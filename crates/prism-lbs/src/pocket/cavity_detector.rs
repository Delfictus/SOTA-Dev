//! Real cavity-based pocket detection using alpha spheres
//!
//! This implements a proper pocket detection algorithm similar to fpocket:
//! 1. Generate alpha spheres at protein surface
//! 2. Cluster spheres by spatial proximity (DBSCAN)
//! 3. Filter clusters by volume and druggability features
//! 4. Rank pockets by druggability score

use crate::graph::ProteinGraph;
use crate::pocket::properties::Pocket;
use crate::scoring::{Components, DrugabilityClass, DruggabilityScore};

/// Configuration for cavity-based pocket detection
#[derive(Debug, Clone)]
pub struct CavityDetectorConfig {
    /// Minimum alpha sphere radius (Angstroms) - smaller = finer surface detail
    pub min_alpha_radius: f32,
    /// Maximum alpha sphere radius (Angstroms) - larger catches bigger cavities
    pub max_alpha_radius: f32,
    /// DBSCAN epsilon (clustering distance threshold in Angstroms)
    pub cluster_eps: f32,
    /// DBSCAN min_samples (minimum spheres to form a cluster)
    pub cluster_min_samples: usize,
    /// Minimum pocket volume (Angstroms^3)
    pub min_volume: f32,
    /// Probe radius for surface calculation
    pub probe_radius: f32,
    /// Grid resolution for alpha sphere generation
    pub grid_resolution: f32,
}

impl Default for CavityDetectorConfig {
    fn default() -> Self {
        Self {
            min_alpha_radius: 3.4,   // Typical for small molecule binding (~3.4Å)
            max_alpha_radius: 8.0,   // Catches drug-sized cavities (not huge voids)
            cluster_eps: 2.5,        // Tighter clustering for distinct pockets
            cluster_min_samples: 8,  // Minimum 8 spheres for valid pocket
            min_volume: 150.0,       // 150 Å³ minimum (drug-like)
            probe_radius: 1.4,       // Water probe radius
            grid_resolution: 1.5,    // 1.5 Å grid spacing (faster, less noise)
        }
    }
}

/// Alpha sphere representing a cavity point
#[derive(Debug, Clone)]
pub struct AlphaSphere {
    pub center: [f32; 3],
    pub radius: f32,
    pub contact_atoms: Vec<usize>,  // Atom indices in contact
}

/// Detected cavity/pocket from alpha sphere clustering
#[derive(Debug, Clone)]
pub struct DetectedCavity {
    pub spheres: Vec<AlphaSphere>,
    pub centroid: [f32; 3],
    pub volume: f32,
    pub atom_indices: Vec<usize>,
    pub residue_indices: Vec<usize>,
}

/// Cavity-based pocket detector using alpha sphere approach
pub struct CavityDetector {
    config: CavityDetectorConfig,
}

impl CavityDetector {
    pub fn new(config: CavityDetectorConfig) -> Self {
        Self { config }
    }

    /// Detect pockets using alpha sphere clustering
    pub fn detect(&self, graph: &ProteinGraph) -> Vec<Pocket> {
        // Step 1: Generate alpha spheres at surface cavities
        let spheres = self.generate_alpha_spheres(graph);
        log::info!("Generated {} alpha spheres", spheres.len());

        if spheres.is_empty() {
            log::warn!("No alpha spheres generated - check surface calculation");
            return Vec::new();
        }

        // Step 2: Cluster spheres using DBSCAN
        let clusters = self.cluster_spheres(&spheres);
        log::info!("Found {} sphere clusters", clusters.len());

        // Step 3: Convert clusters to cavities
        let cavities: Vec<DetectedCavity> = clusters
            .into_iter()
            .map(|cluster_indices| self.cluster_to_cavity(&spheres, &cluster_indices, graph))
            .collect();

        // Step 4: Filter by volume threshold
        let filtered: Vec<DetectedCavity> = cavities
            .into_iter()
            .filter(|c| c.volume >= self.config.min_volume)
            .collect();
        log::info!("{} cavities pass volume threshold (>{}Å³)",
                   filtered.len(), self.config.min_volume);

        // Step 5: Convert to Pocket structs with druggability scoring
        let mut pockets: Vec<Pocket> = filtered
            .into_iter()
            .map(|cavity| self.cavity_to_pocket(cavity, graph))
            .collect();

        // Step 6: Sort by druggability score descending
        pockets.sort_by(|a, b| {
            b.druggability_score.total
                .partial_cmp(&a.druggability_score.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        pockets
    }

    /// Generate alpha spheres at surface cavity points
    fn generate_alpha_spheres(&self, graph: &ProteinGraph) -> Vec<AlphaSphere> {
        let structure = &graph.structure_ref;
        let atoms = &structure.atoms;

        if atoms.is_empty() {
            return Vec::new();
        }

        // Compute bounding box
        let (min_coords, max_coords) = self.compute_bounds(atoms);

        // Expand bounds by max alpha radius
        let padding = self.config.max_alpha_radius + self.config.probe_radius;
        let min_x = min_coords[0] - padding;
        let min_y = min_coords[1] - padding;
        let min_z = min_coords[2] - padding;
        let max_x = max_coords[0] + padding;
        let max_y = max_coords[1] + padding;
        let max_z = max_coords[2] + padding;

        let mut spheres = Vec::new();
        let res = self.config.grid_resolution;

        // Grid-based alpha sphere generation
        let mut x = min_x;
        while x <= max_x {
            let mut y = min_y;
            while y <= max_y {
                let mut z = min_z;
                while z <= max_z {
                    if let Some(sphere) = self.try_create_alpha_sphere([x, y, z], atoms) {
                        spheres.push(sphere);
                    }
                    z += res;
                }
                y += res;
            }
            x += res;
        }

        spheres
    }

    /// Try to create an alpha sphere at a grid point
    fn try_create_alpha_sphere(
        &self,
        point: [f32; 3],
        atoms: &[crate::structure::Atom],
    ) -> Option<AlphaSphere> {
        // Find atoms within max_alpha_radius + their vdW radius
        let mut contact_atoms: Vec<(usize, f32)> = Vec::new();

        for (i, atom) in atoms.iter().enumerate() {
            let dx = point[0] - atom.coord[0] as f32;
            let dy = point[1] - atom.coord[1] as f32;
            let dz = point[2] - atom.coord[2] as f32;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            let vdw = atom.vdw_radius() as f32;
            let max_dist = self.config.max_alpha_radius + vdw + self.config.probe_radius;

            if dist <= max_dist {
                contact_atoms.push((i, dist - vdw));
            }
        }

        // Need at least 4 contact atoms for valid alpha sphere (tetrahedron)
        if contact_atoms.len() < 4 {
            return None;
        }

        // Sort by distance to atom surface
        contact_atoms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Alpha sphere radius = distance to closest atom surface
        let radius = contact_atoms[0].1;

        // Filter: radius must be in valid range
        if radius < self.config.min_alpha_radius || radius > self.config.max_alpha_radius {
            return None;
        }

        // Check: point must be OUTSIDE all atoms (not buried inside)
        let any_inside = contact_atoms.iter().any(|(_, surf_dist)| *surf_dist < 0.0);
        if any_inside {
            return None;
        }

        // Additional filter: sphere center should be in a "cavity"
        // (surrounded by atoms on multiple sides)
        let quadrant_coverage = self.check_quadrant_coverage(&point, &contact_atoms, atoms);
        if quadrant_coverage < 5 {
            // Less than 5 octants have nearby atoms = likely exposed surface, not true cavity
            return None;
        }

        Some(AlphaSphere {
            center: point,
            radius,
            contact_atoms: contact_atoms.iter().take(8).map(|(i, _)| *i).collect(),
        })
    }

    /// Check how many octants around a point have nearby atoms
    fn check_quadrant_coverage(
        &self,
        center: &[f32; 3],
        contacts: &[(usize, f32)],
        atoms: &[crate::structure::Atom],
    ) -> usize {
        let mut octants = [false; 8];

        for (atom_idx, _) in contacts.iter().take(20) {
            let atom = &atoms[*atom_idx];
            let dx = atom.coord[0] as f32 - center[0];
            let dy = atom.coord[1] as f32 - center[1];
            let dz = atom.coord[2] as f32 - center[2];

            let octant = ((dx >= 0.0) as usize)
                       + ((dy >= 0.0) as usize) * 2
                       + ((dz >= 0.0) as usize) * 4;
            octants[octant] = true;
        }

        octants.iter().filter(|&&x| x).count()
    }

    /// Cluster alpha spheres using DBSCAN algorithm
    fn cluster_spheres(&self, spheres: &[AlphaSphere]) -> Vec<Vec<usize>> {
        let n = spheres.len();
        if n == 0 {
            return Vec::new();
        }

        let eps_sq = self.config.cluster_eps * self.config.cluster_eps;
        let min_pts = self.config.cluster_min_samples;

        // Build distance-based neighbor lists
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = spheres[i].center[0] - spheres[j].center[0];
                let dy = spheres[i].center[1] - spheres[j].center[1];
                let dz = spheres[i].center[2] - spheres[j].center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq <= eps_sq {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                }
            }
        }

        // DBSCAN clustering
        let mut labels: Vec<i32> = vec![-1; n];  // -1 = unvisited
        let mut cluster_id = 0;

        for i in 0..n {
            if labels[i] != -1 {
                continue;  // Already processed
            }

            // Check if core point
            if neighbors[i].len() < min_pts {
                labels[i] = -2;  // Noise
                continue;
            }

            // Expand cluster
            let mut cluster = vec![i];
            let mut queue = neighbors[i].clone();
            labels[i] = cluster_id;

            while let Some(j) = queue.pop() {
                if labels[j] == -2 {
                    labels[j] = cluster_id;  // Border point
                    cluster.push(j);
                }
                if labels[j] != -1 {
                    continue;  // Already in a cluster
                }

                labels[j] = cluster_id;
                cluster.push(j);

                if neighbors[j].len() >= min_pts {
                    // Core point - add its neighbors to queue
                    for &k in &neighbors[j] {
                        if labels[k] == -1 || labels[k] == -2 {
                            queue.push(k);
                        }
                    }
                }
            }

            cluster_id += 1;
        }

        // Group indices by cluster
        let num_clusters = cluster_id as usize;
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];

        for (i, &label) in labels.iter().enumerate() {
            if label >= 0 {
                clusters[label as usize].push(i);
            }
        }

        clusters
    }

    /// Convert a sphere cluster to a DetectedCavity
    fn cluster_to_cavity(
        &self,
        spheres: &[AlphaSphere],
        cluster_indices: &[usize],
        graph: &ProteinGraph,
    ) -> DetectedCavity {
        let cluster_spheres: Vec<AlphaSphere> = cluster_indices
            .iter()
            .map(|&i| spheres[i].clone())
            .collect();

        // Compute centroid
        let mut centroid = [0.0f32; 3];
        for sphere in &cluster_spheres {
            centroid[0] += sphere.center[0];
            centroid[1] += sphere.center[1];
            centroid[2] += sphere.center[2];
        }
        let n = cluster_spheres.len() as f32;
        centroid[0] /= n;
        centroid[1] /= n;
        centroid[2] /= n;

        // Collect unique contact atoms
        let mut atom_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for sphere in &cluster_spheres {
            for &atom_idx in &sphere.contact_atoms {
                atom_set.insert(atom_idx);
            }
        }
        let atom_indices: Vec<usize> = atom_set.into_iter().collect();

        // Map to PDB residue sequence numbers (RESSEQ)
        let residue_indices: Vec<usize> = atom_indices
            .iter()
            .filter_map(|&atom_idx| {
                let atom = &graph.structure_ref.atoms[atom_idx];
                // Use PDB RESSEQ (seq_number) directly, not internal index
                Some(atom.residue_seq as usize)
            })
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Estimate volume from alpha spheres (sum of sphere volumes, adjusted for overlap)
        let volume: f32 = cluster_spheres
            .iter()
            .map(|s| (4.0 / 3.0) * std::f32::consts::PI * s.radius.powi(3))
            .sum::<f32>()
            * 0.6;  // Overlap factor

        DetectedCavity {
            spheres: cluster_spheres,
            centroid,
            volume,
            atom_indices,
            residue_indices,
        }
    }

    /// Convert a DetectedCavity to a Pocket with druggability scoring
    fn cavity_to_pocket(&self, cavity: DetectedCavity, graph: &ProteinGraph) -> Pocket {
        let structure = &graph.structure_ref;
        let atoms = &structure.atoms;

        // Compute pocket properties
        let mut total_hydro = 0.0f32;
        let mut total_sasa = 0.0f32;
        let mut total_depth = 0.0f32;
        let mut total_flex = 0.0f32;
        let mut donors = 0usize;
        let mut acceptors = 0usize;

        for &atom_idx in &cavity.atom_indices {
            if atom_idx < atoms.len() {
                let atom = &atoms[atom_idx];
                total_hydro += atom.hydrophobicity as f32;
                total_sasa += atom.sasa as f32;
                total_depth += atom.depth as f32;
                total_flex += atom.b_factor as f32;
                if atom.is_hbond_donor() {
                    donors += 1;
                }
                if atom.is_hbond_acceptor() {
                    acceptors += 1;
                }
            }
        }

        let count = cavity.atom_indices.len().max(1) as f32;

        // Compute druggability score
        let druggability = self.compute_druggability(
            cavity.volume,
            total_hydro / count,
            total_depth / count,
            donors,
            acceptors,
            cavity.atom_indices.len(),
        );

        Pocket {
            atom_indices: cavity.atom_indices,
            residue_indices: cavity.residue_indices,
            centroid: [
                cavity.centroid[0] as f64,
                cavity.centroid[1] as f64,
                cavity.centroid[2] as f64,
            ],
            volume: cavity.volume as f64,
            enclosure_ratio: 0.8,  // Alpha sphere method implies good enclosure
            mean_hydrophobicity: (total_hydro / count) as f64,
            mean_sasa: (total_sasa / count) as f64,
            mean_depth: (total_depth / count) as f64,
            mean_flexibility: (total_flex / count) as f64,
            mean_conservation: 0.0,  // Would need MSA data
            persistence_score: 0.0,
            hbond_donors: donors,
            hbond_acceptors: acceptors,
            druggability_score: druggability,
            boundary_atoms: Vec::new(),
            mean_electrostatic: 0.0,
            gnn_embedding: Vec::new(),
            gnn_druggability: 0.0,
        }
    }

    /// Compute druggability score based on pocket properties
    fn compute_druggability(
        &self,
        volume: f32,
        mean_hydro: f32,
        mean_depth: f32,
        donors: usize,
        acceptors: usize,
        atom_count: usize,
    ) -> DruggabilityScore {
        // Volume score (optimal 300-800 Å³)
        let vol_score = if volume < 100.0 {
            volume / 100.0
        } else if volume < 300.0 {
            0.5 + 0.5 * (volume - 100.0) / 200.0
        } else if volume < 800.0 {
            1.0
        } else if volume < 1500.0 {
            1.0 - 0.3 * (volume - 800.0) / 700.0
        } else {
            0.5
        };

        // Hydrophobicity score (positive = hydrophobic = good for drug binding)
        let hydro_norm = ((mean_hydro + 4.5) / 9.0).clamp(0.0, 1.0);

        // Depth score (deeper = more enclosed = better)
        let depth_score = (mean_depth / 12.0).clamp(0.0, 1.0);

        // H-bond capacity (need some, but not too many)
        let hbond_total = donors + acceptors;
        let hbond_score = if hbond_total < 2 {
            0.3
        } else if hbond_total <= 10 {
            0.8 + 0.2 * (hbond_total as f32 - 2.0) / 8.0
        } else if hbond_total <= 20 {
            1.0
        } else {
            1.0 - 0.3 * ((hbond_total - 20) as f32 / 20.0).min(1.0)
        };

        // Size score (need enough atoms to bind a drug)
        let size_score = if atom_count < 10 {
            atom_count as f32 / 10.0
        } else if atom_count < 50 {
            1.0
        } else {
            1.0 - 0.2 * ((atom_count - 50) as f32 / 100.0).min(1.0)
        };

        // Weighted combination
        let total = 0.25 * vol_score
                  + 0.20 * hydro_norm
                  + 0.20 * depth_score
                  + 0.15 * hbond_score
                  + 0.10 * size_score
                  + 0.10;  // Base score

        let classification = if total > 0.7 {
            DrugabilityClass::HighlyDruggable
        } else if total > 0.5 {
            DrugabilityClass::Druggable
        } else if total > 0.3 {
            DrugabilityClass::DifficultTarget
        } else {
            DrugabilityClass::Undruggable
        };

        DruggabilityScore {
            total: total as f64,
            classification,
            components: Components {
                volume: vol_score as f64,
                hydro: hydro_norm as f64,
                enclosure: 0.8,
                depth: depth_score as f64,
                hbond: hbond_score as f64,
                flex: 0.5,
                cons: 0.0,
                topo: 0.0,
            },
        }
    }

    fn compute_bounds(&self, atoms: &[crate::structure::Atom]) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for atom in atoms {
            for i in 0..3 {
                min[i] = min[i].min(atom.coord[i] as f32);
                max[i] = max[i].max(atom.coord[i] as f32);
            }
        }

        (min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbscan_clustering() {
        let detector = CavityDetector::new(CavityDetectorConfig {
            cluster_eps: 2.0,
            cluster_min_samples: 2,
            ..Default::default()
        });

        // Create 2 clusters of spheres
        // Each cluster needs at least min_pts+1 points for DBSCAN to form clusters
        let spheres = vec![
            // First cluster
            AlphaSphere { center: [0.0, 0.0, 0.0], radius: 1.0, contact_atoms: vec![] },
            AlphaSphere { center: [1.0, 0.0, 0.0], radius: 1.0, contact_atoms: vec![] },
            AlphaSphere { center: [0.5, 0.5, 0.0], radius: 1.0, contact_atoms: vec![] },
            // Second cluster far away - needs 3 points for min_pts=2
            AlphaSphere { center: [10.0, 10.0, 10.0], radius: 1.0, contact_atoms: vec![] },
            AlphaSphere { center: [11.0, 10.0, 10.0], radius: 1.0, contact_atoms: vec![] },
            AlphaSphere { center: [10.5, 10.5, 10.0], radius: 1.0, contact_atoms: vec![] },
        ];

        let clusters = detector.cluster_spheres(&spheres);
        assert_eq!(clusters.len(), 2);
    }
}
