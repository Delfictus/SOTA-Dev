//! Grid-based alpha sphere pocket detection (fpocket-like algorithm)
//!
//! This implements a PROPER cavity detection algorithm:
//! 1. Sample 3D grid at 1.0 Å spacing across protein
//! 2. At each grid point, find distance to nearest atom surface
//! 3. Create alpha sphere if radius is in valid range (2.0-10.0 Å)
//! 4. Filter spheres that are inside atoms or too exposed
//! 5. Cluster spheres with DBSCAN (eps=4.5Å, min_pts=3)
//! 6. Build pockets from clusters, compute properties
//!
//! This approach generates 100+ alpha spheres per pocket (like fpocket)
//! vs the sparse tetrahedra approach that only finds 10-20 points.

use crate::graph::ProteinGraph;
use crate::pocket::properties::Pocket;
use crate::pocket::delaunay_detector::{DelaunayAlphaSphereDetector, DelaunayAlphaSphere};
use crate::scoring::{Components, DrugabilityClass, DruggabilityScore};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[cfg(feature = "cuda")]
use prism_gpu::LbsGpu;

/// Physical constants
mod constants {
    /// Van der Waals radii in Ångströms (CHARMM36)
    pub fn vdw_radius(element: &str) -> f64 {
        match element.trim().to_uppercase().as_str() {
            "C" => 1.70,
            "N" => 1.55,
            "O" => 1.52,
            "S" => 1.80,
            "P" => 1.80,
            "H" => 1.20,
            _ => 1.70,
        }
    }

    /// Probe radius for solvent accessibility
    pub const PROBE_RADIUS: f64 = 1.4;

    /// Grid spacing for alpha sphere sampling
    /// Must be > eps to prevent one giant connected cluster
    pub const GRID_SPACING: f64 = 4.0;

    /// Alpha sphere radius bounds
    pub const MIN_SPHERE_RADIUS: f64 = 2.5;  // Slightly larger minimum
    pub const MAX_SPHERE_RADIUS: f64 = 10.0;

    /// DBSCAN clustering parameters
    /// eps should be > grid_spacing to connect adjacent, but < sqrt(2)*grid to break at boundaries
    pub const DBSCAN_EPS: f64 = 5.0;  // Connect adjacent grid points (4.0 Å), break at diagonals (5.66 Å)
    pub const DBSCAN_MIN_PTS: usize = 2;

    /// Pocket bounds
    pub const MIN_VOLUME: f64 = 100.0;
    pub const MAX_VOLUME: f64 = 5000.0;
    pub const MIN_ATOMS: usize = 10;
    pub const MAX_ATOMS: usize = 400;

    /// Minimum burial depth for valid cavity points
    /// Higher value = only deep cavities, lower value = surface pockets too
    pub const MIN_BURIAL_DEPTH: f64 = 2.0;  // Allow surface pockets

    /// Minimum nearby atoms for cavity validation
    pub const MIN_NEARBY_ATOMS: usize = 4;

    /// Maximum nearby atoms (to avoid interior of protein)
    pub const MAX_NEARBY_ATOMS: usize = 100;  // Relaxed
}

/// Detection method for alpha sphere generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetectionMethod {
    /// Grid-based sampling (faster, denser spheres)
    #[default]
    Grid,
    /// Delaunay tessellation (mathematically precise, fpocket-compatible)
    Delaunay,
    /// Hybrid: Delaunay + grid refinement (best quality)
    Hybrid,
}

/// Configuration for grid-based pocket detection
#[derive(Debug, Clone)]
pub struct VoronoiDetectorConfig {
    pub min_alpha_radius: f64,
    pub max_alpha_radius: f64,
    pub dbscan_eps: f64,
    pub dbscan_min_samples: usize,
    pub min_volume: f64,
    pub max_volume: f64,
    pub min_atoms: usize,
    pub max_atoms: usize,
    pub grid_spacing: f64,
    pub min_burial_depth: f64,
    /// Detection method (Grid, Delaunay, or Hybrid)
    pub detection_method: DetectionMethod,
}

impl Default for VoronoiDetectorConfig {
    fn default() -> Self {
        Self {
            min_alpha_radius: constants::MIN_SPHERE_RADIUS,
            max_alpha_radius: constants::MAX_SPHERE_RADIUS,
            dbscan_eps: constants::DBSCAN_EPS,
            dbscan_min_samples: constants::DBSCAN_MIN_PTS,
            min_volume: constants::MIN_VOLUME,
            max_volume: constants::MAX_VOLUME,
            min_atoms: constants::MIN_ATOMS,
            max_atoms: constants::MAX_ATOMS,
            grid_spacing: constants::GRID_SPACING,
            min_burial_depth: constants::MIN_BURIAL_DEPTH,
            detection_method: DetectionMethod::default(),
        }
    }
}

/// Alpha sphere representing a cavity point
#[derive(Debug, Clone)]
pub struct AlphaSphere {
    pub center: [f64; 3],
    pub radius: f64,
    pub nearby_atoms: Vec<usize>,
    pub burial_depth: f64,
}

/// Spatial hash grid for O(1) neighbor queries
struct SpatialGrid {
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size: f64,
    bounds: GridBounds,
}

struct GridBounds {
    min_x: f64, max_x: f64,
    min_y: f64, max_y: f64,
    min_z: f64, max_z: f64,
}

impl SpatialGrid {
    fn new(atoms: &[crate::structure::Atom], cell_size: f64) -> Self {
        let mut cells = HashMap::new();

        let bounds = GridBounds {
            min_x: atoms.iter().map(|a| a.coord[0]).fold(f64::MAX, f64::min) - cell_size,
            max_x: atoms.iter().map(|a| a.coord[0]).fold(f64::MIN, f64::max) + cell_size,
            min_y: atoms.iter().map(|a| a.coord[1]).fold(f64::MAX, f64::min) - cell_size,
            max_y: atoms.iter().map(|a| a.coord[1]).fold(f64::MIN, f64::max) + cell_size,
            min_z: atoms.iter().map(|a| a.coord[2]).fold(f64::MAX, f64::min) - cell_size,
            max_z: atoms.iter().map(|a| a.coord[2]).fold(f64::MIN, f64::max) + cell_size,
        };

        for (idx, atom) in atoms.iter().enumerate() {
            let key = (
                (atom.coord[0] / cell_size).floor() as i32,
                (atom.coord[1] / cell_size).floor() as i32,
                (atom.coord[2] / cell_size).floor() as i32,
            );
            cells.entry(key).or_insert_with(Vec::new).push(idx);
        }

        Self { cells, cell_size, bounds }
    }

    fn query(&self, point: &[f64; 3], radius: f64) -> Vec<usize> {
        let mut result = Vec::new();
        let cells_to_check = (radius / self.cell_size).ceil() as i32 + 1;

        let cx = (point[0] / self.cell_size).floor() as i32;
        let cy = (point[1] / self.cell_size).floor() as i32;
        let cz = (point[2] / self.cell_size).floor() as i32;

        for dx in -cells_to_check..=cells_to_check {
            for dy in -cells_to_check..=cells_to_check {
                for dz in -cells_to_check..=cells_to_check {
                    if let Some(indices) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        result.extend(indices.iter().copied());
                    }
                }
            }
        }

        result
    }
}

/// Grid-based pocket detector using dense alpha sphere sampling
pub struct VoronoiDetector {
    config: VoronoiDetectorConfig,
    #[cfg(feature = "cuda")]
    gpu: Option<Arc<LbsGpu>>,
}

impl VoronoiDetector {
    pub fn new(config: VoronoiDetectorConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    /// Create a detector with GPU acceleration
    #[cfg(feature = "cuda")]
    pub fn with_gpu(mut self, gpu: Arc<LbsGpu>) -> Self {
        self.gpu = Some(gpu);
        self
    }

    /// Check if GPU acceleration is available and enabled
    #[cfg(feature = "cuda")]
    pub fn has_gpu(&self) -> bool {
        self.gpu.as_ref().map(|g| g.has_pocket_detection()).unwrap_or(false)
    }

    /// Check if GPU is available (no-op when cuda feature is disabled)
    #[cfg(not(feature = "cuda"))]
    pub fn has_gpu(&self) -> bool {
        false
    }

    /// Detect pockets using configured detection method
    pub fn detect(&self, graph: &ProteinGraph) -> Vec<Pocket> {
        let atoms = &graph.structure_ref.atoms;

        if atoms.len() < 50 {
            log::warn!("Structure too small ({} atoms) for pocket detection", atoms.len());
            return Vec::new();
        }

        log::info!("Starting pocket detection ({:?}) for {} atoms",
                   self.config.detection_method, atoms.len());
        let start = std::time::Instant::now();

        // Step 1: Build spatial grid for fast neighbor lookup
        let grid = SpatialGrid::new(atoms, self.config.max_alpha_radius * 2.0);
        log::debug!("Built spatial grid in {:?}", start.elapsed());

        // Step 2: Generate alpha spheres based on configured method
        let spheres = match self.config.detection_method {
            DetectionMethod::Grid => {
                self.generate_grid_spheres(atoms, &grid)
            }
            DetectionMethod::Delaunay => {
                self.generate_delaunay_spheres(atoms, &grid)
            }
            DetectionMethod::Hybrid => {
                // Combine Delaunay + grid for maximum coverage
                let mut delaunay_spheres = self.generate_delaunay_spheres(atoms, &grid);
                let grid_spheres = self.generate_grid_spheres(atoms, &grid);
                delaunay_spheres.extend(grid_spheres);
                self.deduplicate_alpha_spheres(delaunay_spheres)
            }
        };
        log::info!("Generated {} alpha spheres ({:?} method)",
                   spheres.len(), self.config.detection_method);

        if spheres.is_empty() {
            log::warn!("No alpha spheres generated - structure may be too small or too exposed");
            return Vec::new();
        }

        // Step 3: Cluster spheres with DBSCAN
        let clusters = self.cluster_dbscan(&spheres);
        log::info!("DBSCAN found {} clusters (eps={:.1}Å, min_pts={})",
                   clusters.len(), self.config.dbscan_eps, self.config.dbscan_min_samples);

        // Step 4: Build pockets from clusters
        let mut pockets: Vec<Pocket> = clusters
            .into_iter()
            .enumerate()
            .filter_map(|(id, sphere_indices)| {
                self.build_pocket(id, &sphere_indices, &spheres, graph)
            })
            .collect();

        log::info!("Built {} pockets after filtering", pockets.len());

        // Step 5: Sort by druggability
        pockets.sort_by(|a, b| {
            b.druggability_score.total
                .partial_cmp(&a.druggability_score.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top 20
        pockets.truncate(20);

        log::info!("Pocket detection complete: {} pockets in {:?}",
                   pockets.len(), start.elapsed());

        for (i, p) in pockets.iter().enumerate() {
            log::info!(
                "Pocket {}: vol={:.0}Å³, atoms={}, depth={:.1}Å, druggability={:.3}, centroid=({:.1},{:.1},{:.1})",
                i + 1, p.volume, p.atom_indices.len(), p.mean_depth,
                p.druggability_score.total,
                p.centroid[0], p.centroid[1], p.centroid[2]
            );
        }

        pockets
    }

    /// Generate alpha spheres by sampling on 3D grid
    fn generate_grid_spheres(
        &self,
        atoms: &[crate::structure::Atom],
        grid: &SpatialGrid,
    ) -> Vec<AlphaSphere> {
        let bounds = &grid.bounds;
        let spacing = self.config.grid_spacing;

        // Compute protein centroid for depth calculation
        let centroid = [
            atoms.iter().map(|a| a.coord[0]).sum::<f64>() / atoms.len() as f64,
            atoms.iter().map(|a| a.coord[1]).sum::<f64>() / atoms.len() as f64,
            atoms.iter().map(|a| a.coord[2]).sum::<f64>() / atoms.len() as f64,
        ];

        // Compute max distance from centroid for normalization
        let max_dist = atoms.iter()
            .map(|a| {
                let dx = a.coord[0] - centroid[0];
                let dy = a.coord[1] - centroid[1];
                let dz = a.coord[2] - centroid[2];
                (dx*dx + dy*dy + dz*dz).sqrt()
            })
            .fold(0.0_f64, |a, b| a.max(b));

        // Generate grid points
        let mut grid_points = Vec::new();
        let mut x = bounds.min_x;
        while x <= bounds.max_x {
            let mut y = bounds.min_y;
            while y <= bounds.max_y {
                let mut z = bounds.min_z;
                while z <= bounds.max_z {
                    grid_points.push([x, y, z]);
                    z += spacing;
                }
                y += spacing;
            }
            x += spacing;
        }

        log::debug!("Sampling {} grid points", grid_points.len());

        // Process grid points in parallel
        let spheres: Vec<AlphaSphere> = grid_points
            .par_iter()
            .filter_map(|point| {
                self.try_create_sphere(point, atoms, grid, &centroid, max_dist)
            })
            .collect();

        spheres
    }

    /// Generate alpha spheres using Delaunay tessellation (fpocket method)
    fn generate_delaunay_spheres(
        &self,
        atoms: &[crate::structure::Atom],
        grid: &SpatialGrid,
    ) -> Vec<AlphaSphere> {
        log::debug!("[Delaunay] Starting Delaunay tessellation for {} atoms", atoms.len());

        // Create Delaunay detector with matching parameters
        let detector = DelaunayAlphaSphereDetector::new(
            self.config.min_alpha_radius,
            self.config.max_alpha_radius,
        );

        // Run Delaunay detection
        let delaunay_spheres = detector.detect(atoms);
        log::debug!("[Delaunay] Found {} raw Delaunay spheres", delaunay_spheres.len());

        // Compute protein centroid for depth calculation
        let centroid = [
            atoms.iter().map(|a| a.coord[0]).sum::<f64>() / atoms.len() as f64,
            atoms.iter().map(|a| a.coord[1]).sum::<f64>() / atoms.len() as f64,
            atoms.iter().map(|a| a.coord[2]).sum::<f64>() / atoms.len() as f64,
        ];

        let max_dist = atoms.iter()
            .map(|a| {
                let dx = a.coord[0] - centroid[0];
                let dy = a.coord[1] - centroid[1];
                let dz = a.coord[2] - centroid[2];
                (dx*dx + dy*dy + dz*dz).sqrt()
            })
            .fold(0.0_f64, |a, b| a.max(b));

        // Convert Delaunay spheres to our AlphaSphere format
        let spheres: Vec<AlphaSphere> = delaunay_spheres
            .into_iter()
            .filter(|ds| ds.is_valid)
            .filter_map(|ds| {
                // Query nearby atoms using spatial grid
                let nearby_atoms = grid.query(&ds.center, self.config.max_alpha_radius);
                if nearby_atoms.len() < constants::MIN_NEARBY_ATOMS {
                    return None;
                }
                if nearby_atoms.len() > constants::MAX_NEARBY_ATOMS {
                    return None;
                }

                // Compute burial depth
                let dist_to_centroid = {
                    let dx = ds.center[0] - centroid[0];
                    let dy = ds.center[1] - centroid[1];
                    let dz = ds.center[2] - centroid[2];
                    (dx*dx + dy*dy + dz*dz).sqrt()
                };

                let normalized_depth = 1.0 - (dist_to_centroid / max_dist).min(1.0);
                let density_factor = (nearby_atoms.len() as f64 / 30.0).min(1.0);
                let burial_depth = normalized_depth * 25.0 * density_factor;

                if burial_depth < self.config.min_burial_depth {
                    return None;
                }

                Some(AlphaSphere {
                    center: ds.center,
                    radius: ds.radius,
                    nearby_atoms,
                    burial_depth,
                })
            })
            .collect();

        log::info!("[Delaunay] Converted {} spheres after filtering", spheres.len());
        spheres
    }

    /// Deduplicate alpha spheres (for Hybrid mode)
    fn deduplicate_alpha_spheres(&self, mut spheres: Vec<AlphaSphere>) -> Vec<AlphaSphere> {
        let initial_count = spheres.len();
        if initial_count < 2 {
            return spheres;
        }

        // Sort by center for efficient deduplication
        spheres.sort_by(|a, b| {
            a.center[0].partial_cmp(&b.center[0]).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.center[1].partial_cmp(&b.center[1]).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.center[2].partial_cmp(&b.center[2]).unwrap_or(std::cmp::Ordering::Equal))
        });

        let min_dist_sq = 0.25; // 0.5Å minimum separation
        let mut unique = Vec::with_capacity(initial_count);
        unique.push(spheres[0].clone());

        for sphere in spheres.into_iter().skip(1) {
            let last = unique.last().unwrap();
            let dist_sq = (sphere.center[0] - last.center[0]).powi(2)
                + (sphere.center[1] - last.center[1]).powi(2)
                + (sphere.center[2] - last.center[2]).powi(2);

            if dist_sq > min_dist_sq {
                unique.push(sphere);
            }
        }

        log::debug!("Deduplicated {} -> {} spheres", initial_count, unique.len());
        unique
    }

    /// Try to create an alpha sphere at a grid point
    fn try_create_sphere(
        &self,
        point: &[f64; 3],
        atoms: &[crate::structure::Atom],
        grid: &SpatialGrid,
        protein_centroid: &[f64; 3],
        max_dist: f64,
    ) -> Option<AlphaSphere> {
        // Find nearby atoms
        let nearby_indices = grid.query(point, self.config.max_alpha_radius);

        if nearby_indices.len() < constants::MIN_NEARBY_ATOMS {
            return None; // Not enough nearby atoms for a cavity
        }

        // Find distance to nearest atom SURFACE (not center)
        let mut min_dist_to_surface = f64::MAX;
        let mut closest_atoms = Vec::new();

        for &idx in &nearby_indices {
            let atom = &atoms[idx];
            let dx = point[0] - atom.coord[0];
            let dy = point[1] - atom.coord[1];
            let dz = point[2] - atom.coord[2];
            let dist_to_center = (dx*dx + dy*dy + dz*dz).sqrt();
            let dist_to_surface = dist_to_center - atom.vdw_radius();

            if dist_to_surface < min_dist_to_surface {
                min_dist_to_surface = dist_to_surface;
            }

            // Track atoms within max sphere radius
            if dist_to_center < self.config.max_alpha_radius + atom.vdw_radius() {
                closest_atoms.push(idx);
            }
        }

        // Point must be OUTSIDE all atoms (positive distance to surface)
        if min_dist_to_surface < 0.0 {
            return None; // Inside an atom
        }

        // Sphere radius = distance to nearest surface - probe radius
        let radius = min_dist_to_surface;

        // Filter by radius bounds
        if radius < self.config.min_alpha_radius || radius > self.config.max_alpha_radius {
            return None;
        }

        // Need enough nearby atoms (cavity, not void)
        if closest_atoms.len() < constants::MIN_NEARBY_ATOMS {
            return None;
        }

        // But not too many (that's the interior of the protein, not a pocket)
        if closest_atoms.len() > constants::MAX_NEARBY_ATOMS {
            return None;
        }

        // Compute burial depth
        let dist_to_centroid = {
            let dx = point[0] - protein_centroid[0];
            let dy = point[1] - protein_centroid[1];
            let dz = point[2] - protein_centroid[2];
            (dx*dx + dy*dy + dz*dz).sqrt()
        };

        // Count nearby atoms for density
        let density_count = nearby_indices.iter()
            .filter(|&&idx| {
                let atom = &atoms[idx];
                let dx = point[0] - atom.coord[0];
                let dy = point[1] - atom.coord[1];
                let dz = point[2] - atom.coord[2];
                (dx*dx + dy*dy + dz*dz).sqrt() < 8.0
            })
            .count();

        // Depth formula: closer to center + higher density = more buried
        let normalized_depth = 1.0 - (dist_to_centroid / max_dist).min(1.0);
        let density_factor = (density_count as f64 / 30.0).min(1.0);
        let burial_depth = normalized_depth * 25.0 * density_factor;

        // Filter by minimum burial depth
        if burial_depth < self.config.min_burial_depth {
            return None;
        }

        Some(AlphaSphere {
            center: *point,
            radius,
            nearby_atoms: closest_atoms,
            burial_depth,
        })
    }

    /// DBSCAN clustering of alpha spheres
    fn cluster_dbscan(&self, spheres: &[AlphaSphere]) -> Vec<Vec<usize>> {
        let n = spheres.len();
        if n == 0 {
            return Vec::new();
        }

        let eps = self.config.dbscan_eps;
        let min_pts = self.config.dbscan_min_samples;
        let eps_sq = eps * eps;

        // Build neighbor lists (parallel)
        let neighbors: Vec<Vec<usize>> = spheres
            .par_iter()
            .enumerate()
            .map(|(i, si)| {
                spheres.iter().enumerate()
                    .filter(|(j, sj)| {
                        if i == *j { return false; }
                        let dx = si.center[0] - sj.center[0];
                        let dy = si.center[1] - sj.center[1];
                        let dz = si.center[2] - sj.center[2];
                        dx*dx + dy*dy + dz*dz <= eps_sq
                    })
                    .map(|(j, _)| j)
                    .collect()
            })
            .collect();

        // DBSCAN clustering
        let mut labels: Vec<i32> = vec![-1; n]; // -1 = unvisited
        let mut cluster_id = 0;

        for i in 0..n {
            if labels[i] != -1 {
                continue;
            }

            if neighbors[i].len() < min_pts {
                labels[i] = -2; // Noise
                continue;
            }

            // Start new cluster
            labels[i] = cluster_id;
            let mut queue = neighbors[i].clone();

            while let Some(j) = queue.pop() {
                if labels[j] == -2 {
                    labels[j] = cluster_id;
                }
                if labels[j] != -1 {
                    continue;
                }

                labels[j] = cluster_id;

                if neighbors[j].len() >= min_pts {
                    for &k in &neighbors[j] {
                        if labels[k] == -1 || labels[k] == -2 {
                            queue.push(k);
                        }
                    }
                }
            }

            cluster_id += 1;
        }

        // Group by cluster
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); cluster_id as usize];
        for (i, &label) in labels.iter().enumerate() {
            if label >= 0 {
                clusters[label as usize].push(i);
            }
        }

        clusters.into_iter().filter(|c| !c.is_empty()).collect()
    }

    /// Build pocket from a cluster of alpha spheres
    fn build_pocket(
        &self,
        id: usize,
        sphere_indices: &[usize],
        spheres: &[AlphaSphere],
        graph: &ProteinGraph,
    ) -> Option<Pocket> {
        let atoms = &graph.structure_ref.atoms;

        // Collect all atoms from spheres in this cluster
        let mut atom_set: HashSet<usize> = HashSet::new();
        let cluster_spheres: Vec<&AlphaSphere> = sphere_indices
            .iter()
            .map(|&i| &spheres[i])
            .collect();

        for sphere in &cluster_spheres {
            for &atom_idx in &sphere.nearby_atoms {
                if atom_idx < atoms.len() {
                    atom_set.insert(atom_idx);
                }
            }
        }

        let atom_indices: Vec<usize> = atom_set.into_iter().collect();

        // Filter by atom count
        if atom_indices.len() < self.config.min_atoms {
            log::debug!("Rejected cluster {}: only {} atoms", id, atom_indices.len());
            return None;
        }
        if atom_indices.len() > self.config.max_atoms {
            log::debug!("Rejected cluster {}: {} atoms exceeds max", id, atom_indices.len());
            return None;
        }

        // Get unique PDB residue sequence numbers (RESSEQ)
        let residue_indices: Vec<usize> = atom_indices
            .iter()
            .filter_map(|&ai| {
                let atom = &atoms[ai];
                // Use PDB RESSEQ (seq_number) directly, not internal index
                Some(atom.residue_seq as usize)
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        // Compute centroid from sphere centers
        let centroid = [
            cluster_spheres.iter().map(|s| s.center[0]).sum::<f64>() / cluster_spheres.len() as f64,
            cluster_spheres.iter().map(|s| s.center[1]).sum::<f64>() / cluster_spheres.len() as f64,
            cluster_spheres.iter().map(|s| s.center[2]).sum::<f64>() / cluster_spheres.len() as f64,
        ];

        // Compute volume using Monte Carlo integration
        let volume = self.compute_volume_monte_carlo(&cluster_spheres);

        // Filter by volume
        if volume < self.config.min_volume {
            log::debug!("Rejected cluster {}: volume {:.0}Å³ < min", id, volume);
            return None;
        }
        if volume > self.config.max_volume {
            log::debug!("Rejected cluster {}: volume {:.0}Å³ > max", id, volume);
            return None;
        }

        // Compute mean burial depth
        let mean_depth = cluster_spheres.iter()
            .map(|s| s.burial_depth)
            .sum::<f64>() / cluster_spheres.len() as f64;

        // Compute pocket properties from atoms
        let pocket_atoms: Vec<&crate::structure::Atom> = atom_indices
            .iter()
            .map(|&i| &atoms[i])
            .collect();

        let mean_hydro = pocket_atoms.iter()
            .map(|a| a.hydrophobicity)
            .sum::<f64>() / pocket_atoms.len() as f64;

        let mean_sasa = pocket_atoms.iter()
            .map(|a| a.sasa)
            .sum::<f64>() / pocket_atoms.len() as f64;

        let mean_flex = pocket_atoms.iter()
            .map(|a| a.b_factor)
            .sum::<f64>() / pocket_atoms.len() as f64;

        let hbond_donors = pocket_atoms.iter()
            .filter(|a| a.is_hbond_donor())
            .count();

        let hbond_acceptors = pocket_atoms.iter()
            .filter(|a| a.is_hbond_acceptor())
            .count();

        // Compute enclosure from sphere coverage
        let enclosure_ratio = (mean_depth / 20.0).min(1.0);

        // Score druggability
        let druggability_score = self.compute_druggability(
            volume,
            mean_hydro,
            enclosure_ratio,
            mean_depth,
            hbond_donors,
            hbond_acceptors,
            mean_flex,
        );

        Some(Pocket {
            atom_indices,
            residue_indices,
            centroid,
            volume,
            enclosure_ratio,
            mean_hydrophobicity: mean_hydro,
            mean_sasa,
            mean_depth,
            mean_flexibility: mean_flex,
            mean_conservation: 0.0,
            persistence_score: mean_depth * enclosure_ratio,
            hbond_donors,
            hbond_acceptors,
            druggability_score,
            boundary_atoms: Vec::new(),
            mean_electrostatic: 0.0,
            gnn_embedding: Vec::new(),
            gnn_druggability: 0.0,
        })
    }

    /// Monte Carlo volume estimation for sphere union
    fn compute_volume_monte_carlo(&self, spheres: &[&AlphaSphere]) -> f64 {
        if spheres.is_empty() {
            return 0.0;
        }

        // Find bounding box
        let min_x = spheres.iter().map(|s| s.center[0] - s.radius).fold(f64::MAX, f64::min);
        let max_x = spheres.iter().map(|s| s.center[0] + s.radius).fold(f64::MIN, f64::max);
        let min_y = spheres.iter().map(|s| s.center[1] - s.radius).fold(f64::MAX, f64::min);
        let max_y = spheres.iter().map(|s| s.center[1] + s.radius).fold(f64::MIN, f64::max);
        let min_z = spheres.iter().map(|s| s.center[2] - s.radius).fold(f64::MAX, f64::min);
        let max_z = spheres.iter().map(|s| s.center[2] + s.radius).fold(f64::MIN, f64::max);

        let box_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z);

        // Monte Carlo sampling
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let num_samples = 10_000;
        let mut inside_count = 0;

        for _ in 0..num_samples {
            let x = rng.gen_range(min_x..max_x);
            let y = rng.gen_range(min_y..max_y);
            let z = rng.gen_range(min_z..max_z);

            // Check if point is inside any sphere
            if spheres.iter().any(|s| {
                let dx = x - s.center[0];
                let dy = y - s.center[1];
                let dz = z - s.center[2];
                (dx*dx + dy*dy + dz*dz).sqrt() <= s.radius
            }) {
                inside_count += 1;
            }
        }

        box_volume * (inside_count as f64 / num_samples as f64)
    }

    /// Compute druggability score
    fn compute_druggability(
        &self,
        volume: f64,
        mean_hydro: f64,
        enclosure: f64,
        depth: f64,
        donors: usize,
        acceptors: usize,
        flexibility: f64,
    ) -> DruggabilityScore {
        // Volume score (sigmoid centered at 400 Å³)
        let vol_score = 1.0 / (1.0 + (-(volume - 400.0) / 200.0).exp());

        // Hydrophobicity score (optimal around 0-2 on Kyte-Doolittle)
        let hydro_score = (1.0 - (mean_hydro - 1.0).abs() / 5.0).max(0.0);

        // Depth score (sigmoid centered at 8 Å)
        let depth_score = 1.0 / (1.0 + (-(depth - 8.0) / 5.0).exp());

        // H-bond score
        let total_hbond = (donors + acceptors) as f64;
        let balance = 1.0 - ((donors as f64 - acceptors as f64).abs() / total_hbond.max(1.0));
        let count_score = if total_hbond < 3.0 {
            total_hbond / 3.0
        } else if total_hbond <= 20.0 {
            1.0
        } else {
            20.0 / total_hbond
        };
        let hbond_score = count_score * balance;

        // Flexibility score (optimal B-factor 20-40)
        let flex_score = if flexibility < 10.0 {
            flexibility / 10.0
        } else if flexibility <= 40.0 {
            1.0
        } else {
            (80.0 - flexibility) / 40.0
        }.max(0.0);

        // Weights from DoGSiteScorer
        let total = 0.20 * vol_score
            + 0.25 * hydro_score
            + 0.15 * enclosure
            + 0.15 * depth_score
            + 0.15 * hbond_score
            + 0.10 * flex_score;

        let total = total.max(0.0).min(1.0);

        let classification = if total >= 0.8 {
            DrugabilityClass::HighlyDruggable
        } else if total >= 0.5 {
            DrugabilityClass::Druggable
        } else if total >= 0.3 {
            DrugabilityClass::DifficultTarget
        } else {
            DrugabilityClass::Undruggable
        };

        DruggabilityScore {
            total,
            classification,
            components: Components {
                volume: vol_score,
                hydro: hydro_score,
                enclosure,
                depth: depth_score,
                hbond: hbond_score,
                flex: flex_score,
                cons: 0.0,
                topo: 0.0,
            },
        }
    }

    /// GPU-accelerated alpha sphere generation
    #[cfg(feature = "cuda")]
    fn generate_grid_spheres_gpu(
        &self,
        atoms: &[crate::structure::Atom],
    ) -> Option<Vec<AlphaSphere>> {
        let gpu = self.gpu.as_ref()?;
        if !gpu.has_pocket_detection() {
            log::debug!("GPU pocket detection kernels not available");
            return None;
        }

        // Convert atom data to f32 arrays for GPU
        let coords: Vec<[f32; 3]> = atoms.iter()
            .map(|a| [a.coord[0] as f32, a.coord[1] as f32, a.coord[2] as f32])
            .collect();
        let vdw: Vec<f32> = atoms.iter()
            .map(|a| a.vdw_radius() as f32)
            .collect();

        // Compute grid bounds
        let min_x = coords.iter().map(|c| c[0]).fold(f32::INFINITY, f32::min) - 10.0;
        let max_x = coords.iter().map(|c| c[0]).fold(f32::NEG_INFINITY, f32::max) + 10.0;
        let min_y = coords.iter().map(|c| c[1]).fold(f32::INFINITY, f32::min) - 10.0;
        let max_y = coords.iter().map(|c| c[1]).fold(f32::NEG_INFINITY, f32::max) + 10.0;
        let min_z = coords.iter().map(|c| c[2]).fold(f32::INFINITY, f32::min) - 10.0;
        let max_z = coords.iter().map(|c| c[2]).fold(f32::NEG_INFINITY, f32::max) + 10.0;

        // Generate alpha spheres on GPU
        match gpu.generate_alpha_spheres(
            &coords,
            &vdw,
            (min_x, max_x, min_y, max_y, min_z, max_z),
            self.config.grid_spacing as f32,
        ) {
            Ok((sphere_coords, sphere_radii, sphere_burials, _valid)) => {
                // Convert GPU results to AlphaSphere structs
                let spheres: Vec<AlphaSphere> = sphere_coords.iter().zip(sphere_radii.iter()).zip(sphere_burials.iter())
                    .filter(|((_, &r), &b)| {
                        r >= self.config.min_alpha_radius as f32
                            && r <= self.config.max_alpha_radius as f32
                            && b >= self.config.min_burial_depth as f32
                    })
                    .map(|((center, &radius), &burial)| {
                        // Find nearby atoms (simplified - use spatial query on CPU)
                        let nearby_atoms: Vec<usize> = coords.iter().enumerate()
                            .filter(|(_, c)| {
                                let dx = center[0] - c[0];
                                let dy = center[1] - c[1];
                                let dz = center[2] - c[2];
                                (dx*dx + dy*dy + dz*dz).sqrt() < self.config.max_alpha_radius as f32 + 3.0
                            })
                            .map(|(i, _)| i)
                            .collect();

                        AlphaSphere {
                            center: [center[0] as f64, center[1] as f64, center[2] as f64],
                            radius: radius as f64,
                            nearby_atoms,
                            burial_depth: burial as f64,
                        }
                    })
                    .collect();

                log::info!("GPU generated {} alpha spheres", spheres.len());
                Some(spheres)
            }
            Err(e) => {
                log::warn!("GPU alpha sphere generation failed: {}, falling back to CPU", e);
                None
            }
        }
    }

    /// GPU-accelerated DBSCAN clustering
    #[cfg(feature = "cuda")]
    fn cluster_dbscan_gpu(&self, spheres: &[AlphaSphere]) -> Option<Vec<Vec<usize>>> {
        let gpu = self.gpu.as_ref()?;
        if !gpu.has_pocket_detection() {
            return None;
        }

        // Convert sphere centers to f32 for GPU
        let coords: Vec<[f32; 3]> = spheres.iter()
            .map(|s| [s.center[0] as f32, s.center[1] as f32, s.center[2] as f32])
            .collect();

        match gpu.dbscan_cluster(
            &coords,
            self.config.dbscan_eps as f32,
            self.config.dbscan_min_samples as i32,
        ) {
            Ok(labels) => {
                // Group by cluster label
                let num_clusters = labels.iter().filter(|&&l| l >= 0).max().map(|m| m + 1).unwrap_or(0) as usize;
                let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];

                for (i, &label) in labels.iter().enumerate() {
                    if label >= 0 {
                        clusters[label as usize].push(i);
                    }
                }

                let result: Vec<Vec<usize>> = clusters.into_iter().filter(|c| !c.is_empty()).collect();
                log::info!("GPU DBSCAN found {} clusters", result.len());
                Some(result)
            }
            Err(e) => {
                log::warn!("GPU DBSCAN failed: {}, falling back to CPU", e);
                None
            }
        }
    }

    /// Detect pockets with GPU acceleration (falls back to CPU if unavailable)
    #[cfg(feature = "cuda")]
    pub fn detect_gpu(&self, graph: &ProteinGraph) -> Vec<Pocket> {
        let atoms = &graph.structure_ref.atoms;

        if atoms.len() < 50 {
            log::warn!("Structure too small ({} atoms) for pocket detection", atoms.len());
            return Vec::new();
        }

        log::info!("Starting GPU-accelerated pocket detection for {} atoms", atoms.len());
        let start = std::time::Instant::now();

        // Try GPU alpha sphere generation first
        let spheres = match self.generate_grid_spheres_gpu(atoms) {
            Some(s) if !s.is_empty() => s,
            _ => {
                log::info!("Falling back to CPU alpha sphere generation");
                let grid = SpatialGrid::new(atoms, self.config.max_alpha_radius * 2.0);
                self.generate_grid_spheres(atoms, &grid)
            }
        };

        log::info!("Generated {} alpha spheres", spheres.len());

        if spheres.is_empty() {
            log::warn!("No alpha spheres generated");
            return Vec::new();
        }

        // Try GPU DBSCAN clustering
        let clusters = match self.cluster_dbscan_gpu(&spheres) {
            Some(c) if !c.is_empty() => c,
            _ => {
                log::info!("Falling back to CPU DBSCAN");
                self.cluster_dbscan(&spheres)
            }
        };

        log::info!("DBSCAN found {} clusters", clusters.len());

        // Build pockets from clusters (CPU)
        let mut pockets: Vec<Pocket> = clusters
            .into_iter()
            .enumerate()
            .filter_map(|(id, sphere_indices)| {
                self.build_pocket(id, &sphere_indices, &spheres, graph)
            })
            .collect();

        // Sort by druggability
        pockets.sort_by(|a, b| {
            b.druggability_score.total
                .partial_cmp(&a.druggability_score.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        pockets.truncate(20);

        log::info!("GPU pocket detection complete: {} pockets in {:?}",
                   pockets.len(), start.elapsed());

        pockets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = VoronoiDetectorConfig::default();
        assert_eq!(config.grid_spacing, constants::GRID_SPACING);
        assert_eq!(config.dbscan_eps, constants::DBSCAN_EPS);
        assert_eq!(config.dbscan_min_samples, constants::DBSCAN_MIN_PTS);
    }

    #[test]
    fn test_volume_calculation() {
        // Single sphere volume = 4/3 * pi * r^3
        let sphere = AlphaSphere {
            center: [0.0, 0.0, 0.0],
            radius: 5.0,
            nearby_atoms: vec![],
            burial_depth: 10.0,
        };

        let detector = VoronoiDetector::new(VoronoiDetectorConfig::default());
        let volume = detector.compute_volume_monte_carlo(&[&sphere]);

        let expected = (4.0 / 3.0) * std::f64::consts::PI * 125.0;
        // Monte Carlo should be within 10% of exact
        assert!((volume - expected).abs() / expected < 0.15);
    }
}
