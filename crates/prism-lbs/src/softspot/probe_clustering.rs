//! FTMap-style Probe Clustering for cryptic site detection
//!
//! Implements computational fragment mapping to identify binding hot spots
//! regardless of the current conformational state. This directly probes
//! binding propensity using small molecular fragments.
//!
//! ## Scientific Basis
//!
//! - Small molecular probes (CH3, OH, NH3, etc.) bind to favorable regions
//! - Clustering of probe binding sites reveals druggable hot spots
//! - Hot spots persist across conformational states (robust signal)
//! - This method can detect cryptic sites by probing local binding affinity
//!
//! ## Method
//!
//! 1. Generate probe positions on molecular surface
//! 2. Score each position using simplified energy function
//! 3. Cluster favorable positions
//! 4. Rank clusters by binding propensity
//!
//! ## References
//!
//! - Brenke et al. (2009) - Fragment-based identification of druggable hot spots
//! - Kozakov et al. (2015) - FTMap methodology

use crate::structure::Atom;
use std::collections::HashMap;

//=============================================================================
// CONSTANTS
//=============================================================================

/// Grid spacing for probe placement (Angstroms)
pub const PROBE_GRID_SPACING: f64 = 1.0;

/// Minimum distance from protein surface (Angstroms)
pub const PROBE_MIN_DISTANCE: f64 = 2.5;

/// Maximum distance from protein surface (Angstroms)
pub const PROBE_MAX_DISTANCE: f64 = 5.0;

/// Lennard-Jones well depth (kcal/mol, approximate)
pub const LJ_EPSILON: f64 = 0.1;

/// Probe radius (Angstroms)
pub const PROBE_RADIUS: f64 = 1.7;

/// Electrostatic screening constant
pub const ELECTROSTATIC_SCREENING: f64 = 4.0;

/// Clustering distance for probe positions (Angstroms)
pub const CLUSTER_DISTANCE: f64 = 4.0;

/// Minimum probes in a cluster to be significant
pub const MIN_CLUSTER_SIZE: usize = 5;

/// Energy threshold for favorable binding (kcal/mol)
pub const FAVORABLE_ENERGY_THRESHOLD: f64 = -0.5;

/// Weight of probe clustering signal in combined scoring
pub const PROBE_WEIGHT: f64 = 0.20;

//=============================================================================
// TYPES
//=============================================================================

/// A positioned probe with binding energy
#[derive(Debug, Clone)]
pub struct Probe {
    pub position: [f64; 3],
    /// Total binding energy (more negative = better)
    pub energy: f64,
    /// Van der Waals contribution
    pub vdw_energy: f64,
    /// Electrostatic contribution
    pub elec_energy: f64,
    /// Desolvation penalty
    pub desolv_penalty: f64,
}

/// A cluster of favorable probe positions
#[derive(Debug, Clone)]
pub struct ProbeCluster {
    pub id: usize,
    pub centroid: [f64; 3],
    pub num_probes: usize,
    /// Average binding energy in cluster
    pub mean_energy: f64,
    /// Best (most negative) energy in cluster
    pub best_energy: f64,
    /// Residues in contact with this cluster
    pub contact_residues: Vec<i32>,
    /// Binding propensity score (0-1)
    pub binding_score: f64,
}

/// Per-residue binding propensity from probe analysis
#[derive(Debug, Clone)]
pub struct ResidueBindingPropensity {
    pub residue_seq: i32,
    pub chain_id: char,
    /// Number of favorable probes near this residue
    pub probe_count: usize,
    /// Mean energy of nearby probes
    pub mean_probe_energy: f64,
    /// Binding propensity score (0-1)
    pub binding_score: f64,
}

/// Results from probe clustering analysis
#[derive(Debug, Clone)]
pub struct ProbeClusteringResult {
    pub clusters: Vec<ProbeCluster>,
    pub residue_propensities: Vec<ResidueBindingPropensity>,
    pub total_probes_placed: usize,
    pub favorable_probes: usize,
    /// Residues identified as binding hot spots (binding_score > 0.5)
    pub hot_spot_residues: Vec<i32>,
}

//=============================================================================
// PROBE CLUSTERING ANALYZER
//=============================================================================

/// FTMap-style probe clustering analyzer
pub struct ProbeClusteringAnalyzer {
    /// Grid spacing for probe placement
    pub grid_spacing: f64,
    /// Minimum distance from surface
    pub min_distance: f64,
    /// Maximum distance from surface
    pub max_distance: f64,
    /// Clustering distance
    pub cluster_distance: f64,
    /// Minimum cluster size
    pub min_cluster_size: usize,
}

impl Default for ProbeClusteringAnalyzer {
    fn default() -> Self {
        Self {
            grid_spacing: PROBE_GRID_SPACING,
            min_distance: PROBE_MIN_DISTANCE,
            max_distance: PROBE_MAX_DISTANCE,
            cluster_distance: CLUSTER_DISTANCE,
            min_cluster_size: MIN_CLUSTER_SIZE,
        }
    }
}

impl ProbeClusteringAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze protein for binding hot spots using probe clustering
    pub fn analyze(&self, atoms: &[Atom]) -> ProbeClusteringResult {
        if atoms.len() < 10 {
            return ProbeClusteringResult {
                clusters: Vec::new(),
                residue_propensities: Vec::new(),
                total_probes_placed: 0,
                favorable_probes: 0,
                hot_spot_residues: Vec::new(),
            };
        }

        log::debug!("[PROBE] Analyzing {} atoms for binding hot spots", atoms.len());

        // Step 1: Generate probe positions
        let probes = self.generate_probes(atoms);
        let total_probes = probes.len();
        log::debug!("[PROBE] Generated {} probe positions", total_probes);

        // Step 2: Score each probe
        let scored_probes: Vec<Probe> = probes
            .into_iter()
            .map(|pos| self.score_probe(pos, atoms))
            .collect();

        // Step 3: Filter favorable probes
        let favorable: Vec<Probe> = scored_probes
            .into_iter()
            .filter(|p| p.energy < FAVORABLE_ENERGY_THRESHOLD)
            .collect();
        let favorable_count = favorable.len();
        log::debug!("[PROBE] {} favorable probe positions", favorable_count);

        // Step 4: Cluster favorable probes
        let clusters = self.cluster_probes(&favorable, atoms);
        log::debug!("[PROBE] Formed {} significant clusters", clusters.len());

        // Step 5: Calculate per-residue propensities
        let residue_propensities = self.calculate_residue_propensities(atoms, &favorable);

        // Step 6: Identify hot spot residues
        let hot_spot_residues: Vec<i32> = residue_propensities
            .iter()
            .filter(|r| r.binding_score > 0.5)
            .map(|r| r.residue_seq)
            .collect();

        log::debug!(
            "[PROBE] {} hot spot residues identified",
            hot_spot_residues.len()
        );

        ProbeClusteringResult {
            clusters,
            residue_propensities,
            total_probes_placed: total_probes,
            favorable_probes: favorable_count,
            hot_spot_residues,
        }
    }

    /// Generate probe positions on a grid around the protein
    fn generate_probes(&self, atoms: &[Atom]) -> Vec<[f64; 3]> {
        // Find bounding box
        let (mut min_x, mut min_y, mut min_z) = (f64::MAX, f64::MAX, f64::MAX);
        let (mut max_x, mut max_y, mut max_z) = (f64::MIN, f64::MIN, f64::MIN);

        for atom in atoms {
            min_x = min_x.min(atom.coord[0]);
            min_y = min_y.min(atom.coord[1]);
            min_z = min_z.min(atom.coord[2]);
            max_x = max_x.max(atom.coord[0]);
            max_y = max_y.max(atom.coord[1]);
            max_z = max_z.max(atom.coord[2]);
        }

        // Extend box
        let padding = self.max_distance + 2.0;
        min_x -= padding;
        min_y -= padding;
        min_z -= padding;
        max_x += padding;
        max_y += padding;
        max_z += padding;

        let mut probes = Vec::new();
        let mut x = min_x;
        while x <= max_x {
            let mut y = min_y;
            while y <= max_y {
                let mut z = min_z;
                while z <= max_z {
                    let pos = [x, y, z];

                    // Check distance to nearest atom
                    let min_dist = self.min_distance_to_protein(&pos, atoms);

                    if min_dist >= self.min_distance && min_dist <= self.max_distance {
                        probes.push(pos);
                    }

                    z += self.grid_spacing;
                }
                y += self.grid_spacing;
            }
            x += self.grid_spacing;
        }

        // Limit probe count for efficiency
        if probes.len() > 50000 {
            // Subsample uniformly
            let step = probes.len() / 50000;
            probes = probes.into_iter().step_by(step.max(1)).collect();
        }

        probes
    }

    /// Calculate minimum distance from position to any protein atom
    fn min_distance_to_protein(&self, pos: &[f64; 3], atoms: &[Atom]) -> f64 {
        atoms
            .iter()
            .map(|a| {
                let dx = a.coord[0] - pos[0];
                let dy = a.coord[1] - pos[1];
                let dz = a.coord[2] - pos[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(f64::MAX, |a, b| a.min(b))
    }

    /// Score a probe position using simplified energy function
    fn score_probe(&self, position: [f64; 3], atoms: &[Atom]) -> Probe {
        let mut vdw_energy = 0.0;
        let mut elec_energy = 0.0;

        for atom in atoms {
            let dx = atom.coord[0] - position[0];
            let dy = atom.coord[1] - position[1];
            let dz = atom.coord[2] - position[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist < 0.1 {
                continue; // Skip overlapping
            }

            // Get atom radius
            let atom_radius = get_vdw_radius(&atom.element);

            // Lennard-Jones 6-12 potential (simplified)
            let sigma = (PROBE_RADIUS + atom_radius) / 2.0;
            let ratio = sigma / dist;
            let ratio6 = ratio.powi(6);
            let ratio12 = ratio6 * ratio6;
            vdw_energy += 4.0 * LJ_EPSILON * (ratio12 - ratio6);

            // Electrostatic interaction (Coulomb with screening)
            if atom.partial_charge.abs() > 0.01 {
                // Assume probe has small positive charge (like NH3)
                let probe_charge = 0.1;
                elec_energy += 332.0 * probe_charge * atom.partial_charge
                    / (ELECTROSTATIC_SCREENING * dist);
            }
        }

        // Simplified desolvation penalty (based on burial)
        let burial = self.calculate_burial(&position, atoms);
        let desolv_penalty = burial * 0.5; // Penalty for being buried

        let total_energy = vdw_energy + elec_energy + desolv_penalty;

        Probe {
            position,
            energy: total_energy,
            vdw_energy,
            elec_energy,
            desolv_penalty,
        }
    }

    /// Calculate how buried a position is (0-1)
    fn calculate_burial(&self, pos: &[f64; 3], atoms: &[Atom]) -> f64 {
        let radius = 6.0;
        let count = atoms
            .iter()
            .filter(|a| {
                let dx = a.coord[0] - pos[0];
                let dy = a.coord[1] - pos[1];
                let dz = a.coord[2] - pos[2];
                (dx * dx + dy * dy + dz * dz) < radius * radius
            })
            .count();

        // Normalize: ~30 atoms in 6Å sphere is "buried"
        (count as f64 / 30.0).min(1.0)
    }

    /// Cluster favorable probe positions
    fn cluster_probes(&self, probes: &[Probe], atoms: &[Atom]) -> Vec<ProbeCluster> {
        if probes.is_empty() {
            return Vec::new();
        }

        let mut clusters: Vec<Vec<&Probe>> = Vec::new();
        let mut assigned = vec![false; probes.len()];
        let cluster_dist_sq = self.cluster_distance * self.cluster_distance;

        for i in 0..probes.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![&probes[i]];
            assigned[i] = true;

            // Grow cluster
            let mut changed = true;
            while changed {
                changed = false;
                for j in 0..probes.len() {
                    if assigned[j] {
                        continue;
                    }

                    let is_close = cluster.iter().any(|c| {
                        let dx = c.position[0] - probes[j].position[0];
                        let dy = c.position[1] - probes[j].position[1];
                        let dz = c.position[2] - probes[j].position[2];
                        (dx * dx + dy * dy + dz * dz) < cluster_dist_sq
                    });

                    if is_close {
                        cluster.push(&probes[j]);
                        assigned[j] = true;
                        changed = true;
                    }
                }
            }

            if cluster.len() >= self.min_cluster_size {
                clusters.push(cluster);
            }
        }

        // Convert to ProbeCluster structures
        clusters
            .into_iter()
            .enumerate()
            .map(|(id, cluster)| {
                let n = cluster.len() as f64;

                // Calculate centroid
                let centroid = [
                    cluster.iter().map(|p| p.position[0]).sum::<f64>() / n,
                    cluster.iter().map(|p| p.position[1]).sum::<f64>() / n,
                    cluster.iter().map(|p| p.position[2]).sum::<f64>() / n,
                ];

                let mean_energy = cluster.iter().map(|p| p.energy).sum::<f64>() / n;
                let best_energy = cluster
                    .iter()
                    .map(|p| p.energy)
                    .fold(0.0_f64, |a, b| a.min(b));

                // Find contact residues
                let contact_residues = self.find_contact_residues(&centroid, atoms, 5.0);

                // Binding score: normalize energy to 0-1
                let binding_score = (-mean_energy / 2.0).clamp(0.0, 1.0);

                ProbeCluster {
                    id: id + 1,
                    centroid,
                    num_probes: cluster.len(),
                    mean_energy,
                    best_energy,
                    contact_residues,
                    binding_score,
                }
            })
            .collect()
    }

    /// Find residues within distance of a position
    fn find_contact_residues(&self, pos: &[f64; 3], atoms: &[Atom], distance: f64) -> Vec<i32> {
        let dist_sq = distance * distance;
        let mut residues = std::collections::HashSet::new();

        for atom in atoms {
            let dx = atom.coord[0] - pos[0];
            let dy = atom.coord[1] - pos[1];
            let dz = atom.coord[2] - pos[2];

            if dx * dx + dy * dy + dz * dz < dist_sq {
                residues.insert(atom.residue_seq);
            }
        }

        residues.into_iter().collect()
    }

    /// Calculate per-residue binding propensities
    fn calculate_residue_propensities(
        &self,
        atoms: &[Atom],
        favorable_probes: &[Probe],
    ) -> Vec<ResidueBindingPropensity> {
        // Get unique residues
        let mut residues: Vec<(i32, char)> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for atom in atoms {
            let key = (atom.residue_seq, atom.chain_id);
            if !seen.contains(&key) {
                seen.insert(key);
                residues.push(key);
            }
        }

        let contact_dist = 5.0;
        let contact_dist_sq = contact_dist * contact_dist;

        residues
            .into_iter()
            .map(|(res_seq, chain_id)| {
                // Find probes near this residue
                let residue_atoms: Vec<&Atom> = atoms
                    .iter()
                    .filter(|a| a.residue_seq == res_seq && a.chain_id == chain_id)
                    .collect();

                let mut nearby_probes = Vec::new();
                for probe in favorable_probes {
                    for atom in &residue_atoms {
                        let dx = atom.coord[0] - probe.position[0];
                        let dy = atom.coord[1] - probe.position[1];
                        let dz = atom.coord[2] - probe.position[2];
                        if dx * dx + dy * dy + dz * dz < contact_dist_sq {
                            nearby_probes.push(probe);
                            break;
                        }
                    }
                }

                let probe_count = nearby_probes.len();
                let mean_probe_energy = if probe_count > 0 {
                    nearby_probes.iter().map(|p| p.energy).sum::<f64>() / probe_count as f64
                } else {
                    0.0
                };

                // Binding score based on probe count and energy
                let binding_score = if probe_count > 0 {
                    let count_score = (probe_count as f64 / 10.0).min(1.0);
                    let energy_score = (-mean_probe_energy / 2.0).clamp(0.0, 1.0);
                    0.6 * count_score + 0.4 * energy_score
                } else {
                    0.0
                };

                ResidueBindingPropensity {
                    residue_seq: res_seq,
                    chain_id,
                    probe_count,
                    mean_probe_energy,
                    binding_score,
                }
            })
            .collect()
    }

    /// Get residues with high binding propensity
    pub fn get_binding_hotspot_residues(&self, result: &ProbeClusteringResult) -> Vec<i32> {
        result
            .residue_propensities
            .iter()
            .filter(|r| r.binding_score > 0.5)
            .map(|r| r.residue_seq)
            .collect()
    }
}

/// Get Van der Waals radius for an element
fn get_vdw_radius(element: &str) -> f64 {
    match element.to_uppercase().as_str() {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        _ => 1.50,
    }
}

/// Convert probe clustering results to a residue -> score map
pub fn probe_to_score_map(result: &ProbeClusteringResult) -> HashMap<i32, f64> {
    result
        .residue_propensities
        .iter()
        .map(|r| (r.residue_seq, r.binding_score))
        .collect()
}

/// Alias for probe_to_score_map (for API consistency)
pub fn probe_clusters_to_score_map(result: &ProbeClusteringResult) -> HashMap<i32, f64> {
    probe_to_score_map(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_atom(serial: u32, residue_seq: i32, coord: [f64; 3]) -> Atom {
        Atom {
            serial,
            name: "CA".to_string(),
            residue_name: "ALA".to_string(),
            chain_id: 'A',
            residue_seq,
            insertion_code: None,
            coord,
            occupancy: 1.0,
            b_factor: 20.0,
            element: "C".to_string(),
            alt_loc: None,
            model: 1,
            is_hetatm: false,
            sasa: 0.0,
            hydrophobicity: 0.7,
            partial_charge: 0.0,
            is_surface: true,
            depth: 0.0,
            curvature: 0.0,
        }
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ProbeClusteringAnalyzer::new();
        assert_eq!(analyzer.grid_spacing, PROBE_GRID_SPACING);
    }

    #[test]
    fn test_vdw_radii() {
        assert!((get_vdw_radius("C") - 1.70).abs() < 0.01);
        assert!((get_vdw_radius("N") - 1.55).abs() < 0.01);
    }

    #[test]
    fn test_small_structure() {
        let analyzer = ProbeClusteringAnalyzer::new();

        // Create a small cluster of atoms
        let atoms: Vec<Atom> = (0..20)
            .map(|i| {
                let angle = (i as f64) * 0.3;
                let x = 5.0 * angle.cos();
                let y = 5.0 * angle.sin();
                let z = (i as f64) * 0.5;
                make_atom(i as u32, i as i32, [x, y, z])
            })
            .collect();

        let result = analyzer.analyze(&atoms);
        assert!(result.total_probes_placed > 0);
    }

    #[test]
    fn test_empty_input() {
        let analyzer = ProbeClusteringAnalyzer::new();
        let result = analyzer.analyze(&[]);
        assert_eq!(result.total_probes_placed, 0);
    }
}
