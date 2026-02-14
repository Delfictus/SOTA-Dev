//! Stage 1B: Hinge Region Detection
//!
//! Detects flexible hinge regions that enable domain motion using:
//! - B-factor gradient analysis (rate of change of flexibility)
//! - Secondary structure assignment (hinges typically in loops)
//! - Spatial clustering of hinge candidates
//!
//! Hinge regions are critical for allosteric communication as they
//! transmit conformational changes between domains.

use crate::structure::Atom;
use super::types::*;
use super::domain_decomposition::euclidean_distance;
use std::collections::HashMap;

/// Hinge region detector using B-factor gradients
pub struct HingeDetector {
    /// Window size for B-factor gradient calculation
    pub gradient_window: usize,
    /// Minimum gradient to consider a hinge
    pub min_gradient: f64,
    /// Distance cutoff for clustering hinge residues
    pub cluster_cutoff: f64,
    /// Minimum residues in a hinge cluster
    pub min_cluster_size: usize,
}

impl Default for HingeDetector {
    fn default() -> Self {
        Self {
            gradient_window: 3,
            min_gradient: 0.5,
            cluster_cutoff: 8.0,
            min_cluster_size: 2,
        }
    }
}

impl HingeDetector {
    pub fn new(gradient_window: usize, min_gradient: f64) -> Self {
        Self {
            gradient_window,
            min_gradient,
            ..Default::default()
        }
    }

    /// Detect hinge regions in structure
    pub fn detect(&self, atoms: &[Atom]) -> Vec<HingeRegion> {
        // Step 1: Calculate per-residue B-factors
        let residue_bfactors = self.calculate_residue_bfactors(atoms);

        // Step 2: Calculate B-factor gradients (rate of change)
        let gradients = self.calculate_bfactor_gradients(&residue_bfactors);

        // Step 3: Assign secondary structure
        let secondary_structure = self.assign_secondary_structure(atoms);

        // Step 4: Identify hinge candidates (high gradient + loop region)
        let candidates = self.find_hinge_candidates(&gradients, &secondary_structure);

        // Step 5: Cluster nearby hinge residues
        self.cluster_hinges(atoms, candidates)
    }

    /// Detect hinges with domain context
    pub fn detect_with_domains(
        &self,
        atoms: &[Atom],
        domains: &[Domain],
    ) -> Vec<HingeRegion> {
        let mut hinges = self.detect(atoms);

        // Annotate hinges with connected domains
        for hinge in &mut hinges {
            hinge.connected_domains = self.find_connected_domains(
                atoms,
                &hinge.residues,
                domains,
            );
        }

        hinges
    }

    fn calculate_residue_bfactors(&self, atoms: &[Atom]) -> HashMap<i32, f64> {
        let mut residue_bfactors: HashMap<i32, (f64, usize)> = HashMap::new();

        // For each residue, average the B-factors of backbone atoms
        for atom in atoms.iter() {
            let name = atom.name.trim();
            // Backbone atoms: N, CA, C, O
            if name == "N" || name == "CA" || name == "C" || name == "O" {
                let entry = residue_bfactors
                    .entry(atom.residue_seq)
                    .or_insert((0.0, 0));
                entry.0 += atom.b_factor;
                entry.1 += 1;
            }
        }

        residue_bfactors
            .into_iter()
            .filter_map(|(res, (sum, count))| {
                if count > 0 {
                    Some((res, sum / count as f64))
                } else {
                    None
                }
            })
            .collect()
    }

    fn calculate_bfactor_gradients(
        &self,
        bfactors: &HashMap<i32, f64>,
    ) -> HashMap<i32, f64> {
        let mut sorted_residues: Vec<i32> = bfactors.keys().copied().collect();
        sorted_residues.sort();

        let mut gradients = HashMap::new();
        let window = self.gradient_window;

        // Need at least 2*window+1 residues
        if sorted_residues.len() < 2 * window + 1 {
            return gradients;
        }

        // Calculate gradient for interior residues
        for i in window..sorted_residues.len() - window {
            let res = sorted_residues[i];

            // Left average
            let left_avg: f64 = (1..=window)
                .filter_map(|j| {
                    let left_res = sorted_residues[i - j];
                    bfactors.get(&left_res)
                })
                .sum::<f64>()
                / window as f64;

            // Right average
            let right_avg: f64 = (1..=window)
                .filter_map(|j| {
                    let right_res = sorted_residues[i + j];
                    bfactors.get(&right_res)
                })
                .sum::<f64>()
                / window as f64;

            // Gradient magnitude (absolute change across the residue)
            let gradient = (right_avg - left_avg).abs() / (2 * window) as f64;

            gradients.insert(res, gradient);
        }

        gradients
    }

    fn assign_secondary_structure(&self, atoms: &[Atom]) -> HashMap<i32, SecondaryStructure> {
        // Simplified DSSP-like assignment based on backbone geometry
        // For production, integrate actual DSSP or STRIDE

        let ca_atoms: Vec<&Atom> = atoms
            .iter()
            .filter(|a| a.name.trim() == "CA")
            .collect();

        let mut ss_assignment = HashMap::new();

        for (i, ca) in ca_atoms.iter().enumerate() {
            // Calculate local curvature using Cα positions
            let ss = if i >= 2 && i < ca_atoms.len() - 2 {
                let curvature = self.calculate_backbone_curvature(&ca_atoms[i - 2..=i + 2]);
                let twist = self.calculate_backbone_twist(&ca_atoms, i);

                // Classification based on curvature and twist
                if curvature < 0.15 && twist.abs() < 30.0 {
                    SecondaryStructure::Helix
                } else if curvature < 0.30 && twist.abs() > 100.0 {
                    SecondaryStructure::Strand
                } else if curvature > 0.50 {
                    SecondaryStructure::Turn
                } else {
                    SecondaryStructure::Coil
                }
            } else {
                SecondaryStructure::Coil // Terminal regions
            };

            ss_assignment.insert(ca.residue_seq, ss);
        }

        ss_assignment
    }

    fn calculate_backbone_curvature(&self, ca_atoms: &[&Atom]) -> f64 {
        if ca_atoms.len() < 5 {
            return 0.5;
        }

        // Calculate angle at central residue (index 2)
        let p0 = &ca_atoms[0].coord;
        let p1 = &ca_atoms[2].coord;
        let p2 = &ca_atoms[4].coord;

        // Vectors from central to flanking residues
        let v1 = [p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]];
        let v2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];

        // Dot product
        let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];

        // Magnitudes
        let mag1 = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();
        let mag2 = (v2[0].powi(2) + v2[1].powi(2) + v2[2].powi(2)).sqrt();

        if mag1 < 1e-6 || mag2 < 1e-6 {
            return 0.5;
        }

        let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
        let angle = cos_angle.acos();

        // Convert angle to curvature (1.0 = highly curved, 0.0 = straight)
        1.0 - (angle / std::f64::consts::PI)
    }

    fn calculate_backbone_twist(&self, ca_atoms: &[&Atom], center_idx: usize) -> f64 {
        // Calculate dihedral-like twist angle
        if center_idx < 1 || center_idx >= ca_atoms.len() - 1 {
            return 0.0;
        }

        // Use three consecutive Cα atoms
        let p0 = &ca_atoms[center_idx - 1].coord;
        let p1 = &ca_atoms[center_idx].coord;
        let p2 = &ca_atoms[center_idx + 1].coord;

        // Simple twist metric based on out-of-plane deviation
        let v01 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v12 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];

        // Cross product for normal
        let cross = [
            v01[1] * v12[2] - v01[2] * v12[1],
            v01[2] * v12[0] - v01[0] * v12[2],
            v01[0] * v12[1] - v01[1] * v12[0],
        ];

        let cross_mag = (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
        let v01_mag = (v01[0].powi(2) + v01[1].powi(2) + v01[2].powi(2)).sqrt();
        let v12_mag = (v12[0].powi(2) + v12[1].powi(2) + v12[2].powi(2)).sqrt();

        if v01_mag < 1e-6 || v12_mag < 1e-6 {
            return 0.0;
        }

        // Twist angle in degrees
        let sin_twist = cross_mag / (v01_mag * v12_mag);
        sin_twist.asin().to_degrees()
    }

    fn find_hinge_candidates(
        &self,
        gradients: &HashMap<i32, f64>,
        secondary_structure: &HashMap<i32, SecondaryStructure>,
    ) -> Vec<HingeCandidate> {
        let mut candidates = Vec::new();

        for (&residue, &gradient) in gradients {
            if gradient > self.min_gradient {
                let ss = secondary_structure
                    .get(&residue)
                    .cloned()
                    .unwrap_or(SecondaryStructure::Coil);

                // Hinges are typically in loops/coils/turns
                if matches!(
                    ss,
                    SecondaryStructure::Coil | SecondaryStructure::Turn
                ) {
                    candidates.push(HingeCandidate {
                        residue_seq: residue,
                        gradient,
                        secondary_structure: ss,
                    });
                }
            }
        }

        candidates
    }

    fn cluster_hinges(
        &self,
        atoms: &[Atom],
        candidates: Vec<HingeCandidate>,
    ) -> Vec<HingeRegion> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // Get Cα coordinates for candidates
        let ca_coords: HashMap<i32, [f64; 3]> = atoms
            .iter()
            .filter(|a| a.name.trim() == "CA")
            .map(|a| (a.residue_seq, a.coord))
            .collect();

        // Cluster candidates spatially
        let mut clusters: Vec<Vec<HingeCandidate>> = Vec::new();
        let mut assigned = vec![false; candidates.len()];

        for (i, candidate) in candidates.iter().enumerate() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![candidate.clone()];
            assigned[i] = true;

            let coord_i = match ca_coords.get(&candidate.residue_seq) {
                Some(c) => c,
                None => continue,
            };

            // Find all nearby candidates
            for (j, other) in candidates.iter().enumerate().skip(i + 1) {
                if assigned[j] {
                    continue;
                }

                if let Some(coord_j) = ca_coords.get(&other.residue_seq) {
                    let dist = euclidean_distance(coord_i, coord_j);
                    if dist < self.cluster_cutoff {
                        cluster.push(other.clone());
                        assigned[j] = true;
                    }
                }
            }

            clusters.push(cluster);
        }

        // Convert clusters to HingeRegion objects
        clusters
            .into_iter()
            .filter(|c| c.len() >= self.min_cluster_size)
            .enumerate()
            .map(|(_, cluster)| {
                let residues: Vec<i32> = cluster.iter().map(|c| c.residue_seq).collect();

                // Find center residue (highest gradient)
                let center = cluster
                    .iter()
                    .max_by(|a, b| a.gradient.partial_cmp(&b.gradient).unwrap())
                    .map(|c| c.residue_seq)
                    .unwrap_or(residues[0]);

                // Average gradient magnitude
                let avg_gradient: f64 =
                    cluster.iter().map(|c| c.gradient).sum::<f64>() / cluster.len() as f64;

                // Most common secondary structure
                let ss = cluster
                    .first()
                    .map(|c| c.secondary_structure.clone())
                    .unwrap_or(SecondaryStructure::Coil);

                // Flexibility score (normalized gradient)
                let max_gradient = 2.0; // Empirical maximum
                let flexibility = (avg_gradient / max_gradient).clamp(0.0, 1.0);

                HingeRegion {
                    center_residue: center,
                    residues,
                    gradient_magnitude: avg_gradient,
                    secondary_structure: ss,
                    connected_domains: Vec::new(),
                    flexibility_score: flexibility,
                }
            })
            .collect()
    }

    fn find_connected_domains(
        &self,
        atoms: &[Atom],
        hinge_residues: &[i32],
        domains: &[Domain],
    ) -> Vec<usize> {
        let mut connected = Vec::new();
        let contact_cutoff = 10.0;

        // Get Cα coordinates
        let ca_coords: HashMap<i32, [f64; 3]> = atoms
            .iter()
            .filter(|a| a.name.trim() == "CA")
            .map(|a| (a.residue_seq, a.coord))
            .collect();

        for domain in domains {
            // Check if any hinge residue is close to any domain residue
            let is_connected = hinge_residues.iter().any(|&hinge_res| {
                domain.residues.iter().any(|&domain_res| {
                    if let (Some(h_coord), Some(d_coord)) =
                        (ca_coords.get(&hinge_res), ca_coords.get(&domain_res))
                    {
                        euclidean_distance(h_coord, d_coord) < contact_cutoff
                    } else {
                        false
                    }
                })
            });

            if is_connected {
                connected.push(domain.id);
            }
        }

        connected
    }
}

/// Intermediate hinge candidate
#[derive(Debug, Clone)]
struct HingeCandidate {
    residue_seq: i32,
    gradient: f64,
    secondary_structure: SecondaryStructure,
}
