//! Glycan Shield Modeling (Stage 0 Preprocessing)
//!
//! Nipah F and G proteins are heavily glycosylated. N-linked glycans at N-X-S/T sequons
//! physically occlude epitopes. This module implements glycan masking to adjust SASA
//! calculations and modify cryptic site definitions.

use crate::structure_types::ParamyxoStructure;
use regex::Regex;
use std::collections::HashSet;

/// Radius around glycosylation sites for masking (Angstroms)
pub const GLYCAN_RADIUS: f32 = 10.0;

/// Glycan occlusion factor (70% occlusion)
pub const GLYCAN_OCCLUSION_FACTOR: f32 = 0.3;

/// Sequon pattern for N-linked glycosylation (N-X-S/T where X != P)
pub const SEQUON_PATTERN: &str = r"N[^P][ST]";

/// Glycan masking data structure
#[derive(Debug, Clone)]
pub struct GlycanMask {
    /// Residues within GLYCAN_RADIUS of any sequon Asn
    pub shielded_residues: HashSet<u32>,
    /// Positions of N-glycosylation sequons (Asn positions)
    pub sequon_positions: Vec<u32>,
}

impl GlycanMask {
    /// Create glycan mask from protein sequence and CA coordinates
    pub fn from_sequence(sequence: &str, ca_coords: &[(f32, f32, f32)]) -> Self {
        let regex = Regex::new(SEQUON_PATTERN).unwrap();
        let mut sequon_positions = Vec::new();

        // Find all N-X-S/T sequons in the sequence
        for mat in regex.find_iter(sequence) {
            let asn_idx = mat.start() as u32;
            sequon_positions.push(asn_idx);
        }

        // Compute shielded residues (within GLYCAN_RADIUS of any sequon Asn)
        let mut shielded = HashSet::new();
        for &asn_idx in &sequon_positions {
            if let Some(asn_coord) = ca_coords.get(asn_idx as usize) {
                for (res_idx, coord) in ca_coords.iter().enumerate() {
                    let dist = distance_3d(*asn_coord, *coord);
                    if dist < GLYCAN_RADIUS {
                        shielded.insert(res_idx as u32);
                    }
                }
            }
        }

        Self {
            shielded_residues: shielded,
            sequon_positions,
        }
    }

    /// Create glycan mask from ParamyxoStructure
    pub fn from_structure(structure: &ParamyxoStructure) -> Self {
        let ca_coords = structure.ca_coords();
        Self::from_sequence(&structure.sequence, &ca_coords)
    }

    /// Adjust SASA values by glycan occlusion
    pub fn adjust_sasa(&self, raw_sasa: &[f32]) -> Vec<f32> {
        raw_sasa
            .iter()
            .enumerate()
            .map(|(i, &sasa)| {
                if self.shielded_residues.contains(&(i as u32)) {
                    sasa * GLYCAN_OCCLUSION_FACTOR
                } else {
                    sasa
                }
            })
            .collect()
    }

    /// Check if residue is a valid cryptic site candidate
    /// Excludes glycan-shielded residues from cryptic site consideration
    pub fn is_valid_cryptic_candidate(&self, res_idx: u32) -> bool {
        !self.shielded_residues.contains(&res_idx)
    }

    /// Get number of shielded residues
    pub fn num_shielded(&self) -> usize {
        self.shielded_residues.len()
    }

    /// Get number of glycosylation sites
    pub fn num_glycosylation_sites(&self) -> usize {
        self.sequon_positions.len()
    }

    /// Check if a specific residue is glycan-shielded
    pub fn is_shielded(&self, res_idx: u32) -> bool {
        self.shielded_residues.contains(&res_idx)
    }

    /// Get effective SASA for cryptic site analysis
    pub fn effective_sasa(&self, raw_sasa: &[f32], res_idx: usize) -> f32 {
        if self.shielded_residues.contains(&(res_idx as u32)) {
            raw_sasa[res_idx] * GLYCAN_OCCLUSION_FACTOR
        } else {
            raw_sasa[res_idx]
        }
    }
}

/// Compute 3D Euclidean distance between two points
fn distance_3d(p1: (f32, f32, f32), p2: (f32, f32, f32)) -> f32 {
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;
    let dz = p1.2 - p2.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Known Nipah virus glycosylation sites based on experimental data
pub mod known_sites {
    use super::*;

    /// Known N-glycosylation sites in Nipah G protein
    pub const NIV_G_GLYCOSYLATION_SITES: &[u32] = &[64, 378, 417];

    /// Known N-glycosylation sites in Nipah F protein
    pub const NIV_F_GLYCOSYLATION_SITES: &[u32] = &[67, 238, 414];

    /// Get known glycosylation sites for a given protein
    pub fn get_known_sites(pdb_id: &str, protein_type: crate::structure_types::ProteinType) -> &'static [u32] {
        use crate::structure_types::ProteinType;

        // For Nipah virus structures
        if pdb_id.starts_with("8X") || pdb_id.starts_with("7U") || pdb_id.starts_with("7T") {
            match protein_type {
                ProteinType::GProtein => NIV_G_GLYCOSYLATION_SITES,
                ProteinType::FProtein => NIV_F_GLYCOSYLATION_SITES,
                _ => &[],
            }
        } else {
            // For Hendra virus or other structures, return empty for now
            // Can be extended with HeV-specific sites
            &[]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequon_detection() {
        let sequence = "MKLLNGSKNVTNNSTGDACPRQVPKN"; // Contains N-G-S at position 6
        let ca_coords = vec![(0.0, 0.0, 0.0); sequence.len()];

        let mask = GlycanMask::from_sequence(sequence, &ca_coords);
        assert!(mask.sequon_positions.contains(&6)); // N-G-S sequon
        assert!(!mask.sequon_positions.contains(&0)); // No sequon at start
    }

    #[test]
    fn test_sasa_adjustment() {
        let mask = GlycanMask {
            shielded_residues: [0, 2, 4].iter().cloned().collect(),
            sequon_positions: vec![1],
        };

        let raw_sasa = vec![100.0, 80.0, 60.0, 40.0, 20.0];
        let adjusted = mask.adjust_sasa(&raw_sasa);

        assert_eq!(adjusted[0], 30.0); // 100 * 0.3
        assert_eq!(adjusted[1], 80.0); // Unshielded
        assert_eq!(adjusted[2], 18.0); // 60 * 0.3
        assert_eq!(adjusted[3], 40.0); // Unshielded
        assert_eq!(adjusted[4], 6.0);  // 20 * 0.3
    }
}