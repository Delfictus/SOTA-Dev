//! Production Druggability Scoring for Cryptic Binding Sites
//!
//! Physics-based druggability assessment using validated parameters from:
//! - SiteMap (Schrödinger) - hydrophobic/polar balance
//! - fpocket (INSERM) - pocket geometry
//! - DoGSiteScorer (BioSolveIT) - drug-likeness
//!
//! # Druggability Factors
//!
//! 1. **Hydrophobicity (35%)**: Lipophilic regions for drug binding
//! 2. **Enclosure (35%)**: Pocket depth and burial
//! 3. **H-bond capacity (30%)**: Polar contacts for specificity
//!
//! # Score Interpretation
//!
//! | Score | Classification | Drug Discovery Potential |
//! |-------|---------------|-------------------------|
//! | >0.8  | Highly druggable | Excellent target |
//! | 0.6-0.8 | Druggable | Good target |
//! | 0.4-0.6 | Challenging | Requires optimization |
//! | <0.4 | Difficult | PPI-like, needs fragments |

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Kyte-Doolittle hydrophobicity scale (normalized 0-1)
/// Higher = more hydrophobic
fn kyte_doolittle_normalized(residue: &str) -> f64 {
    match residue.to_uppercase().as_str() {
        "ILE" => 1.000,
        "VAL" => 0.967,
        "LEU" => 0.922,
        "PHE" => 0.811,
        "CYS" => 0.778,
        "MET" => 0.711,
        "ALA" => 0.700,
        "GLY" => 0.456,
        "THR" => 0.422,
        "SER" => 0.411,
        "TRP" => 0.400,
        "TYR" => 0.356,
        "PRO" => 0.322,
        "HIS" => 0.144,
        "GLU" => 0.111,
        "GLN" => 0.111,
        "ASP" => 0.111,
        "ASN" => 0.111,
        "LYS" => 0.067,
        "ARG" => 0.000,
        _ => 0.5, // Unknown residues get neutral score
    }
}

/// H-bond donor capacity (number of potential donors)
fn hbond_donor_count(residue: &str) -> u32 {
    match residue.to_uppercase().as_str() {
        "ARG" => 5, // Guanidinium + backbone
        "LYS" => 2, // Amino + backbone
        "ASN" => 2, // Amide + backbone
        "GLN" => 2, // Amide + backbone
        "HIS" => 2, // Imidazole + backbone
        "SER" => 2, // Hydroxyl + backbone
        "THR" => 2, // Hydroxyl + backbone
        "TYR" => 2, // Hydroxyl + backbone
        "TRP" => 2, // Indole NH + backbone
        "CYS" => 1, // Thiol (weak) + backbone
        _ => 1,     // Backbone NH only
    }
}

/// H-bond acceptor capacity (number of potential acceptors)
fn hbond_acceptor_count(residue: &str) -> u32 {
    match residue.to_uppercase().as_str() {
        "GLU" => 3, // Carboxylate + backbone
        "ASP" => 3, // Carboxylate + backbone
        "ASN" => 2, // Amide O + backbone
        "GLN" => 2, // Amide O + backbone
        "HIS" => 2, // Imidazole N + backbone
        "SER" => 2, // Hydroxyl + backbone
        "THR" => 2, // Hydroxyl + backbone
        "TYR" => 2, // Hydroxyl + backbone
        "MET" => 2, // Thioether + backbone
        _ => 1,     // Backbone C=O only
    }
}

/// Residue volume (Å³) for enclosure calculations
fn residue_volume(residue: &str) -> f64 {
    match residue.to_uppercase().as_str() {
        "GLY" => 60.1,
        "ALA" => 88.6,
        "SER" => 89.0,
        "CYS" => 108.5,
        "ASP" => 111.1,
        "PRO" => 112.7,
        "ASN" => 114.1,
        "THR" => 116.1,
        "GLU" => 138.4,
        "VAL" => 140.0,
        "GLN" => 143.8,
        "HIS" => 153.2,
        "MET" => 162.9,
        "ILE" => 166.7,
        "LEU" => 166.7,
        "LYS" => 168.6,
        "ARG" => 173.4,
        "PHE" => 189.9,
        "TYR" => 193.6,
        "TRP" => 227.8,
        _ => 140.0, // Average
    }
}

/// Druggability score result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DruggabilityScore {
    /// Overall druggability score (0-1)
    pub score: f64,

    /// Classification based on score
    pub classification: DruggabilityClass,

    /// Hydrophobicity component (0-1)
    pub hydrophobicity: f64,

    /// Enclosure/burial component (0-1)
    pub enclosure: f64,

    /// H-bond capacity component (0-1)
    pub hbond_capacity: f64,

    /// Estimated binding affinity range (kcal/mol)
    pub estimated_affinity_range: (f64, f64),

    /// Recommended fragment size (heavy atoms)
    pub recommended_fragment_size: (usize, usize),

    /// Detailed breakdown
    pub details: DruggabilityDetails,
}

/// Druggability classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DruggabilityClass {
    HighlyDruggable,
    Druggable,
    Challenging,
    Difficult,
}

impl DruggabilityClass {
    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            DruggabilityClass::HighlyDruggable
        } else if score >= 0.6 {
            DruggabilityClass::Druggable
        } else if score >= 0.4 {
            DruggabilityClass::Challenging
        } else {
            DruggabilityClass::Difficult
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DruggabilityClass::HighlyDruggable => "Highly Druggable",
            DruggabilityClass::Druggable => "Druggable",
            DruggabilityClass::Challenging => "Challenging",
            DruggabilityClass::Difficult => "Difficult",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            DruggabilityClass::HighlyDruggable => "#22c55e", // Green
            DruggabilityClass::Druggable => "#84cc16",        // Lime
            DruggabilityClass::Challenging => "#eab308",      // Yellow
            DruggabilityClass::Difficult => "#ef4444",        // Red
        }
    }
}

/// Detailed druggability breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DruggabilityDetails {
    /// Number of hydrophobic residues
    pub n_hydrophobic: usize,
    /// Number of polar residues
    pub n_polar: usize,
    /// Number of charged residues
    pub n_charged: usize,
    /// Total H-bond donors
    pub total_donors: u32,
    /// Total H-bond acceptors
    pub total_acceptors: u32,
    /// Pocket volume (Å³)
    pub pocket_volume: f64,
    /// Surface area (Å²)
    pub surface_area: f64,
    /// Burial fraction (0-1)
    pub burial_fraction: f64,
    /// Aromatic fraction
    pub aromatic_fraction: f64,
}

/// Druggability scorer using physics-based parameters
pub struct DruggabilityScorer {
    /// Weight for hydrophobicity component
    pub hydrophobic_weight: f64,
    /// Weight for enclosure component
    pub enclosure_weight: f64,
    /// Weight for H-bond component
    pub hbond_weight: f64,
    /// Optimal pocket volume range (Å³)
    pub optimal_volume_range: (f64, f64),
    /// Optimal hydrophobic fraction
    pub optimal_hydrophobic_fraction: f64,
}

impl Default for DruggabilityScorer {
    fn default() -> Self {
        Self {
            hydrophobic_weight: 0.35,
            enclosure_weight: 0.35,
            hbond_weight: 0.30,
            // Based on SiteMap analysis of drug binding sites
            optimal_volume_range: (200.0, 800.0),
            optimal_hydrophobic_fraction: 0.55, // 55% hydrophobic is optimal
        }
    }
}

impl DruggabilityScorer {
    /// Create scorer with custom weights
    pub fn with_weights(hydrophobic: f64, enclosure: f64, hbond: f64) -> Self {
        Self {
            hydrophobic_weight: hydrophobic,
            enclosure_weight: enclosure,
            hbond_weight: hbond,
            ..Default::default()
        }
    }

    /// Score a pocket defined by its residues and geometry
    ///
    /// # Arguments
    /// * `pocket_residues` - List of (residue_name, residue_id) in the pocket
    /// * `pocket_coords` - Coordinates of pocket-defining atoms [n_atoms][3]
    /// * `pocket_sasa` - Per-residue SASA values
    /// * `pocket_volume` - Pocket volume in Å³
    ///
    /// # Returns
    /// Complete druggability assessment
    pub fn score_pocket(
        &self,
        pocket_residues: &[(String, i32)],
        pocket_coords: &[[f64; 3]],
        pocket_sasa: &[f64],
        pocket_volume: f64,
    ) -> DruggabilityScore {
        if pocket_residues.is_empty() {
            return self.empty_score();
        }

        // Classify residues
        let mut n_hydrophobic = 0usize;
        let mut n_polar = 0usize;
        let mut n_charged = 0usize;
        let mut n_aromatic = 0usize;
        let mut total_donors = 0u32;
        let mut total_acceptors = 0u32;
        let mut hydrophobicity_sum = 0.0;

        let hydrophobic_set: HashSet<&str> = ["ILE", "VAL", "LEU", "PHE", "MET", "ALA", "TRP"]
            .iter().cloned().collect();
        let polar_set: HashSet<&str> = ["SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS"]
            .iter().cloned().collect();
        let charged_set: HashSet<&str> = ["ASP", "GLU", "LYS", "ARG"]
            .iter().cloned().collect();
        let aromatic_set: HashSet<&str> = ["PHE", "TYR", "TRP", "HIS"]
            .iter().cloned().collect();

        for (res_name, _) in pocket_residues {
            let res_upper = res_name.to_uppercase();
            let res_str = res_upper.as_str();

            hydrophobicity_sum += kyte_doolittle_normalized(res_str);
            total_donors += hbond_donor_count(res_str);
            total_acceptors += hbond_acceptor_count(res_str);

            if hydrophobic_set.contains(res_str) {
                n_hydrophobic += 1;
            } else if polar_set.contains(res_str) {
                n_polar += 1;
            }
            if charged_set.contains(res_str) {
                n_charged += 1;
            }
            if aromatic_set.contains(res_str) {
                n_aromatic += 1;
            }
        }

        let n_residues = pocket_residues.len();
        let hydrophobic_fraction = n_hydrophobic as f64 / n_residues as f64;
        let aromatic_fraction = n_aromatic as f64 / n_residues as f64;
        let mean_hydrophobicity = hydrophobicity_sum / n_residues as f64;

        // Calculate hydrophobicity score
        // Optimal is ~55% hydrophobic (based on drug binding site analysis)
        let hydrophobicity_deviation = (hydrophobic_fraction - self.optimal_hydrophobic_fraction).abs();
        let hydrophobicity_score = (1.0 - hydrophobicity_deviation * 2.0).max(0.0);

        // Boost for aromatic residues (good for π-stacking)
        let aromatic_boost = (aromatic_fraction * 0.5).min(0.2);
        let final_hydrophobicity = (hydrophobicity_score + aromatic_boost).min(1.0);

        // Calculate enclosure score
        // Based on pocket volume vs optimal range
        let volume_score = if pocket_volume < self.optimal_volume_range.0 {
            // Too small - linear penalty
            pocket_volume / self.optimal_volume_range.0
        } else if pocket_volume > self.optimal_volume_range.1 {
            // Too large - gradual penalty
            let excess = (pocket_volume - self.optimal_volume_range.1) / self.optimal_volume_range.1;
            (1.0 - excess * 0.5).max(0.3)
        } else {
            // In optimal range
            1.0
        };

        // Calculate burial from SASA (lower SASA = more buried = better enclosure)
        let total_sasa: f64 = pocket_sasa.iter().sum();
        let max_expected_sasa = n_residues as f64 * 120.0; // ~120 Å² per exposed residue
        let burial_fraction = 1.0 - (total_sasa / max_expected_sasa).min(1.0);

        // Combine for enclosure score
        let enclosure_score = 0.6 * volume_score + 0.4 * burial_fraction;

        // Calculate H-bond capacity score
        // Optimal: balanced donors and acceptors, total 4-12 for typical drugs
        let total_hbond = total_donors + total_acceptors;
        let donor_acceptor_ratio = if total_acceptors > 0 {
            total_donors as f64 / total_acceptors as f64
        } else {
            2.0 // Penalize no acceptors
        };

        // Optimal ratio is ~0.5-1.5 (based on Lipinski's rules)
        let ratio_score = if (0.5..=1.5).contains(&donor_acceptor_ratio) {
            1.0
        } else if (0.3..=2.0).contains(&donor_acceptor_ratio) {
            0.7
        } else {
            0.4
        };

        // Optimal total is 6-10 for drug-like binding
        let count_score = if (4..=12).contains(&total_hbond) {
            1.0
        } else if (2..=16).contains(&total_hbond) {
            0.7
        } else {
            0.4
        };

        let hbond_score = 0.6 * ratio_score + 0.4 * count_score;

        // Calculate final weighted score
        let final_score = self.hydrophobic_weight * final_hydrophobicity
            + self.enclosure_weight * enclosure_score
            + self.hbond_weight * hbond_score;

        // Estimate binding affinity range (empirical correlation)
        // Based on SiteMap Dscore to ΔG correlation
        let affinity_min = -6.0 - (final_score * 4.0);  // kcal/mol
        let affinity_max = -4.0 - (final_score * 5.0);  // kcal/mol

        // Recommend fragment size based on pocket volume
        let min_fragment = ((pocket_volume / 25.0) as usize).max(8).min(15);
        let max_fragment = ((pocket_volume / 15.0) as usize).max(15).min(35);

        // Calculate surface area from coordinates
        let surface_area = self.estimate_surface_area(pocket_coords);

        DruggabilityScore {
            score: final_score,
            classification: DruggabilityClass::from_score(final_score),
            hydrophobicity: final_hydrophobicity,
            enclosure: enclosure_score,
            hbond_capacity: hbond_score,
            estimated_affinity_range: (affinity_min, affinity_max),
            recommended_fragment_size: (min_fragment, max_fragment),
            details: DruggabilityDetails {
                n_hydrophobic,
                n_polar,
                n_charged,
                total_donors,
                total_acceptors,
                pocket_volume,
                surface_area,
                burial_fraction,
                aromatic_fraction,
            },
        }
    }

    /// Score a pocket using only residue names (simpler interface)
    pub fn score_simple(&self, residue_names: &[String], pocket_volume: f64) -> DruggabilityScore {
        let pocket_residues: Vec<(String, i32)> = residue_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i as i32))
            .collect();

        // Generate placeholder coordinates and SASA
        let n = residue_names.len();
        let coords: Vec<[f64; 3]> = (0..n).map(|i| [i as f64, 0.0, 0.0]).collect();
        let sasa: Vec<f64> = vec![50.0; n]; // Moderate exposure

        self.score_pocket(&pocket_residues, &coords, &sasa, pocket_volume)
    }

    /// Estimate surface area from coordinates using bounding box
    fn estimate_surface_area(&self, coords: &[[f64; 3]]) -> f64 {
        if coords.is_empty() {
            return 0.0;
        }

        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];

        for c in coords {
            for i in 0..3 {
                min[i] = min[i].min(c[i]);
                max[i] = max[i].max(c[i]);
            }
        }

        let dx = (max[0] - min[0]).max(1.0);
        let dy = (max[1] - min[1]).max(1.0);
        let dz = (max[2] - min[2]).max(1.0);

        // Approximate surface area as ellipsoid
        let a = dx / 2.0;
        let b = dy / 2.0;
        let c = dz / 2.0;

        // Thomsen's approximation for ellipsoid surface area
        let p = 1.6075;
        let ap = a.powf(p);
        let bp = b.powf(p);
        let cp = c.powf(p);
        4.0 * std::f64::consts::PI * ((ap * bp + ap * cp + bp * cp) / 3.0).powf(1.0 / p)
    }

    /// Return empty score for invalid input
    fn empty_score(&self) -> DruggabilityScore {
        DruggabilityScore {
            score: 0.0,
            classification: DruggabilityClass::Difficult,
            hydrophobicity: 0.0,
            enclosure: 0.0,
            hbond_capacity: 0.0,
            estimated_affinity_range: (-4.0, -2.0),
            recommended_fragment_size: (8, 15),
            details: DruggabilityDetails {
                n_hydrophobic: 0,
                n_polar: 0,
                n_charged: 0,
                total_donors: 0,
                total_acceptors: 0,
                pocket_volume: 0.0,
                surface_area: 0.0,
                burial_fraction: 0.0,
                aromatic_fraction: 0.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hydrophobicity_scale() {
        assert!(kyte_doolittle_normalized("ILE") > kyte_doolittle_normalized("ARG"));
        assert!(kyte_doolittle_normalized("ALA") > kyte_doolittle_normalized("ASP"));
        assert!((kyte_doolittle_normalized("GLY") - 0.456).abs() < 0.01);
    }

    #[test]
    fn test_hbond_counts() {
        assert!(hbond_donor_count("ARG") > hbond_donor_count("ALA"));
        assert!(hbond_acceptor_count("GLU") > hbond_acceptor_count("ALA"));
    }

    #[test]
    fn test_druggability_class() {
        assert_eq!(DruggabilityClass::from_score(0.9), DruggabilityClass::HighlyDruggable);
        assert_eq!(DruggabilityClass::from_score(0.7), DruggabilityClass::Druggable);
        assert_eq!(DruggabilityClass::from_score(0.5), DruggabilityClass::Challenging);
        assert_eq!(DruggabilityClass::from_score(0.2), DruggabilityClass::Difficult);
    }

    #[test]
    fn test_hydrophobic_pocket() {
        let scorer = DruggabilityScorer::default();

        // Highly hydrophobic pocket (like p38 DFG-out)
        let residues = vec![
            "LEU".to_string(), "ILE".to_string(), "VAL".to_string(),
            "PHE".to_string(), "MET".to_string(), "ALA".to_string(),
        ];

        let result = scorer.score_simple(&residues, 400.0);

        // Hydrophobic pockets should have decent druggability
        assert!(result.score > 0.4, "Hydrophobic pocket score: {}", result.score);
        assert!(result.details.n_hydrophobic >= 5);
    }

    #[test]
    fn test_polar_pocket() {
        let scorer = DruggabilityScorer::default();

        // Highly polar pocket (PPI-like)
        let residues = vec![
            "ARG".to_string(), "LYS".to_string(), "GLU".to_string(),
            "ASP".to_string(), "SER".to_string(), "THR".to_string(),
        ];

        let result = scorer.score_simple(&residues, 600.0);

        // Polar pockets are typically harder to drug
        assert!(result.details.n_charged >= 4);
        assert!(result.details.total_donors > 10);
    }

    #[test]
    fn test_balanced_pocket() {
        let scorer = DruggabilityScorer::default();

        // Balanced pocket (optimal for drug binding)
        let residues = vec![
            "LEU".to_string(), "ILE".to_string(), "VAL".to_string(), // Hydrophobic
            "PHE".to_string(), "TYR".to_string(),                    // Aromatic
            "SER".to_string(), "ASN".to_string(), "GLU".to_string(), // Polar/charged
        ];

        let result = scorer.score_simple(&residues, 450.0);

        // Balanced pockets should score well (Druggable or HighlyDruggable)
        assert!(result.score > 0.6, "Balanced pocket score: {}", result.score);
        assert!(
            matches!(result.classification, DruggabilityClass::Druggable | DruggabilityClass::HighlyDruggable),
            "Expected Druggable or HighlyDruggable, got {:?}", result.classification
        );
    }

    #[test]
    fn test_volume_effects() {
        let scorer = DruggabilityScorer::default();
        let residues = vec!["LEU".to_string(); 8];

        // Test volume in optimal range
        let result_optimal = scorer.score_simple(&residues, 400.0);

        // Test volume too small
        let result_small = scorer.score_simple(&residues, 80.0);

        // Test volume too large
        let result_large = scorer.score_simple(&residues, 2000.0);

        assert!(result_optimal.enclosure > result_small.enclosure);
        assert!(result_optimal.enclosure > result_large.enclosure);
    }

    #[test]
    fn test_empty_pocket() {
        let scorer = DruggabilityScorer::default();
        let result = scorer.score_simple(&[], 0.0);

        assert_eq!(result.score, 0.0);
        assert_eq!(result.classification, DruggabilityClass::Difficult);
    }
}
