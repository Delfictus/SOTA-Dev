//! Tier 1 and Tier 2 correlation analysis

use crate::inputs::{HoloStructure, TopologyData, TruthResidues};
use crate::sites::CrypticSite;
use serde::{Deserialize, Serialize};

/// Complete correlation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult {
    /// Tier 1 results (if holo provided)
    pub tier1: Option<Tier1Correlation>,
    /// Tier 2 results (if truth provided)
    pub tier2: Option<Tier2Correlation>,
    /// Overall hit metrics
    pub hit_metrics: HitMetrics,
}

/// Tier 1: Holo ligand proximity correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Correlation {
    /// Per-site correlation results
    pub site_correlations: Vec<SiteTier1>,
    /// Best site (closest to ligand)
    pub best_site_id: String,
    /// Best distance to ligand
    pub best_distance: f32,
    /// Mean distance across all sites
    pub mean_distance: f32,
    /// Number of sites within 5Å of ligand
    pub sites_within_5a: usize,
    /// Number of sites within 8Å of ligand
    pub sites_within_8a: usize,
}

/// Per-site Tier 1 correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteTier1 {
    /// Site ID
    pub site_id: String,
    /// Distance to nearest ligand atom (Å)
    pub nearest_ligand_atom_distance_a: f32,
    /// Distance to ligand centroid (Å)
    pub ligand_centroid_distance_a: f32,
    /// Residue recall: fraction of site residues within 4Å of ligand
    pub residue_recall_within_4a: f64,
    /// Is this a "hit" (distance < 5Å)?
    pub is_hit: bool,
}

impl Tier1Correlation {
    /// Compute Tier 1 correlation for sites against holo structure
    pub fn compute(
        sites: &[CrypticSite],
        holo: &HoloStructure,
        topology: &TopologyData,
    ) -> Self {
        let mut site_correlations = Vec::new();
        let mut best_distance = f32::INFINITY;
        let mut best_site_id = String::new();
        let mut total_distance = 0.0f32;
        let mut sites_within_5a = 0;
        let mut sites_within_8a = 0;

        for site in sites {
            // Distance from site centroid to nearest ligand atom
            let nearest_dist = holo.distance_to_ligand(site.centroid);
            let centroid_dist = holo.distance_to_centroid(site.centroid);

            // Compute residue recall within 4Å of ligand
            let mut residues_near_ligand = 0;
            for &res_id in &site.residues {
                let res_centroid = topology.residue_centroid(res_id);
                let dist = holo.distance_to_ligand(res_centroid);
                if dist < 4.0 {
                    residues_near_ligand += 1;
                }
            }
            let recall = if site.residues.is_empty() {
                0.0
            } else {
                residues_near_ligand as f64 / site.residues.len() as f64
            };

            let is_hit = nearest_dist < 5.0;

            if nearest_dist < best_distance {
                best_distance = nearest_dist;
                best_site_id = site.site_id.clone();
            }

            total_distance += nearest_dist;

            if nearest_dist < 5.0 {
                sites_within_5a += 1;
            }
            if nearest_dist < 8.0 {
                sites_within_8a += 1;
            }

            site_correlations.push(SiteTier1 {
                site_id: site.site_id.clone(),
                nearest_ligand_atom_distance_a: nearest_dist,
                ligand_centroid_distance_a: centroid_dist,
                residue_recall_within_4a: recall,
                is_hit,
            });
        }

        let mean_distance = if sites.is_empty() {
            0.0
        } else {
            total_distance / sites.len() as f32
        };

        Self {
            site_correlations,
            best_site_id,
            best_distance,
            mean_distance,
            sites_within_5a,
            sites_within_8a,
        }
    }
}

/// Tier 2: Truth residue precision/recall correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier2Correlation {
    /// Per-site correlation results
    pub site_correlations: Vec<SiteTier2>,
    /// Best site by F1
    pub best_site_id: String,
    /// Best F1 score
    pub best_f1: f64,
    /// Hit@1: Is the top-ranked site a hit (F1 > 0.3)?
    pub hit_at_1: bool,
    /// Hit@3: Is any of top 3 sites a hit?
    pub hit_at_3: bool,
    /// Mean F1 across all sites
    pub mean_f1: f64,
}

/// Per-site Tier 2 correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteTier2 {
    /// Site ID
    pub site_id: String,
    /// Precision: correct_residues / predicted_residues
    pub precision: f64,
    /// Recall: correct_residues / truth_residues
    pub recall: f64,
    /// F1 score: 2 * precision * recall / (precision + recall)
    pub f1: f64,
    /// Number of correct (overlapping) residues
    pub correct_residues: usize,
    /// Is this a "hit" (F1 > 0.3)?
    pub is_hit: bool,
}

impl Tier2Correlation {
    /// Compute Tier 2 correlation for sites against truth residues
    pub fn compute(sites: &[CrypticSite], truth: &TruthResidues) -> Self {
        let mut site_correlations = Vec::new();
        let mut best_f1 = 0.0;
        let mut best_site_id = String::new();
        let mut total_f1 = 0.0;

        for site in sites {
            let precision = truth.precision(&site.residues);
            let recall = truth.recall(&site.residues);
            let f1 = truth.f1(&site.residues);

            // Count correct residues
            let truth_set: std::collections::HashSet<_> = truth.residues.iter().collect();
            let correct = site.residues.iter().filter(|r| truth_set.contains(r)).count();

            let is_hit = f1 > 0.3;

            if f1 > best_f1 {
                best_f1 = f1;
                best_site_id = site.site_id.clone();
            }

            total_f1 += f1;

            site_correlations.push(SiteTier2 {
                site_id: site.site_id.clone(),
                precision,
                recall,
                f1,
                correct_residues: correct,
                is_hit,
            });
        }

        // Sort by rank for hit@1 and hit@3
        let mut sorted_sites = site_correlations.clone();
        sorted_sites.sort_by(|a, b| {
            let idx_a = sites.iter().position(|s| s.site_id == a.site_id).unwrap_or(usize::MAX);
            let idx_b = sites.iter().position(|s| s.site_id == b.site_id).unwrap_or(usize::MAX);
            idx_a.cmp(&idx_b)
        });

        let hit_at_1 = sorted_sites.first().map(|s| s.is_hit).unwrap_or(false);
        let hit_at_3 = sorted_sites.iter().take(3).any(|s| s.is_hit);

        let mean_f1 = if sites.is_empty() {
            0.0
        } else {
            total_f1 / sites.len() as f64
        };

        Self {
            site_correlations,
            best_site_id,
            best_f1,
            hit_at_1,
            hit_at_3,
            mean_f1,
        }
    }
}

/// Hit metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitMetrics {
    /// Number of sites
    pub n_sites: usize,
    /// Tier 1 hit@1 (if applicable)
    pub tier1_hit_at_1: Option<bool>,
    /// Tier 1 hit@3
    pub tier1_hit_at_3: Option<bool>,
    /// Tier 2 hit@1 (if applicable)
    pub tier2_hit_at_1: Option<bool>,
    /// Tier 2 hit@3
    pub tier2_hit_at_3: Option<bool>,
}

impl CorrelationResult {
    /// Compute correlation for sites
    pub fn compute(
        sites: &[CrypticSite],
        holo: Option<&HoloStructure>,
        truth: Option<&TruthResidues>,
        topology: &TopologyData,
    ) -> Self {
        let tier1 = holo.map(|h| Tier1Correlation::compute(sites, h, topology));
        let tier2 = truth.map(|t| Tier2Correlation::compute(sites, t));

        let tier1_hit_at_1 = tier1.as_ref().map(|t| {
            t.site_correlations.first().map(|s| s.is_hit).unwrap_or(false)
        });
        let tier1_hit_at_3 = tier1.as_ref().map(|t| {
            t.site_correlations.iter().take(3).any(|s| s.is_hit)
        });

        let hit_metrics = HitMetrics {
            n_sites: sites.len(),
            tier1_hit_at_1,
            tier1_hit_at_3,
            tier2_hit_at_1: tier2.as_ref().map(|t| t.hit_at_1),
            tier2_hit_at_3: tier2.as_ref().map(|t| t.hit_at_3),
        };

        Self {
            tier1,
            tier2,
            hit_metrics,
        }
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("site_id,rank,tier1_nearest_dist,tier1_centroid_dist,tier1_recall_4a,tier1_hit,tier2_precision,tier2_recall,tier2_f1,tier2_hit\n");

        // Combine tier1 and tier2 data
        let tier1_map: std::collections::HashMap<_, _> = self
            .tier1
            .as_ref()
            .map(|t| {
                t.site_correlations
                    .iter()
                    .map(|s| (s.site_id.clone(), s))
                    .collect()
            })
            .unwrap_or_default();

        let tier2_map: std::collections::HashMap<_, _> = self
            .tier2
            .as_ref()
            .map(|t| {
                t.site_correlations
                    .iter()
                    .map(|s| (s.site_id.clone(), s))
                    .collect()
            })
            .unwrap_or_default();

        // Get all site IDs
        let mut all_sites: Vec<_> = tier1_map.keys().chain(tier2_map.keys()).collect();
        all_sites.sort();
        all_sites.dedup();

        for (rank, site_id) in all_sites.iter().enumerate() {
            let t1 = tier1_map.get(*site_id);
            let t2 = tier2_map.get(*site_id);

            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{}\n",
                site_id,
                rank + 1,
                t1.map(|s| format!("{:.2}", s.nearest_ligand_atom_distance_a))
                    .unwrap_or_else(|| "NA".to_string()),
                t1.map(|s| format!("{:.2}", s.ligand_centroid_distance_a))
                    .unwrap_or_else(|| "NA".to_string()),
                t1.map(|s| format!("{:.3}", s.residue_recall_within_4a))
                    .unwrap_or_else(|| "NA".to_string()),
                t1.map(|s| s.is_hit.to_string())
                    .unwrap_or_else(|| "NA".to_string()),
                t2.map(|s| format!("{:.3}", s.precision))
                    .unwrap_or_else(|| "NA".to_string()),
                t2.map(|s| format!("{:.3}", s.recall))
                    .unwrap_or_else(|| "NA".to_string()),
                t2.map(|s| format!("{:.3}", s.f1))
                    .unwrap_or_else(|| "NA".to_string()),
                t2.map(|s| s.is_hit.to_string())
                    .unwrap_or_else(|| "NA".to_string()),
            ));
        }

        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sites::{
        ChemistryMetrics, GeometryMetrics, PersistenceMetrics, SiteMetrics, UvResponseMetrics,
    };

    fn make_test_site(id: &str, residues: Vec<usize>, centroid: [f32; 3]) -> CrypticSite {
        CrypticSite {
            site_id: id.to_string(),
            rank: 1,
            centroid,
            residues,
            residue_names: vec!["ALA".to_string(); 3],
            chain_id: "A".to_string(),
            metrics: SiteMetrics {
                persistence: PersistenceMetrics {
                    present_fraction: 0.5,
                    mean_lifetime_frames: 100.0,
                    replica_agreement: 0.8,
                },
                geometry: GeometryMetrics {
                    volume_mean: 300.0,
                    volume_p50: 280.0,
                    volume_p95: 400.0,
                    volume_min: 180.0,
                    volume_max: 420.0,
                    volume_std: 60.0,
                    breathing_amplitude: 240.0,
                    aspect_ratio: Some(1.5),
                    sphericity: Some(0.67),
                    depth_proxy_pocket_a: None,
                    depth_proxy_surface_a: None,
                    mouth_area_proxy_a2: None,
                    mouth_area_total_a2: None,
                    n_openings: None,
                },
                chemistry: ChemistryMetrics {
                    hydrophobic_fraction: 0.6,
                    donor_count: 5,
                    acceptor_count: 4,
                    aromatic_fraction: 0.2,
                    charged_fraction: 0.1,
                },
                uv_response: UvResponseMetrics::default(),
            },
            rank_score: 0.7,
            confidence: 0.8,
            is_druggable: true,
            first_frame: 0,
            last_frame: 1000,
            representative_frame: 500,
        }
    }

    #[test]
    fn test_tier2_precision_recall() {
        let truth = TruthResidues {
            residues: vec![10, 11, 12, 13, 14],
            site_name: Some("test".to_string()),
            notes: None,
        };

        // Site with 3 correct residues out of 4 predicted
        let predicted = vec![10, 11, 12, 99];
        assert!((truth.precision(&predicted) - 0.75).abs() < 0.01); // 3/4
        assert!((truth.recall(&predicted) - 0.6).abs() < 0.01); // 3/5
    }

    #[test]
    fn test_csv_export() {
        let sites = vec![make_test_site("site_001", vec![1, 2, 3], [0.0, 0.0, 0.0])];
        let truth = TruthResidues {
            residues: vec![1, 2, 5],
            site_name: None,
            notes: None,
        };

        let result = CorrelationResult::compute(&sites, None, Some(&truth), &TopologyData {
            source_pdb: String::new(),
            n_atoms: 0,
            n_residues: 0,
            n_chains: 0,
            positions: vec![],
            atom_names: vec![],
            residue_names: vec![],
            residue_ids: vec![],
            chain_ids: vec![],
            aromatic_targets: vec![],
            n_aromatics: 0,
            ca_indices: vec![],
        });

        let csv = result.to_csv();
        assert!(csv.contains("site_001"));
        assert!(csv.contains("tier2_precision"));
    }
}
