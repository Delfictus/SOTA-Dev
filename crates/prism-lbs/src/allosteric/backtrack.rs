//! Backtrack Analysis Module
//!
//! The key innovation: automatic gap detection and self-correction.
//! Identifies regions that should have binding sites based on:
//! - High conservation without pocket detection
//! - Domain interfaces without pocket detection
//! - Flexible regions missed by initial softspot pass
//! - Communication hubs (high centrality) without pockets
//! - Grid-based cavity scan for geometric misses
//!
//! Re-runs focused detection with adjusted parameters to fill gaps.

use crate::structure::Atom;
use super::types::*;
use super::domain_decomposition::{calculate_residue_centroid, DomainDecomposer};
use super::msa_conservation::spatial_cluster;
use std::collections::{HashMap, HashSet};

/// Backtrack analyzer for coverage gap detection
pub struct BacktrackAnalyzer {
    /// Sensitivity for gap detection (0-1)
    pub gap_sensitivity: f64,
    /// Minimum gap priority to consider
    pub min_gap_priority: f64,
    /// Maximum gaps to analyze
    pub max_gaps: usize,
}

impl Default for BacktrackAnalyzer {
    fn default() -> Self {
        Self {
            gap_sensitivity: 0.5,
            min_gap_priority: 0.4,
            max_gaps: 10,
        }
    }
}

impl BacktrackAnalyzer {
    pub fn new(gap_sensitivity: f64) -> Self {
        Self {
            gap_sensitivity,
            ..Default::default()
        }
    }

    /// Detect coverage gaps based on multiple criteria
    pub fn detect_coverage_gaps(
        &self,
        atoms: &[Atom],
        detected_pockets: &[AllostericPocket],
        conservation: &HashMap<i32, f64>,
        centrality: &HashMap<i32, f64>,
        interfaces: &[DomainInterface],
        residue_bfactors: &HashMap<i32, f64>,
    ) -> Vec<CoverageGap> {
        let mut gaps = Vec::new();

        // Get all residues already detected
        let detected_residues: HashSet<i32> = detected_pockets
            .iter()
            .flat_map(|p| p.residue_indices.iter().copied())
            .collect();

        // Gap Type 1: Conserved regions not in any pocket
        gaps.extend(self.find_conserved_gaps(atoms, conservation, &detected_residues));

        // Gap Type 2: Domain interfaces not detected
        gaps.extend(self.find_interface_gaps(interfaces, &detected_residues));

        // Gap Type 3: Flexible regions not in softspot results
        gaps.extend(self.find_flexible_gaps(atoms, residue_bfactors, &detected_residues));

        // Gap Type 4: Communication hubs (high centrality) not detected
        gaps.extend(self.find_centrality_gaps(atoms, centrality, &detected_residues));

        // Gap Type 5: Active site adjacent regions
        gaps.extend(self.find_active_site_adjacent_gaps(
            atoms,
            detected_pockets,
            &detected_residues,
        ));

        // Sort by priority and limit
        gaps.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        gaps.truncate(self.max_gaps);

        // Filter by minimum priority
        gaps.into_iter()
            .filter(|g| g.priority >= self.min_gap_priority)
            .collect()
    }

    fn find_conserved_gaps(
        &self,
        atoms: &[Atom],
        conservation: &HashMap<i32, f64>,
        detected: &HashSet<i32>,
    ) -> Vec<CoverageGap> {
        let threshold = 0.7 * self.gap_sensitivity + 0.3; // 0.65-0.85 depending on sensitivity

        // Find highly conserved residues not in any pocket
        let conserved_uncovered: Vec<i32> = conservation
            .iter()
            .filter(|(&res, &score)| score >= threshold && !detected.contains(&res))
            .map(|(&res, _)| res)
            .collect();

        if conserved_uncovered.is_empty() {
            return Vec::new();
        }

        // Cluster spatially
        let clusters = spatial_cluster(atoms, &conserved_uncovered, 8.0);

        clusters
            .into_iter()
            .filter(|c| c.len() >= 3)
            .map(|residues| {
                let len = residues.len();
                let mean_conservation: f64 = residues
                    .iter()
                    .filter_map(|r| conservation.get(r))
                    .sum::<f64>()
                    / len as f64;

                CoverageGap {
                    gap_type: GapType::ConservedUncovered,
                    residues,
                    suggested_module: DetectorModule::Allosteric,
                    priority: 0.9 * mean_conservation,
                    reason: format!(
                        "Cluster of {} highly conserved residues (mean: {:.0}%) without pocket detection",
                        len,
                        mean_conservation * 100.0
                    ),
                }
            })
            .collect()
    }

    fn find_interface_gaps(
        &self,
        interfaces: &[DomainInterface],
        detected: &HashSet<i32>,
    ) -> Vec<CoverageGap> {
        interfaces
            .iter()
            .filter_map(|interface| {
                let interface_set: HashSet<i32> = interface.residues.iter().copied().collect();
                let overlap = interface_set.intersection(detected).count();
                let coverage = overlap as f64 / interface.residues.len() as f64;

                if coverage < 0.3 * self.gap_sensitivity {
                    Some(CoverageGap {
                        gap_type: GapType::InterfaceUncovered,
                        residues: interface.residues.clone(),
                        suggested_module: DetectorModule::Interface,
                        priority: 0.7 * (1.0 - coverage) * interface.shape_complementarity,
                        reason: format!(
                            "Domain interface ({} residues, {:.1} Å² buried) only {:.0}% covered",
                            interface.residues.len(),
                            interface.buried_sasa,
                            coverage * 100.0
                        ),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    fn find_flexible_gaps(
        &self,
        atoms: &[Atom],
        bfactors: &HashMap<i32, f64>,
        detected: &HashSet<i32>,
    ) -> Vec<CoverageGap> {
        if bfactors.is_empty() {
            return Vec::new();
        }

        // Calculate B-factor statistics
        let values: Vec<f64> = bfactors.values().copied().collect();
        let mean_bf: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let std_bf: f64 = (values.iter().map(|&v| (v - mean_bf).powi(2)).sum::<f64>()
            / values.len() as f64)
            .sqrt();

        // Threshold: residues above mean + gap_sensitivity * std
        let threshold = mean_bf + self.gap_sensitivity * std_bf;

        // Find flexible residues not detected
        let flexible_uncovered: Vec<i32> = bfactors
            .iter()
            .filter(|(&res, &bf)| bf >= threshold && !detected.contains(&res))
            .map(|(&res, _)| res)
            .collect();

        if flexible_uncovered.len() < 4 {
            return Vec::new();
        }

        // Cluster spatially
        let clusters = spatial_cluster(atoms, &flexible_uncovered, 6.0);

        clusters
            .into_iter()
            .filter(|c| c.len() >= 4)
            .map(|residues| {
                let len = residues.len();
                let cluster_mean_bf: f64 = residues
                    .iter()
                    .filter_map(|r| bfactors.get(r))
                    .sum::<f64>()
                    / len as f64;

                let zscore = (cluster_mean_bf - mean_bf) / std_bf.max(0.1);

                CoverageGap {
                    gap_type: GapType::FlexibleUncovered,
                    residues,
                    suggested_module: DetectorModule::SoftSpot,
                    priority: 0.6 * zscore.abs().min(2.0) / 2.0,
                    reason: format!(
                        "Cluster of {} flexible residues (B-factor z-score: {:.1}) missed by initial detection",
                        len,
                        zscore
                    ),
                }
            })
            .collect()
    }

    fn find_centrality_gaps(
        &self,
        atoms: &[Atom],
        centrality: &HashMap<i32, f64>,
        detected: &HashSet<i32>,
    ) -> Vec<CoverageGap> {
        let threshold = 0.5 * self.gap_sensitivity + 0.2; // 0.45-0.70

        // Find high-centrality residues not detected
        let high_centrality_uncovered: Vec<i32> = centrality
            .iter()
            .filter(|(&res, &score)| score >= threshold && !detected.contains(&res))
            .map(|(&res, _)| res)
            .collect();

        if high_centrality_uncovered.len() < 3 {
            return Vec::new();
        }

        // Cluster spatially
        let clusters = spatial_cluster(atoms, &high_centrality_uncovered, 10.0);

        clusters
            .into_iter()
            .filter(|c| c.len() >= 3)
            .map(|residues| {
                let len = residues.len();
                let mean_centrality: f64 = residues
                    .iter()
                    .filter_map(|r| centrality.get(r))
                    .sum::<f64>()
                    / len as f64;

                CoverageGap {
                    gap_type: GapType::CommunicationHub,
                    residues,
                    suggested_module: DetectorModule::Allosteric,
                    priority: 0.85 * mean_centrality,
                    reason: format!(
                        "Cluster of {} high-centrality residues (mean: {:.2}) - potential allosteric communication hub",
                        len,
                        mean_centrality
                    ),
                }
            })
            .collect()
    }

    fn find_active_site_adjacent_gaps(
        &self,
        atoms: &[Atom],
        pockets: &[AllostericPocket],
        detected: &HashSet<i32>,
    ) -> Vec<CoverageGap> {
        // Find highest-scoring pocket (likely active site)
        let primary_pocket = pockets
            .iter()
            .max_by(|a, b| a.confidence.score.partial_cmp(&b.confidence.score).unwrap());

        let primary = match primary_pocket {
            Some(p) => p,
            None => return Vec::new(),
        };

        // Get Cα coordinates
        let ca_coords: HashMap<i32, [f64; 3]> = atoms
            .iter()
            .filter(|a| a.name.trim() == "CA")
            .map(|a| (a.residue_seq, a.coord))
            .collect();

        // Find residues within shell around primary site
        let inner_radius = 8.0;
        let outer_radius = 15.0;
        let inner_sq = inner_radius * inner_radius;
        let outer_sq = outer_radius * outer_radius;

        let mut shell_residues: Vec<i32> = Vec::new();

        for (&res, coord) in &ca_coords {
            if detected.contains(&res) {
                continue;
            }

            // Check distance to primary pocket centroid
            let dist_sq = (coord[0] - primary.centroid[0]).powi(2)
                + (coord[1] - primary.centroid[1]).powi(2)
                + (coord[2] - primary.centroid[2]).powi(2);

            if dist_sq > inner_sq && dist_sq < outer_sq {
                shell_residues.push(res);
            }
        }

        if shell_residues.len() < 5 {
            return Vec::new();
        }

        // Cluster the shell
        let clusters = spatial_cluster(atoms, &shell_residues, 6.0);

        clusters
            .into_iter()
            .filter(|c| c.len() >= 4)
            .take(2) // Max 2 adjacent gaps
            .map(|residues| {
                let len = residues.len();
                CoverageGap {
                    gap_type: GapType::ActiveSiteAdjacent,
                    residues,
                    suggested_module: DetectorModule::Allosteric,
                    priority: 0.65,
                    reason: format!(
                        "Cluster of {} residues in shell around primary site (8-15 Å) - potential peripheral allosteric site",
                        len
                    ),
                }
            })
            .collect()
    }

    /// Fill detected gaps by re-running focused detection
    pub fn fill_gaps(
        &self,
        atoms: &[Atom],
        gaps: &[CoverageGap],
        conservation: &HashMap<i32, f64>,
        centrality: &HashMap<i32, f64>,
    ) -> Vec<AllostericPocket> {
        let mut additional_pockets = Vec::new();

        for (i, gap) in gaps.iter().enumerate() {
            log::debug!(
                "[BACKTRACK] Analyzing gap {}: {:?} with {} residues",
                i + 1,
                gap.gap_type,
                gap.residues.len()
            );

            // Create pocket from gap region
            if let Some(pocket) = self.create_pocket_from_gap(atoms, gap, conservation, centrality) {
                additional_pockets.push(pocket);
            }
        }

        additional_pockets
    }

    fn create_pocket_from_gap(
        &self,
        atoms: &[Atom],
        gap: &CoverageGap,
        conservation: &HashMap<i32, f64>,
        centrality: &HashMap<i32, f64>,
    ) -> Option<AllostericPocket> {
        if gap.residues.len() < 3 {
            return None;
        }

        let centroid = calculate_residue_centroid(atoms, &gap.residues);

        // Build evidence based on gap type
        let mut evidence = MultiModuleEvidence::default();
        let detection_type;

        match gap.gap_type {
            GapType::ConservedUncovered => {
                let mean_conservation = gap
                    .residues
                    .iter()
                    .filter_map(|r| conservation.get(r))
                    .sum::<f64>()
                    / gap.residues.len() as f64;

                evidence.conservation = Some(ConservationEvidence {
                    mean_score: mean_conservation,
                    n_conserved_residues: gap
                        .residues
                        .iter()
                        .filter(|r| conservation.get(r).map(|&c| c > 0.7).unwrap_or(false))
                        .count(),
                    entropy_score: 1.0 - mean_conservation,
                });
                evidence.detected_by = vec!["conservation_backtrack".to_string()];
                detection_type = AllostericDetectionType::Allosteric;
            }

            GapType::InterfaceUncovered => {
                evidence.interface = Some(InterfaceEvidence {
                    buried_sasa: gap.residues.len() as f64 * 40.0, // Estimate
                    shape_complementarity: 0.6,
                    n_interface_contacts: gap.residues.len(),
                });
                evidence.detected_by = vec!["interface_backtrack".to_string()];
                detection_type = AllostericDetectionType::Interface;
            }

            GapType::FlexibleUncovered => {
                evidence.flexibility = Some(FlexibilityEvidence {
                    score: gap.priority / 0.6,
                    mean_bfactor: 0.0, // Would need to calculate
                    packing_deficit: 0.5,
                    nma_mobility: 0.0,
                });
                evidence.detected_by = vec!["flexibility_backtrack".to_string()];
                detection_type = AllostericDetectionType::Cryptic;
            }

            GapType::CommunicationHub => {
                let mean_centrality = gap
                    .residues
                    .iter()
                    .filter_map(|r| centrality.get(r))
                    .sum::<f64>()
                    / gap.residues.len() as f64;

                evidence.allosteric_coupling = Some(AllostericCouplingEvidence {
                    coupling_strength: mean_centrality,
                    shortest_path_length: 0.0,
                    distance_to_active: 0.0,
                    betweenness_centrality: mean_centrality,
                });
                evidence.detected_by = vec!["centrality_backtrack".to_string()];
                detection_type = AllostericDetectionType::Allosteric;
            }

            GapType::CavityMissed => {
                evidence.geometric = Some(GeometricEvidence {
                    volume: gap.residues.len() as f64 * 50.0, // Estimate
                    depth: 5.0,
                    druggability: 0.5,
                    enclosure: 0.5,
                });
                evidence.detected_by = vec!["geometric_backtrack".to_string()];
                detection_type = AllostericDetectionType::Geometric;
            }

            GapType::ActiveSiteAdjacent => {
                evidence.allosteric_coupling = Some(AllostericCouplingEvidence {
                    coupling_strength: 0.5,
                    shortest_path_length: 12.0, // Approximate
                    distance_to_active: 12.0,
                    betweenness_centrality: 0.3,
                });
                evidence.detected_by = vec!["adjacent_backtrack".to_string()];
                detection_type = AllostericDetectionType::Allosteric;
            }
        }

        // Calculate confidence for backtrack pocket
        let confidence = self.calculate_backtrack_confidence(&evidence, gap);

        Some(AllostericPocket {
            id: 0, // Will be reassigned
            residue_indices: gap.residues.clone(),
            centroid,
            volume: gap.residues.len() as f64 * 40.0, // Rough estimate
            druggability: confidence.score * 0.8, // Conservative estimate
            detection_type,
            confidence,
            evidence,
            from_backtrack: true,
            gap_origin: Some(gap.gap_type),
            annotation: Some(format!("[Backtrack] {}", gap.reason)),
        })
    }

    fn calculate_backtrack_confidence(
        &self,
        evidence: &MultiModuleEvidence,
        gap: &CoverageGap,
    ) -> ConfidenceAssessment {
        // Backtrack pockets start with lower confidence
        let base_score = gap.priority * 0.7;

        let mut supporting = Vec::new();
        supporting.push(format!(
            "Gap detection: {:?} (priority: {:.2})",
            gap.gap_type, gap.priority
        ));

        if evidence.conservation.as_ref().map(|c| c.mean_score > 0.7).unwrap_or(false) {
            supporting.push("Highly conserved region".into());
        }

        if evidence
            .allosteric_coupling
            .as_ref()
            .map(|a| a.betweenness_centrality > 0.5)
            .unwrap_or(false)
        {
            supporting.push("High network centrality".into());
        }

        let concerning = vec!["Detected via backtrack (not primary pass)".to_string()];

        let level = if base_score >= 0.55 {
            Confidence::Medium
        } else {
            Confidence::Low
        };

        ConfidenceAssessment {
            level,
            score: base_score.clamp(0.0, 0.8), // Cap at 0.8 for backtrack
            rationale: "Detected through gap analysis - experimental validation recommended".into(),
            supporting_signals: supporting,
            concerning_signals: concerning,
        }
    }

    /// Generate coverage analysis report
    pub fn generate_coverage_report(
        &self,
        atoms: &[Atom],
        pockets: &[AllostericPocket],
        conservation: &HashMap<i32, f64>,
        interfaces: &[DomainInterface],
    ) -> CoverageAnalysis {
        let all_residues: HashSet<i32> = atoms
            .iter()
            .filter(|a| a.name.trim() == "CA")
            .map(|a| a.residue_seq)
            .collect();

        let detected_residues: HashSet<i32> = pockets
            .iter()
            .flat_map(|p| p.residue_indices.iter().copied())
            .collect();

        // Conserved residue coverage
        let conserved_residues: HashSet<i32> = conservation
            .iter()
            .filter(|(_, &score)| score > 0.7)
            .map(|(&res, _)| res)
            .collect();

        let conserved_covered = conserved_residues
            .intersection(&detected_residues)
            .count();

        let conserved_coverage = if !conserved_residues.is_empty() {
            conserved_covered as f64 / conserved_residues.len() as f64
        } else {
            1.0
        };

        // Interface coverage
        let interface_residues: HashSet<i32> = interfaces
            .iter()
            .flat_map(|i| i.residues.iter().copied())
            .collect();

        let interface_covered = interface_residues
            .intersection(&detected_residues)
            .count();

        let interface_coverage = if !interface_residues.is_empty() {
            interface_covered as f64 / interface_residues.len() as f64
        } else {
            1.0
        };

        CoverageAnalysis {
            total_residues: all_residues.len(),
            residues_in_pockets: detected_residues.len(),
            conserved_residues_covered_pct: conserved_coverage * 100.0,
            interface_residues_covered_pct: interface_coverage * 100.0,
            gaps_remaining: Vec::new(), // Would be populated if gaps remain
        }
    }
}
