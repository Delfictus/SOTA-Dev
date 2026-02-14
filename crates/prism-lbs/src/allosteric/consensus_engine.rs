//! Stage 4: Hybrid Consensus Engine
//!
//! Integrates all detection modules and computes multi-signal consensus scores.
//! Uses sophisticated confidence estimation based on:
//! - Number of independent signals supporting detection
//! - Strength of each signal
//! - Consistency across detection methods
//!
//! The key innovation: evidence-based confidence with transparent rationale.

use crate::structure::Atom;
use super::types::*;
use super::domain_decomposition::{calculate_residue_centroid, DomainDecomposer};
use super::msa_conservation::spatial_cluster;
use std::collections::{HashMap, HashSet};

/// Hybrid consensus engine for multi-module pocket detection
pub struct HybridConsensusEngine {
    /// Minimum confidence to include in results
    pub min_confidence: f64,
    /// Spatial overlap threshold for merging pockets (Å)
    pub merge_distance: f64,
    /// Minimum residues for a valid pocket
    pub min_pocket_residues: usize,
}

impl Default for HybridConsensusEngine {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            merge_distance: 8.0,
            min_pocket_residues: 5,
        }
    }
}

impl HybridConsensusEngine {
    pub fn new(min_confidence: f64) -> Self {
        Self {
            min_confidence,
            ..Default::default()
        }
    }

    /// Build unified pocket list from multiple detection sources
    pub fn build_consensus(
        &self,
        atoms: &[Atom],
        geometric_pockets: &[crate::pocket::Pocket],
        cryptic_candidates: &[crate::softspot::CrypticCandidate],
        allosteric_candidates: &[AllostericCandidateRegion],
        interface_pockets: &[DomainInterface],
        conservation: &HashMap<i32, f64>,
        centrality: &HashMap<i32, f64>,
    ) -> Vec<AllostericPocket> {
        // Build residue index → PDB RESSEQ mapping from atoms
        // The mapping preserves the order of first occurrence (like ProteinGraph.residues)
        let residue_seq_map = build_residue_seq_map(atoms);

        // Step 1: Convert all sources to unified candidates
        let mut candidates: Vec<UnifiedCandidate> = Vec::new();

        // Geometric pockets - convert internal indices to PDB RESSEQ
        for pocket in geometric_pockets {
            let residues: Vec<i32> = pocket
                .residue_indices
                .iter()
                .filter_map(|&idx| residue_seq_map.get(idx).copied())
                .collect();

            candidates.push(UnifiedCandidate {
                residues,
                centroid: pocket.centroid,
                source: DetectionSource::Geometric,
                score: pocket.druggability_score.total as f64,
                evidence: self.build_geometric_evidence(pocket),
            });
        }

        // Cryptic/flexibility candidates
        for candidate in cryptic_candidates {
            candidates.push(UnifiedCandidate {
                residues: candidate.residue_indices.clone(),
                centroid: candidate.centroid,
                source: DetectionSource::Cryptic,
                score: candidate.cryptic_score,
                evidence: self.build_cryptic_evidence(candidate),
            });
        }

        // Allosteric candidates
        for candidate in allosteric_candidates {
            candidates.push(UnifiedCandidate {
                residues: candidate.residues.clone(),
                centroid: candidate.centroid,
                source: DetectionSource::Allosteric,
                score: candidate.coupling_score,
                evidence: self.build_allosteric_evidence(candidate),
            });
        }

        // Interface pockets
        for interface in interface_pockets {
            candidates.push(UnifiedCandidate {
                residues: interface.residues.clone(),
                centroid: interface.centroid,
                source: DetectionSource::Interface,
                score: interface.shape_complementarity,
                evidence: self.build_interface_evidence(interface),
            });
        }

        // Step 2: Cluster spatially overlapping candidates
        let merged = self.merge_overlapping_candidates(atoms, candidates);

        // Step 3: Enrich with conservation and centrality
        let enriched = self.enrich_with_evolutionary_data(merged, conservation, centrality);

        // Step 4: Calculate confidence and filter
        let mut pockets: Vec<AllostericPocket> = enriched
            .into_iter()
            .enumerate()
            .filter_map(|(id, candidate)| {
                let confidence = self.calculate_confidence(&candidate);

                if confidence.score >= self.min_confidence
                    && candidate.residues.len() >= self.min_pocket_residues
                {
                    Some(AllostericPocket {
                        id,
                        residue_indices: candidate.residues.clone(),
                        centroid: candidate.centroid,
                        volume: self.estimate_volume(atoms, &candidate.residues),
                        druggability: candidate.evidence.geometric.as_ref()
                            .map(|g| g.druggability)
                            .unwrap_or(0.5),
                        detection_type: candidate.detection_type,
                        confidence,
                        evidence: candidate.evidence,
                        from_backtrack: false,
                        gap_origin: None,
                        annotation: None,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Step 5: Sort by confidence score
        pockets.sort_by(|a, b| b.confidence.score.partial_cmp(&a.confidence.score).unwrap());

        // Re-assign IDs after sorting
        for (i, pocket) in pockets.iter_mut().enumerate() {
            pocket.id = i + 1;
        }

        pockets
    }

    fn merge_overlapping_candidates(
        &self,
        atoms: &[Atom],
        candidates: Vec<UnifiedCandidate>,
    ) -> Vec<MergedCandidate> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let merge_dist_sq = self.merge_distance * self.merge_distance;
        let mut merged: Vec<MergedCandidate> = Vec::new();
        let mut assigned = vec![false; candidates.len()];

        for (i, candidate) in candidates.iter().enumerate() {
            if assigned[i] {
                continue;
            }

            // Start new cluster
            let mut cluster_residues: HashSet<i32> = candidate.residues.iter().copied().collect();
            let mut sources: Vec<DetectionSource> = vec![candidate.source.clone()];
            let mut evidences: Vec<MultiModuleEvidence> = vec![candidate.evidence.clone()];
            assigned[i] = true;

            // Find overlapping candidates
            for (j, other) in candidates.iter().enumerate().skip(i + 1) {
                if assigned[j] {
                    continue;
                }

                // Check centroid distance
                let dist_sq = (candidate.centroid[0] - other.centroid[0]).powi(2)
                    + (candidate.centroid[1] - other.centroid[1]).powi(2)
                    + (candidate.centroid[2] - other.centroid[2]).powi(2);

                if dist_sq < merge_dist_sq {
                    // Check residue overlap
                    let other_set: HashSet<i32> = other.residues.iter().copied().collect();
                    let overlap = cluster_residues.intersection(&other_set).count();

                    if overlap >= 2 || dist_sq < (self.merge_distance / 2.0).powi(2) {
                        // Merge
                        cluster_residues.extend(other.residues.iter().copied());
                        sources.push(other.source.clone());
                        evidences.push(other.evidence.clone());
                        assigned[j] = true;
                    }
                }
            }

            // Determine detection type based on sources
            let detection_type = self.determine_detection_type(&sources);

            // Merge evidences
            let merged_evidence = self.merge_evidences(evidences);

            // Calculate centroid of merged residues
            let residues: Vec<i32> = cluster_residues.into_iter().collect();
            let centroid = calculate_residue_centroid(atoms, &residues);

            merged.push(MergedCandidate {
                residues,
                centroid,
                sources,
                detection_type,
                evidence: merged_evidence,
            });
        }

        merged
    }

    fn determine_detection_type(&self, sources: &[DetectionSource]) -> AllostericDetectionType {
        if sources.len() > 1 {
            AllostericDetectionType::Consensus
        } else {
            match sources.first() {
                Some(DetectionSource::Geometric) => AllostericDetectionType::Geometric,
                Some(DetectionSource::Cryptic) => AllostericDetectionType::Cryptic,
                Some(DetectionSource::Allosteric) => AllostericDetectionType::Allosteric,
                Some(DetectionSource::Interface) => AllostericDetectionType::Interface,
                None => AllostericDetectionType::Geometric,
            }
        }
    }

    fn merge_evidences(&self, evidences: Vec<MultiModuleEvidence>) -> MultiModuleEvidence {
        let mut merged = MultiModuleEvidence::default();
        let mut detected_by: HashSet<String> = HashSet::new();

        for evidence in evidences {
            // Take best geometric evidence
            if let Some(geo) = evidence.geometric {
                if merged.geometric.is_none()
                    || geo.druggability > merged.geometric.as_ref().unwrap().druggability
                {
                    merged.geometric = Some(geo);
                }
            }

            // Take best flexibility evidence
            if let Some(flex) = evidence.flexibility {
                if merged.flexibility.is_none()
                    || flex.score > merged.flexibility.as_ref().unwrap().score
                {
                    merged.flexibility = Some(flex);
                }
            }

            // Take best conservation evidence
            if let Some(cons) = evidence.conservation {
                if merged.conservation.is_none()
                    || cons.mean_score > merged.conservation.as_ref().unwrap().mean_score
                {
                    merged.conservation = Some(cons);
                }
            }

            // Take best allosteric coupling evidence
            if let Some(allo) = evidence.allosteric_coupling {
                if merged.allosteric_coupling.is_none()
                    || allo.coupling_strength
                        > merged.allosteric_coupling.as_ref().unwrap().coupling_strength
                {
                    merged.allosteric_coupling = Some(allo);
                }
            }

            // Take interface evidence
            if let Some(iface) = evidence.interface {
                if merged.interface.is_none() {
                    merged.interface = Some(iface);
                }
            }

            detected_by.extend(evidence.detected_by);
        }

        merged.detected_by = detected_by.into_iter().collect();
        merged.detected_by.sort();

        merged
    }

    fn enrich_with_evolutionary_data(
        &self,
        candidates: Vec<MergedCandidate>,
        conservation: &HashMap<i32, f64>,
        centrality: &HashMap<i32, f64>,
    ) -> Vec<MergedCandidate> {
        candidates
            .into_iter()
            .map(|mut candidate| {
                // Add conservation evidence if not present
                if candidate.evidence.conservation.is_none() {
                    let residue_conservation: Vec<f64> = candidate
                        .residues
                        .iter()
                        .filter_map(|r| conservation.get(r).copied())
                        .collect();

                    if !residue_conservation.is_empty() {
                        let mean_score =
                            residue_conservation.iter().sum::<f64>() / residue_conservation.len() as f64;
                        let n_conserved = residue_conservation
                            .iter()
                            .filter(|&&c| c > 0.7)
                            .count();

                        candidate.evidence.conservation = Some(ConservationEvidence {
                            mean_score,
                            n_conserved_residues: n_conserved,
                            entropy_score: 1.0 - mean_score, // Inverse
                        });

                        if !candidate.evidence.detected_by.contains(&"conservation".to_string()) {
                            candidate.evidence.detected_by.push("conservation".to_string());
                        }
                    }
                }

                // Add centrality to allosteric coupling evidence
                if candidate.evidence.allosteric_coupling.is_none() {
                    let residue_centrality: Vec<f64> = candidate
                        .residues
                        .iter()
                        .filter_map(|r| centrality.get(r).copied())
                        .collect();

                    if !residue_centrality.is_empty() {
                        let mean_centrality =
                            residue_centrality.iter().sum::<f64>() / residue_centrality.len() as f64;

                        if mean_centrality > 0.3 {
                            candidate.evidence.allosteric_coupling = Some(AllostericCouplingEvidence {
                                coupling_strength: mean_centrality,
                                shortest_path_length: 0.0,
                                distance_to_active: 0.0,
                                betweenness_centrality: mean_centrality,
                            });

                            if !candidate.evidence.detected_by.contains(&"centrality".to_string()) {
                                candidate.evidence.detected_by.push("centrality".to_string());
                            }
                        }
                    }
                }

                candidate
            })
            .collect()
    }

    /// Calculate confidence assessment with detailed rationale
    pub fn calculate_confidence(&self, candidate: &MergedCandidate) -> ConfidenceAssessment {
        let mut score = 0.0;
        let mut supporting = Vec::new();
        let mut concerning = Vec::new();

        // Geometric evidence (strong signal)
        if let Some(geo) = &candidate.evidence.geometric {
            score += 0.25;
            supporting.push(format!("Geometric cavity: {:.0} Å³", geo.volume));

            if geo.druggability > 0.7 {
                score += 0.1;
                supporting.push(format!("High druggability: {:.2}", geo.druggability));
            } else if geo.druggability < 0.4 {
                score -= 0.05;
                concerning.push(format!("Low druggability: {:.2}", geo.druggability));
            }

            if geo.enclosure > 0.6 {
                score += 0.05;
            }
        }

        // Flexibility evidence (medium signal)
        if let Some(flex) = &candidate.evidence.flexibility {
            score += 0.15;
            supporting.push(format!("Flexibility signal: {:.2}", flex.score));

            if flex.packing_deficit > 0.5 {
                score += 0.05;
                supporting.push("High packing deficit (cryptic potential)".into());
            }
        }

        // Conservation evidence (strong signal)
        if let Some(cons) = &candidate.evidence.conservation {
            if cons.mean_score > 0.7 {
                score += 0.25;
                supporting.push(format!(
                    "Highly conserved: {:.0}% ({} residues)",
                    cons.mean_score * 100.0,
                    cons.n_conserved_residues
                ));
            } else if cons.mean_score > 0.5 {
                score += 0.1;
                supporting.push(format!("Moderately conserved: {:.0}%", cons.mean_score * 100.0));
            } else {
                concerning.push(format!("Low conservation: {:.0}%", cons.mean_score * 100.0));
            }
        } else {
            concerning.push("No conservation data available".into());
        }

        // Allosteric coupling evidence (strong signal for allosteric sites)
        if let Some(allo) = &candidate.evidence.allosteric_coupling {
            if allo.coupling_strength > 0.5 {
                score += 0.2;
                supporting.push(format!(
                    "Allosteric coupling: {:.2} (path: {:.1})",
                    allo.coupling_strength, allo.shortest_path_length
                ));
            } else if allo.coupling_strength > 0.3 {
                score += 0.1;
            }

            if allo.betweenness_centrality > 0.5 {
                score += 0.1;
                supporting.push("High network centrality (communication hub)".into());
            }
        }

        // Interface evidence
        if let Some(iface) = &candidate.evidence.interface {
            score += 0.1;
            supporting.push(format!(
                "Domain interface: {:.0} Å² buried",
                iface.buried_sasa
            ));
        }

        // Multi-module consensus bonus
        let n_modules = candidate.sources.len();
        if n_modules >= 3 {
            score += 0.2;
            supporting.push(format!("Multi-module consensus ({} sources)", n_modules));
        } else if n_modules == 2 {
            score += 0.1;
            supporting.push("Two detection methods agree".into());
        } else if n_modules == 1 {
            score -= 0.05;
            concerning.push("Single detection method only".into());
        }

        // Determine level
        let level = if score >= 0.65 {
            Confidence::High
        } else if score >= 0.35 {
            Confidence::Medium
        } else {
            Confidence::Low
        };

        let rationale = match level {
            Confidence::High => "Strong evidence from multiple independent signals",
            Confidence::Medium => "Moderate evidence, may require experimental validation",
            Confidence::Low => "Weak evidence, treat as speculative",
        }
        .to_string();

        ConfidenceAssessment {
            level,
            score: (score as f64).clamp(0.0, 1.0),
            rationale,
            supporting_signals: supporting,
            concerning_signals: concerning,
        }
    }

    fn build_geometric_evidence(&self, pocket: &crate::pocket::Pocket) -> MultiModuleEvidence {
        MultiModuleEvidence {
            geometric: Some(GeometricEvidence {
                volume: pocket.volume,
                depth: pocket.mean_depth,
                druggability: pocket.druggability_score.total as f64,
                enclosure: pocket.enclosure_ratio,
            }),
            detected_by: vec!["geometric".to_string()],
            ..Default::default()
        }
    }

    fn build_cryptic_evidence(
        &self,
        candidate: &crate::softspot::CrypticCandidate,
    ) -> MultiModuleEvidence {
        MultiModuleEvidence {
            flexibility: Some(FlexibilityEvidence {
                score: candidate.cryptic_score,
                mean_bfactor: candidate.flexibility_score,
                packing_deficit: candidate.packing_deficit,
                nma_mobility: 0.0,
            }),
            detected_by: vec!["softspot".to_string()],
            ..Default::default()
        }
    }

    fn build_allosteric_evidence(
        &self,
        candidate: &AllostericCandidateRegion,
    ) -> MultiModuleEvidence {
        MultiModuleEvidence {
            allosteric_coupling: Some(AllostericCouplingEvidence {
                coupling_strength: candidate.coupling_score,
                shortest_path_length: candidate.path_length,
                distance_to_active: candidate.distance_to_active,
                betweenness_centrality: candidate.centrality,
            }),
            detected_by: vec!["allosteric".to_string()],
            ..Default::default()
        }
    }

    fn build_interface_evidence(&self, interface: &DomainInterface) -> MultiModuleEvidence {
        MultiModuleEvidence {
            interface: Some(InterfaceEvidence {
                buried_sasa: interface.buried_sasa,
                shape_complementarity: interface.shape_complementarity,
                n_interface_contacts: interface.residues.len(),
            }),
            detected_by: vec!["interface".to_string()],
            ..Default::default()
        }
    }

    fn estimate_volume(&self, atoms: &[Atom], residues: &[i32]) -> f64 {
        // Rough volume estimate based on residue count and typical volumes
        let avg_residue_volume = 135.0; // Å³ average amino acid volume
        residues.len() as f64 * avg_residue_volume * 0.3 // Pocket is portion of residue volume
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

#[derive(Debug, Clone)]
enum DetectionSource {
    Geometric,
    Cryptic,
    Allosteric,
    Interface,
}

#[derive(Debug)]
struct UnifiedCandidate {
    residues: Vec<i32>,
    centroid: [f64; 3],
    source: DetectionSource,
    score: f64,
    evidence: MultiModuleEvidence,
}

#[derive(Debug)]
struct MergedCandidate {
    residues: Vec<i32>,
    centroid: [f64; 3],
    sources: Vec<DetectionSource>,
    detection_type: AllostericDetectionType,
    evidence: MultiModuleEvidence,
}

/// Candidate allosteric region from coupling analysis
#[derive(Debug, Clone)]
pub struct AllostericCandidateRegion {
    pub residues: Vec<i32>,
    pub centroid: [f64; 3],
    pub coupling_score: f64,
    pub path_length: f64,
    pub distance_to_active: f64,
    pub centrality: f64,
}

/// Build a mapping from internal residue index to PDB RESSEQ
///
/// This replicates how ProteinGraph builds its residue list:
/// atoms are processed in order, and residues are assigned sequential
/// indices (0, 1, 2...) in order of first occurrence.
fn build_residue_seq_map(atoms: &[Atom]) -> Vec<i32> {
    let mut seen: HashSet<(char, i32)> = HashSet::new();
    let mut seq_map: Vec<i32> = Vec::new();

    for atom in atoms {
        let key = (atom.chain_id, atom.residue_seq);
        if !seen.contains(&key) {
            seen.insert(key);
            seq_map.push(atom.residue_seq);
        }
    }

    seq_map
}
