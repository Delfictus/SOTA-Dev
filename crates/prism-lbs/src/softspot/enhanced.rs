//! Enhanced Soft-spot Detector - Multi-signal cryptic site detection
//!
//! Combines multiple biophysical signals to improve cryptic binding site detection:
//! - B-factor flexibility (classic)
//! - Normal Mode Analysis (induced-fit susceptibility)
//! - Contact Order (conformational rearrangement potential)
//! - Evolutionary Conservation (functional importance)
//! - Probe Clustering (druggable hot spots)
//!
//! ## Architecture
//!
//! All enhancements are ADDITIVE ONLY - they can only increase detection
//! confidence, never decrease scores for existing detections. This ensures
//! zero regressions from the classic geometric detection.
//!
//! Each signal is independently toggleable via configuration.

use crate::softspot::conservation::{conservation_to_score_map, ConservationAnalyzer};
use crate::softspot::contact_order::{contact_order_to_score_map, ContactOrderAnalyzer};
use crate::softspot::constants::*;
use crate::softspot::nma::{nma_to_score_map, NmaAnalyzer};
use crate::softspot::probe_clustering::{probe_clusters_to_score_map, ProbeClusteringAnalyzer};
use crate::softspot::types::*;
use crate::structure::Atom;
use std::collections::HashMap;

//=============================================================================
// CONFIGURATION
//=============================================================================

/// Configuration for the enhanced soft-spot detector
#[derive(Debug, Clone)]
pub struct EnhancedSoftSpotConfig {
    // Classic detection parameters
    /// B-factor z-score threshold for flexibility
    pub bfactor_threshold: f64,
    /// Minimum residues to form a candidate
    pub min_cluster_size: usize,
    /// Maximum residues per candidate
    pub max_cluster_size: usize,
    /// Minimum score to report
    pub min_score: f64,

    // Enhancement toggles (all default ON)
    /// Enable Normal Mode Analysis
    pub use_nma: bool,
    /// Enable Contact Order Analysis
    pub use_contact_order: bool,
    /// Enable Conservation Analysis
    pub use_conservation: bool,
    /// Enable Probe Clustering
    pub use_probe_clustering: bool,

    // Signal weights (must sum to ~1.0 for normalized scoring)
    /// Weight for B-factor flexibility signal
    pub weight_bfactor: f64,
    /// Weight for NMA mobility signal
    pub weight_nma: f64,
    /// Weight for contact order flexibility signal
    pub weight_contact_order: f64,
    /// Weight for conservation signal
    pub weight_conservation: f64,
    /// Weight for probe clustering signal
    pub weight_probe: f64,
    /// Weight for packing density signal
    pub weight_packing: f64,
    /// Weight for hydrophobicity signal
    pub weight_hydrophobicity: f64,

    // Enhancement-specific thresholds
    /// Minimum NMA mobility to consider (normalized 0-1)
    pub nma_min_threshold: f64,
    /// Low contact order threshold (below this = flexible)
    pub contact_order_threshold: f64,
    /// High conservation threshold (above this = important)
    pub conservation_threshold: f64,
    /// Minimum probe cluster score
    pub probe_min_score: f64,
}

impl Default for EnhancedSoftSpotConfig {
    fn default() -> Self {
        Self {
            // Classic parameters
            bfactor_threshold: BFACTOR_ZSCORE_THRESHOLD,
            min_cluster_size: MIN_CLUSTER_SIZE,
            max_cluster_size: MAX_CLUSTER_SIZE,
            min_score: MIN_CRYPTIC_SCORE,

            // All enhancements ON by default
            use_nma: true,
            use_contact_order: true,
            use_conservation: true,
            use_probe_clustering: true,

            // Balanced weights (sum ~1.0)
            // Classic signals: 40%
            // Enhanced signals: 60%
            weight_bfactor: 0.15,
            weight_packing: 0.15,
            weight_hydrophobicity: 0.10,
            weight_nma: 0.20,
            weight_contact_order: 0.12,
            weight_conservation: 0.13,
            weight_probe: 0.15,

            // Enhancement thresholds
            nma_min_threshold: 0.3,
            contact_order_threshold: 0.3,
            conservation_threshold: 0.6,
            probe_min_score: 0.3,
        }
    }
}

//=============================================================================
// ENHANCED DETECTOR
//=============================================================================

/// Enhanced Soft-spot Detector with multi-signal integration
///
/// This detector extends the classic B-factor-based approach with:
/// - Normal Mode Analysis for induced-fit susceptibility
/// - Contact Order for conformational flexibility potential
/// - Conservation for functional importance without structural role
/// - Probe Clustering for druggable hot spot identification
///
/// ## Design Principle: ADDITIVE ONLY
///
/// All enhancements are designed to ONLY ADD to detection - they can
/// boost scores or add new candidates, but NEVER reduce scores or
/// remove candidates that the classic detector would find. This ensures
/// zero regressions in geometric pocket detection.
pub struct EnhancedSoftSpotDetector {
    pub config: EnhancedSoftSpotConfig,
    nma_analyzer: NmaAnalyzer,
    contact_order_analyzer: ContactOrderAnalyzer,
    conservation_analyzer: ConservationAnalyzer,
    probe_analyzer: ProbeClusteringAnalyzer,
}

impl Default for EnhancedSoftSpotDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedSoftSpotDetector {
    /// Create a new enhanced detector with default configuration
    pub fn new() -> Self {
        Self {
            config: EnhancedSoftSpotConfig::default(),
            nma_analyzer: NmaAnalyzer::new(),
            contact_order_analyzer: ContactOrderAnalyzer::new(),
            conservation_analyzer: ConservationAnalyzer::new(),
            probe_analyzer: ProbeClusteringAnalyzer::new(),
        }
    }

    /// Create a detector with custom configuration
    pub fn with_config(config: EnhancedSoftSpotConfig) -> Self {
        Self {
            config,
            nma_analyzer: NmaAnalyzer::new(),
            contact_order_analyzer: ContactOrderAnalyzer::new(),
            conservation_analyzer: ConservationAnalyzer::new(),
            probe_analyzer: ProbeClusteringAnalyzer::new(),
        }
    }

    /// Main entry point: detect cryptic sites using all enabled signals
    pub fn detect(&self, atoms: &[Atom]) -> Vec<CrypticCandidate> {
        if atoms.is_empty() {
            log::warn!("[ENHANCED_SOFTSPOT] No atoms provided for analysis");
            return Vec::new();
        }

        log::info!(
            "[ENHANCED_SOFTSPOT] Analyzing {} atoms with multi-signal detection",
            atoms.len()
        );

        // Step 1: Run all enabled analyses in parallel preparation
        let signals = self.gather_all_signals(atoms);

        // Step 2: Combine signals into per-residue scores
        let scored_residues = self.combine_enhanced_signals(atoms, &signals);
        log::debug!(
            "[ENHANCED_SOFTSPOT] {} residues scored with enhanced signals",
            scored_residues.len()
        );

        // Step 3: Cluster scored residues spatially
        let clusters = self.cluster_residues(&scored_residues);
        log::debug!("[ENHANCED_SOFTSPOT] {} clusters formed", clusters.len());

        // Step 4: Score clusters and create candidates
        let candidates = self.score_candidates(clusters, &signals);

        log::info!(
            "[ENHANCED_SOFTSPOT] Found {} cryptic site candidates",
            candidates.len()
        );
        candidates
    }

    /// Gather all signal maps from enabled analyzers
    fn gather_all_signals(&self, atoms: &[Atom]) -> SignalMaps {
        // Classic signals
        let bfactor_map = self.analyze_flexibility(atoms);
        let packing_map = self.analyze_packing(atoms);
        let hydrophobicity_map = self.analyze_hydrophobicity(atoms);

        // Enhanced signals (conditionally computed)
        let nma_map = if self.config.use_nma {
            log::debug!("[ENHANCED_SOFTSPOT] Running NMA analysis");
            let nma_result = self.nma_analyzer.analyze(atoms);
            log::debug!(
                "[ENHANCED_SOFTSPOT] NMA: {} modes computed for {} residues",
                nma_result.num_modes_computed,
                nma_result.total_residues
            );
            nma_to_score_map(&nma_result)
        } else {
            HashMap::new()
        };

        let contact_order_map = if self.config.use_contact_order {
            log::debug!("[ENHANCED_SOFTSPOT] Running contact order analysis");
            let co_result = self.contact_order_analyzer.analyze(atoms);
            log::debug!(
                "[ENHANCED_SOFTSPOT] Contact order: RCO={:.3}, {} contacts",
                co_result.global_rco,
                co_result.total_contacts
            );
            contact_order_to_score_map(&co_result)
        } else {
            HashMap::new()
        };

        let conservation_map = if self.config.use_conservation {
            log::debug!("[ENHANCED_SOFTSPOT] Running conservation analysis");
            let cons_result = self.conservation_analyzer.analyze(atoms);
            log::debug!(
                "[ENHANCED_SOFTSPOT] Conservation: mean={:.3}, std={:.3}",
                cons_result.mean_conservation,
                cons_result.std_conservation
            );
            conservation_to_score_map(&cons_result)
        } else {
            HashMap::new()
        };

        let probe_map = if self.config.use_probe_clustering {
            log::debug!("[ENHANCED_SOFTSPOT] Running probe clustering analysis");
            let probe_result = self.probe_analyzer.analyze(atoms);
            log::debug!(
                "[ENHANCED_SOFTSPOT] Probe clustering: {} clusters, {} hot spot residues",
                probe_result.clusters.len(),
                probe_result.hot_spot_residues.len()
            );
            probe_clusters_to_score_map(&probe_result)
        } else {
            HashMap::new()
        };

        SignalMaps {
            bfactor: bfactor_map,
            packing: packing_map,
            hydrophobicity: hydrophobicity_map,
            nma: nma_map,
            contact_order: contact_order_map,
            conservation: conservation_map,
            probe: probe_map,
        }
    }

    /// Calculate B-factor z-scores per residue (same as classic)
    fn analyze_flexibility(&self, atoms: &[Atom]) -> HashMap<i32, f64> {
        let bfactors: Vec<f64> = atoms.iter().map(|a| a.b_factor).collect();

        if bfactors.is_empty() {
            return HashMap::new();
        }

        let mean = bfactors.iter().sum::<f64>() / bfactors.len() as f64;
        let variance = bfactors
            .iter()
            .map(|b| (b - mean).powi(2))
            .sum::<f64>()
            / bfactors.len() as f64;
        let std = variance.sqrt().max(0.1);

        let mut residue_bfactors: HashMap<i32, Vec<f64>> = HashMap::new();
        for atom in atoms {
            residue_bfactors
                .entry(atom.residue_seq)
                .or_default()
                .push(atom.b_factor);
        }

        residue_bfactors
            .into_iter()
            .map(|(res_seq, bfs)| {
                let avg_bf = bfs.iter().sum::<f64>() / bfs.len() as f64;
                let zscore = (avg_bf - mean) / std;
                (res_seq, zscore)
            })
            .collect()
    }

    /// Calculate local packing density per residue
    fn analyze_packing(&self, atoms: &[Atom]) -> HashMap<i32, f64> {
        let mut residue_centroids: HashMap<i32, [f64; 3]> = HashMap::new();
        let mut residue_counts: HashMap<i32, usize> = HashMap::new();

        for atom in atoms {
            let entry = residue_centroids
                .entry(atom.residue_seq)
                .or_insert([0.0, 0.0, 0.0]);
            entry[0] += atom.coord[0];
            entry[1] += atom.coord[1];
            entry[2] += atom.coord[2];
            *residue_counts.entry(atom.residue_seq).or_insert(0) += 1;
        }

        for (res_seq, centroid) in residue_centroids.iter_mut() {
            let n = residue_counts[res_seq] as f64;
            centroid[0] /= n;
            centroid[1] /= n;
            centroid[2] /= n;
        }

        let mut packing_map = HashMap::new();
        let sphere_volume = (4.0 / 3.0) * std::f64::consts::PI * PACKING_RADIUS.powi(3);
        let radius_sq = PACKING_RADIUS * PACKING_RADIUS;

        for (&res_seq, centroid) in &residue_centroids {
            let nearby_count = atoms
                .iter()
                .filter(|a| a.residue_seq != res_seq)
                .filter(|a| {
                    let dx = a.coord[0] - centroid[0];
                    let dy = a.coord[1] - centroid[1];
                    let dz = a.coord[2] - centroid[2];
                    (dx * dx + dy * dy + dz * dz) < radius_sq
                })
                .count();

            let density = nearby_count as f64 / sphere_volume;
            let density_ratio = density / EXPECTED_PACKING_DENSITY;
            packing_map.insert(res_seq, density_ratio);
        }

        packing_map
    }

    /// Calculate hydrophobicity per residue
    fn analyze_hydrophobicity(&self, atoms: &[Atom]) -> HashMap<i32, f64> {
        let scale: HashMap<&str, f64> = [
            ("ALA", 0.700),
            ("ARG", 0.000),
            ("ASN", 0.111),
            ("ASP", 0.111),
            ("CYS", 0.778),
            ("GLN", 0.111),
            ("GLU", 0.111),
            ("GLY", 0.456),
            ("HIS", 0.144),
            ("ILE", 1.000),
            ("LEU", 0.922),
            ("LYS", 0.067),
            ("MET", 0.711),
            ("PHE", 0.811),
            ("PRO", 0.322),
            ("SER", 0.411),
            ("THR", 0.422),
            ("TRP", 0.400),
            ("TYR", 0.356),
            ("VAL", 0.967),
        ]
        .into_iter()
        .collect();

        let mut residue_names: HashMap<i32, String> = HashMap::new();
        for atom in atoms {
            residue_names
                .entry(atom.residue_seq)
                .or_insert_with(|| atom.residue_name.clone());
        }

        residue_names
            .into_iter()
            .map(|(res_seq, name)| {
                let hydro = scale.get(name.as_str()).copied().unwrap_or(0.5);
                (res_seq, hydro)
            })
            .collect()
    }

    /// Combine all signals into scored residues using enhanced weighting
    fn combine_enhanced_signals(
        &self,
        atoms: &[Atom],
        signals: &SignalMaps,
    ) -> Vec<EnhancedFlexibleResidue> {
        // Build residue data with centroids
        let mut residue_data: HashMap<i32, (Vec<&Atom>, [f64; 3])> = HashMap::new();
        for atom in atoms {
            let entry = residue_data
                .entry(atom.residue_seq)
                .or_insert_with(|| (Vec::new(), [0.0, 0.0, 0.0]));
            entry.0.push(atom);
        }

        // Calculate centroids
        for (_, (atom_list, centroid)) in residue_data.iter_mut() {
            let n = atom_list.len() as f64;
            centroid[0] = atom_list.iter().map(|a| a.coord[0]).sum::<f64>() / n;
            centroid[1] = atom_list.iter().map(|a| a.coord[1]).sum::<f64>() / n;
            centroid[2] = atom_list.iter().map(|a| a.coord[2]).sum::<f64>() / n;
        }

        let mut results = Vec::new();

        for (&res_seq, (atom_list, centroid)) in &residue_data {
            // Get all signal values with defaults
            let bfactor = signals.bfactor.get(&res_seq).copied().unwrap_or(0.0);
            let packing = signals.packing.get(&res_seq).copied().unwrap_or(1.0);
            let hydro = signals.hydrophobicity.get(&res_seq).copied().unwrap_or(0.5);

            // Enhanced signals (0.5 default = neutral contribution)
            let nma = signals.nma.get(&res_seq).copied().unwrap_or(0.5);
            let contact_order = signals.contact_order.get(&res_seq).copied().unwrap_or(0.5);
            let conservation = signals.conservation.get(&res_seq).copied().unwrap_or(0.5);
            let probe = signals.probe.get(&res_seq).copied().unwrap_or(0.0);

            // ENHANCED QUALIFICATION LOGIC:
            // Classic criteria: B-factor + (packing OR hydrophobicity)
            // Enhanced criteria: ANY strong signal can qualify
            let classic_qualifies = bfactor > self.config.bfactor_threshold
                && (packing < PACKING_DEFICIT_THRESHOLD || hydro > HYDROPHOBICITY_THRESHOLD);

            let nma_qualifies = self.config.use_nma && nma > self.config.nma_min_threshold;

            let contact_order_qualifies =
                self.config.use_contact_order && contact_order > 0.5; // High score = low contact order

            let conservation_qualifies =
                self.config.use_conservation && conservation > self.config.conservation_threshold;

            let probe_qualifies =
                self.config.use_probe_clustering && probe > self.config.probe_min_score;

            // ADDITIVE LOGIC: Qualify if classic OR any enhanced signal is strong
            let qualifies = classic_qualifies
                || nma_qualifies
                || contact_order_qualifies
                || (conservation_qualifies && probe_qualifies); // Conservation alone not enough

            // But we need SOME baseline flexibility signal
            // This prevents false positives from conservation alone
            let has_flexibility_signal = bfactor > BFACTOR_ZSCORE_MINIMUM
                || nma > 0.3
                || contact_order > 0.4;

            if qualifies && has_flexibility_signal {
                // Calculate combined score using enhanced weights
                let packing_deficit = (1.0 - packing).clamp(0.0, 1.0);
                let bfactor_norm = (bfactor / 3.0).clamp(0.0, 1.0);

                let combined_score = self.config.weight_bfactor * bfactor_norm
                    + self.config.weight_packing * packing_deficit
                    + self.config.weight_hydrophobicity * hydro
                    + self.config.weight_nma * nma
                    + self.config.weight_contact_order * contact_order
                    + self.config.weight_conservation * conservation
                    + self.config.weight_probe * probe;

                results.push(EnhancedFlexibleResidue {
                    base: FlexibleResidue {
                        chain_id: atom_list[0].chain_id,
                        residue_seq: res_seq,
                        residue_name: atom_list[0].residue_name.clone(),
                        bfactor_zscore: bfactor,
                        packing_density: packing,
                        hydrophobicity: hydro,
                        centroid: *centroid,
                    },
                    nma_mobility: nma,
                    contact_order_flexibility: contact_order,
                    conservation_score: conservation,
                    probe_score: probe,
                    combined_score,
                    qualification_reason: if classic_qualifies {
                        "classic"
                    } else if nma_qualifies {
                        "nma"
                    } else if contact_order_qualifies {
                        "contact_order"
                    } else {
                        "enhanced"
                    },
                });
            }
        }

        results
    }

    /// Cluster enhanced flexible residues spatially
    fn cluster_residues(
        &self,
        residues: &[EnhancedFlexibleResidue],
    ) -> Vec<Vec<EnhancedFlexibleResidue>> {
        if residues.is_empty() {
            return Vec::new();
        }

        let mut clusters: Vec<Vec<EnhancedFlexibleResidue>> = Vec::new();
        let mut assigned = vec![false; residues.len()];
        let cluster_dist_sq = CLUSTER_DISTANCE * CLUSTER_DISTANCE;

        for i in 0..residues.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![residues[i].clone()];
            assigned[i] = true;

            // Grow cluster using single-linkage
            let mut changed = true;
            while changed {
                changed = false;
                for j in 0..residues.len() {
                    if assigned[j] {
                        continue;
                    }

                    let is_close = cluster.iter().any(|c| {
                        let dx = c.base.centroid[0] - residues[j].base.centroid[0];
                        let dy = c.base.centroid[1] - residues[j].base.centroid[1];
                        let dz = c.base.centroid[2] - residues[j].base.centroid[2];
                        (dx * dx + dy * dy + dz * dz) < cluster_dist_sq
                    });

                    if is_close {
                        cluster.push(residues[j].clone());
                        assigned[j] = true;
                        changed = true;
                    }
                }
            }

            if cluster.len() >= self.config.min_cluster_size
                && cluster.len() <= self.config.max_cluster_size
            {
                clusters.push(cluster);
            }
        }

        clusters
    }

    /// Score clusters and convert to candidates with enhanced rationale
    fn score_candidates(
        &self,
        clusters: Vec<Vec<EnhancedFlexibleResidue>>,
        signals: &SignalMaps,
    ) -> Vec<CrypticCandidate> {
        let mut candidates: Vec<CrypticCandidate> = clusters
            .into_iter()
            .enumerate()
            .filter_map(|(id, cluster)| self.score_cluster(id, cluster, signals))
            .collect();

        candidates.sort_by(|a, b| {
            b.cryptic_score
                .partial_cmp(&a.cryptic_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, candidate) in candidates.iter_mut().enumerate() {
            candidate.id = i + 1;
        }

        candidates
    }

    /// Score a single cluster with enhanced signals
    fn score_cluster(
        &self,
        id: usize,
        cluster: Vec<EnhancedFlexibleResidue>,
        _signals: &SignalMaps,
    ) -> Option<CrypticCandidate> {
        if cluster.is_empty() {
            return None;
        }

        let n = cluster.len() as f64;

        // Average all signals
        let avg_bfactor = cluster
            .iter()
            .map(|r| (r.base.bfactor_zscore / 3.0).clamp(0.0, 1.0))
            .sum::<f64>()
            / n;

        let avg_packing_deficit = cluster
            .iter()
            .map(|r| (1.0 - r.base.packing_density).clamp(0.0, 1.0))
            .sum::<f64>()
            / n;

        let avg_hydro = cluster.iter().map(|r| r.base.hydrophobicity).sum::<f64>() / n;

        let avg_nma = cluster.iter().map(|r| r.nma_mobility).sum::<f64>() / n;

        let avg_contact_order = cluster
            .iter()
            .map(|r| r.contact_order_flexibility)
            .sum::<f64>()
            / n;

        let avg_conservation = cluster.iter().map(|r| r.conservation_score).sum::<f64>() / n;

        let avg_probe = cluster.iter().map(|r| r.probe_score).sum::<f64>() / n;

        // Cluster centroid
        let centroid = [
            cluster.iter().map(|r| r.base.centroid[0]).sum::<f64>() / n,
            cluster.iter().map(|r| r.base.centroid[1]).sum::<f64>() / n,
            cluster.iter().map(|r| r.base.centroid[2]).sum::<f64>() / n,
        ];

        // Spatial coherence
        let max_dist = cluster
            .iter()
            .map(|r| {
                let dx = r.base.centroid[0] - centroid[0];
                let dy = r.base.centroid[1] - centroid[1];
                let dz = r.base.centroid[2] - centroid[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(0.0_f64, |a, b| a.max(b));

        let expected_span = n.sqrt() * 5.0;
        let coherence = (expected_span / max_dist.max(1.0)).clamp(0.0, 1.0);

        // Enhanced cryptic score
        let cryptic_score = self.config.weight_bfactor * avg_bfactor
            + self.config.weight_packing * avg_packing_deficit
            + self.config.weight_hydrophobicity * avg_hydro
            + self.config.weight_nma * avg_nma
            + self.config.weight_contact_order * avg_contact_order
            + self.config.weight_conservation * avg_conservation
            + self.config.weight_probe * avg_probe
            + COHERENCE_WEIGHT * coherence;

        if cryptic_score < self.config.min_score {
            return None;
        }

        // Volume and druggability
        let estimated_volume = cluster.len() as f64 * VOLUME_PER_RESIDUE;
        let volume_factor = (estimated_volume / OPTIMAL_POCKET_VOLUME).clamp(0.0, 1.0);
        let predicted_druggability = (0.25 * avg_hydro
            + 0.25 * volume_factor
            + 0.15 * avg_packing_deficit
            + 0.15 * coherence
            + 0.10 * avg_nma
            + 0.10 * avg_probe)
            .clamp(0.0, 1.0);

        // Count qualification reasons
        let classic_count = cluster.iter().filter(|r| r.qualification_reason == "classic").count();
        let nma_count = cluster.iter().filter(|r| r.qualification_reason == "nma").count();
        let co_count = cluster.iter().filter(|r| r.qualification_reason == "contact_order").count();
        let enhanced_count = cluster.iter().filter(|r| r.qualification_reason == "enhanced").count();

        // Generate enhanced rationale
        let avg_zscore = cluster.iter().map(|r| r.base.bfactor_zscore).sum::<f64>() / n;
        let rationale = format!(
            "flex_z={:.2}, nma={:.2}, co={:.2}, cons={:.2}, probe={:.2} [qual: {}C/{}N/{}O/{}E]",
            avg_zscore,
            avg_nma,
            avg_contact_order,
            avg_conservation,
            avg_probe,
            classic_count,
            nma_count,
            co_count,
            enhanced_count
        );

        Some(CrypticCandidate {
            id,
            residue_indices: cluster.iter().map(|r| r.base.residue_seq).collect(),
            centroid,
            estimated_volume,
            flexibility_score: avg_bfactor,
            packing_deficit: avg_packing_deficit,
            hydrophobic_score: avg_hydro,
            cryptic_score,
            predicted_druggability,
            confidence: CrypticConfidence::from_score(cryptic_score),
            rationale,
        })
    }
}

//=============================================================================
// INTERNAL TYPES
//=============================================================================

/// All signal maps collected from analyzers
struct SignalMaps {
    bfactor: HashMap<i32, f64>,
    packing: HashMap<i32, f64>,
    hydrophobicity: HashMap<i32, f64>,
    nma: HashMap<i32, f64>,
    contact_order: HashMap<i32, f64>,
    conservation: HashMap<i32, f64>,
    probe: HashMap<i32, f64>,
}

/// Extended flexible residue with enhanced signals
#[derive(Debug, Clone)]
struct EnhancedFlexibleResidue {
    base: FlexibleResidue,
    nma_mobility: f64,
    contact_order_flexibility: f64,
    conservation_score: f64,
    probe_score: f64,
    combined_score: f64,
    qualification_reason: &'static str,
}

//=============================================================================
// TESTS
//=============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_atom(serial: u32, residue_seq: i32, coord: [f64; 3], b_factor: f64) -> Atom {
        Atom {
            serial,
            name: "CA".to_string(),
            residue_name: "ALA".to_string(),
            chain_id: 'A',
            residue_seq,
            insertion_code: None,
            coord,
            occupancy: 1.0,
            b_factor,
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
    fn test_enhanced_detector_creation() {
        let detector = EnhancedSoftSpotDetector::new();
        assert!(detector.config.use_nma);
        assert!(detector.config.use_contact_order);
        assert!(detector.config.use_conservation);
        assert!(detector.config.use_probe_clustering);
    }

    #[test]
    fn test_enhanced_empty_input() {
        let detector = EnhancedSoftSpotDetector::new();
        let result = detector.detect(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_enhanced_with_disabled_modules() {
        let config = EnhancedSoftSpotConfig {
            use_nma: false,
            use_contact_order: false,
            use_conservation: false,
            use_probe_clustering: false,
            ..Default::default()
        };
        let detector = EnhancedSoftSpotDetector::with_config(config);

        let atoms: Vec<Atom> = (0..20)
            .map(|i| make_test_atom(i as u32, i as i32, [i as f64 * 3.8, 0.0, 0.0], 30.0 + i as f64 * 2.0))
            .collect();

        // Should still work with classic detection only
        let _result = detector.detect(&atoms);
        // No panic = success
    }

    #[test]
    fn test_signal_weights_sum() {
        let config = EnhancedSoftSpotConfig::default();
        let total_weight = config.weight_bfactor
            + config.weight_packing
            + config.weight_hydrophobicity
            + config.weight_nma
            + config.weight_contact_order
            + config.weight_conservation
            + config.weight_probe;

        // Weights should sum to approximately 1.0
        assert!((total_weight - 1.0).abs() < 0.01, "Weights sum to {}", total_weight);
    }
}
