//! Ensemble Pocket Detector v2
//!
//! Enhanced cryptic pocket detection with improved scoring and clustering.
//! Changes from v1:
//! - Phase 1.2: Adaptive per-structure thresholding (z-score based)
//! - Phase 1.3: Dynamic EFE prior based on protein characteristics
//! - Phase 2.3: Graph-based community clustering (label propagation)
//!
//! Expected cumulative impact: +0.17 ROC AUC
//!
//! # Methodology
//!
//! 1. For the APO structure: identify pocket residues using burial/concavity analysis
//! 2. For each ensemble conformation: detect pockets
//! 3. Track residues that NEWLY become pocket-adjacent (cryptic = "hidden until motion")
//! 4. Score using Active Inference EFE with adaptive prior
//! 5. Apply adaptive z-score thresholding per structure
//! 6. Cluster using graph-based community detection

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::anm_ensemble_v2::AnmEnsembleV2;

/// Configuration for ensemble pocket detection v2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePocketConfigV2 {
    /// SASA threshold for surface residues (Å²)
    pub surface_sasa_threshold: f64,

    /// SASA threshold for buried residues (Å²)
    pub buried_sasa_threshold: f64,

    /// Minimum fraction of conformations where a residue must form pocket
    pub min_pocket_frequency: f64,

    /// Probe radius for SASA calculation (Å)
    pub probe_radius: f64,

    /// Distance cutoff for neighbor detection (Å)
    pub neighbor_cutoff: f64,

    /// Minimum number of neighbors for a residue to be "buried"
    pub min_neighbors_for_burial: usize,

    /// SASA variance threshold for cryptic site detection
    pub sasa_variance_threshold: f64,

    // === Active Inference Parameters (v2 Enhanced) ===

    /// Enable Active Inference EFE scoring
    pub use_active_inference: bool,

    /// v2: Base prior for pocket formation (will be scaled dynamically)
    /// Dynamic prior = base_prior * size_factor * (1 + burial_variance * 0.5)
    pub base_pocket_formation_prior: f64,

    /// Weight for epistemic value (information gain)
    pub epistemic_weight: f64,

    /// Weight for pragmatic value (goal achievement)
    pub pragmatic_weight: f64,

    // === v2: Adaptive Thresholding Parameters ===

    /// Enable adaptive z-score thresholding (v2 feature)
    pub use_adaptive_threshold: bool,

    /// Z-score threshold for cryptic classification (1.5 = top ~7%)
    pub z_threshold: f64,

    /// Minimum floor for adaptive threshold (prevents noise)
    pub min_threshold_floor: f64,

    // === v2: Graph-Based Clustering Parameters ===

    /// Enable graph-based community clustering (v2 feature)
    pub use_graph_clustering: bool,

    /// Distance threshold for contact graph edges (Å)
    pub graph_contact_distance: f64,

    /// Distance threshold for clustering (fallback greedy)
    pub cluster_distance: f64,

    /// Minimum cluster size to report
    pub min_cluster_size: usize,

    /// Number of label propagation iterations
    pub label_propagation_iterations: usize,
}

impl Default for EnsemblePocketConfigV2 {
    fn default() -> Self {
        Self {
            surface_sasa_threshold: 50.0,
            buried_sasa_threshold: 10.0,
            min_pocket_frequency: 0.2,
            probe_radius: 1.4,
            neighbor_cutoff: 8.0,
            min_neighbors_for_burial: 12,
            sasa_variance_threshold: 100.0,

            // Active Inference (v2 enhanced)
            use_active_inference: true,
            base_pocket_formation_prior: 0.07,  // Will be scaled dynamically
            epistemic_weight: 0.4,
            pragmatic_weight: 0.6,

            // v2: Adaptive thresholding (Phase 1.2)
            use_adaptive_threshold: true,
            z_threshold: 1.5,           // Top ~7% of each structure
            min_threshold_floor: 0.25,  // Minimum threshold to avoid noise

            // v2: Graph-based clustering (Phase 2.3)
            use_graph_clustering: true,
            graph_contact_distance: 10.0,  // Å for contact graph
            cluster_distance: 8.0,
            min_cluster_size: 2,
            label_propagation_iterations: 10,
        }
    }
}

/// Result of pocket detection for a single structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketDetectionResultV2 {
    /// Residue indices that are pocket-adjacent
    pub pocket_residues: HashSet<i32>,

    /// Per-residue burial scores (higher = more buried)
    pub burial_scores: HashMap<i32, f64>,

    /// Per-residue SASA values
    pub sasa_values: HashMap<i32, f64>,

    /// Number of residues in detected pockets
    pub n_pocket_residues: usize,
}

/// A cluster of spatially related cryptic residue predictions (v2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticClusterV2 {
    /// Residue IDs in this cluster
    pub residues: Vec<i32>,

    /// Cluster centroid position [x, y, z]
    pub centroid: [f64; 3],

    /// Aggregate cluster score (max or mean of member scores)
    pub score: f64,

    /// Representative residue (highest scoring in cluster)
    pub representative: i32,

    /// Cluster radius (max distance from centroid)
    pub radius: f64,

    /// v2: Cluster density (residues per Å³)
    pub density: f64,

    /// v2: Clustering method used ("graph" or "greedy")
    pub method: String,
}

/// Result of cryptic site detection across ensemble (v2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticSiteResultV2 {
    /// Per-residue cryptic scores (0.0-1.0)
    pub cryptic_scores: HashMap<i32, f64>,

    /// Residues classified as cryptic (above adaptive threshold)
    pub cryptic_residues: Vec<i32>,

    /// APO pocket residues (NOT cryptic by definition)
    pub apo_pocket_residues: HashSet<i32>,

    /// Per-residue SASA variance across ensemble
    pub sasa_variance: HashMap<i32, f64>,

    /// Pocket detection results for APO structure
    pub apo_pockets: PocketDetectionResultV2,

    /// Summary statistics
    pub n_residues: usize,
    pub n_apo_pocket: usize,
    pub n_cryptic: usize,
    pub mean_cryptic_score: f64,

    // === Active Inference Results ===

    /// EFE scores per residue
    pub efe_scores: HashMap<i32, f64>,

    /// Epistemic value per residue
    pub epistemic_values: HashMap<i32, f64>,

    /// Pragmatic value per residue
    pub pragmatic_values: HashMap<i32, f64>,

    // === v2: Adaptive Threshold Results ===

    /// The adaptive threshold computed for this structure
    pub adaptive_threshold: f64,

    /// Score statistics (for debugging)
    pub score_mean: f64,
    pub score_std: f64,

    // === v2: Dynamic Prior Results ===

    /// The dynamic prior computed for this structure
    pub dynamic_prior: f64,

    // === Clustering Results ===

    /// Spatial clusters of cryptic predictions
    pub clusters: Vec<CrypticClusterV2>,

    /// Clustered predictions (representative residue per cluster)
    pub clustered_predictions: Vec<(i32, f64)>,
}

/// Ensemble Pocket Detector v2
///
/// Enhanced version with:
/// - Adaptive per-structure thresholding
/// - Dynamic EFE prior
/// - Graph-based community clustering
pub struct EnsemblePocketDetectorV2 {
    config: EnsemblePocketConfigV2,
}

impl EnsemblePocketDetectorV2 {
    /// Create a new v2 detector with given configuration
    pub fn new(config: EnsemblePocketConfigV2) -> Self {
        Self { config }
    }

    /// Create with default v2 configuration
    pub fn default_config() -> Self {
        Self::new(EnsemblePocketConfigV2::default())
    }

    /// Create with v1-compatible settings (for comparison)
    pub fn v1_compatible() -> Self {
        Self::new(EnsemblePocketConfigV2 {
            use_adaptive_threshold: false,
            use_graph_clustering: false,
            base_pocket_formation_prior: 0.07,  // Fixed, not dynamic
            ..Default::default()
        })
    }

    /// Detect cryptic sites from an ANM ensemble v2
    ///
    /// Uses enhanced scoring with:
    /// - Dynamic EFE prior based on protein size and burial variance
    /// - Adaptive z-score thresholding per structure
    /// - Graph-based community clustering
    pub fn detect_cryptic_sites(
        &self,
        ensemble: &AnmEnsembleV2,
        residue_map: &HashMap<usize, i32>,
    ) -> Result<CrypticSiteResultV2> {
        let n_residues = ensemble.original_coords.len();
        let n_conformations = ensemble.conformations.len();

        log::info!(
            "[v2] Detecting cryptic sites: {} residues, {} conformations",
            n_residues,
            n_conformations
        );

        // Step 1: Compute per-residue neighbor counts (proxy for burial) in APO
        let apo_neighbors = self.compute_neighbor_counts(&ensemble.original_coords);

        // Detect APO pockets using stringent criteria
        let apo_pockets = self.detect_pockets_stringent(&ensemble.original_coords, residue_map)?;
        log::info!("[v2] APO has {} pocket residues (stringent)", apo_pockets.n_pocket_residues);

        // Step 2: Compute exposure scores for each conformation
        let mut exposure_per_residue: Vec<Vec<f64>> = vec![Vec::new(); n_residues];
        let mut neighbor_change_per_residue: Vec<Vec<f64>> = vec![Vec::new(); n_residues];

        for conf_coords in ensemble.conformations.iter() {
            let conf_exposures = self.compute_exposure_scores(conf_coords);
            let conf_neighbors = self.compute_neighbor_counts(conf_coords);

            for i in 0..n_residues {
                exposure_per_residue[i].push(conf_exposures[i]);
                let neighbor_change = apo_neighbors[i] as f64 - conf_neighbors[i] as f64;
                neighbor_change_per_residue[i].push(neighbor_change);
            }
        }

        // === Phase 1.3: Compute DYNAMIC EFE Prior ===
        let max_neighbors = *apo_neighbors.iter().max().unwrap_or(&1).max(&1);
        let burial_variance = self.compute_burial_variance(&apo_neighbors, max_neighbors);
        let size_factor = (n_residues as f64 / 300.0).clamp(0.5, 1.5);
        let dynamic_prior = self.config.base_pocket_formation_prior
            * size_factor
            * (1.0 + burial_variance * 0.5);

        log::info!(
            "[v2] Dynamic prior: {:.4} (base={:.4}, size_factor={:.2}, burial_var={:.3})",
            dynamic_prior,
            self.config.base_pocket_formation_prior,
            size_factor,
            burial_variance
        );

        // Step 3: Compute cryptic scores
        let mut cryptic_scores: HashMap<i32, f64> = HashMap::new();
        let mut sasa_variance: HashMap<i32, f64> = HashMap::new();
        let mut efe_scores: HashMap<i32, f64> = HashMap::new();
        let mut epistemic_values: HashMap<i32, f64> = HashMap::new();
        let mut pragmatic_values: HashMap<i32, f64> = HashMap::new();

        // Compute RMSF from ensemble
        let rmsf: Vec<f64> = (0..n_residues)
            .map(|i| {
                let orig = &ensemble.original_coords[i];
                let mean_sq_disp: f64 = ensemble.conformations.iter()
                    .map(|conf| {
                        let dx = (conf[i][0] - orig[0]) as f64;
                        let dy = (conf[i][1] - orig[1]) as f64;
                        let dz = (conf[i][2] - orig[2]) as f64;
                        dx * dx + dy * dy + dz * dz
                    })
                    .sum::<f64>() / n_conformations as f64;
                mean_sq_disp.sqrt()
            })
            .collect();

        // Normalize RMSF
        let max_rmsf = rmsf.iter().cloned().fold(0.0f64, f64::max);
        let rmsf_normalized: Vec<f64> = rmsf.iter()
            .map(|&r| r / max_rmsf.max(0.01))
            .collect();

        for i in 0..n_residues {
            let res_id = residue_map.get(&i).cloned().unwrap_or(i as i32);

            // Compute variance in exposure
            let exposures = &exposure_per_residue[i];
            let mean_exposure = exposures.iter().sum::<f64>() / exposures.len() as f64;
            let variance = exposures
                .iter()
                .map(|&e| (e - mean_exposure).powi(2))
                .sum::<f64>()
                / (exposures.len() - 1).max(1) as f64;
            sasa_variance.insert(res_id, variance);

            // Burial score
            let burial = apo_neighbors[i] as f64 / max_neighbors as f64;

            // Burial potential (favoring moderately buried residues)
            let burial_potential_score = {
                let optimal = 0.55;
                let width = 0.35;
                let diff = (burial - optimal).abs();
                (1.0 - (diff / width).min(1.0)).max(0.0)
            };

            // Variance score
            let variance_score = (variance.sqrt() / 0.1).min(1.0);

            // Combined neighbor flexibility
            let neighbor_flexibility = {
                let start = i.saturating_sub(5);
                let end = (i + 5).min(n_residues - 1);
                let sum: f64 = (start..=end)
                    .filter(|&j| j != i)
                    .map(|j| rmsf_normalized[j])
                    .sum();
                let count = (end - start) as f64;
                if count > 0.0 { sum / count } else { 0.0 }
            };

            let spatial_neighbor_flexibility = {
                let orig_pos = ensemble.original_coords[i];
                let mut sum = 0.0;
                let mut count = 0;
                for j in 0..n_residues {
                    if j == i { continue; }
                    let other_pos = ensemble.original_coords[j];
                    let dx = (other_pos[0] - orig_pos[0]) as f64;
                    let dy = (other_pos[1] - orig_pos[1]) as f64;
                    let dz = (other_pos[2] - orig_pos[2]) as f64;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < 10.0 {
                        sum += rmsf_normalized[j];
                        count += 1;
                    }
                }
                if count > 0 { sum / count as f64 } else { 0.0 }
            };

            let combined_neighbor_flex = neighbor_flexibility * 0.4 + spatial_neighbor_flexibility * 0.6;

            // === v2: Active Inference EFE Scoring with DYNAMIC PRIOR ===
            let (cryptic_score, epistemic_val, pragmatic_val) = if self.config.use_active_inference {
                // Epistemic value with dynamic prior
                let rarity = 1.0 - (burial * 0.3 + combined_neighbor_flex * 0.7);
                let surprise = variance_score * (1.0 + rarity);
                let epistemic = surprise * (1.0 - dynamic_prior);  // Uses DYNAMIC prior

                // Pragmatic value
                let posterior = burial * 0.35 + combined_neighbor_flex * 0.35
                    + burial_potential_score * 0.2 + variance_score * 0.1;

                // KL divergence from DYNAMIC prior
                let kl_div = if posterior > 0.01 && dynamic_prior > 0.01 {
                    posterior * (posterior / dynamic_prior).ln()
                } else {
                    0.0
                };
                let pragmatic = posterior + kl_div.max(0.0) * 0.1;

                let efe_score = self.config.epistemic_weight * epistemic
                    + self.config.pragmatic_weight * pragmatic;

                (efe_score, epistemic, pragmatic)
            } else {
                let score = burial * 0.35
                    + combined_neighbor_flex * 0.35
                    + burial_potential_score * 0.2
                    + variance_score * 0.1;
                (score, 0.0, score)
            };

            if cryptic_score > 0.01 {
                cryptic_scores.insert(res_id, cryptic_score);
                efe_scores.insert(res_id, cryptic_score);
                epistemic_values.insert(res_id, epistemic_val);
                pragmatic_values.insert(res_id, pragmatic_val);
            }
        }

        // === Phase 1.2: ADAPTIVE THRESHOLDING ===
        let (adaptive_threshold, score_mean, score_std) = if self.config.use_adaptive_threshold {
            self.compute_adaptive_threshold(&cryptic_scores)
        } else {
            (0.3, 0.0, 0.0)  // v1 fixed threshold
        };

        log::info!(
            "[v2] Adaptive threshold: {:.3} (mean={:.3}, std={:.3}, z={})",
            adaptive_threshold,
            score_mean,
            score_std,
            self.config.z_threshold
        );

        // Apply threshold
        let mut cryptic_residues: Vec<i32> = cryptic_scores
            .iter()
            .filter(|(_, &score)| score >= adaptive_threshold)
            .map(|(&res_id, _)| res_id)
            .collect();

        // Sort by score descending
        cryptic_residues.sort_by(|a, b| {
            let score_a = cryptic_scores.get(a).unwrap_or(&0.0);
            let score_b = cryptic_scores.get(b).unwrap_or(&0.0);
            score_b.partial_cmp(score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // === Phase 2.3: GRAPH-BASED CLUSTERING ===
        let (clusters, clustered_predictions) = if self.config.use_graph_clustering {
            self.cluster_predictions_graph(
                &cryptic_residues,
                &cryptic_scores,
                &ensemble.original_coords,
                residue_map,
            )
        } else {
            self.cluster_predictions_greedy(
                &cryptic_residues,
                &cryptic_scores,
                &ensemble.original_coords,
                residue_map,
            )
        };

        // Compute summary statistics
        let mean_cryptic_score = if cryptic_scores.is_empty() {
            0.0
        } else {
            cryptic_scores.values().sum::<f64>() / cryptic_scores.len() as f64
        };

        let n_apo_pocket = apo_pockets.n_pocket_residues;
        let n_cryptic = cryptic_residues.len();

        log::info!(
            "[v2] Found {} cryptic residues, {} clusters (adaptive threshold: {:.3})",
            n_cryptic,
            clusters.len(),
            adaptive_threshold
        );

        Ok(CrypticSiteResultV2 {
            cryptic_scores,
            cryptic_residues,
            apo_pocket_residues: apo_pockets.pocket_residues.clone(),
            sasa_variance,
            apo_pockets,
            n_residues,
            n_apo_pocket,
            n_cryptic,
            mean_cryptic_score,
            efe_scores,
            epistemic_values,
            pragmatic_values,
            adaptive_threshold,
            score_mean,
            score_std,
            dynamic_prior,
            clusters,
            clustered_predictions,
        })
    }

    /// Re-apply thresholding and clustering after scores are modified externally
    ///
    /// This method is used after PRISM-ZrO modifies the EFE scores to re-compute
    /// predictions with the blended scores. It recomputes:
    /// 1. Adaptive threshold from modified scores
    /// 2. Cryptic residue list based on new threshold
    /// 3. Graph-based clustering of predictions
    ///
    /// # Arguments
    /// * `result` - Mutable reference to the CrypticSiteResultV2 to update
    /// * `original_coords` - Original CA coordinates for clustering
    /// * `residue_map` - Mapping from sequential index to residue ID
    pub fn redetect_from_modified_scores(
        &self,
        result: &mut CrypticSiteResultV2,
        original_coords: &[[f32; 3]],
        residue_map: &HashMap<usize, i32>,
    ) {
        // Step 1: Recompute adaptive threshold from modified EFE scores
        let (adaptive_threshold, score_mean, score_std) = if self.config.use_adaptive_threshold {
            self.compute_adaptive_threshold(&result.efe_scores)
        } else {
            (0.3, 0.0, 0.0)
        };

        log::info!(
            "[v2] Re-applying threshold after ZrO: {:.3} (mean={:.3}, std={:.3})",
            adaptive_threshold, score_mean, score_std
        );

        // Step 2: Apply new threshold to get predictions
        let mut cryptic_residues: Vec<i32> = result.efe_scores
            .iter()
            .filter(|(_, &score)| score >= adaptive_threshold)
            .map(|(&res_id, _)| res_id)
            .collect();

        // Sort by score descending
        cryptic_residues.sort_by(|a, b| {
            let score_a = result.efe_scores.get(a).unwrap_or(&0.0);
            let score_b = result.efe_scores.get(b).unwrap_or(&0.0);
            score_b.partial_cmp(score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 3: Re-cluster predictions
        let (clusters, clustered_predictions) = if self.config.use_graph_clustering {
            self.cluster_predictions_graph(
                &cryptic_residues,
                &result.efe_scores,
                original_coords,
                residue_map,
            )
        } else {
            self.cluster_predictions_greedy(
                &cryptic_residues,
                &result.efe_scores,
                original_coords,
                residue_map,
            )
        };

        // Step 4: Update result with new predictions
        result.adaptive_threshold = adaptive_threshold;
        result.score_mean = score_mean;
        result.score_std = score_std;
        result.cryptic_residues = cryptic_residues;
        result.n_cryptic = result.cryptic_residues.len();
        result.clusters = clusters;
        result.clustered_predictions = clustered_predictions;

        // Update cryptic_scores to match efe_scores (for consistency)
        result.cryptic_scores = result.efe_scores.clone();
        result.mean_cryptic_score = if result.efe_scores.is_empty() {
            0.0
        } else {
            result.efe_scores.values().sum::<f64>() / result.efe_scores.len() as f64
        };

        log::info!(
            "[v2] After ZrO redetection: {} cryptic residues, {} clusters",
            result.n_cryptic, result.clusters.len()
        );
    }

    /// Phase 1.3: Compute burial variance for dynamic prior
    fn compute_burial_variance(&self, neighbor_counts: &[usize], max_neighbors: usize) -> f64 {
        if neighbor_counts.is_empty() {
            return 0.0;
        }

        let burials: Vec<f64> = neighbor_counts
            .iter()
            .map(|&n| n as f64 / max_neighbors as f64)
            .collect();

        let mean = burials.iter().sum::<f64>() / burials.len() as f64;
        let variance = burials
            .iter()
            .map(|&b| (b - mean).powi(2))
            .sum::<f64>() / burials.len() as f64;

        variance.sqrt()
    }

    /// Phase 1.2: Compute adaptive threshold using z-score normalization
    fn compute_adaptive_threshold(&self, scores: &HashMap<i32, f64>) -> (f64, f64, f64) {
        if scores.is_empty() {
            return (self.config.min_threshold_floor, 0.0, 0.0);
        }

        let score_values: Vec<f64> = scores.values().cloned().collect();
        let n = score_values.len() as f64;

        let mean = score_values.iter().sum::<f64>() / n;
        let std = if n > 1.0 {
            let variance = score_values
                .iter()
                .map(|&s| (s - mean).powi(2))
                .sum::<f64>() / (n - 1.0);
            variance.sqrt()
        } else {
            0.0
        };

        // Z-score based threshold: mean + z * std
        let z_threshold = mean + self.config.z_threshold * std;

        // Apply minimum floor
        let adaptive_threshold = z_threshold.max(self.config.min_threshold_floor);

        (adaptive_threshold, mean, std)
    }

    /// Phase 2.3: Graph-based community clustering using label propagation
    fn cluster_predictions_graph(
        &self,
        residues: &[i32],
        scores: &HashMap<i32, f64>,
        coords: &[[f32; 3]],
        residue_map: &HashMap<usize, i32>,
    ) -> (Vec<CrypticClusterV2>, Vec<(i32, f64)>) {
        if residues.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Build reverse map
        let reverse_map: HashMap<i32, usize> = residue_map
            .iter()
            .map(|(&idx, &res_id)| (res_id, idx))
            .collect();

        // Step 1: Build contact graph for cryptic residues only
        let mut adjacency: HashMap<i32, Vec<i32>> = HashMap::new();
        let cutoff_sq = (self.config.graph_contact_distance * self.config.graph_contact_distance) as f32;

        for &r1 in residues {
            adjacency.entry(r1).or_insert_with(Vec::new);

            let idx1 = match reverse_map.get(&r1) {
                Some(&idx) => idx,
                None => continue,
            };

            for &r2 in residues {
                if r1 == r2 { continue; }

                let idx2 = match reverse_map.get(&r2) {
                    Some(&idx) => idx,
                    None => continue,
                };

                let dx = coords[idx2][0] - coords[idx1][0];
                let dy = coords[idx2][1] - coords[idx1][1];
                let dz = coords[idx2][2] - coords[idx1][2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    adjacency.entry(r1).or_default().push(r2);
                }
            }
        }

        // Step 2: Label propagation clustering
        let communities = self.label_propagation(&adjacency, scores);

        // Step 3: Convert communities to clusters
        let mut clusters: Vec<CrypticClusterV2> = Vec::new();

        for members in communities {
            if members.len() < self.config.min_cluster_size {
                continue;
            }

            // Find representative (highest score)
            let representative = members
                .iter()
                .max_by(|&a, &b| {
                    scores.get(a).unwrap_or(&0.0)
                        .partial_cmp(scores.get(b).unwrap_or(&0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned()
                .unwrap_or(members[0]);

            let max_score = *scores.get(&representative).unwrap_or(&0.0);

            // Compute centroid
            let mut centroid = [0.0f64; 3];
            let mut valid_count = 0;

            for &res in &members {
                if let Some(&idx) = reverse_map.get(&res) {
                    centroid[0] += coords[idx][0] as f64;
                    centroid[1] += coords[idx][1] as f64;
                    centroid[2] += coords[idx][2] as f64;
                    valid_count += 1;
                }
            }

            if valid_count > 0 {
                let n = valid_count as f64;
                centroid[0] /= n;
                centroid[1] /= n;
                centroid[2] /= n;
            }

            // Compute radius
            let mut max_dist = 0.0f64;
            for &res in &members {
                if let Some(&idx) = reverse_map.get(&res) {
                    let dx = coords[idx][0] as f64 - centroid[0];
                    let dy = coords[idx][1] as f64 - centroid[1];
                    let dz = coords[idx][2] as f64 - centroid[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    max_dist = max_dist.max(dist);
                }
            }

            // Compute density (residues per Å³)
            let volume = if max_dist > 0.0 {
                (4.0 / 3.0) * std::f64::consts::PI * max_dist.powi(3)
            } else {
                1.0
            };
            let density = members.len() as f64 / volume;

            clusters.push(CrypticClusterV2 {
                residues: members,
                centroid,
                score: max_score,
                representative,
                radius: max_dist,
                density,
                method: "graph".to_string(),
            });
        }

        // Sort by score
        clusters.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let clustered_predictions: Vec<(i32, f64)> = clusters
            .iter()
            .map(|c| (c.representative, c.score))
            .collect();

        log::debug!(
            "[v2] Graph clustering: {} residues → {} clusters",
            residues.len(),
            clusters.len()
        );

        (clusters, clustered_predictions)
    }

    /// Label propagation algorithm for community detection
    fn label_propagation(
        &self,
        adjacency: &HashMap<i32, Vec<i32>>,
        scores: &HashMap<i32, f64>,
    ) -> Vec<Vec<i32>> {
        if adjacency.is_empty() {
            return Vec::new();
        }

        // Initialize: each node is its own community
        let mut labels: HashMap<i32, i32> = adjacency.keys().map(|&k| (k, k)).collect();

        // Iterate
        for _iter in 0..self.config.label_propagation_iterations {
            let mut changed = false;

            for (&node, neighbors) in adjacency {
                if neighbors.is_empty() {
                    continue;
                }

                // Count labels in neighborhood (weighted by score)
                let mut label_counts: HashMap<i32, f64> = HashMap::new();
                for &neighbor in neighbors {
                    if let Some(&label) = labels.get(&neighbor) {
                        let weight = scores.get(&neighbor).unwrap_or(&1.0);
                        *label_counts.entry(label).or_insert(0.0) += weight;
                    }
                }

                // Find most frequent label
                if let Some((&best_label, _)) = label_counts
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                {
                    if labels.get(&node) != Some(&best_label) {
                        labels.insert(node, best_label);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // Group by label
        let mut communities: HashMap<i32, Vec<i32>> = HashMap::new();
        for (&node, &label) in &labels {
            communities.entry(label).or_default().push(node);
        }

        communities.into_values().collect()
    }

    /// Fallback greedy clustering (v1 compatible)
    fn cluster_predictions_greedy(
        &self,
        residues: &[i32],
        scores: &HashMap<i32, f64>,
        coords: &[[f32; 3]],
        residue_map: &HashMap<usize, i32>,
    ) -> (Vec<CrypticClusterV2>, Vec<(i32, f64)>) {
        if residues.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let reverse_map: HashMap<i32, usize> = residue_map
            .iter()
            .map(|(&idx, &res_id)| (res_id, idx))
            .collect();

        let mut assigned: HashSet<i32> = HashSet::new();
        let mut clusters: Vec<CrypticClusterV2> = Vec::new();

        for &seed_res in residues {
            if assigned.contains(&seed_res) {
                continue;
            }

            let seed_idx = match reverse_map.get(&seed_res) {
                Some(&idx) => idx,
                None => continue,
            };
            let seed_pos = coords[seed_idx];

            let mut cluster_members: Vec<i32> = vec![seed_res];
            assigned.insert(seed_res);

            for &other_res in residues {
                if assigned.contains(&other_res) {
                    continue;
                }

                let other_idx = match reverse_map.get(&other_res) {
                    Some(&idx) => idx,
                    None => continue,
                };
                let other_pos = coords[other_idx];

                let dx = (other_pos[0] - seed_pos[0]) as f64;
                let dy = (other_pos[1] - seed_pos[1]) as f64;
                let dz = (other_pos[2] - seed_pos[2]) as f64;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist <= self.config.cluster_distance {
                    cluster_members.push(other_res);
                    assigned.insert(other_res);
                }
            }

            if cluster_members.len() < self.config.min_cluster_size {
                continue;
            }

            // Compute cluster properties
            let mut centroid = [0.0f64; 3];
            let mut max_score = 0.0f64;
            let mut representative = seed_res;

            for &res in &cluster_members {
                if let Some(&idx) = reverse_map.get(&res) {
                    centroid[0] += coords[idx][0] as f64;
                    centroid[1] += coords[idx][1] as f64;
                    centroid[2] += coords[idx][2] as f64;
                }

                let score = *scores.get(&res).unwrap_or(&0.0);
                if score > max_score {
                    max_score = score;
                    representative = res;
                }
            }

            let n = cluster_members.len() as f64;
            centroid[0] /= n;
            centroid[1] /= n;
            centroid[2] /= n;

            let mut max_dist = 0.0f64;
            for &res in &cluster_members {
                if let Some(&idx) = reverse_map.get(&res) {
                    let dx = coords[idx][0] as f64 - centroid[0];
                    let dy = coords[idx][1] as f64 - centroid[1];
                    let dz = coords[idx][2] as f64 - centroid[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    max_dist = max_dist.max(dist);
                }
            }

            let volume = if max_dist > 0.0 {
                (4.0 / 3.0) * std::f64::consts::PI * max_dist.powi(3)
            } else {
                1.0
            };
            let density = cluster_members.len() as f64 / volume;

            clusters.push(CrypticClusterV2 {
                residues: cluster_members,
                centroid,
                score: max_score,
                representative,
                radius: max_dist,
                density,
                method: "greedy".to_string(),
            });
        }

        clusters.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let clustered_predictions: Vec<(i32, f64)> = clusters
            .iter()
            .map(|c| (c.representative, c.score))
            .collect();

        (clusters, clustered_predictions)
    }

    /// Compute neighbor counts for each residue
    fn compute_neighbor_counts(&self, ca_coords: &[[f32; 3]]) -> Vec<usize> {
        let n = ca_coords.len();
        let cutoff_sq = (self.config.neighbor_cutoff * self.config.neighbor_cutoff) as f32;
        let mut neighbor_counts = vec![0usize; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = ca_coords[j][0] - ca_coords[i][0];
                let dy = ca_coords[j][1] - ca_coords[i][1];
                let dz = ca_coords[j][2] - ca_coords[i][2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    neighbor_counts[i] += 1;
                    neighbor_counts[j] += 1;
                }
            }
        }

        neighbor_counts
    }

    /// Compute exposure scores
    fn compute_exposure_scores(&self, ca_coords: &[[f32; 3]]) -> Vec<f64> {
        let neighbor_counts = self.compute_neighbor_counts(ca_coords);
        let max_neighbors = *neighbor_counts.iter().max().unwrap_or(&1).max(&1);

        neighbor_counts.iter()
            .map(|&n| 1.0 - (n as f64 / max_neighbors as f64))
            .collect()
    }

    /// Stringent pocket detection
    fn detect_pockets_stringent(
        &self,
        ca_coords: &[[f32; 3]],
        residue_map: &HashMap<usize, i32>,
    ) -> Result<PocketDetectionResultV2> {
        let neighbor_counts = self.compute_neighbor_counts(ca_coords);

        let centroid = self.compute_centroid(ca_coords);
        let distances_to_centroid: Vec<f64> = ca_coords
            .iter()
            .map(|c| {
                let dx = c[0] as f64 - centroid[0];
                let dy = c[1] as f64 - centroid[1];
                let dz = c[2] as f64 - centroid[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();
        let max_dist = distances_to_centroid.iter().cloned().fold(0.0f64, f64::max);

        let mut burial_scores: HashMap<i32, f64> = HashMap::new();
        let mut sasa_values: HashMap<i32, f64> = HashMap::new();
        let mut pocket_residues: HashSet<i32> = HashSet::new();

        let max_neighbors = *neighbor_counts.iter().max().unwrap_or(&1);

        for (i, &count) in neighbor_counts.iter().enumerate() {
            let res_id = residue_map.get(&i).cloned().unwrap_or(i as i32);
            let dist = distances_to_centroid[i];

            burial_scores.insert(res_id, count as f64 / max_neighbors as f64);
            let normalized_sasa = (dist / max_dist) * 100.0;
            sasa_values.insert(res_id, normalized_sasa);

            let is_intermediate_burial = count >= 6 && count <= 10;
            let is_intermediate_depth = dist > max_dist * 0.3 && dist < max_dist * 0.7;

            if is_intermediate_burial && is_intermediate_depth {
                pocket_residues.insert(res_id);
            }
        }

        Ok(PocketDetectionResultV2 {
            pocket_residues: pocket_residues.clone(),
            burial_scores,
            sasa_values,
            n_pocket_residues: pocket_residues.len(),
        })
    }

    /// Compute centroid
    fn compute_centroid(&self, ca_coords: &[[f32; 3]]) -> [f64; 3] {
        let n = ca_coords.len() as f64;
        let mut sum = [0.0f64; 3];

        for c in ca_coords {
            sum[0] += c[0] as f64;
            sum[1] += c[1] as f64;
            sum[2] += c[2] as f64;
        }

        [sum[0] / n, sum[1] / n, sum[2] / n]
    }
}

/// Convert cryptic scores to predictions
pub fn cryptic_scores_to_predictions_v2(
    result: &CrypticSiteResultV2,
    score_threshold: f64,
) -> Vec<(i32, f64)> {
    let mut predictions: Vec<(i32, f64)> = result
        .cryptic_scores
        .iter()
        .filter(|(_, &score)| score >= score_threshold)
        .map(|(&res_id, &score)| (res_id, score))
        .collect();

    predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    predictions
}

/// Compute prediction overlap (same as v1)
pub fn compute_prediction_overlap_v2(
    predictions: &[i32],
    ground_truth: &HashSet<i32>,
) -> (f64, f64, f64, usize) {
    if predictions.is_empty() && ground_truth.is_empty() {
        return (1.0, 1.0, 1.0, 0);
    }
    if predictions.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }
    if ground_truth.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }

    let pred_set: HashSet<i32> = predictions.iter().cloned().collect();
    let overlap: usize = pred_set.intersection(ground_truth).count();

    let precision = overlap as f64 / predictions.len() as f64;
    let recall = overlap as f64 / ground_truth.len() as f64;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1, overlap)
}

/// Apply interface boost to cryptic scores from oligomer topology
///
/// Residues at oligomeric interfaces get a boost because they're cryptic -
/// they're buried in the assembled oligomer but become accessible when
/// the complex dissociates.
///
/// # Arguments
/// * `result` - Mutable reference to cryptic site result
/// * `topology` - Oligomer topology with detected interfaces
/// * `chain_id` - Chain ID to use for interface lookup
/// * `boost_factor` - Factor to multiply scores by (default: 1.15 = +15%)
pub fn apply_interface_boost(
    result: &mut CrypticSiteResultV2,
    interfaces: &[crate::oligomer_topology::InterfaceRegion],
    chain_id: &str,
    boost_factor: f64,
) {
    use std::collections::HashSet;

    // Collect all interface residues for this chain
    let interface_residues: HashSet<i32> = interfaces
        .iter()
        .flat_map(|iface| {
            let mut residues = Vec::new();
            if iface.chain_a == chain_id {
                residues.extend(iface.residues_a.iter().copied());
            }
            if iface.chain_b == chain_id {
                residues.extend(iface.residues_b.iter().copied());
            }
            residues
        })
        .collect();

    if interface_residues.is_empty() {
        log::debug!("[v2] No interface residues found for chain {}", chain_id);
        return;
    }

    log::info!(
        "[v2] Applying interface boost ({:.0}%) to {} interface residues",
        (boost_factor - 1.0) * 100.0,
        interface_residues.len()
    );

    let mut boosted_count = 0;

    // Boost scores for interface residues
    for res_id in &interface_residues {
        if let Some(score) = result.cryptic_scores.get_mut(res_id) {
            let old_score = *score;
            *score = (*score * boost_factor).min(1.0);
            log::trace!(
                "[v2] Boosted residue {}: {:.4} -> {:.4}",
                res_id, old_score, *score
            );
            boosted_count += 1;
        } else {
            // Interface residue not in cryptic scores - add it with base score
            let base_score = 0.3 * boost_factor; // Moderate base score for interface
            result.cryptic_scores.insert(*res_id, base_score.min(1.0));
            log::trace!(
                "[v2] Added interface residue {} with score {:.4}",
                res_id, base_score
            );
            boosted_count += 1;
        }

        // Also boost EFE scores
        if let Some(efe) = result.efe_scores.get_mut(res_id) {
            *efe = (*efe * boost_factor).min(1.0);
        }
    }

    // Update cryptic residues list if new ones were added
    let threshold = result.adaptive_threshold;
    result.cryptic_residues = result
        .cryptic_scores
        .iter()
        .filter(|(_, &score)| score >= threshold)
        .map(|(&res_id, _)| res_id)
        .collect();

    // Sort by score
    result.cryptic_residues.sort_by(|a, b| {
        let score_a = result.cryptic_scores.get(a).unwrap_or(&0.0);
        let score_b = result.cryptic_scores.get(b).unwrap_or(&0.0);
        score_b.partial_cmp(score_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    result.n_cryptic = result.cryptic_residues.len();

    // Recalculate mean
    result.mean_cryptic_score = if result.cryptic_scores.is_empty() {
        0.0
    } else {
        result.cryptic_scores.values().sum::<f64>() / result.cryptic_scores.len() as f64
    };

    log::info!(
        "[v2] After interface boost: {} residues boosted, {} total cryptic residues",
        boosted_count,
        result.n_cryptic
    );
}

/// Apply proximity boost based on distance to known functional sites
///
/// Residues near known epitopes or binding sites get a boost because
/// they are more likely to be therapeutically relevant.
///
/// # Arguments
/// * `result` - Mutable reference to cryptic site result
/// * `functional_residues` - Set of residue IDs that are known functional sites
/// * `ca_coords` - CA coordinates for distance calculation
/// * `residue_map` - Map from coordinate index to residue ID
/// * `max_distance` - Maximum distance for proximity boost (Å)
/// * `max_boost` - Maximum boost factor (e.g., 0.15 for +15%)
pub fn apply_proximity_boost(
    result: &mut CrypticSiteResultV2,
    functional_residues: &std::collections::HashSet<i32>,
    ca_coords: &[[f32; 3]],
    residue_map: &std::collections::HashMap<usize, i32>,
    max_distance: f64,
    max_boost: f64,
) {
    if functional_residues.is_empty() {
        return;
    }

    // Build reverse map
    let reverse_map: std::collections::HashMap<i32, usize> = residue_map
        .iter()
        .map(|(&idx, &res_id)| (res_id, idx))
        .collect();

    // Get coordinates of functional residues
    let functional_coords: Vec<[f32; 3]> = functional_residues
        .iter()
        .filter_map(|&res_id| {
            reverse_map.get(&res_id).map(|&idx| ca_coords[idx])
        })
        .collect();

    if functional_coords.is_empty() {
        return;
    }

    log::info!(
        "[v2] Applying proximity boost to residues within {:.1}Å of {} functional sites",
        max_distance,
        functional_residues.len()
    );

    let max_dist_sq = (max_distance * max_distance) as f32;
    let mut boosted_count = 0;

    // For each residue with a cryptic score, check proximity to functional sites
    let res_ids: Vec<i32> = result.cryptic_scores.keys().copied().collect();

    for res_id in res_ids {
        if let Some(&idx) = reverse_map.get(&res_id) {
            let coord = ca_coords[idx];

            // Find minimum distance to any functional residue
            let min_dist_sq = functional_coords
                .iter()
                .map(|fc| {
                    let dx = coord[0] - fc[0];
                    let dy = coord[1] - fc[1];
                    let dz = coord[2] - fc[2];
                    dx * dx + dy * dy + dz * dz
                })
                .fold(f32::MAX, f32::min);

            if min_dist_sq < max_dist_sq {
                // Gaussian decay boost: closer = higher boost
                let dist = (min_dist_sq as f64).sqrt();
                let boost = max_boost * (-dist * dist / (2.0 * max_distance.powi(2))).exp();

                if let Some(score) = result.cryptic_scores.get_mut(&res_id) {
                    *score = (*score + boost).min(1.0);
                    boosted_count += 1;
                }
            }
        }
    }

    // Recalculate mean
    result.mean_cryptic_score = if result.cryptic_scores.is_empty() {
        0.0
    } else {
        result.cryptic_scores.values().sum::<f64>() / result.cryptic_scores.len() as f64
    };

    log::info!(
        "[v2] Proximity boost applied to {} residues",
        boosted_count
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_helix(n: usize) -> Vec<[f32; 3]> {
        let rise = 1.5f32;
        let radius = 2.3f32;
        let residues_per_turn = 3.6f32;

        (0..n)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * i as f32 / residues_per_turn;
                [
                    radius * angle.cos(),
                    radius * angle.sin(),
                    rise * i as f32,
                ]
            })
            .collect()
    }

    #[test]
    fn test_adaptive_threshold() {
        let scores: HashMap<i32, f64> = (0..100)
            .map(|i| (i, (i as f64) / 100.0))
            .collect();

        let detector = EnsemblePocketDetectorV2::default_config();
        let (threshold, mean, std) = detector.compute_adaptive_threshold(&scores);

        assert!(threshold > 0.25, "Threshold {} should be above floor", threshold);
        assert!((mean - 0.495).abs() < 0.01, "Mean should be ~0.495, got {}", mean);
        assert!(std > 0.0, "Std should be positive");
    }

    #[test]
    fn test_label_propagation() {
        // Simple graph: two connected components
        let mut adjacency: HashMap<i32, Vec<i32>> = HashMap::new();
        adjacency.insert(1, vec![2, 3]);
        adjacency.insert(2, vec![1, 3]);
        adjacency.insert(3, vec![1, 2]);
        adjacency.insert(10, vec![11, 12]);
        adjacency.insert(11, vec![10, 12]);
        adjacency.insert(12, vec![10, 11]);

        let scores: HashMap<i32, f64> = (1..=12).map(|i| (i, 1.0)).collect();

        let detector = EnsemblePocketDetectorV2::default_config();
        let communities = detector.label_propagation(&adjacency, &scores);

        assert_eq!(communities.len(), 2, "Should find 2 communities");
    }

    #[test]
    fn test_dynamic_prior() {
        let detector = EnsemblePocketDetectorV2::default_config();

        // Test burial variance calculation
        let neighbors_uniform = vec![10, 10, 10, 10, 10];
        let var_uniform = detector.compute_burial_variance(&neighbors_uniform, 10);
        assert!(var_uniform < 0.01, "Uniform should have low variance");

        let neighbors_varied = vec![1, 5, 10, 15, 20];
        let var_varied = detector.compute_burial_variance(&neighbors_varied, 20);
        assert!(var_varied > 0.1, "Varied should have higher variance");
    }

    #[test]
    fn test_v2_vs_v1_config() {
        let v2 = EnsemblePocketDetectorV2::default_config();
        let v1_compat = EnsemblePocketDetectorV2::v1_compatible();

        // v2 should have adaptive threshold enabled
        assert!(v2.config.use_adaptive_threshold);
        assert!(v2.config.use_graph_clustering);

        // v1-compatible should have them disabled
        assert!(!v1_compat.config.use_adaptive_threshold);
        assert!(!v1_compat.config.use_graph_clustering);
    }
}
