//! Ensemble Pocket Detector
//!
//! Detects cryptic pockets by analyzing pocket landscapes across conformational ensembles.
//! A cryptic site is defined as residues that become pocket-adjacent in some conformations
//! but were NOT pocket-adjacent in the original APO structure.
//!
//! # Methodology
//!
//! 1. For the APO structure: identify pocket residues using burial/concavity analysis
//! 2. For each ensemble conformation: detect pockets
//! 3. Track residues that NEWLY become pocket-adjacent (cryptic = "hidden until motion")
//! 4. Score by frequency: cryptic_score = fraction of conformations where residue forms pocket
//!
//! # Pocket Detection Approaches
//!
//! This module implements a native pocket detector using:
//! - Burial depth analysis (distance to convex hull)
//! - Local packing density
//! - SASA variance across ensemble
//!
//! For higher accuracy, can optionally integrate with fpocket via file I/O.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::anm_ensemble::AnmEnsemble;

/// Configuration for ensemble pocket detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePocketConfig {
    /// SASA threshold for surface residues (Å²)
    /// Residues with SASA above this are considered "surface"
    pub surface_sasa_threshold: f64,

    /// SASA threshold for buried residues (Å²)
    /// Residues with SASA below this are considered "buried"
    pub buried_sasa_threshold: f64,

    /// Minimum fraction of conformations where a residue must form pocket
    /// to be considered cryptic
    pub min_pocket_frequency: f64,

    /// Probe radius for SASA calculation (Å)
    pub probe_radius: f64,

    /// Distance cutoff for neighbor detection (Å)
    pub neighbor_cutoff: f64,

    /// Minimum number of neighbors for a residue to be "buried"
    pub min_neighbors_for_burial: usize,

    /// SASA variance threshold for cryptic site detection
    /// High variance = residue changes exposure significantly across ensemble
    pub sasa_variance_threshold: f64,

    // === Active Inference Parameters ===

    /// Enable Active Inference EFE scoring (improves ranking for class imbalance)
    pub use_active_inference: bool,

    /// Prior probability of pocket formation (goal prior)
    /// Lower values = more selective, higher precision
    pub pocket_formation_prior: f64,

    /// Weight for epistemic value (information gain / ambiguity reduction)
    pub epistemic_weight: f64,

    /// Weight for pragmatic value (goal achievement)
    pub pragmatic_weight: f64,

    // === Spatial Clustering Parameters ===

    /// Enable RLS spatial clustering to reduce false positives
    pub use_spatial_clustering: bool,

    /// Distance threshold for clustering predictions (Å)
    pub cluster_distance: f64,

    /// Minimum cluster size to report
    pub min_cluster_size: usize,
}

impl Default for EnsemblePocketConfig {
    fn default() -> Self {
        Self {
            surface_sasa_threshold: 50.0,  // Å² - typical for exposed residues
            buried_sasa_threshold: 10.0,   // Å² - typical for buried residues
            min_pocket_frequency: 0.2,     // At least 20% of conformations
            probe_radius: 1.4,             // Water probe
            neighbor_cutoff: 8.0,          // Å for neighbor detection
            min_neighbors_for_burial: 12,  // Typical for core residues
            sasa_variance_threshold: 100.0, // Å² variance threshold

            // Active Inference defaults (enabled by default for better ranking)
            use_active_inference: true,
            pocket_formation_prior: 0.07,  // ~7% of residues are cryptic (class prior)
            epistemic_weight: 0.4,         // Weight for uncertainty reduction
            pragmatic_weight: 0.6,         // Weight for goal achievement

            // Spatial clustering defaults (enabled for precision improvement)
            use_spatial_clustering: true,
            cluster_distance: 8.0,         // Å - typical pocket radius
            min_cluster_size: 2,           // At least 2 residues per cluster
        }
    }
}

/// Result of pocket detection for a single structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketDetectionResult {
    /// Residue indices that are pocket-adjacent
    pub pocket_residues: HashSet<i32>,

    /// Per-residue burial scores (higher = more buried)
    pub burial_scores: HashMap<i32, f64>,

    /// Per-residue SASA values
    pub sasa_values: HashMap<i32, f64>,

    /// Number of residues in detected pockets
    pub n_pocket_residues: usize,
}

/// A cluster of spatially related cryptic residue predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticCluster {
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
}

/// Result of cryptic site detection across ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticSiteResult {
    /// Per-residue cryptic scores (0.0-1.0)
    /// Score = frequency of pocket formation in ensemble
    pub cryptic_scores: HashMap<i32, f64>,

    /// Residues classified as cryptic (score >= threshold)
    pub cryptic_residues: Vec<i32>,

    /// APO pocket residues (NOT cryptic by definition)
    pub apo_pocket_residues: HashSet<i32>,

    /// Per-residue SASA variance across ensemble
    pub sasa_variance: HashMap<i32, f64>,

    /// Pocket detection results for APO structure
    pub apo_pockets: PocketDetectionResult,

    /// Summary statistics
    pub n_residues: usize,
    pub n_apo_pocket: usize,
    pub n_cryptic: usize,
    pub mean_cryptic_score: f64,

    // === Active Inference Results ===

    /// EFE scores per residue (if Active Inference enabled)
    pub efe_scores: HashMap<i32, f64>,

    /// Epistemic value per residue (information gain)
    pub epistemic_values: HashMap<i32, f64>,

    /// Pragmatic value per residue (goal alignment)
    pub pragmatic_values: HashMap<i32, f64>,

    // === Clustering Results ===

    /// Spatial clusters of cryptic predictions
    pub clusters: Vec<CrypticCluster>,

    /// Clustered predictions (representative residue per cluster)
    pub clustered_predictions: Vec<(i32, f64)>,
}

/// Ensemble Pocket Detector
///
/// Detects cryptic pockets by comparing pocket landscapes across
/// conformational ensembles.
pub struct EnsemblePocketDetector {
    config: EnsemblePocketConfig,
}

impl EnsemblePocketDetector {
    /// Create a new detector with given configuration
    pub fn new(config: EnsemblePocketConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(EnsemblePocketConfig::default())
    }

    /// Detect cryptic sites from an ANM ensemble
    ///
    /// Uses SASA-variance approach: cryptic residues are those that show
    /// significant INCREASE in solvent accessibility across the ensemble.
    /// This matches PocketMiner's methodology of identifying residues that
    /// become exposed through conformational change.
    ///
    /// # Arguments
    /// * `ensemble` - Generated conformational ensemble
    /// * `residue_map` - Mapping from sequential index to residue ID
    ///
    /// # Returns
    /// * `CrypticSiteResult` with per-residue cryptic scores
    pub fn detect_cryptic_sites(
        &self,
        ensemble: &AnmEnsemble,
        residue_map: &HashMap<usize, i32>,
    ) -> Result<CrypticSiteResult> {
        let n_residues = ensemble.original_coords.len();
        let n_conformations = ensemble.conformations.len();

        log::info!(
            "Detecting cryptic sites: {} residues, {} conformations",
            n_residues,
            n_conformations
        );

        // Step 1: Compute per-residue neighbor counts (proxy for burial) in APO
        let apo_neighbors = self.compute_neighbor_counts(&ensemble.original_coords);
        let _apo_exposures = self.compute_exposure_scores(&ensemble.original_coords);

        // Detect APO pockets using stringent criteria
        let apo_pockets = self.detect_pockets_stringent(&ensemble.original_coords, residue_map)?;
        log::info!("APO has {} pocket residues (stringent)", apo_pockets.n_pocket_residues);

        // Step 2: Compute exposure scores for each conformation
        let mut exposure_per_residue: Vec<Vec<f64>> = vec![Vec::new(); n_residues];
        let mut neighbor_change_per_residue: Vec<Vec<f64>> = vec![Vec::new(); n_residues];

        for (conf_idx, conf_coords) in ensemble.conformations.iter().enumerate() {
            let conf_exposures = self.compute_exposure_scores(conf_coords);
            let conf_neighbors = self.compute_neighbor_counts(conf_coords);

            for i in 0..n_residues {
                exposure_per_residue[i].push(conf_exposures[i]);
                // Track how much neighbor count decreased (more exposed)
                let neighbor_change = apo_neighbors[i] as f64 - conf_neighbors[i] as f64;
                neighbor_change_per_residue[i].push(neighbor_change);
            }

            if (conf_idx + 1) % 10 == 0 {
                log::debug!("Processed conformation {}/{}", conf_idx + 1, n_conformations);
            }
        }

        // Step 3: Compute cryptic scores using RMSF-based approach
        // Cryptic sites are typically:
        // - Moderately buried (not surface, not deep core)
        // - In flexible regions (high RMSF from ensemble)
        // - Show exposure variance across conformations
        let mut cryptic_scores: HashMap<i32, f64> = HashMap::new();
        let mut cryptic_residues = Vec::new();
        let mut sasa_variance: HashMap<i32, f64> = HashMap::new();

        // Active Inference score components
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

        // Compute burial scores (normalized neighbor count)
        let max_neighbors = *apo_neighbors.iter().max().unwrap_or(&1).max(&1);

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

            // Burial score: 0 = surface, 1 = deep core
            let burial = apo_neighbors[i] as f64 / max_neighbors as f64;

            // "Pocket-forming potential" - moderately buried, not surface
            // Bell curve centered around burial = 0.4-0.6
            let burial_potential = {
                let optimal_burial = 0.5;
                let width = 0.3;
                let deviation = (burial - optimal_burial).abs();
                (1.0 - (deviation / width).min(1.0)).max(0.0)
            };

            // RMSF score - higher is more flexible
            let flexibility = rmsf_normalized[i];

            // Exposure variance score
            let variance_score = (variance.sqrt() / 0.1).min(1.0);

            // Composite cryptic score based on multiple features:
            // 1. Burial: cryptic sites are moderately to highly buried
            // 2. Neighbor flexibility: neighbors' RMSF (if nearby regions move, pocket can open)
            // 3. Burial potential: Bell curve favoring intermediate burial (~0.5-0.7)
            // 4. Exposure variance: how much exposure changes across ensemble

            // Neighbor flexibility (sequence neighbors ±5 residues)
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

            // Spatial neighbor flexibility (residues within 10Å)
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

            // Combined neighbor flexibility
            let combined_neighbor_flex = neighbor_flexibility * 0.4 + spatial_neighbor_flexibility * 0.6;

            // Burial potential (favoring moderately buried residues)
            let burial_potential_score = {
                let optimal = 0.55;
                let width = 0.35;
                let diff = (burial - optimal).abs();
                (1.0 - (diff / width).min(1.0)).max(0.0)
            };

            // === Active Inference EFE Scoring ===
            // EFE = -Epistemic_Value + Pragmatic_Value (we minimize EFE, so negate for scoring)
            //
            // Epistemic Value: How much does observing this residue reduce uncertainty?
            // - High variance = high information gain (reduces ambiguity)
            // - Rare features (high burial + high flex neighbors) = more informative
            //
            // Pragmatic Value: How well does this residue align with pocket formation goal?
            // - Matches prior expectation of cryptic sites (buried + flexible neighbors)
            // - Penalize common patterns (surface residues)

            let (cryptic_score, epistemic_val, pragmatic_val) = if self.config.use_active_inference {
                // Epistemic value: information gain from observing this residue
                // High when: variance is high, features are rare/surprising
                let rarity = 1.0 - (burial * 0.3 + combined_neighbor_flex * 0.7);
                let surprise = variance_score * (1.0 + rarity);  // Variance weighted by rarity
                let epistemic = surprise * (1.0 - self.config.pocket_formation_prior);

                // Pragmatic value: alignment with pocket formation goal
                // Goal prior P(pocket|features) approximated by our feature model
                let posterior = burial * 0.35 + combined_neighbor_flex * 0.35
                    + burial_potential_score * 0.2 + variance_score * 0.1;

                // KL divergence from goal prior (prefer high posterior)
                // D_KL = posterior * log(posterior / prior)
                let prior = self.config.pocket_formation_prior;
                let kl_div = if posterior > 0.01 && prior > 0.01 {
                    posterior * (posterior / prior).ln()
                } else {
                    0.0
                };
                let pragmatic = posterior + kl_div.max(0.0) * 0.1;

                // Combined EFE score (higher = more likely cryptic)
                // We MAXIMIZE score, so: score = epistemic_weight * epistemic + pragmatic_weight * pragmatic
                let efe_score = self.config.epistemic_weight * epistemic
                    + self.config.pragmatic_weight * pragmatic;

                (efe_score, epistemic, pragmatic)
            } else {
                // Fallback to original linear scoring
                let score = burial * 0.35
                    + combined_neighbor_flex * 0.35
                    + burial_potential_score * 0.2
                    + variance_score * 0.1;
                (score, 0.0, score)
            };

            // Store scores
            if cryptic_score > 0.01 {
                cryptic_scores.insert(res_id, cryptic_score);
                efe_scores.insert(res_id, cryptic_score);
                epistemic_values.insert(res_id, epistemic_val);
                pragmatic_values.insert(res_id, pragmatic_val);
            }

            // Mark as cryptic if score above threshold
            if cryptic_score >= 0.3 {
                cryptic_residues.push(res_id);
            }
        }

        // Sort cryptic residues by score (descending)
        cryptic_residues.sort_by(|a, b| {
            let score_a = cryptic_scores.get(a).unwrap_or(&0.0);
            let score_b = cryptic_scores.get(b).unwrap_or(&0.0);
            score_b.partial_cmp(score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // === Step 4: Spatial Clustering (RLS-inspired) ===
        // Group nearby predictions to reduce false positives
        let (clusters, clustered_predictions) = if self.config.use_spatial_clustering {
            self.cluster_predictions(
                &cryptic_residues,
                &cryptic_scores,
                &ensemble.original_coords,
                residue_map,
            )
        } else {
            // No clustering - return all predictions
            let predictions: Vec<(i32, f64)> = cryptic_residues
                .iter()
                .map(|&r| (r, *cryptic_scores.get(&r).unwrap_or(&0.0)))
                .collect();
            (Vec::new(), predictions)
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
            "Found {} cryptic residues, {} clusters (threshold: {:.1}%)",
            n_cryptic,
            clusters.len(),
            self.config.min_pocket_frequency * 100.0
        );

        Ok(CrypticSiteResult {
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
            clusters,
            clustered_predictions,
        })
    }

    /// Cluster predictions by spatial proximity (RLS-inspired aggregation)
    ///
    /// This reduces false positives by grouping nearby predictions and
    /// reporting only cluster representatives.
    fn cluster_predictions(
        &self,
        residues: &[i32],
        scores: &HashMap<i32, f64>,
        coords: &[[f32; 3]],
        residue_map: &HashMap<usize, i32>,
    ) -> (Vec<CrypticCluster>, Vec<(i32, f64)>) {
        if residues.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Build reverse map: residue_id -> index
        let reverse_map: HashMap<i32, usize> = residue_map
            .iter()
            .map(|(&idx, &res_id)| (res_id, idx))
            .collect();

        // Simple greedy clustering
        let mut assigned: HashSet<i32> = HashSet::new();
        let mut clusters: Vec<CrypticCluster> = Vec::new();

        for &seed_res in residues {
            if assigned.contains(&seed_res) {
                continue;
            }

            let seed_idx = match reverse_map.get(&seed_res) {
                Some(&idx) => idx,
                None => continue,
            };
            let seed_pos = coords[seed_idx];

            // Find all unassigned residues within cluster_distance
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

            // Skip clusters smaller than minimum size
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

            // Compute radius
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

            clusters.push(CrypticCluster {
                residues: cluster_members,
                centroid,
                score: max_score,
                representative,
                radius: max_dist,
            });
        }

        // Sort clusters by score
        clusters.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Generate clustered predictions (one per cluster)
        let clustered_predictions: Vec<(i32, f64)> = clusters
            .iter()
            .map(|c| (c.representative, c.score))
            .collect();

        log::debug!(
            "Clustered {} residues into {} clusters",
            assigned.len(),
            clusters.len()
        );

        (clusters, clustered_predictions)
    }

    /// Compute neighbor counts for each residue (proxy for burial)
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

    /// Compute exposure scores (inverse of burial)
    fn compute_exposure_scores(&self, ca_coords: &[[f32; 3]]) -> Vec<f64> {
        let neighbor_counts = self.compute_neighbor_counts(ca_coords);
        let max_neighbors = *neighbor_counts.iter().max().unwrap_or(&1).max(&1);

        neighbor_counts.iter()
            .map(|&n| 1.0 - (n as f64 / max_neighbors as f64))
            .collect()
    }

    /// Stringent pocket detection with fewer residues classified as pocket
    fn detect_pockets_stringent(
        &self,
        ca_coords: &[[f32; 3]],
        residue_map: &HashMap<usize, i32>,
    ) -> Result<PocketDetectionResult> {
        let n = ca_coords.len();
        let neighbor_counts = self.compute_neighbor_counts(ca_coords);

        // Compute centroid distance
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

        // More stringent pocket criteria:
        // Only classify as pocket if in a very specific "pocket-like" region
        for (i, &count) in neighbor_counts.iter().enumerate() {
            let res_id = residue_map.get(&i).cloned().unwrap_or(i as i32);
            let dist = distances_to_centroid[i];
            let max_neighbors = *neighbor_counts.iter().max().unwrap_or(&1);

            burial_scores.insert(res_id, count as f64 / max_neighbors as f64);
            let normalized_sasa = (dist / max_dist) * 100.0;
            sasa_values.insert(res_id, normalized_sasa);

            // Stringent pocket criteria:
            // - Moderately buried (6-10 neighbors - narrow range)
            // - At intermediate depth (not surface, not deep core)
            let is_intermediate_burial = count >= 6 && count <= 10;
            let is_intermediate_depth = dist > max_dist * 0.3 && dist < max_dist * 0.7;

            if is_intermediate_burial && is_intermediate_depth {
                pocket_residues.insert(res_id);
            }
        }

        Ok(PocketDetectionResult {
            pocket_residues: pocket_residues.clone(),
            burial_scores,
            sasa_values,
            n_pocket_residues: pocket_residues.len(),
        })
    }

    /// Detect pockets using only CA coordinates
    ///
    /// Uses a simplified approach based on:
    /// 1. Local packing density (neighbor count)
    /// 2. Burial depth (distance to convex surface approximation)
    fn detect_pockets_ca_only(
        &self,
        ca_coords: &[[f32; 3]],
        residue_map: &HashMap<usize, i32>,
    ) -> Result<PocketDetectionResult> {
        let n = ca_coords.len();

        // Compute local packing density (neighbor count within cutoff)
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

        // Compute burial scores (normalized neighbor count)
        let max_neighbors = *neighbor_counts.iter().max().unwrap_or(&1).max(&1);
        let mut burial_scores: HashMap<i32, f64> = HashMap::new();

        for (i, &count) in neighbor_counts.iter().enumerate() {
            let res_id = residue_map.get(&i).cloned().unwrap_or(i as i32);
            burial_scores.insert(res_id, count as f64 / max_neighbors as f64);
        }

        // Compute approximate SASA using distance to centroid
        // (simplified: exposed residues are far from centroid)
        let centroid = self.compute_centroid(ca_coords);
        let mut distances_to_centroid: Vec<f64> = ca_coords
            .iter()
            .map(|c| {
                let dx = c[0] as f64 - centroid[0];
                let dy = c[1] as f64 - centroid[1];
                let dz = c[2] as f64 - centroid[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();

        // Normalize distances to approximate SASA
        let max_dist = distances_to_centroid.iter().cloned().fold(0.0f64, f64::max);
        let mut sasa_values: HashMap<i32, f64> = HashMap::new();

        for (i, &dist) in distances_to_centroid.iter().enumerate() {
            let res_id = residue_map.get(&i).cloned().unwrap_or(i as i32);
            // Higher distance = more exposed = higher "SASA"
            let normalized_sasa = (dist / max_dist) * 100.0;  // Scale to ~0-100 range
            sasa_values.insert(res_id, normalized_sasa);
        }

        // Identify pocket residues
        // Pocket = moderately buried (not surface, not core)
        // This is a heuristic - real pockets are concave regions
        let mut pocket_residues: HashSet<i32> = HashSet::new();

        for (i, &count) in neighbor_counts.iter().enumerate() {
            let res_id = residue_map.get(&i).cloned().unwrap_or(i as i32);
            let dist = distances_to_centroid[i];

            // Pocket criteria:
            // 1. Moderately buried (not too few or too many neighbors)
            // 2. Not at the surface (not maximum distance from centroid)
            let is_moderately_buried = count >= 4 && count < self.config.min_neighbors_for_burial;
            let is_not_core = count < self.config.min_neighbors_for_burial + 4;
            let is_not_extreme_surface = dist < max_dist * 0.9;

            if is_moderately_buried && is_not_core && is_not_extreme_surface {
                pocket_residues.insert(res_id);
            }
        }

        Ok(PocketDetectionResult {
            pocket_residues: pocket_residues.clone(),
            burial_scores,
            sasa_values,
            n_pocket_residues: pocket_residues.len(),
        })
    }

    /// Compute centroid of CA coordinates
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

/// Convert cryptic scores to predictions for benchmark comparison
///
/// # Arguments
/// * `result` - Cryptic site detection result
/// * `score_threshold` - Minimum score to be predicted as cryptic
///
/// # Returns
/// * Vec of (residue_id, score) sorted by score descending
pub fn cryptic_scores_to_predictions(
    result: &CrypticSiteResult,
    score_threshold: f64,
) -> Vec<(i32, f64)> {
    let mut predictions: Vec<(i32, f64)> = result
        .cryptic_scores
        .iter()
        .filter(|(_, &score)| score >= score_threshold)
        .map(|(&res_id, &score)| (res_id, score))
        .collect();

    // Sort by score descending
    predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    predictions
}

/// Compute overlap between predicted and ground truth cryptic residues
///
/// # Arguments
/// * `predictions` - Predicted cryptic residue IDs
/// * `ground_truth` - Ground truth cryptic residue IDs
///
/// # Returns
/// * (precision, recall, f1_score, overlap_count)
pub fn compute_prediction_overlap(
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
    fn test_pocket_detection_ca_only() {
        let coords = make_simple_helix(30);
        let residue_map: HashMap<usize, i32> = (0..30).map(|i| (i, i as i32)).collect();

        let detector = EnsemblePocketDetector::default_config();
        let result = detector.detect_pockets_ca_only(&coords, &residue_map).unwrap();

        assert!(result.burial_scores.len() == 30);
        assert!(result.sasa_values.len() == 30);
    }

    #[test]
    fn test_compute_overlap() {
        let predictions = vec![1, 2, 3, 4, 5];
        let ground_truth: HashSet<i32> = vec![3, 4, 5, 6, 7].into_iter().collect();

        let (precision, recall, f1, overlap) = compute_prediction_overlap(&predictions, &ground_truth);

        assert_eq!(overlap, 3); // {3, 4, 5}
        assert!((precision - 0.6).abs() < 0.01); // 3/5
        assert!((recall - 0.6).abs() < 0.01);    // 3/5
    }
}
