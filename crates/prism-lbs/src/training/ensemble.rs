//! Ensemble prediction with multi-method voting
//!
//! Combines predictions from multiple pocket detection approaches
//! using configurable voting strategies.

use crate::{LbsConfig, LbsError, Pocket, PrismLbs, ProteinStructure};
use crate::scoring::{DruggabilityScore, ScoringWeights};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Voting method for ensemble predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingMethod {
    /// Simple majority voting
    Majority,
    /// Weighted voting by method confidence
    Weighted,
    /// Union of all predictions with score averaging
    Union,
    /// Intersection (only pockets found by all methods)
    Intersection,
    /// Rank-based voting (Borda count)
    RankBased,
}

impl Default for VotingMethod {
    fn default() -> Self {
        VotingMethod::Weighted
    }
}

/// Configuration for a single prediction method in the ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodConfig {
    /// Method name/identifier
    pub name: String,
    /// Weight in ensemble (0.0-1.0)
    pub weight: f64,
    /// LBS configuration for this method
    pub lbs_config: LbsConfig,
    /// Enable/disable this method
    pub enabled: bool,
}

/// Ensemble predictor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Voting method
    pub voting_method: VotingMethod,
    /// Methods in the ensemble
    pub methods: Vec<MethodConfig>,
    /// Distance threshold for pocket matching (Å)
    pub match_distance: f64,
    /// Minimum votes required (for majority voting)
    pub min_votes: usize,
    /// Maximum pockets to return
    pub top_n: usize,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            voting_method: VotingMethod::Weighted,
            methods: Self::default_methods(),
            match_distance: 4.0,
            min_votes: 2,
            top_n: 10,
        }
    }
}

impl EnsembleConfig {
    /// Create default ensemble with 3 complementary methods
    fn default_methods() -> Vec<MethodConfig> {
        vec![
            // Method 1: Default PRISM-LBS (balanced)
            MethodConfig {
                name: "prism_balanced".to_string(),
                weight: 0.4,
                lbs_config: LbsConfig::default(),
                enabled: true,
            },
            // Method 2: Geometry-focused (higher enclosure weight)
            MethodConfig {
                name: "prism_geometry".to_string(),
                weight: 0.35,
                lbs_config: {
                    let mut cfg = LbsConfig::default();
                    cfg.scoring = ScoringWeights {
                        volume: 0.25,
                        hydrophobicity: 0.10,
                        enclosure: 0.25,
                        depth: 0.20,
                        hbond_capacity: 0.05,
                        flexibility: 0.05,
                        conservation: 0.05,
                        topology: 0.05,
                    };
                    cfg
                },
                enabled: true,
            },
            // Method 3: Chemistry-focused (higher hydrophobicity/H-bond weight)
            MethodConfig {
                name: "prism_chemistry".to_string(),
                weight: 0.25,
                lbs_config: {
                    let mut cfg = LbsConfig::default();
                    cfg.scoring = ScoringWeights {
                        volume: 0.10,
                        hydrophobicity: 0.30,
                        enclosure: 0.10,
                        depth: 0.10,
                        hbond_capacity: 0.20,
                        flexibility: 0.05,
                        conservation: 0.10,
                        topology: 0.05,
                    };
                    cfg
                },
                enabled: true,
            },
        ]
    }
}

/// Ensemble pocket prediction result
#[derive(Debug, Clone)]
pub struct EnsemblePocket {
    /// Merged pocket data
    pub pocket: Pocket,
    /// Votes from each method
    pub votes: HashMap<String, f64>,
    /// Number of methods that found this pocket
    pub num_votes: usize,
    /// Ensemble confidence score
    pub confidence: f64,
    /// Consensus druggability score
    pub consensus_score: f64,
}

/// Ensemble predictor combining multiple methods
pub struct EnsemblePredictor {
    config: EnsembleConfig,
    predictors: Vec<(String, PrismLbs)>,
}

impl EnsemblePredictor {
    /// Create new ensemble predictor
    pub fn new(config: EnsembleConfig) -> Result<Self, LbsError> {
        let mut predictors = Vec::new();

        for method in &config.methods {
            if method.enabled {
                let lbs = PrismLbs::new(method.lbs_config.clone())?;
                predictors.push((method.name.clone(), lbs));
            }
        }

        if predictors.is_empty() {
            return Err(LbsError::Config("No enabled methods in ensemble".to_string()));
        }

        Ok(Self { config, predictors })
    }

    /// Create with default configuration
    pub fn default_ensemble() -> Result<Self, LbsError> {
        Self::new(EnsembleConfig::default())
    }

    /// Run ensemble prediction
    pub fn predict(&self, structure: &ProteinStructure) -> Result<Vec<EnsemblePocket>, LbsError> {
        // Collect predictions from all methods
        let mut all_predictions: Vec<(String, Vec<Pocket>)> = Vec::new();

        for (name, predictor) in &self.predictors {
            match predictor.predict(structure) {
                Ok(pockets) => {
                    log::debug!("Method '{}' found {} pockets", name, pockets.len());
                    all_predictions.push((name.clone(), pockets));
                }
                Err(e) => {
                    log::warn!("Method '{}' failed: {}", name, e);
                }
            }
        }

        if all_predictions.is_empty() {
            return Ok(Vec::new());
        }

        // Combine predictions based on voting method
        let ensemble_pockets = match self.config.voting_method {
            VotingMethod::Majority => self.majority_voting(&all_predictions),
            VotingMethod::Weighted => self.weighted_voting(&all_predictions),
            VotingMethod::Union => self.union_voting(&all_predictions),
            VotingMethod::Intersection => self.intersection_voting(&all_predictions),
            VotingMethod::RankBased => self.rank_based_voting(&all_predictions),
        };

        // Sort by consensus score and return top N
        let mut results = ensemble_pockets;
        results.sort_by(|a, b| b.consensus_score.partial_cmp(&a.consensus_score).unwrap());
        results.truncate(self.config.top_n);

        Ok(results)
    }

    /// Majority voting: pockets found by >= min_votes methods
    fn majority_voting(&self, predictions: &[(String, Vec<Pocket>)]) -> Vec<EnsemblePocket> {
        let mut pocket_clusters = self.cluster_pockets(predictions);

        pocket_clusters
            .into_iter()
            .filter(|(_, pockets)| pockets.len() >= self.config.min_votes)
            .map(|(centroid, pockets)| self.merge_pocket_cluster(&centroid, &pockets, predictions))
            .collect()
    }

    /// Weighted voting: score pockets by method weights
    fn weighted_voting(&self, predictions: &[(String, Vec<Pocket>)]) -> Vec<EnsemblePocket> {
        let pocket_clusters = self.cluster_pockets(predictions);

        pocket_clusters
            .into_iter()
            .map(|(centroid, pockets)| self.merge_pocket_cluster(&centroid, &pockets, predictions))
            .filter(|ep| ep.confidence > 0.3)  // Minimum confidence threshold
            .collect()
    }

    /// Union voting: include all unique pockets
    fn union_voting(&self, predictions: &[(String, Vec<Pocket>)]) -> Vec<EnsemblePocket> {
        let pocket_clusters = self.cluster_pockets(predictions);

        pocket_clusters
            .into_iter()
            .map(|(centroid, pockets)| self.merge_pocket_cluster(&centroid, &pockets, predictions))
            .collect()
    }

    /// Intersection voting: only pockets found by ALL methods
    fn intersection_voting(&self, predictions: &[(String, Vec<Pocket>)]) -> Vec<EnsemblePocket> {
        let num_methods = predictions.len();
        let pocket_clusters = self.cluster_pockets(predictions);

        pocket_clusters
            .into_iter()
            .filter(|(_, pockets)| {
                // Check if all methods contributed
                let methods: std::collections::HashSet<&str> = pockets
                    .iter()
                    .map(|(m, _)| m.as_str())
                    .collect();
                methods.len() >= num_methods
            })
            .map(|(centroid, pockets)| self.merge_pocket_cluster(&centroid, &pockets, predictions))
            .collect()
    }

    /// Rank-based voting (Borda count)
    fn rank_based_voting(&self, predictions: &[(String, Vec<Pocket>)]) -> Vec<EnsemblePocket> {
        let pocket_clusters = self.cluster_pockets(predictions);

        // Assign ranks within each method
        let mut rank_scores: HashMap<[i64; 3], f64> = HashMap::new();

        for (method_name, pockets) in predictions {
            let method_weight = self.get_method_weight(method_name);
            let n = pockets.len();

            for (rank, pocket) in pockets.iter().enumerate() {
                let key = self.centroid_key(&pocket.centroid);
                // Borda score: (n - rank) / n
                let borda = (n - rank) as f64 / n as f64 * method_weight;
                *rank_scores.entry(key).or_insert(0.0) += borda;
            }
        }

        pocket_clusters
            .into_iter()
            .map(|(centroid, pockets)| {
                let mut ep = self.merge_pocket_cluster(&centroid, &pockets, predictions);
                let key = self.centroid_key(&centroid);
                ep.consensus_score = rank_scores.get(&key).copied().unwrap_or(0.0);
                ep
            })
            .collect()
    }

    /// Cluster nearby pockets from different methods
    fn cluster_pockets(&self, predictions: &[(String, Vec<Pocket>)]) -> Vec<([f64; 3], Vec<(String, Pocket)>)> {
        let mut clusters: HashMap<[i64; 3], Vec<(String, Pocket)>> = HashMap::new();

        for (method_name, pockets) in predictions {
            for pocket in pockets {
                let key = self.centroid_key(&pocket.centroid);

                // Find if there's an existing cluster nearby
                let mut found_cluster = None;
                for existing_key in clusters.keys() {
                    let existing_centroid = self.key_to_centroid(existing_key);
                    if self.distance(&pocket.centroid, &existing_centroid) < self.config.match_distance {
                        found_cluster = Some(*existing_key);
                        break;
                    }
                }

                let cluster_key = found_cluster.unwrap_or(key);
                clusters
                    .entry(cluster_key)
                    .or_default()
                    .push((method_name.clone(), pocket.clone()));
            }
        }

        // Convert keys back to f64 centroids
        clusters
            .into_iter()
            .map(|(key, pockets)| (self.key_to_centroid(&key), pockets))
            .collect()
    }

    /// Merge a cluster of pockets into ensemble result
    fn merge_pocket_cluster(
        &self,
        _centroid: &[f64; 3],
        pockets: &[(String, Pocket)],
        predictions: &[(String, Vec<Pocket>)],
    ) -> EnsemblePocket {
        if pockets.is_empty() {
            return EnsemblePocket {
                pocket: Pocket::default(),
                votes: HashMap::new(),
                num_votes: 0,
                confidence: 0.0,
                consensus_score: 0.0,
            };
        }

        // Compute weighted average of properties
        let mut total_weight = 0.0;
        let mut votes = HashMap::new();

        let mut sum_volume = 0.0;
        let mut sum_hydro = 0.0;
        let mut sum_enclosure = 0.0;
        let mut sum_depth = 0.0;
        let mut sum_score = 0.0;
        let mut centroid = [0.0, 0.0, 0.0];
        let mut all_atoms: Vec<usize> = Vec::new();
        let mut all_residues: Vec<usize> = Vec::new();

        for (method_name, pocket) in pockets {
            let weight = self.get_method_weight(method_name);
            votes.insert(method_name.clone(), pocket.druggability_score.total * weight);

            sum_volume += pocket.volume * weight;
            sum_hydro += pocket.mean_hydrophobicity * weight;
            sum_enclosure += pocket.enclosure_ratio * weight;
            sum_depth += pocket.mean_depth * weight;
            sum_score += pocket.druggability_score.total * weight;

            centroid[0] += pocket.centroid[0] * weight;
            centroid[1] += pocket.centroid[1] * weight;
            centroid[2] += pocket.centroid[2] * weight;

            all_atoms.extend(&pocket.atom_indices);
            all_residues.extend(&pocket.residue_indices);

            total_weight += weight;
        }

        if total_weight > 0.0 {
            centroid[0] /= total_weight;
            centroid[1] /= total_weight;
            centroid[2] /= total_weight;
        }

        // Deduplicate atoms/residues
        all_atoms.sort_unstable();
        all_atoms.dedup();
        all_residues.sort_unstable();
        all_residues.dedup();

        // Compute confidence based on agreement
        let num_methods = predictions.len();
        let num_votes = pockets.len();
        let confidence = num_votes as f64 / num_methods as f64;

        let merged_pocket = Pocket {
            atom_indices: all_atoms,
            residue_indices: all_residues,
            centroid,
            volume: if total_weight > 0.0 { sum_volume / total_weight } else { 0.0 },
            enclosure_ratio: if total_weight > 0.0 { sum_enclosure / total_weight } else { 0.0 },
            mean_hydrophobicity: if total_weight > 0.0 { sum_hydro / total_weight } else { 0.0 },
            mean_depth: if total_weight > 0.0 { sum_depth / total_weight } else { 0.0 },
            mean_sasa: pockets.first().map(|(_, p)| p.mean_sasa).unwrap_or(0.0),
            mean_flexibility: pockets.first().map(|(_, p)| p.mean_flexibility).unwrap_or(0.0),
            mean_conservation: pockets.first().map(|(_, p)| p.mean_conservation).unwrap_or(0.0),
            persistence_score: pockets.first().map(|(_, p)| p.persistence_score).unwrap_or(0.0),
            hbond_donors: pockets.iter().map(|(_, p)| p.hbond_donors).max().unwrap_or(0),
            hbond_acceptors: pockets.iter().map(|(_, p)| p.hbond_acceptors).max().unwrap_or(0),
            druggability_score: DruggabilityScore::default(),
            boundary_atoms: Vec::new(),
            mean_electrostatic: 0.0,
            gnn_embedding: Vec::new(),
            gnn_druggability: 0.0,
        };

        let consensus_score = if total_weight > 0.0 {
            (sum_score / total_weight) * confidence
        } else {
            0.0
        };

        EnsemblePocket {
            pocket: merged_pocket,
            votes,
            num_votes,
            confidence,
            consensus_score,
        }
    }

    /// Get method weight from config
    fn get_method_weight(&self, method_name: &str) -> f64 {
        self.config.methods
            .iter()
            .find(|m| m.name == method_name)
            .map(|m| m.weight)
            .unwrap_or(1.0 / self.predictors.len() as f64)
    }

    /// Convert centroid to integer key for clustering
    fn centroid_key(&self, centroid: &[f64; 3]) -> [i64; 3] {
        [
            (centroid[0] / self.config.match_distance * 2.0).round() as i64,
            (centroid[1] / self.config.match_distance * 2.0).round() as i64,
            (centroid[2] / self.config.match_distance * 2.0).round() as i64,
        ]
    }

    /// Convert key back to approximate centroid
    fn key_to_centroid(&self, key: &[i64; 3]) -> [f64; 3] {
        [
            key[0] as f64 * self.config.match_distance / 2.0,
            key[1] as f64 * self.config.match_distance / 2.0,
            key[2] as f64 * self.config.match_distance / 2.0,
        ]
    }

    /// Euclidean distance between two centroids
    fn distance(&self, a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Get enabled method names
    pub fn method_names(&self) -> Vec<&str> {
        self.predictors.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Get number of methods in ensemble
    pub fn num_methods(&self) -> usize {
        self.predictors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_ensemble_config() {
        let config = EnsembleConfig::default();
        assert_eq!(config.methods.len(), 3);
        assert!(config.methods.iter().all(|m| m.enabled));

        // Weights should sum to ~1.0
        let total_weight: f64 = config.methods.iter().map(|m| m.weight).sum();
        assert!((total_weight - 1.0).abs() < 0.01);
    }
}
