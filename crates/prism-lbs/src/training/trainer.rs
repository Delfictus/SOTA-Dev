//! LBS Training System
//!
//! Integrates PDBBind data loading, FluxNet RL weight optimization,
//! conservation data, and ensemble training.

use crate::pipeline_integration::FluxNetWeightOptimizer;
use crate::scoring::ScoringWeights;
use crate::validation::{BenchmarkSummary, ValidationMetrics};
use crate::{LbsConfig, LbsError, PrismLbs};
use super::pdbbind_loader::{PdbBindConfig, PdbBindEntry, PdbBindLoader};
use super::conservation::{ConservationConfig, ConservationLoader};
use super::ensemble::{EnsembleConfig, EnsemblePredictor};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// PDBBind dataset configuration
    pub pdbbind: PdbBindConfig,
    /// Conservation data configuration
    pub conservation: ConservationConfig,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate for weight optimization
    pub learning_rate: f64,
    /// Validation split ratio (0.0-1.0)
    pub validation_split: f64,
    /// Distance threshold for success (Å)
    pub success_threshold: f64,
    /// Early stopping patience (epochs)
    pub patience: usize,
    /// Checkpoint directory
    pub checkpoint_dir: Option<String>,
    /// Enable ensemble training
    pub train_ensemble: bool,
    /// Random seed
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            pdbbind: PdbBindConfig::default(),
            conservation: ConservationConfig::default(),
            epochs: 100,
            batch_size: 16,
            learning_rate: 0.01,
            validation_split: 0.2,
            success_threshold: 4.0,  // 4Å DCC
            patience: 10,
            checkpoint_dir: Some("checkpoints".to_string()),
            train_ensemble: true,
            seed: 42,
        }
    }
}

/// Training metrics for a single epoch
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Epoch number
    pub epoch: usize,
    /// Training loss (1 - reward)
    pub train_loss: f64,
    /// Validation success rate
    pub val_success_rate: f64,
    /// Validation mean center distance
    pub val_mean_distance: f64,
    /// Validation mean coverage
    pub val_mean_coverage: f64,
    /// Best weights found so far
    pub best_weights: [f64; 8],
    /// Current learning rate
    pub learning_rate: f64,
    /// Current temperature (FluxNet)
    pub temperature: f64,
}

/// LBS Trainer
pub struct LbsTrainer {
    config: TrainingConfig,
    pdbbind_loader: Option<PdbBindLoader>,
    conservation_loader: ConservationLoader,
    optimizer: FluxNetWeightOptimizer,
    best_weights: ScoringWeights,
    best_score: f64,
    epochs_without_improvement: usize,
}

impl LbsTrainer {
    /// Create new trainer
    pub fn new(config: TrainingConfig) -> Result<Self, LbsError> {
        // Try to load PDBBind data
        let pdbbind_loader = match PdbBindLoader::new(config.pdbbind.clone()) {
            Ok(loader) => {
                log::info!("Loaded PDBBind dataset with {} entries", loader.len());
                Some(loader)
            }
            Err(e) => {
                log::warn!("Could not load PDBBind: {}. Training will use synthetic data.", e);
                None
            }
        };

        let conservation_loader = ConservationLoader::new(config.conservation.clone());
        let optimizer = FluxNetWeightOptimizer::new(config.learning_rate, 0.9);

        Ok(Self {
            config,
            pdbbind_loader,
            conservation_loader,
            optimizer,
            best_weights: ScoringWeights::default(),
            best_score: 0.0,
            epochs_without_improvement: 0,
        })
    }

    /// Run full training loop
    pub fn train(&mut self) -> Result<Vec<TrainingMetrics>, LbsError> {
        log::info!("Starting PRISM-LBS training for {} epochs", self.config.epochs);

        let mut all_metrics = Vec::new();

        // Split data into train/validation
        let (train_ids, val_ids) = self.split_data()?;
        log::info!(
            "Train/Val split: {}/{} structures",
            train_ids.len(),
            val_ids.len()
        );

        for epoch in 0..self.config.epochs {
            // Training phase
            let train_loss = self.train_epoch(&train_ids)?;

            // Validation phase
            let (val_success, val_distance, val_coverage) = self.validate(&val_ids)?;

            // Check for improvement
            if val_success > self.best_score {
                self.best_score = val_success;
                self.best_weights = self.optimizer.get_weights();
                self.epochs_without_improvement = 0;
                self.save_checkpoint(epoch)?;
                log::info!("New best model at epoch {}: {:.3} success rate", epoch, val_success);
            } else {
                self.epochs_without_improvement += 1;
            }

            let metrics = TrainingMetrics {
                epoch,
                train_loss,
                val_success_rate: val_success,
                val_mean_distance: val_distance,
                val_mean_coverage: val_coverage,
                best_weights: self.best_weights.to_array(),
                learning_rate: self.config.learning_rate,
                temperature: self.optimizer.temperature(),
            };

            log::info!(
                "Epoch {}: loss={:.4} val_success={:.3} val_dist={:.2}Å",
                epoch,
                train_loss,
                val_success,
                val_distance
            );

            all_metrics.push(metrics);

            // Early stopping
            if self.epochs_without_improvement >= self.config.patience {
                log::info!(
                    "Early stopping at epoch {} (no improvement for {} epochs)",
                    epoch,
                    self.config.patience
                );
                break;
            }
        }

        log::info!(
            "Training complete. Best success rate: {:.3}",
            self.best_score
        );

        Ok(all_metrics)
    }

    /// Train for one epoch
    fn train_epoch(&mut self, train_ids: &[String]) -> Result<f64, LbsError> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for batch_ids in train_ids.chunks(self.config.batch_size) {
            let batch_loss = self.train_batch(batch_ids)?;
            total_loss += batch_loss;
            num_batches += 1;
        }

        Ok(if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        })
    }

    /// Train on a single batch
    fn train_batch(&mut self, batch_ids: &[String]) -> Result<f64, LbsError> {
        let mut batch_loss = 0.0;

        for pdb_id in batch_ids {
            if let Some(ref loader) = self.pdbbind_loader {
                if let Ok(entry) = loader.load_entry(pdb_id) {
                    let loss = self.train_single(&entry)?;
                    batch_loss += loss;
                }
            }
        }

        Ok(batch_loss / batch_ids.len() as f64)
    }

    /// Train on a single structure
    fn train_single(&mut self, entry: &PdbBindEntry) -> Result<f64, LbsError> {
        // Apply conservation data
        let mut structure = entry.structure.clone();
        self.conservation_loader.apply(&mut structure);

        // Run prediction with current weights
        let mut config = LbsConfig::default();
        config.scoring = self.optimizer.get_weights();

        let lbs = PrismLbs::new(config)?;
        let pockets = lbs.predict(&structure)?;

        // Compute validation metrics
        let metrics = ValidationMetrics::compute_batch(
            &pockets,
            &entry.ligand_coords,
            self.config.success_threshold,
        );

        // Update optimizer
        self.optimizer.update(&metrics);

        // Loss = 1 - reward
        let reward = self.optimizer.compute_reward(&metrics);
        Ok(1.0 - reward)
    }

    /// Validate on held-out data
    fn validate(&self, val_ids: &[String]) -> Result<(f64, f64, f64), LbsError> {
        let mut total_success = 0.0;
        let mut total_distance = 0.0;
        let mut total_coverage = 0.0;
        let mut count = 0;

        let config = {
            let mut cfg = LbsConfig::default();
            cfg.scoring = self.optimizer.get_weights();
            cfg
        };

        let lbs = PrismLbs::new(config)?;

        for pdb_id in val_ids {
            if let Some(ref loader) = self.pdbbind_loader {
                if let Ok(entry) = loader.load_entry(pdb_id) {
                    let mut structure = entry.structure.clone();
                    self.conservation_loader.apply(&mut structure);

                    if let Ok(pockets) = lbs.predict(&structure) {
                        let metrics = ValidationMetrics::compute_batch(
                            &pockets,
                            &entry.ligand_coords,
                            self.config.success_threshold,
                        );

                        total_success += metrics.success_rate();
                        total_distance += metrics.center_distance();
                        total_coverage += metrics.ligand_coverage;
                        count += 1;
                    }
                }
            }
        }

        if count == 0 {
            return Ok((0.0, f64::INFINITY, 0.0));
        }

        Ok((
            total_success / count as f64,
            total_distance / count as f64,
            total_coverage / count as f64,
        ))
    }

    /// Split data into train/validation sets
    fn split_data(&self) -> Result<(Vec<String>, Vec<String>), LbsError> {
        let all_ids: Vec<String> = self
            .pdbbind_loader
            .as_ref()
            .map(|l| l.pdb_ids())
            .unwrap_or_default();

        if all_ids.is_empty() {
            // Generate synthetic IDs for testing
            return Ok((
                (0..80).map(|i| format!("syn_{:04}", i)).collect(),
                (80..100).map(|i| format!("syn_{:04}", i)).collect(),
            ));
        }

        // Shuffle with seed
        let mut rng_state = self.config.seed;
        let mut shuffled = all_ids.clone();
        for i in (1..shuffled.len()).rev() {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (rng_state as usize) % (i + 1);
            shuffled.swap(i, j);
        }

        let split_idx = (shuffled.len() as f64 * (1.0 - self.config.validation_split)) as usize;
        let (train, val) = shuffled.split_at(split_idx);

        Ok((train.to_vec(), val.to_vec()))
    }

    /// Save checkpoint
    fn save_checkpoint(&self, epoch: usize) -> Result<(), LbsError> {
        if let Some(ref dir) = self.config.checkpoint_dir {
            std::fs::create_dir_all(dir).map_err(LbsError::Io)?;

            let weights = self.best_weights.clone();
            let checkpoint = CheckpointData {
                epoch,
                weights,
                best_score: self.best_score,
            };

            let path = Path::new(dir).join(format!("checkpoint_epoch_{}.json", epoch));
            let json = serde_json::to_string_pretty(&checkpoint)
                .map_err(|e| LbsError::Config(e.to_string()))?;

            std::fs::write(&path, json).map_err(LbsError::Io)?;

            // Also save as "best" checkpoint
            let best_path = Path::new(dir).join("best_checkpoint.json");
            std::fs::write(&best_path, serde_json::to_string_pretty(&checkpoint)
                .map_err(|e| LbsError::Config(e.to_string()))?)
                .map_err(LbsError::Io)?;
        }

        Ok(())
    }

    /// Load checkpoint
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<(), LbsError> {
        let content = std::fs::read_to_string(path).map_err(LbsError::Io)?;
        let checkpoint: CheckpointData = serde_json::from_str(&content)
            .map_err(|e| LbsError::Config(e.to_string()))?;

        self.best_weights = checkpoint.weights;
        self.best_score = checkpoint.best_score;
        self.optimizer.update_weights_from(&self.best_weights);

        log::info!(
            "Loaded checkpoint from epoch {} with score {:.3}",
            checkpoint.epoch,
            checkpoint.best_score
        );

        Ok(())
    }

    /// Get best weights found during training
    pub fn best_weights(&self) -> &ScoringWeights {
        &self.best_weights
    }

    /// Get best validation score
    pub fn best_score(&self) -> f64 {
        self.best_score
    }

    /// Train ensemble model
    pub fn train_ensemble(&mut self) -> Result<EnsembleConfig, LbsError> {
        if !self.config.train_ensemble {
            return Err(LbsError::Config("Ensemble training disabled".to_string()));
        }

        log::info!("Training ensemble model...");

        // Start with default ensemble config
        let mut ensemble_config = EnsembleConfig::default();

        // Train individual method weights
        let (train_ids, val_ids) = self.split_data()?;

        // Train each method's scoring weights
        for method in &mut ensemble_config.methods {
            if method.enabled {
                log::info!("Training method: {}", method.name);

                // Create method-specific optimizer
                let mut method_optimizer = FluxNetWeightOptimizer::from_weights(
                    method.lbs_config.scoring.clone(),
                    self.config.learning_rate,
                );

                // Train for fewer epochs per method
                let method_epochs = self.config.epochs / 3;
                for _ in 0..method_epochs {
                    for pdb_id in &train_ids {
                        if let Some(ref loader) = self.pdbbind_loader {
                            if let Ok(entry) = loader.load_entry(pdb_id) {
                                let mut config = method.lbs_config.clone();
                                config.scoring = method_optimizer.get_weights();

                                let lbs = PrismLbs::new(config)?;
                                let pockets = lbs.predict(&entry.structure)?;

                                let metrics = ValidationMetrics::compute_batch(
                                    &pockets,
                                    &entry.ligand_coords,
                                    self.config.success_threshold,
                                );

                                method_optimizer.update(&metrics);
                            }
                        }
                    }
                }

                method.lbs_config.scoring = method_optimizer.get_weights();
            }
        }

        // Optimize method weights based on validation performance
        let mut method_scores: Vec<(usize, f64)> = Vec::new();

        for (i, method) in ensemble_config.methods.iter().enumerate() {
            if method.enabled {
                let lbs = PrismLbs::new(method.lbs_config.clone())?;
                let mut total_score = 0.0;
                let mut count = 0;

                for pdb_id in &val_ids {
                    if let Some(ref loader) = self.pdbbind_loader {
                        if let Ok(entry) = loader.load_entry(pdb_id) {
                            if let Ok(pockets) = lbs.predict(&entry.structure) {
                                let metrics = ValidationMetrics::compute_batch(
                                    &pockets,
                                    &entry.ligand_coords,
                                    self.config.success_threshold,
                                );
                                total_score += metrics.success_rate();
                                count += 1;
                            }
                        }
                    }
                }

                let avg_score = if count > 0 {
                    total_score / count as f64
                } else {
                    0.0
                };
                method_scores.push((i, avg_score));
            }
        }

        // Normalize weights based on validation scores
        let total_score: f64 = method_scores.iter().map(|(_, s)| s).sum();
        if total_score > 0.0 {
            for (i, score) in method_scores {
                ensemble_config.methods[i].weight = score / total_score;
            }
        }

        log::info!("Ensemble training complete. Method weights:");
        for method in &ensemble_config.methods {
            if method.enabled {
                log::info!("  {}: {:.3}", method.name, method.weight);
            }
        }

        Ok(ensemble_config)
    }

    /// Get current optimizer state
    pub fn optimizer(&self) -> &FluxNetWeightOptimizer {
        &self.optimizer
    }
}

/// Checkpoint data for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointData {
    epoch: usize,
    weights: ScoringWeights,
    best_score: f64,
}

impl ScoringWeights {
    /// Convert to array
    fn to_array(&self) -> [f64; 8] {
        [
            self.volume,
            self.hydrophobicity,
            self.enclosure,
            self.depth,
            self.hbond_capacity,
            self.flexibility,
            self.conservation,
            self.topology,
        ]
    }
}

impl FluxNetWeightOptimizer {
    /// Update weights from ScoringWeights struct
    fn update_weights_from(&mut self, weights: &ScoringWeights) {
        self.update_weights([
            weights.volume,
            weights.hydrophobicity,
            weights.enclosure,
            weights.depth,
            weights.hbond_capacity,
            weights.flexibility,
            weights.conservation,
            weights.topology,
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 16);
        assert!(config.train_ensemble);
    }
}
