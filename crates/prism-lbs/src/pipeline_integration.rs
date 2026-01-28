//! Minimal adapter hooks so prism-pipeline can call PRISM-LBS without tight coupling.
//!
//! The pipeline crate can import `prism_lbs::pipeline_integration::run_lbs` to execute
//! predictions inside existing orchestration.
//!
//! FluxNet RL integration enables learning optimal druggability scoring weights
//! from validation feedback using adaptive weight optimization.

use crate::{LbsConfig, Pocket, PrismLbs, ProteinStructure};
use crate::scoring::{DruggabilityScorer, ScoringWeights};
use crate::validation::ValidationMetrics;
use anyhow::Result;
#[cfg(feature = "cuda")]
use prism_gpu::context::GpuContext;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Execute LBS prediction for a structure and return pockets.
pub fn run_lbs(structure: &ProteinStructure, config: &LbsConfig) -> Result<Vec<Pocket>> {
    #[cfg(feature = "cuda")]
    {
        let lbs = PrismLbs::new_with_gpu(config.clone(), None)?;
        return lbs.predict(structure);
    }

    #[cfg(not(feature = "cuda"))]
    {
        let lbs = PrismLbs::new(config.clone())?;
        lbs.predict(structure)
    }
}

/// Execute LBS prediction using an existing GPU context (preferred in orchestrated runs).
#[cfg(feature = "cuda")]
pub fn run_lbs_with_gpu(
    structure: &ProteinStructure,
    config: &LbsConfig,
    gpu_ctx: Option<Arc<GpuContext>>,
) -> Result<Vec<Pocket>> {
    let lbs = PrismLbs::new_with_gpu(config.clone(), gpu_ctx)?;
    lbs.predict(structure)
}

/// FluxNet RL-based weight optimizer for druggability scoring
///
/// Uses policy gradient with baseline to learn optimal weights from
/// validation feedback (center distance, ligand coverage, precision).
pub struct FluxNetWeightOptimizer {
    weights: [f64; 8],
    learning_rate: f64,
    baseline: f64,
    momentum: [f64; 8],
    temperature: f64,
    episode_count: usize,
}

impl Default for FluxNetWeightOptimizer {
    fn default() -> Self {
        Self::new(0.01, 0.9)
    }
}

impl FluxNetWeightOptimizer {
    /// Create new optimizer with learning rate and initial temperature
    pub fn new(learning_rate: f64, temperature: f64) -> Self {
        let default_weights = ScoringWeights::default();
        Self {
            weights: [
                default_weights.volume,
                default_weights.hydrophobicity,
                default_weights.enclosure,
                default_weights.depth,
                default_weights.hbond_capacity,
                default_weights.flexibility,
                default_weights.conservation,
                default_weights.topology,
            ],
            learning_rate,
            baseline: 0.0,
            momentum: [0.0; 8],
            temperature,
            episode_count: 0,
        }
    }

    /// Initialize from existing weights
    pub fn from_weights(weights: ScoringWeights, learning_rate: f64) -> Self {
        Self {
            weights: [
                weights.volume,
                weights.hydrophobicity,
                weights.enclosure,
                weights.depth,
                weights.hbond_capacity,
                weights.flexibility,
                weights.conservation,
                weights.topology,
            ],
            learning_rate,
            baseline: 0.0,
            momentum: [0.0; 8],
            temperature: 0.9,
            episode_count: 0,
        }
    }

    /// Compute reward from validation metrics
    /// Higher is better: success_rate + coverage - (distance penalty)
    pub fn compute_reward(&self, metrics: &ValidationMetrics) -> f64 {
        let distance_penalty = (metrics.center_distance() / 4.0).min(2.0);
        let reward = metrics.success_rate()
            + 0.5 * metrics.ligand_coverage
            + 0.3 * metrics.pocket_precision
            - 0.4 * distance_penalty;
        reward.clamp(-1.0, 2.0)
    }

    /// Update weights using policy gradient with baseline
    pub fn update(&mut self, metrics: &ValidationMetrics) {
        let reward = self.compute_reward(metrics);

        // Update baseline with exponential moving average
        let alpha = 0.1;
        self.baseline = alpha * reward + (1.0 - alpha) * self.baseline;

        // Advantage = reward - baseline
        let advantage = reward - self.baseline;

        // Policy gradient update with momentum
        let momentum_decay = 0.9;
        for i in 0..8 {
            // Gradient estimate: advantage * weight perturbation direction
            // Using log-derivative trick approximation
            let grad = advantage * (self.weights[i] - 0.125);

            self.momentum[i] = momentum_decay * self.momentum[i] + self.learning_rate * grad;
            self.weights[i] += self.momentum[i];
            self.weights[i] = self.weights[i].clamp(0.01, 0.5);
        }

        // Normalize weights
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }

        // Cool down temperature (simulated annealing effect)
        self.temperature *= 0.995;
        self.temperature = self.temperature.max(0.1);

        self.episode_count += 1;
    }

    /// Get current weights as ScoringWeights
    pub fn get_weights(&self) -> ScoringWeights {
        ScoringWeights {
            volume: self.weights[0],
            hydrophobicity: self.weights[1],
            enclosure: self.weights[2],
            depth: self.weights[3],
            hbond_capacity: self.weights[4],
            flexibility: self.weights[5],
            conservation: self.weights[6],
            topology: self.weights[7],
        }
    }

    /// Get weights as array
    pub fn weights_array(&self) -> [f64; 8] {
        self.weights
    }

    /// Update weights from array (direct set)
    pub fn update_weights(&mut self, new_weights: [f64; 8]) {
        for (i, &w) in new_weights.iter().enumerate() {
            self.weights[i] = w.clamp(0.01, 0.5);
        }

        // Normalize
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }

    /// Run training episode with a batch of structures and known ligand positions
    pub fn train_episode(
        &mut self,
        structures: &[(ProteinStructure, Vec<[f64; 3]>)],
        threshold: f64,
    ) -> Result<f64> {
        let mut total_reward = 0.0;

        for (structure, ligand_coords) in structures {
            let mut config = LbsConfig::default();
            config.scoring = self.get_weights();

            let lbs = PrismLbs::new(config)?;
            let pockets = lbs.predict(structure)?;

            let metrics = ValidationMetrics::compute_batch(&pockets, ligand_coords, threshold);
            total_reward += self.compute_reward(&metrics);
            self.update(&metrics);
        }

        Ok(total_reward / structures.len() as f64)
    }

    /// Get episode count
    pub fn episodes(&self) -> usize {
        self.episode_count
    }

    /// Get current temperature (for logging)
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
}

/// Training configuration for FluxNet weight optimization
pub struct FluxNetTrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub validation_threshold: f64,
    pub early_stop_patience: usize,
}

impl Default for FluxNetTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 8,
            learning_rate: 0.01,
            validation_threshold: 4.0,  // 4Å DCC threshold (PDBBind standard)
            early_stop_patience: 10,
        }
    }
}
