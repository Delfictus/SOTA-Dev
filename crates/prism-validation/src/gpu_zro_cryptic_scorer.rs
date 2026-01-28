//! GPU-accelerated PRISM-ZrO cryptic site scorer
//!
//! Uses full 512-neuron DendriticSNNReservoir with Feature Adapter Protocol
//! and RLS (Recursive Least Squares) online learning.
//!
//! # Zero Fallback Policy
//! This module REQUIRES a valid CUDA context. It will NOT fall back to CPU.
//! If no GPU is available, initialization will fail with an explicit error.
//!
//! # Architecture
//!
//! ```text
//! CrypticFeatures (16-dim)
//!        │
//!        ▼
//! encode_with_velocity() ──→ 40-dim buffer
//!        │
//!        ▼
//! DendriticSNNReservoir (GPU)
//!        │ (internal 40→80 expansion)
//!        ▼
//! 512-dim reservoir state
//!        │
//!        ▼
//! RLS readout ──→ sigmoid ──→ cryptic score [0,1]
//! ```

use anyhow::{bail, Context, Result};
// Note: bail! used in load_weights for dimension mismatch
use std::sync::Arc;

use crate::cryptic_features::CrypticFeatures;

#[cfg(feature = "cryptic-gpu")]
use cudarc::driver::CudaContext;

#[cfg(feature = "cryptic-gpu")]
use prism_gpu::DendriticSNNReservoir;

/// Number of neurons in the reservoir
pub const RESERVOIR_SIZE: usize = 512;

/// RLS forgetting factor
pub const RLS_LAMBDA: f32 = 0.99;

/// GPU-accelerated cryptic site scorer using PRISM-ZrO architecture
///
/// Combines:
/// - 512-neuron dendritic SNN reservoir (GPU-accelerated)
/// - RLS online learning for readout weights
/// - Velocity encoding for temporal dynamics
#[cfg(feature = "cryptic-gpu")]
pub struct GpuZroCrypticScorer {
    /// 512-neuron dendritic SNN reservoir (GPU)
    reservoir: DendriticSNNReservoir,

    /// RLS readout weights [512] -> single score
    readout_weights: Vec<f32>,

    /// RLS precision matrix [512 x 512]
    precision_matrix: Vec<f32>,

    /// Forgetting factor (0.99)
    lambda: f32,

    /// Number of RLS updates performed
    update_count: usize,

    /// Previous features for velocity computation
    prev_features: Option<CrypticFeatures>,

    /// Maximum allowed precision matrix trace (stability)
    max_precision_trace: f32,

    /// Gradient clamp value for stability
    gradient_clamp: f32,
}

#[cfg(feature = "cryptic-gpu")]
impl GpuZroCrypticScorer {
    /// Initialize GPU scorer with CUDA context
    ///
    /// # Errors
    /// Returns error if CUDA context is invalid or GPU initialization fails.
    /// This is intentional - we do NOT fall back to CPU.
    ///
    /// # Zero Fallback Policy
    /// The caller must provide a valid CUDA context. If context creation
    /// failed (no GPU available), this function should not be called.
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        // The context being valid proves a GPU exists.
        // If no GPU were available, CudaContext::new() would have failed.
        let mut reservoir = DendriticSNNReservoir::new(context, RESERVOIR_SIZE)
            .context("Failed to create 512-neuron GPU reservoir. Is CUDA available?")?;

        reservoir
            .initialize(42)
            .context("Failed to initialize reservoir weights")?;

        // Initialize readout weights to zero
        let readout_weights = vec![0.0f32; RESERVOIR_SIZE];

        // Initialize precision matrix to 100 * I (identity scaled)
        let mut precision_matrix = vec![0.0f32; RESERVOIR_SIZE * RESERVOIR_SIZE];
        for i in 0..RESERVOIR_SIZE {
            precision_matrix[i * RESERVOIR_SIZE + i] = 100.0;
        }

        log::info!(
            "GPU ZrO Cryptic Scorer initialized: {} neurons, RLS lambda={}",
            RESERVOIR_SIZE,
            RLS_LAMBDA
        );

        Ok(Self {
            reservoir,
            readout_weights,
            precision_matrix,
            lambda: RLS_LAMBDA,
            update_count: 0,
            prev_features: None,
            max_precision_trace: 1e6,
            gradient_clamp: 1.0,
        })
    }

    /// Process cryptic features through GPU reservoir (inference only)
    ///
    /// Returns a score in [0, 1] indicating cryptic site probability.
    pub fn score_residue(&mut self, features: &CrypticFeatures) -> Result<f32> {
        let mut input = [0.0f32; 40];

        if let Some(ref prev) = self.prev_features {
            features.encode_with_velocity(prev, &mut input);
        } else {
            features.encode_into(&mut input);
        }

        self.prev_features = Some(features.clone());

        // Process through GPU reservoir
        let state = self
            .reservoir
            .process_features(&input)
            .context("GPU reservoir processing failed")?;

        // Compute score via readout weights
        let raw_score: f32 = state
            .iter()
            .zip(&self.readout_weights)
            .map(|(s, w)| s * w)
            .sum();

        Ok(sigmoid(raw_score))
    }

    /// Score with RLS online learning from ground truth label
    ///
    /// Returns the prediction BEFORE the weight update (for fair evaluation).
    pub fn score_and_learn(&mut self, features: &CrypticFeatures, ground_truth: bool) -> Result<f32> {
        let mut input = [0.0f32; 40];

        if let Some(ref prev) = self.prev_features {
            features.encode_with_velocity(prev, &mut input);
        } else {
            features.encode_into(&mut input);
        }

        self.prev_features = Some(features.clone());

        // Process through GPU reservoir
        let state = self
            .reservoir
            .process_features(&input)
            .context("GPU reservoir processing failed")?;

        // Compute current prediction (before update)
        let raw_score: f32 = state
            .iter()
            .zip(&self.readout_weights)
            .map(|(s, w)| s * w)
            .sum();

        let prediction = sigmoid(raw_score);
        let target = if ground_truth { 1.0 } else { 0.0 };

        // Perform RLS update
        self.rls_update(&state, target)
            .context("RLS weight update failed")?;

        Ok(prediction)
    }

    /// Sherman-Morrison RLS update with stability safeguards
    fn rls_update(&mut self, state: &[f32], target: f32) -> Result<()> {
        let n = RESERVOIR_SIZE;
        let k = state;

        // Compute P * k
        let mut pk = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                pk[i] += self.precision_matrix[i * n + j] * k[j];
            }
        }

        // Compute k' * P * k
        let kpk: f32 = k.iter().zip(&pk).map(|(ki, pki)| ki * pki).sum();

        // Compute gain with numerical stability
        let gain = 1.0 / (self.lambda + kpk + 1e-8);

        // Update precision matrix: P = (1/lambda)(P - gain * pk * pk')
        let inv_lambda = 1.0 / self.lambda;
        for i in 0..n {
            for j in 0..n {
                self.precision_matrix[i * n + j] =
                    inv_lambda * (self.precision_matrix[i * n + j] - gain * pk[i] * pk[j]);
            }
        }

        // Compute prediction error with gradient clamp
        let prediction: f32 = k
            .iter()
            .zip(&self.readout_weights)
            .map(|(ki, wi)| ki * wi)
            .sum();
        let error = (target - sigmoid(prediction)).clamp(-self.gradient_clamp, self.gradient_clamp);

        // Update weights: w = w + P * k * error
        for i in 0..n {
            let delta = pk[i] * gain * error;
            self.readout_weights[i] += delta;

            // Clamp weights for stability
            self.readout_weights[i] = self.readout_weights[i].clamp(-10.0, 10.0);
        }

        self.update_count += 1;

        // Periodic stability check
        if self.update_count % 100 == 0 {
            self.stability_check()?;
        }

        Ok(())
    }

    /// Check and fix numerical stability issues
    fn stability_check(&mut self) -> Result<()> {
        // Check for NaN/Inf in weights
        if self.readout_weights.iter().any(|w| !w.is_finite()) {
            log::warn!("NaN/Inf detected in weights, resetting");
            self.readout_weights.fill(0.0);
        }

        // Check precision matrix trace
        let trace: f32 = (0..RESERVOIR_SIZE)
            .map(|i| self.precision_matrix[i * RESERVOIR_SIZE + i])
            .sum();

        if trace > self.max_precision_trace {
            log::warn!(
                "Precision matrix trace ({:.2e}) exceeded threshold, soft reset",
                trace
            );
            self.soft_reset_precision();
        }

        if !trace.is_finite() {
            log::error!("Precision matrix contains NaN/Inf, full reset");
            self.reset()?;
        }

        Ok(())
    }

    /// Soft reset precision matrix while preserving learned weights
    fn soft_reset_precision(&mut self) {
        for i in 0..RESERVOIR_SIZE {
            for j in 0..RESERVOIR_SIZE {
                if i == j {
                    self.precision_matrix[i * RESERVOIR_SIZE + j] = 10.0;
                } else {
                    self.precision_matrix[i * RESERVOIR_SIZE + j] = 0.0;
                }
            }
        }
    }

    /// Full reset (weights and precision matrix)
    pub fn reset(&mut self) -> Result<()> {
        self.readout_weights.fill(0.0);
        for i in 0..RESERVOIR_SIZE {
            for j in 0..RESERVOIR_SIZE {
                self.precision_matrix[i * RESERVOIR_SIZE + j] = if i == j { 100.0 } else { 0.0 };
            }
        }
        self.update_count = 0;
        self.prev_features = None;

        log::debug!("GPU scorer reset complete");
        Ok(())
    }

    /// Reset state for new structure (keep weights, clear state)
    pub fn reset_for_structure(&mut self) {
        self.prev_features = None;
        self.reservoir.reset_state().ok();
    }

    /// Save learned weights to file (binary format via bincode)
    ///
    /// Uses bincode for compact, fast, precision-preserving serialization.
    /// This is the production-grade format for neural network weights.
    pub fn save_weights(&self, path: &str) -> Result<()> {
        let data = bincode::serialize(&self.readout_weights)
            .context("Failed to serialize weights")?;
        std::fs::write(path, data).context("Failed to write weights file")?;

        log::info!("Saved weights to {} ({} updates, {} bytes)",
                   path, self.update_count, self.readout_weights.len() * 4);
        Ok(())
    }

    /// Load pre-trained weights from file (binary format via bincode)
    ///
    /// Expects weights saved with `save_weights()` in bincode format.
    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        let data = std::fs::read(path).context("Failed to read weights file")?;
        self.readout_weights =
            bincode::deserialize(&data).context("Failed to deserialize weights")?;

        if self.readout_weights.len() != RESERVOIR_SIZE {
            bail!(
                "Weight dimension mismatch: expected {}, got {}",
                RESERVOIR_SIZE,
                self.readout_weights.len()
            );
        }

        log::info!("Loaded weights from {}", path);
        Ok(())
    }

    /// Get number of RLS updates performed
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    /// Get current weight statistics for logging
    pub fn weight_stats(&self) -> WeightStats {
        let mean = self.readout_weights.iter().sum::<f32>() / RESERVOIR_SIZE as f32;
        let var = self
            .readout_weights
            .iter()
            .map(|w| (w - mean).powi(2))
            .sum::<f32>()
            / RESERVOIR_SIZE as f32;
        let max = self
            .readout_weights
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min = self
            .readout_weights
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);

        WeightStats {
            mean,
            std: var.sqrt(),
            min,
            max,
        }
    }

    /// Get reference to readout weights (for inspection/analysis)
    pub fn readout_weights(&self) -> &[f32] {
        &self.readout_weights
    }
}

/// Weight statistics for logging and diagnostics
#[derive(Debug, Clone)]
pub struct WeightStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
}

impl std::fmt::Display for WeightStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]",
            self.mean, self.std, self.min, self.max
        )
    }
}

/// Sigmoid activation function
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(all(test, feature = "cryptic-gpu"))]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_scorer_initialization() {
        let context = CudaContext::new(0).expect("CUDA not available");
        let scorer = GpuZroCrypticScorer::new(context);
        assert!(scorer.is_ok(), "Failed to initialize scorer: {:?}", scorer.err());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_score_residue() {
        let context = CudaContext::new(0).expect("CUDA not available");
        let mut scorer = GpuZroCrypticScorer::new(context).expect("Failed to create scorer");

        let features = CrypticFeatures {
            burial_change: 0.5,
            rmsf: 1.2,
            variance: 0.3,
            ..Default::default()
        };

        let score = scorer.score_residue(&features);
        assert!(score.is_ok(), "Scoring failed: {:?}", score.err());

        let score_val = score.unwrap();
        assert!(score_val >= 0.0 && score_val <= 1.0, "Score {} out of range", score_val);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_score_and_learn() {
        let context = CudaContext::new(0).expect("CUDA not available");
        let mut scorer = GpuZroCrypticScorer::new(context).expect("Failed to create scorer");

        let features = CrypticFeatures {
            burial_change: 0.8,
            rmsf: 2.0,
            variance: 0.5,
            contact_density: 10.0,
            ..Default::default()
        };

        // Learn from positive example
        let score1 = scorer.score_and_learn(&features, true).expect("Learning failed");
        assert!(score1 >= 0.0 && score1 <= 1.0);

        // Update count should increase
        assert_eq!(scorer.update_count(), 1);

        // Learn from negative example
        let neg_features = CrypticFeatures::default();
        let score2 = scorer.score_and_learn(&neg_features, false).expect("Learning failed");
        assert!(score2 >= 0.0 && score2 <= 1.0);

        assert_eq!(scorer.update_count(), 2);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_weight_stats() {
        let context = CudaContext::new(0).expect("CUDA not available");
        let scorer = GpuZroCrypticScorer::new(context).expect("Failed to create scorer");

        let stats = scorer.weight_stats();
        // Initial weights are zero
        assert!((stats.mean - 0.0).abs() < 1e-6);
        assert!((stats.std - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
