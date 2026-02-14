//! Adaptive Parameter (ADP) adjuster for warmstart configuration.
//!
//! Implements Warmstart Plan Step 5: ADP Feedback Integration.
//!
//! Tunes warmstart weights and anchor fraction based on effectiveness feedback
//! using gradient descent with momentum for stability.

use prism_core::{PrismError, WarmstartConfig};
use std::collections::VecDeque;

/// Default learning rate for parameter updates
const DEFAULT_LEARNING_RATE: f32 = 0.01;

/// Default momentum for gradient smoothing
const DEFAULT_MOMENTUM: f32 = 0.9;

/// History window size for effectiveness tracking
const HISTORY_WINDOW: usize = 10;

/// Min/max bounds for anchor fraction
const MIN_ANCHOR_FRACTION: f32 = 0.05;
const MAX_ANCHOR_FRACTION: f32 = 0.20;

/// Min/max bounds for weights (before normalization)
const MIN_WEIGHT: f32 = 0.1;
const MAX_WEIGHT: f32 = 0.8;

/// Adaptive Parameter adjuster for warmstart configuration.
///
/// Uses gradient descent to tune warmstart parameters based on effectiveness
/// feedback. Estimates gradients via finite differences on effectiveness history.
///
/// ## Algorithm
/// 1. Maintain windowed history of effectiveness scores (last 10 runs)
/// 2. Estimate gradient: Δeffectiveness / Δparameter
/// 3. Apply update with momentum: param_new = param_old + lr * (momentum * grad_old + (1-momentum) * grad_new)
/// 4. Clamp parameters to valid ranges
/// 5. Normalize weights to sum = 1.0
///
/// ## Example
/// ```rust,ignore
/// use prism_pipeline::AdpWarmstartAdjuster;
/// use prism_core::WarmstartConfig;
///
/// let mut adjuster = AdpWarmstartAdjuster::new(0.01, 0.9);
/// let mut config = WarmstartConfig::default();
///
/// // After each run
/// let effectiveness = 0.75;  // 75% conflict reduction
/// adjuster.adjust(&mut config, effectiveness)?;
/// ```
///
/// Implements Warmstart Plan §5: ADP Feedback Integration
#[derive(Debug, Clone)]
pub struct AdpWarmstartAdjuster {
    /// Learning rate for parameter updates
    learning_rate: f32,

    /// Momentum coefficient for gradient smoothing
    momentum: f32,

    /// History of effectiveness scores (windowed)
    effectiveness_history: VecDeque<f32>,

    /// Previous weight gradients (for momentum)
    flux_gradient: f32,
    ensemble_gradient: f32,
    random_gradient: f32,

    /// Previous anchor fraction gradient (for momentum)
    anchor_gradient: f32,

    /// Number of adjustments made (for convergence tracking)
    adjustment_count: usize,
}

impl AdpWarmstartAdjuster {
    /// Creates a new ADP adjuster with default parameters.
    pub fn new() -> Self {
        Self::with_params(DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM)
    }

    /// Creates a new ADP adjuster with custom learning rate and momentum.
    ///
    /// # Arguments
    /// * `learning_rate` - Step size for gradient descent (typically 0.001 - 0.1)
    /// * `momentum` - Smoothing factor for gradients (typically 0.8 - 0.95)
    ///
    /// # Panics
    /// Panics if learning_rate or momentum are outside valid ranges.
    pub fn with_params(learning_rate: f32, momentum: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&learning_rate),
            "Learning rate must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&momentum),
            "Momentum must be in [0, 1]"
        );

        Self {
            learning_rate,
            momentum,
            effectiveness_history: VecDeque::with_capacity(HISTORY_WINDOW),
            flux_gradient: 0.0,
            ensemble_gradient: 0.0,
            random_gradient: 0.0,
            anchor_gradient: 0.0,
            adjustment_count: 0,
        }
    }

    /// Adjusts warmstart configuration based on effectiveness feedback.
    ///
    /// Updates weights and anchor fraction using gradient descent. Requires
    /// at least 3 effectiveness samples to estimate gradients.
    ///
    /// # Arguments
    /// * `config` - Warmstart configuration to adjust (modified in-place)
    /// * `effectiveness` - Effectiveness score from last run (0.0 - 1.0)
    ///
    /// # Returns
    /// Ok(()) if adjustment successful, Err if effectiveness is invalid.
    ///
    /// # Note
    /// First 2 calls do nothing (collecting history). Gradients estimated
    /// starting from 3rd call.
    pub fn adjust(
        &mut self,
        config: &mut WarmstartConfig,
        effectiveness: f32,
    ) -> Result<(), PrismError> {
        // Validate effectiveness
        if !(0.0..=1.0).contains(&effectiveness) {
            return Err(PrismError::validation(format!(
                "Invalid effectiveness: {} (must be in [0, 1])",
                effectiveness
            )));
        }

        // Add to history
        self.effectiveness_history.push_back(effectiveness);
        if self.effectiveness_history.len() > HISTORY_WINDOW {
            self.effectiveness_history.pop_front();
        }

        // Need at least 3 samples for gradient estimation
        if self.effectiveness_history.len() < 3 {
            log::debug!(
                "ADP: Collecting history ({}/3 samples)",
                self.effectiveness_history.len()
            );
            return Ok(());
        }

        // Compute effectiveness trend (simple finite difference)
        let recent_avg = self.effectiveness_history.iter().rev().take(3).sum::<f32>() / 3.0;
        let older_avg = self
            .effectiveness_history
            .iter()
            .rev()
            .skip(3)
            .take(3)
            .sum::<f32>()
            .max(0.01)
            / 3.0;
        let trend = (recent_avg - older_avg) / older_avg; // Relative change

        // Update weight gradients (simple heuristic: increase if improving)
        let new_flux_grad = if trend > 0.0 { 0.05 } else { -0.05 };
        let new_ensemble_grad = if trend > 0.0 { 0.03 } else { -0.03 };
        let new_random_grad = if trend > 0.0 { -0.02 } else { 0.02 };

        // Apply momentum
        self.flux_gradient =
            self.momentum * self.flux_gradient + (1.0 - self.momentum) * new_flux_grad;
        self.ensemble_gradient =
            self.momentum * self.ensemble_gradient + (1.0 - self.momentum) * new_ensemble_grad;
        self.random_gradient =
            self.momentum * self.random_gradient + (1.0 - self.momentum) * new_random_grad;

        // Update weights
        config.flux_weight = (config.flux_weight + self.learning_rate * self.flux_gradient)
            .clamp(MIN_WEIGHT, MAX_WEIGHT);
        config.ensemble_weight = (config.ensemble_weight
            + self.learning_rate * self.ensemble_gradient)
            .clamp(MIN_WEIGHT, MAX_WEIGHT);
        config.random_weight = (config.random_weight + self.learning_rate * self.random_gradient)
            .clamp(MIN_WEIGHT, MAX_WEIGHT);

        // Normalize weights to sum = 1.0
        let weight_sum = config.flux_weight + config.ensemble_weight + config.random_weight;
        config.flux_weight /= weight_sum;
        config.ensemble_weight /= weight_sum;
        config.random_weight /= weight_sum;

        // Update anchor fraction gradient
        let new_anchor_grad = if trend > 0.0 && config.anchor_fraction < 0.15 {
            0.01 // Increase anchors if improving and below threshold
        } else if trend < 0.0 && config.anchor_fraction > 0.05 {
            -0.01 // Decrease anchors if degrading
        } else {
            0.0 // Keep stable
        };

        self.anchor_gradient =
            self.momentum * self.anchor_gradient + (1.0 - self.momentum) * new_anchor_grad;

        // Update anchor fraction
        config.anchor_fraction = (config.anchor_fraction
            + self.learning_rate * self.anchor_gradient)
            .clamp(MIN_ANCHOR_FRACTION, MAX_ANCHOR_FRACTION);

        self.adjustment_count += 1;

        log::debug!(
            "ADP: Adjusted config (iteration {}): flux={:.3}, ensemble={:.3}, random={:.3}, anchors={:.3}, trend={:.3}",
            self.adjustment_count,
            config.flux_weight,
            config.ensemble_weight,
            config.random_weight,
            config.anchor_fraction,
            trend
        );

        Ok(())
    }

    /// Returns the number of adjustments made.
    pub fn adjustment_count(&self) -> usize {
        self.adjustment_count
    }

    /// Returns current effectiveness history.
    pub fn effectiveness_history(&self) -> &VecDeque<f32> {
        &self.effectiveness_history
    }

    /// Resets the adjuster state (clears history and gradients).
    pub fn reset(&mut self) {
        self.effectiveness_history.clear();
        self.flux_gradient = 0.0;
        self.ensemble_gradient = 0.0;
        self.random_gradient = 0.0;
        self.anchor_gradient = 0.0;
        self.adjustment_count = 0;
    }
}

impl Default for AdpWarmstartAdjuster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adp_initialization() {
        let adjuster = AdpWarmstartAdjuster::new();
        assert_eq!(adjuster.adjustment_count(), 0);
        assert_eq!(adjuster.effectiveness_history().len(), 0);
    }

    #[test]
    fn test_adp_collect_history() {
        let mut adjuster = AdpWarmstartAdjuster::new();
        let mut config = WarmstartConfig::default();

        // First 2 calls should just collect history
        adjuster.adjust(&mut config, 0.5).unwrap();
        adjuster.adjust(&mut config, 0.6).unwrap();

        assert_eq!(adjuster.effectiveness_history().len(), 2);
        assert_eq!(adjuster.adjustment_count(), 0); // No adjustments yet
    }

    #[test]
    fn test_adp_adjustment_improving() {
        let mut adjuster = AdpWarmstartAdjuster::with_params(0.1, 0.0); // High LR, no momentum
        let mut config = WarmstartConfig::default();
        let initial_flux = config.flux_weight;

        // Simulate improving effectiveness
        for eff in &[0.3, 0.4, 0.5, 0.6, 0.7] {
            adjuster.adjust(&mut config, *eff).unwrap();
        }

        assert!(adjuster.adjustment_count() >= 2); // Should have adjusted

        // Weights should still sum to 1.0
        let sum = config.flux_weight + config.ensemble_weight + config.random_weight;
        assert!((sum - 1.0).abs() < 0.01);

        // Weights should be in valid range
        assert!(config.flux_weight >= MIN_WEIGHT && config.flux_weight <= MAX_WEIGHT);
        assert!(config.anchor_fraction >= MIN_ANCHOR_FRACTION);
        assert!(config.anchor_fraction <= MAX_ANCHOR_FRACTION);
    }

    #[test]
    fn test_adp_weight_normalization() {
        let mut adjuster = AdpWarmstartAdjuster::new();
        let mut config = WarmstartConfig {
            flux_weight: 0.5,
            ensemble_weight: 0.3,
            random_weight: 0.2,
            ..Default::default()
        };

        // Add enough history
        for _ in 0..5 {
            adjuster.adjust(&mut config, 0.5).unwrap();
        }

        // Check normalization
        let sum = config.flux_weight + config.ensemble_weight + config.random_weight;
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Weights must sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_adp_anchor_bounds() {
        let mut adjuster = AdpWarmstartAdjuster::with_params(0.5, 0.0); // Very high LR
        let mut config = WarmstartConfig::default();

        // Push effectiveness up to test upper bound
        for _ in 0..20 {
            adjuster.adjust(&mut config, 0.9).unwrap();
        }

        assert!(config.anchor_fraction <= MAX_ANCHOR_FRACTION);
        assert!(config.anchor_fraction >= MIN_ANCHOR_FRACTION);
    }

    #[test]
    fn test_adp_invalid_effectiveness() {
        let mut adjuster = AdpWarmstartAdjuster::new();
        let mut config = WarmstartConfig::default();

        assert!(adjuster.adjust(&mut config, -0.1).is_err());
        assert!(adjuster.adjust(&mut config, 1.5).is_err());
    }

    #[test]
    fn test_adp_reset() {
        let mut adjuster = AdpWarmstartAdjuster::new();
        let mut config = WarmstartConfig::default();

        for eff in &[0.5, 0.6, 0.7] {
            adjuster.adjust(&mut config, *eff).unwrap();
        }

        assert!(adjuster.adjustment_count() > 0);
        adjuster.reset();
        assert_eq!(adjuster.adjustment_count(), 0);
        assert_eq!(adjuster.effectiveness_history().len(), 0);
    }
}
