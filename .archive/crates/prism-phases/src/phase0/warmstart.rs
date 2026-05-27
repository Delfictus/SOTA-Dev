//! Phase 0 Warmstart Prior Generation
//!
//! Implements flux reservoir prior generation using softmax over vertex difficulty
//! and uncertainty metrics from dendritic reservoir computation.

use prism_core::{WarmstartConfig, WarmstartPrior};

/// Builds warmstart priors from reservoir difficulty and uncertainty metrics.
///
/// ## Algorithm
/// 1. For each vertex, compute softmax over difficulty scores
/// 2. Clamp probabilities to [min_prob, max_prob] range
/// 3. Normalize to ensure sum = 1.0
///
/// ## Parameters
/// - `difficulty`: Per-vertex difficulty scores (higher = harder to color)
/// - `uncertainty`: Per-vertex uncertainty scores (higher = more exploration needed)
/// - `config`: Warmstart configuration (max_colors, min_prob, etc.)
///
/// ## Returns
/// Vector of `WarmstartPrior` with probabilistic color distributions.
///
/// ## Example
/// ```rust
/// use prism_core::WarmstartConfig;
/// use prism_phases::phase0::warmstart::build_reservoir_prior;
///
/// let difficulty = vec![0.5, 0.8, 0.3];
/// let uncertainty = vec![0.2, 0.6, 0.1];
/// let config = WarmstartConfig::default();
///
/// let priors = build_reservoir_prior(&difficulty, &uncertainty, &config);
/// assert_eq!(priors.len(), 3);
/// ```
pub fn build_reservoir_prior(
    difficulty: &[f32],
    uncertainty: &[f32],
    config: &WarmstartConfig,
) -> Vec<WarmstartPrior> {
    let num_vertices = difficulty.len();
    assert_eq!(
        num_vertices,
        uncertainty.len(),
        "difficulty and uncertainty must have same length"
    );

    let max_colors = config.max_colors;
    let min_prob = config.min_prob;
    let max_prob = 1.0 - (max_colors as f32 - 1.0) * min_prob;

    let mut priors = Vec::with_capacity(num_vertices);

    for vertex in 0..num_vertices {
        let diff = difficulty[vertex];
        let uncert = uncertainty[vertex];

        // Combine difficulty and uncertainty (weighted sum)
        let combined_score = 0.7 * diff + 0.3 * uncert;

        // Generate color probabilities using softmax-like distribution
        let color_probs =
            softmax_color_distribution(combined_score, max_colors, min_prob, max_prob);

        priors.push(WarmstartPrior {
            vertex,
            color_probabilities: color_probs,
            is_anchor: false,
            anchor_color: None,
        });
    }

    priors
}

/// Generates color probability distribution using softmax-inspired approach.
///
/// Higher difficulty scores lead to more uniform distributions (higher entropy),
/// while lower scores lead to more peaked distributions.
fn softmax_color_distribution(
    score: f32,
    max_colors: usize,
    min_prob: f32,
    max_prob: f32,
) -> Vec<f32> {
    let mut probs = Vec::with_capacity(max_colors);

    // Temperature parameter: higher score -> higher temperature -> more uniform
    let temperature = 1.0 + 2.0 * score;

    // Generate raw softmax weights
    let mut raw_weights = Vec::with_capacity(max_colors);
    let mut sum = 0.0;

    for i in 0..max_colors {
        // Exponentially decaying preference for lower color indices
        let weight = (-(i as f32) / temperature).exp();
        raw_weights.push(weight);
        sum += weight;
    }

    // Normalize and clamp
    for i in 0..max_colors {
        let normalized = raw_weights[i] / sum;
        let clamped = normalized.clamp(min_prob, max_prob);
        probs.push(clamped);
    }

    // Re-normalize to ensure sum = 1.0 after clamping
    let final_sum: f32 = probs.iter().sum();
    if final_sum > 0.0 {
        for prob in &mut probs {
            *prob /= final_sum;
        }
    }

    probs
}

/// Fuses multiple warmstart priors using weighted average.
///
/// Used to combine reservoir priors with other sources (ensemble, curriculum).
///
/// ## Parameters
/// - `priors`: Vector of prior sources to fuse
/// - `weights`: Corresponding weights (must sum to 1.0)
///
/// ## Returns
/// Fused prior with combined probability distributions.
pub fn fuse_priors(priors: &[&WarmstartPrior], weights: &[f32]) -> WarmstartPrior {
    assert_eq!(
        priors.len(),
        weights.len(),
        "priors and weights must have same length"
    );
    assert!(!priors.is_empty(), "Cannot fuse empty prior list");

    let vertex = priors[0].vertex;
    let max_colors = priors[0].color_probabilities.len();

    // Verify all priors are for the same vertex and have same color count
    for prior in priors.iter() {
        assert_eq!(
            prior.vertex, vertex,
            "All priors must be for the same vertex"
        );
        assert_eq!(
            prior.color_probabilities.len(),
            max_colors,
            "All priors must have same number of colors"
        );
    }

    // Weighted average of color probabilities
    let mut fused_probs = vec![0.0; max_colors];
    for (prior, &weight) in priors.iter().zip(weights.iter()) {
        for (i, &prob) in prior.color_probabilities.iter().enumerate() {
            fused_probs[i] += weight * prob;
        }
    }

    // Normalize
    let sum: f32 = fused_probs.iter().sum();
    if sum > 0.0 {
        for prob in &mut fused_probs {
            *prob /= sum;
        }
    }

    // Check if any prior is an anchor (anchors take precedence)
    let (is_anchor, anchor_color) = priors
        .iter()
        .find_map(|p| {
            if p.is_anchor {
                Some((true, p.anchor_color))
            } else {
                None
            }
        })
        .unwrap_or((false, None));

    WarmstartPrior {
        vertex,
        color_probabilities: fused_probs,
        is_anchor,
        anchor_color,
    }
}

/// Applies geometry-based hotspot prioritization to warmstart priors.
///
/// When geometric stress is high, this function boosts the probability of anchoring
/// vertices that are identified as conflict hotspots. This implements metaphysical
/// telemetry coupling by directing warmstart attention to geometrically stressed regions.
///
/// ## Algorithm
/// 1. Check if geometry stress exceeds threshold (0.6)
/// 2. For hotspot vertices, increase first-color probability (anchoring bias)
/// 3. Re-normalize probability distributions
/// 4. Mark hotspot vertices as anchor candidates
///
/// ## Parameters
/// - `priors`: Mutable warmstart priors to adjust
/// - `geometry_metrics`: Optional geometry telemetry from Phase 4/6
/// - `stress_threshold`: Threshold for activating hotspot strategy (default: 0.6)
/// - `hotspot_boost`: Probability boost factor for hotspots (default: 2.0)
///
/// ## Returns
/// Number of hotspots prioritized
pub fn apply_geometry_hotspot_prioritization(
    priors: &mut [WarmstartPrior],
    geometry_metrics: Option<&prism_core::GeometryTelemetry>,
    stress_threshold: f32,
    hotspot_boost: f32,
) -> usize {
    let Some(geom) = geometry_metrics else {
        return 0; // No geometry metrics available
    };

    // Check if stress is high enough to activate hotspot strategy
    if geom.stress_scalar < stress_threshold {
        return 0;
    }

    log::info!(
        "[Warmstart] Geometry stress {:.3} exceeds threshold {:.3}. Prioritizing {} hotspots.",
        geom.stress_scalar,
        stress_threshold,
        geom.anchor_hotspots.len()
    );

    let mut hotspots_prioritized = 0;

    // Boost probability for hotspot vertices
    for hotspot_vertex in &geom.anchor_hotspots {
        if *hotspot_vertex < priors.len() {
            let prior = &mut priors[*hotspot_vertex];

            // Boost first-color probability (anchoring bias)
            // This makes hotspot vertices more likely to be anchored early
            if !prior.color_probabilities.is_empty() {
                prior.color_probabilities[0] *= hotspot_boost;
            }

            // Re-normalize
            let sum: f32 = prior.color_probabilities.iter().sum();
            if sum > 0.0 {
                for prob in &mut prior.color_probabilities {
                    *prob /= sum;
                }
            }

            hotspots_prioritized += 1;
        }
    }

    log::debug!(
        "[Warmstart] Applied hotspot prioritization to {} vertices with boost factor {:.2}x",
        hotspots_prioritized,
        hotspot_boost
    );

    hotspots_prioritized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_reservoir_prior() {
        let difficulty = vec![0.2, 0.5, 0.8];
        let uncertainty = vec![0.1, 0.3, 0.6];
        let config = WarmstartConfig::default();

        let priors = build_reservoir_prior(&difficulty, &uncertainty, &config);

        assert_eq!(priors.len(), 3);
        for prior in &priors {
            // Check probabilities sum to ~1.0
            let sum: f32 = prior.color_probabilities.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Probabilities should sum to 1.0");

            // Check all probabilities are in valid range
            for &prob in &prior.color_probabilities {
                assert!(prob >= config.min_prob);
                assert!(prob <= 1.0);
            }
        }
    }

    #[test]
    fn test_softmax_distribution_properties() {
        let config = WarmstartConfig::default();

        // Low difficulty -> more peaked distribution (lower entropy)
        let low_diff_probs = softmax_color_distribution(0.1, 10, config.min_prob, 1.0);
        let low_entropy: f32 = low_diff_probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum();

        // High difficulty -> more uniform distribution (higher entropy)
        let high_diff_probs = softmax_color_distribution(0.9, 10, config.min_prob, 1.0);
        let high_entropy: f32 = high_diff_probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum();

        assert!(
            high_entropy > low_entropy,
            "High difficulty should have higher entropy"
        );
    }

    #[test]
    fn test_fuse_priors() {
        let max_colors = 5;

        let prior1 = WarmstartPrior {
            vertex: 0,
            color_probabilities: vec![0.5, 0.2, 0.1, 0.1, 0.1],
            is_anchor: false,
            anchor_color: None,
        };

        let prior2 = WarmstartPrior {
            vertex: 0,
            color_probabilities: vec![0.2, 0.2, 0.2, 0.2, 0.2],
            is_anchor: false,
            anchor_color: None,
        };

        let fused = fuse_priors(&[&prior1, &prior2], &[0.6, 0.4]);

        // Check sum = 1.0
        let sum: f32 = fused.color_probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // First color should have highest probability (weighted toward prior1)
        let max_prob = fused
            .color_probabilities
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert_eq!(max_prob, &fused.color_probabilities[0]);
    }

    #[test]
    fn test_fuse_priors_anchor_precedence() {
        let prior_anchor = WarmstartPrior {
            vertex: 0,
            color_probabilities: vec![1.0, 0.0, 0.0],
            is_anchor: true,
            anchor_color: Some(0),
        };

        let prior_normal = WarmstartPrior {
            vertex: 0,
            color_probabilities: vec![0.33, 0.33, 0.34],
            is_anchor: false,
            anchor_color: None,
        };

        let fused = fuse_priors(&[&prior_anchor, &prior_normal], &[0.5, 0.5]);

        // Anchor should be preserved
        assert!(fused.is_anchor);
        assert_eq!(fused.anchor_color, Some(0));
    }
}
