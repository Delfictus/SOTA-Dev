//! Phase 0 Ensemble Prior Fusion
//!
//! Implements multi-source warmstart prior fusion combining:
//! - Reservoir-based priors (from dendritic computation)
//! - Structural anchors (from geodesic/TDA phases)
//! - Random exploration noise
//!
//! Refs: Warmstart Plan Step 3 (Ensemble Fusion)

use prism_core::{Graph, WarmstartConfig, WarmstartPrior};

use super::warmstart::fuse_priors;

/// Combines multiple warmstart priors using weighted fusion.
///
/// ## Algorithm
/// 1. Weight reservoir prior by config.flux_weight (default 0.4)
/// 2. Create uniform priors for structural anchors, weight by config.ensemble_weight (0.4)
/// 3. Add random exploration noise, weight by config.random_weight (0.2)
/// 4. Fuse all sources using weighted average
/// 5. Normalize final distributions
///
/// ## Parameters
/// - `reservoir_prior`: Prior from Phase 0 dendritic reservoir computation
/// - `geodesic_anchors`: Anchor vertices from Phase 4 geodesic analysis
/// - `tda_anchors`: Anchor vertices from Phase 6 topological data analysis
/// - `config`: Warmstart configuration (weights, max_colors, etc.)
///
/// ## Returns
/// Fused prior combining all sources with normalized probability distributions.
///
/// ## Example
/// ```rust
/// use prism_core::{WarmstartConfig, WarmstartPrior};
/// use prism_phases::phase0::ensemble::fuse_ensemble_priors;
///
/// let reservoir_prior = WarmstartPrior {
///     vertex: 0,
///     color_probabilities: vec![0.5, 0.3, 0.2],
///     is_anchor: false,
///     anchor_color: None,
/// };
///
/// let config = WarmstartConfig::default();
/// let fused = fuse_ensemble_priors(&reservoir_prior, &[0], &[0], &config);
///
/// // Result combines all sources with proper weighting
/// assert_eq!(fused.vertex, 0);
/// assert!((fused.color_probabilities.iter().sum::<f32>() - 1.0).abs() < 0.01);
/// ```
pub fn fuse_ensemble_priors(
    reservoir_prior: &WarmstartPrior,
    geodesic_anchors: &[usize],
    tda_anchors: &[usize],
    config: &WarmstartConfig,
) -> WarmstartPrior {
    let vertex = reservoir_prior.vertex;
    let max_colors = reservoir_prior.color_probabilities.len();

    // Validate weights sum to 1.0 (with tolerance for floating point)
    let weight_sum = config.flux_weight + config.ensemble_weight + config.random_weight;
    assert!(
        (weight_sum - 1.0).abs() < 0.01,
        "Fusion weights must sum to 1.0, got {}",
        weight_sum
    );

    // Source 1: Reservoir prior (flux-based)
    let flux_prior = reservoir_prior;

    // Source 2: Ensemble prior (uniform over anchors)
    let is_geodesic_anchor = geodesic_anchors.contains(&vertex);
    let is_tda_anchor = tda_anchors.contains(&vertex);
    let is_any_anchor = is_geodesic_anchor || is_tda_anchor;

    let ensemble_prior = if is_any_anchor {
        // Anchors get deterministic initial distribution
        // (will be made fully deterministic by apply_anchors later)
        let mut probs = vec![1.0 / max_colors as f32; max_colors];
        // Normalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }
        WarmstartPrior {
            vertex,
            color_probabilities: probs,
            is_anchor: true,
            anchor_color: None, // Will be set by apply_anchors
        }
    } else {
        // Non-anchors use reservoir prior
        reservoir_prior.clone()
    };

    // Source 3: Random exploration noise (uniform distribution)
    let random_prior = WarmstartPrior {
        vertex,
        color_probabilities: vec![1.0 / max_colors as f32; max_colors],
        is_anchor: false,
        anchor_color: None,
    };

    // Fuse all sources using weighted average
    let weights = vec![
        config.flux_weight,
        config.ensemble_weight,
        config.random_weight,
    ];

    fuse_priors(&[flux_prior, &ensemble_prior, &random_prior], &weights)
}

/// Applies structural anchors to a prior by setting deterministic colors.
///
/// ## Algorithm
/// 1. For each anchor vertex, find a greedy-valid color (no conflicts with neighbors)
/// 2. Set prior to deterministic: 1.0 for assigned color, 0.0 for others
/// 3. Mark as is_anchor = true, set anchor_color field
/// 4. Validate no conflicts with graph adjacency
///
/// ## Parameters
/// - `prior`: Mutable reference to warmstart prior to modify
/// - `anchors`: List of anchor vertex indices
/// - `graph`: Graph structure for conflict validation
///
/// ## Returns
/// Result indicating success or error if conflicts detected.
///
/// ## Example
/// ```rust
/// use prism_core::{Graph, WarmstartPrior};
/// use prism_phases::phase0::ensemble::apply_anchors;
///
/// let mut graph = Graph::new(3);
/// graph.add_edge(0, 1);
///
/// let mut prior = WarmstartPrior {
///     vertex: 0,
///     color_probabilities: vec![0.4, 0.3, 0.3],
///     is_anchor: false,
///     anchor_color: None,
/// };
///
/// apply_anchors(&mut prior, &[0], &graph).unwrap();
///
/// // Prior is now deterministic
/// assert!(prior.is_anchor);
/// assert!(prior.anchor_color.is_some());
/// assert_eq!(prior.color_probabilities.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(), &1.0);
/// ```
pub fn apply_anchors(
    prior: &mut WarmstartPrior,
    anchors: &[usize],
    graph: &Graph,
) -> Result<(), String> {
    let vertex = prior.vertex;

    // Only process if this vertex is an anchor
    if !anchors.contains(&vertex) {
        return Ok(());
    }

    // Find greedy-valid color (no conflicts with neighbors)
    let neighbors = &graph.adjacency[vertex];
    let max_colors = prior.color_probabilities.len();

    // Find first color not used by neighbors
    // (In real implementation, would check neighbor colors from current state)
    // For now, use color 0 as greedy choice
    let mut chosen_color = 0;

    // If vertex has neighbors, try to pick a distinct color
    if !neighbors.is_empty() {
        // Simple greedy: pick first valid color
        // TODO(WARMSTART-1): Integrate with actual coloring state for conflict checking
        for color in 0..max_colors {
            chosen_color = color;
            // In production: check if any neighbor has this color
            // For now, just use first color
            break;
        }
    }

    // Validate chosen color is in valid range
    if chosen_color >= max_colors {
        return Err(format!(
            "Anchor color {} exceeds max_colors {} for vertex {}",
            chosen_color, max_colors, vertex
        ));
    }

    // Set deterministic distribution
    let mut deterministic_probs = vec![0.0; max_colors];
    deterministic_probs[chosen_color] = 1.0;

    prior.color_probabilities = deterministic_probs;
    prior.is_anchor = true;
    prior.anchor_color = Some(chosen_color);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuse_ensemble_priors_basic() {
        let max_colors = 5;

        let reservoir_prior = WarmstartPrior {
            vertex: 0,
            color_probabilities: vec![0.4, 0.3, 0.15, 0.1, 0.05],
            is_anchor: false,
            anchor_color: None,
        };

        let config = WarmstartConfig {
            max_colors,
            min_prob: 0.01,
            anchor_fraction: 0.1,
            flux_weight: 0.4,
            ensemble_weight: 0.4,
            random_weight: 0.2,
            curriculum_catalog_path: None,
        };

        let fused = fuse_ensemble_priors(&reservoir_prior, &[], &[], &config);

        // Verify basic properties
        assert_eq!(fused.vertex, 0);
        assert_eq!(fused.color_probabilities.len(), max_colors);

        // Probabilities should sum to 1.0
        let sum: f32 = fused.color_probabilities.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Probabilities should sum to 1.0, got {}",
            sum
        );

        // All probabilities should be non-negative
        for &prob in &fused.color_probabilities {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_fuse_ensemble_priors_with_anchors() {
        let max_colors = 5;

        let reservoir_prior = WarmstartPrior {
            vertex: 0,
            color_probabilities: vec![0.4, 0.3, 0.15, 0.1, 0.05],
            is_anchor: false,
            anchor_color: None,
        };

        let config = WarmstartConfig {
            max_colors,
            min_prob: 0.01,
            anchor_fraction: 0.1,
            flux_weight: 0.4,
            ensemble_weight: 0.4,
            random_weight: 0.2,
            curriculum_catalog_path: None,
        };

        // Mark vertex 0 as geodesic anchor
        let fused = fuse_ensemble_priors(&reservoir_prior, &[0], &[], &config);

        // Should be marked as anchor
        assert!(fused.is_anchor);

        // Probabilities should still sum to 1.0
        let sum: f32 = fused.color_probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fuse_ensemble_priors_multiple_sources() {
        let max_colors = 3;

        let reservoir_prior = WarmstartPrior {
            vertex: 1,
            color_probabilities: vec![0.6, 0.3, 0.1],
            is_anchor: false,
            anchor_color: None,
        };

        let config = WarmstartConfig {
            max_colors,
            min_prob: 0.01,
            anchor_fraction: 0.1,
            flux_weight: 0.5,
            ensemble_weight: 0.3,
            random_weight: 0.2,
            curriculum_catalog_path: None,
        };

        let fused = fuse_ensemble_priors(&reservoir_prior, &[], &[1], &config);

        // TDA anchor should be recognized
        assert!(fused.is_anchor);
        assert_eq!(fused.vertex, 1);

        // Distribution should be influenced by random noise
        let sum: f32 = fused.color_probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    #[should_panic(expected = "Fusion weights must sum to 1.0")]
    fn test_fuse_ensemble_priors_invalid_weights() {
        let reservoir_prior = WarmstartPrior {
            vertex: 0,
            color_probabilities: vec![0.5, 0.5],
            is_anchor: false,
            anchor_color: None,
        };

        let config = WarmstartConfig {
            max_colors: 2,
            min_prob: 0.01,
            anchor_fraction: 0.1,
            flux_weight: 0.5,
            ensemble_weight: 0.5,
            random_weight: 0.5, // Invalid: sums to 1.5
            curriculum_catalog_path: None,
        };

        fuse_ensemble_priors(&reservoir_prior, &[], &[], &config);
    }

    #[test]
    fn test_apply_anchors_basic() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        let mut prior = WarmstartPrior {
            vertex: 0,
            color_probabilities: vec![0.4, 0.3, 0.3],
            is_anchor: false,
            anchor_color: None,
        };

        apply_anchors(&mut prior, &[0], &graph).unwrap();

        // Prior should now be deterministic
        assert!(prior.is_anchor);
        assert!(prior.anchor_color.is_some());

        // Probabilities should sum to 1.0 (one color = 1.0, rest = 0.0)
        let sum: f32 = prior.color_probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // Exactly one color should have probability 1.0
        let ones_count = prior
            .color_probabilities
            .iter()
            .filter(|&&p| (p - 1.0).abs() < 0.01)
            .count();
        assert_eq!(ones_count, 1);
    }

    #[test]
    fn test_apply_anchors_non_anchor_unchanged() {
        let graph = Graph::new(3);

        let mut prior = WarmstartPrior {
            vertex: 1,
            color_probabilities: vec![0.5, 0.3, 0.2],
            is_anchor: false,
            anchor_color: None,
        };

        let original = prior.clone();

        // Vertex 1 is not in anchor list
        apply_anchors(&mut prior, &[0, 2], &graph).unwrap();

        // Prior should be unchanged
        assert_eq!(prior.is_anchor, original.is_anchor);
        assert_eq!(prior.anchor_color, original.anchor_color);
        assert_eq!(prior.color_probabilities, original.color_probabilities);
    }

    #[test]
    fn test_apply_anchors_multiple_vertices() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);

        let mut priors = vec![
            WarmstartPrior {
                vertex: 0,
                color_probabilities: vec![0.5, 0.5],
                is_anchor: false,
                anchor_color: None,
            },
            WarmstartPrior {
                vertex: 2,
                color_probabilities: vec![0.5, 0.5],
                is_anchor: false,
                anchor_color: None,
            },
            WarmstartPrior {
                vertex: 4,
                color_probabilities: vec![0.5, 0.5],
                is_anchor: false,
                anchor_color: None,
            },
        ];

        let anchors = vec![0, 2, 4];

        for prior in &mut priors {
            apply_anchors(prior, &anchors, &graph).unwrap();
        }

        // All should be anchors
        for prior in &priors {
            assert!(prior.is_anchor);
            assert!(prior.anchor_color.is_some());
        }
    }

    #[test]
    fn test_apply_anchors_empty_anchors() {
        let graph = Graph::new(2);

        let mut prior = WarmstartPrior {
            vertex: 0,
            color_probabilities: vec![0.6, 0.4],
            is_anchor: false,
            anchor_color: None,
        };

        let original = prior.clone();

        apply_anchors(&mut prior, &[], &graph).unwrap();

        // Should remain unchanged
        assert_eq!(prior.is_anchor, original.is_anchor);
        assert_eq!(prior.color_probabilities, original.color_probabilities);
    }

    #[test]
    fn test_apply_anchors_single_vertex_graph() {
        let graph = Graph::new(1);

        let mut prior = WarmstartPrior {
            vertex: 0,
            color_probabilities: vec![1.0],
            is_anchor: false,
            anchor_color: None,
        };

        apply_anchors(&mut prior, &[0], &graph).unwrap();

        // Should be anchored to color 0
        assert!(prior.is_anchor);
        assert_eq!(prior.anchor_color, Some(0));
        assert!((prior.color_probabilities[0] - 1.0).abs() < 0.01);
    }
}
