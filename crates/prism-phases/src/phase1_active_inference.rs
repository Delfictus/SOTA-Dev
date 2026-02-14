//! Phase 1: Active Inference with GPU Acceleration
//!
//! Implements uncertainty-driven vertex ordering for graph coloring using
//! Active Inference variational free energy minimization.
//!
//! ## Specification Compliance
//! - Implements PRISM GPU Plan §4.1: Phase Controllers
//! - Full Active Inference algorithm (from foundation/active_inference)
//! - GPU-accelerated policy computation (10-20x CPU speedup)
//! - PhaseController trait with telemetry
//!
//! ## Algorithm
//! 1. Compute Active Inference policy on GPU (EFE, uncertainty)
//! 2. Order vertices by uncertainty (descending)
//! 3. Apply greedy coloring with uncertainty-driven exploration
//! 4. Emit telemetry: EFE, VFE, uncertainty metrics
//!
//! ## Performance Targets (from foundation/active_inference/controller.rs)
//! - Action selection: <2ms per action
//! - Full policy computation: <50ms for 250 vertices
//! - Efficient exploration-exploitation balance

use prism_core::{Graph, PhaseContext, PhaseController, PhaseOutcome, PhaseTelemetry, PrismError};
use prism_gpu::{ActiveInferenceGpu, ActiveInferencePolicy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for Phase 1 Active Inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase1Config {
    /// Prior belief precision
    #[serde(default = "default_prior_precision")]
    pub prior_precision: f64,

    /// Likelihood precision
    #[serde(default = "default_likelihood_precision")]
    pub likelihood_precision: f64,

    /// Learning rate for belief updates
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,

    /// Free energy convergence threshold
    #[serde(default = "default_free_energy_threshold")]
    pub free_energy_threshold: f64,

    /// Maximum inference iterations
    #[serde(default = "default_num_iterations")]
    pub num_iterations: usize,

    /// Hidden state dimensionality
    #[serde(default = "default_hidden_states")]
    pub hidden_states: usize,

    /// Policy planning depth
    #[serde(default = "default_policy_depth")]
    pub policy_depth: usize,

    /// Exploration bonus coefficient
    #[serde(default = "default_exploration_bonus")]
    pub exploration_bonus: f64,

    /// Enable GPU acceleration
    #[serde(default = "default_gpu_enabled")]
    pub gpu_enabled: bool,
}

// Default value functions
fn default_prior_precision() -> f64 {
    1.0
}
fn default_likelihood_precision() -> f64 {
    2.0
}
fn default_learning_rate() -> f64 {
    0.001
}
fn default_free_energy_threshold() -> f64 {
    0.01
}
fn default_num_iterations() -> usize {
    1000
}
fn default_hidden_states() -> usize {
    64
}
fn default_policy_depth() -> usize {
    3
}
fn default_exploration_bonus() -> f64 {
    0.1
}
fn default_gpu_enabled() -> bool {
    true
}

impl Default for Phase1Config {
    fn default() -> Self {
        Self {
            prior_precision: default_prior_precision(),
            likelihood_precision: default_likelihood_precision(),
            learning_rate: default_learning_rate(),
            free_energy_threshold: default_free_energy_threshold(),
            num_iterations: default_num_iterations(),
            hidden_states: default_hidden_states(),
            policy_depth: default_policy_depth(),
            exploration_bonus: default_exploration_bonus(),
            gpu_enabled: default_gpu_enabled(),
        }
    }
}

/// Phase 1 controller with GPU-accelerated Active Inference
pub struct Phase1ActiveInference {
    /// GPU Active Inference engine (optional for CPU fallback)
    ai_gpu: Option<Arc<ActiveInferenceGpu>>,

    /// Telemetry from last execution
    last_telemetry: Phase1Telemetry,

    /// Configuration parameters
    prior_precision: f64,
    likelihood_precision: f64,
    learning_rate: f64,
    free_energy_threshold: f64,
    num_iterations: usize,
    hidden_states: usize,
    policy_depth: usize,
    exploration_bonus: f64,
}

/// Telemetry structure for Phase 1
///
/// Captures Active Inference metrics and coloring performance.
#[derive(Debug, Clone, Default)]
struct Phase1Telemetry {
    /// Mean Expected Free Energy across vertices
    mean_efe: f64,

    /// Mean uncertainty across vertices
    mean_uncertainty: f64,

    /// Variational Free Energy (if computed)
    vfe: f64,

    /// Number of colors used
    num_colors: usize,

    /// Execution time (ms)
    execution_time_ms: f64,

    /// Policy computation time (ms)
    policy_time_ms: f64,

    /// Coloring time (ms)
    coloring_time_ms: f64,
}

impl Phase1ActiveInference {
    /// Create new Phase 1 controller (CPU fallback mode)
    pub fn new() -> Self {
        let default_config = Phase1Config::default();
        Self {
            ai_gpu: None,
            last_telemetry: Phase1Telemetry::default(),
            prior_precision: default_config.prior_precision,
            likelihood_precision: default_config.likelihood_precision,
            learning_rate: default_config.learning_rate,
            free_energy_threshold: default_config.free_energy_threshold,
            num_iterations: default_config.num_iterations,
            hidden_states: default_config.hidden_states,
            policy_depth: default_config.policy_depth,
            exploration_bonus: default_config.exploration_bonus,
        }
    }

    /// Create Phase 1 controller with custom config (CPU mode)
    pub fn with_config(config: Phase1Config) -> Self {
        log::info!(
            "Phase1: Initializing with custom TOML config: prior_precision={:.3}, learning_rate={:.4}, policy_depth={}",
            config.prior_precision, config.learning_rate, config.policy_depth
        );

        Self {
            ai_gpu: None,
            last_telemetry: Phase1Telemetry::default(),
            prior_precision: config.prior_precision,
            likelihood_precision: config.likelihood_precision,
            learning_rate: config.learning_rate,
            free_energy_threshold: config.free_energy_threshold,
            num_iterations: config.num_iterations,
            hidden_states: config.hidden_states,
            policy_depth: config.policy_depth,
            exploration_bonus: config.exploration_bonus,
        }
    }

    /// Create Phase 1 controller with custom config and GPU acceleration
    pub fn with_config_and_gpu(config: Phase1Config, ai_gpu: Arc<ActiveInferenceGpu>) -> Self {
        log::info!(
            "Phase1: Initializing with custom TOML config and GPU: prior_precision={:.3}, gpu_enabled={}",
            config.prior_precision, config.gpu_enabled
        );

        Self {
            ai_gpu: if config.gpu_enabled {
                Some(ai_gpu)
            } else {
                None
            },
            last_telemetry: Phase1Telemetry::default(),
            prior_precision: config.prior_precision,
            likelihood_precision: config.likelihood_precision,
            learning_rate: config.learning_rate,
            free_energy_threshold: config.free_energy_threshold,
            num_iterations: config.num_iterations,
            hidden_states: config.hidden_states,
            policy_depth: config.policy_depth,
            exploration_bonus: config.exploration_bonus,
        }
    }

    /// Create Phase 1 controller with GPU acceleration (uses default config)
    ///
    /// # Arguments
    /// * `ai_gpu` - GPU Active Inference engine from prism-gpu
    pub fn new_with_gpu(ai_gpu: Arc<ActiveInferenceGpu>) -> Self {
        let default_config = Phase1Config::default();
        Self::with_config_and_gpu(default_config, ai_gpu)
    }

    /// Compute Active Inference policy (GPU or CPU fallback)
    fn compute_policy(
        &self,
        graph: &Graph,
        coloring: &[usize],
    ) -> Result<ActiveInferencePolicy, PrismError> {
        if let Some(ref ai_gpu) = self.ai_gpu {
            // GPU path
            ai_gpu.compute_policy(graph, coloring)
        } else {
            // CPU fallback: uniform uncertainty
            log::warn!("[Phase1] GPU not available, using CPU fallback (uniform uncertainty)");
            Ok(self.cpu_fallback_policy(graph.num_vertices))
        }
    }

    /// CPU fallback policy (uniform uncertainty)
    fn cpu_fallback_policy(&self, n: usize) -> ActiveInferencePolicy {
        let uniform = 1.0 / n as f64;
        ActiveInferencePolicy {
            uncertainty: vec![uniform; n],
            expected_free_energy: vec![1.0; n],
            pragmatic_value: vec![0.5; n],
            epistemic_value: vec![0.5; n],
        }
    }

    /// Greedy coloring with Active Inference vertex ordering
    ///
    /// Orders vertices by uncertainty (descending) and applies greedy coloring.
    ///
    /// Spec reference: foundation/prct-core/src/gpu_active_inference.rs:53-256
    fn greedy_coloring_with_ai(
        &self,
        graph: &Graph,
        policy: &ActiveInferencePolicy,
    ) -> Result<Vec<usize>, PrismError> {
        let start = std::time::Instant::now();
        let n = graph.num_vertices;

        // Get vertex order from Active Inference policy
        let vertex_order = policy.vertex_order();

        // Initialize coloring (usize::MAX = uncolored)
        let mut coloring = vec![usize::MAX; n];

        // Greedy coloring in uncertainty order
        for &v in &vertex_order {
            // Find neighbors' colors
            let mut neighbor_colors = vec![false; n];
            for &neighbor in &graph.adjacency[v] {
                if coloring[neighbor] != usize::MAX {
                    neighbor_colors[coloring[neighbor]] = true;
                }
            }

            // Assign smallest available color
            let color = neighbor_colors
                .iter()
                .position(|&used| !used)
                .unwrap_or(neighbor_colors.len());

            coloring[v] = color;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        log::debug!("[Phase1] Coloring completed in {:.2}ms", elapsed);

        Ok(coloring)
    }

    /// Count number of colors used
    fn count_colors(coloring: &[usize]) -> usize {
        coloring
            .iter()
            .filter(|&&c| c != usize::MAX)
            .max()
            .map(|&m| m + 1)
            .unwrap_or(0)
    }

    /// Validate coloring is conflict-free
    fn validate_coloring(graph: &Graph, coloring: &[usize]) -> Result<(), PrismError> {
        for v in 0..graph.num_vertices {
            for &neighbor in &graph.adjacency[v] {
                if coloring[v] == coloring[neighbor] && coloring[v] != usize::MAX {
                    return Err(PrismError::validation(format!(
                        "Conflict detected: vertices {} and {} both have color {}",
                        v, neighbor, coloring[v]
                    )));
                }
            }
        }
        Ok(())
    }

    /// Applies geometry prediction error to policy to increase exploration.
    ///
    /// Modulates uncertainty values based on geometry stress:
    /// - Low error (< 0.25): Decrease uncertainty slightly (exploitation)
    /// - Medium error (0.25-0.50): Keep uncertainty unchanged
    /// - High error (0.50-0.80): Increase uncertainty moderately (exploration)
    /// - Critical error (> 0.80): Increase uncertainty significantly (high exploration)
    ///
    /// This implements the metaphysical coupling feedback loop where geometric
    /// stress drives Active Inference to explore more aggressively.
    fn apply_geometry_prediction_error(&self, policy: &mut ActiveInferencePolicy, error: f64) {
        let adjustment_factor = if error > 0.8 {
            1.5 // Critical: +50% uncertainty
        } else if error > 0.5 {
            1.25 // High: +25% uncertainty
        } else if error > 0.25 {
            1.0 // Medium: no change
        } else {
            0.9 // Low: -10% uncertainty (favor exploitation)
        };

        // Apply adjustment to all uncertainty values
        for uncertainty in &mut policy.uncertainty {
            *uncertainty *= adjustment_factor;
        }

        // Clamp to [0.0, 1.0] range
        for uncertainty in &mut policy.uncertainty {
            *uncertainty = uncertainty.clamp(0.0, 1.0);
        }

        log::info!(
            "[Phase1] Applied geometry prediction error ({:.3}) with adjustment factor {:.2}x. New mean_uncertainty={:.4}",
            error,
            adjustment_factor,
            policy.mean_uncertainty()
        );
    }

    /// Applies dendritic reservoir difficulty to policy for targeted exploration.
    ///
    /// Uses per-vertex difficulty from Phase 0 to modulate uncertainty:
    /// - High difficulty vertices (> 0.7): Boost uncertainty by 1.3x
    /// - Medium difficulty (0.3-0.7): Normal uncertainty
    /// - Low difficulty (< 0.3): Slight reduction (0.9x)
    ///
    /// This implements the Phase 0 → Phase 1 coupling where neuromorphic
    /// difficulty estimation guides Active Inference exploration.
    fn apply_dendritic_difficulty(&self, policy: &mut ActiveInferencePolicy, difficulty: &[f32]) {
        if difficulty.len() != policy.uncertainty.len() {
            log::warn!(
                "[Phase1] Dendritic difficulty length mismatch: {} vs {}",
                difficulty.len(),
                policy.uncertainty.len()
            );
            return;
        }

        let mut boosted = 0usize;
        let mut reduced = 0usize;

        for (i, &diff) in difficulty.iter().enumerate() {
            let adjustment = if diff > 0.7 {
                boosted += 1;
                1.3 // High difficulty: boost exploration
            } else if diff < 0.3 {
                reduced += 1;
                0.9 // Low difficulty: slight exploitation
            } else {
                1.0 // Normal
            };

            policy.uncertainty[i] *= adjustment;
            policy.uncertainty[i] = policy.uncertainty[i].clamp(0.0, 1.0);
        }

        log::info!(
            "[Phase1] Dendritic coupling: {} vertices boosted, {} reduced, mean_uncertainty={:.4}",
            boosted,
            reduced,
            policy.mean_uncertainty()
        );
    }
}

impl Default for Phase1ActiveInference {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseController for Phase1ActiveInference {
    fn execute(
        &mut self,
        graph: &Graph,
        context: &mut PhaseContext,
    ) -> Result<PhaseOutcome, PrismError> {
        let start = std::time::Instant::now();
        log::info!("[Phase1] Starting Active Inference coloring");

        // Read geometry metrics from context for metaphysical coupling
        let geometry_prediction_error = if let Some(ref geom) = context.geometry_metrics {
            // Compute geometry-based prediction error: stress_scalar * overlap_density
            let error = (geom.stress_scalar * geom.overlap_density) as f64;
            log::info!(
                "[Phase1] Geometry coupling active: stress={:.3}, overlap={:.3}, prediction_error={:.3}",
                geom.stress_scalar,
                geom.overlap_density,
                error
            );

            // Log adjustment suggestions based on thresholds
            if geom.stress_scalar > 0.8 {
                log::warn!(
                    "[Phase1] CRITICAL geometry stress detected ({:.3}). Suggesting exploration boost.",
                    geom.stress_scalar
                );
            } else if geom.stress_scalar > 0.5 {
                log::info!(
                    "[Phase1] High geometry stress detected ({:.3}). Suggesting moderate exploration increase.",
                    geom.stress_scalar
                );
            } else {
                log::debug!(
                    "[Phase1] Low geometry stress ({:.3}). Normal exploration mode.",
                    geom.stress_scalar
                );
            }

            Some(error)
        } else {
            log::debug!("[Phase1] No geometry metrics available, proceeding without coupling");
            None
        };

        // Initialize coloring (all uncolored)
        let initial_coloring = vec![usize::MAX; graph.num_vertices];

        // Compute Active Inference policy
        let policy_start = std::time::Instant::now();
        let mut policy = self.compute_policy(graph, &initial_coloring)?;
        let policy_time = policy_start.elapsed().as_secs_f64() * 1000.0;

        // Early-phase geometry seeding: Generate synthetic geometry metrics from policy
        // This enables reflexive coupling before Phase 4/6 completes
        if context.geometry_metrics.is_none() {
            let mean_uncertainty = policy.mean_uncertainty() as f32;
            let mean_efe = policy.mean_efe() as f32;
            // Use EFE as difficulty proxy (high EFE = hard to color)
            let mean_difficulty = (mean_efe / 10.0).clamp(0.0, 1.0);

            let vertex_uncertainties: Vec<f32> =
                policy.uncertainty.iter().map(|&u| u as f32).collect();

            let early_geometry = prism_core::GeometryTelemetry::from_early_phase_signals(
                mean_uncertainty,
                mean_difficulty,
                &vertex_uncertainties,
            );

            log::info!(
                "[Phase1] Early-phase geometry seeding: stress={:.3}, overlap={:.3}, {} hotspots",
                early_geometry.stress_scalar,
                early_geometry.overlap_density,
                early_geometry.hotspot_count()
            );

            context.update_geometry_metrics(early_geometry);
        }

        // Apply geometry prediction error to policy if available
        if let Some(error) = geometry_prediction_error {
            self.apply_geometry_prediction_error(&mut policy, error);
        }

        // Apply dendritic reservoir difficulty from Phase 0
        if let Some(difficulty) = context.dendritic_difficulty() {
            log::info!(
                "[Phase1] Dendritic metrics available: {} vertices, mean_difficulty={:.3}",
                difficulty.len(),
                context.mean_difficulty()
            );
            self.apply_dendritic_difficulty(&mut policy, difficulty);
        } else {
            log::debug!("[Phase1] No dendritic metrics available (Phase 0 may not have run)");
        }

        log::info!(
            "[Phase1] Policy computed: mean_uncertainty={:.4}, mean_efe={:.4}, time={:.2}ms",
            policy.mean_uncertainty(),
            policy.mean_efe(),
            policy_time
        );

        // Apply greedy coloring with AI vertex ordering
        let coloring_start = std::time::Instant::now();
        let coloring = self.greedy_coloring_with_ai(graph, &policy)?;
        let coloring_time = coloring_start.elapsed().as_secs_f64() * 1000.0;

        // Validate coloring
        Self::validate_coloring(graph, &coloring)?;

        let num_colors = Self::count_colors(&coloring);
        let total_time = start.elapsed().as_secs_f64() * 1000.0;

        log::info!(
            "[Phase1] Coloring complete: {} colors, {:.2}ms total",
            num_colors,
            total_time
        );

        // Store telemetry
        self.last_telemetry = Phase1Telemetry {
            mean_efe: policy.mean_efe(),
            mean_uncertainty: policy.mean_uncertainty(),
            vfe: 0.0, // Not computed in Phase 1 (would require full inference loop)
            num_colors,
            execution_time_ms: total_time,
            policy_time_ms: policy_time,
            coloring_time_ms: coloring_time,
        };

        // Store coloring in context scratch space
        context
            .scratch
            .insert("phase1_coloring".to_string(), Box::new(coloring.clone()));
        context
            .scratch
            .insert("phase1_num_colors".to_string(), Box::new(num_colors));
        context
            .scratch
            .insert("phase1_policy".to_string(), Box::new(policy));

        // Store geometry prediction error in context for telemetry
        if let Some(error) = geometry_prediction_error {
            context.scratch.insert(
                "phase1_geometry_prediction_error".to_string(),
                Box::new(error),
            );
        }

        Ok(PhaseOutcome::success())
    }

    fn name(&self) -> &'static str {
        "Phase1-ActiveInference"
    }

    fn telemetry(&self) -> &dyn PhaseTelemetry {
        self
    }
}

impl PhaseTelemetry for Phase1ActiveInference {
    fn metrics(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("efe".to_string(), self.last_telemetry.mean_efe);
        m.insert(
            "uncertainty".to_string(),
            self.last_telemetry.mean_uncertainty,
        );
        m.insert("vfe".to_string(), self.last_telemetry.vfe);
        m.insert(
            "num_colors".to_string(),
            self.last_telemetry.num_colors as f64,
        );
        m.insert(
            "execution_time_ms".to_string(),
            self.last_telemetry.execution_time_ms,
        );
        m.insert(
            "policy_time_ms".to_string(),
            self.last_telemetry.policy_time_ms,
        );
        m.insert(
            "coloring_time_ms".to_string(),
            self.last_telemetry.coloring_time_ms,
        );
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        // Triangle graph: 0-1, 1-2, 2-0
        Graph::from_edges(3, vec![(0, 1), (1, 2), (2, 0)])
    }

    #[test]
    fn test_phase1_cpu_fallback() {
        let mut phase1 = Phase1ActiveInference::new();
        let graph = create_test_graph();
        let mut context = PhaseContext::new();

        let result = phase1.execute(&graph, &mut context);
        assert!(result.is_ok(), "Phase1 execution failed: {:?}", result);

        let outcome = result.unwrap();
        assert!(outcome.is_success(), "Phase1 did not succeed");

        // Check coloring in context scratch space
        let coloring_box = context.scratch.get("phase1_coloring").unwrap();
        let coloring = coloring_box.downcast_ref::<Vec<usize>>().unwrap();
        assert_eq!(coloring.len(), 3);

        // Triangle requires 3 colors
        let num_colors = Phase1ActiveInference::count_colors(&coloring);
        assert_eq!(num_colors, 3);

        // Validate no conflicts
        assert!(Phase1ActiveInference::validate_coloring(&graph, &coloring).is_ok());
    }

    #[test]
    fn test_count_colors() {
        let coloring = vec![0, 1, 2, 1, 0];
        assert_eq!(Phase1ActiveInference::count_colors(&coloring), 3);

        let uncolored = vec![usize::MAX, usize::MAX];
        assert_eq!(Phase1ActiveInference::count_colors(&uncolored), 0);
    }

    #[test]
    fn test_validate_coloring_success() {
        let graph = create_test_graph();
        let coloring = vec![0, 1, 2]; // Valid 3-coloring of triangle
        assert!(Phase1ActiveInference::validate_coloring(&graph, &coloring).is_ok());
    }

    #[test]
    fn test_validate_coloring_conflict() {
        let graph = create_test_graph();
        let coloring = vec![0, 0, 1]; // Conflict: vertices 0 and 1 both color 0
        assert!(Phase1ActiveInference::validate_coloring(&graph, &coloring).is_err());
    }

    #[test]
    fn test_telemetry() {
        let phase1 = Phase1ActiveInference {
            ai_gpu: None,
            last_telemetry: Phase1Telemetry {
                mean_efe: 1.23,
                mean_uncertainty: 0.56,
                vfe: 0.0,
                num_colors: 42,
                execution_time_ms: 12.34,
                policy_time_ms: 5.67,
                coloring_time_ms: 6.67,
            },
        };

        let metrics = phase1.metrics();
        assert_eq!(metrics.get("efe"), Some(&1.23));
        assert_eq!(metrics.get("uncertainty"), Some(&0.56));
        assert_eq!(metrics.get("num_colors"), Some(&42.0));
    }
}
