//! WHCR Phase Controller - Multi-Phase Integrated Version
//!
//! This is the updated WHCRPhaseController that supports invocation from
//! multiple PRISM phases (2, 3, 5, 7) with phase-specific configurations.
//!
//! # Key Changes from Original
//! - `CallingPhase` determines behavior instead of single config
//! - `GeometryAccumulator` provides GPU-resident geometry buffers
//! - Phase-specific telemetry tracks color budget compliance
//! - Factory methods for each calling phase
//!
//! # Usage Pattern
//! ```ignore
//! // Create once per pipeline run
//! let geometry = GeometryAccumulator::new(gpu_context.device(), num_vertices)?;
//!
//! // After Phase 2
//! let whcr_p2 = WHCRPhaseController::for_phase2(gpu_context.clone(), graph)?;
//! whcr_p2.execute_with_geometry(graph, context, &geometry)?;
//!
//! // After Phase 3 (geometry now includes Phase 4 stress)
//! let whcr_p3 = WHCRPhaseController::for_phase3(gpu_context.clone(), graph)?;
//! whcr_p3.execute_with_geometry(graph, context, &geometry)?;
//! ```

use anyhow::Result;
use prism_core::{
    ColoringSolution, Graph, PhaseContext, PhaseController, PhaseOutcome, PhaseTelemetry,
    PrismError,
};
use prism_gpu::{DendriticWhcrGpu, GpuContext, WhcrGpu as WaveletHierarchicalRepairGpu};
use std::collections::HashMap;
use std::sync::Arc;

// Multi-phase integration types
use prism_whcr::{
    CallingPhase, GeometryAccumulator, GeometryBuffers, GeometrySources, GeometryWeights,
    PhaseWHCRConfig,
};

/// WHCR Phase Controller with multi-phase support
pub struct WHCRPhaseController {
    /// GPU WHCR implementation
    whcr_gpu: Option<WaveletHierarchicalRepairGpu>,

    /// GPU dendritic reservoir
    dendritic: Option<DendriticWhcrGpu>,

    /// Phase-specific configuration
    config: PhaseWHCRConfig,

    /// Telemetry data
    telemetry: WHCRTelemetry,

    /// GPU context for potential reinitialization
    gpu_context: Option<Arc<GpuContext>>,
}

/// WHCR telemetry with phase-aware metrics
struct WHCRTelemetry {
    calling_phase: CallingPhase,
    initial_conflicts: usize,
    final_conflicts: usize,
    conflicts_reduced: usize,
    repair_iterations: usize,
    precision_level: u32,
    colors_added: i32,
    budget_violated: bool,
    geometry_used: String,
}

impl PhaseTelemetry for WHCRTelemetry {
    fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("calling_phase".to_string(), self.calling_phase as u8 as f64);
        metrics.insert(
            "initial_conflicts".to_string(),
            self.initial_conflicts as f64,
        );
        metrics.insert("final_conflicts".to_string(), self.final_conflicts as f64);
        metrics.insert(
            "conflicts_reduced".to_string(),
            self.conflicts_reduced as f64,
        );
        metrics.insert(
            "repair_iterations".to_string(),
            self.repair_iterations as f64,
        );
        metrics.insert("precision_level".to_string(), self.precision_level as f64);
        metrics.insert("colors_added".to_string(), self.colors_added as f64);
        metrics.insert(
            "budget_violated".to_string(),
            if self.budget_violated { 1.0 } else { 0.0 },
        );
        metrics
    }
}

impl WHCRPhaseController {
    // =========================================================================
    // Factory Methods for Each Calling Phase
    // =========================================================================

    /// Create controller for Phase 2 (Thermodynamic) invocation
    ///
    /// - High aggressiveness: can add up to 5 colors
    /// - Precision: f32 for fast exploration
    /// - Expected geometry: Phase 0, Phase 1
    pub fn for_phase2(gpu_context: Arc<GpuContext>, graph: &Graph) -> Result<Self, PrismError> {
        Self::new_with_phase(gpu_context, graph, CallingPhase::Phase2Thermodynamic)
    }

    /// Create controller for Phase 3 (Quantum) invocation
    ///
    /// - Medium aggressiveness: can add up to 3 colors
    /// - Precision: mixed (f32 coarse, f64 fine)
    /// - Expected geometry: Phase 0, 1, 4
    pub fn for_phase3(gpu_context: Arc<GpuContext>, graph: &Graph) -> Result<Self, PrismError> {
        Self::new_with_phase(gpu_context, graph, CallingPhase::Phase3Quantum)
    }

    /// Create controller for Phase 5 (Membrane) invocation
    ///
    /// - Low aggressiveness: can add up to 2 colors
    /// - Precision: f64 with geometry coupling
    /// - Expected geometry: All phases
    pub fn for_phase5(gpu_context: Arc<GpuContext>, graph: &Graph) -> Result<Self, PrismError> {
        Self::new_with_phase(gpu_context, graph, CallingPhase::Phase5Membrane)
    }

    /// Create controller for Phase 7 (Ensemble) invocation
    ///
    /// - STRICT: cannot add any colors
    /// - Precision: f64 with full geometry coupling
    /// - Maximum iterations
    /// - Expected geometry: All phases
    pub fn for_phase7(gpu_context: Arc<GpuContext>, graph: &Graph) -> Result<Self, PrismError> {
        Self::new_with_phase(gpu_context, graph, CallingPhase::Phase7Ensemble)
    }

    /// Create standalone controller (for testing or manual use)
    pub fn standalone(gpu_context: Arc<GpuContext>, graph: &Graph) -> Result<Self, PrismError> {
        Self::new_with_phase(gpu_context, graph, CallingPhase::Standalone)
    }

    // =========================================================================
    // Core Constructor
    // =========================================================================

    /// Create new WHCR controller for a specific calling phase
    fn new_with_phase(
        gpu_context: Arc<GpuContext>,
        graph: &Graph,
        phase: CallingPhase,
    ) -> Result<Self, PrismError> {
        log::info!(
            "Initializing {} for {} vertices",
            phase.name(),
            graph.num_vertices
        );

        let config = PhaseWHCRConfig::for_phase(phase, graph.num_vertices);

        // Create WHCR GPU instance
        let whcr_gpu = WaveletHierarchicalRepairGpu::new(
            gpu_context.device().clone(),
            graph.num_vertices,
            &graph.adjacency,
        )
        .map_err(|e| {
            PrismError::gpu(
                phase.name(),
                format!("Failed to initialize WHCR GPU: {}", e),
            )
        })?;

        // Create dendritic reservoir if enabled
        // Resolved TODO(GPU-WHCR-1): Fixed kernel parameter passing
        let dendritic = if config.enable_dendritic && phase.update_reservoir() {
            let (row_ptr, col_idx) = adjacency_to_csr(&graph.adjacency);
            let initial_conflicts = vec![0.0f32; graph.num_vertices];

            match DendriticWhcrGpu::new(
                gpu_context.device().clone(),
                graph.num_vertices,
                &row_ptr,
                &col_idx,
                &initial_conflicts,
                config.reservoir_influence,
            ) {
                Ok(reservoir) => {
                    log::info!(
                        "{}: Dendritic reservoir initialized with {} vertices, influence: {:.2}",
                        phase.name(),
                        graph.num_vertices,
                        config.reservoir_influence
                    );
                    Some(reservoir)
                }
                Err(e) => {
                    log::warn!(
                        "{}: Failed to initialize dendritic reservoir: {}. Continuing without neuromorphic co-processing.",
                        phase.name(),
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        let precision = config.precision();
        let iterations = config.iterations();

        log::info!(
            "{} initialized (dendritic: {}, precision: {}, max_iter: {})",
            phase.name(),
            dendritic.is_some(),
            precision,
            iterations
        );

        Ok(Self {
            whcr_gpu: Some(whcr_gpu),
            dendritic,
            config,
            telemetry: WHCRTelemetry {
                calling_phase: phase,
                initial_conflicts: 0,
                final_conflicts: 0,
                conflicts_reduced: 0,
                repair_iterations: 0,
                precision_level: precision,
                colors_added: 0,
                budget_violated: false,
                geometry_used: String::new(),
            },
            gpu_context: Some(gpu_context),
        })
    }

    /// Create CPU-only fallback (no GPU)
    pub fn cpu_fallback(phase: CallingPhase, num_vertices: usize) -> Self {
        log::warn!("{} in CPU-only fallback mode", phase.name());
        Self {
            whcr_gpu: None,
            dendritic: None,
            config: PhaseWHCRConfig::for_phase(phase, num_vertices),
            telemetry: WHCRTelemetry {
                calling_phase: phase,
                initial_conflicts: 0,
                final_conflicts: 0,
                conflicts_reduced: 0,
                repair_iterations: 0,
                precision_level: 0,
                colors_added: 0,
                budget_violated: false,
                geometry_used: "CPU fallback".to_string(),
            },
            gpu_context: None,
        }
    }

    // =========================================================================
    // Execution with Geometry
    // =========================================================================

    /// Execute repair with geometry from accumulator
    ///
    /// This is the primary execution method for multi-phase integration.
    /// It uses geometry buffers accumulated from prior phases.
    pub fn execute_with_geometry(
        &mut self,
        graph: &Graph,
        context: &mut PhaseContext,
        geometry: &GeometryAccumulator,
    ) -> Result<PhaseOutcome, PrismError> {
        let phase = self.config.calling_phase;

        // Validate geometry availability
        if !geometry.satisfies_phase(phase) {
            log::warn!(
                "{}: Geometry requirements not fully satisfied. Expected: {:?}, Available: {:?}",
                phase.name(),
                phase.expected_geometry(),
                geometry.available()
            );
            // Continue anyway - we'll use what's available
        }

        // Get current solution
        let solution = context
            .best_solution
            .as_mut()
            .ok_or_else(|| PrismError::phase(phase.name(), "No solution available in context"))?;

        let initial_conflicts = count_conflicts(&solution.colors, &graph.adjacency);
        self.telemetry.initial_conflicts = initial_conflicts;

        log::info!(
            "{}: Starting repair - {} colors, {} conflicts, geometry: {}",
            phase.name(),
            solution.chromatic_number,
            initial_conflicts,
            geometry.summary()
        );

        // Early exit if already conflict-free
        if initial_conflicts == 0 {
            return Ok(PhaseOutcome::Success {
                message: format!(
                    "{}: Already zero conflicts with {} colors",
                    phase.name(),
                    solution.chromatic_number
                ),
                telemetry: self.build_telemetry_map(),
            });
        }

        // Execute GPU repair
        if let Some(whcr) = &mut self.whcr_gpu {
            // Get geometry buffers (GPU-resident references)
            let buffers = geometry.get_gpu_buffers();

            // Note: Geometry is already synchronized through GeometrySynchronizer
            // The buffers contain GPU-resident CudaSlice references
            // WHCR will access these directly through the buffers parameter
            if self.dendritic.is_some() {
                log::debug!(
                    "{}: Dendritic reservoir active - hotspots available in geometry",
                    phase.name()
                );
            }

            // Log available geometry
            if buffers.has_any() {
                log::debug!("{}: Geometry available - hotspots: {}, stress: {}, persistence: {}, beliefs: {}",
                    phase.name(),
                    buffers.hotspot_mask.is_some(),
                    buffers.stress_scores.is_some(),
                    buffers.persistence_scores.is_some(),
                    buffers.belief_distribution.is_some()
                );
            }

            // Execute phase-aware repair
            let result = prism_whcr::repair_with_phase_config(
                whcr,
                &mut solution.colors,
                solution.chromatic_number,
                &self.config,
                &buffers,
            )
            .map_err(|e| PrismError::gpu(phase.name(), format!("WHCR repair failed: {}", e)))?;

            // Update solution
            let final_conflicts = count_conflicts(&solution.colors, &graph.adjacency);
            solution.conflicts = final_conflicts;
            solution.chromatic_number = result.final_colors;

            // Update telemetry
            self.telemetry.final_conflicts = final_conflicts;
            self.telemetry.conflicts_reduced = initial_conflicts.saturating_sub(final_conflicts);
            self.telemetry.repair_iterations = result.iterations;
            self.telemetry.colors_added = result.colors_added;
            self.telemetry.budget_violated = result.budget_violated;
            self.telemetry.geometry_used = format!(
                "hotspots:{}, stress:{}, persist:{}, beliefs:{}",
                result.geometry_used.hotspots,
                result.geometry_used.stress,
                result.geometry_used.persistence,
                result.geometry_used.beliefs
            );

            log::info!(
                "{}: Completed - {} colors (+{}), {} conflicts (reduced by {}), budget_ok: {}",
                phase.name(),
                result.final_colors,
                result.colors_added.max(0),
                final_conflicts,
                initial_conflicts.saturating_sub(final_conflicts),
                !result.budget_violated
            );

            // Return appropriate outcome
            if result.budget_violated && phase == CallingPhase::Phase7Ensemble {
                // Phase 7 budget violation is a serious issue
                Ok(PhaseOutcome::Escalate {
                    reason: format!(
                        "{}: Budget violated - added {} colors (max: 0)",
                        phase.name(),
                        result.colors_added
                    ),
                })
            } else {
                Ok(PhaseOutcome::Success {
                    message: format!(
                        "{}: {} colors, {} conflicts",
                        phase.name(),
                        result.final_colors,
                        final_conflicts
                    ),
                    telemetry: self.build_telemetry_map(),
                })
            }
        } else {
            // CPU fallback
            self.execute_cpu_fallback(graph, context)
        }
    }

    /// Sync geometry accumulator data to WHCR GPU buffers
    fn sync_geometry_to_whcr(&mut self, geometry: &GeometryAccumulator) -> Result<(), PrismError> {
        // The existing WHCR uses update_geometry() which uploads fresh data.
        // For now, we need to download from accumulator and re-upload.
        // TODO: Future optimization - share CudaSlice references directly

        // This is a temporary solution. The proper fix is to modify
        // WaveletHierarchicalRepairGpu to accept CudaSlice references
        // instead of owning the buffers.

        Ok(())
    }

    /// Build telemetry HashMap from internal state
    fn build_telemetry_map(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert(
            "calling_phase".to_string(),
            serde_json::json!(self.telemetry.calling_phase.name()),
        );
        map.insert(
            "initial_conflicts".to_string(),
            serde_json::json!(self.telemetry.initial_conflicts),
        );
        map.insert(
            "final_conflicts".to_string(),
            serde_json::json!(self.telemetry.final_conflicts),
        );
        map.insert(
            "conflicts_reduced".to_string(),
            serde_json::json!(self.telemetry.conflicts_reduced),
        );
        map.insert(
            "repair_iterations".to_string(),
            serde_json::json!(self.telemetry.repair_iterations),
        );
        map.insert(
            "precision_level".to_string(),
            serde_json::json!(self.telemetry.precision_level),
        );
        map.insert(
            "colors_added".to_string(),
            serde_json::json!(self.telemetry.colors_added),
        );
        map.insert(
            "budget_violated".to_string(),
            serde_json::json!(self.telemetry.budget_violated),
        );
        map.insert(
            "geometry_used".to_string(),
            serde_json::json!(self.telemetry.geometry_used),
        );
        map
    }

    /// CPU fallback execution
    fn execute_cpu_fallback(
        &mut self,
        graph: &Graph,
        context: &mut PhaseContext,
    ) -> Result<PhaseOutcome, PrismError> {
        let phase = self.config.calling_phase;

        log::warn!("{}: Using CPU fallback (greedy repair)", phase.name());

        let solution = context
            .best_solution
            .as_mut()
            .ok_or_else(|| PrismError::phase(phase.name(), "No solution available"))?;

        let initial_conflicts = count_conflicts(&solution.colors, &graph.adjacency);
        cpu_greedy_repair(&mut solution.colors, &graph.adjacency);
        let final_conflicts = count_conflicts(&solution.colors, &graph.adjacency);

        solution.conflicts = final_conflicts;

        self.telemetry.final_conflicts = final_conflicts;
        self.telemetry.conflicts_reduced = initial_conflicts.saturating_sub(final_conflicts);

        Ok(PhaseOutcome::Success {
            message: format!(
                "{} (CPU): {} colors, {} conflicts",
                phase.name(),
                solution.chromatic_number,
                final_conflicts
            ),
            telemetry: self.build_telemetry_map(),
        })
    }

    // =========================================================================
    // Configuration Access
    // =========================================================================

    /// Get the calling phase
    #[inline]
    pub fn calling_phase(&self) -> CallingPhase {
        self.config.calling_phase
    }

    /// Get mutable reference to config for customization
    pub fn config_mut(&mut self) -> &mut PhaseWHCRConfig {
        &mut self.config
    }

    /// Check if GPU acceleration is available
    #[inline]
    pub fn has_gpu(&self) -> bool {
        self.whcr_gpu.is_some()
    }
}

// ============================================================================
// PhaseController Trait Implementation
// ============================================================================

impl PhaseController for WHCRPhaseController {
    fn execute(
        &mut self,
        graph: &Graph,
        context: &mut PhaseContext,
    ) -> Result<PhaseOutcome, PrismError> {
        // For trait-based execution without explicit geometry,
        // create an empty geometry accumulator.
        // This maintains backward compatibility but loses multi-phase benefits.

        log::warn!(
            "{}: execute() called without geometry - using empty accumulator",
            self.config.calling_phase.name()
        );

        if let Some(gpu_ctx) = &self.gpu_context {
            let geometry = GeometryAccumulator::new(gpu_ctx.device().clone(), graph.num_vertices)
                .map_err(|e| {
                PrismError::gpu(
                    self.config.calling_phase.name(),
                    format!("Failed to create geometry accumulator: {}", e),
                )
            })?;

            self.execute_with_geometry(graph, context, &geometry)
        } else {
            self.execute_cpu_fallback(graph, context)
        }
    }

    fn name(&self) -> &'static str {
        self.config.calling_phase.name()
    }

    fn telemetry(&self) -> &dyn PhaseTelemetry {
        &self.telemetry
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn adjacency_to_csr(adjacency: &[Vec<usize>]) -> (Vec<i32>, Vec<i32>) {
    let mut row_ptr = vec![0i32];
    let mut col_idx = Vec::new();

    for neighbors in adjacency {
        for &n in neighbors {
            col_idx.push(n as i32);
        }
        row_ptr.push(col_idx.len() as i32);
    }

    (row_ptr, col_idx)
}

fn count_conflicts(coloring: &[usize], adjacency: &[Vec<usize>]) -> usize {
    let mut conflicts = 0;
    for (v, neighbors) in adjacency.iter().enumerate() {
        for &u in neighbors {
            if u > v && coloring[v] == coloring[u] {
                conflicts += 1;
            }
        }
    }
    conflicts
}

fn cpu_greedy_repair(coloring: &mut [usize], adjacency: &[Vec<usize>]) {
    let max_color = *coloring.iter().max().unwrap_or(&0);

    for v in 0..coloring.len() {
        let has_conflict = adjacency[v].iter().any(|&u| coloring[u] == coloring[v]);

        if has_conflict {
            for c in 0..=max_color + 1 {
                let conflicts = adjacency[v].iter().filter(|&&u| coloring[u] == c).count();
                if conflicts == 0 {
                    coloring[v] = c;
                    break;
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_controller_names() {
        // Verify each phase has distinct name
        let phases = [
            CallingPhase::Phase2Thermodynamic,
            CallingPhase::Phase3Quantum,
            CallingPhase::Phase5Membrane,
            CallingPhase::Phase7Ensemble,
        ];

        let names: Vec<_> = phases.iter().map(|p| p.name()).collect();

        // All unique
        for (i, name) in names.iter().enumerate() {
            for (j, other) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(name, other, "Phases {} and {} have same name", i, j);
                }
            }
        }
    }

    #[test]
    fn test_cpu_greedy_repair() {
        let adjacency = vec![
            vec![1, 2],    // 0 -> 1, 2
            vec![0, 2],    // 1 -> 0, 2
            vec![0, 1, 3], // 2 -> 0, 1, 3
            vec![2],       // 3 -> 2
        ];

        // Start with conflicting coloring
        let mut coloring = vec![0, 0, 0, 0]; // All same color = max conflicts

        cpu_greedy_repair(&mut coloring, &adjacency);

        // Should have no conflicts after repair
        let conflicts = count_conflicts(&coloring, &adjacency);
        assert_eq!(conflicts, 0, "CPU repair should eliminate all conflicts");
    }
}
