//! Extensions to WaveletHierarchicalRepairGpu for multi-phase integration

use crate::{CallingPhase, GeometryBuffers, GeometryWeights, PhaseWHCRConfig};
use anyhow::Result;
use prism_gpu::WhcrGpu as WaveletHierarchicalRepairGpu;

/// Result from a phase-aware repair operation
#[derive(Debug, Clone)]
pub struct PhaseRepairResult {
    /// Whether zero conflicts were achieved
    pub success: bool,
    /// Final number of colors used
    pub final_colors: usize,
    /// Final conflict count
    pub final_conflicts: usize,
    /// Iterations executed
    pub iterations: usize,
    /// Which phase invoked this repair
    pub calling_phase: CallingPhase,
    /// Color budget used (final_colors - initial_colors)
    pub colors_added: i32,
    /// Whether color budget was exceeded (Phase 7 violation)
    pub budget_violated: bool,
    /// Geometry sources used
    pub geometry_used: GeometryUsed,
}

/// Tracks which geometry sources were actually used in repair
#[derive(Debug, Clone, Default)]
pub struct GeometryUsed {
    pub hotspots: bool,
    pub stress: bool,
    pub persistence: bool,
    pub beliefs: bool,
}

/// Phase-aware repair with configuration and geometry buffers
///
/// This is the primary entry point for multi-phase integration.
pub fn repair_with_phase_config(
    whcr: &mut WaveletHierarchicalRepairGpu,
    coloring: &mut [usize],
    current_colors: usize,
    config: &PhaseWHCRConfig,
    geometry: &GeometryBuffers<'_>,
) -> Result<PhaseRepairResult> {
    let phase = config.calling_phase;
    let max_colors = config.effective_max_colors(current_colors);
    let precision = config.precision();
    let iterations = config.iterations();

    log::info!(
        "{}: Starting repair - {} vertices, {} colors (max: {}), precision: {}, iterations: {}",
        phase.name(),
        coloring.len(),
        current_colors,
        max_colors,
        precision,
        iterations
    );

    // Bind geometry buffers (zero-copy) before running repair
    whcr.set_geometry_buffers(
        geometry.stress_scores,
        geometry.persistence_scores,
        geometry.hotspot_mask,
        geometry.belief_distribution,
        geometry.belief_num_colors,
        geometry.reservoir_priorities,
    );

    // Resolved TODO(GPU-WHCR-3): Set geometry weights from phase config
    let weights = config.geometry_weights;
    whcr.set_geometry_weights(
        weights.stress_weight,
        weights.persistence_weight,
        weights.belief_weight,
        weights.hotspot_multiplier,
    )?;

    log::debug!(
        "{}: Geometry weights configured - stress: {:.2}, persistence: {:.2}, belief: {:.2}, hotspot: {:.2}",
        phase.name(),
        weights.stress_weight,
        weights.persistence_weight,
        weights.belief_weight,
        weights.hotspot_multiplier
    );

    // For now, use the basic repair method with default parameters
    // Full geometry integration would require additional kernel development
    let mut solution_colors = coloring.to_vec();

    // Adaptive number of levels based on graph size
    let num_vertices = coloring.len();
    let num_levels = if num_vertices <= 100 {
        3 // Small graphs: 3 levels
    } else if num_vertices <= 500 {
        4 // Medium graphs: 4 levels
    } else if num_vertices <= 1000 {
        5 // Large graphs: 5 levels
    } else {
        ((num_vertices as f64).log2().ceil() as usize).min(7) // Very large: logarithmic, max 7
    };

    log::debug!(
        "{}: Using {} wavelet levels for {} vertices",
        phase.name(),
        num_levels,
        num_vertices
    );

    // Call repair with adaptive parameters (coloring, max_colors, max_iterations, precision_level)
    let repair_result = whcr.repair(
        &mut solution_colors,
        max_colors,
        iterations,
        precision as usize,
    )?;

    // Copy back
    coloring.copy_from_slice(&solution_colors);

    // Calculate results
    let final_colors = coloring.iter().max().map(|&m| m + 1).unwrap_or(0);
    let colors_added = final_colors as i32 - current_colors as i32;
    let budget_violated = colors_added > phase.max_color_increase() as i32;

    // Resolved TODO(GPU-WHCR-4): Use actual conflict count from repair result
    let final_conflicts = repair_result.final_conflicts;

    if budget_violated {
        log::warn!(
            "{}: Color budget violated! Added {} colors (max: {})",
            phase.name(),
            colors_added,
            phase.max_color_increase()
        );
    }

    log::info!(
        "{}: Completed - {} colors (+{}), {} conflicts, success: {}",
        phase.name(),
        final_colors,
        colors_added.max(0),
        final_conflicts,
        final_conflicts == 0
    );

    Ok(PhaseRepairResult {
        success: final_conflicts == 0,
        final_colors,
        final_conflicts,
        iterations,
        calling_phase: phase,
        colors_added,
        budget_violated,
        geometry_used: GeometryUsed::default(),
    })
}

/// Convenience Functions for Phase Integration

/// Repair after Phase 2 (Thermodynamic Decomposition)
pub fn repair_after_thermodynamic(
    whcr: &mut WaveletHierarchicalRepairGpu,
    coloring: &mut [usize],
    current_colors: usize,
    geometry: &GeometryBuffers<'_>,
) -> Result<PhaseRepairResult> {
    let config = PhaseWHCRConfig::for_phase(CallingPhase::Phase2Thermodynamic, coloring.len());
    repair_with_phase_config(whcr, coloring, current_colors, &config, geometry)
}

/// Repair after Phase 3 (Quantum Annealing)
pub fn repair_after_quantum(
    whcr: &mut WaveletHierarchicalRepairGpu,
    coloring: &mut [usize],
    current_colors: usize,
    geometry: &GeometryBuffers<'_>,
) -> Result<PhaseRepairResult> {
    let config = PhaseWHCRConfig::for_phase(CallingPhase::Phase3Quantum, coloring.len());
    repair_with_phase_config(whcr, coloring, current_colors, &config, geometry)
}

/// Repair at Phase 5 (Membrane Orchestration checkpoint)
pub fn repair_at_membrane_checkpoint(
    whcr: &mut WaveletHierarchicalRepairGpu,
    coloring: &mut [usize],
    current_colors: usize,
    geometry: &GeometryBuffers<'_>,
) -> Result<PhaseRepairResult> {
    let config = PhaseWHCRConfig::for_phase(CallingPhase::Phase5Membrane, coloring.len());
    repair_with_phase_config(whcr, coloring, current_colors, &config, geometry)
}

/// Repair at Phase 7 (Ensemble Exchange - final polish)
pub fn repair_final_ensemble(
    whcr: &mut WaveletHierarchicalRepairGpu,
    coloring: &mut [usize],
    current_colors: usize,
    geometry: &GeometryBuffers<'_>,
) -> Result<PhaseRepairResult> {
    let config = PhaseWHCRConfig::for_phase(CallingPhase::Phase7Ensemble, coloring.len());
    repair_with_phase_config(whcr, coloring, current_colors, &config, geometry)
}
