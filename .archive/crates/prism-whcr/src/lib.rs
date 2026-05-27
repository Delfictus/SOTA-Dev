//! PRISM WHCR Multi-Phase Integration
//!
//! This module provides the multi-phase integration layer for Wavelet-Hierarchical
//! Conflict Repair (WHCR). It enables WHCR to be invoked from phases 2, 3, 5, and 7
//! with phase-specific configurations and accumulated geometry.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        PRISM Pipeline                                   │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► ...       │
//! │     │           │           │           │           │                   │
//! │     ▼           ▼           │           │           ▼                   │
//! │  ┌─────────────────────────┴───────────┴───────────────────────────┐   │
//! │  │              GeometryAccumulator (GPU-resident)                  │   │
//! │  │  - Phase 0: hotspots, priorities                                 │   │
//! │  │  - Phase 1: beliefs, free_energy                                 │   │
//! │  │  - Phase 4: stress, embedding                                    │   │
//! │  │  - Phase 6: persistence, betti                                   │   │
//! │  └──────────────────────────┬──────────────────────────────────────┘   │
//! │                              │                                          │
//! │                              ▼                                          │
//! │  ┌──────────────────────────────────────────────────────────────────┐  │
//! │  │              WHCRPhaseController                                  │  │
//! │  │  ┌────────────────────────────────────────────────────────────┐  │  │
//! │  │  │ CallingPhase (2/3/5/7) → PhaseWHCRConfig                   │  │  │
//! │  │  │  - precision_level (f32/mixed/f64)                         │  │  │
//! │  │  │  - max_color_increase (5/3/2/0)                            │  │  │
//! │  │  │  - iterations (200/300/100/500)                            │  │  │
//! │  │  │  - geometry_weights                                        │  │  │
//! │  │  └────────────────────────────────────────────────────────────┘  │  │
//! │  │                              │                                    │  │
//! │  │                              ▼                                    │  │
//! │  │  ┌────────────────────────────────────────────────────────────┐  │  │
//! │  │  │ WaveletHierarchicalRepairGpu (existing, unchanged)         │  │  │
//! │  │  │  - CUDA kernels (PTX compiled)                             │  │  │
//! │  │  │  - cudarc FFI bindings                                     │  │  │
//! │  │  │  - Mixed precision (f32/f64)                               │  │  │
//! │  │  └────────────────────────────────────────────────────────────┘  │  │
//! │  └──────────────────────────────────────────────────────────────────┘  │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Module Structure
//!
//! - `calling_phase`: Defines CallingPhase enum and phase-specific configurations
//! - `geometry_accumulator`: GPU-resident geometry buffer management
//! - `whcr_phase_extension`: Extensions to WaveletHierarchicalRepairGpu
//! - `phase_whcr_multiphase`: Updated WHCRPhaseController with multi-phase support
//!
//! # Usage Example
//!
//! ```ignore
//! use prism_whcr::{
//!     CallingPhase, GeometryAccumulator, WHCRPhaseController,
//! };
//!
//! // Initialize geometry accumulator (once per pipeline)
//! let mut geometry = GeometryAccumulator::new(device.clone(), num_vertices)?;
//!
//! // After Phase 0: Register dendritic hotspots
//! geometry.set_phase0_hotspots(&hotspot_indices)?;
//!
//! // After Phase 1: Register belief distribution
//! geometry.set_phase1_beliefs(&beliefs, num_colors)?;
//!
//! // === Phase 2 WHCR Invocation ===
//! let mut whcr_p2 = WHCRPhaseController::for_phase2(gpu_context.clone(), &graph)?;
//! whcr_p2.execute_with_geometry(&graph, &mut context, &geometry)?;
//!
//! // After Phase 4: Register stress scores
//! geometry.set_phase4_stress(&stress_tensor)?;
//!
//! // === Phase 3 WHCR Invocation ===
//! let mut whcr_p3 = WHCRPhaseController::for_phase3(gpu_context.clone(), &graph)?;
//! whcr_p3.execute_with_geometry(&graph, &mut context, &geometry)?;
//!
//! // After Phase 6: Register persistence scores
//! geometry.set_phase6_persistence(&persistence)?;
//!
//! // === Phase 5 WHCR Invocation (checkpoint) ===
//! let mut whcr_p5 = WHCRPhaseController::for_phase5(gpu_context.clone(), &graph)?;
//! whcr_p5.execute_with_geometry(&graph, &mut context, &geometry)?;
//!
//! // === Phase 7 WHCR Invocation (final polish) ===
//! let mut whcr_p7 = WHCRPhaseController::for_phase7(gpu_context.clone(), &graph)?;
//! whcr_p7.execute_with_geometry(&graph, &mut context, &geometry)?;
//! // Note: Phase 7 has max_color_increase = 0 (strict bound)
//! ```
//!
//! # Competitive Moat
//!
//! This multi-phase integration creates a compound moat because:
//!
//! 1. **Geometry Accumulation**: Each WHCR invocation has access to MORE geometry
//!    than the previous one. Phase 7 uses geometry from ALL prior phases.
//!
//! 2. **Phase-Specific Tuning**: Early phases explore aggressively (f32, +5 colors),
//!    late phases refine carefully (f64, +0 colors).
//!
//! 3. **Architectural Lock-In**: WHCR requires the specific geometry outputs from
//!    PRISM's Phase 0/1/4/6. Competitors cannot replicate without the entire pipeline.
//!
//! 4. **GPU-Resident State**: Geometry stays on GPU between phases, eliminating
//!    CPU roundtrip overhead that would kill performance.

// Core phase definitions
pub mod calling_phase;
pub use calling_phase::{CallingPhase, GeometrySources, GeometryWeights, PhaseWHCRConfig};

// GPU-resident geometry accumulation
pub mod geometry_accumulator;
pub use geometry_accumulator::{GeometryAccumulator, GeometryBuffers};

// Extensions to WaveletHierarchicalRepairGpu
pub mod whcr_extensions;
pub use whcr_extensions::{
    repair_after_quantum, repair_after_thermodynamic, repair_at_membrane_checkpoint,
    repair_final_ensemble, repair_with_phase_config, GeometryUsed, PhaseRepairResult,
};

// Advanced geometry extraction and synchronization
pub mod geometry_sync;
#[cfg(feature = "cuda")]
pub use geometry_sync::GeometrySynchronizer;
pub use geometry_sync::{ExtractionConfig, GeometryExtractor};

// Updated phase controller
// Note: WHCRPhaseController is now in prism-phases/src/phase_whcr.rs
// pub mod phase_whcr_multiphase;
// pub use phase_whcr_multiphase::WHCRPhaseController;

// Re-export from prism-gpu for convenience
// (These would be actual imports in the real crate)
// pub use prism_gpu::{WaveletHierarchicalRepairGpu, DendriticWhcrGpu};

/// Version of the multi-phase integration
pub const VERSION: &str = "1.0.0";

/// Summary of multi-phase configuration
pub fn configuration_summary() -> String {
    format!(
        r#"WHCR Multi-Phase Configuration:
  Phase 2 (Thermodynamic): precision=f32, max_colors=+5, iterations=200
  Phase 3 (Quantum):       precision=mixed, max_colors=+3, iterations=300
  Phase 5 (Membrane):      precision=f64, max_colors=+2, iterations=100
  Phase 7 (Ensemble):      precision=f64, max_colors=+0, iterations=500
"#
    )
}
