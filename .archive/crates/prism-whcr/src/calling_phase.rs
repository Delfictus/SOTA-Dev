//! Calling Phase Definitions for WHCR Multi-Phase Integration
//!
//! Defines which PRISM phase is invoking WHCR and the corresponding
//! configuration that controls GPU kernel behavior.
//!
//! # Design Principles
//! - Phase identity determines GPU kernel selection and parameters
//! - Earlier phases are more aggressive (allow color increases, use f32)
//! - Later phases are more conservative (strict bounds, use f64 with geometry)
//! - Geometry availability increases as phases progress

use std::collections::HashSet;

/// Identifies which PRISM phase is invoking WHCR
///
/// Each phase has different:
/// - Available geometry (what prior phases have computed)
/// - Repair aggressiveness (color budget, iteration count)
/// - Precision requirements (f32 exploration vs f64 refinement)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CallingPhase {
    /// Phase 2: Thermodynamic decomposition
    /// - Geometry: Phase 0 (dendritic), Phase 1 (active inference)
    /// - High aggressiveness: can add up to 5 colors
    /// - Precision: f32 for fast exploration
    Phase2Thermodynamic = 2,

    /// Phase 3: Quantum annealing
    /// - Geometry: Phase 0, 1, 4 (geodesic stress)
    /// - Medium aggressiveness: can add up to 3 colors
    /// - Precision: mixed (f32 coarse, f64 fine)
    Phase3Quantum = 3,

    /// Phase 5: Membrane orchestration (checkpoint)
    /// - Geometry: All phases (0, 1, 4, 6)
    /// - Low aggressiveness: can add up to 2 colors
    /// - Precision: f64 with full geometry coupling
    Phase5Membrane = 5,

    /// Phase 7: Ensemble exchange (final polish)
    /// - Geometry: All phases
    /// - No aggressiveness: cannot add colors (strict bound)
    /// - Precision: f64 with full geometry, maximum iterations
    Phase7Ensemble = 7,

    /// Standalone invocation (testing, manual use)
    /// - No geometry assumed
    /// - Default configuration
    Standalone = 0,
}

impl CallingPhase {
    /// Get precision level for GPU kernel selection
    ///
    /// Returns:
    /// - 0: f32 only (fast exploration)
    /// - 1: mixed precision (f32 coarse, f64 fine)
    /// - 2: f64 with geometry coupling
    #[inline]
    pub fn precision_level(&self) -> u32 {
        match self {
            CallingPhase::Phase2Thermodynamic => 0, // f32 for speed
            CallingPhase::Phase3Quantum => 1,       // mixed
            CallingPhase::Phase5Membrane => 2,      // f64 + geometry
            CallingPhase::Phase7Ensemble => 2,      // f64 + geometry
            CallingPhase::Standalone => 1,          // mixed default
        }
    }

    /// Maximum color increase allowed at this phase
    #[inline]
    pub fn max_color_increase(&self) -> usize {
        match self {
            CallingPhase::Phase2Thermodynamic => 9, // allow up to ~20 colors when starting near 11
            CallingPhase::Phase3Quantum => 3,
            CallingPhase::Phase5Membrane => 2,
            CallingPhase::Phase7Ensemble => 0, // STRICT: no increase
            CallingPhase::Standalone => 3,
        }
    }

    /// Default iteration budget for this phase
    #[inline]
    pub fn default_iterations(&self) -> usize {
        match self {
            CallingPhase::Phase2Thermodynamic => 200,
            CallingPhase::Phase3Quantum => 300,
            CallingPhase::Phase5Membrane => 100, // Quick checkpoint
            CallingPhase::Phase7Ensemble => 500, // Thorough polish
            CallingPhase::Standalone => 300,
        }
    }

    /// Which geometry sources are expected to be available
    #[inline]
    pub fn expected_geometry(&self) -> GeometrySources {
        match self {
            CallingPhase::Phase2Thermodynamic => GeometrySources {
                phase0_dendritic: true,
                phase1_beliefs: true,
                phase4_stress: false,
                phase6_persistence: false,
            },
            CallingPhase::Phase3Quantum => GeometrySources {
                phase0_dendritic: true,
                phase1_beliefs: true,
                phase4_stress: true,
                phase6_persistence: false,
            },
            CallingPhase::Phase5Membrane | CallingPhase::Phase7Ensemble => GeometrySources {
                phase0_dendritic: true,
                phase1_beliefs: true,
                phase4_stress: true,
                phase6_persistence: true,
            },
            CallingPhase::Standalone => GeometrySources::none(),
        }
    }

    /// Whether to update dendritic reservoir after repair
    ///
    /// Phase 7 doesn't update because it's the final polish -
    /// no future phases will benefit from the learning.
    #[inline]
    pub fn update_reservoir(&self) -> bool {
        match self {
            CallingPhase::Phase7Ensemble => false,
            _ => true,
        }
    }

    /// Early exit threshold (0 = must reach zero conflicts)
    #[inline]
    pub fn early_exit_threshold(&self) -> usize {
        match self {
            CallingPhase::Phase2Thermodynamic => 10, // Tolerate some conflicts
            CallingPhase::Phase3Quantum => 5,
            CallingPhase::Phase5Membrane => 2,
            CallingPhase::Phase7Ensemble => 0, // Must be zero
            CallingPhase::Standalone => 0,
        }
    }

    /// Human-readable name for logging/telemetry
    pub fn name(&self) -> &'static str {
        match self {
            CallingPhase::Phase2Thermodynamic => "WHCR-Phase2-Thermodynamic",
            CallingPhase::Phase3Quantum => "WHCR-Phase3-Quantum",
            CallingPhase::Phase5Membrane => "WHCR-Phase5-Membrane",
            CallingPhase::Phase7Ensemble => "WHCR-Phase7-Ensemble",
            CallingPhase::Standalone => "WHCR-Standalone",
        }
    }
}

/// Tracks which geometry sources are available/expected
#[derive(Debug, Clone, Copy, Default)]
pub struct GeometrySources {
    pub phase0_dendritic: bool,
    pub phase1_beliefs: bool,
    pub phase4_stress: bool,
    pub phase6_persistence: bool,
}

impl GeometrySources {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn all() -> Self {
        Self {
            phase0_dendritic: true,
            phase1_beliefs: true,
            phase4_stress: true,
            phase6_persistence: true,
        }
    }

    /// Check if actual geometry meets expected requirements
    pub fn satisfies(&self, required: &GeometrySources) -> bool {
        (!required.phase0_dendritic || self.phase0_dendritic)
            && (!required.phase1_beliefs || self.phase1_beliefs)
            && (!required.phase4_stress || self.phase4_stress)
            && (!required.phase6_persistence || self.phase6_persistence)
    }
}

/// Phase-specific WHCR configuration
///
/// This extends WHCRConfig with phase-aware parameters that
/// control GPU kernel behavior.
#[derive(Debug, Clone)]
pub struct PhaseWHCRConfig {
    /// Which phase is calling WHCR
    pub calling_phase: CallingPhase,

    /// Number of wavelet decomposition levels
    pub num_levels: usize,

    /// Maximum repair iterations (overrides phase default if set)
    pub max_iterations: Option<usize>,

    /// Precision level (overrides phase default if set)
    /// 0=f32, 1=mixed, 2=f64+geometry
    pub precision_level: Option<u32>,

    /// Maximum colors allowed (current + max_color_increase)
    pub max_colors: Option<usize>,

    /// Reservoir influence weight [0.0, 1.0]
    pub reservoir_influence: f32,

    /// Enable dendritic reservoir co-processing
    pub enable_dendritic: bool,

    /// Geometry weights for move evaluation
    pub geometry_weights: GeometryWeights,
}

impl PhaseWHCRConfig {
    /// Create config for a specific calling phase with defaults
    pub fn for_phase(phase: CallingPhase, num_vertices: usize) -> Self {
        let num_levels = ((num_vertices as f64).log2().ceil() as usize)
            .max(3)
            .min(10);

        Self {
            calling_phase: phase,
            num_levels,
            max_iterations: None,  // Use phase default
            precision_level: None, // Use phase default
            max_colors: None,
            reservoir_influence: match phase {
                CallingPhase::Phase2Thermodynamic => 0.2,
                CallingPhase::Phase3Quantum => 0.3,
                CallingPhase::Phase5Membrane => 0.4,
                CallingPhase::Phase7Ensemble => 0.5,
                CallingPhase::Standalone => 0.3,
            },
            enable_dendritic: true,
            geometry_weights: GeometryWeights::for_phase(phase),
        }
    }

    /// Get effective iteration count
    #[inline]
    pub fn iterations(&self) -> usize {
        self.max_iterations
            .unwrap_or_else(|| self.calling_phase.default_iterations())
    }

    /// Get effective precision level
    #[inline]
    pub fn precision(&self) -> u32 {
        self.precision_level
            .unwrap_or_else(|| self.calling_phase.precision_level())
    }

    /// Get effective max colors given current color count
    #[inline]
    pub fn effective_max_colors(&self, current_colors: usize) -> usize {
        self.max_colors
            .unwrap_or_else(|| current_colors + self.calling_phase.max_color_increase())
    }
}

/// Weights for geometry-coupled move evaluation
///
/// These control how much each geometry source influences
/// the GPU kernel's move evaluation function.
#[derive(Debug, Clone, Copy)]
pub struct GeometryWeights {
    /// Weight for geodesic stress (higher = prioritize high-stress vertices)
    pub stress_weight: f32,

    /// Weight for TDA persistence (higher = protect topologically stable regions)
    pub persistence_weight: f32,

    /// Weight for belief distribution (higher = prefer high-belief colors)
    pub belief_weight: f32,

    /// Multiplier for dendritic hotspots
    pub hotspot_multiplier: f32,
}

impl GeometryWeights {
    /// Phase-specific default weights
    pub fn for_phase(phase: CallingPhase) -> Self {
        match phase {
            CallingPhase::Phase2Thermodynamic => Self {
                stress_weight: 0.0,      // No stress available yet
                persistence_weight: 0.0, // No TDA available yet
                belief_weight: 0.3,      // Use Phase 1 beliefs
                hotspot_multiplier: 1.5, // Moderate hotspot boost
            },
            CallingPhase::Phase3Quantum => Self {
                stress_weight: 0.4,      // Phase 4 stress available
                persistence_weight: 0.0, // No TDA yet
                belief_weight: 0.3,
                hotspot_multiplier: 2.0,
            },
            CallingPhase::Phase5Membrane => Self {
                stress_weight: 0.5,
                persistence_weight: 0.4, // Phase 6 TDA available
                belief_weight: 0.3,
                hotspot_multiplier: 2.0,
            },
            CallingPhase::Phase7Ensemble => Self {
                stress_weight: 0.5,
                persistence_weight: 0.5, // Full geometry coupling
                belief_weight: 0.4,
                hotspot_multiplier: 2.5, // Maximum hotspot priority
            },
            CallingPhase::Standalone => Self {
                stress_weight: 0.3,
                persistence_weight: 0.3,
                belief_weight: 0.3,
                hotspot_multiplier: 1.5,
            },
        }
    }

    /// Pack weights into f32 array for GPU kernel parameter
    ///
    /// Format: [stress, persistence, belief, hotspot_mult]
    #[inline]
    pub fn as_gpu_params(&self) -> [f32; 4] {
        [
            self.stress_weight,
            self.persistence_weight,
            self.belief_weight,
            self.hotspot_multiplier,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_configurations() {
        // Phase 2 should be most aggressive
        assert_eq!(CallingPhase::Phase2Thermodynamic.max_color_increase(), 5);
        assert_eq!(CallingPhase::Phase2Thermodynamic.precision_level(), 0);

        // Phase 7 should be strictest
        assert_eq!(CallingPhase::Phase7Ensemble.max_color_increase(), 0);
        assert_eq!(CallingPhase::Phase7Ensemble.precision_level(), 2);
        assert_eq!(CallingPhase::Phase7Ensemble.early_exit_threshold(), 0);
    }

    #[test]
    fn test_geometry_sources() {
        let phase2_expected = CallingPhase::Phase2Thermodynamic.expected_geometry();
        assert!(phase2_expected.phase0_dendritic);
        assert!(phase2_expected.phase1_beliefs);
        assert!(!phase2_expected.phase4_stress);
        assert!(!phase2_expected.phase6_persistence);

        let phase7_expected = CallingPhase::Phase7Ensemble.expected_geometry();
        assert!(phase7_expected.phase4_stress);
        assert!(phase7_expected.phase6_persistence);
    }

    #[test]
    fn test_config_defaults() {
        let config = PhaseWHCRConfig::for_phase(CallingPhase::Phase3Quantum, 1000);
        assert_eq!(config.iterations(), 300);
        assert_eq!(config.precision(), 1); // mixed
        assert_eq!(config.effective_max_colors(10), 13); // 10 + 3
    }
}
