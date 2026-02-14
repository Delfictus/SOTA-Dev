//! Variant prediction data structures and logic

use serde::{Serialize, Deserialize};

/// Evolutionary cycle phase (6 phases)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Never under immune selection (freq <1%, velocity <1%)
    Naive = 0,
    /// Actively rising under selection (velocity >5%, freq <50%)
    Exploring = 1,
    /// Dominant escape variant (freq >50%, velocity ≥-2%)
    Escaped = 2,
    /// Fitness cost accumulating (freq >20%, velocity <-2%, γ <0)
    Costly = 3,
    /// Returning to wild-type (velocity <-5%)
    Reverting = 4,
    /// Stable compensated escape (freq >80%, |velocity| <2%, γ >-0.1)
    Fixed = 5,
}

impl Phase {
    /// Create from integer value (from GPU kernel)
    pub fn from_i32(val: i32) -> Self {
        match val {
            0 => Phase::Naive,
            1 => Phase::Exploring,
            2 => Phase::Escaped,
            3 => Phase::Costly,
            4 => Phase::Reverting,
            5 => Phase::Fixed,
            _ => Phase::Exploring,  // Default
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Phase::Naive => "Never under immune selection",
            Phase::Exploring => "Currently rising under selection",
            Phase::Escaped => "Dominant escape variant",
            Phase::Costly => "Fitness cost accumulating",
            Phase::Reverting => "Returning to wild-type",
            Phase::Fixed => "Stable compensated escape",
        }
    }

    /// Is this phase ready for emergence?
    pub fn emergence_ready(&self) -> bool {
        matches!(self, Phase::Naive | Phase::Exploring)
    }
}

/// Time horizon for predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeHorizon {
    /// 3-month forecast
    ThreeMonths,
    /// 6-month forecast
    SixMonths,
    /// 12-month forecast
    TwelveMonths,
}

impl TimeHorizon {
    pub fn months(&self) -> f32 {
        match self {
            TimeHorizon::ThreeMonths => 3.0,
            TimeHorizon::SixMonths => 6.0,
            TimeHorizon::TwelveMonths => 12.0,
        }
    }
}

/// Variant dynamics prediction (for VASIL benchmark)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantPrediction {
    /// Lineage name (e.g., "BA.5")
    pub lineage: String,

    /// Country
    pub country: String,

    /// Assessment date
    pub date: String,

    /// Prediction: "RISE" or "FALL"
    pub prediction: String,

    /// Fitness score (γ)
    /// γ > 0 → variant rising
    /// γ < 0 → variant falling
    pub gamma: f32,

    /// Emergence probability (0-1)
    pub emergence_prob: f32,

    /// Cycle phase
    pub phase: Phase,

    /// Prediction confidence (0-1)
    pub confidence: f32,
}

impl VariantPrediction {
    /// Is this a correct prediction?
    ///
    /// Compares predicted direction (RISE/FALL) to observed frequency change.
    pub fn is_correct(&self, observed_direction: &str) -> bool {
        self.prediction == observed_direction
    }
}

/// Comprehensive variant assessment (full PRISM-VE output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantAssessment {
    /// Variant information
    pub variant_name: String,
    pub assessment_date: String,

    // Escape module outputs
    pub escape_probability: f32,
    pub escape_rank: f32,
    pub antibody_specific_escape: Vec<f32>,  // Per epitope class

    // Fitness module outputs
    pub ddg_binding: f32,
    pub ddg_stability: f32,
    pub expression_score: f32,
    pub relative_fitness: f32,  // γ
    pub viable: bool,

    // Cycle module outputs
    pub cycle_phase: Phase,
    pub phase_confidence: f32,
    pub current_frequency: f32,
    pub velocity: f32,  // Δfreq/month

    // Integrated predictions
    pub emergence_probability: f32,
    pub predicted_timing: String,  // "1-3 months", "3-6 months", etc.
    pub months_to_dominance: f32,
    pub predicted_peak_frequency: f32,

    // Metadata
    pub overall_confidence: f32,
    pub processing_time_ms: u64,
}

/// Cycle state for a position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleState {
    pub position: i32,
    pub phase: Phase,
    pub phase_confidence: f32,
    pub current_frequency: f32,
    pub velocity: f32,
    pub acceleration: f32,
    pub time_in_phase_days: f32,
    pub emergence_probability: f32,
    pub predicted_timing: String,
}
