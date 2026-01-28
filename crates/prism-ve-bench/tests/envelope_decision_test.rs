//! Golden File Test for Envelope Decision Rule
//!
//! Verifies VASIL Extended Data Fig 6a envelope decision logic:
//! - Entirely positive envelope → Rising (decided)
//! - Entirely negative envelope → Falling (decided)
//! - Envelope crosses zero → Undecided (MUST EXCLUDE from accuracy)
//!
//! This is a CRITICAL test for scientific integrity.

use prism_ve_bench::{EnvelopeDecision, DayDirection};

#[test]
fn test_envelope_entirely_positive_is_rising() {
    // Case: All 75 PK values produce positive gamma
    // Min > 0, Max > 0 → Envelope entirely positive → Rising
    let min = 0.15;
    let max = 0.45;

    let decision = EnvelopeDecision::from_envelope(min, max);

    assert_eq!(decision, EnvelopeDecision::Rising,
        "Entirely positive envelope (min={}, max={}) should be Rising", min, max);
    assert!(decision.is_decided(),
        "Entirely positive envelope should be decided");
}

#[test]
fn test_envelope_entirely_negative_is_falling() {
    // Case: All 75 PK values produce negative gamma
    // Min < 0, Max < 0 → Envelope entirely negative → Falling
    let min = -0.45;
    let max = -0.12;

    let decision = EnvelopeDecision::from_envelope(min, max);

    assert_eq!(decision, EnvelopeDecision::Falling,
        "Entirely negative envelope (min={}, max={}) should be Falling", min, max);
    assert!(decision.is_decided(),
        "Entirely negative envelope should be decided");
}

#[test]
fn test_envelope_crosses_zero_is_undecided() {
    // Case: Some PKs produce positive gamma, some negative
    // Min < 0, Max > 0 → Envelope crosses zero → Undecided
    // CRITICAL: Must EXCLUDE from accuracy calculation per VASIL spec
    let min = -0.18;
    let max = 0.24;

    let decision = EnvelopeDecision::from_envelope(min, max);

    assert_eq!(decision, EnvelopeDecision::Undecided,
        "Envelope crossing zero (min={}, max={}) should be Undecided", min, max);
    assert!(!decision.is_decided(),
        "Envelope crossing zero should NOT be decided");
    assert_eq!(decision.to_day_direction(), None,
        "Undecided envelope should not have direction");
}

#[test]
fn test_envelope_edge_case_zero_min() {
    // Edge case: min exactly at zero, max positive
    // This is still crossing zero → Undecided
    let min = 0.0;
    let max = 0.35;

    let decision = EnvelopeDecision::from_envelope(min, max);

    // With min=0, envelope includes zero → Undecided
    assert_eq!(decision, EnvelopeDecision::Undecided,
        "Envelope with min=0 (edge case) should be Undecided");
}

#[test]
fn test_envelope_edge_case_zero_max() {
    // Edge case: max exactly at zero, min negative
    // This is still crossing zero → Undecided
    let min = -0.25;
    let max = 0.0;

    let decision = EnvelopeDecision::from_envelope(min, max);

    // With max=0, envelope includes zero → Undecided
    assert_eq!(decision, EnvelopeDecision::Undecided,
        "Envelope with max=0 (edge case) should be Undecided");
}

#[test]
fn test_envelope_barely_positive() {
    // Edge case: Very small positive envelope
    // Min > 0 (barely), Max > 0 → Rising
    let min = 0.001;
    let max = 0.035;

    let decision = EnvelopeDecision::from_envelope(min, max);

    assert_eq!(decision, EnvelopeDecision::Rising,
        "Barely positive envelope (min={}, max={}) should still be Rising", min, max);
}

#[test]
fn test_envelope_barely_negative() {
    // Edge case: Very small negative envelope
    // Min < 0, Max < 0 (barely) → Falling
    let min = -0.042;
    let max = -0.002;

    let decision = EnvelopeDecision::from_envelope(min, max);

    assert_eq!(decision, EnvelopeDecision::Falling,
        "Barely negative envelope (min={}, max={}) should still be Falling", min, max);
}

#[test]
fn test_envelope_to_day_direction_conversion() {
    // Test conversion from EnvelopeDecision to DayDirection

    let rising = EnvelopeDecision::Rising;
    assert_eq!(rising.to_day_direction(), Some(DayDirection::Rising));

    let falling = EnvelopeDecision::Falling;
    assert_eq!(falling.to_day_direction(), Some(DayDirection::Falling));

    let undecided = EnvelopeDecision::Undecided;
    assert_eq!(undecided.to_day_direction(), None,
        "Undecided envelope should not convert to DayDirection");
}

#[test]
fn test_realistic_vasil_envelopes() {
    // Realistic envelope values from VASIL paper uncertainty
    // Typical envelope width: ±0.2 around mean

    // Case 1: Strong rising signal (mean=0.3, envelope [0.1, 0.5])
    let decision1 = EnvelopeDecision::from_envelope(0.1, 0.5);
    assert_eq!(decision1, EnvelopeDecision::Rising,
        "Strong rising envelope should be Rising");

    // Case 2: Strong falling signal (mean=-0.25, envelope [-0.45, -0.05])
    let decision2 = EnvelopeDecision::from_envelope(-0.45, -0.05);
    assert_eq!(decision2, EnvelopeDecision::Falling,
        "Strong falling envelope should be Falling");

    // Case 3: Uncertain signal (mean=0.05, envelope [-0.15, 0.25])
    let decision3 = EnvelopeDecision::from_envelope(-0.15, 0.25);
    assert_eq!(decision3, EnvelopeDecision::Undecided,
        "Uncertain envelope crossing zero should be Undecided");
}

#[test]
fn test_envelope_symmetry() {
    // Envelope decision should be symmetric around zero

    // Positive envelope
    let pos = EnvelopeDecision::from_envelope(0.1, 0.4);
    assert_eq!(pos, EnvelopeDecision::Rising);

    // Symmetric negative envelope
    let neg = EnvelopeDecision::from_envelope(-0.4, -0.1);
    assert_eq!(neg, EnvelopeDecision::Falling);

    // Both should be decided
    assert!(pos.is_decided());
    assert!(neg.is_decided());
}
