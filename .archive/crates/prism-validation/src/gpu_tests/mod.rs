//! Test modules for prism-validation
//!
//! These tests require GPU for most functionality.
//! Tests are gated by `#[cfg(feature = "cryptic-gpu")]` where appropriate.

#[cfg(feature = "cryptic-gpu")]
pub mod gpu_scorer_tests;
