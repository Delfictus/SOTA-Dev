//! Shadow Pipeline
//!
//! Provides comparison of NOVA and AMBER outputs for validation.

mod comparator;

pub use comparator::{DivergenceMetrics, DivergenceVerdict, ShadowComparator, ShadowResult};
