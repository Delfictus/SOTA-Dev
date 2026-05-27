//! Sequence conservation placeholder

use crate::structure::Atom;

/// Placeholder conservation feature; in production this should be populated from MSA
pub fn conservation_feature(_atom: &Atom) -> f64 {
    0.0
}
