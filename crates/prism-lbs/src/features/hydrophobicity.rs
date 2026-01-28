//! Hydrophobicity feature helpers

use crate::structure::hydrophobicity_scale;
use crate::structure::Atom;

pub fn hydrophobicity_feature(atom: &Atom) -> f64 {
    hydrophobicity_scale(&atom.residue_name)
}
