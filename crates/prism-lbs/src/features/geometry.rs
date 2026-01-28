//! Geometric feature helpers

use crate::structure::Atom;

pub fn curvature_feature(atom: &Atom) -> f64 {
    atom.curvature
}

pub fn depth_feature(atom: &Atom) -> f64 {
    atom.depth
}
