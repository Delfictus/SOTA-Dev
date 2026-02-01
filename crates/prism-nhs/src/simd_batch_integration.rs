///! Integration layer for AmberSimdBatch with NHS batch processor
///!
///! Converts PrismPrepTopology to StructureTopology format for concurrent batch MD

use anyhow::{Context, Result};
use std::collections::HashSet;
use prism_gpu::amber_simd_batch::StructureTopology;
use crate::input::PrismPrepTopology;

/// Convert PrismPrepTopology to StructureTopology for AmberSimdBatch
pub fn convert_to_structure_topology(prism_topo: &PrismPrepTopology) -> Result<StructureTopology> {
    // Extract LJ parameters (sigma, epsilon) from lj_params
    let mut sigmas = Vec::with_capacity(prism_topo.n_atoms);
    let mut epsilons = Vec::with_capacity(prism_topo.n_atoms);

    for lj in &prism_topo.lj_params {
        sigmas.push(lj.sigma as f32);
        epsilons.push(lj.epsilon as f32);
    }

    // Convert bonds to tuple format (i, j, r0, k)
    let bonds: Vec<(usize, usize, f32, f32)> = prism_topo.bonds
        .iter()
        .map(|b| (b.i, b.j, b.r0 as f32, b.k as f32))
        .collect();

    // Convert angles to tuple format (i, j, k, theta0, k)
    let angles: Vec<(usize, usize, usize, f32, f32)> = prism_topo.angles
        .iter()
        .map(|a| (a.i, a.j, a.k_idx, a.theta0 as f32, a.force_k as f32))
        .collect();

    // Convert dihedrals to tuple format (i, j, k, l, periodicity, phase, k)
    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = prism_topo.dihedrals
        .iter()
        .map(|d| (
            d.i,
            d.j,
            d.k_idx,
            d.l,  // Fourth atom index
            d.periodicity as f32,
            d.phase as f32,
            d.force_k as f32
        ))
        .collect();

    // Convert exclusions Vec<Vec<usize>> to Vec<HashSet<usize>>
    let exclusions: Vec<HashSet<usize>> = prism_topo.exclusions
        .iter()
        .map(|excl_vec| excl_vec.iter().cloned().collect())
        .collect();

    Ok(StructureTopology {
        positions: prism_topo.positions.clone(),
        masses: prism_topo.masses.clone(),
        charges: prism_topo.charges.clone(),
        sigmas,
        epsilons,
        bonds,
        angles,
        dihedrals,
        exclusions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_basic() {
        // Test that conversion preserves atom count and basic properties
        // This will be tested with actual topology files
    }
}
