//! Electrostatic feature helpers
//!
//! Computes electrostatic potential contributions from partial charges
//! using Coulomb-like distance-weighted summation.

use crate::structure::{Atom, ProteinStructure};

/// Returns pre-computed partial charge
pub fn electrostatic_feature(atom: &Atom) -> f64 {
    atom.partial_charge
}

/// Compute local electrostatic potential at an atom position
/// using distance-weighted sum of nearby charges
pub fn local_electrostatic_potential(
    atom_idx: usize,
    structure: &ProteinStructure,
    cutoff: f64,
) -> f64 {
    let atom = &structure.atoms[atom_idx];
    let cutoff_sq = cutoff * cutoff;
    let mut potential = 0.0;

    for (j, other) in structure.atoms.iter().enumerate() {
        if j == atom_idx {
            continue;
        }

        let dx = atom.coord[0] - other.coord[0];
        let dy = atom.coord[1] - other.coord[1];
        let dz = atom.coord[2] - other.coord[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        if dist_sq < cutoff_sq && dist_sq > 0.01 {
            let dist = dist_sq.sqrt();
            // Coulomb potential: q / r (normalized)
            potential += other.partial_charge / dist;
        }
    }

    potential
}

/// Compute mean electrostatic potential for a set of atom indices
pub fn mean_electrostatic_potential(
    atom_indices: &[usize],
    structure: &ProteinStructure,
    cutoff: f64,
) -> f64 {
    if atom_indices.is_empty() {
        return 0.0;
    }

    let sum: f64 = atom_indices
        .iter()
        .map(|&idx| local_electrostatic_potential(idx, structure, cutoff))
        .sum();

    sum / atom_indices.len() as f64
}

/// Compute electrostatic polarity score (variance of charges)
/// High polarity indicates good hydrogen bonding environment
pub fn electrostatic_polarity(atom_indices: &[usize], structure: &ProteinStructure) -> f64 {
    if atom_indices.is_empty() {
        return 0.0;
    }

    let charges: Vec<f64> = atom_indices
        .iter()
        .filter_map(|&idx| structure.atoms.get(idx).map(|a| a.partial_charge))
        .collect();

    if charges.is_empty() {
        return 0.0;
    }

    let mean: f64 = charges.iter().sum::<f64>() / charges.len() as f64;
    let variance: f64 = charges.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / charges.len() as f64;

    // Normalize variance to [0, 1] range (typical charge variance is 0.0-0.5)
    (variance * 4.0).min(1.0)
}

/// Compute charge complementarity score for drug binding
/// Ideal pockets have balanced positive/negative regions
pub fn charge_complementarity(atom_indices: &[usize], structure: &ProteinStructure) -> f64 {
    if atom_indices.is_empty() {
        return 0.0;
    }

    let mut positive_count = 0;
    let mut negative_count = 0;
    let mut neutral_count = 0;

    for &idx in atom_indices {
        if let Some(atom) = structure.atoms.get(idx) {
            if atom.partial_charge > 0.1 {
                positive_count += 1;
            } else if atom.partial_charge < -0.1 {
                negative_count += 1;
            } else {
                neutral_count += 1;
            }
        }
    }

    let total = (positive_count + negative_count + neutral_count) as f64;
    if total == 0.0 {
        return 0.5;
    }

    // Ideal: balanced positive/negative with some neutral
    // Score based on how well-balanced the charges are
    let pos_ratio = positive_count as f64 / total;
    let neg_ratio = negative_count as f64 / total;

    // Best score when pos_ratio ~= neg_ratio ~= 0.25-0.35
    let balance = 1.0 - (pos_ratio - neg_ratio).abs();
    let ideal_charged_ratio = (pos_ratio + neg_ratio).min(0.7).max(0.3);

    balance * 0.7 + ideal_charged_ratio * 0.3
}
