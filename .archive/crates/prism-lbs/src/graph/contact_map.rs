//! Residue contact map construction

use crate::structure::ProteinStructure;

/// Compute a binary residue contact map based on CA-CA distance threshold
pub fn residue_contact_map(structure: &ProteinStructure, threshold: f64) -> Vec<Vec<bool>> {
    let n = structure.residues.len();
    let mut contacts = vec![vec![false; n]; n];
    let thresh_sq = threshold * threshold;

    // Precompute CA coords
    let mut ca_coords: Vec<Option<[f64; 3]>> = vec![None; n];
    for (idx, residue) in structure.residues.iter().enumerate() {
        if let Some(ca_idx) = residue.ca_index(&structure.atoms) {
            ca_coords[idx] = Some(structure.atoms[ca_idx].coord);
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            match (ca_coords[i], ca_coords[j]) {
                (Some(a), Some(b)) => {
                    let dx = a[0] - b[0];
                    let dy = a[1] - b[1];
                    let dz = a[2] - b[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 <= thresh_sq {
                        contacts[i][j] = true;
                        contacts[j][i] = true;
                    }
                }
                _ => {}
            }
        }
    }

    contacts
}
