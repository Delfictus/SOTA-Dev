//! Topology Loader for Prism-Prep Output
//!
//! Loads sanitized structures from prism-prep JSON topology files.
//! This is the CORRECT input format for production cryptic site detection.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Lennard-Jones parameters (sigma, epsilon)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LjParams {
    pub sigma: f64,
    pub epsilon: f64,
}

/// Bond parameters from prism-prep
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondParams {
    pub i: usize,
    pub j: usize,
    pub r0: f64,
    pub k: f64,
}

/// Angle parameters from prism-prep
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngleParams {
    pub i: usize,
    pub j: usize,
    pub k_idx: usize,
    pub theta0: f64,
    pub force_k: f64,
}

/// Dihedral parameters from prism-prep
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DihedralParams {
    pub i: usize,
    pub j: usize,
    pub k_idx: usize,
    pub l: usize,
    pub periodicity: i32,
    pub phase: f64,
    pub force_k: f64,
}

/// Hydrogen cluster parameters from prism-prep
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HCluster {
    #[serde(rename = "type")]
    pub cluster_type: i32,
    pub central_atom: usize,
    pub hydrogen_atoms: Vec<i32>,
    pub bond_lengths: Vec<f64>,
    pub n_hydrogens: i32,
    pub inv_mass_central: f64,
    pub inv_mass_h: f64,
}

/// Topology structure from prism-prep JSON output
///
/// This struct matches the exact output format of prism-prep.
/// For cryptic site detection, we primarily need: positions, ca_indices,
/// residue_names, residue_ids, chain_ids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismTopology {
    /// Source PDB file name
    pub source_pdb: String,
    /// Total number of atoms
    pub n_atoms: usize,
    /// Number of residues
    pub n_residues: usize,
    /// Number of chains
    pub n_chains: usize,
    /// Flat array of positions [x1, y1, z1, x2, y2, z2, ...]
    pub positions: Vec<f64>,
    /// Atomic masses
    pub masses: Vec<f64>,
    /// Element symbols
    pub elements: Vec<String>,
    /// Atom names (e.g., "CA", "N", "C")
    pub atom_names: Vec<String>,
    /// Residue names (e.g., "ALA", "GLY")
    pub residue_names: Vec<String>,
    /// Residue sequence numbers
    pub residue_ids: Vec<i32>,
    /// Chain identifiers
    pub chain_ids: Vec<String>,
    /// Indices of Cα atoms (0-based)
    pub ca_indices: Vec<usize>,
    /// Partial charges
    pub charges: Vec<f64>,
    /// Lennard-Jones parameters {sigma, epsilon}
    pub lj_params: Vec<LjParams>,
    /// Bond parameters {i, j, r0, k}
    pub bonds: Vec<BondParams>,
    /// Angle parameters {i, j, k_idx, theta0, force_k}
    pub angles: Vec<AngleParams>,
    /// Dihedral parameters {i, j, k_idx, l, periodicity, phase, force_k}
    pub dihedrals: Vec<DihedralParams>,
    /// Water oxygen indices (if any)
    #[serde(default)]
    pub water_oxygens: Vec<usize>,
    /// Hydrogen clusters for SETTLE/SHAKE
    #[serde(default)]
    pub h_clusters: Vec<HCluster>,
    /// Exclusion adjacency list: exclusions[i] = list of atoms excluded from atom i
    #[serde(default)]
    pub exclusions: Vec<Vec<usize>>,
}

impl PrismTopology {
    /// Load topology from prism-prep JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read topology file: {}", path.display()))?;

        let topology: PrismTopology = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse topology JSON: {}", path.display()))?;

        // Validate consistency
        topology.validate()?;

        Ok(topology)
    }

    /// Validate topology consistency
    fn validate(&self) -> Result<()> {
        // Check positions array length
        if self.positions.len() != self.n_atoms * 3 {
            anyhow::bail!(
                "Position array length mismatch: expected {} ({}*3), got {}",
                self.n_atoms * 3, self.n_atoms, self.positions.len()
            );
        }

        // Check CA indices are valid
        for &idx in &self.ca_indices {
            if idx >= self.n_atoms {
                anyhow::bail!("Invalid CA index {} (n_atoms={})", idx, self.n_atoms);
            }
        }

        // Check CA count matches n_residues
        if self.ca_indices.len() != self.n_residues {
            anyhow::bail!(
                "CA indices count mismatch: {} CA atoms but {} residues",
                self.ca_indices.len(), self.n_residues
            );
        }

        Ok(())
    }

    /// Extract Cα coordinates as [n_residues][3] array
    pub fn get_ca_coordinates(&self) -> Vec<[f32; 3]> {
        self.ca_indices.iter().map(|&idx| {
            let base = idx * 3;
            [
                self.positions[base] as f32,
                self.positions[base + 1] as f32,
                self.positions[base + 2] as f32,
            ]
        }).collect()
    }

    /// Extract all atom coordinates as [n_atoms][3] array
    pub fn get_all_coordinates(&self) -> Vec<[f64; 3]> {
        (0..self.n_atoms).map(|i| {
            let base = i * 3;
            [
                self.positions[base],
                self.positions[base + 1],
                self.positions[base + 2],
            ]
        }).collect()
    }

    /// Get residue name for each CA (i.e., per-residue)
    pub fn get_ca_residue_names(&self) -> Vec<String> {
        self.ca_indices.iter().map(|&idx| {
            self.residue_names.get(idx).cloned().unwrap_or_else(|| "UNK".to_string())
        }).collect()
    }

    /// Get residue IDs for each CA
    pub fn get_ca_residue_ids(&self) -> Vec<i32> {
        self.ca_indices.iter().map(|&idx| {
            self.residue_ids.get(idx).copied().unwrap_or(0)
        }).collect()
    }

    /// Get chain IDs for each CA
    pub fn get_ca_chain_ids(&self) -> Vec<char> {
        self.ca_indices.iter().map(|&idx| {
            self.chain_ids.get(idx)
                .and_then(|s| s.chars().next())
                .unwrap_or('A')
        }).collect()
    }

    /// Get PDB ID from source filename
    pub fn get_pdb_id(&self) -> String {
        Path::new(&self.source_pdb)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    }

    /// Get flat position array for GPU SASA calculation
    ///
    /// Returns positions as f32 for GPU compatibility: [x1, y1, z1, x2, y2, z2, ...]
    pub fn get_positions_flat_f32(&self) -> Vec<f32> {
        self.positions.iter().map(|&p| p as f32).collect()
    }

    /// Get atom type indices for GPU LCPO SASA
    ///
    /// Maps element symbols to LCPO atom type indices:
    /// - 0: Carbon (sp3)
    /// - 1: Carbon (sp2) - same as sp3 for now
    /// - 2: Nitrogen
    /// - 3: Oxygen
    /// - 4: Sulfur
    /// - 5: Phosphorus
    /// - 6: Hydrogen
    /// - 7: Other
    pub fn get_sasa_atom_types(&self) -> Vec<i32> {
        self.elements.iter().map(|e| {
            match e.trim().to_uppercase().as_str() {
                "C" => 0,
                "N" => 2,
                "O" => 3,
                "S" => 4,
                "P" => 5,
                "H" => 6,
                _ => 7,
            }
        }).collect()
    }

    /// Get VDW radii for GPU SASA calculation
    ///
    /// Returns per-atom VDW radii in Angstroms, derived from element symbols.
    pub fn get_vdw_radii(&self) -> Vec<f32> {
        self.elements.iter().map(|e| {
            match e.trim().to_uppercase().as_str() {
                "C" => 1.70,
                "N" => 1.55,
                "O" => 1.52,
                "S" => 1.80,
                "P" => 1.80,
                "H" => 1.20,
                _ => 1.70,
            }
        }).collect()
    }

    /// Build a mapping from atom index to residue index
    ///
    /// Returns a vector where entry[i] is the residue index (0-based) for atom i.
    /// Used for aggregating per-atom SASA to per-residue SASA.
    pub fn get_atom_to_residue_map(&self) -> Vec<i32> {
        // Build a mapping using residue_ids
        // Each atom has a residue_id, we need to convert to 0-based index
        let unique_res_ids: Vec<i32> = {
            let mut ids: Vec<i32> = self.residue_ids.clone();
            ids.sort();
            ids.dedup();
            ids
        };

        self.residue_ids.iter().map(|&res_id| {
            unique_res_ids.iter().position(|&id| id == res_id).unwrap_or(0) as i32
        }).collect()
    }

    /// Check if an atom is a hydrogen
    pub fn is_hydrogen(&self, atom_idx: usize) -> bool {
        self.elements.get(atom_idx)
            .map(|e| e.trim().to_uppercase() == "H")
            .unwrap_or(false)
    }

    /// Get indices of heavy atoms (non-hydrogen)
    pub fn get_heavy_atom_indices(&self) -> Vec<usize> {
        (0..self.n_atoms)
            .filter(|&i| !self.is_hydrogen(i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_topology() {
        let path = Path::new("data/curated_14/topologies/1L2Y_topology.json");
        if path.exists() {
            let topology = PrismTopology::load(path).unwrap();
            assert_eq!(topology.n_residues, 20);
            assert_eq!(topology.ca_indices.len(), 20);

            let ca_coords = topology.get_ca_coordinates();
            assert_eq!(ca_coords.len(), 20);
        }
    }
}
