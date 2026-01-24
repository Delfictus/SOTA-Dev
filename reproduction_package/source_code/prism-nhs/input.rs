//! NHS Input Module - PRISM-PREP Topology Loader
//!
//! Loads sanitized topology JSON from PRISM-PREP and prepares it for NHS processing.
//!
//! **CRITICAL**: Only PRISM-PREP sanitized structures are valid input.
//! Raw PDB files must go through `prism-prep` first.
//!
//! ## Pipeline Flow
//!
//! ```text
//! Raw PDB → prism-prep → topology.json → NHS-UV Detection Engine
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use prism_nhs::input::PrismPrepTopology;
//!
//! // Load PRISM-PREP output
//! let topology = PrismPrepTopology::load("1L2Y_topology.json")?;
//!
//! // Get NHS-compatible atom data
//! let (positions, types, charges, residues) = topology.to_nhs_format();
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// ATOM TYPES FOR NHS
// ============================================================================

/// Atom type classification for NHS exclusion field
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(i32)]
pub enum NhsAtomType {
    /// Hydrophobic atoms (C aliphatic, S in Met/Cys)
    Hydrophobic = 0,
    /// Polar atoms (O, N with H-bond capability)
    Polar = 1,
    /// Positively charged (Lys NZ, Arg NH)
    ChargedPositive = 2,
    /// Negatively charged (Asp OD, Glu OE)
    ChargedNegative = 3,
    /// Aromatic atoms (Trp, Tyr, Phe rings) - UV bias targets
    Aromatic = 4,
    /// Backbone atoms (N, CA, C, O)
    Backbone = 5,
    /// Water oxygen (if present)
    Water = 6,
}

impl NhsAtomType {
    /// Convert to i32 for GPU
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

// ============================================================================
// PRISM-PREP TOPOLOGY FORMAT
// ============================================================================

/// Bond parameter structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondParam {
    /// First atom index
    pub i: usize,
    /// Second atom index
    pub j: usize,
    /// Equilibrium distance
    pub r0: f64,
    /// Force constant
    pub k: f64,
}

/// Angle parameter structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngleParam {
    /// First atom index
    pub i: usize,
    /// Second atom index
    pub j: usize,
    /// Third atom index (named k_idx to avoid conflict with force constant)
    pub k_idx: usize,
    /// Equilibrium angle
    pub theta0: f64,
    /// Force constant
    pub force_k: f64,
}

/// Dihedral parameter structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DihedralParam {
    /// First atom index
    pub i: usize,
    /// Second atom index
    pub j: usize,
    /// Third atom index (named k_idx to avoid conflict with force constant)
    pub k_idx: usize,
    /// Fourth atom index
    pub l: usize,
    /// Periodicity
    #[serde(default)]
    pub periodicity: i32,
    /// Phase
    #[serde(default)]
    pub phase: f64,
    /// Force constant
    #[serde(default)]
    pub force_k: f64,
}

/// LJ parameter structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LjParam {
    /// Sigma (distance)
    pub sigma: f64,
    /// Epsilon (well depth)
    pub epsilon: f64,
}

/// Hydrogen cluster for SHAKE/RATTLE constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HCluster {
    /// Cluster type (1=CH, 2=CH2, 3=CH3, 4=NH, 5=OH, etc.)
    #[serde(rename = "type")]
    pub cluster_type: i32,
    /// Central atom index
    pub central_atom: usize,
    /// Hydrogen atom indices (-1 for unused slots)
    pub hydrogen_atoms: Vec<i32>,
    /// Bond lengths to each hydrogen
    pub bond_lengths: Vec<f64>,
    /// Number of hydrogens in cluster
    pub n_hydrogens: usize,
    /// Inverse mass of central atom
    pub inv_mass_central: f64,
    /// Inverse mass of hydrogen
    pub inv_mass_h: f64,
}

/// PRISM-PREP topology JSON structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismPrepTopology {
    /// Source PDB file
    pub source_pdb: String,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of residues
    pub n_residues: usize,
    /// Number of chains
    pub n_chains: usize,
    /// Atom positions [x0, y0, z0, x1, y1, z1, ...]
    pub positions: Vec<f32>,
    /// Element symbols
    pub elements: Vec<String>,
    /// Atom names (CA, CB, N, O, etc.)
    pub atom_names: Vec<String>,
    /// Residue names (ALA, GLY, etc.)
    pub residue_names: Vec<String>,
    /// Residue IDs for each atom
    pub residue_ids: Vec<usize>,
    /// Chain IDs
    pub chain_ids: Vec<String>,
    /// Partial charges
    pub charges: Vec<f32>,
    /// Masses
    pub masses: Vec<f32>,
    /// CA atom indices
    pub ca_indices: Vec<usize>,
    /// Bond parameters
    pub bonds: Vec<BondParam>,
    /// Angle parameters
    pub angles: Vec<AngleParam>,
    /// Dihedral parameters
    pub dihedrals: Vec<DihedralParam>,
    /// LJ parameters per atom
    pub lj_params: Vec<LjParam>,
    /// Exclusions per atom (list of atom indices to exclude from nonbonded for each atom)
    pub exclusions: Vec<Vec<usize>>,
    /// Hydrogen clusters for SHAKE/RATTLE constraints
    #[serde(default)]
    pub h_clusters: Vec<HCluster>,
    /// Water oxygen indices
    #[serde(default)]
    pub water_oxygens: Vec<usize>,
}

impl PrismPrepTopology {
    /// Load topology from PRISM-PREP JSON file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        log::info!("Loading PRISM-PREP topology from: {}", path.display());

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read topology file: {}", path.display()))?;

        let topology: Self = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse topology JSON: {}", path.display()))?;

        log::info!(
            "Loaded topology: {} atoms, {} residues, {} chains",
            topology.n_atoms,
            topology.n_residues,
            topology.n_chains
        );

        Ok(topology)
    }

    /// Classify atoms for NHS exclusion field
    ///
    /// Returns atom type for each atom based on:
    /// - Element (C, N, O, S, H)
    /// - Residue type (aromatic: TRP, TYR, PHE)
    /// - Atom name (backbone vs sidechain)
    /// - Partial charge (charged residues)
    pub fn classify_atoms(&self) -> Vec<NhsAtomType> {
        let mut types = Vec::with_capacity(self.n_atoms);

        // Aromatic residues
        let aromatic_residues = ["TRP", "TYR", "PHE", "HIS"];

        // Backbone atom names
        let backbone_atoms = ["N", "CA", "C", "O", "OXT"];

        for i in 0..self.n_atoms {
            let element = &self.elements[i];
            let atom_name = &self.atom_names[i];
            let res_name = &self.residue_names[i];
            let charge = self.charges[i];

            let atom_type = if self.water_oxygens.contains(&i) {
                NhsAtomType::Water
            } else if backbone_atoms.contains(&atom_name.as_str()) {
                NhsAtomType::Backbone
            } else if aromatic_residues.contains(&res_name.as_str())
                && Self::is_ring_atom(atom_name, res_name)
            {
                NhsAtomType::Aromatic
            } else if charge > 0.3 {
                NhsAtomType::ChargedPositive
            } else if charge < -0.3 {
                NhsAtomType::ChargedNegative
            } else if Self::is_polar_atom(element, atom_name, res_name) {
                NhsAtomType::Polar
            } else {
                NhsAtomType::Hydrophobic
            };

            types.push(atom_type);
        }

        types
    }

    /// Check if atom is part of aromatic ring
    fn is_ring_atom(atom_name: &str, res_name: &str) -> bool {
        match res_name {
            "PHE" => matches!(
                atom_name,
                "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ"
            ),
            "TYR" => matches!(
                atom_name,
                "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ" | "OH"
            ),
            "TRP" => matches!(
                atom_name,
                "CG" | "CD1" | "CD2" | "NE1" | "CE2" | "CE3" | "CZ2" | "CZ3" | "CH2"
            ),
            "HIS" => matches!(
                atom_name,
                "CG" | "ND1" | "CD2" | "CE1" | "NE2"
            ),
            _ => false,
        }
    }

    /// Check if atom is polar
    fn is_polar_atom(element: &str, atom_name: &str, res_name: &str) -> bool {
        match element {
            "O" => true,
            "N" => {
                // Amide nitrogen is polar, but aromatic N in His handled separately
                !matches!(res_name, "PRO") || atom_name != "N"
            }
            "S" => {
                // Cys SG can be polar (H-bond acceptor)
                res_name == "CYS" && atom_name == "SG"
            }
            _ => false,
        }
    }

    /// Convert to NHS GPU format
    ///
    /// Returns (positions, types, charges, residue_ids) ready for NhsGpuEngine
    pub fn to_nhs_format(&self) -> (Vec<f32>, Vec<i32>, Vec<f32>, Vec<i32>) {
        let types = self.classify_atoms();
        let types_i32: Vec<i32> = types.iter().map(|t| t.as_i32()).collect();
        let residues_i32: Vec<i32> = self.residue_ids.iter().map(|&r| r as i32).collect();

        (
            self.positions.clone(),
            types_i32,
            self.charges.clone(),
            residues_i32,
        )
    }

    /// Get atom positions as array of [x,y,z] triples
    pub fn positions_as_xyz(&self) -> Vec<[f32; 3]> {
        self.positions
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect()
    }

    /// Compute bounding box for grid origin
    pub fn bounding_box(&self) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for chunk in self.positions.chunks(3) {
            for i in 0..3 {
                min[i] = min[i].min(chunk[i]);
                max[i] = max[i].max(chunk[i]);
            }
        }

        (min, max)
    }

    /// Get grid origin with padding
    pub fn grid_origin(&self, padding: f32) -> [f32; 3] {
        let (min, _) = self.bounding_box();
        [min[0] - padding, min[1] - padding, min[2] - padding]
    }

    /// Get required grid dimension for given spacing
    pub fn required_grid_dim(&self, spacing: f32, padding: f32) -> usize {
        let (min, max) = self.bounding_box();
        let size = [
            max[0] - min[0] + 2.0 * padding,
            max[1] - min[1] + 2.0 * padding,
            max[2] - min[2] + 2.0 * padding,
        ];
        let max_size = size[0].max(size[1]).max(size[2]);
        ((max_size / spacing).ceil() as usize).max(32)
    }

    /// Get aromatic residue indices for UV bias targeting
    pub fn aromatic_residues(&self) -> Vec<usize> {
        let aromatic = ["TRP", "TYR", "PHE"];
        let mut residues = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for (i, res_name) in self.residue_names.iter().enumerate() {
            let res_id = self.residue_ids[i];
            if aromatic.contains(&res_name.as_str()) && !seen.contains(&res_id) {
                residues.push(res_id);
                seen.insert(res_id);
            }
        }

        residues
    }

    /// Get atom type statistics
    pub fn atom_type_stats(&self) -> HashMap<NhsAtomType, usize> {
        let types = self.classify_atoms();
        let mut stats = HashMap::new();

        for t in types {
            *stats.entry(t).or_insert(0) += 1;
        }

        stats
    }
}

// ============================================================================
// NHS INPUT PREPARED FOR GPU
// ============================================================================

/// Prepared input data for NHS GPU processing
#[derive(Debug, Clone)]
pub struct NhsPreparedInput {
    /// Source topology
    pub topology: PrismPrepTopology,
    /// Flat positions array
    pub positions: Vec<f32>,
    /// Atom types as i32
    pub types: Vec<i32>,
    /// Partial charges
    pub charges: Vec<f32>,
    /// Residue IDs as i32
    pub residues: Vec<i32>,
    /// Grid origin
    pub grid_origin: [f32; 3],
    /// Required grid dimension
    pub grid_dim: usize,
}

impl NhsPreparedInput {
    /// Create prepared input from topology
    pub fn from_topology(topology: PrismPrepTopology, grid_spacing: f32, padding: f32) -> Self {
        let (positions, types, charges, residues) = topology.to_nhs_format();
        let grid_origin = topology.grid_origin(padding);
        let grid_dim = topology.required_grid_dim(grid_spacing, padding);

        log::info!(
            "Prepared NHS input: {} atoms, grid {}³ at origin {:?}",
            topology.n_atoms,
            grid_dim,
            grid_origin
        );

        Self {
            topology,
            positions,
            types,
            charges,
            residues,
            grid_origin,
            grid_dim,
        }
    }

    /// Load and prepare from file
    pub fn load(path: impl AsRef<Path>, grid_spacing: f32, padding: f32) -> Result<Self> {
        let topology = PrismPrepTopology::load(path)?;
        Ok(Self::from_topology(topology, grid_spacing, padding))
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_type_values() {
        assert_eq!(NhsAtomType::Hydrophobic.as_i32(), 0);
        assert_eq!(NhsAtomType::Aromatic.as_i32(), 4);
    }

    #[test]
    fn test_is_ring_atom() {
        assert!(PrismPrepTopology::is_ring_atom("CG", "PHE"));
        assert!(PrismPrepTopology::is_ring_atom("NE1", "TRP"));
        assert!(!PrismPrepTopology::is_ring_atom("CB", "PHE"));
    }

    #[test]
    fn test_topology_load() {
        // This test requires an actual file
        let path = "../../data/curated_14/topologies/1L2Y_topology.json";
        if std::path::Path::new(path).exists() {
            let topology = PrismPrepTopology::load(path).expect("Failed to load topology");
            assert!(topology.n_atoms > 0);
            assert!(topology.n_residues > 0);

            let stats = topology.atom_type_stats();
            println!("Atom type stats: {:?}", stats);
        }
    }
}
