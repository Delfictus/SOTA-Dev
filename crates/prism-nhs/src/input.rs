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

use crate::config::SolventMode;
use crate::rt_targets::{identify_rt_targets, RtTargets};
use crate::solvate::solvate_protein;

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
            let res_id = self.residue_ids[i];
            let res_name = &self.residue_names[res_id];
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

    // === RT Integration Fields [STAGE-1-PREP] ===
    /// Solvent mode (implicit/explicit/hybrid)
    pub solvent_mode: SolventMode,
    /// Water atom indices (if explicit/hybrid)
    pub water_atoms: Option<Vec<usize>>,
    /// RT probe targets for spatial sensing
    pub rt_targets: RtTargets,
    /// Total atom count (protein + waters)
    pub total_atoms: usize,
}

impl NhsPreparedInput {
    /// Create prepared input from topology with RT support
    ///
    /// # Arguments
    /// * `topology` - PRISM-PREP topology
    /// * `grid_spacing` - Grid spacing for detection
    /// * `padding` - Padding around protein
    /// * `solvent_mode` - Solvent mode for RT integration
    ///
    /// # RT Integration [STAGE-1-PREP]
    ///
    /// This method now supports explicit solvation and RT target identification:
    /// 1. If Explicit/Hybrid: Generate water box around protein
    /// 2. Identify RT targets (protein atoms, water O, aromatic centers)
    /// 3. Compute total atom count (protein + waters)
    ///
    /// # Returns
    /// * `Result<Self>` - Prepared input with RT targets
    pub fn from_topology(
        mut topology: PrismPrepTopology,
        grid_spacing: f32,
        padding: f32,
        solvent_mode: &SolventMode,
    ) -> Result<Self> {
        log::info!("Preparing system with {:?} solvent mode", solvent_mode);

        // Step 1: Validate solvent mode
        solvent_mode.validate()
            .context("Invalid solvent mode configuration")?;

        // Step 2: Solvate if explicit mode (hybrid starts implicit, adds waters later in Stage 2b)
        let water_atoms = if solvent_mode.starts_explicit() {
            let padding_angstroms = match solvent_mode {
                SolventMode::Explicit { padding_angstroms } => *padding_angstroms,
                SolventMode::Hybrid { .. } => padding,  // Use grid padding
                SolventMode::Implicit => 0.0,  // Won't reach here due to requires_water check
            };

            let protein_coords = topology.positions.clone();
            let (water_coords, water_indices) = solvate_protein(&topology, &protein_coords, padding_angstroms)
                .context("Failed to solvate protein")?;

            let n_waters = water_indices.len();
            log::info!("Added {} water molecules ({} atoms)", n_waters, water_coords.len() / 3);

            // Append water positions to topology
            topology.positions.extend_from_slice(&water_coords);

            // Append water metadata to all arrays
            for _ in 0..n_waters {
                topology.elements.push("O".to_string());  // Water oxygen
                topology.atom_names.push("O".to_string());
                topology.residue_ids.push(topology.n_residues);  // All waters in same "residue"
                topology.chain_ids.push("W".to_string());  // Water chain
                topology.charges.push(-0.834);  // TIP3P oxygen charge
                topology.masses.push(15.9994);  // Oxygen mass
            }

            topology.n_atoms += n_waters;
            topology.residue_names.push("HOH".to_string());  // Water residue name
            topology.n_residues += 1;  // All waters count as one "residue" for simplicity
            topology.water_oxygens = water_indices.clone();

            Some(water_indices)
        } else {
            log::info!("Implicit mode: no explicit waters added");
            None
        };

        // Step 3: Identify RT targets
        let rt_targets = identify_rt_targets(&topology, solvent_mode)
            .context("Failed to identify RT targets")?;

        log::info!("{}", rt_targets.summary());

        // Step 4: Compute total atom count
        let total_atoms = topology.n_atoms;

        // Step 5: Convert to NHS format
        let (positions, types, charges, residues) = topology.to_nhs_format();
        let grid_origin = topology.grid_origin(padding);
        let grid_dim = topology.required_grid_dim(grid_spacing, padding);

        log::info!(
            "Prepared NHS input: {} atoms ({} protein{}), grid {}³ at origin {:?}",
            total_atoms,
            total_atoms - water_atoms.as_ref().map_or(0, |w| w.len()),
            if water_atoms.is_some() {
                format!(" + {} waters", water_atoms.as_ref().unwrap().len())
            } else {
                String::new()
            },
            grid_dim,
            grid_origin
        );

        Ok(Self {
            topology,
            positions,
            types,
            charges,
            residues,
            grid_origin,
            grid_dim,
            solvent_mode: solvent_mode.clone(),
            water_atoms,
            rt_targets,
            total_atoms,
        })
    }

    /// Load and prepare from file with RT support
    ///
    /// # Arguments
    /// * `path` - Path to PRISM-PREP topology JSON
    /// * `grid_spacing` - Grid spacing for detection
    /// * `padding` - Padding around protein
    /// * `solvent_mode` - Solvent mode for RT integration
    ///
    /// # Returns
    /// * `Result<Self>` - Prepared input with RT targets
    pub fn load(
        path: impl AsRef<Path>,
        grid_spacing: f32,
        padding: f32,
        solvent_mode: &SolventMode,
    ) -> Result<Self> {
        let topology = PrismPrepTopology::load(path)?;
        Self::from_topology(topology, grid_spacing, padding, solvent_mode)
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

    // === RT Integration Tests [STAGE-1-PREP] ===

    fn create_minimal_topology() -> PrismPrepTopology {
        PrismPrepTopology {
            source_pdb: "test.pdb".into(),
            n_atoms: 5,
            n_residues: 2,
            n_chains: 1,
            positions: vec![
                0.0, 0.0, 0.0,   // Atom 0
                1.0, 0.0, 0.0,   // Atom 1
                2.0, 0.0, 0.0,   // Atom 2
                3.0, 0.0, 0.0,   // Atom 3
                4.0, 0.0, 0.0,   // Atom 4
            ],
            elements: vec!["C".into(), "N".into(), "O".into(), "C".into(), "C".into()],
            atom_names: vec!["CA".into(), "N".into(), "O".into(), "CB".into(), "CG".into()],
            residue_names: vec!["ALA".into(), "PHE".into()],
            residue_ids: vec![0, 0, 0, 1, 1],
            chain_ids: vec!["A".into(); 5],
            charges: vec![0.0; 5],
            masses: vec![12.0, 14.0, 16.0, 12.0, 12.0],
            ca_indices: vec![0],
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            lj_params: Vec::new(),
            exclusions: Vec::new(),
            h_clusters: Vec::new(),
            water_oxygens: Vec::new(),
        }
    }

    #[test]
    fn test_prepared_input_implicit_mode() {
        let topology = create_minimal_topology();
        let solvent_mode = SolventMode::Implicit;

        let prepared = NhsPreparedInput::from_topology(
            topology,
            0.5,  // grid_spacing
            10.0, // padding
            &solvent_mode,
        ).expect("Failed to prepare input");

        // Check basic fields
        assert_eq!(prepared.total_atoms, 5, "Should have 5 protein atoms");
        assert!(prepared.water_atoms.is_none(), "Implicit mode should have no waters");

        // Check RT targets
        assert_eq!(prepared.rt_targets.protein_atoms.len(), 5, "Should have 5 protein heavy atoms");
        assert!(prepared.rt_targets.water_atoms.is_none(), "Should have no water targets");
        assert_eq!(prepared.rt_targets.aromatic_centers.len(), 1, "Should have 1 aromatic center (PHE)");

        // Check total target count: 5 protein + 1 aromatic = 6
        assert_eq!(prepared.rt_targets.total_targets, 6);

        println!("Implicit mode: {}", prepared.rt_targets.summary());
    }

    #[test]
    fn test_prepared_input_explicit_mode() {
        let topology = create_minimal_topology();
        let solvent_mode = SolventMode::Explicit { padding_angstroms: 5.0 };

        let prepared = NhsPreparedInput::from_topology(
            topology,
            0.5,  // grid_spacing
            10.0, // padding
            &solvent_mode,
        ).expect("Failed to prepare input");

        // Check that waters were added
        assert!(prepared.water_atoms.is_some(), "Explicit mode should have waters");
        let n_waters = prepared.water_atoms.as_ref().unwrap().len();
        assert!(n_waters > 0, "Should have added some waters");

        // Total atoms = protein + waters
        assert_eq!(prepared.total_atoms, 5 + n_waters);

        // Check RT targets include waters
        assert!(prepared.rt_targets.water_atoms.is_some(), "Should have water targets");
        assert_eq!(prepared.rt_targets.water_atoms.as_ref().unwrap().len(), n_waters);

        println!("Explicit mode: {}", prepared.rt_targets.summary());
        println!("Added {} waters", n_waters);
    }

    #[test]
    fn test_prepared_input_fields_populated() {
        let topology = create_minimal_topology();
        let solvent_mode = SolventMode::Implicit;

        let prepared = NhsPreparedInput::from_topology(
            topology.clone(),
            0.5,
            10.0,
            &solvent_mode,
        ).expect("Failed to prepare input");

        // Check all standard fields
        assert_eq!(prepared.positions.len(), 15, "Should have 5 atoms * 3 coords");
        assert_eq!(prepared.types.len(), 5);
        assert_eq!(prepared.charges.len(), 5);
        assert_eq!(prepared.residues.len(), 5);
        assert_eq!(prepared.grid_dim, topology.required_grid_dim(0.5, 10.0));

        // Check RT fields
        assert!(matches!(prepared.solvent_mode, SolventMode::Implicit));
        assert_eq!(prepared.total_atoms, 5);
    }
}
