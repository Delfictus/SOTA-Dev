//! Oligomer Topology: Biological Assembly Parsing and Interface Detection
//!
//! This module parses REMARK 350 records from PDB files to reconstruct
//! biological assemblies (oligomers) and detect inter-chain interfaces
//! where cryptic binding sites may be hidden.
//!
//! # Key Concepts
//!
//! **Biological Assembly**: The functionally relevant oligomeric state of a protein,
//! as defined by REMARK 350 records. For example, Nipah virus G protein is a tetramer.
//!
//! **Interface Residues**: Residues at chain-chain interfaces that become buried
//! when chains assemble. These are often cryptic sites - accessible in the monomer
//! but hidden in the oligomer.
//!
//! # Example
//!
//! ```rust,ignore
//! use prism_validation::oligomer_topology::{parse_biological_assembly, OligomerTopology};
//!
//! let pdb_content = std::fs::read_to_string("2VWD.pdb")?;
//! let assembly = parse_biological_assembly(&pdb_content, 1)?;
//!
//! // Detect interface residues
//! let interfaces = assembly.detect_interfaces(5.0); // 5Å cutoff
//! for interface in &interfaces {
//!     println!("Interface between chains {} and {}: {} residues",
//!              interface.chain_a, interface.chain_b, interface.residues_a.len());
//! }
//! ```

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// 4x4 transformation matrix for biological assembly
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Matrix4x4 {
    pub data: [[f64; 4]; 4],
}

impl Matrix4x4 {
    /// Identity matrix
    pub fn identity() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Apply transformation to a 3D point
    pub fn transform_point(&self, point: &[f32; 3]) -> [f32; 3] {
        let x = point[0] as f64;
        let y = point[1] as f64;
        let z = point[2] as f64;

        let new_x = self.data[0][0] * x + self.data[0][1] * y + self.data[0][2] * z + self.data[0][3];
        let new_y = self.data[1][0] * x + self.data[1][1] * y + self.data[1][2] * z + self.data[1][3];
        let new_z = self.data[2][0] * x + self.data[2][1] * y + self.data[2][2] * z + self.data[2][3];

        [new_x as f32, new_y as f32, new_z as f32]
    }

    /// Check if this is essentially the identity matrix
    pub fn is_identity(&self) -> bool {
        let eps = 1e-6;
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (self.data[i][j] - expected).abs() > eps {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for Matrix4x4 {
    fn default() -> Self {
        Self::identity()
    }
}

/// Biological assembly definition from REMARK 350
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalAssembly {
    /// Biomolecule number (1, 2, etc.)
    pub biomolecule_id: u32,
    /// Chains included in this assembly
    pub chains: Vec<char>,
    /// Transformation matrices to apply
    pub transforms: Vec<Matrix4x4>,
    /// Author-determined assembly (vs software-determined)
    pub is_author_determined: bool,
    /// Total number of chains after applying transforms
    pub total_chains: usize,
}

impl BiologicalAssembly {
    /// Create a new biological assembly
    pub fn new(biomolecule_id: u32) -> Self {
        Self {
            biomolecule_id,
            chains: Vec::new(),
            transforms: Vec::new(),
            is_author_determined: false,
            total_chains: 0,
        }
    }

    /// Add a chain to the assembly
    pub fn add_chain(&mut self, chain_id: char) {
        if !self.chains.contains(&chain_id) {
            self.chains.push(chain_id);
        }
    }

    /// Add a transformation matrix
    pub fn add_transform(&mut self, transform: Matrix4x4) {
        self.transforms.push(transform);
    }

    /// Calculate total chains after applying all transforms
    pub fn calculate_total_chains(&mut self) {
        self.total_chains = self.chains.len() * self.transforms.len().max(1);
    }

    /// Get oligomeric state name
    pub fn oligomeric_state(&self) -> &'static str {
        match self.total_chains {
            1 => "monomer",
            2 => "dimer",
            3 => "trimer",
            4 => "tetramer",
            5 => "pentamer",
            6 => "hexamer",
            8 => "octamer",
            12 => "dodecamer",
            _ => "oligomer",
        }
    }
}

/// Atom with chain information for oligomer topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OligomerAtom {
    /// Atom serial number
    pub serial: i32,
    /// Atom name (CA, CB, etc.)
    pub name: String,
    /// Residue name (ALA, GLY, etc.)
    pub residue_name: String,
    /// Chain identifier
    pub chain_id: char,
    /// Residue sequence number
    pub residue_num: i32,
    /// Insertion code (if any)
    pub insertion_code: char,
    /// Coordinates
    pub coords: [f32; 3],
    /// B-factor
    pub b_factor: f32,
    /// Is this atom from the original asymmetric unit?
    pub is_original: bool,
    /// Transform index (which symmetry mate)
    pub transform_index: usize,
}

impl OligomerAtom {
    /// Get unique chain identifier including transform
    pub fn full_chain_id(&self) -> String {
        if self.transform_index == 0 {
            self.chain_id.to_string()
        } else {
            format!("{}{}", self.chain_id, self.transform_index)
        }
    }
}

/// Topology of an oligomeric assembly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OligomerTopology {
    /// PDB ID
    pub pdb_id: String,
    /// Which biological assembly this represents
    pub assembly_id: u32,
    /// All atoms organized by chain
    pub chains: HashMap<String, Vec<OligomerAtom>>,
    /// CA-only coordinates per chain (for faster distance calculations)
    pub ca_coords: HashMap<String, Vec<(i32, [f32; 3])>>,
    /// Inter-chain contacts
    pub inter_chain_contacts: Vec<(AtomRef, AtomRef, f32)>,
    /// Interface residues per chain
    pub interface_residues: HashMap<String, HashSet<i32>>,
    /// Oligomeric state
    pub oligomeric_state: String,
    /// Number of chains
    pub n_chains: usize,
}

/// Reference to an atom for contact tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AtomRef {
    pub chain_id: String,
    pub residue_num: i32,
    pub atom_name: String,
}

/// Region at the interface between chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceRegion {
    /// First chain
    pub chain_a: String,
    /// Second chain
    pub chain_b: String,
    /// Residues from chain A at the interface
    pub residues_a: Vec<i32>,
    /// Residues from chain B at the interface
    pub residues_b: Vec<i32>,
    /// SASA lost upon complex formation (Å²)
    pub buried_sasa: f64,
    /// Total interface area (Å²)
    pub interface_area: f64,
    /// Center of the interface
    pub center: [f32; 3],
    /// Cryptic score boost for interface residues
    pub cryptic_boost: f64,
}

impl InterfaceRegion {
    /// Get all residues in this interface
    pub fn all_residues(&self) -> Vec<(String, i32)> {
        let mut residues = Vec::new();
        for &res in &self.residues_a {
            residues.push((self.chain_a.clone(), res));
        }
        for &res in &self.residues_b {
            residues.push((self.chain_b.clone(), res));
        }
        residues
    }
}

impl OligomerTopology {
    /// Create a new empty topology
    pub fn new(pdb_id: &str, assembly_id: u32) -> Self {
        Self {
            pdb_id: pdb_id.to_string(),
            assembly_id,
            chains: HashMap::new(),
            ca_coords: HashMap::new(),
            inter_chain_contacts: Vec::new(),
            interface_residues: HashMap::new(),
            oligomeric_state: "unknown".to_string(),
            n_chains: 0,
        }
    }

    /// Build topology from PDB content and biological assembly
    pub fn from_biological_assembly(
        pdb_content: &str,
        assembly: &BiologicalAssembly,
    ) -> Result<Self> {
        let mut topology = Self::new("", assembly.biomolecule_id);

        // Parse atoms from PDB
        let atoms = parse_pdb_atoms(pdb_content)?;

        // Filter to chains in the assembly
        let assembly_chains: HashSet<char> = assembly.chains.iter().copied().collect();
        let filtered_atoms: Vec<_> = atoms
            .into_iter()
            .filter(|a| assembly_chains.contains(&a.chain_id))
            .collect();

        // Apply transforms to create full assembly
        for (transform_idx, transform) in assembly.transforms.iter().enumerate() {
            for atom in &filtered_atoms {
                let transformed_coords = if transform.is_identity() {
                    atom.coords
                } else {
                    transform.transform_point(&atom.coords)
                };

                let transformed_atom = OligomerAtom {
                    serial: atom.serial,
                    name: atom.name.clone(),
                    residue_name: atom.residue_name.clone(),
                    chain_id: atom.chain_id,
                    residue_num: atom.residue_num,
                    insertion_code: atom.insertion_code,
                    coords: transformed_coords,
                    b_factor: atom.b_factor,
                    is_original: transform_idx == 0,
                    transform_index: transform_idx,
                };

                let chain_key = transformed_atom.full_chain_id();

                // Add to chains map
                topology.chains
                    .entry(chain_key.clone())
                    .or_insert_with(Vec::new)
                    .push(transformed_atom.clone());

                // Track CA coordinates separately
                if transformed_atom.name == "CA" {
                    topology.ca_coords
                        .entry(chain_key)
                        .or_insert_with(Vec::new)
                        .push((transformed_atom.residue_num, transformed_atom.coords));
                }
            }
        }

        // If no transforms were applied, use original atoms
        if assembly.transforms.is_empty() {
            for atom in filtered_atoms {
                let chain_key = atom.chain_id.to_string();

                // Track CA coordinates
                if atom.name == "CA" {
                    topology.ca_coords
                        .entry(chain_key.clone())
                        .or_insert_with(Vec::new)
                        .push((atom.residue_num, atom.coords));
                }

                topology.chains
                    .entry(chain_key)
                    .or_insert_with(Vec::new)
                    .push(OligomerAtom {
                        serial: atom.serial,
                        name: atom.name,
                        residue_name: atom.residue_name,
                        chain_id: atom.chain_id,
                        residue_num: atom.residue_num,
                        insertion_code: atom.insertion_code,
                        coords: atom.coords,
                        b_factor: atom.b_factor,
                        is_original: true,
                        transform_index: 0,
                    });
            }
        }

        topology.n_chains = topology.chains.len();
        topology.oligomeric_state = match topology.n_chains {
            1 => "monomer".to_string(),
            2 => "dimer".to_string(),
            3 => "trimer".to_string(),
            4 => "tetramer".to_string(),
            6 => "hexamer".to_string(),
            _ => format!("{}-mer", topology.n_chains),
        };

        Ok(topology)
    }

    /// Detect inter-chain interfaces
    ///
    /// # Arguments
    /// * `distance_cutoff` - Maximum CA-CA distance for interface residue (typically 8-10 Å)
    ///
    /// # Returns
    /// Vector of interface regions between chain pairs
    pub fn detect_interfaces(&mut self, distance_cutoff: f32) -> Vec<InterfaceRegion> {
        let cutoff_sq = distance_cutoff * distance_cutoff;
        let chain_ids: Vec<String> = self.ca_coords.keys().cloned().collect();
        let mut interfaces = Vec::new();

        // Compare all chain pairs
        for i in 0..chain_ids.len() {
            for j in (i + 1)..chain_ids.len() {
                let chain_a = &chain_ids[i];
                let chain_b = &chain_ids[j];

                let coords_a = &self.ca_coords[chain_a];
                let coords_b = &self.ca_coords[chain_b];

                let mut residues_a = HashSet::new();
                let mut residues_b = HashSet::new();
                let mut contact_coords = Vec::new();

                // Find interface residues
                for &(res_a, coord_a) in coords_a {
                    for &(res_b, coord_b) in coords_b {
                        let dx = coord_a[0] - coord_b[0];
                        let dy = coord_a[1] - coord_b[1];
                        let dz = coord_a[2] - coord_b[2];
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        if dist_sq < cutoff_sq {
                            residues_a.insert(res_a);
                            residues_b.insert(res_b);
                            contact_coords.push(coord_a);
                            contact_coords.push(coord_b);

                            // Track contact
                            let dist = dist_sq.sqrt();
                            self.inter_chain_contacts.push((
                                AtomRef {
                                    chain_id: chain_a.clone(),
                                    residue_num: res_a,
                                    atom_name: "CA".to_string(),
                                },
                                AtomRef {
                                    chain_id: chain_b.clone(),
                                    residue_num: res_b,
                                    atom_name: "CA".to_string(),
                                },
                                dist,
                            ));
                        }
                    }
                }

                // Only create interface if there are contacts
                if !residues_a.is_empty() {
                    // Compute interface center
                    let center = if !contact_coords.is_empty() {
                        let n = contact_coords.len() as f32;
                        let sum_x: f32 = contact_coords.iter().map(|c| c[0]).sum();
                        let sum_y: f32 = contact_coords.iter().map(|c| c[1]).sum();
                        let sum_z: f32 = contact_coords.iter().map(|c| c[2]).sum();
                        [sum_x / n, sum_y / n, sum_z / n]
                    } else {
                        [0.0, 0.0, 0.0]
                    };

                    // Estimate interface area (rough: ~100 Å² per residue)
                    let interface_area = (residues_a.len() + residues_b.len()) as f64 * 100.0;

                    // Store interface residues
                    self.interface_residues
                        .entry(chain_a.clone())
                        .or_insert_with(HashSet::new)
                        .extend(&residues_a);
                    self.interface_residues
                        .entry(chain_b.clone())
                        .or_insert_with(HashSet::new)
                        .extend(&residues_b);

                    interfaces.push(InterfaceRegion {
                        chain_a: chain_a.clone(),
                        chain_b: chain_b.clone(),
                        residues_a: residues_a.into_iter().collect(),
                        residues_b: residues_b.into_iter().collect(),
                        buried_sasa: interface_area * 0.5, // Rough estimate
                        interface_area,
                        center,
                        cryptic_boost: 0.30, // +30% boost for interface residues (Phase 5.2)
                    });
                }
            }
        }

        interfaces
    }

    /// Check if a residue is at an interface
    pub fn is_interface_residue(&self, chain_id: &str, residue_num: i32) -> bool {
        self.interface_residues
            .get(chain_id)
            .map_or(false, |residues| residues.contains(&residue_num))
    }

    /// Get the interface boost for a residue
    ///
    /// Interface residues are potentially cryptic - they're buried in the oligomer
    /// but become accessible when the complex dissociates.
    ///
    /// # Phase 5.2 Enhancement
    /// Increased from 15% to 30% based on:
    /// - CrypTothML shows interface residues are 2x more likely to be cryptic
    /// - Epitopes cluster at protein-protein interfaces
    /// - Interface burial creates druggable pockets upon dissociation
    pub fn get_interface_boost(&self, chain_id: &str, residue_num: i32) -> f64 {
        if self.is_interface_residue(chain_id, residue_num) {
            0.30 // +30% boost to cryptic score (Phase 5.2)
        } else {
            0.0
        }
    }

    /// Get statistics about the oligomer
    pub fn stats(&self) -> OligomerStats {
        let total_residues: usize = self.ca_coords.values().map(|v| v.len()).sum();
        let interface_residues: usize = self.interface_residues.values()
            .map(|s| s.len())
            .sum();

        OligomerStats {
            n_chains: self.n_chains,
            oligomeric_state: self.oligomeric_state.clone(),
            total_residues,
            interface_residues,
            interface_fraction: if total_residues > 0 {
                interface_residues as f64 / total_residues as f64
            } else {
                0.0
            },
            n_inter_chain_contacts: self.inter_chain_contacts.len(),
        }
    }
}

/// Statistics about an oligomer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OligomerStats {
    pub n_chains: usize,
    pub oligomeric_state: String,
    pub total_residues: usize,
    pub interface_residues: usize,
    pub interface_fraction: f64,
    pub n_inter_chain_contacts: usize,
}

/// Parse REMARK 350 records from PDB content
pub fn parse_remark_350(pdb_content: &str) -> Vec<BiologicalAssembly> {
    let mut assemblies: HashMap<u32, BiologicalAssembly> = HashMap::new();
    let mut current_biomol: Option<u32> = None;
    let mut current_matrix: [[f64; 4]; 4] = [[0.0; 4]; 4];
    let mut current_matrix_id: Option<usize> = None;

    for line in pdb_content.lines() {
        if !line.starts_with("REMARK 350") {
            continue;
        }

        let content = if line.len() > 11 { &line[11..] } else { "" };

        // Parse BIOMOLECULE line
        if content.contains("BIOMOLECULE:") {
            if let Some(start) = content.find("BIOMOLECULE:") {
                let num_str = content[start + 12..].trim().split_whitespace().next();
                if let Some(num) = num_str.and_then(|s| s.parse::<u32>().ok()) {
                    current_biomol = Some(num);
                    assemblies.entry(num).or_insert_with(|| BiologicalAssembly::new(num));
                }
            }
        }

        // Parse APPLY THE FOLLOWING TO CHAINS line
        if content.contains("APPLY THE FOLLOWING TO CHAINS:") || content.contains("AND CHAINS:") {
            if let Some(biomol) = current_biomol {
                // Extract chain IDs
                let chain_part = if content.contains("CHAINS:") {
                    content.split("CHAINS:").last().unwrap_or("")
                } else {
                    ""
                };

                for part in chain_part.split(',') {
                    let chain_str = part.trim();
                    if chain_str.len() == 1 {
                        if let Some(assembly) = assemblies.get_mut(&biomol) {
                            assembly.add_chain(chain_str.chars().next().unwrap());
                        }
                    }
                }
            }
        }

        // Parse BIOMT lines
        if content.trim().starts_with("BIOMT") {
            let parts: Vec<&str> = content.split_whitespace().collect();
            if parts.len() >= 5 {
                // BIOMT1, BIOMT2, or BIOMT3
                let row = match parts[0] {
                    "BIOMT1" => Some(0),
                    "BIOMT2" => Some(1),
                    "BIOMT3" => Some(2),
                    _ => None,
                };

                if let Some(row_idx) = row {
                    // Parse matrix ID
                    if let Ok(matrix_id) = parts[1].parse::<usize>() {
                        if current_matrix_id != Some(matrix_id) {
                            // Start new matrix
                            current_matrix = [[0.0; 4]; 4];
                            current_matrix[3] = [0.0, 0.0, 0.0, 1.0];
                            current_matrix_id = Some(matrix_id);
                        }

                        // Parse rotation and translation
                        if parts.len() >= 6 {
                            current_matrix[row_idx][0] = parts[2].parse().unwrap_or(0.0);
                            current_matrix[row_idx][1] = parts[3].parse().unwrap_or(0.0);
                            current_matrix[row_idx][2] = parts[4].parse().unwrap_or(0.0);
                            if parts.len() >= 7 {
                                current_matrix[row_idx][3] = parts[5].parse().unwrap_or(0.0);
                            }
                        }

                        // If this was the last row (BIOMT3), add the matrix
                        if row_idx == 2 {
                            if let Some(biomol) = current_biomol {
                                if let Some(assembly) = assemblies.get_mut(&biomol) {
                                    assembly.add_transform(Matrix4x4 { data: current_matrix });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Parse author determination
        if content.contains("AUTHOR DETERMINED") {
            if let Some(biomol) = current_biomol {
                if let Some(assembly) = assemblies.get_mut(&biomol) {
                    assembly.is_author_determined = true;
                }
            }
        }
    }

    // Calculate total chains for each assembly
    let mut result: Vec<BiologicalAssembly> = assemblies.into_values().collect();
    for assembly in &mut result {
        assembly.calculate_total_chains();
    }

    // Sort by biomolecule ID
    result.sort_by_key(|a| a.biomolecule_id);
    result
}

/// Parse a biological assembly by ID
pub fn parse_biological_assembly(pdb_content: &str, assembly_id: u32) -> Result<BiologicalAssembly> {
    let assemblies = parse_remark_350(pdb_content);

    assemblies
        .into_iter()
        .find(|a| a.biomolecule_id == assembly_id)
        .ok_or_else(|| anyhow::anyhow!("Biological assembly {} not found", assembly_id))
}

/// Parse atom records from PDB content
fn parse_pdb_atoms(pdb_content: &str) -> Result<Vec<OligomerAtom>> {
    let mut atoms = Vec::new();

    for line in pdb_content.lines() {
        if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
            continue;
        }

        if line.len() < 54 {
            continue;
        }

        // Parse atom fields (PDB format is fixed-width)
        let serial: i32 = line[6..11].trim().parse().unwrap_or(0);
        let name = line[12..16].trim().to_string();
        let residue_name = line[17..20].trim().to_string();
        let chain_id = line[21..22].chars().next().unwrap_or(' ');
        let residue_num: i32 = line[22..26].trim().parse().unwrap_or(0);
        let insertion_code = line[26..27].chars().next().unwrap_or(' ');

        let x: f32 = line[30..38].trim().parse().unwrap_or(0.0);
        let y: f32 = line[38..46].trim().parse().unwrap_or(0.0);
        let z: f32 = line[46..54].trim().parse().unwrap_or(0.0);

        let b_factor: f32 = if line.len() >= 66 {
            line[60..66].trim().parse().unwrap_or(0.0)
        } else {
            0.0
        };

        // Skip non-standard residues for now (HOH, ligands, etc.)
        if line.starts_with("HETATM") {
            continue;
        }

        atoms.push(OligomerAtom {
            serial,
            name,
            residue_name,
            chain_id,
            residue_num,
            insertion_code,
            coords: [x, y, z],
            b_factor,
            is_original: true,
            transform_index: 0,
        });
    }

    Ok(atoms)
}

/// Compute cryptic score boost based on interface proximity
///
/// Residues near interfaces get a boost because they become accessible
/// when the oligomer dissociates.
pub fn compute_interface_boost(
    residue_num: i32,
    chain_id: &str,
    interfaces: &[InterfaceRegion],
) -> f64 {
    // Check if directly at interface
    for interface in interfaces {
        if (interface.chain_a == chain_id && interface.residues_a.contains(&residue_num))
            || (interface.chain_b == chain_id && interface.residues_b.contains(&residue_num))
        {
            return interface.cryptic_boost;
        }
    }

    0.0
}

/// Henipavirus structure collection for validation
pub mod henipavirus {
    //! Henipavirus structure collection
    //!
    //! Reference structures for Nipah and Hendra virus validation.

    /// Henipavirus structure entry
    #[derive(Debug, Clone)]
    pub struct HenipavirusStructure {
        pub pdb_id: &'static str,
        pub protein: &'static str,
        pub virus: &'static str,
        pub state: &'static str,
        pub resolution: f64,
        pub oligomeric_state: &'static str,
        pub notes: &'static str,
    }

    /// Collection of Henipavirus structures for benchmarking
    pub const HENIPAVIRUS_STRUCTURES: &[HenipavirusStructure] = &[
        HenipavirusStructure {
            pdb_id: "2VWD",
            protein: "G (Attachment Glycoprotein)",
            virus: "Nipah",
            state: "Apo",
            resolution: 3.5,
            oligomeric_state: "dimer",
            notes: "Dimeric in asymmetric unit, tetramer in biological assembly",
        },
        HenipavirusStructure {
            pdb_id: "3D11",
            protein: "G + m102.4 Fab",
            virus: "Nipah",
            state: "Holo (antibody complex)",
            resolution: 3.0,
            oligomeric_state: "dimer",
            notes: "m102.4 antibody complex - validation target",
        },
        HenipavirusStructure {
            pdb_id: "6VY4",
            protein: "G (Attachment Glycoprotein)",
            virus: "Hendra",
            state: "Apo",
            resolution: 2.3,
            oligomeric_state: "monomer",
            notes: "Higher resolution HeV G structure",
        },
        HenipavirusStructure {
            pdb_id: "6CMG",
            protein: "F (Fusion Glycoprotein)",
            virus: "Nipah",
            state: "Pre-fusion",
            resolution: 3.2,
            oligomeric_state: "trimer",
            notes: "Fusion protein in pre-fusion state",
        },
    ];

    /// Get structure by PDB ID
    pub fn get_structure(pdb_id: &str) -> Option<&'static HenipavirusStructure> {
        HENIPAVIRUS_STRUCTURES.iter().find(|s| s.pdb_id == pdb_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_identity() {
        let m = Matrix4x4::identity();
        assert!(m.is_identity());

        let point = [1.0f32, 2.0, 3.0];
        let transformed = m.transform_point(&point);
        assert!((transformed[0] - 1.0).abs() < 0.001);
        assert!((transformed[1] - 2.0).abs() < 0.001);
        assert!((transformed[2] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_matrix_translation() {
        let mut m = Matrix4x4::identity();
        m.data[0][3] = 10.0; // Translate X by 10
        m.data[1][3] = 20.0; // Translate Y by 20
        m.data[2][3] = 30.0; // Translate Z by 30

        let point = [1.0f32, 2.0, 3.0];
        let transformed = m.transform_point(&point);
        assert!((transformed[0] - 11.0).abs() < 0.001);
        assert!((transformed[1] - 22.0).abs() < 0.001);
        assert!((transformed[2] - 33.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_remark_350_simple() {
        let pdb_content = r#"
REMARK 350
REMARK 350 BIOMOLECULE: 1
REMARK 350 AUTHOR DETERMINED BIOLOGICAL UNIT: TETRAMER
REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, B
REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
REMARK 350   BIOMT1   2 -1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   2  0.000000 -1.000000  0.000000        0.00000
REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000
"#;

        let assemblies = parse_remark_350(pdb_content);
        assert_eq!(assemblies.len(), 1);

        let assembly = &assemblies[0];
        assert_eq!(assembly.biomolecule_id, 1);
        assert!(assembly.is_author_determined);
        assert_eq!(assembly.chains.len(), 2);
        assert!(assembly.chains.contains(&'A'));
        assert!(assembly.chains.contains(&'B'));
        assert_eq!(assembly.transforms.len(), 2);

        // First transform should be identity
        assert!(assembly.transforms[0].is_identity());

        // Second transform should have rotation
        assert!(!assembly.transforms[1].is_identity());
    }

    #[test]
    fn test_biological_assembly_oligomeric_state() {
        let mut assembly = BiologicalAssembly::new(1);
        assembly.chains = vec!['A'];
        assembly.calculate_total_chains();
        assert_eq!(assembly.oligomeric_state(), "monomer");

        assembly.chains = vec!['A', 'B'];
        assembly.transforms = vec![Matrix4x4::identity()];
        assembly.calculate_total_chains();
        assert_eq!(assembly.oligomeric_state(), "dimer");

        assembly.transforms = vec![Matrix4x4::identity(), Matrix4x4::identity()];
        assembly.calculate_total_chains();
        assert_eq!(assembly.oligomeric_state(), "tetramer");
    }

    #[test]
    fn test_oligomer_stats() {
        let mut topology = OligomerTopology::new("TEST", 1);
        topology.n_chains = 4;
        topology.oligomeric_state = "tetramer".to_string();

        topology.ca_coords.insert("A".to_string(), vec![(1, [0.0, 0.0, 0.0])]);
        topology.ca_coords.insert("B".to_string(), vec![(1, [0.0, 0.0, 0.0])]);

        topology.interface_residues.insert("A".to_string(), [1].into_iter().collect());

        let stats = topology.stats();
        assert_eq!(stats.n_chains, 4);
        assert_eq!(stats.total_residues, 2);
        assert_eq!(stats.interface_residues, 1);
    }
}
