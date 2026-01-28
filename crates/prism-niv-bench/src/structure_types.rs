//! Data structures for paramyxovirus proteins (Nipah and Hendra)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VirusType {
    Nipah,
    Hendra,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProteinType {
    GProtein,      // Attachment protein (receptor binding)
    FProtein,      // Fusion protein
    MatrixProtein, // Matrix protein
    Polymerase,    // L/P polymerase complex
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainInfo {
    pub id: String,
    pub protein_type: ProteinType,
    pub start_residue: u32,
    pub end_residue: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub id: u32,
    pub name: String,
    pub element: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub residue_id: u32,
    pub chain_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Residue {
    pub id: u32,
    pub name: String,
    pub chain_id: String,
    pub sequence_number: u32,
    pub ca_coords: (f32, f32, f32),
    pub atoms: Vec<Atom>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamyxoStructure {
    pub pdb_id: String,
    pub virus: VirusType,
    pub protein: ProteinType,
    pub chains: Vec<ChainInfo>,
    pub atoms: Vec<Atom>,
    pub residues: Vec<Residue>,
    pub sequence: String,
    pub resolution: Option<f32>,
    pub experimental_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpitopeDefinition {
    pub name: String,
    pub antibody_pdb: String,
    pub interface_residues: Vec<u32>,
    pub is_cryptic: bool,
    pub is_neutralizing: bool,
    pub cross_reactive: bool, // Binds both NiV and HeV
    pub ic50_niv: Option<f32>, // IC50 for Nipah virus (if available)
    pub ic50_hev: Option<f32>, // IC50 for Hendra virus (if available)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticSite {
    pub residues: Vec<u32>,
    pub sasa_apo: Vec<f32>,    // SASA in apo structure
    pub sasa_holo: Vec<f32>,   // SASA in holo structure
    pub p_rmsd: f32,           // Positional RMSD between states
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscapeMutation {
    pub pdb_id: String,
    pub position: u32,
    pub from_aa: char,
    pub to_aa: char,
    pub fold_change_ic50: f32,
    pub antibody: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NivBenchDataset {
    pub structures: Vec<ParamyxoStructure>,
    pub epitopes: HashMap<String, Vec<EpitopeDefinition>>,
    pub cryptic_sites: HashMap<String, Vec<CrypticSite>>,
    pub known_escape_mutations: Vec<EscapeMutation>,
    pub train_structures: Vec<String>,
    pub validation_structures: Vec<String>,
    pub test_structures: Vec<String>,
}

impl ParamyxoStructure {
    /// Get CA (alpha carbon) coordinates for all residues
    pub fn ca_coords(&self) -> Vec<(f32, f32, f32)> {
        self.residues.iter().map(|r| r.ca_coords).collect()
    }

    /// Get number of residues
    pub fn n_residues(&self) -> usize {
        self.residues.len()
    }

    /// Get residue by sequence number
    pub fn get_residue(&self, seq_num: u32) -> Option<&Residue> {
        self.residues.iter().find(|r| r.sequence_number == seq_num)
    }

    /// Check if structure has epitope data
    pub fn has_epitope_data(&self) -> bool {
        // Check if this is a complex with antibody binding data
        self.chains.len() > 1 || self.experimental_method == Some("cryo-EM".to_string())
    }
}