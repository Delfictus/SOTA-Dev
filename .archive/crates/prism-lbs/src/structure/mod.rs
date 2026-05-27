//! Protein structure representation and parsing

pub mod atom;
pub mod pdb_parser;
pub mod residue;
pub mod surface;

pub use atom::Atom;
pub use pdb_parser::{PdbParseOptions, ProteinStructure};
pub use residue::Residue;
pub use surface::SurfaceComputer;

use lazy_static::lazy_static;
use std::collections::HashMap;

lazy_static! {
    /// Van der Waals radii for common elements (in Angstroms)
    pub static ref VDW_RADII: HashMap<String, f64> = {
        let mut m = HashMap::new();
        m.insert("H".to_string(), 1.20);
        m.insert("C".to_string(), 1.70);
        m.insert("N".to_string(), 1.55);
        m.insert("O".to_string(), 1.52);
        m.insert("S".to_string(), 1.80);
        m.insert("P".to_string(), 1.80);
        m.insert("F".to_string(), 1.47);
        m.insert("CL".to_string(), 1.75);
        m.insert("BR".to_string(), 1.85);
        m.insert("I".to_string(), 1.98);
        m.insert("FE".to_string(), 1.40);
        m.insert("ZN".to_string(), 1.39);
        m.insert("CA".to_string(), 1.97);
        m.insert("MG".to_string(), 1.73);
        m.insert("NA".to_string(), 2.27);
        m.insert("K".to_string(), 2.75);
        m
    };
}

/// Kyte-Doolittle hydrophobicity scale
pub fn hydrophobicity_scale(residue: &str) -> f64 {
    match residue {
        "ILE" => 4.5,
        "VAL" => 4.2,
        "LEU" => 3.8,
        "PHE" => 2.8,
        "CYS" => 2.5,
        "MET" => 1.9,
        "ALA" => 1.8,
        "GLY" => -0.4,
        "THR" => -0.7,
        "SER" => -0.8,
        "TRP" => -0.9,
        "TYR" => -1.3,
        "PRO" => -1.6,
        "HIS" => -3.2,
        "GLU" => -3.5,
        "GLN" => -3.5,
        "ASP" => -3.5,
        "ASN" => -3.5,
        "LYS" => -3.9,
        "ARG" => -4.5,
        _ => 0.0,
    }
}

/// Check if residue can be hydrogen bond donor
pub fn is_hbond_donor(atom_name: &str, residue_name: &str) -> bool {
    match atom_name {
        "N" | "NE" | "NH1" | "NH2" | "ND1" | "ND2" | "NE1" | "NE2" | "NZ" => true,
        "O" | "OG" | "OH" => {
            residue_name == "SER" || residue_name == "THR" || residue_name == "TYR"
        }
        "SG" => residue_name == "CYS",
        _ => false,
    }
}

/// Check if atom can be hydrogen bond acceptor
pub fn is_hbond_acceptor(atom_name: &str) -> bool {
    matches!(
        atom_name,
        "O" | "OD1" | "OD2" | "OE1" | "OE2" | "OG" | "OH" | "SD" | "SG"
    )
}

/// Compute distance between two 3D points
pub fn distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt()
}

/// Compute distance squared (avoids sqrt for performance)
pub fn distance_squared(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    (p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)
}
