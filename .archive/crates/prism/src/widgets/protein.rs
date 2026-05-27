//! Protein visualization widgets
//!
//! ASCII protein structure rendering with secondary structure.

/// Protein visualization state
pub struct ProteinWidget {
    pub residues: Vec<Residue>,
    pub secondary_structure: Vec<SecondaryStructure>,
    pub pockets: Vec<Pocket>,
    pub ligand: Option<Ligand>,
}

pub struct Residue {
    pub name: String,
    pub chain: char,
    pub position: (f64, f64, f64),
}

pub enum SecondaryStructure {
    Helix { start: usize, end: usize },
    Sheet { start: usize, end: usize },
    Loop { start: usize, end: usize },
}

pub struct Pocket {
    pub id: usize,
    pub residues: Vec<usize>,
    pub center: (f64, f64, f64),
    pub volume: f64,
    pub druggability: f64,
}

pub struct Ligand {
    pub atoms: Vec<(f64, f64, f64)>,
    pub bonds: Vec<(usize, usize)>,
}

impl ProteinWidget {
    pub fn new() -> Self {
        Self {
            residues: Vec::new(),
            secondary_structure: Vec::new(),
            pockets: Vec::new(),
            ligand: None,
        }
    }
}

impl Default for ProteinWidget {
    fn default() -> Self {
        Self::new()
    }
}
