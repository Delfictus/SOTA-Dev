//! Residue representation for protein structures

use super::Atom;
use serde::{Deserialize, Serialize};

/// Represents a residue (amino acid) in a protein structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Residue {
    /// Residue name (three-letter code)
    pub name: String,

    /// Chain identifier
    pub chain_id: char,

    /// Sequence number in chain
    pub seq_number: i32,

    /// Model number (defaults to 1)
    pub model: usize,

    /// Insertion code (if any)
    pub insertion_code: Option<char>,

    /// Whether the residue originates from a HETATM record
    pub is_hetatm: bool,

    /// Indices of atoms belonging to this residue
    pub atom_indices: Vec<usize>,

    /// Centroid position
    pub centroid: [f64; 3],

    /// Secondary structure assignment
    pub secondary_structure: SecondaryStructure,

    /// Computed properties
    pub sasa: f64,
    pub hydrophobicity: f64,
    pub conservation_score: f64,
    pub flexibility: f64,
}

/// Secondary structure types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecondaryStructure {
    AlphaHelix,
    BetaSheet,
    Turn,
    Loop,
    Unknown,
}

impl Residue {
    /// Create a new residue
    pub fn new(
        name: String,
        chain_id: char,
        seq_number: i32,
        insertion_code: Option<char>,
    ) -> Self {
        let normalized_name = name.to_ascii_uppercase();
        Self {
            name: normalized_name.clone(),
            chain_id,
            seq_number,
            model: 1,
            insertion_code,
            is_hetatm: false,
            atom_indices: Vec::new(),
            centroid: [0.0, 0.0, 0.0],
            secondary_structure: SecondaryStructure::Unknown,
            sasa: 0.0,
            hydrophobicity: super::hydrophobicity_scale(&normalized_name),
            conservation_score: 0.0,
            flexibility: 0.0,
        }
    }

    /// Update centroid based on atom positions
    pub fn update_centroid(&mut self, atoms: &[Atom]) {
        if self.atom_indices.is_empty() {
            return;
        }

        let mut sum = [0.0, 0.0, 0.0];
        let mut count = 0;

        for &idx in &self.atom_indices {
            if let Some(atom) = atoms.get(idx) {
                sum[0] += atom.coord[0];
                sum[1] += atom.coord[1];
                sum[2] += atom.coord[2];
                count += 1;
            }
        }

        if count > 0 {
            self.centroid = [
                sum[0] / count as f64,
                sum[1] / count as f64,
                sum[2] / count as f64,
            ];
        }
    }

    /// Get CA (alpha carbon) atom index if present
    pub fn ca_index(&self, atoms: &[Atom]) -> Option<usize> {
        self.atom_indices
            .iter()
            .find(|&&idx| atoms.get(idx).map_or(false, |a| a.name == "CA"))
            .copied()
    }

    /// Register an atom index and keep track of hetero/non-hetero composition
    pub fn add_atom(&mut self, atom_idx: usize, atoms: &[Atom]) {
        self.atom_indices.push(atom_idx);
        if let Some(atom) = atoms.get(atom_idx) {
            self.is_hetatm |= atom.is_hetero();
        }
    }

    /// Get backbone atom indices
    pub fn backbone_indices(&self, atoms: &[Atom]) -> Vec<usize> {
        self.atom_indices
            .iter()
            .filter(|&&idx| atoms.get(idx).map_or(false, |a| a.is_backbone()))
            .copied()
            .collect()
    }

    /// Get side chain atom indices
    pub fn sidechain_indices(&self, atoms: &[Atom]) -> Vec<usize> {
        self.atom_indices
            .iter()
            .filter(|&&idx| atoms.get(idx).map_or(false, |a| !a.is_backbone()))
            .copied()
            .collect()
    }

    /// Check if this is a hydrophobic residue
    pub fn is_hydrophobic(&self) -> bool {
        self.hydrophobicity > 0.0
    }

    /// Check if this is a charged residue
    pub fn is_charged(&self) -> bool {
        matches!(self.name.as_str(), "ARG" | "LYS" | "ASP" | "GLU" | "HIS")
    }

    /// Check if this is a polar residue
    pub fn is_polar(&self) -> bool {
        matches!(
            self.name.as_str(),
            "SER" | "THR" | "ASN" | "GLN" | "TYR" | "CYS" | "ARG" | "LYS" | "ASP" | "GLU" | "HIS"
        )
    }

    /// Check if this is an aromatic residue
    pub fn is_aromatic(&self) -> bool {
        matches!(self.name.as_str(), "PHE" | "TYR" | "TRP" | "HIS")
    }

    /// Get one-letter code for this residue
    pub fn one_letter_code(&self) -> char {
        match self.name.as_str() {
            "ALA" => 'A',
            "ARG" => 'R',
            "ASN" => 'N',
            "ASP" => 'D',
            "CYS" => 'C',
            "GLN" => 'Q',
            "GLU" => 'E',
            "GLY" => 'G',
            "HIS" => 'H',
            "ILE" => 'I',
            "LEU" => 'L',
            "LYS" => 'K',
            "MET" => 'M',
            "PHE" => 'F',
            "PRO" => 'P',
            "SER" => 'S',
            "THR" => 'T',
            "TRP" => 'W',
            "TYR" => 'Y',
            "VAL" => 'V',
            _ => 'X',
        }
    }

    /// Calculate total SASA for this residue
    pub fn calculate_sasa(&mut self, atoms: &[Atom]) {
        self.sasa = self
            .atom_indices
            .iter()
            .filter_map(|&idx| atoms.get(idx))
            .map(|a| a.sasa)
            .sum();
    }

    /// Calculate average B-factor (flexibility)
    pub fn calculate_flexibility(&mut self, atoms: &[Atom]) {
        let b_factors: Vec<f64> = self
            .atom_indices
            .iter()
            .filter_map(|&idx| atoms.get(idx))
            .map(|a| a.b_factor)
            .collect();

        if !b_factors.is_empty() {
            self.flexibility = b_factors.iter().sum::<f64>() / b_factors.len() as f64;
        }
    }

    /// Update centroid, SASA, and flexibility together for convenience
    pub fn refresh_derived_properties(&mut self, atoms: &[Atom]) {
        self.update_centroid(atoms);
        self.calculate_sasa(atoms);
        self.calculate_flexibility(atoms);
    }
}

/// Residue properties for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueProperties {
    pub hydrophobicity: f64,
    pub charge: i32,
    pub polarity: bool,
    pub aromaticity: bool,
    pub size: ResidueSize,
    pub hbond_donor_count: usize,
    pub hbond_acceptor_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidueSize {
    Tiny,   // GLY, ALA
    Small,  // SER, THR, CYS, PRO, ASN, ASP
    Medium, // VAL, ILE, LEU, MET, GLN, GLU, HIS
    Large,  // PHE, TYR, TRP, ARG, LYS
}

impl Residue {
    /// Get comprehensive residue properties
    pub fn get_properties(&self) -> ResidueProperties {
        let size = match self.name.as_str() {
            "GLY" | "ALA" => ResidueSize::Tiny,
            "SER" | "THR" | "CYS" | "PRO" | "ASN" | "ASP" => ResidueSize::Small,
            "VAL" | "ILE" | "LEU" | "MET" | "GLN" | "GLU" | "HIS" => ResidueSize::Medium,
            "PHE" | "TYR" | "TRP" | "ARG" | "LYS" => ResidueSize::Large,
            _ => ResidueSize::Medium,
        };

        let charge = match self.name.as_str() {
            "ARG" | "LYS" => 1,
            "ASP" | "GLU" => -1,
            _ => 0,
        };

        let (hbond_donor_count, hbond_acceptor_count) = match self.name.as_str() {
            "ARG" => (5, 1),
            "LYS" => (3, 1),
            "ASN" => (2, 2),
            "GLN" => (2, 2),
            "HIS" => (2, 1),
            "SER" => (2, 2),
            "THR" => (2, 2),
            "TYR" => (2, 2),
            "CYS" => (1, 1),
            "TRP" => (1, 1),
            "ASP" => (1, 3),
            "GLU" => (1, 3),
            _ => (1, 2), // backbone only
        };

        ResidueProperties {
            hydrophobicity: self.hydrophobicity,
            charge,
            polarity: self.is_polar(),
            aromaticity: self.is_aromatic(),
            size,
            hbond_donor_count,
            hbond_acceptor_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residue_creation() {
        let residue = Residue::new("ALA".to_string(), 'A', 1, None);
        assert_eq!(residue.name, "ALA");
        assert_eq!(residue.chain_id, 'A');
        assert_eq!(residue.one_letter_code(), 'A');
    }

    #[test]
    fn test_residue_properties() {
        let arg = Residue::new("ARG".to_string(), 'A', 1, None);
        assert!(arg.is_charged());
        assert!(arg.is_polar());
        assert!(!arg.is_hydrophobic());

        let ile = Residue::new("ILE".to_string(), 'A', 2, None);
        assert!(!ile.is_charged());
        assert!(!ile.is_polar());
        assert!(ile.is_hydrophobic());
    }
}
