//! Atom representation for protein structures

use serde::{Deserialize, Serialize};

/// Represents a single atom in a protein structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    /// Atom serial number
    pub serial: u32,

    /// Atom name (e.g., "CA", "CB", "N")
    pub name: String,

    /// Residue name (e.g., "ALA", "GLY")
    pub residue_name: String,

    /// Chain identifier
    pub chain_id: char,

    /// Residue sequence number
    pub residue_seq: i32,

    /// Optional insertion code (PDB column 27)
    pub insertion_code: Option<char>,

    /// 3D coordinates [x, y, z] in Angstroms
    pub coord: [f64; 3],

    /// Occupancy factor (0.0-1.0)
    pub occupancy: f64,

    /// B-factor (temperature factor)
    pub b_factor: f64,

    /// Element symbol (e.g., "C", "N", "O")
    pub element: String,

    /// Alternate location identifier (if present)
    pub alt_loc: Option<char>,

    /// Model number (defaults to 1 for single-model structures)
    pub model: usize,

    /// Whether this came from a HETATM record
    pub is_hetatm: bool,

    // Computed properties
    /// Solvent-accessible surface area (SASA) in Å²
    pub sasa: f64,

    /// Hydrophobicity value (Kyte-Doolittle scale)
    pub hydrophobicity: f64,

    /// Partial charge
    pub partial_charge: f64,

    /// Whether atom is on the protein surface
    pub is_surface: bool,

    /// Distance to protein center of mass
    pub depth: f64,

    /// Local curvature at this atom
    pub curvature: f64,
}

impl Atom {
    /// Create a new atom with default computed properties
    pub fn new(
        serial: u32,
        name: String,
        residue_name: String,
        chain_id: char,
        residue_seq: i32,
        coord: [f64; 3],
        occupancy: f64,
        b_factor: f64,
        element: String,
    ) -> Self {
        let residue_name = residue_name.to_ascii_uppercase();
        let residue_name_clone = residue_name.clone();
        let mut atom = Self {
            serial,
            name,
            residue_name,
            chain_id,
            residue_seq,
            insertion_code: None,
            coord,
            occupancy,
            b_factor,
            element: element.to_ascii_uppercase(),
            alt_loc: None,
            model: 1,
            is_hetatm: false,
            sasa: 0.0,
            hydrophobicity: super::hydrophobicity_scale(&residue_name_clone),
            partial_charge: 0.0,
            is_surface: false,
            depth: 0.0,
            curvature: 0.0,
        };
        atom.estimate_partial_charge();
        atom
    }

    /// Check if this is a backbone atom
    pub fn is_backbone(&self) -> bool {
        matches!(self.name.as_str(), "N" | "CA" | "C" | "O")
    }

    /// Check if this is a heavy atom (non-hydrogen)
    pub fn is_heavy(&self) -> bool {
        self.element != "H"
    }

    /// Check if this is a heteroatom (from ligand, water, etc.)
    pub fn is_hetero(&self) -> bool {
        self.is_hetatm
            || self.residue_name.len() != 3
            || matches!(self.residue_name.as_str(), "HOH" | "WAT" | "SOL")
    }

    /// Distance to another atom
    pub fn distance_to(&self, other: &Atom) -> f64 {
        super::distance(&self.coord, &other.coord)
    }

    /// Distance squared (more efficient when comparing distances)
    pub fn distance_squared_to(&self, other: &Atom) -> f64 {
        super::distance_squared(&self.coord, &other.coord)
    }

    /// Estimate partial charge based on atom type
    /// This is a simplified approximation
    pub fn estimate_partial_charge(&mut self) {
        self.partial_charge = match (self.element.as_str(), self.name.as_str()) {
            ("N", _) if self.is_backbone() => -0.47,
            ("N", "NZ") if self.residue_name == "LYS" => 1.0,
            ("N", _) if self.residue_name == "ARG" => 0.5,
            ("O", _) if self.is_backbone() => -0.51,
            ("O", "OD1") | ("O", "OD2") if self.residue_name == "ASP" => -0.5,
            ("O", "OE1") | ("O", "OE2") if self.residue_name == "GLU" => -0.5,
            ("S", _) => -0.2,
            ("C", _) if self.is_backbone() => 0.51,
            _ => 0.0,
        };
    }

    /// Van der Waals radius lookup (falls back to 1.5 Å)
    pub fn vdw_radius(&self) -> f64 {
        super::VDW_RADII.get(&self.element).copied().unwrap_or(1.5)
    }

    /// Whether this atom can donate a hydrogen bond
    pub fn is_hbond_donor(&self) -> bool {
        super::is_hbond_donor(&self.name, &self.residue_name)
    }

    /// Whether this atom can accept a hydrogen bond
    pub fn is_hbond_acceptor(&self) -> bool {
        super::is_hbond_acceptor(&self.name)
    }

    /// Update hydrophobicity and charge estimates based on residue context
    pub fn refresh_annotations(&mut self) {
        self.hydrophobicity = super::hydrophobicity_scale(&self.residue_name);
        self.estimate_partial_charge();
    }

    /// Create an atom from parsed PDB fields
    #[allow(clippy::too_many_arguments)]
    pub fn from_pdb_fields(
        serial: u32,
        name: String,
        residue_name: String,
        chain_id: char,
        residue_seq: i32,
        insertion_code: Option<char>,
        coord: [f64; 3],
        occupancy: f64,
        b_factor: f64,
        element: String,
        alt_loc: Option<char>,
        model: usize,
        is_hetatm: bool,
    ) -> Self {
        let mut atom = Self::new(
            serial,
            name,
            residue_name,
            chain_id,
            residue_seq,
            coord,
            occupancy,
            b_factor,
            element,
        );

        atom.insertion_code = insertion_code;
        atom.alt_loc = alt_loc;
        atom.model = model;
        atom.is_hetatm = is_hetatm;
        atom.refresh_annotations();
        atom
    }
}

/// Atom type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomType {
    BackboneN,
    BackboneCA,
    BackboneC,
    BackboneO,
    SideChain,
    Hetero,
}

impl Atom {
    /// Get the atom type classification
    pub fn atom_type(&self) -> AtomType {
        if self.is_hetero() {
            AtomType::Hetero
        } else {
            match self.name.as_str() {
                "N" => AtomType::BackboneN,
                "CA" => AtomType::BackboneCA,
                "C" => AtomType::BackboneC,
                "O" => AtomType::BackboneO,
                _ => AtomType::SideChain,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_creation() {
        let atom = Atom::new(
            1,
            "CA".to_string(),
            "ALA".to_string(),
            'A',
            1,
            [0.0, 0.0, 0.0],
            1.0,
            20.0,
            "C".to_string(),
        );

        assert_eq!(atom.serial, 1);
        assert_eq!(atom.name, "CA");
        assert!(atom.is_backbone());
        assert!(atom.is_heavy());
        assert!(!atom.is_hetero());
    }

    #[test]
    fn test_distance_calculation() {
        let atom1 = Atom::new(
            1,
            "CA".to_string(),
            "ALA".to_string(),
            'A',
            1,
            [0.0, 0.0, 0.0],
            1.0,
            20.0,
            "C".to_string(),
        );

        let atom2 = Atom::new(
            2,
            "CA".to_string(),
            "ALA".to_string(),
            'A',
            2,
            [3.0, 4.0, 0.0],
            1.0,
            20.0,
            "C".to_string(),
        );

        assert_eq!(atom1.distance_to(&atom2), 5.0);
        assert_eq!(atom1.distance_squared_to(&atom2), 25.0);
    }

    #[test]
    fn test_radius_and_hbond_flags() {
        let atom = Atom::new(
            1,
            "OG".to_string(),
            "SER".to_string(),
            'A',
            10,
            [1.0, 1.0, 1.0],
            1.0,
            10.0,
            "O".to_string(),
        );

        assert!(atom.vdw_radius() > 1.0);
        assert!(atom.is_hbond_donor());
        assert!(atom.is_hbond_acceptor());
    }
}
