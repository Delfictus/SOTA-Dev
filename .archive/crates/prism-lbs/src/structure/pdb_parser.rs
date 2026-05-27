//! PDB parsing and protein structure representation

use crate::LbsError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::{Atom, Residue};

/// Options controlling how PDB files are parsed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdbParseOptions {
    /// Include HETATM records (ligands, waters, ions)
    pub include_hetatm: bool,
    /// Parse all models instead of just the first
    pub include_all_models: bool,
    /// Keep alternate location records (B, C, etc.). If false, only blank/A altLoc are used
    pub keep_alternate_locations: bool,
}

impl Default for PdbParseOptions {
    fn default() -> Self {
        Self {
            include_hetatm: true,
            include_all_models: false,
            keep_alternate_locations: false,
        }
    }
}

/// In-memory representation of a protein structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinStructure {
    /// Concatenated TITLE records
    pub title: String,
    /// PDB identifier if present
    pub pdb_id: Option<String>,
    /// Experimental resolution (Å)
    pub resolution: Option<f64>,
    /// Experimental method (e.g., X-RAY DIFFRACTION)
    pub experimental_method: Option<String>,
    /// All atoms (ordered as parsed)
    pub atoms: Vec<Atom>,
    /// All residues
    pub residues: Vec<Residue>,
    /// Residue indices grouped by chain identifier
    pub chain_residue_indices: HashMap<char, Vec<usize>>,
    /// Indices for atoms coming from HETATM lines
    pub hetero_atom_indices: Vec<usize>,
    /// Number of models present (parsed or skipped)
    pub models: usize,
    /// Whether the source file contained alternate locations
    pub has_alternate_locations: bool,
    /// Geometric center of all atoms
    pub center_of_mass: [f64; 3],
    /// Axis-aligned bounding box (min, max)
    pub bounding_box: ([f64; 3], [f64; 3]),
    /// Path to source PDB file (if loaded from file, needed for fpocket integration)
    #[serde(skip)]
    pub pdb_path: Option<std::path::PathBuf>,
}

impl Default for ProteinStructure {
    fn default() -> Self {
        Self {
            title: String::new(),
            pdb_id: None,
            resolution: None,
            experimental_method: None,
            atoms: Vec::new(),
            residues: Vec::new(),
            chain_residue_indices: HashMap::new(),
            hetero_atom_indices: Vec::new(),
            models: 1,
            has_alternate_locations: false,
            center_of_mass: [0.0; 3],
            bounding_box: ([0.0; 3], [0.0; 3]),
            pdb_path: None,
        }
    }
}

impl ProteinStructure {
    /// Parse a PDB file from disk
    pub fn from_pdb_file(path: &Path) -> Result<Self, LbsError> {
        let pdb_data = fs::read_to_string(path)?;
        let mut structure = Self::from_pdb_str_with_options(&pdb_data, PdbParseOptions::default())?;
        structure.pdb_path = Some(path.to_path_buf());
        Ok(structure)
    }

    /// Parse a PDB file with custom options
    pub fn from_pdb_file_with_options(
        path: &Path,
        options: PdbParseOptions,
    ) -> Result<Self, LbsError> {
        let pdb_data = fs::read_to_string(path)?;
        let mut structure = Self::from_pdb_str_with_options(&pdb_data, options)?;
        structure.pdb_path = Some(path.to_path_buf());
        Ok(structure)
    }

    /// Parse a PDB structure from an in-memory string
    pub fn from_pdb_str(contents: &str) -> Result<Self, LbsError> {
        Self::from_pdb_str_with_options(contents, PdbParseOptions::default())
    }

    /// Parse a PDB structure from an in-memory string with custom options
    pub fn from_pdb_str_with_options(
        contents: &str,
        options: PdbParseOptions,
    ) -> Result<Self, LbsError> {
        let mut structure = ProteinStructure::default();
        let mut residue_lookup: HashMap<(usize, char, i32, Option<char>), usize> = HashMap::new();
        let mut current_model: usize = 1;

        for line in contents.lines() {
            if line.starts_with("HEADER") {
                if let Some(pdb_id) = line.get(62..66).and_then(extract_token) {
                    structure.pdb_id = Some(pdb_id.to_string());
                }
            } else if line.starts_with("TITLE") {
                if let Some(fragment) = line.get(10..).map(str::trim) {
                    if !fragment.is_empty() {
                        if !structure.title.is_empty() {
                            structure.title.push(' ');
                        }
                        structure.title.push_str(fragment);
                    }
                }
            } else if line.starts_with("EXPDTA") {
                if let Some(method) = line.get(10..).map(str::trim) {
                    if !method.is_empty() {
                        structure.experimental_method = Some(method.to_string());
                    }
                }
            } else if line.starts_with("REMARK   2") && structure.resolution.is_none() {
                structure.resolution = parse_resolution(line);
            } else if line.starts_with("MODEL") {
                current_model = parse_usize_field(line, 10..14).unwrap_or(current_model + 1);
                structure.models = structure.models.max(current_model);
                if !options.include_all_models && current_model > 1 {
                    continue;
                }
            } else if line.starts_with("ENDMDL") && !options.include_all_models {
                break;
            }

            let record = line.get(0..6).unwrap_or("").trim();
            if record != "ATOM" && record != "HETATM" {
                continue;
            }

            let is_hetatm = record == "HETATM";
            if is_hetatm && !options.include_hetatm {
                continue;
            }

            let alt_loc = line.get(16..17).and_then(extract_char);
            if alt_loc.is_some() {
                structure.has_alternate_locations = true;
            }
            if alt_loc.is_some()
                && !options.keep_alternate_locations
                && !matches!(alt_loc, Some('A') | Some(' '))
            {
                continue;
            }

            if !options.include_all_models && current_model > 1 {
                continue;
            }

            let atom = parse_atom_line(line, current_model, is_hetatm, alt_loc)
                .map_err(LbsError::PdbParse)?;

            let residue_key = (
                atom.model,
                atom.chain_id,
                atom.residue_seq,
                atom.insertion_code,
            );
            let residue_index = *residue_lookup.entry(residue_key).or_insert_with(|| {
                let mut residue = Residue::new(
                    atom.residue_name.clone(),
                    atom.chain_id,
                    atom.residue_seq,
                    atom.insertion_code,
                );
                residue.model = atom.model;
                residue.is_hetatm = is_hetatm;
                structure.residues.push(residue);
                let idx = structure.residues.len() - 1;
                structure
                    .chain_residue_indices
                    .entry(atom.chain_id)
                    .or_default()
                    .push(idx);
                idx
            });

            let atom_index = structure.atoms.len();
            structure.atoms.push(atom);
            if let Some(residue) = structure.residues.get_mut(residue_index) {
                residue.atom_indices.push(atom_index);
            }

            if is_hetatm {
                structure.hetero_atom_indices.push(atom_index);
                if let Some(residue) = structure.residues.get_mut(residue_index) {
                    residue.is_hetatm = true;
                }
            }
        }

        structure.recompute_geometry();
        structure.refresh_residue_properties();
        Ok(structure)
    }

    /// Number of atoms in the structure
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Number of residues in the structure
    pub fn residue_count(&self) -> usize {
        self.residues.len()
    }

    /// Unique chain identifiers present in the structure (sorted)
    pub fn chain_ids(&self) -> Vec<char> {
        let mut ids: Vec<char> = self.chain_residue_indices.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Compute SASA and surface flags using the default surface computer
    pub fn compute_surface_accessibility(&mut self) -> Result<(), LbsError> {
        let computer = super::surface::SurfaceComputer::default();
        computer.compute(self)
    }

    /// Recompute centroids and SASA for all residues from atom data
    pub fn refresh_residue_properties(&mut self) {
        for residue in &mut self.residues {
            residue.refresh_derived_properties(&self.atoms);
        }
    }

    /// Update center of mass and bounding box from atom coordinates
    pub fn recompute_geometry(&mut self) {
        if self.atoms.is_empty() {
            self.center_of_mass = [0.0, 0.0, 0.0];
            self.bounding_box = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
            return;
        }

        let mut min_corner = [f64::INFINITY; 3];
        let mut max_corner = [f64::NEG_INFINITY; 3];
        let mut sum = [0.0, 0.0, 0.0];

        for atom in &self.atoms {
            for i in 0..3 {
                min_corner[i] = min_corner[i].min(atom.coord[i]);
                max_corner[i] = max_corner[i].max(atom.coord[i]);
                sum[i] += atom.coord[i];
            }
        }

        let count = self.atoms.len() as f64;
        self.center_of_mass = [sum[0] / count, sum[1] / count, sum[2] / count];
        self.bounding_box = (min_corner, max_corner);
    }
}

fn parse_atom_line(
    line: &str,
    model: usize,
    is_hetatm: bool,
    alt_loc: Option<char>,
) -> Result<Atom, String> {
    let serial = parse_u32_field(line, 6..11)
        .ok_or_else(|| format!("Invalid atom serial in line: {line}"))?;
    let name = line
        .get(12..16)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("Missing atom name in line: {line}"))?
        .to_string();

    let residue_name = line
        .get(17..20)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("Missing residue name in line: {line}"))?
        .to_ascii_uppercase();

    let chain_id = line.get(21..22).and_then(extract_char).unwrap_or('A');

    let residue_seq = parse_i32_field(line, 22..26).unwrap_or(0);
    let insertion_code = line.get(26..27).and_then(extract_char);

    let x = parse_f64_field(line, 30..38)
        .ok_or_else(|| format!("Missing X coordinate in line: {line}"))?;
    let y = parse_f64_field(line, 38..46)
        .ok_or_else(|| format!("Missing Y coordinate in line: {line}"))?;
    let z = parse_f64_field(line, 46..54)
        .ok_or_else(|| format!("Missing Z coordinate in line: {line}"))?;

    let occupancy = parse_f64_field(line, 54..60).unwrap_or(1.0);
    let b_factor = parse_f64_field(line, 60..66).unwrap_or(0.0);
    let element_field = line.get(76..78).map(str::trim).unwrap_or("");
    let element = resolve_element(element_field, &name);

    Ok(Atom::from_pdb_fields(
        serial,
        name,
        residue_name,
        chain_id,
        residue_seq,
        insertion_code,
        [x, y, z],
        occupancy,
        b_factor,
        element,
        alt_loc,
        model,
        is_hetatm,
    ))
}

fn parse_resolution(line: &str) -> Option<f64> {
    let upper = line.to_ascii_uppercase();
    let needle = "RESOLUTION";
    let pos = upper.find(needle)?;
    let tail = &line[pos + needle.len()..];
    tail.split_whitespace()
        .filter_map(|token| token.parse::<f64>().ok())
        .next()
}

fn parse_f64_field(line: &str, range: std::ops::Range<usize>) -> Option<f64> {
    line.get(range)?.trim().parse::<f64>().ok()
}

fn parse_i32_field(line: &str, range: std::ops::Range<usize>) -> Option<i32> {
    line.get(range)?.trim().parse::<i32>().ok()
}

fn parse_u32_field(line: &str, range: std::ops::Range<usize>) -> Option<u32> {
    line.get(range)?.trim().parse::<u32>().ok()
}

fn parse_usize_field(line: &str, range: std::ops::Range<usize>) -> Option<usize> {
    line.get(range)?.trim().parse::<usize>().ok()
}

fn extract_char(slice: &str) -> Option<char> {
    let trimmed = slice.trim();
    if trimmed.is_empty() {
        None
    } else {
        trimmed.chars().next()
    }
}

fn extract_token(slice: &str) -> Option<&str> {
    let trimmed = slice.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn resolve_element(element_field: &str, atom_name: &str) -> String {
    if !element_field.is_empty() {
        return element_field.to_ascii_uppercase();
    }

    // Derive element from atom name (strip digits and capitalization)
    let letters: String = atom_name
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .collect();

    letters
        .chars()
        .take(2)
        .collect::<String>()
        .to_ascii_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pdb_line(
        record: &str,
        serial: u32,
        name: &str,
        alt_loc: Option<char>,
        res_name: &str,
        chain_id: char,
        res_seq: i32,
        coords: (f64, f64, f64),
        occupancy: f64,
        b_factor: f64,
        element: &str,
    ) -> String {
        format!(
            "{:<6}{:>5} {:<4}{:1}{:<3} {:1}{:>4}{:1}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}           {:>2}",
            record,
            serial,
            name,
            alt_loc.unwrap_or(' '),
            res_name,
            chain_id,
            res_seq,
            "",
            coords.0,
            coords.1,
            coords.2,
            occupancy,
            b_factor,
            element
        )
    }

    fn test_pdb_string() -> String {
        let mut lines = Vec::new();
        lines.push("HEADER    TEST STRUCTURE                                       01-JAN-00   0000              ".to_string());
        lines.push("TITLE     SIMPLE TEST STRUCTURE".to_string());
        lines.push("REMARK   2 RESOLUTION.    1.80 ANGSTROMS.".to_string());
        lines.push(pdb_line(
            "ATOM",
            1,
            "N",
            None,
            "ALA",
            'A',
            1,
            (11.104, 13.207, 9.247),
            1.00,
            20.00,
            "N",
        ));
        lines.push(pdb_line(
            "ATOM",
            2,
            "CA",
            None,
            "ALA",
            'A',
            1,
            (12.560, 13.250, 9.406),
            1.00,
            19.00,
            "C",
        ));
        lines.push(pdb_line(
            "ATOM",
            3,
            "C",
            None,
            "ALA",
            'A',
            1,
            (13.047, 14.699, 9.745),
            1.00,
            18.00,
            "C",
        ));
        lines.push(pdb_line(
            "ATOM",
            4,
            "CB",
            Some('A'),
            "ALA",
            'A',
            1,
            (12.800, 12.000, 8.600),
            0.50,
            15.00,
            "C",
        ));
        lines.push(pdb_line(
            "ATOM",
            5,
            "CB",
            Some('B'),
            "ALA",
            'A',
            1,
            (12.900, 12.100, 8.700),
            0.50,
            15.00,
            "C",
        ));
        lines.push(pdb_line(
            "HETATM",
            6,
            "O",
            None,
            "HOH",
            'A',
            201,
            (10.000, 10.000, 10.000),
            1.00,
            10.00,
            "O",
        ));
        lines.push("END".to_string());
        lines.join("\n")
    }

    #[test]
    fn parses_basic_metadata_and_counts() {
        let pdb = test_pdb_string();
        let structure = ProteinStructure::from_pdb_str(&pdb).expect("parse failed");
        assert!(structure.pdb_id.is_some());
        assert_eq!(structure.title.trim(), "SIMPLE TEST STRUCTURE");
        assert_eq!(structure.resolution, Some(1.80));
        assert_eq!(structure.atom_count(), 5); // altLoc B is skipped by default
        assert_eq!(structure.residue_count(), 2); // ALA + HOH
        assert_eq!(structure.chain_ids(), vec!['A']);
        assert_eq!(structure.models, 1);
        assert!(structure.has_alternate_locations);
        assert_eq!(structure.hetero_atom_indices.len(), 1);
    }

    #[test]
    fn can_exclude_hetatm_records() {
        let pdb = test_pdb_string();
        let options = PdbParseOptions {
            include_hetatm: false,
            ..Default::default()
        };
        let structure =
            ProteinStructure::from_pdb_str_with_options(&pdb, options).expect("parse failed");
        assert_eq!(structure.atom_count(), 4);
        assert_eq!(structure.hetero_atom_indices.len(), 0);
    }

    #[test]
    fn keeps_all_alt_locations_when_requested() {
        let pdb = test_pdb_string();
        let options = PdbParseOptions {
            keep_alternate_locations: true,
            ..Default::default()
        };
        let structure =
            ProteinStructure::from_pdb_str_with_options(&pdb, options).expect("parse failed");
        assert_eq!(structure.atom_count(), 6);
    }
}
