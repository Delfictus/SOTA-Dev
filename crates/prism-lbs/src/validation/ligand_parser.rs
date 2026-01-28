//! Ligand Parser Module
//!
//! Parses ligand coordinates from various file formats:
//! - PDB HETATM records (co-crystallized ligands)
//! - SDF/MOL files (standard drug format)
//! - MOL2 files (Tripos format)
//! - XYZ files (simple coordinate format)
//!
//! Used for extracting reference ligand positions for DCC/DCA validation.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Common solvent and ion residue names to exclude
const EXCLUDE_RESIDUES: &[&str] = &[
    "HOH", "WAT", "H2O", "DOD", "D2O", // Water
    "SO4", "PO4", "NO3", "CO3", // Common ions
    "NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN", "CU", "NI", "CO", // Metal ions
    "GOL", "EDO", "PEG", "DMS", "ACT", "BME", "TRS", "IMD", // Common additives
    "MPD", "PGE", "1PE", "P6G", "7PE", // PEG variants
    "CIT", "FMT", "ACE", "NH4", // Buffers
];

/// Parsed ligand information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ligand {
    /// Ligand name/identifier
    pub name: String,
    /// Chain ID (if from PDB)
    pub chain_id: Option<char>,
    /// Residue number (if from PDB)
    pub residue_number: Option<i32>,
    /// Atom coordinates
    pub atoms: Vec<LigandAtom>,
    /// Centroid of the ligand
    pub centroid: [f64; 3],
    /// Number of heavy atoms
    pub heavy_atom_count: usize,
    /// Molecular weight estimate
    pub molecular_weight: f64,
}

/// Single atom in a ligand
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LigandAtom {
    /// Atom name
    pub name: String,
    /// Element symbol
    pub element: String,
    /// 3D coordinates
    pub coord: [f64; 3],
    /// Atom type (if available)
    pub atom_type: Option<String>,
    /// Partial charge (if available)
    pub charge: Option<f64>,
}

impl Ligand {
    /// Get all coordinates as an array
    pub fn coordinates(&self) -> Vec<[f64; 3]> {
        self.atoms.iter().map(|a| a.coord).collect()
    }

    /// Get heavy atom coordinates only (non-hydrogen)
    pub fn heavy_atom_coordinates(&self) -> Vec<[f64; 3]> {
        self.atoms
            .iter()
            .filter(|a| !a.element.eq_ignore_ascii_case("H"))
            .map(|a| a.coord)
            .collect()
    }

    /// Calculate centroid from atoms
    fn calculate_centroid(atoms: &[LigandAtom]) -> [f64; 3] {
        if atoms.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        let n = atoms.len() as f64;
        let sum = atoms.iter().fold([0.0, 0.0, 0.0], |acc, a| {
            [acc[0] + a.coord[0], acc[1] + a.coord[1], acc[2] + a.coord[2]]
        });
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }

    /// Estimate molecular weight from elements
    fn estimate_molecular_weight(atoms: &[LigandAtom]) -> f64 {
        atoms
            .iter()
            .map(|a| match a.element.to_uppercase().as_str() {
                "C" => 12.01,
                "N" => 14.01,
                "O" => 16.00,
                "S" => 32.07,
                "P" => 30.97,
                "H" => 1.008,
                "F" => 19.00,
                "CL" => 35.45,
                "BR" => 79.90,
                "I" => 126.90,
                _ => 12.0, // Default to carbon
            })
            .sum()
    }
}

/// Ligand parser supporting multiple formats
pub struct LigandParser;

impl LigandParser {
    /// Parse ligands from a PDB file (HETATM records)
    pub fn parse_pdb(path: &Path) -> std::io::Result<Vec<Ligand>> {
        let content = fs::read_to_string(path)?;
        Self::parse_pdb_string(&content)
    }

    /// Parse ligands from PDB content string
    pub fn parse_pdb_string(content: &str) -> std::io::Result<Vec<Ligand>> {
        let exclude_set: HashSet<&str> = EXCLUDE_RESIDUES.iter().copied().collect();

        // Group HETATM records by (resname, chain, resnum)
        let mut ligand_map: std::collections::HashMap<(String, char, i32), Vec<LigandAtom>> =
            std::collections::HashMap::new();

        for line in content.lines() {
            if !line.starts_with("HETATM") {
                continue;
            }

            if line.len() < 54 {
                continue;
            }

            let res_name = line[17..20].trim().to_uppercase();
            if exclude_set.contains(res_name.as_str()) {
                continue;
            }

            let chain_id = line.chars().nth(21).unwrap_or('A');
            let res_num: i32 = line[22..26].trim().parse().unwrap_or(0);

            let x: f64 = line[30..38].trim().parse().unwrap_or(0.0);
            let y: f64 = line[38..46].trim().parse().unwrap_or(0.0);
            let z: f64 = line[46..54].trim().parse().unwrap_or(0.0);

            let atom_name = line[12..16].trim().to_string();
            let element = if line.len() >= 78 {
                line[76..78].trim().to_string()
            } else {
                // Infer from atom name
                atom_name
                    .chars()
                    .take(1)
                    .collect::<String>()
                    .to_uppercase()
            };

            let atom = LigandAtom {
                name: atom_name,
                element,
                coord: [x, y, z],
                atom_type: None,
                charge: None,
            };

            ligand_map
                .entry((res_name, chain_id, res_num))
                .or_default()
                .push(atom);
        }

        // Convert to Ligand structs
        let mut ligands: Vec<Ligand> = ligand_map
            .into_iter()
            .filter(|(_, atoms)| atoms.len() >= 5) // Filter small fragments
            .map(|((name, chain, res_num), atoms)| {
                let centroid = Ligand::calculate_centroid(&atoms);
                let heavy_count = atoms.iter().filter(|a| a.element != "H").count();
                let mw = Ligand::estimate_molecular_weight(&atoms);

                Ligand {
                    name,
                    chain_id: Some(chain),
                    residue_number: Some(res_num),
                    atoms,
                    centroid,
                    heavy_atom_count: heavy_count,
                    molecular_weight: mw,
                }
            })
            .collect();

        // Sort by size (largest first - usually the drug-like molecule)
        ligands.sort_by(|a, b| b.heavy_atom_count.cmp(&a.heavy_atom_count));

        Ok(ligands)
    }

    /// Parse an SDF/MOL file
    pub fn parse_sdf(path: &Path) -> std::io::Result<Vec<Ligand>> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut ligands = Vec::new();
        let mut lines: Vec<String> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            lines.push(line.clone());

            // SDF record separator
            if line.starts_with("$$$$") {
                if let Some(lig) = Self::parse_mol_block(&lines) {
                    ligands.push(lig);
                }
                lines.clear();
            }
        }

        // Handle single MOL file (no $$$$ terminator)
        if !lines.is_empty() {
            if let Some(lig) = Self::parse_mol_block(&lines) {
                ligands.push(lig);
            }
        }

        Ok(ligands)
    }

    /// Parse a MOL block (V2000 format)
    fn parse_mol_block(lines: &[String]) -> Option<Ligand> {
        if lines.len() < 4 {
            return None;
        }

        // First line is molecule name
        let name = lines[0].trim().to_string();

        // Line 4 (index 3) is counts line: num_atoms num_bonds ...
        let counts: Vec<&str> = lines[3].split_whitespace().collect();
        if counts.len() < 2 {
            return None;
        }

        let num_atoms: usize = counts[0].parse().ok()?;

        // Atom block starts at line 5 (index 4)
        let mut atoms = Vec::new();
        for i in 0..num_atoms {
            let atom_line = lines.get(4 + i)?;
            let parts: Vec<&str> = atom_line.split_whitespace().collect();

            if parts.len() < 4 {
                continue;
            }

            let x: f64 = parts[0].parse().ok()?;
            let y: f64 = parts[1].parse().ok()?;
            let z: f64 = parts[2].parse().ok()?;
            let element = parts[3].to_string();

            atoms.push(LigandAtom {
                name: format!("{}{}", element, i + 1),
                element,
                coord: [x, y, z],
                atom_type: None,
                charge: None,
            });
        }

        if atoms.is_empty() {
            return None;
        }

        let centroid = Ligand::calculate_centroid(&atoms);
        let heavy_count = atoms.iter().filter(|a| a.element != "H").count();
        let mw = Ligand::estimate_molecular_weight(&atoms);

        Some(Ligand {
            name: if name.is_empty() {
                "Unknown".to_string()
            } else {
                name
            },
            chain_id: None,
            residue_number: None,
            atoms,
            centroid,
            heavy_atom_count: heavy_count,
            molecular_weight: mw,
        })
    }

    /// Parse a MOL2 file (Tripos format)
    pub fn parse_mol2(path: &Path) -> std::io::Result<Vec<Ligand>> {
        let content = fs::read_to_string(path)?;
        let mut ligands = Vec::new();

        let mut current_name = String::new();
        let mut atoms = Vec::new();
        let mut in_atom_block = false;

        for line in content.lines() {
            if line.starts_with("@<TRIPOS>MOLECULE") {
                // Save previous molecule if exists
                if !atoms.is_empty() {
                    let centroid = Ligand::calculate_centroid(&atoms);
                    let heavy_count = atoms.iter().filter(|a| a.element != "H").count();
                    let mw = Ligand::estimate_molecular_weight(&atoms);

                    ligands.push(Ligand {
                        name: current_name.clone(),
                        chain_id: None,
                        residue_number: None,
                        atoms: std::mem::take(&mut atoms),
                        centroid,
                        heavy_atom_count: heavy_count,
                        molecular_weight: mw,
                    });
                }
                in_atom_block = false;
            } else if line.starts_with("@<TRIPOS>ATOM") {
                in_atom_block = true;
            } else if line.starts_with("@<TRIPOS>") {
                in_atom_block = false;
            } else if in_atom_block {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 6 {
                    let atom_name = parts[1].to_string();
                    let x: f64 = parts[2].parse().unwrap_or(0.0);
                    let y: f64 = parts[3].parse().unwrap_or(0.0);
                    let z: f64 = parts[4].parse().unwrap_or(0.0);
                    let atom_type = parts[5].to_string();

                    // Extract element from atom type (e.g., "C.3" -> "C")
                    let element = atom_type
                        .split('.')
                        .next()
                        .unwrap_or("C")
                        .to_uppercase();

                    let charge = if parts.len() >= 9 {
                        parts[8].parse().ok()
                    } else {
                        None
                    };

                    atoms.push(LigandAtom {
                        name: atom_name,
                        element,
                        coord: [x, y, z],
                        atom_type: Some(atom_type),
                        charge,
                    });
                }
            } else if atoms.is_empty() && !line.starts_with("@") && !line.is_empty() {
                // Molecule name (first non-empty line after @<TRIPOS>MOLECULE)
                current_name = line.trim().to_string();
            }
        }

        // Save last molecule
        if !atoms.is_empty() {
            let centroid = Ligand::calculate_centroid(&atoms);
            let heavy_count = atoms.iter().filter(|a| a.element != "H").count();
            let mw = Ligand::estimate_molecular_weight(&atoms);

            ligands.push(Ligand {
                name: current_name,
                chain_id: None,
                residue_number: None,
                atoms,
                centroid,
                heavy_atom_count: heavy_count,
                molecular_weight: mw,
            });
        }

        Ok(ligands)
    }

    /// Parse an XYZ file (simple coordinate format)
    pub fn parse_xyz(path: &Path) -> std::io::Result<Vec<Ligand>> {
        let content = fs::read_to_string(path)?;
        let mut atoms = Vec::new();
        let mut line_iter = content.lines();

        // First line might be atom count
        let first_line = line_iter.next().unwrap_or("");
        let num_atoms: Option<usize> = first_line.trim().parse().ok();

        // Second line might be comment/name
        let name = if num_atoms.is_some() {
            line_iter.next().unwrap_or("Unknown").trim().to_string()
        } else {
            "Unknown".to_string()
        };

        // If first line wasn't a number, process it as an atom
        if num_atoms.is_none() {
            if let Some(atom) = Self::parse_xyz_line(first_line) {
                atoms.push(atom);
            }
        }

        // Parse remaining lines
        for line in line_iter {
            if let Some(atom) = Self::parse_xyz_line(line) {
                atoms.push(atom);
            }
        }

        if atoms.is_empty() {
            return Ok(Vec::new());
        }

        let centroid = Ligand::calculate_centroid(&atoms);
        let heavy_count = atoms.iter().filter(|a| a.element != "H").count();
        let mw = Ligand::estimate_molecular_weight(&atoms);

        Ok(vec![Ligand {
            name,
            chain_id: None,
            residue_number: None,
            atoms,
            centroid,
            heavy_atom_count: heavy_count,
            molecular_weight: mw,
        }])
    }

    fn parse_xyz_line(line: &str) -> Option<LigandAtom> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return None;
        }

        let element = parts[0].to_uppercase();
        let x: f64 = parts[1].parse().ok()?;
        let y: f64 = parts[2].parse().ok()?;
        let z: f64 = parts[3].parse().ok()?;

        Some(LigandAtom {
            name: element.clone(),
            element,
            coord: [x, y, z],
            atom_type: None,
            charge: None,
        })
    }

    /// Auto-detect format and parse
    pub fn parse(path: &Path) -> std::io::Result<Vec<Ligand>> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "pdb" | "ent" => Self::parse_pdb(path),
            "sdf" | "mol" => Self::parse_sdf(path),
            "mol2" => Self::parse_mol2(path),
            "xyz" => Self::parse_xyz(path),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unsupported ligand format: {}", ext),
            )),
        }
    }

    /// Extract the primary ligand (largest drug-like molecule)
    pub fn extract_primary_ligand(path: &Path) -> std::io::Result<Option<Ligand>> {
        let ligands = Self::parse(path)?;

        // Filter for drug-like molecules (MW > 100, heavy atoms > 10)
        let drug_like: Vec<_> = ligands
            .into_iter()
            .filter(|l| l.molecular_weight > 100.0 && l.heavy_atom_count > 10)
            .collect();

        Ok(drug_like.into_iter().next())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exclude_solvents() {
        let pdb_content = r#"
HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00 10.00           O
HETATM    2  C1  ATP A   2       1.000   0.000   0.000  1.00 10.00           C
HETATM    3  C2  ATP A   2       2.000   0.000   0.000  1.00 10.00           C
HETATM    4  C3  ATP A   2       3.000   0.000   0.000  1.00 10.00           C
HETATM    5  C4  ATP A   2       4.000   0.000   0.000  1.00 10.00           C
HETATM    6  C5  ATP A   2       5.000   0.000   0.000  1.00 10.00           C
"#;

        let ligands = LigandParser::parse_pdb_string(pdb_content).unwrap();

        // Should only have ATP, not HOH
        assert_eq!(ligands.len(), 1);
        assert_eq!(ligands[0].name, "ATP");
    }

    #[test]
    fn test_centroid_calculation() {
        let atoms = vec![
            LigandAtom {
                name: "C1".into(),
                element: "C".into(),
                coord: [0.0, 0.0, 0.0],
                atom_type: None,
                charge: None,
            },
            LigandAtom {
                name: "C2".into(),
                element: "C".into(),
                coord: [2.0, 0.0, 0.0],
                atom_type: None,
                charge: None,
            },
        ];

        let centroid = Ligand::calculate_centroid(&atoms);
        assert!((centroid[0] - 1.0).abs() < 0.001);
    }
}
