//! World-class PDB Parser for VASIL Benchmark
//!
//! Parses PDB files to extract:
//! - Atom coordinates
//! - CA indices
//! - B-factors (temperature factors)
//! - Residue types (amino acid identities)
//!
//! Also computes:
//! - Burial (solvent accessibility proxy)
//! - Conservation (from known data)

use anyhow::{Result, Context, bail};
use std::collections::HashMap;
use std::path::Path;
use std::fs;

/// Amino acid 3-letter to 1-letter mapping
pub fn aa3_to_index(aa3: &str) -> i32 {
    match aa3 {
        "ALA" => 0,
        "CYS" => 1,
        "ASP" => 2,
        "GLU" => 3,
        "PHE" => 4,
        "GLY" => 5,
        "HIS" => 6,
        "ILE" => 7,
        "LYS" => 8,
        "LEU" => 9,
        "MET" => 10,
        "ASN" => 11,
        "PRO" => 12,
        "GLN" => 13,
        "ARG" => 14,
        "SER" => 15,
        "THR" => 16,
        "VAL" => 17,
        "TRP" => 18,
        "TYR" => 19,
        _ => 0,  // Unknown defaults to ALA
    }
}

/// Single-letter AA to index
pub fn aa1_to_index(aa1: char) -> i32 {
    match aa1 {
        'A' => 0,
        'C' => 1,
        'D' => 2,
        'E' => 3,
        'F' => 4,
        'G' => 5,
        'H' => 6,
        'I' => 7,
        'K' => 8,
        'L' => 9,
        'M' => 10,
        'N' => 11,
        'P' => 12,
        'Q' => 13,
        'R' => 14,
        'S' => 15,
        'T' => 16,
        'V' => 17,
        'W' => 18,
        'Y' => 19,
        _ => 0,
    }
}

/// Index to single-letter AA
pub fn index_to_aa1(idx: i32) -> char {
    match idx {
        0 => 'A',
        1 => 'C',
        2 => 'D',
        3 => 'E',
        4 => 'F',
        5 => 'G',
        6 => 'H',
        7 => 'I',
        8 => 'K',
        9 => 'L',
        10 => 'M',
        11 => 'N',
        12 => 'P',
        13 => 'Q',
        14 => 'R',
        15 => 'S',
        16 => 'T',
        17 => 'V',
        18 => 'W',
        19 => 'Y',
        _ => 'X',
    }
}

/// Parsed PDB structure
#[derive(Debug, Clone)]
pub struct PdbStructure {
    /// Atom coordinates [n_atoms * 3] (x, y, z for each atom)
    pub atoms: Vec<f32>,

    /// Indices of CA atoms [n_residues]
    pub ca_indices: Vec<i32>,

    /// B-factors for each residue [n_residues]
    pub bfactors: Vec<f32>,

    /// Residue types (AA index 0-19) [n_residues]
    pub residue_types: Vec<i32>,

    /// Residue numbers from PDB [n_residues]
    pub residue_numbers: Vec<i32>,

    /// Chain IDs [n_residues]
    pub chain_ids: Vec<char>,

    /// Number of residues
    pub n_residues: usize,

    /// Number of atoms
    pub n_atoms: usize,
}

/// Parsed atom from PDB ATOM record
#[derive(Debug, Clone)]
struct PdbAtom {
    serial: i32,
    name: String,
    res_name: String,
    chain_id: char,
    res_seq: i32,
    x: f32,
    y: f32,
    z: f32,
    bfactor: f32,
}

impl PdbStructure {
    /// Parse a PDB file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .context(format!("Failed to read PDB file: {:?}", path))?;

        Self::from_string(&content)
    }

    /// Parse PDB from string content
    pub fn from_string(content: &str) -> Result<Self> {
        let mut atoms_vec: Vec<PdbAtom> = Vec::new();

        // Parse ATOM records
        for line in content.lines() {
            if line.starts_with("ATOM") || line.starts_with("HETATM") {
                if let Some(atom) = Self::parse_atom_line(line) {
                    atoms_vec.push(atom);
                }
            }
        }

        if atoms_vec.is_empty() {
            bail!("No atoms found in PDB");
        }

        // Group atoms by residue
        let mut residue_atoms: HashMap<(char, i32), Vec<&PdbAtom>> = HashMap::new();
        for atom in &atoms_vec {
            residue_atoms
                .entry((atom.chain_id, atom.res_seq))
                .or_insert_with(Vec::new)
                .push(atom);
        }

        // Sort residues by chain and residue number
        let mut residue_keys: Vec<_> = residue_atoms.keys().cloned().collect();
        residue_keys.sort_by(|a, b| {
            a.0.cmp(&b.0).then(a.1.cmp(&b.1))
        });

        // Build output arrays
        let mut atoms: Vec<f32> = Vec::new();
        let mut ca_indices: Vec<i32> = Vec::new();
        let mut bfactors: Vec<f32> = Vec::new();
        let mut residue_types: Vec<i32> = Vec::new();
        let mut residue_numbers: Vec<i32> = Vec::new();
        let mut chain_ids: Vec<char> = Vec::new();

        let mut atom_idx = 0;

        for (chain_id, res_seq) in &residue_keys {
            let res_atoms = &residue_atoms[&(*chain_id, *res_seq)];

            // Find CA atom and get residue info
            let mut ca_idx: Option<i32> = None;
            let mut res_bfactor = 0.0;
            let mut res_type = 0;
            let mut ca_count = 0;

            for atom in res_atoms {
                // Add all atom coordinates
                atoms.push(atom.x);
                atoms.push(atom.y);
                atoms.push(atom.z);

                // Track CA atom
                if atom.name.trim() == "CA" {
                    ca_idx = Some(atom_idx);
                    res_bfactor = atom.bfactor;
                    res_type = aa3_to_index(&atom.res_name);
                    ca_count += 1;
                }

                atom_idx += 1;
            }

            // Only add residues that have a CA atom
            if let Some(ca) = ca_idx {
                ca_indices.push(ca);
                bfactors.push(res_bfactor);
                residue_types.push(res_type);
                residue_numbers.push(*res_seq);
                chain_ids.push(*chain_id);
            }
        }

        let n_residues = ca_indices.len();
        let n_atoms = atoms.len() / 3;

        log::info!("Parsed PDB: {} residues, {} atoms", n_residues, n_atoms);

        Ok(PdbStructure {
            atoms,
            ca_indices,
            bfactors,
            residue_types,
            residue_numbers,
            chain_ids,
            n_residues,
            n_atoms,
        })
    }

    /// Parse a single ATOM line from PDB format
    fn parse_atom_line(line: &str) -> Option<PdbAtom> {
        if line.len() < 54 {
            return None;
        }

        // PDB format is fixed-width columns:
        // ATOM   4879  N   THR E 333     -34.808  16.588  48.236  1.00107.78           N
        // 1-6    Record name
        // 7-11   Serial number
        // 13-16  Atom name
        // 17     Alternate location
        // 18-20  Residue name
        // 22     Chain ID
        // 23-26  Residue sequence number
        // 31-38  X coordinate
        // 39-46  Y coordinate
        // 47-54  Z coordinate
        // 55-60  Occupancy
        // 61-66  B-factor

        let serial: i32 = line.get(6..11)?.trim().parse().ok()?;
        let name = line.get(12..16)?.to_string();
        let res_name = line.get(17..20)?.trim().to_string();
        let chain_id = line.chars().nth(21)?;
        let res_seq: i32 = line.get(22..26)?.trim().parse().ok()?;
        let x: f32 = line.get(30..38)?.trim().parse().ok()?;
        let y: f32 = line.get(38..46)?.trim().parse().ok()?;
        let z: f32 = line.get(46..54)?.trim().parse().ok()?;

        // B-factor (optional, default to 50.0)
        let bfactor: f32 = line.get(60..66)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(50.0);

        Some(PdbAtom {
            serial,
            name,
            res_name,
            chain_id,
            res_seq,
            x,
            y,
            z,
            bfactor,
        })
    }

    /// Extract a specific chain from the structure
    pub fn extract_chain(&self, chain: char) -> Result<Self> {
        let mut atoms: Vec<f32> = Vec::new();
        let mut ca_indices: Vec<i32> = Vec::new();
        let mut bfactors: Vec<f32> = Vec::new();
        let mut residue_types: Vec<i32> = Vec::new();
        let mut residue_numbers: Vec<i32> = Vec::new();
        let mut chain_ids: Vec<char> = Vec::new();

        // This is a simplified extraction - for full implementation
        // we'd need to track which atoms belong to which residues
        for (i, &c) in self.chain_ids.iter().enumerate() {
            if c == chain {
                ca_indices.push(self.ca_indices[i]);
                bfactors.push(self.bfactors[i]);
                residue_types.push(self.residue_types[i]);
                residue_numbers.push(self.residue_numbers[i]);
                chain_ids.push(c);
            }
        }

        // For atoms, we need to extract only atoms belonging to chain
        // This simplified version just uses all atoms (correct for single-chain)
        let atoms = self.atoms.clone();

        let n_residues = ca_indices.len();
        let n_atoms = atoms.len() / 3;

        if n_residues == 0 {
            bail!("Chain {} not found in structure", chain);
        }

        Ok(PdbStructure {
            atoms,
            ca_indices,
            bfactors,
            residue_types,
            residue_numbers,
            chain_ids,
            n_residues,
            n_atoms,
        })
    }

    /// Compute burial (solvent accessibility proxy) for each residue
    /// Uses neighbor counting within 10Å as a proxy for burial
    pub fn compute_burial(&self) -> Vec<f32> {
        let mut burial = vec![0.0f32; self.n_residues];
        let contact_dist = 10.0f32;  // Angstroms
        let max_contacts = 15.0f32;  // Normalization

        for i in 0..self.n_residues {
            let ca_i = self.ca_indices[i] as usize;
            let xi = self.atoms[ca_i * 3];
            let yi = self.atoms[ca_i * 3 + 1];
            let zi = self.atoms[ca_i * 3 + 2];

            let mut contacts = 0.0;

            for j in 0..self.n_residues {
                if i == j {
                    continue;
                }

                let ca_j = self.ca_indices[j] as usize;
                let xj = self.atoms[ca_j * 3];
                let yj = self.atoms[ca_j * 3 + 1];
                let zj = self.atoms[ca_j * 3 + 2];

                let dx = xi - xj;
                let dy = yi - yj;
                let dz = zi - zj;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < contact_dist {
                    contacts += 1.0;
                }
            }

            // Normalize: more contacts = more buried
            burial[i] = (contacts / max_contacts).min(1.0);
        }

        burial
    }

    /// Normalize B-factors to 0-1 range
    pub fn normalize_bfactors(&self) -> Vec<f32> {
        if self.bfactors.is_empty() {
            return Vec::new();
        }

        let min_b = self.bfactors.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_b = self.bfactors.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_b - min_b).max(1.0);

        self.bfactors
            .iter()
            .map(|&b| (b - min_b) / range)
            .collect()
    }

    /// Get residue index for a given residue number
    pub fn residue_index(&self, res_num: i32) -> Option<usize> {
        self.residue_numbers.iter().position(|&r| r == res_num)
    }

    /// Apply a mutation to the structure
    /// Returns a new structure with the mutation applied
    pub fn apply_mutation(&self, res_num: i32, new_aa: char) -> Result<Self> {
        let mut new_structure = self.clone();

        // Find residue index
        let res_idx = self.residue_index(res_num)
            .context(format!("Residue {} not found", res_num))?;

        // Update residue type
        new_structure.residue_types[res_idx] = aa1_to_index(new_aa);

        log::debug!("Applied mutation at position {}: {} -> {}",
                    res_num,
                    index_to_aa1(self.residue_types[res_idx]),
                    new_aa);

        Ok(new_structure)
    }

    /// Apply multiple mutations (e.g., "K417N/L452R/T478K")
    pub fn apply_mutations(&self, mutations_str: &str) -> Result<Self> {
        let mut structure = self.clone();

        for mutation in mutations_str.split('/') {
            let mutation = mutation.trim();
            if mutation.is_empty() {
                continue;
            }

            // Parse mutation format: K417N (original AA, position, new AA)
            if mutation.len() < 3 {
                log::warn!("Skipping invalid mutation: {}", mutation);
                continue;
            }

            // Extract position (digits in middle)
            let chars: Vec<char> = mutation.chars().collect();
            let orig_aa = chars[0];
            let new_aa = chars[chars.len() - 1];

            // Extract position number
            let pos_str: String = chars[1..chars.len()-1].iter().collect();
            let res_num: i32 = match pos_str.parse() {
                Ok(n) => n,
                Err(_) => {
                    log::warn!("Skipping mutation with invalid position: {}", mutation);
                    continue;
                }
            };

            // Apply mutation (we ignore original AA check for flexibility)
            match structure.apply_mutation(res_num, new_aa) {
                Ok(s) => structure = s,
                Err(e) => {
                    log::warn!("Failed to apply mutation {}: {}", mutation, e);
                }
            }
        }

        Ok(structure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aa_conversion() {
        assert_eq!(aa3_to_index("ALA"), 0);
        assert_eq!(aa3_to_index("LYS"), 8);
        assert_eq!(aa1_to_index('A'), 0);
        assert_eq!(aa1_to_index('K'), 8);
        assert_eq!(index_to_aa1(0), 'A');
        assert_eq!(index_to_aa1(8), 'K');
    }

    #[test]
    fn test_parse_atom_line() {
        let line = "ATOM   4879  N   THR E 333     -34.808  16.588  48.236  1.00107.78           N  ";
        let atom = PdbStructure::parse_atom_line(line);
        assert!(atom.is_some());
        let atom = atom.unwrap();
        assert_eq!(atom.serial, 4879);
        assert_eq!(atom.res_name, "THR");
        assert_eq!(atom.chain_id, 'E');
        assert_eq!(atom.res_seq, 333);
    }
}
