//! PDB Sanitization for GPU Safety
//!
//! This module provides PDB structure sanitization to ensure GPU kernels
//! receive clean, consistent input that won't cause crashes or undefined behavior.
//!
//! # GPU Safety Requirements
//!
//! Raw PDB files from the RCSB can crash GPU kernels due to:
//! - HETATM records (non-standard atoms)
//! - Water molecules (thousands of extra atoms)
//! - Non-standard residues (modified amino acids)
//! - Alternate conformations (duplicate atoms)
//! - Missing atoms (incomplete residues)
//! - Insertion codes (non-sequential numbering)
//!
//! This sanitizer produces a clean structure with:
//! - Only standard amino acid ATOM records
//! - Sequential atom numbering (1, 2, 3, ...)
//! - Sequential residue numbering (1, 2, 3, ...)
//! - Single conformation (A or first)
//! - Extracted Cα coordinates for coarse-grained analysis
//!
//! # Zero Fallback Policy
//!
//! This module does NOT fall back to approximate behavior.
//! If a structure cannot be sanitized, it returns an error.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Standard amino acid 3-letter codes
pub const STANDARD_AMINO_ACIDS: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
];

/// Common water molecule residue names to filter
pub const WATER_RESIDUES: &[&str] = &["HOH", "WAT", "H2O", "DOD", "TIP", "TIP3", "SOL"];

/// Sanitized atom representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizedAtom {
    /// Sequential atom index (1-based for PDB compatibility)
    pub index: usize,
    /// Atom name (e.g., "CA", "N", "C", "O")
    pub name: String,
    /// Residue name (3-letter code)
    pub residue_name: String,
    /// Sequential residue index (1-based)
    pub residue_index: usize,
    /// Chain identifier
    pub chain_id: char,
    /// 3D coordinates [x, y, z] in Angstroms
    pub position: [f32; 3],
    /// Original PDB residue sequence number (for mapping back)
    pub original_res_seq: i32,
}

/// Sanitized Cα-only representation for coarse-grained analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalphaResidue {
    /// Sequential residue index (1-based)
    pub index: usize,
    /// Residue name (3-letter code)
    pub residue_name: String,
    /// Chain identifier
    pub chain_id: char,
    /// Cα coordinates [x, y, z] in Angstroms
    pub position: [f32; 3],
    /// Original PDB residue sequence number
    pub original_res_seq: i32,
}

/// Sanitized PDB structure ready for GPU processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizedStructure {
    /// Source identifier (PDB ID or filename)
    pub source_id: String,
    /// All sanitized atoms
    pub atoms: Vec<SanitizedAtom>,
    /// Cα-only residues for coarse-grained analysis
    pub ca_residues: Vec<CalphaResidue>,
    /// Chain IDs present in the structure
    pub chains: Vec<char>,
    /// Number of residues per chain
    pub residues_per_chain: Vec<usize>,
    /// Sanitization statistics
    pub stats: SanitizationStats,
}

/// Statistics from sanitization process
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SanitizationStats {
    /// Total lines processed
    pub lines_processed: usize,
    /// ATOM records processed
    pub atom_records: usize,
    /// HETATM records removed
    pub hetatm_removed: usize,
    /// Water molecules removed
    pub waters_removed: usize,
    /// Non-standard residues removed
    pub nonstandard_removed: usize,
    /// Alternate conformations removed
    pub altloc_removed: usize,
    /// Final atom count
    pub final_atom_count: usize,
    /// Final residue count
    pub final_residue_count: usize,
    /// Final Cα count
    pub final_ca_count: usize,
}

impl SanitizedStructure {
    /// Get Cα coordinates as flat array for GPU
    ///
    /// Returns coordinates in the format expected by GPU kernels:
    /// [x1, y1, z1, x2, y2, z2, ...]
    pub fn get_ca_coords_flat(&self) -> Vec<f32> {
        let mut coords = Vec::with_capacity(self.ca_residues.len() * 3);
        for res in &self.ca_residues {
            coords.push(res.position[0]);
            coords.push(res.position[1]);
            coords.push(res.position[2]);
        }
        coords
    }

    /// Get Cα coordinates as array of [x, y, z]
    pub fn get_ca_coords(&self) -> Vec<[f32; 3]> {
        self.ca_residues.iter().map(|r| r.position).collect()
    }

    /// Get number of atoms
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Get number of residues
    pub fn n_residues(&self) -> usize {
        self.ca_residues.len()
    }

    /// Check if structure is within NOVA atom limit (512)
    pub fn fits_nova_limit(&self) -> bool {
        self.atoms.len() <= 512
    }

    /// Get all atom coordinates as flat array for GPU
    pub fn get_all_coords_flat(&self) -> Vec<f32> {
        let mut coords = Vec::with_capacity(self.atoms.len() * 3);
        for atom in &self.atoms {
            coords.push(atom.position[0]);
            coords.push(atom.position[1]);
            coords.push(atom.position[2]);
        }
        coords
    }

    /// Regenerate PDB content from sanitized structure
    pub fn to_pdb_string(&self) -> String {
        let mut pdb = String::with_capacity(self.atoms.len() * 80);

        for atom in &self.atoms {
            // PDB ATOM format (columns are 1-indexed in spec, 0-indexed here)
            // ATOM  serial name altLoc resName chainID resSeq iCode x y z occupancy tempFactor element charge
            pdb.push_str(&format!(
                "ATOM  {:5} {:^4} {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00           {:>2}\n",
                atom.index,
                atom.name,
                atom.residue_name,
                atom.chain_id,
                atom.residue_index,
                atom.position[0],
                atom.position[1],
                atom.position[2],
                atom.name.chars().next().unwrap_or('X'),
            ));
        }

        pdb.push_str("END\n");
        pdb
    }
}

/// PDB Sanitizer for GPU-safe structure preparation
#[derive(Debug, Clone)]
pub struct PdbSanitizer {
    /// Whether to keep HETATM records (default: false)
    keep_hetatm: bool,
    /// Whether to keep waters (default: false)
    keep_waters: bool,
    /// Additional residue names to include (besides standard AAs)
    additional_residues: HashSet<String>,
    /// Chain filter (None = all chains)
    chain_filter: Option<char>,
}

impl Default for PdbSanitizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PdbSanitizer {
    /// Create a new sanitizer with default settings
    ///
    /// Default behavior:
    /// - Remove all HETATM records
    /// - Remove all water molecules
    /// - Keep only standard amino acids
    /// - Process all chains
    pub fn new() -> Self {
        Self {
            keep_hetatm: false,
            keep_waters: false,
            additional_residues: HashSet::new(),
            chain_filter: None,
        }
    }

    /// Set whether to keep HETATM records
    pub fn with_hetatm(mut self, keep: bool) -> Self {
        self.keep_hetatm = keep;
        self
    }

    /// Set whether to keep water molecules
    pub fn with_waters(mut self, keep: bool) -> Self {
        self.keep_waters = keep;
        self
    }

    /// Add additional residue names to keep (e.g., modified amino acids)
    pub fn with_additional_residues(mut self, residues: &[&str]) -> Self {
        for r in residues {
            self.additional_residues.insert(r.to_uppercase());
        }
        self
    }

    /// Set chain filter (only process specified chain)
    pub fn with_chain(mut self, chain: char) -> Self {
        self.chain_filter = Some(chain);
        self
    }

    /// Sanitize PDB content and return a GPU-safe structure
    ///
    /// # Arguments
    /// * `content` - Raw PDB file content
    /// * `source_id` - Identifier for this structure (e.g., "1AKE")
    ///
    /// # Errors
    /// Returns error if:
    /// - No valid atoms found after filtering
    /// - No Cα atoms found
    /// - Coordinate parsing fails
    pub fn sanitize(&self, content: &str, source_id: &str) -> Result<SanitizedStructure> {
        let mut stats = SanitizationStats::default();
        let mut atoms = Vec::new();
        let mut ca_residues = Vec::new();

        let mut current_res_key = String::new();
        let mut residue_counter = 0usize;
        let mut atom_counter = 0usize;
        let mut chains_seen = HashSet::new();
        let mut residues_per_chain: std::collections::HashMap<char, usize> = std::collections::HashMap::new();

        // Build set of valid residue names
        let valid_residues: HashSet<&str> = STANDARD_AMINO_ACIDS.iter().cloned().collect();

        for line in content.lines() {
            stats.lines_processed += 1;

            // Determine record type
            let is_atom = line.starts_with("ATOM");
            let is_hetatm = line.starts_with("HETATM");

            if !is_atom && !is_hetatm {
                continue;
            }

            // Count ATOM records
            if is_atom {
                stats.atom_records += 1;
            }

            // Ensure line is long enough for parsing
            if line.len() < 54 {
                continue;
            }

            // Parse residue name first to check for waters (before HETATM filter)
            let residue_name = line.get(17..20).unwrap_or("").trim().to_uppercase();

            // Filter waters (check before HETATM filter so we count them correctly)
            if WATER_RESIDUES.contains(&residue_name.as_str()) {
                if !self.keep_waters {
                    stats.waters_removed += 1;
                    continue;
                }
            }

            // Filter HETATM if not keeping (after water check)
            if is_hetatm && !self.keep_hetatm {
                stats.hetatm_removed += 1;
                continue;
            }

            // Parse remaining PDB fields (fixed-width format)
            let atom_name = line.get(12..16).unwrap_or("").trim();
            let alt_loc = line.get(16..17).unwrap_or(" ").chars().next().unwrap_or(' ');
            let chain_id = line.get(21..22).unwrap_or(" ").chars().next().unwrap_or('A');
            let res_seq_str = line.get(22..26).unwrap_or("0").trim();

            // Apply chain filter
            if let Some(target) = self.chain_filter {
                if chain_id != target {
                    continue;
                }
            }

            // Filter non-standard residues
            if !valid_residues.contains(residue_name.as_str())
                && !self.additional_residues.contains(&residue_name)
            {
                stats.nonstandard_removed += 1;
                continue;
            }

            // Handle alternate conformations (keep only A or first)
            if alt_loc != ' ' && alt_loc != 'A' {
                stats.altloc_removed += 1;
                continue;
            }

            // Parse residue sequence number
            let res_seq: i32 = res_seq_str.parse().unwrap_or(0);

            // Parse coordinates
            let x: f32 = line
                .get(30..38)
                .unwrap_or("0")
                .trim()
                .parse()
                .context("Failed to parse X coordinate")?;
            let y: f32 = line
                .get(38..46)
                .unwrap_or("0")
                .trim()
                .parse()
                .context("Failed to parse Y coordinate")?;
            let z: f32 = line
                .get(46..54)
                .unwrap_or("0")
                .trim()
                .parse()
                .context("Failed to parse Z coordinate")?;

            // Track residue changes (chain + seq number)
            let res_key = format!("{}:{}", chain_id, res_seq);
            if res_key != current_res_key {
                residue_counter += 1;
                current_res_key = res_key;
                chains_seen.insert(chain_id);
                *residues_per_chain.entry(chain_id).or_insert(0) += 1;
            }

            // Increment atom counter
            atom_counter += 1;

            // Create sanitized atom
            let sanitized_atom = SanitizedAtom {
                index: atom_counter,
                name: atom_name.to_string(),
                residue_name: residue_name.clone(),
                residue_index: residue_counter,
                chain_id,
                position: [x, y, z],
                original_res_seq: res_seq,
            };

            // If this is a Cα atom, also add to CA residues
            if atom_name == "CA" {
                ca_residues.push(CalphaResidue {
                    index: residue_counter,
                    residue_name: residue_name.clone(),
                    chain_id,
                    position: [x, y, z],
                    original_res_seq: res_seq,
                });
            }

            atoms.push(sanitized_atom);
        }

        // Validate results
        if atoms.is_empty() {
            bail!(
                "No valid atoms found after sanitization of '{}'. \
                 Check that the PDB contains standard amino acid ATOM records.",
                source_id
            );
        }

        if ca_residues.is_empty() {
            bail!(
                "No Cα atoms found in '{}'. \
                 Structure may be missing backbone atoms or use non-standard naming.",
                source_id
            );
        }

        // Finalize statistics
        stats.final_atom_count = atoms.len();
        stats.final_residue_count = residue_counter;
        stats.final_ca_count = ca_residues.len();

        // Sort chains
        let mut chains: Vec<char> = chains_seen.into_iter().collect();
        chains.sort();

        let residues_per_chain_vec: Vec<usize> = chains
            .iter()
            .map(|c| *residues_per_chain.get(c).unwrap_or(&0))
            .collect();

        log::debug!(
            "Sanitized '{}': {} atoms, {} residues, {} Cα atoms, {} chains",
            source_id,
            stats.final_atom_count,
            stats.final_residue_count,
            stats.final_ca_count,
            chains.len()
        );

        Ok(SanitizedStructure {
            source_id: source_id.to_string(),
            atoms,
            ca_residues,
            chains,
            residues_per_chain: residues_per_chain_vec,
            stats,
        })
    }
}

/// Convenience function to sanitize PDB content with default settings
pub fn sanitize_pdb(content: &str, source_id: &str) -> Result<SanitizedStructure> {
    PdbSanitizer::new().sanitize(content, source_id)
}

/// Convenience function to sanitize PDB content for a specific chain
pub fn sanitize_pdb_chain(content: &str, source_id: &str, chain: char) -> Result<SanitizedStructure> {
    PdbSanitizer::new()
        .with_chain(chain)
        .sanitize(content, source_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PDB: &str = r#"HEADER    TEST STRUCTURE
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C
ATOM      6  N   GLY A   2       3.326   1.545   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       3.941   2.869   0.000  1.00  0.00           C
ATOM      8  C   GLY A   2       5.445   2.810   0.000  1.00  0.00           C
ATOM      9  O   GLY A   2       6.042   1.735   0.000  1.00  0.00           O
HETATM   10  O   HOH A 100      10.000  10.000  10.000  1.00  0.00           O
HETATM   11  MG  MG  A 101      15.000  15.000  15.000  1.00  0.00          MG
END
"#;

    #[test]
    fn test_basic_sanitization() {
        let result = sanitize_pdb(SAMPLE_PDB, "TEST");
        assert!(result.is_ok(), "Sanitization failed: {:?}", result.err());

        let structure = result.unwrap();
        assert_eq!(structure.source_id, "TEST");
        assert_eq!(structure.atoms.len(), 9); // 5 ALA + 4 GLY
        assert_eq!(structure.ca_residues.len(), 2); // 2 residues
        assert_eq!(structure.chains, vec!['A']);
    }

    #[test]
    fn test_hetatm_removal() {
        let result = sanitize_pdb(SAMPLE_PDB, "TEST").unwrap();

        // Should not contain HETATM atoms
        assert!(
            !result.atoms.iter().any(|a| a.residue_name == "HOH" || a.residue_name == "MG"),
            "HETATM records should be removed"
        );

        // Water is counted separately, so only MG is counted as hetatm_removed
        assert_eq!(result.stats.hetatm_removed, 1); // MG only
        assert_eq!(result.stats.waters_removed, 1); // HOH
    }

    #[test]
    fn test_water_removal() {
        let result = sanitize_pdb(SAMPLE_PDB, "TEST").unwrap();
        assert_eq!(result.stats.waters_removed, 1);
    }

    #[test]
    fn test_ca_extraction() {
        let result = sanitize_pdb(SAMPLE_PDB, "TEST").unwrap();

        assert_eq!(result.ca_residues.len(), 2);
        assert_eq!(result.ca_residues[0].residue_name, "ALA");
        assert_eq!(result.ca_residues[1].residue_name, "GLY");

        // Check coordinates
        assert!((result.ca_residues[0].position[0] - 1.458).abs() < 0.001);
    }

    #[test]
    fn test_sequential_numbering() {
        let result = sanitize_pdb(SAMPLE_PDB, "TEST").unwrap();

        // Atoms should be numbered 1, 2, 3, ...
        for (i, atom) in result.atoms.iter().enumerate() {
            assert_eq!(atom.index, i + 1);
        }

        // Residues should be numbered 1, 2
        assert_eq!(result.ca_residues[0].index, 1);
        assert_eq!(result.ca_residues[1].index, 2);
    }

    #[test]
    fn test_get_ca_coords_flat() {
        let result = sanitize_pdb(SAMPLE_PDB, "TEST").unwrap();
        let coords = result.get_ca_coords_flat();

        assert_eq!(coords.len(), 6); // 2 residues * 3 coords
        assert!((coords[0] - 1.458).abs() < 0.001); // First CA x
    }

    #[test]
    fn test_to_pdb_string() {
        let result = sanitize_pdb(SAMPLE_PDB, "TEST").unwrap();
        let pdb_str = result.to_pdb_string();

        assert!(pdb_str.contains("ATOM"));
        assert!(pdb_str.contains("ALA"));
        assert!(pdb_str.contains("GLY"));
        assert!(pdb_str.contains("END"));
        assert!(!pdb_str.contains("HOH")); // No waters
    }

    #[test]
    fn test_chain_filter() {
        let multi_chain_pdb = r#"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA B   1       5.000   5.000   5.000  1.00  0.00           C
END
"#;

        let result = PdbSanitizer::new()
            .with_chain('A')
            .sanitize(multi_chain_pdb, "TEST")
            .unwrap();

        assert_eq!(result.ca_residues.len(), 1);
        assert_eq!(result.ca_residues[0].chain_id, 'A');
    }

    #[test]
    fn test_alternate_conformation_handling() {
        let altloc_pdb = r#"ATOM      1  CA AALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA BALA A   1       0.100   0.100   0.100  1.00  0.00           C
END
"#;

        let result = sanitize_pdb(altloc_pdb, "TEST").unwrap();

        // Should only keep A conformation
        assert_eq!(result.atoms.len(), 1);
        assert!((result.atoms[0].position[0] - 0.0).abs() < 0.001);
        assert_eq!(result.stats.altloc_removed, 1);
    }

    #[test]
    fn test_empty_structure_error() {
        let empty_pdb = "HEADER EMPTY\nEND\n";
        let result = sanitize_pdb(empty_pdb, "EMPTY");

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No valid atoms"));
    }

    #[test]
    fn test_no_ca_error() {
        // Structure with only sidechain atoms
        let no_ca_pdb = r#"ATOM      1  CB  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
END
"#;
        let result = sanitize_pdb(no_ca_pdb, "NOCA");

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No Cα atoms"));
    }

    #[test]
    fn test_fits_nova_limit() {
        let result = sanitize_pdb(SAMPLE_PDB, "TEST").unwrap();
        assert!(result.fits_nova_limit()); // 9 atoms << 512
    }

    #[test]
    fn test_nonstandard_residue_removal() {
        let nonstandard_pdb = r#"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  MSE A   2       5.000   5.000   5.000  1.00  0.00           C
END
"#;

        let result = sanitize_pdb(nonstandard_pdb, "TEST").unwrap();
        assert_eq!(result.ca_residues.len(), 1);
        assert_eq!(result.stats.nonstandard_removed, 1);
    }

    #[test]
    fn test_with_additional_residues() {
        let nonstandard_pdb = r#"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  MSE A   2       5.000   5.000   5.000  1.00  0.00           C
END
"#;

        let result = PdbSanitizer::new()
            .with_additional_residues(&["MSE"])
            .sanitize(nonstandard_pdb, "TEST")
            .unwrap();

        assert_eq!(result.ca_residues.len(), 2);
    }
}
