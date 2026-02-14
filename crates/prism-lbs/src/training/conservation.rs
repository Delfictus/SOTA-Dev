//! Conservation data loader (ConSurf-style)
//!
//! Loads sequence conservation scores from MSA-derived data.
//! Supports ConSurf grades format and custom conservation profiles.

use crate::{LbsError, ProteinStructure};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Conservation data for a protein
#[derive(Debug, Clone, Default)]
pub struct ConservationData {
    /// Conservation scores by residue (chain_id, residue_number) -> score
    pub residue_scores: HashMap<(char, i32), f64>,
    /// Conservation scores by atom index
    pub atom_scores: Vec<f64>,
    /// Sequence identity of MSA
    pub sequence_identity: f64,
    /// Number of sequences in MSA
    pub num_sequences: usize,
}

/// Conservation score source type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConservationSource {
    /// ConSurf server output
    ConSurf,
    /// Rate4Site output
    Rate4Site,
    /// BLAST-derived conservation
    Blast,
    /// Custom format (column index for score)
    Custom(usize),
}

/// Conservation data loader configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationConfig {
    /// Source format
    pub source: ConservationSource,
    /// Directory containing conservation files
    pub data_dir: Option<String>,
    /// Use precomputed database
    pub use_database: bool,
    /// Fallback value for missing residues
    pub default_score: f64,
}

impl Default for ConservationConfig {
    fn default() -> Self {
        Self {
            source: ConservationSource::ConSurf,
            data_dir: None,
            use_database: true,
            default_score: 0.5,  // Neutral conservation
        }
    }
}

/// Conservation data loader
pub struct ConservationLoader {
    config: ConservationConfig,
    /// Precomputed conservation database (PDB ID -> data)
    database: HashMap<String, ConservationData>,
}

impl ConservationLoader {
    /// Create new conservation loader
    pub fn new(config: ConservationConfig) -> Self {
        Self {
            config,
            database: HashMap::new(),
        }
    }

    /// Load conservation database from directory
    pub fn load_database(&mut self, dir: &Path) -> Result<usize, LbsError> {
        if !dir.exists() {
            return Ok(0);
        }

        let mut count = 0;
        for entry in fs::read_dir(dir).map_err(LbsError::Io)? {
            let entry = entry.map_err(LbsError::Io)?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("cons") {
                if let Some(pdb_id) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(data) = self.load_file(&path) {
                        self.database.insert(pdb_id.to_lowercase(), data);
                        count += 1;
                    }
                }
            }
        }

        log::info!("Loaded {} conservation profiles", count);
        Ok(count)
    }

    /// Load conservation data for a structure
    pub fn load_for_structure(&self, structure: &ProteinStructure) -> ConservationData {
        // Try database lookup first
        let pdb_id = structure.title.chars().take(4).collect::<String>().to_lowercase();
        if let Some(data) = self.database.get(&pdb_id) {
            return self.apply_to_structure(data, structure);
        }

        // Try loading from file
        if let Some(ref data_dir) = self.config.data_dir {
            let path = Path::new(data_dir).join(format!("{}.cons", pdb_id));
            if path.exists() {
                if let Ok(data) = self.load_file(&path) {
                    return self.apply_to_structure(&data, structure);
                }
            }
        }

        // Return default scores
        self.default_conservation(structure)
    }

    /// Load conservation from file
    pub fn load_file(&self, path: &Path) -> Result<ConservationData, LbsError> {
        let content = fs::read_to_string(path).map_err(LbsError::Io)?;

        match self.config.source {
            ConservationSource::ConSurf => self.parse_consurf(&content),
            ConservationSource::Rate4Site => self.parse_rate4site(&content),
            ConservationSource::Blast => self.parse_blast(&content),
            ConservationSource::Custom(col) => self.parse_custom(&content, col),
        }
    }

    /// Parse ConSurf grades file format
    fn parse_consurf(&self, content: &str) -> Result<ConservationData, LbsError> {
        let mut data = ConservationData::default();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("POS") {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                // Format: POS CHAIN RES GRADE [SCORE] [...]
                // ConSurf grades: 1-9 (1=variable, 9=conserved)
                if let Ok(pos) = parts[0].parse::<i32>() {
                    let chain = parts[1].chars().next().unwrap_or('A');

                    // Grade is typically in column 3 or 4
                    let grade: f64 = parts.get(3)
                        .or(parts.get(2))
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(5.0);

                    // Normalize grade 1-9 to 0-1
                    let score = (grade - 1.0) / 8.0;
                    data.residue_scores.insert((chain, pos), score.clamp(0.0, 1.0));
                }
            }
        }

        Ok(data)
    }

    /// Parse Rate4Site output format
    fn parse_rate4site(&self, content: &str) -> Result<ConservationData, LbsError> {
        let mut data = ConservationData::default();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                // Format: POS AA SCORE [...]
                if let (Ok(pos), Ok(rate)) = (parts[0].parse::<i32>(), parts[2].parse::<f64>()) {
                    // Rate4Site: negative = conserved, positive = variable
                    // Convert to 0-1 where 1 = conserved
                    let score = 1.0 / (1.0 + (rate).exp());  // Sigmoid transform
                    data.residue_scores.insert(('A', pos), score);
                }
            }
        }

        Ok(data)
    }

    /// Parse BLAST-derived conservation (simple position-specific scoring)
    fn parse_blast(&self, content: &str) -> Result<ConservationData, LbsError> {
        let mut data = ConservationData::default();

        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Simple format: one score per line, or space-separated scores
            for (j, part) in line.split_whitespace().enumerate() {
                if let Ok(score) = part.parse::<f64>() {
                    let pos = (i * 20 + j + 1) as i32;  // Approximate position
                    data.residue_scores.insert(('A', pos), score.clamp(0.0, 1.0));
                }
            }
        }

        Ok(data)
    }

    /// Parse custom format with specified score column
    fn parse_custom(&self, content: &str, score_col: usize) -> Result<ConservationData, LbsError> {
        let mut data = ConservationData::default();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split([',', '\t', ' '])
                .filter(|s| !s.is_empty())
                .collect();

            if parts.len() > score_col {
                if let (Ok(pos), Ok(score)) = (
                    parts[0].parse::<i32>(),
                    parts[score_col].parse::<f64>()
                ) {
                    let chain = parts.get(1)
                        .and_then(|s| s.chars().next())
                        .unwrap_or('A');
                    data.residue_scores.insert((chain, pos), score.clamp(0.0, 1.0));
                }
            }
        }

        Ok(data)
    }

    /// Apply conservation data to structure atoms
    fn apply_to_structure(&self, data: &ConservationData, structure: &ProteinStructure) -> ConservationData {
        let mut result = data.clone();
        result.atom_scores = Vec::with_capacity(structure.atoms.len());

        for atom in &structure.atoms {
            let key = (atom.chain_id, atom.residue_seq);
            let score = data.residue_scores.get(&key)
                .copied()
                .unwrap_or(self.config.default_score);
            result.atom_scores.push(score);
        }

        result
    }

    /// Create default conservation (neutral scores)
    fn default_conservation(&self, structure: &ProteinStructure) -> ConservationData {
        ConservationData {
            residue_scores: HashMap::new(),
            atom_scores: vec![self.config.default_score; structure.atoms.len()],
            sequence_identity: 0.0,
            num_sequences: 0,
        }
    }

    /// Apply conservation to protein structure (mutates structure)
    pub fn apply(&self, structure: &mut ProteinStructure) {
        let data = self.load_for_structure(structure);

        for (i, atom) in structure.atoms.iter_mut().enumerate() {
            if i < data.atom_scores.len() {
                // Update atom's conservation-related properties
                // (This would need atom struct to have conservation field)
            }
        }

        // Update residue conservation scores
        for residue in &mut structure.residues {
            let key = (residue.chain_id, residue.seq_number);
            if let Some(&score) = data.residue_scores.get(&key) {
                residue.conservation_score = score;
            }
        }
    }

    /// Compute conservation from MSA alignment (simple entropy-based)
    pub fn compute_from_msa(alignment: &[String]) -> ConservationData {
        if alignment.is_empty() {
            return ConservationData::default();
        }

        let seq_len = alignment[0].len();
        let num_seqs = alignment.len() as f64;
        let mut data = ConservationData::default();
        data.num_sequences = alignment.len();

        for pos in 0..seq_len {
            let mut counts: HashMap<char, usize> = HashMap::new();

            for seq in alignment {
                if let Some(aa) = seq.chars().nth(pos) {
                    if aa != '-' && aa != '.' {
                        *counts.entry(aa.to_ascii_uppercase()).or_insert(0) += 1;
                    }
                }
            }

            // Shannon entropy
            let total: usize = counts.values().sum();
            if total == 0 {
                continue;
            }

            let entropy: f64 = counts.values()
                .map(|&c| {
                    let p = c as f64 / total as f64;
                    if p > 0.0 { -p * p.ln() } else { 0.0 }
                })
                .sum();

            // Normalize entropy (max entropy = ln(20) for amino acids)
            let max_entropy = 20.0_f64.ln();
            let normalized = 1.0 - (entropy / max_entropy);  // 1 = conserved, 0 = variable

            data.residue_scores.insert(('A', (pos + 1) as i32), normalized);
        }

        // Compute sequence identity
        if alignment.len() >= 2 {
            let ref_seq = &alignment[0];
            let mut total_identity = 0.0;
            for seq in alignment.iter().skip(1) {
                let matches: usize = ref_seq.chars()
                    .zip(seq.chars())
                    .filter(|(a, b)| a == b && *a != '-')
                    .count();
                let len = ref_seq.len().max(seq.len());
                total_identity += matches as f64 / len as f64;
            }
            data.sequence_identity = total_identity / (alignment.len() - 1) as f64;
        }

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_conservation() {
        let alignment = vec![
            "ACDEFGHIKLMNPQRSTVWY".to_string(),
            "ACDEFGHIKLMNPQRSTVWY".to_string(),
            "ACDEFGHIKLMNPQRSTVWY".to_string(),
        ];

        let data = ConservationLoader::compute_from_msa(&alignment);

        // Identical sequences should have high conservation
        for (_, score) in data.residue_scores {
            assert!(score > 0.9, "Identical positions should be highly conserved");
        }
    }
}
