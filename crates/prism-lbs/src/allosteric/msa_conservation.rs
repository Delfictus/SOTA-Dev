//! Stage 2: MSA Conservation Analysis
//!
//! Analyzes Multiple Sequence Alignments to identify conserved regions.
//! Uses Shannon entropy-based conservation scoring with pseudocounts.
//!
//! Supports:
//! - FASTA format MSA files
//! - Stockholm format MSA files
//! - Clustal format MSA files
//! - Position-to-structure residue mapping

use crate::structure::Atom;
use super::types::*;
use super::domain_decomposition::{calculate_residue_centroid, euclidean_distance};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// MSA conservation analyzer
pub struct ConservationAnalyzer {
    /// Minimum sequence identity threshold
    pub min_identity: f64,
    /// Pseudocount for frequency calculation
    pub pseudocount: f64,
    /// Gap penalty factor
    pub gap_penalty: f64,
}

impl Default for ConservationAnalyzer {
    fn default() -> Self {
        Self {
            min_identity: 0.3,
            pseudocount: 1.0,
            gap_penalty: 0.5,
        }
    }
}

impl ConservationAnalyzer {
    pub fn new(pseudocount: f64) -> Self {
        Self {
            pseudocount,
            ..Default::default()
        }
    }

    /// Parse MSA from file (auto-detects format)
    pub fn parse_msa(&self, msa_path: &Path) -> Result<MSA, String> {
        let content = std::fs::read_to_string(msa_path)
            .map_err(|e| format!("Failed to read MSA file: {}", e))?;

        if content.starts_with('>') {
            self.parse_fasta_msa(&content)
        } else if content.starts_with("# STOCKHOLM") {
            self.parse_stockholm_msa(&content)
        } else if content.contains("CLUSTAL") {
            self.parse_clustal_msa(&content)
        } else {
            // Try FASTA by default
            self.parse_fasta_msa(&content)
        }
    }

    /// Parse FASTA format MSA
    pub fn parse_fasta_msa(&self, content: &str) -> Result<MSA, String> {
        let mut sequences = Vec::new();
        let mut sequence_ids = Vec::new();
        let mut current_id = String::new();
        let mut current_seq = String::new();

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('>') {
                // Save previous sequence
                if !current_seq.is_empty() {
                    sequences.push(current_seq.clone());
                    sequence_ids.push(current_id.clone());
                }
                // Parse new header
                current_id = line[1..].split_whitespace().next().unwrap_or("").to_string();
                current_seq.clear();
            } else if !line.is_empty() && !line.starts_with(';') {
                current_seq.push_str(line);
            }
        }

        // Don't forget last sequence
        if !current_seq.is_empty() {
            sequences.push(current_seq);
            sequence_ids.push(current_id);
        }

        if sequences.is_empty() {
            return Err("No sequences found in MSA".to_string());
        }

        // Validate alignment length
        let alignment_length = sequences[0].len();
        for (i, seq) in sequences.iter().enumerate() {
            if seq.len() != alignment_length {
                return Err(format!(
                    "Sequence {} has length {}, expected {}",
                    sequence_ids[i],
                    seq.len(),
                    alignment_length
                ));
            }
        }

        Ok(MSA::new(sequences, sequence_ids))
    }

    /// Parse Stockholm format MSA
    pub fn parse_stockholm_msa(&self, content: &str) -> Result<MSA, String> {
        let mut seq_data: HashMap<String, String> = HashMap::new();
        let mut seq_order: Vec<String> = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // Skip comments and markup lines
            if line.is_empty()
                || line.starts_with('#')
                || line.starts_with("//")
                || line.starts_with("GC")
                || line.starts_with("GR")
            {
                continue;
            }

            // Parse sequence line: "seqid  SEQUENCE"
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let seq_id = parts[0].to_string();
                let seq_part = parts[1..].join("");

                if !seq_data.contains_key(&seq_id) {
                    seq_order.push(seq_id.clone());
                }
                seq_data
                    .entry(seq_id)
                    .or_insert_with(String::new)
                    .push_str(&seq_part);
            }
        }

        let sequences: Vec<String> = seq_order.iter().map(|id| seq_data[id].clone()).collect();

        if sequences.is_empty() {
            return Err("No sequences found in Stockholm MSA".to_string());
        }

        Ok(MSA::new(sequences, seq_order))
    }

    /// Parse Clustal format MSA
    pub fn parse_clustal_msa(&self, content: &str) -> Result<MSA, String> {
        let mut seq_data: HashMap<String, String> = HashMap::new();
        let mut seq_order: Vec<String> = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // Skip header and conservation lines
            if line.is_empty()
                || line.starts_with("CLUSTAL")
                || line.chars().all(|c| c == '*' || c == ':' || c == '.' || c == ' ')
            {
                continue;
            }

            // Parse sequence line: "seqid  SEQUENCE"
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let seq_id = parts[0].to_string();
                let seq_part = parts[1];

                if !seq_data.contains_key(&seq_id) {
                    seq_order.push(seq_id.clone());
                }
                seq_data
                    .entry(seq_id)
                    .or_insert_with(String::new)
                    .push_str(seq_part);
            }
        }

        let sequences: Vec<String> = seq_order.iter().map(|id| seq_data[id].clone()).collect();

        if sequences.is_empty() {
            return Err("No sequences found in Clustal MSA".to_string());
        }

        Ok(MSA::new(sequences, seq_order))
    }

    /// Calculate per-position conservation scores
    pub fn calculate_conservation(&self, msa: &MSA) -> Vec<ConservationScore> {
        let n_seqs = msa.sequences.len() as f64;
        let mut scores = Vec::with_capacity(msa.alignment_length);

        // Standard amino acid alphabet
        let amino_acids: Vec<char> = "ACDEFGHIKLMNPQRSTVWY".chars().collect();
        let max_entropy = (amino_acids.len() as f64).ln();

        for pos in 0..msa.alignment_length {
            // Count amino acid frequencies at this position
            let mut aa_counts: HashMap<char, f64> = HashMap::new();
            let mut gap_count = 0.0;

            for seq in &msa.sequences {
                let aa = seq.chars().nth(pos).unwrap_or('-').to_ascii_uppercase();
                if aa == '-' || aa == '.' || aa == 'X' {
                    gap_count += 1.0;
                } else {
                    *aa_counts.entry(aa).or_insert(self.pseudocount) += 1.0;
                }
            }

            // Total with pseudocounts
            let total: f64 = aa_counts.values().sum();

            // Shannon entropy
            let entropy: f64 = if total > 0.0 {
                aa_counts
                    .values()
                    .map(|&count| {
                        let p = count / total;
                        if p > 0.0 {
                            -p * p.ln()
                        } else {
                            0.0
                        }
                    })
                    .sum()
            } else {
                max_entropy
            };

            // Conservation = 1 - normalized entropy
            let conservation = 1.0 - (entropy / max_entropy);

            // Gap penalty
            let gap_fraction = gap_count / n_seqs;
            let adjusted_conservation = conservation * (1.0 - self.gap_penalty * gap_fraction);

            // Find dominant amino acid
            let (dominant_aa, dominant_freq) = aa_counts
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(&aa, &count)| (aa, count / total))
                .unwrap_or(('-', 0.0));

            scores.push(ConservationScore {
                position: pos,
                conservation: adjusted_conservation.clamp(0.0, 1.0),
                entropy,
                gap_fraction,
                dominant_aa,
                dominant_frequency: dominant_freq,
            });
        }

        scores
    }

    /// Generate sequence mapping from alignment to structure
    pub fn generate_sequence_mapping(
        &self,
        msa: &MSA,
        atoms: &[Atom],
    ) -> SequenceMapping {
        let mut mapping = SequenceMapping::new();

        // Get reference sequence (first in alignment)
        let ref_seq = &msa.sequences[msa.reference_index];

        // Get structure sequence from atoms
        let mut structure_residues: Vec<(i32, char)> = Vec::new();
        let mut seen_residues = HashSet::new();

        for atom in atoms {
            if atom.name.trim() == "CA" && !seen_residues.contains(&atom.residue_seq) {
                seen_residues.insert(atom.residue_seq);
                structure_residues.push((atom.residue_seq, atom.residue_name_to_one_letter()));
            }
        }
        structure_residues.sort_by_key(|(seq, _)| *seq);

        // Align reference sequence to structure sequence
        // Simple ungapped alignment (assuming they match)
        let mut struct_idx = 0;
        for (align_pos, ref_char) in ref_seq.chars().enumerate() {
            let ref_aa = ref_char.to_ascii_uppercase();

            // Skip gaps in alignment
            if ref_aa == '-' || ref_aa == '.' {
                continue;
            }

            // Match to structure
            while struct_idx < structure_residues.len() {
                let (res_seq, struct_aa) = structure_residues[struct_idx];

                if ref_aa == struct_aa {
                    mapping.add_mapping(align_pos, res_seq);
                    struct_idx += 1;
                    break;
                } else {
                    // Mismatch - try next structure residue
                    struct_idx += 1;
                }
            }
        }

        mapping
    }

    /// Map conservation scores to structure residues
    pub fn map_to_structure(
        &self,
        conservation: &[ConservationScore],
        mapping: &SequenceMapping,
    ) -> HashMap<i32, f64> {
        let mut residue_conservation = HashMap::new();

        for score in conservation {
            if let Some(residue_seq) = mapping.alignment_to_structure(score.position) {
                residue_conservation.insert(residue_seq, score.conservation);
            }
        }

        residue_conservation
    }

    /// Find spatially clustered conserved residues
    pub fn find_conserved_clusters(
        &self,
        atoms: &[Atom],
        residue_conservation: &HashMap<i32, f64>,
        threshold: f64,
    ) -> Vec<ConservedCluster> {
        // Get highly conserved residues
        let conserved_residues: Vec<i32> = residue_conservation
            .iter()
            .filter(|(_, &score)| score >= threshold)
            .map(|(&res, _)| res)
            .collect();

        if conserved_residues.is_empty() {
            return Vec::new();
        }

        // Spatial clustering using distance cutoff
        let clusters = spatial_cluster(atoms, &conserved_residues, 8.0);

        clusters
            .into_iter()
            .filter(|c| c.len() >= 3)
            .map(|residues| {
                let mean_conservation = residues
                    .iter()
                    .filter_map(|r| residue_conservation.get(r))
                    .sum::<f64>()
                    / residues.len() as f64;

                let centroid = calculate_residue_centroid(atoms, &residues);

                ConservedCluster {
                    residues,
                    mean_conservation,
                    centroid,
                    annotation: None,
                }
            })
            .collect()
    }

    /// Calculate sequence identity between two sequences
    pub fn sequence_identity(&self, seq1: &str, seq2: &str) -> f64 {
        if seq1.len() != seq2.len() {
            return 0.0;
        }

        let matches: usize = seq1
            .chars()
            .zip(seq2.chars())
            .filter(|(a, b)| {
                let aa = a.to_ascii_uppercase();
                let bb = b.to_ascii_uppercase();
                aa != '-' && aa != '.' && aa == bb
            })
            .count();

        let non_gap: usize = seq1
            .chars()
            .zip(seq2.chars())
            .filter(|(a, b)| {
                let aa = a.to_ascii_uppercase();
                let bb = b.to_ascii_uppercase();
                aa != '-' && aa != '.' && bb != '-' && bb != '.'
            })
            .count();

        if non_gap > 0 {
            matches as f64 / non_gap as f64
        } else {
            0.0
        }
    }

    /// Filter MSA to remove sequences below identity threshold
    pub fn filter_by_identity(&self, msa: &MSA, min_identity: f64) -> MSA {
        let ref_seq = &msa.sequences[msa.reference_index];

        let filtered: Vec<(String, String)> = msa
            .sequences
            .iter()
            .zip(msa.sequence_ids.iter())
            .filter(|(seq, _)| {
                let identity = self.sequence_identity(ref_seq, seq);
                identity >= min_identity
            })
            .map(|(seq, id)| (seq.clone(), id.clone()))
            .collect();

        let (sequences, sequence_ids): (Vec<String>, Vec<String>) = filtered.into_iter().unzip();

        MSA::new(sequences, sequence_ids)
    }
}

/// Spatial clustering of residues using distance cutoff
pub fn spatial_cluster(atoms: &[Atom], residues: &[i32], cutoff: f64) -> Vec<Vec<i32>> {
    if residues.is_empty() {
        return Vec::new();
    }

    // Get Cα coordinates
    let ca_coords: HashMap<i32, [f64; 3]> = atoms
        .iter()
        .filter(|a| a.name.trim() == "CA")
        .map(|a| (a.residue_seq, a.coord))
        .collect();

    let cutoff_sq = cutoff * cutoff;
    let mut clusters: Vec<Vec<i32>> = Vec::new();
    let mut assigned: HashSet<i32> = HashSet::new();

    for &res in residues {
        if assigned.contains(&res) {
            continue;
        }

        let coord = match ca_coords.get(&res) {
            Some(c) => c,
            None => continue,
        };

        // Start new cluster
        let mut cluster = vec![res];
        assigned.insert(res);
        let mut frontier = vec![res];

        // Grow cluster
        while let Some(current) = frontier.pop() {
            let current_coord = match ca_coords.get(&current) {
                Some(c) => c,
                None => continue,
            };

            for &other in residues {
                if assigned.contains(&other) {
                    continue;
                }

                if let Some(other_coord) = ca_coords.get(&other) {
                    let dist_sq = (current_coord[0] - other_coord[0]).powi(2)
                        + (current_coord[1] - other_coord[1]).powi(2)
                        + (current_coord[2] - other_coord[2]).powi(2);

                    if dist_sq < cutoff_sq {
                        cluster.push(other);
                        assigned.insert(other);
                        frontier.push(other);
                    }
                }
            }
        }

        clusters.push(cluster);
    }

    clusters
}

/// Generate a simple conservation profile without full MSA (B-factor approximation)
pub fn estimate_conservation_from_bfactors(atoms: &[Atom]) -> HashMap<i32, f64> {
    // B-factor can be a rough proxy for conservation/flexibility
    // Lower B-factor often correlates with conserved core residues

    let mut residue_bfactors: HashMap<i32, (f64, usize)> = HashMap::new();

    for atom in atoms {
        if atom.name.trim() == "CA" {
            let entry = residue_bfactors
                .entry(atom.residue_seq)
                .or_insert((0.0, 0));
            entry.0 += atom.b_factor;
            entry.1 += 1;
        }
    }

    // Get B-factor range for normalization
    let bfactors: Vec<f64> = residue_bfactors
        .values()
        .filter_map(|(sum, count)| {
            if *count > 0 {
                Some(sum / *count as f64)
            } else {
                None
            }
        })
        .collect();

    if bfactors.is_empty() {
        return HashMap::new();
    }

    let min_bf = bfactors.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_bf = bfactors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_bf - min_bf;

    // Convert: low B-factor → high "conservation" estimate
    residue_bfactors
        .into_iter()
        .filter_map(|(res, (sum, count))| {
            if count > 0 && range > 0.0 {
                let avg_bf = sum / count as f64;
                // Invert: low B-factor = high score
                let normalized = 1.0 - (avg_bf - min_bf) / range;
                Some((res, normalized.clamp(0.0, 1.0)))
            } else {
                None
            }
        })
        .collect()
}

// Extension trait for Atom to convert residue name to single letter code
trait AtomExt {
    fn residue_name_to_one_letter(&self) -> char;
}

impl AtomExt for Atom {
    fn residue_name_to_one_letter(&self) -> char {
        match self.residue_name.trim().to_uppercase().as_str() {
            "ALA" => 'A',
            "CYS" => 'C',
            "ASP" => 'D',
            "GLU" => 'E',
            "PHE" => 'F',
            "GLY" => 'G',
            "HIS" => 'H',
            "ILE" => 'I',
            "LYS" => 'K',
            "LEU" => 'L',
            "MET" => 'M',
            "ASN" => 'N',
            "PRO" => 'P',
            "GLN" => 'Q',
            "ARG" => 'R',
            "SER" => 'S',
            "THR" => 'T',
            "VAL" => 'V',
            "TRP" => 'W',
            "TYR" => 'Y',
            _ => 'X',
        }
    }
}
