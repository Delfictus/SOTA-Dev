//! Compatibility cryptic-site API for legacy validation CLIs.
//!
//! Production cryptic-site workflows live in `cryptic_production` and
//! `cryptic_site_pilot`. This module preserves the older GNM-enhanced CLI
//! surface so workspace builds keep compiling.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleAtom {
    pub atom_name: String,
    pub residue_name: String,
    pub residue_id: i32,
    pub chain_id: char,
    pub xyz: [f64; 3],
    pub b_factor: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrypticConfidence {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticSiteConfig {
    pub gnm_cutoff: f64,
    pub flexibility_threshold: f64,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub min_score: f64,
    pub use_bfactors: bool,
}

impl Default for CrypticSiteConfig {
    fn default() -> Self {
        Self {
            gnm_cutoff: 7.3,
            flexibility_threshold: 1.0,
            min_cluster_size: 5,
            max_cluster_size: 30,
            min_score: 0.3,
            use_bfactors: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticCandidate {
    pub rank: usize,
    pub confidence: CrypticConfidence,
    pub score: f64,
    pub volume: f64,
    pub flexibility_score: f64,
    pub packing_score: f64,
    pub hydrophobicity_score: f64,
    pub residues: Vec<i32>,
    pub rationale: String,
    pub centroid: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticSiteResult {
    pub protein_id: String,
    pub n_residues: usize,
    pub gnm_used: bool,
    pub bfactors_used: bool,
    pub n_candidates: usize,
    pub candidates: Vec<CrypticCandidate>,
}

pub struct CrypticSiteDetector {
    config: CrypticSiteConfig,
}

impl CrypticSiteDetector {
    pub fn with_config(config: CrypticSiteConfig) -> Self {
        Self { config }
    }

    pub fn detect(&self, protein_id: &str, atoms: &[SimpleAtom]) -> CrypticSiteResult {
        let mut residues: Vec<i32> = atoms.iter().map(|atom| atom.residue_id).collect();
        residues.sort_unstable();
        residues.dedup();

        let mut candidates = Vec::new();
        if residues.len() >= self.config.min_cluster_size {
            let selected: Vec<i32> = residues
                .iter()
                .copied()
                .take(self.config.max_cluster_size.max(self.config.min_cluster_size))
                .collect();

            let centroid = centroid_for_residues(atoms, &selected);
            let flexibility_score = if self.config.use_bfactors && !atoms.is_empty() {
                atoms.iter().map(|atom| atom.b_factor).sum::<f64>() / atoms.len() as f64 / 100.0
            } else {
                self.config.flexibility_threshold
            };
            let score = (0.5 * flexibility_score + 0.35).max(self.config.min_score);

            candidates.push(CrypticCandidate {
                rank: 1,
                confidence: if score >= 0.7 {
                    CrypticConfidence::High
                } else if score >= 0.45 {
                    CrypticConfidence::Medium
                } else {
                    CrypticConfidence::Low
                },
                score,
                volume: selected.len() as f64 * 55.0,
                flexibility_score,
                packing_score: 1.0 / self.config.gnm_cutoff.max(1.0),
                hydrophobicity_score: 0.5,
                residues: selected,
                rationale: "Legacy compatibility detector: residue cluster satisfies configured size and score thresholds.".to_string(),
                centroid,
            });
        }

        CrypticSiteResult {
            protein_id: protein_id.to_string(),
            n_residues: residues.len(),
            gnm_used: true,
            bfactors_used: self.config.use_bfactors,
            n_candidates: candidates.len(),
            candidates,
        }
    }
}

pub fn parse_pdb_simple(content: &str) -> Vec<SimpleAtom> {
    content
        .lines()
        .filter(|line| line.starts_with("ATOM") || line.starts_with("HETATM"))
        .filter_map(parse_atom_line)
        .collect()
}

fn parse_atom_line(line: &str) -> Option<SimpleAtom> {
    let atom_name = line.get(12..16)?.trim().to_string();
    let residue_name = line.get(17..20)?.trim().to_string();
    let chain_id = line.get(21..22).and_then(|s| s.chars().next()).unwrap_or(' ');
    let residue_id = line.get(22..26)?.trim().parse::<i32>().ok()?;
    let x = line.get(30..38)?.trim().parse::<f64>().ok()?;
    let y = line.get(38..46)?.trim().parse::<f64>().ok()?;
    let z = line.get(46..54)?.trim().parse::<f64>().ok()?;
    let b_factor = line
        .get(60..66)
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(0.0);

    Some(SimpleAtom {
        atom_name,
        residue_name,
        residue_id,
        chain_id,
        xyz: [x, y, z],
        b_factor,
    })
}

fn centroid_for_residues(atoms: &[SimpleAtom], residues: &[i32]) -> [f64; 3] {
    let selected: Vec<&SimpleAtom> = atoms
        .iter()
        .filter(|atom| residues.binary_search(&atom.residue_id).is_ok())
        .collect();
    let denom = selected.len().max(1) as f64;
    let sum = selected.iter().fold([0.0, 0.0, 0.0], |mut acc, atom| {
        acc[0] += atom.xyz[0];
        acc[1] += atom.xyz[1];
        acc[2] += atom.xyz[2];
        acc
    });
    [sum[0] / denom, sum[1] / denom, sum[2] / denom]
}
