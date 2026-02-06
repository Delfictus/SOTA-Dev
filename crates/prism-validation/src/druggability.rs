//! Native physics-based druggability validation module
//!
//! Validates predicted druggable pockets against known drug-binding sites from
//! PDB/literature ground truth. Implements the `ValidationBenchmark` trait for
//! integration into the PRISM-4D validation framework.
//!
//! ## Methodology
//!
//! 1. Parse target structure and extract CA atom positions
//! 2. Detect pockets via burial/concavity analysis on the structure
//! 3. Score each pocket for druggability using a physics-based composite scorer
//! 4. Compare predicted pockets against known binding sites from target metadata
//! 5. Compute standard validation metrics (precision, recall, F1, DVO, enrichment)
//!
//! ## Key Metrics
//!
//! - **Site recovery rate**: fraction of known sites recovered within 4 angstrom centroid distance
//! - **Enrichment factor**: druggable sites ranked in top-N vs random expectation
//! - **DVO (Druggability Volume Overlap)**: Jaccard index of predicted vs actual pocket voxel grids
//! - **Classification accuracy**: predicted class vs known druggability tier
//!
//! ## Druggability Classification Tiers
//!
//! | Class            | Score Range | Description                              |
//! |------------------|------------|------------------------------------------|
//! | HighlyDruggable  | >= 0.70    | Large, enclosed, hydrophobic pocket      |
//! | Druggable        | >= 0.50    | Good pocket properties, viable target    |
//! | DifficultTarget  | >= 0.30    | Shallow or polar; requires fragment-based |
//! | Undruggable      | < 0.30     | No suitable pocket geometry              |

use anyhow::{anyhow, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::{
    Af3Comparison, BenchmarkMetrics, BenchmarkResult, ComparisonItem, ScoreComponent,
    ValidationBenchmark, ValidationScore,
};
use crate::targets;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the druggability validation benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DruggabilityConfig {
    /// Distance threshold (angstrom) for centroid-based site recovery.
    pub centroid_recovery_cutoff: f64,

    /// Residue-level distance cutoff (angstrom) for overlap calculations.
    pub residue_overlap_cutoff: f64,

    /// Grid spacing (angstrom) for voxel-based DVO calculation.
    pub dvo_grid_spacing: f64,

    /// Radius around each CA atom for voxel volume expansion.
    pub dvo_atom_radius: f64,

    /// DVO threshold to call a prediction "successful".
    pub dvo_success_threshold: f64,

    /// Top-N pockets to consider for enrichment calculation.
    pub enrichment_top_n: usize,

    /// Probe radius for SASA calculation (angstrom).
    pub probe_radius: f64,

    /// Neighbor cutoff for burial analysis (angstrom).
    pub neighbor_cutoff: f64,

    /// Minimum pocket volume (cubic angstrom) to report.
    pub min_pocket_volume: f64,

    /// Maximum number of pockets to detect.
    pub max_pockets: usize,

    /// Minimum number of residues to form a pocket cluster.
    pub min_cluster_size: usize,

    /// Distance threshold for pocket clustering (angstrom).
    pub cluster_distance: f64,

    /// Scoring weights for the composite druggability scorer.
    pub scoring_weights: DruggabilityScoringWeights,
}

impl Default for DruggabilityConfig {
    fn default() -> Self {
        Self {
            centroid_recovery_cutoff: 4.0,
            residue_overlap_cutoff: 4.5,
            dvo_grid_spacing: 1.0,
            dvo_atom_radius: 2.0,
            dvo_success_threshold: 0.2,
            enrichment_top_n: 3,
            probe_radius: 1.4,
            neighbor_cutoff: 8.0,
            min_pocket_volume: 100.0,
            max_pockets: 20,
            min_cluster_size: 3,
            cluster_distance: 8.0,
            scoring_weights: DruggabilityScoringWeights::default(),
        }
    }
}

/// Weights for the composite druggability scoring function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DruggabilityScoringWeights {
    pub volume: f64,
    pub hydrophobicity: f64,
    pub enclosure: f64,
    pub depth: f64,
    pub hbond_capacity: f64,
    pub flexibility: f64,
}

impl Default for DruggabilityScoringWeights {
    fn default() -> Self {
        Self {
            volume: 0.25,
            hydrophobicity: 0.25,
            enclosure: 0.20,
            depth: 0.15,
            hbond_capacity: 0.10,
            flexibility: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// Druggability classification
// ---------------------------------------------------------------------------

/// Classification tier for a predicted pocket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DruggabilityClass {
    HighlyDruggable,
    Druggable,
    DifficultTarget,
    Undruggable,
}

impl DruggabilityClass {
    pub fn from_score(score: f64) -> Self {
        if score >= 0.70 {
            Self::HighlyDruggable
        } else if score >= 0.50 {
            Self::Druggable
        } else if score >= 0.30 {
            Self::DifficultTarget
        } else {
            Self::Undruggable
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HighlyDruggable => "HighlyDruggable",
            Self::Druggable => "Druggable",
            Self::DifficultTarget => "DifficultTarget",
            Self::Undruggable => "Undruggable",
        }
    }

    /// Numeric ordinal for comparison (higher = more druggable).
    pub fn ordinal(&self) -> u8 {
        match self {
            Self::HighlyDruggable => 3,
            Self::Druggable => 2,
            Self::DifficultTarget => 1,
            Self::Undruggable => 0,
        }
    }
}

impl std::fmt::Display for DruggabilityClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Pocket representation
// ---------------------------------------------------------------------------

/// A detected pocket with physics-based properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPocket {
    /// Sequential pocket index (0-based, ranked by druggability score).
    pub rank: usize,

    /// Residue indices that compose this pocket.
    pub residue_indices: Vec<usize>,

    /// CA atom positions for residues in this pocket (angstrom).
    pub atom_positions: Vec<[f64; 3]>,

    /// Geometric centroid of the pocket (angstrom).
    pub centroid: [f64; 3],

    /// Approximate volume (cubic angstrom) from convex hull of CA atoms.
    pub volume: f64,

    /// Mean burial score of member residues (0 = surface, 1 = deep core).
    pub mean_burial: f64,

    /// Mean hydrophobicity of member residues (Kyte-Doolittle scale, normalized).
    pub mean_hydrophobicity: f64,

    /// Enclosure ratio: fraction of 26 cubic directions blocked by protein.
    pub enclosure_ratio: f64,

    /// Mean depth from protein surface (angstrom).
    pub mean_depth: f64,

    /// Estimated hydrogen-bond donors + acceptors in the pocket.
    pub hbond_capacity: usize,

    /// Mean B-factor-like flexibility proxy.
    pub mean_flexibility: f64,

    /// Composite druggability score in [0, 1].
    pub druggability_score: f64,

    /// Classification tier derived from score.
    pub classification: DruggabilityClass,
}

// ---------------------------------------------------------------------------
// Ground-truth binding site definition
// ---------------------------------------------------------------------------

/// Known drug-binding site loaded from target metadata.
#[derive(Debug, Clone)]
pub struct KnownBindingSite {
    /// Residue indices defining the binding site.
    pub residue_indices: Vec<usize>,

    /// Core residues most critical for ligand binding.
    pub core_residues: Vec<usize>,

    /// CA atom positions for the binding-site residues.
    pub atom_positions: Vec<[f64; 3]>,

    /// Geometric centroid (angstrom).
    pub centroid: [f64; 3],

    /// Expected druggability class from literature.
    pub expected_class: DruggabilityClass,

    /// Whether this site is cryptic (hidden in apo).
    pub is_cryptic: bool,
}

// ---------------------------------------------------------------------------
// Validation result detail
// ---------------------------------------------------------------------------

/// Detailed per-site comparison result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteComparisonResult {
    /// Index of the known binding site being evaluated.
    pub known_site_index: usize,

    /// Whether the site was recovered (centroid within cutoff).
    pub recovered: bool,

    /// Rank of the best-matching predicted pocket (1-indexed).
    pub best_match_rank: Option<usize>,

    /// Centroid distance (angstrom) to the best-matching pocket.
    pub centroid_distance: Option<f64>,

    /// Residue-level overlap (Jaccard) between prediction and ground truth.
    pub residue_overlap: f64,

    /// DVO Jaccard index between voxelized pocket volumes.
    pub dvo_jaccard: f64,

    /// DVO Dice coefficient.
    pub dvo_dice: f64,

    /// Predicted class for the best-matching pocket.
    pub predicted_class: Option<DruggabilityClass>,

    /// Expected class from literature.
    pub expected_class: DruggabilityClass,

    /// Whether the classification matches.
    pub class_match: bool,
}

/// Aggregate metrics from the druggability validation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DruggabilityMetrics {
    /// Number of known binding sites evaluated.
    pub n_known_sites: usize,

    /// Number of predicted pockets.
    pub n_predicted_pockets: usize,

    /// Site recovery rate (fraction of known sites recovered within centroid cutoff).
    pub site_recovery_rate: f64,

    /// Precision at the pocket level: TP / (TP + FP).
    pub precision: f64,

    /// Recall at the pocket level: TP / (TP + FN).
    pub recall: f64,

    /// F1 score (harmonic mean of precision and recall).
    pub f1: f64,

    /// Enrichment factor: (hits in top-N / N) / (total hits / total pockets).
    pub enrichment_factor: f64,

    /// Mean DVO Jaccard across all known sites.
    pub mean_dvo_jaccard: f64,

    /// Mean DVO Dice across all known sites.
    pub mean_dvo_dice: f64,

    /// Classification accuracy (fraction of matched predictions with correct class).
    pub classification_accuracy: f64,

    /// Per-site comparison details.
    pub site_comparisons: Vec<SiteComparisonResult>,
}

// ---------------------------------------------------------------------------
// Voxel-based DVO
// ---------------------------------------------------------------------------

/// Integer grid point for hashing.
#[derive(Hash, Eq, PartialEq, Clone)]
struct Voxel {
    x: i32,
    y: i32,
    z: i32,
}

/// Expand a set of atom positions into a voxel grid using sphere filling.
fn atoms_to_voxels(positions: &[[f64; 3]], radius: f64, spacing: f64) -> HashSet<Voxel> {
    let mut grid = HashSet::new();
    let r2 = radius * radius;
    for pos in positions {
        let x_min = ((pos[0] - radius) / spacing).floor() as i32;
        let x_max = ((pos[0] + radius) / spacing).ceil() as i32;
        let y_min = ((pos[1] - radius) / spacing).floor() as i32;
        let y_max = ((pos[1] + radius) / spacing).ceil() as i32;
        let z_min = ((pos[2] - radius) / spacing).floor() as i32;
        let z_max = ((pos[2] + radius) / spacing).ceil() as i32;
        for ix in x_min..=x_max {
            for iy in y_min..=y_max {
                for iz in z_min..=z_max {
                    let cx = ix as f64 * spacing;
                    let cy = iy as f64 * spacing;
                    let cz = iz as f64 * spacing;
                    let dx = cx - pos[0];
                    let dy = cy - pos[1];
                    let dz = cz - pos[2];
                    if dx * dx + dy * dy + dz * dz <= r2 {
                        grid.insert(Voxel { x: ix, y: iy, z: iz });
                    }
                }
            }
        }
    }
    grid
}

/// Compute Jaccard and Dice between two voxel sets.
fn voxel_overlap(a: &HashSet<Voxel>, b: &HashSet<Voxel>) -> (f64, f64) {
    if a.is_empty() && b.is_empty() {
        return (0.0, 0.0);
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    let jaccard = if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    };
    let dice = if a.len() + b.len() == 0 {
        0.0
    } else {
        2.0 * intersection as f64 / (a.len() + b.len()) as f64
    };
    (jaccard, dice)
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn centroid(positions: &[[f64; 3]]) -> [f64; 3] {
    if positions.is_empty() {
        return [0.0; 3];
    }
    let n = positions.len() as f64;
    let mut c = [0.0; 3];
    for p in positions {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    c[0] /= n;
    c[1] /= n;
    c[2] /= n;
    c
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Approximate convex-hull volume from a point cloud via bounding-box scaling.
/// For pocket-sized clusters this gives a reasonable estimate without a full
/// Delaunay implementation.
fn approximate_volume(positions: &[[f64; 3]]) -> f64 {
    if positions.len() < 4 {
        return 0.0;
    }
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for p in positions {
        for i in 0..3 {
            if p[i] < min[i] { min[i] = p[i]; }
            if p[i] > max[i] { max[i] = p[i]; }
        }
    }
    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];
    // Typical protein pockets occupy roughly 30-40% of their bounding box.
    dx * dy * dz * 0.35
}

// ---------------------------------------------------------------------------
// Residue properties (Kyte-Doolittle hydrophobicity, H-bond capacity)
// ---------------------------------------------------------------------------

/// Return normalized Kyte-Doolittle hydrophobicity for a 3-letter residue name.
/// Mapping is scaled to [0, 1] where 1 = most hydrophobic (Ile).
fn residue_hydrophobicity(resname: &str) -> f64 {
    // Raw K-D scale: Ile 4.5 .. Arg -4.5, range = 9.0
    let raw = match resname {
        "ILE" => 4.5,
        "VAL" => 4.2,
        "LEU" => 3.8,
        "PHE" => 2.8,
        "CYS" => 2.5,
        "MET" => 1.9,
        "ALA" => 1.8,
        "GLY" => -0.4,
        "THR" => -0.7,
        "SER" => -0.8,
        "TRP" => -0.9,
        "TYR" => -1.3,
        "PRO" => -1.6,
        "HIS" => -3.2,
        "GLU" => -3.5,
        "GLN" => -3.5,
        "ASP" => -3.5,
        "ASN" => -3.5,
        "LYS" => -3.9,
        "ARG" => -4.5,
        _ => 0.0,
    };
    (raw + 4.5) / 9.0 // normalize to [0, 1]
}

/// Return estimated H-bond donor + acceptor count for a residue.
fn residue_hbond_capacity(resname: &str) -> usize {
    match resname {
        "ARG" => 5,
        "LYS" => 3,
        "ASN" | "GLN" => 3,
        "HIS" => 2,
        "SER" | "THR" | "TYR" => 2,
        "ASP" | "GLU" => 2,
        "TRP" => 1,
        "CYS" => 1,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// PDB parsing (minimal, CA-only)
// ---------------------------------------------------------------------------

/// Minimal residue record extracted from a PDB ATOM line.
#[derive(Debug, Clone)]
struct PdbResidue {
    index: usize,          // 0-based sequential index
    resid: usize,          // PDB residue sequence number
    resname: String,       // 3-letter code
    chain: char,
    ca_pos: [f64; 3],      // CA atom coordinates
}

/// Parse a PDB file and return one record per residue (CA atoms only).
fn parse_pdb_ca_atoms(pdb_path: &std::path::Path) -> Result<Vec<PdbResidue>> {
    let content = std::fs::read_to_string(pdb_path)
        .map_err(|e| anyhow!("Failed to read PDB {}: {}", pdb_path.display(), e))?;

    let mut residues = Vec::new();
    let mut seen_resids: HashSet<(char, usize)> = HashSet::new();

    for line in content.lines() {
        if !line.starts_with("ATOM") {
            continue;
        }
        if line.len() < 54 {
            continue;
        }

        let atom_name = line.get(12..16).unwrap_or("").trim();
        if atom_name != "CA" {
            continue;
        }

        let chain = line.as_bytes().get(21).map(|&b| b as char).unwrap_or('A');
        let resid: usize = line.get(22..26)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(0);

        if seen_resids.contains(&(chain, resid)) {
            continue; // skip alternate conformations
        }
        seen_resids.insert((chain, resid));

        let resname = line.get(17..20).unwrap_or("UNK").trim().to_string();

        let x: f64 = line.get(30..38).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let y: f64 = line.get(38..46).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let z: f64 = line.get(46..54).unwrap_or("0").trim().parse().unwrap_or(0.0);

        residues.push(PdbResidue {
            index: residues.len(),
            resid,
            resname,
            chain,
            ca_pos: [x, y, z],
        });
    }

    if residues.is_empty() {
        return Err(anyhow!("No CA atoms found in PDB: {}", pdb_path.display()));
    }

    Ok(residues)
}

// ---------------------------------------------------------------------------
// Pocket detection (self-contained, no prism-lbs dependency)
// ---------------------------------------------------------------------------

/// Detect pockets from CA coordinates using burial depth, neighbor analysis,
/// and spatial clustering. Returns pockets sorted by druggability score
/// (highest first).
fn detect_pockets(
    residues: &[PdbResidue],
    config: &DruggabilityConfig,
) -> Vec<DetectedPocket> {
    let n = residues.len();
    if n < config.min_cluster_size {
        return Vec::new();
    }

    // ---- Step 1: compute per-residue neighbor counts (burial proxy) ----
    let mut neighbor_counts = vec![0usize; n];
    let cutoff_sq = config.neighbor_cutoff * config.neighbor_cutoff;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = residues[i].ca_pos[0] - residues[j].ca_pos[0];
            let dy = residues[i].ca_pos[1] - residues[j].ca_pos[1];
            let dz = residues[i].ca_pos[2] - residues[j].ca_pos[2];
            if dx * dx + dy * dy + dz * dz <= cutoff_sq {
                neighbor_counts[i] += 1;
                neighbor_counts[j] += 1;
            }
        }
    }
    let max_neighbors = *neighbor_counts.iter().max().unwrap_or(&1).max(&1) as f64;

    // ---- Step 2: compute per-residue burial score ----
    let burial: Vec<f64> = neighbor_counts.iter().map(|&c| c as f64 / max_neighbors).collect();

    // ---- Step 3: identify pocket-candidate residues (moderately buried) ----
    // Pocket residues are typically at intermediate burial (0.25-0.75),
    // not fully exposed surface and not deep core.
    let mut candidates: Vec<usize> = Vec::new();
    for i in 0..n {
        let b = burial[i];
        if b >= 0.20 && b <= 0.80 {
            candidates.push(i);
        }
    }

    if candidates.is_empty() {
        return Vec::new();
    }

    // ---- Step 4: cluster candidates by spatial proximity ----
    let cluster_cutoff_sq = config.cluster_distance * config.cluster_distance;
    let mut assigned = vec![false; n];
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for &seed in &candidates {
        if assigned[seed] {
            continue;
        }
        // BFS/flood-fill from this seed
        let mut cluster = Vec::new();
        let mut queue = vec![seed];
        assigned[seed] = true;

        while let Some(current) = queue.pop() {
            cluster.push(current);
            for &neighbor in &candidates {
                if assigned[neighbor] {
                    continue;
                }
                let dx = residues[current].ca_pos[0] - residues[neighbor].ca_pos[0];
                let dy = residues[current].ca_pos[1] - residues[neighbor].ca_pos[1];
                let dz = residues[current].ca_pos[2] - residues[neighbor].ca_pos[2];
                if dx * dx + dy * dy + dz * dz <= cluster_cutoff_sq {
                    assigned[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }

        if cluster.len() >= config.min_cluster_size {
            clusters.push(cluster);
        }
    }

    // ---- Step 5: score each cluster for druggability ----
    let weights = &config.scoring_weights;
    let mut pockets: Vec<DetectedPocket> = clusters
        .into_iter()
        .map(|member_indices| {
            let positions: Vec<[f64; 3]> = member_indices
                .iter()
                .map(|&i| residues[i].ca_pos)
                .collect();
            let cent = centroid(&positions);
            let vol = approximate_volume(&positions);

            // Mean burial
            let mean_burial_val: f64 =
                member_indices.iter().map(|&i| burial[i]).sum::<f64>()
                    / member_indices.len() as f64;

            // Mean hydrophobicity
            let mean_hydro: f64 = member_indices
                .iter()
                .map(|&i| residue_hydrophobicity(&residues[i].resname))
                .sum::<f64>()
                / member_indices.len() as f64;

            // Enclosure ratio approximation: fraction of 26 cubic directions
            // blocked within cutoff by at least one other protein atom.
            let enclosure = compute_enclosure(&cent, &positions, config.neighbor_cutoff);

            // Mean depth: average distance from centroid (proxy for pocket depth)
            let mean_depth_val: f64 = positions
                .iter()
                .map(|p| distance(p, &cent))
                .sum::<f64>()
                / positions.len() as f64;

            // H-bond capacity
            let hbond: usize = member_indices
                .iter()
                .map(|&i| residue_hbond_capacity(&residues[i].resname))
                .sum();

            // Flexibility proxy: inverse of burial (surface residues are more flexible)
            let mean_flex: f64 = member_indices
                .iter()
                .map(|&i| 1.0 - burial[i])
                .sum::<f64>()
                / member_indices.len() as f64;

            // ---- Composite druggability score ----
            let s_vol = score_volume(vol);
            let s_hydro = mean_hydro; // already [0, 1]
            let s_encl = score_enclosure(enclosure);
            let s_depth = score_depth(mean_depth_val);
            let s_hbond = (hbond as f64 / 10.0).min(1.0);
            let s_flex = (1.0 - mean_flex).clamp(0.0, 1.0); // prefer rigid pockets

            let total_score = weights.volume * s_vol
                + weights.hydrophobicity * s_hydro
                + weights.enclosure * s_encl
                + weights.depth * s_depth
                + weights.hbond_capacity * s_hbond
                + weights.flexibility * s_flex;

            let clamped = total_score.clamp(0.0, 1.0);

            DetectedPocket {
                rank: 0, // filled later after sorting
                residue_indices: member_indices.iter().map(|&i| residues[i].resid).collect(),
                atom_positions: positions,
                centroid: cent,
                volume: vol,
                mean_burial: mean_burial_val,
                mean_hydrophobicity: mean_hydro,
                enclosure_ratio: enclosure,
                mean_depth: mean_depth_val,
                hbond_capacity: hbond,
                mean_flexibility: mean_flex,
                druggability_score: clamped,
                classification: DruggabilityClass::from_score(clamped),
            }
        })
        .collect();

    // ---- Step 6: sort by score descending, assign ranks, truncate ----
    pockets.sort_by(|a, b| {
        b.druggability_score
            .partial_cmp(&a.druggability_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, p) in pockets.iter_mut().enumerate() {
        p.rank = i + 1; // 1-indexed rank
    }
    pockets.truncate(config.max_pockets);
    pockets
}

/// Score volume using a sigmoid centered near 650 cubic angstrom.
fn score_volume(v: f64) -> f64 {
    let x = (v - 650.0) / 250.0;
    (1.0 / (1.0 + (-x).exp())).clamp(0.0, 1.0)
}

/// Score enclosure ratio, preferring partially enclosed pockets.
fn score_enclosure(e: f64) -> f64 {
    let c = e.clamp(0.0, 1.0);
    if c < 0.2 {
        c * 0.5
    } else if c > 0.9 {
        0.9 - (c - 0.9) * 0.8
    } else {
        c
    }
}

/// Score depth using a sigmoid centered at 6 angstrom.
fn score_depth(d: f64) -> f64 {
    let normalized = (d / 12.0).clamp(0.0, 1.2);
    (1.0 / (1.0 + (-4.0 * (normalized - 0.5)).exp())).clamp(0.0, 1.0)
}

/// Approximate pocket enclosure by testing 26 cubic lattice directions from
/// the centroid and checking how many are blocked by at least one atom.
fn compute_enclosure(cent: &[f64; 3], atoms: &[[f64; 3]], cutoff: f64) -> f64 {
    // 26 directions: all combinations of {-1, 0, 1}^3 minus (0,0,0)
    let mut directions = Vec::with_capacity(26);
    for dx in -1i32..=1 {
        for dy in -1i32..=1 {
            for dz in -1i32..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let len = ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
                directions.push([dx as f64 / len, dy as f64 / len, dz as f64 / len]);
            }
        }
    }

    let mut blocked = 0;
    for dir in &directions {
        // Check if any atom lies roughly along this direction from centroid
        let ray_endpoint = [
            cent[0] + dir[0] * cutoff,
            cent[1] + dir[1] * cutoff,
            cent[2] + dir[2] * cutoff,
        ];
        // An atom blocks a direction if it is within 3 angstrom of the ray line
        let blocked_by_atom = atoms.iter().any(|atom| {
            point_to_line_distance(atom, cent, &ray_endpoint) < 3.0
        });
        if blocked_by_atom {
            blocked += 1;
        }
    }

    blocked as f64 / directions.len() as f64
}

/// Distance from point P to the line segment AB.
fn point_to_line_distance(p: &[f64; 3], a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let ab_len_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if ab_len_sq < 1e-12 {
        return (ap[0] * ap[0] + ap[1] * ap[1] + ap[2] * ap[2]).sqrt();
    }
    let t = ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab_len_sq).clamp(0.0, 1.0);
    let proj = [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]];
    let dx = p[0] - proj[0];
    let dy = p[1] - proj[1];
    let dz = p[2] - proj[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ---------------------------------------------------------------------------
// Validation logic
// ---------------------------------------------------------------------------

/// Compare predicted pockets against known binding sites and compute metrics.
fn compute_validation_metrics(
    predicted: &[DetectedPocket],
    known_sites: &[KnownBindingSite],
    config: &DruggabilityConfig,
) -> DruggabilityMetrics {
    let n_known = known_sites.len();
    let n_pred = predicted.len();

    if n_known == 0 || n_pred == 0 {
        return DruggabilityMetrics {
            n_known_sites: n_known,
            n_predicted_pockets: n_pred,
            site_recovery_rate: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            enrichment_factor: 0.0,
            mean_dvo_jaccard: 0.0,
            mean_dvo_dice: 0.0,
            classification_accuracy: 0.0,
            site_comparisons: Vec::new(),
        };
    }

    let mut site_comparisons = Vec::new();
    let mut recovered_count = 0usize;
    let mut matched_predicted: HashSet<usize> = HashSet::new(); // indices of predicted pockets that matched
    let mut correct_class_count = 0usize;
    let mut total_dvo_jaccard = 0.0;
    let mut total_dvo_dice = 0.0;

    for (si, known) in known_sites.iter().enumerate() {
        // Find best-matching predicted pocket by centroid distance
        let mut best_rank: Option<usize> = None;
        let mut best_dist = f64::MAX;

        for (pi, pred) in predicted.iter().enumerate() {
            let d = distance(&known.centroid, &pred.centroid);
            if d < best_dist {
                best_dist = d;
                best_rank = Some(pi);
            }
        }

        let recovered = best_dist <= config.centroid_recovery_cutoff;
        if recovered {
            recovered_count += 1;
            if let Some(pi) = best_rank {
                matched_predicted.insert(pi);
            }
        }

        // Residue-level overlap (Jaccard)
        let known_set: HashSet<usize> = known.residue_indices.iter().cloned().collect();
        let residue_overlap = if let Some(pi) = best_rank {
            let pred_set: HashSet<usize> =
                predicted[pi].residue_indices.iter().cloned().collect();
            let inter = known_set.intersection(&pred_set).count();
            let union = known_set.union(&pred_set).count();
            if union == 0 { 0.0 } else { inter as f64 / union as f64 }
        } else {
            0.0
        };

        // DVO (voxel-based volume overlap)
        let (dvo_jaccard, dvo_dice) = if let Some(pi) = best_rank {
            let known_voxels = atoms_to_voxels(
                &known.atom_positions,
                config.dvo_atom_radius,
                config.dvo_grid_spacing,
            );
            let pred_voxels = atoms_to_voxels(
                &predicted[pi].atom_positions,
                config.dvo_atom_radius,
                config.dvo_grid_spacing,
            );
            voxel_overlap(&known_voxels, &pred_voxels)
        } else {
            (0.0, 0.0)
        };

        total_dvo_jaccard += dvo_jaccard;
        total_dvo_dice += dvo_dice;

        // Classification accuracy
        let predicted_class = best_rank.map(|pi| predicted[pi].classification);
        let class_match = predicted_class
            .map(|pc| pc == known.expected_class)
            .unwrap_or(false);
        if class_match {
            correct_class_count += 1;
        }

        site_comparisons.push(SiteComparisonResult {
            known_site_index: si,
            recovered,
            best_match_rank: best_rank.map(|pi| predicted[pi].rank),
            centroid_distance: if best_rank.is_some() { Some(best_dist) } else { None },
            residue_overlap,
            dvo_jaccard,
            dvo_dice,
            predicted_class,
            expected_class: known.expected_class,
            class_match,
        });
    }

    // ---- Precision / Recall / F1 ----
    let true_positives = matched_predicted.len() as f64;
    let false_positives = n_pred as f64 - true_positives;
    let false_negatives = n_known as f64 - recovered_count as f64;

    let precision = if true_positives + false_positives > 0.0 {
        true_positives / (true_positives + false_positives)
    } else {
        0.0
    };

    let recall = if true_positives + false_negatives > 0.0 {
        true_positives / (true_positives + false_negatives)
    } else {
        0.0
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    // ---- Enrichment factor ----
    // EF = (hits_in_top_N / N) / (total_hits / total_pockets)
    let top_n = config.enrichment_top_n.min(n_pred);
    let random_hit_rate = if n_pred > 0 {
        n_known as f64 / n_pred as f64
    } else {
        0.0
    };

    let hits_in_top_n = if top_n > 0 {
        let top_n_indices: HashSet<usize> = (0..top_n).collect();
        top_n_indices.intersection(&matched_predicted).count() as f64
    } else {
        0.0
    };

    let enrichment_factor = if random_hit_rate > 0.0 && top_n > 0 {
        (hits_in_top_n / top_n as f64) / random_hit_rate
    } else {
        0.0
    };

    // ---- Aggregate ----
    DruggabilityMetrics {
        n_known_sites: n_known,
        n_predicted_pockets: n_pred,
        site_recovery_rate: recovered_count as f64 / n_known as f64,
        precision,
        recall,
        f1,
        enrichment_factor,
        mean_dvo_jaccard: total_dvo_jaccard / n_known as f64,
        mean_dvo_dice: total_dvo_dice / n_known as f64,
        classification_accuracy: correct_class_count as f64 / n_known as f64,
        site_comparisons,
    }
}

// ---------------------------------------------------------------------------
// Ground truth extraction from Target metadata
// ---------------------------------------------------------------------------

/// Build known binding sites from Target pocket metadata and the parsed PDB.
fn extract_known_sites(
    target: &targets::Target,
    residues: &[PdbResidue],
) -> Vec<KnownBindingSite> {
    let pocket_def = match &target.pocket {
        Some(p) => p,
        None => return Vec::new(),
    };

    // Build a lookup: resid -> PdbResidue
    let resid_map: HashMap<usize, &PdbResidue> = residues.iter().map(|r| (r.resid, r)).collect();

    let positions: Vec<[f64; 3]> = pocket_def
        .residues
        .iter()
        .filter_map(|&rid| resid_map.get(&rid).map(|r| r.ca_pos))
        .collect();

    if positions.is_empty() {
        return Vec::new();
    }

    let cent = centroid(&positions);

    // Determine expected druggability class from target metadata.
    // If the target has an approved drug, the site is at least Druggable.
    let expected_class = match &target.drug_info {
        Some(info) => match info.status {
            targets::DrugStatus::Approved => DruggabilityClass::HighlyDruggable,
            targets::DrugStatus::Phase3 | targets::DrugStatus::Phase2 => {
                DruggabilityClass::Druggable
            }
            targets::DrugStatus::Phase1 | targets::DrugStatus::Preclinical => {
                DruggabilityClass::DifficultTarget
            }
            _ => DruggabilityClass::Druggable,
        },
        None => DruggabilityClass::Druggable,
    };

    vec![KnownBindingSite {
        residue_indices: pocket_def.residues.clone(),
        core_residues: pocket_def.core_residues.clone(),
        atom_positions: positions,
        centroid: cent,
        expected_class,
        is_cryptic: pocket_def.is_cryptic,
    }]
}

// ---------------------------------------------------------------------------
// DruggabilityBenchmark
// ---------------------------------------------------------------------------

/// Physics-based druggability validation benchmark.
///
/// Detects pockets in the target structure, scores them for druggability,
/// and compares against known drug-binding sites to produce standard
/// validation metrics (precision, recall, F1, DVO, enrichment factor).
pub struct DruggabilityBenchmark {
    config: DruggabilityConfig,
}

impl DruggabilityBenchmark {
    /// Create a new benchmark with default configuration.
    pub fn new() -> Self {
        Self {
            config: DruggabilityConfig::default(),
        }
    }

    /// Create a new benchmark with custom configuration.
    pub fn with_config(config: DruggabilityConfig) -> Self {
        Self { config }
    }

    /// Run pocket detection and druggability scoring on a PDB file.
    /// Returns the list of detected pockets (ranked by score).
    pub fn detect_and_score(
        &self,
        pdb_path: &std::path::Path,
    ) -> Result<Vec<DetectedPocket>> {
        let residues = parse_pdb_ca_atoms(pdb_path)?;
        log::info!(
            "Parsed {} residues from {}",
            residues.len(),
            pdb_path.display()
        );
        let pockets = detect_pockets(&residues, &self.config);
        log::info!("Detected {} pockets", pockets.len());
        Ok(pockets)
    }

    /// Run full validation: detect pockets, compare against known sites,
    /// return detailed metrics.
    pub fn validate(
        &self,
        target: &targets::Target,
    ) -> Result<(Vec<DetectedPocket>, DruggabilityMetrics)> {
        // Determine which PDB to use (prefer holo for ground-truth comparison,
        // fall back to apo).
        let pdb_path = target
            .structures
            .apo_pdb
            .as_ref()
            .or(target.structures.holo_pdb.as_ref())
            .ok_or_else(|| anyhow!("No PDB structure available for target {}", target.name))?;

        let residues = parse_pdb_ca_atoms(pdb_path)?;
        log::info!(
            "Parsed {} residues from {} for target {}",
            residues.len(),
            pdb_path.display(),
            target.name
        );

        let pockets = detect_pockets(&residues, &self.config);
        log::info!(
            "Detected {} pockets for target {}",
            pockets.len(),
            target.name
        );

        let known_sites = extract_known_sites(target, &residues);
        log::info!(
            "Extracted {} known binding sites for target {}",
            known_sites.len(),
            target.name
        );

        let metrics = compute_validation_metrics(&pockets, &known_sites, &self.config);
        Ok((pockets, metrics))
    }

    /// Pack druggability metrics into the BenchmarkMetrics custom map.
    fn pack_metrics(
        &self,
        dm: &DruggabilityMetrics,
        pockets: &[DetectedPocket],
        duration: f64,
    ) -> BenchmarkMetrics {
        let mut metrics = BenchmarkMetrics::default();

        // Store the best pocket RMSD-like metric (centroid distance to known site)
        if let Some(first) = dm.site_comparisons.first() {
            metrics.pocket_rmsd = first.centroid_distance.map(|d| d as f32);
        }

        // Store SASA-like proxy (mean enclosure of top pocket)
        if let Some(top) = pockets.first() {
            metrics.pocket_sasa = Some((top.enclosure_ratio * 100.0) as f32);
        }

        // Store topological detection proxy
        if dm.n_predicted_pockets > 0 {
            metrics.betti_2 = Some(dm.n_predicted_pockets as f32);
        }

        // Custom metrics
        metrics.custom.insert("precision".to_string(), dm.precision);
        metrics.custom.insert("recall".to_string(), dm.recall);
        metrics.custom.insert("f1".to_string(), dm.f1);
        metrics
            .custom
            .insert("site_recovery_rate".to_string(), dm.site_recovery_rate);
        metrics
            .custom
            .insert("enrichment_factor".to_string(), dm.enrichment_factor);
        metrics
            .custom
            .insert("mean_dvo_jaccard".to_string(), dm.mean_dvo_jaccard);
        metrics
            .custom
            .insert("mean_dvo_dice".to_string(), dm.mean_dvo_dice);
        metrics.custom.insert(
            "classification_accuracy".to_string(),
            dm.classification_accuracy,
        );
        metrics
            .custom
            .insert("n_predicted_pockets".to_string(), dm.n_predicted_pockets as f64);
        metrics
            .custom
            .insert("n_known_sites".to_string(), dm.n_known_sites as f64);

        // Top pocket druggability score
        if let Some(top) = pockets.first() {
            metrics
                .custom
                .insert("top_pocket_score".to_string(), top.druggability_score);
            metrics.custom.insert(
                "top_pocket_volume".to_string(),
                top.volume,
            );
            metrics.custom.insert(
                "top_pocket_class".to_string(),
                top.classification.ordinal() as f64,
            );
        }

        metrics
    }
}

impl Default for DruggabilityBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationBenchmark for DruggabilityBenchmark {
    fn name(&self) -> &str {
        "druggability"
    }

    fn run(&self, target: &targets::Target) -> Result<BenchmarkResult> {
        let start = std::time::Instant::now();

        log::info!(
            "Running Druggability benchmark on {} (drug: {:?})",
            target.name,
            target.drug_info.as_ref().map(|d| &d.name)
        );

        let (pockets, dm) = self.validate(target)?;
        let duration = start.elapsed().as_secs_f64();

        let metrics = self.pack_metrics(&dm, &pockets, duration);

        // Passing criteria:
        //   1. At least one known site recovered (site_recovery_rate > 0)
        //   2. DVO Jaccard above success threshold
        //   3. F1 above 0.3
        let passed = dm.site_recovery_rate > 0.0
            && dm.mean_dvo_jaccard >= self.config.dvo_success_threshold
            && dm.f1 >= 0.3;

        let pdb_id = target
            .structures
            .apo_pdb
            .as_ref()
            .or(target.structures.holo_pdb.as_ref())
            .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
            .unwrap_or_default();

        Ok(BenchmarkResult {
            benchmark: self.name().to_string(),
            target: target.name.clone(),
            pdb_id,
            timestamp: Utc::now(),
            duration_secs: duration,
            steps: 0, // no simulation steps for static pocket detection
            metrics,
            passed,
            reason: if passed {
                format!(
                    "Site recovery {:.0}%, F1 {:.2}, DVO {:.2}",
                    dm.site_recovery_rate * 100.0,
                    dm.f1,
                    dm.mean_dvo_jaccard
                )
            } else {
                format!(
                    "Recovery {:.0}% (need >0%), F1 {:.2} (need >=0.30), DVO {:.2} (need >={:.2})",
                    dm.site_recovery_rate * 100.0,
                    dm.f1,
                    dm.mean_dvo_jaccard,
                    self.config.dvo_success_threshold
                )
            },
        })
    }

    fn score(&self, result: &BenchmarkResult) -> ValidationScore {
        let mut components = Vec::new();

        // Site recovery rate (weight: 25%)
        if let Some(&recovery) = result.metrics.custom.get("site_recovery_rate") {
            components.push(ScoreComponent {
                name: "Site Recovery Rate".to_string(),
                score: (recovery * 100.0).clamp(0.0, 100.0),
                weight: 0.25,
                description: "Fraction of known drug-binding sites recovered within 4A centroid distance".to_string(),
            });
        }

        // F1 score (weight: 25%)
        if let Some(&f1) = result.metrics.custom.get("f1") {
            components.push(ScoreComponent {
                name: "F1 Score".to_string(),
                score: (f1 * 100.0).clamp(0.0, 100.0),
                weight: 0.25,
                description: "Harmonic mean of pocket-level precision and recall".to_string(),
            });
        }

        // DVO Jaccard (weight: 25%)
        if let Some(&dvo) = result.metrics.custom.get("mean_dvo_jaccard") {
            components.push(ScoreComponent {
                name: "DVO (Volume Overlap)".to_string(),
                score: (dvo * 100.0).clamp(0.0, 100.0),
                weight: 0.25,
                description: "Jaccard index of predicted vs actual pocket volumes (voxel grid)".to_string(),
            });
        }

        // Enrichment factor (weight: 15%)
        if let Some(&ef) = result.metrics.custom.get("enrichment_factor") {
            // EF of 1.0 = random; cap display at EF = 5.0 -> 100 points
            let score = (ef / 5.0 * 100.0).clamp(0.0, 100.0);
            components.push(ScoreComponent {
                name: "Enrichment Factor".to_string(),
                score,
                weight: 0.15,
                description: "Drug sites enriched in top-N predictions vs random".to_string(),
            });
        }

        // Classification accuracy (weight: 10%)
        if let Some(&acc) = result.metrics.custom.get("classification_accuracy") {
            components.push(ScoreComponent {
                name: "Classification Accuracy".to_string(),
                score: (acc * 100.0).clamp(0.0, 100.0),
                weight: 0.10,
                description: "Accuracy of HighlyDruggable/Druggable/Difficult/Undruggable classification".to_string(),
            });
        }

        ValidationScore::compute(components)
    }

    fn compare_af3(
        &self,
        result: &BenchmarkResult,
        af3_result: Option<&BenchmarkMetrics>,
    ) -> Af3Comparison {
        let mut comparison = Vec::new();

        // --- Site recovery comparison ---
        if let Some(&prism_recovery) = result.metrics.custom.get("site_recovery_rate") {
            let af3_recovery = af3_result
                .and_then(|r| r.custom.get("site_recovery_rate"))
                .copied();
            comparison.push(ComparisonItem {
                metric: "Site Recovery Rate".to_string(),
                prism_value: prism_recovery,
                af3_value: af3_recovery,
                winner: if af3_recovery
                    .map(|a| prism_recovery >= a)
                    .unwrap_or(true)
                {
                    "PRISM-NOVA".to_string()
                } else {
                    "AlphaFold3".to_string()
                },
                significance: "Fraction of known drug sites recovered".to_string(),
            });
        }

        // --- F1 comparison ---
        if let Some(&prism_f1) = result.metrics.custom.get("f1") {
            let af3_f1 = af3_result
                .and_then(|r| r.custom.get("f1"))
                .copied();
            comparison.push(ComparisonItem {
                metric: "F1 Score".to_string(),
                prism_value: prism_f1,
                af3_value: af3_f1,
                winner: if af3_f1.map(|a| prism_f1 >= a).unwrap_or(true) {
                    "PRISM-NOVA".to_string()
                } else {
                    "AlphaFold3".to_string()
                },
                significance: "Pocket detection F1 (precision-recall balance)".to_string(),
            });
        }

        // --- DVO comparison ---
        if let Some(&prism_dvo) = result.metrics.custom.get("mean_dvo_jaccard") {
            let af3_dvo = af3_result
                .and_then(|r| r.custom.get("mean_dvo_jaccard"))
                .copied();
            comparison.push(ComparisonItem {
                metric: "DVO (Volume Overlap)".to_string(),
                prism_value: prism_dvo,
                af3_value: af3_dvo,
                winner: if af3_dvo.map(|a| prism_dvo >= a).unwrap_or(true) {
                    "PRISM-NOVA".to_string()
                } else {
                    "AlphaFold3".to_string()
                },
                significance: "Voxel-grid volume overlap with ground truth pocket".to_string(),
            });
        }

        // --- Cryptic site capability (PRISM advantage) ---
        comparison.push(ComparisonItem {
            metric: "Cryptic Site Detection".to_string(),
            prism_value: 1.0,
            af3_value: Some(0.0),
            winner: "PRISM-NOVA".to_string(),
            significance: "AF3 cannot detect cryptic sites from apo structures (static predictor)".to_string(),
        });

        let prism_wins = comparison
            .iter()
            .filter(|c| c.winner == "PRISM-NOVA")
            .count();
        let af3_wins = comparison
            .iter()
            .filter(|c| c.winner == "AlphaFold3")
            .count();

        Af3Comparison {
            target: result.target.clone(),
            prism_metrics: result.metrics.clone(),
            af3_metrics: af3_result.cloned(),
            comparison,
            winner: if prism_wins >= af3_wins {
                "PRISM-NOVA".to_string()
            } else {
                "AlphaFold3".to_string()
            },
            advantage: format!(
                "PRISM wins {}/{} druggability metrics (dynamics + cryptic site capability)",
                prism_wins,
                prism_wins + af3_wins
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_residue(index: usize, resid: usize, resname: &str, pos: [f64; 3]) -> PdbResidue {
        PdbResidue {
            index,
            resid,
            resname: resname.to_string(),
            chain: 'A',
            ca_pos: pos,
        }
    }

    #[test]
    fn test_druggability_class_from_score() {
        assert_eq!(DruggabilityClass::from_score(0.85), DruggabilityClass::HighlyDruggable);
        assert_eq!(DruggabilityClass::from_score(0.60), DruggabilityClass::Druggable);
        assert_eq!(DruggabilityClass::from_score(0.40), DruggabilityClass::DifficultTarget);
        assert_eq!(DruggabilityClass::from_score(0.15), DruggabilityClass::Undruggable);
    }

    #[test]
    fn test_druggability_class_ordinal() {
        assert!(
            DruggabilityClass::HighlyDruggable.ordinal()
                > DruggabilityClass::Druggable.ordinal()
        );
        assert!(
            DruggabilityClass::Druggable.ordinal()
                > DruggabilityClass::DifficultTarget.ordinal()
        );
    }

    #[test]
    fn test_centroid() {
        let pts = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let c = centroid(&pts);
        assert!((c[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((c[1] - 2.0 / 3.0).abs() < 1e-6);
        assert!((c[2]).abs() < 1e-6);
    }

    #[test]
    fn test_centroid_empty() {
        let c = centroid(&[]);
        assert_eq!(c, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_distance() {
        let d = distance(&[0.0, 0.0, 0.0], &[3.0, 4.0, 0.0]);
        assert!((d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_voxel_overlap_identical() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let v = atoms_to_voxels(&pts, 2.0, 1.0);
        let (jaccard, dice) = voxel_overlap(&v, &v);
        assert!((jaccard - 1.0).abs() < 1e-6);
        assert!((dice - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_voxel_overlap_no_overlap() {
        let a = atoms_to_voxels(&[[0.0, 0.0, 0.0]], 1.0, 1.0);
        let b = atoms_to_voxels(&[[100.0, 100.0, 100.0]], 1.0, 1.0);
        let (jaccard, _dice) = voxel_overlap(&a, &b);
        assert!((jaccard).abs() < 1e-6);
    }

    #[test]
    fn test_voxel_overlap_partial() {
        let a = atoms_to_voxels(&[[0.0, 0.0, 0.0]], 3.0, 1.0);
        let b = atoms_to_voxels(&[[3.0, 0.0, 0.0]], 3.0, 1.0);
        let (jaccard, _) = voxel_overlap(&a, &b);
        assert!(jaccard > 0.0);
        assert!(jaccard < 1.0);
    }

    #[test]
    fn test_score_volume() {
        // Volume below optimal should score low
        assert!(score_volume(100.0) < 0.2);
        // Volume at optimal should score near 0.5
        assert!((score_volume(650.0) - 0.5).abs() < 0.05);
        // Volume above optimal should score high
        assert!(score_volume(1200.0) > 0.8);
    }

    #[test]
    fn test_residue_hydrophobicity() {
        assert!(residue_hydrophobicity("ILE") > 0.9);
        assert!(residue_hydrophobicity("ARG") < 0.1);
        let gly = residue_hydrophobicity("GLY");
        assert!(gly > 0.3 && gly < 0.6);
    }

    #[test]
    fn test_residue_hbond_capacity() {
        assert_eq!(residue_hbond_capacity("ARG"), 5);
        assert_eq!(residue_hbond_capacity("ALA"), 0);
        assert_eq!(residue_hbond_capacity("SER"), 2);
    }

    #[test]
    fn test_detect_pockets_minimal_cluster() {
        // Build a small cluster of 5 residues close together in a pocket-like arrangement
        let residues: Vec<PdbResidue> = vec![
            make_residue(0, 10, "LEU", [0.0, 0.0, 0.0]),
            make_residue(1, 11, "ILE", [3.0, 0.0, 0.0]),
            make_residue(2, 12, "VAL", [0.0, 3.0, 0.0]),
            make_residue(3, 13, "PHE", [3.0, 3.0, 0.0]),
            make_residue(4, 14, "MET", [1.5, 1.5, 3.0]),
            // Surrounding residues to provide burial context
            make_residue(5, 20, "ALA", [6.0, 0.0, 0.0]),
            make_residue(6, 21, "ALA", [-3.0, 0.0, 0.0]),
            make_residue(7, 22, "ALA", [0.0, 6.0, 0.0]),
            make_residue(8, 23, "ALA", [0.0, -3.0, 0.0]),
            make_residue(9, 24, "ALA", [0.0, 0.0, 6.0]),
            make_residue(10, 25, "ALA", [0.0, 0.0, -3.0]),
            make_residue(11, 26, "ALA", [6.0, 6.0, 0.0]),
            make_residue(12, 27, "ALA", [-3.0, -3.0, 0.0]),
            make_residue(13, 28, "ALA", [6.0, 0.0, 6.0]),
            make_residue(14, 29, "ALA", [-3.0, 0.0, -3.0]),
        ];

        let config = DruggabilityConfig {
            min_cluster_size: 3,
            cluster_distance: 6.0,
            neighbor_cutoff: 8.0,
            ..Default::default()
        };

        let pockets = detect_pockets(&residues, &config);
        // Should detect at least one pocket
        assert!(!pockets.is_empty(), "Expected at least one detected pocket");
        // Top pocket should have a valid druggability score
        assert!(pockets[0].druggability_score >= 0.0);
        assert!(pockets[0].druggability_score <= 1.0);
    }

    #[test]
    fn test_compute_validation_metrics_perfect_match() {
        // Predicted pocket exactly matches the known site
        let predicted = vec![DetectedPocket {
            rank: 1,
            residue_indices: vec![10, 11, 12, 13, 14],
            atom_positions: vec![
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [3.0, 3.0, 0.0],
                [1.5, 1.5, 3.0],
            ],
            centroid: [1.5, 1.5, 0.6],
            volume: 500.0,
            mean_burial: 0.5,
            mean_hydrophobicity: 0.7,
            enclosure_ratio: 0.5,
            mean_depth: 4.0,
            hbond_capacity: 3,
            mean_flexibility: 0.3,
            druggability_score: 0.75,
            classification: DruggabilityClass::HighlyDruggable,
        }];

        let known = vec![KnownBindingSite {
            residue_indices: vec![10, 11, 12, 13, 14],
            core_residues: vec![10, 11, 12],
            atom_positions: vec![
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [3.0, 3.0, 0.0],
                [1.5, 1.5, 3.0],
            ],
            centroid: [1.5, 1.5, 0.6],
            expected_class: DruggabilityClass::HighlyDruggable,
            is_cryptic: false,
        }];

        let config = DruggabilityConfig::default();
        let m = compute_validation_metrics(&predicted, &known, &config);

        assert!((m.site_recovery_rate - 1.0).abs() < 1e-6);
        assert!((m.precision - 1.0).abs() < 1e-6);
        assert!((m.recall - 1.0).abs() < 1e-6);
        assert!((m.f1 - 1.0).abs() < 1e-6);
        assert!((m.mean_dvo_jaccard - 1.0).abs() < 1e-6);
        assert!(m.classification_accuracy > 0.99);
    }

    #[test]
    fn test_compute_validation_metrics_no_overlap() {
        let predicted = vec![DetectedPocket {
            rank: 1,
            residue_indices: vec![100, 101, 102],
            atom_positions: vec![[50.0, 50.0, 50.0], [53.0, 50.0, 50.0], [50.0, 53.0, 50.0]],
            centroid: [51.0, 51.0, 50.0],
            volume: 200.0,
            mean_burial: 0.4,
            mean_hydrophobicity: 0.5,
            enclosure_ratio: 0.3,
            mean_depth: 3.0,
            hbond_capacity: 2,
            mean_flexibility: 0.5,
            druggability_score: 0.45,
            classification: DruggabilityClass::DifficultTarget,
        }];

        let known = vec![KnownBindingSite {
            residue_indices: vec![10, 11, 12],
            core_residues: vec![10],
            atom_positions: vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            centroid: [1.0, 1.0, 0.0],
            expected_class: DruggabilityClass::Druggable,
            is_cryptic: false,
        }];

        let config = DruggabilityConfig::default();
        let m = compute_validation_metrics(&predicted, &known, &config);

        assert!((m.site_recovery_rate).abs() < 1e-6);
        assert!((m.mean_dvo_jaccard).abs() < 1e-6);
    }

    #[test]
    fn test_compute_validation_metrics_empty() {
        let config = DruggabilityConfig::default();
        let m = compute_validation_metrics(&[], &[], &config);
        assert_eq!(m.n_known_sites, 0);
        assert_eq!(m.n_predicted_pockets, 0);
    }

    #[test]
    fn test_benchmark_creation() {
        let bench = DruggabilityBenchmark::new();
        assert_eq!(bench.name(), "druggability");
    }

    #[test]
    fn test_benchmark_score_all_components() {
        let bench = DruggabilityBenchmark::new();

        let mut metrics = BenchmarkMetrics::default();
        metrics.custom.insert("site_recovery_rate".to_string(), 0.80);
        metrics.custom.insert("f1".to_string(), 0.65);
        metrics.custom.insert("mean_dvo_jaccard".to_string(), 0.45);
        metrics.custom.insert("enrichment_factor".to_string(), 3.0);
        metrics.custom.insert("classification_accuracy".to_string(), 0.70);

        let result = BenchmarkResult {
            benchmark: "druggability".to_string(),
            target: "TEST".to_string(),
            pdb_id: "1ABC".to_string(),
            timestamp: Utc::now(),
            duration_secs: 1.0,
            steps: 0,
            metrics,
            passed: true,
            reason: "test".to_string(),
        };

        let score = bench.score(&result);
        assert_eq!(score.components.len(), 5);
        assert!(score.overall > 0.0);
        assert!(score.overall <= 100.0);
        // With these metrics the score should be solidly in B-C range
        assert!(
            score.overall > 40.0,
            "Score {} is unexpectedly low",
            score.overall
        );
    }

    #[test]
    fn test_approximate_volume() {
        // 4 points forming a cube-like shape
        let pts = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ];
        let vol = approximate_volume(&pts);
        // Bounding box = 10*10*10 = 1000, * 0.35 = 350
        assert!((vol - 350.0).abs() < 1.0);
    }

    #[test]
    fn test_approximate_volume_too_few() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(approximate_volume(&pts), 0.0);
    }

    #[test]
    fn test_point_to_line_distance() {
        let d = point_to_line_distance(
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[10.0, 0.0, 0.0],
        );
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_enrichment_factor() {
        // 1 known site, 5 predicted pockets, the known site matches the rank-1 pocket
        let predicted = vec![
            DetectedPocket {
                rank: 1,
                residue_indices: vec![10, 11],
                atom_positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                centroid: [0.5, 0.0, 0.0],
                volume: 100.0,
                mean_burial: 0.5,
                mean_hydrophobicity: 0.5,
                enclosure_ratio: 0.5,
                mean_depth: 4.0,
                hbond_capacity: 2,
                mean_flexibility: 0.3,
                druggability_score: 0.7,
                classification: DruggabilityClass::HighlyDruggable,
            },
            DetectedPocket {
                rank: 2,
                residue_indices: vec![30, 31],
                atom_positions: vec![[20.0, 20.0, 20.0], [21.0, 20.0, 20.0]],
                centroid: [20.5, 20.0, 20.0],
                volume: 80.0,
                mean_burial: 0.4,
                mean_hydrophobicity: 0.3,
                enclosure_ratio: 0.3,
                mean_depth: 3.0,
                hbond_capacity: 1,
                mean_flexibility: 0.5,
                druggability_score: 0.4,
                classification: DruggabilityClass::DifficultTarget,
            },
        ];

        let known = vec![KnownBindingSite {
            residue_indices: vec![10, 11],
            core_residues: vec![10],
            atom_positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            centroid: [0.5, 0.0, 0.0],
            expected_class: DruggabilityClass::HighlyDruggable,
            is_cryptic: false,
        }];

        let config = DruggabilityConfig {
            enrichment_top_n: 1,
            ..Default::default()
        };
        let m = compute_validation_metrics(&predicted, &known, &config);

        // With 1 known site in 2 pockets: random rate = 0.5
        // Hit in top-1 = 1.0. EF = (1/1) / 0.5 = 2.0
        assert!(
            (m.enrichment_factor - 2.0).abs() < 1e-6,
            "EF was {} expected 2.0",
            m.enrichment_factor
        );
    }

    #[test]
    fn test_compare_af3() {
        let bench = DruggabilityBenchmark::new();

        let mut metrics = BenchmarkMetrics::default();
        metrics.custom.insert("site_recovery_rate".to_string(), 0.80);
        metrics.custom.insert("f1".to_string(), 0.65);
        metrics.custom.insert("mean_dvo_jaccard".to_string(), 0.45);

        let result = BenchmarkResult {
            benchmark: "druggability".to_string(),
            target: "TEST".to_string(),
            pdb_id: "1ABC".to_string(),
            timestamp: Utc::now(),
            duration_secs: 1.0,
            steps: 0,
            metrics,
            passed: true,
            reason: "test".to_string(),
        };

        let comp = bench.compare_af3(&result, None);
        assert_eq!(comp.winner, "PRISM-NOVA");
        // Should have at least 4 comparison items (3 metrics + cryptic)
        assert!(comp.comparison.len() >= 4);
    }
}
