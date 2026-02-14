//! PocketMiner Benchmark Dataset Handler
//!
//! Loads and manages the PocketMiner 39-pocket benchmark dataset
//! for cryptic site detector validation.
//!
//! ## Dataset Structure
//!
//! ```text
//! data/benchmarks/pocketminer/
//! ├── manifest.json           # Dataset manifest with ground truth
//! ├── apo/                    # Apo structures (no ligand)
//! │   ├── 1ex6_apo.pdb
//! │   └── ...
//! └── holo/                   # Holo structures (with ligand)
//!     ├── 1ex6_holo.pdb
//!     └── ...
//! ```
//!
//! ## Reference
//! PocketMiner: Nature Communications 2023
//! https://github.com/Mickdub/gvp/tree/pocket_pred

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Classification of cryptic pocket mechanism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PocketType {
    /// Loop or coil movement
    LoopShift,
    /// Secondary structure unwinding
    HelixUnwinding,
    /// Large domain motion
    DomainMotion,
    /// Sidechain rotamer change
    SidechainRotamer,
    /// Unknown or mixed mechanism
    Unknown,
}

impl Default for PocketType {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Single entry in the PocketMiner dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketMinerEntry {
    /// PDB ID (e.g., "1ex6")
    pub pdb_id: String,
    /// Path to apo structure (no ligand)
    pub apo_path: PathBuf,
    /// Path to holo structure (with ligand)
    pub holo_path: PathBuf,
    /// Ground truth cryptic residue IDs
    pub cryptic_residues: Vec<i32>,
    /// Chain IDs for multi-chain proteins
    pub chain_ids: Vec<String>,
    /// Ligand atom coordinates (from holo structure)
    pub ligand_coords: Vec<[f64; 3]>,
    /// Centroid of cryptic pocket
    pub pocket_centroid: [f64; 3],
    /// Mechanism classification
    #[serde(default)]
    pub pocket_type: PocketType,
    /// Ligand name/ID if known
    pub ligand_id: Option<String>,
    /// Number of pocket residues within 4.5Å of ligand
    pub n_pocket_residues: usize,
}

impl PocketMinerEntry {
    /// Get set of cryptic residue IDs for evaluation
    pub fn ground_truth_set(&self) -> HashSet<i32> {
        self.cryptic_residues.iter().cloned().collect()
    }

    /// Check if structure exists on disk
    pub fn validate_paths(&self) -> Result<()> {
        if !self.apo_path.exists() {
            return Err(anyhow!("Apo structure not found: {:?}", self.apo_path));
        }
        if !self.holo_path.exists() {
            return Err(anyhow!("Holo structure not found: {:?}", self.holo_path));
        }
        Ok(())
    }
}

/// Full PocketMiner benchmark dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketMinerDataset {
    /// Dataset version/source
    pub version: String,
    /// Source reference
    pub reference: String,
    /// Base directory for PDB files
    #[serde(skip)]
    pub base_dir: PathBuf,
    /// All benchmark entries
    pub entries: Vec<PocketMinerEntry>,
    /// Number of unique apo structures
    pub n_structures: usize,
    /// Total number of cryptic pockets (may be >n_structures for multi-pocket proteins)
    pub n_pockets: usize,
}

impl PocketMinerDataset {
    /// Load dataset from manifest file
    pub fn load(manifest_path: &Path) -> Result<Self> {
        if !manifest_path.exists() {
            return Err(anyhow!("Manifest not found: {:?}", manifest_path));
        }

        let content = fs::read_to_string(manifest_path)?;
        let mut dataset: PocketMinerDataset = serde_json::from_str(&content)?;

        // Set base directory from manifest location
        dataset.base_dir = manifest_path
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();

        // Update counts
        dataset.n_structures = dataset.entries.len();
        dataset.n_pockets = dataset.entries.iter().map(|e| e.cryptic_residues.len().max(1)).sum();

        Ok(dataset)
    }

    /// Create empty dataset (for building manually)
    pub fn new(base_dir: &Path) -> Self {
        Self {
            version: "1.0".to_string(),
            reference: "PocketMiner Nature Communications 2023".to_string(),
            base_dir: base_dir.to_path_buf(),
            entries: Vec::new(),
            n_structures: 0,
            n_pockets: 0,
        }
    }

    /// Add an entry to the dataset
    pub fn add_entry(&mut self, entry: PocketMinerEntry) {
        self.entries.push(entry);
        self.n_structures = self.entries.len();
        self.n_pockets = self.entries.iter().map(|e| e.cryptic_residues.len().max(1)).sum();
    }

    /// Get all PDB IDs in the dataset
    pub fn pdb_ids(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.pdb_id.clone()).collect()
    }

    /// Iterator over apo structures
    pub fn iter_apo_structures(&self) -> impl Iterator<Item = &PocketMinerEntry> {
        self.entries.iter()
    }

    /// Get entry by PDB ID
    pub fn get_entry(&self, pdb_id: &str) -> Option<&PocketMinerEntry> {
        let normalized = pdb_id.to_lowercase();
        self.entries.iter().find(|e| e.pdb_id.to_lowercase() == normalized)
    }

    /// Validate all entries exist on disk
    pub fn validate(&self) -> Result<Vec<String>> {
        let mut missing = Vec::new();

        for entry in &self.entries {
            if let Err(e) = entry.validate_paths() {
                missing.push(format!("{}: {}", entry.pdb_id, e));
            }
        }

        if missing.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(missing)
        }
    }

    /// Save dataset manifest
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Filter entries by pocket type
    pub fn filter_by_type(&self, pocket_type: PocketType) -> Vec<&PocketMinerEntry> {
        self.entries
            .iter()
            .filter(|e| e.pocket_type == pocket_type)
            .collect()
    }

    /// Get summary statistics
    pub fn summary(&self) -> DatasetSummary {
        let mut by_type = std::collections::HashMap::new();
        for entry in &self.entries {
            *by_type.entry(entry.pocket_type).or_insert(0usize) += 1;
        }

        let mean_residues = if self.entries.is_empty() {
            0.0
        } else {
            self.entries.iter().map(|e| e.n_pocket_residues).sum::<usize>() as f64
                / self.entries.len() as f64
        };

        DatasetSummary {
            n_structures: self.n_structures,
            n_pockets: self.n_pockets,
            by_mechanism: by_type,
            mean_pocket_residues: mean_residues,
        }
    }
}

/// Dataset summary statistics
#[derive(Debug, Clone, Serialize)]
pub struct DatasetSummary {
    pub n_structures: usize,
    pub n_pockets: usize,
    pub by_mechanism: std::collections::HashMap<PocketType, usize>,
    pub mean_pocket_residues: f64,
}

/// Extract ligand coordinates from a holo PDB file
///
/// Looks for HETATM records that are likely ligands (excluding water, ions)
pub fn extract_ligand_coords(pdb_path: &Path) -> Result<Vec<[f64; 3]>> {
    let content = fs::read_to_string(pdb_path)?;
    let mut coords = Vec::new();

    // Common non-ligand HETATM residue names to skip
    let skip_residues: HashSet<&str> = [
        "HOH", "WAT", "H2O", "DOD", "D2O",  // Water
        "NA", "CL", "K", "CA", "MG", "ZN", "FE", "CU", "MN",  // Ions
        "SO4", "PO4", "ACT", "GOL", "EDO", "PEG",  // Common crystallization additives
    ].into_iter().collect();

    for line in content.lines() {
        if line.starts_with("HETATM") && line.len() >= 54 {
            let res_name = line[17..20].trim();

            if skip_residues.contains(res_name) {
                continue;
            }

            // Parse coordinates
            let x: f64 = line[30..38].trim().parse().unwrap_or(0.0);
            let y: f64 = line[38..46].trim().parse().unwrap_or(0.0);
            let z: f64 = line[46..54].trim().parse().unwrap_or(0.0);

            coords.push([x, y, z]);
        }
    }

    Ok(coords)
}

/// Extract cryptic residues from apo/holo comparison
///
/// **IMPORTANT**: This function requires the structure and ligand to be in
/// the same coordinate frame. For PocketMiner data, use `extract_cryptic_residues_from_holo`
/// instead, which extracts residues from the HOLO structure where ligand coordinates match.
///
/// Residues within `cutoff` Å of ligand atoms are considered part of the cryptic binding site.
pub fn extract_cryptic_residues(
    pdb_path: &Path,
    ligand_coords: &[[f64; 3]],
    cutoff: f64,
) -> Result<Vec<i32>> {
    let content = fs::read_to_string(pdb_path)?;
    let cutoff_sq = cutoff * cutoff;

    let mut cryptic_residues: HashSet<i32> = HashSet::new();

    for line in content.lines() {
        if line.starts_with("ATOM") && line.len() >= 54 {
            // Parse residue number
            let res_seq: i32 = line[22..26].trim().parse().unwrap_or(0);

            // Parse coordinates
            let x: f64 = line[30..38].trim().parse().unwrap_or(0.0);
            let y: f64 = line[38..46].trim().parse().unwrap_or(0.0);
            let z: f64 = line[46..54].trim().parse().unwrap_or(0.0);

            // Check distance to any ligand atom
            for lig_coord in ligand_coords {
                let dx = x - lig_coord[0];
                let dy = y - lig_coord[1];
                let dz = z - lig_coord[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    cryptic_residues.insert(res_seq);
                    break;
                }
            }
        }
    }

    let mut sorted: Vec<i32> = cryptic_residues.into_iter().collect();
    sorted.sort();
    Ok(sorted)
}

/// Extract cryptic residues from HOLO structure
///
/// This is the correct function for PocketMiner benchmark:
/// - Ligand coordinates are in HOLO coordinate frame
/// - We find protein residues in HOLO that are near the ligand
/// - These residue IDs are used as ground truth (same residue numbering as APO)
///
/// The assumption is that APO and HOLO have the same residue numbering.
pub fn extract_cryptic_residues_from_holo(
    holo_path: &Path,
    cutoff: f64,
) -> Result<(Vec<i32>, Vec<[f64; 3]>)> {
    let content = fs::read_to_string(holo_path)?;
    let cutoff_sq = cutoff * cutoff;

    // First extract ligand coordinates
    let ligand_coords = extract_ligand_coords(holo_path)?;
    if ligand_coords.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut cryptic_residues: HashSet<i32> = HashSet::new();

    for line in content.lines() {
        if line.starts_with("ATOM") && line.len() >= 54 {
            // Parse residue number
            let res_seq: i32 = line[22..26].trim().parse().unwrap_or(0);

            // Parse coordinates
            let x: f64 = line[30..38].trim().parse().unwrap_or(0.0);
            let y: f64 = line[38..46].trim().parse().unwrap_or(0.0);
            let z: f64 = line[46..54].trim().parse().unwrap_or(0.0);

            // Check distance to any ligand atom
            for lig_coord in &ligand_coords {
                let dx = x - lig_coord[0];
                let dy = y - lig_coord[1];
                let dz = z - lig_coord[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    cryptic_residues.insert(res_seq);
                    break;
                }
            }
        }
    }

    let mut sorted: Vec<i32> = cryptic_residues.into_iter().collect();
    sorted.sort();
    Ok((sorted, ligand_coords))
}

/// Compute centroid of a set of 3D points
pub fn compute_centroid(coords: &[[f64; 3]]) -> [f64; 3] {
    if coords.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let n = coords.len() as f64;
    let sum_x: f64 = coords.iter().map(|c| c[0]).sum();
    let sum_y: f64 = coords.iter().map(|c| c[1]).sum();
    let sum_z: f64 = coords.iter().map(|c| c[2]).sum();

    [sum_x / n, sum_y / n, sum_z / n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pocket_type_default() {
        let pocket_type = PocketType::default();
        assert_eq!(pocket_type, PocketType::Unknown);
    }

    #[test]
    fn test_compute_centroid() {
        let coords = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let centroid = compute_centroid(&coords);

        assert!((centroid[0] - 0.666).abs() < 0.01);
        assert!((centroid[1] - 0.666).abs() < 0.01);
        assert!((centroid[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_centroid() {
        let coords: Vec<[f64; 3]> = Vec::new();
        let centroid = compute_centroid(&coords);
        assert_eq!(centroid, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ground_truth_set() {
        let entry = PocketMinerEntry {
            pdb_id: "test".to_string(),
            apo_path: PathBuf::new(),
            holo_path: PathBuf::new(),
            cryptic_residues: vec![10, 15, 20, 25],
            chain_ids: vec!["A".to_string()],
            ligand_coords: vec![],
            pocket_centroid: [0.0, 0.0, 0.0],
            pocket_type: PocketType::Unknown,
            ligand_id: None,
            n_pocket_residues: 4,
        };

        let gt = entry.ground_truth_set();
        assert!(gt.contains(&10));
        assert!(gt.contains(&25));
        assert!(!gt.contains(&30));
    }

    #[test]
    fn test_dataset_new() {
        let dataset = PocketMinerDataset::new(Path::new("/tmp"));
        assert_eq!(dataset.n_structures, 0);
        assert_eq!(dataset.n_pockets, 0);
    }
}
