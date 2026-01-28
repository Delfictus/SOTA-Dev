//! PDBBind dataset loader for training PRISM-LBS
//!
//! Supports PDBBind refined set (~5,000 complexes) for training
//! druggability scoring weights via FluxNet RL.

use crate::{LbsError, ProteinStructure};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// PDBBind dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdbBindConfig {
    /// Path to PDBBind root directory
    pub root_dir: PathBuf,
    /// Subset to use: "refined", "general", "core"
    pub subset: String,
    /// Year filter (e.g., 2020)
    pub year: Option<u32>,
    /// Maximum number of entries to load
    pub max_entries: Option<usize>,
    /// Filter by resolution (Å)
    pub max_resolution: Option<f64>,
    /// Filter by binding affinity (pKd/pKi)
    pub min_affinity: Option<f64>,
}

impl Default for PdbBindConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("data/pdbbind"),
            subset: "refined".to_string(),
            year: Some(2020),
            max_entries: None,
            max_resolution: Some(2.5),
            min_affinity: Some(4.0),  // pKd >= 4.0
        }
    }
}

/// Single PDBBind entry
#[derive(Debug, Clone)]
pub struct PdbBindEntry {
    /// PDB ID (4-letter code)
    pub pdb_id: String,
    /// Protein structure
    pub structure: ProteinStructure,
    /// Ligand coordinates (heavy atoms)
    pub ligand_coords: Vec<[f64; 3]>,
    /// Ligand center of mass
    pub ligand_center: [f64; 3],
    /// Binding affinity (pKd or pKi)
    pub affinity: f64,
    /// Affinity type ("Kd", "Ki", "IC50")
    pub affinity_type: String,
    /// Resolution (Å)
    pub resolution: f64,
    /// Year of structure
    pub year: u32,
}

/// PDBBind dataset loader
pub struct PdbBindLoader {
    config: PdbBindConfig,
    index: HashMap<String, PdbBindIndexEntry>,
}

#[derive(Debug, Clone)]
struct PdbBindIndexEntry {
    pdb_id: String,
    resolution: f64,
    year: u32,
    affinity: f64,
    affinity_type: String,
    ligand_name: String,
}

impl PdbBindLoader {
    /// Create new PDBBind loader
    pub fn new(config: PdbBindConfig) -> Result<Self, LbsError> {
        let mut loader = Self {
            config,
            index: HashMap::new(),
        };
        loader.load_index()?;
        Ok(loader)
    }

    /// Load the PDBBind index file
    fn load_index(&mut self) -> Result<(), LbsError> {
        let index_path = self.config.root_dir
            .join(&self.config.subset)
            .join("index")
            .join(format!("{}_INDEX.csv", self.config.subset.to_uppercase()));

        // Try alternative path formats
        let index_paths = vec![
            index_path.clone(),
            self.config.root_dir.join("INDEX_refined_data.csv"),
            self.config.root_dir.join("index.csv"),
            self.config.root_dir.join(format!("{}_INDEX.csv", self.config.subset)),
        ];

        let mut content = None;
        for path in &index_paths {
            if path.exists() {
                content = Some(fs::read_to_string(path)
                    .map_err(|e| LbsError::Io(e))?);
                log::info!("Loaded PDBBind index from: {}", path.display());
                break;
            }
        }

        let content = content.ok_or_else(|| {
            LbsError::Config(format!(
                "PDBBind index not found. Tried: {:?}",
                index_paths
            ))
        })?;

        // Parse index file (CSV format varies by PDBBind version)
        for line in content.lines().skip(1) {  // Skip header
            if let Some(entry) = self.parse_index_line(line) {
                // Apply filters
                if let Some(max_res) = self.config.max_resolution {
                    if entry.resolution > max_res {
                        continue;
                    }
                }
                if let Some(min_aff) = self.config.min_affinity {
                    if entry.affinity < min_aff {
                        continue;
                    }
                }
                if let Some(year) = self.config.year {
                    if entry.year < year - 5 {  // Allow 5-year window
                        continue;
                    }
                }

                self.index.insert(entry.pdb_id.clone(), entry);

                if let Some(max) = self.config.max_entries {
                    if self.index.len() >= max {
                        break;
                    }
                }
            }
        }

        log::info!("Loaded {} PDBBind entries", self.index.len());
        Ok(())
    }

    /// Parse a single index line
    fn parse_index_line(&self, line: &str) -> Option<PdbBindIndexEntry> {
        let parts: Vec<&str> = line.split([',', '\t', ' '])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if parts.len() < 4 {
            return None;
        }

        // Format: PDB_ID  resolution  year  affinity  ligand_name
        // or: PDB_ID,resolution,release_year,-logKd/Ki,Kd/Ki,ligand
        let pdb_id = parts[0].to_lowercase();
        if pdb_id.len() != 4 {
            return None;
        }

        let resolution: f64 = parts.get(1)?.parse().ok()?;
        let year: u32 = parts.get(2)?.parse().ok()?;

        // Parse affinity (handle various formats)
        let affinity_str = parts.get(3)?;
        let (affinity, affinity_type) = self.parse_affinity(affinity_str)?;

        let ligand_name = parts.get(4).unwrap_or(&"UNK").to_string();

        Some(PdbBindIndexEntry {
            pdb_id,
            resolution,
            year,
            affinity,
            affinity_type,
            ligand_name,
        })
    }

    /// Parse affinity string (e.g., "Kd=1.5nM" or "6.82")
    fn parse_affinity(&self, s: &str) -> Option<(f64, String)> {
        // Direct pKd/pKi value
        if let Ok(val) = s.parse::<f64>() {
            return Some((val, "pKd".to_string()));
        }

        // Parse "Kd=1.5nM" format
        let s_lower = s.to_lowercase();
        let affinity_type = if s_lower.contains("kd") {
            "Kd"
        } else if s_lower.contains("ki") {
            "Ki"
        } else if s_lower.contains("ic50") {
            "IC50"
        } else {
            "Kd"
        };

        // Extract numeric value
        let num_str: String = s.chars()
            .filter(|c| c.is_numeric() || *c == '.' || *c == '-' || *c == 'e' || *c == 'E')
            .collect();

        let value: f64 = num_str.parse().ok()?;

        // Convert to pKd (assuming nM if no unit)
        let multiplier = if s_lower.contains("um") || s_lower.contains("μm") {
            1e-6
        } else if s_lower.contains("nm") {
            1e-9
        } else if s_lower.contains("pm") {
            1e-12
        } else if s_lower.contains("mm") {
            1e-3
        } else {
            1e-9  // Default to nM
        };

        let molar = value * multiplier;
        let pkd = -molar.log10();

        Some((pkd, affinity_type.to_string()))
    }

    /// Load a single entry
    pub fn load_entry(&self, pdb_id: &str) -> Result<PdbBindEntry, LbsError> {
        let index_entry = self.index.get(pdb_id)
            .ok_or_else(|| LbsError::Config(format!("PDB ID {} not in index", pdb_id)))?;

        // Find protein PDB file
        let protein_paths = vec![
            self.config.root_dir.join(&self.config.subset).join(pdb_id).join(format!("{}_protein.pdb", pdb_id)),
            self.config.root_dir.join(pdb_id).join(format!("{}_protein.pdb", pdb_id)),
            self.config.root_dir.join(format!("{}_protein.pdb", pdb_id)),
        ];

        let protein_path = protein_paths.iter()
            .find(|p| p.exists())
            .ok_or_else(|| LbsError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Protein file not found for {}", pdb_id)
            )))?;

        let structure = ProteinStructure::from_pdb_file(protein_path)?;

        // Find ligand file
        let ligand_paths = vec![
            self.config.root_dir.join(&self.config.subset).join(pdb_id).join(format!("{}_ligand.mol2", pdb_id)),
            self.config.root_dir.join(&self.config.subset).join(pdb_id).join(format!("{}_ligand.sdf", pdb_id)),
            self.config.root_dir.join(pdb_id).join(format!("{}_ligand.mol2", pdb_id)),
        ];

        let ligand_coords = ligand_paths.iter()
            .find(|p| p.exists())
            .and_then(|p| self.parse_ligand_coords(p).ok())
            .unwrap_or_default();

        let ligand_center = if ligand_coords.is_empty() {
            [0.0, 0.0, 0.0]
        } else {
            let n = ligand_coords.len() as f64;
            [
                ligand_coords.iter().map(|c| c[0]).sum::<f64>() / n,
                ligand_coords.iter().map(|c| c[1]).sum::<f64>() / n,
                ligand_coords.iter().map(|c| c[2]).sum::<f64>() / n,
            ]
        };

        Ok(PdbBindEntry {
            pdb_id: pdb_id.to_string(),
            structure,
            ligand_coords,
            ligand_center,
            affinity: index_entry.affinity,
            affinity_type: index_entry.affinity_type.clone(),
            resolution: index_entry.resolution,
            year: index_entry.year,
        })
    }

    /// Parse ligand coordinates from mol2 or sdf file
    fn parse_ligand_coords(&self, path: &Path) -> Result<Vec<[f64; 3]>, LbsError> {
        let content = fs::read_to_string(path)?;
        let mut coords = Vec::new();

        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");

        if ext == "mol2" {
            // Parse MOL2 format
            let mut in_atoms = false;
            for line in content.lines() {
                if line.starts_with("@<TRIPOS>ATOM") {
                    in_atoms = true;
                    continue;
                }
                if line.starts_with("@<TRIPOS>") && in_atoms {
                    break;
                }
                if in_atoms {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 5 {
                        if let (Ok(x), Ok(y), Ok(z)) = (
                            parts[2].parse::<f64>(),
                            parts[3].parse::<f64>(),
                            parts[4].parse::<f64>(),
                        ) {
                            // Skip hydrogens
                            let atom_type = parts.get(5).unwrap_or(&"");
                            if !atom_type.starts_with('H') {
                                coords.push([x, y, z]);
                            }
                        }
                    }
                }
            }
        } else if ext == "sdf" {
            // Parse SDF format
            let lines: Vec<&str> = content.lines().collect();
            if lines.len() > 4 {
                // Line 4 contains atom count
                let counts: Vec<&str> = lines[3].split_whitespace().collect();
                if let Some(n_atoms) = counts.first().and_then(|s| s.parse::<usize>().ok()) {
                    for i in 0..n_atoms {
                        if let Some(line) = lines.get(4 + i) {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 4 {
                                if let (Ok(x), Ok(y), Ok(z)) = (
                                    parts[0].parse::<f64>(),
                                    parts[1].parse::<f64>(),
                                    parts[2].parse::<f64>(),
                                ) {
                                    // Skip hydrogens
                                    let elem = parts.get(3).unwrap_or(&"");
                                    if *elem != "H" {
                                        coords.push([x, y, z]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(coords)
    }

    /// Get all PDB IDs in the index
    pub fn pdb_ids(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Load all entries (iterator)
    pub fn iter(&self) -> impl Iterator<Item = Result<PdbBindEntry, LbsError>> + '_ {
        self.index.keys().map(|id| self.load_entry(id))
    }

    /// Load entries in batches
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = Vec<Result<PdbBindEntry, LbsError>>> + '_ {
        let ids: Vec<_> = self.index.keys().cloned().collect();
        ids.chunks(batch_size)
            .map(|chunk| chunk.iter().map(|id| self.load_entry(id)).collect())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affinity_parsing() {
        let loader = PdbBindLoader {
            config: PdbBindConfig::default(),
            index: HashMap::new(),
        };

        // Test direct pKd value
        let (val, typ) = loader.parse_affinity("6.82").unwrap();
        assert!((val - 6.82).abs() < 0.01);
        assert_eq!(typ, "pKd");

        // Test Kd=X format would need actual parsing
    }
}
