//! fpocket FFI integration for gold-standard pocket detection
//!
//! fpocket (Ligand Binding Site Detection Algorithm) is a published, validated
//! method for detecting protein pockets using Voronoi tessellation and alpha spheres.
//!
//! This module provides two integration strategies:
//! - Binary execution: Call fpocket as subprocess (no compilation needed)
//! - Native FFI: Link to libfpocket.so (requires fpocket development headers)
//!
//! Reference: Le Guilloux et al. (2009) BMC Bioinformatics 10:168

use crate::pocket::properties::Pocket;
use crate::scoring::DruggabilityScore;
use crate::LbsError;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// fpocket integration mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpocketMode {
    /// Call fpocket binary as subprocess (recommended)
    Binary,
    /// Link to libfpocket.so via FFI (requires development headers)
    #[allow(dead_code)]
    NativeLib,
}

/// Configuration for fpocket integration
#[derive(Debug, Clone)]
pub struct FpocketConfig {
    /// Integration mode (binary or native library)
    pub mode: FpocketMode,
    /// Path to fpocket binary (if mode == Binary)
    pub fpocket_path: Option<PathBuf>,
    /// Minimum alpha sphere radius (Angstroms)
    pub min_alpha_radius: f64,
    /// Maximum number of pockets to return
    pub max_pockets: usize,
    /// Druggability score threshold (0.0-1.0)
    pub druggability_threshold: f64,
}

impl Default for FpocketConfig {
    fn default() -> Self {
        Self {
            mode: FpocketMode::Binary,
            fpocket_path: None, // Will search PATH
            min_alpha_radius: 3.0,
            max_pockets: 20,
            druggability_threshold: 0.0, // No filtering
        }
    }
}

/// Check if fpocket is available on the system
pub fn fpocket_available() -> bool {
    which_fpocket().is_some()
}

/// Find fpocket binary in PATH
pub fn which_fpocket() -> Option<PathBuf> {
    if let Ok(output) = Command::new("which").arg("fpocket").output() {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout);
            let path = path_str.trim();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }
    None
}

/// Run fpocket on a PDB file and parse results
///
/// # Arguments
/// * `pdb_path` - Path to input PDB file
/// * `config` - fpocket configuration
///
/// # Returns
/// Vector of detected pockets with properties from fpocket analysis
///
/// # Errors
/// Returns error if fpocket execution fails or output parsing fails
pub fn run_fpocket(pdb_path: &Path, config: &FpocketConfig) -> Result<Vec<Pocket>, LbsError> {
    match config.mode {
        FpocketMode::Binary => run_fpocket_binary(pdb_path, config),
        FpocketMode::NativeLib => {
            // Future: implement native FFI binding
            Err(LbsError::Config(
                "Native library mode not yet implemented. Use Binary mode.".to_string(),
            ))
        }
    }
}

/// Run fpocket as external binary and parse output
fn run_fpocket_binary(pdb_path: &Path, config: &FpocketConfig) -> Result<Vec<Pocket>, LbsError> {
    // Validate input
    if !pdb_path.exists() {
        return Err(LbsError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("PDB file not found: {}", pdb_path.display()),
        )));
    }

    // Find fpocket executable
    let fpocket_exe = config
        .fpocket_path
        .clone()
        .or_else(which_fpocket)
        .ok_or_else(|| {
            LbsError::Config(
                "fpocket not found in PATH. Install fpocket or specify fpocket_path.".to_string(),
            )
        })?;

    log::info!("Running fpocket on: {}", pdb_path.display());
    log::debug!(
        "fpocket executable: {}, min_radius: {}",
        fpocket_exe.display(),
        config.min_alpha_radius
    );

    // Run fpocket: fpocket -f input.pdb -m <min_radius>
    let output = Command::new(&fpocket_exe)
        .arg("-f")
        .arg(pdb_path)
        .arg("-m")
        .arg(config.min_alpha_radius.to_string())
        .output()
        .map_err(|e| {
            LbsError::Gpu(format!("Failed to execute fpocket: {}. Ensure fpocket is installed and in PATH.", e))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(LbsError::Gpu(format!(
            "fpocket execution failed: {}",
            stderr
        )));
    }

    // fpocket creates output directory: <input>_out/
    let pdb_stem = pdb_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| LbsError::PdbParse("Invalid PDB filename".to_string()))?;

    let output_dir = pdb_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{}_out", pdb_stem));

    if !output_dir.exists() {
        return Err(LbsError::Gpu(
            "fpocket did not create output directory. Check fpocket installation.".to_string(),
        ));
    }

    log::info!("fpocket output directory: {}", output_dir.display());

    // Parse fpocket output files
    parse_fpocket_output(&output_dir, pdb_path, config)
}

/// Parse fpocket output directory to extract pockets
///
/// fpocket creates:
/// - <input>_out/<input>_info.txt - Pocket statistics (volume, druggability, etc.)
/// - <input>_out/<input>_pockets.pdb - All pockets as HETATM spheres
/// - <input>_out/pockets/pocket<N>_atm.pdb - Individual pocket atoms
fn parse_fpocket_output(
    output_dir: &Path,
    original_pdb: &Path,
    config: &FpocketConfig,
) -> Result<Vec<Pocket>, LbsError> {
    let pdb_stem = original_pdb
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| LbsError::PdbParse("Invalid PDB filename".to_string()))?;

    // Parse info file for pocket statistics
    let info_path = output_dir.join(format!("{}_info.txt", pdb_stem));
    if !info_path.exists() {
        return Err(LbsError::Gpu(format!(
            "fpocket info file not found: {}",
            info_path.display()
        )));
    }

    log::debug!("Parsing fpocket info file: {}", info_path.display());
    let pocket_info = parse_fpocket_info(&info_path)?;

    // Parse pocket PDB files
    let pockets_dir = output_dir.join("pockets");
    if !pockets_dir.exists() {
        log::warn!(
            "fpocket pockets directory not found: {}",
            pockets_dir.display()
        );
        return Ok(Vec::new());
    }

    let mut pockets = Vec::new();
    for pocket_id in 1..=config.max_pockets {
        let pocket_pdb = pockets_dir.join(format!("pocket{}_atm.pdb", pocket_id));
        if !pocket_pdb.exists() {
            break; // No more pockets
        }

        // Parse pocket atoms
        if let Ok(pocket) = parse_fpocket_pocket_pdb(&pocket_pdb, pocket_id, &pocket_info) {
            // Filter by druggability threshold
            if pocket.druggability_score.total >= config.druggability_threshold {
                pockets.push(pocket);
            }
        }
    }

    log::info!("fpocket detected {} pockets", pockets.len());
    Ok(pockets)
}

/// Parse fpocket info.txt file for pocket statistics
///
/// Format:
/// ```text
/// Pocket 1 :
///     Score:                  0.8932
///     Druggability Score:     0.67
///     Number of Alpha Spheres:    85
///     Total SASA:             650.34
///     Polar SASA:             234.12
///     Apolar SASA:            416.22
///     Volume:                 584.3
///     Mean local hydrophobic density: 15.23
///     Mean alpha sphere radius:       3.87
/// ```
fn parse_fpocket_info(info_path: &Path) -> Result<HashMap<usize, PocketInfo>, LbsError> {
    let content = fs::read_to_string(info_path).map_err(|e| {
        LbsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to read fpocket info file: {}", e),
        ))
    })?;

    let mut pockets = HashMap::new();
    let mut current_pocket: Option<usize> = None;
    let mut current_info = PocketInfo::default();

    for line in content.lines() {
        let trimmed = line.trim();

        // Detect pocket header: "Pocket 1 :"
        if trimmed.starts_with("Pocket ") && trimmed.ends_with(':') {
            // Save previous pocket
            if let Some(id) = current_pocket {
                pockets.insert(id, current_info.clone());
            }

            // Parse new pocket ID
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(id) = parts[1].parse::<usize>() {
                    current_pocket = Some(id);
                    current_info = PocketInfo::default();
                }
            }
        } else if current_pocket.is_some() {
            // Parse property lines
            if let Some((key, value)) = trimmed.split_once(':') {
                let key = key.trim();
                let value = value.trim();

                match key {
                    "Score" => {
                        current_info.score = value.parse().unwrap_or(0.0);
                    }
                    "Druggability Score" => {
                        current_info.druggability = value.parse().unwrap_or(0.0);
                    }
                    "Number of Alpha Spheres" => {
                        current_info.num_spheres = value.parse().unwrap_or(0);
                    }
                    "Volume" => {
                        current_info.volume = value.parse().unwrap_or(0.0);
                    }
                    "Total SASA" => {
                        current_info.total_sasa = value.parse().unwrap_or(0.0);
                    }
                    "Polar SASA" => {
                        current_info.polar_sasa = value.parse().unwrap_or(0.0);
                    }
                    "Apolar SASA" => {
                        current_info.apolar_sasa = value.parse().unwrap_or(0.0);
                    }
                    "Mean local hydrophobic density" => {
                        current_info.mean_hydrophobicity = value.parse().unwrap_or(0.0);
                    }
                    "Mean alpha sphere radius" => {
                        current_info.mean_radius = value.parse().unwrap_or(0.0);
                    }
                    _ => {} // Ignore unknown fields
                }
            }
        }
    }

    // Save last pocket
    if let Some(id) = current_pocket {
        pockets.insert(id, current_info);
    }

    Ok(pockets)
}

/// Parse individual pocket PDB file (pocket<N>_atm.pdb)
fn parse_fpocket_pocket_pdb(
    pocket_pdb: &Path,
    pocket_id: usize,
    info_map: &HashMap<usize, PocketInfo>,
) -> Result<Pocket, LbsError> {
    let content = fs::read_to_string(pocket_pdb)?;

    // Parse PDB to extract atom indices and coordinates
    let mut atom_indices = Vec::new();
    let mut coords = Vec::new();
    let mut residue_indices = Vec::new();

    for line in content.lines() {
        if line.starts_with("ATOM") || line.starts_with("HETATM") {
            // Parse PDB ATOM/HETATM record
            if line.len() >= 54 {
                // Extract serial number (columns 7-11)
                if let Ok(serial) = line[6..11].trim().parse::<usize>() {
                    atom_indices.push(serial - 1); // 0-indexed
                }

                // Extract coordinates (columns 31-38, 39-46, 47-54)
                let x = line[30..38].trim().parse::<f64>().unwrap_or(0.0);
                let y = line[38..46].trim().parse::<f64>().unwrap_or(0.0);
                let z = line[46..54].trim().parse::<f64>().unwrap_or(0.0);
                coords.push([x, y, z]);

                // Extract residue sequence number (columns 23-26)
                if let Ok(res_seq) = line[22..26].trim().parse::<usize>() {
                    if !residue_indices.contains(&res_seq) {
                        residue_indices.push(res_seq);
                    }
                }
            }
        }
    }

    // Calculate centroid
    let mut centroid = [0.0, 0.0, 0.0];
    if !coords.is_empty() {
        for coord in &coords {
            centroid[0] += coord[0];
            centroid[1] += coord[1];
            centroid[2] += coord[2];
        }
        let n = coords.len() as f64;
        centroid[0] /= n;
        centroid[1] /= n;
        centroid[2] /= n;
    }

    // Get info from parsed info file
    let info = info_map.get(&pocket_id).cloned().unwrap_or_default();

    Ok(Pocket {
        atom_indices,
        residue_indices,
        centroid,
        volume: info.volume,
        enclosure_ratio: 0.0, // fpocket doesn't provide this
        mean_hydrophobicity: info.mean_hydrophobicity / 100.0, // Normalize
        mean_sasa: if coords.is_empty() {
            0.0
        } else {
            info.total_sasa / coords.len() as f64
        },
        mean_depth: 0.0, // Not provided by fpocket
        mean_flexibility: 0.0,
        mean_conservation: 0.0,
        persistence_score: info.score,
        hbond_donors: 0,    // Not provided by fpocket
        hbond_acceptors: 0, // Not provided by fpocket
        druggability_score: DruggabilityScore {
            total: info.druggability,
            classification: classify_druggability(info.druggability),
            components: crate::scoring::Components {
                volume: (info.volume / 1000.0).min(1.0),
                hydro: (info.mean_hydrophobicity / 20.0).min(1.0),
                enclosure: 0.5,
                depth: 0.0,
                hbond: 0.0,
                flex: 0.0,
                cons: 0.0,
                topo: 0.0,
            },
        },
        boundary_atoms: Vec::new(),
        mean_electrostatic: 0.0,
        gnn_embedding: Vec::new(),
        gnn_druggability: 0.0,
    })
}

/// Classify druggability score into categories
/// fpocket scores range from 0.0 to 1.0, map to PRISM's classification
fn classify_druggability(score: f64) -> crate::scoring::DrugabilityClass {
    if score >= 0.7 {
        crate::scoring::DrugabilityClass::HighlyDruggable
    } else if score >= 0.5 {
        crate::scoring::DrugabilityClass::Druggable
    } else if score >= 0.3 {
        crate::scoring::DrugabilityClass::DifficultTarget
    } else {
        crate::scoring::DrugabilityClass::Undruggable
    }
}

/// fpocket pocket information from info.txt
#[derive(Debug, Clone, Default)]
struct PocketInfo {
    score: f64,
    druggability: f64,
    num_spheres: usize,
    volume: f64,
    total_sasa: f64,
    polar_sasa: f64,
    apolar_sasa: f64,
    mean_hydrophobicity: f64,
    mean_radius: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fpocket_availability() {
        // Non-critical test - fpocket may not be installed
        let available = fpocket_available();
        if available {
            println!("fpocket found: {:?}", which_fpocket());
        } else {
            println!("fpocket not found (this is OK for testing)");
        }
    }

    #[test]
    fn test_fpocket_config_defaults() {
        let config = FpocketConfig::default();
        assert_eq!(config.mode, FpocketMode::Binary);
        assert_eq!(config.min_alpha_radius, 3.0);
        assert_eq!(config.max_pockets, 20);
    }

    #[test]
    fn test_druggability_classification() {
        use crate::scoring::DrugabilityClass;

        assert!(matches!(
            classify_druggability(0.8),
            DrugabilityClass::HighlyDruggable
        ));
        assert!(matches!(
            classify_druggability(0.6),
            DrugabilityClass::Druggable
        ));
        assert!(matches!(
            classify_druggability(0.4),
            DrugabilityClass::DifficultTarget
        ));
        assert!(matches!(
            classify_druggability(0.2),
            DrugabilityClass::Undruggable
        ));
    }
}
