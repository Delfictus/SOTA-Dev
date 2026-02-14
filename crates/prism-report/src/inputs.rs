//! Input data structures and parsers for cryo-probe outputs

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Cryo-probe results from NHS pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryoProbeResults {
    /// Input topology file
    pub input: String,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of residues
    pub n_residues: usize,
    /// Temperature protocol name
    pub protocol: String,
    /// Start temperature (K)
    pub start_temp: f32,
    /// End temperature (K)
    pub end_temp: f32,
    /// Total simulation steps
    pub total_steps: i32,
    /// Total spikes detected
    pub total_spikes: usize,
    /// Spikes by phase
    pub phase_spikes: PhaseSpikes,
    /// Elapsed time (seconds)
    pub elapsed_seconds: f64,
    /// Steps per second
    pub steps_per_second: f64,
    /// ns/day performance
    pub ns_per_day: f64,
    /// UV configuration
    pub uv_config: UvConfig,
    /// Trajectory information (optional)
    #[serde(default)]
    pub trajectory: Option<TrajectoryInfo>,
}

impl CryoProbeResults {
    /// Load from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read cryo probe results: {}", path.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse cryo probe results: {}", path.display()))
    }
}

/// Spike counts by temperature phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSpikes {
    /// Cold phase spikes (primary hydrophobic mapping)
    pub cold: usize,
    /// Ramp phase spikes (cryptic site emergence)
    pub ramp: usize,
    /// Warm phase spikes (physiological validation)
    pub warm: usize,
}

impl PhaseSpikes {
    /// Total spikes
    pub fn total(&self) -> usize {
        self.cold + self.ramp + self.warm
    }

    /// Cold phase fraction
    pub fn cold_fraction(&self) -> f64 {
        let total = self.total();
        if total == 0 { 0.0 } else { self.cold as f64 / total as f64 }
    }

    /// Ramp phase fraction
    pub fn ramp_fraction(&self) -> f64 {
        let total = self.total();
        if total == 0 { 0.0 } else { self.ramp as f64 / total as f64 }
    }
}

/// UV probe configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvConfig {
    /// Burst energy (kcal/mol)
    pub burst_energy: f32,
    /// Burst interval (timesteps)
    pub burst_interval: i32,
    /// Number of aromatic targets
    pub n_targets: usize,
    /// Spectroscopy mode enabled
    #[serde(default)]
    pub spectroscopy_mode: bool,
    /// Frequency hopping enabled
    #[serde(default)]
    pub frequency_hopping: bool,
    /// Scanned wavelengths (nm)
    #[serde(default)]
    pub wavelengths_nm: Option<Vec<f32>>,
}

/// Trajectory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryInfo {
    /// Total frames
    pub total_frames: usize,
    /// Spike-triggered frames
    pub spike_triggered_frames: usize,
    /// Interval frames
    pub interval_frames: usize,
    /// Time range (ps)
    pub time_range_ps: (f32, f32),
    /// Temperature range (K)
    pub temperature_range_k: (f32, f32),
    /// Ensemble PDB path
    pub ensemble_pdb: String,
}

/// Topology data from PRISM-PREP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyData {
    /// Source PDB file
    pub source_pdb: String,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of residues
    pub n_residues: usize,
    /// Number of chains
    pub n_chains: usize,
    /// Atom positions (flat x,y,z array)
    pub positions: Vec<f32>,
    /// Atom names
    pub atom_names: Vec<String>,
    /// Residue names (per atom)
    pub residue_names: Vec<String>,
    /// Residue IDs (per atom)
    pub residue_ids: Vec<usize>,
    /// Chain IDs (per atom)
    pub chain_ids: Vec<String>,
    /// Aromatic targets for UV
    #[serde(default)]
    pub aromatic_targets: Vec<AromaticTarget>,
    /// Number of aromatics
    #[serde(default)]
    pub n_aromatics: usize,
    /// C-alpha indices
    #[serde(default)]
    pub ca_indices: Vec<usize>,
}

impl TopologyData {
    /// Load from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read topology: {}", path.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse topology: {}", path.display()))
    }

    /// Get position of atom i as [x, y, z]
    pub fn get_position(&self, i: usize) -> [f32; 3] {
        let base = i * 3;
        [
            self.positions[base],
            self.positions[base + 1],
            self.positions[base + 2],
        ]
    }

    /// Get residue centroid
    pub fn residue_centroid(&self, res_id: usize) -> [f32; 3] {
        let mut sum = [0.0f32; 3];
        let mut count = 0;
        for (i, &rid) in self.residue_ids.iter().enumerate() {
            if rid == res_id {
                let pos = self.get_position(i);
                sum[0] += pos[0];
                sum[1] += pos[1];
                sum[2] += pos[2];
                count += 1;
            }
        }
        if count == 0 {
            return [0.0; 3];
        }
        [
            sum[0] / count as f32,
            sum[1] / count as f32,
            sum[2] / count as f32,
        ]
    }

    /// Get unique residues as (res_id, res_name, chain_id)
    pub fn unique_residues(&self) -> Vec<(usize, String, String)> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for i in 0..self.n_atoms {
            let key = (self.residue_ids[i], self.chain_ids[i].clone());
            if !seen.contains(&key) {
                seen.insert(key.clone());
                result.push((
                    self.residue_ids[i],
                    self.residue_names[i].clone(),
                    self.chain_ids[i].clone(),
                ));
            }
        }
        result
    }
}

/// Aromatic target for UV spectroscopy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AromaticTarget {
    /// Residue index
    pub residue_idx: usize,
    /// Residue name (TRP, TYR, PHE)
    pub residue_name: String,
    /// Residue ID
    pub residue_id: i32,
    /// Ring atom indices
    pub ring_atom_indices: Vec<usize>,
    /// Ring center position
    pub ring_center: [f32; 3],
    /// Extinction coefficient at 280nm
    pub extinction_280: f32,
}

/// Holo structure for Tier 1 correlation
#[derive(Debug, Clone)]
pub struct HoloStructure {
    /// Atom positions (CA only for alignment)
    pub ca_positions: Vec<[f32; 3]>,
    /// Ligand atom positions (HETATM)
    pub ligand_atoms: Vec<LigandAtom>,
    /// Ligand centroid
    pub ligand_centroid: [f32; 3],
    /// Residue IDs within 4Å of ligand
    pub contact_residues: Vec<usize>,
}

/// Ligand atom from HETATM records
#[derive(Debug, Clone)]
pub struct LigandAtom {
    /// Atom name
    pub name: String,
    /// Residue name (ligand code)
    pub resname: String,
    /// Position
    pub position: [f32; 3],
    /// Element
    pub element: String,
}

impl HoloStructure {
    /// Load from PDB file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read holo PDB: {}", path.display()))?;

        let mut ca_positions = Vec::new();
        let mut ligand_atoms = Vec::new();

        // Skip these as non-ligand HETATMs
        let skip_resnames = ["HOH", "WAT", "NA", "CL", "MG", "CA", "ZN", "FE", "K", "MN"];

        for line in content.lines() {
            if line.starts_with("ATOM") {
                // Parse CA atoms
                let atom_name = line.get(12..16).unwrap_or("").trim();
                if atom_name == "CA" {
                    let x: f32 = line.get(30..38).unwrap_or("0").trim().parse().unwrap_or(0.0);
                    let y: f32 = line.get(38..46).unwrap_or("0").trim().parse().unwrap_or(0.0);
                    let z: f32 = line.get(46..54).unwrap_or("0").trim().parse().unwrap_or(0.0);
                    ca_positions.push([x, y, z]);
                }
            } else if line.starts_with("HETATM") {
                let resname = line.get(17..20).unwrap_or("").trim();
                if !skip_resnames.contains(&resname) {
                    let atom_name = line.get(12..16).unwrap_or("").trim().to_string();
                    let x: f32 = line.get(30..38).unwrap_or("0").trim().parse().unwrap_or(0.0);
                    let y: f32 = line.get(38..46).unwrap_or("0").trim().parse().unwrap_or(0.0);
                    let z: f32 = line.get(46..54).unwrap_or("0").trim().parse().unwrap_or(0.0);
                    let element = line.get(76..78).unwrap_or("").trim().to_string();

                    ligand_atoms.push(LigandAtom {
                        name: atom_name,
                        resname: resname.to_string(),
                        position: [x, y, z],
                        element,
                    });
                }
            }
        }

        // Compute ligand centroid
        let ligand_centroid = if ligand_atoms.is_empty() {
            [0.0; 3]
        } else {
            let mut sum = [0.0f32; 3];
            for atom in &ligand_atoms {
                sum[0] += atom.position[0];
                sum[1] += atom.position[1];
                sum[2] += atom.position[2];
            }
            let n = ligand_atoms.len() as f32;
            [sum[0] / n, sum[1] / n, sum[2] / n]
        };

        Ok(Self {
            ca_positions,
            ligand_atoms,
            ligand_centroid,
            contact_residues: Vec::new(), // Computed after alignment
        })
    }

    /// Distance from point to nearest ligand atom
    pub fn distance_to_ligand(&self, point: [f32; 3]) -> f32 {
        self.ligand_atoms
            .iter()
            .map(|atom| {
                let dx = point[0] - atom.position[0];
                let dy = point[1] - atom.position[1];
                let dz = point[2] - atom.position[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(f32::INFINITY, f32::min)
    }

    /// Distance from point to ligand centroid
    pub fn distance_to_centroid(&self, point: [f32; 3]) -> f32 {
        let dx = point[0] - self.ligand_centroid[0];
        let dy = point[1] - self.ligand_centroid[1];
        let dz = point[2] - self.ligand_centroid[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Truth residues for Tier 2 correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthResidues {
    /// Known binding site residue IDs
    pub residues: Vec<usize>,
    /// Optional site name
    #[serde(default)]
    pub site_name: Option<String>,
    /// Optional notes
    #[serde(default)]
    pub notes: Option<String>,
}

impl TruthResidues {
    /// Load from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read truth residues: {}", path.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse truth residues: {}", path.display()))
    }

    /// Compute precision for predicted residues
    pub fn precision(&self, predicted: &[usize]) -> f64 {
        if predicted.is_empty() {
            return 0.0;
        }
        let truth_set: std::collections::HashSet<_> = self.residues.iter().collect();
        let correct = predicted.iter().filter(|r| truth_set.contains(r)).count();
        correct as f64 / predicted.len() as f64
    }

    /// Compute recall for predicted residues
    pub fn recall(&self, predicted: &[usize]) -> f64 {
        if self.residues.is_empty() {
            return 0.0;
        }
        let pred_set: std::collections::HashSet<_> = predicted.iter().collect();
        let correct = self.residues.iter().filter(|r| pred_set.contains(r)).count();
        correct as f64 / self.residues.len() as f64
    }

    /// Compute F1 score
    pub fn f1(&self, predicted: &[usize]) -> f64 {
        let p = self.precision(predicted);
        let r = self.recall(predicted);
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

/// Frame data from trajectory JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryFrame {
    /// Frame index
    pub frame_idx: usize,
    /// Timestep
    pub timestep: i32,
    /// Temperature (K)
    pub temperature: f32,
    /// Time (ps)
    pub time_ps: f32,
    /// Atom positions (flat x,y,z)
    pub positions: Vec<f32>,
    /// Spike-triggered frame
    #[serde(default)]
    pub spike_triggered: bool,
    /// Spike count (if triggered)
    #[serde(default)]
    pub spike_count: Option<usize>,
    /// Spike voxel indices
    #[serde(default)]
    pub spike_voxels: Option<Vec<usize>>,
    /// UV wavelength (spectroscopy mode)
    #[serde(default)]
    pub wavelength_nm: Option<f32>,
}

/// Load trajectory frames from JSON
pub fn load_trajectory_frames(path: &Path) -> Result<Vec<TrajectoryFrame>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read trajectory frames: {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse trajectory frames: {}", path.display()))
}
