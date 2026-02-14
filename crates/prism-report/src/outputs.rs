//! Output contract structures and writers

use crate::ablation::AblationResults;
use crate::config::RankingWeights;
use crate::correlation::CorrelationResult;
use crate::sites::CrypticSite;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Output directory contract (must match exactly)
#[derive(Debug, Clone)]
pub struct OutputContract {
    /// Base output directory
    pub base_dir: PathBuf,
}

impl OutputContract {
    /// Create output contract and ensure directory structure
    pub fn new(base_dir: &Path) -> Result<Self> {
        let contract = Self {
            base_dir: base_dir.to_path_buf(),
        };
        contract.create_directories()?;
        Ok(contract)
    }

    /// Create required directory structure
    fn create_directories(&self) -> Result<()> {
        let dirs = [
            self.base_dir.clone(),
            self.sites_dir(),
            self.volumes_dir(),
            self.trajectories_dir(),
            self.provenance_dir(),
        ];

        for dir in &dirs {
            fs::create_dir_all(dir)
                .with_context(|| format!("Failed to create directory: {}", dir.display()))?;
        }

        Ok(())
    }

    // Path accessors
    pub fn report_html(&self) -> PathBuf {
        self.base_dir.join("report.html")
    }

    pub fn report_pdf(&self) -> PathBuf {
        self.base_dir.join("report.pdf")
    }

    pub fn summary_json(&self) -> PathBuf {
        self.base_dir.join("summary.json")
    }

    pub fn correlation_csv(&self) -> PathBuf {
        self.base_dir.join("correlation.csv")
    }

    /// New primary name for site metrics CSV
    pub fn site_metrics_csv(&self) -> PathBuf {
        self.base_dir.join("site_metrics.csv")
    }

    pub fn sites_dir(&self) -> PathBuf {
        self.base_dir.join("sites")
    }

    pub fn volumes_dir(&self) -> PathBuf {
        self.base_dir.join("volumes")
    }

    pub fn trajectories_dir(&self) -> PathBuf {
        self.base_dir.join("trajectories")
    }

    pub fn provenance_dir(&self) -> PathBuf {
        self.base_dir.join("provenance")
    }

    /// Get site-specific directory
    pub fn site_dir(&self, site_id: &str) -> PathBuf {
        self.sites_dir().join(site_id)
    }

    /// Create site output structure
    pub fn create_site_output(&self, site_id: &str) -> Result<SiteOutput> {
        SiteOutput::new(&self.site_dir(site_id))
    }
}

/// Per-site output structure
#[derive(Debug, Clone)]
pub struct SiteOutput {
    /// Site directory
    pub base_dir: PathBuf,
}

impl SiteOutput {
    pub fn new(base_dir: &Path) -> Result<Self> {
        fs::create_dir_all(base_dir)?;
        fs::create_dir_all(base_dir.join("figures"))?;

        Ok(Self {
            base_dir: base_dir.to_path_buf(),
        })
    }

    pub fn site_pdb(&self) -> PathBuf {
        self.base_dir.join("site.pdb")
    }

    pub fn site_mol2(&self) -> PathBuf {
        self.base_dir.join("site.mol2")
    }

    pub fn residues_txt(&self) -> PathBuf {
        self.base_dir.join("residues.txt")
    }

    pub fn correlation_json(&self) -> PathBuf {
        self.base_dir.join("correlation.json")
    }

    pub fn figures_dir(&self) -> PathBuf {
        self.base_dir.join("figures")
    }

    pub fn pymol_session(&self) -> PathBuf {
        self.base_dir.join("pymol_session.pse")
    }

    pub fn chimerax_session(&self) -> PathBuf {
        self.base_dir.join("chimerax_session.cxs")
    }

    // Figure paths
    pub fn pocket_overlay_png(&self) -> PathBuf {
        self.figures_dir().join("pocket_overlay.png")
    }

    pub fn pocket_volume_vs_time_png(&self) -> PathBuf {
        self.figures_dir().join("pocket_volume_vs_time.png")
    }

    pub fn persistence_vs_replica_png(&self) -> PathBuf {
        self.figures_dir().join("persistence_vs_replica.png")
    }

    pub fn uv_vs_control_deltasasa_png(&self) -> PathBuf {
        self.figures_dir().join("uv_vs_control_deltaSASA.png")
    }

    pub fn holo_distance_hist_png(&self) -> PathBuf {
        self.figures_dir().join("holo_distance_hist.png")
    }
}

/// Summary JSON structure (matches output contract)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryJson {
    /// PRISM4D version
    pub version: String,
    /// Timestamp
    pub timestamp: String,
    /// Input configuration
    pub input: SummaryInput,
    /// Site summary
    pub sites: Vec<SiteSummary>,
    /// Ablation results
    pub ablation: AblationSummary,
    /// Correlation results
    pub correlation: Option<CorrelationSummary>,
    /// Ranking weights used
    pub ranking_weights: RankingWeights,
    /// Run statistics
    pub statistics: RunStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryInput {
    pub pdb_file: String,
    pub replicates: usize,
    pub wavelengths: Vec<f32>,
    pub holo_file: Option<String>,
    pub truth_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteSummary {
    pub site_id: String,
    pub rank: usize,
    pub centroid: [f32; 3],
    pub residues: Vec<usize>,
    pub residue_count: usize,
    pub chain_id: String,
    pub rank_score: f64,
    pub confidence: f64,
    pub is_druggable: bool,
    pub persistence: f64,
    pub volume_mean: f64,
    pub hydrophobic_fraction: f64,
    pub uv_delta_sasa: f64,
    pub uv_delta_volume: f64,
}

impl From<&CrypticSite> for SiteSummary {
    fn from(site: &CrypticSite) -> Self {
        Self {
            site_id: site.site_id.clone(),
            rank: site.rank,
            centroid: site.centroid,
            residues: site.residues.clone(),
            residue_count: site.residues.len(),
            chain_id: site.chain_id.clone(),
            rank_score: site.rank_score,
            confidence: site.confidence,
            is_druggable: site.is_druggable,
            persistence: site.metrics.persistence.present_fraction,
            volume_mean: site.metrics.geometry.volume_mean,
            hydrophobic_fraction: site.metrics.chemistry.hydrophobic_fraction,
            uv_delta_sasa: site.metrics.uv_response.delta_sasa,
            uv_delta_volume: site.metrics.uv_response.delta_volume,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationSummary {
    pub baseline_spikes: usize,
    pub cryo_only_spikes: usize,
    pub cryo_uv_spikes: usize,
    pub cryo_vs_baseline_delta: i64,
    pub cryouv_vs_cryo_delta: i64,
    pub cryo_contrast_significant: bool,
    pub uv_response_significant: bool,
    pub interpretation: String,
}

impl From<&AblationResults> for AblationSummary {
    fn from(results: &AblationResults) -> Self {
        Self {
            baseline_spikes: results.baseline.total_spikes,
            cryo_only_spikes: results.cryo_only.total_spikes,
            cryo_uv_spikes: results.cryo_uv.total_spikes,
            cryo_vs_baseline_delta: results.deltas.spikes_cryo_vs_baseline,
            cryouv_vs_cryo_delta: results.deltas.spikes_cryouv_vs_cryo,
            cryo_contrast_significant: results.comparison.cryo_contrast_significant,
            uv_response_significant: results.comparison.uv_response_significant,
            interpretation: results.comparison.interpretation.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationSummary {
    pub tier1_available: bool,
    pub tier1_best_distance: Option<f32>,
    pub tier1_hit_at_1: Option<bool>,
    pub tier1_hit_at_3: Option<bool>,
    pub tier2_available: bool,
    pub tier2_best_f1: Option<f64>,
    pub tier2_hit_at_1: Option<bool>,
    pub tier2_hit_at_3: Option<bool>,
}

impl From<&CorrelationResult> for CorrelationSummary {
    fn from(result: &CorrelationResult) -> Self {
        Self {
            tier1_available: result.tier1.is_some(),
            tier1_best_distance: result.tier1.as_ref().map(|t| t.best_distance),
            tier1_hit_at_1: result.hit_metrics.tier1_hit_at_1,
            tier1_hit_at_3: result.hit_metrics.tier1_hit_at_3,
            tier2_available: result.tier2.is_some(),
            tier2_best_f1: result.tier2.as_ref().map(|t| t.best_f1),
            tier2_hit_at_1: result.hit_metrics.tier2_hit_at_1,
            tier2_hit_at_3: result.hit_metrics.tier2_hit_at_3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStatistics {
    pub total_runtime_seconds: f64,
    pub total_spikes_detected: usize,
    pub total_frames_analyzed: usize,
    pub replicates_completed: usize,
}

/// Provenance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceManifest {
    /// All output files
    pub files: Vec<ProvenanceFile>,
    /// Generation timestamp
    pub generated_at: String,
    /// PRISM4D version
    pub prism_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceFile {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceVersions {
    pub prism_report: String,
    pub prism_nhs: String,
    pub rust_version: String,
    pub platform: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceSeeds {
    pub replicates: Vec<u64>,
    pub master_seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceParams {
    pub temperature_protocol: String,
    pub start_temp: f32,
    pub end_temp: f32,
    /// Steps to hold at cold (cryogenic) temperature before ramp
    pub cold_hold_steps: i32,
    /// Steps for temperature ramp from start_temp to end_temp
    pub ramp_steps: i32,
    /// Steps to hold at warm (physiological) temperature after ramp
    pub warm_hold_steps: i32,
    pub uv_wavelengths: Vec<f32>,
    pub uv_burst_energy: f32,
    pub uv_burst_interval: i32,
    pub grid_spacing: f32,
}

/// Write PDB file for site residues
pub fn write_site_pdb(
    path: &Path,
    site: &CrypticSite,
    positions: &[f32],
    atom_names: &[String],
    residue_names: &[String],
    residue_ids: &[usize],
    chain_ids: &[String],
) -> Result<()> {
    use std::io::Write;

    let mut file = fs::File::create(path)?;

    // Write header
    writeln!(file, "REMARK   Generated by PRISM4D prism-report")?;
    writeln!(file, "REMARK   Site ID: {}", site.site_id)?;
    writeln!(
        file,
        "REMARK   Centroid: {:.2} {:.2} {:.2}",
        site.centroid[0], site.centroid[1], site.centroid[2]
    )?;

    let site_residues: std::collections::HashSet<_> = site.residues.iter().collect();
    let n_atoms = positions.len() / 3;
    let mut atom_serial = 1;

    for i in 0..n_atoms {
        if site_residues.contains(&residue_ids[i]) {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            writeln!(
                file,
                "ATOM  {:5} {:4} {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00",
                atom_serial,
                atom_names[i],
                residue_names[i],
                chain_ids[i],
                residue_ids[i],
                x,
                y,
                z
            )?;
            atom_serial += 1;
        }
    }

    writeln!(file, "END")?;
    Ok(())
}

/// Write MOL2 file for site
pub fn write_site_mol2(
    path: &Path,
    site: &CrypticSite,
    positions: &[f32],
    atom_names: &[String],
    residue_names: &[String],
    residue_ids: &[usize],
) -> Result<()> {
    use std::io::Write;

    let mut file = fs::File::create(path)?;

    let site_residues: std::collections::HashSet<_> = site.residues.iter().collect();
    let n_atoms = positions.len() / 3;

    // Count atoms in site
    let mut site_atoms = Vec::new();
    for i in 0..n_atoms {
        if site_residues.contains(&residue_ids[i]) {
            site_atoms.push(i);
        }
    }

    // Write header
    writeln!(file, "@<TRIPOS>MOLECULE")?;
    writeln!(file, "{}", site.site_id)?;
    writeln!(file, "{} 0 0 0 0", site_atoms.len())?;
    writeln!(file, "SMALL")?;
    writeln!(file, "NO_CHARGES")?;
    writeln!(file)?;

    // Write atoms
    writeln!(file, "@<TRIPOS>ATOM")?;
    for (serial, &i) in site_atoms.iter().enumerate() {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];

        // Simple element guess from atom name
        let element = atom_names[i].chars().next().unwrap_or('C');

        writeln!(
            file,
            "{:6} {:4} {:8.4} {:8.4} {:8.4} {:4} {} {}",
            serial + 1,
            atom_names[i],
            x,
            y,
            z,
            element,
            residue_ids[i],
            residue_names[i]
        )?;
    }

    Ok(())
}

/// Write residues.txt file
pub fn write_residues_txt(path: &Path, site: &CrypticSite) -> Result<()> {
    use std::io::Write;

    let mut file = fs::File::create(path)?;
    writeln!(file, "# Site {} residues", site.site_id)?;
    writeln!(file, "# Chain: {}", site.chain_id)?;

    for (res_id, res_name) in site.residues.iter().zip(site.residue_names.iter()) {
        writeln!(file, "{}:{}", res_name, res_id)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_output_contract_directories() {
        let tmp = TempDir::new().unwrap();
        let contract = OutputContract::new(tmp.path()).unwrap();

        assert!(contract.sites_dir().exists());
        assert!(contract.volumes_dir().exists());
        assert!(contract.trajectories_dir().exists());
        assert!(contract.provenance_dir().exists());
    }

    #[test]
    fn test_site_output_directories() {
        let tmp = TempDir::new().unwrap();
        let site_dir = tmp.path().join("site_001");
        let output = SiteOutput::new(&site_dir).unwrap();

        assert!(output.figures_dir().exists());
        assert_eq!(output.site_pdb().file_name().unwrap(), "site.pdb");
    }
}
