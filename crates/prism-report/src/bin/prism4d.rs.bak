//! PRISM4D Unified CLI Entry Point
//!
//! Single command runs engine then finalize:
//! ```
//! prism4d run --pdb structure.pdb --mode cryo-uv --replicates 5 \
//!     --wavelengths 258,274,280 --out results/ID \
//!     [--holo HOLO.pdb] [--truth-residues truth.json]
//! ```
//!
//! Pipeline flow:
//! 1. Engine (prism-nhs) writes events.jsonl during MD stepping
//! 2. FinalizeStage reads events.jsonl and produces evidence pack
//!
//! INVARIANT: No fallback paths. Real engine output required.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use prism_report::{
    config::{AblationConfig, ReportConfig, TemperatureProtocol},
    finalize::FinalizeStage,
    find_chimerax, find_pdf_renderer, find_pymol,
};
use serde::{Deserialize, Serialize};

// =============================================================================
// SEED CONFIG (Deterministic Reproducibility)
// =============================================================================

/// Complete configuration snapshot for deterministic reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedConfig {
    /// Master seed
    pub seed: u64,
    /// Timestamp of run
    pub timestamp: String,
    /// PRISM4D version
    pub version: String,
    /// Input PDB file
    pub input_pdb: String,
    /// Temperature protocol
    pub temperature: TemperatureParams,
    /// UV parameters
    pub uv: UvParams,
    /// Distance filtering
    pub filtering: FilterParams,
    /// Cluster detection
    pub clustering: ClusterParams,
    /// Site acceptance
    pub acceptance: AcceptanceParams,
    /// Run statistics (populated after run)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<RunStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureParams {
    pub start_temp: f32,
    pub end_temp: f32,
    pub cold_hold_steps: i32,
    pub ramp_steps: i32,
    pub warm_hold_steps: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvParams {
    pub energy: f32,
    pub interval: i32,
    pub dwell_steps: i32,
    pub wavelengths: Vec<f32>,
    pub wavelength_sweep: bool,
    pub wavelength_min: f32,
    pub wavelength_max: f32,
    pub wavelength_step: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterParams {
    pub initial_dist: f32,
    pub max_dist: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterParams {
    pub min_events: usize,
    pub eps: f32,
    pub min_residues: usize,
    pub residue_query_radius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceParams {
    pub min_persistence: f32,
    pub min_volume: f32,
    pub min_replica_agreement: f32,
    pub replicates: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStats {
    pub total_events: usize,
    pub sites_detected: usize,
    pub druggable_sites: usize,
    pub runtime_seconds: f64,
    pub events_per_second: f64,
}

impl SeedConfig {
    /// Save seed config to JSON file
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load seed config from JSON file
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&json)?;
        Ok(config)
    }
}

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
#[cfg(feature = "gpu")]
use prism_nhs::{
    fused_engine::{NhsAmberFusedEngine, TemperatureProtocol as NhsTemperatureProtocol, UvProbeConfig},
    input::PrismPrepTopology,
};
#[cfg(feature = "gpu")]
use prism_report::event_cloud::{AblationPhase, EventWriter, PocketEvent, RawSpikeEvent, TempPhase};
use std::io::Write;

/// PRISM4D: Phase Resonance Integrated Solver Machine for Molecular Dynamics
#[derive(Parser, Debug)]
#[command(name = "prism4d")]
#[command(author = "PRISM4D Team")]
#[command(version)]
#[command(about = "Cryptic binding site detection with cryo-UV contrast", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run complete pipeline (engine + finalize)
    Run(RunArgs),

    /// Run finalize stage only (from existing events.jsonl)
    Finalize(FinalizeArgs),

    /// Check system dependencies
    Check,

    /// Show version and build info
    Version,
}

#[derive(Parser, Debug)]
struct FinalizeArgs {
    /// Path to events.jsonl from engine output
    #[arg(long, required = true)]
    events: PathBuf,

    /// Path to topology.json (MANDATORY - no fallback to PDB)
    #[arg(long, required = true)]
    topology: PathBuf,

    /// Output directory
    #[arg(long, short = 'o', required = true)]
    out: PathBuf,

    /// Input PDB file (for visualization sessions only - NOT for spatial metrics)
    #[arg(long)]
    pdb: Option<PathBuf>,

    // ═══════════════════════════════════════════════════════════════════════
    // SEED MANAGEMENT (Deterministic Reproducibility)
    // ═══════════════════════════════════════════════════════════════════════

    /// Master seed for deterministic operations
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Load seed and parameters from a previous run's seed_config.json
    #[arg(long)]
    load_seed_from: Option<PathBuf>,

    /// Save seed and all parameters to seed_config.json in output directory
    #[arg(long, default_value = "true")]
    save_seed: bool,

    // ═══════════════════════════════════════════════════════════════════════
    // DISTANCE FILTERING
    // ═══════════════════════════════════════════════════════════════════════

    /// Initial (tight) filter distance from protein atoms (Å)
    #[arg(long, default_value = "8.0")]
    filter_initial_dist: f32,

    /// Global maximum filter distance from protein atoms (Å)
    #[arg(long, default_value = "18.0")]
    filter_max_dist: f32,

    // ═══════════════════════════════════════════════════════════════════════
    // CLUSTER DETECTION TUNING
    // ═══════════════════════════════════════════════════════════════════════

    /// Minimum events per cluster (DBSCAN min_samples)
    #[arg(long, default_value = "100")]
    cluster_min_events: usize,

    /// Cluster spatial threshold / epsilon (Å) for DBSCAN
    #[arg(long, default_value = "5.0")]
    cluster_eps: f32,

    /// Minimum residues required per site
    #[arg(long, default_value = "5")]
    cluster_min_residues: usize,

    /// Residue query radius for site mapping (Å)
    #[arg(long, default_value = "8.0")]
    residue_query_radius: f32,

    /// Number of replicates (for replica agreement filtering)
    #[arg(long, default_value = "1")]
    replicates: usize,

    // ═══════════════════════════════════════════════════════════════════════
    // SITE ACCEPTANCE CRITERIA
    // ═══════════════════════════════════════════════════════════════════════

    /// Minimum site persistence (fraction of frames, 0.0-1.0)
    #[arg(long, default_value = "0.002")]
    min_persistence: f32,

    /// Minimum site volume (Å³)
    #[arg(long, default_value = "100.0")]
    min_volume: f32,

    /// Minimum replica agreement (fraction, 0.0-1.0)
    #[arg(long, default_value = "0.3")]
    min_replica_agreement: f32,

    /// Skip PDF generation
    #[arg(long)]
    no_pdf: bool,

    /// Skip PyMOL session generation
    #[arg(long)]
    no_pymol: bool,

    /// Skip ChimeraX session generation
    #[arg(long)]
    no_chimerax: bool,

    // ═══════════════════════════════════════════════════════════════════════
    // BENCHMARK VALIDATION (Tier1/Tier2 Correlation)
    // ═══════════════════════════════════════════════════════════════════════

    /// Holo structure PDB for Tier 1 correlation (distance to ligand)
    #[arg(long)]
    holo: Option<PathBuf>,

    /// Truth residues JSON for Tier 2 correlation (precision/recall)
    #[arg(long)]
    truth_residues: Option<PathBuf>,

    /// Contact distance cutoff for auto-extracting truth residues from holo (Angstroms)
    /// Used when --holo is provided but --truth-residues is not (structure-agnostic validation)
    #[arg(long, default_value = "4.5")]
    contact_cutoff: f32,
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Input topology JSON file (from prism-prep). If not provided, auto-runs prep stage.
    #[arg(long)]
    topology: Option<PathBuf>,

    /// Input PDB file (required)
    #[arg(long, required = true)]
    pdb: PathBuf,

    /// Skip AMBER H-placement in prep stage (use fast heuristic instead)
    #[arg(long)]
    no_amber: bool,

    /// Path to prism-prep script (auto-detected if not specified)
    #[arg(long)]
    prep_script: Option<PathBuf>,

    /// Execution mode
    #[arg(long, default_value = "cryo-uv")]
    mode: String,

    /// Number of replicates
    #[arg(long, default_value = "3")]
    replicates: usize,

    /// UV wavelengths (nm), comma-separated
    #[arg(long, value_delimiter = ',', default_value = "250,258,274,280")]
    wavelengths: Vec<f32>,

    /// Output directory
    #[arg(long, short = 'o', required = true)]
    out: PathBuf,

    /// Holo structure for Tier 1 correlation (optional)
    #[arg(long)]
    holo: Option<PathBuf>,

    /// Truth residues JSON for Tier 2 correlation (optional)
    #[arg(long)]
    truth_residues: Option<PathBuf>,

    /// Contact distance cutoff for auto-extracting truth residues from holo (Angstroms)
    #[arg(long, default_value = "4.5")]
    contact_cutoff: f32,

    /// Start temperature (K) - cryogenic
    #[arg(long, default_value = "50.0")]
    start_temp: f32,

    /// End temperature (K) - physiological
    #[arg(long, default_value = "300.0")]
    end_temp: f32,

    /// Steps to hold at cold (cryogenic) temperature before ramp
    #[arg(long, default_value = "20000")]
    cold_hold_steps: i32,

    /// Steps for temperature ramp from start_temp to end_temp
    #[arg(long, default_value = "30000")]
    ramp_steps: i32,

    /// Steps to hold at warm (physiological) temperature after ramp
    #[arg(long, default_value = "50000")]
    warm_hold_steps: i32,

    /// Grid spacing in Angstroms
    #[arg(long, default_value = "1.0")]
    grid_spacing: f32,

    /// UV burst energy (kcal/mol)
    #[arg(long, default_value = "5.0")]
    uv_energy: f32,

    /// UV burst interval (timesteps between bursts)
    #[arg(long, default_value = "500")]
    uv_interval: i32,

    /// UV burst dwell time (timesteps per burst)
    #[arg(long, default_value = "10")]
    uv_dwell_steps: i32,

    /// Enable wavelength frequency sweep (cycles through wavelengths)
    #[arg(long)]
    wavelength_sweep: bool,

    /// Minimum wavelength for sweep mode (nm)
    #[arg(long, default_value = "250.0")]
    wavelength_min: f32,

    /// Maximum wavelength for sweep mode (nm)
    #[arg(long, default_value = "300.0")]
    wavelength_max: f32,

    /// Wavelength sweep step size (nm)
    #[arg(long, default_value = "5.0")]
    wavelength_step: f32,

    // ═══════════════════════════════════════════════════════════════════════
    // SNAPSHOT TRIGGERS
    // ═══════════════════════════════════════════════════════════════════════

    /// Delta SASA threshold to trigger activity-based snapshot (Ų)
    #[arg(long, default_value = "50.0")]
    snapshot_sasa_threshold: f32,

    /// Enable spike activity-triggered snapshots
    #[arg(long, default_value = "true")]
    snapshot_on_spike: bool,

    /// Enable UV response-triggered snapshots
    #[arg(long, default_value = "true")]
    snapshot_on_uv: bool,

    /// Enable temperature transition-triggered snapshots (80K, 150K, 200K, 250K, 273K, 300K)
    #[arg(long, default_value = "true")]
    snapshot_on_temp_transition: bool,

    // ═══════════════════════════════════════════════════════════════════════
    // DUAL-STAGE DISTANCE FILTERING
    // ═══════════════════════════════════════════════════════════════════════

    /// Initial (tight) filter distance from protein atoms (Å).
    /// Events beyond this distance are flagged for secondary review.
    #[arg(long, default_value = "8.0")]
    filter_initial_dist: f32,

    /// Global maximum filter distance from protein atoms (Å).
    /// Events beyond this distance are discarded entirely.
    #[arg(long, default_value = "18.0")]
    filter_max_dist: f32,

    // ═══════════════════════════════════════════════════════════════════════
    // CLUSTER DETECTION TUNING
    // ═══════════════════════════════════════════════════════════════════════

    /// Minimum events per cluster (DBSCAN min_samples)
    #[arg(long, default_value = "100")]
    cluster_min_events: usize,

    /// Cluster spatial threshold / epsilon (Å) for DBSCAN
    #[arg(long, default_value = "5.0")]
    cluster_eps: f32,

    /// Minimum residues required per site (post-mapping filter)
    #[arg(long, default_value = "5")]
    cluster_min_residues: usize,

    /// Minimum cluster density (events per Å³)
    #[arg(long, default_value = "0.01")]
    cluster_min_density: f32,

    /// Maximum cluster radius (Å) - reject overly diffuse clusters
    #[arg(long, default_value = "25.0")]
    cluster_max_radius: f32,

    // ═══════════════════════════════════════════════════════════════════════
    // SITE ACCEPTANCE CRITERIA
    // ═══════════════════════════════════════════════════════════════════════

    /// Minimum site persistence (fraction of frames, 0.0-1.0)
    #[arg(long, default_value = "0.002")]
    min_persistence: f32,

    /// Minimum site volume (Å³)
    #[arg(long, default_value = "100.0")]
    min_volume: f32,

    /// Minimum replica agreement (fraction, 0.0-1.0)
    #[arg(long, default_value = "0.3")]
    min_replica_agreement: f32,

    /// Residue query radius for site mapping (Å)
    #[arg(long, default_value = "8.0")]
    residue_query_radius: f32,

    /// CUDA device ID
    #[arg(long, default_value = "0")]
    device: i32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Skip PDF generation
    #[arg(long)]
    no_pdf: bool,

    /// Skip PyMOL session generation (emits .pml script only)
    #[arg(long)]
    no_pymol: bool,

    /// Skip ChimeraX session generation (emits .cxc script only)
    #[arg(long)]
    no_chimerax: bool,

    /// Skip finalize stage (engine output only, no evidence pack)
    #[arg(long)]
    no_finalize: bool,

    /// Skip ablation phases (run only Cryo+UV, no Baseline or Cryo-only)
    /// This makes runs 3x faster but removes ablation comparison data
    #[arg(long)]
    skip_ablation: bool,

    /// Master seed for deterministic operations
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Maximum distance (Å) from any atom for events to be included in voxelization.
    /// Events farther than this from all atoms are filtered out before volume generation.
    #[arg(long, default_value = "15.0")]
    max_event_atom_dist: f32,

    /// Maximum distance (Å) from any atom for events to be included in clustering.
    /// Must be <= residue mapping radius (5Å) + margin to ensure site centroids
    /// are close enough for residue mapping to succeed.
    #[arg(long, default_value = "15.0")]
    cluster_max_event_atom_dist: f32,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run(args) => run_pipeline(args),
        Commands::Finalize(args) => run_finalize(args),
        Commands::Check => check_dependencies(),
        Commands::Version => show_version(),
    }
}

// =============================================================================
// EVENTS.JSONL VALIDATION (MANDATORY)
// =============================================================================

/// Validate that events.jsonl exists, is non-empty, and contains valid data.
/// This is a HARD requirement - no fallback paths.
fn validate_events_file(events_path: &PathBuf) -> Result<usize> {
    use std::io::{BufRead, BufReader};

    // Check file exists
    if !events_path.exists() {
        bail!(
            "FATAL: events.jsonl not found at {}\n\
             The engine must produce this file. No fallback path exists.\n\
             Check engine logs for errors.",
            events_path.display()
        );
    }

    // Check file is non-empty
    let metadata = std::fs::metadata(events_path)
        .with_context(|| format!("Failed to read metadata for {}", events_path.display()))?;

    if metadata.len() == 0 {
        bail!(
            "FATAL: events.jsonl is empty at {}\n\
             The engine ran but produced no events. This is a hard failure.\n\
             Possible causes:\n\
             - No spikes detected during simulation\n\
             - Engine crashed before writing events\n\
             - Incorrect topology or parameters",
            events_path.display()
        );
    }

    // Validate first line is valid JSON with required fields
    let file = std::fs::File::open(events_path)
        .with_context(|| format!("Failed to open {}", events_path.display()))?;
    let reader = BufReader::new(file);

    let mut event_count = 0;
    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result
            .with_context(|| format!("Failed to read line {} of {}", line_num + 1, events_path.display()))?;

        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!(
                "FATAL: Invalid JSON on line {} of {}\n\
                 Line content: {}\n\
                 events.jsonl must contain valid JSONL (one JSON object per line)",
                line_num + 1, events_path.display(), line
            ))?;

        // Validate required fields on first event
        if event_count == 0 {
            let required_fields = ["center_xyz", "volume_a3", "phase", "replicate_id", "frame_idx"];
            for field in &required_fields {
                if json.get(field).is_none() {
                    bail!(
                        "FATAL: events.jsonl missing required field '{}' on line 1\n\
                         Required fields: {:?}\n\
                         This indicates corrupted or incompatible event data.",
                        field, required_fields
                    );
                }
            }
        }

        event_count += 1;
    }

    if event_count == 0 {
        bail!(
            "FATAL: events.jsonl contains no valid events at {}\n\
             File exists but has no parseable event lines.",
            events_path.display()
        );
    }

    log::info!("Validated events.jsonl: {} events", event_count);
    Ok(event_count)
}

// =============================================================================
// PREP STAGE (PDB -> Topology)
// =============================================================================

/// Run prism-prep to generate topology.json from a PDB file.
/// Uses AMBER H-placement by default for pharma-grade quality.
fn run_prep_stage(
    pdb_path: &PathBuf,
    topology_path: &PathBuf,
    use_amber: bool,
    custom_prep_script: Option<&PathBuf>,
) -> Result<()> {
    use std::process::Command;

    // Find prism-prep script
    let prep_script = if let Some(custom) = custom_prep_script {
        custom.clone()
    } else {
        // Try common locations
        let candidates = [
            // Relative to binary
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("../scripts/prep/prism-prep"))),
            // Absolute paths
            Some(PathBuf::from("/home/diddy/Desktop/PRISM4D_RELEASE/scripts/prep/prism-prep")),
            Some(PathBuf::from("./scripts/prep/prism-prep")),
        ];

        candidates
            .into_iter()
            .flatten()
            .find(|p| p.exists())
            .ok_or_else(|| anyhow::anyhow!(
                "prism-prep script not found. Specify with --prep-script or provide --topology"
            ))?
    };

    log::info!("Running prep stage: {} -> {}", pdb_path.display(), topology_path.display());
    log::info!("  AMBER H-placement: {}", if use_amber { "enabled" } else { "disabled" });

    // Ensure output directory exists
    if let Some(parent) = topology_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Build command
    let mut cmd = Command::new("python3");
    cmd.arg(&prep_script)
        .arg(pdb_path)
        .arg(topology_path);

    if use_amber {
        cmd.arg("--use-amber");
    }

    // Execute
    let output = cmd
        .output()
        .with_context(|| format!("Failed to execute prism-prep at {}", prep_script.display()))?;

    // Print stdout/stderr
    if !output.stdout.is_empty() {
        print!("{}", String::from_utf8_lossy(&output.stdout));
    }
    if !output.stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }

    if !output.status.success() {
        bail!(
            "prism-prep failed with exit code: {:?}\n\
             Check that the PDB file is valid and AMBER tools are installed.",
            output.status.code()
        );
    }

    // Validate output was created
    if !topology_path.exists() {
        bail!(
            "prism-prep completed but topology not created at {}\n\
             Check prep script output for errors.",
            topology_path.display()
        );
    }

    log::info!("Prep stage complete: {}", topology_path.display());
    println!();
    Ok(())
}

// =============================================================================
// REAL ENGINE EXECUTION (GPU REQUIRED)
// =============================================================================

/// Run the prism-nhs fused engine and produce events.jsonl.
/// This is the ONLY path - no fallbacks.
#[cfg(feature = "gpu")]
fn run_cryo_uv_engine(
    topology_path: &PathBuf,
    events_path: &PathBuf,
    config: &EngineConfig,
) -> Result<EngineResult> {
    use std::sync::Arc;
    use std::time::Instant;

    log::info!("Loading topology from: {}", topology_path.display());
    let topology = PrismPrepTopology::load(topology_path)
        .with_context(|| format!("Failed to load topology from {}", topology_path.display()))?;

    log::info!(
        "Structure: {} atoms, {} residues, {} chains",
        topology.n_atoms,
        topology.n_residues,
        topology.n_chains
    );

    // Initialize CUDA
    log::info!("Initializing CUDA device {}...", config.device_id);
    let context = CudaContext::new(config.device_id as usize)
        .context("Failed to initialize CUDA. Ensure GPU is available and drivers are installed.")?;

    // Compute grid dimension from structure size
    let (min_pos, max_pos) = topology.bounding_box();
    let padding = 5.0f32;
    let box_size = [
        max_pos[0] - min_pos[0] + 2.0 * padding,
        max_pos[1] - min_pos[1] + 2.0 * padding,
        max_pos[2] - min_pos[2] + 2.0 * padding,
    ];
    let max_dim = box_size[0].max(box_size[1]).max(box_size[2]);
    let grid_dim = ((max_dim / config.grid_spacing).ceil() as usize).min(128).max(32);

    log::info!("Grid: {}^3 voxels at {:.2} A spacing", grid_dim, config.grid_spacing);

    // Create temperature protocol for the engine
    // Note: prism-nhs uses ramp_steps + hold_steps internally, we compute from our explicit phases
    let total_cryo_steps = config.total_cryo_steps();
    let temp_protocol = NhsTemperatureProtocol {
        start_temp: config.start_temp,
        end_temp: config.end_temp,
        // Engine ramp_steps = our cold_hold + ramp (transition period)
        ramp_steps: config.cold_hold_steps + config.ramp_steps,
        // Engine hold_steps = our warm_hold
        hold_steps: config.warm_hold_steps,
        current_step: 0,
    };

    log::info!(
        "Temperature protocol: {:.1}K -> {:.1}K",
        config.start_temp, config.end_temp
    );
    log::info!(
        "  Phases: cold_hold={}, ramp={}, warm_hold={} (total={})",
        config.cold_hold_steps, config.ramp_steps, config.warm_hold_steps, total_cryo_steps
    );

    // Create UV probe configuration with wavelength sweep
    let aromatics = topology.aromatic_residues();
    let has_250nm = config.wavelengths.iter().any(|&w| (w - 250.0).abs() < 1.0);
    let uv_config = UvProbeConfig {
        enabled: true,
        burst_energy: config.uv_energy,
        burst_interval: config.uv_interval,
        burst_duration: 10,
        target_sequence: (0..aromatics.len()).collect(),
        current_target: 0,
        timestep_counter: 0,
        // Enable frequency hopping with user-specified wavelengths
        frequency_hopping_enabled: true,
        scan_wavelengths: config.wavelengths.clone(),
        dwell_steps: 1000, // Steps per wavelength before hopping
        // Enable disulfide targeting if 250nm is in the sweep
        target_disulfides: has_250nm,
        ..Default::default()
    };

    log::info!("UV targets: {} aromatic residues", aromatics.len());
    log::info!("UV wavelength sweep: {:?} nm (disulfides: {})", config.wavelengths, has_250nm);
    log::info!("UV Protocol: energy={:.1} kcal/mol, interval={}", config.uv_energy, config.uv_interval);

    // Create fused engine
    log::info!("Creating NHS-AMBER Fused Engine...");
    let mut engine = NhsAmberFusedEngine::new(
        context,
        &topology,
        grid_dim,
        config.grid_spacing,
    )?;

    engine.set_temperature_protocol(temp_protocol.clone())?;
    engine.set_uv_config(uv_config);

    // Create event writers for streaming output
    let mut event_writer = EventWriter::new(events_path)
        .with_context(|| format!("Failed to create event writer at {}", events_path.display()))?;

    // Create spike event writer for raw spike data (enables UV enrichment calculation)
    let spike_events_path = events_path.with_file_name("spike_events.jsonl");
    let spike_writer_file = std::fs::File::create(&spike_events_path)
        .with_context(|| format!("Failed to create spike_events.jsonl at {}", spike_events_path.display()))?;
    let mut spike_writer = std::io::BufWriter::new(spike_writer_file);
    log::info!("  Spike events will be saved to: {}", spike_events_path.display());

    // Run ablation protocol: Baseline -> CryoOnly -> CryoUv
    // If skip_ablation is true, only run CryoUv phase (3x faster)
    let all_phases = [
        (AblationPhase::Baseline, "Baseline (no cryo, no UV)", 0usize),
        (AblationPhase::CryoOnly, "Cryo-only (cold, no UV)", 1usize),
        (AblationPhase::CryoUv, "Cryo+UV (cold + UV bursts)", 2usize),
    ];

    let phases_to_run: Vec<_> = if config.skip_ablation {
        log::info!("Ablation SKIPPED: running only Cryo+UV phase");
        all_phases.iter().filter(|(p, _, _)| *p == AblationPhase::CryoUv).collect()
    } else {
        all_phases.iter().collect()
    };

    let start_time = Instant::now();
    let mut total_events = 0usize;
    let mut phase_events = [0usize; 3];

    // Tiered counters for logging
    let mut raw_candidates_found = 0usize;
    let mut events_written_total = 0usize;
    let mut events_written_last_10k = 0usize;
    let mut uv_bursts_applied_last_10k = 0usize;
    let mut max_candidate_volume_last_10k = 0.0f32;
    let mut max_spike_intensity_last_10k = 0.0f32;

    for (replicate_id, _) in (0..config.replicates).enumerate() {
        log::info!("=== Replicate {}/{} ===", replicate_id + 1, config.replicates);

        for (phase, phase_name, phase_idx) in &phases_to_run {
            log::info!("  Phase: {}", phase_name);

            // Configure engine for this phase
            let enable_uv = *phase == AblationPhase::CryoUv;
            engine.set_uv_enabled(enable_uv);

            // Reset temperature protocol for each phase
            // Baseline: runs at constant 300K for warm_hold_steps only
            // Cryo phases: run full cold_hold -> ramp -> warm_hold sequence
            engine.set_temperature_protocol(NhsTemperatureProtocol {
                start_temp: if *phase == AblationPhase::Baseline { 300.0 } else { config.start_temp },
                end_temp: config.end_temp,
                ramp_steps: if *phase == AblationPhase::Baseline { 0 } else { config.cold_hold_steps + config.ramp_steps },
                hold_steps: config.warm_hold_steps,
                current_step: 0,
            })?;

            let phase_total_steps = if *phase == AblationPhase::Baseline {
                config.warm_hold_steps // Baseline runs at constant 300K (warm only)
            } else {
                total_cryo_steps // Cryo phases run full cold+ramp+warm
            };

            // Run this phase
            for step in 0..phase_total_steps {
                let result = engine.step()?;

                // Track UV bursts
                if result.uv_burst_active {
                    uv_bursts_applied_last_10k += 1;
                }

                // Download and emit spike events when available
                // The engine syncs every 100 steps, so spike_count > 0 means we have data
                if result.spike_count > 0 {
                    // Download full spike events from GPU
                    let gpu_spikes = engine.download_full_spike_events(result.spike_count)?;
                    raw_candidates_found += gpu_spikes.len();

                    // Compute temperature phase based on explicit schedule boundaries
                    let temp_phase = if *phase == AblationPhase::Baseline {
                        TempPhase::Warm
                    } else {
                        if step < config.cold_end_step() {
                            TempPhase::Cold
                        } else if step < config.ramp_end_step() {
                            TempPhase::Ramp
                        } else {
                            TempPhase::Warm
                        }
                    };

                    for spike in &gpu_spikes {
                        // Copy packed struct fields to avoid unaligned reference issues
                        let spike_timestep = spike.timestep;
                        let spike_intensity = spike.intensity;
                        let spike_position = spike.position;
                        let spike_n_residues = spike.n_residues;
                        let spike_nearby_residues = spike.nearby_residues;

                        // Save raw spike event for UV enrichment calculation
                        let nearby_res_vec: Vec<i32> = spike_nearby_residues[..spike_n_residues.min(8) as usize]
                            .iter()
                            .filter(|&&r| r >= 0)
                            .copied()
                            .collect();

                        let raw_spike = RawSpikeEvent {
                            timestep: spike_timestep,
                            position: spike_position,
                            nearby_residues: nearby_res_vec,
                            intensity: spike_intensity,
                            phase: *phase,
                            replicate_id,
                        };

                        // Write raw spike event (for true UV enrichment calculation)
                        let spike_json = serde_json::to_string(&raw_spike)?;
                        writeln!(spike_writer, "{}", spike_json)?;

                        // Track statistics
                        let volume = estimate_pocket_volume(spike_n_residues, spike_intensity);
                        max_candidate_volume_last_10k = max_candidate_volume_last_10k.max(volume);
                        max_spike_intensity_last_10k = max_spike_intensity_last_10k.max(spike_intensity);

                        // Build residue list from nearby_residues
                        let n_res = (spike_n_residues as usize).min(8);
                        let residues: Vec<u32> = spike_nearby_residues[..n_res]
                            .iter()
                            .filter(|&&r| r >= 0)
                            .map(|&r| r as u32)
                            .collect();

                        let event = PocketEvent {
                            center_xyz: spike_position,
                            volume_a3: volume,
                            spike_count: 1,
                            phase: *phase,
                            temp_phase,
                            replicate_id,
                            frame_idx: step as usize,
                            residues,
                            confidence: compute_spike_confidence(spike_intensity, result.temperature),
                            wavelength_nm: if enable_uv && result.uv_burst_active {
                                result.current_wavelength_nm
                            } else {
                                None
                            },
                        };
                        event_writer.write_event(&event)?;
                        total_events += 1;
                        events_written_total += 1;
                        events_written_last_10k += 1;
                        phase_events[*phase_idx] += 1;
                    }
                }

                // Progress logging every 10k steps with tiered counters
                if step > 0 && step % 10000 == 0 {
                    log::info!(
                        "    Step {}/{}: T={:.1}K | candidates={} events={} uv_bursts={} max_vol={:.1}Å³ max_int={:.2}",
                        step, phase_total_steps, result.temperature,
                        raw_candidates_found, events_written_last_10k,
                        uv_bursts_applied_last_10k, max_candidate_volume_last_10k, max_spike_intensity_last_10k
                    );
                    // Reset interval counters
                    events_written_last_10k = 0;
                    uv_bursts_applied_last_10k = 0;
                    max_candidate_volume_last_10k = 0.0;
                    max_spike_intensity_last_10k = 0.0;
                }
            }
        }
    }

    event_writer.flush()?;
    spike_writer.flush()?;
    log::info!("  ✓ Spike events saved for UV enrichment calculation");

    let elapsed = start_time.elapsed();
    let n_phases = if config.skip_ablation { 1 } else { 3 };
    let steps_per_sec = (total_cryo_steps as usize * config.replicates * n_phases) as f64 / elapsed.as_secs_f64();

    log::info!("Engine complete: raw_candidates={} events_written={} in {:.2}s ({:.0} steps/s)",
        raw_candidates_found, events_written_total, elapsed.as_secs_f64(), steps_per_sec);

    Ok(EngineResult {
        total_events,
        events_by_phase: phase_events,
        elapsed_seconds: elapsed.as_secs_f64(),
        steps_per_second: steps_per_sec,
    })
}

#[cfg(feature = "gpu")]
fn estimate_pocket_volume(n_residues: i32, spike_intensity: f32) -> f32 {
    // Estimate volume from residue count and intensity
    // Each residue contributes ~35Å³ to accessible pocket volume
    // Intensity modulates the "openness" factor
    let base_volume = 50.0_f32; // Minimum cavity
    let per_residue_volume = 35.0_f32;
    let openness_factor = 1.0 + (spike_intensity / 15.0).min(1.5);

    let estimated = base_volume + (n_residues as f32 * per_residue_volume * openness_factor);
    estimated.clamp(100.0, 2000.0)
}

#[cfg(feature = "gpu")]
fn compute_spike_confidence(intensity: f32, temperature: f32) -> f32 {
    // Confidence is higher at lower temperatures (cleaner signal)
    // and higher intensities
    let temp_factor = (300.0 / temperature).min(2.0);
    let intensity_factor = (intensity / 10.0).min(1.0);
    (0.5 + 0.3 * temp_factor * intensity_factor).min(1.0)
}

#[cfg(not(feature = "gpu"))]
fn run_cryo_uv_engine(
    _topology_path: &PathBuf,
    _events_path: &PathBuf,
    _config: &EngineConfig,
) -> Result<EngineResult> {
    bail!(
        "FATAL: prism4d run requires GPU support.\n\
         This binary was compiled without the 'gpu' feature.\n\
         \n\
         To enable GPU support, rebuild with:\n\
         \n\
         cargo build --release --features gpu -p prism-report\n\
         \n\
         There is NO fallback path. Real engine execution is mandatory."
    );
}

#[derive(Debug, Clone)]
struct EngineConfig {
    start_temp: f32,
    end_temp: f32,
    /// Steps to hold at cold (cryogenic) temperature before ramp
    cold_hold_steps: i32,
    /// Steps for temperature ramp from start_temp to end_temp
    ramp_steps: i32,
    /// Steps to hold at warm (physiological) temperature after ramp
    warm_hold_steps: i32,
    grid_spacing: f32,
    uv_energy: f32,
    uv_interval: i32,
    device_id: i32,
    replicates: usize,
    /// UV wavelengths for frequency-hopping sweep (nm)
    wavelengths: Vec<f32>,
    /// Skip ablation phases (run only Cryo+UV)
    skip_ablation: bool,
}

impl EngineConfig {
    /// Total steps for the complete cryo protocol (cold + ramp + warm)
    fn total_cryo_steps(&self) -> i32 {
        self.cold_hold_steps + self.ramp_steps + self.warm_hold_steps
    }

    /// Get the step boundary where cold phase ends and ramp begins
    fn cold_end_step(&self) -> i32 {
        self.cold_hold_steps
    }

    /// Get the step boundary where ramp phase ends and warm begins
    fn ramp_end_step(&self) -> i32 {
        self.cold_hold_steps + self.ramp_steps
    }
}

#[derive(Debug)]
struct EngineResult {
    total_events: usize,
    events_by_phase: [usize; 3],
    elapsed_seconds: f64,
    steps_per_second: f64,
}

// =============================================================================
// FINALIZE STAGE (REQUIRES VALID EVENTS.JSONL)
// =============================================================================

/// Run finalize stage only (standalone mode)
fn run_finalize(args: FinalizeArgs) -> Result<()> {
    println!("============================================================");
    println!("  PRISM4D v{} - Finalize Stage", prism_report::VERSION);
    println!("============================================================");
    println!();

    // MANDATORY: Validate events file
    let event_count = validate_events_file(&args.events)?;

    // MANDATORY: Validate topology file exists
    if !args.topology.exists() {
        bail!(
            "FATAL: Topology file not found: {}\n\
             The --topology argument is MANDATORY. There is NO fallback to PDB parsing.\n\
             Generate with: prism-prep your_structure.pdb output.json --use-amber --strict",
            args.topology.display()
        );
    }

    println!("Events:   {} ({} events)", args.events.display(), event_count);
    println!("Topology: {}", args.topology.display());
    println!("Output:   {}", args.out.display());
    println!("Seed:     {}", args.seed);
    println!();

    // Save PDB path for later use in seed config (before it gets moved)
    let pdb_path_str = args.pdb.as_ref().map(|p| p.display().to_string()).unwrap_or_default();

    // Build minimal config - PDB is for visuals only, not for spatial metrics
    let config = ReportConfig {
        input_pdb: args.pdb.unwrap_or_else(|| args.out.join("reference.pdb")),
        output_dir: args.out.clone(),
        holo_pdb: args.holo.clone(),          // Tier1 correlation (optional)
        truth_residues: args.truth_residues.clone(), // Tier2 correlation (optional)
        contact_cutoff: Some(args.contact_cutoff),   // Auto-truth extraction cutoff
        replicates: args.replicates, // For replica agreement filtering
        output_formats: prism_report::config::OutputFormats {
            html: true,
            pdf: !args.no_pdf,
            json: true,
            csv: true,
            pymol: !args.no_pymol,
            chimerax: !args.no_chimerax,
            figures: true,
            mrc_volumes: true,
        },
        site_detection: prism_report::config::SiteDetectionConfig {
            min_cluster_size: args.cluster_min_events,
            cluster_threshold: args.cluster_eps,
            min_volume: args.min_volume,
            min_persistence: args.min_persistence,
            min_confidence: 0.3,
            residue_query_radius_a: args.residue_query_radius,
            min_replica_agreement: args.min_replica_agreement,
        },
        ..Default::default()
    };

    if args.replicates > 1 && args.min_replica_agreement > 0.0 {
        log::info!("Replica agreement filtering ENABLED: requiring sites in ≥{:.0}% of {} replicates",
                   args.min_replica_agreement * 100.0, args.replicates);
    } else if args.replicates > 1 {
        log::info!("Replica agreement filtering DISABLED (min_replica_agreement=0.0)");
    }

    // Create finalize stage with mandatory topology path
    // Uses two-tier filtering: filter_initial_dist for tight pass, filter_max_dist for global max
    let stage = FinalizeStage::new_with_topology_full(
        config,
        args.events.clone(),
        args.topology.clone(),
        args.seed,
        args.filter_initial_dist,
        args.filter_max_dist,
    )?;

    let start_time = std::time::Instant::now();
    let result = stage.run()?;
    let elapsed = start_time.elapsed();

    // Save seed config for reproducibility
    if args.save_seed {
        let events_per_sec = if elapsed.as_secs_f64() > 0.0 {
            event_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let seed_config = SeedConfig {
            seed: args.seed,
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: prism_report::PRISM4D_RELEASE.to_string(),
            input_pdb: pdb_path_str.clone(),
            temperature: TemperatureParams {
                start_temp: 50.0,  // Default - engine params not available in finalize-only mode
                end_temp: 300.0,
                cold_hold_steps: 20000,
                ramp_steps: 30000,
                warm_hold_steps: 50000,
            },
            uv: UvParams {
                energy: 5.0,
                interval: 500,
                dwell_steps: 10,
                wavelengths: vec![258.0, 274.0, 280.0],
                wavelength_sweep: false,
                wavelength_min: 250.0,
                wavelength_max: 300.0,
                wavelength_step: 5.0,
            },
            filtering: FilterParams {
                initial_dist: args.filter_initial_dist,
                max_dist: args.filter_max_dist,
            },
            clustering: ClusterParams {
                min_events: args.cluster_min_events,
                eps: args.cluster_eps,
                min_residues: args.cluster_min_residues,
                residue_query_radius: args.residue_query_radius,
            },
            acceptance: AcceptanceParams {
                min_persistence: args.min_persistence,
                min_volume: args.min_volume,
                min_replica_agreement: args.min_replica_agreement,
                replicates: args.replicates,
            },
            stats: Some(RunStats {
                total_events: event_count,
                sites_detected: result.n_sites,
                druggable_sites: result.n_druggable,
                runtime_seconds: elapsed.as_secs_f64(),
                events_per_second: events_per_sec,
            }),
        };

        let seed_path = args.out.join("seed_config.json");
        if let Err(e) = seed_config.save(&seed_path) {
            log::warn!("Failed to save seed config: {}", e);
        } else {
            log::info!("Seed config saved to: {}", seed_path.display());
        }
    }

    println!();
    println!("============================================================");
    println!("  FINALIZE COMPLETE");
    println!("============================================================");
    println!();
    println!("Results: {}", result.output_dir.display());
    println!("Sites:   {} ({} druggable)", result.n_sites, result.n_druggable);
    println!("Seed:    {} (saved to seed_config.json)", args.seed);
    println!("Runtime: {:.2}s", elapsed.as_secs_f64());
    println!();

    Ok(())
}

// =============================================================================
// FULL PIPELINE (ENGINE + FINALIZE)
// =============================================================================

fn run_pipeline(args: RunArgs) -> Result<()> {
    println!("============================================================");
    println!("  PRISM4D v{} - Cryptic Binding Site Detection", prism_report::VERSION);
    println!("  Mode: {}", args.mode);
    println!("============================================================");
    println!();

    // Validate mode
    if args.mode != "cryo-uv" {
        bail!(
            "Unsupported mode: '{}'. Currently supported: cryo-uv",
            args.mode
        );
    }

    // Validate PDB exists
    if !args.pdb.exists() {
        bail!("Input PDB not found: {}", args.pdb.display());
    }

    // Determine topology path - auto-run prep if not provided
    let topology_path = if let Some(ref topo) = args.topology {
        if !topo.exists() {
            bail!(
                "Topology file not found: {}\n\
                 Generate with: prism-prep your_structure.pdb output.json --use-amber",
                topo.display()
            );
        }
        topo.clone()
    } else {
        // Auto-run prep stage
        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│  STAGE 1: PREP (auto-generating topology)                   │");
        println!("└─────────────────────────────────────────────────────────────┘");

        let topology_path = args.out.join("topology.json");
        run_prep_stage(&args.pdb, &topology_path, !args.no_amber, args.prep_script.as_ref())?;
        topology_path
    };

    // Check optional inputs
    if let Some(ref holo) = args.holo {
        if !holo.exists() {
            bail!("Holo PDB not found: {}", holo.display());
        }
    }
    if let Some(ref truth) = args.truth_residues {
        if !truth.exists() {
            bail!("Truth residues file not found: {}", truth.display());
        }
    }

    // Check dependencies and warn
    check_dependencies_quiet(&args)?;

    // Create output directory
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("Failed to create output directory: {}", args.out.display()))?;

    // Build report configuration
    let report_config = ReportConfig {
        input_pdb: args.pdb.clone(),
        output_dir: args.out.clone(),
        replicates: args.replicates,
        wavelengths: args.wavelengths.clone(),
        holo_pdb: args.holo.clone(),
        truth_residues: args.truth_residues.clone(),
        contact_cutoff: Some(args.contact_cutoff),  // Auto-truth extraction cutoff
        temperature_protocol: TemperatureProtocol {
            start_temp: args.start_temp,
            end_temp: args.end_temp,
            cold_hold_steps: args.cold_hold_steps,
            ramp_steps: args.ramp_steps,
            warm_hold_steps: args.warm_hold_steps,
        },
        ablation: AblationConfig::default(),
        output_formats: prism_report::config::OutputFormats {
            html: true,
            pdf: !args.no_pdf,
            json: true,
            csv: true,
            pymol: !args.no_pymol,
            chimerax: !args.no_chimerax,
            figures: true,
            mrc_volumes: true,
        },
        device_id: args.device,
        verbose: args.verbose,
        ..Default::default()
    };

    println!("Configuration:");
    println!("  Topology:    {}", topology_path.display());
    println!("  PDB:         {}", report_config.input_pdb.display());
    println!("  Output:      {}", report_config.output_dir.display());
    println!("  Replicates:  {}", report_config.replicates);
    println!("  Wavelengths: {:?} nm", report_config.wavelengths);
    println!("  Temp range:  {}K -> {}K", args.start_temp, args.end_temp);
    println!("  Seed:        {}", args.seed);
    if let Some(ref holo) = report_config.holo_pdb {
        println!("  Holo:        {}", holo.display());
    }
    if let Some(ref truth) = report_config.truth_residues {
        println!("  Truth:       {}", truth.display());
    }
    println!();

    // =========================================================================
    // STAGE 1: Engine - Run MD stepping, produce events.jsonl
    // =========================================================================
    println!("[Stage 1/2] Running cryo-UV MD engine...");

    let events_path = report_config.output_dir.join("events.jsonl");

    let engine_config = EngineConfig {
        start_temp: args.start_temp,
        end_temp: args.end_temp,
        cold_hold_steps: args.cold_hold_steps,
        ramp_steps: args.ramp_steps,
        warm_hold_steps: args.warm_hold_steps,
        grid_spacing: args.grid_spacing,
        uv_energy: args.uv_energy,
        uv_interval: args.uv_interval,
        device_id: args.device,
        replicates: args.replicates,
        wavelengths: args.wavelengths.clone(),
        skip_ablation: args.skip_ablation,
    };

    // Run the REAL engine (no fallback)
    let engine_result = run_cryo_uv_engine(&topology_path, &events_path, &engine_config)?;

    // MANDATORY: Validate events.jsonl was produced correctly
    let event_count = validate_events_file(&events_path)?;

    println!("  Engine complete:");
    println!("    Events:     {}", event_count);
    println!("    Baseline:   {}", engine_result.events_by_phase[0]);
    println!("    Cryo-only:  {}", engine_result.events_by_phase[1]);
    println!("    Cryo+UV:    {}", engine_result.events_by_phase[2]);
    println!("    Time:       {:.2}s ({:.0} steps/s)",
        engine_result.elapsed_seconds, engine_result.steps_per_second);
    println!();

    // =========================================================================
    // STAGE 2: Finalize - Generate evidence pack from events
    // =========================================================================
    if args.no_finalize {
        println!("[Stage 2/2] Finalize stage skipped (--no-finalize)");
        println!();
        println!("Engine output available at: {}", report_config.output_dir.display());
        println!("To generate evidence pack, run:");
        println!("  prism4d finalize --events {} --out {}",
            events_path.display(), report_config.output_dir.display());
        return Ok(());
    }

    println!("[Stage 2/2] Running finalize stage...");

    // Use new_with_topology_full with explicit topology path (mandatory)
    // Two-tier filtering: max_event_atom_dist for voxelization, cluster_max_event_atom_dist for clustering
    let stage = FinalizeStage::new_with_topology_full(
        report_config.clone(),
        events_path,
        topology_path.clone(),
        args.seed,
        args.max_event_atom_dist,
        args.cluster_max_event_atom_dist,
    )?;
    let result = stage.run()?;

    // Print summary
    println!();
    println!("============================================================");
    println!("  PIPELINE COMPLETE");
    println!("============================================================");
    println!();
    println!("Results saved to: {}", result.output_dir.display());
    println!();
    println!("Sites detected:    {} ({} druggable)", result.n_sites, result.n_druggable);
    println!("Cryo contrast:     {}", if result.cryo_significant { "SIGNIFICANT" } else { "not significant" });
    println!("UV response:       {}", if result.uv_significant { "SIGNIFICANT" } else { "not significant" });
    println!();

    // Verify output contract
    verify_output_contract(&result.output_dir)?;

    Ok(())
}

// =============================================================================
// DEPENDENCY CHECKS
// =============================================================================

fn check_dependencies() -> Result<()> {
    println!("PRISM4D Dependency Check");
    println!("========================");
    println!();

    let mut all_ok = true;

    // GPU
    print!("GPU:       ");
    #[cfg(feature = "gpu")]
    {
        match cudarc::driver::CudaContext::new(0) {
            Ok(_) => println!("OK (CUDA available)"),
            Err(e) => {
                println!("ERROR: {}", e);
                all_ok = false;
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        println!("NOT COMPILED (rebuild with --features gpu)");
        all_ok = false;
    }

    // PyMOL
    print!("PyMOL:     ");
    match find_pymol() {
        Some(path) => println!("OK ({})", path.display()),
        None => {
            println!("NOT FOUND (optional)");
        }
    }

    // ChimeraX
    print!("ChimeraX:  ");
    match find_chimerax() {
        Some(path) => println!("OK ({})", path.display()),
        None => {
            println!("NOT FOUND (optional)");
        }
    }

    // PDF renderer
    print!("PDF:       ");
    match find_pdf_renderer() {
        Some((name, path)) => println!("OK ({}: {})", name, path.display()),
        None => {
            println!("NOT FOUND (optional)");
        }
    }

    println!();

    if !all_ok {
        println!("CRITICAL: GPU support is REQUIRED for prism4d run.");
        println!();
        #[cfg(not(feature = "gpu"))]
        {
            println!("Rebuild with GPU support:");
            println!("  cargo build --release --features gpu -p prism-report");
        }
    } else {
        println!("All critical dependencies available!");
    }

    Ok(())
}

fn check_dependencies_quiet(args: &RunArgs) -> Result<()> {
    if !args.no_pymol && find_pymol().is_none() {
        log::warn!("PyMOL not found. Use --no-pymol to suppress this warning.");
    }

    if !args.no_chimerax && find_chimerax().is_none() {
        log::warn!("ChimeraX not found. Use --no-chimerax to suppress this warning.");
    }

    if !args.no_pdf && find_pdf_renderer().is_none() {
        log::warn!("PDF renderer not found. Use --no-pdf to skip PDF generation.");
    }

    Ok(())
}

fn show_version() -> Result<()> {
    println!("prism4d {}", prism_report::VERSION);
    println!("prism-report {}", prism_report::VERSION);
    println!();
    println!("Build info:");
    println!("  Platform: {}-{}", std::env::consts::OS, std::env::consts::ARCH);
    #[cfg(feature = "gpu")]
    println!("  GPU:      enabled");
    #[cfg(not(feature = "gpu"))]
    println!("  GPU:      disabled (rebuild with --features gpu)");

    Ok(())
}

/// Verify all required output files exist
fn verify_output_contract(output_dir: &PathBuf) -> Result<()> {
    println!("Verifying output contract...");

    let required_files = [
        "report.html",
        "summary.json",
        "correlation.csv",
    ];

    let required_dirs = [
        "sites",
        "volumes",
        "trajectories",
        "provenance",
    ];

    let provenance_files = [
        "provenance/manifest.json",
        "provenance/versions.json",
        "provenance/seeds.json",
        "provenance/params.json",
    ];

    let mut missing = Vec::new();

    for file in &required_files {
        let path = output_dir.join(file);
        if !path.exists() {
            missing.push(file.to_string());
        }
    }

    for dir in &required_dirs {
        let path = output_dir.join(dir);
        if !path.is_dir() {
            missing.push(format!("{}/", dir));
        }
    }

    for file in &provenance_files {
        let path = output_dir.join(file);
        if !path.exists() {
            missing.push(file.to_string());
        }
    }

    if !missing.is_empty() {
        println!();
        println!("WARNING: Missing required outputs:");
        for m in &missing {
            println!("  - {}", m);
        }
        println!();
    }

    println!();
    println!("Output Contract Checklist:");
    println!("  [{}] report.html", if output_dir.join("report.html").exists() { "x" } else { " " });
    println!("  [{}] report.pdf", if output_dir.join("report.pdf").exists() { "x" } else { " " });
    println!("  [{}] summary.json", if output_dir.join("summary.json").exists() { "x" } else { " " });
    println!("  [{}] correlation.csv", if output_dir.join("correlation.csv").exists() { "x" } else { " " });
    println!("  [{}] sites/", if output_dir.join("sites").is_dir() { "x" } else { " " });
    println!("  [{}] volumes/", if output_dir.join("volumes").is_dir() { "x" } else { " " });
    println!("  [{}] trajectories/", if output_dir.join("trajectories").is_dir() { "x" } else { " " });
    println!("  [{}] provenance/", if output_dir.join("provenance").is_dir() { "x" } else { " " });

    Ok(())
}
