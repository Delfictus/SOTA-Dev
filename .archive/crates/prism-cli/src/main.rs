//! PRISM-Fold CLI entry point.
//!
//! Phase Resonance Integrated Solver Machine for Molecular Folding
//! GPU-accelerated graph coloring and ligand binding site prediction.

use anyhow::Result;
use clap::Parser;
use prism_core::{Graph, WarmstartConfig};
use prism_fluxnet::{RLConfig, UniversalRLController};
use prism_pipeline::{PipelineConfig, PipelineOrchestrator};
use std::io::Write;
use std::path::Path;

/// PRISM-Fold version from Cargo.toml
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(name = "prism-cli")]
#[command(version = VERSION)]
#[command(about = "PRISM-Fold: GPU-accelerated graph coloring with FluxNet RL", long_about = None)]
struct Args {
    /// Input graph file path (required for coloring mode)
    #[arg(short, long)]
    input: Option<String>,

    /// Number of vertices (for testing)
    #[arg(long, default_value = "10")]
    vertices: usize,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    // ========================================================================
    // Mode Selection
    // ========================================================================
    /// Execution mode: coloring (default), biomolecular, materials, mec-only
    ///
    /// - coloring: Standard graph coloring pipeline (default)
    /// - biomolecular: Protein structure prediction and drug binding
    /// - materials: Materials discovery and property prediction
    /// - mec-only: Molecular dynamics diagnostics (MEC phase only)
    ///
    /// Example: --mode biomolecular
    #[arg(long, default_value = "coloring")]
    mode: String,

    /// FASTA sequence file for biomolecular mode
    ///
    /// Example: --sequence benchmarks/biomolecular/nipah_glycoprotein.fasta
    #[arg(long)]
    sequence: Option<String>,

    /// Ligand SMILES string for biomolecular mode
    ///
    /// Example: --ligand "CC(C)Oc1ccc(C(=O)N[C@H](C(=O)O)C(C)C)cc1"
    #[arg(long)]
    ligand: Option<String>,

    // ========================================================================
    // Warmstart Configuration
    // ========================================================================
    /// Enable warmstart system with priors from Phase 0
    ///
    /// When enabled, Phase 1-7 solvers receive probabilistic color priors
    /// derived from flux reservoir dynamics, ensemble methods, and curriculum learning.
    ///
    /// Example: --warmstart
    #[arg(long, default_value = "false")]
    warmstart: bool,

    /// Warmstart flux weight (reservoir prior contribution)
    ///
    /// Controls the influence of Phase 0 flux reservoir dynamics on color priors.
    /// Higher values favor neuromorphic reservoir states.
    ///
    /// Constraint: flux_weight + ensemble_weight + random_weight = 1.0
    ///
    /// Example: --warmstart-flux-weight 0.5
    #[arg(long, default_value = "0.4")]
    warmstart_flux_weight: f32,

    /// Warmstart ensemble weight (structural anchor contribution)
    ///
    /// Controls the influence of ensemble methods (greedy, DSATUR, etc.) on priors.
    /// Higher values favor structural heuristics.
    ///
    /// Constraint: flux_weight + ensemble_weight + random_weight = 1.0
    ///
    /// Example: --warmstart-ensemble-weight 0.5
    #[arg(long, default_value = "0.4")]
    warmstart_ensemble_weight: f32,

    /// Warmstart random weight (exploration contribution)
    ///
    /// Controls the influence of uniform random priors on color distribution.
    /// Higher values increase exploration and entropy.
    ///
    /// Constraint: flux_weight + ensemble_weight + random_weight = 1.0
    ///
    /// Example: --warmstart-random-weight 0.3
    #[arg(long, default_value = "0.2")]
    warmstart_random_weight: f32,

    /// Warmstart anchor fraction (0.0-1.0)
    ///
    /// Fraction of vertices to designate as structural anchors with deterministic
    /// color assignments. Anchors reduce search space and guide solver convergence.
    ///
    /// Target: 0.10 for DSJC250 (25 anchors), 0.15 for dense graphs.
    ///
    /// Example: --warmstart-anchor-fraction 0.15
    #[arg(long, default_value = "0.10")]
    warmstart_anchor_fraction: f32,

    /// Warmstart maximum colors in prior distribution
    ///
    /// Defines the size of the color probability vector for each vertex.
    /// Should be >= expected chromatic number.
    ///
    /// Example: --warmstart-max-colors 100
    #[arg(long, default_value = "50")]
    warmstart_max_colors: usize,

    /// Path to curriculum profile catalog (optional)
    ///
    /// JSON file containing curriculum profiles for graph classes.
    /// If not specified, uses default curriculum selection heuristics.
    ///
    /// Example: --warmstart-curriculum-path profiles/curriculum/catalog.json
    #[arg(long)]
    warmstart_curriculum_path: Option<String>,

    // ========================================================================
    // FluxNet RL Configuration
    // ========================================================================
    /// Path to pretrained FluxNet Q-table (binary format)
    ///
    /// Loads pretrained Q-table for FluxNet RL controller.
    /// Q-tables can be trained using the fluxnet_train binary:
    ///   cargo run --release --bin fluxnet_train -- <graph.col> <epochs> <output.bin>
    ///
    /// This enables warmstart for RL, allowing the controller to start with
    /// learned policies rather than random initialization.
    ///
    /// Example: --fluxnet-qtable profiles/curriculum/qtable_dsjc250.bin
    #[arg(long)]
    fluxnet_qtable: Option<String>,

    /// FluxNet RL epsilon (exploration rate)
    ///
    /// Controls exploration vs exploitation trade-off.
    /// Lower values favor exploitation (greedy action selection).
    /// Higher values increase exploration (random action selection).
    ///
    /// Range: [0.0, 1.0], typical: 0.1-0.3
    ///
    /// Example: --fluxnet-epsilon 0.2
    #[arg(long, default_value = "0.2")]
    fluxnet_epsilon: f64,

    /// FluxNet RL learning rate (alpha)
    ///
    /// Controls Q-value update step size.
    /// Higher values learn faster but may be unstable.
    ///
    /// Range: [0.0, 1.0], typical: 0.05-0.2
    ///
    /// Example: --fluxnet-alpha 0.1
    #[arg(long, default_value = "0.1")]
    fluxnet_alpha: f64,

    /// FluxNet RL discount factor (gamma)
    ///
    /// Controls importance of future rewards.
    /// Higher values prioritize long-term planning.
    ///
    /// Range: [0.0, 1.0], typical: 0.9-0.99
    ///
    /// Example: --fluxnet-gamma 0.95
    #[arg(long, default_value = "0.95")]
    fluxnet_gamma: f64,

    // ========================================================================
    // CMA-ES Configuration
    // ========================================================================
    /// Enable CMA-ES evolutionary optimization phase
    ///
    /// When enabled, uses Covariance Matrix Adaptation Evolution Strategy
    /// to optimize graph coloring via transfer entropy minimization.
    ///
    /// Example: --enable-cma
    #[arg(long, default_value = "false")]
    enable_cma: bool,

    /// CMA-ES population size
    ///
    /// Number of candidate solutions per generation.
    /// Larger populations explore more thoroughly but increase computation.
    ///
    /// Range: [4, 200], typical: 20-100 depending on graph size
    ///
    /// Example: --cma-population-size 50
    #[arg(long, default_value = "50")]
    cma_population_size: usize,

    /// CMA-ES maximum generations
    ///
    /// Maximum number of evolutionary generations to run.
    /// Can terminate early if convergence criteria are met.
    ///
    /// Range: [10, 10000], typical: 100-1000
    ///
    /// Example: --cma-generations 500
    #[arg(long, default_value = "100")]
    cma_generations: usize,

    /// CMA-ES initial step size (sigma)
    ///
    /// Initial standard deviation for sampling solutions.
    /// Affects exploration range - larger values explore more broadly.
    ///
    /// Range: [0.01, 2.0], typical: 0.3-0.7
    ///
    /// Example: --cma-sigma 0.5
    #[arg(long, default_value = "0.5")]
    cma_sigma: f32,

    /// CMA-ES target fitness (optional)
    ///
    /// Stop optimization if this fitness level is achieved.
    /// Useful for problems with known optimal solutions.
    ///
    /// Example: --cma-target-fitness 0.001
    #[arg(long)]
    cma_target_fitness: Option<f32>,

    // ========================================================================
    // GPU Configuration
    // ========================================================================
    /// Enable GPU acceleration
    ///
    /// When enabled, GPU-accelerated phases (Phase 0, 3, 4, 6) will use CUDA kernels.
    /// Requires CUDA-capable GPU and compiled with --features gpu.
    ///
    /// Example: --gpu
    #[arg(long, default_value = "true")]
    gpu: bool,

    /// CUDA device IDs (comma-separated for multi-GPU)
    ///
    /// Specifies which CUDA devices to use (0-indexed).
    /// Use nvidia-smi to list available devices.
    /// For multi-GPU, provide comma-separated list.
    ///
    /// Example: --gpu-devices 0,1,2
    #[arg(long, value_delimiter = ',', default_value = "0")]
    gpu_devices: Vec<usize>,

    /// Multi-GPU scheduling policy
    ///
    /// Determines how workloads are distributed across GPUs:
    /// - round-robin: Simple cyclic assignment (default)
    /// - least-loaded: Select device with lowest utilization
    /// - memory-aware: Select device with most available memory
    ///
    /// Example: --gpu-scheduling-policy least-loaded
    #[arg(long, default_value = "round-robin")]
    gpu_scheduling_policy: String,

    /// PTX directory path
    ///
    /// Directory containing compiled PTX kernel files.
    /// Should contain: dendritic_reservoir.ptx, floyd_warshall.ptx, tda.ptx, quantum.ptx
    ///
    /// Example: --gpu-ptx-dir /opt/prism/ptx
    #[arg(long, default_value = "target/ptx")]
    gpu_ptx_dir: String,

    /// Enable GPU secure mode (require signed PTX)
    ///
    /// When enabled, all PTX files must have valid .sha256 signature files.
    /// Prevents loading of tampered or untrusted kernels.
    ///
    /// Example: --gpu-secure
    #[arg(long, default_value = "false")]
    gpu_secure: bool,

    /// Trusted PTX directory (for secure mode)
    ///
    /// When --gpu-secure is enabled, PTX files are loaded only from this directory.
    /// Each PTX file must have a corresponding .sha256 signature file.
    ///
    /// Example: --gpu-trusted-ptx-dir /opt/prism/trusted_ptx
    #[arg(long)]
    gpu_trusted_ptx_dir: Option<String>,

    /// Disable NVRTC runtime compilation
    ///
    /// When set, disallows runtime kernel compilation via NVRTC.
    /// Only pre-compiled PTX from --gpu-ptx-dir is loaded.
    /// Recommended for production deployments.
    ///
    /// Example: --disable-nvrtc
    #[arg(long, default_value = "false")]
    disable_nvrtc: bool,

    /// NVML polling interval (milliseconds)
    ///
    /// Interval for collecting GPU telemetry (utilization, memory, temperature).
    /// Set to 0 to disable NVML polling.
    ///
    /// Example: --gpu-nvml-interval 500
    #[arg(long, default_value = "1000")]
    gpu_nvml_interval: u64,

    // ========================================================================
    // Metrics and Profiling
    // ========================================================================
    /// Enable Prometheus metrics server
    ///
    /// Starts HTTP server exposing /metrics endpoint for Prometheus scraping.
    /// Access metrics at http://localhost:<port>/metrics
    ///
    /// Example: --enable-metrics
    #[arg(long, default_value = "false")]
    enable_metrics: bool,

    /// Metrics server port
    ///
    /// TCP port for Prometheus metrics endpoint.
    /// Default: 9090
    ///
    /// Example: --metrics-port 9100
    #[arg(long, default_value = "9090")]
    metrics_port: u16,

    /// Enable performance profiler
    ///
    /// Captures detailed timing and resource usage for all phases and kernels.
    /// Results exported to JSON/CSV at end of pipeline execution.
    ///
    /// Example: --enable-profiler
    #[arg(long, default_value = "false")]
    enable_profiler: bool,

    /// Profiler output path (JSON)
    ///
    /// File path for performance profile JSON report.
    /// If not specified, exports to profiler_report.json
    ///
    /// Example: --profiler-output profile.json
    #[arg(long)]
    profiler_output: Option<String>,

    // ========================================================================
    // Multi-Attempt World-Record Optimization
    // ========================================================================
    /// Configuration file path (TOML format)
    ///
    /// Loads pipeline configuration from TOML file. Overrides default values and
    /// can specify memetic algorithm parameters, GPU settings, phase configs.
    ///
    /// Example: --config configs/dsjc250_memetic.toml
    #[arg(long)]
    config: Option<String>,

    /// Number of pipeline attempts for world-record optimization
    ///
    /// Runs the full pipeline multiple times with different random seeds.
    /// Each attempt is independent - finds optimal solution through massive parallelism.
    ///
    /// Strategy:
    /// - 1: Single attempt (default, fast validation)
    /// - 100-1,000: Good for initial optimization
    /// - 10,000-100,000: World-record hunting (GPU embarrassingly parallel)
    /// - 1,000,000+: Extreme optimization (RTX 3060 can handle in minutes)
    ///
    /// Best chromatic number and conflicts tracked across all attempts.
    ///
    /// Example: --attempts 10000
    #[arg(long, default_value = "1")]
    attempts: usize,

    // ========================================================================
    // Phase 2 Hyperparameters (Thermodynamic Annealing)
    // ========================================================================
    /// Phase 2: Number of annealing iterations
    ///
    /// Controls convergence quality vs runtime trade-off.
    /// Higher values improve solution quality but increase execution time.
    ///
    /// Recommended:
    /// - DSJC125: 20,000-30,000
    /// - DSJC250: 50,000-75,000
    /// - DSJC500: 100,000-150,000
    ///
    /// Example: --phase2-iterations 50000
    #[arg(long, default_value = "10000")]
    phase2_iterations: usize,

    /// Phase 2: Number of temperature replicas
    ///
    /// Parallel tempering uses multiple temperature schedules simultaneously.
    /// More replicas improve exploration but increase GPU memory usage.
    ///
    /// Range: 4-16, typical: 8
    ///
    /// Example: --phase2-replicas 12
    #[arg(long, default_value = "8")]
    phase2_replicas: usize,

    /// Phase 2: Minimum temperature
    ///
    /// Lowest temperature in parallel tempering schedule.
    /// Lower values enable fine-grained search but may get stuck in local optima.
    ///
    /// Range: 0.001-0.1, typical: 0.01
    ///
    /// Example: --phase2-temp-min 0.005
    #[arg(long, default_value = "0.01")]
    phase2_temp_min: f32,

    /// Phase 2: Maximum temperature
    ///
    /// Highest temperature in parallel tempering schedule.
    /// Higher values increase exploration but may lose convergence.
    ///
    /// Range: 5.0-50.0, typical: 10.0
    ///
    /// Example: --phase2-temp-max 20.0
    #[arg(long, default_value = "10.0")]
    phase2_temp_max: f32,
}

/// Run biomolecular mode (protein structure prediction and drug binding)
fn run_biomolecular_mode(args: &Args) -> Result<()> {
    use prism_core::domain::{BiomolecularAdapter, BiomolecularConfig};

    log::info!("=== Biomolecular Mode: Protein Structure & Drug Binding ===");

    // Validate required arguments
    let sequence_path = args
        .sequence
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--sequence is required for biomolecular mode"))?;

    let ligand_smiles = args
        .ligand
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--ligand is required for biomolecular mode"))?;

    // Create adapter with config
    let config = BiomolecularConfig {
        contact_distance: 8.0,
        num_poses: 10,
        use_gpu: args.gpu,
    };

    let adapter = BiomolecularAdapter::new(config);

    // Step 1: Predict structure
    log::info!("Step 1: Predicting protein structure...");
    let structure = adapter.predict_structure(sequence_path)?;

    // Step 2: Identify binding sites
    log::info!("Step 2: Identifying binding sites...");
    let binding_sites = adapter.identify_binding_sites(&structure)?;

    // Step 3: Predict binding affinity
    log::info!("Step 3: Predicting binding affinity...");
    let _affinity = adapter.predict_binding(&structure, ligand_smiles)?;

    // Step 4: Generate docking poses (for first binding site)
    if !binding_sites.is_empty() {
        log::info!("Step 4: Generating docking poses...");
        let poses = adapter.generate_poses(&structure, ligand_smiles, &binding_sites[0])?;

        if !poses.is_empty() {
            log::info!("=== Results Summary ===");
            log::info!(
                "Structure: {} residues, confidence={:.2}, RMSD={:.2} Å",
                structure.length,
                structure.confidence,
                structure.rmsd
            );
            log::info!("Binding sites: {} identified", binding_sites.len());
            log::info!("Best binding affinity: {:.2} kcal/mol", poses[0].affinity);
            log::info!("Top 3 docking poses:");
            for (i, pose) in poses.iter().take(3).enumerate() {
                log::info!("  Pose {}: {:.2} kcal/mol", i + 1, pose.affinity);
            }

            // Emit telemetry
            let telemetry_path = "telemetry_biomolecular.jsonl";
            let telemetry = serde_json::json!({
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "mode": "biomolecular",
                "sequence_path": sequence_path,
                "ligand_smiles": ligand_smiles,
                "results": {
                    "residues": structure.length,
                    "confidence": structure.confidence,
                    "rmsd": structure.rmsd,
                    "binding_sites": binding_sites.len(),
                    "best_affinity": poses[0].affinity,
                    "num_poses": poses.len()
                }
            });

            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(telemetry_path)?
                .write_all(format!("{}\n", telemetry).as_bytes())?;

            log::info!("Telemetry written to: {}", telemetry_path);
        }
    }

    Ok(())
}

/// Run materials mode (materials discovery and property prediction)
fn run_materials_mode(args: &Args) -> Result<()> {
    use prism_core::domain::{MaterialsAdapter, MaterialsConfig, TargetProperties};

    log::info!("=== Materials Mode: Materials Discovery ===");

    // Create adapter with config
    let config = MaterialsConfig {
        num_candidates: 20,
        use_gpu: args.gpu,
        seed: 42,
    };

    let adapter = MaterialsAdapter::new(config);

    // Define target properties (from config if available, or defaults)
    let target = TargetProperties {
        band_gap_range: (1.0, 3.0),
        max_formation_energy: -1.0,
        min_stability: 0.75,
        required_elements: vec![
            "Li".to_string(),
            "Fe".to_string(),
            "P".to_string(),
            "O".to_string(),
        ],
        forbidden_elements: vec!["Hg".to_string(), "Pb".to_string()],
    };

    // Discover materials
    log::info!("Discovering materials matching target properties...");
    let candidates = adapter.discover_material(&target)?;

    // Display results
    log::info!("=== Results Summary ===");
    log::info!("Generated {} candidate materials", candidates.len());
    log::info!("Top 5 candidates:");
    for (i, candidate) in candidates.iter().take(5).enumerate() {
        log::info!(
            "  {}. {} (confidence={:.2}, band_gap={:.2} eV, stability={:.2})",
            i + 1,
            candidate.composition,
            candidate.confidence,
            candidate.properties.band_gap,
            candidate.properties.stability
        );
    }

    // Emit telemetry
    let telemetry_path = "telemetry_materials.jsonl";
    if !candidates.is_empty() {
        let best = &candidates[0];
        let telemetry = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "mode": "materials",
            "target": {
                "band_gap_range": target.band_gap_range,
                "max_formation_energy": target.max_formation_energy,
                "min_stability": target.min_stability
            },
            "results": {
                "num_candidates": candidates.len(),
                "best_composition": best.composition,
                "best_confidence": best.confidence,
                "best_band_gap": best.properties.band_gap,
                "best_formation_energy": best.properties.formation_energy,
                "best_stability": best.properties.stability
            }
        });

        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(telemetry_path)?
            .write_all(format!("{}\n", telemetry).as_bytes())?;

        log::info!("Telemetry written to: {}", telemetry_path);
    }

    Ok(())
}

/// Run MEC-only diagnostic mode
///
/// This mode runs only the Molecular Emergent Computing (MEC) phase for diagnostics.
/// It simulates molecular dynamics on a small test system and emits telemetry.
fn run_mec_only_mode(args: &Args) -> Result<()> {
    use prism_core::traits::PhaseController;
    use prism_mec::{MecConfig, MecPhaseController};
    use std::collections::HashMap;

    log::info!("═══════════════════════════════════════════════════════");
    log::info!("=== MEC-Only Mode: Molecular Dynamics Diagnostics ===");
    log::info!("═══════════════════════════════════════════════════════");

    // Create a small test system (125 molecules)
    let num_molecules = 125;
    log::info!("Creating test system with {} molecules", num_molecules);

    // Create MEC config
    let mec_config = MecConfig {
        time_step: 1e-15,   // 1 femtosecond
        iterations: 1000,   // 1000 timesteps
        temperature: 300.0, // Room temperature
        use_gpu: args.gpu,
        reaction_rates: HashMap::new(),
    };

    log::info!("MEC Configuration:");
    log::info!("  Time step: {} fs", mec_config.time_step * 1e15);
    log::info!("  Iterations: {}", mec_config.iterations);
    log::info!("  Temperature: {} K", mec_config.temperature);
    log::info!("  GPU: {}", mec_config.use_gpu);

    // Initialize MEC phase controller
    let mut mec_controller = MecPhaseController::new(mec_config.clone());

    // Create a dummy graph for molecular system
    let graph = prism_core::types::Graph::new(num_molecules);

    // Create phase context
    let mut context = prism_core::traits::PhaseContext::new();

    // Execute MEC phase
    log::info!("Running molecular dynamics simulation...");
    let start = std::time::Instant::now();
    let outcome = mec_controller.execute(&graph, &mut context)?;
    let duration = start.elapsed();

    log::info!("MEC simulation completed in {:.3}s", duration.as_secs_f64());

    // Extract MEC metrics from context
    let free_energy = context
        .get_metadata("mec_free_energy")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let entropy = context
        .get_metadata("mec_entropy")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let pattern_index = context
        .get_metadata("mec_pattern_index")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    log::info!("=== MEC Diagnostics Results ===");
    log::info!("  Free energy: {:.2} kJ/mol", free_energy);
    log::info!("  Entropy: {:.3} J/(mol·K)", entropy);
    log::info!("  Pattern index: {}", pattern_index);
    log::info!("  Temperature: {:.1} K", mec_config.temperature);
    log::info!("  Timesteps: {}", mec_config.iterations);

    // Emit telemetry
    let telemetry = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "mode": "mec-only",
        "num_molecules": num_molecules,
        "config": {
            "time_step": mec_config.time_step,
            "iterations": mec_config.iterations,
            "temperature": mec_config.temperature,
            "use_gpu": mec_config.use_gpu,
        },
        "results": {
            "free_energy": free_energy,
            "entropy": entropy,
            "pattern_index": pattern_index,
            "simulation_time_secs": duration.as_secs_f64(),
        },
        "outcome": match outcome {
            prism_core::traits::PhaseOutcome::Success { message, .. } => message,
            prism_core::traits::PhaseOutcome::Retry { reason, .. } => format!("Retry: {}", reason),
            prism_core::traits::PhaseOutcome::Escalate { reason } => format!("Escalate: {}", reason),
        },
    });

    // Write telemetry to file
    let telemetry_path = "telemetry_mec_only.jsonl";
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(telemetry_path)?;
    use std::io::Write;
    writeln!(file, "{}", serde_json::to_string(&telemetry)?)?;
    log::info!("Telemetry written to: {}", telemetry_path);

    log::info!("MEC-only diagnostics completed successfully!");

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logger
    if args.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }

    log::info!("PRISM-Fold {} - Starting", VERSION);

    // ========================================================================
    // Mode Selection & Dispatch
    // ========================================================================

    match args.mode.as_str() {
        "biomolecular" => {
            return run_biomolecular_mode(&args);
        }
        "materials" => {
            return run_materials_mode(&args);
        }
        "mec-only" => {
            return run_mec_only_mode(&args);
        }
        "coloring" => {
            // Continue with normal graph coloring pipeline
            log::info!("Running graph coloring mode");
        }
        unknown => {
            anyhow::bail!(
                "Unknown mode: {}. Valid modes: coloring, biomolecular, materials, mec-only",
                unknown
            );
        }
    }

    // ========================================================================
    // Warmstart Configuration & Validation
    // ========================================================================

    let warmstart_config = if args.warmstart {
        // Validate weight constraints
        let weight_sum = args.warmstart_flux_weight
            + args.warmstart_ensemble_weight
            + args.warmstart_random_weight;

        if (weight_sum - 1.0).abs() > 0.01 {
            anyhow::bail!(
                "Warmstart weights must sum to 1.0 (got {:.3}). \
                 flux={:.2}, ensemble={:.2}, random={:.2}",
                weight_sum,
                args.warmstart_flux_weight,
                args.warmstart_ensemble_weight,
                args.warmstart_random_weight
            );
        }

        // Validate anchor fraction
        if args.warmstart_anchor_fraction < 0.0 || args.warmstart_anchor_fraction > 1.0 {
            anyhow::bail!(
                "Warmstart anchor fraction must be in [0.0, 1.0] (got {:.3})",
                args.warmstart_anchor_fraction
            );
        }

        // Validate max colors
        if args.warmstart_max_colors == 0 {
            anyhow::bail!("Warmstart max_colors must be > 0");
        }

        let config = WarmstartConfig {
            max_colors: args.warmstart_max_colors,
            min_prob: 0.001, // Default epsilon to prevent zero probabilities
            anchor_fraction: args.warmstart_anchor_fraction,
            flux_weight: args.warmstart_flux_weight,
            ensemble_weight: args.warmstart_ensemble_weight,
            random_weight: args.warmstart_random_weight,
            curriculum_catalog_path: args.warmstart_curriculum_path.clone(),
        };

        log::info!("Warmstart enabled:");
        log::info!("  Max colors: {}", config.max_colors);
        log::info!("  Anchor fraction: {:.2}", config.anchor_fraction);
        log::info!("  Flux weight: {:.2}", config.flux_weight);
        log::info!("  Ensemble weight: {:.2}", config.ensemble_weight);
        log::info!("  Random weight: {:.2}", config.random_weight);
        if let Some(ref path) = config.curriculum_catalog_path {
            log::info!("  Curriculum catalog: {}", path);
        }

        Some(config)
    } else {
        log::info!("Warmstart disabled (use --warmstart to enable)");
        None
    };

    // ========================================================================
    // GPU Configuration
    // ========================================================================

    use prism_pipeline::config::GpuConfig;
    use std::path::PathBuf;

    // Multi-GPU support: Use first device for GpuConfig (legacy), full list for MultiGpuManager
    let primary_device = args.gpu_devices.first().copied().unwrap_or(0);

    let gpu_config = GpuConfig {
        enabled: args.gpu,
        device_id: primary_device,
        ptx_dir: PathBuf::from(&args.gpu_ptx_dir),
        allow_nvrtc: !args.disable_nvrtc,
        require_signed_ptx: args.gpu_secure,
        trusted_ptx_dir: args.gpu_trusted_ptx_dir.as_ref().map(PathBuf::from),
        nvml_poll_interval_ms: args.gpu_nvml_interval,
    };

    if gpu_config.enabled {
        log::info!("GPU acceleration enabled:");
        log::info!("  Devices: {:?}", args.gpu_devices);
        log::info!("  Scheduling policy: {}", args.gpu_scheduling_policy);
        log::info!("  PTX directory: {}", gpu_config.ptx_dir.display());
        log::info!(
            "  Security mode: {}",
            if gpu_config.require_signed_ptx {
                "ENABLED (signed PTX required)"
            } else {
                "disabled"
            }
        );
        log::info!(
            "  NVRTC: {}",
            if gpu_config.allow_nvrtc {
                "allowed"
            } else {
                "DISABLED (pre-compiled PTX only)"
            }
        );
        if let Some(ref trusted_dir) = gpu_config.trusted_ptx_dir {
            log::info!("  Trusted PTX directory: {}", trusted_dir.display());
        }
        log::info!("  NVML polling: {} ms", gpu_config.nvml_poll_interval_ms);
    } else {
        log::info!("GPU acceleration disabled (use --gpu to enable)");
    }

    // ========================================================================
    // Metrics and Profiling Setup
    // ========================================================================

    use prism_pipeline::telemetry::prometheus::PrometheusMetrics;
    use prism_pipeline::PerformanceProfiler;
    use std::sync::Arc;

    // Initialize Prometheus metrics if enabled
    let metrics = if args.enable_metrics {
        let m = PrometheusMetrics::new()?;
        log::info!("Prometheus metrics enabled:");
        log::info!(
            "  Metrics endpoint: http://0.0.0.0:{}/metrics",
            args.metrics_port
        );
        log::info!(
            "  Health endpoint:  http://0.0.0.0:{}/health",
            args.metrics_port
        );
        Some(m)
    } else {
        log::info!("Prometheus metrics disabled (use --enable-metrics to enable)");
        None
    };

    // Initialize performance profiler if enabled
    let profiler = if args.enable_profiler {
        let p = PerformanceProfiler::new();
        let output_path = args
            .profiler_output
            .as_deref()
            .unwrap_or("profiler_report.json");
        log::info!("Performance profiler enabled:");
        log::info!("  Output path: {}", output_path);
        Some(p)
    } else {
        log::info!("Performance profiler disabled (use --enable-profiler to enable)");
        None
    };

    // Start metrics server in background if enabled
    if let Some(ref m) = metrics {
        let metrics_clone = Arc::clone(m);
        let port = args.metrics_port;

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
            rt.block_on(async {
                if let Err(e) =
                    prism_cli::metrics_server::start_metrics_server(port, metrics_clone).await
                {
                    log::error!("Metrics server failed: {}", e);
                }
            });
        });

        // Give server time to start
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // ========================================================================
    // Pipeline Configuration
    // ========================================================================

    // Validate input is provided for coloring mode
    let input_path = args
        .input
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--input is required for coloring mode"))?;

    // Load graph from file or create test graph
    let graph = if Path::new(input_path).exists() {
        log::info!("Loading graph from DIMACS file: {}", input_path);
        prism_core::dimacs::parse_dimacs_file(input_path)
            .map_err(|e| anyhow::anyhow!("Failed to parse DIMACS file '{}': {}", input_path, e))?
    } else {
        log::warn!(
            "Input file '{}' not found, creating test graph with {} vertices",
            input_path,
            args.vertices
        );
        Graph::new(args.vertices)
    };

    // ========================================================================
    // Phase 2 Configuration
    // ========================================================================

    use prism_pipeline::Phase2Config;

    let mut phase2_config = Phase2Config {
        iterations: args.phase2_iterations,
        replicas: args.phase2_replicas,
        temp_min: args.phase2_temp_min,
        temp_max: args.phase2_temp_max,
    };

    log::info!("Phase 2 hyperparameters:");
    log::info!("  Iterations: {}", phase2_config.iterations);
    log::info!("  Replicas: {}", phase2_config.replicas);
    log::info!(
        "  Temperature range: [{:.3}, {:.1}]",
        phase2_config.temp_min,
        phase2_config.temp_max
    );

    // Load config from TOML file if provided
    use prism_core::Phase3Config;
    use prism_pipeline::{
        GnnConfig as PipelineGnnConfig, MemeticConfig, MetaphysicalCouplingConfig,
    };
    let mut memetic_config: Option<MemeticConfig> = None;
    let mut metaphysical_coupling_config: Option<MetaphysicalCouplingConfig> = None;
    let mut gnn_config: Option<PipelineGnnConfig> = None;
    let mut phase0_config: Option<prism_phases::phase0::Phase0Config> = None;
    let mut phase1_config: Option<prism_phases::phase1_active_inference::Phase1Config> = None;
    let mut phase3_config: Option<Phase3Config> = None;
    let mut phase4_config: Option<prism_phases::phase4_geodesic::Phase4Config> = None;
    let mut phase6_config: Option<prism_phases::phase6_tda::Phase6Config> = None;
    let mut phase7_config: Option<prism_phases::phase7_ensemble::Phase7Config> = None;
    let mut telemetry_path_override: Option<String> = None;

    if let Some(config_path) = &args.config {
        log::info!("Loading configuration from: {}", config_path);

        // Load and parse config using serde
        use prism_cli::config::PrismConfig;
        let prism_config = PrismConfig::from_file(config_path)?;
        prism_config.validate()?;

        // Extract phase2 config if present and override defaults
        if let Some(ref p2) = prism_config.phase2.or(prism_config.phase2_thermodynamic) {
            phase2_config.iterations = p2.iterations;
            phase2_config.replicas = p2.replicas;
            phase2_config.temp_min = p2.temp_min;
            phase2_config.temp_max = p2.temp_max;

            log::info!("Phase 2 configuration overridden from TOML");
            log::info!("  Iterations: {}", phase2_config.iterations);
            log::info!("  Replicas: {}", phase2_config.replicas);
            log::info!(
                "  Temperature range: [{:.3}, {:.1}]",
                phase2_config.temp_min,
                phase2_config.temp_max
            );
        }

        // Extract Phase 0 dendritic reservoir config if present
        if let Some(ref cfg) = prism_config.phase0_dendritic {
            phase0_config = Some(cfg.clone());
            log::info!("Phase 0 dendritic reservoir configuration loaded from TOML");
        }

        // Extract Phase 1 active inference config if present
        if let Some(ref cfg) = prism_config.phase1_active_inference {
            phase1_config = Some(cfg.clone());
            log::info!("Phase 1 active inference configuration loaded from TOML");
        }

        // Extract phase3_quantum config if present
        if let Some(ref cfg) = prism_config.phase3_quantum {
            phase3_config = Some(cfg.clone());
            log::info!("Phase 3 quantum configuration loaded from TOML");
            log::info!("  Evolution time: {}", cfg.evolution_time);
            log::info!("  Coupling strength: {}", cfg.coupling_strength);
            log::info!("  Max colors: {}", cfg.max_colors);
        }

        // Extract Phase 4 geodesic config if present
        if let Some(ref cfg) = prism_config.phase4_geodesic {
            phase4_config = Some(cfg.clone());
            log::info!("Phase 4 geodesic configuration loaded from TOML");
        }

        // Extract Phase 6 TDA config if present
        if let Some(ref cfg) = prism_config.phase6_tda {
            phase6_config = Some(cfg.clone());
            log::info!("Phase 6 TDA configuration loaded from TOML");
        }

        // Extract Phase 7 Ensemble config if present
        if let Some(ref cfg) = prism_config.phase7_ensemble {
            phase7_config = Some(cfg.clone());
            log::info!("Phase 7 Ensemble configuration loaded from TOML");
        }

        // Extract memetic config if present
        if let Some(ref cfg) = prism_config.memetic {
            memetic_config = Some(cfg.clone());
            log::info!("Memetic algorithm configuration loaded from TOML");
        }

        // Extract metaphysical coupling config if present
        if let Some(ref cfg) = prism_config.metaphysical_coupling {
            metaphysical_coupling_config = Some(cfg.clone());
            log::info!("Metaphysical coupling configuration loaded from TOML");
        }

        // Extract GNN config if present
        if let Some(ref cfg) = prism_config.gnn {
            gnn_config = Some(cfg.clone());
            log::info!("GNN configuration loaded from TOML");
        }

        // Extract telemetry path from pipeline config if present
        if let Some(ref path) = prism_config.pipeline.telemetry_path {
            telemetry_path_override = Some(path.clone());
            log::info!("Telemetry path override from TOML: {}", path);
        }
    }

    // ========================================================================
    // FluxNet RL Controller
    // ========================================================================

    // Create RL controller with CLI-specified hyperparameters and optional reward log threshold from config
    let reward_log_threshold = metaphysical_coupling_config
        .as_ref()
        .map(|c| c.reward_log_threshold)
        .unwrap_or(0.001);

    let rl_config = RLConfig::builder()
        .epsilon(args.fluxnet_epsilon)
        .alpha(args.fluxnet_alpha)
        .gamma(args.fluxnet_gamma)
        .reward_log_threshold(reward_log_threshold)
        .build();

    log::info!("FluxNet RL controller initialized:");
    log::info!("  Epsilon (exploration): {:.3}", rl_config.epsilon);
    log::info!("  Alpha (learning rate): {:.3}", rl_config.alpha);
    log::info!("  Gamma (discount): {:.3}", rl_config.gamma);
    log::info!(
        "  Reward log threshold: {:.4}",
        rl_config.reward_log_threshold
    );

    let rl_controller = UniversalRLController::new(rl_config);

    // Load pretrained Q-table if specified
    if let Some(ref qtable_path) = args.fluxnet_qtable {
        log::info!("Loading pretrained Q-table from: {}", qtable_path);

        // Try binary format first, fall back to JSON
        let result = if qtable_path.ends_with(".bin") {
            rl_controller.load_qtables_binary(qtable_path)
        } else {
            rl_controller.load_qtables(qtable_path)
        };

        match result {
            Ok(_) => {
                log::info!("Q-table loaded successfully!");

                // Print Q-table stats for verification
                for phase_name in &[
                    "Phase0-DendriticReservoir",
                    "Phase1-ActiveInference",
                    "Phase2-Thermodynamic",
                    "Phase3-QuantumClassical",
                    "Phase4-Geodesic",
                    "Phase6-TDA",
                    "Phase7-Ensemble",
                ] {
                    let (mean, min, max) = rl_controller.qtable_stats(phase_name);
                    log::info!(
                        "  {}: mean={:.3}, range=[{:.3}, {:.3}]",
                        phase_name,
                        mean,
                        min,
                        max
                    );
                }
            }
            Err(e) => {
                log::warn!("Failed to load Q-table: {}", e);
                log::warn!("Continuing with default Q-table initialization");
            }
        }
    } else {
        log::info!("No pretrained Q-table specified (use --fluxnet-qtable to load)");
        log::info!("Starting with randomly initialized Q-tables");
    }

    // Build pipeline config with optional warmstart and GPU
    let mut pipeline_builder = PipelineConfig::builder()
        .max_vertices(10000)
        .gpu(gpu_config)
        .phase2(phase2_config);

    if let Some(warmstart_cfg) = warmstart_config {
        pipeline_builder = pipeline_builder.warmstart(warmstart_cfg);
    }

    if let Some(memetic_cfg) = memetic_config {
        pipeline_builder = pipeline_builder.memetic(memetic_cfg);
    }

    if let Some(coupling_cfg) = metaphysical_coupling_config {
        pipeline_builder = pipeline_builder.metaphysical_coupling(coupling_cfg);
    }

    if let Some(gnn_cfg) = gnn_config {
        pipeline_builder = pipeline_builder.gnn(gnn_cfg);
    }

    // Add CMA-ES configuration if enabled
    if args.enable_cma {
        let cma_config = prism_pipeline::config::CmaEsConfig {
            enabled: true,
            population_size: args.cma_population_size,
            initial_sigma: args.cma_sigma,
            max_iterations: args.cma_generations,
            target_fitness: args.cma_target_fitness,
            use_gpu: args.gpu, // Use GPU if --gpu flag is set
        };
        pipeline_builder = pipeline_builder.cma_es(cma_config);
    }

    if let Some(path) = telemetry_path_override {
        pipeline_builder = pipeline_builder.telemetry_path(path);
    }

    let pipeline_config = pipeline_builder.build()?;

    // Create orchestrator (clone config so we can reference it later for memetic)
    let mut orchestrator = PipelineOrchestrator::new(pipeline_config.clone(), rl_controller);

    // Pass phase0 config to orchestrator if loaded from TOML
    if let Some(cfg) = phase0_config {
        orchestrator.set_phase0_config(cfg);
        log::info!("Phase 0 config passed to orchestrator");
    }

    // Pass phase1 config to orchestrator if loaded from TOML
    if let Some(cfg) = phase1_config {
        orchestrator.set_phase1_config(cfg);
        log::info!("Phase 1 config passed to orchestrator");
    }

    // Pass phase3 config to orchestrator if loaded from TOML
    if let Some(cfg) = phase3_config {
        orchestrator.set_phase3_config(cfg);
        log::info!("Phase 3 config passed to orchestrator");
    }

    // Pass phase4 config to orchestrator if loaded from TOML
    if let Some(cfg) = phase4_config {
        orchestrator.set_phase4_config(cfg);
        log::info!("Phase 4 config passed to orchestrator");
    }

    // Pass phase6 config to orchestrator if loaded from TOML
    if let Some(cfg) = phase6_config {
        orchestrator.set_phase6_config(cfg);
        log::info!("Phase 6 config passed to orchestrator");
    }

    // Pass phase7 config to orchestrator if loaded from TOML
    if let Some(cfg) = phase7_config {
        orchestrator.set_phase7_config(cfg);
        log::info!("Phase 7 config passed to orchestrator");
    }

    // Note: Phase initialization happens inside run() after GPU context is set up
    // This ensures phases receive GPU context when available

    // Run pipeline with multi-attempt world-record optimization
    log::info!(
        "Running pipeline on graph with {} vertices",
        graph.num_vertices
    );

    // Check if memetic algorithm is enabled
    let memetic_enabled = pipeline_config
        .memetic
        .as_ref()
        .map(|m| m.enabled)
        .unwrap_or(false);

    if args.attempts > 1 {
        log::info!("Multi-attempt mode enabled: {} attempts", args.attempts);
        if memetic_enabled {
            log::info!("Strategy: GPU population generation + CPU memetic evolution");
        } else {
            log::info!("Strategy: Embarrassingly parallel search for optimal random seeds");
        }
    }

    let mut best_solution = None;
    let mut best_chromatic = usize::MAX;
    let mut best_conflicts = usize::MAX;
    let mut all_solutions = Vec::new(); // Collect all solutions for memetic algorithm
    let overall_start = std::time::Instant::now();

    for attempt in 1..=args.attempts {
        let start_time = std::time::Instant::now();
        let solution = orchestrator.run_with_seed(&graph, attempt as u64)?;
        let pipeline_duration = start_time.elapsed();

        // Collect solution for memetic algorithm if enabled
        if memetic_enabled {
            all_solutions.push(solution.clone());
        }

        // Track best solution across all attempts
        let is_improvement = solution.chromatic_number < best_chromatic
            || (solution.chromatic_number == best_chromatic && solution.conflicts < best_conflicts);

        if is_improvement {
            best_chromatic = solution.chromatic_number;
            best_conflicts = solution.conflicts;
            best_solution = Some(solution.clone());

            log::info!(
                "Attempt {}/{}: ⭐ NEW BEST - {} colors, {} conflicts ({:.2}s)",
                attempt,
                args.attempts,
                solution.chromatic_number,
                solution.conflicts,
                pipeline_duration.as_secs_f64()
            );
        } else if attempt % 100 == 0 || attempt == args.attempts {
            // Log progress every 100 attempts or on final attempt
            log::info!(
                "Attempt {}/{}: {} colors, {} conflicts (best: {} colors, {} conflicts)",
                attempt,
                args.attempts,
                solution.chromatic_number,
                solution.conflicts,
                best_chromatic,
                best_conflicts
            );
        }

        // Update metrics with best-so-far if enabled
        if let Some(ref m) = metrics {
            m.update_pipeline_best_chromatic(best_chromatic as u32)?;
            m.record_pipeline_runtime(pipeline_duration.as_secs_f64())?;
            if best_conflicts == 0 {
                m.record_solution_found()?;
            }
        }
    }

    let mut solution = best_solution.unwrap();
    let mut total_duration = overall_start.elapsed();

    // Run memetic algorithm if enabled
    if memetic_enabled {
        if let Some(memetic_config) = &pipeline_config.memetic {
            log::info!("═══════════════════════════════════════════════════════");
            log::info!("Starting memetic algorithm evolution");
            log::info!("═══════════════════════════════════════════════════════");
            log::info!("  Initial population size: {}", all_solutions.len());
            log::info!(
                "  Target population size: {}",
                memetic_config.population_size
            );
            log::info!("  Generations: {}", memetic_config.generations);
            log::info!("  Crossover rate: {}", memetic_config.crossover_rate);
            log::info!("  Mutation rate: {}", memetic_config.mutation_rate);
            log::info!(
                "  Local search iterations: {}",
                memetic_config.local_search_iterations
            );
            log::info!("  Elitism count: {}", memetic_config.elitism_count);
            log::info!("  Tournament size: {}", memetic_config.tournament_size);
            log::info!(
                "  Convergence threshold: {}",
                memetic_config.convergence_threshold
            );
            log::info!(
                "  Best GPU solution: {} colors, {} conflicts",
                solution.chromatic_number,
                solution.conflicts
            );

            let memetic_start = std::time::Instant::now();

            // Create memetic algorithm with config parameters
            use prism_phases::MemeticAlgorithm;
            let memetic = MemeticAlgorithm::new(
                memetic_config.population_size,
                memetic_config.generations,
                memetic_config.crossover_rate,
                memetic_config.mutation_rate,
                memetic_config.local_search_iterations,
                memetic_config.elitism_count,
                memetic_config.tournament_size,
                memetic_config.convergence_threshold,
            );

            // Evolve population
            // Evolve with geometry metrics if available from context
            // (For now, passing None - full pipeline integration will provide metrics)
            match memetic.evolve(&graph, all_solutions, None) {
                Ok(evolved_solution) => {
                    let memetic_duration = memetic_start.elapsed();
                    total_duration = overall_start.elapsed();

                    log::info!("═══════════════════════════════════════════════════════");
                    log::info!("Memetic evolution completed!");
                    log::info!("═══════════════════════════════════════════════════════");
                    log::info!(
                        "  Evolution time: {:.3}s ({:.2} hours)",
                        memetic_duration.as_secs_f64(),
                        memetic_duration.as_secs_f64() / 3600.0
                    );
                    log::info!(
                        "  Final chromatic number: {}",
                        evolved_solution.chromatic_number
                    );
                    log::info!("  Final conflicts: {}", evolved_solution.conflicts);

                    if evolved_solution.chromatic_number < solution.chromatic_number
                        || (evolved_solution.chromatic_number == solution.chromatic_number
                            && evolved_solution.conflicts < solution.conflicts)
                    {
                        log::info!(
                            "  ⭐ IMPROVEMENT: {} → {} colors",
                            solution.chromatic_number,
                            evolved_solution.chromatic_number
                        );
                        solution = evolved_solution;
                    } else {
                        log::info!("  No improvement over GPU best (keeping GPU solution)");
                    }
                }
                Err(e) => {
                    log::error!("Memetic evolution failed: {}", e);
                    log::warn!("Falling back to best GPU solution");
                }
            }
        }
    }

    // ========================================================================
    // GNN Inference (Optional)
    // ========================================================================

    if let Some(gnn_config) = &pipeline_config.gnn {
        if gnn_config.enabled {
            log::info!("═══════════════════════════════════════════════════════");
            log::info!("Running GNN inference");
            log::info!("═══════════════════════════════════════════════════════");

            use petgraph::graph::DiGraph;
            use prism_core::domain::GnnState;
            use prism_gnn::{
                E3EquivariantGnn, GnnArchitecture, GnnConfig as GnnLibConfig, OnnxGnn,
            };

            let gnn_start = std::time::Instant::now();

            // Convert prism_core::Graph to petgraph::DiGraph for GNN processing
            let mut petgraph = DiGraph::<f32, f32>::new();
            let mut node_map = Vec::new();

            // Add nodes
            for _ in 0..graph.num_vertices {
                let node = petgraph.add_node(0.0);
                node_map.push(node);
            }

            // Add edges
            for (u, neighbors) in graph.adjacency.iter().enumerate() {
                for &v in neighbors {
                    if u < v {
                        // Add each undirected edge only once
                        petgraph.add_edge(node_map[u], node_map[v], 1.0);
                    }
                }
            }

            // Convert pipeline config to GNN library config
            let gnn_lib_config = GnnLibConfig {
                hidden_dim: gnn_config.hidden_dim,
                num_layers: gnn_config.num_layers,
                dropout: gnn_config.dropout,
                learning_rate: gnn_config.learning_rate,
                architecture: match gnn_config.architecture.as_str() {
                    "e3_equivariant" => GnnArchitecture::GCN,
                    "onnx" => GnnArchitecture::GCN,
                    _ => GnnArchitecture::GCN,
                },
                use_gpu: gnn_config.use_gpu,
            };

            // Run inference based on architecture type
            let prediction_result = if gnn_config.architecture == "onnx" {
                if let Some(ref model_path) = gnn_config.onnx_model_path {
                    log::info!("  Architecture: ONNX");
                    log::info!("  Model path: {}", model_path);
                    let onnx_gnn =
                        OnnxGnn::load(model_path, gnn_config.hidden_dim, gnn_config.hidden_dim)?;
                    onnx_gnn.predict(&petgraph)
                } else {
                    log::warn!("  ONNX architecture selected but no model_path provided, falling back to E3-Equivariant");
                    log::info!("  Architecture: E3-Equivariant GNN (fallback)");
                    let e3_gnn = E3EquivariantGnn::new(gnn_lib_config);
                    e3_gnn.predict(&petgraph)
                }
            } else {
                log::info!("  Architecture: E3-Equivariant GNN");
                log::info!("  Hidden dimensions: {}", gnn_config.hidden_dim);
                log::info!("  Number of layers: {}", gnn_config.num_layers);
                let e3_gnn = E3EquivariantGnn::new(gnn_lib_config);
                e3_gnn.predict(&petgraph)
            };

            match prediction_result {
                Ok(prediction) => {
                    let gnn_duration = gnn_start.elapsed();

                    log::info!(
                        "GNN inference completed in {:.3}s",
                        gnn_duration.as_secs_f64()
                    );
                    log::info!(
                        "  Predicted chromatic number: {}",
                        prediction.chromatic_number
                    );
                    log::info!("  Actual chromatic number: {}", solution.chromatic_number);
                    log::info!("  Prediction confidence: {:.3}", prediction.confidence);
                    log::info!(
                        "  Manifold dimension: {:.2}",
                        prediction.manifold_features.dimension
                    );
                    log::info!(
                        "  Manifold curvature: {:.3}",
                        prediction.manifold_features.curvature
                    );
                    log::info!(
                        "  Geodesic complexity: {:.2}",
                        prediction.manifold_features.geodesic_complexity
                    );
                    log::info!(
                        "  Betti numbers: {:?}",
                        prediction.manifold_features.betti_numbers
                    );

                    // Create GNN state for telemetry (could be used for PhaseContext in future)
                    let _gnn_state = GnnState {
                        predicted_chromatic: prediction.chromatic_number,
                        embedding_dim: gnn_config.hidden_dim,
                        confidence: prediction.confidence,
                        manifold_dimension: prediction.manifold_features.dimension,
                        manifold_curvature: prediction.manifold_features.curvature,
                        geodesic_complexity: prediction.manifold_features.geodesic_complexity,
                        betti_count: prediction.manifold_features.betti_numbers.len(),
                        model_type: gnn_config.architecture.clone(),
                    };

                    // Emit GNN telemetry
                    let telemetry = serde_json::json!({
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                        "mode": "gnn",
                        "graph_vertices": graph.num_vertices,
                        "graph_edges": graph.num_edges,
                        "predicted_chromatic": prediction.chromatic_number,
                        "actual_chromatic": solution.chromatic_number,
                        "confidence": prediction.confidence,
                        "manifold_features": {
                            "dimension": prediction.manifold_features.dimension,
                            "curvature": prediction.manifold_features.curvature,
                            "geodesic_complexity": prediction.manifold_features.geodesic_complexity,
                            "betti_numbers": prediction.manifold_features.betti_numbers,
                        },
                        "inference_time_secs": gnn_duration.as_secs_f64(),
                        "model_type": gnn_config.architecture,
                        "embedding_dim": gnn_config.hidden_dim,
                    });

                    // Write telemetry to file
                    let telemetry_path = "telemetry_gnn.jsonl";
                    let mut file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(telemetry_path)?;
                    use std::io::Write;
                    writeln!(file, "{}", serde_json::to_string(&telemetry)?)?;
                    log::info!("  GNN telemetry written to: {}", telemetry_path);
                }
                Err(e) => {
                    log::error!("GNN inference failed: {}", e);
                    log::warn!("Continuing without GNN analysis");
                }
            }
        }
    }

    log::info!("Multi-attempt optimization completed!");
    log::info!("  Total attempts: {}", args.attempts);
    log::info!("  Best chromatic number: {}", solution.chromatic_number);
    log::info!("  Best conflicts: {}", solution.conflicts);
    log::info!("  Valid: {}", solution.is_valid());
    log::info!("  Total runtime: {:.3}s", total_duration.as_secs_f64());
    log::info!(
        "  Avg per attempt: {:.3}s",
        total_duration.as_secs_f64() / args.attempts as f64
    );

    // Final metrics update (already done in loop, but log confirmation)
    if metrics.is_some() {
        log::info!("Metrics updated successfully");
    }

    // Export profiler results if enabled
    if let Some(p) = profiler {
        let output_path = args
            .profiler_output
            .as_deref()
            .unwrap_or("profiler_report.json");
        p.export_json(std::path::Path::new(output_path))?;

        // Also export CSV
        let csv_path = output_path.replace(".json", ".csv");
        p.export_csv(std::path::Path::new(&csv_path))?;

        let report = p.generate_report();
        log::info!("Performance profiler results:");
        log::info!("  Total duration: {:.3}s", report.total_duration_secs);
        log::info!("  Phase iterations: {}", report.total_phase_iterations);
        log::info!("  Kernel launches: {}", report.total_kernel_launches);
        log::info!("  Report exported to: {}", output_path);
        log::info!("  CSV exported to: {}", csv_path);
    }

    // Keep metrics server alive if enabled
    if metrics.is_some() {
        log::info!("Metrics server still running. Press Ctrl+C to exit.");
        log::info!(
            "Access metrics at http://localhost:{}/metrics",
            args.metrics_port
        );

        // Sleep indefinitely (until Ctrl+C)
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    Ok(())
}
