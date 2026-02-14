//! PRISM-Fold: World-class interactive computational research interface
//!
//! Phase Resonance Integrated Solver Machine for Molecular Folding
//! GPU-accelerated graph coloring and ligand binding site prediction
//! with AI-native conversational interface and real-time visualization.

use anyhow::Result;
use clap::Parser;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use std::io::{self, stdout};
use std::path::Path;

mod ai;
pub mod config;
pub mod metrics_server;
mod runtime;
mod streaming;
mod ui;
mod widgets;

pub use config::PrismConfig;
pub use runtime::PrismRuntime;
pub use ui::App;

/// PRISM-Fold version
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(name = "prism")]
#[command(version = VERSION)]
#[command(about = "PRISM-Fold: World-class GPU-accelerated graph coloring and molecular analysis")]
#[command(long_about = r#"
╔══════════════════════════════════════════════════════════════════════════╗
║  ◆ PRISM-Fold                                                            ║
║                                                                          ║
║  Phase Resonance Integrated Solver Machine for Molecular Folding         ║
║                                                                          ║
║  A bleeding-edge computational research platform featuring:              ║
║  • GPU-accelerated 7-phase optimization pipeline                         ║
║  • Real-time visualization of optimization dynamics                      ║
║  • AI-native conversational interface                                    ║
║  • Ligand binding site prediction with GNN                               ║
║  • Quantum-classical hybrid algorithms                                   ║
║  • Dendritic reservoir computing                                         ║
║                                                                          ║
║  Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.            ║
╚══════════════════════════════════════════════════════════════════════════╝
"#)]
struct Args {
    /// Input file (graph .col or protein .pdb)
    #[arg(short, long)]
    input: Option<String>,

    /// Output file path (JSON for results)
    #[arg(short, long)]
    output: Option<String>,

    /// Output format: json, csv
    #[arg(long, default_value = "json")]
    format: String,

    /// Mode: auto, coloring, biomolecular, materials
    #[arg(short, long, default_value = "coloring")]
    mode: String,

    /// Number of vertices (for testing)
    #[arg(long, default_value = "10")]
    vertices: usize,

    /// Start in non-interactive batch mode
    #[arg(long)]
    batch: bool,

    /// Enable verbose logging to file
    #[arg(short, long)]
    verbose: bool,

    /// Skip TUI, run with minimal output
    #[arg(long)]
    headless: bool,

    // ========================================================================
    // Warmstart Configuration
    // ========================================================================
    /// Enable warmstart system with priors from Phase 0
    #[arg(long, default_value = "false")]
    warmstart: bool,

    /// Warmstart flux weight (reservoir prior contribution)
    #[arg(long, default_value = "0.4")]
    warmstart_flux_weight: f32,

    /// Warmstart ensemble weight (structural anchor contribution)
    #[arg(long, default_value = "0.4")]
    warmstart_ensemble_weight: f32,

    /// Warmstart random weight (exploration contribution)
    #[arg(long, default_value = "0.2")]
    warmstart_random_weight: f32,

    /// Warmstart anchor fraction (0.0-1.0)
    #[arg(long, default_value = "0.10")]
    warmstart_anchor_fraction: f32,

    /// Warmstart maximum colors in prior distribution
    #[arg(long, default_value = "50")]
    warmstart_max_colors: usize,

    /// Path to curriculum profile catalog (optional)
    #[arg(long)]
    warmstart_curriculum_path: Option<String>,

    // ========================================================================
    // FluxNet RL Configuration
    // ========================================================================
    /// Path to pretrained FluxNet Q-table (binary format)
    #[arg(long)]
    fluxnet_qtable: Option<String>,

    /// FluxNet RL epsilon (exploration rate)
    #[arg(long, default_value = "0.2")]
    fluxnet_epsilon: f64,

    /// FluxNet RL learning rate (alpha)
    #[arg(long, default_value = "0.1")]
    fluxnet_alpha: f64,

    /// FluxNet RL discount factor (gamma)
    #[arg(long, default_value = "0.95")]
    fluxnet_gamma: f64,

    // ========================================================================
    // GPU Configuration
    // ========================================================================
    /// Enable GPU acceleration
    #[arg(long, default_value = "true")]
    gpu: bool,

    /// CUDA device IDs (comma-separated for multi-GPU)
    #[arg(long, value_delimiter = ',', default_value = "0")]
    gpu_devices: Vec<usize>,

    /// Multi-GPU scheduling policy
    #[arg(long, default_value = "round-robin")]
    gpu_scheduling_policy: String,

    /// PTX directory path
    #[arg(long, default_value = "target/ptx")]
    gpu_ptx_dir: String,

    /// Enable GPU secure mode (require signed PTX)
    #[arg(long, default_value = "false")]
    gpu_secure: bool,

    /// Trusted PTX directory (for secure mode)
    #[arg(long)]
    gpu_trusted_ptx_dir: Option<String>,

    /// Disable NVRTC runtime compilation
    #[arg(long, default_value = "false")]
    disable_nvrtc: bool,

    /// NVML polling interval (milliseconds)
    #[arg(long, default_value = "1000")]
    gpu_nvml_interval: u64,

    /// Enable AATGS async kernel scheduling
    #[arg(long, default_value = "false")]
    aatgs_async: bool,

    /// Enable multi-GPU parallel execution (requires multiple --gpu-devices)
    #[arg(long, default_value = "false")]
    multi_gpu: bool,

    /// Enable Ultra Kernel (fused 8-component GPU kernel)
    #[arg(long, default_value = "false")]
    ultra_kernel: bool,

    // ========================================================================
    // Phase 2 Hyperparameters (Thermodynamic Annealing)
    // ========================================================================
    /// Phase 2: Number of annealing iterations
    #[arg(long, default_value = "10000")]
    phase2_iterations: usize,

    /// Phase 2: Number of temperature replicas
    #[arg(long, default_value = "8")]
    phase2_replicas: usize,

    /// Phase 2: Minimum temperature
    #[arg(long, default_value = "0.01")]
    phase2_temp_min: f32,

    /// Phase 2: Maximum temperature
    #[arg(long, default_value = "10.0")]
    phase2_temp_max: f32,

    // ========================================================================
    // CMA-ES Configuration
    // ========================================================================
    /// Enable CMA-ES evolutionary optimization phase
    #[arg(long, default_value = "false")]
    enable_cma: bool,

    /// CMA-ES population size
    #[arg(long, default_value = "50")]
    cma_population_size: usize,

    /// CMA-ES maximum generations
    #[arg(long, default_value = "100")]
    cma_generations: usize,

    /// CMA-ES initial step size (sigma)
    #[arg(long, default_value = "0.5")]
    cma_sigma: f32,

    /// CMA-ES target fitness (optional)
    #[arg(long)]
    cma_target_fitness: Option<f32>,

    // ========================================================================
    // Metrics and Profiling
    // ========================================================================
    /// Enable Prometheus metrics server
    #[arg(long, default_value = "false")]
    enable_metrics: bool,

    /// Metrics server port
    #[arg(long, default_value = "9090")]
    metrics_port: u16,

    /// Enable performance profiler
    #[arg(long, default_value = "false")]
    enable_profiler: bool,

    /// Profiler output path (JSON)
    #[arg(long)]
    profiler_output: Option<String>,

    // ========================================================================
    // Multi-Attempt World-Record Optimization
    // ========================================================================
    /// Configuration file path (TOML format)
    #[arg(long)]
    config: Option<String>,

    /// Number of pipeline attempts for world-record optimization
    #[arg(long, default_value = "1")]
    attempts: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging (to file in TUI mode)
    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("prism=debug,prism_gpu=debug,prism_pipeline=info")
            .with_writer(std::fs::File::create("prism.log")?)
            .init();
    }

    // Handle biomolecular mode separately - no TUI, direct LBS pipeline
    // Also auto-detect biomolecular mode for .pdb files in batch mode
    let is_pdb_input = args.input.as_ref()
        .map(|p| p.to_lowercase().ends_with(".pdb"))
        .unwrap_or(false);

    if args.mode == "biomolecular" || (args.batch && is_pdb_input) {
        return run_biomolecular(args);
    }

    if args.headless || args.batch {
        // Headless/batch mode - minimal output, no TUI
        run_headless(args)
    } else {
        // Full TUI mode
        run_tui(args)
    }
}

/// Run biomolecular mode (LBS prediction) - DIRECT LIBRARY INTEGRATION
/// Supports --batch mode with -o output.json for validation suite
fn run_biomolecular(args: Args) -> Result<()> {
    use prism_lbs::{LbsConfig, PrismLbs, ProteinStructure};

    let input_path = args
        .input
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--input is required for biomolecular mode (PDB file)"))?;

    if !Path::new(input_path).exists() {
        return Err(anyhow::anyhow!("PDB file not found: {}", input_path));
    }

    // Determine output path - use --output if provided, else default to lbs_results/
    let (output_json, output_dir) = if let Some(ref out_path) = args.output {
        let out = std::path::PathBuf::from(out_path);
        let dir = out.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| std::path::PathBuf::from("."));
        (Some(out), dir)
    } else {
        (None, std::path::PathBuf::from("lbs_results"))
    };

    // In batch mode, minimize output
    let quiet = args.batch && args.output.is_some();

    if !quiet {
        println!("◆ PRISM-Fold {} - Biomolecular Mode\n", VERSION);
        println!("  Input: {}", input_path);
    }

    // Initialize logger for headless mode
    if args.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }

    // Create output directory
    std::fs::create_dir_all(&output_dir)?;
    if !quiet {
        println!("  Output: {}/\n", output_dir.display());
    }

    // Load protein structure directly
    if !quiet { println!("  Loading protein structure..."); }
    let structure = ProteinStructure::from_pdb_file(Path::new(input_path))?;

    if !quiet {
        println!("  Loaded: {} residues, {} atoms, {} chains",
            structure.residues.len(),
            structure.atoms.len(),
            structure.chain_residue_indices.len()
        );
    }

    // Create LBS predictor with default config (GPU-accelerated)
    if !quiet { println!("\n  Initializing PRISM-LBS predictor (GPU: {})...", args.gpu); }
    let mut config = LbsConfig::default();
    config.use_gpu = args.gpu;

    let predictor = PrismLbs::new(config)?;

    // Run prediction
    if !quiet { println!("  Running 7-phase binding site detection...\n"); }
    let start = std::time::Instant::now();
    let pockets = predictor.predict(&structure)?;
    let duration = start.elapsed();

    // Write results
    if !quiet { println!("\n  Writing results..."); }

    // If custom output path specified (batch mode), write JSON there directly
    if let Some(ref json_path) = output_json {
        prism_lbs::output::write_pockets_json(&pockets, &structure, json_path)?;
    } else {
        // Default behavior: write to lbs_results/
        prism_lbs::output::write_pockets_pdb(&pockets, &structure, &output_dir.join("pockets.pdb"))?;
        prism_lbs::output::write_pockets_json(&pockets, &structure, &output_dir.join("pockets.json"))?;
    }

    if !quiet && output_dir.join("visualize_pockets.pml").exists() {
        println!("  PyMOL script: {}/visualize_pockets.pml", output_dir.display());
    }

    // Print summary (always show for successful batch runs, but minimal)
    if quiet {
        // Batch mode: just output pocket count for scripts to parse
        println!("{}", pockets.len());
    } else {
        println!("\n✅ LBS prediction complete in {:.2}s!", duration.as_secs_f64());
        if let Some(ref json_path) = output_json {
            println!("   Results saved to: {}", json_path.display());
        } else {
            println!("   Results saved to: {}/", output_dir.display());
        }
    }

    if !quiet && !pockets.is_empty() {
        println!("\n   Found {} pockets:", pockets.len());
        for (i, pocket) in pockets.iter().take(5).enumerate() {
            let class = pocket.druggability_score.classification.as_str();
            println!(
                "   {}. Vol: {:.1}Å³, Druggability: {:.3} ({}), {} residues",
                i + 1,
                pocket.volume,
                pocket.druggability_score.total,
                class,
                pocket.residue_indices.len()
            );
        }
    } else if !quiet {
        println!("\n   No pockets detected (structure may be too small or lack cavities)");
    }

    Ok(())
}

fn run_tui(args: Args) -> Result<()> {
    // Create async runtime for the PRISM runtime system
    let rt = tokio::runtime::Runtime::new()?;

    rt.block_on(async {
        // Initialize PRISM runtime
        let runtime_config = runtime::RuntimeConfig {
            event_bus_capacity: 1024,
            ring_buffer_size: 1000,
            gpu_poll_interval_ms: 100,
            max_actors: 16,
        };

        let mut prism_runtime = PrismRuntime::new(runtime_config)?;

        // Start runtime (spawns all actors)
        prism_runtime.start().await?;

        // Get event receiver for UI updates
        let event_rx = prism_runtime.subscribe();

        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Extract GPU device ID for TUI
        let gpu_device = args.gpu_devices.first().copied().unwrap_or(0);

        // Create app with runtime integration
        let mut app = App::new_with_runtime(
            args.input,
            args.mode,
            gpu_device,
            prism_runtime.state.clone(),
            event_rx,
        )?;

        let result = app.run(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        // Shutdown runtime
        prism_runtime.shutdown().await?;

        // Handle any errors from the app
        if let Err(e) = result {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }

        Ok(())
    })
}

fn run_headless(args: Args) -> Result<()> {
    use prism_core::{Graph, WarmstartConfig};
    use prism_fluxnet::{RLConfig, UniversalRLController};
    use prism_pipeline::{PipelineConfig, PipelineOrchestrator};
    use std::io::Write;
    use std::path::Path;

    println!("◆ PRISM-Fold {} - Headless Mode", VERSION);
    println!();

    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(if args.verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .init();

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
                    crate::metrics_server::start_metrics_server(port, metrics_clone).await
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
        use crate::config::PrismConfig;
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
                "Attempt {}/{}: NEW BEST - {} colors, {} conflicts ({:.2}s)",
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
    let total_duration = overall_start.elapsed();

    // Run memetic algorithm if enabled
    if memetic_enabled {
        if let Some(memetic_config) = &pipeline_config.memetic {
            log::info!("═══════════════════════════════════════════════════════");
            log::info!("Starting memetic algorithm evolution");
            log::info!("═══════════════════════════════════════════════════════");

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

            match memetic.evolve(&graph, all_solutions, None) {
                Ok(evolved_solution) => {
                    let memetic_duration = memetic_start.elapsed();

                    log::info!("Memetic evolution completed in {:.3}s", memetic_duration.as_secs_f64());
                    log::info!("  Final chromatic number: {}", evolved_solution.chromatic_number);
                    log::info!("  Final conflicts: {}", evolved_solution.conflicts);

                    if evolved_solution.chromatic_number < solution.chromatic_number
                        || (evolved_solution.chromatic_number == solution.chromatic_number
                            && evolved_solution.conflicts < solution.conflicts)
                    {
                        log::info!(
                            "  IMPROVEMENT: {} → {} colors",
                            solution.chromatic_number,
                            evolved_solution.chromatic_number
                        );
                        solution = evolved_solution;
                    }
                }
                Err(e) => {
                    log::error!("Memetic evolution failed: {}", e);
                    log::warn!("Falling back to best GPU solution");
                }
            }
        }
    }

    log::info!("═══════════════════════════════════════════════════════");
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
