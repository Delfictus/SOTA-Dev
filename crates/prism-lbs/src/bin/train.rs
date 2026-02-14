//! PRISM-LBS Training CLI
//!
//! Train druggability scoring weights using PDBBind dataset
//! and FluxNet reinforcement learning.

use clap::Parser;
use std::path::PathBuf;

use prism_lbs::training::{
    ConservationConfig, EnsembleConfig, LbsTrainer, PdbBindConfig, TrainingConfig,
};

/// PRISM-LBS Training Tool
#[derive(Parser)]
#[command(name = "prism-lbs-train")]
#[command(about = "Train PRISM-LBS druggability scoring weights", long_about = None)]
struct Cli {
    /// PDBBind dataset directory
    #[arg(long)]
    pdbbind_dir: Option<PathBuf>,

    /// PDBBind subset (refined, general, core)
    #[arg(long, default_value = "refined")]
    subset: String,

    /// Conservation data directory
    #[arg(long)]
    conservation_dir: Option<PathBuf>,

    /// Number of training epochs
    #[arg(long, default_value_t = 100)]
    epochs: usize,

    /// Batch size
    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value_t = 0.01)]
    learning_rate: f64,

    /// Validation split ratio
    #[arg(long, default_value_t = 0.2)]
    validation_split: f64,

    /// Distance threshold for success (Å)
    #[arg(long, default_value_t = 4.0)]
    threshold: f64,

    /// Early stopping patience
    #[arg(long, default_value_t = 10)]
    patience: usize,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: PathBuf,

    /// Load checkpoint to resume training
    #[arg(long)]
    resume_from: Option<PathBuf>,

    /// Train ensemble model
    #[arg(long)]
    ensemble: bool,

    /// Output weights file
    #[arg(long)]
    output_weights: Option<PathBuf>,

    /// Output metrics file (JSON)
    #[arg(long)]
    output_metrics: Option<PathBuf>,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum entries to load (for testing)
    #[arg(long)]
    max_entries: Option<usize>,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let log_level = match cli.verbose {
        0 => log::LevelFilter::Info,
        1 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    env_logger::Builder::from_default_env()
        .filter_level(log_level)
        .init();

    log::info!("PRISM-LBS Training Tool v{}", env!("CARGO_PKG_VERSION"));
    log::info!("========================================");

    // Build configuration
    let mut config = TrainingConfig {
        pdbbind: PdbBindConfig {
            root_dir: cli.pdbbind_dir.clone().unwrap_or_else(|| PathBuf::from("data/pdbbind")),
            subset: cli.subset.clone(),
            max_entries: cli.max_entries,
            ..Default::default()
        },
        conservation: ConservationConfig {
            data_dir: cli.conservation_dir.map(|p| p.to_string_lossy().to_string()),
            ..Default::default()
        },
        epochs: cli.epochs,
        batch_size: cli.batch_size,
        learning_rate: cli.learning_rate,
        validation_split: cli.validation_split,
        success_threshold: cli.threshold,
        patience: cli.patience,
        checkpoint_dir: Some(cli.checkpoint_dir.to_string_lossy().to_string()),
        train_ensemble: cli.ensemble,
        seed: cli.seed,
    };

    // Create trainer
    let mut trainer = LbsTrainer::new(config.clone())?;

    // Resume from checkpoint if specified
    if let Some(ref checkpoint_path) = cli.resume_from {
        log::info!("Resuming from checkpoint: {}", checkpoint_path.display());
        trainer.load_checkpoint(checkpoint_path)?;
    }

    // Run training
    log::info!("Starting training...");
    log::info!("  Epochs: {}", cli.epochs);
    log::info!("  Batch size: {}", cli.batch_size);
    log::info!("  Learning rate: {}", cli.learning_rate);
    log::info!("  Threshold: {} Å", cli.threshold);

    let metrics = trainer.train()?;

    // Train ensemble if requested
    let ensemble_config = if cli.ensemble {
        log::info!("Training ensemble model...");
        Some(trainer.train_ensemble()?)
    } else {
        None
    };

    // Output results
    log::info!("========================================");
    log::info!("Training Complete!");
    log::info!("  Best validation score: {:.3}", trainer.best_score());
    log::info!("  Total epochs: {}", metrics.len());

    // Print best weights
    let best_weights = trainer.best_weights();
    log::info!("Best weights:");
    log::info!("  volume:         {:.4}", best_weights.volume);
    log::info!("  hydrophobicity: {:.4}", best_weights.hydrophobicity);
    log::info!("  enclosure:      {:.4}", best_weights.enclosure);
    log::info!("  depth:          {:.4}", best_weights.depth);
    log::info!("  hbond_capacity: {:.4}", best_weights.hbond_capacity);
    log::info!("  flexibility:    {:.4}", best_weights.flexibility);
    log::info!("  conservation:   {:.4}", best_weights.conservation);
    log::info!("  topology:       {:.4}", best_weights.topology);

    // Save weights file
    if let Some(ref path) = cli.output_weights {
        let weights_json = serde_json::to_string_pretty(best_weights)?;
        std::fs::write(path, weights_json)?;
        log::info!("Saved weights to: {}", path.display());
    }

    // Save metrics file
    if let Some(ref path) = cli.output_metrics {
        let output = TrainingOutput {
            metrics,
            best_weights: best_weights.clone(),
            best_score: trainer.best_score(),
            ensemble_config,
        };
        let json = serde_json::to_string_pretty(&output)?;
        std::fs::write(path, json)?;
        log::info!("Saved metrics to: {}", path.display());
    }

    Ok(())
}

#[derive(serde::Serialize)]
struct TrainingOutput {
    metrics: Vec<prism_lbs::training::TrainingMetrics>,
    best_weights: prism_lbs::scoring::ScoringWeights,
    best_score: f64,
    ensemble_config: Option<EnsembleConfig>,
}
