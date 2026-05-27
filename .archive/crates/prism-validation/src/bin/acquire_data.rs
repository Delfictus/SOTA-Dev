//! PRISM Benchmark Data Acquisition CLI
//!
//! Downloads and manages benchmark datasets for heterogeneous dynamics evaluation:
//! - ATLAS: MD-derived RMSF for SOTA comparability
//! - NMR Ensembles: Experimental grounding
//! - MISATO: Protein-ligand MD for drug discovery validation
//!
//! # Usage
//!
//! ```bash
//! # Download all benchmark data
//! cargo run --release -p prism-validation --bin acquire-data -- --all
//!
//! # Download specific datasets
//! cargo run --release -p prism-validation --bin acquire-data -- --atlas --nmr
//!
//! # Check what data is available
//! cargo run --release -p prism-validation --bin acquire-data -- --check
//! ```

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use log::info;

use prism_validation::data_acquisition::{DataAcquisition, DataSource};

#[derive(Parser, Debug)]
#[command(name = "acquire-data")]
#[command(about = "Download benchmark datasets for PRISM heterogeneous dynamics evaluation")]
struct Args {
    /// Base directory for benchmark data
    #[arg(long, default_value = "data/benchmarks")]
    data_dir: PathBuf,

    /// Download ATLAS MD benchmark data (Layer 1: SOTA comparability)
    #[arg(long)]
    atlas: bool,

    /// Download NMR ensemble structures (Layer 2: Experimental grounding)
    #[arg(long)]
    nmr: bool,

    /// Download MISATO protein-ligand data (Layer 3: Drug discovery)
    #[arg(long)]
    misato: bool,

    /// Download all benchmark datasets
    #[arg(long)]
    all: bool,

    /// Check what data is currently available
    #[arg(long)]
    check: bool,

    /// Minimum NMR models required per ensemble
    #[arg(long, default_value = "5")]
    min_nmr_models: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(
        if std::env::args().any(|a| a == "-v" || a == "--verbose") {
            "debug"
        } else {
            "info"
        },
    ))
    .init();

    let args = Args::parse();

    info!("╔═══════════════════════════════════════════════════════════════════════════╗");
    info!("║      PRISM Heterogeneous Benchmark Data Acquisition                        ║");
    info!("╚═══════════════════════════════════════════════════════════════════════════╝");
    info!("");
    info!("  📁 Data directory: {:?}", args.data_dir);
    info!("");

    let mut acq = DataAcquisition::new(&args.data_dir);
    acq.nmr_config.min_models = args.min_nmr_models;

    // Check mode
    if args.check {
        let availability = acq.check_available_data();

        info!("═══════════════════════════════════════════════════════════════════════════");
        info!("                         DATA AVAILABILITY                                  ");
        info!("═══════════════════════════════════════════════════════════════════════════");
        info!("");
        info!("  ┌─────────────────────┬───────────┬──────────┬─────────────────────────┐");
        info!("  │ Dataset             │ Available │ Complete │ Description             │");
        info!("  ├─────────────────────┼───────────┼──────────┼─────────────────────────┤");

        let atlas_status = if availability.atlas_complete {
            "✅"
        } else {
            "⚠️ "
        };
        info!(
            "  │ ATLAS (MD RMSF)     │ {:>9} │    {}    │ SOTA comparability      │",
            availability.atlas_targets, atlas_status
        );

        let nmr_status = if availability.nmr_complete {
            "✅"
        } else {
            "⚠️ "
        };
        info!(
            "  │ NMR Ensembles       │ {:>9} │    {}    │ Experimental grounding  │",
            availability.nmr_ensembles, nmr_status
        );

        let misato_status = if availability.misato_complete {
            "✅"
        } else {
            "⚠️ "
        };
        info!(
            "  │ MISATO (Drug)       │ {:>9} │    {}    │ Drug discovery          │",
            availability.misato_complexes, misato_status
        );

        info!("  └─────────────────────┴───────────┴──────────┴─────────────────────────┘");
        info!("");

        if !availability.atlas_complete
            || !availability.nmr_complete
            || !availability.misato_complete
        {
            info!("  💡 To download missing data, run:");
            info!("     cargo run --release -p prism-validation --bin acquire-data -- --all");
        } else {
            info!("  ✅ All benchmark datasets are available!");
        }

        return Ok(());
    }

    // Download mode
    let download_all = args.all || (!args.atlas && !args.nmr && !args.misato);

    if download_all {
        acq.download_all().await?;
    } else {
        if args.atlas {
            info!("═══════════════════════════════════════════════════════════════════════════");
            info!("  Layer 1: ATLAS MD Benchmark (SOTA Comparability)                          ");
            info!("═══════════════════════════════════════════════════════════════════════════");
            info!("");
            acq.download_atlas().await?;
            info!("");
        }

        if args.nmr {
            info!("═══════════════════════════════════════════════════════════════════════════");
            info!("  Layer 2: NMR Ensembles (Experimental Grounding)                           ");
            info!("═══════════════════════════════════════════════════════════════════════════");
            info!("");
            acq.download_nmr_ensembles().await?;
            info!("");
        }

        if args.misato {
            info!("═══════════════════════════════════════════════════════════════════════════");
            info!("  Layer 3: MISATO (Drug Discovery Validation)                               ");
            info!("═══════════════════════════════════════════════════════════════════════════");
            info!("");
            acq.download_misato().await?;
            info!("");
        }
    }

    info!("");
    info!("╔═══════════════════════════════════════════════════════════════════════════╗");
    info!("║                         NEXT STEPS                                         ║");
    info!("╚═══════════════════════════════════════════════════════════════════════════╝");
    info!("");
    info!("  Run the heterogeneous benchmark:");
    info!("");
    info!("  # Layer 1: ATLAS MD benchmark (SOTA comparability)");
    info!("  cargo run --release -p prism-validation --bin run-dynamics-bench -- \\");
    info!("      --mode enhanced-gnm --data-dir {:?}", acq.atlas_dir());
    info!("");
    info!("  # Layer 2: NMR experimental grounding");
    info!("  cargo run --release -p prism-validation --bin run-dynamics-bench -- \\");
    info!("      --mode enhanced-gnm --data-dir {:?}", acq.nmr_dir());
    info!("");
    info!("  # Full heterogeneous evaluation");
    info!("  cargo run --release -p prism-validation --bin run-heterogeneous-bench -- \\");
    info!("      --data-dir {:?}", args.data_dir);

    Ok(())
}
