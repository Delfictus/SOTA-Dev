//! Demonstration of fpocket integration for gold-standard pocket detection
//!
//! This example shows how to use the fpocket FFI integration to detect
//! binding pockets using the validated fpocket algorithm.
//!
//! Requirements:
//! - fpocket must be installed and in PATH
//! - Run: `sudo apt-get install fpocket` (Ubuntu/Debian)
//!   or download from: https://github.com/Discngine/fpocket
//!
//! Usage:
//! ```bash
//! cargo run --release --example fpocket_demo --features fpocket -- <path/to/protein.pdb>
//! ```

use prism_lbs::pocket::fpocket_ffi::{fpocket_available, run_fpocket, FpocketConfig};
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Check command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path/to/protein.pdb>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  cargo run --release --example fpocket_demo --features fpocket -- data/1hiv.pdb");
        std::process::exit(1);
    }

    let pdb_path = Path::new(&args[1]);

    // Check if fpocket is available
    if !fpocket_available() {
        eprintln!("Error: fpocket not found in PATH");
        eprintln!("\nTo install fpocket:");
        eprintln!("  Ubuntu/Debian: sudo apt-get install fpocket");
        eprintln!("  macOS: brew install fpocket");
        eprintln!("  Source: https://github.com/Discngine/fpocket");
        std::process::exit(1);
    }

    println!("fpocket found!");
    println!("Analyzing: {}", pdb_path.display());
    println!();

    // Configure fpocket
    let config = FpocketConfig {
        min_alpha_radius: 3.0,
        max_pockets: 20,
        druggability_threshold: 0.0, // Accept all pockets
        ..Default::default()
    };

    // Run fpocket
    println!("Running fpocket...");
    let pockets = run_fpocket(pdb_path, &config)?;

    println!("\nfpocket detected {} pockets:", pockets.len());
    println!();

    // Display results
    for (i, pocket) in pockets.iter().enumerate() {
        println!("Pocket {} (ID: {})", i + 1, i);
        println!("  Centroid:       ({:.2}, {:.2}, {:.2}) Å",
                 pocket.centroid[0], pocket.centroid[1], pocket.centroid[2]);
        println!("  Volume:         {:.2} ų", pocket.volume);
        println!("  Druggability:   {:.3}", pocket.druggability_score.total);
        println!("  Classification: {:?}", pocket.druggability_score.classification);
        println!("  Atoms:          {}", pocket.atom_indices.len());
        println!("  Residues:       {}", pocket.residue_indices.len());
        println!("  Mean SASA:      {:.2} ų", pocket.mean_sasa);
        println!("  Hydrophobicity: {:.3}", pocket.mean_hydrophobicity);
        println!();
    }

    // Summary statistics
    if !pockets.is_empty() {
        let avg_volume: f64 = pockets.iter().map(|p| p.volume).sum::<f64>() / pockets.len() as f64;
        let avg_drug: f64 = pockets.iter().map(|p| p.druggability_score.total).sum::<f64>()
            / pockets.len() as f64;

        println!("Summary:");
        println!("  Average volume:       {:.2} ų", avg_volume);
        println!("  Average druggability: {:.3}", avg_drug);

        // Best pocket
        let best = pockets
            .iter()
            .max_by(|a, b| a.druggability_score.total.partial_cmp(&b.druggability_score.total).unwrap())
            .unwrap();
        println!("\nBest pocket (highest druggability):");
        println!("  Centroid: ({:.2}, {:.2}, {:.2}) Å",
                 best.centroid[0], best.centroid[1], best.centroid[2]);
        println!("  Volume:   {:.2} ų", best.volume);
        println!("  Score:    {:.3}", best.druggability_score.total);
    }

    Ok(())
}
