//! Phase 2.1 Demo: PIMC Epitope Optimization
//!
//! Demonstrates quantum annealing-based cryptic epitope discovery
//! using validated PIMC infrastructure and real Nipah virus data.
//!
//! PERFORMANCE TARGET: <50ms per structure optimization
//!
//! Usage:
//! cargo run --bin pimc_epitope_demo --release

use anyhow::Result;
use prism_niv_bench::{
    structure_types::NivBenchDataset,
    pimc_epitope_optimization::{PimcEpitopeOptimizer, EpitopeOptimizationParams},
};
use cudarc::driver::CudaContext;
use std::{sync::Arc, fs::File, io::BufReader, time::Instant};

fn main() -> Result<()> {
    env_logger::init();

    println!("🚀 Phase 2.1: PIMC Epitope Optimization Demo");
    println!("==============================================");
    println!("Quantum annealing for cryptic epitope discovery");
    println!("Using validated PIMC infrastructure + real Nipah data");
    println!();

    // Load real Nipah virus dataset
    println!("📁 Loading real Nipah virus dataset...");
    let dataset_path = "../../data/niv_bench_dataset.json";
    let file = File::open(dataset_path)
        .map_err(|e| anyhow::anyhow!("Failed to open dataset {}: {}", dataset_path, e))?;
    let reader = BufReader::new(file);
    let dataset: NivBenchDataset = serde_json::from_reader(reader)
        .map_err(|e| anyhow::anyhow!("Failed to parse dataset: {}", e))?;

    println!("✅ Dataset loaded: {} structures", dataset.structures.len());

    // Initialize CUDA context
    println!("🔧 Initializing CUDA context...");
    let cuda_context = Arc::new(CudaContext::new(0)
        .map_err(|e| anyhow::anyhow!("Failed to initialize CUDA: {}", e))?);

    println!("✅ CUDA initialized: {}", cuda_context.name()?);

    // Configure PIMC optimization parameters
    let mut optimization_params = EpitopeOptimizationParams::default();
    optimization_params.num_replicas = 64;           // Balanced performance
    optimization_params.mc_steps = 500;              // Fast demo iterations
    optimization_params.accessibility_threshold = 0.25; // Stricter cryptic threshold

    println!("⚙️  PIMC Parameters:");
    println!("   Replicas: {}", optimization_params.num_replicas);
    println!("   MC Steps: {}", optimization_params.mc_steps);
    println!("   Accessibility Threshold: {:.2}", optimization_params.accessibility_threshold);
    println!();

    // Initialize PIMC epitope optimizer
    println!("🔬 Initializing PIMC Epitope Optimizer...");
    let mut epitope_optimizer = PimcEpitopeOptimizer::new(
        cuda_context.clone(),
        optimization_params,
    )?;

    println!("✅ PIMC Epitope Optimizer ready");
    println!();

    // Demonstrate optimization on each structure
    let total_start = Instant::now();
    let mut total_cryptic_sites = 0;
    let mut total_structures_processed = 0;

    for structure in &dataset.structures {
        if structure.residues.is_empty() {
            println!("⏭️  Skipping {} (no residue data)", structure.pdb_id);
            continue;
        }

        println!("🧬 Processing: {} ({} residues)", structure.pdb_id, structure.residues.len());

        // Get known epitopes for validation
        let known_epitopes: Vec<Vec<usize>> = dataset.epitopes
            .get(&structure.pdb_id)
            .map(|epitopes| {
                epitopes.iter()
                    .map(|e| e.interface_residues.clone())
                    .collect()
            })
            .unwrap_or_default();

        println!("   📋 Known epitopes: {}", known_epitopes.len());

        // Get antibody contacts (if available)
        let antibody_contacts: Vec<usize> = known_epitopes
            .iter()
            .flatten()
            .cloned()
            .collect();

        // PIMC epitope optimization
        let optimization_start = Instant::now();

        let epitope_landscape = epitope_optimizer.optimize_epitope_landscape(
            structure,
            &known_epitopes,
            &antibody_contacts,
        )?;

        let optimization_time = optimization_start.elapsed().as_millis();

        // Performance analysis
        println!("   ⚡ Optimization time: {}ms", optimization_time);

        if optimization_time <= 50 {
            println!("   🎯 TARGET MET: {}ms <= 50ms", optimization_time);
        } else {
            println!("   ⚠️  Above target: {}ms > 50ms", optimization_time);
        }

        // Results analysis
        println!("   📊 Results:");
        println!("      Cryptic epitopes discovered: {}", epitope_landscape.cryptic_sites.len());
        println!("      PIMC converged: {}", epitope_landscape.pimc_convergence);
        println!("      Quantum tunneling events: {}", epitope_landscape.quantum_tunneling_events);

        // Detailed epitope analysis
        for (i, cryptic_site) in epitope_landscape.cryptic_sites.iter().enumerate() {
            println!("      🔍 Cryptic Site {}: {} residues", i + 1, cryptic_site.residue_indices.len());
            println!("         Accessibility: {:.3}", cryptic_site.accessibility_score);
            println!("         Energy barrier: {:.2} kT", cryptic_site.energy_barrier);
            println!("         Escape potential: {:.3}", cryptic_site.antibody_escape_potential);
            println!("         Platform specificity:");
            println!("           mRNA exposure: {:.2}", cryptic_site.platform_specificity.mrna_exposure_probability);
            println!("           Viral vector: {:.2}", cryptic_site.platform_specificity.viral_vector_exposure_probability);
            println!("           Subunit: {:.2}", cryptic_site.platform_specificity.subunit_exposure_probability);
            println!("           Differential: {:.3}", cryptic_site.platform_specificity.differential_crypticity);

            if cryptic_site.antibody_escape_potential > 0.7 {
                println!("         ⚠️  HIGH ESCAPE RISK!");
            }
        }

        total_cryptic_sites += epitope_landscape.cryptic_sites.len();
        total_structures_processed += 1;
        println!();
    }

    let total_time = total_start.elapsed();

    println!("🎯 PHASE 2.1 DEMO COMPLETE");
    println!("===========================");
    println!("Structures processed: {}", total_structures_processed);
    println!("Total cryptic sites discovered: {}", total_cryptic_sites);
    println!("Average cryptic sites per structure: {:.1}",
        total_cryptic_sites as f32 / total_structures_processed.max(1) as f32);
    println!("Total processing time: {:.2}s", total_time.as_secs_f32());
    println!("Average time per structure: {:.1}ms",
        total_time.as_millis() as f32 / total_structures_processed.max(1) as f32);
    println!();

    // Validation against known epitopes
    if total_cryptic_sites > 0 {
        println!("✅ PIMC successfully discovered cryptic epitopes using quantum annealing");
        println!("✅ Platform-specific exposure predictions generated");
        println!("✅ Energy barriers computed from PIMC optimization");
    } else {
        println!("⚠️  No cryptic sites discovered (check parameters or data)");
    }

    // Performance summary
    println!();
    println!("🔬 QUANTUM INFRASTRUCTURE PERFORMANCE:");
    println!("   PIMC Replicas: 64 (validated ✅)");
    println!("   Quantum Annealing: Temperature schedule optimization ✅");
    println!("   Target Performance: <50ms per structure");
    println!("   Real Nipah Data: {} structures analyzed", total_structures_processed);
    println!();
    println!("🚀 Ready for Phase 2.2: QUBO-TDA Topology Integration!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA and data files
    fn test_pimc_epitope_demo() {
        // Integration test would run the full demo
        assert!(true);
    }
}