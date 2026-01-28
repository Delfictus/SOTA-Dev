//! NiV-Bench: Neuromorphic Benchmark for Cryptic Epitope Prediction
//!
//! CLI implementation for NiV-Bench.

use anyhow::{Result, Context};
use clap::{Parser, Subcommand};
use log::{info, warn, error};
use std::path::Path;
use prism_niv_bench::structure_types::{ParamyxoStructure, VirusType, ProteinType};
use prism_niv_bench::pdb_fetcher;
use prism_niv_bench::data_loader::DataLoader;
use prism_niv_bench::ground_truth;
// use prism_niv_bench::{PackedBatch, BatchStructureDesc};  // Disabled for now
// use prism_niv_bench::gpu_parallel::ParallelGpuPipeline;  // Disabled until vendored fixed
// use prism_niv_bench::glycan_mask::GlycanMask;  // Disabled for now

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Download and validate NiV/HeV PDB structures
    Download {
        /// Output directory for structures
        #[arg(short, long, default_value = "data/niv_structures")]
        output_dir: String,
    },
    /// Run benchmark mode
    Benchmark {
        /// Run actual GPU kernels
        #[arg(long, default_value = "true")]
        use_gpu: bool,
    },
    /// Train FluxNet agent using Evolutionary Strategy (Zero-Copy)
    Train {
        #[arg(long, default_value = "10")]
        generations: usize,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    let args = Args::parse();

    println!("NiV-Bench v0.3.0 - Nipah Virus Cryptic Epitope Benchmark");
    println!("{}", "=".repeat(60));

    match args.command {
        Commands::Download { output_dir } => {
            download_structures(&output_dir).await?;
        }
        Commands::Benchmark { use_gpu: _ } => {
            info!("Running data validation benchmark...");
            tokio::task::spawn_blocking(move || run_data_benchmark()).await??;
        }
        Commands::Train { generations } => {
            info!("Starting Evolutionary Training (Zero-Copy GPU)...");
            tokio::task::spawn_blocking(move || run_training(generations)).await??;
        }
    }

    Ok(())
}

fn run_training(_generations: usize) -> Result<()> {
    // TODO: Re-enable after fixing vendored dependencies
    anyhow::bail!("Training mode temporarily disabled - fix vendored imports first");

    /* DISABLED UNTIL VENDORED FIXED
    use prism_niv_bench::gpu_parallel::ParallelGpuPipeline;
    use prism_niv_bench::vendored::evolution_gpu::{EvolutionGpu, EvolutionConfig};
    use prism_niv_bench::vendored::fluxnet_gpu::DQNWeights;
    use prism_niv_bench::fluxnet_dqn_zero_copy::{compute_cryptic_reward, CrypticAction};
    use prism_niv_bench::fluxnet_niv::GroundTruth;
    use prism_niv_bench::structure_types::NivBenchDataset;
    use std::fs::File;
    use std::io::BufReader;

    // 1. Load Ground Truth
    let dataset_path = "data/niv_bench_dataset_real.json";
    if !std::path::Path::new(dataset_path).exists() {
        anyhow::bail!("Dataset not found at {}. Please provide data.", dataset_path);
    }
    let file = File::open(dataset_path)?;
    let reader = BufReader::new(file);
    let dataset: NivBenchDataset = serde_json::from_reader(reader)?;
    let ground_truth = GroundTruth::new(&dataset);

    // 2. Initialize GPU Pipeline
    let mut pipeline = ParallelGpuPipeline::new()?;
    let ptx_path = std::path::Path::new("kernels/ptx");
    
    // 3. Initialize Evolution Strategy
    let evo_config = EvolutionConfig::default();
    let mut evo = EvolutionGpu::new(pipeline.device.clone(), ptx_path, evo_config.clone())?;
    
    // Initialize Mean Weights on GPU
    let mut mean_weights = pipeline.stream.alloc_zeros::<f32>(evo_config.n_params)?;
    
    // 4. Prepare Training Data (8XPS)
    // We compute features ONCE on GPU to save time
    let pdb_path = "data/niv_structures/8XPS.pdb";
    let mut structure = pdb_fetcher::parse_pdb(pdb_path).context("Failed to parse 8XPS")?;
    
    // Fake sequence for masking
    let seq: String = structure.residues.iter().map(|_| 'A').collect(); 
    structure.sequence = seq.clone();
    let mut seq_map = std::collections::HashMap::new();
    seq_map.insert("8XPS".to_string(), seq);

    println!("Extracting features for 8XPS...");
    let mut batch = create_batch_from_structure(&structure)?;
    // Run pipeline to populate features (we ignore q_values for now)
    // We need to capture the intermediate features.
    // Hack: execute_batch runs FluxNet with current (random) weights.
    // But we want to reuse the FEATURES for the population.
    // `ParallelGpuPipeline` doesn't expose features easily.
    // For this demo, we will run the FULL pipeline for each individual.
    // It's 42ms * 64 = 2.6s per generation. Acceptable.
    
    println!("Training for {} generations (Pop: {})", generations, evo_config.population_size);
    
    for gen in 0..generations {
        let start = std::time::Instant::now();
        
        // A. Perturb Weights
        evo.perturb(&mean_weights)?;
        
        // B. Evaluate Population
        let mut fitness_scores = vec![0.0f32; evo_config.population_size];
        let mut total_reward = 0.0;
        
        for i in 0..evo_config.population_size {
            // 1. Get weights for individual i
            let ind_weights = evo.get_individual(i)?;
            
            // 2. Load into FluxNet (Device to Device copy would be better, but CPU roundtrip works)
            // ind_weights is Vec<f32>. We need to upload as bytes.
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    ind_weights.as_ptr() as *const u8,
                    ind_weights.len() * 4
                )
            };
            pipeline.stream.memcpy_htod(byte_slice, &mut pipeline.fluxnet_gpu.weights_buffer)?;
            
            // 3. Run Inference
            let result = pipeline.execute_batch(&mut batch, Some(&seq_map))?;
            
            // 4. Compute Reward
            // Sum rewards across all residues in 8XPS
            let mut ind_reward = 0.0;
            if let Some(s) = result.structures.first() {
                for (res_idx, chunk) in s.q_values.chunks(4).enumerate() {
                    // Argmax
                    let mut best_action = 0;
                    let mut max_q = chunk[0];
                    for k in 1..4 {
                        if chunk[k] > max_q { max_q = chunk[k]; best_action = k; }
                    }
                    
                    let action = CrypticAction::from(best_action);
                    let is_cryptic = ground_truth.is_cryptic("8XPS", res_idx);
                    let is_epitope = ground_truth.is_epitope("8XPS", res_idx);
                    
                    ind_reward += compute_cryptic_reward(action, is_cryptic, is_epitope, 1.0);
                }
            }
            
            fitness_scores[i] = ind_reward;
            total_reward += ind_reward;
        }
        
        // C. Update Weights
        let mut d_fitness = pipeline.stream.alloc_zeros::<f32>(evo_config.population_size)?;
        pipeline.stream.memcpy_htod(&fitness_scores, &mut d_fitness)?;
        
        evo.update(&mut mean_weights, &d_fitness)?;
        
        let avg_reward = total_reward / evo_config.population_size as f32;
        println!("Generation {}: Avg Reward = {:.2} (Time: {:.2?} ms)", gen, avg_reward, start.elapsed().as_millis());
    }
    
    // Save Best Weights (Mean of Population)
    println!("Saving best weights to models/fluxnet_best.bin...");
    let mut best_weights_vec = vec![0.0f32; evo_config.n_params];
    pipeline.stream.memcpy_dtoh(&mean_weights, &mut best_weights_vec)?;
    
    // Check for NaNs
    if best_weights_vec.iter().any(|&x| x.is_nan()) {
        log::error!("TRAINING FAILED: Evolved weights contain NaN. Skipping save.");
        return Ok(());
    }

    // Convert to bytes and save
    let best_bytes = unsafe {
        std::slice::from_raw_parts(
            best_weights_vec.as_ptr() as *const u8,
            best_weights_vec.len() * 4
        )
    };
    DQNWeights::save_bytes(best_bytes, std::path::Path::new("models/fluxnet_best.bin"))?;
    println!("Weights saved successfully.");

    println!("Training Complete. Weights evolved on GPU.");
    Ok(())
    */ // END DISABLED BLOCK
}

async fn download_structures(output_dir: &str) -> Result<()> {
    info!("Downloading NiV/HeV PDB structures to {}", output_dir);
    // ... (Same download logic as before) ...
    let niv_structures = vec![
        "8XPS", "8XQ3", "7UPK", "7UPD", "7UPB",
        "8ZPV", "7SKT", "7TY0"
    ];
    let hev_structures = vec![
        "2X9M", "5EJB", "6CMG", "7UPH"
    ];

    std::fs::create_dir_all(output_dir).context("Failed to create output directory")?;
    let client = reqwest::Client::new();

    for pdb_id in niv_structures.iter().chain(hev_structures.iter()) {
        let url = format!("https://files.rcsb.org/download/{}.pdb", pdb_id);
        let output_path = format!("{}/{}.pdb", output_dir, pdb_id);

        if Path::new(&output_path).exists() {
            info!("Structure {} already exists", pdb_id);
            continue;
        }
        info!("Downloading {} -> {}", pdb_id, output_path);
        let response = client.get(&url).send().await?;
        if response.status().is_success() {
            let content = response.text().await?;
            std::fs::write(&output_path, content)?;
            info!("Downloaded {} ({} bytes)", pdb_id, std::fs::metadata(&output_path)?.len());
        } else {
            warn!("Failed to download {}: HTTP {}", pdb_id, response.status());
        }
    }
    info!("Structure download complete");
    Ok(())
}

fn run_data_benchmark() -> Result<()> {
    println!("=== NiV-Bench Data Validation ===\n");

    let data_dir = "data/niv_structures";
    let loader = DataLoader::new(data_dir);

    let targets = vec![
        ("8XPS", "Nipah G Protein (Monomer + n425 sdAb)"),
        ("8XQ3", "Nipah G Protein (Tetramer + n425 sdAb)"),
        ("7TY0", "Nipah G + nAH1.3"),
        ("7UPK", "Nipah F + Fab 1A9"),
        ("2X9M", "Hendra G Protein (Monomer)"),
    ];

    let mut structures = Vec::new();

    println!("Loading {} PDB structures...\n", targets.len());
    for (pdb_id, desc) in targets {
        let pdb_path = format!("{}/{}.pdb", data_dir, pdb_id);
        if !std::path::Path::new(&pdb_path).exists() {
            warn!("{} not found at {}", pdb_id, pdb_path);
            continue;
        }

        println!("Parsing {} ({})...", pdb_id, desc);
        match loader.parse_pdb(pdb_id) {
            Ok(s) => {
                println!("  ✓ {} residues, {} atoms", s.residues.len(), s.atoms.len());
                structures.push(s);
            },
            Err(e) => {
                error!("  ✗ Parse failed: {}", e);
            }
        }
    }

    println!("\n=== Ground Truth Extraction ===\n");
    match ground_truth::extract_ground_truth(&structures) {
        Ok(dataset) => {
            println!("Dataset created successfully:");
            println!("  Structures: {}", dataset.structures.len());
            println!("  Cryptic sites: {}", dataset.cryptic_sites.len());
            println!("  Epitopes: {}", dataset.epitopes.len());
            println!("  Train/Test/Val splits: {}/{}/{}",
                dataset.train_structures.len(),
                dataset.test_structures.len(),
                dataset.validation_structures.len());

            // Save dataset
            let output_path = "data/niv_bench_dataset_validated.json";
            let file = std::fs::File::create(output_path)?;
            serde_json::to_writer_pretty(file, &dataset)?;
            println!("\n✓ Dataset saved to {}", output_path);
        },
        Err(e) => {
            error!("Ground truth extraction failed: {}", e);
        }
    }

    println!("\n=== Benchmark Complete ===");
    Ok(())
}

/* DISABLED: GPU batch processing not needed for data validation
fn create_batch_from_structure(structure: &ParamyxoStructure) -> Result<PackedBatch> {
    let mut atoms_packed = Vec::new();
    let mut ca_indices_packed = Vec::new();
    let mut residue_types_packed = Vec::new();
    let mut bfactor_packed = Vec::new();
    let mut burial_packed = Vec::new();
    
    // Flatten atoms
    for atom in &structure.atoms {
        atoms_packed.push(atom.x);
        atoms_packed.push(atom.y);
        atoms_packed.push(atom.z);
    }
    
    // Find CA indices
    for res in &structure.residues {
        // Find CA atom index in the flattened list (stride 3)
        // structure.atoms is the source of truth
        if let Some(idx) = structure.atoms.iter().position(|a| a.id == res.atoms.iter().find(|ra| ra.name == "CA").map(|ca| ca.id).unwrap_or(0)) {
            ca_indices_packed.push(idx as i32);
        } else {
             // Fallback: use first atom
             if let Some(first) = res.atoms.first() {
                 if let Some(idx) = structure.atoms.iter().position(|a| a.id == first.id) {
                     ca_indices_packed.push(idx as i32);
                 } else {
                     ca_indices_packed.push(0); // Should not happen
                 }
             } else {
                 ca_indices_packed.push(0);
             }
        }
        
        // Residue type (simplified mapping)
        let type_id = match res.name.as_str() {
            "ALA" => 0, "CYS" => 1, "ASP" => 2, "GLU" => 3, "PHE" => 4,
            "GLY" => 5, "HIS" => 6, "ILE" => 7, "LYS" => 8, "LEU" => 9,
            "MET" => 10, "ASN" => 11, "PRO" => 12, "GLN" => 13, "ARG" => 14,
            "SER" => 15, "THR" => 16, "VAL" => 17, "TRP" => 18, "TYR" => 19,
            _ => 20, // Unknown
        };
        residue_types_packed.push(type_id);
        
        // B-factor (avg of atoms or CA)
        // let b = res.atoms.iter().map(|a| 0.0).sum::<f32>(); // PDB parser didn't extract B-factor!
        // My parser didn't parse B-factor. I should update it or use 0.0
        bfactor_packed.push(0.0);
        
        // Default burial to 1.0 (Exposed). Glycan masking (Stage 0) will set to 0.0 (Buried).
        burial_packed.push(1.0);
    }

    let n_residues = structure.residues.len() as i32;
    let n_atoms = structure.atoms.len() as i32;

    let desc = BatchStructureDesc {
        atom_offset: 0,
        residue_offset: 0,
        n_atoms,
        n_residues,
    };

    Ok(PackedBatch {
        descriptors: vec![desc],
        ids: vec![structure.pdb_id.clone()],
        atoms_packed,
        ca_indices_packed,
        conservation_packed: vec![0.5; n_residues as usize], // Dummy conservation
        bfactor_packed,
        burial_packed,
        residue_types_packed,
        total_atoms: structure.atoms.len(),
        total_residues: structure.residues.len(),
        frequencies_packed: vec![],
        velocities_packed: vec![],
        p_neut_time_series_75pk_packed: vec![],
        current_immunity_levels_75_packed: vec![],
        pk_params_packed: vec![],
        epitope_escape_packed: vec![],
    })
}
*/  // END DISABLED GPU batch processing
