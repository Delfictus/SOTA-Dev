//! PRISM-LBS Reservoir Readout Training for CryptoBench
//!
//! Trains a linear readout layer on 70-dim combined features using ridge regression.
//! Uses the SAME inference kernel (mega_fused_pocket_kernel) for training to ensure
//! feature consistency between training and inference.
//!
//! ## SESSION 9 SOTA Feature Set (70-dim)
//! - 16 base geometric features (centrality, conservation, burial, etc.)
//! - 12 reservoir state features (dendritic branches)
//! - 12 physics features (thermodynamic, quantum, information-theoretic)
//! - 30 SOTA biochemical features (hydrophobicity, electrostatics, shape, etc.)
//!
//! ## Critical Design Decision
//! We use the inference kernel (not batch training kernel) because:
//! - Inference kernel computes full 70-dim SOTA features
//! - Training and inference MUST see identical features!
//!
//! Usage:
//!   cargo run --release --bin train-readout -- \
//!     --pdb-dir /path/to/cryptobench/pdb-files \
//!     --dataset /path/to/cryptobench/dataset.json \
//!     --folds /path/to/cryptobench/folds.json \
//!     --output readout_weights.bin

use clap::Parser;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use prism_lbs::ProteinStructure;

#[cfg(feature = "cuda")]
use prism_gpu::{
    MegaFusedGpu, MegaFusedConfig,
    readout_training::{ReservoirStateCollector, TrainedReadout},
};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;

/// Dimension of combined features from inference kernel (SESSION 9 SOTA)
const INFERENCE_FEATURE_DIM: usize = 70;  // 16 base + 12 reservoir + 12 physics + 30 SOTA

/// PRISM-LBS Readout Training Tool for CryptoBench
#[derive(Parser)]
#[command(name = "train-readout")]
#[command(about = "Train reservoir readout weights on CryptoBench dataset using inference kernel")]
struct Cli {
    /// Directory containing PDB files
    #[arg(long)]
    pdb_dir: PathBuf,

    /// CryptoBench dataset.json file
    #[arg(long)]
    dataset: PathBuf,

    /// CryptoBench folds.json file
    #[arg(long)]
    folds: PathBuf,

    /// Output weights file
    #[arg(long, default_value = "readout_weights.bin")]
    output: PathBuf,

    /// Ridge regularization lambda
    #[arg(long, default_value_t = 1e-4)]
    lambda: f32,

    /// Batch size for GPU processing (smaller = safer for large structures)
    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    /// Maximum structures to process (for testing)
    #[arg(long)]
    max_structures: Option<usize>,

    /// Skip test evaluation
    #[arg(long)]
    skip_eval: bool,

    /// PTX directory
    #[arg(long, env = "PRISM_PTX_DIR")]
    ptx_dir: Option<PathBuf>,

    /// Disable z-score normalization
    #[arg(long)]
    no_normalize: bool,

    /// Positive class weight for imbalanced training (default: sqrt of imbalance ratio)
    /// Lower values (1.0-4.0) = higher precision, lower recall
    /// Higher values (8.0+) = higher recall, lower precision
    #[arg(long)]
    pos_weight: Option<f32>,

    /// Export features to NPY directory instead of training (SESSION 10B)
    #[arg(long)]
    export_features: Option<PathBuf>,
}

/// Structure data for training (simpler than batch input)
#[cfg(feature = "cuda")]
struct TrainingStructure {
    id: String,
    atoms: Vec<f32>,           // [n_atoms * 3]
    ca_indices: Vec<i32>,      // [n_residues]
    conservation: Vec<f32>,    // [n_residues]
    bfactor: Vec<f32>,         // [n_residues]
    burial: Vec<f32>,          // [n_residues]
    gt_mask: Vec<u8>,          // [n_residues] ground truth labels
}

/// SESSION 10B: Helper to convert ReservoirStateCollector to flat per-residue arrays
#[cfg(feature = "cuda")]
fn collector_to_flat_arrays(collector: &ReservoirStateCollector) -> (Vec<Vec<f32>>, Vec<u8>) {
    let mut all_features = Vec::new();
    let mut all_labels = Vec::new();

    for (features_flat, labels) in &collector.data {
        let n_residues = labels.len();
        let feature_dim = if n_residues > 0 {
            features_flat.len() / n_residues
        } else {
            INFERENCE_FEATURE_DIM
        };

        // Extract per-residue features
        for r in 0..n_residues {
            let offset = r * feature_dim;
            if offset + feature_dim <= features_flat.len() {
                let residue_features: Vec<f32> = features_flat[offset..offset + feature_dim].to_vec();
                all_features.push(residue_features);
                all_labels.push(labels[r]);
            }
        }
    }

    (all_features, all_labels)
}

/// SESSION 10B: Export features to NPY format for Python ensemble training
#[cfg(feature = "cuda")]
fn export_features_to_npy(
    train_features: &[Vec<f32>],
    train_labels: &[u8],
    test_features: &[Vec<f32>],
    test_labels: &[u8],
    output_dir: &PathBuf,
) -> anyhow::Result<()> {
    use ndarray::{Array1, Array2};
    use ndarray_npy::WriteNpyExt;
    use std::fs::File;
    use std::io::BufWriter;

    std::fs::create_dir_all(output_dir)?;

    let n_features = if train_features.is_empty() { 70 } else { train_features[0].len() };

    log::info!("═══════════════════════════════════════════════════════════════");
    log::info!("EXPORTING FEATURES TO NPY");
    log::info!("═══════════════════════════════════════════════════════════════");
    log::info!("Train samples: {}", train_features.len());
    log::info!("Test samples:  {}", test_features.len());
    log::info!("Features:      {}", n_features);
    log::info!("Output dir:    {}", output_dir.display());

    // Convert train features to ndarray
    let n_train = train_features.len();
    let mut train_array = Array2::<f32>::zeros((n_train, n_features));
    for (i, feat) in train_features.iter().enumerate() {
        for (j, &val) in feat.iter().enumerate() {
            if j < n_features {
                train_array[[i, j]] = val;
            }
        }
    }

    // Convert train labels to f32 for compatibility
    let train_labels_array = Array1::<f32>::from_iter(
        train_labels.iter().map(|&l| l as f32)
    );

    // Convert test features
    let n_test = test_features.len();
    let mut test_array = Array2::<f32>::zeros((n_test, n_features));
    for (i, feat) in test_features.iter().enumerate() {
        for (j, &val) in feat.iter().enumerate() {
            if j < n_features {
                test_array[[i, j]] = val;
            }
        }
    }

    // Convert test labels
    let test_labels_array = Array1::<f32>::from_iter(
        test_labels.iter().map(|&l| l as f32)
    );

    // Write train features
    let train_feat_path = output_dir.join("train_features.npy");
    let file = File::create(&train_feat_path)?;
    let mut writer = BufWriter::new(file);
    train_array.write_npy(&mut writer)?;
    log::info!("✅ Saved: {} ({} x {})", train_feat_path.display(), n_train, n_features);

    // Write train labels
    let train_label_path = output_dir.join("train_labels.npy");
    let file = File::create(&train_label_path)?;
    let mut writer = BufWriter::new(file);
    train_labels_array.write_npy(&mut writer)?;
    log::info!("✅ Saved: {} ({})", train_label_path.display(), n_train);

    // Write test features
    let test_feat_path = output_dir.join("test_features.npy");
    let file = File::create(&test_feat_path)?;
    let mut writer = BufWriter::new(file);
    test_array.write_npy(&mut writer)?;
    log::info!("✅ Saved: {} ({} x {})", test_feat_path.display(), n_test, n_features);

    // Write test labels
    let test_label_path = output_dir.join("test_labels.npy");
    let file = File::create(&test_label_path)?;
    let mut writer = BufWriter::new(file);
    test_labels_array.write_npy(&mut writer)?;
    log::info!("✅ Saved: {} ({})", test_label_path.display(), n_test);

    // Print class distribution
    let train_pos = train_labels.iter().filter(|&&l| l == 1).count();
    let train_neg = train_labels.len() - train_pos;
    let test_pos = test_labels.iter().filter(|&&l| l == 1).count();
    let test_neg = test_labels.len() - test_pos;

    log::info!("");
    log::info!("Class Distribution:");
    log::info!("  Train: {} positive, {} negative (ratio: {:.1}:1)",
          train_pos, train_neg, train_neg as f64 / train_pos.max(1) as f64);
    log::info!("  Test:  {} positive, {} negative (ratio: {:.1}:1)",
          test_pos, test_neg, test_neg as f64 / test_pos.max(1) as f64);
    log::info!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    log::info!("═══════════════════════════════════════════════════════════════════");
    log::info!("  PRISM-LBS Reservoir Readout Training (70-DIM INFERENCE KERNEL)");
    log::info!("  Dataset: {:?}", cli.dataset);
    log::info!("  PDB dir: {:?}", cli.pdb_dir);
    log::info!("  Lambda: {}", cli.lambda);
    log::info!("  Feature dimension: {} (16 base + 12 reservoir + 12 physics + 30 SOTA)", INFERENCE_FEATURE_DIM);
    log::info!("  Z-score normalization: {}", !cli.no_normalize);
    if let Some(pw) = cli.pos_weight {
        log::info!("  Pos weight: {} (manual)", pw);
    } else {
        log::info!("  Pos weight: sqrt(imbalance) (auto)");
    }
    log::info!("═══════════════════════════════════════════════════════════════════");

    // Set PTX path if provided
    if let Some(ref ptx_dir) = cli.ptx_dir {
        std::env::set_var("PRISM_PTX_DIR", ptx_dir);
    }

    #[cfg(feature = "cuda")]
    {
        run_training(&cli)?;
    }

    #[cfg(not(feature = "cuda"))]
    {
        anyhow::bail!("This binary requires CUDA feature to be enabled");
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn run_training(cli: &Cli) -> anyhow::Result<()> {
    let total_start = Instant::now();

    // 1. Parse folds.json to get train/test split
    let folds: serde_json::Value = serde_json::from_str(&fs::read_to_string(&cli.folds)?)?;

    let test_set: HashSet<String> = folds.get("test")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
            .collect())
        .unwrap_or_default();

    // Collect all train PDB IDs (train-0, train-1, train-2, train-3)
    let mut train_set: HashSet<String> = HashSet::new();
    for key in ["train-0", "train-1", "train-2", "train-3"] {
        if let Some(arr) = folds.get(key).and_then(|v| v.as_array()) {
            for v in arr {
                if let Some(s) = v.as_str() {
                    train_set.insert(s.to_lowercase());
                }
            }
        }
    }

    log::info!("Loaded folds: {} train, {} test structures", train_set.len(), test_set.len());

    // 2. Parse dataset.json for ground truth
    let dataset: serde_json::Value = serde_json::from_str(&fs::read_to_string(&cli.dataset)?)?;
    let ground_truth = parse_ground_truth(&dataset)?;
    log::info!("Parsed ground truth for {} structures", ground_truth.len());

    // 3. Load training structures with ground truth
    let load_start = Instant::now();
    let mut train_structures = Vec::new();
    let mut missing_pdb = 0usize;
    let mut missing_gt = 0usize;

    for pdb_id in &train_set {
        // Check limit
        if let Some(max) = cli.max_structures {
            if train_structures.len() >= max {
                break;
            }
        }

        // Find PDB file
        let pdb_path = cli.pdb_dir.join(format!("{}.pdb", pdb_id));
        if !pdb_path.exists() {
            missing_pdb += 1;
            continue;
        }

        // Load structure
        let structure = match ProteinStructure::from_pdb_file(&pdb_path) {
            Ok(s) => s,
            Err(e) => {
                log::debug!("Failed to load {}: {}", pdb_id, e);
                continue;
            }
        };

        // Get ground truth
        let gt_residues = match ground_truth.get(pdb_id) {
            Some(residues) => residues,
            None => {
                missing_gt += 1;
                continue;
            }
        };

        // Create training structure
        let ts = create_training_structure(&structure, pdb_id, gt_residues)?;
        train_structures.push(ts);
    }

    log::info!(
        "Loaded {} train structures in {:?} (missing: {} PDB, {} GT)",
        train_structures.len(), load_start.elapsed(), missing_pdb, missing_gt
    );

    if train_structures.is_empty() {
        anyhow::bail!("No training structures loaded");
    }

    // 4. Initialize GPU with INFERENCE kernel (not batch kernel!)
    let gpu_start = Instant::now();

    // Get PTX directory from env or CLI
    let ptx_dir = cli.ptx_dir.clone()
        .or_else(|| std::env::var("PRISM_PTX_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("target/ptx"));

    if !ptx_dir.exists() {
        anyhow::bail!("PTX directory not found: {:?}. Set PRISM_PTX_DIR or use --ptx-dir", ptx_dir);
    }

    // Create CUDA context
    let context = CudaContext::new(0)
        .map_err(|e| anyhow::anyhow!("Failed to create CUDA context: {}", e))?;

    // Use INFERENCE kernel (MegaFusedGpu) - NOT batch training kernel!
    let mut gpu = MegaFusedGpu::new(context, &ptx_dir)?;
    log::info!("GPU initialized with INFERENCE kernel in {:?}", gpu_start.elapsed());

    // 5. Extract 70-dim features using inference kernel for each structure
    let config = MegaFusedConfig::default();
    let mut collector = ReservoirStateCollector::new();

    let n_structures = train_structures.len();
    for (idx, ts) in train_structures.iter().enumerate() {
        let structure_start = Instant::now();

        // Call INFERENCE kernel (detect_pockets) - same as production inference!
        let output = gpu.detect_pockets(
            &ts.atoms,
            &ts.ca_indices,
            &ts.conservation,
            &ts.bfactor,
            &ts.burial,
            None,  // No residue types (legacy compatibility)
            &config,
        ).map_err(|e| anyhow::anyhow!("GPU inference failed for {}: {}", ts.id, e))?;

        let n_residues = ts.ca_indices.len();

        // Validate output size
        let expected_features = n_residues * INFERENCE_FEATURE_DIM;
        if output.combined_features.len() != expected_features {
            log::warn!(
                "Feature size mismatch for {}: got {} expected {} ({} residues * {} dim)",
                ts.id, output.combined_features.len(), expected_features,
                n_residues, INFERENCE_FEATURE_DIM
            );
            continue;
        }

        // Add to collector (features are already 70-dim from inference kernel)
        collector.add_structure(&ts.id, output.combined_features, ts.gt_mask.clone());

        if (idx + 1) % 50 == 0 || idx == 0 {
            let (_, n_res, n_pos) = collector.stats();
            log::info!(
                "Processed {}/{}: {} total residues ({} positive) - {:?}/structure",
                idx + 1, n_structures, n_res, n_pos, structure_start.elapsed()
            );
        }
    }

    let (n_struct, n_res, n_pos) = collector.stats();
    log::info!(
        "Feature extraction complete: {} structures, {} residues ({} positive, {:.2}% rate)",
        n_struct, n_res, n_pos, 100.0 * n_pos as f64 / n_res as f64
    );

    // 6. Apply z-score normalization if enabled
    let training_data = if !cli.no_normalize {
        log::info!("Applying z-score normalization to {} features...", INFERENCE_FEATURE_DIM);
        let (normalized, means, stds) = z_score_normalize(&collector.data, INFERENCE_FEATURE_DIM);

        // Log normalization stats
        let nonzero_stds = stds.iter().filter(|&&s| s > 1e-6).count();
        log::info!(
            "Normalization: {} features with non-zero std, mean range [{:.4}, {:.4}], std range [{:.6}, {:.4}]",
            nonzero_stds,
            means.iter().cloned().fold(f32::INFINITY, f32::min),
            means.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            stds.iter().cloned().fold(f32::INFINITY, f32::min),
            stds.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );

        // Save normalization stats for inference
        let norm_path = cli.output.with_extension("norm.json");
        let norm_json = serde_json::json!({
            "means": means,
            "stds": stds,
            "feature_dim": INFERENCE_FEATURE_DIM,
        });
        fs::write(&norm_path, serde_json::to_string_pretty(&norm_json)?)?;
        log::info!("Saved normalization stats to {:?}", norm_path);

        normalized
    } else {
        collector.data.clone()
    };

    // 7. Train readout weights on 70-dim features
    let train_start = Instant::now();
    log::info!(
        "Training readout on {}-dim features: {} structures, {} residues ({} positive)",
        INFERENCE_FEATURE_DIM, n_struct, n_res, n_pos
    );

    let readout = TrainedReadout::train_with_weight(&training_data, INFERENCE_FEATURE_DIM, cli.lambda, cli.pos_weight)
        .map_err(|e| anyhow::anyhow!("Training failed: {}", e))?;
    log::info!("Training completed in {:?}", train_start.elapsed());

    // 8. Save weights
    readout.save(&cli.output)?;
    log::info!("Saved weights to {:?}", cli.output);

    // Also generate Rust const for compile-time embedding
    let const_path = cli.output.with_extension("rs");
    fs::write(&const_path, readout.to_rust_const())?;
    log::info!("Saved Rust const to {:?}", const_path);

    // 9. Evaluate on test set (if not skipped)
    if !cli.skip_eval && !test_set.is_empty() {
        log::info!("Evaluating on test set...");

        let mut test_structures = Vec::new();
        for pdb_id in &test_set {
            let pdb_path = cli.pdb_dir.join(format!("{}.pdb", pdb_id));
            if !pdb_path.exists() {
                continue;
            }

            let structure = match ProteinStructure::from_pdb_file(&pdb_path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let gt_residues = match ground_truth.get(pdb_id) {
                Some(r) => r,
                None => continue,
            };

            if let Ok(ts) = create_training_structure(&structure, pdb_id, gt_residues) {
                test_structures.push(ts);
            }
        }

        log::info!("Loaded {} test structures", test_structures.len());

        if !test_structures.is_empty() {
            // Extract 70-dim features for test set using SAME inference kernel
            let mut test_collector = ReservoirStateCollector::new();

            for ts in &test_structures {
                let output = gpu.detect_pockets(
                    &ts.atoms,
                    &ts.ca_indices,
                    &ts.conservation,
                    &ts.bfactor,
                    &ts.burial,
                    None,  // No residue types
                    &config,
                ).map_err(|e| anyhow::anyhow!("Test GPU inference failed: {}", e))?;

                test_collector.add_structure(&ts.id, output.combined_features, ts.gt_mask.clone());
            }

            // Apply same normalization to test data
            let test_data = if !cli.no_normalize {
                let norm_path = cli.output.with_extension("norm.json");
                let norm_json: serde_json::Value = serde_json::from_str(&fs::read_to_string(&norm_path)?)?;
                let means: Vec<f32> = serde_json::from_value(norm_json["means"].clone())?;
                let stds: Vec<f32> = serde_json::from_value(norm_json["stds"].clone())?;
                apply_normalization(&test_collector.data, &means, &stds, INFERENCE_FEATURE_DIM)
            } else {
                test_collector.data.clone()
            };

            // Evaluate
            let (acc, prec, rec, f1, auc) = readout.evaluate(&test_data);

            log::info!("═══════════════════════════════════════════════════════════════════");
            log::info!("  TEST SET RESULTS (70-DIM INFERENCE KERNEL)");
            log::info!("  Accuracy:  {:.4}", acc);
            log::info!("  Precision: {:.4}", prec);
            log::info!("  Recall:    {:.4}", rec);
            log::info!("  F1 Score:  {:.4}", f1);
            log::info!("  AUC-ROC:   {:.4}", auc);
            log::info!("═══════════════════════════════════════════════════════════════════");

            // SESSION 10B: Export features if requested
            if let Some(export_dir) = &cli.export_features {
                // Convert collector data to flat arrays for NPY export
                let (train_feats, train_labs) = collector_to_flat_arrays(&collector);
                let (test_feats, test_labs) = collector_to_flat_arrays(&test_collector);

                export_features_to_npy(
                    &train_feats,
                    &train_labs,
                    &test_feats,
                    &test_labs,
                    export_dir,
                )?;

                log::info!("✅ Feature export complete. Ready for Python ensemble training.");
                return Ok(());  // Exit without normal training
            }
        }
    }

    let total_time = total_start.elapsed();
    log::info!("Total training time: {:?}", total_time);

    Ok(())
}

/// Z-score normalize features: (x - mean) / std
#[cfg(feature = "cuda")]
fn z_score_normalize(
    data: &[(Vec<f32>, Vec<u8>)],
    feature_dim: usize,
) -> (Vec<(Vec<f32>, Vec<u8>)>, Vec<f32>, Vec<f32>) {
    // Compute means
    let mut sums = vec![0.0f64; feature_dim];
    let mut counts = vec![0usize; feature_dim];

    for (features, _) in data {
        let n_residues = features.len() / feature_dim;
        for r in 0..n_residues {
            for f in 0..feature_dim {
                let val = features[r * feature_dim + f] as f64;
                if val.is_finite() {
                    sums[f] += val;
                    counts[f] += 1;
                }
            }
        }
    }

    let means: Vec<f32> = sums.iter().zip(&counts)
        .map(|(&sum, &count)| if count > 0 { (sum / count as f64) as f32 } else { 0.0 })
        .collect();

    // Compute stds
    let mut sq_diffs = vec![0.0f64; feature_dim];

    for (features, _) in data {
        let n_residues = features.len() / feature_dim;
        for r in 0..n_residues {
            for f in 0..feature_dim {
                let val = features[r * feature_dim + f] as f64;
                if val.is_finite() {
                    let diff = val - means[f] as f64;
                    sq_diffs[f] += diff * diff;
                }
            }
        }
    }

    let stds: Vec<f32> = sq_diffs.iter().zip(&counts)
        .map(|(&sq_diff, &count)| {
            if count > 1 {
                ((sq_diff / (count - 1) as f64).sqrt()).max(1e-6) as f32
            } else {
                1.0  // Avoid divide by zero
            }
        })
        .collect();

    // Normalize
    let normalized: Vec<(Vec<f32>, Vec<u8>)> = data.iter().map(|(features, labels)| {
        let n_residues = features.len() / feature_dim;
        let mut norm_features = vec![0.0f32; features.len()];

        for r in 0..n_residues {
            for f in 0..feature_dim {
                let idx = r * feature_dim + f;
                let val = features[idx];
                if val.is_finite() {
                    norm_features[idx] = (val - means[f]) / stds[f];
                } else {
                    norm_features[idx] = 0.0;  // Replace NaN/Inf with 0
                }
            }
        }

        (norm_features, labels.clone())
    }).collect();

    (normalized, means, stds)
}

/// Apply pre-computed normalization to data
#[cfg(feature = "cuda")]
fn apply_normalization(
    data: &[(Vec<f32>, Vec<u8>)],
    means: &[f32],
    stds: &[f32],
    feature_dim: usize,
) -> Vec<(Vec<f32>, Vec<u8>)> {
    data.iter().map(|(features, labels)| {
        let n_residues = features.len() / feature_dim;
        let mut norm_features = vec![0.0f32; features.len()];

        for r in 0..n_residues {
            for f in 0..feature_dim {
                let idx = r * feature_dim + f;
                let val = features[idx];
                if val.is_finite() && f < means.len() && f < stds.len() {
                    norm_features[idx] = (val - means[f]) / stds[f];
                } else {
                    norm_features[idx] = 0.0;
                }
            }
        }

        (norm_features, labels.clone())
    }).collect()
}

/// Parse ground truth from CryptoBench dataset.json
/// Format: { "pdb_id": [{ "apo_pocket_selection": ["B_12", "B_14", ...] }] }
fn parse_ground_truth(dataset: &serde_json::Value) -> anyhow::Result<HashMap<String, Vec<(char, i32)>>> {
    let mut ground_truth: HashMap<String, Vec<(char, i32)>> = HashMap::new();

    if let serde_json::Value::Object(map) = dataset {
        for (pdb_id, entries) in map {
            let pdb_id_lower = pdb_id.to_lowercase();
            let mut residues: Vec<(char, i32)> = Vec::new();

            if let serde_json::Value::Array(arr) = entries {
                for entry in arr {
                    if let Some(pocket_sel) = entry.get("apo_pocket_selection") {
                        if let serde_json::Value::Array(selections) = pocket_sel {
                            for sel in selections {
                                if let serde_json::Value::String(s) = sel {
                                    // Parse "B_12" -> ('B', 12)
                                    let parts: Vec<&str> = s.split('_').collect();
                                    if parts.len() == 2 {
                                        let chain = parts[0].chars().next().unwrap_or('A');
                                        if let Ok(res_num) = parts[1].parse::<i32>() {
                                            residues.push((chain, res_num));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Deduplicate
            residues.sort();
            residues.dedup();

            if !residues.is_empty() {
                ground_truth.insert(pdb_id_lower, residues);
            }
        }
    }

    Ok(ground_truth)
}

/// Hydrophobicity scale (Kyte-Doolittle) normalized to [0, 1]
fn residue_hydrophobicity(res_name: &str) -> f32 {
    match res_name.trim() {
        "ILE" => 1.00, "VAL" => 0.97, "LEU" => 0.93, "PHE" => 0.80, "CYS" => 0.72,
        "MET" => 0.57, "ALA" => 0.55, "GLY" => 0.21, "THR" => 0.19, "SER" => 0.16,
        "TRP" => 0.15, "TYR" => 0.08, "PRO" => 0.05, "HIS" => 0.03, "GLU" => 0.01,
        "GLN" => 0.01, "ASP" => 0.01, "ASN" => 0.00, "LYS" => 0.00, "ARG" => 0.00,
        _ => 0.5, // Unknown
    }
}

/// Compute burial score based on neighbor count within 10Å sphere
fn compute_burial_scores(atoms: &[prism_lbs::Atom], ca_indices: &[i32]) -> Vec<f32> {
    let cutoff_sq = 10.0_f32 * 10.0;
    let max_neighbors = 25.0_f32; // Normalization factor

    ca_indices.iter().map(|&ca_idx| {
        let ca = &atoms[ca_idx as usize];
        let ca_x = ca.coord[0] as f32;
        let ca_y = ca.coord[1] as f32;
        let ca_z = ca.coord[2] as f32;

        let mut neighbor_count = 0.0_f32;
        for atom in atoms {
            // Only count heavy atoms (not hydrogen)
            if atom.element == "H" {
                continue;
            }
            let dx = atom.coord[0] as f32 - ca_x;
            let dy = atom.coord[1] as f32 - ca_y;
            let dz = atom.coord[2] as f32 - ca_z;
            let dist_sq = dx*dx + dy*dy + dz*dz;
            if dist_sq < cutoff_sq && dist_sq > 1.0 {
                neighbor_count += 1.0;
            }
        }
        // More neighbors = more buried = higher score
        (neighbor_count / max_neighbors).min(1.0)
    }).collect()
}

/// Create TrainingStructure from protein structure and ground truth
#[cfg(feature = "cuda")]
fn create_training_structure(
    structure: &ProteinStructure,
    pdb_id: &str,
    gt_residues: &[(char, i32)],
) -> anyhow::Result<TrainingStructure> {
    // Extract atoms as flat array
    let mut atoms: Vec<f32> = Vec::with_capacity(structure.atoms.len() * 3);
    for atom in &structure.atoms {
        atoms.push(atom.coord[0] as f32);
        atoms.push(atom.coord[1] as f32);
        atoms.push(atom.coord[2] as f32);
    }

    // Find CA atoms and build residue info
    let mut ca_indices: Vec<i32> = Vec::new();
    let mut residue_info: Vec<(char, i32)> = Vec::new(); // (chain, res_num) for matching GT

    // Build per-residue data
    let mut current_res_id = None;
    for (idx, atom) in structure.atoms.iter().enumerate() {
        let res_id = (atom.chain_id, atom.residue_seq);

        if current_res_id != Some(res_id) {
            current_res_id = Some(res_id);
            residue_info.push((atom.chain_id, atom.residue_seq));

            // Default to this atom as CA, will be updated if we find actual CA
            ca_indices.push(idx as i32);
        }

        // Update CA index if this is the CA atom
        if atom.name.trim() == "CA" {
            if let Some(last_idx) = ca_indices.last_mut() {
                *last_idx = idx as i32;
            }
        }
    }

    let n_residues = ca_indices.len();

    // Create ground truth mask
    let gt_set: HashSet<(char, i32)> = gt_residues.iter().cloned().collect();
    let gt_mask: Vec<u8> = residue_info.iter()
        .map(|&(chain, res_num)| {
            if gt_set.contains(&(chain, res_num)) {
                1u8
            } else {
                0u8
            }
        })
        .collect();

    // Use hydrophobicity as conservation proxy (binding sites often have hydrophobic patches)
    let conservation: Vec<f32> = ca_indices.iter()
        .map(|&idx| {
            let atom = &structure.atoms[idx as usize];
            residue_hydrophobicity(&atom.residue_name)
        })
        .collect();

    // Extract B-factors normalized to [0, 1]
    let mut bfactor: Vec<f32> = Vec::with_capacity(n_residues);
    for &ca_idx in &ca_indices {
        let atom = &structure.atoms[ca_idx as usize];
        // Normalize B-factor (typical range 10-80)
        let b_norm = ((atom.b_factor - 10.0) / 70.0).clamp(0.0, 1.0) as f32;
        bfactor.push(b_norm);
    }

    // Compute actual burial scores from neighbor counts
    let burial = compute_burial_scores(&structure.atoms, &ca_indices);

    Ok(TrainingStructure {
        id: pdb_id.to_string(),
        atoms,
        ca_indices,
        conservation,
        bfactor,
        burial,
        gt_mask,
    })
}
