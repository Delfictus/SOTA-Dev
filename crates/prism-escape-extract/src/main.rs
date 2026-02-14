//! Real PRISM-GPU feature extraction for viral escape benchmark
//! Uses MegaFusedBatchGpu to extract actual features

use anyhow::{Result, Context};
use prism_gpu::{MegaFusedBatchGpu, StructureInput, PackedBatch};
use cudarc::driver::CudaContext;
use std::sync::Arc;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════");
    println!("  PRISM-GPU Feature Extraction (REAL)");
    println!("═══════════════════════════════════════════════════\n");

    // Load PDB structure
    let pdb_path = "prism-escape-benchmark/data/raw/structures/6m0j.pdb";
    println!("Loading structure: {}", pdb_path);

    let pdb_content = fs::read_to_string(pdb_path)
        .context("Failed to read PDB file")?;

    // Parse PDB to get coordinates and residue info
    let (atoms, ca_indices, conservation, bfactor) = parse_pdb(&pdb_content)?;
    let num_residues = ca_indices.len();

    println!("  ✓ Loaded {} residues, {} atoms", num_residues, atoms.len() / 3);

    // Initialize GPU
    println!("\nInitializing PRISM-GPU...");
    let context = Arc::new(CudaContext::new(0)?);
    let gpu = MegaFusedBatchGpu::new(context, Path::new("target/ptx"))?;

    println!("  ✓ GPU initialized");

    // Create structure input
    let structure = StructureInput {
        id: "6m0j".to_string(),
        atoms,
        ca_indices,
        conservation,
        bfactor,
    };

    // Create batch
    let batch = PackedBatch::from_structures(&[structure])?;

    println!("\nExtracting features via mega_fused_batch_detection kernel...");
    let output = gpu.process_batch_detection(&batch)?;

    let features = &output.features[0];
    println!("  ✓ Extracted {} × 136 features", features.len());

    // Save as numpy array
    let output_dir = "prism-escape-benchmark/extracted_features";
    fs::create_dir_all(output_dir)?;

    let output_path = format!("{}/6m0j_RESIDUE_TYPES_FIXED.npy", output_dir);
    save_numpy(&features, &output_path)?;

    println!("\n✅ REAL features saved to: {}", output_path);
    println!("═══════════════════════════════════════════════════");

    Ok(())
}

fn parse_pdb(content: &str) -> Result<(Vec<f32>, Vec<i32>, Vec<f32>, Vec<f32>)> {
    let mut atoms = Vec::new(); // flat [x0, y0, z0, x1, y1, z1, ...]
    let mut ca_indices = Vec::new();
    let mut bfactors = Vec::new();
    let mut atom_count = 0;

    for line in content.lines() {
        if !line.starts_with("ATOM") {
            continue;
        }

        let x: f32 = line[30..38].trim().parse()?;
        let y: f32 = line[38..46].trim().parse()?;
        let z: f32 = line[46..54].trim().parse()?;
        let bfactor: f32 = line[60..66].trim().parse().unwrap_or(0.0);

        atoms.extend_from_slice(&[x, y, z]);

        // Track CA atoms
        let atom_name = line[12..16].trim();
        if atom_name == "CA" {
            ca_indices.push(atom_count);
            bfactors.push(bfactor);
        }

        atom_count += 1;
    }

    // Normalize bfactors
    let max_b = bfactors.iter().fold(0.0f32, |a, &b| a.max(b));
    if max_b > 0.0 {
        for b in &mut bfactors {
            *b /= max_b;
        }
    }

    // Default conservation (uniform)
    let conservation = vec![0.5f32; ca_indices.len()];

    Ok((atoms, ca_indices, conservation, bfactors))
}

fn save_numpy(features: &[f32], num_features: usize, path: &str) -> Result<()> {
    // Simple NPY format (magic + header + data)
    use std::io::Write;

    let mut file = fs::File::create(path)?;

    let num_residues = features.len() / num_features;

    // NPY magic number
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[0x01, 0x00])?; // version 1.0

    // Header
    let shape = format!("({}, {})", num_residues, num_features);
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}", shape);
    let header_len = header.len() as u16;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header.as_bytes())?;

    // Data (already flat row-major)
    for &val in features {
        file.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}
