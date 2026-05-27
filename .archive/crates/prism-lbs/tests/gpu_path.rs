#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;

use prism_lbs::{
    graph::{GraphConfig, ProteinGraphBuilder},
    structure::{ProteinStructure, SurfaceComputer},
};

fn simple_pdb() -> String {
    [
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N",
        "ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 10.00           C",
        "ATOM      3  C   ALA A   1       1.500   1.500   0.000  1.00 10.00           C",
        "ATOM      4  O   ALA A   1       0.000   1.500   0.000  1.00 10.00           O",
        "ATOM      5  CB  ALA A   1       1.500  -0.750  -1.200  1.00 10.00           C",
        "ATOM      6  N   ALA A   2       3.000   0.000   0.000  1.00 10.00           N",
        "END",
    ]
    .join("\n")
}

#[test]
fn gpu_surface_and_graph_path_smoke() -> anyhow::Result<()> {
    // Skip gracefully if no GPU or PTX is available.
    let ptx_dir = std::env::var("PRISM_PTX_DIR").unwrap_or_else(|_| "target/ptx".to_string());
    let device_id = std::env::var("PRISM_GPU_DEVICE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let gpu_ctx = match prism_gpu::context::GpuContext::new(
        device_id,
        prism_gpu::context::GpuSecurityConfig::default(),
        Path::new(&ptx_dir),
    ) {
        Ok(ctx) => Arc::new(ctx),
        Err(e) => {
            eprintln!("Skipping CUDA path test (no GPU/PTX): {}", e);
            return Ok(());
        }
    };

    let mut structure = ProteinStructure::from_pdb_str(&simple_pdb())?;

    // Surface computation on GPU
    SurfaceComputer::default().compute_gpu(&mut structure, &gpu_ctx)?;
    assert!(
        structure.atoms.iter().any(|a| a.is_surface),
        "GPU surface computation should mark surface atoms"
    );

    // Graph construction on GPU
    let mut graph_cfg = GraphConfig::default();
    graph_cfg.surface_only = false;
    graph_cfg.use_gpu = true;
    let graph = ProteinGraphBuilder::new(graph_cfg).build_with_gpu(&structure, Some(&gpu_ctx))?;
    assert_eq!(graph.atom_indices.len(), structure.atoms.len());
    Ok(())
}
