//! Graph representation of protein surface atoms

use crate::structure::{hydrophobicity_scale, ProteinStructure};
use crate::LbsError;
#[cfg(feature = "cuda")]
use log::warn;
#[cfg(feature = "cuda")]
use prism_gpu::{context::GpuContext, global_context::GlobalGpuContext, LbsGpu};
use serde::{Deserialize, Serialize};

/// Graph construction parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    pub distance_threshold: f64,
    pub surface_only: bool,
    pub min_sasa: f64,
    pub weighted_edges: bool,
    /// Enable GPU distance matrix + clustering when GPU is available
    pub use_gpu: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            distance_threshold: 4.5,
            surface_only: true,
            min_sasa: 0.5,
            weighted_edges: true,
            use_gpu: true,  // GPU-first: always prefer GPU acceleration
        }
    }
}

/// Per-vertex feature arrays
#[derive(Debug, Clone)]
pub struct VertexFeatures {
    pub hydrophobicity: Vec<f64>,
    pub electrostatic: Vec<f64>,
    pub curvature: Vec<f64>,
    pub depth: Vec<f64>,
    pub conservation: Vec<f64>,
    pub flexibility: Vec<f64>,
}

impl VertexFeatures {
    pub fn new(n: usize) -> Self {
        Self {
            hydrophobicity: vec![0.0; n],
            electrostatic: vec![0.0; n],
            curvature: vec![0.0; n],
            depth: vec![0.0; n],
            conservation: vec![0.0; n],
            flexibility: vec![0.0; n],
        }
    }
}

/// Graph built from atoms
#[derive(Debug, Clone)]
pub struct ProteinGraph {
    pub atom_indices: Vec<usize>,
    pub adjacency: Vec<Vec<usize>>,
    pub edge_weights: Vec<Vec<f64>>,
    pub vertex_features: VertexFeatures,
    pub structure_ref: ProteinStructure,
}

/// Build protein graphs from structures
#[derive(Debug, Clone)]
pub struct ProteinGraphBuilder {
    config: GraphConfig,
}

impl ProteinGraphBuilder {
    pub fn new(config: GraphConfig) -> Self {
        Self { config }
    }

    pub fn build(&self, structure: &ProteinStructure) -> Result<ProteinGraph, LbsError> {
        let selected = self.select_atoms(structure);
        let n = selected.len();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut edge_weights: Vec<Vec<f64>> = vec![Vec::new(); n];
        self.fill_cpu_edges(structure, &selected, &mut adjacency, &mut edge_weights);

        let vertex_features = self.compute_features(structure, &selected)?;

        Ok(ProteinGraph {
            atom_indices: selected,
            adjacency,
            edge_weights,
            vertex_features,
            structure_ref: structure.clone(),
        })
    }

    /// Build protein graph using a provided GPU context when enabled; falls back to CPU.
    #[cfg(feature = "cuda")]
    pub fn build_with_gpu(
        &self,
        structure: &ProteinStructure,
        gpu_ctx: Option<&GpuContext>,
    ) -> Result<ProteinGraph, LbsError> {
        let selected = self.select_atoms(structure);
        let n = selected.len();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut edge_weights: Vec<Vec<f64>> = vec![Vec::new(); n];

        if self.config.use_gpu {
            if let Some(ctx) = gpu_ctx {
                let coords: Vec<[f32; 3]> = selected
                    .iter()
                    .filter_map(|&i| structure.atoms.get(i))
                    .map(|a| [a.coord[0] as f32, a.coord[1] as f32, a.coord[2] as f32])
                    .collect();
                // Try pre-loaded LbsGpu from GlobalGpuContext first (zero PTX overhead)
                let gpu_result = if let Some(gpu) = GlobalGpuContext::try_get().ok().and_then(|g| g.lbs_locked()) {
                    log::debug!("Using pre-loaded LbsGpu for distance matrix (zero PTX overhead)");
                    gpu.distance_matrix(&coords)
                } else {
                    log::debug!("GlobalGpuContext LbsGpu not available, creating new instance");
                    match LbsGpu::new(ctx.device().clone(), &ctx.ptx_dir()) {
                        Ok(gpu) => gpu.distance_matrix(&coords),
                        Err(e) => Err(e)
                    }
                };
                match gpu_result {
                    Ok(dist_mat) => {
                        for i in 0..n {
                            for j in (i + 1)..n {
                                let d = dist_mat[i * n + j] as f64;
                                if d <= self.config.distance_threshold {
                                    let w = if self.config.weighted_edges {
                                        1.0 - (d / self.config.distance_threshold)
                                    } else {
                                        1.0
                                    };
                                    adjacency[i].push(j);
                                    adjacency[j].push(i);
                                    edge_weights[i].push(w);
                                    edge_weights[j].push(w);
                                }
                            }
                        }
                    }
                    Err(e) => warn!("GPU distance matrix failed; falling back to CPU: {}", e),
                }
            } else {
                warn!("GPU graph requested but no GPU context provided; falling back to CPU");
            }
        }

        if adjacency.iter().all(|nbrs| nbrs.is_empty()) {
            self.fill_cpu_edges(structure, &selected, &mut adjacency, &mut edge_weights);
        }

        let vertex_features = self.compute_features(structure, &selected)?;

        Ok(ProteinGraph {
            atom_indices: selected,
            adjacency,
            edge_weights,
            vertex_features,
            structure_ref: structure.clone(),
        })
    }

    fn compute_features(
        &self,
        structure: &ProteinStructure,
        selected: &[usize],
    ) -> Result<VertexFeatures, LbsError> {
        let n = selected.len();
        let mut f = VertexFeatures::new(n);

        for (idx, &atom_idx) in selected.iter().enumerate() {
            let atom = &structure.atoms[atom_idx];
            f.hydrophobicity[idx] = hydrophobicity_scale(&atom.residue_name);
            f.electrostatic[idx] = atom.partial_charge;
            f.curvature[idx] = atom.curvature;
            f.depth[idx] = atom.depth;

            if let Some(res_idx) = structure
                .residues
                .iter()
                .position(|r| r.seq_number == atom.residue_seq && r.chain_id == atom.chain_id)
            {
                let res = &structure.residues[res_idx];
                f.conservation[idx] = res.conservation_score;
                f.flexibility[idx] = res.flexibility;
            }
        }

        Ok(f)
    }

    fn select_atoms(&self, structure: &ProteinStructure) -> Vec<usize> {
        structure
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, atom)| {
                let passes_surface = !self.config.surface_only || atom.is_surface;
                passes_surface && atom.sasa >= self.config.min_sasa
            })
            .map(|(i, _)| i)
            .collect()
    }

    fn fill_cpu_edges(
        &self,
        structure: &ProteinStructure,
        selected: &[usize],
        adjacency: &mut [Vec<usize>],
        edge_weights: &mut [Vec<f64>],
    ) {
        let n = selected.len();
        let thresh_sq = self.config.distance_threshold.powi(2);

        for i in 0..n {
            let ai = &structure.atoms[selected[i]];
            for j in (i + 1)..n {
                let aj = &structure.atoms[selected[j]];
                let dx = ai.coord[0] - aj.coord[0];
                let dy = ai.coord[1] - aj.coord[1];
                let dz = ai.coord[2] - aj.coord[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 <= thresh_sq {
                    let w = if self.config.weighted_edges {
                        let d = d2.sqrt();
                        1.0 - (d / self.config.distance_threshold)
                    } else {
                        1.0
                    };
                    adjacency[i].push(j);
                    adjacency[j].push(i);
                    edge_weights[i].push(w);
                    edge_weights[j].push(w);
                }
            }
        }
    }
}
