//! GPU helpers for PRISM-LBS geometry and clustering.

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use prism_core::PrismError;
use std::path::Path;
use std::sync::Arc;

/// GPU executor for LBS kernels (surface accessibility, distance matrix, clustering, scoring).
pub struct LbsGpu {
    device: Arc<CudaDevice>,
}

impl LbsGpu {
    /// Load LBS PTX modules from `ptx_dir`. Expects:
    /// - lbs_surface_accessibility.ptx with `surface_accessibility_kernel`
    /// - lbs_distance_matrix.ptx with `distance_matrix_kernel`
    /// - lbs_pocket_clustering.ptx with `pocket_clustering_kernel`
    /// - lbs_druggability_scoring.ptx with `druggability_score_kernel`
    pub fn new(device: Arc<CudaDevice>, ptx_dir: &Path) -> Result<Self, PrismError> {
        let path = ptx_dir.join("lbs_surface_accessibility.ptx");
        let ptx = cudarc::nvrtc::Ptx::from_file(&path);
        device
            .load_ptx(
                ptx,
                "lbs_surface_accessibility",
                &["surface_accessibility_kernel"],
            )
            .map_err(|e| {
                PrismError::gpu(
                    "lbs_surface_accessibility",
                    format!("Failed to load PTX: {}", e),
                )
            })?;

        let path = ptx_dir.join("lbs_distance_matrix.ptx");
        let ptx = cudarc::nvrtc::Ptx::from_file(&path);
        device
            .load_ptx(ptx, "lbs_distance_matrix", &["distance_matrix_kernel"])
            .map_err(|e| {
                PrismError::gpu("lbs_distance_matrix", format!("Failed to load PTX: {}", e))
            })?;

        let path = ptx_dir.join("lbs_pocket_clustering.ptx");
        let ptx = cudarc::nvrtc::Ptx::from_file(&path);
        device
            .load_ptx(ptx, "lbs_pocket_clustering", &["pocket_clustering_kernel"])
            .map_err(|e| {
                PrismError::gpu(
                    "lbs_pocket_clustering",
                    format!("Failed to load PTX: {}", e),
                )
            })?;

        let path = ptx_dir.join("lbs_druggability_scoring.ptx");
        let ptx = cudarc::nvrtc::Ptx::from_file(&path);
        device
            .load_ptx(
                ptx,
                "lbs_druggability_scoring",
                &["druggability_score_kernel"],
            )
            .map_err(|e| {
                PrismError::gpu(
                    "lbs_druggability_scoring",
                    format!("Failed to load PTX: {}", e),
                )
            })?;

        Ok(Self { device })
    }

    /// Compute SASA and surface flags for atoms.
    pub fn surface_accessibility(
        &self,
        coords: &[[f32; 3]],
        radii: &[f32],
        samples: i32,
        probe_radius: f32,
    ) -> Result<(Vec<f32>, Vec<u8>), PrismError> {
        let n = coords.len();
        if radii.len() != n {
            return Err(PrismError::gpu(
                "lbs_surface_accessibility",
                "radii length mismatch",
            ));
        }
        let x: Vec<f32> = coords.iter().map(|c| c[0]).collect();
        let y: Vec<f32> = coords.iter().map(|c| c[1]).collect();
        let z: Vec<f32> = coords.iter().map(|c| c[2]).collect();

        let d_x = self
            .device
            .htod_copy(x)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let d_y = self
            .device
            .htod_copy(y)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let d_z = self
            .device
            .htod_copy(z)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let d_r = self
            .device
            .htod_copy(radii.to_vec())
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let mut d_sasa = self
            .device
            .alloc_zeros::<f32>(n)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let mut d_surface = self
            .device
            .alloc_zeros::<u8>(n)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let kernel: CudaFunction = self
            .device
            .get_func("lbs_surface_accessibility", "surface_accessibility_kernel")
            .ok_or_else(|| PrismError::gpu("lbs_surface_accessibility", "kernel not found"))?;
        unsafe {
            kernel
                .launch(
                    cfg,
                    (
                        &d_x,
                        &d_y,
                        &d_z,
                        &d_r,
                        n as i32,
                        samples,
                        probe_radius,
                        &mut d_sasa,
                        &mut d_surface,
                    ),
                )
                .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        }

        let sasa = self
            .device
            .dtoh_sync_copy(&d_sasa)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        let surface = self
            .device
            .dtoh_sync_copy(&d_surface)
            .map_err(|e| PrismError::gpu("lbs_surface_accessibility", e.to_string()))?;
        Ok((sasa, surface))
    }

    /// Compute full pairwise distance matrix (n x n).
    pub fn distance_matrix(&self, coords: &[[f32; 3]]) -> Result<Vec<f32>, PrismError> {
        let n = coords.len();
        let x: Vec<f32> = coords.iter().map(|c| c[0]).collect();
        let y: Vec<f32> = coords.iter().map(|c| c[1]).collect();
        let z: Vec<f32> = coords.iter().map(|c| c[2]).collect();
        let d_x = self
            .device
            .htod_copy(x)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        let d_y = self
            .device
            .htod_copy(y)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        let d_z = self
            .device
            .htod_copy(z)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        let mut d_out = self
            .device
            .alloc_zeros::<f32>(n * n)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;

        let block = (16, 16, 1);
        let grid = (
            (n as u32 + block.0 - 1) / block.0,
            (n as u32 + block.1 - 1) / block.1,
            1,
        );
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };
        let kernel: CudaFunction = self
            .device
            .get_func("lbs_distance_matrix", "distance_matrix_kernel")
            .ok_or_else(|| PrismError::gpu("lbs_distance_matrix", "kernel not found"))?;
        unsafe {
            kernel
                .launch(cfg, (&d_x, &d_y, &d_z, n as i32, &mut d_out))
                .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?;
        }
        Ok(self
            .device
            .dtoh_sync_copy(&d_out)
            .map_err(|e| PrismError::gpu("lbs_distance_matrix", e.to_string()))?)
    }

    /// Greedy pocket clustering (graph coloring) on GPU.
    pub fn pocket_clustering(
        &self,
        row_ptr: &[i32],
        col_idx: &[i32],
        max_colors: i32,
    ) -> Result<Vec<i32>, PrismError> {
        let n = row_ptr.len().saturating_sub(1);
        let d_row = self
            .device
            .htod_copy(row_ptr.to_vec())
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;
        let d_col = self
            .device
            .htod_copy(col_idx.to_vec())
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;
        let mut d_colors = self
            .device
            .alloc_zeros::<i32>(n)
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let kernel: CudaFunction = self
            .device
            .get_func("lbs_pocket_clustering", "pocket_clustering_kernel")
            .ok_or_else(|| PrismError::gpu("lbs_pocket_clustering", "kernel not found"))?;
        unsafe {
            kernel
                .launch(cfg, (&d_row, &d_col, n as i32, max_colors, &mut d_colors))
                .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?;
        }
        Ok(self
            .device
            .dtoh_sync_copy(&d_colors)
            .map_err(|e| PrismError::gpu("lbs_pocket_clustering", e.to_string()))?)
    }

    /// GPU aggregation of druggability scores.
    pub fn druggability_score(
        &self,
        volume: &[f32],
        hydrophobicity: &[f32],
        enclosure: &[f32],
        depth: &[f32],
        hbond: &[f32],
        flexibility: &[f32],
        conservation: &[f32],
        topology: &[f32],
        weights: [f32; 8],
    ) -> Result<Vec<f32>, PrismError> {
        let n = volume.len();
        let inputs = [
            hydrophobicity.len(),
            enclosure.len(),
            depth.len(),
            hbond.len(),
            flexibility.len(),
            conservation.len(),
            topology.len(),
        ];
        if inputs.iter().any(|&l| l != n) {
            return Err(PrismError::gpu(
                "lbs_druggability_scoring",
                "input length mismatch",
            ));
        }

        let d_volume = self
            .device
            .htod_copy(volume.to_vec())
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_hydro = self
            .device
            .htod_copy(hydrophobicity.to_vec())
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_enclosure = self
            .device
            .htod_copy(enclosure.to_vec())
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_depth = self
            .device
            .htod_copy(depth.to_vec())
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_hbond = self
            .device
            .htod_copy(hbond.to_vec())
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_flex = self
            .device
            .htod_copy(flexibility.to_vec())
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_cons = self
            .device
            .htod_copy(conservation.to_vec())
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_topo = self
            .device
            .htod_copy(topology.to_vec())
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let d_weights = self
            .device
            .htod_copy(weights.to_vec())
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        let mut d_out = self
            .device
            .alloc_zeros::<f32>(n)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;

        let cfg = LaunchConfig::for_num_elems(n as u32);
        let kernel: CudaFunction = self
            .device
            .get_func("lbs_druggability_scoring", "druggability_score_kernel")
            .ok_or_else(|| PrismError::gpu("lbs_druggability_scoring", "kernel not found"))?;
        unsafe {
            kernel
                .launch(
                    cfg,
                    (
                        &d_volume,
                        &d_hydro,
                        &d_enclosure,
                        &d_depth,
                        &d_hbond,
                        &d_flex,
                        &d_cons,
                        &d_topo,
                        &d_weights,
                        n as i32,
                        &mut d_out,
                    ),
                )
                .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?;
        }
        Ok(self
            .device
            .dtoh_sync_copy(&d_out)
            .map_err(|e| PrismError::gpu("lbs_druggability_scoring", e.to_string()))?)
    }
}
