//! PRISM-TWIN GPU kernel bindings
//!
//! Loads ring_buffer.ptx and tensor_ccf.ptx at runtime.
//! Provides safe wrappers for ring buffer threshold coupling
//! and Tensor Core cross-correlation computation.

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaStream, CudaModule, CudaSlice, CudaFunction, LaunchConfig, PushKernelArg};
use std::sync::Arc;
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// PTX file discovery
// ─────────────────────────────────────────────────────────────────────────────

/// Find a TWIN PTX file by name, searching standard paths.
pub fn find_twin_ptx(name: &str) -> Result<String> {
    let candidates = [
        format!("target/ptx/{}", name),
        format!("../../target/ptx/{}", name),
        format!("crates/prism-gpu/src/kernels/{}", name.replace(".ptx", ".ptx")),
    ];
    // Also check OUT_DIR from build.rs
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let path = format!("{}/ptx/{}", out_dir, name);
        if Path::new(&path).exists() {
            return Ok(path);
        }
    }
    for c in &candidates {
        if Path::new(c).exists() {
            return Ok(c.clone());
        }
    }
    // Search in build output directories
    let build_glob = format!("target/debug/build/prism-gpu-*/out/ptx/{}", name);
    if let Ok(entries) = glob::glob(&build_glob) {
        for entry in entries.flatten() {
            return Ok(entry.to_string_lossy().to_string());
        }
    }
    anyhow::bail!("TWIN PTX '{}' not found in any search path", name)
}

// ─────────────────────────────────────────────────────────────────────────────
// Ring Buffer
// ─────────────────────────────────────────────────────────────────────────────

/// RingSpikeEvent matches the CUDA struct in ring_buffer.cu (48 bytes).
/// This is a SUBSET of GpuSpikeEvent (92 bytes) — only the fields needed
/// for threshold adaptation are included.
#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct RingSpikeEvent {
    pub timestep: i32,
    pub voxel_idx: i32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub vibrational_energy: f32,
    pub water_density: f32,
    pub n_nearby_excited: i32,
    pub spike_source: i32,
    pub wavelength_nm: f32,
    pub pad: i32,
}

/// GPU-side ring buffer for spike exchange between twin observation groups.
pub struct TwinRingBuffer {
    buffer: CudaSlice<u8>,
    head: CudaSlice<u32>,
    tail: CudaSlice<u32>,
    overflow: CudaSlice<u32>,
    capacity: u32,
    /// CPU-side staging for spike compaction (GpuSpikeEvent → RingSpikeEvent)
    staging: Vec<RingSpikeEvent>,
    /// GPU staging buffer for compacted spikes
    d_staging: CudaSlice<u8>,
    // Kernel functions
    push_batch_fn: CudaFunction,
    read_adapt_fn: CudaFunction,
    recovery_fn: CudaFunction,
    reset_fn: CudaFunction,
}

impl TwinRingBuffer {
    pub fn new(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        capacity: u32,
    ) -> Result<Self> {
        let spike_size = std::mem::size_of::<RingSpikeEvent>();
        assert_eq!(spike_size, 48, "RingSpikeEvent must be exactly 48 bytes");

        let buffer = stream.alloc_zeros::<u8>((capacity as usize) * spike_size)?;
        let head = stream.alloc_zeros::<u32>(1)?;
        let tail = stream.alloc_zeros::<u32>(1)?;
        let overflow = stream.alloc_zeros::<u32>(1)?;

        // Staging buffer: max 4096 spikes per push (can grow if needed)
        let staging_capacity = 4096;
        let d_staging = stream.alloc_zeros::<u8>(staging_capacity * spike_size)?;

        let push_batch_fn = module.load_function("ring_buffer_push_batch")
            .context("ring_buffer_push_batch not found in PTX")?;
        let read_adapt_fn = module.load_function("ring_buffer_read_and_adapt")
            .context("ring_buffer_read_and_adapt not found in PTX")?;
        let recovery_fn = module.load_function("ring_buffer_threshold_recovery")
            .context("ring_buffer_threshold_recovery not found in PTX")?;
        let reset_fn = module.load_function("ring_buffer_reset")
            .context("ring_buffer_reset not found in PTX")?;

        Ok(Self {
            buffer, head, tail, overflow, capacity,
            staging: Vec::with_capacity(staging_capacity),
            d_staging,
            push_batch_fn, read_adapt_fn, recovery_fn, reset_fn,
        })
    }

    /// Compact GpuSpikeEvents from the engine into RingSpikeEvents and push to ring buffer.
    ///
    /// This handles the 92B→48B struct conversion on CPU, then uploads the
    /// compacted spikes and calls the ring buffer push kernel.
    pub fn push_compacted(
        &mut self,
        stream: &Arc<CudaStream>,
        gpu_spikes: &[crate::fused_engine::GpuSpikeEvent],
    ) -> Result<()> {
        if gpu_spikes.is_empty() {
            return Ok(());
        }

        // Compact on CPU: extract the 10 fields RingSpikeEvent needs
        self.staging.clear();
        for s in gpu_spikes {
            self.staging.push(RingSpikeEvent {
                timestep: s.timestep,
                voxel_idx: s.voxel_idx,
                x: s.position[0],
                y: s.position[1],
                z: s.position[2],
                intensity: s.intensity,
                vibrational_energy: s.vibrational_energy,
                water_density: s.water_density,
                n_nearby_excited: s.n_nearby_excited,
                spike_source: s.spike_source,
                wavelength_nm: s.wavelength_nm,
                pad: 0,
            });
        }

        let n_spikes = self.staging.len();

        // Reallocate GPU staging if needed
        let needed_bytes = n_spikes * 48;
        if needed_bytes > self.d_staging.len() {
            self.d_staging = stream.alloc_zeros::<u8>(needed_bytes)?;
        }

        // Upload compacted spikes to GPU staging buffer
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.staging.as_ptr() as *const u8,
                n_spikes * 48,
            )
        };
        stream.memcpy_htod(bytes, &mut self.d_staging)?;

        // Launch push_batch kernel on the staging buffer
        let n_blocks = ((n_spikes as u32) + 255) / 256;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let prev_count = 0u32;
        let curr_count = n_spikes as u32;
        unsafe {
            stream.launch_builder(&self.push_batch_fn)
                .arg(&mut self.buffer)
                .arg(&mut self.head)
                .arg(&mut self.tail)
                .arg(&mut self.overflow)
                .arg(&self.capacity)
                .arg(&self.d_staging)
                .arg(&prev_count)
                .arg(&curr_count)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Read spikes from ring buffer and modify target engine's thresholds.
    pub fn read_and_adapt(
        &mut self,
        stream: &Arc<CudaStream>,
        osc_thresholds: &mut CudaSlice<f32>,
        base_thresholds: &CudaSlice<f32>,
        grid_dims: (i32, i32, i32),
        grid_origin: (f32, f32, f32),
        voxel_size: f32,
        sensitivity_boost: f32,
        max_reduction: f32,
        current_step: u32,
        decay_constant: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&self.read_adapt_fn)
                .arg(&self.buffer)
                .arg(&mut self.head)
                .arg(&mut self.tail)
                .arg(&self.capacity)
                .arg(osc_thresholds)
                .arg(base_thresholds)
                .arg(&grid_dims.0).arg(&grid_dims.1).arg(&grid_dims.2)
                .arg(&grid_origin.0).arg(&grid_origin.1).arg(&grid_origin.2)
                .arg(&voxel_size)
                .arg(&sensitivity_boost)
                .arg(&max_reduction)
                .arg(&current_step)
                .arg(&decay_constant)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Gradually restore thresholds toward baseline.
    pub fn threshold_recovery(
        &self,
        stream: &Arc<CudaStream>,
        osc_thresholds: &mut CudaSlice<f32>,
        base_thresholds: &CudaSlice<f32>,
        n_voxels_total: u32,
        recovery_rate: f32,
    ) -> Result<()> {
        let n_blocks = (n_voxels_total + 255) / 256;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&self.recovery_fn)
                .arg(osc_thresholds)
                .arg(base_thresholds)
                .arg(&n_voxels_total)
                .arg(&recovery_rate)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Reset ring buffer state.
    pub fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&self.reset_fn)
                .arg(&mut self.head)
                .arg(&mut self.tail)
                .arg(&mut self.overflow)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Read overflow counter from GPU.
    pub fn overflow_count(&self, stream: &Arc<CudaStream>) -> Result<u32> {
        let mut count = [0u32; 1];
        stream.memcpy_dtoh(&self.overflow, &mut count)?;
        Ok(count[0])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor Core CCF
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-side Tensor Core cross-correlation compute context.
pub struct TwinCcfCompute {
    spike_matrix_a: CudaSlice<u16>,  // FP16 as u16 (half::f16 → u16 bitcast)
    spike_matrix_b: CudaSlice<u16>,
    ccf_output: CudaSlice<f32>,
    norm_a: CudaSlice<f32>,
    norm_b: CudaSlice<f32>,
    n_res: i32,
    n_res_padded: i32,
    n_bins_padded: i32,
    ccf_compute_fn: CudaFunction,
    norms_fn: CudaFunction,
}

impl TwinCcfCompute {
    /// Allocate CCF buffers. Pads dimensions to multiples of 16 for WMMA.
    pub fn new(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        n_res: i32,
        n_bins: i32,
    ) -> Result<Self> {
        let n_res_padded = ((n_res + 15) / 16) * 16;
        let n_bins_padded = ((n_bins + 15) / 16) * 16;

        let mat_size = (n_res_padded * n_bins_padded) as usize;
        let spike_matrix_a = stream.alloc_zeros::<u16>(mat_size)?;
        let spike_matrix_b = stream.alloc_zeros::<u16>(mat_size)?;
        let ccf_output = stream.alloc_zeros::<f32>((n_res * n_res) as usize)?;
        let norm_a = stream.alloc_zeros::<f32>(n_res as usize)?;
        let norm_b = stream.alloc_zeros::<f32>(n_res as usize)?;

        let ccf_compute_fn = module.load_function("tensor_ccf_compute")
            .context("tensor_ccf_compute not found in PTX")?;
        let norms_fn = module.load_function("compute_spike_norms")
            .context("compute_spike_norms not found in PTX")?;

        Ok(Self {
            spike_matrix_a, spike_matrix_b, ccf_output, norm_a, norm_b,
            n_res, n_res_padded, n_bins_padded,
            ccf_compute_fn, norms_fn,
        })
    }

    /// Upload mean-centered spike matrices (CPU → GPU).
    /// Input is f32, converted to FP16 (u16 bitcast) before upload.
    pub fn upload_matrices(
        &mut self,
        stream: &Arc<CudaStream>,
        matrix_a_f32: &[f32],
        matrix_b_f32: &[f32],
    ) -> Result<()> {
        // Convert f32 → f16 → u16 bitcast
        let to_u16 = |vals: &[f32]| -> Vec<u16> {
            vals.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect()
        };
        let a_u16 = to_u16(matrix_a_f32);
        let b_u16 = to_u16(matrix_b_f32);

        stream.memcpy_htod(&a_u16, &mut self.spike_matrix_a)?;
        stream.memcpy_htod(&b_u16, &mut self.spike_matrix_b)?;
        Ok(())
    }

    /// Compute norms, then CCF = A × B^T / (norm_a × norm_b).
    pub fn compute(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        // Step 1: per-row norms
        let norm_blocks = ((self.n_res as u32) + 255) / 256;
        let norm_cfg = LaunchConfig {
            grid_dim: (norm_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&self.norms_fn)
                .arg(&self.spike_matrix_a)
                .arg(&mut self.norm_a)
                .arg(&self.n_res)
                .arg(&self.n_bins_padded)
                .launch(norm_cfg)?;
            stream.launch_builder(&self.norms_fn)
                .arg(&self.spike_matrix_b)
                .arg(&mut self.norm_b)
                .arg(&self.n_res)
                .arg(&self.n_bins_padded)
                .launch(norm_cfg)?;
        }

        // Step 2: WMMA CCF
        // Grid: tiles over n_res_padded, Block: (32, 4) = 4 warps
        let tile_x = ((self.n_res_padded as u32) + 16 * 4 - 1) / (16 * 4);
        let tile_y = ((self.n_res_padded as u32) + 15) / 16;
        let ccf_cfg = LaunchConfig {
            grid_dim: (tile_x.max(1), tile_y.max(1), 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 4 * 16 * 16 * 4, // 4 warps × 16×16 × sizeof(float)
        };
        unsafe {
            stream.launch_builder(&self.ccf_compute_fn)
                .arg(&self.spike_matrix_a)
                .arg(&self.spike_matrix_b)
                .arg(&mut self.ccf_output)
                .arg(&self.norm_a)
                .arg(&self.norm_b)
                .arg(&self.n_res)
                .arg(&self.n_res_padded)
                .arg(&self.n_bins_padded)
                .launch(ccf_cfg)?;
        }
        Ok(())
    }

    /// Download CCF matrix to CPU.
    pub fn download_ccf(&self, stream: &Arc<CudaStream>) -> Result<Vec<f32>> {
        let size = (self.n_res * self.n_res) as usize;
        let mut host = vec![0.0f32; size];
        stream.memcpy_dtoh(&self.ccf_output, &mut host)?;
        Ok(host)
    }

    /// Get matrix dimensions for external use.
    pub fn dimensions(&self) -> (i32, i32, i32) {
        (self.n_res, self.n_res_padded, self.n_bins_padded)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CCF matrix feature extraction (CPU-side)
// ─────────────────────────────────────────────────────────────────────────────

/// Per-residue features extracted from the CCF matrix.
#[derive(Debug, Clone, Default)]
pub struct CcfResidueFeatures {
    pub ccf_peak_lag: i32,
    pub ccf_peak_value: f32,
    pub ccf_width: f32,
    pub ccf_asymmetry: f32,
    pub ccf_reproducibility: f32,
}

/// Extract per-residue CCF features from n_res × n_res CCF matrix.
pub fn extract_ccf_features(ccf_matrix: &[f32], n_res: usize) -> Vec<CcfResidueFeatures> {
    (0..n_res).map(|r| {
        let row = &ccf_matrix[r * n_res..(r + 1) * n_res];

        // Peak value and column (max off-diagonal)
        let (peak_col, peak_val) = row.iter().enumerate()
            .filter(|(j, _)| *j != r)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, &v)| (j as i32, v))
            .unwrap_or((0, 0.0));

        // Width: columns with CCF > peak_val * 0.5
        let half_max = peak_val * 0.5;
        let width = row.iter().filter(|&&v| v > half_max).count() as f32;

        // Asymmetry: upper triangle mean vs lower triangle mean
        let upper: f32 = if r + 1 < n_res { row[r+1..].iter().sum() } else { 0.0 };
        let lower: f32 = row[..r].iter().sum();
        let denom = (upper.abs() + lower.abs()).max(1e-8);
        let asymmetry = (upper - lower) / denom;

        // Reproducibility: fraction of residues with CCF > 0.1
        let reproducibility = row.iter()
            .filter(|&&v| v > 0.1)
            .count() as f32 / n_res.max(1) as f32;

        CcfResidueFeatures {
            ccf_peak_lag: peak_col - r as i32,
            ccf_peak_value: peak_val,
            ccf_width: width,
            ccf_asymmetry: asymmetry,
            ccf_reproducibility: reproducibility,
        }
    }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Spike matrix builder (CPU-side prep for CCF)
// ─────────────────────────────────────────────────────────────────────────────

/// Build time-binned spike matrices for CCF computation.
/// Returns (matrix_a, matrix_b) as f32 vectors, mean-centered, padded.
pub fn build_ccf_matrices(
    spikes_a: &[crate::fused_engine::GpuSpikeEvent],
    spikes_b: &[crate::fused_engine::GpuSpikeEvent],
    n_residues: usize,
    total_steps: i32,
    bin_size: i32,
) -> (Vec<f32>, Vec<f32>, i32, i32) {
    let n_bins = (total_steps / bin_size.max(1)) as usize;
    let n_res_padded = ((n_residues + 15) / 16) * 16;
    let n_bins_padded = ((n_bins + 15) / 16) * 16;

    let mut mat_a = vec![0.0f32; n_res_padded * n_bins_padded];
    let mut mat_b = vec![0.0f32; n_res_padded * n_bins_padded];

    // Accumulate spike intensities into bins
    let accumulate = |mat: &mut Vec<f32>, spikes: &[crate::fused_engine::GpuSpikeEvent]| {
        for spike in spikes {
            let bin = (spike.timestep / bin_size.max(1)) as usize;
            if bin >= n_bins { continue; }
            let n = spike.n_residues.min(8) as usize;
            for r in 0..n {
                let resid = spike.nearby_residues[r];
                if resid < 0 || resid >= n_residues as i32 { continue; }
                mat[resid as usize * n_bins_padded + bin] += spike.intensity;
            }
        }
    };
    accumulate(&mut mat_a, spikes_a);
    accumulate(&mut mat_b, spikes_b);

    // Mean-center each row (critical for WMMA — raw counts saturate FP16)
    for r in 0..n_residues {
        let start = r * n_bins_padded;
        let mean: f32 = mat_a[start..start + n_bins].iter().sum::<f32>() / n_bins.max(1) as f32;
        for v in mat_a[start..start + n_bins].iter_mut() { *v -= mean; }

        let mean: f32 = mat_b[start..start + n_bins].iter().sum::<f32>() / n_bins.max(1) as f32;
        for v in mat_b[start..start + n_bins].iter_mut() { *v -= mean; }
    }

    (mat_a, mat_b, n_res_padded as i32, n_bins_padded as i32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_spike_event_size() {
        assert_eq!(
            std::mem::size_of::<RingSpikeEvent>(), 48,
            "RingSpikeEvent must be exactly 48 bytes to match CUDA struct"
        );
    }

    #[test]
    fn test_ccf_feature_extraction_identity() {
        // 3×3 CCF with known structure
        let ccf = vec![
            1.0, 0.2, 0.1,
            0.2, 1.0, 0.3,
            0.1, 0.3, 1.0,
        ];
        let features = extract_ccf_features(&ccf, 3);
        assert_eq!(features.len(), 3);
        // Residue 2's highest off-diagonal is column 1 (0.3)
        assert_eq!(features[2].ccf_peak_lag, 1 - 2); // relative: -1
        assert!((features[2].ccf_peak_value - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_ccf_feature_extraction_zeros() {
        let ccf = vec![0.0; 9]; // 3×3 all zeros
        let features = extract_ccf_features(&ccf, 3);
        assert_eq!(features.len(), 3);
        for f in &features {
            assert_eq!(f.ccf_peak_value, 0.0);
        }
    }

    #[test]
    fn test_build_ccf_matrices_padding() {
        // 5 residues → padded to 16
        // 100 steps / bin_size=10 = 10 bins → padded to 16
        let (mat_a, _mat_b, n_res_p, n_bins_p) = build_ccf_matrices(
            &[], &[], 5, 100, 10,
        );
        assert_eq!(n_res_p, 16);
        assert_eq!(n_bins_p, 16);
        assert_eq!(mat_a.len(), 16 * 16);
    }

    #[test]
    fn test_build_ccf_matrices_mean_centering() {
        use crate::fused_engine::GpuSpikeEvent;
        // Create spikes at residue 0, various bins
        let spikes: Vec<GpuSpikeEvent> = (0..10).map(|t| {
            let mut s = GpuSpikeEvent::default();
            s.timestep = t * 10;
            s.intensity = 1.0;
            s.nearby_residues[0] = 0;
            s.n_residues = 1;
            s
        }).collect();

        let (mat_a, _, n_res_p, n_bins_p) = build_ccf_matrices(
            &spikes, &[], 3, 100, 10,
        );

        // After mean-centering, row 0 should sum to ~0
        let row_sum: f32 = mat_a[0..n_bins_p as usize].iter().sum();
        assert!(row_sum.abs() < 0.01, "Mean-centered row should sum to ~0, got {}", row_sum);
    }
}
