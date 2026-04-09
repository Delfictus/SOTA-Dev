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
// Stream Completion Signaling
// ─────────────────────────────────────────────────────────────────────────────

/// Device-side stream completion flags for persistent coupling kernel.
///
/// Each physics stream appends `signal_stream_done(flag, step)` after its
/// LADD kernels. The persistent coupling kernel spin-waits on both flags.
/// Step-numbered flags prevent ABA race conditions.
pub struct TwinSignal {
    pub flag_a: CudaSlice<u32>,    // [1] — stream A completion flag
    pub flag_b: CudaSlice<u32>,    // [1] — stream B completion flag
    signal_fn: CudaFunction,
    clear_fn: CudaFunction,
}

impl TwinSignal {
    pub fn new(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
    ) -> Result<Self> {
        let flag_a = stream.alloc_zeros::<u32>(1)?;
        let flag_b = stream.alloc_zeros::<u32>(1)?;
        let signal_fn = module.load_function("signal_stream_done")
            .context("signal_stream_done not found in PTX")?;
        let clear_fn = module.load_function("clear_signals")
            .context("clear_signals not found in PTX")?;
        Ok(Self { flag_a, flag_b, signal_fn, clear_fn })
    }

    /// Signal that stream A has completed step `step_number`.
    /// Must be launched on stream A's CUDA stream.
    pub fn signal_a(&mut self, stream: &Arc<CudaStream>, step_number: u32) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 };
        unsafe {
            stream.launch_builder(&self.signal_fn)
                .arg(&mut self.flag_a)
                .arg(&step_number)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Signal that stream B has completed step `step_number`.
    /// Must be launched on stream B's CUDA stream.
    pub fn signal_b(&mut self, stream: &Arc<CudaStream>, step_number: u32) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 };
        unsafe {
            stream.launch_builder(&self.signal_fn)
                .arg(&mut self.flag_b)
                .arg(&step_number)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Clear both flags (called from host or coupling kernel after exchange).
    pub fn clear(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 };
        unsafe {
            stream.launch_builder(&self.clear_fn)
                .arg(&mut self.flag_a)
                .arg(&mut self.flag_b)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Read flag values from GPU (for diagnostics).
    pub fn read_flags(&self, stream: &Arc<CudaStream>) -> Result<(u32, u32)> {
        let mut a = [0u32; 1];
        let mut b = [0u32; 1];
        stream.memcpy_dtoh(&self.flag_a, &mut a)?;
        stream.memcpy_dtoh(&self.flag_b, &mut b)?;
        Ok((a[0], b[0]))
    }
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
    pub primary_residue_id: i32,  // lossless residue attribution (was: pad)
}

/// GPU-side ring buffer for spike exchange between twin observation groups.
pub struct TwinRingBuffer {
    // Public for cross-crate access (prism-cuda-ext device compactor)
    pub buffer: CudaSlice<u8>,
    pub head: CudaSlice<u32>,
    pub tail: CudaSlice<u32>,
    pub overflow: CudaSlice<u32>,
    pub capacity: u32,
    /// CPU-side staging for spike compaction (GpuSpikeEvent → RingSpikeEvent)
    staging: Vec<RingSpikeEvent>,
    /// GPU staging buffer for compacted spikes
    d_staging: CudaSlice<u8>,
    // Kernel functions
    push_batch_fn: CudaFunction,
    compact_push_fn: Option<CudaFunction>,  // Gate 3: device-side compact+push (zero CPU memcpy)
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
        let compact_push_fn = module.load_function("compact_and_push").ok();
        if compact_push_fn.is_some() {
            log::info!("  Ring buffer compact_and_push (device-side): loaded");
        }
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
            push_batch_fn, compact_push_fn, read_adapt_fn, recovery_fn, reset_fn,
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

        // Compact on CPU: extract the 10 fields RingSpikeEvent needs.
        // Only push the most recent spikes up to ring capacity to avoid
        // overwhelming the ring buffer. The ring is for RECENT evidence —
        // if we push 1M spikes into a 8192 ring, 99.9% overflow and the
        // adapter only sees the last 8192 anyway. Better to truncate here.
        let max_to_push = self.capacity as usize;
        let spikes_to_push = if gpu_spikes.len() > max_to_push {
            // Take the MOST RECENT spikes (tail of the vector, which are
            // the latest timesteps since accumulated_spikes is append-only)
            &gpu_spikes[gpu_spikes.len() - max_to_push..]
        } else {
            gpu_spikes
        };

        self.staging.clear();
        for s in spikes_to_push {
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
                primary_residue_id: if s.n_residues > 0 { s.nearby_residues[0] } else { -1 },
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

    /// Check if device-side compact_and_push is available.
    pub fn has_device_push(&self) -> bool {
        self.compact_push_fn.is_some()
    }

    /// GPU-direct spike compaction and push — zero CPU memcpy (Gate 3).
    ///
    /// Reads GpuSpikeEvent (92 bytes) directly from device memory,
    /// extracts 12 fields into RingSpikeEvent (48 bytes), and pushes
    /// to ring buffer. All on GPU, never touches CPU or PCIe bus.
    ///
    /// This replaces push_compacted() for the autonomous graph path.
    pub fn push_device(
        &mut self,
        stream: &Arc<CudaStream>,
        d_spike_events: &CudaSlice<u8>,   // engine's d_spike_events buffer
        d_spike_count: &CudaSlice<i32>,    // engine's d_spike_count
    ) -> Result<()> {
        let compact_fn = match &self.compact_push_fn {
            Some(f) => f,
            None => anyhow::bail!("compact_and_push kernel not loaded — cannot use device path"),
        };

        // Launch compact_and_push: one thread per spike, reads spike_count from device
        // We launch max_spikes threads — the kernel checks spike_count internally
        let max_spikes = 65536u32; // upper bound, kernel exits early for tid >= *spike_count
        let n_blocks = (max_spikes + 255) / 256;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(compact_fn)
                .arg(d_spike_events)     // const GpuSpikeEvent*
                .arg(d_spike_count)       // const int*
                .arg(&mut self.buffer)    // ring buffer storage
                .arg(&mut self.head)      // monotonic head counter
                .arg(&mut self.overflow)  // overflow counter
                .arg(&self.capacity)      // ring capacity
                .launch(cfg)?;
        }

        Ok(())
    }

    /// Read spikes from ring buffer and modify target engine's thresholds.
    /// If `d_protocol_state` is provided, writes ASC steering fields on hotspot detection.
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
        self.read_and_adapt_with_steering(
            stream, osc_thresholds, base_thresholds,
            grid_dims, grid_origin, voxel_size,
            sensitivity_boost, max_reduction, current_step, decay_constant,
            None,
        )
    }

    /// Read spikes from ring buffer with optional ASC steering.
    pub fn read_and_adapt_with_steering(
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
        d_protocol_state: Option<&mut CudaSlice<u8>>,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let null_ptr = 0u64; // nullptr for ASC when no protocol state
        if let Some(ps) = d_protocol_state {
            unsafe {
                stream.launch_builder(&self.read_adapt_fn)
                    .arg(&self.buffer).arg(&mut self.head).arg(&mut self.tail)
                    .arg(&self.capacity).arg(osc_thresholds).arg(base_thresholds)
                    .arg(&grid_dims.0).arg(&grid_dims.1).arg(&grid_dims.2)
                    .arg(&grid_origin.0).arg(&grid_origin.1).arg(&grid_origin.2)
                    .arg(&voxel_size).arg(&sensitivity_boost).arg(&max_reduction)
                    .arg(&current_step).arg(&decay_constant)
                    .arg(ps) // ASC: ProtocolState*
                    .launch(cfg)?;
            }
        } else {
            unsafe {
                stream.launch_builder(&self.read_adapt_fn)
                    .arg(&self.buffer).arg(&mut self.head).arg(&mut self.tail)
                    .arg(&self.capacity).arg(osc_thresholds).arg(base_thresholds)
                    .arg(&grid_dims.0).arg(&grid_dims.1).arg(&grid_dims.2)
                    .arg(&grid_origin.0).arg(&grid_origin.1).arg(&grid_origin.2)
                    .arg(&voxel_size).arg(&sensitivity_boost).arg(&max_reduction)
                    .arg(&current_step).arg(&decay_constant)
                    .arg(&null_ptr) // ASC: nullptr
                    .launch(cfg)?;
            }
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

/// Sanitize a float: replace NaN/Inf with 0.0.
/// GPU kernels can produce NaN from 0/0 or Inf from overflow.
fn sanitize(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Extract per-residue CCF features from n_res × n_res CCF matrix.
/// Handles NaN values from GPU computation (norm division by zero when
/// a residue has zero spikes → zero norm → NaN in CCF output).
pub fn extract_ccf_features(ccf_matrix: &[f32], n_res: usize) -> Vec<CcfResidueFeatures> {
    if ccf_matrix.len() != n_res * n_res {
        log::warn!("CCF matrix size {} != expected {}×{}, returning empty features",
            ccf_matrix.len(), n_res, n_res);
        return vec![CcfResidueFeatures::default(); n_res];
    }

    (0..n_res).map(|r| {
        let row = &ccf_matrix[r * n_res..(r + 1) * n_res];

        // Filter out NaN/Inf values before any computation
        let clean_row: Vec<f32> = row.iter().map(|&v| sanitize(v)).collect();

        // Peak value and column (max off-diagonal, excluding self-correlation)
        let (peak_col, peak_val) = clean_row.iter().enumerate()
            .filter(|(j, _)| *j != r)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, &v)| (j as i32, v))
            .unwrap_or((0, 0.0));

        // Width: count of columns with CCF > peak_val * 0.5 (FWHM proxy)
        let half_max = peak_val * 0.5;
        let width = clean_row.iter().filter(|&&v| v > half_max).count() as f32;

        // Asymmetry: (sum upper triangle - sum lower triangle) / total
        // Positive asymmetry = more correlation with higher-index residues
        let upper: f32 = if r + 1 < n_res { clean_row[r+1..].iter().sum() } else { 0.0 };
        let lower: f32 = clean_row[..r].iter().sum();
        let denom = (upper.abs() + lower.abs()).max(1e-8);
        let asymmetry = (upper - lower) / denom;

        // Reproducibility: fraction of residues with CCF > 0.1
        // High reproducibility = this residue correlates with many others
        let reproducibility = clean_row.iter()
            .filter(|&&v| v > 0.1)
            .count() as f32 / n_res.max(1) as f32;

        CcfResidueFeatures {
            ccf_peak_lag: peak_col - r as i32,
            ccf_peak_value: sanitize(peak_val),
            ccf_width: width,
            ccf_asymmetry: sanitize(asymmetry),
            ccf_reproducibility: sanitize(reproducibility),
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

    // Mean-center each row (critical for WMMA — raw counts saturate FP16).
    // Also clamp to FP16 representable range [-65504, 65504] to prevent
    // NaN propagation in the WMMA kernel.
    let fp16_max = 65504.0f32;
    for r in 0..n_residues {
        let start = r * n_bins_padded;

        let sum_a: f32 = mat_a[start..start + n_bins].iter().sum();
        let mean_a = sum_a / n_bins.max(1) as f32;
        for v in mat_a[start..start + n_bins].iter_mut() {
            *v = (*v - mean_a).clamp(-fp16_max, fp16_max);
        }

        let sum_b: f32 = mat_b[start..start + n_bins].iter().sum();
        let mean_b = sum_b / n_bins.max(1) as f32;
        for v in mat_b[start..start + n_bins].iter_mut() {
            *v = (*v - mean_b).clamp(-fp16_max, fp16_max);
        }
    }

    (mat_a, mat_b, n_res_padded as i32, n_bins_padded as i32)
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU Transfer Entropy
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-side binned Transfer Entropy computation.
///
/// Uses the same time-binned spike matrices as the CCF kernel.
/// Produces [n_res × n_res] TE matrices (A→B and B→A) plus
/// per-residue mutual information and causal flow direction.
pub struct TwinTeCompute {
    te_a_to_b: CudaSlice<f32>,     // [n_res × n_res]
    te_b_to_a: CudaSlice<f32>,     // [n_res × n_res]
    mean_a: CudaSlice<f32>,         // [n_res]
    mean_b: CudaSlice<f32>,         // [n_res]
    mutual_info: CudaSlice<f32>,    // [n_res]
    causal_flow: CudaSlice<f32>,    // [n_res]
    n_res: i32,
    n_bins: i32,
    n_res_padded: i32,
    n_bins_padded: i32,
    mean_fn: CudaFunction,
    te_fn: CudaFunction,
    mi_fn: CudaFunction,
}

impl TwinTeCompute {
    pub fn new(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        n_res: i32,
        n_bins: i32,
        n_res_padded: i32,
        n_bins_padded: i32,
    ) -> Result<Self> {
        let te_a_to_b = stream.alloc_zeros::<f32>((n_res * n_res) as usize)?;
        let te_b_to_a = stream.alloc_zeros::<f32>((n_res * n_res) as usize)?;
        let mean_a = stream.alloc_zeros::<f32>(n_res as usize)?;
        let mean_b = stream.alloc_zeros::<f32>(n_res as usize)?;
        let mutual_info = stream.alloc_zeros::<f32>(n_res as usize)?;
        let causal_flow = stream.alloc_zeros::<f32>(n_res as usize)?;

        let mean_fn = module.load_function("twin_compute_row_means")
            .context("twin_compute_row_means not found in PTX")?;
        let te_fn = module.load_function("twin_binned_te")
            .context("twin_binned_te not found in PTX")?;
        let mi_fn = module.load_function("twin_compute_mutual_info")
            .context("twin_compute_mutual_info not found in PTX")?;

        Ok(Self {
            te_a_to_b, te_b_to_a, mean_a, mean_b,
            mutual_info, causal_flow,
            n_res, n_bins, n_res_padded, n_bins_padded,
            mean_fn, te_fn, mi_fn,
        })
    }

    /// Compute means, TE matrices, MI, and causal flow.
    /// Spike matrices must already be uploaded to GPU (same as CCF input).
    pub fn compute(
        &mut self,
        stream: &Arc<CudaStream>,
        spike_matrix_a: &CudaSlice<u16>,  // FP16 as u16, same as CCF
        spike_matrix_b: &CudaSlice<u16>,
    ) -> Result<()> {
        // Step 1: compute per-row means for binarization
        let mean_blocks = ((self.n_res as u32) + 255) / 256;
        let mean_cfg = LaunchConfig {
            grid_dim: (mean_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&self.mean_fn)
                .arg(spike_matrix_a)
                .arg(&mut self.mean_a)
                .arg(&self.n_res)
                .arg(&self.n_bins)
                .arg(&self.n_bins_padded)
                .launch(mean_cfg)?;
            stream.launch_builder(&self.mean_fn)
                .arg(spike_matrix_b)
                .arg(&mut self.mean_b)
                .arg(&self.n_res)
                .arg(&self.n_bins)
                .arg(&self.n_bins_padded)
                .launch(mean_cfg)?;
        }

        // Step 2: compute TE matrices
        let tile = 16u32;
        let grid_x = ((self.n_res as u32) + tile - 1) / tile;
        let grid_y = ((self.n_res as u32) + tile - 1) / tile;
        let te_cfg = LaunchConfig {
            grid_dim: (grid_x.max(1), grid_y.max(1), 1),
            block_dim: (tile, tile, 1),
            shared_mem_bytes: 0,
        };
        let lag = 1i32;
        unsafe {
            stream.launch_builder(&self.te_fn)
                .arg(spike_matrix_a)
                .arg(spike_matrix_b)
                .arg(&mut self.te_a_to_b)
                .arg(&mut self.te_b_to_a)
                .arg(&self.mean_a)
                .arg(&self.mean_b)
                .arg(&self.n_res)
                .arg(&self.n_res_padded)
                .arg(&self.n_bins)
                .arg(&self.n_bins_padded)
                .arg(&lag)
                .launch(te_cfg)?;
        }

        // Step 3: compute per-residue MI and causal flow from TE matrices
        let mi_blocks = ((self.n_res as u32) + 255) / 256;
        let mi_cfg = LaunchConfig {
            grid_dim: (mi_blocks.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&self.mi_fn)
                .arg(&self.te_a_to_b)
                .arg(&self.te_b_to_a)
                .arg(&mut self.mutual_info)
                .arg(&mut self.causal_flow)
                .arg(&self.n_res)
                .launch(mi_cfg)?;
        }

        Ok(())
    }

    /// Download per-residue MI and causal flow to CPU.
    pub fn download_per_residue(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut mi = vec![0.0f32; self.n_res as usize];
        let mut cf = vec![0.0f32; self.n_res as usize];
        stream.memcpy_dtoh(&self.mutual_info, &mut mi)?;
        stream.memcpy_dtoh(&self.causal_flow, &mut cf)?;
        Ok((mi, cf))
    }

    /// Download per-residue TE(A→B) sum (total outgoing TE from each residue).
    pub fn download_te_per_residue(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<f32>> {
        let n = self.n_res as usize;
        let mut te_ab = vec![0.0f32; n * n];
        stream.memcpy_dtoh(&self.te_a_to_b, &mut te_ab)?;
        // Sum across columns for each row: total outgoing TE from residue i
        let per_res: Vec<f32> = (0..n).map(|i| {
            te_ab[i * n..(i + 1) * n].iter()
                .filter(|&&v| v > 0.001)
                .sum()
        }).collect();
        Ok(per_res)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistent Cooperative Coupling Kernel
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the persistent coupling kernel launch.
pub struct TwinCouplingConfig {
    pub sensitivity_boost: f32,
    pub max_reduction_fraction: f32,
    pub decay_constant: f32,
    pub recovery_rate: f32,
    pub recovery_interval: u32,
    pub total_steps: u32,
}

/// GPU state for the persistent cooperative coupling kernel.
///
/// This kernel runs for the ENTIRE simulation on a dedicated CUDA stream.
/// It handles ring buffer exchange + threshold adaptation without returning
/// to host. Physics kernels signal completion via TwinSignal atomic flags.
pub struct TwinCouplingPersistent {
    kernel_fn: CudaFunction,
}

impl TwinCouplingPersistent {
    /// Load the persistent coupling kernel from PTX.
    pub fn new(module: &Arc<CudaModule>) -> Result<Self> {
        let kernel_fn = module.load_function("twin_coupling_persistent")
            .context("twin_coupling_persistent not found in PTX")?;
        Ok(Self { kernel_fn })
    }

    /// Launch the persistent coupling kernel. This call returns immediately —
    /// the kernel runs asynchronously for `config.total_steps` steps.
    ///
    /// The kernel spin-waits on signal flags, so the physics streams must
    /// call `TwinSignal::signal_a/b()` after each step's LADD kernels.
    ///
    /// # Safety
    /// All buffer references must remain valid for the duration of the
    /// simulation (the kernel doesn't return until total_steps complete).
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &Arc<CudaStream>,
        // Signal flags
        signal: &mut TwinSignal,
        // Ring buffers
        ring_a: &mut TwinRingBuffer,
        ring_b: &mut TwinRingBuffer,
        ring_capacity: u32,
        // Spike buffers (from physics engines)
        spike_buf_a: &CudaSlice<u8>,
        spike_count_a: &CudaSlice<i32>,
        spike_buf_b: &CudaSlice<u8>,
        spike_count_b: &CudaSlice<i32>,
        // Threshold buffers
        thresh_a: &mut CudaSlice<f32>,
        base_a: &CudaSlice<f32>,
        thresh_b: &mut CudaSlice<f32>,
        base_b: &CudaSlice<f32>,
        // Grid geometry
        grid_dims: (i32, i32, i32),
        grid_origin: (f32, f32, f32),
        voxel_size: f32,
        // Config
        config: &TwinCouplingConfig,
    ) -> Result<()> {
        // 2 blocks × 256 threads — minimal SM footprint.
        // The persistent kernel only needs 2 blocks (one per stream direction).
        // Using 168 blocks would occupy ALL SMs and starve the physics kernels.
        // With 2 blocks: grid.sync() across 2 blocks is near-instant, and
        // the remaining 82 SMs are free for the fused physics kernel.
        let launch_cfg = LaunchConfig {
            grid_dim: (2, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(&self.kernel_fn)
                // Signal flags (params 0-1)
                .arg(&mut signal.flag_a)
                .arg(&mut signal.flag_b)
                // Ring A: buffer, head, tail, overflow (params 2-5)
                .arg(&mut ring_a.buffer)
                .arg(&mut ring_a.head)
                .arg(&mut ring_a.tail)
                .arg(&mut ring_a.overflow)
                // Ring B: buffer, head, tail, overflow (params 6-9)
                .arg(&mut ring_b.buffer)
                .arg(&mut ring_b.head)
                .arg(&mut ring_b.tail)
                .arg(&mut ring_b.overflow)
                // Ring capacity (param 10)
                .arg(&ring_capacity)
                // Spike buffers (params 11-14)
                .arg(spike_buf_a)
                .arg(spike_count_a)
                .arg(spike_buf_b)
                .arg(spike_count_b)
                // Threshold buffers (params 15-18)
                .arg(thresh_a)
                .arg(base_a)
                .arg(thresh_b)
                .arg(base_b)
                // Grid geometry (params 19-25)
                .arg(&grid_dims.0)
                .arg(&grid_dims.1)
                .arg(&grid_dims.2)
                .arg(&grid_origin.0)
                .arg(&grid_origin.1)
                .arg(&grid_origin.2)
                .arg(&voxel_size)
                // Coupling parameters (params 26-30)
                .arg(&config.sensitivity_boost)
                .arg(&config.max_reduction_fraction)
                .arg(&config.decay_constant)
                .arg(&config.recovery_rate)
                .arg(&config.recovery_interval)
                // Total steps (param 31)
                .arg(&config.total_steps)
                .launch_cooperative(launch_cfg)?;
        }

        log::info!("  Persistent coupling kernel launched: 2 blocks × 256 threads (minimal SM footprint), {} steps",
            config.total_steps);
        Ok(())
    }
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
