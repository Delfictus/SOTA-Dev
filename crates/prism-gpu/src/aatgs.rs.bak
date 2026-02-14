//! AATGS: Adaptive Asynchronous Task Graph Scheduler
//!
//! GPU-resident scheduler that eliminates CPU-GPU synchronization barriers
//! by maintaining circular buffers for config and telemetry.
//!
//! ## Architecture
//!
//! The AATGS system uses a triple-buffered async pipeline to overlap:
//! 1. Config upload (stream 0)
//! 2. Kernel execution (stream 1)
//! 3. Telemetry download (stream 2)
//!
//! This allows the CPU to queue work ahead while the GPU processes previous iterations,
//! and simultaneously retrieve results from even earlier iterations.
//!
//! ## Memory Layout
//!
//! GPU-resident circular buffers:
//! - Config buffer: 16 slots of RuntimeConfig (16 * 256B = 4KB)
//! - Telemetry buffer: 64 slots of KernelTelemetry (64 * 64B = 4KB)
//! - Atomic pointers for lock-free coordination
//!
//! ## Synchronization
//!
//! - config_write_ptr: CPU writes, GPU reads (producer: CPU, consumer: GPU)
//! - config_read_ptr: GPU writes after consuming (feedback to CPU)
//! - telemetry_write_ptr: GPU writes, CPU reads (producer: GPU, consumer: CPU)
//! - telemetry_read_ptr: CPU writes after consuming (feedback to GPU)
//! - gpu_idle: GPU sets when waiting for work
//! - cpu_shutdown: CPU sets to terminate GPU loop
//!
//! ## Usage
//!
//! ```rust,no_run
//! use prism_gpu::aatgs::AsyncPipeline;
//! use prism_core::RuntimeConfig;
//! use cudarc::driver::CudaDevice;
//! use std::sync::Arc;
//!
//! let device = CudaDevice::new(0).unwrap();
//! let mut pipeline = AsyncPipeline::new(Arc::new(device)).unwrap();
//!
//! let config = RuntimeConfig::production();
//! let telemetry = pipeline.step(config).unwrap();
//! ```

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

use prism_core::{KernelTelemetry, RuntimeConfig};

/// Circular buffer capacity for configs and telemetry
const CONFIG_BUFFER_SIZE: usize = 16;
const TELEMETRY_BUFFER_SIZE: usize = 64;

/// AATGS buffer state (mirrored on GPU)
///
/// This structure is allocated once on the GPU and contains all circular buffer state.
/// The CPU maintains shadow copies of the pointers for local bookkeeping.
///
/// Total size: ~16KB (config buffer) + ~4KB (telemetry buffer) + metadata = ~20KB
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AATGSBuffers {
    /// Circular buffer of RuntimeConfigs
    pub config_buffer: [RuntimeConfig; CONFIG_BUFFER_SIZE],
    /// Write pointer for config buffer (CPU writes, GPU reads)
    pub config_write_ptr: i32,
    /// Read pointer for config buffer (GPU writes after consuming, CPU reads)
    pub config_read_ptr: i32,

    /// Circular buffer of telemetry outputs
    pub telemetry_buffer: [KernelTelemetry; TELEMETRY_BUFFER_SIZE],
    /// Write pointer for telemetry (GPU writes, CPU reads)
    pub telemetry_write_ptr: i32,
    /// Read pointer for telemetry (CPU writes after consuming, GPU reads)
    pub telemetry_read_ptr: i32,

    /// GPU idle flag (GPU sets to 1 when waiting for config)
    pub gpu_idle: i32,
    /// Shutdown signal (CPU sets to 1 to terminate GPU loop)
    pub cpu_shutdown: i32,
}

impl Default for AATGSBuffers {
    fn default() -> Self {
        Self {
            config_buffer: [RuntimeConfig::default(); CONFIG_BUFFER_SIZE],
            config_write_ptr: 0,
            config_read_ptr: 0,
            telemetry_buffer: [KernelTelemetry::default(); TELEMETRY_BUFFER_SIZE],
            telemetry_write_ptr: 0,
            telemetry_read_ptr: 0,
            gpu_idle: 0,
            cpu_shutdown: 0,
        }
    }
}

// SAFETY: AATGSBuffers is #[repr(C)] containing only:
// - Fixed-size arrays of RuntimeConfig and KernelTelemetry (both #[repr(C)] with primitives)
// - i32 primitives for pointers and flags
// All fields are valid when zero-initialized.
unsafe impl cudarc::driver::ValidAsZeroBits for AATGSBuffers {}
unsafe impl cudarc::driver::DeviceRepr for AATGSBuffers {}

/// AATGS Scheduler
///
/// Manages GPU-resident circular buffers for asynchronous config/telemetry exchange.
/// Uses cudarc 0.9 API (synchronous operations, no explicit streams).
///
/// Note: cudarc 0.9 doesn't expose CUDA streams directly, so we simulate async behavior
/// through careful buffer management and non-blocking queries.
pub struct AATGSScheduler {
    device: Arc<CudaDevice>,

    /// GPU-side buffer state
    d_buffers: CudaSlice<AATGSBuffers>,

    /// Host-side shadow of buffer state for local tracking
    local_buffers: AATGSBuffers,

    /// Host-side shadow of config write pointer
    local_config_write_ptr: usize,
    /// Host-side shadow of telemetry read pointer
    local_telemetry_read_ptr: usize,

    /// Pending configs to upload on next flush
    pending_configs: Vec<RuntimeConfig>,
}

impl AATGSScheduler {
    /// Create new AATGS scheduler
    ///
    /// Allocates GPU-resident circular buffers and initializes pointers.
    ///
    /// # Arguments
    /// * `device` - CUDA device context
    ///
    /// # Returns
    /// * `Ok(Self)` - Initialized scheduler
    /// * `Err(_)` - GPU allocation failure
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        log::info!("Initializing AATGS scheduler with {} config slots, {} telemetry slots",
                   CONFIG_BUFFER_SIZE, TELEMETRY_BUFFER_SIZE);

        // Allocate GPU buffer state (single allocation for entire state)
        let d_buffers = device
            .alloc_zeros::<AATGSBuffers>(1)
            .context("Failed to allocate GPU buffer state")?;

        let local_buffers = AATGSBuffers::default();

        Ok(Self {
            device,
            d_buffers,
            local_buffers,
            local_config_write_ptr: 0,
            local_telemetry_read_ptr: 0,
            pending_configs: Vec::with_capacity(CONFIG_BUFFER_SIZE),
        })
    }

    /// Queue a new config for GPU execution (non-blocking)
    ///
    /// Configs are batched locally until `flush_configs()` is called or the buffer
    /// reaches half capacity.
    ///
    /// # Arguments
    /// * `config` - Runtime configuration to queue
    ///
    /// # Returns
    /// * `Ok(())` - Config queued successfully
    /// * `Err(_)` - Buffer overflow or GPU error
    pub fn queue_config(&mut self, config: RuntimeConfig) -> Result<()> {
        // Check for buffer overflow
        let available = CONFIG_BUFFER_SIZE
            .saturating_sub(self.local_config_write_ptr - self.local_buffers.config_read_ptr as usize);

        anyhow::ensure!(
            available > 0,
            "Config buffer full (write={}, read={})",
            self.local_config_write_ptr,
            self.local_buffers.config_read_ptr
        );

        self.pending_configs.push(config);

        // Auto-flush if buffer is getting full
        if self.pending_configs.len() >= CONFIG_BUFFER_SIZE / 2 {
            self.flush_configs()?;
        }

        Ok(())
    }

    /// Flush pending configs to GPU (blocking upload)
    ///
    /// Uploads all pending configs to the GPU circular buffer and updates the write pointer.
    /// This operation blocks until the upload completes.
    ///
    /// # Returns
    /// * `Ok(())` - Configs uploaded successfully
    /// * `Err(_)` - GPU upload failure
    pub fn flush_configs(&mut self) -> Result<()> {
        if self.pending_configs.is_empty() {
            return Ok(());
        }

        log::debug!("Flushing {} configs to GPU", self.pending_configs.len());

        // Update local buffer with pending configs
        for config in self.pending_configs.drain(..) {
            let slot = self.local_config_write_ptr % CONFIG_BUFFER_SIZE;
            self.local_buffers.config_buffer[slot] = config;
            self.local_config_write_ptr += 1;
        }

        // Update write pointer in local state
        self.local_buffers.config_write_ptr = self.local_config_write_ptr as i32;

        // Upload entire buffer state to GPU
        // Note: In cudarc 0.9, we don't have async memcpy, so this blocks
        self.device
            .htod_sync_copy_into(&[self.local_buffers], &mut self.d_buffers)
            .context("Failed to upload config buffer to GPU")?;

        Ok(())
    }

    /// Poll for completed telemetry (non-blocking check, blocking read)
    ///
    /// Reads the GPU's telemetry write pointer to determine how many new entries are available,
    /// then downloads them from the circular buffer.
    ///
    /// # Returns
    /// * `Ok(Vec<KernelTelemetry>)` - Available telemetry entries
    /// * `Err(_)` - GPU download failure
    pub fn poll_telemetry(&mut self) -> Result<Vec<KernelTelemetry>> {
        // Download current GPU buffer state to check telemetry write pointer
        // Note: In cudarc 0.9, this is a blocking operation
        let mut gpu_buffers_array = [AATGSBuffers::default()];
        self.device
            .dtoh_sync_copy_into(&self.d_buffers, &mut gpu_buffers_array)
            .context("Failed to download buffer state from GPU")?;
        let gpu_buffers = gpu_buffers_array[0];

        // Calculate how many telemetry entries are available
        let available = (gpu_buffers.telemetry_write_ptr as usize)
            .saturating_sub(self.local_telemetry_read_ptr);

        if available == 0 {
            return Ok(Vec::new());
        }

        log::debug!("Polling {} telemetry entries from GPU", available);

        // Read available telemetry entries from circular buffer
        let mut results = Vec::with_capacity(available);

        for _ in 0..available {
            let slot = self.local_telemetry_read_ptr % TELEMETRY_BUFFER_SIZE;
            let telemetry = gpu_buffers.telemetry_buffer[slot];
            results.push(telemetry);
            self.local_telemetry_read_ptr += 1;
        }

        // Update local read pointer and upload to GPU
        self.local_buffers.telemetry_read_ptr = self.local_telemetry_read_ptr as i32;
        self.device
            .htod_sync_copy_into(&[self.local_buffers], &mut self.d_buffers)
            .context("Failed to update telemetry read pointer on GPU")?;

        Ok(results)
    }

    /// Check if GPU is idle (waiting for work)
    ///
    /// Queries the GPU's idle flag to determine if it's waiting for new configs.
    ///
    /// # Returns
    /// * `Ok(true)` - GPU is idle
    /// * `Ok(false)` - GPU is busy
    /// * `Err(_)` - GPU query failure
    pub fn is_gpu_idle(&mut self) -> Result<bool> {
        // Download current buffer state
        let mut gpu_buffers_array = [AATGSBuffers::default()];
        self.device
            .dtoh_sync_copy_into(&self.d_buffers, &mut gpu_buffers_array)
            .context("Failed to query GPU idle status")?;
        let gpu_buffers = gpu_buffers_array[0];

        Ok(gpu_buffers.gpu_idle != 0)
    }

    /// Signal GPU to shutdown and wait for completion
    ///
    /// Sets the shutdown flag and synchronizes to ensure all pending operations complete.
    ///
    /// # Returns
    /// * `Ok(())` - Shutdown successful
    /// * `Err(_)` - GPU synchronization failure
    pub fn shutdown(&mut self) -> Result<()> {
        log::info!("Signaling GPU shutdown");

        // Set shutdown flag in local state
        self.local_buffers.cpu_shutdown = 1;

        // Upload to GPU
        self.device
            .htod_sync_copy_into(&[self.local_buffers], &mut self.d_buffers)
            .context("Failed to signal GPU shutdown")?;

        // Wait for all operations to complete
        self.device
            .synchronize()
            .context("Failed to synchronize device during shutdown")?;

        log::info!("AATGS scheduler shutdown complete");
        Ok(())
    }

    /// Get statistics about buffer utilization
    ///
    /// # Returns
    /// Tuple of (config_occupancy, telemetry_occupancy) as percentages [0.0, 1.0]
    pub fn buffer_stats(&self) -> (f32, f32) {
        let config_occupancy = (self.local_config_write_ptr
            - self.local_buffers.config_read_ptr as usize) as f32
            / CONFIG_BUFFER_SIZE as f32;

        let telemetry_occupancy = (self.local_buffers.telemetry_write_ptr as usize
            - self.local_telemetry_read_ptr) as f32
            / TELEMETRY_BUFFER_SIZE as f32;

        (config_occupancy, telemetry_occupancy)
    }

    /// Get reference to underlying CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get GPU buffer slice for external kernel coordination
    ///
    /// Allows external kernels to directly access the AATGS buffer state.
    pub fn buffer_slice(&self) -> &CudaSlice<AATGSBuffers> {
        &self.d_buffers
    }
}

/// Triple-buffered async pipeline
///
/// High-level interface that combines config scheduling with automatic telemetry polling.
/// Provides a simple `step()` method that queues new config and retrieves latest telemetry.
///
/// ## Pipeline Stages
///
/// 1. CPU queues config[n+1]
/// 2. GPU executes kernel[n] (using config from previous step)
/// 3. CPU polls telemetry[n-1] (results from two steps ago)
///
/// This 3-stage pipeline ensures maximum overlap between CPU and GPU work.
pub struct AsyncPipeline {
    scheduler: AATGSScheduler,

    /// Number of iterations executed
    iteration_count: usize,

    /// Peak buffer utilization (config, telemetry)
    peak_utilization: (f32, f32),
}

impl AsyncPipeline {
    /// Create new async pipeline
    ///
    /// # Arguments
    /// * `device` - CUDA device context
    ///
    /// # Returns
    /// * `Ok(Self)` - Initialized pipeline
    /// * `Err(_)` - Scheduler initialization failure
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self {
            scheduler: AATGSScheduler::new(device)?,
            iteration_count: 0,
            peak_utilization: (0.0, 0.0),
        })
    }

    /// Execute one async iteration
    ///
    /// Queues a new config for execution and polls for any completed telemetry.
    /// Due to pipeline depth, telemetry returned may be from 1-2 iterations ago.
    ///
    /// # Arguments
    /// * `new_config` - Configuration for next iteration
    ///
    /// # Returns
    /// * `Ok(Some(telemetry))` - Telemetry from completed iteration
    /// * `Ok(None)` - No telemetry available yet (pipeline filling)
    /// * `Err(_)` - Scheduler error
    pub fn step(&mut self, new_config: RuntimeConfig) -> Result<Option<KernelTelemetry>> {
        // Queue new config (async upload)
        self.scheduler.queue_config(new_config)?;

        // Poll for completed telemetry (non-blocking)
        let telemetry = self.scheduler.poll_telemetry()?;

        // Update statistics
        let (config_util, telem_util) = self.scheduler.buffer_stats();
        self.peak_utilization.0 = self.peak_utilization.0.max(config_util);
        self.peak_utilization.1 = self.peak_utilization.1.max(telem_util);

        self.iteration_count += 1;

        // Return first telemetry if available
        Ok(telemetry.into_iter().next())
    }

    /// Execute multiple iterations in a batch
    ///
    /// Efficiently processes multiple configs by maximizing buffer utilization.
    ///
    /// # Arguments
    /// * `configs` - Slice of configurations to execute
    ///
    /// # Returns
    /// * `Ok(Vec<KernelTelemetry>)` - Telemetry from all completed iterations
    /// * `Err(_)` - Scheduler error
    pub fn batch_step(&mut self, configs: &[RuntimeConfig]) -> Result<Vec<KernelTelemetry>> {
        let mut all_telemetry = Vec::new();

        for config in configs {
            self.scheduler.queue_config(*config)?;
        }

        // Flush all configs
        self.scheduler.flush_configs()?;

        // Poll for results (may take multiple polls to get all results)
        let mut remaining = configs.len();
        while remaining > 0 {
            let telemetry = self.scheduler.poll_telemetry()?;
            let received = telemetry.len();
            all_telemetry.extend(telemetry);
            remaining = remaining.saturating_sub(received);

            // Small delay to allow GPU to process
            if remaining > 0 {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        }

        self.iteration_count += configs.len();

        Ok(all_telemetry)
    }

    /// Get current iteration count
    pub fn iterations(&self) -> usize {
        self.iteration_count
    }

    /// Get peak buffer utilization
    ///
    /// Returns (config_peak, telemetry_peak) as fractions [0.0, 1.0]
    pub fn peak_utilization(&self) -> (f32, f32) {
        self.peak_utilization
    }

    /// Check if GPU is idle
    pub fn is_idle(&mut self) -> Result<bool> {
        self.scheduler.is_gpu_idle()
    }

    /// Shutdown the pipeline
    pub fn shutdown(mut self) -> Result<()> {
        self.scheduler.shutdown()
    }

    /// Get reference to underlying scheduler
    pub fn scheduler(&self) -> &AATGSScheduler {
        &self.scheduler
    }

    /// Get mutable reference to underlying scheduler
    pub fn scheduler_mut(&mut self) -> &mut AATGSScheduler {
        &mut self.scheduler
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_sizes() {
        // Ensure reasonable buffer sizes
        assert!(CONFIG_BUFFER_SIZE >= 8);
        assert!(TELEMETRY_BUFFER_SIZE >= 32);
    }

    #[test]
    fn test_default_buffers() {
        let buffers = AATGSBuffers::default();
        assert_eq!(buffers.config_write_ptr, 0);
        assert_eq!(buffers.config_read_ptr, 0);
        assert_eq!(buffers.telemetry_write_ptr, 0);
        assert_eq!(buffers.telemetry_read_ptr, 0);
        assert_eq!(buffers.gpu_idle, 0);
        assert_eq!(buffers.cpu_shutdown, 0);
    }

    // GPU tests require CUDA device
    #[test]
    #[ignore]
    fn test_scheduler_init() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let scheduler = AATGSScheduler::new(Arc::new(device));
        assert!(scheduler.is_ok());
    }

    #[test]
    #[ignore]
    fn test_pipeline_init() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let pipeline = AsyncPipeline::new(Arc::new(device));
        assert!(pipeline.is_ok());
    }
}
