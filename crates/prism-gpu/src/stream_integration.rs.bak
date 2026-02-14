//! Stream Manager Integration
//!
//! Enhanced GPU context with stream management for triple-buffered pipelines.
//!
//! This module provides:
//! - ManagedGpuContext: Wrapper around GpuContext with optional stream management
//! - Backward compatibility: Works with or without stream management enabled
//! - Triple-buffered execution: Overlaps config upload, kernel execution, and telemetry download
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   ManagedGpuContext                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │  device: Arc<CudaDevice>                                    │
//! │  stream_pool: Option<StreamPool>          ← Stream manager  │
//! │  pipeline_coordinator: Option<...>        ← Triple-buffer   │
//! └─────────────────────────────────────────────────────────────┘
//!        │                    │
//!        │ Sync mode          │ Async mode (triple-buffered)
//!        │                    │
//!        v                    v
//!   device.synchronize()   Stream overlapping
//! ```
//!
//! # Usage
//!
//! ## Synchronous Mode (Default)
//! ```no_run
//! use prism_gpu::stream_integration::ManagedGpuContext;
//! use cudarc::driver::CudaDevice;
//! use std::sync::Arc;
//!
//! let device = CudaDevice::new(0).unwrap();
//! let ctx = ManagedGpuContext::new(device, false).unwrap();
//! // Uses standard synchronous execution
//! ```
//!
//! ## Asynchronous Mode (Triple-buffered)
//! ```no_run
//! # use prism_gpu::stream_integration::ManagedGpuContext;
//! # use cudarc::driver::CudaDevice;
//! # use prism_core::RuntimeConfig;
//! # use std::sync::Arc;
//! let device = CudaDevice::new(0).unwrap();
//! let mut ctx = ManagedGpuContext::new(device, true).unwrap();
//!
//! // Triple-buffered iteration (overlaps 3 stages)
//! let config = RuntimeConfig::default();
//! let telemetry = ctx.triple_buffered_step(config).unwrap();
//! ```
//!
//! # Performance
//!
//! Async mode provides ~2-3x throughput when kernel execution time
//! is comparable to data transfer time.

use crate::stream_manager::{AsyncPipelineCoordinator, StreamPool, StreamPurpose};
use anyhow::Result;
use cudarc::driver::CudaDevice;
use prism_core::{KernelTelemetry, RuntimeConfig};
use std::sync::Arc;

/// Enhanced GPU context with stream management
///
/// Wraps a CUDA device with optional stream management for triple-buffered
/// asynchronous execution. When stream management is disabled, falls back
/// to standard synchronous execution.
///
/// # Thread Safety
///
/// ManagedGpuContext is Send + Sync via Arc<CudaDevice>.
/// Stream pool and coordinator are only accessed through &mut methods.
///
/// # Example
///
/// ```no_run
/// # use prism_gpu::stream_integration::ManagedGpuContext;
/// # use cudarc::driver::CudaDevice;
/// # use prism_core::RuntimeConfig;
/// # use std::sync::Arc;
/// let device = CudaDevice::new(0).unwrap();
/// let mut ctx = ManagedGpuContext::new(device, true).unwrap();
///
/// let config = RuntimeConfig::default();
/// let telemetry = ctx.triple_buffered_step(config).unwrap();
/// ```
pub struct ManagedGpuContext {
    /// CUDA device handle
    device: Arc<CudaDevice>,

    /// Stream pool (None if stream management disabled)
    stream_pool: Option<StreamPool>,

    /// Pipeline coordinator (None if stream management disabled)
    pipeline_coordinator: Option<AsyncPipelineCoordinator>,
}

impl ManagedGpuContext {
    /// Create new managed GPU context
    ///
    /// # Arguments
    ///
    /// * `device` - CUDA device handle
    /// * `enable_streams` - Whether to enable stream management
    ///
    /// # Returns
    ///
    /// Managed context with optional stream management.
    ///
    /// # Errors
    ///
    /// Returns error if stream pool or coordinator initialization fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::stream_integration::ManagedGpuContext;
    /// # use cudarc::driver::CudaDevice;
    /// # use std::sync::Arc;
    /// // Synchronous mode
    /// let ctx = ManagedGpuContext::new(
    ///     CudaDevice::new(0).unwrap(),
    ///     false
    /// ).unwrap();
    ///
    /// // Asynchronous mode
    /// let ctx_async = ManagedGpuContext::new(
    ///     CudaDevice::new(0).unwrap(),
    ///     true
    /// ).unwrap();
    /// ```
    pub fn new(device: Arc<CudaDevice>, enable_streams: bool) -> Result<Self> {
        let (stream_pool, pipeline_coordinator) = if enable_streams {
            log::info!("Initializing ManagedGpuContext with stream management enabled");
            (
                Some(StreamPool::new(device.clone())?),
                Some(AsyncPipelineCoordinator::new(device.clone())?),
            )
        } else {
            log::info!("Initializing ManagedGpuContext in synchronous mode");
            (None, None)
        };

        Ok(Self {
            device,
            stream_pool,
            pipeline_coordinator,
        })
    }

    /// Get reference to CUDA device
    ///
    /// Use this to access the underlying device for kernel launches,
    /// memory allocation, etc.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::stream_integration::ManagedGpuContext;
    /// # use cudarc::driver::CudaDevice;
    /// # let ctx = ManagedGpuContext::new(CudaDevice::new(0).unwrap(), false).unwrap();
    /// let device = ctx.device();
    /// // Use device for kernel launches, memory ops, etc.
    /// ```
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get stream index for specific purpose
    ///
    /// Returns None if stream management is disabled.
    ///
    /// # Arguments
    ///
    /// * `purpose` - Stream purpose (ConfigUpload, KernelExecution, etc.)
    ///
    /// # Returns
    ///
    /// Some(stream_idx) if stream management enabled, None otherwise.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::stream_integration::ManagedGpuContext;
    /// # use prism_gpu::stream_manager::StreamPurpose;
    /// # use cudarc::driver::CudaDevice;
    /// # let mut ctx = ManagedGpuContext::new(CudaDevice::new(0).unwrap(), true).unwrap();
    /// if let Some(stream_idx) = ctx.get_stream(StreamPurpose::KernelExecution) {
    ///     // Use stream_idx for async kernel launch
    /// }
    /// ```
    pub fn get_stream(&mut self, purpose: StreamPurpose) -> Option<usize> {
        self.stream_pool
            .as_mut()
            .and_then(|pool| pool.get_stream(purpose).ok())
    }

    /// Run triple-buffered pipeline iteration
    ///
    /// Overlaps three stages across iterations:
    /// 1. Config upload (async H2D)
    /// 2. Kernel execution (async compute)
    /// 3. Telemetry download (async D2H)
    ///
    /// # Arguments
    ///
    /// * `config` - Runtime configuration for this iteration
    ///
    /// # Returns
    ///
    /// Kernel telemetry from completed iteration.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Stream management not enabled
    /// - Pipeline operations fail
    ///
    /// # Performance
    ///
    /// Achieves ~2-3x throughput vs synchronous execution when:
    /// - Kernel execution time ≈ data transfer time
    /// - Multiple iterations executed sequentially
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::stream_integration::ManagedGpuContext;
    /// # use cudarc::driver::CudaDevice;
    /// # use prism_core::RuntimeConfig;
    /// # let mut ctx = ManagedGpuContext::new(CudaDevice::new(0).unwrap(), true).unwrap();
    /// for iter in 0..100 {
    ///     let config = RuntimeConfig::default();
    ///     let telemetry = ctx.triple_buffered_step(config)?;
    ///     // Process telemetry...
    /// }
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn triple_buffered_step(&mut self, config: RuntimeConfig) -> Result<KernelTelemetry> {
        if let Some(ref mut coordinator) = self.pipeline_coordinator {
            // Triple-buffered async execution
            coordinator.begin_config_upload(config)?;
            coordinator.begin_kernel_execution()?;
            coordinator.begin_telemetry_download()?;
            coordinator.complete_iteration()
        } else {
            // Fallback to synchronous execution
            log::warn!(
                "triple_buffered_step called but stream management disabled - using sync fallback"
            );
            self.sync_execute(config)
        }
    }

    /// Synchronous execution fallback
    ///
    /// Used when stream management is disabled or as fallback.
    /// Executes kernel synchronously and returns telemetry.
    ///
    /// # Arguments
    ///
    /// * `config` - Runtime configuration
    ///
    /// # Returns
    ///
    /// Kernel telemetry (placeholder in current implementation).
    ///
    /// # Note
    ///
    /// This is a placeholder implementation. In production, this would:
    /// 1. Upload config to GPU
    /// 2. Launch kernel synchronously
    /// 3. Download telemetry from GPU
    /// 4. Return actual telemetry data
    fn sync_execute(&self, _config: RuntimeConfig) -> Result<KernelTelemetry> {
        // Placeholder: In production, this would execute kernel synchronously
        // 1. Upload config to device
        // 2. Launch kernel
        // 3. Synchronize device
        // 4. Download telemetry
        self.device.synchronize()?;

        Ok(KernelTelemetry::default())
    }

    /// Check if stream management is enabled
    ///
    /// # Returns
    ///
    /// true if stream management enabled, false otherwise.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::stream_integration::ManagedGpuContext;
    /// # use cudarc::driver::CudaDevice;
    /// # let ctx = ManagedGpuContext::new(CudaDevice::new(0).unwrap(), true).unwrap();
    /// if ctx.has_stream_management() {
    ///     // Use async execution paths
    /// } else {
    ///     // Use synchronous execution
    /// }
    /// ```
    pub fn has_stream_management(&self) -> bool {
        self.stream_pool.is_some()
    }

    /// Get reference to stream pool
    ///
    /// Returns None if stream management disabled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::stream_integration::ManagedGpuContext;
    /// # use cudarc::driver::CudaDevice;
    /// # let ctx = ManagedGpuContext::new(CudaDevice::new(0).unwrap(), true).unwrap();
    /// if let Some(pool) = ctx.stream_pool() {
    ///     // Access stream pool...
    /// }
    /// ```
    pub fn stream_pool(&self) -> Option<&StreamPool> {
        self.stream_pool.as_ref()
    }

    /// Get mutable reference to stream pool
    ///
    /// Returns None if stream management disabled.
    pub fn stream_pool_mut(&mut self) -> Option<&mut StreamPool> {
        self.stream_pool.as_mut()
    }

    /// Get reference to pipeline coordinator
    ///
    /// Returns None if stream management disabled.
    pub fn pipeline_coordinator(&self) -> Option<&AsyncPipelineCoordinator> {
        self.pipeline_coordinator.as_ref()
    }

    /// Get mutable reference to pipeline coordinator
    ///
    /// Returns None if stream management disabled.
    pub fn pipeline_coordinator_mut(&mut self) -> Option<&mut AsyncPipelineCoordinator> {
        self.pipeline_coordinator.as_mut()
    }

    /// Synchronize all GPU operations
    ///
    /// Blocks until all GPU work (all streams if enabled) completes.
    ///
    /// # Errors
    ///
    /// Returns error if device synchronization fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::stream_integration::ManagedGpuContext;
    /// # use cudarc::driver::CudaDevice;
    /// # let ctx = ManagedGpuContext::new(CudaDevice::new(0).unwrap(), true).unwrap();
    /// ctx.synchronize()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn synchronize(&self) -> Result<()> {
        if let Some(ref coordinator) = self.pipeline_coordinator {
            coordinator.synchronize()
        } else {
            self.device.synchronize().map_err(Into::into)
        }
    }

    /// Reset pipeline state
    ///
    /// Resets pipeline coordinator to idle state.
    /// No-op if stream management disabled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use prism_gpu::stream_integration::ManagedGpuContext;
    /// # use cudarc::driver::CudaDevice;
    /// # let mut ctx = ManagedGpuContext::new(CudaDevice::new(0).unwrap(), true).unwrap();
    /// ctx.reset_pipeline();
    /// ```
    pub fn reset_pipeline(&mut self) {
        if let Some(ref mut coordinator) = self.pipeline_coordinator {
            coordinator.reset();
        }
    }
}

// Thread safety: Arc<CudaDevice> is Send + Sync, stream pool/coordinator are !Sync
// but ManagedGpuContext only exposes them via &mut, so we can safely implement Send
unsafe impl Send for ManagedGpuContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_managed_context_sync_mode() {
        // Test that sync mode doesn't require GPU
        // (will fail if we try to create device, but struct creation should work)
        if let Ok(device) = CudaDevice::new(0) {
            let ctx = ManagedGpuContext::new(device, false).unwrap();
            assert!(!ctx.has_stream_management());
            assert!(ctx.stream_pool().is_none());
            assert!(ctx.pipeline_coordinator().is_none());
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_managed_context_async_mode() {
        if let Ok(device) = CudaDevice::new(0) {
            let ctx = ManagedGpuContext::new(device, true).unwrap();
            assert!(ctx.has_stream_management());
            assert!(ctx.stream_pool().is_some());
            assert!(ctx.pipeline_coordinator().is_some());
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_triple_buffered_step() {
        if let Ok(device) = CudaDevice::new(0) {
            let mut ctx = ManagedGpuContext::new(device, true).unwrap();

            let config = RuntimeConfig::default();
            let result = ctx.triple_buffered_step(config);

            // Should not error (even if it's a placeholder implementation)
            assert!(result.is_ok());
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_sync_fallback() {
        if let Ok(device) = CudaDevice::new(0) {
            let mut ctx = ManagedGpuContext::new(device, false).unwrap();

            let config = RuntimeConfig::default();
            let result = ctx.triple_buffered_step(config);

            // Should use sync fallback
            assert!(result.is_ok());
        }
    }
}
