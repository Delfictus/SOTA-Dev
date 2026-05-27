//! Feature Merge GPU Module (Vendored for NiV-Bench)

use anyhow::{Context, Result};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, DeviceSlice, PushKernelArg, DeviceRepr};
use cudarc::driver::safe::ValidAsZeroBits;
use cudarc::driver::CudaContext;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Configuration for Feature Merge GPU operations
#[derive(Debug, Clone)]
pub struct FeatureMergeConfig {
    /// Block size for CUDA kernel
    pub block_size: u32,
    /// Maximum number of residues per batch
    pub max_residues: usize,
}

impl Default for FeatureMergeConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            max_residues: 1_048_576,
        }
    }
}

/// GPU-accelerated Feature Merge operations
pub struct FeatureMergeGpu {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    feature_merge_kernel: CudaFunction,
    config: FeatureMergeConfig,
}

impl FeatureMergeGpu {
    pub fn new(device: Arc<CudaContext>, config: FeatureMergeConfig) -> Result<Self> {
        // Load PTX
        let ptx_paths = [
            std::path::PathBuf::from("kernels/ptx/feature_merge.ptx"),
            std::path::PathBuf::from("../../kernels/ptx/feature_merge.ptx"),
            std::path::PathBuf::from("target/ptx/feature_merge.ptx"),
            std::path::PathBuf::from("../../target/ptx/feature_merge.ptx"),
        ];
        
        let ptx_path = ptx_paths.iter().find(|p| p.exists())
            .context("Failed to find feature_merge.ptx in kernels/ptx or target/ptx")?;
            
        let ptx_src = std::fs::read_to_string(ptx_path)?;
        
        let ptx = Ptx::from_src(ptx_src);
        
        // Load module
        let module = device.load_module(ptx)?;
        let feature_merge_kernel = module.load_function("feature_merge_kernel")
            .context("Failed to load feature_merge_kernel function")?;

        let stream = device.new_stream()?;

        Ok(Self {
            device,
            stream,
            feature_merge_kernel,
            config,
        })
    }

    pub fn merge_features(
        &self,
        main_features: &CudaSlice<f32>,
        cryptic_features: &CudaSlice<f32>,
        n_residues: usize,
    ) -> Result<CudaSlice<f32>> {
        if n_residues == 0 {
            return Err(anyhow::anyhow!("n_residues must be > 0"));
        }
        
        // Allocate output using stream
        let output_size = n_residues * 140;
        let mut combined_features = self.stream.alloc_zeros::<f32>(output_size)?;

        let block_size = self.config.block_size;
        let grid_size = (n_residues as u32 + block_size - 1) / block_size;

        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch_builder(&self.feature_merge_kernel)
                .arg(main_features)
                .arg(cryptic_features)
                .arg(&mut combined_features)
                .arg(&(n_residues as i32))
                .launch(launch_config)?;
        }
        
        // No sync needed if returning CudaSlice (it's on device)
        // But caller might use it on another stream?
        // Usually fine if on same context?
        // But we should sync if we want to ensure completion before drop?
        // self.stream.synchronize()?;
        
        Ok(combined_features)
    }
}
