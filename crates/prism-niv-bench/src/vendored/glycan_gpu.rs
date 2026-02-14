//! GPU Glycan Masking Module (Vendored for NiV-Bench)

use anyhow::{Context, Result};
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, DeviceSlice, DeviceRepr, PushKernelArg};
use cudarc::driver::CudaContext;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use std::path::Path;

/// Configuration for Glycan GPU operations
#[derive(Debug, Clone)]
pub struct GlycanGpuConfig {
    pub block_size: u32,
}

impl Default for GlycanGpuConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
        }
    }
}

/// GPU-accelerated Glycan Masking
pub struct GlycanGpu {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    mask_kernel: CudaFunction,
    config: GlycanGpuConfig,
}

impl GlycanGpu {
    pub fn new(device: Arc<CudaContext>, ptx_dir: &Path, config: GlycanGpuConfig) -> Result<Self> {
        let ptx_path = ptx_dir.join("glycan_mask.ptx");
        let ptx_src = std::fs::read_to_string(&ptx_path)
            .with_context(|| format!("Failed to read {}", ptx_path.display()))?;
            
        let ptx = Ptx::from_src(ptx_src);
        
        // Load module
        let module = device.load_module(ptx)?;
        let mask_kernel = module.load_function("glycan_mask_kernel")
            .context("Failed to load glycan_mask_kernel function")?;

        let stream = device.new_stream()?;

        Ok(Self {
            device,
            stream,
            mask_kernel,
            config,
        })
    }

    /// Compute glycan mask on GPU
    /// 
    /// sequence: [n_residues] byte array of single-letter codes
    /// ca_coords: [n_residues * 3] float array of CA coordinates (x, y, z)
    /// Returns: [n_residues] uint8 array (1 = shielded, 0 = exposed)
    pub fn compute_mask(
        &self,
        sequence: &[u8],
        ca_coords: &[f32],
    ) -> Result<Vec<u8>> {
        let n_residues = sequence.len();
        if n_residues == 0 {
            return Ok(vec![]);
        }
        
        if ca_coords.len() != n_residues * 3 {
            return Err(anyhow::anyhow!("Coordinates length mismatch"));
        }

        // Allocate and upload inputs
        let mut d_seq = self.stream.alloc_zeros::<u8>(n_residues)?;
        self.stream.memcpy_htod(sequence, &mut d_seq)?;

        let mut d_coords = self.stream.alloc_zeros::<f32>(n_residues * 3)?;
        self.stream.memcpy_htod(ca_coords, &mut d_coords)?;

        // Allocate output
        let d_mask = self.stream.alloc_zeros::<u8>(n_residues)?;

        let block_size = self.config.block_size;
        let grid_size = (n_residues as u32 + block_size - 1) / block_size;

        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch_builder(&self.mask_kernel)
                .arg(&d_seq)
                .arg(&d_coords)
                .arg(&(n_residues as i32))
                .arg(&d_mask)
                .launch(launch_config)?;
        }
        
        self.stream.synchronize()?;

        // Download result
        let mut mask = vec![0u8; n_residues];
        self.stream.memcpy_dtoh(&d_mask, &mut mask)?;
        
        self.stream.synchronize()?;
        
        Ok(mask)
    }
}
