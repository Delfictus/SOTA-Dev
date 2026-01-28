//! FluxNet GPU Module (Vendored for NiV-Bench)
//! Implements Zero-Copy Inference using Tensor Core DQN Kernel

use anyhow::{Context, Result};
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, CudaSlice, DeviceRepr, PushKernelArg, ValidAsZeroBits};
use cudarc::driver::CudaContext;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use std::path::Path;
use rand::Rng;
use std::mem::size_of;

// Dimensions
const INPUT_DIM: usize = 140;
const HIDDEN1_DIM: usize = 256;
const HIDDEN2_DIM: usize = 128;
const HIDDEN3_DIM: usize = 64;
const VALUE_DIM: usize = 1;
const ADVANTAGE_DIM: usize = 4;

/// Packed DQN weight structure (matches CUDA struct layout)
/// Using f32 for compatibility with Evolution Strategy
/// 
/// NOTE: This struct is ~310KB. DO NOT allocate on stack.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DQNWeights {
    pub weights_fc1: [f32; INPUT_DIM * HIDDEN1_DIM],
    pub bias_fc1: [f32; HIDDEN1_DIM],
    pub weights_fc2: [f32; HIDDEN1_DIM * HIDDEN2_DIM],
    pub bias_fc2: [f32; HIDDEN2_DIM],
    pub weights_fc3: [f32; HIDDEN2_DIM * HIDDEN3_DIM],
    pub bias_fc3: [f32; HIDDEN3_DIM],
    pub weights_value: [f32; HIDDEN3_DIM],
    pub bias_value: [f32; VALUE_DIM],
    pub weights_advantage: [f32; HIDDEN3_DIM * ADVANTAGE_DIM],
    pub bias_advantage: [f32; ADVANTAGE_DIM],
}

// Implement DeviceRepr safely since it's POD
unsafe impl DeviceRepr for DQNWeights {}
unsafe impl ValidAsZeroBits for DQNWeights {}

impl DQNWeights {
    /// Create a new random weight set directly on the heap as a byte vector
    pub fn create_random_heap() -> Vec<u8> {
        let size = size_of::<Self>();
        let num_floats = size / 4;
        let mut rng = rand::thread_rng();
        
        // Create vector of floats directly
        let mut floats = Vec::with_capacity(num_floats);
        for _ in 0..num_floats {
            floats.push(rng.gen_range(-0.05..0.05));
        }
        
        // Convert to bytes
        // Safety: floats are just bytes, layout matches C-struct of arrays of floats
        unsafe {
            let ratio = size_of::<f32>() / size_of::<u8>();
            let length = floats.len() * ratio;
            let capacity = floats.capacity() * ratio;
            let ptr = floats.as_mut_ptr() as *mut u8;
            std::mem::forget(floats);
            Vec::from_raw_parts(ptr, length, capacity)
        }
    }

    pub fn save_bytes(bytes: &[u8], path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load_bytes(path: &Path) -> Result<Vec<u8>> {
        let bytes = std::fs::read(path)?;
        if bytes.len() != size_of::<Self>() {
            anyhow::bail!("File size mismatch for DQNWeights");
        }
        Ok(bytes)
    }
}

pub struct FluxNetGpu {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    inference_kernel: CudaFunction,
    pub weights_buffer: CudaSlice<u8>, // Raw bytes on GPU
}

impl FluxNetGpu {
    pub fn new(device: Arc<CudaContext>, ptx_dir: &Path) -> Result<Self> {
        let ptx_path = ptx_dir.join("dqn_tensor_core.ptx");
        let ptx_src = std::fs::read_to_string(&ptx_path)
            .with_context(|| format!("Failed to read {}", ptx_path.display()))?;
            
        let ptx = Ptx::from_src(ptx_src);
        
        let module = device.load_module(ptx)?;
        let inference_kernel = module.load_function("dqn_batch_inference")
            .context("Failed to load dqn_batch_inference function")?;

        let stream = device.new_stream()?;

        // Try to load weights from file, otherwise default
        let weights_path = Path::new("models/fluxnet_best.bin");
        let weights_bytes = if weights_path.exists() {
            log::info!("Loading trained weights from {:?}", weights_path);
            DQNWeights::load_bytes(weights_path).unwrap_or_else(|e| {
                log::warn!("Failed to load weights: {}. Using random initialization.", e);
                DQNWeights::create_random_heap()
            })
        } else {
            DQNWeights::create_random_heap()
        };

        let mut weights_buffer = stream.alloc_zeros::<u8>(weights_bytes.len())?;
        stream.memcpy_htod(&weights_bytes, &mut weights_buffer)?;

        Ok(Self {
            device,
            stream,
            inference_kernel,
            weights_buffer,
        })
    }

    /// Run Zero-Copy Inference
    /// 
    /// features: [n_residues * 140] Input features on GPU
    /// Returns: [n_residues * 4] Q-values on GPU
    pub fn predict_batch(
        &self,
        features: &CudaSlice<f32>,
        n_residues: usize,
    ) -> Result<CudaSlice<f32>> {
        if n_residues == 0 {
            return Ok(self.stream.alloc_zeros::<f32>(0)?);
        }

        let output_size = n_residues * 4;
        let mut q_values = self.stream.alloc_zeros::<f32>(output_size)?;

        let block_size = 256;
        let grid_size = (n_residues as u32 + block_size - 1) / block_size;

        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Args: features, q_values, total_residues, weights_ptr
        unsafe {
            self.stream.launch_builder(&self.inference_kernel)
                .arg(features)
                .arg(&mut q_values)
                .arg(&(n_residues as i32))
                .arg(&self.weights_buffer) // Pass raw buffer as pointer
                .launch(launch_config)?;
        }
        
        // No sync needed for returning GPU handle, but good for benchmark timing
        self.stream.synchronize()?;
        
        Ok(q_values)
    }
}
