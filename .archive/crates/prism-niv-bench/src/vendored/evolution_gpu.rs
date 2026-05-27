//! GPU Evolution Strategy (OpenAI-ES)
//! Optimized for large Neural Network weight vectors (35k+ params)

use anyhow::{Context, Result};
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, CudaSlice, DeviceRepr, PushKernelArg};
use cudarc::driver::CudaContext;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub sigma: f32,
    pub learning_rate: f32,
    pub n_params: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 64,
            sigma: 0.02,          // Stable exploration
            learning_rate: 0.005, // Stable learning speed
            n_params: 77573,
        }
    }
}

pub struct EvolutionGpu {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    perturb_kernel: CudaFunction,
    update_kernel: CudaFunction,
    init_rng_kernel: CudaFunction,
    
    // State buffers
    pub d_population: CudaSlice<f32>,     // [pop_size * n_params]
    pub d_noise: CudaSlice<f32>,          // [pop_size * n_params]
    pub d_rng_states: CudaSlice<u8>,      // [pop_size * n_params * 48 bytes approx] - actually curandState is ~48 bytes
    
    config: EvolutionConfig,
}

impl EvolutionGpu {
    pub fn new(device: Arc<CudaContext>, ptx_dir: &Path, config: EvolutionConfig) -> Result<Self> {
        let ptx_path = ptx_dir.join("simple_es.ptx");
        let ptx_src = std::fs::read_to_string(&ptx_path)
            .with_context(|| format!("Failed to read {}", ptx_path.display()))?;
            
        let ptx = Ptx::from_src(ptx_src);
        let module = device.load_module(ptx)?;
        
        let perturb_kernel = module.load_function("perturb_weights")?;
        let update_kernel = module.load_function("update_weights")?;
        let init_rng_kernel = module.load_function("init_rng_states")?;
        
        let stream = device.new_stream()?;

        // Allocate buffers
        let total_elements = config.population_size * config.n_params;
        let d_population = stream.alloc_zeros::<f32>(total_elements)?;
        let d_noise = stream.alloc_zeros::<f32>(total_elements)?;
        
        // curandState is 48 bytes. allocating u8 buffer of size 48 * total_elements
        let curand_state_size = 48; 
        let d_rng_states = stream.alloc_zeros::<u8>(total_elements * curand_state_size)?;

        let mut evo = Self {
            device,
            stream,
            perturb_kernel,
            update_kernel,
            init_rng_kernel,
            d_population,
            d_noise,
            d_rng_states,
            config,
        };
        
        evo.init_rng(12345)?; // Seed
        
        Ok(evo)
    }

    fn init_rng(&mut self, seed: u64) -> Result<()> {
        let n_threads = self.config.population_size * self.config.n_params;
        let block_size = 256;
        let grid_size = (n_threads + block_size - 1) / block_size;
        
        unsafe {
            self.stream.launch_builder(&self.init_rng_kernel)
                .arg(&mut self.d_rng_states)
                .arg(&(seed as u64))
                .arg(&(n_threads as i32))
                .launch(LaunchConfig {
                    grid_dim: (grid_size as u32, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// Generate perturbed population around mean weights
    pub fn perturb(&mut self, mean_weights: &CudaSlice<f32>) -> Result<()> {
        let n_params = self.config.n_params;
        let pop_size = self.config.population_size;
        let total_elements = n_params * pop_size;
        
        let block_size = 256;
        let grid_size = (total_elements + block_size - 1) / block_size;

        unsafe {
            self.stream.launch_builder(&self.perturb_kernel)
                .arg(mean_weights)
                .arg(&mut self.d_population)
                .arg(&mut self.d_noise)
                .arg(&mut self.d_rng_states)
                .arg(&(self.config.sigma))
                .arg(&(n_params as i32))
                .arg(&(pop_size as i32))
                .launch(LaunchConfig {
                    grid_dim: (grid_size as u32, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        // self.stream.synchronize()?; // Async
        Ok(())
    }

    /// Update mean weights based on fitness
    /// fitness_scores: [pop_size] on GPU (or we upload it)
    pub fn update(&mut self, mean_weights: &mut CudaSlice<f32>, fitness_scores: &CudaSlice<f32>) -> Result<()> {
        let n_params = self.config.n_params;
        let pop_size = self.config.population_size;
        
        let block_size = 256;
        let grid_size = (n_params + block_size - 1) / block_size; // Parallel over params

        unsafe {
            self.stream.launch_builder(&self.update_kernel)
                .arg(mean_weights) // In/Out
                .arg(&self.d_noise)
                .arg(fitness_scores)
                .arg(&(self.config.learning_rate))
                .arg(&(self.config.sigma))
                .arg(&(n_params as i32))
                .arg(&(pop_size as i32))
                .launch(LaunchConfig {
                    grid_dim: (grid_size as u32, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// Get a specific individual's weights from the population buffer (CPU download)
    pub fn get_individual(&self, idx: usize) -> Result<Vec<f32>> {
        let n_params = self.config.n_params;
        let start = idx * n_params;
        let end = start + n_params;
        
        let slice = self.d_population.slice(start..end);
        let mut host_vec = vec![0.0f32; n_params];
        self.stream.memcpy_dtoh(&slice, &mut host_vec)?;
        self.stream.synchronize()?;
        
        Ok(host_vec)
    }
}
