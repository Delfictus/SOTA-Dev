//! Path Integral Monte Carlo (PIMC) GPU acceleration wrapper
//!
//! ASSUMPTIONS:
//! - PTX module "pimc" loaded in GPU context
//! - States represented as f32 arrays
//! - MAX_REPLICAS = 512, MAX_DIMENSIONS = 1024
//! - Requires sm_70+ for atomic operations
//!
//! PERFORMANCE TARGETS:
//! - 100-replica evolution: < 50ms per MC sweep
//! - Replica exchange: < 5ms per exchange round
//! - Memory: < 100MB for typical problem sizes
//!
//! SPARSE MODE (Stage 6):
//! - Config flag `use_sparse` added for future CSR optimization
//! - Currently defaults to dense matrices (sufficient for < 1024 dimensions)
//! - Sparse CSR implementation deferred to future performance optimization phase
//! - No ownership issues exist as sparse mode not yet implemented
//!
//! REFERENCE: PRISM Spec Section 3.3 "Quantum Annealing via PIMC"

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// PIMC parameters for GPU kernel
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PimcParams {
    pub num_replicas: i32,
    pub dimensions: i32,
    pub beta: f32,
    pub delta_tau: f32,
    pub transverse_field: f32,
    pub coupling_strength: f32,
    pub mc_steps: i32,
    pub seed: u64,
    /// Enable sparse matrix mode (CSR format) for large-scale problems
    ///
    /// NOTE (Stage 6): Currently defaults to false (dense mode).
    /// Sparse mode implementation deferred to future optimization phase.
    /// Dense matrices are sufficient for typical PIMC problems (< 1024 dimensions).
    pub use_sparse: bool,
}

impl Default for PimcParams {
    fn default() -> Self {
        Self {
            num_replicas: 32,
            dimensions: 100,
            beta: 1.0,
            delta_tau: 0.1,
            transverse_field: 1.0,
            coupling_strength: 1.0,
            mc_steps: 100,
            seed: 42,
            use_sparse: false, // Dense mode by default (Stage 6)
        }
    }
}

/// Path Integral Monte Carlo GPU accelerator
pub struct PimcGpu {
    device: Arc<CudaDevice>,

    // Device memory allocations
    replicas: CudaSlice<f32>,
    coupling_matrix: CudaSlice<f32>,
    energies: CudaSlice<f32>,
    magnetizations: CudaSlice<f32>,
    acceptance_rates: CudaSlice<f32>,
    temperatures: CudaSlice<f32>,
    exchange_counts: CudaSlice<i32>,

    // Configuration
    params: PimcParams,
    max_replicas: usize,
    max_dimensions: usize,
}

impl PimcGpu {
    /// Creates a new PIMC GPU accelerator
    ///
    /// # Arguments
    /// * `device` - CUDA device handle
    /// * `num_replicas` - Number of Trotter slices
    /// * `dimensions` - Problem dimensionality
    ///
    /// # Returns
    /// Initialized PIMC accelerator with pre-allocated GPU memory
    ///
    /// # Errors
    /// Returns error if:
    /// - Dimensions exceed MAX_DIMENSIONS (1024)
    /// - Replicas exceed MAX_REPLICAS (512)
    /// - GPU memory allocation fails
    pub fn new(device: Arc<CudaDevice>, num_replicas: usize, dimensions: usize) -> Result<Self> {
        // Validation
        anyhow::ensure!(
            dimensions <= 1024,
            "Dimensions {} exceed maximum 1024",
            dimensions
        );
        anyhow::ensure!(
            num_replicas <= 512,
            "Replicas {} exceed maximum 512",
            num_replicas
        );

        log::info!(
            "Initializing PIMC GPU: {} replicas, {} dimensions",
            num_replicas,
            dimensions
        );

        // Load PTX module with explicit kernel list from file system
        let ptx_path = std::path::Path::new("target/ptx/pimc.ptx");
        let ptx = std::fs::read_to_string(ptx_path)
            .with_context(|| format!("Failed to read PIMC PTX file: {:?}", ptx_path))?;
        device
            .load_ptx(
                ptx.into(),
                "pimc",
                &[
                    "pimc_evolution_kernel",
                    "replica_exchange_kernel",
                    "update_annealing_schedule",
                    "compute_ensemble_statistics",
                    "pimc_performance_metrics",
                    "initialize_random_spins",
                ],
            )
            .context("Failed to load PIMC PTX module")?;

        log::debug!("PIMC PTX module loaded with 6 kernels");

        // Allocate device memory
        let replicas_size = num_replicas * dimensions;
        let coupling_size = dimensions * dimensions;

        let replicas = device
            .alloc_zeros::<f32>(replicas_size)
            .context("Failed to allocate replicas memory")?;

        let coupling_matrix = device
            .alloc_zeros::<f32>(coupling_size)
            .context("Failed to allocate coupling matrix")?;

        let energies = device
            .alloc_zeros::<f32>(num_replicas)
            .context("Failed to allocate energies")?;

        let magnetizations = device
            .alloc_zeros::<f32>(num_replicas)
            .context("Failed to allocate magnetizations")?;

        let acceptance_rates = device
            .alloc_zeros::<f32>(num_replicas)
            .context("Failed to allocate acceptance rates")?;

        let temperatures = device
            .alloc_zeros::<f32>(num_replicas)
            .context("Failed to allocate temperatures")?;

        let exchange_counts = device
            .alloc_zeros::<i32>(num_replicas * num_replicas)
            .context("Failed to allocate exchange counts")?;

        let params = PimcParams {
            num_replicas: num_replicas as i32,
            dimensions: dimensions as i32,
            ..Default::default()
        };

        Ok(Self {
            device,
            replicas,
            coupling_matrix,
            energies,
            magnetizations,
            acceptance_rates,
            temperatures,
            exchange_counts,
            params,
            max_replicas: num_replicas,
            max_dimensions: dimensions,
        })
    }

    /// Initializes random spin configurations
    pub fn initialize_random(&mut self, seed: u64) -> Result<()> {
        self.params.seed = seed;

        // Launch initialization kernel
        let threads_per_block = 256;
        let total_spins = self.params.num_replicas * self.params.dimensions;
        let blocks = ((total_spins + threads_per_block - 1) / threads_per_block) as u32;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get kernel function
        let kernel = self
            .device
            .get_func("pimc", "initialize_random_spins")
            .context("Failed to get initialization kernel")?;

        unsafe {
            kernel.launch(
                config,
                (
                    &self.replicas,
                    self.params.num_replicas,
                    self.params.dimensions,
                    seed,
                ),
            )
        }
        .context("Failed to launch initialization kernel")?;

        self.device.synchronize()?;
        Ok(())
    }

    /// Sets the coupling matrix for the Ising model
    pub fn set_coupling_matrix(&mut self, coupling: &[f32]) -> Result<()> {
        anyhow::ensure!(
            coupling.len() == (self.params.dimensions * self.params.dimensions) as usize,
            "Coupling matrix size mismatch"
        );

        self.device
            .htod_sync_copy_into(&coupling, &mut self.coupling_matrix)
            .context("Failed to upload coupling matrix")?;

        Ok(())
    }

    /// Runs PIMC evolution for specified Monte Carlo steps
    pub fn evolve(&mut self, mc_steps: usize) -> Result<()> {
        self.params.mc_steps = mc_steps as i32;

        // Calculate launch configuration
        let threads_per_block = 256;
        let blocks = self.params.num_replicas as u32;
        let shared_mem_size = threads_per_block * self.params.dimensions as usize * 4;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: shared_mem_size as u32,
        };

        // Get evolution kernel
        let kernel = self
            .device
            .get_func("pimc", "pimc_evolution_kernel")
            .context("Failed to get evolution kernel")?;

        // Launch kernel
        unsafe {
            kernel.launch(
                config,
                (
                    &self.replicas,
                    &self.coupling_matrix,
                    &self.energies,
                    &self.magnetizations,
                    &self.acceptance_rates,
                    self.params.num_replicas,
                    self.params.dimensions,
                    self.params.beta,
                    self.params.transverse_field,
                    self.params.mc_steps,
                    self.params.seed,
                ),
            )
        }
        .context("Failed to launch evolution kernel")?;

        self.device.synchronize()?;
        Ok(())
    }

    /// Performs replica exchange between temperature replicas
    pub fn replica_exchange(&mut self) -> Result<()> {
        // Update temperature schedule
        self.update_temperatures()?;

        // Launch exchange kernel
        let threads_per_block = 128;
        let num_pairs = self.params.num_replicas / 2;
        let blocks = ((num_pairs + threads_per_block - 1) / threads_per_block) as u32;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = self
            .device
            .get_func("pimc", "replica_exchange_kernel")
            .context("Failed to get exchange kernel")?;

        unsafe {
            kernel.launch(
                config,
                (
                    &self.replicas,
                    &self.temperatures,
                    &self.energies,
                    &self.exchange_counts,
                    self.params.num_replicas,
                    self.params.seed,
                ),
            )
        }
        .context("Failed to launch exchange kernel")?;

        self.device.synchronize()?;
        Ok(())
    }

    /// Updates annealing schedule (transverse field and temperatures)
    pub fn update_annealing_schedule(&mut self, progress: f32) -> Result<()> {
        anyhow::ensure!(
            progress >= 0.0 && progress <= 1.0,
            "Progress must be in [0, 1]"
        );

        // Update transverse field (quantum fluctuations)
        self.params.transverse_field = 1.0 * (1.0 - progress);

        // Launch schedule update kernel
        let threads_per_block = 256;
        let blocks =
            ((self.params.num_replicas + threads_per_block - 1) / threads_per_block) as u32;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = self
            .device
            .get_func("pimc", "update_annealing_schedule")
            .context("Failed to get schedule kernel")?;

        // Create transverse fields array
        let transverse_fields = self
            .device
            .alloc_zeros::<f32>(self.params.num_replicas as usize)?;

        unsafe {
            kernel.launch(
                config,
                (
                    &transverse_fields,
                    &self.temperatures,
                    progress,
                    1.0f32, // total_time
                    self.params.num_replicas,
                ),
            )
        }
        .context("Failed to launch schedule kernel")?;

        self.device.synchronize()?;
        Ok(())
    }

    /// Computes ensemble statistics for CMA-ES integration
    pub fn compute_ensemble_statistics(&self) -> Result<EnsembleStatistics> {
        // Allocate output buffers
        let mean_state = self
            .device
            .alloc_zeros::<f32>(self.params.dimensions as usize)?;

        let covariance_size = (self.params.dimensions * self.params.dimensions) as usize;
        let covariance = self.device.alloc_zeros::<f32>(covariance_size)?;

        let entropy = self.device.alloc_zeros::<f32>(1)?;

        // Launch statistics kernel
        let threads_per_block = 256;
        let blocks = self.params.dimensions as u32;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: threads_per_block * 4,
        };

        let kernel = self
            .device
            .get_func("pimc", "compute_ensemble_statistics")
            .context("Failed to get statistics kernel")?;

        unsafe {
            kernel.launch(
                config,
                (
                    &self.replicas,
                    &self.energies,
                    &mean_state,
                    &covariance,
                    &entropy,
                    self.params.num_replicas,
                    self.params.dimensions,
                    self.params.beta,
                ),
            )
        }
        .context("Failed to launch statistics kernel")?;

        self.device.synchronize()?;

        // Download results
        let mean_vec = self.device.dtoh_sync_copy(&mean_state)?;
        let cov_vec = self.device.dtoh_sync_copy(&covariance)?;
        let entropy_val = self.device.dtoh_sync_copy(&entropy)?;

        Ok(EnsembleStatistics {
            mean_state: mean_vec,
            covariance: cov_vec,
            entropy: entropy_val[0],
        })
    }

    /// Gets current observables (energy, magnetization, acceptance)
    pub fn get_observables(&self) -> Result<PimcObservables> {
        let energies = self.device.dtoh_sync_copy(&self.energies)?;
        let magnetizations = self.device.dtoh_sync_copy(&self.magnetizations)?;
        let acceptance_rates = self.device.dtoh_sync_copy(&self.acceptance_rates)?;

        // Compute averages
        let num_replicas = self.params.num_replicas as usize;
        let avg_energy = energies.iter().sum::<f32>() / num_replicas as f32;
        let avg_magnetization = magnetizations.iter().sum::<f32>() / num_replicas as f32;
        let avg_acceptance = acceptance_rates.iter().sum::<f32>() / num_replicas as f32;

        Ok(PimcObservables {
            energies,
            magnetizations,
            acceptance_rates,
            avg_energy,
            avg_magnetization,
            avg_acceptance,
        })
    }

    /// Gets the best (lowest energy) configuration
    pub fn get_best_configuration(&self) -> Result<Vec<f32>> {
        // Download energies to find best replica
        let energies = self.device.dtoh_sync_copy(&self.energies)?;

        let (best_idx, _) = energies
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Download best replica
        let replicas = self.device.dtoh_sync_copy(&self.replicas)?;
        let start = best_idx * self.params.dimensions as usize;
        let end = start + self.params.dimensions as usize;

        Ok(replicas[start..end].to_vec())
    }

    /// Updates temperature ladder for parallel tempering
    fn update_temperatures(&mut self) -> Result<()> {
        // Geometric temperature spacing
        let t_min = 0.01f32;
        let t_max = 10.0f32;
        let num_replicas = self.params.num_replicas as usize;

        let mut temps = vec![0.0f32; num_replicas];
        let alpha = (t_max / t_min).powf(1.0 / (num_replicas - 1) as f32);

        for i in 0..num_replicas {
            temps[i] = t_min * alpha.powi(i as i32);
        }

        self.device
            .htod_sync_copy_into(&temps, &mut self.temperatures)
            .context("Failed to update temperatures")?;

        Ok(())
    }

    /// Gets performance metrics
    pub fn get_performance_metrics(&self) -> Result<PimcMetrics> {
        // Allocate metrics buffer
        let metrics = self.device.alloc_zeros::<f32>(4)?;

        // Launch metrics kernel
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = self
            .device
            .get_func("pimc", "pimc_performance_metrics")
            .context("Failed to get metrics kernel")?;

        let total_exchanges = 100; // Placeholder

        unsafe {
            kernel.launch(
                config,
                (
                    &self.acceptance_rates,
                    &self.exchange_counts,
                    &metrics,
                    self.params.num_replicas,
                    total_exchanges,
                ),
            )
        }
        .context("Failed to launch metrics kernel")?;

        self.device.synchronize()?;

        let metrics_vec = self.device.dtoh_sync_copy(&metrics)?;

        Ok(PimcMetrics {
            mc_efficiency: metrics_vec[0],
            tunneling_rate: metrics_vec[1],
            equilibration: metrics_vec[2],
        })
    }

    /// Updates parameters
    pub fn set_params(&mut self, params: PimcParams) {
        self.params = params;
    }

    /// Gets current parameters
    pub fn params(&self) -> &PimcParams {
        &self.params
    }
}

/// Ensemble statistics for CMA-ES integration
#[derive(Debug, Clone)]
pub struct EnsembleStatistics {
    pub mean_state: Vec<f32>,
    pub covariance: Vec<f32>,
    pub entropy: f32,
}

/// PIMC observables
#[derive(Debug, Clone)]
pub struct PimcObservables {
    pub energies: Vec<f32>,
    pub magnetizations: Vec<f32>,
    pub acceptance_rates: Vec<f32>,
    pub avg_energy: f32,
    pub avg_magnetization: f32,
    pub avg_acceptance: f32,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PimcMetrics {
    pub mc_efficiency: f32,
    pub tunneling_rate: f32,
    pub equilibration: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_pimc_initialization() {
        let device = CudaDevice::new(0).unwrap();
        let pimc = PimcGpu::new(Arc::new(device), 32, 100);
        assert!(pimc.is_ok());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_pimc_evolution() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let mut pimc = PimcGpu::new(device, 16, 50).unwrap();

        pimc.initialize_random(42).unwrap();
        pimc.evolve(100).unwrap();

        let obs = pimc.get_observables().unwrap();
        assert!(obs.avg_acceptance > 0.0);
    }
}
