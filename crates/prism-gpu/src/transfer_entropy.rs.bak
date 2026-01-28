//! Transfer Entropy GPU acceleration using KSG estimator
//!
//! ASSUMPTIONS:
//! - PTX module "transfer_entropy" loaded in GPU context
//! - Time series data as contiguous f32 arrays
//! - MAX_SERIES_LENGTH = 10000, MAX_VARIABLES = 256
//! - k=4 nearest neighbors for KSG estimator
//! - Requires sm_70+ for efficient sorting
//!
//! PERFORMANCE TARGETS:
//! - 100-variable causal graph: < 500ms inference
//! - Sliding window TE: < 100ms per window
//! - Memory: < 200MB for typical datasets
//!
//! REFERENCE: PRISM Spec Section 5.3 "Causal Discovery via Transfer Entropy"

use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// Transfer entropy computation parameters
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TEParams {
    pub num_variables: i32,
    pub series_length: i32,
    pub history_length: i32,
    pub prediction_lag: i32,
    pub noise_level: f32,
    pub bootstrap_samples: i32,
}

impl Default for TEParams {
    fn default() -> Self {
        Self {
            num_variables: 10,
            series_length: 1000,
            history_length: 3,
            prediction_lag: 1,
            noise_level: 1e-10,
            bootstrap_samples: 100,
        }
    }
}

/// Transfer Entropy GPU accelerator
pub struct TransferEntropyGpu {
    device: Arc<CudaDevice>,

    // Device memory
    time_series: CudaSlice<f32>,
    te_matrix: CudaSlice<f32>,
    significance: CudaSlice<f32>,
    neighbor_counts: CudaSlice<i32>,
    cte_matrix: Option<CudaSlice<f32>>, // Conditional TE

    // Configuration
    params: TEParams,
    max_variables: usize,
    max_series_length: usize,
}

impl TransferEntropyGpu {
    /// Creates a new Transfer Entropy GPU accelerator
    ///
    /// # Arguments
    /// * `device` - CUDA device handle
    /// * `num_variables` - Number of time series variables
    /// * `series_length` - Length of each time series
    ///
    /// # Returns
    /// Initialized TE accelerator with pre-allocated GPU memory
    pub fn new(
        device: Arc<CudaDevice>,
        num_variables: usize,
        series_length: usize,
    ) -> Result<Self> {
        // Validation
        anyhow::ensure!(
            num_variables <= 256,
            "Variables {} exceed maximum 256",
            num_variables
        );
        anyhow::ensure!(
            series_length <= 10000,
            "Series length {} exceeds maximum 10000",
            series_length
        );

        log::info!(
            "Initializing Transfer Entropy GPU: {} variables, {} time points",
            num_variables,
            series_length
        );

        // Load PTX module with explicit kernel list from file system
        let ptx_path = std::path::Path::new("target/ptx/transfer_entropy.ptx");
        let ptx = std::fs::read_to_string(ptx_path)
            .with_context(|| format!("Failed to read Transfer Entropy PTX file: {:?}", ptx_path))?;
        device
            .load_ptx(
                ptx.into(),
                "transfer_entropy",
                &[
                    "transfer_entropy_ksg_kernel",
                    "conditional_te_kernel",
                    "multivariate_te_kernel",
                    "sliding_window_te_kernel",
                    "ensemble_te_kernel",
                    "te_performance_metrics",
                ],
            )
            .context("Failed to load Transfer Entropy PTX module")?;

        log::debug!("Transfer Entropy PTX module loaded with 6 kernels");

        // Allocate device memory
        let ts_size = num_variables * series_length;
        let matrix_size = num_variables * num_variables;

        let time_series = device
            .alloc_zeros::<f32>(ts_size)
            .context("Failed to allocate time series memory")?;

        let te_matrix = device
            .alloc_zeros::<f32>(matrix_size)
            .context("Failed to allocate TE matrix")?;

        let significance = device
            .alloc_zeros::<f32>(matrix_size)
            .context("Failed to allocate significance matrix")?;

        let neighbor_counts = device
            .alloc_zeros::<i32>(series_length * num_variables)
            .context("Failed to allocate neighbor counts")?;

        let params = TEParams {
            num_variables: num_variables as i32,
            series_length: series_length as i32,
            ..Default::default()
        };

        Ok(Self {
            device,
            time_series,
            te_matrix,
            significance,
            neighbor_counts,
            cte_matrix: None,
            params,
            max_variables: num_variables,
            max_series_length: series_length,
        })
    }

    /// Uploads time series data to GPU
    pub fn set_time_series(&mut self, data: &[f32]) -> Result<()> {
        let expected_size = (self.params.num_variables * self.params.series_length) as usize;
        anyhow::ensure!(
            data.len() == expected_size,
            "Data size {} doesn't match expected {}",
            data.len(),
            expected_size
        );

        self.device
            .htod_sync_copy_into(data, &mut self.time_series)
            .context("Failed to upload time series")?;

        Ok(())
    }

    /// Computes transfer entropy matrix using KSG estimator
    pub fn compute_transfer_entropy(&mut self) -> Result<TEMatrix> {
        // Calculate launch configuration
        let threads_per_block = 256;
        let total_pairs = (self.params.num_variables * self.params.num_variables) as u32;
        let blocks = (total_pairs + threads_per_block - 1) / threads_per_block;

        let shared_mem_size =
            (threads_per_block as usize * self.params.series_length as usize * 4) as u32;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: shared_mem_size,
        };

        // Get KSG kernel
        let kernel = self
            .device
            .get_func("transfer_entropy", "transfer_entropy_ksg_kernel")
            .context("Failed to get KSG kernel")?;

        // Launch kernel
        unsafe {
            kernel.launch(
                config,
                (
                    &self.time_series,
                    &self.te_matrix,
                    &self.significance,
                    &self.neighbor_counts,
                    self.params.num_variables,
                    self.params.series_length,
                    self.params.history_length,
                    self.params.prediction_lag,
                    self.params.noise_level,
                ),
            )
        }
        .context("Failed to launch KSG kernel")?;

        self.device.synchronize()?;

        // Download results
        let te_values = self.device.dtoh_sync_copy(&self.te_matrix)?;
        let p_values = self.device.dtoh_sync_copy(&self.significance)?;

        Ok(TEMatrix {
            values: te_values,
            significance: p_values,
            num_variables: self.params.num_variables as usize,
        })
    }

    /// Computes conditional transfer entropy (controlling for confounders)
    pub fn compute_conditional_te(&mut self, conditioning_vars: &[usize]) -> Result<TEMatrix> {
        anyhow::ensure!(
            !conditioning_vars.is_empty(),
            "Must specify conditioning variables"
        );

        // Allocate CTE matrix if needed
        if self.cte_matrix.is_none() {
            let matrix_size = (self.params.num_variables * self.params.num_variables) as usize;
            self.cte_matrix = Some(self.device.alloc_zeros::<f32>(matrix_size)?);
        }

        // Upload conditioning variables
        let conditioning_gpu = self.device.htod_sync_copy(
            &conditioning_vars
                .iter()
                .map(|&v| v as i32)
                .collect::<Vec<_>>(),
        )?;

        // Launch configuration
        let threads_per_block = 256;
        let total_pairs = (self.params.num_variables * self.params.num_variables) as u32;
        let blocks = (total_pairs + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = self
            .device
            .get_func("transfer_entropy", "conditional_te_kernel")
            .context("Failed to get CTE kernel")?;

        let cte_matrix = self.cte_matrix.as_ref().unwrap();

        unsafe {
            kernel.launch(
                config,
                (
                    &self.time_series,
                    &conditioning_gpu,
                    cte_matrix,
                    self.params.num_variables,
                    self.params.series_length,
                    conditioning_vars.len() as i32,
                ),
            )
        }
        .context("Failed to launch CTE kernel")?;

        self.device.synchronize()?;

        let cte_values = self.device.dtoh_sync_copy(cte_matrix)?;

        Ok(TEMatrix {
            values: cte_values,
            significance: vec![
                1.0;
                (self.params.num_variables * self.params.num_variables) as usize
            ],
            num_variables: self.params.num_variables as usize,
        })
    }

    /// Computes sliding window transfer entropy for dynamic analysis
    pub fn compute_sliding_window_te(
        &mut self,
        source_idx: usize,
        target_idx: usize,
        window_size: usize,
        window_stride: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            source_idx < self.params.num_variables as usize,
            "Invalid source index"
        );
        anyhow::ensure!(
            target_idx < self.params.num_variables as usize,
            "Invalid target index"
        );

        let num_windows = ((self.params.series_length as usize - window_size) / window_stride) + 1;

        // Allocate output buffer
        let te_timeseries = self.device.alloc_zeros::<f32>(num_windows)?;

        // Launch configuration
        let threads_per_block = 256;
        let blocks = ((num_windows + threads_per_block - 1) / threads_per_block) as u32;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = self
            .device
            .get_func("transfer_entropy", "sliding_window_te_kernel")
            .context("Failed to get sliding window kernel")?;

        unsafe {
            kernel.launch(
                config,
                (
                    &self.time_series,
                    &te_timeseries,
                    source_idx as i32,
                    target_idx as i32,
                    window_size as i32,
                    window_stride as i32,
                    self.params.series_length,
                ),
            )
        }
        .context("Failed to launch sliding window kernel")?;

        self.device.synchronize()?;

        Ok(self.device.dtoh_sync_copy(&te_timeseries)?)
    }

    /// Computes multivariate transfer entropy (multiple sources to one target)
    pub fn compute_multivariate_te(
        &mut self,
        source_indices: &[usize],
        target_idx: usize,
    ) -> Result<f32> {
        anyhow::ensure!(!source_indices.is_empty(), "Must specify source indices");
        anyhow::ensure!(
            target_idx < self.params.num_variables as usize,
            "Invalid target index"
        );

        // Upload source indices
        let sources_gpu = self
            .device
            .htod_sync_copy(&source_indices.iter().map(|&v| v as i32).collect::<Vec<_>>())?;

        // Allocate result
        let mte_result = self.device.alloc_zeros::<f32>(1)?;

        // Launch configuration
        let threads_per_block = 256;
        let blocks =
            ((self.params.series_length + threads_per_block - 1) / threads_per_block) as u32;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: threads_per_block as u32 * 4,
        };

        let kernel = self
            .device
            .get_func("transfer_entropy", "multivariate_te_kernel")
            .context("Failed to get multivariate TE kernel")?;

        unsafe {
            kernel.launch(
                config,
                (
                    &self.time_series,
                    &sources_gpu,
                    source_indices.len() as i32,
                    target_idx as i32,
                    &mte_result,
                    self.params.series_length,
                    self.params.history_length,
                ),
            )
        }
        .context("Failed to launch multivariate TE kernel")?;

        self.device.synchronize()?;

        let result = self.device.dtoh_sync_copy(&mte_result)?;
        Ok(result[0])
    }

    /// Gets performance metrics for the computed TE matrix
    pub fn get_performance_metrics(&self) -> Result<TEMetrics> {
        // Allocate metrics buffer
        let metrics = self.device.alloc_zeros::<f32>(3)?;

        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = self
            .device
            .get_func("transfer_entropy", "te_performance_metrics")
            .context("Failed to get metrics kernel")?;

        unsafe {
            kernel.launch(
                config,
                (&self.te_matrix, &metrics, self.params.num_variables),
            )
        }
        .context("Failed to launch metrics kernel")?;

        self.device.synchronize()?;

        let metrics_vec = self.device.dtoh_sync_copy(&metrics)?;

        Ok(TEMetrics {
            sparsity: metrics_vec[0],
            avg_strength: metrics_vec[1],
            max_te: metrics_vec[2],
        })
    }

    /// Builds causal graph from TE matrix with threshold
    pub fn build_causal_graph(&self, threshold: f32) -> Result<CausalGraph> {
        let te_values = self.device.dtoh_sync_copy(&self.te_matrix)?;
        let n = self.params.num_variables as usize;

        let mut edges = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let te = te_values[i * n + j];
                    if te > threshold {
                        edges.push(CausalEdge {
                            source: i,
                            target: j,
                            strength: te,
                        });
                    }
                }
            }
        }

        Ok(CausalGraph {
            num_nodes: n,
            edges,
        })
    }

    /// Updates computation parameters
    pub fn set_params(&mut self, params: TEParams) {
        self.params = params;
    }

    /// Gets current parameters
    pub fn params(&self) -> &TEParams {
        &self.params
    }
}

/// Transfer entropy matrix result
#[derive(Debug, Clone)]
pub struct TEMatrix {
    pub values: Vec<f32>,
    pub significance: Vec<f32>,
    pub num_variables: usize,
}

impl TEMatrix {
    /// Gets TE value from source to target
    pub fn get(&self, source: usize, target: usize) -> f32 {
        self.values[source * self.num_variables + target]
    }

    /// Gets p-value for source to target
    pub fn get_pvalue(&self, source: usize, target: usize) -> f32 {
        self.significance[source * self.num_variables + target]
    }

    /// Returns significant edges (p < alpha)
    pub fn significant_edges(&self, alpha: f32) -> Vec<(usize, usize, f32)> {
        let mut edges = Vec::new();
        for i in 0..self.num_variables {
            for j in 0..self.num_variables {
                if i != j && self.get_pvalue(i, j) < alpha {
                    edges.push((i, j, self.get(i, j)));
                }
            }
        }
        edges
    }
}

/// Causal graph structure
#[derive(Debug, Clone)]
pub struct CausalGraph {
    pub num_nodes: usize,
    pub edges: Vec<CausalEdge>,
}

#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub source: usize,
    pub target: usize,
    pub strength: f32,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct TEMetrics {
    pub sparsity: f32,
    pub avg_strength: f32,
    pub max_te: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_te_initialization() {
        let device = CudaDevice::new(0).unwrap();
        let te = TransferEntropyGpu::new(Arc::new(device), 10, 1000);
        assert!(te.is_ok());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_te_computation() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let mut te = TransferEntropyGpu::new(device, 5, 500).unwrap();

        // Generate synthetic time series
        let data: Vec<f32> = (0..2500).map(|i| (i as f32 * 0.01).sin()).collect();
        te.set_time_series(&data).unwrap();

        let matrix = te.compute_transfer_entropy().unwrap();
        assert_eq!(matrix.num_variables, 5);
        assert_eq!(matrix.values.len(), 25);
    }
}
