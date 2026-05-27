//! Phase 0: Dendritic Reservoir with FluxNet RL and GPU Acceleration.
//!
//! Provides neuromorphic dendritic computation for warmstart prior generation.
//! Supports both GPU-accelerated and CPU fallback modes.

use prism_core::{Graph, PhaseContext, PhaseController, PhaseOutcome, PhaseTelemetry, PrismError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
use prism_gpu::DendriticReservoirGpu;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Configuration for Phase 0 Dendritic Reservoir
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase0Config {
 #[serde(default = "default_num_branches")]
 pub num_branches: usize,

 #[serde(default = "default_branch_depth")]
 pub branch_depth: usize,

 #[serde(default = "default_learning_rate")]
 pub learning_rate: f32,

 #[serde(default = "default_plasticity")]
 pub plasticity: f32,

 #[serde(default = "default_activation_threshold")]
 pub activation_threshold: f32,

 #[serde(default = "default_reservoir_size")]
 pub reservoir_size: usize,

 #[serde(default = "default_readout_size")]
 pub readout_size: usize,

 #[serde(default = "default_gpu_enabled")]
 pub gpu_enabled: bool,
}

fn default_num_branches() -> usize {
 10
}
fn default_branch_depth() -> usize {
 6
}
fn default_learning_rate() -> f32 {
 0.01
}
fn default_plasticity() -> f32 {
 0.05
}
fn default_activation_threshold() -> f32 {
 0.5
}
fn default_reservoir_size() -> usize {
 512
}
fn default_readout_size() -> usize {
 128
}
fn default_gpu_enabled() -> bool {
 true
}

impl Default for Phase0Config {
 fn default() -> Self {
 Self {
 num_branches: default_num_branches(),
 branch_depth: default_branch_depth(),
 learning_rate: default_learning_rate(),
 plasticity: default_plasticity(),
 activation_threshold: default_activation_threshold(),
 reservoir_size: default_reservoir_size(),
 readout_size: default_readout_size(),
 gpu_enabled: default_gpu_enabled(),
 }
 }
}

/// Phase 0: Dendritic Reservoir controller with GPU acceleration.
///
/// Computes difficulty and uncertainty metrics for each vertex using
/// multi-branch dendritic reservoir computation. These metrics drive
/// the warmstart prior softmax distribution.
///
/// ## GPU Mode
/// When CUDA is available and enabled:
/// - Uses GPU-accelerated dendritic reservoir kernels
/// - Target: 250 vertices in < 100ms
/// - Configurable branches, leak rate, iterations
///
/// ## CPU Fallback Mode
/// When GPU is unavailable:
/// - Uses simplified heuristics (degree-based difficulty)
/// - Faster but less accurate than full reservoir computation
///
/// Resolved TODO(GPU-Phase0): Dendritic reservoir kernel integrated.
pub struct Phase0DendriticReservoir {
 /// GPU reservoir (if CUDA enabled and available)
 #[cfg(feature = "cuda")]
 gpu_reservoir: Option<Arc<DendriticReservoirGpu>>,

 /// Whether to use GPU acceleration
 use_gpu: bool,

 /// Last computed difficulty metrics (for telemetry)
 last_difficulty: Vec<f32>,

 /// Last computed uncertainty metrics (for telemetry)
 last_uncertainty: Vec<f32>,

 /// Structured telemetry for Phase 0 (Warmstart Plan Step 2)
 telemetry: Option<prism_core::Phase0Telemetry>,

 /// Execution time for last run (milliseconds)
 last_execution_time_ms: f64,

 /// Number of iterations for reservoir convergence
 reservoir_iterations: usize,

 /// Final convergence loss
 convergence_loss: f32,

 /// Configuration parameters
 num_branches: usize,
 branch_depth: usize,
 learning_rate: f32,
 plasticity: f32,
 activation_threshold: f32,
 reservoir_size: usize,
 readout_size: usize,
}

impl Default for Phase0DendriticReservoir {
 fn default() -> Self {
 Self::new()
 }
}

impl Phase0DendriticReservoir {
 /// Creates a new Phase0 controller (CPU mode)
 pub fn new() -> Self {
 let default_config = Phase0Config::default();
 log::info!("Phase0DendriticReservoir: CPU mode");
 Self {
 #[cfg(feature = "cuda")]
 gpu_reservoir: None,
 use_gpu: false,
 last_difficulty: Vec::new(),
 last_uncertainty: Vec::new(),
 telemetry: None,
 last_execution_time_ms: 0.0,
 reservoir_iterations: 0,
 convergence_loss: 0.0,
 num_branches: default_config.num_branches,
 branch_depth: default_config.branch_depth,
 learning_rate: default_config.learning_rate,
 plasticity: default_config.plasticity,
 activation_threshold: default_config.activation_threshold,
 reservoir_size: default_config.reservoir_size,
 readout_size: default_config.readout_size,
 }
 }

 /// Creates Phase0 controller with GPU acceleration
 ///
 /// # Arguments
 /// * `ptx_path` - Path to dendritic_reservoir.ptx (e.g., "target/ptx/dendritic_reservoir.ptx")
 ///
 /// # Errors
 /// Returns error if:
 /// - CUDA device initialization fails
 /// - PTX module loading fails
 ///
 /// # Example
 /// ```rust,no_run
 /// use prism_phases::phase0::Phase0DendriticReservoir;
 ///
 /// let phase0 = Phase0DendriticReservoir::new_with_gpu(
 /// "target/ptx/dendritic_reservoir.ptx"
 /// ).expect("GPU initialization failed");
 /// ```
 #[cfg(feature = "cuda")]
 pub fn new_with_gpu(ptx_path: &str) -> Result<Self, PrismError> {
 log::info!("Phase0DendriticReservoir: Initializing GPU mode");

 // Initialize CUDA context (device 0)
 let device = CudaContext::new(0)
 .map_err(|e| PrismError::gpu("Phase0", format!("CUDA context init failed: {}", e)))?;

 // Load dendritic reservoir PTX module
 let reservoir = DendriticReservoirGpu::new(device, ptx_path)
 .map_err(|e| PrismError::gpu("Phase0", format!("PTX loading failed: {}", e)))?;

 log::info!("Phase0DendriticReservoir: GPU initialized successfully");

 let default_config = Phase0Config::default();
 Ok(Self {
 gpu_reservoir: Some(Arc::new(reservoir)),
 use_gpu: true,
 last_difficulty: Vec::new(),
 last_uncertainty: Vec::new(),
 telemetry: None,
 last_execution_time_ms: 0.0,
 reservoir_iterations: 0,
 convergence_loss: 0.0,
 num_branches: default_config.num_branches,
 branch_depth: default_config.branch_depth,
 learning_rate: default_config.learning_rate,
 plasticity: default_config.plasticity,
 activation_threshold: default_config.activation_threshold,
 reservoir_size: default_config.reservoir_size,
 readout_size: default_config.readout_size,
 })
 }

 /// Creates Phase0 controller with custom GPU parameters
 ///
 /// # Arguments
 /// * `ptx_path` - Path to PTX module
 /// * `num_branches` - Number of dendritic branches (1-32)
 /// * `leak_rate` - Leak rate for temporal dynamics (0.0-1.0)
 /// * `iterations` - Number of propagation iterations
 #[cfg(feature = "cuda")]
 pub fn new_with_gpu_params(
 ptx_path: &str,
 num_branches: usize,
 leak_rate: f32,
 iterations: usize,
 ) -> Result<Self, PrismError> {
 log::info!(
 "Phase0DendriticReservoir: Initializing GPU mode (branches={}, leak={}, iter={})",
 num_branches,
 leak_rate,
 iterations
 );

 // Initialize CUDA context
 let device = CudaContext::new(0)
 .map_err(|e| PrismError::gpu("Phase0", format!("CUDA context init failed: {}", e)))?;

 let reservoir = DendriticReservoirGpu::new_with_params(
 device,
 ptx_path,
 num_branches,
 leak_rate,
 iterations,
 )
 .map_err(|e| PrismError::gpu("Phase0", format!("PTX loading failed: {}", e)))?;

 let default_config = Phase0Config::default();
 Ok(Self {
 gpu_reservoir: Some(Arc::new(reservoir)),
 use_gpu: true,
 last_difficulty: Vec::new(),
 last_uncertainty: Vec::new(),
 telemetry: None,
 last_execution_time_ms: 0.0,
 reservoir_iterations: iterations,
 convergence_loss: 0.0,
 num_branches: default_config.num_branches,
 branch_depth: default_config.branch_depth,
 learning_rate: default_config.learning_rate,
 plasticity: default_config.plasticity,
 activation_threshold: default_config.activation_threshold,
 reservoir_size: default_config.reservoir_size,
 readout_size: default_config.readout_size,
 })
 }

 /// Creates Phase0 controller with custom config (CPU mode)
 pub fn with_config(config: Phase0Config) -> Self {
 log::info!(
 "Phase0: Initializing with config: branches={}, depth={}, lr={:.3}",
 config.num_branches,
 config.branch_depth,
 config.learning_rate
 );

 Self {
 #[cfg(feature = "cuda")]
 gpu_reservoir: None,
 use_gpu: false,
 last_difficulty: Vec::new(),
 last_uncertainty: Vec::new(),
 telemetry: None,
 last_execution_time_ms: 0.0,
 reservoir_iterations: 0,
 convergence_loss: 0.0,
 num_branches: config.num_branches,
 branch_depth: config.branch_depth,
 learning_rate: config.learning_rate,
 plasticity: config.plasticity,
 activation_threshold: config.activation_threshold,
 reservoir_size: config.reservoir_size,
 readout_size: config.readout_size,
 }
 }

 /// Creates Phase0 controller with custom config and GPU acceleration
 #[cfg(feature = "cuda")]
 pub fn with_config_and_gpu(config: Phase0Config, ptx_path: &str) -> Result<Self, PrismError> {
 let mut phase = Self::with_config(config.clone());
 if config.gpu_enabled {
 let device = CudaContext::new(0).map_err(|e| {
 PrismError::gpu("Phase0", format!("CUDA context init failed: {}", e))
 })?;

 match DendriticReservoirGpu::new(device, ptx_path) {
 Ok(gpu) => {
 phase.gpu_reservoir = Some(Arc::new(gpu));
 phase.use_gpu = true;
 log::info!("Phase0: GPU acceleration enabled with custom config");
 }
 Err(e) => {
 log::warn!("Phase0: GPU init failed: {}, using CPU", e);
 }
 }
 }
 Ok(phase)
 }

 /// Computes reservoir metrics (difficulty and uncertainty) for graph
 ///
 /// Uses GPU acceleration if available, otherwise falls back to CPU heuristics.
 ///
 /// # Returns
 /// Tuple of (difficulty, uncertainty) vectors, each of length num_vertices.
 pub fn compute_reservoir_metrics(
 &mut self,
 graph: &Graph,
 ) -> Result<(Vec<f32>, Vec<f32>), PrismError> {
 #[cfg(feature = "cuda")]
 if self.use_gpu {
 return self.compute_gpu_metrics(graph);
 }

 // CPU fallback
 self.compute_cpu_metrics(graph)
 }

 /// GPU path: Compute metrics using CUDA kernels
 #[cfg(feature = "cuda")]
 fn compute_gpu_metrics(&mut self, graph: &Graph) -> Result<(Vec<f32>, Vec<f32>), PrismError> {
 let reservoir = self
 .gpu_reservoir
 .as_ref()
 .ok_or_else(|| PrismError::gpu("Phase0", "GPU reservoir not initialized"))?;

 log::debug!(
 "Computing reservoir metrics on GPU for {} vertices",
 graph.num_vertices
 );

 let (difficulty, uncertainty) = reservoir
 .compute_metrics(&graph.adjacency, graph.num_vertices)
 .map_err(|e| {
 PrismError::gpu("Phase0", format!("GPU reservoir computation failed: {}", e))
 })?;

 // Cache for telemetry
 self.last_difficulty = difficulty.clone();
 self.last_uncertainty = uncertainty.clone();

 log::debug!(
 "GPU metrics computed: mean_difficulty={:.3}, mean_uncertainty={:.3}",
 self.last_difficulty.iter().sum::<f32>() / self.last_difficulty.len() as f32,
 self.last_uncertainty.iter().sum::<f32>() / self.last_uncertainty.len() as f32
 );

 Ok((difficulty, uncertainty))
 }

 /// CPU fallback: Compute metrics using simple heuristics
 ///
 /// Difficulty based on normalized vertex degree.
 /// Uncertainty based on local clustering coefficient.
 fn compute_cpu_metrics(&mut self, graph: &Graph) -> Result<(Vec<f32>, Vec<f32>), PrismError> {
 log::debug!(
 "Computing reservoir metrics on CPU (fallback) for {} vertices",
 graph.num_vertices
 );

 let num_vertices = graph.num_vertices;
 let adjacency = &graph.adjacency;

 let mut difficulty = Vec::with_capacity(num_vertices);
 let mut uncertainty = Vec::with_capacity(num_vertices);

 // Find max degree for normalization
 let max_degree = adjacency
 .iter()
 .map(|neighbors| neighbors.len())
 .max()
 .unwrap_or(1);

 for vertex in 0..num_vertices {
 let degree = adjacency[vertex].len();

 // Difficulty: Normalized degree (high degree = hard to color)
 let diff = if max_degree > 0 {
 degree as f32 / max_degree as f32
 } else {
 0.0
 };
 difficulty.push(diff);

 // Uncertainty: Based on neighbor degree variance
 // High variance = unpredictable structure = high uncertainty
 if degree > 0 {
 let neighbor_degrees: Vec<f32> = adjacency[vertex]
 .iter()
 .map(|&n| adjacency[n].len() as f32)
 .collect();

 let mean_neighbor_deg: f32 =
 neighbor_degrees.iter().sum::<f32>() / neighbor_degrees.len() as f32;
 let variance: f32 = neighbor_degrees
 .iter()
 .map(|&d| (d - mean_neighbor_deg).powi(2))
 .sum::<f32>()
 / neighbor_degrees.len() as f32;

 let std_dev = variance.sqrt();
 let uncert = (std_dev / max_degree as f32).min(1.0);
 uncertainty.push(uncert);
 } else {
 uncertainty.push(0.0);
 }
 }

 // Cache for telemetry
 self.last_difficulty = difficulty.clone();
 self.last_uncertainty = uncertainty.clone();

 log::debug!(
 "CPU metrics computed: mean_difficulty={:.3}, mean_uncertainty={:.3}",
 self.last_difficulty.iter().sum::<f32>() / self.last_difficulty.len() as f32,
 self.last_uncertainty.iter().sum::<f32>() / self.last_uncertainty.len() as f32
 );

 Ok((difficulty, uncertainty))
 }

 /// Returns whether GPU mode is enabled
 pub fn is_gpu_enabled(&self) -> bool {
 self.use_gpu
 }
}

impl PhaseController for Phase0DendriticReservoir {
 fn execute(
 &mut self,
 graph: &Graph,
 context: &mut PhaseContext,
 ) -> Result<PhaseOutcome, PrismError> {
 use std::time::Instant;

 log::info!("Phase 0: Dendritic Reservoir (GPU: {})", self.use_gpu);

 // Time execution for telemetry
 let start_time = Instant::now();

 // Compute reservoir metrics
 let (difficulty, uncertainty) = self.compute_reservoir_metrics(graph)?;

 // Record execution time
 self.last_execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

 // Store in context for orchestrator to use in warmstart
 context
 .scratch
 .insert("phase0_difficulty".to_string(), Box::new(difficulty));
 context
 .scratch
 .insert("phase0_uncertainty".to_string(), Box::new(uncertainty));

 // Build telemetry (Warmstart Plan Step 2)
 let telemetry = prism_core::Phase0Telemetry::from_metrics(
 &self.last_difficulty,
 &self.last_uncertainty,
 self.reservoir_iterations,
 self.convergence_loss,
 self.last_execution_time_ms,
 self.use_gpu,
 );

 // Store structured telemetry in context for orchestrator
 context
 .scratch
 .insert("phase0_telemetry".to_string(), Box::new(telemetry.clone()));

 // Cache telemetry for PhaseTelemetry trait
 self.telemetry = Some(telemetry);

 log::info!(
 "Phase 0 completed: metrics stored (exec_time={:.2}ms, iterations={}, gpu={})",
 self.last_execution_time_ms,
 self.reservoir_iterations,
 self.use_gpu
 );

 Ok(PhaseOutcome::success())
 }

 fn name(&self) -> &'static str {
 "Phase0-DendriticReservoir"
 }

 fn telemetry(&self) -> &dyn PhaseTelemetry {
 self
 }
}

impl PhaseTelemetry for Phase0DendriticReservoir {
 fn metrics(&self) -> HashMap<String, f64> {
 // Use structured telemetry if available (preferred path)
 if let Some(ref telemetry) = self.telemetry {
 return telemetry.to_hashmap();
 }

 // Fallback: Compute metrics from raw data (legacy path)
 let mut m = HashMap::new();

 if !self.last_difficulty.is_empty() {
 let mean_difficulty =
 self.last_difficulty.iter().sum::<f32>() / self.last_difficulty.len() as f32;
 let mean_uncertainty =
 self.last_uncertainty.iter().sum::<f32>() / self.last_uncertainty.len() as f32;

 m.insert("mean_difficulty".to_string(), mean_difficulty as f64);
 m.insert("mean_uncertainty".to_string(), mean_uncertainty as f64);

 // Entropy: Measure of uniformity in difficulty distribution
 let max_diff = self.last_difficulty.iter().cloned().fold(0.0f32, f32::max);
 if max_diff > 0.0 {
 let normalized: Vec<f32> =
 self.last_difficulty.iter().map(|d| d / max_diff).collect();
 let entropy: f32 = normalized
 .iter()
 .filter(|&&p| p > 0.0)
 .map(|&p| -p * p.ln())
 .sum();
 m.insert("difficulty_entropy".to_string(), entropy as f64);
 }

 // Sparsity: Fraction of low-uncertainty vertices
 let low_uncertainty_count = self.last_uncertainty.iter().filter(|&&u| u < 0.3).count();
 let sparsity = low_uncertainty_count as f64 / self.last_uncertainty.len() as f64;
 m.insert("uncertainty_sparsity".to_string(), sparsity);
 } else {
 // Default values if no metrics computed yet
 m.insert("mean_difficulty".to_string(), 0.5);
 m.insert("mean_uncertainty".to_string(), 0.5);
 }

 m.insert(
 "gpu_enabled".to_string(),
 if self.use_gpu { 1.0 } else { 0.0 },
 );

 m
 }
}
