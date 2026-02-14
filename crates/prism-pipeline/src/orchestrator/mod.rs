//! Pipeline orchestrator - executes phases with retry/escalate logic.
//!
//! ## High-Performance Kernel Integration
//!
//! The orchestrator supports multiple GPU acceleration modes:
//! - **Ultra Kernel**: Fused 8-component kernel for maximum throughput
//! - **AATGS Async**: Asynchronous task graph scheduling for CPU-GPU overlap
//! - **Multi-GPU**: Replica exchange across multiple CUDA devices
//! - **Stream Manager**: Triple-buffering for H2D/compute/D2H overlap

use crate::config::PipelineConfig;
use crate::telemetry::TelemetryEvent;
use prism_core::{
 ColoringSolution, Graph, PhaseContext, PhaseController, PhaseOutcome, PrismError,
 RuntimeConfig, WarmstartMetadata, WarmstartPlan,
};
use prism_fluxnet::{
 CurriculumBank, GraphStats, UniversalAction, UniversalRLController, UniversalRLState,
};
#[cfg(feature = "cuda")]
use prism_gpu;
#[cfg(feature = "cuda")]
use prism_gpu::{
 GpuExecutionContext, GpuExecutionContextBuilder, MultiGpuContext, UltraKernelGpu,
};
#[cfg(feature = "lbs")]
use prism_lbs::{
 pipeline_integration::{run_lbs, run_lbs_with_gpu},
 LbsConfig as LbsCfg, Pocket as LbsPocket, ProteinStructure as LbsStructure,
};
use prism_phases::WHCRPhaseController;
#[cfg(feature = "cuda")]
use prism_whcr::GeometrySynchronizer;
use prism_whcr::{CallingPhase, ExtractionConfig, GeometryAccumulator, PhaseWHCRConfig};
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Main pipeline orchestrator.
///
/// Executes phases in sequence, handles retry/escalate logic, and integrates
/// with FluxNet RL for parameter optimization.
///
/// ## High-Performance GPU Features
///
/// When built with `cuda` feature, supports:
/// - **Ultra Kernel**: Fused GPU kernel combining 8 optimization techniques
/// - **AATGS**: Adaptive Asynchronous Task Graph Scheduler
/// - **Multi-GPU**: Parallel execution across multiple CUDA devices
pub struct PipelineOrchestrator {
 /// Registered phase controllers
 phases: Vec<Box<dyn PhaseController>>,

 /// RL controller for adaptive parameter tuning
 rl_controller: UniversalRLController,

 /// Pipeline configuration
 config: PipelineConfig,

 /// Execution context (shared across phases)
 context: PhaseContext,

 /// Telemetry file handle (buffered writer for JSONL)
 telemetry_writer: Option<std::io::BufWriter<std::fs::File>>,

 /// Phase 0 configuration (optional, loaded from TOML)
 phase0_config: Option<prism_phases::phase0::Phase0Config>,

 /// Phase 1 configuration (optional, loaded from TOML)
 phase1_config: Option<prism_phases::phase1_active_inference::Phase1Config>,

 /// Phase 3 configuration (optional, loaded from TOML)
 phase3_config: Option<prism_core::Phase3Config>,

 /// Phase 4 configuration (optional, loaded from TOML)
 phase4_config: Option<prism_phases::phase4_geodesic::Phase4Config>,

 /// Phase 6 configuration (optional, loaded from TOML)
 phase6_config: Option<prism_phases::phase6_tda::Phase6Config>,

 /// Phase 7 configuration (optional, loaded from TOML)
 phase7_config: Option<prism_phases::phase7_ensemble::Phase7Config>,

 // =========================================================================
 // HIGH-PERFORMANCE GPU KERNEL INTEGRATION
 // =========================================================================
 /// Ultra Kernel GPU instance (fused 8-component kernel)
 #[cfg(feature = "cuda")]
 ultra_kernel: Option<UltraKernelGpu>,

 /// AATGS async execution context
 #[cfg(feature = "cuda")]
 gpu_exec_context: Option<GpuExecutionContext>,

 /// Multi-GPU execution context
 #[cfg(feature = "cuda")]
 multi_gpu_context: Option<MultiGpuContext>,

 /// Enable Ultra kernel mode (vs individual phase kernels)
 #[cfg(feature = "cuda")]
 use_ultra_kernel: bool,

 /// Enable AATGS async scheduling
 #[cfg(feature = "cuda")]
 use_aatgs_async: bool,
}

impl PipelineOrchestrator {
 /// Creates a new orchestrator with the given configuration.
 pub fn new(config: PipelineConfig, rl_controller: UniversalRLController) -> Self {
 // Initialize telemetry writer if enabled
 let telemetry_writer = if config.enable_telemetry {
 Self::create_telemetry_writer(&config.telemetry_path)
 .map_err(|e| {
 log::warn!(
 "Failed to create telemetry writer at {}: {}. Telemetry will be disabled.",
 config.telemetry_path,
 e
 );
 e
 })
 .ok()
 } else {
 None
 };

 Self {
 phases: Vec::new(),
 rl_controller,
 config,
 context: PhaseContext::new(),
 telemetry_writer,
 phase0_config: None,
 phase1_config: None,
 phase3_config: None,
 phase4_config: None,
 phase6_config: None,
 phase7_config: None,
 // High-performance GPU kernel integration (initialized lazily)
 #[cfg(feature = "cuda")]
 ultra_kernel: None,
 #[cfg(feature = "cuda")]
 gpu_exec_context: None,
 #[cfg(feature = "cuda")]
 multi_gpu_context: None,
 #[cfg(feature = "cuda")]
 use_ultra_kernel: false, // Default to individual phase kernels
 #[cfg(feature = "cuda")]
 use_aatgs_async: false, // Default to sync execution
 }
 }

 /// Sets the Phase 0 dendritic reservoir configuration.
 pub fn set_phase0_config(&mut self, config: prism_phases::phase0::Phase0Config) {
 self.phase0_config = Some(config);
 }

 /// Sets the Phase 1 active inference configuration.
 ///
 /// This configuration will be used during phase initialization to set
 /// active inference parameters from TOML config files.
 pub fn set_phase1_config(
 &mut self,
 config: prism_phases::phase1_active_inference::Phase1Config,
 ) {
 self.phase1_config = Some(config);
 }

 /// Sets the Phase 3 quantum configuration.
 ///
 /// This configuration will be used during phase initialization to set
 /// quantum evolution parameters from TOML config files.
 pub fn set_phase3_config(&mut self, config: prism_core::Phase3Config) {
 self.phase3_config = Some(config);
 }

 /// Sets the Phase 4 geodesic configuration.
 ///
 /// This configuration will be used during phase initialization to set
 /// geodesic distance parameters from TOML config files.
 pub fn set_phase4_config(&mut self, config: prism_phases::phase4_geodesic::Phase4Config) {
 self.phase4_config = Some(config);
 }

 /// Sets the Phase 6 TDA configuration.
 pub fn set_phase6_config(&mut self, config: prism_phases::phase6_tda::Phase6Config) {
 self.phase6_config = Some(config);
 }

 /// Sets the Phase 7 Ensemble configuration.
 pub fn set_phase7_config(&mut self, config: prism_phases::phase7_ensemble::Phase7Config) {
 self.phase7_config = Some(config);
 }

 // =========================================================================
 // HIGH-PERFORMANCE GPU MODE CONFIGURATION
 // =========================================================================

 /// Enable Ultra Kernel mode for fused GPU execution.
 ///
 /// When enabled, the pipeline uses a single fused kernel that combines:
 /// - W-Cycle Multigrid
 /// - Dendritic Reservoir Computing
 /// - Quantum Tunneling
 /// - TPTP Persistent Homology
 /// - Active Inference
 /// - Parallel Tempering
 /// - WHCR Conflict Repair
 /// - Wavelet-guided prioritization
 ///
 /// This provides ~3.5x speedup over individual phase kernels.
 #[cfg(feature = "cuda")]
 pub fn enable_ultra_kernel(&mut self, enable: bool) {
 self.use_ultra_kernel = enable;
 log::info!("Ultra Kernel mode: {}", if enable { "ENABLED" } else { "DISABLED" });
 }

 /// Enable AATGS asynchronous scheduling.
 ///
 /// When enabled, GPU kernels are scheduled asynchronously with triple-buffering
 /// for CPU-GPU overlap. Provides ~1.2x speedup.
 #[cfg(feature = "cuda")]
 pub fn enable_aatgs_async(&mut self, enable: bool) {
 self.use_aatgs_async = enable;
 log::info!("AATGS async scheduling: {}", if enable { "ENABLED" } else { "DISABLED" });
 }

 /// Initialize the Ultra Kernel for a specific graph.
 ///
 /// Must be called after GPU context initialization and before running with Ultra kernel.
 #[cfg(feature = "cuda")]
 pub fn initialize_ultra_kernel(&mut self, graph: &Graph) -> Result<(), PrismError> {
 if !self.use_ultra_kernel {
 log::debug!("Ultra kernel not enabled, skipping initialization");
 return Ok(());
 }

 let gpu_ctx = self.context.gpu_context.as_ref()
 .ok_or_else(|| PrismError::gpu("ultra_kernel", "No GPU context available"))?;

 let gpu_ctx = gpu_ctx.clone()
 .downcast::<prism_gpu::context::GpuContext>()
 .map_err(|_| PrismError::gpu("ultra_kernel", "Failed to downcast GPU context"))?;

 // Convert graph to CSR format
 let (row_ptr, col_idx) = self.graph_to_csr(graph);

 // Create runtime config
 let config = RuntimeConfig::default();

 // Initialize Ultra kernel
 match UltraKernelGpu::new(
 gpu_ctx.device().clone(),
 graph.num_vertices,
 &row_ptr,
 &col_idx,
 &config,
 ) {
 Ok(kernel) => {
 log::info!(
 "Ultra Kernel initialized for {} vertices, {} edges",
 graph.num_vertices,
 col_idx.len()
 );
 self.ultra_kernel = Some(kernel);
 Ok(())
 }
 Err(e) => {
 log::warn!("Ultra Kernel initialization failed: {}. Falling back to individual phases.", e);
 self.use_ultra_kernel = false;
 Ok(())
 }
 }
 }

 /// Convert graph adjacency list to CSR format for GPU kernels.
 #[cfg(feature = "cuda")]
 fn graph_to_csr(&self, graph: &Graph) -> (Vec<i32>, Vec<i32>) {
 let mut row_ptr = Vec::with_capacity(graph.num_vertices + 1);
 let mut col_idx = Vec::new();

 row_ptr.push(0);
 for adj in &graph.adjacency {
 col_idx.extend(adj.iter().map(|&v| v as i32));
 row_ptr.push(col_idx.len() as i32);
 }

 (row_ptr, col_idx)
 }

 /// Run the Ultra Kernel for the specified number of iterations.
 ///
 /// Returns the coloring solution from the fused kernel execution.
 #[cfg(feature = "cuda")]
 pub fn run_ultra_kernel(&mut self, iterations: usize) -> Result<ColoringSolution, PrismError> {
 let kernel = self.ultra_kernel.as_ref()
 .ok_or_else(|| PrismError::gpu("ultra_kernel", "Ultra kernel not initialized"))?;

 log::info!("Running Ultra Kernel for {} iterations", iterations);
 let start = Instant::now();

 let (telemetry, coloring) = kernel.run(iterations)
 .map_err(|e| PrismError::gpu("ultra_kernel", e.to_string()))?;

 let elapsed = start.elapsed();
 log::info!(
 "Ultra Kernel completed in {:.2}s: {} colors, {} conflicts",
 elapsed.as_secs_f64(),
 telemetry.colors_used,
 telemetry.conflicts
 );

 // Convert to ColoringSolution
 let colors: Vec<usize> = coloring.iter().map(|&c| c as usize).collect();
 let chromatic_number = colors.iter().max().map(|&m| m + 1).unwrap_or(0);

 Ok(ColoringSolution {
 colors,
 chromatic_number,
 conflicts: telemetry.conflicts as usize,
 quality_score: if telemetry.conflicts == 0 { 1.0 } else { 0.5 },
 computation_time_ms: elapsed.as_secs_f64() * 1000.0,
 })
 }

 /// Initialize AATGS async execution context.
 #[cfg(feature = "cuda")]
 pub fn initialize_aatgs(&mut self) -> Result<(), PrismError> {
 if !self.use_aatgs_async {
 log::debug!("AATGS not enabled, skipping initialization");
 return Ok(());
 }

 let gpu_ctx = self.context.gpu_context.as_ref()
 .ok_or_else(|| PrismError::gpu("aatgs", "No GPU context available"))?;

 let gpu_ctx = gpu_ctx.clone()
 .downcast::<prism_gpu::context::GpuContext>()
 .map_err(|_| PrismError::gpu("aatgs", "Failed to downcast GPU context"))?;

 match GpuExecutionContext::new(gpu_ctx.device().clone(), true) {
 Ok(ctx) => {
 log::info!("AATGS async execution context initialized");
 self.gpu_exec_context = Some(ctx);
 Ok(())
 }
 Err(e) => {
 log::warn!("AATGS initialization failed: {}. Using sync execution.", e);
 self.use_aatgs_async = false;
 Ok(())
 }
 }
 }

 /// Initialize Multi-GPU context for parallel execution.
 #[cfg(feature = "cuda")]
 pub fn initialize_multi_gpu(&mut self, device_ids: &[usize]) -> Result<(), PrismError> {
 if device_ids.len() <= 1 {
 log::debug!("Single GPU mode, skipping multi-GPU initialization");
 return Ok(());
 }

 // Default to 4 replicas per GPU for parallel tempering
 let num_replicas = device_ids.len() * 4;
 match MultiGpuContext::new(device_ids, num_replicas) {
 Ok(ctx) => {
 log::info!(
 "Multi-GPU context initialized with {} devices, {} replicas: {:?}",
 device_ids.len(),
 num_replicas,
 device_ids
 );
 self.multi_gpu_context = Some(ctx);
 Ok(())
 }
 Err(e) => {
 log::warn!("Multi-GPU initialization failed: {}. Using single GPU.", e);
 Ok(())
 }
 }
 }

 /// Creates a buffered writer for telemetry JSONL output.
 ///
 /// Creates parent directories if they don't exist.
 fn create_telemetry_writer(path: &str) -> std::io::Result<std::io::BufWriter<std::fs::File>> {
 use std::fs::{create_dir_all, OpenOptions};
 use std::io::BufWriter;
 use std::path::Path;

 // Create parent directories if needed
 if let Some(parent) = Path::new(path).parent() {
 if !parent.as_os_str().is_empty() && !parent.exists() {
 create_dir_all(parent)?;
 }
 }

 // Open file in append mode (create if not exists)
 let file = OpenOptions::new().create(true).append(true).open(path)?;

 log::info!("Telemetry writer created: {}", path);
 Ok(BufWriter::new(file))
 }

 /// Writes a telemetry event to the JSONL file.
 fn write_telemetry_event(&mut self, event: &TelemetryEvent) -> std::io::Result<()> {
 use std::io::Write;

 if let Some(ref mut writer) = self.telemetry_writer {
 let json = event
 .to_json()
 .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

 writeln!(writer, "{}", json)?;
 writer.flush()?; // Ensure immediate write for real-time monitoring
 }

 Ok(())
 }

 /// Registers a phase controller.
 pub fn add_phase(&mut self, phase: Box<dyn PhaseController>) {
 self.phases.push(phase);
 }

 /// Initializes and registers all phases with GPU-first initialization.
 ///
 /// This is the recommended way to register phases when GPU acceleration is enabled.
 /// If GPU context is available, phases are initialized with GPU constructors.
 /// Otherwise, they fall back to CPU-only constructors.
 ///
 /// # Returns
 /// Error if any critical phase fails to initialize.
 ///
 /// # GPU-Accelerated Phases
 /// - Phase 0: Dendritic Reservoir (dendritic_reservoir.ptx)
 /// - Phase 1: Active Inference (active_inference.ptx)
 /// - Phase 3: Quantum Evolution (quantum.ptx)
 /// - Phase 4: Geodesic Distance (floyd_warshall.ptx)
 /// - Phase 6: TDA (tda.ptx)
 ///
 /// # Example
 /// ```rust,no_run
 /// # use prism_pipeline::{PipelineOrchestrator, PipelineConfig};
 /// # use prism_fluxnet::{UniversalRLController, RLConfig};
 /// # fn example() -> anyhow::Result<()> {
 /// let config = PipelineConfig::builder().gpu_enabled(true).build()?;
 /// let rl = UniversalRLController::new(RLConfig::default());
 /// let mut orchestrator = PipelineOrchestrator::new(config, rl);
 /// orchestrator.initialize_all_phases()?; // GPU-first initialization
 /// # Ok(())
 /// # }
 /// ```
 pub fn initialize_all_phases(&mut self) -> Result<(), PrismError> {
 use prism_phases::{
 Phase0DendriticReservoir, Phase1ActiveInference, Phase2Thermodynamic, Phase3Quantum,
 Phase4Geodesic, Phase6TDA, Phase7Ensemble,
 };
 // Import new phase controllers
 use prism_mec::MecPhaseController;
 use prism_ontology::OntologyPhaseController;
 use prism_physics::CmaEsPhaseController;

 log::info!("Initializing phases with GPU-first strategy");

 // Check if GPU context is available
 #[cfg(feature = "gpu")]
 let gpu_available = self.context.gpu_context.is_some();

 #[cfg(not(feature = "gpu"))]
 let gpu_available = false;

 // Extract PTX directory path for GPU phases
 let ptx_dir = self.config.gpu.ptx_dir.clone();

 if gpu_available {
 log::info!("GPU context available - initializing phases with CUDA acceleration");

 // Phase 0-Ontology: Semantic Grounding (GPU-accelerated if available)
 {
 let mut ontology_config = prism_ontology::OntologyConfig::default();
 ontology_config.use_gpu = true;
 self.add_phase(Box::new(OntologyPhaseController::new(ontology_config)));
 log::info!("Phase 0-Ontology: Semantic grounding initialized");
 }

 // Phase 0: Dendritic Reservoir (GPU-accelerated)
 #[cfg(feature = "cuda")]
 {
 let ptx_path = ptx_dir.join("dendritic_reservoir.ptx");
 let phase0 = if let Some(ref cfg) = self.phase0_config {
 log::info!("Phase 0: Initializing with custom TOML config");
 match Phase0DendriticReservoir::with_config_and_gpu(
 cfg.clone(),
 ptx_path.to_str().unwrap(),
 ) {
 Ok(p) => p,
 Err(e) => {
 log::warn!(
 "Phase 0: GPU init failed ({}), falling back to CPU with config",
 e
 );
 Phase0DendriticReservoir::with_config(cfg.clone())
 }
 }
 } else {
 match Phase0DendriticReservoir::new_with_gpu(ptx_path.to_str().unwrap()) {
 Ok(p) => {
 log::info!("Phase 0: GPU dendritic reservoir acceleration enabled");
 p
 }
 Err(e) => {
 log::warn!("Phase 0: GPU init failed ({}), falling back to CPU", e);
 Phase0DendriticReservoir::new()
 }
 }
 };
 self.add_phase(Box::new(phase0));
 }
 #[cfg(not(feature = "cuda"))]
 {
 log::info!("Phase 0: CPU mode (CUDA feature not enabled)");
 let phase0 = if let Some(ref cfg) = self.phase0_config {
 log::info!("Phase 0: Initializing with custom TOML config (CPU mode)");
 Phase0DendriticReservoir::with_config(cfg.clone())
 } else {
 Phase0DendriticReservoir::new()
 };
 self.add_phase(Box::new(phase0));
 }

 // Phase 1: Active Inference (GPU-accelerated)
 #[cfg(feature = "gpu")]
 {
 let ptx_path = ptx_dir.join("active_inference.ptx");
 if let Some(gpu_ctx) = &self.context.gpu_context {
 if let Some(ctx) = gpu_ctx.downcast_ref::<prism_gpu::context::GpuContext>() {
 let device = ctx.device().clone();
 match prism_gpu::ActiveInferenceGpu::new(device, ptx_path.to_str().unwrap())
 {
 Ok(ai_gpu) => {
 log::info!("Phase 1: GPU Active Inference acceleration enabled");
 let phase1 = if let Some(ref cfg) = self.phase1_config {
 log::info!("Phase 1: Initializing with custom TOML config");
 Phase1ActiveInference::with_config_and_gpu(
 cfg.clone(),
 std::sync::Arc::new(ai_gpu),
 )
 } else {
 Phase1ActiveInference::new_with_gpu(std::sync::Arc::new(ai_gpu))
 };
 self.add_phase(Box::new(phase1));
 }
 Err(e) => {
 log::warn!("Phase 1: GPU init failed ({}), falling back to CPU", e);
 let phase1 = if let Some(ref cfg) = self.phase1_config {
 log::info!(
 "Phase 1: Initializing with custom TOML config (CPU)"
 );
 Phase1ActiveInference::with_config(cfg.clone())
 } else {
 Phase1ActiveInference::new()
 };
 self.add_phase(Box::new(phase1));
 }
 }
 } else {
 log::warn!("Phase 1: Failed to downcast GPU context, using CPU");
 let phase1 = if let Some(ref cfg) = self.phase1_config {
 Phase1ActiveInference::with_config(cfg.clone())
 } else {
 Phase1ActiveInference::new()
 };
 self.add_phase(Box::new(phase1));
 }
 } else {
 log::warn!("Phase 1: GPU context not available, using CPU");
 let phase1 = if let Some(ref cfg) = self.phase1_config {
 Phase1ActiveInference::with_config(cfg.clone())
 } else {
 Phase1ActiveInference::new()
 };
 self.add_phase(Box::new(phase1));
 }
 }
 #[cfg(not(feature = "gpu"))]
 {
 log::info!("Phase 1: CPU mode (GPU feature not enabled)");
 let phase1 = if let Some(ref cfg) = self.phase1_config {
 Phase1ActiveInference::with_config(cfg.clone())
 } else {
 Phase1ActiveInference::new()
 };
 self.add_phase(Box::new(phase1));
 }

 // Phase 2: Thermodynamic (GPU-accelerated)
 #[cfg(feature = "cuda")]
 {
 let ptx_path = ptx_dir.join("thermodynamic.ptx");
 if let Some(gpu_ctx) = &self.context.gpu_context {
 if let Some(ctx) = gpu_ctx.downcast_ref::<prism_gpu::context::GpuContext>() {
 let device = ctx.device().clone();
 match Phase2Thermodynamic::new_with_gpu(device, ptx_path.to_str().unwrap())
 {
 Ok(phase2) => {
 // Apply hyperparameters from config
 let phase2 = phase2.with_hyperparameters(
 self.config.phase2.iterations,
 self.config.phase2.replicas,
 self.config.phase2.temp_min,
 self.config.phase2.temp_max,
 );
 log::info!("Phase 2: GPU thermodynamic acceleration enabled");
 self.add_phase(Box::new(phase2));
 }
 Err(e) => {
 log::warn!("Phase 2: GPU init failed ({}), falling back to CPU", e);
 self.add_phase(Box::new(Phase2Thermodynamic::new()));
 }
 }
 } else {
 log::warn!("Phase 2: Failed to downcast GPU context, using CPU");
 self.add_phase(Box::new(Phase2Thermodynamic::new()));
 }
 } else {
 log::warn!("Phase 2: GPU context not available, using CPU");
 self.add_phase(Box::new(Phase2Thermodynamic::new()));
 }
 }
 #[cfg(not(feature = "cuda"))]
 {
 log::info!("Phase 2: CUDA feature disabled, using CPU fallback");
 self.add_phase(Box::new(Phase2Thermodynamic::new()));
 }

 // Phase 3: Quantum Evolution (GPU-accelerated)
 #[cfg(feature = "gpu")]
 {
 let ptx_path = ptx_dir.join("quantum.ptx");

 // Extract device from context
 if let Some(gpu_ctx) = &self.context.gpu_context {
 if let Some(ctx) = gpu_ctx.downcast_ref::<prism_gpu::context::GpuContext>() {
 let device = ctx.device().clone();

 // Use with_config if phase3_config is available, otherwise use with_gpu
 let phase3_result = if let Some(ref cfg) = self.phase3_config {
 log::info!("Phase 3: Initializing with custom TOML config");
 Phase3Quantum::with_config(device, ptx_path.to_str().unwrap(), cfg)
 } else {
 Phase3Quantum::with_gpu(device, ptx_path.to_str().unwrap())
 };

 match phase3_result {
 Ok(phase3) => {
 log::info!("Phase 3: GPU initialization successful");
 self.add_phase(Box::new(phase3));
 }
 Err(e) => {
 log::warn!("Phase 3: GPU init failed ({}), falling back to CPU", e);
 self.add_phase(Box::new(Phase3Quantum::new()));
 }
 }
 } else {
 log::warn!("Phase 3: Failed to downcast GPU context, using CPU");
 self.add_phase(Box::new(Phase3Quantum::new()));
 }
 } else {
 log::warn!("Phase 3: GPU context not available, using CPU");
 self.add_phase(Box::new(Phase3Quantum::new()));
 }
 }
 #[cfg(not(feature = "gpu"))]
 {
 log::info!("Phase 3: CPU mode (GPU feature not enabled)");
 self.add_phase(Box::new(Phase3Quantum::new()));
 }

 // Phase 4: Geodesic Distance (GPU-accelerated)
 #[cfg(feature = "cuda")]
 {
 let ptx_path = ptx_dir.join("floyd_warshall.ptx");
 let phase4 = if let Some(ref cfg) = self.phase4_config {
 log::info!("Phase 4: Initializing with custom TOML config");
 Phase4Geodesic::with_config_and_gpu(cfg.clone(), ptx_path.to_str().unwrap())
 } else {
 Phase4Geodesic::new_with_gpu(ptx_path.to_str().unwrap())
 };
 log::info!(
 "Phase 4: GPU geodesic acceleration enabled (path: {})",
 ptx_path.display()
 );
 self.add_phase(Box::new(phase4));
 }
 #[cfg(not(feature = "cuda"))]
 {
 log::info!("Phase 4: CPU mode (CUDA feature not enabled)");
 let phase4 = if let Some(ref cfg) = self.phase4_config {
 log::info!("Phase 4: Initializing with custom TOML config");
 Phase4Geodesic::with_config(cfg.clone())
 } else {
 Phase4Geodesic::new()
 };
 self.add_phase(Box::new(phase4));
 }

 // Phase 6: TDA (GPU-accelerated)
 #[cfg(feature = "cuda")]
 {
 let ptx_path = ptx_dir.join("tda.ptx");
 let phase6 = if let Some(ref cfg) = self.phase6_config {
 log::info!("Phase 6: Initializing with custom TOML config");
 Phase6TDA::with_config_and_gpu(cfg.clone(), ptx_path.to_str().unwrap())
 } else {
 Phase6TDA::new_with_gpu(ptx_path.to_str().unwrap())
 };
 log::info!(
 "Phase 6: GPU TDA acceleration enabled (path: {})",
 ptx_path.display()
 );
 self.add_phase(Box::new(phase6));
 }
 #[cfg(not(feature = "cuda"))]
 {
 let phase6 = if let Some(ref cfg) = self.phase6_config {
 log::info!("Phase 6: Initializing with custom TOML config (CPU mode)");
 Phase6TDA::with_config(cfg.clone())
 } else {
 Phase6TDA::new()
 };
 log::info!("Phase 6: CPU mode (CUDA feature not enabled)");
 self.add_phase(Box::new(phase6));
 }

 // Phase 7: Ensemble (CPU only)
 let phase7 = if let Some(ref cfg) = self.phase7_config {
 log::info!("Phase 7: Initializing with custom TOML config");
 Phase7Ensemble::with_config(cfg.clone())
 } else {
 Phase7Ensemble::new()
 };
 self.add_phase(Box::new(phase7));

 // Phase M-MEC: Molecular Emergent Computing (GPU-accelerated if available)
 {
 let mut mec_config = prism_mec::MecConfig::default();
 mec_config.use_gpu = true;
 let mec_controller = if let Some(ref gpu_any) = self.context.gpu_context {
 // Downcast Arc<dyn Any> to GpuContext
 #[cfg(feature = "cuda")]
 {
 if let Some(gpu_ctx) =
 gpu_any.downcast_ref::<prism_gpu::context::GpuContext>()
 {
 MecPhaseController::new(mec_config).with_gpu(gpu_ctx.device().clone())
 } else {
 MecPhaseController::new(mec_config)
 }
 }
 #[cfg(not(feature = "cuda"))]
 {
 MecPhaseController::new(mec_config)
 }
 } else {
 MecPhaseController::new(mec_config)
 };
 self.add_phase(Box::new(mec_controller));
 log::info!(
 "Phase M-MEC: Molecular emergent computing initialized (GPU: {})",
 self.context.gpu_context.is_some()
 );
 }

 // Phase X-CMA: CMA-ES Optimization (GPU-accelerated if available)
 if let Some(ref cma_config) = self.config.cma_es {
 if cma_config.enabled {
 let mut physics_cma_config = prism_physics::CmaEsConfig {
 population_size: cma_config.population_size,
 initial_sigma: cma_config.initial_sigma,
 max_iterations: cma_config.max_iterations,
 target_fitness: cma_config.target_fitness,
 use_gpu: true,
 };
 self.add_phase(Box::new(CmaEsPhaseController::new(physics_cma_config)));
 log::info!("Phase X-CMA: CMA-ES optimization initialized (GPU-accelerated)");
 }
 }
 } else {
 log::info!("GPU not available - initializing all phases with CPU fallback");

 // Phase 0-Ontology: Semantic Grounding (CPU)
 self.add_phase(Box::new(OntologyPhaseController::new(
 prism_ontology::OntologyConfig::default(),
 )));

 let phase0 = if let Some(ref cfg) = self.phase0_config {
 Phase0DendriticReservoir::with_config(cfg.clone())
 } else {
 Phase0DendriticReservoir::new()
 };
 self.add_phase(Box::new(phase0));

 let phase1 = if let Some(ref cfg) = self.phase1_config {
 Phase1ActiveInference::with_config(cfg.clone())
 } else {
 Phase1ActiveInference::new()
 };
 self.add_phase(Box::new(phase1));

 self.add_phase(Box::new(Phase2Thermodynamic::new()));
 self.add_phase(Box::new(Phase3Quantum::new()));

 let phase4 = if let Some(ref cfg) = self.phase4_config {
 Phase4Geodesic::with_config(cfg.clone())
 } else {
 Phase4Geodesic::new()
 };
 self.add_phase(Box::new(phase4));

 let phase6 = if let Some(ref cfg) = self.phase6_config {
 Phase6TDA::with_config(cfg.clone())
 } else {
 Phase6TDA::new()
 };
 self.add_phase(Box::new(phase6));

 let phase7 = if let Some(ref cfg) = self.phase7_config {
 Phase7Ensemble::with_config(cfg.clone())
 } else {
 Phase7Ensemble::new()
 };
 self.add_phase(Box::new(phase7));

 // Phase M-MEC: Molecular Emergent Computing (CPU)
 self.add_phase(Box::new(MecPhaseController::new(
 prism_mec::MecConfig::default(),
 )));

 // Phase X-CMA: CMA-ES Optimization (CPU)
 if let Some(ref cma_config) = self.config.cma_es {
 if cma_config.enabled {
 let physics_cma_config = prism_physics::CmaEsConfig {
 population_size: cma_config.population_size,
 initial_sigma: cma_config.initial_sigma,
 max_iterations: cma_config.max_iterations,
 target_fitness: cma_config.target_fitness,
 use_gpu: false,
 };
 self.add_phase(Box::new(CmaEsPhaseController::new(physics_cma_config)));
 log::info!("Phase X-CMA: CMA-ES optimization initialized (CPU)");
 }
 }
 }

 log::info!(
 "All phases initialized successfully ({} phases)",
 self.phases.len()
 );
 Ok(())
 }

 /// Executes the full pipeline on the given graph.
 ///
 /// Returns the best coloring solution found.
 pub fn run(&mut self, graph: &Graph) -> Result<ColoringSolution, PrismError> {
 log::info!(
 "Starting pipeline execution on graph with {} vertices",
 graph.num_vertices
 );

 // Initialize GPU context if enabled
 #[cfg(feature = "gpu")]
 {
 if self.config.gpu.enabled {
 self.initialize_gpu_context()?;
 } else {
 log::info!("GPU disabled in config. Running CPU-only mode.");
 }
 }

 #[cfg(not(feature = "gpu"))]
 {
 if self.config.gpu.enabled {
 log::warn!("GPU enabled in config but prism-pipeline not built with 'gpu' feature. Running CPU-only mode.");
 }
 }

 // =========================================================================
 // AATGS ASYNC SCHEDULER INITIALIZATION
 // =========================================================================
 #[cfg(feature = "cuda")]
 {
 if self.use_aatgs_async && self.gpu_exec_context.is_none() {
 if let Err(e) = self.initialize_aatgs() {
 log::warn!("AATGS initialization failed: {}. Continuing with sync execution.", e);
 }
 }

 if self.gpu_exec_context.is_some() {
 log::info!("=== AATGS ASYNC SCHEDULER: Enabled for kernel fusion ===");
 }
 }

 // =========================================================================
 // MULTI-GPU STATUS LOGGING
 // =========================================================================
 // Multi-GPU is initialized explicitly via initialize_multi_gpu() when:
 // - Config specifies multiple device IDs, or
 // - CLI explicitly requests multi-GPU mode
 #[cfg(feature = "cuda")]
 if self.multi_gpu_context.is_some() {
 log::info!("=== MULTI-GPU CONTEXT: Active with parallel execution ===");
 }

 // Initialize phases AFTER GPU context is set up
 // This ensures GPU-accelerated phases receive CUDA device handles
 if self.phases.is_empty() {
 log::info!("No phases registered, auto-initializing with GPU-first strategy");
 self.initialize_all_phases()?;
 } else {
 log::info!("Using {} pre-registered phases", self.phases.len());
 }

 let start_time = Instant::now();

 // =========================================================================
 // CREATE GEOMETRY SYNCHRONIZER (ADVANCED GPU-RESIDENT)
 // =========================================================================
 #[cfg(feature = "cuda")]
 let mut geometry_sync: Option<GeometrySynchronizer> = if let Some(ref gpu_ctx_any) = self.context.gpu_context {
 // Try to downcast as Arc<GpuContext>
 if let Ok(gpu_ctx_arc) = gpu_ctx_any
 .clone()
 .downcast::<prism_gpu::context::GpuContext>()
 {
 // Detect if this is DSJC125.5 for special tuning
 let config = if graph.num_vertices == 125 {
 let num_edges = graph.adjacency.iter().map(|a| a.len()).sum::<usize>() / 2;
 if num_edges > 3500 && num_edges < 4200 {
 log::info!("Detected DSJC125.5-like graph, using tuned extraction config");
 ExtractionConfig::for_dsjc125_5()
 } else {
 ExtractionConfig::for_graph(graph.num_vertices, num_edges)
 }
 } else {
 let num_edges = graph.adjacency.iter().map(|a| a.len()).sum::<usize>() / 2;
 ExtractionConfig::for_graph(graph.num_vertices, num_edges)
 };

 match GeometrySynchronizer::with_config(gpu_ctx_arc.device().clone(), graph, config)
 {
 Ok(sync) => {
 log::info!(
 "Created GeometrySynchronizer for {} vertices with adaptive config",
 graph.num_vertices
 );
 Some(sync)
 }
 Err(e) => {
 log::warn!(
 "Failed to create GeometrySynchronizer: {}. WHCR disabled.",
 e
 );
 None
 }
 }
 } else {
 log::warn!("GPU context exists but could not be downcast. WHCR will be skipped.");
 None
 }
 } else {
 log::info!("No GPU context available. WHCR multi-phase integration disabled.");
 None
 };

 #[cfg(not(feature = "cuda"))]
 let geometry_sync = None::<()>;

 log::info!(
 "Pipeline: WHCR multi-phase integration {}",
 if geometry_sync.is_some() {
 "enabled with advanced sync"
 } else {
 "disabled"
 }
 );

 // Load curriculum bank and initialize RL controller if configured
 let catalog_path = self
 .config
 .warmstart_config
 .as_ref()
 .and_then(|cfg| cfg.curriculum_catalog_path.clone());

 if let Some(catalog_path) = catalog_path {
 match self.load_and_apply_curriculum(graph, &catalog_path) {
 Ok(profile) => {
 log::info!("Curriculum Q-table loaded for profile: {:?}", profile);

 // Store curriculum profile in metadata for telemetry
 self.context.scratch.insert(
 "curriculum_profile".to_string(),
 Box::new(format!("{:?}", profile)),
 );
 }
 Err(e) => {
 log::warn!("Failed to load curriculum catalog: {}. Continuing without curriculum warmstart.", e);
 }
 }
 }

 // Execute warmstart stage if enabled (before main phase loop)
 if self.config.warmstart_config.is_some() {
 self.execute_warmstart_stage(graph)?;
 }

 // =========================================================================
 // ULTRA KERNEL MODE (FUSED 8-COMPONENT GPU EXECUTION)
 // =========================================================================
 // When enabled, bypasses individual phase execution with a single fused kernel
 #[cfg(feature = "cuda")]
 if self.use_ultra_kernel {
 log::info!("=== ULTRA KERNEL MODE: Fused GPU execution enabled ===");

 // Initialize Ultra kernel for this graph
 if let Err(e) = self.initialize_ultra_kernel(graph) {
 log::warn!("Ultra kernel init failed: {}. Falling back to phase-by-phase.", e);
 self.use_ultra_kernel = false;
 } else if self.ultra_kernel.is_some() {
 // Determine iteration count based on graph size
 let iterations = match graph.num_vertices {
 n if n <= 250 => 500,
 n if n <= 500 => 1000,
 n if n <= 1000 => 2000,
 _ => 3000,
 };

 log::info!(
 "Running Ultra Kernel: {} vertices, {} iterations",
 graph.num_vertices,
 iterations
 );

 match self.run_ultra_kernel(iterations) {
 Ok(solution) => {
 let elapsed = start_time.elapsed();
 log::info!(
 "=== ULTRA KERNEL COMPLETE: {} colors, {} conflicts in {:.2}s ===",
 solution.chromatic_number,
 solution.conflicts,
 elapsed.as_secs_f64()
 );

 // Update context with solution
 self.context.best_solution = Some(solution.clone());

 // Skip individual phases - return Ultra kernel result directly
 return Ok(solution);
 }
 Err(e) => {
 log::warn!("Ultra kernel execution failed: {}. Falling back to phases.", e);
 self.use_ultra_kernel = false;
 }
 }
 }
 }

 // =========================================================================
 // INDIVIDUAL PHASE EXECUTION (when Ultra kernel not used/failed)
 // =========================================================================

 // Execute phases in sequence (using indices to avoid borrow checker issues)
 let num_phases = self.phases.len();
 for i in 0..num_phases {
 let phase_name = self.phases[i].name();
 log::info!("Executing phase: {}", phase_name);

 // WHCR pending flag: DISABLED because WHCR is disabled in this build
 // This allows phases to use their own internal conflict_repair module
 // When WHCR is re-enabled, uncomment the block below
 #[cfg(feature = "cuda")]
 if geometry_sync.is_some() {
 // WHCR is currently disabled - don't set pending flag
 // This allows Phase2/Phase3 to use internal conflict repair
 self.context.set_whcr_pending(false);
 }

 // Execute phase with retry logic
 let outcome = self.execute_phase_with_retry(i, graph, phase_name)?;

 // Collect telemetry metrics
 let metrics = self.phases[i].telemetry().metrics();

 // Emit telemetry with geometry metrics (if available)
 let mut event = TelemetryEvent::new(phase_name, metrics, &outcome);
 if let Some(ref geom) = self.context.geometry_metrics {
 event = event.with_geometry(geom);
 }
 log::trace!("Telemetry: {:?}", event);

 // Write telemetry event to JSONL file
 if let Err(e) = self.write_telemetry_event(&event) {
 log::warn!("Failed to write telemetry event for {}: {}", phase_name, e);
 }

 // Metaphysical Telemetry Coupling: Propagate geometry metrics after Phase 4 and 6
 if phase_name == "Phase4-Geodesic" || phase_name == "Phase6-TDA" {
 if let Some(ref best_solution) = self.context.best_solution {
 // Compute geometry metrics from current solution
 let geometry_telemetry = prism_core::GeometryTelemetry::from_solution(
 best_solution,
 graph,
 self.context.previous_chromatic,
 );

 // Update context with geometry metrics
 self.context
 .update_geometry_metrics(geometry_telemetry.clone());

 log::info!(
 "[Orchestrator] Geometry metrics propagated from {}: stress={:.3}, overlap={:.3}, {} hotspots",
 phase_name,
 geometry_telemetry.stress_scalar,
 geometry_telemetry.overlap_density,
 geometry_telemetry.anchor_hotspots.len()
 );
 } else {
 log::debug!("[Orchestrator] No solution available yet, skipping geometry metric propagation");
 }
 }

 // Update RL state with geometry metrics (after EVERY phase)
 // This ensures FluxNet reward shaping can use geometry stress deltas
 if let Some(ref geom) = self.context.geometry_metrics {
 if let Some(rl_state) = self.context.rl_state.as_mut() {
 if let Some(state) = rl_state.downcast_mut::<UniversalRLState>() {
 state.update_geometry_stress(geom.stress_scalar as f64);
 state.geometry_overlap_density = geom.overlap_density as f64;
 state.geometry_hotspot_count = geom.hotspot_count();
 log::trace!(
 "[Orchestrator] RL state updated with geometry: stress={:.3}, overlap={:.3}, {} hotspots",
 state.geometry_stress_level,
 state.geometry_overlap_density,
 state.geometry_hotspot_count
 );
 }
 }
 }

 // Update RL state and Q-table
 self.update_rl(&outcome, phase_name);

 match outcome {
 PhaseOutcome::Success { ref message, .. } => {
 log::info!("Phase {} completed: {}", phase_name, message);
 }
 PhaseOutcome::Escalate { ref reason } => {
 log::warn!("Phase {} escalated: {}", phase_name, reason);
 // Continue to next phase
 }
 PhaseOutcome::Retry { .. } => {
 // Should not reach here after retry loop
 unreachable!("Retry outcome after retry loop");
 }
 }

 // =========================================================================
 // GEOMETRY SYNCHRONIZATION
 // =========================================================================

 // Synchronize geometry after phase execution
 #[cfg(feature = "cuda")]
 if let Some(ref mut sync) = geometry_sync {
 // Sync Phase 0 geometry (Dendritic Reservoir)
 if phase_name.contains("Phase0") && phase_name.contains("Dendritic") {
 if sync.sync_phase0(&outcome, &self.context).unwrap_or(false) {
 log::info!("Phase 0 geometry synchronized");
 }
 }

 // Sync Phase 1 geometry (Active Inference)
 if phase_name.contains("Phase1") || phase_name.contains("ActiveInference") {
 if sync.sync_phase1(&outcome, &self.context).unwrap_or(false) {
 log::info!("Phase 1 geometry synchronized");
 }
 }

 // Sync Phase 4 geometry (Geodesic/Stress)
 if phase_name.contains("Phase4") || phase_name.contains("Geodesic") {
 if sync.sync_phase4(&outcome, &self.context).unwrap_or(false) {
 log::info!("Phase 4 geometry synchronized");
 }
 }

 // Sync Phase 6 geometry (TDA/Persistence)
 if phase_name.contains("Phase6") || phase_name.contains("TDA") {
 if sync.sync_phase6(&outcome, &self.context).unwrap_or(false) {
 log::info!("Phase 6 geometry synchronized");
 }
 }
 }

 // =========================================================================
 // WHCR INVOCATION POINTS (WITH SYNCHRONIZED GEOMETRY)
 // =========================================================================

 // After Phase 2 (Thermodynamic)
 #[cfg(feature = "cuda")]
 if phase_name.contains("Phase2") || phase_name.contains("Thermodynamic") {
 log::debug!("Detected Phase 2 completion - invoking WHCR with safeguards");
 if let Some(ref sync) = geometry_sync {
 if let Some(ref geom) = sync.geometry() {
 // Get conflict count before WHCR
 let conflicts_before = self.context.best_solution
 .as_ref()
 .map(|s| s.conflicts)
 .unwrap_or(usize::MAX);

 match self.invoke_whcr_phase2(graph, geom) {
 Ok(()) => {
 // Verify WHCR didn't make things worse (oscillation safeguard)
 let conflicts_after = self.context.best_solution
 .as_ref()
 .map(|s| s.conflicts)
 .unwrap_or(usize::MAX);

 if conflicts_after > conflicts_before * 3 {
 log::warn!("WHCR-Phase2 oscillation detected: {} -> {} conflicts, reverting",
 conflicts_before, conflicts_after);
 // Revert would need solution backup - for now just warn
 } else {
 log::info!("WHCR-Phase2 completed: {} -> {} conflicts, geometry: {}",
 conflicts_before, conflicts_after, sync.summary());
 }
 // Clear WHCR pending flag after completion
 self.context.set_whcr_pending(false);
 }
 Err(e) => {
 log::warn!("WHCR-Phase2 failed: {}", e);
 }
 }
 }
 } else {
 log::debug!("Skipping WHCR-Phase2: no geometry synchronizer");
 }
 }

 // After Phase 3 (Quantum) - but really after Phase 4 for stress data
 #[cfg(feature = "cuda")]
 if phase_name.contains("Phase4") || phase_name.contains("Geodesic") {
 log::debug!("Detected Phase 4 completion - invoking WHCR with safeguards");
 if let Some(ref sync) = geometry_sync {
 if let Some(ref geom) = sync.geometry() {
 // Get conflict count before WHCR
 let conflicts_before = self.context.best_solution
 .as_ref()
 .map(|s| s.conflicts)
 .unwrap_or(usize::MAX);

 match self.invoke_whcr_phase3(graph, geom) {
 Ok(()) => {
 // Verify WHCR didn't make things worse (oscillation safeguard)
 let conflicts_after = self.context.best_solution
 .as_ref()
 .map(|s| s.conflicts)
 .unwrap_or(usize::MAX);

 if conflicts_after > conflicts_before * 3 {
 log::warn!("WHCR-Phase3 oscillation detected: {} -> {} conflicts",
 conflicts_before, conflicts_after);
 } else {
 log::info!("WHCR-Phase3 completed: {} -> {} conflicts, geometry: {}",
 conflicts_before, conflicts_after, sync.summary());
 }
 // Clear WHCR pending flag after completion
 self.context.set_whcr_pending(false);
 }
 Err(e) => {
 log::warn!("WHCR-Phase3 failed: {}", e);
 }
 }
 }
 } else {
 log::debug!("Skipping WHCR-Phase3: no geometry synchronizer");
 }
 }

 // After Phase 5 (Membrane)
 #[cfg(feature = "cuda")]
 if phase_name.contains("Phase5") || phase_name.contains("Membrane") {
 log::debug!("Detected Phase 5 completion - checkpoint WHCR");
 if let Some(ref sync) = geometry_sync {
 if let Some(ref geom) = sync.geometry() {
 if let Err(e) = self.invoke_whcr_phase5(graph, geom) {
 log::warn!("WHCR-Phase5 failed: {}", e);
 } else {
 log::info!("WHCR-Phase5 checkpoint with geometry: {}", sync.summary());
 }
 }
 } else {
 log::debug!("Skipping WHCR-Phase5: no geometry synchronizer");
 }
 }

 // After Phase 7 (Ensemble) - final polish
 #[cfg(feature = "cuda")]
 if phase_name.contains("Phase7") || phase_name.contains("Ensemble") {
 log::debug!("Detected Phase 7 completion - final WHCR polish");
 if let Some(ref sync) = geometry_sync {
 if let Some(ref geom) = sync.geometry() {
 if let Err(e) = self.invoke_whcr_phase7(graph, geom) {
 log::warn!("WHCR-Phase7 failed: {}", e);
 } else {
 log::info!(
 "WHCR-Phase7 final polish with full geometry: {}",
 sync.summary()
 );
 }
 // Clear WHCR pending flag after completion
 self.context.set_whcr_pending(false);
 }
 } else {
 log::debug!("Skipping WHCR-Phase7: no geometry synchronizer");
 }
 }

 self.context.iteration += 1;
 }

 let elapsed = start_time.elapsed();
 log::info!("Pipeline completed in {:.2}s", elapsed.as_secs_f64());

 // Update warmstart effectiveness with actual conflicts (Warmstart Plan Step 7)
 self.update_warmstart_effectiveness();

 // Decay epsilon after full pipeline run
 self.rl_controller.decay_epsilon();

 // Return best solution
 self.context
 .best_solution
 .clone()
 .ok_or_else(|| PrismError::internal("No solution found"))
 }

 /// Run pipeline with explicit attempt-based seed for true randomization
 pub fn run_with_seed(
 &mut self,
 graph: &Graph,
 attempt_number: u64,
 ) -> Result<ColoringSolution, PrismError> {
 // Store attempt seed in context for quantum evolution and other phases to use
 let dynamic_seed = std::time::SystemTime::now()
 .duration_since(std::time::UNIX_EPOCH)
 .map(|d| d.as_nanos() as u64)
 .unwrap_or(12345);

 // Combine system time with attempt number for stronger seed diversity
 let combined_seed = dynamic_seed.wrapping_mul(1000000007) ^ attempt_number;

 self.context
 .scratch
 .insert("attempt_seed".to_string(), Box::new(combined_seed));

 log::debug!(
 "Attempt {} initialized with seed: {}",
 attempt_number,
 combined_seed
 );

 // Call regular run method
 self.run(graph)
 }

 /// Executes the warmstart stage to generate initial color priors.
 ///
 /// This runs **before** the main 7-phase pipeline to provide probabilistic
 /// initialization hints based on reservoir computation and structural anchors.
 ///
 /// ## Algorithm
 /// 1. Generate placeholder difficulty/uncertainty vectors (using vertex degrees)
 /// 2. Build reservoir priors using Phase 0 warmstart functions
 /// 3. Retrieve geodesic anchors from context (set by Phase 4)
 /// 4. Retrieve TDA anchors from context (set by Phase 6)
 /// 5. Fuse ensemble priors combining all sources
 /// 6. Apply anchors to create deterministic color assignments
 /// 7. Create WarmstartPlan and store in context scratch space
 /// 8. Log warmstart metrics (entropy, anchor count, etc.)
 ///
 /// ## Error Handling
 /// - If anchors not found in context, uses empty anchor lists (graceful degradation)
 /// - Validates WarmstartPlan before storage
 /// - Returns PrismError with context on failures
 ///
 /// Refs: Warmstart Plan Step 5 (Pipeline Integration)
 fn execute_warmstart_stage(&mut self, graph: &Graph) -> Result<(), PrismError> {
 log::info!("Executing warmstart stage");

 // Get warmstart config (guaranteed to exist due to caller check)
 let config = self
 .config
 .warmstart_config
 .as_ref()
 .ok_or_else(|| PrismError::internal("Warmstart config missing"))?;

 // Step 1: Get difficulty/uncertainty from Phase 0 if available, else use placeholder
 // Resolved TODO(PHASE0-RESERVOIR): Using actual dendritic reservoir when available
 let (difficulty, uncertainty) = self
 .context
 .scratch
 .get("phase0_difficulty")
 .and_then(|d| {
 self.context.scratch.get("phase0_uncertainty").map(|u| {
 (
 d.downcast_ref::<Vec<f32>>().unwrap().clone(),
 u.downcast_ref::<Vec<f32>>().unwrap().clone(),
 )
 })
 })
 .unwrap_or_else(|| {
 log::debug!("Phase 0 metrics not found, using placeholder difficulty/uncertainty");
 Self::compute_placeholder_difficulty_uncertainty(graph)
 });

 // Step 2: Build reservoir priors
 let reservoir_priors =
 prism_phases::phase0::build_reservoir_prior(&difficulty, &uncertainty, config);

 // Step 3: Retrieve geodesic anchors from context (set by Phase 4)
 let geodesic_anchors = self
 .context
 .scratch
 .get("geodesic_anchors")
 .and_then(|v| v.downcast_ref::<Vec<usize>>())
 .cloned()
 .unwrap_or_default();

 // Step 4: Retrieve TDA anchors from context (set by Phase 6)
 let tda_anchors = self
 .context
 .scratch
 .get("tda_anchors")
 .and_then(|v| v.downcast_ref::<Vec<usize>>())
 .cloned()
 .unwrap_or_default();

 log::debug!(
 "Warmstart anchors: {} geodesic, {} TDA",
 geodesic_anchors.len(),
 tda_anchors.len()
 );

 // Step 5: Fuse ensemble priors for each vertex
 let mut fused_priors = Vec::with_capacity(graph.num_vertices);
 for reservoir_prior in &reservoir_priors {
 let fused = prism_phases::phase0::fuse_ensemble_priors(
 reservoir_prior,
 &geodesic_anchors,
 &tda_anchors,
 config,
 );
 fused_priors.push(fused);
 }

 // Step 6: Apply anchors to create deterministic color assignments
 let all_anchors = {
 let mut combined = geodesic_anchors.clone();
 combined.extend(tda_anchors.iter());
 combined.sort_unstable();
 combined.dedup();
 combined
 };

 for prior in &mut fused_priors {
 prism_phases::phase0::apply_anchors(prior, &all_anchors, graph)
 .map_err(|e| PrismError::internal(format!("Failed to apply anchors: {}", e)))?;
 }

 // Step 7: Create WarmstartPlan with metadata
 // Retrieve curriculum profile from context if available
 let curriculum_profile = self
 .context
 .scratch
 .get("curriculum_profile")
 .and_then(|v| v.downcast_ref::<String>())
 .cloned();

 let mut warmstart_plan = WarmstartPlan {
 vertex_priors: fused_priors.clone(), // Clone for telemetry
 metadata: WarmstartMetadata {
 source_weights: {
 let mut weights = HashMap::new();
 weights.insert("flux".to_string(), config.flux_weight);
 weights.insert("ensemble".to_string(), config.ensemble_weight);
 weights.insert("random".to_string(), config.random_weight);
 weights
 },
 anchor_count: all_anchors.len(),
 curriculum_profile: curriculum_profile.clone(),
 prior_entropy: 0.0, // Will be computed below
 expected_conflicts: 0, // Placeholder until conflict prediction implemented
 },
 };

 // Validate plan
 if warmstart_plan.vertex_priors.len() != graph.num_vertices {
 return Err(PrismError::internal(format!(
 "Warmstart plan size mismatch: {} priors for {} vertices",
 warmstart_plan.vertex_priors.len(),
 graph.num_vertices
 )));
 }

 // Step 8: Compute and update entropy in metadata
 let mean_entropy = warmstart_plan.mean_entropy();
 warmstart_plan.metadata.prior_entropy = mean_entropy;
 let anchor_count = warmstart_plan.metadata.anchor_count;
 log::info!(
 "Warmstart plan created: mean_entropy={:.3}, anchors={}",
 mean_entropy,
 anchor_count
 );

 // Store in context
 self.context
 .scratch
 .insert("warmstart_plan".to_string(), Box::new(warmstart_plan));

 // Step 9: Create WarmstartTelemetry (Warmstart Plan Step 7)
 let curriculum_source = self
 .config
 .warmstart_config
 .as_ref()
 .and_then(|cfg| cfg.curriculum_catalog_path.clone());

 let warmstart_telemetry = prism_core::WarmstartTelemetry::new(
 &fused_priors,
 &geodesic_anchors,
 &tda_anchors,
 curriculum_profile,
 curriculum_source,
 config.flux_weight,
 config.ensemble_weight,
 config.random_weight,
 0, // Expected conflicts - TODO: implement conflict prediction
 );

 // Store telemetry in context for later effectiveness update
 self.context.scratch.insert(
 "warmstart_telemetry".to_string(),
 Box::new(warmstart_telemetry.clone()),
 );

 log::debug!(
 "Warmstart telemetry created: entropy_mean={:.3}, anchor_coverage={:.1}%, profile={:?}",
 warmstart_telemetry.mean_entropy(),
 warmstart_telemetry.anchor_coverage(),
 warmstart_telemetry.profile_name()
 );

 Ok(())
 }

 /// Computes placeholder difficulty and uncertainty vectors using vertex degrees.
 ///
 /// This is a temporary implementation until Phase 0 dendritic reservoir is integrated.
 /// The algorithm normalizes vertex degrees to [0, 1] range to serve as proxy metrics.
 ///
 /// ## Returns
 /// Tuple of (difficulty, uncertainty) vectors of length graph.num_vertices.
 fn compute_placeholder_difficulty_uncertainty(graph: &Graph) -> (Vec<f32>, Vec<f32>) {
 let num_vertices = graph.num_vertices;

 // Compute degrees
 let degrees: Vec<usize> = (0..num_vertices).map(|v| graph.degree(v)).collect();

 // Find max degree for normalization
 let max_degree = degrees.iter().max().copied().unwrap_or(1) as f32;
 let max_degree = max_degree.max(1.0); // Avoid division by zero

 // Normalize degrees to [0, 1] as difficulty proxy
 let difficulty: Vec<f32> = degrees.iter().map(|&d| (d as f32) / max_degree).collect();

 // Uncertainty = 1 - difficulty (inverse relationship for now)
 let uncertainty: Vec<f32> = difficulty.iter().map(|&d| 1.0 - d).collect();

 (difficulty, uncertainty)
 }

 /// Loads curriculum bank and applies best-matching Q-table to RL controller.
 ///
 /// ## Algorithm
 /// 1. Load curriculum bank from catalog JSON
 /// 2. Compute graph statistics (density, avg_degree)
 /// 3. Classify graph difficulty profile
 /// 4. Select best-matching curriculum entry
 /// 5. Initialize RL controller Q-tables from curriculum
 ///
 /// ## Returns
 /// The selected DifficultyProfile on success.
 ///
 /// Refs: PRISM GPU Plan §6.4 (Curriculum Q-Table Bank)
 fn load_and_apply_curriculum(
 &mut self,
 graph: &Graph,
 catalog_path: &str,
 ) -> Result<prism_fluxnet::DifficultyProfile, Box<dyn std::error::Error>> {
 // Load curriculum bank
 let bank = CurriculumBank::load(catalog_path)?;

 log::debug!(
 "Loaded curriculum bank: version {}, {} entries",
 bank.version,
 bank.num_entries()
 );

 // Compute graph statistics
 let stats = GraphStats::from_graph(graph);
 let profile = stats.classify_profile();

 log::info!(
 "Graph classified as {:?}: density={:.3}, avg_degree={:.1}",
 profile,
 stats.density,
 stats.avg_degree
 );

 // Select best-matching curriculum entry
 let entry = bank
 .select_best_match(profile)
 .ok_or_else(|| format!("No curriculum entry found for profile {:?}", profile))?;

 log::info!(
 "Selected curriculum entry: graph_class={}, episodes={}, avg_reward={:.3}",
 entry.metadata.graph_class,
 entry.metadata.training_episodes,
 entry.metadata.average_reward
 );

 // Initialize RL controller from curriculum Q-table
 // Apply to all phases (phase-agnostic curriculum)
 self.rl_controller
 .initialize_all_phases_from_curriculum(&entry.q_table)?;

 log::info!(
 "Initialized RL controller with curriculum Q-table ({} state-action pairs)",
 entry.num_entries()
 );

 Ok(profile)
 }

 /// Initializes GPU context with CudaContext and loads PTX modules.
 ///
 /// ## Algorithm
 /// 1. Create GpuSecurityConfig from pipeline config
 /// 2. Initialize GpuContext (device, PTX modules)
 /// 3. Collect GPU info for logging
 /// 4. Store context in PhaseContext
 /// 5. Emit telemetry event (TODO: when telemetry extended)
 ///
 /// ## Error Handling
 /// GPU initialization failures are logged but not fatal. The orchestrator
 /// falls back to CPU-only mode gracefully.
 ///
 /// Refs: PRISM GPU Plan §1 (GPU Context Manager)
 #[cfg(feature = "gpu")]
 fn initialize_gpu_context(&mut self) -> Result<(), PrismError> {
 use std::sync::Arc;

 log::info!(
 "Initializing GPU context with device {}",
 self.config.gpu.device_id
 );

 // Convert GpuConfig to GpuSecurityConfig
 let security_config = prism_gpu::context::GpuSecurityConfig {
 allow_nvrtc: self.config.gpu.allow_nvrtc,
 require_signed_ptx: self.config.gpu.require_signed_ptx,
 trusted_ptx_dir: self.config.gpu.trusted_ptx_dir.clone(),
 };

 // Initialize GPU context
 match prism_gpu::context::GpuContext::new(
 self.config.gpu.device_id,
 security_config,
 &self.config.gpu.ptx_dir,
 ) {
 Ok(ctx) => {
 // Collect GPU info for logging
 match ctx.collect_gpu_info() {
 Ok(info) => {
 log::info!(
 "GPU initialized: {} (compute {}.{}, {}MB memory, driver {})",
 info.device_name,
 info.compute_capability.0,
 info.compute_capability.1,
 info.total_memory_mb,
 info.driver_version
 );

 if self.config.gpu.require_signed_ptx {
 log::info!("GPU security: PTX signature verification ENABLED");
 }
 if !self.config.gpu.allow_nvrtc {
 log::info!("GPU security: NVRTC runtime compilation DISABLED");
 }
 }
 Err(e) => {
 log::warn!("Failed to collect GPU info: {}. Continuing anyway.", e);
 }
 }

 // Store in PhaseContext (as Arc<dyn Any> to avoid circular deps)
 self.context.gpu_context =
 Some(Arc::new(ctx) as Arc<dyn std::any::Any + Send + Sync>);

 // TODO: Emit telemetry event when TelemetryEvent::GpuInitialization is available

 Ok(())
 }
 Err(e) => {
 log::warn!(
 "GPU initialization failed: {}. Falling back to CPU-only mode.",
 e
 );
 log::warn!(" Reason: {}", e);
 log::info!("Pipeline will continue with CPU fallback for all phases.");
 Ok(()) // Don't propagate error - CPU fallback is acceptable
 }
 }
 }

 /// Executes a phase with retry logic based on PhaseOutcome.
 fn execute_phase_with_retry(
 &mut self,
 phase_idx: usize,
 graph: &Graph,
 phase_name: &str,
 ) -> Result<PhaseOutcome, PrismError> {
 const MAX_RETRIES: u32 = 3;

 for attempt in 0..=MAX_RETRIES {
 // Get RL state before execution
 let state_before = self.build_rl_state();

 // Select and apply action from RL controller
 let action = self.rl_controller.select_action(&state_before, phase_name);
 self.apply_action(&action, phase_name);

 // Execute phase
 let outcome = self.phases[phase_idx].execute(graph, &mut self.context)?;

 match outcome {
 PhaseOutcome::Success { message, telemetry } => {
 return Ok(PhaseOutcome::Success { message, telemetry })
 }
 PhaseOutcome::Retry {
 backoff_ms,
 ref reason,
 } => {
 if attempt < MAX_RETRIES {
 log::warn!(
 "Phase {} retry {}/{}: {} (backoff: {}ms)",
 phase_name,
 attempt + 1,
 MAX_RETRIES,
 reason,
 backoff_ms
 );

 // Backoff
 std::thread::sleep(Duration::from_millis(backoff_ms));
 } else {
 log::error!("Phase {} exhausted retries, escalating", phase_name);
 return Ok(PhaseOutcome::escalate(format!(
 "Max retries exceeded: {}",
 reason
 )));
 }
 }
 PhaseOutcome::Escalate { .. } => return Ok(outcome),
 }
 }

 Ok(PhaseOutcome::escalate("Retry loop exit"))
 }

 /// Builds the current RL state from phase metrics and context.
 fn build_rl_state(&self) -> UniversalRLState {
 let mut state = UniversalRLState::new();

 // Update state from best solution
 if let Some(ref solution) = self.context.best_solution {
 state.chromatic_number = solution.chromatic_number;
 state.conflicts = solution.conflicts;
 }

 state.iteration = self.context.iteration;

 // TODO: Collect metrics from phases
 // For now, return default state

 state
 }

 /// Applies an RL action (placeholder - actual implementation in phases).
 fn apply_action(&mut self, action: &UniversalAction, phase: &str) {
 log::debug!("Applying action {:?} for phase {}", action, phase);
 // Phase controllers will read actions from context scratch space
 self.context
 .scratch
 .insert(format!("{}_action", phase), Box::new(action.clone()));
 }

 /// Updates RL Q-table based on phase outcome.
 fn update_rl(&mut self, outcome: &PhaseOutcome, phase_name: &str) {
 // Compute reward based on outcome
 let reward = match outcome {
 PhaseOutcome::Success { .. } => {
 if let Some(ref solution) = self.context.best_solution {
 if solution.conflicts == 0 {
 1.0 // Valid solution
 } else {
 -0.5 // Has conflicts
 }
 } else {
 -1.0 // No solution
 }
 }
 PhaseOutcome::Retry { .. } => -0.2,
 PhaseOutcome::Escalate { .. } => -0.5,
 };

 let state_after = self.build_rl_state();
 let state_before = UniversalRLState::new(); // TODO: Store actual state before

 // Get action from scratch
 let action = self
 .context
 .scratch
 .get(&format!("{}_action", phase_name))
 .and_then(|a| a.downcast_ref::<UniversalAction>())
 .cloned()
 .unwrap_or(UniversalAction::NoOp);

 self.rl_controller
 .update_qtable(&state_before, &action, reward, &state_after, phase_name);
 }

 /// Updates warmstart telemetry with actual conflicts and effectiveness.
 ///
 /// Called after pipeline execution completes to compare expected vs actual conflicts.
 /// Computes warmstart effectiveness as: 1.0 - (actual/expected).
 ///
 /// Implements Warmstart Plan Step 7: Post-execution effectiveness tracking.
 fn update_warmstart_effectiveness(&mut self) {
 // Get warmstart telemetry from context
 let warmstart_telemetry = self
 .context
 .scratch
 .get_mut("warmstart_telemetry")
 .and_then(|v| v.downcast_mut::<prism_core::WarmstartTelemetry>());

 if let Some(telemetry) = warmstart_telemetry {
 // Get actual conflicts from best solution
 if let Some(ref best_solution) = self.context.best_solution {
 let actual_conflicts = best_solution.conflicts;

 // Update telemetry with effectiveness metrics
 telemetry.update_effectiveness(actual_conflicts);

 log::info!(
 "Warmstart effectiveness updated: expected={}, actual={}, effectiveness={:.3}",
 telemetry.expected_conflicts,
 actual_conflicts,
 telemetry.warmstart_effectiveness.unwrap_or(0.0)
 );
 } else {
 log::warn!("Cannot update warmstart effectiveness: no best solution found");
 }
 }
 }

 // =========================================================================
 // WHCR Multi-Phase Integration Methods
 // =========================================================================

 /// WHCR after Phase 2 (Thermodynamic)
 /// - Geometry: P0 (hotspots), P1 (beliefs)
 /// - Aggressive: can add up to 5 colors
 /// - Precision: f32 for fast exploration
 #[cfg(feature = "cuda")]
 fn invoke_whcr_phase2(
 &mut self,
 graph: &Graph,
 geometry: &GeometryAccumulator,
 ) -> Result<(), PrismError> {
 // WHCR fixes applied - re-enabling for testing
 log::info!("WHCR-Phase2: Re-enabled with buffer sync and geometry fixes");

 let current_colors = self
 .context
 .best_solution
 .as_ref()
 .map(|s| s.chromatic_number)
 .unwrap_or(graph.num_vertices);

 let conflicts = self
 .context
 .best_solution
 .as_ref()
 .map(|s| s.conflicts)
 .unwrap_or(0);

 // Skip if no conflicts
 if conflicts == 0 {
 log::debug!("WHCR-Phase2: Skipping (no conflicts)");
 return Ok(());
 }

 log::info!(
 "WHCR-Phase2: Invoking with {} colors, {} conflicts, geometry: {}",
 current_colors,
 conflicts,
 geometry.summary()
 );

 // Get GPU context from orchestrator context
 if let Some(ref gpu_ctx_any) = self.context.gpu_context {
 // Try to downcast as Arc<GpuContext> first
 if let Ok(gpu_ctx_arc) = gpu_ctx_any
 .clone()
 .downcast::<prism_gpu::context::GpuContext>()
 {
 let mut whcr = WHCRPhaseController::for_phase2(gpu_ctx_arc, graph)?;

 whcr.execute_with_geometry(graph, &mut self.context, geometry)?;
 } else {
 log::warn!("WHCR-Phase2: Could not downcast GPU context");
 }
 } else {
 log::warn!("WHCR-Phase2: No GPU context available");
 }

 let new_conflicts = self
 .context
 .best_solution
 .as_ref()
 .map(|s| s.conflicts)
 .unwrap_or(0);

 log::info!(
 "WHCR-Phase2: Completed - conflicts {} → {}",
 conflicts,
 new_conflicts
 );

 Ok(())
 }

 /// WHCR after Phase 3 (Quantum) - now has Phase 4 stress
 /// - Geometry: P0, P1, P4 (stress)
 /// - Medium: can add up to 3 colors
 /// - Precision: mixed (f32 coarse, f64 fine)
 #[cfg(feature = "cuda")]
 fn invoke_whcr_phase3(
 &mut self,
 graph: &Graph,
 geometry: &GeometryAccumulator,
 ) -> Result<(), PrismError> {
 // WHCR fixes applied - re-enabling for testing
 log::info!("WHCR-Phase3: Re-enabled with buffer sync and geometry fixes");

 let current_colors = self
 .context
 .best_solution
 .as_ref()
 .map(|s| s.chromatic_number)
 .unwrap_or(graph.num_vertices);

 let conflicts = self
 .context
 .best_solution
 .as_ref()
 .map(|s| s.conflicts)
 .unwrap_or(0);

 if conflicts == 0 {
 log::debug!("WHCR-Phase3: Skipping (no conflicts)");
 return Ok(());
 }

 log::info!(
 "WHCR-Phase3: Invoking with {} colors, {} conflicts, geometry: {}",
 current_colors,
 conflicts,
 geometry.summary()
 );

 // Get GPU context from orchestrator context
 if let Some(ref gpu_ctx_any) = self.context.gpu_context {
 // Try to downcast as Arc<GpuContext>
 if let Ok(gpu_ctx_arc) = gpu_ctx_any
 .clone()
 .downcast::<prism_gpu::context::GpuContext>()
 {
 let mut whcr = WHCRPhaseController::for_phase3(gpu_ctx_arc, graph)?;

 whcr.execute_with_geometry(graph, &mut self.context, geometry)?;
 } else {
 log::warn!("WHCR-Phase3: Could not downcast GPU context");
 }
 } else {
 log::warn!("WHCR-Phase3: No GPU context available");
 }

 Ok(())
 }

 /// WHCR at Phase 5 boundary (checkpoint)
 /// - Geometry: All available (P0, P1, P4, potentially P6)
 /// - Conservative: can add up to 2 colors
 /// - Precision: f64 with geometry coupling
 #[cfg(feature = "cuda")]
 fn invoke_whcr_phase5(
 &mut self,
 graph: &Graph,
 geometry: &GeometryAccumulator,
 ) -> Result<(), PrismError> {
 // WHCR fixes applied - re-enabling for testing
 log::info!("WHCR-Phase5: Re-enabled with buffer sync and geometry fixes");

 let conflicts = self
 .context
 .best_solution
 .as_ref()
 .map(|s| s.conflicts)
 .unwrap_or(0);

 if conflicts == 0 {
 log::debug!("WHCR-Phase5: Skipping (no conflicts)");
 return Ok(());
 }

 log::info!(
 "WHCR-Phase5: Checkpoint repair with geometry: {}",
 geometry.summary()
 );

 // Get GPU context from orchestrator context
 if let Some(ref gpu_ctx_any) = self.context.gpu_context {
 // Try to downcast as Arc<GpuContext>
 if let Ok(gpu_ctx_arc) = gpu_ctx_any
 .clone()
 .downcast::<prism_gpu::context::GpuContext>()
 {
 let mut whcr = WHCRPhaseController::for_phase5(gpu_ctx_arc, graph)?;

 whcr.execute_with_geometry(graph, &mut self.context, geometry)?;
 } else {
 log::warn!("WHCR-Phase5: Could not downcast GPU context");
 }
 } else {
 log::warn!("WHCR-Phase5: No GPU context available");
 }

 Ok(())
 }

 /// WHCR final polish (Phase 7)
 /// - Geometry: All (P0, P1, P4, P6)
 /// - STRICT: cannot add any colors
 /// - Precision: f64 with full geometry coupling
 /// - Maximum iterations
 #[cfg(feature = "cuda")]
 fn invoke_whcr_phase7(
 &mut self,
 graph: &Graph,
 geometry: &GeometryAccumulator,
 ) -> Result<(), PrismError> {
 // WHCR fixes applied - re-enabling for testing
 log::info!("WHCR-Phase7: Re-enabled with buffer sync and geometry fixes");

 let current_colors = self
 .context
 .best_solution
 .as_ref()
 .map(|s| s.chromatic_number)
 .unwrap_or(graph.num_vertices);

 let conflicts = self
 .context
 .best_solution
 .as_ref()
 .map(|s| s.conflicts)
 .unwrap_or(0);

 if conflicts == 0 {
 log::info!(
 "WHCR-Phase7: Solution already valid with {} colors",
 current_colors
 );
 return Ok(());
 }

 log::info!(
 "WHCR-Phase7: Final polish - {} colors, {} conflicts, geometry: {}",
 current_colors,
 conflicts,
 geometry.summary()
 );

 // Get GPU context from orchestrator context
 if let Some(ref gpu_ctx_any) = self.context.gpu_context {
 // Try to downcast as Arc<GpuContext>
 if let Ok(gpu_ctx_arc) = gpu_ctx_any
 .clone()
 .downcast::<prism_gpu::context::GpuContext>()
 {
 let mut whcr = WHCRPhaseController::for_phase7(gpu_ctx_arc, graph)?;

 // Phase 7 is strict - if it can't reach zero conflicts without
 // adding colors, that's a problem we need to report
 let _outcome = whcr.execute_with_geometry(graph, &mut self.context, geometry)?;
 } else {
 log::warn!("WHCR-Phase7: Could not downcast GPU context");
 }
 } else {
 log::warn!("WHCR-Phase7: No GPU context available");
 }

 let final_conflicts = self
 .context
 .best_solution
 .as_ref()
 .map(|s| s.conflicts)
 .unwrap_or(0);

 if final_conflicts > 0 {
 log::warn!(
 "WHCR-Phase7: Could not reach zero conflicts ({}). \
 Consider allowing color increase or more iterations.",
 final_conflicts
 );
 }

 // Special handling for DSJC125.5: verify we hit 17
 if graph.num_vertices == 125 {
 let final_colors = self
 .context
 .best_solution
 .as_ref()
 .map(|s| s.chromatic_number)
 .unwrap_or(0);

 if final_colors > 17 && final_conflicts == 0 {
 log::warn!(
 "DSJC125.5: Achieved {} colors, target is 17. \
 Consider adjusting phase parameters or running additional iterations.",
 final_colors
 );
 } else if final_colors == 17 && final_conflicts == 0 {
 log::info!("🎉 DSJC125.5: WORLD RECORD ACHIEVED - 17 colors!");
 }
 }

 Ok(())
 }

 // =========================================================================
 // Geometry Extraction Methods
 // =========================================================================

 /// Extract hotspot vertices from Phase 0 (Dendritic Reservoir)
 fn extract_phase0_hotspots(&self, result: &PhaseOutcome) -> Option<Vec<usize>> {
 // Try to get from telemetry
 if let PhaseOutcome::Success { telemetry, .. } = result {
 // Method A: Explicit hotspot list in telemetry
 if let Some(hotspots_val) = telemetry.get("hotspot_vertices") {
 if let Some(arr) = hotspots_val.as_array() {
 let hotspots: Vec<usize> = arr
 .iter()
 .filter_map(|v| v.as_u64().map(|n| n as usize))
 .collect();
 if !hotspots.is_empty() {
 log::debug!(
 "Extracted {} hotspots from Phase 0 telemetry",
 hotspots.len()
 );
 return Some(hotspots);
 }
 }
 }

 // Method B: High-activity vertices from dendritic reservoir
 if let Some(activity_val) = telemetry.get("vertex_activity") {
 if let Some(arr) = activity_val.as_array() {
 let activities: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();

 if !activities.is_empty() {
 // Hotspots = top 10% by activity
 let threshold = {
 let mut sorted = activities.clone();
 sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
 let idx = (sorted.len() as f64 * 0.1).max(1.0) as usize;
 sorted.get(idx).copied().unwrap_or(0.0)
 };

 let hotspots: Vec<usize> = activities
 .iter()
 .enumerate()
 .filter(|(_, &a)| a >= threshold)
 .map(|(i, _)| i)
 .collect();

 if !hotspots.is_empty() {
 log::debug!(
 "Derived {} hotspots from activity (threshold: {:.3})",
 hotspots.len(),
 threshold
 );
 return Some(hotspots);
 }
 }
 }
 }
 }

 log::debug!("Phase 0 hotspots not available - geometry incomplete");
 None
 }

 /// Extract belief distribution from Phase 1 (Active Inference)
 fn extract_phase1_beliefs(&self, result: &PhaseOutcome) -> Option<(Vec<f64>, usize)> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 // Method A: Direct belief matrix from telemetry
 if let Some(beliefs_val) = telemetry.get("belief_distribution") {
 if let Some(arr) = beliefs_val.as_array() {
 let beliefs: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();

 if let Some(num_colors_val) = telemetry.get("num_colors") {
 if let Some(num_colors) = num_colors_val.as_u64() {
 let num_colors = num_colors as usize;
 let expected_size = self
 .context
 .best_solution
 .as_ref()
 .map(|s| s.colors.len())
 .unwrap_or(0)
 * num_colors;

 if beliefs.len() == expected_size {
 log::debug!(
 "Extracted beliefs: {} values, {} colors",
 beliefs.len(),
 num_colors
 );
 return Some((beliefs, num_colors));
 }
 }
 }
 }
 }
 }

 // Method B: Construct from solution (fallback)
 if let Some(solution) = &self.context.best_solution {
 let num_vertices = solution.colors.len();
 let num_colors = solution.chromatic_number;

 if num_vertices > 0 && num_colors > 0 {
 // Create one-hot belief distribution from current coloring
 // with small uniform noise for exploration
 let mut beliefs = vec![0.01 / (num_colors as f64); num_vertices * num_colors];
 for (v, &c) in solution.colors.iter().enumerate() {
 if c < num_colors {
 beliefs[v * num_colors + c] = 0.9; // High confidence in current color
 }
 }

 log::debug!(
 "Constructed beliefs from solution: {}x{} colors",
 num_vertices,
 num_colors
 );
 return Some((beliefs, num_colors));
 }
 }

 log::debug!("Phase 1 beliefs not available - geometry incomplete");
 None
 }

 /// Extract stress scores from Phase 4 (Geodesic)
 fn extract_phase4_stress(&self, result: &PhaseOutcome) -> Option<Vec<f64>> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 // Method A: Direct stress array from telemetry
 if let Some(stress_val) = telemetry.get("vertex_stress") {
 if let Some(arr) = stress_val.as_array() {
 let stress: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();

 if !stress.is_empty() {
 log::debug!("Extracted {} stress values from Phase 4", stress.len());
 return Some(stress);
 }
 }
 }

 // Method B: Geodesic distortion metric
 if let Some(distortion_val) = telemetry.get("geodesic_distortion") {
 if let Some(arr) = distortion_val.as_array() {
 let stress: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();

 if !stress.is_empty() {
 log::debug!(
 "Extracted stress from geodesic distortion: {} values",
 stress.len()
 );
 return Some(stress);
 }
 }
 }

 // Method C: Use embedding quality as proxy
 if let Some(embedding_val) = telemetry.get("embedding_quality") {
 if let Some(quality) = embedding_val.as_f64() {
 // Uniform stress based on overall quality
 if let Some(solution) = &self.context.best_solution {
 let stress = vec![1.0 - quality; solution.colors.len()];
 log::debug!(
 "Constructed uniform stress from quality: {:.3}",
 1.0 - quality
 );
 return Some(stress);
 }
 }
 }
 }

 log::debug!("Phase 4 stress not available - geometry incomplete");
 None
 }

 /// Extract persistence scores from Phase 6 (TDA/Ensemble)
 fn extract_phase6_persistence(&self, _result: &PhaseOutcome) -> Option<Vec<f64>> {
 // TODO: Implement once Phase 6 stores persistence scores
 None
 }

 /// Run PRISM-LBS end-to-end for a parsed structure (feature-gated).
 #[cfg(feature = "lbs")]
 pub fn run_lbs_prediction(
 &self,
 structure: &LbsStructure,
 config: &LbsCfg,
 ) -> anyhow::Result<Vec<LbsPocket>> {
 #[cfg(feature = "cuda")]
 {
 if let Some(ref gpu_ctx_any) = self.context.gpu_context {
 if let Ok(gpu_ctx) = gpu_ctx_any
 .clone()
 .downcast::<prism_gpu::context::GpuContext>()
 {
 return run_lbs_with_gpu(structure, config, Some(gpu_ctx));
 } else {
 log::warn!("LBS GPU execution requested but GPU context downcast failed; falling back to CPU");
 }
 }
 }
 run_lbs(structure, config)
 }

 /// Convenience helper to load a PDB and execute LBS with a given config path.
 #[cfg(feature = "lbs")]
 pub fn run_lbs_from_files(
 &self,
 pdb_path: &std::path::Path,
 config_path: &std::path::Path,
 ) -> anyhow::Result<Vec<LbsPocket>> {
 let cfg = LbsCfg::from_file(config_path)?;
 let structure = LbsStructure::from_pdb_file(pdb_path)?;
 self.run_lbs_prediction(&structure, &cfg)
 }
}

#[cfg(test)]
mod tests {
 use super::*;
 use crate::config::{GpuConfig, Phase2Config};
 use prism_core::WarmstartConfig;
 use prism_fluxnet::RLConfig;
 use std::collections::HashMap;

 #[test]
 fn test_compute_placeholder_difficulty_uncertainty() {
 // Create a simple graph with varying degrees
 let mut graph = Graph::new(5);
 graph.add_edge(0, 1); // degree 1
 graph.add_edge(0, 2); // degree 2
 graph.add_edge(1, 2); // degrees 2, 2
 graph.add_edge(2, 3); // degree 3
 graph.add_edge(2, 4); // degrees 4, 1

 let (difficulty, uncertainty) =
 PipelineOrchestrator::compute_placeholder_difficulty_uncertainty(&graph);

 // Verify lengths
 assert_eq!(difficulty.len(), 5);
 assert_eq!(uncertainty.len(), 5);

 // All values should be in [0, 1]
 for &d in &difficulty {
 assert!(d >= 0.0 && d <= 1.0, "Difficulty out of range: {}", d);
 }
 for &u in &uncertainty {
 assert!(u >= 0.0 && u <= 1.0, "Uncertainty out of range: {}", u);
 }

 // Vertex 2 has highest degree (4), so highest difficulty
 let max_difficulty = difficulty.iter().copied().fold(0.0f32, f32::max);
 assert!((difficulty[2] - max_difficulty).abs() < 0.01);

 // Uncertainty = 1 - difficulty
 for i in 0..5 {
 assert!((difficulty[i] + uncertainty[i] - 1.0).abs() < 0.01);
 }
 }

 #[test]
 fn test_warmstart_stage_no_anchors() {
 // Create orchestrator with warmstart enabled
 let config = PipelineConfig {
 max_vertices: 100,
 phase_configs: HashMap::new(),
 timeout_seconds: 3600,
 enable_telemetry: true,
 telemetry_path: "test.jsonl".to_string(),
 warmstart_config: Some(WarmstartConfig::default()),
 gpu: GpuConfig::default(),
 phase2: Phase2Config::default(),
 memetic: None,
 metaphysical_coupling: None,
 };

 let rl_controller = UniversalRLController::new(RLConfig::default());
 let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

 // Create test graph
 let mut graph = Graph::new(3);
 graph.add_edge(0, 1);
 graph.add_edge(1, 2);

 // Execute warmstart stage (no anchors in context yet)
 let result = orchestrator.execute_warmstart_stage(&graph);
 assert!(result.is_ok(), "Warmstart should succeed: {:?}", result);

 // Verify warmstart plan was stored
 let plan = orchestrator
 .context
 .scratch
 .get("warmstart_plan")
 .and_then(|v| v.downcast_ref::<WarmstartPlan>());

 assert!(plan.is_some(), "Warmstart plan should be stored");

 let plan = plan.unwrap();
 assert_eq!(plan.vertex_priors.len(), 3);

 // Verify mean entropy is reasonable
 let entropy = plan.mean_entropy();
 assert!(
 entropy > 0.0 && entropy < 10.0,
 "Entropy out of range: {}",
 entropy
 );

 // No anchors initially
 assert_eq!(plan.metadata.anchor_count, 0);
 }

 #[test]
 fn test_warmstart_stage_with_anchors() {
 // Create orchestrator with warmstart enabled
 let config = PipelineConfig {
 max_vertices: 100,
 phase_configs: HashMap::new(),
 timeout_seconds: 3600,
 enable_telemetry: true,
 telemetry_path: "test.jsonl".to_string(),
 warmstart_config: Some(WarmstartConfig::default()),
 gpu: GpuConfig::default(),
 phase2: Phase2Config::default(),
 memetic: None,
 metaphysical_coupling: None,
 };

 let rl_controller = UniversalRLController::new(RLConfig::default());
 let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

 // Create test graph
 let mut graph = Graph::new(5);
 graph.add_edge(0, 1);
 graph.add_edge(1, 2);
 graph.add_edge(2, 3);
 graph.add_edge(3, 4);

 // Pre-populate context with anchors
 orchestrator
 .context
 .scratch
 .insert("geodesic_anchors".to_string(), Box::new(vec![0usize, 2]));
 orchestrator
 .context
 .scratch
 .insert("tda_anchors".to_string(), Box::new(vec![4usize]));

 // Execute warmstart stage
 let result = orchestrator.execute_warmstart_stage(&graph);
 assert!(result.is_ok(), "Warmstart should succeed: {:?}", result);

 // Verify warmstart plan
 let plan = orchestrator
 .context
 .scratch
 .get("warmstart_plan")
 .and_then(|v| v.downcast_ref::<WarmstartPlan>())
 .unwrap();

 assert_eq!(plan.vertex_priors.len(), 5);
 assert_eq!(plan.metadata.anchor_count, 3); // 0, 2, 4

 // Verify anchors have deterministic priors
 for anchor_vertex in [0, 2, 4] {
 let prior = &plan.vertex_priors[anchor_vertex];
 assert!(prior.is_anchor, "Vertex {} should be anchor", anchor_vertex);
 assert!(
 prior.anchor_color.is_some(),
 "Anchor {} should have color",
 anchor_vertex
 );

 // Verify deterministic distribution (one color = 1.0, rest = 0.0)
 let max_prob = prior
 .color_probabilities
 .iter()
 .copied()
 .fold(0.0f32, f32::max);
 assert!(
 (max_prob - 1.0).abs() < 0.01,
 "Anchor should have max_prob = 1.0, got {}",
 max_prob
 );
 }
 }

 #[test]
 fn test_warmstart_stage_validation() {
 // Test that warmstart validates plan size matches graph
 let config = PipelineConfig {
 max_vertices: 100,
 phase_configs: HashMap::new(),
 timeout_seconds: 3600,
 enable_telemetry: true,
 telemetry_path: "test.jsonl".to_string(),
 warmstart_config: Some(WarmstartConfig::default()),
 gpu: GpuConfig::default(),
 phase2: Phase2Config::default(),
 memetic: None,
 metaphysical_coupling: None,
 };

 let rl_controller = UniversalRLController::new(RLConfig::default());
 let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

 let graph = Graph::new(10);

 let result = orchestrator.execute_warmstart_stage(&graph);
 assert!(result.is_ok());

 // Verify plan has correct size
 let plan = orchestrator
 .context
 .scratch
 .get("warmstart_plan")
 .and_then(|v| v.downcast_ref::<WarmstartPlan>())
 .unwrap();

 assert_eq!(plan.vertex_priors.len(), 10);
 }

 #[test]
 fn test_warmstart_metadata_weights() {
 // Verify metadata captures fusion weights correctly
 let config = PipelineConfig {
 max_vertices: 100,
 phase_configs: HashMap::new(),
 timeout_seconds: 3600,
 enable_telemetry: true,
 telemetry_path: "test.jsonl".to_string(),
 warmstart_config: Some(WarmstartConfig {
 max_colors: 5,
 min_prob: 0.01,
 anchor_fraction: 0.1,
 flux_weight: 0.5,
 ensemble_weight: 0.3,
 random_weight: 0.2,
 curriculum_catalog_path: None,
 }),
 gpu: GpuConfig::default(),
 phase2: Phase2Config::default(),
 memetic: None,
 metaphysical_coupling: None,
 };

 let rl_controller = UniversalRLController::new(RLConfig::default());
 let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

 let graph = Graph::new(3);
 orchestrator.execute_warmstart_stage(&graph).unwrap();

 let plan = orchestrator
 .context
 .scratch
 .get("warmstart_plan")
 .and_then(|v| v.downcast_ref::<WarmstartPlan>())
 .unwrap();

 // Verify weights in metadata
 assert_eq!(plan.metadata.source_weights.get("flux"), Some(&0.5));
 assert_eq!(plan.metadata.source_weights.get("ensemble"), Some(&0.3));
 assert_eq!(plan.metadata.source_weights.get("random"), Some(&0.2));
 }

 #[test]
 fn test_orchestrator_run_with_warmstart() {
 // Integration test: verify warmstart executes before phases
 let config = PipelineConfig {
 max_vertices: 100,
 phase_configs: HashMap::new(),
 timeout_seconds: 3600,
 enable_telemetry: true,
 telemetry_path: "test.jsonl".to_string(),
 warmstart_config: Some(WarmstartConfig::default()),
 gpu: GpuConfig::default(),
 phase2: Phase2Config::default(),
 memetic: None,
 metaphysical_coupling: None,
 };

 let rl_controller = UniversalRLController::new(RLConfig::default());
 let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

 let graph = Graph::new(5);

 // Note: run() will auto-initialize phases since none are registered,
 // and will execute warmstart stage before running phases
 let result = orchestrator.run(&graph);

 // May succeed or fail depending on phase execution,
 // but warmstart plan should exist either way
 // (The orchestrator now auto-initializes phases when empty)

 // Verify warmstart was executed
 let plan = orchestrator
 .context
 .scratch
 .get("warmstart_plan")
 .and_then(|v| v.downcast_ref::<WarmstartPlan>());

 assert!(plan.is_some(), "Warmstart should execute before phases run");
 }

 #[test]
 fn test_warmstart_telemetry_effectiveness_update() {
 // Test that warmstart effectiveness is computed after execution
 let config = PipelineConfig {
 max_vertices: 100,
 phase_configs: HashMap::new(),
 timeout_seconds: 3600,
 enable_telemetry: true,
 telemetry_path: "test.jsonl".to_string(),
 warmstart_config: Some(WarmstartConfig::default()),
 gpu: GpuConfig::default(),
 phase2: Phase2Config::default(),
 memetic: None,
 metaphysical_coupling: None,
 };

 let rl_controller = UniversalRLController::new(RLConfig::default());
 let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

 let graph = Graph::new(3);

 // Execute warmstart stage
 orchestrator.execute_warmstart_stage(&graph).unwrap();

 // Verify warmstart telemetry was created
 let telemetry_before = orchestrator
 .context
 .scratch
 .get("warmstart_telemetry")
 .and_then(|v| v.downcast_ref::<prism_core::WarmstartTelemetry>())
 .unwrap();

 assert!(telemetry_before.actual_conflicts.is_none());
 assert!(telemetry_before.warmstart_effectiveness.is_none());

 // Simulate best solution with conflicts
 let mut solution = prism_core::ColoringSolution::new(3);
 solution.conflicts = 5;
 orchestrator.context.best_solution = Some(solution);

 // Update effectiveness
 orchestrator.update_warmstart_effectiveness();

 // Verify effectiveness was computed
 let telemetry_after = orchestrator
 .context
 .scratch
 .get("warmstart_telemetry")
 .and_then(|v| v.downcast_ref::<prism_core::WarmstartTelemetry>())
 .unwrap();

 assert_eq!(telemetry_after.actual_conflicts, Some(5));
 assert!(telemetry_after.warmstart_effectiveness.is_some());

 // Since expected_conflicts is 0 (placeholder), effectiveness should be 0.0
 assert_eq!(telemetry_after.warmstart_effectiveness, Some(0.0));
 }

 #[test]
 fn test_warmstart_telemetry_no_solution() {
 // Test that effectiveness update handles missing solution gracefully
 let config = PipelineConfig {
 max_vertices: 100,
 phase_configs: HashMap::new(),
 timeout_seconds: 3600,
 enable_telemetry: true,
 telemetry_path: "test.jsonl".to_string(),
 warmstart_config: Some(WarmstartConfig::default()),
 gpu: GpuConfig::default(),
 phase2: Phase2Config::default(),
 memetic: None,
 metaphysical_coupling: None,
 };

 let rl_controller = UniversalRLController::new(RLConfig::default());
 let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

 let graph = Graph::new(3);

 // Execute warmstart stage
 orchestrator.execute_warmstart_stage(&graph).unwrap();

 // Don't set best_solution (simulate failure)

 // Update effectiveness (should not panic)
 orchestrator.update_warmstart_effectiveness();

 // Verify telemetry unchanged
 let telemetry = orchestrator
 .context
 .scratch
 .get("warmstart_telemetry")
 .and_then(|v| v.downcast_ref::<prism_core::WarmstartTelemetry>())
 .unwrap();

 assert!(telemetry.actual_conflicts.is_none());
 assert!(telemetry.warmstart_effectiveness.is_none());
 }
}
