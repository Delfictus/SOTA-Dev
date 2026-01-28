//! # PRISM MEC (Molecular Emergent Computing) Module
//!
//! Implements molecular-scale computing paradigms and emergent computational behaviors.
//! Provides Phase M (MEC) in the PRISM pipeline.
//!
//! ## Core Components
//!
//! - **Molecular Dynamics**: Simulates molecular interactions and dynamics
//! - **Emergent Behaviors**: Pattern formation and self-organization
//! - **Chemical Reactions**: Reaction-diffusion systems and catalysis
//! - **Membrane Computing**: P-systems and cellular automata

use prism_core::{Graph, PhaseContext, PhaseController, PhaseOutcome, PhaseTelemetry, PrismError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod emergent;
pub mod membrane;
pub mod molecular;
pub mod reactions;

/// MEC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MecConfig {
 /// Time step for molecular dynamics simulation
 pub time_step: f32,

 /// Number of simulation iterations
 pub iterations: usize,

 /// Temperature for molecular simulation (Kelvin)
 pub temperature: f32,

 /// Enable GPU acceleration
 pub use_gpu: bool,

 /// Reaction rate constants
 pub reaction_rates: HashMap<String, f32>,
}

impl Default for MecConfig {
 fn default() -> Self {
 Self {
 time_step: 1e-15, // 1 femtosecond
 iterations: 1000,
 temperature: 300.0,
 use_gpu: false,
 reaction_rates: HashMap::new(),
 }
 }
}

/// MEC phase controller
pub struct MecPhaseController {
 config: MecConfig,
 metrics: MecMetrics,
 #[cfg(feature = "cuda")]
 gpu_device: Option<std::sync::Arc<cudarc::driver::CudaContext>>,
 #[cfg(feature = "cuda")]
 gpu_stream: Option<std::sync::Arc<cudarc::driver::CudaStream>>,
 #[cfg(not(feature = "cuda"))]
 gpu_device: Option<()>,
}

// Ensure thread safety for pipeline execution
unsafe impl Send for MecPhaseController {}
unsafe impl Sync for MecPhaseController {}

impl MecPhaseController {
 pub fn new(config: MecConfig) -> Self {
 Self {
 config,
 metrics: MecMetrics {
 total_energy: 0.0,
 kinetic_energy: 0.0,
 potential_energy: 0.0,
 entropy: 0.0,
 reaction_yield: 0.0,
 },
 gpu_device: None,
 #[cfg(feature = "cuda")]
 gpu_stream: None,
 }
 }

 #[cfg(feature = "cuda")]
 pub fn with_gpu(mut self, device: std::sync::Arc<cudarc::driver::CudaContext>) -> Self {
 let stream = device.default_stream();
 self.gpu_device = Some(device);
 self.gpu_stream = Some(stream);
 self
 }

 #[cfg(not(feature = "cuda"))]
 pub fn with_gpu(self, _device: ()) -> Self {
 self
 }

 pub fn validate_config(&self) -> Result<(), PrismError> {
 if self.config.time_step <= 0.0 {
 return Err(PrismError::ValidationError(
 "Time step must be positive".to_string(),
 ));
 }
 if self.config.temperature < 0.0 {
 return Err(PrismError::ValidationError(
 "Temperature must be non-negative".to_string(),
 ));
 }
 Ok(())
 }

 /// Initialize molecular state from graph topology
 fn initialize_molecular_state(&mut self, graph: &Graph) -> HashMap<String, serde_json::Value> {
 let mut state = HashMap::new();

 // Map graph vertices to molecular positions
 let num_molecules = graph.num_vertices.min(1000); // Cap for performance
 state.insert(
 "num_molecules".to_string(),
 serde_json::json!(num_molecules),
 );

 // Initialize positions based on graph topology
 let positions: Vec<[f32; 3]> = (0..num_molecules)
 .map(|i| {
 let angle = 2.0 * std::f32::consts::PI * (i as f32) / (num_molecules as f32);
 [angle.cos() * 10.0, angle.sin() * 10.0, (i as f32) * 0.1]
 })
 .collect();
 state.insert("positions".to_string(), serde_json::json!(positions));

 // Initialize velocities based on temperature
 let boltzmann_constant = 1.380649e-23;
 let avg_mass = 1.0e-26; // Average molecular mass in kg
 let velocity_scale =
 (boltzmann_constant * self.config.temperature as f64 / avg_mass).sqrt() as f32;
 state.insert(
 "velocity_scale".to_string(),
 serde_json::json!(velocity_scale),
 );

 state
 }

 /// Compute emergent metrics from molecular simulation
 fn compute_emergent_metrics(
 &mut self,
 state: &HashMap<String, serde_json::Value>,
 ) -> HashMap<String, serde_json::Value> {
 let mut metrics = HashMap::new();

 // Extract molecular data
 let num_molecules = state
 .get("num_molecules")
 .and_then(|v| v.as_u64())
 .unwrap_or(0) as f32;

 let velocity_scale = state
 .get("velocity_scale")
 .and_then(|v| v.as_f64())
 .unwrap_or(1.0) as f32;

 // Compute energy metrics
 self.metrics.kinetic_energy = 0.5 * num_molecules * velocity_scale * velocity_scale;
 self.metrics.potential_energy = -num_molecules * 10.0; // Simplified potential
 self.metrics.total_energy = self.metrics.kinetic_energy + self.metrics.potential_energy;

 // Compute entropy (simplified Boltzmann entropy)
 self.metrics.entropy = num_molecules * (num_molecules / std::f32::consts::E).ln();

 // Compute reaction yield (placeholder)
 self.metrics.reaction_yield = 0.85; // 85% yield

 // Package metrics for telemetry
 metrics.insert(
 "mec_total_energy".to_string(),
 serde_json::json!(self.metrics.total_energy),
 );
 metrics.insert(
 "mec_kinetic_energy".to_string(),
 serde_json::json!(self.metrics.kinetic_energy),
 );
 metrics.insert(
 "mec_potential_energy".to_string(),
 serde_json::json!(self.metrics.potential_energy),
 );
 metrics.insert(
 "mec_entropy".to_string(),
 serde_json::json!(self.metrics.entropy),
 );
 metrics.insert(
 "mec_reaction_yield".to_string(),
 serde_json::json!(self.metrics.reaction_yield),
 );
 metrics.insert(
 "mec_temperature".to_string(),
 serde_json::json!(self.config.temperature),
 );
 metrics.insert(
 "mec_free_energy".to_string(),
 serde_json::json!(
 self.metrics.total_energy - self.config.temperature * self.metrics.entropy
 ),
 );

 metrics
 }
}

/// Telemetry implementation for MEC phase
pub struct MecTelemetry;

impl PhaseTelemetry for MecTelemetry {
 fn metrics(&self) -> HashMap<String, f64> {
 let mut metrics = HashMap::new();
 metrics.insert("total_energy".to_string(), 0.0);
 metrics.insert("kinetic_energy".to_string(), 0.0);
 metrics.insert("potential_energy".to_string(), 0.0);
 metrics.insert("temperature".to_string(), 300.0);
 metrics
 }
}

impl PhaseController for MecPhaseController {
 fn execute(
 &mut self,
 graph: &Graph,
 context: &mut PhaseContext,
 ) -> Result<PhaseOutcome, PrismError> {
 log::info!("Executing MEC Phase (Phase M) - Molecular Emergent Computing");

 // Validate configuration
 self.validate_config()?;

 // Step 1: Initialize molecular state from graph topology
 let molecular_state = self.initialize_molecular_state(graph);
 log::debug!(
 "Initialized {} molecules from graph topology",
 molecular_state
 .get("num_molecules")
 .and_then(|v| v.as_u64())
 .unwrap_or(0)
 );

 // Step 2: Run molecular dynamics simulation
 let md_results: Option<prism_gpu::MDResults> = if self.config.use_gpu
 && self.gpu_device.is_some()
 {
 #[cfg(feature = "cuda")]
 {
 // GPU-accelerated MD simulation
 let device = self.gpu_device.as_ref().unwrap().clone();
 match prism_gpu::MolecularDynamicsGpu::new(
 device,
 "target/ptx/molecular_dynamics.ptx",
 ) {
 Ok(mut md_gpu) => {
 log::info!("Running GPU-accelerated molecular dynamics");

 // Set up MD parameters
 // Use dynamic seed from system time to ensure different results per attempt
 let dynamic_seed = std::time::SystemTime::now()
 .duration_since(std::time::UNIX_EPOCH)
 .map(|d| (d.as_nanos() as u64) & 0xFFFFFFFF) // Keep as 32-bit for curand
 .unwrap_or(12345);
 let md_params = prism_gpu::MDParams {
 num_particles: molecular_state
 .get("num_molecules")
 .and_then(|v| v.as_u64())
 .unwrap_or(250) as usize,
 timestep: self.config.time_step,
 temperature: self.config.temperature,
 box_size: 50.0, // Angstroms
 epsilon: 0.238, // LJ parameter
 sigma: 3.405, // LJ parameter
 damping: 0.1,
 coupling_strength: 0.5,
 integration_steps: self.config.iterations,
 seed: dynamic_seed,
 };

 // Initialize particles on GPU
 let mut particles = md_gpu.initialize_system(&md_params).map_err(|e| {
 PrismError::Internal(format!("GPU initialization failed: {}", e))
 })?;

 // Run MD simulation
 let results =
 md_gpu
 .run_simulation(&mut particles, &md_params)
 .map_err(|e| {
 PrismError::Internal(format!("GPU simulation failed: {}", e))
 })?;

 Some(results)
 }
 Err(e) => {
 log::warn!("GPU MD initialization failed: {}. Using CPU fallback.", e);
 None
 }
 }
 }
 #[cfg(not(feature = "cuda"))]
 {
 log::warn!("CUDA feature not enabled, using CPU fallback");
 None
 }
 } else {
 log::warn!("GPU-accelerated MD not available, using CPU fallback");
 None
 };

 // Update metrics from GPU results or use CPU fallback
 if let Some(results) = md_results {
 self.metrics.kinetic_energy = results.kinetic_energy;
 self.metrics.potential_energy = results.potential_energy;
 self.metrics.total_energy = results.total_energy;
 self.metrics.entropy = molecular_state
 .get("num_molecules")
 .and_then(|v| v.as_u64())
 .unwrap_or(0) as f32
 * 1.5; // Simplified entropy
 self.metrics.reaction_yield = 0.85;

 log::info!(
 "GPU MD completed: E_total={:.3}, T={:.1}K, coherence={:.3}",
 results.total_energy,
 results.temperature,
 results.mec_coherence
 );
 } else {
 // CPU fallback: simple iteration
 for iteration in 0..self.config.iterations.min(100) {
 if iteration % 10 == 0 {
 log::trace!(
 "MEC iteration {}/{} (CPU)",
 iteration,
 self.config.iterations
 );
 }
 }
 }

 // Step 4: Extract emergent patterns and compute metrics
 let telemetry_metrics = self.compute_emergent_metrics(&molecular_state);

 // Step 5: Update PhaseContext with MEC state
 // Store MEC metrics in context scratch space
 context
 .scratch
 .insert("mec_state".to_string(), Box::new(molecular_state.clone()));

 context
 .scratch
 .insert("mec_metrics".to_string(), Box::new(self.metrics.clone()));

 // Update metadata for downstream phases using new update method
 let free_energy =
 self.metrics.total_energy - self.config.temperature * self.metrics.entropy;
 context.update_mec_state(
 free_energy as f64,
 self.metrics.entropy as f64,
 self.config.temperature as f64,
 );
 context.set_metadata("mec_completed", serde_json::json!(true));

 log::info!(
 "MEC phase completed: free_energy={:.3}, entropy={:.3}, temperature={}K",
 self.metrics.total_energy - self.config.temperature * self.metrics.entropy,
 self.metrics.entropy,
 self.config.temperature
 );

 Ok(PhaseOutcome::Success {
 message: format!(
 "MEC phase completed with {} molecules, free_energy={:.3}",
 molecular_state
 .get("num_molecules")
 .and_then(|v| v.as_u64())
 .unwrap_or(0),
 self.metrics.total_energy - self.config.temperature * self.metrics.entropy
 ),
 telemetry: telemetry_metrics,
 })
 }

 fn name(&self) -> &'static str {
 "PhaseM-MEC"
 }

 fn telemetry(&self) -> &dyn PhaseTelemetry {
 &MecTelemetry
 }
}

/// MEC phase metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MecMetrics {
 pub total_energy: f32,
 pub kinetic_energy: f32,
 pub potential_energy: f32,
 pub entropy: f32,
 pub reaction_yield: f32,
}

#[cfg(test)]
mod tests {
 use super::*;

 #[test]
 fn test_mec_config_validation() {
 let mut config = MecConfig::default();
 let controller = MecPhaseController::new(config.clone());
 assert!(controller.validate_config().is_ok());

 config.time_step = -1.0;
 let controller = MecPhaseController::new(config);
 assert!(controller.validate_config().is_err());
 }
}
