//! Phase 2: Thermodynamic Simulated Annealing with FluxNet RL.
//!
//! Implements parallel tempering simulated annealing for graph coloring with GPU acceleration.
//! Integrates with FluxNet RL for dynamic hyperparameter tuning.
//!
//! Implements §4.2 (Phase 2: Thermodynamic) of the PRISM GPU Plan.

use crate::conflict_repair::{repair_phase_output, ConflictRepairConfig, ConflictRepairEngine};
use prism_core::{
 ColoringSolution, Graph, PhaseContext, PhaseController, PhaseOutcome, PhaseTelemetry,
 PrismError,
};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
use prism_gpu::ThermodynamicGpu;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Phase 2: Thermodynamic Simulated Annealing.
///
/// Applies parallel tempering simulated annealing with multiple temperature schedules.
/// When GPU is available, uses CUDA kernel for acceleration. Otherwise, falls back to CPU greedy heuristics.
///
/// ## RL Integration
///
/// FluxNet actions control:
/// - Temperature min/max range
/// - Cooling rate
/// - Number of annealing iterations
/// - Number of temperature replicas
///
/// ## Telemetry
///
/// Emits:
/// - `temperature`: Current temperature (lowest replica)
/// - `cooling_rate`: Cooling schedule parameter
/// - `compaction_ratio`: Color space reduction achieved
/// - `guard_triggers`: Number of conflicts encountered
/// - `gpu_enabled`: Whether GPU acceleration is active
pub struct Phase2Thermodynamic {
 temperature: f64,
 cooling_rate: f64,
 compaction_ratio: f64,
 guard_triggers: usize,

 // GPU support
 #[cfg(feature = "cuda")]
 thermodynamic_gpu: Option<ThermodynamicGpu>,
 use_gpu: bool,

 // Tuning parameters (RL-adjustable)
 num_replicas: usize,
 iterations: usize,
 temp_min: f32,
 temp_max: f32,
}

impl Default for Phase2Thermodynamic {
 fn default() -> Self {
 Self::new()
 }
}

impl Phase2Thermodynamic {
 /// Creates a new Phase 2 controller with CPU fallback.
 pub fn new() -> Self {
 Self {
 temperature: 10.0,
 cooling_rate: 0.95,
 compaction_ratio: 0.0,
 guard_triggers: 0,
 #[cfg(feature = "cuda")]
 thermodynamic_gpu: None,
 use_gpu: false,
 num_replicas: 8,
 iterations: 10000,
 temp_min: 0.01,
 temp_max: 10.0,
 }
 }

 /// Creates a new Phase 2 controller with GPU acceleration.
 ///
 /// # Arguments
 /// * `device` - CUDA device handle
 /// * `ptx_path` - Path to thermodynamic.ptx module
 ///
 /// # Errors
 /// Returns error if GPU initialization fails.
 #[cfg(feature = "cuda")]
 pub fn new_with_gpu(device: Arc<CudaContext>, ptx_path: &str) -> Result<Self, PrismError> {
 let thermodynamic_gpu = ThermodynamicGpu::new(device, ptx_path).map_err(|e| {
 PrismError::gpu("Phase2", format!("Failed to init thermodynamic GPU: {}", e))
 })?;

 log::info!("Phase2: GPU thermodynamic acceleration enabled");

 Ok(Self {
 temperature: 10.0,
 cooling_rate: 0.95,
 compaction_ratio: 0.0,
 guard_triggers: 0,
 thermodynamic_gpu: Some(thermodynamic_gpu),
 use_gpu: true,
 num_replicas: 8,
 iterations: 10000,
 temp_min: 0.01,
 temp_max: 10.0,
 })
 }

 /// Configure hyperparameters for Phase 2.
 ///
 /// Allows runtime tuning of annealing parameters without recompilation.
 ///
 /// # Arguments
 /// * `iterations` - Number of annealing iterations (default: 10000, recommended: 50000-75000 for DSJC250)
 /// * `replicas` - Number of temperature replicas (default: 8, range: 4-16)
 /// * `temp_min` - Minimum temperature (default: 0.01, range: 0.001-0.1)
 /// * `temp_max` - Maximum temperature (default: 10.0, range: 5.0-50.0)
 pub fn with_hyperparameters(
 mut self,
 iterations: usize,
 replicas: usize,
 temp_min: f32,
 temp_max: f32,
 ) -> Self {
 self.iterations = iterations;
 self.num_replicas = replicas;
 self.temp_min = temp_min;
 self.temp_max = temp_max;

 log::info!("Phase2: Configured hyperparameters: iterations={}, replicas={}, temp_range=[{:.3}, {:.1}]",
 iterations, replicas, temp_min, temp_max);

 self
 }

 /// Apply RL action to adjust thermodynamic parameters.
 ///
 /// Maps ThermodynamicAction to parameter adjustments:
 /// - IncreaseTemperature: temp_max *= 1.5
 /// - DecreaseTemperature: temp_max *= 0.75
 /// - IncreaseCoolingRate: cooling_rate += 0.05 (capped at 0.99)
 /// - DecreaseCoolingRate: cooling_rate -= 0.05 (minimum 0.5)
 /// - AnnealFast: iterations /= 2 (minimum 1000)
 /// - AnnealSlow: iterations *= 2 (maximum 100000)
 /// - Reheat: temp_min *= 2.0, temp_max *= 1.5
 fn apply_rl_action_raw(&mut self, action_idx: usize) {
 // Phase 2 actions are in range [16, 23]
 let relative_idx = if (16..=23).contains(&action_idx) {
 action_idx - 16
 } else {
 return; // Invalid action for this phase
 };

 log::debug!("Phase2: Applying RL action index {}", relative_idx);

 match relative_idx {
 0 => {
 // IncreaseTemperature
 self.temp_max *= 1.5;
 self.temp_max = self.temp_max.min(50.0); // Cap at 50
 log::debug!("Phase2: Increased temp_max to {:.2}", self.temp_max);
 }
 1 => {
 // DecreaseTemperature
 self.temp_max *= 0.75;
 self.temp_max = self.temp_max.max(1.0); // Minimum 1.0
 log::debug!("Phase2: Decreased temp_max to {:.2}", self.temp_max);
 }
 2 => {
 // IncreaseCoolingRate
 self.cooling_rate += 0.05;
 self.cooling_rate = self.cooling_rate.min(0.99);
 log::debug!("Phase2: Increased cooling_rate to {:.3}", self.cooling_rate);
 }
 3 => {
 // DecreaseCoolingRate
 self.cooling_rate -= 0.05;
 self.cooling_rate = self.cooling_rate.max(0.5);
 log::debug!("Phase2: Decreased cooling_rate to {:.3}", self.cooling_rate);
 }
 4 => {
 // AnnealFast
 self.iterations = (self.iterations / 2).max(1000);
 log::debug!("Phase2: Reduced iterations to {}", self.iterations);
 }
 5 => {
 // AnnealSlow
 self.iterations = (self.iterations * 2).min(100000);
 log::debug!("Phase2: Increased iterations to {}", self.iterations);
 }
 6 => {
 // Reheat
 self.temp_min *= 2.0;
 self.temp_max *= 1.5;
 self.temp_min = self.temp_min.min(1.0);
 self.temp_max = self.temp_max.min(50.0);
 log::debug!(
 "Phase2: Reheated - temp_min={:.3}, temp_max={:.2}",
 self.temp_min,
 self.temp_max
 );
 }
 7 => {
 // NoOp
 log::debug!("Phase2: NoOp action");
 }
 _ => {
 log::warn!("Phase2: Unknown RL action index {}", relative_idx);
 }
 }
 }
}

impl PhaseController for Phase2Thermodynamic {
 fn execute(
 &mut self,
 graph: &Graph,
 context: &mut PhaseContext,
 ) -> Result<PhaseOutcome, PrismError> {
 log::info!("Phase 2: Thermodynamic (GPU: {})", self.use_gpu);

 // Read RL action from context scratch space
 if let Some(&action_idx) = context
 .scratch
 .get("rl_action")
 .and_then(|a| a.downcast_ref::<usize>())
 {
 self.apply_rl_action_raw(action_idx);
 }

 // Apply dendritic reservoir difficulty for temperature adjustment
 // Higher mean difficulty → higher starting temperature for better exploration
 if context.has_dendritic_metrics() {
 let mean_diff = context.mean_difficulty();
 let temp_boost = 1.0 + (mean_diff - 0.5) * 0.5; // Range: 0.75x to 1.25x

 let original_max = self.temp_max;
 self.temp_max *= temp_boost;
 self.temp_max = self.temp_max.clamp(5.0, 50.0);

 log::info!(
 "[Phase2] Dendritic coupling: mean_difficulty={:.3}, temp_boost={:.2}x, temp_max: {:.2}→{:.2}",
 mean_diff,
 temp_boost,
 original_max,
 self.temp_max
 );
 }

 // Get initial coloring from context or warmstart
 let initial_colors = if let Some(sol) = &context.best_solution {
 log::info!(
 "Phase2: Starting from warmstart with {} colors",
 sol.chromatic_number
 );
 sol.colors.clone()
 } else {
 log::info!("Phase2: No warmstart available, using greedy initialization");
 self.greedy_coloring(graph)
 };

 let solution = if self.use_gpu {
 #[cfg(feature = "cuda")]
 {
 self.run_gpu(graph, &initial_colors, context)?
 }
 #[cfg(not(feature = "cuda"))]
 {
 log::warn!("Phase2: GPU requested but CUDA feature not enabled");
 self.run_cpu(graph, &initial_colors)
 }
 } else {
 self.run_cpu(graph, &initial_colors)
 };

 // Update context
 let prev_chromatic = context
 .best_solution
 .as_ref()
 .map(|s| s.chromatic_number)
 .unwrap_or(graph.num_vertices);

 let improvement = if solution.chromatic_number < prev_chromatic {
 let reduction =
 (prev_chromatic - solution.chromatic_number) as f64 / prev_chromatic as f64;
 log::info!(
 "Phase2: Improved chromatic number from {} to {} ({:.1}% reduction)",
 prev_chromatic,
 solution.chromatic_number,
 reduction * 100.0
 );
 true
 } else {
 false
 };

 self.compaction_ratio = 1.0 - (solution.chromatic_number as f64 / prev_chromatic as f64);
 self.guard_triggers = solution.conflicts;

 context.update_best_solution(solution.clone());

 // Update RL state for next phase
 if let Some(rl_state) = context.rl_state.as_mut() {
 if let Some(state) = rl_state.downcast_mut::<prism_fluxnet::UniversalRLState>() {
 state.thermodynamic_temp = self.temperature;
 state.chromatic_number = solution.chromatic_number;
 state.conflicts = solution.conflicts;
 }
 }

 // Determine outcome
 if solution.conflicts == 0 {
 if improvement {
 Ok(PhaseOutcome::success())
 } else {
 Ok(PhaseOutcome::success()) // Still valid, no improvement
 }
 } else if solution.conflicts < graph.num_edges / 10 {
 // Check if WHCR is pending - if so, skip internal repair
 if context.is_whcr_pending() {
 log::info!(
 "Phase2: {} conflicts with {} colors - deferring to WHCR for repair",
 solution.conflicts,
 solution.chromatic_number
 );
 Ok(PhaseOutcome::success())
 } else {
 // Attempt conflict repair before retrying
 log::warn!(
 "Phase2: {} conflicts with {} colors, attempting repair",
 solution.conflicts,
 solution.chromatic_number
 );

 match repair_phase_output(graph, solution.clone(), "Phase2") {
 Ok(repaired) => {
 if repaired.conflicts == 0 {
 log::info!(
 "Phase2: ✓ Repaired to {} colors, 0 conflicts",
 repaired.chromatic_number
 );
 context.update_best_solution(repaired);
 Ok(PhaseOutcome::success())
 } else {
 log::warn!(
 "Phase2: Repair incomplete: {} conflicts remaining",
 repaired.conflicts
 );
 Ok(PhaseOutcome::retry(
 "Thermodynamic annealing has conflicts after repair, retrying",
 1,
 ))
 }
 }
 Err(e) => {
 log::error!("Phase2: Repair failed: {}", e);
 Ok(PhaseOutcome::retry(
 "Thermodynamic annealing has conflicts, retrying",
 1,
 ))
 }
 }
 }
 } else {
 // Check if WHCR is pending - if so, skip internal repair
 if context.is_whcr_pending() {
 log::info!(
 "Phase2: {} conflicts (high) with {} colors - deferring to WHCR for repair",
 solution.conflicts,
 solution.chromatic_number
 );
 Ok(PhaseOutcome::success())
 } else {
 // High conflicts - attempt desperate repair before escalating
 log::error!(
 "Phase2: {} conflicts (high) with {} colors, attempting aggressive repair",
 solution.conflicts,
 solution.chromatic_number
 );

 // Use more aggressive repair config for high conflicts
 let repair_config = ConflictRepairConfig {
 max_iterations: 1000,
 population_size: 50,
 mutation_rate: 0.5,
 allow_color_increase: true,
 max_color_increase: 5,
 use_kempe_chains: true,
 use_local_search: true,
 };

 let repair_engine = ConflictRepairEngine::new(repair_config);
 match repair_engine.repair(graph, solution.clone()) {
 Ok(repaired) => {
 if repaired.conflicts == 0 {
 log::info!(
 "Phase2: ✓ Aggressive repair succeeded: {} colors",
 repaired.chromatic_number
 );
 context.update_best_solution(repaired);
 Ok(PhaseOutcome::success())
 } else {
 log::warn!(
 "Phase2: Aggressive repair reduced conflicts: {} → {}",
 solution.conflicts,
 repaired.conflicts
 );
 Ok(PhaseOutcome::escalate(
 "Thermodynamic annealing failed with conflicts despite repair",
 ))
 }
 }
 Err(e) => {
 log::error!("Phase2: Aggressive repair failed: {}", e);
 Ok(PhaseOutcome::escalate(
 "Thermodynamic annealing failed with high conflicts",
 ))
 }
 }
 }
 }
 }

 fn name(&self) -> &'static str {
 "Phase2-Thermodynamic"
 }

 fn telemetry(&self) -> &dyn PhaseTelemetry {
 self
 }
}

impl Phase2Thermodynamic {
 /// Run thermodynamic annealing on GPU.
 #[cfg(feature = "cuda")]
 fn run_gpu(
 &self,
 graph: &Graph,
 initial: &[usize],
 context: &PhaseContext,
 ) -> Result<ColoringSolution, PrismError> {
 let thermo = self
 .thermodynamic_gpu
 .as_ref()
 .ok_or_else(|| PrismError::gpu("Phase2", "GPU not initialized"))?;

 // Extract geometry stress scalar from context if available
 let stress_scalar = context
 .geometry_metrics
 .as_ref()
 .map(|geo| geo.stress_scalar)
 .unwrap_or(0.0);

 // Adjust temperature based on geometry stress
 let mut adjusted_temp_max = self.temp_max;
 let stress_factor = stress_scalar / 100.0;
 if stress_factor > 0.5 {
 adjusted_temp_max *= 1.0 + (stress_factor - 0.5);
 log::debug!(
 "Phase2: Geometry stress {:.2} → adjusted temp_max to {:.3}",
 stress_scalar,
 adjusted_temp_max
 );
 }

 let start = std::time::Instant::now();

 let colors = thermo
 .run(
 &graph.adjacency,
 graph.num_vertices,
 initial,
 self.num_replicas,
 self.iterations,
 self.temp_min,
 adjusted_temp_max,
 stress_scalar,
 )
 .map_err(|e| PrismError::gpu("Phase2", format!("Thermodynamic GPU failed: {}", e)))?;

 let mut solution = ColoringSolution::new(graph.num_vertices);
 solution.colors = colors;
 solution.compute_chromatic_number();
 solution.conflicts = solution.validate(graph);
 solution.computation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
 solution.quality_score = if solution.conflicts == 0 {
 1.0 / solution.chromatic_number as f64
 } else {
 -(solution.conflicts as f64)
 };

 log::info!(
 "Phase2 GPU: {} colors, {} conflicts, {:.2}ms",
 solution.chromatic_number,
 solution.conflicts,
 solution.computation_time_ms
 );

 Ok(solution)
 }

 /// Run thermodynamic annealing on CPU (greedy fallback).
 fn run_cpu(&self, graph: &Graph, initial: &[usize]) -> ColoringSolution {
 log::warn!("Phase2: Using CPU fallback (greedy coloring)");

 let start = std::time::Instant::now();

 // If initial coloring is valid, use it
 let colors = if self.is_valid_coloring(graph, initial) {
 initial.to_vec()
 } else {
 self.greedy_coloring(graph)
 };

 let mut solution = ColoringSolution::new(graph.num_vertices);
 solution.colors = colors;
 solution.compute_chromatic_number();
 solution.conflicts = solution.validate(graph);
 solution.computation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
 solution.quality_score = if solution.conflicts == 0 {
 1.0 / solution.chromatic_number as f64
 } else {
 -(solution.conflicts as f64)
 };

 log::info!(
 "Phase2 CPU: {} colors, {} conflicts, {:.2}ms",
 solution.chromatic_number,
 solution.conflicts,
 solution.computation_time_ms
 );

 solution
 }

 /// Greedy coloring heuristic (DSATUR variant).
 ///
 /// Colors vertices in order of decreasing saturation degree.
 fn greedy_coloring(&self, graph: &Graph) -> Vec<usize> {
 let mut colors = vec![0; graph.num_vertices];
 let mut colored = vec![false; graph.num_vertices];
 let mut saturation = vec![0; graph.num_vertices]; // Number of different colors in neighbors

 for _ in 0..graph.num_vertices {
 // Find uncolored vertex with max saturation (ties broken by max degree)
 let v = (0..graph.num_vertices)
 .filter(|&v| !colored[v])
 .max_by_key(|&v| (saturation[v], graph.adjacency[v].len()))
 .unwrap();

 // Find smallest available color
 let mut used = vec![false; graph.num_vertices + 1];
 for &neighbor in &graph.adjacency[v] {
 if colors[neighbor] > 0 {
 used[colors[neighbor]] = true;
 }
 }

 let color = (1..=graph.num_vertices)
 .find(|&c| !used[c])
 .unwrap_or(graph.num_vertices);

 colors[v] = color;
 colored[v] = true;

 // Update saturation of neighbors
 for &neighbor in &graph.adjacency[v] {
 if !colored[neighbor] {
 let mut neighbor_colors = std::collections::HashSet::new();
 for &nn in &graph.adjacency[neighbor] {
 if colors[nn] > 0 {
 neighbor_colors.insert(colors[nn]);
 }
 }
 saturation[neighbor] = neighbor_colors.len();
 }
 }
 }

 colors
 }

 /// Check if a coloring is valid (no adjacent vertices share a color).
 fn is_valid_coloring(&self, graph: &Graph, colors: &[usize]) -> bool {
 if colors.len() != graph.num_vertices {
 return false;
 }

 for v in 0..graph.num_vertices {
 for &neighbor in &graph.adjacency[v] {
 if colors[v] == colors[neighbor] && colors[v] > 0 {
 return false;
 }
 }
 }

 true
 }
}

impl PhaseTelemetry for Phase2Thermodynamic {
 fn metrics(&self) -> HashMap<String, f64> {
 let mut m = HashMap::new();
 m.insert("temperature".to_string(), self.temperature);
 m.insert("cooling_rate".to_string(), self.cooling_rate);
 m.insert("compaction_ratio".to_string(), self.compaction_ratio);
 m.insert("guard_triggers".to_string(), self.guard_triggers as f64);
 m.insert(
 "gpu_enabled".to_string(),
 if self.use_gpu { 1.0 } else { 0.0 },
 );
 m.insert("num_replicas".to_string(), self.num_replicas as f64);
 m.insert("iterations".to_string(), self.iterations as f64);
 m.insert("temp_min".to_string(), self.temp_min as f64);
 m.insert("temp_max".to_string(), self.temp_max as f64);
 m
 }

 fn status(&self) -> String {
 format!(
 "Phase2 Thermodynamic: temp=[{:.3}, {:.1}], replicas={}, iters={}, GPU={}",
 self.temp_min, self.temp_max, self.num_replicas, self.iterations, self.use_gpu
 )
 }
}

#[cfg(test)]
mod tests {
 use super::*;

 fn create_triangle_graph() -> Graph {
 Graph {
 num_vertices: 3,
 num_edges: 3,
 adjacency: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
 degrees: None,
 edge_weights: None,
 }
 }

 #[test]
 fn test_greedy_coloring() {
 let phase2 = Phase2Thermodynamic::new();
 let graph = create_triangle_graph();

 let colors = phase2.greedy_coloring(&graph);

 assert_eq!(colors.len(), 3);
 // Verify validity
 assert_ne!(colors[0], colors[1]);
 assert_ne!(colors[1], colors[2]);
 assert_ne!(colors[0], colors[2]);
 }

 #[test]
 fn test_is_valid_coloring() {
 let phase2 = Phase2Thermodynamic::new();
 let graph = create_triangle_graph();

 let valid = vec![1, 2, 3];
 assert!(phase2.is_valid_coloring(&graph, &valid));

 let invalid = vec![1, 1, 2];
 assert!(!phase2.is_valid_coloring(&graph, &invalid));
 }

 #[test]
 fn test_cpu_execution() {
 env_logger::try_init().ok();

 let mut phase2 = Phase2Thermodynamic::new();
 let graph = create_triangle_graph();
 let mut context = PhaseContext::new();

 let result = phase2.execute(&graph, &mut context);
 assert!(result.is_ok());

 let solution = context.best_solution.expect("No solution produced");
 assert_eq!(solution.conflicts, 0);
 assert!(solution.chromatic_number <= 5);
 }

 #[test]
 fn test_rl_action_application() {
 let mut phase2 = Phase2Thermodynamic::new();

 let initial_temp_max = phase2.temp_max;

 // Apply IncreaseTemperature action (index 16)
 phase2.apply_rl_action_raw(16);

 assert!(phase2.temp_max > initial_temp_max);
 }
}
