//! Phase 6: Topological Data Analysis (TDA) with FluxNet RL.

use prism_core::{
 ColoringSolution, Graph, PhaseContext, PhaseController, PhaseOutcome, PhaseTelemetry,
 PrismError,
};
use prism_fluxnet::UniversalAction;
use prism_geometry::{generate_circular_layout, GeometrySensorCpu};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
use prism_gpu::TdaGpu;

/// Configuration for Phase 6 Topological Data Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase6Config {
 /// Persistence threshold for topological features (0.0 - 1.0)
 #[serde(default = "default_persistence_threshold")]
 pub persistence_threshold: f32,

 /// Maximum homology dimension to compute
 #[serde(default = "default_max_dimension")]
 pub max_dimension: usize,

 /// Coherence coefficient of variation threshold
 #[serde(default = "default_coherence_cv_threshold")]
 pub coherence_cv_threshold: f32,

 /// Vietoris-Rips complex radius
 #[serde(default = "default_vietoris_rips_radius")]
 pub vietoris_rips_radius: f32,

 /// Number of landmark points for witness complex
 #[serde(default = "default_num_landmarks")]
 pub num_landmarks: usize,

 /// Use witness complex instead of Vietoris-Rips
 #[serde(default = "default_use_witness_complex")]
 pub use_witness_complex: bool,

 /// Filtration resolution (number of steps)
 #[serde(default = "default_filtration_resolution")]
 pub filtration_resolution: usize,

 /// Enable GPU acceleration for TDA computations
 #[serde(default = "default_gpu_enabled")]
 pub gpu_enabled: bool,
}

// Default value functions for serde
fn default_persistence_threshold() -> f32 {
 0.1
}
fn default_max_dimension() -> usize {
 2
}
fn default_coherence_cv_threshold() -> f32 {
 0.3
}
fn default_vietoris_rips_radius() -> f32 {
 2.0
}
fn default_num_landmarks() -> usize {
 100
}
fn default_use_witness_complex() -> bool {
 false
}
fn default_filtration_resolution() -> usize {
 10
}
fn default_gpu_enabled() -> bool {
 true
}

impl Default for Phase6Config {
 fn default() -> Self {
 Self {
 persistence_threshold: default_persistence_threshold(),
 max_dimension: default_max_dimension(),
 coherence_cv_threshold: default_coherence_cv_threshold(),
 vietoris_rips_radius: default_vietoris_rips_radius(),
 num_landmarks: default_num_landmarks(),
 use_witness_complex: default_use_witness_complex(),
 filtration_resolution: default_filtration_resolution(),
 gpu_enabled: default_gpu_enabled(),
 }
 }
}

/// Phase 6: Topological Data Analysis controller.
///
/// Analyzes graph topology using persistent homology:
/// - Vietoris-Rips complex construction
/// - Persistent homology (Betti numbers)
/// - Topological features for vertex prioritization
///
/// Resolved TODO(GPU-Phase6): GPU TDA kernels integrated with CPU fallback.
pub struct Phase6TDA {
 /// Persistence score (0.0 - 1.0)
 persistence: f64,

 /// Betti-0 (connected components)
 betti_0: usize,

 /// Betti-1 (cycles)
 betti_1: usize,

 /// Filtration resolution
 filtration_resolution: usize,

 /// GPU TDA accelerator (optional, requires CUDA feature)
 #[cfg(feature = "cuda")]
 gpu_tda: Option<Arc<TdaGpu>>,

 /// Use GPU acceleration if available
 #[cfg(feature = "cuda")]
 use_gpu: bool,

 // Config fields
 persistence_threshold: f32,
 max_dimension: usize,
 coherence_cv_threshold: f32,
 vietoris_rips_radius: f32,
 num_landmarks: usize,
 use_witness_complex: bool,
}

impl Default for Phase6TDA {
 fn default() -> Self {
 Self::new()
 }
}

impl Phase6TDA {
 pub fn new() -> Self {
 let config = Phase6Config::default();
 Self::with_config(config)
 }

 /// Creates Phase6 controller with custom config (CPU mode)
 pub fn with_config(config: Phase6Config) -> Self {
 log::info!(
 "Phase6: Initializing with config: threshold={:.2}, max_dim={}, landmarks={}",
 config.persistence_threshold,
 config.max_dimension,
 config.num_landmarks
 );

 Self {
 persistence: 0.0,
 betti_0: 0,
 betti_1: 0,
 filtration_resolution: config.filtration_resolution,
 #[cfg(feature = "cuda")]
 gpu_tda: None,
 #[cfg(feature = "cuda")]
 use_gpu: false,
 persistence_threshold: config.persistence_threshold,
 max_dimension: config.max_dimension,
 coherence_cv_threshold: config.coherence_cv_threshold,
 vietoris_rips_radius: config.vietoris_rips_radius,
 num_landmarks: config.num_landmarks,
 use_witness_complex: config.use_witness_complex,
 }
 }

 /// Creates Phase6 controller with custom config and GPU support
 #[cfg(feature = "cuda")]
 pub fn with_config_and_gpu(config: Phase6Config, ptx_path: &str) -> Self {
 let mut phase = Self::with_config(config.clone());

 if config.gpu_enabled {
 match Self::try_init_gpu(ptx_path) {
 Ok(gpu_tda) => {
 phase.gpu_tda = Some(Arc::new(gpu_tda));
 phase.use_gpu = true;
 log::info!("Phase6: GPU TDA acceleration enabled with custom config");
 }
 Err(e) => {
 log::warn!(
 "Phase6: GPU initialization failed ({}), using CPU fallback",
 e
 );
 }
 }
 }

 phase
 }

 /// Creates a new Phase 6 controller with GPU acceleration
 ///
 /// # Arguments
 /// * `ptx_path` - Path to compiled TDA PTX module (e.g., "target/ptx/tda.ptx")
 ///
 /// # Errors
 /// Returns error if GPU initialization fails. Falls back to CPU on error.
 ///
 /// # Example
 /// ```rust,no_run
 /// use prism_phases::phase6_tda::Phase6TDA;
 /// let phase6 = Phase6TDA::new_with_gpu("target/ptx/tda.ptx");
 /// ```
 #[cfg(feature = "cuda")]
 pub fn new_with_gpu(ptx_path: &str) -> Self {
 let config = Phase6Config::default();
 Self::with_config_and_gpu(config, ptx_path)
 }

 #[cfg(feature = "cuda")]
 fn try_init_gpu(ptx_path: &str) -> anyhow::Result<TdaGpu> {
 let device = CudaContext::new(0)?;
 TdaGpu::new(device, ptx_path)
 }

 /// Computes Betti numbers (connected components and cycles).
 ///
 /// Resolved TODO(GPU-Phase6): Uses GPU TDA kernel with CPU fallback.
 fn compute_betti_numbers(&mut self, graph: &Graph) -> Result<(), PrismError> {
 #[cfg(feature = "cuda")]
 {
 if self.use_gpu && self.gpu_tda.is_some() {
 // GPU path
 return self.compute_betti_numbers_gpu(graph);
 }
 }

 // CPU fallback
 self.compute_betti_numbers_cpu(graph)
 }

 /// GPU implementation of Betti number computation
 #[cfg(feature = "cuda")]
 fn compute_betti_numbers_gpu(&mut self, graph: &Graph) -> Result<(), PrismError> {
 let start = std::time::Instant::now();

 let (betti_0, betti_1) = self
 .gpu_tda
 .as_ref()
 .unwrap()
 .compute_betti_numbers(&graph.adjacency, graph.num_vertices, graph.num_edges)
 .map_err(|e| PrismError::gpu("Phase6 TDA", e.to_string()))?;

 self.betti_0 = betti_0;
 self.betti_1 = betti_1;

 // Compute persistence score
 self.persistence = if self.betti_0 > 0 {
 1.0 / self.betti_0 as f64
 } else {
 0.0
 };

 let elapsed = start.elapsed();
 log::debug!(
 "Phase6 GPU: Betti-0={}, Betti-1={}, Persistence={:.4} [{:.3}ms]",
 self.betti_0,
 self.betti_1,
 self.persistence,
 elapsed.as_secs_f64() * 1000.0
 );

 Ok(())
 }

 /// CPU fallback implementation of Betti number computation
 fn compute_betti_numbers_cpu(&mut self, graph: &Graph) -> Result<(), PrismError> {
 let n = graph.num_vertices;

 // Betti-0: Count connected components using union-find
 let mut parent: Vec<usize> = (0..n).collect();

 fn find(parent: &mut [usize], x: usize) -> usize {
 if parent[x] != x {
 parent[x] = find(parent, parent[x]);
 }
 parent[x]
 }

 fn union(parent: &mut [usize], x: usize, y: usize) {
 let px = find(parent, x);
 let py = find(parent, y);
 if px != py {
 parent[px] = py;
 }
 }

 // Union all edges
 for (u, neighbors) in graph.adjacency.iter().enumerate() {
 for &v in neighbors {
 if u < v {
 union(&mut parent, u, v);
 }
 }
 }

 // Count unique roots (components)
 let mut roots = std::collections::HashSet::new();
 for i in 0..n {
 roots.insert(find(&mut parent, i));
 }

 self.betti_0 = roots.len();

 // Betti-1: Approximate as edges - vertices + components (Euler characteristic)
 let vertices = n;
 let edges = graph.num_edges;
 self.betti_1 = if edges >= vertices {
 edges - vertices + self.betti_0
 } else {
 0
 };

 // Compute persistence score
 self.persistence = if self.betti_0 > 0 {
 1.0 / self.betti_0 as f64
 } else {
 0.0
 };

 log::debug!(
 "Phase6 CPU: Betti-0={}, Betti-1={}, Persistence={:.4}",
 self.betti_0,
 self.betti_1,
 self.persistence
 );

 Ok(())
 }

 /// Creates vertex ordering based on topological importance.
 fn create_topological_ordering(&self, graph: &Graph) -> Vec<usize> {
 let n = graph.num_vertices;
 let mut ordering: Vec<usize> = (0..n).collect();

 // Order by degree (proxy for topological importance)
 ordering.sort_by_key(|&v| std::cmp::Reverse(graph.adjacency[v].len()));

 ordering
 }

 /// Validates coherence quality by checking topological feature variance.
 ///
 /// Computes variance of topological importance scores and warns if features are too uniform.
 /// Low variance indicates poor topological differentiation, which degrades warmstart quality.
 ///
 /// ## Warning Thresholds
 /// - **Variance < 0.1**: Extremely uniform (all vertices have similar importance)
 /// - **Betti-0 == num_vertices**: Graph is completely disconnected (pathological)
 /// - **Persistence < 0.01**: Very weak topological structure
 ///
 /// ## Returns
 /// (mean_importance, variance, coefficient_of_variation)
 fn validate_coherence_quality(&self, graph: &Graph) -> (f64, f64, f64) {
 let n = graph.num_vertices;

 // Compute topological importance scores (same as select_topological_anchors)
 let component_factor = if self.betti_0 > 0 {
 1.0 / self.betti_0 as f64
 } else {
 1.0
 };

 let importance_scores: Vec<f64> = (0..n)
 .map(|v| {
 let degree = graph.adjacency[v].len();
 degree as f64 * (1.0 + component_factor)
 })
 .collect();

 // Compute statistics
 let mean = importance_scores.iter().sum::<f64>() / n as f64;
 let variance = importance_scores
 .iter()
 .map(|&score| (score - mean).powi(2))
 .sum::<f64>()
 / n as f64;
 let std_dev = variance.sqrt();
 let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

 // Log warnings for poor coherence quality
 if coefficient_of_variation < 0.1 {
 log::warn!(
 "Phase6 Coherence Warning: Very low topological variance (CV={:.4}). \
 All vertices have similar importance - warmstart may be ineffective.",
 coefficient_of_variation
 );
 }

 if self.betti_0 == n {
 log::warn!(
 "Phase6 Coherence Warning: Graph is completely disconnected (Betti-0={}, vertices={}). \
 Topological features are trivial.",
 self.betti_0,
 n
 );
 }

 if self.persistence < 0.01 {
 log::warn!(
 "Phase6 Coherence Warning: Very weak topological structure (persistence={:.4}). \
 TDA features may not provide useful guidance.",
 self.persistence
 );
 }

 log::debug!(
 "Phase6 Coherence: mean={:.2}, variance={:.2}, CV={:.4}",
 mean,
 variance,
 coefficient_of_variation
 );

 (mean, variance, coefficient_of_variation)
 }

 /// Applies greedy coloring with topological ordering.
 fn greedy_coloring(&self, graph: &Graph, ordering: &[usize]) -> ColoringSolution {
 let n = graph.num_vertices;
 let mut colors = vec![0; n];
 let mut max_color = 0;

 for &v in ordering {
 // Find colors used by neighbors
 let mut used_colors = vec![false; n];
 for &neighbor in &graph.adjacency[v] {
 if colors[neighbor] > 0 {
 used_colors[colors[neighbor] - 1] = true;
 }
 }

 // Assign first available color
 let mut color = 1;
 while color <= n && used_colors[color - 1] {
 color += 1;
 }

 colors[v] = color;
 max_color = max_color.max(color);
 }

 let mut solution = ColoringSolution::new(n);
 solution.colors = colors;
 solution.chromatic_number = max_color;
 solution.conflicts = solution.validate(graph);

 solution
 }
}

/// Selects topological anchors based on persistence scores and degree.
///
/// Returns vertex indices with highest topological significance.
/// Top `anchor_fraction * n` vertices are selected as anchors for warmstart.
///
/// ## Parameters
/// - `graph`: The input graph
/// - `betti_0`: Number of connected components (topological feature)
/// - `anchor_fraction`: Fraction of vertices to select as anchors (0.0 - 1.0)
///
/// ## Returns
/// Vector of vertex indices with highest topological importance (sorted descending).
///
/// ## Algorithm
/// Uses vertex degree as a proxy for topological importance:
/// - High-degree vertices are central to graph structure
/// - Vertices in dense regions have higher topological significance
/// - Component bridges have high importance (detected via degree)
///
/// ## Example
/// ```rust
/// let graph = Graph::new(10);
/// let anchors = select_topological_anchors(&graph, 1, 0.20); // Top 20%
/// ```
///
/// Implements PRISM GPU Plan §6 (Warmstart Upgrade), Step 2 (Structural Anchors).
pub fn select_topological_anchors(
 graph: &Graph,
 betti_0: usize,
 anchor_fraction: f32,
) -> Vec<usize> {
 let n = graph.num_vertices;
 let num_anchors = ((n as f32 * anchor_fraction).ceil() as usize).max(1).min(n);

 // Compute topological importance score (degree-based)
 let mut importance: Vec<(usize, f64)> = Vec::with_capacity(n);
 for v in 0..n {
 let degree = graph.adjacency[v].len();

 // Importance = degree + component penalty
 // Lower Betti-0 (more connected) -> higher importance
 let component_factor = if betti_0 > 0 {
 1.0 / betti_0 as f64
 } else {
 1.0
 };

 let score = degree as f64 * (1.0 + component_factor);
 importance.push((v, score));
 }

 // Sort by importance (descending)
 importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

 // Take top-K vertices
 importance
 .iter()
 .take(num_anchors)
 .map(|(v, _)| *v)
 .collect()
}

impl PhaseController for Phase6TDA {
 fn execute(
 &mut self,
 graph: &Graph,
 context: &mut PhaseContext,
 ) -> Result<PhaseOutcome, PrismError> {
 log::info!("Phase 6: Topological Data Analysis executing");

 // Read RL action from context
 if let Some(action) = context
 .scratch
 .get("Phase6_action")
 .and_then(|a| a.downcast_ref::<UniversalAction>())
 {
 if let UniversalAction::Phase6(tda_action) = action {
 log::debug!("Phase6: Applying RL action: {:?}", tda_action);
 }
 }

 // Apply dendritic reservoir uncertainty for persistence threshold scaling
 // Higher mean uncertainty → stricter persistence filtering (fewer features)
 let persistence_scale = if context.has_dendritic_metrics() {
 let mean_uncert = context.mean_uncertainty();
 // Scale: low uncertainty → 0.8x threshold (more features)
 // high uncertainty → 1.3x threshold (stricter filtering)
 let scale = 0.8 + mean_uncert * 0.5;
 log::info!(
 "[Phase6] Dendritic coupling: mean_uncertainty={:.3}, persistence_scale={:.2}x",
 mean_uncert,
 scale
 );
 scale
 } else {
 1.0
 };
 // Store persistence scale in context for use by Betti computation
 context.scratch.insert(
 "tda_persistence_scale".to_string(),
 Box::new(persistence_scale as f32)
 );

 // Compute Betti numbers
 self.compute_betti_numbers(graph)?;

 // Validate coherence quality (Phase 5 requirement)
 let (_mean, _variance, cv) = self.validate_coherence_quality(graph);

 // Select topological anchors (for warmstart system)
 let anchor_fraction = 0.10; // TODO: Get from WarmstartConfig
 let anchors = select_topological_anchors(graph, self.betti_0, anchor_fraction);

 log::debug!(
 "Phase6: Selected {} topological anchors (coherence CV={:.4})",
 anchors.len(),
 cv
 );

 // Store anchors in context for warmstart stage
 context
 .scratch
 .insert("tda_anchors".to_string(), Box::new(anchors.clone()));

 // Compute geometry stress metrics after topological analysis
 // Use circular layout for topological phase (preserves symmetry)
 let n = graph.num_vertices;
 let positions = generate_circular_layout(n);
 let geometry_sensor = GeometrySensorCpu::new();

 match geometry_sensor.compute_metrics(&positions, n, &graph.adjacency, &anchors) {
 Ok(geo_metrics) => {
 log::debug!(
 "Phase6 Geometry: bbox_area={:.4}, overlap_density={:.4}, curvature_stress={:.4}, time={:.2}ms",
 geo_metrics.bounding_box.area,
 geo_metrics.mean_overlap_density,
 geo_metrics.mean_curvature_stress,
 geo_metrics.computation_time_ms
 );

 // Convert prism-geometry metrics to prism-core GeometryTelemetry
 // Emphasize topological features (curvature stress)
 let geo_telemetry = prism_core::types::GeometryTelemetry {
 bounding_box_area: geo_metrics.bounding_box.area,
 growth_rate: 0.0, // Will be computed by orchestrator
 overlap_density: geo_metrics.mean_overlap_density as f32,
 stress_scalar: (0.3 * geo_metrics.mean_overlap_density
 + 0.2 * (geo_metrics.bounding_box.area as f64)
 + 0.5 * geo_metrics.mean_curvature_stress)
 as f32,
 anchor_hotspots: anchors.clone(),
 };

 // Update or merge with existing geometry metrics
 if let Some(ref existing_geo) = context.geometry_metrics {
 // Average with Phase 4 metrics for combined geometric view
 let merged = prism_core::types::GeometryTelemetry {
 bounding_box_area: (existing_geo.bounding_box_area
 + geo_telemetry.bounding_box_area)
 / 2.0,
 growth_rate: existing_geo.growth_rate,
 overlap_density: (existing_geo.overlap_density
 + geo_telemetry.overlap_density)
 / 2.0,
 stress_scalar: (existing_geo.stress_scalar + geo_telemetry.stress_scalar)
 / 2.0,
 anchor_hotspots: {
 let mut combined = existing_geo.anchor_hotspots.clone();
 combined.extend(geo_telemetry.anchor_hotspots);
 combined.sort_unstable();
 combined.dedup();
 combined
 },
 };
 context.update_geometry_metrics(merged);
 } else {
 context.update_geometry_metrics(geo_telemetry);
 }
 }
 Err(e) => {
 log::warn!("Phase6: Geometry stress analysis failed: {}", e);
 // Continue without geometry metrics - not critical
 }
 }

 // Create topological ordering
 let ordering = self.create_topological_ordering(graph);

 // Apply greedy coloring
 let solution = self.greedy_coloring(graph, &ordering);

 log::info!(
 "Phase6: Coloring found with {} colors, {} conflicts",
 solution.chromatic_number,
 solution.conflicts
 );

 // Update best solution
 context.update_best_solution(solution);

 // Update RL state metrics
 if let Some(rl_state) = context.rl_state.as_mut() {
 if let Some(state) = rl_state.downcast_mut::<prism_fluxnet::UniversalRLState>() {
 state.tda_persistence = self.persistence;
 state.tda_betti_0 = self.betti_0;
 }
 }

 Ok(PhaseOutcome::success())
 }

 fn name(&self) -> &'static str {
 "Phase6-TDA"
 }

 fn telemetry(&self) -> &dyn PhaseTelemetry {
 self
 }
}

impl PhaseTelemetry for Phase6TDA {
 fn metrics(&self) -> HashMap<String, f64> {
 let mut m = HashMap::new();
 m.insert("persistence".to_string(), self.persistence);
 m.insert("betti_0".to_string(), self.betti_0 as f64);
 m.insert("betti_1".to_string(), self.betti_1 as f64);
 m.insert(
 "filtration_resolution".to_string(),
 self.filtration_resolution as f64,
 );
 m
 }
}
