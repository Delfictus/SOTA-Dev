//! Phase 4/5: Geodesic Distance + FluxNet RL.

use prism_core::{
 ColoringSolution, Graph, PhaseContext, PhaseController, PhaseOutcome, PhaseTelemetry,
 PrismError,
};
use prism_fluxnet::UniversalAction;
use prism_geometry::{generate_spring_layout, GeometrySensorCpu};
use prism_gpu::FloydWarshallGpu;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for Phase 4 Geodesic Distance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase4Config {
 /// Maximum geodesic distance for influence
 #[serde(default = "default_distance_threshold")]
 pub distance_threshold: f64,

 /// Betweenness centrality weight
 #[serde(default = "default_centrality_weight")]
 pub centrality_weight: f64,

 /// Graph diameter penalty coefficient
 #[serde(default = "default_diameter_penalty")]
 pub diameter_penalty: f64,

 /// Use betweenness centrality
 #[serde(default = "default_use_betweenness")]
 pub use_betweenness: bool,

 /// Use closeness centrality
 #[serde(default = "default_use_closeness")]
 pub use_closeness: bool,

 /// Use eigenvector centrality
 #[serde(default = "default_use_eigenvector")]
 pub use_eigenvector: bool,

 /// Enable GPU Floyd-Warshall acceleration
 #[serde(default = "default_gpu_enabled")]
 pub gpu_enabled: bool,

 /// Fraction of vertices to select as structural anchors (0.0 - 1.0)
 #[serde(default = "default_anchor_fraction")]
 pub anchor_fraction: f32,
}

// Default value functions
fn default_distance_threshold() -> f64 {
 3.0
}
fn default_centrality_weight() -> f64 {
 1.0
}
fn default_diameter_penalty() -> f64 {
 0.5
}
fn default_use_betweenness() -> bool {
 true
}
fn default_use_closeness() -> bool {
 false
}
fn default_use_eigenvector() -> bool {
 false
}
fn default_gpu_enabled() -> bool {
 true
}
fn default_anchor_fraction() -> f32 {
 0.10
}

impl Default for Phase4Config {
 fn default() -> Self {
 Self {
 distance_threshold: default_distance_threshold(),
 centrality_weight: default_centrality_weight(),
 diameter_penalty: default_diameter_penalty(),
 use_betweenness: default_use_betweenness(),
 use_closeness: default_use_closeness(),
 use_eigenvector: default_use_eigenvector(),
 gpu_enabled: default_gpu_enabled(),
 anchor_fraction: default_anchor_fraction(),
 }
 }
}

/// Phase 4/5: Geodesic Distance controller.
///
/// Computes shortest path metrics and uses them for vertex ordering:
/// - Floyd-Warshall APSP (All-Pairs Shortest Paths) with GPU acceleration
/// - Betweenness centrality
/// - Graph eccentricity and diameter
///
/// GPU ACCELERATION:
/// - Uses CUDA Floyd-Warshall kernel when GPU available
/// - Automatic CPU fallback for simulation mode or GPU unavailable
/// - Target: DSJC500 (500 vertices) in < 1.5 seconds on RTX 3060
///
/// Resolved TODO(GPU-Phase4): Floyd-Warshall CUDA kernel integrated.
pub struct Phase4Geodesic {
 /// Mean betweenness centrality
 mean_centrality: f64,

 /// Graph diameter (max shortest path)
 diameter: f64,

 /// Use Floyd-Warshall vs Dijkstra
 use_floyd_warshall: bool,

 /// Distance matrix (computed lazily)
 distances_cached: bool,

 /// GPU Floyd-Warshall kernel (None if GPU unavailable)
 gpu_floyd_warshall: Option<Arc<FloydWarshallGpu>>,

 /// Enable GPU acceleration
 use_gpu: bool,

 /// Configuration parameters
 distance_threshold: f64,
 centrality_weight: f64,
 diameter_penalty: f64,
 use_betweenness: bool,
 use_closeness: bool,
 use_eigenvector: bool,
 anchor_fraction: f32,
}

impl Default for Phase4Geodesic {
 fn default() -> Self {
 Self::new()
 }
}

impl Phase4Geodesic {
 pub fn new() -> Self {
 let default_config = Phase4Config::default();
 Self {
 mean_centrality: 0.0,
 diameter: 0.0,
 use_floyd_warshall: true,
 distances_cached: false,
 gpu_floyd_warshall: None,
 use_gpu: false,
 distance_threshold: default_config.distance_threshold,
 centrality_weight: default_config.centrality_weight,
 diameter_penalty: default_config.diameter_penalty,
 use_betweenness: default_config.use_betweenness,
 use_closeness: default_config.use_closeness,
 use_eigenvector: default_config.use_eigenvector,
 anchor_fraction: default_config.anchor_fraction,
 }
 }

 /// Creates Phase4 controller with custom config (CPU mode)
 pub fn with_config(config: Phase4Config) -> Self {
 log::info!(
 "Phase4: Initializing with custom TOML config: distance_threshold={:.2}, centrality_weight={:.2}, anchor_fraction={:.2}",
 config.distance_threshold, config.centrality_weight, config.anchor_fraction
 );

 Self {
 mean_centrality: 0.0,
 diameter: 0.0,
 use_floyd_warshall: true,
 distances_cached: false,
 gpu_floyd_warshall: None,
 use_gpu: false,
 distance_threshold: config.distance_threshold,
 centrality_weight: config.centrality_weight,
 diameter_penalty: config.diameter_penalty,
 use_betweenness: config.use_betweenness,
 use_closeness: config.use_closeness,
 use_eigenvector: config.use_eigenvector,
 anchor_fraction: config.anchor_fraction,
 }
 }

 /// Creates Phase4 controller with custom config and GPU support
 pub fn with_config_and_gpu(config: Phase4Config, ptx_path: &str) -> Self {
 use cudarc::driver::CudaContext;

 log::info!(
 "Phase4: Initializing with custom TOML config and GPU: distance_threshold={:.2}, gpu_enabled={}",
 config.distance_threshold, config.gpu_enabled
 );

 let gpu_floyd_warshall = if config.gpu_enabled {
 match CudaContext::new(0) {
 Ok(context) => match FloydWarshallGpu::new(context, ptx_path) {
 Ok(fw_gpu) => {
 log::info!("Phase4: GPU Floyd-Warshall initialized successfully");
 Some(Arc::new(fw_gpu))
 }
 Err(e) => {
 log::warn!("Phase4: Failed to initialize GPU Floyd-Warshall: {}. Using CPU fallback.", e);
 None
 }
 },
 Err(e) => {
 log::warn!(
 "Phase4: CUDA device not available: {}. Using CPU fallback.",
 e
 );
 None
 }
 }
 } else {
 log::info!("Phase4: GPU disabled by config");
 None
 };

 let use_gpu = gpu_floyd_warshall.is_some();

 Self {
 mean_centrality: 0.0,
 diameter: 0.0,
 use_floyd_warshall: true,
 distances_cached: false,
 gpu_floyd_warshall,
 use_gpu,
 distance_threshold: config.distance_threshold,
 centrality_weight: config.centrality_weight,
 diameter_penalty: config.diameter_penalty,
 use_betweenness: config.use_betweenness,
 use_closeness: config.use_closeness,
 use_eigenvector: config.use_eigenvector,
 anchor_fraction: config.anchor_fraction,
 }
 }

 /// Creates a new Phase 4 controller with GPU support (uses default config)
 ///
 /// Attempts to initialize GPU Floyd-Warshall kernel.
 /// Falls back to CPU if GPU unavailable.
 ///
 /// # Arguments
 /// * `ptx_path` - Path to floyd_warshall.ptx module (e.g., "target/ptx/floyd_warshall.ptx")
 pub fn new_with_gpu(ptx_path: &str) -> Self {
 let default_config = Phase4Config::default();
 Self::with_config_and_gpu(default_config, ptx_path)
 }

 /// Computes all-pairs shortest paths.
 ///
 /// Uses GPU-accelerated Floyd-Warshall kernel if available,
 /// otherwise falls back to CPU implementation.
 ///
 /// Resolved TODO(GPU-Phase4): CUDA Floyd-Warshall kernel integrated.
 fn compute_apsp(&mut self, graph: &Graph) -> Result<Vec<Vec<f64>>, PrismError> {
 let n = graph.num_vertices;

 // Try GPU path first if enabled
 if self.use_gpu && self.gpu_floyd_warshall.is_some() {
 log::debug!("Phase4: Computing APSP on GPU ({} vertices)", n);

 let start = std::time::Instant::now();

 let gpu_result = self
 .gpu_floyd_warshall
 .as_ref()
 .unwrap()
 .compute_apsp(&graph.adjacency, n);

 match gpu_result {
 Ok(dist_f32) => {
 let elapsed = start.elapsed();
 log::info!(
 "Phase4: GPU APSP completed in {:.3}s ({} vertices)",
 elapsed.as_secs_f64(),
 n
 );

 // Convert f32 to f64 for compatibility with rest of phase
 let dist: Vec<Vec<f64>> = dist_f32
 .iter()
 .map(|row| row.iter().map(|&v| v as f64).collect())
 .collect();

 self.distances_cached = true;
 return Ok(dist);
 }
 Err(e) => {
 log::warn!("Phase4: GPU APSP failed: {}. Falling back to CPU.", e);
 // Fall through to CPU implementation
 }
 }
 }

 // CPU fallback implementation
 log::debug!("Phase4: Computing APSP on CPU ({} vertices)", n);
 let start = std::time::Instant::now();

 let mut dist = vec![vec![f64::INFINITY; n]; n];

 // Initialize diagonal and edges
 for i in 0..n {
 dist[i][i] = 0.0;
 for &j in &graph.adjacency[i] {
 dist[i][j] = 1.0;
 }
 }

 if self.use_floyd_warshall {
 // CPU Floyd-Warshall
 for k in 0..n {
 for i in 0..n {
 for j in 0..n {
 if dist[i][k] + dist[k][j] < dist[i][j] {
 dist[i][j] = dist[i][k] + dist[k][j];
 }
 }
 }
 }
 }

 let elapsed = start.elapsed();
 log::info!(
 "Phase4: CPU APSP completed in {:.3}s ({} vertices)",
 elapsed.as_secs_f64(),
 n
 );

 self.distances_cached = true;
 Ok(dist)
 }

 /// Computes betweenness centrality from distance matrix.
 fn compute_centrality(&mut self, distances: &[Vec<f64>]) {
 let n = distances.len();
 let mut centrality_sum = 0.0;
 let mut max_dist = 0.0;

 for i in 0..n {
 for j in 0..n {
 if distances[i][j] != f64::INFINITY && distances[i][j] > max_dist {
 max_dist = distances[i][j];
 }
 if i != j && distances[i][j] != f64::INFINITY {
 centrality_sum += 1.0 / (distances[i][j] + 1.0);
 }
 }
 }

 self.mean_centrality = centrality_sum / (n * (n - 1)) as f64;
 self.diameter = max_dist;

 log::debug!(
 "Phase4: Centrality={:.4}, Diameter={:.1}",
 self.mean_centrality,
 self.diameter
 );
 }

 /// Creates vertex ordering based on centrality.
 fn create_ordering(&self, graph: &Graph, distances: &[Vec<f64>]) -> Vec<usize> {
 let n = graph.num_vertices;
 let mut ordering: Vec<usize> = (0..n).collect();

 // Compute per-vertex centrality
 let mut vertex_centrality: Vec<f64> = vec![0.0; n];
 for i in 0..n {
 for j in 0..n {
 if i != j && distances[i][j] != f64::INFINITY {
 vertex_centrality[i] += 1.0 / (distances[i][j] + 1.0);
 }
 }
 }

 // Sort by centrality (descending)
 ordering.sort_by(|&a, &b| {
 vertex_centrality[b]
 .partial_cmp(&vertex_centrality[a])
 .unwrap()
 });

 ordering
 }

 /// Applies greedy coloring with given ordering.
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

/// Selects structural anchors based on betweenness centrality.
///
/// Returns vertex indices sorted by centrality (highest first).
/// Top `anchor_fraction * n` vertices are selected as anchors for warmstart.
///
/// ## Parameters
/// - `betweenness`: Per-vertex betweenness centrality scores
/// - `anchor_fraction`: Fraction of vertices to select as anchors (0.0 - 1.0)
///
/// ## Returns
/// Vector of vertex indices with highest centrality (sorted descending).
///
/// ## Example
/// ```rust
/// let betweenness = vec![0.5, 0.8, 0.2, 0.9];
/// let anchors = select_structural_anchors(&betweenness, 0.25); // Top 25%
/// // Returns [3, 1] (vertices with centrality 0.9 and 0.8)
/// ```
///
/// Implements PRISM GPU Plan §6 (Warmstart Upgrade), Step 2 (Structural Anchors).
pub fn select_structural_anchors(betweenness: &[f64], anchor_fraction: f32) -> Vec<usize> {
 let n = betweenness.len();
 let num_anchors = ((n as f32 * anchor_fraction).ceil() as usize).max(1).min(n);

 // Create (vertex, centrality) pairs
 let mut vertices: Vec<(usize, f64)> = betweenness
 .iter()
 .enumerate()
 .map(|(i, &c)| (i, c))
 .collect();

 // Sort by centrality (descending)
 vertices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

 // Take top-K vertices
 vertices.iter().take(num_anchors).map(|(v, _)| *v).collect()
}

impl PhaseController for Phase4Geodesic {
 fn execute(
 &mut self,
 graph: &Graph,
 context: &mut PhaseContext,
 ) -> Result<PhaseOutcome, PrismError> {
 log::info!("Phase 4/5: Geodesic Distance executing");

 // Read RL action from context
 if let Some(action) = context
 .scratch
 .get("Phase4_action")
 .and_then(|a| a.downcast_ref::<UniversalAction>())
 {
 if let UniversalAction::Phase4(geo_action) = action {
 log::debug!("Phase4: Applying RL action: {:?}", geo_action);
 // Apply action (e.g., toggle Floyd-Warshall vs Dijkstra)
 }
 }

 // Compute APSP
 let distances = self.compute_apsp(graph)?;

 // Compute centrality and diameter
 self.compute_centrality(&distances);

 // Compute per-vertex betweenness centrality for anchor selection
 let n = graph.num_vertices;
 let mut vertex_centrality: Vec<f64> = vec![0.0; n];
 for i in 0..n {
 for j in 0..n {
 if i != j && distances[i][j] != f64::INFINITY {
 vertex_centrality[i] += 1.0 / (distances[i][j] + 1.0);
 }
 }
 }

 // Apply dendritic reservoir difficulty to boost centrality of hard vertices
 // High difficulty vertices are prioritized for anchor selection
 if context.has_dendritic_metrics() {
 let mean_diff = context.mean_difficulty() as f64;
 for i in 0..n {
 let vertex_diff = context.vertex_difficulty(i) as f64;
 // Boost factor: 1.0 for average difficulty, up to 1.5 for high difficulty
 let boost = 1.0 + (vertex_diff - mean_diff).max(0.0) * 0.5;
 vertex_centrality[i] *= boost;
 }
 log::info!(
 "[Phase4] Dendritic coupling: mean_difficulty={:.3}, centrality boosted for {} high-difficulty vertices",
 mean_diff,
 (0..n).filter(|&i| context.vertex_difficulty(i) as f64 > mean_diff).count()
 );
 }

 // Select structural anchors (for warmstart system)
 let anchors = select_structural_anchors(&vertex_centrality, self.anchor_fraction);

 log::debug!("Phase4: Selected {} structural anchors", anchors.len());

 // Store anchors in context for warmstart stage
 context
 .scratch
 .insert("geodesic_anchors".to_string(), Box::new(anchors.clone()));

 // Compute geometry stress metrics after geodesic computation
 // Use spring layout based on distance matrix for spatial embedding
 let positions = generate_spring_layout(n, &graph.adjacency, 50, context.iteration as u64);
 let geometry_sensor = GeometrySensorCpu::new();

 match geometry_sensor.compute_metrics(&positions, n, &graph.adjacency, &anchors) {
 Ok(geo_metrics) => {
 log::debug!(
 "Phase4 Geometry: bbox_area={:.4}, overlap_density={:.4}, num_hotspots={}, time={:.2}ms",
 geo_metrics.bounding_box.area,
 geo_metrics.mean_overlap_density,
 geo_metrics.num_hotspots,
 geo_metrics.computation_time_ms
 );

 // Convert prism-geometry metrics to prism-core GeometryTelemetry
 let geo_telemetry = prism_core::types::GeometryTelemetry {
 bounding_box_area: geo_metrics.bounding_box.area,
 growth_rate: 0.0, // Will be computed by orchestrator comparing iterations
 overlap_density: geo_metrics.mean_overlap_density as f32,
 stress_scalar: (0.4 * geo_metrics.mean_overlap_density
 + 0.3 * (geo_metrics.bounding_box.area as f64)
 + 0.3 * geo_metrics.mean_curvature_stress)
 as f32,
 anchor_hotspots: (0..n)
 .filter(|&v| {
 // Mark vertices with high overlap as hotspots
 // TODO: Use actual hotspot scores from GPU sensor
 anchors.contains(&v)
 })
 .collect(),
 };

 context.update_geometry_metrics(geo_telemetry);
 }
 Err(e) => {
 log::warn!("Phase4: Geometry stress analysis failed: {}", e);
 // Continue without geometry metrics - not critical for execution
 }
 }

 // Create vertex ordering
 let ordering = self.create_ordering(graph, &distances);

 // Apply greedy coloring
 let solution = self.greedy_coloring(graph, &ordering);

 log::info!(
 "Phase4: Coloring found with {} colors, {} conflicts",
 solution.chromatic_number,
 solution.conflicts
 );

 // Update best solution
 context.update_best_solution(solution);

 // Update RL state metrics
 if let Some(rl_state) = context.rl_state.as_mut() {
 if let Some(state) = rl_state.downcast_mut::<prism_fluxnet::UniversalRLState>() {
 state.geodesic_centrality = self.mean_centrality;
 state.geodesic_diameter = self.diameter;
 }
 }

 Ok(PhaseOutcome::success())
 }

 fn name(&self) -> &'static str {
 "Phase4-Geodesic"
 }

 fn telemetry(&self) -> &dyn PhaseTelemetry {
 self
 }
}

impl PhaseTelemetry for Phase4Geodesic {
 fn metrics(&self) -> HashMap<String, f64> {
 let mut m = HashMap::new();
 m.insert("centrality".to_string(), self.mean_centrality);
 m.insert("diameter".to_string(), self.diameter);
 m.insert(
 "floyd_warshall".to_string(),
 if self.use_floyd_warshall { 1.0 } else { 0.0 },
 );
 m
 }
}
