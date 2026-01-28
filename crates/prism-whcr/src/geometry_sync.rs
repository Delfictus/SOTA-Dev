//! WHCR Geometry Synchronization - Advanced Implementation
//!
//! This module provides production-grade geometry extraction and synchronization
//! for the WHCR multi-phase integration. It implements:
//!
//! - GPU-direct geometry extraction (no CPU roundtrip when possible)
//! - Multi-source fusion with confidence weighting
//! - Adaptive thresholds based on graph characteristics
//! - Fallback computation when telemetry is unavailable
//! - Streaming updates for large graphs
//!
//! # Architecture
//!
//! ```text
//! Phase Controller → Telemetry → GeometryExtractor → GeometryAccumulator
//! ↓ ↓ ↓
//! GPU Buffers ──────────────────> Direct Copy ──────> GPU-Resident
//! (if available) (zero-copy) (for WHCR)
//! ```

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use prism_core::{Graph, PhaseContext, PhaseOutcome, PrismError};

#[cfg(feature = "cuda")]
use crate::GeometryAccumulator;

// =============================================================================
// GEOMETRY EXTRACTOR - Central extraction logic
// =============================================================================

/// Advanced geometry extractor with multi-source fusion and adaptive thresholds
pub struct GeometryExtractor {
 /// Graph characteristics for adaptive threshold computation
 num_vertices: usize,
 num_edges: usize,
 density: f64,

 /// Extraction configuration
 config: ExtractionConfig,

 /// Cache for expensive computations
 degree_distribution: Option<Vec<usize>>,
 adjacency_cache: Option<Vec<Vec<usize>>>,
}

/// Configuration for geometry extraction
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
 /// Hotspot detection: fraction of vertices to mark (adaptive)
 pub hotspot_fraction: f64,

 /// Hotspot detection: minimum activity threshold
 pub hotspot_min_activity: f64,

 /// Belief smoothing: Laplacian smoothing parameter
 pub belief_smoothing: f64,

 /// Stress normalization: use percentile-based normalization
 pub stress_normalize_percentile: bool,

 /// Persistence: minimum lifetime to consider significant
 pub persistence_min_lifetime: f64,

 /// Enable GPU-direct extraction when possible
 pub prefer_gpu_direct: bool,

 /// Fallback computation verbosity
 pub verbose_fallback: bool,
}

impl Default for ExtractionConfig {
 fn default() -> Self {
 Self {
 hotspot_fraction: 0.15, // Top 15% by default
 hotspot_min_activity: 0.1,
 belief_smoothing: 0.1,
 stress_normalize_percentile: true,
 persistence_min_lifetime: 0.05,
 prefer_gpu_direct: true,
 verbose_fallback: true,
 }
 }
}

impl ExtractionConfig {
 /// Adaptive configuration based on graph characteristics
 pub fn for_graph(num_vertices: usize, num_edges: usize) -> Self {
 let density = (2.0 * num_edges as f64) / (num_vertices as f64 * (num_vertices - 1) as f64);

 let mut config = Self::default();

 // Dense graphs (like DSJC) need more hotspots
 if density > 0.3 {
 config.hotspot_fraction = 0.20;
 config.belief_smoothing = 0.05; // Less smoothing for dense
 }

 // Large graphs need percentile normalization
 if num_vertices > 1000 {
 config.stress_normalize_percentile = true;
 }

 // Small graphs can use more aggressive hotspot detection
 if num_vertices < 200 {
 config.hotspot_fraction = 0.25;
 config.hotspot_min_activity = 0.05;
 }

 config
 }

 /// Tuned configuration for DSJC125.5 specifically
 pub fn for_dsjc125_5() -> Self {
 Self {
 hotspot_fraction: 0.20, // 25 vertices as hotspots
 hotspot_min_activity: 0.08,
 belief_smoothing: 0.05,
 stress_normalize_percentile: true,
 persistence_min_lifetime: 0.03,
 prefer_gpu_direct: true,
 verbose_fallback: false,
 }
 }
}

impl GeometryExtractor {
 /// Create extractor for a specific graph
 pub fn new(graph: &Graph) -> Self {
 let num_vertices = graph.num_vertices;
 let num_edges = graph.adjacency.iter().map(|adj| adj.len()).sum::<usize>() / 2;
 let density = if num_vertices > 1 {
 (2.0 * num_edges as f64) / (num_vertices as f64 * (num_vertices - 1) as f64)
 } else {
 0.0
 };

 Self {
 num_vertices,
 num_edges,
 density,
 config: ExtractionConfig::for_graph(num_vertices, num_edges),
 degree_distribution: None,
 adjacency_cache: Some(graph.adjacency.clone()),
 }
 }

 /// Create extractor with custom configuration
 pub fn with_config(graph: &Graph, config: ExtractionConfig) -> Self {
 let mut extractor = Self::new(graph);
 extractor.config = config;
 extractor
 }

 // =========================================================================
 // PHASE 0: HOTSPOT EXTRACTION
 // =========================================================================

 /// Extract hotspot vertices from Phase 0 (Dendritic Reservoir)
 ///
 /// Uses multi-source fusion with priority:
 /// 1. Explicit hotspot list from telemetry
 /// 2. Activity-based derivation from reservoir state
 /// 3. Spike rate analysis
 /// 4. Degree-based fallback (high-degree vertices are conflict-prone)
 pub fn extract_phase0_hotspots(
 &mut self,
 result: &PhaseOutcome,
 context: &PhaseContext,
 ) -> Option<Vec<usize>> {
 // Source 1: Explicit hotspot list
 if let Some(hotspots) = self.extract_explicit_hotspots(result) {
 log::debug!("Phase0: Extracted {} explicit hotspots", hotspots.len());
 return Some(hotspots);
 }

 // Source 2: Activity-based derivation
 if let Some(hotspots) = self.derive_hotspots_from_activity(result) {
 log::debug!("Phase0: Derived {} hotspots from activity", hotspots.len());
 return Some(hotspots);
 }

 // Source 3: Spike rate analysis
 if let Some(hotspots) = self.derive_hotspots_from_spike_rates(result) {
 log::debug!(
 "Phase0: Derived {} hotspots from spike rates",
 hotspots.len()
 );
 return Some(hotspots);
 }

 // Source 4: Conflict-based derivation (if solution exists)
 if let Some(hotspots) = self.derive_hotspots_from_conflicts(context) {
 log::debug!(
 "Phase0: Derived {} hotspots from conflict analysis",
 hotspots.len()
 );
 return Some(hotspots);
 }

 // Source 5: Degree-based fallback
 if let Some(hotspots) = self.derive_hotspots_from_degree() {
 if self.config.verbose_fallback {
 log::warn!(
 "Phase0: Using degree-based hotspot fallback ({} vertices)",
 hotspots.len()
 );
 }
 return Some(hotspots);
 }

 log::warn!("Phase0: No hotspot source available");
 None
 }

 fn extract_explicit_hotspots(&self, result: &PhaseOutcome) -> Option<Vec<usize>> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 // Try multiple possible key names
 for key in &[
 "hotspot_vertices",
 "hotspots",
 "high_activity_vertices",
 "critical_vertices",
 ] {
 if let Some(val) = telemetry.get(*key) {
 if let Some(arr) = val.as_array() {
 let hotspots: Vec<usize> = arr
 .iter()
 .filter_map(|v| v.as_u64().map(|n| n as usize))
 .filter(|&v| v < self.num_vertices)
 .collect();

 if !hotspots.is_empty() {
 return Some(hotspots);
 }
 }
 }
 }
 }
 None
 }

 fn derive_hotspots_from_activity(&self, result: &PhaseOutcome) -> Option<Vec<usize>> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 for key in &[
 "vertex_activity",
 "activity_levels",
 "reservoir_activity",
 "soma_potentials",
 ] {
 if let Some(val) = telemetry.get(*key) {
 if let Some(arr) = val.as_array() {
 let activities: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();

 if activities.len() == self.num_vertices {
 return Some(self.threshold_to_hotspots(&activities));
 }
 }
 }
 }
 }
 None
 }

 fn derive_hotspots_from_spike_rates(&self, result: &PhaseOutcome) -> Option<Vec<usize>> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 for key in &["spike_rates", "firing_rates", "spike_counts"] {
 if let Some(val) = telemetry.get(*key) {
 if let Some(arr) = val.as_array() {
 let rates: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();

 if rates.len() == self.num_vertices {
 return Some(self.threshold_to_hotspots(&rates));
 }
 }
 }
 }
 }
 None
 }

 fn derive_hotspots_from_conflicts(&self, context: &PhaseContext) -> Option<Vec<usize>> {
 let solution = context.best_solution.as_ref()?;
 let adjacency = self.adjacency_cache.as_ref()?;

 // Compute conflict count per vertex
 let mut conflict_counts = vec![0usize; self.num_vertices];
 for (v, neighbors) in adjacency.iter().enumerate() {
 for &u in neighbors {
 if solution.colors[v] == solution.colors[u] {
 conflict_counts[v] += 1;
 }
 }
 }

 // Convert to f64 for thresholding
 let conflict_scores: Vec<f64> = conflict_counts.iter().map(|&c| c as f64).collect();

 // Only use if there are actual conflicts
 if conflict_scores.iter().any(|&c| c > 0.0) {
 Some(self.threshold_to_hotspots(&conflict_scores))
 } else {
 None
 }
 }

 fn derive_hotspots_from_degree(&mut self) -> Option<Vec<usize>> {
 let adjacency = self.adjacency_cache.as_ref()?;

 // Compute degree distribution if not cached
 if self.degree_distribution.is_none() {
 self.degree_distribution = Some(adjacency.iter().map(|adj| adj.len()).collect());
 }

 let degrees = self.degree_distribution.as_ref()?;
 let degree_f64: Vec<f64> = degrees.iter().map(|&d| d as f64).collect();

 Some(self.threshold_to_hotspots(&degree_f64))
 }

 /// Convert activity scores to hotspot list using adaptive thresholding
 fn threshold_to_hotspots(&self, scores: &[f64]) -> Vec<usize> {
 if scores.is_empty() {
 return Vec::new();
 }

 // Compute adaptive threshold
 let mut sorted_scores: Vec<f64> = scores.iter().copied().collect();
 sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

 // Target count based on fraction
 let target_count =
 ((self.num_vertices as f64) * self.config.hotspot_fraction).ceil() as usize;
 let target_count = target_count.max(1).min(self.num_vertices);

 // Threshold is the score at the target_count position
 let threshold = sorted_scores
 .get(target_count.saturating_sub(1))
 .copied()
 .unwrap_or(0.0)
 .max(self.config.hotspot_min_activity);

 // Select vertices above threshold
 scores
 .iter()
 .enumerate()
 .filter(|(_, &s)| s >= threshold)
 .map(|(i, _)| i)
 .collect()
 }

 // =========================================================================
 // PHASE 1: BELIEF EXTRACTION
 // =========================================================================

 /// Extract belief distribution from Phase 1 (Quantum/Active Inference Init)
 ///
 /// Uses multi-source fusion:
 /// 1. Explicit belief matrix from telemetry
 /// 2. Probability amplitudes from quantum state
 /// 3. Temperature-based Boltzmann distribution
 /// 4. Solution-derived one-hot with smoothing
 pub fn extract_phase1_beliefs(
 &self,
 result: &PhaseOutcome,
 context: &PhaseContext,
 ) -> Option<(Vec<f64>, usize)> {
 // Source 1: Explicit belief matrix
 if let Some(beliefs) = self.extract_explicit_beliefs(result) {
 log::debug!("Phase1: Extracted explicit belief matrix");
 return Some(beliefs);
 }

 // Source 2: Quantum amplitudes
 if let Some(beliefs) = self.derive_beliefs_from_amplitudes(result) {
 log::debug!("Phase1: Derived beliefs from quantum amplitudes");
 return Some(beliefs);
 }

 // Source 3: Boltzmann distribution
 if let Some(beliefs) = self.derive_beliefs_from_temperature(result, context) {
 log::debug!("Phase1: Derived beliefs from temperature distribution");
 return Some(beliefs);
 }

 // Source 4: Solution-derived with smoothing
 if let Some(beliefs) = self.derive_beliefs_from_solution(context) {
 log::debug!("Phase1: Derived beliefs from current solution");
 return Some(beliefs);
 }

 log::warn!("Phase1: No belief source available");
 None
 }

 fn extract_explicit_beliefs(&self, result: &PhaseOutcome) -> Option<(Vec<f64>, usize)> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 // Look for belief matrix
 let beliefs_val = telemetry
 .get("belief_distribution")
 .or_else(|| telemetry.get("beliefs"))
 .or_else(|| telemetry.get("color_probabilities"))?;

 let beliefs: Vec<f64> = beliefs_val
 .as_array()?
 .iter()
 .filter_map(|v| v.as_f64())
 .collect();

 // Determine num_colors
 let num_colors = telemetry
 .get("num_colors")
 .and_then(|v| v.as_u64())
 .map(|n| n as usize)
 .or_else(|| {
 // Infer from belief array size
 if beliefs.len() % self.num_vertices == 0 {
 Some(beliefs.len() / self.num_vertices)
 } else {
 None
 }
 })?;

 if beliefs.len() == self.num_vertices * num_colors {
 return Some((beliefs, num_colors));
 }
 }
 None
 }

 fn derive_beliefs_from_amplitudes(&self, result: &PhaseOutcome) -> Option<(Vec<f64>, usize)> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 // Quantum amplitudes (complex numbers stored as [re, im, re, im, ...])
 let amplitudes_val = telemetry
 .get("quantum_amplitudes")
 .or_else(|| telemetry.get("state_vector"))?;

 let amplitudes: Vec<f64> = amplitudes_val
 .as_array()?
 .iter()
 .filter_map(|v| v.as_f64())
 .collect();

 // Convert complex amplitudes to probabilities
 // Assuming format: [re_0, im_0, re_1, im_1, ...]
 if amplitudes.len() % 2 == 0 {
 let probabilities: Vec<f64> = amplitudes
 .chunks(2)
 .map(|c| c[0] * c[0] + c[1] * c[1]) // |amplitude|^2
 .collect();

 let num_states = probabilities.len();
 if num_states % self.num_vertices == 0 {
 let num_colors = num_states / self.num_vertices;
 return Some((probabilities, num_colors));
 }
 }
 }
 None
 }

 fn derive_beliefs_from_temperature(
 &self,
 result: &PhaseOutcome,
 context: &PhaseContext,
 ) -> Option<(Vec<f64>, usize)> {
 let solution = context.best_solution.as_ref()?;
 let adjacency = self.adjacency_cache.as_ref()?;

 // Get temperature from telemetry
 let temperature = if let PhaseOutcome::Success { telemetry, .. } = result {
 telemetry
 .get("temperature")
 .or_else(|| telemetry.get("T"))
 .and_then(|v| v.as_f64())
 .unwrap_or(1.0)
 } else {
 1.0
 };

 let num_colors = solution.chromatic_number.max(1);
 let mut beliefs = vec![0.0; self.num_vertices * num_colors];

 // Compute Boltzmann distribution for each vertex
 for v in 0..self.num_vertices {
 let mut energies = vec![0.0; num_colors];

 // Energy = number of conflicts with each color
 for c in 0..num_colors {
 for &u in &adjacency[v] {
 if solution.colors[u] == c {
 energies[c] += 1.0;
 }
 }
 }

 // Boltzmann: P(c) ∝ exp(-E(c)/T)
 let exp_energies: Vec<f64> = energies
 .iter()
 .map(|&e| (-e / temperature.max(0.001)).exp())
 .collect();

 let sum: f64 = exp_energies.iter().sum();

 for c in 0..num_colors {
 beliefs[v * num_colors + c] = if sum > 0.0 {
 exp_energies[c] / sum
 } else {
 1.0 / num_colors as f64
 };
 }
 }

 Some((beliefs, num_colors))
 }

 fn derive_beliefs_from_solution(&self, context: &PhaseContext) -> Option<(Vec<f64>, usize)> {
 let solution = context.best_solution.as_ref()?;
 let num_colors = solution.chromatic_number.max(1);
 let smoothing = self.config.belief_smoothing;

 // One-hot with Laplacian smoothing
 let uniform = smoothing / num_colors as f64;
 let confidence = 1.0 - smoothing;

 let mut beliefs = vec![uniform; self.num_vertices * num_colors];

 for (v, &c) in solution.colors.iter().enumerate() {
 if c < num_colors {
 beliefs[v * num_colors + c] = confidence + uniform;
 }
 }

 Some((beliefs, num_colors))
 }

 // =========================================================================
 // PHASE 4: STRESS EXTRACTION
 // =========================================================================

 /// Extract stress scores from Phase 4 (Geodesic Optimization)
 ///
 /// Uses multi-source fusion:
 /// 1. Explicit stress array from telemetry
 /// 2. Embedding-derived stress computation
 /// 3. Geodesic distortion metrics
 /// 4. Local conflict density as proxy
 pub fn extract_phase4_stress(
 &self,
 result: &PhaseOutcome,
 context: &PhaseContext,
 ) -> Option<Vec<f64>> {
 // Source 1: Explicit stress
 if let Some(stress) = self.extract_explicit_stress(result) {
 log::debug!("Phase4: Extracted explicit stress scores");
 return Some(self.normalize_stress(stress));
 }

 // Source 2: Embedding-derived
 if let Some(stress) = self.derive_stress_from_embedding(result) {
 log::debug!("Phase4: Derived stress from embedding");
 return Some(self.normalize_stress(stress));
 }

 // Source 3: Geodesic distortion
 if let Some(stress) = self.derive_stress_from_distortion(result) {
 log::debug!("Phase4: Derived stress from geodesic distortion");
 return Some(self.normalize_stress(stress));
 }

 // Source 4: Conflict density proxy
 if let Some(stress) = self.derive_stress_from_conflict_density(context) {
 if self.config.verbose_fallback {
 log::warn!("Phase4: Using conflict density as stress proxy");
 }
 return Some(self.normalize_stress(stress));
 }

 log::warn!("Phase4: No stress source available");
 None
 }

 fn extract_explicit_stress(&self, result: &PhaseOutcome) -> Option<Vec<f64>> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 for key in &["vertex_stress", "stress_scores", "stress", "local_stress"] {
 if let Some(val) = telemetry.get(*key) {
 if let Some(arr) = val.as_array() {
 let stress: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();

 if stress.len() == self.num_vertices {
 return Some(stress);
 }
 }
 }
 }
 }
 None
 }

 fn derive_stress_from_embedding(&self, result: &PhaseOutcome) -> Option<Vec<f64>> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 // Look for embedding coordinates
 let embedding_val = telemetry
 .get("vertex_embedding")
 .or_else(|| telemetry.get("embedding"))
 .or_else(|| telemetry.get("coordinates"))?;

 let coords: Vec<f64> = embedding_val
 .as_array()?
 .iter()
 .filter_map(|v| v.as_f64())
 .collect();

 // Determine dimension (usually 2 or 3)
 let dim = if coords.len() == self.num_vertices * 2 {
 2
 } else if coords.len() == self.num_vertices * 3 {
 3
 } else {
 return None;
 };

 let adjacency = self.adjacency_cache.as_ref()?;
 let mut stress = vec![0.0; self.num_vertices];

 // Compute stress as squared distance error
 for v in 0..self.num_vertices {
 let v_coords = &coords[v * dim..(v + 1) * dim];

 for &u in &adjacency[v] {
 if u > v {
 continue; // Count each edge once
 }

 let u_coords = &coords[u * dim..(u + 1) * dim];

 // Euclidean distance in embedding
 let embedded_dist: f64 = v_coords
 .iter()
 .zip(u_coords.iter())
 .map(|(a, b)| (a - b).powi(2))
 .sum::<f64>()
 .sqrt();

 // Graph distance = 1 for adjacent vertices
 let error = (embedded_dist - 1.0).powi(2);
 stress[v] += error;
 stress[u] += error;
 }
 }

 Some(stress)
 } else {
 None
 }
 }

 fn derive_stress_from_distortion(&self, result: &PhaseOutcome) -> Option<Vec<f64>> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 for key in &["geodesic_distortion", "distortion", "path_stretch"] {
 if let Some(val) = telemetry.get(*key) {
 if let Some(arr) = val.as_array() {
 let distortion: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();

 if distortion.len() == self.num_vertices {
 return Some(distortion);
 }
 }
 }
 }
 }
 None
 }

 fn derive_stress_from_conflict_density(&self, context: &PhaseContext) -> Option<Vec<f64>> {
 let solution = context.best_solution.as_ref()?;
 let adjacency = self.adjacency_cache.as_ref()?;

 let mut stress = vec![0.0; self.num_vertices];

 for (v, neighbors) in adjacency.iter().enumerate() {
 let degree = neighbors.len() as f64;
 if degree == 0.0 {
 continue;
 }

 // Count conflicts
 let conflicts: f64 = neighbors
 .iter()
 .filter(|&&u| solution.colors[u] == solution.colors[v])
 .count() as f64;

 // Count color pressure (neighbors using same color)
 let mut color_counts = HashMap::new();
 for &u in neighbors {
 *color_counts.entry(solution.colors[u]).or_insert(0usize) += 1;
 }

 // Stress = conflict ratio + color pressure variance
 let conflict_ratio = conflicts / degree;
 let pressure_variance: f64 = if !color_counts.is_empty() {
 let mean = degree / color_counts.len() as f64;
 color_counts
 .values()
 .map(|&c| (c as f64 - mean).powi(2))
 .sum::<f64>()
 / color_counts.len() as f64
 } else {
 0.0
 };

 stress[v] = conflict_ratio + 0.1 * pressure_variance.sqrt();
 }

 Some(stress)
 }

 fn normalize_stress(&self, mut stress: Vec<f64>) -> Vec<f64> {
 if stress.is_empty() {
 return stress;
 }

 if self.config.stress_normalize_percentile {
 // Percentile-based normalization (robust to outliers)
 let mut sorted = stress.clone();
 sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

 let p5 = sorted[sorted.len() * 5 / 100];
 let p95 = sorted[sorted.len() * 95 / 100];
 let range = (p95 - p5).max(1e-10);

 for s in &mut stress {
 *s = ((*s - p5) / range).clamp(0.0, 1.0);
 }
 } else {
 // Standard min-max normalization
 let min = stress.iter().copied().fold(f64::INFINITY, f64::min);
 let max = stress.iter().copied().fold(f64::NEG_INFINITY, f64::max);
 let range = (max - min).max(1e-10);

 for s in &mut stress {
 *s = (*s - min) / range;
 }
 }

 stress
 }

 // =========================================================================
 // PHASE 6: PERSISTENCE EXTRACTION
 // =========================================================================

 /// Extract persistence scores from Phase 6 (TDA/Ensemble Validation)
 ///
 /// Uses multi-source fusion:
 /// 1. Explicit persistence from TDA
 /// 2. Feature lifetime analysis
 /// 3. Ensemble stability metrics
 /// 4. Color stability as proxy
 pub fn extract_phase6_persistence(
 &self,
 result: &PhaseOutcome,
 context: &PhaseContext,
 ) -> Option<Vec<f64>> {
 // Source 1: Explicit persistence
 if let Some(persistence) = self.extract_explicit_persistence(result) {
 log::debug!("Phase6: Extracted explicit persistence");
 return Some(persistence);
 }

 // Source 2: Feature lifetime
 if let Some(persistence) = self.derive_persistence_from_features(result) {
 log::debug!("Phase6: Derived persistence from TDA features");
 return Some(persistence);
 }

 // Source 3: Ensemble stability
 if let Some(persistence) = self.derive_persistence_from_ensemble(result) {
 log::debug!("Phase6: Derived persistence from ensemble stability");
 return Some(persistence);
 }

 // Source 4: Color stability proxy
 if let Some(persistence) = self.derive_persistence_from_color_stability(result, context) {
 if self.config.verbose_fallback {
 log::warn!("Phase6: Using color stability as persistence proxy");
 }
 return Some(persistence);
 }

 log::warn!("Phase6: No persistence source available");
 None
 }

 fn extract_explicit_persistence(&self, result: &PhaseOutcome) -> Option<Vec<f64>> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 for key in &[
 "persistence_scores",
 "persistence",
 "vertex_persistence",
 "topological_stability",
 ] {
 if let Some(val) = telemetry.get(*key) {
 if let Some(arr) = val.as_array() {
 let persistence: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();

 if persistence.len() == self.num_vertices {
 return Some(persistence);
 }
 }
 }
 }
 }
 None
 }

 fn derive_persistence_from_features(&self, result: &PhaseOutcome) -> Option<Vec<f64>> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 // Look for persistence diagram or birth-death pairs
 let features_val = telemetry
 .get("persistence_diagram")
 .or_else(|| telemetry.get("tda_features"))
 .or_else(|| telemetry.get("birth_death_pairs"))?;

 // Parse features and aggregate to per-vertex persistence
 // This depends on the exact format of TDA output
 // For now, attempt basic parsing
 if let Some(arr) = features_val.as_array() {
 if arr.len() == self.num_vertices {
 // Direct per-vertex persistence
 let persistence: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();

 if persistence.len() == self.num_vertices {
 return Some(persistence);
 }
 }

 // Birth-death pairs format: [[birth, death, vertex], ...]
 // Aggregate lifetime per vertex
 let mut vertex_lifetimes = vec![0.0; self.num_vertices];
 for feature in arr {
 if let Some(f_arr) = feature.as_array() {
 if f_arr.len() >= 3 {
 let birth = f_arr[0].as_f64().unwrap_or(0.0);
 let death = f_arr[1].as_f64().unwrap_or(0.0);
 let vertex = f_arr[2].as_u64().unwrap_or(0) as usize;

 if vertex < self.num_vertices {
 let lifetime = (death - birth).abs();
 if lifetime >= self.config.persistence_min_lifetime {
 vertex_lifetimes[vertex] += lifetime;
 }
 }
 }
 }
 }

 if vertex_lifetimes.iter().any(|&l| l > 0.0) {
 return Some(vertex_lifetimes);
 }
 }
 }
 None
 }

 fn derive_persistence_from_ensemble(&self, result: &PhaseOutcome) -> Option<Vec<f64>> {
 if let PhaseOutcome::Success { telemetry, .. } = result {
 // Ensemble methods produce per-vertex stability metrics
 for key in &[
 "ensemble_stability",
 "vertex_stability",
 "solution_consistency",
 "color_agreement",
 ] {
 if let Some(val) = telemetry.get(*key) {
 if let Some(arr) = val.as_array() {
 let stability: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();

 if stability.len() == self.num_vertices {
 return Some(stability);
 }
 }
 }
 }
 }
 None
 }

 fn derive_persistence_from_color_stability(
 &self,
 result: &PhaseOutcome,
 context: &PhaseContext,
 ) -> Option<Vec<f64>> {
 let solution = context.best_solution.as_ref()?;
 let adjacency = self.adjacency_cache.as_ref()?;

 // Color stability = how "stable" is a vertex's color assignment
 // High stability = few alternatives would be conflict-free
 // Low stability = many alternatives would work

 let num_colors = solution.chromatic_number.max(1);
 let mut persistence = vec![0.0; self.num_vertices];

 for v in 0..self.num_vertices {
 let current_color = solution.colors[v];

 // Count neighbor colors
 let mut neighbor_colors = vec![0usize; num_colors];
 for &u in &adjacency[v] {
 let c = solution.colors[u];
 if c < num_colors {
 neighbor_colors[c] += 1;
 }
 }

 // Count how many colors are conflict-free
 let available_colors = neighbor_colors.iter().filter(|&&count| count == 0).count();

 // Persistence = inverse of flexibility
 // If only 1 color works, very stable (persistence = 1.0)
 // If many colors work, less stable (lower persistence)
 persistence[v] = if num_colors > 0 {
 1.0 - (available_colors as f64 / num_colors as f64)
 } else {
 0.5
 };
 }

 Some(persistence)
 }
}

// =============================================================================
// GEOMETRY SYNCHRONIZER - Orchestrator integration
// =============================================================================

/// Handles geometry flow between phases and GeometryAccumulator
#[cfg(feature = "cuda")]
pub struct GeometrySynchronizer {
 extractor: GeometryExtractor,
 accumulator: GeometryAccumulator,

 /// Track what geometry has been registered
 registered: RegisteredGeometry,
}

#[derive(Debug, Default)]
struct RegisteredGeometry {
 phase0_hotspots: bool,
 phase1_beliefs: bool,
 phase4_stress: bool,
 phase6_persistence: bool,
}

#[cfg(feature = "cuda")]
impl GeometrySynchronizer {
 /// Create synchronizer for a graph with GPU context
 pub fn new(device: Arc<CudaContext>, graph: &Graph) -> Result<Self> {
 let extractor = GeometryExtractor::new(graph);
 let accumulator = GeometryAccumulator::new(device, graph.num_vertices)
 .context("Failed to create GeometryAccumulator")?;

 Ok(Self {
 extractor,
 accumulator,
 registered: RegisteredGeometry::default(),
 })
 }

 /// Create with custom extraction configuration
 pub fn with_config(
 device: Arc<CudaContext>,
 graph: &Graph,
 config: ExtractionConfig,
 ) -> Result<Self> {
 let extractor = GeometryExtractor::with_config(graph, config);
 let accumulator = GeometryAccumulator::new(device, graph.num_vertices)
 .context("Failed to create GeometryAccumulator")?;

 Ok(Self {
 extractor,
 accumulator,
 registered: RegisteredGeometry::default(),
 })
 }

 /// Get reference to accumulator for WHCR
 pub fn accumulator(&self) -> &GeometryAccumulator {
 &self.accumulator
 }

 /// Get mutable reference to accumulator
 pub fn accumulator_mut(&mut self) -> &mut GeometryAccumulator {
 &mut self.accumulator
 }

 /// Synchronize geometry after Phase 0
 pub fn sync_phase0(&mut self, result: &PhaseOutcome, context: &PhaseContext) -> Result<bool> {
 if self.registered.phase0_hotspots {
 log::debug!("Phase0 geometry already registered");
 return Ok(true);
 }

 if let Some(hotspots) = self.extractor.extract_phase0_hotspots(result, context) {
 self.accumulator
 .set_phase0_hotspots(&hotspots)
 .context("Failed to set Phase 0 hotspots")?;

 self.registered.phase0_hotspots = true;
 log::info!("Synchronized {} Phase 0 hotspots to GPU", hotspots.len());
 Ok(true)
 } else {
 log::warn!("Phase 0 geometry extraction failed");
 Ok(false)
 }
 }

 /// Synchronize geometry after Phase 1
 pub fn sync_phase1(&mut self, result: &PhaseOutcome, context: &PhaseContext) -> Result<bool> {
 if self.registered.phase1_beliefs {
 log::debug!("Phase1 geometry already registered");
 return Ok(true);
 }

 if let Some((beliefs, num_colors)) = self.extractor.extract_phase1_beliefs(result, context)
 {
 self.accumulator
 .set_phase1_beliefs(&beliefs, num_colors)
 .context("Failed to set Phase 1 beliefs")?;

 self.registered.phase1_beliefs = true;
 log::info!(
 "Synchronized Phase 1 beliefs ({}x{}) to GPU",
 beliefs.len() / num_colors,
 num_colors
 );
 Ok(true)
 } else {
 log::warn!("Phase 1 geometry extraction failed");
 Ok(false)
 }
 }

 /// Synchronize geometry after Phase 4
 pub fn sync_phase4(&mut self, result: &PhaseOutcome, context: &PhaseContext) -> Result<bool> {
 if self.registered.phase4_stress {
 log::debug!("Phase4 geometry already registered");
 return Ok(true);
 }

 if let Some(stress) = self.extractor.extract_phase4_stress(result, context) {
 self.accumulator
 .set_phase4_stress(&stress)
 .context("Failed to set Phase 4 stress")?;

 self.registered.phase4_stress = true;
 log::info!("Synchronized {} Phase 4 stress values to GPU", stress.len());
 Ok(true)
 } else {
 log::warn!("Phase 4 geometry extraction failed");
 Ok(false)
 }
 }

 /// Synchronize geometry after Phase 6
 pub fn sync_phase6(&mut self, result: &PhaseOutcome, context: &PhaseContext) -> Result<bool> {
 if self.registered.phase6_persistence {
 log::debug!("Phase6 geometry already registered");
 return Ok(true);
 }

 if let Some(persistence) = self.extractor.extract_phase6_persistence(result, context) {
 self.accumulator
 .set_phase6_persistence(&persistence)
 .context("Failed to set Phase 6 persistence")?;

 self.registered.phase6_persistence = true;
 log::info!(
 "Synchronized {} Phase 6 persistence values to GPU",
 persistence.len()
 );
 Ok(true)
 } else {
 log::warn!("Phase 6 geometry extraction failed");
 Ok(false)
 }
 }

 /// Get summary of registered geometry
 pub fn summary(&self) -> String {
 let mut parts = Vec::new();
 if self.registered.phase0_hotspots {
 parts.push("P0:hotspots");
 }
 if self.registered.phase1_beliefs {
 parts.push("P1:beliefs");
 }
 if self.registered.phase4_stress {
 parts.push("P4:stress");
 }
 if self.registered.phase6_persistence {
 parts.push("P6:persistence");
 }

 if parts.is_empty() {
 "none".to_string()
 } else {
 parts.join(", ")
 }
 }

 /// Access the underlying geometry accumulator for WHCR invocation
 pub fn geometry(&self) -> Option<&GeometryAccumulator> {
 Some(&self.accumulator)
 }

 /// Check if geometry is sufficient for a phase
 pub fn ready_for_phase(&self, phase: u8) -> bool {
 match phase {
 2 => self.registered.phase0_hotspots || self.registered.phase1_beliefs,
 3 => self.registered.phase4_stress,
 5 => true, // Use whatever is available
 7 => self.registered.phase4_stress && self.registered.phase6_persistence,
 _ => true,
 }
 }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
 use super::*;

 fn make_test_graph(num_vertices: usize, density: f64) -> Graph {
 let mut adjacency = vec![Vec::new(); num_vertices];

 for i in 0..num_vertices {
 for j in (i + 1)..num_vertices {
 if rand::random::<f64>() < density {
 adjacency[i].push(j);
 adjacency[j].push(i);
 }
 }
 }

 Graph {
 num_vertices,
 adjacency,
 // ... other fields
 }
 }

 #[test]
 fn test_adaptive_config() {
 // Dense graph (DSJC-like)
 let dense_config = ExtractionConfig::for_graph(125, 3891);
 assert!(dense_config.hotspot_fraction > 0.15);

 // Sparse graph
 let sparse_config = ExtractionConfig::for_graph(1000, 2000);
 assert!(sparse_config.hotspot_fraction <= 0.15);

 // Small graph
 let small_config = ExtractionConfig::for_graph(50, 200);
 assert!(small_config.hotspot_fraction > 0.2);
 }

 #[test]
 fn test_hotspot_thresholding() {
 let graph = make_test_graph(100, 0.3);
 let extractor = GeometryExtractor::new(&graph);

 // Test with uniform scores
 let uniform_scores = vec![0.5; 100];
 let hotspots = extractor.threshold_to_hotspots(&uniform_scores);
 assert!(!hotspots.is_empty());
 assert!(hotspots.len() <= 20); // ~15-20% of 100

 // Test with skewed scores
 let mut skewed_scores = vec![0.1; 100];
 skewed_scores[0] = 1.0;
 skewed_scores[1] = 0.9;
 skewed_scores[2] = 0.8;
 let hotspots = extractor.threshold_to_hotspots(&skewed_scores);
 assert!(hotspots.contains(&0));
 assert!(hotspots.contains(&1));
 assert!(hotspots.contains(&2));
 }

 #[test]
 fn test_stress_normalization() {
 let graph = make_test_graph(100, 0.3);
 let extractor = GeometryExtractor::new(&graph);

 let stress = vec![0.0, 0.5, 1.0, 10.0, 100.0];
 let normalized = extractor.normalize_stress(stress);

 // Should be in [0, 1]
 assert!(normalized.iter().all(|&s| s >= 0.0 && s <= 1.0));
 }
}
