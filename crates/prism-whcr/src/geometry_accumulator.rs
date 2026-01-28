//! Geometry Accumulator for WHCR Multi-Phase Integration
//!
//! Maintains GPU-resident geometry buffers that accumulate outputs from
//! multiple PRISM phases. This is the key to WHCR's compound advantage:
//! later invocations have access to ALL prior phase geometry.
//!
//! # Memory Management
//! - Buffers are allocated lazily when first populated
//! - Buffers remain on GPU between phase invocations (no CPU roundtrip)
//! - CPU-side copies are optional (for debugging/telemetry only)
//!
//! # Integration Pattern
//! ```ignore
//! // In orchestrator setup
//! let mut geometry = GeometryAccumulator::new(device.clone(), num_vertices)?;
//!
//! // After Phase 0
//! geometry.set_phase0_hotspots(&hotspot_indices)?;
//!
//! // After Phase 1
//! geometry.set_phase1_beliefs(&belief_matrix)?;
//!
//! // After Phase 4
//! geometry.set_phase4_stress(&stress_tensor)?;
//!
//! // After Phase 6
//! geometry.set_phase6_persistence(&persistence_values)?;
//!
//! // WHCR invocation gets accumulated geometry
//! whcr_gpu.set_geometry_buffers(geometry.get_gpu_buffers())?;
//! ```

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaStream, CudaSlice};
use std::sync::Arc;

use super::calling_phase::{CallingPhase, GeometrySources};

/// GPU-resident geometry buffers accumulated across PRISM phases
///
/// Each field is `Option<CudaSlice<T>>` - allocated only when populated.
/// This avoids wasting GPU memory for geometry that isn't available yet.
pub struct GeometryAccumulator {
 device: Arc<CudaContext>,
 stream: Arc<CudaStream>,
 num_vertices: usize,

 // === Phase 0: Dendritic Reservoir ===
 /// Hotspot vertex mask (1 = hotspot, 0 = normal)
 /// Shape: [num_vertices]
 d_hotspot_mask: Option<CudaSlice<i32>>,

 /// Priority modulation from reservoir dynamics
 /// Shape: [num_vertices]
 d_reservoir_priorities: Option<CudaSlice<f32>>,

 // === Phase 1: Active Inference ===
 /// Belief distribution over colors
 /// Shape: [num_vertices * max_colors] (flattened 2D)
 d_belief_distribution: Option<CudaSlice<f64>>,
 belief_num_colors: usize,

 /// Free energy per vertex (lower = more confident)
 /// Shape: [num_vertices]
 d_free_energy: Option<CudaSlice<f64>>,

 // === Phase 4: Geodesic Optimization ===
 /// Stress tensor per vertex (geometric distortion)
 /// Shape: [num_vertices]
 d_stress_scores: Option<CudaSlice<f64>>,

 /// 3D embedding coordinates (for visualization/debug)
 /// Shape: [num_vertices * 3]
 d_embedding: Option<CudaSlice<f64>>,

 // === Phase 6: TDA ===
 /// Persistence scores per vertex
 /// Shape: [num_vertices]
 d_persistence_scores: Option<CudaSlice<f64>>,

 /// Betti numbers (topological invariants)
 /// Shape: [max_dimension + 1]
 d_betti_numbers: Option<CudaSlice<i32>>,

 // === Tracking ===
 /// Which geometry sources have been populated
 available: GeometrySources,
}

impl GeometryAccumulator {
 /// Create new accumulator with no geometry populated
 pub fn new(device: Arc<CudaContext>, num_vertices: usize) -> Result<Self> {
 let stream = device.default_stream();
 Ok(Self {
 device,
 stream,
 num_vertices,
 d_hotspot_mask: None,
 d_reservoir_priorities: None,
 d_belief_distribution: None,
 belief_num_colors: 0,
 d_free_energy: None,
 d_stress_scores: None,
 d_embedding: None,
 d_persistence_scores: None,
 d_betti_numbers: None,
 available: GeometrySources::none(),
 })
 }

 /// Get number of vertices this accumulator is sized for
 #[inline]
 pub fn num_vertices(&self) -> usize {
 self.num_vertices
 }

 /// Check which geometry sources are currently available
 #[inline]
 pub fn available(&self) -> &GeometrySources {
 &self.available
 }

 /// Check if geometry is sufficient for a calling phase
 pub fn satisfies_phase(&self, phase: CallingPhase) -> bool {
 self.available.satisfies(&phase.expected_geometry())
 }

 // =========================================================================
 // Phase 0: Dendritic Reservoir
 // =========================================================================

 /// Set hotspot mask from Phase 0 dendritic reservoir
 ///
 /// # Arguments
 /// * `hotspot_indices` - Indices of vertices identified as hotspots
 pub fn set_phase0_hotspots(&mut self, hotspot_indices: &[usize]) -> Result<()> {
 // Create mask (1 for hotspots, 0 otherwise)
 let mut mask = vec![0i32; self.num_vertices];
 for &idx in hotspot_indices {
 if idx < self.num_vertices {
 mask[idx] = 1;
 }
 }

 let gpu_mask = self.stream.clone_htod(&mask)?;
 self.d_hotspot_mask = Some(gpu_mask);
 self.available.phase0_dendritic = true;

 log::debug!(
 "GeometryAccumulator: Set {} Phase 0 hotspots",
 hotspot_indices.len()
 );
 Ok(())
 }

 /// Set priority modulation from reservoir dynamics
 pub fn set_phase0_priorities(&mut self, priorities: &[f32]) -> Result<()> {
 if priorities.len() != self.num_vertices {
 anyhow::bail!(
 "Priority length {} != num_vertices {}",
 priorities.len(),
 self.num_vertices
 );
 }

 let gpu_priorities = self.stream.clone_htod(priorities)?;
 self.d_reservoir_priorities = Some(gpu_priorities);
 Ok(())
 }

 /// Get GPU hotspot mask buffer (if available)
 #[inline]
 pub fn hotspot_mask(&self) -> Option<&CudaSlice<i32>> {
 self.d_hotspot_mask.as_ref()
 }

 /// Get GPU reservoir priorities buffer (if available)
 #[inline]
 pub fn reservoir_priorities(&self) -> Option<&CudaSlice<f32>> {
 self.d_reservoir_priorities.as_ref()
 }

 // =========================================================================
 // Phase 1: Active Inference
 // =========================================================================

 /// Set belief distribution from Phase 1 active inference
 ///
 /// # Arguments
 /// * `beliefs` - Flattened belief matrix [num_vertices * num_colors]
 /// * `num_colors` - Number of color options
 pub fn set_phase1_beliefs(&mut self, beliefs: &[f64], num_colors: usize) -> Result<()> {
 let expected_len = self.num_vertices * num_colors;
 if beliefs.len() != expected_len {
 anyhow::bail!(
 "Beliefs length {} != expected {} ({}v * {}c)",
 beliefs.len(),
 expected_len,
 self.num_vertices,
 num_colors
 );
 }

 let gpu_beliefs = self.stream.clone_htod(beliefs)?;
 self.d_belief_distribution = Some(gpu_beliefs);
 self.belief_num_colors = num_colors;
 self.available.phase1_beliefs = true;

 log::debug!(
 "GeometryAccumulator: Set Phase 1 beliefs ({}x{})",
 self.num_vertices,
 num_colors
 );
 Ok(())
 }

 /// Set free energy values from Phase 1
 pub fn set_phase1_free_energy(&mut self, free_energy: &[f64]) -> Result<()> {
 if free_energy.len() != self.num_vertices {
 anyhow::bail!(
 "Free energy length {} != num_vertices {}",
 free_energy.len(),
 self.num_vertices
 );
 }

 let gpu_free_energy = self.stream.clone_htod(free_energy)?;
 self.d_free_energy = Some(gpu_free_energy);
 Ok(())
 }

 /// Get GPU belief distribution buffer (if available)
 #[inline]
 pub fn belief_distribution(&self) -> Option<&CudaSlice<f64>> {
 self.d_belief_distribution.as_ref()
 }

 /// Get number of colors in belief distribution
 #[inline]
 pub fn belief_num_colors(&self) -> usize {
 self.belief_num_colors
 }

 // =========================================================================
 // Phase 4: Geodesic Optimization
 // =========================================================================

 /// Set stress scores from Phase 4 geodesic optimization
 pub fn set_phase4_stress(&mut self, stress: &[f64]) -> Result<()> {
 if stress.len() != self.num_vertices {
 anyhow::bail!(
 "Stress length {} != num_vertices {}",
 stress.len(),
 self.num_vertices
 );
 }

 let gpu_stress = self.stream.clone_htod(stress)?;
 self.d_stress_scores = Some(gpu_stress);
 self.available.phase4_stress = true;

 log::debug!("GeometryAccumulator: Set Phase 4 stress scores");
 Ok(())
 }

 /// Set 3D embedding from Phase 4
 pub fn set_phase4_embedding(&mut self, embedding: &[f64]) -> Result<()> {
 let expected_len = self.num_vertices * 3;
 if embedding.len() != expected_len {
 anyhow::bail!(
 "Embedding length {} != expected {} ({}v * 3)",
 embedding.len(),
 expected_len,
 self.num_vertices
 );
 }

 let gpu_embedding = self.stream.clone_htod(embedding)?;
 self.d_embedding = Some(gpu_embedding);
 Ok(())
 }

 /// Get GPU stress scores buffer (if available)
 #[inline]
 pub fn stress_scores(&self) -> Option<&CudaSlice<f64>> {
 self.d_stress_scores.as_ref()
 }

 // =========================================================================
 // Phase 6: TDA
 // =========================================================================

 /// Set persistence scores from Phase 6 TDA
 pub fn set_phase6_persistence(&mut self, persistence: &[f64]) -> Result<()> {
 if persistence.len() != self.num_vertices {
 anyhow::bail!(
 "Persistence length {} != num_vertices {}",
 persistence.len(),
 self.num_vertices
 );
 }

 let gpu_persistence = self.stream.clone_htod(persistence)?;
 self.d_persistence_scores = Some(gpu_persistence);
 self.available.phase6_persistence = true;

 log::debug!("GeometryAccumulator: Set Phase 6 persistence scores");
 Ok(())
 }

 /// Set Betti numbers from Phase 6
 pub fn set_phase6_betti(&mut self, betti: &[i32]) -> Result<()> {
 let gpu_betti = self.stream.clone_htod(betti)?;
 self.d_betti_numbers = Some(gpu_betti);
 Ok(())
 }

 /// Get GPU persistence scores buffer (if available)
 #[inline]
 pub fn persistence_scores(&self) -> Option<&CudaSlice<f64>> {
 self.d_persistence_scores.as_ref()
 }

 // =========================================================================
 // Bulk Operations
 // =========================================================================

 /// Get all available geometry buffers for WHCR GPU
 ///
 /// Returns a struct that can be passed directly to
 /// `WaveletHierarchicalRepairGpu::set_geometry_buffers()`
 pub fn get_gpu_buffers(&self) -> GeometryBuffers<'_> {
 GeometryBuffers {
 hotspot_mask: self.d_hotspot_mask.as_ref(),
 stress_scores: self.d_stress_scores.as_ref(),
 persistence_scores: self.d_persistence_scores.as_ref(),
 belief_distribution: self.d_belief_distribution.as_ref(),
 belief_num_colors: self.belief_num_colors,
 reservoir_priorities: self.d_reservoir_priorities.as_ref(),
 free_energy: self.d_free_energy.as_ref(),
 }
 }

 /// Clear all geometry (for reset or reuse)
 pub fn clear(&mut self) {
 self.d_hotspot_mask = None;
 self.d_reservoir_priorities = None;
 self.d_belief_distribution = None;
 self.belief_num_colors = 0;
 self.d_free_energy = None;
 self.d_stress_scores = None;
 self.d_embedding = None;
 self.d_persistence_scores = None;
 self.d_betti_numbers = None;
 self.available = GeometrySources::none();
 }

 /// Get summary for logging/telemetry
 pub fn summary(&self) -> String {
 let mut parts = Vec::new();
 if self.available.phase0_dendritic {
 let count = self
 .d_hotspot_mask
 .as_ref()
 .map(|_| "set")
 .unwrap_or("missing");
 parts.push(format!("P0:{}", count));
 }
 if self.available.phase1_beliefs {
 parts.push(format!("P1:{}c", self.belief_num_colors));
 }
 if self.available.phase4_stress {
 parts.push("P4:stress".to_string());
 }
 if self.available.phase6_persistence {
 parts.push("P6:persist".to_string());
 }
 if parts.is_empty() {
 "none".to_string()
 } else {
 parts.join(", ")
 }
 }
}

/// References to GPU geometry buffers for WHCR
///
/// This is a lightweight struct holding references to CudaSlice buffers.
/// It's used to pass geometry to the GPU repair kernels without copying.
#[derive(Debug)]
pub struct GeometryBuffers<'a> {
 pub hotspot_mask: Option<&'a CudaSlice<i32>>,
 pub stress_scores: Option<&'a CudaSlice<f64>>,
 pub persistence_scores: Option<&'a CudaSlice<f64>>,
 pub belief_distribution: Option<&'a CudaSlice<f64>>,
 pub belief_num_colors: usize,
 pub reservoir_priorities: Option<&'a CudaSlice<f32>>,
 pub free_energy: Option<&'a CudaSlice<f64>>,
}

impl<'a> GeometryBuffers<'a> {
 /// Check if any geometry is available
 pub fn has_any(&self) -> bool {
 self.hotspot_mask.is_some()
 || self.stress_scores.is_some()
 || self.persistence_scores.is_some()
 || self.belief_distribution.is_some()
 }

 /// Check if full geometry is available (all four sources)
 pub fn has_full(&self) -> bool {
 self.hotspot_mask.is_some()
 && self.stress_scores.is_some()
 && self.persistence_scores.is_some()
 && self.belief_distribution.is_some()
 }
}

#[cfg(test)]
mod tests {
 use super::*;

 // Note: These tests require CUDA device, so they're marked ignore
 // Run with: cargo test -- --ignored

 #[test]
 #[ignore]
 fn test_geometry_accumulator_creation() {
 let device = Arc::new(CudaContext::new(0).unwrap());
 let accumulator = GeometryAccumulator::new(device, 1000).unwrap();

 assert_eq!(accumulator.num_vertices(), 1000);
 assert!(!accumulator.available().phase0_dendritic);
 assert!(!accumulator.available().phase1_beliefs);
 }

 #[test]
 #[ignore]
 fn test_phase_satisfaction() {
 let device = Arc::new(CudaContext::new(0).unwrap());
 let mut accumulator = GeometryAccumulator::new(device, 100).unwrap();

 // Initially satisfies no phases
 assert!(!accumulator.satisfies_phase(CallingPhase::Phase2Thermodynamic));

 // Add Phase 0 geometry
 accumulator.set_phase0_hotspots(&[0, 5, 10]).unwrap();

 // Still doesn't satisfy Phase 2 (needs Phase 1 too)
 assert!(!accumulator.satisfies_phase(CallingPhase::Phase2Thermodynamic));

 // Add Phase 1 geometry
 let beliefs = vec![0.5f64; 100 * 10]; // 100 vertices, 10 colors
 accumulator.set_phase1_beliefs(&beliefs, 10).unwrap();

 // Now satisfies Phase 2
 assert!(accumulator.satisfies_phase(CallingPhase::Phase2Thermodynamic));

 // But not Phase 3 (needs Phase 4 stress)
 assert!(!accumulator.satisfies_phase(CallingPhase::Phase3Quantum));
 }
}
