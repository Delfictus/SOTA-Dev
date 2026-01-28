//! Universal RL State representation.
//!
//! Aggregates metrics from all 7 PRISM phases into a single, discretized state space.
//!
//! Implements PRISM GPU Plan §3.1: UniversalRLState.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Discretization mode for state space compression.
///
/// Determines the size of the state space:
/// - Compact: 4096 states (12-bit hash)
/// - Extended: 65536 states (16-bit hash)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscretizationMode {
    /// Compact mode: 4096 states (faster learning, less precision)
    Compact,
    /// Extended mode: 65536 states (slower learning, more precision)
    Extended,
}

impl DiscretizationMode {
    /// Returns the number of discrete states for this mode.
    pub fn num_states(self) -> usize {
        match self {
            DiscretizationMode::Compact => 4096,   // 2^12
            DiscretizationMode::Extended => 65536, // 2^16
        }
    }
}

/// Universal RL State capturing metrics from all 7 phases.
///
/// ## State Components
///
/// ### Phase 0: Dendritic Reservoir
/// - `reservoir_entropy`: Neuron activation entropy (0.0 - 1.0)
/// - `reservoir_sparsity`: Fraction of inactive neurons (0.0 - 1.0)
///
/// ### Phase 1: Active Inference
/// - `active_inference_efe`: Expected Free Energy (lower = better)
/// - `active_inference_vfe`: Variational Free Energy
///
/// ### Phase 2: Thermodynamic
/// - `thermodynamic_temp`: Current temperature (dimensionless)
/// - `thermodynamic_energy`: System energy
///
/// ### Phase 3: Quantum-Classical Hybrid
/// - `quantum_purity`: State purity (0.0 - 1.0)
/// - `quantum_entanglement`: Entanglement metric
/// - `quantum_amplitude_variance`: Amplitude spread (complex evolution)
/// - `quantum_coherence`: Phase coherence (interference quality)
///
/// ### Phases 4/5: Geodesic
/// - `geodesic_centrality`: Mean vertex centrality
/// - `geodesic_diameter`: Graph diameter (shortest path metric)
///
/// ### Phase 6: TDA
/// - `tda_persistence`: Topological persistence (0.0 - 1.0)
/// - `tda_betti_0`: Betti number (connected components)
///
/// ### Phase 7: Ensemble
/// - `ensemble_diversity`: Solution diversity metric
/// - `ensemble_consensus`: Agreement across replicas
///
/// ### Global Metrics
/// - `num_vertices`: Number of vertices in the graph
/// - `chromatic_number`: Current best chromatic number
/// - `conflicts`: Number of edge conflicts
/// - `iteration`: Global iteration counter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalRLState {
    // Phase 0: Dendritic Reservoir
    pub reservoir_entropy: f64,
    pub reservoir_sparsity: f64,

    // Phase 1: Active Inference
    pub active_inference_efe: f64,
    pub active_inference_vfe: f64,

    // Phase 2: Thermodynamic
    pub thermodynamic_temp: f64,
    pub thermodynamic_energy: f64,

    // Phase 3: Quantum-Classical
    pub quantum_purity: f64,
    pub quantum_entanglement: f64,

    // Phase 3: Complex Quantum Evolution (Stage 5)
    /// Amplitude variance (spread of quantum amplitudes)
    pub quantum_amplitude_variance: f64,
    /// Phase coherence (quantum interference quality)
    pub quantum_coherence: f64,

    // Phases 4/5: Geodesic
    pub geodesic_centrality: f64,
    pub geodesic_diameter: f64,

    // Phase 6: TDA
    pub tda_persistence: f64,
    pub tda_betti_0: usize,

    // Phase 7: Ensemble
    pub ensemble_diversity: f64,
    pub ensemble_consensus: f64,

    // FluxNet v2: Warmstart & multi-attempt metrics
    /// Phase 2 annealing progress (0.0 = start, 1.0 = fully cooled)
    pub phase2_temperature_stage: f64,
    /// Phase 6 TDA coherence coefficient of variation (higher = better structural diversity)
    pub coherence_cv: f64,
    /// Warmstart quality: fraction of anchors selected (0.0 - 1.0)
    pub warmstart_quality: f64,
    /// Multi-attempt progress: current attempt / total attempts
    pub attempt_progress: f64,
    /// Memetic evolution: current generation number
    pub memetic_generation: usize,
    /// Memetic improvement rate: Δchromatic per generation (smoothed)
    pub memetic_improvement_rate: f64,

    // Metaphysical Coupling: Geometry stress metrics
    /// Geometric stress level from Phase 4/6 (0.0 = no stress, 1.0 = critical)
    /// Influences Phase 1/2/3/7 parameter adjustments via coupling feedback loop
    pub geometry_stress_level: f64,
    /// Overlap density: fraction of edges with same-color endpoints (0.0 - 1.0)
    pub geometry_overlap_density: f64,
    /// Number of geometric hotspots (high-conflict vertices)
    pub geometry_hotspot_count: usize,
    /// Previous geometry stress level (for reward shaping)
    /// Used to compute stress delta: positive reward when stress decreases
    pub previous_geometry_stress: f64,

    // New subsystem metrics (PRISM GPU Plan §7.4)
    /// MEC (Molecular Emergent Computing) free energy
    pub mec_free_energy: f64,
    /// CMA-ES transfer entropy mean
    pub cma_te_mean: f64,
    /// Ontology semantic conflicts count
    pub ontology_conflicts: u32,
    /// Biomolecular RMSD prediction error
    pub bio_rmsd: f64,
    /// Materials band gap prediction
    pub mat_band_gap: f64,
    /// GNN embedding loss
    pub gnn_loss: f64,

    // Global state
    pub num_vertices: usize,
    pub chromatic_number: usize,
    pub conflicts: usize,
    pub iteration: usize,
}

impl UniversalRLState {
    /// Creates a new default RL state.
    pub fn new() -> Self {
        Self {
            reservoir_entropy: 0.5,
            reservoir_sparsity: 0.5,
            active_inference_efe: 0.0,
            active_inference_vfe: 0.0,
            thermodynamic_temp: 1.0,
            thermodynamic_energy: 0.0,
            quantum_purity: 1.0,
            quantum_entanglement: 0.0,
            quantum_amplitude_variance: 0.0,
            quantum_coherence: 1.0,
            geodesic_centrality: 0.0,
            geodesic_diameter: 0.0,
            tda_persistence: 0.0,
            tda_betti_0: 0,
            ensemble_diversity: 0.0,
            ensemble_consensus: 0.0,
            phase2_temperature_stage: 0.0,
            coherence_cv: 0.0,
            warmstart_quality: 0.0,
            attempt_progress: 0.0,
            memetic_generation: 0,
            memetic_improvement_rate: 0.0,
            geometry_stress_level: 0.0,
            geometry_overlap_density: 0.0,
            geometry_hotspot_count: 0,
            previous_geometry_stress: 0.0,
            mec_free_energy: 0.0,
            cma_te_mean: 0.0,
            ontology_conflicts: 0,
            bio_rmsd: 0.0,
            mat_band_gap: 0.0,
            gnn_loss: 0.0,
            num_vertices: 0,
            chromatic_number: 0,
            conflicts: 0,
            iteration: 0,
        }
    }

    /// Updates geometry stress and returns the delta for reward shaping.
    ///
    /// # Returns
    /// Stress delta: negative = stress increased (bad), positive = stress decreased (good)
    pub fn update_geometry_stress(&mut self, new_stress: f64) -> f64 {
        let delta = self.geometry_stress_level - new_stress;
        self.previous_geometry_stress = self.geometry_stress_level;
        self.geometry_stress_level = new_stress;
        delta
    }

    /// Computes geometry-based reward bonus for reinforcement learning.
    ///
    /// # Algorithm
    /// - Stress decrease: positive reward proportional to delta
    /// - Stress increase: negative reward (penalty)
    /// - Scale factor: 2.0 (makes geometry feedback significant)
    ///
    /// # Example
    /// - Stress drops from 0.8 to 0.5: reward = +0.6 (good!)
    /// - Stress rises from 0.3 to 0.6: reward = -0.6 (bad!)
    ///
    /// # Returns
    /// Reward bonus in range [-2.0, +2.0]
    pub fn compute_geometry_reward_bonus(&self) -> f64 {
        let stress_delta = self.previous_geometry_stress - self.geometry_stress_level;
        // Scale by 2.0 to make geometry feedback significant relative to base rewards
        stress_delta * 2.0
    }

    /// Discretizes the continuous state into a single integer index.
    ///
    /// Uses a hash function to map the continuous state vector to a discrete bin.
    ///
    /// ## Algorithm
    /// 1. Normalize each metric to [0, 1]
    /// 2. Hash the normalized vector
    /// 3. Modulo by the number of states (4096 or 65536)
    ///
    /// This provides a deterministic, uniform mapping from continuous to discrete states.
    pub fn discretize(&self, mode: DiscretizationMode) -> usize {
        let mut hasher = DefaultHasher::new();

        // Hash normalized metrics (quantized to reduce hash collisions)
        let quantize = |x: f64| -> u32 { (x.clamp(0.0, 1.0) * 255.0) as u32 };

        quantize(self.reservoir_entropy).hash(&mut hasher);
        quantize(self.reservoir_sparsity).hash(&mut hasher);
        quantize(self.thermodynamic_temp.min(10.0) / 10.0).hash(&mut hasher);
        quantize(self.quantum_purity).hash(&mut hasher);
        quantize(self.geodesic_centrality).hash(&mut hasher);
        quantize(self.tda_persistence).hash(&mut hasher);
        quantize(self.ensemble_diversity).hash(&mut hasher);

        // FluxNet v2 metrics
        quantize(self.phase2_temperature_stage).hash(&mut hasher);
        quantize(self.coherence_cv).hash(&mut hasher);
        quantize(self.warmstart_quality).hash(&mut hasher);
        quantize(self.attempt_progress).hash(&mut hasher);
        quantize(self.memetic_improvement_rate).hash(&mut hasher);

        // Geometry stress metrics
        quantize(self.geometry_stress_level).hash(&mut hasher);
        quantize(self.geometry_overlap_density).hash(&mut hasher);

        // New subsystem metrics
        quantize(self.mec_free_energy.abs().min(100.0) / 100.0).hash(&mut hasher);
        quantize(self.cma_te_mean.abs().min(1.0)).hash(&mut hasher);
        quantize(self.bio_rmsd.min(10.0) / 10.0).hash(&mut hasher);
        quantize(self.mat_band_gap.min(10.0) / 10.0).hash(&mut hasher);
        quantize(self.gnn_loss.min(1.0)).hash(&mut hasher);

        // Include discrete metrics
        (self.chromatic_number % 256).hash(&mut hasher);
        (self.conflicts % 256).hash(&mut hasher);
        (self.memetic_generation % 256).hash(&mut hasher);
        (self.geometry_hotspot_count % 256).hash(&mut hasher);
        (self.ontology_conflicts % 256).hash(&mut hasher);

        let hash = hasher.finish();
        (hash as usize) % mode.num_states()
    }

    /// Creates a state from a graph and phase metrics.
    pub fn from_metrics(
        num_vertices: usize,
        chromatic_number: usize,
        conflicts: usize,
        iteration: usize,
    ) -> Self {
        Self {
            num_vertices,
            chromatic_number,
            conflicts,
            iteration,
            ..Self::new()
        }
    }

    /// Updates a specific phase metric.
    pub fn set_phase_metric(&mut self, phase: &str, metric: &str, value: f64) {
        match (phase, metric) {
            ("Phase0", "entropy") => self.reservoir_entropy = value,
            ("Phase0", "sparsity") => self.reservoir_sparsity = value,
            ("Phase1", "efe") => self.active_inference_efe = value,
            ("Phase1", "vfe") => self.active_inference_vfe = value,
            ("Phase2", "temperature") => self.thermodynamic_temp = value,
            ("Phase2", "energy") => self.thermodynamic_energy = value,
            ("Phase2", "temperature_stage") => self.phase2_temperature_stage = value,
            ("Phase3", "purity") => self.quantum_purity = value,
            ("Phase3", "entanglement") => self.quantum_entanglement = value,
            ("Phase4", "centrality") | ("Phase5", "centrality") => self.geodesic_centrality = value,
            ("Phase4", "diameter") | ("Phase5", "diameter") => self.geodesic_diameter = value,
            ("Phase6", "persistence") => self.tda_persistence = value,
            ("Phase6", "coherence_cv") => self.coherence_cv = value,
            ("Phase7", "diversity") => self.ensemble_diversity = value,
            ("Phase7", "consensus") => self.ensemble_consensus = value,
            ("Warmstart", "quality") => self.warmstart_quality = value,
            ("MultiAttempt", "progress") => self.attempt_progress = value,
            ("Memetic", "generation") => self.memetic_generation = value as usize,
            ("Memetic", "improvement_rate") => self.memetic_improvement_rate = value,
            ("Geometry", "stress_level") => self.geometry_stress_level = value,
            ("Geometry", "overlap_density") => self.geometry_overlap_density = value,
            ("Geometry", "hotspot_count") => self.geometry_hotspot_count = value as usize,
            ("MEC", "free_energy") => self.mec_free_energy = value,
            ("CMA", "te_mean") => self.cma_te_mean = value,
            ("Ontology", "conflicts") => self.ontology_conflicts = value as u32,
            ("Biomolecular", "rmsd") => self.bio_rmsd = value,
            ("Materials", "band_gap") => self.mat_band_gap = value,
            ("GNN", "loss") => self.gnn_loss = value,
            _ => log::warn!("Unknown phase metric: {}.{}", phase, metric),
        }
    }
}

impl Default for UniversalRLState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discretization() {
        let state = UniversalRLState::new();

        let compact_index = state.discretize(DiscretizationMode::Compact);
        assert!(compact_index < 4096);

        let extended_index = state.discretize(DiscretizationMode::Extended);
        assert!(extended_index < 65536);
    }

    #[test]
    fn test_discretization_deterministic() {
        let state1 = UniversalRLState::new();
        let state2 = UniversalRLState::new();

        assert_eq!(
            state1.discretize(DiscretizationMode::Compact),
            state2.discretize(DiscretizationMode::Compact)
        );
    }

    #[test]
    fn test_set_phase_metric() {
        let mut state = UniversalRLState::new();

        state.set_phase_metric("Phase0", "entropy", 0.75);
        assert_eq!(state.reservoir_entropy, 0.75);

        state.set_phase_metric("Phase2", "temperature", 2.5);
        assert_eq!(state.thermodynamic_temp, 2.5);
    }
}
