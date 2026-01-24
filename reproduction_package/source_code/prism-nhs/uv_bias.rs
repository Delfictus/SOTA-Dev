//! UV Bias Perturbation Engine (Stage 6)
//!
//! Implements targeted perturbation of aromatic residues using simulated UV absorption.
//! This creates a pump-probe style molecular dynamics where:
//!
//! - **PUMP**: UV burst to aromatic residues (Trp, Tyr, Phe)
//! - **PROBE**: Neuromorphic detection of dewetting spikes
//! - **CORRELATION**: Causal link between perturbation and pocket opening
//!
//! # Physical Basis
//!
//! Aromatic residues absorb UV at 280nm:
//! - Tryptophan: ε ≈ 5,600 M⁻¹cm⁻¹ (STRONG)
//! - Tyrosine: ε ≈ 1,400 M⁻¹cm⁻¹ (MODERATE)
//! - Phenylalanine: ε ≈ 200 M⁻¹cm⁻¹ (WEAK)
//! - **Water: ε ≈ 0 (TRANSPARENT)**
//!
//! Water's transparency at 280nm means perturbations create signal on a silent
//! background - enabling causal inference about pocket opening mechanisms.
//!
//! # The Holographic Negative Insight
//!
//! By perturbing what EXCLUDES water (hydrophobic aromatics), we observe
//! water's RESPONSE (leaving = dewetting spike) without perturbing water itself.
//! This is the computational equivalent of pump-probe spectroscopy.

use crate::config::{NhsConfig, UvBiasConfig, ABSORPTION_NORMALIZATION};
use crate::config::{TRP_EXTINCTION_280, TYR_EXTINCTION_280, PHE_EXTINCTION_280};
use anyhow::Result;
use rand::Rng;
use rand_distr::{Distribution, Normal, UnitSphere};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// =============================================================================
// AROMATIC TARGET STRUCTURES
// =============================================================================

/// Aromatic residue targeted for UV perturbation
#[derive(Debug, Clone)]
pub struct AromaticTarget {
    /// Residue index in topology
    pub residue_idx: usize,

    /// Residue type (TRP, TYR, PHE)
    pub residue_type: AromaticType,

    /// Atom indices forming the aromatic ring
    pub ring_atoms: Vec<usize>,

    /// Center of mass of the ring
    pub ring_center: [f32; 3],

    /// Ring plane normal vector
    pub ring_normal: [f32; 3],

    /// Two orthogonal vectors in the ring plane
    pub ring_plane_vectors: [[f32; 3]; 2],

    /// Relative absorption strength (Trp = 1.0)
    pub absorption_strength: f32,

    /// Solvent accessible surface area (for surface filtering)
    pub sasa: f32,

    /// Nearest pocket probability voxel value
    pub pocket_probability: f32,

    /// Is this target currently active?
    pub active: bool,
}

/// Aromatic residue types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AromaticType {
    Tryptophan,  // Strongest absorber
    Tyrosine,    // Moderate absorber
    Phenylalanine, // Weak absorber
}

impl AromaticType {
    /// Get 3-letter code
    pub fn code(&self) -> &'static str {
        match self {
            AromaticType::Tryptophan => "TRP",
            AromaticType::Tyrosine => "TYR",
            AromaticType::Phenylalanine => "PHE",
        }
    }

    /// Get absorption coefficient at 280nm
    pub fn extinction_280(&self) -> f32 {
        match self {
            AromaticType::Tryptophan => TRP_EXTINCTION_280,
            AromaticType::Tyrosine => TYR_EXTINCTION_280,
            AromaticType::Phenylalanine => PHE_EXTINCTION_280,
        }
    }

    /// Get normalized absorption strength (Trp = 1.0)
    pub fn absorption_strength(&self) -> f32 {
        self.extinction_280() / ABSORPTION_NORMALIZATION
    }

    /// From residue name
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "TRP" | "W" => Some(AromaticType::Tryptophan),
            "TYR" | "Y" => Some(AromaticType::Tyrosine),
            "PHE" | "F" => Some(AromaticType::Phenylalanine),
            _ => None,
        }
    }
}

// =============================================================================
// BURST EVENT TRACKING
// =============================================================================

/// A single UV burst event for correlation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstEvent {
    /// Frame when burst was applied
    pub frame: usize,

    /// Target residue indices that were perturbed
    pub targets: Vec<usize>,

    /// Total energy deposited (arbitrary units)
    pub energy_deposited: f32,

    /// Burst pattern ID (for deconvolution)
    pub pattern_id: u32,
}

/// Spike event for correlation (from neuromorphic layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeEvent {
    /// Frame when spike occurred
    pub frame: usize,

    /// Neuron/voxel indices that spiked
    pub neurons: Vec<usize>,

    /// Associated residue indices (if known)
    pub residues: Vec<usize>,
}

/// Causal correlation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalCorrelation {
    /// Target residue that was perturbed
    pub target_residue: usize,

    /// Spike location (residue or voxel)
    pub spike_location: usize,

    /// Time lag between burst and spike (frames)
    pub lag_frames: usize,

    /// Correlation coefficient
    pub correlation: f32,

    /// Number of observations
    pub n_observations: usize,

    /// Is this a significant causal link?
    pub is_causal: bool,
}

// =============================================================================
// UV BIAS ENGINE
// =============================================================================

/// UV Bias Perturbation Engine
///
/// Manages targeted perturbation of aromatic residues and tracks
/// causal correlations with dewetting events.
pub struct UvBiasEngine {
    config: UvBiasConfig,

    /// All aromatic targets in the system
    all_targets: Vec<AromaticTarget>,

    /// Currently active targets (near high-probability pockets)
    active_targets: Vec<usize>,

    /// Burst history for correlation
    burst_history: VecDeque<BurstEvent>,

    /// Spike history for correlation
    spike_history: VecDeque<SpikeEvent>,

    /// Computed causal correlations
    correlations: HashMap<(usize, usize), CausalCorrelation>,

    /// Current frame counter
    current_frame: usize,

    /// Burst state machine
    burst_state: BurstState,

    /// Next burst pattern ID
    next_pattern_id: u32,

    /// Random number generator
    rng: rand::rngs::ThreadRng,
}

/// Burst generation state machine
#[derive(Debug, Clone)]
enum BurstState {
    /// Waiting for next burst (observation window)
    Observing { frames_remaining: u32 },

    /// Currently in a burst
    Bursting {
        pulses_remaining: u32,
        frames_to_next_pulse: u32,
        pattern_id: u32,
    },
}

impl UvBiasEngine {
    /// Create new UV bias engine
    pub fn new(config: UvBiasConfig) -> Self {
        let burst_state = if config.burst_mode {
            BurstState::Observing {
                frames_remaining: config.inter_burst_interval,
            }
        } else {
            // Continuous mode: always "bursting" with 1 pulse
            BurstState::Bursting {
                pulses_remaining: u32::MAX,
                frames_to_next_pulse: 1,
                pattern_id: 0,
            }
        };

        Self {
            config,
            all_targets: Vec::new(),
            active_targets: Vec::new(),
            burst_history: VecDeque::with_capacity(1000),
            spike_history: VecDeque::with_capacity(1000),
            correlations: HashMap::new(),
            current_frame: 0,
            burst_state,
            next_pattern_id: 0,
            rng: rand::thread_rng(),
        }
    }

    /// Initialize targets from protein topology
    ///
    /// # Arguments
    /// * `residue_names` - Residue names indexed by residue number
    /// * `atom_residues` - Residue index for each atom
    /// * `positions` - Flat array of atom positions [x0, y0, z0, x1, y1, z1, ...]
    pub fn initialize_targets(
        &mut self,
        residue_names: &[String],
        atom_residues: &[usize],
        positions: &[f32],
    ) -> Result<()> {
        self.all_targets.clear();

        for (res_idx, res_name) in residue_names.iter().enumerate() {
            // Check if aromatic
            let aromatic_type = match AromaticType::from_name(res_name) {
                Some(t) => t,
                None => continue,
            };

            // Check if target is enabled in config
            if !self.config.is_valid_target(res_name) {
                continue;
            }

            // Find ring atoms for this residue
            let ring_atoms = self.find_ring_atoms(res_idx, &aromatic_type, atom_residues);
            if ring_atoms.is_empty() {
                continue;
            }

            // Compute ring geometry
            let ring_center = compute_ring_center(&ring_atoms, positions);
            let (ring_normal, ring_plane_vectors) =
                compute_ring_plane(&ring_atoms, positions, &ring_center);

            let target = AromaticTarget {
                residue_idx: res_idx,
                residue_type: aromatic_type,
                ring_atoms,
                ring_center,
                ring_normal,
                ring_plane_vectors,
                absorption_strength: aromatic_type.absorption_strength(),
                sasa: 0.0,        // Updated later
                pocket_probability: 0.0, // Updated later
                active: false,
            };

            self.all_targets.push(target);
        }

        log::info!(
            "UvBiasEngine: Initialized {} aromatic targets (Trp: {}, Tyr: {}, Phe: {})",
            self.all_targets.len(),
            self.all_targets.iter().filter(|t| t.residue_type == AromaticType::Tryptophan).count(),
            self.all_targets.iter().filter(|t| t.residue_type == AromaticType::Tyrosine).count(),
            self.all_targets.iter().filter(|t| t.residue_type == AromaticType::Phenylalanine).count(),
        );

        Ok(())
    }

    /// Find ring atom indices for an aromatic residue
    fn find_ring_atoms(
        &self,
        residue_idx: usize,
        aromatic_type: &AromaticType,
        atom_residues: &[usize],
    ) -> Vec<usize> {
        // Get all atoms in this residue
        let residue_atoms: Vec<usize> = atom_residues
            .iter()
            .enumerate()
            .filter(|(_, &res)| res == residue_idx)
            .map(|(i, _)| i)
            .collect();

        // For simplicity, return all residue atoms
        // In production, filter to actual ring atoms (CG, CD1, CD2, CE1, CE2, CZ for Phe/Tyr)
        // and (CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2 for Trp)
        residue_atoms
    }

    /// Update target selection based on pocket probability field
    ///
    /// Targets aromatics near high-probability pocket regions
    pub fn update_target_selection(&mut self, pocket_probability: &[f32], grid_spacing: f32) {
        self.active_targets.clear();

        let threshold = self.config.pocket_probability_threshold;
        let radius = self.config.target_selection_radius;

        for (target_idx, target) in self.all_targets.iter_mut().enumerate() {
            // Find maximum pocket probability near this aromatic
            let max_prob = find_max_probability_near(
                &target.ring_center,
                pocket_probability,
                radius,
                grid_spacing,
            );

            target.pocket_probability = max_prob;
            target.active = max_prob >= threshold;

            if target.active {
                self.active_targets.push(target_idx);
            }
        }

        log::debug!(
            "UvBiasEngine: {} / {} targets active (prob >= {:.2})",
            self.active_targets.len(),
            self.all_targets.len(),
            threshold
        );
    }

    /// Process one frame - may apply perturbation
    ///
    /// Returns perturbation velocities to add to system (or empty if no perturbation)
    pub fn step(&mut self, positions: &[f32]) -> Option<PerturbationResult> {
        self.current_frame += 1;

        // Update burst state machine
        let should_perturb = self.advance_burst_state();

        if !should_perturb || self.active_targets.is_empty() {
            return None;
        }

        // Generate perturbation
        let result = self.generate_perturbation(positions);

        // Record burst event
        let burst = BurstEvent {
            frame: self.current_frame,
            targets: self.active_targets.clone(),
            energy_deposited: result.total_energy,
            pattern_id: self.next_pattern_id,
        };
        self.burst_history.push_back(burst);

        // Limit history size
        while self.burst_history.len() > 1000 {
            self.burst_history.pop_front();
        }

        Some(result)
    }

    /// Advance burst state machine, return true if should perturb
    fn advance_burst_state(&mut self) -> bool {
        match &mut self.burst_state {
            BurstState::Observing { frames_remaining } => {
                if *frames_remaining > 0 {
                    *frames_remaining -= 1;
                    false
                } else {
                    // Start new burst
                    self.next_pattern_id += 1;
                    self.burst_state = BurstState::Bursting {
                        pulses_remaining: self.config.pulses_per_burst,
                        frames_to_next_pulse: 0,
                        pattern_id: self.next_pattern_id,
                    };
                    true
                }
            }
            BurstState::Bursting {
                pulses_remaining,
                frames_to_next_pulse,
                pattern_id: _,
            } => {
                if *frames_to_next_pulse > 0 {
                    *frames_to_next_pulse -= 1;
                    false
                } else if *pulses_remaining > 0 {
                    *pulses_remaining -= 1;
                    *frames_to_next_pulse = self.config.intra_burst_interval;
                    true
                } else {
                    // Burst complete, start observation
                    self.burst_state = BurstState::Observing {
                        frames_remaining: self.config.inter_burst_interval,
                    };
                    false
                }
            }
        }
    }

    /// Generate perturbation velocities for active targets
    fn generate_perturbation(&mut self, positions: &[f32]) -> PerturbationResult {
        let mut velocity_deltas: HashMap<usize, [f32; 3]> = HashMap::new();
        let mut total_energy = 0.0f32;

        // Clone to avoid borrow issues
        let active_targets = self.active_targets.clone();
        let base_intensity = self.config.base_intensity;
        let scale_by_absorption = self.config.scale_by_absorption;
        let ring_plane_perturbation = self.config.ring_plane_perturbation;
        let direction_randomness = self.config.direction_randomness;

        for &target_idx in &active_targets {
            let target = &self.all_targets[target_idx];

            // Scale intensity by absorption strength
            let intensity = if scale_by_absorption {
                base_intensity * target.absorption_strength
            } else {
                base_intensity
            };

            // Update ring geometry from current positions
            let ring_atoms = target.ring_atoms.clone();
            let ring_center = compute_ring_center(&ring_atoms, positions);
            let (_normal, plane_vectors) = compute_ring_plane(&ring_atoms, positions, &ring_center);

            // Perturb each ring atom
            for &atom_idx in &ring_atoms {
                let delta = generate_atom_perturbation(
                    &mut self.rng,
                    intensity,
                    &plane_vectors,
                    ring_plane_perturbation,
                    direction_randomness,
                );
                velocity_deltas.insert(atom_idx, delta);

                // Estimate energy: 0.5 * m * v^2 (assume m=12 for carbon)
                let v_sq = delta[0].powi(2) + delta[1].powi(2) + delta[2].powi(2);
                total_energy += 0.5 * 12.0 * v_sq;
            }
        }

        PerturbationResult {
            frame: self.current_frame,
            velocity_deltas,
            total_energy,
            targets_perturbed: active_targets.len(),
        }
    }

    /// Record spike events from neuromorphic layer
    pub fn record_spikes(&mut self, neurons: Vec<usize>, residues: Vec<usize>) {
        let spike = SpikeEvent {
            frame: self.current_frame,
            neurons,
            residues,
        };
        self.spike_history.push_back(spike);

        // Limit history size
        while self.spike_history.len() > 1000 {
            self.spike_history.pop_front();
        }

        // Update correlations if tracking enabled
        if self.config.track_causality {
            self.update_correlations();
        }
    }

    /// Update causal correlations between bursts and spikes
    fn update_correlations(&mut self) {
        // Need sufficient history
        if self.burst_history.len() < 10 || self.spike_history.len() < 10 {
            return;
        }

        // For each recent burst, check for correlated spikes
        for burst in self.burst_history.iter().rev().take(20) {
            for spike in &self.spike_history {
                // Check if spike follows burst within correlation window
                if spike.frame <= burst.frame {
                    continue;
                }

                let lag = spike.frame - burst.frame;
                if lag > self.config.max_correlation_lag {
                    continue;
                }

                // Check for target-spike overlap
                for &target_idx in &burst.targets {
                    let target_res = self.all_targets[target_idx].residue_idx;

                    for &spike_res in &spike.residues {
                        let key = (target_res, spike_res);

                        let entry = self.correlations.entry(key).or_insert(CausalCorrelation {
                            target_residue: target_res,
                            spike_location: spike_res,
                            lag_frames: lag,
                            correlation: 0.0,
                            n_observations: 0,
                            is_causal: false,
                        });

                        entry.n_observations += 1;
                        // Update running correlation estimate
                        entry.correlation = 1.0 / (1.0 + (lag as f32 / 5.0));
                        entry.is_causal =
                            entry.correlation >= self.config.min_correlation_threshold
                                && entry.n_observations >= 3;
                    }
                }
            }
        }
    }

    /// Get all established causal correlations
    pub fn get_causal_links(&self) -> Vec<&CausalCorrelation> {
        self.correlations
            .values()
            .filter(|c| c.is_causal)
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> UvBiasStats {
        UvBiasStats {
            total_targets: self.all_targets.len(),
            active_targets: self.active_targets.len(),
            bursts_applied: self.burst_history.len(),
            spikes_recorded: self.spike_history.len(),
            causal_links: self.correlations.values().filter(|c| c.is_causal).count(),
            current_frame: self.current_frame,
        }
    }

    /// Reset for new trajectory
    pub fn reset(&mut self) {
        self.active_targets.clear();
        self.burst_history.clear();
        self.spike_history.clear();
        self.correlations.clear();
        self.current_frame = 0;
        self.next_pattern_id = 0;

        self.burst_state = if self.config.burst_mode {
            BurstState::Observing {
                frames_remaining: self.config.inter_burst_interval,
            }
        } else {
            BurstState::Bursting {
                pulses_remaining: u32::MAX,
                frames_to_next_pulse: 1,
                pattern_id: 0,
            }
        };

        for target in &mut self.all_targets {
            target.active = false;
            target.pocket_probability = 0.0;
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Find maximum pocket probability near a position (free function to avoid borrow issues)
fn find_max_probability_near(
    _pos: &[f32; 3],
    probability: &[f32],
    _radius: f32,
    _grid_spacing: f32,
) -> f32 {
    // Simplified: just return a placeholder
    // In production, sample the 3D grid around the position
    if probability.is_empty() {
        0.0
    } else {
        // Return max value as approximation
        probability.iter().copied().fold(0.0f32, f32::max) * 0.5
    }
}

/// Compute center of mass of ring atoms
fn compute_ring_center(ring_atoms: &[usize], positions: &[f32]) -> [f32; 3] {
    if ring_atoms.is_empty() {
        return [0.0; 3];
    }

    let mut center = [0.0f32; 3];
    for &atom_idx in ring_atoms {
        let base = atom_idx * 3;
        if base + 2 < positions.len() {
            center[0] += positions[base];
            center[1] += positions[base + 1];
            center[2] += positions[base + 2];
        }
    }

    let n = ring_atoms.len() as f32;
    [center[0] / n, center[1] / n, center[2] / n]
}

/// Compute ring plane normal and in-plane vectors
fn compute_ring_plane(
    ring_atoms: &[usize],
    positions: &[f32],
    center: &[f32; 3],
) -> ([f32; 3], [[f32; 3]; 2]) {
    // Default to XY plane if not enough atoms
    if ring_atoms.len() < 3 {
        return ([0.0, 0.0, 1.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    }

    // Get first two atoms relative to center
    let get_pos = |idx: usize| -> [f32; 3] {
        let base = idx * 3;
        if base + 2 < positions.len() {
            [
                positions[base] - center[0],
                positions[base + 1] - center[1],
                positions[base + 2] - center[2],
            ]
        } else {
            [0.0, 0.0, 0.0]
        }
    };

    let v1 = get_pos(ring_atoms[0]);
    let v2 = get_pos(ring_atoms[1]);

    // Cross product for normal
    let normal = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];

    // Normalize
    let mag = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
    let normal = if mag > 1e-6 {
        [normal[0] / mag, normal[1] / mag, normal[2] / mag]
    } else {
        [0.0, 0.0, 1.0]
    };

    // Normalize v1 for first in-plane vector
    let mag_v1 = (v1[0].powi(2) + v1[1].powi(2) + v1[2].powi(2)).sqrt();
    let plane_v1 = if mag_v1 > 1e-6 {
        [v1[0] / mag_v1, v1[1] / mag_v1, v1[2] / mag_v1]
    } else {
        [1.0, 0.0, 0.0]
    };

    // Second in-plane vector: normal × v1
    let plane_v2 = [
        normal[1] * plane_v1[2] - normal[2] * plane_v1[1],
        normal[2] * plane_v1[0] - normal[0] * plane_v1[2],
        normal[0] * plane_v1[1] - normal[1] * plane_v1[0],
    ];

    (normal, [plane_v1, plane_v2])
}

/// Generate velocity perturbation for single atom
fn generate_atom_perturbation<R: rand::Rng>(
    rng: &mut R,
    intensity: f32,
    plane_vectors: &[[f32; 3]; 2],
    ring_plane_perturbation: bool,
    direction_randomness: f32,
) -> [f32; 3] {
    if ring_plane_perturbation {
        // Perturbation in ring plane (mimics π→π* excitation)
        let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let mut delta = [0.0f32; 3];
        for i in 0..3 {
            delta[i] = intensity * (cos_a * plane_vectors[0][i] + sin_a * plane_vectors[1][i]);
        }

        // Add random component
        if direction_randomness > 0.0 {
            let random_vec: [f32; 3] = UnitSphere.sample(rng);
            for i in 0..3 {
                delta[i] += intensity * direction_randomness * random_vec[i];
            }
        }

        delta
    } else {
        // Random 3D perturbation
        let random_vec: [f32; 3] = UnitSphere.sample(rng);
        [
            intensity * random_vec[0],
            intensity * random_vec[1],
            intensity * random_vec[2],
        ]
    }
}

// =============================================================================
// OUTPUT STRUCTURES
// =============================================================================

/// Result of perturbation application
#[derive(Debug, Clone)]
pub struct PerturbationResult {
    /// Frame number
    pub frame: usize,

    /// Velocity deltas by atom index (Å/ps)
    pub velocity_deltas: HashMap<usize, [f32; 3]>,

    /// Total energy deposited (kcal/mol)
    pub total_energy: f32,

    /// Number of targets perturbed
    pub targets_perturbed: usize,
}

impl PerturbationResult {
    /// Apply perturbation to velocity array
    pub fn apply_to_velocities(&self, velocities: &mut [f32]) {
        for (&atom_idx, delta) in &self.velocity_deltas {
            let base = atom_idx * 3;
            if base + 2 < velocities.len() {
                velocities[base] += delta[0];
                velocities[base + 1] += delta[1];
                velocities[base + 2] += delta[2];
            }
        }
    }
}

/// UV Bias engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvBiasStats {
    pub total_targets: usize,
    pub active_targets: usize,
    pub bursts_applied: usize,
    pub spikes_recorded: usize,
    pub causal_links: usize,
    pub current_frame: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aromatic_type_absorption() {
        assert!(AromaticType::Tryptophan.absorption_strength() > AromaticType::Tyrosine.absorption_strength());
        assert!(AromaticType::Tyrosine.absorption_strength() > AromaticType::Phenylalanine.absorption_strength());
    }

    #[test]
    fn test_burst_state_machine() {
        let config = UvBiasConfig {
            burst_mode: true,
            pulses_per_burst: 3,
            intra_burst_interval: 2,
            inter_burst_interval: 10,
            ..Default::default()
        };

        let mut engine = UvBiasEngine::new(config);

        // Should be observing initially
        for _ in 0..10 {
            assert!(!engine.advance_burst_state());
        }

        // Should start bursting
        assert!(engine.advance_burst_state());

        // Intra-burst interval
        for _ in 0..2 {
            assert!(!engine.advance_burst_state());
        }

        // Second pulse
        assert!(engine.advance_burst_state());
    }

    #[test]
    fn test_perturbation_generation() {
        let config = UvBiasConfig::default();
        let mut engine = UvBiasEngine::new(config);

        // Create dummy target
        engine.all_targets.push(AromaticTarget {
            residue_idx: 0,
            residue_type: AromaticType::Tryptophan,
            ring_atoms: vec![0, 1, 2, 3, 4, 5],
            ring_center: [0.0, 0.0, 0.0],
            ring_normal: [0.0, 0.0, 1.0],
            ring_plane_vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            absorption_strength: 1.0,
            sasa: 1.0,
            pocket_probability: 0.5,
            active: true,
        });
        engine.active_targets.push(0);

        let positions = vec![0.0; 18]; // 6 atoms × 3 coords
        let result = engine.generate_perturbation(&positions);

        assert_eq!(result.targets_perturbed, 1);
        assert!(!result.velocity_deltas.is_empty());
    }
}
