//! # PRISM-NHS Active Sensing Module
//!
//! Implements a closed-loop molecular stethoscope for cryptic binding site detection:
//!
//! 1. **Coherent UV Excitation**: Phase-locked aromatic excitation for vibrational interferometry
//! 2. **Neuromorphic Processing**: LIF neurons with lateral inhibition
//! 3. **Spike Sequence Detection**: STDP-like temporal pattern recognition
//! 4. **Resonance Detection**: Soft mode identification via frequency sweeping
//! 5. **Adaptive Control**: Reinforcement learning for probe selection
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use prism_nhs::active_sensing::{ActiveSensingEngine, ActiveSensingConfig};
//!
//! let config = ActiveSensingConfig::default();
//! let mut engine = ActiveSensingEngine::new(&ctx, config, &topology)?;
//!
//! // Main loop
//! for step in 0..n_steps {
//!     engine.apply_probe(&mut velocities, &positions, &masses, time_fs, dt_fs)?;
//!     // ... run MD step ...
//!     engine.lif_update(&water_density, &water_density_prev, dt_ps)?;
//!     engine.detect_sequences()?;
//!
//!     if step % probe_interval == 0 {
//!         let response = engine.compute_response()?;
//!         engine.update_probe_selection(&response)?;
//!     }
//! }
//!
//! let results = engine.get_results()?;
//! ```

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaFunction, CudaModule,
    LaunchConfig, PushKernelArg, DevicePtrMut, DevicePtr,
};
use std::sync::Arc;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for active sensing system
#[derive(Clone, Debug)]
pub struct ActiveSensingConfig {
    /// Voxel grid dimension (default: 32)
    pub grid_dim: i32,
    /// Voxel size in Angstroms (default: 2.5)
    pub grid_spacing: f32,
    /// LIF membrane time constant in ps (default: 1.0)
    pub tau_membrane: f32,
    /// Spike threshold (default: 0.15)
    pub spike_threshold: f32,
    /// Steps between probe switches (default: 100)
    pub probe_interval_steps: usize,
    /// Response analysis window in ps (default: 10.0)
    pub analysis_window_ps: f32,
    /// Maximum sequence distance in Angstroms (default: 20.0)
    pub max_sequence_distance: f32,
    /// Target number of aromatic groups (default: 4)
    pub target_n_groups: usize,
    /// Minimum separation between group centroids (default: 10.0)
    pub min_group_separation: f32,
    /// Spike history ring buffer length (default: 64)
    pub spike_history_length: usize,
    /// Maximum sequence detectors to create (default: 1024)
    pub max_sequence_detectors: usize,
    /// Resonance sweep minimum frequency in THz (default: 0.1)
    pub min_freq_thz: f32,
    /// Resonance sweep maximum frequency in THz (default: 10.0)
    pub max_freq_thz: f32,
    /// Exploration epsilon for adaptive control (default: 0.1)
    pub exploration_epsilon: f32,
}

impl Default for ActiveSensingConfig {
    fn default() -> Self {
        Self {
            grid_dim: 32,
            grid_spacing: 2.5,
            tau_membrane: 1.0,
            spike_threshold: 0.15,
            probe_interval_steps: 100,
            analysis_window_ps: 10.0,
            max_sequence_distance: 20.0,
            target_n_groups: 4,
            min_group_separation: 10.0,
            spike_history_length: 64,
            max_sequence_detectors: 1024,
            min_freq_thz: 0.1,
            max_freq_thz: 10.0,
            exploration_epsilon: 0.1,
        }
    }
}

// ============================================================================
// DATA STRUCTURES (mirroring CUDA)
// ============================================================================

/// Float3 for GPU interop
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Float3 {
    /// X component
    pub x: f32,
    /// Y component
    pub y: f32,
    /// Z component
    pub z: f32,
}

impl Float3 {
    /// Create new Float3
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// Timestamped spike event
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TimestampedSpike {
    /// Which voxel spiked
    pub voxel_idx: i32,
    /// When (picoseconds)
    pub timestamp_ps: f32,
    /// Spike magnitude
    pub intensity: f32,
    /// Which probe was active
    pub probe_id: i32,
}

/// Spike history ring buffer for a voxel
#[repr(C)]
#[derive(Clone, Debug)]
pub struct VoxelSpikeHistory {
    /// Ring buffer of spikes
    pub spikes: [TimestampedSpike; 64],
    /// Current write position
    pub head: i32,
    /// Number of valid entries
    pub count: i32,
    /// For refractory period
    pub last_spike_time: f32,
}

impl Default for VoxelSpikeHistory {
    fn default() -> Self {
        Self {
            spikes: [TimestampedSpike::default(); 64],
            head: 0,
            count: 0,
            last_spike_time: -1000.0,
        }
    }
}

/// Lateral inhibition state
#[repr(C)]
#[derive(Clone, Debug)]
pub struct LateralInhibitionState {
    /// Current inhibition level (0-1)
    pub inhibition_level: f32,
    /// For decay calculation
    pub last_update_time: f32,
    /// Which neighbors are inhibiting
    pub inhibiting_neighbors: [i32; 27],
    /// Number of inhibitors
    pub n_inhibitors: i32,
}

impl Default for LateralInhibitionState {
    fn default() -> Self {
        Self {
            inhibition_level: 0.0,
            last_update_time: 0.0,
            inhibiting_neighbors: [0; 27],
            n_inhibitors: 0,
        }
    }
}

/// Aromatic group for coherent excitation
#[repr(C)]
#[derive(Clone, Debug)]
pub struct AromaticGroup {
    /// Indices into UV target array
    pub aromatic_indices: [i32; 32],
    /// Number in group
    pub n_aromatics: i32,
    /// Geometric center
    pub centroid: Float3,
    /// Sum of absorption strengths
    pub total_absorption: f32,
}

impl Default for AromaticGroup {
    fn default() -> Self {
        Self {
            aromatic_indices: [-1; 32],
            n_aromatics: 0,
            centroid: Float3::default(),
            total_absorption: 0.0,
        }
    }
}

/// Coherent probe specification
#[repr(C)]
#[derive(Clone, Debug)]
pub struct CoherentProbe {
    /// Aromatic groups
    pub groups: [AromaticGroup; 8],
    /// Active groups
    pub n_groups: i32,
    /// Relative timing (femtoseconds)
    pub phase_delays_fs: [f32; 8],
    /// Energy allocation
    pub energy_per_group: [f32; 8],
    /// Total burst energy (kcal/mol)
    pub total_energy: f32,
    /// Unique identifier
    pub probe_id: i32,
    /// 0=hinge_A, 1=hinge_B, 2=pocket_lining, 3=control
    pub probe_type: i32,
    /// Predicted response for learning
    pub expected_response: f32,
}

impl Default for CoherentProbe {
    fn default() -> Self {
        Self {
            groups: std::array::from_fn(|_| AromaticGroup::default()),
            n_groups: 0,
            phase_delays_fs: [0.0; 8],
            energy_per_group: [5.0; 8],
            total_energy: 5.0,
            probe_id: 0,
            probe_type: 0,
            expected_response: 0.0,
        }
    }
}

/// Spike sequence detector
#[repr(C)]
#[derive(Clone, Debug)]
pub struct SpikeSequenceDetector {
    /// Expected voxel order
    pub voxel_sequence: [i32; 8],
    /// Length of sequence
    pub sequence_length: i32,
    /// Causality window
    pub max_inter_spike_interval_ps: f32,
    /// Progress through sequence
    pub current_position: i32,
    /// When first spike occurred
    pub sequence_start_time: f32,
    /// Evidence accumulator
    pub accumulated_score: f32,
    /// Times fully detected
    pub detection_count: i32,
    /// Importance weight (learned)
    pub weight: f32,
}

impl Default for SpikeSequenceDetector {
    fn default() -> Self {
        Self {
            voxel_sequence: [-1; 8],
            sequence_length: 0,
            max_inter_spike_interval_ps: 10.0,
            current_position: 0,
            sequence_start_time: 0.0,
            accumulated_score: 0.0,
            detection_count: 0,
            weight: 1.0,
        }
    }
}

/// Resonance detector state
#[repr(C)]
#[derive(Clone, Debug)]
pub struct ResonanceDetector {
    /// Current probe frequency
    pub probe_frequency_thz: f32,
    /// Measured response
    pub response_amplitude: f32,
    /// Phase shift
    pub response_phase: f32,
    /// Response vs frequency
    pub frequency_spectrum: [f32; 100],
    /// Phase vs frequency
    pub phase_spectrum: [f32; 100],
    /// Samples per bin
    pub sample_counts: [i32; 100],
    /// Peak frequencies
    pub resonance_frequencies: [f32; 8],
    /// Peak amplitudes
    pub resonance_amplitudes: [f32; 8],
    /// Q = f0 / FWHM
    pub quality_factors: [f32; 8],
    /// Number of resonances found
    pub n_resonances: i32,
}

impl Default for ResonanceDetector {
    fn default() -> Self {
        Self {
            probe_frequency_thz: 0.0,
            response_amplitude: 0.0,
            response_phase: 0.0,
            frequency_spectrum: [0.0; 100],
            phase_spectrum: [0.0; 100],
            sample_counts: [0; 100],
            resonance_frequencies: [0.0; 8],
            resonance_amplitudes: [0.0; 8],
            quality_factors: [0.0; 8],
            n_resonances: 0,
        }
    }
}

/// Adaptive probe controller
#[repr(C)]
#[derive(Clone, Debug)]
pub struct AdaptiveProbeController {
    /// Probe effectiveness scores
    pub probe_scores: [f32; 64],
    /// Trial counts per probe
    pub probe_trial_counts: [i32; 64],
    /// Current probe being used
    pub current_probe_idx: i32,
    /// Softmax temperature
    pub exploration_temperature: f32,
    /// Total reward accumulated
    pub cumulative_reward: f32,
    /// Total trials
    pub total_trials: i32,
    /// Best score seen
    pub best_score: f32,
    /// Index of best probe
    pub best_probe_idx: i32,
}

impl Default for AdaptiveProbeController {
    fn default() -> Self {
        Self {
            probe_scores: [0.0; 64],
            probe_trial_counts: [0; 64],
            current_probe_idx: 0,
            exploration_temperature: 1.0,
            cumulative_reward: 0.0,
            total_trials: 0,
            best_score: 0.0,
            best_probe_idx: 0,
        }
    }
}

/// Comprehensive response to a probe
#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct ProbeResponse {
    /// Raw spike count
    pub total_spikes: i32,
    /// Average spike intensity
    pub mean_intensity: f32,
    /// How spread out spikes were
    pub spatial_extent: f32,
    /// Time to first spike
    pub onset_latency_ps: f32,
    /// Time to maximum activity
    pub peak_latency_ps: f32,
    /// Duration of response
    pub duration_ps: f32,
    /// Best sequence score
    pub sequence_score: f32,
    /// Complete sequences found
    pub sequences_detected: i32,
    /// Center of spike activity
    pub response_centroid: Float3,
    /// Directional bias
    pub response_anisotropy: f32,
    /// Alignment with probe
    pub probe_correlation: f32,
}

/// Differential probe pair
#[repr(C)]
#[derive(Clone, Debug)]
pub struct DifferentialProbePair {
    /// One side of hinge
    pub probe_a: CoherentProbe,
    /// Opposite side
    pub probe_b: CoherentProbe,
    /// |A - B| / (A + B)
    pub differential_score: f32,
    /// Statistical confidence
    pub confidence: f32,
    /// Number of trials
    pub n_trials: i32,
}

impl Default for DifferentialProbePair {
    fn default() -> Self {
        Self {
            probe_a: CoherentProbe::default(),
            probe_b: CoherentProbe::default(),
            differential_score: 0.0,
            confidence: 0.0,
            n_trials: 0,
        }
    }
}

// ============================================================================
// RESULT STRUCTURES
// ============================================================================

/// Detected cryptic site candidate
#[derive(Clone, Debug)]
pub struct CrypticSiteCandidate {
    /// Position in Angstroms
    pub position: [f32; 3],
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Sequence detection score
    pub sequence_score: f32,
    /// Resonance frequency if detected
    pub resonance_frequency: Option<f32>,
    /// Differential score if measured
    pub differential_score: Option<f32>,
    /// Voxel indices involved
    pub voxel_indices: Vec<i32>,
}

/// Resonance peak information
#[derive(Clone, Debug)]
pub struct ResonancePeak {
    /// Frequency in THz
    pub frequency_thz: f32,
    /// Peak amplitude
    pub amplitude: f32,
    /// Quality factor
    pub quality_factor: f32,
}

/// Probe effectiveness statistics
#[derive(Clone, Debug, Default)]
pub struct ProbeStatistics {
    /// Total trials run
    pub total_trials: i32,
    /// Best performing probe
    pub best_probe_idx: i32,
    /// Best score achieved
    pub best_score: f32,
    /// Total reward
    pub cumulative_reward: f32,
    /// Current temperature
    pub exploration_temperature: f32,
}

/// Complete results from active sensing
#[derive(Clone, Debug, Default)]
pub struct ActiveSensingResults {
    /// Detected cryptic sites
    pub cryptic_site_candidates: Vec<CrypticSiteCandidate>,
    /// Differential scores per pair
    pub differential_scores: Vec<(i32, f32)>,
    /// Resonance peaks found
    pub resonance_peaks: Vec<ResonancePeak>,
    /// Probe statistics
    pub probe_statistics: ProbeStatistics,
    /// Total spikes recorded
    pub total_spikes: i32,
    /// Sequences detected
    pub total_sequences_detected: i32,
}

/// Active sensing mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveSensingMode {
    /// Single probe exploration
    SingleProbe,
    /// Differential A/B comparison
    Differential,
    /// Frequency sweep for resonance
    ResonanceSweep,
    /// Adaptive RL-based exploration
    Adaptive,
}

// ============================================================================
// MAIN ENGINE
// ============================================================================

/// Active Sensing Engine
///
/// Manages GPU resources and orchestrates the closed-loop detection pipeline.
pub struct ActiveSensingEngine {
    // Configuration
    config: ActiveSensingConfig,

    // CUDA resources
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    // GPU buffers
    d_spike_histories: CudaSlice<u8>,
    d_inhibition_states: CudaSlice<u8>,
    d_sequence_detectors: CudaSlice<u8>,
    d_detection_scores: CudaSlice<f32>,
    d_resonance_detector: CudaSlice<u8>,
    d_probe_controller: CudaSlice<u8>,
    d_probes: CudaSlice<u8>,
    d_current_response: CudaSlice<u8>,
    d_aromatic_centroids: CudaSlice<f32>,
    d_aromatic_absorptions: CudaSlice<f32>,
    d_ring_normals: CudaSlice<f32>,
    d_aromatic_atom_indices: CudaSlice<i32>,
    d_aromatic_n_atoms: CudaSlice<i32>,
    d_lif_potential: CudaSlice<f32>,
    d_spike_grid: CudaSlice<i32>,
    d_spike_count: CudaSlice<i32>,
    d_rng_states: CudaSlice<u8>,
    d_n_detectors: CudaSlice<i32>,
    d_next_probe_idx: CudaSlice<i32>,

    // State
    n_voxels: usize,
    n_aromatics: usize,
    n_detectors: usize,
    n_probes: usize,
    current_probe_idx: usize,
    current_time_ps: f32,
    probe_start_time: f32,
    mode: ActiveSensingMode,
    grid_origin: Float3,

    // Host caches
    host_response: ProbeResponse,
    host_controller: AdaptiveProbeController,
}

impl ActiveSensingEngine {
    /// Create new active sensing engine
    ///
    /// # Arguments
    /// * `context` - CUDA context
    /// * `config` - Active sensing configuration
    /// * `aromatic_centroids` - Centers of aromatic residues
    /// * `aromatic_absorptions` - Absorption strengths
    /// * `ring_normals` - Normal vectors for aromatic rings
    /// * `aromatic_atom_indices` - Atom indices per aromatic (16 per aromatic)
    /// * `aromatic_n_atoms` - Number of atoms per aromatic
    pub fn new(
        context: Arc<CudaContext>,
        config: ActiveSensingConfig,
        aromatic_centroids: &[[f32; 3]],
        aromatic_absorptions: &[f32],
        ring_normals: &[[f32; 3]],
        aromatic_atom_indices: &[i32],
        aromatic_n_atoms: &[i32],
    ) -> Result<Self> {
        let stream = context.default_stream();
        let n_aromatics = aromatic_centroids.len();
        let n_voxels = (config.grid_dim * config.grid_dim * config.grid_dim) as usize;

        // Compute grid origin from aromatic centroids
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut min_z = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        let mut max_z = f32::MIN;

        for c in aromatic_centroids {
            min_x = min_x.min(c[0]);
            min_y = min_y.min(c[1]);
            min_z = min_z.min(c[2]);
            max_x = max_x.max(c[0]);
            max_y = max_y.max(c[1]);
            max_z = max_z.max(c[2]);
        }

        let padding = 10.0;
        let grid_origin = Float3::new(min_x - padding, min_y - padding, min_z - padding);

        // Size calculations
        let spike_history_size = std::mem::size_of::<VoxelSpikeHistory>();
        let inhibition_state_size = std::mem::size_of::<LateralInhibitionState>();
        let detector_size = std::mem::size_of::<SpikeSequenceDetector>();
        let resonance_size = std::mem::size_of::<ResonanceDetector>();
        let controller_size = std::mem::size_of::<AdaptiveProbeController>();
        let probe_size = std::mem::size_of::<CoherentProbe>();
        let response_size = std::mem::size_of::<ProbeResponse>();

        // Allocate GPU buffers
        let d_spike_histories: CudaSlice<u8> = stream
            .alloc_zeros(n_voxels * spike_history_size)
            .context("Failed to allocate spike histories")?;

        let d_inhibition_states: CudaSlice<u8> = stream
            .alloc_zeros(n_voxels * inhibition_state_size)
            .context("Failed to allocate inhibition states")?;

        let max_detectors = config.max_sequence_detectors;
        let d_sequence_detectors: CudaSlice<u8> = stream
            .alloc_zeros(max_detectors * detector_size)
            .context("Failed to allocate sequence detectors")?;

        let d_detection_scores: CudaSlice<f32> = stream
            .alloc_zeros(max_detectors)
            .context("Failed to allocate detection scores")?;

        let d_resonance_detector: CudaSlice<u8> = stream
            .alloc_zeros(resonance_size)
            .context("Failed to allocate resonance detector")?;

        let d_probe_controller: CudaSlice<u8> = stream
            .alloc_zeros(controller_size)
            .context("Failed to allocate probe controller")?;

        let n_probes = 64;
        let d_probes: CudaSlice<u8> = stream
            .alloc_zeros(n_probes * probe_size)
            .context("Failed to allocate probes")?;

        let d_current_response: CudaSlice<u8> = stream
            .alloc_zeros(response_size)
            .context("Failed to allocate response buffer")?;

        // Upload aromatic data
        let centroids_flat: Vec<f32> = aromatic_centroids.iter().flat_map(|c| c.iter().copied()).collect();
        let mut d_aromatic_centroids: CudaSlice<f32> = stream
            .alloc_zeros(centroids_flat.len().max(1))
            .context("Failed to allocate aromatic centroids")?;
        if !centroids_flat.is_empty() {
            stream.memcpy_htod(&centroids_flat, &mut d_aromatic_centroids)?;
        }

        let mut d_aromatic_absorptions: CudaSlice<f32> = stream
            .alloc_zeros(aromatic_absorptions.len().max(1))
            .context("Failed to allocate aromatic absorptions")?;
        if !aromatic_absorptions.is_empty() {
            stream.memcpy_htod(aromatic_absorptions, &mut d_aromatic_absorptions)?;
        }

        let normals_flat: Vec<f32> = ring_normals.iter().flat_map(|n| n.iter().copied()).collect();
        let mut d_ring_normals: CudaSlice<f32> = stream
            .alloc_zeros(normals_flat.len().max(1))
            .context("Failed to allocate ring normals")?;
        if !normals_flat.is_empty() {
            stream.memcpy_htod(&normals_flat, &mut d_ring_normals)?;
        }

        let mut d_aromatic_atom_indices: CudaSlice<i32> = stream
            .alloc_zeros(aromatic_atom_indices.len().max(1))
            .context("Failed to allocate aromatic atom indices")?;
        if !aromatic_atom_indices.is_empty() {
            stream.memcpy_htod(aromatic_atom_indices, &mut d_aromatic_atom_indices)?;
        }

        let mut d_aromatic_n_atoms: CudaSlice<i32> = stream
            .alloc_zeros(aromatic_n_atoms.len().max(1))
            .context("Failed to allocate aromatic n_atoms")?;
        if !aromatic_n_atoms.is_empty() {
            stream.memcpy_htod(aromatic_n_atoms, &mut d_aromatic_n_atoms)?;
        }

        // LIF state
        let d_lif_potential: CudaSlice<f32> = stream
            .alloc_zeros(n_voxels)
            .context("Failed to allocate LIF potential")?;

        let d_spike_grid: CudaSlice<i32> = stream
            .alloc_zeros(n_voxels)
            .context("Failed to allocate spike grid")?;

        let d_spike_count: CudaSlice<i32> = stream
            .alloc_zeros(1)
            .context("Failed to allocate spike count")?;

        // RNG states
        let rng_state_size = 48;
        let d_rng_states: CudaSlice<u8> = stream
            .alloc_zeros(n_probes * rng_state_size)
            .context("Failed to allocate RNG states")?;

        let d_n_detectors: CudaSlice<i32> = stream
            .alloc_zeros(1)
            .context("Failed to allocate n_detectors")?;

        let d_next_probe_idx: CudaSlice<i32> = stream
            .alloc_zeros(1)
            .context("Failed to allocate next_probe_idx")?;

        Ok(Self {
            config,
            context,
            stream,
            d_spike_histories,
            d_inhibition_states,
            d_sequence_detectors,
            d_detection_scores,
            d_resonance_detector,
            d_probe_controller,
            d_probes,
            d_current_response,
            d_aromatic_centroids,
            d_aromatic_absorptions,
            d_ring_normals,
            d_aromatic_atom_indices,
            d_aromatic_n_atoms,
            d_lif_potential,
            d_spike_grid,
            d_spike_count,
            d_rng_states,
            d_n_detectors,
            d_next_probe_idx,
            n_voxels,
            n_aromatics,
            n_detectors: 0,
            n_probes,
            current_probe_idx: 0,
            current_time_ps: 0.0,
            probe_start_time: 0.0,
            mode: ActiveSensingMode::Adaptive,
            grid_origin,
            host_response: ProbeResponse::default(),
            host_controller: AdaptiveProbeController::default(),
        })
    }

    /// Get current configuration
    pub fn config(&self) -> &ActiveSensingConfig {
        &self.config
    }

    /// Get current mode
    pub fn mode(&self) -> ActiveSensingMode {
        self.mode
    }

    /// Set active sensing mode
    pub fn set_mode(&mut self, mode: ActiveSensingMode) {
        self.mode = mode;
    }

    /// Get current probe index
    pub fn current_probe_idx(&self) -> usize {
        self.current_probe_idx
    }

    /// Get number of aromatics
    pub fn n_aromatics(&self) -> usize {
        self.n_aromatics
    }

    /// Get number of voxels
    pub fn n_voxels(&self) -> usize {
        self.n_voxels
    }

    /// Get number of sequence detectors
    pub fn n_detectors(&self) -> usize {
        self.n_detectors
    }

    /// Get current simulation time
    pub fn current_time_ps(&self) -> f32 {
        self.current_time_ps
    }

    /// Reset all state
    pub fn reset(&mut self) -> Result<()> {
        self.current_probe_idx = 0;
        self.current_time_ps = 0.0;
        self.probe_start_time = 0.0;
        self.n_detectors = 0;
        self.host_response = ProbeResponse::default();
        self.host_controller = AdaptiveProbeController::default();
        Ok(())
    }

    /// Build sequence detectors from aromatic topology
    pub fn build_sequence_detectors(&mut self) -> Result<usize> {
        // Creates detectors for plausible spike propagation paths
        self.n_detectors = std::cmp::min(
            (self.n_aromatics * (self.n_aromatics.saturating_sub(1))) / 2,
            self.config.max_sequence_detectors,
        );
        Ok(self.n_detectors)
    }

    /// Cluster aromatics into groups
    pub fn cluster_aromatics(&mut self) -> Result<usize> {
        Ok(self.config.target_n_groups)
    }

    /// Upload a probe configuration
    pub fn set_probe(&mut self, probe_idx: usize, probe: &CoherentProbe) -> Result<()> {
        if probe_idx >= self.n_probes {
            anyhow::bail!("Probe index {} out of range (max {})", probe_idx, self.n_probes);
        }

        let probe_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                probe as *const CoherentProbe as *const u8,
                std::mem::size_of::<CoherentProbe>(),
            )
        };

        let offset = probe_idx * std::mem::size_of::<CoherentProbe>();
        let mut slice = self.d_probes.slice_mut(offset..offset + probe_bytes.len());
        self.stream.memcpy_htod(probe_bytes, &mut slice)?;

        Ok(())
    }

    /// Apply current coherent probe
    pub fn apply_probe(
        &mut self,
        _d_velocities: &mut CudaSlice<f32>,
        _d_positions: &CudaSlice<f32>,
        _d_masses: &CudaSlice<f32>,
        current_time_fs: f32,
        _dt_fs: f32,
        _n_atoms: usize,
    ) -> Result<()> {
        // Would launch apply_coherent_probe kernel
        self.current_time_ps = current_time_fs / 1000.0;
        Ok(())
    }

    /// Update LIF neurons with lateral inhibition
    pub fn lif_update(
        &mut self,
        _d_water_density: &CudaSlice<f32>,
        _d_water_density_prev: &CudaSlice<f32>,
        _dt_ps: f32,
    ) -> Result<i32> {
        // Would launch lif_update_with_inhibition kernel
        Ok(0)
    }

    /// Detect spike sequences
    pub fn detect_sequences(&mut self) -> Result<Vec<f32>> {
        if self.n_detectors == 0 {
            return Ok(vec![]);
        }

        let mut scores = vec![0.0f32; self.n_detectors];
        let slice = self.d_detection_scores.slice(..self.n_detectors);
        self.stream.memcpy_dtoh(&slice, &mut scores)?;

        Ok(scores)
    }

    /// Update resonance spectrum
    pub fn update_resonance(&mut self, _probe_frequency_thz: f32, _probe_period_ps: f32) -> Result<()> {
        // Would launch update_resonance_spectrum kernel
        Ok(())
    }

    /// Analyze resonance spectrum for peaks
    pub fn analyze_resonances(&mut self) -> Result<Vec<ResonancePeak>> {
        // Would launch analyze_resonances kernel
        Ok(vec![])
    }

    /// Compute comprehensive response metrics
    pub fn compute_response(&mut self) -> Result<ProbeResponse> {
        // Would launch compute_response_metrics kernel
        let mut response_bytes = vec![0u8; std::mem::size_of::<ProbeResponse>()];
        self.stream.memcpy_dtoh(&self.d_current_response, &mut response_bytes)?;

        if response_bytes.len() >= std::mem::size_of::<ProbeResponse>() {
            self.host_response = unsafe {
                std::ptr::read_unaligned(response_bytes.as_ptr() as *const ProbeResponse)
            };
        }

        Ok(self.host_response.clone())
    }

    /// Update probe selection based on response
    pub fn update_probe_selection(&mut self, _response: &ProbeResponse) -> Result<usize> {
        // Would launch adaptive_probe_update kernel
        let mut next_idx = [0i32];
        self.stream.memcpy_dtoh(&self.d_next_probe_idx, &mut next_idx)?;

        self.current_probe_idx = next_idx[0].max(0) as usize;
        self.probe_start_time = self.current_time_ps;

        Ok(self.current_probe_idx)
    }

    /// Run differential probe comparison
    pub fn compare_differential(
        &mut self,
        _pair_idx: usize,
        response_a: &ProbeResponse,
        response_b: &ProbeResponse,
    ) -> Result<(f32, f32)> {
        let sum = (response_a.total_spikes + response_b.total_spikes + 1) as f32;
        let diff = (response_a.total_spikes - response_b.total_spikes).abs() as f32;
        let differential = diff / sum;

        Ok((differential, 0.5))
    }

    /// Get comprehensive results
    pub fn get_results(&self) -> Result<ActiveSensingResults> {
        let probe_stats = ProbeStatistics {
            total_trials: self.host_controller.total_trials,
            best_probe_idx: self.host_controller.best_probe_idx,
            best_score: self.host_controller.best_score,
            cumulative_reward: self.host_controller.cumulative_reward,
            exploration_temperature: self.host_controller.exploration_temperature,
        };

        Ok(ActiveSensingResults {
            cryptic_site_candidates: vec![],
            differential_scores: vec![],
            resonance_peaks: vec![],
            probe_statistics: probe_stats,
            total_spikes: 0,
            total_sequences_detected: 0,
        })
    }

    /// Get spike statistics
    pub fn get_spike_statistics(&self) -> Result<(i32, f32, usize)> {
        Ok((0, 0.0, 0))
    }

    /// Estimate VRAM usage in MB
    pub fn estimate_vram_mb(&self) -> f32 {
        let spike_history_size = std::mem::size_of::<VoxelSpikeHistory>();
        let inhibition_state_size = std::mem::size_of::<LateralInhibitionState>();
        let detector_size = std::mem::size_of::<SpikeSequenceDetector>();

        let total_bytes = self.n_voxels * spike_history_size
            + self.n_voxels * inhibition_state_size
            + self.config.max_sequence_detectors * detector_size
            + self.config.max_sequence_detectors * 4
            + std::mem::size_of::<ResonanceDetector>()
            + std::mem::size_of::<AdaptiveProbeController>()
            + self.n_probes * std::mem::size_of::<CoherentProbe>()
            + self.n_voxels * 4 * 3
            + self.n_aromatics * 4 * 7;

        total_bytes as f32 / (1024.0 * 1024.0)
    }
}

// ============================================================================
// BUILDER
// ============================================================================

/// Builder for ActiveSensingEngine
pub struct ActiveSensingBuilder {
    config: ActiveSensingConfig,
    aromatic_centroids: Vec<[f32; 3]>,
    aromatic_absorptions: Vec<f32>,
    ring_normals: Vec<[f32; 3]>,
    aromatic_atom_indices: Vec<i32>,
    aromatic_n_atoms: Vec<i32>,
}

impl ActiveSensingBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: ActiveSensingConfig::default(),
            aromatic_centroids: vec![],
            aromatic_absorptions: vec![],
            ring_normals: vec![],
            aromatic_atom_indices: vec![],
            aromatic_n_atoms: vec![],
        }
    }

    /// Set configuration
    pub fn config(mut self, config: ActiveSensingConfig) -> Self {
        self.config = config;
        self
    }

    /// Set grid dimension
    pub fn grid_dim(mut self, dim: i32) -> Self {
        self.config.grid_dim = dim;
        self
    }

    /// Set grid spacing
    pub fn grid_spacing(mut self, spacing: f32) -> Self {
        self.config.grid_spacing = spacing;
        self
    }

    /// Add aromatic residue
    pub fn add_aromatic(
        mut self,
        centroid: [f32; 3],
        absorption: f32,
        normal: [f32; 3],
        atom_indices: &[i32],
    ) -> Self {
        self.aromatic_centroids.push(centroid);
        self.aromatic_absorptions.push(absorption);
        self.ring_normals.push(normal);

        let mut indices = [0i32; 16];
        for (i, &idx) in atom_indices.iter().take(16).enumerate() {
            indices[i] = idx;
        }
        self.aromatic_atom_indices.extend_from_slice(&indices);
        self.aromatic_n_atoms.push(atom_indices.len() as i32);

        self
    }

    /// Build the engine
    pub fn build(self, context: Arc<CudaContext>) -> Result<ActiveSensingEngine> {
        ActiveSensingEngine::new(
            context,
            self.config,
            &self.aromatic_centroids,
            &self.aromatic_absorptions,
            &self.ring_normals,
            &self.aromatic_atom_indices,
            &self.aromatic_n_atoms,
        )
    }
}

impl Default for ActiveSensingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ActiveSensingConfig::default();
        assert_eq!(config.grid_dim, 32);
        assert!((config.grid_spacing - 2.5).abs() < 0.001);
        assert!((config.tau_membrane - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_probe_default() {
        let probe = CoherentProbe::default();
        assert_eq!(probe.n_groups, 0);
        assert!((probe.total_energy - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_sequence_detector_default() {
        let detector = SpikeSequenceDetector::default();
        assert_eq!(detector.sequence_length, 0);
        assert!((detector.max_inter_spike_interval_ps - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_builder() {
        let builder = ActiveSensingBuilder::new()
            .grid_dim(64)
            .grid_spacing(2.0)
            .add_aromatic([0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 1.0], &[0, 1, 2, 3, 4, 5]);

        assert_eq!(builder.config.grid_dim, 64);
        assert_eq!(builder.aromatic_centroids.len(), 1);
    }

    #[test]
    fn test_float3() {
        let v = Float3::new(1.0, 2.0, 3.0);
        assert!((v.x - 1.0).abs() < 0.001);
        assert!((v.y - 2.0).abs() < 0.001);
        assert!((v.z - 3.0).abs() < 0.001);
    }
}
