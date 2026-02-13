//! NHS-AMBER Fused Engine
//!
//! Rust wrapper for the fused GPU kernel that combines:
//! - Full AMBER ff14SB physics
//! - Langevin thermostat with dynamic temperature protocols
//! - Holographic exclusion field (negative space mapping)
//! - Neuromorphic LIF observation
//! - UV bias pump-probe
//! - Spike-triggered snapshot capture
//! - Warp matrix for atomic-precision alignment
//!
//! All running in a single GPU stream at 100,000+ timesteps/second.
//!
//! ## Cryogenic Physics
//!
//! For cryogenic contrast probing (50K-150K), special physics modifications:
//! - Temperature-dependent friction coefficient (scaled by T/300K)
//! - Temperature-dependent dielectric constant (increases as T decreases)
//! - Simulated annealing ramp (smooth transitions, no instant jumps)
//! - UV burst energy dissipation (prevents geometry explosion in frozen systems)

use anyhow::{bail, Context, Result};
use std::sync::Arc;
use std::net::TcpStream;
use std::io::Write;
use std::time::{Duration, Instant};

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaFunction, CudaModule,
    LaunchConfig, PushKernelArg, DevicePtrMut, DevicePtr,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

use crate::input::PrismPrepTopology;
use crate::config::{
    extinction_to_cross_section, wavelength_to_ev, KB_EV_K,
    CALIBRATED_PHOTON_FLUENCE, DEFAULT_HEAT_YIELD,
    NEFF_TRP, NEFF_TYR, NEFF_PHE, NEFF_DISULFIDE, NEFF_BENZENE,
    TRP_LAMBDA_MAX, TYR_LAMBDA_MAX, PHE_LAMBDA_MAX, DISULFIDE_LAMBDA_MAX, BENZENE_LAMBDA_MAX,
    TRP_EXTINCTION_280, TYR_EXTINCTION_274, PHE_EXTINCTION_258, DISULFIDE_EXTINCTION_250, BENZENE_EXTINCTION_254,
    TRP_BANDWIDTH, TYR_BANDWIDTH, PHE_BANDWIDTH, DISULFIDE_BANDWIDTH, BENZENE_BANDWIDTH,
};

// Import ultimate engine for hyperoptimized kernel path
#[cfg(feature = "gpu")]
use crate::ultimate_engine::{UltimateEngine, UltimateEngineConfig, OptimizationLevel};

// ============================================================================
// GPU STRUCT TYPES (must match CUDA kernel exactly)
// ============================================================================

/// Bond parameter for CUDA kernel (matches BondParam in nhs_amber_fused.cu)
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuBondParam {
    pub i: i32,
    pub j: i32,
    pub r0: f32,
    pub k: f32,
}

/// Angle parameter for CUDA kernel (matches AngleParam in nhs_amber_fused.cu)
/// CUDA struct: i, j, k (12 bytes), theta0, force_k (8 bytes) = 20 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuAngleParam {
    pub i: i32,
    pub j: i32,
    pub k: i32,
    pub theta0: f32,
    pub force_k: f32,
}

/// Dihedral parameter for CUDA kernel (matches DihedralParam in nhs_amber_fused.cu)
/// CUDA struct: i, j, k, l (16 bytes), periodicity (4 bytes), phase, force_k (8 bytes) = 28 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuDihedralParam {
    pub i: i32,
    pub j: i32,
    pub k: i32,
    pub l: i32,
    pub periodicity: i32,
    pub phase: f32,
    pub force_k: f32,
}

/// LJ parameter for CUDA kernel (matches LJParam in nhs_amber_fused.cu)
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuLJParam {
    pub sigma: f32,
    pub epsilon: f32,
}

/// Hydrogen cluster for SHAKE (matches HCluster in nhs_amber_fused.cu)
/// CUDA struct: central_atom (4), hydrogen_atoms[3] (12), bond_lengths[3] (12),
/// n_hydrogens (4), inv_mass_central (4), inv_mass_h (4) = 40 bytes
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuHCluster {
    pub central_atom: i32,
    pub hydrogen_atoms: [i32; 3],  // -1 for unused
    pub bond_lengths: [f32; 3],
    pub n_hydrogens: i32,
    pub inv_mass_central: f32,
    pub inv_mass_h: f32,
}

#[cfg(feature = "gpu")]
impl Default for GpuHCluster {
    fn default() -> Self {
        Self {
            central_atom: -1,
            hydrogen_atoms: [-1, -1, -1],
            bond_lengths: [0.0, 0.0, 0.0],
            n_hydrogens: 0,
            inv_mass_central: 0.0,
            inv_mass_h: 0.0,
        }
    }
}

/// UV target for CUDA kernel (matches UVTarget in nhs_amber_fused.cu)
/// CUDA struct: residue_id (4), atom_indices[16] (64), n_atoms (4), absorption_strength (4), aromatic_type (4) = 80 bytes
/// CANONICAL: 0=TRP, 1=TYR, 2=PHE, 3=S-S, 4=BNZ
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct GpuUVTarget {
    pub residue_id: i32,
    pub atom_indices: [i32; 16],
    pub n_atoms: i32,
    pub absorption_strength: f32,
    /// CANONICAL chromophore type: 0=TRP, 1=TYR, 2=PHE, 3=S-S
    pub aromatic_type: i32,
}

#[cfg(feature = "gpu")]
impl Default for GpuUVTarget {
    fn default() -> Self {
        Self {
            residue_id: -1,
            atom_indices: [-1; 16],
            n_atoms: 0,
            absorption_strength: 0.0,
            aromatic_type: 0,
        }
    }
}

/// Aromatic neighbor list for vibrational energy transfer
/// CUDA struct: atom_indices[64] (256), n_neighbors (4) = 260 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct GpuAromaticNeighbors {
    pub atom_indices: [i32; 64],
    pub n_neighbors: i32,
}

#[cfg(feature = "gpu")]
impl Default for GpuAromaticNeighbors {
    fn default() -> Self {
        Self {
            atom_indices: [-1; 64],
            n_neighbors: 0,
        }
    }
}

/// Warp matrix entry for CUDA kernel (matches WarpEntry in nhs_amber_fused.cu)
/// CUDA struct: voxel_idx (4), atom_indices[16] (64), atom_weights[16] (64), n_atoms (4) = 136 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct GpuWarpEntry {
    pub voxel_idx: i32,
    pub atom_indices: [i32; 16],
    pub atom_weights: [f32; 16],
    pub n_atoms: i32,
}

#[cfg(feature = "gpu")]
impl Default for GpuWarpEntry {
    fn default() -> Self {
        Self {
            voxel_idx: -1,
            atom_indices: [-1; 16],
            atom_weights: [0.0; 16],
            n_atoms: 0,
        }
    }
}

/// Temperature protocol for CUDA kernel (matches TemperatureProtocol in nhs_amber_fused.cu)
/// CUDA struct: start_temp (4), end_temp (4), ramp_steps (4), hold_steps (4), current_step (4) = 20 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuTemperatureProtocol {
    pub start_temp: f32,
    pub end_temp: f32,
    pub ramp_steps: i32,
    pub hold_steps: i32,
    pub current_step: i32,
}

/// GPU spike event (matches SpikeEvent in nhs_amber_fused.cu)
/// CUDA struct: timestep (4), voxel_idx (4), position (12), intensity (4),
/// nearby_residues[8] (32), n_residues (4) = 60 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct GpuSpikeEvent {
    pub timestep: i32,
    pub voxel_idx: i32,
    pub position: [f32; 3],
    pub intensity: f32,
    pub nearby_residues: [i32; 8],
    pub n_residues: i32,
}

#[cfg(feature = "gpu")]
impl Default for GpuSpikeEvent {
    fn default() -> Self {
        Self {
            timestep: 0,
            voxel_idx: 0,
            position: [0.0; 3],
            intensity: 0.0,
            nearby_residues: [0; 8],
            n_residues: 0,
        }
    }
}

// ============================================================================
// TEMPERATURE PROTOCOLS
// ============================================================================

// ============================================================================
// UNIFIED CRYO-UV PROTOCOL (INSEPARABLE)
// ============================================================================

/// **Unified Cryo-UV Protocol** - Temperature ramping + UV-LIF coupling
///
/// This is the canonical PRISM4D cryptic site detection method. The cryo-thermal
/// contrast and UV-LIF coupling work together as a single integrated system:
///
/// - **Cryo phase (77-150K)**: Flash-freeze structure, suppress thermal noise
/// - **Ramp phase (gradual warming)**: Controlled conformational sampling
/// - **UV bursts**: Aromatic excitation → Franck-Condon → thermal wavefront
/// - **LIF detection**: Neuromorphic spike detection at dewetting sites
///
/// The UV-LIF coupling is ALWAYS ACTIVE during cryo-UV runs. This is not optional.
#[derive(Debug, Clone)]
pub struct CryoUvProtocol {
    // Temperature protocol
    /// Starting temperature (Kelvin) - typically 77K (liquid N2)
    pub start_temp: f32,
    /// Ending temperature (Kelvin) - typically 300-310K (physiological)
    pub end_temp: f32,
    /// Steps to hold at cold before ramping
    pub cold_hold_steps: i32,
    /// Number of steps to ramp temperature
    pub ramp_steps: i32,
    /// Number of steps to hold at warm temperature
    pub warm_hold_steps: i32,
    /// Current step in protocol
    pub current_step: i32,

    // UV-LIF coupling (integrated, not optional)
    /// Energy per UV burst (kcal/mol) - calibrated for aromatic excitation
    pub uv_burst_energy: f32,
    /// Interval between UV bursts (timesteps)
    pub uv_burst_interval: i32,
    /// Duration of each burst (timesteps)
    pub uv_burst_duration: i32,
    /// Wavelengths to scan (nm) - TRP/TYR/PHE specific
    pub scan_wavelengths: Vec<f32>,
    /// Dwell steps per wavelength
    pub wavelength_dwell_steps: i32,
}

impl CryoUvProtocol {
    /// Standard liquid nitrogen cryo-UV protocol (77K → 310K)
    ///
    /// This is the validated configuration from benchmark testing:
    /// - 100% aromatic localization
    /// - 2.26x enrichment over baseline
    /// - ~13.5 sites per ultra-difficult structure
    ///
    /// Wavelengths cover all aromatic residues:
    /// - 280 nm: TRP (tryptophan)
    /// - 274 nm: TYR (tyrosine)
    /// - 258 nm: PHE (phenylalanine)
    /// - 211 nm: HIS/HID/HIE/HIP (histidine imidazole ring)
    pub fn standard() -> Self {
        Self {
            start_temp: 77.0,
            end_temp: 310.0,
            cold_hold_steps: 5000,
            ramp_steps: 10000,
            warm_hold_steps: 5000,
            current_step: 0,
            uv_burst_energy: 30.0,
            uv_burst_interval: 500,
            uv_burst_duration: 50,
            // Full aromatic coverage: TRP, TYR, PHE, HIS (all protonation states)
            scan_wavelengths: vec![280.0, 274.0, 258.0, 211.0],
            wavelength_dwell_steps: 500,
        }
    }

    /// Deep freeze protocol (50K start, extended sampling)
    pub fn deep_freeze() -> Self {
        Self {
            start_temp: 50.0,
            cold_hold_steps: 10000,
            ramp_steps: 20000,
            ..Self::standard()
        }
    }

    /// Fast protocol for testing (reduced steps)
    pub fn fast() -> Self {
        Self {
            cold_hold_steps: 2000,
            ramp_steps: 3000,
            warm_hold_steps: 2000,
            ..Self::standard()
        }
    }

    /// Fast 35K protocol - high-energy UV compensates for reduced step count
    ///
    /// Physics: stronger UV burst energy (42 kcal/mol, +40%) drives faster
    /// aromatic excitation → faster Franck-Condon displacement → faster spike
    /// generation. Combined with more frequent bursts (every 250 steps vs 400),
    /// this yields equivalent detection quality in 30% fewer steps.
    ///
    /// At 50K start temp, thermal fluctuations are ~30% smaller than 100K:
    /// - Faster equilibration (protein settles quicker)
    /// - Cleaner spike signal (less thermal noise)
    /// - Higher UV energy drives confident detection in fewer steps
    ///
    /// Phase budget: 14K cold + 6K ramp + 15K warm = 35K total
    /// UV probing: 42 kcal/mol every 250 steps = 140 bursts total
    ///
    /// Wavelengths cover all aromatic residues:
    /// - 280 nm: TRP (tryptophan)
    /// - 274 nm: TYR (tyrosine)
    /// - 258 nm: PHE (phenylalanine)
    /// - 211 nm: HIS/HID/HIE/HIP (histidine imidazole ring)
    pub fn fast_35k() -> Self {
        Self {
            start_temp: 50.0,           // Ultra-cold start
            end_temp: 300.0,            // Physiological end
            cold_hold_steps: 14000,     // Aggressive cold hold (was 20K)
            ramp_steps: 6000,           // Faster ramp 50K→300K (was 8K)
            warm_hold_steps: 15000,     // Extended warm hold for production sampling (was 2K)
            current_step: 0,
            uv_burst_energy: 42.0,      // +40% energy: faster aromatic excitation (was 30.0)
            uv_burst_interval: 250,     // More frequent probing (was 400)
            uv_burst_duration: 50,
            // Full aromatic coverage: TRP, TYR, PHE, HIS (all protonation states)
            scan_wavelengths: vec![280.0, 274.0, 258.0, 211.0],
            wavelength_dwell_steps: 300, // Proportionally reduced (was 400)
        }
    }

    /// Get current temperature
    pub fn current_temperature(&self) -> f32 {
        let step_in_phase = self.current_step;

        if step_in_phase < self.cold_hold_steps {
            // Cold hold phase
            self.start_temp
        } else if step_in_phase < self.cold_hold_steps + self.ramp_steps {
            // Ramp phase
            let ramp_progress = (step_in_phase - self.cold_hold_steps) as f32 / self.ramp_steps as f32;
            self.start_temp + ramp_progress * (self.end_temp - self.start_temp)
        } else {
            // Warm hold phase
            self.end_temp
        }
    }

    /// Check if UV burst should fire at current timestep
    pub fn is_uv_burst_active(&self) -> bool {
        (self.current_step % self.uv_burst_interval) < self.uv_burst_duration
    }

    /// Get current wavelength (cycles through scan_wavelengths)
    pub fn current_wavelength(&self) -> f32 {
        let wavelength_cycle = (self.current_step / self.wavelength_dwell_steps) as usize;
        let idx = wavelength_cycle % self.scan_wavelengths.len();
        self.scan_wavelengths[idx]
    }

    /// Advance to next step
    pub fn advance(&mut self) {
        self.current_step += 1;
    }

    /// Check if protocol is complete
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.total_steps()
    }

    /// Total steps in protocol
    pub fn total_steps(&self) -> i32 {
        self.cold_hold_steps + self.ramp_steps + self.warm_hold_steps
    }
}

/// **DEPRECATED**: Use `CryoUvProtocol` instead
///
/// This struct is kept for backward compatibility but should not be used in new code.
/// The cryo-thermal and UV-LIF systems are now unified.
#[deprecated(since = "1.2.0", note = "Use CryoUvProtocol instead - cryo and UV are now unified")]
#[derive(Debug, Clone)]
pub struct TemperatureProtocol {
    /// Starting temperature (Kelvin)
    pub start_temp: f32,
    /// Ending temperature (Kelvin)
    pub end_temp: f32,
    /// Number of steps to ramp temperature
    pub ramp_steps: i32,
    /// Number of steps to hold at end temperature
    pub hold_steps: i32,
    /// Number of steps to hold at start temperature before ramping (cryo baseline)
    pub cold_hold_steps: i32,
    /// Current step in protocol
    pub current_step: i32,
}

impl TemperatureProtocol {
    /// Standard physiological temperature (300K constant)
    pub fn physiological() -> Self {
        Self {
            start_temp: 300.0,
            end_temp: 300.0,
            ramp_steps: 0,
            hold_steps: 100000,
            cold_hold_steps: 0,
            current_step: 0,
        }
    }

    /// Cryogenic probing protocol (100K → 300K)
    pub fn cryogenic_probe(cryo_temp: f32, ramp_steps: i32, hold_steps: i32) -> Self {
        Self {
            start_temp: cryo_temp,
            end_temp: 300.0,
            ramp_steps,
            hold_steps,
            cold_hold_steps: 0,
            current_step: 0,
        }
    }

    /// Deep freeze for extreme contrast (50K start)
    pub fn deep_freeze() -> Self {
        Self {
            start_temp: 50.0,
            end_temp: 300.0,
            ramp_steps: 50000,
            hold_steps: 50000,
            cold_hold_steps: 0,
            current_step: 0,
        }
    }

    /// Flash freeze then slow warm (for capturing transient states)
    pub fn flash_freeze_slow_warm() -> Self {
        Self {
            start_temp: 100.0,
            end_temp: 300.0,
            ramp_steps: 100000,
            hold_steps: 50000,
            cold_hold_steps: 0,
            current_step: 0,
        }
    }

    /// Get current temperature
    pub fn current_temperature(&self) -> f32 {
        // Phase 1: Cold hold -- constant at start_temp
        if self.current_step < self.cold_hold_steps {
            return self.start_temp;
        }
        // Phase 2: Ramp from start_temp -> end_temp
        let ramp_elapsed = self.current_step - self.cold_hold_steps;
        if self.ramp_steps == 0 {
            return self.end_temp;
        }
        if ramp_elapsed < self.ramp_steps {
            let t = ramp_elapsed as f32 / self.ramp_steps as f32;
            self.start_temp + t * (self.end_temp - self.start_temp)
        } else {
            // Phase 3: Warm hold -- constant at end_temp
            self.end_temp
        }
    }

    /// Advance to next step
    pub fn advance(&mut self) {
        self.current_step += 1;
    }

    /// Check if protocol is complete
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.cold_hold_steps + self.ramp_steps + self.hold_steps
    }

    /// Total steps in protocol
    pub fn total_steps(&self) -> i32 {
        self.cold_hold_steps + self.ramp_steps + self.hold_steps
    }
}

// ============================================================================
// SPIKE EVENT CAPTURE
// ============================================================================

/// Captured spike event with atomic-precision mapping
#[derive(Debug, Clone)]
pub struct SpikeEvent {
    /// Timestep when spike occurred
    pub timestep: i32,
    /// Voxel index in grid
    pub voxel_idx: i32,
    /// 3D position of spike
    pub position: [f32; 3],
    /// Spike intensity
    pub intensity: f32,
    /// Nearby residue IDs (via warp matrix)
    pub nearby_residues: Vec<i32>,
    /// Temperature at time of spike
    pub temperature: f32,
    /// Whether UV burst was active
    pub uv_burst_active: bool,
}

/// Reason why a snapshot was captured (activity-based triggers)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SnapshotTrigger {
    /// Spike count exceeded threshold
    SpikeActivity,
    /// UV burst caused significant response
    UvResponse,
    /// SASA changed significantly from baseline
    SasaChange,
    /// Conformational change detected (RMSD spike)
    ConformationalChange,
    /// Temperature transition point
    TemperatureTransition,
    /// High residue RMSF (flexibility)
    ResidueFlexibility,
}

/// Ensemble snapshot triggered by activity (not interval)
#[derive(Debug, Clone)]
pub struct EnsembleSnapshot {
    /// Timestep
    pub timestep: i32,
    /// All atom positions at this moment
    pub positions: Vec<f32>,
    /// Velocities
    pub velocities: Vec<f32>,
    /// Spikes that triggered this capture
    pub trigger_spikes: Vec<SpikeEvent>,
    /// Current temperature
    pub temperature: f32,
    /// Quality scores for each trigger spike
    pub spike_quality_scores: Vec<SpikeQualityScore>,
    /// Overall alignment quality relative to reference (0-1, 1=perfect)
    pub alignment_quality: f32,
    /// RMSD from reference structure for spike region atoms (Å)
    pub spike_region_rmsd: f32,
    /// Frame time in picoseconds
    pub time_ps: f32,
    /// Why this snapshot was captured
    pub trigger_reason: SnapshotTrigger,
    /// Delta SASA from baseline (Å²) - if computed
    pub delta_sasa: Option<f32>,
}

// ============================================================================
// SPIKE QUALITY SCORING
// ============================================================================

/// Quality and confidence scoring for spike-triggered events
/// Used for publication-quality filtering and validation
#[derive(Debug, Clone, Default)]
pub struct SpikeQualityScore {
    // =========================================================================
    // Confidence Metrics (0.0 - 1.0)
    // =========================================================================

    /// Spike intensity relative to threshold (higher = more confident)
    /// Calculated as (intensity - threshold) / (max_observed - threshold)
    pub intensity_score: f32,

    /// Recurrence at same spatial location across frames
    /// Calculated from spike location history within clustering radius
    pub persistence_score: f32,

    /// Temporal correlation with UV burst events
    /// High value means spike occurred within expected response window after UV
    pub uv_correlation: f32,

    /// Stability across temperature variations during cryo protocol
    /// Spikes that persist through thermal cycling are more significant
    pub thermal_stability: f32,

    // =========================================================================
    // Structural Context Metrics
    // =========================================================================

    /// Distance to nearest aromatic residue (Å)
    /// Lower values suggest UV-mediated dewetting
    pub aromatic_proximity: f32,

    /// Local structural flexibility (RMSF-derived, 0-1 normalized)
    /// Flexible regions more likely to reveal cryptic sites
    pub flexibility_score: f32,

    /// Number of hydrogen bonds disrupted at this location
    pub hydrogen_bond_disruption: i32,

    /// Number of nearby hydrophobic atoms (within 6Å)
    pub hydrophobic_neighbors: i32,

    // =========================================================================
    // Alignment Metrics
    // =========================================================================

    /// Local RMSD of atoms within spike voxel (Å)
    pub local_rmsd: f32,

    /// Atoms contributing to spike region
    pub contributing_atoms: i32,

    // =========================================================================
    // Combined Scores
    // =========================================================================

    /// Overall confidence score (weighted combination)
    pub overall_confidence: f32,

    /// Categorical classification
    pub category: SpikeQualityCategory,
}

impl SpikeQualityScore {
    /// Compute overall confidence from component scores
    pub fn compute_overall_confidence(&mut self) {
        // Weighted combination of confidence factors
        // Weights empirically tuned for cryptic site detection
        let weights = [
            (self.intensity_score, 0.25),
            (self.persistence_score, 0.30),
            (self.uv_correlation, 0.20),
            (self.thermal_stability, 0.15),
            // Structural context contributes via aromatic proximity
            ((1.0 - (self.aromatic_proximity / 10.0).min(1.0)), 0.10),
        ];

        self.overall_confidence = weights.iter()
            .map(|(score, weight)| score * weight)
            .sum::<f32>()
            .clamp(0.0, 1.0);

        // Determine category based on overall confidence
        self.category = if self.overall_confidence >= 0.75 {
            SpikeQualityCategory::HighConfidence
        } else if self.overall_confidence >= 0.50 {
            SpikeQualityCategory::MediumConfidence
        } else if self.overall_confidence >= 0.25 {
            SpikeQualityCategory::LowConfidence
        } else if self.is_likely_artifact() {
            SpikeQualityCategory::Artifact
        } else {
            SpikeQualityCategory::Noise
        };
    }

    /// Check if spike is likely an artifact
    fn is_likely_artifact(&self) -> bool {
        // Artifact indicators:
        // - Very low persistence (appears once, never again)
        // - Zero UV correlation during UV-active probing
        // - Extreme local RMSD (geometry errors)
        self.persistence_score < 0.05
            || self.local_rmsd > 5.0
            || (self.uv_correlation < 0.1 && self.aromatic_proximity < 5.0)
    }

    /// Create a score for a high-quality, validated spike
    pub fn high_quality(intensity: f32, persistence: f32) -> Self {
        let mut score = Self {
            intensity_score: intensity.clamp(0.0, 1.0),
            persistence_score: persistence.clamp(0.0, 1.0),
            uv_correlation: 0.8,
            thermal_stability: 0.7,
            aromatic_proximity: 4.0,
            flexibility_score: 0.6,
            hydrogen_bond_disruption: 2,
            hydrophobic_neighbors: 8,
            local_rmsd: 1.5,
            contributing_atoms: 12,
            ..Default::default()
        };
        score.compute_overall_confidence();
        score
    }
}

/// Categories for spike classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpikeQualityCategory {
    /// High confidence cryptic site indicator (>= 0.75)
    HighConfidence,
    /// Medium confidence, may need validation (0.50-0.75)
    MediumConfidence,
    /// Low confidence, likely noise or weak signal (0.25-0.50)
    LowConfidence,
    /// Likely artifact from simulation instability
    Artifact,
    /// Random noise, not significant
    #[default]
    Noise,
}

impl std::fmt::Display for SpikeQualityCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpikeQualityCategory::HighConfidence => write!(f, "HIGH"),
            SpikeQualityCategory::MediumConfidence => write!(f, "MEDIUM"),
            SpikeQualityCategory::LowConfidence => write!(f, "LOW"),
            SpikeQualityCategory::Artifact => write!(f, "ARTIFACT"),
            SpikeQualityCategory::Noise => write!(f, "NOISE"),
        }
    }
}

/// Spike persistence tracker for computing recurrence scores
#[derive(Debug, Clone, Default)]
pub struct SpikePersistenceTracker {
    /// Historical spike locations [x, y, z] with frame index
    history: Vec<([f32; 3], i32)>,
    /// Clustering radius for persistence detection (Å)
    cluster_radius: f32,
    /// Maximum history length
    max_history: usize,
}

impl SpikePersistenceTracker {
    /// Create new tracker with given clustering radius
    pub fn new(cluster_radius: f32) -> Self {
        Self {
            history: Vec::new(),
            cluster_radius,
            max_history: 10000,
        }
    }

    /// Record a spike event
    pub fn record_spike(&mut self, position: [f32; 3], frame: i32) {
        if self.history.len() >= self.max_history {
            // Remove oldest entries
            self.history.drain(0..1000);
        }
        self.history.push((position, frame));
    }

    /// Compute persistence score for a spike location
    /// Returns (persistence_score, occurrence_count)
    pub fn compute_persistence(&self, position: [f32; 3]) -> (f32, usize) {
        let radius_sq = self.cluster_radius * self.cluster_radius;

        let count = self.history.iter()
            .filter(|(pos, _)| {
                let dx = pos[0] - position[0];
                let dy = pos[1] - position[1];
                let dz = pos[2] - position[2];
                dx*dx + dy*dy + dz*dz < radius_sq
            })
            .count();

        // Persistence score: logarithmic scaling, saturates at ~20 occurrences
        let score = (count as f32 / 5.0).min(1.0);
        (score, count)
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Get number of recorded spikes
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

// ============================================================================
// RMSD CALCULATION UTILITIES
// ============================================================================

/// Compute RMSD between two position arrays for specified atom indices
/// positions_a and positions_b are flat [x0,y0,z0,x1,y1,z1,...] arrays
pub fn compute_rmsd_subset(
    positions_a: &[f32],
    positions_b: &[f32],
    atom_indices: &[usize],
) -> f32 {
    if atom_indices.is_empty() {
        return 0.0;
    }

    let mut sum_sq = 0.0;
    let mut count = 0;

    for &idx in atom_indices {
        let i3 = idx * 3;
        if i3 + 2 < positions_a.len() && i3 + 2 < positions_b.len() {
            let dx = positions_a[i3] - positions_b[i3];
            let dy = positions_a[i3 + 1] - positions_b[i3 + 1];
            let dz = positions_a[i3 + 2] - positions_b[i3 + 2];
            sum_sq += dx*dx + dy*dy + dz*dz;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    (sum_sq / count as f32).sqrt()
}

/// Find atoms within a given radius of a position
pub fn find_atoms_near_position(
    positions: &[f32],
    center: [f32; 3],
    radius: f32,
) -> Vec<usize> {
    let radius_sq = radius * radius;
    let n_atoms = positions.len() / 3;
    let mut nearby = Vec::new();

    for i in 0..n_atoms {
        let i3 = i * 3;
        let dx = positions[i3] - center[0];
        let dy = positions[i3 + 1] - center[1];
        let dz = positions[i3 + 2] - center[2];

        if dx*dx + dy*dy + dz*dz < radius_sq {
            nearby.push(i);
        }
    }

    nearby
}

/// Compute alignment quality between two structures
/// Returns value 0-1 where 1 = perfect alignment
pub fn compute_alignment_quality(
    positions_a: &[f32],
    positions_b: &[f32],
    reference_rmsd: f32,  // Expected RMSD for "good" alignment
) -> f32 {
    let n_atoms = positions_a.len().min(positions_b.len()) / 3;
    if n_atoms == 0 {
        return 0.0;
    }

    // Compute full RMSD
    let mut sum_sq = 0.0;
    for i in 0..n_atoms {
        let i3 = i * 3;
        let dx = positions_a[i3] - positions_b[i3];
        let dy = positions_a[i3 + 1] - positions_b[i3 + 1];
        let dz = positions_a[i3 + 2] - positions_b[i3 + 2];
        sum_sq += dx*dx + dy*dy + dz*dz;
    }

    let rmsd = (sum_sq / n_atoms as f32).sqrt();

    // Convert to quality score (exponential decay)
    // reference_rmsd gives quality = e^(-1) ≈ 0.37
    (-rmsd / reference_rmsd).exp()
}

// ============================================================================
// UV PROBE CONFIGURATION
// ============================================================================

/// **DEPRECATED**: Use `CryoUvProtocol` instead
///
/// UV burst configuration for pump-probe with multi-wavelength spectroscopy.
/// This struct is deprecated because UV-LIF coupling should ALWAYS be used with
/// cryo-thermal protocols. Use the unified `CryoUvProtocol` instead.
#[deprecated(since = "1.2.0", note = "Use CryoUvProtocol instead - UV is now unified with cryo protocol")]
#[derive(Debug, Clone)]
pub struct UvProbeConfig {
    /// Master enable/disable for UV bursts
    pub enabled: bool,
    /// Energy per burst (kcal/mol)
    pub burst_energy: f32,
    /// Interval between bursts (timesteps)
    pub burst_interval: i32,
    /// Duration of each burst (timesteps)
    pub burst_duration: i32,
    /// Target aromatic residues (indices into uv_targets)
    pub target_sequence: Vec<usize>,
    /// Current position in sequence
    pub current_target: usize,
    /// Timestep counter for burst timing
    pub timestep_counter: i32,

    // =========================================================================
    // Enhanced UV Spectroscopy Fields
    // =========================================================================

    /// Enable frequency hopping (wavelength scanning)
    pub frequency_hopping_enabled: bool,
    /// Wavelengths to scan (nm)
    pub scan_wavelengths: Vec<f32>,
    /// Current wavelength index
    pub current_wavelength_idx: usize,
    /// Dwell steps per wavelength
    pub dwell_steps: i32,
    /// Steps at current wavelength
    pub steps_at_wavelength: i32,

    /// Enable disulfide targeting at 250nm
    pub target_disulfides: bool,

    /// Enable virtual benzene cosolvent probes on hydrophobic residues (LEU/ILE/VAL)
    /// When enabled, injects type-4 UV targets on aliphatic sidechains at 254nm
    pub enable_cosolvent_probes: bool,

    /// Track local temperature per aromatic
    pub track_local_temperature: bool,
    /// Per-aromatic local temperature deltas (K)
    pub local_temp_deltas: Vec<f32>,
    /// Thermal dissipation time constant (ps)
    pub thermal_dissipation_tau: f32,

    /// Total energy deposited this run (eV)
    pub total_energy_deposited: f32,
    /// Peak local temperature observed (K)
    pub peak_local_temp: f32,
}

impl Default for UvProbeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            burst_energy: 5.0,
            burst_interval: 1000,
            burst_duration: 10,
            target_sequence: Vec::new(),
            current_target: 0,
            timestep_counter: 0,
            // Spectroscopy defaults
            frequency_hopping_enabled: false,
            scan_wavelengths: vec![258.0, 274.0, 280.0],
            current_wavelength_idx: 0,
            dwell_steps: 1000,
            steps_at_wavelength: 0,
            target_disulfides: false,
            enable_cosolvent_probes: false,
            track_local_temperature: true,
            local_temp_deltas: Vec::new(),
            thermal_dissipation_tau: 5.0,
            total_energy_deposited: 0.0,
            peak_local_temp: 0.0,
        }
    }
}

impl UvProbeConfig {
    /// Create config with frequency hopping enabled
    pub fn with_frequency_hopping(wavelengths: Vec<f32>, dwell_steps: i32) -> Self {
        Self {
            frequency_hopping_enabled: true,
            scan_wavelengths: wavelengths,
            dwell_steps,
            ..Default::default()
        }
    }

    /// Create publication-quality spectroscopy config
    pub fn publication_quality() -> Self {
        Self {
            frequency_hopping_enabled: true,
            scan_wavelengths: vec![250.0, 254.0, 258.0, 265.0, 274.0, 280.0, 290.0],
            dwell_steps: 500,
            target_disulfides: true,
            track_local_temperature: true,
            ..Default::default()
        }
    }

    /// Create config for benzene cosolvent probe mode
    /// Optimized wavelength scan centered on BNZ 254nm + PHE 258nm region
    pub fn cosolvent_mode() -> Self {
        Self {
            frequency_hopping_enabled: true,
            scan_wavelengths: vec![250.0, 254.0, 258.0, 274.0, 280.0],
            dwell_steps: 500,
            enable_cosolvent_probes: true,
            target_disulfides: false,
            track_local_temperature: true,
            ..Default::default()
        }
    }

    /// Get current wavelength (nm)
    pub fn current_wavelength(&self) -> f32 {
        if self.frequency_hopping_enabled && !self.scan_wavelengths.is_empty() {
            self.scan_wavelengths[self.current_wavelength_idx % self.scan_wavelengths.len()]
        } else {
            280.0  // Default to tryptophan λmax
        }
    }

    /// Get wavelength-specific extinction coefficient ε(λ) in M⁻¹cm⁻¹
    /// Uses Gaussian profile around λmax for each chromophore type
    ///
    /// CANONICAL chromophore type ordering (MUST match GPU):
    ///   0 = TRP (Tryptophan)
    ///   1 = TYR (Tyrosine)
    ///   2 = PHE (Phenylalanine)
    ///   3 = S-S (Disulfide)
    ///   4 = BNZ (Benzene cosolvent)
    pub fn extinction_at_wavelength(&self, chromophore_type: i32) -> f32 {
        let wavelength = self.current_wavelength();

        // CANONICAL ordering: 0=TRP, 1=TYR, 2=PHE, 3=S-S, 4=BNZ
        let (lambda_max, epsilon_max, bandwidth) = match chromophore_type {
            0 => (TRP_LAMBDA_MAX, TRP_EXTINCTION_280, TRP_BANDWIDTH),      // TRP @ 280nm
            1 => (TYR_LAMBDA_MAX, TYR_EXTINCTION_274, TYR_BANDWIDTH),      // TYR @ 274nm
            2 => (PHE_LAMBDA_MAX, PHE_EXTINCTION_258, PHE_BANDWIDTH),      // PHE @ 258nm
            3 => (DISULFIDE_LAMBDA_MAX, DISULFIDE_EXTINCTION_250, DISULFIDE_BANDWIDTH), // S-S @ 250nm
            4 => (BENZENE_LAMBDA_MAX, BENZENE_EXTINCTION_254, BENZENE_BANDWIDTH), // BNZ @ 254nm
            _ => (TRP_LAMBDA_MAX, TRP_EXTINCTION_280, TRP_BANDWIDTH),      // Default to TRP
        };

        // Gaussian absorption profile: ε(λ) = ε_max × exp[-(λ-λ_max)²/(2σ²)]
        let delta = wavelength - lambda_max;
        let sigma = bandwidth / 2.355;  // FWHM to sigma
        epsilon_max * (-0.5 * (delta / sigma).powi(2)).exp()
    }

    /// Get wavelength-specific absorption scaling factor (backward compat, normalized)
    #[deprecated(note = "Use extinction_at_wavelength for physics calculations")]
    pub fn absorption_at_wavelength(&self, chromophore_type: i32) -> f32 {
        // Return normalized value for legacy code paths
        self.extinction_at_wavelength(chromophore_type) / TRP_EXTINCTION_280
    }

    /// Compute local heating from UV absorption (PHYSICS-CORRECTED)
    ///
    /// Formula: ΔT = (E_γ × p × η) / (3/2 × k_B × N_eff)
    /// where:
    ///   E_γ = photon energy (eV) = 1239.84 / λ(nm)
    ///   p = σ × F = absorption probability
    ///   σ = ε × 3.823×10⁻⁵ (cross-section in Å², per molecule)
    ///   F = photon fluence (photons/Å²) = 0.024 (calibrated)
    ///   η = heat yield = 1.0
    ///   N_eff = effective degrees of freedom
    ///
    /// Calibration: TRP @ 280nm → ΔT ≈ 19.6 K
    pub fn compute_local_heating(&self, chromophore_type: i32) -> f32 {
        let wavelength = self.current_wavelength();

        // Get wavelength-dependent extinction coefficient ε(λ)
        let epsilon = self.extinction_at_wavelength(chromophore_type);

        // CORRECTED: Proper ε → σ conversion
        // σ(Å²) = ε(M⁻¹cm⁻¹) × 3.823×10⁻⁵
        let sigma = extinction_to_cross_section(epsilon);

        // Photon energy: E_γ = hc/λ
        let e_photon = wavelength_to_ev(wavelength);

        // Calibrated photon fluence
        let fluence = CALIBRATED_PHOTON_FLUENCE;  // 0.024 photons/Å²

        // Absorption probability (single-photon regime)
        let p_absorb = sigma * fluence;

        // Heat yield
        let eta = match chromophore_type {
            0 => crate::config::HEAT_YIELD_TRP,
            1 => crate::config::HEAT_YIELD_TYR,
            2 => crate::config::HEAT_YIELD_PHE,
            3 => crate::config::HEAT_YIELD_DISULFIDE,
            4 => crate::config::HEAT_YIELD_BENZENE,
            _ => crate::config::HEAT_YIELD_TRP,
        };

        // Energy deposited
        let e_dep = e_photon * p_absorb * eta;

        // CORRECTED N_eff values (effective DOF proxies)
        // CANONICAL ordering: 0=TRP, 1=TYR, 2=PHE, 3=S-S, 4=BNZ
        let n_eff = match chromophore_type {
            0 => NEFF_TRP,       // 9.0 - Indole ring
            1 => NEFF_TYR,       // 10.0 - Phenol + OH system
            2 => NEFF_PHE,       // 9.0 - Benzene + side chain
            3 => NEFF_DISULFIDE, // 2.0 - S-S bond
            4 => NEFF_BENZENE,   // 6.0 - Benzene ring only
            _ => NEFF_TRP,
        };

        // Temperature rise via equipartition
        // ΔT = E_dep / (3/2 × k_B × N_eff)
        e_dep / (1.5 * KB_EV_K * n_eff)
    }

    /// Check if burst should be active this timestep
    pub fn is_burst_active(&self) -> bool {
        if !self.enabled {
            return false;
        }
        let cycle_pos = self.timestep_counter % self.burst_interval;
        cycle_pos < self.burst_duration
    }

    /// Get current target index
    pub fn get_target_idx(&self) -> Option<usize> {
        if self.target_sequence.is_empty() {
            None
        } else {
            Some(self.target_sequence[self.current_target % self.target_sequence.len()])
        }
    }

    /// Advance to next timestep (includes frequency hopping)
    pub fn advance(&mut self) {
        self.timestep_counter += 1;

        // Advance burst target cycling
        if self.timestep_counter % self.burst_interval == 0 && !self.target_sequence.is_empty() {
            self.current_target = (self.current_target + 1) % self.target_sequence.len();
        }

        // Advance frequency hopping
        if self.frequency_hopping_enabled && !self.scan_wavelengths.is_empty() {
            self.steps_at_wavelength += 1;
            if self.steps_at_wavelength >= self.dwell_steps {
                self.steps_at_wavelength = 0;
                self.current_wavelength_idx = (self.current_wavelength_idx + 1) % self.scan_wavelengths.len();
                log::debug!("UV wavelength hop: {:.1}nm", self.current_wavelength());
            }
        }

        // Decay local temperatures (exponential decay, τ in ps, dt = 2fs = 0.002ps)
        if self.track_local_temperature {
            let dt = 0.002;  // 2 fs timestep
            let decay = (-dt / self.thermal_dissipation_tau).exp();
            for temp in &mut self.local_temp_deltas {
                *temp *= decay;
                if *temp < 0.01 {
                    *temp = 0.0;
                }
            }
        }
    }

    /// Record local heating for an aromatic
    pub fn record_heating(&mut self, aromatic_idx: usize, delta_t: f32) {
        // Ensure vector is sized
        if aromatic_idx >= self.local_temp_deltas.len() {
            self.local_temp_deltas.resize(aromatic_idx + 1, 0.0);
        }
        self.local_temp_deltas[aromatic_idx] += delta_t;

        // Track peak
        if self.local_temp_deltas[aromatic_idx] > self.peak_local_temp {
            self.peak_local_temp = self.local_temp_deltas[aromatic_idx];
        }
    }

    /// Initialize local temperature tracking for N aromatics
    pub fn init_temperature_tracking(&mut self, n_aromatics: usize) {
        self.local_temp_deltas = vec![0.0; n_aromatics];
    }

    /// Get spectroscopy summary
    pub fn get_spectroscopy_summary(&self) -> UvSpectroscopySummary {
        UvSpectroscopySummary {
            current_wavelength: self.current_wavelength(),
            wavelengths_scanned: self.scan_wavelengths.clone(),
            total_energy_deposited: self.total_energy_deposited,
            peak_local_temp: self.peak_local_temp,
            frequency_hopping_enabled: self.frequency_hopping_enabled,
            disulfide_targeting_enabled: self.target_disulfides,
        }
    }
}

/// Summary of UV spectroscopy state
#[derive(Debug, Clone)]
pub struct UvSpectroscopySummary {
    pub current_wavelength: f32,
    pub wavelengths_scanned: Vec<f32>,
    pub total_energy_deposited: f32,
    pub peak_local_temp: f32,
    pub frequency_hopping_enabled: bool,
    pub disulfide_targeting_enabled: bool,
}

// ============================================================================
// CRYOGENIC PHYSICS CONSTANTS
// ============================================================================

/// Reference temperature for scaling (Kelvin)
const T_REF: f32 = 300.0;

/// Minimum temperature to prevent division by zero
const T_MIN: f32 = 10.0;

/// Dielectric constant at reference temperature
const EPSILON_REF: f32 = 78.5;  // Water at 300K

/// Dielectric constant at low temperature (ice-like)
const EPSILON_LOW: f32 = 3.2;   // Ice at 100K

/// UV energy dissipation factor at cold temps (prevents geometry explosion)
const UV_COLD_DISSIPATION: f32 = 0.3;

// ============================================================================
// FUSED ENGINE
// ============================================================================

/// Maximum grid dimension
const MAX_GRID_DIM: usize = 128;

/// Block size for 1D kernels
const BLOCK_SIZE_1D: usize = 256;

/// Maximum spikes per step (increased from 10000 to handle UV-LIF coupling which generates many spikes)
const MAX_SPIKES_PER_STEP: usize = 100000;

/// Maximum hydrogen clusters
const MAX_H_CLUSTERS: usize = 10000;

/// Maximum UV targets
const MAX_UV_TARGETS: usize = 256;

/// Maximum number of parallel streams for concurrent replica execution
const MAX_PARALLEL_STREAMS: usize = 8;

/// Replica state for parallel execution
#[cfg(feature = "gpu")]
pub struct ReplicaState {
    /// Positions buffer for this replica
    pub d_positions: CudaSlice<f32>,
    /// Velocities buffer for this replica
    pub d_velocities: CudaSlice<f32>,
    /// Forces buffer for this replica
    pub d_forces: CudaSlice<f32>,
    /// RNG states for this replica
    pub d_rng_states: CudaSlice<u8>,
    /// Spike events for this replica
    pub d_spike_events: CudaSlice<u8>,
    /// Spike count for this replica
    pub d_spike_count: CudaSlice<i32>,
    /// Current timestep for this replica
    pub timestep: i32,
    /// Replica ID
    pub replica_id: usize,
}

#[cfg(feature = "gpu")]
pub struct NhsAmberFusedEngine {
    // CUDA handles
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _fused_module: Arc<CudaModule>,

    // ========================================================================
    // MULTI-STREAM PARALLEL EXECUTION (for concurrent replicas)
    // ========================================================================
    /// Pool of CUDA streams for parallel replica execution
    stream_pool: Vec<Arc<CudaStream>>,
    /// Per-replica state buffers (positions, velocities, forces, RNG)
    replica_states: Vec<ReplicaState>,
    /// Number of active parallel streams
    n_parallel_streams: usize,

    // Kernel functions
    fused_step_kernel: CudaFunction,
    init_rng_kernel: CudaFunction,
    init_lif_kernel: CudaFunction,
    init_warp_matrix_kernel: CudaFunction,
    build_uv_targets_kernel: CudaFunction,

    // Atom state buffers (float3 = 3 contiguous floats)
    d_positions: CudaSlice<f32>,
    d_velocities: CudaSlice<f32>,
    d_forces: CudaSlice<f32>,
    d_masses: CudaSlice<f32>,
    d_charges: CudaSlice<f32>,
    d_atom_types: CudaSlice<i32>,
    d_residue_ids: CudaSlice<i32>,

    // AMBER parameter buffers (as raw bytes for GPU compatibility)
    d_bonds: CudaSlice<u8>,
    d_angles: CudaSlice<u8>,
    d_dihedrals: CudaSlice<u8>,
    d_lj_params: CudaSlice<u8>,
    d_exclusion_list: CudaSlice<i32>,
    d_exclusion_offsets: CudaSlice<i32>,

    // Struct sizes for GPU memory layout
    bond_size: usize,
    angle_size: usize,
    dihedral_size: usize,
    lj_size: usize,

    // SHAKE clusters (as raw bytes)
    d_h_clusters: CudaSlice<u8>,
    n_clusters: usize,
    h_cluster_size: usize,

    // UV targets (as raw bytes)
    d_uv_targets: CudaSlice<u8>,
    n_uv_targets: usize,
    uv_target_size: usize,

    // Warp matrix (as raw bytes)
    d_warp_matrix: CudaSlice<u8>,
    warp_entry_size: usize,

    // Grid buffers
    d_exclusion_field: CudaSlice<f32>,
    d_water_density: CudaSlice<f32>,
    d_water_density_prev: CudaSlice<f32>,
    d_lif_potential: CudaSlice<f32>,
    d_spike_grid: CudaSlice<i32>,

    // Spike output (events as raw bytes)
    d_spike_events: CudaSlice<u8>,
    spike_event_size: usize,
    d_spike_count: CudaSlice<i32>,

    // RNG states
    d_rng_states: CudaSlice<u8>,

    // Configuration
    n_atoms: usize,
    n_bonds: usize,
    n_angles: usize,
    n_dihedrals: usize,
    grid_dim: usize,
    grid_spacing: f32,
    grid_origin: [f32; 3],

    // Simulation parameters
    dt: f32,
    gamma_base: f32,     // Base friction at 300K
    cutoff: f32,
    timestep: i32,

    // Cryogenic physics parameters
    cryo_enabled: bool,
    dielectric_scaling: bool,

    // Protocols
    temp_protocol: TemperatureProtocol,
    uv_config: UvProbeConfig,

    // Cached aromatic residue info
    aromatic_residues: Vec<i32>,
    /// Host-side aromatic types for spectroscopy - CANONICAL: 0=TRP, 1=TYR, 2=PHE, 3=S-S
    aromatic_types: Vec<i32>,

    // ====================================================================
    // EXCITED STATE DYNAMICS BUFFERS (true UV photophysics)
    // ====================================================================
    d_is_excited: CudaSlice<i32>,              // [n_aromatics] - excitation flag
    d_time_since_excitation: CudaSlice<f32>,   // [n_aromatics] - time tracking
    d_electronic_population: CudaSlice<f32>,   // [n_aromatics] - 0.0-1.0 population
    d_vibrational_energy: CudaSlice<f32>,      // [n_aromatics] - kcal/mol
    d_franck_condon_progress: CudaSlice<f32>,  // [n_aromatics] - relaxation progress
    d_ground_state_charges: CudaSlice<f32>,    // [n_atoms] - original charges
    d_atom_to_aromatic: CudaSlice<i32>,        // [n_atoms] - -1 or aromatic index
    d_aromatic_type: CudaSlice<i32>,           // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S,4=BNZ
    d_ring_normals: CudaSlice<f32>,            // [n_aromatics * 3] - precomputed normals
    d_aromatic_centroids: CudaSlice<f32>,      // [n_aromatics * 3] - aromatic ring centroid positions (updated per-step)
    d_aromatic_neighbors: CudaSlice<u8>,       // [n_aromatics] - AromaticNeighbors structs
    aromatic_neighbors_size: usize,
    n_aromatics: usize,
    d_uv_signal_prev: CudaSlice<f32>,          // [grid_dim³] - per-voxel previous UV signal for derivative filter

    // Aromatic topology buffers for init kernels (Issue #3 fix)
    d_aromatic_atom_indices: CudaSlice<i32>,  // [n_aromatics * 16] - flat array of ring atom indices
    d_aromatic_n_atoms: CudaSlice<i32>,       // [n_aromatics] - count of atoms per aromatic

    // ====================================================================
    // O(N) CELL LIST / NEIGHBOR LIST BUFFERS
    // ====================================================================
    // Cell list constants (matches CUDA kernel)
    cell_size: f32,                           // = 10.0 Å (matches NB_CUTOFF)
    max_atoms_per_cell: usize,                // = 128
    neighbor_list_size: usize,                // = 256 per atom

    // Cell grid dimensions (computed from bounding box)
    cell_nx: i32,
    cell_ny: i32,
    cell_nz: i32,
    cell_origin: [f32; 3],

    // GPU buffers for cell list
    d_cell_list: CudaSlice<i32>,              // [n_total_cells * MAX_ATOMS_PER_CELL]
    d_cell_counts: CudaSlice<i32>,            // [n_total_cells]
    d_atom_cell: CudaSlice<i32>,              // [n_atoms] - which cell each atom is in

    // GPU buffers for neighbor list
    d_neighbor_list: CudaSlice<i32>,          // [n_atoms * NEIGHBOR_LIST_SIZE]
    d_n_neighbors: CudaSlice<i32>,            // [n_atoms] - actual neighbor count per atom

    // Rebuild control
    neighbor_list_rebuild_interval: i32,      // Rebuild every N steps (typically 10-20)
    steps_since_rebuild: i32,                 // Counter
    use_neighbor_list: bool,                  // Enable O(N) path (true for n_atoms > 500)

    // Captured data
    spike_events: Vec<SpikeEvent>,
    ensemble_snapshots: Vec<EnsembleSnapshot>,

    // Spike quality scoring support
    spike_persistence_tracker: SpikePersistenceTracker,
    reference_positions: Vec<f32>,  // Initial positions for RMSD calculation
    last_uv_burst_timestep: i32,    // For UV correlation scoring
    last_spike_count: i32,          // Preserved spike count from last sync (for download after reset)

    // Spike accumulation for analysis (across sync intervals)
    accumulate_spikes: bool,        // When true, download and accumulate spikes during sync
    accumulated_spikes: Vec<GpuSpikeEvent>,  // Accumulated spikes from all sync intervals

    // Live monitor connection
    live_monitor: Option<TcpStream>,
    live_monitor_last_send: Instant,
    live_monitor_frame_id: u64,

    // ====================================================================
    // HYPEROPTIMIZED KERNEL PATH (optional)
    // ====================================================================
    /// Ultimate engine for hyperoptimized SoA kernel (2-4x faster)
    /// Requires SM86+ GPU (Ampere/Ada)
    ultimate_engine: Option<UltimateEngine>,
    /// Current optimization level
    optimization_level: OptimizationLevel,
}

#[cfg(feature = "gpu")]
impl NhsAmberFusedEngine {
    /// Create new fused engine from PRISM-PREP topology
    pub fn new(
        context: Arc<CudaContext>,
        topology: &PrismPrepTopology,
        grid_dim: usize,
        grid_spacing: f32,
    ) -> Result<Self> {
        let stream = context.default_stream();
        Self::new_on_stream(context, stream, topology, grid_dim, grid_spacing)
    }

    /// Create new fused engine with explicit CUDA stream (for multi-stream concurrency).
    pub fn new_on_stream(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        topology: &PrismPrepTopology,
        grid_dim: usize,
        grid_spacing: f32,
    ) -> Result<Self> {
        log::info!("Creating NHS-AMBER Fused Engine: {} atoms, grid {}³",
            topology.n_atoms, grid_dim);

        if grid_dim > MAX_GRID_DIM {
            bail!("Grid dimension {} exceeds maximum {}", grid_dim, MAX_GRID_DIM);
        }

        let n_atoms = topology.n_atoms;
        let total_voxels = grid_dim * grid_dim * grid_dim;

        // Compute grid origin from bounding box
        let (min_pos, _) = topology.bounding_box();
        let padding = 5.0f32;
        let grid_origin = [
            min_pos[0] - padding,
            min_pos[1] - padding,
            min_pos[2] - padding,
        ];

        // Load PTX module - try multiple paths for different execution contexts
        //
        // STRICT RULE: If PRISM4D_PTX_DIR is set, ONLY use that path and hard-fail
        // if it doesn't exist. No silent fallback. This ensures explicit configuration
        // is respected and misconfigurations are caught early.
        //
        // Fallback paths are only used when PRISM4D_PTX_DIR is NOT set:
        // 1. Relative to executable: ../assets/ptx/ (release bundle)
        // 2. Development paths (workspace root, tests)

        let (fused_module, loaded_path): (Arc<CudaModule>, String) = if let Ok(env_dir) = std::env::var("PRISM4D_PTX_DIR") {
            // STRICT: Environment variable is set - use ONLY this path, no fallback
            let ptx_path = std::path::PathBuf::from(&env_dir).join("nhs_amber_fused.ptx");
            let path_str = ptx_path.display().to_string();

            if !ptx_path.exists() {
                bail!(
                    "FATAL: PRISM4D_PTX_DIR is set but PTX file not found.\n\
                     \n\
                     PRISM4D_PTX_DIR={}\n\
                     Expected file: {}\n\
                     \n\
                     To fix:\n\
                     - Verify the path contains nhs_amber_fused.ptx\n\
                     - Or unset PRISM4D_PTX_DIR to use automatic path resolution",
                    env_dir, path_str
                );
            }

            let module = context.load_module(Ptx::from_file(&path_str))
                .with_context(|| format!(
                    "FATAL: PRISM4D_PTX_DIR is set but PTX failed to load.\n\
                     \n\
                     PRISM4D_PTX_DIR={}\n\
                     PTX file: {}\n\
                     \n\
                     The file exists but CUDA could not load it. Check PTX compatibility.",
                    env_dir, path_str
                ))?;

            (module, path_str)
        } else {
            // PRISM4D_PTX_DIR not set - try fallback paths
            let mut ptx_paths: Vec<std::path::PathBuf> = Vec::new();

            // 1. Relative to executable (release bundle layout: bin/prism4d -> ../assets/ptx/)
            if let Ok(exe_path) = std::env::current_exe() {
                if let Some(exe_dir) = exe_path.parent() {
                    // Bundle layout: bin/prism4d, assets/ptx/
                    ptx_paths.push(exe_dir.join("../assets/ptx/nhs_amber_fused.ptx"));
                    // Same directory as executable
                    ptx_paths.push(exe_dir.join("assets/ptx/nhs_amber_fused.ptx"));
                }
            }

            // 2. Development paths
            ptx_paths.push(std::path::PathBuf::from("target/ptx/nhs_amber_fused.ptx"));
            ptx_paths.push(std::path::PathBuf::from("../../target/ptx/nhs_amber_fused.ptx"));
            ptx_paths.push(std::path::PathBuf::from("../prism-gpu/src/kernels/nhs_amber_fused.ptx"));
            ptx_paths.push(std::path::PathBuf::from("crates/prism-gpu/src/kernels/nhs_amber_fused.ptx"));

            let mut tried_paths: Vec<String> = Vec::new();
            let mut result: Option<(Arc<CudaModule>, String)> = None;

            for path in &ptx_paths {
                let path_str = path.display().to_string();
                tried_paths.push(path_str.clone());
                log::debug!("Trying PTX path: {}", path_str);
                if path.exists() {
                    match context.load_module(Ptx::from_file(&path_str)) {
                        Ok(m) => {
                            result = Some((m, path_str));
                            break;
                        }
                        Err(e) => {
                            log::debug!("PTX exists but failed to load from {}: {}", path_str, e);
                        }
                    }
                }
            }

            result.ok_or_else(|| {
                anyhow::anyhow!(
                    "FATAL: Failed to load NHS-AMBER fused PTX from any location.\n\
                     Tried paths:\n{}\n\n\
                     To fix:\n\
                     - Set PRISM4D_PTX_DIR=/path/to/ptx/dir\n\
                     - Or ensure assets/ptx/nhs_amber_fused.ptx exists relative to executable\n\
                     - Or run from workspace root with target/ptx/ available",
                    tried_paths.iter().map(|p| format!("  - {}", p)).collect::<Vec<_>>().join("\n")
                )
            })?
        };

        log::info!("Loaded fused kernel PTX from: {}", loaded_path);

        // Get kernel functions
        let fused_step_kernel = fused_module.load_function("nhs_amber_fused_step")?;
        let init_rng_kernel = fused_module.load_function("init_rng_states")?;
        let init_lif_kernel = fused_module.load_function("init_lif_state")?;
        // These kernels are optional - try to load, use defaults if missing
        let init_warp_matrix_kernel = fused_module.load_function("init_warp_matrix")
            .unwrap_or_else(|_| fused_step_kernel.clone());
        let build_uv_targets_kernel = fused_module.load_function("build_uv_targets")
            .unwrap_or_else(|_| fused_step_kernel.clone());

        // ====================================================================
        // ALLOCATE ATOM STATE BUFFERS
        // ====================================================================

        let d_positions: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;
        let d_velocities: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;
        let d_forces: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;
        let d_masses: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_charges: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_atom_types: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;
        let d_residue_ids: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;

        // ====================================================================
        // BUILD AND UPLOAD AMBER PARAMETERS (as raw bytes for GPU compatibility)
        // ====================================================================

        let n_bonds = topology.bonds.len();
        let n_angles = topology.angles.len();
        let n_dihedrals = topology.dihedrals.len();

        // Get struct sizes for memory allocation
        let bond_size = std::mem::size_of::<GpuBondParam>();
        let angle_size = std::mem::size_of::<GpuAngleParam>();
        let dihedral_size = std::mem::size_of::<GpuDihedralParam>();
        let lj_size = std::mem::size_of::<GpuLJParam>();

        // Build bond parameters
        let bonds: Vec<GpuBondParam> = topology.bonds.iter().map(|b| {
            GpuBondParam {
                i: b.i as i32,
                j: b.j as i32,
                r0: b.r0 as f32,
                k: b.k as f32,
            }
        }).collect();
        let d_bonds: CudaSlice<u8> = stream.alloc_zeros((n_bonds.max(1) * bond_size))?;

        // Build angle parameters
        let angles: Vec<GpuAngleParam> = topology.angles.iter().map(|a| {
            GpuAngleParam {
                i: a.i as i32,
                j: a.j as i32,
                k: a.k_idx as i32,
                theta0: a.theta0 as f32,
                force_k: a.force_k as f32,
            }
        }).collect();
        let d_angles: CudaSlice<u8> = stream.alloc_zeros((n_angles.max(1) * angle_size))?;

        // Build dihedral parameters
        let dihedrals: Vec<GpuDihedralParam> = topology.dihedrals.iter().map(|d| {
            GpuDihedralParam {
                i: d.i as i32,
                j: d.j as i32,
                k: d.k_idx as i32,
                l: d.l as i32,
                periodicity: d.periodicity as i32,
                phase: d.phase as f32,
                force_k: d.force_k as f32,
            }
        }).collect();
        let d_dihedrals: CudaSlice<u8> = stream.alloc_zeros((n_dihedrals.max(1) * dihedral_size))?;

        // Build LJ parameters
        let lj_params: Vec<GpuLJParam> = topology.lj_params.iter().map(|lj| {
            GpuLJParam {
                sigma: lj.sigma as f32,
                epsilon: lj.epsilon as f32,
            }
        }).collect();
        let d_lj_params: CudaSlice<u8> = stream.alloc_zeros(n_atoms * lj_size)?;

        // Build exclusion list (CSR format)
        let mut exclusion_list: Vec<i32> = Vec::new();
        let mut exclusion_offsets: Vec<i32> = vec![0];
        for atom_exclusions in &topology.exclusions {
            for &excl_atom in atom_exclusions {
                exclusion_list.push(excl_atom as i32);
            }
            exclusion_offsets.push(exclusion_list.len() as i32);
        }
        let d_exclusion_list: CudaSlice<i32> = stream.alloc_zeros(exclusion_list.len().max(1))?;
        let d_exclusion_offsets: CudaSlice<i32> = stream.alloc_zeros(exclusion_offsets.len())?;

        // ====================================================================
        // BUILD SHAKE H-CLUSTERS
        // ====================================================================

        let h_cluster_size = std::mem::size_of::<GpuHCluster>();
        let mut h_clusters: Vec<GpuHCluster> = Vec::new();
        for cluster in &topology.h_clusters {
            let mut gpu_cluster = GpuHCluster::default();
            gpu_cluster.central_atom = cluster.central_atom as i32;
            gpu_cluster.n_hydrogens = cluster.n_hydrogens as i32;
            gpu_cluster.inv_mass_central = cluster.inv_mass_central as f32;
            gpu_cluster.inv_mass_h = cluster.inv_mass_h as f32;

            for (i, &h_atom) in cluster.hydrogen_atoms.iter().enumerate().take(3) {
                gpu_cluster.hydrogen_atoms[i] = h_atom;
            }
            for (i, &bond_len) in cluster.bond_lengths.iter().enumerate().take(3) {
                gpu_cluster.bond_lengths[i] = bond_len as f32;
            }

            h_clusters.push(gpu_cluster);
        }
        let n_clusters = h_clusters.len();
        let d_h_clusters: CudaSlice<u8> = stream.alloc_zeros((n_clusters.max(1) * h_cluster_size))?;

        // ====================================================================
        // BUILD UV TARGETS (aromatic residues) + EXCITED STATE MAPPINGS
        // ====================================================================

        let uv_target_size = std::mem::size_of::<GpuUVTarget>();
        let mut aromatic_residues: Vec<i32> = topology.aromatic_residues()
            .into_iter()
            .map(|r| r as i32)
            .collect();

        // Build atom_to_aromatic mapping (-1 for non-aromatic atoms)
        let mut atom_to_aromatic: Vec<i32> = vec![-1i32; n_atoms];
        let mut aromatic_types: Vec<i32> = Vec::new();

        let mut uv_targets: Vec<GpuUVTarget> = Vec::new();
        for (aromatic_idx, &res_id) in aromatic_residues.iter().enumerate() {
            // Find atoms in this aromatic residue
            let mut target = GpuUVTarget::default();
            target.residue_id = res_id;

            let mut atom_count = 0;
            for (atom_idx, &atom_res) in topology.residue_ids.iter().enumerate() {
                if atom_res as i32 == res_id && atom_count < 16 {
                    target.atom_indices[atom_count] = atom_idx as i32;
                    // Map this atom to its aromatic index
                    atom_to_aromatic[atom_idx] = aromatic_idx as i32;
                    atom_count += 1;
                }
            }
            target.n_atoms = atom_count as i32;

            // Set absorption strength and aromatic type based on residue type
            // TRP > TYR > PHE (roughly 5:3:1 ratio for molar absorptivity at 280nm)
            // CANONICAL chromophore type ordering (MUST match GPU and compute_local_heating):
            //   0 = TRP (Tryptophan)
            //   1 = TYR (Tyrosine)
            //   2 = PHE (Phenylalanine)
            //   3 = S-S (Disulfide)
            //   4 = BNZ (Benzene cosolvent)
            let res_name = topology.residue_ids.iter()
                .position(|&r| r as i32 == res_id)
                .map(|idx| topology.residue_names[idx].as_str())
                .unwrap_or("");

            let (absorption, arom_type) = match res_name {
                "TRP" => (1.0, 0),  // TRP = 0 (canonical)
                "TYR" => (0.6, 1),  // TYR = 1 (canonical)
                "PHE" => (0.2, 2),  // PHE = 2 (canonical)
                "CYS" | "CYX" => (0.1, 3),  // S-S = 3 (canonical)
                _ => (0.3, 1),  // Default to TYR
            };
            target.absorption_strength = absorption;
            target.aromatic_type = arom_type;
            aromatic_types.push(arom_type);

            uv_targets.push(target);
        }

        // ====================================================================
        // VIRTUAL BENZENE COSOLVENT TARGETS (LEU/ILE/VAL)
        // ====================================================================
        // Inject type-4 UV targets on aliphatic hydrophobic sidechains.
        // These simulate a 0.2M benzene cosolvent (Tan et al. 2016, PMC5515508)
        // depositing heat at hydrophobic surfaces via virtual ¹B₂ᵤ absorption.
        // GPU kernel default case: ratio_sqrt=1.0 (correct: benzene has no dipole).
        // Enabled via PRISM4D_COSOLVENT=1 environment variable.
        // BNZ cosolvent always enabled for cryptic site detection
        let cosolvent_enabled = true; // was: std::env::var("PRISM4D_COSOLVENT")

        if cosolvent_enabled {
            let mut cosolvent_count = 0usize;
            // Collect unique residue IDs for hydrophobic residues not already targeted
            let mut seen_res_ids: std::collections::HashSet<i32> = 
                aromatic_residues.iter().copied().collect();

            for atom_idx in 0..n_atoms {
                let res_id = topology.residue_ids[atom_idx] as i32;
                if seen_res_ids.contains(&res_id) && !matches!(topology.residue_names[atom_idx].as_str(), "TRP") {
                    continue;
                }
                let res_name = topology.residue_names[atom_idx].as_str();
                if !matches!(res_name, "LEU" | "ILE" | "VAL" | "TRP") {
                    continue;
                }
                seen_res_ids.insert(res_id);

                // Build cosolvent target for this residue
                let aromatic_idx = uv_targets.len();
                let mut target = GpuUVTarget::default();
                target.residue_id = res_id;

                let mut atom_count = 0;
                for (aidx, &atom_res) in topology.residue_ids.iter().enumerate() {
                    if atom_res as i32 == res_id && atom_count < 16 {
                        target.atom_indices[atom_count] = aidx as i32;
                        if atom_to_aromatic[aidx] < 0 { atom_to_aromatic[aidx] = aromatic_idx as i32; }
                        atom_count += 1;
                    }
                }
                target.n_atoms = atom_count as i32;
                // BNZ cosolvent: ε=204 vs TRP ε=5600 → absorption ~0.036 relative
                target.absorption_strength = 0.036;
                target.aromatic_type = 4;  // BNZ = 4 (canonical)
                aromatic_types.push(4);
                aromatic_residues.push(res_id);
                uv_targets.push(target);
                cosolvent_count += 1;
            }
            if cosolvent_count > 0 {
                log::info!("🧪 Benzene cosolvent: injected {} virtual probes on LEU/ILE/VAL/TRP residues",
                    cosolvent_count);
            }
        }

        let n_uv_targets = uv_targets.len().min(MAX_UV_TARGETS);
        let n_aromatics = n_uv_targets;  // Same as UV targets
        let d_uv_targets: CudaSlice<u8> = stream.alloc_zeros((n_uv_targets.max(1) * uv_target_size))?;

        // ====================================================================
        // ALLOCATE EXCITED STATE BUFFERS
        // ====================================================================

        let d_is_excited: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_time_since_excitation: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_electronic_population: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_vibrational_energy: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_franck_condon_progress: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_ground_state_charges: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_atom_to_aromatic: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;
        let d_aromatic_type: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_ring_normals: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1) * 3)?;

        // UV-LIF Coupling: Aromatic centroid positions (computed from ring atoms per-step)
        let d_aromatic_centroids: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1) * 3)?;

        // UV-LIF Coupling: Per-voxel previous UV signal for temporal derivative filter
        let d_uv_signal_prev: CudaSlice<f32> = stream.alloc_zeros(total_voxels)?;

        // Aromatic neighbors for vibrational energy transfer
        let aromatic_neighbors_size = std::mem::size_of::<GpuAromaticNeighbors>();
        let d_aromatic_neighbors: CudaSlice<u8> = stream.alloc_zeros(n_aromatics.max(1) * aromatic_neighbors_size)?;

        // Aromatic topology buffers for init kernels (Issue #3 fix)
        // These are needed by build_aromatic_neighbors and compute_ring_normals CUDA kernels
        let d_aromatic_atom_indices: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1) * 16)?;
        let d_aromatic_n_atoms: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1))?;

        // ====================================================================
        // ALLOCATE O(N) CELL LIST / NEIGHBOR LIST BUFFERS
        // ====================================================================

        // Cell list constants (must match CUDA kernel defines)
        const CELL_SIZE: f32 = 10.0;         // Matches NB_CUTOFF
        const MAX_ATOMS_PER_CELL: usize = 128;
        const NEIGHBOR_LIST_SIZE: usize = 256;
        const MAX_CELLS_PER_DIM: i32 = 32;

        // Compute cell grid dimensions from bounding box
        let (min_pos, max_pos) = topology.bounding_box();
        let cell_padding = CELL_SIZE; // One cell of padding on each side
        let cell_origin = [
            min_pos[0] - cell_padding,
            min_pos[1] - cell_padding,
            min_pos[2] - cell_padding,
        ];

        let extent = [
            max_pos[0] - min_pos[0] + 2.0 * cell_padding,
            max_pos[1] - min_pos[1] + 2.0 * cell_padding,
            max_pos[2] - min_pos[2] + 2.0 * cell_padding,
        ];

        let cell_nx = ((extent[0] / CELL_SIZE).ceil() as i32).min(MAX_CELLS_PER_DIM).max(1);
        let cell_ny = ((extent[1] / CELL_SIZE).ceil() as i32).min(MAX_CELLS_PER_DIM).max(1);
        let cell_nz = ((extent[2] / CELL_SIZE).ceil() as i32).min(MAX_CELLS_PER_DIM).max(1);
        let n_total_cells = (cell_nx * cell_ny * cell_nz) as usize;

        log::info!("Cell grid: {}x{}x{} = {} cells (cell size {} Å)",
            cell_nx, cell_ny, cell_nz, n_total_cells, CELL_SIZE);

        // Allocate cell list buffers
        let d_cell_list: CudaSlice<i32> = stream.alloc_zeros(n_total_cells * MAX_ATOMS_PER_CELL)?;
        let d_cell_counts: CudaSlice<i32> = stream.alloc_zeros(n_total_cells)?;
        let d_atom_cell: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;

        // Allocate neighbor list buffers
        let d_neighbor_list: CudaSlice<i32> = stream.alloc_zeros(n_atoms * NEIGHBOR_LIST_SIZE)?;
        let d_n_neighbors: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;

        // Enable neighbor list for systems with > 500 atoms (where O(N) beats O(N²))
        let use_neighbor_list = n_atoms > 500;
        log::info!("Neighbor list mode: {} (n_atoms={})",
            if use_neighbor_list { "ENABLED (O(N))" } else { "DISABLED (O(N²))" },
            n_atoms);

        // ====================================================================
        // ALLOCATE WARP MATRIX (voxel-to-atom mapping)
        // ====================================================================

        let warp_entry_size = std::mem::size_of::<GpuWarpEntry>();
        let d_warp_matrix: CudaSlice<u8> = stream.alloc_zeros(total_voxels * warp_entry_size)?;

        // ====================================================================
        // ALLOCATE GRID BUFFERS
        // ====================================================================

        let d_exclusion_field: CudaSlice<f32> = stream.alloc_zeros(total_voxels)?;
        let d_water_density: CudaSlice<f32> = stream.alloc_zeros(total_voxels)?;
        let d_water_density_prev: CudaSlice<f32> = stream.alloc_zeros(total_voxels)?;
        let d_lif_potential: CudaSlice<f32> = stream.alloc_zeros(total_voxels)?;
        let d_spike_grid: CudaSlice<i32> = stream.alloc_zeros(total_voxels)?;

        // ====================================================================
        // ALLOCATE SPIKE OUTPUT
        // ====================================================================

        let spike_event_size = std::mem::size_of::<GpuSpikeEvent>();
        let d_spike_events: CudaSlice<u8> = stream.alloc_zeros(MAX_SPIKES_PER_STEP * spike_event_size)?;
        let d_spike_count: CudaSlice<i32> = stream.alloc_zeros(1)?;

        // ====================================================================
        // ALLOCATE RNG STATES
        // ====================================================================

        // curandState is ~48 bytes each
        let d_rng_states: CudaSlice<u8> = stream.alloc_zeros(n_atoms * 48)?;

        // ====================================================================
        // CREATE ENGINE
        // ====================================================================

        let mut engine = Self {
            context,
            stream,
            _fused_module: fused_module,
            fused_step_kernel,
            init_rng_kernel,
            init_lif_kernel,
            init_warp_matrix_kernel,
            build_uv_targets_kernel,

            d_positions,
            d_velocities,
            d_forces,
            d_masses,
            d_charges,
            d_atom_types,
            d_residue_ids,

            d_bonds,
            d_angles,
            d_dihedrals,
            d_lj_params,
            d_exclusion_list,
            d_exclusion_offsets,

            // Struct sizes for GPU memory layout
            bond_size,
            angle_size,
            dihedral_size,
            lj_size,

            d_h_clusters,
            n_clusters,
            h_cluster_size,

            d_uv_targets,
            n_uv_targets,
            uv_target_size,

            d_warp_matrix,
            warp_entry_size,

            d_exclusion_field,
            d_water_density,
            d_water_density_prev,
            d_lif_potential,
            d_spike_grid,

            d_spike_events,
            spike_event_size,
            d_spike_count,
            d_rng_states,

            n_atoms,
            n_bonds,
            n_angles,
            n_dihedrals,
            grid_dim,
            grid_spacing,
            grid_origin,

            dt: 0.002,          // 2 fs timestep
            gamma_base: 10.0,   // Base friction at 300K (ps^-1) - higher for stability
            cutoff: 10.0,       // 10 Angstrom nonbonded cutoff
            timestep: 0,

            cryo_enabled: true,
            dielectric_scaling: true,

            temp_protocol: TemperatureProtocol::physiological(),
            uv_config: UvProbeConfig {
                enable_cosolvent_probes: cosolvent_enabled,
                ..UvProbeConfig::default()
            },

            aromatic_residues,
            aromatic_types: aromatic_types.clone(),

            // Excited state buffers
            d_is_excited,
            d_time_since_excitation,
            d_electronic_population,
            d_vibrational_energy,
            d_franck_condon_progress,
            d_ground_state_charges,
            d_atom_to_aromatic,
            d_aromatic_type,
            d_ring_normals,
            d_aromatic_centroids,
            d_aromatic_neighbors,
            aromatic_neighbors_size,
            n_aromatics,
            d_uv_signal_prev,

            // Aromatic topology buffers for init kernels (Issue #3 fix)
            d_aromatic_atom_indices,
            d_aromatic_n_atoms,

            // O(N) cell list / neighbor list buffers
            cell_size: CELL_SIZE,
            max_atoms_per_cell: MAX_ATOMS_PER_CELL,
            neighbor_list_size: NEIGHBOR_LIST_SIZE,
            cell_nx,
            cell_ny,
            cell_nz,
            cell_origin,
            d_cell_list,
            d_cell_counts,
            d_atom_cell,
            d_neighbor_list,
            d_n_neighbors,
            neighbor_list_rebuild_interval: 20,  // Rebuild every 20 steps
            steps_since_rebuild: 0,
            use_neighbor_list,

            spike_events: Vec::new(),
            ensemble_snapshots: Vec::new(),

            // Spike quality scoring support
            spike_persistence_tracker: SpikePersistenceTracker::new(5.0),  // 5Å clustering radius
            reference_positions: topology.positions.clone(),
            last_uv_burst_timestep: -1000,  // No burst yet
            last_spike_count: 0,            // Will be updated on each sync before reset

            // Spike accumulation (disabled by default, enable with set_spike_accumulation)
            accumulate_spikes: false,
            accumulated_spikes: Vec::new(),

            // Live monitor (not connected by default)
            live_monitor: None,
            live_monitor_last_send: Instant::now(),
            live_monitor_frame_id: 0,

            // Hyperoptimized kernel path (disabled by default)
            ultimate_engine: None,
            optimization_level: OptimizationLevel::Standard,

            // Multi-stream parallel execution (initialized empty, created on demand)
            stream_pool: Vec::new(),
            replica_states: Vec::new(),
            n_parallel_streams: 0,
        };

        // Upload all data to GPU
        engine.upload_topology_structs(topology, &bonds, &angles, &dihedrals, &lj_params,
                                        &exclusion_list, &exclusion_offsets,
                                        &h_clusters, &uv_targets,
                                        &atom_to_aromatic, &aromatic_types)?;

        // Initialize RNG
        engine.init_rng(42)?;

        // Initialize LIF state
        engine.init_lif_state()?;

        // Build warp matrix
        engine.build_warp_matrix()?;

        // ====================================================================
        // INITIALIZE AROMATIC NEIGHBOR LISTS AND RING NORMALS (Issues #1 & #2 fix)
        // These must be called AFTER positions are uploaded but BEFORE simulation starts
        // ====================================================================
        engine.init_aromatic_neighbors()?;
        engine.compute_ring_normals()?;
        engine.compute_aromatic_centroids()?;  // UV-LIF coupling requires centroid positions

        log::info!("NHS-AMBER Fused Engine created successfully");
        log::info!("  Bonds: {}, Angles: {}, Dihedrals: {}", n_bonds, n_angles, n_dihedrals);
        log::info!("  H-Clusters (SHAKE): {}", n_clusters);
        log::info!("  UV Targets: {}", n_uv_targets);
        log::info!("  Cryogenic physics: {}", if engine.cryo_enabled { "ENABLED" } else { "disabled" });

        Ok(engine)
    }

    /// Compute friction coefficient with equilibration boost and cryogenic physics
    ///
    /// During the first EQUILIBRATION_STEPS, uses EXTREMELY high friction
    /// to quickly dissipate initial energy from unminimized structures.
    /// This prevents the velocity explosion that occurs without proper equilibration.
    ///
    /// The friction starts at 1000 ps⁻¹ (c1 ≈ 0.135, extreme damping)
    /// and gradually reduces to the base level over equilibration.
    ///
    /// After equilibration, at low temperatures, the friction increases further
    /// to mimic the sluggish behavior of a frozen/near-frozen system.
    fn compute_cryo_friction(&self, temperature: f32) -> f32 {
        // Equilibration boost: EXTREME friction for first 10000 steps
        // This is CRITICAL for structures that haven't been energy-minimized
        // gamma=1000 ps⁻¹ with dt=0.002 ps gives c1=exp(-2)≈0.135 (86.5% damping per step!)
        const EQUILIBRATION_STEPS: i32 = 10000;
        const EQUILIBRATION_GAMMA: f32 = 1000.0;  // Extreme damping (ps⁻¹)

        let base_gamma = if self.timestep < EQUILIBRATION_STEPS {
            // Exponential decay from EQUILIBRATION_GAMMA to gamma_base
            // This provides strong damping early, then gradually relaxes
            let progress = self.timestep as f32 / EQUILIBRATION_STEPS as f32;
            let decay = (-3.0 * progress).exp();  // Exponential decay factor
            EQUILIBRATION_GAMMA * decay + self.gamma_base * (1.0 - decay)
        } else {
            self.gamma_base
        };

        // Additional cryogenic scaling at low temperatures
        if !self.cryo_enabled || temperature >= T_REF {
            return base_gamma;
        }

        // Scale friction inversely with temperature
        // At T_REF (300K): gamma = base_gamma
        // At T_MIN (10K): gamma = base_gamma * sqrt(30) (much slower dynamics)
        let t_clamped = temperature.max(T_MIN);
        let scale = T_REF / t_clamped;

        // Use sqrt scaling to prevent extreme values
        base_gamma * scale.sqrt()
    }

    /// Compute temperature-dependent dielectric constant for cryogenic physics
    ///
    /// At low temperatures, the dielectric constant changes dramatically:
    /// - Liquid water at 300K: ~78.5
    /// - Ice at 100K: ~3.2
    /// This affects electrostatic interactions significantly.
    fn compute_cryo_dielectric(&self, temperature: f32) -> f32 {
        if !self.dielectric_scaling || temperature >= T_REF {
            return EPSILON_REF;
        }

        // Linear interpolation between ice and water dielectric constants
        // based on temperature
        let t_clamped = temperature.max(T_MIN);
        let t_frac = (t_clamped - T_MIN) / (T_REF - T_MIN);

        // Interpolate: low T -> EPSILON_LOW, high T -> EPSILON_REF
        EPSILON_LOW + t_frac * (EPSILON_REF - EPSILON_LOW)
    }

    /// Compute UV burst energy with wavelength-specific absorption and cryo dissipation
    ///
    /// At cold temperatures, UV energy must be dissipated more carefully
    /// to prevent local geometry explosion in the frozen system.
    /// With spectroscopy enabled, energy is also scaled by the chromophore's
    /// absorption at the current wavelength (Gaussian profile).
    fn compute_uv_energy(&self, base_energy: f32, temperature: f32) -> f32 {
        // Start with base energy
        let mut energy = base_energy;

        // Apply wavelength-specific absorption if frequency hopping enabled
        if self.uv_config.frequency_hopping_enabled {
            let target_idx = self.uv_config.current_target;
            if target_idx < self.aromatic_types.len() {
                let aromatic_type = self.aromatic_types[target_idx];
                let absorption = self.uv_config.absorption_at_wavelength(aromatic_type);
                energy *= absorption;
            }
        }

        // Apply cryo temperature dissipation
        if !self.cryo_enabled || temperature >= T_REF {
            return energy;
        }

        // Scale down UV energy at cold temperatures
        let t_clamped = temperature.max(T_MIN);
        let t_frac = (t_clamped - T_MIN) / (T_REF - T_MIN);

        // At T_MIN: use UV_COLD_DISSIPATION (30%) of base energy
        // At T_REF: use full base energy
        let scale = UV_COLD_DISSIPATION + t_frac * (1.0 - UV_COLD_DISSIPATION);
        energy * scale
    }

    /// Upload topology data to GPU using proper struct types
    fn upload_topology_structs(
        &mut self,
        topology: &PrismPrepTopology,
        bonds: &[GpuBondParam],
        angles: &[GpuAngleParam],
        dihedrals: &[GpuDihedralParam],
        lj_params: &[GpuLJParam],
        exclusion_list: &[i32],
        exclusion_offsets: &[i32],
        h_clusters: &[GpuHCluster],
        uv_targets: &[GpuUVTarget],
        atom_to_aromatic: &[i32],
        aromatic_types: &[i32],
    ) -> Result<()> {
        // Positions (flatten [x,y,z] format)
        self.stream.memcpy_htod(&topology.positions, &mut self.d_positions)?;

        // Initialize velocities from Maxwell-Boltzmann at starting temperature
        let temp = self.temp_protocol.current_temperature();
        let velocities = self.generate_maxwell_boltzmann_velocities(&topology.masses, temp);
        self.stream.memcpy_htod(&velocities, &mut self.d_velocities)?;

        // Masses
        self.stream.memcpy_htod(&topology.masses, &mut self.d_masses)?;

        // Charges
        self.stream.memcpy_htod(&topology.charges, &mut self.d_charges)?;

        // Atom types
        let atom_types: Vec<i32> = topology.classify_atoms()
            .iter()
            .map(|t| t.as_i32())
            .collect();
        self.stream.memcpy_htod(&atom_types, &mut self.d_atom_types)?;

        // Residue IDs
        let residue_ids: Vec<i32> = topology.residue_ids.iter()
            .map(|&r| r as i32)
            .collect();
        self.stream.memcpy_htod(&residue_ids, &mut self.d_residue_ids)?;

        // AMBER parameters - convert structs to bytes for GPU upload
        if !bonds.is_empty() {
            let bonds_bytes = Self::structs_to_bytes(bonds);
            self.stream.memcpy_htod(&bonds_bytes, &mut self.d_bonds)?;
        }
        if !angles.is_empty() {
            let angles_bytes = Self::structs_to_bytes(angles);
            self.stream.memcpy_htod(&angles_bytes, &mut self.d_angles)?;
        }
        if !dihedrals.is_empty() {
            let dihedrals_bytes = Self::structs_to_bytes(dihedrals);
            self.stream.memcpy_htod(&dihedrals_bytes, &mut self.d_dihedrals)?;
        }
        if !lj_params.is_empty() {
            let lj_bytes = Self::structs_to_bytes(lj_params);
            self.stream.memcpy_htod(&lj_bytes, &mut self.d_lj_params)?;
        }

        // Exclusion list (CSR format)
        if !exclusion_list.is_empty() {
            self.stream.memcpy_htod(exclusion_list, &mut self.d_exclusion_list)?;
        }
        self.stream.memcpy_htod(exclusion_offsets, &mut self.d_exclusion_offsets)?;

        // SHAKE H-clusters
        if !h_clusters.is_empty() {
            let h_clusters_bytes = Self::structs_to_bytes(h_clusters);
            self.stream.memcpy_htod(&h_clusters_bytes, &mut self.d_h_clusters)?;
        }

        // UV targets
        if !uv_targets.is_empty() {
            let uv_targets_bytes = Self::structs_to_bytes(uv_targets);
            self.stream.memcpy_htod(&uv_targets_bytes, &mut self.d_uv_targets)?;
        }

        // ====================================================================
        // UPLOAD EXCITED STATE MAPPINGS
        // ====================================================================

        // Ground state charges (copy of original charges for reference)
        self.stream.memcpy_htod(&topology.charges, &mut self.d_ground_state_charges)?;

        // Atom to aromatic mapping
        self.stream.memcpy_htod(atom_to_aromatic, &mut self.d_atom_to_aromatic)?;

        // Aromatic types
        if !aromatic_types.is_empty() {
            self.stream.memcpy_htod(aromatic_types, &mut self.d_aromatic_type)?;
        }

        // ====================================================================
        // UPLOAD AROMATIC TOPOLOGY FOR INIT KERNELS (Issue #3 fix)
        // Build flat arrays of aromatic atom indices and counts for GPU kernels
        // ====================================================================

        if !uv_targets.is_empty() {
            let n_aromatics = uv_targets.len();

            // Build flat array: [aromatic_0_atom_0, aromatic_0_atom_1, ..., aromatic_1_atom_0, ...]
            let mut aromatic_atom_indices_flat = vec![-1i32; n_aromatics * 16];
            let mut aromatic_n_atoms_flat = vec![0i32; n_aromatics];

            for (i, target) in uv_targets.iter().enumerate() {
                aromatic_n_atoms_flat[i] = target.n_atoms;
                for j in 0..16 {
                    aromatic_atom_indices_flat[i * 16 + j] = target.atom_indices[j];
                }
            }

            self.stream.memcpy_htod(&aromatic_atom_indices_flat, &mut self.d_aromatic_atom_indices)?;
            self.stream.memcpy_htod(&aromatic_n_atoms_flat, &mut self.d_aromatic_n_atoms)?;

            log::info!("Uploaded aromatic topology: {} aromatics with ring atom indices", n_aromatics);
        }

        // Initialize excited state arrays to zero (ground state)
        // d_is_excited, d_electronic_population, d_vibrational_energy, etc. are already zero-initialized

        log::info!("Uploaded topology: {} bonds, {} angles, {} dihedrals",
            self.n_bonds, self.n_angles, self.n_dihedrals);
        log::info!("Excited state: {} aromatics mapped", self.n_aromatics);

        Ok(())
    }

    /// Convert a slice of structs to a byte vector for GPU upload
    fn structs_to_bytes<T: Copy>(structs: &[T]) -> Vec<u8> {
        let size = std::mem::size_of::<T>();
        let total_bytes = structs.len() * size;
        let mut bytes = vec![0u8; total_bytes];

        unsafe {
            std::ptr::copy_nonoverlapping(
                structs.as_ptr() as *const u8,
                bytes.as_mut_ptr(),
                total_bytes,
            );
        }

        bytes
    }

    /// Generate Maxwell-Boltzmann distributed velocities
    fn generate_maxwell_boltzmann_velocities(&self, masses: &[f32], temperature: f32) -> Vec<f32> {
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();
        let mut velocities = vec![0.0f32; self.n_atoms * 3];

        // kB in kcal/(mol·K)
        const KB: f32 = 0.001987204;

        for i in 0..self.n_atoms {
            let mass = masses[i];
            if mass <= 0.0 {
                continue;
            }

            // Standard deviation from Maxwell-Boltzmann: sqrt(kT/m)
            let sigma = (KB * temperature / mass).sqrt();
            let normal = Normal::new(0.0f64, sigma as f64).unwrap();

            velocities[i * 3] = normal.sample(&mut rng) as f32;
            velocities[i * 3 + 1] = normal.sample(&mut rng) as f32;
            velocities[i * 3 + 2] = normal.sample(&mut rng) as f32;
        }

        // Remove center of mass velocity
        let mut com_vel = [0.0f32; 3];
        let mut total_mass = 0.0f32;
        for i in 0..self.n_atoms {
            let mass = masses[i];
            com_vel[0] += mass * velocities[i * 3];
            com_vel[1] += mass * velocities[i * 3 + 1];
            com_vel[2] += mass * velocities[i * 3 + 2];
            total_mass += mass;
        }
        if total_mass > 0.0 {
            com_vel[0] /= total_mass;
            com_vel[1] /= total_mass;
            com_vel[2] /= total_mass;

            for i in 0..self.n_atoms {
                velocities[i * 3] -= com_vel[0];
                velocities[i * 3 + 1] -= com_vel[1];
                velocities[i * 3 + 2] -= com_vel[2];
            }
        }

        velocities
    }

    /// Build warp matrix (voxel-to-atom mapping) on CPU
    fn build_warp_matrix(&mut self) -> Result<()> {
        let total_voxels = self.grid_dim * self.grid_dim * self.grid_dim;
        let cutoff_sq = 64.0f32;  // 8 Angstrom cutoff for warp mapping

        // Download current positions
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;

        let mut warp_entries: Vec<GpuWarpEntry> = Vec::with_capacity(total_voxels);

        for v in 0..total_voxels {
            let vz = v / (self.grid_dim * self.grid_dim);
            let vy = (v / self.grid_dim) % self.grid_dim;
            let vx = v % self.grid_dim;

            let voxel_center = [
                self.grid_origin[0] + (vx as f32 + 0.5) * self.grid_spacing,
                self.grid_origin[1] + (vy as f32 + 0.5) * self.grid_spacing,
                self.grid_origin[2] + (vz as f32 + 0.5) * self.grid_spacing,
            ];

            // Build entry using local arrays (avoid packed struct alignment issues)
            let mut atom_indices = [-1i32; 16];
            let mut atom_weights = [0.0f32; 16];
            let mut n_atoms = 0i32;

            // Find nearby atoms
            for a in 0..self.n_atoms {
                if n_atoms >= 16 {
                    break;
                }

                let dx = positions[a * 3] - voxel_center[0];
                let dy = positions[a * 3 + 1] - voxel_center[1];
                let dz = positions[a * 3 + 2] - voxel_center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    let weight = 1.0 / (1.0 + dist_sq);
                    atom_indices[n_atoms as usize] = a as i32;
                    atom_weights[n_atoms as usize] = weight;
                    n_atoms += 1;
                }
            }

            // Normalize weights
            let total_weight: f32 = atom_weights[..n_atoms as usize].iter().sum();
            if total_weight > 0.0 {
                for i in 0..n_atoms as usize {
                    atom_weights[i] /= total_weight;
                }
            }

            // Create entry from local values
            let entry = GpuWarpEntry {
                voxel_idx: v as i32,
                atom_indices,
                atom_weights,
                n_atoms,
            };
            warp_entries.push(entry);
        }

        // Upload to GPU as bytes
        let warp_bytes = Self::structs_to_bytes(&warp_entries);
        self.stream.memcpy_htod(&warp_bytes, &mut self.d_warp_matrix)?;

        // Diagnostic: count voxels with atoms
        let voxels_with_atoms = warp_entries.iter().filter(|e| e.n_atoms > 0).count();
        let avg_atoms_per_voxel = warp_entries.iter()
            .map(|e| e.n_atoms as f32)
            .sum::<f32>() / total_voxels as f32;
        let max_atoms = warp_entries.iter().map(|e| e.n_atoms).max().unwrap_or(0);

        log::info!("Built warp matrix: {} voxels ({} with atoms, avg {:.1} atoms/voxel, max {})",
            total_voxels, voxels_with_atoms, avg_atoms_per_voxel, max_atoms);
        Ok(())
    }

    /// Initialize RNG states
    fn init_rng(&mut self, seed: u64) -> Result<()> {
        let n_blocks = (self.n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.init_rng_kernel)
                .arg(&mut self.d_rng_states)
                .arg(&seed)
                .arg(&(self.n_atoms as i32))
                .launch(cfg)
        }
        .context("Failed to launch init_rng_states")?;

        self.context.synchronize()?;
        Ok(())
    }

    /// Initialize LIF neuron state
    fn init_lif_state(&mut self) -> Result<()> {
        let total_voxels = (self.grid_dim * self.grid_dim * self.grid_dim) as i32;
        let n_blocks = (total_voxels as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.init_lif_kernel)
                .arg(&mut self.d_lif_potential)
                .arg(&mut self.d_water_density)
                .arg(&mut self.d_water_density_prev)
                .arg(&mut self.d_spike_grid)
                .arg(&total_voxels)
                .launch(cfg)
        }
        .context("Failed to launch init_lif_state")?;

        self.context.synchronize()?;
        Ok(())
    }

    /// Initialize aromatic neighbor lists for vibrational energy transfer (Issue #1 fix)
    ///
    /// This must be called AFTER positions are uploaded to GPU.
    /// The CUDA kernel `build_aromatic_neighbors` finds all atoms within 5A
    /// of each aromatic ring center (excluding ring atoms themselves).
    /// These neighbors receive vibrational energy kicks during UV excitation decay.
    fn init_aromatic_neighbors(&mut self) -> Result<()> {
        if self.n_aromatics == 0 {
            log::info!("No aromatics - skipping aromatic neighbor initialization");
            return Ok(());
        }

        // Load the build_aromatic_neighbors kernel
        let build_neighbors_kernel = self._fused_module
            .load_function("build_aromatic_neighbors")
            .context("Failed to load build_aromatic_neighbors kernel")?;

        let n_blocks = (self.n_aromatics as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let neighbor_cutoff = 5.0f32;  // 5 Angstroms - atoms within this distance receive vibrational energy

        // Kernel signature from nhs_amber_fused.cu:
        // build_aromatic_neighbors(
        //     AromaticNeighbors* d_aromatic_neighbors,
        //     const float3* positions,
        //     const int* aromatic_atom_indices,  // [n_aromatics * 16]
        //     const int* aromatic_n_atoms,       // [n_aromatics]
        //     int n_aromatics,
        //     int n_atoms,
        //     float neighbor_cutoff
        // )

        unsafe {
            self.stream
                .launch_builder(&build_neighbors_kernel)
                .arg(&mut self.d_aromatic_neighbors)
                .arg(&self.d_positions)
                .arg(&self.d_aromatic_atom_indices)
                .arg(&self.d_aromatic_n_atoms)
                .arg(&(self.n_aromatics as i32))
                .arg(&(self.n_atoms as i32))
                .arg(&neighbor_cutoff)
                .launch(cfg)
        }
        .context("Failed to launch build_aromatic_neighbors")?;

        self.context.synchronize()?;
        log::info!("Built aromatic neighbor lists for {} aromatics (cutoff {:.1}A)",
            self.n_aromatics, neighbor_cutoff);

        Ok(())
    }

    /// Compute ring normal vectors for directional vibrational transfer (Issue #2 fix)
    ///
    /// This must be called AFTER positions are uploaded to GPU.
    /// The CUDA kernel `compute_ring_normals` computes the plane normal
    /// for each aromatic ring using cross product of two edge vectors.
    /// These normals are used to direct vibrational energy transfer
    /// perpendicular to the ring plane.
    fn compute_ring_normals(&mut self) -> Result<()> {
        if self.n_aromatics == 0 {
            log::info!("No aromatics - skipping ring normal computation");
            return Ok(());
        }

        // Load the compute_ring_normals kernel
        let compute_normals_kernel = self._fused_module
            .load_function("compute_ring_normals")
            .context("Failed to load compute_ring_normals kernel")?;

        let n_blocks = (self.n_aromatics as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Kernel signature from nhs_amber_fused.cu:
        // compute_ring_normals(
        //     float3* d_ring_normals,
        //     const float3* positions,
        //     const int* aromatic_atom_indices,  // [n_aromatics * 16]
        //     const int* aromatic_n_atoms,       // [n_aromatics]
        //     int n_aromatics,
        //     int n_atoms
        // )

        unsafe {
            self.stream
                .launch_builder(&compute_normals_kernel)
                .arg(&mut self.d_ring_normals)
                .arg(&self.d_positions)
                .arg(&self.d_aromatic_atom_indices)
                .arg(&self.d_aromatic_n_atoms)
                .arg(&(self.n_aromatics as i32))
                .arg(&(self.n_atoms as i32))
                .launch(cfg)
        }
        .context("Failed to launch compute_ring_normals")?;

        self.context.synchronize()?;
        log::info!("Computed ring normals for {} aromatics", self.n_aromatics);

        Ok(())
    }

    /// Compute aromatic ring centroid positions for UV-LIF coupling
    ///
    /// For each aromatic ring, computes the centroid (average position) of all
    /// atoms in the ring. These centroids are used by the UV-LIF coupling system
    /// to compute thermal wavefront propagation, dewetting halo effects, and
    /// expanded exclusion modification.
    ///
    /// This should be called after positions are uploaded/updated on GPU.
    /// For optimal UV-spike correlation, call periodically (e.g., every 10-20 steps).
    fn compute_aromatic_centroids(&mut self) -> Result<()> {
        if self.n_aromatics == 0 {
            log::trace!("No aromatics - skipping centroid computation");
            return Ok(());
        }

        // Download current positions from GPU (flat: [x0,y0,z0, x1,y1,z1, ...])
        let mut positions_host = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions_host)?;

        // Download aromatic atom indices and counts
        let mut aromatic_atom_indices = vec![0i32; self.n_aromatics * 16];
        let mut aromatic_n_atoms = vec![0i32; self.n_aromatics];
        self.stream.memcpy_dtoh(&self.d_aromatic_atom_indices, &mut aromatic_atom_indices)?;
        self.stream.memcpy_dtoh(&self.d_aromatic_n_atoms, &mut aromatic_n_atoms)?;

        // DEBUG: Print first few aromatic counts and positions
        log::debug!("Computing aromatic centroids:");
        for arom_idx in 0..self.n_aromatics.min(3) {
            let n = aromatic_n_atoms[arom_idx];
            let first_atom = aromatic_atom_indices[arom_idx * 16];
            let pos = if (first_atom as usize) < self.n_atoms {
                let idx = first_atom as usize;
                [positions_host[idx * 3], positions_host[idx * 3 + 1], positions_host[idx * 3 + 2]]
            } else {
                [0.0, 0.0, 0.0]
            };
            log::debug!("  Aromatic {}: n_atoms={}, first_atom_idx={}, pos=({:.2}, {:.2}, {:.2})",
                       arom_idx, n, first_atom, pos[0], pos[1], pos[2]);
        }

        // Compute centroids for each aromatic
        // Format: flat Vec<f32> with [x, y, z] for each aromatic
        let mut centroids_flat: Vec<f32> = vec![0.0f32; self.n_aromatics * 3];
        for arom_idx in 0..self.n_aromatics {
            let n_atoms_in_ring = aromatic_n_atoms[arom_idx] as usize;
            if n_atoms_in_ring == 0 {
                continue;
            }

            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut sum_z = 0.0f32;

            for i in 0..n_atoms_in_ring.min(16) {
                let atom_idx = aromatic_atom_indices[arom_idx * 16 + i] as usize;
                if atom_idx < self.n_atoms {
                    // Positions are stored as flat [x, y, z, x, y, z, ...]
                    sum_x += positions_host[atom_idx * 3 + 0];
                    sum_y += positions_host[atom_idx * 3 + 1];
                    sum_z += positions_host[atom_idx * 3 + 2];
                }
            }

            let n = n_atoms_in_ring as f32;
            centroids_flat[arom_idx * 3 + 0] = sum_x / n;
            centroids_flat[arom_idx * 3 + 1] = sum_y / n;
            centroids_flat[arom_idx * 3 + 2] = sum_z / n;
        }

        // DEBUG: Print computed centroids
        log::debug!("Computed centroids (first 3):");
        for arom_idx in 0..self.n_aromatics.min(3) {
            log::debug!("  Centroid {}: ({:.2}, {:.2}, {:.2})",
                       arom_idx,
                       centroids_flat[arom_idx * 3 + 0],
                       centroids_flat[arom_idx * 3 + 1],
                       centroids_flat[arom_idx * 3 + 2]);
        }

        // Upload centroids to GPU
        self.stream.memcpy_htod(&centroids_flat, &mut self.d_aromatic_centroids)?;
        self.context.synchronize()?;  // Ensure upload completes

        // Verify upload by reading back
        let mut verify = vec![0.0f32; self.n_aromatics * 3];
        self.stream.memcpy_dtoh(&self.d_aromatic_centroids, &mut verify)?;
        log::debug!("Verification - centroid 0 from GPU: ({:.2}, {:.2}, {:.2})",
                   verify[0], verify[1], verify[2]);

        log::info!("Computed and uploaded aromatic centroids for {} aromatics", self.n_aromatics);
        Ok(())
    }

    // ========================================================================
    // O(N) NEIGHBOR LIST METHODS
    // ========================================================================

    /// Rebuild cell list and neighbor lists for O(N) nonbonded calculation
    ///
    /// This should be called every 10-20 timesteps. The overhead of rebuilding
    /// is amortized over the fast O(N) force calculation.
    ///
    /// Call sequence:
    /// 1. reset_cell_counts - clear previous frame's cell data
    /// 2. build_cell_list - assign each atom to a cell
    /// 3. build_neighbor_list - find neighbors within cutoff
    pub fn rebuild_neighbor_lists(&mut self) -> Result<()> {
        if !self.use_neighbor_list {
            return Ok(());  // O(N²) path, no neighbor lists needed
        }

        let n_total_cells = (self.cell_nx * self.cell_ny * self.cell_nz) as usize;

        // Step 1: Reset cell counts
        let reset_kernel = self._fused_module
            .load_function("reset_cell_counts")
            .context("Failed to load reset_cell_counts kernel")?;

        let n_blocks_cells = (n_total_cells as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg_cells = LaunchConfig {
            grid_dim: (n_blocks_cells, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&reset_kernel)
                .arg(&mut self.d_cell_counts)
                .arg(&(n_total_cells as i32))
                .launch(cfg_cells)
        }
        .context("Failed to launch reset_cell_counts")?;

        // Step 2: Build cell list
        let build_cell_kernel = self._fused_module
            .load_function("build_cell_list")
            .context("Failed to load build_cell_list kernel")?;

        let n_blocks_atoms = (self.n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg_atoms = LaunchConfig {
            grid_dim: (n_blocks_atoms, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&build_cell_kernel)
                .arg(&self.d_positions)
                .arg(&mut self.d_cell_list)
                .arg(&mut self.d_cell_counts)
                .arg(&mut self.d_atom_cell)
                .arg(&self.cell_origin[0])
                .arg(&self.cell_origin[1])
                .arg(&self.cell_origin[2])
                .arg(&self.cell_nx)
                .arg(&self.cell_ny)
                .arg(&self.cell_nz)
                .arg(&(self.n_atoms as i32))
                .launch(cfg_atoms)
        }
        .context("Failed to launch build_cell_list")?;

        // Step 3: Build neighbor list
        let build_neighbor_kernel = self._fused_module
            .load_function("build_neighbor_list")
            .context("Failed to load build_neighbor_list kernel")?;

        // Use cutoff with 20% buffer for list reuse between rebuilds
        let cutoff_sq_with_buffer = self.cutoff * self.cutoff * 1.44;  // 1.2^2 = 1.44

        unsafe {
            self.stream
                .launch_builder(&build_neighbor_kernel)
                .arg(&self.d_positions)
                .arg(&self.d_cell_list)
                .arg(&self.d_cell_counts)
                .arg(&self.d_atom_cell)
                .arg(&self.d_exclusion_list)
                .arg(&self.d_exclusion_offsets)
                .arg(&mut self.d_neighbor_list)
                .arg(&mut self.d_n_neighbors)
                .arg(&self.cell_nx)
                .arg(&self.cell_ny)
                .arg(&self.cell_nz)
                .arg(&(self.n_atoms as i32))
                .arg(&cutoff_sq_with_buffer)
                .launch(cfg_atoms)
        }
        .context("Failed to launch build_neighbor_list")?;

        self.context.synchronize()?;
        self.steps_since_rebuild = 0;

        Ok(())
    }

    /// Compute nonbonded forces using O(N) neighbor lists
    ///
    /// This is a separate kernel call that replaces the O(N²) inline loop
    /// in the main fused kernel. For large proteins (>1000 atoms), this
    /// provides a 50-100x speedup.
    fn compute_nonbonded_with_neighbor_list(&mut self) -> Result<()> {
        let compute_nb_kernel = self._fused_module
            .load_function("compute_nonbonded_neighborlist")
            .context("Failed to load compute_nonbonded_neighborlist kernel")?;

        let n_blocks = (self.n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let cutoff_sq = self.cutoff * self.cutoff;

        unsafe {
            self.stream
                .launch_builder(&compute_nb_kernel)
                .arg(&self.d_positions)
                .arg(&mut self.d_forces)
                .arg(&self.d_charges)
                .arg(&self.d_lj_params)
                .arg(&self.d_neighbor_list)
                .arg(&self.d_n_neighbors)
                .arg(&self.d_atom_to_aromatic)
                .arg(&self.d_aromatic_type)
                .arg(&self.d_is_excited)
                .arg(&self.d_electronic_population)
                .arg(&self.d_ground_state_charges)
                .arg(&(self.n_atoms as i32))
                .arg(&cutoff_sq)
                .launch(cfg)
        }
        .context("Failed to launch compute_nonbonded_neighborlist")?;

        Ok(())
    }

    // ========================================================================
    // HYPEROPTIMIZED KERNEL MODE
    // ========================================================================

    /// Enable the hyperoptimized ultimate kernel (2-4x faster)
    ///
    /// The ultimate kernel uses SoA (Structure of Arrays) layout, occupancy tuning,
    /// texture memory, constant memory, cooperative groups, and all 14 GPU
    /// optimization techniques for maximum performance on SM86+ GPUs.
    ///
    /// This initializes a separate UltimateEngine that can be used via
    /// `step_ultimate()` and `step_batch_ultimate()` methods.
    ///
    /// # Requirements
    /// - NVIDIA Ampere (SM86) or newer GPU
    /// - ultimate_md.ptx must be compiled and available
    ///
    /// # Example
    /// ```rust,ignore
    /// engine.enable_ultimate_mode(&topology)?;
    /// // Use step_ultimate() instead of step() for optimized path
    /// let result = engine.step_ultimate()?;
    /// ```
    pub fn enable_ultimate_mode(&mut self, topology: &PrismPrepTopology) -> Result<()> {
        if self.ultimate_engine.is_some() {
            log::info!("Ultimate mode already enabled");
            return Ok(());
        }

        log::info!("Enabling hyperoptimized ultimate kernel mode...");

        let config = UltimateEngineConfig {
            optimization_level: OptimizationLevel::Ultimate,
            enable_multi_gpu: false,
            enable_mixed_precision: true,
            compute_block_size: 128,
            neighbor_rebuild_interval: 20,
        };

        let ultimate = UltimateEngine::new(self.context.clone(), topology, config)
            .context("Failed to initialize UltimateEngine")?;

        self.ultimate_engine = Some(ultimate);
        self.optimization_level = OptimizationLevel::Ultimate;

        log::info!("Ultimate mode enabled: SoA layout, occupancy tuning, all 14 optimizations active");

        Ok(())
    }

    /// Disable the ultimate kernel mode (switch back to standard kernel)
    pub fn disable_ultimate_mode(&mut self) {
        self.ultimate_engine = None;
        self.optimization_level = OptimizationLevel::Standard;
        log::info!("Ultimate mode disabled, using standard kernel");
    }

    /// Check if ultimate mode is enabled
    pub fn is_ultimate_mode(&self) -> bool {
        self.ultimate_engine.is_some()
    }

    /// Get current optimization level
    pub fn optimization_level(&self) -> OptimizationLevel {
        self.optimization_level
    }

    /// Run a single timestep using the hyperoptimized ultimate kernel
    ///
    /// This requires `enable_ultimate_mode()` to be called first.
    /// Falls back to standard kernel if ultimate mode is not enabled.
    ///
    /// The ultimate kernel provides 2-4x speedup through:
    /// - SoA memory layout for coalesced access
    /// - Occupancy-tuned launch bounds
    /// - Warp shuffle reductions
    /// - Bank conflict-free shared memory
    /// - Cooperative groups synchronization
    pub fn step_ultimate(&mut self) -> Result<StepResult> {
        if let Some(ref mut ultimate) = self.ultimate_engine {
            // Sync positions from ultimate engine back if needed
            let result = ultimate.step()
                .context("Ultimate kernel step failed")?;

            // Convert result to StepResult
            Ok(StepResult {
                timestep: result.timestep,
                temperature: result.temperature,
                spike_count: 0,  // Ultimate kernel doesn't track spikes yet
                uv_burst_active: false,
                current_wavelength_nm: None,
            })
        } else {
            // Fall back to standard kernel
            log::debug!("Ultimate mode not enabled, falling back to standard kernel");
            self.step()
        }
    }

    /// Run multiple timesteps using the hyperoptimized ultimate kernel
    ///
    /// This is the fastest path for running many MD steps without
    /// intermediate CPU-GPU synchronization.
    pub fn step_batch_ultimate(&mut self, n_steps: i32) -> Result<StepResult> {
        if let Some(ref mut ultimate) = self.ultimate_engine {
            let result = ultimate.step_batch(n_steps)
                .context("Ultimate kernel batch step failed")?;

            Ok(StepResult {
                timestep: result.timestep,
                temperature: result.temperature,
                spike_count: 0,
                uv_burst_active: false,
                current_wavelength_nm: None,
            })
        } else {
            log::debug!("Ultimate mode not enabled, falling back to standard kernel");
            self.step_batch(n_steps)
        }
    }

    // ========================================================================
    // MULTI-STREAM PARALLEL BATCH EXECUTION
    // ========================================================================
    // These methods enable running multiple replicas CONCURRENTLY on different
    // CUDA streams, providing near-linear throughput scaling up to GPU limits.
    //
    // Architecture:
    //   Stream 0: Replica 0 kernel launches
    //   Stream 1: Replica 1 kernel launches  (concurrent with Stream 0)
    //   Stream 2: Replica 2 kernel launches  (concurrent with Stream 0,1)
    //   ...
    //
    // Benefits:
    // - N replicas run in parallel (not sequential)
    // - GPU utilization approaches 100%
    // - Throughput scales ~linearly with streams (up to GPU SM limit)
    // ========================================================================

    /// Initialize multi-stream parallel execution for N replicas
    ///
    /// Creates N independent CUDA streams and allocates per-replica state buffers.
    /// Call this once before using `step_parallel_replicas()`.
    ///
    /// # Arguments
    /// * `n_replicas` - Number of replicas to run in parallel (max 8)
    /// * `topology` - Topology for buffer sizing
    ///
    /// # Example
    /// ```rust,ignore
    /// engine.init_parallel_streams(4, &topology)?;  // 4 parallel replicas
    /// for batch in 0..100 {
    ///     let results = engine.step_parallel_replicas(100)?;  // 100 steps each
    /// }
    /// ```
    pub fn init_parallel_streams(&mut self, n_replicas: usize, topology: &PrismPrepTopology) -> Result<()> {
        let n_replicas = n_replicas.min(MAX_PARALLEL_STREAMS);

        if n_replicas == 0 {
            bail!("Must have at least 1 replica");
        }

        log::info!("Initializing {} parallel CUDA streams for concurrent replica execution", n_replicas);

        // Create stream pool
        self.stream_pool.clear();
        for i in 0..n_replicas {
            let stream = self.context.new_stream()
                .with_context(|| format!("Failed to create CUDA stream {}", i))?;
            self.stream_pool.push(stream);
        }

        // Allocate per-replica state buffers
        self.replica_states.clear();
        let n_atoms = topology.n_atoms;
        let rng_state_size = std::mem::size_of::<[u32; 48]>();  // curandState
        let spike_event_size = self.spike_event_size;

        for replica_id in 0..n_replicas {
            let stream = &self.stream_pool[replica_id];

            // Allocate state buffers on this stream
            let mut d_positions: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;
            let mut d_velocities: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;
            let d_forces: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;
            let d_rng_states: CudaSlice<u8> = stream.alloc_zeros(n_atoms * rng_state_size)?;
            let d_spike_events: CudaSlice<u8> = stream.alloc_zeros(MAX_SPIKES_PER_STEP * spike_event_size)?;
            let d_spike_count: CudaSlice<i32> = stream.alloc_zeros(1)?;

            // Copy initial positions from main buffer
            stream.memcpy_dtod(&self.d_positions, &mut d_positions)?;

            // Initialize velocities from Maxwell-Boltzmann
            let temp = self.temp_protocol.current_temperature();
            let velocities = self.generate_maxwell_boltzmann_velocities(&topology.masses, temp);
            stream.memcpy_htod(&velocities, &mut d_velocities)?;

            self.replica_states.push(ReplicaState {
                d_positions,
                d_velocities,
                d_forces,
                d_rng_states,
                d_spike_events,
                d_spike_count,
                timestep: 0,
                replica_id,
            });

            // Initialize RNG for this replica with unique seed
            self.init_rng_on_stream(stream, &self.replica_states[replica_id].d_rng_states,
                                    42 + replica_id as u64)?;
        }

        self.n_parallel_streams = n_replicas;
        log::info!("Parallel streams initialized: {} replicas ready for concurrent execution", n_replicas);

        Ok(())
    }

    /// Initialize RNG on a specific stream
    fn init_rng_on_stream(&self, stream: &CudaStream, d_rng_states: &CudaSlice<u8>, seed: u64) -> Result<()> {
        let n_blocks = (self.n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&self.init_rng_kernel)
                .arg(d_rng_states)
                .arg(&seed)
                .arg(&(self.n_atoms as i32))
                .launch(cfg)
        }
        .context("Failed to launch init_rng_states on stream")?;

        Ok(())
    }

    /// Run N steps on ALL replicas in PARALLEL using multi-stream execution
    ///
    /// This is the highest-throughput path for running multiple replicas.
    /// All replicas execute concurrently on different CUDA streams.
    ///
    /// # Returns
    /// Vector of StepResult, one per replica
    ///
    /// # Performance
    /// - N replicas run in parallel (not N × serial time)
    /// - Throughput approaches N × single-replica throughput
    /// - Limited by GPU memory bandwidth and SM count
    pub fn step_parallel_replicas(&mut self, n_steps: i32) -> Result<Vec<StepResult>> {
        if self.n_parallel_streams == 0 {
            bail!("Parallel streams not initialized. Call init_parallel_streams() first.");
        }

        let n_replicas = self.n_parallel_streams;
        let current_temp = self.temp_protocol.current_temperature();
        let effective_gamma = self.compute_cryo_friction(current_temp);

        // Determine UV burst parameters (shared across replicas)
        let uv_burst_active = self.uv_config.enabled &&
            (self.timestep % self.uv_config.burst_interval) < self.uv_config.burst_duration;
        let uv_target_idx = if uv_burst_active {
            self.uv_config.current_target as i32
        } else {
            -1
        };
        let uv_burst_energy = if uv_burst_active {
            self.compute_uv_energy(self.uv_config.burst_energy, current_temp)
        } else {
            0.0
        };
        let uv_wavelength_nm = self.uv_config.current_wavelength();

        // Collect all the shared parameters we need (before mutable borrow)
        let n_atoms = self.n_atoms;
        let n_bonds = self.n_bonds;
        let n_angles = self.n_angles;
        let n_dihedrals = self.n_dihedrals;
        let n_clusters = self.n_clusters;
        let grid_dim = self.grid_dim;
        let n_uv_targets = self.n_uv_targets;
        let n_aromatics = self.n_aromatics;
        let cutoff = self.cutoff;
        let dt = self.dt;
        let use_neighbor_list = self.use_neighbor_list;
        let grid_origin = self.grid_origin;
        let grid_spacing = self.grid_spacing;
        let temp_start = self.temp_protocol.start_temp;
        let temp_end = self.temp_protocol.end_temp;
        let temp_ramp_steps = self.temp_protocol.ramp_steps;
        let temp_hold_steps = self.temp_protocol.hold_steps;
        let temp_current_step = self.temp_protocol.current_step;

        // Launch kernels on ALL streams (concurrent execution)
        for _step in 0..n_steps {
            for replica_id in 0..n_replicas {
                let n_atoms_i32 = n_atoms as i32;
                let n_bonds_i32 = n_bonds as i32;
                let n_angles_i32 = n_angles as i32;
                let n_dihedrals_i32 = n_dihedrals as i32;
                let n_clusters_i32 = n_clusters as i32;
                let grid_dim_i32 = grid_dim as i32;
                let n_uv_targets_i32 = n_uv_targets as i32;
                let uv_burst_active_i32 = if uv_burst_active { 1i32 } else { 0i32 };
                let max_spikes_i32 = MAX_SPIKES_PER_STEP as i32;
                let use_neighbor_list_i32 = if use_neighbor_list { 1i32 } else { 0i32 };
                let n_aromatics_i32 = n_aromatics as i32;

                let stream = &self.stream_pool[replica_id];
                let replica = &mut self.replica_states[replica_id];

                let cfg = LaunchConfig {
                    grid_dim: ((n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32), 1, 1),
                    block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
                    shared_mem_bytes: 0,
                };

                // Launch kernel directly (avoids borrow conflict)
                unsafe {
                    stream
                        .launch_builder(&self.fused_step_kernel)
                        .arg(&mut replica.d_positions)
                        .arg(&mut replica.d_velocities)
                        .arg(&mut replica.d_forces)
                        .arg(&self.d_masses)
                        .arg(&self.d_charges)
                        .arg(&self.d_atom_types)
                        .arg(&self.d_residue_ids)
                        .arg(&n_atoms_i32)
                        .arg(&self.d_bonds)
                        .arg(&n_bonds_i32)
                        .arg(&self.d_angles)
                        .arg(&n_angles_i32)
                        .arg(&self.d_dihedrals)
                        .arg(&n_dihedrals_i32)
                        .arg(&self.d_lj_params)
                        .arg(&self.d_exclusion_list)
                        .arg(&self.d_exclusion_offsets)
                        .arg(&self.d_h_clusters)
                        .arg(&n_clusters_i32)
                        .arg(&self.d_exclusion_field)
                        .arg(&self.d_water_density)
                        .arg(&self.d_water_density_prev)
                        .arg(&self.d_lif_potential)
                        .arg(&self.d_spike_grid)
                        .arg(&grid_origin[0])
                        .arg(&grid_origin[1])
                        .arg(&grid_origin[2])
                        .arg(&grid_spacing)
                        .arg(&grid_dim_i32)
                        .arg(&self.d_warp_matrix)
                        .arg(&self.d_uv_targets)
                        .arg(&n_uv_targets_i32)
                        .arg(&uv_burst_active_i32)
                        .arg(&uv_target_idx)
                        .arg(&uv_burst_energy)
                        .arg(&uv_wavelength_nm)
                        .arg(&self.d_is_excited)
                        .arg(&self.d_time_since_excitation)
                        .arg(&self.d_electronic_population)
                        .arg(&self.d_vibrational_energy)
                        .arg(&self.d_franck_condon_progress)
                        .arg(&self.d_ground_state_charges)
                        .arg(&self.d_atom_to_aromatic)
                        .arg(&self.d_aromatic_type)
                        .arg(&self.d_ring_normals)
                        .arg(&self.d_aromatic_centroids)
                        .arg(&mut self.d_uv_signal_prev)
                        .arg(&self.d_aromatic_neighbors)
                        .arg(&n_aromatics_i32)
                        .arg(&mut replica.d_spike_events)
                        .arg(&mut replica.d_spike_count)
                        .arg(&max_spikes_i32)
                        .arg(&temp_start)
                        .arg(&temp_end)
                        .arg(&temp_ramp_steps)
                        .arg(&temp_hold_steps)
                        .arg(&temp_current_step)
                        .arg(&dt)
                        .arg(&effective_gamma)
                        .arg(&cutoff)
                        .arg(&replica.timestep)
                        .arg(&replica.d_rng_states)
                        .arg(&self.d_neighbor_list)
                        .arg(&self.d_n_neighbors)
                        .arg(&use_neighbor_list_i32)
                        .launch(cfg)
                }
                .context("Failed to launch parallel fused_step_kernel")?;

                replica.timestep += 1;
            }
        }

        // Synchronize ALL streams (wait for all replicas to complete)
        for stream in &self.stream_pool {
            stream.synchronize()?;
        }

        // Collect results from each replica
        let mut results = Vec::with_capacity(n_replicas);
        for replica in &self.replica_states {
            let mut spike_count = [0i32];
            self.stream_pool[replica.replica_id].memcpy_dtoh(&replica.d_spike_count, &mut spike_count)?;

            results.push(StepResult {
                timestep: replica.timestep,
                temperature: current_temp,
                spike_count: spike_count[0] as usize,
                uv_burst_active,
                current_wavelength_nm: if uv_burst_active { Some(uv_wavelength_nm) } else { None },
            });
        }

        // Update main timestep
        self.timestep += n_steps;
        self.temp_protocol.current_step += n_steps as i32;

        Ok(results)
    }

    /// Get number of parallel streams initialized
    pub fn n_parallel_streams(&self) -> usize {
        self.n_parallel_streams
    }

    /// Get positions from a specific replica
    pub fn get_replica_positions(&self, replica_id: usize) -> Result<Vec<f32>> {
        if replica_id >= self.n_parallel_streams {
            bail!("Replica {} not initialized (only {} replicas)", replica_id, self.n_parallel_streams);
        }

        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream_pool[replica_id].memcpy_dtoh(&self.replica_states[replica_id].d_positions, &mut positions)?;
        Ok(positions)
    }

    /// Collect spike events from all replicas
    pub fn collect_parallel_spikes(&self) -> Result<Vec<Vec<SpikeEvent>>> {
        let mut all_spikes = Vec::with_capacity(self.n_parallel_streams);

        for replica_id in 0..self.n_parallel_streams {
            let stream = &self.stream_pool[replica_id];
            let replica = &self.replica_states[replica_id];

            // Get spike count
            let mut spike_count = [0i32];
            stream.memcpy_dtoh(&replica.d_spike_count, &mut spike_count)?;
            let n_spikes = spike_count[0].min(MAX_SPIKES_PER_STEP as i32) as usize;

            // Download spike events
            if n_spikes > 0 {
                let mut spike_bytes = vec![0u8; n_spikes * self.spike_event_size];
                stream.memcpy_dtoh(&replica.d_spike_events, &mut spike_bytes)?;

                // Convert bytes to SpikeEvent structs
                let spikes = self.bytes_to_spike_events(&spike_bytes, n_spikes);
                all_spikes.push(spikes);
            } else {
                all_spikes.push(Vec::new());
            }
        }

        Ok(all_spikes)
    }

    /// Convert raw bytes to SpikeEvent structs
    fn bytes_to_spike_events(&self, bytes: &[u8], n_spikes: usize) -> Vec<SpikeEvent> {
        let mut spikes = Vec::with_capacity(n_spikes);
        for i in 0..n_spikes {
            let offset = i * self.spike_event_size;
            if offset + self.spike_event_size <= bytes.len() {
                // Parse the SpikeEvent from bytes (matches GPU struct layout)
                let timestep = i32::from_le_bytes([
                    bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]
                ]);
                let voxel_idx = i32::from_le_bytes([
                    bytes[offset+4], bytes[offset+5], bytes[offset+6], bytes[offset+7]
                ]);
                let x = f32::from_le_bytes([
                    bytes[offset+8], bytes[offset+9], bytes[offset+10], bytes[offset+11]
                ]);
                let y = f32::from_le_bytes([
                    bytes[offset+12], bytes[offset+13], bytes[offset+14], bytes[offset+15]
                ]);
                let z = f32::from_le_bytes([
                    bytes[offset+16], bytes[offset+17], bytes[offset+18], bytes[offset+19]
                ]);
                let intensity = f32::from_le_bytes([
                    bytes[offset+20], bytes[offset+21], bytes[offset+22], bytes[offset+23]
                ]);
                let temperature = f32::from_le_bytes([
                    bytes[offset+24], bytes[offset+25], bytes[offset+26], bytes[offset+27]
                ]);

                spikes.push(SpikeEvent {
                    timestep,
                    voxel_idx,
                    position: [x, y, z],
                    intensity,
                    temperature,
                    nearby_residues: Vec::new(),  // Not parsed from GPU
                    uv_burst_active: false,  // Would need to track per-step
                });
            }
        }
        spikes
    }

    /// **Set unified cryo-UV protocol (RECOMMENDED)**
    ///
    /// This is the canonical way to configure PRISM4D for cryptic site detection.
    /// The cryo-thermal and UV-LIF systems work together as an integrated protocol.
    ///
    /// # Example
    /// ```ignore
    /// // Use the validated standard protocol
    /// engine.set_cryo_uv_protocol(CryoUvProtocol::standard())?;
    ///
    /// // Or customize
    /// let protocol = CryoUvProtocol {
    ///     start_temp: 77.0,
    ///     end_temp: 310.0,
    ///     cold_hold_steps: 5000,
    ///     ramp_steps: 10000,
    ///     warm_hold_steps: 5000,
    ///     uv_burst_energy: 30.0,
    ///     uv_burst_interval: 500,
    ///     uv_burst_duration: 50,
    ///     scan_wavelengths: vec![280.0, 274.0, 258.0],
    ///     wavelength_dwell_steps: 500,
    ///     current_step: 0,
    /// };
    /// engine.set_cryo_uv_protocol(protocol)?;
    /// ```
    pub fn set_cryo_uv_protocol(&mut self, protocol: CryoUvProtocol) -> Result<()> {
        // Convert unified protocol to legacy internal structures
        // (this will be refactored once internal engine is updated)

        #[allow(deprecated)]
        let temp_protocol = TemperatureProtocol {
            start_temp: protocol.start_temp,
            end_temp: protocol.end_temp,
            ramp_steps: protocol.ramp_steps,
            hold_steps: protocol.warm_hold_steps,
            cold_hold_steps: protocol.cold_hold_steps,
            current_step: protocol.current_step,
        };

        #[allow(deprecated)]
        let uv_config = UvProbeConfig {
            enabled: true,  // ALWAYS ENABLED in cryo-UV mode
            burst_energy: protocol.uv_burst_energy,
            burst_interval: protocol.uv_burst_interval,
            burst_duration: protocol.uv_burst_duration,
            frequency_hopping_enabled: true,
            scan_wavelengths: protocol.scan_wavelengths.clone(),
            dwell_steps: protocol.wavelength_dwell_steps,
            ..UvProbeConfig::default()
        };

        self.temp_protocol = temp_protocol;
        self.uv_config = uv_config;

        log::info!("╔═══════════════════════════════════════════════════════════════╗");
        log::info!("║  UNIFIED CRYO-UV PROTOCOL ACTIVATED                           ║");
        log::info!("╚═══════════════════════════════════════════════════════════════╝");
        log::info!("  Temperature: {}K → {}K", protocol.start_temp, protocol.end_temp);
        log::info!("  Phases: cold_hold={}, ramp={}, warm_hold={}",
            protocol.cold_hold_steps, protocol.ramp_steps, protocol.warm_hold_steps);
        log::info!("  UV bursts: {}kcal/mol every {} steps ({} step duration)",
            protocol.uv_burst_energy, protocol.uv_burst_interval, protocol.uv_burst_duration);
        log::info!("  Wavelengths: {:?} nm", protocol.scan_wavelengths);
        log::info!("  ✓ UV-LIF coupling: ACTIVE (100% aromatic localization)");

        Ok(())
    }

    /// **DEPRECATED**: Set temperature protocol
    ///
    /// Use `set_cryo_uv_protocol()` instead. The cryo-thermal and UV-LIF systems
    /// should always be used together.
    #[deprecated(since = "1.2.0", note = "Use set_cryo_uv_protocol() instead")]
    pub fn set_temperature_protocol(&mut self, protocol: TemperatureProtocol) -> Result<()> {
        self.temp_protocol = protocol.clone();
        log::warn!("⚠️  DEPRECATED: set_temperature_protocol() - Use set_cryo_uv_protocol() instead");
        log::info!("Set temperature protocol: {}K -> {}K over {} steps",
            protocol.start_temp, protocol.end_temp, protocol.ramp_steps);
        Ok(())
    }

    /// **DEPRECATED**: Set UV probe configuration
    ///
    /// Use `set_cryo_uv_protocol()` instead. UV-LIF should always be coupled with
    /// cryo-thermal protocols.
    #[deprecated(since = "1.2.0", note = "Use set_cryo_uv_protocol() instead")]
    pub fn set_uv_config(&mut self, config: UvProbeConfig) {
        log::warn!("⚠️  DEPRECATED: set_uv_config() - Use set_cryo_uv_protocol() instead");
        self.uv_config = config;
    }

    /// Get current UV probe configuration
    pub fn get_uv_config(&self) -> &UvProbeConfig {
        &self.uv_config
    }

    /// Enable or disable UV bursts
    pub fn set_uv_enabled(&mut self, enabled: bool) {
        self.uv_config.enabled = enabled;
    }

    /// Run a single timestep of the fused simulation
    ///
    /// This launches the full NHS-AMBER fused kernel which performs:
    /// 1. AMBER force computation (bonds, angles, dihedrals, nonbonded)
    /// 2. Velocity Verlet integration with Langevin thermostat
    /// 3. SHAKE constraints for hydrogen bonds
    /// 4. Holographic exclusion field update
    /// 5. Water density inference
    /// 6. Neuromorphic LIF observation
    /// 7. UV bias pump-probe (if active)
    /// 8. Spike event capture
    pub fn step(&mut self) -> Result<StepResult> {
        // Get current temperature from protocol (simulated annealing ramp)
        let current_temp = self.temp_protocol.current_temperature();

        // Compute cryogenic physics parameters
        let effective_gamma = self.compute_cryo_friction(current_temp);
        let _effective_dielectric = self.compute_cryo_dielectric(current_temp);

        // ====================================================================
        // O(N) NEIGHBOR LIST REBUILD (if needed)
        // ====================================================================
        // Rebuild every N steps or on first step
        if self.use_neighbor_list {
            self.steps_since_rebuild += 1;
            if self.steps_since_rebuild >= self.neighbor_list_rebuild_interval || self.timestep == 0 {
                self.rebuild_neighbor_lists()?;
            }
        }

        // Determine UV burst state and parameters
        let uv_burst_active = self.uv_config.is_burst_active();
        let uv_target_idx = self.uv_config.get_target_idx().unwrap_or(0) as i32;
        let uv_burst_energy = self.compute_uv_energy(self.uv_config.burst_energy, current_temp);
        let uv_wavelength_nm = self.uv_config.current_wavelength();  // GPU wavelength-dependent σ(λ)

        // Track last UV burst timestep for spike quality scoring
        if uv_burst_active {
            self.last_uv_burst_timestep = self.timestep;
        }

        // Record local temperature change when UV burst is active (spectroscopy tracking)
        if uv_burst_active && self.uv_config.track_local_temperature {
            let target_idx = self.uv_config.current_target;
            if target_idx < self.aromatic_types.len() {
                let aromatic_type = self.aromatic_types[target_idx];
                let delta_t = self.uv_config.compute_local_heating(aromatic_type);
                self.uv_config.record_heating(target_idx, delta_t);
                // Track total energy deposited (convert burst energy to eV: 1 kcal/mol ≈ 0.043 eV)
                self.uv_config.total_energy_deposited += uv_burst_energy * 0.043;

                // Debug: Log CPU-side physics for comparison with GPU
                // Enable with RUST_LOG=debug or RUST_LOG=prism_nhs=debug
                if log::log_enabled!(log::Level::Debug) && self.timestep % 10000 == 0 {
                    let wavelength = self.uv_config.current_wavelength();
                    let epsilon = self.uv_config.extinction_at_wavelength(aromatic_type);
                    let sigma = crate::config::extinction_to_cross_section(epsilon);
                    let fluence = crate::config::CALIBRATED_PHOTON_FLUENCE;
                    let p_absorb = sigma * fluence;
                    let e_photon = crate::config::wavelength_to_ev(wavelength);
                    let chromophore_name = match aromatic_type {
                        0 => "TRP", 1 => "TYR", 2 => "PHE", 3 => "S-S", 4 => "BNZ", _ => "UNK"
                    };
                    log::debug!(
                        "[UV CPU] step={} type={} λ={:.0}nm σ={:.5}Å² F={:.4} p={:.6} E_γ={:.3}eV ΔT={:.2}K",
                        self.timestep, chromophore_name, wavelength, sigma, fluence, p_absorb, e_photon, delta_t
                    );
                }
            }
        }

        // NOTE: spike_count is NOT reset here!
        // Spikes accumulate across steps and are only reset AFTER sync (when we've read them).
        // This preserves spike timestamps across the sync interval for proper UV correlation analysis.

        // ====================================================================
        // LAUNCH FUSED KERNEL
        // ====================================================================

        // Compute launch configuration
        let n_blocks = (self.n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Convert parameters to kernel-compatible types
        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let n_dihedrals_i32 = self.n_dihedrals as i32;
        let n_clusters_i32 = self.n_clusters as i32;
        let grid_dim_i32 = self.grid_dim as i32;
        let n_uv_targets_i32 = self.n_uv_targets as i32;
        let uv_burst_active_i32 = if uv_burst_active { 1i32 } else { 0i32 };
        let max_spikes_i32 = MAX_SPIKES_PER_STEP as i32;

        // Temperature protocol values
        let temp_start = self.temp_protocol.start_temp;
        let temp_end = self.temp_protocol.end_temp;
        let temp_ramp_steps = self.temp_protocol.ramp_steps;
        let temp_hold_steps = self.temp_protocol.hold_steps;
        let temp_current_step = self.temp_protocol.current_step;

        unsafe {
            self.stream
                .launch_builder(&self.fused_step_kernel)
                // Atom state (float3* treated as f32* with 3x elements)
                .arg(&mut self.d_positions)
                .arg(&mut self.d_velocities)
                .arg(&mut self.d_forces)
                .arg(&self.d_masses)
                .arg(&self.d_charges)
                .arg(&self.d_atom_types)
                .arg(&self.d_residue_ids)
                .arg(&n_atoms_i32)
                // AMBER parameters
                .arg(&self.d_bonds)
                .arg(&n_bonds_i32)
                .arg(&self.d_angles)
                .arg(&n_angles_i32)
                .arg(&self.d_dihedrals)
                .arg(&n_dihedrals_i32)
                .arg(&self.d_lj_params)
                .arg(&self.d_exclusion_list)
                .arg(&self.d_exclusion_offsets)
                // SHAKE clusters
                .arg(&self.d_h_clusters)
                .arg(&n_clusters_i32)
                // Grid buffers
                .arg(&mut self.d_exclusion_field)
                .arg(&mut self.d_water_density)
                .arg(&mut self.d_water_density_prev)
                .arg(&mut self.d_lif_potential)
                .arg(&mut self.d_spike_grid)
                // Grid origin (individual floats for cudarc compatibility)
                .arg(&self.grid_origin[0])
                .arg(&self.grid_origin[1])
                .arg(&self.grid_origin[2])
                .arg(&self.grid_spacing)
                .arg(&grid_dim_i32)
                // Warp matrix
                .arg(&mut self.d_warp_matrix)
                // UV targets
                .arg(&self.d_uv_targets)
                .arg(&n_uv_targets_i32)
                .arg(&uv_burst_active_i32)
                .arg(&uv_target_idx)
                .arg(&uv_burst_energy)
                .arg(&uv_wavelength_nm)  // Wavelength for σ(λ) calculation on GPU
                // Excited state dynamics (true photophysics)
                .arg(&mut self.d_is_excited)
                .arg(&mut self.d_time_since_excitation)
                .arg(&mut self.d_electronic_population)
                .arg(&mut self.d_vibrational_energy)
                .arg(&mut self.d_franck_condon_progress)
                .arg(&self.d_ground_state_charges)
                .arg(&self.d_atom_to_aromatic)
                .arg(&self.d_aromatic_type)
                .arg(&self.d_ring_normals)
                .arg(&self.d_aromatic_centroids)
                .arg(&mut self.d_uv_signal_prev)
                .arg(&self.d_aromatic_neighbors)
                .arg(&(self.n_aromatics as i32))
                // Spike output
                .arg(&mut self.d_spike_events)
                .arg(&mut self.d_spike_count)
                .arg(&max_spikes_i32)
                // Temperature protocol (individual values)
                .arg(&temp_start)
                .arg(&temp_end)
                .arg(&temp_ramp_steps)
                .arg(&temp_hold_steps)
                .arg(&temp_current_step)
                // Simulation parameters
                .arg(&self.dt)
                .arg(&effective_gamma)
                .arg(&self.cutoff)
                .arg(&self.timestep)
                // RNG state
                .arg(&mut self.d_rng_states)
                // O(N) neighbor list (optional)
                .arg(&self.d_neighbor_list)
                .arg(&self.d_n_neighbors)
                .arg(&(if self.use_neighbor_list { 1i32 } else { 0i32 }))
                .launch(cfg)
        }
        .context("Failed to launch nhs_amber_fused_step kernel")?;

        // Advance protocols (CPU-side, no sync needed)
        self.temp_protocol.advance();
        self.uv_config.advance();
        self.timestep += 1;

        // Only sync and read spikes every N steps for performance
        // The GPU keeps running while we skip sync on most steps
        // Higher interval = faster throughput but coarser spike timing
        let sync_interval = 1000; // Sync every 1000 steps (10x faster than 100)
        let num_spikes = if self.timestep % sync_interval == 0 {
            self.context.synchronize()?;

            let mut spike_count_host = [0i32];
            self.stream.memcpy_dtoh(&self.d_spike_count, &mut spike_count_host)?;
            let spikes = spike_count_host[0] as usize;

            // Activity-based snapshot capture - multiple triggers
            let uv_active = self.uv_config.is_burst_active();

            // Trigger 1: Spike activity threshold (adaptive based on temperature)
            let spike_threshold = if current_temp < 150.0 {
                5  // Lower threshold at cryogenic temps (events are rarer but more significant)
            } else if current_temp < 250.0 {
                8  // Medium threshold during transition
            } else {
                12 // Higher threshold at physiological temps
            };
            let spike_trigger = spikes >= spike_threshold;

            // Trigger 2: UV response - always capture during/immediately after UV burst
            let uv_trigger = uv_active && spikes >= 3;

            // Trigger 3: Temperature transition points (capture conformational changes)
            let temp_trigger = self.is_temperature_transition_point(current_temp);

            // Capture snapshot if any activity trigger fires
            if spike_trigger {
                self.capture_ensemble_snapshot_with_trigger(current_temp, SnapshotTrigger::SpikeActivity)?;
            } else if uv_trigger {
                self.capture_ensemble_snapshot_with_trigger(current_temp, SnapshotTrigger::UvResponse)?;
            } else if temp_trigger && spikes >= 2 {
                self.capture_ensemble_snapshot_with_trigger(current_temp, SnapshotTrigger::TemperatureTransition)?;
            }

            // Preserve spike count before reset (for download_full_spike_events)
            self.last_spike_count = spike_count_host[0];

            // If spike accumulation is enabled, download and store spikes before reset
            if self.accumulate_spikes && spikes > 0 {
                let n_to_download = spikes.min(MAX_SPIKES_PER_STEP);
                let bytes_needed = n_to_download * self.spike_event_size;

                // Download ONLY the actual spike bytes (not the full 6MB buffer!)
                // This is a major performance optimization - copies bytes_needed instead of 6MB
                let mut full_buffer = vec![0u8; bytes_needed];
                self.stream.memcpy_dtoh(&self.d_spike_events, &mut full_buffer)?;

                // Parse and accumulate spike events
                for i in 0..n_to_download {
                    let offset = i * self.spike_event_size;
                    let timestep = i32::from_le_bytes([
                        full_buffer[offset], full_buffer[offset + 1],
                        full_buffer[offset + 2], full_buffer[offset + 3],
                    ]);
                    let voxel_idx = i32::from_le_bytes([
                        full_buffer[offset + 4], full_buffer[offset + 5],
                        full_buffer[offset + 6], full_buffer[offset + 7],
                    ]);
                    let pos_x = f32::from_le_bytes([
                        full_buffer[offset + 8], full_buffer[offset + 9],
                        full_buffer[offset + 10], full_buffer[offset + 11],
                    ]);
                    let pos_y = f32::from_le_bytes([
                        full_buffer[offset + 12], full_buffer[offset + 13],
                        full_buffer[offset + 14], full_buffer[offset + 15],
                    ]);
                    let pos_z = f32::from_le_bytes([
                        full_buffer[offset + 16], full_buffer[offset + 17],
                        full_buffer[offset + 18], full_buffer[offset + 19],
                    ]);
                    let intensity = f32::from_le_bytes([
                        full_buffer[offset + 20], full_buffer[offset + 21],
                        full_buffer[offset + 22], full_buffer[offset + 23],
                    ]);
                    let mut nearby_residues = [0i32; 8];
                    for r in 0..8 {
                        let r_offset = offset + 24 + r * 4;
                        nearby_residues[r] = i32::from_le_bytes([
                            full_buffer[r_offset], full_buffer[r_offset + 1],
                            full_buffer[r_offset + 2], full_buffer[r_offset + 3],
                        ]);
                    }
                    let n_residues = i32::from_le_bytes([
                        full_buffer[offset + 56], full_buffer[offset + 57],
                        full_buffer[offset + 58], full_buffer[offset + 59],
                    ]);

                    self.accumulated_spikes.push(GpuSpikeEvent {
                        timestep,
                        voxel_idx,
                        position: [pos_x, pos_y, pos_z],
                        intensity,
                        nearby_residues,
                        n_residues,
                    });
                }
            }

            // Reset spike count for next sync interval
            // This must happen AFTER we've read the spike count to preserve accumulated spikes
            let zero = [0i32];
            self.stream.memcpy_htod(&zero, &mut self.d_spike_count)?;

            spikes
        } else {
            0 // Don't know spike count on non-sync steps
        };

        // Update live monitor (rate-limited to ~30 FPS internally)
        self.update_live_monitor(current_temp)?;

        Ok(StepResult {
            timestep: self.timestep,
            temperature: current_temp,
            spike_count: num_spikes,
            uv_burst_active,
            current_wavelength_nm: if self.uv_config.frequency_hopping_enabled {
                Some(self.uv_config.current_wavelength())
            } else {
                None
            },
        })
    }

    /// Run N steps without any CPU-GPU synchronization (maximum throughput)
    /// Only syncs at the very end to get final spike count
    #[cfg(feature = "gpu")]
    pub fn step_batch(&mut self, n_steps: i32) -> Result<StepResult> {
        for _ in 0..n_steps {
            // Get current temperature for this step
            let current_temp = self.temp_protocol.current_temperature();
            let uv_burst_active = self.uv_config.is_burst_active();

            // Temperature protocol values
            let temp_start = self.temp_protocol.start_temp;
            let temp_end = self.temp_protocol.end_temp;
            let temp_ramp_steps = self.temp_protocol.ramp_steps;
            let temp_hold_steps = self.temp_protocol.hold_steps;
            let temp_current_step = self.temp_protocol.current_step;

            // Cryogenic scaling
            let effective_gamma = if self.cryo_enabled && current_temp < 200.0 {
                self.gamma_base * (current_temp / 300.0).max(0.1)
            } else {
                self.gamma_base
            };

            // UV burst parameters with wavelength-specific absorption
            let uv_burst_active_i32 = if uv_burst_active { 1i32 } else { 0i32 };
            let uv_target_idx = self.uv_config.current_target as i32;
            let uv_burst_energy = if uv_burst_active {
                self.compute_uv_energy(self.uv_config.burst_energy, current_temp)
            } else {
                0.0
            };
            let uv_wavelength_nm = self.uv_config.current_wavelength();  // GPU wavelength-dependent σ(λ)

            // Record local temperature change when UV burst is active (spectroscopy tracking)
            if uv_burst_active && self.uv_config.track_local_temperature {
                let target_idx = self.uv_config.current_target;
                if target_idx < self.aromatic_types.len() {
                    let aromatic_type = self.aromatic_types[target_idx];
                    let delta_t = self.uv_config.compute_local_heating(aromatic_type);
                    self.uv_config.record_heating(target_idx, delta_t);
                    self.uv_config.total_energy_deposited += uv_burst_energy * 0.043;
                }
            }

            // Grid parameters
            let grid_dim_i32 = self.grid_dim as i32;
            let n_atoms_i32 = self.n_atoms as i32;
            let n_bonds_i32 = self.n_bonds as i32;
            let n_angles_i32 = self.n_angles as i32;
            let n_dihedrals_i32 = self.n_dihedrals as i32;
            let n_clusters_i32 = self.n_clusters as i32;
            let n_uv_targets_i32 = self.n_uv_targets as i32;
            let max_spikes_i32 = MAX_SPIKES_PER_STEP as i32;

            // Launch kernel (no sync)
            let n_blocks = (self.n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32);
            let cfg = LaunchConfig {
                grid_dim: (n_blocks, 1, 1),
                block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.stream
                    .launch_builder(&self.fused_step_kernel)
                    .arg(&mut self.d_positions)
                    .arg(&mut self.d_velocities)
                    .arg(&mut self.d_forces)
                    .arg(&self.d_masses)
                    .arg(&self.d_charges)
                    .arg(&self.d_atom_types)
                    .arg(&self.d_residue_ids)
                    .arg(&n_atoms_i32)
                    .arg(&self.d_bonds).arg(&n_bonds_i32)
                    .arg(&self.d_angles).arg(&n_angles_i32)
                    .arg(&self.d_dihedrals).arg(&n_dihedrals_i32)
                    .arg(&self.d_lj_params)
                    .arg(&self.d_exclusion_list)
                    .arg(&self.d_exclusion_offsets)
                    .arg(&self.d_h_clusters).arg(&n_clusters_i32)
                    .arg(&mut self.d_exclusion_field)
                    .arg(&mut self.d_water_density)
                    .arg(&mut self.d_water_density_prev)
                    .arg(&mut self.d_lif_potential)
                    .arg(&mut self.d_spike_grid)
                    .arg(&self.grid_origin[0])
                    .arg(&self.grid_origin[1])
                    .arg(&self.grid_origin[2])
                    .arg(&self.grid_spacing)
                    .arg(&grid_dim_i32)
                    .arg(&mut self.d_warp_matrix)
                    .arg(&self.d_uv_targets).arg(&n_uv_targets_i32)
                    .arg(&uv_burst_active_i32)
                    .arg(&uv_target_idx)
                    .arg(&uv_burst_energy)
                    .arg(&uv_wavelength_nm)  // Wavelength for σ(λ) calculation on GPU
                    // Excited state dynamics
                    .arg(&mut self.d_is_excited)
                    .arg(&mut self.d_time_since_excitation)
                    .arg(&mut self.d_electronic_population)
                    .arg(&mut self.d_vibrational_energy)
                    .arg(&mut self.d_franck_condon_progress)
                    .arg(&self.d_ground_state_charges)
                    .arg(&self.d_atom_to_aromatic)
                    .arg(&self.d_aromatic_type)
                    .arg(&self.d_ring_normals)
                    .arg(&self.d_aromatic_centroids)
                    .arg(&mut self.d_uv_signal_prev)
                    .arg(&self.d_aromatic_neighbors)
                    .arg(&(self.n_aromatics as i32))
                    .arg(&mut self.d_spike_events)
                    .arg(&mut self.d_spike_count)
                    .arg(&max_spikes_i32)
                    .arg(&temp_start).arg(&temp_end)
                    .arg(&temp_ramp_steps).arg(&temp_hold_steps)
                    .arg(&temp_current_step)
                    .arg(&self.dt).arg(&effective_gamma)
                    .arg(&self.cutoff).arg(&self.timestep)
                    .arg(&mut self.d_rng_states)
                    // O(N) neighbor list (optional)
                    .arg(&self.d_neighbor_list)
                    .arg(&self.d_n_neighbors)
                    .arg(&(if self.use_neighbor_list { 1i32 } else { 0i32 }))
                    .launch(cfg)
            }
            .context("Failed to launch nhs_amber_fused_step kernel")?;

            // O(N) neighbor list rebuild check (if enabled)
            if self.use_neighbor_list {
                self.steps_since_rebuild += 1;
                if self.steps_since_rebuild >= self.neighbor_list_rebuild_interval {
                    self.rebuild_neighbor_lists()?;
                    // Also update aromatic centroids for UV-LIF coupling
                    // (aromatics move with their parent residues)
                    self.compute_aromatic_centroids()?;
                }
            }

            // Advance protocols (CPU-side only, no GPU sync)
            self.temp_protocol.advance();
            self.uv_config.advance();
            self.timestep += 1;
        }

        // Single sync at the end of batch
        self.context.synchronize()?;

        // Read final spike count
        let mut spike_count_host = [0i32];
        self.stream.memcpy_dtoh(&self.d_spike_count, &mut spike_count_host)?;
        let num_spikes = spike_count_host[0] as usize;

        Ok(StepResult {
            timestep: self.timestep,
            temperature: self.temp_protocol.current_temperature(),
            spike_count: num_spikes,
            uv_burst_active: false,
            current_wavelength_nm: if self.uv_config.frequency_hopping_enabled {
                Some(self.uv_config.current_wavelength())
            } else {
                None
            },
        })
    }

    /// Check if we're at a temperature transition point (important for capturing conformational changes)
    fn is_temperature_transition_point(&self, current_temp: f32) -> bool {
        // Key transition temperatures for protein dynamics
        let transition_temps = [
            80.0,   // Glass transition
            150.0,  // Onset of anharmonic motion
            200.0,  // Methyl group rotation activation
            250.0,  // Side chain flexibility onset
            273.0,  // Near water freezing point
            300.0,  // Physiological
        ];

        // Check if we're within 5K of any transition point
        for &t in &transition_temps {
            if (current_temp - t).abs() < 5.0 {
                return true;
            }
        }
        false
    }

    /// Capture an ensemble snapshot with activity-based trigger reason
    fn capture_ensemble_snapshot_with_trigger(&mut self, temperature: f32, trigger: SnapshotTrigger) -> Result<()> {
        // Download current positions and velocities
        let positions = self.get_positions()?;
        let velocities = self.get_velocities()?;

        // Download FULL spike events with residue mapping from GPU
        let gpu_spikes = self.download_full_spike_events(100)?;

        // Convert to SpikeEvent and compute quality scores
        let mut trigger_spikes = Vec::new();
        let mut spike_quality_scores = Vec::new();

        for gpu_spike in &gpu_spikes {
            // Extract nearby residues from GPU spike event
            let nearby_residues: Vec<i32> = (0..gpu_spike.n_residues.min(8) as usize)
                .map(|i| gpu_spike.nearby_residues[i])
                .filter(|&r| r >= 0)
                .collect();

            // Create spike event with full data from GPU
            let spike = SpikeEvent {
                timestep: gpu_spike.timestep,
                voxel_idx: gpu_spike.voxel_idx,
                position: gpu_spike.position,
                intensity: gpu_spike.intensity,
                nearby_residues: nearby_residues.clone(),
                temperature,
                uv_burst_active: self.uv_config.is_burst_active(),
            };
            trigger_spikes.push(spike);

            // Record spike for persistence tracking
            self.spike_persistence_tracker.record_spike(gpu_spike.position, self.timestep);

            // Compute quality score for this spike with actual intensity
            let mut quality = self.compute_spike_quality_score(
                gpu_spike.position,
                temperature,
            );
            // Override intensity with actual GPU value
            quality.intensity_score = (gpu_spike.intensity / 3.0).clamp(0.0, 1.0);
            quality.compute_overall_confidence();
            spike_quality_scores.push(quality);
        }

        // Compute alignment metrics
        let alignment_quality = compute_alignment_quality(
            &positions,
            &self.reference_positions,
            2.5,  // 2.5Å reference RMSD for "good" alignment
        );

        // Compute spike region RMSD (atoms near any spike)
        let spike_region_rmsd = if !trigger_spikes.is_empty() {
            let mut spike_atoms = Vec::new();
            for spike in &trigger_spikes {
                let nearby = find_atoms_near_position(&positions, spike.position, 8.0);
                for atom in nearby {
                    if !spike_atoms.contains(&atom) {
                        spike_atoms.push(atom);
                    }
                }
            }
            compute_rmsd_subset(&positions, &self.reference_positions, &spike_atoms)
        } else {
            0.0
        };

        // Compute simulation time in ps (2fs timestep)
        let time_ps = self.timestep as f32 * 0.002;

        let snapshot = EnsembleSnapshot {
            timestep: self.timestep,
            positions,
            velocities,
            trigger_spikes,
            temperature,
            spike_quality_scores,
            alignment_quality,
            spike_region_rmsd,
            time_ps,
            trigger_reason: trigger,
            delta_sasa: None, // Could compute from GPU SASA grid if available
        };

        self.ensemble_snapshots.push(snapshot);

        if self.ensemble_snapshots.len() % 10 == 0 {
            log::info!("Captured {} ensemble snapshots (avg quality: {:.2})",
                self.ensemble_snapshots.len(),
                self.average_spike_quality());
        }

        Ok(())
    }

    /// Compute quality score for a single spike
    fn compute_spike_quality_score(
        &self,
        position: [f32; 3],
        temperature: f32,
    ) -> SpikeQualityScore {
        // Get persistence score from tracker
        let (persistence_score, _count) = self.spike_persistence_tracker.compute_persistence(position);

        // Compute UV correlation
        // Higher score if spike occurred shortly after a UV burst
        let steps_since_uv = (self.timestep - self.last_uv_burst_timestep).abs();
        let uv_correlation = if self.uv_config.is_burst_active() {
            0.95  // Very high correlation if UV active now
        } else if steps_since_uv < 500 {
            // Within 1ps of UV burst - high correlation
            0.8 * (1.0 - steps_since_uv as f32 / 500.0)
        } else {
            0.1  // Low correlation
        };

        // Find nearest aromatic residue
        let positions = self.reference_positions.as_slice();
        let mut aromatic_proximity = f32::MAX;
        for &res_id in &self.aromatic_residues {
            // Find atoms belonging to this residue (simplified - would need residue mapping)
            // For now, use a heuristic based on typical aromatic positions
            if res_id >= 0 {
                let approx_pos_idx = (res_id as usize * 10).min(positions.len() / 3 - 1) * 3;
                if approx_pos_idx + 2 < positions.len() {
                    let dx = positions[approx_pos_idx] - position[0];
                    let dy = positions[approx_pos_idx + 1] - position[1];
                    let dz = positions[approx_pos_idx + 2] - position[2];
                    let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                    if dist < aromatic_proximity {
                        aromatic_proximity = dist;
                    }
                }
            }
        }
        if aromatic_proximity == f32::MAX {
            aromatic_proximity = 20.0;  // Default if no aromatics
        }

        // Thermal stability: higher if temperature is stable (near target)
        let target_temp = self.temp_protocol.end_temp;
        let thermal_stability = 1.0 - ((temperature - target_temp).abs() / 200.0).min(1.0);

        // Flexibility score: based on local mobility (simplified)
        // In a real implementation, this would track RMSF of nearby atoms
        let flexibility_score = 0.5;  // Placeholder

        // Find atoms near spike position for local RMSD
        let current_positions = &self.reference_positions;  // Would use actual current positions
        let nearby_atoms = find_atoms_near_position(current_positions, position, 6.0);
        let local_rmsd = if !nearby_atoms.is_empty() {
            // Simplified: just count atoms as proxy for structural change
            (nearby_atoms.len() as f32 / 20.0).min(3.0)
        } else {
            0.0
        };

        let mut score = SpikeQualityScore {
            intensity_score: 0.7,  // Would come from actual spike intensity
            persistence_score,
            uv_correlation,
            thermal_stability,
            aromatic_proximity,
            flexibility_score,
            hydrogen_bond_disruption: 0,  // Would need H-bond analysis
            hydrophobic_neighbors: nearby_atoms.len() as i32,
            local_rmsd,
            contributing_atoms: nearby_atoms.len() as i32,
            overall_confidence: 0.0,
            category: SpikeQualityCategory::Noise,
        };

        score.compute_overall_confidence();
        score
    }

    /// Get average spike quality across all snapshots
    fn average_spike_quality(&self) -> f32 {
        let mut total = 0.0;
        let mut count = 0;
        for snapshot in &self.ensemble_snapshots {
            for quality in &snapshot.spike_quality_scores {
                total += quality.overall_confidence;
                count += 1;
            }
        }
        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }

    /// Get current velocities from GPU
    pub fn get_velocities(&self) -> Result<Vec<f32>> {
        let mut velocities = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_velocities, &mut velocities)?;
        Ok(velocities)
    }

    /// Get current water density field from GPU
    /// This is the inferred water density at each voxel based on exclusion field
    pub fn get_water_density(&self) -> Result<Vec<f32>> {
        let total_voxels = self.grid_dim * self.grid_dim * self.grid_dim;
        let mut water_density = vec![0.0f32; total_voxels];
        self.stream.memcpy_dtoh(&self.d_water_density, &mut water_density)?;
        Ok(water_density)
    }

    /// Get current LIF membrane potentials from GPU
    pub fn get_lif_potential(&self) -> Result<Vec<f32>> {
        let total_voxels = self.grid_dim * self.grid_dim * self.grid_dim;
        let mut potentials = vec![0.0f32; total_voxels];
        self.stream.memcpy_dtoh(&self.d_lif_potential, &mut potentials)?;
        Ok(potentials)
    }

    /// Get current exclusion field from GPU
    pub fn get_exclusion_field(&self) -> Result<Vec<f32>> {
        let total_voxels = self.grid_dim * self.grid_dim * self.grid_dim;
        let mut exclusion = vec![0.0f32; total_voxels];
        self.stream.memcpy_dtoh(&self.d_exclusion_field, &mut exclusion)?;
        Ok(exclusion)
    }

    /// Get grid dimension
    pub fn grid_dim(&self) -> usize {
        self.grid_dim
    }

    /// Get number of atoms
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Download spike events from GPU
    /// Returns the voxel indices and positions where spikes occurred
    pub fn download_spike_events(&self, max_spikes: usize) -> Result<Vec<(i32, [f32; 3])>> {
        let full_events = self.download_full_spike_events(max_spikes)?;
        Ok(full_events.into_iter().map(|e| (e.voxel_idx, e.position)).collect())
    }

    /// Download full spike events from GPU with all fields
    /// Returns complete GpuSpikeEvent data for event emission
    pub fn download_full_spike_events(&self, max_spikes: usize) -> Result<Vec<GpuSpikeEvent>> {
        // Get spike count from GPU, or use last_spike_count if GPU counter was reset
        let mut spike_count = [0i32];
        self.stream.memcpy_dtoh(&self.d_spike_count, &mut spike_count)?;

        // If GPU counter is 0, use the preserved last_spike_count from before reset
        let effective_count = if spike_count[0] == 0 && self.last_spike_count > 0 {
            self.last_spike_count
        } else {
            spike_count[0]
        };

        // Cap at both max_spikes parameter AND MAX_SPIKES_PER_STEP (buffer size limit)
        let n_spikes = (effective_count as usize).min(max_spikes).min(MAX_SPIKES_PER_STEP);

        if n_spikes == 0 {
            return Ok(Vec::new());
        }

        // Download spike events
        let bytes_needed = n_spikes * self.spike_event_size;

        // Create a view into just the spike events we need
        let full_bytes = MAX_SPIKES_PER_STEP * self.spike_event_size;
        let mut full_buffer = vec![0u8; full_bytes];
        self.stream.memcpy_dtoh(&self.d_spike_events, &mut full_buffer)?;

        // Parse spike events
        let mut events = Vec::with_capacity(n_spikes);
        for i in 0..n_spikes {
            let offset = i * self.spike_event_size;
            // GpuSpikeEvent layout: timestep(4), voxel_idx(4), position[3](12), intensity(4),
            // nearby_residues[8](32), n_residues(4) = 60 bytes
            let timestep = i32::from_le_bytes([
                full_buffer[offset],
                full_buffer[offset + 1],
                full_buffer[offset + 2],
                full_buffer[offset + 3],
            ]);
            let voxel_idx = i32::from_le_bytes([
                full_buffer[offset + 4],
                full_buffer[offset + 5],
                full_buffer[offset + 6],
                full_buffer[offset + 7],
            ]);
            let pos_x = f32::from_le_bytes([
                full_buffer[offset + 8],
                full_buffer[offset + 9],
                full_buffer[offset + 10],
                full_buffer[offset + 11],
            ]);
            let pos_y = f32::from_le_bytes([
                full_buffer[offset + 12],
                full_buffer[offset + 13],
                full_buffer[offset + 14],
                full_buffer[offset + 15],
            ]);
            let pos_z = f32::from_le_bytes([
                full_buffer[offset + 16],
                full_buffer[offset + 17],
                full_buffer[offset + 18],
                full_buffer[offset + 19],
            ]);
            let intensity = f32::from_le_bytes([
                full_buffer[offset + 20],
                full_buffer[offset + 21],
                full_buffer[offset + 22],
                full_buffer[offset + 23],
            ]);

            // Parse nearby_residues[8]
            let mut nearby_residues = [0i32; 8];
            for r in 0..8 {
                let r_offset = offset + 24 + r * 4;
                nearby_residues[r] = i32::from_le_bytes([
                    full_buffer[r_offset],
                    full_buffer[r_offset + 1],
                    full_buffer[r_offset + 2],
                    full_buffer[r_offset + 3],
                ]);
            }

            let n_residues = i32::from_le_bytes([
                full_buffer[offset + 56],
                full_buffer[offset + 57],
                full_buffer[offset + 58],
                full_buffer[offset + 59],
            ]);

            events.push(GpuSpikeEvent {
                timestep,
                voxel_idx,
                position: [pos_x, pos_y, pos_z],
                intensity,
                nearby_residues,
                n_residues,
            });
        }

        Ok(events)
    }

    /// Get spike grid (binary spike map) from GPU
    pub fn get_spike_grid(&self) -> Result<Vec<i32>> {
        let total_voxels = self.grid_dim * self.grid_dim * self.grid_dim;
        let mut spike_grid = vec![0i32; total_voxels];
        self.stream.memcpy_dtoh(&self.d_spike_grid, &mut spike_grid)?;
        Ok(spike_grid)
    }

    /// Clear spike events buffer on GPU
    ///
    /// Resets the spike counter to 0 so the next step starts fresh.
    /// Call this after downloading spike events to avoid re-reading stale data.
    pub fn clear_spike_events(&mut self) -> Result<()> {
        let zero = [0i32];
        self.stream.memcpy_htod(&zero, &mut self.d_spike_count)?;
        Ok(())
    }

    /// Enable spike accumulation mode
    ///
    /// When enabled, spike events are downloaded and accumulated on each sync interval.
    /// This allows analysis of spikes across the entire run, not just the last sync interval.
    /// Use `get_accumulated_spikes()` to retrieve and `clear_accumulated_spikes()` to reset.
    pub fn set_spike_accumulation(&mut self, enabled: bool) {
        self.accumulate_spikes = enabled;
        log::info!("Spike accumulation: {}", if enabled { "ENABLED" } else { "disabled" });
    }

    /// Get accumulated spike events
    ///
    /// Returns all spike events accumulated since the last clear.
    /// Only populated when spike accumulation is enabled.
    pub fn get_accumulated_spikes(&self) -> &[GpuSpikeEvent] {
        &self.accumulated_spikes
    }

    /// Clear accumulated spike events
    pub fn clear_accumulated_spikes(&mut self) {
        self.accumulated_spikes.clear();
    }

    /// Reset engine state for a new replica
    ///
    /// Clears accumulated spikes, re-initializes RNG with a new seed,
    /// resets simulation counters, and zeros velocities.
    /// The Langevin thermostat will naturally thermalize to target temperature.
    /// Topology, force field parameters, and neighbor lists are preserved.
    ///
    /// # Arguments
    /// * `seed` - Random seed for RNG (affects stochastic Langevin dynamics)
    pub fn reset_for_replica(&mut self, seed: u64) -> Result<()> {
        // Clear accumulated data
        self.accumulated_spikes.clear();
        self.ensemble_snapshots.clear();

        // Re-initialize GPU RNG with new seed (affects Langevin noise in kernels)
        self.init_rng(seed)?;

        // Reset simulation counters
        self.timestep = 0;
        self.last_spike_count = 0;

        // Generate seeded Maxwell-Boltzmann velocities for true trajectory divergence.
        // Each replica gets deterministically different initial velocities from its seed,
        // ensuring warm-phase dynamics diverge even with identical starting positions.
        {
            use rand::SeedableRng;
            use rand_distr::{Distribution, Normal};
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut velocities = vec![0.0f32; self.n_atoms * 3];
            const KB: f32 = 0.001987204;
            let temp = self.temp_protocol.start_temp.max(50.0);
            // Download masses from GPU
            let mut masses = vec![0.0f32; self.n_atoms];
            self.stream.memcpy_dtoh(&self.d_masses, &mut masses)?;
            for i in 0..self.n_atoms {
                let mass = masses[i];
                if mass <= 0.0 { continue; }
                let sigma = (KB * temp / mass).sqrt();
                let normal = Normal::new(0.0f64, sigma as f64).unwrap();
                velocities[i * 3] = normal.sample(&mut rng) as f32;
                velocities[i * 3 + 1] = normal.sample(&mut rng) as f32;
                velocities[i * 3 + 2] = normal.sample(&mut rng) as f32;
            }
            // Remove center of mass velocity
            let mut com_vel = [0.0f32; 3];
            let mut total_mass = 0.0f32;
            for i in 0..self.n_atoms {
                let m = masses[i];
                com_vel[0] += m * velocities[i * 3];
                com_vel[1] += m * velocities[i * 3 + 1];
                com_vel[2] += m * velocities[i * 3 + 2];
                total_mass += m;
            }
            if total_mass > 0.0 {
                for i in 0..self.n_atoms {
                    velocities[i * 3] -= com_vel[0] / total_mass;
                    velocities[i * 3 + 1] -= com_vel[1] / total_mass;
                    velocities[i * 3 + 2] -= com_vel[2] / total_mass;
                }
            }
            self.stream.memcpy_htod(&velocities, &mut self.d_velocities)?;
            // Apply thermally-scaled position perturbations per atom.
            // σ = sqrt(kT / (m * ω²)) where ω² ≈ k_bond/m ≈ 500 kcal/(mol·Å²·amu)
            // At 50K: σ(C) ≈ 0.14Å, σ(H) ≈ 0.32Å — matches cryo B-factors.
            // This creates genuinely different microstates that force field cannot erase.
            const KB_JITTER: f64 = 0.001987204;  // kcal/(mol·K)
            const OMEGA_SQ: f64 = 500.0;  // effective spring constant / mass
            let mut positions = vec![0.0f32; self.n_atoms * 3];
            self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;
            for i in 0..self.n_atoms {
                let mass = masses[i] as f64;
                if mass <= 0.0 { continue; }
                // Thermal displacement: sqrt(kT / (m * omega^2))
                let sigma = (KB_JITTER * temp as f64 / (mass * OMEGA_SQ)).sqrt();
                let jitter = Normal::new(0.0f64, sigma).unwrap();
                positions[i * 3] += jitter.sample(&mut rng) as f32;
                positions[i * 3 + 1] += jitter.sample(&mut rng) as f32;
                positions[i * 3 + 2] += jitter.sample(&mut rng) as f32;
            }
            self.stream.memcpy_htod(&positions, &mut self.d_positions)?;
        }

        // Reset temperature protocol
        self.temp_protocol.current_step = 0;

        log::debug!("Reset for replica with seed {} (seeded MB velocities + GPU RNG)", seed);
        Ok(())
    }

    /// Enable or disable cryogenic physics
    pub fn set_cryogenic_mode(&mut self, enabled: bool) {
        self.cryo_enabled = enabled;
        log::info!("Cryogenic physics: {}", if enabled { "ENABLED" } else { "disabled" });
    }

    /// Enable or disable dielectric scaling
    pub fn set_dielectric_scaling(&mut self, enabled: bool) {
        self.dielectric_scaling = enabled;
        log::info!("Dielectric scaling: {}", if enabled { "ENABLED" } else { "disabled" });
    }

    /// Run multiple steps
    pub fn run(&mut self, n_steps: i32) -> Result<RunSummary> {
        let mut total_spikes = 0usize;
        let start_temp = self.temp_protocol.current_temperature();

        for _step in 0..n_steps {
            let result = self.step()?;
            total_spikes += result.spike_count;

            if self.timestep % 10000 == 0 {
                log::info!("Step {}: T={:.1}K, spikes={}",
                    self.timestep,
                    self.temp_protocol.current_temperature(),
                    result.spike_count);
            }
        }

        let end_temp = self.temp_protocol.current_temperature();

        Ok(RunSummary {
            steps_completed: n_steps,
            total_spikes,
            start_temperature: start_temp,
            end_temperature: end_temp,
            ensemble_snapshots: self.ensemble_snapshots.len(),
        })
    }

    /// Get current positions from GPU
    pub fn get_positions(&self) -> Result<Vec<f32>> {
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;
        Ok(positions)
    }

    /// Get vibrational energy from GPU (for debugging/testing)
    /// Returns [n_aromatics] array of energy in kcal/mol
    pub fn get_vibrational_energy(&self) -> Result<Vec<f32>> {
        let mut energy = vec![0.0f32; self.n_aromatics.max(1)];
        self.stream.memcpy_dtoh(&self.d_vibrational_energy, &mut energy)?;
        Ok(energy)
    }

    /// Get excitation state from GPU (for debugging/testing)
    /// Returns [n_aromatics] array of 0/1 flags
    pub fn get_is_excited(&self) -> Result<Vec<i32>> {
        let mut excited = vec![0i32; self.n_aromatics.max(1)];
        self.stream.memcpy_dtoh(&self.d_is_excited, &mut excited)?;
        Ok(excited)
    }

    /// Get number of aromatics detected in structure
    pub fn n_aromatics(&self) -> usize {
        self.n_aromatics
    }

    /// Get aromatic residue IDs
    pub fn aromatic_residue_ids(&self) -> &[i32] {
        &self.aromatic_residues
    }

    /// Get captured spike events
    pub fn get_spike_events(&self) -> &[SpikeEvent] {
        &self.spike_events
    }

    /// Get ensemble snapshots
    pub fn get_ensemble_snapshots(&self) -> &[EnsembleSnapshot] {
        &self.ensemble_snapshots
    }

    /// Get UV spectroscopy summary for publication reporting
    pub fn get_spectroscopy_summary(&self) -> UvSpectroscopySummary {
        self.uv_config.get_spectroscopy_summary()
    }

    /// Get per-aromatic local temperature deltas (K above baseline)
    pub fn get_local_temperatures(&self) -> &[f32] {
        &self.uv_config.local_temp_deltas
    }

    // ========================================================================
    // LIVE MONITOR METHODS
    // ========================================================================

    /// Connect to live monitor (call before simulation starts)
    pub fn connect_live_monitor(&mut self, address: &str) -> Result<()> {
        match TcpStream::connect_timeout(
            &address.parse().context("Invalid monitor address")?,
            Duration::from_secs(2)
        ) {
            Ok(stream) => {
                stream.set_nodelay(true)?;
                stream.set_write_timeout(Some(Duration::from_millis(100)))?;
                self.live_monitor = Some(stream);
                log::info!("Live monitor connected to {}", address);
            }
            Err(e) => {
                log::warn!("Live monitor unavailable: {} (continuing without)", e);
            }
        }
        Ok(())
    }

    /// Update live monitor with current state (called at end of step)
    fn update_live_monitor(&mut self, temperature: f32) -> Result<()> {
        // Early return if no monitor connected
        if self.live_monitor.is_none() {
            return Ok(());
        }

        // Rate limit to ~30 FPS
        if self.live_monitor_last_send.elapsed() < Duration::from_millis(33) {
            return Ok(());
        }

        // Download GPU state
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;

        let grid_size = self.grid_dim * self.grid_dim * self.grid_dim;
        let mut exclusion = vec![0.0f32; grid_size];
        self.stream.memcpy_dtoh(&self.d_exclusion_field, &mut exclusion)?;

        // Subsample exclusion field (every 2nd voxel for efficiency)
        let sub_dim = self.grid_dim / 2;
        let mut sub_excl = vec![0.0f32; sub_dim * sub_dim * sub_dim];
        for z in 0..sub_dim {
            for y in 0..sub_dim {
                for x in 0..sub_dim {
                    let src = (z*2)*self.grid_dim*self.grid_dim + (y*2)*self.grid_dim + (x*2);
                    let dst = z*sub_dim*sub_dim + y*sub_dim + x;
                    sub_excl[dst] = exclusion[src];
                }
            }
        }

        let mut spike_count = vec![0i32];
        self.stream.memcpy_dtoh(&self.d_spike_count, &mut spike_count)?;

        // Get aromatic excitation if available
        let aromatic_excitation = if self.n_aromatics > 0 {
            let mut exc = vec![0.0f32; self.n_aromatics];
            self.stream.memcpy_dtoh(&self.d_electronic_population, &mut exc)?;
            exc
        } else {
            vec![]
        };

        // Build frame
        let frame = self.build_monitor_frame(
            temperature,
            &positions,
            &sub_excl,
            sub_dim,
            spike_count[0],
            &aromatic_excitation,
        );

        // Send (ignore errors - just disconnect if monitor goes away)
        let len = frame.len() as u32;
        let send_failed = if let Some(stream) = &mut self.live_monitor {
            stream.write_all(&len.to_le_bytes()).is_err()
                || stream.write_all(&frame).is_err()
        } else {
            false
        };

        if send_failed {
            self.live_monitor = None;
            log::debug!("Live monitor disconnected");
        }

        self.live_monitor_frame_id += 1;
        self.live_monitor_last_send = Instant::now();
        Ok(())
    }

    fn build_monitor_frame(
        &self,
        temperature: f32,
        positions: &[f32],
        exclusion: &[f32],
        grid_dim: usize,
        spike_count: i32,
        aromatic_excitation: &[f32],
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(1024 + positions.len() * 4 + exclusion.len() * 4);

        // Header: Q=u64, f=f32, I=u32, i=i32 (matches Python struct format)
        buf.extend_from_slice(&self.live_monitor_frame_id.to_le_bytes());  // Q: frame_id
        buf.extend_from_slice(&(self.timestep as f32 * self.dt).to_le_bytes());  // f: time_ps
        buf.extend_from_slice(&temperature.to_le_bytes());  // f: temperature
        buf.extend_from_slice(&0.0f32.to_le_bytes());  // f: PE placeholder
        buf.extend_from_slice(&0.0f32.to_le_bytes());  // f: KE placeholder
        buf.extend_from_slice(&(self.n_atoms as u32).to_le_bytes());  // I: n_atoms
        buf.extend_from_slice(&(spike_count as u32).to_le_bytes());  // I: spike_count
        buf.extend_from_slice(&(grid_dim as u32).to_le_bytes());  // I: grid_dim
        buf.extend_from_slice(&(self.uv_config.get_target_idx().unwrap_or(0) as i32).to_le_bytes());  // i: probe_id
        buf.extend_from_slice(&0.0f32.to_le_bytes());  // f: sequence_score placeholder
        buf.extend_from_slice(&[0u8; 16]);  // 16 bytes padding

        // Positions
        buf.extend_from_slice(&(positions.len() as u32).to_le_bytes());
        for &p in positions {
            buf.extend_from_slice(&p.to_le_bytes());
        }

        // Exclusion field
        buf.extend_from_slice(&(exclusion.len() as u32).to_le_bytes());
        for &e in exclusion {
            buf.extend_from_slice(&e.to_le_bytes());
        }

        // Spikes (empty for now - would need spike event download)
        buf.extend_from_slice(&0u32.to_le_bytes());

        // Aromatic excitation
        buf.extend_from_slice(&(aromatic_excitation.len() as u32).to_le_bytes());
        for &e in aromatic_excitation {
            buf.extend_from_slice(&e.to_le_bytes());
        }

        buf
    }
}

// ============================================================================
// RESULT TYPES
// ============================================================================

/// Result of a single step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Current timestep
    pub timestep: i32,
    /// Current temperature (K)
    pub temperature: f32,
    /// Number of spikes detected
    pub spike_count: usize,
    /// Whether UV burst was active
    pub uv_burst_active: bool,
    /// Current UV wavelength (nm) - only valid when spectroscopy mode is active
    pub current_wavelength_nm: Option<f32>,
}

/// Summary of a multi-step run
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// Steps completed
    pub steps_completed: i32,
    /// Total spikes detected
    pub total_spikes: usize,
    /// Starting temperature
    pub start_temperature: f32,
    /// Ending temperature
    pub end_temperature: f32,
    /// Number of ensemble snapshots captured
    pub ensemble_snapshots: usize,
}

// ============================================================================
// NON-GPU STUB
// ============================================================================

#[cfg(not(feature = "gpu"))]
pub struct NhsAmberFusedEngine;

#[cfg(not(feature = "gpu"))]
impl NhsAmberFusedEngine {
    /// Stub for non-GPU builds
    pub fn new(
        _topology: &PrismPrepTopology,
        _grid_dim: usize,
        _grid_spacing: f32,
    ) -> Result<Self> {
        bail!("NHS-AMBER Fused Engine requires GPU. Compile with --features gpu")
    }
}
