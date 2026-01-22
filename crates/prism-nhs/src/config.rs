//! NHS Configuration and Constants
//!
//! Central configuration for the Neuromorphic Holographic Stream pipeline.
//! All magic numbers are defined here with physical justification.
//!
//! # Physical Constants
//!
//! The UV bias mechanism is based on real aromatic absorption physics:
//! - Tryptophan: ε ≈ 5,600 M⁻¹cm⁻¹ at 280nm
//! - Tyrosine: ε ≈ 1,400 M⁻¹cm⁻¹ at 280nm
//! - Phenylalanine: ε ≈ 200 M⁻¹cm⁻¹ at 280nm
//! - Water: ε ≈ 0 (TRANSPARENT at 280nm)

use serde::{Deserialize, Serialize};

// =============================================================================
// PHYSICAL CONSTANTS
// =============================================================================

/// Water probe radius (Å) - standard value for SASA calculations
pub const WATER_PROBE_RADIUS: f32 = 1.4;

/// Bulk water density (molecules/Å³) at 300K
pub const BULK_WATER_DENSITY: f32 = 0.0334;

/// Boltzmann constant × Temperature (kcal/mol at 300K)
pub const KT_300K: f32 = 0.596;

/// Boltzmann constant in kcal/(mol·K)
pub const KB_KCAL_MOL_K: f32 = 0.001987204;

/// Grid spacing for exclusion field (Å)
/// 0.5Å provides good resolution without excessive memory
pub const DEFAULT_GRID_SPACING: f32 = 0.5;

/// Padding around protein bounding box (Å)
pub const GRID_PADDING: f32 = 8.0;

/// Dewetting spike threshold (fraction of bulk density)
/// Below this = dewetted = spike
pub const DEWETTING_THRESHOLD: f32 = 0.3;

/// Minimum avalanche size to report as cryptic site (number of spikes)
pub const MIN_AVALANCHE_SIZE: usize = 5;

/// Minimum pocket volume to be druggable (Å³)
pub const MIN_DRUGGABLE_VOLUME: f32 = 100.0;

// =============================================================================
// UV ABSORPTION CONSTANTS - MULTI-WAVELENGTH SPECTROSCOPY
// =============================================================================

// ----------------------------- λmax Values (nm) -----------------------------
/// Tryptophan λmax - La band π→π* transition
pub const TRP_LAMBDA_MAX: f32 = 280.0;
/// Tyrosine λmax - phenol π→π* transition
pub const TYR_LAMBDA_MAX: f32 = 274.0;
/// Phenylalanine λmax - benzyl π→π* transition
pub const PHE_LAMBDA_MAX: f32 = 258.0;
/// Disulfide λmax - σ→σ* transition
pub const DISULFIDE_LAMBDA_MAX: f32 = 250.0;

// ------------------------- Extinction Coefficients (M⁻¹cm⁻¹) ----------------
/// Tryptophan molar extinction coefficient at 280nm (M⁻¹cm⁻¹)
/// Strongest UV absorber - primary target for UV bias
pub const TRP_EXTINCTION_280: f32 = 5600.0;
/// Tyrosine molar extinction coefficient at 274nm
pub const TYR_EXTINCTION_274: f32 = 1490.0;
/// Tyrosine at 280nm (secondary peak)
pub const TYR_EXTINCTION_280: f32 = 1400.0;
/// Phenylalanine molar extinction coefficient at 258nm
pub const PHE_EXTINCTION_258: f32 = 200.0;
/// Phenylalanine at 280nm (off-peak)
pub const PHE_EXTINCTION_280: f32 = 200.0;
/// Disulfide molar extinction coefficient at 250nm
pub const DISULFIDE_EXTINCTION_250: f32 = 300.0;

// ----------------------------- Spectral Bandwidths (nm FWHM) ----------------
/// Tryptophan bandwidth
pub const TRP_BANDWIDTH: f32 = 15.0;
/// Tyrosine bandwidth
pub const TYR_BANDWIDTH: f32 = 12.0;
/// Phenylalanine bandwidth
pub const PHE_BANDWIDTH: f32 = 10.0;
/// Disulfide bandwidth (broader due to conformational heterogeneity)
pub const DISULFIDE_BANDWIDTH: f32 = 20.0;

/// Water extinction at 280nm - TRANSPARENT
/// This is the key insight: water doesn't absorb UV at aromatic wavelengths
pub const WATER_EXTINCTION_280: f32 = 0.0;

/// Normalization factor for absorption strengths (relative to Trp)
pub const ABSORPTION_NORMALIZATION: f32 = TRP_EXTINCTION_280;

// ----------------------------- Physical Constants ---------------------------
/// Boltzmann constant in eV/K
pub const KB_EV_K: f32 = 8.617e-5;
/// Planck's constant in eV·s
pub const PLANCK_EV_S: f32 = 4.136e-15;
/// Speed of light in nm/s
pub const SPEED_OF_LIGHT_NM_S: f32 = 2.998e17;

/// Convert wavelength (nm) to photon energy (eV)
/// E = hc/λ
pub fn wavelength_to_ev(wavelength_nm: f32) -> f32 {
    // hc = 1239.84 eV·nm
    1239.84 / wavelength_nm
}

/// Standard wavelengths for frequency hopping protocol (nm)
pub const FREQUENCY_HOP_WAVELENGTHS: [f32; 5] = [258.0, 265.0, 274.0, 280.0, 290.0];

/// Disulfide bond maximum distance (Å) for S-S detection
pub const DISULFIDE_BOND_MAX_DISTANCE: f32 = 2.5;

// =============================================================================
// MAIN CONFIGURATION
// =============================================================================

/// NHS Pipeline Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NhsConfig {
    // =========================================================================
    // Grid parameters
    // =========================================================================

    /// Grid spacing in Angstroms (default: 0.5Å)
    pub grid_spacing: f32,

    /// Padding around protein bounding box in Angstroms
    pub grid_padding: f32,

    // =========================================================================
    // Exclusion field parameters
    // =========================================================================

    /// Gaussian width = VdW_radius × this factor
    pub exclusion_sigma_scale: f32,

    /// Scaling for polar attraction field
    pub polar_attraction_scale: f32,

    // =========================================================================
    // Neuromorphic parameters
    // =========================================================================

    /// Water density threshold for spike (fraction of bulk)
    pub spike_threshold: f32,

    /// LIF neuron membrane time constant (arbitrary units)
    pub membrane_tau: f32,

    /// Lateral synaptic connection weight
    pub synaptic_strength: f32,

    /// Refractory period after spike (frames)
    pub refractory_period: u32,

    // =========================================================================
    // Avalanche detection
    // =========================================================================

    /// Minimum spikes to form valid avalanche
    pub min_avalanche_spikes: usize,

    /// Maximum distance to cluster spikes (Å)
    pub avalanche_spatial_threshold: f32,

    /// Frames to integrate for temporal clustering
    pub avalanche_temporal_window: usize,

    // =========================================================================
    // Site classification
    // =========================================================================

    /// Minimum pocket volume (Å³)
    pub min_volume: f32,

    /// CV(SASA) threshold for cryptic classification
    pub cv_sasa_threshold: f32,

    /// Minimum open frequency
    pub open_freq_min: f32,

    /// Maximum open frequency
    pub open_freq_max: f32,

    // =========================================================================
    // UV Bias parameters (Stage 6)
    // =========================================================================

    /// Enable UV bias perturbation mechanism
    pub uv_bias_enabled: bool,

    /// UV bias configuration
    pub uv_bias: UvBiasConfig,

    // =========================================================================
    // Performance tuning
    // =========================================================================

    /// Use FFT acceleration for holographic encoding
    pub use_fft_acceleration: bool,

    /// Streaming output buffer size
    pub stream_buffer_size: usize,
}

impl Default for NhsConfig {
    fn default() -> Self {
        Self {
            // Grid
            grid_spacing: DEFAULT_GRID_SPACING,
            grid_padding: GRID_PADDING,

            // Exclusion
            exclusion_sigma_scale: 0.3,
            polar_attraction_scale: 2.0,

            // Neuromorphic
            spike_threshold: DEWETTING_THRESHOLD,
            membrane_tau: 5.0,
            synaptic_strength: 0.1,
            refractory_period: 3,

            // Avalanche
            min_avalanche_spikes: MIN_AVALANCHE_SIZE,
            avalanche_spatial_threshold: 6.0,
            avalanche_temporal_window: 10,

            // Site classification (matching existing PRISM-Cryptic)
            min_volume: MIN_DRUGGABLE_VOLUME,
            cv_sasa_threshold: 0.20,
            open_freq_min: 0.05,
            open_freq_max: 0.90,

            // UV Bias - enabled by default
            uv_bias_enabled: true,
            uv_bias: UvBiasConfig::default(),

            // Performance
            use_fft_acceleration: true,
            stream_buffer_size: 1024,
        }
    }
}

// =============================================================================
// UV BIAS CONFIGURATION
// =============================================================================

/// UV Bias Perturbation Configuration
///
/// Controls the pump-probe style targeted perturbation of aromatic residues.
/// Water is transparent at 280nm, so perturbations create signal on silent background.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvBiasConfig {
    // =========================================================================
    // Target selection
    // =========================================================================

    /// Minimum pocket probability to target nearby aromatics
    pub pocket_probability_threshold: f32,

    /// Maximum distance from high-probability voxel to target aromatic (Å)
    pub target_selection_radius: f32,

    /// Include tryptophan as target (strongest absorber)
    pub target_trp: bool,

    /// Include tyrosine as target
    pub target_tyr: bool,

    /// Include phenylalanine as target (weakest absorber)
    pub target_phe: bool,

    /// Only target surface-exposed aromatics
    pub surface_only: bool,

    /// Minimum solvent accessibility to be considered "surface"
    pub min_sasa_exposure: f32,

    // =========================================================================
    // Burst generation
    // =========================================================================

    /// Use burst mode (true) or continuous perturbation (false)
    pub burst_mode: bool,

    /// Number of pulses per burst
    pub pulses_per_burst: u32,

    /// Frames between pulses within a burst
    pub intra_burst_interval: u32,

    /// Frames between bursts (observation window)
    pub inter_burst_interval: u32,

    /// Base perturbation intensity (velocity boost in Å/ps)
    pub base_intensity: f32,

    /// Scale intensity by absorption coefficient
    pub scale_by_absorption: bool,

    // =========================================================================
    // Perturbation physics
    // =========================================================================

    /// Apply perturbation in aromatic ring plane (mimics π→π* excitation)
    pub ring_plane_perturbation: bool,

    /// Random direction component (0 = fully directed, 1 = fully random)
    pub direction_randomness: f32,

    /// Temperature equivalent of perturbation (K above ambient)
    pub effective_temperature_boost: f32,

    // =========================================================================
    // Response correlation
    // =========================================================================

    /// Enable causal correlation tracking
    pub track_causality: bool,

    /// Maximum lag to check for pump-probe correlation (frames)
    pub max_correlation_lag: usize,

    /// Minimum correlation coefficient to establish causality
    pub min_correlation_threshold: f32,

    /// Window size for correlation computation
    pub correlation_window: usize,
}

impl Default for UvBiasConfig {
    fn default() -> Self {
        Self {
            // Target selection
            pocket_probability_threshold: 0.3,
            target_selection_radius: 8.0,
            target_trp: true,   // Primary target
            target_tyr: true,   // Secondary target
            target_phe: false,  // Weak absorber, skip by default
            surface_only: true,
            min_sasa_exposure: 0.2,

            // Burst generation - pump-probe style
            burst_mode: true,
            pulses_per_burst: 3,
            intra_burst_interval: 2,
            inter_burst_interval: 20,  // Observation window
            base_intensity: 0.5,       // Å/ps velocity boost
            scale_by_absorption: true,

            // Perturbation physics
            ring_plane_perturbation: true,
            direction_randomness: 0.3,
            effective_temperature_boost: 50.0,  // +50K local heating

            // Response correlation
            track_causality: true,
            max_correlation_lag: 15,
            min_correlation_threshold: 0.5,
            correlation_window: 50,
        }
    }
}

impl UvBiasConfig {
    /// Get relative absorption strength for a residue type
    pub fn absorption_strength(&self, residue: &str) -> f32 {
        match residue.to_uppercase().as_str() {
            "TRP" | "W" => TRP_EXTINCTION_280 / ABSORPTION_NORMALIZATION,
            "TYR" | "Y" => TYR_EXTINCTION_280 / ABSORPTION_NORMALIZATION,
            "PHE" | "F" => PHE_EXTINCTION_280 / ABSORPTION_NORMALIZATION,
            _ => 0.0,  // Non-aromatic: no absorption
        }
    }

    /// Check if a residue type is a valid target
    pub fn is_valid_target(&self, residue: &str) -> bool {
        match residue.to_uppercase().as_str() {
            "TRP" | "W" => self.target_trp,
            "TYR" | "Y" => self.target_tyr,
            "PHE" | "F" => self.target_phe,
            _ => false,
        }
    }
}

// =============================================================================
// UV SPECTROSCOPY CONFIGURATION (Enhanced Multi-Wavelength)
// =============================================================================

/// UV Spectroscopy Configuration for full multi-wavelength pump-probe
///
/// Extends UvBiasConfig with:
/// - Frequency hopping protocol
/// - Disulfide bond targeting
/// - Local temperature tracking
/// - π→π* electronic state modeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvSpectroscopyConfig {
    /// Base UV bias config (backward compatible)
    pub base: UvBiasConfig,

    // =========================================================================
    // Frequency Hopping Protocol
    // =========================================================================

    /// Enable frequency hopping (wavelength scanning)
    pub frequency_hopping_enabled: bool,

    /// Wavelengths to scan (nm)
    pub scan_wavelengths: Vec<f32>,

    /// Dwell time per wavelength (MD steps)
    pub dwell_steps: u32,

    /// Number of full spectral scans
    pub n_scans: u32,

    // =========================================================================
    // Disulfide Bond Targeting
    // =========================================================================

    /// Enable disulfide bond (S-S) targeting at 250nm
    pub target_disulfides: bool,

    /// Maximum S-S bond distance for detection (Å)
    pub disulfide_max_distance: f32,

    // =========================================================================
    // Local Temperature Tracking
    // =========================================================================

    /// Enable local temperature tracking from photon absorption
    pub track_local_temperature: bool,

    /// Photon fluence for energy deposition calculation (photons/Å²)
    pub photon_fluence: f32,

    /// Thermal dissipation time constant (ps)
    pub thermal_dissipation_tau: f32,

    /// Number of atoms to include in local temperature calculation
    pub local_temp_shell_atoms: usize,

    // =========================================================================
    // Electronic State Modeling
    // =========================================================================

    /// Enable π→π* transition modeling
    pub model_electronic_transitions: bool,

    /// Excited state lifetime (ps) - vibrational relaxation
    pub excited_state_lifetime: f32,

    /// Fraction of energy deposited to ring atoms (vs dissipated)
    pub energy_deposition_fraction: f32,
}

impl Default for UvSpectroscopyConfig {
    fn default() -> Self {
        Self {
            base: UvBiasConfig::default(),

            // Frequency hopping - disabled by default for backward compat
            frequency_hopping_enabled: false,
            scan_wavelengths: vec![258.0, 265.0, 274.0, 280.0, 290.0],
            dwell_steps: 1000,  // 2 ps per wavelength at 2 fs timestep
            n_scans: 5,

            // Disulfide targeting - disabled by default
            target_disulfides: false,
            disulfide_max_distance: DISULFIDE_BOND_MAX_DISTANCE,

            // Local temperature tracking - enabled
            track_local_temperature: true,
            photon_fluence: 1.0,           // photons/Å²
            thermal_dissipation_tau: 5.0,  // 5 ps decay
            local_temp_shell_atoms: 20,

            // Electronic state modeling - enabled
            model_electronic_transitions: true,
            excited_state_lifetime: 10.0,  // 10 ps for vibrational relaxation
            energy_deposition_fraction: 0.8,
        }
    }
}

impl UvSpectroscopyConfig {
    /// Create config with full frequency hopping enabled
    pub fn with_frequency_hopping() -> Self {
        Self {
            frequency_hopping_enabled: true,
            ..Default::default()
        }
    }

    /// Create config with disulfide targeting enabled
    pub fn with_disulfides() -> Self {
        Self {
            target_disulfides: true,
            ..Default::default()
        }
    }

    /// Create full publication-quality config
    pub fn publication_quality() -> Self {
        Self {
            frequency_hopping_enabled: true,
            target_disulfides: true,
            track_local_temperature: true,
            model_electronic_transitions: true,
            n_scans: 10,
            ..Default::default()
        }
    }

    /// Get chromophore spec for a residue type at a wavelength
    pub fn get_chromophore_absorption(&self, residue: &str, wavelength: f32) -> f32 {
        let (lambda_max, epsilon_max, bandwidth) = match residue.to_uppercase().as_str() {
            "TRP" | "W" => (TRP_LAMBDA_MAX, TRP_EXTINCTION_280, TRP_BANDWIDTH),
            "TYR" | "Y" => (TYR_LAMBDA_MAX, TYR_EXTINCTION_274, TYR_BANDWIDTH),
            "PHE" | "F" => (PHE_LAMBDA_MAX, PHE_EXTINCTION_258, PHE_BANDWIDTH),
            "CYS" | "C" | "CYX" => (DISULFIDE_LAMBDA_MAX, DISULFIDE_EXTINCTION_250, DISULFIDE_BANDWIDTH),
            _ => return 0.0,
        };

        // Gaussian absorption profile
        let delta = wavelength - lambda_max;
        let sigma = bandwidth / 2.355;  // FWHM to sigma
        epsilon_max * (-0.5 * (delta / sigma).powi(2)).exp()
    }

    /// Get current wavelength for frequency hopping at given step
    pub fn current_wavelength(&self, step: u64) -> f32 {
        if !self.frequency_hopping_enabled || self.scan_wavelengths.is_empty() {
            return 280.0;  // Default to 280nm
        }

        let scan_length = self.scan_wavelengths.len() as u64 * self.dwell_steps as u64;
        let position = ((step % scan_length) / self.dwell_steps as u64) as usize;
        self.scan_wavelengths[position % self.scan_wavelengths.len()]
    }

    /// Compute local temperature increase from photon absorption
    ///
    /// ΔT = E / (3/2 * k_B * N_atoms)
    pub fn compute_local_heating(&self, wavelength: f32, extinction: f32) -> f32 {
        let photon_energy = wavelength_to_ev(wavelength);

        // Absorption cross-section (simplified: proportional to extinction)
        let absorption_cross = extinction / 1000.0;  // Å² approximation

        // Energy deposited per chromophore
        let energy_deposited = photon_energy * absorption_cross * self.photon_fluence;

        // Convert to temperature increase (equipartition)
        let n_ring_atoms = 6.0;  // Approximate for benzene ring
        let delta_t = energy_deposited / (1.5 * KB_EV_K * n_ring_atoms);

        delta_t * self.energy_deposition_fraction
    }
}

// =============================================================================
// HYDROPHOBICITY THRESHOLDS
// =============================================================================

/// Hydrophobic atom classification thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydrophobicityThresholds {
    /// Residues with hydrophobicity > this are "hydrophobic"
    pub hydrophobic_cutoff: f32,

    /// Residues with hydrophobicity < this are "hydrophilic"
    pub hydrophilic_cutoff: f32,
}

impl Default for HydrophobicityThresholds {
    fn default() -> Self {
        Self {
            hydrophobic_cutoff: 0.6,  // ILE, LEU, VAL, PHE, MET, etc.
            hydrophilic_cutoff: 0.4,  // ARG, LYS, ASP, GLU, etc.
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absorption_strengths() {
        let config = UvBiasConfig::default();

        // Trp should be strongest (normalized to 1.0)
        assert!((config.absorption_strength("TRP") - 1.0).abs() < 0.01);

        // Tyr should be ~0.25
        assert!((config.absorption_strength("TYR") - 0.25).abs() < 0.05);

        // Phe should be ~0.036
        assert!(config.absorption_strength("PHE") < 0.1);

        // Water should be 0
        assert_eq!(config.absorption_strength("HOH"), 0.0);

        // Alanine should be 0
        assert_eq!(config.absorption_strength("ALA"), 0.0);
    }

    #[test]
    fn test_target_selection() {
        let config = UvBiasConfig::default();

        assert!(config.is_valid_target("TRP"));
        assert!(config.is_valid_target("TYR"));
        assert!(!config.is_valid_target("PHE"));  // Disabled by default
        assert!(!config.is_valid_target("ALA"));
    }

    #[test]
    fn test_water_transparency() {
        // This is the key physical insight
        assert_eq!(WATER_EXTINCTION_280, 0.0);
    }
}
