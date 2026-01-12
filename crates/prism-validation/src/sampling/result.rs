//! Unified Result Types for Sampling Backends
//!
//! All sampling backends produce the same output format, enabling
//! downstream code to be path-agnostic.
//!
//! # Stability
//!
//! - Required fields are FROZEN after Phase 6
//! - Optional fields (marked `Option<T>`) can be added in future phases
//! - Never remove or change the type of existing fields

use serde::{Deserialize, Serialize};

/// Backend identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendId {
    /// NOVA path - TDA + Active Inference (≤512 atoms)
    Nova,
    /// AMBER mega-fused HMC - proven MD (no limit)
    AmberMegaFused,
    /// Mock backend for testing
    Mock,
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendId::Nova => write!(f, "NOVA"),
            BackendId::AmberMegaFused => write!(f, "AMBER"),
            BackendId::Mock => write!(f, "Mock"),
        }
    }
}

/// Backend capabilities
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Supports TDA (Betti numbers, persistence diagrams)
    pub tda: bool,
    /// Supports Active Inference goal-directed sampling
    pub active_inference: bool,
    /// Maximum atoms supported (None = no limit)
    pub max_atoms: Option<usize>,
    /// Uses GPU acceleration
    pub gpu_accelerated: bool,
}

impl BackendCapabilities {
    /// Create capabilities for NOVA path
    pub fn nova() -> Self {
        Self {
            tda: true,
            active_inference: true,
            max_atoms: Some(512),
            gpu_accelerated: true,
        }
    }

    /// Create capabilities for AMBER path
    pub fn amber() -> Self {
        Self {
            tda: false,
            active_inference: false,
            max_atoms: None,
            gpu_accelerated: true,
        }
    }

    /// Create capabilities for mock path
    pub fn mock() -> Self {
        Self {
            tda: false,
            active_inference: false,
            max_atoms: None,
            gpu_accelerated: false,
        }
    }

    /// Check if structure with n_atoms can be processed
    pub fn can_handle(&self, n_atoms: usize) -> bool {
        match self.max_atoms {
            Some(max) => n_atoms <= max,
            None => true,
        }
    }
}

/// Sampling configuration (backend-agnostic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Number of conformational samples to generate
    pub n_samples: usize,
    /// Steps between saved samples (decorrelation)
    pub steps_per_sample: usize,
    /// Temperature in Kelvin
    pub temperature: f32,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Optional: timestep in femtoseconds (default: 2.0)
    pub timestep_fs: Option<f32>,
    /// Optional: leapfrog steps per HMC proposal (default: 5)
    pub leapfrog_steps: Option<usize>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            n_samples: 500,
            steps_per_sample: 100,
            temperature: 310.0,
            seed: 42,
            timestep_fs: Some(2.0),
            leapfrog_steps: Some(5),
        }
    }
}

impl SamplingConfig {
    /// Create config for quick testing
    pub fn quick() -> Self {
        Self {
            n_samples: 50,
            steps_per_sample: 10,
            ..Default::default()
        }
    }

    /// Create config for production runs
    pub fn production() -> Self {
        Self {
            n_samples: 500,
            steps_per_sample: 100,
            ..Default::default()
        }
    }

    /// Create config for extended sampling (Phase 7+)
    pub fn extended() -> Self {
        Self {
            n_samples: 2000,
            steps_per_sample: 200,
            ..Default::default()
        }
    }
}

/// Unified sampling result
///
/// # Stability
///
/// All fields marked as `Option` can be added in future phases.
/// Required fields (`conformations`, `energies`, `metadata`) are FROZEN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingResult {
    /// Cα coordinates per sample: [n_samples][n_residues][3]
    pub conformations: Vec<Vec<[f32; 3]>>,

    /// Energy per sample (kcal/mol)
    pub energies: Vec<f32>,

    /// Betti numbers per sample (only from TDA-capable backends)
    /// [n_samples][3] where [beta_0, beta_1, beta_2]
    pub betti: Option<Vec<[i32; 3]>>,

    /// Metadata about the run
    pub metadata: SamplingMetadata,

    // === PHASE 7+ ADDITIONS (optional, backward compatible) ===
    // pub persistence_diagrams: Option<Vec<PersistenceDiagram>>,
    // pub active_inference_scores: Option<Vec<f32>>,
}

impl SamplingResult {
    /// Create empty result (for testing)
    pub fn empty(backend: BackendId, n_residues: usize) -> Self {
        Self {
            conformations: Vec::new(),
            energies: Vec::new(),
            betti: None,
            metadata: SamplingMetadata {
                backend,
                n_atoms: 0,
                n_residues,
                n_samples: 0,
                has_tda: false,
                has_active_inference: false,
                elapsed_ms: 0,
                acceptance_rate: None,
            },
        }
    }

    /// Get number of samples
    pub fn n_samples(&self) -> usize {
        self.conformations.len()
    }

    /// Get number of residues (from first conformation)
    pub fn n_residues(&self) -> usize {
        self.conformations.first().map(|c| c.len()).unwrap_or(0)
    }

    /// Check if TDA data is available
    pub fn has_tda(&self) -> bool {
        self.betti.is_some()
    }

    /// Get mean energy
    pub fn mean_energy(&self) -> f32 {
        if self.energies.is_empty() {
            0.0
        } else {
            self.energies.iter().sum::<f32>() / self.energies.len() as f32
        }
    }

    /// Get energy standard deviation
    pub fn energy_std(&self) -> f32 {
        if self.energies.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_energy();
        let var: f32 = self
            .energies
            .iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f32>()
            / (self.energies.len() - 1) as f32;
        var.sqrt()
    }
}

/// Metadata about a sampling run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingMetadata {
    /// Which backend produced this result
    pub backend: BackendId,
    /// Number of atoms in structure
    pub n_atoms: usize,
    /// Number of residues
    pub n_residues: usize,
    /// Number of samples generated
    pub n_samples: usize,
    /// Whether TDA was computed
    pub has_tda: bool,
    /// Whether Active Inference was used
    pub has_active_inference: bool,
    /// Total elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// HMC acceptance rate (if applicable)
    pub acceptance_rate: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id_display() {
        assert_eq!(format!("{}", BackendId::Nova), "NOVA");
        assert_eq!(format!("{}", BackendId::AmberMegaFused), "AMBER");
        assert_eq!(format!("{}", BackendId::Mock), "Mock");
    }

    #[test]
    fn test_capabilities_can_handle() {
        let nova = BackendCapabilities::nova();
        assert!(nova.can_handle(100));
        assert!(nova.can_handle(512));
        assert!(!nova.can_handle(513));

        let amber = BackendCapabilities::amber();
        assert!(amber.can_handle(100));
        assert!(amber.can_handle(10000));
    }

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.n_samples, 500);
        assert_eq!(config.steps_per_sample, 100);
        assert_eq!(config.temperature, 310.0);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_sampling_config_presets() {
        let quick = SamplingConfig::quick();
        assert_eq!(quick.n_samples, 50);

        let prod = SamplingConfig::production();
        assert_eq!(prod.n_samples, 500);

        let ext = SamplingConfig::extended();
        assert_eq!(ext.n_samples, 2000);
    }

    #[test]
    fn test_sampling_result_empty() {
        let result = SamplingResult::empty(BackendId::Mock, 100);
        assert_eq!(result.n_samples(), 0);
        assert_eq!(result.n_residues(), 0);
        assert!(!result.has_tda());
    }

    #[test]
    fn test_sampling_result_stats() {
        let result = SamplingResult {
            conformations: vec![vec![[0.0; 3]; 10]; 5],
            energies: vec![100.0, 110.0, 90.0, 105.0, 95.0],
            betti: None,
            metadata: SamplingMetadata {
                backend: BackendId::Mock,
                n_atoms: 100,
                n_residues: 10,
                n_samples: 5,
                has_tda: false,
                has_active_inference: false,
                elapsed_ms: 1000,
                acceptance_rate: Some(0.75),
            },
        };

        assert_eq!(result.n_samples(), 5);
        assert_eq!(result.n_residues(), 10);
        assert!((result.mean_energy() - 100.0).abs() < 0.01);
        assert!(result.energy_std() > 0.0);
    }
}
