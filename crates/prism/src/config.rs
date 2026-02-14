//! PRISM Configuration Module
//!
//! Provides serde-based TOML configuration parsing for the PRISM pipeline.
//! Replaces brittle manual TOML extraction with type-safe structs.
//!
//! # Architecture
//! This module serves as the single source of truth for all PRISM configuration.
//! It provides a hierarchical configuration structure that maps directly to TOML files.
//!
//! # Example TOML
//! ```toml
//! [global]
//! max_attempts = 1000
//! enable_fluxnet_rl = true
//!
//! [gpu]
//! enabled = true
//! device_id = 0
//! ptx_dir = "target/ptx"
//!
//! [phase2]
//! iterations = 50000
//! replicas = 8
//! temp_min = 0.01
//! temp_max = 10.0
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};

// Re-export phase configs from their respective crates for convenience
pub use prism_core::Phase3Config;
pub use prism_phases::phase0::Phase0Config;
pub use prism_phases::phase1_active_inference::Phase1Config;
pub use prism_phases::phase4_geodesic::Phase4Config;
pub use prism_phases::phase6_tda::Phase6Config;
pub use prism_phases::phase7_ensemble::Phase7Config;
pub use prism_pipeline::{GnnConfig, MemeticConfig, MetaphysicalCouplingConfig};

/// Root configuration for PRISM pipeline.
///
/// This is the top-level configuration struct that contains all subsystem configurations.
/// It can be loaded from a TOML file or constructed programmatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismConfig {
    /// Global settings that apply across all phases
    #[serde(default)]
    pub global: GlobalConfig,

    /// Core pipeline settings
    #[serde(default)]
    pub pipeline: PipelineConfig,

    /// GPU acceleration settings
    #[serde(default)]
    pub gpu: GpuConfig,

    /// Warmstart system configuration (optional)
    #[serde(default)]
    pub warmstart: Option<WarmstartConfig>,

    /// FluxNet RL controller settings
    #[serde(default)]
    pub fluxnet: FluxNetConfig,

    /// Phase 0: Dendritic Reservoir configuration
    #[serde(default)]
    pub phase0_dendritic: Option<Phase0Config>,

    /// Phase 1: Active Inference configuration
    #[serde(default)]
    pub phase1_active_inference: Option<Phase1Config>,

    /// Phase 2: Thermodynamic Annealing configuration
    #[serde(default)]
    pub phase2: Option<Phase2Config>,

    /// Phase 2 alternative name for compatibility
    #[serde(default)]
    pub phase2_thermodynamic: Option<Phase2Config>,

    /// Phase 3: Quantum-Classical Hybrid configuration
    #[serde(default)]
    pub phase3_quantum: Option<Phase3Config>,

    /// Phase 4: Geodesic Distance configuration
    #[serde(default)]
    pub phase4_geodesic: Option<Phase4Config>,

    /// Phase 6: Topological Data Analysis configuration
    #[serde(default)]
    pub phase6_tda: Option<Phase6Config>,

    /// Phase 7: Ensemble Aggregation configuration
    #[serde(default)]
    pub phase7_ensemble: Option<Phase7Config>,

    /// Memetic Algorithm configuration
    #[serde(default)]
    pub memetic: Option<MemeticConfig>,

    /// Metaphysical Coupling configuration
    #[serde(default)]
    pub metaphysical_coupling: Option<MetaphysicalCouplingConfig>,

    /// GNN inference configuration
    #[serde(default)]
    pub gnn: Option<GnnConfig>,

    /// Telemetry and observability settings
    #[serde(default)]
    pub telemetry: TelemetryConfig,

    /// CMA-ES evolutionary optimization configuration
    #[serde(default)]
    pub cma_es: Option<CmaEsConfig>,
}

impl PrismConfig {
    /// Load configuration from a TOML file.
    ///
    /// # Arguments
    /// * `path` - Path to the TOML configuration file
    ///
    /// # Returns
    /// Parsed configuration or error if file cannot be read or parsed
    ///
    /// # Example
    /// ```rust,no_run
    /// use prism::config::PrismConfig;
    ///
    /// let config = PrismConfig::from_file("configs/dsjc500.toml")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_str(&content)
    }

    /// Parse configuration from a TOML string.
    ///
    /// # Arguments
    /// * `content` - TOML content as a string
    ///
    /// # Returns
    /// Parsed configuration or error if content is invalid TOML
    pub fn from_str(content: &str) -> Result<Self> {
        Ok(toml::from_str(content)?)
    }

    /// Validate configuration consistency and constraints.
    ///
    /// Performs semantic validation beyond basic parsing:
    /// - Warmstart weights must sum to 1.0
    /// - Temperature ranges must be valid
    /// - Replica counts must be positive
    ///
    /// # Returns
    /// Ok(()) if valid, or error describing the validation failure
    pub fn validate(&self) -> Result<()> {
        // Validate warmstart weights if present
        if let Some(ref ws) = self.warmstart {
            ws.validate()?;
        }

        // Validate phase2 config
        if let Some(ref p2) = self.phase2 {
            p2.validate()?;
        }
        if let Some(ref p2) = self.phase2_thermodynamic {
            p2.validate()?;
        }

        Ok(())
    }
}

impl Default for PrismConfig {
    fn default() -> Self {
        Self {
            global: GlobalConfig::default(),
            pipeline: PipelineConfig::default(),
            gpu: GpuConfig::default(),
            warmstart: None,
            fluxnet: FluxNetConfig::default(),
            phase0_dendritic: None,
            phase1_active_inference: None,
            phase2: None,
            phase2_thermodynamic: None,
            phase3_quantum: None,
            phase4_geodesic: None,
            phase6_tda: None,
            phase7_ensemble: None,
            memetic: None,
            metaphysical_coupling: None,
            gnn: None,
            telemetry: TelemetryConfig::default(),
            cma_es: None,
        }
    }
}

// =============================================================================
// Global Configuration
// =============================================================================

/// Global settings that apply across the entire PRISM pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// Maximum number of optimization attempts
    #[serde(default = "default_max_attempts")]
    pub max_attempts: usize,

    /// Enable FluxNet RL controller
    #[serde(default = "default_true")]
    pub enable_fluxnet_rl: bool,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            enable_fluxnet_rl: true,
        }
    }
}

// =============================================================================
// Pipeline Configuration
// =============================================================================

/// Core pipeline settings controlling execution behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Maximum supported graph vertices
    #[serde(default = "default_max_vertices")]
    pub max_vertices: usize,

    /// Phase retry limit before escalation
    #[serde(default = "default_retry_limit")]
    pub retry_limit: usize,

    /// Custom telemetry output path (optional)
    #[serde(default)]
    pub telemetry_path: Option<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_vertices: 10000,
            retry_limit: 3,
            telemetry_path: None,
        }
    }
}

// =============================================================================
// GPU Configuration
// =============================================================================

/// GPU acceleration settings for CUDA-enabled phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Primary CUDA device ID
    #[serde(default)]
    pub device_id: usize,

    /// List of CUDA devices for multi-GPU
    #[serde(default)]
    pub devices: Vec<usize>,

    /// PTX kernel directory path
    #[serde(default = "default_ptx_dir")]
    pub ptx_dir: String,

    /// Alternative PTX directory field
    #[serde(default)]
    pub ptx_directory: Option<String>,

    /// Allow NVRTC runtime compilation
    #[serde(default)]
    pub allow_nvrtc: bool,

    /// Require cryptographically signed PTX
    #[serde(default)]
    pub require_signed_ptx: bool,

    /// Trusted PTX directory for signed kernels
    #[serde(default)]
    pub trusted_ptx_dir: Option<String>,

    /// NVML telemetry polling interval (ms)
    #[serde(default = "default_nvml_interval")]
    pub nvml_poll_interval_ms: u64,

    /// Multi-GPU scheduling policy
    #[serde(default = "default_scheduling_policy")]
    pub scheduling_policy: String,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_id: 0,
            devices: vec![0],
            ptx_dir: "target/ptx".to_string(),
            ptx_directory: None,
            allow_nvrtc: false,
            require_signed_ptx: false,
            trusted_ptx_dir: None,
            nvml_poll_interval_ms: 1000,
            scheduling_policy: "round-robin".to_string(),
        }
    }
}

// =============================================================================
// Warmstart Configuration
// =============================================================================

/// Warmstart system configuration for probabilistic color priors.
///
/// The warmstart system provides Phase 1-7 solvers with informed initial
/// color distributions derived from Phase 0 reservoir dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmstartConfig {
    /// Maximum colors in prior distribution
    #[serde(default = "default_max_colors")]
    pub max_colors: usize,

    /// Minimum probability (prevents zero probabilities)
    #[serde(default = "default_min_prob")]
    pub min_prob: f32,

    /// Fraction of vertices to designate as anchors
    #[serde(default = "default_anchor_fraction")]
    pub anchor_fraction: f32,

    /// Weight for flux reservoir contribution
    #[serde(default = "default_flux_weight")]
    pub flux_weight: f32,

    /// Weight for ensemble method contribution
    #[serde(default = "default_ensemble_weight")]
    pub ensemble_weight: f32,

    /// Weight for random exploration contribution
    #[serde(default = "default_random_weight")]
    pub random_weight: f32,

    /// Path to curriculum profile catalog (optional)
    #[serde(default)]
    pub curriculum_catalog_path: Option<String>,
}

impl WarmstartConfig {
    /// Validate warmstart configuration constraints.
    pub fn validate(&self) -> Result<()> {
        let weight_sum = self.flux_weight + self.ensemble_weight + self.random_weight;
        if (weight_sum - 1.0).abs() > 0.01 {
            anyhow::bail!(
                "Warmstart weights must sum to 1.0 (got {:.3}). flux={:.2}, ensemble={:.2}, random={:.2}",
                weight_sum, self.flux_weight, self.ensemble_weight, self.random_weight
            );
        }

        if self.anchor_fraction < 0.0 || self.anchor_fraction > 1.0 {
            anyhow::bail!(
                "Warmstart anchor fraction must be in [0.0, 1.0] (got {:.3})",
                self.anchor_fraction
            );
        }

        if self.max_colors == 0 {
            anyhow::bail!("Warmstart max_colors must be > 0");
        }

        Ok(())
    }
}

// =============================================================================
// FluxNet RL Configuration
// =============================================================================

/// FluxNet reinforcement learning controller configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxNetConfig {
    /// Exploration rate (epsilon-greedy)
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,

    /// Learning rate (Q-value update step size)
    #[serde(default = "default_alpha")]
    pub alpha: f64,

    /// Discount factor (future reward importance)
    #[serde(default = "default_gamma")]
    pub gamma: f64,

    /// Reward logging threshold
    #[serde(default = "default_reward_log_threshold")]
    pub reward_log_threshold: f64,

    /// Path to pretrained Q-table (optional)
    #[serde(default)]
    pub qtable_path: Option<String>,
}

impl Default for FluxNetConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.2,
            alpha: 0.1,
            gamma: 0.95,
            reward_log_threshold: 0.001,
            qtable_path: None,
        }
    }
}

// =============================================================================
// Phase 2: Thermodynamic Annealing Configuration
// =============================================================================

/// Phase 2 thermodynamic simulated annealing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase2Config {
    /// Number of annealing iterations
    #[serde(default = "default_phase2_iterations")]
    pub iterations: usize,

    /// Number of parallel temperature replicas
    #[serde(default = "default_phase2_replicas")]
    pub replicas: usize,

    /// Minimum temperature
    #[serde(default = "default_phase2_temp_min")]
    pub temp_min: f32,

    /// Maximum temperature
    #[serde(default = "default_phase2_temp_max")]
    pub temp_max: f32,

    /// Alternative: initial temperature
    #[serde(default)]
    pub initial_temperature: Option<f32>,

    /// Alternative: cooling rate
    #[serde(default)]
    pub cooling_rate: Option<f32>,

    /// Alternative: steps per temperature
    #[serde(default)]
    pub steps_per_temp: Option<usize>,

    /// Alternative: number of temperatures
    #[serde(default)]
    pub num_temps: Option<usize>,
}

impl Phase2Config {
    /// Validate Phase 2 configuration constraints.
    pub fn validate(&self) -> Result<()> {
        if self.temp_min >= self.temp_max {
            anyhow::bail!(
                "Phase2 temp_min ({}) must be < temp_max ({})",
                self.temp_min,
                self.temp_max
            );
        }

        if self.replicas == 0 {
            anyhow::bail!("Phase2 replicas must be > 0");
        }

        Ok(())
    }
}

// =============================================================================
// Telemetry Configuration
// =============================================================================

/// Telemetry and observability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable telemetry collection
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Output file path
    #[serde(default = "default_telemetry_path")]
    pub path: String,

    /// Include geometry metrics
    #[serde(default = "default_true")]
    pub include_geometry: bool,

    /// Include quantum metrics
    #[serde(default = "default_true")]
    pub include_quantum: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "telemetry.jsonl".to_string(),
            include_geometry: true,
            include_quantum: true,
        }
    }
}

// =============================================================================
// CMA-ES Configuration
// =============================================================================

/// CMA-ES evolutionary optimization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmaEsConfig {
    /// Enable CMA-ES optimization
    #[serde(default)]
    pub enabled: bool,

    /// Population size
    #[serde(default = "default_cma_population_size")]
    pub population_size: usize,

    /// Initial step size (sigma)
    #[serde(default = "default_cma_sigma")]
    pub initial_sigma: f32,

    /// Maximum iterations
    #[serde(default = "default_cma_iterations")]
    pub max_iterations: usize,

    /// Target fitness (optional early stopping)
    #[serde(default)]
    pub target_fitness: Option<f32>,

    /// Enable GPU acceleration for CMA-ES
    #[serde(default = "default_true")]
    pub use_gpu: bool,
}

// =============================================================================
// Default Value Functions
// =============================================================================

fn default_max_attempts() -> usize { 1 }
fn default_true() -> bool { true }
fn default_max_vertices() -> usize { 10000 }
fn default_retry_limit() -> usize { 3 }
fn default_ptx_dir() -> String { "target/ptx".to_string() }
fn default_nvml_interval() -> u64 { 1000 }
fn default_scheduling_policy() -> String { "round-robin".to_string() }
fn default_max_colors() -> usize { 50 }
fn default_min_prob() -> f32 { 0.01 }
fn default_anchor_fraction() -> f32 { 0.10 }
fn default_flux_weight() -> f32 { 0.4 }
fn default_ensemble_weight() -> f32 { 0.4 }
fn default_random_weight() -> f32 { 0.2 }
fn default_epsilon() -> f64 { 0.2 }
fn default_alpha() -> f64 { 0.1 }
fn default_gamma() -> f64 { 0.95 }
fn default_reward_log_threshold() -> f64 { 0.001 }
fn default_phase2_iterations() -> usize { 10000 }
fn default_phase2_replicas() -> usize { 8 }
fn default_phase2_temp_min() -> f32 { 0.01 }
fn default_phase2_temp_max() -> f32 { 10.0 }
fn default_telemetry_path() -> String { "telemetry.jsonl".to_string() }
fn default_cma_population_size() -> usize { 50 }
fn default_cma_sigma() -> f32 { 0.5 }
fn default_cma_iterations() -> usize { 100 }

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PrismConfig::default();
        assert!(config.global.enable_fluxnet_rl);
        assert!(config.gpu.enabled);
        assert_eq!(config.pipeline.max_vertices, 10000);
    }

    #[test]
    fn test_warmstart_validation() {
        let mut ws = WarmstartConfig {
            max_colors: 50,
            min_prob: 0.01,
            anchor_fraction: 0.1,
            flux_weight: 0.4,
            ensemble_weight: 0.4,
            random_weight: 0.2,
            curriculum_catalog_path: None,
        };
        assert!(ws.validate().is_ok());

        // Invalid weights
        ws.random_weight = 0.5;
        assert!(ws.validate().is_err());
    }

    #[test]
    fn test_phase2_validation() {
        let mut p2 = Phase2Config {
            iterations: 10000,
            replicas: 8,
            temp_min: 0.01,
            temp_max: 10.0,
            initial_temperature: None,
            cooling_rate: None,
            steps_per_temp: None,
            num_temps: None,
        };
        assert!(p2.validate().is_ok());

        // Invalid temperature range
        p2.temp_min = 20.0;
        assert!(p2.validate().is_err());
    }

    #[test]
    fn test_toml_parsing() {
        let toml = r#"
            [global]
            max_attempts = 100

            [gpu]
            enabled = true
            device_id = 0

            [phase2]
            iterations = 50000
            replicas = 12
        "#;

        let config = PrismConfig::from_str(toml).unwrap();
        assert_eq!(config.global.max_attempts, 100);
        assert!(config.gpu.enabled);
        assert_eq!(config.phase2.as_ref().unwrap().iterations, 50000);
        assert_eq!(config.phase2.as_ref().unwrap().replicas, 12);
    }
}
