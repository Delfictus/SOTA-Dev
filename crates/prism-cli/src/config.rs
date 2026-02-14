//! PRISM Configuration Module
//!
//! Provides serde-based TOML configuration parsing for the PRISM pipeline.
//! Replaces brittle manual TOML extraction with type-safe structs.

use anyhow::Result;
use serde::{Deserialize, Serialize};

// Re-export phase configs from their respective crates
pub use prism_core::Phase3Config;
pub use prism_phases::phase0::Phase0Config;
pub use prism_phases::phase1_active_inference::Phase1Config;
pub use prism_phases::phase4_geodesic::Phase4Config;
pub use prism_phases::phase6_tda::Phase6Config;
pub use prism_phases::phase7_ensemble::Phase7Config;
pub use prism_pipeline::{GnnConfig, MemeticConfig, MetaphysicalCouplingConfig};

/// Root configuration for PRISM pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismConfig {
    #[serde(default)]
    pub global: GlobalConfig,

    #[serde(default)]
    pub pipeline: PipelineConfig,

    #[serde(default)]
    pub gpu: GpuConfig,

    #[serde(default)]
    pub warmstart: Option<WarmstartConfig>,

    #[serde(default)]
    pub fluxnet: FluxNetConfig,

    #[serde(default)]
    pub phase0_dendritic: Option<Phase0Config>,

    #[serde(default)]
    pub phase1_active_inference: Option<Phase1Config>,

    #[serde(default)]
    pub phase2: Option<Phase2Config>,

    #[serde(default)]
    pub phase2_thermodynamic: Option<Phase2Config>,

    #[serde(default)]
    pub phase3_quantum: Option<Phase3Config>,

    #[serde(default)]
    pub phase4_geodesic: Option<Phase4Config>,

    #[serde(default)]
    pub phase6_tda: Option<Phase6Config>,

    #[serde(default)]
    pub phase7_ensemble: Option<Phase7Config>,

    #[serde(default)]
    pub memetic: Option<MemeticConfig>,

    #[serde(default)]
    pub metaphysical_coupling: Option<MetaphysicalCouplingConfig>,

    #[serde(default)]
    pub gnn: Option<GnnConfig>,

    #[serde(default)]
    pub telemetry: TelemetryConfig,

    #[serde(default)]
    pub cma_es: Option<CmaEsConfig>,
}

impl PrismConfig {
    /// Load configuration from TOML file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_str(&content)
    }

    /// Parse configuration from TOML string
    pub fn from_str(content: &str) -> Result<Self> {
        Ok(toml::from_str(content)?)
    }

    /// Validate configuration consistency
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    #[serde(default = "default_max_attempts")]
    pub max_attempts: usize,

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    #[serde(default = "default_max_vertices")]
    pub max_vertices: usize,

    #[serde(default = "default_retry_limit")]
    pub retry_limit: usize,

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default)]
    pub device_id: usize,

    #[serde(default)]
    pub devices: Vec<usize>,

    #[serde(default = "default_ptx_dir")]
    pub ptx_dir: String,

    #[serde(default)]
    pub ptx_directory: Option<String>,

    #[serde(default)]
    pub allow_nvrtc: bool,

    #[serde(default)]
    pub require_signed_ptx: bool,

    #[serde(default)]
    pub trusted_ptx_dir: Option<String>,

    #[serde(default = "default_nvml_interval")]
    pub nvml_poll_interval_ms: u64,

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmstartConfig {
    #[serde(default = "default_max_colors")]
    pub max_colors: usize,

    #[serde(default = "default_min_prob")]
    pub min_prob: f32,

    #[serde(default = "default_anchor_fraction")]
    pub anchor_fraction: f32,

    #[serde(default = "default_flux_weight")]
    pub flux_weight: f32,

    #[serde(default = "default_ensemble_weight")]
    pub ensemble_weight: f32,

    #[serde(default = "default_random_weight")]
    pub random_weight: f32,

    #[serde(default)]
    pub curriculum_catalog_path: Option<String>,
}

impl WarmstartConfig {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxNetConfig {
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,

    #[serde(default = "default_alpha")]
    pub alpha: f64,

    #[serde(default = "default_gamma")]
    pub gamma: f64,

    #[serde(default = "default_reward_log_threshold")]
    pub reward_log_threshold: f64,

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
// Phase 0: Dendritic Reservoir
// =============================================================================
// Phase0Config is re-exported at the top of this module

// =============================================================================
// Phase 1: Active Inference
// =============================================================================
// Phase1Config is re-exported at the top of this module

// =============================================================================
// Phase 2: Thermodynamic Annealing
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase2Config {
    #[serde(default = "default_phase2_iterations")]
    pub iterations: usize,

    #[serde(default = "default_phase2_replicas")]
    pub replicas: usize,

    #[serde(default = "default_phase2_temp_min")]
    pub temp_min: f32,

    #[serde(default = "default_phase2_temp_max")]
    pub temp_max: f32,

    // Alternative field names for compatibility
    #[serde(default)]
    pub initial_temperature: Option<f32>,

    #[serde(default)]
    pub cooling_rate: Option<f32>,

    #[serde(default)]
    pub steps_per_temp: Option<usize>,

    #[serde(default)]
    pub num_temps: Option<usize>,
}

impl Phase2Config {
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
// Phase 3: Quantum-Classical Hybrid
// =============================================================================
// Phase3Config is re-exported at the top of this module

// =============================================================================
// Phase 4: Geodesic Distance
// =============================================================================
// Phase4Config is re-exported at the top of this module

// =============================================================================
// Phase 6: Topological Data Analysis (TDA)
// =============================================================================
// Phase6Config is re-exported at the top of this module

// =============================================================================
// Phase 7: Ensemble Aggregation
// =============================================================================
// Phase7Config is re-exported at the top of this module

// =============================================================================
// Memetic Algorithm Configuration
// =============================================================================
// MemeticConfig is re-exported at the top of this module

// =============================================================================
// Metaphysical Coupling Configuration
// =============================================================================
// MetaphysicalCouplingConfig is re-exported at the top of this module

// =============================================================================
// GNN Configuration
// =============================================================================
// GnnConfig is re-exported at the top of this module

// =============================================================================
// Telemetry Configuration
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_telemetry_path")]
    pub path: String,

    #[serde(default = "default_true")]
    pub include_geometry: bool,

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmaEsConfig {
    #[serde(default)]
    pub enabled: bool,

    #[serde(default = "default_cma_population_size")]
    pub population_size: usize,

    #[serde(default = "default_cma_sigma")]
    pub initial_sigma: f32,

    #[serde(default = "default_cma_iterations")]
    pub max_iterations: usize,

    #[serde(default)]
    pub target_fitness: Option<f32>,

    #[serde(default = "default_true")]
    pub use_gpu: bool,
}

// =============================================================================
// Default value functions
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
