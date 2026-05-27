//! Pipeline configuration and validation.

use prism_core::{PhaseConfig, PrismError, WarmstartConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Maximum number of vertices (safety guardrail).
pub const MAX_VERTICES: usize = 10000;

/// Pipeline configuration.
///
/// Implements PRISM GPU Plan §4.3: Configuration Validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Maximum vertices allowed
    pub max_vertices: usize,

    /// Per-phase configurations
    pub phase_configs: HashMap<String, PhaseConfig>,

    /// Global timeout (seconds)
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    /// Enable telemetry
    #[serde(default = "default_true")]
    pub enable_telemetry: bool,

    /// Telemetry output path
    #[serde(default = "default_telemetry_path")]
    pub telemetry_path: String,

    /// Warmstart configuration (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmstart_config: Option<WarmstartConfig>,

    /// GPU configuration
    #[serde(default)]
    pub gpu: GpuConfig,

    /// Phase 2 hyperparameters
    #[serde(default)]
    pub phase2: Phase2Config,

    /// Memetic algorithm configuration (optional, Phase 7)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memetic: Option<MemeticConfig>,

    /// Metaphysical telemetry coupling configuration (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metaphysical_coupling: Option<MetaphysicalCouplingConfig>,

    /// Ontology phase configuration (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ontology: Option<OntologyConfig>,

    /// MEC (Molecular Emergent Computing) configuration (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mec: Option<MecConfig>,

    /// CMA-ES optimization configuration (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cma_es: Option<CmaEsConfig>,

    /// GNN (Graph Neural Network) configuration (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gnn: Option<GnnConfig>,
}

fn default_timeout() -> u64 {
    3600 // 1 hour
}

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_telemetry_path() -> String {
    "telemetry.jsonl".to_string()
}

impl PipelineConfig {
    /// Creates a new configuration builder.
    pub fn builder() -> PipelineConfigBuilder {
        PipelineConfigBuilder::default()
    }

    /// Validates the configuration.
    ///
    /// Returns an error if any constraints are violated.
    pub fn validate(&self) -> Result<(), PrismError> {
        // Check max_vertices
        if self.max_vertices == 0 {
            return Err(PrismError::config("max_vertices must be greater than 0"));
        }

        if self.max_vertices > MAX_VERTICES {
            return Err(PrismError::config(format!(
                "max_vertices ({}) exceeds MAX_VERTICES ({})",
                self.max_vertices, MAX_VERTICES
            )));
        }

        // Check timeout
        if self.timeout_seconds == 0 {
            return Err(PrismError::config("timeout_seconds must be greater than 0"));
        }

        // Validate each phase config
        for (phase_name, phase_config) in &self.phase_configs {
            if phase_config.max_iterations == 0 {
                return Err(PrismError::config(format!(
                    "Phase {} has max_iterations = 0",
                    phase_name
                )));
            }

            if phase_config.convergence_threshold < 0.0 {
                return Err(PrismError::config(format!(
                    "Phase {} has negative convergence_threshold",
                    phase_name
                )));
            }
        }

        Ok(())
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_vertices: MAX_VERTICES,
            phase_configs: HashMap::new(),
            timeout_seconds: default_timeout(),
            enable_telemetry: true,
            telemetry_path: default_telemetry_path(),
            warmstart_config: None,
            gpu: GpuConfig::default(),
            phase2: Phase2Config::default(),
            memetic: None,
            metaphysical_coupling: None,
            ontology: None,
            mec: None,
            cma_es: None,
            gnn: None,
        }
    }
}

/// Phase 2 (Thermodynamic Annealing) hyperparameters.
///
/// Controls simulated annealing parameters for GPU parallel tempering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase2Config {
    /// Number of annealing iterations
    pub iterations: usize,

    /// Number of temperature replicas
    pub replicas: usize,

    /// Minimum temperature
    pub temp_min: f32,

    /// Maximum temperature
    pub temp_max: f32,
}

impl Default for Phase2Config {
    fn default() -> Self {
        Self {
            iterations: 10000,
            replicas: 8,
            temp_min: 0.01,
            temp_max: 10.0,
        }
    }
}

/// Memetic Algorithm Configuration (Phase 7 Ensemble).
///
/// Hybrid CPU genetic algorithm for world-record optimization:
/// - GPU generates diverse initial population (multi-attempt pipeline runs)
/// - CPU evolves population via crossover, mutation, and local search
/// - Combines exploration (GPU randomness) with exploitation (CPU refinement)
///
/// ## Strategy
/// 1. **Initial Population**: Run pipeline N times with different seeds → N solutions
/// 2. **Selection**: Tournament or roulette wheel (fitness = chromatic number)
/// 3. **Crossover**: Recombine two parent colorings (vertex-wise or block-wise)
/// 4. **Mutation**: Random color changes (exploration)
/// 5. **Local Search**: Greedy conflict reduction (exploitation)
/// 6. **Elitism**: Preserve best solutions across generations
///
/// ## Performance Expectations (DSJC250.5)
/// - Population 128, Generations 500: ~350 hours (14.6 days)
/// - Expected progression: 41 → 38 → 36 → 34 → 32 colors
/// - Convergence: 50 generations without improvement triggers early stop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemeticConfig {
    /// Enable memetic algorithm
    #[serde(default = "default_false")]
    pub enabled: bool,

    /// Population size (number of solutions in gene pool)
    #[serde(default = "default_population_size")]
    pub population_size: usize,

    /// Number of evolutionary generations
    #[serde(default = "default_generations")]
    pub generations: usize,

    /// Crossover probability (0.0 - 1.0)
    #[serde(default = "default_crossover_rate")]
    pub crossover_rate: f64,

    /// Mutation probability per vertex (0.0 - 1.0)
    #[serde(default = "default_mutation_rate")]
    pub mutation_rate: f64,

    /// Local search iterations per offspring
    #[serde(default = "default_local_search_iterations")]
    pub local_search_iterations: usize,

    /// Number of elite solutions preserved
    #[serde(default = "default_elitism_count")]
    pub elitism_count: usize,

    /// Selection strategy: "tournament" or "roulette"
    #[serde(default = "default_selection_strategy")]
    pub selection_strategy: String,

    /// Tournament size (if using tournament selection)
    #[serde(default = "default_tournament_size")]
    pub tournament_size: usize,

    /// Maintain diversity (penalize duplicates)
    #[serde(default = "default_true")]
    pub maintain_diversity: bool,

    /// Early stop if no improvement for N generations
    #[serde(default = "default_convergence_threshold")]
    pub convergence_threshold: usize,
}

fn default_population_size() -> usize {
    128
}

fn default_generations() -> usize {
    500
}

fn default_crossover_rate() -> f64 {
    0.8
}

fn default_mutation_rate() -> f64 {
    0.05
}

fn default_local_search_iterations() -> usize {
    100
}

fn default_elitism_count() -> usize {
    10
}

fn default_selection_strategy() -> String {
    "tournament".to_string()
}

fn default_tournament_size() -> usize {
    5
}

fn default_convergence_threshold() -> usize {
    50
}

impl Default for MemeticConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            population_size: 128,
            generations: 500,
            crossover_rate: 0.8,
            mutation_rate: 0.05,
            local_search_iterations: 100,
            elitism_count: 10,
            selection_strategy: "tournament".to_string(),
            tournament_size: 5,
            maintain_diversity: true,
            convergence_threshold: 50,
        }
    }
}

/// Metaphysical Telemetry Coupling Configuration.
///
/// Enables feedback loop where geometric stress telemetry from Phase 4/6
/// influences all phases: Active Inference adjusts exploration, thermodynamic
/// adjusts temperature, FluxNet learns stress-responsive actions, and warmstart/
/// memetic phases prioritize stressed regions.
///
/// ## Algorithm
/// 1. Phase 4/6 compute geometry telemetry (stress_scalar, overlap_density, hotspots)
/// 2. Orchestrator propagates metrics to PhaseContext
/// 3. Phase 1: Adjusts exploration based on stress thresholds
/// 4. Phase 2: Modulates temperature: `temp *= (1.0 + alpha * stress_scalar)`
/// 5. Phase 0/7: Prioritize hotspot vertices for anchoring/mutation
/// 6. FluxNet: Learns geometry-responsive policies (8 new actions)
///
/// ## Configuration
/// - `enabled`: Master toggle for all coupling behavior
/// - `enable_early_phase_seeding`: Use Phase 0/1 signals for geometry before Phase 4 (default: true)
/// - `enable_reward_shaping`: Apply geometry stress bonuses to FluxNet rewards (default: true)
/// - `reward_shaping_scale`: Multiplier for geometry reward bonuses (default: 2.0)
/// - `stress_hot_threshold`: Activates moderate adjustments (default: 0.5)
/// - `stress_critical_threshold`: Activates aggressive adjustments (default: 0.8)
/// - `warmstart_bias_weight`: Hotspot anchoring boost factor (default: 2.0)
/// - `memetic_hotspot_boost`: Mutation rate multiplier for hotspots (default: 2.0)
/// - `phase1_exploration_boost`: Uncertainty multiplier at critical stress (default: 1.5)
/// - `phase2_temp_alpha`: Temperature scaling coefficient (default: 0.5)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaphysicalCouplingConfig {
    /// Enable metaphysical telemetry coupling
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Enable early-phase geometry seeding from Phase 0/1 proxy metrics.
    /// When true, Phase 1 computes synthetic geometry telemetry from uncertainty/difficulty
    /// before Phase 4/6 run, enabling earlier metaphysical coupling feedback.
    #[serde(default = "default_true")]
    pub enable_early_phase_seeding: bool,

    /// Enable FluxNet reward shaping based on geometry stress deltas.
    /// When true, RL controller receives bonus rewards when geometry stress decreases,
    /// encouraging actions that reduce geometric conflicts.
    #[serde(default = "default_true")]
    pub enable_reward_shaping: bool,

    /// Scaling factor for geometry reward bonuses (applied to stress delta).
    /// Higher values make geometry feedback more influential in RL learning.
    /// Default 2.0 balances geometry and outcome rewards.
    #[serde(default = "default_reward_shaping_scale")]
    pub reward_shaping_scale: f64,

    /// Stress threshold for moderate adjustments (0.0-1.0)
    #[serde(default = "default_stress_hot")]
    pub stress_hot_threshold: f32,

    /// Stress threshold for aggressive adjustments (0.0-1.0)
    #[serde(default = "default_stress_critical")]
    pub stress_critical_threshold: f32,

    /// Warmstart hotspot anchoring boost factor
    #[serde(default = "default_warmstart_bias")]
    pub warmstart_bias_weight: f32,

    /// Memetic mutation rate multiplier for hotspot vertices
    #[serde(default = "default_memetic_boost")]
    pub memetic_hotspot_boost: f32,

    /// Phase 1 exploration boost at critical stress
    #[serde(default = "default_phase1_boost")]
    pub phase1_exploration_boost: f32,

    /// Phase 2 temperature scaling coefficient
    #[serde(default = "default_phase2_alpha")]
    pub phase2_temp_alpha: f32,

    /// Minimum reward bonus magnitude for logging (default: 0.001)
    /// Logs appear when |geometry_bonus| > threshold OR when new best reward is achieved
    #[serde(default = "default_reward_log_threshold")]
    pub reward_log_threshold: f64,
}

fn default_stress_hot() -> f32 {
    0.5
}

fn default_stress_critical() -> f32 {
    0.8
}

fn default_warmstart_bias() -> f32 {
    2.0
}

fn default_memetic_boost() -> f32 {
    2.0
}

fn default_phase1_boost() -> f32 {
    1.5
}

fn default_phase2_alpha() -> f32 {
    0.5
}

fn default_reward_shaping_scale() -> f64 {
    2.0
}

fn default_reward_log_threshold() -> f64 {
    0.001
}

impl Default for MetaphysicalCouplingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_early_phase_seeding: true,
            enable_reward_shaping: true,
            reward_shaping_scale: 2.0,
            stress_hot_threshold: 0.5,
            stress_critical_threshold: 0.8,
            warmstart_bias_weight: 2.0,
            memetic_hotspot_boost: 2.0,
            phase1_exploration_boost: 1.5,
            phase2_temp_alpha: 0.5,
            reward_log_threshold: 0.001,
        }
    }
}

/// GPU acceleration configuration.
///
/// Implements PRISM GPU Plan §1: GPU Context Manager Configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,

    /// CUDA device ID (default: 0)
    pub device_id: usize,

    /// Directory containing PTX files
    pub ptx_dir: PathBuf,

    /// Allow NVRTC runtime compilation (default: false for production)
    pub allow_nvrtc: bool,

    /// Require PTX signature verification (default: false, set true in production)
    pub require_signed_ptx: bool,

    /// Directory with trusted PTX + .sha256 signatures
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trusted_ptx_dir: Option<PathBuf>,

    /// NVML polling interval in milliseconds (0 = disabled)
    pub nvml_poll_interval_ms: u64,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_id: 0,
            ptx_dir: PathBuf::from("target/ptx"),
            allow_nvrtc: false,
            require_signed_ptx: false,
            trusted_ptx_dir: None,
            nvml_poll_interval_ms: 1000,
        }
    }
}

/// Builder for PipelineConfig.
#[derive(Debug, Default)]
pub struct PipelineConfigBuilder {
    config: PipelineConfig,
}

impl PipelineConfigBuilder {
    pub fn max_vertices(mut self, max: usize) -> Self {
        self.config.max_vertices = max;
        self
    }

    pub fn timeout_seconds(mut self, timeout: u64) -> Self {
        self.config.timeout_seconds = timeout;
        self
    }

    pub fn add_phase(mut self, name: impl Into<String>, config: PhaseConfig) -> Self {
        self.config.phase_configs.insert(name.into(), config);
        self
    }

    pub fn telemetry_path(mut self, path: impl Into<String>) -> Self {
        self.config.telemetry_path = path.into();
        self
    }

    /// Enables warmstart with the given configuration.
    pub fn warmstart(mut self, config: WarmstartConfig) -> Self {
        self.config.warmstart_config = Some(config);
        self
    }

    /// Sets GPU configuration.
    pub fn gpu(mut self, config: GpuConfig) -> Self {
        self.config.gpu = config;
        self
    }

    /// Sets Phase 2 hyperparameters.
    pub fn phase2(mut self, config: Phase2Config) -> Self {
        self.config.phase2 = config;
        self
    }

    /// Enables memetic algorithm with the given configuration.
    pub fn memetic(mut self, config: MemeticConfig) -> Self {
        self.config.memetic = Some(config);
        self
    }

    /// Enables metaphysical telemetry coupling with the given configuration.
    pub fn metaphysical_coupling(mut self, config: MetaphysicalCouplingConfig) -> Self {
        self.config.metaphysical_coupling = Some(config);
        self
    }

    pub fn cma_es(mut self, config: CmaEsConfig) -> Self {
        self.config.cma_es = Some(config);
        self
    }

    /// Enables GNN inference with the given configuration.
    pub fn gnn(mut self, config: GnnConfig) -> Self {
        self.config.gnn = Some(config);
        self
    }

    pub fn build(self) -> Result<PipelineConfig, PrismError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let mut config = PipelineConfig::default();
        assert!(config.validate().is_ok());

        config.max_vertices = 0;
        assert!(config.validate().is_err());

        config.max_vertices = MAX_VERTICES + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = PipelineConfig::builder()
            .max_vertices(5000)
            .timeout_seconds(1800)
            .build()
            .unwrap();

        assert_eq!(config.max_vertices, 5000);
        assert_eq!(config.timeout_seconds, 1800);
    }

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert!(config.enabled);
        assert_eq!(config.device_id, 0);
        assert!(!config.allow_nvrtc);
        assert!(!config.require_signed_ptx);
        assert_eq!(config.nvml_poll_interval_ms, 1000);
    }
}

/// Ontology phase configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyConfig {
    /// Enable ontology phase
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Maximum depth for concept hierarchy traversal
    #[serde(default = "default_ontology_depth")]
    pub max_hierarchy_depth: usize,

    /// Semantic similarity threshold for concept matching
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,

    /// Enable GPU acceleration for graph operations
    #[serde(default = "default_false")]
    pub use_gpu: bool,

    /// Knowledge base path
    pub knowledge_base_path: Option<String>,
}

fn default_ontology_depth() -> usize {
    10
}

fn default_similarity_threshold() -> f32 {
    0.7
}

impl Default for OntologyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_hierarchy_depth: 10,
            similarity_threshold: 0.7,
            use_gpu: false,
            knowledge_base_path: None,
        }
    }
}

/// MEC (Molecular Emergent Computing) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MecConfig {
    /// Enable MEC phase
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Time step for molecular dynamics simulation
    #[serde(default = "default_mec_timestep")]
    pub time_step: f32,

    /// Number of simulation iterations
    #[serde(default = "default_mec_iterations")]
    pub iterations: usize,

    /// Temperature for molecular simulation (Kelvin)
    #[serde(default = "default_mec_temperature")]
    pub temperature: f32,

    /// Enable GPU acceleration
    #[serde(default = "default_false")]
    pub use_gpu: bool,

    /// Reaction rate constants
    #[serde(default)]
    pub reaction_rates: HashMap<String, f32>,
}

fn default_mec_timestep() -> f32 {
    1e-15 // 1 femtosecond
}

fn default_mec_iterations() -> usize {
    1000
}

fn default_mec_temperature() -> f32 {
    300.0 // Room temperature
}

impl Default for MecConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            time_step: 1e-15,
            iterations: 1000,
            temperature: 300.0,
            use_gpu: false,
            reaction_rates: HashMap::new(),
        }
    }
}

/// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmaEsConfig {
    /// Enable CMA-ES phase
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Population size
    #[serde(default = "default_cma_population")]
    pub population_size: usize,

    /// Initial step size (sigma)
    #[serde(default = "default_cma_sigma")]
    pub initial_sigma: f32,

    /// Maximum iterations
    #[serde(default = "default_cma_iterations")]
    pub max_iterations: usize,

    /// Target fitness value
    pub target_fitness: Option<f32>,

    /// Enable GPU acceleration
    #[serde(default = "default_false")]
    pub use_gpu: bool,
}

fn default_cma_population() -> usize {
    50
}

fn default_cma_sigma() -> f32 {
    0.5
}

fn default_cma_iterations() -> usize {
    1000
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            population_size: 50,
            initial_sigma: 0.5,
            max_iterations: 1000,
            target_fitness: None,
            use_gpu: false,
        }
    }
}

/// GNN (Graph Neural Network) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnConfig {
    /// Enable GNN processing
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Number of hidden dimensions
    #[serde(default = "default_gnn_hidden")]
    pub hidden_dim: usize,

    /// Number of layers
    #[serde(default = "default_gnn_layers")]
    pub num_layers: usize,

    /// Dropout rate
    #[serde(default = "default_gnn_dropout")]
    pub dropout: f32,

    /// Learning rate
    #[serde(default = "default_gnn_lr")]
    pub learning_rate: f32,

    /// GNN architecture type (e3_equivariant or onnx)
    #[serde(default = "default_gnn_architecture")]
    pub architecture: String,

    /// ONNX model path (optional, required if architecture is "onnx")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub onnx_model_path: Option<String>,

    /// Enable GPU acceleration
    #[serde(default = "default_false")]
    pub use_gpu: bool,
}

fn default_gnn_hidden() -> usize {
    128
}

fn default_gnn_layers() -> usize {
    3
}

fn default_gnn_dropout() -> f32 {
    0.5
}

fn default_gnn_lr() -> f32 {
    0.001
}

fn default_gnn_architecture() -> String {
    "e3_equivariant".to_string()
}

impl Default for GnnConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hidden_dim: 128,
            num_layers: 3,
            dropout: 0.5,
            learning_rate: 0.001,
            architecture: "e3_equivariant".to_string(),
            onnx_model_path: None,
            use_gpu: false,
        }
    }
}
