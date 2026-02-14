//! MegaFused Config and Params (Vendored)

use serde::{Serialize, Deserialize};

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct MegaFusedParams {
    pub contact_cutoff: f32,
    pub contact_sigma: f32,
    pub power_iterations: i32,
    pub kempe_iterations: i32,
    pub thresh_geometric: f32,
    pub thresh_conservation: f32,
    pub thresh_centrality: f32,
    pub thresh_flexibility: f32,
    pub min_signals: i32,
    pub consensus_threshold: f32,
    pub branch_weight_local: f32,
    pub branch_weight_neighbor: f32,
    pub branch_weight_global: f32,
    pub branch_weight_recurrent: f32,
    pub recurrent_decay: f32,
    pub consensus_weight_geometric: f32,
    pub consensus_weight_conservation: f32,
    pub consensus_weight_centrality: f32,
    pub consensus_weight_flexibility: f32,
    pub signal_bonus_0: f32,
    pub signal_bonus_1: f32,
    pub signal_bonus_2: f32,
    pub signal_bonus_3: f32,
    pub confidence_high_score: f32,
    pub confidence_medium_score: f32,
    pub confidence_high_signals: i32,
    pub confidence_medium_signals: i32,
    pub kempe_contact_threshold: f32,
    pub kempe_swap_threshold: f32,
    pub centrality_degree_weight: f32,
    pub centrality_eigenvector_weight: f32,
    pub min_pocket_volume: f32,
    pub max_pocket_volume: f32,
    pub min_druggability: f32,
    pub max_pocket_residues: i32,
    pub max_pockets: i32,
}

impl Default for MegaFusedParams {
    fn default() -> Self {
        Self {
            contact_cutoff: 12.0,
            contact_sigma: 6.0,
            power_iterations: 15,
            kempe_iterations: 10,
            thresh_geometric: 0.40,
            thresh_conservation: 0.50,
            thresh_centrality: 0.30,
            thresh_flexibility: 0.45,
            min_signals: 2,
            consensus_threshold: 0.35,
            branch_weight_local: 0.40,
            branch_weight_neighbor: 0.30,
            branch_weight_global: 0.20,
            branch_weight_recurrent: 0.10,
            recurrent_decay: 0.90,
            consensus_weight_geometric: 0.30,
            consensus_weight_conservation: 0.25,
            consensus_weight_centrality: 0.25,
            consensus_weight_flexibility: 0.20,
            signal_bonus_0: 0.70,
            signal_bonus_1: 1.00,
            signal_bonus_2: 1.15,
            signal_bonus_3: 1.30,
            confidence_high_score: 0.70,
            confidence_medium_score: 0.40,
            confidence_high_signals: 3,
            confidence_medium_signals: 2,
            kempe_contact_threshold: 0.20,
            kempe_swap_threshold: 1.10,
            centrality_degree_weight: 0.60,
            centrality_eigenvector_weight: 0.40,
            min_pocket_volume: 160.0,
            max_pocket_volume: 4800.0,
            min_druggability: 0.60,
            max_pocket_residues: 80,
            max_pockets: 10,
        }
    }
}

impl MegaFusedParams {
    pub fn from_config(config: &MegaFusedConfig) -> Self {
        let mut params = Self::default();
        params.contact_sigma = config.contact_sigma;
        params.consensus_threshold = config.consensus_threshold;
        params.power_iterations = config.power_iterations;
        params.kempe_iterations = config.kempe_iterations;
        params
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MegaFusedMode {
    UltraPrecise,
    #[default]
    Balanced,
    Screening,
}

#[derive(Debug, Clone)]
pub struct MegaFusedConfig {
    pub use_fp16: bool,
    pub contact_sigma: f32,
    pub consensus_threshold: f32,
    pub mode: MegaFusedMode,
    pub kempe_iterations: i32,
    pub power_iterations: i32,
    pub pure_gpu_mode: bool,
}

impl Default for MegaFusedConfig {
    fn default() -> Self {
        Self {
            use_fp16: true,
            contact_sigma: 6.0,
            consensus_threshold: 0.35,
            mode: MegaFusedMode::Balanced,
            kempe_iterations: 10,
            power_iterations: 15,
            pure_gpu_mode: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MegaFusedOutput {
    pub consensus_scores: Vec<f32>,
    pub confidence: Vec<i32>,
    pub signal_mask: Vec<i32>,
    pub pocket_assignment: Vec<i32>,
    pub centrality: Vec<f32>,
    pub combined_features: Vec<f32>,
}

// Telemetry structs omitted for brevity in vendored version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProvenanceData {} 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelTelemetryEvent {}
pub struct GpuTelemetry;
impl GpuTelemetry {
    pub fn new() -> Self { Self }
    pub fn get_clock_mhz(&self) -> Option<u32> { None }
    pub fn get_memory_clock_mhz(&self) -> Option<u32> { None }
    pub fn get_temperature(&self) -> Option<u32> { None }
    pub fn get_memory_used(&self) -> Option<u64> { None }
    pub fn get_gpu_name(&self) -> Option<String> { None }
    pub fn get_driver_version(&self) -> Option<String> { None }
}

pub mod confidence {
    pub const LOW: i32 = 0;
    pub const MEDIUM: i32 = 1;
    pub const HIGH: i32 = 2;
}

pub mod signals {
    pub const GEOMETRIC: i32 = 0x01;
    pub const CONSERVATION: i32 = 0x02;
    pub const CENTRALITY: i32 = 0x04;
    pub const FLEXIBILITY: i32 = 0x08;
}
