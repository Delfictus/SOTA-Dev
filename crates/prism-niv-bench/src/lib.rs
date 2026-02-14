//! NiV-Bench: Neuromorphic Benchmark for Cryptic Epitope Prediction
//!
//! A novel benchmark for PRISM>4D that demonstrates cryptic epitope prediction
//! on Nipah virus proteins using GPU-accelerated neuromorphic computing and
//! deep reinforcement learning.
//!
//! ## Key Features
//!
//! - **Cryptic Site Detection**: Uses 3 dedicated CUDA kernels for eigenmode analysis
//! - **FluxNet-DQN**: Dueling Deep Q-Network for continuous state space learning
//! - **Glycan Shield Modeling**: Stage 0 preprocessing for N-glycosylation masking
//! - **Dual-Stream GPU Pipeline**: Parallel execution to avoid register pressure
//! - **Cross-Reactivity Testing**: Paramyxovirus generalization (NiV + HeV)
//!
//! ## Architecture
//!
//! ```text
//! PDB Structures → Glycan Masking → GPU Pipeline → FluxNet-DQN → Predictions
//!     ↓               ↓               ↓              ↓            ↓
//! NiV/HeV       Stage 0        140-dim features   Q-learning   Cryptic sites
//! ```

pub mod error;
pub mod pdb_fetcher;
pub mod data_loader;
pub mod structure_types;
pub mod ground_truth;
pub mod glycan_mask;
pub mod glycan_dynamics;
pub mod nipah_platform_validation;
pub mod pimc_epitope_optimization;
pub mod qubo_tda_integration;
pub mod conservation_analysis;
// pub mod gpu_parallel;  // DISABLED: needs vendored types
// pub mod vendored;  // DISABLED: Use prism-gpu directly
// pub mod gpu_graph_pipeline;
// pub mod gpu_pipeline;
// pub mod ultra_benchmark;     // DISABLED: needs gpu_parallel
#[cfg(feature = "dqn")]
pub mod fluxnet_dqn;
// pub mod fluxnet_dqn_zero_copy;  // DISABLED: needs vendored types
// pub mod provenance_recorder;   // DISABLED: needs vendored types
// pub mod memory_proof_validator; // DISABLED: needs vendored types
// pub mod benchmark;  // DISABLED: needs fluxnet_niv
pub mod metrics;
pub mod baseline;
// pub mod visualize;  // DISABLED: needs fluxnet_niv
// pub mod report;  // DISABLED: needs benchmark
// pub mod fluxnet_niv;  // DISABLED: needs vendored types

// Use prism-gpu exports directly instead of vendored duplicates
pub use prism_gpu::{MegaFusedBatchGpu, BatchStructureDesc, StructureInput, PackedBatch};

pub use error::NivBenchError;
pub use structure_types::*;
pub use glycan_mask::GlycanMask;
// pub use gpu_parallel::ParallelGpuPipeline;  // DISABLED

/// Result type alias for NiV-Bench operations
pub type Result<T> = std::result::Result<T, NivBenchError>;

/// Current feature dimension count (will expand from 136 to 140 with cryptic features)
pub const CURRENT_FEATURE_DIM: usize = 136;
pub const TARGET_FEATURE_DIM: usize = 140;

/// Cryptic feature indices (F136-F139 when implemented)
pub const CRYPTIC_EIGENMODE_IDX: usize = 136;
pub const CRYPTIC_FLEXIBILITY_IDX: usize = 137;
pub const CRYPTIC_PROBE_IDX: usize = 138;
pub const CRYPTIC_COMBINED_IDX: usize = 139;