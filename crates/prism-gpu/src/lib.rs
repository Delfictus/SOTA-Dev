//! # prism-gpu
//!
//! GPU acceleration layer for PRISM v2.
//! Optimized for PRISM-VE Benchmark.

pub mod context;
pub mod global_context;
pub mod feature_merge;
pub mod dendritic_reservoir;
pub mod dendritic_snn;
pub mod mega_fused;
pub mod mega_fused_batch;
pub mod reservoir_construction;
pub mod ve_swarm;
pub mod polycentric_immunity;
pub mod active_inference;
pub mod bio_chemistry_gpu;
pub mod prism_nova;

// AMBER ff14SB bonded force calculator
pub mod amber_forces;

// Mega-fused AMBER HMC dynamics (full GPU acceleration)
pub mod amber_mega_fused;

// PME (Particle Mesh Ewald) for long-range electrostatics
pub mod cufft_sys;
pub mod pme;

// SETTLE constraint solver for rigid water
pub mod settle;

// H-bond constraint solver for protein X-H bonds
pub mod h_constraints;

// SIMD Batched AMBER MD (Tier 1: 10-50x throughput, identical physics)
pub mod amber_simd_batch;

// SOTA Performance Optimizations
// Verlet neighbor lists: 2-3× speedup for non-bonded force computation
pub mod verlet_list;
// Tensor Core (WMMA) accelerated force computation: 2-4× speedup
pub mod tensor_core_forces;
// Async CUDA streams: 1.1-1.3× speedup from latency hiding
pub mod async_md_pipeline;

// Essential exports
pub use context::{GpuContext, GpuInfo, GpuSecurityConfig};
pub use global_context::{GlobalGpuContext, GlobalGpuError};
pub use feature_merge::{FeatureMergeGpu, FeatureMergeConfig, FeatureMergeResult};
pub use dendritic_reservoir::DendriticReservoirGpu;
pub use dendritic_snn::{DendriticSNNReservoir, INPUT_DIM as SNN_INPUT_DIM, EXPANDED_INPUT_DIM as SNN_EXPANDED_INPUT_DIM, DEFAULT_RESERVOIR_SIZE};
pub use mega_fused::{MegaFusedGpu, MegaFusedConfig, MegaFusedMode, MegaFusedOutput, MegaFusedParams, GpuProvenanceData, KernelTelemetryEvent, GpuTelemetry, confidence, signals};
pub use mega_fused_batch::{MegaFusedBatchGpu, BatchStructureDesc, StructureInput, StructureMetadata, PkParams, ImmunityMetadataV2, CountryImmunityTimeSeriesV2, PackedBatch, BatchStructureOutput, BatchOutput, TrainingOutput};
pub use reservoir_construction::{BioReservoir, SparseConnection, compute_readout_weights};
pub use polycentric_immunity::{PolycentricImmunityGpu, N_EPITOPE_CENTERS, N_PK_SCENARIOS, POLYCENTRIC_OUTPUT_DIM, DEFAULT_CROSS_REACTIVITY};
pub use active_inference::{ActiveInferenceGpu, ActiveInferencePolicy};
pub use bio_chemistry_gpu::{BiochemistryGpu, GpuAtomicMetadata, MAX_ATOMS as BIO_MAX_ATOMS};
pub use prism_nova::{PrismNova, NovaConfig, NovaStepResult, RESERVOIR_SIZE as NOVA_RESERVOIR_SIZE};
pub use amber_forces::{AmberBondedForces, TopologyBuilder, BondParam, AngleParam, DihedralParam, NB14Param, EnergyComponents, Bond, Angle, Dihedral, Pair14};
pub use amber_mega_fused::{
    AmberMegaFusedHmc, HmcRunResult, EnergyRecord, ConstraintInfo,
    KB_KCAL_MOL_K, build_exclusion_lists as build_amber_exclusions,
    MAX_EXCLUSIONS as AMBER_MAX_EXCLUSIONS,
    // Phase 7: Mixed precision exports
    MixedPrecisionConfig, MixedPrecisionBuffers,
    f32_to_f16_bits, f16_bits_to_f32,
};
pub use pme::{PME, compute_ewald_beta, DEFAULT_PME_TOLERANCE};
pub use settle::Settle;
pub use h_constraints::{HConstraints, HConstraintCluster, ClusterType, build_h_clusters};
pub use amber_simd_batch::{
    AmberSimdBatch, StructureTopology, BatchMdResult, OptimizationConfig, SotaStats,
    BATCH_SPATIAL_OFFSET, MAX_BATCH_SIZE, NB_CUTOFF,
};
pub use verlet_list::{
    VerletList, VERLET_SKIN, VERLET_SKIN_HALF, VERLET_LIST_CUTOFF, MAX_NEIGHBORS_PER_ATOM,
};
pub use tensor_core_forces::{
    TensorCoreForces, TC_TILE_SIZE, TC_BLOCK_SIZE,
};
pub use async_md_pipeline::{
    AsyncMdPipeline, AsyncPipelineConfig, PipelineStats, MdPhase, PipelineExecutor, SyncPoint,
};
pub use memory::{VramGuard, VramInfo, VramGuardError, init_global_vram_guard, global_vram_guard};
pub use whcr::{WhcrGpu, RepairResult as WhcrRepairResult};

// Commented out unused modules to isolate benchmark requirements
// pub mod aatgs;
// pub mod aatgs_integration;
// pub mod cma;
// pub mod cma_es;
// pub mod cryptic_gpu;
// pub mod glycan_gpu;
// pub mod dendritic_whcr;
// pub mod floyd_warshall;
// pub mod lbs;
// pub mod readout_training;
// pub mod molecular;
// pub mod multi_device_pool;
// pub mod multi_gpu;
// pub mod multi_gpu_integration;
// pub mod pimc;
// pub mod quantum;
pub mod stream_integration;
pub mod stream_manager;

// VRAM Safety Guard - Battle-tested memory management
pub mod memory;
// pub mod tda;
// pub mod thermodynamic;
// pub mod transfer_entropy;
// pub mod ultra_kernel;
pub mod whcr;  // Re-enabled for prism-whcr dependency
// pub mod batch_tda;
// pub mod mega_fused_integrated;
// pub mod training;