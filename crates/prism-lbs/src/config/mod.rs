//! Configuration module for PRISM-LBS unified hybrid detection
//!
//! Provides configurable parameters for:
//! - Detection quality modes (Turbo, Standard, Publication)
//! - Benchmark output formats (LIGYSIS, CryptoBench)
//! - Provenance tracking levels
//! - GPU optimization settings

pub mod unified_hybrid;

// Re-export main configuration types
pub use unified_hybrid::{
    UnifiedHybridConfig,
    DetectionMode,
    ProvenanceLevel,
    DruggabilityWeights,
    QualityPreset,
};
