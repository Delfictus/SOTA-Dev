//! # prism-pipeline
//!
//! Pipeline orchestration, telemetry, and configuration for PRISM v2.
//!
//! This crate provides the execution engine that coordinates all phases,
//! integrates with FluxNet RL, and emits telemetry.

pub mod adp;
pub mod config;
pub mod orchestrator;
pub mod profiler;
pub mod telemetry;

// Re-export commonly used items
pub use adp::AdpWarmstartAdjuster;
pub use config::{
    GnnConfig, MemeticConfig, MetaphysicalCouplingConfig, Phase2Config, PipelineConfig,
};
pub use orchestrator::PipelineOrchestrator;
pub use profiler::PerformanceProfiler;
pub use telemetry::TelemetryEvent;
