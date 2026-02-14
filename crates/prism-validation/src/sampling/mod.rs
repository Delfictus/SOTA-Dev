//! Parallel Sampling Implementation
//!
//! This module provides the hybrid sampling infrastructure for PRISM cryptic site detection.
//! It supports parallel NOVA (greenfield) and AMBER (stable) paths with shadow comparison.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    PUBLIC API: HybridSampler                        │
//! │  • load_structure(pdb) -> BackendId                                 │
//! │  • sample(config) -> SamplingResult                                 │
//! │  • Downstream code uses this ONLY                                   │
//! └─────────────────────────────────────────────────────────────────────┘
//!                                    │
//!                    ┌───────────────┼───────────────┐
//!                    ▼               ▼               ▼
//!        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
//!        │   NOVA PATH     │ │   AMBER PATH    │ │  SHADOW MODE    │
//!        │   (Greenfield)  │ │   (Stable)      │ │  (Comparison)   │
//!        │ • ≤512 atoms    │ │ • No limit      │ │ • Runs both     │
//!        │ • TDA + AI      │ │ • Proven MD     │ │ • Compares      │
//!        └─────────────────┘ └─────────────────┘ └─────────────────┘
//! ```
//!
//! # Module Structure
//!
//! - [`contract`]: The `SamplingBackend` trait (THE LAW - frozen after release)
//! - [`result`]: Unified result types (`SamplingResult`, `SamplingConfig`)
//! - [`paths`]: Parallel implementations (NOVA greenfield, AMBER stable)
//! - [`router`]: Routing layer and `HybridSampler` entry point
//! - [`shadow`]: Shadow pipeline for comparing outputs
//! - [`migration`]: Strangler pattern support and feature flags
//!
//! # Usage
//!
//! ```ignore
//! use prism_validation::sampling::{HybridSampler, SamplingConfig};
//! use prism_validation::pdb_sanitizer::sanitize_pdb;
//!
//! // Sanitize structure first
//! let structure = sanitize_pdb(&pdb_content, "1AKE")?;
//!
//! // Create sampler
//! let mut sampler = HybridSampler::new(context)?;
//!
//! // Load structure (auto-selects backend based on size)
//! let backend = sampler.load_structure(&structure)?;
//! println!("Using backend: {:?}", backend);
//!
//! // Sample
//! let result = sampler.sample(&SamplingConfig::default())?;
//! println!("Generated {} conformations", result.conformations.len());
//! ```
//!
//! # Phase Evolution
//!
//! - **Phase 6**: Establish parallel paths, shadow comparison
//! - **Phase 7**: Enhance NOVA with persistent homology, adaptive sampling
//! - **Phase 8**: Ensemble voting, transfer learning
//!
//! The AMBER path remains FROZEN after Phase 6 release.

// Core contract (THE LAW)
pub mod contract;

// Result types
pub mod result;

// Parallel path implementations
pub mod paths;

// Routing layer
pub mod router;

// Shadow pipeline
pub mod shadow;

// Migration control
pub mod migration;

// Re-export key types for convenience
pub use contract::{SamplingBackend, CONTRACT_VERSION};
pub use result::{
    BackendCapabilities, BackendId, SamplingConfig, SamplingMetadata, SamplingResult,
};

// Re-export router types
pub use router::{HybridSampler, RoutingStrategy};

// Re-export path types (for testing/debugging only)
pub use paths::{AmberPath, NovaPath, NOVA_MAX_ATOMS};

// Re-export migration types
pub use migration::{MigrationFlags, MigrationStage};

// Re-export shadow types
pub use shadow::{DivergenceMetrics, DivergenceVerdict, ShadowComparator, ShadowResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all key types are exported
        let _: BackendId = BackendId::Nova;
        let _: BackendId = BackendId::AmberMegaFused;
        let _: BackendId = BackendId::Mock;

        let config = SamplingConfig::default();
        assert_eq!(config.n_samples, 500);
        assert_eq!(config.temperature, 310.0);

        assert_eq!(CONTRACT_VERSION, "1.0.0");
        assert_eq!(NOVA_MAX_ATOMS, 512);
    }

    #[test]
    fn test_backend_capabilities_default() {
        let caps = BackendCapabilities::default();
        assert!(!caps.tda);
        assert!(!caps.active_inference);
        assert!(caps.max_atoms.is_none());
        assert!(!caps.gpu_accelerated);
    }

    #[test]
    fn test_routing_strategy_default() {
        let strategy = RoutingStrategy::default();
        assert!(matches!(strategy, RoutingStrategy::Auto));
    }

    #[test]
    fn test_migration_stage_transitions() {
        let mut flags = MigrationFlags::new(MigrationStage::StableOnly);
        assert_eq!(flags.stage(), MigrationStage::StableOnly);
        assert!(!flags.use_greenfield());

        assert!(flags.advance());
        assert_eq!(flags.stage(), MigrationStage::Shadow);
        assert!(!flags.use_greenfield());
        assert!(flags.run_shadow());
    }
}
