//! # prism-phases
//!
//! Phase controller implementations for PRISM v2.
//!
//! Implements all 7 phases with FluxNet RL integration.

pub mod conflict_repair;
pub mod dendritic_reservoir_whcr;
pub mod phase0;
pub mod phase1_active_inference;
pub mod phase2_thermodynamic;
pub mod phase3_quantum;
pub mod phase4_geodesic;
pub mod phase6_tda;
pub mod phase7_ensemble;
pub mod phase_whcr;
pub mod wavelet_conflict_repair;

// Re-export phase controllers
pub use phase0::Phase0DendriticReservoir;
pub use phase1_active_inference::Phase1ActiveInference;
pub use phase2_thermodynamic::Phase2Thermodynamic;
pub use phase3_quantum::Phase3Quantum;
pub use phase4_geodesic::Phase4Geodesic;
pub use phase6_tda::Phase6TDA;
pub use phase7_ensemble::{MemeticAlgorithm, Phase7Config, Phase7Ensemble};
pub use phase_whcr::WHCRPhaseController;
