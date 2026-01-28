//! Phase 0: Dendritic Reservoir module
//!
//! Implements neuromorphic dendritic computing with warmstart prior generation.

pub mod controller;
pub mod ensemble;
pub mod warmstart;

pub use controller::{Phase0Config, Phase0DendriticReservoir};
pub use ensemble::{apply_anchors, fuse_ensemble_priors};
pub use warmstart::{build_reservoir_prior, fuse_priors};
