//! Training infrastructure for PRISM-LBS
//!
//! Includes PDBBind data loading, FluxNet weight optimization,
//! and ensemble model training.

pub mod conservation;
pub mod ensemble;
pub mod pdbbind_loader;
pub mod trainer;

pub use conservation::{ConservationConfig, ConservationData, ConservationLoader};
pub use ensemble::{EnsembleConfig, EnsemblePredictor, VotingMethod};
pub use pdbbind_loader::{PdbBindConfig, PdbBindEntry, PdbBindLoader};
pub use trainer::{LbsTrainer, TrainingConfig, TrainingMetrics};
