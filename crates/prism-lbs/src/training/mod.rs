//! Training infrastructure for PRISM-LBS
//!
//! Includes PDBBind data loading, FluxNet weight optimization,
//! and ensemble model training.

pub mod pdbbind_loader;
pub mod conservation;
pub mod ensemble;
pub mod trainer;

pub use pdbbind_loader::{PdbBindLoader, PdbBindEntry, PdbBindConfig};
pub use conservation::{ConservationLoader, ConservationData, ConservationConfig};
pub use ensemble::{EnsemblePredictor, EnsembleConfig, VotingMethod};
pub use trainer::{LbsTrainer, TrainingConfig, TrainingMetrics};
