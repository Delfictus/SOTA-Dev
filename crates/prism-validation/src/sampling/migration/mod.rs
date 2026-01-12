//! Migration Control (Strangler Pattern)
//!
//! Controls gradual rollout from stable (AMBER) to greenfield (NOVA) path.

mod feature_flags;

pub use feature_flags::{MigrationFlags, MigrationStage};
