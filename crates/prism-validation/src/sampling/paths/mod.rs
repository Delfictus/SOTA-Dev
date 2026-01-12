//! Parallel Path Implementations
//!
//! This module contains the parallel sampling backends:
//!
//! - [`NovaPath`]: Greenfield implementation with TDA + Active Inference (≤512 atoms)
//! - [`AmberPath`]: Stable implementation with proven AMBER MD (no limit)
//! - [`MockPath`]: Testing implementation with deterministic output
//!
//! # Isolation Rules
//!
//! **CRITICAL**: These paths are isolated from each other:
//!
//! - `nova_path.rs` MUST NEVER import from `amber_path.rs`
//! - `amber_path.rs` MUST NEVER import from `nova_path.rs`
//! - Both MUST implement `SamplingBackend` trait exactly
//!
//! # Evolution
//!
//! - **NOVA Path**: Evolves through Phase 7-8 with enhanced capabilities
//! - **AMBER Path**: FROZEN after Phase 6 release - serves as stable reference

mod nova_path;
mod amber_path;
mod mock_path;

pub use nova_path::{NovaPath, NOVA_MAX_ATOMS};
pub use amber_path::AmberPath;
pub use mock_path::MockPath;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::contract::SamplingBackend;
    use crate::sampling::result::BackendId;

    #[test]
    fn test_nova_path_id() {
        let path = NovaPath::new_mock();
        assert_eq!(path.id(), BackendId::Nova);
    }

    #[test]
    fn test_amber_path_id() {
        let path = AmberPath::new_mock();
        assert_eq!(path.id(), BackendId::AmberMegaFused);
    }

    #[test]
    fn test_mock_path_id() {
        let path = MockPath::new();
        assert_eq!(path.id(), BackendId::Mock);
    }

    #[test]
    fn test_nova_max_atoms_constant() {
        assert_eq!(NOVA_MAX_ATOMS, 512);
    }

    #[test]
    fn test_capabilities_match_expectations() {
        let nova = NovaPath::new_mock();
        let caps = nova.capabilities();
        assert!(caps.tda);
        assert!(caps.active_inference);
        assert_eq!(caps.max_atoms, Some(512));

        let amber = AmberPath::new_mock();
        let caps = amber.capabilities();
        assert!(!caps.tda);
        assert!(!caps.active_inference);
        assert!(caps.max_atoms.is_none());
    }
}
