//! Sampling Backend Contract
//!
//! THE LAW: All sampling backends must implement the `SamplingBackend` trait.
//!
//! # Stability Guarantee
//!
//! This trait signature is **FROZEN** after Phase 6 release.
//! Phase 7-8 enhancements **MUST NOT** modify this trait.
//! Add extension traits if new capabilities are needed.
//!
//! # Version
//!
//! Contract version 1.0.0 - DO NOT MODIFY AFTER PHASE 6 RELEASE

use anyhow::Result;

use super::result::{BackendCapabilities, BackendId, SamplingConfig, SamplingResult};
use crate::pdb_sanitizer::SanitizedStructure;

/// Contract version - increment only for breaking changes (which should never happen)
pub const CONTRACT_VERSION: &str = "1.0.0";

/// THE CONTRACT: All sampling backends must implement this trait
///
/// # Stability Guarantee
///
/// This trait signature is FROZEN after Phase 6 release.
/// Phase 7-8 enhancements MUST NOT modify this trait.
/// Add extension traits if new capabilities needed.
///
/// # Implementor Requirements
///
/// 1. `id()` must return stable, unique BackendId
/// 2. `capabilities()` must accurately reflect supported features
/// 3. `load_structure()` must validate against capabilities.max_atoms
/// 4. `sample()` output must conform to SamplingResult schema
/// 5. `sample()` must be deterministic given same seed
/// 6. Errors must be descriptive and include backend ID
pub trait SamplingBackend: Send + Sync {
    /// Unique identifier for this backend
    fn id(&self) -> BackendId;

    /// What this backend supports
    fn capabilities(&self) -> BackendCapabilities;

    /// Load and validate structure
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Structure exceeds `capabilities().max_atoms`
    /// - Structure is invalid for this backend
    fn load_structure(&mut self, structure: &SanitizedStructure) -> Result<()>;

    /// Run sampling
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No structure loaded
    /// - GPU error during sampling
    /// - Numerical instability detected
    fn sample(&mut self, config: &SamplingConfig) -> Result<SamplingResult>;

    /// Reset for reuse with new structure
    fn reset(&mut self) -> Result<()>;

    /// Estimate VRAM usage in megabytes
    fn estimate_vram_mb(&self, n_atoms: usize) -> f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_version() {
        assert_eq!(CONTRACT_VERSION, "1.0.0");
    }
}
