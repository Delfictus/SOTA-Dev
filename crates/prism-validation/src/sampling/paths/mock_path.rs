//! Mock Path - Testing Implementation
//!
//! Provides deterministic sampling output for testing.
//! No GPU required.

use anyhow::Result;

use crate::pdb_sanitizer::SanitizedStructure;
use crate::sampling::contract::SamplingBackend;
use crate::sampling::result::{
    BackendCapabilities, BackendId, SamplingConfig, SamplingMetadata, SamplingResult,
};

/// Mock Path for testing
///
/// Provides deterministic output without GPU.
/// Useful for:
/// - Unit tests
/// - Integration tests
/// - Pipeline validation
pub struct MockPath {
    structure: Option<SanitizedStructure>,
    /// Whether to include mock TDA data
    include_tda: bool,
}

impl MockPath {
    /// Create a new mock path
    pub fn new() -> Self {
        Self {
            structure: None,
            include_tda: false,
        }
    }

    /// Create mock path that includes TDA data
    pub fn with_tda() -> Self {
        Self {
            structure: None,
            include_tda: true,
        }
    }
}

impl Default for MockPath {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingBackend for MockPath {
    fn id(&self) -> BackendId {
        BackendId::Mock
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            tda: self.include_tda,
            active_inference: false,
            max_atoms: None,
            gpu_accelerated: false,
        }
    }

    fn load_structure(&mut self, structure: &SanitizedStructure) -> Result<()> {
        self.structure = Some(structure.clone());
        Ok(())
    }

    fn sample(&mut self, config: &SamplingConfig) -> Result<SamplingResult> {
        let structure = self
            .structure
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MockPath: No structure loaded"))?;

        let n_residues = structure.n_residues();
        let n_samples = config.n_samples;

        // Deterministic mock conformations
        let base_coords = structure.get_ca_coords();
        let conformations: Vec<Vec<[f32; 3]>> = (0..n_samples)
            .map(|i| {
                base_coords
                    .iter()
                    .map(|&[x, y, z]| {
                        let offset = (i as f32 * 0.01).sin() * 0.1;
                        [x + offset, y + offset, z + offset]
                    })
                    .collect()
            })
            .collect();

        // Deterministic mock energies
        let energies: Vec<f32> = (0..n_samples)
            .map(|i| -100.0 + (i as f32 * 0.1).sin() * 10.0)
            .collect();

        // Optional TDA
        let betti = if self.include_tda {
            Some((0..n_samples).map(|_| [1, 2, 0]).collect())
        } else {
            None
        };

        Ok(SamplingResult {
            conformations,
            energies,
            betti,
            metadata: SamplingMetadata {
                backend: BackendId::Mock,
                n_atoms: structure.n_atoms(),
                n_residues,
                n_samples,
                has_tda: self.include_tda,
                has_active_inference: false,
                elapsed_ms: 1,
                acceptance_rate: Some(1.0),
            },
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.structure = None;
        Ok(())
    }

    fn estimate_vram_mb(&self, _n_atoms: usize) -> f32 {
        0.0 // No GPU usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_structure() -> SanitizedStructure {
        use crate::pdb_sanitizer::sanitize_pdb;

        let pdb = r#"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C
END
"#;
        sanitize_pdb(pdb, "TEST").unwrap()
    }

    #[test]
    fn test_mock_path() {
        let mut path = MockPath::new();
        assert_eq!(path.id(), BackendId::Mock);

        let caps = path.capabilities();
        assert!(!caps.tda);
        assert!(!caps.gpu_accelerated);
    }

    #[test]
    fn test_mock_with_tda() {
        let path = MockPath::with_tda();
        assert!(path.capabilities().tda);
    }

    #[test]
    fn test_mock_sampling() {
        let mut path = MockPath::new();
        let structure = create_test_structure();

        path.load_structure(&structure).unwrap();

        let config = SamplingConfig::quick();
        let result = path.sample(&config).unwrap();

        assert_eq!(result.n_samples(), config.n_samples);
        assert!(!result.has_tda());
    }

    #[test]
    fn test_mock_deterministic() {
        let structure = create_test_structure();
        let config = SamplingConfig::quick();

        let mut path1 = MockPath::new();
        path1.load_structure(&structure).unwrap();
        let result1 = path1.sample(&config).unwrap();

        let mut path2 = MockPath::new();
        path2.load_structure(&structure).unwrap();
        let result2 = path2.sample(&config).unwrap();

        // Results should be identical
        assert_eq!(result1.energies, result2.energies);
    }
}
