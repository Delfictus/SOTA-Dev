//! HybridSampler - Main Entry Point
//!
//! The public API for sampling. Downstream code uses this class ONLY.
//!
//! Features:
//! - Auto-selects backend based on structure size
//! - Can run shadow comparisons
//! - Supports migration stages
//! - Provides unified output

use anyhow::Result;

use crate::pdb_sanitizer::SanitizedStructure;
use crate::sampling::contract::SamplingBackend;
use crate::sampling::migration::{MigrationFlags, MigrationStage};
use crate::sampling::paths::{AmberPath, MockPath, NovaPath, NOVA_MAX_ATOMS};
use crate::sampling::result::{BackendId, SamplingConfig, SamplingResult};
use crate::sampling::shadow::{DivergenceMetrics, ShadowResult};

/// Routing strategy for backend selection
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Auto-select: Currently routes ALL structures to AmberMegaFused
    ///
    /// NOTE: NovaPath is deprecated for benchmarks due to:
    /// - Missing physics fixes (protonation, clamping, bond params)
    /// - ~100x slower performance
    /// - AmberMegaFused now handles all structure sizes with O(N) cell lists
    #[default]
    Auto,
    /// Force NOVA path (DEPRECATED - lacks physics fixes)
    ForceNova,
    /// Force AMBER path (recommended - full physics)
    ForceAmber,
    /// Use mock backend (for testing)
    Mock,
}

/// Main entry point for sampling
///
/// # Example
///
/// ```ignore
/// let mut sampler = HybridSampler::new_mock();
/// let backend = sampler.load_structure(&structure)?;
/// let result = sampler.sample(&SamplingConfig::default())?;
/// ```
pub struct HybridSampler {
    nova: NovaPath,
    amber: AmberPath,
    mock: MockPath,
    strategy: RoutingStrategy,
    flags: MigrationFlags,
    selected: Option<BackendId>,
    structure: Option<SanitizedStructure>,
}

impl HybridSampler {
    /// Create a new hybrid sampler with GPU context
    #[cfg(feature = "cryptic-gpu")]
    pub fn new(context: std::sync::Arc<cudarc::driver::CudaContext>) -> Result<Self> {
        Ok(Self {
            nova: NovaPath::new(std::sync::Arc::clone(&context))?,
            amber: AmberPath::new(context)?,
            mock: MockPath::new(),
            strategy: RoutingStrategy::Auto,
            flags: MigrationFlags::new(MigrationStage::Shadow),
            selected: None,
            structure: None,
        })
    }

    /// Create a mock hybrid sampler for testing (no GPU required)
    pub fn new_mock() -> Self {
        Self {
            nova: NovaPath::new_mock(),
            amber: AmberPath::new_mock(),
            mock: MockPath::new(),
            strategy: RoutingStrategy::Auto,
            flags: MigrationFlags::new(MigrationStage::StableOnly),
            selected: None,
            structure: None,
        }
    }

    /// Set routing strategy
    pub fn with_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set migration stage
    pub fn with_migration_stage(mut self, stage: MigrationStage) -> Self {
        self.flags = MigrationFlags::new(stage);
        self
    }

    /// Load structure and select backend
    ///
    /// Returns the selected backend ID.
    pub fn load_structure(&mut self, structure: &SanitizedStructure) -> Result<BackendId> {
        let backend = self.select_backend(structure);

        match backend {
            BackendId::Nova => self.nova.load_structure(structure)?,
            BackendId::AmberMegaFused => self.amber.load_structure(structure)?,
            BackendId::Mock => self.mock.load_structure(structure)?,
        }

        self.selected = Some(backend);
        self.structure = Some(structure.clone());

        log::info!(
            "HybridSampler: {} atoms -> {:?}",
            structure.n_atoms(),
            backend
        );

        Ok(backend)
    }

    /// Run sampling with selected backend
    pub fn sample(&mut self, config: &SamplingConfig) -> Result<SamplingResult> {
        let backend = self
            .selected
            .ok_or_else(|| anyhow::anyhow!("HybridSampler: No structure loaded"))?;

        match backend {
            BackendId::Nova => self.nova.sample(config),
            BackendId::AmberMegaFused => self.amber.sample(config),
            BackendId::Mock => self.mock.sample(config),
        }
    }

    /// Run shadow comparison (both backends, compare results)
    ///
    /// Only works for structures that fit in NOVA (≤512 atoms).
    pub fn sample_with_shadow(&mut self, config: &SamplingConfig) -> Result<ShadowResult> {
        let structure = self
            .structure
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No structure loaded"))?
            .clone();

        // Only run shadow if structure fits in NOVA
        if structure.n_atoms() > NOVA_MAX_ATOMS {
            let result = self.amber.sample(config)?;
            return Ok(ShadowResult::SkippedLargeStructure(result));
        }

        // Run NOVA
        self.nova.reset()?;
        self.nova.load_structure(&structure)?;
        let nova_result = self.nova.sample(config)?;

        // Run AMBER
        self.amber.reset()?;
        self.amber.load_structure(&structure)?;
        let amber_result = self.amber.sample(config)?;

        // Compare
        let divergence = compare_results(&nova_result, &amber_result);

        log::info!(
            "Shadow comparison: RMSD={:.2}A, energy_corr={:.3}",
            divergence.mean_rmsd,
            divergence.energy_correlation
        );

        Ok(ShadowResult::Compared {
            primary: nova_result,
            shadow: amber_result,
            divergence,
        })
    }

    /// Select backend based on strategy and structure
    ///
    /// NOTE: Auto mode now ALWAYS selects AmberMegaFused because:
    /// 1. AmberMegaFused has full physics (protonation, clamping, bond params)
    /// 2. NovaPath is ~100x slower and lacks these critical fixes
    /// 3. AmberMegaFused handles all structure sizes with O(N) cell lists
    fn select_backend(&self, structure: &SanitizedStructure) -> BackendId {
        match self.strategy {
            RoutingStrategy::Auto => {
                // DEPRECATED: NovaPath routing disabled
                // Previous: if structure.n_atoms() <= NOVA_MAX_ATOMS { Nova } else { Amber }
                // Now: Always use AmberMegaFused for correct physics
                let _ = structure.n_atoms(); // Silence unused warning
                let _ = NOVA_MAX_ATOMS; // Silence unused warning
                BackendId::AmberMegaFused
            }
            RoutingStrategy::ForceNova => BackendId::Nova,
            RoutingStrategy::ForceAmber => BackendId::AmberMegaFused,
            RoutingStrategy::Mock => BackendId::Mock,
        }
    }

    /// Get selected backend
    pub fn selected_backend(&self) -> Option<BackendId> {
        self.selected
    }

    /// Check if TDA is available for current structure
    pub fn has_tda(&self) -> bool {
        matches!(self.selected, Some(BackendId::Nova))
    }

    /// Reset sampler for reuse
    pub fn reset(&mut self) -> Result<()> {
        self.nova.reset()?;
        self.amber.reset()?;
        self.mock.reset()?;
        self.selected = None;
        self.structure = None;
        Ok(())
    }

    /// Get routing strategy
    pub fn strategy(&self) -> RoutingStrategy {
        self.strategy
    }

    /// Get migration flags
    pub fn migration_flags(&self) -> &MigrationFlags {
        &self.flags
    }
}

/// Compare two sampling results
fn compare_results(a: &SamplingResult, b: &SamplingResult) -> DivergenceMetrics {
    // Compute mean RMSD between conformations
    let mut total_rmsd = 0.0f32;
    let mut max_rmsd = 0.0f32;
    let n_comparisons = a.conformations.len().min(b.conformations.len());

    for i in 0..n_comparisons {
        let conf_a = &a.conformations[i];
        let conf_b = &b.conformations[i];

        if conf_a.len() != conf_b.len() {
            continue;
        }

        let rmsd = compute_rmsd(conf_a, conf_b);
        total_rmsd += rmsd;
        max_rmsd = max_rmsd.max(rmsd);
    }

    let mean_rmsd = if n_comparisons > 0 {
        total_rmsd / n_comparisons as f32
    } else {
        0.0
    };

    // Compute energy correlation
    let energy_correlation = compute_correlation(&a.energies, &b.energies);

    DivergenceMetrics {
        mean_rmsd,
        max_rmsd,
        energy_correlation,
    }
}

/// Compute RMSD between two conformations
fn compute_rmsd(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let sum_sq: f32 = a
        .iter()
        .zip(b)
        .map(|(pa, pb)| {
            let dx = pa[0] - pb[0];
            let dy = pa[1] - pb[1];
            let dz = pa[2] - pb[2];
            dx * dx + dy * dy + dz * dz
        })
        .sum();

    (sum_sq / a.len() as f32).sqrt()
}

/// Compute Pearson correlation
fn compute_correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.len() < 2 {
        return 0.0;
    }

    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;

    for (ai, bi) in a.iter().zip(b) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        cov / denom
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
    fn test_routing_auto_selects_amber() {
        // Test that Auto strategy correctly maps to AmberMegaFused
        let sampler = HybridSampler::new_mock();
        let structure = create_test_structure();

        // Call select_backend directly to test routing logic (doesn't require GPU)
        let backend = sampler.select_backend(&structure);
        // NOTE: Auto mode now ALWAYS routes to AmberMegaFused (NovaPath deprecated)
        assert_eq!(backend, BackendId::AmberMegaFused);
    }

    #[test]
    fn test_routing_strategy_mock() {
        let mut sampler = HybridSampler::new_mock().with_strategy(RoutingStrategy::Mock);
        let structure = create_test_structure();

        let backend = sampler.load_structure(&structure).unwrap();
        assert_eq!(backend, BackendId::Mock);
    }

    #[test]
    fn test_routing_strategy_force_amber_selection() {
        // Test that ForceAmber strategy correctly maps to AmberMegaFused
        let sampler = HybridSampler::new_mock().with_strategy(RoutingStrategy::ForceAmber);
        let structure = create_test_structure();

        // Call select_backend directly to test routing logic (doesn't require GPU)
        let backend = sampler.select_backend(&structure);
        assert_eq!(backend, BackendId::AmberMegaFused);
    }

    #[test]
    fn test_sample_mock() {
        let mut sampler = HybridSampler::new_mock().with_strategy(RoutingStrategy::Mock);
        let structure = create_test_structure();

        sampler.load_structure(&structure).unwrap();
        let result = sampler.sample(&SamplingConfig::quick()).unwrap();

        assert_eq!(result.n_samples(), 50);
    }

    #[test]
    fn test_has_tda_with_mock() {
        // Use Mock strategy to test has_tda (Mock returns false for TDA)
        let mut sampler = HybridSampler::new_mock().with_strategy(RoutingStrategy::Mock);
        let structure = create_test_structure();

        sampler.load_structure(&structure).unwrap();
        assert!(!sampler.has_tda()); // Mock doesn't have TDA
    }

    #[test]
    fn test_reset_with_mock() {
        let mut sampler = HybridSampler::new_mock().with_strategy(RoutingStrategy::Mock);
        let structure = create_test_structure();

        sampler.load_structure(&structure).unwrap();
        assert!(sampler.selected_backend().is_some());

        sampler.reset().unwrap();
        assert!(sampler.selected_backend().is_none());
    }

    #[test]
    fn test_compute_rmsd() {
        let a = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!((compute_rmsd(&a, &b) - 0.0).abs() < 1e-6);

        let c = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert!(compute_rmsd(&a, &c) > 0.0);
    }

    #[test]
    fn test_compute_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((compute_correlation(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((compute_correlation(&a, &c) - (-1.0)).abs() < 1e-6);
    }
}
