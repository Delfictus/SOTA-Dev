# PRISM Parallel Implementation Architecture
## Phase 6 Foundation for Phase 7-8 Enhancement Path

**Document Version**: 1.0  
**Created**: 2026-01-12  
**Applies To**: Phase 6 Implementation (Required Foundation)  
**Purpose**: Establish parallel paths and shadow pipeline infrastructure  

---

## Overview

This document specifies the **parallel implementation architecture** that MUST be implemented during Phase 6 to support the Phase 7-8 enhancement trajectory. This architecture enables:

1. **Safe parallel development** - NOVA (greenfield) evolves while AMBER (stable) remains unchanged
2. **Shadow validation** - Compare greenfield against stable before promotion
3. **Gradual migration** - Strangler pattern for controlled rollout
4. **Automatic rollback** - Revert to stable on failures

**CRITICAL**: This architecture is NOT optional. It is required infrastructure for Phase 7-8 success.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRISM SAMPLING ARCHITECTURE                              │
│                    Parallel Implementation with Shadow Pipeline              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PUBLIC API: HybridSampler                        │   │
│  │  • load_structure(pdb) -> BackendId                                 │   │
│  │  • sample(config) -> SamplingResult                                 │   │
│  │  • Downstream code uses this ONLY                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                    ┌───────────────┼───────────────┐                       │
│                    │               │               │                       │
│                    ▼               ▼               ▼                       │
│        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│        │   NOVA PATH     │ │   AMBER PATH    │ │  SHADOW MODE    │        │
│        │   (Greenfield)  │ │   (Stable)      │ │  (Comparison)   │        │
│        │                 │ │                 │ │                 │        │
│        │ • ≤512 atoms    │ │ • No limit      │ │ • Runs both     │        │
│        │ • TDA + AI      │ │ • Proven MD     │ │ • Compares      │        │
│        │ • Evolves Ph7-8 │ │ • Never changes │ │ • Logs diffs    │        │
│        └─────────────────┘ └─────────────────┘ └─────────────────┘        │
│                    │               │               │                       │
│                    └───────────────┼───────────────┘                       │
│                                    ▼                                        │
│        ┌─────────────────────────────────────────────────────────────┐     │
│        │              UNIFIED OUTPUT: SamplingResult                 │     │
│        │  • Same format from all paths                               │     │
│        │  • Downstream processing is path-agnostic                   │     │
│        └─────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure (Phase 6)

```
crates/prism-validation/src/sampling/
├── mod.rs                    # Public API exports
├── contract.rs               # SamplingBackend trait (THE LAW)
├── result.rs                 # SamplingResult, SamplingConfig
│
├── paths/                    # PARALLEL IMPLEMENTATIONS
│   ├── mod.rs               # Path exports
│   ├── nova_path.rs         # Greenfield: TDA + Active Inference (≤512 atoms)
│   ├── amber_path.rs        # Stable: Proven AMBER MD (no limit)
│   └── mock_path.rs         # Testing: Deterministic fake
│
├── router/                   # ROUTING LAYER
│   ├── mod.rs               # Router exports
│   ├── hybrid_sampler.rs    # Main entry point
│   ├── auto_router.rs       # Size-based backend selection
│   └── strategy.rs          # Routing strategies
│
├── shadow/                   # SHADOW PIPELINE
│   ├── mod.rs               # Shadow exports
│   ├── comparator.rs        # Run both paths, compare
│   ├── divergence.rs        # Divergence metrics
│   └── log.rs               # Divergence logging
│
└── migration/                # STRANGLER PATTERN SUPPORT
    ├── mod.rs               # Migration exports
    ├── feature_flags.rs     # Stage control
    └── rollback.rs          # Automatic rollback
```

---

## Core Components

### 1. The Contract (contract.rs)

**Purpose**: Define the interface ALL backends must implement. This NEVER changes.

```rust
//! sampling/contract.rs
//!
//! THE LAW: All sampling backends must implement this trait.
//! Version: 1.0.0 - DO NOT MODIFY AFTER PHASE 6 RELEASE

use anyhow::Result;
use crate::pdb_sanitizer::SanitizedStructure;
use super::result::{SamplingResult, SamplingConfig, BackendCapabilities, BackendId};

/// Contract version - increment only for breaking changes (which should never happen)
pub const CONTRACT_VERSION: &str = "1.0.0";

/// THE CONTRACT: All sampling backends must implement this trait
///
/// # Stability Guarantee
/// This trait signature is FROZEN after Phase 6 release.
/// Phase 7-8 enhancements MUST NOT modify this trait.
/// Add extension traits if new capabilities needed.
///
/// # Implementor Requirements
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
    fn load_structure(&mut self, structure: &SanitizedStructure) -> Result<()>;
    
    /// Run sampling
    fn sample(&mut self, config: &SamplingConfig) -> Result<SamplingResult>;
    
    /// Reset for reuse
    fn reset(&mut self) -> Result<()>;
    
    /// Estimate VRAM usage
    fn estimate_vram_mb(&self, n_atoms: usize) -> f32;
}
```

### 2. Result Types (result.rs)

**Purpose**: Define unified output format that all paths produce.

```rust
//! sampling/result.rs
//!
//! Unified result types for all sampling backends.
//! STABLE: Do not add required fields, only optional ones.

/// Backend identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendId {
    Nova,
    AmberMegaFused,
    Mock,
}

/// Backend capabilities
#[derive(Debug, Clone, Copy, Default)]
pub struct BackendCapabilities {
    pub tda: bool,
    pub active_inference: bool,
    pub max_atoms: Option<usize>,
    pub gpu_accelerated: bool,
}

/// Sampling configuration (backend-agnostic)
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub n_samples: usize,
    pub steps_per_sample: usize,
    pub temperature: f32,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            n_samples: 500,
            steps_per_sample: 100,
            temperature: 310.0,
            seed: 42,
        }
    }
}

/// Unified sampling result
///
/// STABILITY: All fields marked as Option can be added in future phases.
/// Required fields (conformations, energies, metadata) are FROZEN.
#[derive(Debug, Clone)]
pub struct SamplingResult {
    /// Cα coordinates per sample [n_samples][n_residues][3]
    pub conformations: Vec<Vec<[f32; 3]>>,
    
    /// Energy per sample
    pub energies: Vec<f32>,
    
    /// Betti numbers (only from TDA-capable backends)
    pub betti: Option<Vec<[i32; 3]>>,
    
    /// Metadata about the run
    pub metadata: SamplingMetadata,
    
    // PHASE 7+ ADDITIONS (optional, backward compatible)
    // pub persistence_diagrams: Option<Vec<PersistenceDiagram>>,
    // pub active_inference_scores: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct SamplingMetadata {
    pub backend: BackendId,
    pub n_atoms: usize,
    pub n_residues: usize,
    pub n_samples: usize,
    pub has_tda: bool,
    pub has_active_inference: bool,
    pub elapsed_ms: u64,
}
```

### 3. NOVA Path (paths/nova_path.rs)

**Purpose**: Greenfield implementation with TDA + Active Inference. Evolves through phases.

```rust
//! sampling/paths/nova_path.rs
//!
//! NOVA Path - Greenfield Implementation
//!
//! STATUS: Greenfield (evolves through Phase 7-8)
//! CAPABILITIES: TDA + Active Inference
//! LIMITATION: ≤512 atoms (shared memory constraint)
//!
//! PHASE EVOLUTION:
//! - Phase 6: Basic TDA, 500 samples
//! - Phase 7: Persistent homology, 2000 samples, adaptive bias
//! - Phase 8: Part of ensemble

use super::super::contract::SamplingBackend;
use super::super::result::*;
use crate::pdb_sanitizer::SanitizedStructure;
use anyhow::{Result, bail, Context};
use std::sync::Arc;
use cudarc::driver::CudaContext;

/// NOVA atom limit
pub const NOVA_MAX_ATOMS: usize = 512;

pub struct NovaPath {
    context: Arc<CudaContext>,
    nova: Option<prism_gpu::PrismNova>,
    structure: Option<SanitizedStructure>,
}

impl NovaPath {
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            context,
            nova: None,
            structure: None,
        })
    }
}

impl SamplingBackend for NovaPath {
    fn id(&self) -> BackendId {
        BackendId::Nova
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            tda: true,
            active_inference: true,
            max_atoms: Some(NOVA_MAX_ATOMS),
            gpu_accelerated: true,
        }
    }
    
    fn load_structure(&mut self, structure: &SanitizedStructure) -> Result<()> {
        if structure.n_atoms > NOVA_MAX_ATOMS {
            bail!(
                "NovaPath: {} atoms exceeds limit of {}",
                structure.n_atoms, NOVA_MAX_ATOMS
            );
        }
        
        // Initialize NOVA...
        self.structure = Some(structure.clone());
        Ok(())
    }
    
    fn sample(&mut self, config: &SamplingConfig) -> Result<SamplingResult> {
        let structure = self.structure.as_ref()
            .ok_or_else(|| anyhow::anyhow!("NovaPath: No structure loaded"))?;
        
        // Run sampling...
        // Return with betti: Some(...) because capabilities.tda == true
        
        todo!("Implement NOVA sampling")
    }
    
    fn reset(&mut self) -> Result<()> {
        self.structure = None;
        Ok(())
    }
    
    fn estimate_vram_mb(&self, n_atoms: usize) -> f32 {
        50.0 + (n_atoms as f32 * 0.5)
    }
}
```

### 4. AMBER Path (paths/amber_path.rs)

**Purpose**: Stable reference implementation. NEVER changes after Phase 6.

```rust
//! sampling/paths/amber_path.rs
//!
//! AMBER Path - Stable Implementation
//!
//! STATUS: STABLE - DO NOT MODIFY AFTER PHASE 6 RELEASE
//! CAPABILITIES: Proven AMBER ff14SB molecular dynamics
//! LIMITATION: None (O(N) cell lists)
//!
//! This path serves as the stable reference for shadow comparison.
//! Phase 7-8 enhancements DO NOT touch this file.

use super::super::contract::SamplingBackend;
use super::super::result::*;
use crate::pdb_sanitizer::SanitizedStructure;
use anyhow::{Result, Context};
use std::sync::Arc;
use cudarc::driver::CudaContext;

pub struct AmberPath {
    context: Arc<CudaContext>,
    amber: Option<prism_gpu::AmberMegaFusedHmc>,
    structure: Option<SanitizedStructure>,
}

impl AmberPath {
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            context,
            amber: None,
            structure: None,
        })
    }
}

impl SamplingBackend for AmberPath {
    fn id(&self) -> BackendId {
        BackendId::AmberMegaFused
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            tda: false,              // AMBER doesn't have TDA
            active_inference: false, // AMBER doesn't have AI
            max_atoms: None,         // No limit
            gpu_accelerated: true,
        }
    }
    
    fn load_structure(&mut self, structure: &SanitizedStructure) -> Result<()> {
        // No atom limit check - AMBER handles any size
        self.structure = Some(structure.clone());
        Ok(())
    }
    
    fn sample(&mut self, config: &SamplingConfig) -> Result<SamplingResult> {
        let structure = self.structure.as_ref()
            .ok_or_else(|| anyhow::anyhow!("AmberPath: No structure loaded"))?;
        
        // Run sampling...
        // Return with betti: None because capabilities.tda == false
        
        todo!("Implement AMBER sampling")
    }
    
    fn reset(&mut self) -> Result<()> {
        self.structure = None;
        Ok(())
    }
    
    fn estimate_vram_mb(&self, n_atoms: usize) -> f32 {
        30.0 + (n_atoms as f32 * 0.2)
    }
}
```

### 5. Hybrid Sampler (router/hybrid_sampler.rs)

**Purpose**: Main entry point that routes to appropriate backend.

```rust
//! sampling/router/hybrid_sampler.rs
//!
//! HybridSampler - The public API for sampling
//!
//! Downstream code uses this class ONLY. It:
//! 1. Auto-selects backend based on structure size
//! 2. Can run shadow comparisons
//! 3. Supports migration stages
//! 4. Provides unified output

use super::super::contract::SamplingBackend;
use super::super::result::*;
use super::super::paths::{NovaPath, AmberPath};
use super::super::shadow::ShadowComparator;
use super::super::migration::{MigrationFlags, MigrationStage};
use crate::pdb_sanitizer::SanitizedStructure;
use anyhow::Result;
use std::sync::Arc;
use cudarc::driver::CudaContext;

/// Routing strategy
#[derive(Debug, Clone, Copy, Default)]
pub enum RoutingStrategy {
    /// Auto-select based on atom count
    #[default]
    Auto,
    /// Force NOVA (fails if too large)
    ForceNova,
    /// Force AMBER (loses TDA)
    ForceAmber,
}

/// Main entry point for sampling
pub struct HybridSampler {
    context: Arc<CudaContext>,
    nova: NovaPath,
    amber: AmberPath,
    strategy: RoutingStrategy,
    flags: MigrationFlags,
    selected: Option<BackendId>,
    structure: Option<SanitizedStructure>,
}

impl HybridSampler {
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            nova: NovaPath::new(Arc::clone(&context))?,
            amber: AmberPath::new(Arc::clone(&context))?,
            strategy: RoutingStrategy::Auto,
            flags: MigrationFlags::new(MigrationStage::Shadow),
            selected: None,
            structure: None,
            context,
        })
    }
    
    pub fn with_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = strategy;
        self
    }
    
    pub fn with_migration_stage(mut self, stage: MigrationStage) -> Self {
        self.flags = MigrationFlags::new(stage);
        self
    }
    
    /// Load structure and select backend
    pub fn load_structure(&mut self, structure: &SanitizedStructure) -> Result<BackendId> {
        let backend = self.select_backend(structure);
        
        match backend {
            BackendId::Nova => self.nova.load_structure(structure)?,
            BackendId::AmberMegaFused => self.amber.load_structure(structure)?,
            BackendId::Mock => unreachable!(),
        }
        
        self.selected = Some(backend);
        self.structure = Some(structure.clone());
        
        log::info!("HybridSampler: {} atoms -> {}", structure.n_atoms, backend);
        
        Ok(backend)
    }
    
    /// Run sampling
    pub fn sample(&mut self, config: &SamplingConfig) -> Result<SamplingResult> {
        let backend = self.selected
            .ok_or_else(|| anyhow::anyhow!("HybridSampler: No structure loaded"))?;
        
        match backend {
            BackendId::Nova => self.nova.sample(config),
            BackendId::AmberMegaFused => self.amber.sample(config),
            BackendId::Mock => unreachable!(),
        }
    }
    
    /// Run shadow comparison (both backends, compare results)
    pub fn sample_with_shadow(&mut self, config: &SamplingConfig) -> Result<ShadowResult> {
        let structure = self.structure.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No structure loaded"))?;
        
        // Only run shadow if structure fits in NOVA
        if structure.n_atoms > super::super::paths::nova_path::NOVA_MAX_ATOMS {
            return Ok(ShadowResult::SkippedLargeStructure(
                self.amber.sample(config)?
            ));
        }
        
        // Run NOVA
        self.nova.reset()?;
        self.nova.load_structure(structure)?;
        let nova_result = self.nova.sample(config)?;
        
        // Run AMBER
        self.amber.reset()?;
        self.amber.load_structure(structure)?;
        let amber_result = self.amber.sample(config)?;
        
        // Compare
        let divergence = compare_results(&nova_result, &amber_result);
        
        log::info!("Shadow comparison: RMSD={:.2}Å, energy_corr={:.3}",
                   divergence.mean_rmsd, divergence.energy_correlation);
        
        Ok(ShadowResult::Compared {
            primary: nova_result,
            shadow: amber_result,
            divergence,
        })
    }
    
    fn select_backend(&self, structure: &SanitizedStructure) -> BackendId {
        match self.strategy {
            RoutingStrategy::Auto => {
                if structure.n_atoms <= super::super::paths::nova_path::NOVA_MAX_ATOMS {
                    BackendId::Nova
                } else {
                    BackendId::AmberMegaFused
                }
            }
            RoutingStrategy::ForceNova => BackendId::Nova,
            RoutingStrategy::ForceAmber => BackendId::AmberMegaFused,
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
    
    /// Reset
    pub fn reset(&mut self) -> Result<()> {
        self.nova.reset()?;
        self.amber.reset()?;
        self.selected = None;
        self.structure = None;
        Ok(())
    }
}

/// Shadow comparison result
pub enum ShadowResult {
    Compared {
        primary: SamplingResult,
        shadow: SamplingResult,
        divergence: DivergenceMetrics,
    },
    SkippedLargeStructure(SamplingResult),
}

/// Divergence metrics
#[derive(Debug, Clone)]
pub struct DivergenceMetrics {
    pub mean_rmsd: f32,
    pub max_rmsd: f32,
    pub energy_correlation: f32,
}

fn compare_results(a: &SamplingResult, b: &SamplingResult) -> DivergenceMetrics {
    // Compute RMSD between conformations
    // Compute energy correlation
    todo!("Implement comparison")
}
```

### 6. Migration Flags (migration/feature_flags.rs)

**Purpose**: Control gradual rollout from stable to greenfield.

```rust
//! sampling/migration/feature_flags.rs
//!
//! Controls gradual migration from stable to greenfield path.
//! Supports strangler pattern for safe rollout.

/// Migration stage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationStage {
    /// 100% stable (AMBER)
    StableOnly,
    /// Run both, use stable result, log comparison
    Shadow,
    /// X% to greenfield (NOVA)
    Canary(u8),
    /// Greenfield primary, stable fallback
    GreenfieldPrimary,
    /// 100% greenfield
    GreenfieldOnly,
}

pub struct MigrationFlags {
    stage: MigrationStage,
    request_count: std::sync::atomic::AtomicU32,
    failure_count: std::sync::atomic::AtomicU32,
    max_failures_before_rollback: u32,
}

impl MigrationFlags {
    pub fn new(stage: MigrationStage) -> Self {
        Self {
            stage,
            request_count: std::sync::atomic::AtomicU32::new(0),
            failure_count: std::sync::atomic::AtomicU32::new(0),
            max_failures_before_rollback: 3,
        }
    }
    
    pub fn stage(&self) -> MigrationStage {
        self.stage
    }
    
    pub fn use_greenfield(&self) -> bool {
        use std::sync::atomic::Ordering;
        
        match self.stage {
            MigrationStage::StableOnly => false,
            MigrationStage::Shadow => false,
            MigrationStage::Canary(pct) => {
                let count = self.request_count.fetch_add(1, Ordering::Relaxed);
                (count % 100) < pct as u32
            }
            MigrationStage::GreenfieldPrimary => true,
            MigrationStage::GreenfieldOnly => true,
        }
    }
    
    pub fn run_shadow(&self) -> bool {
        matches!(self.stage, MigrationStage::Shadow)
    }
    
    pub fn report_success(&self) {
        use std::sync::atomic::Ordering;
        self.failure_count.store(0, Ordering::Relaxed);
    }
    
    pub fn report_failure(&self) -> bool {
        use std::sync::atomic::Ordering;
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        failures >= self.max_failures_before_rollback
    }
    
    pub fn advance(&mut self) -> bool {
        let next = match self.stage {
            MigrationStage::StableOnly => MigrationStage::Shadow,
            MigrationStage::Shadow => MigrationStage::Canary(10),
            MigrationStage::Canary(p) if p < 50 => MigrationStage::Canary(p + 10),
            MigrationStage::Canary(_) => MigrationStage::GreenfieldPrimary,
            MigrationStage::GreenfieldPrimary => MigrationStage::GreenfieldOnly,
            MigrationStage::GreenfieldOnly => return false,
        };
        log::info!("Migration: {:?} -> {:?}", self.stage, next);
        self.stage = next;
        true
    }
    
    pub fn rollback(&mut self) -> bool {
        let prev = match self.stage {
            MigrationStage::StableOnly => return false,
            MigrationStage::Shadow => MigrationStage::StableOnly,
            MigrationStage::Canary(_) => MigrationStage::Shadow,
            MigrationStage::GreenfieldPrimary => MigrationStage::Canary(50),
            MigrationStage::GreenfieldOnly => MigrationStage::GreenfieldPrimary,
        };
        log::warn!("Migration ROLLBACK: {:?} -> {:?}", self.stage, prev);
        self.stage = prev;
        true
    }
}
```

---

## Integration Tests (Required for Phase 6)

```rust
//! tests/parallel_implementation_tests.rs
//!
//! These tests MUST pass before Phase 6 is complete.

use prism_validation::sampling::*;

/// Test that both paths implement contract correctly
#[test]
fn test_contract_compliance() {
    // Test NovaPath satisfies SamplingBackend
    // Test AmberPath satisfies SamplingBackend
}

/// Test that auto-routing works
#[test]
fn test_auto_routing_small_structure() {
    // Small structure -> NOVA
}

#[test]
fn test_auto_routing_large_structure() {
    // Large structure -> AMBER
}

/// Test shadow comparison
#[test]
#[ignore] // Requires GPU
fn test_shadow_comparison() {
    // Run both paths
    // Verify comparable results
    // Log divergence
}

/// Test output format compatibility
#[test]
#[ignore] // Requires GPU
fn test_output_format_identical() {
    // Both paths produce same SamplingResult structure
    // Downstream can process either transparently
}

/// Test rollback mechanism
#[test]
fn test_migration_rollback() {
    // Simulate failures
    // Verify automatic rollback
}
```

---

## Phase 6 Checklist for Parallel Architecture

```
□ contract.rs created with SamplingBackend trait
□ result.rs created with SamplingResult types
□ nova_path.rs implements SamplingBackend
□ amber_path.rs implements SamplingBackend
□ hybrid_sampler.rs routes correctly
□ feature_flags.rs controls migration
□ Shadow comparison working
□ All parallel implementation tests pass
□ Documentation complete
```

---

## Phase 7-8 Evolution Points

The parallel architecture enables these future enhancements WITHOUT breaking changes:

### Phase 7 Changes (nova_path.rs only)

```rust
// ADDITIONS to NovaPath (backward compatible)
impl NovaPath {
    // New: Extended sampling
    pub fn sample_extended(&mut self, config: &ExtendedConfig) -> Result<SamplingResult> {
        // 2000 samples, adaptive bias, temperature annealing
    }
    
    // New: Persistence diagrams (added to SamplingResult.persistence_diagrams)
    fn compute_persistence(&self) -> Vec<PersistenceDiagram> {
        // Full TDA
    }
}
```

### Phase 8 Changes (new files, wrap existing)

```rust
// NEW: ensemble_reservoir.rs
pub struct EnsembleReservoir {
    reservoirs: [HierarchicalReservoir; 5],
    // Wraps hierarchical, doesn't modify it
}

// NEW: transfer_learning.rs  
pub struct TransferManager {
    // Completely new, doesn't touch existing paths
}
```

**AMBER path remains UNTOUCHED through all phases.**

---

## Summary

This architecture provides:

| Capability | Implementation |
|------------|----------------|
| **Parallel paths** | nova_path.rs (greenfield) + amber_path.rs (stable) |
| **Unified interface** | SamplingBackend trait + SamplingResult type |
| **Shadow validation** | ShadowComparator runs both, compares |
| **Safe evolution** | NOVA evolves, AMBER frozen |
| **Gradual rollout** | MigrationFlags (canary, strangler pattern) |
| **Automatic rollback** | On consecutive failures |

**This architecture is REQUIRED infrastructure for Phase 6.**  
**It enables Phase 7-8 enhancements without breaking changes.**

---

**END OF PARALLEL IMPLEMENTATION ARCHITECTURE DOCUMENT**
