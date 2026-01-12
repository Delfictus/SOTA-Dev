# PRISM Phase 7-8: Enhancement Implementation Plan
## From Competitive to Category Leader

**Document Version**: 1.0  
**Generated**: 2026-01-12  
**Prerequisites**: Phase 6 Complete (ROC AUC ≥0.70)  
**Timeline**: 12-16 Weeks  
**Target**: ROC AUC ≥0.90, PR AUC ≥0.40  
**Classification**: Zero Fallback / Zero External Dependencies  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Phase 6 Baseline Requirements](#2-phase-6-baseline-requirements)
3. [Phase 7: Architectural Enhancements (Weeks 1-8)](#3-phase-7-architectural-enhancements)
4. [Phase 8: Advanced Capabilities (Weeks 9-16)](#4-phase-8-advanced-capabilities)
5. [Component Specifications](#5-component-specifications)
6. [Implementation Timeline](#6-implementation-timeline)
7. [File Manifest](#7-file-manifest)
8. [Verification Strategy](#8-verification-strategy)
9. [Risk Mitigation](#9-risk-mitigation)

---

## 1. Executive Summary

### Objective

Transform PRISM from **competitive** (Phase 6: ~0.75 AUC) to **category leader** (Phase 8: ~0.90 AUC) while maintaining **complete sovereignty**.

### Target Metrics Progression

| Metric | Phase 6 (Baseline) | Phase 7 (Target) | Phase 8 (Target) | SOTA Reference |
|--------|-------------------|------------------|------------------|----------------|
| ROC AUC | 0.75 | **0.82** | **0.90** | PocketMiner 0.87 |
| PR AUC | 0.25 | **0.32** | **0.40** | PocketMiner 0.44 |
| Success Rate | 85% | **88%** | **92%** | - |
| Top-1 Accuracy | 90% | **92%** | **95%** | - |
| Time/Structure | <1s | **<1.5s** | **<2s** | - |

### Enhancement Components

#### Phase 7: Architectural (Weeks 1-8)
1. **Hierarchical Neuromorphic Reservoir** - Cortical column architecture
2. **Persistent Homology Integration** - Full TDA beyond Betti numbers
3. **Extended NOVA Sampling** - 2000 conformations with adaptive biasing
4. **Multi-Scale Feature Extraction** - Local + regional + global features

#### Phase 8: Advanced (Weeks 9-16)
5. **Ensemble Reservoir Voting** - Multiple reservoirs with learned combination
6. **Cross-Structure Transfer Learning** - Protein family knowledge transfer
7. **Uncertainty Quantification** - Confidence scores for predictions
8. **Active Learning Pipeline** - Prioritized structure selection for improvement

### Expected Gains by Component

| Component | Expected Δ AUC | Cumulative | Confidence |
|-----------|----------------|------------|------------|
| Phase 6 Baseline | - | 0.75 | Achieved |
| Hierarchical Reservoir | +0.03 | 0.78 | High |
| Persistent Homology | +0.02 | 0.80 | High |
| Extended Sampling | +0.02 | 0.82 | Medium |
| Multi-Scale Features | +0.02 | 0.84 | Medium |
| Ensemble Voting | +0.03 | 0.87 | High |
| Transfer Learning | +0.03 | 0.90 | Medium |
| **Total** | **+0.15** | **0.90** | - |

### Architectural Constraints (Unchanged)

```
✅ REQUIRED:
   - Native Rust/CUDA implementations only
   - GPU-mandatory execution
   - Zero external ML dependencies
   - Full IP sovereignty

❌ FORBIDDEN:
   - PyTorch, TensorFlow, JAX
   - ESM-2, ProtTrans, AlphaFold
   - External API calls
   - Cloud dependencies
```

---

## 2. Phase 6 Baseline Requirements

### Minimum Requirements to Begin Phase 7

```
□ CryptoBench ROC AUC ≥ 0.70
□ CryptoBench PR AUC ≥ 0.20
□ Success Rate ≥ 80%
□ GPU scorer operational (512 neurons)
□ NOVA sampling functional (500 conformations)
□ RLS online learning stable
□ TDA Betti numbers integrated
□ All Phase 6 tests passing
```

### Baseline Metrics Documentation

Before starting Phase 7, document in `results/PHASE6_FINAL_METRICS.md`:

```markdown
# Phase 6 Final Metrics

**Date**: [DATE]
**Commit**: [COMMIT_HASH]

## CryptoBench Test Set (222 structures)
| Metric | Value |
|--------|-------|
| ROC AUC | [VALUE] |
| PR AUC | [VALUE] |
| Success Rate | [VALUE]% |
| Top-1 Accuracy | [VALUE]% |
| Mean Time/Structure | [VALUE]s |
| Peak VRAM | [VALUE] GB |

## Component Status
- [x] GPU ZrO Scorer (512 neurons)
- [x] NOVA Sampler (500 conformations)
- [x] RLS Online Learning (λ=0.99)
- [x] TDA Betti Numbers (β₀, β₁, β₂)
- [x] Apo-Holo Benchmark (X/15 success)
- [x] Ablation Study Complete
```

---

## 3. Phase 7: Architectural Enhancements (Weeks 1-8)

### 3.1 Hierarchical Neuromorphic Reservoir

#### Objective

Replace flat 512-neuron reservoir with biologically-inspired cortical column hierarchy.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              HIERARCHICAL NEUROMORPHIC RESERVOIR                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 3: Global Context (1 × 256 neurons)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Global integration, cross-region patterns              │   │
│  │  Receptive field: entire protein                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↑                                      │
│  Layer 2: Regional Integration (4 × 128 neurons)               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Region 1 │ │ Region 2 │ │ Region 3 │ │ Region 4 │          │
│  │ Dynamics │ │ Structure│ │ Chemical │ │ Spatial  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│       ↑            ↑            ↑            ↑                  │
│  Layer 1: Local Feature Detectors (8 × 64 neurons)             │
│  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐             │
│  │ L1 ││ L2 ││ L3 ││ L4 ││ L5 ││ L6 ││ L7 ││ L8 │             │
│  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘             │
│                          ↑                                      │
│  Input: 40-dim feature vector (16 features + velocities)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Total neurons: 8×64 + 4×128 + 256 = 512 + 512 + 256 = 1,280
VRAM estimate: ~3.2 MB (fits easily on RTX 3060)
```

#### Specifications

| Layer | Columns | Neurons/Column | Total | Function |
|-------|---------|----------------|-------|----------|
| L1 (Local) | 8 | 64 | 512 | Feature-specific detectors |
| L2 (Regional) | 4 | 128 | 512 | Feature category integration |
| L3 (Global) | 1 | 256 | 256 | Whole-protein context |
| **Total** | - | - | **1,280** | - |

#### Layer Connectivity

```
L1 → L2 Mapping:
  L1[0,1] (burial, rmsf)      → L2[0] (Dynamics)
  L1[2,3] (ss_flex, b_factor) → L2[1] (Structure)
  L1[4,5] (charge, hydro)     → L2[2] (Chemical)
  L1[6,7] (contact, interface)→ L2[3] (Spatial)

L2 → L3 Mapping:
  All L2 columns → L3 (full connectivity, 10% sparse)

Lateral Connections:
  Within-layer inhibition (20% inhibitory)
  Cross-column excitation (5% sparse)
```

#### Implementation

**File**: `crates/prism-gpu/src/hierarchical_reservoir.rs`

```rust
//! Hierarchical Neuromorphic Reservoir
//!
//! Biologically-inspired cortical column architecture with:
//! - Layer 1: Local feature detectors (8 columns × 64 neurons)
//! - Layer 2: Regional integrators (4 columns × 128 neurons)
//! - Layer 3: Global context (1 column × 256 neurons)
//!
//! Total: 1,280 neurons with structured sparse connectivity

use anyhow::{Result, Context};
use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig};

/// Configuration for hierarchical reservoir
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Layer 1: Number of local columns
    pub l1_columns: usize,
    /// Layer 1: Neurons per column
    pub l1_neurons_per_column: usize,
    
    /// Layer 2: Number of regional columns
    pub l2_columns: usize,
    /// Layer 2: Neurons per column
    pub l2_neurons_per_column: usize,
    
    /// Layer 3: Global column neurons
    pub l3_neurons: usize,
    
    /// Intra-layer connectivity density
    pub intra_density: f32,
    /// Inter-layer connectivity density
    pub inter_density: f32,
    /// Lateral connectivity density
    pub lateral_density: f32,
    
    /// Excitatory/Inhibitory ratio
    pub excitatory_ratio: f32,
    
    /// Time constants (ms)
    pub tau_fast: f32,
    pub tau_slow: f32,
    
    /// Spectral radius for echo state property
    pub spectral_radius: f32,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            l1_columns: 8,
            l1_neurons_per_column: 64,
            l2_columns: 4,
            l2_neurons_per_column: 128,
            l3_neurons: 256,
            intra_density: 0.10,
            inter_density: 0.15,
            lateral_density: 0.05,
            excitatory_ratio: 0.80,
            tau_fast: 5.0,
            tau_slow: 50.0,
            spectral_radius: 0.95,
        }
    }
}

/// Single cortical column with LIF neurons
#[derive(Debug)]
pub struct CorticalColumn {
    /// Number of neurons
    pub n_neurons: usize,
    
    /// Membrane potentials (GPU)
    pub membrane: CudaSlice<f32>,
    
    /// Spike states (GPU)
    pub spikes: CudaSlice<f32>,
    
    /// Recurrent weights within column (GPU, sparse)
    pub recurrent_weights: SparseMatrix,
    
    /// Time constants per neuron (GPU)
    pub tau: CudaSlice<f32>,
    
    /// Threshold potentials
    pub threshold: f32,
    
    /// Reset potential
    pub reset: f32,
}

/// Sparse matrix for efficient connectivity
#[derive(Debug)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub row_ptr: CudaSlice<i32>,
    pub col_idx: CudaSlice<i32>,
    pub values: CudaSlice<f32>,
}

/// Hierarchical neuromorphic reservoir
pub struct HierarchicalReservoir {
    context: Arc<CudaContext>,
    config: HierarchicalConfig,
    
    /// Layer 1: Local feature detectors
    l1_columns: Vec<CorticalColumn>,
    
    /// Layer 2: Regional integrators
    l2_columns: Vec<CorticalColumn>,
    
    /// Layer 3: Global context
    l3_column: CorticalColumn,
    
    /// L1 → L2 projection weights
    l1_to_l2: Vec<SparseMatrix>,
    
    /// L2 → L3 projection weights
    l2_to_l3: SparseMatrix,
    
    /// Lateral connections within each layer
    l1_lateral: SparseMatrix,
    l2_lateral: SparseMatrix,
    
    /// Input projection to L1
    input_weights: CudaSlice<f32>,
    
    /// Readout weights from all layers
    readout_weights: CudaSlice<f32>,
    
    /// Total neuron count
    total_neurons: usize,
    
    /// CUDA kernels
    kernel_l1_step: CudaFunction,
    kernel_l2_step: CudaFunction,
    kernel_l3_step: CudaFunction,
    kernel_readout: CudaFunction,
}

impl HierarchicalReservoir {
    /// Create new hierarchical reservoir
    pub fn new(context: Arc<CudaContext>, config: HierarchicalConfig) -> Result<Self> {
        let total_l1 = config.l1_columns * config.l1_neurons_per_column;
        let total_l2 = config.l2_columns * config.l2_neurons_per_column;
        let total_l3 = config.l3_neurons;
        let total_neurons = total_l1 + total_l2 + total_l3;
        
        log::info!("Creating hierarchical reservoir: {} total neurons", total_neurons);
        log::info!("  L1: {} columns × {} = {} neurons", 
                   config.l1_columns, config.l1_neurons_per_column, total_l1);
        log::info!("  L2: {} columns × {} = {} neurons",
                   config.l2_columns, config.l2_neurons_per_column, total_l2);
        log::info!("  L3: {} neurons", total_l3);
        
        // Create L1 columns
        let mut l1_columns = Vec::with_capacity(config.l1_columns);
        for i in 0..config.l1_columns {
            l1_columns.push(Self::create_column(
                &context,
                config.l1_neurons_per_column,
                config.intra_density,
                config.excitatory_ratio,
                config.tau_fast,
                config.tau_slow,
            )?);
        }
        
        // Create L2 columns
        let mut l2_columns = Vec::with_capacity(config.l2_columns);
        for i in 0..config.l2_columns {
            l2_columns.push(Self::create_column(
                &context,
                config.l2_neurons_per_column,
                config.intra_density,
                config.excitatory_ratio,
                config.tau_fast,
                config.tau_slow,
            )?);
        }
        
        // Create L3 column
        let l3_column = Self::create_column(
            &context,
            config.l3_neurons,
            config.intra_density,
            config.excitatory_ratio,
            config.tau_fast,
            config.tau_slow,
        )?;
        
        // Create inter-layer projections
        let l1_to_l2 = Self::create_l1_to_l2_projections(&context, &config)?;
        let l2_to_l3 = Self::create_projection(
            &context,
            total_l2,
            total_l3,
            config.inter_density,
        )?;
        
        // Create lateral connections
        let l1_lateral = Self::create_lateral(&context, total_l1, config.lateral_density)?;
        let l2_lateral = Self::create_lateral(&context, total_l2, config.lateral_density)?;
        
        // Create input weights (40 inputs → L1)
        let input_weights = Self::create_input_weights(&context, 40, total_l1)?;
        
        // Create readout weights (all neurons → 1 output)
        let readout_weights = context.alloc_zeros::<f32>(total_neurons)?;
        
        // Load CUDA kernels
        let module = context.load_module(HIERARCHICAL_KERNEL_PTX)?;
        let kernel_l1_step = module.get_function("hierarchical_l1_step")?;
        let kernel_l2_step = module.get_function("hierarchical_l2_step")?;
        let kernel_l3_step = module.get_function("hierarchical_l3_step")?;
        let kernel_readout = module.get_function("hierarchical_readout")?;
        
        Ok(Self {
            context,
            config,
            l1_columns,
            l2_columns,
            l3_column,
            l1_to_l2,
            l2_to_l3,
            l1_lateral,
            l2_lateral,
            input_weights,
            readout_weights,
            total_neurons,
            kernel_l1_step,
            kernel_l2_step,
            kernel_l3_step,
            kernel_readout,
        })
    }
    
    /// Process input through hierarchy and return final state
    pub fn process(&mut self, input: &[f32; 40]) -> Result<Vec<f32>> {
        // Upload input
        let input_gpu = self.context.htod_sync_copy(input)?;
        
        // Step 1: Input → L1
        self.step_l1(&input_gpu)?;
        
        // Step 2: L1 → L2 + lateral
        self.step_l2()?;
        
        // Step 3: L2 → L3 + lateral
        self.step_l3()?;
        
        // Collect full state
        self.collect_state()
    }
    
    /// Process with temporal integration (multiple timesteps)
    pub fn process_temporal(&mut self, input: &[f32; 40], timesteps: usize) -> Result<Vec<f32>> {
        let input_gpu = self.context.htod_sync_copy(input)?;
        
        for _ in 0..timesteps {
            self.step_l1(&input_gpu)?;
            self.step_l2()?;
            self.step_l3()?;
        }
        
        self.collect_state()
    }
    
    /// Step Layer 1 (input + recurrent + lateral)
    fn step_l1(&mut self, input: &CudaSlice<f32>) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(
            self.config.l1_columns * self.config.l1_neurons_per_column
        );
        
        unsafe {
            self.kernel_l1_step.launch(cfg, (
                input,
                &self.input_weights,
                // ... column states, recurrent weights, lateral weights
            ))?;
        }
        
        Ok(())
    }
    
    /// Step Layer 2 (L1 input + recurrent + lateral)
    fn step_l2(&mut self) -> Result<()> {
        // Aggregate L1 spikes into L2 columns based on mapping
        // Apply recurrent dynamics
        // Apply lateral inhibition
        Ok(())
    }
    
    /// Step Layer 3 (L2 input + recurrent)
    fn step_l3(&mut self) -> Result<()> {
        // Aggregate L2 spikes into L3
        // Apply recurrent dynamics
        Ok(())
    }
    
    /// Collect state from all layers
    fn collect_state(&self) -> Result<Vec<f32>> {
        let mut state = Vec::with_capacity(self.total_neurons);
        
        // Collect L1 states
        for col in &self.l1_columns {
            let col_state = self.context.dtoh_sync_copy(&col.membrane)?;
            state.extend(col_state);
        }
        
        // Collect L2 states
        for col in &self.l2_columns {
            let col_state = self.context.dtoh_sync_copy(&col.membrane)?;
            state.extend(col_state);
        }
        
        // Collect L3 state
        let l3_state = self.context.dtoh_sync_copy(&self.l3_column.membrane)?;
        state.extend(l3_state);
        
        Ok(state)
    }
    
    /// Reset all column states
    pub fn reset(&mut self) -> Result<()> {
        for col in &mut self.l1_columns {
            self.context.memset_zeros(&mut col.membrane)?;
            self.context.memset_zeros(&mut col.spikes)?;
        }
        for col in &mut self.l2_columns {
            self.context.memset_zeros(&mut col.membrane)?;
            self.context.memset_zeros(&mut col.spikes)?;
        }
        self.context.memset_zeros(&mut self.l3_column.membrane)?;
        self.context.memset_zeros(&mut self.l3_column.spikes)?;
        
        Ok(())
    }
    
    /// Get total neuron count
    pub fn total_neurons(&self) -> usize {
        self.total_neurons
    }
    
    // Helper functions for initialization...
    fn create_column(
        context: &Arc<CudaContext>,
        n_neurons: usize,
        density: f32,
        exc_ratio: f32,
        tau_fast: f32,
        tau_slow: f32,
    ) -> Result<CorticalColumn> {
        // Implementation...
        todo!()
    }
    
    fn create_l1_to_l2_projections(
        context: &Arc<CudaContext>,
        config: &HierarchicalConfig,
    ) -> Result<Vec<SparseMatrix>> {
        // L1 columns 0,1 → L2 column 0 (Dynamics)
        // L1 columns 2,3 → L2 column 1 (Structure)
        // L1 columns 4,5 → L2 column 2 (Chemical)
        // L1 columns 6,7 → L2 column 3 (Spatial)
        todo!()
    }
    
    fn create_projection(
        context: &Arc<CudaContext>,
        from_size: usize,
        to_size: usize,
        density: f32,
    ) -> Result<SparseMatrix> {
        todo!()
    }
    
    fn create_lateral(
        context: &Arc<CudaContext>,
        size: usize,
        density: f32,
    ) -> Result<SparseMatrix> {
        todo!()
    }
    
    fn create_input_weights(
        context: &Arc<CudaContext>,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<CudaSlice<f32>> {
        todo!()
    }
}

/// CUDA kernel source for hierarchical operations
const HIERARCHICAL_KERNEL_PTX: &[u8] = include_bytes!("../kernels/hierarchical_reservoir.ptx");
```

---

### 3.2 Persistent Homology Integration

#### Objective

Extend TDA beyond Betti numbers to full persistence diagrams, capturing **when** topological features form and **how long** they persist.

#### Background

```
Current (Phase 6):
  β₀ = 1  (connected components)
  β₁ = 2  (loops)
  β₂ = 1  (voids/cavities)
  
  Problem: Only counts features, not their significance

Enhanced (Phase 7):
  Persistence Diagram for each dimension:
  - Birth time: when feature appears (filtration parameter)
  - Death time: when feature disappears
  - Lifetime = death - birth (significance measure)
  
  Captures:
  - Transient pockets (short lifetime) vs stable pockets (long lifetime)
  - Pocket depth (birth time)
  - Pocket volume evolution (persistence landscape)
```

#### Persistence Features

```
┌─────────────────────────────────────────────────────────────────┐
│                    PERSISTENCE FEATURES                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Per-Dimension Features (β₀, β₁, β₂):                          │
│  ├── Count: number of features                                 │
│  ├── Total Persistence: Σ(death - birth)                       │
│  ├── Max Lifetime: max(death - birth)                          │
│  ├── Mean Lifetime: mean(death - birth)                        │
│  ├── Persistence Entropy: -Σ(p_i × log(p_i))                   │
│  └── Birth/Death Statistics: mean, std, min, max               │
│                                                                 │
│  Cryptic-Specific Features (β₂ focus):                         │
│  ├── Void Birth Threshold: earliest void birth                 │
│  ├── Stable Void Count: voids with lifetime > threshold        │
│  ├── Void Volume Proxy: Σ(lifetime × birth_radius³)            │
│  └── Pocket Opening Score: derivative of β₂ persistence        │
│                                                                 │
│  Total: 24 additional features                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation

**File**: `crates/prism-gpu/src/persistent_homology.rs`

```rust
//! Persistent Homology for Cryptic Site Detection
//!
//! Computes persistence diagrams and derived features from
//! alpha complex filtration of protein structures.

use anyhow::{Result, Context};
use std::sync::Arc;
use cudarc::driver::CudaContext;

/// A single persistence pair (birth, death)
#[derive(Debug, Clone, Copy, Default)]
pub struct PersistencePair {
    pub birth: f32,
    pub death: f32,
}

impl PersistencePair {
    pub fn lifetime(&self) -> f32 {
        self.death - self.birth
    }
    
    pub fn midpoint(&self) -> f32 {
        (self.birth + self.death) / 2.0
    }
}

/// Persistence diagram for one homology dimension
#[derive(Debug, Clone, Default)]
pub struct PersistenceDiagram {
    pub dimension: usize,
    pub pairs: Vec<PersistencePair>,
}

impl PersistenceDiagram {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            pairs: Vec::new(),
        }
    }
    
    /// Total persistence (sum of lifetimes)
    pub fn total_persistence(&self) -> f32 {
        self.pairs.iter().map(|p| p.lifetime()).sum()
    }
    
    /// Maximum lifetime
    pub fn max_lifetime(&self) -> f32 {
        self.pairs.iter()
            .map(|p| p.lifetime())
            .fold(0.0f32, f32::max)
    }
    
    /// Mean lifetime
    pub fn mean_lifetime(&self) -> f32 {
        if self.pairs.is_empty() {
            return 0.0;
        }
        self.total_persistence() / self.pairs.len() as f32
    }
    
    /// Persistence entropy
    pub fn entropy(&self) -> f32 {
        let total = self.total_persistence();
        if total < 1e-10 {
            return 0.0;
        }
        
        -self.pairs.iter()
            .map(|p| {
                let prob = p.lifetime() / total;
                if prob > 1e-10 {
                    prob * prob.ln()
                } else {
                    0.0
                }
            })
            .sum::<f32>()
    }
    
    /// Count of features above lifetime threshold
    pub fn count_above_threshold(&self, threshold: f32) -> usize {
        self.pairs.iter()
            .filter(|p| p.lifetime() > threshold)
            .count()
    }
    
    /// Birth statistics
    pub fn birth_stats(&self) -> DescriptiveStats {
        let births: Vec<f32> = self.pairs.iter().map(|p| p.birth).collect();
        DescriptiveStats::compute(&births)
    }
    
    /// Death statistics
    pub fn death_stats(&self) -> DescriptiveStats {
        let deaths: Vec<f32> = self.pairs.iter().map(|p| p.death).collect();
        DescriptiveStats::compute(&deaths)
    }
}

/// Descriptive statistics
#[derive(Debug, Clone, Default)]
pub struct DescriptiveStats {
    pub count: usize,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
}

impl DescriptiveStats {
    pub fn compute(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        
        let count = values.len();
        let mean = values.iter().sum::<f32>() / count as f32;
        let var = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / count as f32;
        let std = var.sqrt();
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        Self { count, mean, std, min, max }
    }
}

/// Full persistence features for cryptic site detection
#[derive(Debug, Clone, Default)]
pub struct PersistenceFeatures {
    // Per-dimension features (β₀, β₁, β₂)
    pub betti_0: BettiFeatures,
    pub betti_1: BettiFeatures,
    pub betti_2: BettiFeatures,
    
    // Cryptic-specific features
    pub void_birth_threshold: f32,
    pub stable_void_count: usize,
    pub void_volume_proxy: f32,
    pub pocket_opening_score: f32,
    
    // Cross-dimension features
    pub total_topological_complexity: f32,
    pub persistence_ratio_2_1: f32,  // β₂/β₁ persistence ratio
}

/// Features for single Betti dimension
#[derive(Debug, Clone, Default)]
pub struct BettiFeatures {
    pub count: usize,
    pub total_persistence: f32,
    pub max_lifetime: f32,
    pub mean_lifetime: f32,
    pub entropy: f32,
    pub birth_mean: f32,
    pub birth_std: f32,
    pub death_mean: f32,
    pub death_std: f32,
}

impl BettiFeatures {
    pub fn from_diagram(diagram: &PersistenceDiagram) -> Self {
        let birth_stats = diagram.birth_stats();
        let death_stats = diagram.death_stats();
        
        Self {
            count: diagram.pairs.len(),
            total_persistence: diagram.total_persistence(),
            max_lifetime: diagram.max_lifetime(),
            mean_lifetime: diagram.mean_lifetime(),
            entropy: diagram.entropy(),
            birth_mean: birth_stats.mean,
            birth_std: birth_stats.std,
            death_mean: death_stats.mean,
            death_std: death_stats.std,
        }
    }
    
    /// Convert to feature array (9 features)
    pub fn to_array(&self) -> [f32; 9] {
        [
            self.count as f32,
            self.total_persistence,
            self.max_lifetime,
            self.mean_lifetime,
            self.entropy,
            self.birth_mean,
            self.birth_std,
            self.death_mean,
            self.death_std,
        ]
    }
}

impl PersistenceFeatures {
    /// Compute from persistence diagrams
    pub fn from_diagrams(
        diagram_0: &PersistenceDiagram,
        diagram_1: &PersistenceDiagram,
        diagram_2: &PersistenceDiagram,
    ) -> Self {
        let betti_0 = BettiFeatures::from_diagram(diagram_0);
        let betti_1 = BettiFeatures::from_diagram(diagram_1);
        let betti_2 = BettiFeatures::from_diagram(diagram_2);
        
        // Cryptic-specific: focus on voids (β₂)
        let void_birth_threshold = diagram_2.pairs.iter()
            .map(|p| p.birth)
            .fold(f32::INFINITY, f32::min);
        
        let stable_void_count = diagram_2.count_above_threshold(1.0);
        
        // Volume proxy: larger birth radius + longer lifetime = larger pocket
        let void_volume_proxy = diagram_2.pairs.iter()
            .map(|p| p.lifetime() * p.birth.powi(3))
            .sum();
        
        // Pocket opening score: rapid β₂ increase indicates pocket formation
        let pocket_opening_score = if betti_2.count > 0 {
            betti_2.total_persistence / (betti_2.birth_mean + 0.1)
        } else {
            0.0
        };
        
        // Cross-dimension
        let total_complexity = betti_0.total_persistence 
            + betti_1.total_persistence 
            + betti_2.total_persistence;
        
        let persistence_ratio = if betti_1.total_persistence > 0.01 {
            betti_2.total_persistence / betti_1.total_persistence
        } else {
            0.0
        };
        
        Self {
            betti_0,
            betti_1,
            betti_2,
            void_birth_threshold,
            stable_void_count,
            void_volume_proxy,
            pocket_opening_score,
            total_topological_complexity: total_complexity,
            persistence_ratio_2_1: persistence_ratio,
        }
    }
    
    /// Convert to flat feature array (31 features total)
    pub fn to_array(&self) -> [f32; 31] {
        let mut arr = [0.0f32; 31];
        
        // β₀ features (0-8)
        arr[0..9].copy_from_slice(&self.betti_0.to_array());
        
        // β₁ features (9-17)
        arr[9..18].copy_from_slice(&self.betti_1.to_array());
        
        // β₂ features (18-26)
        arr[18..27].copy_from_slice(&self.betti_2.to_array());
        
        // Cryptic-specific (27-30)
        arr[27] = self.void_birth_threshold;
        arr[28] = self.stable_void_count as f32;
        arr[29] = self.void_volume_proxy;
        arr[30] = self.pocket_opening_score;
        
        arr
    }
}

/// GPU-accelerated persistence computation
pub struct PersistenceComputer {
    context: Arc<CudaContext>,
    // CUDA kernels for alpha complex and persistence
}

impl PersistenceComputer {
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        Ok(Self { context })
    }
    
    /// Compute persistence from point cloud (Cα coordinates)
    pub fn compute(&self, coords: &[[f32; 3]]) -> Result<(
        PersistenceDiagram,
        PersistenceDiagram,
        PersistenceDiagram,
    )> {
        // Build alpha complex filtration
        let filtration = self.build_alpha_complex(coords)?;
        
        // Compute persistence via matrix reduction
        let diagrams = self.reduce_boundary_matrix(&filtration)?;
        
        Ok(diagrams)
    }
    
    /// Compute persistence features
    pub fn compute_features(&self, coords: &[[f32; 3]]) -> Result<PersistenceFeatures> {
        let (d0, d1, d2) = self.compute(coords)?;
        Ok(PersistenceFeatures::from_diagrams(&d0, &d1, &d2))
    }
    
    fn build_alpha_complex(&self, coords: &[[f32; 3]]) -> Result<AlphaComplex> {
        // Delaunay triangulation + alpha filtration
        todo!()
    }
    
    fn reduce_boundary_matrix(&self, filtration: &AlphaComplex) -> Result<(
        PersistenceDiagram,
        PersistenceDiagram,
        PersistenceDiagram,
    )> {
        // Standard persistence algorithm on GPU
        todo!()
    }
}

struct AlphaComplex {
    // Simplicial complex with filtration values
}
```

---

### 3.3 Extended NOVA Sampling

#### Objective

Increase sampling from 500 to 2000 conformations with adaptive biasing toward pocket-opening states.

#### Enhancements

| Parameter | Phase 6 | Phase 7 | Rationale |
|-----------|---------|---------|-----------|
| Samples | 500 | 2000 | More coverage of conformational space |
| Steps/Sample | 100 | 50 | Faster decorrelation with better integrator |
| Adaptive Bias | Fixed goal | Learned bias | Focus on productive regions |
| Temperature | 310K fixed | 310K ± annealing | Escape local minima |
| TDA Feedback | None | Real-time β₂ | Bias toward void formation |

#### Adaptive Biasing Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                 ADAPTIVE BIASING STRATEGY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Initial Exploration (samples 1-500):                       │
│     - Standard HMC at T=310K                                   │
│     - No bias, explore freely                                  │
│     - Track β₂ history                                         │
│                                                                 │
│  2. Identify Productive Directions (sample 500):               │
│     - Find conformations with highest β₂                       │
│     - Compute collective variable (CV) gradient                │
│     - Define biasing potential                                 │
│                                                                 │
│  3. Biased Sampling (samples 501-1500):                        │
│     - Apply metadynamics-like bias                             │
│     - Gaussian hills deposited at β₂ minima                    │
│     - Push system toward pocket-open states                    │
│                                                                 │
│  4. Refinement (samples 1501-2000):                            │
│     - Temperature annealing (310K → 290K)                      │
│     - Sample around best pocket-open states                    │
│     - High-resolution pocket characterization                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation

**File**: `crates/prism-validation/src/extended_nova_sampler.rs`

```rust
//! Extended NOVA Sampler with Adaptive Biasing
//!
//! Enhancements over Phase 6:
//! - 2000 conformations (4x increase)
//! - Adaptive biasing toward pocket-opening
//! - Temperature annealing
//! - Real-time TDA feedback

use anyhow::{Result, Context};
use std::sync::Arc;
use cudarc::driver::CudaContext;

use crate::nova_cryptic_sampler::{NovaCrypticSampler, NovaCrypticConfig, NovaSamplingResult};
use crate::persistent_homology::{PersistenceComputer, PersistenceFeatures};

/// Extended sampling configuration
#[derive(Debug, Clone)]
pub struct ExtendedNovaConfig {
    /// Base NOVA configuration
    pub base: NovaCrypticConfig,
    
    /// Total conformations to sample
    pub total_samples: usize,
    
    /// Enable adaptive biasing
    pub adaptive_bias: bool,
    
    /// Samples before activating bias
    pub exploration_samples: usize,
    
    /// Enable temperature annealing
    pub temperature_annealing: bool,
    
    /// Final temperature for annealing
    pub final_temperature: f32,
    
    /// Annealing start sample
    pub annealing_start: usize,
    
    /// Metadynamics hill height
    pub hill_height: f32,
    
    /// Metadynamics hill width
    pub hill_width: f32,
    
    /// Deposit frequency (samples between hills)
    pub hill_frequency: usize,
}

impl Default for ExtendedNovaConfig {
    fn default() -> Self {
        let mut base = NovaCrypticConfig::default();
        base.steps_per_sample = 50;  // Faster decorrelation
        
        Self {
            base,
            total_samples: 2000,
            adaptive_bias: true,
            exploration_samples: 500,
            temperature_annealing: true,
            final_temperature: 290.0,
            annealing_start: 1500,
            hill_height: 0.5,
            hill_width: 0.2,
            hill_frequency: 50,
        }
    }
}

/// Metadynamics bias potential
#[derive(Debug, Clone)]
pub struct MetadynamicsBias {
    /// Deposited Gaussian hills (β₂_value, height)
    hills: Vec<(f32, f32)>,
    
    /// Hill width (σ)
    width: f32,
}

impl MetadynamicsBias {
    pub fn new(width: f32) -> Self {
        Self {
            hills: Vec::new(),
            width,
        }
    }
    
    /// Deposit a new hill at current β₂ value
    pub fn deposit_hill(&mut self, beta2: f32, height: f32) {
        self.hills.push((beta2, height));
    }
    
    /// Compute bias potential at given β₂
    pub fn potential(&self, beta2: f32) -> f32 {
        self.hills.iter()
            .map(|(center, height)| {
                let diff = beta2 - center;
                height * (-diff * diff / (2.0 * self.width * self.width)).exp()
            })
            .sum()
    }
    
    /// Compute bias force (negative gradient)
    pub fn force(&self, beta2: f32) -> f32 {
        self.hills.iter()
            .map(|(center, height)| {
                let diff = beta2 - center;
                let gaussian = (-diff * diff / (2.0 * self.width * self.width)).exp();
                height * gaussian * diff / (self.width * self.width)
            })
            .sum()
    }
}

/// Extended sampling result with additional analysis
#[derive(Debug, Clone)]
pub struct ExtendedSamplingResult {
    /// Base result
    pub base: NovaSamplingResult,
    
    /// Persistence features per sample
    pub persistence_history: Vec<PersistenceFeatures>,
    
    /// Best pocket-opening conformations (indices)
    pub best_pocket_indices: Vec<usize>,
    
    /// Bias potential history
    pub bias_history: Vec<f32>,
    
    /// Temperature history
    pub temperature_history: Vec<f32>,
    
    /// Sampling phase boundaries
    pub phase_boundaries: SamplingPhases,
}

#[derive(Debug, Clone, Default)]
pub struct SamplingPhases {
    pub exploration_end: usize,
    pub biased_end: usize,
    pub refinement_end: usize,
}

/// Extended NOVA sampler
pub struct ExtendedNovaSampler {
    base_sampler: NovaCrypticSampler,
    persistence: PersistenceComputer,
    config: ExtendedNovaConfig,
    bias: MetadynamicsBias,
}

impl ExtendedNovaSampler {
    pub fn new(context: Arc<CudaContext>, config: ExtendedNovaConfig) -> Result<Self> {
        let base_sampler = NovaCrypticSampler::new(Arc::clone(&context))?
            .with_config(config.base.clone());
        
        let persistence = PersistenceComputer::new(context)?;
        let bias = MetadynamicsBias::new(config.hill_width);
        
        Ok(Self {
            base_sampler,
            persistence,
            config,
            bias,
        })
    }
    
    /// Load structure
    pub fn load_structure(&mut self, pdb_content: &str) -> Result<()> {
        self.base_sampler.load_structure(pdb_content)
    }
    
    /// Run extended sampling with adaptive biasing
    pub fn sample(&mut self) -> Result<ExtendedSamplingResult> {
        let mut conformations = Vec::with_capacity(self.config.total_samples);
        let mut betti_history = Vec::with_capacity(self.config.total_samples);
        let mut persistence_history = Vec::with_capacity(self.config.total_samples);
        let mut energy_history = Vec::with_capacity(self.config.total_samples);
        let mut bias_history = Vec::with_capacity(self.config.total_samples);
        let mut temperature_history = Vec::with_capacity(self.config.total_samples);
        
        let start_time = std::time::Instant::now();
        let mut current_temp = self.config.base.temperature;
        
        log::info!("Starting extended sampling: {} conformations", self.config.total_samples);
        
        for sample_idx in 0..self.config.total_samples {
            // Phase determination
            let phase = self.determine_phase(sample_idx);
            
            // Temperature annealing
            if self.config.temperature_annealing && sample_idx >= self.config.annealing_start {
                let progress = (sample_idx - self.config.annealing_start) as f32 
                    / (self.config.total_samples - self.config.annealing_start) as f32;
                current_temp = self.config.base.temperature 
                    + progress * (self.config.final_temperature - self.config.base.temperature);
                self.base_sampler.set_temperature(current_temp)?;
            }
            temperature_history.push(current_temp);
            
            // Run sampling steps
            let step_result = self.base_sampler.step_n(self.config.base.steps_per_sample)?;
            
            // Get conformation
            let coords = self.base_sampler.get_ca_coords()?;
            
            // Compute persistence features
            let pers_features = self.persistence.compute_features(&coords)?;
            let beta2 = pers_features.betti_2.count as f32;
            
            // Apply/update bias
            let current_bias = if self.config.adaptive_bias && phase == SamplingPhase::Biased {
                // Deposit hill if at local β₂ minimum
                if sample_idx % self.config.hill_frequency == 0 && beta2 < 1.0 {
                    self.bias.deposit_hill(beta2, self.config.hill_height);
                }
                self.bias.potential(beta2)
            } else {
                0.0
            };
            bias_history.push(current_bias);
            
            // Record
            conformations.push(coords);
            betti_history.push([1, 0, pers_features.betti_2.count as i32]);
            persistence_history.push(pers_features);
            energy_history.push(step_result.energy);
            
            // Progress logging
            if (sample_idx + 1) % 200 == 0 {
                log::info!("Sample {}/{}: phase={:?}, T={:.0}K, β₂={}, bias={:.2}",
                           sample_idx + 1, self.config.total_samples,
                           phase, current_temp, beta2 as i32, current_bias);
            }
        }
        
        // Find best pocket-opening conformations
        let best_indices = self.find_best_pocket_conformations(&persistence_history);
        
        let elapsed = start_time.elapsed();
        log::info!("Extended sampling complete: {} samples in {:.1}s",
                   self.config.total_samples, elapsed.as_secs_f32());
        
        // Build base result
        let base = NovaSamplingResult {
            conformations: conformations.clone(),
            full_conformations: vec![], // Not storing full atoms
            betti_history,
            energy_history,
            efe_history: vec![],
            acceptance_rates: vec![],
            quality_metrics: None,
            stats: Default::default(),
        };
        
        Ok(ExtendedSamplingResult {
            base,
            persistence_history,
            best_pocket_indices: best_indices,
            bias_history,
            temperature_history,
            phase_boundaries: SamplingPhases {
                exploration_end: self.config.exploration_samples,
                biased_end: self.config.annealing_start,
                refinement_end: self.config.total_samples,
            },
        })
    }
    
    fn determine_phase(&self, sample_idx: usize) -> SamplingPhase {
        if sample_idx < self.config.exploration_samples {
            SamplingPhase::Exploration
        } else if sample_idx < self.config.annealing_start {
            SamplingPhase::Biased
        } else {
            SamplingPhase::Refinement
        }
    }
    
    fn find_best_pocket_conformations(&self, history: &[PersistenceFeatures]) -> Vec<usize> {
        // Score each conformation by pocket-opening potential
        let mut scored: Vec<(usize, f32)> = history.iter()
            .enumerate()
            .map(|(i, p)| {
                let score = p.betti_2.count as f32 * 10.0
                    + p.betti_2.total_persistence
                    + p.void_volume_proxy * 0.1
                    + p.pocket_opening_score;
                (i, score)
            })
            .collect();
        
        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top 10%
        let top_count = (history.len() / 10).max(10);
        scored.into_iter()
            .take(top_count)
            .map(|(i, _)| i)
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SamplingPhase {
    Exploration,
    Biased,
    Refinement,
}
```

---

### 3.4 Multi-Scale Feature Extraction

#### Objective

Extract features at local (residue), regional (secondary structure), and global (domain) scales.

#### Feature Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                 MULTI-SCALE FEATURES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scale 1: Local (per-residue) - 16 features                    │
│  ├── Existing Phase 6 features                                 │
│  └── Same as CrypticFeatures struct                            │
│                                                                 │
│  Scale 2: Regional (5Å neighborhood) - 12 features             │
│  ├── Neighbor count (within 5Å, 8Å, 12Å)                       │
│  ├── Regional RMSF (mean, std of neighbors)                    │
│  ├── Regional charge distribution                              │
│  ├── Secondary structure context                               │
│  ├── Regional hydrophobic moment                               │
│  └── Local packing density                                     │
│                                                                 │
│  Scale 3: Global (whole protein) - 8 features                  │
│  ├── Distance to centroid                                      │
│  ├── Distance to surface                                       │
│  ├── Relative position in sequence                             │
│  ├── Domain membership (if multi-domain)                       │
│  ├── Global flexibility rank                                   │
│  └── Allosteric network centrality                             │
│                                                                 │
│  Total: 36 features (16 + 12 + 8)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation in**: `crates/prism-validation/src/multiscale_features.rs`

```rust
//! Multi-Scale Feature Extraction
//!
//! Extracts features at three scales:
//! - Local: per-residue properties
//! - Regional: neighborhood context
//! - Global: whole-protein context

use crate::cryptic_features::CrypticFeatures;
use crate::pdb_sanitizer::SanitizedStructure;

/// Regional features (5-12Å neighborhood)
#[derive(Debug, Clone, Default)]
pub struct RegionalFeatures {
    // Neighbor counts at different radii
    pub neighbors_5a: f32,
    pub neighbors_8a: f32,
    pub neighbors_12a: f32,
    
    // Regional dynamics
    pub regional_rmsf_mean: f32,
    pub regional_rmsf_std: f32,
    
    // Regional chemistry
    pub regional_charge: f32,
    pub regional_hydrophobicity: f32,
    pub hydrophobic_moment: f32,
    
    // Secondary structure context
    pub helix_fraction: f32,
    pub sheet_fraction: f32,
    pub coil_fraction: f32,
    
    // Packing
    pub packing_density: f32,
}

/// Global features (whole protein context)
#[derive(Debug, Clone, Default)]
pub struct GlobalFeatures {
    // Spatial position
    pub distance_to_centroid: f32,
    pub distance_to_surface: f32,
    pub relative_sequence_position: f32,
    
    // Domain context
    pub domain_id: f32,
    pub inter_domain_interface: f32,
    
    // Network properties
    pub flexibility_rank: f32,
    pub contact_network_centrality: f32,
    pub allosteric_coupling: f32,
}

/// Combined multi-scale features
#[derive(Debug, Clone, Default)]
pub struct MultiScaleFeatures {
    pub local: CrypticFeatures,      // 16 features
    pub regional: RegionalFeatures,  // 12 features
    pub global: GlobalFeatures,      // 8 features
}

impl MultiScaleFeatures {
    /// Total feature count
    pub const TOTAL_FEATURES: usize = 36;
    
    /// Convert to flat array
    pub fn to_array(&self) -> [f32; 36] {
        let mut arr = [0.0f32; 36];
        
        // Local (0-15)
        let mut local_buf = [0.0f32; 40];
        self.local.encode_into(&mut local_buf);
        arr[0..16].copy_from_slice(&local_buf[0..16]);
        
        // Regional (16-27)
        arr[16] = self.regional.neighbors_5a;
        arr[17] = self.regional.neighbors_8a;
        arr[18] = self.regional.neighbors_12a;
        arr[19] = self.regional.regional_rmsf_mean;
        arr[20] = self.regional.regional_rmsf_std;
        arr[21] = self.regional.regional_charge;
        arr[22] = self.regional.regional_hydrophobicity;
        arr[23] = self.regional.hydrophobic_moment;
        arr[24] = self.regional.helix_fraction;
        arr[25] = self.regional.sheet_fraction;
        arr[26] = self.regional.coil_fraction;
        arr[27] = self.regional.packing_density;
        
        // Global (28-35)
        arr[28] = self.global.distance_to_centroid;
        arr[29] = self.global.distance_to_surface;
        arr[30] = self.global.relative_sequence_position;
        arr[31] = self.global.domain_id;
        arr[32] = self.global.inter_domain_interface;
        arr[33] = self.global.flexibility_rank;
        arr[34] = self.global.contact_network_centrality;
        arr[35] = self.global.allosteric_coupling;
        
        arr
    }
}

/// Multi-scale feature extractor
pub struct MultiScaleExtractor {
    // Precomputed structure properties
    centroid: [f32; 3],
    contact_matrix: Vec<Vec<bool>>,
    surface_residues: Vec<bool>,
}

impl MultiScaleExtractor {
    pub fn new(structure: &SanitizedStructure) -> Self {
        let centroid = Self::compute_centroid(structure);
        let contact_matrix = Self::compute_contacts(structure, 8.0);
        let surface_residues = Self::identify_surface(structure);
        
        Self {
            centroid,
            contact_matrix,
            surface_residues,
        }
    }
    
    /// Extract multi-scale features for a residue
    pub fn extract(
        &self,
        residue_idx: usize,
        structure: &SanitizedStructure,
        local_features: &CrypticFeatures,
        ensemble_rmsf: &[f32],
    ) -> MultiScaleFeatures {
        let regional = self.extract_regional(residue_idx, structure, ensemble_rmsf);
        let global = self.extract_global(residue_idx, structure, ensemble_rmsf);
        
        MultiScaleFeatures {
            local: local_features.clone(),
            regional,
            global,
        }
    }
    
    fn extract_regional(
        &self,
        residue_idx: usize,
        structure: &SanitizedStructure,
        ensemble_rmsf: &[f32],
    ) -> RegionalFeatures {
        // Count neighbors at different radii
        // Compute regional statistics
        // ...
        RegionalFeatures::default()
    }
    
    fn extract_global(
        &self,
        residue_idx: usize,
        structure: &SanitizedStructure,
        ensemble_rmsf: &[f32],
    ) -> GlobalFeatures {
        // Distance to centroid
        // Distance to surface
        // Network centrality
        // ...
        GlobalFeatures::default()
    }
    
    fn compute_centroid(structure: &SanitizedStructure) -> [f32; 3] {
        let coords = structure.get_ca_coords();
        let n = coords.len() as f32;
        let sum: [f32; 3] = coords.iter()
            .fold([0.0, 0.0, 0.0], |acc, c| {
                [acc[0] + c[0], acc[1] + c[1], acc[2] + c[2]]
            });
        [sum[0] / n, sum[1] / n, sum[2] / n]
    }
    
    fn compute_contacts(structure: &SanitizedStructure, cutoff: f32) -> Vec<Vec<bool>> {
        let coords = structure.get_ca_coords();
        let n = coords.len();
        let cutoff_sq = cutoff * cutoff;
        
        let mut matrix = vec![vec![false; n]; n];
        for i in 0..n {
            for j in (i+1)..n {
                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                let dist_sq = dx*dx + dy*dy + dz*dz;
                
                if dist_sq < cutoff_sq {
                    matrix[i][j] = true;
                    matrix[j][i] = true;
                }
            }
        }
        matrix
    }
    
    fn identify_surface(structure: &SanitizedStructure) -> Vec<bool> {
        // Simple heuristic: residues with few contacts are surface
        let contacts = Self::compute_contacts(structure, 10.0);
        contacts.iter()
            .map(|row| row.iter().filter(|&&b| b).count() < 12)
            .collect()
    }
}
```

---

## Week 1-4 Checklist (Phase 7 Part 1)

```
Week 1-2: Hierarchical Reservoir
□ hierarchical_reservoir.rs compiles
□ Cortical column struct implemented
□ L1 → L2 → L3 forward pass works
□ Lateral inhibition functional
□ CUDA kernels for each layer
□ Unit tests pass
□ Benchmark: 1,280 neurons < 5ms/step

Week 3-4: Persistence + Multi-Scale
□ persistent_homology.rs compiles
□ Alpha complex construction works
□ Persistence computation correct (test on known shapes)
□ 31 persistence features extracted
□ multiscale_features.rs compiles
□ Regional features computed correctly
□ Global features computed correctly
□ Total: 36 + 31 = 67 new features
```

*[Continued in Part 2: Phase 7 Weeks 5-8 and Phase 8]*
