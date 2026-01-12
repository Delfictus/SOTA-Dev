# PRISM Phase 6: Complete Implementation Plan
## Cryptic Site Detection SOTA Achievement

**Document Version**: 2.0 (Consolidated)
**Generated**: 2026-01-11
**Status**: APPROVED FOR EXECUTION
**Timeline**: 8 Weeks
**Classification**: Zero Fallback / Zero Mock Implementation

---

## Table of Contents

1. Executive Summary
2. Success Criteria (Non-Negotiable)
3. Week 0: Pre-Implementation Setup
4. Weeks 1-2: GPU SNN Scale-Up
5. Weeks 3-4: PRISM-NOVA Integration
6. Weeks 5-6: CryptoBench & Ablation
7. Weeks 7-8: Publication & Analysis
8. File Manifest
9. Verification Commands
10. Risk Mitigation
11. Appendix: Quick Reference & Task List

---

## 1. Executive Summary

### Objective

Achieve **publication-ready** cryptic site detection that **exceeds SOTA** using **ONLY native PRISM infrastructure**.

### Target Metrics

| Metric | Phase 5 Baseline | Target | Minimum | SOTA Reference |
|--------|------------------|--------|---------|----------------|
| ROC AUC | 0.487 | **>0.75** | 0.70 | PocketMiner 0.87 |
| PR AUC | 0.081 | **>0.25** | 0.20 | CryptoBank 0.17 |
| Success Rate | 71.7% | **>85%** | 80% | Schrodinger 83% |
| Top-1 Accuracy | 82.6% | **>90%** | 85% | CrypTothML 78% |
| Time/Structure | N/A | **<1s** | <5s | RTX 3060 |
| Peak VRAM | N/A | **<2GB** | <4GB | RTX 3060 |
| Apo-Holo Recovery | N/A | **<2.5A** | <3.5A | Min RMSD to holo |
| Ensemble Diversity | N/A | **1-3A** | 0.5-5A | Mean pairwise RMSD |

### Key Deliverables

1. **GPU DendriticSNNReservoir** (512 neurons, RLS online learning)
2. **PRISM-NOVA Sampler** (Neural HMC, TDA-guided, Active Inference)
3. **CryptoBench Validation** (1107 structures, 885 train / 222 test)
4. **Ablation Study** (6 variants proving component contributions)
5. **Apo-Holo Benchmark** (15 classic pairs demonstrating conformational prediction)
6. **Failure Case Analysis** (categorized limitations)
7. **Publication Package** (LaTeX tables, figures, methods section)

### Architectural Constraints

```
REQUIRED:
   - PRISM-ZrO (SNN + RLS) for adaptive learning
   - PRISM-NOVA (HMC + AMBER) for enhanced sampling
   - Native Rust/CUDA implementations
   - GPU-mandatory execution (no silent CPU fallback)
   - Explicit error on missing GPU

FORBIDDEN:
   - PyTorch, TensorFlow, or external ML models
   - Silent fallback to CPU (must fail explicitly)
   - Mock implementations or placeholder returns
   - Data leakage between train/test splits
   - Metric regression from Phase 5
   - todo!() or unimplemented!() in production code
```

---

## 2. Success Criteria (Non-Negotiable)

### Primary Metrics

| Criterion | Target | Minimum | Measurement Method |
|-----------|--------|---------|-------------------|
| ROC AUC | >0.75 | 0.70 | CryptoBench test set (222 structures) |
| PR AUC | >0.25 | 0.20 | CryptoBench test set |
| Success Rate | >85% | 80% | Overlap >=30% with ground truth |
| Top-1 Accuracy | >90% | 85% | Predicted site within 8A of true site |
| GPU Performance | <1s/structure | <5s | RTX 3060, averaged over test set |
| Memory | <2GB | <4GB | Peak VRAM during inference |

### Secondary Metrics (Validation Quality)

| Criterion | Target | Rationale |
|-----------|--------|-----------|
| Apo-Holo Recovery | <2.5A | Proves conformational prediction capability |
| Ensemble Diversity | 1-3A mean pairwise RMSD | Validates sampling quality |
| Ablation Delta | >+0.20 AUC vs ANM-only | Proves component contributions |
| RLS Stability | No NaN/Inf over 10k updates | Numerical robustness |

### Zero Fallback Verification Test

```bash
# This command MUST FAIL if no GPU is present
CUDA_VISIBLE_DEVICES="" cargo test --release -p prism-validation --features cuda gpu_scorer

# Expected output: Error - CUDA device not found
# If this passes, there is a hidden fallback - FIX IMMEDIATELY
```

### Zero Mock Verification Checklist

Every function must:
- [ ] Return real computed values (no hardcoded returns)
- [ ] Process actual input data (no ignored parameters)
- [ ] Interact with GPU hardware (no simulated responses)
- [ ] Have unit tests that verify real computation
- [ ] Log meaningful intermediate values

---

## 3. Week 0: Pre-Implementation Setup

**Duration**: 2-3 days before Week 1

### Task 0.1: Environment Verification

```bash
# Verify Rust toolchain
rustc --version  # Must be 1.75+
cargo --version

# Verify CUDA
nvcc --version   # Must be 12.0+
nvidia-smi       # Verify RTX 3060 visible

# Verify prism-gpu compiles
cd /path/to/prism-4d
cargo check -p prism-gpu --features cuda

# Run CUDA smoke test
cargo run --release -p prism-gpu --example test-cuda
```

**Pass Criteria**: All commands succeed, GPU detected with >6GB VRAM

### Task 0.2: Download CryptoBench Dataset

```bash
# Create directory structure
mkdir -p data/benchmarks/cryptobench/structures
mkdir -p data/benchmarks/cryptobench/manifests

# Clone CryptoBench repository
git clone https://github.com/skrhakv/CryptoBench.git /tmp/cryptobench

# Copy relevant data
cp -r /tmp/cryptobench/data/* data/benchmarks/cryptobench/

# Verify structure count
find data/benchmarks/cryptobench/structures -name "*.pdb" | wc -l
# Expected: ~1107 files

# Create manifest with proper train/test split
python scripts/create_cryptobench_manifest.py \
    --input data/benchmarks/cryptobench/ \
    --output data/benchmarks/cryptobench/manifest.json \
    --train-ratio 0.8 \
    --seed 42
```

### Task 0.3: Download Apo-Holo Pairs

```bash
# Create apo-holo directory
mkdir -p data/benchmarks/apo_holo

# Download 15 classic pairs from RCSB
APO_HOLO_PAIRS=(
    "1AKE:4AKE:Adenylate_kinase"
    "2LAO:1LST:Lysine_binding_protein"
    "1GGG:1WDN:Calmodulin"
    "1OMP:1ANF:Maltose_binding_protein"
    "1RX2:1RX4:Ribonuclease"
    "3CHY:2CHE:CheY"
    "1EX6:1EX7:Galectin"
    "1STP:1SWB:Streptavidin"
    "1AJJ:1AJK:Guanylate_kinase"
    "1PHP:1PHN:Phosphotransferase"
    "1BTL:1BTM:Beta_lactamase"
    "2CPL:1CWA:Cyclophilin"
    "1BMD:1BMC:Biotin_binding_protein"
    "1URN:1URP:Ubiquitin"
    "1HOE:1HOF:Alpha_amylase_inhibitor"
)

for pair in "${APO_HOLO_PAIRS[@]}"; do
    IFS=':' read -r apo holo name <<< "$pair"
    wget -q "https://files.rcsb.org/download/${apo}.pdb" \
         -O "data/benchmarks/apo_holo/${apo}_apo.pdb"
    wget -q "https://files.rcsb.org/download/${holo}.pdb" \
         -O "data/benchmarks/apo_holo/${holo}_holo.pdb"
    echo "Downloaded: $name ($apo -> $holo)"
done
```

### Task 0.4: Document Phase 5 Baseline

Create `results/BASELINE_METRICS.md`:

```markdown
# Phase 5 Baseline Metrics (Pre-Phase 6)

**Date**: [DATE]
**Commit**: [COMMIT_HASH]

## CryptoBench Subset Results (46 structures)

| Metric | Value | Notes |
|--------|-------|-------|
| ROC AUC | 0.487 | Near random-level |
| PR AUC | 0.081 | Very low precision |
| Success Rate | 71.7% | Overlap >=30% |
| Top-1 Accuracy | 82.6% | Any overlap |

## Root Causes Identified

1. **Harmonic-only sampling (ANM)**: Cannot cross energy barriers
2. **CPU-only SNN (64 neurons)**: Underutilized capacity
3. **No TDA integration**: Missing void formation signals
4. **No online learning**: No adaptation to structure
5. **Small dataset**: Only 46 structures

## Phase 6 Expected Improvements

| Component | Expected Gain | Mechanism |
|-----------|---------------|-----------|
| GPU SNN (512 neurons) | +0.05 AUC | Better feature processing |
| PRISM-NOVA sampling | +0.10 AUC | Anharmonic motions |
| TDA integration | +0.05 AUC | Void detection |
| RLS learning | +0.05 AUC | Per-structure adaptation |
| Full CryptoBench | +0.05 AUC | Better training signal |
| **Total** | **+0.30 AUC** | **0.49 -> 0.79** |
```

### Task 0.5: Git Setup & CI Configuration

```bash
# Create feature branch
git checkout -b feature/phase-6-cryptic-sota
git push -u origin feature/phase-6-cryptic-sota

# Create CI workflow for metric regression
cat > .github/workflows/phase6-metrics.yml << 'EOF'
name: Phase 6 Metric Validation

on:
  push:
    branches: [feature/phase-6-cryptic-sota]
  pull_request:
    branches: [main]

jobs:
  validate-metrics:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3

      - name: Run metric validation
        run: |
          cargo test --release -p prism-validation --features cuda

      - name: Check for regression
        run: |
          cargo run --release -p prism-validation --bin check-regression -- \
            --baseline results/BASELINE_METRICS.json \
            --current results/current_metrics.json
EOF
```

### Week 0 Checklist

```
[ ] Rust 1.75+ installed and verified
[ ] CUDA 12.0+ installed and verified
[ ] prism-gpu compiles with cuda feature
[ ] CryptoBench dataset downloaded (1107 structures)
[ ] manifest.json created with 885/222 train/test split
[ ] Apo-holo pairs downloaded (15 pairs, 30 PDB files)
[ ] Phase 5 baseline documented with commit hash
[ ] Git branch created: feature/phase-6-cryptic-sota
[ ] CI pipeline configured for metric regression testing
[ ] Test structure (3CSY Ebola GP) downloads successfully
```

---

## 4. Weeks 1-2: GPU SNN Scale-Up

### Objective

Replace CPU ZrO scorer with full GPU DendriticSNNReservoir (512 neurons)

### Component Specifications

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Reservoir Size | 512 neurons | Matches PRISM-ZrO architecture |
| Input Dimension | 40 (16 features + 16 velocities + 8 padding) | Feature Adapter Protocol |
| Topology | Structured sparse (~10% connectivity) | Echo state property |
| E/I Balance | 80% excitatory / 20% inhibitory | Biological plausibility |
| Time Constants | 5-50ms gradient | Temporal dynamics |
| VRAM Usage | ~1.2 MB for reservoir | Fits any modern GPU |
| RLS lambda | 0.99 | Forgetting factor |
| Precision Init | P = 100 * I | Initial precision matrix |
| Gradient Clamp | +/-1.0 | Stability |
| Max Precision Trace | 1e6 | Soft reset trigger |

### Task 1.1: Create CrypticFeatures Struct

**File**: `crates/prism-validation/src/cryptic_features.rs`

```rust
//! Cryptic site feature vector definition
//!
//! 16-dimensional feature vector capturing dynamics, structural,
//! chemical, distance, and tertiary properties of each residue.

use serde::{Deserialize, Serialize};

/// Cryptic site feature vector (16 dimensions)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrypticFeatures {
    // === Dynamics Features (5) ===
    /// Change in burial upon conformational sampling
    pub burial_change: f32,
    /// Root mean square fluctuation from ensemble
    pub rmsf: f32,
    /// Variance of position across ensemble
    pub variance: f32,
    /// Average flexibility of neighboring residues
    pub neighbor_flexibility: f32,
    /// Predicted burial potential (from ANM/NOVA)
    pub burial_potential: f32,

    // === Structural Features (3) ===
    /// Secondary structure flexibility score
    pub ss_flexibility: f32,
    /// Side chain rotamer flexibility
    pub sidechain_flexibility: f32,
    /// Crystallographic B-factor (normalized)
    pub b_factor: f32,

    // === Chemical Features (3) ===
    /// Net charge of residue
    pub net_charge: f32,
    /// Hydrophobicity (Kyte-Doolittle scale)
    pub hydrophobicity: f32,
    /// Hydrogen bonding potential
    pub h_bond_potential: f32,

    // === Distance Features (3) ===
    /// Local contact density (neighbors within 8A)
    pub contact_density: f32,
    /// Change in solvent accessible surface area
    pub sasa_change: f32,
    /// Distance to nearest charged residue
    pub nearest_charged_dist: f32,

    // === Tertiary Features (2) ===
    /// Interface score for multi-chain proteins
    pub interface_score: f32,
    /// Proximity to known allosteric sites
    pub allosteric_proximity: f32,
}

impl CrypticFeatures {
    /// Encode features into 40-dim input buffer
    /// Layout: [16 features][16 velocities][8 padding]
    pub fn encode_into(&self, buffer: &mut [f32; 40]) {
        buffer[0] = self.burial_change;
        buffer[1] = self.rmsf;
        buffer[2] = self.variance;
        buffer[3] = self.neighbor_flexibility;
        buffer[4] = self.burial_potential;
        buffer[5] = self.ss_flexibility;
        buffer[6] = self.sidechain_flexibility;
        buffer[7] = self.b_factor;
        buffer[8] = self.net_charge;
        buffer[9] = self.hydrophobicity;
        buffer[10] = self.h_bond_potential;
        buffer[11] = self.contact_density;
        buffer[12] = self.sasa_change;
        buffer[13] = self.nearest_charged_dist;
        buffer[14] = self.interface_score;
        buffer[15] = self.allosteric_proximity;

        // Velocity slots (16-31) - set by encode_with_velocity
        // Padding slots (32-39) - zeros
        for i in 16..40 {
            buffer[i] = 0.0;
        }
    }

    /// Encode with velocity information from previous frame
    pub fn encode_with_velocity(&self, prev: &CrypticFeatures, buffer: &mut [f32; 40]) {
        self.encode_into(buffer);

        // Compute velocities (deltas from previous)
        buffer[16] = self.burial_change - prev.burial_change;
        buffer[17] = self.rmsf - prev.rmsf;
        buffer[18] = self.variance - prev.variance;
        buffer[19] = self.neighbor_flexibility - prev.neighbor_flexibility;
        buffer[20] = self.burial_potential - prev.burial_potential;
        buffer[21] = self.ss_flexibility - prev.ss_flexibility;
        buffer[22] = self.sidechain_flexibility - prev.sidechain_flexibility;
        buffer[23] = self.b_factor - prev.b_factor;
        buffer[24] = self.net_charge - prev.net_charge;
        buffer[25] = self.hydrophobicity - prev.hydrophobicity;
        buffer[26] = self.h_bond_potential - prev.h_bond_potential;
        buffer[27] = self.contact_density - prev.contact_density;
        buffer[28] = self.sasa_change - prev.sasa_change;
        buffer[29] = self.nearest_charged_dist - prev.nearest_charged_dist;
        buffer[30] = self.interface_score - prev.interface_score;
        buffer[31] = self.allosteric_proximity - prev.allosteric_proximity;
    }

    /// Create from raw array (for testing)
    pub fn from_array(arr: &[f32; 16]) -> Self {
        Self {
            burial_change: arr[0],
            rmsf: arr[1],
            variance: arr[2],
            neighbor_flexibility: arr[3],
            burial_potential: arr[4],
            ss_flexibility: arr[5],
            sidechain_flexibility: arr[6],
            b_factor: arr[7],
            net_charge: arr[8],
            hydrophobicity: arr[9],
            h_bond_potential: arr[10],
            contact_density: arr[11],
            sasa_change: arr[12],
            nearest_charged_dist: arr[13],
            interface_score: arr[14],
            allosteric_proximity: arr[15],
        }
    }

    /// Normalize features to [0, 1] range
    pub fn normalize(&mut self) {
        // Apply sigmoid to unbounded features
        self.burial_change = sigmoid(self.burial_change);
        self.variance = sigmoid(self.variance);
        self.neighbor_flexibility = sigmoid(self.neighbor_flexibility);

        // Clamp bounded features
        self.rmsf = self.rmsf.clamp(0.0, 10.0) / 10.0;
        self.b_factor = self.b_factor.clamp(0.0, 100.0) / 100.0;
        self.contact_density = self.contact_density.clamp(0.0, 30.0) / 30.0;
        self.sasa_change = sigmoid(self.sasa_change);
        self.nearest_charged_dist = self.nearest_charged_dist.clamp(0.0, 20.0) / 20.0;
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_roundtrip() {
        let features = CrypticFeatures {
            burial_change: 0.5,
            rmsf: 1.2,
            net_charge: -1.0,
            ..Default::default()
        };

        let mut buffer = [0.0f32; 40];
        features.encode_into(&mut buffer);

        assert!((buffer[0] - 0.5).abs() < 1e-6);
        assert!((buffer[1] - 1.2).abs() < 1e-6);
        assert!((buffer[8] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_velocity_encoding() {
        let prev = CrypticFeatures {
            burial_change: 0.3,
            rmsf: 1.0,
            ..Default::default()
        };

        let curr = CrypticFeatures {
            burial_change: 0.5,
            rmsf: 1.5,
            ..Default::default()
        };

        let mut buffer = [0.0f32; 40];
        curr.encode_with_velocity(&prev, &mut buffer);

        // Check velocity slots
        assert!((buffer[16] - 0.2).abs() < 1e-6); // burial_change velocity
        assert!((buffer[17] - 0.5).abs() < 1e-6); // rmsf velocity
    }
}
```

### Task 1.2: Create GpuZroCrypticScorer

**File**: `crates/prism-validation/src/gpu_zro_cryptic_scorer.rs`

```rust
//! GPU-accelerated PRISM-ZrO cryptic site scorer
//!
//! Uses full 512-neuron DendriticSNNReservoir with Feature Adapter Protocol
//! and RLS (Recursive Least Squares) online learning.
//!
//! # Zero Fallback Policy
//! This module REQUIRES a valid CUDA context. It will NOT fall back to CPU.
//! If no GPU is available, initialization will fail with an explicit error.

use anyhow::{Result, Context, bail};
use std::sync::Arc;
use cudarc::driver::CudaContext;
use prism_gpu::{DendriticSNNReservoir, SNN_INPUT_DIM};

use crate::cryptic_features::CrypticFeatures;

/// GPU-accelerated cryptic site scorer using PRISM-ZrO architecture
pub struct GpuZroCrypticScorer {
    /// 512-neuron dendritic SNN reservoir (GPU)
    reservoir: DendriticSNNReservoir,

    /// RLS readout weights [512] -> single score
    readout_weights: Vec<f32>,

    /// RLS precision matrix [512 x 512]
    precision_matrix: Vec<f32>,

    /// Forgetting factor (0.99)
    lambda: f32,

    /// Number of RLS updates performed
    update_count: usize,

    /// Previous features for velocity computation
    prev_features: Option<CrypticFeatures>,

    /// Maximum allowed precision matrix trace (stability)
    max_precision_trace: f32,

    /// Gradient clamp value for stability
    gradient_clamp: f32,
}

impl GpuZroCrypticScorer {
    /// Initialize GPU scorer with CUDA context
    ///
    /// # Errors
    /// Returns error if CUDA context is invalid or GPU initialization fails.
    /// This is intentional - we do NOT fall back to CPU.
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        // Verify GPU is actually available
        let device_count = cudarc::driver::result::device::count()
            .context("Failed to query CUDA device count")?;

        if device_count == 0 {
            bail!("No CUDA devices found. GPU scorer requires a CUDA-capable GPU. \
                   This is NOT a fallback scenario - GPU is mandatory.");
        }

        let mut reservoir = DendriticSNNReservoir::new(context, 512)
            .context("Failed to create 512-neuron GPU reservoir")?;

        reservoir.initialize(42)
            .context("Failed to initialize reservoir weights")?;

        // Initialize readout weights to zero
        let readout_weights = vec![0.0f32; 512];

        // Initialize precision matrix to 100 * I
        let mut precision_matrix = vec![0.0f32; 512 * 512];
        for i in 0..512 {
            precision_matrix[i * 512 + i] = 100.0;
        }

        log::info!("GPU ZrO Cryptic Scorer initialized: 512 neurons, RLS enabled");

        Ok(Self {
            reservoir,
            readout_weights,
            precision_matrix,
            lambda: 0.99,
            update_count: 0,
            prev_features: None,
            max_precision_trace: 1e6,
            gradient_clamp: 1.0,
        })
    }

    /// Process cryptic features through GPU reservoir (inference only)
    pub fn score_residue(&mut self, features: &CrypticFeatures) -> Result<f32> {
        let mut input = [0.0f32; 40];

        if let Some(ref prev) = self.prev_features {
            features.encode_with_velocity(prev, &mut input);
        } else {
            features.encode_into(&mut input);
        }

        self.prev_features = Some(features.clone());

        // Process through GPU reservoir
        let state = self.reservoir.process_features(&input)
            .context("GPU reservoir processing failed")?;

        // Compute score via readout weights
        let raw_score: f32 = state.iter()
            .zip(&self.readout_weights)
            .map(|(s, w)| s * w)
            .sum();

        Ok(sigmoid(raw_score))
    }

    /// Score with RLS online learning from ground truth label
    pub fn score_and_learn(
        &mut self,
        features: &CrypticFeatures,
        ground_truth: bool,
    ) -> Result<f32> {
        let mut input = [0.0f32; 40];

        if let Some(ref prev) = self.prev_features {
            features.encode_with_velocity(prev, &mut input);
        } else {
            features.encode_into(&mut input);
        }

        self.prev_features = Some(features.clone());

        // Process through GPU reservoir
        let state = self.reservoir.process_features(&input)
            .context("GPU reservoir processing failed")?;

        // Compute current prediction
        let raw_score: f32 = state.iter()
            .zip(&self.readout_weights)
            .map(|(s, w)| s * w)
            .sum();

        let prediction = sigmoid(raw_score);
        let target = if ground_truth { 1.0 } else { 0.0 };

        // Perform RLS update
        self.rls_update(&state, target)
            .context("RLS weight update failed")?;

        Ok(prediction)
    }

    /// Sherman-Morrison RLS update with stability safeguards
    fn rls_update(&mut self, state: &[f32], target: f32) -> Result<()> {
        let n = 512;
        let k = state;

        // Compute P * k
        let mut pk = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                pk[i] += self.precision_matrix[i * n + j] * k[j];
            }
        }

        // Compute k' * P * k
        let kpk: f32 = k.iter().zip(&pk).map(|(ki, pki)| ki * pki).sum();

        // Compute gain with numerical stability
        let gain = 1.0 / (self.lambda + kpk + 1e-8);

        // Update precision matrix: P = (1/lambda)(P - gain * pk * pk')
        let inv_lambda = 1.0 / self.lambda;
        for i in 0..n {
            for j in 0..n {
                self.precision_matrix[i * n + j] = inv_lambda * (
                    self.precision_matrix[i * n + j] - gain * pk[i] * pk[j]
                );
            }
        }

        // Compute prediction error with gradient clamp
        let prediction: f32 = k.iter()
            .zip(&self.readout_weights)
            .map(|(ki, wi)| ki * wi)
            .sum();
        let error = (target - sigmoid(prediction))
            .clamp(-self.gradient_clamp, self.gradient_clamp);

        // Update weights: w = w + P * k * error
        for i in 0..n {
            let delta = pk[i] * gain * error;
            self.readout_weights[i] += delta;

            // Clamp weights for stability
            self.readout_weights[i] = self.readout_weights[i].clamp(-10.0, 10.0);
        }

        self.update_count += 1;

        // Periodic stability check
        if self.update_count % 100 == 0 {
            self.stability_check()?;
        }

        Ok(())
    }

    /// Check and fix numerical stability issues
    fn stability_check(&mut self) -> Result<()> {
        // Check for NaN/Inf in weights
        if self.readout_weights.iter().any(|w| !w.is_finite()) {
            log::warn!("NaN/Inf detected in weights, resetting");
            self.readout_weights.fill(0.0);
        }

        // Check precision matrix trace
        let trace: f32 = (0..512).map(|i| self.precision_matrix[i * 512 + i]).sum();

        if trace > self.max_precision_trace {
            log::warn!("Precision matrix trace ({:.2e}) exceeded threshold, soft reset", trace);
            self.soft_reset_precision();
        }

        if !trace.is_finite() {
            log::error!("Precision matrix contains NaN/Inf, full reset");
            self.reset()?;
        }

        Ok(())
    }

    /// Soft reset precision matrix while preserving learned weights
    fn soft_reset_precision(&mut self) {
        for i in 0..512 {
            for j in 0..512 {
                if i == j {
                    self.precision_matrix[i * 512 + j] = 10.0;
                } else {
                    self.precision_matrix[i * 512 + j] = 0.0;
                }
            }
        }
    }

    /// Full reset (weights and precision matrix)
    pub fn reset(&mut self) -> Result<()> {
        self.readout_weights.fill(0.0);
        for i in 0..512 {
            for j in 0..512 {
                self.precision_matrix[i * 512 + j] = if i == j { 100.0 } else { 0.0 };
            }
        }
        self.update_count = 0;
        self.prev_features = None;

        log::debug!("GPU scorer reset complete");
        Ok(())
    }

    /// Reset state for new structure (keep weights, clear state)
    pub fn reset_for_structure(&mut self) {
        self.prev_features = None;
        self.reservoir.reset_state().ok();
    }

    /// Save learned weights to file
    pub fn save_weights(&self, path: &str) -> Result<()> {
        let data = bincode::serialize(&self.readout_weights)
            .context("Failed to serialize weights")?;
        std::fs::write(path, data)
            .context("Failed to write weights file")?;

        log::info!("Saved weights to {} ({} updates)", path, self.update_count);
        Ok(())
    }

    /// Load pre-trained weights from file
    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        let data = std::fs::read(path)
            .context("Failed to read weights file")?;
        self.readout_weights = bincode::deserialize(&data)
            .context("Failed to deserialize weights")?;

        if self.readout_weights.len() != 512 {
            bail!("Weight dimension mismatch: expected 512, got {}",
                  self.readout_weights.len());
        }

        log::info!("Loaded weights from {}", path);
        Ok(())
    }

    /// Get number of RLS updates performed
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    /// Get current weight statistics for logging
    pub fn weight_stats(&self) -> WeightStats {
        let mean = self.readout_weights.iter().sum::<f32>() / 512.0;
        let var = self.readout_weights.iter()
            .map(|w| (w - mean).powi(2))
            .sum::<f32>() / 512.0;
        let max = self.readout_weights.iter().cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min = self.readout_weights.iter().cloned()
            .fold(f32::INFINITY, f32::min);

        WeightStats { mean, std: var.sqrt(), min, max }
    }
}

#[derive(Debug, Clone)]
pub struct WeightStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
```

### Task 1.3: Create GPU Scorer Tests

**File**: `crates/prism-validation/src/tests/gpu_scorer_tests.rs`

```rust
//! GPU Scorer Integration Tests
//!
//! Tests verify:
//! 1. No CPU fallback (must fail without GPU)
//! 2. RLS stability over many updates
//! 3. Correct feature encoding
//! 4. Weight persistence
//! 5. Performance benchmarks

use crate::gpu_zro_cryptic_scorer::*;
use crate::cryptic_features::CrypticFeatures;
use std::sync::Arc;
use cudarc::driver::CudaContext;

/// CRITICAL TEST: Verify no CPU fallback exists
#[test]
fn test_no_cpu_fallback() {
    // Save current CUDA_VISIBLE_DEVICES
    let old_val = std::env::var("CUDA_VISIBLE_DEVICES").ok();

    // Hide all GPUs
    std::env::set_var("CUDA_VISIBLE_DEVICES", "");

    // Attempt to create context - should fail
    let result = CudaContext::new(0);

    // Restore CUDA_VISIBLE_DEVICES
    match old_val {
        Some(val) => std::env::set_var("CUDA_VISIBLE_DEVICES", val),
        None => std::env::remove_var("CUDA_VISIBLE_DEVICES"),
    }

    // THIS MUST FAIL - if it succeeds, we have a hidden CPU fallback
    assert!(result.is_err(),
        "CRITICAL: CudaContext creation should fail without GPU. \
         If this test fails, there is a hidden CPU fallback that violates \
         the Zero Fallback requirement.");
}

/// Test RLS stability over 1000 updates
#[test]
fn test_rls_stability_1000_updates() {
    let context = match CudaContext::new(0) {
        Ok(ctx) => Arc::new(ctx),
        Err(_) => {
            eprintln!("Skipping test_rls_stability - no GPU available");
            return;
        }
    };

    let mut scorer = GpuZroCrypticScorer::new(context).unwrap();

    let mut scores = Vec::new();
    for i in 0..1000 {
        let features = CrypticFeatures {
            burial_change: (i as f32 * 0.01).sin(),
            rmsf: (i as f32 * 0.02).cos() + 1.0,
            variance: 0.5,
            sasa_change: (i as f32 * 0.03).sin() * 0.5,
            ..Default::default()
        };

        let target = i % 3 == 0; // 33% positive rate
        let score = scorer.score_and_learn(&features, target).unwrap();

        scores.push(score);

        // Verify score validity
        assert!(score.is_finite(),
            "Score became NaN/Inf at iteration {}", i);
        assert!(score >= 0.0 && score <= 1.0,
            "Score {} out of [0,1] at iteration {}", score, i);
    }

    // Verify learning occurred
    let early_mean: f32 = scores[..100].iter().sum::<f32>() / 100.0;
    let late_mean: f32 = scores[900..].iter().sum::<f32>() / 100.0;
    let diff = (late_mean - early_mean).abs();

    assert!(diff > 0.01,
        "RLS learning appears stalled: early={:.4}, late={:.4}, diff={:.4}",
        early_mean, late_mean, diff);

    // Log weight statistics
    let stats = scorer.weight_stats();
    println!("After 1000 updates: mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]",
             stats.mean, stats.std, stats.min, stats.max);
}

/// Test RLS stability over 10000 updates (extended stress test)
#[test]
#[ignore] // Run with --ignored for extended tests
fn test_rls_stability_10000_updates() {
    let context = match CudaContext::new(0) {
        Ok(ctx) => Arc::new(ctx),
        Err(_) => {
            eprintln!("Skipping extended RLS test - no GPU");
            return;
        }
    };

    let mut scorer = GpuZroCrypticScorer::new(context).unwrap();

    for i in 0..10000 {
        let features = CrypticFeatures {
            burial_change: (i as f32 * 0.01).sin(),
            rmsf: (i as f32 * 0.02).cos().abs() + 0.5,
            variance: ((i as f32 * 0.005).sin() + 1.0) * 0.5,
            ..Default::default()
        };

        let target = (i % 5 < 2); // 40% positive rate
        let score = scorer.score_and_learn(&features, target).unwrap();

        assert!(score.is_finite(), "Score became NaN/Inf at iteration {}", i);

        if i % 1000 == 0 {
            let stats = scorer.weight_stats();
            println!("Iteration {}: mean={:.4}, std={:.4}", i, stats.mean, stats.std);
        }
    }

    assert_eq!(scorer.update_count(), 10000);
}

/// Test weight save/load roundtrip
#[test]
fn test_weight_persistence() {
    let context = match CudaContext::new(0) {
        Ok(ctx) => Arc::new(ctx),
        Err(_) => {
            eprintln!("Skipping weight persistence test - no GPU");
            return;
        }
    };

    let mut scorer1 = GpuZroCrypticScorer::new(Arc::clone(&context)).unwrap();

    // Train
    for i in 0..100 {
        let features = CrypticFeatures {
            burial_change: i as f32 * 0.1,
            rmsf: (i as f32 * 0.05).sin() + 1.0,
            ..Default::default()
        };
        scorer1.score_and_learn(&features, i % 2 == 0).unwrap();
    }

    // Save
    let tmp_path = "/tmp/prism_test_weights.bin";
    scorer1.save_weights(tmp_path).unwrap();

    // Load into new scorer
    let mut scorer2 = GpuZroCrypticScorer::new(context).unwrap();
    scorer2.load_weights(tmp_path).unwrap();

    // Verify same predictions
    for i in 0..10 {
        let features = CrypticFeatures {
            burial_change: i as f32 * 0.3,
            rmsf: 1.5,
            ..Default::default()
        };

        let score1 = scorer1.score_residue(&features).unwrap();
        // Reset state between scorers
        scorer1.reset_for_structure();
        scorer2.reset_for_structure();
        let score2 = scorer2.score_residue(&features).unwrap();

        assert!((score1 - score2).abs() < 1e-5,
            "Loaded weights produce different scores: {} vs {}", score1, score2);
    }

    // Cleanup
    std::fs::remove_file(tmp_path).ok();
}

/// Benchmark GPU scorer throughput
#[test]
fn bench_gpu_scorer_throughput() {
    let context = match CudaContext::new(0) {
        Ok(ctx) => Arc::new(ctx),
        Err(_) => {
            eprintln!("Skipping throughput benchmark - no GPU");
            return;
        }
    };

    let mut scorer = GpuZroCrypticScorer::new(context).unwrap();

    let features = CrypticFeatures {
        burial_change: 0.5,
        rmsf: 1.2,
        variance: 0.3,
        neighbor_flexibility: 0.4,
        contact_density: 5.0,
        ..Default::default()
    };

    // Warmup
    for _ in 0..100 {
        scorer.score_residue(&features).unwrap();
    }

    // Benchmark
    let n_iterations = 10000;
    let start = std::time::Instant::now();

    for _ in 0..n_iterations {
        scorer.score_residue(&features).unwrap();
    }

    let elapsed = start.elapsed();
    let throughput = n_iterations as f64 / elapsed.as_secs_f64();

    println!("GPU Scorer Throughput: {:.0} residues/second", throughput);
    println!("Time per residue: {:.2} us", elapsed.as_micros() as f64 / n_iterations as f64);

    // Target: >10k residues/second for real-time performance
    assert!(throughput > 10000.0,
        "GPU scorer too slow: {:.0} residues/sec (target: >10000)", throughput);
}

/// Test that scorer rejects invalid input dimensions
#[test]
fn test_input_validation() {
    let context = match CudaContext::new(0) {
        Ok(ctx) => Arc::new(ctx),
        Err(_) => {
            eprintln!("Skipping input validation test - no GPU");
            return;
        }
    };

    let mut scorer = GpuZroCrypticScorer::new(context).unwrap();

    // Valid input should work
    let features = CrypticFeatures::default();
    assert!(scorer.score_residue(&features).is_ok());

    // Test with extreme values (should be handled gracefully)
    let extreme_features = CrypticFeatures {
        burial_change: 1e10,
        rmsf: -1e10,
        ..Default::default()
    };

    let score = scorer.score_residue(&extreme_features).unwrap();
    assert!(score.is_finite(), "Extreme inputs should not produce NaN/Inf");
}
```

### Week 1-2 Checklist

```
[ ] cryptic_features.rs compiles and tests pass
[ ] gpu_zro_cryptic_scorer.rs compiles
[ ] Zero fallback test fails when GPU hidden
[ ] RLS stability test passes (1000 updates, no NaN/Inf)
[ ] Extended stability test passes (10000 updates) [--ignored]
[ ] Weight persistence test passes (save/load roundtrip)
[ ] Throughput benchmark: >10k residues/sec
[ ] Weight statistics logged at end of training
[ ] Soft reset triggers when precision trace exceeds 1e6
[ ] Full reset clears weights and precision matrix
```

---

## 5. Weeks 3-4: PRISM-NOVA Integration

### Task 3.1: Create PDB Sanitizer

**File**: `crates/prism-validation/src/pdb_sanitizer.rs`

```rust
//! PDB sanitization for GPU safety
//!
//! CRITICAL: Raw PDBs can crash GPU kernels. This module ensures
//! all structures are clean before processing.

use anyhow::{Result, Context};

pub struct PdbSanitizer {
    remove_hetatm: bool,
    remove_waters: bool,
    filter_standard_aa: bool,
}

impl PdbSanitizer {
    pub fn new() -> Self {
        Self {
            remove_hetatm: true,
            remove_waters: true,
            filter_standard_aa: true,
        }
    }

    /// Sanitize PDB content for GPU processing
    pub fn sanitize(&self, pdb_content: &str) -> Result<SanitizedStructure> {
        let mut atoms = Vec::new();
        let mut ca_coords = Vec::new();
        let mut atom_serial = 1;

        for line in pdb_content.lines() {
            if line.starts_with("ATOM") {
                // Filter to standard amino acids if enabled
                let resname = &line[17..20].trim();
                if self.filter_standard_aa && !is_standard_aa(resname) {
                    continue;
                }

                // Extract coordinates
                let x: f32 = line[30..38].trim().parse()
                    .context("Failed to parse X coordinate")?;
                let y: f32 = line[38..46].trim().parse()
                    .context("Failed to parse Y coordinate")?;
                let z: f32 = line[46..54].trim().parse()
                    .context("Failed to parse Z coordinate")?;

                // Track CA atoms
                let atom_name = line[12..16].trim();
                if atom_name == "CA" {
                    ca_coords.push([x, y, z]);
                }

                // Renumber atom
                let mut new_line = line.to_string();
                let serial_str = format!("{:5}", atom_serial);
                new_line.replace_range(6..11, &serial_str);

                atoms.push(new_line);
                atom_serial += 1;
            } else if line.starts_with("HETATM") && !self.remove_hetatm {
                atoms.push(line.to_string());
            } else if line.starts_with("TER") || line.starts_with("END") {
                atoms.push(line.to_string());
            }
        }

        Ok(SanitizedStructure {
            pdb_lines: atoms,
            ca_coords,
            n_atoms: atom_serial - 1,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SanitizedStructure {
    pub pdb_lines: Vec<String>,
    pub ca_coords: Vec<[f32; 3]>,
    pub n_atoms: usize,
}

impl SanitizedStructure {
    pub fn get_ca_coords(&self) -> Vec<[f32; 3]> {
        self.ca_coords.clone()
    }

    pub fn to_pdb_string(&self) -> String {
        self.pdb_lines.join("\n")
    }
}

fn is_standard_aa(resname: &str) -> bool {
    matches!(resname,
        "ALA" | "ARG" | "ASN" | "ASP" | "CYS" |
        "GLN" | "GLU" | "GLY" | "HIS" | "ILE" |
        "LEU" | "LYS" | "MET" | "PHE" | "PRO" |
        "SER" | "THR" | "TRP" | "TYR" | "VAL")
}

impl Default for PdbSanitizer {
    fn default() -> Self {
        Self::new()
    }
}
```

### Task 3.2: Create Apo-Holo Benchmark

**File**: `crates/prism-validation/src/apo_holo_benchmark.rs`

```rust
//! Apo-Holo Conformational Change Benchmark
//!
//! Tests PRISM's ability to predict conformational changes:
//! 1. Start from apo (ligand-free) structure
//! 2. Sample with NOVA
//! 3. Check if any conformation approaches holo (ligand-bound) state

use anyhow::{Result, Context};
use std::sync::Arc;
use cudarc::driver::CudaContext;

use crate::nova_cryptic_sampler::{NovaCrypticSampler, NovaCrypticConfig};
use crate::pdb_sanitizer::PdbSanitizer;

/// Classic apo-holo pairs with known conformational changes
pub const APO_HOLO_PAIRS: &[ApoHoloPair] = &[
    ApoHoloPair { apo: "1AKE", holo: "4AKE", name: "Adenylate kinase", motion: MotionType::DomainClosure },
    ApoHoloPair { apo: "2LAO", holo: "1LST", name: "Lysine-binding protein", motion: MotionType::HingeMotion },
    ApoHoloPair { apo: "1GGG", holo: "1WDN", name: "Calmodulin", motion: MotionType::DomainRotation },
    ApoHoloPair { apo: "1OMP", holo: "1ANF", name: "Maltose-binding protein", motion: MotionType::DomainClosure },
    ApoHoloPair { apo: "1RX2", holo: "1RX4", name: "Ribonuclease", motion: MotionType::LoopMotion },
    ApoHoloPair { apo: "3CHY", holo: "2CHE", name: "CheY", motion: MotionType::SmallRotation },
    ApoHoloPair { apo: "1EX6", holo: "1EX7", name: "Galectin", motion: MotionType::LoopMotion },
    ApoHoloPair { apo: "1STP", holo: "1SWB", name: "Streptavidin", motion: MotionType::LoopMotion },
    ApoHoloPair { apo: "1AJJ", holo: "1AJK", name: "Guanylate kinase", motion: MotionType::DomainClosure },
    ApoHoloPair { apo: "1PHP", holo: "1PHN", name: "Phosphotransferase", motion: MotionType::HingeMotion },
    ApoHoloPair { apo: "1BTL", holo: "1BTM", name: "Beta-lactamase", motion: MotionType::SmallRotation },
    ApoHoloPair { apo: "2CPL", holo: "1CWA", name: "Cyclophilin", motion: MotionType::LoopMotion },
    ApoHoloPair { apo: "1BMD", holo: "1BMC", name: "Biotin-binding", motion: MotionType::LoopMotion },
    ApoHoloPair { apo: "1URN", holo: "1URP", name: "Ubiquitin", motion: MotionType::SmallRotation },
    ApoHoloPair { apo: "1HOE", holo: "1HOF", name: "Alpha-amylase inhibitor", motion: MotionType::LoopMotion },
];

#[derive(Debug, Clone, Copy)]
pub struct ApoHoloPair {
    pub apo: &'static str,
    pub holo: &'static str,
    pub name: &'static str,
    pub motion: MotionType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MotionType {
    DomainClosure,    // Large hinge-bending (>5A)
    DomainRotation,   // Rigid body rotation
    HingeMotion,      // Classic hinge movement
    LoopMotion,       // Flexible loop rearrangement
    SmallRotation,    // Minor conformational shifts (<2A)
}

impl MotionType {
    /// Success threshold (min RMSD to holo) for this motion type
    pub fn success_threshold(&self) -> f32 {
        match self {
            MotionType::SmallRotation => 1.5,
            MotionType::LoopMotion => 2.0,
            MotionType::HingeMotion => 2.5,
            MotionType::DomainRotation => 3.0,
            MotionType::DomainClosure => 3.5,
        }
    }
}

/// Result for single apo-holo validation
#[derive(Debug, Clone)]
pub struct ApoHoloResult {
    pub apo_pdb: String,
    pub holo_pdb: String,
    pub name: String,
    pub motion_type: MotionType,
    pub apo_holo_rmsd: f32,
    pub min_rmsd_to_holo: f32,
    pub best_sample_idx: usize,
    pub rmsd_improvement: f32,
    pub success: bool,
    pub time_to_open: Option<usize>,
    pub rmsd_trajectory: Vec<f32>,
}

/// Benchmark runner
pub struct ApoHoloBenchmark {
    context: Arc<CudaContext>,
    config: NovaCrypticConfig,
    data_dir: String,
    results: Vec<ApoHoloResult>,
}

impl ApoHoloBenchmark {
    pub fn new(context: Arc<CudaContext>, data_dir: &str) -> Self {
        Self {
            context,
            config: NovaCrypticConfig::default(),
            data_dir: data_dir.to_string(),
            results: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: NovaCrypticConfig) -> Self {
        self.config = config;
        self
    }

    /// Run benchmark on all pairs
    pub fn run_all(&mut self) -> Result<ApoHoloBenchmarkSummary> {
        log::info!("Starting apo-holo benchmark on {} pairs", APO_HOLO_PAIRS.len());

        for pair in APO_HOLO_PAIRS {
            match self.run_pair(pair) {
                Ok(result) => {
                    log::info!("  {} {}: {:.2}A -> {:.2}A ({})",
                               if result.success { "OK" } else { "FAIL" },
                               pair.name,
                               result.apo_holo_rmsd,
                               result.min_rmsd_to_holo,
                               if result.success { "SUCCESS" } else { "FAILED" });
                    self.results.push(result);
                }
                Err(e) => {
                    log::error!("  FAIL {} FAILED: {}", pair.name, e);
                }
            }
        }

        Ok(self.summarize())
    }

    /// Run benchmark on single pair
    pub fn run_pair(&self, pair: &ApoHoloPair) -> Result<ApoHoloResult> {
        let apo_path = format!("{}/{}_apo.pdb", self.data_dir, pair.apo);
        let holo_path = format!("{}/{}_holo.pdb", self.data_dir, pair.holo);

        let apo_content = std::fs::read_to_string(&apo_path)
            .context(format!("Failed to read {}", apo_path))?;
        let holo_content = std::fs::read_to_string(&holo_path)
            .context(format!("Failed to read {}", holo_path))?;

        let sanitizer = PdbSanitizer::new();
        let apo_struct = sanitizer.sanitize(&apo_content)?;
        let holo_struct = sanitizer.sanitize(&holo_content)?;

        let apo_ca = apo_struct.get_ca_coords();
        let holo_ca = holo_struct.get_ca_coords();

        let n = apo_ca.len().min(holo_ca.len());
        let apo_ca: Vec<_> = apo_ca.into_iter().take(n).collect();
        let holo_ca: Vec<_> = holo_ca.into_iter().take(n).collect();

        let apo_holo_rmsd = compute_rmsd(&apo_ca, &holo_ca);

        let mut sampler = NovaCrypticSampler::new(Arc::clone(&self.context))?
            .with_config(self.config.clone());
        sampler.load_structure(&apo_content)?;

        let sampling_result = sampler.sample()?;

        let rmsd_trajectory: Vec<f32> = sampling_result.conformations.iter()
            .map(|conf| {
                let trimmed: Vec<_> = conf.iter().take(n).cloned().collect();
                compute_rmsd(&trimmed, &holo_ca)
            })
            .collect();

        let (best_idx, &min_rmsd) = rmsd_trajectory.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let rmsd_improvement = apo_holo_rmsd - min_rmsd;
        let success = min_rmsd < pair.motion.success_threshold();

        let opening_threshold = apo_holo_rmsd * 0.7;
        let time_to_open = rmsd_trajectory.iter()
            .position(|&r| r < opening_threshold);

        Ok(ApoHoloResult {
            apo_pdb: pair.apo.to_string(),
            holo_pdb: pair.holo.to_string(),
            name: pair.name.to_string(),
            motion_type: pair.motion,
            apo_holo_rmsd,
            min_rmsd_to_holo: min_rmsd,
            best_sample_idx: best_idx,
            rmsd_improvement,
            success,
            time_to_open,
            rmsd_trajectory,
        })
    }

    pub fn summarize(&self) -> ApoHoloBenchmarkSummary {
        let n_total = self.results.len();
        let n_success = self.results.iter().filter(|r| r.success).count();

        let mean_improvement = if n_total > 0 {
            self.results.iter().map(|r| r.rmsd_improvement).sum::<f32>() / n_total as f32
        } else { 0.0 };

        let mean_min_rmsd = if n_total > 0 {
            self.results.iter().map(|r| r.min_rmsd_to_holo).sum::<f32>() / n_total as f32
        } else { 0.0 };

        ApoHoloBenchmarkSummary {
            n_total,
            n_success,
            success_rate: n_success as f32 / n_total.max(1) as f32,
            mean_rmsd_improvement: mean_improvement,
            mean_min_rmsd_to_holo: mean_min_rmsd,
            results: self.results.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApoHoloBenchmarkSummary {
    pub n_total: usize,
    pub n_success: usize,
    pub success_rate: f32,
    pub mean_rmsd_improvement: f32,
    pub mean_min_rmsd_to_holo: f32,
    pub results: Vec<ApoHoloResult>,
}

impl ApoHoloBenchmarkSummary {
    pub fn to_report(&self) -> String {
        let mut s = String::new();
        s.push_str("# Apo-Holo Benchmark Results\n\n");
        s.push_str(&format!("**Success Rate**: {}/{} ({:.0}%)\n\n",
                            self.n_success, self.n_total, self.success_rate * 100.0));
        s.push_str(&format!("**Mean Improvement**: {:.2} A\n", self.mean_rmsd_improvement));
        s.push_str(&format!("**Mean Min RMSD**: {:.2} A\n\n", self.mean_min_rmsd_to_holo));

        s.push_str("| Protein | Apo->Holo | Best | Delta | Status |\n");
        s.push_str("|---------|----------|------|---|--------|\n");
        for r in &self.results {
            s.push_str(&format!("| {} | {:.2}A | {:.2}A | {:.2}A | {} |\n",
                                r.name, r.apo_holo_rmsd, r.min_rmsd_to_holo,
                                r.rmsd_improvement,
                                if r.success { "OK" } else { "FAIL" }));
        }
        s
    }

    pub fn to_latex(&self) -> String {
        let mut s = String::new();
        s.push_str("\\begin{table}[h]\n\\centering\n");
        s.push_str("\\caption{Apo-Holo Conformational Change Prediction}\n");
        s.push_str("\\begin{tabular}{lcccc}\n\\toprule\n");
        s.push_str("Protein & Apo$\\to$Holo & Min RMSD & $\\Delta$ & Success \\\\\n");
        s.push_str("\\midrule\n");

        for r in &self.results {
            s.push_str(&format!("{} & {:.2}\\AA & {:.2}\\AA & {:.2}\\AA & {} \\\\\n",
                                r.name.replace("_", "\\_"),
                                r.apo_holo_rmsd, r.min_rmsd_to_holo, r.rmsd_improvement,
                                if r.success { "\\checkmark" } else { "$\\times$" }));
        }

        s.push_str("\\midrule\n");
        s.push_str(&format!("\\textbf{{Total}} & & {:.2}\\AA & {:.2}\\AA & {:.0}\\% \\\\\n",
                            self.mean_min_rmsd_to_holo, self.mean_rmsd_improvement,
                            self.success_rate * 100.0));
        s.push_str("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
        s
    }
}

fn compute_rmsd(conf1: &[[f32; 3]], conf2: &[[f32; 3]]) -> f32 {
    assert_eq!(conf1.len(), conf2.len());
    let n = conf1.len() as f32;
    let sum_sq: f32 = conf1.iter().zip(conf2.iter())
        .map(|(a, b)| {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            dx*dx + dy*dy + dz*dz
        })
        .sum();
    (sum_sq / n).sqrt()
}
```

### Week 3-4 Checklist

```
[ ] pdb_sanitizer.rs compiles and tests pass
[ ] nova_cryptic_sampler.rs compiles
[ ] apo_holo_benchmark.rs compiles
[ ] Test on small PDB (dipeptide) passes
[ ] Test on 3CSY (Ebola GP trimer) passes
[ ] Interface residue detection works for multi-chain
[ ] TDA Betti-2 tracking shows void formation
[ ] Acceptance rate > 20% (adjust temperature if needed)
[ ] Ensemble quality metrics computed
[ ] Apo-holo benchmark runs on 1AKE/4AKE pair
[ ] At least 10/15 apo-holo pairs show improvement
```

---

## 6. Weeks 5-6: CryptoBench & Ablation Study

### Task 5.1: CryptoBench Dataset Loader

**File**: `crates/prism-validation/src/cryptobench_dataset.rs`

```rust
//! CryptoBench dataset loader
//!
//! Handles the 1107-structure benchmark with 885/222 train/test split.

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoBenchEntry {
    pub structure_id: String,
    pub pdb_path: String,
    pub chain_id: String,
    pub binding_residues: Vec<i32>,
    pub pocket_type: Option<String>,
    pub ligand_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoBenchManifest {
    pub name: String,
    pub version: String,
    pub train_entries: Vec<CryptoBenchEntry>,
    pub test_entries: Vec<CryptoBenchEntry>,
}

pub struct CryptoBenchDataset {
    pub manifest: CryptoBenchManifest,
    pub base_path: String,
}

impl CryptoBenchDataset {
    pub fn load(manifest_path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(manifest_path)
            .context("Failed to read manifest")?;
        let manifest: CryptoBenchManifest = serde_json::from_str(&content)
            .context("Failed to parse manifest JSON")?;

        let base_path = Path::new(manifest_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());

        log::info!("Loaded CryptoBench: {} train, {} test structures",
                   manifest.train_entries.len(), manifest.test_entries.len());

        Ok(Self { manifest, base_path })
    }

    pub fn train_entries(&self) -> &[CryptoBenchEntry] {
        &self.manifest.train_entries
    }

    pub fn test_entries(&self) -> &[CryptoBenchEntry] {
        &self.manifest.test_entries
    }

    pub fn ground_truth(&self, structure_id: &str) -> Option<HashSet<i32>> {
        self.manifest.train_entries.iter()
            .chain(self.manifest.test_entries.iter())
            .find(|e| e.structure_id == structure_id)
            .map(|e| e.binding_residues.iter().cloned().collect())
    }

    pub fn load_pdb(&self, entry: &CryptoBenchEntry) -> Result<String> {
        let path = format!("{}/{}", self.base_path, entry.pdb_path);
        std::fs::read_to_string(&path)
            .context(format!("Failed to load PDB: {}", path))
    }

    pub fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();

        for entry in self.manifest.train_entries.iter()
            .chain(self.manifest.test_entries.iter())
        {
            report.total += 1;

            let path = format!("{}/{}", self.base_path, entry.pdb_path);
            if !Path::new(&path).exists() {
                report.missing_pdbs.push(entry.structure_id.clone());
                continue;
            }

            if entry.binding_residues.is_empty() {
                report.no_binding_site.push(entry.structure_id.clone());
                continue;
            }

            report.valid += 1;
        }

        Ok(report)
    }
}

#[derive(Debug, Default)]
pub struct ValidationReport {
    pub total: usize,
    pub valid: usize,
    pub missing_pdbs: Vec<String>,
    pub no_binding_site: Vec<String>,
}

impl ValidationReport {
    pub fn is_ok(&self) -> bool {
        self.missing_pdbs.is_empty() && self.no_binding_site.is_empty()
    }
}
```

### Task 5.2: Ablation Study Framework

**File**: `crates/prism-validation/src/ablation.rs`

```rust
//! Ablation study framework
//!
//! Tests 6 variants to prove component contributions:
//! 1. ANM only (baseline)
//! 2. ANM + GPU-SNN
//! 3. NOVA only
//! 4. NOVA + CPU-SNN (for comparison only, not production)
//! 5. NOVA + GPU-SNN (no TDA)
//! 6. Full pipeline (NOVA + GPU-SNN + TDA + RLS)

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AblationVariant {
    AnmOnly,
    AnmGpuSnn,
    NovaOnly,
    NovaCpuSnn,
    NovaGpuSnn,
    Full,
}

impl AblationVariant {
    pub fn name(&self) -> &'static str {
        match self {
            Self::AnmOnly => "ANM Only",
            Self::AnmGpuSnn => "ANM + GPU-SNN",
            Self::NovaOnly => "NOVA Only",
            Self::NovaCpuSnn => "NOVA + CPU-SNN",
            Self::NovaGpuSnn => "NOVA + GPU-SNN",
            Self::Full => "Full Pipeline",
        }
    }

    pub fn uses_nova(&self) -> bool {
        matches!(self, Self::NovaOnly | Self::NovaCpuSnn | Self::NovaGpuSnn | Self::Full)
    }

    pub fn uses_gpu_snn(&self) -> bool {
        matches!(self, Self::AnmGpuSnn | Self::NovaGpuSnn | Self::Full)
    }

    pub fn uses_tda(&self) -> bool {
        matches!(self, Self::Full)
    }

    pub fn uses_rls(&self) -> bool {
        matches!(self, Self::Full)
    }

    pub fn all() -> &'static [AblationVariant] {
        &[
            Self::AnmOnly,
            Self::AnmGpuSnn,
            Self::NovaOnly,
            Self::NovaCpuSnn,
            Self::NovaGpuSnn,
            Self::Full,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationResult {
    pub variant: AblationVariant,
    pub roc_auc: f32,
    pub pr_auc: f32,
    pub success_rate: f32,
    pub top1_accuracy: f32,
    pub time_per_structure: f32,
    pub n_structures: usize,
}

impl AblationResult {
    pub fn delta_from(&self, baseline: &AblationResult) -> AblationDelta {
        AblationDelta {
            variant: self.variant,
            delta_roc_auc: self.roc_auc - baseline.roc_auc,
            delta_pr_auc: self.pr_auc - baseline.pr_auc,
            delta_success_rate: self.success_rate - baseline.success_rate,
            speedup: baseline.time_per_structure / self.time_per_structure.max(0.001),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AblationDelta {
    pub variant: AblationVariant,
    pub delta_roc_auc: f32,
    pub delta_pr_auc: f32,
    pub delta_success_rate: f32,
    pub speedup: f32,
}

#[derive(Debug, Clone, Default)]
pub struct AblationStudy {
    pub results: Vec<AblationResult>,
}

impl AblationStudy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_result(&mut self, result: AblationResult) {
        self.results.push(result);
    }

    pub fn get_baseline(&self) -> Option<&AblationResult> {
        self.results.iter().find(|r| r.variant == AblationVariant::AnmOnly)
    }

    pub fn get_full(&self) -> Option<&AblationResult> {
        self.results.iter().find(|r| r.variant == AblationVariant::Full)
    }

    pub fn to_markdown_table(&self) -> String {
        let baseline = self.get_baseline();

        let mut s = String::new();
        s.push_str("| Variant | ROC AUC | Delta AUC | PR AUC | Success | Time |\n");
        s.push_str("|---------|---------|-------|--------|---------|------|\n");

        for r in &self.results {
            let delta = baseline.map(|b| r.roc_auc - b.roc_auc).unwrap_or(0.0);
            let delta_str = if delta > 0.0 { format!("+{:.3}", delta) }
                           else { format!("{:.3}", delta) };

            s.push_str(&format!("| {} | {:.3} | {} | {:.3} | {:.1}% | {:.2}s |\n",
                                r.variant.name(),
                                r.roc_auc,
                                delta_str,
                                r.pr_auc,
                                r.success_rate * 100.0,
                                r.time_per_structure));
        }
        s
    }

    pub fn to_latex_table(&self) -> String {
        let baseline = self.get_baseline();

        let mut s = String::new();
        s.push_str("\\begin{table}[h]\n\\centering\n");
        s.push_str("\\caption{Ablation Study Results}\n");
        s.push_str("\\begin{tabular}{lccccc}\n\\toprule\n");
        s.push_str("Variant & ROC AUC & $\\Delta$ & PR AUC & Success & Time \\\\\n");
        s.push_str("\\midrule\n");

        for r in &self.results {
            let delta = baseline.map(|b| r.roc_auc - b.roc_auc).unwrap_or(0.0);
            let delta_str = if delta > 0.0 { format!("+{:.3}", delta) }
                           else if delta < 0.0 { format!("{:.3}", delta) }
                           else { "---".to_string() };

            s.push_str(&format!("{} & {:.3} & {} & {:.3} & {:.1}\\% & {:.2}s \\\\\n",
                                r.variant.name().replace("_", "\\_"),
                                r.roc_auc,
                                delta_str,
                                r.pr_auc,
                                r.success_rate * 100.0,
                                r.time_per_structure));
        }

        s.push_str("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
        s
    }
}
```

### Task 5.3: Failure Case Analysis

**File**: `crates/prism-validation/src/failure_analysis.rs`

```rust
//! Failure case analysis for limitations section

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureReason {
    PocketTooDeep,
    LargeConformationalChange,
    AllostericSite,
    CrystalContact,
    MultiplePockets,
    PoorStructureQuality,
    SmallPocket,
    CofactorDependent,
    Unknown,
}

impl FailureReason {
    pub fn description(&self) -> &'static str {
        match self {
            Self::PocketTooDeep => "Pocket buried beyond sampling reach",
            Self::LargeConformationalChange => "Requires large backbone motion (>5A)",
            Self::AllostericSite => "Allosteric site distal from active region",
            Self::CrystalContact => "Pocket is crystal packing artifact",
            Self::MultiplePockets => "Multiple pockets cause annotation ambiguity",
            Self::PoorStructureQuality => "Structure has quality issues",
            Self::SmallPocket => "Very small pocket (<100 A^3)",
            Self::CofactorDependent => "Pocket requires cofactor to form",
            Self::Unknown => "Unclassified failure",
        }
    }

    pub fn is_fundamental_limitation(&self) -> bool {
        matches!(self,
            Self::LargeConformationalChange |
            Self::AllostericSite |
            Self::CofactorDependent)
    }

    pub fn is_data_issue(&self) -> bool {
        matches!(self,
            Self::CrystalContact |
            Self::MultiplePockets |
            Self::PoorStructureQuality)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureCaseAnalysis {
    pub structure_id: String,
    pub predicted_auc: f32,
    pub reason: FailureReason,
    pub notes: String,
}

#[derive(Debug, Clone, Default)]
pub struct FailureReport {
    pub cases: Vec<FailureCaseAnalysis>,
}

impl FailureReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, case: FailureCaseAnalysis) {
        self.cases.push(case);
    }

    pub fn count_by_reason(&self) -> Vec<(FailureReason, usize)> {
        let mut counts = std::collections::HashMap::new();
        for case in &self.cases {
            *counts.entry(case.reason).or_insert(0) += 1;
        }
        let mut result: Vec<_> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("## Failure Case Analysis\n\n");
        s.push_str(&format!("Total failures: {}\n\n", self.cases.len()));

        s.push_str("### By Category\n\n");
        for (reason, count) in self.count_by_reason() {
            s.push_str(&format!("- **{}**: {} ({:.0}%)\n",
                                reason.description(),
                                count,
                                count as f32 / self.cases.len() as f32 * 100.0));
        }

        let fundamental = self.cases.iter()
            .filter(|c| c.reason.is_fundamental_limitation())
            .count();
        let data_issues = self.cases.iter()
            .filter(|c| c.reason.is_data_issue())
            .count();

        s.push_str(&format!("\n**Fundamental limitations**: {} ({:.0}%)\n",
                            fundamental, fundamental as f32 / self.cases.len() as f32 * 100.0));
        s.push_str(&format!("**Data issues**: {} ({:.0}%)\n",
                            data_issues, data_issues as f32 / self.cases.len() as f32 * 100.0));

        s
    }
}
```

### Week 5-6 Checklist

```
[ ] cryptobench_dataset.rs compiles
[ ] Dataset loads all 1107 structures
[ ] Train/test split verified (885/222, no leakage)
[ ] ablation.rs compiles
[ ] All 6 ablation variants run
[ ] Full pipeline > ANM-only by >0.20 AUC
[ ] failure_analysis.rs compiles
[ ] Bottom 10% failures categorized
[ ] Metrics computed on test set only
[ ] Results serialized to JSON
```

---

## 7. Weeks 7-8: Publication & Final Validation

### Task 7.1: Publication Outputs

**File**: `crates/prism-validation/src/publication_outputs.rs`

```rust
//! Publication-ready output generation

use anyhow::Result;
use serde::Serialize;
use std::fs;

#[derive(Debug, Clone, Serialize)]
pub struct PublicationResults {
    pub main_metrics: MainMetrics,
    pub ablation_study: Vec<AblationRow>,
    pub apo_holo_results: Vec<ApoHoloRow>,
    pub failure_summary: FailureSummary,
    pub timing_stats: TimingStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct MainMetrics {
    pub roc_auc: f32,
    pub roc_auc_ci: (f32, f32),
    pub pr_auc: f32,
    pub pr_auc_ci: (f32, f32),
    pub success_rate: f32,
    pub top1_accuracy: f32,
    pub n_test_structures: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AblationRow {
    pub variant: String,
    pub roc_auc: f32,
    pub delta_auc: f32,
    pub pr_auc: f32,
    pub success_rate: f32,
    pub time_seconds: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApoHoloRow {
    pub name: String,
    pub apo_pdb: String,
    pub holo_pdb: String,
    pub start_rmsd: f32,
    pub best_rmsd: f32,
    pub improvement: f32,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FailureSummary {
    pub total_failures: usize,
    pub by_category: Vec<(String, usize)>,
    pub fundamental_limitations: usize,
    pub data_issues: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimingStats {
    pub mean_time_per_structure: f32,
    pub median_time_per_structure: f32,
    pub total_test_time_hours: f32,
    pub gpu_model: String,
    pub peak_vram_gb: f32,
}

impl PublicationResults {
    pub fn save_all(&self, output_dir: &str) -> Result<()> {
        fs::create_dir_all(output_dir)?;

        let json = serde_json::to_string_pretty(self)?;
        fs::write(format!("{}/results.json", output_dir), json)?;

        fs::write(
            format!("{}/table_main_results.tex", output_dir),
            self.generate_main_table()
        )?;

        fs::write(
            format!("{}/table_ablation.tex", output_dir),
            self.generate_ablation_table()
        )?;

        fs::write(
            format!("{}/table_apo_holo.tex", output_dir),
            self.generate_apo_holo_table()
        )?;

        fs::write(
            format!("{}/ablation_data.csv", output_dir),
            self.generate_ablation_csv()
        )?;

        fs::write(
            format!("{}/methods_section.md", output_dir),
            self.generate_methods()
        )?;

        log::info!("Publication outputs saved to {}", output_dir);
        Ok(())
    }

    fn generate_main_table(&self) -> String {
        format!(r#"\begin{{table}}[h]
\centering
\caption{{PRISM-ZrO Cryptic Site Detection Performance on CryptoBench}}
\label{{tab:main_results}}
\begin{{tabular}}{{lc}}
\toprule
Metric & Value \\
\midrule
ROC AUC & {:.3} ({:.3}--{:.3}) \\
PR AUC & {:.3} ({:.3}--{:.3}) \\
Success Rate ($\geq$30\% overlap) & {:.1}\% \\
Top-1 Accuracy & {:.1}\% \\
Test Set Size & {} structures \\
\bottomrule
\end{{tabular}}
\end{{table}}
"#,
            self.main_metrics.roc_auc,
            self.main_metrics.roc_auc_ci.0,
            self.main_metrics.roc_auc_ci.1,
            self.main_metrics.pr_auc,
            self.main_metrics.pr_auc_ci.0,
            self.main_metrics.pr_auc_ci.1,
            self.main_metrics.success_rate * 100.0,
            self.main_metrics.top1_accuracy * 100.0,
            self.main_metrics.n_test_structures)
    }

    fn generate_ablation_table(&self) -> String {
        let mut s = String::from(r#"\begin{table}[h]
\centering
\caption{Ablation Study: Component Contributions}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Configuration & ROC AUC & $\Delta$ & PR AUC & Success & Time \\
\midrule
"#);

        for row in &self.ablation_study {
            let delta = if row.delta_auc > 0.0 {
                format!("+{:.3}", row.delta_auc)
            } else if row.delta_auc < 0.0 {
                format!("{:.3}", row.delta_auc)
            } else {
                "---".to_string()
            };

            s.push_str(&format!(
                "{} & {:.3} & {} & {:.3} & {:.1}\\% & {:.2}s \\\\\n",
                row.variant.replace("_", "\\_"),
                row.roc_auc,
                delta,
                row.pr_auc,
                row.success_rate * 100.0,
                row.time_seconds
            ));
        }

        s.push_str(r#"\bottomrule
\end{tabular}
\end{table}
"#);
        s
    }

    fn generate_apo_holo_table(&self) -> String {
        let mut s = String::from(r#"\begin{table}[h]
\centering
\caption{Apo-Holo Conformational Change Prediction}
\label{tab:apo_holo}
\begin{tabular}{lcccc}
\toprule
Protein & Start & Best & $\Delta$ & Success \\
\midrule
"#);

        for row in &self.apo_holo_results {
            s.push_str(&format!(
                "{} & {:.2}\\AA & {:.2}\\AA & {:.2}\\AA & {} \\\\\n",
                row.name.replace("_", "\\_"),
                row.start_rmsd,
                row.best_rmsd,
                row.improvement,
                if row.success { "\\checkmark" } else { "$\\times$" }
            ));
        }

        s.push_str(r#"\bottomrule
\end{tabular}
\end{table}
"#);
        s
    }

    fn generate_ablation_csv(&self) -> String {
        let mut s = String::from("variant,roc_auc,delta_auc,pr_auc,success_rate,time_seconds\n");
        for row in &self.ablation_study {
            s.push_str(&format!("{},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                                row.variant, row.roc_auc, row.delta_auc,
                                row.pr_auc, row.success_rate, row.time_seconds));
        }
        s
    }

    fn generate_methods(&self) -> String {
        format!(r#"# Methods

## Cryptic Site Detection Pipeline

PRISM-ZrO combines neuromorphic computing with enhanced conformational sampling
for cryptic binding site detection.

### Conformational Sampling

Structures were sampled using PRISM-NOVA, a neural Hamiltonian Monte Carlo
implementation running on GPU. Parameters:
- Temperature: 310 K
- Timestep: 2 fs
- Leapfrog steps: 5
- Samples per structure: 500
- Decorrelation steps: 100

### Feature Extraction

Per-residue features (16 dimensions) were computed including:
- Dynamics: burial change, RMSF, variance, neighbor flexibility
- Structural: secondary structure flexibility, B-factor
- Chemical: charge, hydrophobicity, H-bond potential
- Spatial: contact density, SASA change, interface proximity

### Neural Scoring

Features were processed through a 512-neuron GPU-accelerated dendritic
spiking neural network reservoir with RLS online learning (lambda=0.99).

### Evaluation

Performance was evaluated on the CryptoBench benchmark:
- {} test structures (held out from training)
- Primary metrics: ROC AUC, PR AUC
- Secondary metrics: Success rate (>=30% overlap), Top-1 accuracy

### Hardware

All experiments were performed on:
- GPU: {}
- Peak VRAM: {:.1} GB
- Mean time per structure: {:.2} seconds
"#,
            self.main_metrics.n_test_structures,
            self.timing_stats.gpu_model,
            self.timing_stats.peak_vram_gb,
            self.timing_stats.mean_time_per_structure)
    }
}
```

### Week 7-8 Checklist

```
[ ] publication_outputs.rs compiles
[ ] All LaTeX tables generate correctly
[ ] CSV data exports for plotting
[ ] ROC curve data exported
[ ] PR curve data exported
[ ] Figure generation scripts work
[ ] Methods section draft complete
[ ] Final benchmark sweep matches targets
[ ] All results reproducible from clean state
[ ] Code packaged for release
[ ] README updated with usage instructions
```

---

## 8. Complete File Manifest

### New Files (14 total)

| File | Purpose | Week |
|------|---------|------|
| `cryptic_features.rs` | Feature vector definition | 1 |
| `gpu_zro_cryptic_scorer.rs` | GPU 512-neuron scorer | 1 |
| `ensemble_cryptic_model.rs` | Ensemble weight learning | 1 |
| `ensemble_quality_metrics.rs` | Sampling validation | 1 |
| `tests/gpu_scorer_tests.rs` | GPU scorer tests | 2 |
| `pdb_sanitizer.rs` | PDB preprocessing | 3 |
| `nova_cryptic_sampler.rs` | NOVA HMC wrapper | 3 |
| `apo_holo_benchmark.rs` | Conformational validation | 4 |
| `cryptobench_dataset.rs` | Dataset loader | 5 |
| `cryptobench_benchmark.rs` | Full benchmark runner | 5 |
| `ablation.rs` | Ablation study framework | 5 |
| `failure_analysis.rs` | Failure categorization | 6 |
| `publication_outputs.rs` | LaTeX/figure generation | 7 |
| `scripts/generate_figures.py` | Plotting scripts | 8 |

### Files to Modify (3)

| File | Changes |
|------|---------|
| `blind_validation_pipeline.rs` | Add GPU scorer, NOVA options |
| `lib.rs` | Export Phase 6 modules |
| `Cargo.toml` | Add `cuda` feature, dependencies |

---

## 9. Verification Commands

```bash
# Week 0: Setup verification
cargo check -p prism-validation --features cuda

# Week 2: GPU scorer tests
cargo test --release -p prism-validation --features cuda gpu_scorer -- --nocapture
CUDA_VISIBLE_DEVICES="" cargo test test_no_cpu_fallback  # MUST FAIL

# Week 4: NOVA sampling test
cargo run --release -p prism-validation --bin test-nova-sampler -- \
    --pdb data/test/1ake.pdb --samples 100 --verbose

# Week 4: Apo-holo benchmark
cargo run --release -p prism-validation --bin apo-holo-benchmark -- \
    --data-dir data/benchmarks/apo_holo --output results/apo_holo.json

# Week 6: Full CryptoBench benchmark
cargo run --release -p prism-validation --bin cryptobench-benchmark -- \
    --manifest data/benchmarks/cryptobench/manifest.json \
    --output results/cryptobench_full.json

# Week 6: Ablation study
cargo run --release -p prism-validation --bin ablation-study -- \
    --manifest data/benchmarks/cryptobench/manifest.json \
    --output results/ablation.json

# Week 8: Generate publication outputs
cargo run --release -p prism-validation --bin generate-publication -- \
    --results results/cryptobench_full.json \
    --ablation results/ablation.json \
    --apo-holo results/apo_holo.json \
    --output results/publication/
```

---

## 10. Risk Mitigation

| Risk | Prevention | Recovery |
|------|------------|----------|
| GPU memory overflow | Batch processing | Reduce reservoir to 256 neurons |
| NOVA divergence | Conservative dt (2fs) | Lower temperature, shorter leapfrog |
| RLS instability | lambda=0.99, gradient clamp | Soft reset precision matrix |
| Low acceptance | Monitor rate | Increase temperature |
| Dataset issues | Pre-validate all PDBs | Skip corrupt entries |
| Metric regression | Per-structure logging | Bisect to find cause |

---

## 11. Appendix: Quick Reference & Task List

### Claude Code Task List by Week

**Week 0: Setup**
```
[ ] Verify environment (rustc 1.75+, nvcc 12.0+)
[ ] cargo check -p prism-gpu --features cuda
[ ] Download CryptoBench dataset (1107 structures)
[ ] Create manifest.json with 885/222 train/test split
[ ] Download apo-holo pairs (15 pairs)
[ ] Document Phase 5 baseline metrics
```

**Weeks 1-2: GPU SNN**
```
[ ] Create cryptic_features.rs with 16-dim feature vector
[ ] Create gpu_zro_cryptic_scorer.rs with 512-neuron reservoir
[ ] Implement RLS online learning with Sherman-Morrison updates
[ ] Create gpu_scorer_tests.rs with zero-fallback test
[ ] Verify throughput >10k residues/sec
```

**Weeks 3-4: NOVA**
```
[ ] Create pdb_sanitizer.rs for GPU safety
[ ] Create nova_cryptic_sampler.rs with TDA tracking
[ ] Create apo_holo_benchmark.rs for 15 pairs
[ ] Test on 3CSY (Ebola GP trimer)
[ ] Verify acceptance rate >20%
```

**Weeks 5-6: CryptoBench**
```
[ ] Create cryptobench_dataset.rs loader
[ ] Create ablation.rs with 6 variants
[ ] Create failure_analysis.rs
[ ] Run full benchmark on 222 test structures
[ ] Verify Full > ANM-only by >0.20 AUC
```

**Weeks 7-8: Publication**
```
[ ] Create publication_outputs.rs
[ ] Generate LaTeX tables
[ ] Create figure generation scripts
[ ] Final benchmark sweep
[ ] Package for release
```

### Verification Checkpoints

**Week 2 Checkpoint**
```bash
cargo test --release -p prism-validation --features cuda gpu_scorer
CUDA_VISIBLE_DEVICES="" cargo test test_no_cpu_fallback  # MUST FAIL
cargo test bench_gpu_scorer_throughput -- --nocapture
# Expected: >10,000 residues/second
```

**Week 4 Checkpoint**
```bash
cargo run --release -p prism-validation --bin test-nova -- \
    --pdb test.pdb --samples 100
cargo run --release -p prism-validation --bin apo-holo-single -- \
    --apo 1AKE --holo 4AKE
# Expected: min RMSD < 3.5A
```

**Week 6 Checkpoint**
```bash
cargo run --release -p prism-validation --bin cryptobench -- \
    --manifest manifest.json --output results.json
jq '.roc_auc, .pr_auc, .success_rate' results.json
# Expected: >0.70, >0.20, >0.80
```

**Week 8 Checkpoint**
```bash
cargo run --release -p prism-validation --bin publication -- \
    --output results/publication/
ls results/publication/*.tex
```

### Risk Quick Reference

| Issue | Solution |
|-------|----------|
| RLS explodes | Reset precision matrix when trace > 1e6 |
| Low acceptance | Increase temperature or reduce leapfrog steps |
| GPU OOM | Reduce reservoir to 256 neurons |
| Metric regression | Check per-structure logs, bisect |
| PDB crashes GPU | Run through sanitizer first |

### Success Definition

Phase 6 is complete when:

1. ROC AUC > 0.70 on CryptoBench test set
2. Ablation proves Full > Baseline by >0.20 AUC
3. Apo-holo achieves >60% success rate
4. No CPU fallback exists (verified by failing test)
5. Publication outputs generated (LaTeX, figures)
6. Code is reproducible from clean state

---

**Document Version**: 2.0 (Consolidated)
**Status**: APPROVED FOR EXECUTION
**Ready for execution. Start with Week 0 setup tasks.**
