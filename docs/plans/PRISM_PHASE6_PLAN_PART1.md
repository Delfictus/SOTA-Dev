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

---

## 1. Executive Summary

### Objective

Achieve **publication-ready** cryptic site detection that **exceeds SOTA** using **ONLY native PRISM infrastructure**.

### Target Metrics

| Metric | Phase 5 Baseline | Target | Minimum | SOTA Reference |
|--------|------------------|--------|---------|----------------|
| ROC AUC | 0.487 | **>0.75** | 0.70 | PocketMiner 0.87 |
| PR AUC | 0.081 | **>0.25** | 0.20 | CryptoBank 0.17 |
| Success Rate | 71.7% | **>85%** | 80% | Schrödinger 83% |
| Top-1 Accuracy | 82.6% | **>90%** | 85% | CrypTothML 78% |
| Time/Structure | N/A | **<1s** | <5s | RTX 3060 |
| Peak VRAM | N/A | **<2GB** | <4GB | RTX 3060 |
| Apo-Holo Recovery | N/A | **<2.5Å** | <3.5Å | Min RMSD to holo |
| Ensemble Diversity | N/A | **1-3Å** | 0.5-5Å | Mean pairwise RMSD |

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
✅ REQUIRED:
   - PRISM-ZrO (SNN + RLS) for adaptive learning
   - PRISM-NOVA (HMC + AMBER) for enhanced sampling
   - Native Rust/CUDA implementations
   - GPU-mandatory execution (no silent CPU fallback)
   - Explicit error on missing GPU

❌ FORBIDDEN:
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
| Success Rate | >85% | 80% | Overlap ≥30% with ground truth |
| Top-1 Accuracy | >90% | 85% | Predicted site within 8Å of true site |
| GPU Performance | <1s/structure | <5s | RTX 3060, averaged over test set |
| Memory | <2GB | <4GB | Peak VRAM during inference |

### Secondary Metrics (Validation Quality)

| Criterion | Target | Rationale |
|-----------|--------|-----------|
| Apo-Holo Recovery | <2.5Å | Proves conformational prediction capability |
| Ensemble Diversity | 1-3Å mean pairwise RMSD | Validates sampling quality |
| Ablation Δ | >+0.20 AUC vs ANM-only | Proves component contributions |
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
    echo "Downloaded: $name ($apo → $holo)"
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
| Success Rate | 71.7% | Overlap ≥30% |
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
| **Total** | **+0.30 AUC** | **0.49 → 0.79** |
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
□ Rust 1.75+ installed and verified
□ CUDA 12.0+ installed and verified
□ prism-gpu compiles with cuda feature
□ CryptoBench dataset downloaded (1107 structures)
□ manifest.json created with 885/222 train/test split
□ Apo-holo pairs downloaded (15 pairs, 30 PDB files)
□ Phase 5 baseline documented with commit hash
□ Git branch created: feature/phase-6-cryptic-sota
□ CI pipeline configured for metric regression testing
□ Test structure (3CSY Ebola GP) downloads successfully
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
| RLS λ | 0.99 | Forgetting factor |
| Precision Init | P = 100 * I | Initial precision matrix |
| Gradient Clamp | ±1.0 | Stability |
| Max Precision Trace | 1e6 | Soft reset trigger |

### Files to Create

1. `crates/prism-validation/src/gpu_zro_cryptic_scorer.rs`
2. `crates/prism-validation/src/ensemble_cryptic_model.rs`
3. `crates/prism-validation/src/ensemble_quality_metrics.rs`
4. `crates/prism-validation/src/tests/gpu_scorer_tests.rs`

---

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
    /// Local contact density (neighbors within 8Å)
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

---

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
    
    /// RLS readout weights [512] → single score
    readout_weights: Vec<f32>,
    
    /// RLS precision matrix [512 × 512]
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
        
        // Update precision matrix: P = (1/λ)(P - gain * pk * pk')
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

---

### Task 1.3: Create Unit Tests

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
    println!("Time per residue: {:.2} µs", elapsed.as_micros() as f64 / n_iterations as f64);
    
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

---

### Week 1-2 Verification Commands

```bash
# Run all GPU scorer tests
cargo test --release -p prism-validation --features cuda gpu_scorer -- --nocapture

# Run extended stability test
cargo test --release -p prism-validation --features cuda test_rls_stability_10000 -- --ignored --nocapture

# Verify no CPU fallback (this should FAIL if no GPU)
CUDA_VISIBLE_DEVICES="" cargo test --release -p prism-validation --features cuda test_no_cpu_fallback

# Benchmark throughput
cargo test --release -p prism-validation --features cuda bench_gpu_scorer_throughput -- --nocapture
```

### Week 1-2 Checklist

```
□ cryptic_features.rs compiles and tests pass
□ gpu_zro_cryptic_scorer.rs compiles
□ Zero fallback test fails when GPU hidden
□ RLS stability test passes (1000 updates, no NaN/Inf)
□ Extended stability test passes (10000 updates) [--ignored]
□ Weight persistence test passes (save/load roundtrip)
□ Throughput benchmark: >10k residues/sec
□ Weight statistics logged at end of training
□ Soft reset triggers when precision trace exceeds 1e6
□ Full reset clears weights and precision matrix
```

---

*[Continue to Part 2 for Weeks 3-8]*
