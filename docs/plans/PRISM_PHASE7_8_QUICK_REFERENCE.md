# PRISM Phase 7-8: Quick Reference

## Prerequisites

**Phase 6 must be complete with:**
- CryptoBench ROC AUC ≥ 0.70
- All Phase 6 tests passing
- GPU scorer (512 neurons) operational
- NOVA sampling functional

---

## Target Progression

| Phase | ROC AUC | PR AUC | Status |
|-------|---------|--------|--------|
| 6 (Baseline) | 0.75 | 0.25 | Required |
| 7 | **0.82** | **0.32** | +0.07 |
| 8 | **0.90** | **0.40** | +0.08 |
| **SOTA** | 0.87 | 0.44 | PocketMiner |

**Phase 8 target exceeds published SOTA**

---

## Enhancement Components

### Phase 7 (Weeks 1-8): Architecture

| Component | Δ AUC | Key Change |
|-----------|-------|------------|
| Hierarchical Reservoir | +0.03 | 512 → 1,280 neurons (cortical columns) |
| Persistent Homology | +0.02 | Betti → Full persistence diagrams |
| Extended Sampling | +0.02 | 500 → 2,000 conformations + adaptive bias |
| Multi-Scale Features | +0.02 | 16 → 67 features (local+regional+global) |

### Phase 8 (Weeks 9-16): Advanced

| Component | Δ AUC | Key Change |
|-----------|-------|------------|
| Ensemble Voting | +0.03 | 5 reservoirs with learned weights |
| Transfer Learning | +0.03 | Family-level backbone weights |
| Uncertainty | (quality) | Calibrated confidence intervals |
| Active Learning | (efficiency) | Smart structure selection |

---

## Architecture Summary

### Phase 7: Hierarchical Reservoir

```
Layer 3: Global Context     [1 × 256 = 256 neurons]
         ↑
Layer 2: Regional Integration [4 × 128 = 512 neurons]
         ↑
Layer 1: Local Detectors    [8 × 64 = 512 neurons]
         ↑
Input: 80-dim features

Total: 1,280 neurons (2.5× Phase 6)
```

### Phase 8: Ensemble + Transfer

```
Input Features (80-dim)
       ↓
┌──────────────────────────────┐
│ Reservoir 1 → Score₁        │
│ Reservoir 2 → Score₂        │
│ Reservoir 3 → Score₃  ──────┼──→ Weighted Mean + Uncertainty
│ Reservoir 4 → Score₄        │
│ Reservoir 5 → Score₅        │
└──────────────────────────────┘
       +
Transfer Learning (family backbone weights)
       +
Uncertainty Quantification (confidence intervals)
```

---

## Feature Dimensions

| Scale | Features | Total |
|-------|----------|-------|
| Local (Phase 6) | 16 | 16 |
| Regional (5-12Å) | 12 | 28 |
| Global (protein) | 8 | 36 |
| Persistence (TDA) | 31 | **67** |
| Velocities | +16 | **83** |
| Padding | +N | **80** (input buffer) |

---

## File Manifest

### Phase 7 (8 files)

```
hierarchical_reservoir.rs    - Cortical column architecture
persistent_homology.rs       - TDA with persistence diagrams
extended_nova_sampler.rs     - 2000-sample adaptive biasing
multiscale_features.rs       - Local + regional + global
phase7_scorer.rs             - Integrated pipeline
kernels/hierarchical_reservoir.cu
kernels/persistence.cu
tests/phase7_tests.rs
```

### Phase 8 (6 files)

```
ensemble_reservoir.rs        - 5-reservoir voting
transfer_learning.rs         - Cross-structure transfer
uncertainty.rs               - Calibrated confidence
active_learning.rs           - Structure prioritization
phase8_scorer.rs             - Complete scorer
tests/phase8_tests.rs
```

---

## Timeline

```
Week 1-2:  Hierarchical Reservoir
Week 3-4:  Persistence + Multi-Scale Features
Week 5-6:  Extended NOVA Sampling
Week 7-8:  Phase 7 Integration + Validation
           ──────────────────────────────────
           CHECKPOINT: AUC ≥ 0.82
           ──────────────────────────────────
Week 9-10:  Ensemble Voting
Week 11-13: Transfer Learning
Week 14:    Uncertainty Quantification
Week 15-16: Active Learning + Final Validation
           ──────────────────────────────────
           FINAL: AUC ≥ 0.90
           ──────────────────────────────────
```

---

## Key Specifications

### Extended NOVA Sampling

| Parameter | Phase 6 | Phase 7+ |
|-----------|---------|----------|
| Samples | 500 | **2,000** |
| Steps/Sample | 100 | **50** |
| Adaptive Bias | No | **Yes** |
| Temperature | Fixed 310K | **310K → 290K annealing** |
| TDA Feedback | None | **Real-time β₂** |

### Ensemble Configuration

| Parameter | Value |
|-----------|-------|
| Reservoirs | 5 |
| Seeds | 42, 123, 456, 789, 1011 |
| Combination | Learned weights |
| Uncertainty | σ(scores) |

### Transfer Learning

| Level | Application |
|-------|-------------|
| Family | Same protein family (kinase→kinase) |
| Global | Cross-family (weaker, 0.3× discount) |
| Instance | Per-structure RLS adaptation |

---

## Verification Commands

```bash
# Phase 7 checkpoint (Week 8)
cargo test --release -p prism-validation --features cuda phase7

cargo run --release -p prism-validation --bin cryptobench -- \
    --config phase7 \
    --output results/phase7.json

# Check: AUC ≥ 0.82

# Phase 8 final (Week 16)
cargo test --release -p prism-validation --features cuda phase8

cargo run --release -p prism-validation --bin cryptobench -- \
    --config phase8 \
    --output results/phase8.json

# Check: AUC ≥ 0.90
```

---

## Success Criteria

### Phase 7 (Week 8)
```
□ Hierarchical reservoir compiles and runs
□ 1,280 neurons, <10ms/step
□ Persistence features extracted (31-dim)
□ Extended sampling works (2000 conformations)
□ ROC AUC ≥ 0.82 on CryptoBench
```

### Phase 8 (Week 16)
```
□ Ensemble voting functional (5 reservoirs)
□ Transfer learning saves/loads family backbones
□ Uncertainty calibration ECE < 0.10
□ ROC AUC ≥ 0.90 on CryptoBench
□ Matches or exceeds PocketMiner (0.87)
□ Zero external dependencies maintained
```

---

## Strategic Outcome

After Phase 8:

| Attribute | Status |
|-----------|--------|
| Accuracy | **Category Leader** (0.90 vs 0.87 SOTA) |
| Sovereignty | **100%** (zero external deps) |
| Confidence | **Calibrated** (uncertainty quantification) |
| Efficiency | **Optimized** (active learning) |
| Scalability | **Compounding** (transfer learning) |

**PRISM becomes the only sovereign AI system that matches or exceeds published deep learning methods for cryptic site detection.**

---

## Documents

| Document | Content |
|----------|---------|
| `PRISM_PHASE7_8_PLAN_PART1.md` | Weeks 1-4, Hierarchical + TDA |
| `PRISM_PHASE7_8_PLAN_PART2.md` | Weeks 5-16, Ensemble + Transfer |
| `PRISM_PHASE7_8_QUICK_REFERENCE.md` | This document |

---

**Execute after Phase 6 completion.**
