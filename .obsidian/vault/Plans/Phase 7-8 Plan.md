# Phase 7-8: SOTA Enhancement Plan

> **Status**: PENDING (Requires Phase 6 completion)
> **Prerequisites**: Phase 6 ROC AUC >= 0.70
> **Timeline**: 16 Weeks (Phase 7: 8 weeks, Phase 8: 8 weeks)
> **Target**: ROC AUC >= 0.90, PR AUC >= 0.40

---

## Full Implementation Plans (IN VAULT)

| Document | Vault Path | Content |
|----------|------------|---------|
| **Part 1** | [[Plans/Full/PRISM_PHASE7_8_PLAN_PART1\|Phase 7-8 Part 1]] | Phase 7 detailed spec |
| **Part 2** | [[Plans/Full/PRISM_PHASE7_8_PLAN_PART2\|Phase 7-8 Part 2]] | Phase 8 detailed spec |
| **Quick Reference** | [[Plans/Full/PRISM_PHASE7_8_QUICK_REFERENCE\|Phase 7-8 Quick Ref]] | Summary card |
| **Master Trajectory** | [[Plans/Full/PRISM_MASTER_IMPLEMENTATION_TRAJECTORY\|Trajectory]] | Overall roadmap |

---

## Objective

Transform PRISM from **competitive** (Phase 6: ~0.75 AUC) to **category leader** (Phase 8: ~0.90 AUC) while maintaining complete sovereignty and AMBER-primary physics.

---

## Target Metrics Progression

| Metric | Phase 6 | Phase 7 | Phase 8 | SOTA |
|--------|---------|---------|---------|------|
| ROC AUC | 0.75 | **0.82** | **0.90** | PocketMiner 0.87 |
| PR AUC | 0.25 | **0.32** | **0.40** | PocketMiner 0.44 |
| Success Rate | 85% | **88%** | **92%** | - |
| Time/Structure | <1s | **<1.5s** | **<2s** | - |

---

## Architecture Principle

```
AMBER-PRIMARY:
- AMBER is the PRIMARY physics engine (all structures)
- NOVA is OPTIONAL for small proteins (<=512 atoms)
- Enhancements apply to SCORING LAYER (backend-agnostic)
- Hierarchical reservoir processes features from EITHER backend

CORE FEATURES MAINTAINED:
* Betti Numbers (TDA): beta_0, beta_1, beta_2
* Blake3 Hashing: fingerprints, caching, integrity

WHAT PHASE 7-8 ENHANCES:
+ Scoring: 512 -> 1,280 hierarchical neurons
+ Features: 16-dim -> 67-dim (multi-scale + persistence)
+ TDA: Betti counts -> Full persistence diagrams
+ Ensemble: Single scorer -> 5-reservoir voting
+ Learning: Per-structure -> Transfer learning
```

---

## Phase 7: Architectural Enhancements (8 weeks)

### Weeks 1-2: Hierarchical Reservoir
- [ ] `scoring/hierarchical_reservoir.rs`
- Architecture: 1,280 neurons (8x64 + 4x128 + 1x256)
- Layer 1: Local detectors (512 neurons)
- Layer 2: Regional integration (512 neurons)
- Layer 3: Global context (256 neurons)

### Weeks 3-4: Persistent Homology
- [ ] `features/persistence_diagrams.rs`
- Full persistence diagrams (birth, death, persistence)
- 24-dim TDA features per conformation
- Blake3-cached results

### Weeks 5-6: Extended Sampling
- [ ] `sampling/extended_ensemble.rs`
- 500 -> 2,000 conformations for AMBER
- Stratified temperature sampling
- Metadynamics for barrier crossing

### Weeks 7-8: Multi-Scale Features
- [ ] `features/multi_scale.rs`
- 67-dim extended feature vector
- Local (residue), regional (10A), global (chain) scales
- Surface/core context features

---

## Phase 8: Intelligence Enhancements (8 weeks)

### Weeks 1-2: Ensemble Voting
- [ ] `scoring/ensemble_scorer.rs`
- 5 reservoirs with different initializations
- Weighted voting by confidence
- Disagreement-based uncertainty

### Weeks 3-4: Transfer Learning
- [ ] `learning/transfer_backbone.rs`
- Pre-trained family-specific weights
- Kinase, GPCR, protease backbones
- Fine-tuning protocol

### Weeks 5-6: Uncertainty Quantification
- [ ] `scoring/uncertainty.rs`
- Ensemble variance
- Epistemic vs aleatoric
- Calibrated confidence intervals

### Weeks 7-8: Active Learning
- [ ] `pipeline/active_learning.rs`
- High-uncertainty sample selection
- Human-in-loop validation
- Continuous improvement protocol

---

## Gate Requirements

### Phase 6 -> Phase 7
```
[ ] ROC AUC >= 0.70
[ ] PR AUC >= 0.20
[ ] Success Rate >= 80%
[ ] All Phase 6 tests passing
[ ] results/PHASE6_FINAL.json exists
```

### Phase 7 -> Phase 8
```
[ ] ROC AUC >= 0.80
[ ] Hierarchical reservoir operational
[ ] Persistence features extracted
[ ] Shadow validation passed vs Phase 6
[ ] results/PHASE7_FINAL.json exists
```

---

## Links

- [[Plans/Phase 6 Plan|Phase 6 Plan]]
- [[Architecture/Architecture Overview|Architecture Overview]]
- [[Sessions/Current Session|Current Session]]
- [[PRISM Dashboard|Dashboard]]

---

*DO NOT IMPLEMENT until Phase 6 gates pass. Full specifications in the linked plan documents above.*
