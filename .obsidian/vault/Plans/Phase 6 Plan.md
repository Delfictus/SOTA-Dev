# Phase 6: Cryptic Site Detection SOTA

> **Status**: APPROVED FOR EXECUTION
> **Timeline**: 8 Weeks
> **Current Unit**: 1.3 (tests/gpu_scorer_tests.rs)

---

## Full Implementation Plans (IN VAULT)

| Document | Vault Path | Content |
|----------|------------|---------|
| **Part 1** | [[Plans/Full/PRISM_PHASE6_PLAN_PART1\|Phase 6 Part 1]] | Weeks 0-2 detailed spec |
| **Part 2** | [[Plans/Full/PRISM_PHASE6_PLAN_PART2\|Phase 6 Part 2]] | Weeks 3-8 detailed spec |
| **Quick Reference** | [[Plans/Full/PRISM_PHASE6_QUICK_REFERENCE\|Phase 6 Quick Ref]] | Summary card |
| **Parallel Architecture** | [[Plans/Full/PRISM_PARALLEL_IMPLEMENTATION_ARCHITECTURE\|Architecture]] | NOVA/AMBER paths |
| **Master Trajectory** | [[Plans/Full/PRISM_MASTER_IMPLEMENTATION_TRAJECTORY\|Trajectory]] | Overall roadmap |

---

## Objective

Achieve **publication-ready** cryptic site detection that **exceeds SOTA** using **ONLY native PRISM infrastructure**.

---

## Target Metrics

| Metric | Minimum | Target | SOTA Reference |
|--------|---------|--------|----------------|
| ROC AUC | 0.70 | >0.75 | PocketMiner 0.87 |
| PR AUC | 0.20 | >0.25 | CryptoBank 0.17 |
| Success Rate | 80% | >85% | Schrodinger 83% |
| Top-1 Accuracy | 85% | >90% | CrypTothML 78% |
| Time/Structure | <5s | <1s | RTX 3060 |
| Peak VRAM | <4GB | <2GB | RTX 3060 |

---

## Weekly Schedule

### Week 0: Setup
- [x] Environment verification (rustc 1.75+, nvcc 12.0+)
- [ ] Download CryptoBench dataset (1107 structures)
- [ ] Download apo-holo pairs (15 pairs)
- [ ] Document Phase 5 baseline

### Weeks 1-2: GPU SNN Scale-Up
- [x] `cryptic_features.rs` - 16-dim feature vector (Unit 1.1) - commit: c4d88c2
- [x] `gpu_zro_cryptic_scorer.rs` - 512-neuron reservoir + RLS (Unit 1.2) - commit: 5e55a7a
- [ ] `tests/gpu_scorer_tests.rs` - Zero fallback verification (Unit 1.3) **<- ACTIVE**

### Weeks 3-4: Parallel Sampling
- [ ] `pdb_sanitizer.rs` - Structure preprocessing (Unit 3.1)
- [ ] `sampling/contract.rs` - SamplingBackend trait (Unit 3.2)
- [ ] `sampling/paths/nova_path.rs` - TDA + Active Inference (Unit 3.3)
- [ ] `sampling/paths/amber_path.rs` - Full-atom AMBER MD (Unit 3.4)
- [ ] `sampling/router/mod.rs` - Hybrid router (Unit 3.5)
- [ ] `sampling/shadow/comparator.rs` - Output comparison (Unit 3.6)
- [ ] `sampling/migration/feature_flags.rs` - Rollout control (Unit 3.7)
- [ ] `apo_holo_benchmark.rs` - Conformational validation (Unit 3.8)

### Weeks 5-6: Benchmarking
- [ ] `cryptobench_dataset.rs` - Dataset loader (Unit 5.1)
- [ ] `ablation.rs` - 6-variant study (Unit 5.2)
- [ ] `failure_analysis.rs` - Error categorization (Unit 5.3)

### Weeks 7-8: Publication
- [ ] `publication_outputs.rs` - LaTeX tables/figures (Unit 7.1)

---

## Key Architecture

```
Hybrid Sampler:
- n_atoms <= 512: PRISM-NOVA (TDA + Active Inference)
- n_atoms > 512: AMBER MegaFused (Full MD, no TDA)

Zero Fallback Policy:
- GPU REQUIRED - explicit error if unavailable
- No silent CPU fallback
- No mock implementations
```

---

## Checkpoints

| Checkpoint | Gate Criteria |
|------------|---------------|
| Week 2 | GPU scorer tests pass, zero fallback verified |
| Week 4 | Apo-holo RMSD < 3.5A, shadow comparison pass |
| Week 6 | ROC AUC >= 0.70, ablation delta > 0.15 |
| Week 8 | All metrics pass, publication ready |

---

## Links

- [[Progress/implementation_status.json|Machine State]]
- [[Sessions/Current Session|Current Session]]
- [[PRISM Dashboard|Dashboard]]
- [[Plans/Phase 7-8 Plan|Phase 7-8 Plan]]

---

*Full specifications: Click the plan links above to read complete details.*
