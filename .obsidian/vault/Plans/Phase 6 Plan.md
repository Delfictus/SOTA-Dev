# Phase 6: Cryptic Site Detection SOTA

> **Full Plan**: `results/phase6_sota_plan.md`
> **Status**: APPROVED FOR EXECUTION
> **Timeline**: 8 Weeks

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
- [ ] Environment verification (rustc 1.75+, nvcc 12.0+)
- [ ] Download CryptoBench dataset (1107 structures)
- [ ] Download apo-holo pairs (15 pairs)
- [ ] Document Phase 5 baseline

### Weeks 1-2: GPU SNN Scale-Up
- [x] `cryptic_features.rs` - 16-dim feature vector
- [ ] `gpu_zro_cryptic_scorer.rs` - 512-neuron reservoir + RLS
- [ ] `tests/gpu_scorer_tests.rs` - Zero fallback verification

### Weeks 3-4: Parallel Sampling
- [ ] `pdb_sanitizer.rs` - Structure preprocessing
- [ ] `sampling/contract.rs` - SamplingBackend trait (THE LAW)
- [ ] `sampling/paths/nova_path.rs` - TDA + Active Inference
- [ ] `sampling/paths/amber_path.rs` - Full-atom AMBER MD
- [ ] `sampling/router/mod.rs` - Hybrid router
- [ ] `sampling/shadow/comparator.rs` - Output comparison
- [ ] `sampling/migration/feature_flags.rs` - Rollout control
- [ ] `apo_holo_benchmark.rs` - Conformational validation

### Weeks 5-6: CryptoBench & Ablation
- [ ] `cryptobench_dataset.rs` - Dataset loader
- [ ] `ablation.rs` - 6-variant ablation study
- [ ] `failure_analysis.rs` - Error categorization

### Weeks 7-8: Publication
- [ ] `publication_outputs.rs` - LaTeX tables/figures
- [ ] Final metrics validation
- [ ] Publication package

---

## Architectural Constraints

```
REQUIRED:
- PRISM-ZrO (SNN + RLS) for adaptive learning
- PRISM-NOVA (HMC + AMBER) for enhanced sampling
- Native Rust/CUDA implementations
- GPU-mandatory execution (no silent CPU fallback)

FORBIDDEN:
- PyTorch, TensorFlow, or external ML models
- Silent fallback to CPU
- Mock implementations or placeholder returns
- Data leakage between train/test splits
- todo!() or unimplemented!() in production code
```

---

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Reservoir neurons | 512 |
| RLS lambda | 0.99 |
| Feature dimensions | 16 (40 with velocity) |
| NOVA atom limit | 512 |
| Train/test split | 885/222 |

---

## Checkpoints

| Week | Gate | Command |
|------|------|---------|
| 2 | GPU scorer tests pass | `cargo test -p prism-validation --features cuda gpu_scorer` |
| 4 | Apo-holo RMSD < 3.5A | `cargo run --bin apo-holo-single -- --apo 1AKE --holo 4AKE` |
| 6 | ROC AUC >= 0.70 | `cargo run --bin cryptobench -- --manifest manifest.json` |
| 8 | All metrics pass | Full publication validation |

---

## Links

- [[Architecture Overview]]
- [[Phase 7-8 Plan]]
- [[Current Session]]
- [[PRISM Dashboard]]

---

*See `results/phase6_sota_plan.md` for complete implementation details.*
