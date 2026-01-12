# PRISM Phase 6: Quick Reference & Claude Code Task List

## Document Overview

This implementation plan is split into two detailed parts:
- **Part 1** (`PRISM_PHASE6_PLAN_PART1.md`): Weeks 0-2, GPU SNN Scale-Up
- **Part 2** (`PRISM_PHASE6_PLAN_PART2.md`): Weeks 3-8, NOVA, CryptoBench, Publication

---

## Executive Summary

**Goal**: Achieve SOTA cryptic site detection (ROC AUC >0.75) using ONLY native PRISM infrastructure.

**Timeline**: 8 weeks

**Constraints**:
- ✅ Native Rust/CUDA only
- ❌ NO PyTorch, TensorFlow, external ML
- ❌ NO CPU fallback (GPU mandatory)
- ❌ NO mock/placeholder implementations

---

## Target Metrics

| Metric | Current | Target | SOTA |
|--------|---------|--------|------|
| ROC AUC | 0.487 | **>0.75** | 0.87 |
| PR AUC | 0.081 | **>0.25** | 0.17 |
| Success Rate | 71.7% | **>85%** | 83% |
| Top-1 | 82.6% | **>90%** | 78% |
| Time | N/A | **<1s** | - |

---

## Claude Code Task List

### Week 0: Setup (Pre-requisite)

```
Task 0.1: Verify environment
- rustc --version (1.75+)
- nvcc --version (12.0+)
- cargo check -p prism-gpu --features cuda

Task 0.2: Download CryptoBench dataset (1107 structures)
- git clone https://github.com/skrhakv/CryptoBench.git
- Create manifest.json with 885/222 train/test split

Task 0.3: Download apo-holo pairs (15 pairs)
- Download from RCSB (see Part 1 for script)

Task 0.4: Document Phase 5 baseline metrics
- Create results/BASELINE_METRICS.md
```

### Weeks 1-2: GPU SNN Scale-Up

```
Task 1.1: Create cryptic_features.rs
"Implement CrypticFeatures struct with 16-dim feature vector.
Include encode_into() and encode_with_velocity() methods.
Add normalization and unit tests."

Task 1.2: Create gpu_zro_cryptic_scorer.rs  
"Implement GpuZroCrypticScorer with 512-neuron reservoir.
Include RLS online learning with Sherman-Morrison updates.
Add stability safeguards (gradient clamp, precision matrix reset).
CRITICAL: Must fail explicitly if no GPU available - NO CPU fallback."

Task 1.3: Create ensemble_cryptic_model.rs
"Implement ensemble model combining EFE, ZrO, TDA, interface scores.
Use RLS for adaptive weight learning.
Include save/load for trained weights."

Task 1.4: Create ensemble_quality_metrics.rs
"Implement metrics to validate sampling quality:
- Mean pairwise RMSD (target: 1-3Å)
- Radius of gyration variance
- Quality verdict (Excellent/Good/Poor/Failed)"

Task 1.5: Create gpu_scorer_tests.rs
"Write tests that verify:
1. test_no_cpu_fallback - MUST FAIL without GPU
2. test_rls_stability_1000_updates - no NaN/Inf
3. test_weight_persistence - save/load roundtrip
4. bench_gpu_scorer_throughput - >10k residues/sec"

Task 1.6: Integrate into pipeline
"Modify blind_validation_pipeline.rs to add --gpu-scorer flag.
Verify end-to-end test passes with GPU scorer."
```

### Weeks 3-4: NOVA Integration

```
Task 3.1: Create pdb_sanitizer.rs
"Implement PDB sanitization for GPU safety:
- Remove HETATM, waters
- Filter to standard amino acids
- Renumber atoms sequentially
- Extract Cα coordinates
CRITICAL: Raw PDBs crash GPU kernels."

Task 3.2: Create nova_cryptic_sampler.rs
"Implement NOVA HMC wrapper with:
- Temperature 310K, dt 2fs, 5 leapfrog steps
- 500 samples, 100 steps decorrelation
- TDA Betti-2 tracking
- Active Inference goal-directed sampling
Include ensemble quality validation."

Task 3.3: Create apo_holo_benchmark.rs
"Implement benchmark on 15 classic apo-holo pairs.
Compute min RMSD to holo state from ensemble.
Success criteria varies by motion type (1.5-3.5Å).
Generate markdown and LaTeX reports."

Task 3.4: Test on 3CSY (Ebola GP trimer)
"Run NOVA on multi-chain quaternary structure.
Verify interface residue detection works.
Check acceptance rate >20%."
```

### Weeks 5-6: CryptoBench & Ablation

```
Task 5.1: Create cryptobench_dataset.rs
"Implement dataset loader for 1107 structures.
Handle train/test split (885/222).
Include validation and ground truth lookup."

Task 5.2: Create cryptobench_benchmark.rs
"Implement full benchmark runner.
Compute ROC AUC, PR AUC, Success Rate, Top-1.
Generate per-structure results JSON."

Task 5.3: Create ablation.rs
"Implement 6 ablation variants:
1. ANM only (baseline)
2. ANM + GPU-SNN
3. NOVA only
4. NOVA + CPU-SNN (comparison only)
5. NOVA + GPU-SNN
6. Full pipeline
Generate comparison tables."

Task 5.4: Create failure_analysis.rs
"Categorize failures by reason:
- PocketTooDeep, LargeConformationalChange
- AllostericSite, CrystalContact
- MultiplePockets, etc.
Generate limitations summary."

Task 5.5: Run ablation study
"Execute all 6 variants on CryptoBench test set.
Verify Full > ANM-only by >0.20 AUC.
Generate ablation table."
```

### Weeks 7-8: Publication

```
Task 7.1: Create publication_outputs.rs
"Generate publication-ready outputs:
- LaTeX tables (main results, ablation, apo-holo)
- CSV data for plotting
- Methods section draft"

Task 7.2: Create figure generation scripts
"Python scripts for:
- ROC curve with CI
- PR curve
- Ablation bar chart
- Per-structure heatmap"

Task 7.3: Final benchmark sweep
"Run full pipeline on all test structures.
Verify metrics meet targets.
Document any failures."

Task 7.4: Package for release
"Update README with usage instructions.
Verify reproducibility from clean state.
Tag release version."
```

---

## Verification Checkpoints

### Week 2 Checkpoint
```bash
# GPU scorer tests pass
cargo test --release -p prism-validation --features cuda gpu_scorer

# No CPU fallback (this MUST FAIL)
CUDA_VISIBLE_DEVICES="" cargo test test_no_cpu_fallback

# Throughput benchmark
cargo test bench_gpu_scorer_throughput -- --nocapture
# Expected: >10,000 residues/second
```

### Week 4 Checkpoint
```bash
# NOVA sampling test
cargo run --release -p prism-validation --bin test-nova -- \
    --pdb test.pdb --samples 100

# Apo-holo on single pair
cargo run --release -p prism-validation --bin apo-holo-single -- \
    --apo 1AKE --holo 4AKE
# Expected: min RMSD < 3.5Å
```

### Week 6 Checkpoint
```bash
# Full CryptoBench
cargo run --release -p prism-validation --bin cryptobench -- \
    --manifest manifest.json --output results.json

# Check metrics
jq '.roc_auc, .pr_auc, .success_rate' results.json
# Expected: >0.70, >0.20, >0.80
```

### Week 8 Checkpoint
```bash
# Generate publication outputs
cargo run --release -p prism-validation --bin publication -- \
    --output results/publication/

# Verify all tables generated
ls results/publication/*.tex
```

---

## Key Technical Details

### Feature Vector (16-dim)
```
Dynamics (5): burial_change, rmsf, variance, neighbor_flexibility, burial_potential
Structural (3): ss_flexibility, sidechain_flexibility, b_factor
Chemical (3): net_charge, hydrophobicity, h_bond_potential
Distance (3): contact_density, sasa_change, nearest_charged_dist
Tertiary (2): interface_score, allosteric_proximity
```

### GPU Reservoir Parameters
```
Neurons: 512
Input dim: 40 (16 features + 16 velocities + 8 padding)
Topology: 10% sparse, 80% excitatory
RLS λ: 0.99
Precision init: 100 * I
Gradient clamp: ±1.0
```

### NOVA Sampling Parameters
```
Temperature: 310 K
Timestep: 2 fs
Leapfrog steps: 5
Samples: 500
Steps per sample: 100
Goal strength: 0.2 (Active Inference)
```

---

## Risk Quick Reference

| Issue | Solution |
|-------|----------|
| RLS explodes | Reset precision matrix when trace > 1e6 |
| Low acceptance | Increase temperature or reduce leapfrog steps |
| GPU OOM | Reduce reservoir to 256 neurons |
| Metric regression | Check per-structure logs, bisect |
| PDB crashes GPU | Run through sanitizer first |

---

## Success Definition

Phase 6 is complete when:

1. ✅ ROC AUC > 0.70 on CryptoBench test set
2. ✅ Ablation proves Full > Baseline by >0.20 AUC
3. ✅ Apo-holo achieves >60% success rate
4. ✅ No CPU fallback exists (verified by failing test)
5. ✅ Publication outputs generated (LaTeX, figures)
6. ✅ Code is reproducible from clean state

---

**Ready for execution. Start with Week 0 setup tasks.**
