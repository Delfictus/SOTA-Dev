# PRISM Phase 7-8: Enhancement Implementation Plan
## From Competitive to Category Leader (AMBER-Primary Architecture)

**Document Version**: 1.0
**Generated**: 2026-01-12
**Prerequisites**: Phase 6 Complete (ROC AUC >= 0.70)
**Timeline**: 16 Weeks (Phase 7: 8 weeks, Phase 8: 8 weeks)
**Target**: ROC AUC >= 0.90, PR AUC >= 0.40
**Classification**: Zero Fallback / Zero External Dependencies

---

## CRITICAL: AMBER-Primary Architecture

**Phase 7-8 enhancements build on the FULL-ATOM AMBER foundation established in Phase 6.**

```
ARCHITECTURE PRINCIPLE:
- AMBER is the PRIMARY physics engine (all structures)
- NOVA is OPTIONAL for small proteins (<=512 atoms) where TDA adds value
- Enhancements apply to SCORING LAYER (backend-agnostic)
- Hierarchical reservoir processes features from EITHER backend
- TDA can be computed post-hoc on AMBER trajectories

CORE PRISM FEATURES (MAINTAINED THROUGHOUT):
  * Betti Numbers (TDA topology):
    - beta_0: Connected components
    - beta_1: Loops/tunnels (pocket indicators)
    - beta_2: Voids/cavities (cryptic site signatures)
  * Blake3 Hashing (integrity & caching):
    - Structure fingerprinting
    - Conformation cache keys
    - Result integrity verification
    - Reproducibility checksums

WHAT PHASE 7-8 ENHANCES:
  + Scoring: 512 neurons -> 1,280 hierarchical neurons
  + Features: 16-dim -> 67-dim (multi-scale + persistence)
  + TDA: Betti counts -> Full persistence diagrams
  + Ensemble: Single scorer -> 5-reservoir voting
  + Learning: Per-structure -> Transfer learning

WHAT PHASE 7-8 DOES NOT CHANGE:
  - AMBER physics engine (frozen after Phase 6)
  - SamplingBackend contract (frozen)
  - Hybrid router logic (frozen)
  - Zero fallback policy (maintained)
  - Blake3 hashing infrastructure
  - Betti number computation (extended, not replaced)
```

---

## 1. Executive Summary

### Objective

Transform PRISM from **competitive** (Phase 6: ~0.75 AUC) to **category leader** (Phase 8: ~0.90 AUC) while maintaining complete sovereignty and AMBER-primary physics.

### Target Metrics Progression

| Metric | Phase 6 | Phase 7 | Phase 8 | SOTA |
|--------|---------|---------|---------|------|
| ROC AUC | 0.75 | **0.82** | **0.90** | PocketMiner 0.87 |
| PR AUC | 0.25 | **0.32** | **0.40** | PocketMiner 0.44 |
| Success Rate | 85% | **88%** | **92%** | - |
| Time/Structure | <1s | **<1.5s** | **<2s** | - |

### Enhancement Components

| Phase | Component | Expected Gain | Layer Affected |
|-------|-----------|---------------|----------------|
| 7 | Hierarchical Reservoir | +0.03 | Scoring |
| 7 | Persistent Homology | +0.02 | Features |
| 7 | Extended Sampling | +0.02 | Sampling (AMBER) |
| 7 | Multi-Scale Features | +0.02 | Features |
| 8 | Ensemble Voting | +0.03 | Scoring |
| 8 | Transfer Learning | +0.03 | Learning |
| 8 | Uncertainty | (quality) | Output |
| 8 | Active Learning | (efficiency) | Pipeline |

---

## 2. Phase Transition Requirements

### Phase 6 -> Phase 7 Gate (MANDATORY)

```
[ ] CryptoBench ROC AUC >= 0.70
[ ] CryptoBench PR AUC >= 0.20
[ ] Success Rate >= 80%
[ ] GPU scorer operational (512 neurons)
[ ] AMBER sampling functional (all sizes)
[ ] Hybrid router tested
[ ] All Phase 6 tests passing
[ ] results/PHASE6_FINAL.json exists
```

### Phase 7 -> Phase 8 Gate (MANDATORY)

```
[ ] CryptoBench ROC AUC >= 0.80
[ ] Hierarchical reservoir operational (1,280 neurons)
[ ] Persistence features extracted
[ ] Extended sampling working (2,000 conformations)
[ ] Shadow validation passed vs Phase 6
[ ] results/PHASE7_FINAL.json exists
```

---

## 3. Phase 7: Architectural Enhancements (Weeks 1-8)

### 3.1 Hierarchical Neuromorphic Reservoir (Weeks 1-2)

**Objective**: Replace flat 512-neuron reservoir with cortical column hierarchy.

```
Architecture:

Layer 3: Global Context     [1 x 256 = 256 neurons]
         ^
Layer 2: Regional Integration [4 x 128 = 512 neurons]
         ^
Layer 1: Local Detectors    [8 x 64 = 512 neurons]
         ^
Input: 80-dim features (from AMBER or NOVA sampling)

Total: 1,280 neurons (2.5x Phase 6)
VRAM: ~3.2 MB (fits on RTX 3060)
```

**Key Point**: Features come from AMBER sampling. The reservoir is backend-agnostic.

**File**: `crates/prism-validation/src/scoring/hierarchical_reservoir.rs`

### 3.2 Persistent Homology with Full Betti Enhancement (Weeks 3-4)

**Objective**: Extend Betti number computation to full persistence diagrams on AMBER trajectories.

```
BETTI NUMBER EVOLUTION:

Phase 6 (Existing):
  - beta_0: Count of connected components
  - beta_1: Count of loops/tunnels
  - beta_2: Count of voids/cavities
  - Computed via prism_gpu::tda module
  - 3 features per conformation

Phase 7 (Enhancement):
  - PRESERVES all Phase 6 Betti computation
  - ADDS persistence diagram analysis:
    * Birth/death times for each feature
    * Lifetime = death - birth (significance)
    * Persistence entropy
  - 31 additional features (34 total TDA features)

PERSISTENCE FEATURES (per Betti dimension):
  | Feature | Description | Cryptic Relevance |
  |---------|-------------|-------------------|
  | count | Number of features | Pocket count |
  | total_persistence | Sum of lifetimes | Overall topology |
  | max_lifetime | Longest-lived feature | Dominant pocket |
  | mean_lifetime | Average significance | Pocket stability |
  | entropy | Feature distribution | Complexity |
  | birth_mean/std | When features form | Pocket depth |
  | death_mean/std | When features close | Pocket transience |

CRYPTIC-SPECIFIC (beta_2 focus):
  - void_birth_threshold: Earliest cavity formation
  - stable_void_count: Cavities with lifetime > 1.0
  - void_volume_proxy: lifetime * birth_radius^3
  - pocket_opening_score: Rate of beta_2 increase

Computed ON:
- AMBER conformations (Calpha coordinates)
- Blake3 hash for conformation caching
- GPU-accelerated via prism_gpu::tda
```

**Files**:
- `crates/prism-validation/src/features/persistent_homology.rs`
- `crates/prism-gpu/src/kernels/persistence.cu`

**Integration with Blake3**:
```rust
// Each conformation's TDA results are cached with Blake3 hash
let conformation_hash = blake3::hash(&coords_bytes);
if let Some(cached) = tda_cache.get(&conformation_hash) {
    return cached;
}
let features = compute_persistence(&coords);
tda_cache.insert(conformation_hash, features.clone());
```

### 3.3 Extended AMBER Sampling (Weeks 5-6)

**Objective**: Increase AMBER conformational sampling from 500 to 2,000 samples.

```
Parameter Changes:

| Parameter | Phase 6 | Phase 7 |
|-----------|---------|---------|
| Samples | 500 | 2,000 |
| Steps/Sample | 100 | 50 |
| Temperature | 310K | 310K -> 290K annealing |

NOTE: Uses existing AmberMegaFusedHmc - no physics changes.
Extended sampling is purely more iterations.
```

**File**: `crates/prism-validation/src/sampling/extended_amber_sampler.rs`

### 3.4 Multi-Scale Features (Weeks 5-6)

**Objective**: Extract features at local, regional, and global scales.

```
Scale 1: Local (per-residue) - 16 features (Phase 6)
Scale 2: Regional (5-12A neighborhood) - 12 features
Scale 3: Global (whole protein) - 8 features

Total: 36 static + 31 persistence = 67 features
With velocities: 67 + 16 = 83 -> padded to 80 input
```

**File**: `crates/prism-validation/src/features/multiscale_features.rs`

### 3.5 Phase 7 Integration (Weeks 7-8)

**Objective**: Integrate all Phase 7 components into unified scorer.

**File**: `crates/prism-validation/src/phase7_scorer.rs`

---

## 4. Phase 8: Advanced Capabilities (Weeks 9-16)

### 4.1 Ensemble Reservoir Voting (Weeks 9-10)

**Objective**: 5 reservoirs with different seeds, learned combination weights.

```
Input Features (80-dim from AMBER)
       |
+------+------+------+------+------+
|      |      |      |      |      |
R1     R2     R3     R4     R5     (different seeds)
|      |      |      |      |      |
+------+------+------+------+------+
               |
        Weighted Mean + Uncertainty

Expected Gain: +0.03 AUC (variance reduction)
```

**File**: `crates/prism-validation/src/scoring/ensemble_reservoir.rs`

### 4.2 Transfer Learning (Weeks 11-13)

**Objective**: Transfer learned patterns between related protein structures.

```
Strategy:
1. Train on structure A (kinase) -> backbone weights W_A
2. Train on structure B (kinase) -> backbone weights W_B
3. Aggregate: W_family = mean(W_A, W_B, ...)
4. New kinase: Initialize with W_family, adapt via RLS

Works with AMBER because:
- Features are structure-derived (not backend-specific)
- Transfer is at scoring layer
- AMBER provides consistent conformational sampling
```

**File**: `crates/prism-validation/src/transfer/transfer_learning.rs`

### 4.3 Uncertainty Quantification (Week 14)

**Objective**: Calibrated confidence scores for predictions.

```
Sources of Uncertainty:
- Epistemic: Ensemble disagreement (5 reservoirs)
- Aleatoric: Feature variance from AMBER sampling

Output:
- Point prediction
- 95% confidence interval
- Calibrated confidence score (ECE < 0.10)
```

**File**: `crates/prism-validation/src/uncertainty/uncertainty.rs`

### 4.4 Active Learning (Weeks 15-16)

**Objective**: Prioritize structures for maximum model improvement.

```
Acquisition Function:
Score = alpha * Uncertainty + beta * Diversity + gamma * Info_Gain

Selects structures where AMBER sampling + scoring is most valuable.
```

**File**: `crates/prism-validation/src/active/active_learning.rs`

---

## 5. File Manifest

### Phase 7 Files (8 new)

| File | Location | Purpose |
|------|----------|---------|
| `hierarchical_reservoir.rs` | `scoring/` | 1,280-neuron cortical columns |
| `hierarchical_reservoir.cu` | `kernels/` | CUDA kernels for layers |
| `persistent_homology.rs` | `features/` | TDA on AMBER conformations |
| `persistence.cu` | `kernels/` | CUDA for TDA |
| `multiscale_features.rs` | `features/` | Local + regional + global |
| `extended_amber_sampler.rs` | `sampling/` | 2,000-sample AMBER wrapper |
| `phase7_scorer.rs` | root | Integrated Phase 7 pipeline |
| `phase7_tests.rs` | `tests/` | Phase 7 validation |

### Phase 8 Files (6 new)

| File | Location | Purpose |
|------|----------|---------|
| `ensemble_reservoir.rs` | `scoring/` | 5-reservoir voting |
| `transfer_learning.rs` | `transfer/` | Family backbone transfer |
| `uncertainty.rs` | `uncertainty/` | Calibrated confidence |
| `active_learning.rs` | `active/` | Structure prioritization |
| `phase8_scorer.rs` | root | Complete Phase 8 system |
| `phase8_tests.rs` | `tests/` | Phase 8 validation |

---

## 6. Directory Structure (All Phases)

```
crates/prism-validation/src/
|-- lib.rs
|
|-- sampling/                    # PHASE 6 (FROZEN)
|   |-- contract.rs              # SamplingBackend trait
|   |-- paths/
|   |   |-- nova_path.rs         # TDA + AI (<=512 atoms)
|   |   +-- amber_path.rs        # Primary physics (all sizes)
|   |-- router/
|   +-- extended_amber_sampler.rs # PHASE 7: More iterations
|
|-- scoring/                     # NEUROMORPHIC LAYER
|   |-- gpu_zro_scorer.rs        # Phase 6: 512 neurons
|   |-- hierarchical_reservoir.rs # Phase 7: 1,280 neurons
|   +-- ensemble_reservoir.rs    # Phase 8: 5 x 1,280 neurons
|
|-- features/                    # FEATURE EXTRACTION
|   |-- cryptic_features.rs      # Phase 6: 16-dim
|   |-- multiscale_features.rs   # Phase 7: 36-dim
|   +-- persistent_homology.rs   # Phase 7: +31-dim TDA
|
|-- transfer/                    # PHASE 8
|   +-- transfer_learning.rs
|
|-- uncertainty/                 # PHASE 8
|   +-- uncertainty.rs
|
+-- active/                      # PHASE 8
    +-- active_learning.rs
```

---

## 7. AMBER Integration Points

### What Uses AMBER Directly

| Component | AMBER Usage |
|-----------|-------------|
| `amber_path.rs` | Primary sampling for all structures |
| `extended_amber_sampler.rs` | Wrapper for 2,000-sample runs |
| Feature extraction | Computes on AMBER conformations |
| Persistence | Computes on AMBER Calpha trajectories |

### What Is Backend-Agnostic

| Component | Works With |
|-----------|------------|
| `hierarchical_reservoir.rs` | Features from any backend |
| `ensemble_reservoir.rs` | Features from any backend |
| `transfer_learning.rs` | Structure-derived features |
| `uncertainty.rs` | Scores from any scorer |

### AMBER Components (DO NOT MODIFY)

```
FROZEN after Phase 6:
- prism_gpu::AmberMegaFusedHmc
- prism_gpu::AmberBondedForces
- prism_physics::amber_topology
- crates/prism-gpu/src/kernels/amber_bonded.cu
```

---

## 8. Verification Commands

### Phase 7 Checkpoint (Week 8)

```bash
# Compile check
cargo check -p prism-validation --features cuda

# Hierarchical reservoir tests
cargo test --release -p prism-validation --features cuda hierarchical

# TDA feature tests
cargo test --release -p prism-validation --features cuda persistence

# Full Phase 7 benchmark
cargo run --release -p prism-validation --bin cryptobench -- \
    --config phase7 \
    --output results/phase7_full.json

# Verify: ROC AUC >= 0.82
```

### Phase 8 Final (Week 16)

```bash
# Ensemble tests
cargo test --release -p prism-validation --features cuda ensemble

# Transfer learning tests
cargo test --release -p prism-validation --features cuda transfer

# Full Phase 8 benchmark
cargo run --release -p prism-validation --bin cryptobench -- \
    --config phase8 \
    --output results/phase8_full.json

# Verify: ROC AUC >= 0.90
```

---

## 9. Success Criteria

### Phase 7 Complete (Week 8)

```
[ ] Hierarchical reservoir: 1,280 neurons, <10ms/step
[ ] Persistence features: 31 dimensions from AMBER conformations
[ ] Multi-scale features: 36 dimensions total
[ ] Extended AMBER sampling: 2,000 conformations
[ ] CryptoBench ROC AUC >= 0.82
[ ] All tests passing
```

### Phase 8 Complete (Week 16)

```
[ ] Ensemble: 5 reservoirs, <50ms combined
[ ] Transfer learning: 5+ family backbones
[ ] Uncertainty: ECE < 0.10
[ ] CryptoBench ROC AUC >= 0.90
[ ] Exceeds PocketMiner (0.87 AUC)
[ ] Zero external dependencies
[ ] Publication-ready results
```

---

## 10. Strategic Outcome

After Phase 8, PRISM will be:

| Attribute | Status |
|-----------|--------|
| Accuracy | **Category Leader** (0.90 vs 0.87 SOTA) |
| Physics | **Full-Atom AMBER** (proven, stable) |
| Sovereignty | **100%** (zero external deps) |
| Confidence | **Calibrated** (uncertainty quantification) |
| Scalability | **Compounding** (transfer learning) |

**PRISM becomes the only sovereign AI system using full-atom AMBER physics that matches or exceeds published deep learning methods for cryptic site detection.**

---

**Execute Phase 7 ONLY after Phase 6 completion and gate requirements met.**
