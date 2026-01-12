# PRISM Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRISM CRYPTIC SITE DETECTION                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                                                                      │
│  ┌─────────────┐                                                           │
│  │ PDB Structure│──────────────────────────────────────────────┐           │
│  └─────────────┘                                               │           │
│         │                                                       │           │
│         ▼                                                       ▼           │
│  ┌─────────────┐                                    ┌─────────────────┐    │
│  │PdbSanitizer │                                    │   Blake3 Hash   │    │
│  └─────────────┘                                    │  (fingerprint)  │    │
│         │                                           └─────────────────┘    │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         HYBRID SAMPLER                               │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                      SAMPLING ROUTER                          │  │   │
│  │  │  n_atoms <= 512 ?  ──YES──> NOVA Path (TDA + Active Inference)│  │   │
│  │  │                    ──NO───> AMBER Path (Full Atom MD)         │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │              SamplingResult (conformations + energy)          │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         FEATURE EXTRACTION                           │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │   │
│  │  │ CrypticFeatures│  │ Betti Numbers  │  │ Blake3 Cache Key       │ │   │
│  │  │   (16-dim)     │  │ (β₀, β₁, β₂)  │  │ (TDA result caching)   │ │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │   │
│  │         │                    │                                       │   │
│  │         └────────────────────┴───────────────────┐                   │   │
│  │                                                  ▼                   │   │
│  │                              ┌────────────────────────────────────┐  │   │
│  │                              │  40-dim Input (16 feat + velocity) │  │   │
│  │                              └────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      GPU ZrO CRYPTIC SCORER                          │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │              DendriticSNNReservoir (512 neurons)               │ │   │
│  │  │  • LIF neurons with dendritic compartments                     │ │   │
│  │  │  • Sparse connectivity (10%)                                   │ │   │
│  │  │  • Spectral radius 0.95                                        │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │              RLS Online Learning (λ=0.99)                      │ │   │
│  │  │  • Readout weights updated per residue                         │ │   │
│  │  │  • Numerical stability guards                                  │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  OUTPUT                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Per-Residue Cryptic Scores (0-1)                                   │   │
│  │  • Score > 0.5: Predicted cryptic site                              │   │
│  │  • Spatial clustering for site identification                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. AMBER Physics Engine (Primary)
- Full-atom ff14SB force field
- Bonds, angles, dihedrals, nonbonded
- O(N) cell lists for scaling
- No atom limit

### 2. NOVA Sampler (Optional, ≤512 atoms)
- Neural HMC integration
- TDA-guided sampling (Betti feedback)
- Active Inference (EFE minimization)
- Shared memory limited to 512 atoms

### 3. TDA Module
- Alpha complex filtration
- Betti number computation (β₀, β₁, β₂)
- GPU-accelerated via prism_gpu::tda
- Blake3 caching for results

### 4. Neuromorphic Reservoir
- 512 LIF neurons
- Dendritic compartments
- 10% sparse connectivity
- Spectral radius 0.95

### 5. RLS Online Learning
- Lambda = 0.99 (forgetting factor)
- Per-residue weight updates
- Stability guards (clamp, reset)

---

## Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Reservoir neurons | 512 | gpu_zro_cryptic_scorer.rs |
| RLS lambda | 0.99 | gpu_zro_cryptic_scorer.rs |
| Feature dimensions | 16 | cryptic_features.rs |
| Input with velocity | 40 | cryptic_features.rs |
| NOVA atom limit | 512 | prism_nova.rs |
| Spectral radius | 0.95 | reservoir config |
| Connectivity density | 0.10 | reservoir config |

---

## Data Flow

```
PDB → Sanitize → Route → Sample → Extract Features → Score → Output
         │                  │            │
         ▼                  ▼            ▼
    Blake3 hash      TDA (Betti)   Blake3 cache
```

---

## Phase Evolution

| Phase | Scoring | Sampling | Features | Target AUC |
|-------|---------|----------|----------|------------|
| 6 | 512 neurons | 500 samples | 16-dim | 0.75 |
| 7 | 1,280 neurons | 2,000 samples | 67-dim | 0.82 |
| 8 | 5×1,280 ensemble | 2,000 samples | 67-dim | 0.90 |

---

*Reference document for architecture decisions.*
