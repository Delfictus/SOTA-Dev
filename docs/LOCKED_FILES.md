# PRISM-Delta Locked Files Registry

**Last Updated:** 2026-01-09
**Best Achieved Performance:** CA-GNM + Distance Weighting = ρ = 0.6204

---

## Purpose

This document registers all files that are **LOCKED** and must NOT be modified.
These files represent validated, working implementations. Any new features must
be implemented in NEW files using the **copy-and-create** pattern.

---

## Locked Files by Module

### prism-physics (Core Physics Engine)

| File | Purpose | Performance | Status |
|------|---------|-------------|--------|
| `src/gnm.rs` | Plain Gaussian Network Model | ρ = 0.6007 | 🔒 LOCKED |
| `src/gnm_enhanced.rs` | Distance-weighted GNM | ρ = 0.6140 | 🔒 LOCKED |
| `src/gnm_chemistry.rs` | Chemistry-Aware GNM | ρ = 0.6204 | 🔒 LOCKED |
| `src/residue_chemistry.rs` | AA flexibility factors | - | 🔒 LOCKED |
| `src/dynamics_engine.rs` | Unified dynamics interface | - | 🔒 LOCKED |
| `src/secondary_structure.rs` | Helix/sheet/loop detection | - | 🔒 LOCKED |
| `src/sidechain_analysis.rs` | Residue flexibility factors | - | 🔒 LOCKED |
| `src/tertiary_analysis.rs` | SASA, burial depth | - | 🔒 LOCKED |

### prism-validation (Benchmarking)

| File | Purpose | Status |
|------|---------|--------|
| `src/bin/run_heterogeneous_bench.rs` | Main GNM benchmark runner | 🔒 LOCKED |
| `src/bin/gnm_chemistry_bench.rs` | CA-GNM benchmark with ablation | 🔒 LOCKED |
| `src/bin/gnm_ablation.rs` | Ablation study tool | 🔒 LOCKED |
| `src/bin/gnm_breakthrough.rs` | Physics experiments | 🔒 LOCKED |
| `Cargo.toml` | Binary definitions | 🔒 LOCKED (add only) |

### prism-gpu (GPU Kernels)

| File | Purpose | Status |
|------|---------|--------|
| `src/kernels/prism_nova.cu` | Core CUDA kernels | 🔒 LOCKED |
| `src/prism_nova.rs` | Rust GPU wrapper | 🔒 LOCKED |

### Configuration & Data

| File | Purpose | Status |
|------|---------|--------|
| `data/atlas_alphaflow/atlas_targets.json` | Benchmark targets | 🔒 LOCKED |
| `data/atlas_alphaflow/pdb/*.pdb` | Reference structures | 🔒 LOCKED |

---

## Rules for New Development

### DO ✅

1. **CREATE new files** for new functionality
2. **COPY working code** as starting point (then modify the copy)
3. **ADD new binaries** to Cargo.toml (don't modify existing binary configs)
4. **CREATE new benchmark runners** for new modes
5. **Document new files** in this registry when they become stable

### DO NOT ❌

1. **NEVER modify** any file marked 🔒 LOCKED
2. **NEVER add conditional logic** to locked files for new features
3. **NEVER refactor** locked files "for improvement"
4. **NEVER change defaults** in locked configuration structs
5. **NEVER rename** functions/structs in locked files

---

## New File Locations (Phase 2+)

### Phase 2: AMBER MD (All-Atom Dynamics)

```
crates/prism-physics/src/
├── amber_topology.rs      # NEW: Bond/angle/dihedral topology
├── amber_ff14sb.rs        # NEW: AMBER ff14SB parameters
├── amber_dynamics.rs      # NEW: All-atom integrator
└── amber_analysis.rs      # NEW: Trajectory analysis

crates/prism-gpu/src/
├── kernels/amber_bonded.cu  # NEW: Angle/dihedral CUDA kernels
└── amber_simulator.rs       # NEW: GPU wrapper for AMBER

crates/prism-validation/src/bin/
└── run_amber_bench.rs       # NEW: AMBER benchmark runner
```

### Phase 3: ML-Corrected GNM

```
crates/prism-physics/src/
├── gnm_ml_refiner.rs      # NEW: ML residual correction
└── per_residue_features.rs # NEW: Feature extraction

crates/prism-validation/src/bin/
├── train_ml_gnm.rs        # NEW: Training script
└── run_ml_gnm_bench.rs    # NEW: ML-GNM benchmark
```

### Phase 4: LBS/Cryptic Site Detection

```
crates/prism-lbs/src/
├── cryptic_detector.rs    # NEW: Cryptic site detection
├── ensemble_pockets.rs    # NEW: Multi-conformer pocket analysis
└── druggability_ml.rs     # NEW: ML druggability scoring

crates/prism-validation/src/bin/
└── run_cryptic_bench.rs   # NEW: Cryptic site benchmark
```

---

## Version History

| Date | Commit | Change | Best ρ |
|------|--------|--------|--------|
| 2026-01-09 | `928f4fb` | Enhanced GNM ablation optimization | 0.615 |
| 2026-01-09 | `7945087` | Chemistry-Aware GNM implementation | 0.6204 |

---

## Verification Command

To verify the locked GNM performance hasn't regressed:

```bash
cargo run --release -p prism-validation --bin gnm-chemistry-bench --features prism-physics
```

Expected output:
```
CA-GNM + DW (BEST)  ρ = 0.6204  ≥0.7: 26/81  ≥0.6: 48/81  pass=96.3%
```

---

## Contact

If you believe a locked file needs modification, document:
1. The specific change needed
2. Why it can't be done in a new file
3. Risk assessment for existing functionality
4. Rollback plan if regression occurs
