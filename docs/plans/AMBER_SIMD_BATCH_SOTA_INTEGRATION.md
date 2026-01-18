# AMBER SIMD Batch SOTA Integration Plan

## Overview

This document details the integration of SOTA performance optimizations into `amber_simd_batch.rs`.

**Expected Speedup**: 5-12× combined improvement over current implementation

## Current Architecture Analysis

### Current Per-Step Flow (amber_simd_batch.rs)
```
For each MD step:
  1. Zero energy accumulators                    [~0.01ms]
  2. Zero cell counts                            [~0.05ms]
  3. Build cell list (ALL atoms)                 [~0.5ms]  ← BOTTLENECK
  4. For each structure:
     a. Upload single batch descriptor           [~0.02ms]
     b. Launch MD step kernel (phase 1):
        - Compute bonded forces                  [~0.3ms]
        - Compute non-bonded forces (cell list)  [~2-5ms] ← BOTTLENECK
        - Half-kick 1, drift
     c. Synchronize
  5. Rebuild cell list at new positions          [~0.5ms]  ← DUPLICATE
  6. For each structure:
     a. Launch MD step kernel (phase 2):
        - Recompute forces at new positions      [~2-5ms] ← DUPLICATE
        - Half-kick 2, thermostat
     c. Synchronize
  7. Apply H-constraints                         [~0.1ms]
```

**Current bottlenecks:**
- Cell list rebuilt EVERY step (even when atoms haven't moved much)
- Non-bonded forces recomputed TWICE per step (phase 1 & phase 2)
- No overlap between bonded and non-bonded computation
- FP32 parameters use full memory bandwidth
- Sequential structure processing (no pipeline overlap)

## Integration Todo List

### Phase 1: Verlet List Integration (2-3× speedup)

| Task | File | Description |
|------|------|-------------|
| 1.1 | amber_simd_batch.rs | Add `VerletList` member field |
| 1.2 | amber_simd_batch.rs | Initialize Verlet list in `new()` |
| 1.3 | amber_simd_batch.rs | Build Verlet list in `finalize_batch()` |
| 1.4 | amber_simd_batch.rs | Replace per-step cell list with `needs_rebuild()` check |
| 1.5 | amber_simd_batch.rs | Conditional rebuild only when skin exceeded |

**Impact**: Reduces neighbor list construction from every step to every 10-20 steps.

### Phase 2: FP16 Mixed Precision (1.3-1.5× speedup)

| Task | File | Description |
|------|------|-------------|
| 2.1 | amber_simd_batch.rs | Add FP16 parameter buffers |
| 2.2 | amber_simd_batch.rs | Convert LJ params in `finalize_batch()` |
| 2.3 | amber_simd_batch.rs | Add position-to-FP16 conversion per step |
| 2.4 | amber_simd_batch.cu | Update kernel to use FP16 params |

**Impact**: 50% reduction in memory bandwidth for parameter reads.

### Phase 3: Tensor Core Integration (2-4× speedup)

| Task | File | Description |
|------|------|-------------|
| 3.1 | amber_simd_batch.rs | Add `TensorCoreForces` member (optional) |
| 3.2 | amber_simd_batch.rs | Detect Tensor Core capability at init |
| 3.3 | amber_simd_batch.rs | Add tensor core non-bonded path |
| 3.4 | amber_simd_batch.rs | Fallback to standard kernel on non-TC GPUs |

**Impact**: WMMA instructions process 16×16 tiles in hardware for distance computation.

### Phase 4: Async Pipeline (1.1-1.3× speedup)

| Task | File | Description |
|------|------|-------------|
| 4.1 | amber_simd_batch.rs | Add `AsyncMdPipeline` member |
| 4.2 | amber_simd_batch.rs | Create separate streams for bonded/non-bonded |
| 4.3 | amber_simd_batch.rs | Overlap bonded forces with non-bonded |
| 4.4 | amber_simd_batch.rs | Pipeline Verlet check with MD step |

**Impact**: Hide latency by overlapping independent computations.

### Phase 5: Configuration & Testing

| Task | File | Description |
|------|------|-------------|
| 5.1 | amber_simd_batch.rs | Add `OptimizationConfig` struct |
| 5.2 | amber_simd_batch.rs | Add runtime feature flags |
| 5.3 | tests/ | Add correctness validation tests |
| 5.4 | benches/ | Add performance benchmarks |

## End-to-End Pipeline Impact

### Before Optimization (Current)
```
Structure Load → Topology Build → GPU Upload → MD Steps → Results
                                      ↓
                                 Per Step:
                                 ┌─────────────────────────────────────┐
                                 │ Build Cell List (0.5ms)             │
                                 │         ↓                           │
                                 │ Bonded Forces (0.3ms)               │
                                 │         ↓                           │
                                 │ Non-bonded Forces (3ms) ← O(N²) ish │
                                 │         ↓                           │
                                 │ Integration (0.1ms)                 │
                                 │         ↓                           │
                                 │ Rebuild Cell List (0.5ms)           │
                                 │         ↓                           │
                                 │ Non-bonded Forces (3ms) ← DUPLICATE │
                                 │         ↓                           │
                                 │ Integration (0.1ms)                 │
                                 │         ↓                           │
                                 │ H-Constraints (0.1ms)               │
                                 └─────────────────────────────────────┘
                                 Total: ~7.6ms per step
                                 → ~130 steps/sec
                                 → ~0.26 ns/day (1000 atom protein)
```

### After Optimization (Target)
```
Structure Load → Topology Build → GPU Upload → FP16 Convert → Verlet Build → MD Steps → Results
                                                    ↓                ↓
                                              One-time cost     One-time cost

                                 Per Step (Optimized):
                                 ┌─────────────────────────────────────────────────┐
                                 │                                                 │
                                 │  ┌─── Stream A ────┐  ┌─── Stream B ────┐      │
                                 │  │ Verlet Check    │  │ (parallel)      │      │
                                 │  │ (0.1ms)         │  │                 │      │
                                 │  │      ↓          │  │                 │      │
                                 │  │ [Rebuild?]──No──│──│─────────────────│──┐   │
                                 │  │      │Yes       │  │                 │  │   │
                                 │  │      ↓          │  │                 │  │   │
                                 │  │ Rebuild Verlet  │  │                 │  │   │
                                 │  │ (0.5ms/10-20)   │  │                 │  │   │
                                 │  └─────────────────┘  └─────────────────┘  │   │
                                 │           ↓                                │   │
                                 │  ┌─── Stream A ────┐  ┌─── Stream B ────┐  │   │
                                 │  │ Bonded Forces   │  │ Non-bonded TC   │  │   │
                                 │  │ (0.3ms)         │  │ (0.8ms w/WMMA)  │←─┘   │
                                 │  └────────┬────────┘  └────────┬────────┘      │
                                 │           └──────────┬─────────┘               │
                                 │                      ↓                         │
                                 │              Force Accumulate                  │
                                 │                      ↓                         │
                                 │              Integration (0.1ms)               │
                                 │                      ↓                         │
                                 │              H-Constraints (0.1ms)             │
                                 └─────────────────────────────────────────────────┘
                                 Total: ~1.0-1.5ms per step (amortized)
                                 → ~700-1000 steps/sec
                                 → ~1.4-2.0 ns/day (1000 atom protein)

                                 SPEEDUP: 5-8× per structure
```

## Runtime Architecture Flow Map

### Complete End-to-End Flow (Optimized)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              PRISM4D AMBER SIMD BATCH PIPELINE                            │
│                                    (SOTA OPTIMIZED)                                       │
└──────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════
                                    INITIALIZATION PHASE
═══════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PDB Files     │────▶│  Topology       │────▶│  Batch Upload   │────▶│  GPU Buffers    │
│   (1-128)       │     │  Generation     │     │  (finalize)     │     │  Allocated      │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        │                               │                               │
                        ▼                               ▼                               ▼
               ┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
               │  FP16 Param     │             │  Initial Cell   │             │  Verlet List    │
               │  Conversion     │             │  List Build     │             │  Build          │
               │  (one-time)     │             │  (one-time)     │             │  (one-time)     │
               └─────────────────┘             └─────────────────┘             └─────────────────┘
                        │                               │                               │
                        │  d_sigma_fp16                 │  d_cell_list                  │  d_neighbor_indices
                        │  d_epsilon_fp16              │  d_cell_counts                │  d_neighbor_offsets
                        │  d_positions_fp16            │  d_atom_cell                  │  d_ref_positions
                        │                               │                               │
                        └───────────────────────────────┴───────────────────────────────┘
                                                        │
                                                        ▼
               ┌────────────────────────────────────────────────────────────────────────────┐
               │                         MINIMIZATION (Optional)                            │
               │   run_internal(n_steps=1000, dt=0.2, T=0, γ=0.5) + velocity reset         │
               └────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
               ┌────────────────────────────────────────────────────────────────────────────┐
               │                         EQUILIBRATION PHASE                                │
               │   equilibrate_staged() or equilibrate_with_gamma()                        │
               │   - Temperature ramping: 50K → 100K → ... → 300K                          │
               │   - Strong thermostat (γ=0.1) then relaxing (γ=0.01)                      │
               └────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼

═══════════════════════════════════════════════════════════════════════════════════════════
                                    PRODUCTION MD PHASE
═══════════════════════════════════════════════════════════════════════════════════════════

                    ┌───────────────────────────────────────────────────────────────────┐
                    │                         run(n_steps, dt, T, γ)                    │
                    └───────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                    FOR EACH MD STEP                                       │
│══════════════════════════════════════════════════════════════════════════════════════════│
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              STEP 1: VERLET CHECK                                  │  │
│  │                                                                                    │  │
│  │   ┌─────────────────────┐                                                         │  │
│  │   │ verlet.needs_rebuild│                                                         │  │
│  │   │ (d_positions)       │                                                         │  │
│  │   │                     │                                                         │  │
│  │   │ GPU Kernel:         │                                                         │  │
│  │   │ - For each atom:    │     ┌──────────────────────────────────────────────┐   │  │
│  │   │   disp² = |r - r₀|² │────▶│ max_disp² > (skin/2)² ?                      │   │  │
│  │   │ - atomicMax(max_d²) │     │                                              │   │  │
│  │   └─────────────────────┘     │  NO: Skip rebuild (90% of steps)             │   │  │
│  │         │                     │      Use cached neighbor list                │   │  │
│  │         │                     │                                              │   │  │
│  │         ▼                     │  YES: Rebuild Verlet list (~10% of steps)    │   │  │
│  │   Cost: ~0.1ms                │       verlet.build(d_positions, cell_list)   │   │  │
│  │                               │       Cost: ~0.5ms                           │   │  │
│  │                               └──────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                              │
│                                           ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                         STEP 2: FORCE COMPUTATION (PARALLEL STREAMS)               │  │
│  │                                                                                    │  │
│  │   ════════════════════════════════════════════════════════════════════════════    │  │
│  │   STREAM A (Bonded)                    STREAM B (Non-bonded)                      │  │
│  │   ════════════════════════════════════════════════════════════════════════════    │  │
│  │                                                                                    │  │
│  │   ┌─────────────────────┐              ┌─────────────────────────────────────┐    │  │
│  │   │ Zero Forces         │              │ Convert Positions to FP16           │    │  │
│  │   │ d_forces[:] = 0     │              │ verlet.convert_positions_to_fp16()  │    │  │
│  │   └──────────┬──────────┘              └──────────────────┬──────────────────┘    │  │
│  │              │                                            │                       │  │
│  │              ▼                                            ▼                       │  │
│  │   ┌─────────────────────┐              ┌─────────────────────────────────────┐    │  │
│  │   │ Bond Forces         │              │ Tensor Core Non-bonded              │    │  │
│  │   │                     │              │                                     │    │  │
│  │   │ F_bond = -k(r-r₀)r̂  │              │ IF has_tensor_cores:                │    │  │
│  │   │ Per bond: O(1)      │              │   tensor_core.compute_nonbonded()   │    │  │
│  │   │ Total: O(N_bonds)   │              │   - WMMA 16×16 distance tiles       │    │  │
│  │   └──────────┬──────────┘              │   - D²[i,j] = ||r_i||² + ||r_j||²   │    │  │
│  │              │                         │              - 2(r_i · r_j)         │    │  │
│  │              ▼                         │   - FP16 params, FP32 accumulate    │    │  │
│  │   ┌─────────────────────┐              │                                     │    │  │
│  │   │ Angle Forces        │              │ ELSE:                               │    │  │
│  │   │                     │              │   verlet.compute_nonbonded()        │    │  │
│  │   │ F_θ = -k_θ(θ-θ₀)∇θ  │              │   - Newton's 3rd law optimization   │    │  │
│  │   │ Per angle: O(1)     │              │   - Cached neighbor list lookup     │    │  │
│  │   │ Total: O(N_angles)  │              │                                     │    │  │
│  │   └──────────┬──────────┘              │ LJ: F = 24ε[(2σ¹²/r¹³) - (σ⁶/r⁷)]  │    │  │
│  │              │                         │ Coulomb: F = q_i q_j / (4πε₀r²)    │    │  │
│  │              ▼                         │                                     │    │  │
│  │   ┌─────────────────────┐              │ Per pair: O(1)                      │    │  │
│  │   │ Dihedral Forces     │              │ Total: O(N × avg_neighbors)         │    │  │
│  │   │                     │              │       ≈ O(N) with neighbor list     │    │  │
│  │   │ F_φ = -k_φ cos(nφ-δ)│              └──────────────────┬──────────────────┘    │  │
│  │   │ Per dihedral: O(1)  │                                 │                       │  │
│  │   │ Total: O(N_dihed)   │                                 │                       │  │
│  │   └──────────┬──────────┘                                 │                       │  │
│  │              │                                            │                       │  │
│  │              │         ┌──────────────────────────────────┘                       │  │
│  │              │         │                                                          │  │
│  │              ▼         ▼                                                          │  │
│  │   ════════════════════════════════════════════════════════════════════════════    │  │
│  │              SYNCHRONIZATION POINT (cudaStreamSynchronize)                        │  │
│  │   ════════════════════════════════════════════════════════════════════════════    │  │
│  │                                                                                    │  │
│  │   Cost: Bonded ~0.3ms  |  Non-bonded (TC) ~0.8ms  |  Non-bonded (std) ~2ms        │  │
│  │   With overlap: max(0.3, 0.8) = 0.8ms effective                                   │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                              │
│                                           ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                           STEP 3: VELOCITY VERLET INTEGRATION                      │  │
│  │                                                                                    │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────────┐  │  │
│  │   │                         HALF-KICK 1                                         │  │  │
│  │   │                                                                             │  │  │
│  │   │   v(t + dt/2) = v(t) + (dt/2) × F(t) / m                                   │  │  │
│  │   │                                                                             │  │  │
│  │   │   For each atom i:                                                          │  │  │
│  │   │     v_i += 0.5 * dt * F_i / m_i                                            │  │  │
│  │   └─────────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                           │                                        │  │
│  │                                           ▼                                        │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────────┐  │  │
│  │   │                              DRIFT                                          │  │  │
│  │   │                                                                             │  │  │
│  │   │   r(t + dt) = r(t) + dt × v(t + dt/2)                                      │  │  │
│  │   │                                                                             │  │  │
│  │   │   For each atom i:                                                          │  │  │
│  │   │     r_i += dt * v_i                                                        │  │  │
│  │   └─────────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                           │                                        │  │
│  │   Cost: ~0.05ms                           │                                        │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                              │
│                                           ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                      STEP 4: FORCES AT NEW POSITIONS (if proper VV)               │  │
│  │                                                                                    │  │
│  │   NOTE: For proper Velocity Verlet, we need F(t+dt) for the second half-kick.     │  │
│  │   This requires recomputing forces at the new positions.                          │  │
│  │                                                                                    │  │
│  │   WITH OPTIMIZATION: Use Verlet list (still valid - atoms moved < skin/2)         │  │
│  │   - Skip cell list rebuild (already checked in Step 1)                            │  │
│  │   - Recompute forces using same neighbor list                                     │  │
│  │                                                                                    │  │
│  │   Cost: ~0.8ms (Tensor Core) or ~2ms (standard)                                   │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                              │
│                                           ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                           STEP 5: COMPLETE INTEGRATION                             │  │
│  │                                                                                    │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────────┐  │  │
│  │   │                         HALF-KICK 2                                         │  │  │
│  │   │                                                                             │  │  │
│  │   │   v(t + dt) = v(t + dt/2) + (dt/2) × F(t + dt) / m                         │  │  │
│  │   │                                                                             │  │  │
│  │   │   For each atom i:                                                          │  │  │
│  │   │     v_i += 0.5 * dt * F_i / m_i                                            │  │  │
│  │   └─────────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                           │                                        │  │
│  │                                           ▼                                        │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────────┐  │  │
│  │   │                      LANGEVIN THERMOSTAT                                    │  │  │
│  │   │                                                                             │  │  │
│  │   │   v_i = v_i × (1 - γ×dt) + √(2×γ×kT×dt/m_i) × η_i                         │  │  │
│  │   │                                                                             │  │  │
│  │   │   Where:                                                                    │  │  │
│  │   │     γ = friction coefficient (0.01 - 0.1 fs⁻¹)                             │  │  │
│  │   │     η_i = Gaussian random noise                                            │  │  │
│  │   │     T = target temperature                                                 │  │  │
│  │   └─────────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                           │                                        │  │
│  │   Cost: ~0.05ms                           │                                        │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                              │
│                                           ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                           STEP 6: H-BOND CONSTRAINTS (SHAKE/RATTLE)               │  │
│  │                                                                                    │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────────┐  │  │
│  │   │                      HConstraints::apply()                                  │  │  │
│  │   │                                                                             │  │  │
│  │   │   For each X-H constraint cluster:                                         │  │  │
│  │   │     1. Single H: Simple distance correction                                 │  │  │
│  │   │     2. Two H (NH2, CH2): Triangle constraint solver                        │  │  │
│  │   │     3. Three H (CH3, NH3): Tetrahedral constraint solver                   │  │  │
│  │   │                                                                             │  │  │
│  │   │   Allows dt = 2fs (vs 1fs without constraints)                             │  │  │
│  │   │   → 2× fewer steps for same simulation time                                │  │  │
│  │   └─────────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                           │                                        │  │
│  │   Cost: ~0.1ms                            │                                        │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                              │
│  ════════════════════════════════════════════════════════════════════════════════════   │
│  TOTAL STEP TIME (Optimized):                                                           │
│    Verlet check:     0.1ms (amortized rebuild: +0.05ms)                                │
│    Force compute:    0.8ms (Tensor Core, overlapped)                                   │
│    Integration:      0.1ms                                                              │
│    Force recompute:  0.8ms (for proper VV)                                             │
│    Constraints:      0.1ms                                                              │
│    ─────────────────────────                                                            │
│    TOTAL:            ~1.9ms per step                                                    │
│                                                                                          │
│  vs CURRENT:         ~7.6ms per step                                                    │
│  SPEEDUP:            ~4× per step                                                       │
│  ════════════════════════════════════════════════════════════════════════════════════   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌───────────────────────────────────────────────────────────────────┐
                    │                         get_all_results()                         │
                    │   - Download positions, velocities, energies                      │
                    │   - Remove spatial offsets                                        │
                    │   - Compute temperature from KE                                   │
                    └───────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌───────────────────────────────────────────────────────────────────┐
                    │                      Vec<BatchMdResult>                           │
                    │   - Per-structure positions                                       │
                    │   - Per-structure velocities                                      │
                    │   - Per-structure energies (PE, KE)                              │
                    │   - Per-structure temperature                                     │
                    └───────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

### GPU Buffer Layout (Optimized)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              GPU MEMORY LAYOUT                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  POSITIONS (FP32)              POSITIONS (FP16)           REFERENCE POSITIONS       │
│  ┌───────────────────┐         ┌───────────────────┐      ┌───────────────────┐    │
│  │ x₀ y₀ z₀ x₁ y₁...│         │ x₀ y₀ z₀ x₁ y₁...│      │ x₀ y₀ z₀ x₁ y₁...│    │
│  │ [total_atoms × 3] │         │ [total_atoms × 3] │      │ [total_atoms × 3] │    │
│  └───────────────────┘         └───────────────────┘      └───────────────────┘    │
│                                        │                                            │
│                                        │ 50% bandwidth                              │
│                                        ▼ savings                                    │
│                                                                                     │
│  LJ PARAMETERS (FP32)          LJ PARAMETERS (FP16)                                │
│  ┌───────────────────┐         ┌───────────────────┐                               │
│  │ σ₀ ε₀ σ₁ ε₁ ...  │         │ σ₀ ε₀ σ₁ ε₁ ...  │                               │
│  │ [total_atoms × 2] │         │ [total_atoms × 2] │                               │
│  └───────────────────┘         └───────────────────┘                               │
│                                                                                     │
│  VERLET NEIGHBOR LIST                                                              │
│  ┌───────────────────────────────────────────────────────────────────────────┐     │
│  │ neighbor_counts[N]     │ neighbor_offsets[N]    │ neighbor_indices[N×512] │     │
│  │ [count per atom]       │ [CSR-style offsets]    │ [actual neighbor IDs]   │     │
│  └───────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
│  BONDED TOPOLOGY                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                    │
│  │ bond_atoms[2×B] │  │ angle_atoms[4×A]│  │ dihedral_atoms  │                    │
│  │ bond_params[2×B]│  │ angle_params[2×A]│ │ dihedral_params │                    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                    │
│                                                                                     │
│  EXCLUSION LIST                           ENERGIES                                 │
│  ┌───────────────────────┐               ┌───────────────────┐                    │
│  │ excl_list[N×32]       │               │ PE₀ KE₀ PE₁ KE₁...│                    │
│  │ n_excl[N]             │               │ [n_structures × 2]│                    │
│  └───────────────────────┘               └───────────────────┘                    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Performance Expectations

### Per-Step Timing Comparison

| Component | Current | Optimized | Speedup |
|-----------|---------|-----------|---------|
| Neighbor list build | 1.0ms (every step) | 0.55ms (every 10 steps avg) | 18× |
| Non-bonded forces | 5.0ms (×2 phases) | 0.8ms (TC) or 2.0ms (std) | 2.5-6× |
| Bonded forces | 0.3ms | 0.3ms (overlapped) | - |
| Integration | 0.2ms | 0.2ms | - |
| H-constraints | 0.1ms | 0.1ms | - |
| **Total per step** | **7.6ms** | **1.5-2.5ms** | **3-5×** |

### Throughput Comparison (1000-atom protein)

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| Steps/second | ~130 | ~500-700 | 4-5× |
| ns/day | ~0.26 | ~1.0-1.4 | 4-5× |
| GPU utilization | ~40% | ~70-85% | ~2× |

### Batch Throughput (32 structures × 1000 atoms)

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| Steps/second | ~4 | ~15-20 | 4-5× |
| Structures/minute | ~240 | ~900-1200 | 4-5× |
| Total ns/day | ~8 | ~35-45 | 4-5× |

## Implementation Order

1. **Week 1**: Verlet list integration (highest ROI)
   - Add VerletList to AmberSimdBatch
   - Replace cell list rebuild with skin check
   - Validate correctness with existing tests

2. **Week 2**: FP16 conversion
   - Add FP16 buffers
   - Convert parameters in finalize_batch
   - Update CUDA kernel to use FP16

3. **Week 3**: Tensor Core integration
   - Add TensorCoreForces (optional path)
   - Detect GPU capability
   - Route to TC kernel when available

4. **Week 4**: Async pipeline
   - Add multi-stream execution
   - Overlap bonded/non-bonded
   - Pipeline Verlet check

5. **Week 5**: Testing and benchmarks
   - Correctness validation
   - Performance benchmarks
   - Documentation

## Risk Mitigation

1. **Numerical precision**: FP16 can lose precision at large distances. Use FP32 accumulation.
2. **Verlet skin too small**: Monitor rebuild frequency. Increase skin if >20% steps rebuild.
3. **Tensor Core compatibility**: Graceful fallback to standard kernel on older GPUs.
4. **Race conditions**: Async pipeline needs careful synchronization points.

## Success Criteria

- [ ] All existing tests pass with optimizations enabled
- [ ] Energy conservation within 0.1% of current implementation
- [ ] Temperature stability within ±10K of target
- [ ] 4× or greater speedup demonstrated
- [ ] No increase in VRAM usage (FP16 should decrease it)
