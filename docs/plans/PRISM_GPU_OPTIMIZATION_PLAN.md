# PRISM4D GPU Optimization Implementation Plan

## Overview

This plan implements a comprehensive GPU optimization stack for the PRISM4D AMBER MD engine, targeting:
- **5-6x throughput improvement** over current implementation
- **100% reproducibility** (deterministic results across runs)
- **100x better energy accuracy** (FP64 accumulation with Kahan summation)

### Hardware Target
- **GPU**: RTX 3060 Laptop (Compute Capability 8.6, Ampere)
- **Tensor Cores**: 3rd generation (FP16→FP32, TF32, BF16)
- **VRAM**: 6GB
- **CUDA Cores**: 3840

### Validation Strategy
All optimizations validated using **curated 14 structures** with the latest preprocessing pipeline:
- Smart multi-chain routing (STANDARD/MULTICHAIN/WHOLE)
- Hybrid hydrogen placement (PDBFixer)
- Full AMBER ff14SB topology generation
- Energy minimization + staged equilibration

---

## Precision Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRISM4D PRECISION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  COORDINATES & VELOCITIES                                                   │
│  ─────────────────────────                                                  │
│  Storage: FP32 (sufficient for ~1e-7 Å precision)                          │
│  Integration: FP32 (Verlet/Langevin updates)                               │
│                                                                             │
│  FORCE COMPUTATION                                                          │
│  ─────────────────                                                          │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   Distances     │────▶│  Force Terms    │────▶│  Accumulation   │       │
│  │   FP32          │     │  FP16/FP32      │     │  Tree Reduce    │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                         │                   │
│                                                         ▼                   │
│                                              ┌─────────────────────┐        │
│                                              │  Per-Atom Forces    │        │
│                                              │  FP32 (deterministic)│        │
│                                              └─────────────────────┘        │
│                                                                             │
│  ENERGY COMPUTATION                                                         │
│  ──────────────────                                                         │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  Energy Terms   │────▶│  Tree Reduce    │────▶│  Kahan Sum      │       │
│  │  FP32           │     │  FP64 partials  │     │  FP64 total     │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                             │
│  TENSOR CORE OPERATIONS (Optional acceleration)                             │
│  ────────────────────────────────────────────                               │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  Pair Distances │────▶│  WMMA FP16×FP16 │────▶│  Forces FP32    │       │
│  │  FP16 tiles     │     │  8×8×8 fragments│     │  output         │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                             │
│  PME ELECTROSTATICS (Optional)                                              │
│  ─────────────────────────────                                              │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  Charge Grid    │────▶│  cuFFT TF32     │────▶│  Interpolation  │       │
│  │  FP32           │     │  (automatic)    │     │  FP32           │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Deterministic Reductions (Priority: CRITICAL)
**Timeline**: 3-4 days
**Speedup**: 1.5-3x (reduced contention)
**Enables**: All subsequent phases (reproducibility foundation)

#### Current Problem
```cuda
// Current: Non-deterministic due to atomicAdd ordering
atomicAdd(&forces[atom_id].x, fx);  // Order depends on thread scheduling
atomicAdd(&forces[atom_id].y, fy);
atomicAdd(&forces[atom_id].z, fz);
```

#### Solution: Tree Reduction with Warp Primitives
```cuda
// Phase 1.1: Intra-warp reduction (warp-level deterministic)
__device__ float3 warp_reduce_sum(float3 val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, offset);
        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, offset);
        val.z += __shfl_down_sync(0xFFFFFFFF, val.z, offset);
    }
    return val;
}

// Phase 1.2: Inter-warp reduction via shared memory
__shared__ float3 warp_sums[32];  // One per warp
int lane = threadIdx.x % 32;
int warp_id = threadIdx.x / 32;

float3 reduced = warp_reduce_sum(local_force);
if (lane == 0) {
    warp_sums[warp_id] = reduced;
}
__syncthreads();

// Final reduction in first warp
if (warp_id == 0 && lane < (blockDim.x / 32)) {
    float3 final = warp_reduce_sum(warp_sums[lane]);
    if (lane == 0) {
        // Single atomic write per block (deterministic order)
        atomicAdd(&forces[atom_id].x, final.x);
    }
}
```

#### Files to Modify
| File | Changes |
|------|---------|
| `crates/prism-gpu/src/kernels/amber_simd_batch.cu` | Replace ~50 atomicAdd with tree reductions |
| `crates/prism-gpu/src/kernels/amber_bonded.cu` | Tree reductions for bonded forces |
| `crates/prism-gpu/src/kernels/reduction_primitives.cuh` | NEW: Shared reduction primitives header |

#### Validation Checkpoint
```bash
# Run same seed 3 times - must produce IDENTICAL results
for i in 1 2 3; do
  cargo run --release -p prism-validation --bin generate-ensemble-simd -- \
    --topology data/prepared/1L2Y_topology.json \
    --n-samples 10 --n-steps 100 --seed 42 \
    --output results/determinism_test_$i/
done

# Compare outputs
diff results/determinism_test_1/ensemble.pdb results/determinism_test_2/ensemble.pdb
diff results/determinism_test_2/ensemble.pdb results/determinism_test_3/ensemble.pdb
# Expected: No differences
```

---

### Phase 2: FP64 Energy Accumulation (Priority: HIGH)
**Timeline**: 1 day
**Accuracy**: 100x improvement in energy precision
**Enables**: Proper energy conservation tracking, better thermodynamics

#### Current Problem
```cuda
// Current: FP32 accumulation loses precision with large systems
float total_energy = 0.0f;
for (int i = 0; i < n_atoms; i++) {
    total_energy += energy[i];  // Catastrophic cancellation
}
```

#### Solution: Kahan Summation with FP64
```cuda
// Kahan summation for numerically stable accumulation
struct KahanAccumulator {
    double sum;
    double compensation;
};

__device__ void kahan_add(KahanAccumulator* acc, double value) {
    double y = value - acc->compensation;
    double t = acc->sum + y;
    acc->compensation = (t - acc->sum) - y;
    acc->sum = t;
}

// Tree reduction with FP64 partials
__device__ double block_reduce_kahan(float local_energy) {
    __shared__ double partial_sums[32];

    // Warp-level FP64 reduction
    double d_energy = (double)local_energy;
    for (int offset = 16; offset > 0; offset /= 2) {
        d_energy += __shfl_down_sync(0xFFFFFFFF, d_energy, offset);
    }

    // Store warp result
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        partial_sums[warp_id] = d_energy;
    }
    __syncthreads();

    // Final reduction in warp 0
    if (warp_id == 0 && lane < (blockDim.x / 32)) {
        d_energy = partial_sums[lane];
        for (int offset = 16; offset > 0; offset /= 2) {
            d_energy += __shfl_down_sync(0xFFFFFFFF, d_energy, offset);
        }
    }

    return d_energy;
}
```

#### Files to Modify
| File | Changes |
|------|---------|
| `crates/prism-gpu/src/kernels/amber_simd_batch.cu` | FP64 energy reduction |
| `crates/prism-gpu/src/kernels/reduction_primitives.cuh` | Add Kahan accumulator |
| `crates/prism-gpu/src/amber_simd_batch.rs` | Change energy return type to f64 |

#### Validation Checkpoint
```bash
# Energy conservation test - run 10000 steps NVE
cargo run --release -p prism-validation --bin energy-conservation-test -- \
  --topology data/prepared/1L2Y_topology.json \
  --n-steps 10000 --dt 1.0 --temperature 0.0

# Expected: ΔE/E < 1e-6 (was ~1e-3 with FP32)
```

---

### Phase 3: Persistent Neighbor Lists (Priority: HIGH)
**Timeline**: 4-5 days
**Speedup**: 2-3x (avoid rebuilding every step)
**Memory**: Slight increase for padding (~10%)

#### Current Problem
```cuda
// Current: Rebuild cell list every step - expensive!
__global__ void simd_batch_populate_cells(...) {
    // Called every integration step
    // O(N) rebuild even when atoms barely moved
}
```

#### Solution: Verlet Lists with Skin Distance
```cuda
// Build neighbor list with skin buffer
const float SKIN_DISTANCE = 2.0f;  // Ångstroms
const float CUTOFF = 9.0f;
const float LIST_CUTOFF = CUTOFF + SKIN_DISTANCE;  // 11.0 Å

struct VerletList {
    int* neighbors;      // [n_atoms * max_neighbors]
    int* n_neighbors;    // [n_atoms]
    float3* ref_coords;  // Positions at last rebuild
    int rebuild_counter; // Steps since last rebuild
};

// Check if rebuild needed (every step, very cheap)
__global__ void check_displacement(
    float3* coords,
    float3* ref_coords,
    int n_atoms,
    float skin_half,
    int* needs_rebuild  // Output flag
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    float3 d = coords[i] - ref_coords[i];
    float dist_sq = d.x*d.x + d.y*d.y + d.z*d.z;

    // If any atom moved > skin/2, flag for rebuild
    if (dist_sq > skin_half * skin_half) {
        atomicOr(needs_rebuild, 1);
    }
}

// Rebuild only when needed (typically every 10-20 steps)
if (*needs_rebuild) {
    build_verlet_list<<<...>>>(coords, list, LIST_CUTOFF);
    cudaMemcpy(list.ref_coords, coords, ...);
    *needs_rebuild = 0;
}
```

#### Memory Layout for Coalesced Access
```cuda
// Structure of Arrays (SoA) for neighbor data
struct VerletListSoA {
    int* neighbor_counts;   // [n_atoms] - count per atom
    int* neighbor_offsets;  // [n_atoms] - CSR-style offsets
    int* neighbor_indices;  // [total_pairs] - packed neighbor list
    float3* ref_positions;  // [n_atoms] - reference for displacement check
};
```

#### Files to Create/Modify
| File | Changes |
|------|---------|
| `crates/prism-gpu/src/kernels/verlet_list.cu` | NEW: Verlet list build/update |
| `crates/prism-gpu/src/kernels/amber_simd_batch.cu` | Use Verlet list in non-bonded |
| `crates/prism-gpu/src/amber_simd_batch.rs` | Add VerletList management |

#### Validation Checkpoint
```bash
# Verify physics unchanged with Verlet lists
cargo run --release -p prism-validation --bin verlet-validation -- \
  --topology data/prepared/1L2Y_topology.json \
  --n-steps 1000 --compare-cell-list

# Expected: Force differences < 1e-5, energy drift < 1e-6
# Measure: Average rebuilds per 100 steps (should be 5-10)
```

---

### Phase 4: Tensor Core Force Computation (Priority: MEDIUM)
**Timeline**: 1-2 weeks
**Speedup**: 2-4x for non-bonded forces
**Complexity**: HIGH (requires matrix restructuring)

#### Concept: Distance Matrix Tiling
```
Traditional:
  For each pair (i,j): compute distance, compute force

Tensor Core:
  1. Tile atoms into 8×8 blocks
  2. Compute 64 distances in parallel via WMMA
  3. Compute 64 forces via vectorized operations
  4. Accumulate to per-atom totals
```

#### WMMA Implementation
```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// Fragment types for 8x8x8 FP16 multiply
fragment<matrix_a, 8, 8, 8, half, row_major> a_frag;
fragment<matrix_b, 8, 8, 8, half, col_major> b_frag;
fragment<accumulator, 8, 8, 8, float> c_frag;

__global__ void tensor_core_distances(
    half* coords_x,  // [n_atoms] FP16 x coordinates
    half* coords_y,  // [n_atoms] FP16 y coordinates
    half* coords_z,  // [n_atoms] FP16 z coordinates
    float* dist_sq,  // [n_tiles * 64] output distance² matrix
    int n_atoms
) {
    // Each warp handles one 8×8 tile
    int tile_i = blockIdx.x;
    int tile_j = blockIdx.y;

    // Load coordinate tiles
    // Compute (x_i - x_j)² + (y_i - y_j)² + (z_i - z_j)²
    // Using WMMA for parallel computation

    fill_fragment(c_frag, 0.0f);

    // X component: outer product trick
    // dist_x² = x_i² - 2*x_i*x_j + x_j²
    load_matrix_sync(a_frag, &coords_x[tile_i * 8], 8);
    load_matrix_sync(b_frag, &coords_x[tile_j * 8], 8);
    mma_sync(c_frag, a_frag, b_frag, c_frag);  // Accumulates x_i * x_j

    // ... similar for Y, Z

    store_matrix_sync(dist_sq, c_frag, 8, mem_row_major);
}
```

#### Files to Create/Modify
| File | Changes |
|------|---------|
| `crates/prism-gpu/src/kernels/tensor_core_forces.cu` | NEW: WMMA force kernels |
| `crates/prism-gpu/src/kernels/tensor_core_distances.cuh` | NEW: Distance matrix tiles |
| `crates/prism-gpu/src/amber_simd_batch.rs` | Integrate TC path |
| `crates/prism-gpu/build.rs` | Add WMMA compilation flags |

#### Validation Checkpoint
```bash
# Compare Tensor Core vs scalar forces
cargo run --release -p prism-validation --bin tensor-core-validation -- \
  --topology data/prepared/1L2Y_topology.json \
  --n-steps 100 --compare-scalar

# Expected: Max force error < 1e-3 (FP16 precision)
# Measure: Throughput improvement (target 2-4x)
```

---

### Phase 5: TF32 PME Electrostatics (Priority: LOW)
**Timeline**: 2-3 days
**Speedup**: 1.5-2x for electrostatics (if PME enabled)
**Prerequisite**: PME must be implemented first

#### Note: PME Not Currently Active
The current AMBER implementation uses direct Coulomb summation with cutoff.
PME would be needed for periodic boundary conditions (PBC) at scale.
This phase is **deferred** until PBC/PME is added to the codebase.

#### When Implemented:
```cuda
// cuFFT automatically uses TF32 on Ampere
cufftHandle plan;
cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C);

// Set TF32 math mode
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

// FFT operations run with TF32 acceleration
cufftExecC2C(plan, grid_data, grid_data, CUFFT_FORWARD);
```

---

## Implementation Schedule

```
Week 1:
├── Day 1-2: Phase 1a - Implement reduction_primitives.cuh
├── Day 3-4: Phase 1b - Replace atomicAdd in amber_simd_batch.cu
└── Day 5: Phase 1c - Replace atomicAdd in amber_bonded.cu + validation

Week 2:
├── Day 1: Phase 2 - FP64 energy accumulation (Kahan)
├── Day 2-3: Phase 3a - Verlet list data structures
├── Day 4-5: Phase 3b - Verlet list build/update kernels

Week 3:
├── Day 1-2: Phase 3c - Integrate Verlet lists into force kernels
├── Day 3: Phase 3d - Displacement check + auto-rebuild logic
└── Day 4-5: Integration testing + benchmarking

Week 4+ (Optional):
├── Phase 4: Tensor Core forces (if speed critical)
└── Phase 5: TF32 PME (when PBC added)
```

---

## Curated 14 Validation Suite

### Test Matrix
Each phase validated against all 14 structures with current best preprocessing:

| PDB | Atoms | Routing | Current PE/atom | Test Focus |
|-----|-------|---------|-----------------|------------|
| 1L2Y | 304 | STANDARD | 47.9 | Baseline speed |
| 1AKE | 3341 | STANDARD | TBD | Medium size |
| 4AKE | 3459 | STANDARD | TBD | Conformational |
| 5IRE | 8364 | MULTICHAIN | 266 | Multi-chain |
| 1HXY | 1641 | WHOLE | TBD | Disulfide-rich |
| 6LU7 | 4894 | STANDARD | 81.6 | Production |
| ... | ... | ... | ... | ... |

### Validation Commands
```bash
# Full validation script
./scripts/validate_optimization_phase.sh --phase 1

# Individual structure test
cargo run --release -p prism-validation --bin validate-curated14 -- \
  --phase deterministic-reductions \
  --structures data/curated_14/

# Performance benchmark
cargo run --release -p prism-validation --bin benchmark-curated14 -- \
  --baseline results/baseline_perf.json \
  --output results/phase1_perf.json
```

### Acceptance Criteria Per Phase

| Phase | Correctness | Performance | Reproducibility |
|-------|-------------|-------------|-----------------|
| 1. Deterministic | Forces match ±1e-5 | ≥1.5x faster | 100% identical |
| 2. FP64 Energy | Energy drift <1e-6 | Same speed | Energy ±1e-12 |
| 3. Verlet Lists | Forces match ±1e-5 | ≥2x faster | Depends on P1 |
| 4. Tensor Cores | Forces match ±1e-3 | ≥3x faster | Depends on P1 |
| 5. TF32 PME | Energy match ±1e-4 | ≥1.5x faster | TF32 precision |

---

## Risk Mitigation

### Rollback Strategy
Each phase is independent. If issues found:
1. Revert to previous phase's code
2. Re-run curated 14 validation
3. Debug isolated kernel changes

### Feature Flags
```rust
// In crates/prism-gpu/src/lib.rs
#[cfg(feature = "deterministic-reductions")]
mod reduction_primitives;

#[cfg(feature = "fp64-energy")]
mod kahan_accumulator;

#[cfg(feature = "verlet-lists")]
mod verlet_list;

#[cfg(feature = "tensor-cores")]
mod tensor_core_forces;
```

### Compatibility Testing
```bash
# Test with all features disabled (baseline)
cargo test --release -p prism-gpu --no-default-features

# Test with each feature individually
cargo test --release -p prism-gpu --features deterministic-reductions
cargo test --release -p prism-gpu --features fp64-energy
cargo test --release -p prism-gpu --features verlet-lists
cargo test --release -p prism-gpu --features tensor-cores

# Test all features together
cargo test --release -p prism-gpu --all-features
```

---

## Expected Outcomes

### Performance Gains (Cumulative)

| After Phase | Throughput | Energy Accuracy | Reproducibility |
|-------------|------------|-----------------|-----------------|
| Baseline | 1.0x | ~1e-3 | Non-deterministic |
| Phase 1 | 1.5-3.0x | ~1e-3 | 100% deterministic |
| Phase 2 | 1.5-3.0x | ~1e-12 | 100% deterministic |
| Phase 3 | 3.0-6.0x | ~1e-12 | 100% deterministic |
| Phase 4 | 6.0-12.0x | ~1e-3* | 100% deterministic |
| Phase 5 | 7.0-15.0x | ~1e-4* | TF32 precision |

*Forces computed at reduced precision, energy at full precision

### Memory Impact

| Phase | Additional VRAM | Notes |
|-------|-----------------|-------|
| Phase 1 | ~0 | Same or less (no atomic contention) |
| Phase 2 | +8 bytes/atom | FP64 energy partials |
| Phase 3 | +20-50% | Verlet list storage + skin buffer |
| Phase 4 | +16 bytes/atom | FP16 coordinate copies |
| Phase 5 | Per-grid | FFT workspace |

---

## Commit Strategy

Each phase completed with atomic commits:

```
feat(gpu): Add deterministic tree reduction primitives
feat(gpu): Replace atomicAdd with tree reductions in SIMD batch kernel
feat(gpu): Add FP64 energy accumulation with Kahan summation
feat(gpu): Implement Verlet neighbor lists with skin buffer
feat(gpu): Add displacement check for auto-rebuild
feat(gpu): Integrate Verlet lists into non-bonded force computation
perf(gpu): Add Tensor Core distance matrix computation
perf(gpu): Integrate Tensor Core forces into main kernel
```

---

## Next Steps

1. **Read current atomicAdd usage** in `amber_simd_batch.cu`
2. **Create `reduction_primitives.cuh`** with warp/block reduce helpers
3. **Modify force accumulation** to use tree reductions
4. **Validate with curated 14** - ensure determinism + correctness
5. **Benchmark throughput** - confirm 1.5-3x improvement
6. **Proceed to Phase 2** after Phase 1 checkpoint passes
