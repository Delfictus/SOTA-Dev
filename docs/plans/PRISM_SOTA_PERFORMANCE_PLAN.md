# PRISM4D SOTA+ Performance Optimization Plan

## Overview

This plan implements four major GPU optimizations to achieve **10-20× speedup** over current implementation, targeting **400-800 ns/day** for implicit solvent MD on RTX 3060.

### Target Performance

| Stage | Optimizations | Expected ns/day | Cumulative Speedup |
|-------|--------------|-----------------|-------------------|
| Current | Baseline | 30 | 1× |
| Phase 1 | Verlet Lists | 60-90 | 2-3× |
| Phase 2 | + Tensor Cores | 150-300 | 5-10× |
| Phase 3 | + FP16 Params | 200-400 | 7-13× |
| Phase 4 | + Async Pipeline | 250-500 | 8-17× |

### Hardware Target
- **GPU**: RTX 3060 Laptop (GA106, Ampere)
- **Compute Capability**: 8.6
- **Tensor Cores**: 80 (3rd gen, FP16→FP32)
- **CUDA Cores**: 3840
- **Memory Bandwidth**: 336 GB/s
- **VRAM**: 6GB

---

## Phase 1: Verlet Neighbor Lists with Skin Buffer

### Current Problem
```cuda
// EVERY MD step:
simd_batch_zero_cell_counts<<<...>>>();   // O(cells)
simd_batch_build_cell_list<<<...>>>();    // O(N) but expensive
// Then 27-cell lookup per atom
```

Cell list rebuild cost: ~15-20% of total step time.

### Solution: Verlet Lists with Adaptive Rebuild

```
┌─────────────────────────────────────────────────────────────────┐
│                    VERLET LIST ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Build Phase (every 10-20 steps):                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Cell List    │───▶│ Find Pairs   │───▶│ Store List   │      │
│  │ (temporary)  │    │ r < r_cut+skin│    │ Per Atom     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
│  Use Phase (every step):                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Check Max    │───▶│ If safe:     │───▶│ Direct Pair  │      │
│  │ Displacement │    │ Use cached   │    │ Iteration    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                                        │
│         │ If max_disp > skin/2                                  │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │ Trigger      │                                               │
│  │ Rebuild      │                                               │
│  └──────────────┘                                               │
│                                                                  │
│  Parameters:                                                     │
│  - r_cutoff = 10.0 Å (LJ/Coulomb cutoff)                       │
│  - r_skin = 2.0 Å (buffer distance)                            │
│  - r_list = 12.0 Å (neighbor list cutoff)                      │
│  - Typical rebuild frequency: every 10-20 steps                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Structures

```cuda
// Per-atom neighbor storage (CSR-like format)
struct VerletList {
    int* neighbor_counts;     // [n_atoms] - number of neighbors per atom
    int* neighbor_offsets;    // [n_atoms] - CSR offset into packed list
    int* neighbor_indices;    // [total_pairs] - packed neighbor atom indices
    float3* ref_positions;    // [n_atoms] - positions at last rebuild
    int needs_rebuild;        // Global flag (set by any thread)
    int rebuild_count;        // Statistics
};

// Configuration
#define VERLET_SKIN 2.0f
#define VERLET_SKIN_HALF 1.0f
#define VERLET_LIST_CUTOFF (NB_CUTOFF + VERLET_SKIN)  // 12.0 Å
#define MAX_NEIGHBORS_PER_ATOM 256
```

### Kernels

```cuda
// 1. Check if rebuild needed (very cheap, every step)
__global__ void check_verlet_displacement(
    const float3* positions,
    const float3* ref_positions,
    int* needs_rebuild,
    int n_atoms
);

// 2. Build neighbor list (expensive, every 10-20 steps)
__global__ void build_verlet_list(
    const float3* positions,
    const int* cell_list,
    const int* cell_counts,
    int* neighbor_counts,
    int* neighbor_offsets,
    int* neighbor_indices,
    float3* ref_positions,
    int n_atoms
);

// 3. Compute forces using Verlet list (every step)
__global__ void compute_nonbonded_verlet(
    const float3* positions,
    float3* forces,
    float* energy,
    const int* neighbor_counts,
    const int* neighbor_offsets,
    const int* neighbor_indices,
    // ... LJ/Coulomb params
);
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `kernels/verlet_list.cu` | CREATE | Verlet list build/check kernels |
| `kernels/amber_simd_batch.cu` | MODIFY | Add `compute_nonbonded_verlet()` |
| `amber_simd_batch.rs` | MODIFY | Add VerletList management |

### Expected Speedup: 2-3×

---

## Phase 2: Tensor Core Force Computation

### Current Problem
```cuda
// Per pair: scalar FP32 operations
float dx = xj - xi;
float r2 = dx*dx + dy*dy + dz*dz;
float inv_r6 = inv_r2 * inv_r2 * inv_r2;
float lj = 4*eps*(sigma12*inv_r6*inv_r6 - sigma6*inv_r6);
```

RTX 3060 has 80 Tensor Cores doing nothing during MD!

### Solution: WMMA-Accelerated Distance and Force Tiles

```
┌─────────────────────────────────────────────────────────────────┐
│                 TENSOR CORE FORCE PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Load 8×8 Atom Tiles into Registers                     │
│  ┌────────────────────────────────────────────────────┐         │
│  │  tile_i[8]: atoms i₀..i₇ (from Verlet list)       │         │
│  │  tile_j[8]: neighbors j₀..j₇                       │         │
│  │  coords stored as FP16: (x, y, z)                  │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
│  Step 2: Compute 64 Distances via WMMA                          │
│  ┌────────────────────────────────────────────────────┐         │
│  │  // Distance² = (xi-xj)² + (yi-yj)² + (zi-zj)²    │         │
│  │  // Reformulate as: ||xi||² - 2*xi·xj + ||xj||²   │         │
│  │                                                    │         │
│  │  wmma::fragment<matrix_a, 8,8,8, half> Xi;        │         │
│  │  wmma::fragment<matrix_b, 8,8,8, half> Xj;        │         │
│  │  wmma::fragment<accumulator, 8,8,8, float> D;     │         │
│  │                                                    │         │
│  │  wmma::mma_sync(D, Xi, Xj, D);  // 64 dot products│         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
│  Step 3: Vectorized LJ/Coulomb (FP32 for precision)             │
│  ┌────────────────────────────────────────────────────┐         │
│  │  for k in 0..64:                                   │         │
│  │    r2 = D[k]                                       │         │
│  │    if r2 < cutoff²:                               │         │
│  │      compute LJ + Coulomb                          │         │
│  │      accumulate forces                             │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
│  Step 4: Warp-Reduce Forces Per Atom                            │
│  ┌────────────────────────────────────────────────────┐         │
│  │  forces_i = warp_reduce_sum(partial_forces)        │         │
│  │  // Single write per atom (deterministic)          │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### WMMA Distance Computation

The key insight: distance² = ||r_i - r_j||² can be computed as:
```
d² = ||r_i||² + ||r_j||² - 2 * r_i · r_j
```

Using WMMA for the dot product term:
```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// Each warp processes 8 atoms × 8 neighbors = 64 pairs
__device__ void tensor_core_distances_8x8(
    const half* __restrict__ coords_i,  // [8, 3] - 8 atoms, xyz
    const half* __restrict__ coords_j,  // [8, 3] - 8 neighbors, xyz
    float* __restrict__ dist_sq         // [64] output
) {
    fragment<matrix_a, 8, 8, 8, half, row_major> a_frag;
    fragment<matrix_b, 8, 8, 8, half, col_major> b_frag;
    fragment<accumulator, 8, 8, 8, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    // Compute -2 * (xi·xj + yi·yj + zi·zj) using WMMA
    // Need to restructure coords for matrix multiply

    // Load and compute
    load_matrix_sync(a_frag, coords_i, 8);
    load_matrix_sync(b_frag, coords_j, 8);
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(dist_sq, c_frag, 8, mem_row_major);
}
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `kernels/tensor_core_forces.cu` | CREATE | WMMA distance + force kernels |
| `kernels/tensor_core_forces.cuh` | CREATE | Shared structures and helpers |
| `amber_simd_batch.rs` | MODIFY | Integrate TC path |
| `build.rs` | MODIFY | Add WMMA compilation flags |

### Expected Speedup: 2-4× (for non-bonded)

---

## Phase 3: FP16 Parameters and Mixed Precision

### Current Problem
```cuda
// All FP32 - 4 bytes per value
const float* __restrict__ nb_sigma;    // [n_atoms]
const float* __restrict__ nb_epsilon;  // [n_atoms]
const float* __restrict__ nb_charge;   // [n_atoms]
```

Memory bandwidth is often the bottleneck for non-bonded forces.

### Solution: FP16 Storage with FP32 Compute

```
┌─────────────────────────────────────────────────────────────────┐
│                   MIXED PRECISION STRATEGY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FP16 Storage (half bandwidth):                                 │
│  ┌────────────────────────────────────────────────────┐         │
│  │  half* nb_sigma_fp16;    // 2 bytes (was 4)       │         │
│  │  half* nb_epsilon_fp16;  // 2 bytes (was 4)       │         │
│  │  half* nb_charge_fp16;   // 2 bytes (was 4)       │         │
│  │  half* coords_fp16;      // For Tensor Cores       │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
│  FP32 Compute (full precision where needed):                    │
│  ┌────────────────────────────────────────────────────┐         │
│  │  float sigma_i = __half2float(nb_sigma_fp16[i]);  │         │
│  │  float eps_i = __half2float(nb_epsilon_fp16[i]);  │         │
│  │  // All force computation in FP32                  │         │
│  │  // Accumulation in FP32                           │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
│  FP32 Storage (precision critical):                             │
│  ┌────────────────────────────────────────────────────┐         │
│  │  float* positions;       // Integration precision  │         │
│  │  float* velocities;      // Integration precision  │         │
│  │  float* forces;          // Accumulation precision │         │
│  │  float* masses;          // Division precision     │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Savings

| Array | Current | Optimized | Savings |
|-------|---------|-----------|---------|
| sigma | N × 4B | N × 2B | 50% |
| epsilon | N × 4B | N × 2B | 50% |
| charge | N × 4B | N × 2B | 50% |
| coords (TC copy) | N/A | N × 6B | New |
| **Total NB params** | N × 12B | N × 6B | **50%** |

For 10K atoms: 120 KB → 60 KB (fits better in L2 cache)

### Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `amber_simd_batch.rs` | MODIFY | Add FP16 buffer allocation |
| `amber_simd_batch.cu` | MODIFY | Add FP16 load paths |
| `tensor_core_forces.cu` | USE | Already FP16 native |

### Expected Speedup: 1.3-1.5× (bandwidth bound kernels)

---

## Phase 4: Async Kernel Pipeline with CUDA Streams

### Current Problem
```rust
// Sequential execution - GPU idles between kernels
build_cell_list();      // GPU busy, CPU waits
compute_bonded();       // GPU busy, CPU waits
compute_nonbonded();    // GPU busy, CPU waits
integrate();            // GPU busy, CPU waits
```

### Solution: Overlapped Execution with Multiple Streams

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASYNC PIPELINE ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stream 0 (Forces):        Stream 1 (Integration):              │
│  ┌────────────────┐        ┌────────────────┐                   │
│  │ Bonded Forces  │───────▶│ Half Kick 1    │                   │
│  │ (independent)  │        │ (needs bonded) │                   │
│  └────────────────┘        └────────────────┘                   │
│         │                          │                             │
│         ▼                          ▼                             │
│  ┌────────────────┐        ┌────────────────┐                   │
│  │ Non-bonded     │───────▶│ Drift          │                   │
│  │ Forces         │        │ (needs NB)     │                   │
│  └────────────────┘        └────────────────┘                   │
│         │                          │                             │
│         ▼                          ▼                             │
│  ┌────────────────┐        ┌────────────────┐                   │
│  │ Event: Forces  │        │ Half Kick 2    │                   │
│  │ Complete       │        │ + Thermostat   │                   │
│  └────────────────┘        └────────────────┘                   │
│                                                                  │
│  Stream 2 (Next Step Prep):                                     │
│  ┌────────────────┐                                             │
│  │ Check Verlet   │  (overlapped with current step integrate)   │
│  │ Displacement   │                                             │
│  └────────────────┘                                             │
│         │                                                        │
│         ▼ (if needed)                                           │
│  ┌────────────────┐                                             │
│  │ Rebuild Verlet │                                             │
│  │ List           │                                             │
│  └────────────────┘                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
pub struct AsyncMdPipeline {
    stream_forces: Arc<CudaStream>,
    stream_integrate: Arc<CudaStream>,
    stream_verlet: Arc<CudaStream>,

    event_bonded_done: CudaEvent,
    event_nonbonded_done: CudaEvent,
    event_forces_done: CudaEvent,
}

impl AsyncMdPipeline {
    pub fn step(&mut self) {
        // Stream 0: Compute forces
        self.stream_forces.launch(bonded_kernel);
        self.event_bonded_done.record(&self.stream_forces);

        self.stream_forces.launch(nonbonded_kernel);
        self.event_forces_done.record(&self.stream_forces);

        // Stream 1: Integration (waits for forces)
        self.stream_integrate.wait_event(&self.event_bonded_done);
        self.stream_integrate.launch(half_kick_1);

        self.stream_integrate.wait_event(&self.event_forces_done);
        self.stream_integrate.launch(drift);
        self.stream_integrate.launch(half_kick_2);

        // Stream 2: Prepare next step (overlapped)
        self.stream_verlet.launch(check_displacement);
    }
}
```

### Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `amber_simd_batch.rs` | MODIFY | Add multi-stream support |
| `stream_integration.rs` | USE | Existing stream utilities |

### Expected Speedup: 1.1-1.3× (latency hiding)

---

## Implementation Order

### Week 1: Verlet Lists
1. Create `verlet_list.cu` with build/check kernels
2. Add `compute_nonbonded_verlet()` device function
3. Modify `amber_simd_batch.rs` for Verlet management
4. Benchmark: Target 2× speedup

### Week 2: Tensor Cores
1. Create `tensor_core_forces.cu` with WMMA kernels
2. Implement 8×8 tiled distance computation
3. Integrate with Verlet list iteration
4. Benchmark: Target additional 2× (cumulative 4×)

### Week 3: FP16 + Async
1. Add FP16 parameter buffers
2. Modify kernels for FP16 loads
3. Implement multi-stream pipeline
4. Benchmark: Target additional 1.5× (cumulative 6×)

### Week 4: Integration & Tuning
1. Full integration testing
2. Auto-tuning for different system sizes
3. Determinism verification
4. Final benchmarks

---

## Validation Requirements

### Correctness Tests
```bash
# Energy conservation (NVE)
cargo test --release -p prism-gpu energy_conservation_verlet
cargo test --release -p prism-gpu energy_conservation_tensor_core

# Determinism (same seed = same trajectory)
cargo test --release -p prism-gpu determinism_verlet
cargo test --release -p prism-gpu determinism_tensor_core

# Force accuracy (vs reference)
cargo test --release -p prism-gpu force_accuracy_fp16
```

### Performance Benchmarks
```bash
# Throughput benchmark
cargo run --release -p prism-validation --bin benchmark-sota -- \
  --structures data/curated_14/ \
  --n-steps 1000 \
  --output results/sota_benchmark.json

# Expected: 400-800 ns/day for 3000-atom protein
```

---

## Risk Mitigation

### Feature Flags
```rust
#[cfg(feature = "verlet-lists")]
mod verlet_list;

#[cfg(feature = "tensor-cores")]
mod tensor_core_forces;

#[cfg(feature = "fp16-params")]
mod fp16_params;

#[cfg(feature = "async-pipeline")]
mod async_pipeline;
```

### Fallback Strategy
Each optimization is independent. If issues found:
1. Disable feature flag
2. Benchmark regression
3. Fix or revert
4. Continue with other optimizations

---

## Success Metrics

| Metric | Current | Target | SOTA Reference |
|--------|---------|--------|----------------|
| ns/day (3K atoms) | 30 | 400-800 | OpenMM: 300-500 |
| ns/day (10K atoms) | 10 | 100-200 | OpenMM: 100-200 |
| Determinism | Yes | Yes | N/A |
| Energy drift | <1e-3 | <1e-4 | <1e-4 |
| Memory overhead | Baseline | +50% | N/A |

**Target: 10-20× speedup, matching or exceeding OpenMM performance**
