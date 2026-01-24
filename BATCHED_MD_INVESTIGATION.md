# SIMD Batched MD Investigation Report

**Date**: 2026-01-18
**Investigation Goal**: Determine why batched parallel processing is slower than sequential

## Executive Summary

The current `AmberSimdBatch` implementation shows **negative speedup** (batched is 30-53% slower than sequential). This is caused by a fundamental architectural issue in how structures are batched.

## Test Results

Using 6LU7 (4,730 atoms) with prism-prep official topologies:

| Clones | Sequential | Batched | Speedup | Notes |
|--------|------------|---------|---------|-------|
| 2 | 9.33s | 13.29s | 0.70× | 30% slower |
| 4 | 18.83s | 37.22s | 0.51× | 49% slower |
| 8 | 37.73s | 80.50s | 0.47× | 53% slower |
| 16 | 73.86s | 152.43s | 0.48× | 52% slower |

**Conclusion**: The more structures batched, the WORSE the performance.

## Root Cause Analysis

### 1. Shared Cell List Architecture

The batched kernel uses a single cell list for ALL structures:

```cuda
// All atoms from all structures go into the same cell grid
int cell_idx = ix + iy * MAX_CELLS_X + iz * MAX_CELLS_X * MAX_CELLS_Y;
atom_cell[tid] = cell_idx;
```

Even though structures are spatially separated by 100Å (BATCH_SPATIAL_OFFSET), the cell list spans all space, causing:
- Atoms from different structures can end up in neighboring cells
- Neighbor searches must filter out cross-structure atoms
- Cell list size scales with batch size

### 2. Per-Step Overhead

Each MD step performs:
```rust
// Zero energy accumulators - O(n_structures)
let zero_energies = vec![0.0f32; self.alloc_energies_size];
self.stream.memcpy_htod(&zero_energies, &mut self.d_energies)?;

// Verlet list checks - O(total_atoms)
verlet.needs_rebuild(&self.d_positions)?
```

With N structures, overhead scales with N × atoms_per_structure.

### 3. True Parallelism Not Achieved

The kernel launches with `total_atoms` threads, but each thread still processes atoms from all structures. There's no structure-level parallelism - it's just processing more data.

## Recommended Solutions

### Option 1: Multi-Stream Parallelism (Easiest)

Run multiple independent AmberSimdBatch instances on separate CUDA streams:

```rust
use std::thread;
use rayon::prelude::*;

// Process structures in parallel using separate streams
let results: Vec<_> = structures
    .par_iter()
    .map(|structure| {
        let context = CudaContext::new(0)?;
        let mut batch = AmberSimdBatch::new_with_config(
            context.clone(),
            structure.n_atoms + 100,
            1,  // Single structure per batch
            OptimizationConfig::default(),
        )?;
        batch.add_structure(structure)?;
        batch.finalize_batch()?;
        batch.run(n_steps, dt, temperature, gamma)?;
        batch.get_all_results()
    })
    .collect();
```

**Expected Speedup**: ~2-4× depending on GPU memory bandwidth

### Option 2: Per-Structure Cell Lists

Modify the kernel to maintain separate cell lists per structure:

```cuda
// Each structure gets its own cell list region
int struct_cell_offset = struct_idx * MAX_CELLS_PER_STRUCT;
int cell_idx = struct_cell_offset + local_ix + local_iy * nx + local_iz * nx * ny;
```

This requires:
- Larger cell list allocation: `MAX_CELLS_PER_STRUCT * n_structures`
- Modified neighbor search to only check within-structure cells

**Expected Speedup**: ~N× for N structures (theoretical)

### Option 3: Warp-Level Structure Assignment

Assign entire warps (32 threads) to single structures:

```cuda
__global__ void batched_md_kernel(...) {
    int warp_id = threadIdx.x / 32;
    int struct_idx = warp_id % n_structures;

    // All threads in warp work on same structure
    BatchStructureDesc desc = batch_descs[struct_idx];
    // ...
}
```

This enables:
- Coalesced memory access within each structure
- Shared memory usage for structure-local data
- Proper warp-level synchronization

**Expected Speedup**: ~N× for N structures fitting in GPU memory

## Immediate Workaround

Until the kernel is fixed, use **sequential processing** for actual work:

```bash
# Process each structure independently (actually faster!)
for structure in structures:
    ./prism_md --input $structure --steps 10000
```

Or use the existing benchmark with batch_size=1.

## Files to Modify for Fixes

1. `crates/prism-gpu/src/kernels/amber_simd_batch.cu` - CUDA kernel
2. `crates/prism-gpu/src/amber_simd_batch.rs` - Host-side batching logic
3. `crates/prism-gpu/src/verlet_list.rs` - Per-structure Verlet lists

## Verification Commands

```bash
# Run ensemble speedup test
cargo run --release --features cuda -p prism-validation --bin ensemble_speedup_test -- \
    --topology results/sota_validation_fresh/6LU7_topology.json \
    --clones 2 4 8 16 \
    --steps 500

# Expected: Batched should be FASTER than sequential after fixes
```

## Conclusion

The current batched MD implementation has a **negative scaling problem** - adding more structures makes it slower. This is a fundamental architectural issue that requires kernel redesign.

For now, **sequential processing is recommended** for production workloads.
