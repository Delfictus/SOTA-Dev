# HANDOFF: SOTA MD Engine Validation

**Date**: 2026-01-18
**Commit**: 9d5ec06
**Branch**: main

---

## SESSION SUMMARY

This session implemented SOTA (State-of-the-Art) performance optimizations for the PRISM4D SIMD batched AMBER MD engine, then attempted validation benchmarks.

### What Was Completed

1. **SOTA Optimization Modules Created**:
   - `crates/prism-gpu/src/verlet_list.rs` + `verlet_list.cu` - Verlet neighbor lists with skin buffer
   - `crates/prism-gpu/src/tensor_core_forces.rs` + `tensor_core_forces.cu` - Tensor Core WMMA acceleration
   - `crates/prism-gpu/src/async_md_pipeline.rs` - Async CUDA stream pipeline

2. **AmberSimdBatch Integration** (`amber_simd_batch.rs`):
   - `OptimizationConfig` struct for selective feature enablement
   - `new_with_config()` constructor for SOTA mode
   - FP16 parameter storage for LJ parameters
   - `SotaStats` for performance reporting
   - Verlet list integration with adaptive rebuilding

3. **Validation Benchmark** (`sota_validation_benchmark.rs`):
   - Raw PDB → Sanitize → AMBER Topology → SIMD Batch MD pipeline
   - Sequential vs Batched performance comparison
   - Tested on 6 structures from curated_14

4. **Structure Analysis** (`data/curated_14/STRUCTURE_ANALYSIS.md`):
   - Categorized 14 structures by size tier
   - Selected 6 for validation (2 per tier): 3SQQ, 4QWO, 6LU7, 1AKE, 6M0J, 4J1G

### Benchmark Results (Current State)

**Problem**: Batched processing showed NO speedup (0.93×) compared to sequential:
- Sequential: 118,989ms for 6 structures
- Batched: 128,318ms for 6 structures

**Likely Causes**:
1. Structures have very different sizes (2K-7K atoms) causing padding overhead
2. The batched approach works best with SAME structure cloned N times (for ensemble generation)
3. Different topologies per structure adds management overhead

---

## NEXT SESSION: MANDATORY ACTIONS

### 1. Use Official Prism-Prep Tool ONLY

**CRITICAL**: The user mandated that ALL future structure preparation MUST use the official prism-prep tool:

```
https://github.com/Delfictus/Prism4D-bio/releases/tag/v1.2.0-prism-prep
```

**DO NOT** use:
- PdbSanitizer
- AmberTopology::from_pdb_atoms()
- Any ad-hoc preprocessing

### 2. Select 3 Test Structures

From this list (in `data/fresh_download/raw/`):
```
1HXY_apo.pdb / 1HXY_fresh.pdb  - 4 chains (tetramer)
2VWD_apo.pdb / 2VWD_fresh.pdb  - 2 chains (Nipah virus)
4B7Q_apo.pdb / 4B7Q_fresh.pdb  - 4 chains (large complex)
5IRE_apo.pdb / 5IRE_fresh.pdb  - 6 chains (hexamer, viral)
6M0J_apo.pdb / 6M0J_fresh.pdb  - 2 chains (SARS-CoV-2 RBD-ACE2)
```

Pick:
- **Easy**: 6M0J (2 chains, well-characterized)
- **Medium**: 1HXY (4 chains, tetramer)
- **Hard**: 5IRE (6 chains, hexamer)

### 3. Pipeline to Execute

```
Raw PDB → [Official Prism-Prep Tool] → Prepared topology → AmberSimdBatch MD
```

---

## KEY FILES

### SOTA Modules (New)
```
crates/prism-gpu/src/verlet_list.rs          # Verlet neighbor list (Rust)
crates/prism-gpu/src/verlet_list.cu          # Verlet neighbor list (CUDA)
crates/prism-gpu/src/tensor_core_forces.rs   # Tensor Core forces (Rust)
crates/prism-gpu/src/tensor_core_forces.cu   # Tensor Core forces (CUDA)
crates/prism-gpu/src/async_md_pipeline.rs    # Async stream pipeline
```

### Modified
```
crates/prism-gpu/src/amber_simd_batch.rs     # SOTA integration (+749 lines)
crates/prism-gpu/src/lib.rs                  # Module exports
crates/prism-gpu/build.rs                    # CUDA kernel compilation
```

### Benchmark
```
crates/prism-validation/src/bin/sota_validation_benchmark.rs
```

### Documentation
```
docs/plans/AMBER_SIMD_BATCH_SOTA_INTEGRATION.md  # Integration plan
data/curated_14/STRUCTURE_ANALYSIS.md            # Structure categorization
```

---

## TECHNICAL CONTEXT

### The MD Engine Architecture

```
AmberSimdBatch (GPU MD Engine)
├── Bonded Forces (bonds, angles, dihedrals)
├── Non-bonded Forces (LJ + Coulomb)
│   ├── Verlet neighbor list (SOTA)
│   └── Tensor Core acceleration (SOTA)
├── Langevin Thermostat
├── Velocity Verlet Integration
└── SIMD Batching (multiple structures in parallel)
```

### OptimizationConfig Options

```rust
pub struct OptimizationConfig {
    pub use_verlet_list: bool,      // Verlet neighbor lists
    pub use_tensor_cores: bool,     // Tensor Core WMMA
    pub use_fp16_params: bool,      // FP16 LJ parameters
    pub use_async_pipeline: bool,   // Async stream overlap
    pub use_batched_forces: bool,   // True batched processing
}

// Presets:
OptimizationConfig::default()      // All SOTA enabled
OptimizationConfig::conservative() // Just Verlet + batching
OptimizationConfig::legacy()       // No SOTA (baseline)
```

### Key Constants

```rust
pub const NB_CUTOFF: f32 = 10.0;           // Non-bonded cutoff (Å)
pub const VERLET_SKIN: f32 = 2.0;          // Verlet skin buffer (Å)
pub const MAX_BATCH_SIZE: usize = 128;     // Max structures per batch
pub const MAX_ATOMS_PER_STRUCT: usize = 8192;
```

---

## KNOWN ISSUES

1. **Batched performance regression**: Different-sized structures cause overhead
2. **High PE values**: Some structures show unrealistic energies (need better equilibration)
3. **Temperature drift**: Final temperatures not always at target (thermostat tuning needed)

---

## COMMANDS FOR NEXT SESSION

```bash
# Build the benchmark
cargo build --release --features cuda -p prism-validation --bin sota_validation_benchmark

# Run validation (after prism-prep)
cargo run --release --features cuda -p prism-validation --bin sota_validation_benchmark -- \
    --pdb-dir data/prepared \
    --pdbs 6M0J 1HXY 5IRE \
    --steps 10000 \
    --output results/sota_validation

# Check SOTA stats
cat results/sota_validation/benchmark_report.json | jq '.summary'
```

---

## USER PREFERENCES NOTED

1. **Official prism-prep tool is MANDATORY** - no exceptions
2. Full-atom AMBER ff14SB implicit solvent MD
3. Focus on batched parallel processing speedup
4. Test across easy/medium/hard structure complexity

---

## CONTINUATION PROMPT

```
Continue PRISM4D SOTA MD validation. Previous session implemented Verlet lists,
Tensor Core forces, and async pipeline in amber_simd_batch.rs. Benchmarks showed
no speedup due to structure size mismatch.

MANDATORY: Download and use the official prism-prep tool from:
https://github.com/Delfictus/Prism4D-bio/releases/tag/v1.2.0-prism-prep

Select 3 structures (easy: 6M0J, medium: 1HXY, hard: 5IRE) from data/fresh_download/raw/,
run through prism-prep, then test with the SOTA MD engine to measure real batched speedup.

Read HANDOFF_SOTA_MD_VALIDATION.md for full context.
```
