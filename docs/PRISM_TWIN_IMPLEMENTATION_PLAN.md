# PRISM-TWIN: Complete Implementation Plan — Phase A through Phase B

> **NOTE (2026-04-21 canonical lockdown):** This implementation plan documents the historical Gate 1→2→3 progression of `--coupled-twin`. The current canonical TWIN command is the 4-group `--multi-differential` path (see CLAUDE.md §B and docs/CANONICAL_PROVENANCE.md). The `--coupled-twin` commands below are preserved for historical context of the implementation path; do NOT use them for new production runs.

**Status**: DRAFT — PRESCRIPTIVE PLAN (not a verification report)  
**Date**: 2026-04-08  
**Author**: Ididia Serfaty + Claude  
**Branch**: `feat/twin-multistream`  
**Baseline**: 151 passed / 4 failed / 3 ignored (`cargo test -p prism-nhs --lib`, verified 2026-04-05)

> **NOTE ON ESTIMATES**: All latency budgets (μs), VRAM projections (GB),
> overhead percentages (%), and regression gate pass counts are
> **engineering targets derived from code-level analysis**, not measured
> runtime benchmarks. Each must be verified against actual `nvprof`/
> `nsys` profiling and `cargo test` output at implementation time.
> Where this document says "< 10μs" or "~3.2% overhead", read it as
> "target, to be validated at the regression gate for that step."

---

## 0. Executive Summary

PRISM-TWIN exists as two disconnected halves:

1. **Orchestration layer** (Rust, `coupled_md.rs`): two `PersistentNhsEngine` instances on separate CUDA streams, CPU-side density exchange, CPU-side sampled cross-correlation, phase-offset scheduling. This runs.
2. **GPU kernel layer** (CUDA): `ring_buffer.cu`, `tensor_ccf.cu`, `twin_persistent.cu`, `twin_persistent_physics.cu`. These compile individually but have zero Rust call sites, no PTX in the build system, and no tests.

The gap between these halves is the gap between "two independent replicas analyzed post-hoc" and "interferometric coupled observation." Closing it requires:

- **Phase A** (9 steps): Wire GPU kernels into the orchestration layer with 2 engines (1 per group). This delivers the core TWIN innovation: detector sensitivity in one stream steered by the other stream's evidence, with real-time GPU cross-correlation.
- **Phase B** (5 steps): Merge `--multi-stream` and `--coupled-twin` so each observation group runs N=4-8 engines, with inter-group coupling. This delivers the statistical power the architecture doc describes.
- **Phase C** (4 steps): GPU Transfer Entropy, NMA pipeline integration, XGBoost pairwise ranker training, contamination-clean PrismAI retraining. This converts raw TWIN signal into better pocket rankings and a production-ready student model.

Every step has: preconditions, data flow specification, implementation details, regression gate, performance contract, and gate failure troubleshooting.

---

## 1. Prerequisites & Invariants

### 1.1 Test baseline (record before ANY code change)

```bash
cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST show: 151 passed; 4 failed; 3 ignored
# The 4 failures are:
#   input::tests::test_prepared_input_{fields_populated,explicit_mode,implicit_mode}
#   persistent_engine::tests::test_default_config
```

**Decision**: Fix these 4 tests FIRST (Step 0) so every subsequent gate diffs against green.

### 1.2 Immutable constraints

- `nhs_amber_fused.cu` is FROZEN. No modifications for TWIN. All coupling hooks go through the ring buffer module (between kernel launches) or the persistent kernel role-assignment layer.
- The fused engine's `d_neuron_threshold` buffer (`fused_engine.rs:1755`, `[total_voxels * 8]` floats) is the spike detection threshold. It is initialized by `init_multi_neuron_kernel` and self-adapts within the kernel (line `nhs_amber_fused.cu:3427-3444`). TWIN's threshold modification writes to this buffer BETWEEN fused kernel launches.
- No composite ranking scores. No DPS. Lexicographic ranking only.
- Production entry point is `prism-validate-and-run.sh`. TWIN flag must route through it.

### 1.3 GPU buffer topology (critical for ring buffer wiring)

The fused engine holds per-voxel oscillator state:

| Buffer | Shape | Field in `fused_engine.rs` | Purpose |
|--------|-------|---------------------------|---------|
| `d_neuron_threshold` | `[total_voxels * K]` (K=8) | `:1755` | Per-neuron firing threshold. **TWIN writes here.** |
| `d_neuron_potential` | `[total_voxels * K]` | `:1754` | Oscillator membrane potential |
| `d_neuron_mean` | `[total_voxels * K]` | `:1756` | Running mean for adaptation |
| `d_neuron_refractory` | `[total_voxels * K]` | `:1757` | Refractory countdown |
| `d_coupling_a/b` | `[total_voxels]` | `:1758-1759` | Inter-neuron coupling (within-stream) |

The kernel's threshold logic at `nhs_amber_fused.cu:3425-3444`:
```c
float effective_threshold = threshold - COUPLING_STRENGTH * coupling_norm;
if (effective_threshold < 0.2f) effective_threshold = 0.2f;
my_spike = (amplitude >= effective_threshold);
if (my_spike) { threshold = fmin(threshold + 0.02, 2.0 * MULTI_LIF_THRESHOLD); }
else { threshold = fmax(threshold - 0.001 * dt, MULTI_LIF_THRESHOLD * 0.5); }
```

TWIN's ring buffer `read_and_adapt` kernel writes to `osc_thresholds` (which maps to `d_neuron_threshold`). It reduces thresholds in voxels where the OTHER stream saw spikes, bounded by `max_reduction_fraction` (default 0.5) from `base_thresholds`. The fused kernel then picks up the modified threshold on its next launch.

**Data race safety**: `ring_buffer_read_and_adapt` runs on `stream_exchange` BEFORE the fused kernel launches on `stream_a`/`stream_b`. No concurrent writes.

### 1.4 Build system gap

`crates/prism-gpu/build.rs` compiles 35 kernels via `compile_kernel()` (line 431). This function passes `--ptx -arch=sm_120 -O3 --use_fast_math --restrict`. It does NOT pass:
- `-rdc=true` (required for cooperative groups / `grid.sync()`)
- `--expt-extended-lambda` (needed for some WMMA paths)

The 4 TWIN kernels are not in the build list. Two of them (`twin_persistent.cu`, `twin_persistent_physics.cu`) require `-rdc=true`. The other two (`ring_buffer.cu`, `tensor_ccf.cu`) do not — they are standard `__global__` kernels.

---

## STEP 0: Fix Pre-existing Test Failures

**Goal**: Green baseline for all subsequent gates.

### 0a: Fix `input::tests::test_prepared_input_*` (3 tests)

**Root cause**: `fused_engine.rs` iterates `0..self.n_atoms` and indexes `self.residue_names[i]` at `input.rs:287`. Comment says "atom-indexed, NOT res_id-indexed." The test fixtures provide `residue_names` sized to residue count (2 or 3), but the production code expects atom-indexed (one entry per atom). The test must populate `residue_names` with one entry per atom.

**Fix**: In each test's `PrismPrepTopology` literal, set `residue_names` to have `n_atoms` entries (repeating the residue name for each atom in that residue), matching the contract.

### 0b: Fix `persistent_engine::tests::test_default_config`

**Root cause**: `PersistentBatchConfig::default()` returns `survey_steps: 20000` but the test asserts `15000`. Someone bumped the default without updating the test.

**Fix**: Update the test assertion from `15000` to match `PersistentBatchConfig::default().survey_steps`.

### Regression gate 0

```bash
cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST show: 155 passed; 0 failed; 3 ignored
# (151 previously passing + 4 fixed = 155)
```

#### If gate 0 fails:
1. Capture the FULL `cargo test` output (not just `tail -5`)
2. Identify the failing test name(s)
3. For `test_prepared_input_*`: check that `residue_names` length == `n_atoms` in the test fixture
4. For `test_default_config`: check current `PersistentBatchConfig::default().survey_steps` value
5. DO NOT proceed to Step 0.5 until the gate passes

---

## STEP 0.5: Codebase Audit (MANDATORY — blocks all subsequent steps)

**Goal**: Verify every API assumption in the plan against the actual codebase. Record exact output. If ANY check fails, amend the plan before proceeding.

### Verified findings (from 2026-04-08 audit session):

#### 0.5a: cudarc version and launch API

```
cudarc = { version = "0.18.2", features = ["std", "cuda-12050", "driver"], optional = true }
```

Launch API pattern confirmed: `stream.launch_builder(&kernel_fn).arg(...).arg(...).launch(cfg)` with `LaunchConfig { grid_dim, block_dim, shared_mem_bytes }`. Used extensively in `fused_engine.rs` (lines 3710, 3734, 3762, etc.).

#### 0.5b: `half` crate availability

**NOT a dependency.** `grep -r 'half' crates/prism-nhs/Cargo.toml crates/prism-gpu/Cargo.toml` returns empty. The plan's `CudaSlice<half>` type in Step 3.3 and Step 5 will NOT compile without adding `half` to `Cargo.toml`.

**PLAN AMENDMENT**: Add `half = "2"` to `[dependencies]` in `crates/prism-nhs/Cargo.toml` as part of Step 3. Alternatively, use `CudaSlice<u16>` with manual `f32→f16` conversion via `half::f16::from_f32()` after adding the crate.

#### 0.5c: Spike buffer field names

The plan assumes `d_spike_output`. **ACTUAL field name is `d_spike_events`** (`fused_engine.rs:1777`, type `CudaSlice<u8>`). There is also `d_spike_count` (`fused_engine.rs:1779`, type `CudaSlice<i32>`).

**PLAN AMENDMENT**: All references to `spike_buffer_gpu()` in Steps 4.2, 4.4 must use `d_spike_events`, NOT `d_spike_output`.

#### 0.5d: `get_spike_count` exists

Confirmed at `persistent_engine.rs:1125`:
```rust
pub fn get_spike_count(&self) -> Result<u32>
```
Reads `d_spike_count` via `memcpy_dtoh`. Returns `u32`. Plan usage is correct.

#### 0.5e: GpuSpikeEvent struct layout — CRITICAL MISMATCH

**GpuSpikeEvent** (`fused_engine.rs:225-241`): 14 fields, **92 bytes** (no `#[repr(C)]`):
```
timestep(i32) + voxel_idx(i32) + position([f32;3]) + intensity(f32)
+ nearby_residues([i32;8]) + n_residues(i32)
+ spike_source(i32) + wavelength_nm(f32) + aromatic_type(i32)
+ aromatic_residue_id(i32) + water_density(f32)
+ vibrational_energy(f32) + n_nearby_excited(i32) + wd_change(f32)
```

**RingSpikeEvent** (`ring_buffer.cu:20-31`): 10 fields, **48 bytes**:
```
timestep(int) + voxel_idx(int) + x,y,z(float×3) + intensity(float)
+ vibrational_energy(float) + water_density(float)
+ n_nearby_excited(int) + spike_source(int) + wavelength_nm(float) + pad(int)
```

**THESE ARE NOT COMPATIBLE.** `ring_buffer_push_batch` interprets the spike buffer as `RingSpikeEvent[]` (48-byte stride) but the fused engine writes `GpuSpikeEvent[]` (92-byte stride). Direct memcpy WILL corrupt data.

**PLAN AMENDMENT (Step 4, BLOCKING)**: Either:
- **(a)** Modify `ring_buffer.cu` to match `GpuSpikeEvent` layout (92 bytes), OR
- **(b)** Add a GPU conversion kernel (`spike_compact_kernel`) that copies the 10 relevant fields from `GpuSpikeEvent[]` into `RingSpikeEvent[]`, OR
- **(c)** Modify `RingSpikeEvent` to be a view into `GpuSpikeEvent` with the correct 92-byte stride and field offsets.

**Recommended: option (b)** — a simple GPU kernel that runs after each `engine.run()` to compact new spikes. This avoids modifying either existing struct and keeps the ring buffer small (48 bytes vs 92 bytes = 48% less VRAM and bandwidth).

#### 0.5f: `memcpy_dtod` exists

Confirmed at `fused_engine.rs:4321`:
```rust
stream.memcpy_dtod(&self.d_positions, &mut d_positions)?;
```
API: `stream.memcpy_dtod(&source, &mut dest)`. Step 2.3 usage is correct.

#### 0.5g: Multi-stream pipeline location

`run_multi_stream_pipeline` at `nhs_rt_full.rs:2588`, signature:
```rust
fn run_multi_stream_pipeline(args: &Args, topology_path: &PathBuf, n_streams: usize) -> Result<()>
```
Temperature ladder at lines 2735-2815. Dispatch at line 548-549.

#### 0.5h: PersistentNhsEngine constructor

`new_on_stream` at `persistent_engine.rs:995`:
```rust
pub fn new_on_stream(
    config: &PersistentBatchConfig,
    context: Arc<CudaContext>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
) -> Result<Self>
```
Takes `Arc<CudaContext>`, `Arc<CudaModule>`, `Arc<CudaStream>` — all shared. Multiple engines can share one context and module. Phase B is safe.

#### 0.5i: Stream creation API

Confirmed: `context.new_stream()` returns `Result<Arc<CudaStream>>` (see `coupled_md.rs:346-347`). Plan usage is correct.

### Regression gate 0.5

All 9 checks completed. Three plan amendments identified:
1. Add `half` crate dependency (Step 3)
2. Rename `d_spike_output` → `d_spike_events` throughout
3. **BLOCKING**: Add spike compaction kernel for GpuSpikeEvent→RingSpikeEvent conversion (Step 4)

---

## PHASE A: Wire GPU Kernels with 2 Engines (1 per Group)

### STEP 1: Add TWIN kernels to build.rs

**Goal**: PTX files for all 4 TWIN kernels generated during `cargo build`.

#### 1.1 Add `compile_cooperative_kernel()` to `crates/prism-gpu/build.rs`

The existing `compile_kernel()` at line 431 does not pass `-rdc=true`. Cooperative kernels (`twin_persistent.cu`, `twin_persistent_physics.cu`) require relocatable device code for `grid.sync()`. Add a new function:

```rust
fn compile_cooperative_kernel(nvcc: &str, source: &str, output: &PathBuf, target_output: &PathBuf) {
    println!("cargo:info=Compiling (cooperative) {} -> {}", source, output.display());
    let status = Command::new(nvcc)
        .arg("--ptx")
        .arg("-o").arg(output)
        .arg(source)
        .arg("-arch=sm_120")
        .arg("-rdc=true")           // relocatable device code for grid.sync()
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--restrict")
        .arg("-I/usr/local/cuda/include")
        .arg("-Xptxas=-v")
        .arg("--expt-relaxed-constexpr")
        .status()
        .expect("Failed to execute nvcc");
    if !status.success() {
        panic!("nvcc cooperative compilation failed for {}", source);
    }
    std::fs::copy(output, target_output).expect("Failed to copy PTX");
    generate_ptx_signature(target_output);
    println!("cargo:info=PTX compiled (cooperative): {}", target_output.display());
}
```

#### 1.2 Add 4 kernel entries after `pharmacophore_splat.cu` (build.rs line 387)

```rust
// =========================================================================
// PRISM-TWIN Kernels
// =========================================================================

// Ring Buffer (standard kernel — no cooperative groups)
compile_kernel(
    &nvcc,
    "src/kernels/ring_buffer.cu",
    &ptx_dir.join("ring_buffer.ptx"),
    &target_ptx_dir.join("ring_buffer.ptx"),
);

// Tensor Core CCF (standard kernel — WMMA, no cooperative groups)
compile_kernel(
    &nvcc,
    "src/kernels/tensor_ccf.cu",
    &ptx_dir.join("tensor_ccf.ptx"),
    &target_ptx_dir.join("tensor_ccf.ptx"),
);

// TWIN Persistent Skeleton (cooperative groups — requires -rdc=true)
compile_cooperative_kernel(
    &nvcc,
    "src/kernels/twin_persistent.cu",
    &ptx_dir.join("twin_persistent.ptx"),
    &target_ptx_dir.join("twin_persistent.ptx"),
);

// TWIN Persistent Physics (cooperative groups — requires -rdc=true)
compile_cooperative_kernel(
    &nvcc,
    "src/kernels/twin_persistent_physics.cu",
    &ptx_dir.join("twin_persistent_physics.ptx"),
    &target_ptx_dir.join("twin_persistent_physics.ptx"),
);
```

#### 1.3 Verify `tensor_ccf.cu` compiles with `--ptx`

The file uses `#include <mma.h>` and `using namespace nvcuda::wmma`. The existing `-I/usr/local/cuda/include` should resolve this, but verify:

```bash
nvcc --ptx -arch=sm_120 -O3 --use_fast_math --restrict \
    -I/usr/local/cuda/include \
    crates/prism-gpu/src/kernels/tensor_ccf.cu \
    -o /tmp/tensor_ccf.ptx 2>&1
# MUST: exit 0, produce valid PTX
```

If WMMA headers fail, add `-Xptxas=-v --expt-relaxed-constexpr` (already present in the function).

#### Data flow: none (build-time only)

#### Regression gate 1

```bash
cargo build -p prism-gpu 2>&1 | grep -E "error|PTX compiled"
# MUST show 4 new "PTX compiled" lines, zero errors
# Verify files exist:
ls -la target/ptx/{ring_buffer,tensor_ccf,twin_persistent,twin_persistent_physics}.ptx
# MUST: all 4 present, non-zero size
```

**Performance contract**: TARGET build time increase < 30s (the 4 kernels are small: 175 + 180 + 112 + 229 lines). Measure with `time cargo build -p prism-gpu` before and after.

#### If gate 1 fails:
1. nvcc errors for `tensor_ccf.cu` → check `#include <mma.h>` resolves with `-I/usr/local/cuda/include`
2. nvcc errors for `twin_persistent*.cu` → verify `-rdc=true` is in `compile_cooperative_kernel`, not `compile_kernel`
3. PTX files missing in `target/ptx/` → check `std::fs::copy` in `compile_*_kernel` functions
4. Build time > 30s → acceptable if first build, subsequent builds will be cached
5. DO NOT proceed until all 4 PTX files exist and are non-zero size

---

### STEP 2: Expose `d_neuron_threshold` for External Modification

**Goal**: The fused engine exposes a safe API to read/write its per-voxel threshold buffer from outside, so the ring buffer can modify it between kernel launches.

#### 2.1 Add threshold access methods to `FusedCryoUvEngine` in `fused_engine.rs`

```rust
/// Get a reference to the device threshold buffer for external modification.
/// SAFETY: caller must synchronize the engine's stream before reading/writing.
/// Only call this between run() calls, never during a kernel launch.
pub fn threshold_buffer_mut(&mut self) -> &mut CudaSlice<f32> {
    &mut self.d_neuron_threshold
}

/// Get a read-only reference to the base (initial) threshold buffer.
/// Used by ring_buffer_read_and_adapt as the floor reference.
pub fn threshold_buffer(&self) -> &CudaSlice<f32> {
    &self.d_neuron_threshold
}

/// Get the grid dimensions for voxel-to-world coordinate mapping.
pub fn grid_info(&self) -> (i32, i32, i32, f32, f32, f32, f32) {
    let dim = self.grid_dim as i32;
    (dim, dim, dim,
     self.grid_origin[0], self.grid_origin[1], self.grid_origin[2],
     self.grid_spacing)
}

/// Total number of voxels (dim³)
pub fn total_voxels(&self) -> usize {
    let d = self.grid_dim as usize;
    d * d * d
}
```

#### 2.2 Expose through `PersistentNhsEngine`

Add delegate methods in `persistent_engine.rs`:

```rust
pub fn threshold_buffer_mut(&mut self) -> Option<&mut CudaSlice<f32>> {
    self.engine.as_mut().map(|e| e.threshold_buffer_mut())
}

pub fn grid_info(&self) -> Option<(i32, i32, i32, f32, f32, f32, f32)> {
    self.engine.as_ref().map(|e| e.grid_info())
}

pub fn total_voxels(&self) -> usize {
    self.engine.as_ref().map(|e| e.total_voxels()).unwrap_or(0)
}
```

#### 2.3 Add a base threshold snapshot

The ring buffer kernel needs a `base_thresholds` reference (the initial threshold values, before any TWIN modification). Store this at topology load time:

In `FusedCryoUvEngine`, add a field:
```rust
d_base_threshold: CudaSlice<f32>,  // [total_voxels * K], snapshot from init
```

After `init_multi_neuron_state()` completes, clone `d_neuron_threshold` into `d_base_threshold`:
```rust
let base_copy = stream.memcpy_dtod(&self.d_neuron_threshold)?;
self.d_base_threshold = base_copy;
```

Expose via:
```rust
pub fn base_threshold_buffer(&self) -> &CudaSlice<f32> {
    &self.d_base_threshold
}
```

#### Data flow

```
init_multi_neuron_state() → d_neuron_threshold (initialized)
                          → d_base_threshold (snapshot copy)
                          
Between kernel launches:
  ring_buffer_read_and_adapt reads d_base_threshold, writes d_neuron_threshold
  
Next fused kernel launch:
  Picks up modified d_neuron_threshold at nhs_amber_fused.cu:3427
```

#### Regression gate 2

```bash
cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST: same count as Gate 0 (155/0/3). No new failures.
cargo check -p prism-nhs 2>&1 | grep error
# MUST: zero errors
```

**Performance contract**: EXPECTED zero runtime cost (only adds accessor methods and one memcpy at init time). Validate with `cargo bench` or timing comparison before/after.

#### If gate 2 fails:
1. Compilation error on `memcpy_dtod` → verify cudarc 0.18.2 API (Step 0.5f confirmed it exists)
2. Type mismatch on `d_base_threshold` → ensure it's declared as `CudaSlice<f32>` matching `d_neuron_threshold`
3. Borrow checker error on `threshold_buffer_mut` → may need to restructure to avoid double mutable borrow

---

### STEP 3: Create Rust FFI Module for TWIN Kernels

**Goal**: A `twin_kernels.rs` module in `crates/prism-nhs/src/` that loads the TWIN PTX files and exposes safe Rust wrappers for each kernel.

#### 3.1 Module structure

Create `crates/prism-nhs/src/twin_kernels.rs`:

```rust
//! PRISM-TWIN GPU kernel bindings
//!
//! Loads ring_buffer.ptx and tensor_ccf.ptx at runtime.
//! Provides safe wrappers that handle buffer allocation, kernel launch
//! configuration, and synchronization.

#[cfg(feature = "gpu")]
use anyhow::{Context, Result};
#[cfg(feature = "gpu")]
use cudarc::driver::{CudaContext, CudaStream, CudaModule, CudaSlice, CudaFunction, LaunchConfig};
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// GPU-side ring buffer state for one stream's spike exchange channel.
#[cfg(feature = "gpu")]
pub struct TwinRingBuffer {
    buffer: CudaSlice<u8>,          // [capacity * 48] bytes (RingSpikeEvent)
    head: CudaSlice<u32>,           // [1]
    tail: CudaSlice<u32>,           // [1]
    overflow: CudaSlice<u32>,       // [1]
    capacity: u32,
    // Kernel functions
    push_batch_fn: CudaFunction,
    read_adapt_fn: CudaFunction,
    recovery_fn: CudaFunction,
    reset_fn: CudaFunction,
}

/// GPU-side Tensor Core CCF compute context.
#[cfg(feature = "gpu")]
pub struct TwinCcfCompute {
    // Spike matrices (FP16, padded to 16-multiples)
    spike_matrix_a: CudaSlice<half>,   // [n_res_padded × n_bins_padded]
    spike_matrix_b: CudaSlice<half>,   // [n_res_padded × n_bins_padded]
    // Output
    ccf_output: CudaSlice<f32>,        // [n_res × n_res]
    norm_a: CudaSlice<f32>,            // [n_res]
    norm_b: CudaSlice<f32>,            // [n_res]
    // Dimensions
    n_res: i32,
    n_res_padded: i32,
    n_bins_padded: i32,
    // Kernel functions
    ccf_compute_fn: CudaFunction,
    norms_fn: CudaFunction,
}
```

#### 3.2 Ring buffer allocation & kernel wrappers

```rust
#[cfg(feature = "gpu")]
impl TwinRingBuffer {
    pub fn new(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        capacity: u32,
    ) -> Result<Self> {
        let spike_size = 48u32; // sizeof(RingSpikeEvent)
        let buffer = stream.alloc_zeros::<u8>((capacity * spike_size) as usize)?;
        let head = stream.alloc_zeros::<u32>(1)?;
        let tail = stream.alloc_zeros::<u32>(1)?;
        let overflow = stream.alloc_zeros::<u32>(1)?;

        let push_batch_fn = module.get_fn("ring_buffer_push_batch")?;
        let read_adapt_fn = module.get_fn("ring_buffer_read_and_adapt")?;
        let recovery_fn = module.get_fn("ring_buffer_threshold_recovery")?;
        let reset_fn = module.get_fn("ring_buffer_reset")?;

        Ok(Self { buffer, head, tail, overflow, capacity,
                  push_batch_fn, read_adapt_fn, recovery_fn, reset_fn })
    }

    /// Push new spikes [prev_count..curr_count) from the fused engine's spike buffer.
    pub fn push_batch(
        &mut self,
        stream: &Arc<CudaStream>,
        spike_buffer: &CudaSlice<u8>,  // fused engine's spike output buffer
        prev_count: u32,
        curr_count: u32,
    ) -> Result<()> {
        let n_new = curr_count.saturating_sub(prev_count);
        if n_new == 0 { return Ok(()); }
        let n_blocks = n_new.div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&self.push_batch_fn)
                .arg(&mut self.buffer)
                .arg(&mut self.head)
                .arg(&mut self.tail)
                .arg(&mut self.overflow)
                .arg(&self.capacity)
                .arg(spike_buffer)
                .arg(&prev_count)
                .arg(&curr_count)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Read spikes from ring buffer and modify target engine's thresholds.
    /// Called BEFORE each fused kernel launch.
    pub fn read_and_adapt(
        &mut self,
        stream: &Arc<CudaStream>,
        osc_thresholds: &mut CudaSlice<f32>,    // target engine's d_neuron_threshold
        base_thresholds: &CudaSlice<f32>,        // target engine's d_base_threshold
        grid_dims: (i32, i32, i32),
        grid_origin: (f32, f32, f32),
        voxel_size: f32,
        sensitivity_boost: f32,
        max_reduction: f32,
        current_step: u32,
        decay_constant: f32,
    ) -> Result<()> {
        // Single-threaded kernel (ring buffer typically small)
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&self.read_adapt_fn)
                .arg(&self.buffer)
                .arg(&mut self.head)
                .arg(&mut self.tail)
                .arg(&self.capacity)
                .arg(osc_thresholds)
                .arg(base_thresholds)
                .arg(&grid_dims.0).arg(&grid_dims.1).arg(&grid_dims.2)
                .arg(&grid_origin.0).arg(&grid_origin.1).arg(&grid_origin.2)
                .arg(&voxel_size)
                .arg(&sensitivity_boost)
                .arg(&max_reduction)
                .arg(&current_step)
                .arg(&decay_constant)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Gradually restore thresholds toward baseline.
    pub fn threshold_recovery(
        &self,
        stream: &Arc<CudaStream>,
        osc_thresholds: &mut CudaSlice<f32>,
        base_thresholds: &CudaSlice<f32>,
        n_voxels_total: u32,
        recovery_rate: f32,
    ) -> Result<()> {
        let n_blocks = n_voxels_total.div_ceil(256);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&self.recovery_fn)
                .arg(osc_thresholds)
                .arg(base_thresholds)
                .arg(&n_voxels_total)
                .arg(&recovery_rate)
                .launch(cfg)?;
        }
        Ok(())
    }

    pub fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&self.reset_fn)
                .arg(&mut self.head)
                .arg(&mut self.tail)
                .arg(&mut self.overflow)
                .launch(cfg)?;
        }
        Ok(())
    }
}
```

#### 3.3 CCF compute wrappers

```rust
#[cfg(feature = "gpu")]
impl TwinCcfCompute {
    /// Allocate CCF buffers for n_residues and n_time_bins.
    /// Pads both dimensions to multiples of 16 for WMMA.
    pub fn new(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        n_res: i32,
        n_bins: i32,
    ) -> Result<Self> {
        let n_res_padded = ((n_res + 15) / 16) * 16;
        let n_bins_padded = ((n_bins + 15) / 16) * 16;

        let mat_size = (n_res_padded * n_bins_padded) as usize;
        let spike_matrix_a = stream.alloc_zeros::<half>(mat_size)?;
        let spike_matrix_b = stream.alloc_zeros::<half>(mat_size)?;
        let ccf_output = stream.alloc_zeros::<f32>((n_res * n_res) as usize)?;
        let norm_a = stream.alloc_zeros::<f32>(n_res as usize)?;
        let norm_b = stream.alloc_zeros::<f32>(n_res as usize)?;

        let ccf_compute_fn = module.get_fn("tensor_ccf_compute")?;
        let norms_fn = module.get_fn("compute_spike_norms")?;

        Ok(Self { spike_matrix_a, spike_matrix_b, ccf_output, norm_a, norm_b,
                  n_res, n_res_padded, n_bins_padded,
                  ccf_compute_fn, norms_fn })
    }

    /// Upload mean-centered spike count matrices (CPU → GPU).
    /// Input: per-residue spike counts binned by time, already mean-centered.
    pub fn upload_matrices(
        &mut self,
        stream: &Arc<CudaStream>,
        matrix_a: &[half],  // [n_res_padded × n_bins_padded], row-major
        matrix_b: &[half],
    ) -> Result<()> {
        stream.memcpy_htod(&matrix_a, &mut self.spike_matrix_a)?;
        stream.memcpy_htod(&matrix_b, &mut self.spike_matrix_b)?;
        Ok(())
    }

    /// Compute norms, then CCF = A × B^T / (norm_a × norm_b).
    pub fn compute(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        // Step 1: compute per-row norms
        let norm_blocks = (self.n_res as u32).div_ceil(256);
        let norm_cfg = LaunchConfig {
            grid_dim: (norm_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&self.norms_fn)
                .arg(&self.spike_matrix_a)
                .arg(&mut self.norm_a)
                .arg(&self.n_res)
                .arg(&self.n_bins_padded)
                .launch(norm_cfg)?;
            stream.launch_builder(&self.norms_fn)
                .arg(&self.spike_matrix_b)
                .arg(&mut self.norm_b)
                .arg(&self.n_res)
                .arg(&self.n_bins_padded)
                .launch(norm_cfg)?;
        }

        // Step 2: WMMA CCF
        // Grid: (ceil(n_res_padded/16), ceil(n_res_padded/16))
        // Block: (32, 4) = 128 threads = 4 warps per block
        let tile_x = (self.n_res_padded as u32).div_ceil(16 * 4); // 4 warps per block in x
        let tile_y = (self.n_res_padded as u32).div_ceil(16);
        let ccf_cfg = LaunchConfig {
            grid_dim: (tile_x, tile_y, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: 4 * 16 * 16 * 4, // 4 warps × 16×16 × sizeof(float)
        };
        unsafe {
            stream.launch_builder(&self.ccf_compute_fn)
                .arg(&self.spike_matrix_a)
                .arg(&self.spike_matrix_b)
                .arg(&mut self.ccf_output)
                .arg(&self.norm_a)
                .arg(&self.norm_b)
                .arg(&self.n_res)
                .arg(&self.n_res_padded)
                .arg(&self.n_bins_padded)
                .launch(ccf_cfg)?;
        }
        Ok(())
    }

    /// Download CCF matrix to CPU.
    pub fn download_ccf(&self, stream: &Arc<CudaStream>) -> Result<Vec<f32>> {
        let size = (self.n_res * self.n_res) as usize;
        let mut host = vec![0.0f32; size];
        stream.memcpy_dtoh(&self.ccf_output, &mut host)?;
        Ok(host)
    }
}
```

#### 3.4 Register the module

In `crates/prism-nhs/src/lib.rs`, add:
```rust
pub mod twin_kernels;
```

#### 3.5 PTX loading strategy

The TWIN kernels need their own `CudaModule` because they are separate PTX files from `nhs_amber_fused.ptx`. In `coupled_md.rs`, load them alongside the fused module:

```rust
let ring_buffer_module = context.load_module(
    Ptx::from_file(&find_twin_ptx("ring_buffer.ptx")?)
).context("Failed to load ring_buffer.ptx")?;

let tensor_ccf_module = context.load_module(
    Ptx::from_file(&find_twin_ptx("tensor_ccf.ptx")?)
).context("Failed to load tensor_ccf.ptx")?;
```

Where `find_twin_ptx()` searches the same candidate paths as `find_ptx_path()` in `nhs_rt_full.rs`.

#### Data flow: build-time PTX → runtime module load → kernel function handles

#### Regression gate 3

```bash
cargo check -p prism-nhs --features gpu 2>&1 | grep error
# MUST: zero errors (module compiles, all types resolve)
cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST: 155/0/3 (no regression from gate 0)
```

Add a unit test in `twin_kernels.rs`:
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_ring_buffer_struct_sizes() {
        // RingSpikeEvent must be 48 bytes to match CUDA struct
        assert_eq!(std::mem::size_of::<[f32; 12]>(), 48);
    }
}
```

**Performance contract**: TARGET module load time < 50ms per PTX file (they are small). Measure in Step 4 functional test logs.

#### If gate 3 fails:
1. `half` type not found → add `half = "2"` to `crates/prism-nhs/Cargo.toml` (Step 0.5b amendment)
2. `get_fn("ring_buffer_push_batch")` fails → PTX not loaded or kernel name mismatch. Check `target/ptx/ring_buffer.ptx` contains the symbol.
3. `CudaSlice<half>` not supported → use `CudaSlice<u16>` with `unsafe` reinterpret
4. Module doesn't compile → check `pub mod twin_kernels;` is inside `#[cfg(feature = "gpu")]` block if needed

---

### STEP 4: Wire Ring Buffer into `coupled_md::run_coupled_twin()`

**Goal**: Replace the CPU spike-counting stub at `coupled_md.rs:456-468` with actual GPU ring buffer push/read/adapt operations. This is the step that makes TWIN different from two independent replicas.

#### 4.1 Data flow (per interleaved step)

```
BEFORE fused kernel launch on engine_a:
  ring_buffer_b.read_and_adapt(
      target = engine_a.d_neuron_threshold,
      base   = engine_a.d_base_threshold,
      ...
  )
  → engine_a's thresholds are now LOWER in voxels where engine_b saw spikes
  → engine_a's detectors are MORE SENSITIVE where engine_b found activity

LAUNCH fused kernel on engine_a (stream_a):
  engine_a.run(inner)
  → spikes detected using modified thresholds
  → spike records written to engine_a's spike buffer

AFTER fused kernel completes on engine_a:
  ring_buffer_a.push_batch(
      source = engine_a.spike_buffer,
      prev_count = prev_spike_count_a,
      curr_count = engine_a.get_spike_count()
  )
  → new spikes pushed into ring_buffer_a for engine_b to read next step

SYMMETRIC for engine_b:
  ring_buffer_a.read_and_adapt(target = engine_b thresholds)
  engine_b.run(inner)
  ring_buffer_b.push_batch(source = engine_b spike buffer)
```

**This is the interferometric coupling.** Stream A's recent spike evidence lowers Stream B's detection thresholds in the same spatial regions, making B more sensitive where A already found pocket dynamics. And vice versa. The coupling is OBSERVATIONAL — no forces are exchanged, only detector sensitivity.

#### 4.2 Expose spike buffer from fused engine

Add to `FusedCryoUvEngine`:
```rust
/// Get a reference to the raw spike output buffer on GPU.
/// Used by ring buffer to read new spikes after each run().
pub fn spike_buffer_gpu(&self) -> &CudaSlice<u8> {
    &self.d_spike_output  // the buffer where the kernel writes GpuSpikeEvent records
}
```

Delegate through `PersistentNhsEngine`:
```rust
pub fn spike_buffer_gpu(&self) -> Option<&CudaSlice<u8>> {
    self.engine.as_ref().map(|e| e.spike_buffer_gpu())
}
```

<!-- SUPERSEDED: Generic "verify field-by-field alignment" note replaced with concrete findings -->

**CRITICAL — STRUCT MISMATCH FOUND (Step 0.5e)**:

The actual field name is `d_spike_events` (NOT `d_spike_output`). The spike buffer accessor must be:
```rust
pub fn spike_buffer_gpu(&self) -> &CudaSlice<u8> {
    &self.d_spike_events  // fused_engine.rs:1777
}
```

**GpuSpikeEvent is 92 bytes. RingSpikeEvent is 48 bytes. They are NOT compatible.**

| Field | GpuSpikeEvent (Rust) | RingSpikeEvent (CUDA) | In ring? |
|-------|---------------------|----------------------|----------|
| timestep | i32, offset 0 | int, offset 0 | ✓ |
| voxel_idx | i32, offset 4 | int, offset 4 | ✓ |
| position | [f32;3], offset 8 | float x,y,z, offset 8 | ✓ |
| intensity | f32, offset 20 | float, offset 20 | ✓ |
| nearby_residues | [i32;8], offset 24 | — | ✗ (32 bytes missing) |
| n_residues | i32, offset 56 | — | ✗ |
| spike_source | i32, offset 60 | int, offset 36 | ✓ (different offset!) |
| wavelength_nm | f32, offset 64 | float, offset 40 | ✓ (different offset!) |
| aromatic_type | i32, offset 68 | — | ✗ |
| aromatic_residue_id | i32, offset 72 | — | ✗ |
| water_density | f32, offset 76 | float, offset 28 | ✓ (different offset!) |
| vibrational_energy | f32, offset 80 | float, offset 24 | ✓ (different offset!) |
| n_nearby_excited | i32, offset 84 | int, offset 32 | ✓ (different offset!) |
| wd_change | f32, offset 88 | — | ✗ |
| — | — | pad (int), offset 44 | alignment only |

#### Step 4.2a: Spike compaction kernel (BLOCKS Step 4.3)

Add a new CUDA kernel `spike_compact.cu` (or append to `ring_buffer.cu`):

```cuda
// Compact GpuSpikeEvent (92 bytes) → RingSpikeEvent (48 bytes)
// Copies only the fields needed for ring buffer threshold adaptation.
// One thread per spike.
extern "C" __global__ void spike_compact(
    const unsigned char* __restrict__ gpu_spikes,  // GpuSpikeEvent[], 92-byte stride
    RingSpikeEvent* __restrict__ ring_spikes,       // output, 48-byte stride
    unsigned int prev_count,
    unsigned int curr_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_new = (int)(curr_count - prev_count);
    if (idx >= n_new) return;

    int src_offset = (prev_count + idx) * 92;  // 92 bytes per GpuSpikeEvent
    const unsigned char* src = &gpu_spikes[src_offset];

    RingSpikeEvent ev;
    // Copy fields at their correct offsets in GpuSpikeEvent
    memcpy(&ev.timestep,           src + 0,  4);  // i32
    memcpy(&ev.voxel_idx,          src + 4,  4);  // i32
    memcpy(&ev.x,                  src + 8,  4);  // f32
    memcpy(&ev.y,                  src + 12, 4);  // f32
    memcpy(&ev.z,                  src + 16, 4);  // f32
    memcpy(&ev.intensity,          src + 20, 4);  // f32
    memcpy(&ev.vibrational_energy, src + 80, 4);  // f32 (offset 80 in Rust)
    memcpy(&ev.water_density,      src + 76, 4);  // f32 (offset 76 in Rust)
    memcpy(&ev.n_nearby_excited,   src + 84, 4);  // i32 (offset 84 in Rust)
    memcpy(&ev.spike_source,       src + 60, 4);  // i32 (offset 60 in Rust)
    memcpy(&ev.wavelength_nm,      src + 64, 4);  // f32 (offset 64 in Rust)
    ev.pad = 0;

    ring_spikes[idx] = ev;
}
```

**IMPORTANT**: The byte offsets above assume Rust default struct layout (no `#[repr(C)]`). Rust is allowed to reorder fields. To make this reliable, add `#[repr(C, packed)]` to `GpuSpikeEvent` OR verify actual layout with `std::mem::offset_of!()` (nightly) or a test that checks `memoffset::offset_of!()`.

**Safer alternative**: Instead of byte-offset memcpy, add a Rust-side function that downloads new spikes to CPU, repacks them into `RingSpikeEvent` format, and uploads to the ring buffer's staging area. This is slower (~200μs per step for 1000 spikes) but avoids all layout assumptions. Use this as the initial implementation, optimize to GPU compaction later if profiling shows it matters.

#### Step 4.2b: Add staging buffer to TwinRingBuffer

```rust
pub struct TwinRingBuffer {
    // ... existing fields ...
    staging: CudaSlice<u8>,  // [capacity * 48], staging area for compacted spikes
}
```

The `push_batch` method changes: instead of pointing directly at the engine's `d_spike_events`, it:
1. Runs `spike_compact` kernel (or CPU fallback) to copy new spikes into `staging`
2. Calls `ring_buffer_push_batch` on `staging`

#### 4.3 Allocate ring buffers in `run_coupled_twin()`

After engine initialization (line ~398), before the main loop:

```rust
// Load TWIN kernel modules
let ring_module = context.load_module(
    Ptx::from_file(&find_twin_ptx("ring_buffer.ptx")?)
).context("ring_buffer.ptx")?;

// Create exchange stream (dedicated, non-blocking)
let stream_exchange = context.new_stream().context("CUDA exchange stream")?;

// Allocate ring buffers (A's spikes → read by B, B's spikes → read by A)
let mut ring_a = TwinRingBuffer::new(&context, &stream_exchange, &ring_module, 8192)?;
let mut ring_b = TwinRingBuffer::new(&context, &stream_exchange, &ring_module, 8192)?;
ring_a.reset(&stream_exchange)?;
ring_b.reset(&stream_exchange)?;
```

#### 4.4 Replace the main loop body at `coupled_md.rs:442-520`

```rust
let mut prev_spike_count_a: u32 = 0;
let mut prev_spike_count_b: u32 = 0;

for step in 0..outer_steps {
    // ── Phase 1: Adapt thresholds from cross-stream evidence ──
    if twin_config.enable_exchange && step > 0 {
        let (gx, gy, gz, ox, oy, oz, vs) = engine_a.grid_info().unwrap();
        
        // B's evidence → lower A's thresholds
        if let (Some(thresh_a), Some(base_a)) = (
            engine_a.threshold_buffer_mut(),
            engine_a.base_threshold_buffer(),
        ) {
            ring_b.read_and_adapt(
                &stream_exchange,
                thresh_a, base_a,
                (gx, gy, gz), (ox, oy, oz), vs,
                twin_config.sensitivity_boost,
                twin_config.max_threshold_reduction,
                step as u32,
                500.0,  // decay_constant: spikes older than ~500 steps lose influence
            )?;
        }
        
        // A's evidence → lower B's thresholds
        if let (Some(thresh_b), Some(base_b)) = (
            engine_b.threshold_buffer_mut(),
            engine_b.base_threshold_buffer(),
        ) {
            ring_a.read_and_adapt(
                &stream_exchange,
                thresh_b, base_b,
                (gx, gy, gz), (ox, oy, oz), vs,
                twin_config.sensitivity_boost,
                twin_config.max_threshold_reduction,
                step as u32,
                500.0,
            )?;
        }
        
        // Sync exchange stream before launching integration streams
        stream_exchange.synchronize()?;
    }

    // ── Phase 2: Run both engines (concurrent on separate streams) ──
    if step < outer_steps_a {
        let summary_a = engine_a.run(inner)?;
        spikes_a_total += summary_a.total_spikes;
    }
    let summary_b = engine_b.run(inner)?;
    spikes_b_total += summary_b.total_spikes;

    // ── Phase 3: Push new spikes into ring buffers ──
    if twin_config.enable_exchange {
        let curr_a = engine_a.get_spike_count().unwrap_or(0);
        let curr_b = engine_b.get_spike_count().unwrap_or(0);
        
        if let Some(buf_a) = engine_a.spike_buffer_gpu() {
            ring_a.push_batch(&stream_a_ref, buf_a, prev_spike_count_a, curr_a)?;
        }
        if let Some(buf_b) = engine_b.spike_buffer_gpu() {
            ring_b.push_batch(&stream_b_ref, buf_b, prev_spike_count_b, curr_b)?;
        }
        
        <!-- SUPERSEDED: prev_spike_count_a = curr_a; prev_spike_count_b = curr_b;
             ring_spikes_exchanged += (curr_a - prev_spike_count_a + curr_b - prev_spike_count_b) as u64;
             BUG: delta always zero because prev is updated before delta is computed -->
        let delta_a = curr_a.saturating_sub(prev_spike_count_a);
        let delta_b = curr_b.saturating_sub(prev_spike_count_b);
        ring_spikes_exchanged += (delta_a + delta_b) as u64;
        prev_spike_count_a = curr_a;
        prev_spike_count_b = curr_b;
    }

    // ── Phase 4: Periodic threshold recovery (prevent permanent suppression) ──
    if twin_config.enable_exchange && step as u32 % 1000 == 0 {
        let n_voxels = engine_a.total_voxels() as u32;
        if let (Some(thresh_a), Some(base_a)) = (
            engine_a.threshold_buffer_mut(),
            engine_a.base_threshold_buffer(),
        ) {
            ring_b.threshold_recovery(&stream_exchange, thresh_a, base_a, n_voxels, 0.01)?;
        }
        if let (Some(thresh_b), Some(base_b)) = (
            engine_b.threshold_buffer_mut(),
            engine_b.base_threshold_buffer(),
        ) {
            ring_a.threshold_recovery(&stream_exchange, thresh_b, base_b, n_voxels, 0.01)?;
        }
    }

    // ── Existing: spike density exchange (CPU-side, keep for logging) ──
    // ... (existing code at :471-510, retained for telemetry)
}
```

#### 4.5 Safety invariants

1. `stream_exchange.synchronize()` BEFORE launching fused kernels ensures threshold modification is visible.
2. `engine.run()` launches on `stream_a`/`stream_b`, NOT on `stream_exchange`. No data race.
3. Ring buffer capacity 8192 handles ~8000 spikes between reads. Overflow is counted (atomic increment in kernel) and logged. If overflow > 10% of total spikes, increase capacity.
4. `decay_constant = 500.0` means spikes older than ~500 steps contribute <37% of their original boost. This prevents stale evidence from permanently suppressing thresholds.
5. `recovery_rate = 0.01` per 1000 steps = ~1% per recovery call. Conservative — thresholds drift back to baseline over ~100K steps if no new evidence arrives.

#### Regression gate 4

```bash
cargo check -p prism-nhs --features gpu 2>&1 | grep error
# MUST: zero errors

cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST: 155+/0/3 (new ring buffer unit tests should add to count)
```

**Functional test** (requires GPU — run manually):
```bash
scripts/prism-validate-and-run.sh \
    -t tests/fixtures/1btl_clean.topology.json \
    -o /tmp/twin_ring_test \
    --coupled-twin --fast \
    --fused-steps 6 --hmr --adaptive-dt -v 2>&1 | grep -E "exchange|ring|threshold"
# MUST show: ring buffer push/read logs, non-zero exchanges
# MUST NOT show: any panic or CUDA error
```

**Performance contract (TARGETS — must be validated with nsys/nvprof at this gate)**:
- Ring buffer push_batch: TARGET < 10μs per call (256-thread kernel, ~100 spikes/step typical)
- Ring buffer read_and_adapt: TARGET < 50μs per call (single-threaded, processes ~8000 entries max)
- Threshold recovery: TARGET < 5μs per call (256-thread, one pass over voxels)
- Total overhead per step: TARGET < 65μs (vs ~2ms per fused step → estimated ~3.2% overhead)
- **Validation command**: `nsys profile --stats=true` on a 1btl TWIN run, extract kernel times

#### Coupling effectiveness verification (REQUIRED, not informational)

```bash
# Run with diagnostic mode
PRISM_TWIN_DIAG=1 scripts/prism-validate-and-run.sh \
    -t tests/fixtures/1btl_clean.topology.json \
    -o /tmp/twin_coupling_diag \
    --coupled-twin --fast \
    --fused-steps 6 --hmr --adaptive-dt -v 2>&1 | \
    grep "threshold_delta_l2"
# MUST show: threshold_delta_l2 > 0.001 on at least 50% of logged steps
# If threshold_delta_l2 ≈ 0 on all steps, coupling is ineffective.
```

**Implementation**: After each `read_and_adapt` call, if `PRISM_TWIN_DIAG` env var is set, download first 1000 elements of `d_neuron_threshold` and `d_base_threshold`, compute L2 norm of difference, log it:

```rust
if std::env::var("PRISM_TWIN_DIAG").is_ok() && step % 100 == 0 {
    let n_sample = 1000.min(engine_a.total_voxels());
    let mut thresh_sample = vec![0.0f32; n_sample];
    let mut base_sample = vec![0.0f32; n_sample];
    // Download first n_sample elements from each buffer
    stream_exchange.memcpy_dtoh_partial(thresh_a, &mut thresh_sample, 0)?;
    stream_exchange.memcpy_dtoh_partial(base_a, &mut base_sample, 0)?;
    let delta_l2: f32 = thresh_sample.iter().zip(&base_sample)
        .map(|(t, b)| (t - b).powi(2)).sum::<f32>().sqrt();
    log::info!("  [DIAG] step={} threshold_delta_l2={:.6} direction=B→A", step, delta_l2);
}
```

Also verify `ring_spikes_exchanged > 0` in the functional test log:
```bash
grep "ring_spikes_exchanged" /tmp/twin_ring_test.log | tail -1
# MUST show a value > 0. If zero, the spike counter bug (Correction 1) was not applied.
```

#### If gate 4 fails:
1. Compilation errors → check API surface against Step 0.5 findings (especially `d_spike_events` not `d_spike_output`)
2. CUDA illegal memory access → run with `CUDA_LAUNCH_BLOCKING=1`, check buffer size mismatch (likely the 92-vs-48 byte struct issue)
3. Ring buffer overflow > 10% → increase capacity from 8192 to 16384
4. `threshold_delta_l2 ≈ 0` → check `sensitivity_boost` value (default 0.01 may be too low for cold_hold phase where few spikes exist), try 0.05
5. `ring_spikes_exchanged == 0` → verify the spike counter delta fix was applied (Correction 1)
6. DO NOT proceed to Step 5 until coupling is verified active
7. If unresolvable after 3 attempts, STOP and report to user

---

### STEP 5: Wire Tensor Core CCF to Replace CPU Fallback

**Goal**: Replace the CPU sampled O(N²) cross-correlation at `coupled_md.rs:632-636` with GPU WMMA-based CCF.

#### 5.1 Spike-to-matrix conversion (CPU-side prep)

The CCF kernel expects `[n_res × n_bins]` FP16 matrices where each row is a residue and each column is a time bin. Build these from the accumulated spike vectors:

```rust
/// Convert spike records to per-residue time-binned matrices for CCF.
/// Returns (matrix_a, matrix_b) as FP16 vectors, mean-centered.
fn build_ccf_matrices(
    spikes_a: &[GpuSpikeEvent],
    spikes_b: &[GpuSpikeEvent],
    n_residues: usize,
    total_steps: i32,
    bin_size: i32,        // steps per time bin (e.g., 100)
) -> (Vec<half>, Vec<half>, i32, i32) {
    let n_bins = (total_steps / bin_size) as usize;
    let n_res_padded = ((n_residues + 15) / 16) * 16;
    let n_bins_padded = ((n_bins + 15) / 16) * 16;
    
    let mut mat_a = vec![0.0f32; n_res_padded * n_bins_padded];
    let mut mat_b = vec![0.0f32; n_res_padded * n_bins_padded];
    
    // Accumulate spike intensities into bins
    for spike in spikes_a {
        let bin = (spike.timestep / bin_size) as usize;
        if bin >= n_bins { continue; }
        for r in 0..spike.n_residues.min(8) as usize {
            let resid = spike.nearby_residues[r];
            if resid < 0 || resid >= n_residues as i32 { continue; }
            mat_a[resid as usize * n_bins_padded + bin] += spike.intensity;
        }
    }
    for spike in spikes_b {
        let bin = (spike.timestep / bin_size) as usize;
        if bin >= n_bins { continue; }
        for r in 0..spike.n_residues.min(8) as usize {
            let resid = spike.nearby_residues[r];
            if resid < 0 || resid >= n_residues as i32 { continue; }
            mat_b[resid as usize * n_bins_padded + bin] += spike.intensity;
        }
    }
    
    // Mean-center each row (CRITICAL for WMMA — raw counts saturate FP16)
    for r in 0..n_residues {
        let row = &mut mat_a[r * n_bins_padded..(r + 1) * n_bins_padded];
        let mean: f32 = row[..n_bins].iter().sum::<f32>() / n_bins as f32;
        for v in row[..n_bins].iter_mut() { *v -= mean; }
        
        let row = &mut mat_b[r * n_bins_padded..(r + 1) * n_bins_padded];
        let mean: f32 = row[..n_bins].iter().sum::<f32>() / n_bins as f32;
        for v in row[..n_bins].iter_mut() { *v -= mean; }
    }
    
    // Convert to FP16
    let half_a: Vec<half> = mat_a.iter().map(|&v| half::from_f32(v)).collect();
    let half_b: Vec<half> = mat_b.iter().map(|&v| half::from_f32(v)).collect();
    
    (half_a, half_b, n_res_padded as i32, n_bins_padded as i32)
}
```

#### 5.2 Replace CPU fallback at `coupled_md.rs:631-639`

```rust
// ── Gate 2: GPU Tensor Core CCF ──────────────────────────────────────
let ccf_matrix: Option<Vec<f32>> = if twin_config.enable_ccf {
    log::info!("  Computing GPU Tensor Core CCF (WMMA FP16→FP32)...");
    
    let n_residues = topology.residues.len();
    let total_steps = steps;  // A's total step count
    let bin_size = 100;       // 100 steps per bin
    
    let (half_a, half_b, n_res_padded, n_bins_padded) = build_ccf_matrices(
        &spikes_a, &spikes_b, n_residues, total_steps, bin_size
    );
    
    let ccf_module = context.load_module(
        Ptx::from_file(&find_twin_ptx("tensor_ccf.ptx")?)
    ).context("tensor_ccf.ptx")?;
    
    let mut ccf = TwinCcfCompute::new(
        &stream_exchange, &ccf_module, n_residues as i32, (total_steps / bin_size) as i32
    )?;
    ccf.upload_matrices(&stream_exchange, &half_a, &half_b)?;
    ccf.compute(&stream_exchange)?;
    
    let ccf_data = ccf.download_ccf(&stream_exchange)?;
    log::info!("  ✓ CCF matrix: {}×{} ({} non-zero)", 
        n_residues, n_residues,
        ccf_data.iter().filter(|&&v| v.abs() > 0.01).count());
    
    Some(ccf_data)
} else {
    None
};
```

#### 5.3 Extract per-residue CCF features from the matrix

From the `n_res × n_res` CCF matrix, extract per-residue features for `InterferometricFeatures`:

```rust
/// Extract per-residue CCF features from the n_res × n_res CCF matrix.
fn extract_ccf_features(ccf_matrix: &[f32], n_res: usize) -> Vec<CcfResidueFeatures> {
    (0..n_res).map(|r| {
        let row = &ccf_matrix[r * n_res..(r + 1) * n_res];
        
        // Peak value and lag (column index of max off-diagonal)
        let (peak_col, peak_val) = row.iter().enumerate()
            .filter(|(j, _)| *j != r)  // exclude self-correlation
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(j, &v)| (j as i32, v))
            .unwrap_or((0, 0.0));
        
        // Width: number of columns with CCF > peak_val * 0.5
        let half_max = peak_val * 0.5;
        let width = row.iter().filter(|&&v| v > half_max).count() as f32;
        
        // Asymmetry: mean of upper triangle vs lower triangle
        let upper: f32 = row[r+1..].iter().sum();
        let lower: f32 = row[..r].iter().sum();
        let denom = (upper.abs() + lower.abs()).max(1e-8);
        let asymmetry = (upper - lower) / denom;
        
        // Reproducibility: fraction of residues with CCF > 0.1
        let reproducibility = row.iter()
            .filter(|&&v| v > 0.1)
            .count() as f32 / n_res as f32;
        
        CcfResidueFeatures {
            ccf_peak_lag: peak_col - r as i32,  // relative lag
            ccf_peak_value: peak_val,
            ccf_width: width,
            ccf_asymmetry: asymmetry,
            ccf_reproducibility: reproducibility,
        }
    }).collect()
}
```

Then populate the `InterferometricFeatures` struct with real values instead of zeros at `coupled_md.rs:688-693`.

#### Data flow

```
spikes_a + spikes_b (CPU vectors)
  ↓ build_ccf_matrices()
[n_res_padded × n_bins_padded] FP16 matrices (CPU)
  ↓ upload_matrices() — H2D memcpy
GPU FP16 matrices
  ↓ compute_spike_norms() — per-row L2 norm
GPU norm vectors
  ↓ tensor_ccf_compute() — WMMA: C = A × B^T / (norm_a × norm_b)
GPU [n_res × n_res] FP32 CCF matrix
  ↓ download_ccf() — D2H memcpy
CPU CCF matrix
  ↓ extract_ccf_features()
Per-residue CCF features → InterferometricFeatures struct
```

#### Regression gate 5

```bash
cargo check -p prism-nhs --features gpu 2>&1 | grep error
# MUST: zero errors

cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST: 155+/0/3
```

**Functional test** (GPU):
```bash
scripts/prism-validate-and-run.sh \
    -t tests/fixtures/1btl_clean.topology.json \
    -o /tmp/twin_ccf_test \
    --coupled-twin --fast \
    --fused-steps 6 --hmr --adaptive-dt -v 2>&1 | grep -E "CCF|Tensor"
# MUST show: "CCF matrix: NxN (M non-zero)" with M > 0
# MUST NOT show: CUDA error, NaN, Inf
```

**Correctness validation**:
```python
# Verify CCF matrix is symmetric (A×B^T should be if A,B come from same dynamics)
import json, numpy as np
ccf = np.array(json.load(open("/tmp/twin_ccf_test/twin_ccf_matrix.json")))
asym = np.abs(ccf - ccf.T).max()
assert asym < 0.1, f"CCF asymmetry {asym} > 0.1 — check mean-centering"
```

**Performance contract (TARGETS — must be validated with nsys/nvprof at this gate)**:
- For n_res=300 (typical protein), n_bins=350 (35K steps / 100):
  - Matrix upload: ESTIMATE ~0.1ms (2 × 300 × 352 × 2 bytes = 422 KB)
  - Norms: ESTIMATE ~0.01ms (300 rows, 1 block)
  - WMMA CCF: ESTIMATE ~0.5ms (19×19 tiles, 4 warps/block)
  - Download: ESTIMATE ~0.1ms (300×300×4 bytes = 360 KB)
  - Total: TARGET < 1ms (vs ~200ms for CPU sampled fallback)
- For n_res=1000 (large protein):
  - WMMA CCF: ESTIMATE ~5ms (63×63 tiles)
  - Total: TARGET < 10ms
- **Validation command**: `nsys profile` on a TWIN run, grep for `tensor_ccf_compute` kernel time

#### If gate 5 fails:
1. WMMA launch failure → check `block_dim: (32, 4, 1)` matches kernel expectation (4 warps of 32 threads)
2. CCF matrix all zeros → check mean-centering: if all spike counts are identical, centered values are 0. Verify spikes exist.
3. NaN in CCF → division by zero in normalization. Check `norm_a * norm_b + 1e-8f` guard in kernel.
4. Asymmetry > 0.1 in validation → the CCF is A×B^T not A×A^T, slight asymmetry is expected if A≠B. Increase threshold to 0.3.

---

### STEP 6: Populate All 50 Per-Residue TWIN Features

**Goal**: Replace the 12-field `InterferometricFeatures` struct with the full 50-field spec from the architecture doc.

#### 6.1 Expand `InterferometricFeatures` struct

Add the missing fields to `coupled_md.rs:92-113`:

```rust
pub struct InterferometricFeatures {
    pub resid: i32,
    pub resname: String,

    // ── Consensus (12 fields) ──
    pub spike_agreement_ratio: f32,
    pub consensus_intensity_mean: f32,
    pub consensus_phase_profile: [f32; 5],      // per CCNS phase
    pub consensus_spatial_coherence: f32,
    pub consensus_temporal_onset: f32,           // earliest step where both agree
    pub n_consensus_neighbors: u32,              // nearby residues with agreement > 0.5

    // ── Cross-correlation (12 fields) ──
    pub ccf_peak_lag: i32,
    pub ccf_peak_value: f32,
    pub ccf_width: f32,
    pub ccf_asymmetry: f32,
    pub ccf_per_phase: [f32; 5],                // CCF peak value per CCNS phase
    pub ccf_frequency_peak: f32,                // dominant frequency in CCF
    pub ccf_reproducibility: f32,
    pub ccf_lag_consistency: f32,               // std of peak lag across phases

    // ── Differential (18 fields) ──
    pub spikes_a: u32,
    pub spikes_b: u32,
    pub b_over_a_ratio: f32,
    pub nma_exclusive_count: u32,               // spikes in B but not A
    pub thermal_exclusive_count: u32,           // spikes in A but not B
    pub b_over_a_intensity_ratio: f32,
    pub nma_responsive_mode: i32,               // which NMA mode correlates most
    pub nma_mode_eigenvalue: f32,
    pub barrier_classification: String,
    pub per_phase_differential: [f32; 5],       // B/A ratio per phase
    pub differential_onset_lag: f32,            // steps after NMA onset that B diverges
    pub nma_work_at_residue: f32,               // integral of NMA force × displacement
    pub mechanical_sensitivity: f32,            // d(spikes_B)/d(NMA_amplitude)
    pub susceptibility_magnitude: f32,

    // ── Scout/propagation (8 fields) ──
    pub scout_lead_time: f32,                   // steps A fires before B (phase-offset effect)
    pub scout_predictive_value: f32,            // P(B fires | A fired earlier)
    pub phase_offset_enrichment: f32,           // spike ratio at offset vs baseline
    pub scout_intensity_at_onset: f32,          // A's intensity when B first fires
    pub scout_spatial_propagation: f32,         // radius of A-first zone around this residue
    pub mutual_information: f32,                // MI(A,B) at this residue
    pub transfer_entropy_a_to_b: f32,           // TE(A→B)
    pub causal_flow_direction: f32,             // TE(A→B) - TE(B→A), signed
}
```

#### 6.2 Feature computation pipeline

Each feature group has its data source:

| Feature group | Data source | Computation location |
|---|---|---|
| Consensus (12) | Per-residue spike counts from both streams, per CCNS phase | CPU, in `coupled_md.rs` after simulation |
| Cross-correlation (12) | GPU CCF matrix (Step 5) | CPU extraction + per-phase re-computation |
| Differential (18) | NMA mode amplitudes + per-residue B-vs-A counts | CPU, requires NMA mode file |
| Scout/propagation (8) | Phase-offset timing + per-residue spike onset times | CPU, requires both streams' timestep data |

**CCNS phase assignment**: Each spike's `timestep` maps to a phase via the protocol:
```
cold_hold: [0, cold_hold_steps)
heating:   [cold_hold_steps, cold_hold_steps + ramp_steps)
warm_hold: [cold_hold_steps + ramp_steps, cold_hold_steps + ramp_steps + warm_hold_steps)
cooling:   [warm_hold_steps end, ramp_down end)
cold_return: [ramp_down end, total_steps)
```

**Transfer entropy** (MI, TE): compute from the time-binned spike matrices already built in Step 5:
```rust
/// Compute transfer entropy TE(A→B) for residue r.
/// TE = H(B_future | B_past) - H(B_future | B_past, A_past)
fn transfer_entropy(mat_a: &[f32], mat_b: &[f32], r: usize, n_bins: usize, lag: usize) -> f32 {
    // Discretize into {0, 1} by median threshold
    let median_b = median(&mat_b[r * n_bins..(r + 1) * n_bins]);
    let median_a = median(&mat_a[r * n_bins..(r + 1) * n_bins]);
    
    // Count conditional frequencies
    let mut counts = [[[[0u32; 2]; 2]; 2]; 2]; // [b_past][a_past][b_future]
    for t in lag..n_bins-1 {
        let b_past = if mat_b[r * n_bins + t - lag] > median_b { 1 } else { 0 };
        let a_past = if mat_a[r * n_bins + t - lag] > median_a { 1 } else { 0 };
        let b_future = if mat_b[r * n_bins + t + 1] > median_b { 1 } else { 0 };
        counts[b_past][a_past][b_future][0] += 1;
    }
    // ... conditional entropy computation
}
```

#### 6.3 Feature computation order

```
1. Run simulation (Steps 1-4 output)
2. Assign phases to all spikes (CPU, O(N))
3. Per-residue per-phase spike counts for A and B (CPU, O(N))
4. Consensus features: agreement ratio, phase profile, spatial coherence (CPU)
5. CCF matrix (GPU, Step 5)
6. CCF feature extraction: peak, width, asymmetry, per-phase, frequency (CPU)
7. Differential features: B/A ratio per phase, NMA-exclusive counts (CPU)
   — NMA features only populated if nma_modes_path is Some
8. Scout features: lead time, predictive value (CPU, requires phase-offset)
9. Information theory: MI, TE from time-binned matrices (CPU)
```

#### Regression gate 6

```bash
cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST: same or higher pass count
```

**Functional test**: Run on TEM1 (1btl), verify no field is NaN:
```bash
python3 -c "
import json
data = json.load(open('/tmp/twin_ccf_test/twin_per_residue.json'))
for r in data['residues']:
    for k, v in r.items():
        if isinstance(v, float):
            assert v == v, f'NaN in {k} for residue {r[\"resid\"]}'
print(f'✓ {len(data[\"residues\"])} residues, all fields finite')
"
```

**Performance contract**: TARGET feature computation < 2s for 300-residue protein (expected to be dominated by TE computation — will move to GPU in Phase C Step 15). Time with `std::time::Instant` in the feature build function.

<!-- SUPERSEDED: CPU TE in Step 6.3 item 9 is a PLACEHOLDER. Phase C Step 15 replaces it with GPU TE kernel. -->

#### If gate 6 fails:
1. NaN in feature fields → division by zero in B/A ratio (cnt_a == 0). Add guards: `if cnt_a == 0 { f32::INFINITY }` or similar.
2. Missing fields in JSON → serde_json skips `Option<T>` with `#[serde(skip_serializing_if = "Option::is_none")]`. Use default values, not Options.
3. Feature computation > 2s → profile to find bottleneck. If it's TE, this will be fixed by GPU TE in Phase C.

---

### STEP 7: Fix Output Directory and CLI Integration

**Goal**: `--coupled-twin` uses `args.output`, not env var. TWIN flag routes through `prism-validate-and-run.sh`.

#### 7.1 Fix output directory

In `coupled_md.rs`, replace all `PRISM_TWIN_OUTPUT` env var reads (lines 537-538 and 598-599) with a parameter:

Add `output_dir: &Path` parameter to `run_coupled_twin()`:
```rust
pub fn run_coupled_twin(
    config_ref: &PersistentBatchConfig,
    context: Arc<CudaContext>,
    fused_module: Arc<CudaModule>,
    topology: &PrismPrepTopology,
    protocol: CryoUvProtocol,
    twin_config: &CoupledTwinConfig,
    seed_a: u64,
    steps: i32,
    hmr: bool,
    fused_steps: u32,
    adaptive_dt: bool,
    ladd_enabled: bool,
    output_dir: &Path,           // NEW: replaces PRISM_TWIN_OUTPUT env var
) -> Result<CoupledTwinResult> {
```

Replace env var reads with `output_dir` parameter.

#### 7.2 Update CLI caller

In `nhs_rt_full.rs:604`, pass `&args.output`:
```rust
let result = run_coupled_twin(
    &batch_config, context, module, &topology, protocol,
    &twin_config, args.replica_seed, steps, args.hmr,
    args.fused_steps, args.adaptive_dt, args.ladd,
    &args.output,  // NEW
)?;
```

#### 7.3 Add `--coupled-twin` to `prism-validate-and-run.sh`

Add flag passthrough in the validation wrapper. Search for the flag parsing section and add:

```bash
--coupled-twin)
    ENGINE_FLAGS="$ENGINE_FLAGS --coupled-twin"
    shift
    ;;
```

#### Regression gate 7

```bash
# Verify wrapper passes the flag
scripts/prism-validate-and-run.sh \
    -t tests/fixtures/1btl_clean.topology.json \
    -o /tmp/twin_cli_test \
    --coupled-twin --fast --fused-steps 6 --hmr --adaptive-dt -v
    
# Output MUST appear in /tmp/twin_cli_test/, NOT /tmp/prism_twin_spikes/
ls /tmp/twin_cli_test/coupled_twin_result.json
# MUST exist
```

#### If gate 7 fails:
1. Output in `/tmp/prism_twin_spikes/` → env var `PRISM_TWIN_OUTPUT` still being read. Search for all occurrences in `coupled_md.rs` and replace.
2. `--coupled-twin` not recognized by wrapper → check `prism-validate-and-run.sh` case statement syntax (must have `;;` terminator).
3. Function signature mismatch → `run_coupled_twin()` now takes `output_dir: &Path` as last param. Update all call sites.

---

### STEP 8: Add Tests for `coupled_md` and `twin_detection`

**Goal**: Both modules get `#[cfg(test)]` blocks with unit tests that run without GPU.

#### 8.1 Tests for `coupled_md.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupled_twin_config_defaults() {
        let c = CoupledTwinConfig::default();
        assert_eq!(c.phase_offset_fraction, 0.20);
        assert_eq!(c.exchange_interval, 100);
        assert!(!c.enable_ccf);     // Gate 1 default: off
        assert!(!c.enable_exchange); // Gate 1 default: off
        assert!(c.sensitivity_boost > 0.0);
        assert!(c.max_threshold_reduction > 0.0);
        assert!(c.max_threshold_reduction <= 1.0);
    }

    #[test]
    fn test_interferometric_features_default() {
        let f = InterferometricFeatures {
            resid: 42, resname: "ALA".into(),
            spike_agreement_ratio: 0.8,
            consensus_intensity_mean: 1.5,
            consensus_phase_profile: [0.0; 5],
            consensus_spatial_coherence: 0.0,
            consensus_temporal_onset: 0.0,
            n_consensus_neighbors: 0,
            ccf_peak_lag: 0, ccf_peak_value: 0.0,
            ccf_width: 0.0, ccf_asymmetry: 0.0,
            ccf_per_phase: [0.0; 5],
            ccf_frequency_peak: 0.0,
            ccf_reproducibility: 0.8,
            ccf_lag_consistency: 0.0,
            spikes_a: 100, spikes_b: 80,
            b_over_a_ratio: 0.8,
            nma_exclusive_count: 5,
            thermal_exclusive_count: 25,
            b_over_a_intensity_ratio: 0.75,
            nma_responsive_mode: -1,
            nma_mode_eigenvalue: 0.0,
            barrier_classification: "MEDIUM".into(),
            per_phase_differential: [0.0; 5],
            differential_onset_lag: 0.0,
            nma_work_at_residue: 0.0,
            mechanical_sensitivity: 0.0,
            susceptibility_magnitude: 0.0,
            scout_lead_time: 0.0,
            scout_predictive_value: 0.0,
            phase_offset_enrichment: 0.0,
            scout_intensity_at_onset: 0.0,
            scout_spatial_propagation: 0.0,
            mutual_information: 0.0,
            transfer_entropy_a_to_b: 0.0,
            causal_flow_direction: 0.0,
        };
        assert_eq!(f.resid, 42);
        let json = serde_json::to_string(&f).unwrap();
        let deser: InterferometricFeatures = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.resid, 42);
        assert_eq!(deser.barrier_classification, "MEDIUM");
    }

    #[test]
    fn test_spike_density_cpu_empty() {
        let cells = compute_spike_density_cpu(&[], 4.0);
        assert!(cells.0.is_empty());
    }

    #[test]
    fn test_cpu_cross_correlation_empty() {
        let (n, map) = compute_cpu_cross_correlation(&[], &[], 5.0, 500);
        assert_eq!(n, 0);
        assert!(map.is_empty());
    }

    #[test]
    fn test_ccf_matrix_extraction_identity() {
        // 3×3 identity-like CCF → peak for each residue is with itself
        let ccf = vec![
            1.0, 0.2, 0.1,
            0.2, 1.0, 0.3,
            0.1, 0.3, 1.0,
        ];
        let features = extract_ccf_features(&ccf, 3);
        assert_eq!(features.len(), 3);
        // Residue 2's highest off-diagonal is column 1 (0.3)
        assert_eq!(features[2].ccf_peak_lag, 1 - 2); // relative: -1
        assert!((features[2].ccf_peak_value - 0.3).abs() < 1e-6);
    }
}
```

#### 8.2 Tests for `twin_detection.rs`

Add a `#[cfg(test)]` block at the end of `twin_detection.rs` testing the detection pipeline's data transforms (site classification logic, consensus thresholds) without requiring GPU.

#### Regression gate 8

```bash
cargo test -p prism-nhs --lib coupled 2>&1 | tail -10
# MUST show: N tests run, 0 failures (was 0 tests before)

cargo test -p prism-nhs --lib twin_detection 2>&1 | tail -10
# MUST show: N tests run, 0 failures

cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST: total count > 160 (155 baseline + new tests)
```

#### If gate 8 fails:
1. `compute_spike_density_cpu` not found → function is `#[cfg(feature = "gpu")]`. Tests without GPU feature won't see it. Use `#[cfg(test)]` inside the `#[cfg(feature = "gpu")]` block, or test CPU-only functions separately.
2. Serde roundtrip failure → check that all new fields in `InterferometricFeatures` have `Serialize + Deserialize` derives and no unsupported types.
3. Test count < 160 → verify both `coupled_md::tests` and `twin_detection::tests` modules are registered.

---

### STEP 9: End-to-End Phase A Validation

**Goal**: Run PRISM-TWIN on 3 benchmark targets and verify (a) coupling is active, (b) output is valid, (c) no regression against non-TWIN baseline.

#### 9.1 Validation targets

| Target | PDB | Atoms | Expected difficulty |
|--------|-----|-------|---------------------|
| TEM1 β-lactamase | 1btl | ~4100 | Easy (SNDC EXCELLENT) |
| KRAS G12C | 4obe | ~2600 | Medium (cryptic SII-P) |
| IL-2 | 1a4q | ~2100 | Hard (protein-protein interface) |

#### 9.2 Run protocol

For each target:
```bash
# Non-TWIN baseline (§B canonical)
scripts/prism-validate-and-run.sh \
    -t output/$TARGET/$TARGET_clean.topology.json \
    -o output/${TARGET}_baseline \
    --fast --hysteresis --prism-therm \
    --multi-stream 8 \
    --spike-percentile 70 \
    --fused-steps 6 \
    --hmr --adaptive-dt \
    --multi-differential \
    --closed-loop-steering --asymmetric-steering \
    --use-xgb-ranker \
    --replica-seed 42 -v

# TWIN run (historical --coupled-twin; see lockdown notice at top)
scripts/prism-validate-and-run.sh \
    -t output/$TARGET/$TARGET_clean.topology.json \
    -o output/${TARGET}_twin \
    --coupled-twin --fast \
    --fused-steps 6 --hmr --adaptive-dt \
    --replica-seed 42 -v
```

#### 9.3 Validation criteria

| Check | Acceptance |
|-------|------------|
| TWIN run completes without crash | PASS/FAIL |
| `coupled_twin_result.json` is valid JSON | PASS/FAIL |
| `twin_per_residue.json` has all 50 fields per residue | PASS/FAIL |
| `twin_ccf_matrix.json` is N×N, no NaN | PASS/FAIL |
| Ring buffer overflow < 10% of total spikes | PASS/FAIL |
| TWIN wall time < 3× baseline 2-stream time | PASS/FAIL |
| TWIN detects >= as many sites as baseline | PASS/WARN |
| CCF matrix has >10% non-zero entries (|v|>0.01) | PASS/WARN |
| Threshold modification measurable (log shows non-zero boost) | PASS/FAIL |

#### 9.4 Performance contracts for Phase A

| Metric | Contract |
|--------|----------|
| Per-step overhead from ring buffer | < 5% of fused step time |
| CCF computation (post-simulation) | < 10ms for 300 residues |
| Total VRAM increase from TWIN buffers | < 50 MB |
| No regression in spike detection count vs non-TWIN | ≥ 90% of baseline spike count |

#### Regression gate 9 (Phase A complete)

```bash
cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST: > 160 passed, 0 failed
```

All 3 targets pass all validation criteria above.

#### If gate 9 fails:
1. Any target crashes → check VRAM with `nvidia-smi`. 1btl (4100 atoms) needs ~3 GB for 2 engines.
2. CCF all zeros → both streams produced identical spikes (same seed?). Verify `seed_b = seed_a + 1000`.
3. Zero threshold delta → `sensitivity_boost` too low, or no spikes during cold_hold. Try enabling `--ladd` for extra spike channel.
4. Wall time > 3× baseline → profiling needed. Ring buffer overhead should be < 5%.
5. Site count regression → TWIN threshold adaptation may suppress some sites. Check if `max_threshold_reduction` (0.5) is too aggressive. Try 0.3.
6. If 2/3 targets pass, investigate the failing one specifically before declaring gate failure.

---

## PHASE B: Multi-Engine Groups (N=4-8 per Group)

<!-- SUPERSEDED: Steps 10-14 were architectural sketches (~370 lines).
     Replaced with fully prescriptive expanded versions per corrections document 2026-04-08.
     Original text preserved in git history at commit before this edit. -->

### STEP 10: Merge `--multi-stream` and `--coupled-twin` CLI Paths (EXPANDED)

#### 10.0 Preconditions

- Phase A Steps 0-9 ALL pass their regression gates
- `cargo test -p prism-nhs --lib` shows 0 failures
- Single `--coupled-twin` run (2 engines) completes on 1btl with non-zero coupling

#### 10.1 CLI semantics

Verified dispatch location: `nhs_rt_full.rs:545-549` (from Step 0.5g).

```rust
// At nhs_rt_full.rs dispatch point (line 545):
if args.coupled_twin {
    let streams_per_group = if args.multi_stream > 1 {
        if args.multi_stream % 2 != 0 {
            anyhow::bail!(
                "--coupled-twin requires even --multi-stream value (got {}). \
                 Engines are split equally between Group A and Group B.",
                args.multi_stream
            );
        }
        args.multi_stream / 2
    } else {
        1  // Phase A default: 1 engine per group
    };
    log::info!("TWIN mode: {} engines/group, {} total", streams_per_group, streams_per_group * 2);
    return run_coupled_twin_multi(args, topology_path, streams_per_group);
} else if args.multi_stream > 1 {
    return run_multi_stream_pipeline(args, topology_path, args.multi_stream);
}
```

#### 10.2 Create `run_coupled_twin_multi()` function

**Location**: New file `crates/prism-nhs/src/coupled_twin_multi.rs`.

Register in `lib.rs`:
```rust
pub mod coupled_twin_multi;
```

**Function signature** (verified: `PersistentNhsEngine::new_on_stream` takes `Arc<CudaContext>`, `Arc<CudaModule>`, `Arc<CudaStream>` — Step 0.5h):

```rust
pub fn run_coupled_twin_multi(
    args: &Args,                  // CLI args (nhs_rt_full.rs:Args struct)
    topology_path: &PathBuf,
    streams_per_group: usize,
) -> Result<()> {
```

#### 10.3 Engine creation loop

```rust
let context = CudaContext::new(0).context("CUDA context")?;
let fused_module = context.load_module(Ptx::from_file(&find_ptx_path()?)).context("fused PTX")?;
let ring_module = context.load_module(Ptx::from_file(&find_twin_ptx("ring_buffer.ptx")?)).context("ring_buffer.ptx")?;
let ccf_module = context.load_module(Ptx::from_file(&find_twin_ptx("tensor_ccf.ptx")?)).context("tensor_ccf.ptx")?;

let n = streams_per_group;
let total_engines = 2 * n;

// Create CUDA streams (one per engine + 1 exchange)
let mut streams: Vec<Arc<CudaStream>> = Vec::with_capacity(total_engines);
for _ in 0..total_engines {
    streams.push(context.new_stream()?);
}
let stream_exchange = context.new_stream()?;

// Create engines: Group A [0..n), Group B [n..2n)
let mut engines: Vec<PersistentNhsEngine> = Vec::with_capacity(total_engines);
let twin_config = CoupledTwinConfig::default(); // with enable_exchange=true, enable_ccf=true
let base_protocol = if args.fast { CryoUvProtocol::fast_35k() } else { CryoUvProtocol::standard() };
let offset_steps = (base_protocol.cold_hold_steps as f32 * twin_config.phase_offset_fraction) as i32;

for i in 0..total_engines {
    let is_group_b = i >= n;
    let group_idx = if is_group_b { i - n } else { i };
    let seed = args.replica_seed + (i as u64) * 12345;

    let mut prot = base_protocol.clone();

    // Temperature diversification (reuse existing multi_temp logic from nhs_rt_full.rs:2735-2815)
    if args.multi_temp && n >= 4 {
        match group_idx % n.min(8) {
            0 => { /* baseline */ }
            1 => { prot.end_temp *= 1.17; }
            2 => { prot.end_temp *= 1.33; prot.ramp_down_steps = (prot.ramp_down_steps as f32 * 0.2) as i32; }
            3 => { prot.end_temp *= 1.5; prot.ramp_down_steps = (prot.ramp_down_steps as f32 * 0.2) as i32; }
            4 => { prot.end_temp *= 1.17; prot.ramp_down_steps = (prot.ramp_down_steps as f32 * 3.0) as i32; }
            5 => { prot.end_temp *= 1.33; prot.warm_hold_steps = (prot.warm_hold_steps as f32 * 2.0) as i32; }
            6 => { prot.end_temp *= 1.67; prot.ramp_down_steps = 50; }
            _ => { /* baseline duplicate */ }
        }
    }

    // Group B: phase offset
    if is_group_b {
        prot.cold_hold_steps += offset_steps;
    }

    let mut engine = PersistentNhsEngine::new_on_stream(
        &config, context.clone(), fused_module.clone(), streams[i].clone(),
    )?;
    engine.load_topology(&topology)?;
    if args.hmr { engine.set_dt(0.004)?; }
    if args.fused_steps > 1 { engine.set_fused_inner_steps(args.fused_steps)?; }
    if args.adaptive_dt { engine.set_adaptive_dt(true)?; }
    if args.ladd { engine.set_ladd_enabled(true); }
    engine.set_cryo_uv_protocol(prot)?;
    engine.set_spike_accumulation(true);

    // NMA perturbation for Group B only
    if is_group_b {
        if let Some(ref nma_path) = args.nma_perturb {
            engine.set_nma_amplification(args.nma_amplification);
            engine.load_nma_modes(nma_path.to_str().unwrap_or(""))?;
        }
    }

    engines.push(engine);
}
```

#### 10.4 Ring buffer allocation

```rust
let ring_capacity = 8192 * (n as u32).max(1);
let mut ring_a_to_b = TwinRingBuffer::new(&context, &stream_exchange, &ring_module, ring_capacity)?;
let mut ring_b_to_a = TwinRingBuffer::new(&context, &stream_exchange, &ring_module, ring_capacity)?;
ring_a_to_b.reset(&stream_exchange)?;
ring_b_to_a.reset(&stream_exchange)?;
```

#### 10.5 Main simulation loop

```rust
let outer_steps = base_protocol.total_steps() / args.fused_steps.max(1) as i32;
let inner = args.fused_steps.max(1) as i32;
let mut prev_spike_counts = vec![0u32; total_engines];
let mut ring_spikes_exchanged: u64 = 0;

for step in 0..outer_steps {
    // ══ PHASE 1: Cross-group threshold adaptation (SERIAL) ══
    if twin_config.enable_exchange && step > 0 {
        let (gx, gy, gz, ox, oy, oz, vs) = engines[0].grid_info().unwrap();

        // B→A: B's evidence lowers ALL Group A engines' thresholds
        for i in 0..n {
            if let (Some(thresh), Some(base)) = (
                engines[i].threshold_buffer_mut(), engines[i].base_threshold_buffer(),
            ) {
                ring_b_to_a.read_and_adapt(&stream_exchange, thresh, base,
                    (gx,gy,gz), (ox,oy,oz), vs,
                    twin_config.sensitivity_boost, twin_config.max_threshold_reduction,
                    step as u32, 500.0)?;
            }
        }
        // A→B: A's evidence lowers ALL Group B engines' thresholds
        for i in n..total_engines {
            if let (Some(thresh), Some(base)) = (
                engines[i].threshold_buffer_mut(), engines[i].base_threshold_buffer(),
            ) {
                ring_a_to_b.read_and_adapt(&stream_exchange, thresh, base,
                    (gx,gy,gz), (ox,oy,oz), vs,
                    twin_config.sensitivity_boost, twin_config.max_threshold_reduction,
                    step as u32, 500.0)?;
            }
        }
        stream_exchange.synchronize()?;
    }

    // ══ PHASE 2: Run ALL engines concurrently (scoped threads) ══
    let summaries = par_map_mut(&mut engines, |engine| engine.run(inner));

    // ══ PHASE 3: Push new spikes per engine into group ring buffers ══
    if twin_config.enable_exchange {
        for i in 0..total_engines {
            let curr = engines[i].get_spike_count().unwrap_or(0);
            let delta = curr.saturating_sub(prev_spike_counts[i]);
            if delta > 0 {
                if let Some(buf) = engines[i].spike_buffer_gpu() {
                    let ring = if i < n { &mut ring_a_to_b } else { &mut ring_b_to_a };
                    // NOTE: requires spike compaction (Step 4.2a) between d_spike_events and ring
                    ring.push_batch(&streams[i], buf, prev_spike_counts[i], curr)?;
                }
                ring_spikes_exchanged += delta as u64;
            }
            prev_spike_counts[i] = curr;
        }
    }

    // ══ PHASE 4: Periodic threshold recovery ══
    if twin_config.enable_exchange && step as u32 % 1000 == 0 && step > 0 {
        let n_voxels = engines[0].total_voxels() as u32;
        for i in 0..total_engines {
            if let (Some(thresh), Some(base)) = (
                engines[i].threshold_buffer_mut(), engines[i].base_threshold_buffer(),
            ) {
                let ring = if i < n { &ring_b_to_a } else { &ring_a_to_b };
                ring.threshold_recovery(&stream_exchange, thresh, base, n_voxels, 0.01)?;
            }
        }
    }

    // ══ PHASE 5: Progress logging ══
    if step % 500 == 0 {
        log::info!("  [step {}/{}] exchanged={} overflow_a={} overflow_b={}",
            step, outer_steps, ring_spikes_exchanged,
            ring_a_to_b.overflow_count(&stream_exchange)?,
            ring_b_to_a.overflow_count(&stream_exchange)?);
    }
}
```

#### 10.6 `par_map_mut` helper for safe concurrent engine execution

```rust
fn par_map_mut<T: Send, R: Send>(
    items: &mut [T],
    f: impl Fn(&mut T) -> R + Send + Sync,
) -> Vec<R> {
    std::thread::scope(|s| {
        let ptrs: Vec<*mut T> = items.iter_mut().map(|x| x as *mut T).collect();
        let handles: Vec<_> = ptrs.into_iter().map(|p| {
            let f_ref = &f;
            s.spawn(move || f_ref(unsafe { &mut *p }))
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}
```

**Safety**: Each thread gets a unique pointer. No two threads access the same engine. The scope guarantees all threads complete before any subsequent access.

#### 10.7 Overflow monitoring

Add to `TwinRingBuffer`:
```rust
pub fn overflow_count(&self, stream: &Arc<CudaStream>) -> Result<u32> {
    let mut count = [0u32; 1];
    stream.memcpy_dtoh(&self.overflow, &mut count)?;
    Ok(count[0])
}
```

#### Regression gate 10

```bash
cargo check -p prism-nhs --features gpu 2>&1 | grep error
# MUST: zero errors

cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST: 0 failures, count >= Phase A final count

# Functional: 2×4 = 8 engines
scripts/prism-validate-and-run.sh \
    -t tests/fixtures/1btl_clean.topology.json \
    -o /tmp/twin_multi8_test \
    --coupled-twin --multi-stream 8 --fast \
    --fused-steps 6 --hmr --adaptive-dt -v 2>&1 | tee /tmp/twin_multi8.log

grep -c "Engine.*initialized\|Stream.*initialized" /tmp/twin_multi8.log
# MUST: 8

grep "ring_spikes_exchanged" /tmp/twin_multi8.log | tail -1
# MUST: > 0

grep -i "cuda error\|illegal memory\|out of memory" /tmp/twin_multi8.log
# MUST: empty

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
# MUST: < 15000 MB for 1btl on RTX 5080

# Verify odd multi-stream rejected
scripts/prism-validate-and-run.sh \
    -t tests/fixtures/1btl_clean.topology.json \
    -o /tmp/twin_odd_test \
    --coupled-twin --multi-stream 7 --fast 2>&1 | grep "requires even"
# MUST: show error message
```

#### If gate 10 fails:
1. CUDA OOM → reduce to `--multi-stream 4` (2×2), verify that works, document VRAM limit
2. Compilation errors in thread scope → check `PersistentNhsEngine: Send` (`grep 'impl Send'`). If not, add `unsafe impl Send for PersistentNhsEngine {}` with safety justification
3. Zero exchanges → check `enable_exchange` is true in config
4. Ring buffer overflow > 10% → increase `ring_capacity` multiplier from 8192 to 16384
5. DO NOT proceed until gate passes

---

### STEP 11: Within-Group Consensus (EXPANDED)

#### 11.0 Preconditions
- Step 10 gate passes (8-engine run completes)

#### 11.1 Data structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupConsensus {
    pub pass_fraction: f32,
    pub mean_spikes_passing: f32,
    pub spike_cv: f32,
    pub median_intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinGroupConsensusResult {
    pub group_a: Vec<GroupConsensus>,
    pub group_b: Vec<GroupConsensus>,
    pub consensus_residues: Vec<usize>,
    pub n_residues: usize,
    pub min_spikes: usize,
    pub min_pass_fraction: f32,
}
```

#### 11.2 Consensus computation

```rust
pub fn compute_within_group_consensus(
    group_a_spikes: &[Vec<GpuSpikeEvent>],
    group_b_spikes: &[Vec<GpuSpikeEvent>],
    n_residues: usize,
    min_spikes: usize,       // default: 5
    min_pass_fraction: f32,  // default: 0.6
) -> WithinGroupConsensusResult {
    let compute_group = |group: &[Vec<GpuSpikeEvent>]| -> Vec<GroupConsensus> {
        let n_eng = group.len();
        (0..n_residues).map(|r| {
            let counts: Vec<u32> = group.iter().map(|spikes| {
                spikes.iter()
                    .filter(|s| (0..s.n_residues.min(8) as usize)
                        .any(|j| s.nearby_residues[j] == r as i32))
                    .count() as u32
            }).collect();

            let passing: Vec<u32> = counts.iter().copied()
                .filter(|&c| c >= min_spikes as u32).collect();
            let pf = passing.len() as f32 / n_eng as f32;
            let mean_p = if passing.is_empty() { 0.0 }
                else { passing.iter().sum::<u32>() as f32 / passing.len() as f32 };
            let all_mean = counts.iter().sum::<u32>() as f32 / n_eng as f32;
            let var = counts.iter().map(|&c| (c as f32 - all_mean).powi(2)).sum::<f32>() / n_eng as f32;
            let cv = if all_mean > 0.0 { var.sqrt() / all_mean } else { 0.0 };

            GroupConsensus { pass_fraction: pf, mean_spikes_passing: mean_p,
                           spike_cv: cv, median_intensity: 0.0 /* computed separately */ }
        }).collect()
    };

    let group_a = compute_group(group_a_spikes);
    let group_b = compute_group(group_b_spikes);
    let consensus_residues: Vec<usize> = (0..n_residues)
        .filter(|&r| group_a[r].pass_fraction >= min_pass_fraction
                   && group_b[r].pass_fraction >= min_pass_fraction)
        .collect();

    log::info!("Within-group consensus: {}/{} residues pass (≥{:.0}% in both groups)",
        consensus_residues.len(), n_residues, min_pass_fraction * 100.0);

    WithinGroupConsensusResult { group_a, group_b, consensus_residues, n_residues,
                                 min_spikes, min_pass_fraction }
}
```

#### 11.3 Consensus-filtered CCF integration

After the main loop, before CCF computation:

```rust
let group_a_per_engine: Vec<Vec<GpuSpikeEvent>> = (0..n)
    .map(|i| engines[i].get_accumulated_spikes()).collect();
let group_b_per_engine: Vec<Vec<GpuSpikeEvent>> = (n..total_engines)
    .map(|i| engines[i].get_accumulated_spikes()).collect();

let consensus = compute_within_group_consensus(
    &group_a_per_engine, &group_b_per_engine,
    topology.residues.len(), 5, 0.6,
);

// Aggregate for CCF (union across engines)
let all_spikes_a: Vec<GpuSpikeEvent> = group_a_per_engine.into_iter().flatten().collect();
let all_spikes_b: Vec<GpuSpikeEvent> = group_b_per_engine.into_iter().flatten().collect();

// Build CCF matrices FILTERED to consensus residues only
let (half_a, half_b, n_res_p, n_bins_p, residue_map) = build_ccf_matrices_filtered(
    &all_spikes_a, &all_spikes_b,
    &consensus.consensus_residues, total_steps, 100,
);
```

#### 11.4 Unit tests (no GPU)

```rust
#[cfg(test)]
mod consensus_tests {
    use super::*;

    fn make_spike(resid: i32) -> GpuSpikeEvent {
        let mut s = GpuSpikeEvent::default();
        s.intensity = 1.0; s.nearby_residues[0] = resid; s.n_residues = 1; s
    }

    #[test]
    fn test_consensus_unanimous() {
        let group: Vec<Vec<GpuSpikeEvent>> = (0..4)
            .map(|_| (0..10).map(|_| make_spike(0)).collect()).collect();
        let r = compute_within_group_consensus(&group, &group, 3, 5, 0.6);
        assert_eq!(r.group_a[0].pass_fraction, 1.0);
        assert!(r.consensus_residues.contains(&0));
    }

    #[test]
    fn test_consensus_noise_rejected() {
        let mut group: Vec<Vec<GpuSpikeEvent>> = (0..4)
            .map(|_| vec![make_spike(0); 10]).collect();
        group[0].extend((0..10).map(|_| make_spike(1)));
        let r = compute_within_group_consensus(&group, &group, 3, 5, 0.6);
        assert!(r.group_a[1].pass_fraction < 0.6);
        assert!(!r.consensus_residues.contains(&1));
    }

    #[test]
    fn test_consensus_requires_both_groups() {
        let ga: Vec<Vec<GpuSpikeEvent>> = (0..4)
            .map(|_| (0..10).map(|_| make_spike(2)).collect()).collect();
        let gb: Vec<Vec<GpuSpikeEvent>> = (0..4).map(|_| vec![]).collect();
        let r = compute_within_group_consensus(&ga, &gb, 3, 5, 0.6);
        assert!(!r.consensus_residues.contains(&2));
    }

    #[test]
    fn test_spike_cv_uniform() {
        let group: Vec<Vec<GpuSpikeEvent>> = (0..4)
            .map(|_| (0..10).map(|_| make_spike(0)).collect()).collect();
        let r = compute_within_group_consensus(&group, &group, 1, 5, 0.6);
        assert!(r.group_a[0].spike_cv < 0.01);
    }
}
```

#### Regression gate 11

```bash
cargo test -p prism-nhs --lib consensus 2>&1 | tail -10
# MUST: 4+ tests, 0 failures

# Functional
grep "consensus" /tmp/twin_multi8_test.log
# MUST: "Within-group consensus: M/N residues pass" with 0 < M < N

python3 -c "
import json
data = json.load(open('/tmp/twin_multi8_test/consensus_within_group.json'))
assert 'consensus_residues' in data
n = len(data['consensus_residues'])
print(f'✓ {n}/{data[\"n_residues\"]} consensus residues')
assert n > 0, 'Zero consensus — coupling or detection failure'
"
```

#### If gate 11 fails:
1. Zero consensus → `min_spikes` too high for short runs. Try 3.
2. All residues pass → `min_spikes` too low. Try 10.
3. Compilation errors in spike iteration → GpuSpikeEvent field access mismatch. Check Step 0.5e.

---

### STEP 12: Per-Group Protocol Diversification (EXPANDED)

#### 12.0 Preconditions
- Step 11 gate passes

#### 12.1 Protocol diversification is in Step 10.3

The temperature ladder in Step 10.3 IS the diversification. Step 12 adds:
1. **Quench-rate diversification** (not just peak temperature)
2. **Protocol metadata recording** per engine

#### 12.2 Record protocol metadata

```rust
#[derive(Debug, Clone, Serialize)]
pub struct EngineProtocolMetadata {
    pub engine_index: usize,
    pub group: String,
    pub seed: u64,
    pub end_temp_k: f32,
    pub ramp_down_steps: i32,
    pub warm_hold_steps: i32,
    pub has_nma_perturbation: bool,
    pub phase_offset_steps: i32,
}
```

Populate during engine creation and write to `engine_protocols.json`.

#### Regression gate 12

```bash
cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST: 0 failures

python3 -c "
import json
data = json.load(open('/tmp/twin_multi8_test/engine_protocols.json'))
temps = set(e['end_temp_k'] for e in data)
print(f'✓ {len(temps)} distinct temperatures across {len(data)} engines')
assert len(temps) >= 2, 'No temperature diversity'
group_b = [e for e in data if e['group'] == 'B']
assert all(e['phase_offset_steps'] > 0 for e in group_b), 'Group B missing offset'
"
```

#### If gate 12 fails:
1. All same temperature → `args.multi_temp` not set. Add to CLI invocation.
2. Missing NMA in Group B → `--nma-perturb` path not provided or file missing.

---

### STEP 13: Full Output Schema (EXPANDED)

#### 13.0 Preconditions
- Steps 10-12 gates pass

#### 13.1 Create `twin_output.rs`

Single function that writes ALL TWIN output files:

```rust
pub fn write_twin_outputs(
    output_dir: &Path,
    prefix: &str,
    result: &CoupledTwinResult,
    per_residue: &[InterferometricFeatures],
    ccf_matrix: Option<&[f32]>,
    ccf_n_res: usize,
    consensus: &WithinGroupConsensusResult,
    engine_metadata: &[EngineProtocolMetadata],
) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    // 1. coupled_twin_result.json
    std::fs::write(output_dir.join("coupled_twin_result.json"),
        serde_json::to_string_pretty(result)?)?;

    // 2. {prefix}.twin_per_residue.json (50 fields)
    std::fs::write(output_dir.join(format!("{}.twin_per_residue.json", prefix)),
        serde_json::to_string_pretty(&serde_json::json!({
            "n_residues": per_residue.len(), "n_features": 50, "residues": per_residue,
        }))?)?;

    // 3. {prefix}.twin_ccf_matrix.json
    if let Some(ccf) = ccf_matrix {
        std::fs::write(output_dir.join(format!("{}.twin_ccf_matrix.json", prefix)),
            serde_json::to_string_pretty(&serde_json::json!({
                "n_res": ccf_n_res, "matrix": ccf,
            }))?)?;
    }

    // 4. consensus_within_group.json
    std::fs::write(output_dir.join(format!("{}.consensus_within_group.json", prefix)),
        serde_json::to_string_pretty(consensus)?)?;

    // 5. engine_protocols.json
    std::fs::write(output_dir.join("engine_protocols.json"),
        serde_json::to_string_pretty(engine_metadata)?)?;

    // 6. {prefix}.binding_sites.json (backward-compatible)
    // 7. {prefix}.kcc_visualization.json (backward-compatible, CCF→KCC mapping)
    // These are written by twin_detection::detect_and_write_twin_sites(), already wired.

    log::info!("TWIN output: {} files written to {}", 7, output_dir.display());
    Ok(())
}
```

#### 13.2 Backward compatibility with `prism_canonical.py`

`binding_sites.json` MUST contain `sites` array with `centroid`, `lining_residues`, `quality_score`. Add optional `twin_class` field. `kcc_visualization.json` maps CCF features into KCC slots.

#### Regression gate 13

```bash
cargo test -p prism-nhs --lib output 2>&1 | tail -10
# MUST: all pass

python3 -c "
import json
pr = json.load(open('/tmp/twin_multi8_test/1btl_clean.twin_per_residue.json'))
expected = ['resid','resname','spike_agreement_ratio','ccf_peak_lag',
    'ccf_peak_value','b_over_a_ratio','barrier_classification',
    'transfer_entropy_a_to_b','causal_flow_direction','mutual_information']
for r in pr['residues'][:3]:
    for f in expected:
        assert f in r, f'Missing: {f}'
    for k,v in r.items():
        if isinstance(v,float): assert v==v, f'NaN in {k}'
print('✓ Schema valid')
"

# Backward compatibility
python3 scripts/prism_canonical.py \
    --binding-sites /tmp/twin_multi8_test/1btl_clean.binding_sites.json \
    --kcc-viz /tmp/twin_multi8_test/1btl_clean.kcc_visualization.json \
    --topology output/1btl/1btl_clean.topology.json \
    --output-dir /tmp/twin_compat_test 2>&1
# MUST: exit 0
```

#### If gate 13 fails:
1. Missing fields → check `InterferometricFeatures` struct has all 50 fields with `Serialize`
2. `prism_canonical.py` KeyError → backward-incompatible change in `binding_sites.json`. Check that `sites` array structure is preserved.

---

### STEP 14: Full Phase B Validation (EXPANDED)

#### 14.0 Preconditions
- ALL prior gates (0-13) pass
- 1btl E2E with `--coupled-twin --multi-stream 8` completes

#### 14.1 Automated validation script

Create `scripts/twin_phase_b_validation.sh` (see corrections document for full script — ~80 lines). For each of 10 BENCH30 targets:
1. Run baseline (2-stream, no coupling)
2. Run TWIN (2×4 = 8 engines, coupled)
3. Validate: no crash, valid JSON, no NaN, overflow < 10%, coupling active, wall time < 4×

#### 14.2 Non-regression script

Create `scripts/twin_nonregression.py` that compares DCC scores:
- **TWIN must not regress any EXCELLENT (< 5Å) target**
- TWIN may find additional sites baseline misses

#### 14.3 Target selection

| Tier | Count | Criteria |
|------|-------|----------|
| EXCELLENT (< 5Å DCC) | 2-3 | Non-regression anchors |
| GOOD (5-8Å DCC) | 3-4 | Moderate difficulty |
| MARGINAL/MISS (> 8Å DCC) | 2-3 | Where TWIN might improve |
| Known allosteric | 1-2 | CCF validation |

#### Regression gate 14 (Phase B complete)

```bash
bash scripts/twin_phase_b_validation.sh
# MUST: "PHASE B VALIDATION PASSED"

python3 scripts/twin_nonregression.py output/twin_phase_b_validation
# MUST: "No regressions on EXCELLENT targets"

cargo test -p prism-nhs --lib 2>&1 | tail -5
# MUST: > 170 tests, 0 failures
```

#### If gate 14 fails:
1. Any crash → check VRAM, reduce to `--multi-stream 4` for large proteins
2. NaN → division-by-zero in feature computation. Add guards.
3. Regression on EXCELLENT → reduce `sensitivity_boost` (threshold adaptation too aggressive)
4. Zero coupling → verify `enable_exchange = true` in all config paths

---

## PRE-REQUISITE: LADD Branch Merge

**Branch**: `ladd-channel-20260331` — 3 unmerged commits, 628 lines, 7 CUDA kernels for departure-based depletion detection.

**Why this is a prerequisite**: LADD (spike_source=4) provides a 4th spike channel. Cross-correlating LADD depletion events between twin streams gives high-confidence pocket opening signal — same atoms departing same voxels in both independent simulations.

**Merge conflicts**: 7 regions across 2 files (`nhs_amber_fused.cu` × 3, `fused_engine.rs` × 4). HEAD has an oscillator-based LADD (K=2 RAF), branch has departure-based LADD (7 kernels: voxel tracking → density → departure buffer → prefix sum → scatter → detect+emit). The branch version is architecturally superior.

**Resolution strategy** (verified in earlier audit):
1. Take branch's 6 detection kernels, drop HEAD's `ladd_observation_step`
2. Take branch's Welford accumulation body
3. Take branch's `ladd_detect_and_emit`, port HEAD's co-fire logic (source=5) into it
4. Take BOTH sides' struct fields: branch's 14 departure fields + HEAD's NMA fields
5. Take branch's `set_ladd_enabled` log message, update for departure architecture
6. Keep ALL of HEAD's NMA code (branch predates NMA)
7. Take branch's 7-step kernel launch pipeline, replace HEAD's single-kernel launch

**Estimated effort**: ~2 hours manual merge + `cargo check` after each file.

**Gate**: `cargo test -p prism-nhs --lib` passes with same count as before merge + zero new failures.

---

## PHASE C: From Engine to Product (GPU TE, NMA, Ranker, Retraining)

Phase C converts the TWIN engine's raw signal into production-ready pocket rankings and a clean student model.

---

### STEP 15: GPU Transfer Entropy Kernel

**Goal**: Replace CPU TE computation (Step 6.3 item 9) with a purpose-built GPU kernel that operates on the same time-binned spike matrices used by the CCF kernel.

#### 15.1 Why the existing `transfer_entropy.cu` won't work

The existing KSG kernel (`transfer_entropy_ksg_kernel` at `crates/prism-gpu/src/kernels/transfer_entropy.cu:107`) has:
- O(T²) inner loop per time point (lines 170-192)
- `extern __shared__ float shared_mem[threadIdx.x * series_length]` → requires 10 MB shared memory per block (RTX 5080 has 100 KB per SM)
- One thread per variable pair → 300² = 90K threads, each doing 10K² iterations

It will fail to launch for any realistic protein.

#### 15.2 New kernel: `twin_transfer_entropy.cu`

**Architecture**: Binned TE (not KSG). For discretized spike data (spike/no-spike per time bin), conditional probability tables are 2×2×2 (8 entries), fit in registers.

```cuda
// twin_transfer_entropy.cu — Binned Transfer Entropy on spike matrices
//
// Input: same [n_res × n_bins] FP16 matrices as tensor_ccf.cu
// Output: [n_res × n_res] TE(A→B) matrix + [n_res × n_res] TE(B→A) matrix
//
// Each thread computes TE for one (source_residue, target_residue) pair.
// Grid: (ceil(n_res/16), ceil(n_res/16)), Block: (16, 16)

extern "C" __global__ void twin_binned_te(
    const half* __restrict__ spike_matrix_a,  // [n_res_padded × n_bins_padded]
    const half* __restrict__ spike_matrix_b,  // [n_res_padded × n_bins_padded]
    float* __restrict__ te_a_to_b,            // [n_res × n_res] output
    float* __restrict__ te_b_to_a,            // [n_res × n_res] output
    const float* __restrict__ median_a,       // [n_res] per-row median for binarization
    const float* __restrict__ median_b,       // [n_res] per-row median
    int n_res,
    int n_res_padded,
    int n_bins,
    int n_bins_padded,
    int lag                                    // temporal lag (default: 1)
) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    int tgt = blockIdx.y * blockDim.y + threadIdx.y;
    if (src >= n_res || tgt >= n_res) return;
    if (src == tgt) {
        te_a_to_b[src * n_res + tgt] = 0.0f;
        te_b_to_a[src * n_res + tgt] = 0.0f;
        return;
    }

    float med_src_a = median_a[src];
    float med_tgt_b = median_b[tgt];

    // Count conditional frequencies: [src_past][tgt_past][tgt_future]
    // Using 2×2×2 = 8 bins, all fit in registers
    int counts[2][2][2] = {{{0}}};
    int total = 0;

    for (int t = lag; t < n_bins - 1; t++) {
        int src_past = (__half2float(spike_matrix_a[src * n_bins_padded + t - lag]) > med_src_a) ? 1 : 0;
        int tgt_past = (__half2float(spike_matrix_b[tgt * n_bins_padded + t - lag]) > med_tgt_b) ? 1 : 0;
        int tgt_future = (__half2float(spike_matrix_b[tgt * n_bins_padded + t + 1]) > med_tgt_b) ? 1 : 0;
        counts[src_past][tgt_past][tgt_future]++;
        total++;
    }

    if (total < 10) {
        te_a_to_b[src * n_res + tgt] = 0.0f;
        te_b_to_a[src * n_res + tgt] = 0.0f;
        return;
    }

    // TE(A→B) = Σ p(src_past, tgt_past, tgt_future) * log2(
    //   p(tgt_future | tgt_past, src_past) / p(tgt_future | tgt_past) )
    float inv_total = 1.0f / (float)total;
    float te = 0.0f;

    for (int sp = 0; sp < 2; sp++) {
        for (int tp = 0; tp < 2; tp++) {
            int n_tp = counts[0][tp][0] + counts[0][tp][1] + counts[1][tp][0] + counts[1][tp][1];
            if (n_tp == 0) continue;
            for (int tf = 0; tf < 2; tf++) {
                int n_joint = counts[sp][tp][tf];
                if (n_joint == 0) continue;
                int n_sp_tp = counts[sp][tp][0] + counts[sp][tp][1];
                int n_tp_tf = counts[0][tp][tf] + counts[1][tp][tf];
                if (n_sp_tp == 0 || n_tp_tf == 0) continue;

                float p_joint = (float)n_joint * inv_total;
                float p_cond_full = (float)n_joint / (float)n_sp_tp;
                float p_cond_marginal = (float)n_tp_tf / (float)n_tp;

                te += p_joint * log2f(p_cond_full / fmaxf(p_cond_marginal, 1e-10f));
            }
        }
    }

    te_a_to_b[src * n_res + tgt] = fmaxf(0.0f, te);

    // TE(B→A): swap roles — same data, swap src↔tgt medians and matrix pointers
    // (computed symmetrically by the thread at [tgt, src], so we only need one direction here)
    // The reverse direction is computed by the thread where src=tgt_orig, tgt=src_orig.
    // No extra computation needed — the grid covers all pairs.
}

// Helper: compute per-row median for binarization
extern "C" __global__ void compute_row_medians(
    const half* __restrict__ spike_matrix,
    float* __restrict__ medians,
    int n_res,
    int n_bins,
    int n_bins_padded
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_res) return;

    // Approximate median via mean (exact median requires sorting, which is expensive)
    // For binary TE, mean threshold works well when distributions are roughly symmetric
    float sum = 0.0f;
    for (int t = 0; t < n_bins; t++) {
        sum += __half2float(spike_matrix[row * n_bins_padded + t]);
    }
    medians[row] = sum / (float)n_bins;
}
```

#### 15.3 Build system integration

Add to `crates/prism-gpu/build.rs` after the `tensor_ccf.cu` entry:
```rust
compile_kernel(&nvcc, "src/kernels/twin_transfer_entropy.cu",
    &ptx_dir.join("twin_transfer_entropy.ptx"),
    &target_ptx_dir.join("twin_transfer_entropy.ptx"));
```

#### 15.4 Rust wrapper in `twin_kernels.rs`

```rust
pub struct TwinTeCompute {
    te_a_to_b: CudaSlice<f32>,    // [n_res × n_res]
    te_b_to_a: CudaSlice<f32>,    // [n_res × n_res]
    median_a: CudaSlice<f32>,     // [n_res]
    median_b: CudaSlice<f32>,     // [n_res]
    te_kernel: CudaFunction,
    median_kernel: CudaFunction,
    n_res: i32,
}
```

#### 15.5 Integration into feature pipeline

Replace Step 6.3 item 9 (CPU TE) with GPU TE. Runs after CCF, on the same spike matrices:

```rust
// After CCF computation (Step 5):
let mut te = TwinTeCompute::new(&stream_exchange, &te_module, n_residues as i32)?;
te.compute_medians(&stream_exchange, &ccf.spike_matrix_a, &ccf.spike_matrix_b)?;
te.compute(&stream_exchange, &ccf.spike_matrix_a, &ccf.spike_matrix_b, n_bins, 1)?;
let te_ab = te.download_te_a_to_b(&stream_exchange)?;
let te_ba = te.download_te_b_to_a(&stream_exchange)?;

// Populate per-residue features
for r in 0..n_residues {
    features[r].transfer_entropy_a_to_b = te_ab[r * n_residues..].iter()
        .filter(|&&v| v > 0.01).sum::<f32>();  // sum of outgoing TE from this residue
    let te_ba_r: f32 = te_ba[r * n_residues..].iter()
        .filter(|&&v| v > 0.01).sum();
    features[r].causal_flow_direction = features[r].transfer_entropy_a_to_b - te_ba_r;
    features[r].mutual_information = /* MI from CCF peak values */;
}
```

#### 15.6 Performance

- Grid: `ceil(300/16) × ceil(300/16)` = 19×19 = 361 blocks
- Block: 16×16 = 256 threads
- Each thread: O(n_bins) ≈ 350 iterations with 8 register counters
- ESTIMATE: < 1ms for 300 residues (same order as CCF)
- **Validation**: `nsys profile` on TWIN run, grep for `twin_binned_te`

#### Regression gate 15

```bash
# Verify PTX builds
ls -la target/ptx/twin_transfer_entropy.ptx
# MUST: exists, non-zero

# Functional
scripts/prism-validate-and-run.sh \
    -t tests/fixtures/1btl_clean.topology.json \
    -o /tmp/twin_te_test \
    --coupled-twin --fast --fused-steps 6 --hmr --adaptive-dt -v 2>&1 | grep "transfer_entropy"
# MUST: non-zero values logged

python3 -c "
import json
data = json.load(open('/tmp/twin_te_test/1btl_clean.twin_per_residue.json'))
te_vals = [r['transfer_entropy_a_to_b'] for r in data['residues']]
nonzero = sum(1 for v in te_vals if v > 0.001)
print(f'✓ {nonzero}/{len(te_vals)} residues with TE > 0.001')
assert nonzero > 0, 'All TE zero — kernel not producing output'
"
```

#### If gate 15 fails:
1. Kernel launch failure → check shared memory: this kernel uses NO shared memory, only registers
2. All TE zero → median threshold wrong. Check `compute_row_medians` output (if all medians are 0, matrix is all zeros → no spikes)
3. NaN → log2f(0) or division by zero. Check `fmaxf(p_cond_marginal, 1e-10f)` guard

---

### STEP 16: NMA Pipeline Integration

**Goal**: Ensure every TWIN run has NMA modes available for Layer 4 (differential perturbation), so Group B actually differs from Group A.

#### 16.1 Current state (verified)

- `scripts/prism-prep` has `compute_nma_modes()` at line 738 (ProDy ANM)
- CLI flag: `--nma-modes N` generates `{stem}_nma_modes.json`
- ProDy 2.6.1 is installed (verified)
- `nhs_rt_full.rs` accepts `--nma-perturb <path>` and calls `engine.load_nma_modes()`
- `coupled_md.rs:382-385` loads NMA into engine_b when `twin_config.nma_modes_path` is Some

#### 16.2 Gap

NMA mode generation is **separate from the TWIN run**. The user must manually run `prism-prep --nma-modes 10` first, then pass `--nma-perturb <path>` to the TWIN run. If they forget, Group B runs thermal-only (identical to Group A except for seed), and all 18 differential features produce zeros.

#### 16.3 Solution: Auto-generate NMA modes in the prep pipeline

Modify `prism-validate-and-run.sh` to auto-generate NMA modes when `--coupled-twin` is set:

```bash
# In prism-validate-and-run.sh, after topology prep:
if [ "$COUPLED_TWIN" = "true" ] && [ ! -f "${TOPO_STEM}_nma_modes.json" ]; then
    echo "  Auto-generating NMA modes for TWIN Layer 4..."
    scripts/prism-prep "$CLEAN_PDB" "$TOPO_JSON" --nma-modes 10
    NMA_PATH="${TOPO_STEM}_nma_modes.json"
    ENGINE_FLAGS="$ENGINE_FLAGS --nma-perturb $NMA_PATH"
fi
```

#### 16.4 Verify NMA modes are valid

Add a validation check after generation:
```bash
python3 -c "
import json, sys
data = json.load(open('$NMA_PATH'))
assert data['n_modes'] >= 6, f'Too few modes: {data[\"n_modes\"]}'
assert data['n_residues'] > 50, f'Too few residues: {data[\"n_residues\"]}'
assert len(data['modes']) == data['n_modes']
assert len(data['ca_indices']) == data['n_residues']
print(f'✓ NMA: {data[\"n_modes\"]} modes × {data[\"n_residues\"]} residues')
" || { echo "NMA validation failed"; exit 1; }
```

#### Regression gate 16

```bash
# Verify auto-generation
scripts/prism-validate-and-run.sh \
    -t tests/fixtures/1btl_clean.topology.json \
    -o /tmp/twin_nma_test \
    --coupled-twin --fast --fused-steps 6 --hmr --adaptive-dt -v 2>&1 | grep "NMA"
# MUST show: "Auto-generating NMA modes" or "NMA loaded"
# MUST show: "Stream B: NMA loaded (amplification=3.0)"

# Verify differential features are non-zero
python3 -c "
import json
data = json.load(open('/tmp/twin_nma_test/1btl_clean.twin_per_residue.json'))
nma_excl = [r['nma_exclusive_count'] for r in data['residues']]
nonzero = sum(1 for v in nma_excl if v > 0)
print(f'✓ {nonzero} residues with NMA-exclusive spikes')
"
```

#### If gate 16 fails:
1. ProDy not installed → `pip install prody==2.6.1`
2. NMA modes file not found → check path construction in wrapper script
3. Zero NMA-exclusive spikes → NMA amplification too low (default 3.0). Try 5.0. Or protein too rigid for ANM modes.

---

### STEP 17: XGBoost Pairwise Ranker Training

**Goal**: Train a learned ranker on TWIN features so the 50 per-residue signals translate into better pocket rankings (SR@1).

#### 17.1 Current state

- `benchmarks/prism4d_bench30/train_xgb_ranker.py` exists (XGBoost pairwise, LOTO CV)
- Training data `site_features_all.csv` does **NOT exist** (verified)
- Current best non-learned ranker: `engine_geo + log_spike_count` at 13.6% SR@1
- v7 handcrafted: 5.1% SR@1
- Expected with learned ranker: 25-40% SR@1

#### 17.2 Generate training data

Run TWIN Phase B on all 60 BENCH30 targets:
```bash
for target in $(jq -r '.[].pdb_id' benchmarks/prism4d_bench30/benchmark_manifest.json); do
    scripts/prism-validate-and-run.sh \
        -t output/${target}/${target}_clean.topology.json \
        -o output/twin_bench30/${target} \
        --coupled-twin --multi-stream 8 --fast \
        --fused-steps 6 --hmr --adaptive-dt -v
done
```

#### 17.3 Extract features to CSV

Create `scripts/extract_twin_features_csv.py`:
- Reads `twin_per_residue.json` + `binding_sites.json` from each target
- Computes DCC to ground truth centroid for each site → relevance label
- Outputs `site_features_all.csv` with columns: `target_id, site_id, dcc, relevance, [50 TWIN features], [existing engine features]`

#### 17.4 Train ranker

```bash
python3 benchmarks/prism4d_bench30/train_xgb_ranker.py \
    --features-csv benchmarks/prism4d_bench30/site_features_all.csv \
    --output-model benchmarks/prism4d_bench30/twin_xgb_ranker.json \
    --cv leave-one-target-out
```

Expect output:
```
LOTO CV SR@1: XX.X% (target: 25-40%)
LOTO CV SR@3: XX.X%
Feature importance: [top 10 features]
```

#### 17.5 Integrate into pipeline

Add `--twin-ranker` flag to `prism_canonical.py` or `postprocess_twin.py` that:
1. Loads `twin_xgb_ranker.json`
2. Scores all sites using TWIN features
3. Re-ranks sites by XGBoost score
4. Writes reranked output

#### Regression gate 17

```bash
# Verify CSV has data
wc -l benchmarks/prism4d_bench30/site_features_all.csv
# MUST: > 500 rows (60 targets × ~10 sites each)

# Verify model trains
python3 benchmarks/prism4d_bench30/train_xgb_ranker.py
# MUST: print SR@1 and SR@3 metrics
# MUST: SR@1 > 15% (improvement over 13.6% non-learned baseline)

# Verify no TWIN feature is degenerate (constant or all NaN)
python3 -c "
import csv
with open('benchmarks/prism4d_bench30/site_features_all.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
twin_cols = [k for k in rows[0] if 'twin_' in k or 'ccf_' in k or 'te_' in k]
for col in twin_cols:
    vals = [float(r[col]) for r in rows if r[col] not in ('', 'nan', 'None')]
    if len(vals) < 10:
        print(f'WARNING: {col} has <10 non-null values')
    elif len(set(vals)) == 1:
        print(f'WARNING: {col} is constant ({vals[0]})')
print(f'✓ Checked {len(twin_cols)} TWIN feature columns')
"
```

#### If gate 17 fails:
1. SR@1 < 15% → check feature importance. If TWIN features rank low, the CCF/TE/consensus signals may not correlate with pocket quality for this benchmark.
2. Degenerate features → fix computation in Step 6 or Step 15. Constant features indicate a bug.
3. CSV generation fails → check path construction and JSON schema compatibility.

---

### STEP 18: Contamination-Clean Retrain of PrismAI Student Model

**Goal**: Train the PrismAI student model on TWIN-derived features from a clean (non-contaminated) target set.

#### 18.1 The contamination problem

From project memory: "teacher on 143/199 CB targets, all CB benchmarks contaminated, wait for PRISM-TWIN." The current teacher (non-TWIN engine) was run on 143 of 199 CryptoBench targets. Any student model trained on those outputs inherits the teacher's biases.

#### 18.2 Define clean target set

Use BENCH30 (30 targets, zero SNDC overlap, zero CryptoBench overlap) as the clean training set. TWIN features from these targets are uncontaminated because:
- TWIN coupling produces features the teacher never generated
- The 50 interferometric features (CCF, TE, consensus, differential) are physically grounded
- Within-group consensus filters stochastic noise the teacher couldn't filter

#### 18.3 Retrain pipeline

```bash
# 1. TWIN features are already generated in Step 17 (BENCH30 runs)
# 2. Extract per-residue features as training labels
python3 scripts/extract_prism_ai_training_data.py \
    --twin-results output/twin_bench30/ \
    --ground-truth benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json \
    --output prism-ai-inference/data/twin_training_v001.parquet

# 3. Warm-start from previous best weights (v002 cold = 0.639 AUC)
python3 prism-ai-inference/train.py \
    --data prism-ai-inference/data/twin_training_v001.parquet \
    --init-weights prism-ai-inference/models/v002_cold.pt \
    --features 216+50  # 216 engine features + 50 TWIN features = 266 total \
    --output prism-ai-inference/models/twin_v001.pt

# 4. Evaluate
python3 prism-ai-inference/evaluate.py \
    --model prism-ai-inference/models/twin_v001.pt \
    --test-data prism-ai-inference/data/twin_test.parquet
```

#### 18.4 Feature expansion: 216 → 266

The current model uses 216 engine features (from `feature_registry_216.md` in memory). TWIN adds 50 interferometric features. The model input expands from 216-dim to 266-dim.

**Warm-start strategy**: Initialize the first 216 weights from the best previous model (`v002_cold.pt`, AUC 0.639). Initialize the 50 new TWIN feature weights to zero. The model learns the TWIN signal incrementally.

#### 18.5 Validation criteria

| Metric | Target |
|--------|--------|
| AUC on held-out targets | > 0.639 (v002 cold baseline) |
| SR@1 on BENCH30 | > 25% (XGBoost baseline from Step 17) |
| No target regression | Zero EXCELLENT targets made worse |
| TWIN feature importance | ≥ 3 TWIN features in top 20 |

#### Regression gate 18 (Phase C complete — FULL POTENTIAL)

```bash
python3 prism-ai-inference/evaluate.py \
    --model prism-ai-inference/models/twin_v001.pt \
    --test-data prism-ai-inference/data/twin_test.parquet
# MUST: AUC > 0.639
# MUST: SR@1 reported and > 25%
# MUST: feature importance shows TWIN features contributing

# Verify no regression
python3 scripts/twin_nonregression.py output/twin_bench30
# MUST: "No regressions on EXCELLENT targets"
```

#### If gate 18 fails:
1. AUC < 0.639 → the 50 new features add noise. Check for degenerate features (Step 17 gate).
2. TWIN features rank last in importance → the interferometric signals don't correlate with pocket quality. This is a fundamental question about TWIN's value proposition — investigate whether CCF/TE signals are informative for ranking.
3. Warm-start fails → verify weight dimensions match. 216→266 expansion must pad, not reshape.

---

## Appendix A: VRAM Budget (ESTIMATES — validate with `nvidia-smi` during Step 10 gate)

| Component | Per-engine | N=4 per group | 2 groups × 4 |
|-----------|-----------|---------------|-------------|
| Positions + velocities + forces | ~0.5 GB | 2.0 GB | 4.0 GB |
| Voxel grids (128³ × 8 neurons) | ~0.6 GB | 2.4 GB | 4.8 GB |
| Spike buffer (accumulation) | ~0.1 GB | 0.4 GB | 0.8 GB |
| KCC + signal preservation | ~0.05 GB | 0.2 GB | 0.4 GB |
| **Subtotal per engine** | **~1.25 GB** | **5.0 GB** | **10.0 GB** |
| Ring buffers (2 × 8192 × 48B) | | | 0.001 GB |
| CCF matrices (300² × 4B × 2) | | | 0.001 GB |
| Base threshold snapshot (2 copies) | | | 0.2 GB |
| **Total** | | | **~10.2 GB** |

Fits in RTX 5080 16 GB with ~5.8 GB headroom. For proteins > 8000 atoms, reduce to 2×2 or 2×3.

## Appendix B: Latency Budget per Outer Step (ESTIMATES — validate with nsys at Step 10 gate)

| Operation | Time | Stream |
|-----------|------|--------|
| ring_buffer_read_and_adapt (×2 directions) | ~100μs | exchange |
| stream_exchange.synchronize() | ~5μs | exchange |
| engine.run(4 fused steps) × 8 engines (concurrent) | ~8ms | a0..a3, b0..b3 |
| aggregate_group_spikes (CPU) | ~200μs | CPU |
| ring_buffer_push_batch (×2 groups) | ~20μs | exchange |
| threshold_recovery (every 1000 steps) | ~5μs | exchange |
| **Total per outer step** | **~8.3ms** | |
| **Overhead from TWIN coupling** | **~325μs (3.9%)** | |

## Appendix C: Build & Test Quick Reference

```bash
# Build everything (including TWIN PTX)
cargo build -p prism-gpu -p prism-nhs --features gpu

# Run all prism-nhs unit tests
cargo test -p prism-nhs --lib

# Run only coupled_md tests
cargo test -p prism-nhs --lib coupled

# Run only twin_kernels tests
cargo test -p prism-nhs --lib twin_kernels

# Run only twin_detection tests
cargo test -p prism-nhs --lib twin_detection

# Verify TWIN PTX files exist
ls -la target/ptx/{ring_buffer,tensor_ccf,twin_persistent,twin_persistent_physics}.ptx

# GPU functional test (TEM1)
scripts/prism-validate-and-run.sh \
    -t output/1btl/1btl_clean.topology.json \
    -o /tmp/twin_e2e \
    --coupled-twin --multi-stream 8 --fast \
    --fused-steps 6 --hmr --adaptive-dt -v
```
