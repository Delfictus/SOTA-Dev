# SO(3) TMA / Async Bulk-Load Plan

**Author:** AGENT 10 — read-only scout
**Branch:** producer-repair-causal-truthing-20260426
**Baseline commit:** 8ca26189
**Target HW:** RTX 5080 sm_120 (Blackwell GB202), CUDA 13.2.51, driver 595.45.04
**Status:** PROFILING-FIRST — no implementation proposed in this pass.

---

## TL;DR

Five SO(3)-related `__global__` kernels exist in the engine. None of them
match the access pattern that motivates Hopper/Blackwell TMA. The hot one
(`prism_so3_project_manifold_kernel`) already uses well-tuned shared-memory
tiling, WMMA tf32 fragments, and per-warp warp-shuffle reductions. The
rest are launch-once init kernels, single-thread post-pass kernels, or
simple per-spike read-modify-write kernels. All five decisions land at
either **A (REUSE)** or **D (REJECT until profiling proves need)**. No
**B (add shmem tiling)** is justified because the high-arithmetic kernel
already tiles, and the low-arithmetic kernels do not reuse data across
threads. No **C (PROTOTYPE TMA)** is justified anywhere — TMA has setup
cost and shines for ≥several-KB tiles with reuse across CTAs, which this
codebase does not have.

**Recommended next action:** run a Nsight Compute profiling pass on the
canonical command (`scripts/prism-validate-and-run.sh ... --multi-stream
8 --fused-steps 6`) to capture DRAM throughput and stall reasons on
`prism_so3_project_manifold_kernel` before any optimization work. Until
the kernel's measured time as a fraction of chunk wall-clock is known,
TMA prototyping is a forbidden speculation.

---

## 1. Files inspected

CUDA sources (exhaustive within prism-nhs/src/cuda/):

- `crates/prism-nhs/src/cuda/so3_project.cuh`        (288 lines) — layout, FFI surface
- `crates/prism-nhs/src/cuda/so3_project.cu`         (1034 lines) — five kernels + host
- `crates/prism-nhs/src/cuda/sh_basis.cuh`           (235 lines) — Y_lm inline evaluator
- `crates/prism-nhs/src/cuda/sh_basis.cu`            (130 lines) — init + test-driver kernels
- `crates/prism-nhs/src/cuda/rich_spike.cuh`         (RichSpike layout, 64 B align(64))
- `crates/prism-nhs/src/cuda/symmetry_consensus.cu`  (consumer of ContactShellTile, NOT SO(3))
- `crates/prism-nhs/src/cuda/adjudicator.cu`         (consumer of ContactShellTile, NOT SO(3))

Rust callers:

- `crates/prism-nhs/src/so3_project.rs`              (1955 lines) — FFI + ContactShellTile mirror
- `crates/prism-nhs/src/sh_basis.rs`                 (521 lines) — SH FFI + init host helper
- `crates/prism-nhs/src/captured_pipeline.rs:3483-3578` — Path C launch sites
- `crates/prism-nhs/src/bin/nhs_rt_full.rs:5247-5808` — n_clusters wiring (4³=64 grid)
- `crates/prism-nhs/build.rs:171-195` — sh_basis + so3_project static-archive compile rules

Build configuration:

- `crates/prism-nhs/build.rs:327` — `-arch=sm_120` for PTX
- `crates/prism-nhs/build.rs:418` — `-arch=sm_120` for static archives
- All CUDA TUs compile sm_120 only (no sm_90 fallback).

Repo-wide TMA reference:

- `Cargo.toml:40` — comment mentions "TMA" as a future capability for
  prism-cuda-ext. No actual TMA code exists.
- No `cuTensorMap*`, `cp.async.bulk`, `cp.async`, or `tensor_map`
  identifier appears anywhere in the workspace.

CUDA / driver verification (run via Bash):

- `nvcc --version` → `release 13.2, V13.2.51` (sm_120 supported).
- `nvidia-smi --query-gpu=compute_cap` → `12.0` (Blackwell sm_120).
- TMA is supported on the silicon (sm_120 ≥ sm_90), but no code paths use it.

---

## 2. Functions / structs / kernels found

### 2.1 The five `__global__` kernels (file:line)

| # | Kernel | Defined at | Launch site | Hot path? |
|---|--------|-----------|-------------|-----------|
| K1 | `prism_so3_project_manifold_kernel` | `cuda/so3_project.cu:140` | `cuda/so3_project.cu:612` (host wrapper); `captured_pipeline.rs:3483` (RELAXED) and `:3567` (PERTURBED) | **YES, twice per chunk replay** |
| K2 | `prism_apply_gradient_gasp_kernel` | `cuda/so3_project.cu:685` | `cuda/so3_project.cu:1018`; `captured_pipeline.rs:3505` | YES, once per chunk replay |
| K3 | `prism_momentum_guard_check_kernel` | `cuda/so3_project.cu:832` | `cuda/so3_project.cu:887` (`<<<1,1>>>`); `captured_pipeline.rs:3533` | post-pass scalar |
| K4 | `prism_apply_com_correction_kernel` | `cuda/so3_project.cu:912` | `cuda/so3_project.cu:944`; `captured_pipeline.rs:3553` | YES, once per chunk replay |
| K5 | `prism_sh_init_constants_kernel` | `cuda/sh_basis.cu:28` | `cuda/sh_basis.cu:96` (`<<<1,1>>>`) — once per process | NO |
| K6 | `prism_sh_eval_kernel` | `cuda/sh_basis.cu:65` | `cuda/sh_basis.cu:113` — test driver only (`prism_sh_eval_run` is called only from `sh_basis.rs` tests; `nhs_rt_full.rs` does not call it) | NO |

K5 + K6 are the SH-basis test/init kernels; they are launched from Rust
test code only. K3 is single-threaded by construction. K2 + K4 are
per-spike RMW. **K1 is the only WMMA / SH-projection kernel on the hot
path**, and it is launched exactly **twice per chunk replay** (relaxed
manifold + perturbed manifold).

### 2.2 Key structs

- `prism_nhs::so3_project::ContactShellTile`
  (`cuda/so3_project.cuh:127`) — 1280 B, `alignas(128)`, FFI-stable.
  Output of K1; consumed by adjudicator + symmetry consensus.
- `prism_nhs::rich_spike::RichSpike`
  (`cuda/rich_spike.cuh:81`) — 64 B, `alignas(64)`. Cache-line-aligned.
  Input to K1, K2, K4.
- `prism_nhs::sh_basis::K_LM[36]`
  (`cuda/sh_basis.cu:19`) — `__device__` constant table (Y_lm
  normalization). Pointer fetched once per process via
  `prism_sh_basis_get_k_lm_dev_ptr`, passed to K1 as a kernel arg.
- `__constant__ d_residue_to_calpha[1024]`
  (`cuda/so3_project.cu:668`) — broadcast LUT for K2.
- `__constant__ d_mu01_lut[4]`, `d_current_md_step`
  (`cuda/so3_project.cu:656,682`) — per-replay constants for K2.

### 2.3 Hot-path runtime numbers

- **n_clusters ≤ 64**, fixed by the host-side spatial hash on a 4³ grid
  in `nhs_rt_full.rs:5247-5295`. Confirmed by the `chain_id_lut: [u8;
  64]` declaration at `nhs_rt_full.rs:5325` and the
  `n_clusters_actual.min(64)` clamps at `:5333, :5340, :5377, :5456`.
- **Block grid:** K1 is launched as `<<<n_clusters, 32, 0, stream>>>`
  (`cuda/so3_project.cu:76, 612`) — single warp per cluster. **At most
  64 blocks total**, 32 threads each → 2048 threads in the entire grid.
- **Spikes per cluster:** typical few-hundreds to few-thousands. Total
  spikes across all clusters bounded by `MAX_SPIKES_PER_STEP = 500_000`
  in `fused_engine.rs:1738`.
- **Replay cadence:** one captured-graph launch per chunk; chunk size =
  500 MD steps (`nhs_rt_full.rs:4899`). For a 25K-step run that is **50
  chunk replays × 2 K1 calls = 100 K1 launches per replica**.
- **Shared memory per K1 block:** ≈ 15.4 KB (see §4 below). Well under
  the 100 KB sm_120 dynamic-shmem ceiling and the 48 KB static-shmem
  default.

---

## 3. Reusable code

K1 already includes everything that justified WMMA tiling on this
codebase: bank-padded shared-memory tiles, WMMA tf32 fragments with fp32
accumulation, explicit `__syncwarp()` discipline, and L2 normalization
of the per-plane `a_lm` accumulator. The Rust mirror of `ContactShellTile`
is FFI-stable with `static_assert`s on both sides
(`cuda/so3_project.cuh:172`, `so3_project.rs:80`). Any TMA work would
inherit this scaffolding unchanged.

K2/K4 use `__ldg` for read-only fetches and 64-byte-aligned struct copies
for outputs. There is no shared memory, no need for cross-thread reuse,
and no opportunity for tile-shaped bulk loads.

---

## 4. SO(3) hot-path inventory — K1 (`prism_so3_project_manifold_kernel`)

### Per-block working set

- **Inputs streamed:**
  - `d_spikes[start..end]`: `n_in_cluster × 64 B` RichSpike records.
    Loaded **once per spike**, only by lane 0..15 in each 16-spike tile
    iteration (`cuda/so3_project.cu:309-310`). Coalesced (64 B aligned,
    contiguous indices in `tile_base + lane`).
  - `d_cluster_offsets[cluster_id..cluster_id+2]`: 8 B, scalar.
  - `d_k_lm[36]`: 144 B SH normalization table. **Reused N times per
    cluster** (once per spike, indirectly via the inline evaluator at
    `cuda/so3_project.cu:328`). Already a 144 B constant — **fits in L1
    constant cache**.
- **Outputs:**
  - `d_tiles_out[cluster_id]`: 1280 B ContactShellTile, written once per
    block at the end of the kernel.

### Shared-memory layout (`cuda/so3_project.cu:159-174, 426`)

```
s_plane_tile[4][16][52]  = 13,312 B   // weighted Y_lm × tf32 staging
s_wmma_stage[16][20]     =  1,280 B   // WMMA store buffer
s_plane_acc[4][48]       =    768 B   // per-plane a_lm accumulator
s_centroid[3] + s_aabb_min[3] + s_aabb_max[3] = 36 B
s_agg_* (3 × u32) + s_sum_w[4]                = 28 B
s_l2_norm[4]                                  = 16 B
─────────────────────────────────
Total ≈ 15.4 KB / block
```

15.4 KB of static shmem × 64 blocks fits comfortably in sm_120's
228 KB-per-SM shared partition. With one warp per block, occupancy is
limited by warps per SM (max 64 warps × 32 threads = 2048 threads), not
by shmem.

### Per-thread global loads (the access pattern that decides A/B/C/D)

- **Pass 1 (centroid / AABB):** `cuda/so3_project.cu:254-258`.
  Stride-32 loop over the cluster's RichSpike range. Each lane loads one
  RichSpike (64 B) per iteration. This is **maximally coalesced**: 32
  threads × 64 B = 2 KB per warp issue, all four 128-byte sectors of two
  L2 lines hit in one transaction pair. `__restrict__` is set on the
  pointer (`:141`). No further gain available.
- **Pass 2 (Y_lm projection):** `cuda/so3_project.cu:309-310`. Lanes
  0..n_in_tile-1 each load one RichSpike. n_in_tile ≤ 16 → **only 16 of
  32 lanes participate** in this load. This is the only sub-coalesced
  pattern in K1, but it costs at most one half-warp per tile iteration
  and the load hits the same L2 line as Pass 1 already brought in.
- **K_LM table:** `cuda/so3_project.cu:328` — 36 floats accessed
  identically by every active lane via `prism_sh_eval_lmax5`. This is a
  textbook **broadcast load** that the L1 constant cache already
  handles in O(1). No TMA win possible — TMA does not accelerate
  broadcast.
- **Reuse pattern:** every RichSpike is loaded **exactly once** by
  exactly one thread in exactly one block. **No cross-thread reuse, no
  cross-block reuse.** This is the empirical disqualifier for TMA: TMA
  shines when a tile is read by many threads (multicast across CTAs) or
  re-read across multiple compute phases. K1 reads each spike, computes
  its 36 Y_lm, weights it 4 ways, writes to shmem, and never touches it
  again.

### Operational intensity (rough)

Per spike: ~36 sincosf-equivalent FLOPs for Y_lm + 4 × 36 = 144 mul-adds
for plane weighting + atomicOr writes. ≈ 500 FLOP per 64 B load = **~8
FLOP/byte**. Above the L4/L40/H100 roofline ridge (~2-4 FLOP/byte) but
not as compute-bound as a dense matmul (>20 FLOP/byte). The kernel sits
in the "moderate compute / coalesced bandwidth" regime where TMA wins
are typically <5%.

### Per-block working set vs TMA tile economics

- TMA's break-even tile size on Hopper is well documented at ~4–8 KB:
  the descriptor encoding + barrier setup costs ~20 cycles, amortizing
  only when the tile carries >>1 KB of data with reuse.
- K1's largest contiguous stream is the RichSpike block: at ~1000 spikes
  × 64 B = 64 KB per cluster, but **chunked into 16-spike (1 KB)
  iterations** because that's the WMMA tile shape. A TMA load of 1 KB
  per iteration is dominated by descriptor cost, not throughput.
- A larger TMA tile (e.g. 4 spikes worth at once, 256 B per lane) would
  break the WMMA per-iteration accumulation contract and require a
  restructure of the warp-synchronous reduction — non-mechanical,
  high-risk.

### TMA-applicability scorecard for K1

| Criterion | Status | Evidence |
|---|---|---|
| Tile ≥ 4 KB per load with reuse | NO | per-spike 64 B, no reuse |
| Hot path | YES (assumed) | 100 launches per replica, but kernel time per launch unmeasured |
| Working set fits shmem | YES at 15.4 KB | `cuda/so3_project.cu:159-426` |
| Cross-CTA multicast useful | NO | each cluster is independent |
| Async overlap with compute | MARGINAL | per-spike compute is light vs 64 B load |
| Layout already tile-compatible | NO | RichSpike is 64 B AoS; TMA wants tiled SoA |
| Profiling shows L2/DRAM-bound | UNKNOWN | needs nsys/ncu pass |

**Conclusion: K1 is not a TMA candidate.** Even if TMA worked here, the
gain is bounded by the fraction of K1 time spent in DRAM transit, which
the static evidence suggests is small (high arithmetic intensity, fully
coalesced, K_LM in constant cache).

---

## 5. Decision matrix — exact plan per kernel

### K1 — `prism_so3_project_manifold_kernel`  →  **D (REJECT until profiling proves need)**

**Justification (file:line):**
- Already shared-memory tiled (`cuda/so3_project.cu:159-174`).
- Already WMMA-accelerated (`:282-405`).
- Per-spike load is coalesced and one-shot
  (`:254-258, 309-310`) — no cross-thread / cross-block reuse for TMA
  to capture.
- Block count is bounded by **n_clusters ≤ 64**
  (`nhs_rt_full.rs:5247-5325`); kernel launch overhead may dominate
  per-launch time more than the load pattern.
- Hot-path status is presumed (twice per chunk replay,
  `captured_pipeline.rs:3483, 3567`) but the **per-call duration as a
  fraction of chunk wall-clock has never been measured** in this branch.
  Without that number, optimization priority is unjustified.

**To resolve to A vs (hypothetical) C, capture (one Nsight Compute pass):**
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — if ≥ 70 %, the
  load pattern matters; if ≤ 20 %, the kernel is compute / latency-bound
  and TMA is irrelevant.
- `smsp__warps_active.avg.pct_of_peak_sustained_active` — current single-
  warp-per-block design caps occupancy at ~32 warps per SM; if achieved
  is far below, the bottleneck is parallelism, not bandwidth.
- `stall_long_sb`, `stall_short_sb`, `stall_membar` — if dominated by
  long-scoreboard global loads, reconsider; if dominated by `mma_pipe`
  or `barrier`, no load-side win available.
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_request` — should
  be 4 sectors (128 B) per request for full coalescing; deviations
  point to the half-warp pattern in Pass 2.

### K2 — `prism_apply_gradient_gasp_kernel`  →  **A (REUSE current layout)**

**Justification (file:line):**
- One thread per spike, 256 threads/block (`cuda/so3_project.cu:1016, 702`).
- Each thread reads its own RichSpike (64 B coalesced via aligned struct
  copy, `:706`), looks up one Cα atom index from `__constant__`
  (broadcast, `:724`), reads 12 B of force + 4 B of mass via `__ldg`
  (`:732-735`), writes one 64 B RichSpike output (`:815`).
- **Zero cross-thread reuse, no shared memory, no opportunity for TMA.**
- Performance is bounded by atomic contention into `d_com_shift[3]` /
  `d_total_mass[1]` (`:797-803`), not by load bandwidth.

### K3 — `prism_momentum_guard_check_kernel`  →  **A (REUSE)**

**Justification (file:line):**
- Single-thread post-pass kernel (`cuda/so3_project.cu:838:
  `if (threadIdx.x != 0 || blockIdx.x != 0) return;`).
- Reads 16 B total (4 floats), writes 16 B. No bandwidth concern.

### K4 — `prism_apply_com_correction_kernel`  →  **A (REUSE)**

**Justification (file:line):**
- One thread per spike, 256 threads/block (`cuda/so3_project.cu:917, 942-944`).
- Reads `d_com_correction[3]` once per thread (12 B broadcast → L1 hit
  for every thread in the warp after the first, `:920-922`).
- Reads + writes one RichSpike per thread (coalesced struct copy,
  `:924-928`). Pure SoA-equivalent stream — already optimal for ldg.

### K5 — `prism_sh_init_constants_kernel`  →  **A (REUSE)**

**Justification (file:line):**
- Single-thread `<<<1,1>>>` launch, runs **once per process**
  (`cuda/sh_basis.cu:28-29, 96`). Outside any meaningful hot path.

### K6 — `prism_sh_eval_kernel`  →  **A (REUSE)**

**Justification (file:line):**
- Test-driver kernel only. Used by `prism_sh_eval_run` which is called
  from `sh_basis.rs` tests (`sh_basis.rs:398, 482`). **Not called by the
  runtime hot path** — the runtime uses `prism_sh_eval_lmax5` inlined
  inside K1 (`cuda/so3_project.cu:328`).

---

## 6. Minimal touch list

**This pass: zero touches.** This document is the only deliverable.

If profiling later proves K1 is bandwidth-bound and TMA is justified
(unlikely based on static evidence), the minimum-change list would
involve:

- `crates/prism-nhs/src/cuda/so3_project.cu` — replace the lane-wise
  RichSpike load in Pass 2 with a TMA bulk load to a shmem stage buffer.
- A new helper TU `crates/prism-nhs/src/cuda/so3_tensor_map.cu` for the
  host-side `cuTensorMapEncodeTiled` setup.
- `crates/prism-nhs/build.rs` — no architectural change (sm_120 already
  supports TMA), but the new TU must be registered.

That work would be a separate commit lane gated on profiling evidence
that does not yet exist in this branch.

---

## 7. New structs / kernels needed

**None for this pass.** Decision gate is profiling, not implementation.

---

## 8. Acceptance tests (for the profiling pass that this plan calls for)

1. **Kernel time fraction.** Run `scripts/prism-validate-and-run.sh -t
   data/targets/4lpk.json.topology.json -o /tmp/so3_profile --fast
   --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70
   --fused-steps 6 --hmr --adaptive-dt --multi-differential
   --closed-loop-steering --asymmetric-steering --site-ranker phase-manifold
   --replica-seed 42 -v` under `nsys profile`. Extract
   `prism_so3_project_manifold_kernel` total time and compute its
   fraction of the chunk-loop wall time. **Pass** if K1 ≥ 5 % of chunk
   time (worth optimizing). **Reject TMA work** if K1 < 5 % (latency
   noise from launch overhead dominates).

2. **DRAM throughput.** `ncu --set roofline --kernel-name
   regex:so3_project_manifold` against the same run. Pass if
   `dram__throughput.avg.pct_of_peak_sustained_elapsed` ≥ 70 % AND
   `sm__warps_active.avg.pct_of_peak_sustained_active` ≤ 50 % (i.e.
   bandwidth-bound, not compute-bound). Otherwise reject TMA.

3. **Stall reasons.** `ncu --section SchedulerStats --section
   WarpStateStats`. Pass if `stall_long_sb` ≥ 30 % of total stall
   cycles. Otherwise the bottleneck is not load latency.

4. **Half-warp Pass-2 audit.** Confirm via `ncu --metrics
   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_request` that the
   per-tile RichSpike load pattern shows the expected 2-sector half-warp
   load (16 lanes × 64 B = 1024 B per warp issue). If it is fully
   coalesced 4-sector, the half-warp is already padded; if it is < 2
   sectors, there is a separate load-coalescing bug worth fixing
   independent of TMA.

If acceptance tests 1+2+3 all pass, escalate to a TMA prototype lane on
a fresh worktree with explicit operator GO. If any fail, the answer is
**A (REUSE) for K1 too** and this plan closes out.

---

## 9. Failure modes

**False-positive TMA proposals to guard against:**

- **"It's a 4-plane SH projection, that smells matmul-shaped."** — It is
  WMMA-shaped, not matmul-shaped. Each plane's reduction is `<ones>·B`
  (column sum), not a true matmul. Already executes on Tensor Cores.
- **"sm_120 supports TMA, therefore use TMA."** — TMA is a tool, not a
  rite. The break-even tile size is ≥ 4 KB with reuse; K1's per-iter
  tile is 1 KB with no reuse.
- **"K_LM is read often, multicast it via TMA."** — K_LM is 144 B; the
  L1 constant cache already broadcasts it for free. TMA descriptor
  setup alone exceeds the win.
- **"Add shared-memory tiling to K2/K4 to cache RichSpike."** — Each
  spike is read by exactly one thread in K2 and K4. Shmem buys nothing.

**Real failure modes the profiling pass must detect:**

- **Launch latency dominates K1.** With n_clusters ≤ 64 and 32
  threads/block, the kernel may complete in microseconds; CUDA graph
  launch overhead and the post-kernel `__threadfence()` at
  `cuda/so3_project.cu:583` may be the bottleneck. TMA cannot fix this.
  **Mitigation lane (separate from TMA):** consolidate K1's 64 single-
  cluster blocks into a persistent multi-cluster kernel, OR fuse K1
  with K2 + K4 to amortize launch overhead. Out of scope for this pass.
- **n_clusters << 64.** If typical runs only fill 8-16 clusters, the SM
  occupancy floor is the issue, not load bandwidth. TMA does not raise
  occupancy.
- **WMMA scheduling stalls.** The 24 mma_sync calls per 16-spike tile
  carry implicit register pressure. If `stall_mma_pipe` ≥ 20 %, the
  reduction itself (not loads) is the bottleneck.

---

## 10. Rollback

This pass writes only `docs/SO3_TMA_ASYNC_LOAD_PLAN.md`. Rollback is
`git restore docs/SO3_TMA_ASYNC_LOAD_PLAN.md` (or `git rm`). Zero
runtime, build, or kernel files modified.

If a future TMA prototype lane lands and needs reverting, the contract
is: the prototype must be gated on a feature flag (`PRISM_SO3_TMA=1`
env var or a new `--so3-tma` CLI arg) and the legacy path
(`prism_so3_project_manifold_kernel` as it stands today) must remain
default. Rollback = unset the flag; remove the new TU; revert
`build.rs`. No other code touches.

---

## Appendix A — Search-term audit

Confirmed absent from the workspace (via `grep -rn` over `crates/`,
`docs/`, `build.rs`, `Cargo.toml`):

- `cuTensorMap` / `cuTensorMapEncodeTiled` — 0 hits.
- `cp.async.bulk` / `cp.async.bulk.tensor` — 0 hits.
- `cp.async` (any form) — 0 hits.
- `tensor_map` / `TensorMap` — 0 hits.
- `cuTensorMap` types — 0 hits.

Confirmed present:

- `__shared__` — present in `adjudicator.cu:280-283`,
  `so3_project.cu:159-174,426` (only). No other kernel uses static
  shmem for tiling; the rest are either single-thread or fully ldg/stg
  streaming.
- `ldmatrix` — referenced by name in `so3_project.cu:83` (comment about
  WMMA's underlying PTX op). The kernel does not call `ldmatrix.sync`
  directly; it goes through `wmma::load_matrix_sync`.
- `sm_120` — every CUDA TU's compile-line comment + `build.rs:327, 418`.
  No sm_90 fallback present.
- `blackwell` / `hopper` — comments only (`adjudicator.cu`,
  `lbvh_tree.cuh`, `vram_pool.cu`, `rich_spike.cuh`, etc.). No code
  paths gate on architecture beyond the OptiX RTX 5080 fallback in
  `nhs_rt_full.rs:5909-5915`.

---

## Appendix B — Why the hot path is structurally TMA-resistant

The PRISM-4D SO(3) projection is a **per-cluster reduction with a small
expansion factor** (36 SH coefficients × 4 planes = 144 floats per
cluster output) over a **moderate spike count per cluster** (typically
10² to 10⁴). The total data read by K1 across all clusters per launch
is bounded by `MAX_SPIKES_PER_STEP × 64 B = 32 MB`. Spread across
`n_clusters ≤ 64` blocks and 100 launches per replica, that's at most
~3 GB of streamed RichSpike data per replica.

TMA exists to accelerate **dense matrix-shaped tile loads with
multicast-style reuse across CTAs** (e.g. cuBLAS GEMM kernels on
Hopper). That is not the access pattern here. The right comparison
point for K1 is a **scatter-gather reduction** (cluster-local SH
projection with one read per element), not a matmul.

For comparison, the kernels that *do* match TMA's strengths in this
codebase — e.g. `lbvh_tree.cu`'s bottom-up AABB reduce, or the
`adjudicator.cu` SIMT 4-plane KL — also do not currently use TMA, and
the same evidence-first decision applies to them: profile before
prototyping.
