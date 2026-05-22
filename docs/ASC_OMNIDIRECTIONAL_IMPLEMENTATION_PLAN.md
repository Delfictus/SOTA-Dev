# ASC Omnidirectional Controller — Auditable Implementation Plan

Agent: 4 (READ-ONLY scout pass)
Branch: `producer-repair-causal-truthing-20260426`
Baseline: `8ca26189`
Date: 2026-05-04

---

## 0. Scope (verbatim from mission)

Turn the existing ASC (asymmetric / closed-loop steering controller) into an
auditable omnidirectional controller. Today ASC may apply forces and modify
trajectories without a complete audit trail. We need a SNAPSHOT/EXPORT contract
so every ASC action emits force summary, reason, target, work/energy effect,
stability flag, and reversibility flag. **No code edits this pass — we plan.**

Invariant: **No unaudited steering.** Every ASC application MUST emit force
summary, reason, target, work/energy effect, stability flag.

---

## 1. Files inspected

All paths absolute. Lines cited where load-bearing.

- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/captured_pipeline.rs`
  - `AscConfig` struct: lines 785-805
  - `ZstrCaptureParams` (post-ASC force-norm capture): lines 807-857
  - `PipelineConfig::asc` field: 928-929
  - W_ext pointer wiring (`d_external_work`): 988-1003, 2703-3018
  - gasp_gain_eta / force_burst_step wire-up: 3021-3146
  - Node D (ASC force inject) capture site: 3741-3768
  - Energy monitor reduce capture: 3848-3881
  - SFA / gearbox capture (consumes adj.adjudication_code post-ASC): 3883-3898
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/persistent_engine.rs`
  - `set_steering_focus_residue` legacy single-residue ASC hook: 1244-1253
  - `write_steering_focus` (Stage-2 closed-loop list): 2257-2336
  - `download_protocol_state` (host-side read of focus_match_count etc.): 2347-2361
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/fused_engine.rs`
  - `d_forces` raw ptr accessor: 2185-2194
  - `d_positions` raw ptr accessor: 2219
  - Raw-storage owners (the buffers ASC mutates): 1757, 2610, 5071
  - Legacy single-residue ASC writeback: 3711-3756
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/cuda/asc_steering.cu` (entire 106 lines — vectorised v4 kernel)
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/cuda/asc_steering.cuh` (entire 53 lines — declarations)
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/cuda/adjudicator.cu`
  - `prism_asc_apply_kernel` (T3 boundary repulsion): 542-604
  - `prism_asc_apply` host launcher: 732-755
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/cuda/adjudicator.cuh`
  - `InterferometricAdjudicatorFfi` struct + offset asserts: 85-205
  - `prism_asc_apply` extern decl: 313-339
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/protocol_state.rs`
  - `SteerEntry` + `STEERING_FOCUS_MAX = 64`: 21-43
  - `ProtocolState` (steering_focus_count, focus_match_count, processed_spike_count, last_seen_*): 122-197
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/zstr.rs`
  - `ZstrFrameHeader` (force_norm @ 28, external_work @ 32, potential_energy @ 40): 67-114
  - G29 trap on non-finite `force_norm` (post-ASC scalar witness): 648-667
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/rescue_controller.rs`
  - Module overview + entry-points: 1-57
  - `ObservationWindow` / `RescueAction` / `RescueController::decide`: 73-630
  - `RescueDecisionRecord` (already a host-side audit record): 515-528
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/ghost_tile.rs`
  - `GhostTileFrame` (4096 B sector, `_reserved_payload[32]` reserved for "ASC-steering work delta"): 80-162
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/cuda/energy_monitor.cuh`
  - CUB DeviceReduce::Sum captured node + `EnergyWindow {prev,cur}`: 33-95
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/twin_kernels.rs`
  - read_and_adapt_with_steering: 384-444 (legacy single-residue path)
- `/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/bin/nhs_rt_full.rs`
  - `--closed-loop-steering` / `--asymmetric-steering` flags: 374-439
  - `AscSharedState` struct (host-side aggregator): 4090-4210
  - Mask alloc + `AscConfig` build (steering_gain_alpha = 0.005 hardcode): 5554-5598
  - PHASE A.4 ASC force-vector trajectory writer (DtoH every chunk): 5142-5158, 7118-7180
  - ASC barrier / event-driven controller block: 7419-7600+
  - ASC sidecar writer (asc_events.bin / acl_contrast.bin / asc_consensus.json /
    gcpid_synergy.json / phasors.bin): 16433-16584

I did NOT find `crates/prism-nhs/src/*telemetry*` containing ASC event records
beyond `ghost_telemetry.rs` (a generic pinned ring) and the ad-hoc
`AscSharedState.event_log: Mutex<Vec<(u32, String)>>` in `nhs_rt_full.rs:4103`.

`DagNodeKind::ASCSnapshot` (Agent 1 territory) is **not yet defined** in the
repo — `rg DagNodeKind crates/` returns empty. The plan below treats it as a
forward dependency to be wired by Agent 1; the snapshot type defined here is
the payload Agent 1 will tag.

---

## 2. Functions / structs found (the existing ASC surface)

### 2.1 Device buffers ASC owns / mutates today

| Buffer | Owner / file:line | Size | Lifetime | Mutated by ASC? |
|---|---|---|---|---|
| `d_forces` (f32 AoS, n×3) | `fused_engine.rs:1757` (engine), exposed via `d_forces_dev_ptr` `fused_engine.rs:2185-2194` | 12·n_atoms B | engine lifetime | **YES** — atomic-add in `prism_asc_apply_kernel` (`adjudicator.cu:584-586`) and `asc_inject_repulsion_v4_kernel` (`asc_steering.cu:86-95`) |
| `d_positions` (f32 AoS, n×3) | `fused_engine.rs` accessor `d_positions_dev_ptr` `:2219` | 12·n_atoms B | engine lifetime | **READ-ONLY** by ASC (kernel input) |
| `d_atom_in_cluster` (u32 mask) | `nhs_rt_full.rs:5560-5570` (per-stream `cuMemAlloc_v2`, `cuMemsetD32_v2 → 1u32`) | 4·n_atoms B | per-stream pipeline build → engine teardown | READ-ONLY by `prism_asc_apply_kernel`; today **all-ones** ("omnidirectional expansion — every atom participates", comment `nhs_rt_full.rs:5554-5556`) |
| `d_pe_components` (f64 per-atom) | engine `d_potential_energy_components_dev_ptr` (`nhs_rt_full.rs:5749`) | 8·n_atoms B | engine lifetime | **YES** — `prism_asc_apply_kernel:599-603` atomic-adds `V_ASC = -½·α·Δ_AB·‖x_i−X_c‖²·mask` into per-atom PE so SFA's drift fuse subtracts steering work |
| `d_external_work` (f64 scalar, F2-pool) | `cfg.d_external_work` (`captured_pipeline.rs:988-1003`); allocated by `engine.allocate_external_work_buffer()` (`nhs_rt_full.rs:5759`) | 8 B | engine lifetime; zeroed each chunk via `cuMemsetD8Async` at head of replay (`captured_pipeline.rs:3346-3358`) | UV velocity-kick + force/velocity clamps atomic-add ΔK; ASC kernel does NOT currently write here directly — it writes V_ASC into per-atom PE which gets reduced |
| `d_pe_scalar` (f64 reduce target) | `energy_pe_scalar_dev` (F2-pool, `captured_pipeline.rs:1156`) | 8 B | pipeline lifetime | written by CUB DeviceReduce::Sum capture; mirrored to `adj.potential_energy` @ offset 112 |
| `EnergyWindow {prev, cur}` (16 B) | `energy_window_dev` (F2-pool, `captured_pipeline.rs:1157`) | 16 B | pipeline lifetime | `prism_energy_monitor_window_update_kernel` rolls per launch — already a captured device-side residue of the post-ASC potential |
| `ProtocolState.steering_focus_residues[64]` (8 B × 64 = 512 B) | `protocol_state.rs:173`, device copy in `engine.d_protocol_state` | 700 B total ProtocolState | engine lifetime | **YES** — host writes every chunk via `write_steering_focus` (`persistent_engine.rs:2282-2336`); kernel `ring_buffer_read_and_adapt` reads, atomic-incs `focus_match_count` (`protocol_state.rs:181`), `processed_spike_count` (`:186`), and stamps `last_seen_focus_id` / `last_seen_spike_residue` (`:191-196`) |
| `cruise_state_dev` (`ChronometricStateTensor`, 16 B) | `captured_pipeline.rs:1133` | 16 B | pipeline lifetime | gear-id ASC saw at retirement (already host-readable post-replay) |
| `adj.current_divergence` (f32 @ offset 48) | F2-pool adj_dev (`captured_pipeline.rs:1105`) | 4 B | pipeline lifetime | input to ASC; readable post-launch via `download_adjudicator_ffi` patterns |
| `adj.adjudication_code` (u32 @ offset 52) | same | 4 B | same | input to ASC — Construct (1) is the only code that fires the kernel (`adjudicator.cu:562-563`) |
| ZSTR pinned slot `force_norm` (f32 @ slot+28) | pinned-host ring base `nhs_rt_full.rs:5632`; computed in-graph by `zstr_force_stage_f4_kernel` + `zstr_force_norm_sqrt_kernel` | 4 B / slot × 5 slots | run lifetime | **already** a post-ASC scalar witness ‖F‖₂; G29 trap reads it (`zstr.rs:648-667`) |
| ZSTR pinned slot `external_work` (f64 @ slot+32) | same | 8 B / slot | same | snapshotted from `*adj.d_external_work` at frame retirement (`zstr.rs:93-100`) |
| ZSTR pinned slot `potential_energy` (f64 @ slot+40) | same | 8 B / slot | same | snapshotted from `adj.potential_energy` (`zstr.rs:101-106`) |

### 2.2 Existing ASC kernels

| Kernel | File:line | Inputs | Outputs |
|---|---|---|---|
| `prism_asc_apply_kernel` (T3 — Boundary Repulsion) | `crates/prism-nhs/src/cuda/adjudicator.cu:542` | `adj`, `d_forces`, `d_atom_positions`, `d_atom_in_cluster`, n_atoms, α, `pe_components` | atomic-add F_i = α·Δ_AB·(x_i − X_c)·mask → d_forces; atomic-add V_ASC into pe_components |
| `prism_asc_apply` (host launcher) | `crates/prism-nhs/src/cuda/adjudicator.cu:732` | same plus stream | grid=(n_atoms+255)/256, block=256 |
| `asc_inject_repulsion_v4_kernel` (vectorised v4 alternative) | `crates/prism-nhs/src/cuda/asc_steering.cu:46` | `float4* d_forces`, `const float4* d_pos`, centroid, α, ΔKL, n_atoms | `atom.global.add.v4.f32` repulsion injection |
| `ring_buffer_read_and_adapt` (steering-focus consumer) | `crates/prism-nhs/src/twin_kernels.rs:384-444` (FFI; CUDA in `find_twin_ptx`) | `ProtocolState*`, spike ring | per-spike threshold-reduction boost when residue ∈ focus list |

Only `prism_asc_apply` is captured into the production graph today (Node D in
`captured_pipeline.rs:3741-3768`); the v4 vectorised variant is wired through
`zstr.rs:37-38` comments as the canonical reference but is not currently the
captured launcher (the simpler scalar kernel in `adjudicator.cu` is what fires).

### 2.3 Existing host-side audit / sidecar exports

| Sidecar | Producer | Schema |
|---|---|---|
| `<name>.asc_events.bin` | `nhs_rt_full.rs:16437-16453` and `:16569-16583` | `b"ASC1"` magic + u32 count + `(u32 chunk_idx, u16 desc_len, [u8] desc)` records — descriptor is a **freeform `String`** |
| `<name>.acl_contrast.bin` | `nhs_rt_full.rs:16456-16469` | `b"ACL1"` + u32 count + `(u32 chunk_idx, f32 ratio)` |
| `<name>.asc_consensus.json` | `nhs_rt_full.rs:16472-16485` | `{n_streams, n_groups, consensus_residues:[{residue_id,n_groups,s_pc}]}` |
| `<name>.gcpid_synergy.json` | `nhs_rt_full.rs:16499-16541` | per-residue PID atoms, sorted by synergy_fraction |
| `<name>.phasors.bin` | `nhs_rt_full.rs:16545-16567` | `b"PHZ1"` + 4 groups × n_residues × (f64 cos, f64 sin, u32 count) |
| `<stem>_stream{i}_asc_trajectory.bin` (PHASE A.4) | `nhs_rt_full.rs:5142-5158, 7118-7180` | header `u64 n_atoms_loaded`; per chunk `(u64 chunk_idx, u64 frame_idx, f32[3·n_atoms] forces)` — full DtoH copy of `d_forces` after each chunk |
| ZSTR `<name>.zstr` (NVMe O_DIRECT) | `zstr.rs:600-700+` | per-frame `[ZstrFrameHeader 4096B][positions n×12B][forces n×12B][pad]` — `force_norm`, `external_work`, `potential_energy` already populated |
| Ghost-tile records (Channel B) | `ghost_tile.rs:80-162`; pushed by `prism_ghost_pipe_stage` | per-cluster per-frame; `_reserved_payload[32]` (offset 128, 128 B free) explicitly reserved for "ASC-steering work delta, gear ID, Φ_sym phase-lock score" |
| `RescueDecisionRecord` (in-memory, not yet exported) | `rescue_controller.rs:515-528, 763-768` | per-chunk observation + deficit + actions; serde-ready |

### 2.4 VRAM-only ASC state with no DtoH path today

Cross-checking the four existing snapshots against an audit-grade contract:

1. `d_atom_in_cluster` mask: per-stream allocated, currently all-ones, **never
   downloaded**. Today the only `target_voxels` semantic is "all atoms".
2. `d_forces` post-ASC delta: full force buffer DtoH'd in PHASE A.4 path
   (`nhs_rt_full.rs:7118-7180`), but **the ASC contribution is not
   isolated** — the host sees F_total, not ΔF_ASC.
3. `V_ASC` per-atom contribution: written into `pe_components` (`adjudicator.cu:599-603`)
   then **subsumed** into the CUB Sum reduce. Host gets only the post-ASC total V_t,
   not ‖V_ASC‖ alone.
4. `ProtocolState.focus_match_count` / `processed_spike_count`: are
   downloaded once via `download_protocol_state` (`persistent_engine.rs:2347-2361`)
   but only **at end of run**, not per-chunk. No per-chunk snapshot.
5. ASC's per-launch reason ("Construct fired" vs "Prune skipped" vs "Violation
   trap"): only inferable from `adj.adjudication_code` post-replay. Today there
   is **no record of which chunk's ASC fired** other than the freeform string
   log (`AscSharedState.event_log`).

### 2.5 ASC interaction with G26 / d_dt / gearbox

ASC is the ONE-BEFORE step that updates pe_components → CUB reduce → SFA reads
drift to choose gear. So `cruise.current_gear` AFTER the SFA kernel fires
(`captured_pipeline.rs:3883-3898`) is causally downstream of the ASC kernel
that ran in the same captured graph. The post-ASC gear is therefore the
authoritative `controller_mode` for the snapshot.

`d_dt` (offset 120, `*float`) is not modified by ASC; the gearbox SWITCH bodies
write through it. ASC is "feed-forward" w.r.t. dt — it fires under whatever dt
the previous SFA step locked.

### 2.6 ASC consumption of future F1/WHILE traces (Agent 2/3 territory)

Today the pipeline is "ASC fires every captured-graph replay; replay = one
chunk". When Agent 2/3 introduces F1-CONDITIONAL bodies (Construct/Prune
sub-graphs) and WHILE-loops inside the chunk window, ASC will fire 0..K times
per chunk. The audit contract here therefore must:

- Allow **0..K snapshots per (stream_id, chunk_id)** — `chunk_id` alone is no
  longer unique. We add `body_idx: u32` (sub-graph body index, 0xFFFF_FFFF when
  not inside a SWITCH) and `iteration: u32` (WHILE iteration counter).
- Read its inputs from a struct Agent 2/3 produces — the F1/WHILE trace will
  expose `body_idx` and `iteration` via the device side; we will need a
  small ringbuffer of `(body_idx, iteration)` per replay that the snapshot
  emitter consumes. **Keep snapshot fields opaque to Agent 4's pass.**

---

## 3. Reusable code (do not reinvent)

| Need | Reuse |
|---|---|
| Pinned-host ring + DtoH | `crates/prism-nhs/src/ghost_telemetry.rs` (`PinnedTelemetryRing<T>`) |
| Sector-aligned record | `crates/prism-nhs/src/ghost_tile.rs` (4096 B `GhostTileFrame`, `_reserved_payload[32]`) |
| Per-atom reduce | `cuda/energy_monitor.cu*` (CUB DeviceReduce::Sum on f64 buffer) — clone-and-rename for ‖F_ASC‖² + Σ V_ASC |
| Per-chunk audit record | `rescue_controller::RescueDecisionRecord` — already serde-ready |
| Sidecar write-out path | `nhs_rt_full.rs:16433-16584` (existing block; append a new sidecar in the same gate) |
| F2-pool buffer alloc (lifetime tied to pipeline) | `prism_vram_pool_alloc_async` (used throughout `captured_pipeline.rs`) |
| Adjudicator FFI offset asserts pattern | `adjudicator.cuh:177-204` |

---

## 4. Unsafe / problematic spots

1. **`AscSharedState.event_log: Mutex<Vec<(u32, String)>>`** (`nhs_rt_full.rs:4103`)
   — freeform string, no enum, no schema. Today this is the sole "reason"
   record. **Do not extend it**; replace its consumers with the typed snapshot
   below. Keep the file format (`b"ASC1"`) for back-compat by serialising the
   typed snapshot's `reason` field there.

2. **Global event_log lock contention** — every stream's per-chunk thread-0
   block today acquires `event_log.lock()`. With 8+ streams this serialises.
   The new snapshot ring should be lock-free (atomic push into a per-stream
   pre-allocated slab; flush on chunk barrier).

3. **PHASE A.4 full-DtoH of `d_forces`** (`nhs_rt_full.rs:7118-7180`) is
   correct for forensic capture but expensive (3·n·4 B PCIe per chunk per
   stream). For the audit contract we want a CHEAP summary — a CUB reduce
   into one f64 (Σ‖F_ASC‖²) is ~12 B DtoH per chunk per stream. Keep PHASE A.4
   as an opt-in deep mode (already the case).

4. **`steering_gain_alpha = 0.005` is hardcoded** (`nhs_rt_full.rs:5582`).
   The snapshot must capture this value verbatim per replay so off-line audit
   distinguishes "α was tuned mid-run" (operator changed CLI) from "α drifted
   at runtime" (would be a violation). Currently captured-graph baked, so
   safe; record explicitly anyway.

5. **`d_atom_in_cluster` is all-ones at allocation** (`nhs_rt_full.rs:5567`,
   `cuMemsetD32_v2(d_mask, 1u32, n_atoms)`). The `target_voxels` field in the
   snapshot must therefore ALSO record the active mask checksum (BLAKE3 of
   the mask, or its popcount) — without that, "we steered atom set X" is not
   verifiable. Until the mask becomes non-trivial, popcount + BLAKE3 of the
   first 64 bytes is enough to detect drift cheaply.

6. **`prism_asc_apply_kernel` early-returns on `code != CONSTRUCT`** but the
   captured node STILL fires (zero work). The snapshot must distinguish
   "kernel fired and applied force" from "kernel fired and skipped because
   adjudication_code != 1". Resolution: the in-kernel reduce (point 3 above)
   atomic-adds `(force_norm² + 1)` into a u64 packing `count|force_sq` so
   `count = 0` ⇔ no atom received force, even when the kernel was scheduled.

7. **The asc_inject_repulsion_v4_kernel is not the captured launcher** but is
   referenced as the canonical kernel in zstr.rs comments. Document the
   discrepancy in the plan; either retire the v4 source-of-truth claim or
   migrate Node D to the v4 path. Out of scope for this lane — flag only.

---

## 5. Exact plan

We propose a four-layer audit chain. NO Rust runtime/Python crossings; ALL
summaries are Rust-emitted.

### 5.1 Device layer — cheap on-replay reduction

**One new captured kernel node** (`prism_asc_summary_reduce_kernel`,
`crates/prism-nhs/src/cuda/asc_steering.cu` — extend the existing namespace):

```c
// Inputs: d_forces_post_asc (mutated by Node D), d_pe_components_post_asc,
//         d_force_pre_asc_snapshot (NEW — F2-pool buffer of n×3 f32 the
//         captured graph copies BEFORE Node D fires; size = same as d_forces),
//         adj (for adjudication_code, current_divergence, gear_id),
//         protocol_state (for steering_focus_count + steering_focus_residues hash).
// Outputs: AscSummaryFfi struct (NEW, 64 B aligned 64) — single device record.
```

**Schema (single 64-byte cache-line, F2-pool):**

```c
struct __align__(64) AscSummaryFfi {
    uint32_t adjudication_code;     //   0..4    snapshot of adj.adjudication_code
    uint32_t fired_atom_count;      //   4..8    popcount(d_atom_in_cluster) ∩ (|ΔF|>ε)
    float    force_norm_l2;         //   8..12   ‖ΔF_ASC‖₂ over fired atoms
    float    force_max_abs;         //  12..16   max_i max_d |ΔF_ASC[i,d]|
    double   v_asc_total;           //  16..24   Σ V_ASC (kcal/mol) — same as Δ in pe_components
    double   work_delta;            //  24..32   ∫ F·dr step-summed (W_ext attribution)
    float    current_divergence;    //  32..36   adj.current_divergence (Δ_AB)
    float    steering_gain_alpha;   //  36..40   α used this replay
    uint32_t gear_id;               //  40..44   cruise.current_gear post-SFA
    uint32_t steering_focus_count;  //  44..48   ProtocolState.steering_focus_count
    uint32_t target_residue_lo;     //  48..52   focus_residues_xxh3[0..32]
    uint32_t target_residue_hi;     //  52..56   focus_residues_xxh3[32..64]
    uint32_t target_mask_popcount;  //  56..60   popcount of d_atom_in_cluster
    uint32_t target_mask_xxh3_lo;   //  60..64   xxh3-32 over first 4096 B of mask
};
static_assert(sizeof(AscSummaryFfi) == 64);
```

The reducer is a single 256-thread block performing warp-shuffle reduces, so
captured-node cost is O(1) launch, ≈ µs at any n_atoms < 10⁶. **Cheaper
than DtoH of d_forces by 5–6 orders of magnitude.**

This is the canonical "reduction kernel over full DtoH" choice mandated by
the brief. The full DtoH path (PHASE A.4) survives as an opt-in deep-trace
mode unchanged.

### 5.2 Pre-ASC snapshot buffer

To compute **ΔF_ASC** rather than F_total, the captured graph adds one node
BEFORE Node D:

`prism_force_snapshot_node` — `cuMemcpyDtoDAsync(d_force_pre_asc, d_forces, n×3×4 B)`.

F2-pool allocation, one-shot at pipeline build (`captured_pipeline.rs:3741`
neighbourhood). Adds 4·n·3 B VRAM (~120 KB at n=10⁴, negligible). This is
the SOLE new device-side dependency — it lets the reducer compute true
ΔF instead of F_total. Without it the snapshot would over-attribute to ASC.

### 5.3 Host layer — pinned-ring DtoH

One pinned-host slot per stream per chunk, 64 B each:

```rust
// crates/prism-nhs/src/asc_summary.rs (NEW)
#[repr(C, align(64))]
pub struct AscSummaryFfi { /* mirror of the C struct above */ }

pub struct AscSummaryRing {
    ring: PinnedTelemetryRing<AscSummaryFfi>, // reuse ghost_telemetry.rs
    // 64 slots = 1 KB pinned per stream — trivial.
}
```

Captured-graph node: `cuMemcpyDtoHAsync(ring_slot, d_summary, 64)` after
the reduce kernel. Rolls per replay using the same `__constant__ active_slot`
pattern ZSTR uses (`captured_pipeline.rs:1170-1177`).

### 5.4 Rust orchestration layer — typed `AscSnapshot` + reason synthesis

The mission-mandated contract (verbatim shape):

```rust
// crates/prism-nhs/src/asc_summary.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AscSnapshot {
    pub stream_id:       u32,
    pub chunk_id:        u64,
    pub phase:           String,         // "cold_hold" | "ramp" | "warm_hold" | "ramp_down"
    pub controller_mode: String,         // "construct" | "prune" | "violation" | "rescue:focus_amp" | "rescue:hold:<reason>"
    pub target_residues: Vec<u32>,       // top-K from ProtocolState.steering_focus_residues
    pub target_voxels:   Vec<u64>,       // morton-encoded voxel ids of ContactShellTile centroids that participated
    pub force_norm:      f32,            // direct from AscSummaryFfi.force_norm_l2
    pub force_vector_hash: String,       // hex(blake3(d_force_pre_asc XOR d_force_post_asc)) — deep-trace mode only; "" otherwise
    pub work_delta:      f32,            // AscSummaryFfi.work_delta cast to f32 (kcal/mol)
    pub energy_budget_pass: bool,        // |work_delta| / |V_t| < 0.01 (configurable)
    pub stability_flag:  String,         // "ok" | "force_norm_nan" | "force_norm_overflow" | "energy_budget_breach" | "g29_trap"
    pub reversibility_flag: String,      // "reversible" (Construct, NVE-safe α) | "irreversible:violation" | "irreversible:rescue_v2"
    pub reason:          String,         // structured one-liner: "burst_2sigma+focus_amp[k=4,a=0.005]" etc.
}
```

**Conflict-with-existing-types check:** none. `AscSnapshot` is new; the existing
freeform string log (`AscSharedState.event_log`) becomes a *consumer* of
`AscSnapshot.reason` rather than the source of truth.

The `reason` synthesis is data-driven from the existing observables:

| Source | reason fragment |
|---|---|
| `delta > mean+2σ` (burst detection, `nhs_rt_full.rs:7449`) | `"burst_{:.1}sigma"` |
| BOCPD `recent_changepoint_probability(2)` > 0.3 | `"changepoint_{:.2}"` |
| `RescueAction::FocusWeightAmplifier{multiplier}` | `"rescue:focus_amp[m={:.2}]"` |
| `RescueAction::Hold{reason}` | `"rescue:hold:{reason}"` |
| Construct + no rescue | `"construct_baseline"` |
| Prune | `"prune_skip"` (kernel fired but did nothing) |
| Violation | `"violation_trap[flags={:#x}]"` (from `adj.adjudication_reason_flags`) |

`stability_flag` synthesis:
- `force_norm.is_nan()` → `"force_norm_nan"` (G29 trap path; matches `zstr.rs:660-666`)
- `force_norm > FORCE_NORM_CEILING` (configurable, default 1e6 kcal/(mol·Å)) → `"force_norm_overflow"`
- `|work_delta| > 0.01 * |V_t|` → `"energy_budget_breach"`
- otherwise → `"ok"`

`reversibility_flag` synthesis:
- `controller_mode == "violation"` → `"irreversible:violation"`
- any `RescueAction::EngineV2*` was applied → `"irreversible:rescue_v2"`
- else (NVE-safe at α≤0.01, all v1 RescueActions) → `"reversible"`

### 5.5 DAG node tag (Agent 1 hand-off)

When Agent 1 lands `DagNodeKind::ASCSnapshot`, this is the payload. The DAG
node carries (`AscSnapshot`, parent_dag_node_id, child_dag_node_ids). For now,
emit `AscSnapshot` records into a per-run `Vec<AscSnapshot>` carried in
`AscSharedState`, and write them out as a sidecar.

### 5.6 New sidecar

`<name>.asc_snapshots.bin` — append in `nhs_rt_full.rs:16433-16584` block:

```text
magic "ASCS"  (4 B)
version u32 = 1
n_records u32
records: [u32 record_len, [u8] bincode-serialized AscSnapshot]+
```

Plus `<name>.asc_snapshots.json` (drained at end-of-run, similar to the
existing rescue history pattern) — same payload, JSON-pretty for human audit.
Decision: emit BOTH; binary for replay, JSON for human review. Already the
established pattern (`asc_consensus.json` + `asc_events.bin`).

---

## 6. Minimal touch list

**No edits this pass.** When implementation lands, the touch list is:

| File | Edit | Why |
|---|---|---|
| `crates/prism-nhs/src/cuda/asc_steering.cu` | **add** `prism_asc_summary_reduce_kernel` + host launcher | the cheap reduce |
| `crates/prism-nhs/src/cuda/asc_steering.cuh` | **add** declaration + `AscSummaryFfi` struct + offset asserts | match Rust mirror |
| `crates/prism-nhs/src/cuda/adjudicator.cu` | **no edit** | T3 kernel already writes V_ASC into pe_components correctly |
| `crates/prism-nhs/src/asc_summary.rs` | **NEW FILE** | `AscSummaryFfi`, `AscSnapshot`, `AscSummaryRing`, reason synthesis |
| `crates/prism-nhs/src/lib.rs` | **add 1 line** | `pub mod asc_summary;` |
| `crates/prism-nhs/src/captured_pipeline.rs` | edit `PipelineConfig` (add `asc_summary: Option<AscSummaryConfig>`); add 3 nodes between `:3741-3768` and `:3848` (pre-ASC snapshot copy → Node D → reducer → DtoH) | snapshot capture |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | host-side ring drain after each chunk barrier; reason synthesis + `AscSnapshot` push into a new `AscSharedState.snapshots: Mutex<Vec<AscSnapshot>>`; sidecar emit (one block after `:16542`) | host-side audit chain |
| `crates/prism-nhs/build.rs` | **no edit** (asc_steering.cu already in build) | n/a |
| `crates/prism-nhs/src/rescue_controller.rs` | **no edit** (its records remain the second feed; `AscSnapshot.reason` quotes them) | n/a |
| `crates/prism-nhs/src/persistent_engine.rs` | **no edit** | `write_steering_focus` and `download_protocol_state` already produce everything we need |

**Total: 1 new file, 4 edits, ZERO touches in `fused_engine.rs` /
`persistent_engine.rs` proper / `rescue_controller.rs` / Python.** The captured
pipeline and CUDA kernels grow by a single reducer + a 4·n·3 B VRAM scratch.

---

## 7. New structs / functions (full list)

### Rust (`crates/prism-nhs/src/asc_summary.rs`)
- `#[repr(C, align(64))] struct AscSummaryFfi { ... }` (sizeof == 64)
- `struct AscSummaryConfig { d_force_pre_snapshot: u64, d_summary: u64, d_protocol_state: u64, ring_pinned_base: *mut AscSummaryFfi, ring_n_slots: u32 }`
- `struct AscSummaryRing { ring: PinnedTelemetryRing<AscSummaryFfi>, n_slots: usize }`
  - `impl AscSummaryRing { fn allocate(n_slots: usize) -> Result<Self, String>; fn read_slot(&self, slot: u32) -> AscSummaryFfi; }`
- `struct AscSnapshot { /* exact contract from §0 */ }` plus `Serialize` / `Deserialize`
- `fn synthesize_reason(summary: &AscSummaryFfi, rescue_actions: &[RescueAction], burst_sigma: f32, bocpd_p: f32) -> String`
- `fn synthesize_stability_flag(summary: &AscSummaryFfi, v_t_now: f64, ceiling: f32) -> String`
- `fn synthesize_reversibility_flag(controller_mode: &str, rescue_v2_applied: bool) -> String`
- `fn build_snapshot(stream_id: u32, chunk_id: u64, phase: &str, summary: AscSummaryFfi, rescue_decision: Option<&RescueDecisionRecord>, deep_trace_force_hash: Option<&str>, v_t_prev: f64, v_t_now: f64) -> AscSnapshot`

### CUDA (extend `asc_steering.cu` / `asc_steering.cuh`)
- `__global__ void prism_asc_summary_reduce_kernel(const float* d_force_pre, const float* d_force_post, const double* d_pe_post, uint32_t n_atoms, const InterferometricAdjudicatorFfi* adj, const ProtocolState* d_protocol, AscSummaryFfi* out_summary)`
- `int prism_asc_summary_reduce_launch(/* matching args */, void* stream)` — host launcher

### Captured-pipeline additions (`captured_pipeline.rs`)
- 3 new captured nodes (pre-snapshot DtoD, reducer kernel, post-DtoH copy)
- 1 new field `asc_summary: Option<AscSummaryConfig>` on `PipelineConfig`
- 1 new F2-pool field `asc_pre_force_snapshot_dev: usize` on `CapturedAdjudicationPipeline`
- 1 new F2-pool field `asc_summary_dev: usize`
- Drop impl extends to free both

### nhs_rt_full.rs additions
- `AscSharedState.snapshots: Mutex<Vec<AscSnapshot>>` field (additive — no removals)
- after-barrier hook calling `build_snapshot` per stream per chunk
- new sidecar block writing `asc_snapshots.bin` and `asc_snapshots.json`
- existing `event_log` block stays for back-compat; can become a thin
  wrapper that pushes `AscSnapshot.reason` strings into the legacy `ASC1`
  format for tools still reading it

---

## 8. Acceptance tests

1. **`prism-nhs/tests/asc_snapshot_layout.rs`** — `assert_eq!(size_of::<AscSummaryFfi>(), 64)`; `assert_eq!(align_of::<AscSummaryFfi>(), 64)`; offset asserts mirroring the `static_assert`s in the .cuh.
2. **`prism-nhs/tests/asc_snapshot_kernel.rs`** (gated `#[cfg(feature = "gpu")]`) — call the reducer directly with synthetic `d_force_pre = 0.0f32`, `d_force_post = (1, 2, 3, 1, 2, 3, ...)` for n_atoms=4; assert `force_norm_l2 == sqrt(56.0 * 4)` and `fired_atom_count == 4`. NaN injection → assert stability_flag synthesises to `"force_norm_nan"`.
3. **`prism-nhs/tests/asc_snapshot_no_steering.rs`** — replay the captured graph with `adj.adjudication_code = PRUNE` (T3 kernel early-returns); assert `fired_atom_count == 0`, `force_norm_l2 == 0.0`, `controller_mode == "prune"`, `reason.starts_with("prune")`.
4. **`prism-nhs/tests/asc_snapshot_consistency.rs`** — for one captured-graph chunk, deep-trace the full DtoH (PHASE A.4 path) AND the reduce path; assert `‖ΔF_PHASE_A4‖₂` matches `summary.force_norm_l2` to 1e-4 relative.
5. **`prism-nhs/tests/asc_snapshot_reversibility.rs`** — synthetic ObservationWindow that triggers `RescueAction::EngineV2NmaAmpMultiplier`; assert `reversibility_flag == "irreversible:rescue_v2"`. Plain Construct → `"reversible"`.
6. **`tests/integration_smoke.rs`** (or extend existing canonical smoke) — run `scripts/prism-validate-and-run.sh` on a 2iyt fixture; assert `<output>/<name>.asc_snapshots.bin` exists, magic == `b"ASCS"`, n_records >= 1, every record decodes, every record has non-empty `reason`, every record has `phase ∈ {cold_hold, ramp, warm_hold, ramp_down}`, every record has `force_norm.is_finite()`.
7. **DAG hand-off test** — once Agent 1 lands `DagNodeKind::ASCSnapshot`, assert each `AscSnapshot` produced by the snapshot path has a matching DAG node id. (Cross-lane; skip until Agent 1 lands.)

---

## 9. Failure modes

| Failure | Detection | Recovery |
|---|---|---|
| F2-pool exhaustion on `asc_pre_force_snapshot_dev` alloc | `prism_vram_pool_alloc_async` rc != CUDA_SUCCESS | LANE-BLOCKED escalation; do NOT silently disable snapshots — fail-loud per invariant |
| Pinned ring slot overflow (chunks fire faster than host drains) | host-side `last_drained < live_idx - N_SLOTS` | log warning, increment `frames_dropped` counter, mint snapshot with `reason = "snapshot_dropped:ring_overflow"` and `stability_flag = "snapshot_lost"` |
| `force_norm` returned as NaN by reducer | `summary.force_norm_l2.is_nan()` host-side | snapshot emitted with `stability_flag = "force_norm_nan"` and `reversibility_flag = "irreversible:violation"`; orchestrator may abort campaign (G29 trap protocol — same as ZSTR's existing path, `zstr.rs:660-666`) |
| `adjudication_code = VIOLATION` (NaN in V_t or momentum guard fired) | `summary.adjudication_code == 2` | snapshot emitted with `controller_mode = "violation"`, `reason = "violation_trap[flags=...]"` from `adj.adjudication_reason_flags` |
| `ProtocolState.steering_focus_count = 64` AND no actual focus uploaded (host upload race) | `summary.target_mask_popcount == 0` AND `steering_focus_count > 0` | snapshot emits `stability_flag = "focus_upload_stale"` — diagnostic-only; controller behaviour unchanged |
| Captured graph rebuilt mid-run (Step-101 heuristic reset, `captured_pipeline.rs:904`) | next replay's `chunk_id` resets to `pipeline_rebuild_count` | emit a synthetic `AscSnapshot` with `reason = "pipeline_rebuilt"`, `controller_mode = "init"`; downstream tools key on `(stream_id, pipeline_epoch, chunk_id)` |
| Agent 2/3 F1/WHILE iterations split a chunk into K sub-records | host sees K summary slots per chunk | snapshot ring grows to `K_max × N_SLOTS`; AscSnapshot carries `body_idx, iteration` (extension fields), default 0 in this lane |
| Async DtoH lags behind host-side `event_log` writes | sidecar `asc_snapshots.bin` records appear out-of-order vs `asc_events.bin` | both sidecars carry `chunk_idx`; downstream merge by `(stream_id, chunk_idx)`. No invariant violation. |

**FORBIDDEN paths (re-stated):**
- No silent overlay fallback when ASC summary is missing — emit a snapshot
  with `reason = "snapshot_missing:<cause>"` or omit the chunk entirely with
  a top-level `audit_note`. Never fabricate fields.
- No Python in runtime — `AscSnapshot` is Rust-emitted; Python's role is
  read-only post-run analysis (`scripts/explicit_solvent/` pattern).
- No removal of existing `asc_events.bin` / `acl_contrast.bin` / `phasors.bin`
  exports — they continue alongside the new snapshot sidecar.

---

## 10. Rollback

Three independent rollback strata, in order of cost:

1. **Disable at build** — guard `PipelineConfig.asc_summary` behind a
   `--asc-snapshot` CLI flag (default on). Off ⇒ the 3 new captured nodes are
   not recorded; the pipeline replays at exactly the pre-change cost. The
   snapshot module compiles but contributes no captured nodes.
2. **Disable at runtime** — `--no-asc-snapshot` flag forces the host to
   skip the DtoH drain and sidecar emit. The captured graph still runs the
   reducer (cost ~µs per replay) but the ring is never read. Useful for
   benchmark runs where we want the engine intact but no audit overhead.
3. **Hard revert** — single revert of the implementation commit. Rollback
   surface is bounded:
   - 1 new file (`asc_summary.rs`) — delete
   - 1 lib.rs line — revert
   - 4 captured_pipeline.rs hunks (new field, 3 captured nodes, Drop fix) — revert
   - 1 asc_steering.cu hunk (new kernel) — revert
   - 1 asc_steering.cuh hunk (new decl + struct) — revert
   - 1 nhs_rt_full.rs hunk (sidecar emit + AscSharedState field) — revert

   All other ASC code (Node D, Stage 2 closed-loop writeback, rescue
   controller, ZSTR force_norm, Hamiltonian Auditor, asc_events.bin) is
   untouched and continues to work.

Default posture if implementation lands but the operator wants to demote the
audit: `--no-asc-snapshot` (stratum 2) — keeps the ABI stable for downstream
analysis tools, kills only the per-replay PCIe traffic.

---

## Appendix A — Field lineage table (proves every snapshot field is sourced, not invented)

| AscSnapshot field | Source | File:line |
|---|---|---|
| `stream_id` | per-stream worker thread index | `nhs_rt_full.rs:5174` (multi-stream fan-out) |
| `chunk_id` | `chunk_idx` loop variable | `nhs_rt_full.rs:5160` |
| `phase` | CCNS phase boundary check on `current_step` | `protocol_state.rs:67-71`, `:305-321` |
| `controller_mode` | derived from `adj.adjudication_code` + active rescue actions | `adjudicator.cuh:92`, `rescue_controller.rs:475-485` |
| `target_residues` | `ProtocolState.steering_focus_residues[0..steering_focus_count]` | `protocol_state.rs:169-173` |
| `target_voxels` | `ContactShellTile` centroids ∈ active clusters (Morton-encoded) | `so3_project.rs` (existing centroid computation) |
| `force_norm` | reducer `Σ‖ΔF_i‖²`-then-sqrt | NEW kernel |
| `force_vector_hash` | optional BLAKE3 of `XOR(d_force_pre, d_force_post)` | NEW; deep-trace only |
| `work_delta` | reducer integrates `Σ_i F_i·Δr_i` (one-step proxy: `α·Δ·Σ‖x_i−X_c‖²·mask` matches `V_ASC` per `adjudicator.cu:599-603`) | reuse PE accumulation logic |
| `energy_budget_pass` | `|work_delta| < 0.01 · |V_t|` where V_t reads `adj.potential_energy` (offset 112) | `adjudicator.cuh:134`, captured `EnergyWindow` |
| `stability_flag` | finite-check + ceiling check on `force_norm`; energy budget | this plan §5.4 |
| `reversibility_flag` | controller_mode + rescue v2 applied | this plan §5.4 |
| `reason` | structured synthesis from BOCPD + RescueAction + adjudication_code | this plan §5.4 |

---

## Appendix B — Conflicts found

- **None at the type level** (`AscSnapshot` is a new public type).
- **One semantic conflict to flag:** the existing freeform
  `AscSharedState.event_log: Vec<(u32, String)>` already produces `asc_events.bin`
  records keyed by chunk_idx. Implementation must keep that file's magic and
  schema bit-identical (downstream tools depend on it) while populating its
  contents from the new structured `AscSnapshot.reason` field. Plan §5.6
  records the dual-emit rule.
- **One open question for Agent 1:** when `DagNodeKind::ASCSnapshot` lands,
  must the snapshot emit happen INSIDE the captured graph (cuGraphAddNode
  for a host-callback) or after replay completion? This plan assumes the
  latter (host drains the ring after `cuGraphLaunch` returns); if Agent 1
  needs in-graph emit (for cross-stream DAG ordering), the pinned-ring
  drain becomes a host-callback node. The `AscSnapshot` payload itself is
  unchanged either way.

— end of plan —
