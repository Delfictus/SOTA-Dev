# Self-Clocking Control Plane — Invariants & Trace Contract (Scout)

Status: SCOUT DELIVERABLE — read-only; no Rust/CUDA/build/runtime files modified.
Author: AGENT 11 — Self-Clocking Gear Shifter / G26-F1-WHILE Control Plane Scout
Branch: `producer-repair-causal-truthing-20260426`
Baseline commit: `8ca26189`
Date: 2026-05-04

Companion plans (this doc references but does not duplicate):
- [docs/F1_PARENT_SWITCH_IMPLEMENTATION_PLAN.md](F1_PARENT_SWITCH_IMPLEMENTATION_PLAN.md) — Agent 2
- [docs/WHILE_NODE_IMPLEMENTATION_PLAN.md](WHILE_NODE_IMPLEMENTATION_PLAN.md) — Agent 3
- [docs/DAG_FOUNDATION_IMPLEMENTATION_PLAN.md](DAG_FOUNDATION_IMPLEMENTATION_PLAN.md) — Agent 1
- [docs/ASC_OMNIDIRECTIONAL_IMPLEMENTATION_PLAN.md](ASC_OMNIDIRECTIONAL_IMPLEMENTATION_PLAN.md) — Agent 4
- [docs/TIER8_GRAPH_TOPOLOGY.md](TIER8_GRAPH_TOPOLOGY.md)
- [docs/EXECUTION_POLICY.md](EXECUTION_POLICY.md)

---

## 0. Bottom line

PRISM-4D's first **device-derived control component** is already live: the
G26 chronometric gearbox runs as a parent-owned SWITCH whose predicate is
computed by a single-thread bridge kernel
([gearbox.cu:436-462](../crates/prism-nhs/src/cuda/gearbox.cu)) and whose
SFA stages the stateful finite automaton on device
([gearbox.cu:198-280](../crates/prism-nhs/src/cuda/gearbox.cu)). The dt
mutation it commits travels through `*(adj->d_dt)` to a fixed device address
inside `d_protocol_state`
([fused_engine.rs:2289-2297](../crates/prism-nhs/src/fused_engine.rs)) and
is read by every integrator kernel via `d_protocol->dt`. **No host write to
dt happens in the hot path while the gearbox is active**
([fused_engine.rs:5602-5612](../crates/prism-nhs/src/fused_engine.rs)
explicitly bypasses the `--adaptive-dt` host write when
`gearbox_active=true`).

This plan formalises **G26 as the template** and binds F1 (Agent 2) and
WHILE (Agent 3) to the same discipline through six invariants and a
seven-field unified trace schema. The plan **does not** duplicate Agent
2's or Agent 3's patch sequence; it defines the schema, file location,
and static-assertion pattern they must obey.

**One control-plane defect found and FLAGGED** — see §4.1 (the
`gear_override` host write at `nhs_rt_full.rs:6755-6788` is the single
remaining hot-path host mutation of device control state). Recommendation
is to retire it once F1's parent-owned SWITCH lands; concrete migration
sketched in §5.7.

---

## 1. Files inspected

All paths absolute under `/home/diddy/Desktop/Prism4D-bio`.

### CUDA / C++ FFI
- [crates/prism-nhs/src/cuda/gearbox.cu](../crates/prism-nhs/src/cuda/gearbox.cu) (entire, 781 lines)
  — predicate bridge kernel (l. 436-462, 464-477), SFA kernel (l. 198-280),
    PointerSwap (l. 102-149), apply_fixed_dt body kernel (l. 555-582),
    populator (l. 654-758).
- [crates/prism-nhs/src/cuda/gearbox.cuh:200-296](../crates/prism-nhs/src/cuda/gearbox.cuh)
  — predicate bridge declaration (l. 217-221), apply_fixed_dt declaration
    (l. 263-266), populator declaration (l. 287-296).
- [crates/prism-nhs/src/cuda/adjudicator.cu:865-1182](../crates/prism-nhs/src/cuda/adjudicator.cu)
  — F1 SWITCH FFI (l. 865-963; LEGACY in-graph variant, NOT to be reused for
    parent-owned wiring per Agent 2 §4), G26 parent-owned helpers
    (l. 1117-1182).
- [crates/prism-nhs/src/cuda/graph_node.cu:1-191](../crates/prism-nhs/src/cuda/graph_node.cu)
  — splice-legality preflight (l. 70+), GRAPH/Memset adders (no WHILE yet).
- [crates/prism-nhs/src/cuda/graph_node.cuh](../crates/prism-nhs/src/cuda/graph_node.cuh)
  — public unified-graph-builder surface.

### Rust orchestration
- [crates/prism-nhs/src/captured_pipeline.rs:686-1019](../crates/prism-nhs/src/captured_pipeline.rs)
  — `extern "C"` G26/F1 FFI (l. 686-779), `PipelineConfig.d_dt` (l. 955),
    `PipelineConfig.g26_parent_cond_handle` (l. 973),
    `PipelineConfig.d_external_work` (l. 988-1003).
- [crates/prism-nhs/src/captured_pipeline.rs:3800-3930](../crates/prism-nhs/src/captured_pipeline.rs)
  — capture-time SFA + predicate-bridge launches; `g26_bridge_node` snapshot.
- [crates/prism-nhs/src/captured_pipeline.rs:4461-4585](../crates/prism-nhs/src/captured_pipeline.rs)
  — parent SWITCH wiring (`create_parent_g26_cond_handle` l. 4470-4497;
    `wire_parent_g26_switch` l. 4511-4585).
- [crates/prism-nhs/src/persistent_engine.rs:1437-1443](../crates/prism-nhs/src/persistent_engine.rs)
  — `d_protocol_dt_dev_ptr` forwarder (the device-side dt address).
- [crates/prism-nhs/src/fused_engine.rs:2274-2297](../crates/prism-nhs/src/fused_engine.rs)
  — `d_protocol_dt_dev_ptr` truth: ProtocolState `dt` field at offset 84
    inside `d_protocol_state`.
- [crates/prism-nhs/src/fused_engine.rs:5602-5612](../crates/prism-nhs/src/fused_engine.rs)
  — adaptive_dt host-write bypass when gearbox is active (the existing
    correct invariant guard).
- [crates/prism-nhs/src/fused_engine.rs:3978-3984](../crates/prism-nhs/src/fused_engine.rs)
  — `set_dt(new_dt)` host accessor; init-time only.
- [crates/prism-nhs/src/gearbox.rs:46-89](../crates/prism-nhs/src/gearbox.rs)
  — `ChronometricStateTensor` Rust mirror (counter, last_burst_frame,
    current_gear, previous_gear, v_prev).
- [crates/prism-nhs/src/ghost_telemetry.rs:536-566](../crates/prism-nhs/src/ghost_telemetry.rs)
  — host-side F1 scraper (`log_f1_switch_events`); current "F1 layer" is
    log-only, not a CUDA-graph conditional.
- [crates/prism-nhs/src/graph_node.rs](../crates/prism-nhs/src/graph_node.rs)
  — TIER 7/8 unified shim; splice-legality types (`SpliceLegalityReport`,
    `SpliceError`); zero existing references to `G26Decision` /
    `F1Decision` / `WhileIteration`.

### Binary
- [crates/prism-nhs/src/bin/nhs_rt_full.rs:1551-1557](../crates/prism-nhs/src/bin/nhs_rt_full.rs)
  — init-time `engine.set_dt(0.004)` for HMR; pre-pipeline-build only.
- [crates/prism-nhs/src/bin/nhs_rt_full.rs:5736-5818](../crates/prism-nhs/src/bin/nhs_rt_full.rs)
  — `cfg.d_dt = engine.d_protocol_dt_dev_ptr() as *mut f32` wiring; the
    device address is plumbed once, never via host write afterward.
- [crates/prism-nhs/src/bin/nhs_rt_full.rs:5870-6075](../crates/prism-nhs/src/bin/nhs_rt_full.rs)
  — monolithic-discovery block: parent-handle create (l. 5927-5944),
    G26 wire (l. 5998-6017), parent post-G26 audit (l. 6017+).
- [crates/prism-nhs/src/bin/nhs_rt_full.rs:6180-6186](../crates/prism-nhs/src/bin/nhs_rt_full.rs)
  — `engine.set_gearbox_active(true)` flag — the operator-mandated
    "Gearbox > Adaptive-DT" hierarchy lock.
- [crates/prism-nhs/src/bin/nhs_rt_full.rs:6735-6788](../crates/prism-nhs/src/bin/nhs_rt_full.rs)
  — **Force-Gear Orchestration Supervisor Shim**. Hot-path host write to
    device-side `gear_override` (offset 100). **FLAGGED — see §4.1.**
- [crates/prism-nhs/src/bin/nhs_rt_full.rs:6837-6862](../crates/prism-nhs/src/bin/nhs_rt_full.rs)
  — current host-side F1 scrape (`log_f1_switch_events`).

### Build / policy
- [crates/prism-nhs/build.rs:29-48, 206-224](../crates/prism-nhs/build.rs)
  — `gearbox.cu`, `adjudicator.cu`, `graph_node.cu` compiled as static
    archives; no build edits required for this lane.
- [docs/EXECUTION_POLICY.md:39](../docs/EXECUTION_POLICY.md)
  — Direct `sys::cuGraphAdd*` from Rust prohibited.
- [docs/TIER8_GRAPH_TOPOLOGY.md:6-11, 25-54](../docs/TIER8_GRAPH_TOPOLOGY.md)
  — GRAPH-SPLICE-001; Option β topology; freeze evidence (l. 68-97).

### Cross-agent plans (read, not modified)
- [docs/F1_PARENT_SWITCH_IMPLEMENTATION_PLAN.md](F1_PARENT_SWITCH_IMPLEMENTATION_PLAN.md) (entire, 571 lines).
- [docs/WHILE_NODE_IMPLEMENTATION_PLAN.md](WHILE_NODE_IMPLEMENTATION_PLAN.md) (entire, 560 lines).
- [docs/DAG_FOUNDATION_IMPLEMENTATION_PLAN.md](DAG_FOUNDATION_IMPLEMENTATION_PLAN.md) (entire, 569 lines).
- [docs/ASC_OMNIDIRECTIONAL_IMPLEMENTATION_PLAN.md](ASC_OMNIDIRECTIONAL_IMPLEMENTATION_PLAN.md) §§3-4 (energy reducers).

---

## 2. Functions / structs / kernels found (G26 today, F1+WHILE planned)

| Layer | Symbol | File:line | Role |
|---|---|---|---|
| CUDA | `prism_gearbox_predicate_bridge_kernel` | gearbox.cu:437-462 | Single-thread; reads `cruise->current_gear` (l. 452) and consults `adj->gear_override` at offset 100 (l. 454-459); calls `cudaGraphSetConditional(handle, gear & 0x3)` (l. 461). **THE G26 PREDICATE BRIDGE.** |
| CUDA | `prism_gearbox_launch_predicate_bridge` | gearbox.cu:464-477 | Host shim that launches the bridge kernel into capture window. |
| CUDA | `prism_gearbox_sfa_kernel` | gearbox.cu:198-280 | Stateful finite automaton; reads `adj->adjudication_code` (offset 52, l. 205), `adj->potential_energy` (offset 112, l. 240), `*adj->d_external_work` (offset 128, l. 247-249); writes `cruise->{current_gear, previous_gear, counter, last_burst_frame, v_prev}` (l. 269-278). **All control-state reads/writes live on device.** |
| CUDA | `prism_gearbox_pointer_swap_kernel` | gearbox.cu:101-149 | Combined SFA + dt write through `*(adj->d_dt)` at offset 120 (l. 144-148). Used in non-Logic/Action-Bifurcation pipelines. |
| CUDA | `prism_gearbox_apply_fixed_dt_kernel` | gearbox.cu:555-582 | SWITCH-body kernel: writes `d_gearbox_table[target_gear * 4]` into `*(adj->d_dt)` (l. 569). **Per-body dt commit is fully device-side.** |
| CUDA | `d_gearbox_table` (`__constant__`) | gearbox.cu:26-31 | 64-byte gear-to-dt table; written once via `prism_gearbox_init_table_async` (l. 153-170). |
| CUDA | `prism_gearbox_create_handle_ffi` | adjudicator.cu:1119 | Parent-owned conditional-handle creator (template for F1 / WHILE). |
| CUDA | `prism_gearbox_wire_with_handle_ffi` | adjudicator.cu:1137 | Parent-owned SWITCH wirer (size=4 hardcoded). |
| Rust | `PipelineConfig.d_dt: *mut f32` | captured_pipeline.rs:955 | Wired to `d_protocol_dt_dev_ptr` (engine offset 84). |
| Rust | `PipelineConfig.g26_parent_cond_handle: u64` | captured_pipeline.rs:973 | Zero ⇒ child wires its own SWITCH; nonzero ⇒ parent owns. |
| Rust | `CapturedAdjudicationPipeline::create_parent_g26_cond_handle` | captured_pipeline.rs:4470-4497 | Default value = 1 (Gear 1 / 2.0 fs). |
| Rust | `CapturedAdjudicationPipeline::wire_parent_g26_switch` | captured_pipeline.rs:4511-4585 | Calls handle wire + body populator. |
| Rust | `g26_bridge_node` snapshot | captured_pipeline.rs:3915-3930 | Frontier handle for the predicate bridge kernel. |
| Rust | `engine.d_protocol_dt_dev_ptr()` | persistent_engine.rs:1439-1443 / fused_engine.rs:2289-2297 | The single canonical address for `d_protocol->dt`. **Static offset 84 inside `d_protocol_state`.** |
| Rust | `engine.set_gearbox_active(active)` | fused_engine.rs:3704; nhs_rt_full.rs:6186 | Gearbox > Adaptive-DT hierarchy guard; bypasses host write at fused_engine.rs:5606. |
| Rust | `ChronometricStateTensor` (32-byte device-resident) | gearbox.rs:46-89 | The persistent control state for G26. |
| Rust | `PipelineConfig.d_external_work: *mut f64` | captured_pipeline.rs:1003 | Source of `W_ext` for SFA's First-Law drift fuse — also our cheap energy-state proxy (see §5.4). |
| F1 (planned) | `prism_f1_predicate_bridge_kernel` | proposed adjudicator.cu append, per Agent 2 §3 P1 | Single-thread; reads `adj->adjudication_code` (offset 52) ; calls `cudaGraphSetConditional(handle, code & 0x3)`. |
| F1 (planned) | `prism_f1_create_handle_ffi`, `prism_f1_wire_with_handle_ffi`, `prism_f1_populate_switch_bodies_ffi`, `prism_f1_add_parent_bridge_ffi` | per Agent 2 §3 P1 | Parent-owned size-3 SWITCH (Prune / Construct / Violation). |
| WHILE (planned) | `prism_graph_create_while_handle_ffi`, `prism_graph_add_while_node_ffi` | per Agent 3 §5 P1 | Sit in `graph_node.cu`; `cudaGraphCondTypeWhile`, `size=1`. |
| WHILE (planned) | `prism_while_drain_predicate_bridge_kernel` | per Agent 3 §5 P4 (`while_drain_bridge.cu`) | Predicate body: `iter_count >= max ⇒ false`, drain remaining check. |

---

## 3. Reusable code (template that F1 + WHILE inherit from)

| Capability | Source | Reuse |
|---|---|---|
| Single-thread predicate bridge kernel | `prism_gearbox_predicate_bridge_kernel` (gearbox.cu:437-462) | F1 mirrors with `adj->adjudication_code` predicate (Agent 2 §7); WHILE mirrors with `iter_count >= max` clamp (Agent 3 §5 P4). |
| Parent-owned conditional handle | `prism_gearbox_create_handle_ffi` (adjudicator.cu:1119) | F1: `prism_f1_create_handle_ffi` (default=0/Prune); WHILE: `prism_graph_create_while_handle_ffi` (default=1/enter once). |
| Parent SWITCH/CONDITIONAL wirer | `prism_gearbox_wire_with_handle_ffi` (adjudicator.cu:1137) | F1: size=3 sibling; WHILE: type=While + size=1 sibling. |
| Body populator template | `prism_gearbox_populate_switch_bodies_ffi` (gearbox.cu:654-758) | F1: 3 bodies (Prune empty / Construct empty / Violation trap); WHILE: 1 body (drain DMA + fence). |
| Trap kernel for abort/violation | `prism_gearbox_trap_kernel` (gearbox.cu:587-592) | Reusable in F1 case 2 (Violation) and WHILE WatchdogTrip (Agent 2 §7; Agent 3 §9). |
| Frontier-snapshot pattern | `cuStreamGetCaptureInfo_v2` snapshot at captured_pipeline.rs:3915-3930 | F1's parent-side bridge node uses the same snapshot pattern (Agent 2 §5). |
| Splice-legality preflight | `prism_graph_is_splice_legal_ffi` (graph_node.cu:70+) / `splice_legality_check` (graph_node.rs:155-178) | Already counts conditional nodes ⇒ catches WHILE inside child template automatically (Agent 3 §3). |
| `Args` whitelist for run-config snapshots | `args_to_run_config_json` helper proposed at Agent 1 §7 | Same helper is the natural place to record the **gear table + SWITCH defaults** when a future DAG-2 captures control-plane provenance. |
| Per-stream NDJSON BufWriter | TIER 3b telemetry pattern (referenced at execution_policy.md:43 and ghost_telemetry.rs:536) | All three trace classes use this — see §6 for the unified writer. |
| Energy reducers | `prism_energy_monitor_launch_reduce` (captured_pipeline.rs:769-779) writes `adj->potential_energy` (offset 112); `*adj->d_external_work` (offset 128) | These supply `source_energy_state` cheaply (§5.4). |

---

## 4. Unsafe / problematic existing code

### 4.1 NO-GO finding: hot-path host write to `gear_override` ⚠

**Site:** [nhs_rt_full.rs:6755-6788](../crates/prism-nhs/src/bin/nhs_rt_full.rs).

**What it does:** At chunk boundaries crossing step 800 / 1000, the binary
calls `cuMemcpyHtoD_v2((adj_dev + 100), &value, 4)` to overwrite the
device-side `gear_override` byte. Value `0x00` forces Gear 0 (0.5 fs);
`0xFF` reverts to AUTO. The predicate bridge kernel
([gearbox.cu:454-459](../crates/prism-nhs/src/cuda/gearbox.cu)) then
short-circuits the SFA's calculated gear with the override on the next
replay.

**Why it is a violation of Invariant #1 (device-derived decisions only):**
This is a host CPU mutation of device control state inside the hot path.
While the *current intent* is "supervisor / hardware-interlock proof,"
its presence means the steady-state contract "no host writes to dt, gear,
branch, or iteration count" is **not yet enforceable**. Until this is
gated behind a debug flag or retired, any agent who reads CLAUDE.md and
greps for `cuMemcpyHtoD_v2.*adj` in the runtime will find a live
counter-example.

**Why it isn't catastrophic today:** The shim writes `adj->gear_override`
(offset 100) which is read by the bridge kernel — that read still happens
**on device**, and the actual dt commit still happens via
`prism_gearbox_apply_fixed_dt_kernel`. The host write is one byte of
*advisory input* to a device-derived decision. But the unified-trace
contract (§5) has no place for this host nudge unless we declare it
explicitly, which would entrench the violation.

**Mitigation (authored here, not implemented):**
1. Gate the entire shim block (lines 6755-6788) behind an explicit
   `args.force_gear_override_supervisor_shim: bool` (default `false`,
   undocumented in canonical `--help`). Operators who need the
   Hardware-Interlock proof opt in.
2. Emit a trace record of class `GearOverride` whenever the shim fires
   (host_mutation = true) — see §6. This makes the violation visible in
   the DAG and unambiguously labels the override as host-originated.
3. Fold the mid-campaign gear-force into a future device-side shim
   kernel: a 1-thread launch that takes a step-counter device pointer,
   compares against thresholds, and writes `gear_override` from device
   space. That removes the host write entirely. Concrete sketch in
   §5.7.

**Recommendation:** Land Agent 2's F1 first; introduce the trace schema
from §6 (with `host_mutation: bool`); then either gate or retire the
shim. The retirement is **not** in this lane's scope — Agent 11 is the
control-plane scout, not the integrator.

### 4.2 Other items inspected — no further violations

| Item | Site | Reason it is not a violation |
|---|---|---|
| `engine.set_dt(0.004)` host calls | nhs_rt_full.rs:1555, 1903, 4759 | Init-time only (pre-pipeline-build). Not in the hot path. |
| `--adaptive-dt` host write to `self.dt` | fused_engine.rs:5606-5612 | Already gated by `!self.gearbox_active` — **the existing correct invariant guard**. When V2 pipeline is live (`set_gearbox_active(true)` at nhs_rt_full.rs:6186), the host write is bypassed. Inverse-check: every monolithic-discovery code path sets `gearbox_active=true`. |
| `engine.download_protocol_state(...)` | nhs_rt_full.rs:8771 | Run-end DtoH for audit. Not in hot path. |
| `prism_gearbox_init_table_async` | gearbox.cu:153-170 | One-shot host→`__constant__` upload at pipeline build. Not in hot path. |
| `prism_zstr_set_active_slot` host call | nhs_rt_full.rs:6729-6733 | Slot rolling for ring-buffer telemetry; rotates a `__constant__` index, not control state (gear/branch/iter/dt). Tracked separately by Path Z. |
| Direct `sys::cuGraphAddDependencies` from Rust | graph_capture.rs:206 (referenced by Agent 2 §4) | Predates TIER 7 policy; tracked as a separate cleanup lane. Not part of control-plane flow. |

---

## 5. Exact plan

### 5.1 The control-plane discipline (six invariants)

These are **the** invariants every gear/branch/iteration decision must
obey. All three of G26 (live), F1 (Agent 2), and WHILE (Agent 3) inherit
them.

1. **Device-derived decisions only.** No host CPU writes back to device
   control state inside the hot path. The host may *read* traces; it
   must not *mutate* dt, gear, branch, or iteration count.
   - G26: Verified live. dt commits via `prism_gearbox_apply_fixed_dt_kernel`
     ([gearbox.cu:555-582](../crates/prism-nhs/src/cuda/gearbox.cu)) and
     `prism_gearbox_pointer_swap_kernel`
     ([gearbox.cu:144-148](../crates/prism-nhs/src/cuda/gearbox.cu));
     gear chosen by SFA on device.
   - F1: Verified by-design (Agent 2 §3 / §5). The new bridge reads
     `adj->adjudication_code` device-side; no host predicate.
   - WHILE: Verified by-design (Agent 3 §5 P4). Iteration counter is a
     device buffer; bridge clamps and writes via `cudaGraphSetConditional`.
2. **Parent-owned conditional handles only** (GRAPH-SPLICE-001).
   - G26: Live (Option β at TIER8_GRAPH_TOPOLOGY.md:25-54).
   - F1: Agent 2 §3 mandates duplicate-as-`f1_*`-siblings; child stays
     conditional-free.
   - WHILE: Agent 3 §"INVARIANTS WHILE MUST UPHOLD" mandates parent-only
     API; no `add_while_node_to_child_template` exists.
3. **Bounded WHILE.** Every WHILE has `max_iterations` and a watchdog.
   - WHILE: Agent 3 §"INVARIANTS WHILE MUST UPHOLD"; `WhileLoopConfig`
     panics on `max_iterations==0`.
   - G26 / F1: N/A (one-shot SWITCH per replay).
4. **Every decision emits a typed trace.** No silent decisions.
   - G26: TODAY there is **no per-decision trace**. The closest existing
     hook is `log_f1_switch_events` (host-side scraper of the pinned
     ring; ghost_telemetry.rs:536-566) which only logs on
     `adjudication_code == 1`. Gear decisions are not traced. **§6
     defines the canonical trace schema for all three.**
5. **Energy-state coupled.** Every decision cites a source energy state
   read on the same chunk so the trace is reconstructable.
   - G26: SFA already reads `adj->potential_energy` and
     `*adj->d_external_work` (gearbox.cu:240-249). The trace emit path
     in §5.4 reuses these device-side reductions — no new DtoH.
6. **Replay-safe.** Given the trace, the sequence of branches /gears /
   iterations is reproducible (modulo nondeterministic kernel order).
   - Replay-safe means: same `replica_seed`, same input topology, same
     gearbox table, and same predicate inputs ⇒ identical trace.
     Determinism caveats are pre-existing
     ([docs/PRISM_TWIN_DETERMINISM_INVARIANTS.md](PRISM_TWIN_DETERMINISM_INVARIANTS.md));
     this lane does not weaken them.

### 5.2 Single-typed-trace-file vs three sidecars — RECOMMEND ONE

**Recommendation:** **single typed trace file per stream**, not three
sidecars.

Justification:
1. **Causality reconstruction.** F1 fires *inside* a chunk, G26 fires
   *after* the child completes, WHILE iterates within a stage. To
   reconstruct "F1 said Construct, G26 said Gear 0 because adj_code=1
   came in mid-burst, drain WHILE ran 4 iterations" you need a single
   monotonic stream of records. Three sidecars force the consumer to
   re-merge by `(stream_id, chunk_id, frame_id)` post-hoc.
2. **Disk pressure.** Three writers per stream × N streams (8 today,
   target 20) = 24-60 file handles open per run. One writer × 8-20 = a
   tractable number.
3. **DAG ingestion.** Agent 1's `transform_dag.json` (DAG-2 follow-up)
   wants typed nodes per decision. A discriminated-union NDJSON record
   maps 1:1 to `DagNodeKind::G26Decision | F1Decision | WhileIteration`
   ([Agent 1 §5.4](DAG_FOUNDATION_IMPLEMENTATION_PLAN.md)). Three
   sidecars would force three readers.
4. **Schema versioning.** A single envelope can carry one
   `schema_version` and a `kind` discriminant per record; bumping the
   schema atomically across the whole control plane is cheaper than
   keeping three sidecar versions in sync.

The single file format is **NDJSON** (one record per line, externally
tagged as `{"kind":"g26","data":{...}} / {"kind":"f1","data":{...}} /
{"kind":"while","data":{...}} / {"kind":"gear_override","data":{...}}`),
written through a single `BufWriter` per stream. Path:
`<output_dir>/prism_control_plane_pid<pid>_stream<sid>.ndjson`.

This **supersedes** Agent 2 §6's per-class `prism_f1_decisions_*.ndjson`
filename. Recommended Agent 2 amendment: rename the F1 sidecar to the
unified file or make `f1_decision_writer` a thin adapter over a shared
`ControlPlaneTraceWriter`. **Cross-agent coordination note recorded in
§9.**

### 5.3 Where the trace emit lands

Two choices considered:

**A. Per-stream NDJSON in artifact dir (RECOMMENDED).**
- Mirrors TIER 3b telemetry persistence pattern referenced at
  [docs/EXECUTION_POLICY.md:43](EXECUTION_POLICY.md).
- Existing analogue: `log_f1_switch_events` already streams F1 burst
  events via `log::info!` from the per-chunk loop at
  nhs_rt_full.rs:6855-6862. Replacing its `log::info!` with a
  `BufWriter::write` to NDJSON is mechanical.
- Disk locality: lives next to `binding_sites.json`, ingested by Agent 1's
  DAG.
- Cost: ~120 bytes per record × ≤2 records/chunk × ≤2000 chunks/stream =
  ~480 KB / stream / run worst-case. Negligible.

**B. ZSTR / ghost ring carry the trace.**
- Existing infrastructure at
  [zstr.rs:93-100](../crates/prism-nhs/src/zstr.rs) already snapshots
  `force_norm`, `external_work`, `potential_energy` per frame
  (see [docs/ASC_OMNIDIRECTIONAL_IMPLEMENTATION_PLAN.md:56-114](ASC_OMNIDIRECTIONAL_IMPLEMENTATION_PLAN.md)).
- Adding three u32 fields (`gear_decision_trace_lo`, `f1_branch_taken`,
  `while_iter_count`) to `ZstrFrameHeader` would batch the trace with
  per-frame state for free.
- Drawback: ZSTR is a fixed-size 4096-byte header with strict
  ABI. Extending requires ABI bump and downstream consumers
  (zstr-archive readers, dossier scripts) to read the new fields. Out of
  scope for this lane.

**Decision: use (A) for v1; reserve (B) as a follow-up enrichment** when
the ZSTR ABI is next bumped. If/when ZSTR carries the trace, the NDJSON
becomes the *replicated* offline view (cheap sanity check; Agent 1's DAG
already consumes both). The two are not mutually exclusive.

### 5.4 How `source_energy_state` is read cheaply (no full DtoH)

The trace must record an energy snapshot per decision so the consumer
can reconstruct *why* the gear/branch/iteration decision was made. A
naive implementation would `cuMemcpyDtoH(d_pe_components)` (n_atoms × 8
bytes) per decision — pathological at high chunk count.

**Cheap path (REUSE):**

The energy reducer already runs **inside the captured pipeline**, before
the SFA, on every replay:

- `prism_energy_monitor_launch_reduce` (captured_pipeline.rs:769-779,
  3863-3881): performs a `cub::DeviceReduce::Sum` on `d_pe_components`
  and writes the scalar V_t into `adj->potential_energy` at offset 112.
- `*adj->d_external_work` (offset 128): F2-pool 8-byte counter,
  zeroed at chunk head via `cuMemsetD8Async`, atomic-added by UV / ASC
  / clamp paths. Holds W_ext for the chunk.

**Both fields are already in `InterferometricAdjudicatorFfi` at known
offsets** ([gearbox.cu:80-85, 246-249](../crates/prism-nhs/src/cuda/gearbox.cu)).
So the trace's `source_energy_state` is two u64-reads from
`adj_dev` — already DMA'd to host once per chunk via the **ring**
(`PinnedTelemetryRing<ContactShellTile>` already populated at chunk
boundary).

**Concrete implementation plan:**

Extend `ContactShellTile` (the per-frame DMA'd struct read at
nhs_rt_full.rs:6841-6852 and ghost_telemetry.rs:541-562) to carry two
new f64 fields: `potential_energy_post_chunk` and `external_work_post_chunk`.
Write them in the same Tier-1 populator that already emits
`adjudication_code` to the tile.

If extending `ContactShellTile` is itself a separate ABI lane,
**fall-back v1**: each control-plane writer issues **one 16-byte
`cuMemcpyDtoH` per decision** of `(potential_energy, external_work)`
read from `(adj_dev + 112, adj_dev + 128)`. 16 bytes × ≤2 decisions /
chunk × ≤2000 chunks = **64 KB total DtoH per stream per run**. Fits
inside the existing event-paced read pattern used by
`log_f1_switch_events` — no new sync barrier.

**Cite for Agent 4 reuse:** the same reducers are documented at
[ASC_OMNIDIRECTIONAL_IMPLEMENTATION_PLAN.md:96-107](ASC_OMNIDIRECTIONAL_IMPLEMENTATION_PLAN.md)
("`d_pe_components`", "`d_external_work`", "`EnergyWindow`").

### 5.5 How the trace integrates with Agent 1's DAG

Agent 1's `DagNodeKind` already declares the three node types
([DAG_FOUNDATION_IMPLEMENTATION_PLAN.md:280-283](DAG_FOUNDATION_IMPLEMENTATION_PLAN.md)):

```rust
#[non_exhaustive]
pub enum DagNodeKind {
    Run, InputTopology, RunConfig, Stream, Chunk, ProtocolPhase,
    GraphLaunch, G26Decision, F1Decision, WhileIteration,
    ASCSnapshot, KCCSnapshot, SpikeBatch, VoxelSupportSet,
    ClusterEvent, SiteCandidate, RankedSite, StaticEquivalentPocket,
    DossierArtifact, ValidationMetric, BinaryArtifact, JsonArtifact,
}
```

DAG-1 (Agent 1's first emission) deliberately does NOT populate these.
DAG-2 will. The contract this plan defines is:

- Each NDJSON record from §6 maps 1:1 to one `DagNode { kind:
  G26Decision | F1Decision | WhileIteration | GearOverride, ... }`.
- The DAG aggregator reads `<output>/prism_control_plane_pid*_stream*.ndjson`
  at run-end (or via an in-memory builder during the run) and emits
  `DagEdge`s of kind `Produces` from `Chunk(chunk_id)` to the decision
  node, plus `Supports` edges to `KCCSnapshot` / `ASCSnapshot` (when
  those are eventually populated).
- The `predicate_history_hash` field (Agent 3 §"WhileTrace schema")
  becomes a `DagInvariant.evidence_node_ids` entry referencing the
  predicate-bridge kernel node by its captured-graph node id. This is
  the replay-safety witness.

**No DAG schema bump is needed** — Agent 1's enum already supports all
three kinds with `#[non_exhaustive]`. This lane only fixes the *NDJSON
ingestion shape* the DAG aggregator must read.

### 5.6 Static assertion that `host_mutation = false`

This is the load-bearing question for Invariant #1. Three options:

**Option 1: compile-time `static_assert`.**
- Not directly applicable — Rust has no `static_assert!` for runtime
  values, and the host-mutation prohibition is a *runtime* property of
  the execution path.
- Could be approximated by feature-gating the `gear_override` write
  path behind `#[cfg(feature = "gear_override_shim")]`. CI builds the
  default profile without the feature; any usage in production builds
  would fail to compile. Cost: one Cargo feature, a `cfg` block, a CI
  check. **Recommended companion to Option 3.**

**Option 2: runtime `debug_assert!`.**
- Pattern used elsewhere in the captured pipeline (e.g., null-pointer
  defenses at gearbox.cu:443, 508, 567). Existing analogue in Rust:
  `debug_assert!(self.gearbox_active || !self.adaptive_dt_enabled, ...)`
  in `step_with_clusters` would catch a re-introduction of the
  adaptive-dt host write while V2 is live.
- Site to add: fused_engine.rs:5606, immediately before the existing
  `if self.adaptive_dt_enabled && !self.gearbox_active` guard. Adding a
  `debug_assert!(!(self.gearbox_active && self.adaptive_dt_enabled))`
  formalises the existing conditional as an assertion.
- **Recommended.**

**Option 3: code-review gate (grep-based).**
- Single grep that catches new violations:
  ```
  rg -n 'cuMemcpyHtoD.*\b(adj_dev|d_protocol_dt|gear_override)\b' \
     crates/prism-nhs/src/bin/ crates/prism-nhs/src/captured_pipeline.rs
  ```
  Every match must be either (a) at pipeline build time (one-shot init,
  outside the chunk loop) or (b) the supervisor shim at lines 6755-6788
  (which §4.1 already flags for gating/retirement). Any new match is
  a violation.
- **Recommended.** A pre-commit hook or CI step that runs this grep
  and fails on novel matches is cheap and uncheatable.

### 5.7 Migration sketch for the `gear_override` shim (NOT in this lane's scope)

For completeness — the path to retiring the §4.1 host write:

1. Add a 1-thread `prism_gearbox_apply_force_gear_kernel(adj_dev,
   d_step_counter, d_thresholds[2], d_force_gears[2])` inside
   `gearbox.cu`. Reads the device-side step counter and writes
   `adj->gear_override` from device space when crossings happen.
2. Allocate `d_step_counter` and `d_thresholds` at engine init; populate
   thresholds once, increment counter once per chunk via a tiny capture
   node.
3. Remove the host `cuMemcpyHtoD_v2` block at nhs_rt_full.rs:6755-6788.
4. The trace emits `GearOverride` records when the kernel fires — same
   schema as today's `host_mutation=true` placeholder, but flipped to
   `host_mutation=false` because the write is now device-derived.

This is sketched, not blocked-on.

---

## 6. Required output fields (unified trace schema)

**Verbatim names from the directive — flag conflicts.** Conflict scan:
`rg -n "(gear_decision_trace|gear_predicate_value|dt_before|dt_after|source_energy_state|switch_branch_taken|next_chunk_consumed_dt|host_mutation|f1_branch_taken|iteration_count|max_iterations|exit_reason|while_node_id|predicate_history_hash|watchdog_status)" crates/`
returned **0 hits** (all names are clean to introduce; no existing field
or struct collisions).

### 6.1 Common fields (every G26 / F1 / WHILE decision)

| Field | Type | Source | Notes |
|---|---|---|---|
| `gear_decision_trace` | `String` | `format!("{stream_id}-{chunk_id}-{frame_id}-{kind}")` | Unique decision id. For F1 / WHILE the "gear" prefix is a misnomer accepted from the directive; documented as the unified record id. |
| `gear_predicate_value` | `u64` | G26: `cruise->current_gear` (low 2 bits) ‖ `(adj->gear_override as u64) << 2`. F1: `adj->adjudication_code as u64`. WHILE: `iter_count as u64`. | Raw predicate bits at the moment of decision. Padded to u64 so a single sidecar can carry all three. |
| `dt_before` | `f32` | Snapshot of `*d_protocol->dt` BEFORE the decision (read from the prior chunk's tail value cached in `cruise->_dt_pre`; can be reconstructed from `previous_gear`). | For F1 / WHILE: identical to `dt_after` (no dt mutation). |
| `dt_after` | `f32` | `d_gearbox_table[final_gear * 4]` (G26); `dt_before` (F1 / WHILE). | Proves the dt commit. |
| `source_energy_state` | `EnergyState { total_pe: f64, total_ke: f64, w_ext: f64 }` | `adj->potential_energy` (offset 112) + `*adj->d_external_work` (offset 128). `total_ke` derived from velocity reduction (TODO: today we only have PE + W_ext; KE reducer is in scope of Agent 4's ASC plan). | See §5.4. v1 may emit `total_ke = 0.0` with a `ke_unmeasured: true` flag in `metadata`. Document the gap. |
| `switch_branch_taken` | `u32` | G26: `final_gear` (0..3). F1: `f1_branch_taken` (0..2). WHILE: ignored (set 0 sentinel). | Sentinel for WHILE keeps the schema homogenous. |
| `next_chunk_consumed_dt` | `f32` | The dt that the *next* chunk will consume = `*d_protocol->dt` after the SWITCH body's `apply_fixed_dt` ran. For G26 = `dt_after`. For F1 / WHILE = `dt_before` (no dt mutation). | Closes the replay invariant: replaying the trace must produce the same `next_chunk_consumed_dt` byte-for-byte. |
| `host_mutation` | `bool` | Constant `false` for G26 / F1 / WHILE in steady state. **`true`** for `GearOverride` records emitted by the §4.1 shim. | The presence of the field is the policy. Static assertion patterns at §5.6. |

### 6.2 Per-class extensions

```rust
// crates/prism-nhs/src/control_plane_trace.rs (proposed home; new file)
// Single envelope; serde-tagged enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum ControlPlaneRecord {
    G26 {
        // Common (§6.1):
        gear_decision_trace: String,
        gear_predicate_value: u64,    // (current_gear & 0x3) | (gear_override << 2)
        dt_before: f32,
        dt_after: f32,
        source_energy_state: EnergyState,
        switch_branch_taken: u32,     // = final_gear
        next_chunk_consumed_dt: f32,
        host_mutation: bool,          // false in steady state

        // G26-specific:
        prev_gear: u32,
        sfa_counter: u32,             // ChronometricStateTensor.counter
        sfa_v_prev: f64,              // ChronometricStateTensor.v_prev
        sfa_drift_tripped: bool,
        gear_override_was_applied: bool,
    },
    F1 {
        gear_decision_trace: String,
        gear_predicate_value: u64,    // = adj.adjudication_code
        dt_before: f32,
        dt_after: f32,
        source_energy_state: EnergyState,
        switch_branch_taken: u32,     // = f1_branch_taken (0..2)
        next_chunk_consumed_dt: f32,
        host_mutation: bool,          // false

        // F1-specific (Agent 2 §6):
        f1_branch_taken: u32,         // 0=Prune, 1=Construct, 2=Violation
        f1_condition_handle: u64,
        f1_branch_trace_hash: u64,    // BLAKE3-low-64
        f1_exit_status: u8,           // 0=clean, 1=trap, 2=launch error
    },
    While {
        gear_decision_trace: String,
        gear_predicate_value: u64,    // = iter_count
        dt_before: f32,
        dt_after: f32,
        source_energy_state: EnergyState,
        switch_branch_taken: u32,     // 0 sentinel (unused for WHILE)
        next_chunk_consumed_dt: f32,
        host_mutation: bool,          // false

        // WHILE-specific (Agent 3 §"WhileTrace schema"):
        while_node_id: u64,
        entry_step: u64,
        exit_step: u64,
        iteration_count: u32,
        max_iterations: u32,
        exit_reason: WhileExitReason,
        predicate_history_hash: u64,
        watchdog_status: u32,
    },
    GearOverride {
        // §4.1 — host-mutation record, emitted ONLY when the supervisor
        // shim fires. Its presence in the trace is the violation report.
        gear_decision_trace: String,
        gear_predicate_value: u64,    // = override_value (0..3 or 0xFF)
        dt_before: f32,
        dt_after: f32,                // unchanged (override is advisory)
        source_energy_state: EnergyState,
        switch_branch_taken: u32,     // intended target gear
        next_chunk_consumed_dt: f32,
        host_mutation: bool,          // ALWAYS true here

        // GearOverride-specific:
        crossed_step: u32,
        override_value: u32,
        host_writer_callsite: String, // file:line of the cuMemcpyHtoD
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EnergyState {
    pub total_pe: f64,
    pub total_ke: f64,        // 0.0 sentinel until KE reducer lands
    pub w_ext:    f64,
    pub ke_unmeasured: bool,  // v1 gap; flips to false in v2
}
```

NDJSON one-line-per-record example:
```json
{"kind":"g26","data":{"gear_decision_trace":"0-15-1500-g26","gear_predicate_value":1,"dt_before":0.002,"dt_after":0.0005,"source_energy_state":{"total_pe":-12345.6,"total_ke":0.0,"w_ext":0.123,"ke_unmeasured":true},"switch_branch_taken":0,"next_chunk_consumed_dt":0.0005,"host_mutation":false,"prev_gear":1,"sfa_counter":42,"sfa_v_prev":-12340.0,"sfa_drift_tripped":false,"gear_override_was_applied":false}}
```

### 6.3 Persistence path & writer

- File: `<output_dir>/prism_control_plane_pid<pid>_stream<sid>.ndjson`.
- Writer: one `BufWriter<File>` per stream, owned by the per-stream
  spawn closure in `nhs_rt_full.rs`. Lives next to the existing
  `log_f1_switch_events` call site
  ([nhs_rt_full.rs:6855-6862](../crates/prism-nhs/src/bin/nhs_rt_full.rs)).
- Flush policy: `flush_on_drop()` + per-record write; no explicit fsync.
  Run-end Drop is guaranteed by Rust ownership.
- Cost: ≈4 KB / chunk worst-case (one G26 + one F1 + one WHILE = three
  records ≤120 bytes each, plus ≤1 GearOverride). Well under the
  200 µs / chunk budget.

---

## 7. New structs / functions

This lane introduces **types only**. Helper functions and integration
sites belong to Agent 2 / Agent 3 / a future Agent 11 implementer.

### Rust (new file: `crates/prism-nhs/src/control_plane_trace.rs`)

| Symbol | Kind | LOC est. | Purpose |
|---|---|---|---|
| `ControlPlaneRecord` (enum) | externally-tagged serde enum | ~80 | Single record type for G26 / F1 / WHILE / GearOverride. |
| `EnergyState` (struct) | serde struct | ~10 | total_pe + total_ke + w_ext + ke_unmeasured flag. |
| `WhileExitReason` (enum) | re-exported from `graph_node::while_trace` (Agent 3 §3) | 0 (re-export) | Avoids duplication. |
| `ControlPlaneTraceWriter` | per-stream BufWriter wrapper | ~40 | `emit(record: ControlPlaneRecord)`, `flush_on_drop()`. |
| `make_decision_trace_id(stream_id, chunk_id, frame_id, kind: &str)` | pub fn | ~5 | Format helper: `"<sid>-<chunk>-<frame>-<kind>"`. |
| `read_energy_state_from_ring(tile: &ContactShellTile)` | pub fn | ~10 | When ContactShellTile carries PE/W_ext (§5.4 fast path). |
| `read_energy_state_dtoh(adj_dev: u64)` | pub unsafe fn | ~20 | v1 fall-back: 16-byte cuMemcpyDtoH from `(adj_dev+112, adj_dev+128)`. |

### Rust (modification: `crates/prism-nhs/src/lib.rs`)

| Symbol | Kind | LOC | Purpose |
|---|---|---|---|
| `pub mod control_plane_trace;` | module re-export | 1 | Public surface; consumed by Agent 1's DAG aggregator. |

### CUDA / C++ (no new symbols)

This lane adds **no new CUDA symbols**. G26's `prism_gearbox_predicate_bridge_kernel`
remains unchanged. F1's bridge kernel and WHILE's drain bridge are owned
by Agent 2 / Agent 3.

### Bin (`nhs_rt_full.rs`) — call sites the implementer must add

| Site | Action | Owner |
|---|---|---|
| Per-stream spawn closure | Construct `ControlPlaneTraceWriter::new(&out_path, stream_id)`. Lifetime ends at stream-spawn close. | Agent 11 (future implementer). |
| Post-launch `log_f1_switch_events` site (l. 6855-6862) | Replace with three writer calls: G26, optional F1, optional WHILE. | Agent 2 (F1 records); future implementer (G26 records). |
| §4.1 supervisor shim (l. 6755-6788) | Replace `log::info!` with `writer.emit(ControlPlaneRecord::GearOverride { host_mutation: true, .. })`; also gate behind `args.force_gear_override_supervisor_shim`. | Future implementer. |

**Net minimal-touch when Agent 2 / Agent 3 land:** ~30 lines per stream
spawn site, additive only.

---

## 8. Acceptance tests

All under `--m1-monolithic-discovery --multi-stream 8 --hysteresis
--no-autonomous-rescue` unless noted.

| # | Check | Method | Pass criterion |
|---|---|---|---|
| A1 | Schema compiles | `cargo check -p prism-nhs` | exit 0 |
| A2 | Roundtrip serde | `cargo test -p prism-nhs control_plane_trace::tests::roundtrip` | exit 0 |
| A3 | Tagged enum field-names match §6 verbatim | `jq '.data \| keys' <ndjson> \| sort -u` per kind | every required field present, no extras |
| A4 | One trace file per stream | `ls <output>/prism_control_plane_pid*_stream*.ndjson \| wc -l` | == n_streams |
| A5 | G26 records emitted on every chunk | `jq -c 'select(.kind=="g26")' <ndjson> \| wc -l` | matches chunks executed |
| A6 | `host_mutation == false` for every G26/F1/WHILE record | `jq -c 'select(.kind!="gear_override") \| .data.host_mutation'` | all `false`; zero `true` |
| A7 | If §4.1 shim disabled (`force_gear_override_supervisor_shim=false`) | `jq -c 'select(.kind=="gear_override")' <ndjson> \| wc -l` | 0 |
| A8 | If §4.1 shim enabled, GearOverride records carry `host_mutation: true` | `jq -c 'select(.kind=="gear_override") \| .data.host_mutation'` | all `true` |
| A9 | Replay determinism | run twice with `--replica-seed 42`; diff sidecars | byte-identical |
| A10 | DAG ingestion shape (DAG-2 prerequisite) | Agent 1 aggregator reads NDJSON ⇒ produces `DagNode { kind: G26Decision, .. }` | one DAG node per record |
| A11 | Static assertion: `debug_assert!(!(gearbox_active && adaptive_dt_enabled))` fires under contradiction | unit test sets both flags | panics in `cfg(debug)`; passes in release without the assertion |
| A12 | Grep gate for novel host writes to control state | `rg -n 'cuMemcpyHtoD.*\b(adj_dev|d_protocol_dt|gear_override)\b' crates/prism-nhs/src/bin/ crates/prism-nhs/src/captured_pipeline.rs` | only the §4.1 shim site; any new match fails CI |
| A13 | TIER 8 freeze evidence unchanged | Re-run all-8 gate (TIER8_GRAPH_TOPOLOGY.md:113-116) | `parent_conditional_nodes` count unchanged from current 1 (G26 only) until F1+WHILE land |
| A14 | Trace cost budget | `wc -c <ndjson>` | ≤ 1 MB / stream / 2000-chunk run |

**End-to-end smoke (post-merge):**
1. Run a short canonical pipeline against `1bzj.topology.json`.
2. Assert `<output>/prism_control_plane_pid*_stream*.ndjson` exists.
3. `jq -c '.kind' <ndjson> | sort | uniq -c` shows at least `g26`
   records.
4. Replay the same run with `--replica-seed 42` twice; diff the two
   ndjsons → expect empty diff.

---

## 9. Failure modes

| Mode | Symptom | Mitigation |
|---|---|---|
| Trace writer disk full | `BufWriter::write` Err | Warn-only emission per Agent 1 §9 mitigations. Run continues; missing records are tolerated by the DAG aggregator. |
| KE reducer not yet present (v1 gap) | `total_ke = 0.0`, `ke_unmeasured = true` in every record | Documented; Agent 4's ASC plan provides the KE reducer in a follow-up; v2 schema flips `ke_unmeasured` to `false`. |
| ContactShellTile does not (yet) carry PE/W_ext | Fall-back DtoH path used (§5.4 v1) adds 16 B / decision | Documented; cost is bounded (~64 KB / stream / run); upgrade path is the ContactShellTile ABI bump. |
| Cross-agent rename: F1's per-class sidecar (Agent 2 §6) vs unified file (this plan §5.2) | Two writer classes ⇒ disk thrash, schema drift | **Coordination required** with Agent 2: rename `F1DecisionWriter` → `ControlPlaneTraceWriter` adapter. See §"Cross-agent coordination" below. |
| Determinism leaks via timestamps | `ts_us` inside the trace forces non-byte-identical replays | Schema in §6 deliberately omits wall-clock timestamps. Reconstruction uses `(stream_id, chunk_id, frame_id)`. The host `log_f1_switch_events` `ts_us` ([ghost_telemetry.rs:548-551](../crates/prism-nhs/src/ghost_telemetry.rs)) stays in `log::info!`, NOT in the NDJSON. |
| Schema bump silently drifts | Older readers fail to parse | NDJSON envelope carries `schema_version: u32` at the top of each record; readers reject records with a higher version they don't understand. v1 = 1. |
| §4.1 shim re-enabled by accident in production | Hot-path host write reappears | `force_gear_override_supervisor_shim` flag gates compilation in addition to runtime; CI grep gate (Test A12) fails on any novel call site. |
| Predicate-history-hash collision (replay drift) | Two distinct iteration histories hash to same u64 | BLAKE3-low-64 truncation has 2⁻³² collision probability per record. Acceptable for replay verification (matches `lineage_integrity_hash` posture at nhs_rt_full.rs:9947 and Agent 1 §3 hash policy). |
| Multi-stream concurrency: writers race on a shared file | None — each stream owns its own file | Per-stream NDJSON path includes `<sid>` ⇒ no shared writer. |
| Parent SWITCH wired but bridge kernel mis-snapshots | Trace records the wrong predicate | Agent 2 §9 + Agent 3 §9 already enumerate this failure mode; the trace itself does not introduce a new mode. |

### Cross-agent coordination

| Agent | Conflict | Resolution |
|---|---|---|
| Agent 1 (DAG) | `transform_dag.json` v1 deliberately empty of decision nodes; v2 needs to consume the NDJSON shape from §6 | This plan defines the shape Agent 1's v2 aggregator reads. No v1 change required. |
| Agent 2 (F1) | Agent 2 §6 proposes a per-class `prism_f1_decisions_*.ndjson` writer | **Recommended amendment**: Agent 2 implements `F1DecisionWriter` as a thin adapter over `ControlPlaneTraceWriter::emit(ControlPlaneRecord::F1 { .. })`. File path becomes `prism_control_plane_*.ndjson` not `prism_f1_decisions_*.ndjson`. Agent 2 §6's eight fields are **all preserved** as the F1 variant in §6.2. |
| Agent 3 (WHILE) | Agent 3 §"WhileTrace schema" defines `WhileTrace` as a struct, not a record | **Recommended amendment**: Agent 3's `WhileTrace` becomes the body of `ControlPlaneRecord::While { .. }`. Field names match exactly. The trace ring proposed in Agent 3 §5 P3 still exists (device-side per-iteration writes), and the run-end aggregator drains it into the NDJSON. |
| Agent 4 (ASC) | ASC's KE reducer | Adds `total_ke` to the v2 EnergyState. Coordinated by `ke_unmeasured` field in v1. |

---

## 10. Rollback

This lane is **additive only**. The single deliverable is a doc.

If the trace types land in code in a follow-up commit and a regression
is observed:

1. **Drop the writer at the call site.** Replace
   `writer.emit(...)` with a no-op `let _ = ();`. Trace stops emitting,
   pipeline continues. Single-line revert.
2. **Disable the module re-export.** Remove `pub mod
   control_plane_trace;` from `lib.rs`. The new file becomes dead code.
3. **Revert the new file.** `git rm
   crates/prism-nhs/src/control_plane_trace.rs`. Single-commit revert.
4. **No graph-topology change** — this lane introduces no new
   conditional / SWITCH / WHILE nodes. The all-8 freeze evidence
   ([TIER8_GRAPH_TOPOLOGY.md:68-97](TIER8_GRAPH_TOPOLOGY.md)) is
   unaffected by this lane.
5. **The §4.1 shim survives** the rollback — it is independently in the
   code today. Retiring or gating it is a separate decision.

The kill switch is **two locations**: the `pub mod` line and the
per-stream writer construction. Either alone disables the trace.

---

## Coordination with Agent 1 / 2 / 3 (summary)

- **Agent 1 (DAG-1):** Reuses `DagNodeKind { G26Decision, F1Decision,
  WhileIteration }` already declared
  ([DAG_FOUNDATION_IMPLEMENTATION_PLAN.md:280-283](DAG_FOUNDATION_IMPLEMENTATION_PLAN.md)).
  DAG-2 will read the NDJSON sidecar this plan defines. **No v1 change.**
- **Agent 2 (F1):** §6 of Agent 2's plan proposes a per-class sidecar.
  This plan recommends Agent 2's writer become a thin adapter over the
  unified `ControlPlaneTraceWriter`. Field names from Agent 2 §6 are
  preserved verbatim under `ControlPlaneRecord::F1 { .. }`. Agent 2's
  patch sequence (§5 P1-P11) is otherwise unchanged.
- **Agent 3 (WHILE):** Agent 3's `WhileTrace` struct
  ([WHILE_NODE_IMPLEMENTATION_PLAN.md:204-217](WHILE_NODE_IMPLEMENTATION_PLAN.md))
  becomes the body of `ControlPlaneRecord::While { .. }`. Field names
  match. Agent 3's bounded-iter / watchdog discipline is preserved; this
  plan only standardises the persistence sidecar.
- **Agent 4 (ASC):** When the KE reducer lands, `EnergyState.total_ke`
  flips from sentinel 0.0 to the real value and `ke_unmeasured` flips
  to `false`. Schema bump v1→v2 only.

---

## Final summary (per the directive)

- **G26 predicate bridge call site:**
  `prism_gearbox_predicate_bridge_kernel` at
  [crates/prism-nhs/src/cuda/gearbox.cu:437-462](../crates/prism-nhs/src/cuda/gearbox.cu);
  host shim `prism_gearbox_launch_predicate_bridge` at
  [crates/prism-nhs/src/cuda/gearbox.cu:464-477](../crates/prism-nhs/src/cuda/gearbox.cu).
  Captured into the child template at
  [crates/prism-nhs/src/captured_pipeline.rs:3899-3913](../crates/prism-nhs/src/captured_pipeline.rs).
  SWITCH `size = 4`. Predicate read: `cruise->current_gear` (offset 8 of
  ChronometricStateTensor) with `adj->gear_override` (offset 100)
  shadow.
- **d_dt write/read (device-side confirmed):**
  - **Write:** `prism_gearbox_apply_fixed_dt_kernel` at
    [crates/prism-nhs/src/cuda/gearbox.cu:555-582](../crates/prism-nhs/src/cuda/gearbox.cu)
    writes `d_gearbox_table[gear*4]` to `*(adj->d_dt)` (offset 120).
    Also `prism_gearbox_pointer_swap_kernel` at
    [gearbox.cu:144-148](../crates/prism-nhs/src/cuda/gearbox.cu).
    **Both are device kernels.**
  - **Read:** Every integrator kernel reads `d_protocol->dt` at
    `d_protocol_state[84..88]`
    ([crates/prism-nhs/src/fused_engine.rs:2274-2297](../crates/prism-nhs/src/fused_engine.rs)).
    No DtoH-then-HtoD round-trip; the address is wired once at pipeline
    build at
    [crates/prism-nhs/src/bin/nhs_rt_full.rs:5744-5816](../crates/prism-nhs/src/bin/nhs_rt_full.rs).
    Host-side adaptive-dt write is bypassed when `gearbox_active=true`
    ([fused_engine.rs:5602-5612](../crates/prism-nhs/src/fused_engine.rs)).
- **Trace schema decision:** **Single typed trace file per stream**
  (NDJSON) at
  `<output_dir>/prism_control_plane_pid<pid>_stream<sid>.ndjson`. One
  externally-tagged serde enum (`ControlPlaneRecord`) carrying G26 / F1
  / WHILE / GearOverride records. Justification: causality
  reconstruction, disk pressure, DAG ingestion, schema atomicity. See
  §5.2.
- **Static assertion pattern for `host_mutation = false`:** Layered.
  (1) Cargo `feature = "gear_override_shim"` gating compilation of the
  §4.1 host write block. (2) `debug_assert!(!(self.gearbox_active &&
  self.adaptive_dt_enabled))` in `step_with_clusters` formalising the
  existing guard at fused_engine.rs:5606. (3) CI grep gate
  `rg 'cuMemcpyHtoD.*\b(adj_dev|d_protocol_dt|gear_override)\b'`
  rejects novel hot-path host writes. See §5.6.
- **Control-plane defects found:** ONE — the **Force-Gear Orchestration
  Supervisor Shim** at
  [crates/prism-nhs/src/bin/nhs_rt_full.rs:6755-6788](../crates/prism-nhs/src/bin/nhs_rt_full.rs).
  Hot-path `cuMemcpyHtoD_v2` to `(adj_dev + 100)` (`gear_override`).
  Live counter-example to Invariant #1. Mitigation: gate behind a
  feature flag and emit an explicit `GearOverride` trace record with
  `host_mutation: true`; retire by migrating to a 1-thread device
  kernel. See §4.1.
