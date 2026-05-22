# WHILE Conditional Node — Implementation Plan (Scout)

Author: Agent 3 (read-only scout)
Date: 2026-05-04
Branch: `producer-repair-causal-truthing-20260426`
Baseline commit: `8ca26189`

Status: PROPOSAL — no code edited; one new doc only.

## Required-finding answers (citations first)

1. **WHILE FFI exposure: NO.** Neither `crates/prism-nhs/src/cuda/graph_node.cuh:42-162` nor
   `crates/prism-nhs/src/cuda/graph_node.cu:1-191` defines a WHILE helper.
   `graph_node.cu` only emits `cudaGraphNodeTypeGraph` (line 43) and
   `cudaGraphNodeTypeMemset` (line 174). The conditional path that exists
   is the SWITCH (G26) family in `crates/prism-nhs/src/cuda/adjudicator.cu`
   with `cudaGraphCondTypeSwitch` at lines 908, 1045, 1157. There is no
   call site anywhere in `prism-nhs` that uses `cudaGraphCondTypeWhile`
   (grep `crates/prism-nhs/ -e cudaGraphCondTypeWhile` → 0 hits;
   only the vendored cudarc enum at `vendor/cudarc/src/runtime/sys/mod.rs:3735`
   and `:3747` references it). The missing surface is:
     - `cudaGraphCondTypeWhile` discriminator (CUDA enum value 1) is not
       referenced from any PRISM TU.
     - There is no helper that does
       `cudaGraphConditionalHandleCreate(graph, defaultLaunchValue=1)` +
       `cudaGraphAddNode(type=Conditional, conditional.type=cudaGraphCondTypeWhile, conditional.size=1)`.
     - The `cudaGraphConditionalHandleFlags::cudaGraphCondAssignDefault`
       value is already used for SWITCH (`adjudicator.cu:883`,`:1025`,`:1131`),
       so flag plumbing exists; only the `type` arm needs to be set to WHILE
       and `size` clamped to **1** per CUDA contract.

2. **CUDA + driver version support.**
   - System nvcc: `13.2.51` (release 13.2, build cuda_13.2.r13.2/compiler.37434383_0).
   - Driver: `595.45.04`; GPU: RTX 5080 Blackwell, compute capability `12.0`
     (sm_120 per `crates/prism-nhs/build.rs:327` and `:418`).
   - WHILE conditional nodes were introduced in CUDA 12.4
     (`#if CUDART_VERSION >= 12040` is the gate already used for the
     SWITCH-based set-conditional kernel at `adjudicator.cu:803`).
     WHILE specifically is documented in CUDA 12.4 (Programming Guide
     §3.2.8.7.4) and is fully supported on CUDA 13.x with no Blackwell-
     specific erratum.
   - System headers confirm: `/usr/local/cuda/include/driver_types.h:3560`
     defines `cudaGraphCondTypeWhile = 1` with the `size==1` constraint
     literally written out at line 3572 — "Allowed values are 1 for
     cudaGraphCondTypeWhile". sm_120 is supported by the cuda 13.2
     toolkit; no driver-side block.
   - **Minimum CUDA toolkit for cudaGraphNodeTypeConditional + WHILE:
     12.4.** PRISM ships on 13.2; the gate already in use for SWITCH
     (`#if CUDART_VERSION >= 12060`, `adjudicator.cu:863`,`:999`,`:1117`)
     is more conservative than necessary for WHILE; `>= 12040` is the
     correct minimum. The new helper should mirror the existing
     `#if CUDART_VERSION >= 12040 ... #else cudaErrorNotSupported ... #endif`
     pattern at `adjudicator.cu:803-846`.

3. **Missing structs / enums to support WHILE.**

   In `crates/prism-nhs/src/cuda/graph_node.cuh`:
     - `prism_graph_create_while_handle_ffi(parent_graph, default_value, *out_handle)`
       — wraps `cudaGraphConditionalHandleCreate` with
       `cudaGraphCondAssignDefault`. (Mirrors
       `prism_gearbox_create_handle_ffi` at `adjudicator.cu:1119`.)
     - `prism_graph_add_while_node_ffi(parent_graph, predicate_node, handle_v,
       *out_conditional_node, *out_body_subgraph /* size 1 */)` — the only
       new graph-builder signature; sets `nodeParams.conditional.type =
       cudaGraphCondTypeWhile`, `conditional.size = 1u`, captures
       `phGraph_out[0]` into the caller's body slot. (Mirrors
       `prism_gearbox_wire_with_handle_ffi` at `adjudicator.cu:1137`,
       parameterized down from 4 bodies to 1.)

   In `crates/prism-nhs/src/graph_node.rs`:
     - `extern "C"` blocks for the two helpers above.
     - `pub struct WhileTrace { ... }` and `pub enum WhileExitReason { ... }`
       (see schema below — verbatim from the prompt). Flag-conflict scan:
       `grep -rn "WhileTrace\|WhileExitReason\|while_node_id" crates/`
       returned 0 hits, so the names are clean.
     - `pub unsafe fn create_while_handle(parent_graph: CUgraph,
       default_value: u32) -> Result<u64, i32>` — direct FFI wrapper.
     - `pub unsafe fn add_while_node(parent_graph: CUgraph, predicate_node:
       CUgraphNode, handle: u64) -> Result<(CUgraphNode, CUgraph), SpliceError>`
       — returns `(while_cond_node, body_subgraph)`; reuses `SpliceError` so
       the legality preflight contract is symmetric with the GRAPH-splice
       path at `graph_node.rs:155-178`.
     - `pub struct WhileLoopConfig { max_iterations: u32, watchdog_step_budget: u32, ... }`
       — stored on the parent graph builder so the iteration cap is
       enforced **at predicate-bridge build time**, not at runtime.
     - `pub struct WhileTrace { ... }` (see Section §"WhileTrace schema").

4. **Generalizing the G26 parent-owned helper, or sibling helper?**

   **Sibling helper.** Justification:
     - `prism_wire_g26_gearbox_ffi` (`adjudicator.cu:1001-1078`) and
       `prism_gearbox_wire_with_handle_ffi` (`adjudicator.cu:1137-1182`)
       both hard-code `nodeParams.conditional.type = cudaGraphCondTypeSwitch`
       and `conditional.size = 4u`. The SWITCH 4-way semantic is fused
       into the function contract: the caller receives a `CUgraph[4]`
       body-subgraph array and a `prism_gearbox_populate_switch_bodies_ffi`
       (`gearbox.cu:654`) populator that assumes 4 bodies (rescale × 3 +
       trap × 1).
     - WHILE has `size = 1` (one body, looped) — the `phGraph_out` array
       is a single-element handle, and the body topology is "predicate
       check + bounded body" rather than "switch-on-discriminator". A
       generalized FFI that accepts `cond_type` + `size` would expose the
       union arm without the per-call-site invariants (size==1 for WHILE)
       that the type-system can otherwise enforce. The G26 helper would
       also have to grow a ctor flag for the 4-body populator vs.
       single-body WHILE topology — that's two functions wearing the same
       name.
     - The existing C++ code style uses one `prism_wire_<feature>_ffi` per
       conditional node class (see F1 SWITCH at `adjudicator.cu:865`, G26
       SWITCH at `:1001`). A `prism_graph_add_while_node_ffi` sibling in
       `cuda/graph_node.cu` keeps the convention and gives the WhileTrace
       hook a clean home.

5. **Where does WHILE live: parent or child?**

   **Parent-owned only.** Code evidence:
     - `docs/TIER8_GRAPH_TOPOLOGY.md:8-11`: "GRAPH-SPLICE-001: any graph
       passed to `cudaGraphAddNode(GRAPH)` as a child template must
       contain no conditional, allocation, or free nodes."
     - `crates/prism-nhs/src/graph_node.rs:96-100`: `SpliceLegalityReport::is_legal`
       requires `conditional_count == 0` for a child template. Adding a
       WHILE node inside a spliceable child would trip the same TIER 8
       preflight that already gates SWITCH (the symmetric failure mode is
       why G26 was lifted to the parent — see
       `captured_pipeline.rs:3803-3811` "TIER 8 Option β" and the
       parent-handle creator at `:4470-4497`).
     - The existing `create_parent_g26_cond_handle` /
       `wire_parent_g26_switch` pair (`captured_pipeline.rs:4470` and
       `:4511`) is the canonical template; WHILE follows the same Option
       β topology, with the predicate-bridge kernel captured inside the
       child so it can read device-resident state and call
       `cudaGraphSetConditional(handle, predicate_value)`, while the
       conditional node itself is installed on the parent post-splice.

6. **First SAFE WHILE use case in PRISM today: bounded post-T7 drain.**

   Recommendation: **bounded post-T7 telemetry drain** — fold the
   `TIER8-CAPTURE deferred-error-state … result=DRAINED …
   deferred_drain_count=1/2` behaviour (`docs/TIER8_GRAPH_TOPOLOGY.md:101-107`,
   `captured_pipeline.rs:295-360`) from a host-loop into a parent-owned
   WHILE on a dedicated drain stream **after** the V2 monolithic graph
   completes its captured chunk.

   Why it is the safest first target:
     - **Clear predicate.** Drain status is already a single u32 device
       counter (`fused_engine.rs:88` `pub drain_count: usize`); the
       predicate "more deferred records pending" is a one-byte read and
       a 1-thread set-conditional (mirrors
       `prism_gearbox_predicate_bridge_kernel` at
       `gearbox.cu:436-462`).
     - **Hard iteration cap is already implied.** Drain is fed by a
       fixed-size ring buffer (the Tier-8 deferred-error-state ring); the
       maximum number of records is bounded by ring capacity per chunk,
       which is a build-time constant. The WHILE `max_iterations` cap can
       be set to `RING_CAPACITY + epsilon` and never trip in practice.
     - **Watchdog.** The drain body is one captured DMA + one fence
       kernel; no kernels touch integrator state, so a watchdog trip is
       safe to surface as `WhileExitReason::WatchdogTrip` and the chunk
       completes normally with truncated telemetry. No correctness risk.
     - **Today's status quo is host-side.** It happens off the critical
       path and never blocks GPU work, so the WHILE migration does not
       have to defend integrator throughput. If the WHILE breaks, fall
       back to today's host drain — already tested.
     - **Visible in the all-8 freeze evidence.** The drain caveat is the
       only known "deferred" loop pattern in the production graph
       (`docs/TIER8_GRAPH_TOPOLOGY.md:99-107`); migrating it gives a
       direct measurable improvement (eliminating 2× drained checks) and
       a reusable pattern for the next two candidates.

   The two candidates we are NOT recommending first, and why:
     - **Bounded ASC convergence loop.** ASC currently runs as a single
       captured kernel (`asc_steering.cuh:42`,
       `captured_pipeline.rs:3760-3768`); converting it into a
       predicate+body WHILE would require splitting the kernel and
       re-validating the gain coefficient under multiple-launch dynamics.
       Steering loop has tight integrator coupling — too many co-changes
       for the first WHILE.
     - **Bounded coherence settle.** Coherence is advisory-only per
       `CLAUDE.md` "IMMUTABLE RULES #5 — Coherence never blocks alone."
       A WHILE here would re-introduce a blocking coherence path, which
       is precisely what the gating-stack rules forbid.

## INVARIANTS WHILE MUST UPHOLD

(From the directive — committed as test conditions in §"Acceptance tests".)

- **Parent-owned only.** Compile-time enforced: the only public Rust
  entry is `add_while_node_to_parent_graph(parent_graph, ...)`. There is
  no `add_while_node_to_child_template` API. `splice_legality_check`
  already rejects any child graph containing a conditional node
  (`graph_node.rs:96-100`).
- **Bounded `max_iterations`.** Required field on `WhileLoopConfig`;
  panic-on-zero in the constructor. Predicate-bridge kernel takes the
  current iteration counter as a kernel-node arg and forces predicate
  to 0 once `iter_count >= max_iterations`.
- **Every WHILE emits a WhileTrace.** The exit-reason write happens in
  the bridge kernel before it sets the conditional to 0 — this is the
  same pattern the SFA kernel uses to write `cruise->current_gear`
  before the bridge calls `cudaGraphSetConditional`
  (`gearbox.cu:268-280` then `:461`).

## WhileTrace schema (verbatim from directive — clean, no conflicts)

```rust
// crates/prism-nhs/src/graph_node.rs (proposed additions)
pub struct WhileTrace {
    pub while_node_id:           u64,
    pub stream_id:               u32,
    pub chunk_id:                u64,
    pub entry_step:              u64,
    pub exit_step:               u64,
    pub iteration_count:         u32,
    pub max_iterations:          u32,
    pub exit_reason:             WhileExitReason,
    pub predicate_history_hash:  u64,
    pub watchdog_status:         u32,
}

pub enum WhileExitReason {
    PredicateFalse,
    MaxIterations,
    WatchdogTrip,
    Error,
    Disabled,
}
```

Conflict scan: `rg "(WhileTrace|WhileExitReason|while_node_id|predicate_history_hash|watchdog_status)" crates/` → 0 hits (this doc is the first introduction).

---

## Ten-section report (per directive)

### 1. Files inspected

| Path | Purpose | Lines read |
|---|---|---|
| `crates/prism-nhs/src/cuda/graph_node.cuh` | unified-graph-builder FFI header | 1-162 (entire) |
| `crates/prism-nhs/src/cuda/graph_node.cu`  | unified-graph-builder FFI impl | 1-191 (entire) |
| `crates/prism-nhs/src/graph_node.rs`       | Rust-side wrappers for above | 1-289 (entire) |
| `crates/prism-nhs/src/cuda/adjudicator.cu` | F1 SWITCH + G26 SWITCH FFI (sibling pattern) | 780-1200 |
| `crates/prism-nhs/src/cuda/gearbox.cu`     | G26 SFA / predicate-bridge / body populator | 200-781 |
| `crates/prism-nhs/src/cuda/gearbox.cuh`    | G26 surface | 200-300 |
| `crates/prism-nhs/src/captured_pipeline.rs`| consumer of all of the above | 686-1200, 3760-4585 |
| `crates/prism-nhs/build.rs`                | nvcc invocation; -arch=sm_120 + CUDA 13.x conventions | 1-495 (entire) |
| `docs/TIER8_GRAPH_TOPOLOGY.md`             | GRAPH-SPLICE-001 invariant + Option β topology | 1-117 (entire) |
| `/usr/local/cuda/include/driver_types.h`   | authoritative WHILE struct constraint | 3555-3614 |
| `vendor/cudarc/src/runtime/sys/mod.rs`     | cudarc enum WHILE binding | 3720-3760 |

### 2. Functions / structs found (load-bearing for WHILE)

- `prism_graph_add_child_node_v3_ffi`
  (`graph_node.cu:18`) — splice GRAPH; the WHILE helper will sit
  next to it.
- `prism_graph_is_splice_legal_ffi` (`graph_node.cu:70`,
  `graph_node.rs:155`) — already counts conditional nodes, will
  count WHILE as conditional automatically (no change needed).
- `prism_wire_f1_switch_ffi` (`adjudicator.cu:865`) — IF/SWITCH
  reference pattern for handle creation + `cudaGraphAddNode(type=Conditional)`.
- `prism_wire_g26_gearbox_ffi` (`adjudicator.cu:1001`) — SWITCH
  4-way reference; identical structure but `size=4`, `phGraph_out[4]`.
- `prism_gearbox_create_handle_ffi` (`adjudicator.cu:1119`) — exact
  shape we mirror for `prism_graph_create_while_handle_ffi`.
- `prism_gearbox_wire_with_handle_ffi` (`adjudicator.cu:1137`) —
  exact shape we mirror for `prism_graph_add_while_node_ffi`,
  parameterized to `size=1, phGraph_out[1]`.
- `prism_gearbox_predicate_bridge_kernel` (`gearbox.cu:436-462`)
  — reference for the WHILE predicate-bridge kernel: 1-thread,
  reads device state, calls `cudaGraphSetConditional`.
- `cudaConditionalNodeParams` (`driver_types.h:3567-3594`) — confirms
  `size==1` is mandatory for WHILE.
- `PrismGraphTemplate::create_parent_g26_cond_handle` (`captured_pipeline.rs:4470`)
  / `wire_parent_g26_switch` (`:4511`) — the parent-owned wiring
  template the WHILE wrapper will follow.
- `cudaGraphConditionalNodeType::cudaGraphCondTypeWhile = 1`
  (`vendor/cudarc/src/runtime/sys/mod.rs:3735`,`:3747`;
  authoritative `/usr/local/cuda/include/driver_types.h:3560`).

### 3. Reusable code (drop-in for the WHILE lane)

- The CUDA-version gate `#if CUDART_VERSION >= 12040 ... #else
  cudaErrorNotSupported ... #endif` at `adjudicator.cu:803-846`.
- The handle creation block at `adjudicator.cu:879-886`.
- The "populate `cudaGraphNodeParams` + call `cudaGraphAddNode`" block
  at `adjudicator.cu:905-934` (change only `conditional.type` and
  `conditional.size`).
- The `phGraph_out[i]` copy-out loop at
  `adjudicator.cu:1071-1075` (down-sized from 4 to 1).
- Predicate-bridge kernel skeleton at `gearbox.cu:436-462`
  (replace gear logic with iteration-count clamp + drain-status read).
- Splice legality preflight `splice_legality_check` (`graph_node.rs:155`)
  — no change; it already detects WHILE because it counts
  `cudaGraphNodeTypeConditional`.
- Trace emission pattern from `cruise->current_gear` writeback at
  `gearbox.cu:268-280` (in our case: WhileTrace fields written to a
  pinned host slot).

### 4. Unsafe / problematic (must NOT do)

- ❌ DO NOT add a WHILE node inside a captured child template — would
  fail `splice_legality_check` and trip GRAPH-SPLICE-001
  (`docs/TIER8_GRAPH_TOPOLOGY.md:8-11`).
- ❌ DO NOT make the iteration cap a runtime device value with no
  build-time bound — violates the "Bounded — no unbounded dynamic
  loops" invariant.
- ❌ DO NOT call `cuGraphAddNode` directly from Rust (TIER 7
  prohibition; see `CLAUDE.md` Execution Policy "Direct sys::cuGraphAdd*
  calls from Rust are prohibited"). The new helper MUST live in
  `cuda/graph_node.cu`.
- ❌ DO NOT remove or weaken `CaptureGuard`, `splice_legality_check`,
  or the deferred-drain reporting in `persistent_engine.rs:1561-1605`
  and `fused_engine.rs:7155-7295`.
- ❌ DO NOT skip the `WhileTrace` emission. Every WHILE must record one,
  even if exit reason is `Disabled`.
- ❌ DO NOT generalize `prism_wire_g26_gearbox_ffi` into a polymorphic
  helper that switches on `cond_type` — the SWITCH body populator
  contract assumes `size=4`.

### 5. Exact plan

Sequenced as 6 micro-patches; each is independently revertable.

**P1. C++ FFI scaffold (graph_node.cu/.cuh).**
   - Add `prism_graph_create_while_handle_ffi(parent_graph,
     default_value, *out_handle)` next to
     `prism_graph_add_child_node_v3_ffi`.
   - Add `prism_graph_add_while_node_ffi(parent_graph, predicate_node,
     handle_v, *out_conditional_node, *out_body_subgraph)` —
     `nodeParams.conditional.type = cudaGraphCondTypeWhile`, `size = 1u`.
   - Both gated `#if CUDART_VERSION >= 12040` else `cudaErrorNotSupported`.
   - `build.rs` already compiles `graph_node.cu` (line 221-226), so the
     archive picks the new symbols up automatically.

**P2. Rust FFI extern + safe wrappers.**
   - Add to `crates/prism-nhs/src/graph_node.rs`:
     - `extern "C"` blocks for the two new FFI symbols.
     - `pub unsafe fn create_while_handle(parent_graph: CUgraph,
       default_value: u32) -> Result<u64, i32>`.
     - `pub unsafe fn add_while_node(parent_graph: CUgraph,
       predicate_node: CUgraphNode, handle: u64)
       -> Result<(CUgraphNode, CUgraph), SpliceError>`.
   - Reuse `SpliceError` taxonomy at `graph_node.rs:115-148`.

**P3. Trace types + ring.**
   - Add `WhileTrace`, `WhileExitReason`, and a small pinned-host ring
     `WhileTraceRing { capacity, head_idx_dev, slots_pinned }` in
     `crates/prism-nhs/src/graph_node.rs` or new submodule
     `graph_node::while_trace`.
   - Predicate-bridge kernel writes one slot per iteration; final exit
     slot stamped with `exit_reason`.

**P4. Predicate-bridge kernel for the post-T7 drain WHILE.**
   - New kernel in a *new* TU, e.g. `cuda/while_drain_bridge.cu`, that
     reads `drain_count` (or its device address) and the iteration
     counter, calls `cudaGraphSetConditional(handle, predicate_value)`.
   - Predicate logic:
     - If `iter_count >= max_iterations` → set 0 + write
       `WhileExitReason::MaxIterations`.
     - Else if `drain_remaining == 0` → set 0 + write `PredicateFalse`.
     - Else if watchdog tripped → set 0 + write `WatchdogTrip`.
     - Else → set 1 + increment `iter_count`, append history hash.
   - 1 thread, 1 block — same shape as
     `prism_gearbox_predicate_bridge_kernel`.

**P5. Parent-graph wiring helpers in `captured_pipeline.rs`.**
   - Add `PrismGraphTemplate::create_parent_while_handle` (mirror
     `:4470-4497`).
   - Add `PrismGraphTemplate::wire_parent_while_node(parent_graph,
     dependency_node, predicate_kernel_args, max_iterations)` (mirror
     `:4511-4585`); inserts the WHILE node after the V2 child has been
     spliced, so parent topology is `ChildAdj → G26_SWITCH →
     PostT7Drain_WHILE → multi_lif`.

**P6. Body-subgraph populator (drain only).**
   - Use the single `phGraph_out[0]` body to capture: drain DMA kernel
     → fence kernel. No SWITCH-style multi-body populator needed.
   - On any internal drain-kernel failure, the WHILE body returns and
     the next iteration's predicate read trips watchdog.

### 6. Minimal touch list (FFI / Rust only — no integrator code)

| File | Action | Bounds |
|---|---|---|
| `crates/prism-nhs/src/cuda/graph_node.cuh`    | EDIT — add 2 prototypes + doc | +~80 lines after line 156 |
| `crates/prism-nhs/src/cuda/graph_node.cu`     | EDIT — add 2 functions + WHILE gate | +~100 lines after line 190 |
| `crates/prism-nhs/src/cuda/while_drain_bridge.cu`  | NEW — predicate-bridge kernel | ~120 lines |
| `crates/prism-nhs/src/cuda/while_drain_bridge.cuh` | NEW — kernel prototype | ~30 lines |
| `crates/prism-nhs/src/graph_node.rs`          | EDIT — extern blocks + 2 wrappers + WhileTrace types | +~200 lines after line 288 |
| `crates/prism-nhs/build.rs`                   | EDIT — register new TU `while_drain_bridge.cu` | +5 lines |
| `crates/prism-nhs/src/captured_pipeline.rs`   | EDIT — `create_parent_while_handle` + `wire_parent_while_node` next to `:4470` and `:4511` | +~150 lines |
| `crates/prism-nhs/tests/while_node_smoke.rs`  | NEW — acceptance tests (see §8) | ~200 lines |
| `docs/TIER8_GRAPH_TOPOLOGY.md`                | EDIT — append "Parent-owned WHILE" section after line 56 | +~30 lines |

No edit to `lib.rs`, `fused_engine.rs`, or `nhs_rt_full.rs` in this lane.
No edit to the V2 child template's capture logic.

### 7. New structs / functions

C++ FFI (in `graph_node.cuh` / `.cu`):

```c
int prism_graph_create_while_handle_ffi(
    cudaGraph_t parent_graph,
    uint32_t    default_value,    // 1 ⇒ enter the body once by default
    uint64_t*   out_handle);

int prism_graph_add_while_node_ffi(
    cudaGraph_t      parent_graph,
    cudaGraphNode_t  predicate_node,    // dependency: predicate-bridge kernel
    uint64_t         handle_v,
    cudaGraphNode_t* out_conditional_node,
    cudaGraph_t*     out_body_subgraph);  // single-element output
```

Rust (in `graph_node.rs`):

```rust
pub struct WhileTrace { /* see §"WhileTrace schema" */ }
pub enum WhileExitReason { PredicateFalse, MaxIterations, WatchdogTrip, Error, Disabled }

pub unsafe fn create_while_handle(parent_graph: CUgraph, default_value: u32) -> Result<u64, i32>;

pub unsafe fn add_while_node(
    parent_graph: CUgraph,
    predicate_node: CUgraphNode,
    handle: u64,
) -> Result<(CUgraphNode, CUgraph), SpliceError>;

pub struct WhileLoopConfig {
    pub max_iterations: u32,           // hard cap; non-zero
    pub watchdog_step_budget: u32,     // host-monotonic upper bound
    pub trace_ring_capacity: u32,
}
```

Captured pipeline (in `captured_pipeline.rs`):

```rust
impl PrismGraphTemplate {
    pub unsafe fn create_parent_while_handle(parent_graph: CUgraph)
        -> Result<u64, BuildError>;

    pub unsafe fn wire_parent_while_node(
        &self,
        parent_graph: CUgraph,
        dependency_node: CUgraphNode,
        cfg: WhileLoopConfig,
        d_predicate_state: *const u32,    // e.g. drain_count addr
    ) -> Result<(CUgraphNode, CUgraph), BuildError>;  // (cond_node, body)
}
```

### 8. Acceptance tests

(In `crates/prism-nhs/tests/while_node_smoke.rs`, all behind
`#[cfg(feature = "gpu")]`.)

1. **`while_handle_create_succeeds_on_parent_graph`** — calls
   `create_parent_while_handle` on a freshly-created `CUgraph`, asserts
   `handle != 0`.
2. **`while_node_added_with_size_one`** — wires WHILE on a parent graph;
   walks parent nodes via `cudaGraphGetNodes` + `cudaGraphNodeGetType`,
   confirms exactly one `cudaGraphNodeTypeConditional`, and uses
   `cudaGraphConditionalHandleGetParams` (or inspects the body via
   `phGraph_out[0]` non-null) to verify `size == 1`.
3. **`splice_legality_rejects_while_in_child_template`** — builds a
   throwaway child graph, adds a WHILE node, calls
   `splice_legality_check`, asserts
   `report.conditional_count == 1` and `report.is_legal() == false`.
4. **`bounded_max_iterations_panics_on_zero`** — constructs
   `WhileLoopConfig { max_iterations: 0, .. }` and asserts a panic
   (Rust-side; never reaches CUDA).
5. **`predicate_bridge_writes_max_iterations_exit`** — runs the WHILE
   with predicate-bridge always returning "more work"; instantiate +
   launch the parent graph; asserts WhileTrace ring has exactly
   `max_iterations` entries with the final entry's
   `exit_reason == MaxIterations`.
6. **`predicate_bridge_writes_predicate_false_exit`** — drain counter
   pre-set to 0; asserts trace has `iteration_count == 0` (or 1, per
   `defaultLaunchValue`) and `exit_reason == PredicateFalse`.
7. **`watchdog_trip_emits_watchdog_status`** — host-side budget exceeded
   (`watchdog_step_budget = 1`, predicate keeps yielding "more work");
   asserts `exit_reason == WatchdogTrip`.
8. **`every_while_emits_a_trace`** — exhaustive over all four exit
   reasons + `Disabled` (compile-time `--no-while` path); asserts
   trace ring head moved by exactly 1 per launch.
9. **`tier8_freeze_evidence_unchanged`** — re-run the all-8 instantiate
   gate (`docs/TIER8_GRAPH_TOPOLOGY.md:68-97`) and confirm:
     - No new `rc=801`, `rc=900`, `rc=901`.
     - `child_conditional_nodes=0` still holds.
     - `parent_conditional_nodes` increases from 1 (G26) to 2
       (G26 + WHILE).
     - No new `STREAM_CAPTURE_INVALIDATED`.

### 9. Failure modes

| Mode | Signal | Mitigation |
|---|---|---|
| `cudaErrorNotSupported` from `cudaGraphAddNode(WHILE)` on toolkit < 12.4 | FFI returns the cudaError | `#if CUDART_VERSION >= 12040` gate (mirrors `adjudicator.cu:803-846`); WHILE flagged `Disabled` and a single trace emitted with `exit_reason::Disabled`. |
| Predicate-bridge fires before drain counter is initialized | First-replay UB | Build-time invariant: bridge kernel checks `iter_count == 0` against the `defaultLaunchValue=1` and returns "false" once if predicate-state buffer is null (mirror of `prism_gearbox_predicate_bridge_kernel`'s null-guard, `gearbox.cu:443`). |
| `phGraph_out[0]` returned null on Blackwell driver (the same "Smoking Gun" the SWITCH populator catches at `gearbox.cu:670-678`) | Body-subgraph null | Defensive: WHILE wrapper checks `body == nullptr` and returns `SpliceError::NullNode`. |
| iteration cap reached without making progress (cycle) | All bodies executed but predicate never falsified | `WhileExitReason::MaxIterations`; trace ring records last 16 predicate-history-hashes for offline diagnosis. |
| Watchdog fires due to host pause/scheduling jitter | Body did not actually mis-behave | `WatchdogTrip` is non-fatal; chunk completes; orchestrator continues. |
| WHILE body deferred-drain DMA itself errors | Drain failure | Body returns; next iteration's predicate read trips a sentinel; bridge sets predicate to 0 with `Error`. |
| Splice-legality regression introduces WHILE into child template | TIER 8 preflight fail | Existing `splice_legality_check` already counts conditional nodes; no change needed. The error message ("splice ILLEGAL — child graph contains conditional={N}…", `graph_node.rs:133`) covers WHILE without code change. |
| Parent graph instantiation rejects WHILE on a CUDA driver where SWITCH was accepted | rc=801 | Driver-version assertion at `create_while_handle` entry; bail with typed error. |

### 10. Rollback path

P-numbered patches are independently revertable; rollback in reverse
order. Specifically:

1. `git revert` P6 (drain populator) — WHILE remains scaffolded but
   has no body; predicate-bridge fires once with `Disabled`.
2. `git revert` P5 (parent wiring helpers) — WHILE never installed
   on parent graph; all-8 freeze gate matches pre-WHILE baseline.
3. `git revert` P4 (predicate bridge) — `while_drain_bridge.cu`
   removed; FFI and Rust types remain compilable.
4. `git revert` P3 (trace types) — WhileTrace removed; no consumers.
5. `git revert` P2 (Rust safe wrappers).
6. `git revert` P1 (C++ FFI). At this point repo is byte-identical to
   `8ca26189` modulo this doc.

Smoke test for each rollback step: `cargo check -p prism-nhs --features gpu`
(must remain green after every revert).

---

## Concise final summary

- **WHILE FFI exposure status:** NO — `crates/prism-nhs/src/cuda/graph_node.cu`
  exposes only GRAPH (`:43`) and Memset (`:174`) node types. The CUDA
  enum value `cudaGraphCondTypeWhile = 1` is defined in vendored cudarc
  (`vendor/cudarc/src/runtime/sys/mod.rs:3735`,`:3747`) and the toolkit
  header (`/usr/local/cuda/include/driver_types.h:3560`) but no PRISM TU
  references it.
- **CUDA version requirement:** WHILE conditional nodes need CUDA toolkit
  ≥ 12.4 (Programming Guide §3.2.8.7.4). System toolkit is 13.2 (driver
  595.45.04, sm_120 RTX 5080) — well above the floor. Gate with
  `#if CUDART_VERSION >= 12040` mirroring `adjudicator.cu:803`.
- **Recommended first use case:** **bounded post-T7 telemetry drain**
  — already a host-side bounded loop today (drain ring capacity
  bounds `iter_count`; see `docs/TIER8_GRAPH_TOPOLOGY.md:99-107`,
  `fused_engine.rs:88` `drain_count`), no integrator coupling, fall-back
  is the existing host loop. Safest GPU-native first migration.
- **Minimal patch sequence:** P1 add 2 FFI prototypes/impls in
  `cuda/graph_node.{cuh,cu}` → P2 Rust extern + safe wrappers in
  `graph_node.rs` → P3 `WhileTrace`/`WhileExitReason` types →
  P4 `cuda/while_drain_bridge.{cu,cuh}` predicate-bridge kernel →
  P5 `PrismGraphTemplate::create_parent_while_handle` +
  `wire_parent_while_node` in `captured_pipeline.rs` (next to
  `:4470` / `:4511`) → P6 single-body drain populator. Each independently
  revertable.
- **Blockers:** None on toolkit / driver / hardware / topology.
  Operational blocker: this lane writes new code and adds a
  `cudaGraphNodeTypeConditional` to the parent graph — the all-8 freeze
  gate (`docs/TIER8_GRAPH_TOPOLOGY.md:68-97`) must be re-run and is
  expected to show `parent_conditional_nodes` increase from 1 to 2.
  Splice legality contract (`child_conditional_nodes=0`) is preserved
  by the parent-only architecture (Option β).
