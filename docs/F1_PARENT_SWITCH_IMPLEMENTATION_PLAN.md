# F1 Parent-Owned SWITCH — Implementation Plan

Status: Scout deliverable, READ-ONLY. No code edits.
Date: 2026-05-04
Baseline commit: 8ca26189
Branch: producer-repair-causal-truthing-20260426
Author: AGENT 2 — F1 Parent-Owned SWITCH Implementation Scout
Companion docs: [docs/TIER8_GRAPH_TOPOLOGY.md](TIER8_GRAPH_TOPOLOGY.md), [docs/EXECUTION_POLICY.md](EXECUTION_POLICY.md)

---

## 0. Bottom line

F1 is currently a **production no-op** at the CUDA-graph layer. The only F1
artefacts in the production path are:

- a host-side log scraper (`ghost_telemetry::log_f1_switch_events`,
  [crates/prism-nhs/src/ghost_telemetry.rs:536](../crates/prism-nhs/src/ghost_telemetry.rs))
  that reads `adjudication_code == 1` from the pinned ring after launch, and
- the `PipelineConfig.adj_dev_ptr` accessor + a documented hook example showing how
  `prism_wire_f1_switch_ffi` could inject a child-level F1 SWITCH
  (test-only path:
  [captured_pipeline.rs:5491](../crates/prism-nhs/src/captured_pipeline.rs)).

**No production caller invokes `prism_wire_f1_switch_ffi`.** The only test
that does
([captured_pipeline.rs:5515](../crates/prism-nhs/src/captured_pipeline.rs))
inserts the SWITCH **inside the captured child template**, which is now
exactly the GRAPH-SPLICE-001 violation TIER 8 forbids — a SWITCH inside a
spliced child is a NO-GO.

The G26 SWITCH was promoted from child-installed (overlay) to parent-installed
(monolithic-discovery) in TIER 8 Option β. The same architectural move now
applies to F1. The G26 helpers
(`create_parent_g26_cond_handle`, `wire_parent_g26_switch`,
`prism_gearbox_create_handle_ffi`, `prism_gearbox_wire_with_handle_ffi`,
`prism_gearbox_predicate_bridge_kernel`) are the templates to generalize.

Recommended generalization strategy: **duplicate as `f1_*` siblings** rather
than parameterize G26's helpers with a `branch_id`. Justification in §3.

---

## 1. Files inspected

Cited file:line for every load-bearing claim.

### Rust orchestration

- [crates/prism-nhs/src/captured_pipeline.rs:686-710](../crates/prism-nhs/src/captured_pipeline.rs)
  — `extern "C"` block declaring `prism_wire_f1_switch_ffi` (3-way SWITCH,
  in-graph; no parent-owned variant exists).
- [crates/prism-nhs/src/captured_pipeline.rs:725-761](../crates/prism-nhs/src/captured_pipeline.rs)
  — G26 FFI block: `prism_wire_g26_gearbox_ffi`,
  `prism_gearbox_create_handle_ffi`, `prism_gearbox_wire_with_handle_ffi`,
  `prism_gearbox_populate_switch_bodies_ffi`. The latter two are the
  parent-owned siblings F1 must mirror.
- [crates/prism-nhs/src/captured_pipeline.rs:965-973](../crates/prism-nhs/src/captured_pipeline.rs)
  — `PipelineConfig.g26_parent_cond_handle: u64` field (Option β switch:
  zero ⇒ child wires its own SWITCH; nonzero ⇒ parent owns it).
- [crates/prism-nhs/src/captured_pipeline.rs:3803-3846](../crates/prism-nhs/src/captured_pipeline.rs)
  — handle-import branch in `build_with_v2_hook`; `g26_switch_owned_by_parent`
  flag drives capture-time SFA + bridge launches.
- [crates/prism-nhs/src/captured_pipeline.rs:3883-3930](../crates/prism-nhs/src/captured_pipeline.rs)
  — capture-time launches: SFA kernel
  (`prism_gearbox_launch_sfa`), predicate bridge
  (`prism_gearbox_launch_predicate_bridge`), and the post-bridge
  frontier snapshot (`g26_bridge_node`).
- [crates/prism-nhs/src/captured_pipeline.rs:4218-4310](../crates/prism-nhs/src/captured_pipeline.rs)
  — child-side post-capture branch: skipped when parent-owned, otherwise
  installs child SWITCH (legacy overlay path).
- [crates/prism-nhs/src/captured_pipeline.rs:4461-4497](../crates/prism-nhs/src/captured_pipeline.rs)
  — `CapturedAdjudicationPipeline::create_parent_g26_cond_handle` (parent-graph
  handle creator). **This is the function F1 must mirror.**
- [crates/prism-nhs/src/captured_pipeline.rs:4499-4585](../crates/prism-nhs/src/captured_pipeline.rs)
  — `CapturedAdjudicationPipeline::wire_parent_g26_switch` (parent-graph
  SWITCH wirer + body populator). **This is the second function F1 must mirror.**
- [crates/prism-nhs/src/captured_pipeline.rs:4754-4768](../crates/prism-nhs/src/captured_pipeline.rs)
  — `cu_graph_raw` and `adj_dev_ptr` accessors used by V2 hook callers.

### CUDA / C++ FFI

- [crates/prism-nhs/src/cuda/adjudicator.cu:865-963](../crates/prism-nhs/src/cuda/adjudicator.cu)
  — `prism_wire_f1_switch_ffi`. **In-graph variant only.** It calls
  `cudaGraphConditionalHandleCreate` on the same graph that owns
  `adjudicator_node` and immediately adds the SWITCH node depending on
  it. No "wire-with-existing-handle" path. No predicate bridge — it relies
  on a separate `prism_adj_set_conditional_kernel` (see
  [adjudicator.cuh:407-408](../crates/prism-nhs/src/cuda/adjudicator.cuh)),
  but **that kernel-launching FFI is never wired into the captured pipeline**.
- [crates/prism-nhs/src/cuda/adjudicator.cu:1117-1182](../crates/prism-nhs/src/cuda/adjudicator.cu)
  — `prism_gearbox_create_handle_ffi` + `prism_gearbox_wire_with_handle_ffi`
  (parent-owned G26 helpers; thin wrappers around `cudaGraphConditionalHandleCreate`
  and `cudaGraphAddNode(CONDITIONAL, type=Switch)` with caller-supplied handle).
- [crates/prism-nhs/src/cuda/adjudicator.cuh:414-446, 448-469](../crates/prism-nhs/src/cuda/adjudicator.cuh)
  — public declarations.
- [crates/prism-nhs/src/cuda/gearbox.cu:437-477](../crates/prism-nhs/src/cuda/gearbox.cu)
  — `prism_gearbox_predicate_bridge_kernel` + `prism_gearbox_launch_predicate_bridge`.
  Single-thread kernel that reads cruise/adj state and calls
  `cudaGraphSetConditional(handle, gear)`. **F1 needs an analogous bridge
  kernel that calls `cudaGraphSetConditional(handle, adj->adjudication_code & 0x3)`
  — the predicate is `adj->adjudication_code` at offset 52, not the cruise gear.**
- [crates/prism-nhs/src/cuda/gearbox.cuh:217](../crates/prism-nhs/src/cuda/gearbox.cuh)
  — public bridge declaration.
- [crates/prism-nhs/src/cuda/graph_node.cu](../crates/prism-nhs/src/cuda/graph_node.cu),
  [crates/prism-nhs/src/cuda/graph_node.cuh](../crates/prism-nhs/src/cuda/graph_node.cuh)
  — TIER 7/8 unified graph-node helpers + splice-legality preflight.
  F1 helpers must live alongside (`adjudicator.cu` is acceptable since
  G26 already does — but the **graph-node FFI ownership rule** says any
  conditional-node ADD path must be in the C++ FFI surface, not in
  Rust). **The new F1 helpers may live in `adjudicator.cu` to stay
  cohesive with `prism_wire_f1_switch_ffi`, OR move to `graph_node.cu`
  for a cleaner registry.** §3 picks `adjudicator.cu` for minimal blast
  radius.

### Rust binary

- [crates/prism-nhs/src/bin/nhs_rt_full.rs:5870-6075](../crates/prism-nhs/src/bin/nhs_rt_full.rs)
  — `if args.m1_monolithic_discovery` block. The G26 parent-owned topology is
  built here:
  - 5900: `engine.capture_autonomous_template()` produces parent template.
  - 5904: `splice_legality_check(parent_graph)` — pre-G26 inspection.
  - 5927: `create_parent_g26_cond_handle(parent_graph)` — parent handle.
  - 5944: `mono_cfg.g26_parent_cond_handle = parent_g26_handle` — child sees
    parent handle.
  - 5945: `CapturedAdjudicationPipeline::build(...)` — child template captured.
  - 5961: `splice_legality_check(adj_tmpl)` — post-build child legality
    audit (must report `child_conditional_nodes=0`).
  - 5989: `auto_tmpl.add_child_graph_node(&[fused_node], adj_tmpl)` — splice.
  - 5999: `mono_pipeline.wire_parent_g26_switch(parent_graph, child_node, ...)`
    — install parent SWITCH downstream of child splice node.
  - 6017: `splice_legality_check(parent_graph)` — post-G26 parent audit
    (must report `parent_conditional_nodes >= 1`).
  - 6038: `auto_tmpl.add_dependency(parent_g26_node, multi_lif_node)` —
    wire G26 SWITCH → next parent stage.
  - 6049: `auto_tmpl.instantiate()` — instantiate the whole monolithic graph.
- [crates/prism-nhs/src/bin/nhs_rt_full.rs:5818](../crates/prism-nhs/src/bin/nhs_rt_full.rs)
  — overlay (non-monolithic) path: `g26_parent_cond_handle: 0` (child wires
  child SWITCH; legacy path; mutually exclusive with monolithic).
- [crates/prism-nhs/src/bin/nhs_rt_full.rs:6837-6863](../crates/prism-nhs/src/bin/nhs_rt_full.rs)
  — current production "F1 layer": post-launch ring scrape via
  `log_f1_switch_events`. This is host-side detection, NOT a CUDA-graph
  conditional. F1's actual SWITCH never executes on device today.
- [crates/prism-nhs/src/graph_node.rs](../crates/prism-nhs/src/graph_node.rs)
  — TIER 7/8 unified Rust shim. **F1's Rust-side wrappers should live in
  `captured_pipeline.rs` next to `wire_parent_g26_switch` for consistency,
  not in `graph_node.rs` (which is reserved for the unified
  `cudaGraphAddNode` taxonomy).**

### Build/policy

- [crates/prism-nhs/build.rs:29-48, 206-224, 287-288](../crates/prism-nhs/build.rs)
  — `adjudicator.cu`, `gearbox.cu`, `graph_node.cu` are all already compiled
  as static archives and re-run-on-change tracked. **No build.rs edit needed.**
- [docs/EXECUTION_POLICY.md:39](../docs/EXECUTION_POLICY.md)
  — Direct `sys::cuGraphAdd*` from Rust is a violation. F1 helpers MUST live
  in C++ FFI (`adjudicator.cu` or `graph_node.cu`).
- [docs/TIER8_GRAPH_TOPOLOGY.md](../docs/TIER8_GRAPH_TOPOLOGY.md)
  — defines `parent → ChildAdj → G26_SWITCH(parent) → multi_lif`. F1 inserts
  in the same parent layer.

---

## 2. Functions / structs found (G26 parent-owned reference)

| Layer | Symbol | File:Line | Role |
|-------|--------|-----------|------|
| Rust | `PipelineConfig.g26_parent_cond_handle: u64` | captured_pipeline.rs:973 | Zero ⇒ child-self-wire; nonzero ⇒ parent-owned |
| Rust | `CapturedAdjudicationPipeline::create_parent_g26_cond_handle` | captured_pipeline.rs:4470 | Calls `prism_gearbox_create_handle_ffi(parent_graph, default=1, &handle)` |
| Rust | `CapturedAdjudicationPipeline::wire_parent_g26_switch` | captured_pipeline.rs:4511 | Calls `prism_gearbox_wire_with_handle_ffi` + `prism_gearbox_populate_switch_bodies_ffi`. Returns parent SWITCH node. |
| Rust | `g26_switch_owned_by_parent` capture-time branch | captured_pipeline.rs:3803 | Skips child handle creation when parent owns the handle |
| Rust | `g26_bridge_node` snapshot | captured_pipeline.rs:3916 | Captured kernel-node frontier handle for the predicate bridge |
| Rust | post-capture child-self-wire | captured_pipeline.rs:4226 | Skipped on `g26_switch_owned_by_parent` |
| C++ | `prism_gearbox_create_handle_ffi` | adjudicator.cu:1119 | `cudaGraphConditionalHandleCreate(graph, default_value, cudaGraphCondAssignDefault)` |
| C++ | `prism_gearbox_wire_with_handle_ffi` | adjudicator.cu:1137 | `cudaGraphAddNode(CONDITIONAL, size=4, type=Switch, handle=existing)`, returns `phGraph_out[4]` |
| C++ | `prism_gearbox_predicate_bridge_kernel` | gearbox.cu:437 | `cudaGraphSetConditional(handle, gear & 0x3)` (single-thread launch-into-capture kernel) |
| C++ | `prism_gearbox_launch_predicate_bridge` (host shim) | gearbox.cu:465 | Launches the bridge kernel on a stream during capture |
| C++ | preflight | graph_node.cu:70 (`prism_graph_is_splice_legal_ffi`) | Counts conditional / alloc / free nodes; used both by Rust and `splice_legality_check` |

The same GRAPH-SPLICE-001 contract applies: conditional, alloc, free
nodes inside a child template are illegal.

---

## 3. Reusable code & generalization decision

### Choice — duplicate as `f1_*` helpers

The brief asks me to choose between (a) renaming G26 helpers to take a
`branch_id` parameter, or (b) duplicating as `f1_*` helpers. **I pick
duplicate.**

Justification:

1. **Body subgraph count differs.** G26 SWITCH has `size=4` (3 gears + trap).
   F1 SWITCH has `size=3` (Prune / Construct / Violation), per
   `prism_wire_f1_switch_ffi`'s in-graph variant
   ([adjudicator.cu:909](../crates/prism-nhs/src/cuda/adjudicator.cu)).
   Parameterising on `size` is doable but the body populator
   (`prism_gearbox_populate_switch_bodies_ffi`) hard-codes 4 bodies and
   gear-specific kernel logic; F1's bodies (Prune-no-op, Construct-no-op,
   Violation-trap) are different kernels entirely, so the populator cannot
   be parameterized cleanly.
2. **Predicate source differs.** G26 reads `cruise->current_gear` (with
   `adj->gear_override` shadow). F1 reads `adj->adjudication_code`
   (offset 52). The bridge kernel logic is structurally similar but the
   I/O contract is different — cleaner as a sibling than as a
   `predicate_source` enum.
3. **Default value differs.** G26 default = 1 (Gear 1 / 2.0fs cruise).
   F1 default = 0 (Prune; pre-burn fallback per
   [adjudicator.cu:881](../crates/prism-nhs/src/cuda/adjudicator.cu)).
4. **Future-proofing.** F1 may eventually need device-pointer-based
   conditionals (driver evolution); G26 will not. Coupling them through
   a shared parameter list locks both sides into the lower-fidelity surface.
5. **Diff size.** Duplication is ~150 LOC across 3 files (Rust + C++ +
   header). Generalisation would touch 5+ files and rewrite at least
   two existing helpers, breaking the all-8 instantiate gate's commit
   bisect locality.

The duplication preserves G26 byte-for-byte, which means the all-8 gate's
freeze evidence
([docs/TIER8_GRAPH_TOPOLOGY.md:68](../docs/TIER8_GRAPH_TOPOLOGY.md))
remains unaffected by F1 work.

---

## 4. Unsafe / problematic existing code

| Item | Where | Why it matters |
|------|-------|----------------|
| `prism_wire_f1_switch_ffi` (existing 3-way SWITCH FFI) creates handle inside the SAME graph it adds the conditional to | adjudicator.cu:865 | This is an **in-graph** wirer (legal in standalone overlay; ILLEGAL inside a spliceable child). It is unsafe to call from inside the captured-child build path because it would emit a CONDITIONAL node into the child template — exactly the GRAPH-SPLICE-001 violation. **F1 must NOT reuse this function** for parent-owned wiring. |
| The single test that calls it (captured_pipeline.rs:5515) places the SWITCH inside the child template | captured_pipeline.rs:5491 | The test predates Option β. It exercises the legacy "monolithic by hook" path. It still passes today because the test's "parent" is the captured child itself (no further splice). For production parent-owned, this test is **misleading reference code** — DO NOT copy its pattern. |
| `prism_get_adjudication_code_devptr` (helper that yields `&adj->adjudication_code`) | adjudicator.cu:775 | Stable accessor; safe to reuse for the F1 bridge kernel argument. |
| `g26_bridge_node` is captured by snapshotting `cuStreamGetCaptureInfo_v2` immediately after the bridge kernel launch | captured_pipeline.rs:3916 | The same pattern is required for the F1 bridge node snapshot. The frontier is exactly `[bridge]` because the bridge kernel is the only single-thread launch on `md_stream` at that point. **DO NOT introduce branching in the capture window before the F1 bridge snapshot.** |
| Both G26 and F1 child bridge kernels run during capture with handle as argument | gearbox.cu:475 | Captured kernel-node arguments are baked into the graph. Re-instantiation is not needed at replay; the kernel calls `cudaGraphSetConditional(handle, value)` every replay. The same captured-kernel pattern works for F1. |
| `auto_tmpl.add_dependency(...)` uses raw `sys::cuGraphAddDependencies` directly from Rust | graph_capture.rs:206 | This predates the TIER 7 policy (route through C++ FFI). Wider question — out of scope for F1, but worth noting. F1 does NOT need to introduce a new direct FFI call from Rust; reusing `auto_tmpl.add_dependency` is acceptable. |

---

## 5. Exact plan

### Topology after F1 lands

```
parent graph:
  fused_step
    └── ChildAdj (GRAPH; F2-pool body, splice-legal)
          └── F1_SWITCH(parent, size=3, handle=parent-owned)
                ├── case 0: Prune body (empty subgraph, no-op)
                ├── case 1: Construct body (empty for v1; reserved for ASC parent-side)
                └── case 2: Violation body (PTX trap; identical kernel to G26 case 3)
          └── G26_SWITCH(parent)  [as today]
                └── multi_lif
```

Order is important: F1 SWITCH lives **upstream of G26 SWITCH** because F1's
Violation case must short-circuit the integrator before G26 selects a
gear. ChildAdj feeds F1 directly because the predicate is the
adjudicator's `adjudication_code` written by Node C inside the child.

Wait — `adjudication_code` is written by a kernel inside the child template.
The child copies-by-value, so the parent-spliced child instance writes to
`adj_dev` (which is pointer-stable across child copies because `adj_dev`
is held by `mono_pipeline`, not embedded in the kernel-node params).
Therefore, after the child node finishes on the parent graph, `adj_dev`
contains the new `adjudication_code`, and a parent-side bridge kernel can
read it and forward via `cudaGraphSetConditional`.

**This means F1 needs a parent-side predicate bridge kernel node**, not a
child-captured one. Unlike G26 (which captures its bridge inside the child
because the gear logic touches `cruise_state_dev` mutated inside the
child), F1's predicate is `adj->adjudication_code` which is fully
visible after the child node completes. The cleanest topology:

```
parent: ... ChildAdj → F1_predicate_bridge(parent kernel node)
                     → F1_SWITCH(parent, deps=[bridge])
                     → G26_SWITCH(parent) → multi_lif
```

The parent-side bridge is added via a captured-stream sequence on the
**parent**: begin a small capture window on the parent_graph's stream,
launch the bridge kernel, end capture, splice the resulting single-node
graph into the parent. Or — simpler — add the bridge as a kernel node
directly via `cudaGraphAddKernelNode` family (which we don't have a
helper for yet) routed through a new C++ FFI helper.

**Recommended approach:** add a new C++ FFI helper
`prism_f1_add_parent_bridge_ffi` that takes the parent graph, the
dependency node (child splice node), the device adjudicator pointer,
and the F1 conditional handle, and adds a kernel node that runs
`prism_f1_predicate_bridge_kernel<<<1,1>>>(handle, adj)`.
This keeps Rust out of `cuGraphAdd*` directly.

### File touch list

**New code:**

1. `crates/prism-nhs/src/cuda/adjudicator.cu` — three new functions:
   - `prism_f1_create_handle_ffi(graph, default_value=0, *out_handle)`
     (mirrors `prism_gearbox_create_handle_ffi`)
   - `prism_f1_wire_with_handle_ffi(graph, predicate_node, handle_v,
     *out_conditional_node, *out_body_subgraphs[3])`
     (mirrors `prism_gearbox_wire_with_handle_ffi` but with `size=3`)
   - `prism_f1_populate_switch_bodies_ffi(body_subgraphs[3])`
     (populates: case 0 empty, case 1 empty, case 2 PTX trap kernel)
   - `prism_f1_add_parent_bridge_ffi(parent_graph, dep_node,
     handle_v, adj_dev, *out_bridge_node)`
     (adds a kernel-node to the parent graph that, on launch, runs the
     F1 predicate-bridge kernel)
   - `__global__ prism_f1_predicate_bridge_kernel(handle, adj)`
     (single-thread; reads `adj->adjudication_code`, calls
     `cudaGraphSetConditional(handle, code & 0x3)`)
2. `crates/prism-nhs/src/cuda/adjudicator.cuh` — declare the four
   `prism_f1_*_ffi` symbols.
3. `crates/prism-nhs/src/captured_pipeline.rs` — three new symbols:
   - `extern "C"` declarations for the four new FFI symbols (alongside
     the existing G26 block at line 725).
   - `PipelineConfig.f1_parent_cond_handle: u64` field (mirrors
     `g26_parent_cond_handle`; defaulted to 0 in all builders).
   - Two new `CapturedAdjudicationPipeline` methods:
     - `create_parent_f1_cond_handle(parent_graph: CUgraph) -> Result<u64, BuildError>`
       (mirrors `create_parent_g26_cond_handle`)
     - `wire_parent_f1_switch(&self, parent_graph, dependency_node) -> Result<CUgraphNode, BuildError>`
       (mirrors `wire_parent_g26_switch`; returns parent F1 SWITCH node;
       internally calls `prism_f1_add_parent_bridge_ffi` THEN
       `prism_f1_wire_with_handle_ffi` THEN
       `prism_f1_populate_switch_bodies_ffi`)
4. `crates/prism-nhs/src/bin/nhs_rt_full.rs` — wire-up at the existing
   monolithic-discovery site:
   - After line 5944 (`mono_cfg.g26_parent_cond_handle = ...`):
     create F1 handle → set `mono_cfg.f1_parent_cond_handle`.
   - After line 5993 (`add_child_graph_node` returns `child_node`):
     call `wire_parent_f1_switch(parent_graph, child_node)` to get
     `parent_f1_node`. Audit: parent_conditional_nodes should be `1` here.
   - After line 5999 (current G26 wire): change G26's dependency from
     `child_node` to `parent_f1_node` so F1 SWITCH happens-before G26.
   - After line 6017 (parent post-G26 audit): expect
     `parent_conditional_nodes == 2` (F1 + G26).
   - After line 6049 instantiate: emit one ENGINE log line summarising
     F1+G26 wiring success.
   - Add F1 trace sidecar emit (see §6).

**No edits to:**

- `build.rs` (already compiles `adjudicator.cu` + `graph_node.cu`).
- `graph_capture.rs` (`add_dependency` + `add_child_graph_node` reused as-is).
- `graph_node.rs` (Rust shim layer; F1 helpers live in
  `captured_pipeline.rs` per existing G26 convention).
- `gearbox.cu/.cuh/.rs` (G26 untouched).
- The existing `prism_wire_f1_switch_ffi` (test-only legacy stays).
- `ghost_telemetry::log_f1_switch_events` (host-side scraper stays as
  redundant cross-check).

### Patch sequence (in order)

| # | File | Lines (approx) | Change |
|---|------|----------------|--------|
| P1 | `crates/prism-nhs/src/cuda/adjudicator.cu` | +120 LOC append after line 1182 | Add `prism_f1_predicate_bridge_kernel`, `prism_f1_create_handle_ffi`, `prism_f1_add_parent_bridge_ffi`, `prism_f1_wire_with_handle_ffi`, `prism_f1_populate_switch_bodies_ffi`. Body case 2 reuses the existing `prism_gearbox_trap_kernel` if exposed; otherwise inline `asm volatile("trap;");`. |
| P2 | `crates/prism-nhs/src/cuda/adjudicator.cuh` | +40 LOC append after line 469 | Forward-declare the five new symbols. |
| P3 | `crates/prism-nhs/src/captured_pipeline.rs` | +5 LOC at line 761 | Add `extern "C"` declarations for the new F1 FFI symbols inside the existing G26 block. |
| P4 | `crates/prism-nhs/src/captured_pipeline.rs` | +12 LOC after line 973 | Add `pub f1_parent_cond_handle: u64` to `PipelineConfig` with full doc-comment mirroring `g26_parent_cond_handle`. |
| P5 | `crates/prism-nhs/src/captured_pipeline.rs` | +6 LOC each at every `PipelineConfig { ... }` literal (8 sites; grep `g26_parent_cond_handle:`) | Set `f1_parent_cond_handle: 0` in every builder. (Production overlay path stays at 0; test fixtures stay at 0.) |
| P6 | `crates/prism-nhs/src/captured_pipeline.rs` | +90 LOC after line 4585 | Add `create_parent_f1_cond_handle` + `wire_parent_f1_switch` (mirror of G26 helpers; PRISM size=3 SWITCH; PTX-trap body case 2; bridge node added as parent-side kernel node). |
| P7 | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | +35 LOC at line 5945 | Insert `parent_f1_handle = create_parent_f1_cond_handle(...)?` and `mono_cfg.f1_parent_cond_handle = parent_f1_handle`. |
| P8 | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | +12 LOC at line 5994 | Insert `parent_f1_node = mono_pipeline.wire_parent_f1_switch(parent_graph, child_node)?`. |
| P9 | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | edit line 6001 | Change `child_node` → `parent_f1_node` in the `wire_parent_g26_switch` call's `dependency_node` arg. |
| P10 | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | edit lines 6019-6024 (audit log) | Update expected `parent_conditional_nodes` from 1 → 2 in the post-G26 audit log. |
| P11 | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | +25 LOC after line 6862 | Emit F1 decision sidecar record (see §6). |

### Optional polish (non-blocking)

| Item | Where | Why |
|------|-------|-----|
| Move `prism_f1_*_ffi` and `prism_gearbox_*_ffi` parent-owned helpers into a new `parent_conditional.cu` TU | new file under `crates/prism-nhs/src/cuda/` | Cleaner registry; today they hide inside `adjudicator.cu`. Strictly cosmetic; do AFTER F1 lands and the gate stays green. |
| Add a `splice_legality_check` invocation immediately AFTER `wire_parent_f1_switch` and BEFORE `wire_parent_g26_switch` | nhs_rt_full.rs:5994 | Catches accidental F1-inside-child regressions. |

---

## 6. Output contract / sidecar emission

When F1 fires, emit per-decision (one record per launch where F1 actually
selected a non-default branch — gated by `adj->adjudication_code != 0`):

| Field | Type | Source |
|-------|------|--------|
| `stream_id` | u32 | `i` in `nhs_rt_full.rs` per-stream loop |
| `chunk_id` | u64 | `chunk_idx` in same loop (line ~6837) |
| `phase` | str | `"capture"` (during build) or `"replay"` (during launch); for the sidecar, always `"replay"` since F1 fires at launch |
| `f1_predicate_value` | u32 | `adj->adjudication_code` read back via `cuMemcpyDtoH` (1 word, 4 B; same approach as `read_burst_marker` at captured_pipeline.rs:4683) |
| `f1_branch_taken` | u32 | identical to `f1_predicate_value & 0x3` (no extra read; derived) |
| `f1_condition_handle` | u64 | `mono_cfg.f1_parent_cond_handle` (constant per replay; recorded once and forward-referenced by handle ID) |
| `f1_branch_trace_hash` | u64 | BLAKE3-of-(stream_id, chunk_id, f1_predicate_value, frame_id) low-64; deterministic for replay verification |
| `f1_exit_status` | u8 | 0 = clean; 1 = trap fired (case 2 / Violation); 2 = launch error after trap |

### Persistence path

Per-stream NDJSON sidecar at `<output_dir>/prism_f1_decisions_pid<pid>_stream<i>.ndjson`,
one record per fired decision. **Reuses Rust's BufWriter pattern from TIER 3b**
(per-stream telemetry persistence, EXECUTION_POLICY.md:43; existing
analogues:
[ghost_telemetry.rs:536](../crates/prism-nhs/src/ghost_telemetry.rs)
log emission already streams decisions, but as `log::info!` not NDJSON).

The sidecar matches Agent 1's planned DAG node kind `F1Decision` —
**Agent 1's deliverable doc was not present in tree at the time of this
scout** (`grep -rln "F1Decision\|DAGNodeKind" docs/` returned empty).
The plan above defines what F1 will emit; Agent 1 should align the
DAG-node persistence schema to consume these eight fields.

### Code site

In `nhs_rt_full.rs` immediately after the existing `log_f1_switch_events`
call (line 6857-6862). The branch is:

```ignore
// Existing (host-side scraper, unchanged):
prism_nhs::ghost_telemetry::log_f1_switch_events(
    pipeline.ring(), frame_id, i,
);
// New (F1 decision sidecar):
let pred_val = pipeline.read_adjudication_code()?;  // 4-byte DtoH
if pred_val != 0 {
    f1_decision_writer.emit(F1DecisionRecord {
        stream_id: i as u32,
        chunk_id: chunk_idx as u64,
        phase: "replay",
        f1_predicate_value: pred_val,
        f1_branch_taken: pred_val & 0x3,
        f1_condition_handle: mono_cfg.f1_parent_cond_handle,
        f1_branch_trace_hash: blake3_hash_low64(...),
        f1_exit_status: 0,
    })?;
}
```

`read_adjudication_code` is a new accessor on
`CapturedAdjudicationPipeline` (~10 LOC; mirrors `read_burst_marker`
at line 4683).

---

## 7. New structs / functions (full inventory)

### Rust (`captured_pipeline.rs`)

| Symbol | Kind | LOC | Mirrors |
|--------|------|-----|---------|
| `PipelineConfig.f1_parent_cond_handle: u64` | field | 1 + doc | `g26_parent_cond_handle` |
| `extern "C" prism_f1_create_handle_ffi` | extern decl | 5 | `prism_gearbox_create_handle_ffi` |
| `extern "C" prism_f1_wire_with_handle_ffi` | extern decl | 7 | `prism_gearbox_wire_with_handle_ffi` |
| `extern "C" prism_f1_populate_switch_bodies_ffi` | extern decl | 4 | `prism_gearbox_populate_switch_bodies_ffi` (smaller) |
| `extern "C" prism_f1_add_parent_bridge_ffi` | extern decl | 7 | (no direct G26 analogue — G26's bridge is inside the child) |
| `CapturedAdjudicationPipeline::create_parent_f1_cond_handle` | method | 30 | `create_parent_g26_cond_handle` |
| `CapturedAdjudicationPipeline::wire_parent_f1_switch` | method | 60 | `wire_parent_g26_switch` |
| `CapturedAdjudicationPipeline::read_adjudication_code` | method | 12 | `read_burst_marker` |
| `F1DecisionRecord` (NDJSON serde struct) | struct | 12 | new |
| `F1DecisionWriter` (BufWriter wrapper) | struct + impl | ~40 | TIER 3b telemetry writer pattern |

### CUDA / C++ (`adjudicator.cu` + `.cuh`)

| Symbol | Kind | LOC | Mirrors |
|--------|------|-----|---------|
| `prism_f1_predicate_bridge_kernel` | `__global__` | 18 | `prism_gearbox_predicate_bridge_kernel` (gearbox.cu:437) |
| `prism_f1_violation_trap_kernel` | `__global__` | 6 | reuse existing trap pattern (`asm volatile("trap;");`) |
| `prism_f1_create_handle_ffi` | `extern "C"` | 18 | `prism_gearbox_create_handle_ffi` (adjudicator.cu:1119) |
| `prism_f1_add_parent_bridge_ffi` | `extern "C"` | 50 | NEW — assembles a kernel-node via `cudaGraphAddNode(KERNEL,…)` with deps=[child_splice_node] |
| `prism_f1_wire_with_handle_ffi` | `extern "C"` | 50 | `prism_gearbox_wire_with_handle_ffi` with `size=3` |
| `prism_f1_populate_switch_bodies_ffi` | `extern "C"` | 35 | populates 3 bodies (case 0 empty; case 1 empty; case 2 trap kernel node) |
| Pre-CUDA-12.6 fallback stubs returning `cudaErrorNotSupported` | x4 | 20 | matches existing `#if CUDART_VERSION < 12060` pattern |

### Binary (`nhs_rt_full.rs`)

| Symbol | Kind | LOC |
|--------|------|-----|
| `parent_f1_handle` local | `let` | 1 |
| `parent_f1_node` local | `let` | 1 |
| F1 sidecar emit block | inline | 25 |
| Per-stream `f1_decision_writer` lifetime | new variable in stream-spawn | 3 |

Total Rust + C++ + bin: ~400 LOC, all additive. No deletions, no
behavioural change to the overlay path (`f1_parent_cond_handle = 0`
preserves today's no-op-at-CUDA-graph-layer behaviour for non-monolithic).

---

## 8. Acceptance tests

All evaluated under `--m1-monolithic-discovery --multi-stream 8 --hysteresis --no-autonomous-rescue`.

| # | Check | Method | Pass criterion |
|---|-------|--------|----------------|
| A1 | `cargo check -p prism-nhs` | `cargo check` | exit 0; no warnings on the new symbols |
| A2 | `cargo build -p prism-nhs --release` | `cargo build --release` | exit 0; static archives compile (`adjudicator.a` rebuilt) |
| A3 | `splice_legality_check(child_template) == legal` | `[TIER8-PREFLIGHT … child_…]` line in run.log | `child_conditional_nodes=0 child_alloc_nodes=0 child_free_nodes=0` |
| A4 | `splice_legality_check(parent_graph)` post-F1 wire | `[TIER8-PREFLIGHT … parent_…]` after `wire_parent_f1_switch` | `parent_conditional_nodes=1` (F1 only at this point) |
| A5 | `splice_legality_check(parent_graph)` post-G26 wire | `[TIER8-PREFLIGHT … parent_…]` after `wire_parent_g26_switch` | `parent_conditional_nodes=2` (F1 + G26) |
| A6 | All 8 streams instantiate | grep `MONO-FUSE … instantiated` and `V2-INSTANTIATE-COMPLETE` in run.log | 8/8 lines, mirror of TIER8 freeze evidence |
| A7 | No CUDA error 801 (`cudaErrorNotSupported`) | grep `rc=801` in run.log | zero matches |
| A8 | No 900/901 (capture invalidation / unjoined) | grep `rc=900\|rc=901` | zero matches |
| A9 | No `STREAM_CAPTURE_INVALIDATED` | grep run.log | zero matches |
| A10 | F1 trace sidecar emitted | `ls <output>/prism_f1_decisions_*.ndjson` | one file per stream; nonzero size when target produces non-Prune adjudications |
| A11 | F1 sidecar schema matches §6 | `jq` on first line of any sidecar | all 8 keys present, types match |
| A12 | Replay determinism | run twice with `--replica-seed 42`; diff sidecars | byte-identical sidecar output |
| A13 | F1 violation case actually traps | inject synthetic `adj->adjudication_code = 2` for one frame; expect `f1_exit_status=1` | sidecar shows the trap was reached |

---

## 9. Failure modes

| Mode | Symptom | Mitigation |
|------|---------|------------|
| F1 SWITCH installed inside child template (operator NO-GO) | A3 fails: `child_conditional_nodes >= 1`; `cudaGraphAddNode(GRAPH)` returns 801 | Acceptance gate A3 fires before any splice attempt; CI blocks |
| F1 handle created on child graph instead of parent | A4 reports `parent_conditional_nodes=0`; CUDA SetConditional silently no-ops; F1 always picks default (Prune) | Test fixture: assert `parent_f1_handle != 0` and that the parent graph (not the child) owns the handle |
| Bridge kernel node added to child template instead of parent | Capture window includes a kernel node that the child copies-by-value; predicate writes to a stale handle | F1 bridge MUST be added to parent_graph via `prism_f1_add_parent_bridge_ffi`, never inside the child capture window |
| F1 SWITCH dependency points at child_node directly, bypassing the bridge | `cudaGraphSetConditional` race: SWITCH may execute before bridge writes | Mandatory: SWITCH deps = [bridge_node], bridge deps = [child_node] |
| WHILE-form regression (someone duplicates G26's bridge but switches `cudaGraphCondTypeSwitch` → `cudaGraphCondTypeWhile`) | Unbounded F1 loop at runtime | Hard-code `cudaGraphCondTypeSwitch` and `size=3` in `prism_f1_wire_with_handle_ffi` (no parameterisation) |
| Parent graph's preflight trips the alloc/free guard | F1 helpers accidentally introduce a memory-pool node | Use only kernel + conditional + dependency primitives; no `cuMemAllocAsync` |
| Sidecar emit corrupts pipeline timing budget (>200µs synchronous DtoH per chunk) | Frame-rate degradation at high chunk count | Use F1's existing event-paced read pattern (same as `log_f1_switch_events`); only emit when `pred_val != 0`; emit at chunk boundary, not per launch |
| `read_adjudication_code` synchronises the stream | Forces full GPU-CPU barrier; kills monolithic latency | Use the pinned ring (already used by F1 host scraper) — the data is already DMA'd to host. No new DtoH needed. **REVISE §6 / P11: read from `pipeline.ring().read_slot_unchecked(frame_id - 2)` and use `tile.adjudication_code`.** |
| F1 wire fails partway, parent graph leaks | `cuGraphDestroy(parent_graph)` not called | `'mono_attempt: { … break … }` block at nhs_rt_full.rs:5894 already handles parent destroy via `auto_tmpl` Drop impl; F1 plan must `break 'mono_attempt` on every error path |
| Test `monolithic_pipeline_v2_ignition_smoke_with_hook` (line 5491) drift | Test asserts `cond_node != 0` post-hook; if F1 Rust API changes types, the test breaks | Plan does NOT touch the test path; existing test uses the legacy `prism_wire_f1_switch_ffi` directly |

---

## 10. Rollback

If a regression introduces 801 / 900 / 901 / `STREAM_CAPTURE_INVALIDATED`:

1. Set `mono_cfg.f1_parent_cond_handle = 0` at nhs_rt_full.rs:5945. F1
   becomes a no-op at the CUDA-graph layer (host-side scraper still
   fires). All 8 streams instantiate as today.
2. Skip the `wire_parent_f1_switch` call at nhs_rt_full.rs:5994. G26's
   dependency reverts to `child_node`. G26 + ChildAdj remain wired.
3. Delete the sidecar emit at nhs_rt_full.rs:6862 (or guard behind
   `if mono_cfg.f1_parent_cond_handle != 0`).
4. The C++ FFI symbols stay compiled (dead code) but cause no runtime
   work — they are pure functions invoked only when the handle field is
   nonzero.
5. Verify with the same all-8 gate command that produced freeze evidence
   (TIER8_GRAPH_TOPOLOGY.md:113-115).
6. Worst case: revert single commit. The plan keeps every change
   additive — no existing function bodies are modified except 2 lines in
   nhs_rt_full.rs (G26 dependency edge swap + audit log expectation).

The kill switch is at **two locations** (P7 handle assignment + P8 wire
call). Either alone disables F1. This bounds the blast radius.

---

## Coordination notes

- **Agent 1 alignment.** Agent 1's DAG-node-kind plan was not present in
  the docs tree at scout time (`grep -rln 'F1Decision\|DAGNodeKind'
  docs/` empty). Agent 1 should mint a `DAGNodeKind::F1Decision` whose
  schema matches the eight fields in §6. The NDJSON path defined in §6
  (`prism_f1_decisions_pid*_stream*.ndjson`) is the persistence sidecar
  Agent 1's DAG aggregator should consume.
- **Test status (today).** The only call to `prism_wire_f1_switch_ffi`
  is in
  `monolithic_pipeline_v2_ignition_smoke_with_hook`
  (captured_pipeline.rs:5491). It tests the **legacy in-graph** F1 SWITCH
  path. The new parent-owned helpers do not interact with this test;
  it stays green. After F1 lands, optionally add a parallel test
  `monolithic_pipeline_parent_owned_f1_smoke` exercising the new
  `wire_parent_f1_switch` against a synthetic parent graph.
- **Glass box.** All TIER8-PREFLIGHT log lines must be retained. Adding
  F1 doubles the conditional-node count on the parent post-instantiate;
  the audit lines at nhs_rt_full.rs:6019-6024 must be updated to expect
  `parent_conditional_nodes=2`, not 1 — this is the audit drift Agent 1
  must NOT silently rebase past.
