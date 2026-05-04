# PRISM-4D Execution Policy

**Status:** Locked, 2026-05-03. Operator-binding. Do not relax without an explicit policy revision committed to this file.

## The three-layer rule

PRISM-4D's runtime is a strict three-layer system. Each layer owns a disjoint set of responsibilities. Crossing the boundary is a policy violation.

### Layer 1 — Rust

**Owns:** orchestration, lifecycle, typing, audit, safety boundaries, error propagation, persistence to host disk, JSON/serde, CLI parsing, logging, replicate management, threading model, FFI typed wrappers, BuildError taxonomy.

**Does not own:** kernel code, graph-node construction, stream capture, graph launch, branch selection at the device level, device buffer lifetime decisions (Rust may *trigger* allocation/free via FFI but the device-side mechanics live in CUDA).

### Layer 2 — CUDA / C++ FFI

**Owns:** kernels (`.cu` files compiled to PTX or static archive), graph-node construction (`cudaGraphAddNode`, `cudaGraphAddDependencies`, conditional handle creation, body subgraph population), stream capture begin/end, graph launch, stream execution semantics, device state mutations including `__constant__` symbol writes, VRAM pool internals, conditional-node bodies.

**Does not own:** orchestration policy, when-to-build decisions, replicate scheduling, audit serialization (CUDA writes raw bytes; Rust serializes to durable formats).

### Layer 3 — Python

**Owns:** offline artifact inspection, dossier aggregation, validation reports, plotting, post-run analysis. Lives in `scripts/` (with `scripts/quarantine/` for unblessed scripts pending operator approval per the SCRIPT EXECUTION POLICY).

**Forbidden in:** the runtime / hot path. Python must NEVER own:
- CUDA graph construction
- Stream capture begin/end
- Graph launch (`cuGraphLaunch` family)
- Branch selection (conditional-node predicates, SWITCH bodies)
- Device buffer lifetime
- Runtime orchestration (replicate scheduling, stream-thread management, in-flight error recovery)

**Allowed:** reading completed artifacts (binding_sites.json, kcc_visualization.json, ghost_tiles.bin, etc.), emitting derived analyses (cryptic_sites.json, dossier reports, plots), validating outputs against ground truth.

## Boundary enforcement

| Check | Layer | Enforcement |
|---|---|---|
| `cudaGraphAddNode` callers | CUDA only | All Rust call sites route through C++ FFI helpers in `crates/prism-nhs/src/cuda/graph_node.cu` (TIER 7). Direct `sys::cuGraphAdd*` from Rust is a violation. |
| Conditional handle creation | CUDA only | `cudaGraphConditionalHandleCreate` is called inside `prism_wire_*_ffi` C++ functions. Rust receives an opaque handle, never the API directly. |
| Splice legality preflight | Rust orchestration owns the policy decision; CUDA owns the inspection | Rust calls `prism_graph_is_splice_legal_ffi` (CUDA-side enumerator) and converts the result into a typed `BuildError::SpliceIllegal { ... }` BEFORE attempting `cudaGraphAddNode(GRAPH)`. PRISM-specific error fires before CUDA returns 801 NotSupported. |
| Stream capture mode | CUDA-side enum, Rust passes the value | TIER 6 fix: Rust selects `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` when calling `cuStreamBeginCapture` via cudarc. The mode is a CUDA primitive, the *choice* is orchestration. |
| Per-stream telemetry persistence | Rust | TIER 3b BufWriter writes to disk are Rust-side. CUDA produces records (binary on device, copied to host via `cuMemcpyDtoH`); Rust owns when, where, and what schema lands on disk. |
| Python in `scripts/quarantine/prism-aggregate-sites.py` (TIER 5) | Allowed (offline) | Reads engine artifacts after the run completes. Cannot be invoked from `nhs_rt_full`. Requires operator GO per SCRIPT EXECUTION POLICY. |

## Why this exists

The runtime is a real-time CUDA capture-and-launch pipeline. Each layer has a different failure model:

- **Rust failures** are typed, propagated via `Result`, recoverable.
- **CUDA failures** are async, returned as `cudaError_t` codes, often with ~100ms latency before symptoms surface; some (e.g. stream-capture invalidation) cascade across other streams.
- **Python failures** are interpreted-loop exceptions with full Python overhead; they do not survive the kind of latency budgets the captured graph operates within.

Mixing layers blurs the failure model. A Python call in the hot path means a GIL acquisition mid-capture, an unbounded interpreter latency budget, and an exception class the orchestrator's `BuildError` taxonomy can't reason about. A Rust call into raw `sys::cuGraphAdd*` from outside the C++ FFI boundary means a `cudaError_t` 801 surfaces deep in the capture pipeline instead of being caught at a typed PRISM-specific gate.

## Drift prevention

This file is the canonical contract. If a future change requires relaxing any rule:

1. The relaxation must be documented here BEFORE the code change lands.
2. The CLAUDE.md reference must update to point at the new section.
3. The boundary-enforcement table must add the new exception with a rationale.

Pre-merge gate: any PR that adds a Python file outside `scripts/` or invokes Python from a Rust runtime path must explain why this policy doesn't apply. Default action is rejection.

## Quick decision rubric

If you're about to write code and uncertain which layer it belongs in:

- **"Does this run during a captured graph launch?"** → CUDA only.
- **"Does this make a control-flow decision based on device state?"** → CUDA conditional node, with Rust orchestrating placement.
- **"Does this serialize state to disk for offline use?"** → Rust (audit boundary) or Python (offline aggregation).
- **"Does this read a finished artifact?"** → Python (offline) is fine.
- **"Does this allocate device memory?"** → CUDA owns the call (`cuMemAlloc_v2`, `cudaMemPool*`, etc.); Rust orchestrates the lifecycle.
- **"Am I tempted to call `sys::cuGraphAdd*` from Rust?"** → STOP. Use or extend `crates/prism-nhs/src/cuda/graph_node.cu`.

---

History:
- 2026-05-03 — Policy locked by operator after TIER 5–7 audit. Initial commit.
