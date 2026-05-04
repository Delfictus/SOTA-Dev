# TIER 8 Graph Topology

Date: 2026-05-04
Status: PASS (all-8 instantiate gate frozen), with deferred-drain caveat tracked

## Invariant

GRAPH-SPLICE-001: any graph passed to `cudaGraphAddNode(GRAPH)` as a
child template must contain no conditional, allocation, or free nodes.
PRISM checks this before the CUDA splice call and reports the node
counts in Rust logs.

## Source-Truth Finding

The active splice-illegal conditional is G26, not F1.

- G26 is captured as SFA + predicate-bridge kernels, then its SWITCH
  conditional node is added post-capture in
  `captured_pipeline.rs`.
- F1 remains deferred/no-op in the production build path.
- The monolithic splice site is `nhs_rt_full.rs`: parent
  `CapturedTemplate` splices the V2 adjudication `CUgraph` between
  `fused_step` and `multi_lif`.

## Option Beta Topology

Parent/control graph:

```text
fused_step -> ChildAdj(GRAPH) -> G26_SWITCH(parent) -> multi_lif
```

Child adjudication template:

```text
SO3 -> Adjudicator -> ASC/Energy -> G26_SFA -> G26_predicate_bridge
     -> telemetry/ZSTR/ghost sidecars
```

The parent graph creates the G26 conditional handle before the
parent-owned child is captured.  The child predicate bridge receives
that handle as a kernel argument and calls `cudaGraphSetConditional`
at replay.  The child does not add a conditional node, so its splice
preflight must report:

```text
child_conditional_nodes = 0
child_alloc_nodes = 0
child_free_nodes = 0
```

After the child graph node is inserted, the parent graph wires the G26
SWITCH using the same handle.  The parent post-wire preflight should
report `parent_conditional_nodes >= 1`.

## Fallback Rule

When `--m1-monolithic-discovery` is selected, monolithic graph assembly
fails closed.  If parent handle creation, parent-owned child build,
splice, parent SWITCH wiring, dependency insertion, or instantiation
fails, V2 is disabled for that stream and the log must state that the
overlay fallback was not retained.

The standalone overlay pipeline remains an explicitly selectable
non-monolithic path.  It must not be used silently during this TIER 8
validation gate.

## Freeze Evidence (All-8 Gate PASS)

Validation run:

`/mnt/storage/prism_tier8_all8_4800_20260504_041001/run.log`

Per-stream instantiate evidence:

| Stream | MONO-FUSE instantiated | V2-INSTANTIATE-COMPLETE |
|---|---|---|
| 0 | 2026-05-04T11:51:38Z | 2026-05-04T11:51:38Z |
| 1 | 2026-05-04T11:51:36Z | 2026-05-04T11:51:36Z |
| 2 | 2026-05-04T11:12:56Z | 2026-05-04T11:12:56Z |
| 3 | 2026-05-04T11:12:30Z | 2026-05-04T11:12:30Z |
| 4 | 2026-05-04T11:27:07Z | 2026-05-04T11:27:07Z |
| 5 | 2026-05-04T11:27:19Z | 2026-05-04T11:27:19Z |
| 6 | 2026-05-04T11:11:06Z | 2026-05-04T11:11:06Z |
| 7 | 2026-05-04T11:11:06Z | 2026-05-04T11:11:06Z |

Gate checks satisfied in the same run:

- No `rc=801`
- No `rc=900`
- No `rc=901`
- No `STREAM_CAPTURE_INVALIDATED`
- No Director launch failure
- Child splice legality remained: `child_conditional_nodes=0`, `child_alloc_nodes=0`, `child_free_nodes=0`
- Parent G26 control remained present: `parent_conditional_nodes=1`
- CaptureGuard commit/cleanup path stayed healthy
- Post-run cleanup returned to baseline (`P8`, no compute process, VRAM ~15 MiB)

## Remaining Caveat

Deferred drain diagnostics still occurred and remain intentionally visible:

- `TIER8-CAPTURE deferred-error-state ... result=DRAINED ... deferred_drain_count=1`
- `TIER8-CAPTURE deferred-error-state ... result=DRAINED ... deferred_drain_count=2`
- One non-fatal `post-gasp-sync` drained check on stream 6

These did not prevent all-8 instantiate success, but they remain tracked and are not normalized away.

## Rollback Path

If regressions reintroduce hard INVALID_VALUE / 801 / 900 / 901 or capture invalidation:

1. Re-enable verbose diagnostics with `PRISM_TIER8_VERBOSE_DIAG=1`.
2. Re-run the focused all-8 gate command (`--m1-monolithic-discovery --multi-stream 8 --hysteresis --no-autonomous-rescue`).
3. Confirm whether failure is pre-capture context guard, post-gasp raw sync gate, Director launch, or capture-end cleanup.
4. If required, revert to the prior commit before the latest cleanup and re-run the same gate for binary comparison.
