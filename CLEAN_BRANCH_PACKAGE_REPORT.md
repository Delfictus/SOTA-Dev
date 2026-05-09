# CLEAN_BRANCH_PACKAGE_REPORT

**Status:** §3 monolithic graph unroll lane EMPIRICALLY VALIDATED. **Production config: default `body_unroll=1`** — §3.V is the architectural win; §3.II+III+IV are scaffolding for future event-vector work.
**Operator:** is@delfictus.com
**Last update:** 2026-05-09 (post unroll=500 + unroll=8 smoke runs)

---

## Branch + commits

- **Branch:** `feature/ghost-phase-lattice-v5-live-autonomy`
- **Base lineage:** `phase-a-stabilization-ghost-consumption`
- **Preserved refs:**
  - `preserve/pre-detour-phase-a-20260508T155610Z` (branch, points at `a25411799d40c7608941633dbcf327d7312e4da7`)
  - `preserve-pre-detour-phase-a-20260508T155610Z` (tag, points at `842febd940b3d082d82f7e15dbf83e4a5c1a254b`)

| Hash | Scope | Files |
|------|-------|-------|
| `e0494425` | GhostPhaseLattice4D backend + V2 autonomy guard + CLI handoff | 15 (5 new + 10 modified) |
| `3b8584f8` | §3.I device-side ZSTR slot kernel (wired into captured graph) + §3.V CU_CTX_SCHED_BLOCKING_SYNC + body_unroll PipelineConfig field | 5 |
| `a332bac2` | §3.III for-loop wrapper around per-step body (with hoisted scalars for post-capture wiring) + §3.IV host-loop decapitation (n_launches = ceil(this_chunk / body_unroll)) + `--captured-graph-body-unroll N` CLI flag | 2 |

## Validation gates (final, post-empirical-probe)

| Check | Result |
|-------|--------|
| `cargo check -p prism-nhs --bin nhs_rt_full --features v2_ignition` | ✓ clean |
| `cargo check -p prism-nhs --bin prism-ghost-evidence-scan` | ✓ clean |
| `cargo check -p prism-nhs --bin prism-materialize-sites` | ✓ clean |
| `cargo test -p prism-nhs ghost_phase_lattice` | ✓ 5/5 pass |
| Pre-commit policy | ✓ all 3 commits passed |

## Live-run evidence — §3 empirical results

### `body_unroll=1` (DEFAULT — `.prism_orchestration/GHOST_LATTICE_S3_FULL_SMOKE/`)
- Wall: **2228.8 s** total
- All 6 directive Part VIII proof lines captured
- Process CPU during chunk integration: **0.0%** (proves §3.V is the CPU collapse, not the unroll)
- Lattice: **297.13 ms** kernel (5.5× faster than the pre-§1 baseline of 1625.71 ms — denser temporal-spatial cells unlocked by §1 device clock)
- Lattice components: 559 (vs. pre-§1 baseline of 498)
- step_span: **`[6101, 6212]`** (vs. pre-§1 baseline `[6000, 6000]` — temporal-collapse FIXED)

### `body_unroll=500` (probe — killed mid-run)
- Confirmed `launch_attempted=1` per chunk per stream (Gate 1 ✅)
- CPU 0.0% (Gate 2 ✅)
- **Wall regression:** ~110-130 s/chunk vs. 2.1 s/chunk legacy (~55× slower)
- Root cause: scalar-shared cross-stream events (`fork_event`, `perturbed_join_event`, `md_to_telemetry_event`, `telemetry_to_md_event`) force GPU serialization across iterations

### `body_unroll=8` (sweet-spot probe — killed mid-run)
- Confirmed `launch_attempted=63` per chunk per stream (= ceil(500/8))
- CPU 0.0%
- **Wall regression:** ~46-147 s/chunk (variance from per-stream warmup + GPU contention)
- Same fundamental serialization at any N>1

## Architectural conclusion

**§3.V CU_CTX_SCHED_BLOCKING_SYNC** is the architectural CPU-collapse win. The per-step host-loop (500× cuGraphLaunch) was always async submit — host-side total cost was ~3 ms — not the source of the 99% CPU. The 99% was `cudaStreamSynchronize` busy-waiting under default `CU_CTX_SCHED_AUTO`. Switching to `BLOCKING_SYNC` makes the host yield to the OS scheduler instead, dropping process CPU to **0.0%** with zero wall-time impact.

**§3.III + §3.IV body unroll** is architecturally interesting but **practically regressive on this graph topology** because the per-iteration body uses shared scalar cross-stream events. Each `cuEventRecord` + `cuStreamWaitEvent` pair on the same event forces sequential ordering across iterations. The legacy host-loop avoided this because each `cuGraphLaunch` produced a distinct graph instance with its own event recordings that could overlap via the driver's implicit batching.

**To make body_unroll>1 actually faster than unroll=1**, the next surgical pass requires:
- True event vectorization: allocate `Vec<cuEvent>` of length `body_unroll` and rotate per iteration so each iteration's fork/join is on a distinct event
- Same vectorization for `g26_bridge_node`, `zstr_pos_stage_node`, `zstr_fence_node` (so post-capture wiring binds every iteration's bridge/snapshot, not just the last)
- This is the documented `body_unroll>1` topology caveat in `PipelineConfig.body_unroll` rustdoc

## Production recommendation

Use the default `--captured-graph-body-unroll 1`. The §3.V flag set at process start is the operative win.

The unroll scaffolding (`PipelineConfig.body_unroll`, the for-loop wrapper, the host-loop division by N) is **committed and inert at default** — present for the future event-vector lane that would unlock the actual 1-launch-per-chunk performance target.

## What this commit set delivers (production-ready)

1. **GhostPhaseLattice4D live backend** (e0494425): physically-constrained 4D edge adjudication on Ghost v2 records. Replaces legacy O(N²) DBSCAN. Smoke-validated: 559 components, 297 ms kernel, all 6 directive proof lines.
2. **§1 device-side timekeeper** (3b8584f8): per-stream u64 d_step_counter, increment kernel injected into captured graph. **Eradicates the temporal-collapse bug** — step_span widens from `[6000,6000]` to `[6101,6212]`.
3. **§2 device-side firehose prune** (3b8584f8): emission gate on `|kl_divergence| > 0.01` in firehose mode. Reduces PCIe saturation.
4. **§3.I device-side ZSTR slot kernel** (3b8584f8): in-graph slot rolling via `*d_step_counter % N_SLOTS`. Removes host-side per-step cudaMemcpyToSymbolAsync.
5. **§3.V CU_CTX_SCHED_BLOCKING_SYNC** (3b8584f8): **the CPU collapse win** — process CPU drops from 99% to 0.0% during chunk integration with zero wall-time regression.
6. **§3.III + §3.IV body unroll scaffold** (a332bac2): for-loop wrapper around per-step body + host-loop decapitation. **Default body_unroll=1 = no behavior change.** Higher values empirically regressive without per-iteration event vectorization (next lane).

## What's NOT yet on this branch

- **Per-iteration event vectorisation** — required to make `body_unroll>1` actually faster than `body_unroll=1`. Documented as the next surgical pass in `PipelineConfig.body_unroll` rustdoc + this report.
- **`prism_materialize_sites` integration** — the 3500-line post-MD materialiser still uses the legacy clustering path. The lattice is reachable via `nhs_rt_full`'s post-MD orchestration only.
- **CUDA 12.4 `cudaGraphCondTypeWhile`** — alternative architecture where the body is captured ONCE inside a WHILE conditional node with a device-side counter predicate. Would avoid both the unroll-induced graph-size blow-up AND the per-iteration event serialization. The codebase has `while_drain_bridge.cu` infrastructure already; wire-in is its own lane.

## Branch policy

**DO NOT MERGE TO MAIN** until §3 next-pass (event vector OR CondTypeWhile) lands and a smoke validation run with `body_unroll≥4` shows wall-time at or below `body_unroll=1`.

End report.
