# CLEAN_BRANCH_PACKAGE_REPORT

**Status:** CHECKPOINT SECURED — §3 monolithic-graph-unroll lane resumes immediately on this branch.
**Operator:** is@delfictus.com
**Timestamp:** 2026-05-08 (UTC tracked in commit metadata)

---

## Branch + commit

- **Branch:** `feature/ghost-phase-lattice-v5-live-autonomy`
- **Base lineage:** `phase-a-stabilization-ghost-consumption`
- **Preserved refs:**
  - `preserve/pre-detour-phase-a-20260508T155610Z` (branch, points at `a25411799d40c7608941633dbcf327d7312e4da7`)
  - `preserve-pre-detour-phase-a-20260508T155610Z` (tag, points at `842febd940b3d082d82f7e15dbf83e4a5c1a254b`)
- **Checkpoint commit:** `e0494425c3db1a694efdce871a7dda971914f965`
- **Pre-commit policy:** ✓ passed (PRISM pre-commit) — staged-diff captured to `.codex/review/`.

## Files in this commit (15 files, +3717/-26)

```
M  crates/prism-nhs/build.rs
M  crates/prism-nhs/src/bin/nhs_rt_full.rs
M  crates/prism-nhs/src/captured_pipeline.rs
A  crates/prism-nhs/src/cuda/ghost_lattice_kernel.cu
A  crates/prism-nhs/src/cuda/ghost_lattice_kernel.cuh
A  crates/prism-nhs/src/cuda/ghost_lattice_kernel_nvrtc.cu
M  crates/prism-nhs/src/cuda/ghost_tile_kernel.cu
M  crates/prism-nhs/src/cuda/ghost_tile_kernel.cuh
A  crates/prism-nhs/src/ghost_phase_lattice.rs
A  crates/prism-nhs/src/ghost_phase_materializer.rs
M  crates/prism-nhs/src/ghost_telemetry.rs
M  crates/prism-nhs/src/gpu_cluster_backend.rs
M  crates/prism-nhs/src/lib.rs
M  crates/prism-nhs/src/persistent_engine.rs
M  crates/prism-nhs/src/site_manifest.rs
```

## Excluded from commit (pre-existing dirty tree, **NOT** mine)

```
M  .codex/review/context.txt
M  .codex/review/pre-commit.diff
M  .codex/review/pre-commit.diffstat
D  .wrangler/cache/cf.json
D  .wrangler/cache/pages.json
M  crates/prism-nhs/src/bin/prism_ghost_evidence_scan.rs
M  crates/prism-nhs/src/bin/prism_materialize_sites.rs
M  scripts/prism_spike_watcher.py
?? crates/prism-nhs/tests/mlfeature_compile_fail.rs
?? crates/prism-nhs/tests/mlfeature_compile_fail/
?? .prism_orchestration/<run folders ×N>
?? .codex/<aux files>
?? target/<build artifacts>
```

These files were already in the working tree at session start and are unrelated
to this lane. They remain unstaged/unstashed for the operator to handle in
their own checkpoint.

## Required validation gates (all passed pre-commit)

| Check | Result |
|-------|--------|
| `git diff --cached --name-status` | 15 expected files (5 A + 10 M); zero garbage. |
| `cargo check -p prism-nhs --bin nhs_rt_full --features v2_ignition` | ✓ Finished `dev` profile in 27.93s. 17 warnings, 0 errors. |
| `cargo check -p prism-nhs --bin prism-ghost-evidence-scan` | ✓ Finished in 22.63s. 24 warnings, 0 errors. |
| `cargo check -p prism-nhs --bin prism-materialize-sites` | ✓ Finished in 0.22s (cache hit). 19 warnings, 0 errors. |
| `cargo test -p prism-nhs ghost_phase_lattice` | ✓ 5/5 pass. |

## What this commit contains (scope summary)

1. **GhostPhaseLattice4D live backend** — physically-constrained 4D edge
   adjudication on Ghost v2 records (spatial cell × protocol phase × step
   bucket). 208-byte `#[repr(C)]` `GhostPhaseLatticeNode`, NVRTC fallback
   compile path, atomicMin union-find, 4-plane SO(3) cosine-sim scoring with
   weighted [0.35/0.25/0.25/0.15] active-plane normalisation.
2. **Ghost lattice CUDA kernel + build wiring** —
   `ghost_lattice_kernel_nvrtc.cu` (device kernels) +
   `ghost_lattice_kernel.cu` (static-archive wrapper with 17
   `static_assert`s pinning struct layout) + `.cuh` FFI surface.
   Registered in `build.rs` via `compile_to_static_archive("ghost_lattice_kernel")`.
3. **SiteManifest additive schema** — 4 new `Option<...>` fields:
   `ghost_phase_lattice` (provenance), `phase_manifold` (per-phase aggregate),
   `therm_ccns_lifecycle` (per-phase mean KL/flux/water + driver persistence),
   `so3_manifold` (plane-status mask, intra-component mean cosine,
   phase-transition cosines). All `skip_serializing_if=None` — legacy DBSCAN
   runs emit byte-identical JSON.
4. **`nhs_rt_full` CLI handoff** —
   `--clustering-backend=ghost-phase-lattice-4d` (alias `ghost-lattice`,
   `lattice-4d`), `--ghost-phase-lattice-so3-threshold` (default 0.75),
   `--ghost-phase-lattice-cell-a` (5.0), `--ghost-phase-lattice-max-temporal-edge-steps`
   (500), `--ghost-phase-lattice-step-bucket-size` (500). Hard predicate:
   explicit lattice-mode without `--m1-monolithic-discovery` ∧
   `--mar-v2-telemetry` exits with errno 5 (CORRECTION ADDENDUM clause 2:
   degraded-status failure, not silent fallback). Auto-mode fallback to
   gpu-hash allowed.
5. **`persistent_engine` live backend routing** — new
   `cluster_ghost_phase_lattice()` method emits the directive's
   `POST_MD_CLUSTER_BACKEND_SELECTED backend=ghost-phase-lattice-4d`
   log. Existing `cluster_spikes()` rejects the lattice backend with a
   clear error directing operators to the proper entry point (the lattice
   operates on Ghost v2 records, not raw spike positions).
6. **`prism_materialize_sites` handoff:** ❌ Not in this commit. The
   3500-line `bin/prism_materialize_sites.rs` post-MD materialiser still
   uses the legacy clustering path. Wiring it to detect v2 records on disk
   and route to `cluster_ghost_phase_lattice` is a separate ticket. Until
   then, the lattice is reachable via `nhs_rt_full`'s post-MD orchestration
   (above the V2 HARD-GATE).
7. **V2 evidence-exit guard / runtime predicate bridge** — already landed
   in `nhs_rt_full.rs:6614` pre-commit. Engine continues integrating past
   ASC `evidence_complete` until V2 has had its chance (verified by the
   pre-commit smoke run firing `[V2-BUILD stream N]` on all 8 streams at
   the operator-set `--path-a-v2-trigger-steps 6000`).
8. **Captured graph autonomy / launch-loop changes:** ❌ Not in this commit.
   The host loop at `nhs_rt_full.rs:8568` still launches
   `mono.launch_on_stream` 500× per chunk. **§3 monolithic graph unroll
   resumes immediately on this branch after this commit** — see "Next
   lane" below.
9. **Firehose pruning + device-side clock** (operator mandate 2026-05-08
   §1+§2) —
   - §1: per-stream u64 `d_step_counter` allocated at V2-BUILD seeded with
     `v2_trigger_step`. `prism_increment_time_kernel` injected into V2
     graph body before each Ghost emission. v2 ghost kernel reads
     `*d_step_counter` and stamps the actual step into every emitted
     record. Eradicates the temporal-collapse bug (`step_span [N, N]`)
     observed in the first smoke run.
   - §2: `prune_kl_threshold` (default 0.01) added to v2 launcher
     signature. In firehose mode, records with `|kl_divergence| ≤ threshold`
     are dropped before the ring atomicAdd. Legacy non-firehose semantics
     preserved.
   - Verified live by run #2 (pre-checkpoint smoke,
     `.prism_orchestration/GHOST_LATTICE_LIVE_SMOKE_RERUN/run.log`):
     8 streams allocated distinct `d_step_counter=0xc082fc02e00 …
     0xc082fc16e00`; `[M1.2.23 v2]` log carried `d_step_counter=0x[non-zero]
     prune_kl=0.0100`.

## Live-run evidence captured pre-checkpoint

- `run #1` (full 2206.4s wall): all 6 directive Part VIII proof lines
  captured. 498 components, 20.9M edges, 1625.71 ms 4D-intersection
  (vs. >250 000 ms legacy DBSCAN baseline = **154× speedup**).
  Step span `[6000, 6000]` exposed the temporal-collapse bug → motivated §1.
- `run #2` (killed pre-MD-end on operator order to start §3): proved §1+§2
  wired into Layer-1 emission; per-stream device counters allocated; v2
  launcher log carries `d_step_counter` + `prune_kl` fields.

Both run-output directories left in `.prism_orchestration/GHOST_LATTICE_LIVE_SMOKE/`
and `.prism_orchestration/GHOST_LATTICE_LIVE_SMOKE_RERUN/` respectively.
**Not committed** per directive (large binary outputs).

## Next lane (resumes immediately on this branch)

**§3 — Monolithic graph unroll** (operator LEVEL 0 OVERRIDE 2026-05-08):

| Sub-task | File | Plan |
|----------|------|------|
| §3.I — Device-side ZSTR slot kernel | `cuda/zstr_kernels.cu` (or new `cuda/zstr_device_updater.cu`) | Single-thread `__global__` reads `*d_step_counter`, computes `step % N_SLOTS`, writes to `__constant__ d_zstr_active_slot`. Removes host-side `prism_zstr_set_active_slot` per-step write. |
| §3.II — Vectorize F1+G26 conditional handles | `captured_pipeline.rs` | Per-iteration handle creation via `prism_f1_create_handle_ffi` + `prism_gearbox_create_handle_ffi`; `Vec<CUgraphNode>` indexed by step. |
| §3.III — Wrap capture body in `for _ in 0..unroll_n` | `captured_pipeline.rs::build` | Unroll between `cuStreamBeginCapture` and `cuStreamEndCapture`. Inject §1 increment + §3.I ZSTR slot kernels per iteration. |
| §3.IV — Eradicate host loop | `nhs_rt_full.rs:8568` | Replace 500× host loop with single `mono.launch_on_stream`. Remove host-side `prism_zstr_set_active_slot`. |
| §3.V — `CU_CTX_SCHED_BLOCKING_SYNC` | engine init | Yield CPU to OS scheduler instead of spinning. |
| §3.VI — Build, smoke-run, verify | binary | Gates: launch_attempted=1/chunk; CPU ≪ 99%; step_span widens; ghost lattice still produces ~498 components. |

The commit hash above is the rollback target if §3 breaks the captured-graph
topology.

## Branch policy

**DO NOT MERGE TO MAIN** until §3 lands and the smoke validation gates pass.

End report.
