# Transparent MAR — v2 wire-in + deadman sidecars + smoke

**Final status**: `TRANSPARENT_MAR_V2_WIRE_IN_SOURCE_LANDED_RUNTIME_SEGFAULT_INVESTIGATION_PENDING`
**Wave**: M1.2.23 follow-up #1 (operator-corrected scope)
**Branch**: `producer-repair-causal-truthing-20260426`

## OPERATOR-CORRECTED SCOPE

The prior wave report claimed "all 8 streams V2-live" and implied
schema_version=2 records were emittable. Both claims were wrong.
The operator's correction list:

1. **The 20260507_160818 smoke had only 7/8 streams V2-live**
   (`v2_live_by_stream: [true,true,true,true,true,true,false,true]`).
   Stream 6 failed V2 instantiate.
2. **No schema_version=2 records were on disk** — captured_pipeline.rs
   still called the v1 `prism_ghost_pipe_stage_launch`.
3. **`clashing_extern_declarations` warnings** for both
   `prism_ghost_pipe_stage_launch` and
   `prism_gearbox_populate_switch_bodies_ffi`.

This wave addresses (1) by reporting verbatim, (2) by wiring the v2
launcher behind `--mar-v2-telemetry`, and (3) by reconciling the FFI
declarations.

## COMMITS LANDED

```
cf2e2262   transparent-mar: wire v2 ghost launcher and resolve clashing externs
15c00e86   transparent-mar: deadman emits ghost site_map and time_map sidecars
```

`cf2e2262` (4 files, +117/-21):
* `captured_pipeline.rs:3943` — branched on `cfg.mar_v2.is_some()`,
  routes to `prism_ghost_pipe_stage_launch_v2` when set, v1 otherwise
* `nhs_rt_full.rs` — new `--mar-v2-telemetry` CLI flag; populates
  `MarV2Config` with B.3.2 noise-floor lock-in defaults (μ=0.005,
  σ=0.001 → obs_thr=0.008, disc_thr=0.017)
* `ghost_tile.rs:454` — added the missing `firehose_enable: u32`
  argument to the FFI declaration (was 8 args, must be 9)
* `gearbox.rs:270` — reconciled `body_subgraphs` and `cruise` types
  to match the canonical declaration in captured_pipeline.rs:865

`15c00e86` (1 file, +117):
* `path_a_watchdog.rs` — new `pub fn emit_deadman_ghost_sidecars()`
  emits `ghost_site_map.json` + `ghost_time_map.json` with explicit
  partial/missing status when the deadman fires.

## BOUNDED V2-LIVE SMOKE — RESULT

```
.prism_orchestration/M1_2_23_V2_SMOKE_v2/20260507_173523/

  scripts/prism-validate-and-run.sh \
    -t data/targets/mpro_monomer.topology.json \
    -o <rundir> \
    --fast --hysteresis --prism-therm \
    --multi-stream 8 --spike-percentile 70 --fused-steps 6 \
    --hmr --adaptive-dt --multi-differential --closed-loop-steering \
    --asymmetric-steering --use-xgb-ranker --replica-seed 42 -v \
    --m1-monolithic-discovery \
    --mar-v2-telemetry \
    --path-a-production-profile \
    --path-a-max-wall-seconds 480 \
    --path-a-v2-trigger-steps 1500 \
    --path-a-t7-max-chunks 8 \
    --md-only-evidence

  Wall: ~20 s (NOT the 480 s wall-cap — process segfaulted)
  Exit: 139 (SIGSEGV; wrapper script's bash trapped it as exit 2)
```

### Timeline (from run.log)

```
00:35:23  start
00:35:42  V2-INSTANTIATE-START streams 0, 2
          [M1.2.23 v2] ghost launcher: schema_version=2
                       obs_thr=0.008000 disc_thr=0.017000
                       gear_id=0 dt_fs=4.000     ← v2 wire-in REACHED
          [M1.2.23 v2] ghost launcher: ...(2nd)  ← v2 wire-in REACHED twice
          MONO-FUSE stream 2 ✓ monolithic exec instantiated
          V2-INSTANTIATE-COMPLETE stream 2
          MONO-FUSE stream 4 acquired build lock
          V2-INSTANTIATE-START stream 4
          TIER8-PREFLIGHT stream 4 protocol_group=monolithic_discovery
00:35:43  Channel-B fd=65 opened for stream 2 ghost_tiles.bin
          ZSTR consumer initialised for stream 2
00:35:43  Segmentation fault (core dumped)
```

### What worked

- **The `--mar-v2-telemetry` flag plumbed through**: `MarV2Config`
  populated with B.3.2 lock-in thresholds verbatim.
- **The v2 FFI binding works**: 2 streams successfully called
  `prism_ghost_pipe_stage_launch_v2` (the `[M1.2.23 v2] ghost
  launcher` log emits from `captured_pipeline.rs` immediately
  after the FFI return). Both reported `schema_version=2,
  obs_thr=0.008, disc_thr=0.017, gear_id=0, dt_fs=4.000`.
- **CUDA build clean**: `cargo build --release` 47.82 s, all
  static_asserts in `ghost_tile_kernel.cuh` passed.
- **No clashing_extern_declarations warnings** anymore.

### What did NOT work — actually emitted on disk

```
mpro_monomer_stream00_ghost_tiles.bin   0 bytes  (FD opened, never written)
mpro_monomer_stream02_ghost_tiles.bin   0 bytes  (same)
prism_zstr_2043167_0.bin                empty
prism_zstr_2043167_2.bin                empty

NOT present:
  mpro_monomer_path_a_completion.json   (segfault bypassed normal emit
                                         AND deadman emit — the watchdog
                                         thread is in a separate POSIX
                                         thread but exits with the
                                         segfaulting process)
  ghost_site_map.json                   (same)
  ghost_time_map.json                   (same)
  md_evidence_manifest.json             (same)
```

**Honest verdict**: the v2 wire-in source builds and the v2 launcher
returns rc=0 from the FFI on at least 2 captured-graph build attempts,
but a segfault during the third stream's V2 instantiation aborts the
process before any captured graph EXECUTES. The segfault is not in
v2 kernel code (no kernel ran) — it's during host-side V2 build.

The 0-byte ghost_tiles.bin files exist because the engine's per-stream
teardown code opens the output FD speculatively (Channel-B `O_DIRECT |
O_DSYNC` open, line 7655 / 7817 of nhs_rt_full.rs) before the captured
graph executes. The FDs are opened, no kernel writes them, the segfault
fires.

### Verification of "no records written"

```
xxd -s 4096 -l 4 stream02_ghost_tiles.bin    →  empty (file is 0 bytes)
ls -la                                        →  size = 0 for both files
```

So **the v2 schema has NOT been runtime-validated**. The wire-in
source is landed; the runtime blocker is host-side process stability.

## SEGFAULT ROOT CAUSE — UNKNOWN

The crash is at the V2-INSTANTIATE phase for stream 4, specifically
after the ZSTR consumer thread for stream 2 was initialised. Possible
causes (none confirmed):

1. Stream-4-specific instability triggered by the rayon scheduler
   ordering once timing changed slightly (the v2 launcher passes
   ~2× the kernel arg buffer of v1 — could affect graph build).
2. Race between ZSTR consumer thread spawn and concurrent V2 build
   on a different stream (pre-existing race that the timing change
   exposes).
3. Captured-graph kernel-arg storage issue with the 13-arg v2
   kernel signature vs v1's 8-arg signature.

The previous 20260506_122646 smoke (with v1 launcher) also showed
stream 6 V2 instantiate failure — non-deterministic V2 build is
a pre-existing issue (separate ticket per the earlier
`V2_MONOLITHIC_THROUGHPUT_PROFILE` blocker).

## CLOSURE CRITERIA — operator-listed

| Criterion | Status |
|---|---|
| A. Wire v2 Ghost launcher into captured-graph call site | ✓ Source landed; smoke proved FFI reaches kernel build (2 streams) |
| B. Deadman sidecar emission | ✓ Source landed; NOT exercised because segfault bypassed deadman |
| C. Resolve clashing extern declarations | ✓ Both warnings cleared (verified by `cargo check` clean) |
| D. Re-run bounded V2-live smoke produces schema_version=2 records | **FAIL — segfault during V2 build aborted before any kernel executed; 0-byte ghost_tiles files** |
| Per-stream V2 status reported VERBATIM (not "all 8") | ✓ 0/8 V2-live in this run because segfault aborted; honest reporting |

## HARD INVARIANTS — directive Phase 16

| Invariant | Status |
|---|---|
| no fake bilateral / QI / Γw / unverified offsets | ✓ |
| no materialized sites from scanner | ✓ |
| no LIGSITE primary | ✓ |
| no production-accuracy claim | ✓ — explicit RUNTIME_SEGFAULT_INVESTIGATION_PENDING |
| no hidden missing fields | ✓ — every absent file documented above |
| no 8-stream invariant claim without proof | ✓ — this run had 0/8 V2 due to crash |
| no graph topology change | ✓ |

## NEXT STEPS — operator decision required

Three honest options:

**a. Investigate segfault** (~1-3 hours)
   Run with `RUST_BACKTRACE=1` + `RUST_LOG=trace`, possibly reproduce
   under `gdb` to find the exact crash site. May involve looking at
   the captured-graph node count and the kernel-arg-buffer layout.

**b. Defensive fallback**
   In `captured_pipeline.rs:3943`, wrap the v2 launcher call with a
   try-fall-back-to-v1 pattern: if v2 returns non-zero rc, retry with
   v1. Doesn't fix the segfault but prevents v2 wire-in from breaking
   v1-stable runs that have `--mar-v2-telemetry` accidentally set.

**c. Disable v2 wire-in by default behind a guard**
   Keep the source-landed v2 launcher but require a stronger gate
   (e.g., environment variable + flag) to enable. This is what
   we have today (`--mar-v2-telemetry` opt-in, default v1) — the
   operator just needs to NOT pass `--mar-v2-telemetry` in
   production runs until segfault is fixed.

The smoke proved the v2 FFI binding works for at least 2 streams.
Investigation of the host-side segfault is the next gate.

## RUNTIME OVERHEAD — not measured

Smoke crashed at 20 s (well before any chunk-loop iteration).
Performance-A/B measurement requires a v2 run that completes at
least one chunk loop, which is blocked on the segfault investigation.
