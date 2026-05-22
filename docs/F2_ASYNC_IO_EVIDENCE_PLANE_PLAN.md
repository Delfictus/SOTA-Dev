# F2 / io_uring / DMA Evidence Plane Plan

Date: 2026-05-04
Status: PLAN ONLY — read-only scout (Agent 9, F2 / io_uring / DMA Evidence Plane Scout).
Author: Agent 9 (read-only). No Rust/CUDA/build/runtime files modified by this scout.

This plan defines the contract for three sidecar JSONs the runtime should
emit at run-end so that IO-commit completeness is auditable without
re-parsing the binary frame streams (`prism_zstr_*.bin`,
`*_streamNN_ghost_tiles.bin`):

- `f2_ring_status.json` — per-ring snapshot at run-end.
- `f2_write_commit_log.json` — per-artifact write-commit ledger.
- `f2_artifact_completeness.json` — run-end completeness verdict.

The emitters are **Rust** (Layer 1, per
[docs/EXECUTION_POLICY.md](EXECUTION_POLICY.md)). No Python in the runtime.
The plan does **not** rewrite `zstr.rs` or `ghost_telemetry.rs`; it adds
accounting on the orchestrator side and harvests existing fields that are
already populated by the consumer thread, the `ZstrRing`, and the
`GhostTileRing`.

The plan is also a **freeze contract** for the deferred-drain caveat
documented in
[docs/TIER8_GRAPH_TOPOLOGY.md:99-107](TIER8_GRAPH_TOPOLOGY.md):
the sidecars expose drain counts and stream provenance, they do not
normalize the diagnostic away.

---

## 1. Files inspected

| File | Lines | Why it is load-bearing |
|---|---|---|
| `crates/prism-nhs/src/zstr.rs` | 1–942 | `ZstrRing`, `ZstrFrameHeader`, `ZstrStats`, `spawn_zstr_consumer` (G21 alignment, G24 Reaper, G29 force-norm trap, io_uring SQ/CQ, Channel-A O_DIRECT, Channel-B O_DIRECT|O_DSYNC). |
| `crates/prism-nhs/src/ghost_tile.rs` | 1–453 | `GhostTileFrame` (4096 B sector-aligned), `GhostTileRing` (pinned-host + device-mapped, counter sector, payload region), `n_frames_written` accessor. |
| `crates/prism-nhs/src/ghost_telemetry.rs` | 1–871 (head only) | `PinnedTelemetryRing<T>` triple-buffer; non-blocking telemetry stream creation (`create_non_blocking_telemetry_stream`); `is_pinned_host` attestation; `log_f1_switch_events`. (Legacy ghost telemetry path; complementary to ghost_tile.rs.) |
| `crates/prism-nhs/src/captured_pipeline.rs` | 295–447, 800–946, 3151–3300 | `tier8_diag_post_gasp_sync_gate`, `post_t7_sync` gate, `ZstrCaptureParams`, `PipelineConfig`. |
| `crates/prism-nhs/src/persistent_engine.rs` | 45–110, 1540–1880 | `TIER8_DEFERRED_DRAIN_COUNT` global atomic; `Tier8DeferredDrainContext` population at `capture_autonomous_template`; capture-window deferred-drain ledger. |
| `crates/prism-nhs/src/fused_engine.rs` | 75–110, 7140–7300 | `Tier8DeferredDrainContext` (struct), thread-local context store, post-T7 deferred-drain reporter. |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 320–340, 4940–5800, 6080–6260, 8568–8640, 9094–9106, 9170–10063 | Args (`ghost_telemetry_io_uring`), per-stream `v2_zstr_ring`/`v2_zstr_consumer`/`v2_ghost_ring`, ZSTR ring alloc, Reaper spawn, Reaper join + `ZstrStats` log, ghost-tile teardown writer (legacy fs::write fallback), V2 ignition summary writer (Phase 8b). |
| `crates/prism-nhs/build.rs` | 1–80 | No `liburing` linkage — io_uring linkage is provided exclusively by the `io-uring` crate (Cargo.toml line 60: `io-uring = "0.5.13"`). |
| `crates/prism-nhs/Cargo.toml` | 50–66 | `libc = "0.2"` for pwrite/fdatasync/fsync/O_DIRECT, `io-uring = "0.5.13"` for SQE submission. |
| `docs/TIER8_GRAPH_TOPOLOGY.md` | 99–107 | Deferred-drain caveat (intentionally visible). |
| `docs/EXECUTION_POLICY.md` | 23, 33, 44, 73 | Layer-1 (Rust) ownership of orchestration + run-end emission; Python is offline-only. |

---

## 2. Functions / structs / types found (with file:line citations)

### 2.1 ZSTR ring + consumer (Channel A — O_DIRECT positions/forces)

| Name | File:line | What it owns |
|---|---|---|
| `ZstrFrameHeader` (struct, repr(C, align(4096)), 4096 B) | `zstr.rs:67-114` | `frame_idx, dt, adjudication_code, completion_fence, n_atoms, gear_id, force_norm, external_work, potential_energy, _padding`. Pinned to 4096 B by `static_assert`s at `zstr.rs:357-383`. |
| `ZstrRing` (struct) | `zstr.rs:130-141` | `raw_pinned_ptr, pinned_ptr, frame_size, n_atoms, alignment_ok`. `N_SLOTS = 5` per `zstr.rs:150`. Allocated via `cuMemHostAlloc(PORTABLE | DEVICEMAP)` (`zstr.rs:195-208`). G21 alignment check at `zstr.rs:215, 233-246`. |
| `ZstrRing::allocate(n_atoms)` | `zstr.rs:159-255` | Computes `header_bytes (4096) + pos_bytes_aligned (16-pad) + force_bytes`, rounds up to 4096-multiple, allocates `frame_size * 5 + 4095`, returns `pinned_ptr` aligned to 4096. Logs G21 PASS/FAIL line at `zstr.rs:220-246`. |
| `ZstrStats` (struct) | `zstr.rs:389-393` | `frames_written, frames_dropped, bytes_written` returned by Reaper join. |
| `spawn_zstr_consumer(ring, output_path, stop, ghost_ring, ghost_output_path)` | `zstr.rs:443-856` | Spawn entry point. Builder name `zstr-consumer`. Returns `JoinHandle<ZstrStats>`. |
| Consumer loop body | `zstr.rs:564-768` | Main spin loop: `slot = frame_idx % N_SLOTS`, spin on `completion_fence == 1` with `SPIN_LIMIT = 3_000_000` (`zstr.rs:562`), io_uring `IORING_OP_WRITE` push (`zstr.rs:600-619`), `submit_and_wait(1)` (`zstr.rs:621-646`), G29 finite-check on `force_norm` (`zstr.rs:654-667`), in-place fence reset with `Release` ordering (`zstr.rs:703-713`), Channel-B drain block (`zstr.rs:715-765`). |
| Channel-A open | `zstr.rs:466-479` | `OpenOptions::write/create/truncate + custom_flags(libc::O_DIRECT)`. |
| Channel-B open | `zstr.rs:490-519` | `custom_flags(libc::O_DIRECT | libc::O_DSYNC)`; on failure logs warn and disables Channel B (`ghost_state = None`). |
| `IoUring::new(32)` | `zstr.rs:535-546` | 32 SQE entries (5 inflight + 27 headroom). No SQPOLL — synchronous submit_and_wait per frame to preserve slot recycle ordering. |
| Channel-A teardown | `zstr.rs:822-830` | `libc::fsync(fd)` after loop exit. Logs `[ZSTR teardown] ✓ Channel-A fsync(...)` on success. |
| Channel-B teardown | `zstr.rs:802-803, 831-842` | `libc::fdatasync(ghost_fd)` mid-drain; final `libc::fsync(ghost_fd)` post-drain. |
| Reaper terminal log | `zstr.rs:845-851` | `[ZSTR consumer] exiting. frames_written={} dropped={} bytes={} MB path={:?}` |

### 2.2 Channel B — Ghost telemetry ring (mapped pinned, kernel writes through DMA)

| Name | File:line | What it owns |
|---|---|---|
| `GhostTileFrame` (struct, repr(C, align(4096)), 4096 B) | `ghost_tile.rs:80-162` | `frame_idx, site_id, chain_id, adjudication_code, telemetry_flags, kl_divergence, power_spectrum[24], thermo_flux[2], causal_lead_residue, _reserved_payload[32], _slack[3840]`. `static_assert`s at `ghost_tile.rs:198-213`. |
| `GhostTileRing` (struct) | `ghost_tile.rs:237-249` | `host_base, device_base, total_bytes, max_records`. `COUNTER_SECTOR_BYTES = 4096` at `ghost_tile.rs:260`. |
| `GhostTileRing::allocate(max_records)` | `ghost_tile.rs:265-310` | `cuMemHostAlloc(PORTABLE | DEVICEMAP) + cuMemHostGetDevicePointer_v2`; first 4096 B is the counter sector (`n_frames_written` u32 at offset 0); records start at offset 4096. |
| `GhostTileRing::n_frames_written()` | `ghost_tile.rs:316-320` | `read_volatile` on the GPU-incremented counter. |
| `GhostTileRing::payload_bytes()` | `ghost_tile.rs:325-334` | Live-record byte slice (clamped to `min(n_frames_written, max_records) * 4096`). |

### 2.3 Legacy ghost telemetry (3-slot DMA ring, parallel to above)

| Name | File:line | What it owns |
|---|---|---|
| `PinnedTelemetryRing<T>` (generic) | `ghost_telemetry.rs:95-99` | `base, elems_per_slot`. `N_SLOTS = 3`. |
| `create_non_blocking_telemetry_stream()` | `ghost_telemetry.rs:242-257` | `cuStreamCreate(CU_STREAM_NON_BLOCKING)`. |
| `is_pinned_host(host_ptr)` | `ghost_telemetry.rs:279-298` | `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_MEMORY_TYPE)` — used as **DMA-eligibility attestation** (mapped: `f2_ring_status.json::pinned_attest`). |
| `schedule_async_tile_copy(...)` | `ghost_telemetry.rs:321-349` | `cuMemcpyDtoHAsync_v2` to write-slot. Returns `Err` for `CUDA_ERROR_INVALID_VALUE` overflow / null source. |
| `log_f1_switch_events(...)` | `ghost_telemetry.rs:536-566` | Scans the safe-to-read slot for `adjudication_code == 1` events. |

### 2.4 Capture-window deferred drain (TIER 8)

| Name | File:line | What it owns |
|---|---|---|
| `Tier8DeferredDrainContext` | `fused_engine.rs:84-92` | `stream_slot, raw_stream, call_site, drain_count, rc, cuda_name, cuda_string`. Thread-local store at `fused_engine.rs:95-110`. |
| `TIER8_DEFERRED_DRAIN_COUNT` (global atomic) | `persistent_engine.rs:54-55` | `AtomicUsize::new(0)`. Incremented at `persistent_engine.rs:1581-1605` whenever `ctx.check_err()` drains a deferred state during pre-capture. |
| `tier8_set_deferred_drain_context` / `tier8_get_deferred_drain_context` | `fused_engine.rs:101-110` | Thread-local set/get of the most-recent `Tier8DeferredDrainContext`. Consumed at `fused_engine.rs:7157-7300` (post-T7 reporter). |
| `tier8_diag_post_gasp_sync_gate` | `captured_pipeline.rs:295-447` | Raw-CUDA 5-step gate (ctx.check_err → cuCtxGetCurrent → cuCtxSetCurrent → cuStreamSynchronize → ctx.check_err). Drains-and-reports posture is the same as `capture_autonomous_template`. |

### 2.5 Run-end finalize / orchestrator (where the emit lives)

| Name | File:line | Why it matters |
|---|---|---|
| Per-stream Reaper join | `nhs_rt_full.rs:8624-8637` | Sets `v2_zstr_stop`, joins handle, logs `frames_written/dropped/bytes`. **First place where `ZstrStats` is materialized on the host.** |
| Channel-B legacy teardown writer | `nhs_rt_full.rs:8568-8612` | Writes `{stem}_streamNN_ghost_tiles.bin` from `ring.payload_bytes()` via `std::fs::write` when the Reaper did **not** drive Channel B (i.e. `--ghost-telemetry-io-uring=false`). |
| V2 trajectory writer join | `nhs_rt_full.rs:9094-9105` | Drops `v2_snap_tx`; joins `v2_writer_handle`; `v2_n_frames` returned. |
| V2 ignition summary writer | `nhs_rt_full.rs:9942-10048` | **Recommended emit site for the three sidecars** (Phase 8b sibling). Already computes `lineage_integrity_hash` over `binding_sites.json`, `kcc_visualization.json`, and the head/tail/size fingerprint of `ghost_tiles.bin`. |
| `args.ghost_telemetry_io_uring` | `nhs_rt_full.rs:329-337` | Selector that routes Channel B through io_uring (default `true`) vs. legacy `std::fs::write` at teardown. The sidecars must record which path was active. |
| `phase3_run_id: String` | `nhs_rt_full.rs:3959-3963` | Reused as the `run_id` field of `f2_artifact_completeness.json`. |
| `output_base: PathBuf` | `nhs_rt_full.rs:10736` (`= args.output.join(&structure_name)`) | DAG / sidecar file paths derive from this. |

---

## 3. Existing reusable code

| Capability | Where it lives | Reuse strategy |
|---|---|---|
| FNV-1a 64-bit hashing | `nhs_rt_full.rs:9948-9957` | Sidecars reuse the **identical** local helper to hash artifact bytes. Same algorithm class as `lineage_integrity_hash`. (Operator already accepted FNV-1a as the audit-gate hash.) |
| File hashing pattern | `nhs_rt_full.rs:9959-9963` | `fs::read` → `fnv1a64`. For files > 64 MiB the head/tail/size fingerprint at `9972-10005` is reused. |
| Pretty-write + warn-on-failure | `nhs_rt_full.rs:9938-9939, 10043-10047` | Exact idiom for emitting `f2_*.json` files. |
| `ZstrStats` (return tuple) | `zstr.rs:389-393, 845-853` | Already carries `frames_written / frames_dropped / bytes_written`. **No new accounting needed for these three fields.** |
| `GhostTileRing::n_frames_written()` | `ghost_tile.rs:316-320` | Provides `frames_consumed` for ghost ring. |
| `GhostTileRing::max_records` | `ghost_tile.rs:248` | Provides `capacity_frames` for ghost ring. |
| `is_pinned_host(host_ptr)` | `ghost_telemetry.rs:279-298` | Authoritative attestation that a ring's `host_base` is page-locked. Sidecar consumes this for `pinned_attest: bool`. |
| `serde_json::to_string_pretty` writers | `nhs_rt_full.rs:9938, 10043, 15504` | Same idiom for sidecars. |
| Stream slot map | `persistent_engine.rs:58-66` (`tier8_capture_stream_slot`) | Maps `raw_stream: usize` → `stream_slot: u32`. Sidecar reuses this when populating `stream_id` on per-ring records. |
| `chrono::Utc::now()` | `nhs_rt_full.rs:3962` | Sidecar `created_utc` timestamp. |
| `phase3_run_id` | `nhs_rt_full.rs:3959-3963` | Sidecar `run_id`. |

**No new crate dependencies are needed.** `serde`, `serde_json`, `anyhow`,
`log`, `chrono` are already in `prism-nhs` Cargo.toml. `libc` is already
present for `O_DIRECT`/`fsync`.

---

## 4. Existing unsafe / problematic surfaces

| Issue | File:line | Conflict severity | Mitigation |
|---|---|---|---|
| **`ZstrStats` is reduced once per stream and logged, then dropped** — there is no global aggregate. | `nhs_rt_full.rs:8624-8637` | **Medium.** Must be captured on the orchestrator side **before** the per-stream `cfg(v2_ignition)` block exits. | Per-stream Reaper join populates a new `Vec<ZstrRingStatus>` shared across the multi-stream rayon scope (use `Arc<Mutex<Vec<...>>>` or extend the per-stream return tuple at `nhs_rt_full.rs:9151`). The minimal-touch path is the **return-tuple extension** — no new Mutex. |
| **`ZstrStats` does not currently carry `consumer_exit_status`** — Reaper exits via four separate paths (G21 fail, file open fail, io_uring init fail, normal stop) but only one logs. | `zstr.rs:457-464, 473-478, 535-545, 845-851` | **Medium.** Sidecar field `consumer_exit_status` cannot be populated without an extension. | **Extend `ZstrStats`** with `exit_status: ZstrExitStatus { Aborted_G21, FileOpenFail, UringInitFail, NormalStop }` enum. Localized change in `zstr.rs`; no cross-cutting impact. |
| **Frame-write counters do NOT separately track `frames_overflow` (SQ-full drops) vs `frames_dropped` (fence-timeout drops).** Both currently increment `frames_dropped` at `zstr.rs:610-619` and `zstr.rs:585-591`. | `zstr.rs:584-591, 610-619` | **Medium.** The audit cannot distinguish SQ saturation from fence stalls. | **Extend `ZstrStats`** with `frames_overflow: u64` (incremented at `zstr.rs:617`), keep `frames_dropped` for fence timeouts only. |
| **No `backpressure_events` counter.** When `submit_and_wait` blocks waiting for a CQE, the producer (GPU pipeline) cannot stall — but the consumer thread can fall behind. The current code does not record backpressure. | `zstr.rs:621-646` | **Low.** Backpressure is observable indirectly through `frames_overflow` and `bytes_written`/`(frames_consumed * frame_size)` divergence. | New counter `backpressure_events: u64` increments when `IoUring` SQ has zero free slots OR when `submit_and_wait` returns Err. **Optional in v1 of the sidecar** (mark field as `null` if unwired). |
| **`alignment_offset` is not stored on `ZstrRing`** — only logged at `zstr.rs:213, 220-232`. | `zstr.rs:130-141` | **Low.** The G21 log line carries the value; sidecar can recompute it from `(pinned_ptr - raw_pinned_ptr)` since both pointers are stored. | Sidecar emitter computes `alignment_offset = (ring.pinned_ptr as usize) - (ring.raw_pinned_ptr as usize)` at run-end. **No struct change needed.** |
| **Ghost-tile counter is GPU-mapped — no host-side `frames_dropped` exists** (kernel skips writes silently when `n_frames_written >= max_records` per `ghost_tile.rs:354-368`). | `ghost_tile.rs:354-368` | **Medium.** Without a kernel-side overflow flag, the sidecar must report `frames_dropped: null` (unknown, exposed) for the ghost ring rather than zero. | Sidecar reports `frames_dropped: null, overflow_inferred: (n_frames_written == max_records)` for the ghost ring; **does not** silently fabricate zero. (Fail loud, per agent-9 forbidden list.) |
| **Two-channel B paths.** When `--ghost-telemetry-io-uring=true` (default) Channel B writes via the Reaper. When `false`, Channel B writes via `std::fs::write` at teardown (`nhs_rt_full.rs:8589-8611`). | `nhs_rt_full.rs:8589-8611, 6088-6110, 6227-6253` | **High.** The artifact path the same in both paths but the open-flags / commit semantics differ. | Sidecar **MUST** record the active path (`io_uring_o_direct_dsync` vs `legacy_pwrite`) so the consumer knows which `close_status` / `truncation_guarantee` semantics apply. |
| **Deferred drain count is a process-global atomic** — TIER 8 drains accumulate across streams. | `persistent_engine.rs:54-55, 1581` | None for IO. | Sidecar reads `TIER8_DEFERRED_DRAIN_COUNT.load(Relaxed)` at run-end; surfaced as a top-level `tier8_deferred_drain_count` in `f2_artifact_completeness.json::reasons[]` when ≥ 1. **Does not paper over the caveat — it is loaded into the report.** |
| **No global `fence_was_ready` counter.** The string `fence_was_ready` only appears in the per-1000-frame INFO log at `zstr.rs:687-696`. | `zstr.rs:687-696` | **Low.** Per-frame fence readiness is implicit in `frames_written > 0`. The sidecar `fence_was_ready` field reduces to `(stats.frames_written > 0) && (stats.frames_dropped < frames_produced * 0.01)`. | Compute at emit time. **Does not require runtime counter.** |

**No `unsafe { ... }` blocks are touched by this plan.** The three
sidecars are pure Rust + serde + filesystem-write; the only state read
from the runtime is the Reaper's `ZstrStats`, the ring structs, and
host-only globals.

---

## 5. Exact proposed implementation plan

### 5.1 Schema location

Three sidecar files written into `args.output` (same directory as
`v2_ignition_summary.json`, `binding_sites.json`,
`*_streamNN_ghost_tiles.bin`):

```
{output_dir}/{stem}.f2_ring_status.json
{output_dir}/{stem}.f2_write_commit_log.json
{output_dir}/{stem}.f2_artifact_completeness.json
```

Naming convention: `output_base.with_extension("f2_ring_status.json")` —
mirrors `output_base.with_extension("v2_ignition_summary.json")` at
`nhs_rt_full.rs:10042`.

### 5.2 Schema 1 — `f2_ring_status.json`

Per-ring snapshot at run-end, one record per (stream × channel). Channel A
is the ZSTR ring; Channel B is the Ghost-tile ring; Channel C is the
optional legacy `PinnedTelemetryRing<ContactShellTile>` (only present if
the Reaper-less path is selected — emit `kind: "ghost_telemetry_legacy"`
only when wired).

```jsonc
{
  "schema_version": 1,
  "run_id": "<phase3_run_id>",
  "rings": [
    {
      "ring_id":             "zstr_channel_a_stream00",      // String (deterministic)
      "kind":                "zstr_o_direct",                // String
      "stream_id":           0,                              // u32 (None for non-CUDA rings → omit field)
      "raw_alloc_bytes":     1310720,                        // u64
      "usable_bytes":        1306624,                        // u64 (frame_size * N_SLOTS)
      "alignment_offset":    0,                              // u64 ((pinned_ptr - raw_pinned_ptr))
      "alignment_ok":        true,                           // bool (G21)
      "capacity_frames":     5,                              // u64 (ZstrRing::N_SLOTS)
      "head_seq":            17234,                          // u64 (frames_written - this stream)
      "tail_seq":            17234,                          // u64 (frame_idx at exit)
      "frames_produced":     17234,                          // u64 (head_seq + frames_dropped + frames_overflow)
      "frames_consumed":     17234,                          // u64 (== ZstrStats.frames_written)
      "frames_dropped":      0,                              // u64 (fence timeouts)
      "frames_overflow":     0,                              // u64 (io_uring SQ-full drops; new field)
      "backpressure_events": 0,                              // u64 (optional; null if unwired)
      "consumer_spawned":    true,                           // bool
      "consumer_exit_status":"normal_stop",                  // String enum
      "fence_was_ready":     true,                           // bool (derived)
      "pinned_attest":       true,                           // bool (cuPointerGetAttribute → CU_MEMORYTYPE_HOST)
      "open_flags":          "O_DIRECT|O_WRONLY|O_CREAT",    // String
      "iouring_sq_depth":    32                              // u32 (zstr.rs:535 IoUring::new(32))
    },
    {
      "ring_id":             "ghost_channel_b_stream00",
      "kind":                "ghost_telemetry",
      "stream_id":           0,
      "raw_alloc_bytes":     134221824,                      // ghost_max_records * 4096 + 4096 counter sector
      "usable_bytes":        134217728,                      // max_records * 4096
      "alignment_offset":    0,                              // counter sector at offset 0; records start at 4096
      "alignment_ok":        true,                           // implicit (allocation is page-aligned)
      "capacity_frames":     32768,                          // u64 (max_records)
      "head_seq":            14802,                          // u64 (n_frames_written)
      "tail_seq":            14802,                          // u64 (last_ghost_slot_emitted at Reaper exit)
      "frames_produced":     14802,                          // u64 (n_frames_written)
      "frames_consumed":     14802,                          // u64 (records flushed to disk; mode-dependent)
      "frames_dropped":      null,                           // null = unknown (kernel-side overflow not exposed)
      "frames_overflow":     0,                              // u64 (only nonzero if Reaper SQ full at ghost slot)
      "backpressure_events": null,
      "consumer_spawned":    true,                           // shared with Channel A Reaper
      "consumer_exit_status":"normal_stop",
      "fence_was_ready":     null,                           // ghost ring has no per-slot fence (counter-based safety lag)
      "pinned_attest":       true,
      "open_flags":          "O_DIRECT|O_DSYNC|O_WRONLY|O_CREAT",
      "iouring_sq_depth":    32,
      "channel_b_path":      "io_uring_o_direct_dsync"        // String enum: "io_uring_o_direct_dsync" | "legacy_pwrite"
    }
  ]
}
```

### 5.3 Schema 2 — `f2_write_commit_log.json`

Per-artifact write-commit ledger.

```jsonc
{
  "schema_version": 1,
  "run_id": "<phase3_run_id>",
  "commits": [
    {
      "artifact_path":          "/mnt/storage/.../prism_zstr_<pid>_0.bin",
      "ring_id":                "zstr_channel_a_stream00",
      "bytes_written":          1763631104,
      "frames_written":         17234,
      "first_seq":              0,
      "last_seq":               17233,
      "open_flags":             "O_DIRECT|O_WRONLY|O_CREAT",
      "close_status":           "fsync_ok",                   // String enum: "fsync_ok" | "no_fsync" | "error:<msg>"
      "truncation_guarantee":   "padded_to_block",            // every record is `frame_size`-aligned (4096 multiple); file size = N * frame_size
      "hash":                   "fnv1a64:<head4k+tail4k+size>"
    },
    {
      "artifact_path":          "/mnt/storage/.../<stem>_stream00_ghost_tiles.bin",
      "ring_id":                "ghost_channel_b_stream00",
      "bytes_written":          60628992,
      "frames_written":         14802,
      "first_seq":              0,
      "last_seq":               14801,
      "open_flags":             "O_DIRECT|O_DSYNC|O_WRONLY|O_CREAT",
      "close_status":           "fsync_ok",
      "truncation_guarantee":   "exact",                      // 4096 B per record × N records = exact file size
      "hash":                   "fnv1a64:<head4k+tail4k+size>"
    },
    {
      "artifact_path":          "/mnt/storage/.../<stem>.binding_sites.json",
      "ring_id":                null,                         // not a ring artifact
      "bytes_written":          <fs::metadata(...).len()>,
      "frames_written":         null,
      "first_seq":              null,
      "last_seq":               null,
      "open_flags":             "O_WRONLY|O_CREAT|O_TRUNC",   // std::fs::write semantics
      "close_status":           "no_fsync",
      "truncation_guarantee":   "exact",
      "hash":                   "fnv1a64:<full bytes>"
    }
  ]
}
```

`truncation_guarantee` semantics:

- `"exact"` — file size equals `bytes_written` (no padding); reader can compute `frames_written = size / record_size`.
- `"padded_to_block"` — `O_DIRECT` writes the slot's full `frame_size` (4096-padded); reader iterates by `frame_size` and trusts headers.
- `"unknown"` — non-deterministic (e.g. fsync failed mid-write).

### 5.4 Schema 3 — `f2_artifact_completeness.json`

Run-end completeness verdict. **MUST** cross-reference Agent 1's
`transform_dag.json::artifacts[]`.

```jsonc
{
  "schema_version": 1,
  "run_id":         "<phase3_run_id>",
  "expected_artifacts": [
    "phasor_kcc_state.json",
    "prism_therm_telemetry.json",
    "kcc_visualization.json",
    "spatial_grid_state.json",
    "t7_calibration.json",
    "aromatic_centroids_map.json",
    "<stem>_v2_final.pdb",
    "<stem>.binding_sites.json",
    "<stem>.v2_ignition_summary.json",
    "<stem>_stream00_ghost_tiles.bin",
    "prism_zstr_<pid>_0.bin",
    "transform_dag.json"                             // Agent 1 deliverable
  ],
  "emitted_artifacts": [
    "phasor_kcc_state.json",
    ...
  ],
  "missing":                  [],
  "partial":                  [
    {
      "path":   "<stem>_stream03_ghost_tiles.bin",
      "reason": "stream 3 monolithic splice failed (V2-INSTANTIATE-COMPLETE flagged); pipeline disabled; no ring allocated"
    }
  ],
  "fence_pass":               true,                  // bool: all ZSTR rings reported fence_was_ready == true
  "alignment_pass":           true,                  // bool: G21 alignment_ok == true on every allocated ZstrRing
  "drain_pass":               true,                  // bool: Σ frames_dropped == 0 across all rings
  "tier8_deferred_drain_count": 2,                   // u64: TIER8_DEFERRED_DRAIN_COUNT.load() at run-end
  "tier8_deferred_drain_caveat": true,               // bool: count > 0 → caveat is live (per TIER8_GRAPH_TOPOLOGY.md:99-107)
  "overall_status":           "complete",            // String enum: "complete" | "partial" | "failed"
  "reasons": [
    "tier8_deferred_drain_count=2 — see docs/TIER8_GRAPH_TOPOLOGY.md:99-107"
  ]
}
```

`overall_status` rule:

- `"complete"` — `missing == [] && partial == [] && fence_pass && alignment_pass && drain_pass`.
- `"partial"` — `missing` empty but `partial` non-empty, OR any of {fence, alignment, drain} failed but rings were allocated.
- `"failed"` — any artifact in `missing[]`.

The `tier8_deferred_drain_caveat` field is **always emitted** (true or
false) — the caveat is exposed, not normalized away.

### 5.5 Population mapping

| Sidecar field | Source struct | Source file:line | New accounting? |
|---|---|---|---|
| `f2_ring_status.json::rings[zstr].raw_alloc_bytes` | computed: `frame_size * N_SLOTS + 4095` | `zstr.rs:184-185` | **Existing**; reuse `frame_size * 5 + 4095`. |
| `f2_ring_status.json::rings[zstr].usable_bytes` | computed: `frame_size * 5` | `zstr.rs:184` | **Existing**. |
| `f2_ring_status.json::rings[zstr].alignment_offset` | computed: `pinned_ptr - raw_pinned_ptr` | `zstr.rs:210-214` | **Existing**; computed at emit time. |
| `f2_ring_status.json::rings[zstr].alignment_ok` | `ZstrRing::alignment_ok` | `zstr.rs:140, 215` | **Existing**. |
| `f2_ring_status.json::rings[zstr].capacity_frames` | `ZstrRing::N_SLOTS` | `zstr.rs:150` | **Existing**. |
| `f2_ring_status.json::rings[zstr].frames_consumed` | `ZstrStats::frames_written` | `zstr.rs:390, 853` | **Existing**. |
| `f2_ring_status.json::rings[zstr].frames_dropped` | `ZstrStats::frames_dropped` | `zstr.rs:391, 853` | **Existing** (fence-timeout drops only after split). |
| `f2_ring_status.json::rings[zstr].frames_overflow` | `ZstrStats::frames_overflow` | `zstr.rs:617` | **NEW** — split SQ-full from fence-timeout. |
| `f2_ring_status.json::rings[zstr].consumer_exit_status` | `ZstrStats::exit_status` (new enum field) | `zstr.rs:463, 477, 544, 854` | **NEW** — extend `ZstrStats`. |
| `f2_ring_status.json::rings[zstr].pinned_attest` | `is_pinned_host(ring.pinned_ptr)` | `ghost_telemetry.rs:279-298` | **Existing** (live attestation at emit time; reused on the ZSTR ring). |
| `f2_ring_status.json::rings[zstr].iouring_sq_depth` | constant `32` | `zstr.rs:535` | **Existing** (compile-time constant). |
| `f2_ring_status.json::rings[ghost].head_seq` | `GhostTileRing::n_frames_written()` | `ghost_tile.rs:316-320` | **Existing**. |
| `f2_ring_status.json::rings[ghost].capacity_frames` | `GhostTileRing::max_records` | `ghost_tile.rs:248` | **Existing**. |
| `f2_ring_status.json::rings[ghost].channel_b_path` | `args.ghost_telemetry_io_uring` | `nhs_rt_full.rs:336` | **Existing**. |
| `f2_write_commit_log.json::commits[zstr_a].bytes_written` | `ZstrStats::bytes_written` | `zstr.rs:392, 853` | **Existing**. |
| `f2_write_commit_log.json::commits[zstr_a].close_status` | new field reduced from teardown logs | `zstr.rs:822-830` | **NEW** — extend `ZstrStats` with `fsync_status: Result<(), i32>`. |
| `f2_write_commit_log.json::commits[ghost_b].bytes_written` | `ZstrStats::ghost_bytes_written` (currently a thread-local; not in `ZstrStats`) | `zstr.rs:521-523, 805-810` | **NEW** — extend `ZstrStats` with `ghost_records_written: u64, ghost_bytes_written: u64, ghost_fsync_status: Option<Result<(),i32>>`. |
| `f2_artifact_completeness.json::tier8_deferred_drain_count` | `TIER8_DEFERRED_DRAIN_COUNT.load(Relaxed)` | `persistent_engine.rs:54-55` | **Existing** — read at emit time. |
| `f2_artifact_completeness.json::expected_artifacts[]` | static list aligned with Agent 1 DAG | `nhs_rt_full.rs:10013-10022` | **Existing template** (current legacy list); reused with two additions (`ghost_tiles.bin`, `prism_zstr_*.bin`, `transform_dag.json`). |
| `f2_artifact_completeness.json::emitted_artifacts[]` | filesystem scan at emit time | n/a | **NEW** — `std::fs::metadata()` on each expected path. |

### 5.6 Where the emit call lives

**Recommended site: `nhs_rt_full.rs:10048` (immediately after the
`v2_ignition_summary.json` write).** Reasons:

1. The Reapers have already joined (per-stream join at `nhs_rt_full.rs:8624-8637`);
   `ZstrStats` are fully materialized by the time control reaches Phase 8b.
2. Channel-B legacy fallback writes have completed (`nhs_rt_full.rs:8589-8611`).
3. `output_base`, `phase3_run_id`, `args` are all in scope.
4. Sibling to Agent 1's DAG emit (also recommended for Phase 8b per
   [docs/DAG_FOUNDATION_IMPLEMENTATION_PLAN.md](DAG_FOUNDATION_IMPLEMENTATION_PLAN.md)).
5. `lineage_integrity_hash` (`nhs_rt_full.rs:10025-10030`) is the
   already-canonical example of "filesystem scan + fingerprint"
   pattern; the F2 emitter is structurally identical.

**The emit call is on the orchestrator (main thread), not the Reaper
thread.** The Reaper has already exited by the time of emit. This is
required because:

- The orchestrator is the only place that sees **all** per-stream
  `ZstrStats` simultaneously (one Reaper per stream).
- Filesystem stat()ing the ghost-tile bin / zstr bin requires those
  files to be `fsync`-committed, which only happens at Reaper teardown
  (`zstr.rs:822-842`).
- `TIER8_DEFERRED_DRAIN_COUNT` is a process-global atomic; reading it
  from the orchestrator is correct.

A second emit site at `nhs_rt_full.rs:16584-16599` (multi-stream non-V2
exit) is **out of scope for v1** — the V2 ignition path is the production
target. The non-V2 fallback emits a degenerate sidecar (no rings, no
commits) only if explicitly required by a follow-up.

### 5.7 Cross-reference with Agent 1 (DAG)

Agent 1's `transform_dag.json` (per
[DAG_FOUNDATION_IMPLEMENTATION_PLAN.md](DAG_FOUNDATION_IMPLEMENTATION_PLAN.md))
emits `DagArtifactRef { path, hash, size_bytes }` for every artifact. The
F2 sidecars and the DAG MUST agree on artifact paths. The contract:

- F2 emits `f2_artifact_completeness.json::expected_artifacts[]` and
  `emitted_artifacts[]` as **plain `String` paths** (relative to
  `args.output`, like the legacy `artifacts: [...]` list at
  `nhs_rt_full.rs:10013-10022`).
- Agent 1's DAG emits `DagArtifactRef::path: PathBuf` for the same set.
- A trivial `path-set equality` test in the validator confirms the two
  agree.

**No central registry is needed.** Both sidecars derive from the same
constant artifact list (extended versions of the
`nhs_rt_full.rs:10013-10022` block). The emit order is:

1. All artifacts that are part of `lineage_integrity_hash` already exist on disk.
2. F2 emits.
3. DAG emits.
4. (Future: completeness validator script reads both, asserts equality.)

If Agent 0 (TIER 8.1 / ZSTR alignment) lands a fix that changes
`alignment_offset` semantics or introduces a new G21 sub-gate, the F2
sidecar `rings[].alignment_offset` and `alignment_ok` fields are the
**single source of truth** for that gate's run-end status.

### 5.8 Exposing the deferred-drain caveat (TIER 8 §99-107)

Per the agent-9 forbidden list ("No removing deferred-drain reporting;
expose it through these sidecars"), the plan exposes the caveat through:

1. `f2_artifact_completeness.json::tier8_deferred_drain_count` — always
   emitted (could be 0).
2. `f2_artifact_completeness.json::tier8_deferred_drain_caveat` — bool
   flag derived from `count > 0`.
3. `f2_artifact_completeness.json::reasons[]` — appends the human-readable
   string `"tier8_deferred_drain_count=N — see docs/TIER8_GRAPH_TOPOLOGY.md:99-107"`
   when count > 0.
4. `overall_status` is **NOT** demoted to `"partial"` solely on
   `tier8_deferred_drain_count > 0`; the all-8 gate has explicitly frozen
   "drains may occur and remain non-fatal" (`TIER8_GRAPH_TOPOLOGY.md:107`).
   The drain count is reported alongside the per-stream
   `Tier8DeferredDrainContext` evidence in the structured log, not used
   to fail the run.

---

## 6. Minimal touch list

| File | Lines | What changes |
|---|---|---|
| `crates/prism-nhs/src/zstr.rs` | 389-393 | **Extend `ZstrStats`** with `frames_overflow: u64`, `exit_status: ZstrExitStatus`, `ghost_records_written: u64`, `ghost_bytes_written: u64`, `fsync_status_a: i32` (errno; 0 = OK), `fsync_status_b: i32`. Add `ZstrExitStatus` enum: `NormalStop, AbortedG21Alignment, FileOpenFailure, IoUringInitFailure, ConsumerPanic`. |
| `crates/prism-nhs/src/zstr.rs` | 463, 477, 544, 853 | Populate `exit_status` at each early return. |
| `crates/prism-nhs/src/zstr.rs` | 617, 588 | Increment `frames_overflow` (was `frames_dropped`) when SQ push fails; keep `frames_dropped` only for fence timeouts. |
| `crates/prism-nhs/src/zstr.rs` | 821-842 | Capture `fsync` return values into `fsync_status_a` / `fsync_status_b`. |
| `crates/prism-nhs/src/zstr.rs` | 521-523, 760-762, 798 | Assign `ghost_records_written` / `ghost_bytes_written` to the local-mut counters (already exist); fold them into the returned `ZstrStats` at `zstr.rs:853`. |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | ~9151 | **Extend the per-stream return tuple** at the rayon scope to include `Option<ZstrStats>` and `Option<f2_ring_snapshot::RingMetadata>` (raw_alloc_bytes, alignment_offset, alignment_ok, capacity_frames, kind, stream_id). |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 8624-8637 | After Reaper join, populate per-stream `Option<ZstrStats>` slot; also populate `RingMetadata` from `v2_zstr_ring` (read alignment_ok, raw_pinned_ptr, pinned_ptr, frame_size, n_atoms) and `v2_ghost_ring` (host_base, total_bytes, max_records, n_frames_written). |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | new (after 10048) | **Phase 8c: F2 evidence-plane sidecars.** Build the three JSON values, write via `std::fs::write` (warn-on-failure). Reuses `fnv1a64`, `phase3_run_id`, `output_base`, `args`, the per-stream `ZstrStats` collected via the extended return tuple, and `TIER8_DEFERRED_DRAIN_COUNT.load(Relaxed)` (import the static path or expose via a helper). |
| `crates/prism-nhs/src/persistent_engine.rs` | 54-55 | **No change to the static itself**, but expose a `pub fn tier8_deferred_drain_count() -> usize` helper at module top-level so `nhs_rt_full.rs` does not have to reach into a `pub(crate)` static. |
| `crates/prism-nhs/src/lib.rs` | re-export | Re-export `tier8_deferred_drain_count`, `ZstrExitStatus`, the new fields on `ZstrStats` if they need to be public (already `pub` on `ZstrStats`; `ZstrExitStatus` becomes `pub`). |

**No CUDA / kernel / build.rs changes.** No new crate dependencies. No
graph topology changes. No Python in the runtime.

---

## 7. New structs / functions (Rust)

```rust
// In crates/prism-nhs/src/zstr.rs (extend existing module)

/// Reaper exit posture. Surfaces in `f2_ring_status.json::consumer_exit_status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum ZstrExitStatus {
    /// Stop signal observed; loop exited cleanly.
    NormalStop,
    /// G21 alignment_ok was false at consumer entry; aborted before any write.
    AbortedG21Alignment,
    /// `O_DIRECT|O_WRONLY|O_CREAT` open failed.
    FileOpenFailure,
    /// `IoUring::new(32)` failed; consumer aborted.
    IoUringInitFailure,
    /// Reserved for `JoinHandle::join()` returning Err — orchestrator-side only.
    ConsumerPanic,
}

/// Reaper exit summary returned by spawn_zstr_consumer's join handle.
/// EXTENDED with channel-B and exit-status accounting for the F2 sidecars.
pub struct ZstrStats {
    pub frames_written:        u64,
    pub frames_dropped:        u64,    // fence-timeout drops ONLY (post-split)
    pub frames_overflow:       u64,    // io_uring SQ-full drops (NEW)
    pub bytes_written:         u64,
    pub ghost_records_written: u64,    // NEW (Channel B record count)
    pub ghost_bytes_written:   u64,    // NEW
    pub exit_status:           ZstrExitStatus, // NEW
    pub fsync_status_a:        i32,    // NEW; 0 = OK, else errno
    pub fsync_status_b:        i32,    // NEW; -1 if Channel B disabled, 0 = OK, else errno
}
```

```rust
// In crates/prism-nhs/src/persistent_engine.rs (new pub helper)

/// Read the process-global TIER 8 deferred-drain count. Surfaces in
/// `f2_artifact_completeness.json::tier8_deferred_drain_count`.
/// See docs/TIER8_GRAPH_TOPOLOGY.md:99-107 for the caveat.
pub fn tier8_deferred_drain_count() -> usize {
    TIER8_DEFERRED_DRAIN_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}
```

```rust
// In crates/prism-nhs/src/bin/nhs_rt_full.rs, new private module
// scoped to the V2 teardown block (Phase 8c).

mod f2_evidence {
    use serde::Serialize;
    use std::path::PathBuf;

    #[derive(Serialize)]
    pub struct RingStatus {
        pub ring_id:              String,
        pub kind:                 String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream_id:            Option<u32>,
        pub raw_alloc_bytes:      u64,
        pub usable_bytes:         u64,
        pub alignment_offset:     u64,
        pub alignment_ok:         bool,
        pub capacity_frames:      u64,
        pub head_seq:             u64,
        pub tail_seq:             u64,
        pub frames_produced:      u64,
        pub frames_consumed:      u64,
        pub frames_dropped:       Option<u64>,
        pub frames_overflow:      u64,
        pub backpressure_events:  Option<u64>,
        pub consumer_spawned:     bool,
        pub consumer_exit_status: String,
        pub fence_was_ready:      Option<bool>,
        pub pinned_attest:        bool,
        pub open_flags:           String,
        pub iouring_sq_depth:     u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub channel_b_path:       Option<String>,
    }

    #[derive(Serialize)]
    pub struct WriteCommit {
        pub artifact_path:        PathBuf,
        pub ring_id:              Option<String>,
        pub bytes_written:        u64,
        pub frames_written:       Option<u64>,
        pub first_seq:            Option<u64>,
        pub last_seq:             Option<u64>,
        pub open_flags:           String,
        pub close_status:         String,
        pub truncation_guarantee: String,
        pub hash:                 Option<String>,
    }

    #[derive(Serialize)]
    pub struct ArtifactCompleteness {
        pub schema_version:                  u32,
        pub run_id:                          String,
        pub expected_artifacts:              Vec<String>,
        pub emitted_artifacts:               Vec<String>,
        pub missing:                         Vec<String>,
        pub partial:                         Vec<PartialArtifact>,
        pub fence_pass:                      bool,
        pub alignment_pass:                  bool,
        pub drain_pass:                      bool,
        pub tier8_deferred_drain_count:      u64,
        pub tier8_deferred_drain_caveat:     bool,
        pub overall_status:                  String,
        pub reasons:                         Vec<String>,
    }

    #[derive(Serialize)]
    pub struct PartialArtifact {
        pub path:   String,
        pub reason: String,
    }
}

/// Phase 8c — emit the three F2 evidence-plane sidecars.
/// Called from `nhs_rt_full.rs` at run-end (after the v2_ignition_summary
/// emit at line 10048). Reuses `fnv1a64`, `output_base`, `phase3_run_id`,
/// `args`, and the per-stream `ZstrStats` collected via the extended
/// rayon return tuple.
fn emit_f2_evidence_plane(
    output_base:    &std::path::Path,
    run_id:         &str,
    args:           &Args,
    per_stream_stats: &[(usize, Option<prism_nhs::zstr::ZstrStats>, Option<f2_evidence::RingStatus>, Option<f2_evidence::RingStatus>)],
    ghost_artifact_paths: &[std::path::PathBuf],
    zstr_artifact_paths:  &[std::path::PathBuf],
    expected_artifacts:   &[String],
) {
    // Build f2_ring_status.json
    // Build f2_write_commit_log.json
    // Build f2_artifact_completeness.json
    // Three std::fs::write calls; warn-only on failure (matches Phase 8b posture).
}
```

---

## 8. Acceptance tests

| # | Test | Expected outcome |
|---|---|---|
| 1 | Run a healthy 8-stream V2 ignition (canonical command) on 4LPK; check that all three sidecars exist in `args.output`. | Three files present; valid JSON; `overall_status == "complete"`. |
| 2 | Same run; assert `f2_ring_status.json::rings[].alignment_ok == true` for all 8 ZSTR rings AND all 8 ghost rings. | Pass. |
| 3 | Same run; assert `Σ rings[].frames_consumed == Σ rings[].head_seq` (no in-flight frames at run-end). | Pass. |
| 4 | Same run; assert `f2_write_commit_log.json::commits[].close_status == "fsync_ok"` for every io_uring path. | Pass. |
| 5 | Same run; assert `f2_artifact_completeness.json::expected_artifacts == transform_dag.json::artifacts[].path` (set equality, ignoring order). | Pass — confirms Agent 1 / Agent 9 cross-reference. |
| 6 | Force a G21 alignment failure (test fixture `--g21-fault-inject`, future); assert `consumer_exit_status == "AbortedG21Alignment"`, `frames_consumed == 0`, `overall_status == "partial"`. | Pass. |
| 7 | Run with `--ghost-telemetry-io-uring=false`; assert `rings[ghost].channel_b_path == "legacy_pwrite"` and the `commits` entry for `*_ghost_tiles.bin` carries `close_status == "no_fsync"`. | Pass. |
| 8 | Run a campaign that triggers a deferred drain (TIER 8 §99-107); assert `tier8_deferred_drain_count >= 1`, `tier8_deferred_drain_caveat == true`, `reasons` contains the doc-line citation, `overall_status == "complete"` (drain alone does not fail the run). | Pass — caveat exposed, run still complete. |
| 9 | Pure-host unit test in `prism-nhs::zstr::tests`: drive `ZstrStats` through each `ZstrExitStatus` variant and assert serde-JSON output matches the schema. | Pass. |
| 10 | Validator script (`scripts/quarantine/f2_completeness_check.py` — Python, offline) loads all three sidecars, asserts schema, asserts cross-references. | Pass — offline only, never invoked from runtime. |

---

## 9. Failure modes (and how the sidecars expose them)

| Failure | Surface in sidecars |
|---|---|
| G21 4096-alignment fails on a stream's ZSTR ring | `f2_ring_status.json::rings[].alignment_ok == false`, `consumer_spawned == false`, `consumer_exit_status == "AbortedG21Alignment"`, `frames_consumed == 0`. `f2_artifact_completeness.json::alignment_pass == false`, `overall_status == "partial"`. |
| Channel-A `O_DIRECT` open fails (e.g. filesystem doesn't support O_DIRECT) | `consumer_exit_status == "FileOpenFailure"`, `frames_consumed == 0`, `bytes_written == 0`. `f2_write_commit_log.json::commits` entry for that path is **absent** (artifact not emitted) and `f2_artifact_completeness.json::missing[]` includes `prism_zstr_<pid>_<stream>.bin`. |
| `IoUring::new(32)` fails (kernel built without `CONFIG_IO_URING=y`) | `consumer_exit_status == "IoUringInitFailure"`. Same `missing[]` entry as above. |
| io_uring SQ saturation under burst | `frames_overflow > 0`. `frames_consumed < frames_produced`. `overall_status` may be `"partial"` if `frames_overflow / frames_produced > 0.01`. |
| Fence timeout (GPU stalled / abort-trap) | `frames_dropped > 0`. `overall_status == "partial"`. |
| G29 force-norm trap (NaN/Inf in steered force) | `frames_dropped > 0` (slot bytes ARE on disk; G29 logs the trap; the drop count reflects the broken slot). Operator reads the structured log line `[ZSTR G29 TRAP]` for triage. |
| Ghost ring overflow (kernel skipped writes — `n_frames_written >= max_records`) | `rings[ghost].head_seq == max_records`, `frames_dropped: null` (kernel-side overflow not exposed). `f2_artifact_completeness.json::partial[]` adds an entry citing the saturation. |
| `--ghost-telemetry-io-uring=false` (legacy fallback) | `rings[ghost].channel_b_path == "legacy_pwrite"`, `commits[ghost].close_status == "no_fsync"`. |
| TIER 8 deferred drain (capture-window) | `tier8_deferred_drain_count > 0`, `tier8_deferred_drain_caveat == true`. **Does not** fail the run. |
| Reaper thread panicked between stop-signal and exit | Orchestrator catches `JoinHandle::join() == Err(_)`; sets `consumer_exit_status == "ConsumerPanic"`, `consumer_spawned == true`, `frames_consumed = 0` (or last-known head_seq with a `panic_at_seq` reason in `partial[]`). |
| `fsync` returns non-zero | `fsync_status_a != 0` propagates to `commits[].close_status == "error:errno=<N>"`. `truncation_guarantee` demoted to `"unknown"`. |
| Two-channel race at teardown (both Reaper + legacy fs::write attempt to write the same ghost path) | Mutually exclusive at the orchestrator: legacy fallback only fires when `ghost_clone.is_none()` (`nhs_rt_full.rs:6088-6092, 6227-6231`). The sidecar emits only one `commits[]` entry per artifact — the active path. |
| Per-stream Reaper exits before all chunks complete (e.g. early stop signal) | `frames_consumed < frames_produced`, `frames_dropped > 0`. `overall_status == "partial"`. |

---

## 10. Rollback path

The plan is **purely additive** — three new JSON files, four new fields
on `ZstrStats`, one new pub helper, one new emit block. No existing
artifact paths change; no graph topology changes; no kernel changes.

If a regression is observed:

1. **Phase 1 rollback:** comment out the Phase 8c emit block at
   `nhs_rt_full.rs:10049+`. Engine continues to emit
   `v2_ignition_summary.json` with `lineage_integrity_hash` exactly as
   before; no F2 sidecars are written; no run-end behavior changes.
2. **Phase 2 rollback:** revert the `ZstrStats` struct extension. The
   runtime keeps the original three-field `ZstrStats`; the per-stream
   Reaper join log line at `nhs_rt_full.rs:8624-8637` remains
   structurally identical (only the format string mentions the
   newly-removed fields).
3. **Phase 3 rollback:** revert the per-stream return-tuple extension at
   `nhs_rt_full.rs:9151`. Restores the original `Vec<f32>`-only return.
4. **Tier-8 helper rollback:** revert the
   `pub fn tier8_deferred_drain_count` helper in `persistent_engine.rs`.
   The atomic itself is untouched.

Each rollback step is independently reversible and does not require a
rebuild of the CUDA kernels (PTX cache hits unchanged) — only Rust
recompilation of `prism-nhs` and `nhs_rt_full`.

---

## 11. Conflicts with sibling agents

| Agent | Territory | Conflict | Resolution |
|---|---|---|---|
| Agent 0 (TIER 8.1 / ZSTR alignment) | G21 4096-alignment fix in `zstr.rs:159-255` | None. F2 sidecar **reads** `alignment_ok` and `alignment_offset` from the post-fix ring; it does not modify the alignment logic. If Agent 0 promotes the G21 check from `bool` to a richer `G21Status { ok, offset, sub_gates }` enum, the sidecar tracks the new shape via a one-line update to `RingStatus::alignment_ok` (drop) + new `g21_subgates: HashMap<String, bool>` field. | F2 plan stays out of `zstr.rs:159-255`. |
| Agent 1 (Transform / Evidence DAG Foundation) | `transform_dag.json` emission at `nhs_rt_full.rs:~10048` | **Coordinated, not conflicting.** F2 emit lives at the same Phase 8c emit site. Both consume `phase3_run_id`, `output_base`, `args`. Both reuse `fnv1a64`. `f2_artifact_completeness.json::expected_artifacts[]` MUST equal `transform_dag.json::artifacts[].path` as sets. | **Deliberate cross-reference.** F2 emits **after** DAG (so the DAG file itself appears in F2's `emitted_artifacts[]`). A future validator script asserts the equality. |
| Captured-pipeline lane (existing) | `captured_pipeline.rs:295-447` post-gasp-sync gate | None. The gate's drain-and-report behavior populates `Tier8DeferredDrainContext` which the F2 sidecar reads through `tier8_deferred_drain_count()`. | F2 plan does not touch the gate. |
| Per-stream rayon handles | `nhs_rt_full.rs:9151-9156` | **Touch.** The return tuple is extended from `(spikes, snapshots, sig_data, kcc_data)` to `(spikes, snapshots, sig_data, kcc_data, zstr_stats: Option<ZstrStats>, ring_metadata: Option<F2RingMetadata>)`. Pass-through; no per-stream logic changes. | Coordinate with multi-stream lane; the change is mechanical. |

---

## 12. Open questions / known unknowns

1. **Ghost-tile kernel-side overflow flag.** The kernel at
   `ghost_tile.rs:354-368` (FFI to `prism_ghost_pipe_stage_launch`)
   silently skips writes once `n_frames_written >= max_records`. To
   distinguish "ring sized correctly, no overflow" from "ring saturated,
   N frames lost," the kernel would need to atomic-or a host-readable
   `overflow_seen` flag. **In v1 of the sidecar this is reported as
   `frames_dropped: null` with `overflow_inferred = (head_seq == max_records)`.**
   Future kernel patch can promote this to a real count.
2. **Backpressure events.** Currently `null`. To populate, the Reaper
   would need to record SQ-occupancy histograms (e.g. fraction of frames
   where `submit_and_wait` blocked > N µs). This is a future optimization
   and is not blocking.
3. **Multi-replicate consensus runs.** The F2 sidecars are emitted
   **per single run**. Replicate consensus (`scripts/prism_replicate.py`)
   reads N copies; an aggregate `f2_consensus_completeness.json` is a
   future Python (offline) deliverable, not part of v1.
4. **Per-stream `transform_dag.json::artifacts[]` granularity.** Agent 1
   has not yet specified whether ghost / zstr binaries are one
   `DagArtifactRef` per stream or aggregated. F2 emits one `commits[]`
   entry per file — straightforward to align with whatever shape Agent 1
   chooses.

---

## Appendix A — Citation index (file:line for every load-bearing claim)

| Claim | Citation |
|---|---|
| ZSTR consumer entry point | `crates/prism-nhs/src/zstr.rs:443` |
| ZSTR consumer thread name | `crates/prism-nhs/src/zstr.rs:451` |
| ZSTR consumer loop body | `crates/prism-nhs/src/zstr.rs:564-768` |
| ZSTR consumer SPIN_LIMIT constant | `crates/prism-nhs/src/zstr.rs:562` |
| io_uring init (size 32) | `crates/prism-nhs/src/zstr.rs:535` |
| io_uring `IORING_OP_WRITE` push | `crates/prism-nhs/src/zstr.rs:600-619` |
| Channel-A O_DIRECT open | `crates/prism-nhs/src/zstr.rs:466-479` |
| Channel-B O_DIRECT|O_DSYNC open | `crates/prism-nhs/src/zstr.rs:494-498` |
| Channel-A fsync (teardown) | `crates/prism-nhs/src/zstr.rs:822-830` |
| Channel-B fsync (teardown) | `crates/prism-nhs/src/zstr.rs:831-842` |
| `ZstrStats` struct | `crates/prism-nhs/src/zstr.rs:389-393` |
| `ZstrRing` struct + alignment | `crates/prism-nhs/src/zstr.rs:130-141, 215` |
| G21 alignment check | `crates/prism-nhs/src/zstr.rs:215, 233-246, 457-464` |
| G29 force-norm finite trap | `crates/prism-nhs/src/zstr.rs:654-667` |
| ZSTR `cuMemHostAlloc(PORTABLE\|DEVICEMAP)` | `crates/prism-nhs/src/zstr.rs:195-208` |
| ZSTR ring slot count `N_SLOTS = 5` | `crates/prism-nhs/src/zstr.rs:150` |
| ZSTR header size 4096 | `crates/prism-nhs/src/zstr.rs:67-114, 357-383` |
| `GhostTileFrame` size 4096 | `crates/prism-nhs/src/ghost_tile.rs:80-162, 198-213` |
| `GhostTileRing` struct | `crates/prism-nhs/src/ghost_tile.rs:237-249` |
| `GhostTileRing::n_frames_written()` | `crates/prism-nhs/src/ghost_tile.rs:316-320` |
| `GhostTileRing::COUNTER_SECTOR_BYTES` | `crates/prism-nhs/src/ghost_tile.rs:260` |
| `GhostTileRing::allocate` | `crates/prism-nhs/src/ghost_tile.rs:265-310` |
| `PinnedTelemetryRing<T>` (legacy 3-slot) | `crates/prism-nhs/src/ghost_telemetry.rs:95-99` |
| `create_non_blocking_telemetry_stream` | `crates/prism-nhs/src/ghost_telemetry.rs:242-257` |
| `is_pinned_host` (DMA attestation) | `crates/prism-nhs/src/ghost_telemetry.rs:279-298` |
| `Tier8DeferredDrainContext` struct | `crates/prism-nhs/src/fused_engine.rs:84-92` |
| `TIER8_DEFERRED_DRAIN_COUNT` (atomic) | `crates/prism-nhs/src/persistent_engine.rs:54-55` |
| Drain count increment | `crates/prism-nhs/src/persistent_engine.rs:1581-1605` |
| `tier8_diag_post_gasp_sync_gate` | `crates/prism-nhs/src/captured_pipeline.rs:295-447` |
| `post_t7_sync` gate string | `crates/prism-nhs/src/captured_pipeline.rs:3224, 3262` |
| `args.ghost_telemetry_io_uring` selector | `crates/prism-nhs/src/bin/nhs_rt_full.rs:329-337` |
| Per-stream Reaper join + log | `crates/prism-nhs/src/bin/nhs_rt_full.rs:8624-8637` |
| Channel-B legacy teardown writer | `crates/prism-nhs/src/bin/nhs_rt_full.rs:8589-8611` |
| ZSTR ring alloc site (V2) | `crates/prism-nhs/src/bin/nhs_rt_full.rs:5603-5656` |
| Ghost ring alloc site (V2) | `crates/prism-nhs/src/bin/nhs_rt_full.rs:5769-5795` |
| `spawn_zstr_consumer` call site (overlay) | `crates/prism-nhs/src/bin/nhs_rt_full.rs:6248-6253` |
| `spawn_zstr_consumer` call site (monolithic) | `crates/prism-nhs/src/bin/nhs_rt_full.rs:6106-6111` |
| V2 ignition summary writer (Phase 8b) | `crates/prism-nhs/src/bin/nhs_rt_full.rs:9942-10048` |
| `fnv1a64` local helper | `crates/prism-nhs/src/bin/nhs_rt_full.rs:9948-9957` |
| ghost_tiles head/tail/size fingerprint | `crates/prism-nhs/src/bin/nhs_rt_full.rs:9972-10005` |
| `phase3_run_id` mint | `crates/prism-nhs/src/bin/nhs_rt_full.rs:3959-3963` |
| `output_base` mint | `crates/prism-nhs/src/bin/nhs_rt_full.rs:9181, 10736` |
| TIER 8 deferred-drain caveat doc | `docs/TIER8_GRAPH_TOPOLOGY.md:99-107` |
| EXECUTION_POLICY Layer 1 (Rust) | `docs/EXECUTION_POLICY.md:23, 33` |
| `io-uring = "0.5.13"` dependency | `crates/prism-nhs/Cargo.toml:60` |
| `libc = "0.2"` dependency | `crates/prism-nhs/Cargo.toml:50-52` |
| build.rs has no liburing linkage | `crates/prism-nhs/build.rs:1-80` (verified absent) |
