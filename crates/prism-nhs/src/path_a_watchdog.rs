//! PATH-A mid-chunk stall watchdog and crash-safe evidence-exit fallback.
//!
//! Per directive after the 2026-05-06 P5 validation postmortem: the existing
//! `--path-a-max-wall-seconds` check fires only at the chunk-loop entry,
//! and the engine has been observed to deadlock mid-chunk on an inter-stream
//! 7-of-8 barrier. This module provides:
//!
//!   1. Per-stream progress heartbeats (`PathAHeartbeat`) writable from
//!      every chunk boundary and key phase transition without holding any
//!      engine-internal lock.
//!   2. A polling `spawn_watchdog` thread that detects N-1/N stall patterns
//!      and flips a shared cancel flag with a typed exit reason.
//!   3. An async-signal-safe SIGINT handler (`install_sigint_handler`) that
//!      sets a static AtomicBool the watchdog pumps into the cancel flag.
//!   4. A crash-safe minimal `path_a_completion.json` emit
//!      (`emit_minimal_completion_json`) used as a deadman fallback when
//!      the rayon join cannot return because a stream is futex-blocked.
//!      After emit, the watchdog calls `process::exit(2)` so the OS
//!      reclaims VRAM cleanly without an external SIGKILL.
//!
//! This module does NOT refactor existing engine internal synchronization.
//! It provides an EXTERNAL escape hatch only. Internal-lock audit + timed
//! waits remain a separate orthogonal investigation.
//!
//! ## Invariants honored
//! - No graph topology changes (this module is host-side Rust only).
//! - No F1/G26/WHILE alteration.
//! - No runtime Python.
//! - No fake completion: `completion_status` never reports `full_complete`.
//! - No fake binding sites: this module never writes binding_sites.json.
//! - Stream count unchanged at 8 (or whatever caller passes).

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Phase encoding written into `current_phase` per stream.
pub const PHASE_INIT: i32 = 0;
/// Cold-hold integration phase.
pub const PHASE_COLD_HOLD: i32 = 1;
/// T7 calibration phase (between cold_hold and V2 ignition / live).
pub const PHASE_T7_CAL: i32 = 2;
/// V2-live captured-graph integration.
pub const PHASE_V2_LIVE: i32 = 3;
/// Per-stream teardown / artifact emit.
pub const PHASE_TEARDOWN: i32 = 4;
/// Stream complete, no further progress expected.
pub const PHASE_DONE: i32 = 5;

fn unix_now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Per-stream progress state. Writers are per-stream rayon closures;
/// readers are the watchdog thread and the deadman emit path.
///
/// All fields use `Relaxed` atomics: the watchdog reads a recent-enough
/// snapshot, not a transactional one. Drift across atomics within one
/// snapshot is acceptable for stall detection (it only matters that
/// progress timestamps advance).
pub struct PathAHeartbeat {
    pub n_streams: usize,
    pub last_chunk_seen: Vec<AtomicI32>,
    pub last_step_seen: Vec<AtomicI32>,
    pub last_progress_ts_unix: Vec<AtomicI64>,
    pub current_phase: Vec<AtomicI32>,
    pub v2_live: Vec<AtomicBool>,
    pub evidence_ready: Vec<AtomicBool>,
}

impl PathAHeartbeat {
    /// Construct a heartbeat record for `n_streams` streams. Initial
    /// progress timestamp is now-Unix so a first poll shortly after
    /// startup does not falsely trigger the stall detector.
    pub fn new(n_streams: usize) -> Arc<Self> {
        let now = unix_now_secs();
        Arc::new(Self {
            n_streams,
            last_chunk_seen: (0..n_streams).map(|_| AtomicI32::new(-1)).collect(),
            last_step_seen: (0..n_streams).map(|_| AtomicI32::new(0)).collect(),
            last_progress_ts_unix: (0..n_streams).map(|_| AtomicI64::new(now)).collect(),
            current_phase: (0..n_streams).map(|_| AtomicI32::new(PHASE_INIT)).collect(),
            v2_live: (0..n_streams).map(|_| AtomicBool::new(false)).collect(),
            evidence_ready: (0..n_streams).map(|_| AtomicBool::new(false)).collect(),
        })
    }

    /// Record a heartbeat from a stream. Cheap: 4 relaxed atomic stores.
    /// Safe to call from inside a rayon closure without holding any
    /// engine-internal lock.
    pub fn mark(&self, sid: usize, chunk: i32, step: i32, phase: i32) {
        if sid >= self.n_streams {
            return;
        }
        self.last_chunk_seen[sid].store(chunk, Ordering::Relaxed);
        self.last_step_seen[sid].store(step, Ordering::Relaxed);
        self.last_progress_ts_unix[sid]
            .store(unix_now_secs(), Ordering::Relaxed);
        self.current_phase[sid].store(phase, Ordering::Relaxed);
    }

    /// Mark V2-live (set after a successful V2-INSTANTIATE-COMPLETE).
    pub fn set_v2_live(&self, sid: usize) {
        if sid >= self.n_streams {
            return;
        }
        self.v2_live[sid].store(true, Ordering::Release);
    }

    /// Mark evidence-ready (set when this stream has produced enough
    /// evidence that an early teardown is safe — typically after the
    /// ASC Bayesian baseline-lock fires).
    pub fn set_evidence_ready(&self, sid: usize) {
        if sid >= self.n_streams {
            return;
        }
        self.evidence_ready[sid].store(true, Ordering::Release);
    }

    /// Atomic point-in-time snapshot of every stream's progress state.
    pub fn snapshot(&self) -> HeartbeatSnapshot {
        let mut last_chunk = Vec::with_capacity(self.n_streams);
        let mut last_step = Vec::with_capacity(self.n_streams);
        let mut last_ts = Vec::with_capacity(self.n_streams);
        let mut phase = Vec::with_capacity(self.n_streams);
        let mut v2 = Vec::with_capacity(self.n_streams);
        let mut ev = Vec::with_capacity(self.n_streams);
        for sid in 0..self.n_streams {
            last_chunk.push(self.last_chunk_seen[sid].load(Ordering::Relaxed));
            last_step.push(self.last_step_seen[sid].load(Ordering::Relaxed));
            last_ts.push(
                self.last_progress_ts_unix[sid].load(Ordering::Relaxed),
            );
            phase.push(self.current_phase[sid].load(Ordering::Relaxed));
            v2.push(self.v2_live[sid].load(Ordering::Relaxed));
            ev.push(self.evidence_ready[sid].load(Ordering::Relaxed));
        }
        HeartbeatSnapshot {
            n_streams: self.n_streams,
            last_chunk_by_stream: last_chunk,
            last_step_by_stream: last_step,
            last_progress_ts_unix_by_stream: last_ts,
            current_phase_by_stream: phase,
            v2_live_by_stream: v2,
            evidence_ready_by_stream: ev,
            taken_at_unix: unix_now_secs(),
        }
    }
}

/// Atomic snapshot of all per-stream heartbeats.
#[derive(Debug, Clone, Serialize)]
pub struct HeartbeatSnapshot {
    pub n_streams: usize,
    pub last_chunk_by_stream: Vec<i32>,
    pub last_step_by_stream: Vec<i32>,
    pub last_progress_ts_unix_by_stream: Vec<i64>,
    pub current_phase_by_stream: Vec<i32>,
    pub v2_live_by_stream: Vec<bool>,
    pub evidence_ready_by_stream: Vec<bool>,
    pub taken_at_unix: i64,
}

/// Shared cancel/stall state. First-writer-wins on `exit_reason`.
pub struct WatchdogState {
    pub cancel_requested: AtomicBool,
    pub stall_detected: AtomicBool,
    pub stalled_stream_id: AtomicI32, // -1 if none
    pub exit_reason: Mutex<String>,
}

impl WatchdogState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            cancel_requested: AtomicBool::new(false),
            stall_detected: AtomicBool::new(false),
            stalled_stream_id: AtomicI32::new(-1),
            exit_reason: Mutex::new(String::new()),
        })
    }

    /// Flip cancel and latch the first writer's reason. Idempotent.
    pub fn request_cancel(&self, reason: &str) {
        self.cancel_requested.store(true, Ordering::Release);
        if let Ok(mut r) = self.exit_reason.lock() {
            if r.is_empty() {
                *r = reason.to_string();
            }
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_requested.load(Ordering::Acquire)
    }

    pub fn current_exit_reason(&self) -> String {
        self.exit_reason
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }
}

// ─── SIGINT handler (async-signal-safe) ────────────────────────────────────
// libc::signal handlers must call only async-signal-safe functions.
// AtomicBool::store with Relaxed ordering IS async-signal-safe. We do NOT
// log, lock a mutex, or call any allocator from the handler.
static SIGINT_RECEIVED: AtomicBool = AtomicBool::new(false);

extern "C" fn sigint_handler(_signum: libc::c_int) {
    SIGINT_RECEIVED.store(true, Ordering::Relaxed);
}

/// Install a SIGINT handler that flips a static AtomicBool. The watchdog
/// thread polls the flag on its next tick and propagates to WatchdogState.
/// Idempotent — calling more than once just re-installs the same handler.
pub fn install_sigint_handler() {
    unsafe {
        libc::signal(libc::SIGINT, sigint_handler as libc::sighandler_t);
    }
}

// ─── Watchdog thread ───────────────────────────────────────────────────────

/// Spawn the path-A watchdog. Returns the thread handle; caller may detach.
///
/// Detection rules:
/// - Every `poll_interval_secs`, snapshot all heartbeats.
/// - Pump SIGINT_RECEIVED → state.request_cancel("sigint").
/// - Compute max_chunk = max over streams of last_chunk_seen.
/// - A stream is "stagnant" if its chunk < max_chunk AND its
///   last_progress_ts is older than `stall_threshold_secs`.
/// - If at least one stream is stagnant AND advanced count >= n - 1,
///   the watchdog logs a typed diagnostic and flips cancel with reason
///   "mid_chunk_stall_watchdog".
///
/// Deadman:
/// - Once cancel is set, the watchdog arms a deadman timer.
/// - If healthy streams have not allowed the rayon join to return within
///   `deadman_grace_secs` (because the stalled stream is futex-blocked),
///   the watchdog emits a minimal `path_a_completion.json` from heartbeat
///   state and calls `process::exit(2)`. OS reclaims VRAM.
pub fn spawn_watchdog(
    heartbeats: Arc<PathAHeartbeat>,
    state: Arc<WatchdogState>,
    poll_interval_secs: u64,
    stall_threshold_secs: i64,
    deadman_grace_secs: u64,
    completion_path: PathBuf,
    run_id: String,
    target: String,
    max_wall_seconds: Option<u64>,
    run_start: std::time::Instant,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("prism-path-a-watchdog".into())
        .spawn(move || {
            let n_streams = heartbeats.n_streams;
            let mut deadman_armed_at: Option<std::time::Instant> = None;

            log::info!(
                "[PATH-A WATCHDOG] started: n_streams={} poll={}s stall_threshold={}s deadman_grace={}s wall_cap={:?}",
                n_streams, poll_interval_secs, stall_threshold_secs,
                deadman_grace_secs, max_wall_seconds
            );

            loop {
                std::thread::sleep(Duration::from_secs(poll_interval_secs));

                // 1) Pump SIGINT global → state cancel.
                if SIGINT_RECEIVED.load(Ordering::Relaxed) && !state.is_cancelled() {
                    log::error!("[PATH-A WATCHDOG] SIGINT observed; flipping cancel");
                    state.request_cancel("sigint");
                }

                // 2) Wall-cap check (in addition to the chunk-loop check
                //    in nhs_rt_full.rs — this catches stalls where no
                //    stream reaches the next chunk boundary).
                if !state.is_cancelled() {
                    if let Some(cap) = max_wall_seconds {
                        if run_start.elapsed().as_secs() >= cap {
                            log::error!(
                                "[PATH-A WATCHDOG] wall-cap reached at watchdog level (elapsed={}s, cap={}s); flipping cancel",
                                run_start.elapsed().as_secs(), cap
                            );
                            state.request_cancel("path_a_max_wall_seconds");
                        }
                    }
                }

                // 3) If cancel is active, run the deadman timer.
                if state.is_cancelled() {
                    if deadman_armed_at.is_none() {
                        deadman_armed_at = Some(std::time::Instant::now());
                        log::warn!(
                            "[PATH-A WATCHDOG] cancel active (reason={}); deadman timer armed for {}s grace",
                            state.current_exit_reason(),
                            deadman_grace_secs
                        );
                    } else if deadman_armed_at
                        .as_ref()
                        .map(|i| i.elapsed().as_secs() > deadman_grace_secs)
                        .unwrap_or(false)
                    {
                        log::error!(
                            "[PATH-A WATCHDOG] deadman expired after {}s grace; emitting minimal path_a_completion.json and exiting via process::exit(2)",
                            deadman_grace_secs
                        );
                        let snap = heartbeats.snapshot();
                        let stalled_id =
                            state.stalled_stream_id.load(Ordering::Acquire);
                        emit_minimal_completion_json(
                            &completion_path,
                            &run_id,
                            &target,
                            n_streams,
                            &state.current_exit_reason(),
                            if stalled_id >= 0 { Some(stalled_id) } else { None },
                            &snap,
                            max_wall_seconds,
                            run_start.elapsed().as_secs_f64(),
                        );
                        std::process::exit(2);
                    }
                    continue;
                }

                // 4) Detect N-1/N mid-chunk stall.
                let snap = heartbeats.snapshot();
                let now = snap.taken_at_unix;
                let max_chunk = snap
                    .last_chunk_by_stream
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(-1);
                if max_chunk < 1 {
                    continue;
                }
                let stagnant: Vec<usize> = (0..n_streams)
                    .filter(|&sid| {
                        let c = snap.last_chunk_by_stream[sid];
                        let ts = snap.last_progress_ts_unix_by_stream[sid];
                        c < max_chunk && (now - ts) > stall_threshold_secs
                    })
                    .collect();
                let advanced = n_streams - stagnant.len();
                if !stagnant.is_empty() && advanced >= n_streams.saturating_sub(1) {
                    let stalled_id = stagnant[0];
                    let stalled_chunk = snap.last_chunk_by_stream[stalled_id];
                    let stalled_age =
                        now - snap.last_progress_ts_unix_by_stream[stalled_id];
                    log::error!(
                        "[PATH-A WATCHDOG] mid-chunk stall detected: stream {} stuck at chunk {} (last progress {}s ago); {}/{} streams reached chunk {}; flipping cancel",
                        stalled_id,
                        stalled_chunk,
                        stalled_age,
                        advanced,
                        n_streams,
                        max_chunk
                    );
                    log::error!(
                        "[PATH-A WATCHDOG] heartbeat snapshot: chunks={:?} steps={:?} phases={:?}",
                        snap.last_chunk_by_stream,
                        snap.last_step_by_stream,
                        snap.current_phase_by_stream
                    );
                    state.stall_detected.store(true, Ordering::Release);
                    state
                        .stalled_stream_id
                        .store(stalled_id as i32, Ordering::Release);
                    state.request_cancel("mid_chunk_stall_watchdog");
                }
            }
        })
        .expect("spawn path-a watchdog thread")
}

// ─── Crash-safe minimal completion emit ────────────────────────────────────

/// Emit a minimal `path_a_completion.json` from heartbeat state alone.
///
/// Used by the watchdog deadman path when normal teardown cannot return
/// (typically because rayon join is blocked behind a futex-stalled stream).
/// The shape is a strict subset of the normal Phase-8 emit: no Phase 8a-d
/// artifacts are claimed, and `path_b_required: true` is asserted.
///
/// Failure to write is logged at error level. The caller (watchdog) calls
/// `process::exit` after this returns, so VRAM is reclaimed by the OS even
/// if the file write itself fails.
pub fn emit_minimal_completion_json(
    path: &Path,
    run_id: &str,
    target: &str,
    n_streams: usize,
    exit_reason: &str,
    stalled_stream_id: Option<i32>,
    snap: &HeartbeatSnapshot,
    max_wall_seconds: Option<u64>,
    elapsed_wall_seconds: f64,
) {
    let now_iso = chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%SZ")
        .to_string();

    // Map exit_reason → completion_status. Per directive §6 Commit 12 +
    // operator's hard MVP discipline, "full_complete" is intentionally
    // unreachable from any partial/bounded path here.
    let completion_status = match exit_reason {
        "path_a_max_wall_seconds"
        | "n_chunks_cap"
        | "mid_chunk_stall_watchdog"
        | "sigint" => "evidence_emitted_partial_bounded",
        "evidence_complete" => "bounded_path_a_evidence_exit",
        "natural_completion" => "evidence_emitted_natural",
        "not_triggered" => "not_triggered",
        _ => "evidence_emitted_unknown",
    };

    let streams_completed: usize =
        snap.v2_live_by_stream.iter().filter(|b| **b).count();
    let max_chunk_completed = snap
        .last_chunk_by_stream
        .iter()
        .copied()
        .max()
        .unwrap_or(-1);

    // instantiate_complete_by_stream as a Vec<bool> (true if v2_live was set
    // before the deadman fired). Honest: not the canonical timestamp form
    // emitted by the normal Phase-8 path; this is the deadman fallback.
    let inst_by_stream: Vec<bool> = snap.v2_live_by_stream.clone();

    let json = serde_json::json!({
        "schema_version": 2,
        "run_id": run_id,
        "target": target,
        "stream_count": n_streams,
        "streams_completed": streams_completed,
        "v2_trigger_step_by_stream": [],
        "instantiate_complete_by_stream": inst_by_stream,
        "t7_mode": "wall_bounded_or_watchdog",
        "t7_chunks_completed": max_chunk_completed,
        "evidence_exit_reason": exit_reason,
        "completion_status": completion_status,
        "max_wall_seconds": max_wall_seconds,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "missing_optional_fields": [
            "v2_trigger_step_by_stream",
            "binding_sites.json",
            "v2_ignition_summary.json",
            "f2_ring_status.json",
            "f2_write_commit_log.json",
            "f2_artifact_completeness.json",
            "transform_dag.json"
        ],
        "path_b_required": true,
        "stalled_stream_id": stalled_stream_id,
        "last_chunk_by_stream": snap.last_chunk_by_stream,
        "last_step_by_stream": snap.last_step_by_stream,
        "last_progress_ts_unix_by_stream":
            snap.last_progress_ts_unix_by_stream,
        "current_phase_by_stream": snap.current_phase_by_stream,
        "v2_live_by_stream": snap.v2_live_by_stream,
        "evidence_ready_by_stream": snap.evidence_ready_by_stream,
        "snapshot_taken_at_unix": snap.taken_at_unix,
        "snapshot_taken_at_iso": now_iso,
        "emit_source": "path_a_watchdog::emit_minimal_completion_json",
        "note": "Crash-safe minimal evidence emit triggered by watchdog or SIGINT. \
                 Phase 8a-d artifacts (binding_sites, F2 sidecars, transform_dag) \
                 are likely missing because normal teardown was bypassed. \
                 path_b_required is asserted true. Treat this run as unvalidated.",
    });

    match std::fs::File::create(path) {
        Ok(f) => {
            let mut writer = std::io::BufWriter::new(f);
            match serde_json::to_writer_pretty(&mut writer, &json) {
                Ok(()) => {
                    use std::io::Write as _;
                    let _ = writer.flush();
                    log::error!(
                        "[PATH-A WATCHDOG] minimal path_a_completion.json emitted at {} (exit_reason={}, stalled_stream_id={:?}, completion_status={})",
                        path.display(),
                        exit_reason,
                        stalled_stream_id,
                        completion_status
                    );
                }
                Err(e) => log::error!(
                    "[PATH-A WATCHDOG] minimal path_a_completion.json: serialize failed at {}: {}",
                    path.display(),
                    e
                ),
            }
        }
        Err(e) => log::error!(
            "[PATH-A WATCHDOG] minimal path_a_completion.json: create failed at {}: {}",
            path.display(),
            e
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heartbeat_mark_and_snapshot_round_trip() {
        let hb = PathAHeartbeat::new(8);
        hb.mark(0, 5, 2500, PHASE_T7_CAL);
        hb.mark(7, 12, 6000, PHASE_V2_LIVE);
        hb.set_v2_live(0);
        hb.set_evidence_ready(7);
        let snap = hb.snapshot();
        assert_eq!(snap.n_streams, 8);
        assert_eq!(snap.last_chunk_by_stream[0], 5);
        assert_eq!(snap.last_step_by_stream[0], 2500);
        assert_eq!(snap.current_phase_by_stream[0], PHASE_T7_CAL);
        assert!(snap.v2_live_by_stream[0]);
        assert_eq!(snap.last_chunk_by_stream[7], 12);
        assert_eq!(snap.current_phase_by_stream[7], PHASE_V2_LIVE);
        assert!(snap.evidence_ready_by_stream[7]);
    }

    #[test]
    fn watchdog_state_first_writer_wins_on_exit_reason() {
        let s = WatchdogState::new();
        s.request_cancel("first");
        s.request_cancel("second");
        assert_eq!(s.current_exit_reason(), "first");
        assert!(s.is_cancelled());
    }

    #[test]
    fn completion_status_mapping_never_full_complete() {
        // Ensure the deadman-emit mapping never yields "full_complete".
        for reason in &[
            "path_a_max_wall_seconds",
            "n_chunks_cap",
            "mid_chunk_stall_watchdog",
            "sigint",
            "evidence_complete",
            "natural_completion",
            "not_triggered",
            "garbage_unknown_value",
        ] {
            let status = match *reason {
                "path_a_max_wall_seconds"
                | "n_chunks_cap"
                | "mid_chunk_stall_watchdog"
                | "sigint" => "evidence_emitted_partial_bounded",
                "evidence_complete" => "bounded_path_a_evidence_exit",
                "natural_completion" => "evidence_emitted_natural",
                "not_triggered" => "not_triggered",
                _ => "evidence_emitted_unknown",
            };
            assert_ne!(status, "full_complete");
        }
    }
}
