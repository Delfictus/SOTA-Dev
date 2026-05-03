//! ZSTR — Zero-Stall Telemetry Ring
//!
//! Implements the "Compute-Stream-Consume" exfiltration architecture for
//! V2 coordinate snapshots.  Three asynchronous planes:
//!
//! 1. **Device Plane** (CUDA graph) — physics + adjudication + position
//!    staging via `zstr_pos_stage_f4_kernel` + fence via
//!    `zstr_signal_completion_kernel`.
//!
//! 2. **Exfiltration Plane** (ZSTR consumer thread) — spins on pinned
//!    `completion_fence`; on READY, writes the frame to NVMe via
//!    `O_DIRECT` (bypasses page cache), resets the fence, and advances
//!    the ring index.
//!
//! 3. **Control Plane** (Rust orchestrator) — allocates the ring before
//!    the CUDA graph capture, passes device-side pointers into the graph
//!    nodes, spawns the consumer thread, and joins it at teardown.
//!
//! # Triple-buffer ring geometry
//!
//! ```text
//! slot = frame_idx % 3
//!
//! slot N   : GPU writing (current cudaMemcpyAsync target)
//! slot N-1 : in-flight / verification buffer
//! slot N-2 : host consumer locked for O_DIRECT write
//! ```
//!
//! # Gate G21 — Alignment verification
//!
//! `ZstrRing::allocate` asserts `pinned_ptr % 4096 == 0` and sets
//! `alignment_ok`.  The consumer thread panics if the gate is not passed
//! before it attempts any `O_DIRECT` write.
//!
//! # Gate G22 — Atomic v4 (CUDA kernel side)
//!
//! See `asc_steering.cu` for `atom.global.add.v4.f32` PTX; run:
//! `nvcc -arch=sm_120 --ptx asc_steering.cu | grep "atom.global.add.v4"`
//! to close G22.

#![cfg(feature = "gpu")]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

// ─── ZstrFrameHeader ────────────────────────────────────────────────────────

/// Header for one ZSTR ring slot (pinned, 4096-byte aligned).
///
/// T11 (Action-Recovery): the slot now carries forces in addition to
/// positions; the header gains `n_atoms`, `gear_id`, and `force_norm`
/// so the offline replay can reconstruct (a) the system size, (b) the
/// G26 chronometric gear active during the frame (set by Wave B; default 0
/// in Wave A), and (c) the L2 norm of the integrator's total force after
/// ASC steering — a single scalar witness that the steering is bounded
/// and finite.
///
/// Immediately followed in the same allocation by:
///   [positions: n_atoms * 3 * f32]
///   [forces:    n_atoms * 3 * f32]
///   [padding to next 4096-byte boundary]
///
/// The full slot size is computed and padded to 4096 bytes so that
/// `O_DIRECT` writes are always sector-aligned.
#[repr(C, align(4096))]
pub struct ZstrFrameHeader {
    /// Monotonically increasing epoch counter (0-based).
    pub frame_idx: u64,                  // 0..8
    /// Integration timestep in picoseconds at the time of capture.
    pub dt: f32,                         // 8..12
    /// adjudication_code written by the InterferometricAdjudicator
    /// (0 = Prune, 1 = Construct, 2 = Violation).
    pub adjudication_code: u32,          // 12..16
    /// 0 = Dirty/Writing (GPU owns).  1 = Ready (host may flush).
    /// Written by `zstr_signal_completion_kernel` after __threadfence_system().
    pub completion_fence: u32,           // 16..20
    /// Atom count for this slot (replay payload size = n_atoms * 3 * 4
    /// bytes for positions and again for forces). Stamped at capture
    /// time so the consumer doesn't need a side-channel size.
    pub n_atoms: u32,                    // 20..24
    /// G26 chronometric gear in effect when this frame retired.
    /// 0 = 0.5 fs (high-res), 1 = 2.0 fs (default), 2 = 4.0 fs (HMR sprint),
    /// 3 = abort-trap. Wave A leaves this at 0 (gear logic is Wave B);
    /// the field is reserved here so no header layout change is needed
    /// when the SWITCH lands.
    pub gear_id: u32,                    // 24..28
    /// L2 norm ‖F‖₂ = sqrt(Σ_i Σ_d F_id²) of the post-ASC integrator
    /// force buffer at this slot. NaN ⇒ G29 trap (steering produced
    /// non-physical forces; campaign aborts on detection).
    pub force_norm: f32,                 // 28..32
    /// **M1.2.18.5 — Total External Work (kcal/mol)** integrated over
    /// this chunk window.  Snapshotted from `*adj.d_external_work`
    /// (FFI offset 128 → F2-pool W_ext buffer) at frame retirement.
    /// `nan` is reserved (write zero when the W_ext pointer is null
    /// pre-M1.2.18.5 wire-up).  Off-line replay subtracts this from
    /// `potential_energy` deltas to derive the conservative-only
    /// component of ΔV (First Law audit).
    pub external_work: f64,              // 32..40
    /// **M1.2.18.5 — System Potential Energy V_t (kcal/mol)** as written
    /// by the captured Hamiltonian-Auditor reduce node into
    /// `adj.d_potential_energy` (FFI offset 112).  Snapshotted at frame
    /// retirement so off-line replay can rebuild the V trace without
    /// re-reducing per-atom components.  Zero pre-Hamiltonian wire-up.
    pub potential_energy: f64,           // 40..48
    /// Pads the header to a full 4096-byte sector. align(4096) on the
    /// struct already enforces tail padding, but making it explicit
    /// here pins the slot layout so the offline replay format does not
    /// silently drift if the struct is later extended.
    /// M1.2.18.5: 1016 → 1012 (lost 16 B = 4 × u32 to external_work +
    /// potential_energy f64 fields above).
    pub _padding: [u32; 1012],           // 48..4096
}

// ZstrFrameHeader meaningful payload (post-M1.2.18.5): 48 bytes.
// size_of::<ZstrFrameHeader>() == 4096 (explicit + align(4096)).
// Slot layout post-T11:
//   [header: 4096 B] [positions: n_atoms*3*4 B] [forces: n_atoms*3*4 B] [pad→4096]

// ─── ZstrRing ────────────────────────────────────────────────────────────────

/// Triple-buffered pinned-host ring for ZSTR frame exfiltration.
///
/// Owns a single `cuMemAllocHost_v2` allocation covering 3 slots of
/// `frame_size` bytes each.  `frame_size` is computed from `n_atoms`
/// and padded to the next 4096-byte multiple for `O_DIRECT` compat.
pub struct ZstrRing {
    /// Base of the pinned allocation (page-locked; DMA-accessible).
    pub pinned_ptr: *mut u8,
    /// Per-slot byte size (4096-byte padded).
    pub frame_size: usize,
    /// Atom count (determines position payload size).
    pub n_atoms: usize,
    /// G21 gate: true iff `pinned_ptr % 4096 == 0`.
    pub alignment_ok: bool,
}

unsafe impl Send for ZstrRing {}
unsafe impl Sync for ZstrRing {}

impl ZstrRing {
    /// Number of ring slots (5-slot resilience ring — Amendment 2.2).
    /// 5 slots × ~61 KB/frame = ~305 KB pinned headroom vs. P99 NVMe jitter.
    /// Contention policy: frame dropped (not queued) when slot still dirty.
    pub const N_SLOTS: usize = 5;

    /// Allocate the triple-buffer in pinned host memory.
    ///
    /// Frame layout per slot (T11 — forces co-resident with positions):
    /// ```
    /// [ZstrFrameHeader: 4096 B] [positions: n_atoms*3*4 B]
    ///   [forces: n_atoms*3*4 B] [pad→next 4096 multiple]
    /// ```
    pub fn allocate(n_atoms: usize) -> Result<Self, String> {
        // ZstrFrameHeader is align(4096) → size_of = 4096 (padded by Rust).
        // Meaningful header data occupies only the first 32 bytes.
        let header_bytes = std::mem::size_of::<ZstrFrameHeader>(); // == 4096
        // Operator Amendment 3.11 — ZSTR Payload Alignment Rectification.
        //
        // `zstr_force_stage_f4_kernel` issues `STG.E.128` (float4 store) at
        // base `slot_base + force_offset_in_slot`.  Blackwell sm_120
        // requires the destination address to be 16-byte aligned or the
        // hardware traps with MISALIGNED_ADDRESS (kernel-printf triage
        // 2026-05-02 isolated the trap to this kernel).
        //
        // For any `n_atoms` where `n_atoms mod 4 != 0`, the raw pos
        // payload size `n_atoms * 12` is not a multiple of 16, leaving
        // `force_offset = 4096 + 12·n_atoms` at an 8-aligned (not 16-
        // aligned) offset within the slot.  Padding the pos payload up
        // to the next 16-byte boundary fixes the offset for the force
        // payload — the slot is still 4096-pad'd at the end so the
        // O_DIRECT total size is preserved.
        let pos_bytes_raw     = n_atoms * 3 * 4; // n_atoms × xyz × f32
        let pos_bytes_aligned = (pos_bytes_raw + 15) & !15;
        let force_bytes       = n_atoms * 3 * 4; // T11: forces appended
        let raw_frame    = header_bytes + pos_bytes_aligned + force_bytes;
        // Round up to next 4096-byte multiple for O_DIRECT sector alignment.
        let frame_size   = (raw_frame + 4095) & !4095;
        let total_bytes  = frame_size * Self::N_SLOTS;

        // M1.2.19.D — Mapped-memory standard (operator Amendment 3.13 §1.2).
        // Use cuMemHostAlloc with PORTABLE + DEVICEMAP so the page-locked
        // host buffer carries an explicit device-side virtual-address alias
        // (vs. the implicit-mapped behavior of cuMemAllocHost_v2).  On
        // Blackwell sm_120 + PCIe Gen5 the driver collapses them to the
        // same VA window, but the explicit flag makes the contract part
        // of the source — required for multi-node B200 portability where
        // the implicit behavior is not guaranteed.
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let flags = cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE
                  | cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP;
        let rc = unsafe {
            cudarc::driver::sys::cuMemHostAlloc(&mut ptr, total_bytes, flags)
        };

        if !matches!(rc, cudarc::driver::sys::CUresult::CUDA_SUCCESS) {
            return Err(format!(
                "cuMemHostAlloc(PORTABLE|DEVICEMAP) failed: {:?} \
                 (requested {} bytes for {} atoms × 3 slots)",
                rc, total_bytes, n_atoms
            ));
        }

        let pinned_ptr   = ptr as *mut u8;
        let alignment_ok = (pinned_ptr as usize) % 4096 == 0;

        // Zero-init: clears all completion_fence fields to 0 (GPU-owned).
        unsafe { std::ptr::write_bytes(pinned_ptr, 0u8, total_bytes); }

        if !alignment_ok {
            log::error!(
                "[ZSTR G21 FAIL] pinned_ptr {:p} not 4096-aligned — \
                 O_DIRECT writes will fault.  cuMemAllocHost_v2 \
                 guarantees page alignment; this is a driver bug.",
                pinned_ptr
            );
        } else {
            log::info!(
                "[ZSTR G21 PASS] pinned_ptr {:p} 4096-aligned ✓  \
                 frame_size={} bytes  total={} bytes  n_atoms={}",
                pinned_ptr, frame_size, total_bytes, n_atoms
            );
        }

        Ok(Self { pinned_ptr, frame_size, n_atoms, alignment_ok })
    }

    /// Pointer to the `ZstrFrameHeader` for ring slot `slot` (0..N_SLOTS).
    ///
    /// # Safety
    /// Caller must ensure no concurrent GPU write to the same slot.
    pub unsafe fn header_ptr(&self, slot: usize) -> *mut ZstrFrameHeader {
        debug_assert!(slot < Self::N_SLOTS);
        self.pinned_ptr.add(slot * self.frame_size) as *mut ZstrFrameHeader
    }

    /// Raw byte pointer to the positions payload inside slot `slot`.
    /// This is what `zstr_pos_stage_f4_kernel` writes into.
    ///
    /// Offset = slot * frame_size + sizeof(ZstrFrameHeader).
    pub fn positions_host_ptr(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < Self::N_SLOTS);
        let header_off = std::mem::size_of::<ZstrFrameHeader>();
        unsafe { self.pinned_ptr.add(slot * self.frame_size + header_off) }
    }

    /// Device-side CUdeviceptr for the positions payload of slot `slot`.
    /// Pass this to `cuMemcpyDtoHAsync_v2` as the destination.
    /// (Pinned memory is directly addressable from the GPU.)
    pub fn positions_cu_ptr(&self, slot: usize) -> u64 {
        self.positions_host_ptr(slot) as u64
    }

    /// Byte offset from a slot's base to the forces payload.
    /// Header (4096) + positions payload aligned up to 16-byte boundary.
    ///
    /// Operator Amendment 3.11 — pos payload `n_atoms*12` is rounded up
    /// to the next 16-byte boundary so this offset (and therefore the
    /// `slot_base + force_offset` address consumed by
    /// `zstr_force_stage_f4_kernel`) is 16-aligned for `STG.E.128`.
    /// Without padding, any `n_atoms mod 4 != 0` produced an 8-aligned
    /// offset → MISALIGNED_ADDRESS hardware trap on the first launch.
    pub fn forces_offset_in_slot(&self) -> usize {
        let header_bytes      = std::mem::size_of::<ZstrFrameHeader>();
        let pos_bytes_raw     = self.n_atoms * 3 * 4;
        let pos_bytes_aligned = (pos_bytes_raw + 15) & !15;
        header_bytes + pos_bytes_aligned
    }

    /// Raw byte pointer to the forces payload inside slot `slot` (T11).
    /// This is what `zstr_force_stage_f4_kernel` writes into.
    pub fn forces_host_ptr(&self, slot: usize) -> *mut u8 {
        debug_assert!(slot < Self::N_SLOTS);
        unsafe { self.pinned_ptr.add(slot * self.frame_size + self.forces_offset_in_slot()) }
    }

    /// Device-side CUdeviceptr for the forces payload of slot `slot` (T11).
    pub fn forces_cu_ptr(&self, slot: usize) -> u64 {
        self.forces_host_ptr(slot) as u64
    }

    /// Byte offset of `force_norm` within `ZstrFrameHeader`. Stable: 28.
    pub const fn force_norm_offset_in_slot() -> usize {
        offset_of_force_norm()
    }

    /// Device-side CUdeviceptr for the `completion_fence` field of slot `slot`.
    /// Pass to `zstr_signal_completion_kernel` as its argument.
    pub fn fence_cu_ptr(&self, slot: usize) -> u64 {
        let hdr = unsafe { self.header_ptr(slot) };
        // completion_fence offset within ZstrFrameHeader
        let field_off = offset_of_completion_fence();
        (hdr as u64) + field_off as u64
    }
}

impl Drop for ZstrRing {
    fn drop(&mut self) {
        if !self.pinned_ptr.is_null() {
            unsafe {
                let _ = cudarc::driver::sys::cuMemFreeHost(
                    self.pinned_ptr as *mut c_void
                );
            }
            self.pinned_ptr = std::ptr::null_mut();
        }
    }
}

/// Byte offset of `completion_fence` within the raw header bytes.
/// frame_idx(8) + dt(4) + adjudication_code(4) = 16.
const fn offset_of_completion_fence() -> usize { 16 }

/// Byte offset of `force_norm` within the raw header bytes.
/// frame_idx(8) + dt(4) + adjudication_code(4) + completion_fence(4)
/// + n_atoms(4) + gear_id(4) = 28.
const fn offset_of_force_norm() -> usize { 28 }

/// **M1.2.18.5** Byte offset of `external_work` within the raw header bytes.
/// `force_norm` ends at 32; `external_work` (f64) lives at 32..40.
pub const fn offset_of_external_work() -> usize { 32 }

/// **M1.2.18.5** Byte offset of `potential_energy` within the raw header bytes.
/// `external_work` ends at 40; `potential_energy` (f64) lives at 40..48.
pub const fn offset_of_potential_energy() -> usize { 40 }

// Compile-time pin: the header layout MUST match the offsets the C-side
// kernels and the offline replay parser depend on.  Drift here ⇒ silently
// corrupt force_norm / gear_id reads in the consumer thread.
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<ZstrFrameHeader>() == 4096);
    // M1.2.18.5 — header packed payload is 48 bytes:
    //   frame_idx       u64       8 B
    //   dt              f32       4 B
    //   adjudication    u32       4 B
    //   completion      u32       4 B
    //   n_atoms         u32       4 B
    //   gear_id         u32       4 B
    //   force_norm      f32       4 B
    //   external_work   f64       8 B
    //   potential_energy f64      8 B
    //   _padding       [u32;1012] 4048 B
    //   ─────────────────────────────────
    //   TOTAL                     4096 B
    assert!(
        size_of::<u64>()      // frame_idx
        + 6 * size_of::<u32>()    // dt + adj + fence + n_atoms + gear_id + force_norm
        + 2 * size_of::<f64>()    // M1.2.18.5: external_work + potential_energy
        + 1012 * size_of::<u32>() // tail padding
        == 4096
    );
};

// ─── ZSTR Consumer Thread ────────────────────────────────────────────────────

/// Message passed from the ZSTR consumer thread to the caller when
/// the run completes.
pub struct ZstrStats {
    pub frames_written: u64,
    pub frames_dropped: u64,   // fence-timeout drops
    pub bytes_written:  u64,
}

/// Spawn the io_uring + O_DIRECT ZSTR consumer thread.
///
/// **Wave 0 / #69 — Beast-Mode I/O.** Replaces the pwrite(2) path with
/// pure io_uring SQE submission.  No VFS read/write copy, no page
/// cache (O_DIRECT bypasses it; io_uring bypasses the buffered-IO
/// stack entirely).  Each ready ring slot is slammed straight into
/// the NVMe submission queue as an `IORING_OP_WRITE` SQE pointing at
/// the pinned-host slot bytes; the kernel's blk-mq layer hands them
/// to the device without an intermediate copy.
///
/// The thread:
/// 1. Opens `output_path` with `O_DIRECT | O_CREAT | O_WRONLY`
///    (path: under `args.output` in the campaign output dir; the
///    legacy `/tmp` path was eliminated in this commit).
/// 2. Initialises a 16-entry `IoUring` (5 inflight + headroom).
/// 3. For each frame epoch:
///    a. Computes `slot = frame_idx % N_SLOTS`.
///    b. Spin-polls `ZstrFrameHeader::completion_fence == 1` (GPU
///       has fenced all writes for this slot) or `stop` is set.
///    c. Reads + traps non-finite `force_norm` (G29).
///    d. Pushes one `IORING_OP_WRITE` SQE: fd + sector-aligned
///       offset + slot pointer + frame_size, `submit_and_wait(1)`,
///       reaps the CQE, checks `cqe.result() >= 0`.
///    e. Resets `force_norm` and `completion_fence` to 0 (returns
///       the slot to the GPU's writable pool).
///    f. Advances `frame_idx`.
/// 4. Flushes (fdatasync) and closes the file when `stop` fires.
///
/// **M1.2.20.C-H / T23 — Multi-Channel AMS Reaper**.  When invoked
/// with `ghost_ring=Some(_)` + `ghost_output_path=Some(_)`, the same
/// Reaper thread also drains the Channel-B `GhostTileRing` to its
/// own NVMe file.  The Ghost ring has no per-slot completion fence
/// (the kernel atomicAdds the leading counter, threadfence_system,
/// then writes the 1408-byte record), so the Reaper polls the
/// counter and flushes any slots that are at least
/// `GHOST_SAFETY_LAG=1` slots behind the live counter — this avoids
/// racing with a kernel still mid-write of the most-recent slot.
/// O_DIRECT is NOT used for Channel B because the 1408-byte record
/// stride is not 4096-aligned; Channel B uses regular buffered I/O
/// via the same io_uring SQ.  Channel A retains its O_DIRECT path.
///
/// Returns a `JoinHandle<ZstrStats>`.  `stats.bytes_written` is the
/// Channel-A total only; Channel-B byte count is logged separately
/// at thread teardown.
///
/// # Safety
/// `ring` and (if Some) `ghost_ring` must remain valid for the
/// lifetime of the returned thread.  The caller joins before dropping.
pub fn spawn_zstr_consumer(
    ring: Arc<ZstrRing>,
    output_path: std::path::PathBuf,
    stop: Arc<AtomicBool>,
    ghost_ring: Option<Arc<crate::ghost_tile::GhostTileRing>>,
    ghost_output_path: Option<std::path::PathBuf>,
) -> JoinHandle<ZstrStats> {
    std::thread::Builder::new()
        .name(format!("zstr-consumer"))
        .spawn(move || {
            use std::os::unix::fs::OpenOptionsExt;
            use io_uring::{IoUring, opcode, types};

            // G21 gate enforcement — consumer must not proceed if alignment failed.
            if !ring.alignment_ok {
                log::error!(
                    "[ZSTR consumer] G21_ALIGNMENT_PASS not met — \
                     pinned buffer not 4096-aligned.  O_DIRECT writes \
                     will fault.  Aborting consumer."
                );
                return ZstrStats { frames_written: 0, frames_dropped: 0, bytes_written: 0 };
            }

            let file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .custom_flags(libc::O_DIRECT)
                .open(&output_path);

            let file = match file {
                Ok(f) => f,
                Err(e) => {
                    log::error!("[ZSTR consumer] Cannot open {:?}: {}", output_path, e);
                    return ZstrStats { frames_written: 0, frames_dropped: 0, bytes_written: 0 };
                }
            };

            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();

            // **M1.2.20.C-H / T23** — Optional Channel-B ghost output file.
            // 1408-byte record stride is NOT 4096-aligned, so this fd is
            // opened WITHOUT O_DIRECT (regular buffered IO; same io_uring
            // submission queue).  When the operator authorizes a 4096-pad
            // record-stride change, we'll switch to O_DIRECT here.
            let ghost_state: Option<(std::fs::File, i32)> = ghost_output_path
                .as_ref()
                .filter(|_| ghost_ring.is_some())
                .and_then(|p| {
                    match std::fs::OpenOptions::new()
                        .write(true).create(true).truncate(true).open(p)
                    {
                        Ok(f) => {
                            let raw = f.as_raw_fd();
                            log::info!(
                                "[AMS T23] Channel-B fd={} opened (buffered IO; \
                                 1408 B stride non-O_DIRECT) → {:?}",
                                raw, p
                            );
                            Some((f, raw))
                        }
                        Err(e) => {
                            log::warn!(
                                "[AMS T23] Cannot open ghost output {:?}: {} \
                                 — Channel-B drain disabled, teardown-only flush",
                                p, e
                            );
                            None
                        }
                    }
                });
            let ghost_fd: i32 = ghost_state.as_ref().map(|(_, fd)| *fd).unwrap_or(-1);
            let mut last_ghost_slot_emitted: u32 = 0;
            let mut ghost_records_written: u64  = 0;
            let mut ghost_bytes_written:   u64  = 0;
            const GHOST_SAFETY_LAG:        u32  = 1;
            const GHOST_RECORD_STRIDE:     usize = 1408;

            // Wave 0 / #69 — io_uring instance.  16 entries: 5 ring slots
            // can be inflight at most, the rest is queue headroom for
            // bursts.  Build with default flags (no SQPOLL — we keep the
            // submit/wait synchronous-per-frame contract so slot recycle
            // ordering is preserved; switching to SQPOLL is a follow-up
            // once we pipeline >1 SQE per frame).
            let mut uring: IoUring = match IoUring::new(16) {
                Ok(u) => u,
                Err(e) => {
                    log::error!(
                        "[ZSTR consumer] io_uring init failed: {} \
                         — falling back to no-op (frames will all drop). \
                         Check kernel build for CONFIG_IO_URING=y.",
                        e
                    );
                    return ZstrStats { frames_written: 0, frames_dropped: 0, bytes_written: 0 };
                }
            };
            log::info!(
                "[ZSTR consumer] io_uring initialised: 16 SQEs, O_DIRECT \
                 fd={} → {:?}",
                fd, output_path
            );

            let frame_size   = ring.frame_size;
            let mut frame_idx: u64 = 0;
            let mut frames_written: u64 = 0;
            let mut frames_dropped: u64 = 0;
            let mut bytes_written:  u64 = 0;

            // Spin-timeout: ~1 ms at ~3 GHz before declaring a drop.
            // Avoids infinite stall if a kernel panic or graph abort
            // leaves the fence permanently at 0.
            const SPIN_LIMIT: u64 = 3_000_000;

            loop {
                if stop.load(Ordering::Relaxed) { break; }

                let slot = (frame_idx as usize) % ZstrRing::N_SLOTS;
                let hdr  = unsafe { &*ring.header_ptr(slot) };

                // Non-blocking spin on completion_fence.
                let mut spins: u64 = 0;
                while unsafe {
                    std::ptr::read_volatile(
                        &hdr.completion_fence as *const u32
                    )
                } == 0 {
                    std::hint::spin_loop();
                    spins += 1;
                    if spins >= SPIN_LIMIT {
                        break;
                    }
                    if stop.load(Ordering::Relaxed) { break; }
                }

                if spins >= SPIN_LIMIT {
                    // Fence timed out — GPU stalled or run ended.
                    if stop.load(Ordering::Relaxed) { break; }
                    frames_dropped += 1;
                    frame_idx += 1;
                    continue;
                }

                // Slot is READY.  io_uring zero-copy submit:
                //   pinned slot ptr → IORING_OP_WRITE SQE → NVMe queue.
                let slot_ptr = unsafe {
                    ring.pinned_ptr.add(slot * frame_size)
                } as *const u8;
                let offset_i64 = (frames_written * frame_size as u64) as i64;

                let write_e = opcode::Write::new(
                        types::Fd(fd),
                        slot_ptr,
                        frame_size as u32,
                    )
                    .offset(offset_i64)
                    .build()
                    .user_data(frame_idx);

                let push_rc = unsafe { uring.submission().push(&write_e) };
                if push_rc.is_err() {
                    // Submission queue full — drop frame, keep going.
                    log::warn!(
                        "[ZSTR consumer] io_uring SQ full at frame={}; \
                         dropping",
                        frame_idx
                    );
                    frames_dropped += 1;
                    frame_idx += 1;
                    continue;
                }
                let submit_rc = uring.submit_and_wait(1);
                let written: isize = match submit_rc {
                    Err(e) => {
                        log::warn!(
                            "[ZSTR consumer] io_uring submit_and_wait \
                             frame={} failed: {}",
                            frame_idx, e
                        );
                        -1
                    }
                    Ok(_) => {
                        // Reap the single completion we waited on.
                        match uring.completion().next() {
                            Some(cqe) => cqe.result() as isize,
                            None => {
                                log::warn!(
                                    "[ZSTR consumer] submit_and_wait \
                                     returned but no CQE present at \
                                     frame={}",
                                    frame_idx
                                );
                                -1
                            }
                        }
                    }
                };

                // ── G29 — Action-Recovery Zero-Trust trap.  Read the
                //    sqrt'd L2 norm written by zstr_force_norm_sqrt_kernel.
                //    Non-finite ⇒ ASC produced a non-physical force; we log
                //    loudly and drop the frame (the campaign should be
                //    aborted by the orchestrator on first detection — the
                //    Reaper does not have authority to kill the engine).
                let force_norm_val = unsafe {
                    std::ptr::read_volatile(
                        std::ptr::addr_of!((*ring.header_ptr(slot)).force_norm)
                    )
                };
                let g29_finite = force_norm_val.is_finite();
                if !g29_finite {
                    log::error!(
                        "[ZSTR G29 TRAP] frame={} slot={} force_norm={:?} non-finite — \
                         Action-Recovery Invariant violated; ASC steered into NaN/Inf. \
                         Frame dropped. Operator must abort campaign.",
                        frame_idx, slot, force_norm_val
                    );
                }

                if written < 0 {
                    // io_uring CQE.result() returns -errno on failure.
                    log::warn!(
                        "[ZSTR consumer] io_uring write frame={} failed: \
                         cqe.result={} (negative ⇒ -errno)",
                        frame_idx, written
                    );
                    frames_dropped += 1;
                } else if !g29_finite {
                    // Slot bytes ARE on disk (we wrote first), but the run
                    // is poisoned. Count it as a drop so the stats line
                    // reflects the broken slot count rather than a clean
                    // success.
                    frames_dropped += 1;
                } else {
                    frames_written += 1;
                    bytes_written  += written as u64;

                    if frame_idx % 1000 == 0 {
                        log::info!(
                            "[ZSTR] frame={} adj_code={} ‖F‖₂={:.6e} fence_was_ready  \
                             written={} MB",
                            frame_idx,
                            hdr.adjudication_code,
                            force_norm_val,
                            bytes_written >> 20,
                        );
                    }
                }

                // Reset fence + force_norm + completion_fence, returning the
                // slot to the GPU's writable pool.  Zeroing force_norm is
                // mandatory: the next force_stage kernel atomic-adds Σ‖F‖²
                // INTO this same field expecting it to start at 0.
                unsafe {
                    let hdr_mut = ring.header_ptr(slot);
                    let fence_ptr      = std::ptr::addr_of_mut!((*hdr_mut).completion_fence);
                    let force_norm_ptr = std::ptr::addr_of_mut!((*hdr_mut).force_norm);
                    std::ptr::write_volatile(force_norm_ptr, 0.0f32);
                    // Force fence write to land AFTER the force_norm reset
                    // so a racing GPU launch on this slot never observes
                    // fence=0 with a stale non-zero force_norm.
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                    std::ptr::write_volatile(fence_ptr, 0u32);
                }

                // ── M1.2.20.C-H / T23 — Channel-B Ghost drain ──
                // Poll the host-mapped ring counter.  For every slot at
                // least GHOST_SAFETY_LAG behind the live counter (kernel
                // is past it, write is committed), submit an io_uring
                // Write SQE for the 1408-byte record.  Bounded by
                // max_records to avoid out-of-buffer flushes.
                if let (Some(ghost), Some(_)) = (ghost_ring.as_ref(), ghost_state.as_ref()) {
                    let counter_now = ghost.n_frames_written().min(ghost.max_records);
                    let safe_target = counter_now.saturating_sub(GHOST_SAFETY_LAG);
                    while last_ghost_slot_emitted < safe_target {
                        let slot_g = last_ghost_slot_emitted;
                        let src_off = crate::ghost_tile::GhostTileRing::COUNTER_SECTOR_BYTES
                            + (slot_g as usize) * GHOST_RECORD_STRIDE;
                        let dst_off = ghost_records_written * GHOST_RECORD_STRIDE as u64;
                        let src_ptr = unsafe {
                            ghost.host_base.add(src_off)
                        } as *const u8;
                        let g_write = opcode::Write::new(
                                types::Fd(ghost_fd),
                                src_ptr,
                                GHOST_RECORD_STRIDE as u32,
                            )
                            .offset(dst_off as i64)
                            .build()
                            .user_data(0xB000_0000_0000_0000u64 | slot_g as u64);
                        let g_push = unsafe { uring.submission().push(&g_write) };
                        if g_push.is_err() {
                            // SQ full — break and let next outer iter retry.
                            break;
                        }
                        let g_submit = uring.submit_and_wait(1);
                        let g_result: i32 = match g_submit {
                            Ok(_) => uring.completion().next()
                                .map(|c| c.result()).unwrap_or(-1),
                            Err(_) => -1,
                        };
                        if g_result < 0 {
                            log::warn!(
                                "[AMS T23] ghost slot {} write failed cqe.result={}; \
                                 dropping",
                                slot_g, g_result
                            );
                            // Skip this slot but advance the cursor so we
                            // don't loop forever on a broken record.
                        } else {
                            ghost_records_written += 1;
                            ghost_bytes_written   += g_result as u64;
                        }
                        last_ghost_slot_emitted += 1;
                    }
                }

                frame_idx += 1;
            }

            // **M1.2.20.C-H / T23** — Drain any remaining Channel-B
            // slots that the in-loop polling missed because of the
            // GHOST_SAFETY_LAG.  The kernel has retired by this point
            // (engine teardown invoked stop+join) so every slot up
            // to counter-1 is committed.  Final flush to fsync.
            if let (Some(ghost), Some((_, _))) = (ghost_ring.as_ref(), ghost_state.as_ref()) {
                let counter_final = ghost.n_frames_written().min(ghost.max_records);
                while last_ghost_slot_emitted < counter_final {
                    let slot_g = last_ghost_slot_emitted;
                    let src_off = crate::ghost_tile::GhostTileRing::COUNTER_SECTOR_BYTES
                        + (slot_g as usize) * GHOST_RECORD_STRIDE;
                    let dst_off = ghost_records_written * GHOST_RECORD_STRIDE as u64;
                    let src_ptr = unsafe { ghost.host_base.add(src_off) } as *const u8;
                    let entry = opcode::Write::new(
                            types::Fd(ghost_fd), src_ptr, GHOST_RECORD_STRIDE as u32,
                        )
                        .offset(dst_off as i64)
                        .build()
                        .user_data(0xB000_0000_0000_0000u64 | slot_g as u64);
                    if unsafe { uring.submission().push(&entry) }.is_err() {
                        break;
                    }
                    let _ = uring.submit_and_wait(1);
                    if let Some(c) = uring.completion().next() {
                        let r = c.result();
                        if r > 0 {
                            ghost_records_written += 1;
                            ghost_bytes_written   += r as u64;
                        }
                    }
                    last_ghost_slot_emitted += 1;
                }
                if ghost_fd >= 0 {
                    unsafe { libc::fdatasync(ghost_fd); }
                }
                log::info!(
                    "[AMS T23] Channel-B io_uring drain summary: \
                     records_written={} bytes_written={} (~{} MB) ghost_path={:?}",
                    ghost_records_written, ghost_bytes_written,
                    ghost_bytes_written >> 20, ghost_output_path
                );
            }

            // fdatasync to ensure NVMe commits all sectors.
            unsafe { libc::fdatasync(fd); }

            log::info!(
                "[ZSTR consumer] exiting. frames_written={} dropped={} \
                 bytes={} MB  path={:?}",
                frames_written, frames_dropped,
                bytes_written >> 20,
                output_path
            );

            ZstrStats { frames_written, frames_dropped, bytes_written }
        })
        .expect("Failed to spawn ZSTR consumer thread")
}

// ─── ZSTR C-ABI Launcher FFI ────────────────────────────────────────────────
//
// C-ABI wrappers compiled in libzstr_kernels.a.  Called during the
// cuStreamBeginCapture window on `telemetry_stream`; each call records a
// kernel node into the in-progress CUgraph.
//
// `zstr_launch_pos_stage`  — launches zstr_pos_stage_f4_kernel (n_atoms × 3
//   floats, VRAM → pinned slot, vectorised float4 STG.NC).
// `zstr_launch_fence_signal` — launches zstr_signal_completion_kernel (1×1
//   thread; __threadfence_system() + atomicExch(fence, 1)).

pub mod ffi {
    use std::ffi::c_void;

    extern "C" {
        /// Path Z device-slot launcher: kernel reads `d_zstr_active_slot` at
        /// execution time. `base_pinned` is slot-0 header start;
        /// `inter_slot_stride` is bytes between slot bases (== frame_size);
        /// `pos_offset_in_slot` is the header-size offset to positions payload.
        pub fn zstr_launch_pos_stage(
            base_pinned:        *mut c_void,
            inter_slot_stride:  u32,
            pos_offset_in_slot: u32,
            src_vram:           *const c_void,
            n_atoms:            u32,
            stream:             *mut c_void,
        ) -> i32;

        /// Path Z device-slot launcher: fence signal. `base_fence` is slot-0
        /// completion_fence; kernel rolls slot via `d_zstr_active_slot`.
        pub fn zstr_launch_fence_signal(
            base_fence:        *mut c_void,
            inter_slot_stride: u32,
            stream:            *mut c_void,
        ) -> i32;

        /// Path Z host helper: stream-ordered update of d_zstr_active_slot.
        /// Caller invokes BEFORE each cuGraphLaunch on the same stream.
        pub fn prism_zstr_set_active_slot(slot: u32, stream: *mut c_void) -> i32;

        /// T11 — Action-Recovery force exfiltration (DMA + Σ‖F‖² atomic-add).
        ///
        /// Records ONE kernel node onto `stream`:
        /// `zstr_force_stage_f4_kernel` — float4 LDG.E.128 from `d_forces` →
        /// STG.E.128 to the pinned slot's force payload, plus per-warp
        /// `__shfl_down_sync` butterfly reduce of Σ Fᵢ² with the warp leader
        /// `atomicAdd`-ing directly into the active slot's `force_norm`
        /// field (offset 28). The field holds the running sum-of-squares
        /// at this point; `zstr_launch_force_norm_sqrt` finalises it.
        ///
        /// `base_pinned`              : slot-0 header start.
        /// `inter_slot_stride`        : `ZstrRing::frame_size`.
        /// `force_offset_in_slot`     : forces payload offset
        ///                              (== `ZstrRing::forces_offset_in_slot()`).
        /// `force_norm_offset_in_slot`: 28 (offset_of_force_norm).
        /// `src_d_forces`             : `d_forces` device pointer
        ///                              (NhsAmberFusedEngine).
        /// `n_atoms`                  : atom count.
        ///
        /// Caller contract: the pinned `force_norm` field MUST be 0.0f
        /// before this kernel atomic-adds. The Reaper resets it alongside
        /// `completion_fence` once it has read the slot; the initial state
        /// is zero from `ZstrRing::allocate`.
        pub fn zstr_launch_force_stage(
            base_pinned:                *mut c_void,
            inter_slot_stride:          u32,
            force_offset_in_slot:       u32,
            force_norm_offset_in_slot:  u32,
            src_d_forces:               *const c_void,
            n_atoms:                    u32,
            stream:                     *mut c_void,
        ) -> i32;

        /// T11 — Single-thread post-pass: in-place sqrtf of the active
        /// slot's `force_norm`, converting the accumulated Σ‖F‖² to
        /// ‖F‖₂ before the consumer reads it.  NaN propagates verbatim
        /// (host-side G29 Reaper traps non-finite values).
        pub fn zstr_launch_force_norm_sqrt(
            base_pinned:                *mut c_void,
            inter_slot_stride:          u32,
            force_norm_offset_in_slot:  u32,
            stream:                     *mut c_void,
        ) -> i32;
    }
}
