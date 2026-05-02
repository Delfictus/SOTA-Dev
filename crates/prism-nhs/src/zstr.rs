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
    /// Pads the header to a full 4096-byte sector. align(4096) on the
    /// struct already enforces tail padding, but making it explicit
    /// here pins the slot layout so the offline replay format does not
    /// silently drift if the struct is later extended.
    pub _padding: [u32; 1016],           // 32..4096
}

// ZstrFrameHeader meaningful payload: 32 bytes.
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
        let pos_bytes    = n_atoms * 3 * 4;                         // n_atoms × xyz × f32
        let force_bytes  = n_atoms * 3 * 4;                         // T11: forces appended
        let raw_frame    = header_bytes + pos_bytes + force_bytes;
        // Round up to next 4096-byte multiple for O_DIRECT sector alignment.
        let frame_size   = (raw_frame + 4095) & !4095;
        let total_bytes  = frame_size * Self::N_SLOTS;

        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, total_bytes)
        };

        if !matches!(rc, cudarc::driver::sys::CUresult::CUDA_SUCCESS) {
            return Err(format!(
                "cuMemAllocHost_v2 failed: {:?} \
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
    /// Equals `sizeof(header) + n_atoms*3*4` — positions sit immediately
    /// after the header, forces immediately after positions.
    pub fn forces_offset_in_slot(&self) -> usize {
        std::mem::size_of::<ZstrFrameHeader>() + self.n_atoms * 3 * 4
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

// Compile-time pin: the header layout MUST match the offsets the C-side
// kernels and the offline replay parser depend on.  Drift here ⇒ silently
// corrupt force_norm / gear_id reads in the consumer thread.
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<ZstrFrameHeader>() == 4096);
    // We can't take addr_of on uninitialised memory in const, but we can
    // assert the field pack: 8+4+4+4+4+4+4 = 32 used bytes + 4064 padding.
    assert!(size_of::<u64>() + 6 * size_of::<u32>() + 1016 * size_of::<u32>() == 4096);
};

// ─── ZSTR Consumer Thread ────────────────────────────────────────────────────

/// Message passed from the ZSTR consumer thread to the caller when
/// the run completes.
pub struct ZstrStats {
    pub frames_written: u64,
    pub frames_dropped: u64,   // fence-timeout drops
    pub bytes_written:  u64,
}

/// Spawn the O_DIRECT ZSTR consumer thread.
///
/// The thread:
/// 1. Opens `output_path` with `O_DIRECT | O_CREAT | O_WRONLY`.
/// 2. For each frame epoch:
///    a. Computes `slot = frame_idx % 3`.
///    b. Spin-polls `ZstrFrameHeader::completion_fence` until `== 1`
///       (GPU has fenced all writes for this slot) or `stop` is set.
///    c. Writes `frame_size` bytes via `pwrite` (sector-aligned offset).
///    d. Resets `completion_fence` to 0 (returns slot to GPU).
///    e. Advances `frame_idx`.
/// 3. Flushes and closes the file when `stop` fires.
///
/// Returns a `JoinHandle<ZstrStats>`.
///
/// # Safety
/// `ring_ptr` must remain valid for the lifetime of the returned thread.
/// The caller is responsible for joining before dropping the ring.
pub fn spawn_zstr_consumer(
    ring: Arc<ZstrRing>,
    output_path: std::path::PathBuf,
    stop: Arc<AtomicBool>,
) -> JoinHandle<ZstrStats> {
    std::thread::Builder::new()
        .name(format!("zstr-consumer"))
        .spawn(move || {
            use std::os::unix::fs::OpenOptionsExt;

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

                // Slot is READY.  O_DIRECT pwrite at sector-aligned offset.
                let slot_ptr = unsafe {
                    ring.pinned_ptr.add(slot * frame_size)
                } as *const u8;
                let buf = unsafe { std::slice::from_raw_parts(slot_ptr, frame_size) };
                let offset = (frames_written * frame_size as u64) as i64;

                let written = unsafe {
                    libc::pwrite(
                        fd,
                        buf.as_ptr() as *const libc::c_void,
                        frame_size,
                        offset,
                    )
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
                    log::warn!(
                        "[ZSTR consumer] pwrite frame {} failed: errno={}",
                        frame_idx,
                        std::io::Error::last_os_error()
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

                frame_idx += 1;
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
