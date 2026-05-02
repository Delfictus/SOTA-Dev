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
/// Immediately followed in the same allocation by:
///   [positions: n_atoms * 3 * f32]
///   [padding to next 4096-byte boundary]
///
/// The full slot size is computed and padded to 4096 bytes so that
/// `O_DIRECT` writes are always sector-aligned.
#[repr(C, align(4096))]
pub struct ZstrFrameHeader {
    /// Monotonically increasing epoch counter (0-based).
    pub frame_idx: u64,
    /// Integration timestep in picoseconds at the time of capture.
    pub dt: f32,
    /// adjudication_code written by the InterferometricAdjudicator
    /// (0 = NOISE, 1 = BURST/F1, 2 = ONSET, 3 = WAIT_SENTINEL).
    pub adjudication_code: u32,
    /// 0 = Dirty/Writing (GPU owns).  1 = Ready (host may flush).
    /// Written by `zstr_signal_completion_kernel` after __threadfence_system().
    pub completion_fence: u32,
    /// Reserved — keeps total header size at 32 bytes.
    pub _padding: [u32; 3],
}

// ZstrFrameHeader meaningful payload: 32 bytes.
// size_of::<ZstrFrameHeader>() == 4096 (padded by align(4096)).
// The positions payload begins at the 4096-byte-aligned slot base,
// after the header struct, for a slot layout of:
//   [header: 4096 B] [positions: n_atoms*3*4 B] [pad→next 4096 multiple]

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
    /// Frame layout per slot:
    /// ```
    /// [ZstrFrameHeader: 32 B] [positions: n_atoms*3*4 B] [pad→4096]
    /// ```
    pub fn allocate(n_atoms: usize) -> Result<Self, String> {
        // ZstrFrameHeader is align(4096) → size_of = 4096 (padded by Rust).
        // Meaningful header data occupies only the first 32 bytes.
        let header_bytes = std::mem::size_of::<ZstrFrameHeader>(); // == 4096
        let pos_bytes    = n_atoms * 3 * 4;                         // n_atoms × xyz × f32
        let raw_frame    = header_bytes + pos_bytes;
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

                let slot = (frame_idx % 3) as usize;
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

                if written < 0 {
                    log::warn!(
                        "[ZSTR consumer] pwrite frame {} failed: errno={}",
                        frame_idx,
                        std::io::Error::last_os_error()
                    );
                    frames_dropped += 1;
                } else {
                    frames_written += 1;
                    bytes_written  += written as u64;

                    if frame_idx % 1000 == 0 {
                        log::info!(
                            "[ZSTR] frame={} adj_code={} fence_was_ready  \
                             written={} MB",
                            frame_idx,
                            hdr.adjudication_code,
                            bytes_written >> 20,
                        );
                    }
                }

                // Reset fence: return slot to GPU.
                // Use addr_of_mut! to avoid going through a shared reference
                // (&T → *mut T is UB per Rust's aliasing rules).
                unsafe {
                    let fence_ptr = std::ptr::addr_of_mut!(
                        (*ring.header_ptr(slot)).completion_fence
                    );
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

pub(crate) mod ffi {
    use std::ffi::c_void;

    extern "C" {
        /// C-ABI launcher: zstr_pos_stage_f4_kernel<<<ceil(n_atoms*3/4)/256, 256>>>.
        /// Returns cudaError_t cast to i32; 0 == success.
        pub fn zstr_launch_pos_stage(
            dst_pinned: *mut c_void,
            src_vram:   *const c_void,
            n_atoms:    u32,
            stream:     *mut c_void,
        ) -> i32;

        /// C-ABI launcher: zstr_signal_completion_kernel<<<1, 1>>>.
        /// Returns cudaError_t cast to i32; 0 == success.
        pub fn zstr_launch_fence_signal(
            slot_fence: *mut c_void,
            stream:     *mut c_void,
        ) -> i32;
    }
}
