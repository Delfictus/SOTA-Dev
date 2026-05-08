//! F2 — Stream-Ordered Memory Pool + VRAM Audit telemetry.
//!
//! Per Blackwell Convergence mandate §3 (operator directive
//! 2026-04-29). Replaces static "worst-case" pre-allocation with a
//! `cudaMemPool`-backed stream-ordered allocator (`cudaMallocFromPoolAsync`
//! / `cudaFreeAsync`) and a global VRAM telemetry struct that the F1
//! Adjudicator's SWITCH node consumes as the Case-2 Violation trigger.
//!
//! # Caller contract
//!
//! 1. Construct one [`VramPool`] per device. The Drop impl returns
//!    the pool handle to the driver via `cudaMemPoolDestroy`. With
//!    `ReleaseThreshold = UINT64_MAX` (set at create time) the
//!    driver does NOT return pool memory to the OS while the pool
//!    is alive — every captured-graph replay sees a stable mempool.
//! 2. Construct one [`VramAuditDevice`] per stream/replica. The
//!    Drop impl frees the device-side audit struct.
//! 3. For every allocation:
//!      a. `pool.alloc_async(size, stream)` → device pointer.
//!      b. IMMEDIATELY: `audit.record_alloc(size, budget, stream)`.
//!    The two kernel launches are stream-ordered, so the telemetry
//!    update is guaranteed to retire after the allocation node in
//!    any captured graph.
//! 4. For every free:
//!      a. `pool.free_async(ptr, stream)`.
//!      b. `audit.record_free(size, stream)`.
//! 5. The host reads the audit via [`VramAuditDevice::snapshot`]
//!    after `stream.synchronize()`. The F1 SWITCH node will read it
//!    via a captured-graph conditional-node update path (M1.2.5).
//!
//! # Concurrency model
//!
//! Every telemetry kernel runs as a single-thread launch
//! (`<<<1, 1, 0, stream>>>`). Single-threaded launches keep the
//! semantics race-free under concurrent invocations from different
//! streams: every atomic op on the audit struct is a single-
//! instruction memory transaction, so cross-stream concurrent
//! updates compose deterministically.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ============================================================================
// VramAudit POD struct (FFI-stable)
// ============================================================================

/// Mirror of the C-side `VramAudit` in `vram_pool.cuh`. Layout-pinned
/// at 24 bytes, 8-byte aligned. The trailing 4-byte padding (after
/// the u32 `pool_exhaustion_flag`) is implicit per Rust + C ABI rules
/// — both sides round size up to align-of-largest-field (8 B).
///
/// Three fields:
///
/// * [`current_allocated_bytes`] — sum of every `alloc_size` recorded
///   minus every `free_size`. The "live" working-set count.
/// * [`peak_high_water_mark`] — max value `current_allocated_bytes`
///   ever reached during the audit's lifetime. Monotonic; never
///   decrements on free.
/// * [`pool_exhaustion_flag`] — 0 = OK, 1 = ABORT. Set sticky-
///   irreversibly when any `record_alloc` would push
///   `current_allocated_bytes` above the supplied `budget`.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VramAudit {
    pub current_allocated_bytes: u64,
    pub peak_high_water_mark: u64,
    pub pool_exhaustion_flag: u32,
    /// Trailing padding to align the next array element to 8 bytes.
    /// Initialized to zero by the C-side `prism_vram_audit_zero_kernel`
    /// (which writes the whole struct, padding included, via
    /// individual field assignments) and never read.
    _pad: u32,
}

impl VramAudit {
    /// All-zeros initial state. Equivalent to `audit_init`'s effect
    /// on the device-side struct. Used by tests + as a sane default.
    pub const fn zero() -> Self {
        Self {
            current_allocated_bytes: 0,
            peak_high_water_mark: 0,
            pool_exhaustion_flag: 0,
            _pad: 0,
        }
    }

    /// Sentinel for `pool_exhaustion_flag`: pool budget intact.
    pub const FLAG_OK: u32 = 0;
    /// Sentinel for `pool_exhaustion_flag`: budget exceeded; routes
    /// the F1 SWITCH node to Case-2 Violation.
    pub const FLAG_VIOLATION: u32 = 1;

    /// True iff `pool_exhaustion_flag == FLAG_VIOLATION`.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.pool_exhaustion_flag == Self::FLAG_VIOLATION
    }
}

impl Default for VramAudit {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// FFI surface
// ============================================================================

#[cfg(feature = "gpu")]
#[allow(dead_code)]
mod ffi {
    use super::VramAudit;

    pub type CudaError = i32;
    pub const CUDA_SUCCESS: CudaError = 0;

    extern "C" {
        /// Sentinel `0xF2A4`. Confirms the static archive linked
        /// correctly and the FFI ABI is round-tripping.
        pub fn prism_vram_pool_link_probe() -> u32;

        pub fn prism_vram_pool_create(
            device_id: i32,
            out_pool: *mut *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_vram_pool_destroy(pool: *mut std::ffi::c_void) -> CudaError;

        pub fn prism_vram_pool_alloc_async(
            pool: *mut std::ffi::c_void,
            size: u64,
            stream: *mut std::ffi::c_void,
            out_ptr: *mut *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_vram_pool_free_async(
            ptr: *mut std::ffi::c_void,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_vram_audit_init(
            d_audit: *mut VramAudit,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_vram_audit_record_alloc(
            d_audit: *mut VramAudit,
            alloc_size: u64,
            budget: u64,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;

        pub fn prism_vram_audit_record_free(
            d_audit: *mut VramAudit,
            free_size: u64,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;
    }
}

/// Safe wrapper over the FFI link-probe. Returns `0xF2A4`.
#[cfg(feature = "gpu")]
pub fn link_probe() -> u32 {
    unsafe { ffi::prism_vram_pool_link_probe() }
}

// ============================================================================
// VramPool — RAII wrapper around `cudaMemPool_t`
// ============================================================================

/// RAII handle for a stream-ordered `cudaMemPool`.
///
/// On construction a fresh pool is created on the requested device
/// with the three Blackwell-required attributes:
///   - `cudaMemPoolAttrReleaseThreshold` = UINT64_MAX (driver never
///     returns pool memory to the OS during the pool's lifetime —
///     load-bearing for captured-graph stability).
///   - `cudaMemPoolAttrReuseAllowOpportunistic` = false.
///   - `cudaMemPoolAttrReuseAllowInternalDependencies` = false.
///
/// On Drop the pool is destroyed via `cudaMemPoolDestroy`. If a Drop
/// returns an error it is logged via `eprintln!` rather than panicking;
/// CUDA pool-destroy errors at process shutdown are typically benign
/// (e.g., already-shutdown context) and panicking here would
/// double-fault the program.
#[cfg(feature = "gpu")]
pub struct VramPool {
    handle: *mut std::ffi::c_void,
    device_id: i32,
}

#[cfg(feature = "gpu")]
unsafe impl Send for VramPool {}
// VramPool is NOT Sync: the underlying cudaMemPool_t is documented
// as thread-safe for alloc/free calls but the wrapper's Drop
// implementation forbids concurrent destroy attempts. Restrict to
// single-thread or `Arc<Mutex<VramPool>>` if cross-thread use is
// needed.

#[cfg(feature = "gpu")]
impl VramPool {
    /// Create a new pool on `device_id`. Returns the wrapper or a
    /// formatted error string.
    pub fn new(device_id: i32) -> Result<Self, String> {
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { ffi::prism_vram_pool_create(device_id, &mut handle) };
        if rc != ffi::CUDA_SUCCESS {
            return Err(format!(
                "prism_vram_pool_create failed: cudaError {} on device {}",
                rc, device_id
            ));
        }
        if handle.is_null() {
            return Err(format!(
                "prism_vram_pool_create returned cudaSuccess but null handle on device {}",
                device_id
            ));
        }
        Ok(Self { handle, device_id })
    }

    /// Raw pool handle as `usize`. Threaded into FFI calls that need
    /// the underlying `cudaMemPool_t`.
    #[inline]
    pub fn raw_handle(&self) -> usize {
        self.handle as usize
    }

    /// Device id this pool was created on.
    #[inline]
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Allocate `size` bytes from this pool on `stream`. Returns the
    /// raw device pointer as `usize`. Caller is responsible for
    /// pairing this with [`Self::free_async`] on a stream where the
    /// last user of the pointer has retired (stream-ordered free
    /// semantics — see CUDA Mempool docs).
    pub fn alloc_async(&self, size: u64, stream_handle: usize) -> Result<usize, String> {
        let mut out: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe {
            ffi::prism_vram_pool_alloc_async(
                self.handle,
                size,
                stream_handle as *mut std::ffi::c_void,
                &mut out,
            )
        };
        if rc != ffi::CUDA_SUCCESS {
            return Err(format!(
                "prism_vram_pool_alloc_async({} bytes) failed: cudaError {}",
                size, rc
            ));
        }
        if out.is_null() {
            return Err(format!(
                "prism_vram_pool_alloc_async({} bytes) returned cudaSuccess but null pointer",
                size
            ));
        }
        // Operator Amendment 3.9 §4 — F2 Pool 256-byte alignment guard.
        //
        // cudaMallocFromPoolAsync (the C++-side backing allocator) is
        // documented to return at least 256-byte aligned addresses.
        // Blackwell sm_120 LDG.E.128 / RED.E.ADD.V4 vector ops trap on
        // any sub-16-byte misalignment; cache-sector-aligned (256 B)
        // exceeds that floor.  This assertion catches any future
        // allocator regression (e.g., a sub-allocator that subdivides
        // a parent block on non-256-byte boundaries) at the point of
        // mint, before the pointer enters the captured graph and the
        // hardware traps deep inside a kernel launch.
        let ptr = out as usize;
        assert!(
            ptr % 256 == 0,
            "F2 POOL ALIGNMENT VIOLATION: alloc_async({} bytes) returned \
             0x{:x} which is not 256-byte aligned (mod 256 = {}). \
             Blackwell sm_120 vector loads will trap. HALT.",
            size,
            ptr,
            ptr % 256
        );
        Ok(ptr)
    }

    /// Free a pointer back to this pool on `stream`. Stream-ordered.
    pub fn free_async(&self, device_ptr: usize, stream_handle: usize) -> Result<(), String> {
        let rc = unsafe {
            ffi::prism_vram_pool_free_async(
                device_ptr as *mut std::ffi::c_void,
                stream_handle as *mut std::ffi::c_void,
            )
        };
        if rc != ffi::CUDA_SUCCESS {
            return Err(format!(
                "prism_vram_pool_free_async failed: cudaError {}",
                rc
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl Drop for VramPool {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        let rc = unsafe { ffi::prism_vram_pool_destroy(self.handle) };
        if rc != ffi::CUDA_SUCCESS {
            // Process-shutdown / context-already-destroyed errors
            // here are benign; panicking would mask the real cause
            // of an outer failure. Log and continue.
            eprintln!(
                "[VramPool::drop] prism_vram_pool_destroy returned cudaError {}",
                rc
            );
        }
        self.handle = std::ptr::null_mut();
    }
}

// ============================================================================
// VramAuditDevice — RAII wrapper around the device-side audit struct
// ============================================================================

/// Owns one device-side [`VramAudit`] struct (24 bytes) and the
/// stream telemetry kernels are launched on. The host reads via
/// [`Self::snapshot`] after `stream.synchronize()`.
#[cfg(feature = "gpu")]
pub struct VramAuditDevice {
    /// 24 bytes of device memory holding the [`VramAudit`] struct.
    /// Owned by this wrapper; freed on Drop.
    d_audit: cudarc::driver::CudaSlice<u8>,
    /// Stream the telemetry kernels are launched on. Held as a
    /// reference so the wrapper can launch additional updates later.
    stream: Arc<cudarc::driver::CudaStream>,
}

#[cfg(feature = "gpu")]
impl VramAuditDevice {
    /// Allocate the device-side audit struct on `stream` and zero
    /// every field. The zero pass is enqueued onto the same stream
    /// so it's stream-ordered with subsequent telemetry updates.
    pub fn new(stream: Arc<cudarc::driver::CudaStream>) -> Result<Self, String> {
        let d_audit = stream
            .alloc_zeros::<u8>(std::mem::size_of::<VramAudit>())
            .map_err(|e| format!("alloc d_audit: {:?}", e))?;
        // Explicit zero kernel — the alloc_zeros above already
        // zeroes the bytes via cudaMemset, but the explicit kernel
        // makes the "init" semantics observable on a stream and
        // ensures any future change to the audit struct that adds
        // padding-sensitive fields is exercised by the same code path.
        //
        // Scoped block: the cudarc `device_ptr` borrow guard
        // (`_g`) must drop before we move `d_audit` and `stream`
        // into `Self` at the end of this fn.
        {
            use cudarc::driver::DevicePtr;
            let (audit_dev, _g) = d_audit.device_ptr(&stream);
            let raw_stream = stream.cu_stream() as usize;
            let rc = unsafe {
                ffi::prism_vram_audit_init(
                    audit_dev as *mut VramAudit,
                    raw_stream as *mut std::ffi::c_void,
                )
            };
            if rc != ffi::CUDA_SUCCESS {
                return Err(format!("prism_vram_audit_init failed: cudaError {}", rc));
            }
        }
        Ok(Self { d_audit, stream })
    }

    /// Record a single allocation. Idempotent w.r.t. the
    /// pool_exhaustion_flag (atomicCAS sets it 0→1 once; subsequent
    /// allocations over budget are no-ops on the flag).
    pub fn record_alloc(&self, alloc_size: u64, budget: u64) -> Result<(), String> {
        use cudarc::driver::DevicePtr;
        let (audit_dev, _g) = self.d_audit.device_ptr(&self.stream);
        let raw_stream = self.stream.cu_stream() as usize;
        let rc = unsafe {
            ffi::prism_vram_audit_record_alloc(
                audit_dev as *mut VramAudit,
                alloc_size,
                budget,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        if rc != ffi::CUDA_SUCCESS {
            return Err(format!(
                "prism_vram_audit_record_alloc failed: cudaError {}",
                rc
            ));
        }
        Ok(())
    }

    /// Record a single free. Subtracts `free_size` from
    /// `current_allocated_bytes`; does not touch `peak_high_water_mark`
    /// or `pool_exhaustion_flag` (those are sticky high-water signals).
    pub fn record_free(&self, free_size: u64) -> Result<(), String> {
        use cudarc::driver::DevicePtr;
        let (audit_dev, _g) = self.d_audit.device_ptr(&self.stream);
        let raw_stream = self.stream.cu_stream() as usize;
        let rc = unsafe {
            ffi::prism_vram_audit_record_free(
                audit_dev as *mut VramAudit,
                free_size,
                raw_stream as *mut std::ffi::c_void,
            )
        };
        if rc != ffi::CUDA_SUCCESS {
            return Err(format!(
                "prism_vram_audit_record_free failed: cudaError {}",
                rc
            ));
        }
        Ok(())
    }

    /// Synchronize the stream, then dtoh-copy the device-side audit
    /// struct into a host-side [`VramAudit`]. Called by the F1
    /// Adjudicator's host-side decision point AND by tests.
    pub fn snapshot(&self) -> Result<VramAudit, String> {
        self.stream
            .synchronize()
            .map_err(|e| format!("stream sync: {:?}", e))?;
        let mut bytes = vec![0u8; std::mem::size_of::<VramAudit>()];
        self.stream
            .memcpy_dtoh(&self.d_audit, &mut bytes)
            .map_err(|e| format!("dtoh audit: {:?}", e))?;
        // SAFETY: we just dtoh'd exactly size_of::<VramAudit>() bytes
        // into a properly-aligned-via-Vec<u8> buffer, and VramAudit is
        // #[repr(C, align(8))] with no internal references. The
        // alignment guarantee from Vec<u8> is 1 byte; we copy through
        // an unaligned read via std::ptr::read_unaligned.
        let audit = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const VramAudit) };
        Ok(audit)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_audit_layout_is_24_bytes_8_aligned() {
        // The C-side static_assert pins sizeof(VramAudit) == 24 and
        // alignof == 8. The Rust mirror MUST match.
        assert_eq!(std::mem::size_of::<VramAudit>(), 24);
        assert_eq!(std::mem::align_of::<VramAudit>(), 8);
    }

    #[test]
    fn vram_audit_zero_default_state() {
        let a = VramAudit::zero();
        assert_eq!(a.current_allocated_bytes, 0);
        assert_eq!(a.peak_high_water_mark, 0);
        assert_eq!(a.pool_exhaustion_flag, VramAudit::FLAG_OK);
        assert!(!a.is_exhausted());
    }

    #[test]
    fn vram_audit_is_exhausted_returns_true_when_flag_set() {
        let mut a = VramAudit::zero();
        a.pool_exhaustion_flag = VramAudit::FLAG_VIOLATION;
        assert!(a.is_exhausted());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn link_probe_returns_sentinel() {
        // Confirms vram_pool.cu linked into the static archive and
        // the FFI ABI is round-tripping. Sentinel pinned at 0xF2A4.
        assert_eq!(super::link_probe(), 0x0000_F2A4);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn pool_create_destroy_round_trip() {
        use cudarc::driver::CudaContext;
        let _ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[vram_pool] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        // Create + Drop. The Drop impl calls destroy. If destroy
        // fails the test prints an eprintln but does not panic;
        // we assert only on the create path.
        let pool = VramPool::new(0).expect("pool create");
        assert_ne!(pool.raw_handle(), 0, "pool handle must be non-null");
        assert_eq!(pool.device_id(), 0);
        // Implicit drop here via end of scope — exercises destroy.
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn pool_alloc_free_round_trip() {
        use cudarc::driver::CudaContext;
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[vram_pool alloc] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream create");

        let pool = VramPool::new(0).expect("pool create");
        let stream_handle = stream.cu_stream() as usize;

        // Allocate 4 KiB — small enough not to OOM any device.
        let dev_ptr = pool.alloc_async(4096, stream_handle).expect("alloc 4 KiB");
        assert_ne!(dev_ptr, 0, "device pointer must be non-null");

        // Free + sync. The free is stream-ordered: returns
        // immediately, actual reclaim happens after stream catches up.
        pool.free_async(dev_ptr, stream_handle).expect("free");
        stream.synchronize().expect("stream sync");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn audit_telemetry_high_water_and_exhaustion_flag() {
        // ── F2 telemetry struct verification (operator's mandate ──
        // §3.2 audit gate).
        //
        // Scenario: budget = 5000 bytes. Record 5 allocations of
        // 1000 bytes each → current = 5000 (boundary, NOT
        // exhausted). Record 1 more (alloc_size = 1000) → current =
        // 6000 > budget, exhaustion flag set. peak = 6000.
        // Record 2 frees of 1000 each → current = 4000.
        // peak_high_water_mark stays at 6000 (sticky monotonic).
        // pool_exhaustion_flag stays at 1 (sticky).
        //
        // This exercises every field of the audit struct and every
        // telemetry kernel exactly once.

        use cudarc::driver::CudaContext;
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "[vram_audit telemetry] CUDA unavailable: {:?} — skipping",
                    e
                );
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream create");

        let audit = VramAuditDevice::new(stream.clone()).expect("audit new");

        // Initial snapshot: all zeros.
        let s0 = audit.snapshot().expect("snapshot 0");
        assert_eq!(
            s0,
            VramAudit::zero(),
            "audit init kernel did not zero every field"
        );

        // 5 record_alloc of 1000 bytes each, budget = 5000.
        let budget: u64 = 5000;
        for _ in 0..5 {
            audit.record_alloc(1000, budget).expect("record_alloc");
        }
        let s1 = audit.snapshot().expect("snapshot 1");
        assert_eq!(s1.current_allocated_bytes, 5000);
        assert_eq!(s1.peak_high_water_mark, 5000);
        assert_eq!(
            s1.pool_exhaustion_flag,
            VramAudit::FLAG_OK,
            "5000 bytes at boundary 5000 must NOT trip exhaustion (>, not >=)"
        );

        // 1 more record_alloc — pushes over budget.
        audit
            .record_alloc(1000, budget)
            .expect("record_alloc over budget");
        let s2 = audit.snapshot().expect("snapshot 2");
        assert_eq!(s2.current_allocated_bytes, 6000);
        assert_eq!(s2.peak_high_water_mark, 6000);
        assert_eq!(
            s2.pool_exhaustion_flag,
            VramAudit::FLAG_VIOLATION,
            "6000 > budget 5000 must trip exhaustion flag"
        );
        assert!(s2.is_exhausted());

        // 2 record_free of 1000 each — current drops, peak stays,
        // flag stays.
        audit.record_free(1000).expect("record_free 1");
        audit.record_free(1000).expect("record_free 2");
        let s3 = audit.snapshot().expect("snapshot 3");
        assert_eq!(
            s3.current_allocated_bytes, 4000,
            "current_allocated_bytes must decrement on record_free"
        );
        assert_eq!(
            s3.peak_high_water_mark, 6000,
            "peak_high_water_mark must NOT decrement on free (monotonic)"
        );
        assert_eq!(
            s3.pool_exhaustion_flag,
            VramAudit::FLAG_VIOLATION,
            "exhaustion flag must stay sticky after a free recovers headroom"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn pool_plus_audit_end_to_end() {
        // Full F2 path: pool alloc → telemetry record → pool free
        // → telemetry record. Pool budget intentionally LARGER than
        // the audit budget so the pool succeeds while the audit
        // detects the budget violation. The audit is the SOURCE OF
        // TRUTH for exhaustion, not the pool's own success/fail.
        use cudarc::driver::CudaContext;
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[vram pool+audit e2e] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream create");
        let pool = VramPool::new(0).expect("pool create");
        let audit = VramAuditDevice::new(stream.clone()).expect("audit new");
        let stream_handle = stream.cu_stream() as usize;

        let small_budget: u64 = 4096; // 4 KiB
        let alloc_size: u64 = 8192; // 8 KiB — over budget intentionally

        // Pool alloc succeeds (the device has plenty of memory) but
        // the audit budget is violated.
        let dev_ptr = pool
            .alloc_async(alloc_size, stream_handle)
            .expect("pool alloc");
        audit
            .record_alloc(alloc_size, small_budget)
            .expect("audit record_alloc");

        let s = audit.snapshot().expect("snapshot");
        assert_eq!(s.current_allocated_bytes, alloc_size);
        assert_eq!(s.peak_high_water_mark, alloc_size);
        assert!(
            s.is_exhausted(),
            "single 8 KiB alloc against 4 KiB budget must trip the flag"
        );

        // Cleanup.
        pool.free_async(dev_ptr, stream_handle).expect("pool free");
        audit.record_free(alloc_size).expect("audit record_free");
        stream.synchronize().expect("stream sync");
    }
}
