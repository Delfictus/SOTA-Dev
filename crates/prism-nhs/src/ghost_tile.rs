//! M1.2.19.B / Amendment 3.13 — Asynchronous Manifold Sequencer **Channel B**
//! (GhostTileFrame stream).
//!
//! The previous V2 telemetry pipeline persisted only **end-of-run snapshots**
//! (phasor_kcc_state.json, spatial_grid_state.json, etc.) — adequate for
//! steady-state diagnostics but useless for the per-frame causal/coherence
//! analysis the operator's Bilateral Phase-Coherence (Φ_sym) and
//! Flux-Coupling Efficiency (η) metrics require.  Channel B closes that gap:
//! every captured-graph replay where `adj.adjudication_code >= 1` (Construct
//! or Violation), the GPU pushes a self-describing `GhostTileFrame` header
//! followed by the 1280-byte `ContactShellTile` payload to a pinned-host,
//! device-mapped ring buffer.  The host reads it directly (zero DtoH copy)
//! on teardown and serializes the entire stream to
//! `{output_dir}/{stem}_ghost_tiles.bin`.
//!
//! ## Record layout (per active site per frame)
//!
//! ```text
//! offset  size  field
//! ──────  ────  ──────
//!     0   128   GhostTileFrame header  (this struct)
//!   128  1280   ContactShellTile        (so3_project::ContactShellTile)
//! ──────  ────
//!  total 1408   bytes per record
//! ```
//!
//! ## GhostTileFrame header (128 bytes, align(128))
//!
//! Operator spec (Amendment 3.13 §2.1):
//!
//! ```text
//! offset  size  field                  purpose
//! ──────  ────  ────────────────────   ──────────────────────────────────
//!     0     8   frame_idx u64          monotonic frame counter
//!     8     4   site_id   u32          cluster index within frame
//!    12     1   chain_id  u8           'A' (0x41) or 'B' (0x42), or 0
//!    13     1   adjudication_code u8   0=Prune, 1=Construct, 2=Violation
//!    14     2   _pad             [u8;2]  alignment
//!    16     4   kl_divergence f32      Σ_planes Δ_AB at this frame
//!    20    24   power_spectrum [f32;6] SO(3) C_l from geo plane (Φ_sym)
//!    44     8   thermo_flux    [f32;2] [wd_change, vib_energy]   (η)
//!    52     4   causal_lead_residue u32  driver_id for Lag Persistence
//!    56    72   _reserved      [u32;18] pads to 128 bytes; future fields
//! ```
//!
//! ## Why mapped pinned memory (operator §1.2 Mapped Memory Standard)
//!
//! The buffer is allocated via `cuMemHostAlloc(.., PORTABLE | DEVICEMAP)`,
//! which produces page-locked host memory addressable from both host and
//! GPU through a single virtual-address window.  The captured kernel
//! writes through the device-mapped pointer; the host reads through the
//! host pointer with no `cudaMemcpy` round-trip.  On Blackwell sm_120 +
//! PCIe Gen5 this lands as direct DMA from L2 to host memory.

#![cfg(feature = "gpu")]

use cudarc::driver::sys::{
    cuMemFreeHost, cuMemHostAlloc, cuMemHostGetDevicePointer_v2, CUresult,
    CU_MEMHOSTALLOC_DEVICEMAP, CU_MEMHOSTALLOC_PORTABLE,
};
use std::ffi::c_void;

// ─── GhostTileFrame — 128-byte aligned header ───────────────────────────────

/// Self-describing temporal/site tag preceding each exfiltrated
/// `ContactShellTile` in the Channel-B stream.
///
/// Layout pinned by `static_assert`s on the C++ mirror in
/// `cuda/ghost_tile_kernel.cuh` and the const-context offset asserts at
/// the bottom of this module.  Any drift breaks the offline replay.
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
pub struct GhostTileFrame {
    /// Monotonic frame counter — increments once per V2 captured-graph
    /// replay that fires Channel B (i.e., per chunk where
    /// adjudication_code >= 1 on at least one cluster).  Drives the
    /// time-axis for Φ_sym integrals.
    pub frame_idx: u64,

    /// Cluster index within the frame (0..n_clusters).  For 7C8R dimer
    /// with n_clusters=1, always 0.  Multi-cluster targets (KRAS, MYC)
    /// will produce one record per active cluster.
    pub site_id: u32,

    /// Chain identifier ASCII byte: 0x41='A', 0x42='B', or 0=unknown.
    /// Resolved from the cluster's centroid via the merged-topology
    /// chain map.  Zero in this commit (chain resolution is a follow-up
    /// — bilateral coherence Φ_sym needs it but the kernel doesn't yet
    /// do the lookup; site_id alone is enough to disambiguate the 7C8R
    /// chain pair given the symmetry).
    pub chain_id: u8,

    /// Mirror of the F1 SWITCH selector: 0=Prune, 1=Construct, 2=Violation.
    /// Filtered upstream — a record only exists when this is >= 1.
    /// Stored anyway so the offline replay can distinguish Construct
    /// (legitimate burst) from Violation (numerical trap).
    pub adjudication_code: u8,

    /// **Wave 1 / Q1+P4** — bitfield of per-record telemetry flags.
    /// Bit 0 (`CLASS_TAINTED` = 0x0001): set when the kernel substituted
    ///   a NaN/Inf value in `thermo_flux[0..2]` with 0.0 (the upstream
    ///   manifold has not yet populated the water-density derivative or
    ///   vibrational-energy slot).
    /// Bits 1..15 reserved for future quality flags (chain-resolution
    ///   confidence, KCC-stale, gear-id snapshot, etc.).
    /// Replaces the alignment-only `_pad: [u8; 2]` placeholder; offset
    /// and size unchanged so the on-disk layout is bit-for-bit
    /// equivalent for records produced before the flag was wired.
    pub telemetry_flags: u16,

    /// Σ_planes Δ_AB — total weighted KL divergence across all 4 planes
    /// at this frame.  Mirrors `adj.current_divergence` at the moment
    /// the adjudicator wrote the SWITCH code.
    pub kl_divergence: f32,

    /// Geometry-plane SO(3) power spectrum `C_l[0..6]` from the relaxed
    /// manifold.  Φ_sym integrates this against the Chain-B partner's
    /// spectrum across time.
    pub power_spectrum: [f32; 6],

    /// `[wd_change, vib_energy]` — water-density derivative + per-tile
    /// vibrational-energy snapshot.  Drives Flux-Coupling Efficiency
    /// η = ΔV_wd / ΔV_vib.  In this commit the kernel writes
    /// `[NaN, NaN]` until upstream telemetry buses surface them — gates
    /// the field as wired but unpopulated, so downstream code can
    /// recognize "not yet measured" vs "zero".
    pub thermo_flux: [f32; 2],

    /// Driver residue id from the KCC causal plane.  In this commit the
    /// kernel writes `u32::MAX` (sentinel for "not resolved at GPU
    /// time") since per-frame KCC argmax requires reading the phasor
    /// state which is not yet wired into the captured graph.  Stable
    /// FFI slot for the follow-up commit that adds it.
    pub causal_lead_residue: u32,

    /// Tail padding so the struct's total size is exactly 128 bytes.
    /// Reserved for future fields (Φ_sym phase-lock score,
    /// Γ_w desolvation tax, gear ID, hardware clock, etc.).
    pub _reserved: [u32; 18],
}

/// **Wave 1 / P4** — telemetry_flags bit definitions.
///
/// Bit 0 — `CLASS_TAINTED`: set when the Ghost stage kernel had to
/// substitute a NaN/Inf in `thermo_flux[0]` (water-density derivative)
/// or `thermo_flux[1]` (vibrational energy) with 0.0.  Indicates the
/// upstream manifold has not yet populated those slots; downstream
/// consumers (offline Φ_sym/η integrators) MUST exclude tainted records
/// from the η = ΔV_wd / ΔV_vib ratio computation.
pub const GHOST_TELEMETRY_CLASS_TAINTED: u16 = 0x0001;

impl GhostTileFrame {
    /// All-zero seed.  Matches the cuMemHostAlloc-zero'd buffer state;
    /// kernels overwrite these fields atomically per-record.
    pub const fn zero() -> Self {
        Self {
            frame_idx: 0,
            site_id: 0,
            chain_id: 0,
            adjudication_code: 0,
            telemetry_flags: 0,
            kl_divergence: 0.0,
            power_spectrum: [0.0; 6],
            thermo_flux: [0.0; 2],
            causal_lead_residue: 0,
            _reserved: [0u32; 18],
        }
    }
}

unsafe impl Send for GhostTileFrame {}
unsafe impl Sync for GhostTileFrame {}

// Compile-time layout invariants (operator spec §2.1).
const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<GhostTileFrame>() == 128);
    assert!(align_of::<GhostTileFrame>() == 128);
    assert!(offset_of!(GhostTileFrame, frame_idx) == 0);
    assert!(offset_of!(GhostTileFrame, site_id) == 8);
    assert!(offset_of!(GhostTileFrame, chain_id) == 12);
    assert!(offset_of!(GhostTileFrame, adjudication_code) == 13);
    assert!(offset_of!(GhostTileFrame, telemetry_flags) == 14);
    assert!(offset_of!(GhostTileFrame, kl_divergence) == 16);
    assert!(offset_of!(GhostTileFrame, power_spectrum) == 20);
    assert!(offset_of!(GhostTileFrame, thermo_flux) == 44);
    assert!(offset_of!(GhostTileFrame, causal_lead_residue) == 52);
    assert!(offset_of!(GhostTileFrame, _reserved) == 56);
};

/// Per-record byte size: 128 (header) + 1280 (ContactShellTile).
pub const GHOST_RECORD_BYTES: usize = 128 + 1280;

// ─── GhostTileRing — pinned-host, device-mapped ring buffer ─────────────────

/// Pinned-host buffer with a device-mapped alias.  The captured kernel
/// writes records through the device pointer; the host reads through the
/// host pointer with no DtoH copy.
///
/// Layout in the buffer:
/// ```text
///   [u32 n_frames_written]         offset 0..4
///   [u8  _pad[124]]                offset 4..128 (header sector)
///   [GhostTileFrame + ContactShellTile] × max_records  offset 128..end
/// ```
///
/// The 128-byte leading "header sector" reserves space for the atomic
/// `n_frames_written` counter that the GPU kernel atomicAdds and the
/// host reads on teardown.  Aligning the first record to offset 128
/// preserves Blackwell L1-sector alignment for the 1408-byte payload
/// records that follow.
pub struct GhostTileRing {
    /// Pinned-host base pointer (page-locked, mapped).  The first 128 B
    /// hold the `n_frames_written` u32 counter + padding; records start
    /// at offset 128.
    pub host_base: *mut u8,
    /// Device-mapped alias of `host_base` — kernels write through this.
    pub device_base: u64,
    /// Total bytes allocated (counter sector + max_records * 1408).
    pub total_bytes: usize,
    /// Maximum number of records the buffer can hold without overflow.
    /// Kernel bounds-checks against this before writing.
    pub max_records: u32,
}

unsafe impl Send for GhostTileRing {}
unsafe impl Sync for GhostTileRing {}

impl GhostTileRing {
    /// Reserved counter-sector size at the head of the buffer.  The
    /// first record starts at offset `COUNTER_SECTOR_BYTES`.
    pub const COUNTER_SECTOR_BYTES: usize = 128;

    /// Allocate a pinned-host, device-mapped ring sized for
    /// `max_records` records of 1408 bytes each.  All bytes zero on
    /// alloc (CUDA pinned-alloc behavior).
    pub fn allocate(max_records: u32) -> Result<Self, String> {
        let total_bytes = Self::COUNTER_SECTOR_BYTES
            + (max_records as usize) * GHOST_RECORD_BYTES;

        let mut host_ptr: *mut c_void = std::ptr::null_mut();
        let flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
        let rc = unsafe { cuMemHostAlloc(&mut host_ptr, total_bytes, flags) };
        if !matches!(rc, CUresult::CUDA_SUCCESS) {
            return Err(format!(
                "cuMemHostAlloc(GhostTileRing, {} B) failed: rc={:?}",
                total_bytes, rc
            ));
        }
        if host_ptr.is_null() {
            return Err("cuMemHostAlloc returned null".to_string());
        }

        // Get the device-side mapped alias.  On Blackwell + PCIe Gen5
        // this is the same VA window as the host pointer, so memcpy is
        // not required — kernel STG.E.* lands directly on host RAM.
        let mut dev_ptr: u64 = 0;
        let rc = unsafe {
            cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0)
        };
        if !matches!(rc, CUresult::CUDA_SUCCESS) {
            unsafe { let _ = cuMemFreeHost(host_ptr); }
            return Err(format!(
                "cuMemHostGetDevicePointer_v2 failed: rc={:?}", rc
            ));
        }

        // Zero the leading counter sector (cuMemHostAlloc may not
        // zero-init — be defensive).
        unsafe {
            std::ptr::write_bytes(host_ptr as *mut u8,
                                  0u8,
                                  Self::COUNTER_SECTOR_BYTES);
        }

        Ok(Self {
            host_base: host_ptr as *mut u8,
            device_base: dev_ptr,
            total_bytes,
            max_records,
        })
    }

    /// Read the GPU-written `n_frames_written` counter.  Volatile so the
    /// load isn't hoisted past the post-kernel synchronize.  Returns
    /// the count of (header + tile) records currently resident in the
    /// buffer.
    pub fn n_frames_written(&self) -> u32 {
        unsafe {
            std::ptr::read_volatile(self.host_base as *const u32)
        }
    }

    /// Byte slice of the live record payload (excluding the leading
    /// counter sector).  Length is clamped to `min(n_frames_written,
    /// max_records) * 1408`.
    pub fn payload_bytes(&self) -> &[u8] {
        let n = self.n_frames_written().min(self.max_records);
        let nbytes = (n as usize) * GHOST_RECORD_BYTES;
        unsafe {
            std::slice::from_raw_parts(
                self.host_base.add(Self::COUNTER_SECTOR_BYTES),
                nbytes,
            )
        }
    }
}

impl Drop for GhostTileRing {
    fn drop(&mut self) {
        if !self.host_base.is_null() {
            unsafe { let _ = cuMemFreeHost(self.host_base as *mut c_void); }
            self.host_base = std::ptr::null_mut();
        }
    }
}

// ─── FFI declarations for the C-side capture kernel ─────────────────────────

extern "C" {
    /// Captured-graph kernel that pushes one GhostTileFrame+ContactShellTile
    /// record to the ring when `adj.adjudication_code >= 1`.  Single-thread
    /// per cluster; n_clusters threads in one block.
    ///
    /// `ring_base_dev`: device-mapped pointer (from `GhostTileRing::device_base`)
    /// `tiles`:         baseline manifold pointer (n_clusters × ContactShellTile)
    /// `adj`:           adjudicator FFI struct
    /// `frame_idx`:     monotonic frame counter (host-supplied per launch)
    /// `n_clusters`:    cluster count
    /// `max_records`:   bounds check — kernel skips writes once counter >= this
    pub fn prism_ghost_pipe_stage_launch(
        ring_base_dev: u64,
        tiles: *const std::ffi::c_void,
        adj: *const std::ffi::c_void,
        d_kcc_lead: *const std::ffi::c_void,   // Wave 1 / Q2 — nullable
        frame_idx: u64,
        n_clusters: u32,
        max_records: u32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Wave 1 / Q1 — populator for the __constant__ d_cluster_to_repr_residue[64]
    /// table.  Stream-ordered cudaMemcpyToSymbolAsync.  `n` clamped to 64.
    pub fn prism_ghost_set_cluster_repr_residue(
        repr_residues_host: *const u32,
        n:                  u32,
        stream:             *mut std::ffi::c_void,
    ) -> i32;

    /// **M1.2.20.C-C / T20 + Amendment 3.20** — Topology-Driven Chain
    /// Boundary populator.  Host parses topology.chain_ids[] and
    /// produces a `[k]` cumulative-residue-boundary array where
    /// `offsets_host[i]` is the first residue id of chain `i`.
    /// Stream-ordered cudaMemcpyToSymbolAsync; `n` clamped to 8.
    /// Target-agnostic — replaces all prior hardcoded chain numbers.
    pub fn prism_ghost_set_chain_offsets(
        offsets_host: *const u32,
        n:            u32,
        stream:       *mut std::ffi::c_void,
    ) -> i32;
}

/// **M1.2.20.C-C / T20 + Amendment 3.20** — Public host wrapper for
/// the topology-driven chain offsets populator.  The underlying
/// `prism_ghost_set_chain_offsets` extern is declared at module-level
/// (not in a sub-module), so this just provides a documented Rust
/// signature for the bin call site.
///
/// # Safety
/// `offsets_host` must point to at least `n` valid `u32` values.
/// `stream` must be a valid CUstream owned by the active context.
pub unsafe fn set_chain_offsets(
    offsets_host: *const u32,
    n:            u32,
    stream:       *mut std::ffi::c_void,
) -> i32 {
    prism_ghost_set_chain_offsets(offsets_host, n, stream)
}
