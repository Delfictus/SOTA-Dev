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
//! v9D' (Amendment 3.14 / Sector-Lock): each record is a single 4096-byte
//! GhostTileFrame written via `O_DIRECT | O_DSYNC`.  The trailing 1280-byte
//! ContactShellTile body has been retired from the on-disk format — its
//! per-plane SO(3) spectra are now folded into the expanded 4-plane
//! `power_spectrum[24]` field of the frame itself.
//!
//! ```text
//! offset  size  field
//! ──────  ────  ──────
//!     0  4096   GhostTileFrame header  (this struct, sector-aligned)
//! ──────  ────
//!  total 4096   bytes per record  (= one NVMe physical sector)
//! ```
//!
//! ## GhostTileFrame header (4096 bytes, align(4096))
//!
//! Operator spec (Amendment 3.14 §1.1 — v9D' Sector-Lock):
//!
//! ```text
//! offset  size  field                  purpose
//! ──────  ────  ────────────────────   ──────────────────────────────────
//!     0     8   frame_idx u64          monotonic frame counter
//!     8     4   site_id   u32          cluster index within frame
//!    12     1   chain_id  u8           'A' (0x41) or 'B' (0x42), or 0
//!    13     1   adjudication_code u8   0=Prune, 1=Construct, 2=Violation
//!    14     2   telemetry_flags u16    bit-packed: LQI, T7, Burst, Drift
//!    16     4   kl_divergence f32      Σ_planes Δ_AB at this frame
//!    20    96   power_spectrum [f32;24]  4 planes × 6 bands (L=0..5)
//!   116     8   thermo_flux    [f32;2] [wd_change, vib_energy]   (η)
//!   124     4   causal_lead_residue u32  driver_id for Lag Persistence
//!   128   128   _reserved_payload [u32;32]  Pillar 5 expansion slab
//!   256  3840   _slack [u8; 3840]       4096-sector alignment pad
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

// ─── GhostTileFrame — 4096-byte aligned header (v9D' Sector-Lock) ────────────

/// Self-describing temporal/site tag for the Channel-B exfiltration
/// stream.  v9D' (Amendment 3.14): the frame is now sized to exactly one
/// NVMe physical sector (4096 bytes, align 4096) so that `O_DIRECT |
/// O_DSYNC` writes hit a unique sector per record without read-modify-
/// write penalty on the Samsung 990 PRO controller.
///
/// Layout pinned by `static_assert`s on the C++ mirror in
/// `cuda/ghost_tile_kernel.cuh` and the const-context offset asserts at
/// the bottom of this module.  Any drift breaks the offline replay.
#[repr(C, align(4096))]
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

    /// 4-plane SO(3) power spectrum: `C_l[0..6]` for each of
    /// {geometry, causal, thermo, chemistry} = 24 floats total.
    /// Layout: `power_spectrum[plane * 6 + l]` for `plane ∈ 0..4`,
    /// `l ∈ 0..6`.  Φ_sym integrates plane 0 against the Chain-B
    /// partner's spectrum across time; planes 1..3 feed the
    /// 4-plane interferometric KL.  Until the kernel populates
    /// planes 1..3 they are written as 0.0 (not NaN) — the geo
    /// plane is the only currently-wired source.
    pub power_spectrum: [f32; 24],

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

    /// Pillar 5 expansion slab — reserved for future per-record
    /// fields (Φ_sym phase-lock score, Γ_w desolvation tax, gear ID,
    /// hardware clock, ASC-steering work delta, etc.).  Zeroed by
    /// `cuMemHostAlloc`; kernels treat as inert until wired.
    pub _reserved_payload: [u32; 32],

    /// Sector-alignment slack: pads the struct to exactly 4096 bytes
    /// so each record consumes one NVMe physical sector under
    /// `O_DIRECT | O_DSYNC`.  Not a "dead space" — this is the
    /// resilience layer that enables zero read-modify-write
    /// exfiltration on PCIe Gen5.
    pub _slack: [u8; 3840],
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
            power_spectrum: [0.0; 24],
            thermo_flux: [0.0; 2],
            causal_lead_residue: 0,
            _reserved_payload: [0u32; 32],
            _slack: [0u8; 3840],
        }
    }
}

unsafe impl Send for GhostTileFrame {}
unsafe impl Sync for GhostTileFrame {}

// Compile-time layout invariants (operator spec Amendment 3.14 §1.1 — v9D').
const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<GhostTileFrame>() == 4096);
    assert!(align_of::<GhostTileFrame>() == 4096);
    assert!(offset_of!(GhostTileFrame, frame_idx) == 0);
    assert!(offset_of!(GhostTileFrame, site_id) == 8);
    assert!(offset_of!(GhostTileFrame, chain_id) == 12);
    assert!(offset_of!(GhostTileFrame, adjudication_code) == 13);
    assert!(offset_of!(GhostTileFrame, telemetry_flags) == 14);
    assert!(offset_of!(GhostTileFrame, kl_divergence) == 16);
    assert!(offset_of!(GhostTileFrame, power_spectrum) == 20);
    assert!(offset_of!(GhostTileFrame, thermo_flux) == 116);
    assert!(offset_of!(GhostTileFrame, causal_lead_residue) == 124);
    assert!(offset_of!(GhostTileFrame, _reserved_payload) == 128);
    assert!(offset_of!(GhostTileFrame, _slack) == 256);
};

/// Per-record byte size: 4096 (one sector, header-only — ContactShellTile
/// retired from on-disk format in v9D' / Amendment 3.14).
pub const GHOST_RECORD_BYTES: usize = 4096;

// ─── M1.2.23 §5 — GhostTileFrame v2 MAR payload schema overlay ──────────────
//
// The 128-byte `_reserved_payload [u32; 32]` region at offset 128 is partitioned
// into structured Transparent-MAR fields. Backward compatibility is preserved:
//
//   * v1 producer zeros the whole 128-byte region (existing kernel behavior)
//     → schema_version reads as 0 → consumers treat as legacy v1.
//
//   * v2 producer (Commit 4 wires this) writes schema_version=2 and populates
//     the structured fields.
//
// On-disk layout is BYTE-IDENTICAL between v1 and v2; the only producer-side
// change is what bytes get written into the reserved region. No struct-field
// rename. No size change. No alignment change. CUDA mirror remains a 4096-byte
// align(4096) struct with the same field list — the v2 fields are accessed
// via byte offsets / typed views below.
//
// Schema versioning policy:
//   * GHOST_FRAME_SCHEMA_V1_LEGACY = 0 (zeros — no v2 fields populated)
//   * GHOST_FRAME_SCHEMA_V2        = 2 (v2 fields populated by producer)
// schema_version=1 is RESERVED (never used) so a "1" reading is unambiguous
// drift / corruption and the scanner can flag it.
//
// Field layout within the 128-byte reserved region (offsets relative to
// the START of the GhostTileFrame, not relative to the reserved region):
//
//   offset 128  schema_version          u32   (0=v1 legacy, 2=v2 explicit)
//   offset 132  observation_pass        u8    (0/1 — bool)
//   offset 133  discovery_pass          u8    (0/1 — bool)
//   offset 134  perturbation_channel    u8    (UV bit code 0..3, or 0xFF unknown)
//   offset 135  _pad8                   u8
//   offset 136  uv_wavelength_nm        u16   (260, 280, 305, 320, or 0)
//   offset 138  field_completeness_flags u16
//   offset 140  gear_id                 u32   (Wave A default 0)
//   offset 144  dt_fs                   f32
//   offset 148  _pad32_for_step_align   u32   (slot reserved so step_idx u64 lands 8-aligned)
//   offset 152  step_idx                u64   (8-aligned — required for STG.E.64 stores)
//   offset 160  aabb_min                [f32; 3]   (160, 164, 168)
//   offset 172  aabb_max                [f32; 3]   (172, 176, 180)
//   offset 184  centroid_xyz            [f32; 3]   (184, 188, 192)
//   offset 196  _v2_reserved            [u32; 15]  (196..256, 60 B for future)
//
//   offset 256  _slack [u8; 3840]       (unchanged; ends at 4096)
//
// M1.2.24 fix: step_idx was at 148 in the v2 schema landed by Commit 3.
// 148 mod 8 = 4 → not 8-aligned → CUDA STG.E.64 traps with
// CUDA_ERROR_MISALIGNED_ADDRESS (deferred). The misaligned u64 store in
// the v2 kernel was the root cause of the M1.2.24 segfault. Swapping the
// pad and step_idx so step_idx lands at 152 (152 mod 8 = 0) fixes the
// alignment. No on-disk format compat issue exists because v2 records
// were never written (smoke aborted before any kernel completed).
//
// Total payload usage: 128 B (offsets 128..256). _slack and all v1 offsets
// are byte-identical to v1. CUDA mirror's _reserved_payload[32] still spans
// the same 128 bytes; the v2 typed view simply reinterprets that span.

pub const GHOST_FRAME_SCHEMA_V1_LEGACY: u32 = 0;
pub const GHOST_FRAME_SCHEMA_V2: u32 = 2;

pub const GHOST_V2_OFFSET_SCHEMA_VERSION:    usize = 128;
pub const GHOST_V2_OFFSET_OBSERVATION_PASS:  usize = 132;
pub const GHOST_V2_OFFSET_DISCOVERY_PASS:    usize = 133;
pub const GHOST_V2_OFFSET_PERTURBATION_CHAN: usize = 134;
pub const GHOST_V2_OFFSET_UV_WAVELENGTH_NM:  usize = 136;
pub const GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS: usize = 138;
pub const GHOST_V2_OFFSET_GEAR_ID:           usize = 140;
pub const GHOST_V2_OFFSET_DT_FS:             usize = 144;
pub const GHOST_V2_OFFSET_STEP_IDX:          usize = 152;
pub const GHOST_V2_OFFSET_AABB_MIN:          usize = 160;
pub const GHOST_V2_OFFSET_AABB_MAX:          usize = 172;
pub const GHOST_V2_OFFSET_CENTROID:          usize = 184;
pub const GHOST_V2_OFFSET_V2_RESERVED:       usize = 196;

/// **GHOST_NATIVE_SPATIAL_MAPPING_WIRE — field_completeness_flags bits.**
///
/// Bits 0-3 are populated by the v2 kernel directly (kl-finite, gear, dt,
/// not-tainted). Bit 4 is set when AABB / centroid spatial fields at
/// offsets 160 / 172 / 184 were natively populated from
/// `ContactShellTile.aabb_min/aabb_max` at write time. When Bit 4 == 0,
/// the spatial fields are sentinel-zero and must be treated as missing
/// by any downstream consumer; the host audit (probe / scanner) reads
/// this bit to distinguish native-populated from sentinel-zero records.
///
/// The selected centroid view at offset 184 is the AABB midpoint —
/// labelled as `aabb_midpoint_native_contact_shell_tile`. It is NOT a
/// phase-manifold complete centroid; richer views (spike_density,
/// kcc_driver, phasor_coherent, thermo_weighted, ghost_zstr_event_weighted)
/// are computed offline by the SiteManifest materializer.
pub const GHOST_FCF_BIT_KL_FINITE:             u16 = 0x0001;
pub const GHOST_FCF_BIT_GEAR_ID_NONZERO:       u16 = 0x0002;
pub const GHOST_FCF_BIT_DT_FS_POSITIVE:        u16 = 0x0004;
pub const GHOST_FCF_BIT_THERMO_NOT_TAINTED:    u16 = 0x0008;
pub const GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT: u16 = 0x0010;

// The v2 region MUST end at or before _slack (offset 256).
const _: () = {
    assert!(GHOST_V2_OFFSET_SCHEMA_VERSION    >= 128);
    assert!(GHOST_V2_OFFSET_OBSERVATION_PASS  == 132);
    assert!(GHOST_V2_OFFSET_DISCOVERY_PASS    == 133);
    assert!(GHOST_V2_OFFSET_PERTURBATION_CHAN == 134);
    assert!(GHOST_V2_OFFSET_UV_WAVELENGTH_NM  == 136);
    assert!(GHOST_V2_OFFSET_GEAR_ID           == 140);
    assert!(GHOST_V2_OFFSET_DT_FS             == 144);
    assert!(GHOST_V2_OFFSET_STEP_IDX          == 152);
    assert!(GHOST_V2_OFFSET_STEP_IDX % 8       == 0); // 8-aligned for STG.E.64
    assert!(GHOST_V2_OFFSET_AABB_MIN          == 160);
    assert!(GHOST_V2_OFFSET_AABB_MAX          == 172);
    assert!(GHOST_V2_OFFSET_CENTROID          == 184);
    assert!(GHOST_V2_OFFSET_V2_RESERVED       == 196);
    // v2 fields end at: 196 + 60 = 256 — must equal _slack offset.
    assert!(GHOST_V2_OFFSET_V2_RESERVED + 60   == 256);
    // Reserved-payload region begins at 128 and ends just before _slack at 256.
    assert!(GHOST_V2_OFFSET_SCHEMA_VERSION + 128 == 256);

    // ── M1.2.24 — natural-alignment audit ─────────────────────────────────
    // Every offset must be naturally aligned for its load/store width on
    // CUDA Blackwell (sm_120 issues STG.E.64/STG.E.32/STG.E.16 which trap
    // CUDA_ERROR_MISALIGNED_ADDRESS on misaligned destinations). Prior
    // step_idx misalignment at offset 148 was the M1.2.24 root cause.
    // This block is regression protection.
    assert!(GHOST_V2_OFFSET_SCHEMA_VERSION       % 4 == 0); // u32
    assert!(GHOST_V2_OFFSET_OBSERVATION_PASS     % 1 == 0); // u8
    assert!(GHOST_V2_OFFSET_DISCOVERY_PASS       % 1 == 0); // u8
    assert!(GHOST_V2_OFFSET_PERTURBATION_CHAN    % 1 == 0); // u8
    assert!(GHOST_V2_OFFSET_UV_WAVELENGTH_NM     % 2 == 0); // u16
    assert!(GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS % 2 == 0); // u16
    assert!(GHOST_V2_OFFSET_GEAR_ID              % 4 == 0); // u32
    assert!(GHOST_V2_OFFSET_DT_FS                % 4 == 0); // f32
    assert!(GHOST_V2_OFFSET_STEP_IDX             % 8 == 0); // u64 ← M1.2.24 fix
    assert!(GHOST_V2_OFFSET_AABB_MIN             % 4 == 0); // f32[3] component-wise
    assert!(GHOST_V2_OFFSET_AABB_MAX             % 4 == 0); // f32[3]
    assert!(GHOST_V2_OFFSET_CENTROID             % 4 == 0); // f32[3]
    assert!(GHOST_V2_OFFSET_V2_RESERVED          % 4 == 0); // u32 array
};

/// UV wavelength bit-code → wavelength_nm. Cited mapping authoritative source:
/// `crates/prism-nhs/src/interferometric_adjudicator.rs:712-735`
/// (`QI_SHIFT = 30` + `MU_01_SQ_LUT` index → wavelength).
pub const GHOST_UV_WAVELENGTH_NM_BY_BITCODE: [u16; 4] = [260, 280, 305, 320];

/// Sentinel for "perturbation channel unknown / not applicable to this record".
pub const GHOST_PERTURBATION_CHANNEL_UNKNOWN: u8 = 0xFF;

/// Read schema_version from a 4096-byte record. Returns the raw u32 — caller
/// matches against `GHOST_FRAME_SCHEMA_V1_LEGACY` / `GHOST_FRAME_SCHEMA_V2`.
#[inline]
pub fn ghost_v2_read_schema_version(record_bytes: &[u8; GHOST_RECORD_BYTES]) -> u32 {
    let off = GHOST_V2_OFFSET_SCHEMA_VERSION;
    let mut b = [0u8; 4];
    b.copy_from_slice(&record_bytes[off..off + 4]);
    u32::from_le_bytes(b)
}

// ─── GhostTileRing — pinned-host, device-mapped ring buffer ─────────────────

/// Pinned-host buffer with a device-mapped alias.  The captured kernel
/// writes records through the device pointer; the host reads through the
/// host pointer with no DtoH copy.
///
/// Layout in the buffer (v9D' / Amendment 3.14):
/// ```text
///   [u32 n_frames_written]         offset 0..4
///   [u8  _pad[4092]]               offset 4..4096 (counter sector)
///   [GhostTileFrame] × max_records offset 4096..end (4096 B per record)
/// ```
///
/// The 4096-byte leading counter sector reserves space for the atomic
/// `n_frames_written` counter that the GPU kernel atomicAdds and the
/// host reads on teardown.  Aligning the first record to offset 4096
/// gives every record a unique NVMe sector under `O_DIRECT | O_DSYNC`
/// (no read-modify-write penalty on PCIe Gen5).
pub struct GhostTileRing {
    /// Pinned-host base pointer (page-locked, mapped).  The first 128 B
    /// hold the `n_frames_written` u32 counter + padding; records start
    /// at offset 128.
    pub host_base: *mut u8,
    /// Device-mapped alias of `host_base` — kernels write through this.
    pub device_base: u64,
    /// Total bytes allocated (counter sector + max_records * 4096).
    pub total_bytes: usize,
    /// Maximum number of records the buffer can hold without overflow.
    /// Kernel bounds-checks against this before writing.
    pub max_records: u32,
}

unsafe impl Send for GhostTileRing {}
unsafe impl Sync for GhostTileRing {}

impl GhostTileRing {
    /// Reserved counter-sector size at the head of the buffer.  The
    /// first record starts at offset `COUNTER_SECTOR_BYTES`.  Sized to
    /// 4096 B in v9D' so the counter sector and the first record both
    /// land on aligned NVMe physical sectors when the buffer base is
    /// page-aligned (cuMemHostAlloc returns page-aligned pointers).
    pub const COUNTER_SECTOR_BYTES: usize = 4096;

    /// Allocate a pinned-host, device-mapped ring sized for
    /// `max_records` records of 4096 bytes each.  All bytes zero on
    /// alloc (CUDA pinned-alloc behavior).
    pub fn allocate(max_records: u32) -> Result<Self, String> {
        let total_bytes = Self::COUNTER_SECTOR_BYTES + (max_records as usize) * GHOST_RECORD_BYTES;

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
        let rc = unsafe { cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0) };
        if !matches!(rc, CUresult::CUDA_SUCCESS) {
            unsafe {
                let _ = cuMemFreeHost(host_ptr);
            }
            return Err(format!("cuMemHostGetDevicePointer_v2 failed: rc={:?}", rc));
        }

        // Zero the leading counter sector (cuMemHostAlloc may not
        // zero-init — be defensive).
        unsafe {
            std::ptr::write_bytes(host_ptr as *mut u8, 0u8, Self::COUNTER_SECTOR_BYTES);
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
        unsafe { std::ptr::read_volatile(self.host_base as *const u32) }
    }

    /// Byte slice of the live record payload (excluding the leading
    /// counter sector).  Length is clamped to `min(n_frames_written,
    /// max_records) * 4096`.
    pub fn payload_bytes(&self) -> &[u8] {
        let n = self.n_frames_written().min(self.max_records);
        let nbytes = (n as usize) * GHOST_RECORD_BYTES;
        unsafe {
            std::slice::from_raw_parts(self.host_base.add(Self::COUNTER_SECTOR_BYTES), nbytes)
        }
    }
}

impl Drop for GhostTileRing {
    fn drop(&mut self) {
        if !self.host_base.is_null() {
            unsafe {
                let _ = cuMemFreeHost(self.host_base as *mut c_void);
            }
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
    /// Reconciled with the canonical declaration in
    /// `captured_pipeline.rs:723` — adds the `firehose_enable: u32` final
    /// argument that matches the actual CUDA `extern "C"` signature in
    /// `crates/prism-nhs/src/cuda/ghost_tile_kernel.cu:233`. The previous
    /// 8-argument declaration triggered `clashing_extern_declarations`
    /// warnings at every full-tree compile.
    pub fn prism_ghost_pipe_stage_launch(
        ring_base_dev: u64,
        tiles: *const std::ffi::c_void,
        adj: *const std::ffi::c_void,
        d_kcc_lead: *const std::ffi::c_void, // Wave 1 / Q2 — nullable
        frame_idx: u64,
        n_clusters: u32,
        max_records: u32,
        stream: *mut std::ffi::c_void,
        firehose_enable: u32,
    ) -> i32;

    /// Wave 1 / Q1 — populator for the __constant__ d_cluster_to_repr_residue[64]
    /// table.  Stream-ordered cudaMemcpyToSymbolAsync.  `n` clamped to 64.
    pub fn prism_ghost_set_cluster_repr_residue(
        repr_residues_host: *const u32,
        n: u32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// **M1.2.20.C-C / T20 + Amendment 3.20** — Topology-Driven Chain
    /// Boundary populator.  Host parses topology.chain_ids[] and
    /// produces a `[k]` cumulative-residue-boundary array where
    /// `offsets_host[i]` is the first residue id of chain `i`.
    /// Stream-ordered cudaMemcpyToSymbolAsync; `n` clamped to 8.
    /// Target-agnostic — replaces all prior hardcoded chain numbers.
    pub fn prism_ghost_set_chain_offsets(
        offsets_host: *const u32,
        n: u32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// **Path Ω Phase 2** — geometry-emergent chain_id LUT populator.
    /// `chain_ids_host[i]` ∈ {0, 1, 0xFF} for cluster i.  Computed
    /// host-side from cluster centroids vs dyad axis at V2 build.
    /// Sentinel 0xFF disables the geometry-emergent path for that
    /// slot (kernel falls back to residue-id boundary scan).
    pub fn prism_ghost_set_cluster_chain_id(
        chain_ids_host: *const u8,
        n: u32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

/// **Path Ω Phase 2** — Public host wrapper for the geometry-emergent
/// chain_id LUT populator.
///
/// # Safety
/// `chain_ids_host` must point to at least `n` valid `u8` values.
/// `stream` must be a valid CUstream owned by the active context.
pub unsafe fn set_cluster_chain_id(
    chain_ids_host: *const u8,
    n: u32,
    stream: *mut std::ffi::c_void,
) -> i32 {
    prism_ghost_set_cluster_chain_id(chain_ids_host, n, stream)
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
    n: u32,
    stream: *mut std::ffi::c_void,
) -> i32 {
    prism_ghost_set_chain_offsets(offsets_host, n, stream)
}

/// **Wave 1 / Q1 — TIER 1 leak-fix (2026-05-03)** — Public host wrapper
/// for the cluster→representative-residue LUT populator.  Loads the
/// __constant__ d_cluster_to_repr_residue[64] slab consumed by the
/// ghost-tile kernel for cluster→residue lookup (lining residue
/// fallback when F2-pool d_kcc_lead is unavailable).  Without this
/// LUT being populated, every cluster slot reads its kernel default
/// (zero), so all per-cluster lining_residues records collapse onto
/// residue 0.
///
/// # Safety
/// `repr_residues_host` must point to at least `n` valid `u32` values.
/// `n` is clamped to 64 inside the FFI; values past 64 are silently
/// dropped.  `stream` must be a valid CUstream owned by the active
/// context.
pub unsafe fn set_cluster_repr_residue(
    repr_residues_host: *const u32,
    n: u32,
    stream: *mut std::ffi::c_void,
) -> i32 {
    prism_ghost_set_cluster_repr_residue(repr_residues_host, n, stream)
}
