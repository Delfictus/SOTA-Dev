//! GhostPhaseLattice4D — physically-constrained spatiotemporal connected
//! components for PRISM Ghost v2 records.
//!
//! Replaces the legacy O(N²) point-cloud DBSCAN on `_stream{:02}_ghost_tiles.bin`
//! records with a 4-D phase lattice: nodes are bucketed by spatial cell ×
//! protocol phase × step bucket, and edges are adjudicated by:
//!
//!   * **Boolean constraints (hard fails)** — arrow of time, temporal adjacency,
//!     AABB overlap, monotone protocol-phase transition.
//!   * **Continuous physics scoring** — SO(3) cosine similarity weighted across
//!     the 4 GhostTileFrame planes (geometry / causality / thermo / chemistry),
//!     causal-driver continuity, KL-divergence smoothness.
//!
//! Edges with a composite score above `so3_threshold` (default 0.75) trigger a
//! union-find merge on the GPU; components survive into a per-site
//! [`PhaseManifold`] / [`ThermCcnsLifecycle`] / [`So3Manifold`] block on the
//! [`crate::site_manifest::SiteManifest`] structure.
//!
//! ## Why this is not "just a faster DBSCAN"
//!
//! Legacy DBSCAN flattened ~12 M raw spikes into a static 3-D grid, averaging
//! across kinematic phases (open / closed / burst) into one dead centroid and
//! merging spatial lobes (e.g. Y > 10 vs Y < 0) that share a global thermo
//! transition — anchoring the centroid in empty solvent. The 4-D lattice
//! preserves the temporal trajectory and refuses to merge across illegal
//! protocol-phase transitions or across the empty-solvent void.
//!
//! ## Provenance — Ghost v2 record schema
//!
//! Source-of-truth for the raw bytes: [`crate::ghost_tile`] (offsets at
//! `GHOST_V2_OFFSET_*`). This module never mutates the ring; it consumes a
//! borrowed slice of [`GhostTileFrame`] records (or a borrowed `&[u8]` payload
//! buffer) and projects each one to a [`GhostPhaseLatticeNode`].

#![cfg(feature = "gpu")]

use anyhow::{anyhow, bail, Context, Result};
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::ghost_tile::{
    GhostTileFrame, GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT, GHOST_FRAME_SCHEMA_V2,
    GHOST_RECORD_BYTES, GHOST_TELEMETRY_CLASS_TAINTED, GHOST_V2_OFFSET_AABB_MAX,
    GHOST_V2_OFFSET_AABB_MIN, GHOST_V2_OFFSET_CENTROID, GHOST_V2_OFFSET_DT_FS,
    GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS, GHOST_V2_OFFSET_GEAR_ID,
    GHOST_V2_OFFSET_PERTURBATION_CHAN, GHOST_V2_OFFSET_SCHEMA_VERSION, GHOST_V2_OFFSET_STEP_IDX,
};

// ─── Sentinels (mandatory by Part V — never zero-fill missing physical data) ──

/// Sentinel for "this node has no resolved causal lead residue". Mirrors the
/// kernel-level u32 sentinel written by the Ghost v2 producer when the per-frame
/// KCC argmax has not yet been wired into the captured graph.
pub const NODE_CAUSAL_LEAD_NONE: u32 = u32::MAX;

/// Sentinel for "this node has no resolved gear id". Distinguishes
/// "Wave A default 0" (which is a real gear) from "missing/unknown" emission.
pub const NODE_GEAR_ID_NONE: u32 = u32::MAX;

/// Sentinel for "ccns phase bin not provided by the upstream telemetry plane".
pub const NODE_CCNS_PHASE_NONE: u16 = 0xFFFF;

/// SO(3) plane bit flags — Bit 0=Geometry, 1=Causality, 2=Thermo, 3=Chemistry.
pub const SO3_PLANE_GEOMETRY: u8 = 0b0001;
pub const SO3_PLANE_CAUSALITY: u8 = 0b0010;
pub const SO3_PLANE_THERMO: u8 = 0b0100;
pub const SO3_PLANE_CHEMISTRY: u8 = 0b1000;

/// Default acceptance threshold for the composite physics score (SO(3) ×
/// causal × thermo). Edges below this threshold are rejected pre-union.
pub const DEFAULT_SO3_THRESHOLD: f32 = 0.75;

/// Default spatial cell size in Å. Picked to match the legacy 5.0 Å DBSCAN
/// epsilon so the lattice's neighborhood radius (3³ cell box) covers the same
/// physical neighborhood that the legacy backend admitted.
pub const DEFAULT_SPATIAL_CELL_A: f32 = 5.0;

/// Default temporal-edge cap (in MD steps). Edges that span more than this
/// number of steps are rejected — beyond ~500 steps the kinematic continuity
/// assumption fails for cryptic-pocket lifecycles.
pub const DEFAULT_MAX_TEMPORAL_EDGE_STEPS: u64 = 500;

/// Default bucket size for `step_bucket = floor(step_idx / bucket_size)`.
/// Mirrors the temporal-edge cap so a single bucket spans one max-edge window.
pub const DEFAULT_STEP_BUCKET_SIZE: u64 = 500;

/// Number of protocol phases recognised by the kernel.
pub const N_PROTOCOL_PHASES: u8 = 4;

/// Protocol-phase ordinals matching directive Part II.1.
pub const PHASE_COLD_HOLD: u8 = 0;
pub const PHASE_HEATING: u8 = 1;
pub const PHASE_WARM_HOLD: u8 = 2;
pub const PHASE_COOLING: u8 = 3;

// ─── Node ontology ──────────────────────────────────────────────────────────

/// A single Ghost v2 record projected into the 4-D phase lattice.
///
/// This struct is `#[repr(C)]` so the host-side packing matches the on-device
/// view passed to the edge kernel. Total size on x86_64 / sm_120 is 208 B
/// (the directive comment-counts of 16 / 12 / 24 / 20 / 96 are *logical*
/// groupings; the actual byte counts include `#[repr(C)]` padding).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GhostPhaseLatticeNode {
    // ── Identity & provenance ──
    /// Original index in the `_stream{:02}_ghost_tiles.bin` file.
    pub tile_index: u32,
    /// Source stream id (one stream per `--multi-stream` slot).
    pub stream_id: u16,
    /// `_pad0`: alignment slack so `site_id` lands 4-byte aligned.
    pub _pad0: u16,
    /// Cluster index within the producing frame. Mirrors
    /// [`GhostTileFrame::site_id`].
    pub site_id: u32,
    /// `_pad1`: alignment slack so `frame_idx` lands 8-byte aligned.
    pub _pad1: u32,
    /// Producer-side monotonic frame counter
    /// ([`GhostTileFrame::frame_idx`]).
    pub frame_idx: u64,
    /// Producer-side MD step index — Ghost v2 `step_idx` at offset 152.
    pub step_idx: u64,

    // ── Phase & state ──
    /// Protocol phase ordinal. 0=cold_hold, 1=heating, 2=warm_hold, 3=cooling.
    /// Resolved by the host from `step_idx` via the supplied
    /// [`PhaseSchedule`].
    pub protocol_phase: u8,
    /// `_pad2`: alignment slack so `step_bucket` lands 4-byte aligned.
    pub _pad2: [u8; 3],
    /// `floor(step_idx / step_bucket_size)`. Reused as a temporal coordinate
    /// in the lattice key.
    pub step_bucket: u32,
    /// CCNS phase bin (16-bit ordinal). [`NODE_CCNS_PHASE_NONE`] when the
    /// ccns telemetry plane has not yet been populated for this record.
    pub ccns_phase_bin: u16,
    /// `_pad3`: alignment slack so `gear_id` lands 4-byte aligned.
    pub _pad3: u16,
    /// Active gearbox gear at write time. [`NODE_GEAR_ID_NONE`] when the
    /// `gear_id` field of the v2 record is the Wave A default and the
    /// completeness flag bit is clear.
    pub gear_id: u32,

    // ── Spatial extent (Å) ──
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// AABB midpoint by default — labelled `centroid_xyz` here only as an
    /// alias. The phase-manifold-complete centroid is materialised offline
    /// by [`crate::site_manifest::PhaseManifold`] using AABB-volume-weighted
    /// aggregation across the constituent nodes.
    pub centroid_xyz: [f32; 3],

    // ── Thermodynamic & causal state ──
    /// `Σ_planes Δ_AB` total weighted KL divergence at this frame.
    /// Mirrors [`GhostTileFrame::kl_divergence`].
    pub kl_divergence: f32,
    /// `[wd_change, vib_energy]`. Either component may be `NaN` when the
    /// upstream manifold has not yet populated it; the kernel treats `NaN`
    /// as "unavailable_neutral" and defaults the multiplier to 1.0.
    pub thermo_flux: [f32; 2],
    /// Local water density. `NaN` when the prism-therm sidecar has not been
    /// written for this frame.
    pub water_density: f32,
    /// Driver residue id. [`NODE_CAUSAL_LEAD_NONE`] when unresolved.
    pub causal_lead_residue: u32,

    // ── SO(3) spherical manifold ──
    /// 4 planes × 6 bands of rotationally invariant SO(3) power spectrum
    /// `C_l[0..6]`. Plane order: 0=Geometry, 1=Causality, 2=Thermo,
    /// 3=Chemistry.
    pub so3_power_spectrum: [[f32; 6]; 4],
    /// Bitmask: which planes are populated (vs sentinel-zero).
    pub so3_plane_status: u8,
    /// `_pad4`: trailing alignment slack so the struct's total alignment
    /// matches its largest field (`u64`, 8-byte) and downstream
    /// `static_assert` size checks land on a stable footprint.
    pub _pad4: [u8; 7],
}

impl GhostPhaseLatticeNode {
    /// Construct an all-sentinel node — used by the host as a starting state
    /// before populating from a Ghost v2 record.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            tile_index: 0,
            stream_id: 0,
            _pad0: 0,
            site_id: 0,
            _pad1: 0,
            frame_idx: 0,
            step_idx: 0,
            protocol_phase: PHASE_COLD_HOLD,
            _pad2: [0; 3],
            step_bucket: 0,
            ccns_phase_bin: NODE_CCNS_PHASE_NONE,
            _pad3: 0,
            gear_id: NODE_GEAR_ID_NONE,
            aabb_min: [0.0; 3],
            aabb_max: [0.0; 3],
            centroid_xyz: [0.0; 3],
            kl_divergence: f32::NAN,
            thermo_flux: [f32::NAN, f32::NAN],
            water_density: f32::NAN,
            causal_lead_residue: NODE_CAUSAL_LEAD_NONE,
            so3_power_spectrum: [[0.0; 6]; 4],
            so3_plane_status: 0,
            _pad4: [0; 7],
        }
    }

    /// True when the node has *any* SO(3) plane populated (geometry plane is
    /// the minimum guarantee from the v2 producer).
    #[inline]
    pub fn has_any_so3_plane(&self) -> bool {
        self.so3_plane_status != 0
    }

    /// True when the AABB is non-degenerate (every component max >= min).
    #[inline]
    pub fn has_valid_aabb(&self) -> bool {
        self.aabb_max[0] >= self.aabb_min[0]
            && self.aabb_max[1] >= self.aabb_min[1]
            && self.aabb_max[2] >= self.aabb_min[2]
    }

    /// AABB volume in Å³.
    #[inline]
    pub fn aabb_volume(&self) -> f32 {
        let dx = (self.aabb_max[0] - self.aabb_min[0]).max(0.0);
        let dy = (self.aabb_max[1] - self.aabb_min[1]).max(0.0);
        let dz = (self.aabb_max[2] - self.aabb_min[2]).max(0.0);
        dx * dy * dz
    }
}

// Compile-time layout pin — protects the on-device kernel against silent host
// repacking. The C++ mirror in `cuda/ghost_lattice_kernel.cuh` carries an
// identical `static_assert` on its own struct.
const _: () = {
    use std::mem::{align_of, size_of};
    assert!(size_of::<GhostPhaseLatticeNode>() == 208);
    assert!(align_of::<GhostPhaseLatticeNode>() == 8);
};

/// 4-D lattice key used to bucket nodes for the edge kernel.
///
/// `(spatial_cell_x, spatial_cell_y, spatial_cell_z, protocol_phase,
/// step_bucket)` — the cartesian-product of the 3-D spatial cell and the 2-D
/// temporal coordinates (phase + step bucket). Two nodes can only share an
/// edge if their keys are within the configured neighborhood radius (default:
/// 1 spatial cell in each direction, ±1 step bucket, ±1 phase).
#[repr(C)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct GhostPhaseLatticeKey {
    pub spatial_cell_x: i32,
    pub spatial_cell_y: i32,
    pub spatial_cell_z: i32,
    pub protocol_phase: u8,
    pub _pad: [u8; 3],
    pub step_bucket: u32,
}

// ─── Phase schedule ─────────────────────────────────────────────────────────

/// Protocol-phase resolution policy. The kernel needs a `protocol_phase: u8`
/// per record; this trait is how the host computes it from the v2 record's
/// `step_idx` (and optionally `gear_id`).
pub trait PhaseSchedule: Send + Sync {
    /// Resolve `(step_idx, gear_id)` to a phase ordinal in `0..N_PROTOCOL_PHASES`.
    fn phase_for(&self, step_idx: u64, gear_id: u32) -> u8;
}

/// Quartile-based phase schedule — partitions the observed step-range into
/// four equal quartiles and labels them cold_hold / heating / warm_hold /
/// cooling in that order. Default fallback when the operator has not supplied
/// an explicit protocol schedule.
#[derive(Clone, Copy, Debug)]
pub struct QuartilePhaseSchedule {
    pub min_step: u64,
    pub max_step: u64,
}

impl PhaseSchedule for QuartilePhaseSchedule {
    #[inline]
    fn phase_for(&self, step_idx: u64, _gear_id: u32) -> u8 {
        if self.max_step <= self.min_step {
            return PHASE_COLD_HOLD;
        }
        let span = self.max_step - self.min_step;
        let pos = step_idx.saturating_sub(self.min_step);
        let q = ((pos as u128 * 4u128) / span as u128).min(3) as u8;
        q
    }
}

/// Explicit interval schedule — pass `[(start_step, end_step, phase_ordinal)]`
/// and any `step_idx` inside `[start, end)` resolves to the matching phase.
#[derive(Clone, Debug)]
pub struct IntervalPhaseSchedule {
    pub intervals: Vec<(u64, u64, u8)>,
    pub default_phase: u8,
}

impl PhaseSchedule for IntervalPhaseSchedule {
    fn phase_for(&self, step_idx: u64, _gear_id: u32) -> u8 {
        for &(s, e, p) in &self.intervals {
            if step_idx >= s && step_idx < e {
                return p.min(N_PROTOCOL_PHASES - 1);
            }
        }
        self.default_phase.min(N_PROTOCOL_PHASES - 1)
    }
}

// ─── Ghost v2 → lattice node projection ─────────────────────────────────────

/// Build a [`GhostPhaseLatticeNode`] from a single 4096-byte Ghost v2 record.
///
/// The `record` slice must be exactly [`GHOST_RECORD_BYTES`] long. Returns
/// `None` if the record is `schema_version != 2` (legacy v1 records do not
/// carry the AABB / step_idx fields the lattice needs and must be filtered
/// out upstream).
pub fn project_ghost_v2_record(
    record: &[u8],
    tile_index: u32,
    stream_id: u16,
    schedule: &dyn PhaseSchedule,
    step_bucket_size: u64,
) -> Option<GhostPhaseLatticeNode> {
    if record.len() != GHOST_RECORD_BYTES {
        return None;
    }

    // Schema gate — only v2 records carry AABB + step_idx.
    let schema_version = u32::from_le_bytes(
        record[GHOST_V2_OFFSET_SCHEMA_VERSION..GHOST_V2_OFFSET_SCHEMA_VERSION + 4]
            .try_into()
            .ok()?,
    );
    if schema_version != GHOST_FRAME_SCHEMA_V2 {
        return None;
    }

    let mut n = GhostPhaseLatticeNode::empty();

    // ── v1 header fields ──
    n.tile_index = tile_index;
    n.stream_id = stream_id;
    n.frame_idx = u64::from_le_bytes(record[0..8].try_into().ok()?);
    n.site_id = u32::from_le_bytes(record[8..12].try_into().ok()?);
    let telemetry_flags = u16::from_le_bytes(record[14..16].try_into().ok()?);
    n.kl_divergence = f32::from_le_bytes(record[16..20].try_into().ok()?);

    // 4-plane SO(3) power spectrum at offset 20: 24 floats × 4 B each.
    let mut spectrum = [[0.0f32; 6]; 4];
    let mut plane_status: u8 = 0;
    for plane in 0..4 {
        let mut populated_in_plane = false;
        for band in 0..6 {
            let off = 20 + (plane * 6 + band) * 4;
            let v = f32::from_le_bytes(record[off..off + 4].try_into().ok()?);
            spectrum[plane][band] = v;
            if v.is_finite() && v != 0.0 {
                populated_in_plane = true;
            }
        }
        if populated_in_plane {
            plane_status |= 1u8 << plane;
        }
    }
    n.so3_power_spectrum = spectrum;
    n.so3_plane_status = plane_status;

    // thermo_flux at offset 116. NaN-tainted records keep the NaN through —
    // the kernel handles missing thermo as "unavailable_neutral" (multiplier
    // 1.0) per directive Part V.
    let tf0 = f32::from_le_bytes(record[116..120].try_into().ok()?);
    let tf1 = f32::from_le_bytes(record[120..124].try_into().ok()?);
    let class_tainted = (telemetry_flags & GHOST_TELEMETRY_CLASS_TAINTED) != 0;
    n.thermo_flux = if class_tainted { [f32::NAN, f32::NAN] } else { [tf0, tf1] };
    n.water_density = f32::NAN; // populated downstream from the prism-therm
                                // sidecar; v2 records don't carry it.

    n.causal_lead_residue = u32::from_le_bytes(record[124..128].try_into().ok()?);
    if n.causal_lead_residue == 0 || n.causal_lead_residue == NODE_CAUSAL_LEAD_NONE {
        n.causal_lead_residue = NODE_CAUSAL_LEAD_NONE;
    }

    // ── v2 fields ──
    let dt_fs = f32::from_le_bytes(
        record[GHOST_V2_OFFSET_DT_FS..GHOST_V2_OFFSET_DT_FS + 4]
            .try_into()
            .ok()?,
    );
    let _ = dt_fs; // available but not currently used in adjudication

    let gear_id = u32::from_le_bytes(
        record[GHOST_V2_OFFSET_GEAR_ID..GHOST_V2_OFFSET_GEAR_ID + 4]
            .try_into()
            .ok()?,
    );
    n.gear_id = if gear_id == 0 { NODE_GEAR_ID_NONE } else { gear_id };

    let perturbation = record[GHOST_V2_OFFSET_PERTURBATION_CHAN];
    let _ = perturbation;

    n.step_idx = u64::from_le_bytes(
        record[GHOST_V2_OFFSET_STEP_IDX..GHOST_V2_OFFSET_STEP_IDX + 8]
            .try_into()
            .ok()?,
    );

    // AABB only valid when the producer set the SPATIAL_NATIVE bit. Sentinel-
    // zero AABBs collapse to a degenerate point AABB at the origin which the
    // kernel's overlap test will then never accept — exactly the directive's
    // "must not zero-fill missing physical data" rule.
    let fcf = u16::from_le_bytes(
        record[GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS
            ..GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS + 2]
            .try_into()
            .ok()?,
    );
    if (fcf & GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT) != 0 {
        for i in 0..3 {
            let off_min = GHOST_V2_OFFSET_AABB_MIN + i * 4;
            let off_max = GHOST_V2_OFFSET_AABB_MAX + i * 4;
            let off_cen = GHOST_V2_OFFSET_CENTROID + i * 4;
            n.aabb_min[i] = f32::from_le_bytes(record[off_min..off_min + 4].try_into().ok()?);
            n.aabb_max[i] = f32::from_le_bytes(record[off_max..off_max + 4].try_into().ok()?);
            n.centroid_xyz[i] = f32::from_le_bytes(record[off_cen..off_cen + 4].try_into().ok()?);
        }
    } else {
        // No spatial — the lattice cannot place this record. Caller must
        // discard nodes for which `has_valid_aabb` returns false (the
        // empty AABB will collapse to a zero-volume point that fails every
        // overlap test, but pre-filtering on the host avoids burning a
        // cell key on an unplaceable record).
        return None;
    }

    // ── Phase resolution ──
    n.protocol_phase = schedule.phase_for(n.step_idx, n.gear_id);
    n.step_bucket = (n.step_idx / step_bucket_size.max(1)) as u32;

    Some(n)
}

/// Project an entire `_stream{:02}_ghost_tiles.bin` payload (excluding the
/// 4096 B counter sector) into a vector of lattice nodes. Records that fail
/// projection (legacy v1, sentinel AABB, malformed) are skipped — the
/// returned vector contains only placeable nodes.
pub fn project_ghost_v2_payload(
    payload: &[u8],
    stream_id: u16,
    schedule: &dyn PhaseSchedule,
    step_bucket_size: u64,
) -> Vec<GhostPhaseLatticeNode> {
    let n_records = payload.len() / GHOST_RECORD_BYTES;
    let mut out = Vec::with_capacity(n_records);
    for i in 0..n_records {
        let off = i * GHOST_RECORD_BYTES;
        let record = &payload[off..off + GHOST_RECORD_BYTES];
        if let Some(node) =
            project_ghost_v2_record(record, i as u32, stream_id, schedule, step_bucket_size)
        {
            out.push(node);
        }
    }
    out
}

// ─── Backend ────────────────────────────────────────────────────────────────

/// Configuration for [`GhostPhaseLattice4D`]. All fields are physical
/// parameters that the directive Part III specifies as backend defaults; the
/// operator overrides them via CLI flags `--ghost-phase-lattice-*`.
#[derive(Clone, Copy, Debug)]
pub struct GhostPhaseLatticeConfig {
    pub spatial_cell_size_a: f32,
    pub max_temporal_edge_steps: u64,
    pub step_bucket_size: u64,
    pub so3_threshold: f32,
}

impl Default for GhostPhaseLatticeConfig {
    fn default() -> Self {
        Self {
            spatial_cell_size_a: DEFAULT_SPATIAL_CELL_A,
            max_temporal_edge_steps: DEFAULT_MAX_TEMPORAL_EDGE_STEPS,
            step_bucket_size: DEFAULT_STEP_BUCKET_SIZE,
            so3_threshold: DEFAULT_SO3_THRESHOLD,
        }
    }
}

/// One connected component returned by the backend, carrying the indices of
/// the constituent [`GhostPhaseLatticeNode`]s plus aggregate edge telemetry
/// for the [`So3Manifold`] block.
#[derive(Clone, Debug)]
pub struct GhostPhaseLatticeComponent {
    /// Indices into the input slice — every node in this group passed the
    /// boolean constraints + scoring threshold against at least one other
    /// node in the group (or is a singleton).
    pub node_indices: Vec<u32>,
    /// Number of accepted edges that fell *inside* this component
    /// (intra-component edge count). Used by the [`So3Manifold`] block to
    /// compute the mean-cosine summary.
    pub n_intra_edges: u32,
    /// Sum of `total_edge_score` over accepted intra-component edges. The
    /// mean cosine emitted into `so3_manifold.intra_component_mean_cosine`
    /// is `intra_edge_score_sum / max(1, n_intra_edges)`.
    pub intra_edge_score_sum: f64,
}

/// Backend-level summary aggregating edge-construction telemetry. Surfaced
/// in the per-site `ghost_phase_lattice` provenance block (directive 4.1).
#[derive(Clone, Debug, Default)]
pub struct GhostPhaseLatticeRunStats {
    pub n_nodes: u32,
    pub n_lattice_cells: u32,
    pub n_pairs_evaluated: u64,
    pub n_pairs_phase_legal: u64,
    pub n_pairs_aabb_overlap: u64,
    pub n_directed_edges: u64,
    pub kernel_elapsed_ms: f64,
    pub host_elapsed_ms: f64,
    pub min_step_idx: u64,
    pub max_step_idx: u64,
    pub phases_present: u8, // bitmask of phases that appeared in the input
}

/// Outcome of a [`GhostPhaseLattice4D::cluster`] call.
#[derive(Clone, Debug)]
pub struct GhostPhaseLatticeOutcome {
    pub components: Vec<GhostPhaseLatticeComponent>,
    pub stats: GhostPhaseLatticeRunStats,
    pub config: GhostPhaseLatticeConfig,
}

/// CUDA source for the lattice's host-side fallback path. The primary kernel
/// lives in the static-archived `ghost_lattice_kernel.cu` (compiled by
/// `build.rs`); this NVRTC-compiled module is the no-static-archive
/// alternative, used when the build was configured without the
/// `ghost_lattice` archive (e.g. when bisecting).
const NVRTC_SRC: &str = include_str!("cuda/ghost_lattice_kernel_nvrtc.cu");

/// FFI surface to the static-archived host orchestrator
/// (`prism_ghost_phase_lattice_run`). Defined in
/// `cuda/ghost_lattice_kernel.cu`.
extern "C" {
    /// Run the full edge-adjudication + union-find + path-compression
    /// pipeline on the device. Returns 0 on success, non-zero on a kernel
    /// launch / sync failure (the integer value mirrors a `cudaError_t`).
    ///
    /// All pointers are device pointers obtained via cudarc allocations.
    /// `nodes_dev` points to `n_nodes * sizeof(GhostPhaseLatticeNode)`
    /// bytes; the kernel reads it but does not mutate it.
    /// `cell_first_dev` / `cell_count_dev` / `permutation_dev` are the
    /// host-built lattice index tables (one entry per non-empty cell;
    /// permutation maps cell-sorted slot → original node index).
    /// `parent_dev` is the union-find parent array (n_nodes ints).
    /// `edge_score_sum_dev` and `edge_count_dev` are 1-element buffers
    /// for atomicAdd-accumulated intra-component edge telemetry.
    pub fn prism_ghost_phase_lattice_run(
        nodes_dev: u64,
        n_nodes: u32,
        permutation_dev: u64,
        cell_first_dev: u64,
        cell_count_dev: u64,
        n_cells: u32,
        cell_table_dev: u64, // packed [cx,cy,cz,phase,step_bucket] per cell
        parent_dev: u64,
        edge_score_sum_dev: u64,
        edge_count_dev: u64,
        pair_count_dev: u64,
        phase_legal_count_dev: u64,
        aabb_overlap_count_dev: u64,
        spatial_cell_size_a: f32,
        max_temporal_edge_steps: u64,
        so3_threshold: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

/// GPU-resident edge-adjudication backend. Implements the directive's Part II
/// boolean + continuous physics scoring. Constructed once per engine; the
/// CUDA module is compiled lazily on the first `cluster` call.
pub struct GhostPhaseLattice4D {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    config: GhostPhaseLatticeConfig,
    /// NVRTC-compiled fallback module — used only when the static archive
    /// FFI symbol isn't reachable (test-only bisection path). On the
    /// canonical build path the kernel comes from
    /// `cuda/ghost_lattice_kernel.cu` via `compile_to_static_archive`.
    nvrtc_module: Option<NvrtcKernels>,
}

struct NvrtcKernels {
    _module: Arc<CudaModule>,
    f_init_parent: CudaFunction,
    f_edge_kernel: CudaFunction,
    f_path_compress: CudaFunction,
}

impl GhostPhaseLattice4D {
    /// Construct the backend. The CUDA module is compiled lazily on the first
    /// `cluster` call (NVRTC path) — the static-archive path resolves
    /// directly via the extern "C" symbol with no compilation required.
    pub fn new(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        config: GhostPhaseLatticeConfig,
    ) -> Self {
        Self {
            context,
            stream,
            config,
            nvrtc_module: None,
        }
    }

    /// Compile the NVRTC fallback once. Idempotent.
    fn ensure_nvrtc(&mut self) -> Result<()> {
        if self.nvrtc_module.is_some() {
            return Ok(());
        }
        log::info!(
            "  [GHOST-LATTICE-4D] nvrtc-compiling fallback edge kernel ({} bytes)",
            NVRTC_SRC.len()
        );
        let t0 = std::time::Instant::now();
        let ptx = compile_ptx(NVRTC_SRC).map_err(|e| {
            anyhow!(
                "nvrtc compile of ghost_phase_lattice fallback kernel failed: {:?}",
                e
            )
        })?;
        let module = self
            .context
            .load_module(ptx)
            .context("load lattice fallback PTX")?;
        let f_init_parent = module
            .load_function("ghost_lattice_init_parent")
            .context("load ghost_lattice_init_parent")?;
        let f_edge_kernel = module
            .load_function("ghost_lattice_edge_kernel")
            .context("load ghost_lattice_edge_kernel")?;
        let f_path_compress = module
            .load_function("ghost_lattice_path_compress")
            .context("load ghost_lattice_path_compress")?;
        log::info!(
            "  [GHOST-LATTICE-4D] fallback kernel ready in {} ms",
            t0.elapsed().as_millis()
        );
        self.nvrtc_module = Some(NvrtcKernels {
            _module: module,
            f_init_parent,
            f_edge_kernel,
            f_path_compress,
        });
        Ok(())
    }

    /// Run the full pipeline against `nodes`. Empty input returns an empty
    /// outcome (no kernel launches).
    pub fn cluster(&mut self, nodes: &[GhostPhaseLatticeNode]) -> Result<GhostPhaseLatticeOutcome> {
        let t_total = std::time::Instant::now();
        if nodes.is_empty() {
            return Ok(GhostPhaseLatticeOutcome {
                components: Vec::new(),
                stats: GhostPhaseLatticeRunStats::default(),
                config: self.config,
            });
        }

        // ── Step 1. Build host-side lattice cell tables ──────────────
        let t_host = std::time::Instant::now();
        let (permutation, cell_first, cell_count, cell_table, span) =
            self.build_lattice_cells(nodes);
        let n_nodes = nodes.len() as u32;
        let n_cells = cell_first.len() as u32;
        log::info!(
            "  [GHOST-LATTICE-4D] lattice build: nodes={} cells={} step_span=[{}, {}] | {} ms",
            n_nodes,
            n_cells,
            span.min_step,
            span.max_step,
            t_host.elapsed().as_millis()
        );

        // ── Step 2. Upload to device ─────────────────────────────────
        let t_upload = std::time::Instant::now();
        let nodes_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                nodes.as_ptr() as *const u8,
                nodes.len() * std::mem::size_of::<GhostPhaseLatticeNode>(),
            )
        };
        let d_nodes: CudaSlice<u8> = self
            .stream
            .memcpy_stod(nodes_bytes)
            .context("upload lattice nodes")?;
        let d_permutation: CudaSlice<u32> = self
            .stream
            .memcpy_stod(&permutation)
            .context("upload permutation")?;
        let d_cell_first: CudaSlice<u32> = self
            .stream
            .memcpy_stod(&cell_first)
            .context("upload cell_first")?;
        let d_cell_count: CudaSlice<u32> = self
            .stream
            .memcpy_stod(&cell_count)
            .context("upload cell_count")?;
        let d_cell_table: CudaSlice<i32> = self
            .stream
            .memcpy_stod(&cell_table)
            .context("upload cell_table")?;
        let mut d_parent: CudaSlice<i32> = unsafe {
            self.stream
                .alloc::<i32>(nodes.len())
                .context("alloc parent")?
        };
        let mut d_edge_score_sum: CudaSlice<u64> =
            self.stream.memcpy_stod(&[0u64; 1]).context("alloc score sum")?;
        let mut d_edge_count: CudaSlice<u64> =
            self.stream.memcpy_stod(&[0u64; 1]).context("alloc edge count")?;
        let mut d_pair_count: CudaSlice<u64> =
            self.stream.memcpy_stod(&[0u64; 1]).context("alloc pair count")?;
        let mut d_phase_legal: CudaSlice<u64> = self
            .stream
            .memcpy_stod(&[0u64; 1])
            .context("alloc phase legal count")?;
        let mut d_aabb_overlap: CudaSlice<u64> = self
            .stream
            .memcpy_stod(&[0u64; 1])
            .context("alloc aabb overlap count")?;
        log::info!(
            "  [GHOST-LATTICE-4D] device upload: {} ms ({} KB nodes + {} KB cells)",
            t_upload.elapsed().as_millis(),
            nodes_bytes.len() / 1024,
            (cell_first.len() + cell_count.len() + cell_table.len() / 5) * 4 / 1024
        );

        // ── Step 3. Launch kernels (try static-archive FFI first; fall
        //            back to NVRTC if the symbol isn't linked) ────────
        let t_kernel = std::time::Instant::now();
        let static_rc = {
            // Each `.device_ptr(stream)` call returns a (raw_ptr, guard)
            // tuple where the guard pins the slice's lifetime against the
            // stream's outstanding work. Hold every guard in scope until
            // after the FFI call returns so cudarc doesn't reclaim the
            // allocation while the kernel is mid-launch.
            let (p_nodes, _g_nodes) = d_nodes.device_ptr(&self.stream);
            let (p_perm, _g_perm) = d_permutation.device_ptr(&self.stream);
            let (p_cf, _g_cf) = d_cell_first.device_ptr(&self.stream);
            let (p_cc, _g_cc) = d_cell_count.device_ptr(&self.stream);
            let (p_ct, _g_ct) = d_cell_table.device_ptr(&self.stream);
            let (p_par, _g_par) = d_parent.device_ptr_mut(&self.stream);
            let (p_score, _g_score) = d_edge_score_sum.device_ptr_mut(&self.stream);
            let (p_ec, _g_ec) = d_edge_count.device_ptr_mut(&self.stream);
            let (p_pc, _g_pc) = d_pair_count.device_ptr_mut(&self.stream);
            let (p_pl, _g_pl) = d_phase_legal.device_ptr_mut(&self.stream);
            let (p_ao, _g_ao) = d_aabb_overlap.device_ptr_mut(&self.stream);
            let raw_stream = self.stream.cu_stream() as *mut std::ffi::c_void;
            unsafe {
                prism_ghost_phase_lattice_run(
                    p_nodes,
                    n_nodes,
                    p_perm,
                    p_cf,
                    p_cc,
                    n_cells,
                    p_ct,
                    p_par,
                    p_score,
                    p_ec,
                    p_pc,
                    p_pl,
                    p_ao,
                    self.config.spatial_cell_size_a,
                    self.config.max_temporal_edge_steps,
                    self.config.so3_threshold,
                    raw_stream,
                )
            }
        };

        if static_rc != 0 {
            log::warn!(
                "  [GHOST-LATTICE-4D] static-archive run returned rc={} — falling back to NVRTC",
                static_rc
            );
            self.ensure_nvrtc()?;
            self.run_via_nvrtc(
                &d_nodes,
                n_nodes,
                &d_permutation,
                &d_cell_first,
                &d_cell_count,
                n_cells,
                &d_cell_table,
                &mut d_parent,
                &mut d_edge_score_sum,
                &mut d_edge_count,
                &mut d_pair_count,
                &mut d_phase_legal,
                &mut d_aabb_overlap,
            )?;
        }

        self.stream.synchronize().context("sync lattice kernels")?;
        let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "  [GHOST-LATTICE-4D] kernel: {:.2} ms",
            kernel_ms
        );

        // ── Step 4. Download parent + edge telemetry ────────────────
        let t_dl = std::time::Instant::now();
        let mut parent_host: Vec<i32> = vec![0; nodes.len()];
        self.stream
            .memcpy_dtoh(&d_parent, &mut parent_host)
            .context("download parent")?;
        let mut edge_score_sum_host = [0u64; 1];
        let mut edge_count_host = [0u64; 1];
        let mut pair_count_host = [0u64; 1];
        let mut phase_legal_host = [0u64; 1];
        let mut aabb_overlap_host = [0u64; 1];
        self.stream
            .memcpy_dtoh(&d_edge_score_sum, &mut edge_score_sum_host)?;
        self.stream
            .memcpy_dtoh(&d_edge_count, &mut edge_count_host)?;
        self.stream
            .memcpy_dtoh(&d_pair_count, &mut pair_count_host)?;
        self.stream
            .memcpy_dtoh(&d_phase_legal, &mut phase_legal_host)?;
        self.stream
            .memcpy_dtoh(&d_aabb_overlap, &mut aabb_overlap_host)?;
        log::info!(
            "  [GHOST-LATTICE-4D] download: {} ms",
            t_dl.elapsed().as_millis()
        );

        // ── Step 5. Component extraction (root → indices map) ───────
        let t_extract = std::time::Instant::now();
        let mut roots: std::collections::HashMap<i32, Vec<u32>> =
            std::collections::HashMap::new();
        for (i, &p) in parent_host.iter().enumerate() {
            // The path-compress kernel is supposed to land every node on its
            // canonical root; defend against a partial compression by
            // walking once on host.
            let mut r = p;
            while parent_host[r as usize] != r {
                r = parent_host[r as usize];
            }
            roots.entry(r).or_default().push(i as u32);
        }

        // Total decoded edge telemetry — kernel atomically accumulated the
        // score-sum × 1e6 (u64 fixed-point) so we recover the float here.
        let total_edges = edge_count_host[0];
        let total_score = edge_score_sum_host[0] as f64 / 1.0e6;

        // Per-component edge counts — derived from the parent array. The
        // exact intra-edge count requires re-running the predicate on host
        // because the kernel writes into a global accumulator. Apportion
        // the global telemetry by component size² (proportional to the
        // expected pair count under uniform edge probability) — this is a
        // *deterministic* allocation, not a measurement, and the JSON
        // emitter calls it out as `intra_component_mean_cosine_estimate`.
        let total_pair_estimator: u64 = roots
            .values()
            .map(|v| {
                let n = v.len() as u64;
                if n < 2 {
                    0
                } else {
                    n * (n - 1) / 2
                }
            })
            .sum();
        let mut components: Vec<GhostPhaseLatticeComponent> = roots
            .into_values()
            .map(|node_indices| {
                let n = node_indices.len() as u64;
                let intra_pair_estimate = if n < 2 { 0 } else { n * (n - 1) / 2 };
                let alloc_frac = if total_pair_estimator > 0 {
                    intra_pair_estimate as f64 / total_pair_estimator as f64
                } else {
                    0.0
                };
                let n_intra_edges = (total_edges as f64 * alloc_frac).round() as u32;
                let intra_edge_score_sum = total_score * alloc_frac;
                GhostPhaseLatticeComponent {
                    node_indices,
                    n_intra_edges,
                    intra_edge_score_sum,
                }
            })
            .collect();
        components.sort_by(|a, b| b.node_indices.len().cmp(&a.node_indices.len()));
        log::info!(
            "  [GHOST-LATTICE-4D] components: {} (largest={}) | extract {} ms",
            components.len(),
            components.first().map(|c| c.node_indices.len()).unwrap_or(0),
            t_extract.elapsed().as_millis()
        );

        let host_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        let mut phases_present: u8 = 0;
        for n in nodes {
            phases_present |= 1u8 << n.protocol_phase.min(7);
        }

        let stats = GhostPhaseLatticeRunStats {
            n_nodes,
            n_lattice_cells: n_cells,
            n_pairs_evaluated: pair_count_host[0],
            n_pairs_phase_legal: phase_legal_host[0],
            n_pairs_aabb_overlap: aabb_overlap_host[0],
            n_directed_edges: total_edges,
            kernel_elapsed_ms: kernel_ms,
            host_elapsed_ms: host_ms,
            min_step_idx: span.min_step,
            max_step_idx: span.max_step,
            phases_present,
        };

        log::info!(
            "POST_MD_CLUSTER_GHOST_LATTICE_4D_DONE components={} pairs={} edges={} | {:.2} ms total",
            components.len(),
            stats.n_pairs_evaluated,
            stats.n_directed_edges,
            host_ms
        );

        Ok(GhostPhaseLatticeOutcome {
            components,
            stats,
            config: self.config,
        })
    }

    /// Per-cell index table. Sorts nodes by lattice key so the kernel can
    /// iterate one cell's nodes contiguously, then builds a 27-cell × 4-phase
    /// × 3-step-bucket neighbour list per query.
    fn build_lattice_cells(
        &self,
        nodes: &[GhostPhaseLatticeNode],
    ) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<i32>, StepSpan) {
        let cell_size = self.config.spatial_cell_size_a.max(1e-3);

        // Step span (also used for stats).
        let mut min_step = u64::MAX;
        let mut max_step = 0u64;
        for n in nodes {
            min_step = min_step.min(n.step_idx);
            max_step = max_step.max(n.step_idx);
        }
        if min_step == u64::MAX {
            min_step = 0;
        }
        let span = StepSpan { min_step, max_step };

        // Project each node to its lattice key. The kernel's neighborhood is
        // ±1 cell in (x, y, z), ±1 phase, ±1 step bucket. Spatial cells use
        // a signed integer floor; the host packs them into a stable 64-bit
        // composite so the sort drops contiguous cells next to each other.
        let mut keys: Vec<(u64, u32)> = Vec::with_capacity(nodes.len());
        for (i, n) in nodes.iter().enumerate() {
            let cx = (n.centroid_xyz[0] / cell_size).floor() as i32;
            let cy = (n.centroid_xyz[1] / cell_size).floor() as i32;
            let cz = (n.centroid_xyz[2] / cell_size).floor() as i32;
            // Pack i32 cell coords into u16 ranges biased by 32_768 so the
            // composite key is monotone under sort. For a 1000 Å spatial
            // box at 5 Å cells, |cx| < 200 — well inside the 16-bit range.
            let bias = 32_768i32;
            let kx = ((cx + bias).clamp(0, 65_535)) as u64;
            let ky = ((cy + bias).clamp(0, 65_535)) as u64;
            let kz = ((cz + bias).clamp(0, 65_535)) as u64;
            // Composite: phase (2 bits) | step_bucket (16 bits) | kx (16) |
            // ky (16) | kz (14). Fits in 64 bits and is monotone in the
            // tuple order (phase, step_bucket, x, y, z) which is what the
            // kernel iterates.
            let phase = (n.protocol_phase & 0b11) as u64;
            let bucket = (n.step_bucket & 0xFFFF) as u64;
            let composite = (phase << 62)
                | (bucket << 46)
                | (kx << 30)
                | (ky << 14)
                | (kz & 0x3FFF);
            keys.push((composite, i as u32));
        }
        keys.sort_by_key(|&(k, _)| k);

        // Build cell tables: one entry per *unique* composite key.
        let mut cell_first: Vec<u32> = Vec::new();
        let mut cell_count: Vec<u32> = Vec::new();
        let mut cell_table: Vec<i32> = Vec::new();
        let mut permutation: Vec<u32> = Vec::with_capacity(nodes.len());

        let mut cur_key: Option<u64> = None;
        let mut run_start: u32 = 0;
        for (slot, &(k, idx)) in keys.iter().enumerate() {
            permutation.push(idx);
            match cur_key {
                Some(ck) if ck == k => {}
                _ => {
                    if let Some(_ck) = cur_key {
                        cell_count.push(slot as u32 - run_start);
                    }
                    // Decode the composite back into (cx, cy, cz, phase, bucket)
                    // for the kernel's neighbour walk.
                    let phase = ((k >> 62) & 0b11) as i32;
                    let bucket = ((k >> 46) & 0xFFFF) as i32;
                    let kx = ((k >> 30) & 0xFFFF) as i32;
                    let ky = ((k >> 14) & 0xFFFF) as i32;
                    let kz = (k & 0x3FFF) as i32;
                    let cx = kx - 32_768;
                    let cy = ky - 32_768;
                    let cz = kz - 32_768;
                    cell_table.extend_from_slice(&[cx, cy, cz, phase, bucket]);
                    cell_first.push(slot as u32);
                    cur_key = Some(k);
                    run_start = slot as u32;
                }
            }
        }
        if cur_key.is_some() {
            cell_count.push(keys.len() as u32 - run_start);
        }

        (permutation, cell_first, cell_count, cell_table, span)
    }

    /// NVRTC fallback launch sequence — only used when the static-archive
    /// host orchestrator is unreachable (test bisection). Matches the
    /// static-archive's behavior byte-for-byte.
    #[allow(clippy::too_many_arguments)]
    fn run_via_nvrtc(
        &mut self,
        d_nodes: &CudaSlice<u8>,
        n_nodes: u32,
        d_permutation: &CudaSlice<u32>,
        d_cell_first: &CudaSlice<u32>,
        d_cell_count: &CudaSlice<u32>,
        n_cells: u32,
        d_cell_table: &CudaSlice<i32>,
        d_parent: &mut CudaSlice<i32>,
        d_edge_score_sum: &mut CudaSlice<u64>,
        d_edge_count: &mut CudaSlice<u64>,
        d_pair_count: &mut CudaSlice<u64>,
        d_phase_legal: &mut CudaSlice<u64>,
        d_aabb_overlap: &mut CudaSlice<u64>,
    ) -> Result<()> {
        let kernels = self
            .nvrtc_module
            .as_ref()
            .ok_or_else(|| anyhow!("NVRTC fallback not initialised"))?;

        let block = 128u32;
        let grid_n = (n_nodes + block - 1) / block;
        let grid_cells = (n_cells + block - 1) / block;

        let cfg_n = LaunchConfig {
            grid_dim: (grid_n, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let cfg_cells = LaunchConfig {
            grid_dim: (grid_cells, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        // Re-borrow each `&mut CudaSlice<T>` parameter on every launch —
        // cudarc's `LaunchBuilder::arg` consumes its argument by value, so
        // passing the same `&mut CudaSlice` directly across multiple
        // launches would produce a "use of moved value" error. The
        // `&mut *d_parent` form creates a fresh reborrow per launch with
        // the same lifetime as the function-parameter borrow.
        unsafe {
            self.stream
                .launch_builder(&kernels.f_init_parent)
                .arg(&mut *d_parent)
                .arg(&n_nodes)
                .launch(cfg_n.clone())
        }
        .context("launch ghost_lattice_init_parent")?;

        unsafe {
            self.stream
                .launch_builder(&kernels.f_edge_kernel)
                .arg(&*d_nodes)
                .arg(&n_nodes)
                .arg(&*d_permutation)
                .arg(&*d_cell_first)
                .arg(&*d_cell_count)
                .arg(&n_cells)
                .arg(&*d_cell_table)
                .arg(&mut *d_parent)
                .arg(&mut *d_edge_score_sum)
                .arg(&mut *d_edge_count)
                .arg(&mut *d_pair_count)
                .arg(&mut *d_phase_legal)
                .arg(&mut *d_aabb_overlap)
                .arg(&self.config.spatial_cell_size_a)
                .arg(&self.config.max_temporal_edge_steps)
                .arg(&self.config.so3_threshold)
                .launch(cfg_cells.clone())
        }
        .context("launch ghost_lattice_edge_kernel")?;

        unsafe {
            self.stream
                .launch_builder(&kernels.f_path_compress)
                .arg(&mut *d_parent)
                .arg(&n_nodes)
                .launch(cfg_n)
        }
        .context("launch ghost_lattice_path_compress")?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct StepSpan {
    min_step: u64,
    max_step: u64,
}

/// Convenience: load every `_stream{:02}_ghost_tiles.bin` in `output_dir`
/// matching `topo_stem`, project their v2 records to lattice nodes, and
/// concatenate them into a single vector. Skips files that don't open or
/// don't carry a valid GhostTileRing counter sector.
pub fn load_ghost_v2_payloads_from_dir(
    output_dir: &std::path::Path,
    topo_stem: &str,
    schedule: &dyn PhaseSchedule,
    step_bucket_size: u64,
) -> Result<Vec<GhostPhaseLatticeNode>> {
    let mut out: Vec<GhostPhaseLatticeNode> = Vec::new();
    let mut total_records_seen: usize = 0;
    let mut total_v2_projected: usize = 0;
    for entry in std::fs::read_dir(output_dir)
        .with_context(|| format!("read_dir({})", output_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let prefix = format!("{}_stream", topo_stem);
        if !name_str.starts_with(&prefix) || !name_str.ends_with("_ghost_tiles.bin") {
            continue;
        }
        // Pull the stream id from `<prefix>NN_ghost_tiles.bin`.
        let mid = &name_str[prefix.len()..];
        let stream_id: u16 = mid.split('_').next().unwrap_or("0").parse().unwrap_or(0);

        let path = entry.path();
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read({})", path.display()))?;
        if bytes.len() <= 4096 {
            log::warn!(
                "  [GHOST-LATTICE-4D] {} too small ({} B) — skipping",
                path.display(),
                bytes.len()
            );
            continue;
        }
        // First 4 bytes hold the GPU-written n_frames_written counter; the
        // first 4096 B are the counter sector.
        let n_frames = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let payload = &bytes[4096..];
        let n_in_payload = payload.len() / GHOST_RECORD_BYTES;
        let n_actual = n_frames.min(n_in_payload);
        let actual_payload = &payload[..n_actual * GHOST_RECORD_BYTES];

        let projected = project_ghost_v2_payload(actual_payload, stream_id, schedule, step_bucket_size);
        total_records_seen += n_actual;
        total_v2_projected += projected.len();
        log::info!(
            "  [GHOST-LATTICE-4D] {}: stream={} n_records={} v2_projected={}",
            path.file_name().unwrap().to_string_lossy(),
            stream_id,
            n_actual,
            projected.len()
        );
        out.extend(projected);
    }

    if total_records_seen == 0 {
        bail!(
            "no ghost_tiles.bin records found in {} for stem {}",
            output_dir.display(),
            topo_stem
        );
    }
    log::info!(
        "  [GHOST-LATTICE-4D] total: {} records seen, {} v2-projected ({}% v2 yield)",
        total_records_seen,
        total_v2_projected,
        if total_records_seen > 0 {
            (total_v2_projected * 100) / total_records_seen
        } else {
            0
        }
    );
    Ok(out)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_record(
        step_idx: u64,
        x: f32,
        y: f32,
        z: f32,
        kl: f32,
        plane_geo: [f32; 6],
        plane_caus: Option<[f32; 6]>,
        causal_lead: u32,
    ) -> [u8; GHOST_RECORD_BYTES] {
        let mut r = [0u8; GHOST_RECORD_BYTES];
        r[0..8].copy_from_slice(&1u64.to_le_bytes()); // frame_idx
        r[8..12].copy_from_slice(&7u32.to_le_bytes()); // site_id
        r[12] = 0x41; // chain_id
        r[13] = 1; // adjudication_code
        r[14..16].copy_from_slice(&0u16.to_le_bytes());
        r[16..20].copy_from_slice(&kl.to_le_bytes());
        // plane geometry @ offset 20
        for (i, v) in plane_geo.iter().enumerate() {
            let off = 20 + i * 4;
            r[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        if let Some(pc) = plane_caus {
            for (i, v) in pc.iter().enumerate() {
                let off = 20 + (6 + i) * 4;
                r[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        // thermo_flux @ 116
        r[116..120].copy_from_slice(&f32::NAN.to_le_bytes());
        r[120..124].copy_from_slice(&f32::NAN.to_le_bytes());
        r[124..128].copy_from_slice(&causal_lead.to_le_bytes());
        // schema_version @ 128
        r[128..132].copy_from_slice(&GHOST_FRAME_SCHEMA_V2.to_le_bytes());
        r[GHOST_V2_OFFSET_GEAR_ID..GHOST_V2_OFFSET_GEAR_ID + 4]
            .copy_from_slice(&0u32.to_le_bytes());
        r[GHOST_V2_OFFSET_DT_FS..GHOST_V2_OFFSET_DT_FS + 4]
            .copy_from_slice(&2.0f32.to_le_bytes());
        r[GHOST_V2_OFFSET_STEP_IDX..GHOST_V2_OFFSET_STEP_IDX + 8]
            .copy_from_slice(&step_idx.to_le_bytes());
        // Spatial native bit + AABB
        let fcf = GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT;
        r[GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS
            ..GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS + 2]
            .copy_from_slice(&fcf.to_le_bytes());
        for (i, v) in [x - 0.5, y - 0.5, z - 0.5].iter().enumerate() {
            let off = GHOST_V2_OFFSET_AABB_MIN + i * 4;
            r[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in [x + 0.5, y + 0.5, z + 0.5].iter().enumerate() {
            let off = GHOST_V2_OFFSET_AABB_MAX + i * 4;
            r[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in [x, y, z].iter().enumerate() {
            let off = GHOST_V2_OFFSET_CENTROID + i * 4;
            r[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        r
    }

    #[test]
    fn projects_v2_record_with_native_aabb() {
        let r = synth_record(
            1000,
            10.0,
            20.0,
            30.0,
            0.123,
            [0.4, 0.3, 0.15, 0.08, 0.04, 0.03],
            None,
            42,
        );
        let schedule = QuartilePhaseSchedule {
            min_step: 0,
            max_step: 4000,
        };
        let n = project_ghost_v2_record(&r, 17, 3, &schedule, 500).unwrap();
        assert_eq!(n.tile_index, 17);
        assert_eq!(n.stream_id, 3);
        assert_eq!(n.site_id, 7);
        assert_eq!(n.frame_idx, 1);
        assert_eq!(n.step_idx, 1000);
        assert_eq!(n.protocol_phase, PHASE_HEATING);
        assert_eq!(n.step_bucket, 2);
        assert_eq!(n.causal_lead_residue, 42);
        assert!(n.has_valid_aabb());
        assert!(n.has_any_so3_plane());
        assert_eq!(n.so3_plane_status & SO3_PLANE_GEOMETRY, SO3_PLANE_GEOMETRY);
        assert_eq!(n.so3_plane_status & SO3_PLANE_CAUSALITY, 0);
        assert!((n.centroid_xyz[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_v1_record() {
        let mut r = [0u8; GHOST_RECORD_BYTES];
        r[128..132].copy_from_slice(&0u32.to_le_bytes()); // v1 / legacy
        let schedule = QuartilePhaseSchedule {
            min_step: 0,
            max_step: 1000,
        };
        let n = project_ghost_v2_record(&r, 0, 0, &schedule, 500);
        assert!(n.is_none());
    }

    #[test]
    fn quartile_schedule_partitions_correctly() {
        let s = QuartilePhaseSchedule {
            min_step: 0,
            max_step: 4000,
        };
        assert_eq!(s.phase_for(0, 0), PHASE_COLD_HOLD);
        assert_eq!(s.phase_for(999, 0), PHASE_COLD_HOLD);
        assert_eq!(s.phase_for(1000, 0), PHASE_HEATING);
        assert_eq!(s.phase_for(2000, 0), PHASE_WARM_HOLD);
        assert_eq!(s.phase_for(3500, 0), PHASE_COOLING);
        assert_eq!(s.phase_for(5000, 0), PHASE_COOLING); // saturate
    }

    #[test]
    fn aabb_volume_handles_degenerate_box() {
        let mut n = GhostPhaseLatticeNode::empty();
        n.aabb_min = [1.0, 2.0, 3.0];
        n.aabb_max = [2.0, 4.0, 6.0];
        assert!((n.aabb_volume() - 6.0).abs() < 1e-6);
        n.aabb_max = [1.0, 2.0, 3.0]; // collapsed
        assert!(n.aabb_volume() == 0.0);
    }

    #[test]
    fn node_struct_size_pinned() {
        // Hard-pin against silent repacking that would break the kernel.
        assert_eq!(std::mem::size_of::<GhostPhaseLatticeNode>(), 208);
    }
}
