//! Per-spike Apache Arrow IPC writer (Stage 1B-1).
//!
//! Replaces the per-site `*.spike_events.json` dump (15 GB across 5 sites,
//! partial schema, JSON serialization on the critical path) with a single
//! per-target `*.spike_events.arrow` file (~3-4 GB total uncompressed,
//! columnar, 31 columns of full per-spike tagging including the stratified
//! background spike bucket).
//!
//! ## Why Apache Arrow IPC and not Parquet?
//!
//! Parquet is a *rest* format, not a *streaming* format. The Apache Arrow
//! IPC streaming format has the property that **the in-memory layout is the
//! wire format** — there is no serialize step, so the cost of "writing"
//! a batch is just the cost of constructing the columnar arrays and a
//! single `cuMemcpyDtoHAsync` (when used in a streaming pipeline). For the
//! initial Stage 1B-1 implementation we write at end-of-pipeline rather
//! than streaming per-chunk, but the format choice keeps the door open
//! for that upgrade in Stage 1B-2 / Stage 4.
//!
//! Arrow IPC is also the *lingua franca* for downstream analytics: Polars,
//! DuckDB, cuDF, Pandas, R, Julia, MATLAB can all read it without
//! conversion. Compared to JSON, an Arrow IPC file is:
//!
//!   • ~5× smaller on disk (binary, columnar, no key repetition)
//!   • ~100× faster to read (mmap, no parser)
//!   • Strongly typed (no string ↔ int conversion bugs)
//!   • Schema-evolving (downstream can add columns without breaking readers)
//!
//! Reference: https://arrow.apache.org/docs/format/Columnar.html#serialization-and-interprocess-communication-ipc
//!
//! ## What's in the schema (31 columns)
//!
//! Each row is one spike, with three classes of fields. Counts: 6
//! provenance + 19 physical state + 6 classification = 31 columns total.
//!
//! ### Provenance (6 columns)
//!   - `spike_id`         u64  — sequential global ID across the run
//!   - `replica_seed`     u64  — `--replica-seed` value (reproducibility)
//!   - `stream_id`        u8   — stream index 0..n_streams-1
//!   - `group_id`         u8   — TWIN group: 0=TS, 1=EQ, 2=UV, 3=HY
//!   - `chunk_idx`        u16  — chunk index where this spike was emitted
//!                                (derived host-side from cumulative steps)
//!   - `voxel_idx`        i32  — unique voxel ID in the engine grid
//!                                (Opp #4 — was previously dropped on dump)
//!
//! ### Physical state (12 columns from GpuSpikeEvent)
//!   - `timestep`         i32  — engine MD step number
//!   - `frame_index`      u16  — `timestep / 1000` (back-compat with JSON)
//!   - `x`, `y`, `z`      f32 × 3 — voxel center position
//!   - `intensity`        f32  — spike intensity
//!   - `spike_source`     i32  — raw numeric source (Opp #1, was string)
//!   - `mechanism_tag`    utf8 — stable mechanism class for downstream ML;
//!                                UNK + LIF maps to LIF_THERMAL_SHAPE
//!   - `aromatic_type`    i32  — raw numeric type (Opp #3, was string)
//!   - `aromatic_residue_id` i32
//!   - `phase_bits`       u32  — 10-bit CCNS phase (Opp #2, was dropped)
//!   - `n_residues`       u8   — count for nearby_residues
//!   - `nearby_residues`  FixedSizeList<i32, 8> — Opp #2, was dropped
//!   - `n_nearby_excited` u8
//!   - `vibrational_energy` f32
//!   - `water_density`    f32
//!   - `wd_change`        f32  — Opp #5 (was computed but never exported)
//!   - `wavelength_nm`    f32
//!   - `ccns_phase`       u8   — 0=cold_hold, 1=ramp, 2=warm_hold, 3=cooling
//!
//! ### Classification (10 columns, host-side derived)
//!   - `site_id`          i32  — consensus site cluster_id, -1 for background
//!   - `nearest_site_id`  i32  — closest consensus site (== site_id when assigned)
//!   - `nearest_site_dist` f32 — distance in Å
//!   - `background_class` u8   — 0=primary_site, 1=bulk_thermal,
//!                                2=surface_noise, 3=near_miss,
//!                                4=relabel_candidate
//!   - `burial_score`     f32  — atom-density-based burial proxy [0,1]
//!   - `intensity_percentile` u8 — rank within (channel) for this run [0,100]
//!
//! ## Background spike stratification (no magic constants)
//!
//! Background spikes (those that fell outside any consensus site's radius)
//! are NOT discarded. They are preserved with `site_id = -1` and
//! sub-classified by `background_class` so the training pipeline can sample
//! them deliberately:
//!
//!   0 = primary_site         (site_id != -1, in a consensus site)
//!   1 = bulk_thermal         (deep solvent / surface, low burial, far from sites)
//!   2 = surface_noise        (surface, moderate burial, mid-distance from sites)
//!   3 = near_miss            (just outside a site's radius — closest 10% of bg)
//!   4 = relabel_candidate    (high burial AND high intensity AND close to a site)
//!
//! The thresholds for these classes are **per-run percentiles** computed
//! from the actual distribution of background spikes — no fixed numeric
//! cutoffs. See `classify_background_class` below.

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, FixedSizeListBuilder, Float32Builder, Int32Array, Int32Builder, RecordBatch,
    StringBuilder, UInt16Builder, UInt32Builder, UInt64Builder, UInt8Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;

use crate::fused_engine::GpuSpikeEvent;

/// Number of nearby residues stored per spike (matches GpuSpikeEvent).
pub const NEARBY_RESIDUES_LEN: i32 = 8;

/// Build the canonical Arrow schema for the per-spike output.
///
/// 31 columns total — see the module-level documentation for the field
/// list and rationale. Field order is **stable** and used by downstream
/// readers; do not reorder without bumping a schema version.
pub fn build_spike_schema() -> Arc<Schema> {
    // Note: the inner field of `nearby_residues` MUST be `nullable: true`
    // even though the builder always pads unused slots with -1 — this is
    // because `arrow::array::FixedSizeListBuilder` defaults its inner
    // field to nullable, and `RecordBatch::try_new` does a strict type
    // equality check including the nullable flag. Caught by
    // `record_batch_builds_from_synthetic_spikes` test.
    let nearby_field = Arc::new(Field::new("item", DataType::Int32, true));
    let fields = vec![
        // Provenance
        Field::new("spike_id", DataType::UInt64, false),
        Field::new("replica_seed", DataType::UInt64, false),
        Field::new("stream_id", DataType::UInt8, false),
        Field::new("group_id", DataType::UInt8, false),
        Field::new("chunk_idx", DataType::UInt16, false),
        Field::new("voxel_idx", DataType::Int32, false),
        // Physical state
        Field::new("timestep", DataType::Int32, false),
        Field::new("frame_index", DataType::UInt16, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("z", DataType::Float32, false),
        Field::new("intensity", DataType::Float32, false),
        Field::new("spike_source", DataType::Int32, false),
        Field::new("mechanism_tag", DataType::Utf8, false),
        Field::new("aromatic_type", DataType::Int32, false),
        Field::new("aromatic_residue_id", DataType::Int32, false),
        Field::new("phase_bits", DataType::UInt32, false),
        Field::new("n_residues", DataType::UInt8, false),
        Field::new(
            "nearby_residues",
            DataType::FixedSizeList(nearby_field, NEARBY_RESIDUES_LEN),
            false,
        ),
        Field::new("n_nearby_excited", DataType::UInt8, false),
        Field::new("vibrational_energy", DataType::Float32, false),
        Field::new("water_density", DataType::Float32, false),
        Field::new("wd_change", DataType::Float32, false),
        Field::new("wavelength_nm", DataType::Float32, false),
        Field::new("ccns_phase", DataType::UInt8, false),
        // Classification
        Field::new("site_id", DataType::Int32, false),
        Field::new("nearest_site_id", DataType::Int32, false),
        Field::new("nearest_site_dist", DataType::Float32, false),
        Field::new("background_class", DataType::UInt8, false),
        Field::new("burial_score", DataType::Float32, false),
        Field::new("intensity_percentile", DataType::UInt8, false),
    ];
    Arc::new(Schema::new(fields))
}

/// Per-spike classification metadata produced by the host before writing.
///
/// One `SpikeClassification` corresponds to one `GpuSpikeEvent` in the
/// input slice (parallel arrays). The writer expects them in matching
/// order.
#[derive(Debug, Clone, Copy)]
pub struct SpikeClassification {
    /// Stream index that emitted this spike (0..n_streams-1).
    pub stream_id: u8,
    /// TWIN group derived from stream_id (0..3).
    pub group_id: u8,
    /// Chunk index when this spike was emitted (derived from timestep on
    /// the host using the chunk-boundary table — fixed `chunk_size = 500`
    /// for now; BOCPD-derived chunks land in Stage 1B-2).
    pub chunk_idx: u16,
    /// CCNS protocol phase: 0=cold_hold, 1=ramp, 2=warm_hold, 3=cooling.
    pub ccns_phase: u8,
    /// Consensus site cluster_id this spike was assigned to, or -1 for
    /// background (no site within radius).
    pub site_id: i32,
    /// Closest consensus site cluster_id (== `site_id` when assigned;
    /// for background spikes, this is the nearest site even though the
    /// spike is outside its radius).
    pub nearest_site_id: i32,
    /// Distance in Å to `nearest_site_id`.
    pub nearest_site_dist: f32,
    /// Background sub-classification (see module doc).
    pub background_class: u8,
    /// Burial score (atoms within 12 Å sphere, normalized) [0, 1].
    pub burial_score: f32,
    /// Intensity rank within this spike's source channel [0, 100].
    pub intensity_percentile: u8,
}

/// Background sub-classes for stratified training data.
pub mod background_class {
    pub const PRIMARY_SITE: u8 = 0;
    pub const BULK_THERMAL: u8 = 1;
    pub const SURFACE_NOISE: u8 = 2;
    pub const NEAR_MISS: u8 = 3;
    pub const RELABEL_CANDIDATE: u8 = 4;
}

#[inline]
pub fn mechanism_tag_for_spike(spike: &GpuSpikeEvent) -> &'static str {
    match spike.spike_source {
        1 => "UV_AROMATIC_PERTURBATION",
        3 => "EFP_ELECTROSTATIC_FIELD",
        4 => "LADD_ATOM_DEPARTURE",
        5 => "COFIRE_COHERENCE",
        _ if spike.aromatic_type < 0 => "LIF_THERMAL_SHAPE",
        _ => "LIF_LOCAL_INTENSITY",
    }
}

/// Compute the TWIN group_id from a stream index given the multi-differential
/// engines-per-group layout.
///
/// In `--multi-differential` mode the streams are organized as 4 groups
/// (TS, EQ, UV, HY) with `n_streams / 4` engines per group. The mapping
/// is `group = stream_idx / engines_per_group`. For non-multi-diff runs
/// (where there is only one logical group), all streams map to group 0.
#[inline]
pub fn group_id_for_stream(stream_idx: usize, n_streams: usize, multi_diff: bool) -> u8 {
    if !multi_diff || n_streams < 4 {
        return 0;
    }
    let epg = (n_streams / 4).max(1);
    ((stream_idx / epg).min(3)) as u8
}

/// Build the chunk-boundary lookup table for host-side derivation of
/// `chunk_idx` per spike.
///
/// In Stage 1B-1 the chunk size is fixed (the `chunk_size = 500` constant
/// in the autonomous chunk loop). Each spike's `chunk_idx` is computed as
/// `min(timestep / chunk_size, max_chunk_idx)`. When BOCPD lands in Stage
/// 1B-2 the chunk boundaries become non-uniform; this function will be
/// replaced by a lookup against the BOCPD-emitted boundary table.
#[inline]
pub fn chunk_idx_for_timestep(timestep: i32, chunk_size: i32, max_chunks: i32) -> u16 {
    if chunk_size <= 0 || timestep < 0 {
        return 0;
    }
    let idx = (timestep / chunk_size).min(max_chunks - 1).max(0);
    idx as u16
}

/// Map a CCNS protocol step + phase boundaries to a phase code.
///
/// 0 = cold_hold, 1 = ramp, 2 = warm_hold. Matches the convention in the
/// standard step() path's phase computation at fused_engine.rs:5889 and the
/// Stage 1A.5 KCC drive at nhs_rt_full.rs:3460+.
#[inline]
pub fn ccns_phase_for_step(timestep: i32, cold_hold: i32, ramp: i32) -> u8 {
    if timestep < cold_hold {
        0
    } else if timestep < cold_hold + ramp {
        1
    } else {
        2
    }
}

/// Classify a background spike using **per-run percentile thresholds**
/// (no magic constants — every cutoff is derived from the run's actual
/// distribution) PLUS the per-channel intensity rank (which is itself a
/// percentile so requires no additional run-level statistics).
///
/// Inputs:
/// - `site_id` of this spike (-1 if background)
/// - `nearest_site_dist` of this spike in Å
/// - `intensity_percentile` of this spike (0-100, rank within source channel)
/// - `burial_score` of this spike (0-1, linear normalization of n_residues / 8)
/// - `bg_dist_p10` — 10th percentile of nearest_site_dist across all background
/// - `bg_dist_p50` — median nearest_site_dist across all background (currently
///                   computed but reserved for future refinements; not used in
///                   the v5 decision tree)
///
/// Decision tree (in order):
///   1. site_id != -1                                                    → PRIMARY_SITE (0)
///   2. nearest_site_dist < bg_dist_p10                                  → NEAR_MISS (3)
///   3. intensity_percentile > 75 AND burial_score >= 1.0                → RELABEL_CANDIDATE (4)
///   4. intensity_percentile >= 50                                       → SURFACE_NOISE (2)
///   5. otherwise (low intensity rank within channel)                    → BULK_THERMAL (1)
///
/// ## Why intensity_percentile primary, distance+burial secondary
///
/// The classifier went through three iterations on the 4LPK validation:
///
/// **v3 (percentile burial cutoffs)** — `burial > bg_burial_p50` for
/// surface_noise. The `n_residues` distribution for background spikes is
/// heavily skewed to one end on 4LPK (and similar mid-size proteins),
/// putting `bg_burial_p50 == bg_burial_p75 == 1.000`. Strict inequality
/// `burial > 1.0` matched NO spikes. surface_noise + relabel_candidate
/// were both empty.
///
/// **v4 (binary burial)** — `burial < 1.0` for surface_noise. I assumed
/// the saturation meant "75% of background is buried" but the actual
/// distribution on 4LPK was the inverse: ~90% of background is surface
/// (`burial < 1.0`), ~10% is internal (`burial == 1.0`). v4 collapsed
/// almost everything into surface_noise (12M of 13.3M = 90.4%). Bulk
/// thermal got 420 spikes, relabel_candidate got 122 — neither bucket
/// was usable for training.
///
/// **v5 (this version) — intensity_percentile primary**:
///
/// `intensity_percentile` is the only feature in the classifier that is
/// guaranteed to be well-distributed by construction (it's the per-spike
/// rank within the source channel, so it always spans 0-100 evenly with
/// no possibility of saturation). Using it as the surface_noise vs
/// bulk_thermal divider gives a roughly even split of the residual
/// background population, regardless of the underlying burial distribution.
///
/// distance is still used for near_miss (the spatial frontier — hard
/// negatives), and burial is still used as a tiebreaker for
/// relabel_candidate (the high-intensity-AND-internal class — the
/// audit channel for missed pockets), but neither distance nor burial
/// drives the bulk surface/thermal split anymore. That avoids the
/// bimodal-distribution failure mode that broke v3 and v4.
///
/// Predicted v5 distribution on 4LPK background (13.3M spikes):
///   - NEAR_MISS:         ~10% = 1.3M (closest distance decile, by definition)
///   - RELABEL_CANDIDATE: ~2-3% (high intensity AND buried)
///   - SURFACE_NOISE:     ~40-45% (above-median intensity)
///   - BULK_THERMAL:      ~40-45% (below-median intensity)
///
/// All four background classes have meaningful population, suitable for
/// stratified training data sampling by difficulty.
///
/// `burial_score` is still emitted as an Arrow column for downstream
/// fine-grained queries — the classification field is a coarse 5-way
/// stratification, not a complete description. Users with different
/// distributions on different proteins can re-classify offline using
/// the raw columns.
#[inline]
pub fn classify_background(
    site_id: i32,
    nearest_site_dist: f32,
    intensity_percentile: u8,
    burial_score: f32,
    bg_dist_p10: f32,
    _bg_dist_p50: f32,
) -> u8 {
    if site_id != -1 {
        return background_class::PRIMARY_SITE;
    }
    if nearest_site_dist < bg_dist_p10 {
        return background_class::NEAR_MISS;
    }
    if intensity_percentile > 75 && burial_score >= 1.0 {
        return background_class::RELABEL_CANDIDATE;
    }
    if intensity_percentile >= 50 {
        return background_class::SURFACE_NOISE;
    }
    background_class::BULK_THERMAL
}

/// Compute the per-run percentiles needed for `classify_background`.
///
/// Walks the background spike list once, collecting `nearest_site_dist`,
/// sorts, returns `(dist_p10, dist_p50)`. Returns zeros if there are no
/// background spikes.
///
/// Burial percentiles are no longer needed — see the v3 smoke validation
/// note in `classify_background` for why burial is used as a binary
/// feature instead of a percentile-ranked one.
pub fn compute_background_percentiles(
    background_classifications: &[SpikeClassification],
) -> (f32, f32) {
    if background_classifications.is_empty() {
        return (0.0, 0.0);
    }
    let mut dists: Vec<f32> = background_classifications
        .iter()
        .filter(|c| c.site_id == -1)
        .map(|c| c.nearest_site_dist)
        .collect();
    if dists.is_empty() {
        return (0.0, 0.0);
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = dists.len();
    let p10_idx = ((n as f32) * 0.10) as usize;
    let p50_idx = n / 2;
    (
        dists.get(p10_idx.min(n - 1)).copied().unwrap_or(0.0),
        dists.get(p50_idx.min(n - 1)).copied().unwrap_or(0.0),
    )
}

/// Build a `RecordBatch` from parallel slices of spikes and classifications.
///
/// Both slices must have the same length. Returns an error if any column
/// builder fails. The output batch contains all 31 columns of the canonical
/// schema.
pub fn build_spike_record_batch(
    spikes: &[GpuSpikeEvent],
    classifications: &[SpikeClassification],
    replica_seed: u64,
) -> Result<RecordBatch> {
    if spikes.len() != classifications.len() {
        anyhow::bail!(
            "spike/classification length mismatch: {} vs {}",
            spikes.len(),
            classifications.len()
        );
    }
    let n = spikes.len();

    // Allocate column builders with capacity hints.
    let mut spike_id = UInt64Builder::with_capacity(n);
    let mut replica_seed_b = UInt64Builder::with_capacity(n);
    let mut stream_id_b = UInt8Builder::with_capacity(n);
    let mut group_id_b = UInt8Builder::with_capacity(n);
    let mut chunk_idx_b = UInt16Builder::with_capacity(n);
    let mut voxel_idx_b = Int32Builder::with_capacity(n);
    let mut timestep_b = Int32Builder::with_capacity(n);
    let mut frame_index_b = UInt16Builder::with_capacity(n);
    let mut x_b = Float32Builder::with_capacity(n);
    let mut y_b = Float32Builder::with_capacity(n);
    let mut z_b = Float32Builder::with_capacity(n);
    let mut intensity_b = Float32Builder::with_capacity(n);
    let mut spike_source_b = Int32Builder::with_capacity(n);
    let mut mechanism_tag_b = StringBuilder::with_capacity(n, n * 24);
    let mut aromatic_type_b = Int32Builder::with_capacity(n);
    let mut aromatic_resid_b = Int32Builder::with_capacity(n);
    let mut phase_bits_b = UInt32Builder::with_capacity(n);
    let mut n_residues_b = UInt8Builder::with_capacity(n);
    let mut nearby_b = FixedSizeListBuilder::with_capacity(
        Int32Builder::with_capacity(n * NEARBY_RESIDUES_LEN as usize),
        NEARBY_RESIDUES_LEN,
        n,
    );
    let mut n_excited_b = UInt8Builder::with_capacity(n);
    let mut vib_energy_b = Float32Builder::with_capacity(n);
    let mut water_density_b = Float32Builder::with_capacity(n);
    let mut wd_change_b = Float32Builder::with_capacity(n);
    let mut wavelength_b = Float32Builder::with_capacity(n);
    let mut ccns_phase_b = UInt8Builder::with_capacity(n);
    let mut site_id_b = Int32Builder::with_capacity(n);
    let mut nearest_site_id_b = Int32Builder::with_capacity(n);
    let mut nearest_site_dist_b = Float32Builder::with_capacity(n);
    let mut background_class_b = UInt8Builder::with_capacity(n);
    let mut burial_score_b = Float32Builder::with_capacity(n);
    let mut intensity_pct_b = UInt8Builder::with_capacity(n);

    for (i, (spike, cls)) in spikes.iter().zip(classifications.iter()).enumerate() {
        spike_id.append_value(i as u64);
        replica_seed_b.append_value(replica_seed);
        stream_id_b.append_value(cls.stream_id);
        group_id_b.append_value(cls.group_id);
        chunk_idx_b.append_value(cls.chunk_idx);
        voxel_idx_b.append_value(spike.voxel_idx);
        timestep_b.append_value(spike.timestep);
        frame_index_b.append_value((spike.timestep / 1000).max(0).min(u16::MAX as i32) as u16);
        x_b.append_value(spike.position[0]);
        y_b.append_value(spike.position[1]);
        z_b.append_value(spike.position[2]);
        intensity_b.append_value(spike.intensity);
        spike_source_b.append_value(spike.spike_source);
        mechanism_tag_b.append_value(mechanism_tag_for_spike(spike));
        aromatic_type_b.append_value(spike.aromatic_type);
        aromatic_resid_b.append_value(spike.aromatic_residue_id);
        phase_bits_b.append_value(spike.phase_bits);
        n_residues_b.append_value(spike.n_residues.clamp(0, NEARBY_RESIDUES_LEN) as u8);

        // Fixed-size list of 8 nearby residues. Always exactly 8 entries; pad
        // unused slots with -1 so the schema is rectangular.
        let n_valid = spike.n_residues.clamp(0, NEARBY_RESIDUES_LEN) as usize;
        for j in 0..NEARBY_RESIDUES_LEN as usize {
            if j < n_valid {
                nearby_b.values().append_value(spike.nearby_residues[j]);
            } else {
                nearby_b.values().append_value(-1);
            }
        }
        nearby_b.append(true);

        n_excited_b.append_value(spike.n_nearby_excited.clamp(0, u8::MAX as i32) as u8);
        vib_energy_b.append_value(spike.vibrational_energy);
        water_density_b.append_value(spike.water_density);
        wd_change_b.append_value(spike.wd_change);
        wavelength_b.append_value(spike.wavelength_nm);
        ccns_phase_b.append_value(cls.ccns_phase);
        site_id_b.append_value(cls.site_id);
        nearest_site_id_b.append_value(cls.nearest_site_id);
        nearest_site_dist_b.append_value(cls.nearest_site_dist);
        background_class_b.append_value(cls.background_class);
        burial_score_b.append_value(cls.burial_score);
        intensity_pct_b.append_value(cls.intensity_percentile);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(spike_id.finish()),
        Arc::new(replica_seed_b.finish()),
        Arc::new(stream_id_b.finish()),
        Arc::new(group_id_b.finish()),
        Arc::new(chunk_idx_b.finish()),
        Arc::new(voxel_idx_b.finish()),
        Arc::new(timestep_b.finish()),
        Arc::new(frame_index_b.finish()),
        Arc::new(x_b.finish()),
        Arc::new(y_b.finish()),
        Arc::new(z_b.finish()),
        Arc::new(intensity_b.finish()),
        Arc::new(spike_source_b.finish()),
        Arc::new(mechanism_tag_b.finish()),
        Arc::new(aromatic_type_b.finish()),
        Arc::new(aromatic_resid_b.finish()),
        Arc::new(phase_bits_b.finish()),
        Arc::new(n_residues_b.finish()),
        Arc::new(nearby_b.finish()),
        Arc::new(n_excited_b.finish()),
        Arc::new(vib_energy_b.finish()),
        Arc::new(water_density_b.finish()),
        Arc::new(wd_change_b.finish()),
        Arc::new(wavelength_b.finish()),
        Arc::new(ccns_phase_b.finish()),
        Arc::new(site_id_b.finish()),
        Arc::new(nearest_site_id_b.finish()),
        Arc::new(nearest_site_dist_b.finish()),
        Arc::new(background_class_b.finish()),
        Arc::new(burial_score_b.finish()),
        Arc::new(intensity_pct_b.finish()),
    ];

    // Diagnostic: log column lengths and types so RecordBatch::try_new failures
    // can be diagnosed without running the full pipeline twice.
    let schema = build_spike_schema();
    if log::log_enabled!(log::Level::Debug) || cfg!(debug_assertions) {
        for (i, (field, col)) in schema.fields().iter().zip(columns.iter()).enumerate() {
            log::debug!(
                "  Arrow col {}: {} expected={:?} got={:?} len={}",
                i,
                field.name(),
                field.data_type(),
                col.data_type(),
                col.len()
            );
        }
    }
    RecordBatch::try_new(schema.clone(), columns).map_err(|e| {
        // Re-walk on failure to give actionable diagnostics in the log.
        log::warn!(
            "  RecordBatch::try_new failed with {} expected rows: {}",
            n,
            e
        );
        anyhow::anyhow!("RecordBatch::try_new failed: {}", e)
    })
}

/// Write a `RecordBatch` to disk as an Arrow IPC file (`.arrow` extension).
///
/// Uses `arrow::ipc::writer::FileWriter` which produces a self-describing
/// file with the schema header followed by one or more record batches.
/// For Stage 1B-1 we write a single batch containing all spikes from the
/// run; Stage 1B-2 will switch to streaming per-chunk batches.
pub fn write_spike_arrow_file(path: &Path, batch: &RecordBatch) -> Result<()> {
    use std::fs::File;
    let file = File::create(path)
        .with_context(|| format!("Failed to create Arrow file: {}", path.display()))?;
    let mut writer =
        FileWriter::try_new(file, &batch.schema()).context("FileWriter::try_new failed")?;
    writer.write(batch).context("FileWriter::write failed")?;
    writer.finish().context("FileWriter::finish failed")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_has_expected_columns() {
        // 6 provenance + 19 physical state + 6 classification = 31 columns total
        // (see module-level documentation for the exact field list)
        let s = build_spike_schema();
        assert_eq!(s.fields().len(), 31);
        assert!(s.field_with_name("mechanism_tag").is_ok());
    }

    #[test]
    fn group_id_for_stream_handles_multi_diff() {
        // 8 streams in multi-diff: 2 engines per group, groups 0..3
        assert_eq!(group_id_for_stream(0, 8, true), 0);
        assert_eq!(group_id_for_stream(1, 8, true), 0);
        assert_eq!(group_id_for_stream(2, 8, true), 1);
        assert_eq!(group_id_for_stream(3, 8, true), 1);
        assert_eq!(group_id_for_stream(7, 8, true), 3);
        // Without multi-diff, all streams are group 0
        assert_eq!(group_id_for_stream(5, 8, false), 0);
        // Edge case: fewer than 4 streams
        assert_eq!(group_id_for_stream(2, 3, true), 0);
    }

    #[test]
    fn classify_background_decision_tree() {
        // v5 args: (site_id, nearest_dist, intensity_pct, burial,
        //          bg_dist_p10, bg_dist_p50)

        // Primary site (assigned)
        assert_eq!(
            classify_background(2, 0.0, 50, 0.5, 5.0, 12.0),
            background_class::PRIMARY_SITE
        );
        // Near miss (closer than p10 = 5.0)
        assert_eq!(
            classify_background(-1, 4.0, 50, 1.0, 5.0, 12.0),
            background_class::NEAR_MISS
        );
        // Relabel candidate (high intensity AND fully buried)
        assert_eq!(
            classify_background(-1, 15.0, 90, 1.0, 5.0, 12.0),
            background_class::RELABEL_CANDIDATE
        );
        // Surface noise (above-median intensity_percentile, not buried-and-high)
        assert_eq!(
            classify_background(-1, 15.0, 60, 0.5, 5.0, 12.0),
            background_class::SURFACE_NOISE
        );
        // Surface noise (above-median intensity, fully buried but
        // intensity_pct <= 75 so doesn't qualify for relabel)
        assert_eq!(
            classify_background(-1, 15.0, 60, 1.0, 5.0, 12.0),
            background_class::SURFACE_NOISE
        );
        // Bulk thermal (low intensity_percentile, anything else)
        assert_eq!(
            classify_background(-1, 15.0, 30, 1.0, 5.0, 12.0),
            background_class::BULK_THERMAL
        );
        assert_eq!(
            classify_background(-1, 15.0, 10, 0.3, 5.0, 12.0),
            background_class::BULK_THERMAL
        );
    }

    #[test]
    fn ccns_phase_boundaries() {
        let cold = 14000;
        let ramp = 6000;
        assert_eq!(ccns_phase_for_step(0, cold, ramp), 0); // cold
        assert_eq!(ccns_phase_for_step(13999, cold, ramp), 0); // cold
        assert_eq!(ccns_phase_for_step(14000, cold, ramp), 1); // ramp
        assert_eq!(ccns_phase_for_step(19999, cold, ramp), 1); // ramp
        assert_eq!(ccns_phase_for_step(20000, cold, ramp), 2); // warm
    }

    #[test]
    fn chunk_idx_clamps_to_max() {
        assert_eq!(chunk_idx_for_timestep(0, 500, 110), 0);
        assert_eq!(chunk_idx_for_timestep(499, 500, 110), 0);
        assert_eq!(chunk_idx_for_timestep(500, 500, 110), 1);
        assert_eq!(chunk_idx_for_timestep(54999, 500, 110), 109); // would be 109
        assert_eq!(chunk_idx_for_timestep(99999, 500, 110), 109); // clamped
    }

    /// End-to-end RecordBatch build test with 3 synthetic spikes.
    /// Catches schema/column-length/type mismatches that the full smoke
    /// test would otherwise only find after 14 minutes of MD.
    #[test]
    fn record_batch_builds_from_synthetic_spikes() {
        let mk = |i: usize| GpuSpikeEvent {
            timestep: (i * 100) as i32,
            voxel_idx: i as i32,
            position: [i as f32, i as f32 * 2.0, i as f32 * 3.0],
            intensity: 50.0 + i as f32,
            nearby_residues: [i as i32, i as i32 + 1, i as i32 + 2, -1, -1, -1, -1, -1],
            n_residues: 3,
            spike_source: 1,
            wavelength_nm: 280.0,
            aromatic_type: 1,
            aromatic_residue_id: i as i32,
            water_density: 0.013,
            vibrational_energy: 0.001,
            n_nearby_excited: 2,
            wd_change: 0.0001,
            phase_bits: 100 + i as u32,
        };
        let spikes: Vec<GpuSpikeEvent> = (0..3).map(mk).collect();
        let cls: Vec<SpikeClassification> = (0..3)
            .map(|i| SpikeClassification {
                stream_id: (i % 8) as u8,
                group_id: ((i % 8) / 2) as u8,
                chunk_idx: i as u16,
                ccns_phase: 1,
                site_id: if i == 0 { 2 } else { -1 },
                nearest_site_id: 2,
                nearest_site_dist: i as f32,
                background_class: 0,
                burial_score: 0.5,
                intensity_percentile: (i * 30) as u8,
            })
            .collect();
        let batch = build_spike_record_batch(&spikes, &cls, 42)
            .expect("RecordBatch build should succeed for synthetic spikes");
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 30);
        // All columns should have length 3
        for (i, col) in batch.columns().iter().enumerate() {
            assert_eq!(col.len(), 3, "column {} length mismatch", i);
        }
    }
}
