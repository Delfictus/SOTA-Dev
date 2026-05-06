// Path B materializer (ad412680+1).
//
// Offline consumer of the MD-only evidence handoff bundle produced by
// `nhs_rt_full --md-only-evidence`. Reads md_evidence_manifest.json,
// validates every per-stream PRSPK001 / PRSGD001 / PRKCC001 binary
// envelope, and materializes honest binding-site candidates from PRISM
// dynamic evidence using spatiotemporal + phase-coherence support.
//
// EMITS:
//
//   materialization_report.json                — per-artifact validation + summaries
//   site_candidates.json                        — voxel-density seed candidates
//   validation_inputs.json                      — paths Path B downstream needs
//   field_completeness_report.pathb.json        — Path B view of what's available
//   binding_sites.materialized.json             — materialized sites with full evidence
//                                                 (only sites meeting minimum criteria;
//                                                  others recorded in
//                                                  `non_materialized_candidates`)
//   site_accuracy_report.json                   — DCC vs. reference ligand if valid;
//                                                 explicit "reference_unavailable" when
//                                                 ground-truth ligand is on a different
//                                                 chain (see ground_truth.json)
//   materialization_field_completeness.json     — per-evidence-source availability matrix
//
// HONEST SCOPE
//
// This is NOT a static-pocket detector. Density peaks are SEED candidates
// only; a site is promoted to "materialized_dynamic_site" only after it
// accumulates spatiotemporal evidence:
//
//   * stream_support       — which of the 8 streams observe it
//   * temporal_support     — timestep extent, recurrence across chunks
//   * phase_support        — phase_bits histogram + Rayleigh phase coherence
//   * residue_support      — nearby_residues from spikes
//   * centroid_manifold    — whole_site / lining_mass / causal_driver /
//                            hot_phase / cold_phase / ligand_adjacent
//
// LIGSITE / static geometry is NOT used. find_neighbors_union DBSCAN
// is NOT used. XGBoost rerank is NOT used.
//
// Warp-matrix / forces / ASC-vectors are noted in field_completeness as
// `deferred` because their on-disk formats are not parsed in this
// commit (no PRISM envelope; raw blobs whose format must be ported
// from the engine emit code in a follow-up). This means
// `so3_support.orientation_coherence_score` from those sources is
// `null`; `phase_coherence_rayleigh_r` is computed from spike
// `phase_bits` and IS a valid SO(3)-adjacent signal in PRISM's
// CCNS protocol-phase space.
//
// Materialization levels:
//
//   candidate_density_only            seeds only; no per-spike evidence
//   candidate_spatiotemporal_partial  has stream + temporal + phase, but
//                                     missing manifold or sub-min support
//   materialized_dynamic_site         all minimum criteria satisfied
//   materialized_dynamic_site_with_validation
//                                     above + DCC against valid reference
//
// `binding_sites.materialized.json` is written only when at least one
// site reaches `materialized_dynamic_site`. Otherwise the file is still
// written with `site_count: 0` and `status: "no_sites_met_criteria"` for
// honest reporting.

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

// ─── CLI ───────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "prism-materialize-sites")]
#[command(about = "Path B offline materializer skeleton — consumes MD-only evidence handoff bundle.")]
struct Args {
    /// md_evidence_manifest.json path produced by `nhs_rt_full --md-only-evidence`.
    #[arg(short, long)]
    manifest: PathBuf,

    /// Output dir. Defaults to the manifest's parent directory.
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Voxel size for density-peak candidate generation (Å).
    #[arg(long, default_value = "2.0")]
    voxel_size_a: f32,

    /// Stage A — number of raw density peaks pulled from the voxel grid
    /// before consolidation. Default raised from 50 to 200 in v2 so that
    /// non-N-terminal regions (e.g., catalytic pockets) survive into the
    /// consolidation stage.
    #[arg(long, default_value = "200")]
    top_k_candidates: usize,

    /// Stage B — centroid distance threshold (Å) for consolidating raw
    /// peaks into candidate regions.
    #[arg(long, default_value = "4.0")]
    consolidation_radius_a: f32,

    /// Stage B — minimum Jaccard overlap of top-12 lining-residue sets
    /// at which two peaks merge regardless of centroid distance.
    #[arg(long, default_value = "0.5")]
    consolidation_jaccard: f32,

    /// Final number of materialized sites to retain after PRISM-native
    /// scoring (top-N by `final_prism_score`). Smaller than
    /// `top_k_candidates` because consolidation collapses duplicates.
    #[arg(long, default_value = "20")]
    materialized_top_n: usize,

    /// Sub-chunk bin size (steps) for refined temporal-persistence scoring.
    /// 100 steps gives 100 bins on a 10000-step run, restoring
    /// discrimination that the engine's `chunk_set` (1000-step bins) loses.
    #[arg(long, default_value = "100")]
    persistence_bin_steps: i32,

    /// Skip the spike-density pass entirely (faster validation-only run).
    #[arg(long, default_value = "false")]
    headers_only: bool,

    /// Verbose logging.
    #[arg(short, long, default_value = "false")]
    verbose: bool,
}

// ─── Manifest schema (read-only mirror of what the engine writes) ──────────

#[derive(Debug, Deserialize)]
struct ManifestArtifact {
    kind: String,
    stream_id: Option<u32>,
    path: String,
    size_bytes: u64,
    schema_version: Option<u32>,
    checksum_fnv1a_64_hex: Option<String>,
    record_count: Option<u64>,
    present: bool,
    note: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u32,
    schema_kind: String,
    run_id: String,
    target: String,
    topology_input: String,
    stream_count: usize,
    streams_serialized: usize,
    total_spikes_md: u64,
    artifacts: Vec<ManifestArtifact>,
    compression_status: String,
    checksum_algorithm: String,
    endian: String,
    v2_was_live: bool,
    path_b_required: bool,
    serialization_failure: Option<String>,
}

// ─── PRISM binary envelope parser ──────────────────────────────────────────

#[derive(Debug, Clone)]
struct EnvelopeHeader {
    magic: [u8; 8],
    schema_version: u32,
    endian_marker: u32,
    stream_id: u32,
    run_id: String,
    stem: String,
    record_count: u64,
    byte_stride: u64,
    payload_size: u64,
}

#[derive(Debug, Serialize)]
struct ArtifactValidation {
    artifact_path: String,
    expected_kind: String,
    magic_observed: String,
    magic_match: bool,
    schema_version: u32,
    endian_marker_hex: String,
    endian_ok_le: bool,
    stream_id: u32,
    stream_id_matches_filename: Option<bool>,
    run_id_in_header: String,
    run_id_matches_manifest: bool,
    stem_in_header: String,
    stem_matches_manifest: bool,
    record_count: u64,
    record_count_matches_manifest: Option<bool>,
    byte_stride: u64,
    payload_size: u64,
    declared_total_size: u64,
    on_disk_size: u64,
    size_consistency: bool,
    payload_checksum_fnv1a_64_hex: String,
    checksum_matches_manifest: Option<bool>,
    trailer_observed: String,
    trailer_match: bool,
    valid: bool,
    failure_reason: Option<String>,
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn fnv1a_64_streamed<R: Read>(r: &mut R, total: u64) -> Result<u64> {
    let mut h = FNV_OFFSET;
    let mut buf = vec![0u8; 1 << 20];
    let mut remaining = total;
    while remaining > 0 {
        let to_read = std::cmp::min(remaining as usize, buf.len());
        r.read_exact(&mut buf[..to_read])?;
        for &b in &buf[..to_read] {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        remaining -= to_read as u64;
    }
    Ok(h)
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_bytes<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    r.read_exact(&mut v)?;
    Ok(v)
}

/// Parse the PRISM envelope header from a file. Leaves the cursor positioned
/// at the start of the payload bytes. Returns the header.
fn read_envelope_header<R: Read>(r: &mut R) -> Result<EnvelopeHeader> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    let schema_version = read_u32_le(r)?;
    let endian_marker = read_u32_le(r)?;
    let stream_id = read_u32_le(r)?;
    let run_id_len = read_u64_le(r)? as usize;
    let run_id_bytes = read_bytes(r, run_id_len)?;
    let run_id = String::from_utf8(run_id_bytes)
        .context("envelope: run_id not utf-8")?;
    let stem_len = read_u64_le(r)? as usize;
    let stem_bytes = read_bytes(r, stem_len)?;
    let stem = String::from_utf8(stem_bytes).context("envelope: stem not utf-8")?;
    let record_count = read_u64_le(r)?;
    let byte_stride = read_u64_le(r)?;
    let payload_size = read_u64_le(r)?;
    Ok(EnvelopeHeader {
        magic,
        schema_version,
        endian_marker,
        stream_id,
        run_id,
        stem,
        record_count,
        byte_stride,
        payload_size,
    })
}

/// Validate an artifact, optionally streaming through the payload to compute
/// the FNV-1a checksum. Returns the validation record + computed checksum.
fn validate_artifact(
    artifact_path: &Path,
    expected_magic: &[u8; 8],
    expected_trailer: &[u8; 8],
    expected_kind: &str,
    manifest_record_count: Option<u64>,
    manifest_checksum_hex: Option<&str>,
    manifest_run_id: &str,
    manifest_stem: &str,
    manifest_filename_stream_id: Option<u32>,
    headers_only: bool,
) -> Result<(ArtifactValidation, Option<EnvelopeHeader>)> {
    let on_disk_size = std::fs::metadata(artifact_path)
        .with_context(|| format!("stat {}", artifact_path.display()))?
        .len();
    let f = File::open(artifact_path)
        .with_context(|| format!("open {}", artifact_path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let header = read_envelope_header(&mut r)?;
    let header_size_so_far: u64 = 8 + 4 + 4 + 4
        + 8 + header.run_id.len() as u64
        + 8 + header.stem.len() as u64
        + 8 + 8 + 8;

    let declared_total = header_size_so_far + header.payload_size + 8 /* checksum */ + 8 /* trailer */;
    let size_consistency = declared_total == on_disk_size;

    let magic_match = &header.magic == expected_magic;
    let endian_ok_le = header.endian_marker == 0x01020304u32;

    // Compute payload checksum (streaming) — unless headers_only.
    let computed_checksum: u64 = if headers_only || !magic_match || !endian_ok_le || !size_consistency {
        0
    } else {
        fnv1a_64_streamed(&mut r, header.payload_size)?
    };

    // Read trailing 8 bytes checksum + 8 bytes trailer.
    let mut tail_checksum_bytes = [0u8; 8];
    let mut trailer_bytes = [0u8; 8];
    if !headers_only && magic_match && endian_ok_le && size_consistency {
        r.read_exact(&mut tail_checksum_bytes)?;
        r.read_exact(&mut trailer_bytes)?;
    }
    let stored_checksum = if headers_only {
        0
    } else {
        u64::from_le_bytes(tail_checksum_bytes)
    };
    let trailer_match = !headers_only && &trailer_bytes == expected_trailer;

    let checksum_matches_manifest = manifest_checksum_hex.map(|hex| {
        format!("{:016x}", computed_checksum) == hex.to_ascii_lowercase()
    });

    let record_count_matches_manifest = manifest_record_count
        .map(|n| n == header.record_count);

    let stream_id_matches_filename =
        manifest_filename_stream_id.map(|id| id == header.stream_id);

    let valid = magic_match
        && endian_ok_le
        && size_consistency
        && (headers_only || trailer_match)
        && (headers_only || stored_checksum == computed_checksum)
        && record_count_matches_manifest.unwrap_or(true)
        && checksum_matches_manifest.unwrap_or(true);

    let failure_reason: Option<String> = if !magic_match {
        Some(format!(
            "magic mismatch: expected {:?}, got {:?}",
            std::str::from_utf8(expected_magic).unwrap_or("?"),
            std::str::from_utf8(&header.magic).unwrap_or("?")
        ))
    } else if !endian_ok_le {
        Some(format!(
            "endian marker not LE: 0x{:08x}",
            header.endian_marker
        ))
    } else if !size_consistency {
        Some(format!(
            "size mismatch: declared total {}, on-disk {}",
            declared_total, on_disk_size
        ))
    } else if !headers_only && !trailer_match {
        Some(format!(
            "trailer mismatch: expected {:?}, got {:?}",
            std::str::from_utf8(expected_trailer).unwrap_or("?"),
            std::str::from_utf8(&trailer_bytes).unwrap_or("?")
        ))
    } else if !headers_only && stored_checksum != computed_checksum {
        Some(format!(
            "checksum mismatch: stored 0x{:016x}, computed 0x{:016x}",
            stored_checksum, computed_checksum
        ))
    } else if record_count_matches_manifest == Some(false) {
        Some("record_count mismatch with manifest".to_string())
    } else if checksum_matches_manifest == Some(false) {
        Some("checksum mismatch with manifest".to_string())
    } else {
        None
    };

    let v = ArtifactValidation {
        artifact_path: artifact_path.display().to_string(),
        expected_kind: expected_kind.to_string(),
        magic_observed: String::from_utf8_lossy(&header.magic).to_string(),
        magic_match,
        schema_version: header.schema_version,
        endian_marker_hex: format!("{:08x}", header.endian_marker),
        endian_ok_le,
        stream_id: header.stream_id,
        stream_id_matches_filename,
        run_id_in_header: header.run_id.clone(),
        run_id_matches_manifest: header.run_id == manifest_run_id,
        stem_in_header: header.stem.clone(),
        stem_matches_manifest: header.stem == manifest_stem,
        record_count: header.record_count,
        record_count_matches_manifest,
        byte_stride: header.byte_stride,
        payload_size: header.payload_size,
        declared_total_size: declared_total,
        on_disk_size,
        size_consistency,
        payload_checksum_fnv1a_64_hex: format!("{:016x}", computed_checksum),
        checksum_matches_manifest,
        trailer_observed: String::from_utf8_lossy(&trailer_bytes).to_string(),
        trailer_match,
        valid,
        failure_reason,
    };
    Ok((v, Some(header)))
}

// ─── Spike density pass ───────────────────────────────────────────────────

#[derive(Debug, Default, Serialize)]
struct StreamSpikeSummary {
    stream_id: u32,
    n_spikes: u64,
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
    intensity_min: f32,
    intensity_max: f32,
    intensity_mean: f64,
    n_uv: u64,
    n_lif: u64,
    n_other_source: u64,
    n_aromatic_trp: u64,
    n_aromatic_tyr: u64,
    n_aromatic_phe: u64,
    n_aromatic_ss: u64,
    n_aromatic_none: u64,
    elapsed_ms: u64,
}

/// Spike record byte offsets (must match GpuSpikeEvent in fused_engine.rs:294).
/// We do NOT mirror the whole struct — we only parse the fields needed for
/// density-peak generation and basic summary stats.
mod spike_offsets {
    pub const POSITION_X:    usize = 8;   // f32
    pub const POSITION_Y:    usize = 12;  // f32
    pub const POSITION_Z:    usize = 16;  // f32
    pub const INTENSITY:     usize = 20;  // f32
    pub const SPIKE_SOURCE:  usize = 60;  // i32 — 1=UV, 2=LIF
    pub const AROMATIC_TYPE: usize = 68;  // i32 — 0=TRP, 1=TYR, 2=PHE, 3=SS, -1=none
}

fn read_f32_at(buf: &[u8], offset: usize) -> f32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[offset..offset + 4]);
    f32::from_le_bytes(b)
}

fn read_i32_at(buf: &[u8], offset: usize) -> i32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[offset..offset + 4]);
    i32::from_le_bytes(b)
}

#[allow(clippy::too_many_arguments)]
fn process_spikes_stream(
    artifact_path: &Path,
    expected_record_count: u64,
    byte_stride: u64,
    voxel_size_a: f32,
    voxel_grid: &mut std::collections::HashMap<(i32, i32, i32), u32>,
) -> Result<StreamSpikeSummary> {
    let start = std::time::Instant::now();
    let f = File::open(artifact_path)
        .with_context(|| format!("open spikes {}", artifact_path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let header = read_envelope_header(&mut r)?;

    let mut summary = StreamSpikeSummary {
        stream_id: header.stream_id,
        n_spikes: 0,
        bbox_min: [f32::INFINITY; 3],
        bbox_max: [f32::NEG_INFINITY; 3],
        intensity_min: f32::INFINITY,
        intensity_max: f32::NEG_INFINITY,
        intensity_mean: 0.0,
        ..Default::default()
    };

    let stride = byte_stride as usize;
    if stride < 72 {
        anyhow::bail!(
            "spikes byte_stride {} too small (need >= 72 to access aromatic_type at offset 68)",
            stride
        );
    }
    let inv_voxel = 1.0f32 / voxel_size_a;

    // Stream in batches of records to avoid loading 1.26 GB into RAM.
    let records_per_batch: u64 = 1 << 16; // 64K records (~6 MB at 96 B/record)
    let mut intensity_sum: f64 = 0.0;
    let mut remaining = header.record_count;
    let mut buf = vec![0u8; (records_per_batch * byte_stride) as usize];
    while remaining > 0 {
        let batch = std::cmp::min(remaining, records_per_batch);
        let bytes = (batch * byte_stride) as usize;
        r.read_exact(&mut buf[..bytes])?;
        for k in 0..(batch as usize) {
            let off = k * stride;
            let rec = &buf[off..off + stride];
            let px = read_f32_at(rec, spike_offsets::POSITION_X);
            let py = read_f32_at(rec, spike_offsets::POSITION_Y);
            let pz = read_f32_at(rec, spike_offsets::POSITION_Z);
            let intensity = read_f32_at(rec, spike_offsets::INTENSITY);
            let source = read_i32_at(rec, spike_offsets::SPIKE_SOURCE);
            let arom = read_i32_at(rec, spike_offsets::AROMATIC_TYPE);
            if px < summary.bbox_min[0] { summary.bbox_min[0] = px; }
            if py < summary.bbox_min[1] { summary.bbox_min[1] = py; }
            if pz < summary.bbox_min[2] { summary.bbox_min[2] = pz; }
            if px > summary.bbox_max[0] { summary.bbox_max[0] = px; }
            if py > summary.bbox_max[1] { summary.bbox_max[1] = py; }
            if pz > summary.bbox_max[2] { summary.bbox_max[2] = pz; }
            if intensity < summary.intensity_min { summary.intensity_min = intensity; }
            if intensity > summary.intensity_max { summary.intensity_max = intensity; }
            intensity_sum += intensity as f64;
            match source {
                1 => summary.n_uv += 1,
                2 => summary.n_lif += 1,
                _ => summary.n_other_source += 1,
            }
            match arom {
                0 => summary.n_aromatic_trp += 1,
                1 => summary.n_aromatic_tyr += 1,
                2 => summary.n_aromatic_phe += 1,
                3 => summary.n_aromatic_ss += 1,
                _ => summary.n_aromatic_none += 1,
            }
            // Voxel bin (sparse hashmap).
            let vx = (px * inv_voxel).floor() as i32;
            let vy = (py * inv_voxel).floor() as i32;
            let vz = (pz * inv_voxel).floor() as i32;
            *voxel_grid.entry((vx, vy, vz)).or_insert(0) += 1;
            summary.n_spikes += 1;
        }
        remaining -= batch;
    }
    if header.record_count != expected_record_count {
        anyhow::bail!(
            "spikes record_count mismatch: header {} vs manifest {}",
            header.record_count, expected_record_count
        );
    }
    if summary.n_spikes > 0 {
        summary.intensity_mean = intensity_sum / summary.n_spikes as f64;
    }
    summary.elapsed_ms = start.elapsed().as_millis() as u64;
    Ok(summary)
}

// ─── Voxel-density peak extraction (HONEST candidates, no ML rerank) ───────

#[derive(Debug, Serialize)]
struct DensityPeak {
    peak_id: u32,
    voxel_index: [i32; 3],
    centroid_a: [f32; 3],
    spike_count: u32,
    rank: u32,
    kind: &'static str,
}

fn extract_top_density_peaks(
    voxel_grid: &std::collections::HashMap<(i32, i32, i32), u32>,
    voxel_size_a: f32,
    top_k: usize,
) -> Vec<DensityPeak> {
    let mut entries: Vec<((i32, i32, i32), u32)> =
        voxel_grid.iter().map(|(k, v)| (*k, *v)).collect();
    entries.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    entries
        .into_iter()
        .take(top_k)
        .enumerate()
        .map(|(idx, ((vx, vy, vz), count))| DensityPeak {
            peak_id: idx as u32,
            voxel_index: [vx, vy, vz],
            centroid_a: [
                (vx as f32 + 0.5) * voxel_size_a,
                (vy as f32 + 0.5) * voxel_size_a,
                (vz as f32 + 0.5) * voxel_size_a,
            ],
            spike_count: count,
            rank: idx as u32,
            kind: "voxel_density_peak",
        })
        .collect()
}

// ─── Filename → stream_id parser (sanity check vs header) ──────────────────

fn parse_stream_id_from_filename(p: &Path) -> Option<u32> {
    let name = p.file_name()?.to_string_lossy();
    // patterns: "..._streamN_..." or "..._stream0N_..."
    let idx = name.find("_stream")? + "_stream".len();
    let tail = &name[idx..];
    // collect digits
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().ok()
}

// ─── Topology loader (minimal — for residue centers + ligand reference) ──

#[derive(Debug, Deserialize)]
struct TopologyMin {
    n_atoms: usize,
    n_residues: usize,
    positions: Vec<f32>,
    residue_ids: Vec<i32>,
    residue_names: Vec<String>,
    ca_indices: Vec<i32>,
}

impl TopologyMin {
    fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)
            .with_context(|| format!("read topology {}", path.display()))?;
        let t: TopologyMin = serde_json::from_str(&s)
            .with_context(|| format!("parse topology {}", path.display()))?;
        anyhow::ensure!(
            t.positions.len() == 3 * t.n_atoms,
            "topology positions len {} != 3 * n_atoms {}",
            t.positions.len(),
            t.n_atoms
        );
        Ok(t)
    }

    /// CA position per residue_id. Returns Vec<Option<[f32;3]>> indexed by residue_id.
    fn ca_positions_by_residue(&self) -> Vec<Option<[f32; 3]>> {
        let mut out: Vec<Option<[f32; 3]>> = vec![None; self.n_residues];
        for (rid, &ca_idx) in self.ca_indices.iter().enumerate() {
            if ca_idx < 0 || (ca_idx as usize) >= self.n_atoms {
                continue;
            }
            let i = (ca_idx as usize) * 3;
            out[rid] = Some([self.positions[i], self.positions[i + 1], self.positions[i + 2]]);
        }
        out
    }
}

// ─── Ground truth loader (for accuracy validation) ──────────────────────────

#[derive(Debug, Deserialize)]
struct GroundTruth {
    pdb_id: Option<String>,
    target_chain: Option<String>,
    valid_for_dcc_validation: bool,
    skip_reason: Option<String>,
    ligand_centroid: Option<[f32; 3]>,
    #[serde(default)]
    #[allow(dead_code)]
    ligand: Option<serde_json::Value>,
}

// ─── KCC v2-full reader (PRKCC001) ──────────────────────────────────────────

#[derive(Debug, Default, Clone)]
struct KccPerStream {
    stream_id: u32,
    n_residues: usize,
    /// Per-residue active_causal counts (used to identify driver residues).
    active_causal: Vec<u32>,
}

fn read_kcc_v2full(path: &Path) -> Result<KccPerStream> {
    let f = File::open(path).with_context(|| format!("open kcc {}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 16, f);
    let h = read_envelope_header(&mut r)?;
    anyhow::ensure!(&h.magic == b"PRKCC001", "kcc bad magic");
    // Payload layout:
    //   n_residues u64
    //   field_count u64
    //   for each field: { name_len u64, name bytes, dtype u8,
    //                     section_size u64, data section_size bytes }
    let n_res = read_u64_le(&mut r)? as usize;
    let field_count = read_u64_le(&mut r)?;
    let mut active_causal: Vec<u32> = Vec::new();
    for _ in 0..field_count {
        let name_len = read_u64_le(&mut r)? as usize;
        let name = String::from_utf8(read_bytes(&mut r, name_len)?)?;
        let mut dtype_buf = [0u8; 1];
        r.read_exact(&mut dtype_buf)?;
        let dtype = dtype_buf[0];
        let section_size = read_u64_le(&mut r)? as usize;
        let data = read_bytes(&mut r, section_size)?;
        if name == "active_causal" && dtype == 2 /* u32 */ && data.len() == n_res * 4 {
            let mut v = vec![0u32; n_res];
            for k in 0..n_res {
                let mut b = [0u8; 4];
                b.copy_from_slice(&data[k * 4..k * 4 + 4]);
                v[k] = u32::from_le_bytes(b);
            }
            active_causal = v;
        }
        // Other fields (temporal_corr, direction_score, ...) are read and
        // discarded for this commit — only active_causal is needed for
        // driver-residue identification.
    }
    Ok(KccPerStream {
        stream_id: h.stream_id,
        n_residues: n_res,
        active_causal,
    })
}

// ─── Per-candidate evidence accumulator ─────────────────────────────────────

#[derive(Debug, Default, Clone)]
struct CandidateEvidence {
    seed_peak_id: u32,
    seed_voxel: [i32; 3],
    seed_centroid_a: [f32; 3],
    /// Total spikes attributed to this candidate's region.
    n_spikes: u64,
    intensity_sum: f64,
    /// Weighted position sum for whole_site centroid.
    pos_sum_x: f64,
    pos_sum_y: f64,
    pos_sum_z: f64,
    /// Per-stream spike counts.
    per_stream_spikes: std::collections::HashMap<u32, u64>,
    /// Set of unique voxels touched per stream (for per_stream_voxel_counts).
    per_stream_voxels: std::collections::HashMap<u32, std::collections::HashSet<(i32, i32, i32)>>,
    /// Timestep range.
    timestep_min: i32,
    timestep_max: i32,
    /// Recurrence: number of unique 1000-step chunks observed.
    chunk_set: std::collections::HashSet<i32>,
    /// Phase histogram (10 bins of phase_bits 0-1023).
    phase_hist: [u32; 10],
    /// Rayleigh r-stat accumulators.
    phase_cos_sum: f64,
    phase_sin_sum: f64,
    /// Hot phase (phase_bits >= 512) position sum.
    hot_pos_sum: [f64; 3],
    n_hot: u64,
    /// Cold phase (phase_bits < 512) position sum.
    cold_pos_sum: [f64; 3],
    n_cold: u64,
    /// Source / aromatic counts.
    n_uv: u64,
    n_lif: u64,
    n_other_source: u64,
    n_arom: [u64; 5], // [TRP, TYR, PHE, SS, none]
    /// Residue support: residue_id → count.
    residue_support: std::collections::HashMap<i32, u32>,
    /// Refined sub-chunk persistence — bin_id (= timestep / persistence_bin_steps) → spikes.
    bin_counts: std::collections::HashMap<i32, u64>,
}

impl CandidateEvidence {
    fn new(seed_peak_id: u32, seed_voxel: [i32; 3], seed_centroid_a: [f32; 3]) -> Self {
        Self {
            seed_peak_id,
            seed_voxel,
            seed_centroid_a,
            timestep_min: i32::MAX,
            timestep_max: i32::MIN,
            ..Default::default()
        }
    }
}

// ─── Materialized site (output JSON shape) ──────────────────────────────────

#[derive(Debug, Serialize)]
struct CentroidManifold {
    whole_site: Option<[f32; 3]>,
    lining_mass: Option<[f32; 3]>,
    causal_driver: Option<[f32; 3]>,
    hot_phase: Option<[f32; 3]>,
    cold_phase: Option<[f32; 3]>,
    return_phase: Option<[f32; 3]>,
    ligand_adjacent: Option<[f32; 3]>,
}

#[derive(Debug, Serialize)]
struct StreamSupport {
    n_streams_supporting: u32,
    supporting_streams: Vec<u32>,
    per_stream_spike_counts: serde_json::Map<String, serde_json::Value>,
    per_stream_voxel_counts: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct TemporalSupport {
    first_observed_step: Option<i32>,
    last_observed_step: Option<i32>,
    /// "Recurrence" = unique 1000-step chunks observed.
    recurrence_count: u32,
    chunk_support: Vec<i32>,
    n_phase_bins_nonzero: u32,
}

#[derive(Debug, Serialize)]
struct PhaseSupport {
    phase_bits_histogram: [u32; 10],
    rayleigh_r_stat: f64,
    /// Mean phase angle in radians; null if rayleigh_r_stat ≈ 0.
    mean_phase_radians: Option<f64>,
    n_hot: u64,
    n_cold: u64,
}

#[derive(Debug, Serialize)]
struct So3Support {
    /// Phase coherence (Rayleigh r) — duplicated from PhaseSupport for
    /// SO(3)-block ergonomics. CCNS phase angle is a rotation in protocol
    /// phase space, so concentrated phase is a valid orientation-coherence
    /// signal.
    phase_coherence_rayleigh_r: f64,
    /// Per-source-class breakdown: separate r for UV vs LIF if useful.
    /// In this commit, single value across all spikes.
    orientation_coherence_score: Option<f64>,
    /// Reserved fields for future extension once warp_matrix / forces /
    /// asc_vectors parsers are implemented.
    local_frame_stability: Option<f64>,
    rotation_dispersion: Option<f64>,
    warp_matrix_support: &'static str,    // "deferred_format_not_parsed"
    asc_vector_support: &'static str,     // "deferred_format_not_parsed"
    force_direction_support: &'static str, // "deferred_format_not_parsed"
    evidence_files: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FieldCompleteness {
    spikes: &'static str,
    phase_bits: &'static str,
    warp_matrix: &'static str,
    asc_vectors: &'static str,
    forces: &'static str,
    signal_grid: &'static str,
    kcc: &'static str,
}

#[derive(Debug, Serialize)]
struct SpatiotemporalSo3Evidence {
    status: &'static str, // "available" | "partial" | "missing"
    stream_support: StreamSupport,
    temporal_support: TemporalSupport,
    phase_support: PhaseSupport,
    so3_support: So3Support,
    centroid_manifold: CentroidManifold,
    field_completeness: FieldCompleteness,
}

#[derive(Debug, Serialize)]
struct Provenance {
    source_manifest: String,
    run_id: String,
    target: String,
    seed_voxel_index: [i32; 3],
    seed_density_peak_rank: u32,
    spikes_files_consumed: Vec<String>,
    kcc_files_consumed: Vec<String>,
    topology_input: String,
}

#[derive(Debug, Serialize)]
struct MaterializedSite {
    site_id: String,
    rank: u32,
    materialization_status: &'static str, // see MaterializationLevel
    materialization_level: &'static str,
    centroid_xyz: [f32; 3],
    centroid_manifold: CentroidManifold,
    representative_residues: Vec<i32>,
    lining_residues: Vec<i32>,
    driver_residues: Vec<i32>,
    n_spikes: u64,
    intensity_sum: f64,
    spike_support: serde_json::Value,
    stream_support: StreamSupport,
    phase_support: PhaseSupport,
    spatiotemporal_so3_evidence: SpatiotemporalSo3Evidence,
    field_completeness: FieldCompleteness,
    provenance: Provenance,
    limitations: Vec<&'static str>,
    accuracy: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct NonMaterializedCandidate {
    seed_peak_id: u32,
    seed_voxel: [i32; 3],
    seed_centroid_a: [f32; 3],
    n_spikes: u64,
    n_streams_supporting: u32,
    materialization_level: &'static str,
    blocking_reasons: Vec<&'static str>,
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn vec3_div(p: [f64; 3], n: f64) -> [f32; 3] {
    [(p[0] / n) as f32, (p[1] / n) as f32, (p[2] / n) as f32]
}

fn dist3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Gini coefficient for a vector of non-negative counts (low gini = balanced).
fn gini_coefficient(counts: &[u64]) -> f64 {
    if counts.is_empty() {
        return 0.0;
    }
    let n = counts.len() as f64;
    let total: f64 = counts.iter().map(|&c| c as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut sorted: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cumsum = 0.0;
    let mut weighted_sum = 0.0;
    for (i, c) in sorted.iter().enumerate() {
        cumsum += *c;
        weighted_sum += cumsum;
    }
    let gini = (n + 1.0 - 2.0 * weighted_sum / total) / n;
    gini.clamp(0.0, 1.0)
}

/// Jaccard overlap between two integer sets.
fn jaccard_overlap(a: &[i32], b: &[i32]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let sa: std::collections::HashSet<i32> = a.iter().copied().collect();
    let sb: std::collections::HashSet<i32> = b.iter().copied().collect();
    let inter = sa.intersection(&sb).count();
    let union = sa.union(&sb).count();
    if union == 0 {
        return 0.0;
    }
    inter as f32 / union as f32
}

// ─── Stage-A raw peak (per top-K voxel, with full evidence accumulator) ────

#[derive(Debug, Clone)]
struct RawPeak {
    raw_peak_id: u32,
    voxel: [i32; 3],
    seed_centroid_a: [f32; 3],
    /// Detailed evidence accumulator from the spike re-pass.
    evidence: CandidateEvidence,
    /// Top-12 lining residues (cached for consolidation Jaccard checks).
    top12_lining_residues: Vec<i32>,
}

// ─── Stage-B consolidated candidate region ─────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
struct CandidateRegion {
    region_id: u32,
    representative_raw_peak_id: u32,
    merged_raw_peak_ids: Vec<u32>,
    aggregate_centroid_a: [f32; 3],
    region_radius_a: f32,
    /// Aggregate evidence (sums of merged raw peaks).
    n_spikes: u64,
    intensity_sum: f64,
    pos_sum_x: f64,
    pos_sum_y: f64,
    pos_sum_z: f64,
    per_stream_spikes: std::collections::HashMap<u32, u64>,
    per_stream_voxels: std::collections::HashMap<u32, std::collections::HashSet<(i32, i32, i32)>>,
    timestep_min: i32,
    timestep_max: i32,
    chunk_set: std::collections::HashSet<i32>,
    /// Refined sub-chunk persistence bin set (using args.persistence_bin_steps).
    persistence_bin_set: std::collections::HashSet<i32>,
    /// Per-bin spike counts (for burstiness scoring).
    bin_counts: std::collections::HashMap<i32, u64>,
    phase_hist: [u32; 10],
    phase_cos_sum: f64,
    phase_sin_sum: f64,
    hot_pos_sum: [f64; 3],
    n_hot: u64,
    cold_pos_sum: [f64; 3],
    n_cold: u64,
    n_uv: u64,
    n_lif: u64,
    n_other_source: u64,
    n_arom: [u64; 5],
    residue_support: std::collections::HashMap<i32, u32>,
    merge_reasons: Vec<String>,
    /// First and last steps observed (for temporal status). When refined
    /// timesteps are not available (e.g., headers_only), this is left
    /// at its sentinel min/max.
    centroid_spread_a: f32,
    residue_shell_overlap_with_rep: f32,
}

// ─── PRISM-native score components (every factor is emitted) ───────────────

#[derive(Debug, Default, Serialize, Clone)]
struct PrismScoreComponents {
    density_score: f64,
    stream_balance_factor: f64,
    temporal_persistence_factor: f64,
    residue_shell_plausibility_factor: f64,
    centroid_manifold_consistency_factor: f64,
    kcc_driver_factor: f64,
    field_completeness_factor: f64,
    duplicate_region_factor: f64,
    tail_penalty_factor: f64,
    final_prism_score: f64,
    score_explanation: Vec<String>,
}

// ─── Residue-shell plausibility helpers ────────────────────────────────────

#[derive(Debug, Default, Serialize, Clone)]
struct ResidueShellPlausibility {
    residue_shell_size: u32,
    residue_diversity_score: f64,
    /// Mean pairwise distance between supporting residue CAs (Å); lower = more compact.
    compactness_a: Option<f64>,
    /// Heuristic enclosure proxy (function of compactness × residue count).
    enclosure_proxy: Option<f64>,
    tail_penalty: f64,
    terminal_residue_fraction: f64,
    backbone_exposure_proxy: Option<f64>,
    aromatic_count: u32,
    hydrophobic_count: u32,
    polar_count: u32,
    catalytic_like_pocket_support: f64,
    residue_shell_reasoning: Vec<String>,
}

const HYDROPHOBIC: &[&str] = &[
    "ALA", "VAL", "LEU", "ILE", "PRO", "MET", "TRP", "PHE", "GLY",
];
const POLAR: &[&str] = &[
    "SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "ASP", "GLU", "LYS", "ARG",
];
const AROMATIC: &[&str] = &["PHE", "TYR", "TRP", "HIS"];

// ─── Manifold consistency block ────────────────────────────────────────────

#[derive(Debug, Default, Serialize, Clone)]
struct ManifoldConsistency {
    d_whole_lining_a: Option<f32>,
    d_whole_driver_a: Option<f32>,
    d_hot_cold_a: Option<f32>,
    manifold_dispersion_a: Option<f32>,
    manifold_consistency_score: f64,
    n_views_present: u32,
}

// ─── Refined temporal support ──────────────────────────────────────────────

#[derive(Debug, Default, Serialize, Clone)]
struct RefinedTemporalSupport {
    persistence_bin_size_steps: i32,
    persistence_bins_observed: u32,
    persistence_bins_total_possible: u32,
    persistence_fraction: f64,
    burstiness_score: f64,
    temporal_entropy: f64,
    /// `coarse_non_discriminative` when timesteps are unavailable or all
    /// candidates saturate the same bins; otherwise `available`.
    temporal_support_status: &'static str,
}

// ─── KCC driver block (active in scoring) ──────────────────────────────────

#[derive(Debug, Default, Serialize, Clone)]
struct KccDriverBlock {
    driver_residue_count: u32,
    driver_residue_fraction_of_shell: f64,
    driver_centroid_present: bool,
    driver_centroid_distance_to_whole_site_a: Option<f32>,
    active_causal_support_sum: u64,
    kcc_field_completeness: &'static str,
    kcc_driver_score: f64,
    kcc_driver_reasoning: Vec<String>,
}

// ─── Stream balance block ──────────────────────────────────────────────────

#[derive(Debug, Default, Serialize, Clone)]
struct StreamBalance {
    n_streams_supporting: u32,
    per_stream_spike_counts: serde_json::Map<String, serde_json::Value>,
    gini_coefficient: f64,
    stream_balance_score: f64,
    per_stream_support_balance_status: &'static str,
}

// ─── Duplicate-cluster report record ───────────────────────────────────────

#[derive(Debug, Serialize)]
struct DuplicateClusterRecord {
    duplicate_cluster_id: u32,
    kept_region_id: u32,
    kept_centroid_a: [f32; 3],
    suppressed_raw_peak_ids: Vec<u32>,
    n_suppressed: u32,
    centroid_spread_a: f32,
    residue_shell_overlap_with_rep: f32,
    merge_reasons: Vec<String>,
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();
    let log_level = if args.verbose { "info" } else { "warn" };
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", log_level);
    }
    env_logger::init();

    eprintln!("prism-materialize-sites — Path B skeleton");
    eprintln!("manifest: {}", args.manifest.display());

    let manifest_text = std::fs::read_to_string(&args.manifest)
        .with_context(|| format!("read manifest {}", args.manifest.display()))?;
    let manifest: Manifest = serde_json::from_str(&manifest_text)
        .with_context(|| format!("parse manifest {}", args.manifest.display()))?;

    if manifest.schema_kind != "md_evidence_manifest" {
        anyhow::bail!(
            "manifest schema_kind = {:?}, expected \"md_evidence_manifest\"",
            manifest.schema_kind
        );
    }
    if manifest.schema_version != 1 {
        anyhow::bail!(
            "manifest schema_version = {}, this binary supports 1",
            manifest.schema_version
        );
    }

    let output_dir = args
        .output_dir
        .clone()
        .or_else(|| args.manifest.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("create output dir {}", output_dir.display()))?;

    eprintln!("output dir: {}", output_dir.display());
    eprintln!(
        "run_id: {}  target: {}  streams: {}/{}",
        manifest.run_id,
        manifest.target,
        manifest.streams_serialized,
        manifest.stream_count
    );
    eprintln!("topology: {}", manifest.topology_input);
    eprintln!(
        "compression={}  checksum={}  endian={}",
        manifest.compression_status, manifest.checksum_algorithm, manifest.endian
    );
    if manifest.serialization_failure.is_some() {
        eprintln!(
            "WARNING: manifest reports serialization_failure: {:?}",
            manifest.serialization_failure
        );
    }

    // ─── Validate every present binary artifact ────────────────────────────
    let mut validations: Vec<ArtifactValidation> = Vec::new();
    let mut spike_artifacts: Vec<(u32, String, u64)> = Vec::new(); // stream_id, path, record_count

    for art in manifest.artifacts.iter() {
        if !art.present {
            continue;
        }
        let p = PathBuf::from(&art.path);
        let (magic, trailer): (Option<[u8; 8]>, Option<[u8; 8]>) = match art.kind.as_str() {
            "spikes"      => (Some(*b"PRSPK001"), Some(*b"PRSPKEND")),
            "signal_grid" => (Some(*b"PRSGD001"), Some(*b"PRSGDEND")),
            "kcc_v2full"  => (Some(*b"PRKCC001"), Some(*b"PRKCCEND")),
            // rayon_scope_emit (warp_matrix, forces, etc.) — no PRISM envelope.
            _ => (None, None),
        };
        let (Some(magic), Some(trailer)) = (magic, trailer) else {
            continue;
        };
        let filename_stream_id = parse_stream_id_from_filename(&p);
        match validate_artifact(
            &p,
            &magic,
            &trailer,
            &art.kind,
            art.record_count,
            art.checksum_fnv1a_64_hex.as_deref(),
            &manifest.run_id,
            &manifest.target,
            filename_stream_id,
            args.headers_only,
        ) {
            Ok((v, _hdr)) => {
                if v.valid {
                    eprintln!(
                        "  ✓ {} kind={} stream={} records={} stride={}",
                        p.display(), art.kind, v.stream_id, v.record_count, v.byte_stride
                    );
                    if art.kind == "spikes" {
                        spike_artifacts.push((v.stream_id, art.path.clone(), v.record_count));
                    }
                } else {
                    eprintln!(
                        "  ✗ {} kind={} INVALID: {}",
                        p.display(), art.kind, v.failure_reason.as_deref().unwrap_or("(unknown)")
                    );
                }
                validations.push(v);
            }
            Err(e) => {
                eprintln!("  ✗ {} kind={} READ-ERROR: {}", p.display(), art.kind, e);
                validations.push(ArtifactValidation {
                    artifact_path: p.display().to_string(),
                    expected_kind: art.kind.clone(),
                    magic_observed: String::new(),
                    magic_match: false,
                    schema_version: 0,
                    endian_marker_hex: String::new(),
                    endian_ok_le: false,
                    stream_id: 0,
                    stream_id_matches_filename: None,
                    run_id_in_header: String::new(),
                    run_id_matches_manifest: false,
                    stem_in_header: String::new(),
                    stem_matches_manifest: false,
                    record_count: 0,
                    record_count_matches_manifest: None,
                    byte_stride: 0,
                    payload_size: 0,
                    declared_total_size: 0,
                    on_disk_size: 0,
                    size_consistency: false,
                    payload_checksum_fnv1a_64_hex: String::new(),
                    checksum_matches_manifest: None,
                    trailer_observed: String::new(),
                    trailer_match: false,
                    valid: false,
                    failure_reason: Some(format!("read error: {}", e)),
                });
            }
        }
    }

    // ─── Density-peak pass over spikes ─────────────────────────────────────
    let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), u32> =
        std::collections::HashMap::new();
    let mut stream_summaries: Vec<StreamSpikeSummary> = Vec::new();
    let mut total_processed: u64 = 0;
    let density_pass_skipped = args.headers_only;
    if !args.headers_only {
        for (stream_id, path, record_count) in spike_artifacts.iter() {
            eprintln!(
                "  [density] stream {} processing {} ({} records)…",
                stream_id, path, record_count
            );
            let p = PathBuf::from(path);
            // Read byte_stride from the header to feed process_spikes_stream
            // — we already validated the file above; reread just the header.
            let f = File::open(&p)?;
            let mut r = BufReader::new(f);
            let h = read_envelope_header(&mut r)?;
            drop(r);
            match process_spikes_stream(
                &p,
                *record_count,
                h.byte_stride,
                args.voxel_size_a,
                &mut voxel_grid,
            ) {
                Ok(s) => {
                    eprintln!(
                        "    ✓ stream {}: {} spikes, bbox=[{:.1},{:.1},{:.1}]→[{:.1},{:.1},{:.1}], elapsed={} ms",
                        s.stream_id, s.n_spikes,
                        s.bbox_min[0], s.bbox_min[1], s.bbox_min[2],
                        s.bbox_max[0], s.bbox_max[1], s.bbox_max[2],
                        s.elapsed_ms
                    );
                    total_processed += s.n_spikes;
                    stream_summaries.push(s);
                }
                Err(e) => {
                    eprintln!("    ✗ stream {}: {}", stream_id, e);
                }
            }
        }
    }

    let candidates: Vec<DensityPeak> = if !args.headers_only && !voxel_grid.is_empty() {
        extract_top_density_peaks(&voxel_grid, args.voxel_size_a, args.top_k_candidates)
    } else {
        Vec::new()
    };

    // ─── Emit materialization_report.json ──────────────────────────────────
    let report_path = output_dir.join("materialization_report.json");
    let report = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_materialization_report",
        "tool":           "prism-materialize-sites",
        "tool_version":   env!("CARGO_PKG_VERSION"),
        "manifest_path":  args.manifest.display().to_string(),
        "manifest_run_id": manifest.run_id,
        "manifest_target": manifest.target,
        "manifest_stream_count":     manifest.stream_count,
        "manifest_streams_serialized": manifest.streams_serialized,
        "manifest_total_spikes_md":  manifest.total_spikes_md,
        "manifest_v2_was_live":      manifest.v2_was_live,
        "manifest_serialization_failure": manifest.serialization_failure,
        "validations":   validations,
        "stream_summaries": stream_summaries,
        "voxel_grid_unique_voxels": voxel_grid.len() as u64,
        "voxel_size_a":  args.voxel_size_a,
        "spike_density_pass_skipped": density_pass_skipped,
        "total_spikes_processed":     total_processed,
        "matches_manifest_total":     total_processed == manifest.total_spikes_md,
        "ml_rerank_status":           "not_run",
        "ligsite_status":             "not_run",
        "ph_refinement_status":       "not_run",
        "find_neighbors_union_status":"not_run",
        "binding_sites_materialization_status": "not_emitted_skeleton_does_not_materialize",
        "path_b_required":            true,
    });
    {
        let f = File::create(&report_path)
            .with_context(|| format!("create {}", report_path.display()))?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &report)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", report_path.display());

    // ─── Emit site_candidates.json (HONEST density peaks, NOT binding sites) ──
    let candidates_path = output_dir.join("site_candidates.json");
    let candidates_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_site_candidates",
        "kind":           "voxel_density_peak",
        "kind_note":      "These are coarse spike-density voxel peaks. They are NOT materialized binding sites. \
                           Path B downstream stages (LIGSITE / PH / find_neighbors_union / XGB rerank) \
                           are required for site materialization.",
        "manifest_run_id": manifest.run_id,
        "manifest_target": manifest.target,
        "voxel_size_a":   args.voxel_size_a,
        "top_k":          args.top_k_candidates,
        "n_candidates":   candidates.len(),
        "n_spikes_basis": total_processed,
        "candidates":     candidates,
        "ml_reranked":    false,
        "ligsite_applied":false,
        "ph_refined":     false,
        "binding_sites_materialized": false,
    });
    {
        let f = File::create(&candidates_path)
            .with_context(|| format!("create {}", candidates_path.display()))?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &candidates_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", candidates_path.display());

    // ─── Emit validation_inputs.json (paths Path B downstream needs) ──────
    let validation_path = output_dir.join("validation_inputs.json");
    let validation_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_validation_inputs",
        "manifest_run_id": manifest.run_id,
        "manifest_target": manifest.target,
        "topology_input": manifest.topology_input,
        "topology_input_present": Path::new(&manifest.topology_input).exists(),
        "candidates_path": candidates_path.display().to_string(),
        "report_path":    report_path.display().to_string(),
        "downstream_stages_pending": [
            "rt_clustering_offline",
            "dynamic_ligsite",
            "cubical_ph_refinement",
            "xgb_rerank",
            "binding_sites_materialize",
            "ground_truth_validation"
        ],
        "ground_truth_input": serde_json::Value::Null,
        "ground_truth_status": "not_supplied_in_skeleton",
    });
    {
        let f = File::create(&validation_path)
            .with_context(|| format!("create {}", validation_path.display()))?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &validation_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", validation_path.display());

    // ─── Emit field_completeness_report.pathb.json ────────────────────────
    let fcr_path = output_dir.join("field_completeness_report.pathb.json");
    let fcr = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_field_completeness_report",
        "manifest_run_id": manifest.run_id,
        "skeleton_version": env!("CARGO_PKG_VERSION"),
        "fields": {
            "spikes_per_stream":            { "status": "available", "source": "PRSPK001 binaries on disk" },
            "signal_grid_per_stream":       { "status": "partial",   "source": "PRSGD001 binaries on disk (some streams source-None at engine teardown)" },
            "kcc_v2full_per_stream":        { "status": "partial",   "source": "PRKCC001 binaries on disk (some streams source-None at engine teardown)" },
            "voxel_density_candidates":     { "status": "available", "source": "this skeleton, voxel-binned spike density" },
            "all_stream_spikes_concat":     { "status": "implicit_streamed", "source": "spikes were streamed in-memory per file; not concatenated to disk" },
            "merged_kcc":                   { "status": "deferred", "source": "Path B follow-up: merge per-stream PRKCC001 files" },
            "merged_signal_grid":           { "status": "deferred", "source": "Path B follow-up: sum per-stream PRSGD001 files" },
            "rt_clustering_sites":          { "status": "deferred", "source": "Path B follow-up: find_neighbors_union over all_stream_spikes" },
            "ligsite_pockets":              { "status": "deferred", "source": "Path B follow-up: deterministic on topology+all_stream_spikes" },
            "cubical_ph_centroids":         { "status": "deferred", "source": "Path B follow-up: density grid + 0-dim PH" },
            "xgb_reranked_sites":           { "status": "deferred", "source": "Path B follow-up: XGB v3 model on cluster output" },
            "binding_sites_materialized":   { "status": "deferred", "source": "Path B follow-up: post-rerank materialization (NOT emitted by skeleton)" },
            "ground_truth_validation":      { "status": "deferred", "source": "Path B follow-up: requires explicit ground-truth input" },
            "transform_dag":                { "status": "absent", "reason": "v2_was_live=false in MD-only mode" }
        },
        "path_b_required": true,
        "binding_sites_materialized": false,
    });
    {
        let f = File::create(&fcr_path)
            .with_context(|| format!("create {}", fcr_path.display()))?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &fcr)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", fcr_path.display());

    // ═══════════════════════════════════════════════════════════════════════
    // MATERIALIZATION PASS — spatiotemporal SO(3) evidence per candidate
    // ═══════════════════════════════════════════════════════════════════════
    //
    // For each top-K density peak, scan spikes again and accumulate:
    //   * stream_support, temporal_support, phase_support
    //   * residue_support, source/aromatic histograms
    //   * hot/cold-phase position centroids
    // Then load per-stream KCC and identify driver residues. Then load
    // topology and compute residue-position-weighted lining_mass +
    // causal_driver centroids. Then load ground_truth.json and decide
    // whether DCC validation is appropriate.

    if args.headers_only || candidates.is_empty() {
        eprintln!("[materialize] skipped (headers_only or no candidates)");
        // Emit empty materialized stubs honestly.
        let materialized_path = output_dir.join("binding_sites.materialized.json");
        let stub = serde_json::json!({
            "schema_version": 1,
            "status":         "no_candidates_or_headers_only",
            "target":         manifest.target,
            "run_id":         manifest.run_id,
            "source_manifest": args.manifest.display().to_string(),
            "site_count":     0,
            "binding_sites":  Vec::<serde_json::Value>::new(),
            "non_materialized_candidates": Vec::<serde_json::Value>::new(),
            "missing_fields": ["spikes_pass_skipped"],
            "path_b_required": true,
        });
        if let Ok(f) = File::create(&materialized_path) {
            let mut bw = std::io::BufWriter::new(f);
            let _ = serde_json::to_writer_pretty(&mut bw, &stub);
            use std::io::Write as _;
            let _ = bw.flush();
            eprintln!("✓ wrote {} (stub)", materialized_path.display());
        }
        return Ok(());
    }

    // ─── Build voxel→candidate-id lookup (radius 4 Å = 2 voxels at 2.0 Å) ─
    let radius_voxels: i32 = (4.0_f32 / args.voxel_size_a).ceil() as i32;
    let mut voxel_to_cands: std::collections::HashMap<(i32, i32, i32), Vec<u32>> =
        std::collections::HashMap::new();
    for (cand_idx, peak) in candidates.iter().enumerate() {
        for dx in -radius_voxels..=radius_voxels {
            for dy in -radius_voxels..=radius_voxels {
                for dz in -radius_voxels..=radius_voxels {
                    let key = (
                        peak.voxel_index[0] + dx,
                        peak.voxel_index[1] + dy,
                        peak.voxel_index[2] + dz,
                    );
                    voxel_to_cands.entry(key).or_default().push(cand_idx as u32);
                }
            }
        }
    }
    eprintln!(
        "[materialize] voxel→candidate lookup built: {} voxel keys (radius={} voxels)",
        voxel_to_cands.len(), radius_voxels
    );

    // ─── Pass 2 — accumulate per-candidate evidence ─────────────────────────
    let mut evidence: Vec<CandidateEvidence> = candidates
        .iter()
        .map(|p| CandidateEvidence::new(p.peak_id, p.voxel_index, p.centroid_a))
        .collect();
    let inv_voxel = 1.0_f32 / args.voxel_size_a;

    for (stream_id, path, record_count) in spike_artifacts.iter() {
        let p = PathBuf::from(path);
        let f = File::open(&p)?;
        let mut r = BufReader::with_capacity(1 << 20, f);
        let h = read_envelope_header(&mut r)?;
        let stride = h.byte_stride as usize;
        anyhow::ensure!(stride >= 84, "spikes stride {} too small", stride);
        let mut remaining = h.record_count;
        let records_per_batch: u64 = 1 << 16;
        let mut buf = vec![0u8; (records_per_batch * h.byte_stride) as usize];
        while remaining > 0 {
            let batch = std::cmp::min(remaining, records_per_batch);
            let bytes = (batch * h.byte_stride) as usize;
            r.read_exact(&mut buf[..bytes])?;
            for k in 0..(batch as usize) {
                let off = k * stride;
                let rec = &buf[off..off + stride];
                let px = read_f32_at(rec, spike_offsets::POSITION_X);
                let py = read_f32_at(rec, spike_offsets::POSITION_Y);
                let pz = read_f32_at(rec, spike_offsets::POSITION_Z);
                let vx = (px * inv_voxel).floor() as i32;
                let vy = (py * inv_voxel).floor() as i32;
                let vz = (pz * inv_voxel).floor() as i32;
                let Some(cand_ids) = voxel_to_cands.get(&(vx, vy, vz)) else {
                    continue;
                };
                let intensity = read_f32_at(rec, spike_offsets::INTENSITY);
                let timestep = read_i32_at(rec, 0);
                let source = read_i32_at(rec, spike_offsets::SPIKE_SOURCE);
                let arom = read_i32_at(rec, spike_offsets::AROMATIC_TYPE);
                // phase_bits at offset 92 (last u32 in 96-byte struct)
                let phase_bits: u32 = if stride >= 96 {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&rec[92..96]);
                    u32::from_le_bytes(b)
                } else {
                    0
                };
                // nearby_residues[8] starts at offset 24 (after timestep+voxel_idx+pos[3]+intensity)
                let mut nearby_res = [0i32; 8];
                let n_res = read_i32_at(rec, 56).max(0).min(8) as usize;
                for j in 0..n_res {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&rec[24 + j * 4..24 + j * 4 + 4]);
                    nearby_res[j] = i32::from_le_bytes(b);
                }
                let phase_angle = (phase_bits as f64 / 1024.0) * 2.0 * std::f64::consts::PI;
                let cs = phase_angle.cos();
                let sn = phase_angle.sin();
                let phase_bin = ((phase_bits as usize) * 10 / 1024).min(9);
                let chunk_id = timestep / 1000;
                let persistence_bin_id = timestep / args.persistence_bin_steps.max(1);
                for &ci in cand_ids.iter() {
                    let ev = &mut evidence[ci as usize];
                    ev.n_spikes += 1;
                    ev.intensity_sum += intensity as f64;
                    ev.pos_sum_x += px as f64 * intensity as f64;
                    ev.pos_sum_y += py as f64 * intensity as f64;
                    ev.pos_sum_z += pz as f64 * intensity as f64;
                    *ev.per_stream_spikes.entry(*stream_id).or_insert(0) += 1;
                    ev.per_stream_voxels.entry(*stream_id).or_default().insert((vx, vy, vz));
                    if timestep < ev.timestep_min { ev.timestep_min = timestep; }
                    if timestep > ev.timestep_max { ev.timestep_max = timestep; }
                    ev.chunk_set.insert(chunk_id);
                    *ev.bin_counts.entry(persistence_bin_id).or_insert(0) += 1;
                    ev.phase_hist[phase_bin] += 1;
                    ev.phase_cos_sum += cs;
                    ev.phase_sin_sum += sn;
                    if phase_bits >= 512 {
                        ev.hot_pos_sum[0] += px as f64;
                        ev.hot_pos_sum[1] += py as f64;
                        ev.hot_pos_sum[2] += pz as f64;
                        ev.n_hot += 1;
                    } else {
                        ev.cold_pos_sum[0] += px as f64;
                        ev.cold_pos_sum[1] += py as f64;
                        ev.cold_pos_sum[2] += pz as f64;
                        ev.n_cold += 1;
                    }
                    match source {
                        1 => ev.n_uv += 1,
                        2 => ev.n_lif += 1,
                        _ => ev.n_other_source += 1,
                    }
                    let arom_bin = match arom { 0 => 0, 1 => 1, 2 => 2, 3 => 3, _ => 4 };
                    ev.n_arom[arom_bin] += 1;
                    for j in 0..n_res {
                        *ev.residue_support.entry(nearby_res[j]).or_insert(0) += 1;
                    }
                }
            }
            remaining -= batch;
        }
        let _ = record_count;
        let total_attributed: u64 = evidence.iter().map(|e| e.per_stream_spikes.get(stream_id).copied().unwrap_or(0)).sum();
        eprintln!("[materialize]  stream {} contributed {} attributed spikes", stream_id, total_attributed);
    }

    // ─── Pass 3 — load per-stream KCC, identify driver residues ─────────────
    let mut kcc_per_stream: Vec<KccPerStream> = Vec::new();
    let mut kcc_files_consumed: Vec<String> = Vec::new();
    for art in manifest.artifacts.iter() {
        if !art.present || art.kind != "kcc_v2full" {
            continue;
        }
        let p = PathBuf::from(&art.path);
        match read_kcc_v2full(&p) {
            Ok(k) => {
                kcc_per_stream.push(k);
                kcc_files_consumed.push(art.path.clone());
            }
            Err(e) => {
                eprintln!("[materialize] kcc read failed for {}: {}", p.display(), e);
            }
        }
    }
    // Driver residues = residues with sum(active_causal) > 0 across streams.
    let mut driver_active_causal_sum: std::collections::HashMap<usize, u64> =
        std::collections::HashMap::new();
    for k in kcc_per_stream.iter() {
        for (rid, &count) in k.active_causal.iter().enumerate() {
            if count > 0 {
                *driver_active_causal_sum.entry(rid).or_insert(0) += count as u64;
            }
        }
    }
    eprintln!(
        "[materialize] {} streams with KCC; {} driver residues (active_causal>0)",
        kcc_per_stream.len(),
        driver_active_causal_sum.len()
    );

    // ─── Pass 4 — load topology for residue centers (CA-based) ──────────────
    let topo_path = PathBuf::from(&manifest.topology_input);
    let (topology, residue_centers): (Option<TopologyMin>, Vec<Option<[f32; 3]>>) =
        match TopologyMin::load(&topo_path) {
            Ok(t) => {
                let centers = t.ca_positions_by_residue();
                eprintln!(
                    "[materialize] topology loaded: {} atoms, {} residues, {} CA centers resolved",
                    t.n_atoms, t.n_residues,
                    centers.iter().filter(|c| c.is_some()).count()
                );
                (Some(t), centers)
            }
            Err(e) => {
                eprintln!("[materialize] topology load FAILED: {} (residue-based centroids skipped)", e);
                (None, Vec::new())
            }
        };

    // ─── Pass 5 — load ground truth and decide validation status ────────────
    let gt_path = output_dir.join(format!("{}_ground_truth.json", manifest.target));
    let (ground_truth, gt_status, gt_skip_reason): (Option<GroundTruth>, &'static str, Option<String>) =
        match std::fs::read_to_string(&gt_path) {
            Ok(s) => match serde_json::from_str::<GroundTruth>(&s) {
                Ok(gt) => {
                    let valid = gt.valid_for_dcc_validation && gt.ligand_centroid.is_some();
                    let status = if valid { "available" } else { "reference_unavailable" };
                    let reason = gt.skip_reason.clone();
                    (Some(gt), status, reason)
                }
                Err(e) => {
                    eprintln!("[materialize] ground_truth.json parse failed: {}", e);
                    (None, "reference_unavailable", Some(format!("parse_error: {}", e)))
                }
            },
            Err(_) => (None, "reference_unavailable", Some("ground_truth_file_missing".to_string())),
        };
    eprintln!("[materialize] ground_truth status: {}", gt_status);

    // ═══════════════════════════════════════════════════════════════════════
    // PRISM-NATIVE PIPELINE v2 (commit dc175006+1):
    //   Stage A — raw peaks (per top-K voxel)
    //   Stage B — consolidation by centroid distance + residue-shell Jaccard
    //   Pass C — per-region: residue-shell plausibility, manifold consistency,
    //            KCC driver, refined temporal, stream balance
    //   Pass D — composite PRISM-native score
    //   Pass E — promotion (no automatic promotion of all candidates)
    //   Pass F — post-hoc accuracy (Chain C reference is NOT used in ranking)
    // ═══════════════════════════════════════════════════════════════════════

    let min_spikes_per_stream: u64 = 50;
    let min_supporting_streams: u32 = 3;

    let mut spikes_files_consumed: Vec<String> = manifest
        .artifacts
        .iter()
        .filter(|a| a.present && a.kind == "spikes")
        .map(|a| a.path.clone())
        .collect();
    spikes_files_consumed.sort();

    let n_total_kcc_streams = kcc_per_stream.len();
    let signal_grid_present_any: bool = manifest
        .artifacts
        .iter()
        .any(|a| a.present && a.kind == "signal_grid");

    // ─── Stage A — convert per-peak evidence into RawPeak records ────────
    let mut raw_peaks: Vec<RawPeak> = evidence
        .into_iter()
        .enumerate()
        .map(|(idx, ev)| {
            let mut lining_pairs: Vec<(i32, u32)> = ev
                .residue_support
                .iter()
                .map(|(r, c)| (*r, *c))
                .collect();
            lining_pairs.sort_by(|a, b| b.1.cmp(&a.1));
            let top12: Vec<i32> = lining_pairs.iter().take(12).map(|(r, _)| *r).collect();
            RawPeak {
                raw_peak_id: idx as u32,
                voxel: ev.seed_voxel,
                seed_centroid_a: ev.seed_centroid_a,
                evidence: ev,
                top12_lining_residues: top12,
            }
        })
        .collect();
    // Sort by raw n_spikes descending so the highest-density peak seeds each region.
    raw_peaks.sort_by(|a, b| b.evidence.n_spikes.cmp(&a.evidence.n_spikes));
    eprintln!("[v2] Stage A — {} raw peaks (sorted by n_spikes)", raw_peaks.len());

    // ─── Stage B — greedy consolidation into candidate regions ───────────
    fn aggregate_centroid_of_evidence(ev: &CandidateEvidence) -> [f32; 3] {
        if ev.intensity_sum > 0.0 {
            vec3_div([ev.pos_sum_x, ev.pos_sum_y, ev.pos_sum_z], ev.intensity_sum)
        } else {
            ev.seed_centroid_a
        }
    }
    let mut regions: Vec<CandidateRegion> = Vec::new();
    let mut duplicate_records: Vec<DuplicateClusterRecord> = Vec::new();

    for raw in raw_peaks.iter() {
        let raw_centroid = aggregate_centroid_of_evidence(&raw.evidence);
        // Find first region this peak should merge into.
        let mut merge_into: Option<usize> = None;
        let mut merge_reasons: Vec<String> = Vec::new();
        let mut residue_overlap_observed: f32 = 0.0;
        for (ri, region) in regions.iter().enumerate() {
            let d = dist3(&raw_centroid, &region.aggregate_centroid_a);
            let mut rep_top12: Vec<i32> = region
                .residue_support
                .iter()
                .map(|(r, c)| (*r, *c))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|(r, c)| (r, c))
                .collect::<Vec<(i32, u32)>>()
                .iter()
                .take(12)
                .map(|(r, _)| *r)
                .collect();
            // The above pattern is ugly; rebuild more cleanly:
            let mut tmp: Vec<(i32, u32)> = region.residue_support.iter().map(|(r, c)| (*r, *c)).collect();
            tmp.sort_by(|a, b| b.1.cmp(&a.1));
            rep_top12 = tmp.into_iter().take(12).map(|(r, _)| r).collect();
            let jacc = jaccard_overlap(&raw.top12_lining_residues, &rep_top12);
            if d <= args.consolidation_radius_a {
                merge_into = Some(ri);
                merge_reasons.push(format!("centroid_distance_{:.2}A_le_{:.1}", d, args.consolidation_radius_a));
                residue_overlap_observed = jacc;
                break;
            }
            if jacc >= args.consolidation_jaccard {
                merge_into = Some(ri);
                merge_reasons.push(format!("residue_jaccard_{:.2}_ge_{:.2}_centroid_dist_{:.2}A", jacc, args.consolidation_jaccard, d));
                residue_overlap_observed = jacc;
                break;
            }
        }
        if let Some(ri) = merge_into {
            // Aggregate raw evidence into the existing region.
            let region = &mut regions[ri];
            region.merged_raw_peak_ids.push(raw.raw_peak_id);
            region.merge_reasons.extend(merge_reasons);
            region.n_spikes += raw.evidence.n_spikes;
            region.intensity_sum += raw.evidence.intensity_sum;
            region.pos_sum_x += raw.evidence.pos_sum_x;
            region.pos_sum_y += raw.evidence.pos_sum_y;
            region.pos_sum_z += raw.evidence.pos_sum_z;
            for (sid, cnt) in raw.evidence.per_stream_spikes.iter() {
                *region.per_stream_spikes.entry(*sid).or_insert(0) += cnt;
            }
            for (sid, vox) in raw.evidence.per_stream_voxels.iter() {
                let entry = region.per_stream_voxels.entry(*sid).or_default();
                for v in vox.iter() {
                    entry.insert(*v);
                }
            }
            if raw.evidence.timestep_min < region.timestep_min { region.timestep_min = raw.evidence.timestep_min; }
            if raw.evidence.timestep_max > region.timestep_max { region.timestep_max = raw.evidence.timestep_max; }
            for c in raw.evidence.chunk_set.iter() { region.chunk_set.insert(*c); }
            for (b, c) in raw.evidence.bin_counts.iter() {
                *region.bin_counts.entry(*b).or_insert(0) += c;
                region.persistence_bin_set.insert(*b);
            }
            for k in 0..10 { region.phase_hist[k] += raw.evidence.phase_hist[k]; }
            region.phase_cos_sum += raw.evidence.phase_cos_sum;
            region.phase_sin_sum += raw.evidence.phase_sin_sum;
            for k in 0..3 {
                region.hot_pos_sum[k] += raw.evidence.hot_pos_sum[k];
                region.cold_pos_sum[k] += raw.evidence.cold_pos_sum[k];
            }
            region.n_hot += raw.evidence.n_hot;
            region.n_cold += raw.evidence.n_cold;
            region.n_uv += raw.evidence.n_uv;
            region.n_lif += raw.evidence.n_lif;
            region.n_other_source += raw.evidence.n_other_source;
            for k in 0..5 { region.n_arom[k] += raw.evidence.n_arom[k]; }
            for (rid, cnt) in raw.evidence.residue_support.iter() {
                *region.residue_support.entry(*rid).or_insert(0) += cnt;
            }
            // Recompute aggregate centroid from updated sums.
            region.aggregate_centroid_a = if region.intensity_sum > 0.0 {
                vec3_div([region.pos_sum_x, region.pos_sum_y, region.pos_sum_z], region.intensity_sum)
            } else { region.aggregate_centroid_a };
            region.region_radius_a = region.region_radius_a.max(dist3(&raw_centroid, &region.aggregate_centroid_a));
            region.centroid_spread_a = region.region_radius_a;
            region.residue_shell_overlap_with_rep =
                region.residue_shell_overlap_with_rep.max(residue_overlap_observed);
        } else {
            // New region seeded from this raw peak.
            let region_id = regions.len() as u32;
            regions.push(CandidateRegion {
                region_id,
                representative_raw_peak_id: raw.raw_peak_id,
                merged_raw_peak_ids: vec![raw.raw_peak_id],
                aggregate_centroid_a: raw_centroid,
                region_radius_a: 0.0,
                n_spikes: raw.evidence.n_spikes,
                intensity_sum: raw.evidence.intensity_sum,
                pos_sum_x: raw.evidence.pos_sum_x,
                pos_sum_y: raw.evidence.pos_sum_y,
                pos_sum_z: raw.evidence.pos_sum_z,
                per_stream_spikes: raw.evidence.per_stream_spikes.clone(),
                per_stream_voxels: raw.evidence.per_stream_voxels.clone(),
                timestep_min: raw.evidence.timestep_min,
                timestep_max: raw.evidence.timestep_max,
                chunk_set: raw.evidence.chunk_set.clone(),
                persistence_bin_set: raw.evidence.bin_counts.keys().copied().collect(),
                bin_counts: raw.evidence.bin_counts.clone(),
                phase_hist: raw.evidence.phase_hist,
                phase_cos_sum: raw.evidence.phase_cos_sum,
                phase_sin_sum: raw.evidence.phase_sin_sum,
                hot_pos_sum: raw.evidence.hot_pos_sum,
                n_hot: raw.evidence.n_hot,
                cold_pos_sum: raw.evidence.cold_pos_sum,
                n_cold: raw.evidence.n_cold,
                n_uv: raw.evidence.n_uv,
                n_lif: raw.evidence.n_lif,
                n_other_source: raw.evidence.n_other_source,
                n_arom: raw.evidence.n_arom,
                residue_support: raw.evidence.residue_support.clone(),
                merge_reasons: vec!["region_seed".to_string()],
                centroid_spread_a: 0.0,
                residue_shell_overlap_with_rep: 1.0,
            });
        }
    }
    eprintln!(
        "[v2] Stage B — {} raw peaks consolidated to {} regions (radius={} Å, jaccard>={})",
        raw_peaks.len(), regions.len(), args.consolidation_radius_a, args.consolidation_jaccard
    );

    // Build duplicate-cluster records (only for regions that absorbed peaks).
    for (ri, region) in regions.iter().enumerate() {
        if region.merged_raw_peak_ids.len() <= 1 {
            continue;
        }
        let suppressed: Vec<u32> = region.merged_raw_peak_ids
            .iter()
            .filter(|&&id| id != region.representative_raw_peak_id)
            .copied()
            .collect();
        duplicate_records.push(DuplicateClusterRecord {
            duplicate_cluster_id: ri as u32,
            kept_region_id: region.region_id,
            kept_centroid_a: region.aggregate_centroid_a,
            n_suppressed: suppressed.len() as u32,
            suppressed_raw_peak_ids: suppressed,
            centroid_spread_a: region.centroid_spread_a,
            residue_shell_overlap_with_rep: region.residue_shell_overlap_with_rep,
            merge_reasons: region.merge_reasons.clone(),
        });
    }

    // ─── Per-region computation: residue-shell plausibility, manifold,
    // KCC driver, refined temporal, stream balance, composite PRISM score ──
    let topology_residue_names: Vec<String> = topology
        .as_ref()
        .map(|t| t.residue_names.clone())
        .unwrap_or_default();

    let mut materialized_v2: Vec<serde_json::Value> = Vec::new();
    let mut density_only_candidates: Vec<serde_json::Value> = Vec::new();
    let mut spatiotemporal_partial_candidates: Vec<serde_json::Value> = Vec::new();
    let max_n_spikes_for_density_normalize: f64 = regions
        .iter()
        .map(|r| r.n_spikes as f64)
        .fold(0.0, f64::max)
        .max(1.0);

    // Pre-compute density-only ranking (for ablation report) — by raw n_spikes.
    let mut density_only_rank_by_region: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::new();
    {
        let mut ord: Vec<(u32, u64)> = regions.iter().map(|r| (r.region_id, r.n_spikes)).collect();
        ord.sort_by(|a, b| b.1.cmp(&a.1));
        for (i, (rid, _)) in ord.iter().enumerate() {
            density_only_rank_by_region.insert(*rid, i as u32);
        }
    }

    struct ScoredRegion {
        region_id: u32,
        old_density_rank: u32,
        score_components: PrismScoreComponents,
        manifest_payload: serde_json::Value,
        materialization_level: &'static str,
        materialization_status: &'static str,
        promoted: bool,
    }
    let mut scored: Vec<ScoredRegion> = Vec::new();

    let n_residues_topo: usize = topology.as_ref().map(|t| t.n_residues).unwrap_or(306);
    let terminal_threshold: i32 = 5; // residues 0..5 or last-5 are "terminal"

    for region in regions.iter() {
        let region_id = region.region_id;
        // ── stream support / balance ────────────────────────────────────────
        let supporting_streams: Vec<u32> = region.per_stream_spikes
            .iter()
            .filter_map(|(s, c)| if *c >= min_spikes_per_stream { Some(*s) } else { None })
            .collect();
        let mut supporting_sorted = supporting_streams.clone();
        supporting_sorted.sort_unstable();
        let mut per_stream_counts_json = serde_json::Map::new();
        for (s, c) in region.per_stream_spikes.iter() {
            per_stream_counts_json.insert(s.to_string(), serde_json::Value::from(*c));
        }
        let stream_counts_vec: Vec<u64> = region.per_stream_spikes.values().copied().collect();
        let gini = gini_coefficient(&stream_counts_vec);
        let stream_balance_score = (1.0 - gini).max(0.0);
        let stream_balance = StreamBalance {
            n_streams_supporting: supporting_sorted.len() as u32,
            per_stream_spike_counts: per_stream_counts_json.clone(),
            gini_coefficient: gini,
            stream_balance_score,
            per_stream_support_balance_status: if stream_counts_vec.is_empty() {
                "no_streams"
            } else if gini < 0.1 {
                "balanced"
            } else if gini < 0.3 {
                "moderately_balanced"
            } else {
                "imbalanced"
            },
        };

        // ── refined temporal support ────────────────────────────────────────
        let counts: Vec<u64> = region.bin_counts.values().copied().collect();
        let total_bins: i32 = if region.timestep_max > region.timestep_min {
            ((region.timestep_max - region.timestep_min) / args.persistence_bin_steps.max(1)) + 1
        } else { 1 };
        let persistence_fraction = if total_bins > 0 {
            region.persistence_bin_set.len() as f64 / total_bins as f64
        } else { 0.0 };
        let mean_count = if !counts.is_empty() {
            counts.iter().sum::<u64>() as f64 / counts.len() as f64
        } else { 0.0 };
        let var = if counts.len() > 1 {
            let m = mean_count;
            counts.iter().map(|c| (*c as f64 - m).powi(2)).sum::<f64>() / counts.len() as f64
        } else { 0.0 };
        let burstiness_score = if mean_count > 0.0 {
            (var.sqrt() / mean_count).min(2.0) / 2.0
        } else { 0.0 };
        let temporal_entropy = if !counts.is_empty() {
            let total: f64 = counts.iter().map(|&c| c as f64).sum();
            if total > 0.0 {
                let mut ent = 0.0;
                for c in counts.iter() {
                    let p = (*c as f64) / total;
                    if p > 0.0 { ent -= p * p.ln(); }
                }
                ent
            } else { 0.0 }
        } else { 0.0 };
        let temporal_status: &'static str = if region.bin_counts.is_empty() {
            "coarse_non_discriminative"
        } else { "available" };
        let refined_temporal = RefinedTemporalSupport {
            persistence_bin_size_steps: args.persistence_bin_steps,
            persistence_bins_observed: region.persistence_bin_set.len() as u32,
            persistence_bins_total_possible: total_bins.max(0) as u32,
            persistence_fraction,
            burstiness_score,
            temporal_entropy,
            temporal_support_status: temporal_status,
        };
        let temporal_persistence_factor: f64 = if temporal_status == "available" {
            // Reward high persistence, penalize bursty.
            (persistence_fraction.powf(0.5)) * (1.0 - 0.5 * burstiness_score).max(0.0)
        } else { 1.0 };

        // ── lining residues / driver residues / residue shell features ─────
        let mut lining_pairs: Vec<(i32, u32)> = region.residue_support
            .iter()
            .map(|(r, c)| (*r, *c))
            .collect();
        lining_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        let lining_residues: Vec<i32> = lining_pairs.iter().take(12).map(|x| x.0).collect();
        let representative_residues: Vec<i32> = lining_residues.iter().take(5).cloned().collect();

        // Tail penalty + terminal fraction.
        let mut n_terminal: u32 = 0;
        for r in lining_residues.iter() {
            let r = *r;
            if r >= 0 && (r < terminal_threshold || r >= (n_residues_topo as i32 - terminal_threshold)) {
                n_terminal += 1;
            }
        }
        let terminal_residue_fraction = if !lining_residues.is_empty() {
            n_terminal as f64 / lining_residues.len() as f64
        } else { 0.0 };
        let tail_penalty = terminal_residue_fraction;
        let tail_penalty_factor = (1.0 - tail_penalty).max(0.05);

        // Residue diversity = unique residue NAMES / shell_size.
        let mut residue_diversity_score = 1.0_f64;
        let mut aromatic_count: u32 = 0;
        let mut hydrophobic_count: u32 = 0;
        let mut polar_count: u32 = 0;
        if !topology_residue_names.is_empty() && !lining_residues.is_empty() {
            let mut name_set: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for r in lining_residues.iter() {
                if (*r as usize) < topology_residue_names.len() {
                    let name = topology_residue_names[*r as usize].as_str();
                    name_set.insert(name);
                    if AROMATIC.contains(&name) { aromatic_count += 1; }
                    if HYDROPHOBIC.contains(&name) { hydrophobic_count += 1; }
                    if POLAR.contains(&name) { polar_count += 1; }
                }
            }
            residue_diversity_score = (name_set.len() as f64 / lining_residues.len() as f64).clamp(0.0, 1.0);
        }
        // Compactness: mean pairwise CA distance among lining residues.
        let mut compactness_a: Option<f64> = None;
        let mut enclosure_proxy: Option<f64> = None;
        if !residue_centers.is_empty() && lining_residues.len() >= 2 {
            let mut sum_d = 0.0_f64;
            let mut n = 0_u32;
            for i in 0..lining_residues.len() {
                for j in (i + 1)..lining_residues.len() {
                    let ri = lining_residues[i] as usize;
                    let rj = lining_residues[j] as usize;
                    if ri < residue_centers.len() && rj < residue_centers.len() {
                        if let (Some(a), Some(b)) = (residue_centers[ri], residue_centers[rj]) {
                            sum_d += dist3(&a, &b) as f64;
                            n += 1;
                        }
                    }
                }
            }
            if n > 0 {
                let mean_d = sum_d / n as f64;
                compactness_a = Some(mean_d);
                // Pocket-like = compact (mean d <~12 Å) AND multi-residue.
                enclosure_proxy = Some((1.0 / (1.0 + (mean_d - 8.0).max(0.0) / 4.0))
                    * (lining_residues.len() as f64 / 12.0).min(1.0));
            }
        }
        // Catalytic-like proxy = aromatic+polar mix without ligand-truth use.
        let catalytic_like = if !lining_residues.is_empty() {
            let mix = (aromatic_count.min(polar_count)) as f64 / lining_residues.len() as f64;
            (mix * (1.0 - tail_penalty)).min(1.0)
        } else { 0.0 };
        let mut shell_reasoning: Vec<String> = Vec::new();
        if tail_penalty > 0.5 { shell_reasoning.push(format!("dominated_by_terminal_residues ({:.0}%)", tail_penalty * 100.0)); }
        if residue_diversity_score < 0.5 { shell_reasoning.push("low_residue_name_diversity".to_string()); }
        if let Some(c) = compactness_a {
            if c > 14.0 { shell_reasoning.push(format!("residue_shell_loose_mean_d={:.1}A", c)); }
        }
        if catalytic_like > 0.15 { shell_reasoning.push("aromatic_polar_mix_present".to_string()); }
        let residue_shell = ResidueShellPlausibility {
            residue_shell_size: lining_residues.len() as u32,
            residue_diversity_score,
            compactness_a,
            enclosure_proxy,
            tail_penalty,
            terminal_residue_fraction,
            backbone_exposure_proxy: None,
            aromatic_count,
            hydrophobic_count,
            polar_count,
            catalytic_like_pocket_support: catalytic_like,
            residue_shell_reasoning: shell_reasoning,
        };
        let residue_shell_plausibility_factor: f64 = {
            let div = residue_diversity_score.max(0.1);
            let comp = enclosure_proxy.unwrap_or(0.5);
            (div * comp).clamp(0.05, 1.5)
        };

        // ── KCC driver block ────────────────────────────────────────────────
        let mut driver_pairs: Vec<(i32, u64)> = lining_residues
            .iter()
            .filter_map(|r| {
                driver_active_causal_sum.get(&(*r as usize)).map(|ac| (*r, *ac))
            })
            .collect();
        driver_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        let driver_residues: Vec<i32> = driver_pairs.iter().take(8).map(|x| x.0).collect();
        let driver_residue_count = driver_pairs.len() as u32;
        let driver_residue_fraction_of_shell = if !lining_residues.is_empty() {
            driver_residue_count as f64 / lining_residues.len() as f64
        } else { 0.0 };
        let active_causal_support_sum: u64 = driver_pairs.iter().map(|(_, ac)| *ac).sum();
        let causal_driver_centroid: Option<[f32; 3]> = if !residue_centers.is_empty() && !driver_pairs.is_empty() {
            let mut sum = [0.0f64; 3];
            let mut wsum = 0.0f64;
            for (rid, ac) in driver_pairs.iter() {
                let r = *rid as usize;
                if r < residue_centers.len() {
                    if let Some(c) = residue_centers[r] {
                        let w = *ac as f64;
                        sum[0] += c[0] as f64 * w;
                        sum[1] += c[1] as f64 * w;
                        sum[2] += c[2] as f64 * w;
                        wsum += w;
                    }
                }
            }
            if wsum > 0.0 { Some(vec3_div(sum, wsum)) } else { None }
        } else { None };
        let kcc_field_completeness: &'static str = if n_total_kcc_streams >= 4 {
            "partial_strong"
        } else if n_total_kcc_streams > 0 {
            "partial_weak"
        } else { "missing" };
        let kcc_driver_factor: f64 = if n_total_kcc_streams == 0 {
            1.0 // neutral
        } else {
            (1.0 + 0.5 * driver_residue_fraction_of_shell).clamp(0.7, 1.6)
        };
        let kcc_driver_score = if n_total_kcc_streams > 0 {
            driver_residue_fraction_of_shell * (active_causal_support_sum as f64).ln().max(0.0)
        } else { 0.0 };
        let mut kcc_reasoning: Vec<String> = Vec::new();
        if n_total_kcc_streams == 0 { kcc_reasoning.push("kcc_unavailable_neutral_factor".to_string()); }
        if driver_residue_fraction_of_shell > 0.5 { kcc_reasoning.push(format!("driver_dominant_shell_fraction={:.2}", driver_residue_fraction_of_shell)); }
        let kcc_driver_block = KccDriverBlock {
            driver_residue_count,
            driver_residue_fraction_of_shell,
            driver_centroid_present: causal_driver_centroid.is_some(),
            driver_centroid_distance_to_whole_site_a: None, // computed below
            active_causal_support_sum,
            kcc_field_completeness,
            kcc_driver_score,
            kcc_driver_reasoning: kcc_reasoning,
        };

        // ── Centroid manifold ───────────────────────────────────────────────
        let whole_site = if region.intensity_sum > 0.0 {
            Some(vec3_div([region.pos_sum_x, region.pos_sum_y, region.pos_sum_z], region.intensity_sum))
        } else { None };
        let hot_phase = if region.n_hot >= 50 {
            Some(vec3_div(region.hot_pos_sum, region.n_hot as f64))
        } else { None };
        let cold_phase = if region.n_cold >= 50 {
            Some(vec3_div(region.cold_pos_sum, region.n_cold as f64))
        } else { None };
        let lining_mass: Option<[f32; 3]> = if !residue_centers.is_empty() {
            let mut sum = [0.0f64; 3];
            let mut wsum = 0.0f64;
            for (rid, count) in lining_pairs.iter().take(20) {
                let r = *rid as usize;
                if r < residue_centers.len() {
                    if let Some(c) = residue_centers[r] {
                        let w = *count as f64;
                        sum[0] += c[0] as f64 * w;
                        sum[1] += c[1] as f64 * w;
                        sum[2] += c[2] as f64 * w;
                        wsum += w;
                    }
                }
            }
            if wsum > 0.0 { Some(vec3_div(sum, wsum)) } else { None }
        } else { None };
        let causal_driver = causal_driver_centroid;
        let ligand_adjacent: Option<[f32; 3]> = ground_truth
            .as_ref()
            .and_then(|gt| if gt.valid_for_dcc_validation { gt.ligand_centroid } else { None });
        let manifold = CentroidManifold {
            whole_site, lining_mass, causal_driver, hot_phase, cold_phase,
            return_phase: None, ligand_adjacent,
        };

        // Manifold consistency.
        let d_whole_lining = whole_site.zip(lining_mass).map(|(a, b)| dist3(&a, &b));
        let d_whole_driver = whole_site.zip(causal_driver).map(|(a, b)| dist3(&a, &b));
        let d_hot_cold = hot_phase.zip(cold_phase).map(|(a, b)| dist3(&a, &b));
        let mut manifold_dispersion: Option<f32> = None;
        let n_views_present = [whole_site, lining_mass, causal_driver, hot_phase, cold_phase]
            .iter().filter(|v| v.is_some()).count() as u32;
        let views_vec: Vec<[f32; 3]> = [whole_site, lining_mass, causal_driver, hot_phase, cold_phase]
            .iter().flatten().copied().collect();
        if views_vec.len() >= 2 {
            let mut max_d: f32 = 0.0;
            for i in 0..views_vec.len() {
                for j in (i + 1)..views_vec.len() {
                    let d = dist3(&views_vec[i], &views_vec[j]);
                    if d > max_d { max_d = d; }
                }
            }
            manifold_dispersion = Some(max_d);
        }
        // Manifold consistency: exp(-dispersion / 4Å).
        let manifold_consistency_score: f64 = if let Some(d) = manifold_dispersion {
            ((-(d as f64) / 4.0).exp()).clamp(0.0, 1.0)
        } else { 0.5 };
        let centroid_manifold_consistency_factor: f64 = manifold_consistency_score
            * (n_views_present as f64 / 5.0).min(1.0).max(0.2);
        let manifold_block = ManifoldConsistency {
            d_whole_lining_a: d_whole_lining,
            d_whole_driver_a: d_whole_driver,
            d_hot_cold_a: d_hot_cold,
            manifold_dispersion_a: manifold_dispersion,
            manifold_consistency_score,
            n_views_present,
        };
        // Patch driver-distance into KccDriverBlock (we already constructed it above).
        let kcc_driver_block_with_dist = KccDriverBlock {
            driver_centroid_distance_to_whole_site_a: d_whole_driver,
            ..kcc_driver_block
        };

        // ── Stream support struct (legacy shape kept for back-compat) ───────
        let stream_support = StreamSupport {
            n_streams_supporting: supporting_sorted.len() as u32,
            supporting_streams: supporting_sorted.clone(),
            per_stream_spike_counts: per_stream_counts_json.clone(),
            per_stream_voxel_counts: {
                let mut m = serde_json::Map::new();
                for (s, vox) in region.per_stream_voxels.iter() {
                    m.insert(s.to_string(), serde_json::Value::from(vox.len()));
                }
                m
            },
        };

        // ── Phase support / Rayleigh r ──────────────────────────────────────
        let n = region.n_spikes as f64;
        let r_stat = if n > 0.0 {
            (region.phase_cos_sum.powi(2) + region.phase_sin_sum.powi(2)).sqrt() / n
        } else { 0.0 };
        let mean_phase = if r_stat > 1e-3 {
            Some(region.phase_sin_sum.atan2(region.phase_cos_sum))
        } else { None };
        let phase_support = PhaseSupport {
            phase_bits_histogram: region.phase_hist,
            rayleigh_r_stat: r_stat,
            mean_phase_radians: mean_phase,
            n_hot: region.n_hot,
            n_cold: region.n_cold,
        };

        // ── Field completeness factor ───────────────────────────────────────
        let mut sources_present: f64 = 0.0;
        let mut sources_total: f64 = 0.0;
        for (status, weight) in [
            ("present", 1.0),                               // spikes
            ("present", 1.0),                               // phase_bits
            (if signal_grid_present_any { "partial" } else { "missing" }, 1.0),
            (if n_total_kcc_streams > 0 { "partial" } else { "missing" }, 1.0),
        ] {
            sources_total += weight;
            sources_present += match status {
                "present" => 1.0 * weight,
                "partial" => 0.5 * weight,
                _ => 0.0,
            };
        }
        let field_completeness_factor: f64 = (sources_present / sources_total).clamp(0.5, 1.0);
        let field_completeness = FieldCompleteness {
            spikes: "present",
            phase_bits: "present",
            warp_matrix: "deferred_format_not_parsed",
            asc_vectors: "deferred_format_not_parsed",
            forces: "deferred_format_not_parsed",
            signal_grid: if signal_grid_present_any { "partial" } else { "missing" },
            kcc: if n_total_kcc_streams > 0 { "partial" } else { "missing" },
        };

        // ── Density score (log-normalized) ──────────────────────────────────
        let density_score = (region.n_spikes as f64 + 1.0).ln()
            / (max_n_spikes_for_density_normalize + 1.0).ln();
        // ── Duplicate region factor: representative regions have factor 1.0.
        let duplicate_region_factor: f64 = 1.0;

        // ── Composite PRISM score ───────────────────────────────────────────
        let final_prism_score = density_score
            * stream_balance_score.clamp(0.05, 1.0)
            * temporal_persistence_factor.clamp(0.05, 1.0)
            * residue_shell_plausibility_factor
            * centroid_manifold_consistency_factor.clamp(0.05, 1.0)
            * kcc_driver_factor
            * field_completeness_factor
            * duplicate_region_factor
            * tail_penalty_factor;

        let mut score_explanation: Vec<String> = Vec::new();
        score_explanation.push(format!("density_score={:.3}", density_score));
        score_explanation.push(format!("stream_balance={:.3} (gini={:.2})", stream_balance_score, gini));
        score_explanation.push(format!("temporal_persistence={:.3} ({})", temporal_persistence_factor, temporal_status));
        score_explanation.push(format!("residue_shell={:.3} (div={:.2}, encl={:?})", residue_shell_plausibility_factor, residue_diversity_score, enclosure_proxy));
        score_explanation.push(format!("manifold={:.3} ({} views)", centroid_manifold_consistency_factor, n_views_present));
        score_explanation.push(format!("kcc_driver={:.3} (drv_frac={:.2})", kcc_driver_factor, driver_residue_fraction_of_shell));
        score_explanation.push(format!("field_completeness={:.3}", field_completeness_factor));
        score_explanation.push(format!("tail_penalty_factor={:.3} (term_frac={:.2})", tail_penalty_factor, terminal_residue_fraction));
        score_explanation.push(format!("FINAL={:.6}", final_prism_score));

        let score_components = PrismScoreComponents {
            density_score,
            stream_balance_factor: stream_balance_score.clamp(0.05, 1.0),
            temporal_persistence_factor: temporal_persistence_factor.clamp(0.05, 1.0),
            residue_shell_plausibility_factor,
            centroid_manifold_consistency_factor: centroid_manifold_consistency_factor.clamp(0.05, 1.0),
            kcc_driver_factor,
            field_completeness_factor,
            duplicate_region_factor,
            tail_penalty_factor,
            final_prism_score,
            score_explanation,
        };

        // ── Promotion criteria (tightened per directive Part 8) ─────────────
        let mut blocking: Vec<&'static str> = Vec::new();
        if whole_site.is_none() { blocking.push("missing_whole_site_centroid"); }
        if (supporting_sorted.len() as u32) < min_supporting_streams { blocking.push("insufficient_stream_support"); }
        if region.n_spikes < 1000 { blocking.push("insufficient_spike_support"); }
        if region.chunk_set.is_empty() { blocking.push("no_temporal_support"); }
        // At least one non-density PRISM evidence dimension active:
        let kcc_active = n_total_kcc_streams > 0 && driver_residue_count > 0;
        let manifold_active = n_views_present >= 2 && manifold_consistency_score > 0.4;
        let temporal_active = temporal_status == "available" && persistence_fraction > 0.3;
        let shell_active = !lining_residues.is_empty() && tail_penalty < 0.8 && residue_diversity_score > 0.4;
        let signal_grid_active = signal_grid_present_any;
        let n_active_dims = [kcc_active, manifold_active, temporal_active, shell_active, signal_grid_active]
            .iter().filter(|x| **x).count();
        if n_active_dims == 0 { blocking.push("no_non_density_evidence_dimension_active"); }

        let materialization_level: &'static str = if blocking.is_empty() {
            if gt_status == "available" { "materialized_dynamic_site_with_validation" }
            else { "materialized_dynamic_site" }
        } else if !blocking.contains(&"missing_whole_site_centroid")
                  && !blocking.contains(&"no_temporal_support") {
            "candidate_spatiotemporal_partial"
        } else {
            "candidate_density_only"
        };
        let materialization_status: &'static str = if blocking.is_empty() { "materialized" } else { "non_materialized_candidate" };
        let promoted = blocking.is_empty();

        // ── Build manifest payload (one JSON value per region) ──────────────
        let so3_block = serde_json::json!({
            "status": if n_views_present >= 3 && r_stat > 0.05 { "available" }
                      else if n_views_present >= 2 { "partial" }
                      else { "missing" },
            "stream_support": stream_support,
            "temporal_support": {
                "first_observed_step": if region.timestep_min == i32::MAX { None } else { Some(region.timestep_min) },
                "last_observed_step":  if region.timestep_max == i32::MIN { None } else { Some(region.timestep_max) },
                "recurrence_count": region.chunk_set.len() as u32,
                "chunk_support": region.chunk_set.iter().copied().collect::<Vec<i32>>(),
                "n_phase_bins_nonzero": region.phase_hist.iter().filter(|c| **c > 0).count() as u32,
            },
            "phase_support": phase_support,
            "so3_support": {
                "phase_coherence_rayleigh_r": r_stat,
                "orientation_coherence_score": r_stat,
                "local_frame_stability": null,
                "rotation_dispersion": null,
                "warp_matrix_support": "deferred_format_not_parsed",
                "asc_vector_support": "deferred_format_not_parsed",
                "force_direction_support": "deferred_format_not_parsed",
                "evidence_files": spikes_files_consumed,
                "so3_score_status": "deferred_format_not_parsed",
                "missing_so3_fields": ["warp_matrix", "asc_vectors", "forces_final"],
            },
            "centroid_manifold": manifold,
            "field_completeness": field_completeness,
        });
        let accuracy: serde_json::Value = if let Some(lig) = ligand_adjacent {
            // Ligand-adjacent comes from valid ground_truth only — used purely
            // as POST-HOC validation, never in score (per directive Part 9).
            let dcc_whole = whole_site.map(|c| dist3(&c, &lig));
            let dcc_lining = lining_mass.map(|c| dist3(&c, &lig));
            let dcc_driver = causal_driver.map(|c| dist3(&c, &lig));
            let dcc_hot = hot_phase.map(|c| dist3(&c, &lig));
            let dcc_cold = cold_phase.map(|c| dist3(&c, &lig));
            let best = [dcc_whole, dcc_lining, dcc_driver, dcc_hot, dcc_cold]
                .into_iter().flatten().fold(f32::INFINITY, f32::min);
            let grade = if best <= 2.0 { "EXCELLENT" }
                        else if best <= 3.0 { "ACCEPTABLE" }
                        else if best <= 5.0 { "USEFUL" }
                        else if best <= 8.0 { "LENIENT" }
                        else { "MISS" };
            serde_json::json!({
                "validation_status": "computed",
                "ligand_centroid": lig,
                "dcc_whole_site_a": dcc_whole,
                "dcc_lining_mass_a": dcc_lining,
                "dcc_causal_driver_a": dcc_driver,
                "dcc_hot_phase_a": dcc_hot,
                "dcc_cold_phase_a": dcc_cold,
                "best_dcc_a": best,
                "grade": grade,
            })
        } else {
            serde_json::json!({
                "validation_status": gt_status,
                "skip_reason": gt_skip_reason,
            })
        };
        let limitations: Vec<&'static str> = vec![
            "warp_matrix_so3_support_deferred",
            "asc_vector_support_deferred",
            "force_direction_support_deferred",
        ];
        let provenance = serde_json::json!({
            "source_manifest": args.manifest.display().to_string(),
            "run_id": manifest.run_id,
            "target": manifest.target,
            "region_id": region_id,
            "representative_raw_peak_id": region.representative_raw_peak_id,
            "merged_raw_peak_ids": region.merged_raw_peak_ids,
            "n_raw_peaks_merged": region.merged_raw_peak_ids.len() as u32,
            "spikes_files_consumed": spikes_files_consumed,
            "kcc_files_consumed": kcc_files_consumed,
            "topology_input": manifest.topology_input,
        });

        let payload = serde_json::json!({
            "site_id": format!("region_{:03}", region_id),
            "rank": null,
            "old_density_rank": density_only_rank_by_region.get(&region_id).copied().unwrap_or(0),
            "materialization_status": materialization_status,
            "materialization_level": materialization_level,
            "centroid_xyz": whole_site.unwrap_or(region.aggregate_centroid_a),
            "centroid_manifold": manifold,
            "representative_residues": representative_residues,
            "lining_residues": lining_residues,
            "driver_residues": driver_residues,
            "n_spikes": region.n_spikes,
            "intensity_sum": region.intensity_sum,
            "n_raw_peaks_merged": region.merged_raw_peak_ids.len() as u32,
            "region_radius_a": region.region_radius_a,
            "centroid_spread_a": region.centroid_spread_a,
            "spike_support": {
                "n_uv": region.n_uv,
                "n_lif": region.n_lif,
                "n_other_source": region.n_other_source,
                "n_aromatic": {
                    "TRP": region.n_arom[0],
                    "TYR": region.n_arom[1],
                    "PHE": region.n_arom[2],
                    "SS":  region.n_arom[3],
                    "none": region.n_arom[4],
                },
            },
            "stream_support": stream_support,
            "stream_balance": stream_balance,
            "phase_support": phase_support,
            "refined_temporal_support": refined_temporal,
            "residue_shell_plausibility": residue_shell,
            "manifold_consistency": manifold_block,
            "kcc_driver": kcc_driver_block_with_dist,
            "spatiotemporal_so3_evidence": so3_block,
            "field_completeness": field_completeness,
            "score_components": score_components,
            "blocking_reasons": blocking,
            "n_active_non_density_dimensions": n_active_dims as u32,
            "promoted": promoted,
            "provenance": provenance,
            "limitations": limitations,
            "accuracy": accuracy,
        });

        scored.push(ScoredRegion {
            region_id,
            old_density_rank: density_only_rank_by_region.get(&region_id).copied().unwrap_or(0),
            score_components,
            manifest_payload: payload,
            materialization_level,
            materialization_status,
            promoted,
        });
    }

    // ── Re-rank by final_prism_score ─────────────────────────────────────────
    scored.sort_by(|a, b| {
        b.score_components.final_prism_score
            .partial_cmp(&a.score_components.final_prism_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, sr) in scored.iter_mut().enumerate() {
        sr.manifest_payload["rank"] = serde_json::Value::from(i as u32);
        sr.manifest_payload["new_prism_rank"] = serde_json::Value::from(i as u32);
        sr.manifest_payload["site_id"] = serde_json::Value::from(format!("site_{:03}", i));
    }
    // Promotion: top materialized_top_n among those with `promoted=true`.
    for (i, sr) in scored.iter_mut().enumerate() {
        if !sr.promoted {
            // Bucket non-promoted into spatiotemporal_partial vs density_only.
            match sr.materialization_level {
                "candidate_spatiotemporal_partial" => spatiotemporal_partial_candidates.push(sr.manifest_payload.clone()),
                _ => density_only_candidates.push(sr.manifest_payload.clone()),
            }
            continue;
        }
        if materialized_v2.len() < args.materialized_top_n {
            materialized_v2.push(sr.manifest_payload.clone());
        } else {
            // Beyond top-N, retain as spatiotemporal_partial for reporting.
            spatiotemporal_partial_candidates.push(sr.manifest_payload.clone());
        }
        let _ = i;
    }
    eprintln!(
        "[v2] scored {} regions; {} promoted (top-{} materialized); {} spatiotemporal_partial; {} density_only",
        scored.len(), materialized_v2.len(), args.materialized_top_n,
        spatiotemporal_partial_candidates.len(), density_only_candidates.len()
    );

    // Backward-compat shims used by later code paths and emit blocks.
    let materialized: Vec<serde_json::Value> = materialized_v2.clone();
    let non_materialized: Vec<serde_json::Value> = density_only_candidates.clone();

    // ═══════════════════════════════════════════════════════════════════════
    // EMIT v2 OUTPUTS
    // ═══════════════════════════════════════════════════════════════════════

    // ─── Emit binding_sites.materialized.json (v2 schema) ───────────────────
    let materialized_path = output_dir.join("binding_sites.materialized.json");
    let materialized_status: &str = if materialized_v2.is_empty() {
        "no_sites_met_criteria"
    } else if gt_status == "available" {
        "materialized_from_md_evidence_with_validation"
    } else {
        "materialized_from_md_evidence"
    };
    let materialized_json = serde_json::json!({
        "schema_version":   2,
        "schema_kind":      "pathb_binding_sites_materialized",
        "ranking_methodology": "prism_native_composite_v2",
        "status":           materialized_status,
        "target":           manifest.target,
        "run_id":           manifest.run_id,
        "source_manifest":  args.manifest.display().to_string(),
        "site_count":       materialized_v2.len(),
        "n_raw_peaks":      raw_peaks.len(),
        "n_consolidated_regions": regions.len(),
        "consolidation_radius_a": args.consolidation_radius_a,
        "consolidation_jaccard":  args.consolidation_jaccard,
        "binding_sites":    materialized_v2,
        "spatiotemporal_partial_candidates": spatiotemporal_partial_candidates,
        "density_only_candidates": density_only_candidates,
        "missing_fields":   {
            "warp_matrix_so3":         "format_not_parsed_in_this_commit",
            "asc_vector_orientation":  "format_not_parsed_in_this_commit",
            "force_direction":         "format_not_parsed_in_this_commit",
            "ligsite_static_geometry": "intentionally_excluded_per_directive",
        },
        "ground_truth_status":     gt_status,
        "ground_truth_skip_reason": gt_skip_reason,
        "path_b_required":          gt_status != "available",
    });
    {
        let f = File::create(&materialized_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &materialized_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {} (sites={}, st_partial={}, density_only={})",
              materialized_path.display(),
              materialized_v2.len(),
              spatiotemporal_partial_candidates.len(),
              density_only_candidates.len());

    // ─── Emit site_accuracy_report.json (post-hoc validation only) ─────────
    let accuracy_path = output_dir.join("site_accuracy_report.json");
    // Aggregate grade summary if validation was computed.
    let (mut n_excellent, mut n_acceptable, mut n_useful, mut n_lenient, mut n_miss) = (0u32, 0u32, 0u32, 0u32, 0u32);
    let mut best_dcc_overall: Option<f32> = None;
    let mut best_grade_overall: Option<String> = None;
    let mut rank_first_lenient: Option<u32> = None;
    if gt_status == "available" {
        for s in materialized_v2.iter() {
            let acc = &s["accuracy"];
            if let Some(best) = acc["best_dcc_a"].as_f64() {
                let best_f = best as f32;
                if best <= 2.0 { n_excellent += 1; }
                else if best <= 3.0 { n_acceptable += 1; }
                else if best <= 5.0 { n_useful += 1; }
                else if best <= 8.0 { n_lenient += 1; }
                else { n_miss += 1; }
                if best_dcc_overall.is_none() || best_f < best_dcc_overall.unwrap() {
                    best_dcc_overall = Some(best_f);
                    best_grade_overall = acc["grade"].as_str().map(|x| x.to_string());
                }
                if best_f <= 8.0 && rank_first_lenient.is_none() {
                    rank_first_lenient = s["rank"].as_u64().map(|x| x as u32);
                }
            }
        }
    }
    let accuracy_json = serde_json::json!({
        "schema_version": 2,
        "schema_kind":    "pathb_site_accuracy_report",
        "target":         manifest.target,
        "run_id":         manifest.run_id,
        "ground_truth_status":     gt_status,
        "ground_truth_skip_reason": gt_skip_reason,
        "ligand_centroid_used":    ground_truth.as_ref().and_then(|gt| gt.ligand_centroid),
        "n_materialized_sites":    materialized_v2.len(),
        "validation_computable":   gt_status == "available",
        "validation_independent_of_ranking": true,
        "per_site_accuracy":       if gt_status == "available" {
            serde_json::Value::Array(materialized_v2.iter().map(|s| serde_json::json!({
                "site_id":      s["site_id"],
                "rank":         s["rank"],
                "centroid_xyz": s["centroid_xyz"],
                "accuracy":     s["accuracy"],
            })).collect())
        } else { serde_json::Value::Null },
        "summary": {
            "n_excellent_le2A":   if gt_status == "available" { serde_json::Value::from(n_excellent) } else { serde_json::Value::Null },
            "n_acceptable_le3A":  if gt_status == "available" { serde_json::Value::from(n_acceptable) } else { serde_json::Value::Null },
            "n_useful_le5A":      if gt_status == "available" { serde_json::Value::from(n_useful) } else { serde_json::Value::Null },
            "n_lenient_le8A":     if gt_status == "available" { serde_json::Value::from(n_lenient) } else { serde_json::Value::Null },
            "n_miss_gt8A":        if gt_status == "available" { serde_json::Value::from(n_miss) } else { serde_json::Value::Null },
            "best_dcc_a_overall": best_dcc_overall,
            "best_grade_overall": best_grade_overall,
            "rank_first_le8A":    rank_first_lenient,
            "grade_note":         if gt_status == "available" { "validation aggregated post-hoc; ranking did NOT use ligand reference"} else { "validation not computable: reference unavailable per ground_truth.json" },
        },
        "limitations": [
            "warp_matrix_so3_support_not_parsed",
            "asc_vector_orientation_not_parsed",
            "force_direction_not_parsed",
            "ligsite_static_geometry_intentionally_excluded",
        ],
    });
    {
        let f = File::create(&accuracy_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &accuracy_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", accuracy_path.display());

    // ─── Emit materialization_field_completeness.json ──────────────────────
    let mfc_path = output_dir.join("materialization_field_completeness.json");
    let mfc_json = serde_json::json!({
        "schema_version": 2,
        "schema_kind":    "pathb_materialization_field_completeness",
        "evidence_sources": {
            "spikes":         { "status": "available", "format": "PRSPK001", "streams_present": spikes_files_consumed.len() },
            "kcc_v2full":     { "status": if !kcc_per_stream.is_empty() { "partial" } else { "missing" }, "format": "PRKCC001", "streams_present": kcc_per_stream.len() },
            "signal_grid":    { "status": if signal_grid_present_any { "partial" } else { "missing" }, "format": "PRSGD001" },
            "warp_matrix":    { "status": "deferred", "reason": "raw blob format not parsed in this commit" },
            "forces_final":   { "status": "deferred", "reason": "raw blob format not parsed in this commit" },
            "asc_vectors":    { "status": "deferred", "reason": "raw blob format not parsed in this commit" },
            "adaptive_dt":    { "status": "deferred", "reason": "raw blob format not parsed in this commit" },
            "protocol_state": { "status": "available_json", "reason": "loaded for run config; not used in materialization yet" },
            "bocpd_log":      { "status": "available_jsonl", "reason": "not used in materialization yet" },
            "topology":       { "status": if topology.is_some() { "available" } else { "missing" }, "format": "json", "path": manifest.topology_input },
            "ground_truth":   { "status": gt_status, "skip_reason": gt_skip_reason },
        },
        "n_raw_peaks":                     raw_peaks.len(),
        "n_consolidated_regions":          regions.len(),
        "n_promoted_materialized":         materialized_v2.len(),
        "n_spatiotemporal_partial":        spatiotemporal_partial_candidates.len(),
        "n_density_only":                  density_only_candidates.len(),
        "promotion_summary": {
            "materialized_dynamic_site_with_validation":  materialized_v2.iter().filter(|s| s["materialization_level"].as_str() == Some("materialized_dynamic_site_with_validation")).count(),
            "materialized_dynamic_site":                  materialized_v2.iter().filter(|s| s["materialization_level"].as_str() == Some("materialized_dynamic_site")).count(),
            "candidate_spatiotemporal_partial":           spatiotemporal_partial_candidates.len(),
            "candidate_density_only":                     density_only_candidates.len(),
        },
        "ranking_methodology":             "prism_native_composite_v2",
        "ligsite_role":                    "intentionally_excluded_per_directive",
    });
    {
        let f = File::create(&mfc_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &mfc_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", mfc_path.display());

    // ─── Emit duplicate_cluster_report.json ─────────────────────────────────
    let dup_path = output_dir.join("duplicate_cluster_report.json");
    let dup_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_duplicate_cluster_report",
        "consolidation_radius_a":  args.consolidation_radius_a,
        "consolidation_jaccard":   args.consolidation_jaccard,
        "n_raw_peaks":             raw_peaks.len(),
        "n_regions":               regions.len(),
        "n_clusters_with_suppression": duplicate_records.len(),
        "duplicate_clusters":      duplicate_records,
    });
    {
        let f = File::create(&dup_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &dup_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", dup_path.display());

    // ─── Emit ranking_ablation_report.json ──────────────────────────────────
    // Re-rank the SAME consolidated regions under different scoring schemes to
    // show which component moved a region. Each scheme returns the top-10
    // region_ids in rank order.
    fn top_k_by<F: FnMut(&ScoredRegion) -> f64>(
        scored: &[ScoredRegion], k: usize, mut score_fn: F,
    ) -> Vec<serde_json::Value> {
        let mut s: Vec<(u32, f64, [f32; 3], Vec<i32>)> = scored.iter().map(|r| {
            (r.region_id,
             score_fn(r),
             {
                 let v = &r.manifest_payload["centroid_xyz"];
                 [v[0].as_f64().unwrap_or(0.0) as f32,
                  v[1].as_f64().unwrap_or(0.0) as f32,
                  v[2].as_f64().unwrap_or(0.0) as f32]
             },
             r.manifest_payload["lining_residues"].as_array()
                 .map(|a| a.iter().filter_map(|x| x.as_i64().map(|n| n as i32)).take(5).collect())
                 .unwrap_or_default(),
            )
        }).collect();
        s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        s.into_iter().take(k).enumerate().map(|(i, (rid, score, c, lining))| serde_json::json!({
            "rank": i,
            "region_id": rid,
            "score": score,
            "centroid_xyz": c,
            "lining_residues_top5": lining,
        })).collect()
    }
    let abl_path = output_dir.join("ranking_ablation_report.json");
    let abl_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_ranking_ablation_report",
        "n_regions": scored.len(),
        "schemes": {
            "density_only":         top_k_by(&scored, 10, |r| r.score_components.density_score),
            "density_plus_nms":     top_k_by(&scored, 10, |r| r.score_components.density_score * r.score_components.duplicate_region_factor),
            "nms_plus_residue_shell": top_k_by(&scored, 10, |r| r.score_components.density_score * r.score_components.residue_shell_plausibility_factor * r.score_components.tail_penalty_factor),
            "nms_plus_manifold":    top_k_by(&scored, 10, |r| r.score_components.density_score * r.score_components.centroid_manifold_consistency_factor),
            "nms_plus_kcc":         top_k_by(&scored, 10, |r| r.score_components.density_score * r.score_components.kcc_driver_factor),
            "final_prism_score":    top_k_by(&scored, 10, |r| r.score_components.final_prism_score),
        },
    });
    {
        let f = File::create(&abl_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &abl_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", abl_path.display());

    // ─── Emit site_family_report.json ───────────────────────────────────────
    // Group materialized sites whose top-12 lining-residue sets share Jaccard
    // >= 0.4. Reports families useful for cross-site dedup interpretation.
    let family_path = output_dir.join("site_family_report.json");
    let mut families: Vec<Vec<u32>> = Vec::new();
    let mut family_lining: Vec<Vec<i32>> = Vec::new();
    for s in materialized_v2.iter() {
        let region_id = s["provenance"]["region_id"].as_u64().map(|x| x as u32).unwrap_or(0);
        let lining: Vec<i32> = s["lining_residues"].as_array()
            .map(|a| a.iter().filter_map(|x| x.as_i64().map(|n| n as i32)).collect())
            .unwrap_or_default();
        let mut placed = false;
        for (fi, prev) in family_lining.iter_mut().enumerate() {
            if jaccard_overlap(&lining, prev) >= 0.4 {
                families[fi].push(region_id);
                // Update family lining to the union (capped at 12).
                let mut union: std::collections::HashSet<i32> = prev.iter().copied().collect();
                for r in &lining { union.insert(*r); }
                let mut u: Vec<i32> = union.into_iter().collect();
                u.sort();
                u.truncate(12);
                *prev = u;
                placed = true;
                break;
            }
        }
        if !placed {
            families.push(vec![region_id]);
            family_lining.push(lining);
        }
    }
    let family_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_site_family_report",
        "n_families":     families.len(),
        "families":       families.iter().enumerate().zip(family_lining.iter()).map(|((i, members), lining)| serde_json::json!({
            "family_id":       i as u32,
            "n_members":       members.len(),
            "member_region_ids": members,
            "shared_lining":   lining,
        })).collect::<Vec<_>>(),
    });
    {
        let f = File::create(&family_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &family_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", family_path.display());

    // ─── Honest summary ────────────────────────────────────────────────────
    let n_valid: usize = validations.iter().filter(|v| v.valid).count();
    let n_total: usize = validations.len();
    eprintln!(
        "\nDone. {} / {} envelope artifacts validated. {} spikes processed across {} streams. {} unique voxels at {:.1} Å. Top {} density peaks emitted as honest candidates (NOT materialized sites).",
        n_valid,
        n_total,
        total_processed,
        stream_summaries.len(),
        voxel_grid.len(),
        args.voxel_size_a,
        candidates.len(),
    );
    if n_total > 0 && n_valid < n_total {
        eprintln!("WARNING: {} artifacts failed validation — see materialization_report.json", n_total - n_valid);
    }
    eprintln!(
        "Materialization summary: {} sites materialized, {} non_materialized candidates, ground_truth={}",
        materialized.len(),
        non_materialized.len(),
        gt_status,
    );

    Ok(())
}
