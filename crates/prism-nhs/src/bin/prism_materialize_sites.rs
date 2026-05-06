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

    /// Top-K density peaks to emit as site_candidates.
    #[arg(long, default_value = "50")]
    top_k_candidates: usize,

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

    // ─── Pass 6 — finalize per-candidate centroid manifold + promotion ─────
    let mut materialized: Vec<MaterializedSite> = Vec::new();
    let mut non_materialized: Vec<NonMaterializedCandidate> = Vec::new();
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

    for (cand_idx, ev) in evidence.iter().enumerate() {
        // Stream support — count streams with >= min_spikes_per_stream attributed.
        let supporting_streams: Vec<u32> = ev
            .per_stream_spikes
            .iter()
            .filter_map(|(s, c)| if *c >= min_spikes_per_stream { Some(*s) } else { None })
            .collect();
        let mut supporting_streams_sorted = supporting_streams.clone();
        supporting_streams_sorted.sort_unstable();

        // Per-stream JSON support.
        let mut per_stream_spike_counts = serde_json::Map::new();
        let mut per_stream_voxel_counts = serde_json::Map::new();
        for (s, c) in ev.per_stream_spikes.iter() {
            per_stream_spike_counts.insert(s.to_string(), serde_json::Value::from(*c));
        }
        for (s, vox) in ev.per_stream_voxels.iter() {
            per_stream_voxel_counts.insert(s.to_string(), serde_json::Value::from(vox.len()));
        }

        // Whole-site centroid (intensity-weighted).
        let whole_site = if ev.intensity_sum > 0.0 {
            Some(vec3_div([ev.pos_sum_x, ev.pos_sum_y, ev.pos_sum_z], ev.intensity_sum))
        } else { None };

        // Hot/cold phase centroids.
        let hot_phase = if ev.n_hot >= 50 {
            Some(vec3_div(ev.hot_pos_sum, ev.n_hot as f64))
        } else { None };
        let cold_phase = if ev.n_cold >= 50 {
            Some(vec3_div(ev.cold_pos_sum, ev.n_cold as f64))
        } else { None };

        // Lining residues (top-12 by support count) + lining_mass centroid.
        let mut lining_pairs: Vec<(i32, u32)> = ev.residue_support
            .iter()
            .map(|(r, c)| (*r, *c))
            .collect();
        lining_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        let lining_residues: Vec<i32> = lining_pairs.iter().take(12).map(|x| x.0).collect();

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

        // Driver residues = top-K driver_active_causal_sum that are ALSO in this candidate's residue support.
        let mut driver_pairs: Vec<(i32, u64)> = lining_pairs
            .iter()
            .filter_map(|(rid, _c)| {
                driver_active_causal_sum
                    .get(&(*rid as usize))
                    .map(|ac| (*rid, *ac))
            })
            .collect();
        driver_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        let driver_residues: Vec<i32> = driver_pairs.iter().take(8).map(|x| x.0).collect();

        let causal_driver: Option<[f32; 3]> = if !residue_centers.is_empty() && !driver_pairs.is_empty() {
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

        // Ligand-adjacent: only if ground_truth has a valid ligand_centroid.
        let ligand_adjacent: Option<[f32; 3]> = ground_truth
            .as_ref()
            .and_then(|gt| if gt.valid_for_dcc_validation { gt.ligand_centroid } else { None });

        let centroid_manifold = CentroidManifold {
            whole_site,
            lining_mass,
            causal_driver,
            hot_phase,
            cold_phase,
            return_phase: None,
            ligand_adjacent,
        };

        // Phase coherence — Rayleigh r-stat.
        let n = ev.n_spikes as f64;
        let r_stat = if n > 0.0 {
            (ev.phase_cos_sum.powi(2) + ev.phase_sin_sum.powi(2)).sqrt() / n
        } else { 0.0 };
        let mean_phase = if r_stat > 1e-3 {
            Some(ev.phase_sin_sum.atan2(ev.phase_cos_sum))
        } else { None };

        let n_phase_bins_nonzero = ev.phase_hist.iter().filter(|&&c| c > 0).count() as u32;
        let mut chunk_support: Vec<i32> = ev.chunk_set.iter().copied().collect();
        chunk_support.sort_unstable();

        let stream_support = StreamSupport {
            n_streams_supporting: supporting_streams_sorted.len() as u32,
            supporting_streams: supporting_streams_sorted.clone(),
            per_stream_spike_counts,
            per_stream_voxel_counts,
        };
        let temporal_support = TemporalSupport {
            first_observed_step: if ev.timestep_min == i32::MAX { None } else { Some(ev.timestep_min) },
            last_observed_step: if ev.timestep_max == i32::MIN { None } else { Some(ev.timestep_max) },
            recurrence_count: ev.chunk_set.len() as u32,
            chunk_support,
            n_phase_bins_nonzero,
        };
        let phase_support = PhaseSupport {
            phase_bits_histogram: ev.phase_hist,
            rayleigh_r_stat: r_stat,
            mean_phase_radians: mean_phase,
            n_hot: ev.n_hot,
            n_cold: ev.n_cold,
        };
        let so3_support = So3Support {
            phase_coherence_rayleigh_r: r_stat,
            orientation_coherence_score: Some(r_stat),
            local_frame_stability: None,
            rotation_dispersion: None,
            warp_matrix_support: "deferred_format_not_parsed",
            asc_vector_support: "deferred_format_not_parsed",
            force_direction_support: "deferred_format_not_parsed",
            evidence_files: spikes_files_consumed.clone(),
        };
        let field_completeness = FieldCompleteness {
            spikes: "present",
            phase_bits: "present",
            warp_matrix: "deferred_format_not_parsed",
            asc_vectors: "deferred_format_not_parsed",
            forces: "deferred_format_not_parsed",
            signal_grid: if manifest.artifacts.iter().any(|a| a.present && a.kind == "signal_grid") { "partial" } else { "missing" },
            kcc: if n_total_kcc_streams > 0 { "partial" } else { "missing" },
        };

        // Promotion.
        let n_streams_supporting = supporting_streams_sorted.len() as u32;
        let has_min_centroid_views: bool = whole_site.is_some()
            && (lining_mass.is_some() || hot_phase.is_some() || cold_phase.is_some());
        let manifold_count: usize = [whole_site, lining_mass, causal_driver, hot_phase, cold_phase]
            .iter()
            .filter(|x| x.is_some())
            .count();

        let so3_block = SpatiotemporalSo3Evidence {
            status: if manifold_count >= 3 && r_stat > 0.05 {
                "available"
            } else if manifold_count >= 2 {
                "partial"
            } else {
                "missing"
            },
            stream_support: StreamSupport {
                n_streams_supporting: stream_support.n_streams_supporting,
                supporting_streams: stream_support.supporting_streams.clone(),
                per_stream_spike_counts: stream_support.per_stream_spike_counts.clone(),
                per_stream_voxel_counts: stream_support.per_stream_voxel_counts.clone(),
            },
            temporal_support: TemporalSupport {
                first_observed_step: temporal_support.first_observed_step,
                last_observed_step: temporal_support.last_observed_step,
                recurrence_count: temporal_support.recurrence_count,
                chunk_support: temporal_support.chunk_support.clone(),
                n_phase_bins_nonzero: temporal_support.n_phase_bins_nonzero,
            },
            phase_support: PhaseSupport {
                phase_bits_histogram: phase_support.phase_bits_histogram,
                rayleigh_r_stat: phase_support.rayleigh_r_stat,
                mean_phase_radians: phase_support.mean_phase_radians,
                n_hot: phase_support.n_hot,
                n_cold: phase_support.n_cold,
            },
            so3_support,
            centroid_manifold: CentroidManifold {
                whole_site, lining_mass, causal_driver,
                hot_phase, cold_phase, return_phase: None, ligand_adjacent,
            },
            field_completeness: FieldCompleteness {
                spikes: field_completeness.spikes,
                phase_bits: field_completeness.phase_bits,
                warp_matrix: field_completeness.warp_matrix,
                asc_vectors: field_completeness.asc_vectors,
                forces: field_completeness.forces,
                signal_grid: field_completeness.signal_grid,
                kcc: field_completeness.kcc,
            },
        };

        // Materialization criteria.
        let mut blocking: Vec<&'static str> = Vec::new();
        if whole_site.is_none() { blocking.push("missing_whole_site_centroid"); }
        if n_streams_supporting < min_supporting_streams { blocking.push("insufficient_stream_support"); }
        if ev.n_spikes < 1000 { blocking.push("insufficient_spike_support"); }
        if ev.chunk_set.is_empty() { blocking.push("no_temporal_support"); }
        if !has_min_centroid_views { blocking.push("manifold_centroids_below_minimum"); }

        let level: &'static str = if blocking.is_empty() {
            if gt_status == "available" {
                "materialized_dynamic_site_with_validation"
            } else {
                "materialized_dynamic_site"
            }
        } else if !blocking.contains(&"missing_whole_site_centroid")
                  && !blocking.contains(&"no_temporal_support") {
            "candidate_spatiotemporal_partial"
        } else {
            "candidate_density_only"
        };

        if level.starts_with("materialized_dynamic_site") {
            let centroid = whole_site.unwrap();
            let representative_residues: Vec<i32> = lining_residues.iter().take(5).cloned().collect();
            // Accuracy block: only populate DCC if ground truth is valid.
            let accuracy: serde_json::Value = if let Some(lig) = ligand_adjacent {
                let dcc_whole = whole_site.map(|c| dist3(&c, &lig));
                let dcc_lining = lining_mass.map(|c| dist3(&c, &lig));
                let dcc_driver = causal_driver.map(|c| dist3(&c, &lig));
                let dcc_hot = hot_phase.map(|c| dist3(&c, &lig));
                let dcc_cold = cold_phase.map(|c| dist3(&c, &lig));
                let best = [dcc_whole, dcc_lining, dcc_driver, dcc_hot, dcc_cold]
                    .into_iter()
                    .flatten()
                    .fold(f32::INFINITY, f32::min);
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
                "warp_matrix_support_deferred",
                "asc_vector_support_deferred",
                "force_direction_support_deferred",
                if causal_driver.is_some() { "causal_driver_partial_kcc_only_active_causal" } else { "causal_driver_unavailable" },
            ];

            let provenance = Provenance {
                source_manifest: args.manifest.display().to_string(),
                run_id: manifest.run_id.clone(),
                target: manifest.target.clone(),
                seed_voxel_index: ev.seed_voxel,
                seed_density_peak_rank: cand_idx as u32,
                spikes_files_consumed: spikes_files_consumed.clone(),
                kcc_files_consumed: kcc_files_consumed.clone(),
                topology_input: manifest.topology_input.clone(),
            };

            materialized.push(MaterializedSite {
                site_id: format!("site_{:03}", cand_idx),
                rank: cand_idx as u32,
                materialization_status: "materialized",
                materialization_level: level,
                centroid_xyz: centroid,
                centroid_manifold: CentroidManifold {
                    whole_site, lining_mass, causal_driver,
                    hot_phase, cold_phase, return_phase: None, ligand_adjacent,
                },
                representative_residues,
                lining_residues,
                driver_residues,
                n_spikes: ev.n_spikes,
                intensity_sum: ev.intensity_sum,
                spike_support: serde_json::json!({
                    "n_uv": ev.n_uv,
                    "n_lif": ev.n_lif,
                    "n_other_source": ev.n_other_source,
                    "n_aromatic": {
                        "TRP": ev.n_arom[0],
                        "TYR": ev.n_arom[1],
                        "PHE": ev.n_arom[2],
                        "SS":  ev.n_arom[3],
                        "none": ev.n_arom[4],
                    },
                }),
                stream_support,
                phase_support,
                spatiotemporal_so3_evidence: so3_block,
                field_completeness,
                provenance,
                limitations,
                accuracy,
            });
        } else {
            non_materialized.push(NonMaterializedCandidate {
                seed_peak_id: ev.seed_peak_id,
                seed_voxel: ev.seed_voxel,
                seed_centroid_a: ev.seed_centroid_a,
                n_spikes: ev.n_spikes,
                n_streams_supporting,
                materialization_level: level,
                blocking_reasons: blocking,
            });
        }
    }

    // Re-rank materialized sites by spike support × stream support.
    materialized.sort_by(|a, b| {
        let ascore = a.n_spikes as f64 * a.stream_support.n_streams_supporting as f64;
        let bscore = b.n_spikes as f64 * b.stream_support.n_streams_supporting as f64;
        bscore.partial_cmp(&ascore).unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, s) in materialized.iter_mut().enumerate() {
        s.rank = i as u32;
        s.site_id = format!("site_{:03}", i);
    }

    // ─── Emit binding_sites.materialized.json ──────────────────────────────
    let materialized_path = output_dir.join("binding_sites.materialized.json");
    let materialized_status: &str = if materialized.is_empty() {
        "no_sites_met_criteria"
    } else if gt_status == "available" {
        "materialized_from_md_evidence_with_validation"
    } else {
        "materialized_from_md_evidence"
    };
    let materialized_json = serde_json::json!({
        "schema_version":   1,
        "schema_kind":      "pathb_binding_sites_materialized",
        "status":           materialized_status,
        "target":           manifest.target,
        "run_id":           manifest.run_id,
        "source_manifest":  args.manifest.display().to_string(),
        "site_count":       materialized.len(),
        "binding_sites":    materialized,
        "non_materialized_candidates": non_materialized,
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
    eprintln!("✓ wrote {} (sites={}, non_materialized={})",
              materialized_path.display(),
              materialized_json["site_count"], non_materialized.len());

    // ─── Emit site_accuracy_report.json ────────────────────────────────────
    let accuracy_path = output_dir.join("site_accuracy_report.json");
    let accuracy_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_site_accuracy_report",
        "target":         manifest.target,
        "run_id":         manifest.run_id,
        "ground_truth_status":     gt_status,
        "ground_truth_skip_reason": gt_skip_reason,
        "ligand_centroid_used":    ground_truth.as_ref().and_then(|gt| gt.ligand_centroid),
        "n_materialized_sites":    materialized_json["site_count"],
        "validation_computable":   gt_status == "available",
        "per_site_accuracy":       if gt_status == "available" {
            serde_json::Value::Array(materialized_json["binding_sites"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(|s| serde_json::json!({
                    "site_id":  s["site_id"],
                    "rank":     s["rank"],
                    "centroid_xyz": s["centroid_xyz"],
                    "accuracy": s["accuracy"],
                }))
                .collect())
        } else {
            serde_json::Value::Null
        },
        "summary": {
            "n_excellent_le2A":    serde_json::Value::Null,
            "n_acceptable_le3A":   serde_json::Value::Null,
            "n_useful_le5A":       serde_json::Value::Null,
            "n_lenient_le8A":      serde_json::Value::Null,
            "n_miss_gt8A":         serde_json::Value::Null,
            "best_grade_overall":  serde_json::Value::Null,
            "grade_note":          if gt_status == "available" {
                "summary aggregation deferred to follow-up commit; per-site grades present in per_site_accuracy"
            } else {
                "validation not computable: reference unavailable per ground_truth.json"
            },
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
        "schema_version": 1,
        "schema_kind":    "pathb_materialization_field_completeness",
        "evidence_sources": {
            "spikes":         { "status": "available", "format": "PRSPK001", "streams_present": spikes_files_consumed.len() },
            "kcc_v2full":     { "status": if !kcc_per_stream.is_empty() { "partial" } else { "missing" }, "format": "PRKCC001", "streams_present": kcc_per_stream.len() },
            "signal_grid":    { "status": if manifest.artifacts.iter().any(|a| a.present && a.kind == "signal_grid") { "partial" } else { "missing" }, "format": "PRSGD001" },
            "warp_matrix":    { "status": "deferred", "reason": "raw blob format not parsed in this commit" },
            "forces_final":   { "status": "deferred", "reason": "raw blob format not parsed in this commit" },
            "asc_vectors":    { "status": "deferred", "reason": "raw blob format not parsed in this commit" },
            "adaptive_dt":    { "status": "deferred", "reason": "raw blob format not parsed in this commit" },
            "protocol_state": { "status": "available_json", "reason": "loaded for run config; not used in materialization yet" },
            "bocpd_log":      { "status": "available_jsonl", "reason": "not used in materialization yet" },
            "topology":       { "status": if topology.is_some() { "available" } else { "missing" }, "format": "json", "path": manifest.topology_input },
            "ground_truth":   { "status": gt_status, "skip_reason": gt_skip_reason },
        },
        "n_candidates_seeded":           candidates.len(),
        "n_candidates_materialized":     materialized.len(),
        "n_candidates_non_materialized": non_materialized.len(),
        "promotion_summary": {
            "materialized_dynamic_site_with_validation": materialized.iter()
                .filter(|s| s.materialization_level == "materialized_dynamic_site_with_validation").count(),
            "materialized_dynamic_site": materialized.iter()
                .filter(|s| s.materialization_level == "materialized_dynamic_site").count(),
            "candidate_spatiotemporal_partial": non_materialized.iter()
                .filter(|c| c.materialization_level == "candidate_spatiotemporal_partial").count(),
            "candidate_density_only": non_materialized.iter()
                .filter(|c| c.materialization_level == "candidate_density_only").count(),
        },
        "ligsite_role": "intentionally_excluded_per_directive",
    });
    {
        let f = File::create(&mfc_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &mfc_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", mfc_path.display());

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
