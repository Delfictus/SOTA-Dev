// Path B materializer skeleton (cb8285de+1).
//
// Offline consumer of the MD-only evidence handoff bundle produced by
// `nhs_rt_full --md-only-evidence`. Reads md_evidence_manifest.json,
// validates every per-stream PRSPK001 / PRSGD001 / PRKCC001 binary
// envelope, computes honest summary statistics, and emits
//
//   materialization_report.json          — per-artifact validation + summaries
//   site_candidates.json                  — voxel-density density peaks (NOT ML-reranked)
//   validation_inputs.json                — paths/inputs Path B downstream needs
//   field_completeness_report.pathb.json  — Path B view of what's available
//
// HONEST SCOPE — what this skeleton DOES NOT do:
//
//   * No XGBoost rerank. No LIGSITE. No PH centroid refinement. No
//     find_neighbors_union DBSCAN-style clustering. No site-level
//     materialization. The legacy aggregation pipeline at
//     `nhs_rt_full.rs:13093+` is the algorithmic spec for those
//     stages; porting them is a follow-up commit.
//
//   * The "site_candidates" emitted here are coarse voxel-density
//     peaks computed from spike positions only. They are explicitly
//     labelled as `kind: "voxel_density_peak"` so downstream tooling
//     cannot mistake them for materialized binding sites.
//
//   * `binding_sites.json` (materialized form) is NEVER emitted by
//     this binary. The directive explicitly forbids faking it; if
//     downstream consumers want materialized sites, they must run
//     subsequent Path B stages that aren't in this commit.
//
// All field-completeness gaps are recorded in the output JSONs with
// explicit `status: "deferred" | "absent" | "available"` markers.

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

    Ok(())
}
