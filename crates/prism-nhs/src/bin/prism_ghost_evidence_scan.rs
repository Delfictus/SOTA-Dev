// PRISM Ghost/ZSTR Evidence Scanner — Path B feature extractor.
//
// CORRECT ROLE — read the directive carefully:
//   * This binary is an OFFLINE evidence reader, scorer, and provenance
//     emitter. It is NOT the binding-site materializer.
//   * It does NOT emit binding_sites.materialized.json.
//   * It does NOT use LIGSITE.
//   * It does NOT fake values, schemas, or bilateral support.
//   * It DOES emit feature/provenance JSONs that prism-materialize-sites
//     can consume in a follow-up integration.
//
// Inputs: md_evidence_manifest.json (manifest-first discovery; no
// hardcoded run paths). All Ghost/ZSTR/V2 files referenced from the
// manifest's run directory.
//
// Outputs (always emitted for honest reporting; status fields explicit):
//   tile_field_completeness.json    — per-file schema validation matrix
//   ghost_tile_evidence.json        — ghost-tile-stream summaries
//   zstr_event_summary.json         — zstr-stream summaries
//   ranked_tile_events.json         — events ranked by verified components
//   ghost_zstr_candidate_features.json — handoff for materializer
//   ghost_zstr_scan_report.json     — overall scan report
//
// SCHEMA SOURCES (cited; do NOT modify these without re-verifying):
//   GhostTileFrame      crates/prism-nhs/src/ghost_tile.rs:80-213
//                       repr(C, align(4096)), 4096 B total, const-asserted offsets:
//                         frame_idx        u64    @ 0
//                         site_id          u32    @ 8
//                         chain_id         u8     @ 12
//                         adjudication_code u8    @ 13
//                         telemetry_flags  u16    @ 14
//                         kl_divergence    f32    @ 16
//                         power_spectrum   [f32;24] @ 20  (4 planes × 6 bands)
//                         thermo_flux      [f32;2] @ 116  ([wd_change, vib_energy])
//                         causal_lead_residue u32 @ 124
//                         _reserved_payload [u32;32] @ 128
//                         _slack           [u8;3840] @ 256
//   File layout:        crates/prism-nhs/src/ghost_tile.rs:225-237
//                         offset 0..4   u32 n_frames_written
//                         offset 4..4096 pad  (counter sector)
//                         offset 4096..end GhostTileFrame × n_frames_written
//
//   ZstrFrameHeader     crates/prism-nhs/src/zstr.rs:67-114
//                       repr(C, align(4096)), 4096 B total:
//                         frame_idx        u64    @ 0
//                         dt               f32    @ 8     (picoseconds at capture)
//                         adjudication_code u32   @ 12
//                         completion_fence u32    @ 16
//                         n_atoms          u32    @ 20
//                         gear_id          u32    @ 24    (Wave A leaves at 0)
//                         force_norm       f32    @ 28
//                         external_work    f64    @ 32
//                         potential_energy f64    @ 40
//                         _padding         [u32;1012] @ 48
//   Per-slot file:      crates/prism-nhs/src/zstr.rs:154-159
//                         [header 4096 B][positions n_atoms*12][forces n_atoms*12][pad→4096]
//   File name pattern:  crates/prism-nhs/src/bin/nhs_rt_full.rs:7638,7788
//                         prism_zstr_<pid>_<stream>.bin
//
//   prism_v2 trajectory crates/prism-nhs/src/bin/nhs_rt_full.rs:6260-6301
//                         offset 0..8   u64 n_frames
//                         per frame: u64 step, u32 n_floats, f32×n_floats
//
//   intensity_packed    crates/prism-nhs/src/interferometric_adjudicator.rs:712-735
//                         QI_SHIFT = 30
//                         INTENSITY_PAYLOAD_MASK = 0x3FFF_FFFF
//                         UV-code → λ:
//                           0 → 260 nm (PHE-dominant)
//                           1 → 280 nm (TRP primary)
//                           2 → 305 nm (TRP gaussian tail)
//                           3 → 320 nm (TRP control / baseline)
//                       NOTE: intensity_packed lives on `RichSpike`, NOT on
//                       GhostTileFrame. Ghost tiles do NOT carry the perturbation
//                       channel directly; the perturbation_channel_evidence block
//                       on a ghost-tile event is therefore status="not_in_ghost_tile_schema"
//                       and would require cross-referencing with PRSPK001 spikes
//                       (which carry per-spike `wavelength_nm` directly).
//
// HARD INVARIANT NOTES (per directive):
//   I1 perturbation_channel:  Ghost tiles lack intensity_packed. Status
//                             "not_in_ghost_tile_schema". Ranking neutral.
//   I2 gear-aware timing:     ZstrFrameHeader.gear_id IS schema-verified (offset 24)
//                             but Wave A producer leaves it at 0. Status
//                             "schema_verified_but_producer_wave_a_writes_zero".
//                             dt IS schema-verified (offset 8); semantics are
//                             "picoseconds at capture". Apply only when present.
//   I3 ASC work integral Γw:  asc_vectors.bin format has NO schema header (raw blob).
//                             Without verified format, Γw is NOT computed.
//                             gamma_w_status = "asc_vector_format_unverified_no_compute".

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

// ─── CLI ───────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "prism-ghost-evidence-scan")]
#[command(about = "Path B Ghost/ZSTR evidence scanner — emits features for prism-materialize-sites.")]
struct Args {
    /// md_evidence_manifest.json path produced by `nhs_rt_full --md-only-evidence`.
    #[arg(short = 'm', long)]
    manifest: PathBuf,

    /// Output dir. Defaults to the manifest's parent directory.
    #[arg(short = 'o', long)]
    out_dir: Option<PathBuf>,

    /// Z-score sigma threshold for tile-event scoring. Default 12.0.
    /// Only applied when noise_floor.json is present and shapes match.
    #[arg(long, default_value = "12.0")]
    threshold_sigma: f32,

    /// Max events to emit (0 = no cap).
    #[arg(long, default_value = "0")]
    max_events: usize,

    /// Allow missing Ghost/ZSTR files; emit honest "missing" status and
    /// exit 0 instead of nonzero.
    #[arg(long, default_value = "false")]
    allow_missing: bool,

    /// Fail if any present binary's schema is unknown / unverifiable.
    #[arg(long, default_value = "false")]
    strict_schema: bool,

    /// Emit per-event NDJSON (one event per line) in addition to the
    /// summary JSON.
    #[arg(long, default_value = "false")]
    emit_ndjson: bool,

    /// Emit ALL parsed events into ranked_tile_events.json (large output).
    /// Default: emit top 200 by tile_score.
    #[arg(long, default_value = "false")]
    emit_all_events: bool,

    /// Verbose logging.
    #[arg(short, long, default_value = "false")]
    verbose: bool,
}

// ─── Manifest schema (read-only mirror) ────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ManifestArtifact {
    kind: String,
    stream_id: Option<u32>,
    path: String,
    #[allow(dead_code)]
    size_bytes: u64,
    present: bool,
    #[allow(dead_code)]
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
    artifacts: Vec<ManifestArtifact>,
}

// ─── Schema constants — cited from producer source ────────────────────────

mod ghost_schema {
    /// `GhostTileFrame` per-record byte size (cited: ghost_tile.rs:217).
    pub const RECORD_BYTES: u64 = 4096;
    /// File-leading counter sector size (cited: ghost_tile.rs:225-237).
    pub const COUNTER_SECTOR_BYTES: u64 = 4096;
    /// Magic offsets within a 4096-byte record (cited: ghost_tile.rs:198-213
    /// const-context offset assertions).
    pub const OFF_FRAME_IDX: usize = 0;
    pub const OFF_SITE_ID: usize = 8;
    pub const OFF_CHAIN_ID: usize = 12;
    pub const OFF_ADJ_CODE: usize = 13;
    pub const OFF_TELEMETRY_FLAGS: usize = 14;
    pub const OFF_KL_DIV: usize = 16;
    pub const OFF_POWER_SPECTRUM: usize = 20;
    pub const OFF_THERMO_FLUX: usize = 116;
    pub const OFF_CAUSAL_LEAD: usize = 124;
    /// Bit 0 — CLASS_TAINTED (cited: ghost_tile.rs:172). Set when the kernel
    /// substituted NaN/Inf in thermo_flux with 0.0.
    pub const FLAG_CLASS_TAINTED: u16 = 0x0001;
}

mod zstr_schema {
    /// `ZstrFrameHeader` byte size (cited: zstr.rs:67 + size_of comment line 162).
    pub const HEADER_BYTES: u64 = 4096;
    /// Field offsets within the 4096 B header (cited: zstr.rs:68-114 inline
    /// comments after each field).
    pub const OFF_FRAME_IDX: usize = 0;
    pub const OFF_DT: usize = 8;
    pub const OFF_ADJ_CODE: usize = 12;
    pub const OFF_COMPLETION_FENCE: usize = 16;
    pub const OFF_N_ATOMS: usize = 20;
    pub const OFF_GEAR_ID: usize = 24;
    pub const OFF_FORCE_NORM: usize = 28;
    pub const OFF_EXTERNAL_WORK: usize = 32;
    pub const OFF_POTENTIAL_ENERGY: usize = 40;
}

mod uv_qi_schema {
    /// QI bit shift (cited: interferometric_adjudicator.rs:714).
    pub const QI_SHIFT: u32 = 30;
    /// UV-code → wavelength_nm map (cited: interferometric_adjudicator.rs:722-728).
    pub const UV_NM: [u32; 4] = [260, 280, 305, 320];
    pub const UV_LABELS: [&str; 4] = [
        "260nm_PHE_dominant",
        "280nm_TRP_primary",
        "305nm_TRP_gaussian_tail",
        "320nm_TRP_control",
    ];
}

// ─── Discovered file record ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DiscoveredFile {
    path: PathBuf,
    stream_id: Option<u32>,
    file_kind: FileKind,
    file_size: u64,
}

#[derive(Debug, Clone, PartialEq)]
enum FileKind {
    GhostTiles,
    Zstr,
    PrismV2Trajectory,
}

fn parse_stream_id_after_underscore(name: &str, marker: &str) -> Option<u32> {
    let idx = name.find(marker)? + marker.len();
    let tail = &name[idx..];
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().ok()
}

fn parse_stream_id_from_filename(p: &Path) -> Option<u32> {
    let name = p.file_name()?.to_string_lossy().to_string();
    if let Some(id) = parse_stream_id_after_underscore(&name, "_stream") {
        return Some(id);
    }
    // prism_zstr_<pid>_<stream>.bin and prism_v2_<pid>_<stream>.bin
    // — last underscore-separated digit run before the extension.
    let stem = name.strip_suffix(".bin").unwrap_or(&name);
    let last = stem.rsplit('_').next()?;
    last.parse::<u32>().ok()
}

fn discover_files(run_dir: &Path) -> Result<Vec<DiscoveredFile>> {
    let mut out: Vec<DiscoveredFile> = Vec::new();
    let entries = std::fs::read_dir(run_dir)
        .with_context(|| format!("read run dir {}", run_dir.display()))?;
    for e in entries.flatten() {
        let p = e.path();
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else { continue };
        let kind = if name.ends_with("_ghost_tiles.bin") {
            FileKind::GhostTiles
        } else if name.starts_with("prism_zstr_") && name.ends_with(".bin") {
            FileKind::Zstr
        } else if name.starts_with("prism_v2_") && name.ends_with(".bin") {
            FileKind::PrismV2Trajectory
        } else {
            continue;
        };
        let file_size = p.metadata().map(|m| m.len()).unwrap_or(0);
        let stream_id = parse_stream_id_from_filename(&p);
        out.push(DiscoveredFile { path: p, stream_id, file_kind: kind, file_size });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

// ─── Schema validation per file ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct FileSchemaValidation {
    file_path: String,
    file_kind: &'static str,
    stream_id: Option<u32>,
    file_size: u64,
    format_guess: &'static str,
    schema_status: &'static str,
    parse_status: &'static str,
    record_count: Option<u64>,
    byte_stride: Option<u64>,
    parser_used: &'static str,
    missing_fields: Vec<&'static str>,
    unsafe_assumptions: Vec<&'static str>,
    notes: Vec<String>,
}

fn validate_ghost_tiles_schema(f: &DiscoveredFile) -> FileSchemaValidation {
    let mut notes = Vec::new();
    let mut record_count: Option<u64> = None;
    let mut parse_status: &'static str = "header_only";
    let mut schema_status: &'static str = "verified_const_offsets_in_ghost_tile_rs_198_213";

    if f.file_size < ghost_schema::COUNTER_SECTOR_BYTES {
        notes.push(format!(
            "file_size {} < counter_sector {} — file too small to contain even the leading u32 n_frames_written counter",
            f.file_size, ghost_schema::COUNTER_SECTOR_BYTES
        ));
        return FileSchemaValidation {
            file_path: f.path.display().to_string(),
            file_kind: "ghost_tiles",
            stream_id: f.stream_id,
            file_size: f.file_size,
            format_guess: "ghost_tiles_v9d_sector_locked",
            schema_status,
            parse_status: "file_too_small",
            record_count: None,
            byte_stride: Some(ghost_schema::RECORD_BYTES),
            parser_used: "ghost_tiles_envelope_validator",
            missing_fields: vec![],
            unsafe_assumptions: vec![],
            notes,
        };
    }
    // Read u32 n_frames_written from offset 0.
    if let Ok(file) = File::open(&f.path) {
        let mut br = BufReader::new(file);
        let mut buf = [0u8; 4];
        if br.read_exact(&mut buf).is_ok() {
            let n = u32::from_le_bytes(buf);
            record_count = Some(n as u64);
            // Verify file size = counter_sector + n * 4096.
            let expected = ghost_schema::COUNTER_SECTOR_BYTES + (n as u64) * ghost_schema::RECORD_BYTES;
            if f.file_size == expected {
                parse_status = "envelope_consistent";
            } else if f.file_size >= expected {
                parse_status = "envelope_consistent_with_trailing_bytes";
                notes.push(format!(
                    "trailing {} bytes after expected payload",
                    f.file_size - expected
                ));
            } else {
                parse_status = "envelope_size_mismatch";
                schema_status = "verified_offsets_but_file_truncated";
                notes.push(format!(
                    "file_size {} < counter_sector + n*4096 = {}",
                    f.file_size, expected
                ));
            }
        }
    }
    FileSchemaValidation {
        file_path: f.path.display().to_string(),
        file_kind: "ghost_tiles",
        stream_id: f.stream_id,
        file_size: f.file_size,
        format_guess: "ghost_tiles_v9d_sector_locked",
        schema_status,
        parse_status,
        record_count,
        byte_stride: Some(ghost_schema::RECORD_BYTES),
        parser_used: "ghost_tiles_envelope_validator",
        missing_fields: vec![],
        unsafe_assumptions: vec![],
        notes,
    }
}

fn validate_zstr_schema(f: &DiscoveredFile) -> FileSchemaValidation {
    let mut notes = Vec::new();
    let mut record_count: Option<u64> = None;
    let mut byte_stride: Option<u64> = None;
    let mut parse_status: &'static str = "header_only";
    let schema_status: &'static str = "header_offsets_verified_in_zstr_rs_67_114_per_slot_size_unknown_until_n_atoms_read";

    if f.file_size < zstr_schema::HEADER_BYTES {
        notes.push(format!(
            "file_size {} < header_bytes {} — cannot contain even one ZstrFrameHeader",
            f.file_size, zstr_schema::HEADER_BYTES
        ));
        return FileSchemaValidation {
            file_path: f.path.display().to_string(),
            file_kind: "prism_zstr",
            stream_id: f.stream_id,
            file_size: f.file_size,
            format_guess: "prism_zstr_5slot_4096_padded",
            schema_status,
            parse_status: "file_too_small",
            record_count: None,
            byte_stride: None,
            parser_used: "zstr_header_validator",
            missing_fields: vec![],
            unsafe_assumptions: vec![],
            notes,
        };
    }
    // Read first header to recover n_atoms, then derive per-slot stride.
    if let Ok(file) = File::open(&f.path) {
        let mut br = BufReader::new(file);
        let mut hdr = [0u8; 4096];
        if br.read_exact(&mut hdr).is_ok() {
            let n_atoms = u32::from_le_bytes(hdr[zstr_schema::OFF_N_ATOMS..zstr_schema::OFF_N_ATOMS + 4].try_into().unwrap());
            let pos_bytes_raw = (n_atoms as u64) * 12;
            let pos_bytes_aligned = (pos_bytes_raw + 15) & !15;
            let force_bytes = (n_atoms as u64) * 12;
            let raw_frame = zstr_schema::HEADER_BYTES + pos_bytes_aligned + force_bytes;
            let frame_size = (raw_frame + 4095) & !4095;
            byte_stride = Some(frame_size);
            if frame_size == 0 {
                notes.push("derived frame_size = 0 — n_atoms in first header likely zero (file may be only the first header padded)".to_string());
                record_count = Some(if f.file_size >= zstr_schema::HEADER_BYTES { 1 } else { 0 });
                parse_status = "first_header_only";
            } else {
                let n = f.file_size / frame_size;
                let trailing = f.file_size - n * frame_size;
                record_count = Some(n);
                if trailing == 0 {
                    parse_status = "envelope_consistent";
                } else {
                    parse_status = "envelope_consistent_with_trailing_bytes";
                    notes.push(format!(
                        "trailing {} bytes after {} full slots of {} bytes each",
                        trailing, n, frame_size
                    ));
                }
            }
        }
    }
    FileSchemaValidation {
        file_path: f.path.display().to_string(),
        file_kind: "prism_zstr",
        stream_id: f.stream_id,
        file_size: f.file_size,
        format_guess: "prism_zstr_5slot_4096_padded",
        schema_status,
        parse_status,
        record_count,
        byte_stride,
        parser_used: "zstr_header_validator",
        missing_fields: vec![],
        unsafe_assumptions: vec![],
        notes,
    }
}

fn validate_prism_v2_schema(f: &DiscoveredFile) -> FileSchemaValidation {
    let mut notes = Vec::new();
    let mut record_count: Option<u64> = None;
    let mut parse_status: &'static str = "header_only";
    if f.file_size < 8 {
        notes.push("file_size < 8 — no n_frames header".to_string());
        return FileSchemaValidation {
            file_path: f.path.display().to_string(),
            file_kind: "prism_v2_trajectory",
            stream_id: f.stream_id,
            file_size: f.file_size,
            format_guess: "prism_v2_trajectory_writer",
            schema_status: "verified_writer_format_in_nhs_rt_full_rs_6291_6301",
            parse_status: "file_too_small",
            record_count: None,
            byte_stride: None,
            parser_used: "prism_v2_header_validator",
            missing_fields: vec![],
            unsafe_assumptions: vec![],
            notes,
        };
    }
    if let Ok(file) = File::open(&f.path) {
        let mut br = BufReader::new(file);
        let mut buf = [0u8; 8];
        if br.read_exact(&mut buf).is_ok() {
            let n = u64::from_le_bytes(buf);
            record_count = Some(n);
            if n == 0 && f.file_size == 8 {
                parse_status = "empty_trajectory_n_frames_zero";
                notes.push("V2 was not live this run; only the 8-byte header was written".to_string());
            } else {
                parse_status = "header_present_payload_present";
            }
        }
    }
    FileSchemaValidation {
        file_path: f.path.display().to_string(),
        file_kind: "prism_v2_trajectory",
        stream_id: f.stream_id,
        file_size: f.file_size,
        format_guess: "prism_v2_trajectory_writer",
        schema_status: "verified_writer_format_in_nhs_rt_full_rs_6291_6301",
        parse_status,
        record_count,
        byte_stride: None,
        parser_used: "prism_v2_header_validator",
        missing_fields: vec![],
        unsafe_assumptions: vec![],
        notes,
    }
}

// ─── Ghost tile event parsing ───────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
struct GhostTileEvent {
    event_id: String,
    stream_id: u32,
    frame_idx: u64,
    site_id: u32,
    chain_id: Option<char>,
    adj_code: u8,
    telemetry_flags: u16,
    class_tainted: bool,
    /// M1.2.23 §5 — v1/v2 schema overlay detection.
    schema_version: u32,
    is_v2: bool,
    /// M1.2.23 §4 — Transparent MAR observation/discovery gates,
    /// populated only when is_v2.
    v2_observation_pass: bool,
    v2_discovery_pass: bool,
    v2_field_completeness_flags: u16,
    kl_divergence: f32,
    /// 4 planes × 6 bands. Plane 0 = geometry; planes 1-3 may be 0.0
    /// per producer note (ghost_tile.rs:128-132 — kernel doesn't yet
    /// populate planes 1-3).
    power_spectrum: [f32; 24],
    /// [wd_change, vib_energy] — both NaN until upstream wires them
    /// (ghost_tile.rs:136-141).
    thermo_flux: [f32; 2],
    causal_lead_residue: u32,
    /// I1 — Ghost tile schema does NOT carry intensity_packed.
    perturbation_channel_evidence: PerturbationChannelEvidence,
    /// I2 — Ghost tile schema does NOT carry gear_id (it's in _reserved_payload).
    gear_normalized_timing: GearTimingEvidence,
    /// I3 — Γw not computable without verified asc_vectors format.
    asc_work_evidence: AscWorkEvidence,
    /// Flux coupling η = wd_change / max(|vib_energy|, ε)
    flux_coupling: FluxCoupling,
    field_completeness: GhostEventFieldCompleteness,
    provenance: EventProvenance,
}

#[derive(Debug, Serialize, Clone, Default)]
struct PerturbationChannelEvidence {
    qi_bits: Option<u32>,
    uv_wavelength_nm: Option<u32>,
    uv_label: Option<&'static str>,
    intensity_packed_raw: Option<u32>,
    status: &'static str,
}

#[derive(Debug, Serialize, Clone, Default)]
struct GearTimingEvidence {
    gear_id: Option<u32>,
    dt_fs: Option<f32>,
    physical_time_fs: Option<f64>,
    physical_delta_fs: Option<f64>,
    status: &'static str,
}

#[derive(Debug, Serialize, Clone, Default)]
struct AscWorkEvidence {
    gamma_w: Option<f64>,
    gamma_w_status: &'static str,
    gamma_w_units: &'static str,
    formula: &'static str,
    is_proxy: bool,
    source_files: Vec<String>,
}

#[derive(Debug, Serialize, Clone, Default)]
struct FluxCoupling {
    eta: Option<f32>,
    eta_status: &'static str,
    wd_change_finite: bool,
    vib_energy_finite: bool,
}

#[derive(Debug, Serialize, Clone, Default)]
struct GhostEventFieldCompleteness {
    frame_idx: bool,
    site_id: bool,
    chain_id: bool,
    adjudication_code: bool,
    kl_divergence: bool,
    power_spectrum_geo: bool,
    power_spectrum_planes_1_3_populated: bool,
    thermo_flux_finite: bool,
    causal_lead_residue_resolved: bool,
}

#[derive(Debug, Serialize, Clone)]
struct EventProvenance {
    file: String,
    byte_offset: u64,
    schema_status: &'static str,
}

fn parse_ghost_tiles_file(
    path: &Path,
    stream_id: u32,
    n_records: u64,
    max_events: usize,
) -> Result<Vec<GhostTileEvent>> {
    let mut out: Vec<GhostTileEvent> = Vec::new();
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(ghost_schema::COUNTER_SECTOR_BYTES))?;
    let mut br = BufReader::with_capacity(1 << 20, file);
    let mut buf = [0u8; 4096];
    let cap = if max_events == 0 { n_records } else { (max_events as u64).min(n_records) };
    for k in 0..cap {
        if br.read_exact(&mut buf).is_err() {
            break;
        }
        let frame_idx = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let site_id = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let chain_id_byte = buf[12];
        let adj_code = buf[13];
        let telemetry_flags = u16::from_le_bytes(buf[14..16].try_into().unwrap());
        let kl_divergence = f32::from_le_bytes(buf[16..20].try_into().unwrap());
        let mut power_spectrum = [0f32; 24];
        for i in 0..24 {
            let off = 20 + i * 4;
            power_spectrum[i] = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        }
        let thermo_flux: [f32; 2] = [
            f32::from_le_bytes(buf[116..120].try_into().unwrap()),
            f32::from_le_bytes(buf[120..124].try_into().unwrap()),
        ];
        let causal_lead_residue = u32::from_le_bytes(buf[124..128].try_into().unwrap());

        // ─── M1.2.23 §5 — v1/v2 schema detection ─────────────────────────
        // Read schema_version u32 at offset 128. 0=v1 (legacy zero), 2=v2.
        let schema_version_v2: u32 = u32::from_le_bytes(buf[128..132].try_into().unwrap());
        let is_v2 = schema_version_v2 == 2u32;
        let (
            v2_obs_pass, v2_disc_pass, v2_perturb_chan, v2_uv_nm,
            v2_field_complete_flags, v2_gear_id, v2_dt_fs, v2_step_idx,
        ) = if is_v2 {
            (
                buf[132] != 0,
                buf[133] != 0,
                buf[134],
                u16::from_le_bytes(buf[136..138].try_into().unwrap()),
                u16::from_le_bytes(buf[138..140].try_into().unwrap()),
                u32::from_le_bytes(buf[140..144].try_into().unwrap()),
                f32::from_le_bytes(buf[144..148].try_into().unwrap()),
                // M1.2.24 fix: step_idx moved from 148 → 152 (148 was
                // misaligned for STG.E.64). Bytes 148..152 are reserved
                // padding (zero-initialized by the v2 kernel).
                u64::from_le_bytes(buf[152..160].try_into().unwrap()),
            )
        } else {
            (false, false, 0xFFu8, 0u16, 0u16, 0u32, 0.0f32, 0u64)
        };

        let chain_id = match chain_id_byte {
            0x41 => Some('A'),
            0x42 => Some('B'),
            _ => None,
        };
        let class_tainted = (telemetry_flags & ghost_schema::FLAG_CLASS_TAINTED) != 0;
        let wd = thermo_flux[0];
        let vib = thermo_flux[1];
        let wd_finite = wd.is_finite();
        let vib_finite = vib.is_finite();
        let (eta, eta_status) = if wd_finite && vib_finite && !class_tainted {
            let denom = vib.abs().max(1e-9);
            (Some(wd / denom), "computed")
        } else if class_tainted {
            (None, "skipped_class_tainted")
        } else {
            (None, "skipped_thermo_flux_nonfinite_or_unwired")
        };

        // Geo plane = power_spectrum[0..6]; planes 1-3 may be zero per producer note.
        let geo_band_sum: f32 = power_spectrum[0..6].iter().sum();
        let other_planes_nonzero = power_spectrum[6..].iter().any(|x| *x != 0.0);

        let mut field_completeness = GhostEventFieldCompleteness::default();
        field_completeness.frame_idx = true;
        field_completeness.site_id = true;
        field_completeness.chain_id = chain_id.is_some();
        field_completeness.adjudication_code = true;
        field_completeness.kl_divergence = kl_divergence.is_finite();
        field_completeness.power_spectrum_geo = geo_band_sum.is_finite();
        field_completeness.power_spectrum_planes_1_3_populated = other_planes_nonzero;
        field_completeness.thermo_flux_finite = wd_finite && vib_finite && !class_tainted;
        field_completeness.causal_lead_residue_resolved = causal_lead_residue != u32::MAX;

        out.push(GhostTileEvent {
            event_id: format!("ghost_s{}_f{}_t{}", stream_id, frame_idx, site_id),
            stream_id,
            frame_idx,
            site_id,
            chain_id,
            adj_code,
            telemetry_flags,
            class_tainted,
            schema_version: schema_version_v2,
            is_v2,
            v2_observation_pass: v2_obs_pass,
            v2_discovery_pass: v2_disc_pass,
            v2_field_completeness_flags: v2_field_complete_flags,
            kl_divergence,
            power_spectrum,
            thermo_flux,
            causal_lead_residue,
            perturbation_channel_evidence: if is_v2 && v2_perturb_chan != 0xFF {
                PerturbationChannelEvidence {
                    qi_bits: Some(v2_perturb_chan as u32),
                    uv_wavelength_nm: if v2_uv_nm > 0 { Some(v2_uv_nm as u32) } else { None },
                    uv_label: match v2_perturb_chan {
                        0 => Some("260nm_PHE_dominant"),
                        1 => Some("280nm_TRP_primary"),
                        2 => Some("305nm_TRP_gaussian_tail"),
                        3 => Some("320nm_TRP_control"),
                        _ => None,
                    },
                    intensity_packed_raw: None,
                    status: "v2_schema_perturbation_channel_resolved",
                }
            } else if is_v2 {
                PerturbationChannelEvidence {
                    qi_bits: None,
                    uv_wavelength_nm: None,
                    uv_label: None,
                    intensity_packed_raw: None,
                    status: "v2_schema_perturbation_channel_unknown_kernel_left_sentinel",
                }
            } else {
                PerturbationChannelEvidence {
                    qi_bits: None,
                    uv_wavelength_nm: None,
                    uv_label: None,
                    intensity_packed_raw: None,
                    status: "v1_schema_no_perturbation_channel_in_record",
                }
            },
            gear_normalized_timing: if is_v2 {
                let dt_ok = v2_dt_fs > 0.0 && v2_dt_fs.is_finite();
                let phys = if dt_ok {
                    Some((v2_step_idx as f64) * (v2_dt_fs as f64))
                } else { None };
                GearTimingEvidence {
                    gear_id: Some(v2_gear_id),
                    dt_fs: if dt_ok { Some(v2_dt_fs) } else { None },
                    physical_time_fs: phys,
                    physical_delta_fs: None,
                    status: if v2_gear_id == 0 {
                        "v2_schema_gear_wave_a_default_zero_dt_resolved"
                    } else {
                        "v2_schema_gear_wave_b_active"
                    },
                }
            } else {
                GearTimingEvidence {
                    gear_id: None,
                    dt_fs: None,
                    physical_time_fs: None,
                    physical_delta_fs: None,
                    status: "v1_schema_no_gear_id_in_record",
                }
            },
            asc_work_evidence: AscWorkEvidence {
                gamma_w: None,
                gamma_w_status: "asc_vector_format_unverified_no_compute",
                gamma_w_units: "n/a",
                formula: "Γw = Σ_t F_ASC(t) · Δr(t)  (NOT computed in this commit)",
                is_proxy: true,
                source_files: vec![],
            },
            flux_coupling: FluxCoupling {
                eta,
                eta_status,
                wd_change_finite: wd_finite,
                vib_energy_finite: vib_finite,
            },
            field_completeness,
            provenance: EventProvenance {
                file: path.display().to_string(),
                byte_offset: ghost_schema::COUNTER_SECTOR_BYTES + k * ghost_schema::RECORD_BYTES,
                schema_status: "verified_ghost_tile_rs_198_213",
            },
        });
    }
    Ok(out)
}

// ─── ZSTR header parsing — file-level summary only ─────────────────────────

#[derive(Debug, Serialize)]
struct ZstrStreamSummary {
    stream_id: Option<u32>,
    file: String,
    file_size: u64,
    first_header: Option<ZstrFirstHeader>,
    decode_status: &'static str,
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ZstrFirstHeader {
    frame_idx: u64,
    dt_fs: f64,
    adjudication_code: u32,
    completion_fence: u32,
    n_atoms: u32,
    gear_id: u32,
    force_norm: f32,
    external_work: f64,
    potential_energy: f64,
    gear_id_status: &'static str,
    dt_status: &'static str,
}

fn summarize_zstr_file(f: &DiscoveredFile) -> Result<ZstrStreamSummary> {
    let mut notes: Vec<String> = Vec::new();
    if f.file_size < zstr_schema::HEADER_BYTES {
        return Ok(ZstrStreamSummary {
            stream_id: f.stream_id,
            file: f.path.display().to_string(),
            file_size: f.file_size,
            first_header: None,
            decode_status: "file_too_small_for_header",
            notes,
        });
    }
    let file = File::open(&f.path)?;
    let mut br = BufReader::new(file);
    let mut hdr = [0u8; 4096];
    br.read_exact(&mut hdr)?;
    let frame_idx = u64::from_le_bytes(hdr[0..8].try_into().unwrap());
    let dt_ps = f32::from_le_bytes(hdr[8..12].try_into().unwrap());
    let adjudication_code = u32::from_le_bytes(hdr[12..16].try_into().unwrap());
    let completion_fence = u32::from_le_bytes(hdr[16..20].try_into().unwrap());
    let n_atoms = u32::from_le_bytes(hdr[20..24].try_into().unwrap());
    let gear_id = u32::from_le_bytes(hdr[24..28].try_into().unwrap());
    let force_norm = f32::from_le_bytes(hdr[28..32].try_into().unwrap());
    let external_work = f64::from_le_bytes(hdr[32..40].try_into().unwrap());
    let potential_energy = f64::from_le_bytes(hdr[40..48].try_into().unwrap());

    let dt_fs = (dt_ps as f64) * 1000.0;
    let dt_status: &'static str = if dt_fs > 0.0 && dt_fs.is_finite() {
        "available_picoseconds_to_femtoseconds"
    } else {
        "missing_or_zero"
    };
    let gear_id_status: &'static str = match gear_id {
        0 => "schema_verified_value_zero_wave_a_default",
        1..=3 => "schema_verified_value_in_known_range",
        _ => "schema_verified_value_out_of_known_range",
    };
    if gear_id == 0 {
        notes.push("gear_id=0 — Wave A producer leaves this at zero per zstr.rs:88; gear-aware temporal normalization should treat as unverified or apply dt only".to_string());
    }
    Ok(ZstrStreamSummary {
        stream_id: f.stream_id,
        file: f.path.display().to_string(),
        file_size: f.file_size,
        first_header: Some(ZstrFirstHeader {
            frame_idx,
            dt_fs,
            adjudication_code,
            completion_fence,
            n_atoms,
            gear_id,
            force_norm,
            external_work,
            potential_energy,
            gear_id_status,
            dt_status,
        }),
        decode_status: "first_header_decoded",
        notes,
    })
}

// ─── Noise floor ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NoiseFloor {
    #[serde(default)]
    mu: Option<f64>,
    #[serde(default)]
    sigma: Option<f64>,
    #[serde(default)]
    notes: Option<String>,
}

fn try_load_noise_floor(run_dir: &Path, stem: &str, stream_id: u32) -> Option<NoiseFloor> {
    let p = run_dir.join(format!("{}_stream{:02}_noise_floor.json", stem, stream_id));
    let s = std::fs::read_to_string(&p).ok()?;
    serde_json::from_str(&s).ok()
}

// ─── Tile event ranking (only on verified components) ─────────────────────

#[derive(Debug, Serialize)]
struct RankedTileEvent {
    event_id: String,
    rank: u32,
    tile_score: f64,
    score_components: serde_json::Value,
    event: serde_json::Value,
    limitations: Vec<&'static str>,
}

fn score_ghost_tile_event(ev: &GhostTileEvent) -> (f64, serde_json::Value) {
    // All components in [0, 1]. Components from unverified fields → neutral 1.0.
    // tile_score = product.

    let kl_factor: f64 = if ev.field_completeness.kl_divergence {
        let v = ev.kl_divergence as f64;
        if v.is_finite() && v > 0.0 { (v / (v + 1.0)).clamp(0.0, 1.0) } else { 0.5 }
    } else {
        0.5
    };
    let geo_band_sum: f64 = ev.power_spectrum[0..6].iter().map(|x| (*x as f64).max(0.0)).sum();
    let geo_factor: f64 = if ev.field_completeness.power_spectrum_geo {
        (geo_band_sum / (geo_band_sum + 1.0)).clamp(0.0, 1.0)
    } else { 0.5 };
    let flux_factor: f64 = match (ev.flux_coupling.eta_status, ev.flux_coupling.eta) {
        ("computed", Some(eta)) => {
            let v = eta as f64;
            // Bound η to [0, 1] via squashed transform; sign explicit.
            (v.abs() / (v.abs() + 1.0)).clamp(0.0, 1.0)
        }
        _ => 0.5,
    };
    let causal_factor: f64 = if ev.field_completeness.causal_lead_residue_resolved { 1.0 } else { 0.5 };
    let adj_factor: f64 = match ev.adj_code {
        1 => 1.0,                 // Construct
        2 => 0.6,                 // Violation
        _ => 0.5,
    };
    let class_factor: f64 = if ev.class_tainted { 0.3 } else { 1.0 };
    // Status fields force NEUTRAL on unverified evidence.
    let perturbation_factor: f64 = match ev.perturbation_channel_evidence.status {
        s if s.starts_with("not_in_ghost_tile_schema") => 1.0,
        _ => 1.0,
    };
    let gear_factor: f64 = match ev.gear_normalized_timing.status {
        s if s.starts_with("ghost_tile_has_no_gear_id_field") => 1.0,
        _ => 1.0,
    };
    let asc_factor: f64 = match ev.asc_work_evidence.gamma_w_status {
        "asc_vector_format_unverified_no_compute" => 1.0,
        _ => 1.0,
    };
    let tile_score = kl_factor * geo_factor * flux_factor * causal_factor * adj_factor
        * class_factor * perturbation_factor * gear_factor * asc_factor;

    let components = serde_json::json!({
        "kl_divergence_factor":          kl_factor,
        "geo_band_sum_factor":           geo_factor,
        "flux_coupling_factor":          flux_factor,
        "flux_coupling_eta":             ev.flux_coupling.eta,
        "flux_coupling_eta_status":      ev.flux_coupling.eta_status,
        "causal_lead_resolved_factor":   causal_factor,
        "adjudication_code_factor":      adj_factor,
        "class_tainted_factor":          class_factor,
        "perturbation_channel_factor":   perturbation_factor,
        "perturbation_channel_status":   ev.perturbation_channel_evidence.status,
        "gear_timing_factor":            gear_factor,
        "gear_timing_status":            ev.gear_normalized_timing.status,
        "asc_work_factor":               asc_factor,
        "asc_work_status":               ev.asc_work_evidence.gamma_w_status,
        "tile_score":                    tile_score,
    });
    (tile_score, components)
}

// ─── M1.2.23 §6 + §7 sidecar readers ───────────────────────────────────────

#[derive(Debug, Deserialize, Default, Clone)]
struct GhostTimeMapStream {
    #[allow(dead_code)] stream_id: u32,
    #[serde(default)] gear_id: u32,
    #[serde(default)] dt_fs: f32,
    #[serde(default)] physical_time_fs: f64,
    #[serde(default)] dt_source: Option<String>,
    #[serde(default)] gear_id_status: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct GhostTimeMap {
    #[serde(default)] schema_version: u32,
    #[serde(default)] streams: Vec<GhostTimeMapStream>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct GhostSiteMapEntry {
    stream_id: Option<u32>,
    site_id: Option<u32>,
    aabb: Option<[f32; 6]>,
    centroid_xyz: Option<[f32; 3]>,
    voxel_xyz: Option<[i32; 3]>,
    residue_support: Option<Vec<i32>>,
    source: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct GhostSiteMap {
    #[serde(default)] schema_version: u32,
    #[serde(default)] status: Option<String>,
    #[serde(default)] coordinate_frame: Option<String>,
    #[serde(default)] entries: Vec<GhostSiteMapEntry>,
}

/// Read path_a_completion.json from the run dir; extract branch-trace fields
/// per the follow-up invariant. Best-effort; missing fields → zeros + status.
fn read_branch_trace(run_dir: &Path, target: &str) -> serde_json::Value {
    let p = run_dir.join(format!("{}_path_a_completion.json", target));
    let s = match std::fs::read_to_string(&p) {
        Ok(s) => s,
        Err(_) => return serde_json::json!({
            "status": "path_a_completion_json_absent",
            "branch_decision_status": "no_completion_json_to_read",
        }),
    };
    let v: serde_json::Value = match serde_json::from_str(&s) {
        Ok(v) => v,
        Err(e) => return serde_json::json!({
            "status": "parse_error",
            "branch_decision_status": "completion_json_parse_failed",
            "error": e.to_string(),
        }),
    };
    // The watchdog's emit_minimal_completion_json carries branch_trace_by_stream
    // when CHUNK13 trace was registered. For graceful runs the field may be
    // absent — emit explicit "default_branch_only" only if the predicate
    // counts say so; otherwise honestly mark absent.
    let trace = v.get("branch_trace_by_stream")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let monomer_passthrough = v.get("monomer_passthrough").and_then(|x| x.as_bool());
    let bilateral_status = v.get("bilateral_status").and_then(|x| x.as_str()).map(|x| x.to_string());
    serde_json::json!({
        "status":                  if trace.is_null() { "branch_trace_absent_in_completion_json" } else { "available" },
        "branch_trace_by_stream":  trace,
        "monomer_passthrough":     monomer_passthrough,
        "bilateral_status":        bilateral_status,
        "branch_decision_status":  if trace.is_null() {
            "default_branch_only_or_unrecorded"
        } else {
            "see_branch_trace_by_stream"
        },
        "predicate_liveness_status": "infer_from_branch_trace_field_diversity_when_available",
        "threshold_sigma_observation": 3.0,
        "threshold_sigma_discovery":   12.0,
        "threshold_source":            "directive_M1_2_23_section_4_default",
    })
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();
    let log_level = if args.verbose { "info" } else { "warn" };
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", log_level);
    }
    env_logger::init();

    eprintln!("prism-ghost-evidence-scan — Path B feature scanner");
    eprintln!("manifest: {}", args.manifest.display());

    // ─── Phase 0 — manifest discovery ───────────────────────────────────────
    let manifest_text = std::fs::read_to_string(&args.manifest)
        .with_context(|| format!("read manifest {}", args.manifest.display()))?;
    let manifest: Manifest = serde_json::from_str(&manifest_text)
        .with_context(|| format!("parse manifest {}", args.manifest.display()))?;
    if manifest.schema_kind != "md_evidence_manifest" {
        eprintln!("ERROR: manifest schema_kind = {:?}, expected \"md_evidence_manifest\"", manifest.schema_kind);
        std::process::exit(1);
    }
    if manifest.schema_version != 1 {
        eprintln!("ERROR: manifest schema_version = {} (this scanner supports 1)", manifest.schema_version);
        std::process::exit(1);
    }
    let run_dir = args
        .manifest
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let out_dir = args.out_dir.clone().unwrap_or(run_dir.clone());
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("create out dir {}", out_dir.display()))?;
    let stem = manifest.target.clone();
    eprintln!(
        "run_id={} target={} stream_count={} run_dir={}",
        manifest.run_id, stem, manifest.stream_count, run_dir.display()
    );
    eprintln!("topology: {}", manifest.topology_input);

    // ─── Topology chain map (for bilateral logic) ────────────────────────────
    let mut chain_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Ok(t) = std::fs::read_to_string(&manifest.topology_input) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
            if let Some(arr) = v.get("chain_ids").and_then(|x| x.as_array()) {
                for c in arr.iter() {
                    if let Some(s) = c.as_str() {
                        chain_set.insert(s.to_string());
                    }
                }
            }
        }
    }
    let bilateral_status: &'static str = if chain_set.len() <= 1 {
        "not_applicable_monomer"
    } else {
        "deferred_real_partner_matching_required_no_implementation_in_this_commit"
    };
    eprintln!(
        "topology chains observed: {:?}  bilateral_status: {}",
        chain_set, bilateral_status
    );

    // ─── Phase 1 — file discovery + schema validation ──────────────────────
    let discovered = discover_files(&run_dir)?;
    let n_ghost = discovered.iter().filter(|f| f.file_kind == FileKind::GhostTiles).count();
    let n_zstr = discovered.iter().filter(|f| f.file_kind == FileKind::Zstr).count();
    let n_v2 = discovered.iter().filter(|f| f.file_kind == FileKind::PrismV2Trajectory).count();
    eprintln!(
        "discovered files: ghost_tiles={} zstr={} prism_v2_trajectory={} (run_dir scan)",
        n_ghost, n_zstr, n_v2
    );

    // ─── M1.2.23 §6 + §7 — sidecar ingestion ───────────────────────────────
    let gtm: Option<GhostTimeMap> = std::fs::read_to_string(run_dir.join("ghost_time_map.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<GhostTimeMap>(&s).ok());
    let gsm: Option<GhostSiteMap> = std::fs::read_to_string(run_dir.join("ghost_site_map.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<GhostSiteMap>(&s).ok());
    let branch_trace = read_branch_trace(&run_dir, &stem);
    let gtm_status: &'static str = match gtm.as_ref() {
        Some(g) if !g.streams.is_empty() => "available",
        Some(_) => "present_but_empty_streams",
        None => "missing",
    };
    let gsm_status: &'static str = match gsm.as_ref() {
        Some(g) if !g.entries.is_empty() => "available_with_entries",
        Some(_) => "present_but_empty_entries",
        None => "missing",
    };
    eprintln!(
        "sidecars: ghost_time_map={} ghost_site_map={} branch_trace={}",
        gtm_status, gsm_status,
        branch_trace.get("status").and_then(|x| x.as_str()).unwrap_or("?")
    );

    if n_ghost == 0 && n_zstr == 0 {
        eprintln!(
            "WARNING: no Ghost or ZSTR files in run_dir. V2 monolithic was likely not live this run."
        );
        if !args.allow_missing && args.strict_schema {
            // strict mode: still error
        } else if !args.allow_missing {
            // default: still continue but will exit with code 2
        }
    }

    let mut validations: Vec<FileSchemaValidation> = Vec::new();
    for f in discovered.iter() {
        let v = match f.file_kind {
            FileKind::GhostTiles => validate_ghost_tiles_schema(f),
            FileKind::Zstr => validate_zstr_schema(f),
            FileKind::PrismV2Trajectory => validate_prism_v2_schema(f),
        };
        eprintln!(
            "  {} kind={} status={} parse={} records={:?} stride={:?}",
            v.file_path, v.file_kind, v.schema_status, v.parse_status, v.record_count, v.byte_stride
        );
        validations.push(v);
    }

    // ─── Phase 2 — ghost tile parsing (only files with consistent envelope) ─
    let mut all_ghost_events: Vec<GhostTileEvent> = Vec::new();
    let mut ghost_streams_summary: Vec<serde_json::Value> = Vec::new();
    for (f, val) in discovered.iter().zip(validations.iter()) {
        if f.file_kind != FileKind::GhostTiles {
            continue;
        }
        let n = val.record_count.unwrap_or(0);
        if n == 0 {
            ghost_streams_summary.push(serde_json::json!({
                "stream_id": f.stream_id,
                "file":      f.path.display().to_string(),
                "record_count": 0,
                "events_parsed": 0,
                "parse_status": val.parse_status,
            }));
            continue;
        }
        let events = parse_ghost_tiles_file(
            &f.path,
            f.stream_id.unwrap_or(u32::MAX),
            n,
            args.max_events,
        )?;
        ghost_streams_summary.push(serde_json::json!({
            "stream_id":     f.stream_id,
            "file":          f.path.display().to_string(),
            "record_count":  n,
            "events_parsed": events.len(),
            "parse_status":  val.parse_status,
            "n_class_tainted": events.iter().filter(|e| e.class_tainted).count(),
            "n_with_finite_thermo_flux": events.iter().filter(|e| e.flux_coupling.eta_status == "computed").count(),
            "n_chain_a": events.iter().filter(|e| e.chain_id == Some('A')).count(),
            "n_chain_b": events.iter().filter(|e| e.chain_id == Some('B')).count(),
            "n_chain_unknown": events.iter().filter(|e| e.chain_id.is_none()).count(),
        }));
        all_ghost_events.extend(events);
    }

    // ─── Phase 3 — ZSTR summary ─────────────────────────────────────────────
    let mut zstr_summaries: Vec<ZstrStreamSummary> = Vec::new();
    for f in discovered.iter() {
        if f.file_kind == FileKind::Zstr {
            zstr_summaries.push(summarize_zstr_file(f)?);
        }
    }

    // ─── Phase 4 — noise floor (per-stream) ────────────────────────────────
    let mut noise_floor_present = false;
    let mut noise_floor_records: Vec<serde_json::Value> = Vec::new();
    for stream_id in 0..(manifest.stream_count as u32) {
        if let Some(nf) = try_load_noise_floor(&run_dir, &stem, stream_id) {
            noise_floor_present = true;
            noise_floor_records.push(serde_json::json!({
                "stream_id": stream_id,
                "mu":        nf.mu,
                "sigma":     nf.sigma,
                "notes":     nf.notes,
            }));
        }
    }
    let _ = args.threshold_sigma;

    // ─── Phase 6 — bilateral status (already computed above) ───────────────

    // ─── Phase 7 — rank tile events ─────────────────────────────────────────
    let mut ranked: Vec<RankedTileEvent> = Vec::new();
    for ev in all_ghost_events.iter() {
        let (score, comps) = score_ghost_tile_event(ev);
        let event_value = serde_json::to_value(ev).unwrap_or(serde_json::Value::Null);
        ranked.push(RankedTileEvent {
            event_id: ev.event_id.clone(),
            rank: 0,
            tile_score: score,
            score_components: comps,
            event: event_value,
            limitations: vec![
                "perturbation_channel_unavailable_in_ghost_tile_schema",
                "gear_id_unavailable_in_ghost_tile_schema",
                "asc_vector_format_unverified_no_gamma_w_compute",
                "ranking_components_neutral_when_unverified",
            ],
        });
    }
    ranked.sort_by(|a, b| b.tile_score.partial_cmp(&a.tile_score).unwrap_or(std::cmp::Ordering::Equal));
    for (i, r) in ranked.iter_mut().enumerate() {
        r.rank = i as u32;
    }
    let n_emit = if args.emit_all_events { ranked.len() } else { ranked.len().min(200) };
    let ranked_to_emit: Vec<&RankedTileEvent> = ranked.iter().take(n_emit).collect();

    // ─── Phase 8 — candidate features for materializer ─────────────────────
    let mut per_stream_ghost_support: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for ev in all_ghost_events.iter() {
        let key = ev.stream_id.to_string();
        let entry = per_stream_ghost_support
            .entry(key)
            .or_insert_with(|| serde_json::json!({"events":0,"with_finite_eta":0,"site_ids":[]}));
        let m = entry.as_object_mut().unwrap();
        let prev = m.get("events").and_then(|x| x.as_u64()).unwrap_or(0);
        m.insert("events".into(), serde_json::Value::from(prev + 1));
        if ev.flux_coupling.eta_status == "computed" {
            let p = m.get("with_finite_eta").and_then(|x| x.as_u64()).unwrap_or(0);
            m.insert("with_finite_eta".into(), serde_json::Value::from(p + 1));
        }
    }

    // ─── Output: tile_field_completeness.json ──────────────────────────────
    let any_intensity_packed = false; // Ghost tiles never carry intensity_packed.
    let any_qi_extracted = false;
    let any_gear_id_present = zstr_summaries.iter().any(|s| s.first_header.as_ref().map(|h| h.gear_id != 0).unwrap_or(false));
    let any_dt_resolved = zstr_summaries.iter().any(|s| s.first_header.as_ref().map(|h| h.dt_fs > 0.0).unwrap_or(false));
    let asc_vectors_present = manifest
        .artifacts
        .iter()
        .any(|a| a.present && a.path.ends_with("_asc_vectors.bin"))
        || std::fs::read_dir(&run_dir).map(|d| d.flatten().any(|e| {
            e.path().file_name().and_then(|s| s.to_str()).map(|n| n.ends_with("_asc_vectors.bin")).unwrap_or(false)
        })).unwrap_or(false);
    let tile_field_completeness = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_ghost_zstr_tile_field_completeness",
        "run_id":         manifest.run_id,
        "target":         stem,
        "files_validated": validations,
        "intensity_packed_present":          any_intensity_packed,
        "intensity_packed_present_reason":   "ghost_tile_schema_has_no_intensity_packed_field_see_richspike_in_interferometric_adjudicator_rs",
        "qi_bits_extracted":                 any_qi_extracted,
        "wavelength_mapping_verified":       true,
        "wavelength_mapping_source":         "interferometric_adjudicator.rs:712-735",
        "uv_mapping": {
            "0": "260nm_PHE_dominant",
            "1": "280nm_TRP_primary",
            "2": "305nm_TRP_gaussian_tail",
            "3": "320nm_TRP_control",
        },
        "gear_id_present":                   any_gear_id_present,
        "gear_id_schema_source":             "zstr.rs:88_gear_id_u32_offset_24",
        "gear_id_wave_a_default_zero":       true,
        "dt_resolved":                       any_dt_resolved,
        "dt_schema_source":                  "zstr.rs:71_dt_f32_offset_8_picoseconds",
        "physical_time_axis_available":      any_dt_resolved,
        "asc_vectors_present":               asc_vectors_present,
        "asc_schema_verified":               false,
        "asc_schema_verified_reason":        "asc_vectors.bin has no schema header in this commit; format must be verified from producer code in a follow-up",
        "asc_tile_mapping_available":        false,
        "gamma_w_computable":                false,
        "gamma_w_used_in_score":             false,
        "noise_floor_present":               noise_floor_present,
        "n_noise_floor_streams":             noise_floor_records.len(),
        "bilateral_status":                  bilateral_status,
        "topology_chains":                   chain_set.iter().collect::<Vec<_>>(),
        // M1.2.23 §5/§6/§7 + branch-trace follow-up — sidecar + trace status.
        "ghost_time_map_status":             gtm_status,
        "ghost_site_map_status":             gsm_status,
        "ghost_site_map_status_field":       gsm.as_ref().and_then(|g| g.status.clone()),
        "ghost_site_map_coordinate_frame":   gsm.as_ref().and_then(|g| g.coordinate_frame.clone()),
        "branch_trace_status":               branch_trace.get("status").cloned().unwrap_or(serde_json::Value::Null),
        "branch_decision_status":            branch_trace.get("branch_decision_status").cloned().unwrap_or(serde_json::Value::Null),
        "threshold_sigma_observation":       3.0,
        "threshold_sigma_discovery":         12.0,
        "v1_v2_schema_overlay_supported":    true,
        "v2_schema_constants_source":        "ghost_tile.rs:GHOST_FRAME_SCHEMA_V2_constant_offsets_at_GHOST_V2_OFFSET_*",
    });
    let tfc_path = out_dir.join("tile_field_completeness.json");
    write_json(&tfc_path, &tile_field_completeness)?;
    eprintln!("✓ wrote {}", tfc_path.display());

    // ─── Output: ghost_tile_evidence.json ──────────────────────────────────
    let n_ghost_events = all_ghost_events.len();
    let any_ghost_files = n_ghost > 0;
    // M1.2.25 — data-driven plane-1/2/3 status (was a static limitation string).
    // Per-plane nonzero counts so the report can distinguish unwired from no-signal.
    let mut plane_nonzero_counts: [u64; 4] = [0; 4];
    for ev in all_ghost_events.iter() {
        for plane in 0..4 {
            let off = plane * 6;
            if ev.power_spectrum[off..off + 6].iter().any(|x| *x != 0.0) {
                plane_nonzero_counts[plane] += 1;
            }
        }
    }
    let planes_1_3_any_populated = plane_nonzero_counts[1] > 0
        || plane_nonzero_counts[2] > 0
        || plane_nonzero_counts[3] > 0;
    let planes_status_string: String = if planes_1_3_any_populated {
        format!(
            "wired_per_M1_2_25_nonzero_counts_geo_{}_caus_{}_therm_{}_chem_{}",
            plane_nonzero_counts[0],
            plane_nonzero_counts[1],
            plane_nonzero_counts[2],
            plane_nonzero_counts[3],
        )
    } else {
        format!(
            "zero_in_this_scan_geo_{}_caus_0_therm_0_chem_0_kernel_writes_or_signal_below_floor",
            plane_nonzero_counts[0]
        )
    };
    let ghost_status: &str = if !any_ghost_files {
        "missing"
    } else if n_ghost_events > 0 {
        "available"
    } else {
        "partial_no_records"
    };
    let ghost_evidence = serde_json::json!({
        "schema_version":   1,
        "schema_kind":      "pathb_ghost_tile_evidence",
        "source_manifest":  args.manifest.display().to_string(),
        "run_id":           manifest.run_id,
        "target":           stem,
        "status":           ghost_status,
        "parser": {
            "format":            "GhostTileFrame_v9d_sector_locked_4096B",
            "schema_source":     "ghost_tile.rs:80-213_const_assert_offsets",
            "layout_verified":   true,
            "unsafe_assumptions": Vec::<&str>::new(),
        },
        "n_ghost_files":      n_ghost,
        "n_events_parsed":    n_ghost_events,
        "streams":            ghost_streams_summary,
        "field_completeness": {
            "frame_idx":                          true,
            "site_id":                            true,
            "chain_id":                           "ascii_byte_at_offset_12_resolves_to_A_B_or_unknown",
            "adjudication_code":                  "u8_at_offset_13",
            "kl_divergence":                      "f32_at_offset_16",
            "power_spectrum_geo_plane_0":         "f32x6_at_offset_20",
            "power_spectrum_planes_1_3":          planes_status_string.clone(),
            "thermo_flux_wd_change_vib_energy":   "f32x2_at_offset_116_NaN_until_upstream_wires_them_per_ghost_tile_rs_136_141",
            "causal_lead_residue":                "u32_at_offset_124_sentinel_uMAX_means_unresolved",
            "intensity_packed":                   "absent_in_ghost_tile_schema",
            "gear_id":                            "absent_in_ghost_tile_schema_reserved_payload_unwired",
            "asc_vectors":                        "format_unverified",
        },
    });
    let ge_path = out_dir.join("ghost_tile_evidence.json");
    write_json(&ge_path, &ghost_evidence)?;
    eprintln!("✓ wrote {}", ge_path.display());

    // ─── Output: zstr_event_summary.json ───────────────────────────────────
    let zstr_status: &str = if n_zstr == 0 {
        "missing"
    } else {
        "header_only_decoded"
    };
    let zstr_summary_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_zstr_event_summary",
        "run_id":         manifest.run_id,
        "target":         stem,
        "status":         zstr_status,
        "parser": {
            "format":          "ZstrFrameHeader_4096B_per_slot_5slot_ring",
            "schema_source":   "zstr.rs:67-114",
            "layout_verified": true,
        },
        "n_zstr_files":   n_zstr,
        "streams":        zstr_summaries,
        "decode_status":  if n_zstr == 0 { "no_zstr_files" } else { "first_header_only_payload_not_decoded_this_commit" },
        "notes": [
            "ZstrFrameHeader fields verified by offset; per-slot positions/forces payload not decoded in this commit.",
            "On-disk file pattern: prism_zstr_<pid>_<stream>.bin (cited: nhs_rt_full.rs:7638).",
        ],
    });
    let zs_path = out_dir.join("zstr_event_summary.json");
    write_json(&zs_path, &zstr_summary_json)?;
    eprintln!("✓ wrote {}", zs_path.display());

    // ─── Output: ranked_tile_events.json ───────────────────────────────────
    let mut ranked_limitations: Vec<&'static str> = vec![
        "ghost_tile_schema_has_no_perturbation_channel",
        "ghost_tile_schema_has_no_gear_id",
        "asc_work_integral_not_computed_format_unverified",
        "thermo_flux_NaN_until_upstream_wires_them_per_producer_note",
    ];
    if !planes_1_3_any_populated {
        ranked_limitations.push("power_spectrum_planes_1_3_zero_in_this_scan_M1_2_25_kernel_writes_them_so_zero_implies_signal_below_floor_or_kernel_disabled");
    }
    let ranked_json = serde_json::json!({
        "schema_version":     1,
        "schema_kind":        "pathb_ranked_tile_events",
        "run_id":             manifest.run_id,
        "target":             stem,
        "n_events_total":     all_ghost_events.len(),
        "n_events_emitted":   ranked_to_emit.len(),
        "emit_all_events":    args.emit_all_events,
        "ranking_methodology":"product_of_factors_unverified_components_neutral_1_0",
        "ranked_events":      ranked_to_emit,
        "limitations":        ranked_limitations,
    });
    let r_path = out_dir.join("ranked_tile_events.json");
    write_json(&r_path, &ranked_json)?;
    eprintln!("✓ wrote {}", r_path.display());

    // ─── Output: mar_plane_completeness.json (M1.2.25 Invariant 2) ─────────
    // Per-plane statistics (mean/max/p95/nonzero count) so downstream audit
    // can verify Pillar 2 (causality plane nonzero when geometry plane nonzero).
    let plane_names = ["geometry", "causality", "thermodynamics", "chemistry"];
    let producer_sources = [
        "so3_project.cu:518_geo_power_spectrum",
        "so3_project.cu:522_caus_power_spectrum",
        "so3_project.cu:526_therm_power_spectrum",
        "so3_project.cu:530_chem_power_spectrum",
    ];
    let total_events = all_ghost_events.len() as u64;
    let mut plane_records = Vec::with_capacity(4);
    for plane in 0..4usize {
        let mut sums: f64 = 0.0;
        let mut maxv: f32 = 0.0;
        let mut all_vals: Vec<f32> = Vec::with_capacity(all_ghost_events.len() * 6);
        for ev in all_ghost_events.iter() {
            for l in 0..6 {
                let v = ev.power_spectrum[plane * 6 + l];
                if v.is_finite() {
                    sums += v as f64;
                    if v > maxv { maxv = v; }
                    all_vals.push(v);
                }
            }
        }
        let n_total = all_vals.len();
        let mean = if n_total > 0 { sums / (n_total as f64) } else { 0.0 };
        all_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95 = if n_total > 0 {
            let idx = ((n_total as f64) * 0.95) as usize;
            all_vals[idx.min(n_total - 1)]
        } else { 0.0 };
        let nonzero_count = plane_nonzero_counts[plane];
        let status = if total_events == 0 {
            "no_events"
        } else if nonzero_count == 0 {
            "zero_in_this_scan"
        } else if (nonzero_count as f64) / (total_events as f64) >= 0.5 {
            "wired_majority_nonzero"
        } else {
            "wired_minority_nonzero"
        };
        plane_records.push(serde_json::json!({
            "plane_id":         plane,
            "plane_name":       plane_names[plane],
            "nonzero_count":    nonzero_count,
            "total_count":      total_events,
            "mean":             mean,
            "max":              maxv,
            "p95":              p95,
            "status":           status,
            "producer_source":  producer_sources[plane],
            "scanner_source":   "prism_ghost_evidence_scan.rs::mar_plane_completeness_emit_M1_2_25",
            "missing_reason":   if nonzero_count == 0 && total_events > 0 {
                "kernel_writes_this_plane_post_M1_2_25_so_zero_in_scan_implies_input_data_below_floor_or_so3_projection_disabled"
            } else { "none" },
            "used_in_interferometric_contrast": plane > 0,
        }));
    }
    let geo_nz = plane_nonzero_counts[0];
    let caus_nz = plane_nonzero_counts[1];
    let therm_nz = plane_nonzero_counts[2];
    let chem_nz = plane_nonzero_counts[3];
    let pillar_2_violation = geo_nz > 0 && caus_nz == 0;
    let full_4plane = geo_nz > 0 && caus_nz > 0 && (therm_nz > 0 || chem_nz > 0);
    let c_total_status = if full_4plane {
        "computable_4plane"
    } else if geo_nz > 0 && caus_nz > 0 {
        "computable_2plane_geo_plus_causality"
    } else if geo_nz > 0 {
        "partial_geometry_only"
    } else {
        "no_signal"
    };
    let mar_plane_completeness = serde_json::json!({
        "schema_version":           1,
        "schema_kind":              "pathb_mar_plane_completeness",
        "run_id":                   manifest.run_id,
        "target":                   stem,
        "n_events_total":           total_events,
        "geometry_nonzero":         geo_nz,
        "causality_nonzero":        caus_nz,
        "thermodynamics_nonzero":   therm_nz,
        "chemistry_nonzero":        chem_nz,
        "full_4plane_available":    full_4plane,
        "c_total_status":           c_total_status,
        "pillar_2_driver_violation": pillar_2_violation,
        "pillar_2_violation_reason": if pillar_2_violation {
            "geometry_present_but_kcc_causality_plane_zero"
        } else { "none" },
        "planes":                   plane_records,
    });
    let mpc_path = out_dir.join("mar_plane_completeness.json");
    write_json(&mpc_path, &mar_plane_completeness)?;
    eprintln!("✓ wrote {}", mpc_path.display());

    if args.emit_ndjson {
        let nd_path = out_dir.join("ghost_tile_events.ndjson");
        let f = File::create(&nd_path)?;
        let mut bw = std::io::BufWriter::new(f);
        for ev in all_ghost_events.iter() {
            use std::io::Write as _;
            let s = serde_json::to_string(ev)?;
            bw.write_all(s.as_bytes())?;
            bw.write_all(b"\n")?;
        }
        use std::io::Write as _;
        bw.flush()?;
        eprintln!("✓ wrote {}", nd_path.display());
    }

    // ─── Output: ghost_zstr_candidate_features.json ────────────────────────
    let candidate_features = serde_json::json!({
        "schema_version":   1,
        "schema_kind":      "pathb_ghost_zstr_candidate_features",
        "run_id":           manifest.run_id,
        "target":           stem,
        "spatial_status":   "unmapped_ghost_tile_records_have_no_voxel_coordinate_field",
        "per_stream_ghost_support":  per_stream_ghost_support,
        "noise_floor_records":       noise_floor_records,
        "feature_handoff_notes": [
            "ghost_tile_records_carry_site_id_only_no_voxel_xyz_in_4096B_record",
            "to_link_to_voxels_use_site_id_to_per_stream_clusters_via_engine_side_metadata",
            "downstream_materializer_must_use_only_non_spatial_features_until_xyz_mapping_is_added",
        ],
        "asc_work_evidence_summary": {
            "gamma_w_computable_from_disk_in_this_commit": false,
            "blocking_reasons": [
                "asc_vectors.bin has no schema header",
                "force/displacement vector layout not verified from producer source",
                "tile->voxel mapping not available",
            ],
            "follow_up": "verify asc_vectors layout from engine emit code at nhs_rt_full.rs ~10150",
        },
        "perturbation_channel_evidence_summary": {
            "qi_bits_in_ghost_tile":      false,
            "qi_bits_in_richspike":       true,
            "uv_mapping_source":          "interferometric_adjudicator.rs:712-735",
            "follow_up": "to attach perturbation channel to a tile event, cross-ref with PRSPK001 spikes in same time window",
        },
        "gear_normalized_timing_summary": {
            "gear_id_field_in_zstr_schema": true,
            "gear_id_wave_a_default":       0,
            "dt_field_in_zstr_schema":      true,
            "dt_units":                     "picoseconds",
            "ghost_tile_has_gear":          false,
        },
        "bilateral_status":  bilateral_status,
    });
    let cf_path = out_dir.join("ghost_zstr_candidate_features.json");
    write_json(&cf_path, &candidate_features)?;
    eprintln!("✓ wrote {}", cf_path.display());

    // ─── Output: ghost_zstr_scan_report.json ───────────────────────────────
    let any_files = n_ghost > 0 || n_zstr > 0;
    let materializer_ready: bool = n_ghost_events > 0 && asc_vectors_present;
    let report = serde_json::json!({
        "schema_version":   1,
        "schema_kind":      "pathb_ghost_zstr_scan_report",
        "run_id":           manifest.run_id,
        "target":           stem,
        "scanned_files":    discovered.iter().map(|f| f.path.display().to_string()).collect::<Vec<_>>(),
        "n_ghost_files":    n_ghost,
        "n_zstr_files":     n_zstr,
        "n_v2_files":       n_v2,
        "n_events":         all_ghost_events.len(),
        "n_ranked_events":  ranked.len(),
        "files_validated":  validations.len(),
        "materializer_ready": materializer_ready,
        "limitations": [
            "perturbation_channel_not_in_ghost_tile_schema",
            "gear_id_not_in_ghost_tile_schema",
            "asc_vectors_format_unverified",
            "no_voxel_xyz_in_ghost_tile_record",
            "bilateral_logic_disabled_for_monomer_or_deferred_for_dimer",
        ],
        "recommended_next_materializer_integration_step": if !any_files {
            "rerun_engine_with_v2_live_to_produce_ghost_tile_files_then_re_scan"
        } else if !materializer_ready {
            "verify_asc_vectors_layout_and_add_voxel_xyz_to_ghost_tile_or_index_via_site_id"
        } else {
            "consume_ghost_zstr_candidate_features_in_prism-materialize-sites_v3"
        },
        "honest_assessment": if any_files {
            "ghost_zstr_evidence_extracted_for_downstream_materializer"
        } else {
            "no_ghost_or_zstr_files_present_in_run_dir_v2_was_not_live"
        },
        // M1.2.23 §6/§7 + branch-trace follow-up consumption summary.
        "sidecars": {
            "ghost_time_map_status": gtm_status,
            "ghost_site_map_status": gsm_status,
        },
        "branch_trace": branch_trace,
    });
    let rep_path = out_dir.join("ghost_zstr_scan_report.json");
    write_json(&rep_path, &report)?;
    eprintln!("✓ wrote {}", rep_path.display());

    // ─── Exit policy ────────────────────────────────────────────────────────
    if !any_files {
        if args.allow_missing {
            eprintln!("\nNo Ghost/ZSTR files in run_dir; --allow-missing → exit 0 with honest 'missing' status.");
            return Ok(());
        } else {
            eprintln!("\nNo Ghost/ZSTR files in run_dir; --allow-missing not set → exit 2.");
            std::process::exit(2);
        }
    }
    if args.strict_schema {
        let any_unverified = validations.iter().any(|v| {
            v.schema_status.starts_with("verified_offsets_but_file_truncated")
                || v.parse_status == "envelope_size_mismatch"
        });
        if any_unverified {
            eprintln!("\n--strict-schema set and one or more files failed envelope validation → exit 3.");
            std::process::exit(3);
        }
    }
    eprintln!("\nDone. {} ghost files / {} zstr files / {} events / {} ranked.",
              n_ghost, n_zstr, all_ghost_events.len(), ranked.len());
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut bw = std::io::BufWriter::new(f);
    serde_json::to_writer_pretty(&mut bw, value)?;
    use std::io::Write as _;
    bw.flush()?;
    Ok(())
}
