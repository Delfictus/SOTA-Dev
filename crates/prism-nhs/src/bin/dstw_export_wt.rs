//! PRISM-4D → DSTW WT physics exporter (Air-Gap bridge, operator-authorised
//! 2026-05-20).
//!
//! Reads the engine's wild-type prime-run outputs (`<base>.binding_sites.json`
//! containing the `prism_therm` payload with per-site TIDE decomposition +
//! 5-phase hysteresis; optional `<base>.trajectory.arrow` for per-spike water
//! density traces) and emits the canonical `WTPhysicalProfileResponse` JSON
//! plus a matching Parquet that DSTW's `GMMPhysicalStratifier` consumes.
//!
//! Per-residue channel mapping (engine output → DSTW handshake):
//!
//!   * te_out             ←  max(TideResidueResult.transfer_entropy across
//!                           every site this residue appears in).  Causal
//!                           outflow score.
//!   * te_in              ←  max(TideResidueResult.fisher_info across every
//!                           site this residue appears in).  Sensitivity to
//!                           perturbation = transfer-entropy DUAL = inflow.
//!   * delta_hc           ←  max site asymmetry_score across every site this
//!                           residue's TIDE row appears in.  Thermal
//!                           hysteresis area projected onto the residue via
//!                           TIDE membership.
//!   * sigma_hydration_sq ←  variance of nearby-spike water_density across
//!                           the 5-phase trajectory window when the spike
//!                           arrow file is present; 0.0 (with a recorded
//!                           `hydration_status` of "not_observed") when the
//!                           trajectory archive is unavailable.  The DSTW
//!                           schema floor (≥0) is honoured either way.
//!
//! Residues that NEVER appear in any binding-site TIDE row (out-of-pocket
//! residues) receive te_out=te_in=delta_hc=0.0 — the engine's measured
//! "this residue has no detected causal influence in any of the discovered
//! pockets" reading.  DSTW's downstream GMM stratifier handles those rows
//! as a valid (low-everything) cluster (Silent Core).
//!
//! NO ENGINE EXECUTION HAPPENS HERE.  This binary is a strict post-processor.
//! It does not launch `nhs_rt_full`, does not run CUDA kernels, and does not
//! re-compute physics.

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::{
    builder::{Float64Builder, Int32Builder, StringBuilder},
    ArrayRef, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "dstw_export_wt",
    about = "Export PRISM-4D WT prime-run results into the DSTW WTPhysicalProfileResponse schema.",
)]
struct Args {
    /// Path to <base>.binding_sites.json from the engine's WT prime run.
    #[arg(long)]
    binding_sites: PathBuf,

    /// UniProt accession for the target receptor (e.g. P61073 for CXCR4).
    #[arg(long)]
    uniprot_accession: String,

    /// Free-text target name (e.g. "CXCR4").
    #[arg(long)]
    target: String,

    /// Optional PDB anchor used for the inactive-state reference structure
    /// (e.g. "3OE0_chainA_t4l_stripped").
    #[arg(long)]
    pdb_anchor_inactive: Option<String>,

    /// Optional PDB anchor used for the active-state reference structure
    /// (e.g. "7SK3_chainR").
    #[arg(long)]
    pdb_anchor_active: Option<String>,

    /// PRISM-4D run identifier emitted by `nhs_rt_full` (used for provenance).
    #[arg(long)]
    prism_run_id: String,

    /// Where to write the WTPhysicalProfileResponse JSON.
    #[arg(long)]
    out_json: PathBuf,

    /// Where to write the matching per-residue Parquet (DSTW's stratifier
    /// consumes this directly).
    #[arg(long)]
    out_parquet: PathBuf,

    /// Optional spike Arrow file produced by the engine.  Needed for the
    /// sigma_hydration_sq channel; when absent, that channel is set to 0.0
    /// and `hydration_status` is recorded as `not_observed` in the manifest.
    #[arg(long)]
    spike_arrow: Option<PathBuf>,

    /// Lowest valid UniProt residue index in the receptor (defaults to 1;
    /// override when exporting a chain offset, e.g. CXCR4 starts at 1).
    #[arg(long, default_value_t = 1)]
    residue_index_lo: i32,

    /// Highest valid UniProt residue index.  Required so the exporter can
    /// emit zero-physics rows for out-of-pocket residues instead of dropping
    /// them silently.
    #[arg(long)]
    residue_index_hi: i32,
}

// ---------------------------------------------------------------------------
// Engine-side input schema (subset of `binding_sites.json`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct BindingSitesEnvelope {
    /// The full document keeps many other fields; we only need `prism_therm`.
    prism_therm: Option<PrismThermPayload>,
}

#[derive(Debug, Clone, Deserialize)]
struct PrismThermPayload {
    sites: Vec<PrismThermSite>,
    /// Some emit paths nest sites under a different key; tolerate both.
    #[serde(default)]
    site_results: Vec<PrismThermSite>,
}

// Engine emit schema is mirrored here in full so deserialization tolerates
// every field the engine writes today.  We only consume a subset; the rest
// are kept under #[allow(dead_code)] so they survive serde and stay
// available when downstream channels are wired in.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct PrismThermSite {
    #[serde(default)]
    site_id: i32,
    #[serde(default)]
    asymmetry_score: f32,
    #[serde(default)]
    is_hysteretic: bool,
    #[serde(default)]
    relative_asymmetry: f32,
    #[serde(default)]
    therm_class: Option<String>,
    #[serde(default)]
    tide_decomposition: Vec<TideResidue>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct TideResidue {
    residue_id: u32,
    #[serde(default)]
    causal_dg: f32,
    #[serde(default)]
    transfer_entropy: f32,
    #[serde(default)]
    fisher_info: f32,
    #[serde(default)]
    kl_divergence: f32,
    #[serde(default)]
    n_causal_spikes: u32,
}

// ---------------------------------------------------------------------------
// DSTW-side output schema (mirrors `prism_dstw.orchestration.prism_handshake`)
// ---------------------------------------------------------------------------

const PHYSICS_SCHEMA_TAG: &str = "dstw_wt_physical_profile_v1";

#[derive(Debug, Clone, Serialize)]
struct WTResiduePhysics {
    uniprot_residue_index: i32,
    /// 3-letter or 1-letter code; the engine often does not know the
    /// residue name in this binary so we emit a deterministic placeholder
    /// "UNK".  DSTW's stratifier does not depend on residue names for the
    /// GMM fit; the residue_name field is for operator inspection only.
    residue_name: String,
    te_out: f64,
    te_in: f64,
    delta_hc: f64,
    sigma_hydration_sq: f64,
}

/// Dynamic contact edge between two residues.  Mirrors DSTW's
/// `WTDynamicContactEdge`.  Self-edges are forbidden by DSTW; we drop them
/// in the assembler before serialisation.
#[derive(Debug, Clone, Serialize)]
struct WTDynamicContactEdge {
    source_uniprot_residue_index: i32,
    target_uniprot_residue_index: i32,
    contact_probability: f64,
    mean_distance_angstrom: Option<f64>,
    dynamic_correlation: Option<f64>,
    replicate_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct WTPhysicalProfileResponse {
    target: String,
    uniprot_accession: String,
    pdb_anchor_inactive: Option<String>,
    pdb_anchor_active: Option<String>,
    prism_run_id: String,
    completed_at_utc: String,
    residues: Vec<WTResiduePhysics>,
    dynamic_contact_edges: Vec<WTDynamicContactEdge>,
    physics_schema: &'static str,
    /// Provenance side-channel (not part of the DSTW Pydantic schema --
    /// the loader ignores unknown top-level keys because `extra="forbid"`
    /// is only enforced on the residues array.  This metadata block lives
    /// in a sibling provenance file instead).
    #[serde(skip)]
    provenance: ExporterProvenance,
}

#[derive(Debug, Clone, Serialize)]
struct ExporterProvenance {
    binding_sites_path: String,
    spike_arrow_path: Option<String>,
    hydration_status: &'static str,
    residue_index_range: [i32; 2],
    sites_consumed: usize,
    tide_rows_consumed: usize,
    out_of_pocket_residue_count: usize,
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

fn load_binding_sites(path: &Path) -> Result<PrismThermPayload> {
    let file = File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let envelope: BindingSitesEnvelope = serde_json::from_reader(reader)
        .with_context(|| format!("parse {}", path.display()))?;
    envelope.prism_therm.ok_or_else(|| {
        anyhow!(
            "{}: file is missing the `prism_therm` payload; was this run \
             produced with `--prism-therm`?",
            path.display()
        )
    })
}

fn iter_sites(payload: &PrismThermPayload) -> impl Iterator<Item = &PrismThermSite> {
    payload.sites.iter().chain(payload.site_results.iter())
}

// ---------------------------------------------------------------------------
// Aggregator
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Copy)]
struct ResidueAccumulator {
    te_out: f64,
    te_in: f64,
    delta_hc: f64,
    sigma_hydration_sq: f64,
    n_site_appearances: u32,
}

impl ResidueAccumulator {
    fn fold(&mut self, te: f32, fi: f32, site_asym: f32) {
        let te = te as f64;
        let fi = fi as f64;
        let asym = site_asym as f64;
        if te.is_finite() && te > self.te_out {
            self.te_out = te;
        }
        if fi.is_finite() && fi > self.te_in {
            self.te_in = fi;
        }
        if asym.is_finite() && asym > self.delta_hc {
            self.delta_hc = asym;
        }
        self.n_site_appearances = self.n_site_appearances.saturating_add(1);
    }
}

fn aggregate_per_residue(
    payload: &PrismThermPayload,
    residue_index_lo: i32,
    residue_index_hi: i32,
) -> (BTreeMap<i32, ResidueAccumulator>, usize, usize) {
    let mut per_residue: BTreeMap<i32, ResidueAccumulator> = BTreeMap::new();
    // Pre-seed every in-range residue with a zero accumulator so the output
    // contains the full receptor (out-of-pocket residues land as zeros, not
    // silent drops).
    for idx in residue_index_lo..=residue_index_hi {
        per_residue.insert(idx, ResidueAccumulator::default());
    }
    let mut sites_consumed = 0usize;
    let mut tide_rows_consumed = 0usize;
    for site in iter_sites(payload) {
        sites_consumed += 1;
        for row in &site.tide_decomposition {
            tide_rows_consumed += 1;
            let residue_id = row.residue_id as i32;
            if residue_id < residue_index_lo || residue_id > residue_index_hi {
                continue;
            }
            let acc = per_residue.entry(residue_id).or_default();
            acc.fold(row.transfer_entropy, row.fisher_info, site.asymmetry_score);
        }
    }
    (per_residue, sites_consumed, tide_rows_consumed)
}

// ---------------------------------------------------------------------------
// Hydration variance pass (sketched -- requires spike Arrow ingestion)
// ---------------------------------------------------------------------------

/// Status string for the manifest's `hydration_status` field.
/// `not_observed`: no spike Arrow file was provided.
/// `derived_from_spike_arrow`: the per-spike water_density column was read
/// and per-residue variance was computed.
fn populate_hydration_variance(
    per_residue: &mut BTreeMap<i32, ResidueAccumulator>,
    spike_arrow_path: Option<&Path>,
) -> Result<&'static str> {
    // The full implementation reads the spike Arrow file's water_density and
    // nearby_residues columns and computes per-residue variance over the
    // 5-phase trajectory window.  That path is gated behind the file being
    // present.  When the file is absent we return zeros (DSTW's >=0 floor
    // is honoured) and tag the manifest accordingly.
    if let Some(path) = spike_arrow_path {
        let _ = path; // silence the unused-variable warning until the full
                      // Arrow-IPC reader lands.  The skeleton below documents
                      // the intended aggregation; do NOT wire it half-implemented
                      // because partial reads would produce silently biased
                      // variances and DSTW would treat them as ground truth.
        bail!(
            "spike Arrow ingestion is not yet wired up; either re-run \
             without --spike-arrow (variance defaults to 0.0 and \
             hydration_status records `not_observed`), or fill in the \
             arrow-ipc reader path documented in dstw_export_wt.rs"
        );
    }
    for acc in per_residue.values_mut() {
        acc.sigma_hydration_sq = 0.0;
    }
    Ok("not_observed")
}

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

fn iso_8601_utc_now() -> String {
    // Minimal ISO-8601 without an external chrono dep: seconds-precision UTC.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    // Compute YYYY-MM-DDTHH:MM:SSZ from secs.  Avoid chrono because prism-nhs
    // already keeps its dep graph tight; this is a single-use formatter.
    let (year, month, day, hour, minute, second) = epoch_to_civil(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn epoch_to_civil(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    // Howard Hinnant's days_from_civil inverse.  No external deps.
    let secs_in_day: u64 = 86400;
    let z = (secs / secs_in_day) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { (mp + 3) as u32 } else { (mp - 9) as u32 };
    let year = y + (if m <= 2 { 1 } else { 0 });
    let time = secs % secs_in_day;
    let hour = (time / 3600) as u32;
    let minute = ((time % 3600) / 60) as u32;
    let second = (time % 60) as u32;
    (year, m, d, hour, minute, second)
}

/// Build per-pair contact edges from TIDE co-membership.
///
/// Two residues appearing in the same binding site's TIDE list are in
/// dynamic contact; the contact probability is set to the fraction of
/// observed sites where they co-appear out of the maximum co-appearance
/// count across all pairs.  Self-edges are skipped (DSTW forbids them).
///
/// This is a CONSERVATIVE contact-graph derivation: it captures the
/// pocket-level connectivity the engine already reports in
/// binding_sites.json without requiring a separate dynamic-contact
/// emit path.  When the engine later writes a richer per-frame contact
/// trace, the exporter can ingest it and replace this approximation.
fn derive_contact_edges(payload: &PrismThermPayload) -> Vec<WTDynamicContactEdge> {
    let mut pair_counts: BTreeMap<(i32, i32), u32> = BTreeMap::new();
    for site in iter_sites(payload) {
        let residues: Vec<i32> = site
            .tide_decomposition
            .iter()
            .map(|r| r.residue_id as i32)
            .collect();
        for i in 0..residues.len() {
            for j in (i + 1)..residues.len() {
                let a = residues[i];
                let b = residues[j];
                if a == b {
                    continue;
                }
                let key = if a < b { (a, b) } else { (b, a) };
                *pair_counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    let max_co = pair_counts.values().copied().max().unwrap_or(1) as f64;
    pair_counts
        .into_iter()
        .map(|((a, b), n)| WTDynamicContactEdge {
            source_uniprot_residue_index: a,
            target_uniprot_residue_index: b,
            contact_probability: (n as f64 / max_co).clamp(0.0, 1.0),
            mean_distance_angstrom: None,
            dynamic_correlation: None,
            replicate_id: Some("tide_comembership_derived".to_string()),
        })
        .collect()
}

fn build_response(
    args: &Args,
    payload: &PrismThermPayload,
    per_residue: &BTreeMap<i32, ResidueAccumulator>,
    hydration_status: &'static str,
    sites_consumed: usize,
    tide_rows_consumed: usize,
) -> WTPhysicalProfileResponse {
    let mut residues: Vec<WTResiduePhysics> = Vec::with_capacity(per_residue.len());
    let mut out_of_pocket = 0usize;
    for (&idx, acc) in per_residue.iter() {
        if acc.n_site_appearances == 0 {
            out_of_pocket += 1;
        }
        residues.push(WTResiduePhysics {
            uniprot_residue_index: idx,
            residue_name: "UNK".to_string(),
            te_out: acc.te_out,
            te_in: acc.te_in,
            delta_hc: acc.delta_hc,
            sigma_hydration_sq: acc.sigma_hydration_sq,
        });
    }
    let known: std::collections::BTreeSet<i32> =
        residues.iter().map(|r| r.uniprot_residue_index).collect();
    let raw_edges = derive_contact_edges(payload);
    let dynamic_contact_edges: Vec<WTDynamicContactEdge> = raw_edges
        .into_iter()
        .filter(|e| {
            known.contains(&e.source_uniprot_residue_index)
                && known.contains(&e.target_uniprot_residue_index)
        })
        .collect();
    WTPhysicalProfileResponse {
        target: args.target.clone(),
        uniprot_accession: args.uniprot_accession.clone(),
        pdb_anchor_inactive: args.pdb_anchor_inactive.clone(),
        pdb_anchor_active: args.pdb_anchor_active.clone(),
        prism_run_id: args.prism_run_id.clone(),
        completed_at_utc: iso_8601_utc_now(),
        residues,
        dynamic_contact_edges,
        physics_schema: PHYSICS_SCHEMA_TAG,
        provenance: ExporterProvenance {
            binding_sites_path: args.binding_sites.display().to_string(),
            spike_arrow_path: args.spike_arrow.as_ref().map(|p| p.display().to_string()),
            hydration_status,
            residue_index_range: [args.residue_index_lo, args.residue_index_hi],
            sites_consumed,
            tide_rows_consumed,
            out_of_pocket_residue_count: out_of_pocket,
        },
    }
}

fn write_json(response: &WTPhysicalProfileResponse, out: &Path) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(out)
        .with_context(|| format!("create {}", out.display()))?;
    serde_json::to_writer_pretty(file, response)
        .with_context(|| format!("write {}", out.display()))?;
    // Also emit the provenance sidecar next to the manifest.
    let provenance_path = out.with_extension("provenance.json");
    let prov_file = File::create(&provenance_path)
        .with_context(|| format!("create {}", provenance_path.display()))?;
    serde_json::to_writer_pretty(prov_file, &response.provenance)
        .with_context(|| format!("write {}", provenance_path.display()))?;
    Ok(())
}

fn write_parquet(response: &WTPhysicalProfileResponse, out: &Path) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("uniprot_residue_index", DataType::Int32, false),
        Field::new("residue_name", DataType::Utf8, false),
        Field::new("te_out", DataType::Float64, false),
        Field::new("te_in", DataType::Float64, false),
        Field::new("delta_hc", DataType::Float64, false),
        Field::new("sigma_hydration_sq", DataType::Float64, false),
    ]));
    let mut idx_b = Int32Builder::with_capacity(response.residues.len());
    let mut name_b = StringBuilder::new();
    let mut teout_b = Float64Builder::with_capacity(response.residues.len());
    let mut tein_b = Float64Builder::with_capacity(response.residues.len());
    let mut dh_b = Float64Builder::with_capacity(response.residues.len());
    let mut sh_b = Float64Builder::with_capacity(response.residues.len());
    for r in &response.residues {
        idx_b.append_value(r.uniprot_residue_index);
        name_b.append_value(&r.residue_name);
        teout_b.append_value(r.te_out);
        tein_b.append_value(r.te_in);
        dh_b.append_value(r.delta_hc);
        sh_b.append_value(r.sigma_hydration_sq);
    }
    let columns: Vec<ArrayRef> = vec![
        Arc::new(idx_b.finish()),
        Arc::new(name_b.finish()),
        Arc::new(teout_b.finish()),
        Arc::new(tein_b.finish()),
        Arc::new(dh_b.finish()),
        Arc::new(sh_b.finish()),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns)
        .context("RecordBatch::try_new for WT physics")?;
    let file = File::create(out)
        .with_context(|| format!("create {}", out.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();
    if args.residue_index_hi < args.residue_index_lo {
        bail!(
            "residue_index_hi ({}) must be >= residue_index_lo ({})",
            args.residue_index_hi,
            args.residue_index_lo
        );
    }
    let payload = load_binding_sites(&args.binding_sites)?;
    let (mut per_residue, sites_consumed, tide_rows_consumed) = aggregate_per_residue(
        &payload,
        args.residue_index_lo,
        args.residue_index_hi,
    );
    let hydration_status = populate_hydration_variance(
        &mut per_residue,
        args.spike_arrow.as_deref(),
    )?;
    let response = build_response(
        &args,
        &payload,
        &per_residue,
        hydration_status,
        sites_consumed,
        tide_rows_consumed,
    );
    write_json(&response, &args.out_json)?;
    write_parquet(&response, &args.out_parquet)?;

    eprintln!(
        "[dstw_export_wt] sites={} tide_rows={} residues={} out_of_pocket={} hydration_status={}",
        sites_consumed,
        tide_rows_consumed,
        response.residues.len(),
        response.provenance.out_of_pocket_residue_count,
        hydration_status,
    );
    eprintln!("[dstw_export_wt] wrote: {}", args.out_json.display());
    eprintln!("[dstw_export_wt] wrote: {}", args.out_parquet.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_site(
        site_id: i32,
        asym: f32,
        tide: Vec<TideResidue>,
    ) -> PrismThermSite {
        PrismThermSite {
            site_id,
            asymmetry_score: asym,
            is_hysteretic: asym >= 0.2,
            relative_asymmetry: asym,
            therm_class: None,
            tide_decomposition: tide,
        }
    }

    fn make_residue(id: u32, te: f32, fi: f32) -> TideResidue {
        TideResidue {
            residue_id: id,
            causal_dg: 0.0,
            transfer_entropy: te,
            fisher_info: fi,
            kl_divergence: 0.0,
            n_causal_spikes: 1,
        }
    }

    #[test]
    fn aggregator_takes_max_across_sites() {
        let payload = PrismThermPayload {
            sites: vec![
                make_site(0, 0.30, vec![make_residue(10, 1.0, 0.2)]),
                make_site(1, 0.45, vec![make_residue(10, 0.5, 0.9)]),
            ],
            site_results: vec![],
        };
        let (per_residue, sites_consumed, tide_rows) =
            aggregate_per_residue(&payload, 1, 20);
        assert_eq!(sites_consumed, 2);
        assert_eq!(tide_rows, 2);
        let acc = per_residue.get(&10).expect("residue 10 must be present");
        // The engine emits f32; we promote to f64.  Tolerances reflect that.
        let tol: f64 = 1e-6;
        assert!((acc.te_out - 1.0).abs() < tol);
        assert!((acc.te_in - 0.9).abs() < tol);
        assert!((acc.delta_hc - 0.45).abs() < tol);
        assert_eq!(acc.n_site_appearances, 2);
    }

    #[test]
    fn out_of_pocket_residues_are_emitted_as_zeros() {
        let payload = PrismThermPayload {
            sites: vec![make_site(0, 0.30, vec![make_residue(5, 0.7, 0.5)])],
            site_results: vec![],
        };
        let (per_residue, _, _) = aggregate_per_residue(&payload, 1, 10);
        // residue 5 (in-pocket) has non-zero te_out
        assert!(per_residue[&5].te_out > 0.0);
        // residue 1 (out-of-pocket) is present with all-zero physics
        let acc = per_residue.get(&1).expect("must seed all residues");
        assert_eq!(acc.te_out, 0.0);
        assert_eq!(acc.n_site_appearances, 0);
    }

    #[test]
    fn hydration_status_records_not_observed_when_no_spike_arrow() {
        let mut per_residue: BTreeMap<i32, ResidueAccumulator> = BTreeMap::new();
        per_residue.insert(1, ResidueAccumulator::default());
        let status =
            populate_hydration_variance(&mut per_residue, None).expect("absent spike arrow ok");
        assert_eq!(status, "not_observed");
    }

    #[test]
    fn hydration_pass_refuses_partial_arrow_implementation() {
        // The full Arrow path is gated; if a caller supplies a spike_arrow
        // path while the reader is still stubbed, we MUST fail loudly so
        // that DSTW never receives a silently zero-padded variance.
        let mut per_residue: BTreeMap<i32, ResidueAccumulator> = BTreeMap::new();
        per_residue.insert(1, ResidueAccumulator::default());
        let dummy = PathBuf::from("/dev/null");
        let err = populate_hydration_variance(&mut per_residue, Some(&dummy)).unwrap_err();
        assert!(err.to_string().contains("not yet wired up"));
    }
}
