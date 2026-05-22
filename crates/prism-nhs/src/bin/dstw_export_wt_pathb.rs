//! PRISM Path-B MD-only materialization -> DSTW WT physics export.
//!
//! This exporter is deliberately separate from `dstw-export-wt`, which consumes
//! the legacy `prism_therm` payload in `<base>.binding_sites.json`. The GLP-1R
//! WT prime runs in this campaign were executed with `--md-only-evidence`, then
//! materialized offline through `prism-materialize-sites`, producing
//! `binding_sites.materialized.json`.
//!
//! The materializer reports residue identifiers in topology-index space. This
//! binary uses the PRISM-PREP residue-map sidecar to convert them back to the
//! receptor residue index before exporting to DSTW. It does not claim a richer
//! dynamic-contact trace than the engine emitted: contact edges are conservative
//! co-membership edges among materialized site lining residues.

use anyhow::{bail, Context, Result};
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
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

const PHYSICS_SCHEMA_TAG: &str = "dstw_wt_physical_profile_v1";

#[derive(Parser, Debug)]
#[command(
    name = "dstw-export-wt-pathb",
    about = "Export Path-B materialized MD-only evidence into DSTW WT physics parquets."
)]
struct Args {
    /// Path to binding_sites.materialized.json from prism-materialize-sites.
    #[arg(long)]
    materialized_sites: PathBuf,

    /// PRISM-PREP residue_map.json sidecar for converting topology indices to
    /// receptor residue indices.
    #[arg(long)]
    residue_map: PathBuf,

    /// Optional topology JSON used to compute CA-CA distances for contact
    /// graph edges.
    #[arg(long)]
    topology_json: Option<PathBuf>,

    #[arg(long)]
    target: String,

    #[arg(long)]
    uniprot_accession: String,

    #[arg(long)]
    structure_anchor_id: String,

    #[arg(long)]
    receptor_chain_id: String,

    #[arg(long)]
    prism_run_id: String,

    #[arg(long)]
    pdb_anchor_inactive: Option<String>,

    #[arg(long)]
    pdb_anchor_active: Option<String>,

    #[arg(long)]
    out_json: PathBuf,

    #[arg(long)]
    out_parquet: PathBuf,

    #[arg(long)]
    out_contact_parquet: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct ResidueMapEnvelope {
    residues: Vec<ResidueMapRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct ResidueMapRow {
    topology_index: i32,
    chain: String,
    pdb_resid: i32,
    resname: String,
}

#[derive(Debug, Clone)]
struct MappedResidue {
    residue_index: i32,
    residue_name: String,
    chain_id: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct ResidueAccumulator {
    te_out: f64,
    te_in: f64,
    delta_hc: f64,
    sigma_hydration_sq: f64,
    n_site_appearances: u32,
}

impl ResidueAccumulator {
    fn fold(&mut self, te_out: f64, te_in: f64, delta_hc: f64, sigma: f64) {
        if te_out.is_finite() && te_out > self.te_out {
            self.te_out = te_out;
        }
        if te_in.is_finite() && te_in > self.te_in {
            self.te_in = te_in;
        }
        if delta_hc.is_finite() && delta_hc > self.delta_hc {
            self.delta_hc = delta_hc;
        }
        if sigma.is_finite() && sigma > self.sigma_hydration_sq {
            self.sigma_hydration_sq = sigma;
        }
        self.n_site_appearances = self.n_site_appearances.saturating_add(1);
    }
}

#[derive(Debug, Clone, Serialize)]
struct WTResiduePhysics {
    uniprot_residue_index: i32,
    residue_name: String,
    te_out: f64,
    te_in: f64,
    delta_hc: f64,
    sigma_hydration_sq: f64,
}

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
}

fn v_f64(v: &Value, path: &[&str]) -> f64 {
    let mut cur = v;
    for key in path {
        cur = match cur.get(*key) {
            Some(next) => next,
            None => return 0.0,
        };
    }
    cur.as_f64().unwrap_or(0.0)
}

fn v_i64(v: &Value, path: &[&str]) -> Option<i64> {
    let mut cur = v;
    for key in path {
        cur = cur.get(*key)?;
    }
    cur.as_i64()
}

fn v_i32_vec(v: &Value, key: &str) -> Vec<i32> {
    v.get(key)
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_i64())
                .filter_map(|x| i32::try_from(x).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn load_residue_map(path: &Path, receptor_chain_id: &str) -> Result<BTreeMap<i32, MappedResidue>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read residue map {}", path.display()))?;
    let env: ResidueMapEnvelope = serde_json::from_str(&text)
        .with_context(|| format!("parse residue map {}", path.display()))?;
    let mut out = BTreeMap::new();
    for row in env.residues {
        if row.chain != receptor_chain_id {
            continue;
        }
        if row.pdb_resid < 1 {
            bail!(
                "{}: residue map row has non-positive receptor residue index: {:?}",
                path.display(),
                row.pdb_resid
            );
        }
        out.insert(
            row.topology_index,
            MappedResidue {
                residue_index: row.pdb_resid,
                residue_name: row.resname,
                chain_id: row.chain,
            },
        );
    }
    if out.is_empty() {
        bail!(
            "{}: residue map contains no residues for receptor chain {:?}",
            path.display(),
            receptor_chain_id
        );
    }
    Ok(out)
}

fn load_ca_positions(
    topology_path: Option<&Path>,
    residue_map: &BTreeMap<i32, MappedResidue>,
) -> Result<BTreeMap<i32, [f64; 3]>> {
    let Some(path) = topology_path else {
        return Ok(BTreeMap::new());
    };
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read topology {}", path.display()))?;
    let root: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse topology {}", path.display()))?;
    let positions = root
        .get("positions")
        .and_then(|v| v.as_array())
        .context("topology positions must be an array")?;
    let ca_indices = root
        .get("ca_indices")
        .and_then(|v| v.as_array())
        .context("topology ca_indices must be an array")?;

    let mut out = BTreeMap::new();
    for (topo_idx, atom_idx_value) in ca_indices.iter().enumerate() {
        let Some(mapped) = residue_map.get(&(topo_idx as i32)) else {
            continue;
        };
        let Some(atom_idx) = atom_idx_value.as_u64().map(|x| x as usize) else {
            continue;
        };
        let base = atom_idx * 3;
        if base + 2 >= positions.len() {
            continue;
        }
        let xyz = [
            positions[base].as_f64().unwrap_or(0.0),
            positions[base + 1].as_f64().unwrap_or(0.0),
            positions[base + 2].as_f64().unwrap_or(0.0),
        ];
        out.insert(mapped.residue_index, xyz);
    }
    Ok(out)
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn convert_topology_indices(
    topology_indices: Vec<i32>,
    residue_map: &BTreeMap<i32, MappedResidue>,
) -> Vec<i32> {
    let mut seen = BTreeSet::new();
    topology_indices
        .into_iter()
        .filter_map(|idx| residue_map.get(&idx).map(|r| r.residue_index))
        .filter(|idx| seen.insert(*idx))
        .collect()
}

fn load_materialized(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read materialized sites {}", path.display()))?;
    let root: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse materialized sites {}", path.display()))?;
    if root.get("schema_kind").and_then(|v| v.as_str()) != Some("pathb_binding_sites_materialized")
    {
        bail!(
            "{}: expected schema_kind pathb_binding_sites_materialized",
            path.display()
        );
    }
    Ok(root)
}

fn aggregate_pathb(
    root: &Value,
    residue_map: &BTreeMap<i32, MappedResidue>,
) -> Result<(
    BTreeMap<i32, ResidueAccumulator>,
    BTreeMap<(i32, i32), (u32, f64)>,
    usize,
)> {
    let mut per_residue: BTreeMap<i32, ResidueAccumulator> = BTreeMap::new();
    for mapped in residue_map.values() {
        per_residue
            .entry(mapped.residue_index)
            .or_insert_with(ResidueAccumulator::default);
    }

    let sites = root
        .get("binding_sites")
        .and_then(|v| v.as_array())
        .context("binding_sites must be an array")?;
    if sites.is_empty() {
        bail!("materialized file has zero binding_sites");
    }

    let mut pair_counts: BTreeMap<(i32, i32), (u32, f64)> = BTreeMap::new();

    for site in sites {
        let lining = convert_topology_indices(v_i32_vec(site, "lining_residues"), residue_map);
        if lining.is_empty() {
            continue;
        }
        let drivers: BTreeSet<i32> =
            convert_topology_indices(v_i32_vec(site, "driver_residues"), residue_map)
                .into_iter()
                .collect();

        let density = v_f64(site, &["score_components", "density_score"]);
        let temporal = v_f64(site, &["score_components", "temporal_persistence_factor"]);
        let stream_balance = v_f64(site, &["score_components", "stream_balance_factor"]);
        let residue_shell = v_f64(
            site,
            &["score_components", "residue_shell_plausibility_factor"],
        );
        let manifold = v_f64(
            site,
            &["score_components", "centroid_manifold_consistency_factor"],
        )
        .max(v_f64(
            site,
            &["manifold_consistency", "manifold_consistency_score"],
        ));
        let kcc_factor = v_f64(site, &["score_components", "kcc_driver_factor"]).max(1.0);
        let kcc_score = v_f64(site, &["kcc_driver", "kcc_driver_score"]);
        let rayleigh = v_f64(site, &["phase_support", "rayleigh_r_stat"]);
        let n_spikes = v_i64(site, &["n_spikes"]).unwrap_or(0).max(0) as f64;
        let spread = v_f64(site, &["centroid_spread_a"]);

        // The materializer does not emit legacy TIDE TE/Fisher rows. These
        // are conservative, documented channel lifts from materialized MD
        // evidence:
        //   te_out   := density × temporal persistence × stream balance,
        //               scaled by log spike support and KCC driver support.
        //   te_in    := residue-shell plausibility × manifold consistency,
        //               i.e. local sensitivity/enclosure support.
        //   delta_hc := phase coherence × manifold consistency.
        //   sigma    := centroid spread variance proxy; downstream EiV paths
        //               can down-weight high-spread sites.
        let spike_scale = n_spikes.ln_1p().max(1.0);
        let base_te_out =
            density.max(0.0) * temporal.max(0.0) * stream_balance.max(0.0) * spike_scale;
        let base_te_in = residue_shell.max(0.0) * manifold.max(0.0) * stream_balance.max(0.0);
        let site_delta_hc = rayleigh.max(0.0) * manifold.max(0.0);
        let site_sigma = (spread.max(0.0) / 10.0).powi(2);

        for &rid in &lining {
            let is_driver = drivers.contains(&rid);
            let driver_boost = if is_driver {
                kcc_factor.max(1.0) + kcc_score.max(0.0).ln_1p()
            } else {
                1.0
            };
            let acc = per_residue.entry(rid).or_default();
            acc.fold(
                base_te_out * driver_boost,
                base_te_in * if is_driver { 1.2 } else { 1.0 },
                site_delta_hc,
                site_sigma,
            );
        }

        for i in 0..lining.len() {
            for j in (i + 1)..lining.len() {
                let a = lining[i];
                let b = lining[j];
                if a == b {
                    continue;
                }
                let key = if a < b { (a, b) } else { (b, a) };
                let entry = pair_counts.entry(key).or_insert((0, 0.0));
                entry.0 = entry.0.saturating_add(1);
                entry.1 += rayleigh.clamp(-1.0, 1.0);
            }
        }
    }

    Ok((per_residue, pair_counts, sites.len()))
}

fn build_edges(
    pair_counts: BTreeMap<(i32, i32), (u32, f64)>,
    ca_positions: &BTreeMap<i32, [f64; 3]>,
    replicate_id: String,
) -> Vec<WTDynamicContactEdge> {
    let max_count = pair_counts.values().map(|(n, _)| *n).max().unwrap_or(1) as f64;
    pair_counts
        .into_iter()
        .filter_map(|((a, b), (count, corr_sum))| {
            if a == b || count == 0 {
                return None;
            }
            let mean_distance = match (ca_positions.get(&a), ca_positions.get(&b)) {
                (Some(x), Some(y)) => Some(dist3(*x, *y)),
                _ => None,
            };
            Some(WTDynamicContactEdge {
                source_uniprot_residue_index: a,
                target_uniprot_residue_index: b,
                contact_probability: (count as f64 / max_count).clamp(0.0, 1.0),
                mean_distance_angstrom: mean_distance,
                dynamic_correlation: Some((corr_sum / count as f64).clamp(-1.0, 1.0)),
                replicate_id: Some(replicate_id.clone()),
            })
        })
        .collect()
}

fn iso_8601_utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    let (year, month, day, hour, minute, second) = epoch_to_civil(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn epoch_to_civil(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let secs_in_day: u64 = 86400;
    let z = (secs / secs_in_day) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 {
        (mp + 3) as u32
    } else {
        (mp - 9) as u32
    };
    let year = y + (if m <= 2 { 1 } else { 0 });
    let time = secs % secs_in_day;
    let hour = (time / 3600) as u32;
    let minute = ((time % 3600) / 60) as u32;
    let second = (time % 60) as u32;
    (year, m, d, hour, minute, second)
}

fn build_response(
    args: &Args,
    residue_map: &BTreeMap<i32, MappedResidue>,
    per_residue: &BTreeMap<i32, ResidueAccumulator>,
    edges: Vec<WTDynamicContactEdge>,
) -> WTPhysicalProfileResponse {
    let mut residue_name_by_index = BTreeMap::new();
    for mapped in residue_map.values() {
        residue_name_by_index.insert(mapped.residue_index, mapped.residue_name.clone());
    }
    let residues = per_residue
        .iter()
        .map(|(&idx, acc)| WTResiduePhysics {
            uniprot_residue_index: idx,
            residue_name: residue_name_by_index
                .get(&idx)
                .cloned()
                .unwrap_or_else(|| "UNK".to_string()),
            te_out: acc.te_out,
            te_in: acc.te_in,
            delta_hc: acc.delta_hc,
            sigma_hydration_sq: acc.sigma_hydration_sq,
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
        dynamic_contact_edges: edges,
        physics_schema: PHYSICS_SCHEMA_TAG,
    }
}

fn write_json(response: &WTPhysicalProfileResponse, out: &Path) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let f = File::create(out).with_context(|| format!("create {}", out.display()))?;
    serde_json::to_writer_pretty(f, response)
        .with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

fn write_physics_parquet(
    response: &WTPhysicalProfileResponse,
    residue_map: &BTreeMap<i32, MappedResidue>,
    structure_anchor_id: &str,
    out: &Path,
) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("target", DataType::Utf8, false),
        Field::new("uniprot_accession", DataType::Utf8, false),
        Field::new("uniprot_residue_index", DataType::Int32, false),
        Field::new("residue_name", DataType::Utf8, false),
        Field::new("receptor_chain_id", DataType::Utf8, false),
        Field::new("structure_anchor_id", DataType::Utf8, false),
        Field::new("te_out", DataType::Float64, false),
        Field::new("te_in", DataType::Float64, false),
        Field::new("delta_hc", DataType::Float64, false),
        Field::new("sigma_hydration_sq", DataType::Float64, false),
        Field::new("replicate_id", DataType::Utf8, false),
    ]));
    let mut target_b = StringBuilder::new();
    let mut acc_b = StringBuilder::new();
    let mut idx_b = Int32Builder::with_capacity(response.residues.len());
    let mut name_b = StringBuilder::new();
    let mut chain_b = StringBuilder::new();
    let mut anchor_b = StringBuilder::new();
    let mut te_out_b = Float64Builder::with_capacity(response.residues.len());
    let mut te_in_b = Float64Builder::with_capacity(response.residues.len());
    let mut dh_b = Float64Builder::with_capacity(response.residues.len());
    let mut sigma_b = Float64Builder::with_capacity(response.residues.len());
    let mut rep_b = StringBuilder::new();

    let chain_by_index: BTreeMap<i32, String> = residue_map
        .values()
        .map(|r| (r.residue_index, r.chain_id.clone()))
        .collect();
    for r in &response.residues {
        target_b.append_value(&response.target);
        acc_b.append_value(&response.uniprot_accession);
        idx_b.append_value(r.uniprot_residue_index);
        name_b.append_value(&r.residue_name);
        chain_b.append_value(
            chain_by_index
                .get(&r.uniprot_residue_index)
                .map(String::as_str)
                .unwrap_or(""),
        );
        anchor_b.append_value(structure_anchor_id);
        te_out_b.append_value(r.te_out);
        te_in_b.append_value(r.te_in);
        dh_b.append_value(r.delta_hc);
        sigma_b.append_value(r.sigma_hydration_sq);
        rep_b.append_value(&response.prism_run_id);
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(target_b.finish()) as ArrayRef,
            Arc::new(acc_b.finish()),
            Arc::new(idx_b.finish()),
            Arc::new(name_b.finish()),
            Arc::new(chain_b.finish()),
            Arc::new(anchor_b.finish()),
            Arc::new(te_out_b.finish()),
            Arc::new(te_in_b.finish()),
            Arc::new(dh_b.finish()),
            Arc::new(sigma_b.finish()),
            Arc::new(rep_b.finish()),
        ],
    )?;
    write_one_batch(out, schema, batch)
}

fn write_contact_parquet(
    response: &WTPhysicalProfileResponse,
    structure_anchor_id: &str,
    out: &Path,
) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("structure_anchor_id", DataType::Utf8, false),
        Field::new("source_uniprot_residue_index", DataType::Int32, false),
        Field::new("target_uniprot_residue_index", DataType::Int32, false),
        Field::new("contact_probability", DataType::Float64, false),
        Field::new("mean_distance_angstrom", DataType::Float64, true),
        Field::new("dynamic_correlation", DataType::Float64, true),
        Field::new("replicate_id", DataType::Utf8, true),
    ]));
    let mut anchor_b = StringBuilder::new();
    let mut src_b = Int32Builder::with_capacity(response.dynamic_contact_edges.len());
    let mut dst_b = Int32Builder::with_capacity(response.dynamic_contact_edges.len());
    let mut prob_b = Float64Builder::with_capacity(response.dynamic_contact_edges.len());
    let mut dist_b = Float64Builder::with_capacity(response.dynamic_contact_edges.len());
    let mut corr_b = Float64Builder::with_capacity(response.dynamic_contact_edges.len());
    let mut rep_b = StringBuilder::new();
    for e in &response.dynamic_contact_edges {
        anchor_b.append_value(structure_anchor_id);
        src_b.append_value(e.source_uniprot_residue_index);
        dst_b.append_value(e.target_uniprot_residue_index);
        prob_b.append_value(e.contact_probability);
        match e.mean_distance_angstrom {
            Some(v) => dist_b.append_value(v),
            None => dist_b.append_null(),
        }
        match e.dynamic_correlation {
            Some(v) => corr_b.append_value(v),
            None => corr_b.append_null(),
        }
        match e.replicate_id.as_deref() {
            Some(v) => rep_b.append_value(v),
            None => rep_b.append_null(),
        }
    }
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(anchor_b.finish()) as ArrayRef,
            Arc::new(src_b.finish()),
            Arc::new(dst_b.finish()),
            Arc::new(prob_b.finish()),
            Arc::new(dist_b.finish()),
            Arc::new(corr_b.finish()),
            Arc::new(rep_b.finish()),
        ],
    )?;
    write_one_batch(out, schema, batch)
}

fn write_one_batch(out: &Path, schema: Arc<Schema>, batch: RecordBatch) -> Result<()> {
    let file = File::create(out).with_context(|| format!("create {}", out.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let residue_map = load_residue_map(&args.residue_map, &args.receptor_chain_id)?;
    let ca_positions = load_ca_positions(args.topology_json.as_deref(), &residue_map)?;
    let materialized = load_materialized(&args.materialized_sites)?;
    let (per_residue, pair_counts, sites_consumed) = aggregate_pathb(&materialized, &residue_map)?;
    let edges = build_edges(
        pair_counts,
        &ca_positions,
        format!("pathb_lining_comembership:{}", args.structure_anchor_id),
    );
    if edges.is_empty() {
        bail!("Path-B export produced no dynamic contact edges");
    }
    let response = build_response(&args, &residue_map, &per_residue, edges);
    write_json(&response, &args.out_json)?;
    write_physics_parquet(
        &response,
        &residue_map,
        &args.structure_anchor_id,
        &args.out_parquet,
    )?;
    write_contact_parquet(
        &response,
        &args.structure_anchor_id,
        &args.out_contact_parquet,
    )?;
    eprintln!(
        "[dstw_export_wt_pathb] target={} anchor={} sites={} residues={} edges={}",
        response.target,
        args.structure_anchor_id,
        sites_consumed,
        response.residues.len(),
        response.dynamic_contact_edges.len(),
    );
    eprintln!("[dstw_export_wt_pathb] wrote {}", args.out_json.display());
    eprintln!(
        "[dstw_export_wt_pathb] wrote {}",
        args.out_parquet.display()
    );
    eprintln!(
        "[dstw_export_wt_pathb] wrote {}",
        args.out_contact_parquet.display()
    );
    Ok(())
}
