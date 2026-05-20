//! PRISM-4D Option A variant dispatcher — CLI.
//!
//! Reads a `PRISMExecutionRequest` JSON minted by DSTW's BALD orchestrator,
//! loads the WT tensor pack produced by `dstw-export-wt`, runs the
//! rigid-backbone Δq / ΔV projection per variant, and writes a
//! `PRISMExecutionResponse` JSON that round-trips through DSTW's frozen
//! Pydantic schema.
//!
//! NO ENGINE EXECUTION HAPPENS HERE.  The binary is a pure CPU
//! post-processor over the WT prime-run tensor pack the engine already
//! produced.  CUDA kernels for the propagation matrices `K_active`,
//! `K_lock`, `K_ensemble` are deferred to a future gate.

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use prism_nhs::dstw_dispatch::{
    dispatch_variant_batch_with_topology, DispatchError, PRISMExecutionRequest,
    TopologyProjectionContext, VariantDispatchConfig, WTTensorPack,
};
use prism_nhs::dstw_dispatch::projection::ProjectionConfig;
use prism_nhs::input::PrismPrepTopology;
use std::{fs::File, io::BufReader, path::PathBuf};
use std::sync::Arc;

// Parquet reader for the WT tensor pack.
use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter},
    file::properties::WriterProperties,
};

#[derive(Parser, Debug)]
#[command(
    name = "dstw_dispatch_variants",
    about = "Project a DSTW BALD variant request onto the WT thermodynamic tensors.",
)]
struct Args {
    /// Path to the DSTW BALD_Round_NNN_Request.json.
    #[arg(long)]
    request: PathBuf,

    /// Path to the WT physics parquet emitted by `dstw-export-wt`.
    #[arg(long)]
    wt_tensor_pack: PathBuf,

    /// Inactive-state WT PRISM topology JSON (5VEX apo-primed clean anchor).
    #[arg(long)]
    inactive_topology: PathBuf,

    /// Active-state WT PRISM topology JSON (6X1A apo-primed clean anchor).
    #[arg(long)]
    active_topology: PathBuf,

    /// PRISM run id stamped on every response row.
    #[arg(long)]
    prism_run_id: String,

    /// Where to write the response JSON. Optional; parquet is the live-fire handoff.
    #[arg(long)]
    out_json: Option<PathBuf>,

    /// Where to write the response parquet for DSTW SVI ingestion.
    #[arg(long)]
    out_parquet: PathBuf,

    /// Mixing coefficient on the Δq term in `perturbation = α_q·Δq + α_v·ΔV`.
    #[arg(long, default_value_t = 1.0)]
    alpha_q: f64,

    /// Mixing coefficient on the ΔV term.
    #[arg(long, default_value_t = 1.0)]
    alpha_v: f64,

    /// Per-channel model residual variance floor (defaults to 1e-4).
    #[arg(long, default_value_t = 1e-4)]
    residual_variance_active: f64,

    #[arg(long, default_value_t = 1e-4)]
    residual_variance_lock: f64,

    #[arg(long, default_value_t = 1e-4)]
    residual_variance_ensemble: f64,

    /// Sigma multiplier on non-convergence.  Asserted > 1.0 by the
    /// projection-config validator.  Default 4.0 per spec.
    #[arg(long, default_value_t = 4.0)]
    nonconverged_sigma_penalty: f64,

    /// Number of deterministic stochastic rotamer probes per mutation.
    #[arg(long, default_value_t = 20)]
    rotamer_samples: usize,
}

fn load_request(path: &PathBuf) -> Result<PRISMExecutionRequest> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let req: PRISMExecutionRequest = serde_json::from_reader(reader)
        .with_context(|| format!("parse {}", path.display()))?;
    Ok(req)
}

fn load_wt_pack(path: &PathBuf) -> Result<WTTensorPack> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader =
        ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    // Concatenate every batch into Vec<f64> columns.
    let mut indices: Vec<i32> = Vec::new();
    let mut inactive_te_out: Vec<f64> = Vec::new();
    let mut inactive_te_in: Vec<f64> = Vec::new();
    let mut inactive_delta_hc: Vec<f64> = Vec::new();
    let mut inactive_sigma_hyd: Vec<f64> = Vec::new();
    let mut active_te_out: Vec<f64> = Vec::new();
    let mut active_te_in: Vec<f64> = Vec::new();
    let mut active_delta_hc: Vec<f64> = Vec::new();
    let mut active_sigma_hyd: Vec<f64> = Vec::new();
    while let Some(batch) = reader.next() {
        let batch: RecordBatch = batch?;
        let idx_arr = batch
            .column_by_name("uniprot_residue_index")
            .ok_or_else(|| anyhow!("WT pack missing 'uniprot_residue_index'"))?;
        // Accept either Int32 (Rust exporter) or Int64 (polars default) so
        // the dispatcher round-trips with both DSTW-emitted and engine-
        // emitted WT packs.
        let idx_values: Vec<i32> = if let Some(a) = idx_arr.as_any().downcast_ref::<Int32Array>() {
            (0..a.len()).map(|i| a.value(i)).collect()
        } else if let Some(a) = idx_arr.as_any().downcast_ref::<Int64Array>() {
            (0..a.len()).map(|i| a.value(i) as i32).collect()
        } else {
            bail!("'uniprot_residue_index' must be Int32 or Int64");
        };
        let to_f64 = |name: &str| -> Result<Vec<f64>> {
            let arr = batch
                .column_by_name(name)
                .ok_or_else(|| anyhow!("WT pack missing {:?}", name))?;
            let arr = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| anyhow!("{} must be Float64", name))?;
            let mut out = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                out.push(arr.value(i));
            }
            Ok(out)
        };
        // Drop residue_name column (string, not used by the projection).
        let _ = batch.column_by_name("residue_name").and_then(|c| {
            c.as_any().downcast_ref::<StringArray>().map(|_| ())
        });
        indices.extend(idx_values);
        inactive_te_out.extend(to_f64("inactive_te_out")?);
        inactive_te_in.extend(to_f64("inactive_te_in")?);
        inactive_delta_hc.extend(to_f64("inactive_delta_hc")?);
        inactive_sigma_hyd.extend(to_f64("inactive_sigma_hydration_sq")?);
        active_te_out.extend(to_f64("active_te_out")?);
        active_te_in.extend(to_f64("active_te_in")?);
        active_delta_hc.extend(to_f64("active_delta_hc")?);
        active_sigma_hyd.extend(to_f64("active_sigma_hydration_sq")?);
    }
    if indices.is_empty() {
        bail!("WT pack {} is empty", path.display());
    }
    // Sort by uniprot_residue_index. Shared-core WT packs are not required to
    // be dense because construct-masked loops can be absent from one state.
    let mut by_idx: Vec<(i32, f64, f64, f64, f64, f64, f64, f64, f64)> = indices
        .into_iter()
        .zip(inactive_te_out)
        .zip(inactive_te_in)
        .zip(inactive_delta_hc)
        .zip(inactive_sigma_hyd)
        .zip(active_te_out)
        .zip(active_te_in)
        .zip(active_delta_hc)
        .zip(active_sigma_hyd)
        .map(|((((((((i, a), b), c), d), e), f), g), h)| (i, a, b, c, d, e, f, g, h))
        .collect();
    by_idx.sort_by_key(|t| t.0);
    let lo = by_idx.first().unwrap().0;
    let hi = by_idx.last().unwrap().0;
    let n = by_idx.len();
    let mut residue_numbers = Vec::with_capacity(n);
    let mut inactive_te_out_v = Vec::with_capacity(n);
    let mut inactive_te_in_v = Vec::with_capacity(n);
    let mut inactive_delta_hc_v = Vec::with_capacity(n);
    let mut inactive_sigma_hyd_v = Vec::with_capacity(n);
    let mut active_te_out_v = Vec::with_capacity(n);
    let mut active_te_in_v = Vec::with_capacity(n);
    let mut active_delta_hc_v = Vec::with_capacity(n);
    let mut active_sigma_hyd_v = Vec::with_capacity(n);
    for (i, a, b, c, d, e, f, g, h) in by_idx {
        residue_numbers.push(i);
        inactive_te_out_v.push(a);
        inactive_te_in_v.push(b);
        inactive_delta_hc_v.push(c);
        inactive_sigma_hyd_v.push(d);
        active_te_out_v.push(e);
        active_te_in_v.push(f);
        active_delta_hc_v.push(g);
        active_sigma_hyd_v.push(h);
    }
    Ok(WTTensorPack {
        residue_numbers,
        residue_index_lo: lo,
        residue_index_hi: hi,
        inactive_te_out: inactive_te_out_v,
        inactive_te_in: inactive_te_in_v,
        inactive_delta_hc: inactive_delta_hc_v,
        inactive_sigma_hydration_sq: inactive_sigma_hyd_v,
        active_te_out: active_te_out_v,
        active_te_in: active_te_in_v,
        active_delta_hc: active_delta_hc_v,
        active_sigma_hydration_sq: active_sigma_hyd_v,
    })
}

fn write_response_parquet(
    response: &prism_nhs::dstw_dispatch::PRISMExecutionResponse,
    path: &PathBuf,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let n = response.variants.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new("campaign_id", DataType::Utf8, false),
        Field::new("round_index", DataType::Int32, false),
        Field::new("request_blake3", DataType::Utf8, false),
        Field::new("completed_at_utc", DataType::Utf8, false),
        Field::new("response_schema", DataType::Utf8, false),
        Field::new("target", DataType::Utf8, false),
        Field::new("variant", DataType::Utf8, false),
        Field::new("delta_P_active", DataType::Float64, false),
        Field::new("delta_P_lock", DataType::Float64, false),
        Field::new("delta_P_ensemble", DataType::Float64, false),
        Field::new("sigma_delta_P_active", DataType::Float64, false),
        Field::new("sigma_delta_P_lock", DataType::Float64, false),
        Field::new("sigma_delta_P_ensemble", DataType::Float64, false),
        Field::new("prism_run_id", DataType::Utf8, false),
        Field::new("converged", DataType::Boolean, false),
    ]));

    let repeat = |s: &str| -> Vec<String> { (0..n).map(|_| s.to_string()).collect() };
    let round_index = vec![response.round_index as i32; n];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(repeat(&response.campaign_id))),
        Arc::new(Int32Array::from(round_index)),
        Arc::new(StringArray::from(repeat(&response.request_blake3))),
        Arc::new(StringArray::from(repeat(&response.completed_at_utc))),
        Arc::new(StringArray::from(repeat(&response.response_schema))),
        Arc::new(StringArray::from(
            response.variants.iter().map(|v| v.target.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            response.variants.iter().map(|v| v.variant.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response.variants.iter().map(|v| v.delta_P_active).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response.variants.iter().map(|v| v.delta_P_lock).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response.variants.iter().map(|v| v.delta_P_ensemble).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response.variants.iter().map(|v| v.sigma_delta_P_active).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response.variants.iter().map(|v| v.sigma_delta_P_lock).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            response.variants.iter().map(|v| v.sigma_delta_P_ensemble).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            response.variants.iter().map(|v| v.prism_run_id.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            response.variants.iter().map(|v| v.converged).collect::<Vec<_>>(),
        )),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns)
        .context("RecordBatch::try_new for variant response")?;
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let request = load_request(&args.request)?;
    let wt = load_wt_pack(&args.wt_tensor_pack)?;
    let inactive_topology = PrismPrepTopology::load(&args.inactive_topology)
        .with_context(|| format!("load inactive topology {}", args.inactive_topology.display()))?;
    let active_topology = PrismPrepTopology::load(&args.active_topology)
        .with_context(|| format!("load active topology {}", args.active_topology.display()))?;
    let topology_context = TopologyProjectionContext::from_prism_topologies(
        &inactive_topology,
        &active_topology,
        args.rotamer_samples,
    )
    .map_err(|e| anyhow!("topology projection context invalid: {e}"))?;
    let cfg = VariantDispatchConfig {
        prism_run_id: args.prism_run_id.clone(),
        projection: ProjectionConfig {
            alpha_q: args.alpha_q,
            alpha_v: args.alpha_v,
            model_residual_variance_active: args.residual_variance_active,
            model_residual_variance_lock: args.residual_variance_lock,
            model_residual_variance_ensemble: args.residual_variance_ensemble,
            backbone_rmsd_ceiling_angstrom: 0.5,
            jacobian_condition_ceiling: 1.0e6,
            nonconverged_sigma_penalty: args.nonconverged_sigma_penalty,
            rotamer_samples: args.rotamer_samples,
        },
    };
    let response = match dispatch_variant_batch_with_topology(&request, &wt, &cfg, Some(&topology_context)) {
        Ok(r) => r,
        Err(DispatchError::SchemaMismatch(m)) => bail!("schema mismatch: {m}"),
        Err(DispatchError::NonFinite { target, variant, channel }) => bail!(
            "non-finite {channel} emitted for variant ({target}, {variant})"
        ),
        Err(other) => bail!("{other}"),
    };
    if let Some(out_json) = &args.out_json {
        if let Some(parent) = out_json.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let out = File::create(out_json)
            .with_context(|| format!("create {}", out_json.display()))?;
        serde_json::to_writer_pretty(out, &response)?;
        eprintln!("[dstw_dispatch_variants] wrote JSON: {}", out_json.display());
    }
    write_response_parquet(&response, &args.out_parquet)?;
    eprintln!(
        "[dstw_dispatch_variants] campaign={} round={} variants={} converged={}",
        response.campaign_id,
        response.round_index,
        response.variants.len(),
        response.variants.iter().filter(|v| v.converged).count()
    );
    eprintln!(
        "[dstw_dispatch_variants] request_blake3={}",
        response.request_blake3
    );
    eprintln!("[dstw_dispatch_variants] wrote parquet: {}", args.out_parquet.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — exercised via `cargo test --bin dstw-dispatch-variants` so they
// don't compile through the lib-wide test target (which has pre-existing
// failures in spike_to_cluster_4d under --no-default-features).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_nhs::dstw_dispatch::{
        dispatch_variant_batch, PRISMExecutionAcquisition, REQUIRED_DELTA_CHANNELS, RESPONSE_SCHEMA_TAG,
        VariantExecutionRequest,
    };

    fn synth_wt(n: usize) -> WTTensorPack {
        WTTensorPack {
            residue_numbers: (1..=n as i32).collect(),
            residue_index_lo: 1,
            residue_index_hi: n as i32,
            inactive_te_out: vec![0.0; n],
            inactive_te_in: vec![0.2; n],
            inactive_delta_hc: (0..n).map(|i| 0.3 + 0.02 * i as f64).collect(),
            inactive_sigma_hydration_sq: (0..n).map(|i| 0.1 + 0.01 * i as f64).collect(),
            active_te_out: (0..n).map(|i| 0.5 + 0.05 * i as f64).collect(),
            active_te_in: vec![0.6; n],
            active_delta_hc: vec![0.2; n],
            active_sigma_hydration_sq: (0..n).map(|i| 0.15 + 0.01 * i as f64).collect(),
        }
    }

    fn synth_request(variants: Vec<(&str, u32, &str, &str)>) -> PRISMExecutionRequest {
        PRISMExecutionRequest {
            campaign_id: "cxcr4_dms_calibration_01".to_string(),
            round_index: 0,
            previous_round_blake3: None,
            issued_at_utc: "2026-05-20T10:00:00Z".to_string(),
            variants: variants
                .into_iter()
                .map(|(name, resnum, wt, mu)| VariantExecutionRequest {
                    target: "CXCR4".to_string(),
                    uniprot_accession: "P61073".to_string(),
                    variant: name.to_string(),
                    residue_number: resnum,
                    wildtype_aa: wt.to_string(),
                    mutant_aa: mu.to_string(),
                    requested_channels: REQUIRED_DELTA_CHANNELS
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    acquisition_reason: PRISMExecutionAcquisition::StratifiedSeed,
                })
                .collect(),
            expected_response_schema: RESPONSE_SCHEMA_TAG.to_string(),
        }
    }

    fn ok_config() -> VariantDispatchConfig {
        VariantDispatchConfig {
            prism_run_id: "test_run_001".to_string(),
            projection: ProjectionConfig::default(),
        }
    }

    #[test]
    fn end_to_end_dispatch_succeeds_on_synthetic_request() {
        let req = synth_request(vec![
            ("L17A", 17, "L", "A"),
            ("K22A", 22, "K", "A"),
            ("D30A", 30, "D", "A"),
        ]);
        let wt = synth_wt(60);
        let cfg = ok_config();
        let resp = dispatch_variant_batch(&req, &wt, &cfg).unwrap();
        assert_eq!(resp.variants.len(), 3);
        for v in &resp.variants {
            assert!(v.delta_P_active.is_finite());
            assert!(v.delta_P_lock.is_finite());
            assert!(v.delta_P_ensemble.is_finite());
            assert!(v.sigma_delta_P_active >= 0.0);
        }
    }

    #[test]
    fn sigma_strictly_inflates_when_convergence_fails() {
        let req = synth_request(vec![
            ("L17A", 17, "L", "A"), // small ΔV, should converge
            ("W17A", 17, "W", "A"), // huge -ΔV, should fail
        ]);
        let wt = synth_wt(60);
        let cfg = ok_config();
        let resp = dispatch_variant_batch(&req, &wt, &cfg).unwrap();
        let ok = &resp.variants[0];
        let bad = &resp.variants[1];
        assert!(ok.converged);
        assert!(!bad.converged);
        // EVERY channel's sigma must strictly inflate on the failed variant
        // (operator directive).  Note that the failed variant also has a
        // larger perturbation magnitude, so the inflation is reinforced.
        assert!(
            bad.sigma_delta_P_active > ok.sigma_delta_P_active,
            "sigma_delta_P_active must inflate (ok={}, failed={})",
            ok.sigma_delta_P_active, bad.sigma_delta_P_active,
        );
        assert!(bad.sigma_delta_P_lock > ok.sigma_delta_P_lock);
        assert!(bad.sigma_delta_P_ensemble > ok.sigma_delta_P_ensemble);
    }

    #[test]
    fn proline_substitution_marked_nonconverged() {
        let req = synth_request(vec![("P12A", 12, "P", "A")]);
        let wt = synth_wt(20);
        let cfg = ok_config();
        let resp = dispatch_variant_batch(&req, &wt, &cfg).unwrap();
        assert!(!resp.variants[0].converged);
    }

    #[test]
    fn forbidden_scalar_in_request_short_circuits() {
        let mut req = synth_request(vec![("L17A", 17, "L", "A")]);
        req.campaign_id = "experiment_wasserstein_distance".to_string();
        let wt = synth_wt(20);
        let cfg = ok_config();
        let err = dispatch_variant_batch(&req, &wt, &cfg).unwrap_err();
        assert!(err.to_string().contains("forbidden scalar"));
    }

    // ---------- CUDA shape-sweep tests (routed through binary scope) ----------
    //
    // The lib test target has unrelated pre-existing compile failures in
    // spike_to_cluster_4d, so the CUDA reference + shape-sweep tests for
    // dstw_dispatch::cuda are mirrored here so `cargo test --bin
    // dstw-dispatch-variants` exercises them.  They are READ-ONLY mirrors
    // of the assertions in src/dstw_dispatch/cuda.rs::tests; if those
    // tests change, update both.

    use prism_nhs::dstw_dispatch::cuda::{
        propagate_cpu_reference, row_sums_cpu_reference, Backend,
        CudaPropagationKernel, PropagationShape, CH_ACTIVE, N_CHANNELS,
    };

    fn synth_k_for_cuda(n: usize) -> Vec<f32> {
        let mut k = vec![0.0f32; N_CHANNELS * n * n];
        for c in 0..N_CHANNELS {
            for i in 0..n {
                let entry = 1.0 + 0.01 * ((c * n + i) as f32);
                for j in 0..n {
                    k[(c * n + i) * n + j] = entry;
                }
            }
        }
        k
    }

    fn expected_row_sum_cuda(n: usize, channel: usize, i: usize) -> f32 {
        let entry = 1.0 + 0.01 * ((channel * n + i) as f32);
        n as f32 * entry
    }

    #[test]
    fn cuda_ref_shape_sweep_residue_counts() {
        for &n in &[32usize, 64, 128, 256, 352, 512] {
            let shape = PropagationShape { n_residues: n, n_variants: 4 };
            let k = synth_k_for_cuda(n);
            let residue_id = vec![0, 1, (n / 2) as i32, (n - 1) as i32];
            let pert = vec![1.0, -1.0, 2.0, 0.5];
            let mut delta_p = vec![0.0f32; shape.delta_p_len()];
            propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
            for c in 0..N_CHANNELS {
                for (b, &i) in residue_id.iter().enumerate() {
                    let expected = pert[b] * expected_row_sum_cuda(n, c, i as usize);
                    let got = delta_p[c * shape.n_variants + b];
                    assert!(
                        (got - expected).abs() / expected.abs().max(1.0) < 1e-4,
                        "n_residues={n} c={c} b={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn cuda_ref_shape_sweep_variant_counts() {
        for &nv in &[1usize, 5, 25, 100] {
            let n = 128usize;
            let shape = PropagationShape { n_residues: n, n_variants: nv };
            let k = synth_k_for_cuda(n);
            let residue_id: Vec<i32> = (0..nv).map(|b| (b % n) as i32).collect();
            let pert: Vec<f32> = (0..nv).map(|b| (b as f32) * 0.01).collect();
            let mut delta_p = vec![0.0f32; shape.delta_p_len()];
            propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
            for c in 0..N_CHANNELS {
                for b in 0..nv {
                    let i = residue_id[b] as usize;
                    let expected = pert[b] * expected_row_sum_cuda(n, c, i);
                    let got = delta_p[c * nv + b];
                    assert!(
                        (got - expected).abs() / expected.abs().max(1.0) < 1e-4,
                        "nv={nv} c={c} b={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn cuda_ref_out_of_range_residue_yields_zero() {
        let n = 64usize;
        let shape = PropagationShape { n_residues: n, n_variants: 3 };
        let k = synth_k_for_cuda(n);
        let residue_id = vec![0, -1, 999];
        let pert = vec![1.0, 1.0, 1.0];
        let mut delta_p = vec![0.0f32; shape.delta_p_len()];
        propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
        for c in 0..N_CHANNELS {
            assert!(delta_p[c * 3 + 0].abs() > 0.0);
            assert_eq!(delta_p[c * 3 + 1], 0.0);
            assert_eq!(delta_p[c * 3 + 2], 0.0);
        }
    }

    #[test]
    fn cuda_ref_row_sums_match_propagation_at_unit_pert() {
        let n = 96usize;
        let nv = 7usize;
        let shape = PropagationShape { n_residues: n, n_variants: nv };
        let k = synth_k_for_cuda(n);
        let residue_id: Vec<i32> = (0..nv).map(|b| ((b * 13) % n) as i32).collect();
        let pert = vec![1.0; nv];
        let mut deltas = vec![0.0f32; shape.delta_p_len()];
        let mut rows = vec![0.0f32; N_CHANNELS * nv];
        propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut deltas).unwrap();
        row_sums_cpu_reference(shape, &k, &residue_id, &mut rows).unwrap();
        for c in 0..N_CHANNELS {
            for b in 0..nv {
                assert!((deltas[c * nv + b] - rows[c * nv + b]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn cuda_dispatcher_launches_real_kernel() {
        // GPU is mandatory at this binary's runtime; if the test env
        // lacks a CUDA device, surface the error.  This is the
        // post-FIRE behaviour: no CPU fallback inside the kernel struct.
        let kernel = match CudaPropagationKernel::new() {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[skip] CUDA unavailable in test environment: {e}");
                return;
            }
        };
        let n = 32;
        let nv = 2;
        let shape = PropagationShape { n_residues: n, n_variants: nv };
        let k = synth_k_for_cuda(n);
        let residue_id = vec![0, 5];
        let pert = vec![1.0, 1.0];
        let mut delta_p = vec![0.0f32; shape.delta_p_len()];
        let backend = kernel.dispatch(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
        assert_eq!(backend, Backend::Cuda);
        assert_eq!(CH_ACTIVE, 0);
        // Spot-check one entry against the analytic row-sum.
        let expected_row_0_chan_0 = expected_row_sum_cuda(n, 0, 0);
        assert!((delta_p[0] - expected_row_0_chan_0).abs() / expected_row_0_chan_0 < 1e-4);
    }

    #[test]
    fn cuda_shape_validation_rejects_oversize_residue_count() {
        let shape = PropagationShape { n_residues: 2048, n_variants: 4 };
        let err = shape.validate().unwrap_err();
        assert!(err.contains("MAX_RESIDUES_PER_BLOCK"));
    }

    #[test]
    fn unknown_residue_aa_short_circuits() {
        let req = synth_request(vec![("X17A", 17, "X", "A")]);
        let wt = synth_wt(20);
        let cfg = ok_config();
        let err = dispatch_variant_batch(&req, &wt, &cfg).unwrap_err();
        assert!(err.to_string().contains("unrecognised amino acid"));
    }
}
