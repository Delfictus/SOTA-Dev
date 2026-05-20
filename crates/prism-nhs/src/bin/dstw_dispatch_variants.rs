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
    dispatch_variant_batch, DispatchError, PRISMExecutionRequest, VariantDispatchConfig,
    WTTensorPack,
};
use prism_nhs::dstw_dispatch::projection::ProjectionConfig;
use std::{fs::File, io::BufReader, path::PathBuf};

// Parquet reader for the WT tensor pack.
use arrow_array::{Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

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

    /// PRISM run id stamped on every response row.
    #[arg(long)]
    prism_run_id: String,

    /// Where to write the response JSON.
    #[arg(long)]
    out_json: PathBuf,

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
    let mut te_out: Vec<f64> = Vec::new();
    let mut te_in: Vec<f64> = Vec::new();
    let mut delta_hc: Vec<f64> = Vec::new();
    let mut sigma_hyd: Vec<f64> = Vec::new();
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
        te_out.extend(to_f64("te_out")?);
        te_in.extend(to_f64("te_in")?);
        delta_hc.extend(to_f64("delta_hc")?);
        sigma_hyd.extend(to_f64("sigma_hydration_sq")?);
    }
    if indices.is_empty() {
        bail!("WT pack {} is empty", path.display());
    }
    // Sort by uniprot_residue_index so the vectors are dense and contiguous.
    let mut by_idx: Vec<(i32, f64, f64, f64, f64)> = indices
        .into_iter()
        .zip(te_out)
        .zip(te_in)
        .zip(delta_hc)
        .zip(sigma_hyd)
        .map(|((((i, a), b), c), d)| (i, a, b, c, d))
        .collect();
    by_idx.sort_by_key(|t| t.0);
    let lo = by_idx.first().unwrap().0;
    let hi = by_idx.last().unwrap().0;
    let n = (hi - lo + 1) as usize;
    if by_idx.len() != n {
        bail!(
            "WT pack has {} rows but residue range [{}, {}] expects {} dense entries",
            by_idx.len(), lo, hi, n
        );
    }
    let mut te_out_v = Vec::with_capacity(n);
    let mut te_in_v = Vec::with_capacity(n);
    let mut delta_hc_v = Vec::with_capacity(n);
    let mut sigma_hyd_v = Vec::with_capacity(n);
    for (_, a, b, c, d) in by_idx {
        te_out_v.push(a);
        te_in_v.push(b);
        delta_hc_v.push(c);
        sigma_hyd_v.push(d);
    }
    Ok(WTTensorPack {
        residue_index_lo: lo,
        residue_index_hi: hi,
        te_out: te_out_v,
        te_in: te_in_v,
        delta_hc: delta_hc_v,
        sigma_hydration_sq: sigma_hyd_v,
        // Replicate-spread variances are not yet emitted by `dstw-export-wt`.
        // Default to zero so the model_residual_variance terms carry the
        // full uncertainty; when the WT exporter adds these columns the
        // loader will read them.  TODO when that column lands.
        var_te_in: vec![0.0; n],
        var_delta_hc: vec![0.0; n],
        var_sigma_hydration_sq: vec![0.0; n],
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let request = load_request(&args.request)?;
    let wt = load_wt_pack(&args.wt_tensor_pack)?;
    let cfg = VariantDispatchConfig {
        prism_run_id: args.prism_run_id.clone(),
        projection: ProjectionConfig {
            alpha_q: args.alpha_q,
            alpha_v: args.alpha_v,
            sigma_q: 0.0,
            sigma_v: 0.0,
            model_residual_variance_active: args.residual_variance_active,
            model_residual_variance_lock: args.residual_variance_lock,
            model_residual_variance_ensemble: args.residual_variance_ensemble,
            backbone_rmsd_ceiling_angstrom: 0.5,
            jacobian_condition_ceiling: 1.0e6,
            nonconverged_sigma_penalty: args.nonconverged_sigma_penalty,
        },
    };
    let response = match dispatch_variant_batch(&request, &wt, &cfg) {
        Ok(r) => r,
        Err(DispatchError::SchemaMismatch(m)) => bail!("schema mismatch: {m}"),
        Err(DispatchError::NonFinite { target, variant, channel }) => bail!(
            "non-finite {channel} emitted for variant ({target}, {variant})"
        ),
        Err(other) => bail!("{other}"),
    };
    if let Some(parent) = args.out_json.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let out = File::create(&args.out_json)
        .with_context(|| format!("create {}", args.out_json.display()))?;
    serde_json::to_writer_pretty(out, &response)?;
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
    eprintln!("[dstw_dispatch_variants] wrote: {}", args.out_json.display());
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
        PRISMExecutionAcquisition, REQUIRED_DELTA_CHANNELS, RESPONSE_SCHEMA_TAG,
        VariantExecutionRequest,
    };

    fn synth_wt(n: usize) -> WTTensorPack {
        WTTensorPack {
            residue_index_lo: 1,
            residue_index_hi: n as i32,
            te_out: vec![0.0; n],
            te_in: (0..n).map(|i| 0.5 + 0.05 * i as f64).collect(),
            delta_hc: (0..n).map(|i| 0.3 + 0.02 * i as f64).collect(),
            sigma_hydration_sq: (0..n).map(|i| 0.1 + 0.01 * i as f64).collect(),
            var_te_in: vec![0.005; n],
            var_delta_hc: vec![0.005; n],
            var_sigma_hydration_sq: vec![0.005; n],
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
    fn cuda_dispatcher_returns_cpu_fallback_in_draft() {
        let kernel = CudaPropagationKernel::new().unwrap();
        let n = 32;
        let nv = 2;
        let shape = PropagationShape { n_residues: n, n_variants: nv };
        let k = synth_k_for_cuda(n);
        let residue_id = vec![0, 5];
        let pert = vec![1.0, 1.0];
        let mut delta_p = vec![0.0f32; shape.delta_p_len()];
        let backend = kernel.dispatch(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
        assert_eq!(backend, Backend::CpuFallback);
        // Verify channel index constant is wired through.
        assert_eq!(CH_ACTIVE, 0);
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
