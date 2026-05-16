//! Production driver for PRISM Twin teacher consensus packaging.
//!
//! This binary turns the previously manual Stage-B -> ensemble -> Stage-C
//! chain into one fail-closed command surface.  It may consume raw
//! `md_evidence_manifest.json` files or already-materialized Stage-B parquet
//! files, validates the contract at each boundary, and emits a provenance
//! manifest beside the final teacher/student bundles.

use anyhow::{bail, Context, Result};
use chrono::Utc;
use clap::Parser;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Parser, Debug)]
#[command(name = "prism-teacher-pipeline")]
#[command(about = "Run validated PRISM Twin Stage-B/C teacher consensus packaging")]
struct Args {
    /// Raw MD evidence manifests. Each one is materialized through prism-teacher-tags.
    #[arg(long = "replica-manifest")]
    replica_manifests: Vec<PathBuf>,

    /// Existing Stage-B teacher_residue_tags.parquet files.
    #[arg(long = "stage-b-parquet")]
    stage_b_parquets: Vec<PathBuf>,

    #[arg(long)]
    target_id: String,

    #[arg(long)]
    output_dir: PathBuf,

    #[arg(long, default_value_t = 42)]
    base_seed: u64,

    #[arg(long, default_value_t = 3)]
    min_replicas: usize,

    #[arg(long, default_value_t = 0.05)]
    min_trainable_fraction: f64,

    #[arg(long, default_value_t = 2.0)]
    strict_rhat: f64,

    #[arg(long, default_value_t = 5.0)]
    soft_rhat: f64,

    #[arg(long, default_value_t = 3)]
    min_replicas_per_feature: u32,

    #[arg(long, default_value_t = 1)]
    max_outlier_features: u32,

    #[arg(long, default_value_t = 0.55)]
    min_phase_score: f64,

    #[arg(long, default_value_t = 0.60)]
    core_support_fraction: f64,

    #[arg(long, default_value_t = 0.35)]
    fringe_support_fraction: f64,

    #[arg(long, default_value_t = false)]
    allow_partial_stage_b: bool,
}

#[derive(Debug, Clone, Serialize)]
struct StageBInput {
    replica_id: usize,
    source_kind: &'static str,
    source_path: String,
    parquet_path: String,
    parquet_sha256: String,
}

#[derive(Debug, Serialize)]
struct CommandRecord {
    tool: String,
    argv: Vec<String>,
    status_code: i32,
}

#[derive(Debug, Serialize)]
struct PipelineManifest {
    schema_kind: &'static str,
    schema_version: &'static str,
    computed_at: String,
    computed_by: &'static str,
    target_id: String,
    n_replicas: usize,
    base_seed: u64,
    inputs: Vec<StageBInput>,
    commands: Vec<CommandRecord>,
    outputs: BTreeMap<&'static str, ArtifactSummary>,
    acceptance: AcceptanceSummary,
}

#[derive(Debug, Serialize)]
struct ArtifactSummary {
    path: String,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct AcceptanceSummary {
    n_residues: u64,
    n_trainable: u64,
    n_uncertain: u64,
    n_bimodal: u64,
    n_outlier_masked: u64,
    min_trainable_fraction: f64,
    observed_trainable_fraction: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;

    let tools = ToolPaths::discover()?;
    let mut commands = Vec::new();
    let mut stage_b_inputs = Vec::new();

    for (idx, manifest) in args.replica_manifests.iter().enumerate() {
        validate_md_manifest(manifest, args.allow_partial_stage_b)?;
        let stage_b_dir = args
            .output_dir
            .join("stage_b")
            .join(format!("replica_{idx:03}"));
        std::fs::create_dir_all(&stage_b_dir)
            .with_context(|| format!("create {}", stage_b_dir.display()))?;

        let argv = vec![
            "--manifest".into(),
            manifest.as_os_str().to_os_string(),
            "--output-dir".into(),
            stage_b_dir.as_os_str().to_os_string(),
            "--no-update-replica-record".into(),
        ];
        commands.push(run_tool("prism-teacher-tags", &tools.teacher_tags, &argv)?);

        let parquet = stage_b_dir.join("teacher_residue_tags.parquet");
        validate_existing_file(&parquet)?;
        stage_b_inputs.push(StageBInput {
            replica_id: stage_b_inputs.len(),
            source_kind: "md_evidence_manifest",
            source_path: manifest.to_string_lossy().into_owned(),
            parquet_sha256: sha256_file(&parquet)?,
            parquet_path: parquet.to_string_lossy().into_owned(),
        });
    }

    for parquet in &args.stage_b_parquets {
        validate_existing_file(parquet)?;
        stage_b_inputs.push(StageBInput {
            replica_id: stage_b_inputs.len(),
            source_kind: "stage_b_parquet",
            source_path: parquet.to_string_lossy().into_owned(),
            parquet_sha256: sha256_file(parquet)?,
            parquet_path: parquet.to_string_lossy().into_owned(),
        });
    }

    if stage_b_inputs.len() < args.min_replicas {
        bail!(
            "only {} Stage-B inputs after materialization; require at least {}",
            stage_b_inputs.len(),
            args.min_replicas
        );
    }

    let ensemble_dir = args.output_dir.join("ensemble");
    let stage_c_dir = args.output_dir.join("stage_c");
    std::fs::create_dir_all(&ensemble_dir)
        .with_context(|| format!("create {}", ensemble_dir.display()))?;
    std::fs::create_dir_all(&stage_c_dir)
        .with_context(|| format!("create {}", stage_c_dir.display()))?;

    let mut ensemble_argv = Vec::<OsString>::new();
    for input in &stage_b_inputs {
        ensemble_argv.push("--feature-parquet".into());
        ensemble_argv.push(input.parquet_path.clone().into());
    }
    ensemble_argv.extend([
        "--target-id".into(),
        args.target_id.clone().into(),
        "--base-seed".into(),
        args.base_seed.to_string().into(),
        "--output-dir".into(),
        ensemble_dir.as_os_str().to_os_string(),
    ]);
    commands.push(run_tool(
        "prism-v004-ensemble",
        &tools.v004_ensemble,
        &ensemble_argv,
    )?);

    let aggregates = ensemble_dir.join("ensemble_aggregates.parquet");
    let consensus = ensemble_dir.join("ensemble_consensus.json");
    validate_existing_file(&aggregates)?;
    validate_existing_file(&consensus)?;

    let finalize_argv = vec![
        "--aggregates".into(),
        aggregates.as_os_str().to_os_string(),
        "--consensus".into(),
        consensus.as_os_str().to_os_string(),
        "--output-dir".into(),
        stage_c_dir.as_os_str().to_os_string(),
        "--strict-rhat".into(),
        args.strict_rhat.to_string().into(),
        "--soft-rhat".into(),
        args.soft_rhat.to_string().into(),
        "--min-replicas".into(),
        args.min_replicas_per_feature.to_string().into(),
        "--max-outlier-features".into(),
        args.max_outlier_features.to_string().into(),
        "--min-phase-score".into(),
        args.min_phase_score.to_string().into(),
        "--core-support-fraction".into(),
        args.core_support_fraction.to_string().into(),
        "--fringe-support-fraction".into(),
        args.fringe_support_fraction.to_string().into(),
    ];
    commands.push(run_tool(
        "prism-teacher-finalize",
        &tools.teacher_finalize,
        &finalize_argv,
    )?);

    let summary_path = stage_c_dir.join("teacher_finalize_summary.json");
    validate_existing_file(&summary_path)?;
    let acceptance = read_acceptance(&summary_path, args.min_trainable_fraction)?;

    let mut outputs = BTreeMap::new();
    let teacher_labels = stage_c_dir.join("teacher_consensus_residue_labels.parquet");
    let student_bundle = stage_c_dir.join("student_label_bundle.parquet");
    let regions = stage_c_dir.join("teacher_consensus_regions.json");
    for (kind, path) in [
        ("ensemble_aggregates", aggregates),
        ("ensemble_consensus", consensus),
        ("teacher_consensus_residue_labels", teacher_labels),
        ("student_label_bundle", student_bundle),
        ("teacher_consensus_regions", regions),
        ("teacher_finalize_summary", summary_path),
    ] {
        validate_existing_file(&path)?;
        outputs.insert(
            kind,
            ArtifactSummary {
                sha256: sha256_file(&path)?,
                path: path.to_string_lossy().into_owned(),
            },
        );
    }

    let manifest = PipelineManifest {
        schema_kind: "prism_twin_teacher_pipeline",
        schema_version: "1.0.0",
        computed_at: Utc::now().to_rfc3339(),
        computed_by: "prism-teacher-pipeline",
        target_id: args.target_id,
        n_replicas: stage_b_inputs.len(),
        base_seed: args.base_seed,
        inputs: stage_b_inputs,
        commands,
        outputs,
        acceptance,
    };
    let pipeline_manifest_path = args
        .output_dir
        .join("teacher_consensus_pipeline_manifest.json");
    std::fs::write(
        &pipeline_manifest_path,
        serde_json::to_vec_pretty(&manifest)?,
    )
    .with_context(|| format!("write {}", pipeline_manifest_path.display()))?;

    println!("wrote: {}", pipeline_manifest_path.display());
    println!(
        "accepted: residues={} trainable={} uncertain={} trainable_fraction={:.3}",
        manifest.acceptance.n_residues,
        manifest.acceptance.n_trainable,
        manifest.acceptance.n_uncertain,
        manifest.acceptance.observed_trainable_fraction
    );
    Ok(())
}

struct ToolPaths {
    teacher_tags: PathBuf,
    v004_ensemble: PathBuf,
    teacher_finalize: PathBuf,
}

impl ToolPaths {
    fn discover() -> Result<Self> {
        Ok(Self {
            teacher_tags: resolve_tool("prism-teacher-tags")?,
            v004_ensemble: resolve_tool("prism-v004-ensemble")?,
            teacher_finalize: resolve_tool("prism-teacher-finalize")?,
        })
    }
}

fn resolve_tool(name: &str) -> Result<PathBuf> {
    let exe = std::env::current_exe().context("resolve current executable")?;
    if let Some(dir) = exe.parent() {
        let sibling = dir.join(name);
        if sibling.is_file() {
            return Ok(sibling);
        }
    }
    for dir in std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect::<Vec<_>>())
        .unwrap_or_default()
    {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!("could not locate required sibling/PATH tool {name}");
}

fn run_tool(name: &str, path: &Path, argv: &[OsString]) -> Result<CommandRecord> {
    let display_argv = argv
        .iter()
        .map(|s| s.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    log::info!("running {} {:?}", path.display(), display_argv);
    let status = Command::new(path)
        .args(argv)
        .status()
        .with_context(|| format!("spawn {}", path.display()))?;
    let code = status.code().unwrap_or(-1);
    if !status.success() {
        bail!("{name} failed with status {code}: {:?}", display_argv);
    }
    Ok(CommandRecord {
        tool: name.to_string(),
        argv: display_argv,
        status_code: code,
    })
}

fn validate_args(args: &Args) -> Result<()> {
    if args.replica_manifests.is_empty() && args.stage_b_parquets.is_empty() {
        bail!("provide at least one --replica-manifest or --stage-b-parquet");
    }
    if !(0.0..=1.0).contains(&args.min_trainable_fraction) {
        bail!("--min-trainable-fraction must be in [0, 1]");
    }
    if args.strict_rhat <= 0.0 || args.soft_rhat <= 1.0 || args.strict_rhat > args.soft_rhat {
        bail!("require 0 < --strict-rhat <= --soft-rhat and --soft-rhat > 1");
    }
    Ok(())
}

fn validate_md_manifest(path: &Path, allow_partial: bool) -> Result<()> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let complete = value
        .get("required_artifacts_complete")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let status = value
        .get("validation_status")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !allow_partial && (!complete || status != "accepted_required_artifacts_present") {
        bail!(
            "{} is not accepted complete MD evidence: status={status:?} required_artifacts_complete={complete}",
            path.display()
        );
    }
    let stream_count = value
        .get("stream_count")
        .and_then(Value::as_u64)
        .context("manifest missing stream_count")?;
    let streams_serialized = value
        .get("streams_serialized")
        .and_then(Value::as_u64)
        .context("manifest missing streams_serialized")?;
    if !allow_partial && streams_serialized != stream_count {
        bail!(
            "{} streams_serialized {} != stream_count {}",
            path.display(),
            streams_serialized,
            stream_count
        );
    }
    validate_manifest_artifacts(&value, stream_count, allow_partial)
        .with_context(|| format!("validate artifacts in {}", path.display()))
}

fn validate_manifest_artifacts(
    value: &Value,
    stream_count: u64,
    allow_partial: bool,
) -> Result<()> {
    let artifacts = value
        .get("artifacts")
        .and_then(Value::as_array)
        .context("manifest missing artifacts[]")?;
    for kind in ["spikes", "signal_grid", "kcc_v2full"] {
        let present = artifacts
            .iter()
            .filter(|a| {
                a.get("kind").and_then(Value::as_str) == Some(kind)
                    && a.get("present").and_then(Value::as_bool).unwrap_or(false)
                    && a.get("size_bytes").and_then(Value::as_u64).unwrap_or(0) > 0
            })
            .count() as u64;
        if !allow_partial && present != stream_count {
            bail!(
                "{kind} present streams {} != expected {}",
                present,
                stream_count
            );
        }
    }
    Ok(())
}

fn validate_existing_file(path: &Path) -> Result<()> {
    let meta = path
        .metadata()
        .with_context(|| format!("missing file {}", path.display()))?;
    if !meta.is_file() || meta.len() == 0 {
        bail!("{} is not a non-empty file", path.display());
    }
    Ok(())
}

fn read_acceptance(path: &Path, min_trainable_fraction: f64) -> Result<AcceptanceSummary> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let read_u64 = |name: &str| -> Result<u64> {
        value
            .get(name)
            .and_then(Value::as_u64)
            .with_context(|| format!("teacher summary missing {name}"))
    };
    let n_residues = read_u64("n_residues")?;
    let n_trainable = read_u64("n_trainable")?;
    let observed_trainable_fraction = if n_residues == 0 {
        0.0
    } else {
        n_trainable as f64 / n_residues as f64
    };
    if observed_trainable_fraction < min_trainable_fraction {
        bail!(
            "trainable fraction {:.3} < required {:.3}",
            observed_trainable_fraction,
            min_trainable_fraction
        );
    }
    Ok(AcceptanceSummary {
        n_residues,
        n_trainable,
        n_uncertain: read_u64("n_uncertain")?,
        n_bimodal: read_u64("n_bimodal")?,
        n_outlier_masked: read_u64("n_outlier_masked")?,
        min_trainable_fraction,
        observed_trainable_fraction,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_artifact_validation_requires_required_streams() {
        let value = serde_json::json!({
            "artifacts": [
                {"kind": "spikes", "present": true, "size_bytes": 10},
                {"kind": "signal_grid", "present": true, "size_bytes": 10},
                {"kind": "kcc_v2full", "present": true, "size_bytes": 10}
            ]
        });
        validate_manifest_artifacts(&value, 1, false).unwrap();
        let err = validate_manifest_artifacts(&value, 2, false).unwrap_err();
        assert!(err.to_string().contains("present streams 1 != expected 2"));
    }

    #[test]
    fn acceptance_rejects_too_few_trainable_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("summary.json");
        std::fs::write(
            &path,
            r#"{
                "n_residues": 100,
                "n_trainable": 4,
                "n_uncertain": 96,
                "n_bimodal": 0,
                "n_outlier_masked": 0
            }"#,
        )
        .unwrap();
        assert!(read_acceptance(&path, 0.05).is_err());
        assert_eq!(read_acceptance(&path, 0.04).unwrap().n_trainable, 4);
    }
}
