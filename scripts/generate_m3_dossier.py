#!/usr/bin/env python3
"""Render the M3 lead-optimization dossier from Track A audit artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias, cast

import polars as pl
from jinja2 import Environment, FileSystemLoader, StrictUndefined


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN_DIR / "track_a_generative"
TEMPLATE = REPO_ROOT / "00_registry/templates/m3_lead_optimization_dossier.md.j2"
OUTPUT = CAMPAIGN_DIR / "M3_Lead_Optimization_Dossier.md"

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
Row: TypeAlias = dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--top100", type=Path, default=TRACK_A / "gflownet_top_100_candidates_lockmask_rescored.parquet")
    parser.add_argument("--candidates", type=Path, default=None, help="Alias for --top100 used by Epoch 016.")
    parser.add_argument("--tripartite-profiles", type=Path, default=TRACK_A / "gflownet_top_50_tripartite_profiles.parquet")
    parser.add_argument("--medchem-audit", type=Path, default=TRACK_A / "gflownet_medchem_audit.parquet")
    parser.add_argument("--candidate-audit", type=Path, default=TRACK_A / "gflownet_candidate_audit.json")
    parser.add_argument("--training-report", type=Path, default=TRACK_A / "epoch016_execution_report.json")
    parser.add_argument("--pgx-report", type=Path, default=TRACK_A / "gflownet_top_100_pgx_screened_report.json")
    parser.add_argument("--parity-report", type=Path, default=TRACK_A / "wt_projection_parity_report.json")
    parser.add_argument("--infra-report", type=Path, default=TRACK_A / "autonomous_infra_status_epoch017.json")
    parser.add_argument("--population-pgx-report", type=Path, default=CAMPAIGN_DIR / "pgx_full_landscape_report.json")
    parser.add_argument("--variant-manifest", type=Path, default=TRACK_A / "population_pgx/variant_perturbation_manifest.json")
    parser.add_argument("--consensus-report", type=Path, default=TRACK_A / "population_pgx/population_consensus_grid_report.json")
    parser.add_argument("--gpu-dispatch-report", type=Path, default=TRACK_A / "gpu_dispatch_audit_report.json")
    parser.add_argument("--plan", type=Path, default=TRACK_A / "vspace_38b_dendritic_plan.json")
    parser.add_argument("--competitor-scaffold-manifest", type=Path, default=TRACK_A / "competitor_scaffold_o3a_manifest.json")
    parser.add_argument("--phase2d-manifest", type=Path, default=CAMPAIGN_DIR / "phase_2d_variant_grid_manifest.json")
    parser.add_argument("--cbom", type=Path, default=CAMPAIGN_DIR / "PRISM_CBOM_v1.0.json")
    parser.add_argument("--tripartite", action="store_true", default=False)
    parser.add_argument("--lock-positive-count", action="store_true", default=False)
    parser.add_argument("--cross-scaffold", action="store_true", default=False)
    parser.add_argument("--pgx-resilience", action="store_true", default=False)
    parser.add_argument("--bald-ranking", action="store_true", default=False)
    return parser.parse_args()


def load_json(path: Path) -> JsonObject:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return cast(JsonObject, loaded)


def numeric(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int | float | str):
        return float(value)
    return default


def integer(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int | float | str):
        return int(float(value))
    return default


def top_rows(path: Path) -> list[Row]:
    frame = pl.read_parquet(path)
    if "lock_geometry_score" not in frame.columns:
        frame = frame.with_columns(pl.col("pi_clash_lock").alias("lock_geometry_score"))
    if "bias_projection_score" not in frame.columns:
        frame = frame.with_columns(pl.lit(0.0).alias("bias_projection_score"))
    if "epistemic_confidence" not in frame.columns:
        frame = frame.with_columns(pl.lit("L1").alias("epistemic_confidence"))
    frame = (
        frame.sort(["reward", "lock_geometry_score"], descending=[True, True])
        .head(10)
        .with_columns(
            pl.format("SMARTS/Z-matrix projected route via {}", pl.col("anchor_id")).alias("synthetic_route")
            if "anchor_id" in frame.columns
            else pl.lit("SMARTS/Z-matrix projected route").alias("synthetic_route")
        )
    )
    return cast(list[Row], frame.to_dicts())


def medchem_counts(path: Path) -> dict[str, int]:
    frame = pl.scan_parquet(path).collect()
    return {
        "audited_candidates": frame.height,
        "biased_agonism_confirmed": frame.filter(pl.col("biased_agonism_confirmed")).height,
        "pains_pass": frame.filter(pl.col("pains_pass")).height,
        "brenk_pass": frame.filter(pl.col("brenk_pass")).height,
        "oral_pass": frame.filter(pl.col("oral_pass")).height,
    }


def plan_context(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        return {
            "nominal_design_space": "38B nominal REAL-space target",
            "total_valid_pairs": 0,
            "estimated_rotamers": 0,
            "shard_count": 0,
        }
    plan = load_json(path)
    return {
        "nominal_design_space": "38B nominal REAL-space target",
        "total_valid_pairs": integer(plan.get("total_valid_pairs")),
        "estimated_rotamers": integer(plan.get("estimated_rotamers")),
        "shard_count": integer(plan.get("shard_count")),
    }


def scaffold_context(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "scaffold_pool_count": 1,
            "scaffold_pool_names": ["Aleniglipron"],
            "scaffold_manifest_path": path.as_posix(),
            "competitor_o3a_outputs": [],
        }
    manifest = load_json(path)
    outputs = cast(list[JsonObject], manifest.get("outputs", []))
    names = ["Aleniglipron"]
    for item in outputs:
        compound_name = str(item.get("compound_name", "")).split(" (")[0]
        if compound_name and compound_name not in names:
            names.append(compound_name)
    return {
        "scaffold_pool_count": len(names),
        "scaffold_pool_names": names,
        "scaffold_manifest_path": path.as_posix(),
        "competitor_o3a_outputs": outputs,
    }


def variant_grid_context(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "pan_variant_grid_count": 0,
            "queued_variant_count": 0,
            "high_risk_variants": [],
            "phase2d_manifest_path": path.as_posix(),
        }
    manifest = load_json(path)
    strategy = cast(JsonObject, manifest.get("phase_2d_expansion_strategy", {}))
    targets = cast(list[JsonObject], strategy.get("staged_biological_targets", []))
    queued = [target for target in targets if target.get("status") == "queued_for_md_simulation"]
    high_risk = [
        str(target.get("variant"))
        for target in targets
        if target.get("risk_class") == "HIGH RISK" and target.get("status") == "queued_for_md_simulation"
    ]
    return {
        "pan_variant_grid_count": integer(strategy.get("pan_variant_grid_count"), len(queued)),
        "queued_variant_count": len(queued),
        "high_risk_variants": high_risk,
        "phase2d_manifest_path": path.as_posix(),
    }


def tripartite_context(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "tripartite_profile_count": 0,
            "tripartite_lock_positive": 0,
            "tripartite_l1": 0,
            "tripartite_l2": 0,
            "tripartite_l3": 0,
            "tripartite_profile_path": path.as_posix(),
        }
    frame = pl.read_parquet(path)
    if "epistemic_confidence" not in frame.columns:
        return {
            "tripartite_profile_count": frame.height,
            "tripartite_lock_positive": 0,
            "tripartite_l1": frame.height,
            "tripartite_l2": 0,
            "tripartite_l3": 0,
            "tripartite_profile_path": path.as_posix(),
        }
    return {
        "tripartite_profile_count": frame.height,
        "tripartite_lock_positive": frame.filter(pl.col("lock_geometry_score") > 0.0).height
        if "lock_geometry_score" in frame.columns
        else 0,
        "tripartite_l1": frame.filter(pl.col("epistemic_confidence") == "L1").height,
        "tripartite_l2": frame.filter(pl.col("epistemic_confidence") == "L2").height,
        "tripartite_l3": frame.filter(pl.col("epistemic_confidence") == "L3").height,
        "tripartite_profile_path": path.as_posix(),
    }


def pgx_context(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "pgx_status": "deferred",
            "pgx_report_path": path.as_posix(),
            "pgx_variants": [],
            "pgx_worst_case_mean": None,
            "pgx_immune_or_tolerant": 0,
            "pgx_epistemic_class": "UNAVAILABLE",
            "pgx_scoring_method": "unavailable",
            "pgx_wt_parity_status": "unavailable",
        }
    report = load_json(path)
    variants = []
    raw_variants = report.get("variants", {})
    if isinstance(raw_variants, dict):
        for name, payload in raw_variants.items():
            if isinstance(payload, dict):
                variants.append(
                    {
                        "name": str(name),
                        "condition_id": str(payload.get("condition_id", "")),
                        "lock_preserved_count": integer(payload.get("lock_preserved_count")),
                        "classification_counts": payload.get("classification_counts", {}),
                    }
                )
    return {
        "pgx_status": str(report.get("diagnostic_status", "complete")),
        "pgx_report_path": path.as_posix(),
        "pgx_variants": variants,
        "pgx_worst_case_mean": report.get("worst_case_mean"),
        "pgx_immune_or_tolerant": integer(report.get("immune_or_tolerant_worst_case")),
        "pgx_epistemic_class": str(report.get("epistemic_class", "PROJECTED")),
        "pgx_scoring_method": str(report.get("scoring_method", "unknown")),
        "pgx_wt_parity_status": str(report.get("wt_parity_status", "unreported")),
    }


def parity_context(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "parity_status": "unavailable",
            "parity_report_path": path.as_posix(),
            "parity_repair_method": "unavailable",
            "parity_raw_projection_status": "unavailable",
            "parity_projection_native_ratio_mean": None,
            "parity_calibrated_ratio_mean": None,
        }
    report = load_json(path)
    return {
        "parity_status": str(report.get("wt_parity_status", "unknown")),
        "parity_report_path": path.as_posix(),
        "parity_repair_method": str(report.get("repair_method", "unknown")),
        "parity_raw_projection_status": str(report.get("raw_projection_status", "unknown")),
        "parity_projection_native_ratio_mean": report.get("projection_vs_native_ratio_mean"),
        "parity_calibrated_ratio_mean": report.get("calibrated_wt_self_parity_ratio_mean"),
    }


def infra_context(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "infra_status": "unavailable",
            "infra_report_path": path.as_posix(),
            "infra_worker_status": "unavailable",
            "infra_d1_status": "unavailable",
            "infra_vectorize_status": "unavailable",
            "infra_r2_status": "unavailable",
            "infra_queue_status": "unavailable",
        }
    report = load_json(path)
    raw_checks = report.get("checks", [])
    checks = cast(list[JsonObject], raw_checks if isinstance(raw_checks, list) else [])

    def check_status(name: str) -> str:
        for item in checks:
            if item.get("name") == name:
                return str(item.get("status", "UNKNOWN"))
        return "unreported"

    return {
        "infra_status": str(report.get("overall_status", "unknown")),
        "infra_report_path": path.as_posix(),
        "infra_worker_status": check_status("worker_http"),
        "infra_d1_status": check_status("d1_candidate_count"),
        "infra_vectorize_status": check_status("vectorize_info"),
        "infra_r2_status": check_status("r2_bucket_list")
        if check_status("r2_bucket_list") != "unreported"
        else check_status("r2_object_list"),
        "infra_queue_status": check_status("queue_list"),
    }


def population_pgx_context(report_path: Path, manifest_path: Path, consensus_path: Path) -> dict[str, object]:
    context: dict[str, object] = {
        "population_pgx_status": "unavailable",
        "population_pgx_report_path": report_path.as_posix(),
        "population_variant_count": 0,
        "population_tier1_count": 0,
        "population_tier2_count": 0,
        "population_tier3_count": 0,
        "population_tier1_worst_mean": None,
        "population_tier1_ge085_count": 0,
        "population_coverage_mean": None,
        "population_ancestry_rows": [],
        "population_tier1_rows": [],
        "population_consensus_status": "unavailable",
        "population_consensus_activated": 0,
        "population_consensus_report_path": consensus_path.as_posix(),
        "population_variant_manifest_path": manifest_path.as_posix(),
    }
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        variants = cast(list[JsonObject], manifest.get("variants", []))
        context["population_variant_count"] = len(variants)
        context["population_tier1_count"] = len([variant for variant in variants if integer(variant.get("tier")) == 1])
        context["population_tier2_count"] = len([variant for variant in variants if integer(variant.get("tier")) == 2])
        context["population_tier3_count"] = len([variant for variant in variants if integer(variant.get("tier")) == 3])
        tier1_rows = []
        for variant in variants:
            if integer(variant.get("tier")) == 1:
                tier1_rows.append(
                    {
                        "mutation": str(variant.get("mutation", "")),
                        "maf": numeric(variant.get("maf_global")),
                        "domain": str(variant.get("domain", "")),
                        "confidence": str(variant.get("epistemic_confidence", "")),
                        "provenance": str(variant.get("provenance", "")),
                    }
                )
        context["population_tier1_rows"] = tier1_rows
    if consensus_path.is_file():
        consensus = load_json(consensus_path)
        global_context = cast(JsonObject, consensus.get("global", {}))
        context["population_consensus_status"] = str(consensus.get("status", "unknown"))
        context["population_consensus_activated"] = integer(global_context.get("thermally_activated_voxels"))
    if report_path.is_file():
        report = load_json(report_path)
        context["population_pgx_status"] = str(report.get("status", "unknown"))
        context["population_tier1_worst_mean"] = report.get("pgx_tier1_worst_case_mean")
        context["population_tier1_ge085_count"] = integer(report.get("pgx_tier1_worst_case_ge_085_count"))
        context["population_coverage_mean"] = report.get("population_coverage_pct_mean")
        ancestry_rows = []
        raw_ancestry = report.get("ancestry", {})
        if isinstance(raw_ancestry, dict):
            for ancestry, payload in raw_ancestry.items():
                if isinstance(payload, dict):
                    ancestry_rows.append(
                        {
                            "ancestry": str(ancestry),
                            "mean_resilience": payload.get("mean_resilience"),
                            "coverage_ge_085": integer(payload.get("coverage_ge_085")),
                        }
                    )
        context["population_ancestry_rows"] = ancestry_rows
    return context


def gpu_dispatch_context(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "gpu_dispatch_status": "deferred",
            "gpu_dispatch_report_path": path.as_posix(),
            "gpu_dispatch_count": 0,
            "gpu_dispatch_corrected_count": 0,
            "gpu_dispatch_high_priority_count": 0,
            "gpu_dispatch_ready_count": 0,
        }
    report = load_json(path)
    dispatch_count = integer(report.get("dispatch_count"))
    dispatches = report.get("dispatches", [])
    if dispatch_count == 0 and isinstance(dispatches, list):
        dispatch_count = len(dispatches)
    return {
        "gpu_dispatch_status": str(report.get("status", "complete")),
        "gpu_dispatch_report_path": path.as_posix(),
        "gpu_dispatch_count": dispatch_count,
        "gpu_dispatch_corrected_count": integer(report.get("corrected_script_count")),
        "gpu_dispatch_high_priority_count": integer(report.get("high_priority_count")),
        "gpu_dispatch_ready_count": integer(report.get("dispatch_ready_count"), dispatch_count),
    }


def training_context(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "epoch016_training_status": "not_run",
            "epoch016_completed_epochs": 0,
            "epoch016_target_epochs": 500,
            "epoch016_failure_mode": "unreported",
            "epoch016_report_path": path.as_posix(),
        }
    report = load_json(path)
    return {
        "epoch016_training_status": str(report.get("status", "unknown")),
        "epoch016_completed_epochs": integer(report.get("completed_epochs_observed")),
        "epoch016_target_epochs": integer(report.get("target_epochs"), 500),
        "epoch016_failure_mode": str(report.get("failure_mode", "none")),
        "epoch016_report_path": path.as_posix(),
    }


def render(args: argparse.Namespace) -> str:
    if args.candidates is not None:
        args.top100 = args.candidates
    counts = medchem_counts(args.medchem_audit)
    if args.top100.is_file():
        top100_frame = pl.read_parquet(args.top100)
        if "lock_geometry_score" in top100_frame.columns:
            counts["biased_agonism_confirmed"] = top100_frame.filter(pl.col("lock_geometry_score") > 0.5).height
        elif "pi_clash_lock" in top100_frame.columns:
            counts["biased_agonism_confirmed"] = top100_frame.filter(pl.col("pi_clash_lock") > 0.5).height
    top10_source = args.tripartite_profiles if args.tripartite_profiles.is_file() else args.top100
    top10 = top_rows(top10_source)
    top100_unique = (
        pl.scan_parquet(args.top100)
        .select(pl.col("canonical_smiles").n_unique().alias("n_unique"))
        .collect()
        .item()
    )
    context: dict[str, Any] = {
        "campaign_id": "glp1r_aleniglipron",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "top10": top10,
        "top100_unique_smiles": int(top100_unique),
        "lock_threshold": 0.5,
        "completed_shards": "0, 1 local validation; additional shard execution blocked pending R2 streaming",
        "policy_path": (TRACK_A / "gflownet_policy_v1.pt").as_posix(),
        "top100_path": args.top100.as_posix(),
        "medchem_audit_path": args.medchem_audit.as_posix(),
        "audit_json_path": args.candidate_audit.as_posix(),
        "cbom_path": args.cbom.as_posix(),
    }
    context.update(counts)
    context.update(plan_context(args.plan))
    context.update(scaffold_context(args.competitor_scaffold_manifest))
    context.update(variant_grid_context(args.phase2d_manifest))
    context.update(tripartite_context(args.tripartite_profiles))
    context.update(training_context(args.training_report))
    context.update(pgx_context(args.pgx_report))
    context.update(parity_context(args.parity_report))
    context.update(infra_context(args.infra_report))
    context.update(population_pgx_context(args.population_pgx_report, args.variant_manifest, args.consensus_report))
    context.update(gpu_dispatch_context(args.gpu_dispatch_report))
    env = Environment(
        loader=FileSystemLoader(str(args.template.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(args.template.name)
    return str(template.render(**context))


def main() -> int:
    args = parse_args()
    rendered = render(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"m3_dossier_written path={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
