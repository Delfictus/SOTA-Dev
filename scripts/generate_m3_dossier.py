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
TRACK_B = CAMPAIGN_DIR / "track_b_chronological"
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
    parser.add_argument(
        "--motif-registry",
        type=Path,
        default=TRACK_B / "motif_intelligence/thermodynamic_motif_registry.parquet",
    )
    parser.add_argument("--competitor-scaffold-manifest", type=Path, default=TRACK_A / "competitor_scaffold_o3a_manifest.json")
    parser.add_argument("--phase2d-manifest", type=Path, default=CAMPAIGN_DIR / "phase_2d_variant_grid_manifest.json")
    parser.add_argument("--cbom", type=Path, default=CAMPAIGN_DIR / "PRISM_CBOM_v1.0.json")
    parser.add_argument("--tripartite", action="store_true", default=False)
    parser.add_argument("--lock-positive-count", action="store_true", default=False)
    parser.add_argument("--cross-scaffold", action="store_true", default=False)
    parser.add_argument("--pgx-resilience", action="store_true", default=False)
    parser.add_argument("--bald-ranking", action="store_true", default=False)
    parser.add_argument("--full-field-stack", action="store_true", default=False)
    parser.add_argument("--species-selectivity", action="store_true", default=False)
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


def full_field_context(top100_path: Path) -> dict[str, object]:
    context: dict[str, object] = {
        "full_field_status": "unavailable",
        "full_field_candidate_count": 0,
        "full_field_shear_mean": None,
        "full_field_shear_positive_count": 0,
        "full_field_hysteresis_mean": None,
        "full_field_reversibility_mean": None,
        "full_field_pathway_contact_count": 0,
        "full_field_pathway_neighborhood_count": 0,
        "full_field_charge_abs_mean": None,
        "full_field_u_pose_mean": None,
        "full_field_u_pose_sources": "{}",
        "full_field_low_shear_reversible_pathway": 0,
        "species_selectivity_status": "unavailable",
        "species_selectivity_mean": None,
        "species_human_selective_count": 0,
        "species_broad_count": 0,
        "species_rows": [],
    }
    if not top100_path.is_file():
        return context
    frame = pl.read_parquet(top100_path)
    context["full_field_candidate_count"] = frame.height

    def maybe_series(column: str) -> pl.Series | None:
        if column not in frame.columns:
            return None
        return frame.get_column(column).cast(pl.Float64, strict=False)

    shear = maybe_series("sigma_shear_mean")
    hysteresis = maybe_series("hysteresis_mean")
    reversibility = maybe_series("reversibility_mean")
    pathway = maybe_series("pathway_voxels_occupied")
    pathway_neighborhood = maybe_series("pathway_neighborhood_contacts")
    charge = maybe_series("charge_feature_mean")
    u_pose = maybe_series("u_pose")
    if all(series is not None for series in [shear, hysteresis, reversibility, pathway, charge, u_pose]):
        assert shear is not None
        assert hysteresis is not None
        assert reversibility is not None
        assert pathway is not None
        assert charge is not None
        assert u_pose is not None
        context.update(
            {
                "full_field_status": "complete",
                "full_field_shear_mean": numeric(shear.mean()),
                "full_field_shear_positive_count": int(frame.filter(shear > 0).height),
                "full_field_hysteresis_mean": numeric(hysteresis.mean()),
                "full_field_reversibility_mean": numeric(reversibility.mean()),
                "full_field_pathway_contact_count": int(frame.filter(pathway > 0).height),
                "full_field_pathway_neighborhood_count": (
                    int(frame.filter(pathway_neighborhood > 0).height)
                    if pathway_neighborhood is not None
                    else 0
                ),
                "full_field_charge_abs_mean": numeric(charge.abs().mean()),
                "full_field_u_pose_mean": numeric(u_pose.mean()),
                "full_field_u_pose_sources": (
                    json.dumps(
                        {
                            str(row["u_pose_source"]): int(row["count"])
                            for row in frame.get_column("u_pose_source").value_counts().to_dicts()
                        },
                        sort_keys=True,
                    )
                    if "u_pose_source" in frame.columns
                    else "{}"
                ),
                "full_field_low_shear_reversible_pathway": int(
                    frame.with_columns(
                        shear.alias("_shear"),
                        hysteresis.alias("_hysteresis"),
                        reversibility.alias("_reversibility"),
                        pathway.alias("_pathway"),
                        (pathway_neighborhood if pathway_neighborhood is not None else pl.Series("_pathway_neighborhood", [0.0] * frame.height)).alias("_pathway_neighborhood"),
                    )
                    .filter(
                        (pl.col("_shear") <= 5.0)
                        & (pl.col("_hysteresis") <= 0.3)
                        & (pl.col("_reversibility") >= 0.7)
                        & ((pl.col("_pathway") > 0) | (pl.col("_pathway_neighborhood") > 0))
                    )
                    .height
                ),
            }
        )

    species = maybe_series("species_selectivity_score")
    if species is not None:
        species_rows = []
        rows = frame.sort("species_selectivity_score", descending=True).head(5).to_dicts()
        for row in rows:
            species_rows.append(
                {
                    "rank": integer(row.get("rank"), 0),
                    "smiles": str(row.get("canonical_smiles", row.get("smiles", ""))),
                    "score": numeric(row.get("species_selectivity_score")),
                    "predicted_active_in": str(row.get("predicted_active_in", "")),
                }
            )
        context.update(
            {
                "species_selectivity_status": "complete",
                "species_selectivity_mean": numeric(species.mean()),
                "species_human_selective_count": int(frame.filter(species >= 0.7).height),
                "species_broad_count": int(frame.filter(species <= 0.35).height),
                "species_rows": species_rows,
            }
        )
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


def motif_context(path: Path) -> dict[str, object]:
    context: dict[str, object] = {
        "motif_status": "unavailable",
        "motif_registry_path": path.as_posix(),
        "motif_count": 0,
        "motif_lock_wedge_count": 0,
        "motif_phase_conditional_count": 0,
        "motif_evolutionary_invariant_count": 0,
        "motif_came_count": 0,
        "motif_tfgd_count": 0,
        "motif_pr_mcs_count": 0,
        "motif_sad_count": 0,
        "motif_completeness_mean": None,
        "motif_top_lock_rows": [],
        "motif_synthon_rows": [],
        "motif_evolution_rows": [],
    }
    if not path.is_file():
        return context
    frame = pl.read_parquet(path)
    context["motif_status"] = "complete"
    context["motif_count"] = frame.height
    if frame.height == 0:
        return context
    if "thermodynamic_role" in frame.columns:
        context["motif_lock_wedge_count"] = frame.filter(pl.col("thermodynamic_role") == "LOCK_WEDGE").height
    if "discovery_method" in frame.columns:
        context["motif_came_count"] = frame.filter(pl.col("discovery_method") == "CAME").height
        context["motif_tfgd_count"] = frame.filter(pl.col("discovery_method") == "TFGD").height
        context["motif_pr_mcs_count"] = frame.filter(pl.col("discovery_method") == "PR_MCS").height
        context["motif_sad_count"] = frame.filter(pl.col("discovery_method") == "SAD").height
    if "phase_profile" in frame.columns:
        phase_conditional = 0
        for raw in frame.get_column("phase_profile").to_list():
            values = parse_float_list(raw)
            if len(values) == 5 and max(values) - min(values) > 1.0e-6:
                phase_conditional += 1
        context["motif_phase_conditional_count"] = phase_conditional
    if "is_evolutionary_invariant" in frame.columns:
        context["motif_evolutionary_invariant_count"] = frame.filter(pl.col("is_evolutionary_invariant") == True).height
    if "completeness_score" in frame.columns:
        context["motif_completeness_mean"] = numeric(frame.get_column("completeness_score").mean())
    lock_frame = frame
    if "thermodynamic_role" in frame.columns:
        lock_frame = frame.filter(pl.col("thermodynamic_role") == "LOCK_WEDGE")
    if "lock_geometry_contribution" in lock_frame.columns:
        lock_frame = (
            lock_frame.with_columns(pl.col("lock_geometry_contribution").fill_null(0.0).alias("_lock_sort"))
            .sort("_lock_sort", descending=True)
            .drop("_lock_sort")
        )
    top_lock_rows = []
    for row in lock_frame.head(6).to_dicts():
        top_lock_rows.append(
            {
                "motif_id": str(row.get("motif_id", "")),
                "smarts": str(row.get("canonical_smarts", "")),
                "role": str(row.get("thermodynamic_role", "")),
                "lock": numeric(row.get("lock_geometry_contribution")),
                "resilience": numeric(row.get("consensus_resilience")),
                "method": str(row.get("discovery_method", "")),
                "provenance": str(row.get("provenance", "")),
            }
        )
    context["motif_top_lock_rows"] = top_lock_rows
    synthon_rows = []
    if "synthon_sources" in frame.columns:
        sad_frame = frame.filter(pl.col("discovery_method") == "SAD") if "discovery_method" in frame.columns else frame
        if "enrichment_ratio" in sad_frame.columns:
            sad_frame = sad_frame.sort("enrichment_ratio", descending=True)
        for row in sad_frame.head(8).to_dicts():
            sources = parse_string_list(row.get("synthon_sources"))
            prefs = parse_json_object(row.get("exit_vector_preference"))
            synthon_rows.append(
                {
                    "enamine_id": sources[0] if sources else "",
                    "smarts": str(row.get("canonical_smarts", "")),
                    "enrichment": numeric(row.get("enrichment_ratio"), 1.0),
                    "preferred_exit_vector": ",".join(sorted(prefs.keys())) if prefs else "unreported",
                    "priority": "HIGH" if numeric(row.get("enrichment_ratio"), 1.0) >= 1.5 else "MEDIUM",
                }
            )
    context["motif_synthon_rows"] = synthon_rows
    context["motif_evolution_rows"] = [
        {
            "motif_id": str(row.get("motif_id", "")),
            "born": integer(row.get("first_seen_epoch")),
            "status": "PERSISTENT",
            "parent": "",
            "lock_delta": numeric(row.get("lock_geometry_contribution")),
            "lineage": "NOVEL",
        }
        for row in lock_frame.head(5).to_dicts()
    ]
    return context


def parse_float_list(value: object) -> list[float]:
    raw = json.loads(value) if isinstance(value, str) else value
    if not isinstance(raw, list):
        return []
    output: list[float] = []
    for item in raw:
        if isinstance(item, bool) or item is None:
            continue
        if isinstance(item, int | float | str):
            output.append(numeric(item))
    return output


def parse_string_list(value: object) -> list[str]:
    raw = json.loads(value) if isinstance(value, str) else value
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def parse_json_object(value: object) -> dict[str, object]:
    raw = json.loads(value) if isinstance(value, str) else value
    return cast(dict[str, object], raw) if isinstance(raw, dict) else {}


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
    status = str(report.get("validation_status", report.get("status", "unknown")))
    completed_epochs = integer(report.get("completed_epochs", report.get("completed_epochs_observed")))
    target_epochs = integer(report.get("target_epochs"), completed_epochs if completed_epochs else 500)
    failure_mode = str(report.get("failure_mode", "none" if status == "PASS" else "unreported"))
    return {
        "epoch016_training_status": status,
        "epoch016_completed_epochs": completed_epochs,
        "epoch016_target_epochs": target_epochs,
        "epoch016_failure_mode": failure_mode,
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
    context.update(full_field_context(args.top100))
    context.update(motif_context(args.motif_registry))
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
