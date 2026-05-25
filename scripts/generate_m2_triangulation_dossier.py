#!/usr/bin/env python3
"""Render the M2 Phase 2E triangulation dossier from downstream decision artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypeAlias, cast

import polars as pl
from jinja2 import Environment, FileSystemLoader, StrictUndefined


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
TEMPLATE_DIR = REPO_ROOT / "00_registry/templates"
DEFAULT_TEMPLATE = TEMPLATE_DIR / "m2_triangulation_dossier.md.j2"
DEFAULT_OUTPUT = CAMPAIGN_DIR / "M2_Triangulation_Dossier_Final.md"

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
Row: TypeAlias = dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "M2_Replayability_Manifest.json")
    parser.add_argument("--assay-routing", type=Path, default=N80_DIR / "assay_routing_recommendations.parquet")
    parser.add_argument("--chronology", type=Path, default=N80_DIR / "probabilistic_break_clusters.parquet")
    parser.add_argument("--tri-state-fibers", type=Path, default=N80_DIR / "tri_state_ligand_fiber_graph.parquet")
    parser.add_argument("--sar-register", type=Path, default=CAMPAIGN_DIR / "track_0_manual_emulation/sar_contingency_register.parquet")
    parser.add_argument("--risk-map", type=Path, default=N80_DIR / "receptor_durability_risk_map.parquet")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> JsonObject:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not decode to an object")
    return cast(JsonObject, loaded)


def row_dicts(frame: pl.DataFrame) -> list[Row]:
    return cast(list[Row], frame.to_dicts())


def as_object_dict(value: JsonValue, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def environment_rows(manifest: JsonObject) -> list[Row]:
    env = as_object_dict(manifest["environment"], "environment")
    return [{"name": key, "value": str(value)} for key, value in sorted(env.items())]


def schema_rows(manifest: JsonObject) -> list[Row]:
    hashes = as_object_dict(manifest["schema_hashes"], "schema_hashes")
    return [{"path": key, "sha256": str(value)} for key, value in sorted(hashes.items())]


def chronology_summary(path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .group_by("cluster_id")
        .agg(
            [
                pl.col("cluster_member_count").first(),
                pl.col("observed_cluster_centroid_md_step").first(),
                pl.col("cluster_confidence").first(),
                pl.col("temporal_overlap_entropy").first(),
                pl.col("inter_replicate_stability").first(),
            ]
        )
        .sort("cluster_id")
        .collect()
    )


def fiber_counts(path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .group_by("fiber_type")
        .agg(pl.len().alias("edge_count"))
        .sort("fiber_type")
        .collect()
    )


def top_fiber(path: Path) -> Row:
    row = (
        pl.scan_parquet(path)
        .sort(["edge_occupancy_fatigue_index", "ligand_interference_load"], descending=[True, True])
        .head(1)
        .collect()
        .to_dicts()[0]
    )
    fragment = str(row.get("dominant_fragment", "FRAG-NA"))
    row["dominant_fragment_suffix"] = fragment.replace("FRAG-", "")
    return cast(Row, row)


def negative_evidence(path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("edge_from_residue").cast(pl.String),
                        pl.lit("->"),
                        pl.col("edge_to_residue").cast(pl.String),
                    ]
                ).alias("edge_id"),
                pl.when(pl.col("mechanically_pruned"))
                .then(pl.lit("mechanically_pruned"))
                .when(pl.col("validation_status") == "divergent_artifact_warning")
                .then(pl.lit("divergent_artifact_warning"))
                .otherwise(pl.lit("not_rejected"))
                .alias("rejection_basis"),
            ]
        )
        .filter(pl.col("rejection_basis") != "not_rejected")
        .select(
            [
                "condition_id",
                "edge_id",
                "edge_class",
                "rejection_basis",
                "durability_risk_score_raw",
                "validation_status",
            ]
        )
        .sort("durability_risk_score_raw", descending=True)
        .head(20)
        .collect()
    )


def normalized_sar_select(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.select(
        [
            pl.col("sar_rule_id").alias("rule_id"),
            pl.col("growth_vector_id").alias("vector_id"),
            pl.col("action_class").alias("classification"),
            pl.col("condition_family"),
            pl.col("condition_id"),
            pl.col("edge_label"),
            pl.col("cooperative_cluster_id"),
            pl.col("cluster_confidence"),
            pl.concat_str([pl.col("ligand_atom_symbol"), pl.col("ligand_atom_idx").cast(pl.String)]).alias("atom_label"),
            pl.col("ray_id"),
            pl.col("atom_edge_u_pose"),
            pl.col("epistemic_u_pose_threshold"),
            pl.col("pi_complement_integral"),
            pl.col("pi_clash_integral"),
            pl.col("shear_stress_normalized_integral"),
            pl.col("vector_score"),
            pl.col("variant_family_support_count"),
            pl.col("universal_growth_vector"),
            pl.col("medchem_directive").alias("rationale"),
        ]
    )


def sar_context(path: Path) -> dict[str, list[Row]]:
    source = pl.scan_parquet(path)
    universal = (
        normalized_sar_select(source.filter(pl.col("universal_growth_vector")))
        .sort("vector_score", descending=True)
        .head(20)
        .collect()
    )
    mirage = (
        normalized_sar_select(source.filter(pl.col("action_class") == "epistemic_mirage"))
        .sort("atom_edge_u_pose", descending=True)
        .head(20)
        .collect()
    )
    prohibited = (
        normalized_sar_select(source.filter(pl.col("action_class").is_in(["prohibited_rigidification_zone", "shear_fracture_vector"])))
        .sort(["classification", "vector_score"], descending=[False, False])
        .head(20)
        .collect()
    )
    return {
        "universal_growth_vectors": row_dicts(universal),
        "epistemic_mirage_vectors": row_dicts(mirage),
        "prohibited_vectors": row_dicts(prohibited),
    }


def load_context(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_json(Path(args.manifest))
    assay_rows = (
        pl.scan_parquet(Path(args.assay_routing))
        .sort(["recommended_assay", "trigger_value"], descending=[False, True])
        .head(30)
        .collect()
    )
    fiber_rows = (
        pl.scan_parquet(Path(args.tri_state_fibers))
        .sort(["fiber_type", "projected_edge_score"], descending=[False, True])
        .collect()
    )
    context: dict[str, object] = {
        "manifest": manifest,
        "environment_rows": environment_rows(manifest),
        "schema_rows": schema_rows(manifest),
        "assay_rows": row_dicts(assay_rows),
        "chronology_rows": row_dicts(chronology_summary(Path(args.chronology))),
        "fiber_rows": row_dicts(fiber_rows),
        "fiber_counts": row_dicts(fiber_counts(Path(args.tri_state_fibers))),
        "top_fiber": top_fiber(Path(args.tri_state_fibers)),
        "negative_evidence_rows": row_dicts(negative_evidence(Path(args.risk_map))),
    }
    context.update(sar_context(Path(args.sar_register)))
    return context


def render(args: argparse.Namespace) -> None:
    template_path = Path(args.template)
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    rendered = env.get_template(template_path.name).render(**load_context(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def main() -> int:
    args = parse_args()
    required = [
        Path(args.manifest),
        Path(args.assay_routing),
        Path(args.chronology),
        Path(args.tri_state_fibers),
        Path(args.sar_register),
        Path(args.risk_map),
        Path(args.template),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing dossier input(s): {missing}")
    render(args)
    sys.stdout.write(f"wrote {Path(args.output)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
