#!/usr/bin/env python3
"""Render the master campaign data-room index from release artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from jinja2 import Environment, FileSystemLoader, StrictUndefined


Row = dict[str, Any]
CAMPAIGN_DIR = Path("campaigns/glp1r_aleniglipron")
TEMPLATE_DIR = Path("00_registry/templates")
DEFAULT_TEMPLATE = TEMPLATE_DIR / "master_data_room_index.md.j2"
DEFAULT_OUTPUT = CAMPAIGN_DIR / "MASTER_DATA_ROOM_INDEX.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cbom", type=Path, default=CAMPAIGN_DIR / "PRISM_CBOM_v1.0.json")
    parser.add_argument("--cro-plan", type=Path, default=CAMPAIGN_DIR / "CRO_WetLab_Action_Plan.parquet")
    parser.add_argument("--teaser-solutions", type=Path, default=CAMPAIGN_DIR / "track_0_manual_emulation/teaser_solutions.parquet")
    parser.add_argument("--sar-register", type=Path, default=CAMPAIGN_DIR / "track_0_manual_emulation/sar_contingency_register.parquet")
    return parser.parse_args()


def row_dicts(frame: pl.DataFrame) -> list[Row]:
    return [dict(row) for row in frame.to_dicts()]


def optional_rows(path: Path, limit: int, sort_columns: list[str] | None = None) -> list[Row]:
    if not path.exists():
        return []
    frame = pl.scan_parquet(path)
    if sort_columns:
        names = set(frame.collect_schema().names())
        if all(column in names for column in sort_columns):
            frame = frame.sort(sort_columns)
    return row_dicts(frame.head(limit).collect())


def normalized_sar_select(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.select(
        [
            pl.col("sar_rule_id").alias("rule_id"),
            pl.col("growth_vector_id").alias("vector_id"),
            pl.col("action_class").alias("classification"),
            pl.col("condition_family"),
            pl.col("condition_id"),
            pl.col("edge_label"),
            pl.col("cooperative_cluster_id").alias("cooperative_cluster_id"),
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
    if not path.exists():
        return {"universal_growth_vectors": [], "epistemic_mirage_vectors": [], "prohibited_vectors": []}
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


def parquet_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def csv_data_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def status_row(component: str, path: Path, row_count: int | None = None) -> Row:
    if path.exists():
        suffix = f"; rows={row_count}" if row_count is not None else f"; bytes={path.stat().st_size}"
        return {"component": component, "status": f"materialized at `{path.as_posix()}`{suffix}"}
    return {"component": component, "status": f"not materialized at `{path.as_posix()}`"}


def pending_campaigns() -> list[Row]:
    paths = [
        ("Phase 2C Metastable Atlas", CAMPAIGN_DIR / "phase_2c_metastable_atlas_triggers.json"),
        ("Phase 2C Snapshot Windows", CAMPAIGN_DIR / "phase_2c_snapshot_triggers.json"),
        ("Phase 2D Holo Launch Script", Path("bin/launch-n80-holo-aleniglipron.sh")),
        ("Phase 2D Holo Topology", Path("04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json")),
        ("Phase 2E Holo Occupancy Delta", Path("scripts/compute_holo_occupancy_fatigue.py")),
        ("Phase 2D Variant Grid Manifest", CAMPAIGN_DIR / "phase_2d_variant_grid_manifest.json"),
    ]
    return [{"name": name, "path": path.as_posix(), "status": "staged" if path.exists() else "not_materialized"} for name, path in paths]


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    cbom = json.loads(args.cbom.read_text(encoding="utf-8"))
    visualizer_campaign = CAMPAIGN_DIR / "visualizer_app/index.html"
    visualizer_dist = Path("apps/glp1r-teaser-visualizer/dist/index.html")
    visualizer_index = visualizer_campaign if visualizer_campaign.exists() else visualizer_dist
    curated_anchors = CAMPAIGN_DIR / "track_a_generative/115k_curated_anchors.csv"
    calibration_anchors = CAMPAIGN_DIR / "track_a_generative/calibration_anchors_3d.parquet"
    reward_boundaries = CAMPAIGN_DIR / "track_a_generative/gflownet_tso_bridge_boundaries.parquet"
    context: dict[str, Any] = {
        "campaign_id": "glp1r_aleniglipron",
        "release_date": datetime.now(UTC).date().isoformat(),
        "cbom": cbom,
        "executive_dossier": "M2_Pharmacological_Dynamics_Intelligence_Report.md",
        "enterprise_positioning": "ENTERPRISE_POSITIONING_SUMMARY.md",
        "claim_graph": "claim_falsification_graph.json",
        "visualizer_index": visualizer_index.as_posix(),
        "cloudflare_architecture": "00_registry/architecture/Cloudflare_Manifold_Architecture.md",
        "teaser_solutions": optional_rows(args.teaser_solutions, 10, ["solution_rank"]),
        "cro_rows": optional_rows(args.cro_plan, 40, ["assay_category", "priority_score"]),
        "track_a_status": [
            status_row("115k curated action-space CSV", curated_anchors, csv_data_lines(curated_anchors)),
            status_row("Calibration anchors 3D parquet", calibration_anchors, parquet_rows(calibration_anchors)),
            status_row("GFlowNet reward boundaries", reward_boundaries, parquet_rows(reward_boundaries)),
            status_row("Cloudflare tripartite store architecture", Path("00_registry/architecture/Cloudflare_Manifold_Architecture.md")),
        ],
        "pending_gpu_campaigns": pending_campaigns(),
    }
    context.update(sar_context(args.sar_register))
    return context


def render(args: argparse.Namespace) -> str:
    env = Environment(loader=FileSystemLoader(str(args.template.parent)), undefined=StrictUndefined, autoescape=False)
    template = env.get_template(args.template.name)
    return template.render(**load_context(args))


def main() -> None:
    args = parse_args()
    if not args.cbom.exists():
        raise FileNotFoundError(f"CBOM not found; run scripts/build_campaign_cbom.py first: {args.cbom}")
    rendered = render(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
