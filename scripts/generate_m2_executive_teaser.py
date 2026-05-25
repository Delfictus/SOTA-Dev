#!/usr/bin/env python3
"""Render the GLP-1R/Aleniglipron M2 executive readout from fused parquets."""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import polars as pl
from jinja2 import Environment, FileSystemLoader, StrictUndefined


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
TRACK0_DIR = CAMPAIGN_DIR / "track_0_manual_emulation"
TEMPLATE_DIR = REPO_ROOT / "00_registry/templates"
DEFAULT_TEMPLATE = TEMPLATE_DIR / "m2_executive_readout.md.j2"
DEFAULT_OUTPUT = CAMPAIGN_DIR / "M2_Executive_Readout_Final.md"

JsonObject: TypeAlias = dict[str, object]
Row: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class ReadoutPaths:
    stream_phase_counts: Path
    phase_manifold_coherence: Path
    phase_edge_validation: Path
    risk_map: Path
    channel_summary: Path
    temporal_cascade: Path
    hysteresis_tensor: Path
    interferometric_differential: Path
    shear_stress_field: Path
    translation_pathway: Path
    whole_projection: Path
    fragment_attribution: Path
    template: Path
    output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-phase-counts", type=Path, default=N80_DIR / "stream_level_phase_counts.parquet")
    parser.add_argument("--phase-manifold-coherence", type=Path, default=N80_DIR / "phase_manifold_coherence.parquet")
    parser.add_argument("--phase-edge-validation", type=Path, default=N80_DIR / "phase_manifold_edge_validation.parquet")
    parser.add_argument("--risk-map", type=Path, default=N80_DIR / "receptor_durability_risk_map.parquet")
    parser.add_argument("--channel-summary", type=Path, default=N80_DIR / "receptor_durability_channel_summary.parquet")
    parser.add_argument("--temporal-cascade", type=Path, default=N80_DIR / "temporal_cascade.parquet")
    parser.add_argument("--hysteresis-tensor", type=Path, default=N80_DIR / "hysteresis_tensor.parquet")
    parser.add_argument("--interferometric-differential", type=Path, default=N80_DIR / "interferometric_differential.parquet")
    parser.add_argument("--shear-stress-field", type=Path, default=N80_DIR / "shear_stress_field.parquet")
    parser.add_argument("--translation-pathway", type=Path, default=N80_DIR / "translation_pathway_nodes.parquet")
    parser.add_argument(
        "--whole-projection",
        type=Path,
        default=TRACK0_DIR / "layer1_whole_molecule/analog_durability_projection.parquet",
    )
    parser.add_argument("--fragment-attribution", type=Path, default=TRACK0_DIR / "fragment_interference_attribution.parquet")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def readout_paths(args: argparse.Namespace) -> ReadoutPaths:
    return ReadoutPaths(
        stream_phase_counts=cast(Path, args.stream_phase_counts),
        phase_manifold_coherence=cast(Path, args.phase_manifold_coherence),
        phase_edge_validation=cast(Path, args.phase_edge_validation),
        risk_map=cast(Path, args.risk_map),
        channel_summary=cast(Path, args.channel_summary),
        temporal_cascade=cast(Path, args.temporal_cascade),
        hysteresis_tensor=cast(Path, args.hysteresis_tensor),
        interferometric_differential=cast(Path, args.interferometric_differential),
        shear_stress_field=cast(Path, args.shear_stress_field),
        translation_pathway=cast(Path, args.translation_pathway),
        whole_projection=cast(Path, args.whole_projection),
        fragment_attribution=cast(Path, args.fragment_attribution),
        template=cast(Path, args.template),
        output=cast(Path, args.output),
    )


def sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def row_dicts(frame: pl.DataFrame) -> list[Row]:
    return cast(list[Row], frame.to_dicts())


def as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got bool")
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError(f"{label} must be an integer")


def as_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError(f"{label} must be numeric")


def scalar_int(frame: pl.DataFrame, column: str) -> int:
    return as_int(cast(object, frame.get_column(column)[0]), column)


def scalar_float(frame: pl.DataFrame, column: str) -> float:
    return as_float(cast(object, frame.get_column(column)[0]), column)


def table_rows(frame: pl.DataFrame) -> list[Row]:
    return row_dicts(frame)


def critical_edge_table(phase_edges: pl.DataFrame, risk_map: pl.DataFrame) -> pl.DataFrame:
    keys = ["condition_id", "edge_from_residue", "edge_to_residue", "edge_class"]
    risk_cols = [
        *keys,
        "durability_risk_score_raw",
        "durability_class",
        "mechanically_pruned",
    ]
    return (
        phase_edges.lazy()
        .join(risk_map.select(risk_cols).lazy(), on=keys, how="left")
        .select(
            [
                "edge_id",
                "edge_label",
                "edge_class",
                "edge_coherence_score",
                "validation_status",
                "durability_risk_score_raw",
                "durability_class",
                "mechanically_pruned",
            ]
        )
        .sort(["edge_class", "edge_label"])
        .collect()
    )


def durability_summary(risk_map: pl.DataFrame) -> pl.DataFrame:
    return (
        risk_map.lazy()
        .group_by("durability_class")
        .agg(
            [
                pl.len().alias("edge_count"),
                pl.col("durability_risk_score_raw").mean().alias("mean_risk_score"),
                pl.col("mechanically_pruned").cast(pl.UInt32).sum().alias("mechanically_pruned_edges"),
            ]
        )
        .sort("edge_count", descending=True)
        .collect()
    )


def cascade_d_hysteresis(temporal_cascade: pl.DataFrame, phase_edges: pl.DataFrame) -> JsonObject:
    endpoints = pl.concat(
        [
            phase_edges.select(["condition_id", pl.col("edge_from_residue").alias("primary_residue_idx")]),
            phase_edges.select(["condition_id", pl.col("edge_to_residue").alias("primary_residue_idx")]),
        ]
    ).unique()
    joined = (
        temporal_cascade.filter(pl.col("protocol_group") == "D_Hysteresis")
        .join(
            endpoints.with_columns(pl.lit(True).alias("is_critical_endpoint")),
            on=["condition_id", "primary_residue_idx"],
            how="left",
        )
        .with_columns(pl.col("is_critical_endpoint").fill_null(False))
    )
    summary = joined.group_by("is_critical_endpoint").agg(
        [
            pl.len().alias("row_count"),
            pl.col("first_ramp_md_step").median().alias("median_first_ramp"),
            pl.col("first_ramp_md_step").mean().alias("mean_first_ramp"),
        ]
    )
    critical = summary.filter(pl.col("is_critical_endpoint")).head(1)
    noncritical = summary.filter(~pl.col("is_critical_endpoint")).head(1)
    return {
        "critical_median": scalar_float(critical, "median_first_ramp"),
        "noncritical_median": scalar_float(noncritical, "median_first_ramp"),
        "critical_mean": scalar_float(critical, "mean_first_ramp"),
        "noncritical_mean": scalar_float(noncritical, "mean_first_ramp"),
    }


def hysteresis_focus(hysteresis: pl.DataFrame) -> pl.DataFrame:
    return (
        hysteresis.lazy()
        .filter(pl.col("condition_id").is_in(["glp1r_6LN2_A316T", "glp1r_6XOX_WT"]))
        .group_by(["condition_id", "protocol_group"])
        .agg(
            [
                pl.col("thermal_irreversibility").mean().alias("mean_irreversibility"),
                (1.0 - pl.col("thermal_irreversibility").mean()).alias("mean_reversibility"),
                pl.col("cold_hold_spikes").sum().alias("cold_hold_spikes"),
                pl.col("cold_return_spikes").sum().alias("cold_return_spikes"),
            ]
        )
        .sort(["condition_id", "protocol_group"])
        .collect()
    )


def pathway_rows(pathway: pl.DataFrame) -> pl.DataFrame:
    return (
        pathway.lazy()
        .select(
            [
                "residue_name",
                "condition_id",
                "pathway_rank",
                "wire_score",
                "structural_fault_line",
                "violent_kinetic_node",
                "shear_stress",
                "max_burst_motion",
            ]
        )
        .sort(["pathway_rank", "condition_id"])
        .head(10)
        .collect()
    )


def lineage_rows(paths: ReadoutPaths) -> list[Row]:
    artifacts = {
        "stream_level_phase_counts.parquet": paths.stream_phase_counts,
        "phase_manifold_coherence.parquet": paths.phase_manifold_coherence,
        "phase_manifold_edge_validation.parquet": paths.phase_edge_validation,
        "receptor_durability_risk_map.parquet": paths.risk_map,
        "temporal_cascade.parquet": paths.temporal_cascade,
        "hysteresis_tensor.parquet": paths.hysteresis_tensor,
        "interferometric_differential.parquet": paths.interferometric_differential,
        "shear_stress_field.parquet": paths.shear_stress_field,
        "translation_pathway_nodes.parquet": paths.translation_pathway,
        "analog_durability_projection.parquet": paths.whole_projection,
        "fragment_interference_attribution.parquet": paths.fragment_attribution,
    }
    return [{"name": name, "sha256": sha256_path(path)} for name, path in artifacts.items()]


def load_context(paths: ReadoutPaths) -> JsonObject:
    stream_phase_counts = pl.read_parquet(paths.stream_phase_counts)
    phase_coherence = pl.read_parquet(paths.phase_manifold_coherence)
    phase_edges = pl.read_parquet(paths.phase_edge_validation)
    risk_map = pl.read_parquet(paths.risk_map)
    channel_summary = pl.read_parquet(paths.channel_summary)
    temporal_cascade = pl.read_parquet(paths.temporal_cascade)
    hysteresis = pl.read_parquet(paths.hysteresis_tensor)
    interferometric = pl.read_parquet(paths.interferometric_differential)
    shear_stress = pl.scan_parquet(paths.shear_stress_field).select(pl.len().alias("row_count")).collect()
    pathway = pl.read_parquet(paths.translation_pathway)
    whole_projection_frame = pl.read_parquet(paths.whole_projection)
    fragment_attribution = pl.read_parquet(paths.fragment_attribution)
    critical_edges = critical_edge_table(phase_edges, risk_map)
    whole_projection = whole_projection_frame.to_dicts()[0]
    fragment_rows = fragment_attribution.sort("inter_fragment_coupling", descending=True)
    return {
        "spike_count_text": f"{scalar_int(stream_phase_counts.select(pl.col('spike_count').sum().alias('total')), 'total'):,}",
        "stream_phase_rows": stream_phase_counts.height,
        "coherence_rows": phase_coherence.height,
        "hysteresis_rows": hysteresis.height,
        "interferometric_rows": interferometric.height,
        "shear_rows": scalar_int(shear_stress, "row_count"),
        "risk_rows": risk_map.height,
        "mechanically_pruned_edges": scalar_int(
            risk_map.select(pl.col("mechanically_pruned").cast(pl.UInt32).sum().alias("n")),
            "n",
        ),
        "cascade_d_hysteresis": cascade_d_hysteresis(temporal_cascade, phase_edges),
        "durability_summary": table_rows(durability_summary(risk_map)),
        "phase_validation_rows": table_rows(critical_edges),
        "hysteresis_focus_rows": table_rows(hysteresis_focus(hysteresis)),
        "critical_edge_rows": table_rows(critical_edges),
        "pathway_rows": table_rows(pathway_rows(pathway)),
        "whole_projection": whole_projection,
        "total_inter_fragment_coupling": scalar_float(
            fragment_attribution.select(pl.col("inter_fragment_coupling").sum().alias("total")),
            "total",
        ),
        "fragment_rows": table_rows(fragment_rows),
        "lineage_rows": lineage_rows(paths),
        "channel_summary_rows": channel_summary.height,
    }


def render(paths: ReadoutPaths) -> None:
    env = Environment(
        loader=FileSystemLoader(str(paths.template.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(paths.template.name)
    rendered = template.render(**load_context(paths))
    paths.output.parent.mkdir(parents=True, exist_ok=True)
    paths.output.write_text(rendered, encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=str(args.log_level).upper(), format="%(levelname)s %(message)s")
    paths = readout_paths(args)
    missing = [
        path
        for name, path in paths.__dict__.items()
        if name != "output" and isinstance(path, Path) and not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing readout input(s): {missing}")
    render(paths)
    logging.info("m2_readout_written path=%s", paths.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
