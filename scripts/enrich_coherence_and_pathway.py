#!/usr/bin/env python3
"""Enrich phase-manifold coherence, edge validation, and translation pathways.

Derived formulas:
    Temporal break stats aggregate first_break_time_ps and mean_break_time_ps by residue.
    Causal lag stats aggregate KCC causal_lag by residue.
    Force angle computes acos(abs(mean_force dot pocket_to_lock_axis) / ||mean_force||).
    Spike voxel ratios divide activated/destabilized/stable spike counts by total voxel-classified spikes.
    Anchor fraction = total causal_anchor_count / total masked_spike_count.
    Steering convergence = number of conditions steering to residue label / total conditions.
    Propagation speed = causal_lag / betweenness for pathway nodes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import polars as pl
import structlog
from scipy.stats import spearmanr  # type: ignore[import-untyped]


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import sha256_path, write_provenance_parquet
from prism_dstw.ontology import ResidueIdx


JsonObject: TypeAlias = dict[str, object]
Coordinate: TypeAlias = tuple[float, float, float]
N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
TRACK0_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation"

DEFAULT_COHERENCE = N80_DIR / "phase_manifold_coherence.parquet"
DEFAULT_EDGE_VALIDATION = N80_DIR / "phase_manifold_edge_validation.parquet"
DEFAULT_PATHWAY = N80_DIR / "translation_pathway_nodes.parquet"
DEFAULT_SPIKE_RESIDUE = N80_DIR / "spike_residue_fields.parquet"
DEFAULT_KCC = N80_DIR / "kcc_residue_fields.parquet"
DEFAULT_MECHANICAL = N80_DIR / "mechanical_load_network.parquet"
DEFAULT_STEERING = N80_DIR / "autonomous_steering_tensor.parquet"
DEFAULT_RISK_MAP = N80_DIR / "receptor_durability_risk_map.parquet"
DEFAULT_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/topology/residue_index_mapping_matrix.parquet"
DEFAULT_GRID_MAPPING = TRACK0_DIR / "grid_coordinate_mapping.json"
DEFAULT_BINDING_SITE = TRACK0_DIR / "binding_site_reference.json"

DEFAULT_ENRICHED_EDGE = N80_DIR / "enriched_edge_validation.parquet"
DEFAULT_ENRICHED_PATHWAY = N80_DIR / "enriched_pathway_nodes.parquet"
DEFAULT_WIRE_CONSERVATION = N80_DIR / "wire_conservation_matrix.parquet"
DEFAULT_CASCADE = N80_DIR / "temporal_cascade_summary.parquet"
EPSILON = 1.0e-6


@dataclass(frozen=True)
class Inputs:
    coherence: Path
    edge_validation: Path
    pathway: Path
    spike_residue: Path
    kcc: Path
    mechanical: Path
    steering: Path
    risk_map: Path
    mapping: Path
    grid_mapping: Path
    binding_site: Path


@dataclass(frozen=True)
class Outputs:
    enriched_edge: Path
    enriched_pathway: Path
    wire_conservation: Path
    temporal_cascade: Path


@dataclass(frozen=True)
class TopologyProducts:
    atom_to_residue: pl.DataFrame
    residue_coordinates: pl.DataFrame
    topology_hashes: dict[str, str]
    topology_files: dict[str, str]


class BoundLogger(Protocol):
    def info(self, event: str, **kwargs: object) -> None:
        """Emit a structured info event."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coherence", type=Path, default=DEFAULT_COHERENCE)
    parser.add_argument("--edge-validation", type=Path, default=DEFAULT_EDGE_VALIDATION)
    parser.add_argument("--pathway", type=Path, default=DEFAULT_PATHWAY)
    parser.add_argument("--spike-residue", type=Path, default=DEFAULT_SPIKE_RESIDUE)
    parser.add_argument("--kcc", type=Path, default=DEFAULT_KCC)
    parser.add_argument("--mechanical", type=Path, default=DEFAULT_MECHANICAL)
    parser.add_argument("--steering", type=Path, default=DEFAULT_STEERING)
    parser.add_argument("--risk-map", type=Path, default=DEFAULT_RISK_MAP)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--binding-site", type=Path, default=DEFAULT_BINDING_SITE)
    parser.add_argument("--enriched-edge-out", type=Path, default=DEFAULT_ENRICHED_EDGE)
    parser.add_argument("--enriched-pathway-out", type=Path, default=DEFAULT_ENRICHED_PATHWAY)
    parser.add_argument("--wire-conservation-out", type=Path, default=DEFAULT_WIRE_CONSERVATION)
    parser.add_argument("--temporal-cascade-out", type=Path, default=DEFAULT_CASCADE)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> BoundLogger:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.KeyValueRenderer(),
        ],
    )
    return cast(BoundLogger, structlog.get_logger("enrich_coherence_and_pathway"))


def input_paths(args: argparse.Namespace) -> Inputs:
    return Inputs(
        coherence=cast(Path, args.coherence),
        edge_validation=cast(Path, args.edge_validation),
        pathway=cast(Path, args.pathway),
        spike_residue=cast(Path, args.spike_residue),
        kcc=cast(Path, args.kcc),
        mechanical=cast(Path, args.mechanical),
        steering=cast(Path, args.steering),
        risk_map=cast(Path, args.risk_map),
        mapping=cast(Path, args.mapping),
        grid_mapping=cast(Path, args.grid_mapping),
        binding_site=cast(Path, args.binding_site),
    )


def output_paths(args: argparse.Namespace) -> Outputs:
    return Outputs(
        enriched_edge=cast(Path, args.enriched_edge_out),
        enriched_pathway=cast(Path, args.enriched_pathway_out),
        wire_conservation=cast(Path, args.wire_conservation_out),
        temporal_cascade=cast(Path, args.temporal_cascade_out),
    )


def source_parquets(paths: Inputs) -> list[Path]:
    return [
        paths.coherence,
        paths.edge_validation,
        paths.pathway,
        paths.spike_residue,
        paths.kcc,
        paths.mechanical,
        paths.steering,
        paths.risk_map,
        paths.mapping,
    ]


def require_columns(path: Path, required: set[str]) -> None:
    names = set(pl.scan_parquet(path).collect_schema().names())
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def first_existing_column(path: Path, options: tuple[str, ...]) -> str:
    names = set(pl.scan_parquet(path).collect_schema().names())
    for option in options:
        if option in names:
            return option
    raise ValueError(f"{path} has none of the expected columns: {options}")


def load_json_object(path: Path) -> JsonObject:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not decode to an object")
    return cast(JsonObject, loaded)


def json_object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(JsonObject, value)


def json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


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


def residue_value(value: object, label: str) -> int:
    return int(ResidueIdx(as_int(value, label)))


def int_list(value: object, label: str) -> list[int]:
    return [as_int(item, f"{label}[]") for item in json_list(value, label)]


def float_list(value: object, label: str) -> list[float]:
    return [as_float(item, f"{label}[]") for item in json_list(value, label)]


def topology_paths_from_grid_mapping(path: Path) -> dict[str, Path]:
    payload = load_json_object(path)
    raw_conditions = json_object(payload.get("conditions"), f"{path}:conditions")
    out: dict[str, Path] = {}
    for condition_id, raw_geometry in raw_conditions.items():
        geometry = json_object(raw_geometry, f"{path}:conditions[{condition_id}]")
        topology_path = Path(str(geometry["topology_path"]))
        if not topology_path.exists():
            raise FileNotFoundError(topology_path)
        out[str(condition_id)] = topology_path
    return out


def coordinate_at(positions: list[float], atom_idx: int) -> Coordinate:
    base = atom_idx * 3
    return (positions[base], positions[base + 1], positions[base + 2])


def topology_products(grid_mapping: Path) -> TopologyProducts:
    atom_rows: list[dict[str, int | str]] = []
    coord_rows: list[dict[str, int | float | str]] = []
    topology_hashes: dict[str, str] = {}
    topology_files: dict[str, str] = {}
    for condition_id, topology_path in topology_paths_from_grid_mapping(grid_mapping).items():
        topology = load_json_object(topology_path)
        residue_ids = int_list(topology.get("residue_ids"), f"{topology_path.name}:residue_ids")
        ca_indices = int_list(topology.get("ca_indices"), f"{topology_path.name}:ca_indices")
        positions = float_list(topology.get("positions"), f"{topology_path.name}:positions")
        topology_hashes[condition_id] = sha256_path(topology_path)
        topology_files[condition_id] = topology_path.name
        for atom_idx, residue_idx in enumerate(residue_ids):
            atom_rows.append({"condition_id": condition_id, "atom_idx": atom_idx, "residue_idx": residue_value(residue_idx, "residue_idx")})
        for residue_idx, ca_atom_idx in enumerate(ca_indices):
            coord = coordinate_at(positions, ca_atom_idx)
            coord_rows.append(
                {
                    "condition_id": condition_id,
                    "residue_idx": residue_value(residue_idx, "residue_idx"),
                    "ca_x": coord[0],
                    "ca_y": coord[1],
                    "ca_z": coord[2],
                }
            )
    return TopologyProducts(
        atom_to_residue=pl.DataFrame(atom_rows),
        residue_coordinates=pl.DataFrame(coord_rows),
        topology_hashes=topology_hashes,
        topology_files=topology_files,
    )


def residue_mapping(path: Path) -> pl.DataFrame:
    require_columns(path, {"condition_id", "residue_idx", "canonical_residue_label"})
    return (
        pl.scan_parquet(path)
        .select(["condition_id", pl.col("residue_idx").cast(pl.Int32), pl.col("canonical_residue_label").alias("residue_name")])
        .collect()
    )


def residue_stats(paths: Inputs) -> pl.DataFrame:
    require_columns(
        paths.spike_residue,
        {
            "condition_id",
            "residue_idx",
            "first_break_time_ps",
            "mean_break_time_ps",
            "mean_survival_time_ps",
            "short_lived_regime_break_count",
            "thermally_destabilized_spike_count",
            "thermally_activated_spike_count",
            "stable_occupied_spike_count",
            "causal_anchor_count",
            "masked_spike_count",
        },
    )
    srf = pl.scan_parquet(paths.spike_residue)
    return (
        srf.select(
            [
                "condition_id",
                pl.col("residue_idx").cast(pl.Int32),
                "first_break_time_ps",
                "mean_break_time_ps",
                "mean_survival_time_ps",
                "short_lived_regime_break_count",
                "thermally_destabilized_spike_count",
                "thermally_activated_spike_count",
                "stable_occupied_spike_count",
                "causal_anchor_count",
                "masked_spike_count",
            ]
        )
        .group_by(["condition_id", "residue_idx"])
        .agg(
            [
                pl.col("first_break_time_ps").mean().alias("mean_first_break_ps"),
                pl.col("first_break_time_ps").std().fill_null(0.0).alias("std_first_break_ps"),
                pl.col("first_break_time_ps").min().alias("earliest_break_ps"),
                pl.col("first_break_time_ps").max().alias("latest_break_ps"),
                pl.col("mean_break_time_ps").mean().alias("mean_mean_break_ps"),
                pl.col("mean_survival_time_ps").mean().alias("mean_survival_ps"),
                pl.col("mean_survival_time_ps").std().fill_null(0.0).alias("std_survival_ps"),
                pl.col("mean_survival_time_ps").min().alias("min_survival_ps"),
                pl.col("mean_survival_time_ps").max().alias("max_survival_ps"),
                pl.col("short_lived_regime_break_count").sum().alias("total_short_breaks"),
                pl.col("thermally_destabilized_spike_count").sum().alias("total_destab_spikes"),
                pl.col("thermally_activated_spike_count").sum().alias("total_activ_spikes"),
                pl.col("stable_occupied_spike_count").sum().alias("total_stable_spikes"),
                pl.col("causal_anchor_count").sum().alias("total_anchors"),
                pl.col("masked_spike_count").sum().alias("total_spikes_residue"),
            ]
        )
        .with_columns(
            (
                pl.col("total_destab_spikes") + pl.col("total_activ_spikes") + pl.col("total_stable_spikes") + EPSILON
            ).alias("voxel_spike_denominator")
        )
        .with_columns(
            [
                (pl.col("total_activ_spikes") / pl.col("voxel_spike_denominator")).alias("activation_spike_ratio"),
                (pl.col("total_destab_spikes") / pl.col("voxel_spike_denominator")).alias("destab_spike_ratio"),
                (pl.col("total_stable_spikes") / pl.col("voxel_spike_denominator")).alias("stable_spike_ratio"),
                (pl.col("total_anchors") / (pl.col("total_spikes_residue") + EPSILON)).alias("anchor_fraction"),
            ]
        )
        .drop("voxel_spike_denominator")
        .collect()
    )


def kcc_stats(path: Path) -> pl.DataFrame:
    temporal_col = first_existing_column(path, ("temporal_correlation", "temporal_corr"))
    require_columns(path, {"condition_id", "residue_idx", temporal_col, "causal_lag"})
    return (
        pl.scan_parquet(path)
        .select(["condition_id", pl.col("residue_idx").cast(pl.Int32), pl.col(temporal_col).alias("temporal_correlation"), "causal_lag"])
        .group_by(["condition_id", "residue_idx"])
        .agg(
            [
                pl.col("causal_lag").mean().alias("mean_causal_lag"),
                pl.col("causal_lag").std().fill_null(0.0).alias("std_causal_lag"),
                pl.col("causal_lag").median().alias("median_causal_lag"),
                pl.col("temporal_correlation").mean().alias("mean_temporal_correlation"),
            ]
        )
        .collect()
    )


def steering_convergence(paths: Inputs, mapping: pl.DataFrame) -> pl.DataFrame:
    require_columns(paths.steering, {"condition_id", "residue_idx", "steering_weight_mean", "steering_weight_sum"})
    condition_count = mapping.select("condition_id").n_unique()
    per_condition = (
        pl.scan_parquet(paths.steering)
        .select(["condition_id", pl.col("residue_idx").cast(pl.Int32), "steering_weight_sum", "steering_weight_mean"])
        .group_by(["condition_id", "residue_idx"])
        .agg(
            [
                pl.col("steering_weight_sum").sum().alias("condition_steering_weight_sum"),
                pl.col("steering_weight_mean").mean().alias("condition_steering_weight_mean"),
            ]
        )
        .collect()
        .join(mapping, on=["condition_id", "residue_idx"], how="left")
    )
    cross_condition = (
        per_condition.lazy()
        .filter(pl.col("condition_steering_weight_sum") > 0.0)
        .group_by("residue_name")
        .agg(
            [
                pl.col("condition_id").n_unique().alias("n_conditions_steered"),
                pl.col("condition_steering_weight_mean").mean().alias("mean_steering_weight"),
                pl.col("condition_steering_weight_mean").std().fill_null(0.0).alias("std_steering_weight"),
            ]
        )
        .with_columns((pl.col("n_conditions_steered").cast(pl.Float64) / float(condition_count)).alias("steering_convergence"))
        .collect()
    )
    return mapping.select(["condition_id", "residue_idx", "residue_name"]).join(cross_condition, on="residue_name", how="left").with_columns(
        [
            pl.col("n_conditions_steered").fill_null(0).cast(pl.UInt8),
            pl.col("mean_steering_weight").fill_null(0.0),
            pl.col("std_steering_weight").fill_null(0.0),
            pl.col("steering_convergence").fill_null(0.0),
        ]
    )


def endpoint_sets(risk_map_path: Path) -> pl.DataFrame:
    require_columns(risk_map_path, {"condition_id", "edge_from_residue", "edge_to_residue", "edge_class"})
    edges = pl.scan_parquet(risk_map_path).filter(pl.col("edge_class").is_in(["pocket_vector", "downstream_lock"]))
    return pl.concat(
        [
            edges.select(["condition_id", pl.col("edge_class").alias("endpoint_class"), pl.col("edge_from_residue").cast(pl.Int32).alias("residue_idx")]),
            edges.select(["condition_id", pl.col("edge_class").alias("endpoint_class"), pl.col("edge_to_residue").cast(pl.Int32).alias("residue_idx")]),
        ]
    ).unique().collect()


def pathway_axes(risk_map_path: Path, residue_coordinates: pl.DataFrame) -> pl.DataFrame:
    endpoints = endpoint_sets(risk_map_path)
    centroids = (
        endpoints.lazy()
        .join(residue_coordinates.lazy(), on=["condition_id", "residue_idx"], how="inner")
        .group_by(["condition_id", "endpoint_class"])
        .agg(
            [
                pl.col("ca_x").mean().alias("centroid_x"),
                pl.col("ca_y").mean().alias("centroid_y"),
                pl.col("ca_z").mean().alias("centroid_z"),
            ]
        )
        .collect()
    )
    pocket = centroids.filter(pl.col("endpoint_class") == "pocket_vector").select(
        ["condition_id", pl.col("centroid_x").alias("pocket_x"), pl.col("centroid_y").alias("pocket_y"), pl.col("centroid_z").alias("pocket_z")]
    )
    lock = centroids.filter(pl.col("endpoint_class") == "downstream_lock").select(
        ["condition_id", pl.col("centroid_x").alias("lock_x"), pl.col("centroid_y").alias("lock_y"), pl.col("centroid_z").alias("lock_z")]
    )
    return (
        pocket.lazy()
        .join(lock.lazy(), on="condition_id", how="inner")
        .with_columns(
            [
                (pl.col("lock_x") - pl.col("pocket_x")).alias("axis_x"),
                (pl.col("lock_y") - pl.col("pocket_y")).alias("axis_y"),
                (pl.col("lock_z") - pl.col("pocket_z")).alias("axis_z"),
            ]
        )
        .with_columns((pl.col("axis_x") * pl.col("axis_x") + pl.col("axis_y") * pl.col("axis_y") + pl.col("axis_z") * pl.col("axis_z")).sqrt().alias("axis_norm"))
        .with_columns(
            [
                (pl.col("axis_x") / pl.col("axis_norm")).alias("axis_unit_x"),
                (pl.col("axis_y") / pl.col("axis_norm")).alias("axis_unit_y"),
                (pl.col("axis_z") / pl.col("axis_norm")).alias("axis_unit_z"),
            ]
        )
        .collect()
    )


def residue_betweenness(coordinates: pl.DataFrame, axes: pl.DataFrame) -> pl.DataFrame:
    return (
        coordinates.lazy()
        .join(axes.lazy(), on="condition_id", how="inner")
        .with_columns(
            [
                (pl.col("ca_x") - pl.col("pocket_x")).alias("rel_x"),
                (pl.col("ca_y") - pl.col("pocket_y")).alias("rel_y"),
                (pl.col("ca_z") - pl.col("pocket_z")).alias("rel_z"),
            ]
        )
        .with_columns((pl.col("rel_x") * pl.col("axis_x") + pl.col("rel_y") * pl.col("axis_y") + pl.col("rel_z") * pl.col("axis_z")).alias("axis_dot"))
        .with_columns((pl.col("axis_dot") / (pl.col("axis_norm") * pl.col("axis_norm"))).alias("betweenness_raw"))
        .select(["condition_id", "residue_idx", pl.col("betweenness_raw").clip(0.0, 1.0).alias("residue_betweenness")])
        .collect()
    )


def mechanical_force_alignment(paths: Inputs, topologies: TopologyProducts, axes: pl.DataFrame) -> pl.DataFrame:
    require_columns(paths.mechanical, {"condition_id", "atom_idx", "force_x", "force_y", "force_z"})
    forces = (
        pl.scan_parquet(paths.mechanical)
        .select(["condition_id", pl.col("atom_idx").cast(pl.Int32), pl.col("force_x").cast(pl.Float64), pl.col("force_y").cast(pl.Float64), pl.col("force_z").cast(pl.Float64)])
        .join(topologies.atom_to_residue.lazy(), on=["condition_id", "atom_idx"], how="inner")
        .group_by(["condition_id", "residue_idx"])
        .agg(
            [
                pl.col("force_x").mean().alias("mean_fx"),
                pl.col("force_y").mean().alias("mean_fy"),
                pl.col("force_z").mean().alias("mean_fz"),
            ]
        )
        .collect()
        .join(axes.select(["condition_id", "axis_unit_x", "axis_unit_y", "axis_unit_z"]), on="condition_id", how="inner")
    )
    rows: list[dict[str, int | float | str]] = []
    for row in forces.iter_rows(named=True):
        fx = as_float(row["mean_fx"], "mean_fx")
        fy = as_float(row["mean_fy"], "mean_fy")
        fz = as_float(row["mean_fz"], "mean_fz")
        axis_x = as_float(row["axis_unit_x"], "axis_unit_x")
        axis_y = as_float(row["axis_unit_y"], "axis_unit_y")
        axis_z = as_float(row["axis_unit_z"], "axis_unit_z")
        force_norm = math.sqrt(fx * fx + fy * fy + fz * fz)
        if force_norm < 1.0e-10:
            angle = 90.0
        else:
            cos_angle = abs((fx * axis_x + fy * axis_y + fz * axis_z) / force_norm)
            angle = math.degrees(math.acos(min(1.0, max(-1.0, cos_angle))))
        rows.append(
            {
                "condition_id": str(row["condition_id"]),
                "residue_idx": residue_value(row["residue_idx"], "residue_idx"),
                "mean_fx": fx,
                "mean_fy": fy,
                "mean_fz": fz,
                "force_angle_to_pathway_deg": angle,
            }
        )
    return pl.DataFrame(rows)


def endpoint_residue_frame(edge_validation: pl.DataFrame) -> pl.DataFrame:
    base_cols = ["edge_id", "condition_id", "edge_label", "edge_class", "durability_risk_score_raw", "edge_coherence_score", "validation_status"]
    from_rows = edge_validation.select(base_cols + [pl.col("edge_from_residue").alias("residue_idx")]).with_columns(pl.lit("from").alias("edge_endpoint"))
    to_rows = edge_validation.select(base_cols + [pl.col("edge_to_residue").alias("residue_idx")]).with_columns(pl.lit("to").alias("edge_endpoint"))
    return pl.concat([from_rows, to_rows])


def enrich_edges(
    paths: Inputs,
    stats: pl.DataFrame,
    lag_stats: pl.DataFrame,
    steering: pl.DataFrame,
    force_alignment: pl.DataFrame,
) -> pl.DataFrame:
    edge_validation = pl.read_parquet(paths.edge_validation)
    endpoint_rows = (
        endpoint_residue_frame(edge_validation)
        .join(stats, on=["condition_id", "residue_idx"], how="left")
        .join(lag_stats, on=["condition_id", "residue_idx"], how="left")
        .join(steering, on=["condition_id", "residue_idx"], how="left")
        .join(force_alignment, on=["condition_id", "residue_idx"], how="left")
    )
    return (
        endpoint_rows.lazy()
        .group_by(["edge_id", "condition_id", "edge_label", "edge_class", "durability_risk_score_raw", "edge_coherence_score", "validation_status"])
        .agg(
            [
                pl.col("mean_first_break_ps").mean(),
                pl.col("std_first_break_ps").mean(),
                pl.col("earliest_break_ps").min(),
                pl.col("latest_break_ps").max(),
                pl.col("mean_causal_lag").mean(),
                pl.col("std_causal_lag").mean(),
                pl.col("mean_survival_ps").mean(),
                pl.col("std_survival_ps").mean(),
                pl.col("total_short_breaks").sum().alias("short_break_count"),
                pl.col("activation_spike_ratio").mean(),
                pl.col("destab_spike_ratio").mean(),
                pl.col("stable_spike_ratio").mean(),
                pl.col("anchor_fraction").mean(),
                pl.col("steering_convergence").mean(),
                pl.col("force_angle_to_pathway_deg").mean(),
            ]
        )
        .with_columns(
            [
                pl.col("mean_first_break_ps").rank("ordinal").over("condition_id").cast(pl.UInt8).alias("temporal_rank"),
                pl.col("mean_causal_lag").rank("ordinal").over("condition_id").cast(pl.UInt8).alias("causal_lag_rank"),
                pl.when(pl.col("mean_survival_ps") >= pl.col("mean_survival_ps").quantile(0.67).over("condition_id"))
                .then(pl.lit("stable"))
                .when(pl.col("mean_survival_ps") >= pl.col("mean_survival_ps").quantile(0.33).over("condition_id"))
                .then(pl.lit("moderate"))
                .otherwise(pl.lit("flickering"))
                .alias("durability_characterization"),
            ]
        )
        .sort(["condition_id", "temporal_rank"])
        .collect()
    )


def enrich_pathway(
    paths: Inputs,
    stats: pl.DataFrame,
    lag_stats: pl.DataFrame,
    steering: pl.DataFrame,
    force_alignment: pl.DataFrame,
) -> pl.DataFrame:
    pathway = pl.read_parquet(paths.pathway)
    return (
        pathway.lazy()
        .join(stats.lazy(), on=["condition_id", "residue_idx"], how="left")
        .join(lag_stats.lazy(), on=["condition_id", "residue_idx"], how="left")
        .join(steering.lazy(), on=["condition_id", "residue_idx"], how="left")
        .join(force_alignment.lazy(), on=["condition_id", "residue_idx"], how="left")
        .with_columns(
            [
                pl.col("mean_first_break_ps").rank("ordinal").over("condition_id").cast(pl.UInt8).alias("break_time_rank"),
                pl.col("mean_causal_lag").rank("ordinal").over("condition_id").cast(pl.UInt8).alias("causal_lag_rank"),
                (pl.col("mean_causal_lag") / (pl.col("betweenness") + EPSILON)).alias("propagation_speed_ps_per_A"),
            ]
        )
        .sort(["condition_id", "pathway_rank"])
        .collect()
    )


def wire_conservation_matrix(
    paths: Inputs,
    mapping: pl.DataFrame,
    stats: pl.DataFrame,
    lag_stats: pl.DataFrame,
    force_alignment: pl.DataFrame,
    axes: pl.DataFrame,
    topologies: TopologyProducts,
) -> pl.DataFrame:
    pathway = pl.read_parquet(paths.pathway)
    wire_names = pathway.select("residue_name").unique()
    all_condition_names = mapping.select("condition_id").unique().join(wire_names, how="cross")
    load = pl.read_parquet(paths.coherence).select(
        [
            "condition_id",
            "residue_idx",
            "residue_name",
            "mean_abs_load",
            "temporal_correlation",
            "coherence_score",
            "coherence_class",
        ]
    )
    candidate = (
        mapping.select(["condition_id", "residue_idx", "residue_name"])
        .join(wire_names, on="residue_name", how="inner")
        .join(load, on=["condition_id", "residue_idx", "residue_name"], how="left")
        .join(stats.select(["condition_id", "residue_idx", "anchor_fraction"]), on=["condition_id", "residue_idx"], how="left")
        .join(lag_stats.select(["condition_id", "residue_idx", "mean_causal_lag"]), on=["condition_id", "residue_idx"], how="left")
        .join(force_alignment.select(["condition_id", "residue_idx", "force_angle_to_pathway_deg"]), on=["condition_id", "residue_idx"], how="left")
    )
    betweenness = residue_betweenness(topologies.residue_coordinates, axes)
    candidate = candidate.join(betweenness, on=["condition_id", "residue_idx"], how="left")
    thresholds = (
        load.lazy()
        .group_by("condition_id")
        .agg(
            [
                pl.col("mean_abs_load").median().alias("load_threshold"),
                pl.col("temporal_correlation").abs().median().alias("kcc_threshold"),
            ]
        )
        .collect()
    )
    present = pathway.select(["condition_id", "residue_name"]).unique().with_columns(pl.lit(True).alias("is_wire"))
    return (
        all_condition_names.join(candidate, on=["condition_id", "residue_name"], how="left")
        .join(thresholds, on="condition_id", how="left")
        .join(present, on=["condition_id", "residue_name"], how="left")
        .with_columns(pl.col("is_wire").fill_null(False))
        .with_columns(
            pl.when(pl.col("is_wire"))
            .then(pl.lit("present"))
            .when(pl.col("residue_idx").is_null())
            .then(pl.lit("residue_absent"))
            .when(pl.col("mean_abs_load") < pl.col("load_threshold"))
            .then(pl.lit("load_below_threshold"))
            .when(pl.col("temporal_correlation").abs() < pl.col("kcc_threshold"))
            .then(pl.lit("kcc_below_threshold"))
            .when(~pl.col("coherence_class").is_in(["fiber_invariant", "partially_coherent"]))
            .then(pl.lit("coherence_not_supportive"))
            .when((pl.col("residue_betweenness") < 0.1) | (pl.col("residue_betweenness") > 0.9))
            .then(pl.lit("spatial_betweenness_failed"))
            .otherwise(pl.lit("not_top_ranked_wire"))
            .alias("failure_reason")
        )
        .select(
            [
                "condition_id",
                "residue_name",
                "is_wire",
                "failure_reason",
                "residue_idx",
                pl.col("mean_abs_load").alias("load_value"),
                pl.col("temporal_correlation").alias("kcc_value"),
                "coherence_score",
                "coherence_class",
                "residue_betweenness",
                "anchor_fraction",
                "mean_causal_lag",
                "force_angle_to_pathway_deg",
            ]
        )
        .sort(["residue_name", "condition_id"])
    )


def critical_endpoint_rows(paths: Inputs, stats: pl.DataFrame, lag_stats: pl.DataFrame, mapping: pl.DataFrame, betweenness: pl.DataFrame) -> pl.DataFrame:
    edges = pl.read_parquet(paths.edge_validation)
    endpoint_rows = endpoint_residue_frame(edges)
    return (
        endpoint_rows.join(mapping, on=["condition_id", "residue_idx"], how="left")
        .join(stats, on=["condition_id", "residue_idx"], how="left")
        .join(lag_stats, on=["condition_id", "residue_idx"], how="left")
        .join(betweenness, on=["condition_id", "residue_idx"], how="left")
        .with_columns(
            pl.when(pl.col("edge_class") == "pocket_vector")
            .then(pl.lit("pocket_vector"))
            .otherwise(pl.lit("downstream_lock"))
            .alias("role")
        )
        .with_columns(
            pl.when(pl.col("role") == "pocket_vector")
            .then(0.0)
            .otherwise(1.0)
            .alias("spatial_order")
        )
        .select(
            [
                "condition_id",
                "residue_idx",
                "residue_name",
                "role",
                "spatial_order",
                "mean_first_break_ps",
                "std_first_break_ps",
                "mean_causal_lag",
                "std_causal_lag",
            ]
        )
        .unique()
    )


def spearman(values_x: list[float], values_y: list[float]) -> tuple[float, float]:
    if len(values_x) < 3 or len(set(values_x)) < 2 or len(set(values_y)) < 2:
        return (float("nan"), float("nan"))
    rho, pval = spearmanr(values_x, values_y)
    return (float(rho), float(pval))


def temporal_cascade(
    paths: Inputs,
    stats: pl.DataFrame,
    lag_stats: pl.DataFrame,
    mapping: pl.DataFrame,
    betweenness: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, float]]:
    endpoints = critical_endpoint_rows(paths, stats, lag_stats, mapping, betweenness)
    wire = (
        pl.read_parquet(paths.pathway)
        .join(stats, on=["condition_id", "residue_idx"], how="left")
        .join(lag_stats, on=["condition_id", "residue_idx"], how="left")
        .select(
            [
                "condition_id",
                "residue_idx",
                "residue_name",
                pl.lit("wire").alias("role"),
                pl.col("betweenness").alias("spatial_order"),
                "mean_first_break_ps",
                "std_first_break_ps",
                "mean_causal_lag",
                "std_causal_lag",
            ]
        )
    )
    cascade = (
        pl.concat([endpoints, wire], how="diagonal")
        .sort(["condition_id", "mean_first_break_ps"])
        .with_columns(
            [
                pl.col("mean_first_break_ps").rank("ordinal").over("condition_id").cast(pl.UInt8).alias("temporal_rank"),
                pl.col("mean_causal_lag").rank("ordinal").over("condition_id").cast(pl.UInt8).alias("causal_rank"),
            ]
        )
    )
    usable = cascade.filter(pl.col("spatial_order").is_not_null() & pl.col("mean_first_break_ps").is_not_null() & pl.col("mean_causal_lag").is_not_null())
    temporal_rho, temporal_p = spearman(
        [as_float(v, "spatial_order") for v in usable["spatial_order"].to_list()],
        [as_float(v, "mean_first_break_ps") for v in usable["mean_first_break_ps"].to_list()],
    )
    causal_rho, causal_p = spearman(
        [as_float(v, "spatial_order") for v in usable["spatial_order"].to_list()],
        [as_float(v, "mean_causal_lag") for v in usable["mean_causal_lag"].to_list()],
    )
    start = as_float(usable["mean_first_break_ps"].min(), "min break")
    end = as_float(usable["mean_first_break_ps"].max(), "max break")
    temporal_span = end - start
    distance = 1.0
    velocity = float("nan") if temporal_span < EPSILON else distance / temporal_span
    metrics = {
        "temporal_spearman_rho": temporal_rho,
        "temporal_spearman_p": temporal_p,
        "causal_lag_spearman_rho": causal_rho,
        "causal_lag_spearman_p": causal_p,
        "propagation_velocity_A_per_ps": velocity,
    }
    return (
        cascade.with_columns(
            [
                pl.lit(temporal_rho).alias("spearman_rho"),
                pl.lit(temporal_p).alias("spearman_p_value"),
                pl.lit(causal_rho).alias("causal_lag_spearman_rho"),
                pl.lit(causal_p).alias("causal_lag_spearman_p_value"),
                pl.lit(velocity).alias("propagation_velocity_A_per_ps"),
            ]
        ),
        metrics,
    )


def gate_metrics(
    enriched_edges: pl.DataFrame,
    enriched_pathway: pl.DataFrame,
    wire_conservation: pl.DataFrame,
    cascade_metrics: dict[str, float],
) -> dict[str, object]:
    asn182_conditions = wire_conservation.filter((pl.col("residue_name") == "ASN182") & pl.col("is_wire")).height
    force_aligned = enriched_pathway.filter(pl.col("force_angle_to_pathway_deg") < 45.0).height
    edge_anchor_support = enriched_edges.filter(pl.col("anchor_fraction") > 0.1).height
    temporal_rho = cascade_metrics["temporal_spearman_rho"]
    causal_rho = cascade_metrics["causal_lag_spearman_rho"]
    return {
        "temporal_cascade": {
            "rho": temporal_rho,
            "p_value": cascade_metrics["temporal_spearman_p"],
            "passed": temporal_rho > 0.0,
            "reason": "insufficient_nonconstant_first_break_signal" if math.isnan(temporal_rho) else "computed",
        },
        "causal_lag_monotonicity": {
            "rho": causal_rho,
            "p_value": cascade_metrics["causal_lag_spearman_p"],
            "passed": causal_rho > 0.0,
            "reason": "causal_lag_source_values_all_nan_or_constant" if math.isnan(causal_rho) else "computed",
        },
        "wire_conservation": {
            "asn182_wire_conditions": asn182_conditions,
            "passed": asn182_conditions >= 3,
        },
        "force_alignment": {
            "aligned_wire_residues_lt45deg": force_aligned,
            "total_wire_residues": enriched_pathway.height,
            "passed": force_aligned >= 3,
        },
        "signal_quality": {
            "critical_edges_anchor_fraction_gt_0_1": edge_anchor_support,
            "total_edges": enriched_edges.height,
            "passed": edge_anchor_support >= 5,
        },
    }


def ledger_gate_status(gates: dict[str, object]) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for name, raw_metrics in gates.items():
        metrics = json_object(raw_metrics, name)
        status[name] = bool(metrics.get("passed", False))
    return status


def main() -> int:
    args = parse_args()
    log = configure_logging(str(args.log_level))
    inputs = input_paths(args)
    outputs = output_paths(args)
    topologies = topology_products(inputs.grid_mapping)
    mapping = residue_mapping(inputs.mapping)
    stats = residue_stats(inputs)
    lag_stats = kcc_stats(inputs.kcc)
    steering = steering_convergence(inputs, mapping)
    axes = pathway_axes(inputs.risk_map, topologies.residue_coordinates)
    betweenness = residue_betweenness(topologies.residue_coordinates, axes)
    force_alignment = mechanical_force_alignment(inputs, topologies, axes)
    enriched_edges = enrich_edges(inputs, stats, lag_stats, steering, force_alignment)
    enriched_pathway = enrich_pathway(inputs, stats, lag_stats, steering, force_alignment)
    conservation = wire_conservation_matrix(inputs, mapping, stats, lag_stats, force_alignment, axes, topologies)
    cascade, cascade_metrics = temporal_cascade(inputs, stats, lag_stats, mapping, betweenness)
    gates = gate_metrics(enriched_edges, enriched_pathway, conservation, cascade_metrics)
    source_paths = source_parquets(inputs)
    ledger_parameters: JsonObject = {
        "input_sha256": {path.name: sha256_path(path) for path in source_paths},
        "binding_site_reference": inputs.binding_site,
        "binding_site_reference_sha256": sha256_path(inputs.binding_site),
        "grid_mapping": inputs.grid_mapping,
        "grid_mapping_sha256": sha256_path(inputs.grid_mapping),
        "topology_sha256_by_condition": topologies.topology_hashes,
        "topology_files_by_condition": topologies.topology_files,
        "gate_metrics": gates,
        "formulas": {
            "anchor_fraction": "total_anchors/(total_spikes+epsilon)",
            "force_angle": "acos(abs(mean_force dot pathway_axis)/norm(mean_force))",
            "wire_conservation": "presence of current pathway residue names across all mapped conditions",
            "temporal_spearman": "spearman(spatial_order, mean_first_break_ps)",
            "causal_spearman": "spearman(spatial_order, mean_causal_lag)",
        },
    }
    for frame, path, schema, stage, keys in [
        (enriched_edges, outputs.enriched_edge, "enriched_edge_validation.v1", "enriched_edge_validation", ["condition_id", "edge_class"]),
        (enriched_pathway, outputs.enriched_pathway, "enriched_pathway_nodes.v1", "enriched_pathway_nodes", ["condition_id"]),
        (conservation, outputs.wire_conservation, "wire_conservation_matrix.v1", "wire_conservation_matrix", ["residue_name"]),
        (cascade, outputs.temporal_cascade, "temporal_cascade_summary.v1", "temporal_cascade_summary", ["condition_id", "role"]),
    ]:
        write_provenance_parquet(
            frame,
            path,
            producer_script=Path(__file__),
            source_parquets=source_paths,
            schema_version=schema,
            pipeline_stage=stage,
            partition_keys=keys,
            extra_metadata={"enrichment": "full-spectrum evidence fusion from existing parquets"},
            ledger_parameters=ledger_parameters,
            ledger_output_value={"rows": frame.height, "output_path": path},
            ledger_gate_status=ledger_gate_status(gates),
            repo_root=REPO_ROOT,
        )
    log.info(
        "enrichment_complete",
        enriched_edges=enriched_edges.height,
        enriched_pathway=enriched_pathway.height,
        conservation_rows=conservation.height,
        cascade_rows=cascade.height,
        gates=gates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
