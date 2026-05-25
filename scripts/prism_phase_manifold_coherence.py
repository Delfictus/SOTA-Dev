#!/usr/bin/env python3
"""Fuse all PRISM extraction channels into a residue coherence tensor.

Derived formulas:
    ch_spike_entropy = -sum(p_stream_phase * ln(p_stream_phase)) / ln(20 * 5).
    ch_steering_coherence = steering_weight_mean * supporting_stream_count / 20,
        normalized by condition maximum.
    ch_load_directionality = abs(mean_signed_load) / max(mean_abs_load, epsilon).
    ch_kcc_coherence = abs(mean(temporal_correlation)) * (1 - cv), normalized.
    ch_bocpd_alignment = max stream count in any 500-frame BOCPD window / 20.
    ch_strain_coherence = 1.0 for coherent absence of DT reductions, otherwise
        0.5 * normalized strain density + 0.5 * BOCPD coincidence rate.
    ch_aromatic_invariance = 1 - min(displacement_std / p95_displacement_std, 1).
    voxel_context = cKDTree classification of signal-grid voxels within 4 A of residue C-alpha.
    snr_quality = 1 - coefficient_of_variation(snr3_threshold).
    ch_residue_enrichment = residue spike density enrichment normalized by condition maximum.
    coherence_score = 0.7 * geometric_mean(channels) + 0.3 * conservative_min.
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

import numpy as np
import polars as pl
import structlog
from numpy.typing import NDArray
from scipy.spatial import cKDTree  # type: ignore[import-untyped]


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import sha256_path, write_provenance_parquet
from prism_dstw.ontology import ResidueIdx, StreamId


JsonObject: TypeAlias = dict[str, object]
Coordinate: TypeAlias = tuple[float, float, float]
FloatArray: TypeAlias = NDArray[np.float64]

N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
TRACK0_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation"
TOPOLOGY_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/topology/residue_index_mapping_matrix.parquet"

DEFAULT_OUTPUT = N80_DIR / "phase_manifold_coherence.parquet"
DEFAULT_EDGE_OUTPUT = N80_DIR / "phase_manifold_edge_validation.parquet"
DEFAULT_GRID_MAPPING = TRACK0_DIR / "grid_coordinate_mapping.json"
DEFAULT_CRITICAL_EDGES = TRACK0_DIR / "binding_site_reference.json"

EXPECTED_STREAMS = 20
BOCPD_WINDOW_SIZE = 500
VOXEL_RADIUS_A = 4.0
EPSILON = 1.0e-6
THERMAL_PHASES = (
    "phase_cold_hold",
    "phase_ramp_up",
    "phase_warm_hold",
    "phase_ramp_down",
    "phase_cold_return",
)


@dataclass(frozen=True)
class ChannelPaths:
    spike_events: Path
    stream_phase_counts: Path
    steering: Path
    mechanical_load: Path
    kcc: Path
    bocpd: Path
    kinetic_strain: Path
    aromatic: Path
    signal_grid: Path
    snr_masks: Path
    spike_residue_fields: Path
    protocol_state: Path
    channel_summary: Path
    risk_map: Path
    mapping: Path


@dataclass(frozen=True)
class TopologyProducts:
    atom_to_residue: pl.DataFrame
    residue_coordinates: pl.DataFrame
    aromatic_ring_map: pl.DataFrame
    topology_hashes: dict[str, str]
    topology_files: dict[str, str]


class BoundLogger(Protocol):
    def info(self, event: str, **kwargs: object) -> None:
        """Emit a structured info event."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spike-events", type=Path, default=N80_DIR / "spike_events_snr_masked.parquet")
    parser.add_argument("--stream-phase-counts", type=Path, default=N80_DIR / "stream_level_phase_counts.parquet")
    parser.add_argument("--steering", type=Path, default=N80_DIR / "autonomous_steering_tensor.parquet")
    parser.add_argument("--mechanical-load", type=Path, default=N80_DIR / "mechanical_load_network.parquet")
    parser.add_argument("--kcc", type=Path, default=N80_DIR / "kcc_residue_fields.parquet")
    parser.add_argument("--bocpd", type=Path, default=N80_DIR / "bocpd_survival_regimes.parquet")
    parser.add_argument("--kinetic-strain", type=Path, default=N80_DIR / "kinetic_strain_events.parquet")
    parser.add_argument("--aromatic", type=Path, default=N80_DIR / "aromatic_reorganization_tensor.parquet")
    parser.add_argument("--signal-grid", type=Path, default=N80_DIR / "signal_grid_variance_channel.parquet")
    parser.add_argument("--snr-masks", type=Path, default=N80_DIR / "stream_snr_masks.parquet")
    parser.add_argument("--spike-residue-fields", type=Path, default=N80_DIR / "spike_residue_fields.parquet")
    parser.add_argument("--protocol-state", type=Path, default=N80_DIR / "protocol_state_summary.parquet")
    parser.add_argument("--channel-summary", type=Path, default=N80_DIR / "receptor_durability_channel_summary.parquet")
    parser.add_argument("--risk-map", type=Path, default=N80_DIR / "receptor_durability_risk_map.parquet")
    parser.add_argument("--mapping", type=Path, default=TOPOLOGY_MAPPING)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--critical-edges-json", type=Path, default=DEFAULT_CRITICAL_EDGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edge-validation-out", type=Path, default=DEFAULT_EDGE_OUTPUT)
    parser.add_argument("--expected-streams", type=int, default=EXPECTED_STREAMS)
    parser.add_argument("--bocpd-window-size", type=int, default=BOCPD_WINDOW_SIZE)
    parser.add_argument("--voxel-radius-A", type=float, default=VOXEL_RADIUS_A)
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
    return cast(BoundLogger, structlog.get_logger("prism_phase_manifold_coherence"))


def channel_paths(args: argparse.Namespace) -> ChannelPaths:
    return ChannelPaths(
        spike_events=cast(Path, args.spike_events),
        stream_phase_counts=cast(Path, args.stream_phase_counts),
        steering=cast(Path, args.steering),
        mechanical_load=cast(Path, args.mechanical_load),
        kcc=cast(Path, args.kcc),
        bocpd=cast(Path, args.bocpd),
        kinetic_strain=cast(Path, args.kinetic_strain),
        aromatic=cast(Path, args.aromatic),
        signal_grid=cast(Path, args.signal_grid),
        snr_masks=cast(Path, args.snr_masks),
        spike_residue_fields=cast(Path, args.spike_residue_fields),
        protocol_state=cast(Path, args.protocol_state),
        channel_summary=cast(Path, args.channel_summary),
        risk_map=cast(Path, args.risk_map),
        mapping=cast(Path, args.mapping),
    )


def source_parquets(paths: ChannelPaths) -> list[Path]:
    return [
        paths.spike_events,
        paths.stream_phase_counts,
        paths.steering,
        paths.mechanical_load,
        paths.kcc,
        paths.bocpd,
        paths.kinetic_strain,
        paths.aromatic,
        paths.signal_grid,
        paths.snr_masks,
        paths.spike_residue_fields,
        paths.protocol_state,
        paths.channel_summary,
        paths.risk_map,
        paths.mapping,
    ]


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    return lazy_frame.collect(engine="streaming")


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


def stream_value(value: object, label: str) -> int:
    return int(StreamId(as_int(value, label)))


def require_columns(path: Path, required: set[str]) -> None:
    schema_names = set(pl.scan_parquet(path).collect_schema().names())
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def first_existing_column(path: Path, options: tuple[str, ...]) -> str:
    schema_names = set(pl.scan_parquet(path).collect_schema().names())
    for option in options:
        if option in schema_names:
            return option
    raise ValueError(f"{path} has none of the expected columns: {options}")


def float_list(value: object, label: str) -> list[float]:
    return [as_float(item, f"{label}[]") for item in json_list(value, label)]


def int_list(value: object, label: str) -> list[int]:
    return [as_int(item, f"{label}[]") for item in json_list(value, label)]


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


def topology_products(grid_mapping_path: Path) -> TopologyProducts:
    atom_rows: list[dict[str, int | str]] = []
    coord_rows: list[dict[str, int | float | str]] = []
    aromatic_rows: list[dict[str, int | str]] = []
    topology_hashes: dict[str, str] = {}
    topology_files: dict[str, str] = {}
    for condition_id, topology_path in topology_paths_from_grid_mapping(grid_mapping_path).items():
        topology = load_json_object(topology_path)
        residue_ids = int_list(topology.get("residue_ids"), f"{topology_path.name}:residue_ids")
        ca_indices = int_list(topology.get("ca_indices"), f"{topology_path.name}:ca_indices")
        positions = float_list(topology.get("positions"), f"{topology_path.name}:positions")
        topology_hashes[condition_id] = sha256_path(topology_path)
        topology_files[condition_id] = topology_path.name
        for atom_idx, residue_idx in enumerate(residue_ids):
            atom_rows.append(
                {
                    "condition_id": condition_id,
                    "atom_idx": atom_idx,
                    "residue_idx": residue_value(residue_idx, "residue_idx"),
                }
            )
        for residue_idx, ca_atom_idx in enumerate(ca_indices):
            ca = coordinate_at(positions, ca_atom_idx)
            coord_rows.append(
                {
                    "condition_id": condition_id,
                    "residue_idx": residue_value(residue_idx, "residue_idx"),
                    "ca_x": ca[0],
                    "ca_y": ca[1],
                    "ca_z": ca[2],
                }
            )
        for ring_idx, raw_target in enumerate(json_list(topology.get("aromatic_targets"), f"{topology_path.name}:aromatics")):
            target = json_object(raw_target, f"{topology_path.name}:aromatics[]")
            aromatic_rows.append(
                {
                    "condition_id": condition_id,
                    "ring_idx": ring_idx,
                    "residue_idx": residue_value(target["residue_idx"], "aromatic residue_idx"),
                }
            )
    return TopologyProducts(
        atom_to_residue=pl.DataFrame(atom_rows),
        residue_coordinates=pl.DataFrame(coord_rows),
        aromatic_ring_map=pl.DataFrame(aromatic_rows),
        topology_hashes=topology_hashes,
        topology_files=topology_files,
    )


def base_residue_frame(mapping_path: Path) -> pl.DataFrame:
    require_columns(mapping_path, {"condition_id", "residue_idx", "canonical_residue_label"})
    return (
        pl.scan_parquet(mapping_path)
        .select(
            [
                "condition_id",
                pl.col("residue_idx").cast(pl.Int32),
                pl.col("canonical_residue_label").alias("residue_name"),
            ]
        )
        .sort(["condition_id", "residue_idx"])
        .collect()
    )


def normalize_by_condition(frame: pl.DataFrame, raw_col: str, out_col: str) -> pl.DataFrame:
    return (
        frame.lazy()
        .with_columns(pl.col(raw_col).max().over("condition_id").alias("__condition_max"))
        .with_columns(
            pl.when(pl.col("__condition_max") > 0.0)
            .then((pl.col(raw_col) / pl.col("__condition_max")).clip(0.0, 1.0))
            .otherwise(0.0)
            .alias(out_col)
        )
        .drop("__condition_max")
        .collect()
    )


def spike_entropy_channel(path: Path, expected_streams: int) -> pl.DataFrame:
    require_columns(path, {"condition_id", "primary_residue_idx", "stream_id", "thermal_phase", "spike_count"})
    max_entropy = math.log(float(expected_streams * len(THERMAL_PHASES)))
    stream_counts = collect_streaming(
        pl.scan_parquet(path)
        .select(
            [
                "condition_id",
                pl.col("primary_residue_idx").cast(pl.Int32).alias("residue_idx"),
                pl.col("stream_id").cast(pl.UInt8),
                "thermal_phase",
                pl.col("spike_count").cast(pl.UInt64),
            ]
        )
        .filter(pl.col("thermal_phase").is_in(list(THERMAL_PHASES)))
        .group_by(["condition_id", "residue_idx", "stream_id", "thermal_phase"])
        .agg(pl.col("spike_count").sum().cast(pl.UInt64).alias("spike_count"))
    )
    return (
        stream_counts.lazy()
        .with_columns(pl.col("spike_count").sum().over(["condition_id", "residue_idx"]).alias("total_spikes"))
        .with_columns(
            pl.when((pl.col("total_spikes") > 0) & (pl.col("spike_count") > 0))
            .then(pl.col("spike_count").cast(pl.Float64) / pl.col("total_spikes").cast(pl.Float64))
            .otherwise(0.0)
            .alias("p")
        )
        .with_columns(
            pl.when(pl.col("p") > 0.0)
            .then(-(pl.col("p") * pl.col("p").log()))
            .otherwise(0.0)
            .alias("entropy_component")
        )
        .group_by(["condition_id", "residue_idx"])
        .agg(
            [
                pl.col("entropy_component").sum().alias("entropy_raw"),
                pl.col("total_spikes").first().cast(pl.UInt64),
                pl.col("stream_id").filter(pl.col("spike_count") > 0).n_unique().cast(pl.UInt8).alias("n_active_streams"),
                pl.col("thermal_phase").filter(pl.col("spike_count") > 0).n_unique().cast(pl.UInt8).alias("n_active_phases"),
                pl.len().cast(pl.UInt16).alias("n_phase_stream_bins"),
                pl.col("spike_count").filter(pl.col("spike_count") > 0).count().cast(pl.UInt16).alias("n_active_phase_stream_bins"),
            ]
        )
        .with_columns(
            [
                (pl.col("entropy_raw") / max_entropy).clip(0.0, 1.0).alias("ch_spike_entropy"),
                (pl.col("n_active_phases").cast(pl.Float64) / float(len(THERMAL_PHASES))).alias("phase_coverage_ratio"),
            ]
        )
        .select(
            [
                "condition_id",
                "residue_idx",
                "ch_spike_entropy",
                "total_spikes",
                "n_active_streams",
                "n_active_phases",
                "n_phase_stream_bins",
                "n_active_phase_stream_bins",
                "phase_coverage_ratio",
                "entropy_raw",
            ]
        )
        .collect()
    )


def steering_channel(path: Path, expected_streams: int) -> pl.DataFrame:
    require_columns(path, {"condition_id", "residue_idx", "steering_weight_mean", "supporting_stream_count"})
    raw = (
        pl.scan_parquet(path)
        .select(
            [
                "condition_id",
                pl.col("residue_idx").cast(pl.Int32),
                (
                    pl.col("steering_weight_mean").cast(pl.Float64)
                    * (pl.col("supporting_stream_count").cast(pl.Float64) / float(expected_streams))
                ).alias("steering_raw"),
            ]
        )
        .group_by(["condition_id", "residue_idx"])
        .agg(pl.col("steering_raw").sum())
        .collect()
    )
    return normalize_by_condition(raw, "steering_raw", "ch_steering_coherence").select(
        ["condition_id", "residue_idx", "ch_steering_coherence"]
    )


def mechanical_channel(path: Path, atom_map: pl.DataFrame) -> pl.DataFrame:
    load_col = first_existing_column(path, ("load_dot_product", "mechanical_load", "mean_load_dot_product"))
    atom_col = first_existing_column(path, ("atom_index", "atom_idx"))
    require_columns(path, {"condition_id", "stream_id", load_col, atom_col})
    per_stream = collect_streaming(
        pl.scan_parquet(path)
        .select(
            [
                "condition_id",
                pl.col("stream_id").cast(pl.UInt8),
                pl.col(atom_col).cast(pl.Int32).alias("atom_idx"),
                pl.col(load_col).cast(pl.Float64).alias("mechanical_load"),
            ]
        )
        .join(atom_map.lazy(), on=["condition_id", "atom_idx"], how="inner")
        .group_by(["condition_id", "residue_idx", "stream_id"])
        .agg(pl.col("mechanical_load").mean().alias("stream_mean_load"))
    )
    return (
        per_stream.lazy()
        .group_by(["condition_id", "residue_idx"])
        .agg(
            [
                pl.col("stream_mean_load").mean().alias("mean_signed_load"),
                pl.col("stream_mean_load").abs().mean().alias("mean_abs_load"),
                pl.col("stream_mean_load").std().fill_null(0.0).alias("load_std"),
            ]
        )
        .with_columns(
            (
                pl.col("mean_signed_load").abs() / pl.max_horizontal([pl.col("mean_abs_load"), pl.lit(EPSILON)])
            )
            .clip(0.0, 1.0)
            .alias("ch_load_directionality")
        )
        .select(
            [
                "condition_id",
                "residue_idx",
                "ch_load_directionality",
                "mean_abs_load",
                "mean_signed_load",
                "load_std",
            ]
        )
        .collect()
    )


def kcc_channel(path: Path) -> pl.DataFrame:
    temporal_col = first_existing_column(path, ("temporal_correlation", "temporal_corr"))
    require_columns(path, {"condition_id", "stream_id", "residue_idx", temporal_col, "direction_score", "causal_lag"})
    per_stream = collect_streaming(
        pl.scan_parquet(path)
        .select(
            [
                "condition_id",
                pl.col("stream_id").cast(pl.UInt8),
                pl.col("residue_idx").cast(pl.Int32),
                pl.col(temporal_col).cast(pl.Float64).alias("temporal_correlation"),
                pl.col("direction_score").cast(pl.Float64),
                pl.col("causal_lag").cast(pl.Float64),
            ]
        )
        .group_by(["condition_id", "residue_idx", "stream_id"])
        .agg(
            [
                pl.col("temporal_correlation").mean(),
                pl.col("direction_score").mean(),
                pl.col("causal_lag").mean(),
            ]
        )
    )
    raw = (
        per_stream.lazy()
        .group_by(["condition_id", "residue_idx"])
        .agg(
            [
                pl.col("temporal_correlation").mean(),
                pl.col("temporal_correlation").std().fill_null(0.0).alias("temporal_std"),
                pl.col("direction_score").mean(),
                pl.col("causal_lag").mean(),
            ]
        )
        .with_columns(
            (pl.col("temporal_std") / pl.max_horizontal([pl.col("temporal_correlation").abs(), pl.lit(EPSILON)])).alias(
                "temporal_cv"
            )
        )
        .with_columns(
            (
                pl.col("temporal_correlation").abs()
                * (pl.lit(1.0) - pl.col("temporal_cv")).clip(0.0, 1.0)
            ).alias("kcc_raw")
        )
        .collect()
    )
    return normalize_by_condition(raw, "kcc_raw", "ch_kcc_coherence").select(
        [
            "condition_id",
            "residue_idx",
            "temporal_correlation",
            "direction_score",
            "causal_lag",
            "ch_kcc_coherence",
        ]
    )


def bocpd_channel(path: Path, expected_streams: int, window_size: int) -> pl.DataFrame:
    require_columns(path, {"condition_id", "stream_id", "frame_idx", "chunk_close_signal"})
    return (
        pl.scan_parquet(path)
        .filter(pl.col("chunk_close_signal"))
        .select(
            [
                "condition_id",
                pl.col("stream_id").cast(pl.UInt8),
                (pl.col("frame_idx").cast(pl.Int64) // window_size).alias("frame_window"),
            ]
        )
        .group_by(["condition_id", "frame_window"])
        .agg(pl.col("stream_id").n_unique().alias("streams_in_window"))
        .group_by("condition_id")
        .agg(pl.col("streams_in_window").max().fill_null(0).alias("max_streams_in_bocpd_window"))
        .with_columns(
            (pl.col("max_streams_in_bocpd_window").cast(pl.Float64) / float(expected_streams))
            .clip(0.0, 1.0)
            .alias("ch_bocpd_alignment")
        )
        .select(["condition_id", "ch_bocpd_alignment", "max_streams_in_bocpd_window"])
        .collect()
    )


def strain_channel(path: Path) -> pl.DataFrame:
    require_columns(path, {"condition_id", "stream_id", "chunk_idx", "dt_reduction_event"})
    coincidence_col = first_existing_column(path, ("dt_drop_coincident_with_regime_change", "coincides_with_bocpd_change"))
    raw = (
        pl.scan_parquet(path)
        .group_by("condition_id")
        .agg(
            [
                pl.col("stream_id").n_unique().alias("n_streams_with_strain_rows"),
                pl.col("chunk_idx").n_unique().alias("n_chunks_with_strain_rows"),
                pl.col("dt_reduction_event").cast(pl.UInt32).sum().alias("n_strain_events"),
                pl.col(coincidence_col).cast(pl.UInt32).sum().alias("n_coincident_strain_events"),
            ]
        )
        .with_columns(
            (
                pl.col("n_strain_events").cast(pl.Float64)
                / pl.max_horizontal(
                    [
                        (pl.col("n_streams_with_strain_rows") * pl.col("n_chunks_with_strain_rows")).cast(pl.Float64),
                        pl.lit(1.0),
                    ]
                )
            ).alias("strain_density")
        )
        .with_columns(
            (
                pl.col("n_coincident_strain_events").cast(pl.Float64)
                / pl.max_horizontal([pl.col("n_strain_events").cast(pl.Float64), pl.lit(1.0)])
            ).alias("bocpd_coincidence_rate")
        )
        .collect()
    )
    normalized = normalize_by_condition(raw, "strain_density", "strain_density_normalized")
    return (
        normalized.lazy()
        .with_columns(
            pl.when(pl.col("n_strain_events") == 0)
            .then(1.0)
            .otherwise(
                (
                    0.5 * pl.col("strain_density_normalized")
                    + 0.5 * pl.col("bocpd_coincidence_rate").clip(0.0, 1.0)
                ).clip(0.0, 1.0)
            )
            .alias("ch_strain_coherence")
        )
        .select(["condition_id", "ch_strain_coherence", "strain_density", "bocpd_coincidence_rate"])
        .collect()
    )


def aromatic_channel(path: Path, aromatic_map: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    require_columns(path, {"condition_id", "ring_idx", "centroid_displacement_std"})
    p95_value = (
        pl.scan_parquet(path)
        .select(pl.col("centroid_displacement_std").cast(pl.Float64).quantile(0.95).alias("p95"))
        .collect()
        .item()
    )
    d_max = max(as_float(p95_value, "aromatic p95"), EPSILON)
    frame = (
        pl.scan_parquet(path)
        .select(
            [
                "condition_id",
                pl.col("ring_idx").cast(pl.Int32),
                pl.col("centroid_displacement_std").cast(pl.Float64),
            ]
        )
        .join(aromatic_map.lazy(), on=["condition_id", "ring_idx"], how="inner")
        .with_columns((1.0 - (pl.col("centroid_displacement_std") / d_max).clip(0.0, 1.0)).alias("ring_invariance"))
        .group_by(["condition_id", "residue_idx"])
        .agg(pl.col("ring_invariance").mean().alias("ch_aromatic_invariance"))
        .collect()
    )
    return frame, d_max


def signal_grid_class_column(path: Path) -> str:
    return first_existing_column(path, ("variance_classification", "variance_class"))


def voxel_center_array(frame: pl.DataFrame, origin: Coordinate, spacing: float) -> FloatArray:
    return np.column_stack(
        [
            origin[0] + (frame["x_idx"].to_numpy().astype(np.float64) + 0.5) * spacing,
            origin[1] + (frame["y_idx"].to_numpy().astype(np.float64) + 0.5) * spacing,
            origin[2] + (frame["z_idx"].to_numpy().astype(np.float64) + 0.5) * spacing,
        ]
    )


def voxel_context_from_counts(stable: int, destabilized: int, activated: int, void: int) -> str:
    total = stable + destabilized + activated + void
    if total <= 0:
        return "surface"
    stable_fraction = stable / total
    destabilized_fraction = destabilized / total
    activated_fraction = activated / total
    void_fraction = void / total
    if stable_fraction > 0.7:
        return "core"
    if destabilized_fraction > 0.3:
        return "interface"
    if activated_fraction > 0.3:
        return "dynamic"
    if void_fraction > 0.5:
        return "surface"
    return "mixed"


def grid_geometries(path: Path) -> dict[str, tuple[Coordinate, float]]:
    payload = load_json_object(path)
    raw_conditions = json_object(payload.get("conditions"), f"{path}:conditions")
    out: dict[str, tuple[Coordinate, float]] = {}
    for condition_id, raw_geometry in raw_conditions.items():
        geometry = json_object(raw_geometry, f"{path}:conditions[{condition_id}]")
        origin_values = json_list(geometry.get("origin_xyz_angstrom"), f"{path}:origin")
        if len(origin_values) != 3:
            raise ValueError(f"{path}:origin must contain three values")
        origin = (
            as_float(origin_values[0], "origin[0]"),
            as_float(origin_values[1], "origin[1]"),
            as_float(origin_values[2], "origin[2]"),
        )
        out[str(condition_id)] = (origin, as_float(geometry["spacing_angstrom"], "spacing_angstrom"))
    return out


def voxel_context_channel(
    signal_grid_path: Path,
    grid_mapping_path: Path,
    residue_coordinates: pl.DataFrame,
    radius_a: float,
) -> pl.DataFrame:
    class_col = signal_grid_class_column(signal_grid_path)
    geometries = grid_geometries(grid_mapping_path)
    rows: list[dict[str, int | float | str]] = []
    conditions = sorted(residue_coordinates["condition_id"].unique().to_list())
    for condition_id in conditions:
        if condition_id not in geometries:
            continue
        origin, spacing = geometries[str(condition_id)]
        grid = (
            pl.scan_parquet(signal_grid_path)
            .filter(pl.col("condition_id") == condition_id)
            .select(
                [
                    "voxel_idx",
                    "x_idx",
                    "y_idx",
                    "z_idx",
                    pl.col(class_col).alias("variance_class"),
                ]
            )
            .collect()
        )
        centers = voxel_center_array(grid, origin, spacing)
        classes = [str(item) for item in grid["variance_class"].to_list()]
        tree = cKDTree(centers)
        residue_rows = residue_coordinates.filter(pl.col("condition_id") == condition_id)
        for residue in residue_rows.iter_rows(named=True):
            point = (
                as_float(residue["ca_x"], "ca_x"),
                as_float(residue["ca_y"], "ca_y"),
                as_float(residue["ca_z"], "ca_z"),
            )
            indices = tree.query_ball_point(point, r=radius_a)
            stable = 0
            destabilized = 0
            activated = 0
            void = 0
            for idx in indices:
                cls = classes[int(idx)]
                if cls == "stable_occupied":
                    stable += 1
                elif cls == "thermally_destabilized":
                    destabilized += 1
                elif cls == "thermally_activated":
                    activated += 1
                else:
                    void += 1
            total = max(stable + destabilized + activated + void, 1)
            rows.append(
                {
                    "condition_id": str(condition_id),
                    "residue_idx": residue_value(residue["residue_idx"], "residue_idx"),
                    "voxel_context": voxel_context_from_counts(stable, destabilized, activated, void),
                    "voxel_shell_count": total,
                    "voxel_stable_fraction": stable / total,
                    "voxel_destabilized_fraction": destabilized / total,
                    "voxel_activated_fraction": activated / total,
                    "voxel_void_fraction": void / total,
                }
            )
    return pl.DataFrame(rows)


def snr_channel(path: Path) -> pl.DataFrame:
    require_columns(path, {"condition_id", "snr3_threshold"})
    return (
        pl.scan_parquet(path)
        .group_by("condition_id")
        .agg(
            [
                pl.col("snr3_threshold").mean().alias("snr_threshold_mean"),
                pl.col("snr3_threshold").std().fill_null(0.0).alias("snr_threshold_std"),
            ]
        )
        .with_columns(
            (
                1.0
                - (
                    pl.col("snr_threshold_std")
                    / pl.max_horizontal([pl.col("snr_threshold_mean").abs(), pl.lit(EPSILON)])
                )
            )
            .clip(0.0, 1.0)
            .alias("snr_quality")
        )
        .select(["condition_id", "snr_quality", "snr_threshold_mean", "snr_threshold_std"])
        .collect()
    )


def residue_enrichment_channel(path: Path) -> pl.DataFrame:
    require_columns(path, {"condition_id", "residue_idx", "masked_spike_count"})
    raw = (
        pl.scan_parquet(path)
        .select(
            [
                "condition_id",
                pl.col("residue_idx").cast(pl.Int32),
                pl.col("masked_spike_count").cast(pl.Float64),
            ]
        )
        .group_by(["condition_id", "residue_idx"])
        .agg(pl.col("masked_spike_count").mean().alias("mean_residue_spike_count"))
        .with_columns(pl.col("mean_residue_spike_count").mean().over("condition_id").alias("condition_mean_spike_count"))
        .with_columns(
            (
                pl.col("mean_residue_spike_count")
                / pl.max_horizontal([pl.col("condition_mean_spike_count"), pl.lit(EPSILON)])
            ).alias("enrichment_raw")
        )
        .collect()
    )
    return normalize_by_condition(raw, "enrichment_raw", "ch_residue_enrichment").select(
        ["condition_id", "residue_idx", "ch_residue_enrichment", "mean_residue_spike_count"]
    )


def fill_channel_defaults(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        [
            pl.col("ch_spike_entropy").fill_null(0.0),
            pl.col("ch_steering_coherence").fill_null(0.0),
            pl.col("ch_load_directionality").fill_null(0.0),
            pl.col("ch_kcc_coherence").fill_null(0.0),
            pl.col("ch_bocpd_alignment").fill_null(0.0),
            pl.col("ch_strain_coherence").fill_null(0.0),
            pl.col("ch_residue_enrichment").fill_null(0.0),
            pl.col("total_spikes").fill_null(0).cast(pl.UInt64),
            pl.col("n_active_streams").fill_null(0).cast(pl.UInt8),
            pl.col("n_active_phases").fill_null(0).cast(pl.UInt8),
            pl.col("n_phase_stream_bins").fill_null(0).cast(pl.UInt16),
            pl.col("n_active_phase_stream_bins").fill_null(0).cast(pl.UInt16),
            pl.col("phase_coverage_ratio").fill_null(0.0),
            pl.col("snr_quality").fill_null(0.0),
            pl.col("voxel_context").fill_null("surface"),
        ]
    )


def fusion_frame(frame: pl.DataFrame) -> pl.DataFrame:
    score_cols = [
        "spike_weighted",
        "ch_steering_coherence",
        "ch_load_directionality",
        "ch_kcc_coherence",
        "ch_bocpd_alignment",
        "ch_strain_coherence",
        "ch_residue_enrichment",
        "ch_aromatic_invariance",
    ]
    context_expr = (
        pl.when(pl.col("voxel_context") == "core")
        .then(0.5)
        .when(pl.col("voxel_context") == "interface")
        .then(1.0)
        .when(pl.col("voxel_context") == "dynamic")
        .then(1.5)
        .when(pl.col("voxel_context") == "surface")
        .then(0.7)
        .otherwise(1.0)
    )
    log_terms = [
        pl.when(pl.col(col).is_not_null())
        .then(pl.max_horizontal([pl.col(col), pl.lit(EPSILON)]).log())
        .otherwise(0.0)
        for col in score_cols
    ]
    count_terms = [
        pl.when(pl.col(col).is_not_null()).then(1).otherwise(0)
        for col in score_cols
    ]
    concordant_terms = [
        pl.when((pl.col(col).is_not_null()) & (pl.col(col) > 0.5)).then(1).otherwise(0)
        for col in score_cols
    ]
    return (
        frame.lazy()
        .with_columns((pl.col("ch_spike_entropy") * pl.col("snr_quality")).alias("spike_weighted"))
        .with_columns(
            [
                pl.sum_horizontal(log_terms).alias("__log_sum"),
                pl.sum_horizontal(count_terms).cast(pl.UInt8).alias("n_total_channels"),
                pl.sum_horizontal(concordant_terms).cast(pl.UInt8).alias("n_concordant_channels"),
                context_expr.alias("context_weight"),
            ]
        )
        .with_columns((pl.col("__log_sum") / pl.col("n_total_channels").cast(pl.Float64)).exp().alias("geometric_mean"))
        .with_columns(
            pl.min_horizontal(
                [
                    "spike_weighted",
                    "ch_load_directionality",
                    "ch_kcc_coherence",
                    "ch_bocpd_alignment",
                ]
            ).alias("conservative_min")
        )
        .with_columns(
            (
                0.7 * pl.col("geometric_mean")
                + 0.3 * pl.col("conservative_min")
            )
            .clip(0.0, 1.0)
            .alias("coherence_score")
        )
        .with_columns(
            [
                (pl.col("coherence_score") * pl.col("context_weight")).alias("informativeness"),
                (
                    pl.col("n_concordant_channels").cast(pl.Float64)
                    / pl.col("n_total_channels").cast(pl.Float64)
                ).alias("concordance_ratio"),
            ]
        )
        .with_columns(
            pl.when(pl.col("n_total_channels") < 4)
            .then(pl.lit("insufficient_channels"))
            .when((pl.col("concordance_ratio") >= 0.8) & (pl.col("coherence_score") > 0.6))
            .then(pl.lit("fiber_invariant"))
            .when((pl.col("concordance_ratio") >= 0.5) & (pl.col("coherence_score") > 0.3))
            .then(pl.lit("partially_coherent"))
            .when(pl.col("concordance_ratio") < 0.3)
            .then(pl.lit("thermally_activated"))
            .otherwise(pl.lit("mixed_signal"))
            .alias("coherence_class")
        )
        .drop("__log_sum")
        .collect()
    )


def critical_edges_from_json(path: Path) -> pl.DataFrame:
    payload = load_json_object(path)
    rows: list[dict[str, int | float | str]] = []
    for raw_edge in json_list(payload.get("critical_edges"), f"{path}:critical_edges"):
        edge = json_object(raw_edge, f"{path}:critical_edges[]")
        rows.append(
            {
                "edge_id": str(edge["edge_id"]),
                "condition_id": str(edge["condition_id"]),
                "edge_label": str(edge["edge_label"]),
                "edge_class": str(edge["edge_class"]),
                "edge_from_residue": residue_value(edge["edge_from_residue"], "edge_from_residue"),
                "edge_to_residue": residue_value(edge["edge_to_residue"], "edge_to_residue"),
                "durability_risk_score_raw": as_float(edge["durability_risk_score_raw"], "durability_risk_score_raw"),
            }
        )
    return pl.DataFrame(rows)


def edge_validation_frame(edges: pl.DataFrame, coherence: pl.DataFrame) -> pl.DataFrame:
    endpoint_cols = [
        "coherence_score",
        "coherence_class",
        "n_concordant_channels",
        "n_total_channels",
        "concordance_ratio",
        "ch_spike_entropy",
        "ch_load_directionality",
        "ch_kcc_coherence",
        "ch_bocpd_alignment",
        "voxel_context",
        "total_spikes",
    ]
    from_frame = coherence.select(
        ["condition_id", pl.col("residue_idx").alias("edge_from_residue")]
        + [pl.col(col).alias(f"from_{col}") for col in endpoint_cols]
    )
    to_frame = coherence.select(
        ["condition_id", pl.col("residue_idx").alias("edge_to_residue")]
        + [pl.col(col).alias(f"to_{col}") for col in endpoint_cols]
    )
    return (
        edges.with_columns(
            [
                pl.col("edge_from_residue").cast(pl.Int32),
                pl.col("edge_to_residue").cast(pl.Int32),
            ]
        )
        .lazy()
        .join(from_frame.lazy(), on=["condition_id", "edge_from_residue"], how="left")
        .join(to_frame.lazy(), on=["condition_id", "edge_to_residue"], how="left")
        .with_columns(pl.min_horizontal(["from_coherence_score", "to_coherence_score"]).alias("edge_coherence_score"))
        .with_columns(
            pl.when(
                pl.col("from_coherence_class").is_in(["fiber_invariant", "partially_coherent"])
                & pl.col("to_coherence_class").is_in(["fiber_invariant", "partially_coherent"])
                & (pl.col("edge_coherence_score") >= 0.45)
            )
            .then(pl.lit("validated_constitutive"))
            .when(
                (pl.col("from_coherence_class") == "thermally_activated")
                & (pl.col("to_coherence_class") == "thermally_activated")
            )
            .then(pl.lit("divergent_artifact_warning"))
            .when(
                (pl.col("from_coherence_class") == "thermally_activated")
                | (pl.col("to_coherence_class") == "thermally_activated")
            )
            .then(pl.lit("divergent_artifact_warning"))
            .otherwise(pl.lit("partial_validation"))
            .alias("validation_status")
        )
        .sort(["edge_class", "durability_risk_score_raw"], descending=[False, True])
        .collect()
    )


def build_coherence(paths: ChannelPaths, args: argparse.Namespace, topologies: TopologyProducts) -> tuple[pl.DataFrame, dict[str, object]]:
    expected_streams = int(args.expected_streams)
    base = base_residue_frame(paths.mapping)
    spike = spike_entropy_channel(paths.stream_phase_counts, expected_streams)
    steering = steering_channel(paths.steering, expected_streams)
    mechanical = mechanical_channel(paths.mechanical_load, topologies.atom_to_residue)
    kcc = kcc_channel(paths.kcc)
    bocpd = bocpd_channel(paths.bocpd, expected_streams, int(args.bocpd_window_size))
    strain = strain_channel(paths.kinetic_strain)
    aromatic, aromatic_dmax = aromatic_channel(paths.aromatic, topologies.aromatic_ring_map)
    voxel_context = voxel_context_channel(paths.signal_grid, cast(Path, args.grid_mapping), topologies.residue_coordinates, float(args.voxel_radius_A))
    snr = snr_channel(paths.snr_masks)
    enrichment = residue_enrichment_channel(paths.spike_residue_fields)
    joined = (
        base.lazy()
        .join(spike.lazy(), on=["condition_id", "residue_idx"], how="left")
        .join(steering.lazy(), on=["condition_id", "residue_idx"], how="left")
        .join(mechanical.lazy(), on=["condition_id", "residue_idx"], how="left")
        .join(kcc.lazy(), on=["condition_id", "residue_idx"], how="left")
        .join(bocpd.lazy(), on="condition_id", how="left")
        .join(strain.lazy(), on="condition_id", how="left")
        .join(aromatic.lazy(), on=["condition_id", "residue_idx"], how="left")
        .join(voxel_context.lazy(), on=["condition_id", "residue_idx"], how="left")
        .join(snr.lazy(), on="condition_id", how="left")
        .join(enrichment.lazy(), on=["condition_id", "residue_idx"], how="left")
        .collect()
    )
    filled = fill_channel_defaults(joined)
    fused = fusion_frame(filled).sort(["condition_id", "residue_idx"])
    constants: dict[str, object] = {
        "expected_streams": expected_streams,
        "thermal_phases": list(THERMAL_PHASES),
        "spike_entropy_source": paths.stream_phase_counts.name,
        "bocpd_window_size": int(args.bocpd_window_size),
        "voxel_radius_A": float(args.voxel_radius_A),
        "aromatic_p95_displacement_std": aromatic_dmax,
        "epsilon": EPSILON,
        "context_weights": {"core": 0.5, "interface": 1.0, "dynamic": 1.5, "surface": 0.7, "mixed": 1.0},
    }
    return fused, constants


def main() -> int:
    args = parse_args()
    log = configure_logging(str(args.log_level))
    paths = channel_paths(args)
    topologies = topology_products(cast(Path, args.grid_mapping))
    log.info("coherence_extraction_started", output=str(args.output))
    coherence, constants = build_coherence(paths, args, topologies)
    critical_edges = critical_edges_from_json(cast(Path, args.critical_edges_json))
    edge_validation = edge_validation_frame(critical_edges, coherence)
    ledger_parameters: JsonObject = {
        "normalization_constants": constants,
        "topology_sha256_by_condition": topologies.topology_hashes,
        "topology_files_by_condition": topologies.topology_files,
        "input_sha256": {path.name: sha256_path(path) for path in source_parquets(paths)},
        "fusion_formula": "coherence_score=0.7*geometric_mean+0.3*conservative_min",
        "classification": {
            "fiber_invariant": "concordance_ratio >= 0.8 and coherence_score > 0.6",
            "partially_coherent": "concordance_ratio >= 0.5 and coherence_score > 0.3",
            "thermally_activated": "concordance_ratio < 0.3",
        },
        "no_strain_event_semantics": "condition-wide zero DT reductions is coherent absence of kinetic strain, not incoherence",
    }
    write_provenance_parquet(
        coherence,
        cast(Path, args.output),
        producer_script=Path(__file__),
        source_parquets=source_parquets(paths),
        schema_version="phase_manifold_coherence.v1",
        pipeline_stage="phase_manifold_coherence",
        partition_keys=["condition_id"],
        extra_metadata={"fusion_formula": "0.7*geometric_mean+0.3*conservative_min"},
        ledger_parameters=ledger_parameters,
        ledger_output_value={"rows": coherence.height, "output_path": cast(Path, args.output)},
        repo_root=REPO_ROOT,
    )
    write_provenance_parquet(
        edge_validation,
        cast(Path, args.edge_validation_out),
        producer_script=Path(__file__),
        source_parquets=source_parquets(paths),
        schema_version="phase_manifold_edge_validation.v1",
        pipeline_stage="phase_manifold_edge_validation",
        partition_keys=["condition_id", "edge_class"],
        extra_metadata={"edge_formula": "min endpoint coherence score"},
        ledger_parameters=ledger_parameters,
        ledger_output_value={"rows": edge_validation.height, "output_path": cast(Path, args.edge_validation_out)},
        repo_root=REPO_ROOT,
    )
    log.info(
        "coherence_extraction_complete",
        rows=coherence.height,
        fiber_invariant=coherence.filter(pl.col("coherence_class") == "fiber_invariant").height,
        edge_rows=edge_validation.height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
