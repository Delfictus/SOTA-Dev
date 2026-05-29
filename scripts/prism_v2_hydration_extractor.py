#!/usr/bin/env python3
"""Extract V2 hydration statistics from SNR-masked spike events.

This extractor is intentionally condition-partitioned. The source parquet can
be large enough that a global group-by over voxel_idx is unsafe, so each
condition_id is aggregated and appended as its own output row group.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "condition_id",
    "primary_residue_idx",
    "voxel_idx",
    "water_density",
    "wd_change",
    "intensity",
}

OUTPUT_COLUMNS = [
    "condition_id",
    "residue_id",
    "voxel_idx",
    "hydration_tunnel_id",
    "sigma_hyd",
    "solvent_wire_importance",
    "occlusion_risk",
]

INTEGRATION_COLUMNS = [
    "topology_region",
    "topology_region_source",
    "captured_tile_ids",
    "captured_tile_types",
    "captured_graph_tile_hashes",
    "captured_tile_topology_regions",
    "captured_tile_match_basis",
    "captured_tile_count",
    "topology_delta_hashes",
    "basin_delta_hashes",
    "restricted_operator_hashes",
    "c6_operator_hashes",
    "spectral_reward_event_source",
    "spectral_captured_replay_count",
    "spectral_gpu_solve_count",
    "spectral_cpu_solve_count",
    "spectral_reward_cache_hit_rate",
    "dstw_spectral_status",
    "dstw_integration_status",
    "dstw_context_evidence_paths",
]

CAPTURE_FIELDS = [
    "captured_tile_ids",
    "captured_tile_types",
    "captured_graph_tile_hashes",
    "captured_tile_topology_regions",
    "captured_tile_count",
    "topology_delta_hashes",
    "basin_delta_hashes",
    "restricted_operator_hashes",
    "c6_operator_hashes",
]

SPECTRAL_FIELDS = {
    "spectral_reward_event_source": pl.Utf8,
    "spectral_captured_replay_count": pl.Int64,
    "spectral_gpu_solve_count": pl.Int64,
    "spectral_cpu_solve_count": pl.Int64,
    "spectral_reward_cache_hit_rate": pl.Float64,
    "dstw_spectral_status": pl.Utf8,
}

BASE_OUTPUT_ARROW_FIELDS = [
    pa.field("condition_id", pa.large_string()),
    pa.field("residue_id", pa.int64()),
    pa.field("voxel_idx", pa.int64()),
    pa.field("hydration_tunnel_id", pa.large_string()),
    pa.field("sigma_hyd", pa.float64()),
    pa.field("solvent_wire_importance", pa.float64()),
    pa.field("occlusion_risk", pa.float64()),
]

INTEGRATION_ARROW_FIELDS = [
    pa.field("topology_region", pa.large_string()),
    pa.field("topology_region_source", pa.large_string()),
    pa.field("captured_tile_ids", pa.large_string()),
    pa.field("captured_tile_types", pa.large_string()),
    pa.field("captured_graph_tile_hashes", pa.large_string()),
    pa.field("captured_tile_topology_regions", pa.large_string()),
    pa.field("captured_tile_match_basis", pa.large_string()),
    pa.field("captured_tile_count", pa.int64()),
    pa.field("topology_delta_hashes", pa.large_string()),
    pa.field("basin_delta_hashes", pa.large_string()),
    pa.field("restricted_operator_hashes", pa.large_string()),
    pa.field("c6_operator_hashes", pa.large_string()),
    pa.field("spectral_reward_event_source", pa.large_string()),
    pa.field("spectral_captured_replay_count", pa.int64()),
    pa.field("spectral_gpu_solve_count", pa.int64()),
    pa.field("spectral_cpu_solve_count", pa.int64()),
    pa.field("spectral_reward_cache_hit_rate", pa.float64()),
    pa.field("dstw_spectral_status", pa.large_string()),
    pa.field("dstw_integration_status", pa.large_string()),
    pa.field("dstw_context_evidence_paths", pa.large_string()),
]

RESIDUE_NUMBER_RE = re.compile(r"(\d+)")


def output_columns(include_integration: bool) -> list[str]:
    if not include_integration:
        return list(OUTPUT_COLUMNS)
    return [*OUTPUT_COLUMNS, *INTEGRATION_COLUMNS]


def output_arrow_schema(include_integration: bool) -> pa.Schema:
    fields = [*BASE_OUTPUT_ARROW_FIELDS]
    if include_integration:
        fields.extend(INTEGRATION_ARROW_FIELDS)
    return pa.schema(fields)


OUTPUT_ARROW_SCHEMA = pa.schema(
    [
        *BASE_OUTPUT_ARROW_FIELDS,
    ]
)


@dataclass(frozen=True)
class IntegrationContext:
    topology_regions: pl.DataFrame | None = None
    captured_tiles: pl.DataFrame | None = None
    spectral_fields: dict[str, object] | None = None
    evidence_paths: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return (
            self.topology_regions is not None
            or self.captured_tiles is not None
            or bool(self.spectral_fields)
            or bool(self.evidence_paths)
        )


def emit_status(event: str, **fields: object) -> None:
    payload = {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        **fields,
    }
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stderr.flush()


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """Collect with the streaming engine, supporting older Polars call forms."""

    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:
        return lazy_frame.collect(streaming=True)


def validate_input_schema(event_parquet: Path) -> None:
    if not event_parquet.is_file():
        raise SystemExit(f"--event-parquet does not exist or is not a file: {event_parquet}")
    schema_names = set(pl.scan_parquet(str(event_parquet)).collect_schema().names())
    missing = sorted(REQUIRED_COLUMNS - schema_names)
    if missing:
        raise SystemExit(f"{event_parquet}: missing required V2 column(s): {missing}")


def base_event_scan(event_parquet: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(event_parquet)).select(
        [
            pl.col("condition_id").cast(pl.Utf8),
            pl.col("primary_residue_idx").cast(pl.Int64),
            pl.col("voxel_idx").cast(pl.Int64),
            pl.col("water_density").cast(pl.Float64),
            pl.col("wd_change").cast(pl.Float64),
            pl.col("intensity").cast(pl.Float64),
        ]
    )


def discover_condition_ids(event_parquet: Path) -> list[str]:
    frame = collect_streaming(
        pl.scan_parquet(str(event_parquet))
        .select(pl.col("condition_id").cast(pl.Utf8))
        .filter(pl.col("condition_id").is_not_null())
        .unique()
        .sort("condition_id")
    )
    return [str(value) for value in frame.get_column("condition_id").to_list()]


def residue_number(value: object) -> int | None:
    match = RESIDUE_NUMBER_RE.search(str(value))
    if match is None:
        return None
    number = int(match.group(1))
    return number if number > 0 else None


def require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} does not exist or is not a file: {path}")


def load_topology_region_registry(path: Path | None) -> pl.DataFrame | None:
    if path is None:
        return None
    require_existing_file(path, "--topology-region-registry")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "track_b.topology_region_registry.v1":
        raise SystemExit(f"{path}: unsupported topology registry schema_version={payload.get('schema_version')!r}")
    residue_regions: dict[int, set[str]] = {}
    for region, region_payload in dict(payload.get("regions") or {}).items():
        for residue in list(dict(region_payload).get("residues") or []):
            number = residue_number(dict(residue).get("residue_id"))
            if number is not None:
                residue_regions.setdefault(number, set()).add(str(region))
    rows = [
        {
            "residue_id": residue_id,
            "topology_region": "|".join(sorted(regions)),
            "topology_region_source": str(path),
        }
        for residue_id, regions in sorted(residue_regions.items())
    ]
    return pl.DataFrame(
        rows,
        schema={
            "residue_id": pl.Int64,
            "topology_region": pl.Utf8,
            "topology_region_source": pl.Utf8,
        },
        orient="row",
    )


def _joined(values: Iterable[object]) -> str:
    return ",".join(sorted({str(value) for value in values if value is not None and str(value) != ""}))


def load_captured_tile_registry(path: Path | None) -> pl.DataFrame | None:
    if path is None:
        return None
    require_existing_file(path, "--captured-tile-registry")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "prism.log_subtb.captured_tile_registry.v1":
        raise SystemExit(f"{path}: unsupported captured tile registry schema_version={payload.get('schema_version')!r}")
    by_key: dict[int, list[dict[str, Any]]] = {}
    for tile in list(payload.get("tiles") or []):
        tile_row = dict(tile)
        for raw_key in list(tile_row.get("affected_voxel_ids") or []):
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                continue
            by_key.setdefault(key, []).append(tile_row)
    rows = []
    for key, tiles in sorted(by_key.items()):
        rows.append(
            {
                "capture_key": key,
                "captured_tile_ids": _joined(tile.get("tile_id") for tile in tiles),
                "captured_tile_types": _joined(tile.get("tile_type") for tile in tiles),
                "captured_graph_tile_hashes": _joined(tile.get("captured_graph_tile_hash") for tile in tiles),
                "captured_tile_topology_regions": _joined(tile.get("topology_region") for tile in tiles),
                "captured_tile_count": len({str(tile.get("tile_id")) for tile in tiles}),
                "topology_delta_hashes": _joined(tile.get("topology_delta") for tile in tiles),
                "basin_delta_hashes": _joined(tile.get("basin_delta") for tile in tiles),
                "restricted_operator_hashes": _joined(tile.get("restricted_operator_hash") for tile in tiles),
                "c6_operator_hashes": _joined(tile.get("c6_operator_hash") for tile in tiles),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "capture_key": pl.Int64,
            "captured_tile_ids": pl.Utf8,
            "captured_tile_types": pl.Utf8,
            "captured_graph_tile_hashes": pl.Utf8,
            "captured_tile_topology_regions": pl.Utf8,
            "captured_tile_count": pl.Int64,
            "topology_delta_hashes": pl.Utf8,
            "basin_delta_hashes": pl.Utf8,
            "restricted_operator_hashes": pl.Utf8,
            "c6_operator_hashes": pl.Utf8,
        },
        orient="row",
    )


def load_spectral_fields(metrics_path: Path | None, manifest_path: Path | None) -> dict[str, object]:
    fields: dict[str, object] = {}
    if metrics_path is not None:
        require_existing_file(metrics_path, "--spectral-metrics")
        schema = pl.scan_parquet(str(metrics_path)).collect_schema()
        exprs: list[pl.Expr] = []
        for source, dest in [
            ("captured_tile_replay_count", "spectral_captured_replay_count"),
            ("gpu_solve_count", "spectral_gpu_solve_count"),
            ("cpu_solve_count", "spectral_cpu_solve_count"),
            ("reward_cache_hit_rate", "spectral_reward_cache_hit_rate"),
        ]:
            if source in schema:
                exprs.append(pl.col(source).max().alias(dest))
        if exprs:
            fields.update(collect_streaming(pl.scan_parquet(str(metrics_path)).select(exprs)).row(0, named=True))
        if "reward_event_source" in schema:
            sources = pl.read_parquet(metrics_path, columns=["reward_event_source"]).get_column("reward_event_source")
            fields["spectral_reward_event_source"] = _joined(sources.drop_nulls().unique().to_list()) or None
    if manifest_path is not None:
        require_existing_file(manifest_path, "--subtb-run-manifest")
        manifest = json.loads(manifest_path.read_text())
        fields["dstw_spectral_status"] = manifest.get("status")
        reward_manager = manifest.get("spectral_reward_manager")
        if isinstance(reward_manager, dict):
            fields.setdefault("spectral_reward_event_source", reward_manager.get("event_trigger_source"))
            fields.setdefault("spectral_reward_cache_hit_rate", reward_manager.get("reward_cache_hit_rate"))
            fields.setdefault("spectral_gpu_solve_count", reward_manager.get("gpu_solve_count"))
            fields.setdefault("spectral_cpu_solve_count", reward_manager.get("cpu_solve_count"))
        fields.setdefault("spectral_captured_replay_count", manifest.get("captured_graph_replay_count"))
    return fields


def load_integration_context(args: argparse.Namespace) -> IntegrationContext:
    evidence = [
        str(path)
        for path in [
            args.topology_region_registry,
            args.captured_tile_registry,
            args.spectral_metrics,
            args.subtb_run_manifest,
        ]
        if path is not None
    ]
    return IntegrationContext(
        topology_regions=load_topology_region_registry(args.topology_region_registry),
        captured_tiles=load_captured_tile_registry(args.captured_tile_registry),
        spectral_fields=load_spectral_fields(args.spectral_metrics, args.subtb_run_manifest),
        evidence_paths=tuple(evidence),
    )


def aggregate_condition(event_parquet: Path, condition_id: str) -> pl.DataFrame:
    event_count = pl.len().cast(pl.UInt64).alias("_event_count")
    lazy_frame = (
        base_event_scan(event_parquet)
        .filter(pl.col("condition_id") == condition_id)
        .filter(pl.col("water_density").is_not_null() & (pl.col("water_density") != 0.0))
        .group_by(["condition_id", "primary_residue_idx", "voxel_idx"])
        .agg(
            [
                pl.col("water_density").std(ddof=0).fill_null(0.0).alias("sigma_hyd"),
                (pl.col("wd_change").abs() * pl.col("intensity")).mean().alias("solvent_wire_importance"),
                (pl.col("water_density").max() - pl.col("water_density").min()).alias("occlusion_risk"),
                event_count,
            ]
        )
        .rename({"primary_residue_idx": "residue_id"})
        .with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("condition_id"),
                        pl.lit("_res"),
                        pl.col("residue_id").cast(pl.Utf8),
                        pl.lit("_vox"),
                        pl.col("voxel_idx").cast(pl.Utf8),
                    ]
                ).alias("hydration_tunnel_id"),
                pl.col("sigma_hyd").cast(pl.Float64),
                pl.col("solvent_wire_importance").cast(pl.Float64),
                pl.col("occlusion_risk").cast(pl.Float64),
            ]
        )
        .select([*OUTPUT_COLUMNS, "_event_count"])
    )
    return collect_streaming(lazy_frame)


def add_null_column(frame: pl.DataFrame, name: str, dtype: pl.DataType) -> pl.DataFrame:
    return frame.with_columns(pl.lit(None, dtype=dtype).alias(name))


def add_missing_integration_columns(frame: pl.DataFrame) -> pl.DataFrame:
    output = frame
    for name in INTEGRATION_COLUMNS:
        if name in output.columns:
            continue
        if name in {
            "captured_tile_count",
            "spectral_captured_replay_count",
            "spectral_gpu_solve_count",
            "spectral_cpu_solve_count",
        }:
            output = output.with_columns(pl.lit(0, dtype=pl.Int64).alias(name))
        elif name == "spectral_reward_cache_hit_rate":
            output = add_null_column(output, name, pl.Float64)
        else:
            output = add_null_column(output, name, pl.Utf8)
    return output


def with_suffixed_capture_columns(captured_tiles: pl.DataFrame, suffix: str) -> pl.DataFrame:
    return captured_tiles.rename(
        {
            "capture_key": f"capture_key_{suffix}",
            **{name: f"{name}_{suffix}" for name in CAPTURE_FIELDS},
        }
    )


def enrich_condition_chunk(chunk: pl.DataFrame, context: IntegrationContext) -> pl.DataFrame:
    if not context.enabled:
        return chunk

    output = chunk
    if context.topology_regions is not None and context.topology_regions.height > 0:
        output = output.join(context.topology_regions, on="residue_id", how="left")
    else:
        output = add_null_column(output, "topology_region", pl.Utf8)
        output = add_null_column(output, "topology_region_source", pl.Utf8)

    if context.captured_tiles is not None and context.captured_tiles.height > 0:
        voxel_tiles = with_suffixed_capture_columns(context.captured_tiles, "voxel")
        residue_tiles = with_suffixed_capture_columns(context.captured_tiles, "residue")
        output = output.join(voxel_tiles, left_on="voxel_idx", right_on="capture_key_voxel", how="left")
        output = output.join(residue_tiles, left_on="residue_id", right_on="capture_key_residue", how="left")
        output = output.with_columns(
            [
                *[
                    pl.coalesce([pl.col(f"{name}_voxel"), pl.col(f"{name}_residue")]).alias(name)
                    for name in CAPTURE_FIELDS
                    if name != "captured_tile_count"
                ],
                pl.coalesce(
                    [pl.col("captured_tile_count_voxel"), pl.col("captured_tile_count_residue"), pl.lit(0)]
                )
                .cast(pl.Int64)
                .alias("captured_tile_count"),
                pl.when(pl.col("captured_tile_count_voxel").is_not_null())
                .then(pl.lit("voxel_idx"))
                .when(pl.col("captured_tile_count_residue").is_not_null())
                .then(pl.lit("residue_id"))
                .otherwise(None)
                .alias("captured_tile_match_basis"),
            ]
        ).drop(
            [
                name
                for name in output.columns
                if name.endswith("_voxel") or name.endswith("_residue") or name.startswith("capture_key_")
            ]
        )
    else:
        output = output.with_columns([pl.lit(0, dtype=pl.Int64).alias("captured_tile_count")])

    spectral_fields = context.spectral_fields or {}
    spectral_exprs: list[pl.Expr] = []
    for name, dtype in SPECTRAL_FIELDS.items():
        value = spectral_fields.get(name)
        spectral_exprs.append(pl.lit(value, dtype=dtype).alias(name))
    spectral_exprs.append(pl.lit(json.dumps(list(context.evidence_paths)), dtype=pl.Utf8).alias("dstw_context_evidence_paths"))
    output = output.with_columns(spectral_exprs)
    output = add_missing_integration_columns(output)
    output = output.with_columns(
        pl.when((pl.col("captured_tile_count") > 0) & pl.col("spectral_reward_event_source").is_not_null())
        .then(pl.lit("DSTW_CAPTURED_GRAPH_SPECTRAL_LINKED"))
        .when(pl.col("captured_tile_count") > 0)
        .then(pl.lit("DSTW_CAPTURED_GRAPH_LINKED"))
        .when(pl.col("topology_region").is_not_null())
        .then(pl.lit("DSTW_TOPOLOGY_REGION_LINKED"))
        .otherwise(pl.lit("DSTW_UNMAPPED"))
        .alias("dstw_integration_status")
    )
    return output


def write_hydration_statistics(
    event_parquet: Path,
    out_dir: Path,
    conditions: Iterable[str],
    context: IntegrationContext,
) -> tuple[Path, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "hydration_statistics.parquet"
    tmp_path = out_dir / f".hydration_statistics.parquet.tmp-{os.getpid()}"
    tmp_path.unlink(missing_ok=True)

    total_groups = 0
    total_events = 0
    parquet_schema = output_arrow_schema(context.enabled)
    parquet_columns = output_columns(context.enabled)

    try:
        with pq.ParquetWriter(
            tmp_path,
            schema=parquet_schema,
            compression="zstd",
            write_statistics=True,
        ) as writer:
            for index, condition_id in enumerate(conditions, start=1):
                started = time.monotonic()
                emit_status("condition_chunk_start", condition_index=index, condition_id=condition_id)
                chunk = aggregate_condition(event_parquet, condition_id)
                event_rows = int(chunk.get_column("_event_count").sum()) if chunk.height else 0
                output_chunk = enrich_condition_chunk(chunk.drop("_event_count"), context).select(parquet_columns)
                group_rows = output_chunk.height
                if group_rows:
                    table = output_chunk.to_arrow().cast(parquet_schema)
                    writer.write_table(table, row_group_size=100_000)
                total_groups += group_rows
                total_events += event_rows
                emit_status(
                    "condition_chunk_written",
                    condition_index=index,
                    condition_id=condition_id,
                    input_event_rows=event_rows,
                    output_group_rows=group_rows,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )
            if total_groups == 0:
                writer.write_table(pa.Table.from_batches([], schema=parquet_schema))
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return output_path, total_events, total_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-parquet", required=True, help="Path to spike_events_snr_masked.parquet")
    parser.add_argument("--out-dir", required=True, help="Directory to write hydration_statistics.parquet")
    parser.add_argument(
        "--topology-region-registry",
        type=Path,
        help="Optional Track B topology_region_registry.json for residue-to-region annotation.",
    )
    parser.add_argument(
        "--captured-tile-registry",
        type=Path,
        help="Optional captured_tile_registry.json for DSTW captured graph tile annotation.",
    )
    parser.add_argument(
        "--spectral-metrics",
        type=Path,
        help="Optional subtb_training_metrics.parquet for spectral DSTW telemetry annotation.",
    )
    parser.add_argument(
        "--subtb-run-manifest",
        type=Path,
        help="Optional subtb_run_manifest.json for spectral DSTW runtime status annotation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_parquet = Path(args.event_parquet)
    out_dir = Path(args.out_dir)

    started = time.monotonic()
    validate_input_schema(event_parquet)
    context = load_integration_context(args)
    emit_status("hydration_extractor_start", event_parquet=str(event_parquet), out_dir=str(out_dir))
    emit_status(
        "dstw_context_loaded",
        enabled=context.enabled,
        topology_region_rows=0 if context.topology_regions is None else context.topology_regions.height,
        captured_tile_keys=0 if context.captured_tiles is None else context.captured_tiles.height,
        spectral_fields=sorted((context.spectral_fields or {}).keys()),
        evidence_paths=list(context.evidence_paths),
    )
    conditions = discover_condition_ids(event_parquet)
    emit_status("conditions_discovered", condition_count=len(conditions), condition_ids=conditions)
    output_path, total_events, total_groups = write_hydration_statistics(event_parquet, out_dir, conditions, context)
    emit_status(
        "hydration_extractor_complete",
        output_path=str(output_path),
        condition_count=len(conditions),
        input_event_rows=total_events,
        output_group_rows=total_groups,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
