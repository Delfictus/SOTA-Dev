#!/usr/bin/env python3
"""Materialize observed GLP1R variant signal grids embedded in the N80 grid parquet."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN_DIR / "track_a_generative"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
DEFAULT_INPUT = N80_DIR / "signal_grid_variance_channel.parquet"
DEFAULT_REPORT = TRACK_A / "variant_grid_materialization_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=N80_DIR)
    parser.add_argument("--variants", type=str, default="A316T,T149M")
    parser.add_argument("--condition-prefix", type=str, default="glp1r_6XOX")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    outputs: list[dict[str, Any]] = []
    schema = pl.scan_parquet(source).collect_schema()
    if "condition_id" not in schema.names():
        raise RuntimeError(f"{source} has no condition_id column; cannot materialize observed variants")
    available = (
        pl.scan_parquet(source)
        .select(pl.col("condition_id").unique())
        .collect()
        .get_column("condition_id")
        .to_list()
    )
    for variant in variants:
        condition_id = f"{args.condition_prefix}_{variant}"
        if condition_id not in available:
            raise FileNotFoundError(f"embedded condition_id={condition_id} not present in {source}")
        frame = pl.scan_parquet(source).filter(pl.col("condition_id") == condition_id).collect()
        if frame.height == 0:
            raise RuntimeError(f"condition_id={condition_id} produced zero rows")
        output = output_dir / f"signal_grid_variance_channel_{variant}.parquet"
        tmp_output = output.with_suffix(output.suffix + ".tmp")
        frame.write_parquet(tmp_output)
        tmp_output.replace(output)
        nonvoid = frame.filter(pl.col("variance_class") != "void").height if "variance_class" in frame.columns else 0
        outputs.append(
            {
                "variant": variant,
                "condition_id": condition_id,
                "output": output.as_posix(),
                "rows": frame.height,
                "nonvoid": nonvoid,
                "source": source.as_posix(),
                "provenance": "OBSERVED_EMBEDDED_N80_SIGNAL_GRID",
            }
        )
        print(
            "variant_grid_materialized "
            f"variant={variant} condition_id={condition_id} rows={frame.height} nonvoid={nonvoid} output={output}"
        )
    report = {
        "schema_version": "PRISM.variant_grid_materialization.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": source.as_posix(),
        "outputs": outputs,
        "available_conditions": [str(value) for value in available],
    }
    atomic_write_json(Path(args.report), report)
    return 0


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
