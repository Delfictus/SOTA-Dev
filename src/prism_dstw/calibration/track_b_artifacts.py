"""Shared Track B artifact helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import polars as pl

TRACK_B_ROOT = Path("campaigns/glp1r_aleniglipron/track_b_chronological")
N80_ROOT = Path("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale")
TRACK_A_ROOT = Path("campaigns/glp1r_aleniglipron/track_a_generative")
CAMPAIGN_ROOT = Path("campaigns/glp1r_aleniglipron")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, Any], payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_schema_and_count(path: Path) -> tuple[dict[str, str], int]:
    frame = pl.scan_parquet(str(path))
    schema = {name: str(dtype) for name, dtype in frame.collect_schema().items()}
    row_count = int(frame.select(pl.len()).collect().item())
    return schema, row_count


def csv_schema_and_count(path: Path) -> tuple[dict[str, str], int]:
    frame = pl.scan_csv(str(path))
    schema = {name: str(dtype) for name, dtype in frame.collect_schema().items()}
    row_count = int(frame.select(pl.len()).collect().item())
    return schema, row_count


def artifact_metadata(path: Path) -> tuple[dict[str, str], int | None, str | None, int | None]:
    if not path.exists():
        return {}, None, None, None
    if path.suffix == ".parquet":
        schema, row_count = parquet_schema_and_count(path)
    elif path.suffix == ".csv":
        schema, row_count = csv_schema_and_count(path)
    elif path.suffix == ".json":
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            schema = {key: type(value).__name__ for key, value in parsed.items()}
        else:
            schema = {"root": type(parsed).__name__}
        row_count = None
    else:
        schema = {}
        row_count = None
    return schema, row_count, sha256_file(path), path.stat().st_size


def find_first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def provenance_for_path(path: Path, derived: bool = False) -> str:
    if not path.exists():
        return "L0_MISSING"
    if derived:
        return "L3_DERIVED"
    if path.suffix == ".parquet":
        return "L4_RUNTIME_TELEMETRY"
    return "L3_DERIVED"
