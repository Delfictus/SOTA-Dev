#!/usr/bin/env python3
"""Shared helpers for GLP-1R n80 raw extraction scripts."""

from __future__ import annotations

import hashlib
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TypeAlias

import polars as pl

from prism_dstw.io import write_provenance_parquet
from prism_dstw.ontology import StreamId
from prism_dstw.propagation_ledger import append_ledger_entry


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

CAMPAIGN_ID = "glp1r_aleniglipron"
DEFAULT_RAW_ROOT = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/"
    "20260518T031002Z/05_RESULTS/glp1r_aleniglipron_risk_map"
)
DEFAULT_OUTPUT_DIR = Path("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale")

STREAM_PATTERN = re.compile(
    r"^(?P<condition>.+)_stream(?P<stream>\d{1,2})_(?P<suffix>.+)$"
)
PRISM_V2_PATTERN = re.compile(r"^prism_v2_(?P<run>\d+)_(?P<stream>\d+)\.bin$")


@dataclass(frozen=True)
class StreamFile:
    path: Path
    condition_id: str
    replica_id: int
    stream_id: StreamId


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def raw_uri(path: Path) -> str:
    return "raw://" + path.resolve().as_posix().lstrip("/")


def parse_stream_file(path: Path, suffix: str) -> StreamFile | None:
    parent = path.parent.name
    if not parent.startswith("replica_"):
        return None
    try:
        replica_id = int(parent.split("_", 1)[1])
    except ValueError:
        return None
    condition_id = path.parent.parent.name
    if suffix == "prism_v2.bin":
        match = PRISM_V2_PATTERN.match(path.name)
        if match is None:
            return None
        return StreamFile(
            path=path,
            condition_id=condition_id,
            replica_id=replica_id,
            stream_id=StreamId(int(match.group("stream"))),
        )
    match = STREAM_PATTERN.match(path.name)
    if match is None:
        return None
    if match.group("suffix") != suffix:
        return None
    return StreamFile(
        path=path,
        condition_id=condition_id,
        replica_id=replica_id,
        stream_id=StreamId(int(match.group("stream"))),
    )


def discover_stream_files(raw_root: Path, suffix: str) -> list[StreamFile]:
    files: list[StreamFile] = []
    for path in sorted(raw_root.glob(f"*/replica_*/*{suffix}")):
        parsed = parse_stream_file(path, suffix)
        if parsed is not None:
            files.append(parsed)
    return files


def parse_stream_selector(value: str | None) -> set[StreamId] | None:
    if value is None or value.strip() == "":
        return None
    return {StreamId(int(part.strip())) for part in value.split(",") if part.strip()}


def filter_streams(
    files: Sequence[StreamFile],
    *,
    condition_id: str | None,
    replica_id: int | None,
    stream_ids: set[StreamId] | None,
    max_streams: int | None,
) -> list[StreamFile]:
    out = [
        item
        for item in files
        if (condition_id is None or item.condition_id == condition_id)
        and (replica_id is None or item.replica_id == replica_id)
        and (stream_ids is None or item.stream_id in stream_ids)
    ]
    return out[:max_streams] if max_streams is not None else out


def replica_json_for(stream_file: StreamFile, name: str) -> Path:
    return stream_file.path.parent / name


def json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported JSON value type: {type(value).__name__}")


def read_json_object(path: Path) -> JsonObject:
    import json

    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return {str(key): json_value(item) for key, item in loaded.items()}


def json_list(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def json_object(value: JsonValue) -> JsonObject:
    return value if isinstance(value, dict) else {}


def json_int(value: JsonValue, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return default


def json_float(value: JsonValue, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return default


def json_str(value: JsonValue, default: str = "unknown") -> str:
    return value if isinstance(value, str) else default


def raw_checksum_map(paths: Iterable[Path]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted({p.resolve() for p in paths}):
        checksums[raw_uri(path)] = sha256_file(path)
    return checksums


def append_raw_input_ledger(
    *,
    ledger_path: Path,
    module: str,
    raw_inputs: Sequence[Path],
    parameters: Mapping[str, JsonValue],
    output_path: Path,
    row_count: int | None,
    repo_root: Path,
) -> None:
    checksums = raw_checksum_map(raw_inputs)
    inputs: dict[str, JsonValue] = {
        f"raw_{idx}": raw_uri(path)
        for idx, path in enumerate(sorted({p.resolve() for p in raw_inputs}))
    }
    checksum_values: dict[str, JsonValue] = {
        key: value
        for key, value in checksums.items()
    }
    entry: JsonObject = {
        "entry_id": str(uuid.uuid4()),
        "module": module,
        "operation": "raw_input_checksum_capture",
        "inputs": inputs,
        "input_checksums": checksum_values,
        "parameters": dict(parameters),
        "output_value": {"output_path": output_path.as_posix(), "row_count": row_count},
        "output_uncertainty": None,
        "timestamp": datetime.now(UTC).isoformat(),
        "gate_status": {
            "raw_sha256": True,
            "external_raw_uri": True,
            "write_provenance_parquet_used": True,
        },
        "supersedes": None,
    }
    append_ledger_entry(ledger_path, entry, repo_root=repo_root)


def write_n80_parquet(
    frame: pl.DataFrame | pl.LazyFrame,
    output_path: Path,
    *,
    producer_script: Path,
    pipeline_stage: str,
    schema_version: str,
    partition_keys: Sequence[str],
    raw_inputs: Sequence[Path],
    source_parquets: Sequence[Path] = (),
    ledger_parameters: Mapping[str, JsonValue],
    row_count: int | None,
) -> Path:
    repo_root = Path.cwd().resolve()
    written = write_provenance_parquet(
        frame,
        output_path,
        producer_script=producer_script,
        source_parquets=source_parquets,
        schema_version=schema_version,
        pipeline_stage=pipeline_stage,
        partition_keys=partition_keys,
        extra_metadata={
            "campaign_id": CAMPAIGN_ID,
            "raw_input_count": len(raw_inputs),
            "source_parquet_count": len(source_parquets),
        },
        ledger_parameters=dict(ledger_parameters),
        ledger_output_value={"output_path": output_path, "row_count": row_count},
        repo_root=repo_root,
    )
    append_raw_input_ledger(
        ledger_path=written.with_suffix(".propagation.jsonl"),
        module=pipeline_stage,
        raw_inputs=raw_inputs,
        parameters=ledger_parameters,
        output_path=written,
        row_count=row_count,
        repo_root=repo_root,
    )
    return written


def schema_text(frame: pl.DataFrame | pl.LazyFrame) -> str:
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    return "\n".join(f"{name}: {dtype}" for name, dtype in schema.items())
