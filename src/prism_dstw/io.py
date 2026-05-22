"""Arrow-native provenance writers for PRISM-DSTW analytical data."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import polars as pl
import pyarrow as pa
import pyarrow
import pyarrow.parquet as pq

from .exceptions import FatalBoundaryError
from .propagation_ledger import (
    append_propagation_entry,
    build_entry,
    repo_relative_path,
)


ROW_GROUP_SIZE = 100_000
RESERVED_METADATA_KEYS = {
    "created_by",
    "generator_script",
    "generator_hash",
    "source_parquets",
    "dependency_versions",
    "schema_version",
    "pipeline",
    "pipeline_stage",
    "partition_keys",
}

JsonObject = dict[str, Any]
LedgerValue = float | JsonObject | str | None


@dataclass(frozen=True)
class ParquetProvenance:
    """Compatibility provenance contract used by legacy hardened generators."""

    generator_script: Path
    source_parquets: Sequence[Path]
    schema_version: str
    pipeline_stage: str
    partition_keys: Sequence[str]


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _absolute_within_repo(path: Path, repo_root: Path) -> Path:
    root = repo_root.resolve()
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FatalBoundaryError(f"path is outside repository boundary: {path}") from exc
    return candidate


def sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    versions = {
        "polars": pl.__version__,
        "pyarrow": pyarrow.__version__,
        "python": sys.version.split()[0],
    }
    try:
        import numpy as np
    except ImportError:
        return versions
    versions["numpy"] = str(np.__version__)
    return versions


def _metadata_string(value: Any) -> str:
    if isinstance(value, str) and ("/home/" in value or "/Users/" in value or value.startswith("file:")):
        raise FatalBoundaryError("metadata strings must not contain workstation-local paths")
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def provenance_metadata(
    *,
    producer_script: Path,
    source_parquets: Sequence[Path],
    schema_version: str,
    pipeline_stage: str,
    partition_keys: Sequence[str],
    pipeline: str = "prism-dstw",
    extra: JsonObject | None = None,
    repo_root: Path | None = None,
) -> dict[str, str]:
    root = (repo_root or default_repo_root()).resolve()
    non_parquet_sources = [str(path) for path in source_parquets if path.suffix != ".parquet"]
    if non_parquet_sources:
        raise FatalBoundaryError(f"source_parquets must contain only .parquet files: {non_parquet_sources}")

    generator_path = _absolute_within_repo(producer_script, root)
    source_checksums = {
        repo_relative_path(path, root): sha256_path(_absolute_within_repo(path, root))
        for path in source_parquets
    }
    metadata: dict[str, str] = {
        "created_by": f"polars/{pl.__version__}",
        "generator_script": repo_relative_path(generator_path, root),
        "generator_hash": sha256_path(generator_path),
        "source_parquets": json.dumps(source_checksums, sort_keys=True),
        "dependency_versions": json.dumps(dependency_versions(), sort_keys=True),
        "schema_version": schema_version,
        "pipeline": pipeline,
        "pipeline_stage": pipeline_stage,
        "partition_keys": json.dumps(list(partition_keys)),
    }
    if extra is not None:
        reserved = sorted(RESERVED_METADATA_KEYS.intersection(extra))
        if reserved:
            raise FatalBoundaryError(f"extra_metadata cannot overwrite reserved provenance keys: {reserved}")
        for key, value in extra.items():
            metadata[key] = _metadata_string(value)
    return metadata


def _resolve_provenance(
    *,
    provenance: ParquetProvenance | None,
    producer_script: Path | None,
    source_parquets: Sequence[Path] | None,
    schema_version: str | None,
    pipeline_stage: str | None,
    partition_keys: Sequence[str] | None,
) -> ParquetProvenance:
    if provenance is not None:
        return provenance
    if (
        producer_script is None
        or source_parquets is None
        or schema_version is None
        or pipeline_stage is None
        or partition_keys is None
    ):
        raise FatalBoundaryError("incomplete parquet provenance arguments")
    return ParquetProvenance(
        generator_script=producer_script,
        source_parquets=source_parquets,
        schema_version=schema_version,
        pipeline_stage=pipeline_stage,
        partition_keys=partition_keys,
    )


def _write_frame(
    frame: pl.DataFrame | pl.LazyFrame | pa.Table,
    output_path: Path,
    *,
    metadata: dict[str, str],
    compression: str,
) -> None:
    if isinstance(frame, pl.LazyFrame):
        frame.sink_parquet(
            output_path,
            compression=compression,
            statistics=True,
            row_group_size=ROW_GROUP_SIZE,
            metadata=metadata,
        )
        return
    dataframe = pl.from_arrow(frame) if isinstance(frame, pa.Table) else frame
    dataframe.write_parquet(
        output_path,
        compression=compression,
        statistics=True,
        row_group_size=ROW_GROUP_SIZE,
        use_pyarrow=False,
        metadata=metadata,
    )


def write_provenance_parquet(
    df: pl.DataFrame | pl.LazyFrame | pa.Table,
    output_path: Path,
    *,
    producer_script: Path | None = None,
    source_parquets: Sequence[Path] | None = None,
    schema_version: str | None = None,
    pipeline_stage: str | None = None,
    partition_keys: Sequence[str] | None = None,
    extra_metadata: JsonObject | None = None,
    ledger_parameters: JsonObject | None = None,
    ledger_output_value: LedgerValue = None,
    ledger_output_uncertainty: float | None = None,
    ledger_gate_status: dict[str, bool] | None = None,
    provenance: ParquetProvenance | None = None,
    repo_root: Path | None = None,
    compression: str = "zstd",
) -> Path:
    resolved_provenance = _resolve_provenance(
        provenance=provenance,
        producer_script=producer_script,
        source_parquets=source_parquets,
        schema_version=schema_version,
        pipeline_stage=pipeline_stage,
        partition_keys=partition_keys,
    )
    root = (repo_root or default_repo_root()).resolve()
    target_path = _absolute_within_repo(output_path, root)
    metadata = provenance_metadata(
        producer_script=resolved_provenance.generator_script,
        source_parquets=resolved_provenance.source_parquets,
        schema_version=resolved_provenance.schema_version,
        pipeline_stage=resolved_provenance.pipeline_stage,
        partition_keys=resolved_provenance.partition_keys,
        extra=extra_metadata,
        repo_root=root,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _write_frame(df, target_path, metadata=metadata, compression=compression)

    parameters: JsonObject = {
        "schema_version": resolved_provenance.schema_version,
        "pipeline_stage": resolved_provenance.pipeline_stage,
        "partition_keys": list(resolved_provenance.partition_keys),
        "row_group_size": ROW_GROUP_SIZE,
        "compression": compression,
    }
    if ledger_parameters is not None:
        parameters.update(ledger_parameters)
    gate_status = {
        "provenance_metadata": True,
        "arrow_polars_writer": True,
        "append_only_ledger": True,
        "repo_relative_paths": True,
    }
    if ledger_gate_status is not None:
        gate_status.update(ledger_gate_status)
    append_propagation_entry(
        target_path.with_suffix(".propagation.jsonl"),
        build_entry(
            module=resolved_provenance.pipeline_stage,
            operation="write_provenance_parquet",
            inputs={
                f"source_{idx}": path
                for idx, path in enumerate(resolved_provenance.source_parquets)
            },
            parameters=parameters,
            output_value=ledger_output_value or {"output_path": target_path},
            output_uncertainty=ledger_output_uncertainty,
            gate_status=gate_status,
            repo_root=root,
        ),
        repo_root=root,
    )
    return target_path


def read_parquet_metadata(path: Path) -> dict[str, str]:
    """Read Arrow key-value metadata as UTF-8 strings."""

    raw_metadata = pq.read_metadata(path).metadata or {}
    return {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in raw_metadata.items()
    }


__all__ = [
    "FatalBoundaryError",
    "JsonObject",
    "LedgerValue",
    "ParquetProvenance",
    "dependency_versions",
    "provenance_metadata",
    "read_parquet_metadata",
    "repo_relative_path",
    "sha256_path",
    "write_provenance_parquet",
]
