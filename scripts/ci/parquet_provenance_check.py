#!/usr/bin/env python3
"""CI gate: fail on contaminated Parquet producer metadata in production paths."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pyarrow.parquet as pq


PRODUCTION_PATHS = [
    "campaigns",
    "00_registry",
]

ALLOWED_CREATED_BY_MARKERS = [
    "polars",
    "arrow",
    "parquet-rs",
    "parquet-cpp-arrow",
]

FORBIDDEN_CREATED_BY_MARKERS = [
    "duckdb",
    "pandas",
]

REQUIRED_METADATA_KEYS = [
    "created_by",
    "generator_script",
    "generator_hash",
    "source_parquets",
    "dependency_versions",
    "schema_version",
    "pipeline_stage",
    "partition_keys",
]


def emit_stdout(message: str) -> None:
    sys.stdout.write(message + "\n")


def emit_stderr(message: str) -> None:
    sys.stderr.write(message + "\n")


def iter_parquets(root: Path, explicit_paths: list[Path] | None = None) -> list[Path]:
    paths: list[Path] = []
    if explicit_paths:
        for base in explicit_paths:
            if base.is_file() and base.suffix == ".parquet":
                paths.append(base)
            elif base.is_dir():
                paths.extend(path for path in base.rglob("*.parquet") if path.is_file())
    for rel in PRODUCTION_PATHS:
        base = root / rel
        if not base.exists():
            continue
        if base.is_file() and base.suffix == ".parquet":
            paths.append(base)
            continue
        paths.extend(path for path in base.rglob("*.parquet") if path.is_file())
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Optional explicit files/directories to scan.")
    args = parser.parse_args()
    root = Path.cwd()
    violations: list[str] = []
    checked = 0
    for path in iter_parquets(root, args.paths):
        checked += 1
        metadata = pq.read_metadata(path)
        created_by = str(metadata.created_by or "").lower()
        if any(marker in created_by for marker in FORBIDDEN_CREATED_BY_MARKERS):
            violations.append(f"{path}: forbidden created_by={metadata.created_by!r}")
            continue
        if not any(marker in created_by for marker in ALLOWED_CREATED_BY_MARKERS):
            violations.append(f"{path}: unsupported or missing created_by={metadata.created_by!r}")
        kv_metadata = metadata.metadata or {}
        decoded_keys = {key.decode("utf-8", errors="replace") for key in kv_metadata}
        missing_keys = [key for key in REQUIRED_METADATA_KEYS if key not in decoded_keys]
        if missing_keys:
            violations.append(f"{path}: missing provenance metadata keys={missing_keys}")

    if violations:
        emit_stderr("PARQUET PROVENANCE CHECK FAILED")
        for violation in violations:
            emit_stderr(violation)
        return 1

    emit_stdout(f"PARQUET PROVENANCE CHECK PASSED ({checked} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
