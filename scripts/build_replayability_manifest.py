#!/usr/bin/env python3
"""Build a zero-trust replayability manifest for the GLP-1R M2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
SCHEMA_DIR = REPO_ROOT / "00_registry/schemas"
PHYSICAL_CONSTANTS = REPO_ROOT / "00_registry/physical_constants.yml"
DEFAULT_OUTPUT = CAMPAIGN_DIR / "M2_Replayability_Manifest.json"

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN_DIR)
    parser.add_argument("--schema-dir", type=Path, default=SCHEMA_DIR)
    parser.add_argument("--physical-constants", type=Path, default=PHYSICAL_CONSTANTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"
    text = (completed.stdout or completed.stderr).strip()
    return text if text else "unavailable"


def environment_fingerprint() -> JsonObject:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "polars": pl.__version__,
        "os_kernel": platform.release(),
        "platform": platform.platform(),
        "cuda_version": run_command(["nvcc", "--version"]),
        "nvidia_driver": run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
    }


def hash_files(paths: list[Path], root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256_path(path) for path in sorted(paths)}


def merkle_root(items: dict[str, str]) -> str:
    leaves = [hashlib.sha256(f"{key}:{value}".encode("utf-8")).hexdigest() for key, value in sorted(items.items())]
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = leaves
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [
            hashlib.sha256((level[idx] + level[idx + 1]).encode("utf-8")).hexdigest()
            for idx in range(0, len(level), 2)
        ]
    return level[0]


def main() -> int:
    args = parse_args()
    campaign_dir = Path(args.campaign_dir)
    schema_dir = Path(args.schema_dir)
    constants_path = Path(args.physical_constants)
    output = Path(args.output)
    schema_hashes = hash_files(list(schema_dir.glob("*.yml")), REPO_ROOT)
    ledger_hashes = hash_files(list(campaign_dir.rglob("*.propagation.jsonl")), REPO_ROOT)
    constants_hash = sha256_path(constants_path)
    unified_inputs: dict[str, str] = {
        **{f"schema:{key}": value for key, value in schema_hashes.items()},
        **{f"ledger:{key}": value for key, value in ledger_hashes.items()},
        "parameter_freeze:00_registry/physical_constants.yml": constants_hash,
    }
    payload: JsonObject = {
        "schema_version": "m2_replayability_manifest.v1",
        "environment": environment_fingerprint(),
        "schema_hashes": cast(JsonValue, schema_hashes),
        "parameter_freeze": {
            "path": constants_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": constants_hash,
        },
        "propagation_ledger_hashes": cast(JsonValue, ledger_hashes),
        "unified_merkle_root": merkle_root(unified_inputs),
        "replayability_constraints": {
            "schema_files_hashed": len(schema_hashes),
            "propagation_ledgers_hashed": len(ledger_hashes),
            "requires_identical_physical_constants": True,
            "requires_identical_schema_registry": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
