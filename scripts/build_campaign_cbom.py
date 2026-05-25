#!/usr/bin/env python3
"""Build a Merkle-tree cryptographic bill of materials for the campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

DEFAULT_CAMPAIGN_DIR = Path("campaigns/glp1r_aleniglipron")
DEFAULT_OUTPUT = DEFAULT_CAMPAIGN_DIR / "PRISM_CBOM_v1.0.json"
DEFAULT_EXTRA_FILES = (Path("campaigns/amyr_calcitonin_combo/campaign_init_manifest.json"),)
INCLUDED_SUFFIXES = {
    ".parquet",
    ".json",
    ".jsonl",
    ".csv",
    ".sdf",
    ".md",
    ".yml",
    ".yaml",
    ".txt",
    ".html",
    ".js",
    ".css",
    ".wasm",
}
EXCLUDED_NAMES = {
    "PRISM_GLP1R_M2_Release_v1.0.tar.gz",
    "PRISM_GLP1R_M2_Release_v1.0.tar.gz.sha256",
    "PRISM_GLP1R_M3_FINAL_RELEASE_v2.0.tar.gz",
    "PRISM_GLP1R_M3_FINAL_RELEASE_v2.0.tar.gz.sha256",
}
EXCLUDED_PARTS = {"node_modules", "__pycache__", ".git"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--extra-file",
        action="append",
        default=None,
        type=Path,
        help="Additional file outside the campaign root to include in the Merkle tree.",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def include_file(path: Path) -> bool:
    if path.name in EXCLUDED_NAMES:
        return False
    if (
        "track_a_generative" in path.parts
        and "fullscale_shards" in path.parts
        and path.name.startswith("shard_")
        and path.suffix.lower() == ".parquet"
    ):
        return False
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    return path.suffix.lower() in INCLUDED_SUFFIXES


def iter_files(campaign_dir: Path) -> list[Path]:
    return sorted(path for path in campaign_dir.rglob("*") if path.is_file() and include_file(path))


def cbom_key(campaign_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(campaign_dir).as_posix()
    except ValueError:
        return f"external/{path.as_posix().lstrip('/')}"


def file_hashes(campaign_dir: Path, extra_files: Sequence[Path]) -> dict[str, str]:
    hashes = {cbom_key(campaign_dir, path): sha256_file(path) for path in iter_files(campaign_dir)}
    for extra_file in sorted(extra_files):
        if extra_file.exists() and extra_file.is_file() and include_file(extra_file):
            hashes[cbom_key(campaign_dir, extra_file)] = sha256_file(extra_file)
    return hashes


def directory_hashes(campaign_dir: Path, file_hashes: dict[str, str]) -> dict[str, str]:
    directories = sorted({Path(rel).parent for rel in file_hashes}, key=lambda value: len(value.parts), reverse=True)
    directory_digest: dict[str, str] = {}
    for directory in directories:
        children: list[str] = []
        for rel, digest in file_hashes.items():
            if Path(rel).parent == directory:
                children.append(f"file:{Path(rel).name}:{digest}")
        for child_dir, digest in directory_digest.items():
            child_path = Path(child_dir)
            if child_path.parent == directory:
                children.append(f"dir:{child_path.name}:{digest}")
        directory_digest[directory.as_posix()] = sha256_bytes("\n".join(sorted(children)).encode("utf-8"))
    root_children = [f"dir:{Path(path).name}:{digest}" for path, digest in directory_digest.items() if Path(path).parent == Path(".")]
    root_children.extend(f"file:{Path(rel).name}:{digest}" for rel, digest in file_hashes.items() if Path(rel).parent == Path("."))
    directory_digest["."] = sha256_bytes("\n".join(sorted(root_children)).encode("utf-8"))
    return directory_digest


def environment() -> dict[str, JsonValue]:
    return {
        "python_version": sys.version.split()[0],
        "polars_version": pl.__version__,
        "os_kernel": platform.release(),
        "platform": platform.platform(),
    }


def build_cbom(campaign_dir: Path, extra_files: Sequence[Path] = DEFAULT_EXTRA_FILES) -> dict[str, JsonValue]:
    hashes = file_hashes(campaign_dir, extra_files)
    dir_hashes = directory_hashes(campaign_dir, hashes)
    return {
        "schema_version": "PRISM_CBOM.v1.0",
        "campaign_id": "glp1r_aleniglipron",
        "campaign_merkle_root": dir_hashes["."],
        "file_count": len(hashes),
        "directory_count": len(dir_hashes),
        "environment": environment(),
        "files": [{"path": path, "sha256": digest} for path, digest in sorted(hashes.items())],
        "directories": [{"path": path, "sha256": digest} for path, digest in sorted(dir_hashes.items())],
    }


def main() -> None:
    args = parse_args()
    if not args.campaign_dir.exists():
        raise FileNotFoundError(f"campaign directory not found: {args.campaign_dir}")
    extra_files = list(DEFAULT_EXTRA_FILES)
    if args.extra_file is not None:
        extra_files.extend(args.extra_file)
    cbom = build_cbom(args.campaign_dir, extra_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} files={cbom['file_count']} merkle_root={cbom['campaign_merkle_root']}")


if __name__ == "__main__":
    main()
