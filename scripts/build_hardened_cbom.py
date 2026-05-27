#!/usr/bin/env python3
"""Build per-subsystem Merkle CBOM and release checksum manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "release_artifacts/v0.25.0"

SUBSYSTEMS: dict[str, list[str]] = {
    "PRISM_FORGE": ["crates/prism-forge/"],
    "PRISM_DSTW": ["src/prism_dstw/"],
    "PRISM_NHS": ["crates/prism-nhs/"],
    "PRISM_ARCHIVE_SUPPORT": [".archive/crates/"],
    "PRISM_MAT": ["src/prism_dstw/adapters/materials/"],
    "CAMPAIGN_BIO": ["campaigns/glp1r_aleniglipron/"],
    "CAMPAIGN_MAT": ["campaigns/mat_battery_interphase_additive_v1/"],
    "MOTIF_ENGINE": ["src/prism_dstw/motif/"],
    "SUBTB_SPECTRAL": ["campaigns/glp1r_aleniglipron/track_b_chronological/subtb_spectral/"],
    "RELEASE_CONTROL": [
        "VERIFICATION.sh",
        "RELEASE_NOTES_v0.25.0.md",
        "AUDIT_REPORT_v0.25.0.md",
        "release_artifacts/v0.25.0/subagents/",
    ],
    "SCRIPTS": ["scripts/"],
    "TEMPLATES": ["00_registry/"],
    "TESTS": ["tests/"],
    "DEPENDENCIES": [
        "requirements.txt",
        "runpod_training/requirements.txt",
        "pyproject.toml",
        "Cargo.lock",
        "Cargo.toml",
    ],
}

EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "target",
    "node_modules",
    ".venv",
}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _merkle_root(hashes: list[str]) -> str:
    if not hashes:
        return hashlib.sha256(b"EMPTY").hexdigest()
    level = hashes[:]
    while len(level) > 1:
        next_level: list[str] = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else left
            next_level.append(hashlib.sha256((left + right).encode()).hexdigest())
        level = next_level
    return level[0]


def _iter_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = REPO_ROOT / item
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and not any(part in EXCLUDED_PARTS for part in child.parts):
                    files.append(child)
    return sorted(set(files))


def build_merkle_tree(paths: list[Path]) -> dict[str, Any]:
    leaves: list[dict[str, str | int]] = []
    for path in paths:
        rel = path.relative_to(REPO_ROOT).as_posix()
        leaves.append({"path": rel, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    root = _merkle_root([str(leaf["sha256"]) for leaf in leaves])
    return {
        "root": root,
        "n_files": len(leaves),
        "total_size_bytes": sum(int(leaf["size_bytes"]) for leaf in leaves),
        "leaves": leaves,
    }


def build_cbom() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    subsystems = {}
    for name, paths in SUBSYSTEMS.items():
        subsystems[name] = build_merkle_tree(_iter_files(paths))
    root_hash = _merkle_root([subsystems[name]["root"] for name in sorted(subsystems)])
    return {
        "schema_version": "PRISM.hardened_cbom.v2",
        "version": "2.0",
        "build_timestamp_utc": datetime.now(UTC).isoformat(),
        "source_commit": head,
        "root_hash": root_hash,
        "subsystems": subsystems,
    }


def write_outputs(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cbom = build_cbom()
    cbom_path = output_dir / "CBOM_v2.0.json"
    cbom_path.write_text(json.dumps(cbom, indent=2, sort_keys=True) + "\n")
    checksum_path = output_dir / "SHA256SUMS.txt"
    lines: list[str] = []
    for subsystem_name in sorted(cbom["subsystems"]):
        for leaf in cbom["subsystems"][subsystem_name]["leaves"]:
            lines.append(f"{leaf['sha256']}  {leaf['path']}")
    checksum_path.write_text("\n".join(sorted(set(lines))) + "\n")
    manifest_path = output_dir / "hardened_release_manifest.json"

    def display_path(path: Path) -> str:
        try:
            return path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    manifest = {
        "schema_version": "PRISM.hardened_release_manifest.v1",
        "source_commit": cbom["source_commit"],
        "cbom": display_path(cbom_path),
        "sha256sums": display_path(checksum_path),
        "subsystem_count": len(cbom["subsystems"]),
        "root_hash": cbom["root_hash"],
        "total_files": sum(int(tree["n_files"]) for tree in cbom["subsystems"].values()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"cbom": cbom_path.as_posix(), "sha256sums": checksum_path.as_posix(), "manifest": manifest_path.as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = write_outputs(args.output_dir)
    print(
        "hardened_cbom_built "
        f"cbom={paths['cbom']} sha256sums={paths['sha256sums']} manifest={paths['manifest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
