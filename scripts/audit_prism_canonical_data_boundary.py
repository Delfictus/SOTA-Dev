#!/usr/bin/env python3
"""Audit which critical PRISM data assets are inside the canonical seal."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CRITICAL_RELATIVE_PATHS = [
    "campaigns/glp1r_aleniglipron/track_a_generative",
    "campaigns/glp1r_aleniglipron/topologies",
    "campaigns/glp1r_aleniglipron/topology",
    "campaigns/glp1r_aleniglipron/dstw_phase_b",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_phase_manifold_smoke_fixture",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/hydration_v2_full_20260529T090620Z",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale",
]

SCAN_EXTENSIONS = {
    ".parquet",
    ".sdf",
    ".smi",
    ".h5",
    ".hdf5",
    ".duckdb",
    ".sqlite",
    ".sqlite3",
    ".db",
}


@dataclass
class PathAudit:
    relative_path: str
    source_exists: bool
    source_size_bytes: int | None
    source_symlink_target_count: int
    source_symlink_target_bytes: int
    sealed_exists: bool
    sealed_size_bytes: int | None
    sealed_symlink_target_count: int
    sealed_symlink_target_bytes: int
    status: str
    note: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def tree_stats(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.is_symlink():
            return 0, 1, path.resolve().stat().st_size
        return path.stat().st_size, 0, 0
    total = 0
    symlink_targets = 0
    symlink_target_bytes = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                if file_path.is_symlink():
                    symlink_targets += 1
                    symlink_target_bytes += file_path.resolve().stat().st_size
                else:
                    total += file_path.stat().st_size
            except FileNotFoundError:
                continue
    return total, symlink_targets, symlink_target_bytes


def summarize_path(repo_root: Path, seal_root: Path, rel_path: str) -> PathAudit:
    source = repo_root / rel_path
    sealed = seal_root / "build_context" / "rootfs" / "home" / "diddy" / "Desktop" / "Prism4D-bio" / rel_path
    source_exists = source.exists()
    sealed_exists = sealed.exists()
    source_size, source_symlink_count, source_symlink_bytes = tree_stats(source) if source_exists else (None, 0, 0)
    sealed_size, sealed_symlink_count, sealed_symlink_bytes = tree_stats(sealed) if sealed_exists else (None, 0, 0)

    if source_exists and sealed_exists:
        if source_symlink_count:
            status = "SOURCE_HAS_EXTERNAL_SYMLINK_DEPENDENCIES"
            note = "source path includes symlinked targets outside the materialized directory footprint"
        elif source_size == sealed_size:
            status = "SEALED_MATCH"
            note = "source and sealed sizes match"
        else:
            status = "SEALED_DIFFERENT"
            note = "source and sealed sizes differ"
    elif source_exists and not sealed_exists:
        status = "MISSING_FROM_SEAL"
        note = "present in source tree but absent from canonical seal"
    elif not source_exists and sealed_exists:
        status = "SEALED_ONLY"
        note = "present in seal but not source tree"
    else:
        status = "MISSING_BOTH"
        note = "not found in source tree or canonical seal"

    return PathAudit(
        relative_path=rel_path,
        source_exists=source_exists,
        source_size_bytes=source_size,
        source_symlink_target_count=source_symlink_count,
        source_symlink_target_bytes=source_symlink_bytes,
        sealed_exists=sealed_exists,
        sealed_size_bytes=sealed_size,
        sealed_symlink_target_count=sealed_symlink_count,
        sealed_symlink_target_bytes=sealed_symlink_bytes,
        status=status,
        note=note,
    )


def iter_large_external_candidates(
    scan_roots: Iterable[Path],
    repo_root: Path,
    seal_root: Path,
    limit: int,
) -> list[dict]:
    rows: list[tuple[int, str]] = []
    excluded_roots = {
        repo_root.resolve(),
        seal_root.resolve(),
    }
    excluded_prefixes = (
        "/mnt/storage/prism_canonical_seals",
        "/mnt/storage/prism_release",
        "/media/diddy/PRISM-LBS/PRISM-ISOLATED-20260528/frozen_partial_snapshot",
    )
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for root, dirs, files in os.walk(scan_root):
            root_path = Path(root)
            try:
                resolved_root = root_path.resolve()
            except FileNotFoundError:
                continue
            if any(str(resolved_root).startswith(str(ex)) for ex in excluded_roots):
                dirs[:] = []
                continue
            if any(str(resolved_root).startswith(prefix) for prefix in excluded_prefixes):
                dirs[:] = []
                continue
            for name in files:
                path = root_path / name
                if path.suffix.lower() not in SCAN_EXTENSIONS:
                    continue
                try:
                    rows.append((path.stat().st_size, str(path)))
                except FileNotFoundError:
                    continue
    rows.sort(reverse=True)
    out = []
    for size, path in rows[:limit]:
        out.append({"path": path, "size_bytes": size})
    return out


def render_markdown(report: dict) -> str:
    lines = [
        "# PRISM Canonical Data Boundary Audit",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Seal root: `{report['seal_root']}`",
        f"- Repo root: `{report['repo_root']}`",
        "",
        "## Critical Paths",
        "",
        "| Path | Status | Source Bytes | Sealed Bytes | Note |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in report["critical_paths"]:
        lines.append(
            f"| `{row['relative_path']}` | `{row['status']}` | "
            f"{row['source_size_bytes'] or 0} | {row['sealed_size_bytes'] or 0} | {row['note']} |"
        )
    lines.extend(["", "## Large External Candidates", ""])
    if report["large_external_candidates"]:
        for row in report["large_external_candidates"]:
            lines.append(f"- `{row['path']}` ({row['size_bytes']} bytes)")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/home/diddy/Desktop/Prism4D-bio")
    parser.add_argument("--seal-root", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    parser.add_argument(
        "--scan-root",
        action="append",
        default=["/mnt/storage", "/media/diddy/PRISM-LBS"],
        help="Additional roots to scan for large external data candidates.",
    )
    parser.add_argument("--external-limit", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    seal_root = Path(args.seal_root).resolve()
    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    scan_roots = [Path(p).resolve() for p in args.scan_root]

    critical = [summarize_path(repo_root, seal_root, rel_path) for rel_path in CRITICAL_RELATIVE_PATHS]
    large_external = iter_large_external_candidates(scan_roots, repo_root, seal_root, args.external_limit)

    report = {
        "schema_version": "prism.canonical_data_boundary_audit.v1",
        "generated_at_utc": now_utc(),
        "repo_root": str(repo_root),
        "seal_root": str(seal_root),
        "critical_paths": [asdict(row) for row in critical],
        "large_external_candidates": large_external,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_out), "md": str(md_out), "critical_count": len(critical)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
