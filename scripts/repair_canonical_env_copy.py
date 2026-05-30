#!/usr/bin/env python3
"""Repair a copied canonical PRISM conda env by relocating text wrappers."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_ENV_ROOT = Path("/mnt/storage/prism_env_copies/prism_dock_portable_20260529")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--source-prefix", action="append", default=None)
    return parser.parse_args()


def load_manifest(env_root: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = env_root.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing env manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"env manifest is not an object: {manifest_path}")
    return manifest_path, payload


def is_text_blob(blob: bytes) -> bool:
    if b"\x00" in blob:
        return False
    return True


def relocate_text_file(path: Path, source_prefixes: list[str], target_prefix: str) -> bool:
    raw = path.read_bytes()
    if not is_text_blob(raw):
        return False
    updated = raw
    changed = False
    for source_prefix in source_prefixes:
        source_bytes = source_prefix.encode("utf-8")
        if source_bytes not in updated:
            continue
        updated = updated.replace(source_bytes, target_prefix.encode("utf-8"))
        changed = True
    if not changed or updated == raw:
        return False
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(updated)
    tmp.chmod(path.stat().st_mode)
    tmp.replace(path)
    return True


def discover_source_prefixes(env_root: Path, manifest: dict[str, Any], explicit: list[str] | None) -> list[str]:
    prefixes: set[str] = set(explicit or [])
    manifest_source = manifest.get("source_env")
    if isinstance(manifest_source, str) and manifest_source:
        prefixes.add(manifest_source)
    pattern = re.compile(rb"(/[^\s'\":]+?)/bin/(?:python(?:[0-9.]+)?|perl|bash|sh)\b")
    for path in sorted((env_root / "bin").iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        raw = path.read_bytes()
        if not is_text_blob(raw):
            continue
        for match in pattern.findall(raw):
            prefix = match.decode("utf-8", errors="ignore")
            if prefix == str(env_root):
                continue
            if prefix.startswith("/usr") or prefix == "/bin":
                continue
            prefixes.add(prefix)
    return sorted(prefixes)


def count_remaining(env_root: Path, source_prefixes: list[str]) -> list[str]:
    matches: list[str] = []
    for path in sorted((env_root / "bin").iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        raw = path.read_bytes()
        if not is_text_blob(raw):
            continue
        for source_prefix in source_prefixes:
            if source_prefix.encode("utf-8") in raw:
                matches.append(path.name)
                break
    return matches


def main() -> int:
    args = parse_args()
    env_root = args.env_root.resolve()
    manifest_path, manifest = load_manifest(env_root)
    source_prefixes = discover_source_prefixes(env_root, manifest, args.source_prefix)
    if not source_prefixes:
        raise ValueError("no foreign source prefixes discovered in env copy")

    bin_root = env_root / "bin"
    if not bin_root.is_dir():
        raise FileNotFoundError(f"missing env bin directory: {bin_root}")

    patched: list[str] = []
    scanned = 0
    for path in sorted(bin_root.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        scanned += 1
        if relocate_text_file(path, source_prefixes, str(env_root)):
            patched.append(path.name)

    remaining = count_remaining(env_root, source_prefixes)
    report = {
        "schema_version": "prism.canonical_env_relocation.v1",
        "generated_at_utc": utc_now(),
        "env_root": str(env_root),
        "source_prefixes": source_prefixes,
        "patched_text_wrappers": len(patched),
        "patched_files": patched,
        "bin_files_scanned": scanned,
        "remaining_source_backreferences_in_bin": len(remaining),
        "remaining_files": remaining,
        "status": "PASS" if not remaining else "FAIL",
    }

    manifest["relocation_report"] = {
        "generated_at_utc": report["generated_at_utc"],
        "source_prefixes": source_prefixes,
        "patched_text_wrappers": len(patched),
        "remaining_source_backreferences_in_bin": len(remaining),
        "status": report["status"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report_path = env_root.with_suffix(".relocation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
