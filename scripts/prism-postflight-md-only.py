#!/usr/bin/env python3
"""Validate md-only PRISM output directories using md_evidence_manifest.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def validate_md_manifest(path: Path) -> tuple[int, list[str], list[str], dict[str, object]]:
    fails: list[str] = []
    warns: list[str] = []
    details: dict[str, object] = {"path": str(path)}
    if not path.exists():
        fails.append(f"Missing md_evidence_manifest.json: {path.name}")
        return 1, fails, warns, details

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fails.append(f"Invalid JSON in {path.name}: {exc}")
        return 1, fails, warns, details

    details.update(
        {
            "schema_kind": data.get("schema_kind"),
            "target": data.get("target"),
            "run_id": data.get("run_id"),
            "stream_count": data.get("stream_count"),
            "streams_serialized": data.get("streams_serialized"),
            "total_spikes_md": data.get("total_spikes_md"),
            "required_artifacts_complete": data.get("required_artifacts_complete"),
            "validation_status": data.get("validation_status"),
        }
    )

    if data.get("schema_kind") != "md_evidence_manifest":
        fails.append(f"schema_kind={data.get('schema_kind')!r} (expected 'md_evidence_manifest')")
    if data.get("required_artifacts_complete") is not True:
        fails.append("required_artifacts_complete is not true")

    stream_count = int(data.get("stream_count", 0) or 0)
    streams_serialized = int(data.get("streams_serialized", 0) or 0)
    total_spikes_md = int(data.get("total_spikes_md", 0) or 0)
    if stream_count <= 0:
        fails.append(f"stream_count={stream_count} (must be > 0)")
    if streams_serialized != stream_count:
        fails.append(f"streams_serialized={streams_serialized} != stream_count={stream_count}")
    if total_spikes_md <= 0:
        fails.append(f"total_spikes_md={total_spikes_md} (must be > 0)")

    if data.get("validation_status") not in {None, "PASS"}:
        warns.append(f"validation_status={data.get('validation_status')!r}")

    return (0 if not fails else 1), fails, warns, details


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: prism-postflight-md-only.py <output_dir> <prefix>")
        return 1

    output_dir = Path(sys.argv[1])
    prefix = sys.argv[2]
    manifest = output_dir / "md_evidence_manifest.json"
    rc, fails, warns, details = validate_md_manifest(manifest)

    print(f"Postflight MD-only: {prefix}")
    print(f"  md_evidence_manifest: {manifest.name}")
    for key in [
        "schema_kind",
        "target",
        "run_id",
        "stream_count",
        "streams_serialized",
        "total_spikes_md",
        "required_artifacts_complete",
        "validation_status",
    ]:
        if key in details:
            print(f"  {key}: {details[key]}")
    for warn in warns:
        print(f"  WARN: {warn}")
    if fails:
        for fail in fails:
            print(f"  FAIL: {fail}")
        print("POSTFLIGHT_MD_ONLY: FAIL")
        return 1
    print("POSTFLIGHT_MD_ONLY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
