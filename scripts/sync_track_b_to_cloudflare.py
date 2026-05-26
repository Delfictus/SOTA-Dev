#!/usr/bin/env python3
"""Dry-run or execute copy-only Track B Cloudflare sync."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from prism_dstw.calibration.track_b_artifacts import read_json, sha256_file


def _rclone_target(target: str) -> str:
    if not target.startswith("r2://"):
        raise SystemExit(f"unsupported cloud target: {target}")
    return "r2:" + target[len("r2://") :]


def _verify_remote_size(local: Path, remote: str) -> None:
    subprocess.run(
        ["rclone", "check", str(local), remote, "--one-way", "--checksum"],
        check=True,
        text=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        raise SystemExit("choose exactly one of --dry-run or --execute")
    manifest = read_json(args.manifest)
    if args.execute:
        token_present = bool(os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CF_R2_CATALOG_TOKEN"))
        if not token_present:
            raise SystemExit("BLOCKED_WITH_HARD_EVIDENCE: Cloudflare credentials not loaded")
    checked = 0
    for item in manifest.get("artifacts", []):
        path = Path(str(item["path"]))
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")
        if sha256_file(path) != item["sha256"]:
            raise SystemExit(f"hash mismatch: {path}")
        if args.execute:
            remote = _rclone_target(str(item["target"]))
            subprocess.run(["rclone", "copyto", str(path), remote], check=True)
            _verify_remote_size(path, remote)
        checked += 1
    print(f"track_b_cloudflare_sync mode={'execute' if args.execute else 'dry-run'} artifacts={checked} credential_values_printed=false")


if __name__ == "__main__":
    main()
