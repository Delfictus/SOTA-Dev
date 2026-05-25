#!/usr/bin/env python3
"""Preflight R2 shard archival without modifying shard files."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARD_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/fullscale_shards"
DEFAULT_REPORT = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/shard_r2_recovery_preflight.json"

JsonObject: TypeAlias = dict[str, Any]
ShardRow: TypeAlias = dict[str, str | int | float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def env_present(name: str) -> bool:
    return bool(os.environ.get(name))


def main() -> int:
    args = parse_args()
    shard_paths = sorted(Path(args.shard_dir).glob("shard_*.parquet"))
    env_requirements = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "AWS_REGION"]
    missing = [name for name in env_requirements if not env_present(name)]
    shard_rows: list[ShardRow] = [
        {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "size_gb": round(path.stat().st_size / (1024.0**3), 3),
        }
        for path in shard_paths
    ]
    total_size_bytes = sum(row["size_bytes"] for row in shard_rows if isinstance(row["size_bytes"], int))
    payload: JsonObject = {
        "schema_version": "PRISM.shard_r2_recovery_preflight.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "shard_dir": Path(args.shard_dir).as_posix(),
        "shards": shard_rows,
        "total_size_gb": round(total_size_bytes / (1024.0**3), 3),
        "required_env": env_requirements,
        "missing_env": missing,
        "r2_ready": not missing,
        "destructive_actions_permitted": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.report.with_suffix(args.report.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.report)
    status = "READY" if not missing else "BLOCKED_MISSING_R2_ENV"
    print(
        "shard_r2_recovery_preflight "
        f"status={status} shards={len(shard_rows)} total_size_gb={payload['total_size_gb']} "
        f"missing_env={','.join(missing) if missing else 'none'} report={args.report}"
    )
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
