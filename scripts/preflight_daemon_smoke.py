#!/usr/bin/env python3
"""Preflight the bounded Track A daemon smoke-test prerequisites."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_REPORT = TRACK_A / "daemon_smoke_preflight.json"

JsonObject: TypeAlias = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=TRACK_A / "gflownet_policy_v1.pt")
    parser.add_argument("--candidates", type=Path, default=TRACK_A / "gflownet_top_100_candidates.parquet")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required_env = ["CLOUDFLARE_API_URL", "CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET"]
    missing_env = [name for name in required_env if not os.environ.get(name)]
    required_files = [Path(args.policy), Path(args.candidates)]
    missing_files = [path.as_posix() for path in required_files if not path.is_file()]
    ready = not missing_env and not missing_files
    payload: JsonObject = {
        "schema_version": "PRISM.daemon_smoke_preflight.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "required_env": required_env,
        "missing_env": missing_env,
        "required_files": [path.as_posix() for path in required_files],
        "missing_files": missing_files,
        "daemon_smoke_ready": ready,
        "bounded_mode_required": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.report.with_suffix(args.report.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(args.report)
    status = "READY" if ready else "BLOCKED"
    print(
        "daemon_smoke_preflight "
        f"status={status} missing_env={','.join(missing_env) if missing_env else 'none'} "
        f"missing_files={','.join(missing_files) if missing_files else 'none'} report={args.report}"
    )
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
