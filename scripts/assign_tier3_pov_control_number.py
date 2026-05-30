#!/usr/bin/env python3
"""Assign and propagate a control number across a staged Tier 3 PoV run root."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--control-number", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def git_short_head(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    repo_root = run_root.parents[4]
    status_path = run_root / "RUNBOOK_STATUS.json"
    status = load_json(status_path)
    run_id = status["run_id"]
    control_number = args.control_number or status.get("control_number") or (
        f"PRISM-T3POV-{run_id.replace('tier3-pov-', '')}-{git_short_head(repo_root)}".upper()
    )

    updated = []
    for path in sorted(run_root.rglob("*.json")):
        body = load_json(path)
        if isinstance(body, dict):
            body["control_number"] = control_number
            write_json(path, body)
            updated.append(str(path.relative_to(run_root)))
    (run_root / "CONTROL_NUMBER.txt").write_text(control_number + "\n", encoding="utf-8")
    write_json(
        run_root / "verification" / "control_number_propagation_audit.json",
        {
            "schema_version": "prism.tier3_pov.control_number_propagation_audit.v1",
            "generated_at_utc": now_utc(),
            "run_root": str(run_root),
            "control_number": control_number,
            "updated_json_files": updated,
            "updated_json_count": len(updated),
            "control_number_file": "CONTROL_NUMBER.txt",
        },
    )
    print(json.dumps({"control_number": control_number, "updated_json_count": len(updated)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
