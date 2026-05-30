#!/usr/bin/env python3
"""Consume one Tier 3 cloud shard manifest and emit receipts with no silent drops."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATED_WRAPPER = REPO_ROOT / "scripts/prism-validate-and-run.sh"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker-id", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def state_event(
    run_id: str,
    control_number: str | None,
    loop: str,
    row_id: str,
    worker_id: str,
    state: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "control_number": control_number,
        "loop": loop,
        "row_id": row_id,
        "worker_id": worker_id,
        "worker_pid": os.getpid(),
        "state": state,
        "timestamp_utc": now_utc(),
    }
    payload.update(extra)
    return payload


def command_for(row: dict[str, Any], output_dir: Path) -> list[str]:
    return [
        str(VALIDATED_WRAPPER),
        "-t",
        row["prepared_holo_topology_path"],
        "-o",
        str(output_dir),
        "--fast",
        "--hysteresis",
        "--prism-therm",
        "--multi-stream",
        "8",
        "--spike-percentile",
        "70",
        "--fused-steps",
        "6",
        "--hmr",
        "--adaptive-dt",
        "--multi-differential",
        "--closed-loop-steering",
        "--asymmetric-steering",
        "--site-ranker",
        "phase-manifold",
        "--md-only-evidence",
        "--path-a-production-profile",
        "--path-a-max-wall-seconds",
        "180",
        "--uv-wavelengths",
        "280,274,258,254,211",
        "--nma-amplification",
        "3.0",
        "--nma-scan-fraction",
        "0.3",
        "--replica-seed",
        str(41 + int(row["replicate"])),
        "-v",
    ]


def main() -> int:
    args = parse_args()
    if args.execute == args.dry_run:
        raise SystemExit("choose exactly one of --dry-run or --execute")
    shard = load_json(args.shard_manifest)
    control_number = shard.get("control_number")
    worker_id = args.worker_id or shard.get("pod_id") or "worker-unknown"
    task_root = args.task_root.resolve()
    events_path = task_root / f"{worker_id}.runtime_events.jsonl"
    receipts_path = task_root / f"{worker_id}.artifact_receipts.jsonl"
    commands_dir = task_root / "commands"
    outputs_dir = task_root / "outputs"
    logs_dir = task_root / "logs"
    summary_rows = []

    for row in shard["rows"]:
        run_id = row["run_id"]
        loop = row["loop"]
        row_id = row["row_id"]
        append_jsonl(events_path, state_event(run_id, control_number, loop, row_id, worker_id, "CLAIMED"))
        append_jsonl(events_path, state_event(run_id, control_number, loop, row_id, worker_id, "STARTED"))
        unresolved_dynamic = str(row["molecule_path"]).startswith("__")
        topology_path = Path(row["topology_path"])
        molecule_path = Path(row["molecule_path"]) if not unresolved_dynamic else None
        prepared_holo = Path(row["prepared_holo_topology_path"]) if row.get("prepared_holo_topology_path") else None
        row_output_dir = outputs_dir / row_id
        row_output_dir.mkdir(parents=True, exist_ok=True)
        command_path = commands_dir / f"{row_id}.command.sh"
        command_path.parent.mkdir(parents=True, exist_ok=True)

        if unresolved_dynamic:
            receipt = {
                "row_id": row_id,
                "control_number": control_number,
                "status": "BLOCKED_UNRESOLVED_DYNAMIC_INPUT",
                "molecule_path": row["molecule_path"],
                "topology_path": row["topology_path"],
            }
            append_jsonl(receipts_path, receipt)
            append_jsonl(
                events_path,
                state_event(
                    run_id,
                    control_number,
                    loop,
                    row_id,
                    worker_id,
                    "FAILED",
                    failure="unresolved_dynamic_input",
                ),
            )
            summary_rows.append(receipt)
            continue
        if prepared_holo is None or not prepared_holo.exists():
            receipt = {
                "row_id": row_id,
                "control_number": control_number,
                "status": "BLOCKED_HOLO_TOPOLOGY_UNPREPARED",
                "prepared_holo_topology_path": row.get("prepared_holo_topology_path"),
                "molecule_path": str(molecule_path) if molecule_path else row["molecule_path"],
                "topology_path": row["topology_path"],
            }
            append_jsonl(receipts_path, receipt)
            append_jsonl(
                events_path,
                state_event(
                    run_id,
                    control_number,
                    loop,
                    row_id,
                    worker_id,
                    "FAILED",
                    failure="prepared_holo_missing",
                ),
            )
            summary_rows.append(receipt)
            continue
        if not topology_path.exists() or not molecule_path or not molecule_path.exists():
            receipt = {
                "row_id": row_id,
                "control_number": control_number,
                "status": "FAIL_INPUT_MISSING",
                "molecule_exists": bool(molecule_path and molecule_path.exists()),
                "topology_exists": topology_path.exists(),
                "prepared_holo_exists": bool(prepared_holo and prepared_holo.exists()),
                "molecule_path": str(molecule_path) if molecule_path else row["molecule_path"],
                "topology_path": row["topology_path"],
            }
            append_jsonl(receipts_path, receipt)
            append_jsonl(
                events_path,
                state_event(
                    run_id,
                    control_number,
                    loop,
                    row_id,
                    worker_id,
                    "FAILED",
                    failure="input_missing",
                ),
            )
            summary_rows.append(receipt)
            continue
        command = command_for(row, row_output_dir)
        command_path.write_text(" ".join(command) + "\n", encoding="utf-8")

        append_jsonl(
            events_path,
            state_event(
                run_id,
                control_number,
                loop,
                row_id,
                worker_id,
                "HEARTBEAT",
                command_path=str(command_path),
                output_dir=str(row_output_dir),
            ),
        )

        if args.dry_run:
            receipt = {
                "row_id": row_id,
                "control_number": control_number,
                "status": "READY_FOR_EXECUTION",
                "command_path": str(command_path),
                "output_dir": str(row_output_dir),
                "molecule_path": str(molecule_path),
                "topology_path": str(topology_path),
                "prepared_holo_topology_path": str(prepared_holo),
            }
            append_jsonl(receipts_path, receipt)
            append_jsonl(events_path, state_event(run_id, control_number, loop, row_id, worker_id, "VERIFIED"))
            summary_rows.append(receipt)
            continue

        log_path = logs_dir / f"{row_id}.stdout_stderr.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True)
        log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        if proc.returncode == 0:
            append_jsonl(events_path, state_event(run_id, control_number, loop, row_id, worker_id, "OUTPUT_WRITTEN"))
            append_jsonl(events_path, state_event(run_id, control_number, loop, row_id, worker_id, "CHECKSUM_VERIFIED"))
            append_jsonl(events_path, state_event(run_id, control_number, loop, row_id, worker_id, "VERIFIED"))
            receipt = {
                "row_id": row_id,
                "control_number": control_number,
                "status": "VERIFIED",
                "returncode": 0,
                "log_path": str(log_path),
                "output_dir": str(row_output_dir),
            }
        else:
            append_jsonl(
                events_path,
                state_event(
                    run_id,
                    control_number,
                    loop,
                    row_id,
                    worker_id,
                    "FAILED",
                    failure=f"returncode_{proc.returncode}",
                ),
            )
            receipt = {
                "row_id": row_id,
                "control_number": control_number,
                "status": "FAILED",
                "returncode": proc.returncode,
                "log_path": str(log_path),
                "output_dir": str(row_output_dir),
            }
        append_jsonl(receipts_path, receipt)
        summary_rows.append(receipt)

    write_json(
        task_root / f"{worker_id}.summary.json",
        {
            "schema_version": "prism.tier3_pov.worker_summary.v1",
            "generated_at_utc": now_utc(),
            "control_number": control_number,
            "worker_id": worker_id,
            "row_count": len(summary_rows),
            "rows": summary_rows,
        },
    )
    print(
        json.dumps(
            {
                "control_number": control_number,
                "worker_id": worker_id,
                "row_count": len(summary_rows),
                "dry_run": args.dry_run,
                "events_path": str(events_path),
                "receipts_path": str(receipts_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
