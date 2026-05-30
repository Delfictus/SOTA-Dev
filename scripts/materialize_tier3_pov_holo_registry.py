#!/usr/bin/env python3
"""Build and optionally execute pre-dispatch holo topology compile tasks for Tier 3 PoV rows."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lib.prism_runtime import resolve_prism_dock_python


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def default_python_exec() -> str:
    candidate = resolve_prism_dock_python()
    if candidate.exists():
        return str(candidate)
    return "python3"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pair-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--python-exec", default=default_python_exec())
    parser.add_argument("--preserve-input-placement", action="store_true")
    parser.add_argument(
        "--existing-pair",
        action="append",
        default=None,
        help="Register a known-good prepared holo as pair_id=/abs/path/to/topology.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def safe(text: str) -> str:
    return SAFE.sub("_", text)


def pair_id(row: dict[str, Any]) -> str:
    return f"{safe(row['molecule_slot'].lower())}__{safe(row['target_id'].lower())}"


def iter_rows(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for manifest_path in [
        run_root / "loop2/execution_manifest.template.json",
        run_root / "loop3/execution_manifest.template.json",
    ]:
        body = load_json(manifest_path)
        rows.extend(body["rows"])
    return rows


def main() -> int:
    args = parse_args()
    if args.execute == args.dry_run:
        raise SystemExit("choose exactly one of --execute or --dry-run")
    run_root = args.run_root.resolve()
    output_root = run_root / "prepared_holo"
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.pair_id or [])
    existing_map: dict[str, str] = {}
    for item in args.existing_pair or []:
        if "=" not in item:
            raise SystemExit(f"invalid --existing-pair entry: {item}")
        key, value = item.split("=", 1)
        existing_map[key.strip()] = value.strip()

    unique_pairs: dict[str, dict[str, Any]] = {}
    for row in iter_rows(run_root):
        pid = pair_id(row)
        unresolved = str(row["molecule_path"]).startswith("__")
        topology_path = Path(row["topology_path"])
        molecule_path = None if unresolved else Path(row["molecule_path"])
        if pid not in unique_pairs:
            unique_pairs[pid] = {
                "pair_id": pid,
                "loop_origins": [row["loop"]],
                "source_row_ids": [row["row_id"]],
                "molecule_slot": row["molecule_slot"],
                "target_id": row["target_id"],
                "molecule_path": row["molecule_path"],
                "topology_path": row["topology_path"],
                "unresolved_dynamic": unresolved,
                "molecule_exists": bool(molecule_path and molecule_path.exists()),
                "topology_exists": topology_path.exists(),
            }
        else:
            unique_pairs[pid]["source_row_ids"].append(row["row_id"])
            if row["loop"] not in unique_pairs[pid]["loop_origins"]:
                unique_pairs[pid]["loop_origins"].append(row["loop"])

    tasks = []
    count = 0
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    for pid, item in sorted(unique_pairs.items()):
        if selected and pid not in selected:
            continue
        if args.limit and count >= args.limit:
            break
        count += 1
        task_dir = output_root / pid
        task_dir.mkdir(parents=True, exist_ok=True)
        candidate_id = safe(item["molecule_slot"])
        command = [
            args.python_exec,
            str(REPO_ROOT / "scripts/compile_candidate_holo_topology.py"),
            "--candidate-id",
            candidate_id,
            "--sdf",
            item["molecule_path"],
            "--base-receptor-topology",
            item["topology_path"],
            "--output-dir",
            str(task_dir),
            "--condition-prefix",
            f"{item['target_id']}_HOLO",
        ]
        if args.preserve_input_placement:
            command.append("--preserve-input-placement")
        record = dict(item)
        record["compile_command"] = command
        record["task_dir"] = str(task_dir)
        record["status"] = "UNRESOLVED_DYNAMIC_INPUT" if item["unresolved_dynamic"] else "READY"
        if pid in existing_map:
            existing_path = existing_map[pid]
            record["prepared_holo_topology_path"] = existing_path
            record["status"] = "REGISTERED_EXISTING_PREPARED_HOLO" if Path(existing_path).exists() else "REGISTERED_EXISTING_PREPARED_HOLO_MISSING"
            tasks.append(record)
            continue
        if args.execute and record["status"] == "READY":
            proc = subprocess.run(command, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
            (task_dir / "compile.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            if proc.returncode == 0:
                compiled = None
                for line in proc.stdout.splitlines():
                    if line.startswith("compiled_topology="):
                        compiled = line.split("=", 1)[1].strip()
                        break
                record["status"] = "COMPILED" if compiled else "COMPILED_PATH_UNRESOLVED"
                record["prepared_holo_topology_path"] = compiled
            else:
                record["status"] = f"COMPILE_FAILED_{proc.returncode}"
                record["prepared_holo_topology_path"] = None
            record["returncode"] = proc.returncode
        else:
            record["prepared_holo_topology_path"] = None
        tasks.append(record)

    manifest = {
        "schema_version": "prism.tier3_pov.holo_compile_manifest.v1",
        "generated_at_utc": now_utc(),
        "run_root": str(run_root),
        "execute": args.execute,
        "task_count": len(tasks),
        "tasks": tasks,
    }
    registry = {
        "schema_version": "prism.tier3_pov.holo_registry.v1",
        "generated_at_utc": now_utc(),
        "run_root": str(run_root),
        "prepared": [
            {
                "pair_id": task["pair_id"],
                "prepared_holo_topology_path": task.get("prepared_holo_topology_path"),
                "molecule_slot": task["molecule_slot"],
                "target_id": task["target_id"],
                "status": task["status"],
            }
            for task in tasks
        ],
    }
    write_json(output_root / "holo_compile_manifest.json", manifest)
    write_json(output_root / "holo_registry.json", registry)
    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "execute": args.execute,
                "compiled_count": sum(1 for row in tasks if row["status"] == "COMPILED"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
