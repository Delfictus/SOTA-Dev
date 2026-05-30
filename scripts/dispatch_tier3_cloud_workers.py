#!/usr/bin/env python3
"""Stage RunPod-compatible worker job specs for a Tier 3 PoV portable bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--image", required=True, help="Container image reference placeholder or real image.")
    parser.add_argument("--provider", default="runpod", choices=["runpod"])
    parser.add_argument("--gpu-type", default="NVIDIA RTX 4090")
    parser.add_argument("--container-disk-gb", type=int, default=200)
    parser.add_argument("--bundle-mount-root", default="/workspace/bundle")
    parser.add_argument("--task-root-base", default="/workspace/runtime")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    bundle_root = args.bundle_root.resolve()
    bundle_manifest = load_json(bundle_root / "bundle_manifest.json")
    control_number = bundle_manifest.get("control_number")
    pod_plan_path = bundle_root / "cloud/pod_assignment_plan.csv"
    if not pod_plan_path.exists():
        raise SystemExit(f"missing pod assignment plan: {pod_plan_path}")
    import csv

    with pod_plan_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    job_specs = []
    for row in rows:
        loop = row["loop"]
        pod_id = row["pod_id"]
        shard_manifest = bundle_root / "cloud/shards" / loop / f"{pod_id}.manifest.json"
        container_shard_manifest = (
            Path(args.bundle_mount_root) / "cloud" / "shards" / loop / f"{pod_id}.manifest.json"
        )
        container_task_root = Path(args.task_root_base) / pod_id
        job_specs.append(
            {
                "provider": args.provider,
                "pod_id": pod_id,
                "loop": loop,
                "control_number": control_number,
                "image": args.image,
                "gpu_type": args.gpu_type,
                "container_disk_gb": args.container_disk_gb,
                "host_shard_manifest": str(shard_manifest),
                "container_shard_manifest": str(container_shard_manifest),
                "container_task_root": str(container_task_root),
                "env": {
                    "PRISM_CONTROL_NUMBER": control_number,
                    "PRISM_SHARD_MANIFEST": str(container_shard_manifest),
                    "PRISM_TASK_ROOT": str(container_task_root),
                    "PRISM_WORKER_ID": pod_id,
                },
                "command": [
                    "python3",
                    "scripts/run_tier3_cloud_worker.py",
                    "--shard-manifest",
                    str(container_shard_manifest),
                    "--task-root",
                    str(container_task_root),
                    "--execute",
                    "--worker-id",
                    pod_id,
                ],
                "expected_row_count": int(row["row_count"]),
            }
        )

    plan = {
        "schema_version": "prism.tier3_pov.dispatch_plan.v1",
        "generated_at_utc": now_utc(),
        "bundle_root": str(bundle_root),
        "control_number": control_number,
        "provider": args.provider,
        "image": args.image,
        "host_bundle_root": str(bundle_root),
        "container_bundle_root": args.bundle_mount_root,
        "container_task_root_base": args.task_root_base,
        "job_spec_count": len(job_specs),
        "job_specs": job_specs,
    }
    out = args.out or (bundle_root / "cloud/dispatch_plan.json")
    write_json(out, plan)
    print(json.dumps({"dispatch_plan": str(out), "job_spec_count": len(job_specs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
