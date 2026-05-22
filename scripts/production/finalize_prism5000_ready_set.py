#!/usr/bin/env python3
"""
Create the exact verified-ready PRISM5000 engine queue.

Inputs:
* curated chain manifest from curate_prism5000_chain_targets.py
* ready_manifest.jsonl from prepare_prism5000_chain_targets.py

Output:
* prism5000_ready_engine_manifest.jsonl with merged curation + topology paths
* prism5000_ready_engine_queue.tsv for direct campaign scheduling
* prism5000_ready_summary.json with category/readiness audit counts
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-manifest", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-targets", type=int, default=5000)
    args = parser.parse_args()

    curated_rows = read_jsonl(args.curated_manifest)
    ready_rows = read_jsonl(args.ready_manifest)
    curated_by_id = {row["target_id"]: row for row in curated_rows}
    ready_by_id = {row["target_id"]: row for row in ready_rows if row.get("ready") is True}

    final_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for curated in curated_rows:
        target_id = curated["target_id"]
        if target_id in seen or target_id not in ready_by_id:
            continue
        ready = ready_by_id[target_id]
        topology_path = Path(ready["paths"]["topology_json"])
        if not topology_path.exists() or topology_path.stat().st_size <= 1000:
            continue
        merged = dict(curated)
        merged["prep_status"] = "ready"
        merged["prep_metrics"] = ready.get("metrics") or {}
        merged["paths"] = dict(curated.get("paths") or {})
        merged["paths"].update(ready.get("paths") or {})
        final_rows.append(merged)
        seen.add(target_id)
        if len(final_rows) >= args.n_targets:
            break

    if len(final_rows) < args.n_targets:
        raise SystemExit(
            f"not enough ready targets: requested={args.n_targets} ready={len(final_rows)} "
            f"curated={len(curated_rows)} ready_manifest={len(ready_rows)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "prism5000_ready_engine_manifest.jsonl"
    queue_path = args.out_dir / "prism5000_ready_engine_queue.tsv"
    summary_path = args.out_dir / "prism5000_ready_summary.json"
    write_jsonl(manifest_path, final_rows)

    with queue_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            delimiter="\t",
            fieldnames=[
                "target_id",
                "topology_json",
                "pdb_id",
                "auth_asym_id",
                "primary_category",
                "sequence_length",
                "selection_score",
            ],
        )
        writer.writeheader()
        for row in final_rows:
            writer.writerow(
                {
                    "target_id": row["target_id"],
                    "topology_json": row["paths"]["topology_json"],
                    "pdb_id": row["pdb_id"],
                    "auth_asym_id": row["auth_asym_id"],
                    "primary_category": row["primary_category"],
                    "sequence_length": row["sequence_length"],
                    "selection_score": row["selection_score"],
                }
            )

    category_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    source_kind_counts: dict[str, int] = {}
    for row in final_rows:
        category_counts[row["primary_category"]] = category_counts.get(row["primary_category"], 0) + 1
        for tag in row.get("difficulty_tags") or []:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        source_kind = (row.get("prep_metrics") or {}).get("source_kind", "unknown")
        source_kind_counts[source_kind] = source_kind_counts.get(source_kind, 0) + 1

    summary = {
        "requested_targets": args.n_targets,
        "final_ready_targets": len(final_rows),
        "curated_manifest": str(args.curated_manifest),
        "ready_manifest": str(args.ready_manifest),
        "engine_manifest": str(manifest_path),
        "engine_queue_tsv": str(queue_path),
        "category_counts": dict(sorted(category_counts.items())),
        "difficulty_tag_counts": dict(sorted(tag_counts.items())),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
