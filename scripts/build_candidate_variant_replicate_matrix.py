#!/usr/bin/env python3
"""Build the GLP-1R generated-candidate x receptor-variant replicate matrix.

This is a planner, not a launcher. It makes the complete matrix explicit and
records why the full all-candidate/all-target/10-replica matrix cannot be
started blindly on the current workstation capacity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_INDEX = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_b_chronological/expanded_variant_run/"
    "phase3_topology_runnable_target_index.json"
)
DEFAULT_OUTPUT_ROOT = Path("/mnt/storage")
SMOKE_SUMMARY = Path("/tmp/prism_cross_motif_variant_smoke_latest.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=REPO_ROOT / "campaigns/glp1r_aleniglipron")
    parser.add_argument("--target-index", type=Path, default=DEFAULT_TARGET_INDEX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument(
        "--observed-replica-bytes",
        type=int,
        default=None,
        help="Override observed bytes per completed one-candidate/one-target replica.",
    )
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.replicas < 1:
        raise ValueError("--replicas must be >= 1")

    run_id = args.run_id or f"candidate_variant_matrix_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = args.output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_generated_candidates(args.campaign_root)
    targets_raw = json.loads(resolve(args.target_index).read_text(encoding="utf-8")).get("targets", [])
    receptor_targets = [target for target in targets_raw if target.get("kind") == "phase3_receptor_variant"]
    holo_controls = [target for target in targets_raw if target.get("kind") != "phase3_receptor_variant"]
    observed_replica_bytes = args.observed_replica_bytes or infer_observed_replica_bytes()
    free_bytes = shutil.disk_usage(args.output_root).free

    matrix_path = out_dir / "candidate_variant_replicate_matrix.jsonl"
    csv_path = out_dir / "candidate_variant_replicate_matrix.csv"
    rows = list(build_rows(candidates, receptor_targets, args.replicas))
    with matrix_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(csv_path, rows)

    excluded_controls_path = out_dir / "excluded_holo_control_targets.json"
    write_json(
        excluded_controls_path,
        {
            "reason": "Already-holo aleniglipron target topologies are controls/reference ligand surfaces; "
            "generated candidates must be appended to receptor-only targets to avoid double-ligand topology.",
            "count": len(holo_controls),
            "targets": holo_controls,
        },
    )

    estimated_full_bytes = len(rows) * observed_replica_bytes
    capacity_status = "BLOCKED_FULL_MATRIX_EXCEEDS_LOCAL_CAPACITY"
    if estimated_full_bytes < int(free_bytes * 0.80):
        capacity_status = "CAPACITY_OK"
    summary = {
        "schema_version": "PRISM.candidate_variant_replicate_matrix.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "campaign_root": str(resolve(args.campaign_root)),
        "target_index": str(resolve(args.target_index)),
        "candidate_unique_sha_count": len(candidates),
        "candidate_path_count": sum(len(row["duplicate_paths"]) + 1 for row in candidates),
        "receptor_variant_target_count": len(receptor_targets),
        "excluded_holo_control_target_count": len(holo_controls),
        "replicas_per_candidate_target": args.replicas,
        "matrix_run_count": len(rows),
        "observed_replica_bytes": observed_replica_bytes,
        "estimated_full_matrix_bytes": estimated_full_bytes,
        "estimated_full_matrix_tib": estimated_full_bytes / float(1024**4),
        "available_output_bytes": free_bytes,
        "available_output_tib": free_bytes / float(1024**4),
        "capacity_status": capacity_status,
        "matrix_jsonl": str(matrix_path),
        "matrix_csv": str(csv_path),
        "excluded_holo_control_targets": str(excluded_controls_path),
        "smoke_summary": find_smoke_summary(),
        "recommended_next_gate": "RUN_TOP_CANDIDATE_ONE_TARGET_10_REPLICATES_THEN_AGGREGATE",
    }
    write_json(out_dir / "candidate_variant_replicate_matrix_summary.json", summary)
    write_markdown(out_dir / "candidate_variant_replicate_matrix_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_generated_candidates(campaign_root: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(resolve(campaign_root).rglob("cand_*.sdf")):
        grouped.setdefault(sha256(path), []).append(path)
    candidates: list[dict[str, Any]] = []
    for digest, paths in sorted(grouped.items(), key=lambda item: canonical_priority(item[1])):
        primary = canonical_path(paths)
        candidates.append(
            {
                "candidate_id": primary.stem,
                "sdf": str(primary),
                "sdf_sha256": digest,
                "duplicate_paths": [str(path) for path in paths if path != primary],
            }
        )
    return candidates


def canonical_priority(paths: list[Path]) -> tuple[int, str]:
    primary = canonical_path(paths)
    return (0 if "gpu_dispatch_final" in primary.as_posix() else 1, primary.as_posix())


def canonical_path(paths: list[Path]) -> Path:
    priority = ("gpu_dispatch_final/sdf", "track_a_generative/gpu_dispatch/sdf", "epoch019_fixes/gpu_dispatch/sdf")
    for marker in priority:
        for path in paths:
            if marker in path.as_posix():
                return path
    return sorted(paths)[0]


def build_rows(candidates: list[dict[str, Any]], targets: list[dict[str, Any]], replicas: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for target in targets:
            target_path = str(resolve(Path(target["path"])))
            residue_map = target.get("sidecars", {}).get("residue_map", {}).get("path")
            if not residue_map:
                continue
            target_name = Path(target_path).name[: -len(".topology.json")]
            pdb_anchor = infer_pdb_anchor(target_name)
            for replica in range(replicas):
                rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "candidate_sdf": candidate["sdf"],
                        "candidate_sdf_sha256": candidate["sdf_sha256"],
                        "target_id": target_name,
                        "target_kind": target.get("kind"),
                        "target_topology": target_path,
                        "target_topology_sha256": target.get("sha256"),
                        "residue_map": str(resolve(Path(residue_map))),
                        "replica": replica,
                        "seed": 42042 + replica * 1000,
                        "pdb_anchor_active": pdb_anchor if pdb_anchor != "5VEX" else None,
                        "pdb_anchor_inactive": "5VEX" if pdb_anchor == "5VEX" else None,
                        "status": "PLANNED_NOT_RUN",
                    }
                )
    return rows


def infer_pdb_anchor(target_id: str) -> str:
    for anchor in ("5VEX", "6LN2", "6X1A", "6XOX"):
        if anchor in target_id:
            return anchor
    return "6XOX"


def infer_observed_replica_bytes() -> int:
    smoke = find_smoke_root()
    if smoke and smoke.exists():
        total = sum(path.stat().st_size for path in smoke.rglob("*") if path.is_file())
        if total > 0:
            return total
    return int(8.4 * 1024**3)


def find_smoke_root() -> Path | None:
    if SMOKE_SUMMARY.exists():
        text = SMOKE_SUMMARY.read_text(encoding="utf-8").strip()
        if text:
            return Path(text)
    matches = sorted(Path("/mnt/storage").glob("prism_cross_motif_variant_smoke_*"))
    return matches[-1] if matches else None


def find_smoke_summary() -> str | None:
    root = find_smoke_root()
    if not root:
        return None
    matches = sorted(root.rglob("candidate_pathb_validation_summary.json"))
    return str(matches[-1]) if matches else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id",
        "target_id",
        "replica",
        "seed",
        "candidate_sdf",
        "target_topology",
        "residue_map",
        "pdb_anchor_active",
        "pdb_anchor_inactive",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Candidate Variant Replicate Matrix",
        "",
        f"- candidate_unique_sha_count: {summary['candidate_unique_sha_count']}",
        f"- receptor_variant_target_count: {summary['receptor_variant_target_count']}",
        f"- excluded_holo_control_target_count: {summary['excluded_holo_control_target_count']}",
        f"- replicas_per_candidate_target: {summary['replicas_per_candidate_target']}",
        f"- matrix_run_count: {summary['matrix_run_count']}",
        f"- observed_replica_bytes: {summary['observed_replica_bytes']}",
        f"- estimated_full_matrix_tib: {summary['estimated_full_matrix_tib']:.3f}",
        f"- available_output_tib: {summary['available_output_tib']:.3f}",
        f"- capacity_status: {summary['capacity_status']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        "",
        "The full matrix is planned, not launched. Launching all rows blindly would exceed current local capacity.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
