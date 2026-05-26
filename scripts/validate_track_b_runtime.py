#!/usr/bin/env python3
"""Validate an instantiated Track B runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism_dstw.calibration.track_b_artifacts import read_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    required = [
        args.runtime / "config" / "track_b_runtime_config.yaml",
        args.runtime / "config" / "oracle_config.yaml",
        args.runtime / "config" / "calibration_config.yaml",
        args.runtime / "manifests" / "artifact_manifest.json",
        args.runtime / "manifests" / "cloud_sync_manifest.json",
        args.runtime / "manifests" / "vectorize_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    artifact_manifest = read_json(args.runtime / "manifests" / "artifact_manifest.json") if not missing else {"artifacts": []}
    hash_mismatches = []
    for item in artifact_manifest.get("artifacts", []):
        path = Path(str(item["path"]))
        if not path.exists():
            missing.append(str(path))
            continue
        if sha256_file(path) != item["sha256"]:
            hash_mismatches.append(str(path))
    needed_names = {
        "chronology_locked_candidate_audit.json",
        "TRACK_B_CHRONOLOGICAL_CONTROL_DOSSIER.md",
        "nma_continuity_map.parquet",
        "hydration_continuity_map.parquet",
        "thermodynamic_continuity_map.parquet",
        "transition_chronology_tensor.parquet",
        "transition_chronology_report.json",
        "translational_calibration_adequacy_gate.json",
        "oracle_scorer",
    }
    present_names = {Path(str(item["path"])).name for item in artifact_manifest.get("artifacts", [])}
    missing.extend(sorted(str(name) for name in needed_names - present_names))
    runtime_oracle = args.runtime / "bin" / "oracle_scorer"
    build_oracle = Path("target/release/oracle_scorer")
    if not runtime_oracle.exists() and not build_oracle.exists():
        missing.append(str(runtime_oracle))
    if "oracle_scorer" not in present_names:
        missing.append("oracle_scorer missing from artifact_manifest.json")
    verdict = "TRACK_B_RUNTIME_VALID" if not missing and not hash_mismatches else "TRACK_B_RUNTIME_INVALID"
    print(
        json.dumps(
            {
                "runtime": str(args.runtime),
                "verdict": verdict,
                "missing": missing,
                "hash_mismatches": hash_mismatches,
                "artifact_count": len(artifact_manifest.get("artifacts", [])),
            },
            sort_keys=True,
        )
    )
    if verdict != "TRACK_B_RUNTIME_VALID":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
