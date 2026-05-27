#!/usr/bin/env python3
"""Generate Track B release manifest with local hashes."""

from __future__ import annotations

import argparse
from pathlib import Path

from prism_dstw.calibration.track_b_artifacts import sha256_file, write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso


REQUIRED_NAMES = [
    "track_b_data_admissibility.json",
    "track_b_artifact_inventory.parquet",
    "topology_region_registry.json",
    "genealogical_variant_panel.json",
    "variant_manifold_coverage_matrix.parquet",
    "variant_manifold_coverage_report.json",
    "translational_calibration_adequacy_gate.json",
    "phase_3_te_hub_variant_manifest.json",
    "transition_chronology_report.json",
    "nma_continuity_map.parquet",
    "hydration_continuity_map.parquet",
    "thermodynamic_continuity_map.parquet",
    "continuity_map_manifest.json",
    "chronology_locked_top_100_candidates.parquet",
    "chronology_locked_training_report.json",
    "chronology_locked_candidate_audit.json",
    "TRACK_B_CHRONOLOGICAL_CONTROL_DOSSIER.md",
    "TRANSLATIONAL_CALIBRATION_MANIFOLD_DOSSIER.md",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = []
    missing = []
    for name in REQUIRED_NAMES:
        path = args.track_b_root / name
        if not path.exists():
            missing.append(str(path))
            continue
        artifacts.append(
            {
                "name": name,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    chronology_tensor = Path("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/transition_chronology_tensor.parquet")
    if chronology_tensor.exists():
        artifacts.append(
            {
                "name": "transition_chronology_tensor.parquet",
                "path": str(chronology_tensor),
                "sha256": sha256_file(chronology_tensor),
                "size_bytes": chronology_tensor.stat().st_size,
            }
        )
    else:
        missing.append(str(chronology_tensor))
    runtime_root = args.track_b_root / "runtime"
    if runtime_root.exists():
        for path in sorted(runtime_root.rglob("*")):
            if not path.is_file():
                continue
            artifacts.append(
                {
                    "name": str(path.relative_to(args.track_b_root)),
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    else:
        missing.append(str(runtime_root))
    subtb_root = args.track_b_root / "subtb_spectral"
    if subtb_root.exists():
        for path in sorted(subtb_root.rglob("*")):
            if not path.is_file():
                continue
            artifacts.append(
                {
                    "name": f"subtb_spectral/{path.relative_to(subtb_root)}",
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    audit_root = Path(".audit-reports")
    if audit_root.exists():
        for path in sorted(audit_root.glob("track_b_*")):
            if not path.is_file():
                continue
            artifacts.append(
                {
                    "name": f"subagent_reports/{path.name}",
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    payload = {
        "schema_version": "track_b.release_manifest.v1",
        "created_at": utc_now_iso(),
        "id": "PRISM4D_TRACK_B_TRANSLATIONAL_CALIBRATION_RELEASE",
        "provenance_class": "L3_DERIVED",
        "source_artifacts": [item["path"] for item in artifacts],
        "evidence_paths": [item["path"] for item in artifacts],
        "track_b_root": str(args.track_b_root),
        "artifact_count": len(artifacts),
        "missing": missing,
        "artifacts": artifacts,
        "verdict": "RELEASE_MANIFEST_COMPLETE" if not missing else "RELEASE_MANIFEST_INCOMPLETE",
    }
    write_json(args.output, payload)
    print(f"track_b_release_manifest artifacts={len(artifacts)} missing={len(missing)} output={args.output}")


if __name__ == "__main__":
    main()
