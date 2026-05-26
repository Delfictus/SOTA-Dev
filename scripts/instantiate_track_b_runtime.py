#!/usr/bin/env python3
"""Instantiate Track B runtime directory with configs and manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from prism_dstw.calibration.track_b_artifacts import sha256_file, write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso


def _append_artifact(artifacts: list[dict[str, Any]], path: Path) -> None:
    if path.exists() and path.is_file():
        artifacts.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--track-b-root", type=Path, required=True)
    parser.add_argument("--output-runtime", type=Path, required=True)
    args = parser.parse_args()
    for subdir in ("config", "manifests", "logs", "outputs", "bin"):
        (args.output_runtime / subdir).mkdir(parents=True, exist_ok=True)
    config_dir = args.output_runtime / "config"
    manifest_dir = args.output_runtime / "manifests"
    runtime_oracle = args.output_runtime / "bin" / "oracle_scorer"
    source_oracle = Path("target/release/oracle_scorer")
    if source_oracle.exists():
        shutil.copy2(source_oracle, runtime_oracle)
        runtime_oracle.chmod(0o755)

    (config_dir / "track_b_runtime_config.yaml").write_text(
        f"campaign: {args.campaign}\ntrack_b_root: {args.track_b_root}\ncreated_at: {utc_now_iso()}\n",
        encoding="utf-8",
    )
    (config_dir / "oracle_config.yaml").write_text(
        "\n".join(
            [
                f"oracle_binary: {runtime_oracle}",
                "continuity_admissibility: true",
                f"nma_continuity_map: {args.track_b_root / 'nma_continuity_map.parquet'}",
                f"hydration_continuity_map: {args.track_b_root / 'hydration_continuity_map.parquet'}",
                f"thermodynamic_continuity_map: {args.track_b_root / 'thermodynamic_continuity_map.parquet'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "calibration_config.yaml").write_text(
        f"adequacy_gate: {args.track_b_root / 'translational_calibration_adequacy_gate.json'}\n",
        encoding="utf-8",
    )
    artifacts: list[dict[str, Any]] = []
    for path in sorted(args.track_b_root.glob("*")):
        if path.is_file():
            if path.name == "track_b_release_manifest.json":
                continue
            _append_artifact(artifacts, path)
    _append_artifact(
        artifacts,
        args.campaign / "integrated_spike_events" / "n80_full_scale" / "transition_chronology_tensor.parquet",
    )
    _append_artifact(artifacts, runtime_oracle)
    for report in sorted(Path(".audit-reports").glob("track_b_*.md")):
        _append_artifact(artifacts, report)
    for report in sorted(Path(".audit-reports").glob("track_b_*.log")):
        _append_artifact(artifacts, report)
    artifact_manifest = {
        "schema_version": "track_b.runtime_artifact_manifest.v1",
        "created_at": utc_now_iso(),
        "artifacts": artifacts,
    }
    write_json(manifest_dir / "artifact_manifest.json", artifact_manifest)
    cloud_items = [
        {"path": item["path"], "target": f"r2://prism-archive/track_b/{Path(item['path']).name}", "sha256": item["sha256"]}
        for item in artifacts
    ]
    write_json(
        manifest_dir / "cloud_sync_manifest.json",
        {
            "schema_version": "track_b.cloud_sync_manifest.v1",
            "created_at": utc_now_iso(),
            "mode": "copy_only",
            "artifacts": cloud_items,
        },
    )
    write_json(
        manifest_dir / "vectorize_manifest.json",
        {
            "schema_version": "track_b.vectorize_manifest.v1",
            "created_at": utc_now_iso(),
            "metadata_only": True,
            "indexes": ["candidate_embeddings", "chronology_conditioned_embeddings", "manifold_coverage_embeddings"],
        },
    )
    print(f"track_b_runtime_instantiated runtime={args.output_runtime} artifacts={len(artifacts)}")


if __name__ == "__main__":
    main()
