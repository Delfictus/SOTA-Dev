#!/usr/bin/env python3
"""Lane A — freeze and manifest the completed-target extraction package.

Emits /tmp/engine_full_profiles/frozen_package_manifest.json + R2 offload plan.
Read-only; does not delete local copies.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

OUT = Path("/tmp/engine_full_profiles")
MANIFEST_PATH = OUT / "frozen_package_manifest.json"
OFFLOAD_CMD_PATH = OUT / "r2_offload_commands.sh"

TARGETS = ["wrn_apo", "menin_apo", "smarca2_brd_apo", "pkmyt1_apo",
           "kras_g12d_apo", "m1_2nvp", "m1_1xhx"]

ARTIFACT_CLASSES = [
    ("engine_full_profile_full", "{target}.engine_full_profile_full.json"),
    ("site_tag_profiles_full_jsonl", "{target}.site_tag_profiles.full.jsonl"),
    ("spike_events_export_manifest", "{target}.spike_events_export_manifest.json"),
    ("actual_keyset_binding_sites", "{target}.binding_sites.json.actual_keyset.json"),
    ("actual_keyset_kcc_visualization", "{target}.kcc_visualization.json.actual_keyset.json"),
    ("actual_keyset_kcc_validation", "{target}.kcc_validation.json.actual_keyset.json"),
    ("actual_keyset_ensemble_trajectory", "{target}.ensemble_trajectory.json.actual_keyset.json"),
    ("actual_keyset_rerank_result", "{target}.rerank_result.json.actual_keyset.json"),
    ("actual_keyset_ground_truth", "{target}.ground_truth.json.actual_keyset.json"),
    ("actual_keyset_residue_map", "{target}.residue_map.json.actual_keyset.json"),
    ("actual_keyset_topology", "{target}.topology.json.actual_keyset.json"),
    ("actual_keyset_evaluation", "{target}.evaluation.json.actual_keyset.json"),
]

GLOBAL_CLASSES = [
    ("schema_vs_actual_diff", "schema_vs_actual_diff.csv"),
    ("completeness_certificate", "completeness_certificate.json"),
    ("engine_file_manifest", "engine_file_manifest.csv"),
    ("field_coverage_matrix", "field_coverage_matrix.csv"),
    ("completeness_summary", "completeness_summary.csv"),
]


def sha256(p: Path, block=1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = []
    for t in TARGETS:
        for cls, tmpl in ARTIFACT_CLASSES:
            p = OUT / tmpl.format(target=t)
            if not p.exists():
                files.append({"target": t, "artifact_class": cls, "path": str(p),
                              "exists": False, "size_bytes": 0, "sha256": None})
                continue
            files.append({
                "target": t, "artifact_class": cls, "path": str(p),
                "exists": True, "size_bytes": p.stat().st_size,
                "sha256": sha256(p),
            })
    for cls, name in GLOBAL_CLASSES:
        p = OUT / name
        files.append({
            "target": "__global__", "artifact_class": cls, "path": str(p),
            "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0,
            "sha256": sha256(p) if p.exists() else None,
        })

    total_bytes = sum(f["size_bytes"] for f in files)
    manifest = {
        "version": "m1_1_completed_extraction_v1",
        "frozen_at_utc": "2026-04-18T01:45:00Z",
        "n_targets": len(TARGETS),
        "n_files": len(files),
        "total_bytes": total_bytes,
        "target_list": TARGETS,
        "files": files,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    # R2 offload commands — do NOT execute; just write the plan
    bucket = "prism-extraction"
    prefix = "m1_1/completed_targets_v1"
    lines = [
        "#!/bin/bash",
        "# Cloudflare R2 offload plan for frozen completed-target extraction package.",
        "# NON-DESTRUCTIVE: local copies remain at /tmp/engine_full_profiles/ after upload.",
        "# Run with: bash r2_offload_commands.sh",
        "set -euo pipefail",
        "R2_BUCKET=" + bucket,
        "R2_PREFIX=" + prefix,
        "",
        "# upload every file in the frozen manifest",
    ]
    for f in files:
        if not f["exists"]:
            continue
        src = f["path"]
        dest_key = f"{prefix}/" + (f['target'] + '/' if f['target'] != '__global__' else '') + Path(src).name
        lines.append(f"wrangler r2 object put $R2_BUCKET/{dest_key} --file {src}  # sha256={f['sha256']}")
    lines.append("")
    lines.append("# manifest upload (last)")
    lines.append(f"wrangler r2 object put $R2_BUCKET/{prefix}/frozen_package_manifest.json --file {MANIFEST_PATH}")
    lines.append("echo 'R2 offload complete — local copies retained.'")
    OFFLOAD_CMD_PATH.write_text("\n".join(lines))
    OFFLOAD_CMD_PATH.chmod(0o755)

    # print summary
    by_target = {}
    for f in files:
        by_target.setdefault(f["target"], []).append(f)
    print(f"{'target':<18} {'n_files':>8} {'total_bytes':>14}")
    for t, lst in by_target.items():
        nb = sum(f["size_bytes"] for f in lst)
        print(f"  {t:<16} {len(lst):>8} {nb:>14}")
    print()
    print(f"n_files total: {len(files)}")
    print(f"total_bytes: {total_bytes}  ({total_bytes/1024/1024:.2f} MB)")
    print(f"manifest: {MANIFEST_PATH}")
    print(f"offload_plan: {OFFLOAD_CMD_PATH}")


if __name__ == "__main__":
    main()
