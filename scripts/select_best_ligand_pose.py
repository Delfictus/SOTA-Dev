#!/usr/bin/env python3
"""Select the guarded 6XOX ligand pose for O3A Z-matrix pruning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_MANIFEST = TRACK_A / "ALENI-PARENT_6XOX_pose_reconciliation_manifest.json"
DEFAULT_OUTPUT = TRACK_A / "selected_ligand_pose.json"
O3A_RELAXED = TRACK_A / "ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf"
O3A_BEST = TRACK_A / "ALENI-PARENT_6XOX_frame_o3a_best.sdf"
KABSCH_FALLBACK = TRACK_A / "ALENI-PARENT_6XOX_frame_minimized.sdf"

JsonObject: TypeAlias = dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_manifest(path: Path) -> JsonObject:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return cast(JsonObject, payload)


def validation_block(manifest: JsonObject) -> JsonObject:
    comparison = manifest.get("pose_comparison")
    if not isinstance(comparison, dict):
        raise ValueError("pose manifest missing pose_comparison")
    validation = comparison.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("pose manifest missing pose_comparison.validation")
    return cast(JsonObject, validation)


def o3a_relaxed_valid(manifest: JsonObject) -> bool:
    validation = validation_block(manifest)
    return (
        manifest.get("selected_alignment_method") == "compact_pseudo_ligand_o3a"
        and bool(manifest.get("ligand_inside_grid")) is True
        and validation.get("guard_level_passed") in {"standard", "strict"}
        and float(validation.get("no_fly_min_distance_A", -1.0)) >= 4.0
        and float(validation.get("complement_overlap_after_o3a", -1.0))
        > float(validation.get("complement_overlap_after_kabsch", float("inf")))
        and float(validation.get("post_min_heavy_distance_A", -1.0)) >= 1.5
        and O3A_RELAXED.exists()
    )


def o3a_best_valid(manifest: JsonObject) -> bool:
    validation = validation_block(manifest)
    return (
        O3A_BEST.exists()
        and manifest.get("selected_alignment_method") == "compact_pseudo_ligand_o3a"
        and bool(manifest.get("ligand_inside_grid")) is True
        and validation.get("guard_level_passed") in {"standard", "strict"}
        and float(validation.get("no_fly_min_distance_A", -1.0)) >= 4.0
        and float(validation.get("complement_overlap_after_o3a", -1.0))
        > float(validation.get("complement_overlap_after_kabsch", float("inf")))
        and float(validation.get("post_min_heavy_distance_A", -1.0)) >= 1.5
    )


def build_selection(manifest_path: Path = DEFAULT_MANIFEST) -> JsonObject:
    manifest = read_manifest(manifest_path)
    validation = validation_block(manifest)
    fallback_used = False
    fallback_reason = ""

    if o3a_relaxed_valid(manifest):
        selected = O3A_RELAXED
        selected_method = str(manifest["selected_alignment_method"])
    elif o3a_best_valid(manifest):
        selected = O3A_BEST
        selected_method = str(manifest["selected_alignment_method"])
        fallback_reason = "relaxed O3A pose absent; selected best O3A pose"
    else:
        if o3a_relaxed_valid(manifest):
            raise RuntimeError("valid O3A-relaxed pose exists but selector reached fallback branch")
        selected = KABSCH_FALLBACK
        selected_method = "kabsch_fallback"
        fallback_used = True
        fallback_reason = "all O3A poses failed selection guards"

    if selected_method == "kabsch_fallback" and o3a_relaxed_valid(manifest):
        raise RuntimeError("selected Kabsch fallback while a valid O3A-relaxed pose exists")
    if not selected.exists():
        raise FileNotFoundError(selected)

    return {
        "selected_ligand_sdf": rel(selected),
        "selected_pose_method": selected_method,
        "guard_level_passed": validation.get("guard_level_passed"),
        "o3a_score": manifest.get("o3a_score"),
        "complement_overlap_before": validation.get("complement_overlap_after_kabsch"),
        "complement_overlap_after": validation.get("complement_overlap_after_o3a"),
        "collision_distance_before_A": manifest.get("rigid_core_clash_before"),
        "collision_distance_after_A": validation.get("post_min_heavy_distance_A"),
        "no_fly_min_distance_A": validation.get("no_fly_min_distance_A"),
        "ligand_inside_grid": manifest.get("ligand_inside_grid"),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "input_manifest_sha256": sha256_file(manifest_path),
        "selected_sdf_sha256": sha256_file(selected),
    }


def main() -> int:
    payload = build_selection()
    DEFAULT_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "selected_ligand_pose "
        f"selected_ligand_sdf={payload['selected_ligand_sdf']} "
        f"selected_pose_method={payload['selected_pose_method']} "
        f"guard_level_passed={payload['guard_level_passed']} "
        f"o3a_score={payload['o3a_score']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
