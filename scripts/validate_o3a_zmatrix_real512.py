#!/usr/bin/env python3
"""Validate the O3A Z-matrix real512 pre-GFlowNet corpus freeze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
AUDIT_DIR = TRACK_A / "o3a_pre_gflownet_audit"
SELECTED = TRACK_A / "selected_ligand_pose.json"
SURVIVORS = TRACK_A / "vspace_survivors_real512_o3a_zmatrix.parquet"
TELEMETRY = TRACK_A / "vspace_real512_o3a_zmatrix_telemetry.json"
TOP100 = AUDIT_DIR / "top_100_o3a_zmatrix_candidates.parquet"
REPORT = AUDIT_DIR / "O3A_ZMATRIX_VALIDATION_REPORT.md"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def assert_check(checks: list[tuple[str, bool]], label: str, value: bool) -> None:
    checks.append((label, value))


def no_training_artifacts() -> bool:
    forbidden = list(TRACK_A.rglob("gflownet_policy_v1.pt"))
    forbidden.extend(TRACK_A.rglob("*training*.pt"))
    forbidden.extend(TRACK_A.rglob("*gflownet*checkpoint*"))
    return not forbidden


def main() -> int:
    checks: list[tuple[str, bool]] = []
    assert_check(checks, "selected_ligand_pose.json exists", SELECTED.exists())
    selected = read_json(SELECTED)
    selected_sdf = REPO_ROOT / str(selected.get("selected_ligand_sdf", ""))
    assert_check(checks, "selected_pose_method == compact_pseudo_ligand_o3a", selected.get("selected_pose_method") == "compact_pseudo_ligand_o3a")
    assert_check(checks, "selected_ligand_sdf exists", selected_sdf.exists())
    assert_check(checks, "survivor parquet exists", SURVIVORS.exists())
    assert_check(checks, "telemetry JSON exists", TELEMETRY.exists())
    telemetry = read_json(TELEMETRY)
    survivors = pl.read_parquet(SURVIVORS)
    assert_check(checks, "real_anchor_count_loaded >= 512", int(telemetry.get("real_anchor_count_loaded", 0)) >= 512)
    assert_check(checks, "mock_anchor_count == 0", int(telemetry.get("mock_anchor_count", -1)) == 0)
    assert_check(checks, "assembly_mode == zmatrix", telemetry.get("assembly_mode") == "zmatrix")
    assert_check(checks, "z_matrix_active_count > 0", int(telemetry.get("z_matrix_active_count", 0)) > 0)
    assert_check(checks, "rigid_fallback_count == 0", int(telemetry.get("rigid_fallback_count", -1)) == 0)
    assert_check(checks, "no_fly_voxels > 0", int(telemetry.get("no_fly_voxels", 0)) > 0)
    assert_check(checks, "complement_voxels > 0", int(telemetry.get("complement_voxels", 0)) > 0)
    assert_check(checks, "stable_occupied > 0", int(telemetry.get("stable_occupied", 0)) > 0)
    assert_check(checks, "dropped_bounds == 0", int(telemetry.get("dropped_bounds", -1)) == 0)
    assert_check(checks, "survivors > 0", survivors.height > 0)
    assert_check(checks, "unique_smiles_count > 0", survivors["canonical_smiles"].n_unique() > 0)
    assert_check(checks, "top 100 parquet exists", TOP100.exists())
    assert_check(checks, "no GFlowNet training output generated", no_training_artifacts())
    passed = all(value for _label, value in checks)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# O3A Z-Matrix Validation Report\n\n"
        + "\n".join(f"- [{'x' if value else ' '}] {label}" for label, value in checks)
        + f"\n\nValidation status: {'PASS' if passed else 'FAIL'}\n"
    )
    print(f"validation_status={'PASS' if passed else 'FAIL'} report={REPORT}")
    if not passed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
