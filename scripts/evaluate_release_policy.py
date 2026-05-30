#!/usr/bin/env python3
"""Evaluate sealed release outputs against release policy gates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._sealed_release_common import load_json, load_policy, now_utc, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("release_policy.yaml"))
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mark(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": passed, "detail": detail})


def destination_status(matrix_rows: list[dict[str, str]], destination: str) -> str | None:
    for row in matrix_rows:
        if row.get("destination") == destination:
            return row.get("status")
    return None


def main() -> int:
    args = parse_args()
    release_root = args.release_root.resolve()
    policy = load_policy((release_root / args.policy).resolve() if not args.policy.is_absolute() else args.policy)

    manifest_path = release_root / "MANIFEST.json"
    status_path = release_root / "RELEASE_STATUS.json"
    cold_restore_path = release_root / "COLD_RESTORE_VALIDATION.json"
    deficits_path = release_root / "RELEASE_DEFICITS.md"
    matrix_path = release_root / "DESTINATION_MATRIX.csv"
    readback_path = release_root / "DESTINATION_READBACK_VALIDATION.json"
    restore_source_matrix_path = release_root / "RESTORE_SOURCE_MATRIX.json"

    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    release_status = load_json(status_path) if status_path.exists() else {}
    cold_restore = load_json(cold_restore_path) if cold_restore_path.exists() else {}
    readback = load_json(readback_path) if readback_path.exists() else {}
    restore_source_matrix = load_json(restore_source_matrix_path) if restore_source_matrix_path.exists() else {}
    matrix_rows = read_csv_rows(matrix_path)
    checks: list[dict[str, Any]] = []

    mark(
        checks,
        "no_unaccounted_files",
        bool(manifest.get("inventory", {}).get("all_accounted_for")),
        f"all_accounted_for={manifest.get('inventory', {}).get('all_accounted_for')}",
    )
    mark(
        checks,
        "cold_restore",
        cold_restore.get("status") == "PASS",
        f"cold_restore_status={cold_restore.get('status')}",
    )
    mark(
        checks,
        "git_bundle",
        bool(manifest.get("git", {}).get("bundle", {}).get("verified")),
        f"git_bundle_verified={manifest.get('git', {}).get('bundle', {}).get('verified')}",
    )
    mark(
        checks,
        "github_tag",
        bool(release_status.get("git", {}).get("tag_pushed")),
        f"tag_pushed={release_status.get('git', {}).get('tag_pushed')}",
    )
    for destination in ("github", "ssd", "r2"):
        required = bool(policy.get("destinations", {}).get(destination, {}).get("required"))
        status = destination_status(matrix_rows, destination)
        passed = (status in {"PASS", "SOURCE_ONLY_PASS", "FULL_OPERATIONAL_PASS"}) or (not required and status is None)
        mark(checks, f"{destination}_destination", passed, f"status={status}")
    mark(
        checks,
        "manifest_sha256",
        bool(manifest.get("release_hashes", {}).get("manifest_sha256")),
        f"manifest_sha256={manifest.get('release_hashes', {}).get('manifest_sha256')}",
    )
    mark(
        checks,
        "archive_extract_test",
        bool(cold_restore.get("archive_extract", {}).get("verified")),
        f"archive_extract_verified={cold_restore.get('archive_extract', {}).get('verified')}",
    )
    mark(
        checks,
        "checksum_verification",
        bool(cold_restore.get("checksums", {}).get("verified")),
        f"checksums_verified={cold_restore.get('checksums', {}).get('verified')}",
    )
    mark(
        checks,
        "restore_test",
        cold_restore.get("status") == "PASS",
        f"cold_restore_status={cold_restore.get('status')}",
    )
    mark(
        checks,
        "operational_smoke",
        bool(cold_restore.get("smoke_validation", {}).get("passed")),
        f"smoke_passed={cold_restore.get('smoke_validation', {}).get('passed')}",
    )
    mark(
        checks,
        "candidate_manifest",
        bool(manifest.get("candidate_matrix", {}).get("manifest_path")),
        f"candidate_manifest={manifest.get('candidate_matrix', {}).get('manifest_path')}",
    )
    mark(
        checks,
        "status_reconciliation",
        bool(manifest.get("candidate_matrix", {}).get("status_reconciliation_path")),
        f"status_reconciliation={manifest.get('candidate_matrix', {}).get('status_reconciliation_path')}",
    )
    claim_boundary = manifest.get("claim_boundary", {})
    mark(
        checks,
        "claim_boundary_phase2c_status",
        claim_boundary.get("phase2c_status") == "SEALED",
        f"phase2c_status={claim_boundary.get('phase2c_status')}",
    )
    mark(
        checks,
        "claim_boundary_hydration_integration_status",
        claim_boundary.get("hydration_dstw_integration_status") == "IMPLEMENTED_AND_SMOKE_VERIFIED",
        f"hydration_dstw_integration_status={claim_boundary.get('hydration_dstw_integration_status')}",
    )
    mark(
        checks,
        "claim_boundary_hydration_full_run_status",
        claim_boundary.get("hydration_full_run_status") == "NOT_FULLY_RUN_104GB_INPUT_UNMEASURED",
        f"hydration_full_run_status={claim_boundary.get('hydration_full_run_status')}",
    )
    mark(
        checks,
        "readback_validation",
        readback.get("overall_status") == "PASS",
        f"readback_status={readback.get('overall_status')}",
    )
    mark(
        checks,
        "restore_source_matrix_present",
        bool(restore_source_matrix),
        f"restore_source_matrix_present={bool(restore_source_matrix)}",
    )
    mark(
        checks,
        "ssd_restore_source_status",
        restore_source_matrix.get("ssd_full_restore", {}).get("status") == "FULL_OPERATIONAL_PASS",
        f"ssd_full_restore_status={restore_source_matrix.get('ssd_full_restore', {}).get('status')}",
    )
    mark(
        checks,
        "r2_restore_source_status",
        restore_source_matrix.get("r2_full_restore", {}).get("status") == "FULL_OPERATIONAL_PASS",
        f"r2_full_restore_status={restore_source_matrix.get('r2_full_restore', {}).get('status')}",
    )
    deficits_text = deficits_path.read_text(encoding="utf-8") if deficits_path.exists() else ""
    mark(
        checks,
        "no_silent_omission",
        "silent omission" not in deficits_text.lower(),
        "deficits_scanned_for_silent_omission",
    )

    failed = [check for check in checks if not check["passed"]]
    core_names = {
        "no_unaccounted_files",
        "git_bundle",
        "manifest_sha256",
        "candidate_manifest",
        "status_reconciliation",
        "claim_boundary_phase2c_status",
        "claim_boundary_hydration_integration_status",
        "claim_boundary_hydration_full_run_status",
        "no_silent_omission",
    }
    core_passed = all(check["passed"] for check in checks if check["name"] in core_names)
    ssd_intermediate = destination_status(matrix_rows, "ssd") in {
        "SSD_STORAGE_WRITTEN_PENDING_READBACK",
        "SSD_STORAGE_PASS",
        "FULL_OPERATIONAL_PASS",
        "PARTIAL",
    }
    status = "PASS" if not failed else "PARTIAL" if core_passed and ssd_intermediate else "FAIL"
    evaluation = {
        "schema_version": "PRISM.sealed_release_policy_evaluation.v1",
        "evaluated_at_utc": now_utc(),
        "release_root": str(release_root),
        "policy_path": str(args.policy),
        "status": status,
        "checks": checks,
        "failed_checks": [check["name"] for check in failed],
    }
    out_path = args.json_out or (release_root / "POLICY_EVALUATION.json")
    write_json(out_path, evaluation)
    print(f"POLICY_EVALUATION={status}")
    print(f"OUTPUT={out_path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
