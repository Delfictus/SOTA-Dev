#!/usr/bin/env python3
"""Verify a sealed release from its packaged artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._sealed_release_common import (
    detect_symlink_leaks,
    ensure_temp_dir,
    load_json,
    now_utc,
    run,
    tar_extract,
    verify_git_bundle,
    verify_sha256_manifest,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--expected-release-id")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--run-db-checks", action="store_true")
    parser.add_argument("--run-cuda-checks", action="store_true")
    parser.add_argument("--run-candidate-checks", action="store_true")
    parser.add_argument("--skip-expensive-tests", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def check_path(path: Path, checks: list[dict[str, Any]], name: str, detail: str) -> bool:
    passed = path.exists()
    checks.append({"name": name, "passed": passed, "detail": detail, "path": str(path)})
    return passed


def main() -> int:
    args = parse_args()
    release_root = args.release_root.resolve()
    manifest_path = release_root / "MANIFEST.json"
    manifest_jsonl_path = release_root / "MANIFEST.jsonl"
    checksums_path = release_root / "CHECKSUMS.sha256"
    readme_path = release_root / "README_RESTORE.md"
    claim_path = release_root / "RELEASE_CLAIM_BOUNDARY.md"
    release_status_path = release_root / "RELEASE_STATUS.json"
    restore_source_matrix_path = release_root / "RESTORE_SOURCE_MATRIX.json"
    log_path = release_root / "validation" / "verify_restored_release.log"
    checks: list[dict[str, Any]] = []

    check_path(manifest_path, checks, "manifest_json", "MANIFEST.json exists")
    check_path(manifest_jsonl_path, checks, "manifest_jsonl", "MANIFEST.jsonl exists")
    check_path(checksums_path, checks, "checksums", "CHECKSUMS.sha256 exists")
    check_path(readme_path, checks, "restore_readme", "README_RESTORE.md exists")
    check_path(claim_path, checks, "claim_boundary", "RELEASE_CLAIM_BOUNDARY.md exists")
    check_path(release_status_path, checks, "release_status", "RELEASE_STATUS.json exists")
    check_path(restore_source_matrix_path, checks, "restore_source_matrix", "RESTORE_SOURCE_MATRIX.json exists")

    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    claim_boundary = manifest.get("claim_boundary", {}) if isinstance(manifest, dict) else {}
    if args.expected_release_id:
        checks.append(
            {
                "name": "release_id_match",
                "passed": manifest.get("release_id") == args.expected_release_id,
                "detail": f"manifest_release_id={manifest.get('release_id')}",
            }
        )

    checksum_result = verify_sha256_manifest(checksums_path, cwd=release_root, log_path=log_path) if checksums_path.exists() else {"verified": False}
    checks.append(
        {
            "name": "checksum_verification",
            "passed": bool(checksum_result.get("verified")),
            "detail": f"sha256sum_returncode={checksum_result.get('returncode')}",
        }
    )

    bundle_path = release_root / "git" / f"{manifest.get('release_id', 'release')}.all-refs.bundle"
    if not bundle_path.exists():
        bundles = sorted((release_root / "git").glob("*.bundle")) if (release_root / "git").exists() else []
        if bundles:
            bundle_path = bundles[0]
    bundle_result = verify_git_bundle(bundle_path, log_path=log_path) if bundle_path.exists() else {"verified": False}
    checks.append(
        {
            "name": "git_bundle_verification",
            "passed": bool(bundle_result.get("verified")),
            "detail": f"bundle={bundle_path}",
        }
    )

    chunk_manifest_path = release_root / "chunk_manifest.json"
    chunk_manifest = load_json(chunk_manifest_path) if chunk_manifest_path.exists() else {}
    archive_path = release_root / "archives" / "repo_working_tree.tar.zst"
    if not archive_path.exists():
        archive_candidates = sorted((release_root / "archives").glob("repo_working_tree.tar*")) if (release_root / "archives").exists() else []
        if archive_candidates:
            archive_path = archive_candidates[0]
    extract_root = ensure_temp_dir("sealed-restore-")
    if chunk_manifest:
        chunk_paths = [release_root / chunk["path"] for chunk in chunk_manifest.get("chunks", [])]
        chunk_list = " ".join(str(path) for path in chunk_paths)
        archive_extract = run(
            [
                "bash",
                "-lc",
                f"cat {chunk_list} | zstd -dc | tar -xpf - -C {str(extract_root)!r}",
            ],
            check=False,
            log_path=log_path,
        )
        archive_extract = {
            "archive": str(chunk_manifest_path),
            "target_dir": str(extract_root),
            "returncode": archive_extract.returncode,
            "stdout": archive_extract.stdout,
            "stderr": archive_extract.stderr,
            "verified": archive_extract.returncode == 0,
            "chunk_count": len(chunk_paths),
        }
    else:
        archive_extract = tar_extract(archive_path, extract_root, log_path=log_path) if archive_path.exists() else {"verified": False}
    checks.append(
        {
            "name": "archive_extract",
            "passed": bool(archive_extract.get("verified")),
            "detail": f"archive={archive_path}",
        }
    )

    repo_snapshot = manifest.get("repo_snapshot", {})
    extracted_source = extract_root / repo_snapshot.get("top_level_dir", "repo")
    if repo_snapshot.get("restored_repo_relative_path"):
        extracted_source = extracted_source / repo_snapshot["restored_repo_relative_path"]
    if not extracted_source.exists():
        children = [path for path in extract_root.iterdir()] if extract_root.exists() else []
        if len(children) == 1 and children[0].is_dir():
            extracted_source = children[0]
    checks.append(
        {
            "name": "source_tree_present",
            "passed": extracted_source.exists(),
            "detail": f"extracted_source={extracted_source}",
        }
    )

    if extracted_source.exists():
        for relpath in ("scripts", "src", "crates", "campaigns/glp1r_aleniglipron/PHASE2C_SEALED_MANIFEST.json"):
            checks.append(
                {
                    "name": f"source_contains_{relpath.replace('/', '_')}",
                    "passed": (extracted_source / relpath).exists(),
                    "detail": relpath,
                }
            )
        leak_prefixes = ["/home/diddy/Desktop/Prism4D-bio"]
        leaks = detect_symlink_leaks(extracted_source, leak_prefixes)
        checks.append(
            {
                "name": "symlink_leaks",
                "passed": not leaks,
                "detail": f"leak_count={len(leaks)}",
            }
        )
        import_result = run(
            [
                "python3",
                "-c",
                (
                    "import sys; "
                    f"sys.path.insert(0, {str(extracted_source / 'src')!r}); "
                    f"sys.path.insert(0, {str(extracted_source)!r}); "
                    "import prism_dstw; "
                    "print('python_import=PASS')"
                ),
            ],
            check=False,
            log_path=log_path,
        )
        checks.append(
            {
                "name": "python_importability",
                "passed": import_result.returncode == 0,
                "detail": import_result.stdout.strip() or import_result.stderr.strip(),
            }
        )
        for relpath in ("Cargo.toml", "Cargo.lock", "release_policy.yaml"):
            checks.append(
                {
                    "name": f"restored_{relpath.replace('/', '_')}",
                    "passed": (extracted_source / relpath).exists(),
                    "detail": relpath,
                }
            )
        checks.append(
            {
                "name": "claim_boundary_hydration_integration_status",
                "passed": claim_boundary.get("hydration_dstw_integration_status") == "IMPLEMENTED_AND_SMOKE_VERIFIED",
                "detail": f"hydration_dstw_integration_status={claim_boundary.get('hydration_dstw_integration_status')}",
            }
        )
        checks.append(
            {
                "name": "claim_boundary_hydration_full_run_status",
                "passed": claim_boundary.get("hydration_full_run_status") in {"NOT_RUN_FULL_SCALE", "NOT_FULLY_RUN_104GB_INPUT_UNMEASURED"},
                "detail": f"hydration_full_run_status={claim_boundary.get('hydration_full_run_status')}",
            }
        )
        checks.append(
            {
                "name": "claim_boundary_candidate_matrix_completion_status",
                "passed": claim_boundary.get("candidate_matrix_completion_status") == "PARTIAL_ONLY_OBSERVED_OUTPUTS_CLAIMABLE",
                "detail": f"candidate_matrix_completion_status={claim_boundary.get('candidate_matrix_completion_status')}",
            }
        )
        if args.run_cuda_checks:
            cuda_paths = list((extracted_source / "crates").rglob("*.ptx"))
            checks.append(
                {
                    "name": "cuda_ptx_presence",
                    "passed": bool(cuda_paths) or any((release_root / "runtime").rglob("*.ptx")),
                    "detail": f"restored_ptx_count={len(cuda_paths)}",
                }
            )
        if args.run_db_checks:
            db_meta = manifest.get("database_capture", {})
            checks.append(
                {
                    "name": "database_capture_metadata",
                    "passed": bool(db_meta),
                    "detail": f"database_entries={len(db_meta.get('entries', [])) if isinstance(db_meta, dict) else 0}",
                }
            )
        if args.run_candidate_checks:
            candidate_manifest_path = release_root / manifest.get("candidate_matrix", {}).get("manifest_path", "")
            checks.append(
                {
                    "name": "candidate_manifest_exists",
                    "passed": candidate_manifest_path.exists(),
                    "detail": str(candidate_manifest_path),
                }
            )
            phase2c_path = extracted_source / "campaigns/glp1r_aleniglipron/PHASE2C_SEALED_MANIFEST.json"
            checks.append(
                {
                    "name": "phase2c_manifest_exists",
                    "passed": phase2c_path.exists(),
                    "detail": str(phase2c_path),
                }
            )
        if args.run_smoke:
            validate_script = release_root / "validation" / "validate_release.sh"
            if validate_script.exists():
                env = os.environ.copy()
                env["PRISM_RELEASE_ROOT"] = str(release_root)
                smoke_result = run(
                    ["bash", str(validate_script)],
                    check=False,
                    env=env,
                    log_path=log_path,
                )
                checks.append(
                    {
                        "name": "operational_smoke",
                        "passed": smoke_result.returncode == 0,
                        "detail": smoke_result.stdout.strip() or smoke_result.stderr.strip(),
                    }
                )
            pytest_path = extracted_source / "tests" / "release_acceptance"
            if pytest_path.exists():
                env = os.environ.copy()
                env["PRISM_RELEASE_ROOT"] = str(release_root)
                env["PRISM_RESTORED_SOURCE_ROOT"] = str(extracted_source)
                acceptance_result = run(
                    [
                        "python3",
                        "-m",
                        "pytest",
                        str(pytest_path),
                        "-q",
                    ],
                    cwd=extracted_source,
                    env=env,
                    check=False,
                    log_path=log_path,
                )
                checks.append(
                    {
                        "name": "acceptance_suite",
                        "passed": acceptance_result.returncode == 0,
                        "detail": acceptance_result.stdout.strip() or acceptance_result.stderr.strip(),
                    }
                )

    failed = [check for check in checks if not check["passed"]]
    status = "PASS" if not failed else "PARTIAL" if any(check["passed"] for check in checks) else "FAIL"
    result = {
        "schema_version": "PRISM.sealed_release_cold_restore_validation.v1",
        "verified_at_utc": now_utc(),
        "release_root": str(release_root),
        "expected_release_id": args.expected_release_id,
        "status": status,
        "checks": checks,
        "failed_checks": [check["name"] for check in failed],
        "checksums": checksum_result,
        "git_bundle": bundle_result,
        "archive_extract": archive_extract,
        "restored_source_root": str(extracted_source),
        "smoke_validation": {
            "passed": not any(check["name"] == "operational_smoke" and not check["passed"] for check in checks)
        },
    }
    out_path = args.json_out or (release_root / "COLD_RESTORE_VALIDATION.json")
    write_json(out_path, result)
    print(f"COLD_RESTORE_STATUS={status}")
    print(f"OUTPUT={out_path}")
    if extract_root.exists():
        shutil.rmtree(extract_root, ignore_errors=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
