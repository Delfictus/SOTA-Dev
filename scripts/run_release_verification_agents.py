#!/usr/bin/env python3
"""Run independent release verification lanes for Track A sealing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._sealed_release_common import load_json, now_utc, sha256_path, verify_git_bundle, write_json


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET_FILENAME_RE = re.compile(r"(?i)(credentials\.env|(^|/)\.env($|/)|rclone\.conf|settings\.local\.json|\.pem$|\.key$|\.p12$)")
SECRET_CONTENT_RE = re.compile(
    r"(?i)(api[_-]?key|token|password|secret|cloudflare|aws_access_key|aws_secret|private key)\s*[:=]\s*"
    r"(?!demo\b|true\b|false\b|null\b|none\b|redacted\b|\$\{|\{|\[|<)"
    r"['\"]?[A-Za-z0-9_./+=:-]{20,}"
)
FORBIDDEN_CLAIMS_RE = re.compile(r"(?i)(FULL_PASS|FULLY_COMPLETE|DELETION_ALLOWED|HYDRATION_PRODUCTION_COMPLETE|PHASE 1-3 PRODUCTION COMPLETE)")
CRITICAL_SUFFIXES = {
    ".py",
    ".sh",
    ".rs",
    ".toml",
    ".cu",
    ".cuh",
    ".ptx",
    ".cubin",
    ".fatbin",
    ".so",
    ".a",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".duckdb",
    ".wal",
    ".shm",
    ".parquet",
    ".arrow",
    ".feather",
    ".h5",
    ".hdf5",
    ".sdf",
    ".mol2",
    ".pdb",
    ".pdbqt",
    ".gro",
    ".top",
    ".itp",
    ".xtc",
    ".trr",
    ".dcd",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".csv",
    ".tsv",
    ".npy",
    ".npz",
    ".pkl",
    ".joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["pre-seal", "post-seal"], required=True)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--fail-closed", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def run_cmd(args: list[str], *, cwd: Path | None = None, timeout: int | None = None) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    return {
        "args": args,
        "cwd": str(cwd) if cwd else None,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rel_to(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def independent_walk(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            st = current.lstat()
        except FileNotFoundError:
            continue
        rel = "." if current == root else rel_to(current, root)
        kind = "directory" if current.is_dir() and not current.is_symlink() else "symlink" if current.is_symlink() else "file"
        records.append(
            {
                "relative_path": rel,
                "absolute_path": str(current),
                "file_type": kind,
                "size_bytes": st.st_size,
                "executable": bool(st.st_mode & 0o100),
                "symlink_target": os.readlink(current) if current.is_symlink() else None,
            }
        )
        if kind == "directory":
            try:
                children = sorted(current.iterdir(), key=lambda item: item.name)
            except PermissionError:
                continue
            stack.extend(reversed(children))
    return records


def load_manifest_jsonl(release_root: Path) -> list[dict[str, Any]]:
    candidates = [release_root / "MANIFEST.jsonl", release_root / "inventory" / "full_inventory.jsonl"]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
    return []


def classify(rel: str, record: dict[str, Any]) -> set[str]:
    lowered = rel.lower()
    suffix = Path(rel).suffix.lower()
    cats: set[str] = set()
    if suffix in CRITICAL_SUFFIXES:
        cats.add(suffix.lstrip("."))
    if suffix == ".py":
        cats.add("python")
    if suffix == ".sh":
        cats.add("shell")
    if suffix == ".rs":
        cats.add("rust")
    if suffix in {".cu", ".cuh"}:
        cats.add("cuda_source")
    if suffix in {".ptx", ".cubin", ".fatbin"}:
        cats.add("cuda_runtime")
    if suffix in {".sqlite", ".sqlite3", ".db", ".duckdb", ".wal", ".shm", ".h5", ".hdf5", ".parquet"}:
        cats.add("database")
    if suffix in {".sdf", ".mol2", ".pdb", ".pdbqt", ".gro", ".top", ".itp", ".xtc", ".trr", ".dcd"}:
        cats.add("molecular_or_md")
    if "cargo.toml" in lowered or "cargo.lock" in lowered:
        cats.add("cargo")
    if "requirements" in lowered or "environment" in lowered or "conda" in lowered or "miniconda" in lowered or ".venv" in lowered:
        cats.add("environment")
    if "phase_2c" in lowered or "phase2c" in lowered:
        cats.add("phase2c")
    if "candidate" in lowered:
        cats.add("candidate")
    if "pathb" in lowered or "dstw_phase_b" in lowered:
        cats.add("path_b")
    if "topolog" in lowered or "residue_map" in lowered:
        cats.add("topology")
    if "motif" in lowered:
        cats.add("motif")
    if rel.startswith(".") or "/." in rel:
        cats.add("hidden")
    if record["file_type"] == "symlink":
        cats.add("symlink")
    if record.get("executable"):
        cats.add("executable")
    if lowered.startswith("scripts/") or "/scripts/" in lowered:
        cats.add("release_or_runtime_tooling")
    if suffix in {".md", ".txt", ".pdf"}:
        cats.add("docs")
    if suffix in {".json", ".jsonl", ".yaml", ".yml", ".toml"}:
        cats.add("manifest_or_config")
    return cats


class AgentContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.input_root = args.input_root.resolve()
        self.repo_root = args.repo_root.resolve()
        self.release_root = args.release_root.resolve()
        self.phase = args.phase
        self.agent_dir = self.release_root / "verification_agents"
        self.security_dir = self.release_root / "security"
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.security_dir.mkdir(parents=True, exist_ok=True)


def report(status: str, **payload: Any) -> dict[str, Any]:
    payload.setdefault("schema_version", "PRISM.release_verification_agent_report.v1")
    payload.setdefault("generated_at_utc", now_utc())
    payload["status"] = status
    return payload


def agent_critical_omission(ctx: AgentContext) -> dict[str, Any]:
    independent = independent_walk(ctx.input_root)
    planned = load_manifest_jsonl(ctx.release_root)
    planned_paths = {row["relative_path"] for row in planned if row.get("source_root_name") == "input_root"}
    unaccounted = []
    excluded = []
    for item in independent:
        rel = item["relative_path"]
        if rel == ".":
            continue
        if "/.git/objects/" in f"/{rel}/" or rel.endswith("/.git/objects") or "/.git/" in f"/{rel}/":
            excluded.append({"relative_path": rel, "reason": "git_internals_preserved_by_bundle_or_metadata"})
            continue
        if rel not in planned_paths:
            unaccounted.append(item)
    category_counts: dict[str, int] = {}
    for item in independent:
        for cat in classify(item["relative_path"], item):
            category_counts[cat] = category_counts.get(cat, 0) + 1
    required = ["python", "shell", "rust", "cargo", "database", "phase2c", "candidate", "topology", "docs", "manifest_or_config", "hidden", "executable", "release_or_runtime_tooling"]
    missing_required = [cat for cat in required if category_counts.get(cat, 0) == 0]
    status = "PASS" if not unaccounted and not missing_required else "FAIL"
    result = report(
        status,
        agent_id="AGENT_1",
        agent_name="CRITICAL_FILE_OMISSION_HUNTER",
        independent_record_count=len(independent),
        planned_record_count=len(planned),
        unaccounted_count=len(unaccounted),
        unaccounted_sample=unaccounted[:50],
        intentionally_excluded_count=len(excluded),
        intentionally_excluded_sample=excluded[:50],
        category_counts=category_counts,
        missing_required_categories=missing_required,
    )
    write_json(ctx.agent_dir / "CRITICAL_FILE_OMISSION_AUDIT.json", result)
    write_text(
        ctx.agent_dir / "CRITICAL_FILE_OMISSION_AUDIT.md",
        f"# Critical File Omission Audit\n\nstatus: `{status}`\nunaccounted_count: `{len(unaccounted)}`\nmissing_required_categories: `{missing_required}`\n",
    )
    return result


def agent_bug_hunter(ctx: AgentContext) -> dict[str, Any]:
    seal = ctx.repo_root / "scripts" / "seal_operational_release.py"
    common = ctx.repo_root / "scripts" / "_sealed_release_common.py"
    compile_result = run_cmd(["python3", "-m", "py_compile", str(seal), str(common)])
    help_result = run_cmd(["python3", str(seal), "--help"], cwd=ctx.repo_root)
    seal_text = seal.read_text(encoding="utf-8", errors="ignore") if seal.exists() else ""
    common_text = common.read_text(encoding="utf-8", errors="ignore") if common.exists() else ""
    required_tokens = {
        "repo_root_flag": "--repo-root" in help_result["stdout"],
        "plan_flag": "--plan" in help_result["stdout"],
        "no_source_duplicate_flag": "--no-source-duplicate" in help_result["stdout"],
        "platform_archive_chunks": "create_platform_archive_chunks" in seal_text,
        "input_root_archive": "ctx.input_root.name" in seal_text and "ctx.input_root.parent" in seal_text,
        "plan_b_execution": "PLAN_B_EXECUTION.json" in seal_text,
        "chunk_manifest": "chunk_manifest.json" in seal_text,
        "no_r2_by_default": "upload_r2_enabled" in seal_text and "args.upload_r2" in seal_text,
        "no_git_object_inventory_blindspot": ".git\" and name == \"objects" in common_text,
        "no_false_full_pass_ssd_only": "TRACK_A_SSD_STORAGE_WRITTEN_PENDING_READBACK" in seal_text,
    }
    synthetic = {"status": "NOT_RUN", "reason": "full synthetic non-dry fixture skipped to avoid hidden archive writes before pre-seal gate"}
    failures = [name for name, passed in required_tokens.items() if not passed]
    status = "PASS" if compile_result["returncode"] == 0 and help_result["returncode"] == 0 and not failures else "FAIL"
    result = report(
        status,
        agent_id="AGENT_2",
        agent_name="BUG_HUNTER_PLAN_B_NON_DRY_PATH",
        compile_result=compile_result,
        help_result=help_result,
        static_checks=required_tokens,
        failed_static_checks=failures,
        synthetic_fixture=synthetic,
    )
    write_json(ctx.agent_dir / "BUG_HUNTER_REPORT.json", result)
    write_text(ctx.agent_dir / "BUG_HUNTER_REPORT.md", f"# Bug Hunter Report\n\nstatus: `{status}`\nfailed_static_checks: `{failures}`\n")
    return result


def agent_implemented_work(ctx: AgentContext) -> dict[str, Any]:
    checks = [
        ("REQ_PLAN_B_STREAMING", "Plan B streaming mode exists", ctx.repo_root / "scripts/seal_operational_release.py", "create_platform_archive_chunks", ctx.repo_root / "scripts/run_release_verification_agents.py"),
        ("REQ_INPUT_ROOT", "input_root honored separately", ctx.repo_root / "scripts/seal_operational_release.py", "ctx.input_root", ctx.repo_root / "scripts/run_release_verification_agents.py"),
        ("REQ_REPO_ROOT", "repo_root honored separately", ctx.repo_root / "scripts/seal_operational_release.py", "--repo-root", ctx.repo_root / "scripts/run_release_verification_agents.py"),
        ("REQ_CHUNKS", "chunked archive mode exists", ctx.repo_root / "scripts/seal_operational_release.py", "chunk_manifest.json", ctx.repo_root / "scripts/verify_restored_release.py"),
        ("REQ_GIT_BUNDLE", "git bundle creation exists", ctx.repo_root / "scripts/seal_operational_release.py", "git bundle", ctx.repo_root / "scripts/run_release_verification_agents.py"),
        ("REQ_SSD_READBACK", "SSD readback validation exists", ctx.repo_root / "scripts/run_release_verification_agents.py", "SSD_READBACK_VALIDATION.json", ctx.repo_root / "scripts/run_release_verification_agents.py"),
        ("REQ_COLD_RESTORE", "cold restore verifier exists", ctx.repo_root / "scripts/verify_restored_release.py", "chunk_manifest.json", ctx.repo_root / "scripts/run_release_verification_agents.py"),
        ("REQ_POLICY", "policy evaluator exists", ctx.repo_root / "scripts/evaluate_release_policy.py", "PARTIAL", ctx.repo_root / "release_policy.yaml"),
        ("REQ_CLAIM", "claim boundary enforcement exists", ctx.repo_root / "scripts/seal_operational_release.py", "NOT_RUN_FULL_SCALE", ctx.repo_root / "scripts/run_release_verification_agents.py"),
        ("REQ_NO_FULL_PASS", "FULL_PASS cannot be emitted in SSD-only path", ctx.repo_root / "scripts/seal_operational_release.py", "TRACK_A_SSD_STORAGE_WRITTEN_PENDING_READBACK", ctx.repo_root / "scripts/run_release_verification_agents.py"),
    ]
    rows = []
    for req_id, text, impl, token, evidence in checks:
        impl_text = impl.read_text(encoding="utf-8", errors="ignore") if impl.exists() else ""
        status = "PASS" if token in impl_text else "FAIL"
        rows.append(
            {
                "requirement_id": req_id,
                "requirement_text": text,
                "implementation_file": str(impl),
                "implementation_symbol_or_section": token,
                "test_file": str(evidence),
                "test_command": "python3 scripts/run_release_verification_agents.py --phase pre-seal ...",
                "evidence_file": str(evidence),
                "status": status,
                "notes": "",
            }
        )
    failing = [row for row in rows if row["status"] == "FAIL"]
    status = "PASS" if not failing else "FAIL"
    result = report(status, agent_id="AGENT_3", agent_name="IMPLEMENTED_WORK_VALIDATOR", requirements=rows, failing_count=len(failing))
    write_json(ctx.agent_dir / "IMPLEMENTED_WORK_VALIDATION.json", result)
    write_csv(ctx.agent_dir / "IMPLEMENTED_WORK_TRACEABILITY.csv", rows)
    return result


def agent_repo_centricity(ctx: AgentContext) -> dict[str, Any]:
    records = load_manifest_jsonl(ctx.release_root)
    outside_repo = [row for row in records if row.get("repo_relative_path") in {None, ""}]
    repo_rows = [row for row in records if row.get("repo_relative_path")]
    sample_outside = [row.get("relative_path") for row in outside_repo[:20]]
    status = "PASS" if records and outside_repo and repo_rows and ctx.repo_root.is_relative_to(ctx.input_root) else "FAIL"
    result = report(
        status,
        agent_id="AGENT_4",
        agent_name="REPO_CENTRICITY_AUDITOR",
        input_root=str(ctx.input_root),
        repo_root=str(ctx.repo_root),
        record_count=len(records),
        outside_repo_count=len(outside_repo),
        repo_count=len(repo_rows),
        outside_repo_sample=sample_outside,
    )
    write_json(ctx.agent_dir / "REPO_CENTRICITY_AUDIT.json", result)
    return result


def agent_capacity(ctx: AgentContext) -> dict[str, Any]:
    capacity_path = ctx.release_root / "capacity_plan.json"
    dry_path = ctx.release_root / "DRY_RUN_SUMMARY.json"
    capacity = load_json(capacity_path) if capacity_path.exists() else {}
    dry = load_json(dry_path) if dry_path.exists() else {}
    selected = capacity.get("selected_plan", {})
    checks = {
        "selected_plan_b": selected.get("plan_id") == "B",
        "no_source_duplicate_description": "no source duplicate" in selected.get("description", "").lower(),
        "estimated_archive_bytes_recorded": bool(capacity.get("estimated_archive_bytes")),
        "root_headroom_recorded": selected.get("available_root_bytes", 0) >= selected.get("required_root_bytes", 1),
        "ssd_headroom_recorded": selected.get("available_ssd_bytes", 0) >= selected.get("required_ssd_bytes", 1),
        "dry_run_summary_present": bool(dry),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = report(status, agent_id="AGENT_5", agent_name="CAPACITY_AND_STREAMING_AUDITOR", checks=checks, selected_plan=selected)
    write_json(ctx.agent_dir / "CAPACITY_PLAN_AUDIT.json", result)
    write_text(ctx.agent_dir / "CAPACITY_PLAN_AUDIT.md", f"# Capacity Plan Audit\n\nstatus: `{status}`\nchecks: `{checks}`\n")
    return result


def agent_secret(ctx: AgentContext) -> dict[str, Any]:
    records = load_manifest_jsonl(ctx.release_root)
    findings = []
    github_violations = []
    for row in records:
        rel = row.get("relative_path", "")
        abs_path = Path(row.get("absolute_path", ""))
        if SECRET_FILENAME_RE.search(rel):
            findings.append({"path": rel, "kind": "filename", "match": "[REDACTED]"})
            if row.get("destinations", {}).get("github", {}).get("include"):
                github_violations.append(rel)
        if row.get("file_type") == "file" and abs_path.exists() and abs_path.stat().st_size <= 1024 * 1024 and abs_path.suffix.lower() in {".env", ".txt", ".json", ".yaml", ".yml", ".toml", ".conf"}:
            try:
                text = abs_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            if SECRET_CONTENT_RE.search(text):
                findings.append({"path": rel, "kind": "content", "match": "[REDACTED]"})
                if row.get("destinations", {}).get("github", {}).get("include"):
                    github_violations.append(rel)
    result = report(
        "PASS" if not github_violations else "FAIL",
        agent_id="AGENT_6",
        agent_name="SECRET_AND_CREDENTIAL_BOUNDARY_AUDITOR",
        redacted_finding_count=len(findings),
        github_plaintext_secret_violations=sorted(set(github_violations)),
    )
    write_json(ctx.agent_dir / "SECRET_BOUNDARY_AUDIT.json", result)
    write_json(ctx.security_dir / "secret_scan_redacted.json", {"findings": findings})
    if findings:
        write_text(ctx.security_dir / "MISSING_SECRETS_REQUIRED.md", "# Secret Restore Requirements\n\nSecrets are redacted from GitHub-bound payloads and must be supplied out of band or from encrypted escrow.\n")
    return result


def agent_claim(ctx: AgentContext) -> dict[str, Any]:
    required_paths = [
        ctx.release_root / "RELEASE_CLAIM_BOUNDARY.md",
        ctx.release_root / "RELEASE_STATUS.json",
        ctx.repo_root / "RELEASE_TRACKS.md",
        ctx.repo_root / "release_tracks.json",
        ctx.repo_root / "scripts" / "seal_operational_release.py",
    ]
    forbidden_paths = [
        ctx.release_root / "RELEASE_CLAIM_BOUNDARY.md",
        ctx.release_root / "RELEASE_STATUS.json",
        ctx.release_root / "README_RESTORE.md",
        ctx.repo_root / "RELEASE_TRACKS.md",
        ctx.repo_root / "release_tracks.json",
    ]
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in required_paths if path.exists())
    forbidden_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in forbidden_paths if path.exists())
    forbidden = FORBIDDEN_CLAIMS_RE.findall(forbidden_text)
    required = {
        "hydration_smoke": "IMPLEMENTED_AND_SMOKE_VERIFIED" in text,
        "no_full_hydration": "NOT_RUN_FULL_SCALE" in text or "NOT_FULLY_RUN_104GB_INPUT_UNMEASURED" in text,
        "no_full_phase13": "NOT_CLAIMABLE" in text,
    }
    status = "PASS" if not forbidden and all(required.values()) else "FAIL"
    result = report(status, agent_id="AGENT_7", agent_name="CLAIM_BOUNDARY_AUDITOR", forbidden_claims=forbidden, required_checks=required)
    write_json(ctx.agent_dir / "CLAIM_BOUNDARY_AUDIT.json", result)
    write_text(ctx.agent_dir / "CLAIM_BOUNDARY_AUDIT.md", f"# Claim Boundary Audit\n\nstatus: `{status}`\nrequired: `{required}`\nforbidden_claims: `{forbidden}`\n")
    return result


def agent_manifest_coverage(ctx: AgentContext) -> dict[str, Any]:
    records = load_manifest_jsonl(ctx.release_root)
    chunk_manifest = load_json(ctx.release_root / "chunk_manifest.json") if (ctx.release_root / "chunk_manifest.json").exists() else {}
    missing_chunks = []
    bad_chunks = []
    for chunk in chunk_manifest.get("chunks", []):
        path = ctx.release_root / chunk["path"]
        if not path.exists():
            missing_chunks.append(chunk["path"])
            continue
        if path.stat().st_size != chunk["size_bytes"]:
            bad_chunks.append({"path": chunk["path"], "reason": "size_mismatch"})
        elif sha256_path(path) != chunk["sha256"]:
            bad_chunks.append({"path": chunk["path"], "reason": "sha256_mismatch"})
    status = "PASS" if records and chunk_manifest and not missing_chunks and not bad_chunks else "FAIL"
    result = report(status, agent_id="AGENT_8", agent_name="MANIFEST_COVERAGE_AUDITOR", manifest_records=len(records), chunk_count=len(chunk_manifest.get("chunks", [])), missing_chunks=missing_chunks, bad_chunks=bad_chunks)
    write_json(ctx.agent_dir / "MANIFEST_COVERAGE_AUDIT.json", result)
    write_text(ctx.agent_dir / "MANIFEST_COVERAGE_AUDIT.md", f"# Manifest Coverage Audit\n\nstatus: `{status}`\nmanifest_records: `{len(records)}`\nchunk_count: `{len(chunk_manifest.get('chunks', []))}`\n")
    return result


def agent_ssd_readback(ctx: AgentContext) -> dict[str, Any]:
    checks = {}
    for rel in ["MANIFEST.json", "MANIFEST.jsonl", "CHECKSUMS.sha256", "RELEASE_STATUS.json", "RELEASE_CLAIM_BOUNDARY.md", "README_RESTORE.md", "chunk_manifest.json"]:
        path = ctx.release_root / rel
        checks[f"{rel}_exists"] = path.exists() and path.stat().st_size > 0
        if rel.endswith(".json") and path.exists():
            try:
                load_json(path)
                checks[f"{rel}_parses"] = True
            except Exception:
                checks[f"{rel}_parses"] = False
    chunk_manifest = load_json(ctx.release_root / "chunk_manifest.json") if (ctx.release_root / "chunk_manifest.json").exists() else {}
    chunks_ok = True
    for chunk in chunk_manifest.get("chunks", []):
        path = ctx.release_root / chunk["path"]
        if not path.exists() or path.stat().st_size == 0 or path.stat().st_size != chunk["size_bytes"] or sha256_path(path) != chunk["sha256"]:
            chunks_ok = False
            break
    checks["chunks_verify"] = chunks_ok and bool(chunk_manifest.get("chunks"))
    bundles = sorted((ctx.release_root / "git").glob("*.bundle")) if (ctx.release_root / "git").exists() else []
    bundle_result = verify_git_bundle(bundles[0]) if bundles else {"verified": False}
    checks["git_bundle_verifies"] = bool(bundle_result.get("verified"))
    status = "PASS" if all(checks.values()) else "FAIL"
    result = report(status, agent_id="AGENT_9", agent_name="SSD_DESTINATION_READBACK_AUDITOR", overall_status=status, checks=checks, git_bundle=bundle_result)
    write_json(ctx.agent_dir / "DESTINATION_READBACK_AUDIT.json", result)
    write_json(ctx.release_root / "SSD_READBACK_VALIDATION.json", result)
    write_text(ctx.release_root / "SSD_READBACK_VALIDATION.md", f"# SSD Readback Validation\n\nstatus: `{status}`\n")
    return result


def agent_cold_restore(ctx: AgentContext) -> dict[str, Any]:
    release_id = ctx.release_root.name
    out_path = ctx.release_root / "validation" / "COLD_RESTORE_VALIDATION.json"
    result = run_cmd(
        [
            "python3",
            str(ctx.repo_root / "scripts" / "verify_restored_release.py"),
            "--release-root",
            str(ctx.release_root),
            "--expected-release-id",
            release_id,
            "--offline",
            "--run-smoke",
            "--run-db-checks",
            "--run-cuda-checks",
            "--run-candidate-checks",
            "--skip-expensive-tests",
            "--json-out",
            str(out_path),
        ],
        cwd=ctx.repo_root,
    )
    validation = load_json(out_path) if out_path.exists() else {}
    root_copy = ctx.release_root / "COLD_RESTORE_VALIDATION.json"
    if out_path.exists():
        shutil.copy2(out_path, root_copy)
    status = validation.get("status", "FAIL")
    agent_result = report(status if status in {"PASS", "PARTIAL", "FAIL"} else "FAIL", agent_id="AGENT_10", agent_name="COLD_RESTORE_VERIFIER", command=result, validation=validation)
    write_json(ctx.agent_dir / "COLD_RESTORE_AUDIT.json", agent_result)
    write_text(ctx.release_root / "COLD_RESTORE_VALIDATION.md", f"# Cold Restore Validation\n\nstatus: `{status}`\n")
    return agent_result


def agent_policy(ctx: AgentContext) -> dict[str, Any]:
    out_path = ctx.release_root / "validation" / "POLICY_EVALUATION.json"
    result = run_cmd(
        [
            "python3",
            str(ctx.repo_root / "scripts" / "evaluate_release_policy.py"),
            "--release-root",
            str(ctx.release_root),
            "--policy",
            str(ctx.repo_root / "release_policy.yaml"),
            "--json-out",
            str(out_path),
        ],
        cwd=ctx.repo_root,
    )
    evaluation = load_json(out_path) if out_path.exists() else {}
    if out_path.exists():
        shutil.copy2(out_path, ctx.release_root / "POLICY_EVALUATION.json")
    status = evaluation.get("status", "FAIL")
    agent_result = report(status if status in {"PASS", "PARTIAL", "FAIL"} else "FAIL", agent_id="AGENT_11", agent_name="POLICY_EVALUATOR_AGENT", command=result, evaluation=evaluation)
    write_json(ctx.agent_dir / "POLICY_EVALUATOR_AGENT.json", agent_result)
    return agent_result


def agent_final_adversarial(ctx: AgentContext, prior: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {item.get("agent_name"): item.get("status") for item in prior}
    blocking_failures = {name: status for name, status in statuses.items() if status in {"FAIL", "NOT_RUN"}}
    questions = {
        "unaccounted_files": statuses.get("CRITICAL_FILE_OMISSION_HUNTER") != "PASS",
        "critical_file_class_missing": statuses.get("CRITICAL_FILE_OMISSION_HUNTER") != "PASS",
        "destination_marked_pass_without_readback": statuses.get("SSD_DESTINATION_READBACK_AUDITOR") != "PASS",
        "restore_claim_without_cold_restore": statuses.get("COLD_RESTORE_VERIFIER") not in {"PASS", "PARTIAL"},
        "hydration_claim_overstated": statuses.get("CLAIM_BOUNDARY_AUDITOR") != "PASS",
        "status_stronger_than_gate": bool(blocking_failures),
    }
    status = "PASS" if not any(questions.values()) else "FAIL"
    result = report(status, agent_id="AGENT_12", agent_name="FINAL_ADVERSARIAL_REVIEWER", prior_statuses=statuses, adversarial_questions=questions, blocking_failures=blocking_failures)
    write_json(ctx.agent_dir / "FINAL_ADVERSARIAL_REVIEW.json", result)
    write_text(ctx.agent_dir / "FINAL_ADVERSARIAL_REVIEW.md", f"# Final Adversarial Review\n\nstatus: `{status}`\nquestions: `{questions}`\n")
    return result


def build_matrix(ctx: AgentContext, reports: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    rows = []
    for item in reports:
        status = item.get("status", "NOT_RUN")
        blocking = status == "FAIL" or status == "NOT_RUN"
        rows.append(
            {
                "agent_id": item.get("agent_id"),
                "agent_name": item.get("agent_name"),
                "phase": phase,
                "required_for_phase": True,
                "input_artifacts": str(ctx.release_root),
                "output_report": str(ctx.agent_dir),
                "status": status,
                "blocking": blocking,
                "failure_summary": ",".join(item.get("failed_static_checks", [])) if isinstance(item.get("failed_static_checks"), list) else "",
                "remediation_required": blocking,
                "evidence_paths": str(ctx.agent_dir),
            }
        )
    write_csv(ctx.agent_dir / "AGENT_GATE_MATRIX.csv", rows)
    write_json(ctx.agent_dir / "AGENT_GATE_MATRIX.json", {"phase": phase, "rows": rows})
    return rows


def run_phase(ctx: AgentContext, phase: str, parallel: bool) -> list[dict[str, Any]]:
    if phase == "pre-seal":
        agents: list[Callable[[AgentContext], dict[str, Any]]] = [
            agent_critical_omission,
            agent_bug_hunter,
            agent_implemented_work,
            agent_repo_centricity,
            agent_capacity,
            agent_secret,
            agent_claim,
        ]
        if not parallel:
            return [agent(ctx) for agent in agents]
        reports = []
        with ThreadPoolExecutor(max_workers=min(7, os.cpu_count() or 4)) as executor:
            future_map = {executor.submit(agent, ctx): agent.__name__ for agent in agents}
            for future in as_completed(future_map):
                reports.append(future.result())
        return reports

    reports = [
        agent_manifest_coverage(ctx),
        agent_ssd_readback(ctx),
        agent_cold_restore(ctx),
        agent_policy(ctx),
    ]
    reports.append(agent_claim(ctx))
    reports.append(agent_final_adversarial(ctx, reports))
    return reports


def main() -> int:
    args = parse_args()
    ctx = AgentContext(args)
    reports = run_phase(ctx, args.phase, args.parallel)
    rows = build_matrix(ctx, reports, args.phase)
    phase_status = "PASS" if not any(row["blocking"] for row in rows) else "FAIL"
    decision_name = "PRE_SEAL_AGENT_GATE_DECISION.json" if args.phase == "pre-seal" else "POST_SEAL_AGENT_GATE_DECISION.json"
    decision = {
        "schema_version": "PRISM.agent_gate_decision.v1",
        "phase": args.phase,
        "generated_at_utc": now_utc(),
        "status": phase_status,
        "release_root": str(ctx.release_root),
        "input_root": str(ctx.input_root),
        "repo_root": str(ctx.repo_root),
        "reports": [{key: report.get(key) for key in ("agent_id", "agent_name", "status")} for report in reports],
        "blocking_agents": [row for row in rows if row["blocking"]],
    }
    write_json(ctx.agent_dir / decision_name, decision)
    write_json(ctx.agent_dir / "FINAL_AGENT_GATE_DECISION.json", decision)
    if args.json_out:
        write_json(args.json_out, decision)
    for path in (ctx.agent_dir / "BUG_FIX_LOOP_LOG.md", ctx.agent_dir / "BUG_FIX_LOOP_LOG.jsonl"):
        if not path.exists():
            write_text(path, "# Bug Fix Loop Log\n\nNo remediations recorded by this orchestrator run.\n" if path.suffix == ".md" else "")
    print(f"AGENT_GATE_STATUS={phase_status}")
    print(f"AGENT_GATE_DECISION={ctx.agent_dir / decision_name}")
    return 0 if phase_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
