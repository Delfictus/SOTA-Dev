#!/usr/bin/env python3
"""Build a preservation-grade sealed operational release snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._sealed_release_common import (
    append_log,
    blake3_path,
    chunk_size_bytes,
    checkpoint_sqlite,
    compute_release_root_hash,
    default_destination_plan,
    duckdb_checkpoint,
    estimate_archive_bytes,
    file_count_and_bytes,
    find_active_compute_processes,
    free_bytes,
    gather_tool_versions,
    git_status_map,
    human_bytes,
    inventory_records,
    is_database_path,
    json_dumps,
    load_json,
    now_utc,
    relative_records_tree,
    resolve_prism_input_layout,
    rsync_copy,
    run,
    scan_secret_candidates,
    sha256_path,
    sqlite_backup,
    verify_git_bundle,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_REPO_ROOT
CAMPAIGN_ROOT = REPO_ROOT / "campaigns" / "glp1r_aleniglipron"
DEFAULT_SSD_ROOT = Path(os.environ.get("EXTERNAL_SSD_RELEASE_ROOT", "/media/diddy/PRISM-LBS"))
DEFAULT_R2_PREFIX = "r2:prism-archive/sealed-operational"
GITHUB_SAFE_ROOT = SCRIPT_REPO_ROOT / "release_artifacts" / "sealed_operational"
REFERENCE_DATA_RELATIVE_PATHS = [
    "campaigns/glp1r_aleniglipron/track_a_generative/fullscale_shards/shard_0000.parquet",
    "campaigns/glp1r_aleniglipron/track_a_generative/fullscale_shards/shard_0001.parquet",
    "campaigns/glp1r_aleniglipron/track_a_generative/enamine_115k_synthons_3d.parquet",
    "campaigns/glp1r_aleniglipron/track_a_generative/enamine_chunked15k_synthons_3d.parquet",
    "campaigns/glp1r_aleniglipron/track_a_generative/enamine_130k_synthons_3d.parquet",
    "campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_full_scale.parquet",
    "campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_scaffold_consensus_action_corpus.parquet",
    "campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_population_consensus_action_corpus.parquet",
    "campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top_50_tripartite_profiles.parquet",
]
PTX_PATHS = [
    REPO_ROOT / "crates/prism-gpu/src/kernels",
    REPO_ROOT / "target/ptx",
    REPO_ROOT / "crates/prism-nhs/src/cuda",
    REPO_ROOT / "vendor/working_ptx_2026-05-29",
]
GITHUB_COMMIT_PATHS = [
    "scripts/_sealed_release_common.py",
    "scripts/seal_operational_release.py",
    "scripts/verify_restored_release.py",
    "scripts/evaluate_release_policy.py",
    "release_policy.yaml",
    "tests/release_acceptance/test_release_acceptance.py",
    "Makefile",
]


@dataclass
class ReleaseContext:
    release_id: str
    release_root: Path
    input_root: Path
    layout_kind: str
    repo_root: Path
    campaign_root: Path
    env_root: Path | None
    candidate_smoke_root: Path | None
    credentials_path: Path | None
    rclone_config_path: Path | None
    repo_mirror_root: Path
    log_path: Path
    continue_on_error: bool
    skip_expensive_tests: bool
    github_enabled: bool
    ssd_root: Path
    r2_prefix: str | None
    upload_r2_enabled: bool
    dry_run: bool
    plan: str | None
    streaming: bool
    chunk_size_gib: float
    chunk_size_bytes: int
    no_double_materialization: bool
    validate_extract_root: Path
    allow_ssd_single_copy_validation: bool
    capacity_report_out: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id")
    parser.add_argument("--input-root", type=Path, default=SCRIPT_REPO_ROOT)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--github", action="store_true")
    parser.add_argument("--ssd-root", type=Path, default=DEFAULT_SSD_ROOT)
    parser.add_argument("--r2", default=DEFAULT_R2_PREFIX)
    parser.add_argument("--upload-r2", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-expensive-tests", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan", choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--chunk-size-gib", type=float, default=8.0)
    parser.add_argument("--no-double-materialization", action="store_true")
    parser.add_argument("--no-source-duplicate", action="store_true")
    parser.add_argument("--validate-extract-root", type=Path, default=Path("/"))
    parser.add_argument("--allow-ssd-single-copy-validation", action="store_true")
    parser.add_argument("--capacity-report-out", type=Path)
    return parser.parse_args()


def compute_release_id(repo_root: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    head = run(["git", "rev-parse", "--short=12", "HEAD"], cwd=repo_root).stdout.strip() or "nogit"
    return f"sealed-operational-{stamp}-{head}"


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_deficit(ctx: ReleaseContext, deficits: list[dict[str, Any]], step: str, detail: str) -> None:
    deficits.append({"step": step, "detail": detail, "time_utc": now_utc()})
    append_log(ctx.log_path, f"DEFICIT step={step} detail={detail}")


def maybe_fail(ctx: ReleaseContext, deficits: list[dict[str, Any]], step: str, detail: str) -> None:
    record_deficit(ctx, deficits, step, detail)
    if not ctx.continue_on_error:
        raise RuntimeError(detail)


def locate_reference_data(repo_root: Path) -> list[Path]:
    resolved = [(repo_root / rel).resolve() for rel in REFERENCE_DATA_RELATIVE_PATHS]
    missing = [path for path in resolved if not path.exists()]
    if missing:
        missing_lines = "\n".join(str(path) for path in missing)
        raise RuntimeError(
            "Mandatory static reference dataset paths are missing.\n"
            f"{missing_lines}\n"
            "Provide exact absolute paths before continuing."
        )
    return resolved


def classify_repo_mirror_files(release_root: Path, release_id: str) -> list[Path]:
    return [
        release_root / "MANIFEST.json",
        release_root / "RELEASE_STATUS.json",
        release_root / "RELEASE_DEFICITS.md",
        release_root / "RELEASE_CLAIM_BOUNDARY.md",
        release_root / "README_RESTORE.md",
        release_root / "RESTORE_ORDER.md",
        release_root / "RESTORE_SOURCE_MATRIX.json",
        release_root / "DESTINATION_READBACK_VALIDATION.json",
        release_root / "POLICY_EVALUATION.json",
        release_root / "COLD_RESTORE_VALIDATION.json",
        release_root / "DESTINATION_MATRIX.csv",
        release_root / "RELEASE_EVIDENCE_INDEX.md",
        release_root / "provenance/SLSA_PROVENANCE.json",
        release_root / "provenance/IN_TOTO_ATTESTATION.json",
        release_root / "provenance/BUILD_COMMANDS.log",
        release_root / "provenance/INPUT_MATERIALS.json",
        release_root / "provenance/OUTPUT_ARTIFACTS.json",
        release_root / "provenance/TOOL_VERSIONS.json",
    ]


def write_destination_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = ["destination", "status", "detail", "location"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_checksums(root: Path) -> Path:
    output = root / "CHECKSUMS.sha256"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != output)
    with output.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_path(path)}  {path.relative_to(root)}\n")
    return output


def create_repo_archive(ctx: ReleaseContext, reference_paths: list[Path], deficits: list[dict[str, Any]]) -> dict[str, Any]:
    archive_dir = mkdir(ctx.release_root / "archives")
    archive_path = archive_dir / "repo_working_tree.tar.zst"
    reference_relpaths = []
    for path in reference_paths:
        try:
            reference_relpaths.append(str(path.relative_to(ctx.repo_root)))
        except ValueError:
            continue
    exclude_file = archive_dir / "repo_working_tree.exclude"
    exclude_entries = [".git", "release_staging"] + reference_relpaths
    exclude_file.write_text("\n".join(exclude_entries) + "\n", encoding="utf-8")
    cmd = [
        "bash",
        "-lc",
        (
            f"cd {shlex.quote(str(ctx.repo_root.parent))} && "
            f"sudo -n tar --xattrs --acls --numeric-owner --preserve-permissions --warning=no-file-changed "
            f"--exclude-from={shlex.quote(str(exclude_file))} -cpf - {shlex.quote(ctx.repo_root.name)} "
            f"| zstd -T0 --long -19 -o {shlex.quote(str(archive_path))}"
        ),
    ]
    result = run(cmd, check=False, log_path=ctx.log_path)
    if result.returncode != 0:
        maybe_fail(ctx, deficits, "repo_archive", f"repo archive creation failed: {result.stderr}")
    return {
        "path": str(archive_path.relative_to(ctx.release_root)),
        "exists": archive_path.exists(),
        "bytes": archive_path.stat().st_size if archive_path.exists() else None,
        "sha256": sha256_path(archive_path) if archive_path.exists() else None,
        "blake3": blake3_path(archive_path) if archive_path.exists() else None,
        "top_level_dir": ctx.repo_root.name,
        "excluded_reference_paths": reference_relpaths,
    }


def write_non_dry_preflight(ctx: ReleaseContext, capacity_report: dict[str, Any]) -> dict[str, Any]:
    out_dir = mkdir(ctx.release_root / "preflight" / "non_dry_preflight")
    commands: dict[str, list[str]] = {
        "git_status_short.txt": ["git", "status", "--short"],
        "git_rev_parse_head.txt": ["git", "rev-parse", "HEAD"],
        "git_branch_show_current.txt": ["git", "branch", "--show-current"],
        "git_remote_v.txt": ["git", "remote", "-v"],
        "git_diff_stat.txt": ["git", "diff", "--stat"],
        "git_diff_check.txt": ["git", "diff", "--check"],
    }
    command_results = {}
    for filename, command in commands.items():
        result = run(command, cwd=ctx.repo_root, check=False, log_path=ctx.log_path)
        write_text(out_dir / filename, result.stdout + result.stderr)
        command_results[filename] = {"returncode": result.returncode}
    compile_targets = [
        ctx.repo_root / "scripts" / "seal_operational_release.py",
        ctx.repo_root / "scripts" / "_sealed_release_common.py",
        ctx.repo_root / "scripts" / "verify_restored_release.py",
        ctx.repo_root / "scripts" / "evaluate_release_policy.py",
    ]
    compile_result = run(["python3", "-m", "py_compile", *[str(path) for path in compile_targets]], check=False, log_path=ctx.log_path)
    write_text(out_dir / "py_compile.txt", compile_result.stdout + compile_result.stderr)
    head = (out_dir / "git_rev_parse_head.txt").read_text(encoding="utf-8").strip()
    summary = {
        "schema_version": "PRISM.non_dry_preflight.v1",
        "generated_at_utc": now_utc(),
        "release_id": ctx.release_id,
        "input_root": str(ctx.input_root),
        "repo_root": str(ctx.repo_root),
        "head": head,
        "canonical_head_expected": "b40d650b90b00825f29007556d4f92c415ebc935",
        "canonical_head_matches": head == "b40d650b90b00825f29007556d4f92c415ebc935",
        "release_tooling_compiles": compile_result.returncode == 0,
        "selected_plan": capacity_report.get("selected_plan", {}).get("plan_id"),
        "selected_plan_is_b": capacity_report.get("selected_plan", {}).get("plan_id") == "B",
        "no_deletion_will_occur": True,
        "no_source_duplicate_will_be_created": ctx.no_double_materialization,
        "full_canonical_root_is_input": ctx.input_root != ctx.repo_root,
        "track_b_volatility_blocks_track_a": False,
        "command_results": command_results,
    }
    write_json(ctx.release_root / "NON_DRY_PREFLIGHT.json", summary)
    return summary


def create_platform_archive_chunks(
    ctx: ReleaseContext,
    capacity_report: dict[str, Any],
    deficits: list[dict[str, Any]],
) -> dict[str, Any]:
    chunks_dir = mkdir(ctx.release_root / "archive_chunks")
    streaming_log_path = ctx.release_root / "PLAN_B_STREAMING_LOG.jsonl"
    chunk_prefix = chunks_dir / "platform_input_root.tar.zst.part-"
    exclude_file = chunks_dir / "platform_archive.exclude"
    exclude_entries = [
        f"{ctx.input_root.name}/absolute_root/home/diddy/Desktop/Prism4D-bio/.git/objects",
        f"{ctx.input_root.name}/absolute_root/home/diddy/Desktop/Prism4D-bio/.git/objects/*",
    ]
    if ctx.release_root.is_relative_to(ctx.input_root):
        exclude_entries.append(str(ctx.release_root.relative_to(ctx.input_root)))
    exclude_file.write_text("\n".join(exclude_entries) + "\n", encoding="utf-8")
    command = [
        "bash",
        "-lc",
        (
            f"set -euo pipefail; "
            f"cd {shlex.quote(str(ctx.input_root.parent))}; "
            f"sudo -n tar --xattrs --acls --numeric-owner --preserve-permissions --warning=no-file-changed "
            f"--exclude-from={shlex.quote(str(exclude_file))} "
            f"-cpf - {shlex.quote(ctx.input_root.name)} "
            f"| zstd -T0 --long -19 "
            f"| split -b {ctx.chunk_size_bytes} -d -a 5 - {shlex.quote(str(chunk_prefix))}"
        ),
    ]
    write_text(ctx.release_root / "SEAL_COMMAND_USED.txt", " ".join(shlex.quote(arg) for arg in sys.argv) + "\n")
    write_jsonl(
        streaming_log_path,
        [
            {
                "event": "plan_b_stream_start",
                "time_utc": now_utc(),
                "input_root": str(ctx.input_root),
                "release_root": str(ctx.release_root),
                "chunk_size_bytes": ctx.chunk_size_bytes,
                "tar_exclude_file": str(exclude_file),
            }
        ],
    )
    result = run(command, check=False, log_path=ctx.log_path)
    append_log(ctx.log_path, f"PLAN_B_STREAMING returncode={result.returncode}")
    if result.returncode != 0:
        maybe_fail(ctx, deficits, "plan_b_streaming_archive", result.stderr or "Plan B streaming archive failed")
    chunks = sorted(chunks_dir.glob("platform_input_root.tar.zst.part-*"))
    if not chunks:
        maybe_fail(ctx, deficits, "plan_b_streaming_archive", "no archive chunks were produced")
    chunk_rows = []
    for index, chunk in enumerate(chunks):
        chunk_rows.append(
            {
                "index": index,
                "path": str(chunk.relative_to(ctx.release_root)),
                "size_bytes": chunk.stat().st_size,
                "sha256": sha256_path(chunk),
                "blake3": blake3_path(chunk),
            }
        )
    chunk_manifest = {
        "schema_version": "PRISM.archive_chunk_manifest.v1",
        "release_id": ctx.release_id,
        "archive_format": "tar.zst.split",
        "input_root": str(ctx.input_root),
        "top_level_dir": ctx.input_root.name,
        "restored_repo_relative_path": str(ctx.repo_root.relative_to(ctx.input_root)),
        "chunk_size_policy_bytes": ctx.chunk_size_bytes,
        "chunks": chunk_rows,
        "total_archive_bytes": sum(int(row["size_bytes"]) for row in chunk_rows),
    }
    write_json(ctx.release_root / "chunk_manifest.json", chunk_manifest)
    write_csv(ctx.release_root / "chunk_manifest.csv", chunk_rows)
    execution = {
        "schema_version": "PRISM.plan_b_execution.v1",
        "release_id": ctx.release_id,
        "input_root": str(ctx.input_root),
        "repo_root": str(ctx.repo_root),
        "release_root": str(ctx.release_root),
        "selected_plan": "B",
        "dry_run_source_summary_path": str(ctx.release_root / "DRY_RUN_SUMMARY.json"),
        "inventory_record_count": None,
        "input_payload_bytes": capacity_report.get("input_total_bytes"),
        "estimated_archive_bytes": capacity_report.get("estimated_archive_bytes"),
        "actual_archive_bytes": chunk_manifest["total_archive_bytes"],
        "chunk_count": len(chunk_rows),
        "chunk_size_policy": {"gib": ctx.chunk_size_gib, "bytes": ctx.chunk_size_bytes},
        "chunks": chunk_rows,
        "archive_method": "sudo -n tar --xattrs --acls --numeric-owner --preserve-permissions | zstd -T0 --long -19 | split",
        "tar_command": command[-1],
        "compression_method": "zstd",
        "symlink_hardlink_preservation_mode": "tar_metadata_preserving_no_dereference",
        "source_duplicate_avoided": True,
        "scientific_outputs_mutated": False,
        "deletion_performed": False,
    }
    write_json(ctx.release_root / "PLAN_B_EXECUTION.json", execution)
    write_text(
        ctx.release_root / "PLAN_B_EXECUTION.md",
        "\n".join(
            [
                "# Plan B Execution",
                "",
                f"- release_id: `{ctx.release_id}`",
                f"- input_root: `{ctx.input_root}`",
                f"- repo_root: `{ctx.repo_root}`",
                "- selected_plan: `B`",
                "- source_duplicate_avoided: `true`",
                f"- actual_archive_bytes: `{chunk_manifest['total_archive_bytes']}`",
                f"- chunk_count: `{len(chunk_rows)}`",
                "- scientific_outputs_mutated: `false`",
                "- deletion_performed: `false`",
            ]
        )
        + "\n",
    )
    return {
        "path": "archive_chunks/",
        "chunk_manifest": "chunk_manifest.json",
        "exists": bool(chunks),
        "bytes": chunk_manifest["total_archive_bytes"],
        "sha256": None,
        "top_level_dir": ctx.input_root.name,
        "restored_repo_relative_path": str(ctx.repo_root.relative_to(ctx.input_root)),
        "archive_format": "tar.zst.split",
        "chunk_count": len(chunk_rows),
    }


def capture_reference_data(ctx: ReleaseContext, reference_paths: list[Path]) -> dict[str, Any]:
    staged_records = []
    total_bytes = 0
    manifest_only = ctx.no_double_materialization and ctx.layout_kind == "canonical_platform_copy"
    for source in reference_paths:
        relative = source.relative_to(ctx.repo_root)
        target = ctx.release_root / "reference_data" / relative
        if not manifest_only:
            if source.is_dir():
                rsync_copy(source, target, log_path=ctx.log_path, dereference=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            size_bytes = target.stat().st_size if target.is_file() else sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            staged_path = str(target.relative_to(ctx.release_root))
            sha256 = sha256_path(target) if target.is_file() else None
        else:
            size_bytes = source.stat().st_size if source.is_file() else sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
            staged_path = None
            sha256 = sha256_path(source) if source.is_file() else None
        total_bytes += size_bytes
        staged_records.append(
            {
                "original_path": str(source),
                "staged_path": staged_path,
                "overlay_target": str(relative),
                "size_bytes": size_bytes,
                "sha256": sha256,
                "materialization_mode": "manifest_only" if manifest_only else "copied",
            }
        )
    write_json(ctx.release_root / "reference_data" / "REFERENCE_DATA_MANIFEST.json", {"entries": staged_records})
    overlay_script = ctx.release_root / "restore" / "apply_reference_data_overlay.sh"
    overlay_script.parent.mkdir(parents=True, exist_ok=True)
    overlay_script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 <restored_source_root> <release_root/reference_data>" >&2
  exit 2
fi
RESTORED_SOURCE_ROOT="$1"
REFERENCE_DATA_ROOT="$2"
rsync -aHAX --info=progress2 "$REFERENCE_DATA_ROOT/" "$RESTORED_SOURCE_ROOT/"
""",
        encoding="utf-8",
    )
    overlay_script.chmod(0o755)
    return {"entries": staged_records, "total_bytes": total_bytes, "overlay_script": str(overlay_script.relative_to(ctx.release_root))}


def capture_preflight(ctx: ReleaseContext, deficits: list[dict[str, Any]], reference_paths: list[Path]) -> dict[str, Any]:
    out_dir = mkdir(ctx.release_root / "metadata" / "preflight")
    branch = run(["git", "branch", "--show-current"], cwd=ctx.repo_root, log_path=ctx.log_path).stdout.strip()
    head = run(["git", "rev-parse", "HEAD"], cwd=ctx.repo_root, log_path=ctx.log_path).stdout.strip()
    status_short = run(["git", "status", "--short", "--ignored"], cwd=ctx.repo_root, log_path=ctx.log_path).stdout
    remotes = run(["git", "remote", "-v"], cwd=ctx.repo_root, log_path=ctx.log_path).stdout
    findmnt = run(["findmnt", "-rn", "-o", "TARGET,SOURCE,FSTYPE,SIZE,AVAIL"], check=False, log_path=ctx.log_path)
    active_compute = find_active_compute_processes()
    write_text(out_dir / "git_status_short.txt", status_short)
    write_text(out_dir / "git_remotes.txt", remotes)
    write_text(out_dir / "mounts.txt", findmnt.stdout + findmnt.stderr)
    env_var_names = sorted(os.environ.keys())
    write_text(out_dir / "env_var_names.txt", "\n".join(env_var_names) + "\n")
    tool_versions = gather_tool_versions(
        ["git", "python3", "pip", "conda", "rustc", "cargo", "nvcc", "ptxas", "nvidia-smi", "docker", "rclone", "gpg"],
        log_path=ctx.log_path,
    )
    write_json(out_dir / "tool_versions.json", tool_versions)
    write_text(out_dir / "uname.txt", run(["uname", "-a"], log_path=ctx.log_path).stdout)
    write_text(out_dir / "whoami.txt", run(["whoami"], log_path=ctx.log_path).stdout)
    write_text(out_dir / "hostname.txt", run(["hostname"], log_path=ctx.log_path).stdout)
    write_text(out_dir / "pwd.txt", str(ctx.repo_root) + "\n")
    repo_file_count, repo_file_bytes = file_count_and_bytes(ctx.repo_root)
    input_file_count, input_file_bytes = file_count_and_bytes(ctx.input_root)
    env_file_bytes = file_count_and_bytes(ctx.env_root)[1] if ctx.env_root and ctx.env_root.exists() else 0
    candidate_smoke_bytes = (
        file_count_and_bytes(ctx.candidate_smoke_root)[1]
        if ctx.candidate_smoke_root and ctx.candidate_smoke_root.exists()
        else 0
    )
    reference_bytes = sum(path.stat().st_size for path in reference_paths if path.is_file())
    summary = {
        "release_id": ctx.release_id,
        "input_root": str(ctx.input_root),
        "layout_kind": ctx.layout_kind,
        "repo_root": str(ctx.repo_root),
        "campaign_root": str(ctx.campaign_root),
        "branch": branch,
        "head": head,
        "active_compute_processes": active_compute,
        "repo_file_count": repo_file_count,
        "repo_file_bytes": repo_file_bytes,
        "input_file_count": input_file_count,
        "input_file_bytes": input_file_bytes,
        "env_file_bytes": env_file_bytes,
        "candidate_smoke_bytes": candidate_smoke_bytes,
        "reference_dataset_bytes": reference_bytes,
        "root_free_bytes": free_bytes(Path("/")),
        "validate_extract_root": str(ctx.validate_extract_root),
        "validate_extract_root_free_bytes": free_bytes(ctx.validate_extract_root),
        "ssd_free_bytes": free_bytes(ctx.ssd_root),
        "github_enabled": ctx.github_enabled,
        "r2_prefix": ctx.r2_prefix,
        "streaming": ctx.streaming,
        "chunk_size_gib": ctx.chunk_size_gib,
        "no_double_materialization": ctx.no_double_materialization,
        "allow_ssd_single_copy_validation": ctx.allow_ssd_single_copy_validation,
    }
    write_json(out_dir / "preflight_summary.json", summary)
    return summary


def build_capacity_report(ctx: ReleaseContext, records: list[dict[str, Any]], preflight: dict[str, Any]) -> dict[str, Any]:
    input_total_bytes = sum(int(record["size_bytes"]) for record in records if record["file_type"] == "file")
    estimated_archive_bytes = estimate_archive_bytes(records)
    metadata_overhead_bytes = max(2 * 1024**3, int(input_total_bytes * 0.01))
    available_root_bytes = free_bytes(ctx.validate_extract_root)
    available_ssd_bytes = free_bytes(ctx.ssd_root)
    available_r2_bytes = None
    validate_extract_bytes = input_total_bytes

    def plan(
        plan_id: str,
        description: str,
        required_root_bytes: int,
        required_ssd_bytes: int,
        required_r2_bytes: int,
        safety_notes: list[str],
        validation_limitations: list[str],
    ) -> dict[str, Any]:
        return {
            "plan_id": plan_id,
            "description": description,
            "required_root_bytes": required_root_bytes,
            "required_ssd_bytes": required_ssd_bytes,
            "required_r2_bytes": required_r2_bytes,
            "available_root_bytes": available_root_bytes,
            "available_ssd_bytes": available_ssd_bytes,
            "available_r2_bytes": available_r2_bytes,
            "capacity_ok": available_root_bytes >= required_root_bytes and available_ssd_bytes >= required_ssd_bytes,
            "safety_notes": safety_notes,
            "validation_limitations": validation_limitations,
        }

    plans = [
        plan(
            "A",
            "full staging copy + archive + validation extract",
            required_root_bytes=input_total_bytes + validate_extract_bytes + metadata_overhead_bytes,
            required_ssd_bytes=input_total_bytes + estimated_archive_bytes + metadata_overhead_bytes,
            required_r2_bytes=estimated_archive_bytes + metadata_overhead_bytes,
            safety_notes=[
                "Maximal redundancy and easiest local inspection.",
                "Expected to fail on current SSD capacity.",
            ],
            validation_limitations=[],
        ),
        plan(
            "B",
            "no source duplicate, archive streamed directly to SSD",
            required_root_bytes=validate_extract_bytes + metadata_overhead_bytes,
            required_ssd_bytes=estimated_archive_bytes + metadata_overhead_bytes,
            required_r2_bytes=estimated_archive_bytes + metadata_overhead_bytes,
            safety_notes=[
                "Single-copy archive on SSD; no source duplication in release staging.",
                "Requires validation extract capacity on the selected validation filesystem.",
            ],
            validation_limitations=[],
        ),
        plan(
            "C",
            "no source duplicate, archive streamed directly to R2, SSD receives chunked archive plus manifests",
            required_root_bytes=validate_extract_bytes + metadata_overhead_bytes,
            required_ssd_bytes=estimated_archive_bytes + metadata_overhead_bytes,
            required_r2_bytes=estimated_archive_bytes + metadata_overhead_bytes,
            safety_notes=[
                "R2 becomes the authoritative full archive target.",
                "SSD holds a single chunk set and manifests only.",
            ],
            validation_limitations=[
                "R2 restore must be read back and validated before FULL_PASS is possible.",
            ],
        ),
        plan(
            "D",
            "directory-mode rsync to SSD plus archive to R2",
            required_root_bytes=validate_extract_bytes + metadata_overhead_bytes,
            required_ssd_bytes=input_total_bytes + metadata_overhead_bytes,
            required_r2_bytes=estimated_archive_bytes + metadata_overhead_bytes,
            safety_notes=[
                "SSD keeps a directly inspectable directory payload instead of archive chunks.",
                "Avoids a second full archive on SSD.",
            ],
            validation_limitations=[
                "SSD restore validation must be driven from the directory payload, not an SSD archive extract.",
            ],
        ),
        plan(
            "E",
            "R2 full release first, SSD partial only if physical capacity cannot hold full payload",
            required_root_bytes=metadata_overhead_bytes,
            required_ssd_bytes=metadata_overhead_bytes,
            required_r2_bytes=estimated_archive_bytes + metadata_overhead_bytes,
            safety_notes=[
                "Fallback preservation path when SSD cannot hold a full payload.",
                "This cannot qualify for FULL_PASS on SSD.",
            ],
            validation_limitations=[
                "SSD remains PARTIAL if only manifests and partial payload fit.",
            ],
        ),
    ]

    selected_plan = None
    if ctx.plan:
        selected_plan = next((item for item in plans if item["plan_id"] == ctx.plan), None)
        if selected_plan and not selected_plan["capacity_ok"] and ctx.plan != "E":
            selected_plan = {**selected_plan, "forced_by_operator": True}
    if selected_plan is None:
        selected_plan = next((item for item in plans if item["plan_id"] in {"B", "C", "D"} and item["capacity_ok"]), None)
    if selected_plan is None:
        selected_plan = next((item for item in plans if item["plan_id"] == "E"), plans[0])
        if selected_plan["plan_id"] == "E":
            selected_plan["capacity_ok"] = True

    report = {
        "schema_version": "PRISM.capacity_plan.v1",
        "generated_at_utc": now_utc(),
        "release_id": ctx.release_id,
        "input_root": str(ctx.input_root),
        "repo_root": str(ctx.repo_root),
        "layout_kind": ctx.layout_kind,
        "input_total_bytes": input_total_bytes,
        "estimated_archive_bytes": estimated_archive_bytes,
        "metadata_overhead_bytes": metadata_overhead_bytes,
        "validate_extract_root": str(ctx.validate_extract_root),
        "allow_ssd_single_copy_validation": ctx.allow_ssd_single_copy_validation,
        "plans": plans,
        "selected_plan": selected_plan,
        "observed_sizes": {
            "repo_file_bytes": preflight["repo_file_bytes"],
            "input_file_bytes": preflight["input_file_bytes"],
            "env_file_bytes": preflight["env_file_bytes"],
            "candidate_smoke_bytes": preflight["candidate_smoke_bytes"],
            "reference_dataset_bytes": preflight["reference_dataset_bytes"],
        },
    }
    target = ctx.capacity_report_out or (ctx.release_root / "capacity_plan.json")
    write_json(target, report)
    return report


def build_inventory(ctx: ReleaseContext) -> list[dict[str, Any]]:
    inventory_dir = mkdir(ctx.release_root / "inventory")
    source_roots: list[tuple[str, Path]] = [("input_root", ctx.input_root)]
    if (
        ctx.candidate_smoke_root
        and ctx.candidate_smoke_root.exists()
        and not str(ctx.candidate_smoke_root).startswith(str(ctx.input_root))
    ):
        source_roots.append(("candidate_smoke", ctx.candidate_smoke_root))
    git_statuses = git_status_map(ctx.repo_root, log_path=ctx.log_path)
    records = inventory_records(
        source_roots,
        repo_root=ctx.repo_root,
        exclude_dirs={ctx.repo_root / "release_staging"},
        git_statuses=git_statuses,
        compute_hashes=not ctx.dry_run,
    )
    for record in records:
        abs_path = Path(record["absolute_path"])
        try:
            record["repo_relative_path"] = str(abs_path.relative_to(ctx.repo_root))
        except ValueError:
            record["repo_relative_path"] = None
        record["destinations"] = default_destination_plan(record)
    jsonl_path = inventory_dir / "full_inventory.jsonl"
    write_jsonl(jsonl_path, records)
    shutil.copy2(jsonl_path, ctx.release_root / "MANIFEST.jsonl")
    write_csv(inventory_dir / "full_inventory.csv", records)
    write_text(inventory_dir / "file_tree.txt", relative_records_tree(records))
    write_text(inventory_dir / "full_inventory.sha256", f"{sha256_path(jsonl_path)}  full_inventory.jsonl\n")
    large_rows = [record for record in records if record["file_type"] == "file" and record["size_bytes"] > 50 * 1024 * 1024]
    write_text(
        inventory_dir / "large_files_over_50MiB.txt",
        "\n".join(f"{record['size_bytes']}\t{record['source_root_name']}:{record['relative_path']}" for record in large_rows) + "\n",
    )
    db_rows = [record for record in records if record["classification"] == "database"]
    write_text(
        inventory_dir / "database_candidates.txt",
        "\n".join(f"{record['classification']}\t{record['absolute_path']}" for record in db_rows) + "\n",
    )
    return records


def secret_scan(ctx: ReleaseContext, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for record in records:
        if record["file_type"] != "file":
            continue
        path = Path(record["absolute_path"])
        findings.extend(scan_secret_candidates(path))
    unique = []
    seen = set()
    for finding in findings:
        key = (finding["path"], finding["kind"], finding["match"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    security_dir = mkdir(ctx.release_root / "security")
    write_json(security_dir / "secret_scan_redacted.json", {"findings": unique})
    write_text(
        ctx.release_root / "inventory" / "secret_candidates_redacted.txt",
        "\n".join(f"{item['kind']}\t{item['path']}\t{item['match']}" for item in unique) + "\n",
    )
    return unique


def capture_git_state(ctx: ReleaseContext, deficits: list[dict[str, Any]]) -> dict[str, Any]:
    git_dir = mkdir(ctx.release_root / "git")
    write_text(git_dir / "status_short.txt", run(["git", "status", "--short", "--ignored"], cwd=ctx.repo_root, log_path=ctx.log_path).stdout)
    write_text(git_dir / "diff.txt", run(["git", "diff"], cwd=ctx.repo_root, log_path=ctx.log_path).stdout)
    write_text(git_dir / "staged_diff.txt", run(["git", "diff", "--cached"], cwd=ctx.repo_root, log_path=ctx.log_path).stdout)
    write_text(git_dir / "log.txt", run(["git", "log", "--decorate", "--oneline", "-200"], cwd=ctx.repo_root, log_path=ctx.log_path).stdout)
    write_text(git_dir / "reflog.txt", run(["git", "reflog", "--date=iso", "-200"], cwd=ctx.repo_root, log_path=ctx.log_path).stdout)
    write_text(git_dir / "branches.txt", run(["git", "branch", "-a", "-vv"], cwd=ctx.repo_root, log_path=ctx.log_path).stdout)
    write_text(git_dir / "remotes.txt", run(["git", "remote", "-v"], cwd=ctx.repo_root, log_path=ctx.log_path).stdout)
    bundle_path = git_dir / f"{ctx.release_id}.all-refs.bundle"
    result = run(["git", "bundle", "create", str(bundle_path), "--all"], cwd=ctx.repo_root, check=False, log_path=ctx.log_path)
    if result.returncode != 0:
        maybe_fail(ctx, deficits, "git_bundle", f"bundle create failed: {result.stderr}")
    verification = verify_git_bundle(bundle_path, log_path=ctx.log_path) if bundle_path.exists() else {"verified": False}
    if not verification.get("verified"):
        maybe_fail(ctx, deficits, "git_bundle_verify", f"bundle verify failed for {bundle_path}")
    return {"bundle": verification, "bundle_path": str(bundle_path.relative_to(ctx.release_root))}


def capture_database_state(ctx: ReleaseContext, records: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = mkdir(ctx.release_root / "database_backups")
    entries = []
    for record in records:
        if record["file_type"] != "file":
            continue
        path = Path(record["absolute_path"])
        if not is_database_path(path):
            continue
        entry: dict[str, Any] = {
            "source_path": str(path),
            "classification": record["classification"],
            "suffix": path.suffix.lower(),
        }
        if ctx.no_double_materialization:
            entry["logical_backup"] = {
                "status": "skipped_no_double_materialization",
                "reason": "database physical file is preserved inside the Plan B platform archive",
            }
            entries.append(entry)
            continue
        suffix = path.suffix.lower()
        if suffix in {".sqlite", ".sqlite3", ".db"}:
            entry["checkpoint"] = checkpoint_sqlite(path)
            target = out_dir / "sqlite" / f"{path.name}.backup.sqlite"
            entry["logical_backup"] = sqlite_backup(path, target)
        elif suffix == ".duckdb":
            entry["checkpoint"] = duckdb_checkpoint(path)
            target = out_dir / "duckdb" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            entry["physical_copy"] = str(target.relative_to(ctx.release_root))
        else:
            entry["logical_backup"] = {"status": "not_applicable", "reason": "physical preservation via archive/reference data"}
        entries.append(entry)
    write_json(out_dir / "database_capture.json", {"entries": entries})
    return {"entries": entries, "metadata_path": str((out_dir / "database_capture.json").relative_to(ctx.release_root))}


def capture_environment(ctx: ReleaseContext, deficits: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = mkdir(ctx.release_root / "environment")
    exports: dict[str, Any] = {}
    commands = {
        "conda_envs.txt": ["conda", "env", "list"],
        "conda_env_export_prism_dock.yml": ["conda", "env", "export", "-n", "prism_dock", "--no-builds"],
        "conda_list_prism_dock.txt": ["conda", "list", "-n", "prism_dock"],
        "conda_explicit_prism_dock.txt": ["conda", "list", "-n", "prism_dock", "--explicit"],
        "pip_freeze_prism_dock.txt": ["conda", "run", "-n", "prism_dock", "python", "-m", "pip", "freeze"],
        "pip_list_prism_dock.txt": ["conda", "run", "-n", "prism_dock", "python", "-m", "pip", "list"],
    }
    for filename, command in commands.items():
        result = run(command, check=False, log_path=ctx.log_path)
        write_text(out_dir / filename, result.stdout + result.stderr)
        exports[filename] = {"returncode": result.returncode, "path": str((out_dir / filename).relative_to(ctx.release_root))}
    conda_pack = shutil.which("conda-pack")
    if conda_pack and not ctx.no_double_materialization:
        archive = out_dir / "prism_dock_conda_env.tar.gz"
        result = run([conda_pack, "-n", "prism_dock", "-o", str(archive), "--force"], check=False, log_path=ctx.log_path)
        exports["conda_pack"] = {
            "returncode": result.returncode,
            "path": str(archive.relative_to(ctx.release_root)) if archive.exists() else None,
            "exists": archive.exists(),
        }
        write_text(out_dir / "conda_pack_prism_dock.log", result.stdout + result.stderr)
    elif conda_pack and ctx.no_double_materialization:
        exports["conda_pack"] = {
            "returncode": None,
            "exists": False,
            "status": "skipped_no_double_materialization",
        }
    else:
        record_deficit(ctx, deficits, "conda_pack", "conda-pack not available")
        exports["conda_pack"] = {"returncode": None, "exists": False}
    wheelhouse = out_dir / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    if (ctx.repo_root / "requirements.txt").exists():
        result = run(
            ["python3", "-m", "pip", "download", "-r", str(ctx.repo_root / "requirements.txt"), "-d", str(wheelhouse)],
            check=False,
            log_path=ctx.log_path,
        )
        exports["wheelhouse"] = {"returncode": result.returncode, "path": str(wheelhouse.relative_to(ctx.release_root))}
        if result.returncode != 0:
            record_deficit(ctx, deficits, "wheelhouse", "pip download did not complete successfully")
    cargo_metadata = run(["cargo", "metadata", "--format-version", "1", "--locked"], cwd=ctx.repo_root, check=False, log_path=ctx.log_path)
    write_text(out_dir / "cargo_metadata.json", cargo_metadata.stdout or cargo_metadata.stderr)
    return {"exports": exports, "path": str(out_dir.relative_to(ctx.release_root))}


def capture_runtime(ctx: ReleaseContext) -> dict[str, Any]:
    runtime_root = mkdir(ctx.release_root / "runtime")
    bin_dir = mkdir(runtime_root / "bin")
    ptx_dir = mkdir(runtime_root / "ptx")
    ldd_dir = mkdir(runtime_root / "ldd")
    source_dir = mkdir(runtime_root / "source")
    release_dir = ctx.repo_root / "target" / "release"
    copied_bins = []
    if release_dir.exists():
        for path in sorted(release_dir.iterdir()):
            if path.is_file() and (os.access(path, os.X_OK) or path.suffix == ".so"):
                target = bin_dir / path.name
                shutil.copy2(path, target)
                copied_bins.append(str(target.relative_to(ctx.release_root)))
                run(["ldd", str(target)], check=False, log_path=ctx.log_path)
                write_text(ldd_dir / f"{path.name}.ldd.txt", run(["ldd", str(target)], check=False, log_path=ctx.log_path).stdout)
                write_text(ldd_dir / f"{path.name}.readelf.txt", run(["readelf", "-d", str(target)], check=False, log_path=ctx.log_path).stdout)
    ptx_paths = [
        ctx.repo_root / "crates/prism-gpu/src/kernels",
        ctx.repo_root / "target/ptx",
        ctx.repo_root / "crates/prism-nhs/src/cuda",
        ctx.repo_root / "vendor/working_ptx_2026-05-29",
    ]
    for ptx_path in ptx_paths:
        if ptx_path.exists():
            target = ptx_dir / ptx_path.relative_to(ctx.repo_root)
            if ptx_path.is_dir():
                rsync_copy(ptx_path, target, log_path=ctx.log_path)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ptx_path, target)
    source_materialization = "inside_platform_archive_only"
    if not ctx.no_double_materialization:
        source_materialization = "copied_runtime_source_subset"
        source_paths = [
            ctx.repo_root / "scripts",
            ctx.repo_root / "src",
            ctx.repo_root / "crates",
            ctx.repo_root / "tests/test_v2_hydration_extractor.py",
            ctx.repo_root / "tests/test_continuity_maps.py",
            ctx.repo_root / "tests/release_acceptance",
            ctx.repo_root / "Cargo.toml",
            ctx.repo_root / "Cargo.lock",
            ctx.repo_root / "pyproject.toml",
            ctx.campaign_root / "PHASE2C_SEALED_MANIFEST.json",
            ctx.campaign_root / "candidate_dossiers/cand_015_bccda098.json",
            ctx.campaign_root / "track_a_generative/motif_intelligence_tripartite_top50",
        ]
        for source_path in source_paths:
            if not source_path.exists():
                continue
            rel = source_path.relative_to(ctx.repo_root) if source_path.is_relative_to(ctx.repo_root) else source_path.relative_to(ctx.campaign_root.parent.parent)
            target = source_dir / rel
            if source_path.is_dir():
                rsync_copy(source_path, target, log_path=ctx.log_path)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)
    return {
        "runtime_root": str(runtime_root.relative_to(ctx.release_root)),
        "binary_count": len(copied_bins),
        "ptx_count": len(list(ptx_dir.rglob("*.ptx"))),
        "source_materialization": source_materialization,
    }


def write_validation_scripts(ctx: ReleaseContext) -> dict[str, Any]:
    validation_root = mkdir(ctx.release_root / "validation")
    runtime_root = ctx.release_root / "runtime"
    activate = runtime_root / "activate_prism_release.sh"
    activate.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
export PRISM_RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$PRISM_RELEASE_ROOT/runtime/bin:$PATH"
export PYTHONPATH="$PRISM_RELEASE_ROOT/runtime/source:$PRISM_RELEASE_ROOT/runtime/source/src:${PYTHONPATH:-}"
export PRISM_PTX_ROOT="$PRISM_RELEASE_ROOT/runtime/ptx"
""",
        encoding="utf-8",
    )
    activate.chmod(0o755)
    validate = validation_root / "validate_release.sh"
    candidate_smoke_arg = ""
    if ctx.candidate_smoke_root and ctx.candidate_smoke_root.exists() and not ctx.no_double_materialization:
        candidate_smoke_arg = (
            'python3 "$ROOT/runtime/source/scripts/build_candidate_motif_summary.py" '
            '--candidate-id cand_015_bccda098 '
            '--candidate-dossier "$ROOT/runtime/source/campaigns/glp1r_aleniglipron/candidate_dossiers/cand_015_bccda098.json" '
            '--candidate-source "$ROOT/reference_data/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top_50_tripartite_profiles.parquet" '
            '--motif-registry "$ROOT/runtime/source/campaigns/glp1r_aleniglipron/track_a_generative/motif_intelligence_tripartite_top50/thermodynamic_motif_registry.parquet" '
            '--motif-registry-report "$ROOT/runtime/source/campaigns/glp1r_aleniglipron/track_a_generative/motif_intelligence_tripartite_top50/thermodynamic_motif_registry_report.json" '
            '--binding-sites-materialized "$ROOT/external_evidence/candidate_smoke/materialized_openff/cand_015_bccda098_glp1r_6XOX_WT_replica_0/binding_sites.materialized.json" '
            '--md-evidence-manifest "$ROOT/external_evidence/candidate_smoke/engine_openff/cand_015_bccda098_glp1r_6XOX_WT_replica_0/md_evidence_manifest.json" '
            '--output "$ROOT/validation/output/cand_015_bccda098_motif_summary.validation.json" >/dev/null\n'
        )
    validate.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
SRC="${{PRISM_RESTORED_SOURCE_ROOT:-$ROOT/runtime/source}}"
source "$ROOT/runtime/activate_prism_release.sh" >/dev/null
mkdir -p "$ROOT/validation/output"
python3 -m json.tool "$ROOT/MANIFEST.json" >/dev/null
python3 -m json.tool "$SRC/campaigns/glp1r_aleniglipron/PHASE2C_SEALED_MANIFEST.json" >/dev/null
"$ROOT/runtime/bin/prism-materialize-sites" --help >/dev/null
"$ROOT/runtime/bin/dstw_export_wt_pathb" --help >/dev/null
grep -a -q 'producer_frames_enqueued' "$ROOT/runtime/bin/nhs_rt_full"
grep -a -q 'all_hashes_match' "$ROOT/runtime/bin/nhs_rt_full"
{candidate_smoke_arg}echo "release_validation=PASS"
""",
        encoding="utf-8",
    )
    validate.chmod(0o755)
    return {"activate": str(activate.relative_to(ctx.release_root)), "validate": str(validate.relative_to(ctx.release_root))}


def capture_external_evidence(ctx: ReleaseContext) -> dict[str, Any]:
    if not ctx.candidate_smoke_root or not ctx.candidate_smoke_root.exists():
        return {"present": False}
    if ctx.no_double_materialization and str(ctx.candidate_smoke_root).startswith(str(ctx.input_root)):
        return {
            "present": True,
            "materialization_mode": "referenced_from_input_root",
            "path_within_input_root": str(ctx.candidate_smoke_root.relative_to(ctx.input_root)),
        }
    target = ctx.release_root / "external_evidence" / "candidate_smoke"
    rsync_copy(ctx.candidate_smoke_root, target, log_path=ctx.log_path)
    return {"present": True, "path": str(target.relative_to(ctx.release_root))}


def build_candidate_matrix(ctx: ReleaseContext) -> dict[str, Any]:
    entries = []
    conflicts = []
    parquet_status = {}
    parquet_path = ctx.campaign_root / "track_a_generative" / "gflownet_top_50_tripartite_profiles.parquet"
    try:
        import polars as pl  # type: ignore
    except Exception:
        pl = None  # type: ignore
    if pl is not None and parquet_path.exists():
        df = pl.read_parquet(parquet_path)
        for row in df.to_dicts():
            parquet_status[str(row["candidate_id"])] = row
    candidate_ids = set()
    for path in ctx.campaign_root.glob("candidate_dossiers/*.json"):
        candidate_ids.add(path.stem)
    for path in ctx.campaign_root.glob("track_a_generative/candidate_dossiers/*.json"):
        candidate_ids.add(path.stem)
    for path in ctx.campaign_root.glob("candidate_dossiers_population/*.md"):
        candidate_ids.add(path.stem.split(".")[0])
    for candidate_id in sorted(candidate_ids):
        status_sources: dict[str, str] = {}
        entry: dict[str, Any] = {
            "candidate_id": candidate_id,
            "dossier_paths": [],
            "sdf_paths": [],
            "topology_paths": [],
            "expected_outputs": [
                "md_evidence_manifest.json",
                "binding_sites.materialized.json",
                "wt_physics_payload.parquet",
                "wt_contact_graph.parquet",
                "candidate_specific_motif_summary.json",
            ],
            "observed_outputs": [],
            "claim_boundary": "candidate-specific validation may be claimed only for observed outputs present here",
        }
        json_paths = [
            ctx.campaign_root / "candidate_dossiers" / f"{candidate_id}.json",
            ctx.campaign_root / "track_a_generative/candidate_dossiers" / f"{candidate_id}.json",
        ]
        for json_path in json_paths:
            if not json_path.exists():
                continue
            payload = load_json(json_path)
            entry["dossier_paths"].append(str(json_path.relative_to(ctx.repo_root)))
            if payload.get("canonical_smiles") and not entry.get("canonical_smiles"):
                entry["canonical_smiles"] = payload["canonical_smiles"]
            if payload.get("gpu_dispatch_status"):
                status_sources[str(json_path.relative_to(ctx.repo_root))] = str(payload["gpu_dispatch_status"])
        if candidate_id in parquet_status:
            row = parquet_status[candidate_id]
            entry.setdefault("canonical_smiles", row.get("canonical_smiles"))
            status_sources[str(parquet_path.relative_to(ctx.repo_root))] = str(row.get("gpu_dispatch_status"))
        for path in sorted(ctx.campaign_root.glob(f"**/*{candidate_id}*.sdf")):
            entry["sdf_paths"].append(str(path.relative_to(ctx.repo_root)))
        for path in sorted(ctx.campaign_root.glob(f"**/*{candidate_id}*.json")):
            if "topologies" in path.parts and path.is_file():
                entry["topology_paths"].append(str(path.relative_to(ctx.repo_root)))
        observed = []
        for path in sorted(ctx.campaign_root.glob(f"**/*{candidate_id}*")):
            if path.is_file() and path.suffix in {".json", ".parquet", ".sdf", ".sh"}:
                if "candidate_dossiers" in path.parts:
                    continue
                observed.append(str(path.relative_to(ctx.repo_root)))
        if ctx.candidate_smoke_root and ctx.candidate_smoke_root.exists():
            for path in sorted(ctx.candidate_smoke_root.glob(f"**/*{candidate_id}*")):
                if path.is_file():
                    observed.append(str(path))
        entry["observed_outputs"] = observed
        entry["status_sources"] = status_sources
        unique_statuses = sorted(set(status_sources.values()))
        entry["resolved_status"] = unique_statuses[0] if len(unique_statuses) == 1 else (status_sources.get(str((ctx.campaign_root / 'candidate_dossiers' / f'{candidate_id}.json').relative_to(ctx.repo_root))) or (unique_statuses[0] if unique_statuses else "unknown"))
        if len(unique_statuses) > 1:
            conflicts.append({"candidate_id": candidate_id, "statuses": status_sources})
        entries.append(entry)
    matrix_path = ctx.release_root / "candidate_matrix_manifest.json"
    write_json(matrix_path, {"schema_version": "PRISM.candidate_matrix_manifest.v1", "entries": entries})
    reconciliation_path = ctx.release_root / "STATUS_RECONCILIATION.md"
    lines = [
        "# Status Reconciliation",
        "",
        "Conflicts are recorded, not normalized away.",
        "",
    ]
    if conflicts:
        for conflict in conflicts:
            lines.append(f"## {conflict['candidate_id']}")
            for source, status in conflict["statuses"].items():
                lines.append(f"- `{source}` -> `{status}`")
            lines.append("")
    else:
        lines.append("No conflicts detected.")
    write_text(reconciliation_path, "\n".join(lines).rstrip() + "\n")
    return {
        "manifest_path": str(matrix_path.relative_to(ctx.release_root)),
        "status_reconciliation_path": str(reconciliation_path.relative_to(ctx.release_root)),
        "entry_count": len(entries),
        "conflict_count": len(conflicts),
    }


def write_claim_boundary(ctx: ReleaseContext) -> None:
    text = """# Release Claim Boundary

- Phase 2C sealed receptor/variant evidence may be claimed.
- Platform smoke validation may be claimed.
- V2 hydration to DSTW context integration is implemented and smoke-verified.
- Full hydration extraction across the 104GB input surface has not been fully run and remains unmeasured.
- Full Phase 1-3 production completion may not be claimed.
- Candidate matrix completion may not be claimed beyond observed candidate-specific MD/Path-B evidence present in this release.
- Hydration completion may not be claimed unless hydration inputs and outputs are present and validated.
- This release is a sealed operational snapshot containing completed, partial, planned, and in-progress surfaces.
"""
    write_text(ctx.release_root / "RELEASE_CLAIM_BOUNDARY.md", text)


def capture_canonical_status(ctx: ReleaseContext) -> dict[str, Any]:
    out_dir = mkdir(ctx.release_root / "metadata" / "canonical_status")
    copied = {}
    for name in (
        "CANONICAL_LOCAL_SYSTEM_STATUS.md",
        "CANONICAL_LOCAL_SYSTEM_STATUS.json",
        "WORKSTATION_ARCHIVE_PLAN_20260529.md",
        "WORKSTATION_ARCHIVE_R2_BENCHMARK_20260529.md",
    ):
        source = ctx.input_root / name
        if not source.exists():
            continue
        target = out_dir / name
        shutil.copy2(source, target)
        copied[name] = str(target.relative_to(ctx.release_root))
    return copied


def write_restore_docs(ctx: ReleaseContext, reference_capture: dict[str, Any]) -> None:
    restore_order = """# Restore Order

1. Verify checksums: `sha256sum -c CHECKSUMS.sha256`
2. Verify git bundle: `git bundle verify git/<release_id>.all-refs.bundle`
3. Extract `archives/repo_working_tree.tar.zst` into a clean directory.
4. Apply reference data overlay:
   `bash restore/apply_reference_data_overlay.sh <restored_source_root> <release_root>/reference_data`
5. Restore packed conda envs or use exported specs from `environment/`.
6. Run `python3 scripts/verify_restored_release.py --release-root <release_root> --run-smoke --run-db-checks --run-cuda-checks --run-candidate-checks`
"""
    write_text(ctx.release_root / "RESTORE_ORDER.md", restore_order.replace("<release_id>", ctx.release_id))
    readme = f"""# README Restore

Release ID: `{ctx.release_id}`

This package restores the current truthful claim boundary:

- Phase 2C sealed receptor/variant evidence is preserved.
- Platform smoke state is preserved.
- Hydration to DSTW context integration is preserved as implemented-and-smoke-verified.
- Full 104GB hydration extraction remains not fully run / unmeasured.
- Candidate matrix completion is partial and explicitly reconciled.

Cold restore entrypoint:

```bash
python3 scripts/verify_restored_release.py \\
  --release-root <release_root> \\
  --run-smoke --run-db-checks --run-cuda-checks --run-candidate-checks
```

Static reference data override:

The Enamine/V-space corpus is staged under:

`reference_data/campaigns/glp1r_aleniglipron/track_a_generative/`

On an air-gapped machine, restore the repository archive first, then apply the overlay:

```bash
bash restore/apply_reference_data_overlay.sh \\
  <restored_source_root> \\
  <release_root>/reference_data
```

That rehydrates the expected campaign-relative paths the engine uses today.

Acceptance command:

```bash
PRISM_RELEASE_ROOT=<release_root> make release-acceptance
```
"""
    write_text(ctx.release_root / "README_RESTORE.md", readme)


def write_provenance(ctx: ReleaseContext, manifest: dict[str, Any]) -> None:
    prov_dir = mkdir(ctx.release_root / "provenance")
    tool_versions = load_json(ctx.release_root / "metadata/preflight/tool_versions.json")
    input_materials = {
        "repo_head": manifest["git"]["head"],
        "reference_data": manifest["reference_data"]["entries"],
    }
    output_artifacts = {
        "release_root": str(ctx.release_root),
        "archive": manifest["repo_snapshot"],
        "runtime": manifest["runtime"],
    }
    write_json(prov_dir / "TOOL_VERSIONS.json", tool_versions)
    write_json(prov_dir / "INPUT_MATERIALS.json", input_materials)
    write_json(prov_dir / "OUTPUT_ARTIFACTS.json", output_artifacts)
    write_text(prov_dir / "BUILD_COMMANDS.log", ctx.log_path.read_text(encoding="utf-8") if ctx.log_path.exists() else "")
    write_json(
        prov_dir / "SLSA_PROVENANCE.json",
        {
            "schema_version": "PRISM.slsa_provenance.v1",
            "build_time_utc": now_utc(),
            "builder": {"host": os.uname().nodename, "user": os.environ.get("USER", "")},
            "materials": input_materials,
            "outputs": output_artifacts,
        },
    )
    write_json(
        prov_dir / "IN_TOTO_ATTESTATION.json",
        {
            "schema_version": "PRISM.in_toto_attestation.v1",
            "predicate_type": "https://in-toto.io/Statement/v1",
            "subject": [{"name": ctx.release_id}],
            "predicate": {"buildType": "sealed-operational-release", "metadata": {"release_root": str(ctx.release_root)}},
        },
    )


def write_evidence_index(ctx: ReleaseContext, manifest: dict[str, Any]) -> None:
    lines = [
        "# Release Evidence Index",
        "",
        "| Claim | Evidence |",
        "|---|---|",
        f"| Phase 2C sealed | `runtime/source/campaigns/glp1r_aleniglipron/PHASE2C_SEALED_MANIFEST.json` |",
        f"| Hydration DSTW integration | `runtime/source/scripts/prism_v2_hydration_extractor.py`, `runtime/source/tests/test_v2_hydration_extractor.py`, `runtime/source/tests/test_continuity_maps.py`, and `metadata/canonical_status/CANONICAL_LOCAL_SYSTEM_STATUS.md` |",
        f"| Hydration full-run boundary | `RELEASE_CLAIM_BOUNDARY.md` and `metadata/canonical_status/CANONICAL_LOCAL_SYSTEM_STATUS.md` |",
        f"| Candidate reconciliation | `{manifest['candidate_matrix']['status_reconciliation_path']}` |",
        f"| Destination matrix | `DESTINATION_MATRIX.csv` |",
        f"| Restore source matrix | `RESTORE_SOURCE_MATRIX.json` |",
        f"| Release status | `RELEASE_STATUS.json` |",
        f"| Cold restore validation | `COLD_RESTORE_VALIDATION.json` |",
        f"| Policy evaluation | `POLICY_EVALUATION.json` |",
        f"| Destination readback | `DESTINATION_READBACK_VALIDATION.json` |",
    ]
    write_text(ctx.release_root / "RELEASE_EVIDENCE_INDEX.md", "\n".join(lines) + "\n")


def write_manifest(ctx: ReleaseContext, records: list[dict[str, Any]], preflight: dict[str, Any], reference_capture: dict[str, Any], git_state: dict[str, Any], runtime: dict[str, Any], env_capture: dict[str, Any], db_capture: dict[str, Any], candidate_matrix: dict[str, Any], repo_snapshot: dict[str, Any], external_evidence: dict[str, Any], secret_findings: list[dict[str, Any]], deficits: list[dict[str, Any]]) -> dict[str, Any]:
    classification_counts = Counter(record["classification"] for record in records)
    destination_counts = defaultdict(int)
    for record in records:
        for destination, plan in record["destinations"].items():
            if plan["include"]:
                destination_counts[destination] += 1
    manifest = {
        "schema_version": "PRISM.sealed_operational_release_manifest.v1",
        "release_id": ctx.release_id,
        "created_at_utc": now_utc(),
        "input_root": str(ctx.input_root),
        "layout_kind": ctx.layout_kind,
        "repo_root": str(ctx.repo_root),
        "release_root": str(ctx.release_root),
        "inventory": {
            "record_count": len(records),
            "classification_counts": dict(classification_counts),
            "destination_include_counts": dict(destination_counts),
            "all_accounted_for": all("destinations" in record for record in records),
            "global_release_root_hash": compute_release_root_hash(records),
        },
        "preflight": preflight,
        "capacity_report_path": str((ctx.capacity_report_out or (ctx.release_root / "capacity_plan.json")).relative_to(ctx.release_root) if (ctx.capacity_report_out or (ctx.release_root / "capacity_plan.json")).is_relative_to(ctx.release_root) else (ctx.capacity_report_out or (ctx.release_root / "capacity_plan.json"))),
        "reference_data": reference_capture,
        "git": {
            "head": preflight["head"],
            "branch": preflight["branch"],
            **git_state,
        },
        "runtime": runtime,
        "environment": env_capture,
        "database_capture": db_capture,
        "candidate_matrix": candidate_matrix,
        "claim_boundary": {
            "phase2c_status": "SEALED",
            "platform_smoke_status": "VALID",
            "hydration_dstw_integration_status": "IMPLEMENTED_AND_SMOKE_VERIFIED",
            "hydration_full_run_status": "NOT_RUN_FULL_SCALE",
            "hydration_full_run_detail": "NOT_FULLY_RUN_104GB_INPUT_UNMEASURED",
            "hydration_runtime_memory_unmeasured": True,
            "phase1_to_3_full_production_status": "NOT_CLAIMABLE",
            "candidate_matrix_completion_status": "PARTIAL_ONLY_OBSERVED_OUTPUTS_CLAIMABLE",
            "hydration_production_complete": False,
            "release_mix_status": "COMPLETED_PARTIAL_PLANNED_IN_PROGRESS",
        },
        "repo_snapshot": repo_snapshot,
        "external_evidence": external_evidence,
        "security": {"secret_findings_count": len(secret_findings)},
        "deficit_count": len(deficits),
    }
    write_json(ctx.release_root / "MANIFEST.json", manifest)
    return manifest


def write_repo_mirror(ctx: ReleaseContext, deficits: list[dict[str, Any]]) -> None:
    mirror_root = mkdir(ctx.repo_mirror_root)
    write_text(
        mirror_root / "MIRROR_SCOPE.md",
        "This GitHub-safe mirror contains release manifests, restore docs, policy evaluation, and destination stubs.\n",
    )
    stubs = []
    manifest = load_json(ctx.release_root / "MANIFEST.json")
    inventory_path = ctx.release_root / "inventory" / "full_inventory.jsonl"
    with inventory_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not record["destinations"]["github"]["include"]:
                stubs.append(
                    {
                        "relative_path": f"{record['source_root_name']}:{record['relative_path']}",
                        "reason": record["destinations"]["github"]["reason"],
                        "size_bytes": record["size_bytes"],
                        "ssd_path": str(Path(record["absolute_path"])) if record.get("repo_relative_path") else None,
                        "r2_object": f"{ctx.r2_prefix}/{ctx.release_id}/{record['source_root_name']}/{record['relative_path']}" if ctx.r2_prefix else None,
                    }
                )
    write_json(mirror_root / "GITHUB_OMISSIONS.json", {"entries": stubs})
    write_json(
        mirror_root / "MANIFEST_SUMMARY.json",
        {
            "release_id": ctx.release_id,
            "global_release_root_hash": manifest["inventory"]["global_release_root_hash"],
            "deficit_count": manifest["deficit_count"],
            "candidate_matrix": manifest["candidate_matrix"],
        },
    )
    for path in classify_repo_mirror_files(ctx.release_root, ctx.release_id):
        if path.exists():
            target = mirror_root / path.name if path.parent == ctx.release_root else mirror_root / path.relative_to(ctx.release_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    if deficits:
        write_json(mirror_root / "RELEASE_DEFICITS.json", {"entries": deficits})


def upload_r2(ctx: ReleaseContext, deficits: list[dict[str, Any]]) -> dict[str, Any]:
    if not ctx.r2_prefix:
        return {"status": "NOT_STARTED", "detail": "r2_upload_deferred_by_operator"}
    target = f"{ctx.r2_prefix}/{ctx.release_id}"
    result = run(
        [
            "rclone",
            "copy",
            str(ctx.release_root),
            target,
            "--progress",
            "--retries",
            "5",
            "--low-level-retries",
            "10",
            "--transfers",
            "16",
            "--checkers",
            "32",
            "--s3-chunk-size",
            "128M",
            "--fast-list",
            "--stats",
            "30s",
            "--stats-one-line",
            "--log-level",
            "INFO",
        ],
        check=False,
        log_path=ctx.log_path,
    )
    if result.returncode != 0:
        record_deficit(ctx, deficits, "r2_upload", f"rclone copy failed: {result.stderr}")
        return {"status": "FAIL", "detail": result.stderr.strip(), "location": target}
    list_result = run(["rclone", "lsf", target], check=False, log_path=ctx.log_path)
    status = "PARTIAL_PASS"
    detail = "uploaded_and_listed; full cold restore from r2 not executed"
    return {"status": status, "detail": detail, "location": target, "listed": list_result.returncode == 0}


def verify_local_release(ctx: ReleaseContext, deficits: list[dict[str, Any]]) -> dict[str, Any]:
    provisional_checksums = write_checksums(ctx.release_root)
    result = run(
        [
            "python3",
            str(SCRIPT_REPO_ROOT / "scripts/verify_restored_release.py"),
            "--release-root",
            str(ctx.release_root),
            "--expected-release-id",
            ctx.release_id,
            "--run-smoke",
            "--run-db-checks",
            "--run-cuda-checks",
            "--run-candidate-checks",
        ]
        + (["--skip-expensive-tests"] if ctx.skip_expensive_tests else []),
        check=False,
        log_path=ctx.log_path,
    )
    if result.returncode != 0:
        record_deficit(ctx, deficits, "cold_restore", "cold restore verification returned non-zero")
    return load_json(ctx.release_root / "COLD_RESTORE_VALIDATION.json")


def evaluate_policy(ctx: ReleaseContext, deficits: list[dict[str, Any]]) -> dict[str, Any]:
    result = run(
        [
            "python3",
            str(SCRIPT_REPO_ROOT / "scripts/evaluate_release_policy.py"),
            "--release-root",
            str(ctx.release_root),
            "--policy",
            str(SCRIPT_REPO_ROOT / "release_policy.yaml"),
        ],
        check=False,
        log_path=ctx.log_path,
    )
    if result.returncode != 0:
        record_deficit(ctx, deficits, "policy_evaluation", "policy evaluation returned non-zero")
    return load_json(ctx.release_root / "POLICY_EVALUATION.json")


def main() -> int:
    args = parse_args()
    layout = resolve_prism_input_layout(args.input_root)
    if args.repo_root:
        repo_root = args.repo_root.resolve()
        input_root = args.input_root.resolve()
        if not repo_root.exists():
            raise RuntimeError(f"--repo-root does not exist: {repo_root}")
        if not repo_root.is_relative_to(input_root):
            raise RuntimeError(f"--repo-root must be inside --input-root: {repo_root} not under {input_root}")
        layout["repo_root"] = repo_root
        layout["campaign_root"] = repo_root / "campaigns" / "glp1r_aleniglipron"
        layout["input_root"] = input_root
    release_id = compute_release_id(Path(layout["repo_root"]), args.release_id)
    ssd_base = args.ssd_root.resolve()
    release_parent = ssd_base if ssd_base.name == "sealed_operational_releases" else ssd_base / "sealed_operational_releases"
    release_root = release_parent / release_id
    log_path = release_root / "logs" / "seal_operational_release.log"
    no_double_materialization = bool(args.no_double_materialization or args.no_source_duplicate)
    ctx = ReleaseContext(
        release_id=release_id,
        release_root=release_root,
        input_root=Path(layout["input_root"]),
        layout_kind=str(layout["layout_kind"]),
        repo_root=Path(layout["repo_root"]),
        campaign_root=Path(layout["campaign_root"]),
        env_root=Path(layout["env_root"]) if layout.get("env_root") else None,
        candidate_smoke_root=Path(layout["candidate_smoke_root"]) if layout.get("candidate_smoke_root") else None,
        credentials_path=Path(layout["credentials_path"]) if layout.get("credentials_path") else None,
        rclone_config_path=Path(layout["rclone_config_path"]) if layout.get("rclone_config_path") else None,
        repo_mirror_root=GITHUB_SAFE_ROOT / release_id,
        log_path=log_path,
        continue_on_error=args.continue_on_error,
        skip_expensive_tests=args.skip_expensive_tests,
        github_enabled=args.github,
        ssd_root=args.ssd_root.resolve(),
        r2_prefix=args.r2 if args.upload_r2 else None,
        upload_r2_enabled=args.upload_r2,
        dry_run=args.dry_run,
        plan=args.plan,
        streaming=args.streaming,
        chunk_size_gib=args.chunk_size_gib,
        chunk_size_bytes=chunk_size_bytes(args.chunk_size_gib),
        no_double_materialization=no_double_materialization,
        validate_extract_root=args.validate_extract_root.resolve(),
        allow_ssd_single_copy_validation=args.allow_ssd_single_copy_validation,
        capacity_report_out=args.capacity_report_out.resolve() if args.capacity_report_out else None,
    )
    mkdir(release_root)
    deficits: list[dict[str, Any]] = []
    reference_paths = locate_reference_data(ctx.repo_root)
    preflight = capture_preflight(ctx, deficits, reference_paths)
    records = build_inventory(ctx)
    capacity_report = build_capacity_report(ctx, records, preflight)
    secret_findings = [] if ctx.dry_run else secret_scan(ctx, records)
    if ctx.dry_run:
        dry_run_summary = {
            "schema_version": "PRISM.sealed_operational_release_dry_run.v1",
            "release_id": ctx.release_id,
            "release_root": str(ctx.release_root),
            "preflight": preflight,
            "capacity_report_path": str((ctx.capacity_report_out or (ctx.release_root / "capacity_plan.json"))),
            "selected_capacity_plan": capacity_report["selected_plan"],
            "inventory_record_count": len(records),
            "secret_finding_count": len(secret_findings),
            "reference_data_paths": [str(path) for path in reference_paths],
            "status": "DRY_RUN",
        }
        write_json(ctx.release_root / "DRY_RUN_SUMMARY.json", dry_run_summary)
        print(f"RELEASE_ID={ctx.release_id}")
        print(f"RELEASE_ROOT={ctx.release_root}")
        print("FINAL_STATUS=DRY_RUN")
        return 0
    non_dry_preflight = write_non_dry_preflight(ctx, capacity_report)
    if capacity_report.get("selected_plan", {}).get("plan_id") != "B" and args.plan == "B":
        maybe_fail(ctx, deficits, "capacity_plan", "operator forced Plan B but capacity planner did not select B")
    if args.plan == "B" and (not ctx.streaming or not ctx.no_double_materialization):
        maybe_fail(ctx, deficits, "plan_b_flags", "Plan B non-dry requires --streaming and --no-source-duplicate/--no-double-materialization")
    reference_capture = capture_reference_data(ctx, reference_paths)
    git_state = capture_git_state(ctx, deficits)
    db_capture = capture_database_state(ctx, records)
    env_capture = capture_environment(ctx, deficits)
    runtime = capture_runtime(ctx)
    runtime["helpers"] = write_validation_scripts(ctx)
    runtime["canonical_status"] = capture_canonical_status(ctx)
    external_evidence = capture_external_evidence(ctx)
    if capacity_report.get("selected_plan", {}).get("plan_id") == "B" and ctx.streaming and ctx.no_double_materialization:
        repo_snapshot = create_platform_archive_chunks(ctx, capacity_report, deficits)
        plan_b_execution_path = ctx.release_root / "PLAN_B_EXECUTION.json"
        if plan_b_execution_path.exists():
            plan_b_execution = load_json(plan_b_execution_path)
            plan_b_execution["inventory_record_count"] = len(records)
            write_json(plan_b_execution_path, plan_b_execution)
    else:
        repo_snapshot = create_repo_archive(ctx, reference_paths, deficits)
    write_claim_boundary(ctx)
    candidate_matrix = build_candidate_matrix(ctx)
    write_restore_docs(ctx, reference_capture)
    manifest = write_manifest(
        ctx,
        records,
        preflight,
        reference_capture,
        git_state,
        runtime,
        env_capture,
        db_capture,
        candidate_matrix,
        repo_snapshot,
        external_evidence,
        secret_findings,
        deficits,
    )
    write_provenance(ctx, manifest)
    write_evidence_index(ctx, manifest)
    if capacity_report.get("selected_plan", {}).get("plan_id") == "B":
        destination_rows = [
            {"destination": "github", "status": "NOT_STARTED", "detail": "explicitly deferred until SSD proof report is reviewed", "location": "repo mirror under release_artifacts/"},
            {"destination": "ssd", "status": "SSD_STORAGE_WRITTEN_PENDING_READBACK", "detail": "Plan B chunks written; independent post-seal readback required", "location": str(ctx.release_root)},
            {"destination": "r2", "status": "NOT_STARTED", "detail": "explicitly deferred until SSD proof report is reviewed", "location": ctx.r2_prefix or ""},
        ]
        write_destination_matrix(ctx.release_root / "DESTINATION_MATRIX.csv", destination_rows)
        write_json(
            ctx.release_root / "DESTINATION_READBACK_VALIDATION.json",
            {
                "schema_version": "PRISM.destination_readback_validation.v1",
                "verified_at_utc": now_utc(),
                "overall_status": "NOT_RUN",
                "destinations": {
                    "github": {"status": "NOT_STARTED"},
                    "ssd": {"status": "PENDING_READBACK"},
                    "r2": {"status": "NOT_STARTED"},
                },
            },
        )
        write_json(
            ctx.release_root / "RESTORE_SOURCE_MATRIX.json",
            {
                "schema_version": "PRISM.restore_source_matrix.v1",
                "github_only_restore": {"status": "NOT_STARTED"},
                "ssd_full_restore": {"status": "PENDING_READBACK_AND_COLD_RESTORE"},
                "r2_full_restore": {"status": "NOT_STARTED"},
            },
        )
        write_json(
            ctx.release_root / "POLICY_EVALUATION.json",
            {
                "schema_version": "PRISM.sealed_release_policy_evaluation.v1",
                "evaluated_at_utc": now_utc(),
                "release_root": str(ctx.release_root),
                "status": "NOT_RUN",
                "reason": "post-seal agent gate has not executed SSD readback/cold restore yet",
            },
        )
        write_text(
            ctx.release_root / "RELEASE_DEFICITS.md",
            "# Release Deficits\n\n"
            + ("\n".join(f"- {item['time_utc']} `{item['step']}`: {item['detail']}" for item in deficits) if deficits else "- None recorded before post-seal verification.\n"),
        )
        release_status = {
            "schema_version": "PRISM.sealed_release_status.v1",
            "release_id": ctx.release_id,
            "status": "TRACK_A_SSD_STORAGE_WRITTEN_PENDING_READBACK",
            "cold_restore_status": "NOT_RUN",
            "policy_status": "NOT_RUN",
            "destination_statuses": {row["destination"]: row["status"] for row in destination_rows},
            "git": {"branch": preflight["branch"], "head": preflight["head"], "tag_pushed": False},
            "no_deletion_clearance": True,
            "non_dry_preflight": non_dry_preflight,
        }
        write_json(ctx.release_root / "RELEASE_STATUS.json", release_status)
        write_repo_mirror(ctx, deficits)
        checksums_path = write_checksums(ctx.release_root)
        print(f"RELEASE_ID={ctx.release_id}")
        print(f"RELEASE_ROOT={ctx.release_root}")
        print(f"MANIFEST={ctx.release_root / 'MANIFEST.json'}")
        print(f"CHECKSUMS={checksums_path}")
        print("FINAL_STATUS=TRACK_A_SSD_STORAGE_WRITTEN_PENDING_READBACK")
        return 0
    cold_restore = verify_local_release(ctx, deficits)
    destination_rows = [
        {"destination": "github", "status": "PENDING", "detail": "commit_tag_push_not_performed_by_builder", "location": "repo mirror under release_artifacts/"},
        {"destination": "ssd", "status": "FULL_OPERATIONAL_PASS" if cold_restore.get("status") == "PASS" else "PARTIAL", "detail": "release built directly on SSD root", "location": str(ctx.release_root)},
    ]
    r2_status = upload_r2(ctx, deficits)
    destination_rows.append({"destination": "r2", **r2_status})
    write_destination_matrix(ctx.release_root / "DESTINATION_MATRIX.csv", destination_rows)
    readback = {
        "schema_version": "PRISM.destination_readback_validation.v1",
        "verified_at_utc": now_utc(),
        "overall_status": "PARTIAL",
        "destinations": {
            "github": {"status": "PENDING", "detail": "builder does not perform git push/tag"},
            "ssd": {"status": destination_rows[1]["status"], "detail": "cold restore verifier executed against SSD release root"},
            "r2": {"status": r2_status["status"], "detail": r2_status["detail"]},
        },
    }
    write_json(ctx.release_root / "DESTINATION_READBACK_VALIDATION.json", readback)
    restore_source_matrix = {
        "schema_version": "PRISM.restore_source_matrix.v1",
        "github_only_restore": {"status": "PENDING"},
        "ssd_full_restore": {"status": "FULL_OPERATIONAL_PASS" if cold_restore.get("status") == "PASS" else "PARTIAL"},
        "r2_full_restore": {"status": "PARTIAL" if r2_status["status"] == "PARTIAL_PASS" else r2_status["status"]},
    }
    write_json(ctx.release_root / "RESTORE_SOURCE_MATRIX.json", restore_source_matrix)
    policy = evaluate_policy(ctx, deficits)
    final_status = "PARTIAL_PASS"
    if cold_restore.get("status") == "PASS" and policy.get("status") == "PASS" and r2_status["status"] == "PASS":
        final_status = "FULL_PASS"
    elif cold_restore.get("status") == "FAIL":
        final_status = "FAILED"
    write_text(
        ctx.release_root / "RELEASE_DEFICITS.md",
        "# Release Deficits\n\n" + ("\n".join(f"- {item['time_utc']} `{item['step']}`: {item['detail']}" for item in deficits) if deficits else "- None recorded.\n"),
    )
    release_status = {
        "schema_version": "PRISM.sealed_release_status.v1",
        "release_id": ctx.release_id,
        "status": final_status,
        "cold_restore_status": cold_restore.get("status"),
        "policy_status": policy.get("status"),
        "destination_statuses": {row["destination"]: row["status"] for row in destination_rows},
        "git": {"branch": preflight["branch"], "head": preflight["head"], "tag_pushed": False},
    }
    write_json(ctx.release_root / "RELEASE_STATUS.json", release_status)
    write_repo_mirror(ctx, deficits)
    checksums_path = write_checksums(ctx.release_root)
    print(f"RELEASE_ID={ctx.release_id}")
    print(f"RELEASE_ROOT={ctx.release_root}")
    print(f"MANIFEST={ctx.release_root / 'MANIFEST.json'}")
    print(f"CHECKSUMS={checksums_path}")
    print(f"FINAL_STATUS={final_status}")
    return 0 if final_status != "FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
