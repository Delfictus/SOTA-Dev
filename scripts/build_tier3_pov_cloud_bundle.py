#!/usr/bin/env python3
"""Build a portable cloud bundle from a staged Tier 3 PoV run root."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-ref", default="prism4d/tier3-pov:staged")
    parser.add_argument("--image-export-manifest", type=Path, default=None)
    parser.add_argument("--tag-bundle", action="store_true", default=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        for path in src.rglob("*"):
            if path.is_dir():
                continue
            out = dst / path.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def control_number_for(run_root: Path) -> str:
    status = load_json(run_root / "RUNBOOK_STATUS.json")
    control_number = status.get("control_number")
    if control_number:
        return control_number
    control_file = run_root / "CONTROL_NUMBER.txt"
    if control_file.exists():
        return control_file.read_text(encoding="utf-8").strip()
    raise SystemExit(f"missing control number for staged run root: {run_root}")


def docker_image_record(image_ref: str) -> dict[str, Any]:
    inspect = subprocess.run(
        ["docker", "image", "inspect", image_ref],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if inspect.returncode != 0:
        return {
            "image_ref": image_ref,
            "inspect_available": False,
            "inspect_stderr": inspect.stderr.strip(),
        }
    payload = json.loads(inspect.stdout)[0]
    return {
        "image_ref": image_ref,
        "inspect_available": True,
        "image_id": payload.get("Id"),
        "created": payload.get("Created"),
        "repo_tags": payload.get("RepoTags") or [],
        "repo_digests": payload.get("RepoDigests") or [],
        "size_bytes": payload.get("Size"),
        "architecture": payload.get("Architecture"),
        "os": payload.get("Os"),
        "config": {
            "entrypoint": payload.get("Config", {}).get("Entrypoint"),
            "cmd": payload.get("Config", {}).get("Cmd"),
            "env_count": len(payload.get("Config", {}).get("Env") or []),
            "working_dir": payload.get("Config", {}).get("WorkingDir"),
        },
    }


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.exists():
        raise SystemExit(f"missing json artifact: {resolved}")
    return load_json(resolved)


def update_container_contract(
    bundle_root: Path,
    image_ref: str,
    image_record: dict[str, Any],
    image_export_manifest: dict[str, Any] | None,
) -> None:
    path = bundle_root / "cloud/container_contract.json"
    contract = load_json(path)
    contract["container_image"] = image_ref
    contract["container_image_record"] = image_record
    if image_export_manifest is not None:
        contract["image_export_manifest"] = {
            "path": str(Path("cloud") / "image_export_manifest.json"),
            "archive_path": image_export_manifest.get("archive_path"),
            "archive_sha256": image_export_manifest.get("archive_sha256"),
            "archive_blake3": image_export_manifest.get("archive_blake3"),
            "docker_load_command": image_export_manifest.get("docker_load_command"),
        }
    write_json(path, contract)


def update_bundle_readme(
    bundle_root: Path,
    image_ref: str,
    image_export_manifest: dict[str, Any] | None,
) -> None:
    path = bundle_root / "README.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    insert = [
        "",
        "## Sealed Runtime Image",
        f"- image_ref: `{image_ref}`",
    ]
    if image_export_manifest is not None:
        insert.extend(
            [
                f"- archive_sha256: `{image_export_manifest.get('archive_sha256')}`",
                f"- docker_load: `{image_export_manifest.get('docker_load_command')}`",
            ]
        )
    if "## Sealed Runtime Image" not in lines:
        lines.extend(insert)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pair_id(row: dict[str, Any]) -> str:
    return f"{row['molecule_slot'].lower()}__{row['target_id'].lower()}".replace("/", "_")


def bundle_records(bundle_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        records.append(
            {
                "path": str(path.relative_to(bundle_root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def image_baked_requirements() -> list[dict[str, Any]]:
    paths = [
        REPO_ROOT / "scripts/prism-validate-and-run.sh",
        REPO_ROOT / "scripts/prism-preflight.py",
        REPO_ROOT / "scripts/prism-ground-truth.py",
        REPO_ROOT / "scripts/prism-postflight.py",
        REPO_ROOT / "scripts/build_transition_trajectory_tensor.py",
        REPO_ROOT / "scripts/build_thermodynamic_motif_registry.py",
        REPO_ROOT / "scripts/run_tier3_cloud_worker.py",
        REPO_ROOT / "target/release/nhs_rt_full",
        REPO_ROOT / "target/ptx/protocol_director.ptx",
        REPO_ROOT / "target/ptx/nhs_amber_fused.ptx",
        REPO_ROOT / "target/ptx/housekeeping.ptx",
        REPO_ROOT / "target/ptx/ring_buffer.ptx",
        REPO_ROOT / "target/ptx/twin_persistent.ptx",
        REPO_ROOT / "target/ptx/twin_persistent_physics.ptx",
    ]
    rows = []
    for path in paths:
        rows.append(
            {
                "path": rel(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path) if path.exists() else None,
            }
        )
    return rows


def runtime_payload_specs(control_number: str) -> list[dict[str, Any]]:
    rows = []
    payloads = [
        ("scripts/prism-validate-and-run.sh", "image_payload/scripts/prism-validate-and-run.sh"),
        ("scripts/prism-preflight.py", "image_payload/scripts/prism-preflight.py"),
        ("scripts/prism-ground-truth.py", "image_payload/scripts/prism-ground-truth.py"),
        ("scripts/prism-postflight.py", "image_payload/scripts/prism-postflight.py"),
        ("scripts/build_transition_trajectory_tensor.py", "image_payload/scripts/build_transition_trajectory_tensor.py"),
        ("scripts/build_thermodynamic_motif_registry.py", "image_payload/scripts/build_thermodynamic_motif_registry.py"),
        ("scripts/run_tier3_cloud_worker.py", "image_payload/scripts/run_tier3_cloud_worker.py"),
        ("target/release/nhs_rt_full", "image_payload/bin/nhs_rt_full"),
        ("target/ptx/protocol_director.ptx", "image_payload/ptx/protocol_director.ptx"),
        ("target/ptx/nhs_amber_fused.ptx", "image_payload/ptx/nhs_amber_fused.ptx"),
        ("target/ptx/housekeeping.ptx", "image_payload/ptx/housekeeping.ptx"),
        ("target/ptx/ring_buffer.ptx", "image_payload/ptx/ring_buffer.ptx"),
        ("target/ptx/twin_persistent.ptx", "image_payload/ptx/twin_persistent.ptx"),
        ("target/ptx/twin_persistent_physics.ptx", "image_payload/ptx/twin_persistent_physics.ptx"),
    ]
    for source_rel, bundle_rel in payloads:
        source = REPO_ROOT / source_rel
        logical_name = Path(bundle_rel).name.replace(".", "_").replace("-", "_")
        rows.append(
            {
                "control_number": control_number,
                "source_path": str(source),
                "bundle_path": bundle_rel,
                "logical_id": f"{control_number}__runtime__{logical_name}",
                "exists": source.exists(),
                "size_bytes": source.stat().st_size if source.exists() else None,
                "sha256": sha256(source) if source.exists() else None,
                "mode_octal": oct(source.stat().st_mode & 0o777) if source.exists() else None,
                "executable": bool(source.exists() and os.access(source, os.X_OK)),
            }
        )
    return rows


def runtime_fetch_requirements(loop_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    rows = []
    for row in loop_rows:
        for role, raw in [("topology", row["topology_path"]), ("molecule", row["molecule_path"])]:
            if raw in seen:
                continue
            seen.add(raw)
            unresolved = str(raw).startswith("__")
            path = Path(raw) if not unresolved else None
            rows.append(
                {
                    "role": role,
                    "path": raw,
                    "unresolved_dynamic": unresolved,
                    "exists_locally": bool(path and path.exists()),
                    "size_bytes": path.stat().st_size if path and path.exists() else None,
                    "sha256": sha256(path) if path and path.exists() else None,
                }
            )
    return rows


def write_shards(
    bundle_root: Path,
    control_number: str,
    loop_name: str,
    rows: list[dict[str, Any]],
    pod_rows: list[dict[str, Any]],
) -> None:
    shard_dir = bundle_root / "cloud/shards" / loop_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    for row in pod_rows:
        pod_id = row["pod_id"]
        start = int(row["row_start"])
        end = int(row["row_end_exclusive"])
        payload = {
            "schema_version": "prism.tier3_pov.cloud_shard_manifest.v1",
            "generated_at_utc": now_utc(),
            "control_number": control_number,
            "pod_id": pod_id,
            "loop": loop_name,
            "rows": rows[start:end],
        }
        write_json(shard_dir / f"{pod_id}.manifest.json", payload)


def write_dockerfile(bundle_root: Path, control_number: str) -> None:
    dockerfile = f"""FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
    bash \\
    python3 \\
    python3-pip \\
    jq \\
    ca-certificates \\
    coreutils \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/prism
ENV PRISM_CONTROL_NUMBER={control_number}
COPY image_payload/scripts/ scripts/
COPY image_payload/bin/ target/release/
COPY image_payload/ptx/ target/ptx/
COPY image_payload/CONTROL_NUMBER.txt CONTROL_NUMBER.txt
COPY cloud/worker_entrypoint.sh /usr/local/bin/worker_entrypoint.sh
RUN chmod +x /usr/local/bin/worker_entrypoint.sh scripts/prism-validate-and-run.sh

ENTRYPOINT ["/usr/local/bin/worker_entrypoint.sh"]
"""
    (bundle_root / "cloud/Dockerfile.runpod").write_text(dockerfile, encoding="utf-8")


def write_entrypoint(bundle_root: Path) -> None:
    script = """#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PRISM_SHARD_MANIFEST:-}" ]]; then
  echo "missing PRISM_SHARD_MANIFEST" >&2
  exit 1
fi
if [[ -z "${PRISM_TASK_ROOT:-}" ]]; then
  echo "missing PRISM_TASK_ROOT" >&2
  exit 1
fi

exec python3 scripts/run_tier3_cloud_worker.py \\
  --shard-manifest "${PRISM_SHARD_MANIFEST}" \\
  --task-root "${PRISM_TASK_ROOT}" \\
  --execute \\
  --worker-id "${PRISM_WORKER_ID:-worker-unknown}"
"""
    path = bundle_root / "cloud/worker_entrypoint.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def write_engine_flag_audit(bundle_root: Path) -> None:
    payload = {
        "schema_version": "prism.tier3_pov.engine_flag_conflict_audit.v1",
        "generated_at_utc": now_utc(),
        "conflict_detected": True,
        "sources": {
            "agents_md": "AGENTS.md canonical command block",
            "wrapper_header": "scripts/prism-validate-and-run.sh 2026-05-20 red-flag profile",
        },
        "default_profile_for_cloud_bundle": "wrapper_red_flag_2026_05_20",
        "profiles": {
            "wrapper_red_flag_2026_05_20": {
                "site_ranker": "phase-manifold",
                "requires_md_only_evidence": True,
                "requires_path_a_profile": True,
                "forbids_xgb_ranker": True,
            },
            "agents_md_canonical": {
                "use_xgb_ranker": True,
                "md_only_evidence": "not specified",
                "path_a_profile": "not specified",
            },
        },
        "operator_action_required_before_full_dispatch": True,
    }
    write_json(bundle_root / "verification/engine_flag_conflict_audit.json", payload)
    write_json(bundle_root / "cloud/engine_flag_profiles.json", payload["profiles"])


def copy_runtime_payload(bundle_root: Path, runtime_specs: list[dict[str, Any]], control_number: str) -> None:
    for spec in runtime_specs:
        source = Path(spec["source_path"])
        if not source.exists():
            continue
        destination = bundle_root / spec["bundle_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (bundle_root / "image_payload" / "CONTROL_NUMBER.txt").write_text(
        control_number + "\n",
        encoding="utf-8",
    )


def finalize_runtime_registry(
    bundle_root: Path,
    runtime_specs: list[dict[str, Any]],
    control_number: str,
) -> list[dict[str, Any]]:
    entries = []
    for spec in runtime_specs:
        destination = bundle_root / spec["bundle_path"]
        if not destination.exists():
            entries.append({**spec, "bundled_exists": False})
            continue
        sidecar = Path(str(destination) + ".prismtag")
        if sidecar.exists():
            tag_material = sidecar.read_text(encoding="utf-8", errors="ignore")
        else:
            tag_material = destination.read_text(encoding="utf-8", errors="ignore")[:1024]
        entries.append(
            {
                **spec,
                "bundled_exists": True,
                "bundled_path": str(destination.relative_to(bundle_root)),
                "bundled_sha256": sha256(destination),
                "bundled_size_bytes": destination.stat().st_size,
                "bundled_mode_octal": oct(destination.stat().st_mode & 0o777),
                "bundled_executable": bool(os.access(destination, os.X_OK)),
                "tag_sidecar_exists": sidecar.exists(),
                "control_number_present_in_tag_material": control_number in tag_material,
                "logical_id_present_in_tag_material": spec["logical_id"] in tag_material,
            }
        )
    write_json(
        bundle_root / "verification" / "runtime_executable_registry.json",
        {
            "schema_version": "prism.tier3_pov.runtime_executable_registry.v1",
            "generated_at_utc": now_utc(),
            "control_number": control_number,
            "entry_count": len(entries),
            "entries": entries,
        },
    )
    write_json(
        bundle_root / "verification" / "runtime_executable_tag_audit.json",
        {
            "schema_version": "prism.tier3_pov.runtime_executable_tag_audit.v1",
            "generated_at_utc": now_utc(),
            "control_number": control_number,
            "entry_count": len(entries),
            "missing_bundle_entries": [
                row["bundle_path"] for row in entries if not row.get("bundled_exists")
            ],
            "tag_control_number_failures": [
                row["bundle_path"]
                for row in entries
                if row.get("bundled_exists") and not row.get("control_number_present_in_tag_material")
            ],
            "tag_logical_id_failures": [
                row["bundle_path"]
                for row in entries
                if row.get("bundled_exists") and not row.get("logical_id_present_in_tag_material")
            ],
            "status": (
                "PASS"
                if all(
                    row.get("bundled_exists")
                    and row.get("control_number_present_in_tag_material")
                    and row.get("logical_id_present_in_tag_material")
                    for row in entries
                )
                else "FAIL"
            ),
        },
    )
    return entries


def write_checksums(bundle_root: Path) -> None:
    lines = []
    for row in bundle_records(bundle_root):
        lines.append(f"{row['sha256']}  {row['path']}")
    (bundle_root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tag_all_bundle(bundle_root: Path) -> None:
    tag_cmd = [
        "python3",
        str(REPO_ROOT / "scripts/prism_filetag.py"),
        "tag-all",
        "--root",
        str(bundle_root),
        "--include-hidden",
    ]
    subprocess.run(tag_cmd, cwd=str(REPO_ROOT), check=True, text=True, capture_output=True)


def tag_file(bundle_root: Path, target: Path, logical_id: str) -> None:
    subprocess.run(
        [
            "python3",
            str(REPO_ROOT / "scripts/prism_filetag.py"),
            "tag",
            str(target),
            "--id",
            logical_id,
        ],
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )


def tag_runtime_payload(bundle_root: Path, runtime_registry: list[dict[str, Any]]) -> None:
    for entry in runtime_registry:
        tag_file(bundle_root, bundle_root / entry["bundle_path"], entry["logical_id"])


def enforce_runtime_sidecar_ids(bundle_root: Path, runtime_specs: list[dict[str, Any]]) -> None:
    for entry in runtime_specs:
        destination = bundle_root / entry["bundle_path"]
        sidecar = Path(str(destination) + ".prismtag")
        if not sidecar.exists():
            continue
        text = sidecar.read_text(encoding="utf-8", errors="ignore")
        if entry["logical_id"] in text:
            continue
        sidecar.write_text(
            f"PRISM-TAG:{uuid.uuid4()}:{entry['logical_id']}\n",
            encoding="utf-8",
        )


def snapshot_verify_bundle(bundle_root: Path) -> None:
    snapshot = subprocess.run(
        [
            "python3",
            str(REPO_ROOT / "scripts/prism_filetag.py"),
            "snapshot",
            "--root",
            str(bundle_root),
        ],
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )
    (bundle_root / "filetag_manifest.json").write_text(snapshot.stdout, encoding="utf-8")
    subprocess.run(
        [
            "python3",
            str(REPO_ROOT / "scripts/prism_filetag.py"),
            "verify",
            "--root",
            str(bundle_root),
            "--manifest",
            str(bundle_root / "filetag_manifest.json"),
            "--strict",
        ],
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    control_number = control_number_for(run_root)
    bundle_root = args.output_dir.resolve() if args.output_dir else (run_root / "portable_bundle")
    image_export_manifest = load_optional_json(args.image_export_manifest)
    image_record = docker_image_record(args.image_ref)
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    for name in ["README.md", "RUNBOOK_STATUS.json"]:
        copy_tree(run_root / name, bundle_root / name)
    for subdir in ["loop0", "loop1", "loop2", "loop3", "tracking", "verification", "cloud", "prepared_holo"]:
        if not (run_root / subdir).exists():
            continue
        copy_tree(run_root / subdir, bundle_root / subdir)

    loop2 = load_json(run_root / "loop2/execution_manifest.template.json")
    loop3 = load_json(run_root / "loop3/execution_manifest.template.json")
    holo_registry_path = run_root / "prepared_holo/holo_registry.json"
    holo_map: dict[str, str | None] = {}
    if holo_registry_path.exists():
        holo_registry = load_json(holo_registry_path)
        holo_map = {
            row["pair_id"]: row.get("prepared_holo_topology_path")
            for row in holo_registry.get("prepared", [])
        }
        for loop_rows in (loop2["rows"], loop3["rows"]):
            for row in loop_rows:
                row["prepared_holo_topology_path"] = holo_map.get(pair_id(row))
    else:
        for loop_rows in (loop2["rows"], loop3["rows"]):
            for row in loop_rows:
                row["prepared_holo_topology_path"] = None
    pod_rows = []
    with (run_root / "cloud/pod_assignment_plan.csv").open("r", encoding="utf-8") as handle:
        pod_rows = list(csv.DictReader(handle))
    write_shards(
        bundle_root, control_number, "loop2", loop2["rows"], [r for r in pod_rows if r["loop"] == "loop2"]
    )
    write_shards(
        bundle_root, control_number, "loop3", loop3["rows"], [r for r in pod_rows if r["loop"] == "loop3"]
    )
    write_entrypoint(bundle_root)
    write_dockerfile(bundle_root, control_number)
    write_engine_flag_audit(bundle_root)
    update_container_contract(bundle_root, args.image_ref, image_record, image_export_manifest)
    if image_export_manifest is not None:
        write_json(bundle_root / "cloud/image_export_manifest.json", image_export_manifest)
    update_bundle_readme(bundle_root, args.image_ref, image_export_manifest)
    runtime_specs = runtime_payload_specs(control_number)
    copy_runtime_payload(bundle_root, runtime_specs, control_number)
    image_reqs = image_baked_requirements()
    fetch_reqs = runtime_fetch_requirements(loop2["rows"] + loop3["rows"])
    if args.tag_bundle:
        tag_runtime_payload(bundle_root, runtime_specs)
        tag_all_bundle(bundle_root)
        enforce_runtime_sidecar_ids(bundle_root, runtime_specs)
    runtime_registry = finalize_runtime_registry(bundle_root, runtime_specs, control_number)
    bundle_manifest = {
        "schema_version": "prism.tier3_pov.portable_bundle_manifest.v1",
        "generated_at_utc": now_utc(),
        "control_number": control_number,
        "run_root": str(run_root),
        "bundle_root": str(bundle_root),
        "container_image": args.image_ref,
        "container_image_record": image_record,
        "image_export_manifest": image_export_manifest,
        "image_baked_requirements": image_reqs,
        "embedded_runtime_payload": runtime_registry,
        "runtime_fetch_requirements": fetch_reqs,
    }
    write_json(bundle_root / "bundle_manifest.json", bundle_manifest)
    completeness = {
        "schema_version": "prism.tier3_pov.bundle_completeness_report.v1",
        "generated_at_utc": now_utc(),
        "control_number": control_number,
        "image_baked_missing_count": sum(1 for row in image_reqs if not row["exists"]),
        "embedded_runtime_payload_count": len(runtime_registry),
        "embedded_runtime_payload_missing_count": sum(1 for row in runtime_specs if not row["exists"]),
        "runtime_fetch_unresolved_dynamic_count": sum(
            1 for row in fetch_reqs if row["unresolved_dynamic"]
        ),
    }
    write_json(bundle_root / "verification/bundle_completeness_report.json", completeness)
    write_json(
        bundle_root / "verification/bundle_control_exceptions.json",
        {
            "schema_version": "prism.tier3_pov.bundle_control_exceptions.v1",
            "generated_at_utc": now_utc(),
            "untaggable_recursive_outputs": ["filetag_manifest.json"],
        },
    )
    write_checksums(bundle_root)
    if args.tag_bundle:
        tag_file(bundle_root, bundle_root / "CHECKSUMS.sha256", "bundle_checksums_sha256")
    completeness["bundle_file_count"] = len(bundle_records(bundle_root))
    write_json(bundle_root / "verification/bundle_completeness_report.json", completeness)
    if args.tag_bundle:
        tag_file(
            bundle_root,
            bundle_root / "bundle_manifest.json",
            f"{control_number}__bundle__manifest",
        )
        tag_file(
            bundle_root,
            bundle_root / "verification/bundle_completeness_report.json",
            "bundle_completeness_report",
        )
        tag_file(
            bundle_root,
            bundle_root / "verification/runtime_executable_registry.json",
            f"{control_number}__runtime__registry",
        )
        tag_file(
            bundle_root,
            bundle_root / "verification/runtime_executable_tag_audit.json",
            f"{control_number}__runtime__tag_audit",
        )
        tag_file(
            bundle_root,
            bundle_root / "verification/bundle_control_exceptions.json",
            f"{control_number}__bundle__control_exceptions",
        )
        snapshot_verify_bundle(bundle_root)
    print(
        json.dumps(
            {
                "control_number": control_number,
                "bundle_root": str(bundle_root),
                "bundle_file_count": len(bundle_records(bundle_root)),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
