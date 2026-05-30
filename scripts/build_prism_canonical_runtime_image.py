#!/usr/bin/env python3
"""Stage and optionally build the sealed canonical PRISM runtime image."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEAL_BASE = Path("/mnt/storage/prism_canonical_seals")
DEFAULT_ENV_ROOT = Path("/mnt/storage/prism_env_copies/prism_dock_portable_20260529")
DEFAULT_SCRATCH_ROOT = Path("/mnt/storage/prism-scratch/Prism4D-bio")
DEFAULT_CANDIDATE_SMOKE = Path("/mnt/storage/tmp/glp1r_candidate_md_smoke_20260527_233437")
DEFAULT_VERIFIER_REPORT = Path(
    "/mnt/storage/tmp/prism_canonical_runtime_verification_20260529T212225Z_90ffae25ed5f"
)
DEFAULT_TIER3_RUN_ROOT = REPO_ROOT / "campaigns/glp1r_aleniglipron/tier3_pov/staged_runs/tier3-pov-20260529T184259Z"
DEFAULT_ALENI_HOLO = Path(
    "/mnt/storage/root-pressure-relief/20260527T041800Z/srv-prism-root-capacity-storage-tmp/epoch023_replay/04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json"
)
DEFAULT_HOST_CUDA_LIB_DIR = Path("/usr/local/cuda/targets/x86_64-linux/lib")
DEFAULT_HOST_SDST_LIB_DIR = Path("/opt/prism4d/lib")


REPO_COPY_PATHS = [
    "04_TOPOLOGIES",
    "Cargo.toml",
    "Cargo.lock",
    "Makefile",
    "manifests",
    "pyproject.toml",
    "setup.py",
    "docs/PRISM_CANONICAL_RUNTIME.md",
    "docs/PRISM_GLP1R_TIER3_AUTHORITY_FIRST_POV_MASTER_DIRECTIVE.md",
    "docs/PRISM_GLP1R_TIER3_CLOUD_RUNBOOK.md",
    "scripts",
    "src",
    "crates",
    "vendor",
    "tests",
    "target/release",
    "target/ptx",
    "campaigns/glp1r_aleniglipron/M3_Lead_Optimization_Dossier.md",
    "campaigns/glp1r_aleniglipron/PHASE2_PHASE3_DELIVERABLES_AUDIT_20260528.md",
    "campaigns/glp1r_aleniglipron/PHASE2_PHASE3_FULL_OUTPUT_COMPLETENESS_QUALITY_20260528.md",
    "campaigns/glp1r_aleniglipron/PHASE2_PHASE3_FULL_OUTPUT_COMPLETENESS_QUALITY_20260528.json",
    "campaigns/glp1r_aleniglipron/candidate_dossiers",
    "campaigns/glp1r_aleniglipron/track_0_manual_emulation",
    "campaigns/glp1r_aleniglipron/topology",
    "campaigns/glp1r_aleniglipron/dstw_phase_b",
    "campaigns/glp1r_aleniglipron/tier3_pov/staged_runs/tier3-pov-20260529T184259Z",
    "campaigns/glp1r_aleniglipron/topologies",
    "campaigns/glp1r_aleniglipron/track_a_generative",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_phase_manifold_smoke_fixture",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/interferometric_differential.parquet",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/phase_manifold_coherence.parquet",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/residue_phase_tensor.parquet",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/shear_stress_field.parquet",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/translation_pathway_nodes.parquet",
]

OPTIONAL_REPO_COPY_PATHS = [
    "campaigns/glp1r_aleniglipron/track_b_chronological/expanded_variant_run",
]


@dataclass(frozen=True)
class InventoryRow:
    control_number: str
    relative_path: str
    file_type: str
    size_bytes: int | None
    mode: str
    symlink_target: str | None
    sha256: str | None


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def git_head_short() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short=12", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "nogit"


def default_control_number() -> str:
    return f"PRISM-CANONICAL-IMAGE-{utc_stamp()}-{git_head_short()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-number", default=default_control_number())
    parser.add_argument("--seal-base", type=Path, default=DEFAULT_SEAL_BASE)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--candidate-smoke", type=Path, default=DEFAULT_CANDIDATE_SMOKE)
    parser.add_argument("--verifier-report", type=Path, default=DEFAULT_VERIFIER_REPORT)
    parser.add_argument("--tier3-run-root", type=Path, default=DEFAULT_TIER3_RUN_ROOT)
    parser.add_argument("--aleni-holo-topology", type=Path, default=DEFAULT_ALENI_HOLO)
    parser.add_argument("--image-tag", default=None)
    parser.add_argument("--include-expanded-variant-run", action="store_true")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.lstat().st_size if file_path.is_symlink() else file_path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def copy_any(src: Path, dst: Path) -> None:
    if src.is_dir():
        dst.parent.mkdir(parents=True, exist_ok=True)
        run(["rsync", "-aH", "--delete", f"{src}/", f"{dst}/"])
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_path(path: Path) -> tuple[str, int | None, str | None]:
    stat = path.lstat()
    mode = oct(stat.st_mode & 0o777)
    if path.is_symlink():
        return "symlink", None, os.readlink(path)
    if path.is_dir():
        return "directory", None, None
    if path.is_file():
        return "file", stat.st_size, None
    return "other", None, None


def build_inventory(root: Path, control_number: str) -> list[InventoryRow]:
    paths = sorted(root.rglob("*"))
    file_paths = [path for path in paths if path.is_file() and not path.is_symlink()]
    hashes: dict[Path, str] = {}

    with ThreadPoolExecutor(max_workers=max(os.cpu_count() or 2, 4)) as pool:
        for path, digest in zip(file_paths, pool.map(sha256_file, file_paths)):
            hashes[path] = digest

    rows: list[InventoryRow] = []
    for path in paths:
        file_type, size_bytes, symlink_target = classify_path(path)
        rows.append(
            InventoryRow(
                control_number=control_number,
                relative_path=path.relative_to(root).as_posix(),
                file_type=file_type,
                size_bytes=size_bytes,
                mode=oct(path.lstat().st_mode & 0o777),
                symlink_target=symlink_target,
                sha256=hashes.get(path),
            )
        )
    return rows


def write_inventory(rows: Iterable[InventoryRow], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, sort_keys=True) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "control_number",
                "relative_path",
                "file_type",
                "size_bytes",
                "mode",
                "symlink_target",
                "sha256",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_sha256sums(root: Path, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if path == output_path:
                continue
            handle.write(f"{sha256_file(path)}  {path.relative_to(root)}\n")


def dockerfile_text(control_number: str) -> str:
    return f"""FROM nvidia/cuda:12.6.0-base-ubuntu24.04

LABEL org.opencontainers.image.title="PRISM canonical runtime"
LABEL org.opencontainers.image.description="Sealed canonical PRISM runtime for the verified GLP-1R aleniglipron lane"
LABEL org.opencontainers.image.version="{control_number}"
LABEL org.opencontainers.image.revision="{git_head_short()}"

ENV PRISM_CONTROL_NUMBER={control_number}
ENV PRISM_DOCK_ENV=/mnt/storage/prism_env_copies/prism_dock_portable_20260529
ENV PRISM_SCRATCH_ROOT=/mnt/storage/prism-scratch/Prism4D-bio
ENV PRISM_CANONICAL_RUNTIME_STRICT=1
ENV LD_LIBRARY_PATH=/opt/prism4d/lib:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
ENV PATH=/mnt/storage/prism_env_copies/prism_dock_portable_20260529/bin:$PATH
ENV PYTHONPATH=/home/diddy/Desktop/Prism4D-bio:/home/diddy/Desktop/Prism4D-bio/src

COPY rootfs/ /

WORKDIR /home/diddy/Desktop/Prism4D-bio

RUN chmod +x \
    /home/diddy/Desktop/Prism4D-bio/scripts/prism-validate-and-run.sh \
    /home/diddy/Desktop/Prism4D-bio/scripts/prism-prep \
    /home/diddy/Desktop/Prism4D-bio/scripts/prism-canonical-env.sh

ENTRYPOINT ["/bin/bash"]
"""


def verification_command(image_tag: str) -> str:
    return (
        "docker run --rm --gpus all -v /mnt/storage/tmp:/mnt/storage/tmp "
        f"{image_tag} "
        "-lc 'cd /home/diddy/Desktop/Prism4D-bio && "
        "source scripts/prism-canonical-env.sh >/tmp/prism-canonical-env.log && "
        "PYTHONPATH=src python3 scripts/verify_prism_canonical_runtime.py "
        "--report-root /mnt/storage/tmp/prism_container_runtime_verification'"
    )


def build_engine_smoke_command(image_tag: str) -> str:
    return (
        "docker run --rm --gpus all -v /mnt/storage/tmp:/mnt/storage/tmp "
        f"{image_tag} "
        "-lc 'cd /home/diddy/Desktop/Prism4D-bio && "
        "source scripts/prism-canonical-env.sh >/tmp/prism-canonical-env.log && "
        "scripts/prism-validate-and-run.sh "
        "-t /mnt/storage/root-pressure-relief/20260527T041800Z/srv-prism-root-capacity-storage-tmp/epoch023_replay/04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json "
        "-o /mnt/storage/tmp/prism_container_engine_smoke "
        "--fast --hysteresis --prism-therm "
        "--multi-stream 8 "
        "--spike-percentile 70 "
        "--fused-steps 6 "
        "--hmr --adaptive-dt "
        "--multi-differential "
        "--closed-loop-steering --asymmetric-steering "
        "--site-ranker phase-manifold "
        "--md-only-evidence "
        "--path-a-production-profile "
        "--path-a-max-wall-seconds 120 "
        "--uv-wavelengths 280,274,258,254,211 "
        "--nma-amplification 3.0 --nma-scan-fraction 0.3 "
        "--replica-seed 42 -v'"
    )


def main() -> int:
    args = parse_args()
    control_number = args.control_number
    image_tag = args.image_tag or f"prism-canonical-runtime:{control_number.lower()}"
    seal_root = args.seal_base / control_number
    build_context = seal_root / "build_context"
    rootfs = build_context / "rootfs"
    manifests_dir = seal_root / "manifests"
    logs_dir = seal_root / "logs"
    verification_dir = seal_root / "verification"

    if seal_root.exists():
        if not args.overwrite:
            raise SystemExit(f"seal root exists: {seal_root} (use --overwrite)")
        shutil.rmtree(seal_root)

    for path in [build_context, rootfs, manifests_dir, logs_dir, verification_dir]:
        path.mkdir(parents=True, exist_ok=True)

    repo_target_root = rootfs / "home/diddy/Desktop/Prism4D-bio"
    repo_target_root.mkdir(parents=True, exist_ok=True)

    copy_plan: list[dict[str, Any]] = []
    for rel in REPO_COPY_PATHS:
        src = REPO_ROOT / rel
        if not src.exists():
            continue
        dst = repo_target_root / rel
        copy_any(src, dst)
        copy_plan.append({"src": str(src), "dst": str(dst), "required": True, "bytes": dir_size(src) if src.is_dir() else src.stat().st_size})

    if args.include_expanded_variant_run:
        for rel in OPTIONAL_REPO_COPY_PATHS:
            src = REPO_ROOT / rel
            if not src.exists():
                continue
            dst = repo_target_root / rel
            copy_any(src, dst)
            copy_plan.append(
                {"src": str(src), "dst": str(dst), "required": False, "bytes": dir_size(src) if src.is_dir() else src.stat().st_size}
            )

    absolute_copy_pairs = [
        (args.env_root.resolve(), rootfs / args.env_root.resolve().relative_to("/")),
        (args.candidate_smoke.resolve(), rootfs / args.candidate_smoke.resolve().relative_to("/")),
        (args.verifier_report.resolve(), rootfs / args.verifier_report.resolve().relative_to("/")),
        (args.tier3_run_root.resolve(), rootfs / args.tier3_run_root.resolve().relative_to("/")),
        (args.aleni_holo_topology.resolve(), rootfs / args.aleni_holo_topology.resolve().relative_to("/")),
    ]
    env_manifest = args.env_root.resolve().with_suffix(".manifest.json")
    if env_manifest.exists():
        absolute_copy_pairs.append((env_manifest, rootfs / env_manifest.relative_to("/")))
    env_relocation = args.env_root.resolve().with_suffix(".relocation.json")
    if env_relocation.exists():
        absolute_copy_pairs.append((env_relocation, rootfs / env_relocation.relative_to("/")))

    host_cuda_pairs: list[tuple[Path, Path]] = []
    for cudart in sorted(DEFAULT_HOST_CUDA_LIB_DIR.glob("libcudart.so.13*")):
        host_cuda_pairs.append((cudart.resolve() if not cudart.is_symlink() else cudart, rootfs / cudart.relative_to("/")))
    if not host_cuda_pairs:
        raise SystemExit(f"required host CUDA runtime not found under {DEFAULT_HOST_CUDA_LIB_DIR}")
    absolute_copy_pairs.extend(host_cuda_pairs)

    host_sdst_pairs: list[tuple[Path, Path]] = []
    for sdst in sorted(DEFAULT_HOST_SDST_LIB_DIR.glob("libsdst.so*")):
        host_sdst_pairs.append((sdst.resolve() if not sdst.is_symlink() else sdst, rootfs / sdst.relative_to("/")))
    if not host_sdst_pairs:
        raise SystemExit(f"required host sdst runtime not found under {DEFAULT_HOST_SDST_LIB_DIR}")
    absolute_copy_pairs.extend(host_sdst_pairs)

    for src, dst in absolute_copy_pairs:
        if not src.exists():
            raise SystemExit(f"required source missing: {src}")
        copy_any(src, dst)
        copy_plan.append({"src": str(src), "dst": str(dst), "required": True, "bytes": dir_size(src) if src.is_dir() else src.stat().st_size})

    # Seed writable scratch roots inside the image.
    for scratch_path in [
        rootfs / args.scratch_root.resolve().relative_to("/"),
        rootfs / "mnt/storage/tmp",
        rootfs / "mnt/storage/prism_canonical_tmp",
    ]:
        scratch_path.mkdir(parents=True, exist_ok=True)
        (scratch_path / ".keep").write_text(control_number + "\n", encoding="utf-8")

    dockerfile_path = build_context / "Dockerfile"
    dockerfile_path.write_text(dockerfile_text(control_number), encoding="utf-8")
    (build_context / ".dockerignore").write_text("", encoding="utf-8")

    build_script = seal_root / "DOCKER_BUILD_COMMAND.sh"
    build_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"docker build -t {image_tag} {build_context}\n",
        encoding="utf-8",
    )
    build_script.chmod(0o755)

    verify_script = seal_root / "DOCKER_VERIFY_COMMAND.sh"
    verify_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + verification_command(image_tag) + "\n",
        encoding="utf-8",
    )
    verify_script.chmod(0o755)

    engine_smoke_script = seal_root / "DOCKER_ENGINE_SMOKE_COMMAND.sh"
    engine_smoke_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + build_engine_smoke_command(image_tag) + "\n",
        encoding="utf-8",
    )
    engine_smoke_script.chmod(0o755)

    inventory_rows = build_inventory(rootfs, control_number)
    write_inventory(inventory_rows, manifests_dir / "ROOTFS_INVENTORY.jsonl", manifests_dir / "ROOTFS_INVENTORY.csv")
    write_sha256sums(build_context, manifests_dir / "SHA256SUMS")

    df_root = shutil.disk_usage(REPO_ROOT)
    df_storage = shutil.disk_usage(args.seal_base)
    manifest = {
        "schema_version": "prism.canonical_runtime_image_seal.v1",
        "control_number": control_number,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_head": git_head_short(),
        "repo_root": str(REPO_ROOT),
        "image_tag": image_tag,
        "build_context": str(build_context),
        "rootfs": str(rootfs),
        "copy_plan": copy_plan,
        "context_bytes": dir_size(build_context),
        "rootfs_bytes": dir_size(rootfs),
        "seal_root_bytes": dir_size(seal_root),
        "free_bytes_root_fs": df_root.free,
        "free_bytes_storage_fs": df_storage.free,
        "env_root": str(args.env_root.resolve()),
        "candidate_smoke": str(args.candidate_smoke.resolve()),
        "verifier_report": str(args.verifier_report.resolve()),
        "tier3_run_root": str(args.tier3_run_root.resolve()),
        "aleni_holo_topology": str(args.aleni_holo_topology.resolve()),
        "include_expanded_variant_run": args.include_expanded_variant_run,
        "docker_build_command": build_script.read_text(encoding="utf-8").strip().splitlines()[-1],
        "docker_verify_command": verify_script.read_text(encoding="utf-8").strip().splitlines()[-1],
        "docker_engine_smoke_command": engine_smoke_script.read_text(encoding="utf-8").strip().splitlines()[-1],
        "inventory_records": len(inventory_rows),
    }
    manifest_path = manifests_dir / "IMAGE_SEAL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.build_image:
        build = run(["docker", "build", "-t", image_tag, str(build_context)], check=False)
        (logs_dir / "docker_build.stdout.log").write_text(build.stdout, encoding="utf-8")
        (logs_dir / "docker_build.stderr.log").write_text(build.stderr, encoding="utf-8")
        manifest["docker_build"] = {
            "returncode": build.returncode,
            "stdout_log": str(logs_dir / "docker_build.stdout.log"),
            "stderr_log": str(logs_dir / "docker_build.stderr.log"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if build.returncode != 0:
            print(json.dumps({"status": "BUILD_FAIL", "seal_root": str(seal_root), "image_tag": image_tag}, indent=2))
            return build.returncode

    print(
        json.dumps(
            {
                "status": "STAGED" if not args.build_image else "BUILT",
                "control_number": control_number,
                "image_tag": image_tag,
                "seal_root": str(seal_root),
                "context_bytes": manifest["context_bytes"],
                "inventory_records": manifest["inventory_records"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
