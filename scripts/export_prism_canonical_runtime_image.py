#!/usr/bin/env python3
"""Export a sealed canonical PRISM runtime image as a portable archive with checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEAL_BASE = Path("/mnt/storage/prism_canonical_seals")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--seal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--compression-level", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blake3_file(path: Path) -> str | None:
    probe = shutil.which("b3sum")
    if probe is None:
        return None
    proc = run([probe, str(path)])
    return proc.stdout.strip().split()[0]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    seal_root = args.seal_root.resolve()
    if not seal_root.exists():
        raise SystemExit(f"missing seal root: {seal_root}")
    if args.compression_level < 1 or args.compression_level > 19:
        raise SystemExit("compression level must be between 1 and 19")

    artifacts_dir = args.output_dir.resolve() if args.output_dir else (seal_root / "artifacts")
    if artifacts_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"artifact dir exists: {artifacts_dir} (use --overwrite)")
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    safe_name = args.image_tag.replace("/", "_").replace(":", "__")
    archive_path = artifacts_dir / f"{safe_name}.docker.tar.zst"
    inspect_path = artifacts_dir / "IMAGE_INSPECT.json"
    export_manifest_path = artifacts_dir / "IMAGE_EXPORT_MANIFEST.json"
    checksums_path = artifacts_dir / "SHA256SUMS"
    load_script_path = artifacts_dir / "DOCKER_LOAD_COMMAND.sh"

    inspect = run(["docker", "image", "inspect", args.image_tag])
    inspect_payload = json.loads(inspect.stdout)[0]
    write_json(inspect_path, inspect_payload)

    export_cmd = (
        "set -euo pipefail; "
        f"docker save {shlex.quote(args.image_tag)} | "
        f"zstd -T0 -{args.compression_level} -o {shlex.quote(str(archive_path))}"
    )
    export_proc = subprocess.run(
        ["bash", "-lc", export_cmd],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    (artifacts_dir / "docker_save.stdout.log").write_text(export_proc.stdout, encoding="utf-8")
    (artifacts_dir / "docker_save.stderr.log").write_text(export_proc.stderr, encoding="utf-8")
    if export_proc.returncode != 0:
        raise SystemExit(f"docker save export failed: {archive_path}")

    archive_sha256 = sha256_file(archive_path)
    archive_blake3 = blake3_file(archive_path)
    checksums_lines = [f"{archive_sha256}  {archive_path.name}"]
    inspect_sha256 = sha256_file(inspect_path)
    checksums_lines.append(f"{inspect_sha256}  {inspect_path.name}")
    checksums_path.write_text("\n".join(checksums_lines) + "\n", encoding="utf-8")

    load_script = "#!/usr/bin/env bash\nset -euo pipefail\n" f"zstd -dc {archive_path.name} | docker load\n"
    load_script_path.write_text(load_script, encoding="utf-8")
    load_script_path.chmod(0o755)

    seal_manifest_path = seal_root / "manifests/IMAGE_SEAL_MANIFEST.json"
    seal_manifest = json.loads(seal_manifest_path.read_text(encoding="utf-8")) if seal_manifest_path.exists() else None
    export_manifest = {
        "schema_version": "prism.canonical_runtime_image_export.v1",
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "image_tag": args.image_tag,
        "image_id": inspect_payload.get("Id"),
        "repo_tags": inspect_payload.get("RepoTags") or [],
        "repo_digests": inspect_payload.get("RepoDigests") or [],
        "size_bytes": inspect_payload.get("Size"),
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "archive_blake3": archive_blake3,
        "inspect_path": str(inspect_path),
        "inspect_sha256": inspect_sha256,
        "docker_load_command": f"zstd -dc {archive_path.name} | docker load",
        "compression_level": args.compression_level,
        "seal_root": str(seal_root),
        "source_image_seal_manifest": str(seal_manifest_path) if seal_manifest_path.exists() else None,
        "source_image_control_number": seal_manifest.get("control_number") if seal_manifest else None,
    }
    write_json(export_manifest_path, export_manifest)

    tag_targets = [
        (archive_path, f"{export_manifest.get('source_image_control_number', 'PRISM_IMAGE')}__archive"),
        (export_manifest_path, f"{export_manifest.get('source_image_control_number', 'PRISM_IMAGE')}__export_manifest"),
        (checksums_path, f"{export_manifest.get('source_image_control_number', 'PRISM_IMAGE')}__export_checksums"),
        (load_script_path, f"{export_manifest.get('source_image_control_number', 'PRISM_IMAGE')}__docker_load"),
    ]
    for target, logical_id in tag_targets:
        run(
            [
                "python3",
                str(REPO_ROOT / "scripts/prism_filetag.py"),
                "tag",
                str(target),
                "--id",
                logical_id,
            ]
        )

    print(
        json.dumps(
            {
                "status": "EXPORTED",
                "image_tag": args.image_tag,
                "archive_path": str(archive_path),
                "archive_sha256": archive_sha256,
                "archive_blake3": archive_blake3,
                "export_manifest": str(export_manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
