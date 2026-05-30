#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def blake3_file(path: Path) -> str | None:
    b3sum = shutil.which("b3sum")
    if not b3sum:
        return None
    proc = subprocess.run(
        [b3sum, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.split()[0]


@dataclass
class Inventory:
    regular_files: int
    directories: int
    symlinks: int
    total_bytes: int
    largest_files: list[dict[str, Any]]


def build_inventory(root: Path) -> Inventory:
    regular_files = 0
    directories = 0
    symlinks = 0
    total_bytes = 0
    largest: list[tuple[int, Path]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        directories += 1
        base = Path(dirpath)
        for dirname in dirnames:
            child = base / dirname
            if child.is_symlink():
                symlinks += 1
        for filename in filenames:
            child = base / filename
            if child.is_symlink():
                symlinks += 1
                continue
            stat = child.stat()
            regular_files += 1
            total_bytes += stat.st_size
            largest.append((stat.st_size, child))

    largest.sort(reverse=True, key=lambda item: item[0])
    return Inventory(
        regular_files=regular_files,
        directories=directories,
        symlinks=symlinks,
        total_bytes=total_bytes,
        largest_files=[
            {"path": str(path), "size_bytes": size}
            for size, path in largest[:10]
        ],
    )


def run_checked(
    cmd: list[str],
    cwd: Path | None = None,
    stdout: Any | None = None,
    stderr: Any | None = None,
) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=stdout, stderr=stderr)


def load_requirements(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal canonical PRISM data packs to SSD.")
    parser.add_argument(
        "--seal-root",
        type=Path,
        required=True,
        help="Existing canonical image seal root under /mnt/storage.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo root used to resolve relative pack paths.",
    )
    parser.add_argument(
        "--ssd-root",
        type=Path,
        default=Path("/media/diddy/PRISM-LBS/prism_canonical_seals"),
        help="SSD root for sealed data-pack archives.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        help="zstd compression level for pack archives.",
    )
    parser.add_argument(
        "--pack-id",
        action="append",
        default=[],
        help="Optional pack id filter. May be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seal_root = args.seal_root.resolve()
    repo_root = args.repo_root.resolve()
    requirements_path = seal_root / "verification" / "CANONICAL_DATA_PACK_REQUIREMENTS.json"
    requirements = load_requirements(requirements_path)
    requested_pack_ids = set(args.pack_id)
    control_number = seal_root.name
    ssd_seal_root = args.ssd_root.resolve() / control_number
    data_pack_root = ssd_seal_root / "data_packs"
    metadata_root = data_pack_root / "metadata"
    data_pack_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    overall: dict[str, Any] = {
        "schema_version": "prism.canonical_data_pack_seal.v1",
        "generated_at_utc": utc_now(),
        "control_number": control_number,
        "source_seal_root": str(seal_root),
        "ssd_seal_root": str(ssd_seal_root),
        "repo_root": str(repo_root),
        "requirements_path": str(requirements_path),
        "compression_level": args.compression_level,
        "packs": [],
    }

    zstd_program = f"zstd -T0 -{args.compression_level}"

    for pack in requirements["pending_packs"]:
        pack_id = pack["pack_id"]
        if requested_pack_ids and pack_id not in requested_pack_ids:
            continue
        rel_path = Path(pack["path"])
        source_path = (repo_root / rel_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"missing source path for {pack_id}: {source_path}")

        pack_dir = data_pack_root / pack_id
        pack_dir.mkdir(parents=True, exist_ok=True)
        archive_path = pack_dir / f"{pack_id}.tar.zst"
        inventory_path = pack_dir / "inventory.json"
        receipt_path = pack_dir / "receipt.json"
        print(f"[seal-data-pack] start pack_id={pack_id} source={source_path}")

        inventory = build_inventory(source_path)
        inventory_payload = {
            "generated_at_utc": utc_now(),
            "control_number": control_number,
            "pack_id": pack_id,
            "source_path": str(source_path),
            "relative_path": str(rel_path),
            "required_bytes": int(pack["required_bytes"]),
            "reason": pack["reason"],
            "inventory": {
                "regular_files": inventory.regular_files,
                "directories": inventory.directories,
                "symlinks": inventory.symlinks,
                "total_bytes": inventory.total_bytes,
                "largest_files": inventory.largest_files,
            },
        }
        inventory_path.write_text(json.dumps(inventory_payload, indent=2) + "\n", encoding="utf-8")

        tar_cmd = [
            "tar",
            "--sort=name",
            "--mtime=@0",
            "--numeric-owner",
            "--owner=0",
            "--group=0",
            "--acls",
            "--xattrs",
            f"--use-compress-program={zstd_program}",
            "-cf",
            str(archive_path),
            str(rel_path),
        ]
        run_checked(tar_cmd, cwd=repo_root)

        run_checked(["zstd", "-t", str(archive_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run_checked(
            ["tar", f"--use-compress-program={zstd_program}", "-tf", str(archive_path)],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        archive_size = archive_path.stat().st_size
        archive_sha256 = sha256_file(archive_path)
        archive_blake3 = blake3_file(archive_path)

        receipt = {
            "generated_at_utc": utc_now(),
            "control_number": control_number,
            "pack_id": pack_id,
            "source_path": str(source_path),
            "relative_path": str(rel_path),
            "reason": pack["reason"],
            "required_bytes": int(pack["required_bytes"]),
            "source_inventory": inventory_payload["inventory"],
            "archive_path": str(archive_path),
            "archive_size_bytes": archive_size,
            "archive_sha256": archive_sha256,
            "archive_blake3": archive_blake3,
            "inventory_path": str(inventory_path),
            "verification": {
                "zstd_test": "PASS",
                "tar_list": "PASS",
            },
            "status": "PASS",
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        overall["packs"].append(receipt)
        print(
            "[seal-data-pack] done"
            f" pack_id={pack_id}"
            f" source_bytes={inventory.total_bytes}"
            f" archive_bytes={archive_size}"
            f" archive_sha256={archive_sha256}"
        )

    summary_path = metadata_root / "CANONICAL_DATA_PACK_SEAL_RECEIPT.json"
    summary_md_path = metadata_root / "CANONICAL_DATA_PACK_SEAL_RECEIPT.md"
    summary_path.write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Canonical Data Pack Seal Receipt",
        "",
        f"- generated_at_utc: {overall['generated_at_utc']}",
        f"- control_number: {control_number}",
        f"- source_seal_root: {seal_root}",
        f"- ssd_seal_root: {ssd_seal_root}",
        "",
    ]
    for receipt in overall["packs"]:
        lines.extend(
            [
                f"## {receipt['pack_id']}",
                "",
                f"- source_path: {receipt['source_path']}",
                f"- archive_path: {receipt['archive_path']}",
                f"- source_bytes: {receipt['source_inventory']['total_bytes']}",
                f"- archive_size_bytes: {receipt['archive_size_bytes']}",
                f"- archive_sha256: {receipt['archive_sha256']}",
                f"- archive_blake3: {receipt['archive_blake3']}",
                f"- status: {receipt['status']}",
                "",
            ]
        )
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")

    verification_root = seal_root / "verification"
    verification_root.mkdir(parents=True, exist_ok=True)
    (verification_root / "CANONICAL_DATA_PACK_SEAL_RECEIPT.json").write_text(
        summary_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (verification_root / "CANONICAL_DATA_PACK_SEAL_RECEIPT.md").write_text(
        summary_md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
