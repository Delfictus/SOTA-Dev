#!/usr/bin/env python3
"""Fetch a bulk chemical representation table for fragment curation.

This intentionally downloads a single compressed bulk file from the ChEMBL FTP
mirror. It does not call molecule REST endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"
    "chembl_36_chemreps.txt.gz"
)
DEFAULT_CHECKSUMS_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/checksums.txt"
DEFAULT_OUTPUT_DIR = Path("/home/diddy/prism4d_analysis/library")
CHUNK_SIZE = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Bulk compressed source URL.")
    parser.add_argument("--checksums-url", default=DEFAULT_CHECKSUMS_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Redownload even if the file exists.")
    return parser.parse_args()


def filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"Could not infer filename from URL: {url}")
    return name


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fetch_expected_sha256(checksums_url: str, filename: str) -> str | None:
    if not checksums_url:
        return None
    with urlopen(checksums_url, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == filename:
            return parts[0].lower()
    return None


def remote_size(url: str) -> int | None:
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=60) as response:
        size = response.headers.get("Content-Length")
    return int(size) if size else None


def stream_download(url: str, destination: Path, force: bool = False) -> None:
    part = destination.with_suffix(destination.suffix + ".part")
    if force:
        destination.unlink(missing_ok=True)
        part.unlink(missing_ok=True)

    expected_size = remote_size(url)
    if destination.exists() and expected_size is not None and destination.stat().st_size == expected_size:
        print(f"download_exists\t{destination}\t{destination.stat().st_size}")
        return

    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "Prism4D-bulk-fragment-fetch/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = Request(url, headers=headers)
    mode = "ab" if offset else "wb"
    try:
        response = urlopen(request, timeout=120)
    except HTTPError as exc:
        if offset and exc.code == 416:
            part.rename(destination)
            return
        raise

    if offset and response.status != 206:
        offset = 0
        mode = "wb"

    total = expected_size or 0
    downloaded = offset
    last_report = time.time()
    with response, part.open(mode) as handle:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_report >= 10:
                if total:
                    pct = downloaded / total * 100.0
                    print(f"download_progress\t{downloaded}\t{total}\t{pct:.1f}%")
                else:
                    print(f"download_progress\t{downloaded}")
                last_report = now
    part.rename(destination)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = filename_from_url(args.url)
    output_path = args.output_dir / filename

    expected_sha256 = fetch_expected_sha256(args.checksums_url, filename)
    if output_path.exists() and not args.force and expected_sha256:
        observed = sha256_path(output_path)
        if observed == expected_sha256:
            print(f"verified_existing\t{output_path}\tsha256={observed}")
            return 0

    stream_download(args.url, output_path, force=args.force)

    if expected_sha256:
        observed = sha256_path(output_path)
        if observed != expected_sha256:
            print(
                f"checksum_mismatch\t{output_path}\texpected={expected_sha256}\tobserved={observed}",
                file=sys.stderr,
            )
            return 2
        print(f"verified_download\t{output_path}\tsha256={observed}")
    else:
        print(f"downloaded_no_checksum\t{output_path}")

    print(f"source_url\t{args.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
