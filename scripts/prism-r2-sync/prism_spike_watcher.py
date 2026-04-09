#!/usr/bin/env python3
"""
PRISM Spike File Watcher Daemon
================================
Watches PRISM engine output directories for completed spike event JSON files,
uploads BOTH raw JSON and lossless Parquet (zstd) to Cloudflare R2, and only
deletes local copies after BOTH are verified on R2.

LIFECYCLE (non-negotiable):
  1. Engine writes spike JSON to local disk
  2. Detect completed JSON (file closed, stable size)
  3. Upload raw JSON to R2 archive bucket
  4. Convert JSON → Parquet (zstd, lossless)
  5. Upload Parquet to R2 archive bucket
  6. Verify BOTH files exist on R2 with matching sizes
  7. ONLY THEN delete local JSON
  8. Keep local Parquet as working copy
  9. If disk critical: delete local Parquet (after R2 verified)

RULE: NO LOCAL FILE IS EVER DELETED UNTIL ITS R2 COUNTERPART IS VERIFIED.

Install deps:
  pip install pyarrow inotify_simple --break-system-packages

Usage:
  # Run as daemon
  sudo systemctl start prism-spike-watcher

  # Run manually (foreground, for testing)
  python3 prism_spike_watcher.py --foreground

  # Dry-run (show what would happen, no uploads or deletes)
  python3 prism_spike_watcher.py --foreground --dry-run

  # Retroactive scan (process existing files that were missed)
  python3 prism_spike_watcher.py --retroactive --foreground
"""

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directories the daemon watches (add more as needed)
WATCH_DIRS = [
    "/mnt/storage/prism-outputs/runs",
    "/mnt/storage/prism-outputs/twin-runs",
]

# Patterns that match spike event files produced by the engine
SPIKE_PATTERNS = [
    r".*\.site\d+\.spike_events\.json$",          # single-stream: {pdb}.site{N}.spike_events.json
    r".*coupled_spike_events\.json$",              # TWIN: coupled_spike_events.json
    r".*\.spike_events\.json$",                    # catch-all for any spike JSON
]

# R2 bucket routing
# Key: regex matching the parent directory path -> (bucket, prefix)
R2_ROUTING = {
    r"/twin-runs/":      ("prism-archive", "twin-runs"),
    r"/cryptobench199/": ("prism-archive", "cryptobench199"),
    r"/v1\.1-physics/":  ("prism-archive", "v1.1-physics"),
    r"/10k-runs/":       ("prism-archive", "10k-runs"),
}
R2_DEFAULT = ("prism-archive", "runs")  # fallback

# rclone settings
RCLONE_REMOTE = "r2"
RCLONE_UPLOAD_FLAGS = [
    "--transfers", "64",
    "--s3-chunk-size", "128M",
    "--s3-upload-concurrency", "32",
    "--no-check-dest",
]

# File stability: wait this many seconds after last modification before processing
STABILITY_WAIT_SECS = 10
# Re-check interval for stability
STABILITY_CHECK_INTERVAL = 2

# Disk space threshold: if free space drops below this, also delete local Parquet
# after R2 verification (emergency mode)
DISK_CRITICAL_GB = 100

# Manifest file: tracks all processed files for audit trail
MANIFEST_PATH = Path("/mnt/storage/prism-outputs/.r2-sync-manifest.jsonl")

# Log file
LOG_PATH = Path("/var/log/prism-spike-watcher.log")

# PID file
PID_PATH = Path("/run/prism-spike-watcher.pid")

# Managed agent webhook (optional — best-effort notification)
WEBHOOK_URL = "http://localhost:8787/upload-complete"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("prism-spike-watcher")


def setup_logging(foreground: bool = False):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = []
    if foreground:
        handlers.append(logging.StreamHandler(sys.stdout))
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(LOG_PATH)))
    except PermissionError:
        # Fall back to user-writable location
        alt = Path.home() / ".local" / "log" / "prism-spike-watcher.log"
        alt.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(alt)))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Managed Agent Webhook (best-effort)
# ---------------------------------------------------------------------------

def _notify_webhook(payload: dict):
    """POST upload notification to managed agent webhook. Best-effort, never blocks."""
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except (urllib.error.URLError, OSError):
        pass  # Webhook not running — that's fine, it's optional


# ---------------------------------------------------------------------------
# R2 Routing
# ---------------------------------------------------------------------------

def get_r2_destination(local_path: str) -> tuple[str, str]:
    """Determine R2 bucket and prefix from local file path."""
    for pattern, (bucket, prefix) in R2_ROUTING.items():
        if re.search(pattern, local_path):
            return bucket, prefix
    return R2_DEFAULT


def get_r2_remote_path(local_path: str) -> str:
    """
    Build the full R2 remote path for a file.

    Local:  /mnt/storage/prism-outputs/runs/cryptobench199/1btl_20260227/1btl.site0.spike_events.json
    R2:     r2:prism-archive/cryptobench199/1btl_20260227/1btl.site0.spike_events.json

    Local:  /mnt/storage/prism-outputs/twin-runs/1a8d/coupled_spike_events.json
    R2:     r2:prism-archive/twin-runs/1a8d/coupled_spike_events.json
    """
    bucket, prefix = get_r2_destination(local_path)
    p = Path(local_path)

    # Find the target directory (parent of the file, e.g., "1btl_20260227")
    target_dir = p.parent.name
    filename = p.name

    return f"{RCLONE_REMOTE}:{bucket}/{prefix}/{target_dir}/{filename}"


def get_r2_dir(local_path: str) -> str:
    """Get the R2 directory (without filename) for uploading."""
    bucket, prefix = get_r2_destination(local_path)
    p = Path(local_path)
    target_dir = p.parent.name
    return f"{RCLONE_REMOTE}:{bucket}/{prefix}/{target_dir}/"


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

def is_spike_file(path: str) -> bool:
    """Check if a file matches spike event patterns."""
    return any(re.match(pat, path) for pat in SPIKE_PATTERNS)


def file_is_stable(path: str, wait_secs: int = STABILITY_WAIT_SECS) -> bool:
    """Wait until file size stops changing (engine finished writing)."""
    try:
        prev_size = os.path.getsize(path)
        time.sleep(wait_secs)
        curr_size = os.path.getsize(path)
        return prev_size == curr_size and curr_size > 0
    except OSError:
        return False


def get_file_size(path: str) -> Optional[int]:
    """Get file size in bytes, or None if not accessible."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def get_disk_free_gb(path: str = "/mnt/storage") -> float:
    """Get free disk space in GB."""
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / (1024 ** 3)


# ---------------------------------------------------------------------------
# rclone operations
# ---------------------------------------------------------------------------

def rclone_upload(local_path: str, r2_dir: str, dry_run: bool = False) -> bool:
    """Upload a file to R2 via rclone. Returns True on success."""
    cmd = [
        "rclone", "copy", local_path, r2_dir,
        *RCLONE_UPLOAD_FLAGS,
    ]
    logger.info(f"UPLOAD: {local_path} → {r2_dir}")
    if dry_run:
        logger.info(f"  [DRY-RUN] Would execute: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            logger.info(f"  UPLOAD OK: {local_path}")
            return True
        else:
            logger.error(f"  UPLOAD FAILED: {local_path} — {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"  UPLOAD TIMEOUT: {local_path} (>1 hour)")
        return False
    except Exception as e:
        logger.error(f"  UPLOAD ERROR: {local_path} — {e}")
        return False


def rclone_verify(r2_path: str, expected_size: int, tolerance: float = 0.01) -> bool:
    """
    Verify a file exists on R2 and size matches within tolerance.
    r2_path: full rclone path like r2:prism-archive/twin-runs/1a8d/file.json
    """
    cmd = ["rclone", "size", r2_path, "--json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.warning(f"  VERIFY FAILED (not found): {r2_path}")
            return False
        data = json.loads(result.stdout)
        r2_size = data.get("bytes", 0)
        if expected_size == 0:
            return r2_size == 0
        size_diff = abs(r2_size - expected_size) / expected_size
        if size_diff <= tolerance:
            logger.info(f"  VERIFY OK: {r2_path} (local={expected_size}, r2={r2_size}, diff={size_diff:.4f})")
            return True
        else:
            logger.warning(f"  VERIFY SIZE MISMATCH: {r2_path} (local={expected_size}, r2={r2_size}, diff={size_diff:.4f})")
            return False
    except Exception as e:
        logger.error(f"  VERIFY ERROR: {r2_path} — {e}")
        return False


def rclone_ls_check(r2_path: str) -> bool:
    """Quick existence check via rclone ls."""
    cmd = ["rclone", "ls", r2_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Parquet conversion
# ---------------------------------------------------------------------------

def convert_json_to_parquet(json_path: str, dry_run: bool = False) -> Optional[str]:
    """
    Convert spike JSON to Parquet with zstd compression.
    Returns path to Parquet file, or None on failure.
    Parquet is 100% lossless — every field preserved.

    Spike JSONs are nested dicts: {centroid, site_id, n_spikes, spikes: [{...}, ...]}.
    We extract the spikes array and flatten it into a typed Parquet table, matching
    the schema from the original convert_spikes.py.
    """
    parquet_path = json_path.replace(".json", ".parquet")

    if os.path.exists(parquet_path):
        logger.info(f"  PARQUET EXISTS: {parquet_path}")
        return parquet_path

    if dry_run:
        logger.info(f"  [DRY-RUN] Would convert: {json_path} → {parquet_path}")
        return parquet_path

    logger.info(f"  CONVERTING: {json_path} → Parquet (zstd)")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Load JSON with stdlib — handles nested dicts correctly
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Handle both formats: nested dict with spikes array, or flat array
        if isinstance(data, dict):
            spikes = data.get("spikes", [])
            site_id = data.get("site_id", 0)
        elif isinstance(data, list):
            spikes = data
            site_id = 0
        else:
            logger.error(f"  UNEXPECTED FORMAT: {json_path} — top-level type: {type(data).__name__}")
            return None

        if not spikes:
            logger.warning(f"  EMPTY SPIKES: {json_path} — 0 spikes, writing empty Parquet")
            # Write empty Parquet with correct schema
            schema = pa.schema([
                ('timestep', pa.int32()), ('frame_index', pa.int32()),
                ('site_id', pa.int32()), ('spike_source', pa.utf8()),
                ('intensity', pa.float32()),
            ])
            pq.write_table(pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema),
                           parquet_path, compression="zstd")
            return parquet_path

        json_rows = len(spikes)

        # Build typed arrays — matches convert_spikes.py schema
        arrays = {
            'timestep': pa.array([s.get('timestep', 0) for s in spikes], type=pa.int32()),
            'frame_index': pa.array([s.get('frame_index', 0) for s in spikes], type=pa.int32()),
            'site_id': pa.array([site_id] * len(spikes), type=pa.int32()),
            'spike_source': pa.array([str(s.get('spike_source', '')) for s in spikes], type=pa.utf8()),
            'ccns_phase': pa.array([str(s.get('ccns_phase', '')) for s in spikes], type=pa.utf8()),
            'intensity': pa.array([float(s.get('intensity', 0)) for s in spikes], type=pa.float32()),
            'vibrational_energy': pa.array([float(s.get('vibrational_energy', 0)) for s in spikes], type=pa.float32()),
            'n_nearby_excited': pa.array([int(s.get('n_nearby_excited', 0)) for s in spikes], type=pa.int16()),
            'stream_id': pa.array([int(s.get('stream_id', 0)) for s in spikes], type=pa.int8()),
            'aromatic_residue_id': pa.array([int(s.get('aromatic_residue_id', 0)) for s in spikes], type=pa.int32()),
            'type': pa.array([str(s.get('type', '')) for s in spikes], type=pa.utf8()),
        }

        # Optional fields — include if present in first spike
        first = spikes[0]
        if 'water_density' in first:
            arrays['water_density'] = pa.array([float(s.get('water_density', 0)) for s in spikes], type=pa.float32())
        if 'x' in first:
            arrays['x'] = pa.array([float(s.get('x', 0)) for s in spikes], type=pa.float32())
            arrays['y'] = pa.array([float(s.get('y', 0)) for s in spikes], type=pa.float32())
            arrays['z'] = pa.array([float(s.get('z', 0)) for s in spikes], type=pa.float32())
        if 'wavelength_nm' in first:
            arrays['wavelength_nm'] = pa.array([float(s.get('wavelength_nm', 0)) for s in spikes], type=pa.float32())

        table = pa.table(arrays)

        # Write with zstd + dictionary encoding on string columns
        pq.write_table(
            table, parquet_path,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )

        # Verify row count
        parquet_table = pq.read_table(parquet_path)
        parquet_rows = parquet_table.num_rows

        if json_rows != parquet_rows:
            logger.error(
                f"  ROW COUNT MISMATCH: JSON={json_rows}, Parquet={parquet_rows} — "
                f"KEEPING BOTH, not deleting anything"
            )
            return None

        json_size = os.path.getsize(json_path)
        parquet_size = os.path.getsize(parquet_path)
        ratio = json_size / parquet_size if parquet_size > 0 else 0

        logger.info(
            f"  PARQUET OK: {parquet_rows} rows, "
            f"{json_size / 1e9:.2f} GB → {parquet_size / 1e6:.1f} MB "
            f"({ratio:.1f}x compression)"
        )
        return parquet_path

    except ImportError:
        logger.error("  pyarrow not installed — run: pip install pyarrow --break-system-packages")
        return None
    except Exception as e:
        logger.error(f"  PARQUET CONVERSION FAILED: {json_path} — {e}")
        return None


# ---------------------------------------------------------------------------
# Manifest / audit trail
# ---------------------------------------------------------------------------

def write_manifest_entry(entry: dict):
    """Append a JSON line to the sync manifest."""
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Manifest write failed: {e}")


# ---------------------------------------------------------------------------
# Core processing pipeline
# ---------------------------------------------------------------------------

def process_spike_file(json_path: str, dry_run: bool = False) -> bool:
    """
    Full lifecycle for a single spike JSON file.
    Returns True if processing completed successfully.

    LIFECYCLE:
      1. Upload raw JSON to R2
      2. Convert to Parquet
      3. Upload Parquet to R2
      4. Verify BOTH on R2 (size match)
      5. Delete local JSON ONLY if both verified
      6. Keep local Parquet
      7. If disk critical: delete local Parquet too (after verification)
    """
    logger.info(f"PROCESSING: {json_path}")
    json_size = get_file_size(json_path)
    if json_size is None or json_size == 0:
        logger.warning(f"  SKIP: File missing or empty: {json_path}")
        return False

    r2_dir = get_r2_dir(json_path)
    r2_json_path = get_r2_remote_path(json_path)

    # ---- Step 1: Upload raw JSON to R2 ----
    json_on_r2 = rclone_ls_check(r2_json_path)
    if json_on_r2:
        logger.info(f"  JSON already on R2: {r2_json_path}")
    else:
        if not rclone_upload(json_path, r2_dir, dry_run=dry_run):
            logger.error(f"  ABORT: JSON upload failed, keeping local file: {json_path}")
            return False

    # ---- Step 2: Convert to Parquet ----
    parquet_path = convert_json_to_parquet(json_path, dry_run=dry_run)
    if parquet_path is None:
        logger.error(f"  ABORT: Parquet conversion failed, keeping local JSON: {json_path}")
        return False

    # ---- Step 3: Upload Parquet to R2 ----
    r2_parquet_path = get_r2_remote_path(parquet_path)
    parquet_on_r2 = rclone_ls_check(r2_parquet_path)
    if parquet_on_r2:
        logger.info(f"  Parquet already on R2: {r2_parquet_path}")
    else:
        parquet_r2_dir = get_r2_dir(parquet_path)
        if not rclone_upload(parquet_path, parquet_r2_dir, dry_run=dry_run):
            logger.error(f"  ABORT: Parquet upload failed, keeping local files: {json_path}")
            return False

    # ---- Step 4: Verify BOTH on R2 ----
    if not dry_run:
        json_verified = rclone_verify(r2_json_path, json_size)
        parquet_size = get_file_size(parquet_path) or 0
        parquet_verified = rclone_verify(r2_parquet_path, parquet_size)

        if not json_verified:
            logger.error(f"  R2 VERIFY FAILED (JSON): {r2_json_path} — NOT deleting local files")
            return False
        if not parquet_verified:
            logger.error(f"  R2 VERIFY FAILED (Parquet): {r2_parquet_path} — NOT deleting local files")
            return False
    else:
        json_verified = True
        parquet_verified = True

    # ---- Step 5: Delete local JSON (ONLY after both verified) ----
    if json_verified and parquet_verified:
        if dry_run:
            logger.info(f"  [DRY-RUN] Would delete local JSON: {json_path}")
        else:
            try:
                os.remove(json_path)
                logger.info(f"  DELETED LOCAL JSON: {json_path} (R2 has both JSON + Parquet)")
            except OSError as e:
                logger.warning(f"  Could not delete local JSON: {json_path} — {e}")
    else:
        logger.error(f"  NOT DELETING: R2 verification incomplete for {json_path}")
        return False

    # ---- Step 6: Check disk pressure, optionally delete local Parquet ----
    disk_free = get_disk_free_gb()
    if disk_free < DISK_CRITICAL_GB:
        logger.warning(
            f"  DISK CRITICAL: {disk_free:.1f} GB free < {DISK_CRITICAL_GB} GB threshold"
        )
        if parquet_verified:
            if dry_run:
                logger.info(f"  [DRY-RUN] Would delete local Parquet (disk critical): {parquet_path}")
            else:
                try:
                    os.remove(parquet_path)
                    logger.info(f"  DELETED LOCAL PARQUET (disk critical): {parquet_path}")
                except OSError as e:
                    logger.warning(f"  Could not delete local Parquet: {parquet_path} — {e}")
    else:
        logger.info(f"  KEEPING local Parquet as working copy ({disk_free:.1f} GB free)")

    # ---- Manifest entry ----
    write_manifest_entry({
        "action": "processed",
        "json_path": json_path,
        "parquet_path": parquet_path,
        "json_size_bytes": json_size,
        "parquet_size_bytes": get_file_size(parquet_path) or 0,
        "r2_json": r2_json_path,
        "r2_parquet": r2_parquet_path,
        "json_verified": json_verified,
        "parquet_verified": parquet_verified,
        "json_deleted_locally": json_verified and parquet_verified,
        "disk_free_gb": round(disk_free, 1),
        "dry_run": dry_run,
    })

    # ---- Notify managed agent webhook (best-effort) ----
    _notify_webhook({
        "r2_path": r2_parquet_path,
        "r2_json_path": r2_json_path,
        "target": Path(json_path).parent.name,
        "size_bytes": get_file_size(parquet_path) or 0,
        "json_size_bytes": json_size,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    logger.info(f"  DONE: {json_path}")
    return True


# ---------------------------------------------------------------------------
# Retroactive scan
# ---------------------------------------------------------------------------

def retroactive_scan(dry_run: bool = False):
    """
    Scan all watch directories for existing spike JSON files that
    haven't been processed yet. Useful for catching up after daemon restart.
    """
    logger.info("=" * 60)
    logger.info("RETROACTIVE SCAN: Checking for unprocessed spike files...")
    logger.info("=" * 60)

    found = 0
    processed = 0

    for watch_dir in WATCH_DIRS:
        if not os.path.isdir(watch_dir):
            logger.info(f"  Skip (not found): {watch_dir}")
            continue

        for root, dirs, files in os.walk(watch_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                if is_spike_file(fpath) and fpath.endswith(".json"):
                    found += 1
                    # Check if already on R2
                    r2_path = get_r2_remote_path(fpath)
                    if rclone_ls_check(r2_path):
                        logger.info(f"  ALREADY ON R2: {fpath}")
                        # Still check if local JSON should be cleaned up
                        parquet_path = fpath.replace(".json", ".parquet")
                        r2_parquet = get_r2_remote_path(parquet_path)
                        if rclone_ls_check(r2_parquet):
                            logger.info(f"  Both on R2, safe to delete local JSON: {fpath}")
                            if not dry_run:
                                try:
                                    os.remove(fpath)
                                    logger.info(f"  DELETED LOCAL JSON: {fpath}")
                                    processed += 1
                                except OSError as e:
                                    logger.warning(f"  Could not delete: {e}")
                        continue
                    else:
                        if process_spike_file(fpath, dry_run=dry_run):
                            processed += 1

    logger.info(f"RETROACTIVE SCAN COMPLETE: {found} spike JSONs found, {processed} processed")


# ---------------------------------------------------------------------------
# inotify watcher
# ---------------------------------------------------------------------------

def watch_directories(dry_run: bool = False):
    """
    Watch directories using inotify for new/modified spike files.
    Falls back to polling if inotify is not available.
    """
    try:
        from inotify_simple import INotify, flags as iflags
        use_inotify = True
    except ImportError:
        logger.warning("inotify_simple not installed — falling back to polling mode")
        use_inotify = False

    if use_inotify:
        _watch_inotify(dry_run)
    else:
        _watch_polling(dry_run)


def _watch_inotify(dry_run: bool = False):
    """inotify-based file watching."""
    from inotify_simple import INotify, flags as iflags

    inotify = INotify()
    wd_map = {}  # watch descriptor -> path

    # Add watches recursively
    for watch_dir in WATCH_DIRS:
        if not os.path.isdir(watch_dir):
            logger.info(f"Creating watch dir: {watch_dir}")
            os.makedirs(watch_dir, exist_ok=True)
        _add_watches_recursive(inotify, watch_dir, wd_map, iflags)

    logger.info(f"Watching {len(wd_map)} directories via inotify")

    # Track files being written (path -> last_size)
    pending = {}

    while True:
        events = inotify.read(timeout=5000)  # 5s timeout for periodic checks

        for event in events:
            parent = wd_map.get(event.wd, "")
            fpath = os.path.join(parent, event.name) if event.name else parent

            # New subdirectory: add watch
            if event.mask & iflags.CREATE and event.mask & iflags.ISDIR:
                _add_watches_recursive(inotify, fpath, wd_map, iflags)
                continue

            # File closed after writing
            if event.mask & iflags.CLOSE_WRITE:
                if is_spike_file(fpath) and fpath.endswith(".json"):
                    pending[fpath] = time.time()

            # File moved into directory
            if event.mask & iflags.MOVED_TO:
                if is_spike_file(fpath) and fpath.endswith(".json"):
                    pending[fpath] = time.time()

        # Process stable pending files
        now = time.time()
        done = []
        for fpath, seen_time in pending.items():
            if now - seen_time >= STABILITY_WAIT_SECS:
                if os.path.exists(fpath) and file_is_stable(fpath, wait_secs=3):
                    process_spike_file(fpath, dry_run=dry_run)
                    done.append(fpath)
                elif not os.path.exists(fpath):
                    done.append(fpath)  # file was removed, skip
        for fpath in done:
            del pending[fpath]


def _add_watches_recursive(inotify, path, wd_map, iflags):
    """Add inotify watches recursively."""
    try:
        mask = iflags.CLOSE_WRITE | iflags.MOVED_TO | iflags.CREATE
        wd = inotify.add_watch(path, mask)
        wd_map[wd] = path
    except Exception as e:
        logger.warning(f"Cannot watch {path}: {e}")
        return

    try:
        for entry in os.scandir(path):
            if entry.is_dir(follow_symlinks=False):
                _add_watches_recursive(inotify, entry.path, wd_map, iflags)
    except PermissionError:
        pass


def _watch_polling(dry_run: bool = False, interval: int = 30):
    """Polling fallback: scan for new spike JSONs every N seconds."""
    known = set()

    # Initial scan
    for watch_dir in WATCH_DIRS:
        if os.path.isdir(watch_dir):
            for root, dirs, files in os.walk(watch_dir):
                for f in files:
                    fpath = os.path.join(root, f)
                    if is_spike_file(fpath) and fpath.endswith(".json"):
                        known.add(fpath)

    logger.info(f"Polling mode: {len(known)} existing spike files, checking every {interval}s")

    while True:
        time.sleep(interval)
        for watch_dir in WATCH_DIRS:
            if not os.path.isdir(watch_dir):
                continue
            for root, dirs, files in os.walk(watch_dir):
                for f in files:
                    fpath = os.path.join(root, f)
                    if is_spike_file(fpath) and fpath.endswith(".json") and fpath not in known:
                        if file_is_stable(fpath):
                            logger.info(f"NEW SPIKE FILE DETECTED: {fpath}")
                            process_spike_file(fpath, dry_run=dry_run)
                            known.add(fpath)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_shutdown = False


def handle_signal(signum, frame):
    global _shutdown
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    _shutdown = True
    sys.exit(0)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PRISM Spike File Watcher Daemon")
    parser.add_argument("--foreground", action="store_true", help="Run in foreground (not as daemon)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen, no uploads/deletes")
    parser.add_argument("--retroactive", action="store_true", help="Scan and process existing files first")
    parser.add_argument("--retroactive-only", action="store_true", help="Only do retroactive scan, then exit")
    parser.add_argument("--watch-dir", action="append", help="Additional directories to watch")
    args = parser.parse_args()

    setup_logging(foreground=args.foreground)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    if args.watch_dir:
        WATCH_DIRS.extend(args.watch_dir)

    logger.info("=" * 60)
    logger.info("PRISM SPIKE FILE WATCHER")
    logger.info(f"  Watch dirs: {WATCH_DIRS}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  Stability wait: {STABILITY_WAIT_SECS}s")
    logger.info(f"  Disk critical threshold: {DISK_CRITICAL_GB} GB")
    logger.info(f"  Manifest: {MANIFEST_PATH}")
    logger.info("=" * 60)

    # Check rclone connectivity
    try:
        result = subprocess.run(
            ["rclone", "lsd", f"{RCLONE_REMOTE}:"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            logger.error(f"rclone connectivity check FAILED: {result.stderr.strip()}")
            logger.error("Fix rclone config before starting the watcher.")
            sys.exit(1)
        logger.info("rclone R2 connectivity: OK")
    except FileNotFoundError:
        logger.error("rclone not found — install it: sudo apt install rclone")
        sys.exit(1)

    # Check pyarrow
    try:
        import pyarrow
        logger.info(f"pyarrow version: {pyarrow.__version__}")
    except ImportError:
        logger.error("pyarrow not installed — run: pip install pyarrow --break-system-packages")
        sys.exit(1)

    # Retroactive scan
    if args.retroactive or args.retroactive_only:
        retroactive_scan(dry_run=args.dry_run)
        if args.retroactive_only:
            return

    # Enter watch loop
    logger.info("Entering watch loop...")
    watch_directories(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
