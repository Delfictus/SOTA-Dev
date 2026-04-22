#!/usr/bin/env python3
"""Incremental deletion of R2-verified safe files.

Emits per-phase snapshot + appends to r2_deletion_log.txt. Deletes only
rows with safe_to_delete=true that still exist on disk AND are not in
the set of files currently open by the active uploader PID.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/tmp/engine_full_profiles")
MANIFEST = OUT / "r2_sync_manifest.jsonl"
DEL_LOG = OUT / "r2_deletion_log.txt"


def load_manifest():
    rows = {}
    if not MANIFEST.exists():
        return []
    with MANIFEST.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = r.get("r2_key") or r.get("local_path")
            if k:
                rows[k] = r
    return list(rows.values())


def uploader_open_files(pid):
    fd_dir = Path(f"/proc/{pid}/fd")
    openf = set()
    if not fd_dir.exists():
        return openf
    for fd in fd_dir.iterdir():
        try:
            openf.add(os.readlink(fd))
        except Exception:
            pass
    return openf


def already_deleted_paths():
    # Collect paths from prior phase snapshots so we skip them.
    done = set()
    for snap in OUT.glob("phase*_deletion_set.json"):
        try:
            d = json.loads(snap.read_text())
        except Exception:
            continue
        paths = [e["path"] if isinstance(e, dict) else e for e in d.get("safe_paths", []) or d.get("snapshot_paths", [])]
        for p in paths:
            done.add(p)
    return done


def run_phase(phase_name: str, uploader_pid: int = 2008059):
    rows = load_manifest()
    safe = [r for r in rows
            if r.get("safe_to_delete")
            and r.get("classification") != "RETAIN_LOCAL_ONLY"]
    done = already_deleted_paths()
    open_files = uploader_open_files(uploader_pid)
    # eligible this pass: safe + on-disk + not previously enumerated + not open by uploader
    eligible = []
    for r in safe:
        p = Path(r["local_path"])
        if not p.exists():
            continue
        if r["local_path"] in done:
            continue
        if r["local_path"] in open_files:
            continue
        eligible.append(r)

    deleted = 0
    recovered = 0
    with DEL_LOG.open("a") as log:
        log.write(f"\n=== phase-{phase_name} start @ "
                  f"{datetime.now(timezone.utc).isoformat()} ===\n")
        log.write(f"  eligible={len(eligible)}\n")
        for r in eligible:
            p = Path(r["local_path"])
            try:
                size = p.stat().st_size
                p.unlink()
                deleted += 1
                recovered += size
                if deleted <= 3 or deleted % 50 == 0:
                    log.write(f"  DEL {p} ({size} bytes)\n")
            except Exception as ex:
                log.write(f"  FAIL {p}: {ex}\n")
        log.write(f"=== phase-{phase_name} end  deleted={deleted}  bytes={recovered} ===\n")

    snap_path = OUT / f"phase{phase_name}_deletion_set.json"
    snap_path.write_text(json.dumps({
        "phase": phase_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "eligible_count": len(eligible),
        "deleted": deleted,
        "bytes_recovered": recovered,
        "snapshot_paths": [r["local_path"] for r in eligible],
    }, indent=2, default=str))

    print(f"phase-{phase_name}: deleted={deleted}  bytes={recovered:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--uploader-pid", type=int, default=2008059)
    args = ap.parse_args()
    run_phase(args.phase, args.uploader_pid)


if __name__ == "__main__":
    main()
