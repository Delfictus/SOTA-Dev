#!/usr/bin/env python3
"""Safe R2 offload for completed-target raw data + frozen extraction package.

Strict verification per rule §3: remote exists + size matches + streamed
SHA-256 of remote matches local SHA-256. Only after verification is
`safe_to_delete=true` set. Deletion is a SEPARATE phase gated on
`--phase delete`.

Usage:
    python3 r2_offload_completed_targets.py --phase classify
    python3 r2_offload_completed_targets.py --phase upload [--group derivative|small|large|all]
    python3 r2_offload_completed_targets.py --phase delete

Outputs:
    /tmp/engine_full_profiles/r2_sync_manifest.jsonl      (one row per file, append-only)
    /tmp/engine_full_profiles/r2_classification.json      (target/file classification)
    /tmp/engine_full_profiles/r2_upload_log.txt           (per-file progress + verification)
    /tmp/engine_full_profiles/r2_deletion_log.txt         (second phase)
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.client import Config

# ── R2 configuration ──
R2_ENDPOINT = os.environ.get("R2_ENDPOINT",
    "https://0b9ebf4f9a2a36c66302cbb9f32ab1f9.r2.cloudflarestorage.com")
R2_ACCESS = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET = os.environ.get("R2_SECRET_ACCESS_KEY")
BUCKET = "prism-archive"
PREFIX = "m1_1/completed_targets_v1"

OUT = Path("/tmp/engine_full_profiles")
MANIFEST_PATH = OUT / "r2_sync_manifest.jsonl"
CLASSIFICATION_PATH = OUT / "r2_classification.json"
UPLOAD_LOG = OUT / "r2_upload_log.txt"
DELETION_LOG = OUT / "r2_deletion_log.txt"

# Targets classified by completion state
COMPLETE_TARGETS = [
    ("wrn_apo",         "/mnt/storage/prism-outputs/twin-10-patent",      "6yhr"),
    ("menin_apo",       "/mnt/storage/prism-outputs/twin-10-patent",      "3re2"),
    ("smarca2_brd_apo", "/mnt/storage/prism-outputs/twin-10-patent",      "4qy4"),
    ("pkmyt1_apo",      "/mnt/storage/prism-outputs/twin-10-patent",      "3p1a"),
    ("kras_g12d_apo",   "/mnt/storage/prism-outputs/twin-10-patent",      "7f0w"),
    ("m1_2nvp",         "/mnt/storage/prism-outputs/m1-strict-dcc-panel", "2nvp"),
    ("m1_1xhx",         "/mnt/storage/prism-outputs/m1-strict-dcc-panel", "1xhx"),
    ("m1_3bjp",         "/mnt/storage/prism-outputs/m1-strict-dcc-panel", "3bjp"),
    ("m1_2e3k",         "/mnt/storage/prism-outputs/m1-strict-dcc-panel", "2e3k"),
    ("m1_6tyo",         "/mnt/storage/prism-outputs/m1-strict-dcc-panel", "6tyo"),
]
ACTIVE_TARGETS = ["m1_7se6"]
BLOCKED_TARGETS = ["m1_2akr", "m1_1k47", "m1_5yj2", "m1_3umi", "m1_3bl7",
                   "usp1_apo", "polq_apo"]

# File class → (group, size_band)
# group: derivative / small / large
# size_band: used to decide multipart upload threshold (handled by boto3.TransferConfig)
FILE_CLASS_GROUP = {
    "binding_sites.json": "small",
    "kcc_visualization.json": "small",
    "kcc_validation.json": "small",
    "ensemble_trajectory.json": "small",
    "topology.json": "small",
    "residue_map.json": "small",
    "ground_truth": "small",
    "rerank_result.json": "small",
    "evaluation.json": "small",
    "engine_logs": "small",
    "gpu_telemetry": "small",
    "provenance": "small",
    "site_spike_events.json": "large",
    "topology.spike_events.arrow": "large",
}


def _s3_client():
    if not R2_ACCESS or not R2_SECRET:
        raise SystemExit("R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set in env")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS,
        aws_secret_access_key=R2_SECRET,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            read_timeout=600,
            connect_timeout=60,
        ),
    )


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify() -> dict:
    """Walk all directories and classify every file."""
    records = []
    for target, root, stem in COMPLETE_TARGETS:
        td = Path(root) / target / "artifacts"
        if not td.exists():
            continue
        for p in td.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(Path(root) / target)
            cls = classify_file(rel.name, stem)
            records.append({
                "target": target,
                "local_path": str(p),
                "relative_path": str(rel),
                "artifact_class": cls["class"],
                "group": cls["group"],
                "size_bytes": p.stat().st_size,
                "classification": "COMPLETE_RAW_ELIGIBLE",
                "r2_key": f"{PREFIX}/{target}/{rel}",
            })

    for target in ACTIVE_TARGETS:
        records.append({"target": target, "classification": "ACTIVE_DO_NOT_TOUCH",
                        "artifact_class": "N/A", "group": "N/A",
                        "size_bytes": 0, "local_path": "N/A", "r2_key": "N/A"})
    for target in BLOCKED_TARGETS:
        records.append({"target": target, "classification": "BLOCKED",
                        "artifact_class": "N/A", "group": "N/A",
                        "size_bytes": 0, "local_path": "N/A", "r2_key": "N/A"})

    # Derivative package
    dp = OUT
    for p in sorted(dp.rglob("*")):
        if not p.is_file():
            continue
        # skip files that are themselves part of the sync infrastructure
        if p.name in ("r2_sync_manifest.jsonl", "r2_classification.json",
                      "r2_upload_log.txt", "r2_deletion_log.txt"):
            continue
        records.append({
            "target": "__derivative_package__",
            "local_path": str(p),
            "relative_path": str(p.relative_to(dp)),
            "artifact_class": "derivative_package",
            "group": "derivative",
            "size_bytes": p.stat().st_size,
            "classification": "COMPLETE_DERIVATIVE_ELIGIBLE",
            "r2_key": f"{PREFIX}/__derivative__/{p.name}",
        })

    # RETAIN_LOCAL_ONLY: manifests + logs themselves
    for local in [MANIFEST_PATH, CLASSIFICATION_PATH, UPLOAD_LOG, DELETION_LOG]:
        records.append({
            "target": "__manifests_and_logs__",
            "local_path": str(local),
            "classification": "RETAIN_LOCAL_ONLY",
            "artifact_class": "manifest_or_log",
            "group": "local_only",
            "size_bytes": local.stat().st_size if local.exists() else 0,
            "r2_key": "N/A",
        })

    return {"records": records}


def classify_file(name: str, stem: str) -> dict:
    if name.endswith("spike_events.json") and ".site" in name:
        return {"class": "site_spike_events.json", "group": "large"}
    if name.endswith(".topology.spike_events.arrow"):
        return {"class": "topology.spike_events.arrow", "group": "large"}
    if name.endswith(".binding_sites.json"):
        return {"class": "binding_sites.json", "group": "small"}
    if name.endswith(".kcc_visualization.json"):
        return {"class": "kcc_visualization.json", "group": "small"}
    if name.endswith(".kcc_validation.json"):
        return {"class": "kcc_validation.json", "group": "small"}
    if name.endswith(".ensemble_trajectory.json"):
        return {"class": "ensemble_trajectory.json", "group": "small"}
    if name.endswith(".topology.json"):
        return {"class": "topology.json", "group": "small"}
    if name.endswith(".residue_map.json"):
        return {"class": "residue_map.json", "group": "small"}
    if name.endswith("_ground_truth.json"):
        return {"class": "ground_truth", "group": "small"}
    if name == "rerank_result.json":
        return {"class": "rerank_result.json", "group": "small"}
    if name == "evaluation.json":
        return {"class": "evaluation.json", "group": "small"}
    if "engine" in name and (name.endswith(".log") or name.endswith(".stderr.log") or name.endswith(".stdout.log")):
        return {"class": "engine_logs", "group": "small"}
    if name.endswith(".prov.json"):
        return {"class": "provenance", "group": "small"}
    if "gpu_telemetry" in name:
        return {"class": "gpu_telemetry", "group": "small"}
    # Fallback
    if name.endswith(".pdb") or name.endswith(".cif") or name.endswith(".pml") \
       or name.endswith(".md") or name.endswith(".cxc") or name.endswith(".pt") \
       or name.endswith(".bin") or name.endswith(".csv") or name.endswith(".json"):
        return {"class": "auxiliary", "group": "small"}
    return {"class": "auxiliary", "group": "small"}


def _append_manifest(row: dict):
    with MANIFEST_PATH.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def upload_and_verify(s3, rec: dict) -> dict:
    """Upload one file to R2 and verify. Returns annotated record."""
    local = Path(rec["local_path"])
    r2_key = rec["r2_key"]

    t0 = time.time()
    # 1. local SHA-256
    local_hash = sha256_file(local)
    local_size = local.stat().st_size

    # 2. upload
    upload_status = "pending"
    try:
        s3.upload_file(str(local), BUCKET, r2_key)
        upload_status = "uploaded"
    except Exception as e:
        upload_status = f"upload_error:{type(e).__name__}"

    remote_exists = False
    remote_size = None
    stream_hash = None
    safe = False

    if upload_status == "uploaded":
        # 3. HEAD to confirm remote size
        try:
            resp = s3.head_object(Bucket=BUCKET, Key=r2_key)
            remote_exists = True
            remote_size = resp["ContentLength"]
        except Exception as e:
            upload_status = f"head_error:{type(e).__name__}"

    if remote_exists and remote_size == local_size:
        # 4. streamed GET + re-hash (no local duplicate file)
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=r2_key)
            h2 = hashlib.sha256()
            body = resp["Body"]
            for chunk in iter(lambda: body.read(1 << 20), b""):
                h2.update(chunk)
            stream_hash = h2.hexdigest()
            if stream_hash == local_hash:
                safe = True
            else:
                upload_status = "hash_mismatch"
        except Exception as e:
            upload_status = f"get_stream_error:{type(e).__name__}"
    elif remote_exists:
        upload_status = "size_mismatch"

    annotated = {
        **rec,
        "sha256_local": local_hash,
        "r2_bucket": BUCKET,
        "r2_key": r2_key,
        "upload_status": upload_status,
        "remote_exists": remote_exists,
        "remote_size_bytes": remote_size,
        "sha256_stream_verify": stream_hash,
        "safe_to_delete": safe,
        "elapsed_sec": round(time.time() - t0, 2),
        "verification_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_manifest(annotated)
    return annotated


def do_upload(group_filter: str):
    s3 = _s3_client()
    classification = json.loads(CLASSIFICATION_PATH.read_text())
    records = [r for r in classification["records"]
               if r["classification"] in ("COMPLETE_RAW_ELIGIBLE", "COMPLETE_DERIVATIVE_ELIGIBLE")]
    if group_filter != "all":
        records = [r for r in records if r.get("group") == group_filter]
    # Sort: derivative first, then small, then large
    order = {"derivative": 0, "small": 1, "large": 2}
    records.sort(key=lambda r: (order.get(r.get("group"), 9), r.get("size_bytes") or 0))

    # Skip already-verified files (resumability)
    already = set()
    if MANIFEST_PATH.exists():
        for line in MANIFEST_PATH.open():
            try:
                obj = json.loads(line)
                if obj.get("safe_to_delete"):
                    already.add(obj.get("r2_key"))
            except Exception:
                pass

    n_total = len(records)
    n_to_do = sum(1 for r in records if r["r2_key"] not in already)
    n_done = 0
    bytes_done = 0
    bytes_skipped_already = 0

    with UPLOAD_LOG.open("a") as log:
        log.write(f"\n=== upload phase start group={group_filter} @ "
                  f"{datetime.now(timezone.utc).isoformat()} ===\n")
        log.write(f"  n_total={n_total}  n_to_upload={n_to_do}\n")
        for rec in records:
            if rec["r2_key"] in already:
                bytes_skipped_already += rec.get("size_bytes", 0)
                continue
            annotated = upload_and_verify(s3, rec)
            n_done += 1
            bytes_done += rec.get("size_bytes", 0) if annotated["safe_to_delete"] else 0
            safe_flag = "SAFE" if annotated["safe_to_delete"] else "NOT_SAFE"
            log.write(f"  [{n_done:>4}/{n_to_do}] {rec['target']:<20} "
                      f"{Path(rec['local_path']).name[:60]:<60} "
                      f"{rec['size_bytes']:>14}  {annotated['upload_status']:<20}  {safe_flag}\n")
            log.flush()
        log.write(f"=== upload phase end group={group_filter} @ "
                  f"{datetime.now(timezone.utc).isoformat()} "
                  f"done={n_done} bytes_verified={bytes_done} already={bytes_skipped_already} ===\n")

    print(f"group={group_filter}  n_total={n_total}  n_uploaded_this_run={n_done}  "
          f"bytes_verified_this_run={bytes_done}")


def summarize():
    """Walk the current manifest and print per-target + aggregate summary."""
    if not MANIFEST_PATH.exists():
        print("No manifest yet.")
        return {}
    per_target = {}
    totals = {"attempted": 0, "uploaded": 0, "verified_safe": 0, "not_safe": 0,
              "bytes_uploaded": 0, "bytes_safe": 0}
    failures = []
    for line in MANIFEST_PATH.open():
        obj = json.loads(line)
        t = obj.get("target", "?")
        per_target.setdefault(t, {"files": 0, "bytes_uploaded": 0,
                                   "bytes_verified_safe": 0, "failures": []})
        per_target[t]["files"] += 1
        totals["attempted"] += 1
        if obj.get("upload_status") == "uploaded" and obj.get("remote_exists"):
            totals["uploaded"] += 1
            totals["bytes_uploaded"] += obj.get("size_bytes") or 0
            per_target[t]["bytes_uploaded"] += obj.get("size_bytes") or 0
        if obj.get("safe_to_delete"):
            totals["verified_safe"] += 1
            totals["bytes_safe"] += obj.get("size_bytes") or 0
            per_target[t]["bytes_verified_safe"] += obj.get("size_bytes") or 0
        else:
            totals["not_safe"] += 1
            reason = obj.get("upload_status") or "unknown"
            failures.append((t, obj.get("local_path"), reason))
            per_target[t]["failures"].append(reason)
    return {"totals": totals, "per_target": per_target, "failures": failures}


def do_delete():
    """Delete only files marked safe_to_delete=true."""
    if not MANIFEST_PATH.exists():
        print("No manifest — nothing to delete.")
        return
    safe_paths = []
    for line in MANIFEST_PATH.open():
        obj = json.loads(line)
        if obj.get("safe_to_delete") and obj.get("local_path") \
           and Path(obj["local_path"]).exists():
            # Never delete manifest or log files
            if obj.get("classification") == "RETAIN_LOCAL_ONLY":
                continue
            safe_paths.append((obj["local_path"], obj.get("size_bytes") or 0))

    with DELETION_LOG.open("a") as log:
        log.write(f"\n=== deletion phase start @ "
                  f"{datetime.now(timezone.utc).isoformat()} ===\n")
        log.write(f"  n_safe_to_delete={len(safe_paths)}\n")
        deleted = 0
        recovered = 0
        for p, size in safe_paths:
            try:
                Path(p).unlink()
                deleted += 1
                recovered += size
                log.write(f"  DEL {p} ({size} bytes)\n")
            except Exception as e:
                log.write(f"  FAIL {p}: {e}\n")
        log.write(f"=== deletion phase end  deleted={deleted}  "
                  f"recovered_bytes={recovered} ===\n")

    print(f"deleted={deleted}  recovered_bytes={recovered}  "
          f"({recovered/1e9:.2f} GB)")
    # free-space after
    for mp in ("/mnt/storage", "/tmp"):
        try:
            r = subprocess.check_output(["df", "-h", mp], text=True).strip().splitlines()[-1]
            print(f"  {mp}: {r}")
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["classify", "upload", "summary", "delete"], required=True)
    ap.add_argument("--group", choices=["derivative", "small", "large", "all"], default="all")
    args = ap.parse_args()

    if args.phase == "classify":
        cls = classify()
        CLASSIFICATION_PATH.write_text(json.dumps(cls, indent=2, default=str))
        counts = {}
        bytes_by = {}
        for r in cls["records"]:
            c = r.get("classification")
            counts[c] = counts.get(c, 0) + 1
            bytes_by[c] = bytes_by.get(c, 0) + (r.get("size_bytes") or 0)
        print(f"{'classification':<35} {'n_files':>8} {'bytes':>16}")
        for c in sorted(counts):
            print(f"  {c:<33} {counts[c]:>8} {bytes_by[c]:>16,}")
        print(f"classification file: {CLASSIFICATION_PATH}")

    elif args.phase == "upload":
        do_upload(args.group)
        summary = summarize()
        print()
        print(f"totals: {summary['totals']}")

    elif args.phase == "summary":
        summary = summarize()
        print(f"totals: {summary['totals']}")
        print()
        print("per_target:")
        for t, s in sorted(summary["per_target"].items()):
            print(f"  {t:<20} files={s['files']:<5} uploaded={s['bytes_uploaded']:>14}  "
                  f"safe={s['bytes_verified_safe']:>14}  failures={len(s['failures'])}")
        print()
        print("failures:")
        for t, p, reason in summary["failures"][:20]:
            print(f"  {t}  {p}  {reason}")
        if len(summary["failures"]) > 20:
            print(f"  ... +{len(summary['failures']) - 20} more")

    elif args.phase == "delete":
        do_delete()


if __name__ == "__main__":
    main()
