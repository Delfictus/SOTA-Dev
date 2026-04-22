#!/usr/bin/env python3
"""Enrich r2_sync_manifest.jsonl with failure_reason + emit complete proof.

Reads the running manifest (may still be appended while this runs), derives
failure_reason from existing fields, and writes a canonical deduplicated
manifest plus a proof summary.

Does NOT delete any files. Purely a proof + summary generator.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

OUT = Path("/tmp/engine_full_profiles")
SRC = OUT / "r2_sync_manifest.jsonl"
FINAL = OUT / "r2_sync_manifest.jsonl"      # canonical: overwritten with enriched rows
TMP = OUT / "r2_sync_manifest.enriched.jsonl"
PROOF = OUT / "r2_sync_proof_summary.json"
CLASS_SRC = OUT / "r2_classification.json"


def derive_failure_reason(row: dict) -> str | None:
    if row.get("safe_to_delete"):
        return None
    us = row.get("upload_status") or ""
    if us.startswith("upload_error:"):
        return f"upload_error:{us.split(':',1)[1]}"
    if us.startswith("head_error:"):
        return f"head_error:{us.split(':',1)[1]}"
    if us.startswith("get_stream_error:"):
        return f"get_stream_error:{us.split(':',1)[1]}"
    if us == "hash_mismatch":
        return "sha256_stream_verify_differs_from_sha256_local"
    if us == "size_mismatch":
        return "remote_size_bytes_differs_from_size_bytes"
    if not row.get("remote_exists"):
        return "remote_object_not_present_after_upload"
    if row.get("remote_size_bytes") != row.get("size_bytes"):
        return "remote_size_mismatch"
    if row.get("sha256_stream_verify") != row.get("sha256_local"):
        return "sha256_mismatch"
    return "unknown_not_safe"


def load_rows(path: Path):
    # Dedupe by r2_key: keep the LAST row seen (latest attempt wins)
    rows_by_key = {}
    if not path.exists():
        return []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        key = r.get("r2_key") or r.get("local_path")
        if key is None:
            continue
        rows_by_key[key] = r
    return list(rows_by_key.values())


def main():
    rows = load_rows(SRC)
    # enrich
    for r in rows:
        r["failure_reason"] = derive_failure_reason(r)
    # Write to enriched sidecar (do NOT touch the live-appended source file).
    # The live uploader only appends to r2_sync_manifest.jsonl. When the
    # uploader finishes, re-run this enricher once more to capture the
    # remaining rows into the sidecar. Final canonical form = this file.
    with TMP.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    # ── summary ──
    totals = {
        "attempted": 0,
        "uploaded": 0,
        "verified_safe": 0,
        "not_safe": 0,
        "bytes_uploaded": 0,
        "bytes_verified_safe": 0,
        "bytes_not_safe": 0,
    }
    per_target = defaultdict(lambda: {"files": 0, "uploaded": 0, "verified_safe": 0,
                                        "bytes_uploaded": 0, "bytes_verified_safe": 0,
                                        "failures": []})
    safe_list = []
    not_safe_list = []
    for r in rows:
        totals["attempted"] += 1
        t = r.get("target", "?")
        per_target[t]["files"] += 1
        if r.get("upload_status") == "uploaded" and r.get("remote_exists"):
            totals["uploaded"] += 1
            per_target[t]["uploaded"] += 1
            totals["bytes_uploaded"] += r.get("size_bytes") or 0
            per_target[t]["bytes_uploaded"] += r.get("size_bytes") or 0
        if r.get("safe_to_delete"):
            totals["verified_safe"] += 1
            per_target[t]["verified_safe"] += 1
            totals["bytes_verified_safe"] += r.get("size_bytes") or 0
            per_target[t]["bytes_verified_safe"] += r.get("size_bytes") or 0
            safe_list.append({"target": t, "local_path": r["local_path"],
                              "r2_key": r["r2_key"], "size_bytes": r.get("size_bytes")})
        else:
            totals["not_safe"] += 1
            totals["bytes_not_safe"] += r.get("size_bytes") or 0
            per_target[t]["failures"].append({
                "local_path": r.get("local_path"),
                "failure_reason": r.get("failure_reason"),
            })
            not_safe_list.append({"target": t, "local_path": r.get("local_path"),
                                   "r2_key": r.get("r2_key"), "size_bytes": r.get("size_bytes"),
                                   "failure_reason": r.get("failure_reason")})

    proof = {
        "bucket": "prism-archive",
        "prefix": "m1_1/completed_targets_v1",
        "manifest_path": str(FINAL),
        "totals": totals,
        "per_target": {k: dict(v) for k, v in per_target.items()},
        "safe_to_delete_list": safe_list,
        "not_safe_list": not_safe_list,
    }
    PROOF.write_text(json.dumps(proof, indent=2, default=str))

    print(f"bucket: {proof['bucket']}")
    print(f"prefix: {proof['prefix']}")
    print(f"manifest (live, append-only): {SRC}")
    print(f"manifest (enriched, canonical): {TMP}")
    print(f"proof: {PROOF}")
    print()
    print("=== TOTALS ===")
    for k, v in totals.items():
        print(f"  {k:<22} = {v}")
    print()
    print("=== PER-TARGET verified-safe summary ===")
    for t in sorted(per_target.keys()):
        s = per_target[t]
        print(f"  {t:<22}  files={s['files']:<5}  uploaded={s['uploaded']:<5}  "
              f"safe={s['verified_safe']:<5}  bytes_safe={s['bytes_verified_safe']:>14,}  "
              f"failures={len(s['failures'])}")
    print()
    print(f"=== SAFE_TO_DELETE list: {len(safe_list)} files ===")
    for s in safe_list[:10]:
        print(f"  {s['target']:<22}  {s['local_path']}  ({s['size_bytes']} bytes)")
    if len(safe_list) > 10:
        print(f"  ... +{len(safe_list) - 10} more  (full list in {PROOF})")
    print()
    print(f"=== NOT_SAFE list: {len(not_safe_list)} files ===")
    if not not_safe_list:
        print("  (empty)")
    else:
        for s in not_safe_list[:20]:
            print(f"  {s['target']}  {s['local_path']}  reason={s['failure_reason']}")


if __name__ == "__main__":
    main()
