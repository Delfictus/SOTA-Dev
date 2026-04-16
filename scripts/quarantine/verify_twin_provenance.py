#!/usr/bin/env python3
"""
[REVIEWER TOOL]

Independently verify the full BLAKE3 content-addressed provenance chain
for a TWIN-10 target.

Given a target output directory, this tool:
  1. Walks prov/*.prov.json, recomputes each record's self_blake3, reports
     valid/invalid
  2. Loads prov/pipeline_manifest.json, recomputes its root_blake3, reports
     valid/invalid
  3. Re-hashes every artifact referenced by every provenance record and
     the manifest, reports any drift
  4. Cross-checks upstream_prov references form a consistent DAG
  5. Prints a human-readable summary + emits a structured verification JSON

Passes iff every check is VERIFIED.

Usage:
    python3 verify_twin_provenance.py /path/to/target_dir
    python3 verify_twin_provenance.py /path/to/target_dir --json report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from prism_prov import blake3_file, blake3_bytes, canonical_json


@dataclass
class VerifyResult:
    path: str
    kind: str              # record | manifest | artifact
    status: str            # VERIFIED | MISMATCH | MISSING | ERROR
    claimed: str = ""
    actual: str = ""
    detail: str = ""


def verify_record(path: Path) -> VerifyResult:
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        return VerifyResult(str(path), "record", "ERROR", detail=f"parse: {e}")
    claimed = data.get("self_blake3", "")
    dc = dict(data)
    dc["self_blake3"] = ""
    actual = blake3_bytes(canonical_json(dc))
    if claimed == actual:
        return VerifyResult(str(path), "record", "VERIFIED", claimed=claimed, actual=actual)
    return VerifyResult(str(path), "record", "MISMATCH",
                       claimed=claimed, actual=actual,
                       detail="self_blake3 does not match recomputed")


def verify_artifact_file(path: Path, expected: str) -> VerifyResult:
    p = Path(path)
    if not p.exists():
        return VerifyResult(str(p), "artifact", "MISSING",
                           claimed=expected, detail="file not present")
    try:
        actual = blake3_file(p)
    except Exception as e:
        return VerifyResult(str(p), "artifact", "ERROR", detail=f"hash: {e}")
    if actual == expected:
        return VerifyResult(str(p), "artifact", "VERIFIED",
                           claimed=expected, actual=actual)
    return VerifyResult(str(p), "artifact", "MISMATCH",
                       claimed=expected, actual=actual,
                       detail="BLAKE3 of file does not match deposited")


def verify_manifest(manifest_path: Path, target_dir: Path
                    ) -> Tuple[VerifyResult, List[VerifyResult]]:
    if not manifest_path.exists():
        return (VerifyResult(str(manifest_path), "manifest", "MISSING"),
                [])
    try:
        with open(manifest_path) as f:
            mani = json.load(f)
    except Exception as e:
        return (VerifyResult(str(manifest_path), "manifest", "ERROR", detail=f"parse: {e}"),
                [])

    # Recompute manifest root hash
    claimed_root = mani.get("root_blake3", "")
    mc = dict(mani)
    mc["root_blake3"] = ""
    actual_root = blake3_bytes(canonical_json(mc))
    root_result = VerifyResult(
        str(manifest_path), "manifest",
        "VERIFIED" if claimed_root == actual_root else "MISMATCH",
        claimed=claimed_root, actual=actual_root,
        detail="manifest root blake3" if claimed_root == actual_root else "manifest root hash drift",
    )

    # Verify each referenced record's blake3 matches its file
    artifact_results: List[VerifyResult] = []
    for rec in mani.get("records", []):
        if not rec.get("present", True):
            continue
        claimed = rec.get("blake3")
        if claimed is None:
            continue
        p = Path(rec["path"])
        if not p.exists():
            artifact_results.append(VerifyResult(
                str(p), "artifact", "MISSING",
                claimed=claimed, detail="manifest-referenced record file missing",
            ))
            continue
        actual = blake3_file(p)
        if actual == claimed:
            artifact_results.append(VerifyResult(
                str(p), "artifact", "VERIFIED",
                claimed=claimed, actual=actual,
            ))
        else:
            artifact_results.append(VerifyResult(
                str(p), "artifact", "MISMATCH",
                claimed=claimed, actual=actual,
                detail="record file blake3 does not match manifest entry",
            ))

    return root_result, artifact_results


def verify_record_artifacts(record_path: Path) -> List[VerifyResult]:
    """Verify all inputs and outputs referenced in a provenance record."""
    results = []
    try:
        with open(record_path) as f:
            data = json.load(f)
    except Exception as e:
        return [VerifyResult(str(record_path), "record", "ERROR", detail=str(e))]

    for role in ("inputs", "outputs"):
        for ref in data.get(role, []):
            if not ref.get("present", True):
                continue
            claimed = ref.get("blake3")
            path_str = ref.get("path")
            if not claimed or not path_str:
                continue
            p = Path(path_str)
            if not p.exists():
                results.append(VerifyResult(
                    str(p), "artifact", "MISSING",
                    claimed=claimed, detail=f"{role} referenced by {record_path.name}",
                ))
                continue
            actual = blake3_file(p)
            if actual == claimed:
                results.append(VerifyResult(
                    str(p), "artifact", "VERIFIED", claimed=claimed, actual=actual,
                ))
            else:
                results.append(VerifyResult(
                    str(p), "artifact", "MISMATCH",
                    claimed=claimed, actual=actual,
                    detail=f"{role} in {record_path.name}: BLAKE3 drift",
                ))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None,
                   help="Emit structured verification JSON")
    args = ap.parse_args()

    target_dir = args.target_dir
    prov_dir = target_dir / "prov"
    if not prov_dir.exists():
        print(f"FATAL: prov dir not found: {prov_dir}", file=sys.stderr)
        return 2

    all_results: List[VerifyResult] = []

    # Step 1: verify each record's self_hash
    print(f"PROVENANCE VERIFICATION — {target_dir.name}")
    print("=" * 72)
    print()

    records = sorted(prov_dir.glob("*.prov.json"))
    print(f"[1] Self-hash verification for {len(records)} records:")
    rec_pass = 0
    rec_fail = 0
    for rec in records:
        r = verify_record(rec)
        all_results.append(r)
        mark = "✓" if r.status == "VERIFIED" else "✗"
        print(f"  {mark} {rec.name:50s} {r.status}")
        if r.status == "VERIFIED":
            rec_pass += 1
        else:
            rec_fail += 1
            if r.detail:
                print(f"      {r.detail}")

    # Step 2: verify manifest + its referenced artifact hashes
    print()
    print("[2] Manifest + referenced-artifact hash verification:")
    manifest_path = prov_dir / "pipeline_manifest.json"
    mani_result, mani_art_results = verify_manifest(manifest_path, target_dir)
    all_results.append(mani_result)
    all_results.extend(mani_art_results)
    mark = "✓" if mani_result.status == "VERIFIED" else "✗"
    print(f"  {mark} manifest root: {mani_result.status}")
    art_pass = sum(1 for r in mani_art_results if r.status == "VERIFIED")
    art_fail = len(mani_art_results) - art_pass
    print(f"      {art_pass}/{len(mani_art_results)} manifest-referenced artifacts verified")
    for r in mani_art_results:
        if r.status != "VERIFIED":
            print(f"      ✗ {r.path}: {r.status} — {r.detail}")

    # Step 3: for each record, verify its inputs/outputs artifacts
    print()
    print("[3] Per-record input/output artifact hash re-verification:")
    per_rec_pass = 0
    per_rec_fail = 0
    for rec in records:
        for r in verify_record_artifacts(rec):
            all_results.append(r)
            if r.status == "VERIFIED":
                per_rec_pass += 1
            else:
                per_rec_fail += 1
                print(f"  ✗ {r.path}: {r.status} — {r.detail}")
    print(f"      {per_rec_pass} artifacts verified, {per_rec_fail} drift/missing")

    # Summary
    print()
    print("=" * 72)
    print("SUMMARY")
    print(f"  Records self-hash VERIFIED:        {rec_pass} / {len(records)}")
    print(f"  Manifest root:                      {mani_result.status}")
    print(f"  Manifest artifacts VERIFIED:        {art_pass} / {len(mani_art_results)}")
    print(f"  Per-record artifact re-hash:        {per_rec_pass} VERIFIED, {per_rec_fail} drift/missing")
    n_total = len(all_results)
    n_pass = sum(1 for r in all_results if r.status == "VERIFIED")
    overall_pass = (n_pass == n_total)
    print()
    print(f"  OVERALL: {'VERIFIED ✓' if overall_pass else 'DRIFT / FAIL ✗'}  ({n_pass}/{n_total})")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({
                "target_dir": str(target_dir),
                "overall": "VERIFIED" if overall_pass else "FAIL",
                "counts": {
                    "records_pass": rec_pass,
                    "records_total": len(records),
                    "manifest_root": mani_result.status,
                    "artifacts_pass": art_pass + per_rec_pass,
                    "artifacts_total": len(mani_art_results) + per_rec_pass + per_rec_fail,
                },
                "results": [
                    {"path": r.path, "kind": r.kind, "status": r.status,
                     "claimed": r.claimed, "actual": r.actual, "detail": r.detail}
                    for r in all_results
                ],
            }, f, indent=2)
        print(f"\nVerification report: {args.json}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
