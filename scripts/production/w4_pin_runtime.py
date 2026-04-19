#!/usr/bin/env python3
"""W4 runtime-provenance producer — POST /targets/:target/runtime.

Single authorized producer of runtime/artifact provenance for the v4
feature-service hardening pipeline.  Pinned against:

  - cloudflare/workers/feature-pipeline/src/index.js:W4 handler
  - scripts/training/v4_feature_contract.yaml:writers.W4_runtime_endpoint
  - docs/contracts/persistence_contract.md (W5 is separate; not this lane)

Non-invention invariant (hard rule):
  - the producer NEVER fabricates a field.  Every value sent is either
    (a) read from the binding_sites.json artifact (authoritative source),
    (b) computed as a sha256 of an artifact file's bytes,
    (c) passed explicitly by the caller via a CLI flag.
  - engine_time_seconds: not sent unless --engine-time-seconds is supplied.
  - engine_flags:        not sent unless --engine-flags is supplied.
    For legacy pct70 backfill both are omitted; Worker stores NULL.

Usage (legacy pct70 backfill):
    python3 scripts/production/w4_pin_runtime.py \\
        --target 10dc_chainA \\
        --binding-sites /mnt/.../10dc_chainA/10dc_chainA.binding_sites.json \\
        --ground-truth  /mnt/.../10dc_chainA/10dc_chainA_ground_truth.json \\
        --engine-commit "legacy_pct70_campaign_v1" \\
        --api-base https://prism-feature-pipeline.is-0b9.workers.dev

Usage (new run, inside prism-validate-and-run.sh):
    python3 scripts/production/w4_pin_runtime.py \\
        --target $TARGET \\
        --binding-sites $OUT/$TARGET.binding_sites.json \\
        --ground-truth  $OUT/${TARGET}_ground_truth.json \\
        --engine-commit $PRISM_ENGINE_COMMIT \\
        --engine-flags  "$ENGINE_ARGV" \\
        --engine-time-seconds $ELAPSED_SEC \\
        --api-base $PRISM_API
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_bs_top_level(path: Path) -> dict:
    """Read the authoritative binding_sites.json top-level provenance."""
    d = json.load(open(path))
    # Only the 5 keys W4 writes to targets; no derivation beyond pass-through.
    return {
        "engine_n_streams": d.get("n_streams"),
        "engine_mode": d.get("mode"),
        "engine_simulation_time_sec": d.get("simulation_time_sec"),
        "engine_total_steps_per_stream": d.get("total_steps_per_stream"),
        "lining_residue_cutoff_angstroms": d.get("lining_residue_cutoff_angstroms"),
    }


def post(api_base: str, target: str, payload: dict, timeout: int = 30) -> tuple[int, str]:
    req = urllib.request.Request(
        f"{api_base}/targets/{target}/runtime",
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 prism4d-w4-pin-runtime",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace") if e.fp else str(e)
    except urllib.error.URLError as e:
        return 0, str(e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--binding-sites", required=True, type=Path)
    ap.add_argument("--ground-truth", required=True, type=Path)
    ap.add_argument("--engine-commit", required=True,
                    help="Explicit commit label — no default, no fake 'UNKNOWN' tolerated.")
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--engine-flags", default=None,
                    help="Optional. Omit for legacy backfill where flags are not recoverable.")
    ap.add_argument("--engine-time-seconds", type=float, default=None,
                    help="Optional. Omit for legacy backfill where runtime is not uniquely recoverable.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the payload that would be POSTed and exit 0.")
    args = ap.parse_args()

    if not args.binding_sites.exists():
        print(f"FATAL: binding_sites.json missing: {args.binding_sites}", file=sys.stderr)
        return 2
    if not args.ground_truth.exists():
        print(f"FATAL: ground_truth.json missing: {args.ground_truth}", file=sys.stderr)
        return 2
    if not args.engine_commit.strip() or args.engine_commit.strip().upper() == "UNKNOWN":
        print("FATAL: --engine-commit must be a real pin (not empty, not 'UNKNOWN')", file=sys.stderr)
        return 2

    bs_top = read_bs_top_level(args.binding_sites)
    payload = {
        "engine_commit": args.engine_commit,
        "binding_sites_json_sha256": sha256_file(args.binding_sites),
        "ground_truth_json_sha256":  sha256_file(args.ground_truth),
        **bs_top,
    }
    # Non-invention gate: only include recoverable optional fields when supplied.
    if args.engine_flags is not None:
        payload["engine_flags"] = args.engine_flags
    if args.engine_time_seconds is not None:
        payload["engine_time_seconds"] = args.engine_time_seconds

    if args.dry_run:
        print(json.dumps({"target": args.target, "payload": payload}, indent=2))
        return 0

    code, body = post(args.api_base, args.target, payload)
    if code != 200:
        print(f"FAIL: W4 POST returned HTTP {code}: {body}", file=sys.stderr)
        return 1
    print(f"OK: {args.target}  HTTP {code}  {body.strip()}")
    print(f"   engine_commit={payload['engine_commit']}")
    print(f"   binding_sites_json_sha256={payload['binding_sites_json_sha256']}")
    print(f"   ground_truth_json_sha256={payload['ground_truth_json_sha256']}")
    print(f"   engine_n_streams={payload.get('engine_n_streams')}  "
          f"engine_mode={payload.get('engine_mode')}")
    print(f"   engine_simulation_time_sec={payload.get('engine_simulation_time_sec')}  "
          f"engine_total_steps_per_stream={payload.get('engine_total_steps_per_stream')}")
    print(f"   lining_residue_cutoff_angstroms={payload.get('lining_residue_cutoff_angstroms')}")
    skipped = [k for k in ("engine_flags", "engine_time_seconds") if k not in payload]
    if skipped:
        print(f"   (intentionally omitted — caller did not supply: {', '.join(skipped)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
