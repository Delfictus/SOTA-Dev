#!/usr/bin/env python3
"""Deterministic shard partitioner for W3b remaining-targets pass.

Input : comma-separated target list on stdin or --targets-file.
Output: one file per shard under .w3b_shards/shard_{k}.txt,
        where k = int(md5(target).hexdigest(),16) % N_SHARDS.

Ordering within a shard is sorted(asc) to keep worker runs byte-repro.
"""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path

N_SHARDS = 16


def shard_id(target: str) -> int:
    return int(hashlib.md5(target.encode()).hexdigest(), 16) % N_SHARDS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-file", type=Path, required=True,
                    help="file containing a single line of comma-separated targets")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-shards", type=int, default=N_SHARDS)
    args = ap.parse_args()

    raw = args.targets_file.read_text().strip()
    targets = [t.strip() for t in raw.split(",") if t.strip()]
    if not targets:
        print("FATAL: empty targets list", file=sys.stderr)
        return 2

    buckets: list[list[str]] = [[] for _ in range(args.n_shards)]
    for t in targets:
        k = int(hashlib.md5(t.encode()).hexdigest(), 16) % args.n_shards
        buckets[k].append(t)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    sizes = []
    for k, bucket in enumerate(buckets):
        bucket.sort()
        (args.out_dir / f"shard_{k}.txt").write_text(",".join(bucket))
        sizes.append(len(bucket))
        total += len(bucket)

    print(f"n_targets={total}  n_shards={args.n_shards}")
    for k, n in enumerate(sizes):
        print(f"  shard_{k:02d}  n={n}")
    if total != len(targets):
        print("FATAL: target count mismatch", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
