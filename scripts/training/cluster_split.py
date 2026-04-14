"""Sequence-identity-based train/val/test splitting for PRISM-4D training.

MMseqs2 easy-cluster at 30% identity groups homologs into the same cluster.
All members of a cluster are assigned to the SAME split (train/val/test) —
no cluster straddles splits — eliminating homolog leakage.

Uses sequences extracted from the extract_all_features.py .npz bundles
(the `sequence` key is a 1-letter per-residue string for the target).

Usage from a training script:

    from cluster_split import cluster_split_bundles
    train, val, test = cluster_split_bundles(
        bundle_dir=Path("/mnt/storage/spike-audit/features-pct95"),
        targets=["10dc_chainA", ...],
        val_frac=0.15, test_frac=0.05, min_seq_id=0.3,
        cache_path=Path("/mnt/storage/spike-audit/seq_clusters.json"),
    )
    # Returns three lists of target names.

The cluster map is cached so subsequent calls reuse MMseqs2 output.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


MMSEQS_BIN = os.environ.get("MMSEQS_BIN", "mmseqs")


# ─────────────────────────────────────────────────────────────
#  Sequence extraction from .npz bundles
# ─────────────────────────────────────────────────────────────

def _load_sequence(bundle_dir: Path, target: str) -> Optional[str]:
    """Return the 1-letter sequence for a target from its .npz. Supports
    both `{t}_features.npz` and `{t}.features.npz` naming."""
    for stem in (f"{target}_features.npz", f"{target}.features.npz"):
        p = bundle_dir / stem
        if p.exists():
            try:
                d = np.load(p, allow_pickle=False)
            except Exception:
                return None
            if "sequence" in d.files:
                s = d["sequence"]
                return str(s.item() if hasattr(s, "item") and s.ndim == 0 else s)
            return None
    return None


def collect_sequences(bundle_dir: Path, targets: Iterable[str]
                      ) -> Dict[str, str]:
    """Return {target: sequence}. Skips targets without a valid sequence."""
    out: Dict[str, str] = {}
    for t in targets:
        seq = _load_sequence(bundle_dir, t)
        if seq and len(seq) >= 10:
            out[t] = seq
    return out


# ─────────────────────────────────────────────────────────────
#  MMseqs2 easy-cluster driver
# ─────────────────────────────────────────────────────────────

def mmseqs_cluster(sequences: Dict[str, str],
                   min_seq_id: float = 0.3,
                   coverage: float = 0.8,
                   tmp_dir: Optional[Path] = None) -> Dict[str, str]:
    """Run mmseqs easy-cluster on the provided sequences.

    Returns {target: cluster_representative_id}. Every target is mapped to
    the target-name of its cluster representative.

    Parameters
        min_seq_id: identity threshold (0.3 = 30%)
        coverage:   coverage requirement (MMseqs default 0.8)
        tmp_dir:    scratch dir for MMseqs — purged on exit if created here.
    """
    if not sequences:
        return {}
    own_tmp = tmp_dir is None
    if tmp_dir is None:
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix="prism_cluster_"))
    else:
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        fasta_path = tmp_dir / "sequences.fasta"
        with open(fasta_path, "w") as f:
            for name, seq in sequences.items():
                # MMseqs2 requires simple headers; target names are already safe (pdb_chain)
                f.write(f">{name}\n{seq}\n")

        out_prefix = tmp_dir / "clusters"
        mmseqs_tmp = tmp_dir / "mmseqs_work"
        mmseqs_tmp.mkdir(exist_ok=True)

        cmd = [
            MMSEQS_BIN, "easy-cluster", str(fasta_path),
            str(out_prefix), str(mmseqs_tmp),
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
            "--cov-mode", "0",
            "--threads", str(max(1, os.cpu_count() // 2)),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"mmseqs easy-cluster failed:\n{r.stderr[-2000:]}")

        # Output: <prefix>_cluster.tsv — two columns: rep  member
        tsv_path = tmp_dir / "clusters_cluster.tsv"
        if not tsv_path.exists():
            raise RuntimeError(f"mmseqs did not produce {tsv_path}")

        target_to_cluster: Dict[str, str] = {}
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                rep, member = parts
                target_to_cluster[member] = rep
        return target_to_cluster
    finally:
        if own_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
#  Cluster-level split
# ─────────────────────────────────────────────────────────────

def split_by_cluster(target_to_cluster: Dict[str, str],
                     val_frac: float = 0.15,
                     test_frac: float = 0.05,
                     seed: int = 42,
                     ) -> Tuple[List[str], List[str], List[str]]:
    """Partition targets into (train, val, test) lists.

    Splits are computed at the CLUSTER level, then expanded to member
    targets. A single cluster never straddles splits.

    The ratios describe the fraction of CLUSTERS (not targets) assigned to
    each split. Target counts per split can drift from the ratio when
    cluster sizes are heterogeneous — this is the expected, correct
    behavior when preventing homolog leakage.
    """
    cluster_to_members: Dict[str, List[str]] = defaultdict(list)
    for t, c in target_to_cluster.items():
        cluster_to_members[c].append(t)

    clusters = sorted(cluster_to_members.keys())
    rng = random.Random(seed)
    rng.shuffle(clusters)

    n_clusters = len(clusters)
    n_test = max(1, int(round(test_frac * n_clusters)))
    n_val = max(1, int(round(val_frac * n_clusters)))
    test_clusters = set(clusters[:n_test])
    val_clusters = set(clusters[n_test:n_test + n_val])
    train_clusters = set(clusters[n_test + n_val:])

    train, val, test = [], [], []
    for c, members in cluster_to_members.items():
        if c in test_clusters:
            test.extend(members)
        elif c in val_clusters:
            val.extend(members)
        else:
            train.extend(members)
    return sorted(train), sorted(val), sorted(test)


# ─────────────────────────────────────────────────────────────
#  Top-level helper (with caching)
# ─────────────────────────────────────────────────────────────

def cluster_split_bundles(
    bundle_dir: Path,
    targets: List[str],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.05,
    min_seq_id: float = 0.3,
    cache_path: Optional[Path] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """End-to-end: extract sequences → MMseqs2 cluster → split → return lists.

    If `cache_path` is provided and exists, the cluster map is loaded from
    JSON instead of re-running MMseqs2. Delete the cache to force a rebuild.
    """
    sequences = collect_sequences(bundle_dir, targets)
    if verbose:
        print(f"[cluster_split] sequences collected: {len(sequences)}/{len(targets)}")
    if not sequences:
        raise RuntimeError("No sequences found in bundle_dir")

    cache_valid = False
    target_to_cluster: Dict[str, str] = {}
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            # Reuse iff the cached set covers our current targets and params match
            if (cached.get("min_seq_id") == min_seq_id
                    and set(cached["map"].keys()) >= set(sequences.keys())):
                target_to_cluster = {k: v for k, v in cached["map"].items()
                                     if k in sequences}
                cache_valid = True
                if verbose:
                    print(f"[cluster_split] loaded cached cluster map "
                          f"({len(target_to_cluster)} targets, "
                          f"{len(set(target_to_cluster.values()))} clusters)")
        except Exception:
            cache_valid = False

    if not cache_valid:
        if verbose:
            print(f"[cluster_split] running MMseqs2 easy-cluster at "
                  f"{int(min_seq_id*100)}% identity...")
        target_to_cluster = mmseqs_cluster(sequences, min_seq_id=min_seq_id)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({
                "min_seq_id": min_seq_id,
                "map": target_to_cluster,
                "n_targets": len(target_to_cluster),
                "n_clusters": len(set(target_to_cluster.values())),
            }, indent=2))
        if verbose:
            n_c = len(set(target_to_cluster.values()))
            print(f"[cluster_split] {n_c} clusters from {len(target_to_cluster)} "
                  f"targets (avg {len(target_to_cluster)/max(n_c,1):.1f}/cluster)")

    train, val, test = split_by_cluster(target_to_cluster,
                                         val_frac=val_frac,
                                         test_frac=test_frac,
                                         seed=seed)
    if verbose:
        def _n_clusters(names): return len({target_to_cluster[n] for n in names
                                            if n in target_to_cluster})
        print(f"[cluster_split] train={len(train)} ({_n_clusters(train)} clusters)  "
              f"val={len(val)} ({_n_clusters(val)} clusters)  "
              f"test={len(test)} ({_n_clusters(test)} clusters)")
    return train, val, test


def target_to_cluster_map(bundle_dir: Path, targets: List[str],
                          min_seq_id: float = 0.3,
                          cache_path: Optional[Path] = None,
                          ) -> Dict[str, str]:
    """Build just the cluster map without splitting. Used by LOTO trainers
    that need cluster info to drop homologs from the training fold when
    a specific target is held out for validation."""
    sequences = collect_sequences(bundle_dir, targets)
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if (cached.get("min_seq_id") == min_seq_id
                    and set(cached["map"].keys()) >= set(sequences.keys())):
                return {k: v for k, v in cached["map"].items() if k in sequences}
        except Exception:
            pass
    m = mmseqs_cluster(sequences, min_seq_id=min_seq_id)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "min_seq_id": min_seq_id, "map": m,
            "n_targets": len(m),
            "n_clusters": len(set(m.values())),
        }, indent=2))
    return m


if __name__ == "__main__":
    # Smoke test on the extracted pct95 bundles
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path,
                        default=Path("/mnt/storage/spike-audit/features-pct95"))
    parser.add_argument("--cache-path", type=Path,
                        default=Path("/mnt/storage/spike-audit/seq_clusters.json"))
    parser.add_argument("--min-seq-id", type=float, default=0.3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.05)
    args = parser.parse_args()

    all_files = list(args.bundle_dir.glob("*_features.npz"))
    targets = [p.name.replace("_features.npz", "") for p in all_files]
    print(f"Found {len(targets)} bundles in {args.bundle_dir}")

    train, val, test = cluster_split_bundles(
        bundle_dir=args.bundle_dir, targets=targets,
        val_frac=args.val_frac, test_frac=args.test_frac,
        min_seq_id=args.min_seq_id,
        cache_path=args.cache_path,
    )
    print(f"\nSample train[:5]: {train[:5]}")
    print(f"Sample val[:5]:   {val[:5]}")
    print(f"Sample test[:5]:  {test[:5]}")
