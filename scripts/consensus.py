#!/usr/bin/env python3
"""PRISM4D — Cross-Run Metastable Pocket Consensus.

Clusters fully gated site objects across stochastic runs into recurrent
pocket manifolds.  Ranks manifolds lexicographically by persistence,
pass fraction, structural stability, then mean gated quality.

NOT averaging.  NOT voting.  NOT smoothing.
Identifying metastable pocket attractors from a population of
stochastic realizations.

Usage:
    python3 scripts/consensus.py \\
        --run-dirs /tmp/prism_1btl_r0 /tmp/prism_1btl_r1 ... \\
        --target-name 1btl \\
        --out /tmp/prism_1btl_consensus/

Programmatic:
    from scripts.consensus import ConsensusBuilder
    builder = ConsensusBuilder()
    result = builder.build(run_dirs, target_name, n_runs)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from scripts.interfaces.consensus_site import (
    ConsensusSite,
    ConsensusResult,
    MemberSite,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CENTROID_THRESHOLD = 5.0    # Angstrom — max distance for same consensus site
LINING_OVERLAP_MIN = 0.2   # Jaccard threshold for lining residue identity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dist(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Load run data
# ---------------------------------------------------------------------------
def load_run_sites(run_dir: Path, run_id: int) -> List[MemberSite]:
    """Load all site objects from a single run's canonical pipeline output."""
    # Load binding_sites
    bs_files = sorted(run_dir.glob("*.binding_sites.json"))
    if not bs_files:
        return []
    with open(bs_files[0]) as f:
        bs_data = json.load(f)
    sites = bs_data if isinstance(bs_data, list) else bs_data.get("sites", [])

    # Load gating result
    gating_path = run_dir / "design" / "gating_result.json"
    gate_decisions = {}
    if gating_path.exists():
        with open(gating_path) as f:
            gr = json.load(f)
        for d in gr.get("decisions", []):
            gate_decisions[d["site_id"]] = d

    # Load anchor maps
    anchor_data = {}
    for p in sorted(run_dir.glob("design/*.json")):
        if "anchor" in p.name:
            continue  # skip design brief jsons, load from gating
    # Anchors are embedded in design brief JSONs
    for p in sorted((run_dir / "design").glob("*_site*.json")):
        try:
            with open(p) as f:
                brief = json.load(f)
            sid = brief.get("site_id")
            if sid is not None:
                anchors = brief.get("anchor_map", {}).get("anchors", [])
                anchor_data[sid] = [a.get("residue_id", -1) for a in anchors]
        except (json.JSONDecodeError, KeyError):
            continue

    members: List[MemberSite] = []
    for site in sites:
        sid = site.get("id", -1)
        centroid = site.get("centroid", [0, 0, 0])
        gd = gate_decisions.get(sid, {})

        # Extract response selectivity from gate decision
        rp = gd.get("response_profile") or {}
        cr = gd.get("contact_reorg") or {}

        lining = site.get("lining_residues", [])
        lining_ids = [r.get("resid", -1) for r in lining]

        members.append(MemberSite(
            run_id=run_id,
            site_id=sid,
            centroid=(centroid[0], centroid[1], centroid[2]),
            quality_score=site.get("quality_score",
                                   site.get("rank_score", 0.0)),
            volume=site.get("volume", 0.0),
            gate_passed=gd.get("overall_pass", True),
            blocked_by=gd.get("blocked_by"),
            contact_reorg_strength=cr.get("localization_ratio", 0.0),
            response_sharpness=rp.get("sharpness", 0.0),
            response_energy_density=rp.get("energy_density", 0.0),
            anchor_residue_ids=anchor_data.get(sid, []),
            n_anchors=len(anchor_data.get(sid, [])),
            lining_residue_ids=lining_ids,
        ))

    return members


# ---------------------------------------------------------------------------
# Consensus clustering
# ---------------------------------------------------------------------------
def cluster_sites(
    all_members: List[MemberSite],
    centroid_threshold: float = CENTROID_THRESHOLD,
    lining_overlap_min: float = LINING_OVERLAP_MIN,
) -> List[List[MemberSite]]:
    """Cluster member sites across runs into consensus groups.

    Rule: same consensus site if centroid within threshold AND
    lining residue Jaccard overlap >= minimum.
    """
    assigned = [False] * len(all_members)
    clusters: List[List[MemberSite]] = []

    # Sort by quality_score descending — seed clusters from strongest sites
    order = sorted(range(len(all_members)),
                   key=lambda i: all_members[i].quality_score, reverse=True)

    for seed_idx in order:
        if assigned[seed_idx]:
            continue

        seed = all_members[seed_idx]
        cluster = [seed]
        assigned[seed_idx] = True
        seed_lining = set(seed.lining_residue_ids)

        for j in order:
            if assigned[j]:
                continue
            candidate = all_members[j]

            # Primary: centroid distance
            d = _dist(seed.centroid, candidate.centroid)
            if d > centroid_threshold:
                continue

            # Secondary: lining residue overlap
            cand_lining = set(candidate.lining_residue_ids)
            if seed_lining and cand_lining:
                overlap = _jaccard(seed_lining, cand_lining)
                if overlap < lining_overlap_min:
                    continue

            cluster.append(candidate)
            assigned[j] = True

        clusters.append(cluster)

    return clusters


# ---------------------------------------------------------------------------
# Consensus site construction
# ---------------------------------------------------------------------------
def build_consensus_site(
    cluster_id: int,
    members: List[MemberSite],
    n_runs: int,
) -> ConsensusSite:
    """Build a ConsensusSite from a cluster of member sites."""
    # Persistence: fraction of runs represented
    runs_present = set(m.run_id for m in members)
    persistence = len(runs_present) / n_runs

    # Pass fraction
    n_passed = sum(1 for m in members if m.gate_passed)
    pass_fraction = n_passed / len(members) if members else 0.0

    # Centroid mean and variance
    centroids = [m.centroid for m in members]
    cx = sum(c[0] for c in centroids) / len(centroids)
    cy = sum(c[1] for c in centroids) / len(centroids)
    cz = sum(c[2] for c in centroids) / len(centroids)
    centroid_mean = (round(cx, 3), round(cy, 3), round(cz, 3))

    centroid_var = math.sqrt(
        sum(_dist(c, centroid_mean) ** 2 for c in centroids) / len(centroids)
    )

    # Mean metrics
    mean_qs = sum(m.quality_score for m in members) / len(members)
    mean_cr = sum(m.contact_reorg_strength for m in members) / len(members)
    mean_sharp = sum(m.response_sharpness for m in members) / len(members)

    # Anchor consistency: pairwise Jaccard of anchor residue sets
    anchor_sets = [set(m.anchor_residue_ids) for m in members if m.anchor_residue_ids]
    if len(anchor_sets) >= 2:
        jaccards = []
        for i in range(len(anchor_sets)):
            for j in range(i + 1, len(anchor_sets)):
                jaccards.append(_jaccard(anchor_sets[i], anchor_sets[j]))
        anchor_consistency = sum(jaccards) / len(jaccards)
    elif len(anchor_sets) == 1:
        anchor_consistency = 1.0
    else:
        anchor_consistency = 0.0

    # Lining consistency
    lining_sets = [set(m.lining_residue_ids) for m in members if m.lining_residue_ids]
    if len(lining_sets) >= 2:
        jaccards = []
        for i in range(len(lining_sets)):
            for j in range(i + 1, len(lining_sets)):
                jaccards.append(_jaccard(lining_sets[i], lining_sets[j]))
        lining_consistency = sum(jaccards) / len(jaccards)
    elif len(lining_sets) == 1:
        lining_consistency = 1.0
    else:
        lining_consistency = 0.0

    # Gate failure reasons
    failures: Dict[str, int] = Counter()
    for m in members:
        if not m.gate_passed and m.blocked_by:
            failures[m.blocked_by] += 1

    return ConsensusSite(
        cluster_id=cluster_id,
        member_sites=members,
        n_runs_total=n_runs,
        persistence=round(persistence, 4),
        pass_fraction=round(pass_fraction, 4),
        centroid_mean=centroid_mean,
        centroid_variance=round(centroid_var, 3),
        mean_quality_score=round(mean_qs, 6),
        mean_contact_reorg=round(mean_cr, 6),
        mean_response_sharpness=round(mean_sharp, 4),
        anchor_consistency=round(anchor_consistency, 4),
        lining_consistency=round(lining_consistency, 4),
        gate_failure_reasons=dict(failures),
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class ConsensusBuilder:
    """Builds cross-run metastable pocket consensus."""

    def __init__(
        self,
        centroid_threshold: float = CENTROID_THRESHOLD,
        lining_overlap_min: float = LINING_OVERLAP_MIN,
    ):
        self.centroid_threshold = centroid_threshold
        self.lining_overlap_min = lining_overlap_min

    def build(
        self,
        run_dirs: List[str],
        target_name: str,
    ) -> ConsensusResult:
        """Build consensus from multiple run directories.

        Args:
            run_dirs: List of paths to replicate run output directories.
            target_name: Target identifier.

        Returns:
            ConsensusResult with ranked consensus sites.
        """
        n_runs = len(run_dirs)

        # Load all member sites from all runs
        all_members: List[MemberSite] = []
        for run_id, rd in enumerate(run_dirs):
            members = load_run_sites(Path(rd), run_id)
            all_members.extend(members)
            print(f"  Run {run_id}: {len(members)} sites loaded "
                  f"({sum(1 for m in members if m.gate_passed)} passed)")

        print(f"  Total: {len(all_members)} sites across {n_runs} runs")

        # Cluster across runs
        clusters = cluster_sites(
            all_members, self.centroid_threshold, self.lining_overlap_min
        )
        print(f"  Clusters: {len(clusters)}")

        # Build consensus sites
        consensus_sites: List[ConsensusSite] = []
        for cid, cluster in enumerate(clusters):
            cs = build_consensus_site(cid, cluster, n_runs)
            consensus_sites.append(cs)

        # Lexicographic ranking:
        #   1. persistence (desc)
        #   2. pass_fraction (desc)
        #   3. centroid_variance (asc — lower = more stable)
        #   4. mean_quality_score (desc)
        #   5. mean_contact_reorg (desc)
        consensus_sites.sort(key=lambda cs: (
            -cs.persistence,
            -cs.pass_fraction,
            cs.centroid_variance,
            -cs.mean_quality_score,
            -cs.mean_contact_reorg,
        ))

        # Re-assign cluster IDs by rank order
        for i, cs in enumerate(consensus_sites):
            cs.cluster_id = i

        return ConsensusResult(
            target_name=target_name,
            n_replicates=n_runs,
            consensus_sites=consensus_sites,
            n_consensus=len(consensus_sites),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D Cross-Run Metastable Pocket Consensus"
    )
    parser.add_argument(
        "--run-dirs", nargs="+", required=True,
        help="Paths to replicate run output directories",
    )
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--centroid-threshold", type=float, default=5.0)
    parser.add_argument("--lining-overlap", type=float, default=0.2)
    args = parser.parse_args()

    print(f"[Consensus] {args.target_name}: {len(args.run_dirs)} replicates")

    builder = ConsensusBuilder(args.centroid_threshold, args.lining_overlap)
    result = builder.build(args.run_dirs, args.target_name)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "consensus_sites.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"CONSENSUS: {args.target_name}")
    print(f"  Replicates: {result.n_replicates}")
    print(f"  Consensus sites: {result.n_consensus}")
    print()
    print(f"  {'Rank':>4} {'Persist':>7} {'Pass%':>6} {'Var(A)':>6} "
          f"{'mGTCKL':>7} {'mCR':>6} {'AnchJ':>6} {'LinJ':>6} {'Members':>7}")
    print(f"  {'-'*62}")
    for cs in result.consensus_sites[:20]:
        print(f"  {cs.cluster_id:>4} {cs.persistence:>7.2f} "
              f"{cs.pass_fraction:>6.2f} {cs.centroid_variance:>6.2f} "
              f"{cs.mean_quality_score:>7.4f} {cs.mean_contact_reorg:>6.4f} "
              f"{cs.anchor_consistency:>6.2f} {cs.lining_consistency:>6.2f} "
              f"{len(cs.member_sites):>7}")
        if cs.gate_failure_reasons:
            reasons = ", ".join(f"{k}:{v}" for k, v in cs.gate_failure_reasons.items())
            print(f"       failures: {reasons}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
