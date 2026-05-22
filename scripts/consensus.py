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

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
def load_run_sites(
    run_dir: Path,
    run_id: int,
    delta_by_member: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None,
) -> List[MemberSite]:
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

    # Load growth vector data from design briefs
    growth_data: Dict[int, List[Tuple[float, float, float]]] = {}
    for p in sorted((run_dir / "design").glob("*_site*.json")):
        try:
            with open(p) as f:
                brief = json.load(f)
            sid = brief.get("site_id")
            if sid is not None:
                vectors = brief.get("growth_map", {}).get("vectors", [])
                dirs = [
                    (v["direction"][0], v["direction"][1], v["direction"][2])
                    for v in vectors if "direction" in v
                ]
                growth_data[sid] = dirs[:10]  # top 10 vectors
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

        delta = (delta_by_member or {}).get((run_id, sid), {})
        members.append(MemberSite(
            run_id=run_id,
            site_id=sid,
            centroid=(centroid[0], centroid[1], centroid[2]),
            quality_score=site.get("quality_score",
                                   site.get("rank_score", 0.0)),
            volume=site.get("volume", 0.0),
            enclosure=site.get("burial_score", 0.0),
            therm_class=site.get("therm_class", "UNKNOWN"),
            gate_passed=gd.get("overall_pass", True),
            blocked_by=gd.get("blocked_by"),
            contact_reorg_strength=cr.get("localization_ratio", 0.0),
            response_sharpness=rp.get("sharpness", 0.0),
            response_energy_density=rp.get("energy_density", 0.0),
            mean_localization=site.get("localization_score_raw",
                                       site.get("mean_burial", 0.0)),
            anchor_residue_ids=anchor_data.get(sid, []),
            n_anchors=len(anchor_data.get(sid, [])),
            lining_residue_ids=lining_ids,
            growth_vector_directions=growth_data.get(sid, []),
            delta_stability=delta.get("delta_stability"),
            delta_rms=delta.get("delta_rms"),
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

    # Mean localization
    mean_loc = sum(m.mean_localization for m in members) / len(members)

    delta_stabilities = [
        float(m.delta_stability) for m in members if m.delta_stability is not None
    ]
    delta_rms_values = [float(m.delta_rms) for m in members if m.delta_rms is not None]
    mean_delta_stability = (
        sum(delta_stabilities) / len(delta_stabilities)
        if delta_stabilities else None
    )
    mean_delta_rms = (
        sum(delta_rms_values) / len(delta_rms_values)
        if delta_rms_values else None
    )

    # Growth vector consistency: mean pairwise cosine similarity of direction sets
    def _cosine_sim_sets(
        dirs_a: List[Tuple[float, float, float]],
        dirs_b: List[Tuple[float, float, float]],
    ) -> float:
        """Mean best-match cosine similarity between two direction sets."""
        if not dirs_a or not dirs_b:
            return 0.0
        sims = []
        for da in dirs_a:
            best = max(
                sum(da[k] * db[k] for k in range(3))
                / (max(math.sqrt(sum(x**2 for x in da)), 1e-12)
                   * max(math.sqrt(sum(x**2 for x in db)), 1e-12))
                for db in dirs_b
            )
            sims.append(best)
        return sum(sims) / len(sims) if sims else 0.0

    gv_sets = [m.growth_vector_directions for m in members if m.growth_vector_directions]
    if len(gv_sets) >= 2:
        gv_sims = []
        for i in range(len(gv_sets)):
            for j in range(i + 1, len(gv_sets)):
                gv_sims.append(_cosine_sim_sets(gv_sets[i], gv_sets[j]))
        growth_vector_consistency = sum(gv_sims) / len(gv_sims)
    elif len(gv_sets) == 1:
        growth_vector_consistency = 1.0
    else:
        growth_vector_consistency = 0.0

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
        mean_localization=round(mean_loc, 4),
        anchor_consistency=round(anchor_consistency, 4),
        growth_vector_consistency=round(growth_vector_consistency, 4),
        lining_consistency=round(lining_consistency, 4),
        gate_failure_reasons=dict(failures),
        mean_delta_stability=(
            round(mean_delta_stability, 6)
            if mean_delta_stability is not None else None
        ),
        mean_delta_rms=round(mean_delta_rms, 6) if mean_delta_rms is not None else None,
    )


def load_delta_stability_manifest(path: Optional[str]) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load optional medoid-diff stability sidecar keyed by run_id/site_id.

    Accepted rows:
      {"run_id": 0, "site_id": 3, "delta_stability": 0.91, "delta_rms": 0.08}
      {"replicate": 0, "id": 3, "stability": 0.91, "rms": 0.08}
    """
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    rows = data.get("sites", data if isinstance(data, list) else [])
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        run_id = row.get("run_id", row.get("replicate", row.get("replicate_id")))
        site_id = row.get("site_id", row.get("id"))
        if run_id is None or site_id is None:
            continue
        stability = row.get("delta_stability", row.get("stability"))
        delta_rms = row.get("delta_rms", row.get("rms", row.get("residue_delta_rms")))
        values: Dict[str, float] = {}
        if stability is not None:
            values["delta_stability"] = float(stability)
        if delta_rms is not None:
            values["delta_rms"] = float(delta_rms)
        if values:
            out[(int(run_id), int(site_id))] = values
    return out


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class ConsensusBuilder:
    """Builds cross-run metastable pocket consensus."""

    def __init__(
        self,
        centroid_threshold: float = CENTROID_THRESHOLD,
        lining_overlap_min: float = LINING_OVERLAP_MIN,
        delta_by_member: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None,
        rank_mode: str = "rf3",
    ):
        self.centroid_threshold = centroid_threshold
        self.lining_overlap_min = lining_overlap_min
        self.delta_by_member = delta_by_member or {}
        self.rank_mode = rank_mode

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
            members = load_run_sites(Path(rd), run_id, self.delta_by_member)
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

        # RF3 rank fusion: enclosure + persistence + anchor_consistency
        # Found by scripts/rank_search.py exhaustive search over 1023 RF1-RF5
        # combinations on 3 validation targets (1JWP, 1P38, 2HNP).
        # Achieves rank 1 on all 3. Physically interpretable:
        # enc = deeply enclosed pocket, p = persistent across replicates,
        # ac = structurally consistent anchor residues.
        def _rf3_rank(sites: List[ConsensusSite]) -> List[int]:
            """Rank fusion: sum of per-signal ranks (lower = better)."""
            n = len(sites)
            if n == 0:
                return []

            def _ranks(key_fn, reverse=True):
                order = sorted(range(n), key=lambda i: key_fn(sites[i]),
                               reverse=reverse)
                ranks = [0] * n
                for r, idx in enumerate(order):
                    ranks[idx] = r + 1
                return ranks

            enc_r = _ranks(lambda cs: (
                cs.member_sites[0].enclosure if cs.member_sites else 0.0
            ))
            per_r = _ranks(lambda cs: cs.persistence)
            ac_r = _ranks(lambda cs: cs.anchor_consistency)

            return [enc_r[i] + per_r[i] + ac_r[i] for i in range(n)]

        if self.rank_mode == "lexicographic":
            def _stability(cs: ConsensusSite) -> float:
                if cs.mean_delta_stability is not None:
                    return cs.mean_delta_stability
                return 1.0 / (1.0 + max(cs.centroid_variance, 0.0))

            consensus_sites.sort(
                key=lambda cs: (
                    -cs.persistence,
                    -cs.pass_fraction,
                    -_stability(cs),
                    -cs.mean_quality_score,
                )
            )
        else:
            rf3_scores = _rf3_rank(consensus_sites)
            order = sorted(range(len(consensus_sites)),
                           key=lambda i: rf3_scores[i])
            consensus_sites = [consensus_sites[i] for i in order]

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
    parser.add_argument(
        "--delta-stability-manifest",
        default=None,
        help="Optional medoid-diff stability sidecar keyed by run_id/site_id",
    )
    parser.add_argument(
        "--rank-mode",
        choices=("rf3", "lexicographic"),
        default="rf3",
        help="rf3 preserves legacy behavior; lexicographic uses persistence/pass/stability/quality",
    )
    args = parser.parse_args()

    print(f"[Consensus] {args.target_name}: {len(args.run_dirs)} replicates")

    delta_by_member = load_delta_stability_manifest(args.delta_stability_manifest)
    builder = ConsensusBuilder(
        args.centroid_threshold,
        args.lining_overlap,
        delta_by_member=delta_by_member,
        rank_mode=args.rank_mode,
    )
    result = builder.build(args.run_dirs, args.target_name)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "consensus_sites.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # consensus_gate_summary.json — aggregate gate attribution
    gate_summary = {
        "target": args.target_name,
        "n_replicates": result.n_replicates,
        "n_consensus_sites": result.n_consensus,
        "sites": [],
    }
    for cs in result.consensus_sites:
        gate_summary["sites"].append({
            "cluster_id": cs.cluster_id,
            "persistence": cs.persistence,
            "pass_fraction": cs.pass_fraction,
            "mean_delta_stability": cs.mean_delta_stability,
            "mean_delta_rms": cs.mean_delta_rms,
            "gate_failure_reasons": cs.gate_failure_reasons,
            "n_members": len(cs.member_sites),
            "n_passed": sum(1 for m in cs.member_sites if m.gate_passed),
            "n_blocked": sum(1 for m in cs.member_sites if not m.gate_passed),
        })
    with open(out / "consensus_gate_summary.json", "w") as f:
        json.dump(gate_summary, f, indent=2)

    # consensus_design_briefs/ — one brief per consensus site
    briefs_dir = out / "consensus_design_briefs"
    briefs_dir.mkdir(parents=True, exist_ok=True)
    for cs in result.consensus_sites:
        brief = {
            "cluster_id": cs.cluster_id,
            "centroid": list(cs.centroid_mean),
            "persistence": cs.persistence,
            "pass_fraction": cs.pass_fraction,
            "centroid_variance": cs.centroid_variance,
            "mean_quality_score": cs.mean_quality_score,
            "mean_contact_reorg": cs.mean_contact_reorg,
            "mean_response_sharpness": cs.mean_response_sharpness,
            "mean_localization": cs.mean_localization,
            "anchor_consistency": cs.anchor_consistency,
            "growth_vector_consistency": cs.growth_vector_consistency,
            "lining_consistency": cs.lining_consistency,
            "mean_delta_stability": cs.mean_delta_stability,
            "mean_delta_rms": cs.mean_delta_rms,
            "gate_failure_reasons": cs.gate_failure_reasons,
            "n_members": len(cs.member_sites),
            "member_runs": [m.run_id for m in cs.member_sites],
        }
        with open(briefs_dir / f"consensus_site_{cs.cluster_id}.json", "w") as f:
            json.dump(brief, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"CONSENSUS: {args.target_name}")
    print(f"  Replicates: {result.n_replicates}")
    print(f"  Consensus sites: {result.n_consensus}")
    print()
    print(f"  {'Rank':>4} {'Persist':>7} {'Pass%':>6} {'Var(A)':>6} "
          f"{'DeltaS':>7} {'mGTCKL':>7} {'mCR':>6} {'AnchJ':>6} {'LinJ':>6} {'Members':>7}")
    print(f"  {'-'*72}")
    for cs in result.consensus_sites[:20]:
        delta_s = cs.mean_delta_stability
        delta_s_str = f"{delta_s:>7.3f}" if delta_s is not None else f"{'NA':>7}"
        print(f"  {cs.cluster_id:>4} {cs.persistence:>7.2f} "
              f"{cs.pass_fraction:>6.2f} {cs.centroid_variance:>6.2f} "
              f"{delta_s_str} "
              f"{cs.mean_quality_score:>7.4f} {cs.mean_contact_reorg:>6.4f} "
              f"{cs.anchor_consistency:>6.2f} {cs.lining_consistency:>6.2f} "
              f"{len(cs.member_sites):>7}")
        if cs.gate_failure_reasons:
            reasons = ", ".join(f"{k}:{v}" for k, v in cs.gate_failure_reasons.items())
            print(f"       failures: {reasons}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
