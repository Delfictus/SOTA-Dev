#!/usr/bin/env python3
"""
coherence_recluster.py

Offline coherence-first reclustering for existing PRISM engine output files.

Debug-enabled and relaxed-threshold version.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

# Default coherence sweep winner for 9ig2_chainA
MAX_SITE_DIST = 12.0
MIN_ACTIVE_STEPS = 50
MIN_LAG_CORR = 0.20
MIN_LOCAL_COV = 0.05
MIN_WEIGHT = -0.25
MAX_PAIR_DIST = 9.0
MIN_COSINE = -0.20
MAX_LAG_DIFF = 24.0
MAX_CORR_DIFF = 0.35
MIN_EDGE_SCORE = 0.30
MIN_CLUSTER_SIZE = 3
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("This script requires numpy. Install with: pip install numpy") from exc


@dataclass
class ResidueKCC:
    residue_id: int
    residue_name: str
    ca_position: np.ndarray
    active_causal_steps: int
    burst_motion: float
    causal_lag: float
    direction_score: float
    kcc_score: float
    lag_corr_peak: float
    local_cov: float
    motion_efficiency: float
    net_dx: float
    net_dy: float
    net_dz: float

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.net_dx, self.net_dy, self.net_dz], dtype=float)


@dataclass
class SiteInput:
    site_id: int
    rank: Optional[int]
    rank_score: Optional[float]
    centroid: np.ndarray
    residue_ids: List[int]
    classification: Optional[str]
    druggability: Optional[float]
    uv_enrichment_score: Optional[float]
    therm_class: Optional[str]
    volume: Optional[float]


@dataclass
class ClusterMetrics:
    site_id: int
    cluster_id: int
    n_residues: int
    residue_ids: List[int]
    centroid: List[float]
    mean_radius: float
    max_diameter: float
    mean_cosine: float
    lag_std: float
    mean_lag_corr: float
    mean_local_cov: float
    mean_active_causal_steps: float
    mean_kcc_score: float
    mean_motion_efficiency: float
    score_signal: float
    score_vector: float
    score_geometry: float
    score_temporal: float
    score_chem: float
    total_score: float
    passes_hard: bool
    soft_penalty: float
    gt_dcc: Optional[float] = None


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else eps


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = safe_norm(a) * safe_norm(b)
    return float(np.dot(a, b) / denom)


def robust_z(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 1e-12:
        std = float(np.std(arr))
        if std < 1e-12:
            return [0.0 for _ in values]
        z = (arr - np.mean(arr)) / std
    else:
        z = 0.6745 * (arr - med) / mad
    z = np.clip(z, -3.0, 3.0)
    return [float(x) for x in z]


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def pairwise_indices(n: int):
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def find_one(target_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(target_dir.glob(pattern))
    return matches[0] if matches else None


def resolve_files(target_dir: Path) -> Dict[str, Optional[Path]]:
    return {
        "binding_sites": find_one(target_dir, "*.binding_sites.json"),
        "kcc_visualization": find_one(target_dir, "*.kcc_visualization.json"),
        "ground_truth": find_one(target_dir, "*_ground_truth.json"),
        "kcc_validation": find_one(target_dir, "*.kcc_validation.json"),
        "prism_therm": find_one(target_dir, "*.topology.prism_therm.json"),
        "residue_map": find_one(target_dir, "*.residue_map.json"),
    }


def parse_residues(kcc_vis: dict) -> Dict[int, ResidueKCC]:
    residues = {}
    for r in kcc_vis.get("residues", []):
        residue_id = int(r["residue_id"])
        residues[residue_id] = ResidueKCC(
            residue_id=residue_id,
            residue_name=str(r.get("residue_name", "")),
            ca_position=np.array(r["ca_position"], dtype=float),
            active_causal_steps=int(r.get("active_causal_steps", 0)),
            burst_motion=float(r.get("burst_motion", 0.0)),
            causal_lag=float(r.get("causal_lag", 0.0)),
            direction_score=float(r.get("direction_score", 0.0)),
            kcc_score=float(r.get("kcc_score", 0.0)),
            lag_corr_peak=float(r.get("lag_corr_peak", 0.0)),
            local_cov=float(r.get("local_cov", 0.0)),
            motion_efficiency=float(r.get("motion_efficiency", 0.0)),
            net_dx=float(r.get("net_dx", 0.0)),
            net_dy=float(r.get("net_dy", 0.0)),
            net_dz=float(r.get("net_dz", 0.0)),
        )
    return residues


def parse_sites(binding_sites: dict) -> List[SiteInput]:
    out = []
    for s in binding_sites.get("sites", []):
        site_id = int(s.get("id", s.get("site_id", -1)))
        residue_ids = [int(x) for x in s.get("residue_ids", [])]
        out.append(
            SiteInput(
                site_id=site_id,
                rank=int(s["rank"]) if "rank" in s else None,
                rank_score=float(s["rank_score"]) if "rank_score" in s else None,
                centroid=np.array(s["centroid"], dtype=float),
                residue_ids=residue_ids,
                classification=s.get("classification"),
                druggability=float(s["druggability"]) if "druggability" in s else None,
                uv_enrichment_score=float(s["uv_enrichment_score"]) if "uv_enrichment_score" in s else None,
                therm_class=s.get("therm_class"),
                volume=float(s["volume"]) if "volume" in s else None,
            )
        )
    return out


def parse_gt_centroid(gt: Optional[dict]) -> Optional[np.ndarray]:
    if not gt:
        return None
    centroid = gt.get("ligand_centroid")
    if centroid is None:
        return None
    return np.array(centroid, dtype=float)


def compute_candidate_weights(residues: List[ResidueKCC]) -> Dict[int, float]:
    lag_corr_vals = [r.lag_corr_peak for r in residues]
    local_cov_vals = [r.local_cov for r in residues]
    active_vals = [math.log1p(r.active_causal_steps) for r in residues]
    kcc_vals = [r.kcc_score for r in residues]
    motion_vals = [r.motion_efficiency for r in residues]

    z_lag = dict(zip([r.residue_id for r in residues], robust_z(lag_corr_vals)))
    z_cov = dict(zip([r.residue_id for r in residues], robust_z(local_cov_vals)))
    z_act = dict(zip([r.residue_id for r in residues], robust_z(active_vals)))
    z_kcc = dict(zip([r.residue_id for r in residues], robust_z(kcc_vals)))
    z_mot = dict(zip([r.residue_id for r in residues], robust_z(motion_vals)))

    weights = {}
    for r in residues:
        w = (
            0.35 * z_lag[r.residue_id]
            + 0.20 * z_cov[r.residue_id]
            + 0.15 * z_act[r.residue_id]
            + 0.20 * z_kcc[r.residue_id]
            + 0.10 * z_mot[r.residue_id]
        )
        weights[r.residue_id] = float(w)
    return weights


def filter_candidates(site: SiteInput, residues_by_id: Dict[int, ResidueKCC]) -> Tuple[List[ResidueKCC], Dict[int, float]]:
    site_res = [residues_by_id[rid] for rid in site.residue_ids if rid in residues_by_id]
    if not site_res:
        return [], {}

    weights = compute_candidate_weights(site_res)
    cand = []

    for r in site_res:
        if euclidean(r.ca_position, site.centroid) > 12.0:
            continue
        if r.active_causal_steps < 50:
            continue
        if r.lag_corr_peak < 0.20:
            continue
        if r.local_cov < 0.05:
            continue
        if weights.get(r.residue_id, -999.0) < -0.25:
            continue
        cand.append(r)

    return cand, weights


def edge_score(a: ResidueKCC, b: ResidueKCC) -> Optional[float]:
    d = euclidean(a.ca_position, b.ca_position)
    c = cosine(a.vector, b.vector)
    lag_diff = abs(a.causal_lag - b.causal_lag)
    corr_diff = abs(a.lag_corr_peak - b.lag_corr_peak)

    if d > 9.0:
        return None
    if c < -0.05:
        return None
    if lag_diff > 18.0:
        return None
    if corr_diff > 0.35:
        return None

    e = (
        0.35 * (1.0 - d / 9.0)
        + 0.30 * max(-0.05, c)
        + 0.20 * (1.0 - lag_diff / 18.0)
        + 0.15 * (1.0 - corr_diff / 0.35)
    )

    return float(e) if e >= 0.30 else None


def connected_components(candidates: List[ResidueKCC]) -> List[List[ResidueKCC]]:
    n = len(candidates)
    adj = [[] for _ in range(n)]

    edge_count = 0
    for i, j in pairwise_indices(n):
        e = edge_score(candidates[i], candidates[j])
        if e is not None:
            adj[i].append(j)
            adj[j].append(i)
            edge_count += 1

    print(f"[DEBUG] graph: candidates={n} edges={edge_count}")

    seen = [False] * n
    comps = []

    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp_idx = []
        while stack:
            u = stack.pop()
            comp_idx.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append([candidates[k] for k in comp_idx])

    return comps


def raw_cluster_metrics(cluster: List[ResidueKCC]) -> Dict[str, float]:
    pos = np.array([r.ca_position for r in cluster], dtype=float)
    ctr = np.mean(pos, axis=0)

    radii = [euclidean(r.ca_position, ctr) for r in cluster]
    mean_radius = float(np.mean(radii)) if radii else 0.0

    max_diameter = 0.0
    cosines = []
    lag_vals = [r.causal_lag for r in cluster]

    for i, j in pairwise_indices(len(cluster)):
        max_diameter = max(max_diameter, euclidean(cluster[i].ca_position, cluster[j].ca_position))
        cosines.append(cosine(cluster[i].vector, cluster[j].vector))

    mean_cosine = float(np.mean(cosines)) if cosines else 1.0
    lag_std = float(np.std(lag_vals)) if lag_vals else 0.0

    return {
        "mean_radius": mean_radius,
        "max_diameter": max_diameter,
        "mean_cosine": mean_cosine,
        "lag_std": lag_std,
    }


def split_cluster_if_needed(cluster: List[ResidueKCC]) -> List[List[ResidueKCC]]:
    metrics = raw_cluster_metrics(cluster)
    if (
        metrics["mean_radius"] <= 6.5
        and metrics["max_diameter"] <= 14.0
        and metrics["mean_cosine"] >= 0.0
    ):
        return [cluster]

    if len(cluster) < 6:
        return [cluster]

    max_d = -1.0
    piv_a = 0
    piv_b = 1
    for i, j in pairwise_indices(len(cluster)):
        d = euclidean(cluster[i].ca_position, cluster[j].ca_position)
        if d > max_d:
            max_d = d
            piv_a, piv_b = i, j

    A = cluster[piv_a].ca_position
    B = cluster[piv_b].ca_position
    left, right = [], []

    for r in cluster:
        da = euclidean(r.ca_position, A)
        db = euclidean(r.ca_position, B)
        if da <= db:
            left.append(r)
        else:
            right.append(r)

    if len(left) < 3 or len(right) < 3:
        return [cluster]

    out = []
    for child in (left, right):
        child_metrics = raw_cluster_metrics(child)
        if (
            child_metrics["mean_radius"] > 6.5
            or child_metrics["max_diameter"] > 14.0
            or child_metrics["mean_cosine"] < 0.0
        ) and len(child) >= 6:
            out.extend(split_cluster_if_needed(child))
        else:
            out.append(child)
    return out


def weighted_centroid(cluster: List[ResidueKCC], weights: Dict[int, float]) -> np.ndarray:
    ws = []
    xs = []
    for r in cluster:
        w = max(0.05, weights.get(r.residue_id, 0.05))
        ws.append(w)
        xs.append(r.ca_position * w)
    total_w = sum(ws)
    if total_w <= 0:
        return np.mean([r.ca_position for r in cluster], axis=0)
    return np.sum(xs, axis=0) / total_w


def cluster_metrics(
    site: SiteInput,
    cluster_id: int,
    cluster: List[ResidueKCC],
    weights: Dict[int, float],
    gt_centroid: Optional[np.ndarray],
) -> ClusterMetrics:
    ctr = weighted_centroid(cluster, weights)
    radii = [euclidean(r.ca_position, ctr) for r in cluster]
    mean_radius = float(np.mean(radii)) if radii else 0.0

    max_diameter = 0.0
    cosines = []
    for i, j in pairwise_indices(len(cluster)):
        max_diameter = max(max_diameter, euclidean(cluster[i].ca_position, cluster[j].ca_position))
        cosines.append(cosine(cluster[i].vector, cluster[j].vector))
    mean_cosine = float(np.mean(cosines)) if cosines else 1.0

    lag_vals = [r.causal_lag for r in cluster]
    lag_std = float(np.std(lag_vals)) if lag_vals else 0.0

    mean_lag_corr = float(np.mean([r.lag_corr_peak for r in cluster]))
    mean_local_cov = float(np.mean([r.local_cov for r in cluster]))
    mean_active_causal_steps = float(np.mean([r.active_causal_steps for r in cluster]))
    mean_kcc_score = float(np.mean([r.kcc_score for r in cluster]))
    mean_motion_efficiency = float(np.mean([r.motion_efficiency for r in cluster]))

    passes_hard = True
    if len(cluster) < 3:
        passes_hard = False
    if mean_radius > 7.0:
        passes_hard = False
    if max_diameter > 15.0:
        passes_hard = False
    if mean_cosine < -0.05:
        passes_hard = False
    if lag_std > 14.0:
        passes_hard = False
    if mean_active_causal_steps < 100.0:
        passes_hard = False

    soft_penalty = 0.0
    if 6.5 < mean_radius <= 7.0:
        soft_penalty += 0.05
    if 14.0 < max_diameter <= 15.0:
        soft_penalty += 0.05
    if -0.05 <= mean_cosine < 0.10:
        soft_penalty += 0.08
    if 12.0 < lag_std <= 14.0:
        soft_penalty += 0.05
    if 100.0 <= mean_active_causal_steps < 200.0:
        soft_penalty += 0.04

    score_signal = sigmoid(
        1.2 * mean_lag_corr
        + 0.8 * mean_local_cov
        + 0.25 * math.log1p(mean_active_causal_steps)
        + 0.6 * mean_kcc_score
    )

    score_vector = clip01((mean_cosine + 0.10) / 0.70)

    score_geometry = (
        0.5 * clip01(1.0 - mean_radius / 7.0)
        + 0.5 * clip01(1.0 - max_diameter / 15.0)
    )

    score_temporal = clip01(1.0 - lag_std / 12.0)

    chem_terms = []
    if site.uv_enrichment_score is not None:
        chem_terms.append(clip01(site.uv_enrichment_score))
    if site.druggability is not None:
        chem_terms.append(clip01(site.druggability))
    if site.volume is not None:
        chem_terms.append(clip01(math.log1p(site.volume) / math.log(2500.0)))
    score_chem = float(np.mean(chem_terms)) if chem_terms else 0.5

    total_score = (
        0.28 * score_signal
        + 0.24 * score_vector
        + 0.22 * score_geometry
        + 0.14 * score_temporal
        + 0.12 * score_chem
    ) - soft_penalty

    gt_dcc = None
    if gt_centroid is not None:
        gt_dcc = euclidean(ctr, gt_centroid)

    return ClusterMetrics(
        site_id=site.site_id,
        cluster_id=cluster_id,
        n_residues=len(cluster),
        residue_ids=sorted([r.residue_id for r in cluster]),
        centroid=[float(x) for x in ctr],
        mean_radius=mean_radius,
        max_diameter=max_diameter,
        mean_cosine=mean_cosine,
        lag_std=lag_std,
        mean_lag_corr=mean_lag_corr,
        mean_local_cov=mean_local_cov,
        mean_active_causal_steps=mean_active_causal_steps,
        mean_kcc_score=mean_kcc_score,
        mean_motion_efficiency=mean_motion_efficiency,
        score_signal=score_signal,
        score_vector=score_vector,
        score_geometry=score_geometry,
        score_temporal=score_temporal,
        score_chem=score_chem,
        total_score=total_score,
        passes_hard=passes_hard,
        soft_penalty=soft_penalty,
        gt_dcc=gt_dcc,
    )


def recluster_site(
    site: SiteInput,
    residues_by_id: Dict[int, ResidueKCC],
    gt_centroid: Optional[np.ndarray],
) -> List[ClusterMetrics]:
    candidates, weights = filter_candidates(site, residues_by_id)
    print(f"[DEBUG] site {site.site_id}: original residues={len(site.residue_ids)} candidates={len(candidates)}")

    if len(candidates) < 3:
        return []

    initial_components = connected_components(candidates)
    print(f"[DEBUG] site {site.site_id}: connected components={len(initial_components)} sizes={[len(c) for c in initial_components[:10]]}")

    clusters = []
    cluster_id = 1
    for comp in initial_components:
        if len(comp) < 3:
            continue
        split_children = split_cluster_if_needed(comp)
        print(f"[DEBUG] site {site.site_id}: component size={len(comp)} -> children sizes={[len(x) for x in split_children]}")

        for child in split_children:
            if len(child) < 3:
                continue
            metrics = cluster_metrics(site, cluster_id, child, weights, gt_centroid)
            print(
                f"[DEBUG] site {site.site_id} cluster {cluster_id}: "
                f"n={metrics.n_residues} radius={metrics.mean_radius:.2f} "
                f"diam={metrics.max_diameter:.2f} cos={metrics.mean_cosine:.3f} "
                f"lag_std={metrics.lag_std:.2f} pass={metrics.passes_hard}"
            )
            clusters.append(metrics)
            cluster_id += 1

    clusters.sort(key=lambda x: x.total_score, reverse=True)
    return clusters


def summarize_original_sites(sites: List[SiteInput], gt_centroid: Optional[np.ndarray]) -> List[dict]:
    out = []
    for s in sites:
        gt_dcc = euclidean(s.centroid, gt_centroid) if gt_centroid is not None else None
        out.append({
            "site_id": s.site_id,
            "rank": s.rank,
            "rank_score": s.rank_score,
            "classification": s.classification,
            "volume": s.volume,
            "n_residues": len(s.residue_ids),
            "gt_dcc": gt_dcc,
        })
    out.sort(key=lambda x: (999999 if x["rank"] is None else x["rank"]))
    return out


def print_top_original(original: List[dict], top_n: int = 10) -> None:
    print("\n=== Original top sites ===")
    for row in original[:top_n]:
        dcc_str = f"{row['gt_dcc']:.2f}" if row["gt_dcc"] is not None else "NA"
        print(
            f"site={row['site_id']:>3}  rank={str(row['rank']):>3}  "
            f"score={row['rank_score'] if row['rank_score'] is not None else 'NA':>8}  "
            f"nres={row['n_residues']:>3}  vol={row['volume'] if row['volume'] is not None else 'NA':>8}  "
            f"gt_dcc={dcc_str:>6}  class={row['classification']}"
        )


def print_top_reclustered(all_clusters: List[ClusterMetrics], top_n: int = 15) -> None:
    print("\n=== Reclustered top clusters ===")
    for i, c in enumerate(all_clusters[:top_n], start=1):
        dcc_str = f"{c.gt_dcc:.2f}" if c.gt_dcc is not None else "NA"
        print(
            f"new_rank={i:>3}  parent_site={c.site_id:>3}  cluster={c.cluster_id:>2}  "
            f"score={c.total_score:>6.3f}  pass={str(c.passes_hard):>5}  "
            f"nres={c.n_residues:>3}  radius={c.mean_radius:>5.2f}  "
            f"diam={c.max_diameter:>5.2f}  cos={c.mean_cosine:>6.3f}  "
            f"lag_std={c.lag_std:>5.2f}  gt_dcc={dcc_str:>6}"
        )


def choose_best_by_gt_dcc(items: List[dict]) -> Optional[dict]:
    candidates = [x for x in items if x.get("gt_dcc") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x["gt_dcc"])


def choose_best_cluster_by_gt_dcc(items: List[ClusterMetrics]) -> Optional[ClusterMetrics]:
    candidates = [x for x in items if x.gt_dcc is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x.gt_dcc if x.gt_dcc is not None else 1e9)


def write_report(
    out_path: Path,
    files: Dict[str, Optional[Path]],
    original_sites: List[dict],
    all_clusters: List[ClusterMetrics],
    best_orig: Optional[dict],
    best_new: Optional[ClusterMetrics],
) -> None:
    payload = {
        "source_files": {k: (str(v) if v else None) for k, v in files.items()},
        "original_sites": original_sites,
        "reclustered_clusters": [asdict(c) for c in all_clusters],
        "summary": {
            "n_original_sites": len(original_sites),
            "n_reclustered_clusters": len(all_clusters),
            "best_original_by_gt_dcc": best_orig,
            "best_reclustered_by_gt_dcc": asdict(best_new) if best_new else None,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline coherence-first reclustering for PRISM output files.")
    p.add_argument("target_dir", type=Path, help="Directory containing one target's JSON outputs.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSON report path. Default: <target_dir>/coherence_recluster_report.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target_dir: Path = args.target_dir

    if not target_dir.exists() or not target_dir.is_dir():
        raise SystemExit(f"Target directory does not exist or is not a directory: {target_dir}")

    files = resolve_files(target_dir)
    if files["binding_sites"] is None:
        raise SystemExit("Missing *.binding_sites.json")
    if files["kcc_visualization"] is None:
        raise SystemExit("Missing *.kcc_visualization.json")

    binding_sites = load_json(files["binding_sites"])
    kcc_vis = load_json(files["kcc_visualization"])
    gt = load_json(files["ground_truth"]) if files["ground_truth"] else None

    residues_by_id = parse_residues(kcc_vis)
    sites = parse_sites(binding_sites)
    gt_centroid = parse_gt_centroid(gt)

    original_sites = summarize_original_sites(sites, gt_centroid)
    print_top_original(original_sites, top_n=10)

    all_clusters: List[ClusterMetrics] = []
    for site in sites:
        site_clusters = recluster_site(site, residues_by_id, gt_centroid)
        all_clusters.extend(site_clusters)

    all_clusters.sort(key=lambda x: x.total_score, reverse=True)
    print_top_reclustered(all_clusters, top_n=15)

    best_orig = choose_best_by_gt_dcc(original_sites)
    best_new = choose_best_cluster_by_gt_dcc(all_clusters)

    print("\n=== GT-DCC comparison ===")
    if best_orig is not None:
        print(
            f"Best original: site={best_orig['site_id']}  rank={best_orig['rank']}  "
            f"gt_dcc={best_orig['gt_dcc']:.2f}"
        )
    else:
        print("Best original: no ground truth available")

    if best_new is not None:
        recluster_rank = all_clusters.index(best_new) + 1
        print(
            f"Best reclustered: parent_site={best_new.site_id}  cluster={best_new.cluster_id}  "
            f"new_rank={recluster_rank}  gt_dcc={best_new.gt_dcc:.2f}  "
            f"pass={best_new.passes_hard}"
        )
    else:
        print("Best reclustered: no ground truth available")

    out_path = args.out or (target_dir / "coherence_recluster_report.json")
    write_report(out_path, files, original_sites, all_clusters, best_orig, best_new)
    print(f"\nWrote report to: {out_path}")


if __name__ == "__main__":
    main()
