#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Data structures
# -----------------------------

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
class SweepConfig:
    max_site_dist: float = 12.0
    min_active_steps: int = 50
    min_lag_corr: float = 0.20
    min_local_cov: float = 0.05
    min_weight: float = -0.25
    max_pair_dist: float = 9.0
    min_cosine: float = -0.20
    max_lag_diff: float = 24.0
    max_corr_diff: float = 0.35
    min_edge_score: float = 0.30
    min_cluster_size: int = 3


# -----------------------------
# Utilities
# -----------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_one(target_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(target_dir.glob(pattern))
    return matches[0] if matches else None


def resolve_files(target_dir: Path) -> Dict[str, Optional[Path]]:
    return {
        "binding_sites": find_one(target_dir, "*.binding_sites.json"),
        "kcc_visualization": find_one(target_dir, "*.kcc_visualization.json"),
        "ground_truth": find_one(target_dir, "*_ground_truth.json"),
    }


def is_target_dir(path: Path) -> bool:
    files = resolve_files(path)
    return files["binding_sites"] is not None and files["kcc_visualization"] is not None


def list_target_dirs(root: Path) -> List[Path]:
    if is_target_dir(root):
        return [root]
    return sorted([p for p in root.iterdir() if p.is_dir() and is_target_dir(p)])


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    n = float(np.linalg.norm(v))
    return n if n > eps else eps


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (safe_norm(a) * safe_norm(b)))


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


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


def pairwise_indices(n: int):
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def rankdata_average(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman_corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3 or len(y) < 3:
        return None
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    if np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return None
    rx = rankdata_average(xa)
    ry = rankdata_average(ya)
    c = np.corrcoef(rx, ry)[0, 1]
    return float(c)


def pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3 or len(y) < 3:
        return None
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    if np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return None
    c = np.corrcoef(xa, ya)[0, 1]
    return float(c)


def topk_hit_rate(rows: List[dict], k: int, dcc_cutoff: float = 5.0) -> Optional[float]:
    by_target: Dict[str, List[dict]] = {}
    for r in rows:
        if r["gt_dcc"] is None:
            continue
        by_target.setdefault(r["target"], []).append(r)
    if not by_target:
        return None

    hits = 0
    total = 0
    for target, items in by_target.items():
        items = sorted(items, key=lambda r: r["cluster_score"], reverse=True)
        top = items[:k]
        total += 1
        if any((x["gt_dcc"] is not None and x["gt_dcc"] <= dcc_cutoff) for x in top):
            hits += 1
    return hits / total if total else None


# -----------------------------
# Parsing
# -----------------------------

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
    return np.array(centroid, dtype=float) if centroid is not None else None


# -----------------------------
# Candidate filtering
# -----------------------------

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

    out = {}
    for r in residues:
        out[r.residue_id] = (
            0.35 * z_lag[r.residue_id]
            + 0.20 * z_cov[r.residue_id]
            + 0.15 * z_act[r.residue_id]
            + 0.20 * z_kcc[r.residue_id]
            + 0.10 * z_mot[r.residue_id]
        )
    return out


def filter_candidates(site: SiteInput, residues_by_id: Dict[int, ResidueKCC], cfg: SweepConfig) -> Tuple[List[ResidueKCC], Dict[int, float]]:
    site_res = [residues_by_id[rid] for rid in site.residue_ids if rid in residues_by_id]
    if not site_res:
        return [], {}

    weights = compute_candidate_weights(site_res)
    cand = []
    for r in site_res:
        if euclidean(r.ca_position, site.centroid) > cfg.max_site_dist:
            continue
        if r.active_causal_steps < cfg.min_active_steps:
            continue
        if r.lag_corr_peak < cfg.min_lag_corr:
            continue
        if r.local_cov < cfg.min_local_cov:
            continue
        if weights.get(r.residue_id, -999.0) < cfg.min_weight:
            continue
        cand.append(r)
    return cand, weights


# -----------------------------
# Graph and clustering
# -----------------------------

def edge_score(a: ResidueKCC, b: ResidueKCC, cfg: SweepConfig) -> Optional[float]:
    d = euclidean(a.ca_position, b.ca_position)
    c = cosine(a.vector, b.vector)
    lag_diff = abs(a.causal_lag - b.causal_lag)
    corr_diff = abs(a.lag_corr_peak - b.lag_corr_peak)

    if d > cfg.max_pair_dist:
        return None
    if c < cfg.min_cosine:
        return None
    if lag_diff > cfg.max_lag_diff:
        return None
    if corr_diff > cfg.max_corr_diff:
        return None

    c_floor = cfg.min_cosine
    c_range = max(1e-6, 1.0 - c_floor)
    edge = (
        0.40 * (1.0 - d / cfg.max_pair_dist)
        + 0.10 * ((c - c_floor) / c_range)
        + 0.30 * (1.0 - lag_diff / cfg.max_lag_diff)
        + 0.20 * (1.0 - corr_diff / cfg.max_corr_diff)
    )
    return float(edge) if edge >= cfg.min_edge_score else None


def build_graph(cluster_candidates: List[ResidueKCC], cfg: SweepConfig) -> Tuple[List[List[int]], int]:
    n = len(cluster_candidates)
    adj = [[] for _ in range(n)]
    edge_count = 0
    for i, j in pairwise_indices(n):
        e = edge_score(cluster_candidates[i], cluster_candidates[j], cfg)
        if e is not None:
            adj[i].append(j)
            adj[j].append(i)
            edge_count += 1
    return adj, edge_count


def connected_components(candidates: List[ResidueKCC], cfg: SweepConfig) -> Tuple[List[List[ResidueKCC]], List[List[int]], int]:
    adj, edge_count = build_graph(candidates, cfg)
    n = len(candidates)
    seen = [False] * n
    comps = []
    comp_indices = []

    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        idxs = []
        while stack:
            u = stack.pop()
            idxs.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append([candidates[k] for k in idxs])
        comp_indices.append(idxs)

    return comps, comp_indices, edge_count


# -----------------------------
# Cluster metrics
# -----------------------------

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


def compute_cluster_row(
    target: str,
    site: SiteInput,
    cluster_id: int,
    cluster: List[ResidueKCC],
    comp_indices: List[int],
    all_candidates: List[ResidueKCC],
    weights: Dict[int, float],
    cfg: SweepConfig,
    gt_centroid: Optional[np.ndarray],
) -> dict:
    ctr = weighted_centroid(cluster, weights)
    n = len(cluster)

    radii = [euclidean(r.ca_position, ctr) for r in cluster]
    mean_radius = float(np.mean(radii)) if radii else 0.0
    max_radius = float(np.max(radii)) if radii else 0.0

    cosines = []
    dists = []
    lag_diffs = []
    corr_diffs = []
    for i, j in pairwise_indices(n):
        cosines.append(cosine(cluster[i].vector, cluster[j].vector))
        dists.append(euclidean(cluster[i].ca_position, cluster[j].ca_position))
        lag_diffs.append(abs(cluster[i].causal_lag - cluster[j].causal_lag))
        corr_diffs.append(abs(cluster[i].lag_corr_peak - cluster[j].lag_corr_peak))

    mean_pair_dist = float(np.mean(dists)) if dists else 0.0
    max_diameter = float(np.max(dists)) if dists else 0.0
    mean_cosine = float(np.mean(cosines)) if cosines else 1.0
    frac_negative_cos = float(np.mean([1.0 if c < 0 else 0.0 for c in cosines])) if cosines else 0.0
    lag_std = float(np.std([r.causal_lag for r in cluster])) if cluster else 0.0
    lag_mean = float(np.mean([r.causal_lag for r in cluster])) if cluster else 0.0
    mean_lag_diff = float(np.mean(lag_diffs)) if lag_diffs else 0.0
    mean_corr_diff = float(np.mean(corr_diffs)) if corr_diffs else 0.0

    mean_lag_corr = float(np.mean([r.lag_corr_peak for r in cluster]))
    mean_local_cov = float(np.mean([r.local_cov for r in cluster]))
    mean_active = float(np.mean([r.active_causal_steps for r in cluster]))
    mean_kcc = float(np.mean([r.kcc_score for r in cluster]))
    mean_motion_eff = float(np.mean([r.motion_efficiency for r in cluster]))
    mean_direction_score = float(np.mean([r.direction_score for r in cluster]))
    mean_burst_motion = float(np.mean([r.burst_motion for r in cluster]))

    norms = [float(np.linalg.norm(r.vector)) for r in cluster]
    mean_vec_norm = float(np.mean(norms)) if norms else 0.0

    # Graph density within cluster
    local_adj, edge_count = build_graph(cluster, cfg)
    possible_edges = n * (n - 1) / 2
    edge_density = float(edge_count / possible_edges) if possible_edges > 0 else 0.0

    degrees = [len(a) for a in local_adj]
    mean_degree = float(np.mean(degrees)) if degrees else 0.0
    min_degree = float(np.min(degrees)) if degrees else 0.0

    # Isolation / separation
    outside = [r for r in all_candidates if r.residue_id not in {x.residue_id for x in cluster}]
    if outside:
        nearest_outside = min(euclidean(r.ca_position, s.ca_position) for r in cluster for s in outside)
        separation_ratio = float(nearest_outside / max(mean_radius, 1e-6))
    else:
        nearest_outside = float("nan")
        separation_ratio = float("nan")

    # Mechanistic score: internal-only objective
    score_signal = sigmoid(
        1.2 * mean_lag_corr +
        0.8 * mean_local_cov +
        0.25 * math.log1p(mean_active) +
        0.6 * mean_kcc
    )
    score_vector = clip01((mean_cosine + 0.20) / 0.80)
    score_geometry = (
        0.45 * clip01(1.0 - mean_radius / 7.5) +
        0.35 * clip01(1.0 - max_diameter / 16.0) +
        0.20 * clip01(edge_density)
    )
    score_temporal = (
        0.5 * clip01(1.0 - lag_std / 16.0) +
        0.5 * clip01(1.0 - mean_lag_diff / max(cfg.max_lag_diff, 1e-6))
    )
    score_isolation = 0.0 if math.isnan(separation_ratio) else clip01(separation_ratio / 3.0)

    cluster_score = (
        0.30 * score_signal +
        0.22 * score_vector +
        0.20 * score_geometry +
        0.18 * score_temporal +
        0.10 * score_isolation
    )

    gt_dcc = euclidean(ctr, gt_centroid) if gt_centroid is not None else None
    is_good_dcc_2 = (gt_dcc is not None and gt_dcc <= 2.0)
    is_good_dcc_3 = (gt_dcc is not None and gt_dcc <= 3.0)
    is_good_dcc_5 = (gt_dcc is not None and gt_dcc <= 5.0)

    return {
        "target": target,
        "site_id": site.site_id,
        "site_rank": site.rank,
        "site_rank_score": site.rank_score,
        "site_classification": site.classification,
        "site_druggability": site.druggability,
        "site_uv_enrichment_score": site.uv_enrichment_score,
        "site_volume": site.volume,
        "cluster_id": cluster_id,
        "n_residues": n,
        "centroid_x": float(ctr[0]),
        "centroid_y": float(ctr[1]),
        "centroid_z": float(ctr[2]),
        "mean_radius": mean_radius,
        "max_radius": max_radius,
        "max_diameter": max_diameter,
        "mean_pair_dist": mean_pair_dist,
        "edge_density": edge_density,
        "mean_degree": mean_degree,
        "min_degree": min_degree,
        "mean_cosine": mean_cosine,
        "frac_negative_cos": frac_negative_cos,
        "lag_std": lag_std,
        "lag_mean": lag_mean,
        "mean_lag_diff": mean_lag_diff,
        "mean_corr_diff": mean_corr_diff,
        "mean_lag_corr": mean_lag_corr,
        "mean_local_cov": mean_local_cov,
        "mean_active_causal_steps": mean_active,
        "mean_kcc_score": mean_kcc,
        "mean_motion_efficiency": mean_motion_eff,
        "mean_direction_score": mean_direction_score,
        "mean_burst_motion": mean_burst_motion,
        "mean_vector_norm": mean_vec_norm,
        "nearest_outside_dist": None if math.isnan(nearest_outside) else nearest_outside,
        "separation_ratio": None if math.isnan(separation_ratio) else separation_ratio,
        "cluster_score": cluster_score,
        "gt_dcc": gt_dcc,
        "good_dcc_2": is_good_dcc_2,
        "good_dcc_3": is_good_dcc_3,
        "good_dcc_5": is_good_dcc_5,
    }


# -----------------------------
# Main panel analysis
# -----------------------------

def analyze_target(target_dir: Path, cfg: SweepConfig) -> List[dict]:
    files = resolve_files(target_dir)
    binding_sites = load_json(files["binding_sites"])
    kcc_vis = load_json(files["kcc_visualization"])
    gt = load_json(files["ground_truth"]) if files["ground_truth"] else None

    residues_by_id = parse_residues(kcc_vis)
    sites = parse_sites(binding_sites)
    gt_centroid = parse_gt_centroid(gt)

    rows = []

    for site in sites:
        candidates, weights = filter_candidates(site, residues_by_id, cfg)
        if len(candidates) < cfg.min_cluster_size:
            continue

        comps, comp_indices, _ = connected_components(candidates, cfg)
        cluster_id = 1
        for comp, idxs in zip(comps, comp_indices):
            if len(comp) < cfg.min_cluster_size:
                continue
            row = compute_cluster_row(
                target=target_dir.name,
                site=site,
                cluster_id=cluster_id,
                cluster=comp,
                comp_indices=idxs,
                all_candidates=candidates,
                weights=weights,
                cfg=cfg,
                gt_centroid=gt_centroid,
            )
            rows.append(row)
            cluster_id += 1

    return rows


def summarize(rows: List[dict], cfg: SweepConfig) -> dict:
    rows_with_dcc = [r for r in rows if r["gt_dcc"] is not None]
    metric_names = [
        "cluster_score",
        "n_residues",
        "mean_radius",
        "max_diameter",
        "edge_density",
        "mean_degree",
        "mean_cosine",
        "frac_negative_cos",
        "lag_std",
        "mean_lag_diff",
        "mean_corr_diff",
        "mean_lag_corr",
        "mean_local_cov",
        "mean_active_causal_steps",
        "mean_kcc_score",
        "mean_motion_efficiency",
        "mean_direction_score",
        "mean_burst_motion",
        "mean_vector_norm",
        "separation_ratio",
        "site_druggability",
        "site_uv_enrichment_score",
        "site_volume",
    ]

    corr_table = []
    for m in metric_names:
        xs = [r[m] for r in rows_with_dcc if r[m] is not None]
        ys = [r["gt_dcc"] for r in rows_with_dcc if r[m] is not None]
        if len(xs) < 3:
            continue
        corr_table.append({
            "metric": m,
            "spearman_vs_dcc": spearman_corr(xs, ys),
            "pearson_vs_dcc": pearson_corr(xs, ys),
        })

    corr_table = sorted(
        corr_table,
        key=lambda x: abs(x["spearman_vs_dcc"]) if x["spearman_vs_dcc"] is not None else -1,
        reverse=True
    )

    hit1_5 = topk_hit_rate(rows_with_dcc, 1, 5.0)
    hit3_5 = topk_hit_rate(rows_with_dcc, 3, 5.0)
    hit5_5 = topk_hit_rate(rows_with_dcc, 5, 5.0)

    dcc_vals = [r["gt_dcc"] for r in rows_with_dcc]
    score_vals = [r["cluster_score"] for r in rows_with_dcc]

    by_target = {}
    for r in rows_with_dcc:
        by_target.setdefault(r["target"], []).append(r)

    per_target_best = []
    for target, items in by_target.items():
        best_by_score = max(items, key=lambda x: x["cluster_score"])
        best_by_dcc = min(items, key=lambda x: x["gt_dcc"])
        per_target_best.append({
            "target": target,
            "best_cluster_score": best_by_score["cluster_score"],
            "best_cluster_score_dcc": best_by_score["gt_dcc"],
            "oracle_best_dcc": best_by_dcc["gt_dcc"],
            "best_cluster_site_id": best_by_score["site_id"],
            "oracle_site_id": best_by_dcc["site_id"],
        })

    return {
        "config": asdict(cfg),
        "n_clusters_total": len(rows),
        "n_clusters_with_gt": len(rows_with_dcc),
        "n_targets_with_gt": len(by_target),
        "cluster_score_vs_dcc_spearman": spearman_corr(score_vals, dcc_vals),
        "cluster_score_vs_dcc_pearson": pearson_corr(score_vals, dcc_vals),
        "top1_hit_dcc_le_5": hit1_5,
        "top3_hit_dcc_le_5": hit3_5,
        "top5_hit_dcc_le_5": hit5_5,
        "metric_correlations": corr_table[:20],
        "per_target_best": per_target_best,
    }


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mechanistic cluster-objective analysis across a panel.")
    ap.add_argument("root", type=Path, help="Panel root or single target directory")
    ap.add_argument("--out-prefix", default="cluster_objective_analysis")
    ap.add_argument("--max-site-dist", type=float, default=12.0)
    ap.add_argument("--min-active-steps", type=int, default=50)
    ap.add_argument("--min-lag-corr", type=float, default=0.20)
    ap.add_argument("--min-local-cov", type=float, default=0.05)
    ap.add_argument("--min-weight", type=float, default=-0.25)
    ap.add_argument("--max-pair-dist", type=float, default=9.0)
    ap.add_argument("--min-cosine", type=float, default=-0.20)
    ap.add_argument("--max-lag-diff", type=float, default=24.0)
    ap.add_argument("--max-corr-diff", type=float, default=0.35)
    ap.add_argument("--min-edge-score", type=float, default=0.30)
    ap.add_argument("--min-cluster-size", type=int, default=3)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SweepConfig(
        max_site_dist=args.max_site_dist,
        min_active_steps=args.min_active_steps,
        min_lag_corr=args.min_lag_corr,
        min_local_cov=args.min_local_cov,
        min_weight=args.min_weight,
        max_pair_dist=args.max_pair_dist,
        min_cosine=args.min_cosine,
        max_lag_diff=args.max_lag_diff,
        max_corr_diff=args.max_corr_diff,
        min_edge_score=args.min_edge_score,
        min_cluster_size=args.min_cluster_size,
    )

    target_dirs = list_target_dirs(args.root)
    if not target_dirs:
        raise SystemExit(f"No valid target directories found under: {args.root}")

    all_rows = []
    for i, td in enumerate(target_dirs, start=1):
        print(f"[{i}/{len(target_dirs)}] {td.name}")
        try:
            rows = analyze_target(td, cfg)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ! skipped {td.name}: {e}")

    summary = summarize(all_rows, cfg)

    csv_path = Path(f"{args.out_prefix}.clusters.csv")
    json_path = Path(f"{args.out_prefix}.summary.json")

    write_csv(all_rows, csv_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"clusters_total: {summary['n_clusters_total']}")
    print(f"clusters_with_gt: {summary['n_clusters_with_gt']}")
    print(f"targets_with_gt: {summary['n_targets_with_gt']}")
    print(f"cluster_score_vs_dcc_spearman: {summary['cluster_score_vs_dcc_spearman']}")
    print(f"cluster_score_vs_dcc_pearson:  {summary['cluster_score_vs_dcc_pearson']}")
    print(f"top1_hit_dcc<=5: {summary['top1_hit_dcc_le_5']}")
    print(f"top3_hit_dcc<=5: {summary['top3_hit_dcc_le_5']}")
    print(f"top5_hit_dcc<=5: {summary['top5_hit_dcc_le_5']}")

    print("\n=== TOP METRIC CORRELATIONS VS DCC ===")
    for row in summary["metric_correlations"][:12]:
        print(
            f"{row['metric']:<28} "
            f"spearman={row['spearman_vs_dcc']}  "
            f"pearson={row['pearson_vs_dcc']}"
        )

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
