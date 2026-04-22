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
class Config:
    # Cluster construction
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

    # Pocket mapping
    expand_neighbor_dist: float = 8.0
    expand_centroid_dist: float = 10.0
    expand_lag_sigma: float = 8.0
    expand_corr_sigma: float = 0.20
    expand_min_membership: float = 0.42
    max_expand_residues: int = 24

    # Final centroid weighting
    seed_weight_floor: float = 1.0
    added_weight_floor: float = 0.15


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


def gaussian_sim(delta: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    return float(math.exp(-0.5 * (delta / sigma) ** 2))


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
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_indices(n: int):
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


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


def filter_candidates(site: SiteInput, residues_by_id: Dict[int, ResidueKCC], cfg: Config) -> Tuple[List[ResidueKCC], Dict[int, float]]:
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


def edge_score(a: ResidueKCC, b: ResidueKCC, cfg: Config) -> Optional[float]:
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


def connected_components(candidates: List[ResidueKCC], cfg: Config) -> List[List[ResidueKCC]]:
    n = len(candidates)
    adj = [[] for _ in range(n)]
    for i, j in pairwise_indices(n):
        e = edge_score(candidates[i], candidates[j], cfg)
        if e is not None:
            adj[i].append(j)
            adj[j].append(i)

    seen = [False] * n
    comps = []
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
    return comps


def weighted_centroid(points: List[np.ndarray], weights: List[float]) -> np.ndarray:
    w = np.array(weights, dtype=float)
    x = np.array(points, dtype=float)
    sw = float(np.sum(w))
    if sw <= 1e-12:
        return np.mean(x, axis=0)
    return np.sum(x * w[:, None], axis=0) / sw


def compute_seed_weights(cluster: List[ResidueKCC], cand_weights: Dict[int, float], cfg: Config) -> Dict[int, float]:
    out = {}
    for r in cluster:
        base = max(cfg.seed_weight_floor, 1.0 + cand_weights.get(r.residue_id, 0.0))
        signal = (
            0.30 * clip01(r.lag_corr_peak) +
            0.20 * clip01(r.local_cov) +
            0.20 * clip01(r.kcc_score) +
            0.15 * clip01(math.log1p(r.active_causal_steps) / 6.0) +
            0.15 * clip01(r.motion_efficiency)
        )
        out[r.residue_id] = max(cfg.seed_weight_floor, base * (1.0 + signal))
    return out


def map_cluster_to_pocket(
    cluster: List[ResidueKCC],
    site_residues: List[ResidueKCC],
    cand_weights: Dict[int, float],
    cfg: Config,
) -> Tuple[List[ResidueKCC], Dict[int, float], np.ndarray, np.ndarray]:
    cluster_ids = {r.residue_id for r in cluster}
    seed_weights = compute_seed_weights(cluster, cand_weights, cfg)

    raw_cluster_centroid = weighted_centroid(
        [r.ca_position for r in cluster],
        [seed_weights[r.residue_id] for r in cluster],
    )

    mean_lag = float(np.mean([r.causal_lag for r in cluster]))
    mean_corr = float(np.mean([r.lag_corr_peak for r in cluster]))
    mean_cov = float(np.mean([r.local_cov for r in cluster]))
    mean_kcc = float(np.mean([r.kcc_score for r in cluster]))
    mean_vec = np.mean([r.vector for r in cluster], axis=0)
    if np.linalg.norm(mean_vec) < 1e-12:
        mean_vec = np.array([1.0, 0.0, 0.0], dtype=float)

    candidates = []
    for r in site_residues:
        if r.residue_id in cluster_ids:
            continue

        min_seed_dist = min(euclidean(r.ca_position, s.ca_position) for s in cluster)
        ctr_dist = euclidean(r.ca_position, raw_cluster_centroid)

        if min_seed_dist > cfg.expand_neighbor_dist:
            continue
        if ctr_dist > cfg.expand_centroid_dist:
            continue

        dist_aff = 0.55 * clip01(1.0 - min_seed_dist / cfg.expand_neighbor_dist) + \
                   0.45 * clip01(1.0 - ctr_dist / cfg.expand_centroid_dist)

        lag_aff = gaussian_sim(abs(r.causal_lag - mean_lag), cfg.expand_lag_sigma)
        corr_aff = gaussian_sim(abs(r.lag_corr_peak - mean_corr), cfg.expand_corr_sigma)

        vec_aff = clip01((cosine(r.vector, mean_vec) + 1.0) / 2.0)

        signal_aff = (
            0.30 * clip01(r.lag_corr_peak) +
            0.20 * clip01(r.local_cov) +
            0.20 * clip01(r.kcc_score) +
            0.15 * clip01(math.log1p(r.active_causal_steps) / 6.0) +
            0.15 * clip01(r.motion_efficiency)
        )

        profile_aff = (
            0.35 * lag_aff +
            0.25 * corr_aff +
            0.20 * vec_aff +
            0.20 * signal_aff
        )

        membership = 0.55 * dist_aff + 0.45 * profile_aff

        candidates.append({
            "residue": r,
            "membership": membership,
            "dist_aff": dist_aff,
            "profile_aff": profile_aff,
            "vec_aff": vec_aff,
            "lag_aff": lag_aff,
            "corr_aff": corr_aff,
            "signal_aff": signal_aff,
        })

    candidates = sorted(candidates, key=lambda x: x["membership"], reverse=True)

    added = []
    added_weights = {}
    for c in candidates:
        if c["membership"] < cfg.expand_min_membership:
            continue
        if len(added) >= cfg.max_expand_residues:
            break
        r = c["residue"]
        added.append(r)
        added_weights[r.residue_id] = max(cfg.added_weight_floor, c["membership"])

    final_residues = list(cluster) + added
    final_weights = {}
    final_weights.update(seed_weights)
    final_weights.update(added_weights)

    mapped_centroid = weighted_centroid(
        [r.ca_position for r in final_residues],
        [final_weights[r.residue_id] for r in final_residues],
    )

    return final_residues, final_weights, raw_cluster_centroid, mapped_centroid


def cluster_internal_score(cluster: List[ResidueKCC]) -> float:
    n = len(cluster)
    if n == 0:
        return 0.0

    ctr = np.mean([r.ca_position for r in cluster], axis=0)
    radii = [euclidean(r.ca_position, ctr) for r in cluster]
    mean_radius = float(np.mean(radii)) if radii else 0.0

    cosines = []
    lag_diffs = []
    for i, j in pairwise_indices(n):
        cosines.append(cosine(cluster[i].vector, cluster[j].vector))
        lag_diffs.append(abs(cluster[i].causal_lag - cluster[j].causal_lag))

    mean_cos = float(np.mean(cosines)) if cosines else 1.0
    lag_std = float(np.std([r.causal_lag for r in cluster]))
    mean_lag_diff = float(np.mean(lag_diffs)) if lag_diffs else 0.0
    mean_lag_corr = float(np.mean([r.lag_corr_peak for r in cluster]))
    mean_local_cov = float(np.mean([r.local_cov for r in cluster]))
    mean_kcc = float(np.mean([r.kcc_score for r in cluster]))
    mean_active = float(np.mean([r.active_causal_steps for r in cluster]))

    score_signal = sigmoid(
        1.2 * mean_lag_corr +
        0.8 * mean_local_cov +
        0.6 * mean_kcc +
        0.25 * math.log1p(mean_active)
    )
    score_vector = clip01((mean_cos + 0.20) / 0.80)
    score_compact = clip01(1.0 - mean_radius / 7.5)
    score_temporal = 0.5 * clip01(1.0 - lag_std / 16.0) + 0.5 * clip01(1.0 - mean_lag_diff / 16.0)

    return (
        0.32 * score_signal +
        0.24 * score_vector +
        0.22 * score_compact +
        0.22 * score_temporal
    )


def analyze_target(target_dir: Path, cfg: Config) -> List[dict]:
    files = resolve_files(target_dir)
    binding_sites = load_json(files["binding_sites"])
    kcc_vis = load_json(files["kcc_visualization"])
    gt = load_json(files["ground_truth"]) if files["ground_truth"] else None

    residues_by_id = parse_residues(kcc_vis)
    sites = parse_sites(binding_sites)
    gt_centroid = parse_gt_centroid(gt)

    rows = []

    for site in sites:
        site_residues = [residues_by_id[rid] for rid in site.residue_ids if rid in residues_by_id]
        if not site_residues:
            continue

        candidates, cand_weights = filter_candidates(site, residues_by_id, cfg)
        if len(candidates) < cfg.min_cluster_size:
            continue

        comps = connected_components(candidates, cfg)
        cluster_id = 1
        for comp in comps:
            if len(comp) < cfg.min_cluster_size:
                continue

            expanded_residues, expanded_weights, raw_ctr, mapped_ctr = map_cluster_to_pocket(
                cluster=comp,
                site_residues=site_residues,
                cand_weights=cand_weights,
                cfg=cfg,
            )

            raw_dcc = euclidean(raw_ctr, gt_centroid) if gt_centroid is not None else None
            mapped_dcc = euclidean(mapped_ctr, gt_centroid) if gt_centroid is not None else None
            dcc_gain = (raw_dcc - mapped_dcc) if (raw_dcc is not None and mapped_dcc is not None) else None

            internal_score = cluster_internal_score(comp)

            rows.append({
                "target": target_dir.name,
                "site_id": site.site_id,
                "site_rank": site.rank,
                "site_rank_score": site.rank_score,
                "site_classification": site.classification,
                "site_druggability": site.druggability,
                "site_uv_enrichment_score": site.uv_enrichment_score,
                "site_volume": site.volume,
                "cluster_id": cluster_id,
                "seed_n": len(comp),
                "mapped_n": len(expanded_residues),
                "n_added": len(expanded_residues) - len(comp),
                "cluster_internal_score": internal_score,
                "raw_centroid_x": float(raw_ctr[0]),
                "raw_centroid_y": float(raw_ctr[1]),
                "raw_centroid_z": float(raw_ctr[2]),
                "mapped_centroid_x": float(mapped_ctr[0]),
                "mapped_centroid_y": float(mapped_ctr[1]),
                "mapped_centroid_z": float(mapped_ctr[2]),
                "raw_dcc": raw_dcc,
                "mapped_dcc": mapped_dcc,
                "dcc_gain": dcc_gain,
                "improved": (dcc_gain is not None and dcc_gain > 0.0),
                "good_raw_dcc_5": (raw_dcc is not None and raw_dcc <= 5.0),
                "good_mapped_dcc_5": (mapped_dcc is not None and mapped_dcc <= 5.0),
            })
            cluster_id += 1

    return rows


def topk_hit_rate(rows: List[dict], score_key: str, dcc_key: str, k: int, cutoff: float = 5.0) -> Optional[float]:
    by_target: Dict[str, List[dict]] = {}
    for r in rows:
        if r[dcc_key] is None:
            continue
        by_target.setdefault(r["target"], []).append(r)

    if not by_target:
        return None

    hits = 0
    total = 0
    for target, items in by_target.items():
        items = sorted(items, key=lambda x: x[score_key], reverse=True)
        top = items[:k]
        total += 1
        if any((x[dcc_key] is not None and x[dcc_key] <= cutoff) for x in top):
            hits += 1

    return hits / total if total else None


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[dict], cfg: Config) -> dict:
    with_gt = [r for r in rows if r["raw_dcc"] is not None and r["mapped_dcc"] is not None]
    raw_dcc = [r["raw_dcc"] for r in with_gt]
    mapped_dcc = [r["mapped_dcc"] for r in with_gt]
    gains = [r["dcc_gain"] for r in with_gt if r["dcc_gain"] is not None]

    by_target = {}
    for r in with_gt:
        by_target.setdefault(r["target"], []).append(r)

    per_target_best = []
    improved_targets = 0
    for target, items in by_target.items():
        best_raw = min(items, key=lambda x: x["raw_dcc"])
        best_mapped = min(items, key=lambda x: x["mapped_dcc"])
        gain = best_raw["raw_dcc"] - best_mapped["mapped_dcc"]
        if gain > 0:
            improved_targets += 1
        per_target_best.append({
            "target": target,
            "best_raw_dcc": best_raw["raw_dcc"],
            "best_mapped_dcc": best_mapped["mapped_dcc"],
            "best_target_gain": gain,
            "best_raw_site_id": best_raw["site_id"],
            "best_mapped_site_id": best_mapped["site_id"],
        })

    per_target_best = sorted(per_target_best, key=lambda x: x["best_target_gain"], reverse=True)

    cluster_score = [r["cluster_internal_score"] for r in with_gt]

    return {
        "config": asdict(cfg),
        "n_rows": len(rows),
        "n_rows_with_gt": len(with_gt),
        "n_targets_with_gt": len(by_target),
        "mean_raw_dcc": float(np.mean(raw_dcc)) if raw_dcc else None,
        "mean_mapped_dcc": float(np.mean(mapped_dcc)) if mapped_dcc else None,
        "median_raw_dcc": float(np.median(raw_dcc)) if raw_dcc else None,
        "median_mapped_dcc": float(np.median(mapped_dcc)) if mapped_dcc else None,
        "mean_dcc_gain": float(np.mean(gains)) if gains else None,
        "median_dcc_gain": float(np.median(gains)) if gains else None,
        "improved_fraction": float(np.mean([1.0 if g > 0 else 0.0 for g in gains])) if gains else None,
        "improved_targets_fraction": (improved_targets / len(by_target)) if by_target else None,
        "cluster_score_vs_raw_dcc_spearman": spearman_corr(cluster_score, raw_dcc),
        "cluster_score_vs_mapped_dcc_spearman": spearman_corr(cluster_score, mapped_dcc),
        "top1_raw_hit_dcc_le_5": topk_hit_rate(with_gt, "cluster_internal_score", "raw_dcc", 1, 5.0),
        "top1_mapped_hit_dcc_le_5": topk_hit_rate(with_gt, "cluster_internal_score", "mapped_dcc", 1, 5.0),
        "top3_raw_hit_dcc_le_5": topk_hit_rate(with_gt, "cluster_internal_score", "raw_dcc", 3, 5.0),
        "top3_mapped_hit_dcc_le_5": topk_hit_rate(with_gt, "cluster_internal_score", "mapped_dcc", 3, 5.0),
        "top5_raw_hit_dcc_le_5": topk_hit_rate(with_gt, "cluster_internal_score", "raw_dcc", 5, 5.0),
        "top5_mapped_hit_dcc_le_5": topk_hit_rate(with_gt, "cluster_internal_score", "mapped_dcc", 5, 5.0),
        "per_target_best": per_target_best[:50],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Map mechanistic spike clusters into pocket hypotheses.")
    ap.add_argument("root", type=Path)
    ap.add_argument("--out-prefix", default="cluster_to_pocket")
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
    ap.add_argument("--expand-neighbor-dist", type=float, default=8.0)
    ap.add_argument("--expand-centroid-dist", type=float, default=10.0)
    ap.add_argument("--expand-lag-sigma", type=float, default=8.0)
    ap.add_argument("--expand-corr-sigma", type=float, default=0.20)
    ap.add_argument("--expand-min-membership", type=float, default=0.42)
    ap.add_argument("--max-expand-residues", type=int, default=24)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
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
        expand_neighbor_dist=args.expand_neighbor_dist,
        expand_centroid_dist=args.expand_centroid_dist,
        expand_lag_sigma=args.expand_lag_sigma,
        expand_corr_sigma=args.expand_corr_sigma,
        expand_min_membership=args.expand_min_membership,
        max_expand_residues=args.max_expand_residues,
    )

    target_dirs = list_target_dirs(args.root)
    if not target_dirs:
        raise SystemExit(f"No valid target directories found under: {args.root}")

    all_rows = []
    for i, td in enumerate(target_dirs, start=1):
        print(f"[{i}/{len(target_dirs)}] {td.name}")
        try:
            all_rows.extend(analyze_target(td, cfg))
        except Exception as e:
            print(f"  ! skipped {td.name}: {e}")

    summary = summarize(all_rows, cfg)

    csv_path = Path(f"{args.out_prefix}.clusters.csv")
    json_path = Path(f"{args.out_prefix}.summary.json")

    write_csv(all_rows, csv_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"n_rows: {summary['n_rows']}")
    print(f"n_rows_with_gt: {summary['n_rows_with_gt']}")
    print(f"n_targets_with_gt: {summary['n_targets_with_gt']}")
    print(f"mean_raw_dcc: {summary['mean_raw_dcc']}")
    print(f"mean_mapped_dcc: {summary['mean_mapped_dcc']}")
    print(f"median_raw_dcc: {summary['median_raw_dcc']}")
    print(f"median_mapped_dcc: {summary['median_mapped_dcc']}")
    print(f"mean_dcc_gain: {summary['mean_dcc_gain']}")
    print(f"median_dcc_gain: {summary['median_dcc_gain']}")
    print(f"improved_fraction: {summary['improved_fraction']}")
    print(f"improved_targets_fraction: {summary['improved_targets_fraction']}")
    print(f"cluster_score_vs_raw_dcc_spearman: {summary['cluster_score_vs_raw_dcc_spearman']}")
    print(f"cluster_score_vs_mapped_dcc_spearman: {summary['cluster_score_vs_mapped_dcc_spearman']}")
    print(f"top1_raw_hit_dcc<=5: {summary['top1_raw_hit_dcc_le_5']}")
    print(f"top1_mapped_hit_dcc<=5: {summary['top1_mapped_hit_dcc_le_5']}")
    print(f"top3_raw_hit_dcc<=5: {summary['top3_raw_hit_dcc_le_5']}")
    print(f"top3_mapped_hit_dcc<=5: {summary['top3_mapped_hit_dcc_le_5']}")
    print(f"top5_raw_hit_dcc<=5: {summary['top5_raw_hit_dcc_le_5']}")
    print(f"top5_mapped_hit_dcc<=5: {summary['top5_mapped_hit_dcc_le_5']}")

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
