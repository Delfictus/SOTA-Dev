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
    # cluster construction
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

    # void inference
    pair_min_dist: float = 4.0
    pair_max_dist: float = 14.0
    inward_shift: float = 2.5
    candidate_merge_radius: float = 2.0
    max_candidates_per_cluster: int = 24
    support_shell: float = 8.0
    candidate_site_max_dist: float = 12.0

    # scoring
    support_min_cos: float = -0.2
    support_lag_sigma: float = 10.0
    support_corr_sigma: float = 0.25


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


def unit(v: np.ndarray) -> np.ndarray:
    return np.array(v, dtype=float) / safe_norm(v)


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


def cluster_seed_centroid(cluster: List[ResidueKCC], cand_weights: Dict[int, float]) -> np.ndarray:
    ws = []
    xs = []
    for r in cluster:
        w = max(0.2, 1.0 + cand_weights.get(r.residue_id, 0.0))
        bonus = (
            0.30 * clip01(r.lag_corr_peak) +
            0.20 * clip01(r.local_cov) +
            0.20 * clip01(r.kcc_score) +
            0.15 * clip01(math.log1p(r.active_causal_steps) / 6.0) +
            0.15 * clip01(r.motion_efficiency)
        )
        w *= (1.0 + bonus)
        ws.append(w)
        xs.append(r.ca_position)
    return weighted_centroid(xs, ws)


def make_void_candidates(
    cluster: List[ResidueKCC],
    site: SiteInput,
    cfg: Config,
    cand_weights: Dict[int, float],
) -> List[np.ndarray]:
    raw_ctr = cluster_seed_centroid(cluster, cand_weights)
    cluster_geom_ctr = np.mean([r.ca_position for r in cluster], axis=0)
    mean_vec = np.mean([r.vector for r in cluster], axis=0)
    if np.linalg.norm(mean_vec) < 1e-12:
        mean_vec = site.centroid - cluster_geom_ctr
    if np.linalg.norm(mean_vec) < 1e-12:
        mean_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    mean_vec = unit(mean_vec)

    candidates = []
    n = len(cluster)

    # candidate 1: raw weighted cluster centroid
    candidates.append(raw_ctr)

    # candidate 2: site centroid anchor
    candidates.append(site.centroid)

    # candidate 3: shifted cluster center toward site centroid
    toward_site = unit(site.centroid - raw_ctr)
    candidates.append(raw_ctr + 1.5 * toward_site)

    # residue-pair cavity proxies
    for i, j in pairwise_indices(n):
        a = cluster[i]
        b = cluster[j]
        d = euclidean(a.ca_position, b.ca_position)
        if d < cfg.pair_min_dist or d > cfg.pair_max_dist:
            continue

        midpoint = 0.5 * (a.ca_position + b.ca_position)

        # inward axis toward cluster center / site center
        inward1 = unit(raw_ctr - a.ca_position)
        inward2 = unit(raw_ctr - b.ca_position)
        inward = unit(inward1 + inward2)

        # pair opening direction
        pair_axis = unit(b.ca_position - a.ca_position)

        # avoid projecting along the bond itself
        ortho = inward - np.dot(inward, pair_axis) * pair_axis
        if np.linalg.norm(ortho) < 1e-8:
            ortho = toward_site
        ortho = unit(ortho)

        # pair coherence gating
        c = cosine(a.vector, b.vector)
        lag_ok = abs(a.causal_lag - b.causal_lag) <= cfg.max_lag_diff
        if c < cfg.support_min_cos or not lag_ok:
            continue

        # 3 candidate centers from this pair
        candidates.append(midpoint)
        candidates.append(midpoint + cfg.inward_shift * ortho)
        candidates.append(midpoint + 0.5 * cfg.inward_shift * toward_site)

    # residue-local cavity proxies
    for r in cluster:
        inward = unit(raw_ctr - r.ca_position)
        if np.linalg.norm(inward) < 1e-8:
            inward = toward_site
        candidates.append(r.ca_position + cfg.inward_shift * inward)

    # merge near-duplicate candidates
    merged: List[np.ndarray] = []
    for c in candidates:
        if euclidean(c, site.centroid) > cfg.candidate_site_max_dist:
            continue
        placed = False
        for k in range(len(merged)):
            if euclidean(c, merged[k]) <= cfg.candidate_merge_radius:
                merged[k] = 0.5 * (merged[k] + c)
                placed = True
                break
        if not placed:
            merged.append(c)
        if len(merged) >= cfg.max_candidates_per_cluster:
            break

    return merged


def score_candidate_center(
    center: np.ndarray,
    cluster: List[ResidueKCC],
    site_residues: List[ResidueKCC],
    site: SiteInput,
    cfg: Config,
) -> Tuple[float, dict]:
    supporting = [r for r in site_residues if euclidean(r.ca_position, center) <= cfg.support_shell]
    if not supporting:
        return -1e9, {"n_support": 0}

    seed_support = [r for r in cluster if euclidean(r.ca_position, center) <= cfg.support_shell]
    if not seed_support:
        seed_support = cluster

    mean_lag = float(np.mean([r.causal_lag for r in cluster]))
    mean_corr = float(np.mean([r.lag_corr_peak for r in cluster]))
    mean_vec = np.mean([r.vector for r in cluster], axis=0)
    if np.linalg.norm(mean_vec) < 1e-12:
        mean_vec = site.centroid - center
    mean_vec = unit(mean_vec)

    dists = [euclidean(r.ca_position, center) for r in supporting]
    compactness = float(np.mean([clip01(1.0 - d / cfg.support_shell) for d in dists]))

    # inwardness: residues should roughly "face" the center
    inward_scores = []
    for r in supporting:
        to_center = unit(center - r.ca_position)
        inward_scores.append(clip01((cosine(r.vector, to_center) + 1.0) / 2.0))
    inwardness = float(np.mean(inward_scores)) if inward_scores else 0.0

    # support coherence
    lag_scores = [gaussian_sim(abs(r.causal_lag - mean_lag), cfg.support_lag_sigma) for r in supporting]
    corr_scores = [gaussian_sim(abs(r.lag_corr_peak - mean_corr), cfg.support_corr_sigma) for r in supporting]
    vec_scores = [clip01((cosine(r.vector, mean_vec) + 1.0) / 2.0) for r in supporting]
    support_coh = float(np.mean([
        0.40 * lag_scores[i] + 0.30 * corr_scores[i] + 0.30 * vec_scores[i]
        for i in range(len(supporting))
    ]))

    # signal support
    signal_scores = []
    for r in supporting:
        sig = (
            0.30 * clip01(r.lag_corr_peak) +
            0.20 * clip01(r.local_cov) +
            0.20 * clip01(r.kcc_score) +
            0.15 * clip01(math.log1p(r.active_causal_steps) / 6.0) +
            0.15 * clip01(r.motion_efficiency)
        )
        signal_scores.append(sig)
    signal_support = float(np.mean(signal_scores)) if signal_scores else 0.0

    # anchor to site envelope, but softly
    site_anchor = clip01(1.0 - euclidean(center, site.centroid) / max(cfg.candidate_site_max_dist, 1e-6))

    # pair-gap bonus: candidate should be inside support geometry, not on the rim
    center_to_support = [euclidean(center, r.ca_position) for r in supporting]
    if len(center_to_support) >= 2:
        mean_center_dist = float(np.mean(center_to_support))
        spread = float(np.std(center_to_support))
        cavity_bonus = clip01(mean_center_dist / 6.0) * clip01(1.0 - spread / 4.0)
    else:
        cavity_bonus = 0.0

    n_support = len(supporting)
    support_bonus = clip01(math.log1p(n_support) / math.log(12.0))

    score = (
        0.24 * compactness +
        0.20 * inwardness +
        0.18 * support_coh +
        0.16 * signal_support +
        0.10 * cavity_bonus +
        0.07 * site_anchor +
        0.05 * support_bonus
    )

    details = {
        "n_support": n_support,
        "compactness": compactness,
        "inwardness": inwardness,
        "support_coh": support_coh,
        "signal_support": signal_support,
        "cavity_bonus": cavity_bonus,
        "site_anchor": site_anchor,
        "support_bonus": support_bonus,
    }
    return score, details


def infer_best_void_center(
    cluster: List[ResidueKCC],
    site_residues: List[ResidueKCC],
    site: SiteInput,
    cfg: Config,
    cand_weights: Dict[int, float],
) -> Tuple[np.ndarray, float, dict, int]:
    candidates = make_void_candidates(cluster, site, cfg, cand_weights)
    if not candidates:
        ctr = cluster_seed_centroid(cluster, cand_weights)
        return ctr, 0.0, {"fallback": True}, 0

    best_c = None
    best_score = -1e18
    best_details = None
    for c in candidates:
        s, details = score_candidate_center(c, cluster, site_residues, site, cfg)
        if s > best_score:
            best_score = s
            best_c = c
            best_details = details

    return best_c, best_score, best_details, len(candidates)


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

            raw_ctr = cluster_seed_centroid(comp, cand_weights)
            void_ctr, void_score, details, n_candidates = infer_best_void_center(
                cluster=comp,
                site_residues=site_residues,
                site=site,
                cfg=cfg,
                cand_weights=cand_weights,
            )

            raw_dcc = euclidean(raw_ctr, gt_centroid) if gt_centroid is not None else None
            void_dcc = euclidean(void_ctr, gt_centroid) if gt_centroid is not None else None
            dcc_gain = (raw_dcc - void_dcc) if (raw_dcc is not None and void_dcc is not None) else None

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
                "cluster_internal_score": cluster_internal_score(comp),
                "n_void_candidates": n_candidates,
                "void_score": void_score,
                "raw_centroid_x": float(raw_ctr[0]),
                "raw_centroid_y": float(raw_ctr[1]),
                "raw_centroid_z": float(raw_ctr[2]),
                "void_centroid_x": float(void_ctr[0]),
                "void_centroid_y": float(void_ctr[1]),
                "void_centroid_z": float(void_ctr[2]),
                "raw_dcc": raw_dcc,
                "void_dcc": void_dcc,
                "dcc_gain": dcc_gain,
                "improved": (dcc_gain is not None and dcc_gain > 0.0),
                "good_raw_dcc_5": (raw_dcc is not None and raw_dcc <= 5.0),
                "good_void_dcc_5": (void_dcc is not None and void_dcc <= 5.0),
                "void_support_n": details.get("n_support"),
                "void_compactness": details.get("compactness"),
                "void_inwardness": details.get("inwardness"),
                "void_support_coh": details.get("support_coh"),
                "void_signal_support": details.get("signal_support"),
                "void_cavity_bonus": details.get("cavity_bonus"),
                "void_site_anchor": details.get("site_anchor"),
                "void_support_bonus": details.get("support_bonus"),
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
    with_gt = [r for r in rows if r["raw_dcc"] is not None and r["void_dcc"] is not None]
    raw_dcc = [r["raw_dcc"] for r in with_gt]
    void_dcc = [r["void_dcc"] for r in with_gt]
    gains = [r["dcc_gain"] for r in with_gt if r["dcc_gain"] is not None]

    by_target: Dict[str, List[dict]] = {}
    for r in with_gt:
        by_target.setdefault(r["target"], []).append(r)

    per_target_best = []
    improved_targets = 0
    for target, items in by_target.items():
        best_raw = min(items, key=lambda x: x["raw_dcc"])
        best_void = min(items, key=lambda x: x["void_dcc"])
        gain = best_raw["raw_dcc"] - best_void["void_dcc"]
        if gain > 0:
            improved_targets += 1
        per_target_best.append({
            "target": target,
            "best_raw_dcc": best_raw["raw_dcc"],
            "best_void_dcc": best_void["void_dcc"],
            "best_target_gain": gain,
            "best_raw_site_id": best_raw["site_id"],
            "best_void_site_id": best_void["site_id"],
        })
    per_target_best = sorted(per_target_best, key=lambda x: x["best_target_gain"], reverse=True)

    cluster_scores = [r["cluster_internal_score"] for r in with_gt]
    void_scores = [r["void_score"] for r in with_gt]

    return {
        "config": asdict(cfg),
        "n_rows": len(rows),
        "n_rows_with_gt": len(with_gt),
        "n_targets_with_gt": len(by_target),
        "mean_raw_dcc": float(np.mean(raw_dcc)) if raw_dcc else None,
        "mean_void_dcc": float(np.mean(void_dcc)) if void_dcc else None,
        "median_raw_dcc": float(np.median(raw_dcc)) if raw_dcc else None,
        "median_void_dcc": float(np.median(void_dcc)) if void_dcc else None,
        "mean_dcc_gain": float(np.mean(gains)) if gains else None,
        "median_dcc_gain": float(np.median(gains)) if gains else None,
        "improved_fraction": float(np.mean([1.0 if g > 0 else 0.0 for g in gains])) if gains else None,
        "improved_targets_fraction": (improved_targets / len(by_target)) if by_target else None,
        "cluster_score_vs_raw_dcc_spearman": spearman_corr(cluster_scores, raw_dcc),
        "cluster_score_vs_void_dcc_spearman": spearman_corr(cluster_scores, void_dcc),
        "void_score_vs_void_dcc_spearman": spearman_corr(void_scores, void_dcc),
        "top1_raw_hit_dcc_le_5": topk_hit_rate(with_gt, "cluster_internal_score", "raw_dcc", 1, 5.0),
        "top1_void_hit_dcc_le_5": topk_hit_rate(with_gt, "void_score", "void_dcc", 1, 5.0),
        "top3_raw_hit_dcc_le_5": topk_hit_rate(with_gt, "cluster_internal_score", "raw_dcc", 3, 5.0),
        "top3_void_hit_dcc_le_5": topk_hit_rate(with_gt, "void_score", "void_dcc", 3, 5.0),
        "top5_raw_hit_dcc_le_5": topk_hit_rate(with_gt, "cluster_internal_score", "raw_dcc", 5, 5.0),
        "top5_void_hit_dcc_le_5": topk_hit_rate(with_gt, "void_score", "void_dcc", 5, 5.0),
        "per_target_best": per_target_best[:50],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Infer void-centered pocket hypotheses from mechanistic spike clusters.")
    ap.add_argument("root", type=Path)
    ap.add_argument("--out-prefix", default="cluster_void_center")
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
    ap.add_argument("--pair-min-dist", type=float, default=4.0)
    ap.add_argument("--pair-max-dist", type=float, default=14.0)
    ap.add_argument("--inward-shift", type=float, default=2.5)
    ap.add_argument("--candidate-merge-radius", type=float, default=2.0)
    ap.add_argument("--max-candidates-per-cluster", type=int, default=24)
    ap.add_argument("--support-shell", type=float, default=8.0)
    ap.add_argument("--candidate-site-max-dist", type=float, default=12.0)
    ap.add_argument("--support-min-cos", type=float, default=-0.2)
    ap.add_argument("--support-lag-sigma", type=float, default=10.0)
    ap.add_argument("--support-corr-sigma", type=float, default=0.25)
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
        pair_min_dist=args.pair_min_dist,
        pair_max_dist=args.pair_max_dist,
        inward_shift=args.inward_shift,
        candidate_merge_radius=args.candidate_merge_radius,
        max_candidates_per_cluster=args.max_candidates_per_cluster,
        support_shell=args.support_shell,
        candidate_site_max_dist=args.candidate_site_max_dist,
        support_min_cos=args.support_min_cos,
        support_lag_sigma=args.support_lag_sigma,
        support_corr_sigma=args.support_corr_sigma,
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
    print(f"mean_void_dcc: {summary['mean_void_dcc']}")
    print(f"median_raw_dcc: {summary['median_raw_dcc']}")
    print(f"median_void_dcc: {summary['median_void_dcc']}")
    print(f"mean_dcc_gain: {summary['mean_dcc_gain']}")
    print(f"median_dcc_gain: {summary['median_dcc_gain']}")
    print(f"improved_fraction: {summary['improved_fraction']}")
    print(f"improved_targets_fraction: {summary['improved_targets_fraction']}")
    print(f"cluster_score_vs_raw_dcc_spearman: {summary['cluster_score_vs_raw_dcc_spearman']}")
    print(f"cluster_score_vs_void_dcc_spearman: {summary['cluster_score_vs_void_dcc_spearman']}")
    print(f"void_score_vs_void_dcc_spearman: {summary['void_score_vs_void_dcc_spearman']}")
    print(f"top1_raw_hit_dcc<=5: {summary['top1_raw_hit_dcc_le_5']}")
    print(f"top1_void_hit_dcc<=5: {summary['top1_void_hit_dcc_le_5']}")
    print(f"top3_raw_hit_dcc<=5: {summary['top3_raw_hit_dcc_le_5']}")
    print(f"top3_void_hit_dcc<=5: {summary['top3_void_hit_dcc_le_5']}")
    print(f"top5_raw_hit_dcc<=5: {summary['top5_raw_hit_dcc_le_5']}")
    print(f"top5_void_hit_dcc<=5: {summary['top5_void_hit_dcc_le_5']}")

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
