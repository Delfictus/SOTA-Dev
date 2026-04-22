#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
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
class SweepConfig:
    max_site_dist: float
    min_active_steps: int
    min_lag_corr: float
    min_local_cov: float
    min_weight: float
    max_pair_dist: float
    min_cosine: float
    max_lag_diff: float
    max_corr_diff: float
    min_edge_score: float
    min_cluster_size: int


@dataclass
class ClusterMetrics:
    site_id: int
    cluster_id: int
    n_residues: int
    centroid: List[float]
    mean_radius: float
    max_diameter: float
    mean_cosine: float
    lag_std: float
    mean_lag_corr: float
    mean_local_cov: float
    mean_active_causal_steps: float
    mean_kcc_score: float
    total_score: float
    passes_hard: bool
    gt_dcc: Optional[float]


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


def connected_components(candidates: List[ResidueKCC], cfg: SweepConfig) -> Tuple[List[List[ResidueKCC]], int]:
    n = len(candidates)
    adj = [[] for _ in range(n)]
    edge_count = 0

    for i, j in pairwise_indices(n):
        e = edge_score(candidates[i], candidates[j], cfg)
        if e is not None:
            adj[i].append(j)
            adj[j].append(i)
            edge_count += 1

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

    return comps, edge_count


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


def compute_cluster_metrics(
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
    mean_active = float(np.mean([r.active_causal_steps for r in cluster]))
    mean_kcc = float(np.mean([r.kcc_score for r in cluster]))

    passes_hard = True
    if len(cluster) < 3:
        passes_hard = False
    if mean_radius > 7.5:
        passes_hard = False
    if max_diameter > 16.0:
        passes_hard = False
    if mean_cosine < -0.20:
        passes_hard = False
    if lag_std > 16.0:
        passes_hard = False
    if mean_active < 80.0:
        passes_hard = False

    score_signal = sigmoid(
        1.2 * mean_lag_corr
        + 0.8 * mean_local_cov
        + 0.25 * math.log1p(mean_active)
        + 0.6 * mean_kcc
    )
    score_vector = clip01((mean_cosine + 0.20) / 0.80)
    score_geometry = (
        0.5 * clip01(1.0 - mean_radius / 7.5)
        + 0.5 * clip01(1.0 - max_diameter / 16.0)
    )
    score_temporal = clip01(1.0 - lag_std / 16.0)

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
        + 0.22 * score_vector
        + 0.22 * score_geometry
        + 0.14 * score_temporal
        + 0.14 * score_chem
    )

    gt_dcc = euclidean(ctr, gt_centroid) if gt_centroid is not None else None

    return ClusterMetrics(
        site_id=site.site_id,
        cluster_id=cluster_id,
        n_residues=len(cluster),
        centroid=[float(x) for x in ctr],
        mean_radius=mean_radius,
        max_diameter=max_diameter,
        mean_cosine=mean_cosine,
        lag_std=lag_std,
        mean_lag_corr=mean_lag_corr,
        mean_local_cov=mean_local_cov,
        mean_active_causal_steps=mean_active,
        mean_kcc_score=mean_kcc,
        total_score=total_score,
        passes_hard=passes_hard,
        gt_dcc=gt_dcc,
    )


def evaluate_target(target_dir: Path, cfg: SweepConfig) -> dict:
    files = resolve_files(target_dir)
    if files["binding_sites"] is None or files["kcc_visualization"] is None:
        return {"target": target_dir.name, "error": "missing required files"}

    binding_sites = load_json(files["binding_sites"])
    kcc_vis = load_json(files["kcc_visualization"])
    gt = load_json(files["ground_truth"]) if files["ground_truth"] else None

    residues_by_id = parse_residues(kcc_vis)
    sites = parse_sites(binding_sites)
    gt_centroid = parse_gt_centroid(gt)

    original_best_dcc = None
    if gt_centroid is not None:
        dccs = [euclidean(s.centroid, gt_centroid) for s in sites]
        if dccs:
            original_best_dcc = float(min(dccs))

    all_clusters = []
    total_candidates = 0
    total_edges = 0
    component_sizes = []

    for site in sites:
        candidates, weights = filter_candidates(site, residues_by_id, cfg)
        total_candidates += len(candidates)
        if len(candidates) < cfg.min_cluster_size:
            continue

        components, edge_count = connected_components(candidates, cfg)
        total_edges += edge_count
        component_sizes.extend([len(c) for c in components])

        cluster_id = 1
        for comp in components:
            if len(comp) < cfg.min_cluster_size:
                continue
            metrics = compute_cluster_metrics(site, cluster_id, comp, weights, gt_centroid)
            all_clusters.append(metrics)
            cluster_id += 1

    all_clusters.sort(key=lambda x: x.total_score, reverse=True)
    passing_clusters = [c for c in all_clusters if c.passes_hard]

    best_new_dcc = None
    if gt_centroid is not None and all_clusters:
        dccs = [c.gt_dcc for c in all_clusters if c.gt_dcc is not None]
        if dccs:
            best_new_dcc = float(min(dccs))

    best_pass_dcc = None
    if gt_centroid is not None and passing_clusters:
        dccs = [c.gt_dcc for c in passing_clusters if c.gt_dcc is not None]
        if dccs:
            best_pass_dcc = float(min(dccs))

    avg_component_size = float(np.mean(component_sizes)) if component_sizes else 0.0
    avg_cluster_size = float(np.mean([c.n_residues for c in all_clusters])) if all_clusters else 0.0
    avg_cosine = float(np.mean([c.mean_cosine for c in all_clusters])) if all_clusters else 0.0
    pass_rate = (len(passing_clusters) / len(all_clusters)) if all_clusters else 0.0

    dcc_gain = None
    improved = None
    if original_best_dcc is not None and best_new_dcc is not None:
        dcc_gain = original_best_dcc - best_new_dcc
        improved = dcc_gain > 0

    return {
        "target": target_dir.name,
        "config": asdict(cfg),
        "original_best_dcc": original_best_dcc,
        "best_new_dcc": best_new_dcc,
        "best_pass_dcc": best_pass_dcc,
        "dcc_gain": dcc_gain,
        "improved": improved,
        "n_clusters": len(all_clusters),
        "n_passing_clusters": len(passing_clusters),
        "pass_rate": pass_rate,
        "total_candidates": total_candidates,
        "total_edges": total_edges,
        "avg_component_size": avg_component_size,
        "avg_cluster_size": avg_cluster_size,
        "avg_cluster_cosine": avg_cosine,
        "top_clusters": [asdict(c) for c in all_clusters[:10]],
    }


def build_configs(mode: str) -> List[SweepConfig]:
    if mode == "focused":
        max_pair_dist_values = [9.0, 10.0, 11.0]
        min_cosine_values = [-0.20, -0.40]
        max_lag_diff_values = [18.0, 24.0, 30.0]
        min_edge_score_values = [0.30, 0.20, 0.15]
    elif mode == "tight":
        max_pair_dist_values = [8.0, 9.0, 10.0]
        min_cosine_values = [-0.20, 0.00]
        max_lag_diff_values = [18.0, 24.0]
        min_edge_score_values = [0.30, 0.40]
    elif mode == "broad":
        max_pair_dist_values = [8.0, 9.0, 10.0, 11.0, 12.0]
        min_cosine_values = [0.00, -0.20, -0.40]
        max_lag_diff_values = [12.0, 18.0, 24.0, 30.0]
        min_edge_score_values = [0.15, 0.20, 0.30, 0.40]
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    configs = []
    for max_pair_dist, min_cosine, max_lag_diff, min_edge_score in itertools.product(
        max_pair_dist_values,
        min_cosine_values,
        max_lag_diff_values,
        min_edge_score_values,
    ):
        configs.append(
            SweepConfig(
                max_site_dist=12.0,
                min_active_steps=50,
                min_lag_corr=0.20,
                min_local_cov=0.05,
                min_weight=-0.25,
                max_pair_dist=max_pair_dist,
                min_cosine=min_cosine,
                max_lag_diff=max_lag_diff,
                max_corr_diff=0.35,
                min_edge_score=min_edge_score,
                min_cluster_size=3,
            )
        )
    return configs


def list_target_dirs(root: Path) -> List[Path]:
    if is_target_dir(root):
        return [root]
    return sorted([p for p in root.iterdir() if p.is_dir() and is_target_dir(p)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-panel sweep across a target set such as PRT70.")
    ap.add_argument("root", type=Path, help="Root containing target subdirectories, or a single target directory")
    ap.add_argument("--mode", choices=["tight", "focused", "broad"], default="focused")
    ap.add_argument("--out", type=Path, default=Path("prt70_panel_sweep.json"))
    args = ap.parse_args()

    target_dirs = list_target_dirs(args.root)
    if not target_dirs:
        raise SystemExit(f"No valid target directories found under: {args.root}")

    configs = build_configs(args.mode)
    all_config_results = []
    per_target_best = {}

    print(f"[INFO] targets found: {len(target_dirs)}")
    print(f"[INFO] sweep mode: {args.mode}")
    print(f"[INFO] configs: {len(configs)}")

    for idx, cfg in enumerate(configs, start=1):
        print(
            f"[SWEEP] {idx}/{len(configs)} "
            f"pair_dist={cfg.max_pair_dist} min_cos={cfg.min_cosine} "
            f"lag_diff={cfg.max_lag_diff} edge={cfg.min_edge_score}"
        )

        per_target = []
        for target_dir in target_dirs:
            res = evaluate_target(target_dir, cfg)
            per_target.append(res)

            if "error" in res:
                continue

            tname = res["target"]
            current = per_target_best.get(tname)
            better = False
            if current is None:
                better = True
            else:
                curr_gain = current["dcc_gain"] if current["dcc_gain"] is not None else -9999.0
                new_gain = res["dcc_gain"] if res["dcc_gain"] is not None else -9999.0
                if new_gain > curr_gain:
                    better = True
                elif new_gain == curr_gain:
                    if res["pass_rate"] > current["pass_rate"]:
                        better = True
                    elif res["pass_rate"] == current["pass_rate"] and res["avg_cluster_size"] > current["avg_cluster_size"]:
                        better = True
            if better:
                per_target_best[tname] = res

        valid = [r for r in per_target if "error" not in r]
        if not valid:
            continue

        dcc_gain_vals = [r["dcc_gain"] for r in valid if r["dcc_gain"] is not None]
        mean_dcc_gain = float(np.mean(dcc_gain_vals)) if dcc_gain_vals else None
        median_dcc_gain = float(np.median(dcc_gain_vals)) if dcc_gain_vals else None
        positive_gain_frac = float(np.mean([1.0 if (r["dcc_gain"] is not None and r["dcc_gain"] > 0) else 0.0 for r in valid]))
        mean_pass_rate = float(np.mean([r["pass_rate"] for r in valid]))
        mean_cluster_size = float(np.mean([r["avg_cluster_size"] for r in valid]))
        total_passing_clusters = int(sum(r["n_passing_clusters"] for r in valid))

        all_config_results.append({
            "config": asdict(cfg),
            "n_targets": len(valid),
            "mean_dcc_gain": mean_dcc_gain,
            "median_dcc_gain": median_dcc_gain,
            "positive_gain_fraction": positive_gain_frac,
            "mean_pass_rate": mean_pass_rate,
            "mean_cluster_size": mean_cluster_size,
            "total_passing_clusters": total_passing_clusters,
            "per_target": valid,
        })

    def config_rank(x: dict):
        return (
            x["mean_dcc_gain"] if x["mean_dcc_gain"] is not None else -9999.0,
            x["positive_gain_fraction"],
            x["mean_pass_rate"],
            x["mean_cluster_size"],
            x["total_passing_clusters"],
        )

    all_config_results.sort(key=config_rank, reverse=True)

    per_target_best_list = sorted(
        per_target_best.values(),
        key=lambda r: (
            r["dcc_gain"] if r["dcc_gain"] is not None else -9999.0,
            r["pass_rate"],
            r["avg_cluster_size"],
        ),
        reverse=True,
    )

    summary = {
        "root": str(args.root),
        "mode": args.mode,
        "n_targets": len(target_dirs),
        "n_configs": len(configs),
        "top_global_configs": all_config_results[:20],
        "best_config_per_target": per_target_best_list,
        "all_config_results": all_config_results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Top global configs ===")
    for i, r in enumerate(all_config_results[:10], start=1):
        cfg = r["config"]
        print(
            f"{i:>2}. mean_gain={r['mean_dcc_gain']}  "
            f"median_gain={r['median_dcc_gain']}  "
            f"positive_frac={r['positive_gain_fraction']:.3f}  "
            f"pass_rate={r['mean_pass_rate']:.3f}  "
            f"avg_cluster={r['mean_cluster_size']:.2f}  "
            f"passing={r['total_passing_clusters']:>5}  "
            f"pair_dist={cfg['max_pair_dist']}  "
            f"min_cos={cfg['min_cosine']}  "
            f"lag_diff={cfg['max_lag_diff']}  "
            f"edge={cfg['min_edge_score']}"
        )

    print("\n=== Best config per target (top 20 by gain) ===")
    for i, r in enumerate(per_target_best_list[:20], start=1):
        cfg = r["config"]
        print(
            f"{i:>2}. {r['target']:<20} gain={r['dcc_gain']}  "
            f"orig={r['original_best_dcc']}  new={r['best_new_dcc']}  "
            f"pair_dist={cfg['max_pair_dist']} min_cos={cfg['min_cosine']} "
            f"lag_diff={cfg['max_lag_diff']} edge={cfg['min_edge_score']}"
        )

    print(f"\nWrote panel sweep results to: {args.out}")


if __name__ == "__main__":
    main()
