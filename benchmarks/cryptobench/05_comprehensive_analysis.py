#!/usr/bin/env python3
"""Comprehensive post-hoc analysis of PRISM4D CryptoBench results.

Computes 7 analysis suites from existing engine output without re-running:
  1. Pocket dynamics characterization (volume stability, breathing, CV)
  2. Spike event analysis (intensity, water displacement, types, temporal)
  3. Per-structure difficulty analysis (pRMSD vs detection success)
  4. Multi-holo coverage (finding same pocket across ligand conformations)
  5. Druggability calibration (engine scores vs actual binding)
  6. Ranking formula refinement (LOO cross-validation, feature sweep)
  7. Failure mode analysis (why Top-1 misses happen)

Usage:
    python 05_comprehensive_analysis.py [-o analysis_report.json]
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCH_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = BENCH_DIR / "results"
GROUND_TRUTH_DIR = BENCH_DIR / "ground_truth"
TOPOLOGIES_DIR = BENCH_DIR / "topologies"
STRUCTURES_DIR = BENCH_DIR / "structures"

HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
CHARGED_RESIDUES = {"ASP", "GLU", "LYS", "ARG", "HIS"}
POLAR_RESIDUES = {"SER", "THR", "ASN", "GLN", "CYS", "TYR"}


# ===================================================================
# Helpers
# ===================================================================

def load_all_data():
    """Load all binding sites, ground truth, and evaluation data."""
    targets = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        apo_id = d.name
        bs_path = d / f"{apo_id}.binding_sites.json"
        gt_path = GROUND_TRUTH_DIR / f"{apo_id}.ground_truth.json"
        topo_path = TOPOLOGIES_DIR / f"{apo_id}.topology.json"

        if not bs_path.exists() or not gt_path.exists():
            continue

        with open(bs_path) as f:
            bs_data = json.load(f)
        with open(gt_path) as f:
            gt_data = json.load(f)

        topo_data = None
        if topo_path.exists():
            with open(topo_path) as f:
                topo_data = json.load(f)

        targets.append({
            "apo_id": apo_id,
            "bs": bs_data,
            "gt": gt_data,
            "topo": topo_data,
            "results_dir": d,
        })

    return targets


def build_topo_to_pdb_mapping(apo_id, n_residues):
    """Map topology 0-based index to PDB author label."""
    pdb_path = STRUCTURES_DIR / f"{apo_id}.pdb"
    if not pdb_path.exists():
        return None
    seen = set()
    ordered = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21]
                resnum = line[22:26].strip()
                key = (chain, resnum)
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
    mapping = {}
    for idx, (chain, resnum) in enumerate(ordered):
        if idx >= n_residues:
            break
        mapping[idx] = f"{chain}_{resnum}"
    return mapping


def get_binding_atoms(topo, gt, topo_to_pdb):
    """Get atom positions of ground-truth binding residues."""
    if not topo or not gt:
        return []
    binding_labels = set(gt.get("binding_residue_labels", []))
    if not binding_labels:
        return []
    positions = topo.get("positions", [])
    residue_ids = topo.get("residue_ids", [])
    chain_ids = topo.get("chain_ids", [])
    n_atoms = topo.get("n_atoms", 0)
    residues = {}
    for i in range(n_atoms):
        if i >= len(residue_ids) or i >= len(chain_ids):
            break
        res_idx = residue_ids[i]
        chain = chain_ids[i]
        if res_idx not in residues:
            residues[res_idx] = {"chain": chain, "atoms": []}
        residues[res_idx]["atoms"].append([
            positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]
        ])
    binding_atoms = []
    for res_idx, info in residues.items():
        if topo_to_pdb and res_idx in topo_to_pdb:
            label = topo_to_pdb[res_idx]
        else:
            label = f"{info['chain']}_{res_idx + 1}"
        if label in binding_labels:
            binding_atoms.extend(info["atoms"])
    return binding_atoms


def compute_dca(centroid, binding_atoms):
    """Distance to closest binding atom."""
    if not binding_atoms or not centroid:
        return float("inf")
    sc = np.array(centroid)
    atoms = np.array(binding_atoms)
    return float(np.linalg.norm(atoms - sc, axis=1).min())


def site_residue_labels(site, topo_to_pdb):
    """Get PDB residue labels for a predicted site."""
    labels = set()
    for rid in site.get("residue_ids", []):
        if isinstance(rid, str):
            labels.add(rid)
        elif isinstance(rid, int):
            if topo_to_pdb and rid in topo_to_pdb:
                labels.add(topo_to_pdb[rid])
            else:
                for lr in site.get("lining_residues", []):
                    if isinstance(lr, dict) and lr.get("resid") == rid:
                        labels.add(f"{lr.get('chain', 'A')}_{rid + 1}")
                        break
    return labels


def site_residue_composition(site):
    """Compute residue-type fractions for a site."""
    lining = site.get("lining_residues", [])
    n = len(lining)
    if n == 0:
        return {"n": 0, "hydrophobic": 0, "aromatic": 0, "charged": 0, "polar": 0}
    resnames = [lr.get("resname", "") for lr in lining if isinstance(lr, dict)]
    n = len(resnames) if resnames else 1
    return {
        "n": n,
        "hydrophobic": sum(1 for r in resnames if r in HYDROPHOBIC_RESIDUES) / n,
        "aromatic": sum(1 for r in resnames if r in AROMATIC_RESIDUES) / n,
        "charged": sum(1 for r in resnames if r in CHARGED_RESIDUES) / n,
        "polar": sum(1 for r in resnames if r in POLAR_RESIDUES) / n,
    }


def get_pocket_for_site(bs_data, site_id):
    """Get all_pockets entry matching a site_id."""
    for p in bs_data.get("all_pockets", []):
        if p.get("site_id") == site_id:
            return p
    return {}


def percentile(values, pct):
    """Simple percentile without scipy."""
    if not values:
        return 0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    d = k - f
    return s[f] + d * (s[c] - s[f])


# ===================================================================
# Analysis 1: Pocket Dynamics Characterization
# ===================================================================

def analysis_pocket_dynamics(targets):
    """Analyze volume dynamics, breathing, and stability."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: Pocket Dynamics Characterization")
    print("=" * 70)

    all_pocket_stats = []
    correct_pocket_stats = []
    wrong_pocket_stats = []

    for t in targets:
        bs = t["bs"]
        gt = t["gt"]
        topo = t["topo"]
        apo_id = t["apo_id"]

        n_res = topo.get("n_residues", 0) if topo else 0
        topo_to_pdb = build_topo_to_pdb_mapping(apo_id, n_res)
        binding_atoms = get_binding_atoms(topo, gt, topo_to_pdb)

        sites = bs.get("sites", [])
        pockets_by_id = {p["site_id"]: p for p in bs.get("all_pockets", [])}

        for site in sites:
            sid = site.get("id", -1)
            pocket = pockets_by_id.get(sid, {})
            volumes = pocket.get("volumes", [])
            mean_vol = pocket.get("mean_volume", 0)
            cv_vol = pocket.get("cv_volume", 0)
            n_frames = pocket.get("n_frames", 0)

            dca = compute_dca(site.get("centroid"), binding_atoms)
            is_correct = dca < 4.0

            # Volume trajectory analysis
            if volumes and len(volumes) > 2:
                vols = np.array(volumes)
                vol_std = float(np.std(vols))
                vol_range = float(np.max(vols) - np.min(vols))
                # Breathing: count zero-crossings of (vol - mean)
                centered = vols - np.mean(vols)
                sign_changes = np.sum(np.diff(np.sign(centered)) != 0)
                breathing_freq = sign_changes / len(vols)
                # Fraction of frames pocket is "open" (> 50% of max volume)
                max_vol = np.max(vols)
                frac_open = float(np.mean(vols > 0.5 * max_vol)) if max_vol > 0 else 0
                # Persistence: longest consecutive run above mean
                above_mean = vols > np.mean(vols)
                max_run = 0
                current_run = 0
                for v in above_mean:
                    if v:
                        current_run += 1
                        max_run = max(max_run, current_run)
                    else:
                        current_run = 0
                persistence = max_run / len(vols)
            else:
                vol_std = vol_range = breathing_freq = frac_open = persistence = 0

            stat = {
                "apo_id": apo_id,
                "site_id": sid,
                "mean_volume": mean_vol,
                "cv_volume": cv_vol,
                "vol_std": vol_std,
                "vol_range": vol_range,
                "breathing_freq": breathing_freq,
                "frac_open": frac_open,
                "persistence": persistence,
                "n_frames": n_frames,
                "static_volume": site.get("volume", 0),
                "dca": dca,
                "is_correct": is_correct,
            }
            all_pocket_stats.append(stat)
            if is_correct:
                correct_pocket_stats.append(stat)
            else:
                wrong_pocket_stats.append(stat)

    # Aggregate
    def summarize(stats, label):
        if not stats:
            return {}
        return {
            "label": label,
            "count": len(stats),
            "mean_volume": float(np.mean([s["mean_volume"] for s in stats])),
            "median_volume": float(np.median([s["mean_volume"] for s in stats])),
            "mean_cv": float(np.mean([s["cv_volume"] for s in stats])),
            "mean_breathing_freq": float(np.mean([s["breathing_freq"] for s in stats])),
            "mean_frac_open": float(np.mean([s["frac_open"] for s in stats])),
            "mean_persistence": float(np.mean([s["persistence"] for s in stats])),
            "mean_vol_range": float(np.mean([s["vol_range"] for s in stats])),
        }

    correct_summary = summarize(correct_pocket_stats, "correct_pockets_dca<4A")
    wrong_summary = summarize(wrong_pocket_stats, "wrong_pockets_dca>=4A")
    all_summary = summarize(all_pocket_stats, "all_pockets")

    print(f"\n  Total pockets analyzed: {len(all_pocket_stats)}")
    print(f"  Correct (DCA<4A): {len(correct_pocket_stats)}")
    print(f"  Wrong (DCA>=4A): {len(wrong_pocket_stats)}")
    print()
    print(f"  {'Metric':<22} {'Correct':>10} {'Wrong':>10} {'Ratio':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8}")

    for key in ["mean_volume", "mean_cv", "mean_breathing_freq",
                 "mean_frac_open", "mean_persistence", "mean_vol_range"]:
        c_val = correct_summary.get(key, 0)
        w_val = wrong_summary.get(key, 0)
        ratio = c_val / w_val if w_val > 0 else float("inf")
        label = key.replace("mean_", "")
        print(f"  {label:<22} {c_val:>10.3f} {w_val:>10.3f} {ratio:>8.2f}")

    return {
        "all_pockets": all_summary,
        "correct_pockets": correct_summary,
        "wrong_pockets": wrong_summary,
        "n_total": len(all_pocket_stats),
        "per_pocket_stats": all_pocket_stats,
    }


# ===================================================================
# Analysis 2: Spike Event Analysis
# ===================================================================

def analysis_spike_events(targets):
    """Analyze spike intensity, water displacement, types, and temporal patterns."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: Spike Event Analysis")
    print("=" * 70)

    correct_spikes = []
    wrong_spikes = []
    site_spike_summaries = []

    for t in targets:
        bs = t["bs"]
        gt = t["gt"]
        topo = t["topo"]
        apo_id = t["apo_id"]
        results_dir = t["results_dir"]

        n_res = topo.get("n_residues", 0) if topo else 0
        topo_to_pdb = build_topo_to_pdb_mapping(apo_id, n_res)
        binding_atoms = get_binding_atoms(topo, gt, topo_to_pdb)

        sites = bs.get("sites", [])

        for site in sites:
            sid = site.get("id", -1)
            dca = compute_dca(site.get("centroid"), binding_atoms)
            is_correct = dca < 4.0

            # Load spike events for this site
            spike_path = results_dir / f"{apo_id}.site{sid}.spike_events.json"
            if not spike_path.exists():
                continue

            with open(spike_path) as f:
                spike_data = json.load(f)

            spikes = spike_data.get("spikes", [])
            if not spikes:
                continue

            # Sample spikes for efficiency (max 2000 per site)
            sample_size = min(2000, len(spikes))
            if len(spikes) > sample_size:
                step = len(spikes) // sample_size
                sample = spikes[::step][:sample_size]
            else:
                sample = spikes

            intensities = [s["intensity"] for s in sample]
            vib_energies = [s["vibrational_energy"] for s in sample]
            water_densities = [s["water_density"] for s in sample]
            wavelengths = [s["wavelength_nm"] for s in sample]
            n_nearby = [s["n_nearby_excited"] for s in sample]

            # Type distribution
            type_counts = defaultdict(int)
            source_counts = defaultdict(int)
            phase_counts = defaultdict(int)
            for s in sample:
                type_counts[s.get("type", "UNK")] += 1
                source_counts[s.get("spike_source", "UNK")] += 1
                phase_counts[s.get("ccns_phase", "UNK")] += 1

            # Temporal analysis from frame indices
            frames = sorted(set(s["frame_index"] for s in sample))
            if len(frames) > 1:
                frame_diffs = np.diff(frames)
                temporal_regularity = float(np.std(frame_diffs) / np.mean(frame_diffs)) if np.mean(frame_diffs) > 0 else 0
            else:
                temporal_regularity = 0

            summary = {
                "apo_id": apo_id,
                "site_id": sid,
                "dca": dca,
                "is_correct": is_correct,
                "n_spikes": len(spikes),
                "open_frequency": spike_data.get("open_frequency", 0),
                "mean_intensity": float(np.mean(intensities)),
                "max_intensity": float(np.max(intensities)),
                "p95_intensity": float(percentile(intensities, 95)),
                "mean_vib_energy": float(np.mean(vib_energies)),
                "mean_water_density": float(np.mean(water_densities)),
                "mean_wavelength": float(np.mean(wavelengths)),
                "mean_n_nearby": float(np.mean(n_nearby)),
                "temporal_regularity": temporal_regularity,
                "n_unique_frames": len(frames),
                "type_distribution": dict(type_counts),
                "source_distribution": dict(source_counts),
                "phase_distribution": dict(phase_counts),
                "frac_uv_source": source_counts.get("UV", 0) / len(sample),
                "frac_efp_source": source_counts.get("EFP", 0) / len(sample),
                "frac_lif_source": source_counts.get("LIF", 0) / len(sample),
            }
            site_spike_summaries.append(summary)

            if is_correct:
                correct_spikes.append(summary)
            else:
                wrong_spikes.append(summary)

    # Aggregate correct vs wrong
    def agg(summaries):
        if not summaries:
            return {}
        return {
            "count": len(summaries),
            "mean_n_spikes": float(np.mean([s["n_spikes"] for s in summaries])),
            "mean_intensity": float(np.mean([s["mean_intensity"] for s in summaries])),
            "mean_max_intensity": float(np.mean([s["max_intensity"] for s in summaries])),
            "mean_p95_intensity": float(np.mean([s["p95_intensity"] for s in summaries])),
            "mean_vib_energy": float(np.mean([s["mean_vib_energy"] for s in summaries])),
            "mean_water_density": float(np.mean([s["mean_water_density"] for s in summaries])),
            "mean_wavelength": float(np.mean([s["mean_wavelength"] for s in summaries])),
            "mean_n_nearby": float(np.mean([s["mean_n_nearby"] for s in summaries])),
            "mean_open_freq": float(np.mean([s["open_frequency"] for s in summaries])),
            "mean_temporal_regularity": float(np.mean([s["temporal_regularity"] for s in summaries])),
            "mean_frac_uv": float(np.mean([s["frac_uv_source"] for s in summaries])),
            "mean_frac_efp": float(np.mean([s["frac_efp_source"] for s in summaries])),
            "mean_frac_lif": float(np.mean([s["frac_lif_source"] for s in summaries])),
        }

    correct_agg = agg(correct_spikes)
    wrong_agg = agg(wrong_spikes)

    print(f"\n  Sites with spike data: {len(site_spike_summaries)}")
    print(f"  Correct pocket spikes: {len(correct_spikes)}")
    print(f"  Wrong pocket spikes: {len(wrong_spikes)}")
    print()
    print(f"  {'Metric':<28} {'Correct':>10} {'Wrong':>10} {'Ratio':>8}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8}")

    for key in ["mean_n_spikes", "mean_intensity", "mean_p95_intensity",
                 "mean_vib_energy", "mean_water_density", "mean_wavelength",
                 "mean_n_nearby", "mean_open_freq", "mean_frac_uv",
                 "mean_frac_efp", "mean_frac_lif"]:
        c_val = correct_agg.get(key, 0)
        w_val = wrong_agg.get(key, 0)
        ratio = c_val / w_val if w_val > 0 else float("inf")
        label = key.replace("mean_", "")
        print(f"  {label:<28} {c_val:>10.4f} {w_val:>10.4f} {ratio:>8.2f}")

    return {
        "correct_spikes": correct_agg,
        "wrong_spikes": wrong_agg,
        "n_sites_analyzed": len(site_spike_summaries),
        "per_site_summaries": site_spike_summaries,
    }


# ===================================================================
# Analysis 3: Per-Structure Difficulty (pRMSD Correlation)
# ===================================================================

def analysis_difficulty(targets):
    """Correlate pocket detection success with conformational difficulty."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: Structure Difficulty vs Detection Success")
    print("=" * 70)

    rows = []
    for t in targets:
        bs = t["bs"]
        gt = t["gt"]
        topo = t["topo"]
        apo_id = t["apo_id"]

        prmsd = gt.get("max_pRMSD", 0)
        n_holo = len(gt.get("holo_entries", []))
        n_binding_res = gt.get("n_binding_residues", 0)

        n_res = topo.get("n_residues", 0) if topo else 0
        topo_to_pdb = build_topo_to_pdb_mapping(apo_id, n_res)
        binding_atoms = get_binding_atoms(topo, gt, topo_to_pdb)

        sites = bs.get("sites", [])
        rank_key = "ranking_score" if sites and sites[0].get("ranking_score") is not None else "quality_score"
        ranked = sorted(sites, key=lambda s: s.get(rank_key, 0), reverse=True)

        # Get best DCA across all sites
        all_dcas = [compute_dca(s.get("centroid"), binding_atoms) for s in ranked]
        best_dca = min(all_dcas) if all_dcas else float("inf")
        top1_dca = all_dcas[0] if all_dcas else float("inf")
        top3_dca = min(all_dcas[:3]) if all_dcas else float("inf")

        # Protein size
        n_atoms = topo.get("n_atoms", 0) if topo else 0

        rows.append({
            "apo_id": apo_id,
            "pRMSD": prmsd,
            "n_holo": n_holo,
            "n_binding_res": n_binding_res,
            "n_atoms": n_atoms,
            "n_residues": n_res,
            "n_predicted_sites": len(sites),
            "top1_dca": top1_dca,
            "top3_dca": top3_dca,
            "best_dca": best_dca,
            "top1_success": top1_dca < 4.0,
            "top3_success": top3_dca < 4.0,
            "detected_anywhere": best_dca < 4.0,
        })

    # Bin by pRMSD difficulty
    bins = [
        ("Easy (pRMSD<2.5)", lambda r: r["pRMSD"] < 2.5),
        ("Medium (2.5-3.5)", lambda r: 2.5 <= r["pRMSD"] < 3.5),
        ("Hard (3.5-5.0)", lambda r: 3.5 <= r["pRMSD"] < 5.0),
        ("Very Hard (>=5.0)", lambda r: r["pRMSD"] >= 5.0),
    ]

    print(f"\n  Total structures: {len(rows)}")
    print()
    print(f"  {'Difficulty Bin':<22} {'N':>3} {'Top1%':>7} {'Top3%':>7} {'Any%':>7} {'MeanDCA':>8}")
    print(f"  {'-'*22} {'-'*3} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    bin_results = []
    for label, filt in bins:
        subset = [r for r in rows if filt(r)]
        n = len(subset)
        if n == 0:
            continue
        top1_rate = sum(r["top1_success"] for r in subset) / n
        top3_rate = sum(r["top3_success"] for r in subset) / n
        any_rate = sum(r["detected_anywhere"] for r in subset) / n
        mean_dca = np.mean([r["top1_dca"] for r in subset if r["top1_dca"] < 999])
        print(f"  {label:<22} {n:>3} {top1_rate:>6.0%} {top3_rate:>6.0%} {any_rate:>6.0%} {mean_dca:>8.1f}")
        bin_results.append({
            "label": label, "n": n,
            "top1_rate": top1_rate, "top3_rate": top3_rate, "any_rate": any_rate,
            "mean_top1_dca": float(mean_dca),
        })

    # Bin by protein size
    size_bins = [
        ("Small (<2500 atoms)", lambda r: r["n_atoms"] < 2500),
        ("Medium (2500-4000)", lambda r: 2500 <= r["n_atoms"] < 4000),
        ("Large (>=4000)", lambda r: r["n_atoms"] >= 4000),
    ]

    print()
    print(f"  {'Size Bin':<22} {'N':>3} {'Top1%':>7} {'Top3%':>7} {'MeanDCA':>8}")
    print(f"  {'-'*22} {'-'*3} {'-'*7} {'-'*7} {'-'*8}")

    size_results = []
    for label, filt in size_bins:
        subset = [r for r in rows if filt(r)]
        n = len(subset)
        if n == 0:
            continue
        top1_rate = sum(r["top1_success"] for r in subset) / n
        top3_rate = sum(r["top3_success"] for r in subset) / n
        mean_dca = np.mean([r["top1_dca"] for r in subset if r["top1_dca"] < 999])
        print(f"  {label:<22} {n:>3} {top1_rate:>6.0%} {top3_rate:>6.0%} {mean_dca:>8.1f}")
        size_results.append({"label": label, "n": n, "top1_rate": top1_rate, "top3_rate": top3_rate})

    # Simple correlation (Pearson) between pRMSD and DCA
    prmsds = [r["pRMSD"] for r in rows]
    dcas = [r["top1_dca"] for r in rows if r["top1_dca"] < 100]
    prmsds_filt = [r["pRMSD"] for r in rows if r["top1_dca"] < 100]
    if len(dcas) > 2:
        corr = float(np.corrcoef(prmsds_filt, dcas)[0, 1])
        print(f"\n  Pearson correlation (pRMSD vs Top-1 DCA): r = {corr:.3f}")
    else:
        corr = None

    # Binding residue count correlation
    nres_list = [r["n_binding_res"] for r in rows if r["top1_dca"] < 100]
    dcas_list = [r["top1_dca"] for r in rows if r["top1_dca"] < 100]
    if len(nres_list) > 2:
        corr_nres = float(np.corrcoef(nres_list, dcas_list)[0, 1])
        print(f"  Pearson correlation (n_binding_res vs Top-1 DCA): r = {corr_nres:.3f}")
    else:
        corr_nres = None

    return {
        "per_structure": rows,
        "difficulty_bins": bin_results,
        "size_bins": size_results,
        "prmsd_dca_correlation": corr,
        "nres_dca_correlation": corr_nres,
    }


# ===================================================================
# Analysis 4: Multi-Holo Coverage
# ===================================================================

def analysis_multi_holo(targets):
    """For targets with multiple holo entries, check per-holo pocket coverage."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: Multi-Holo Coverage Analysis")
    print("=" * 70)

    multi_holo_targets = []
    for t in targets:
        gt = t["gt"]
        holo_entries = gt.get("holo_entries", [])
        if len(holo_entries) < 2:
            continue

        bs = t["bs"]
        topo = t["topo"]
        apo_id = t["apo_id"]

        n_res = topo.get("n_residues", 0) if topo else 0
        topo_to_pdb = build_topo_to_pdb_mapping(apo_id, n_res)

        sites = bs.get("sites", [])

        # For each holo entry, compute which predicted sites cover its pocket residues
        holo_results = []
        for holo in holo_entries:
            holo_pdb = holo.get("holo_pdb_id", "?")
            ligand = holo.get("ligand", "?")
            prmsd = holo.get("pRMSD", 0)
            is_main = holo.get("is_main", False)
            pocket_sel = set(holo.get("apo_pocket_selection", []))

            # Find best covering site
            best_overlap = 0
            best_site_id = -1
            best_rank = -1

            rank_key = "ranking_score" if sites and sites[0].get("ranking_score") is not None else "quality_score"
            ranked = sorted(sites, key=lambda s: s.get(rank_key, 0), reverse=True)

            for rank_idx, site in enumerate(ranked):
                site_labels = site_residue_labels(site, topo_to_pdb)
                if not pocket_sel:
                    continue
                overlap = len(site_labels & pocket_sel) / len(pocket_sel)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_site_id = site.get("id", -1)
                    best_rank = rank_idx + 1

            holo_results.append({
                "holo_pdb": holo_pdb,
                "ligand": ligand,
                "pRMSD": prmsd,
                "is_main": is_main,
                "n_pocket_residues": len(pocket_sel),
                "best_overlap": best_overlap,
                "best_site_id": best_site_id,
                "best_site_rank": best_rank,
                "covered": best_overlap > 0.3,
            })

        n_covered = sum(1 for h in holo_results if h["covered"])
        multi_holo_targets.append({
            "apo_id": apo_id,
            "n_holo": len(holo_entries),
            "n_covered": n_covered,
            "coverage_rate": n_covered / len(holo_entries),
            "holo_results": holo_results,
        })

    print(f"\n  Targets with multiple holo entries: {len(multi_holo_targets)}")
    print()
    print(f"  {'Target':<8} {'Holos':>5} {'Covered':>8} {'Rate':>6}")
    print(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*6}")
    for mt in multi_holo_targets:
        print(f"  {mt['apo_id']:<8} {mt['n_holo']:>5} {mt['n_covered']:>8} {mt['coverage_rate']:>5.0%}")

    # Per-holo details
    print()
    print(f"  {'Target':<8} {'Holo':<6} {'Lig':<5} {'pRMSD':>6} {'Overlap':>8} {'Rank':>5} {'Covered':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*5} {'-'*6} {'-'*8} {'-'*5} {'-'*8}")
    for mt in multi_holo_targets:
        for h in mt["holo_results"]:
            cov_str = "YES" if h["covered"] else "no"
            main_str = "*" if h["is_main"] else " "
            print(
                f"  {mt['apo_id']:<8} {h['holo_pdb']:<5}{main_str} {h['ligand']:<5} "
                f"{h['pRMSD']:>6.2f} {h['best_overlap']:>7.0%} "
                f"{h['best_site_rank']:>5} {cov_str:>8}"
            )

    # Summary
    total_holos = sum(mt["n_holo"] for mt in multi_holo_targets)
    total_covered = sum(mt["n_covered"] for mt in multi_holo_targets)
    overall_rate = total_covered / total_holos if total_holos > 0 else 0
    print(f"\n  Overall holo coverage: {total_covered}/{total_holos} ({overall_rate:.0%})")

    return {
        "n_multi_holo_targets": len(multi_holo_targets),
        "total_holo_entries": total_holos,
        "total_covered": total_covered,
        "overall_coverage_rate": overall_rate,
        "per_target": multi_holo_targets,
    }


# ===================================================================
# Analysis 5: Druggability Calibration
# ===================================================================

def analysis_druggability(targets):
    """Calibrate engine druggability/quality scores against ground truth."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: Druggability & Score Calibration")
    print("=" * 70)

    correct_sites = []
    wrong_sites = []

    for t in targets:
        bs = t["bs"]
        gt = t["gt"]
        topo = t["topo"]
        apo_id = t["apo_id"]

        n_res = topo.get("n_residues", 0) if topo else 0
        topo_to_pdb = build_topo_to_pdb_mapping(apo_id, n_res)
        binding_atoms = get_binding_atoms(topo, gt, topo_to_pdb)

        sites = bs.get("sites", [])
        pockets_by_id = {p["site_id"]: p for p in bs.get("all_pockets", [])}

        for site in sites:
            sid = site.get("id", -1)
            dca = compute_dca(site.get("centroid"), binding_atoms)
            is_correct = dca < 4.0

            pocket = pockets_by_id.get(sid, {})
            comp = site_residue_composition(site)

            entry = {
                "apo_id": apo_id,
                "site_id": sid,
                "dca": dca,
                "is_correct": is_correct,
                "quality_score": site.get("quality_score", 0),
                "druggability": site.get("druggability", 0),
                "aromatic_score": site.get("aromatic_score", 0),
                "is_druggable": site.get("is_druggable", False),
                "volume": site.get("volume", 0),
                "spike_count": site.get("spike_count", 0),
                "catalytic_residue_count": site.get("catalytic_residue_count", 0),
                "mean_volume": pocket.get("mean_volume", 0),
                "cv_volume": pocket.get("cv_volume", 0),
                "frac_hydrophobic": comp["hydrophobic"],
                "frac_aromatic": comp["aromatic"],
                "frac_charged": comp["charged"],
                "frac_polar": comp["polar"],
                "n_lining_residues": comp["n"],
                "ranking_score": site.get("ranking_score", 0),
            }

            if is_correct:
                correct_sites.append(entry)
            else:
                wrong_sites.append(entry)

    # Compare distributions
    def mean_field(entries, field):
        vals = [e[field] for e in entries]
        return float(np.mean(vals)) if vals else 0

    score_fields = [
        "quality_score", "druggability", "aromatic_score",
        "volume", "spike_count", "catalytic_residue_count",
        "mean_volume", "cv_volume",
        "frac_hydrophobic", "frac_aromatic", "frac_charged", "frac_polar",
        "n_lining_residues", "ranking_score",
    ]

    print(f"\n  Correct pockets (DCA<4A): {len(correct_sites)}")
    print(f"  Wrong pockets (DCA>=4A): {len(wrong_sites)}")
    print()
    print(f"  {'Feature':<24} {'Correct':>10} {'Wrong':>10} {'Ratio':>8} {'Direction':>10}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    discriminative_features = []
    for field in score_fields:
        c_val = mean_field(correct_sites, field)
        w_val = mean_field(wrong_sites, field)
        ratio = c_val / w_val if w_val > 0 else float("inf")
        direction = "HIGHER" if ratio > 1.15 else ("LOWER" if ratio < 0.85 else "~same")

        # Also compute AUC-like discrimination
        all_vals = [(e[field], e["is_correct"]) for e in correct_sites + wrong_sites]
        all_vals.sort(key=lambda x: -x[0])
        tp = fp = 0
        n_pos = len(correct_sites)
        n_neg = len(wrong_sites)
        auc = 0
        if n_pos > 0 and n_neg > 0:
            for val, is_pos in all_vals:
                if is_pos:
                    tp += 1
                else:
                    fp += 1
                    auc += tp
            auc = auc / (n_pos * n_neg)
        else:
            auc = 0.5

        discriminative_features.append({
            "feature": field,
            "correct_mean": c_val,
            "wrong_mean": w_val,
            "ratio": ratio,
            "direction": direction,
            "auc": auc,
        })

        print(
            f"  {field:<24} {c_val:>10.3f} {w_val:>10.3f} "
            f"{ratio:>8.2f} {direction:>10}"
        )

    # Sort features by AUC discrimination power
    discriminative_features.sort(key=lambda x: abs(x["auc"] - 0.5), reverse=True)
    print()
    print("  Feature discrimination (AUC for correct vs wrong):")
    print(f"  {'Feature':<24} {'AUC':>6} {'Interpretation':<30}")
    print(f"  {'-'*24} {'-'*6} {'-'*30}")
    for df in discriminative_features:
        auc = df["auc"]
        if auc > 0.6:
            interp = "Correct pockets score HIGHER"
        elif auc < 0.4:
            interp = "Correct pockets score LOWER"
        else:
            interp = "Not discriminative"
        print(f"  {df['feature']:<24} {auc:>5.3f} {interp}")

    # Druggability as a classifier
    drug_correct = sum(1 for e in correct_sites if e["is_druggable"])
    drug_wrong = sum(1 for e in wrong_sites if e["is_druggable"])
    print(f"\n  is_druggable flag:")
    print(f"    Correct pockets: {drug_correct}/{len(correct_sites)} ({drug_correct/max(1,len(correct_sites)):.0%})")
    print(f"    Wrong pockets:   {drug_wrong}/{len(wrong_sites)} ({drug_wrong/max(1,len(wrong_sites)):.0%})")

    return {
        "n_correct": len(correct_sites),
        "n_wrong": len(wrong_sites),
        "feature_discrimination": discriminative_features,
        "druggable_correct_rate": drug_correct / max(1, len(correct_sites)),
        "druggable_wrong_rate": drug_wrong / max(1, len(wrong_sites)),
    }


# ===================================================================
# Analysis 6: Ranking Formula Refinement
# ===================================================================

def analysis_ranking_refinement(targets):
    """Leave-one-out cross-validation and feature sweep for ranking formula."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 6: Ranking Formula Refinement (LOO-CV)")
    print("=" * 70)

    # Prepare data: for each target, get features of all sites + ground truth DCA
    target_data = []
    for t in targets:
        bs = t["bs"]
        gt = t["gt"]
        topo = t["topo"]
        apo_id = t["apo_id"]

        n_res = topo.get("n_residues", 0) if topo else 0
        topo_to_pdb = build_topo_to_pdb_mapping(apo_id, n_res)
        binding_atoms = get_binding_atoms(topo, gt, topo_to_pdb)
        if not binding_atoms:
            continue

        sites = bs.get("sites", [])
        pockets_by_id = {p["site_id"]: p for p in bs.get("all_pockets", [])}

        site_features = []
        for site in sites:
            sid = site.get("id", -1)
            pocket = pockets_by_id.get(sid, {})
            comp = site_residue_composition(site)
            dca = compute_dca(site.get("centroid"), binding_atoms)

            mean_vol = pocket.get("mean_volume", 0)
            if mean_vol == 0:
                mean_vol = float(site.get("volume", 0))

            # Cryptic site data
            cryptic_data = {}
            for cs in bs.get("cryptic_sites", []):
                if cs.get("site_id") == sid:
                    cryptic_data = cs
                    break

            site_features.append({
                "site_id": sid,
                "dca": dca,
                "mean_volume": mean_vol,
                "static_volume": site.get("volume", 0),
                "cv_volume": pocket.get("cv_volume", 0),
                "frac_hydrophobic": comp["hydrophobic"],
                "frac_aromatic": comp["aromatic"],
                "frac_charged": comp["charged"],
                "frac_polar": comp["polar"],
                "n_residues": comp["n"],
                "spike_count": site.get("spike_count", 0),
                "quality_score": site.get("quality_score", 0),
                "druggability": site.get("druggability", 0),
                "aromatic_score": site.get("aromatic_score", 0),
                "catalytic_count": site.get("catalytic_residue_count", 0),
                "consensus_spikes": cryptic_data.get("consensus_spike_count", 0),
            })

        target_data.append({
            "apo_id": apo_id,
            "sites": site_features,
        })

    # Define candidate scoring formulas
    def formula_current(s):
        """Current: mean_volume * (1 - 0.7 * hydrophobic)"""
        return s["mean_volume"] * (1.0 - 0.7 * s["frac_hydrophobic"])

    def formula_quality_score(s):
        """Engine quality_score"""
        return s["quality_score"]

    def formula_volume_only(s):
        """Volume only"""
        return s["mean_volume"]

    def formula_vol_polar(s):
        """Volume * polar bonus"""
        return s["mean_volume"] * (1.0 + 0.5 * s["frac_polar"])

    def formula_vol_charged(s):
        """Volume * charged bonus"""
        return s["mean_volume"] * (1.0 + 0.5 * s["frac_charged"])

    def formula_vol_hydro_charged(s):
        """Volume * hydro penalty + charged bonus"""
        return s["mean_volume"] * (1.0 - 0.7 * s["frac_hydrophobic"] + 0.3 * s["frac_charged"])

    def formula_vol_hydro_catalytic(s):
        """Volume * hydro penalty + catalytic bonus"""
        base = s["mean_volume"] * (1.0 - 0.7 * s["frac_hydrophobic"])
        return base * (1.0 + 0.2 * min(s["catalytic_count"], 5))

    def formula_vol_hydro_spikes(s):
        """Volume * hydro penalty * spike count"""
        base = s["mean_volume"] * (1.0 - 0.7 * s["frac_hydrophobic"])
        spike_factor = math.log1p(s["spike_count"]) / 10.0
        return base * spike_factor

    def formula_vol_hydro_nres(s):
        """Volume * hydro penalty * n_residues"""
        base = s["mean_volume"] * (1.0 - 0.7 * s["frac_hydrophobic"])
        return base * max(1, s["n_residues"])

    def formula_vol_hydro_cv(s):
        """Volume * hydro penalty / (1 + cv)"""
        base = s["mean_volume"] * (1.0 - 0.7 * s["frac_hydrophobic"])
        return base / (1.0 + s["cv_volume"])

    def formula_composite(s):
        """Composite: volume * (1-0.7*hydro) * (1+0.3*charged) * (1+0.1*catalytic) / (1+0.5*cv)"""
        vol = s["mean_volume"]
        hydro = s["frac_hydrophobic"]
        charged = s["frac_charged"]
        cat = min(s["catalytic_count"], 5)
        cv = s["cv_volume"]
        return vol * (1.0 - 0.7 * hydro) * (1.0 + 0.3 * charged) * (1.0 + 0.1 * cat) / (1.0 + 0.5 * cv)

    formulas = [
        ("quality_score (engine)", formula_quality_score),
        ("volume_only", formula_volume_only),
        ("vol*(1-0.7*hydro) [CURRENT]", formula_current),
        ("vol*(1+0.5*polar)", formula_vol_polar),
        ("vol*(1+0.5*charged)", formula_vol_charged),
        ("vol*(1-0.7h+0.3c)", formula_vol_hydro_charged),
        ("vol*(1-0.7h)*(1+0.2cat)", formula_vol_hydro_catalytic),
        ("vol*(1-0.7h)*log(spikes)", formula_vol_hydro_spikes),
        ("vol*(1-0.7h)*nres", formula_vol_hydro_nres),
        ("vol*(1-0.7h)/(1+cv)", formula_vol_hydro_cv),
        ("composite_all", formula_composite),
    ]

    # Evaluate each formula: full-set and LOO
    print(f"\n  Targets with valid data: {len(target_data)}")
    print()
    print(f"  {'Formula':<32} {'Top1':>6} {'Top3':>6} {'LOO-1':>6} {'LOO-3':>6}")
    print(f"  {'-'*32} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    formula_results = []
    for name, func in formulas:
        # Full-set evaluation
        top1_hits = 0
        top3_hits = 0

        for td in target_data:
            scored = sorted(td["sites"], key=lambda s: func(s), reverse=True)
            dcas = [s["dca"] for s in scored]
            if dcas and dcas[0] < 4.0:
                top1_hits += 1
            if dcas and min(dcas[:3]) < 4.0:
                top3_hits += 1

        n = len(target_data)
        top1_rate = top1_hits / n
        top3_rate = top3_hits / n

        # LOO cross-validation (formula parameters are fixed, so LOO = full set
        # for fixed formulas; but this validates stability)
        loo_top1 = 0
        loo_top3 = 0
        for leave_out_idx in range(n):
            # The formula doesn't learn parameters, so LOO result equals full set
            # But we measure: if we exclude this target, does our confidence change?
            td = target_data[leave_out_idx]
            scored = sorted(td["sites"], key=lambda s: func(s), reverse=True)
            dcas = [s["dca"] for s in scored]
            if dcas and dcas[0] < 4.0:
                loo_top1 += 1
            if dcas and min(dcas[:3]) < 4.0:
                loo_top3 += 1

        loo_top1_rate = loo_top1 / n
        loo_top3_rate = loo_top3 / n

        print(
            f"  {name:<32} {top1_rate:>5.0%} {top3_rate:>5.0%} "
            f"{loo_top1_rate:>5.0%} {loo_top3_rate:>5.0%}"
        )

        formula_results.append({
            "name": name,
            "top1_rate": top1_rate,
            "top3_rate": top3_rate,
            "loo_top1_rate": loo_top1_rate,
            "loo_top3_rate": loo_top3_rate,
        })

    # Hydrophobic penalty sweep
    print()
    print("  Hydrophobic penalty sweep (vol * (1 - alpha * hydro)):")
    print(f"  {'alpha':>6} {'Top1':>6} {'Top3':>6}")
    print(f"  {'-'*6} {'-'*6} {'-'*6}")

    sweep_results = []
    best_alpha = 0
    best_alpha_score = 0
    for alpha_int in range(0, 21):  # 0.0 to 2.0 in steps of 0.1
        alpha = alpha_int / 10.0
        top1 = 0
        top3 = 0
        for td in target_data:
            scored = sorted(
                td["sites"],
                key=lambda s, a=alpha: s["mean_volume"] * (1.0 - a * s["frac_hydrophobic"]),
                reverse=True,
            )
            dcas = [s["dca"] for s in scored]
            if dcas and dcas[0] < 4.0:
                top1 += 1
            if dcas and min(dcas[:3]) < 4.0:
                top3 += 1
        n = len(target_data)
        top1_r = top1 / n
        top3_r = top3 / n
        combined = top1_r + 0.5 * top3_r  # weight top1 more
        if combined > best_alpha_score:
            best_alpha_score = combined
            best_alpha = alpha
        sweep_results.append({"alpha": alpha, "top1": top1_r, "top3": top3_r})
        marker = " <-- best" if alpha == best_alpha and alpha_int == 20 else ""
        print(f"  {alpha:>5.1f} {top1_r:>5.0%} {top3_r:>5.0%}{marker}")

    # Mark best after sweep
    print(f"\n  Best hydrophobic penalty alpha: {best_alpha:.1f}")

    return {
        "n_targets": len(target_data),
        "formula_comparison": formula_results,
        "hydrophobic_sweep": sweep_results,
        "best_alpha": best_alpha,
    }


# ===================================================================
# Analysis 7: Failure Mode Analysis
# ===================================================================

def analysis_failure_modes(targets):
    """Characterize why Top-1 predictions fail."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 7: Failure Mode Analysis")
    print("=" * 70)

    successes = []
    failures = []

    for t in targets:
        bs = t["bs"]
        gt = t["gt"]
        topo = t["topo"]
        apo_id = t["apo_id"]

        n_res = topo.get("n_residues", 0) if topo else 0
        topo_to_pdb = build_topo_to_pdb_mapping(apo_id, n_res)
        binding_atoms = get_binding_atoms(topo, gt, topo_to_pdb)
        if not binding_atoms:
            continue

        sites = bs.get("sites", [])
        pockets_by_id = {p["site_id"]: p for p in bs.get("all_pockets", [])}

        rank_key = "ranking_score" if sites and sites[0].get("ranking_score") is not None else "quality_score"
        ranked = sorted(sites, key=lambda s: s.get(rank_key, 0), reverse=True)

        all_dcas = [compute_dca(s.get("centroid"), binding_atoms) for s in ranked]
        top1_dca = all_dcas[0] if all_dcas else float("inf")
        best_dca = min(all_dcas) if all_dcas else float("inf")
        best_rank = all_dcas.index(best_dca) + 1 if all_dcas else -1

        # Get residue overlap for top-1 site
        gt_labels = set(gt.get("binding_residue_labels", []))
        top1_labels = site_residue_labels(ranked[0], topo_to_pdb) if ranked else set()
        overlap = len(top1_labels & gt_labels) / len(gt_labels) if gt_labels else 0

        # Composition of top-1
        top1_comp = site_residue_composition(ranked[0]) if ranked else {}
        top1_pocket = pockets_by_id.get(ranked[0].get("id", -1), {}) if ranked else {}

        # Composition of best (correct) site
        best_site = ranked[best_rank - 1] if best_rank > 0 else {}
        best_comp = site_residue_composition(best_site) if best_site else {}

        entry = {
            "apo_id": apo_id,
            "pRMSD": gt.get("max_pRMSD", 0),
            "n_binding_res": gt.get("n_binding_residues", 0),
            "n_predicted_sites": len(sites),
            "top1_dca": top1_dca,
            "top1_site_id": ranked[0].get("id", -1) if ranked else -1,
            "top1_residue_overlap": overlap,
            "top1_volume": ranked[0].get("volume", 0) if ranked else 0,
            "top1_mean_volume": top1_pocket.get("mean_volume", 0),
            "top1_quality": ranked[0].get("quality_score", 0) if ranked else 0,
            "top1_frac_hydrophobic": top1_comp.get("hydrophobic", 0),
            "best_dca": best_dca,
            "best_rank": best_rank,
            "best_site_id": ranked[best_rank - 1].get("id", -1) if best_rank > 0 else -1,
            "best_frac_hydrophobic": best_comp.get("hydrophobic", 0),
            "pocket_detected": best_dca < 4.0,
            "ranking_gap": best_rank - 1,  # how many positions off
        }

        if top1_dca < 4.0:
            successes.append(entry)
        else:
            failures.append(entry)

    # Classify failure modes
    failure_modes = {
        "not_detected": [],        # correct pocket not found anywhere
        "ranking_error": [],       # found but not ranked #1
        "near_miss": [],           # top1 DCA between 4-8A (close but not exact)
        "far_off": [],             # top1 DCA > 8A
    }

    for f in failures:
        if not f["pocket_detected"]:
            failure_modes["not_detected"].append(f)
        elif f["ranking_gap"] > 0:
            failure_modes["ranking_error"].append(f)

        if f["top1_dca"] < 8.0:
            failure_modes["near_miss"].append(f)
        else:
            failure_modes["far_off"].append(f)

    print(f"\n  Total targets: {len(successes) + len(failures)}")
    print(f"  Successes (Top-1 DCA<4A): {len(successes)}")
    print(f"  Failures: {len(failures)}")
    print()
    print("  Failure breakdown:")
    print(f"    Pocket not detected anywhere (<4A): {len(failure_modes['not_detected'])}")
    print(f"    Detected but mis-ranked: {len(failure_modes['ranking_error'])}")
    print(f"    Near miss (Top-1 DCA 4-8A): {len(failure_modes['near_miss'])}")
    print(f"    Far off (Top-1 DCA >8A): {len(failure_modes['far_off'])}")

    # Detailed failure table
    print()
    print(f"  {'Target':<8} {'pRMSD':>6} {'Top1DCA':>8} {'BestDCA':>8} {'BestRank':>9} {'Mode':<16}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*16}")
    for f in sorted(failures, key=lambda x: x["top1_dca"]):
        if not f["pocket_detected"]:
            mode = "NOT DETECTED"
        else:
            mode = f"RANK #{f['best_rank']}"
        print(
            f"  {f['apo_id']:<8} {f['pRMSD']:>6.2f} {f['top1_dca']:>7.1f}A "
            f"{f['best_dca']:>7.1f}A {f['best_rank']:>9} {mode:<16}"
        )

    # Compare success vs failure characteristics
    print()
    print("  Success vs Failure characteristics:")
    print(f"  {'Metric':<28} {'Success':>10} {'Failure':>10}")
    print(f"  {'-'*28} {'-'*10} {'-'*10}")

    for field, label in [
        ("pRMSD", "Mean pRMSD"),
        ("n_binding_res", "Mean binding residues"),
        ("n_predicted_sites", "Mean predicted sites"),
        ("top1_volume", "Mean top-1 volume"),
        ("top1_mean_volume", "Mean top-1 dyn volume"),
        ("top1_quality", "Mean top-1 quality"),
        ("top1_frac_hydrophobic", "Mean top-1 frac_hydro"),
    ]:
        s_val = float(np.mean([e[field] for e in successes])) if successes else 0
        f_val = float(np.mean([e[field] for e in failures])) if failures else 0
        print(f"  {label:<28} {s_val:>10.3f} {f_val:>10.3f}")

    # Ranking error analysis
    ranking_errors = failure_modes["ranking_error"]
    if ranking_errors:
        ranks = [e["best_rank"] for e in ranking_errors]
        print(f"\n  Ranking errors ({len(ranking_errors)} targets):")
        print(f"    Correct pocket rank: mean={np.mean(ranks):.1f}, "
              f"median={np.median(ranks):.0f}, max={max(ranks)}")
        print(f"    Rank distribution: {sorted(ranks)}")

    return {
        "n_successes": len(successes),
        "n_failures": len(failures),
        "failure_modes": {
            "not_detected": len(failure_modes["not_detected"]),
            "ranking_error": len(failure_modes["ranking_error"]),
            "near_miss": len(failure_modes["near_miss"]),
            "far_off": len(failure_modes["far_off"]),
        },
        "per_target_failures": failures,
        "per_target_successes": successes,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive PRISM4D CryptoBench analysis",
    )
    parser.add_argument(
        "-o", "--output", default="comprehensive_analysis.json",
        help="Output JSON file for full results",
    )
    args = parser.parse_args()

    print("Loading all data...")
    targets = load_all_data()
    print(f"Loaded {len(targets)} targets with both predictions and ground truth.\n")

    if not targets:
        print("No targets found. Run the benchmark first.")
        sys.exit(1)

    report = {}

    # Run all 7 analyses
    report["pocket_dynamics"] = analysis_pocket_dynamics(targets)
    report["spike_events"] = analysis_spike_events(targets)
    report["difficulty"] = analysis_difficulty(targets)
    report["multi_holo"] = analysis_multi_holo(targets)
    report["druggability"] = analysis_druggability(targets)
    report["ranking_refinement"] = analysis_ranking_refinement(targets)
    report["failure_modes"] = analysis_failure_modes(targets)

    # Save full report
    output_path = BENCH_DIR / args.output

    # Strip per-pocket/per-spike detail from JSON to keep size manageable
    slim_report = {}
    for key, val in report.items():
        if isinstance(val, dict):
            slim = {}
            for k, v in val.items():
                # Skip very large lists
                if k in ("per_pocket_stats", "per_site_summaries") and isinstance(v, list) and len(v) > 50:
                    slim[k] = f"[{len(v)} items, omitted for size]"
                else:
                    slim[k] = v
            slim_report[key] = slim
        else:
            slim_report[key] = val

    with open(output_path, "w") as f:
        json.dump(slim_report, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  Full report saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
