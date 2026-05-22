#!/usr/bin/env python3
"""
PRISM-4D Benchmark v4 — Comprehensive Report Generator

Extracts all data from prism4d_v4_full.h5 and generates:
  1. prism4d_v4_report.json   — Machine-readable full export (for AI tools)
  2. prism4d_v4_report.md     — Human-readable markdown report

Usage:
    python3 scripts/generate_v4_report.py [--h5 <path>] [--output-dir <dir>]
"""
import json
import os
import sys
import argparse
from datetime import datetime

import h5py
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

TARGET_INFO = {
    "1bzj": {"name": "PTP1B", "type": "Phosphatase", "site_type": "Allosteric cryptic",
             "literature": "WPD loop closure exposes allosteric pocket ~20Å from active site"},
    "1r3m": {"name": "BS-RNase", "type": "Ribonuclease (dimer)", "site_type": "Interface",
             "literature": "Obligate dimer; cryptic site at A/B interface. Monomer run expected to miss it."},
    "2iyt": {"name": "MtSK (Shikimate Kinase)", "type": "Kinase", "site_type": "Cryptic lid",
             "literature": "Lid domain opens upon substrate binding, creating cryptic pocket"},
    "3uyi": {"name": "Perakine reductase", "type": "Reductase", "site_type": "Cryptic",
             "literature": "Substrate pocket partially occluded in apo; opens via loop displacement"},
    "4epr": {"name": "KRAS G12D", "type": "GTPase", "site_type": "Switch-II pocket",
             "literature": "Oncogenic mutant; SII-P allosteric pocket targetable by covalent inhibitors"},
    "2j1x": {"name": "TP53 Y220C", "type": "Tumor suppressor", "site_type": "Mutation-created cavity",
             "literature": "Y220C creates ~100Å³ cavity; small for current protocol resolution"},
    "1zg4": {"name": "TEM-1 M182T", "type": "β-lactamase", "site_type": "Omega loop cryptic",
             "literature": "Omega loop (163-179) opens transiently; M182T stabilizing mutant"},
}

TIERS = {"hard4": ["1bzj", "1r3m", "2iyt", "3uyi"], "tier2": ["4epr", "2j1x", "1zg4"]}


def decode_bytes(val):
    """Decode numpy bytes to str."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8").strip()
    return str(val)


def np_to_python(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: np_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [np_to_python(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (bytes, np.bytes_)):
        return obj.decode("utf-8").strip()
    elif isinstance(obj, np.ndarray):
        return np_to_python(obj.tolist())
    return obj


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_benchmark_summary(h5):
    """Extract top-level benchmark summary."""
    summary_ds = h5["benchmark/summary"]
    rows = []
    for row in summary_ds:
        rows.append({
            "pdb": decode_bytes(row["pdb"]),
            "tier": decode_bytes(row["tier"]),
            "n_sites": int(row["n_sites"]),
            "n_cryptic": int(row["n_cryptic"]),
            "best_overlap": int(row["best_overlap"]),
            "gt_residues": int(row["gt_size"]),
            "overlap_fraction": round(float(row["overlap_fraction"]), 4),
            "best_hysteresis": round(float(row["best_hysteresis"]), 4),
            "best_cryptic_rank": int(row["best_cryptic_rank"]),
            "best_gtck_rank": int(row["best_gtck_rank"]),
        })
    return rows


def extract_sites(h5, pdb):
    """Extract all binding sites for a target with full feature vectors."""
    grp = h5[f"targets/{pdb}/binding_sites"]
    sites_ds = grp["sites"]
    fields = sites_ds.dtype.names

    sites = []
    for row in sites_ds:
        site = {}
        for field in fields:
            val = row[field]
            if isinstance(val, (bytes, np.bytes_)):
                site[field] = decode_bytes(val)
            elif isinstance(val, (np.floating,)):
                v = float(val)
                site[field] = round(v, 6) if np.isfinite(v) else None
            else:
                site[field] = int(val)
        sites.append(site)
    return sites


def extract_lining_residues(h5, pdb):
    """Extract lining residues per site."""
    base = f"targets/{pdb}/binding_sites/lining_residues"
    if base not in h5:
        return {}
    result = {}
    for site_key in h5[base]:
        ds = h5[f"{base}/{site_key}"]
        residues = []
        for row in ds:
            residues.append({
                "resid": int(row["resid"]),
                "resname": decode_bytes(row["resname"]),
                "min_dist": round(float(row["min_dist"]), 3),
                "is_catalytic": bool(row["is_catalytic"]),
            })
        result[site_key] = residues
    return result


def extract_tide_triggers(h5, pdb):
    """Extract TIDE trigger residue IDs per site."""
    base = f"targets/{pdb}/binding_sites/tide_trigger_residues"
    if base not in h5:
        return {}
    result = {}
    for site_key in h5[base]:
        result[site_key] = h5[f"{base}/{site_key}"][:].tolist()
    return result


def extract_kcc(h5, pdb):
    """Extract KCC residue-level data."""
    grp = h5[f"targets/{pdb}/kcc"]
    residues_ds = grp["residues"]
    fields = residues_ds.dtype.names

    residues = []
    for row in residues_ds:
        r = {}
        for field in fields:
            val = row[field]
            if isinstance(val, (bytes, np.bytes_)):
                r[field] = decode_bytes(val)
            elif isinstance(val, (np.floating,)):
                r[field] = round(float(val), 6)
            else:
                r[field] = int(val)
        residues.append(r)
    return residues


def extract_kcc_validation(h5, pdb):
    """Extract KCC validation data (topk residues per validated site)."""
    base = f"targets/{pdb}/kcc_validation/sites"
    if base not in h5:
        return {}
    result = {}
    for site_key in h5[base]:
        topk_path = f"{base}/{site_key}/topk_residue_ids"
        if topk_path in h5:
            result[site_key] = h5[topk_path][:].tolist()
    return result


def extract_prism_therm(h5, pdb):
    """Extract PRISM-Therm per-pocket data."""
    base = f"targets/{pdb}/binding_sites/prism_therm_global"
    if base not in h5:
        return {}
    result = {}
    for site_key in h5[base]:
        site_grp = h5[f"{base}/{site_key}"]
        pocket = {}
        for attr_name in site_grp.attrs:
            val = site_grp.attrs[attr_name]
            if isinstance(val, (bytes, np.bytes_)):
                pocket[attr_name] = decode_bytes(val)
            elif isinstance(val, (np.floating, float)):
                v = float(val)
                pocket[attr_name] = round(v, 6) if np.isfinite(v) else None
            else:
                pocket[attr_name] = int(val) if isinstance(val, (np.integer, int)) else val

        # TIDE decomposition
        tide_path = f"{base}/{site_key}/tide_decomposition"
        if tide_path in h5:
            tide_ds = h5[tide_path]
            tide = []
            for row in tide_ds:
                entry = {}
                for field in tide_ds.dtype.names:
                    val = row[field]
                    if isinstance(val, (bytes, np.bytes_)):
                        entry[field] = decode_bytes(val)
                    elif isinstance(val, (np.floating,)):
                        entry[field] = round(float(val), 8)
                    else:
                        entry[field] = int(val)
                tide.append(entry)
            pocket["tide_decomposition"] = tide

        result[site_key] = pocket
    return result


def extract_spike_summary(h5, pdb):
    """Extract spike event summary (counts only, not full data)."""
    base = f"targets/{pdb}/spike_events"
    if base not in h5:
        return {}
    result = {}
    for site_key in h5[base]:
        site_obj = h5[f"{base}/{site_key}"]
        # May be a group with 'spikes' dataset inside, or a direct dataset
        if isinstance(site_obj, h5py.Group):
            if "spikes" in site_obj:
                ds = site_obj["spikes"]
            else:
                continue
        else:
            ds = site_obj

        n_spikes = ds.shape[0]
        if n_spikes == 0:
            result[site_key] = {"n_spikes": 0}
            continue

        # Extract summary stats from spike arrays
        fields = ds.dtype.names if ds.dtype.names else []
        summary = {"n_spikes": n_spikes}
        if "intensity" in fields:
            intensities = ds["intensity"]
            summary["intensity_mean"] = round(float(np.mean(intensities)), 4)
            summary["intensity_max"] = round(float(np.max(intensities)), 4)
            summary["intensity_p95"] = round(float(np.percentile(intensities, 95)), 4)
        if "source" in fields:
            sources = ds["source"]
            for s in range(4):
                count = int(np.sum(sources == s))
                if count > 0:
                    summary[f"source_{s}_count"] = count
        result[site_key] = summary
    return result


def extract_per_stream_stats(h5, pdb):
    """Extract per-stream statistics."""
    path = f"targets/{pdb}/binding_sites/per_stream_stats"
    if path not in h5:
        return None
    return h5[path][:].tolist()


# ── Report Building ──────────────────────────────────────────────────────────

def build_json_report(h5_path):
    """Build the full JSON report from HDF5."""
    h5 = h5py.File(h5_path, "r")

    # Metadata
    meta = {}
    if "metadata" in h5:
        for attr in h5["metadata"].attrs:
            meta[attr] = decode_bytes(h5["metadata"].attrs[attr])

    # Benchmark summary
    benchmark_summary = extract_benchmark_summary(h5)

    # Per-target data
    targets = {}
    target_list = [r["pdb"] for r in benchmark_summary]

    for pdb in target_list:
        info = TARGET_INFO.get(pdb, {"name": pdb, "type": "Unknown", "site_type": "Unknown", "literature": ""})

        sites = extract_sites(h5, pdb)
        lining = extract_lining_residues(h5, pdb)
        tide_triggers = extract_tide_triggers(h5, pdb)
        kcc = extract_kcc(h5, pdb)
        kcc_val = extract_kcc_validation(h5, pdb)
        therm = extract_prism_therm(h5, pdb)
        spikes = extract_spike_summary(h5, pdb)
        stream_stats = extract_per_stream_stats(h5, pdb)

        # Identify best cryptic site
        cryptic_sites = [s for s in sites if s.get("therm_class") == "CRYPTIC"]
        best_cryptic = None
        if cryptic_sites:
            best_cryptic = min(cryptic_sites, key=lambda s: s.get("cryptic_rank", 999))

        # Find matching benchmark row
        bm = next((r for r in benchmark_summary if r["pdb"] == pdb), {})

        targets[pdb] = {
            "info": info,
            "benchmark": bm,
            "n_sites": len(sites),
            "n_cryptic": len(cryptic_sites),
            "best_cryptic_site": best_cryptic,
            "sites": sites,
            "lining_residues": lining,
            "tide_trigger_residues": tide_triggers,
            "kcc_residues": kcc,
            "kcc_validation": kcc_val,
            "prism_therm": therm,
            "spike_summary": spikes,
            "per_stream_stats": stream_stats,
        }

    h5.close()

    report = {
        "report_type": "PRISM-4D Benchmark v4 — Full Data Export",
        "generated": datetime.now().isoformat(),
        "platform": "PRISM-4D (Rust/CUDA neuromorphic spike-driven molecular dynamics)",
        "engine": "nhs_rt_full",
        "protocol": {
            "thermal_protocol": "Cryo-UV: 50K hold → ramp to 300K → 300K hold → ramp to 50K → 50K return",
            "multi_stream": 8,
            "spike_percentile": 70,
            "flags": "--fast --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70 --fused-steps 6 --hmr --adaptive-dt --multi-differential --closed-loop-steering --asymmetric-steering --site-ranker phase-manifold --replica-seed 42 -v",
            "fused_steps": 6,
            "hmr": True,
            "adaptive_dt": True,
        },
        "scoring_systems": {
            "gtck_rank": "G×T×C×K×L lexicographic (geometry × thermo × causal × kinematic × localization)",
            "cryptic_rank": "CRYPTIC-aware: boosts therm_class=CRYPTIC sites, penalizes INERT",
            "composite_v3_rank": "V3 weighted composite (10 features)",
            "composite_audit_rank": "27-feature audit composite",
            "quality_score": "V7 12-signal ranker (burial, lining, log_spikes, ...)",
        },
        "therm_classes": {
            "CRYPTIC": "High hysteresis asymmetry + responds to thermal perturbation — pocket opens/closes with temperature",
            "DYNAMIC": "Moderate response, symmetric — pocket breathes but doesn't lock open/closed",
            "RESPONSIVE": "Low asymmetry, responds to heating — pocket modulates but is accessible",
            "INERT": "No thermal response — static pocket, likely always-open orthosteric site",
        },
        "metadata": meta,
        "benchmark_summary": benchmark_summary,
        "targets": targets,
    }

    return np_to_python(report)


def build_markdown_report(report):
    """Generate a human-readable markdown report."""
    lines = []
    a = lines.append

    a("# PRISM-4D Benchmark v4 — Comprehensive Report")
    a("")
    a(f"**Generated:** {report['generated']}")
    a(f"**Platform:** {report['platform']}")
    a(f"**Engine:** {report['engine']}")
    a("")

    # Protocol
    a("## 1. Simulation Protocol")
    a("")
    p = report["protocol"]
    a(f"- **Thermal protocol:** {p['thermal_protocol']}")
    a(f"- **Multi-stream:** {p['multi_stream']} independent streams per target")
    a(f"- **Spike percentile:** {p['spike_percentile']}th")
    a(f"- **Full command flags:** `{p['flags']}`")
    a("")

    # Scoring systems
    a("## 2. Scoring & Ranking Systems")
    a("")
    for k, v in report["scoring_systems"].items():
        a(f"- **{k}:** {v}")
    a("")

    # Therm classes
    a("## 3. Thermodynamic Classification (PRISM-Therm)")
    a("")
    for k, v in report["therm_classes"].items():
        a(f"- **{k}:** {v}")
    a("")

    # Benchmark summary table
    a("## 4. Benchmark Summary")
    a("")
    a("| PDB | Target | Tier | Sites | Cryptic | GT Overlap | Overlap% | Best Hysteresis | Best Cryptic Rank | Best GTCK Rank |")
    a("|-----|--------|------|-------|---------|------------|----------|-----------------|-------------------|----------------|")
    for r in report["benchmark_summary"]:
        pdb = r["pdb"]
        info = TARGET_INFO.get(pdb, {})
        name = info.get("name", pdb)
        # Get actual best cryptic rank from per-site data
        t = report["targets"].get(pdb, {})
        cryptic_sites = [s for s in t.get("sites", []) if s.get("therm_class") == "CRYPTIC"]
        best_cr = min((s.get("cryptic_rank", 999) for s in cryptic_sites), default=None)
        best_cr_str = str(int(best_cr)) if best_cr is not None else "—"
        # Also get best gtck rank from manifest-reported overlap sites
        best_gr = r.get("best_gtck_rank", 999)
        best_gr_str = str(best_gr) if best_gr < 999 else "—"
        a(f"| {pdb} | {name} | {r['tier']} | {r['n_sites']} | {r['n_cryptic']} | "
          f"{r['best_overlap']}/{r['gt_residues']} | {r['overlap_fraction']*100:.0f}% | "
          f"{r['best_hysteresis']:.3f} | {best_cr_str} | {best_gr_str} |")
    a("")

    # Accuracy summary
    hard4 = [r for r in report["benchmark_summary"] if r["tier"] == "hard4"]
    tier2 = [r for r in report["benchmark_summary"] if r["tier"] == "tier2"]
    all_targets = report["benchmark_summary"]

    detected_hard4 = sum(1 for r in hard4 if r["overlap_fraction"] > 0)
    detected_tier2 = sum(1 for r in tier2 if r["overlap_fraction"] > 0)
    rank1_cryptic = sum(1 for r in all_targets if r["best_cryptic_rank"] == 1 and r["overlap_fraction"] > 0)

    a("### Detection Accuracy")
    a("")
    a(f"- **Hard4 tier:** {detected_hard4}/{len(hard4)} targets with ground truth overlap")
    a(f"- **Tier2:** {detected_tier2}/{len(tier2)} targets with ground truth overlap")
    a(f"- **Rank-1 cryptic (with overlap):** {rank1_cryptic}/{len(all_targets)} targets")
    a("")

    # Per-target deep dive
    a("## 5. Per-Target Analysis")
    a("")

    for pdb in ["1bzj", "1r3m", "2iyt", "3uyi", "4epr", "2j1x", "1zg4"]:
        t = report["targets"].get(pdb)
        if not t:
            continue

        info = t["info"]
        bm = t["benchmark"]

        a(f"### {pdb.upper()} — {info['name']} ({info['type']})")
        a("")
        a(f"**Site type:** {info['site_type']}")
        a(f"**Literature:** {info['literature']}")
        a(f"**Detection:** {bm.get('best_overlap', 0)}/{bm.get('gt_residues', 0)} GT residues overlapping "
          f"({bm.get('overlap_fraction', 0)*100:.0f}%)")
        a(f"**Total sites:** {t['n_sites']} | **Cryptic pockets:** {t['n_cryptic']}")
        a("")

        # Top 5 sites by cryptic rank
        sorted_sites = sorted(t["sites"], key=lambda s: s.get("cryptic_rank", 999))
        a("#### Top 5 Sites (by cryptic_rank)")
        a("")
        a("| Rank | Site ID | Centroid (Å) | Therm Class | Hysteresis | τ (CCNS) | Drug. | Volume | Burial | Spikes | Quality |")
        a("|------|---------|-------------|-------------|------------|----------|-------|--------|--------|--------|---------|")

        for s in sorted_sites[:5]:
            cx = s.get("cx", 0)
            cy = s.get("cy", 0)
            cz = s.get("cz", 0)
            centroid = f"({cx:.1f}, {cy:.1f}, {cz:.1f})" if cx else "N/A"
            therm = s.get("therm_class", "?")
            hyst = s.get("hysteresis_asymmetry", 0)
            tau = s.get("ccns_tau", 0)
            drug = s.get("druggability", 0)
            vol = s.get("volume", 0)
            burial = s.get("burial_score", 0)
            spikes = s.get("spike_count", 0)
            qual = s.get("quality_score", 0)

            a(f"| {int(s.get('cryptic_rank', 0))} | {int(s.get('id', 0))} | {centroid} | "
              f"{therm} | {hyst:.3f} | {tau:.3f} | {drug:.3f} | {vol:.0f} | "
              f"{burial:.3f} | {int(spikes)} | {qual:.4f} |")
        a("")

        # Best cryptic site detail
        bc = t.get("best_cryptic_site")
        if bc:
            sid = int(bc.get("id", 0))
            a(f"#### Best Cryptic Site (ID {sid})")
            a("")
            a(f"- **Therm class:** {bc.get('therm_class', '?')}")
            a(f"- **Hysteresis asymmetry:** {bc.get('hysteresis_asymmetry', 0):.4f}")
            a(f"- **Relative asymmetry:** {bc.get('relative_asymmetry', 0):.4f}")
            a(f"- **CCNS tau:** {bc.get('ccns_tau', 0):.4f}")
            a(f"- **Druggability:** {bc.get('druggability', 0):.4f}")
            a(f"- **Burial score:** {bc.get('burial_score', 0):.4f}")
            a(f"- **Quality score:** {bc.get('quality_score', 0):.6f}")
            a(f"- **Breathing score:** {bc.get('breathing_score', 0):.4f}")
            a(f"- **Spike count:** {int(bc.get('spike_count', 0))}")
            a(f"- **ΔG effective:** {bc.get('effective_delta_g_kcal_mol')} kcal/mol")
            a(f"- **ΔG aromatic:** {bc.get('delta_g_aromatic_kcal_mol')} kcal/mol")
            a(f"- **Kinetic accessibility:** {bc.get('kinetic_accessibility', 0):.4f}")
            a(f"- **TIDE coupling:** {bc.get('tide_coupling_score', 0):.4f}")
            a("")

            # Lining residues for this site
            site_key = f"site_{sid}"
            lining = t["lining_residues"].get(site_key, [])
            if lining:
                a(f"**Lining residues ({len(lining)}):**")
                a("")
                cat_residues = [r for r in lining if r["is_catalytic"]]
                non_cat = [r for r in lining if not r["is_catalytic"]]
                if cat_residues:
                    cat_str = ", ".join(r["resname"] + str(r["resid"]) for r in cat_residues)
                    a(f"- Catalytic: {cat_str}")
                non_cat_str = ", ".join(r["resname"] + str(r["resid"]) for r in non_cat[:15])
                extra = f" (+{len(non_cat)-15} more)" if len(non_cat) > 15 else ""
                a(f"- Lining: {non_cat_str}{extra}")
                a("")

            # TIDE triggers
            triggers = t["tide_trigger_residues"].get(site_key, [])
            if triggers:
                a(f"**TIDE trigger residues:** {triggers}")
                a("")

        # PRISM-Therm breakdown
        therm = t.get("prism_therm", {})
        if therm:
            a("#### PRISM-Therm Pockets")
            a("")
            a("| Site | Therm Class | τ | Druggability | Asymmetry | Rel. Asymmetry | Heat Spikes | Cool Spikes |")
            a("|------|-------------|---|-------------|-----------|----------------|-------------|-------------|")
            for sk, pocket in sorted(therm.items()):
                tc = pocket.get("therm_class", pocket.get("ccns_classification", "?"))
                tau = pocket.get("tau", pocket.get("ccns_tau", 0)) or 0
                drug = pocket.get("druggability", 0) or 0
                asym = pocket.get("asymmetry_score", pocket.get("hysteresis_asymmetry", 0)) or 0
                rasym = pocket.get("relative_asymmetry", 0) or 0
                heat = pocket.get("heating_spike_count", 0) or 0
                cool = pocket.get("cooling_spike_count", 0) or 0
                sid = pocket.get("site_id", sk)
                a(f"| {sid} | {tc} | {tau:.3f} | {drug:.3f} | {asym:.3f} | {rasym:.3f} | {int(heat):,} | {int(cool):,} |")
            a("")

        # KCC summary — sort by burst_motion (weight field often unpopulated)
        kcc = t.get("kcc_residues", [])
        if kcc:
            top_kcc = sorted(kcc, key=lambda r: r.get("burst_motion", 0), reverse=True)[:10]
            a(f"#### Top 10 KCC Residues (by burst motion) — {len(kcc)} total")
            a("")
            a("| ResID | Name | Burst Motion | Direction | Lag Corr | Local Cov | Motion Eff. |")
            a("|-------|------|-------------|-----------|----------|-----------|-------------|")
            for r in top_kcc:
                a(f"| {r['residue_id']} | {r['residue_name']} | {r.get('burst_motion', 0):.4f} | "
                  f"{r.get('direction_score', 0):.4f} | {r.get('lag_corr_peak', 0):.4f} | "
                  f"{r.get('local_cov', 0):.4f} | {r.get('motion_efficiency', 0):.4f} |")
            a("")

        # Spike summary
        spk = t.get("spike_summary", {})
        if spk:
            total_spikes = sum(s.get("n_spikes", 0) for s in spk.values())
            a(f"#### Spike Events Summary — {total_spikes:,} total across {len(spk)} sites")
            a("")
            top_spike_sites = sorted(spk.items(), key=lambda x: x[1].get("n_spikes", 0), reverse=True)[:5]
            a("| Site | Spikes | Mean Int. | Max Int. | P95 Int. |")
            a("|------|--------|-----------|----------|----------|")
            for sk, ss in top_spike_sites:
                a(f"| {sk} | {ss['n_spikes']:,} | {ss.get('intensity_mean', 0):.3f} | "
                  f"{ss.get('intensity_max', 0):.3f} | {ss.get('intensity_p95', 0):.3f} |")
            a("")

        a("---")
        a("")

    # Cross-target comparison
    a("## 6. Cross-Target Feature Comparison")
    a("")
    a("### Best cryptic site features across all targets")
    a("")
    a("| PDB | Target | Hysteresis | τ | Burial | Quality | Spikes | Breathing | KinAccess |")
    a("|-----|--------|------------|---|--------|---------|--------|-----------|-----------|")

    for pdb in ["1bzj", "1r3m", "2iyt", "3uyi", "4epr", "2j1x", "1zg4"]:
        t = report["targets"].get(pdb)
        if not t:
            continue
        info = t["info"]
        bc = t.get("best_cryptic_site")
        if bc:
            a(f"| {pdb} | {info['name']} | {bc.get('hysteresis_asymmetry', 0):.3f} | "
              f"{bc.get('ccns_tau', 0):.3f} | {bc.get('burial_score', 0):.3f} | "
              f"{bc.get('quality_score', 0):.4f} | {int(bc.get('spike_count', 0))} | "
              f"{bc.get('breathing_score', 0):.3f} | {bc.get('kinetic_accessibility', 0):.3f} |")
        else:
            a(f"| {pdb} | {info['name']} | — | — | — | — | — | — | — |")
    a("")

    # Ranking comparison
    a("## 7. Ranking System Comparison")
    a("")
    a("### Best-overlap site rank by each scoring system")
    a("")
    a("| PDB | Target | GTCK Rank | Cryptic Rank | V3 Rank | Audit Rank | Quality Score |")
    a("|-----|--------|-----------|--------------|---------|------------|---------------|")

    for pdb in ["1bzj", "1r3m", "2iyt", "3uyi", "4epr", "2j1x", "1zg4"]:
        t = report["targets"].get(pdb)
        if not t:
            continue
        info = t["info"]
        # Find site with best overlap (highest catalytic count in lining)
        best_site = None
        best_cat = 0
        for s in t["sites"]:
            sid = int(s.get("id", 0))
            lining = t["lining_residues"].get(f"site_{sid}", [])
            cat_count = sum(1 for r in lining if r.get("is_catalytic"))
            if cat_count > best_cat:
                best_cat = cat_count
                best_site = s
        if best_site:
            a(f"| {pdb} | {info['name']} | {int(best_site.get('gtck_rank', 0))} | "
              f"{int(best_site.get('cryptic_rank', 0))} | "
              f"{int(best_site.get('composite_v3_rank', 0))} | "
              f"{int(best_site.get('composite_audit_rank', 0))} | "
              f"{best_site.get('quality_score', 0):.4f} |")
        else:
            a(f"| {pdb} | {info['name']} | — | — | — | — | — |")
    a("")

    # Known limitations
    a("## 8. Known Limitations")
    a("")
    a("- **1r3m (BS-RNase):** Monomer run — obligate dimer interface pocket requires merged A/B chains")
    a("- **2j1x (TP53 Y220C):** Cavity ~100Å³ too small for current protocol resolution; "
      "engine detects flanking dynamic region but not cavity interior")
    a("- **1zg4 (TEM-1 M182T):** Engine detects H10 helix (max signal h=1.000) but misses omega loop (163-179); "
      "correct sensitivity, incorrect specificity")
    a("- **EFP channel sparse:** ~9 spikes/10K steps; too few for per-pocket Jarzynski decomposition")
    a("- **delta_g values:** STI estimator produces unreliable values (e.g., delta_g_dewetting ~147 kcal/mol)")
    a("")

    # Data dictionary
    a("## 9. Data Dictionary")
    a("")
    a("### Site-level fields (57 features per site)")
    a("")
    a("| Field | Description | Range |")
    a("|-------|-------------|-------|")
    a("| id | Engine-assigned site ID | integer |")
    a("| gtck_rank | G×T×C×K×L lexicographic rank | 1 = best |")
    a("| cryptic_rank | CRYPTIC-aware rank | 1 = best |")
    a("| quality_score | V7 12-signal composite | 0.0–1.0 |")
    a("| volume | Pocket volume (ų) | >0 |")
    a("| burial_score | Fraction of buried surface | 0.0–1.0 |")
    a("| sphericity | Shape factor | 0.0–1.0 |")
    a("| hysteresis_asymmetry | Heating/cooling asymmetry (PRISM-Therm) | 0.0–1.0 |")
    a("| ccns_tau | CCNS correlation time | >0 |")
    a("| druggability | PRISM-Therm druggability estimate | 0.0–1.0 |")
    a("| breathing_score | Pocket volume fluctuation | >0 |")
    a("| spike_count | Total spikes detected in pocket | integer |")
    a("| kinetic_accessibility | Kinetic barrier estimate | 0.0–1.0 |")
    a("| tide_coupling_score | TIDE transfer entropy coupling | >0 |")
    a("| therm_class | CRYPTIC, DYNAMIC, RESPONSIVE, INERT | string |")
    a("| effective_delta_g_kcal_mol | Jarzynski ΔG (unreliable) | kcal/mol |")
    a("| engine_geo / engine_phys / engine_chem / engine_vcs | Engine sub-scores | 0.0–1.0 |")
    a("| rank_G / rank_T / rank_C / rank_K / rank_L | GTCKL component ranks | integer |")
    a("| cx, cy, cz | Pocket centroid coordinates (Å) | Å |")
    a("")
    a("### KCC residue fields")
    a("")
    a("| Field | Description |")
    a("|-------|-------------|")
    a("| residue_id | Topology residue index (0-based) |")
    a("| residue_name | 3-letter amino acid code |")
    a("| weight | Combined KCC weight |")
    a("| direction_score | Directional motion consistency |")
    a("| burst_motion | Burst motion amplitude |")
    a("| lag_corr_peak | Peak lagged cross-correlation |")
    a("| motion_efficiency | Net displacement / total displacement |")
    a("")

    a("---")
    a(f"*Report generated {report['generated']} by PRISM-4D v4 benchmark pipeline*")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate PRISM-4D v4 benchmark report")
    parser.add_argument("--h5", default="data/benchmark_v4/prism4d_v4_full.h5",
                        help="Path to HDF5 file")
    parser.add_argument("--output-dir", default="data/benchmark_v4",
                        help="Output directory for report files")
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        print(f"ERROR: {args.h5} not found")
        sys.exit(1)

    print(f"Extracting data from {args.h5}...")
    report = build_json_report(args.h5)

    # Write JSON
    json_path = os.path.join(args.output_dir, "prism4d_v4_report.json")
    print(f"Writing JSON report: {json_path}")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    json_size = os.path.getsize(json_path)
    print(f"  JSON size: {json_size / 1024 / 1024:.1f} MB")

    # Write Markdown
    md_path = os.path.join(args.output_dir, "prism4d_v4_report.md")
    print(f"Writing Markdown report: {md_path}")
    md_content = build_markdown_report(report)
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"  Markdown: {len(md_content)} chars")

    # Summary
    print()
    print("═" * 60)
    print("PRISM-4D Benchmark v4 Report Generated")
    print("═" * 60)
    print(f"  JSON: {json_path} ({json_size / 1024 / 1024:.1f} MB)")
    print(f"  Markdown: {md_path}")
    print()
    for r in report["benchmark_summary"]:
        pdb = r["pdb"]
        info = TARGET_INFO.get(pdb, {})
        name = info.get("name", pdb)
        status = "✓" if r["overlap_fraction"] > 0 else "✗"
        t = report["targets"].get(pdb, {})
        cryptic_sites = [s for s in t.get("sites", []) if s.get("therm_class") == "CRYPTIC"]
        best_cr = min((int(s.get("cryptic_rank", 999)) for s in cryptic_sites), default=None)
        cr_str = str(best_cr) if best_cr is not None else "—"
        print(f"  {status} {pdb} {name:25s} overlap={r['overlap_fraction']*100:3.0f}%  "
              f"cryptic_rank={cr_str}  h={r['best_hysteresis']:.3f}")
    print()


if __name__ == "__main__":
    main()
