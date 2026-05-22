#!/usr/bin/env python3
"""
PRISM-4D Phase Manifold Reranker v3
"""
import json, sys, math
from pathlib import Path
from collections import defaultdict

def load_json(path):
    if path is None: return None
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None

def sequence_blocks(residue_ids, gap=3):
    if not residue_ids: return 0
    s = sorted(residue_ids)
    blocks = 1
    for i in range(1, len(s)):
        if s[i] - s[i-1] > gap: blocks += 1
    return blocks

def euclidean(a, b):
    return math.sqrt(sum((ai-bi)**2 for ai,bi in zip(a,b)))

def p1_spatiotemporal(site):
    spikes = site.get("spike_count", 0)
    log_spikes = math.log1p(spikes)
    cpf = site.get("cold_phase_fraction", {})
    delta = abs(cpf.get("delta", 0.5))
    phase_signal = 0.5 + delta
    heat_rate = cpf.get("heating_spike_rate", 0)
    cool_rate = cpf.get("cooling_spike_rate", 0)
    rate_ratio = heat_rate / max(cool_rate, 1e-6)
    rate_signal = min(rate_ratio, 3.0) / 3.0
    breathing = site.get("breathing_score", 0)
    return log_spikes * phase_signal * (1.0 + rate_signal) * (1.0 + breathing)

def p2_mechanistic(site):
    causome = site.get("causome", {})
    kcc = site.get("kcc", {})
    supports = causome.get("candidate_residue_support", [])
    top_support = supports[0] if supports else 0.0
    kcc_conf = causome.get("kcc_confidence", 0.0)
    dir_score = causome.get("site_direction_score", 0.0)
    local_cov = causome.get("site_local_cov", 0.0)
    tide_score = site.get("tide_coupling_score", 0.0)
    n_tide = len(site.get("tide_trigger_residues", []))
    tide_bonus = min(n_tide, 5) * 0.1
    base = (1.0 + top_support) * (1.0 + kcc_conf) * (1.0 + dir_score) * (1.0 + local_cov)
    return base * (1.0 + tide_score) * (1.0 + tide_bonus)

def p3_thermodynamic(site, therm_pocket):
    tc = site.get("therm_class", "INERT")
    therm_weights = {"DYNAMIC": 1.5, "RESPONSIVE": 1.2, "INERT": 0.6}
    tw = therm_weights.get(tc, 0.8)
    drug = site.get("druggability", 0.5)
    hyst = site.get("hysteresis_asymmetry", 0.5)
    hyst_signal = 1.0 + abs(hyst - 0.5)
    causal_res_count = 0; gateway_count = 0; stabilizer_count = 0
    total_te = 0.0; total_kl = 0.0
    if therm_pocket:
        for r in therm_pocket.get("top_residues", []):
            if r.get("n_causal_spikes", 0) > 0: causal_res_count += 1
            role = r.get("role", "")
            if role == "GATEWAY": gateway_count += 1
            elif role == "STABIL.": stabilizer_count += 1
            total_te += r.get("transfer_entropy", 0.0)
            total_kl += r.get("kl_divergence", 0.0)
    role_signal = 1.0 + (gateway_count * 0.15) + (stabilizer_count * 0.2)
    causal_signal = 1.0 + min(causal_res_count, 15) * 0.05
    te_signal = 1.0 + math.log1p(total_te * 100)
    kl_signal = 1.0 + math.log1p(total_kl)
    return tw * drug * hyst_signal * role_signal * causal_signal * te_signal * kl_signal

def p4_topological(site):
    res_ids = site.get("residue_ids", [])
    n_blocks = sequence_blocks(res_ids)
    topo = min(n_blocks, 6) * 0.4
    if topo == 0: topo = 0.1
    burial = site.get("burial_score", 0.0)
    spher = site.get("sphericity", 0.0)
    arom = site.get("aromatic_score", 0.0)
    n_res = len(res_ids)
    if n_res < 5: size_mult = 0.3
    elif n_res > 60: size_mult = 0.6
    else: size_mult = 1.0
    cat_count = site.get("catalytic_residue_count", 0)
    cat_bonus = 1.0 + min(cat_count, 12) * 0.04
    return topo * (1.0 + burial) * (0.5 + spher) * (0.5 + arom) * size_mult * cat_bonus

def p5_causal_information(site, gcpid_residues, kcc_viz_residues):
    res_ids = set(site.get("residue_ids", []))
    if not res_ids: return 1.0
    syn_vals = [r.get("synergy_fraction",0.0) for r in gcpid_residues if r.get("residue_id") in res_ids]
    avg_synergy = sum(syn_vals)/max(len(syn_vals),1) if syn_vals else 0.0
    kcc_scores=[]; dir_scores=[]; motion_effs=[]
    for r in kcc_viz_residues:
        if r.get("residue_id") in res_ids:
            kcc_scores.append(r.get("kcc_score",0.0))
            dir_scores.append(r.get("direction_score",0.0))
            motion_effs.append(r.get("motion_efficiency",0.0))
    avg_kcc = sum(kcc_scores)/max(len(kcc_scores),1) if kcc_scores else 0.0
    avg_dir = sum(dir_scores)/max(len(dir_scores),1) if dir_scores else 0.0
    avg_meff = sum(motion_effs)/max(len(motion_effs),1) if motion_effs else 0.0
    return (1.0+avg_synergy)*(1.0+avg_kcc)*(1.0+avg_dir*2)*(1.0+avg_meff*10)

def p6_consensus(site, kcc_val_sites, asc_residues):
    site_id = site.get("id")
    res_ids = set(site.get("residue_ids", []))
    kcc_verdict = "UNKNOWN"; kcc_rank_score = 0.0
    for vs in kcc_val_sites:
        if vs.get("site_id") == site_id:
            kcc_verdict = vs.get("verdict","UNKNOWN")
            kcc_rank_score = vs.get("rank_score",0.0)
            break
    verdict_mult = {"PASS":2.0,"PARTIAL":1.2,"FAIL":0.5,"UNKNOWN":0.8}
    vm = verdict_mult.get(kcc_verdict, 0.8)
    overlap = 0; total_spc = 0.0
    for ar in asc_residues:
        if ar.get("residue_id") in res_ids:
            overlap += 1; total_spc += ar.get("s_pc",0.0)
    consensus_signal = 1.0 + overlap*0.3 + total_spc*0.1
    engine_chem = site.get("engine_chem",0.0)
    engine_geo = site.get("engine_geo",0.0)
    engine_phys = site.get("engine_phys",0.0)
    engine_vcs = site.get("engine_vcs",0.0)
    engine_avg = (engine_chem+engine_geo+engine_phys+engine_vcs)/4.0
    return vm * consensus_signal * (1.0+kcc_rank_score) * (1.0+engine_avg*0.5)

def compute_dcc_views(site, ligand_centroid):
    if not ligand_centroid: return {}
    views = {}
    gc = site.get("centroid")
    if gc: views["geometric"] = euclidean(gc, ligand_centroid)
    lining = site.get("lining_residues", [])
    if lining: views["lining_min"] = min(r.get("min_distance",999) for r in lining)
    causome = site.get("causome", {})
    driver_id = causome.get("driver_residue_id")
    if driver_id is not None and lining:
        for r in lining:
            if r.get("resid") == driver_id:
                views["driver_min_dist"] = r.get("min_distance",999)
                break
    return views

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 reranker_v3.py <run_output_dir> [output.json]")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    def find_file(pattern):
        matches = list(run_dir.glob(pattern))
        return matches[0] if matches else None
    bs_file = find_file("*.binding_sites.json")
    kcc_val_file = find_file("*.kcc_validation.json")
    kcc_viz_file = find_file("*.kcc_visualization.json")
    therm_file = find_file("*.topology.prism_therm.json")
    gcpid_file = find_file("*.topology.gcpid_synergy.json")
    asc_file = find_file("*.topology.asc_consensus.json")
    gt_file = find_file("*ground_truth.json")
    if not bs_file:
        print(f"FATAL: No *.binding_sites.json found in {run_dir}")
        sys.exit(1)
    bs_data = load_json(bs_file)
    kcc_val_data = load_json(kcc_val_file) or {}
    kcc_viz_data = load_json(kcc_viz_file) or {}
    therm_data = load_json(therm_file) or {}
    gcpid_data = load_json(gcpid_file) or {}
    asc_data = load_json(asc_file) or {}
    gt_data = load_json(gt_file) or {}
    sites = bs_data.get("sites", [])
    if not sites:
        print("FATAL: No sites in binding_sites.json")
        sys.exit(1)
    therm_pockets = {p["pocket_id"]: p for p in therm_data.get("pockets", [])}
    kcc_val_sites = kcc_val_data.get("sites", [])
    kcc_viz_residues = kcc_viz_data.get("residues", [])
    gcpid_residues = gcpid_data.get("residues", [])
    asc_residues = asc_data.get("consensus_residues", [])
    ligand_centroid = gt_data.get("ligand_centroid")
    print(f"Run dir: {run_dir}")
    print(f"Sites: {len(sites)}  |  KCC-val sites: {len(kcc_val_sites)}  |  "
          f"KCC-viz residues: {len(kcc_viz_residues)}  |  Therm pockets: {len(therm_pockets)}")
    print(f"GCPID residues: {len(gcpid_residues)}  |  ASC consensus: {len(asc_residues)}  |  "
          f"Ground truth: {'YES' if ligand_centroid else 'NO'}")
    results = []
    for site in sites:
        sid = site["id"]
        therm_pocket = therm_pockets.get(sid)
        s1 = p1_spatiotemporal(site)
        s2 = p2_mechanistic(site)
        s3 = p3_thermodynamic(site, therm_pocket)
        s4 = p4_topological(site)
        s5 = p5_causal_information(site, gcpid_residues, kcc_viz_residues)
        s6 = p6_consensus(site, kcc_val_sites, asc_residues)
        raw = s1*s2*s3*s4*s5*s6
        composite = raw**(1.0/6.0) if raw > 0 else 0.0
        dcc_views = compute_dcc_views(site, ligand_centroid)
        results.append({
            "site_id": sid, "composite_v4": composite,
            "pillars": {"P1_spatiotemporal":round(s1,4),"P2_mechanistic":round(s2,4),
                        "P3_thermodynamic":round(s3,4),"P4_topological":round(s4,4),
                        "P5_causal_info":round(s5,4),"P6_consensus":round(s6,4)},
            "classification": site.get("classification","?"),
            "therm_class": site.get("therm_class","?"),
            "spike_count": site.get("spike_count",0),
            "engine_v3_rank": site.get("composite_v3_rank"),
            "engine_gtck_rank": site.get("gtck_rank"),
            "dcc_views": dcc_views,
        })
    results.sort(key=lambda x: x["composite_v4"], reverse=True)
    print(f"\n{'V4':>3} {'ID':>5} {'Composite':>10} {'P1-Spatio':>10} {'P2-Mech':>8} "
          f"{'P3-Therm':>9} {'P4-Topo':>8} {'P5-Causal':>10} {'P6-Cons':>8} "
          f"{'Class':<12} {'Therm':<10} {'v3rk':>4} {'gtck':>4}")
    print("="*140)
    for rank, r in enumerate(results, 1):
        p = r["pillars"]
        print(f"{rank:>3} {r['site_id']:>5} {r['composite_v4']:>10.3f} "
              f"{p['P1_spatiotemporal']:>10.2f} {p['P2_mechanistic']:>8.3f} "
              f"{p['P3_thermodynamic']:>9.3f} {p['P4_topological']:>8.3f} "
              f"{p['P5_causal_info']:>10.3f} {p['P6_consensus']:>8.3f} "
              f"{r['classification']:<12} {r['therm_class']:<10} "
              f"{r.get('engine_v3_rank','?'):>4} {r.get('engine_gtck_rank','?'):>4}")
    if ligand_centroid:
        print(f"\n{'V4':>3} {'ID':>5} {'DCC-geom':>10} {'DCC-lining':>11} {'DCC-driver':>11}")
        print("-"*50)
        for rank, r in enumerate(results, 1):
            dv = r.get("dcc_views",{})
            geo = f"{dv.get('geometric',-1):.2f}" if "geometric" in dv else "n/a"
            lin = f"{dv.get('lining_min',-1):.2f}" if "lining_min" in dv else "n/a"
            drv = f"{dv.get('driver_min_dist',-1):.2f}" if "driver_min_dist" in dv else "n/a"
            print(f"{rank:>3} {r['site_id']:>5} {geo:>10} {lin:>11} {drv:>11}")
    if output_path:
        out = {"schema_kind":"prism4d_reranked_v4","run_dir":str(run_dir),
               "ground_truth_available":ligand_centroid is not None,
               "ligand_centroid":ligand_centroid,"ranked_sites":results}
        output_path.write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    main()
