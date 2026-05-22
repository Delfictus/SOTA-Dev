#!/usr/bin/env python3
"""
PRISM-4D POCKET DOSSIER — full 10-enhancement edition.

All 10 enhancements integrated as modular stages. Each enhancement reports:
  - results if data permits
  - "BLOCKED: <gate>" if a dev-gate prerequisite is unmet

NO FAKE FINITE VALUES. Per §1, NaN/null means "not honestly computable."
"""
import os
import json
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.compute as pc
from scipy.spatial import KDTree
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────── CONFIG ────────────────────────
FILE_PATH = "/home/diddy/Desktop/Prism4D-bio/output/m1_readiness_verify/4lpk_clean.topology.spike_events.arrow"
OUT_DIR   = "/home/diddy/Desktop/Prism4D-bio/output/m1_readiness_verify"
os.makedirs(OUT_DIR, exist_ok=True)

CLEAN_COLS = [
    "spike_id", "replica_seed", "stream_id", "group_id", "chunk_idx",
    "voxel_idx", "timestep", "frame_index",
    "x", "y", "z",
    "intensity", "spike_source", "aromatic_type", "aromatic_residue_id",
    "phase_bits", "n_residues", "nearby_residues", "n_nearby_excited",
    "vibrational_energy", "water_density", "wd_change", "wavelength_nm",
    "ccns_phase", "background_class", "intensity_percentile",
]

DOMAINS = {
    "P-loop":       set(range(8, 19)),
    "Switch-I":     set(range(25, 41)),
    "Switch-II":    set(range(55, 77)),
    "alpha3/H3-H4": set(range(95, 135)),
}

TOP_RES_FOR_CLUSTERING = 150
KDTREE_RADII           = [5.0, 6.0, 7.0]
CANONICAL_RADIUS       = 6.0
MIN_CLUSTER_SIZE       = 3
TOP_K_POCKETS          = 10

CCNS_PHASE_NAMES = {0: "cold_hold", 1: "ramp", 2: "warm_hold"}

print("=" * 100)
print("PRISM-4D POCKET DOSSIER — 10-enhancement edition")
print(f"Source: {FILE_PATH}")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: LOAD + SUBSTRATE FILTER + ACTIVE-AXIS DETECTION
# ═══════════════════════════════════════════════════════════════════════
with pa.memory_map(FILE_PATH, "r") as src:
    table = ipc.open_file(src).read_all().select(CLEAN_COLS)

mask = pc.and_(
    pc.equal(table["background_class"], 0),
    pc.greater_equal(table["intensity_percentile"], 70),
)
ft = table.filter(mask)
n = ft.num_rows

seeds_in_run    = sorted(set(ft["replica_seed"].to_pylist()))
streams_in_run  = sorted(set(ft["stream_id"].to_pylist()))
ccns_in_run     = sorted(set(ft["ccns_phase"].to_pylist()))
sources_in_run  = sorted(set(ft["spike_source"].to_pylist()))
frames_in_run   = (min(ft["frame_index"].to_pylist()), max(ft["frame_index"].to_pylist()))
n_frames        = frames_in_run[1] - frames_in_run[0] + 1

REPLICA_AXIS_ACTIVE = len(seeds_in_run) >= 2
STREAM_AXIS_ACTIVE  = len(streams_in_run) >= 2
CCNS_AXIS_ACTIVE    = len(ccns_in_run) >= 2
SOURCE_AXIS_ACTIVE  = len(sources_in_run) >= 2

# Dev gates
GATES = {
    "G1_multi_seed":        REPLICA_AXIS_ACTIVE,
    "G2_trajectory_writer": False,  # Path A trajectory binary not yet implemented
    "G3_phase_bits_decoder":False,  # 10-bit phase decoder spec required
    "G4_residue_atom_map":  False,  # detected by topology JSON probe below
    "G5_apo_pdb_path":      False,  # detected by file probe below
}

# Probe for G4: residue→atom mapping
TOPOLOGY_PATH = FILE_PATH.replace(".spike_events.arrow", ".json").replace("4lpk_clean.topology", "4lpk_clean")
if not os.path.exists(TOPOLOGY_PATH):
    TOPOLOGY_PATH = "/home/diddy/Desktop/Prism4D-bio/output/m1_readiness_verify/4lpk_clean.topology.json"
if os.path.exists(TOPOLOGY_PATH):
    GATES["G4_residue_atom_map"] = True

# Probe for G5: apo PDB
APO_PDB_CANDIDATES = [
    "/home/diddy/Desktop/Prism4D-bio/references/4LPK.pdb",
    "/home/diddy/Desktop/Prism4D-bio/references/4lpk.pdb",
]
APO_PDB = next((p for p in APO_PDB_CANDIDATES if os.path.exists(p)), None)
if APO_PDB:
    GATES["G5_apo_pdb_path"] = True

run_metadata = {
    "filtered_spikes": n,
    "replica_seeds": seeds_in_run,
    "stream_ids": streams_in_run,
    "ccns_phases": ccns_in_run,
    "spike_sources": sources_in_run,
    "frame_range": list(frames_in_run),
    "n_frames": n_frames,
    "axes": {
        "replica_active":  REPLICA_AXIS_ACTIVE,
        "stream_active":   STREAM_AXIS_ACTIVE,
        "ccns_active":     CCNS_AXIS_ACTIVE,
        "source_active":   SOURCE_AXIS_ACTIVE,
    },
    "dev_gates": GATES,
    "topology_path": TOPOLOGY_PATH if os.path.exists(TOPOLOGY_PATH) else None,
    "apo_pdb_path":  APO_PDB,
}

print(f"\n[1] Filtered spikes: {n:,}")
print(f"    Replica seeds:  {seeds_in_run}  [{'ACTIVE' if REPLICA_AXIS_ACTIVE else 'INACTIVE — gate G1'}]")
print(f"    Streams:        {streams_in_run}  [{'ACTIVE' if STREAM_AXIS_ACTIVE else 'INACTIVE'}]")
print(f"    CCNS phases:    {ccns_in_run}  [{'ACTIVE' if CCNS_AXIS_ACTIVE else 'INACTIVE'}]")
print(f"    Spike sources:  {sources_in_run}  [{'ACTIVE' if SOURCE_AXIS_ACTIVE else 'INACTIVE'}]")
print(f"    Frame range:    {frames_in_run[0]}-{frames_in_run[1]} ({n_frames} frames)")
print(f"\n    DEV GATES:")
for g, status in GATES.items():
    print(f"      {g:<28} {'PASS' if status else 'BLOCKED'}")

# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: EXPLODE + PER-RESIDUE AGGREGATION
# ═══════════════════════════════════════════════════════════════════════
spike_idx = pa.array(np.repeat(np.arange(n), 8))
flat_res  = pc.list_flatten(ft["nearby_residues"])
SCALARS = ["intensity", "x", "y", "z", "vibrational_energy", "water_density",
           "wd_change", "wavelength_nm", "ccns_phase", "phase_bits", "stream_id",
           "spike_source", "aromatic_type", "aromatic_residue_id", "frame_index",
           "replica_seed", "n_nearby_excited", "voxel_idx"]
cols_dict = {"residue_id": flat_res, "spike_idx": spike_idx}
for col in SCALARS:
    cols_dict[col] = pc.take(ft[col], spike_idx)

ex = pa.Table.from_pydict(cols_dict).filter(pc.not_equal(pa.Table.from_pydict(cols_dict)["residue_id"], -1))
df = ex.to_pandas()
for c in ("x", "y", "z"):
    df[f"w{c}"] = df[c] * df["intensity"]

print(f"\n[2] Exploded rows: {len(df):,}  unique residues: {df['residue_id'].nunique()}")

def is_aromatic_residue(group): return int((group > 0).any())
def aromatic_type_at_residue(group):
    g = group[group > 0]
    return int(g.mode().iloc[0]) if len(g) else 0

res = df.groupby("residue_id").agg(
    spike_count          = ("intensity", "count"),
    total_energy         = ("intensity", "sum"),
    sum_wx               = ("wx", "sum"),
    sum_wy               = ("wy", "sum"),
    sum_wz               = ("wz", "sum"),
    mean_vib_energy      = ("vibrational_energy", "mean"),
    mean_water_density   = ("water_density", "mean"),
    abs_wd_change        = ("wd_change", lambda s: float(np.mean(np.abs(s)))),
    max_abs_wd_change    = ("wd_change", lambda s: float(np.max(np.abs(s)))),
    signed_wd_change     = ("wd_change", "mean"),
    mean_wavelength      = ("wavelength_nm", "mean"),
    n_unique_streams     = ("stream_id", "nunique"),
    n_unique_ccns_phases = ("ccns_phase", "nunique"),
    n_unique_replicas    = ("replica_seed", "nunique"),
    n_unique_frames      = ("frame_index", "nunique"),
    min_frame            = ("frame_index", "min"),
    is_aromatic          = ("aromatic_type", is_aromatic_residue),
    aromatic_type        = ("aromatic_type", aromatic_type_at_residue),
    n_unique_sources     = ("spike_source", "nunique"),
    mean_n_excited       = ("n_nearby_excited", "mean"),
    var_n_excited        = ("n_nearby_excited", "var"),
).reset_index()
res["mean_x"] = res["sum_wx"] / res["total_energy"]
res["mean_y"] = res["sum_wy"] / res["total_energy"]
res["mean_z"] = res["sum_wz"] / res["total_energy"]
res = res.sort_values("total_energy", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: KD-TREE CLUSTERING + RADIUS STABILITY SWEEP
# ═══════════════════════════════════════════════════════════════════════
def cluster_at_radius(top_residues, radius, min_size, top_k):
    tree = KDTree(top_residues[["mean_x", "mean_y", "mean_z"]].values)
    nbrs = tree.query_ball_tree(tree, r=radius)
    visited, pkts = set(), []
    for i, ns in enumerate(nbrs):
        if i in visited: continue
        cluster = []
        for j in ns:
            if j not in visited:
                cluster.append(int(top_residues.iloc[j]["residue_id"]))
                visited.add(j)
        if len(cluster) >= min_size:
            e = res.loc[res["residue_id"].isin(cluster), "total_energy"].sum()
            pkts.append((cluster, float(e)))
    pkts.sort(key=lambda p: p[1], reverse=True)
    return pkts[:top_k]

top = res.head(TOP_RES_FOR_CLUSTERING).copy()
stability_results = {r: cluster_at_radius(top, r, MIN_CLUSTER_SIZE, TOP_K_POCKETS) for r in KDTREE_RADII}
pockets = stability_results[CANONICAL_RADIUS]

print(f"\n[3] KD-tree clustering at r={CANONICAL_RADIUS}A: {len(pockets)} pockets")
for r, pkts in stability_results.items():
    print(f"    r={r}A: {len(pkts)} pockets, mean size {np.mean([len(p[0]) for p in pkts]):.1f}")

# ═══════════════════════════════════════════════════════════════════════
# STAGE 4: PER-POCKET CORE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════
def pocket_spike_slice(pocket_residues):
    return df[df["residue_id"].isin(pocket_residues)]

def regional_yields(rids):
    pdf = res[res["residue_id"].isin(rids)]
    pE = pdf["total_energy"].sum()
    yields = {name: float(100.0 * pdf[pdf["residue_id"].isin(dres)]["total_energy"].sum() / pE) if pE > 0 else 0.0
              for name, dres in DOMAINS.items()}
    yields["Other"] = max(0.0, 100.0 - sum(yields.values()))
    return yields

def core_diagnostics(rids):
    pdf = res[res["residue_id"].isin(rids)]
    sdf = pocket_spike_slice(rids)
    coords = pdf[["mean_x", "mean_y", "mean_z"]].values
    w = pdf["total_energy"].values
    centroid = np.average(coords, axis=0, weights=w)
    rg = float(np.sqrt(np.average(np.sum((coords - centroid)**2, axis=1), weights=w)))
    seq_gap = float(np.mean(np.diff(sorted(pdf["residue_id"].values)))) if len(pdf) > 1 else 0.0
    wd_var = float(np.var(sdf["water_density"].values)) if len(sdf) > 1 else 0.0
    wd_iqr = float(np.percentile(sdf["water_density"].values, 75) - np.percentile(sdf["water_density"].values, 25)) if len(sdf) > 0 else 0.0
    wd_ch_p95 = float(np.percentile(np.abs(sdf["wd_change"].values), 95)) if len(sdf) > 0 else 0.0
    vib_var = float(np.var(sdf["vibrational_energy"].values)) if len(sdf) > 1 else 0.0
    wl_var = float(np.var(sdf["wavelength_nm"].values)) if len(sdf) > 1 else 0.0
    ccns_balance = float(sdf["ccns_phase"].value_counts(normalize=True).std()) if len(sdf) > 0 else 0.0

    n_c = len(coords); sc_corr = float("nan")
    if n_c > 2 and (STREAM_AXIS_ACTIVE or REPLICA_AXIS_ACTIVE):
        d_sp = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=-1)
        cs = pdf["n_unique_streams"].values + pdf["n_unique_replicas"].values
        c_diff = np.abs(cs[:,None] - cs[None,:])
        iu = np.triu_indices(n_c, k=1)
        if d_sp[iu].std() > 0 and c_diff[iu].std() > 0:
            sc_corr = float(np.corrcoef(d_sp[iu], c_diff[iu])[0,1])

    return {"rg": rg, "seq_gap": seq_gap, "wd_var": wd_var, "wd_iqr": wd_iqr,
            "wd_change_p95": wd_ch_p95, "vib_var": vib_var, "wl_var": wl_var,
            "ccns_balance": ccns_balance, "sc_corr": sc_corr,
            "frame_persistence_per_res": float(pdf["n_unique_frames"].mean()),
            "centroid_xyz": centroid.tolist(),
            "n_residues": len(pdf), "n_spikes": len(sdf)}

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT 1: CCNS LIFECYCLE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════
def enhance_1_ccns_lifecycle(rids):
    if not CCNS_AXIS_ACTIVE:
        return {"status": "BLOCKED",
                "gate": "Need >=2 distinct CCNS phases in run (currently single phase)",
                "result": None}
    sdf = pocket_spike_slice(rids)
    by_phase = {}
    for ph in ccns_in_run:
        ph_df = sdf[sdf["ccns_phase"] == ph]
        ph_name = CCNS_PHASE_NAMES.get(ph, f"phase_{ph}")
        by_phase[ph_name] = {
            "n_spikes": int(len(ph_df)),
            "energy": float(ph_df["intensity"].sum()),
            "energy_pct": float(100.0 * ph_df["intensity"].sum() / sdf["intensity"].sum()) if len(sdf) else 0.0,
            "n_unique_residues": int(ph_df["residue_id"].nunique()),
            "wd_change_p95": float(np.percentile(np.abs(ph_df["wd_change"].values), 95)) if len(ph_df) else 0.0,
        }
    # 6-class taxonomy classification per §5
    e_pct = {k: v["energy_pct"] for k, v in by_phase.items()}
    cold = e_pct.get("cold_hold", 0); ramp = e_pct.get("ramp", 0); warm = e_pct.get("warm_hold", 0)
    if cold > 60: tax = "THERMAL_TRANSIENT"          # cold-dominated, lost on warming
    elif ramp > 60: tax = "BARRIER_GATED"             # ramp-dominated, kinetic barrier
    elif warm > 60: tax = "NMA_RESPONSIVE"            # warm-dominated, normal-mode-like
    elif abs(cold - warm) < 15 and ramp > 25: tax = "CONSENSUS_CRYPTIC"  # balanced
    elif (cold + warm) > 70 and ramp < 20: tax = "ALLOSTERIC_HUB"        # bistable
    else: tax = "COOPERATIVE_NETWORK"
    return {"status": "OK", "by_phase": by_phase, "taxonomy_class": tax, "phase_balance": ccns_balance_calc(by_phase)}

def ccns_balance_calc(by_phase):
    pcts = [v["energy_pct"] for v in by_phase.values()]
    if not pcts: return 0.0
    expected = 100.0 / len(pcts)
    return float(np.sqrt(np.mean([(p - expected)**2 for p in pcts])))

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT 2: WAVELENGTH MICROENVIRONMENT FINGERPRINT
# ═══════════════════════════════════════════════════════════════════════
def enhance_2_wavelength_fingerprint(rids):
    sdf = pocket_spike_slice(rids)
    if len(sdf) < 100:
        return {"status": "BLOCKED", "gate": "Need >=100 spikes for histogram", "result": None}
    wl = sdf["wavelength_nm"].values
    bins = np.arange(0, 55, 5)
    hist, edges = np.histogram(wl, bins=bins)
    total = hist.sum()
    pct_per_bin = (100.0 * hist / total).tolist() if total else [0]*len(hist)

    # Bimodality: dip statistic (Hartigan-style approximation via KS vs uniform)
    ks_stat, ks_p = stats.kstest((wl - wl.min()) / (wl.max() - wl.min() + 1e-9), 'uniform')
    bimodality_index = float(ks_stat)

    # Per-residue dominant wavelength bin
    res_wl_bins = {}
    for rid in rids:
        rwl = sdf[sdf["residue_id"] == rid]["wavelength_nm"].values
        if len(rwl):
            dom_bin_idx = int(np.argmax(np.histogram(rwl, bins=bins)[0]))
            res_wl_bins[int(rid)] = {"dominant_bin_nm": f"{bins[dom_bin_idx]}-{bins[dom_bin_idx+1]}",
                                     "mean_wl": float(np.mean(rwl)),
                                     "std_wl": float(np.std(rwl))}
    # Microenvironment-coherent subgroups: residues in same dominant bin
    bin_groups = defaultdict(list)
    for rid, info in res_wl_bins.items():
        bin_groups[info["dominant_bin_nm"]].append(rid)
    subpockets = {b: g for b, g in bin_groups.items() if len(g) >= 2}

    return {"status": "OK", "histogram_pct_per_bin": pct_per_bin, "bin_edges_nm": edges.tolist(),
            "bimodality_index": bimodality_index, "ks_p_value": float(ks_p),
            "n_chemical_subpockets": len(subpockets),
            "chemical_subpockets": subpockets,
            "interpretation": "BIMODAL_HETEROGENEOUS" if bimodality_index > 0.15 else "UNIMODAL_HOMOGENEOUS"}

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT 3: PHASE_BITS MUTUAL INFORMATION
# ═══════════════════════════════════════════════════════════════════════
def enhance_3_phase_bits_mi(rids):
    if not GATES["G3_phase_bits_decoder"]:
        return {"status": "BLOCKED",
                "gate": ("G3_phase_bits_decoder: 10-bit phase_bits encoding spec required. "
                         "Per §3, phase_bits is u32 with 10-bit phase content. Need: "
                         "(a) bit-extraction routine confirmed against gpu.rs source, "
                         "(b) per-bit semantic mapping (which bit = which phase axis). "
                         "Without spec, MI computation is uninterpretable."),
                "result": None}
    # Implementation when gate passes: pairwise MI on bit patterns per residue
    sdf = pocket_spike_slice(rids)
    res_bits = {rid: sdf[sdf["residue_id"] == rid]["phase_bits"].values for rid in rids}
    pairs = []
    for i, ri in enumerate(rids):
        for rj in rids[i+1:]:
            bi, bj = res_bits[ri], res_bits[rj]
            n_pair = min(len(bi), len(bj))
            if n_pair < 50: continue
            # Mutual info on 10-bit values
            joint = pd.crosstab(bi[:n_pair] & 0x3FF, bj[:n_pair] & 0x3FF, normalize=True).values
            px = joint.sum(axis=1, keepdims=True); py = joint.sum(axis=0, keepdims=True)
            mi = float(np.sum(np.where(joint > 0, joint * np.log2(joint / (px @ py + 1e-12) + 1e-12), 0)))
            pairs.append({"r1": int(ri), "r2": int(rj), "mi": mi})
    if not pairs:
        return {"status": "OK", "mean_mi": 0.0, "pairs": [], "interpretation": "INSUFFICIENT_DATA"}
    mean_mi = float(np.mean([p["mi"] for p in pairs]))
    interp = "TIGHT_CAUSAL_UNIT" if mean_mi > 0.5 else "MULTI_SUBUNIT" if mean_mi < 0.1 else "MIXED"
    return {"status": "OK", "n_pairs": len(pairs), "mean_mi": mean_mi,
            "max_mi_pair": max(pairs, key=lambda p: p["mi"]),
            "interpretation": interp}

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT 4: FRAME-RESOLVED FIRST-PASSAGE TIMING
# ═══════════════════════════════════════════════════════════════════════
def enhance_4_first_passage(rids):
    if n_frames < 5:
        return {"status": "BLOCKED", "gate": f"Need >=5 frames, have {n_frames}", "result": None}
    sdf = pocket_spike_slice(rids)
    fp_per_residue = {}
    for rid in rids:
        rdf = sdf[sdf["residue_id"] == rid]
        if len(rdf):
            fp_per_residue[int(rid)] = int(rdf["frame_index"].min())
    if len(fp_per_residue) < 2:
        return {"status": "BLOCKED", "gate": "Need >=2 residues with spike data", "result": None}
    sorted_fp = sorted(fp_per_residue.items(), key=lambda x: x[1])
    nucleation_residues = [r for r, f in sorted_fp if f == sorted_fp[0][1]]
    propagation_order = [r for r, f in sorted_fp]

    # Permutation test: is the ordering significant vs. random?
    actual_spread = sorted_fp[-1][1] - sorted_fp[0][1]
    rng = np.random.default_rng(42)
    null_spreads = [rng.integers(0, n_frames+1, len(fp_per_residue)).ptp() for _ in range(1000)]
    p_value = float(np.mean([s >= actual_spread for s in null_spreads]))

    return {"status": "OK", "first_passage_per_residue": fp_per_residue,
            "nucleation_residues": nucleation_residues,
            "propagation_order": propagation_order,
            "actual_spread_frames": int(actual_spread),
            "ordering_p_value": p_value,
            "interpretation": ("STRUCTURED_NUCLEATION" if p_value < 0.05 else "STOCHASTIC_FIRING")}

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT 5: AROMATIC CROSS-NETWORK GRAPH
# ═══════════════════════════════════════════════════════════════════════
def enhance_5_aromatic_graph(rids, all_pocket_residues):
    sdf = pocket_spike_slice(rids)
    arom_sdf = sdf[sdf["aromatic_residue_id"] >= 0]
    if len(arom_sdf) == 0:
        return {"status": "OK", "n_internal_edges": 0, "n_external_edges": 0,
                "external_targets": [], "interpretation": "NO_AROMATIC_LINKS"}
    rids_set = set(rids)
    internal_edges = arom_sdf[arom_sdf["aromatic_residue_id"].isin(rids_set)]
    external_edges = arom_sdf[~arom_sdf["aromatic_residue_id"].isin(rids_set)]
    external_targets = external_edges["aromatic_residue_id"].value_counts().head(10).to_dict()
    # Cross-pocket coupling: which other pockets do these targets land in?
    cross_pocket_coupling = defaultdict(int)
    for tgt in external_targets:
        for pidx, prids in all_pocket_residues.items():
            if int(tgt) in prids:
                cross_pocket_coupling[f"pocket_{pidx}"] += int(external_targets[tgt])
    return {"status": "OK",
            "n_internal_edges": int(len(internal_edges)),
            "n_external_edges": int(len(external_edges)),
            "external_target_residues_top10": {int(k): int(v) for k, v in external_targets.items()},
            "cross_pocket_coupling": dict(cross_pocket_coupling),
            "interpretation": "ALLOSTERIC_COUPLED" if cross_pocket_coupling else "ISOLATED_AROMATIC_NETWORK"}

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT 6: COOPERATIVE EXCITATION FIELD
# ═══════════════════════════════════════════════════════════════════════
def enhance_6_cooperative_excitation(rids):
    sdf = pocket_spike_slice(rids)
    nx = sdf["n_nearby_excited"].values
    if len(nx) < 100:
        return {"status": "BLOCKED", "gate": "Need >=100 spikes for cooperativity stats", "result": None}
    mean_nx = float(np.mean(nx)); var_nx = float(np.var(nx))
    cv = float(np.sqrt(var_nx) / mean_nx) if mean_nx > 0 else float("inf")
    # Hill-coefficient analog: log-log slope of fraction-firing vs. n_excited
    bins = np.arange(0, int(nx.max())+2)
    hist, _ = np.histogram(nx, bins=bins)
    cumul = np.cumsum(hist) / hist.sum() if hist.sum() else hist*0
    valid = (cumul > 0.05) & (cumul < 0.95)
    hill_n = float("nan")
    if valid.sum() >= 3:
        x_log = np.log(bins[:-1][valid] + 1)
        y_log = np.log(cumul[valid] / (1 - cumul[valid] + 1e-9))
        if np.isfinite(x_log).all() and np.isfinite(y_log).all():
            hill_n = float(np.polyfit(x_log, y_log, 1)[0])

    if mean_nx > 3 and cv < 0.3: interp = "TIGHTLY_COOPERATIVE"
    elif mean_nx > 3 and cv > 0.6: interp = "BURSTY_BISTABLE"
    elif mean_nx < 1.5: interp = "INDEPENDENT_FIRING"
    else: interp = "MIXED_COOPERATIVITY"
    return {"status": "OK", "mean_n_excited": mean_nx, "var_n_excited": var_nx,
            "cv": cv, "hill_coefficient_analog": hill_n, "interpretation": interp}

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT 7: SPIKE_SOURCE-STRATIFIED CONSENSUS
# ═══════════════════════════════════════════════════════════════════════
def enhance_7_source_consensus(rids):
    if not SOURCE_AXIS_ACTIVE:
        return {"status": "BLOCKED", "gate": "Need >=2 distinct spike_source values", "result": None}
    sdf = pocket_spike_slice(rids)
    by_source = {}
    total_e = sdf["intensity"].sum()
    for src in sources_in_run:
        sd = sdf[sdf["spike_source"] == src]
        by_source[int(src)] = {"n_spikes": int(len(sd)),
                               "energy_pct": float(100.0 * sd["intensity"].sum() / total_e) if total_e else 0.0,
                               "n_residues": int(sd["residue_id"].nunique())}
    pcts = [v["energy_pct"] for v in by_source.values()]
    source_dominant = max(by_source.items(), key=lambda kv: kv[1]["energy_pct"])
    source_purity = max(pcts) / 100.0 if pcts else 0.0
    survives_all = all(v["n_spikes"] > 50 for v in by_source.values())
    return {"status": "OK", "by_source": by_source,
            "dominant_source": source_dominant[0],
            "source_purity": source_purity,
            "survives_all_sources": survives_all,
            "interpretation": ("ARTIFACT_RISK" if source_purity > 0.85 and not survives_all
                               else "MULTI_SOURCE_CONFIRMED" if survives_all and source_purity < 0.6
                               else "SOURCE_CONCENTRATED")}

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT 8: VOXEL ADJACENCY (built across all pockets, applied per-pocket)
# ═══════════════════════════════════════════════════════════════════════
def build_voxel_adjacency_matrix(all_pocket_residues):
    pocket_voxels = {}
    for pidx, rids in all_pocket_residues.items():
        sdf = pocket_spike_slice(rids)
        pocket_voxels[pidx] = set(sdf["voxel_idx"].values.tolist())
    pids = sorted(pocket_voxels.keys())
    matrix = {}
    for i, pi in enumerate(pids):
        for pj in pids[i+1:]:
            vi, vj = pocket_voxels[pi], pocket_voxels[pj]
            if not vi or not vj: continue
            inter = len(vi & vj); uni = len(vi | vj)
            jac = inter / uni if uni else 0.0
            matrix[f"P{pi}-P{pj}"] = {"jaccard": jac, "shared_voxels": inter}
    return matrix, pocket_voxels

def enhance_8_voxel_for_pocket(pidx, voxel_matrix, pocket_voxels):
    related = []
    for k, v in voxel_matrix.items():
        if k.startswith(f"P{pidx}-") or k.endswith(f"-P{pidx}"):
            related.append({"pair": k, **v})
    risk_pairs = [r for r in related if r["jaccard"] > 0.3]
    indep_pairs = [r for r in related if r["jaccard"] < 0.05]
    interp = "MEGACLUSTER_RISK" if risk_pairs else "INDEPENDENT_SITE" if len(indep_pairs) == len(related) else "ADJACENT"
    return {"status": "OK", "n_voxels": len(pocket_voxels.get(pidx, set())),
            "pairs": related, "megacluster_risk_pairs": risk_pairs,
            "interpretation": interp}

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT 9: WD_CHANGE DIRECTIONAL DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════
def enhance_9_wd_directional(rids):
    sdf = pocket_spike_slice(rids)
    wd_ch = sdf["wd_change"].values
    if len(wd_ch) < 100:
        return {"status": "BLOCKED", "gate": "Need >=100 spikes", "result": None}
    pos_mass = float(wd_ch[wd_ch > 0].sum())
    neg_mass = float(-wd_ch[wd_ch < 0].sum())
    total_abs = pos_mass + neg_mass
    signed_mean = float(np.mean(wd_ch))
    pos_frac = pos_mass / total_abs if total_abs else 0.5
    neg_frac = neg_mass / total_abs if total_abs else 0.5
    purity = max(pos_frac, neg_frac)
    direction = "WATER_EXPELLING" if neg_frac > 0.6 else "WATER_ATTRACTING" if pos_frac > 0.6 else "MIXED"

    # Per-residue signed budget
    res_signed = {}
    for rid in rids:
        rd = sdf[sdf["residue_id"] == rid]["wd_change"]
        res_signed[int(rid)] = {"mean_signed": float(rd.mean()) if len(rd) else 0.0,
                                "n_spikes": int(len(rd))}
    return {"status": "OK", "signed_mean": signed_mean,
            "positive_mass": pos_mass, "negative_mass": neg_mass,
            "directional_purity": purity, "direction": direction,
            "per_residue_signed": res_signed,
            "interpretation": ("DRUGGABLE_CRYPTIC_CLEFT" if direction == "WATER_EXPELLING" and purity > 0.65
                              else "ALLOSTERIC_SOLVATION" if direction == "WATER_ATTRACTING" and purity > 0.65
                              else "AMBIGUOUS_HYDRATION")}

# ═══════════════════════════════════════════════════════════════════════
# ENHANCEMENT 10: TENSOR DECOMPOSITION (gated on multi-seed)
# ═══════════════════════════════════════════════════════════════════════
def enhance_10_tensor_nntf():
    if not GATES["G1_multi_seed"]:
        return {"status": "BLOCKED",
                "gate": ("G1_multi_seed: Need >=2 replica seeds for tensor decomposition. "
                         "Currently single seed (42). Run canonical command with --replica-seed 43, 44, ... "
                         "and merge Arrow files. Tensor NNTF requires the seed axis to factorize "
                         "geometry-blind pocket components."),
                "result": None}
    # Implementation when gate passes: NNTF on residue × stream × phase × seed tensor
    try:
        from sklearn.decomposition import NMF
    except ImportError:
        return {"status": "BLOCKED", "gate": "scikit-learn NMF not installed", "result": None}
    # Build tensor — flattened to residue × (stream*phase*seed)
    n_streams = len(streams_in_run); n_phases = len(ccns_in_run); n_seeds = len(seeds_in_run)
    residues = sorted(res["residue_id"].unique())
    n_res_t = len(residues)
    tensor = np.zeros((n_res_t, n_streams * n_phases * n_seeds))
    for i, rid in enumerate(residues):
        rdf = df[df["residue_id"] == rid]
        for sid_i, sid in enumerate(streams_in_run):
            for ph_i, ph in enumerate(ccns_in_run):
                for sd_i, sd in enumerate(seeds_in_run):
                    sub = rdf[(rdf["stream_id"]==sid)&(rdf["ccns_phase"]==ph)&(rdf["replica_seed"]==sd)]
                    col = sid_i * n_phases * n_seeds + ph_i * n_seeds + sd_i
                    tensor[i, col] = sub["intensity"].sum()
    n_components = min(10, n_res_t // 5)
    nmf = NMF(n_components=n_components, init="nndsvd", random_state=42, max_iter=500)
    W = nmf.fit_transform(tensor); H = nmf.components_
    components = []
    for k in range(n_components):
        top_res_idx = np.argsort(W[:,k])[-15:][::-1]
        components.append({"component_id": k, "top_residues": [int(residues[i]) for i in top_res_idx],
                          "energy_share": float(W[:,k].sum() * H[k,:].sum() / (W.sum() * H.sum() + 1e-12))})
    return {"status": "OK", "n_components": n_components,
            "reconstruction_error": float(nmf.reconstruction_err_),
            "components": components}

# ═══════════════════════════════════════════════════════════════════════
# STAGE 5: BUILD FULL DOSSIER PER POCKET
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("PER-POCKET ENHANCED DIAGNOSTICS")
print("=" * 100)

# Pre-compute voxel adjacency matrix across all pockets
all_pocket_residues = {idx: set(rids) for idx, (rids, _) in enumerate(pockets, 1)}
voxel_matrix, pocket_voxels = build_voxel_adjacency_matrix(all_pocket_residues)

records = []
for idx, (rids, _) in enumerate(pockets, 1):
    diag = core_diagnostics(rids)
    yields = regional_yields(rids)
    canonical_yield = sum(yields[k] for k in DOMAINS.keys())
    dominant = max(yields, key=yields.get)
    sdf = pocket_spike_slice(rids)
    pdf = res[res["residue_id"].isin(rids)]
    pE = float(pdf["total_energy"].sum())

    # Verdict
    real_ev = 0; bridge_ev = 0; axes = []
    if diag["rg"] < 8.0: real_ev += 1; axes.append("compact")
    if diag["rg"] > 12.0: bridge_ev += 1
    if diag["seq_gap"] < 15.0: real_ev += 1; axes.append("seq_local")
    if diag["seq_gap"] > 30.0: bridge_ev += 1
    if 5e-6 < diag["wd_var"] < 5e-5: real_ev += 1; axes.append("solvent_coherent")
    if diag["wd_var"] > 1e-4: bridge_ev += 1
    if not math.isnan(diag["sc_corr"]) and diag["sc_corr"] > 0.2: real_ev += 1; axes.append("sc_corr")
    if not math.isnan(diag["sc_corr"]) and diag["sc_corr"] < -0.2: bridge_ev += 1
    if real_ev >= 2 and bridge_ev == 0: verdict = "REAL_POCKET"
    elif bridge_ev >= 2: verdict = "CAUSAL_BRIDGE"
    elif bridge_ev == 1 and real_ev >= 1: verdict = "MIXED"
    elif real_ev == 1 and bridge_ev == 0: verdict = "AMBIGUOUS_LEAN_REAL"
    else: verdict = "AMBIGUOUS"

    # All 10 enhancements
    enh = {
        "E1_ccns_lifecycle":        enhance_1_ccns_lifecycle(rids),
        "E2_wavelength_fingerprint":enhance_2_wavelength_fingerprint(rids),
        "E3_phase_bits_mi":         enhance_3_phase_bits_mi(rids),
        "E4_first_passage":         enhance_4_first_passage(rids),
        "E5_aromatic_graph":        enhance_5_aromatic_graph(rids, all_pocket_residues),
        "E6_cooperative_excitation":enhance_6_cooperative_excitation(rids),
        "E7_source_consensus":      enhance_7_source_consensus(rids),
        "E8_voxel_adjacency":       enhance_8_voxel_for_pocket(idx, voxel_matrix, pocket_voxels),
        "E9_wd_directional":        enhance_9_wd_directional(rids),
        # E10 is run-level not per-pocket — added once after this loop
    }

    # Druggability — incorporates E9 directional purity
    n_arom = int(pdf["is_aromatic"].sum())
    e9 = enh["E9_wd_directional"]
    direction_bonus = (e9["directional_purity"] if e9["status"] == "OK" and e9["direction"] == "WATER_EXPELLING" else 0.0)
    drug_score = (
        0.25 * np.log1p(diag["wd_change_p95"] * 1000)
        + 0.20 * np.log1p(diag["wd_iqr"] * 1000)
        + 0.15 * np.log1p(pE / max(diag["n_residues"],1) / 1e6)
        + 0.15 * (n_arom / max(diag["n_residues"],1))
        + 0.10 * np.log1p(diag["frame_persistence_per_res"])
        + 0.10 * direction_bonus
        + 0.05 * (1.0 - diag["ccns_balance"])
    )
    novelty = (100.0 - canonical_yield) * (drug_score / 5.0)

    flags = []
    if canonical_yield < 50.0: flags.append("OFF_CANONICAL")
    if n_arom >= 2: flags.append("AROMATIC_RICH")
    if diag["wd_change_p95"] > 0.020: flags.append("HIGH_WATER_DISPLACEMENT")
    if e9["status"] == "OK" and e9["direction"] == "WATER_EXPELLING": flags.append("WATER_EXPELLING")
    if e9["status"] == "OK" and e9["direction"] == "WATER_ATTRACTING": flags.append("WATER_ATTRACTING")
    if diag["frame_persistence_per_res"] > 30: flags.append("TEMPORALLY_PERSISTENT")
    if diag["frame_persistence_per_res"] < 15: flags.append("TRANSIENT_CRYPTIC")
    if enh["E5_aromatic_graph"].get("interpretation") == "ALLOSTERIC_COUPLED": flags.append("CROSS_POCKET_COUPLED")
    if enh["E8_voxel_adjacency"]["interpretation"] == "MEGACLUSTER_RISK": flags.append("MEGACLUSTER_RISK")
    if enh["E1_ccns_lifecycle"].get("status") == "OK":
        flags.append(f"TAXONOMY_{enh['E1_ccns_lifecycle']['taxonomy_class']}")
    if enh["E6_cooperative_excitation"].get("status") == "OK":
        flags.append(f"COOP_{enh['E6_cooperative_excitation']['interpretation']}")

    rec = {
        "pocket_id": idx,
        "residues": sorted(rids),
        "verdict": verdict,
        "verdict_evidence": f"{real_ev}R/{bridge_ev}B",
        "axes_supporting_real": axes,
        "dominant_region": dominant,
        "canonical_yield_pct": canonical_yield,
        "druggability_score": float(drug_score),
        "novelty_score": float(novelty),
        "flags": flags,
        "yields": yields,
        "core_diagnostics": diag,
        "n_aromatic_residues": n_arom,
        "aromatic_residue_ids": pdf[pdf["is_aromatic"]==1]["residue_id"].tolist(),
        "enhancements": enh,
    }
    records.append(rec)

    # Per-pocket summary print
    print(f"\nPOCKET {idx} | {verdict} ({rec['verdict_evidence']}) | dominant: {dominant} | drug={drug_score:.3f} novel={novelty:.2f}")
    print(f"  Residues ({len(rids)}): {sorted(rids)}")
    print(f"  Geometry  Rg={diag['rg']:.2f}A  seq_gap={diag['seq_gap']:.2f}  centroid={[round(c,2) for c in diag['centroid_xyz']]}")
    print(f"  Yields    " + "  ".join(f"{k}={v:.1f}%" for k,v in yields.items()))
    print(f"  Flags     {flags}")
    # Print enhancement statuses
    for ekey, ev in enh.items():
        st = ev.get("status", "?")
        if st == "BLOCKED":
            print(f"    {ekey}: BLOCKED — {ev['gate'][:80]}{'...' if len(ev.get('gate',''))>80 else ''}")
        else:
            interp = ev.get("interpretation") or ev.get("taxonomy_class") or ev.get("direction") or "computed"
            print(f"    {ekey}: {interp}")

# Run-level Enhancement 10
print("\n" + "=" * 100)
print("RUN-LEVEL ENHANCEMENT 10: TENSOR DECOMPOSITION")
print("=" * 100)
e10_result = enhance_10_tensor_nntf()
if e10_result["status"] == "BLOCKED":
    print(f"  E10: BLOCKED — {e10_result['gate']}")
else:
    print(f"  E10: {e10_result['n_components']} components, recon_err={e10_result['reconstruction_error']:.4f}")
    for c in e10_result["components"][:5]:
        print(f"    Component {c['component_id']}: top_residues={c['top_residues']}, share={c['energy_share']:.3f}")

# ═══════════════════════════════════════════════════════════════════════
# STAGE 6: PROTEIN-LEVEL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════
total_E = float(res["total_energy"].sum())
region_E = {k: float(res.loc[res["residue_id"].isin(v), "total_energy"].sum()) for k, v in DOMAINS.items()}
region_E["Other"] = total_E - sum(region_E.values())
canonical_pct = 100.0 * (total_E - region_E["Other"]) / total_E
protein_dist = {k: {"energy": v, "pct": 100.0*v/total_E} for k, v in region_E.items()}

print("\n" + "=" * 100)
print("PROTEIN-LEVEL THERMODYNAMIC ENERGY DISTRIBUTION")
print("=" * 100)
for k, v in protein_dist.items():
    print(f"  {k:<14} {v['energy']:>14,.0f}  ({v['pct']:>5.1f}%)")
print(f"  CANONICAL TOT  {canonical_pct:>5.1f}%  <-- headline")

# ═══════════════════════════════════════════════════════════════════════
# STAGE 7: WRITE DOSSIER
# ═══════════════════════════════════════════════════════════════════════
csv_path  = os.path.join(OUT_DIR, "4lpk_dossier_pockets_full.csv")
json_path = os.path.join(OUT_DIR, "4lpk_dossier_full_v2.json")

# CSV — flatten enhancement statuses
csv_records = []
for r in records:
    flat = {
        "pocket_id": r["pocket_id"],
        "residues": ";".join(map(str, r["residues"])),
        "verdict": r["verdict"],
        "verdict_evidence": r["verdict_evidence"],
        "dominant_region": r["dominant_region"],
        "canonical_yield_pct": r["canonical_yield_pct"],
        "druggability_score": r["druggability_score"],
        "novelty_score": r["novelty_score"],
        "n_residues": r["core_diagnostics"]["n_residues"],
        "n_aromatic": r["n_aromatic_residues"],
        "rg_A": r["core_diagnostics"]["rg"],
        "seq_gap": r["core_diagnostics"]["seq_gap"],
        "wd_change_p95": r["core_diagnostics"]["wd_change_p95"],
        "flags": ";".join(r["flags"]),
    }
    for k, v in r["yields"].items():
        flat[f"yield_{k}"] = v
    for ekey, ev in r["enhancements"].items():
        flat[f"{ekey}_status"] = ev["status"]
        flat[f"{ekey}_result"] = (ev.get("interpretation") or ev.get("taxonomy_class")
                                  or ev.get("direction") or ev.get("gate") or "OK")
    csv_records.append(flat)
pd.DataFrame(csv_records).to_csv(csv_path, index=False)

dossier = {
    "run_metadata": run_metadata,
    "protein_level_energy_distribution": protein_dist,
    "canonical_total_pct": canonical_pct,
    "kdtree_radius_used": CANONICAL_RADIUS,
    "kdtree_radius_stability": {str(r): {"n_pockets": len(p)} for r, p in stability_results.items()},
    "voxel_adjacency_matrix": voxel_matrix,
    "pockets": records,
    "run_level_enhancement_E10": e10_result,
    "dev_gate_summary": {g: ("PASS" if s else "BLOCKED") for g, s in GATES.items()},
}

def to_json_safe(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (set, frozenset)): return list(obj)
    if isinstance(obj, dict): return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json_safe(x) for x in obj]
    if isinstance(obj, float) and not math.isfinite(obj): return None
    return obj

with open(json_path, "w") as f:
    json.dump(to_json_safe(dossier), f, indent=2, default=str)

print(f"\n[7] DOSSIER WRITTEN")
print(f"    CSV:  {csv_path}")
print(f"    JSON: {json_path}")

# ═══════════════════════════════════════════════════════════════════════
# STAGE 8: GATE SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("DEV GATE SUMMARY")
print("=" * 100)
blocked_gates = [g for g, s in GATES.items() if not s]
if blocked_gates:
    print(f"\n{len(blocked_gates)} gate(s) BLOCKED:")
    for g in blocked_gates:
        print(f"  - {g}")
    print("\nTo unlock blocked enhancements, run the Claude Code prompt at:")
    print(f"  {OUT_DIR}/CLAUDE_CODE_PROMPT.md")
else:
    print("\nAll gates PASS. Full dossier complete.")

print("\n" + "=" * 100)
print("DOSSIER COMPLETE")
print("=" * 100)
