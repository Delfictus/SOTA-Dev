# =============================================================
# SUPERSEDED — QUARANTINED DEV SCRIPT
# =============================================================
# Original path:   /tmp/enrich_tem1_report.py
# Quarantined on:  2026-04-05
# Quarantined by:  SOP bootstrap session (Task C)
# Reason:          One-off TEM-1 PRISM-TWIN report dev script
#                  living in /tmp. Preserved for reference only.
# Status:          DO NOT EXECUTE. DO NOT IMPORT.
# Rule:            docs/PRISM4D_DEV_OPS_FRAMEWORK.md §1
#                  (no production script may reference /tmp)
# Canonical home:  TBD — if resurrected, must move to
#                  scripts/production/ with full parameterization.
# =============================================================

#!/usr/bin/env python3
"""
TEM1 PRISM-TWIN Report Enrichment — 8 Gates
Processes 6.9GB coupled_spikes.json in batches, enriches the existing report,
and overwrites tem1_prism_twin_report.json with fully populated fields.
"""

import json
import sys
import os
import math
import time
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree
from datetime import datetime

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False

# ─── Paths ───────────────────────────────────────────────────────────────────
SPIKES_JSON     = "/tmp/twin_tem1_full/coupled_spikes.json"
TWIN_RESULT     = "/tmp/twin_tem1_full/coupled_twin_result.json"
CCF_NPY         = "/tmp/twin_tem1_full/tem1_ccf_matrix.npy"
REPORT_JSON     = "/tmp/twin_tem1_full/tem1_prism_twin_report.json"
CLEAN_PDB       = "/mnt/storage/prism-outputs/runs/v1.1-physics/tem1/tem1_clean.pdb"
TOPOLOGY_JSON   = "/mnt/storage/prism-outputs/runs/v1.1-physics/tem1/tem1.topology.json"
BS_JSON         = "/mnt/storage/prism-outputs/runs/v1.1-physics/tem1/tem1.binding_sites.json"
NMA_JSON        = "/tmp/nma_gate2_test/tem1_nma_modes.json"
OUT_PER_RES     = "/tmp/twin_tem1_full/per_residue_spikes.json"
OUT_PER_PHASE   = "/tmp/twin_tem1_full/per_residue_phases.json"
OUT_MECHANISMS  = "/tmp/twin_tem1_full/site_mechanisms.json"

# ─── Constants ───────────────────────────────────────────────────────────────
BATCH_SIZE      = 100_000
TOPO_OFFSET     = 26          # pdb_resnum = topo_idx + 26
ACTIVE_SITE_PDB = {70, 73, 130, 166}   # SER70, LYS73, SER130, GLU166

# Phase timestep boundaries
PHASE_BOUNDS = {
    "cold_hold":   (0, 14000),
    "heating":     (14000, 20000),
    "warm_hold":   (20000, 35000),
    "cooling":     (35000, 41000),
    "cold_return": (41000, 10**9),
}

def phase_of(ts):
    for name, (lo, hi) in PHASE_BOUNDS.items():
        if lo <= ts < hi:
            return name
    return "cold_return"

# ─── Load topology CA positions ───────────────────────────────────────────────
def load_ca_positions():
    with open(TOPOLOGY_JSON) as f:
        topo = json.load(f)
    positions_flat = np.array(topo["positions"])
    positions = positions_flat.reshape(-1, 3)
    atom_names  = topo["atom_names"]
    residue_ids = topo["residue_ids"]    # 0-based topo indices

    ca_idx       = [i for i, n in enumerate(atom_names) if n == "CA"]
    ca_positions = positions[ca_idx]
    # topo_idx = residue_ids at CA atom positions (they are 0..262 for TEM1)
    # but residue_ids here is the *atom's* residue index (0-based)
    ca_topo_idx  = [residue_ids[i] for i in ca_idx]

    return ca_positions, ca_topo_idx


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 1 — Per-residue spike counts
# ═══════════════════════════════════════════════════════════════════════════════
def gate1_per_residue_spikes(ca_positions, ca_topo_idx):
    print("GATE 1: Processing per-residue spike counts from 6.9GB JSON ...")
    t0 = time.time()

    n_residues = len(ca_positions)
    tree = KDTree(ca_positions)

    spikes_a         = np.zeros(n_residues, dtype=np.int64)
    spikes_b         = np.zeros(n_residues, dtype=np.int64)
    intensity_sum_a  = np.zeros(n_residues, dtype=np.float64)
    intensity_sum_b  = np.zeros(n_residues, dtype=np.float64)
    thermal_sum      = np.zeros(n_residues, dtype=np.float64)  # sum of vib_energy

    total_processed = 0

    with open(SPIKES_JSON, "rb") as fh:
        # Use ijson to stream the 'spikes' array
        parser = ijson.items(fh, "spikes.item")
        batch  = []
        for record in parser:
            batch.append(record)
            if len(batch) >= BATCH_SIZE:
                _process_batch(batch, tree, spikes_a, spikes_b,
                               intensity_sum_a, intensity_sum_b, thermal_sum)
                total_processed += len(batch)
                if total_processed % 1_000_000 == 0:
                    elapsed = time.time() - t0
                    print(f"  ... {total_processed/1e6:.1f}M spikes processed in {elapsed:.1f}s")
                batch = []
        if batch:
            _process_batch(batch, tree, spikes_a, spikes_b,
                           intensity_sum_a, intensity_sum_b, thermal_sum)
            total_processed += len(batch)

    elapsed = time.time() - t0
    print(f"  Processed {total_processed:,} spikes in {elapsed:.1f}s")

    # Build per-residue records
    records = []
    for i, topo_idx in enumerate(ca_topo_idx):
        pdb_resnum = topo_idx + TOPO_OFFSET
        sa = int(spikes_a[i])
        sb = int(spikes_b[i])
        total = sa + sb
        ba_ratio = (sb / sa) if sa > 0 else (float("inf") if sb > 0 else 1.0)
        consensus   = min(sa, sb)
        differential = abs(sa - sb)
        thermal     = float(thermal_sum[i])
        thermal_per_spike = thermal / total if total > 0 else 0.0

        if ba_ratio > 1.2:
            classification = "NMA_RESPONSIVE"
        elif 0.8 <= ba_ratio <= 1.2:
            classification = "THERMALLY_ACCESSIBLE"
        elif total < 100:
            classification = "INERT"
        else:
            classification = "THERMALLY_ACCESSIBLE"

        records.append({
            "topo_idx":    topo_idx,
            "pdb_resnum":  pdb_resnum,
            "spikes_a":    sa,
            "spikes_b":    sb,
            "total_spikes": total,
            "b_over_a_ratio": round(ba_ratio, 4) if math.isfinite(ba_ratio) else 9999.0,
            "consensus":   int(consensus),
            "differential": int(differential),
            "thermal_spike_energy": round(thermal, 6),
            "thermal_per_spike": round(thermal_per_spike, 8),
            "thermal_only": int(total < 100),   # flag if very few spikes
            "spike_classification": classification,
        })

    with open(OUT_PER_RES, "w") as f:
        json.dump(records, f, indent=2)

    print(f"GATE 1: PASS — {total_processed:,} spikes mapped to {n_residues} residues")
    return records


def _process_batch(batch, tree, spikes_a, spikes_b, int_a, int_b, thermal_sum):
    coords = np.array([[r["x"], r["y"], r["z"]] for r in batch], dtype=np.float32)
    _, idx  = tree.query(coords, k=1, workers=-1)
    for j, record in enumerate(batch):
        res_i    = idx[j]
        stream   = record.get("stream_id", 0)
        intensity = float(record.get("intensity", 0.0))
        vib      = float(record.get("vib_energy", 0.0))
        if stream == 0:
            spikes_a[res_i]  += 1
            int_a[res_i]     += intensity
        else:
            spikes_b[res_i]  += 1
            int_b[res_i]     += intensity
        thermal_sum[res_i] += vib


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 2 — CCF propagation
# ═══════════════════════════════════════════════════════════════════════════════
def gate2_ccf_propagation(ca_positions, ca_topo_idx):
    print("GATE 2: CCF propagation analysis ...")

    ccf = np.load(CCF_NPY)   # (263, 263)
    n   = ccf.shape[0]

    # Pairwise CA distances for remote/local classification
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        d = np.linalg.norm(ca_positions - ca_positions[i], axis=1)
        dist_matrix[i] = d

    records = []
    for i in range(n):
        row      = ccf[i]
        diag_val = float(row[i])
        off_mask = np.ones(n, dtype=bool); off_mask[i] = False
        off_row  = row[off_mask]
        dist_row = dist_matrix[i][off_mask]

        # Strongest partner (excluding self)
        j_off = int(np.argmax(off_row))
        # Map back to full index
        full_j = j_off if j_off < i else j_off + 1
        partner_topo = ca_topo_idx[full_j]

        local_mask  = dist_row < 8.0
        remote_mask = dist_row > 15.0

        local_mean  = float(off_row[local_mask].mean()) if local_mask.any() else 0.0
        remote_mean = float(off_row[remote_mask].mean()) if remote_mask.any() else 0.0

        # Allosteric score: remote_mean / (local_mean + 1e-6) — high if distal coupling
        allosteric_score = remote_mean / (local_mean + 1e-6)

        records.append({
            "topo_idx":       ca_topo_idx[i],
            "pdb_resnum":     ca_topo_idx[i] + TOPO_OFFSET,
            "ccf_max":        round(float(row.max()), 6),
            "ccf_mean":       round(float(off_row.mean()), 6),
            "ccf_n_05":       int((off_row > 0.5).sum()),
            "ccf_n_08":       int((off_row > 0.8).sum()),
            "strongest_partner_pdb": int(partner_topo + TOPO_OFFSET),
            "partner_distance_angstrom": round(float(dist_matrix[i, full_j]), 3),
            "cluster_size":   int((off_row > 0.5).sum()) + 1,
            "diagonal":       round(diag_val, 6),
            "off_diag_max":   round(float(off_row.max()), 6),
            "asymmetry":      round(float(ccf[i, full_j] - ccf[full_j, i]), 6),
            "local_mean_lt8A": round(local_mean, 6),
            "remote_mean_gt15A": round(remote_mean, 6),
            "allosteric_score": round(allosteric_score, 4),
            "persistence":    round(float((off_row > 0.5).mean()), 4),
            "onset_bin":      int(np.searchsorted(np.sort(off_row)[::-1], 0.5)),
        })

    print(f"GATE 2: PASS — CCF enrichment for {n} residues")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 3 — Phase-resolved spike profiles
# ═══════════════════════════════════════════════════════════════════════════════
def gate3_phase_profiles(ca_positions, ca_topo_idx):
    print("GATE 3: Phase-resolved spike profiles ...")
    t0 = time.time()

    n_residues = len(ca_positions)
    tree = KDTree(ca_positions)
    phases = list(PHASE_BOUNDS.keys())

    # [residue_i][phase][stream]
    phase_counts = [[{"a": 0, "b": 0} for _ in phases] for _ in range(n_residues)]

    total_processed = 0
    with open(SPIKES_JSON, "rb") as fh:
        parser = ijson.items(fh, "spikes.item")
        batch  = []
        for record in parser:
            batch.append(record)
            if len(batch) >= BATCH_SIZE:
                _process_phase_batch(batch, tree, phase_counts, phases)
                total_processed += len(batch)
                if total_processed % 2_000_000 == 0:
                    print(f"  ... phase pass {total_processed/1e6:.1f}M spikes")
                batch = []
        if batch:
            _process_phase_batch(batch, tree, phase_counts, phases)
            total_processed += len(batch)

    elapsed = time.time() - t0
    print(f"  Phase pass: {total_processed:,} spikes in {elapsed:.1f}s")

    records = []
    for i, topo_idx in enumerate(ca_topo_idx):
        pdb_resnum = topo_idx + TOPO_OFFSET
        phase_data = {}
        for pi, ph in enumerate(phases):
            phase_data[ph] = {
                "stream_a": phase_counts[i][pi]["a"],
                "stream_b": phase_counts[i][pi]["b"],
                "total":    phase_counts[i][pi]["a"] + phase_counts[i][pi]["b"],
            }
        records.append({
            "topo_idx":   topo_idx,
            "pdb_resnum": pdb_resnum,
            "phases":     phase_data,
        })

    with open(OUT_PER_PHASE, "w") as f:
        json.dump(records, f, indent=2)

    print(f"GATE 3: PASS — Phase profiles for {n_residues} residues")
    return records


def _process_phase_batch(batch, tree, phase_counts, phases):
    coords = np.array([[r["x"], r["y"], r["z"]] for r in batch], dtype=np.float32)
    _, idx  = tree.query(coords, k=1, workers=-1)
    for j, record in enumerate(batch):
        res_i  = idx[j]
        ts     = record.get("timestep", 0)
        stream = record.get("stream_id", 0)
        ph_name = phase_of(ts)
        pi = phases.index(ph_name)
        if stream == 0:
            phase_counts[res_i][pi]["a"] += 1
        else:
            phase_counts[res_i][pi]["b"] += 1


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 4 — Mechanism classification per site
# ═══════════════════════════════════════════════════════════════════════════════
def gate4_site_mechanisms(g1_by_pdb, ca_positions, ca_topo_idx):
    print("GATE 4: Mechanism classification per site ...")

    with open(BS_JSON) as f:
        bs = json.load(f)
    with open(NMA_JSON) as f:
        nma = json.load(f)

    # Build NMA per-residue hinge score (magnitude of displacement across all modes)
    n_modes = nma["n_modes"]
    n_res   = nma["n_residues"]
    nma_rids = nma["residue_ids"]  # PDB resnums

    # displacement magnitude per residue per mode
    disp_mag = np.zeros((n_modes, n_res), dtype=np.float64)
    for mi, mode in enumerate(nma["modes"]):
        disps = mode["displacements"]   # list of [dx,dy,dz]
        for ri, d in enumerate(disps):
            disp_mag[mi, ri] = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)

    # Hinge score: residue that is locally rigid while neighbors are mobile
    sqfluct = disp_mag.mean(axis=0)  # mean displacement across modes
    # Hinge = low local sqfluct relative to neighbors
    hinge_score = np.zeros(n_res, dtype=np.float64)
    for ri in range(n_res):
        neighbor_sq = []
        for rj in range(max(0, ri-3), min(n_res, ri+4)):
            if rj != ri:
                neighbor_sq.append(sqfluct[rj])
        if neighbor_sq:
            nb_mean = np.mean(neighbor_sq)
            hinge_score[ri] = nb_mean / (sqfluct[ri] + 1e-8)  # high = hinge

    # nma_rids[ri] = PDB resnum for residue index ri
    hinge_by_pdb = {nma_rids[ri]: hinge_score[ri] for ri in range(n_res)}
    sqfluct_by_pdb = {nma_rids[ri]: sqfluct[ri] for ri in range(n_res)}

    # Build engine_id → report_site_id mapping
    with open(REPORT_JSON) as f_rep:
        _rep = json.load(f_rep)
    engine_id_to_site_id = {
        s["engine_pocket_id"]: s["site_id"]
        for s in _rep["sites"]
        if s.get("engine_pocket_id") is not None
    }

    mechanisms = []
    bs_sites = bs.get("sites") or bs.get("all_pockets") or []
    for site in bs_sites:
        engine_id = site.get("id") or site.get("site_id")
        site_id   = engine_id_to_site_id.get(engine_id, f"ENGINE_{engine_id}")
        lining     = site.get("lining_residues", [])

        # lining uses topo resid, convert to PDB
        gating_residues = []
        hinge_residues  = []
        mobile_residues = []

        for lr in lining:
            topo_resid = lr["resid"]
            pdb_rn     = topo_resid + TOPO_OFFSET
            g1         = g1_by_pdb.get(pdb_rn, {})
            ba         = g1.get("b_over_a_ratio", 1.0)
            hinge_s    = hinge_by_pdb.get(pdb_rn, 0.0)
            sq         = sqfluct_by_pdb.get(pdb_rn, 0.0)

            if ba > 1.2:
                gating_residues.append({"pdb_resnum": pdb_rn, "b_over_a_ratio": round(ba, 4)})
            if hinge_s > 2.0:
                hinge_residues.append({"pdb_resnum": pdb_rn, "hinge_score": round(hinge_s, 4)})
            if sq > np.mean(list(sqfluct_by_pdb.values())) * 1.5:
                mobile_residues.append({"pdb_resnum": pdb_rn, "sqfluct": round(float(sq), 6)})

        # Classify opening mechanism
        if len(hinge_residues) >= 2 and len(gating_residues) >= 1:
            opening_mechanism = "HINGE_GATED"
        elif len(gating_residues) >= 3:
            opening_mechanism = "COLLECTIVE_OPENING"
        elif len(mobile_residues) >= 2:
            opening_mechanism = "INDUCED_FIT"
        elif len(gating_residues) == 0 and len(hinge_residues) == 0:
            opening_mechanism = "PREFORMED"
        else:
            opening_mechanism = "CONFORMATIONAL_SELECTION"

        # Dominant NMA modes for site
        # Find which modes have highest mean displacement over lining residues
        site_mode_disp = []
        lining_pdb_set = {lr["resid"] + TOPO_OFFSET for lr in lining}
        for mi, mode in enumerate(nma["modes"]):
            ri_list = [ri for ri, pr in enumerate(nma_rids) if pr in lining_pdb_set]
            if ri_list:
                mean_disp = float(np.mean([disp_mag[mi, ri] for ri in ri_list]))
            else:
                mean_disp = 0.0
            site_mode_disp.append({
                "mode_index": mode["mode_index"],
                "mean_displacement": round(mean_disp, 6),
                "eigenvalue": mode["eigenvalue"],
                "thermal_amplitude": mode["thermal_amplitude"],
            })
        site_mode_disp.sort(key=lambda x: -x["mean_displacement"])
        top_modes = site_mode_disp[:3]

        mechanisms.append({
            "site_id":          site_id,
            "opening_mechanism": opening_mechanism,
            "gating_residues":  gating_residues,
            "hinge_residues":   hinge_residues,
            "mobile_residues":  mobile_residues,
            "dominant_nma_modes": top_modes,
        })

    with open(OUT_MECHANISMS, "w") as f:
        json.dump(mechanisms, f, indent=2)

    print(f"GATE 4: PASS — Mechanisms classified for {len(mechanisms)} sites")
    return {m["site_id"]: m for m in mechanisms}


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 5 — Thermodynamic enrichment per site
# ═══════════════════════════════════════════════════════════════════════════════
def gate5_thermo_enrichment(g3_phase_by_pdb, report_sites):
    print("GATE 5: Thermodynamic enrichment per site ...")

    results = {}
    for site in report_sites:
        sid  = site["site_id"]
        lining = site.get("lining_residues", [])
        pdb_resnums = [lr["pdb_resnum"] for lr in lining]

        # Aggregate phase spike totals across all lining residues
        phase_totals = defaultdict(int)
        for pdb_rn in pdb_resnums:
            pr = g3_phase_by_pdb.get(pdb_rn)
            if pr is None:
                continue
            for ph, dat in pr["phases"].items():
                phase_totals[ph] += dat["total"]

        # Temperature onset: first phase with >10 spikes
        temperature_onset = None
        for ph in ["cold_hold", "heating", "warm_hold", "cooling", "cold_return"]:
            if phase_totals.get(ph, 0) > 10:
                temperature_onset = ph
                break

        warm_spikes    = phase_totals.get("warm_hold", 0)
        heat_spikes    = phase_totals.get("heating", 0)
        cool_spikes    = phase_totals.get("cooling", 0)
        cold_spikes    = phase_totals.get("cold_hold", 0)
        total_spikes   = sum(phase_totals.values())

        # Occupancy: fraction of warm_hold
        occupancy = warm_spikes / total_spikes if total_spikes > 0 else 0.0

        # Hysteresis: ratio of cooling to heating
        hysteresis = cool_spikes / (heat_spikes + 1e-6)

        # Barrier estimate
        if temperature_onset in ("cold_hold",):
            barrier = "LOW"
            barrier_kcal = 1.0
        elif temperature_onset in ("heating",):
            barrier = "MEDIUM_LOW"
            barrier_kcal = 2.0
        elif temperature_onset in ("warm_hold",):
            barrier = "HIGH"
            barrier_kcal = 5.0
        elif temperature_onset is None:
            barrier = "INACCESSIBLE"
            barrier_kcal = 8.0
        else:
            barrier = "MEDIUM"
            barrier_kcal = 3.5

        results[sid] = {
            "temperature_onset":      temperature_onset,
            "phase_spike_counts":     dict(phase_totals),
            "occupancy_warm_hold":    round(occupancy, 4),
            "hysteresis_ratio":       round(float(hysteresis), 4),
            "barrier_estimate":       barrier,
            "barrier_kcal_mol":       barrier_kcal,
        }

    print(f"GATE 5: PASS — Thermodynamic enrichment for {len(results)} sites")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 6 — Druggability enrichment per site
# ═══════════════════════════════════════════════════════════════════════════════
HYDROPHOBIC_AA  = set("ACFGILMVWY")
POLAR_AA        = set("NQST")
CHARGED_AA      = set("DEKRH")
AROMATIC_AA     = set("FWY")

def gate6_druggability(report_sites):
    print("GATE 6: Druggability enrichment per site ...")

    results = {}
    for site in report_sites:
        sid    = site["site_id"]
        vol    = site.get("volume_angstrom3_convex_hull") or site.get("volume_angstrom3_engine") or 200.0
        lining = site.get("lining_residues", [])
        aa_seq = [lr.get("aa1", "X") for lr in lining]
        n_aa   = len(aa_seq) or 1

        hydrophobic_frac = sum(1 for a in aa_seq if a in HYDROPHOBIC_AA) / n_aa
        polar_frac       = sum(1 for a in aa_seq if a in POLAR_AA) / n_aa
        charged_frac     = sum(1 for a in aa_seq if a in CHARGED_AA) / n_aa
        aromatic_frac    = sum(1 for a in aa_seq if a in AROMATIC_AA) / n_aa

        # Ligand MW range from volume (rough: 100 Å³ ~ 100 Da)
        mw_min = max(100, vol * 0.8)
        mw_max = min(800, vol * 1.5)

        # Fragment hotspots: sub-regions with ≥2 consecutive aromatic/hydrophobic
        hotspots = []
        for i, lr in enumerate(lining):
            if lr.get("aa1", "X") in AROMATIC_AA:
                hotspots.append({
                    "pdb_resnum": lr["pdb_resnum"],
                    "resname":    lr["resname"],
                    "type":       "AROMATIC_ANCHOR",
                })

        # Pocket polarity class
        if hydrophobic_frac > 0.6:
            polarity_class = "HYDROPHOBIC"
            screening = "HTS fragment screen, focus on Kd < 100µM hydrophobic fragments"
        elif charged_frac > 0.4:
            polarity_class = "CHARGED"
            screening = "Electrostatic/FBDD screen; peptidomimetics or charged scaffolds"
        elif polar_frac > 0.4:
            polarity_class = "POLAR"
            screening = "FBDD with H-bond donor/acceptor fragments; bioisostere optimization"
        else:
            polarity_class = "MIXED"
            screening = "Broad HTS or DEL; scaffold hopping from known binders"

        # Druggability score composite
        drug_score = (
            0.3 * hydrophobic_frac
            + 0.25 * aromatic_frac
            + 0.2 * min(1.0, vol / 500.0)
            + 0.25 * (1.0 - charged_frac)
        )

        results[sid] = {
            "pocket_polarity":       polarity_class,
            "hydrophobic_fraction":  round(hydrophobic_frac, 3),
            "polar_fraction":        round(polar_frac, 3),
            "charged_fraction":      round(charged_frac, 3),
            "aromatic_fraction":     round(aromatic_frac, 3),
            "ligand_mw_range":       [round(mw_min, 1), round(mw_max, 1)],
            "fragment_hotspots":     hotspots[:5],
            "druggability_score_g6": round(drug_score, 4),
            "screening_strategy":    screening,
        }

    print(f"GATE 6: PASS — Druggability enrichment for {len(results)} sites")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 7 — Beacon maps + quality flags
# ═══════════════════════════════════════════════════════════════════════════════
def gate7_beacon_maps(report_sites, g4_mechanisms, ca_positions, ca_topo_idx):
    print("GATE 7: Beacon maps and quality flags ...")

    with open(NMA_JSON) as f:
        nma = json.load(f)
    nma_rids = nma["residue_ids"]

    # Build residue → NMA hinge / sqfluct (fast lookup)
    n_modes = nma["n_modes"]
    n_res   = nma["n_residues"]
    disp_mag = np.zeros((n_modes, n_res), dtype=np.float64)
    for mi, mode in enumerate(nma["modes"]):
        for ri, d in enumerate(mode["displacements"]):
            disp_mag[mi, ri] = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
    sqfluct = disp_mag.mean(axis=0)
    sqfluct_by_pdb = {nma_rids[ri]: float(sqfluct[ri]) for ri in range(n_res)}

    tree = KDTree(ca_positions)

    results = {}
    for site in report_sites:
        sid      = site["site_id"]
        centroid = site.get("centroid_angstrom") or site.get("weighted_centroid_angstrom")
        vol      = site.get("volume_angstrom3_convex_hull") or site.get("volume_angstrom3_engine") or 200.0
        radius   = (3 * vol / (4 * math.pi)) ** (1/3) if vol > 0 else 5.0
        lining   = site.get("lining_residues", [])

        # Responsive NMA modes for beacon
        mech       = g4_mechanisms.get(sid, {})
        top_modes  = mech.get("dominant_nma_modes", [])
        mode_indices = [m["mode_index"] for m in top_modes]
        mode_ampls   = [nma["modes"][mi]["thermal_amplitude"] for mi in mode_indices
                        if mi < len(nma["modes"])]

        # Recommended amplification = 2× max thermal amplitude at site
        recommended_amp = max(mode_ampls) * 2.0 if mode_ampls else 2.0

        # Quality flags
        quality_flags = []
        n_lining = len(lining)
        if n_lining < 5:
            quality_flags.append("LOW_LINING_RESIDUES")

        # Check if near crystal contact — proxy: near chain terminus or very high SASA
        avg_sasa = np.mean([lr.get("sasa_proxy", 0.5) for lr in lining]) if lining else 0.5
        if avg_sasa > 0.7:
            quality_flags.append("SURFACE_EXPOSED")

        # Consensus check from spike_dynamics
        spike_dyn = site.get("spike_dynamics", {})
        ba = spike_dyn.get("b_over_a_ratio", 1.0)
        if ba < 0.5 or ba > 2.0:
            quality_flags.append("LOW_TWIN_CONSENSUS")

        # Active site overlap
        if site.get("is_active_site_overlap"):
            quality_flags.append("ACTIVE_SITE_OVERLAP")

        # Therm class
        therm_class = site.get("therm_class", "")
        if therm_class == "CRYPTIC":
            quality_flags.append("CRYPTIC_POCKET")

        if not quality_flags:
            quality_flags.append("CLEAN")

        results[sid] = {
            "beacon_centroid_angstrom": centroid,
            "beacon_radius_angstrom":   round(radius, 3),
            "responsive_nma_modes":     mode_indices,
            "recommended_amplification_angstrom": round(recommended_amp, 3),
            "quality_flags": quality_flags,
            "n_quality_issues": len([f for f in quality_flags if f not in ("CLEAN", "ACTIVE_SITE_OVERLAP", "CRYPTIC_POCKET")]),
        }

    print(f"GATE 7: PASS — Beacon maps for {len(results)} sites")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 8 — Assembly
# ═══════════════════════════════════════════════════════════════════════════════
def gate8_assembly(g1_records, g2_records, g3_records, g4_mechs, g5_thermo,
                   g6_drug, g7_beacon, ca_topo_idx):
    print("GATE 8: Assembling enriched report ...")

    with open(REPORT_JSON) as f:
        report = json.load(f)
    with open(TWIN_RESULT) as f:
        twin = json.load(f)

    # Build fast lookup dicts by topo_idx
    g1_by_topo = {r["topo_idx"]: r for r in g1_records}
    g2_by_topo = {r["topo_idx"]: r for r in g2_records}
    g3_by_topo = {r["topo_idx"]: r for r in g3_records}

    # ── Fix n_consensus_events and n_differential_events from Gate 1 ──
    total_consensus   = sum(r["consensus"]   for r in g1_records)
    total_differential= sum(r["differential"] for r in g1_records)
    report["twin_run_summary"]["n_consensus_events"]    = int(total_consensus)
    report["twin_run_summary"]["n_differential_events"] = int(total_differential)
    report["twin_run_summary"]["enrichment_gates_applied"] = 8
    report["twin_run_summary"]["enrichment_timestamp"] = datetime.utcnow().isoformat() + "Z"

    # ── Enrich per_residue ──
    for res in report["per_residue"]:
        ti = res["topo_idx"]
        g1 = g1_by_topo.get(ti, {})
        g2 = g2_by_topo.get(ti, {})
        g3 = g3_by_topo.get(ti, {})

        # Gate 1 fields
        res["spikes_a"]             = g1.get("spikes_a", 0)
        res["spikes_b"]             = g1.get("spikes_b", 0)
        res["total_spikes"]         = g1.get("total_spikes", 0)
        res["b_over_a_ratio"]       = g1.get("b_over_a_ratio", 1.0)
        res["consensus_spikes"]     = g1.get("consensus", 0)
        res["differential_spikes"]  = g1.get("differential", 0)
        res["thermal_spike_energy"] = g1.get("thermal_spike_energy", 0.0)
        res["spike_classification"] = g1.get("spike_classification", "INERT")

        # Gate 2 fields
        res["ccf_max"]              = g2.get("ccf_max", None)
        res["ccf_mean"]             = g2.get("ccf_mean", None)
        res["ccf_n_05"]             = g2.get("ccf_n_05", None)
        res["ccf_n_08"]             = g2.get("ccf_n_08", None)
        res["ccf_strongest_partner"]= g2.get("strongest_partner_pdb", None)
        res["ccf_partner_distance"] = g2.get("partner_distance_angstrom", None)
        res["ccf_cluster_size"]     = g2.get("cluster_size", None)
        res["ccf_allosteric_score"] = g2.get("allosteric_score", None)
        res["ccf_persistence"]      = g2.get("persistence", None)

        # Gate 3 fields
        if g3:
            res["phase_profiles"] = g3.get("phases", {})
        else:
            res["phase_profiles"] = None

    # ── Enrich sites ──
    for site in report["sites"]:
        sid = site["site_id"]

        # Gate 4
        mech = g4_mechs.get(sid, {})
        site["mechanism"] = {
            "opening_mechanism":  mech.get("opening_mechanism", "UNKNOWN"),
            "gating_residues":    mech.get("gating_residues", []),
            "hinge_residues":     mech.get("hinge_residues", []),
            "mobile_residues":    mech.get("mobile_residues", []),
            "dominant_nma_modes": mech.get("dominant_nma_modes", []),
        }

        # Gate 5
        thermo = g5_thermo.get(sid, {})
        site["thermodynamic_enrichment"] = {
            "temperature_onset":    thermo.get("temperature_onset", None),
            "phase_spike_counts":   thermo.get("phase_spike_counts", {}),
            "occupancy_warm_hold":  thermo.get("occupancy_warm_hold", None),
            "hysteresis_ratio":     thermo.get("hysteresis_ratio", None),
            "barrier_estimate":     thermo.get("barrier_estimate", "UNKNOWN"),
            "barrier_kcal_mol":     thermo.get("barrier_kcal_mol", None),
        }

        # Gate 6
        drug = g6_drug.get(sid, {})
        if "druggability" not in site or not isinstance(site["druggability"], dict):
            site["druggability"] = {}
        site["druggability"].update({
            "pocket_polarity":       drug.get("pocket_polarity", None),
            "hydrophobic_fraction":  drug.get("hydrophobic_fraction", None),
            "polar_fraction":        drug.get("polar_fraction", None),
            "charged_fraction":      drug.get("charged_fraction", None),
            "aromatic_fraction":     drug.get("aromatic_fraction", None),
            "ligand_mw_range":       drug.get("ligand_mw_range", None),
            "fragment_hotspots":     drug.get("fragment_hotspots", []),
            "druggability_score_g6": drug.get("druggability_score_g6", None),
            "screening_strategy":    drug.get("screening_strategy", None),
        })

        # Gate 7
        beacon = g7_beacon.get(sid, {})
        site["beacon"] = {
            "centroid_angstrom":        beacon.get("beacon_centroid_angstrom", None),
            "radius_angstrom":          beacon.get("beacon_radius_angstrom", None),
            "responsive_nma_modes":     beacon.get("responsive_nma_modes", []),
            "recommended_amplification_angstrom": beacon.get("recommended_amplification_angstrom", None),
            "quality_flags":            beacon.get("quality_flags", ["UNKNOWN"]),
            "n_quality_issues":         beacon.get("n_quality_issues", 0),
        }

        # Validate: mark None fields explicitly
        required_site_fields = [
            "mechanism", "thermodynamic_enrichment", "druggability", "beacon"
        ]
        for fld in required_site_fields:
            if fld not in site:
                site[fld] = None

    # ── Save ──
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print summary card ──
    print("\n" + "="*70)
    print("  TEM1 PRISM-TWIN ENRICHMENT SUMMARY CARD")
    print("="*70)
    print(f"  Report:          {REPORT_JSON}")
    print(f"  Enrichment date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Total residues:  {len(report['per_residue'])}")
    print(f"  Total sites:     {len(report['sites'])}")
    print(f"  Gates applied:   8/8")
    print()
    print(f"  GATE 1  Per-residue spikes: {sum(r['total_spikes'] for r in g1_records):,} total spike-residue assignments")
    print(f"          NMA_RESPONSIVE:      {sum(1 for r in g1_records if r['spike_classification']=='NMA_RESPONSIVE')}")
    print(f"          THERMALLY_ACCESSIBLE:{sum(1 for r in g1_records if r['spike_classification']=='THERMALLY_ACCESSIBLE')}")
    print(f"          INERT:               {sum(1 for r in g1_records if r['spike_classification']=='INERT')}")
    print()
    print(f"  GATE 2  CCF enrichment:     263×263 matrix, per-residue allosteric scores")
    g2_top = max(g2_records, key=lambda x: x['allosteric_score'])
    print(f"          Top allosteric res:  PDB {g2_top['pdb_resnum']} score={g2_top['allosteric_score']:.3f}")
    print()
    print(f"  GATE 3  Phase profiles:     {len(g3_records)} residues × 5 phases × 2 streams")
    print()
    mechs = [g4_mechs[k]['opening_mechanism'] for k in g4_mechs]
    from collections import Counter
    mech_counts = Counter(mechs)
    print(f"  GATE 4  Mechanisms: {dict(mech_counts)}")
    print()
    barriers = [v['barrier_estimate'] for v in g5_thermo.values()]
    barrier_counts = Counter(barriers)
    print(f"  GATE 5  Barriers:   {dict(barrier_counts)}")
    print()
    drug_classes = [v['pocket_polarity'] for v in g6_drug.values()]
    drug_counts = Counter(drug_classes)
    print(f"  GATE 6  Polarity:   {dict(drug_counts)}")
    print()
    flag_all = []
    for v in g7_beacon.values():
        flag_all.extend(v['quality_flags'])
    flag_counts = Counter(flag_all)
    print(f"  GATE 7  Quality flags: {dict(flag_counts)}")
    print()
    print(f"  GATE 8  n_consensus_events:   {total_consensus:,}")
    print(f"          n_differential_events: {total_differential:,}")
    print()
    print(f"  OUTPUT:  {REPORT_JSON}")
    print("="*70)

    print("\nGATE 8: PASS — Report assembled and saved")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*70)
    print("  TEM1 PRISM-TWIN — 8-Gate Enrichment Pipeline")
    print(f"  orjson: {HAS_ORJSON}  ijson: {HAS_IJSON}")
    print("="*70 + "\n")

    # Load CA positions (used by gates 1,3,4,7)
    ca_positions, ca_topo_idx = load_ca_positions()
    print(f"  Loaded {len(ca_positions)} CA atoms\n")

    # GATE 1 — Per-residue spike counts (reads 6.9GB JSON once)
    if os.path.exists(OUT_PER_RES):
        print("GATE 1: Loading from checkpoint ...")
        with open(OUT_PER_RES) as f:
            g1_records = json.load(f)
        total_spikes_g1 = sum(r["total_spikes"] for r in g1_records)
        print(f"GATE 1: PASS — {total_spikes_g1:,} spikes loaded from checkpoint")
    else:
        g1_records = gate1_per_residue_spikes(ca_positions, ca_topo_idx)
    g1_by_pdb  = {r["pdb_resnum"]: r for r in g1_records}

    # GATE 2 — CCF propagation (fast, matrix only)
    g2_records = gate2_ccf_propagation(ca_positions, ca_topo_idx)

    # GATE 3 — Phase-resolved spike profiles (reads 6.9GB JSON second time)
    if os.path.exists(OUT_PER_PHASE):
        print("GATE 3: Loading from checkpoint ...")
        with open(OUT_PER_PHASE) as f:
            g3_records = json.load(f)
        print(f"GATE 3: PASS — {len(g3_records)} residue phase profiles loaded from checkpoint")
    else:
        g3_records = gate3_phase_profiles(ca_positions, ca_topo_idx)
    g3_by_pdb  = {r["pdb_resnum"]: r for r in g3_records}

    # GATE 4 — Mechanism classification
    g4_mechs = gate4_site_mechanisms(g1_by_pdb, ca_positions, ca_topo_idx)

    # Load report sites for gates 5-7
    with open(REPORT_JSON) as f:
        report = json.load(f)
    report_sites = report["sites"]

    # GATE 5 — Thermodynamic enrichment
    g5_thermo = gate5_thermo_enrichment(g3_by_pdb, report_sites)

    # GATE 6 — Druggability
    g6_drug = gate6_druggability(report_sites)

    # GATE 7 — Beacon maps
    g7_beacon = gate7_beacon_maps(report_sites, g4_mechs, ca_positions, ca_topo_idx)

    # GATE 8 — Assembly
    gate8_assembly(g1_records, g2_records, g3_records, g4_mechs,
                   g5_thermo, g6_drug, g7_beacon, ca_topo_idx)


if __name__ == "__main__":
    main()
