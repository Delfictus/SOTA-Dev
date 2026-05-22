#!/usr/bin/env python3
"""
prism_pub_baseline_validator.py
================================
Publication-grade target-agnostic baseline comparison validator for PRISM-4D.

Merges:
  - Proven data loading + NW/Kabsch alignment + PRISM site scoring from
    prism_manifold_shell_validator.py (working, debugged)
  - Better baseline tool invocation + LORO + family collapse from Kv31 desktop scripts

Pipeline:
  1. Load PRISM sites from run dir (binding_sites.json + kcc_visualization.json + Arrow)
  2. For each holo reference: download PDB, align backbone to query, transform ligand
  3. Build multi-shell residue sets (4/6/8 Å) using all-atom distances
  4. Score every PRISM site: min_dist, Jaccard@4/6/8, KCC driver in shell, verdict
  5. Score fpocket + p2rank proposals against same transformed ligand shells
  6. Compute SR@K for PRISM vs baselines
  7. LORO blinded validation (leave-one-reference-out)
  8. Family collapse via co-validated ligand instance overlap (union-find)
  9. Write CSV outputs + validation_report.md + pymol_overlay.pml

Usage:
  python prism_pub_baseline_validator.py \\
    --run-dir   /mnt/storage/prism-outputs/runs/MCL1_20260512_run \\
    --query-pdb /path/to/mcl1_clean.pdb \\
    --target    MCL1 \\
    --outdir    /mnt/storage/pub_validation/MCL1 \\
    --max-rank  20
"""
from __future__ import annotations
import argparse, collections, csv, json, math, os, re, shutil, subprocess, sys, urllib.request, urllib.error
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pyarrow.ipc as _ipc
    import pyarrow.compute as _pc
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

# ── Holo reference table ──────────────────────────────────────────────────────
TARGET_HOLOS: dict[str, list[dict]] = {
    "MCL1": [
        # 4HW4/5W62: no small-mol ligands in PDB format — kept for completeness, will be skipped
        {"pdb": "4HW4", "chain": "A",  "note": "MCL1 + fragment (BH3 groove, may be apo)"},
        {"pdb": "5FDR", "chain": "A",  "note": "MCL1 + AM-8621 BH3 groove inhibitor"},
        {"pdb": "5W62", "chain": "A",  "note": "MCL1 + AZD5991 (may be apo in PDB format)"},
        {"pdb": "6OQC", "chain": "A",  "note": "MCL1 self-holo (query structure reference)"},
        {"pdb": "6W8I", "chain": "A",  "note": "MCL1 + TKY BH3-groove inhibitor"},
    ],
    "KRAS_G12C": [
        {"pdb": "6OIM", "chain": "A", "note": "KRAS G12C + ARS-1620 covalent SII-P"},
        {"pdb": "6P8Y", "chain": "A", "note": "KRAS G12C + sotorasib (AMG-510) SII-P"},
        {"pdb": "7T8I", "chain": "A", "note": "KRAS G12C + adagrasib MRTX849"},
        {"pdb": "5V9L", "chain": "A", "note": "KRAS G12C apo + GDP (pocket reference)"},
    ],
    "p53_Y220C": [
        {"pdb": "6GGE", "chain": "A", "note": "p53 Y220C + EYE stabilizer (0.16Å reference)"},
        {"pdb": "6SI3", "chain": "A", "note": "p53 Y220C + PK083 triazole stabilizer"},
        {"pdb": "7ATA", "chain": "A", "note": "p53 Y220C + aminobenzimidazole stabilizer"},
        {"pdb": "7O70", "chain": "A", "note": "p53 + iminoquinol (may need chain B)"},
        {"pdb": "6TP6", "chain": "A", "note": "p53 Y220C + fragment series"},
    ],
    "TEAD3": [
        {"pdb": "6CDY", "chain": "B",  "note": "TEAD1 + EY1 palmitoyl flap-pocket (chain B, 207 CA)"},
        {"pdb": "5OAQ", "chain": "A",  "note": "TEAD2 + lipid-mimetic compound (flap)"},
        {"pdb": "6GE0", "chain": "A",  "note": "TEAD1/2 + VGX-1 YAP-interface allosteric"},
        {"pdb": "5GQM", "chain": "A",  "note": "TEAD2 + nucleotides (prob. artifact; gate check)"},
    ],
    "GLP1R": [
        {"pdb": "7LCI", "chain": "A",  "note": "GLP-1R active state (UK4 GLP-1 peptide via chain R)"},
        {"pdb": "6VCB", "chain": "A",  "note": "GLP-1R + TT-OAD2 non-peptide agonist (TMD)"},
        {"pdb": "5NX2", "chain": "A",  "note": "Gαs structure (expect gate rejection, seqid check)"},
    ],
    "STING": [
        {"pdb": "4KSY", "chain": "A", "note": "hSTING CTD + c-di-AMP CDN pocket (1SY)"},
        {"pdb": "4LOH", "chain": "A", "note": "hSTING CTD + c-di-GMP CDN pocket (1SY)"},
        {"pdb": "5CFQ", "chain": "A", "note": "hSTING CTD + DMXAA analog (1SY)"},
        {"pdb": "7BIQ", "chain": "A", "note": "hSTING + SR-717 TVB (gate check — may fail)"},
    ],
    "AKT1": [
        {"pdb": "5KCV", "chain": "A", "note": "AKT1 PH+kinase + 6S1 covalent allosteric"},
        {"pdb": "4EKK", "chain": "A", "note": "AKT1 kinase + ANP (AMPPNP ATP-site mimic)"},
        {"pdb": "3CQW", "chain": "A", "note": "AKT1 kinase + CQW ATP-competitive inhibitor"},
        {"pdb": "4GV1", "chain": "A", "note": "AKT2 kinase + 0XZ allosteric (isoform ref)"},
    ],
    "Kv31": [
        {"pdb": "7PHH", "chain": "A", "note": "Kv3.1 (7PHH) + PCF lipid at pore fenestration"},
        {"pdb": "6Y7Y", "chain": "A", "note": "Kv channel + SB4 blocker (gating/pore comparison)"},
    ],
    "TRPV1": [
        {"pdb": "5IS0", "chain": None, "note": "TRPV1 + 6ET capsaicin-site compound (vanilloid TM4/5)"},
        {"pdb": "5IRZ", "chain": None, "note": "TRPV1 + 6O8/6ES/6OE RTX-site compounds (vanilloid)"},
    ],
    # ── Blind validation targets B01-B10 (2026-05-13) ──────────────────────────
    "HRAS_Q61H": [
        {"pdb": "6OIM", "chain": "A", "note": "KRAS G12C + sotorasib (AMG510, MOV) SW2 covalent — RAS family SW2 reference"},
        {"pdb": "7RPZ", "chain": "A", "note": "KRAS G12D + MRTX1133 (6IC) SW2 non-covalent — RAS family SW2 reference"},
    ],
    "CDK2_allosteric": [
        {"pdb": "3PXZ", "chain": "A", "note": "CDK2 + JWS648 allosteric inhibitor (JWS) + ANS probe (2AN)"},
        {"pdb": "4GCJ", "chain": "A", "note": "CDK2 + RC-3-89 inhibitor (X64) — binding subsite TBD"},
    ],
    "Kv1.2": [
        {"pdb": "2R9R", "chain": "A", "note": "Kv1.2-Kv2.1 paddle chimera + beta subunit (TM lipids) — PROVISIONAL"},
        {"pdb": "2A79", "chain": "A", "note": "Full Kv1.2 + Kvbeta2 (NAP in regulatory domain) — PROVISIONAL"},
    ],
    "MDM2": [
        {"pdb": "4ODF", "chain": "A", "note": "MDM2 + compound 47 (2U1) — p53-binding groove SM inhibitor"},
        {"pdb": "4ERF", "chain": "A", "note": "MDM2 + compound 29/AM-8553 precursor (0R3) — p53-groove SM inhibitor"},
    ],
    "TP53_apo": [
        {"pdb": "3ZME", "chain": "A", "note": "p53 Y220C + compound 23 — PROVISIONAL: different pocket (Y220C not L1/H2)"},
        {"pdb": "4AGQ", "chain": "A", "note": "p53 Y220C + compound 3 — PROVISIONAL: different pocket (Y220C not L1/H2)"},
    ],
    "cGAS": [
        {"pdb": "4O67", "chain": "A", "note": "Human cGAS + cGAMP enzymatic product (1SY, MW 674 Da)"},
        {"pdb": "5V8N", "chain": "A", "note": "Human cGAS + high-affinity inhibitor (8ZP)"},
    ],
    "TEAD1": [
        {"pdb": "3KYS", "chain": "A", "note": "TEAD1 + P1L palmitate covalent at Cys344 — original pre-strip holo"},
        {"pdb": "5OAQ", "chain": "A", "note": "TEAD4 + myristate (MYR) at equivalent palmitate pocket"},
    ],
    "CRBN": [
        {"pdb": "4CI3", "chain": "B", "note": "DDB1-CRBN + pomalidomide (Y70, chain B = CRBN)"},
        {"pdb": "5FQD", "chain": "B", "note": "DDB1-CRBN + lenalidomide (LVY, chain B = CRBN)"},
    ],
    "Thrombin_exosite": [
        {"pdb": "1HAH", "chain": "H", "note": "Thrombin heavy chain H + hirugen at exosite I (chain I TYS)"},
        {"pdb": "3BEF", "chain": "B", "note": "Thrombin + PAR1 extracellular fragment at exosite I"},
    ],
    "ADRB2": [
        {"pdb": "3SN6", "chain": "R", "note": "beta2-AR + Gs + BI167107 agonist (P0G) — active state orthosteric"},
        {"pdb": "4LDE", "chain": "A", "note": "beta2-AR + BI167107 (P0G) + nanobody — active state orthosteric"},
    ],
}

SOLVENT_EXCL = {
    "HOH","WAT","DOD","SO4","PO4","GOL","EDO","PEG","ACT","FMT","MPD","BME",
    "ACE","NH2","NAG","FUC","DMS","IOD","CL","NA","MG","ZN","CA","K","MN","FE",
    "NI","CU","CO","CD","HG","SEP","TPO","PTR","ALY","MLY","MSE","UNK","CSO",
    "CME","OCS","KCX","LLP","PCA","HYP","DPN","NEP","SNN","4HY","2MR","EPE",
    "MES","TRS","HEZ","BTB","P6G","TAM","BOG","LMT","DIO","NDG","BMA",
}

MIN_LIGAND_HEAVY    = 6
SHELL_CUTOFFS       = (4.0, 6.0, 8.0)
KCC_HIGH_SCORE_THRESH = 0.55
ALIGN_RMSD_GATE     = 5.0    # skip holo if backbone RMSD > this
ALIGN_SEQID_GATE    = 0.50   # skip holo if sequence identity < this
IS_GOOD_DIST_CUTOFF = 4.0    # shared threshold for PRISM + baselines SR@K

AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "HID":"H","HIE":"H","HIP":"H","CYX":"C","MSE":"M","SEP":"S","TPO":"T",
    "PTR":"Y","HYP":"P","CSO":"C","CME":"C",
}

# ── Geometry ──────────────────────────────────────────────────────────────────

def dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def centroid(pts) -> list[float]:
    n = len(pts)
    return [sum(p[i] for p in pts) / n for i in range(3)]


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Optimal rotation R and translation t mapping P → Q. Returns (R, t, rmsd)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    aligned = (P @ R.T) + t
    rmsd = float(np.sqrt(((aligned - Q) ** 2).sum(1).mean()))
    return R, t, rmsd


def needleman(a: str, b: str) -> tuple[float, str, str]:
    """Global Needleman-Wunsch alignment. Returns (score, aligned_a, aligned_b)."""
    MATCH, MISMATCH, GAP = 2, -1, -2
    n, m = len(a), len(b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + GAP
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i-1][j-1] + (MATCH if a[i-1] == b[j-1] else MISMATCH)
            dp[i][j] = max(diag, dp[i-1][j] + GAP, dp[i][j-1] + GAP)
    # traceback
    ra, rb = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag = dp[i-1][j-1] + (MATCH if a[i-1] == b[j-1] else MISMATCH)
            if abs(dp[i][j] - diag) < 1e-9:
                ra.append(a[i-1]); rb.append(b[j-1]); i -= 1; j -= 1; continue
        if i > 0 and abs(dp[i][j] - (dp[i-1][j] + GAP)) < 1e-9:
            ra.append(a[i-1]); rb.append("-"); i -= 1
        else:
            ra.append("-"); rb.append(b[j-1]); j -= 1
    return dp[n][m], "".join(reversed(ra)), "".join(reversed(rb))

# ── PDB parsing ───────────────────────────────────────────────────────────────

def parse_pdb(path: Path) -> dict:
    """
    Parse PDB ATOM + HETATM records.
    Returns {'atoms': [...], 'hetatm': [...], 'chains': {chain_id: {'seq', 'ca', 'res'}}}.
    """
    atoms, hetatm = [], []
    chain_res: dict[str, dict] = collections.defaultdict(dict)

    with open(path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                name    = line[12:16].strip()
                resname = line[17:20].strip()
                chain   = line[21].strip() or "A"
                resnum  = int(line[22:26].strip())
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                altloc  = line[16].strip()
                if altloc and altloc not in ("A", "1", ""):
                    continue
            except (ValueError, IndexError):
                continue

            xyz = (x, y, z)
            a   = {"name": name, "resname": resname, "chain": chain,
                   "resnum": resnum, "xyz": xyz, "record": rec}
            atoms.append(a)
            if rec == "ATOM":
                if resnum not in chain_res[chain]:
                    chain_res[chain][resnum] = (resname, {})
                chain_res[chain][resnum][1][name] = xyz
            else:
                hetatm.append(a)

    chains = {}
    for chain, res_dict in chain_res.items():
        sorted_res = sorted(res_dict.items())
        seq = "".join(AA1.get(rn, "X") for _, (rn, _) in sorted_res)
        ca  = []
        for rnum, (rn, atom_dict) in sorted_res:
            if "CA" in atom_dict:
                ca.append({"resnum": rnum, "resname": rn,
                           "xyz": atom_dict["CA"], "all_atoms": atom_dict})
        chains[chain] = {"seq": seq, "ca": ca, "res": sorted_res}
    return {"atoms": atoms, "hetatm": hetatm, "chains": chains}


def ligand_groups(hetatm_atoms: list[dict]) -> list[dict]:
    """Group HETATM atoms into distinct ligand instances (>= MIN_LIGAND_HEAVY heavy atoms)."""
    by_key: dict = collections.defaultdict(list)
    for a in hetatm_atoms:
        if a["resname"] in SOLVENT_EXCL:
            continue
        if a["name"].startswith("H"):
            continue
        key = (a["chain"], a["resnum"], a["resname"])
        by_key[key].append(a["xyz"])
    return [
        {"chain": k[0], "resnum": k[1], "resname": k[2], "xyz": v}
        for k, v in by_key.items()
        if len(v) >= MIN_LIGAND_HEAVY
    ]

# ── Alignment ─────────────────────────────────────────────────────────────────

def best_align_chains(query_chains: dict, holo_chains: dict,
                      prefer_chain: Optional[str] = None
                      ) -> tuple[str, str, np.ndarray, np.ndarray, float, float]:
    """
    Find best (query_chain, holo_chain) pair by NW sequence identity, then
    Kabsch-align CA atoms. Returns (q_chain, h_chain, R, t, rmsd, seq_identity).
    """
    best = None
    for qc, qdata in query_chains.items():
        if len(qdata["ca"]) < 10:
            continue
        for hc, hdata in holo_chains.items():
            if prefer_chain and hc != prefer_chain:
                continue
            if len(hdata["ca"]) < 10:
                continue
            _, qa, ha = needleman(qdata["seq"], hdata["seq"])
            matches  = sum(1 for a, b in zip(qa, ha) if a == b and a != "-")
            identity = matches / max(len(qdata["seq"]), len(hdata["seq"]))
            if best is None or identity > best[0]:
                best = (identity, qc, hc, qdata["ca"], hdata["ca"])

    if best is None:
        return None, None, None, None, 999.0, 0.0

    identity, qc, hc, qca, hca = best
    _, qa_aln, ha_aln = needleman(
        "".join(AA1.get(d["resname"], "X") for d in qca),
        "".join(AA1.get(d["resname"], "X") for d in hca),
    )
    qi, hi = 0, 0
    qpts, hpts = [], []
    for qa_aa, ha_aa in zip(qa_aln, ha_aln):
        if qa_aa != "-" and ha_aa != "-" and qi < len(qca) and hi < len(hca):
            qpts.append(qca[qi]["xyz"])
            hpts.append(hca[hi]["xyz"])
        if qa_aa != "-": qi += 1
        if ha_aa != "-": hi += 1

    if len(qpts) < 4:
        return qc, hc, None, None, 999.0, identity

    Q_arr = np.array(qpts)
    H_arr = np.array(hpts)
    R, t, rmsd = kabsch(H_arr, Q_arr)
    return qc, hc, R, t, rmsd, identity


def transform_atoms(atoms: list[dict], R: np.ndarray, t: np.ndarray) -> list[tuple]:
    """Apply (R, t) rotation+translation to a list of atom dicts (each has 'xyz')."""
    result = []
    for a in atoms:
        v = np.array(a["xyz"]) if isinstance(a, dict) else np.array(a)
        tv = (R @ v) + t
        result.append(tuple(tv.tolist()))
    return result


def build_shell(query_chains: dict, lig_xyz_transformed: list[tuple],
                cutoff: float) -> set[int]:
    """
    Return set of query residue numbers within cutoff Å of any transformed ligand atom.
    Uses all available heavy atoms per residue.
    """
    shell = set()
    for chain, cdata in query_chains.items():
        for resnum, (resname, atom_dict) in cdata["res"]:
            q_pts = list(atom_dict.values()) if atom_dict else []
            if not q_pts:
                continue
            for lpt in lig_xyz_transformed:
                if any(dist(q, lpt) <= cutoff for q in q_pts):
                    shell.add(resnum)
                    break
    return shell

# ── PDB download ──────────────────────────────────────────────────────────────

_pdb_cache: dict[str, Path] = {}


def fetch_pdb(pdb_id: str, cache_dir: Path) -> Optional[Path]:
    """Download a PDB file to cache_dir and return the local path, or None on failure."""
    key  = pdb_id.upper()
    if key in _pdb_cache:
        return _pdb_cache[key]
    dest = cache_dir / f"{key}.pdb"
    if dest.exists():
        _pdb_cache[key] = dest
        return dest
    url = f"https://files.rcsb.org/download/{key}.pdb"
    try:
        urllib.request.urlretrieve(url, dest)
        _pdb_cache[key] = dest
        return dest
    except Exception as e:
        print(f"  [WARN] could not download {key}: {e}")
        return None

# ── PRISM run loader ──────────────────────────────────────────────────────────

def load_prism_run(run_dir: Path) -> dict:
    """
    Load binding sites from run_dir:
      *.binding_sites.json    — site lining residues, quality, therm_class
      *.kcc_visualization.json — per-site KCC driver, rank_K, centroid
      *.topology.spike_events.arrow — Arrow table with aromatic driver residues (optional)

    Returns dict: {sites, kcc_residues, high_kcc_resids, stem}
    """
    bs_files    = sorted(run_dir.glob("*.binding_sites.json"))
    kcc_files   = sorted(run_dir.glob("*.kcc_visualization.json"))
    arrow_files = sorted(run_dir.glob("*.topology.spike_events.arrow"))

    if not bs_files:
        raise FileNotFoundError(f"No *.binding_sites.json found in {run_dir}")

    bs_data  = json.loads(bs_files[0].read_text())
    kcc_data = json.loads(kcc_files[0].read_text()) if kcc_files else {}

    # Build per-residue dict from kcc_visualization (PDB-author residue IDs)
    kcc_residues: dict[int, dict] = {}
    _kcc_raw = sorted(
        [r for r in kcc_data.get("residues", []) if r.get("residue_id", -1) >= 0],
        key=lambda r: r["residue_id"]
    )
    for r in _kcc_raw:
        rid = r["residue_id"]
        kcc_residues[rid] = {
            "ca_xyz":       tuple(r["ca_position"]),
            "kcc_score":    r.get("kcc_score", 0.0),
            "burst_motion": r.get("burst_motion", 1.0),
            "causal_lag":   r.get("causal_lag", 0.0),
            "resname":      r.get("residue_name", "?"),
        }
    # 0-based sequential index → PDB-author residue_id (for translating lining / Arrow IDs)
    seq_to_pdb: list[int] = [r["residue_id"] for r in _kcc_raw]

    # Per-site KCC info from kcc_visualization
    kcc_site_drivers: dict[int, dict] = {}
    for s in kcc_data.get("sites", []):
        sid = s.get("id", -1)
        kcc = s.get("kcc") or {}
        kcc_site_drivers[sid] = {
            "driver_residue_id":     kcc.get("driver_residue_id", -1),
            "candidate_residue_ids": kcc.get("candidate_residue_ids", []),
            "kcc_confidence":        kcc.get("kcc_confidence", 0.0),
            "burst_motion":          kcc.get("burst_motion", 1.0),
            "centroid":              s.get("centroid", [0, 0, 0]),
            "rank_K":                s.get("rank_K", 0.0),
            "rank_T":                s.get("rank_T", 0.0),
            "rank_L":                s.get("rank_L", 0.0),
            "rank_C":                s.get("rank_C", 0.0),
        }

    high_kcc_resids = {
        rid for rid, rd in kcc_residues.items()
        if rd["kcc_score"] >= KCC_HIGH_SCORE_THRESH
    }

    # Arrow: per-site aromatic driver residues
    arrow_drivers: dict[int, set[int]] = collections.defaultdict(set)
    if HAS_ARROW and arrow_files:
        try:
            table    = _ipc.open_file(arrow_files[0]).read_all()
            site_col = table.column("site_id").to_pylist()
            arid_col = table.column("aromatic_residue_id").to_pylist()
            ipct_col = table.column("intensity_percentile").to_pylist()
            nr_col   = table.column("nearby_residues").to_pylist()

            site_arid_cnt: dict[int, dict[int, int]] = collections.defaultdict(
                lambda: collections.defaultdict(int))
            site_nr_cnt: dict[int, dict[int, int]] = collections.defaultdict(
                lambda: collections.defaultdict(int))

            for sid, arid, ipct, nr in zip(site_col, arid_col, ipct_col, nr_col):
                if sid < 0:
                    continue
                if arid >= 0:
                    site_arid_cnt[sid][arid] += 1
                if ipct >= 90:
                    for r in nr:
                        if r >= 0:
                            site_nr_cnt[sid][r] += 1

            def _seq2pdb(seq_id: int) -> int:
                return seq_to_pdb[seq_id] if 0 <= seq_id < len(seq_to_pdb) else seq_id

            for sid, cnt in site_arid_cnt.items():
                top = sorted(cnt.items(), key=lambda x: -x[1])[:5]
                arrow_drivers[sid] = {_seq2pdb(rid) for rid, _ in top}
            for sid, cnt in site_nr_cnt.items():
                top = sorted(cnt.items(), key=lambda x: -x[1])[:8]
                arrow_drivers[sid].update(_seq2pdb(rid) for rid, _ in top)
        except Exception as e:
            print(f"  [WARN] Arrow extraction failed: {e}")

    # Assemble sites dict
    sites: dict[int, dict] = {}
    for s in bs_data.get("sites", []):
        sid = s.get("id", -1)
        if sid < 0:
            continue
        lining = s.get("lining_residues", s.get("lining", []))
        lining_resids = [
            seq_to_pdb[r["resid"]] if 0 <= r["resid"] < len(seq_to_pdb) else r["resid"]
            for r in lining if isinstance(r, dict)
        ]
        lining_resnames = {
            (seq_to_pdb[r["resid"]] if 0 <= r["resid"] < len(seq_to_pdb) else r["resid"]): r.get("resname", "?")
            for r in lining if isinstance(r, dict)
        }
        lining_ca = [
            kcc_residues[rid]["ca_xyz"]
            for rid in lining_resids
            if rid in kcc_residues
        ]

        kcc_info   = kcc_site_drivers.get(sid, {})
        driver_rid = kcc_info.get("driver_residue_id", -1)
        candidates = set(kcc_info.get("candidate_residue_ids", []))

        kcc_cent = kcc_info.get("centroid")
        bs_cent  = s.get("centroid", s.get("centroid_xyz", [0, 0, 0]))
        cent     = kcc_cent if kcc_cent else bs_cent

        sites[sid] = {
            "id":                   sid,
            "centroid":             cent,
            "lining_resids":        set(lining_resids),
            "lining_resnames":      lining_resnames,
            "lining_ca":            lining_ca,
            "quality_score":        s.get("quality_score", 0.0),
            "therm_class":          s.get("therm_class", "?"),
            "hysteresis_asymmetry": s.get("hysteresis_asymmetry", 0.0),
            "kcc_driver_resid":     driver_rid,
            "kcc_candidates":       candidates,
            "kcc_confidence":       kcc_info.get("kcc_confidence", 0.0),
            "burst_motion":         kcc_info.get("burst_motion", 1.0),
            "rank_K":               kcc_info.get("rank_K", 0.0),
            "rank_T":               kcc_info.get("rank_T", 0.0),
            "rank_L":               kcc_info.get("rank_L", 0.0),
            "rank_C":               kcc_info.get("rank_C", 0.0),
            "arrow_driver_resids":  arrow_drivers.get(sid, set()),
            "high_kcc_in_lining":   len(set(lining_resids) & high_kcc_resids),
        }

    return {
        "sites":           sites,
        "kcc_residues":    kcc_residues,
        "high_kcc_resids": high_kcc_resids,
        "stem":            bs_files[0].stem.replace(".binding_sites", ""),
    }

# ── PRISM site scoring ────────────────────────────────────────────────────────

def score_site_vs_ligand(site: dict, kcc_residues: dict,
                         lig_xyz_t: list[tuple],
                         shells: dict[float, set[int]],
                         query_resnum_map: dict[int, int]) -> dict:
    """
    Score one PRISM site against a transformed holo ligand.
    query_resnum_map: topology_resid → PDB resnum in query (identity map when PRISM
    IDs already match PDB author numbering).
    Returns a detailed scoring dict with is_good key.
    """
    lining_q = {query_resnum_map.get(r, r) for r in site["lining_resids"]}
    site_cent = site["centroid"]

    # Minimum CA-to-ligand distance across all lining residues
    min_dist = 999.0
    if site["lining_ca"]:
        for ca in site["lining_ca"]:
            for lpt in lig_xyz_t:
                d = dist(ca, lpt)
                if d < min_dist:
                    min_dist = d

    # Centroid-to-ligand-centroid distance
    lig_cent  = centroid(lig_xyz_t) if lig_xyz_t else [0, 0, 0]
    cent_dist = dist(site_cent, lig_cent)

    # Jaccard and intersection at each cutoff
    jaccard: dict[float, float] = {}
    intersection_cnt: dict[float, int] = {}
    for c in SHELL_CUTOFFS:
        sh    = shells[c]
        inter = lining_q & sh
        union = lining_q | sh
        jaccard[c]          = len(inter) / max(1, len(union))
        intersection_cnt[c] = len(inter)

    # KCC/Arrow driver features
    shell_8    = shells[8.0]
    kcc_dr_q   = query_resnum_map.get(site["kcc_driver_resid"], site["kcc_driver_resid"])
    kcc_cand_q = {query_resnum_map.get(r, r) for r in site["kcc_candidates"]}
    arrow_q    = {query_resnum_map.get(r, r) for r in site["arrow_driver_resids"]}

    kcc_driver_in_shell   = kcc_dr_q in shell_8 and kcc_dr_q >= 0
    kcc_cand_in_shell     = len(kcc_cand_q & shell_8)
    arrow_driver_in_shell = len(arrow_q & shell_8) > 0

    high_kcc_in_8a_shell = 0
    for rid, rd in kcc_residues.items():
        if rd["kcc_score"] >= KCC_HIGH_SCORE_THRESH:
            qr = query_resnum_map.get(rid, rid)
            if qr in shell_8:
                high_kcc_in_8a_shell += 1

    # Verdict — Directive v2 definitions (METRIC CHANGE 2026-05-13, see audit log)
    # EXCELLENT = min_dist <= 4.0 Å AND intersection_cnt[8.0] >= 3  (Strict SR@k, primary)
    # STRONG    = min_dist <= 6.0 Å AND intersection_cnt[8.0] >= 2  (Practical SR@k, secondary)
    # NEAR      = min_dist <= 8.0 Å AND intersection_cnt[8.0] >= 1  (Exploratory SR@k, tertiary)
    # MISS      = otherwise
    if min_dist <= 4.0 and intersection_cnt[8.0] >= 3:
        verdict = "EXCELLENT"
    elif min_dist <= 6.0 and intersection_cnt[8.0] >= 2:
        verdict = "STRONG"
    elif min_dist <= 8.0 and intersection_cnt[8.0] >= 1:
        verdict = "NEAR"
    else:
        verdict = "MISS"

    return {
        "site_id":                site["id"],
        "therm_class":            site["therm_class"],
        "quality_score":          site["quality_score"],
        "hysteresis_asymmetry":   site["hysteresis_asymmetry"],
        "kcc_confidence":         site["kcc_confidence"],
        "burst_motion":           site["burst_motion"],
        "rank_K":                 site["rank_K"],
        "rank_T":                 site["rank_T"],
        "lining_size":            len(site["lining_resids"]),
        "min_lining_dist_A":      round(min_dist, 3),
        "centroid_to_ligand_A":   round(cent_dist, 3),
        "intersection_4A":        intersection_cnt[4.0],
        "intersection_6A":        intersection_cnt[6.0],
        "intersection_8A":        intersection_cnt[8.0],
        "jaccard_4A":             round(jaccard[4.0], 4),
        "jaccard_6A":             round(jaccard[6.0], 4),
        "jaccard_8A":             round(jaccard[8.0], 4),
        "kcc_driver_in_8A_shell": kcc_driver_in_shell,
        "kcc_candidates_in_8A":   kcc_cand_in_shell,
        "arrow_driver_in_8A":     arrow_driver_in_shell,
        "high_kcc_resids_in_8A":  high_kcc_in_8a_shell,
        "verdict":                verdict,
        "is_good":       verdict == "EXCELLENT",           # Strict (primary)
        "is_practical":  verdict in ("EXCELLENT", "STRONG"),  # Practical (secondary)
        "is_exploratory": verdict in ("EXCELLENT", "STRONG", "NEAR"),  # Exploratory (tertiary)
    }

# ── Baseline tool runners ─────────────────────────────────────────────────────

def run_fpocket(query_pdb: Path, outdir: Path) -> list[dict]:
    """
    Run fpocket via snap-confined binary.  The snap sandbox only permits access
    to $HOME, so we stage the PDB in ~/.prism_fpocket_work/<stem>/, run fpocket
    there (it creates <stem>_out/ in that directory), then copy the pocket PDBs
    back to outdir and clean up.
    """
    # snap fpocket can only access non-hidden dirs under $HOME (e.g. ~/Desktop).
    # Stage the PDB there, run, copy results back, then clean up.
    work_root = Path.home() / "Desktop" / "prism_fpocket_work"
    stem      = query_pdb.stem
    work_dir  = work_root / stem
    work_dir.mkdir(parents=True, exist_ok=True)
    local     = work_dir / query_pdb.name
    shutil.copy2(query_pdb, local)
    try:
        r = subprocess.run(["fpocket", "-f", str(local)],
                           capture_output=True, text=True, timeout=180)
        if "Structure reading failed" in r.stderr or "Structure reading failed" in r.stdout:
            print(f"  [WARN] fpocket: structure reading failed (snap access issue?)")
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  [WARN] fpocket failed: {e}")
        return []

    proposals = []
    out_dir    = work_dir / f"{stem}_out"
    fp_archive = outdir / "fpocket_out"
    if out_dir.exists() and not fp_archive.exists():
        shutil.copytree(out_dir, fp_archive)
    search_dir = fp_archive if fp_archive.exists() else out_dir
    for pf in sorted(search_dir.glob("pockets/pocket*_atm.pdb")):
        m    = re.search(r"pocket(\d+)", pf.name)
        rank = int(m.group(1)) if m else len(proposals) + 1
        pts  = []
        for line in pf.read_text(errors="ignore").splitlines():
            if line[:6] in ("ATOM  ", "HETATM"):
                try:
                    pts.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    pass
        if pts:
            proposals.append({"method": "fpocket", "rank": rank, "points": pts})
    return sorted(proposals, key=lambda x: x["rank"])


def run_p2rank(query_pdb: Path, outdir: Path) -> list[dict]:
    """
    Run /opt/p2rank/prank predict. Returns proposals from *_predictions.csv.
    Each p2rank proposal is represented as a single centroid point.
    """
    p2out = outdir / "p2rank_out"
    p2out.mkdir(exist_ok=True)
    try:
        r = subprocess.run(
            ["/opt/p2rank/prank", "predict", "-f", str(query_pdb),
             "-o", str(p2out), "-threads", "4"],
            capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f"  [WARN] p2rank exit {r.returncode}: {r.stderr[:200]}")
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  [WARN] p2rank failed: {e}")
        return []

    pred = next(p2out.rglob("*_predictions.csv"), None)
    if not pred:
        return []

    proposals = []
    with open(pred, newline="") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            def get(*names):
                for n in names:
                    for k, v in row.items():
                        if k.strip().lower() == n.lower():
                            return v
                return None
            try:
                cx       = float(get("center_x") or 0)
                cy       = float(get("center_y") or 0)
                cz       = float(get("center_z") or 0)
                rank_val = get("rank")
                rank     = int(float(rank_val)) if rank_val else i
                proposals.append({"method": "p2rank", "rank": rank, "points": [(cx, cy, cz)]})
            except (ValueError, TypeError):
                pass
    return proposals

# ── Proposal scoring ──────────────────────────────────────────────────────────

def score_proposal_vs_ligand(proposal: dict, query_res_atoms: dict[int, list],
                              lig_xyz_t: list[tuple],
                              ligand_shell_8: set[int]) -> Optional[dict]:
    """
    Score a baseline proposal (fpocket/p2rank) against a transformed ligand.
    query_res_atoms: resnum → list of (x,y,z) for all heavy atoms.
    ligand_shell_8: set of query resnums within 8Å of ligand.
    Returns scoring dict or None if proposal or ligand is empty.
    """
    pts = proposal["points"]
    if not pts or not lig_xyz_t:
        return None

    # Minimum distance: any proposal point to any ligand atom
    min_d = min(dist(p, l) for p in pts for l in lig_xyz_t)

    # Near-residue set: query residues within 4Å of any proposal point
    near_res: set[int] = set()
    for resnum, atoms in query_res_atoms.items():
        if any(dist(a, p) <= 4.0 for a in atoms for p in pts):
            near_res.add(resnum)

    # Overlap with ligand 8Å shell
    inter = near_res & ligand_shell_8
    union = near_res | ligand_shell_8
    j8    = len(inter) / max(1, len(union))

    # Verdict (Kv31-style)
    if min_d <= 4.0 and len(inter) >= 3:
        verdict = "EXCELLENT"
    elif min_d <= 6.0 and len(inter) >= 2:
        verdict = "STRONG"
    elif min_d <= 8.0 and len(inter) >= 1:
        verdict = "NEAR"
    elif min_d <= 10.0:
        verdict = "WEAK"
    else:
        verdict = "MISS"

    return {
        "method":           proposal["method"],
        "rank":             proposal["rank"],
        "min_dist_A":       round(min_d, 3),
        "near_residues_4A": len(near_res),
        "intersection_8A":  len(inter),
        "jaccard_8A":       round(j8, 4),
        "is_good":          min_d <= IS_GOOD_DIST_CUTOFF,
        "verdict":          verdict,
    }

# ── SR@K ──────────────────────────────────────────────────────────────────────

def compute_sr_at_k(rows: list[dict], method: str,
                    ks: tuple = (1, 3, 5, 10)) -> list[dict]:
    """
    Compute SR@K for one method — three tiers per Directive v2.
    Strict     (primary):   EXCELLENT only          — is_good
    Practical  (secondary): EXCELLENT or STRONG     — is_practical
    Exploratory(tertiary):  EXCELLENT/STRONG/NEAR   — is_exploratory
    rows must contain: ligand_key, rank, is_good, is_practical, is_exploratory.
    Baselines use is_good as the sole hit flag (mapped to EXCELLENT-equivalent threshold).
    """
    by_lig: dict = collections.defaultdict(list)
    for r in rows:
        by_lig[r["ligand_key"]].append(r)

    n = max(1, len(by_lig))
    out = []
    for K in ks:
        strict = sum(
            1 for rs in by_lig.values()
            if any(r["rank"] <= K and r.get("is_good", False) for r in rs)
        )
        practical = sum(
            1 for rs in by_lig.values()
            if any(r["rank"] <= K and r.get("is_practical", r.get("is_good", False)) for r in rs)
        )
        exploratory = sum(
            1 for rs in by_lig.values()
            if any(r["rank"] <= K and r.get("is_exploratory", r.get("is_good", False)) for r in rs)
        )
        out.append({
            "method":               method,
            "K":                    K,
            "ligand_instances":     len(by_lig),
            "strict_hits":          strict,
            "Strict_SR@K":          round(strict / n, 4),
            "practical_hits":       practical,
            "Practical_SR@K":       round(practical / n, 4),
            "exploratory_hits":     exploratory,
            "Exploratory_SR@K":     round(exploratory / n, 4),
            # Legacy field — kept for downstream compat; equals Strict_SR@K
            "hits":                 strict,
            "SR@K":                 round(strict / n, 4),
        })
    return out

# ── LORO blinded validation ───────────────────────────────────────────────────

def loro_blinded(all_prism_rows: list[dict], holo_ids: list[str],
                 max_rank: int = 10) -> list[dict]:
    """
    Leave-one-reference-out blinded validation.
    For each withheld holo:
      - Train: site IDs that achieve is_good=True for ANY other holo in top-max_rank
      - Test:  which trained sites recover the withheld holo's ligand instances?

    Status values:
      BLINDED_REDISCOVERY      — withheld ligand recovered by a trained site (best case)
      TRAINED_SITE_NEAR_WITHHELD — trained site appeared but wasn't marked is_good
      WITHHELD_ONLY_NEW_SITE  — found but only by a new (untrained) site
      FAIL                    — not recovered at all
    """
    out = []
    for withheld in holo_ids:
        train = [r for r in all_prism_rows
                 if r["holo_pdb"] != withheld
                 and r["rank_K"] <= max_rank
                 and r["is_good"]]
        trained_sites = {r["site_id"] for r in train}

        test_by_lig: dict = collections.defaultdict(list)
        for r in all_prism_rows:
            if r["holo_pdb"] == withheld:
                test_by_lig[r["ligand_key"]].append(r)

        for lk, rs in sorted(test_by_lig.items()):
            in_trained      = [r for r in rs if r["site_id"] in trained_sites]
            good_in_trained = [r for r in in_trained if r["is_good"]]
            any_good        = [r for r in rs if r["is_good"]]

            best_good_trained = min(good_in_trained, key=lambda x: x["rank_K"], default=None)
            best_in_trained   = min(in_trained,      key=lambda x: x["rank_K"], default=None)
            best_any          = min(any_good,         key=lambda x: x["rank_K"], default=None)

            if best_good_trained:
                status = "BLINDED_REDISCOVERY"
            elif best_in_trained:
                status = "TRAINED_SITE_NEAR_WITHHELD"
            elif best_any:
                status = "WITHHELD_ONLY_NEW_SITE"
            else:
                status = "FAIL"

            out.append({
                "withheld_holo":          withheld,
                "ligand_key":             "|".join(str(x) for x in lk),
                "ligand_resname":         lk[3] if len(lk) > 3 else "",
                "trained_sites_n":        len(trained_sites),
                "status":                 status,
                "best_good_trained_site": best_good_trained["site_id"] if best_good_trained else "",
                "best_good_trained_rank": best_good_trained["rank_K"]  if best_good_trained else "",
                "best_good_trained_dist": round(best_good_trained["min_lining_dist_A"], 3)
                                          if best_good_trained else "",
                "best_any_good_rank":     best_any["rank_K"] if best_any else "",
            })
    return out

# ── Family collapse ───────────────────────────────────────────────────────────

def collapse_families(all_prism_rows: list[dict], max_rank: int = 10) -> list[dict]:
    """
    Collapse PRISM sites into families by co-validated ligand instances.
    Two sites are merged if they both achieve is_good=True for >= 2 of the same
    ligand instances in the top-max_rank results (union-find).
    Returns list of family summary dicts sorted by coverage.
    """
    # site → set of ligand_keys where it's good and within max_rank
    site_to_ligs: dict = collections.defaultdict(set)
    for r in all_prism_rows:
        if r["rank_K"] <= max_rank and r["is_good"]:
            site_to_ligs[r["site_id"]].add(r["ligand_key"])

    sites  = sorted(site_to_ligs.keys(), key=str)
    parent = {s: s for s in sites}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(sites):
        for b in sites[i+1:]:
            if len(site_to_ligs[a] & site_to_ligs[b]) >= 2:
                union(a, b)

    fams: dict = collections.defaultdict(list)
    for s in sites:
        fams[find(s)].append(s)

    # Build quality lookup for representative site selection
    site_all_rows: dict = collections.defaultdict(list)
    for r in all_prism_rows:
        site_all_rows[r["site_id"]].append(r)

    family_rows = []
    for idx, members in enumerate(
        sorted(fams.values(), key=lambda m: (-len(m), str(min(m, key=str)))), 1
    ):
        all_hits = [r for s in members for r in site_all_rows[s] if r["is_good"]]
        ligs     = {r["ligand_key"] for r in all_hits}
        refs     = {r["holo_pdb"]   for r in all_hits}
        best_d   = min((r["min_lining_dist_A"] for r in all_hits), default=999.0)
        max_j    = max((r["jaccard_8A"]         for r in all_hits), default=0.0)
        max_inter = max((r["intersection_8A"]   for r in all_hits), default=0)

        rep_site = max(members, key=lambda s: max(
            (r["quality_score"] for r in site_all_rows[s]), default=0))

        family_rows.append({
            "family_id":           f"FAM_{idx:02d}",
            "sites":               " ".join(str(s) for s in sorted(members, key=str)),
            "n_sites":             len(members),
            "rep_site":            str(rep_site),
            "n_ligand_instances":  len(ligs),
            "n_reference_pdbs":    len(refs),
            "reference_pdbs":      " ".join(sorted(refs)),
            "best_min_dist_A":     round(best_d, 3),
            "max_jaccard_8A":      round(max_j, 4),
            "max_intersection_8A": max_inter,
        })
    return sorted(family_rows,
                  key=lambda x: (-x["n_ligand_instances"], x["best_min_dist_A"]))

# ── Output writers ────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict]):
    """Write list-of-dicts to CSV. Writes empty file if rows is empty."""
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_report(outdir: Path, target: str,
                 prism_rows: list[dict],
                 baseline_rows: list[dict],
                 loro_rows: list[dict],
                 family_rows: list[dict],
                 holo_summary: list[dict],
                 sr_prism: list[dict],
                 sr_baselines: list[dict]) -> None:
    """
    Write validation_report.md to outdir with:
      - Target summary header
      - SR@K comparison table (PRISM + fpocket + p2rank)
      - LORO blinded summary
      - Holo alignment quality table
      - Top sites by quality
      - Family collapse summary
    """
    n_sites     = len({r["site_id"]  for r in prism_rows})
    n_holos     = len({r["holo_pdb"] for r in prism_rows})
    n_lig_inst  = len({r["ligand_key"] for r in prism_rows})
    n_cryptic   = len({r["site_id"] for r in prism_rows if r.get("therm_class") == "CRYPTIC"})
    n_resp      = len({r["site_id"] for r in prism_rows if r.get("therm_class") == "RESPONSIVE"})

    # LORO summary helpers
    n_blinded  = sum(1 for r in loro_rows if r["status"] == "BLINDED_REDISCOVERY")
    n_trained  = sum(1 for r in loro_rows if r["status"] == "TRAINED_SITE_NEAR_WITHHELD")
    n_fail     = sum(1 for r in loro_rows if r["status"] == "FAIL")
    n_loro     = len(loro_rows)

    # Build SR@K lookup by method for easy table construction
    def sr_lookup(rows: list[dict]) -> dict[str, dict[int, float]]:
        out: dict[str, dict[int, float]] = {}
        for r in rows:
            out.setdefault(r["method"], {})[r["K"]] = r["SR@K"]
        return out

    sr_map    = sr_lookup(sr_prism)
    sr_bl_map = sr_lookup(sr_baselines)
    all_methods_order = ["PRISM"] + [m for m in sr_bl_map if m not in sr_map]

    # Ligand instance count per method
    def n_ligs_for(rows: list[dict], method: str) -> int:
        return next((r["ligand_instances"] for r in rows if r["method"] == method), 0)

    lines = [
        f"# PRISM-4D Publication Baseline Validation Report",
        f"## Target: {target}",
        f"",
        f"**Sites scored**: {n_sites}  "
        f"(CRYPTIC: {n_cryptic}  RESPONSIVE: {n_resp})",
        f"**Holo references used**: {n_holos}  "
        f"**Ligand instances**: {n_lig_inst}",
        f"**Site families collapsed**: {len(family_rows)}",
        f"",
        f"---",
        f"",
        f"### SR@K — PRISM vs Baselines",
        f"",
        f"| Method | SR@1 | SR@3 | SR@5 | SR@10 | N ligands |",
        f"|--------|-----:|-----:|-----:|------:|----------:|",
    ]

    for method in all_methods_order:
        m_map = sr_map.get(method) or sr_bl_map.get(method, {})
        n_l   = n_ligs_for(sr_prism + sr_baselines, method)
        lines.append(
            f"| {method:8s} "
            f"| {m_map.get(1, 0.0):.3f} "
            f"| {m_map.get(3, 0.0):.3f} "
            f"| {m_map.get(5, 0.0):.3f} "
            f"| {m_map.get(10, 0.0):.3f} "
            f"| {n_l} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"### LORO Blinded Summary",
        f"",
        f"Total LORO tests: **{n_loro}** (one per unique ligand instance × withheld holo)",
        f"",
        f"| Status | Count |",
        f"|--------|------:|",
        f"| BLINDED_REDISCOVERY | {n_blinded} |",
        f"| TRAINED_SITE_NEAR_WITHHELD | {n_trained} |",
        f"| WITHHELD_ONLY_NEW_SITE | {sum(1 for r in loro_rows if r['status'] == 'WITHHELD_ONLY_NEW_SITE')} |",
        f"| FAIL | {n_fail} |",
        f"",
        f"**Blinded rediscovery rate**: "
        f"{n_blinded}/{n_loro} = {n_blinded/max(1,n_loro):.1%}",
        f"",
    ]

    # Per-holo LORO breakdown
    by_withheld: dict = collections.defaultdict(list)
    for r in loro_rows:
        by_withheld[r["withheld_holo"]].append(r)
    if by_withheld:
        lines += [
            f"#### Per-holo LORO breakdown",
            f"",
            f"| Withheld holo | BLINDED_REDISCOVERY | Total tests |",
            f"|---------------|--------------------:|------------:|",
        ]
        for holo_id in sorted(by_withheld):
            rs    = by_withheld[holo_id]
            n_br  = sum(1 for r in rs if r["status"] == "BLINDED_REDISCOVERY")
            lines.append(f"| {holo_id} | {n_br} | {len(rs)} |")

    lines += [
        f"",
        f"---",
        f"",
        f"### Holo Alignment Quality",
        f"",
        f"| PDB | Chain | RMSD (Å) | Seq identity | N ligands | Status |",
        f"|-----|-------|--------:|-------------:|----------:|--------|",
    ]
    for h in holo_summary:
        lines.append(
            f"| {h['pdb']} "
            f"| {h.get('holo_chain','?')} "
            f"| {h.get('rmsd_A', 999.0):.2f} "
            f"| {h.get('seq_identity', 0.0):.2f} "
            f"| {h.get('n_ligands', 0)} "
            f"| {h.get('status','?')} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"### Top Sites by Quality",
        f"",
        f"Top 10 PRISM sites ranked by quality_score, "
        f"showing best scoring result across all holo references.",
        f"",
        f"| Site | Therm | Quality | rank_K | Best dist (Å) | Best J@8 | Best verdict |",
        f"|------|-------|--------:|-------:|--------------:|---------:|-------------|",
    ]
    site_ids = sorted(
        {r["site_id"] for r in prism_rows},
        key=lambda sid: -max(
            (r["quality_score"] for r in prism_rows if r["site_id"] == sid), default=0)
    )[:10]
    for sid in site_ids:
        site_r = [r for r in prism_rows if r["site_id"] == sid]
        best_d = min(r["min_lining_dist_A"] for r in site_r)
        best_j = max(r["jaccard_8A"]        for r in site_r)
        best_v = min(site_r, key=lambda r: r["min_lining_dist_A"])["verdict"]
        ex     = site_r[0]
        lines.append(
            f"| {sid} "
            f"| {ex.get('therm_class','?')} "
            f"| {ex.get('quality_score',0):.3f} "
            f"| {ex.get('rank_K','?')} "
            f"| {best_d:.2f} "
            f"| {best_j:.3f} "
            f"| {best_v} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"### Family Collapse Summary",
        f"",
        f"Sites merged by co-validated ligand instances (≥2 shared is_good hits).",
        f"",
        f"| Family | N sites | Rep site | N ligands | N refs | Best dist (Å) | Max J@8 |",
        f"|--------|--------:|---------:|----------:|-------:|--------------:|--------:|",
    ]
    for fam in family_rows[:15]:
        lines.append(
            f"| {fam['family_id']} "
            f"| {fam['n_sites']} "
            f"| {fam['rep_site']} "
            f"| {fam['n_ligand_instances']} "
            f"| {fam['n_reference_pdbs']} "
            f"| {fam['best_min_dist_A']:.2f} "
            f"| {fam['max_jaccard_8A']:.3f} |"
        )

    lines += [f"", f"---", f"", f"*Generated by prism_pub_baseline_validator.py*", f""]
    (outdir / "validation_report.md").write_text("\n".join(lines))


def write_pymol(outdir: Path, target: str, query_pdb: Path,
                prism_rows: list[dict], loro_rows: list[dict]) -> None:
    """
    Write pymol_overlay.pml that loads the query PDB and visualises top-scoring
    PRISM sites as spheres colour-coded by verdict.
    """
    verdict_color = {
        "STRONG":       "red",
        "GOOD_BOTH":    "orange",
        "GOOD_DIST":    "yellow",
        "GOOD_JACCARD": "green",
        "NEAR":         "cyan",
        "WEAK":         "blue",
        "MISS":         "grey60",
    }

    # Build per-site best-verdict across all holos
    site_best: dict[int, dict] = {}
    for r in prism_rows:
        sid = r["site_id"]
        if sid not in site_best or r["min_lining_dist_A"] < site_best[sid]["min_lining_dist_A"]:
            site_best[sid] = r

    # Collect LORO blinded-rediscovery site IDs for annotation
    blinded_sites = {
        r.get("best_good_trained_site")
        for r in loro_rows
        if r.get("status") == "BLINDED_REDISCOVERY" and r.get("best_good_trained_site") != ""
    }

    pml_lines = [
        f"# PRISM-4D pub baseline validation overlay — {target}",
        f"# Generated by prism_pub_baseline_validator.py",
        f"",
    ]

    if query_pdb and query_pdb.exists():
        pml_lines += [
            f"load {query_pdb}, query_{target}",
            f"hide everything, query_{target}",
            f"show cartoon,    query_{target}",
            f"color grey80,    query_{target}",
            f"",
        ]

    # Sort sites: good ones first, then by quality_score descending
    rank_verdict = ["STRONG","GOOD_BOTH","GOOD_DIST","GOOD_JACCARD","NEAR","WEAK","MISS"]
    sorted_sites = sorted(
        site_best.values(),
        key=lambda r: (rank_verdict.index(r["verdict"])
                       if r["verdict"] in rank_verdict else 99,
                       -r["quality_score"])
    )

    for r in sorted_sites[:20]:
        sid   = r["site_id"]
        color = verdict_color.get(r["verdict"], "grey50")
        cent  = r.get("centroid_to_ligand_A", "?")  # not stored; use placeholder
        label = f"site_{sid}"
        blinded_flag = " [BLINDED]" if str(sid) in blinded_sites else ""
        pml_lines += [
            f"# Site {sid} [{r.get('therm_class','?')}] "
            f"q={r.get('quality_score',0):.3f} "
            f"verdict={r['verdict']}{blinded_flag}",
        ]

        # Use site centroid from the kcc_visualization via quality_score row
        # The centroid isn't stored in scoring rows; emit a note and skip sphere
        # if we don't have it.  In production, site centroid is in prism rows
        # if the caller adds it — omit silently here.

    # Add spheres from the site centroids stored in site scoring rows
    # (centroid_to_ligand_A is a distance, not xyz — so we emit pseudoatoms
    #  only when explicit centroid XYZ is available via a separate field)
    for r in sorted_sites[:20]:
        sid   = r["site_id"]
        color = verdict_color.get(r["verdict"], "grey50")
        # Centroid XYZ is not in scoring rows by default; skip pseudoatom if absent
        cx = r.get("centroid_x")
        cy = r.get("centroid_y")
        cz = r.get("centroid_z")
        if cx is not None and cy is not None and cz is not None:
            label = f"site_{sid}"
            pml_lines += [
                f'pseudoatom {label}, pos=[{cx:.2f},{cy:.2f},{cz:.2f}]',
                f"color {color}, {label}",
                f"show sphere, {label}",
                f"set sphere_scale, 1.5, {label}",
                f"",
            ]

    pml_lines += [
        f"zoom query_{target}" if query_pdb and query_pdb.exists() else "zoom",
        f"",
        f"# Verdict colour legend:",
    ]
    for v, c in verdict_color.items():
        pml_lines.append(f"#   {v:15s} -> {c}")

    (outdir / "pymol_overlay.pml").write_text("\n".join(pml_lines) + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="PRISM-4D publication-grade baseline comparison validator")
    ap.add_argument("--run-dir",   required=True,
                    help="PRISM run output directory")
    ap.add_argument("--query-pdb", required=True,
                    help="Clean query PDB (prism-clean output)")
    ap.add_argument("--target",    required=True,
                    help="Target name (must match TARGET_HOLOS key)")
    ap.add_argument("--outdir",    required=True,
                    help="Output directory for all validation artifacts")
    ap.add_argument("--max-rank",  type=int, default=20,
                    help="Max site rank to consider (default 20)")
    args = ap.parse_args()

    run_dir   = Path(args.run_dir)
    query_pdb = Path(args.query_pdb)
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = outdir / ".pdb_cache"
    cache_dir.mkdir(exist_ok=True)

    target = args.target
    holos  = TARGET_HOLOS.get(target)
    if not holos:
        sys.exit(f"Unknown target: {target}. Known targets: {list(TARGET_HOLOS)}")

    print(f"\n=== PRISM Publication Baseline Validator: {target} ===")
    print(f"  run_dir:   {run_dir}")
    print(f"  query_pdb: {query_pdb}")
    print(f"  outdir:    {outdir}")
    print(f"  max_rank:  {args.max_rank}")

    # ── 1. Load PRISM run ─────────────────────────────────────────────────
    print("\n[1] Loading PRISM run data...")
    prism        = load_prism_run(run_dir)
    sites        = prism["sites"]
    kcc_residues = prism["kcc_residues"]
    print(f"  Sites: {len(sites)}  KCC residues: {len(kcc_residues)}")

    # ── 2. Parse query PDB ────────────────────────────────────────────────
    print("\n[2] Parsing query PDB...")
    if not query_pdb.exists():
        sys.exit(f"Query PDB not found: {query_pdb}")
    query_parsed   = parse_pdb(query_pdb)
    query_chains   = query_parsed["chains"]

    # query_res_atoms: resnum → list of (x,y,z) for all heavy atoms (for proposal scoring)
    query_res_atoms: dict[int, list] = {}
    for ch, cdata in query_chains.items():
        for resnum, (resname, atom_dict) in cdata["res"]:
            pts = list(atom_dict.values())
            if pts:
                query_res_atoms[resnum] = pts

    # Identity map: query author resnums map to themselves.
    # PRISM lining resids are already in PDB-author space after seq_to_pdb translation
    # in load_prism_run, so no offset needed when IDs match.
    # Auto-detect offset by comparing kcc_viz CA positions to query PDB CA.
    topo_offset = 0
    if kcc_residues and query_chains:
        qc_data = next(iter(query_chains.values()))
        qca_by_rnum = {d["resnum"]: np.array(d["xyz"]) for d in qc_data["ca"]}
        sample_rids = sorted(kcc_residues.keys())[:10]
        best_off, best_err = 0, 1e9
        for off in range(-5, 6):
            errs = []
            for rid in sample_rids:
                qrnum = rid + off
                if qrnum in qca_by_rnum:
                    kca = np.array(kcc_residues[rid]["ca_xyz"])
                    errs.append(float(np.linalg.norm(kca - qca_by_rnum[qrnum])))
            if errs and sum(errs) / len(errs) < best_err:
                best_err = sum(errs) / len(errs)
                best_off = off
        topo_offset = best_off
        print(f"  Auto-detected topology offset: {topo_offset} "
              f"(mean CA Δ = {best_err:.2f} Å)")

    query_resnum_map: dict[int, int] = {rid: rid + topo_offset for rid in kcc_residues}

    # ── 3. Run baseline tools once per target ─────────────────────────────
    print(f"\n[3] Running baseline tools...")
    print(f"  fpocket...")
    fp_proposals = run_fpocket(query_pdb, outdir)
    print(f"  fpocket: {len(fp_proposals)} proposals")
    print(f"  p2rank...")
    p2_proposals = run_p2rank(query_pdb, outdir)
    print(f"  p2rank: {len(p2_proposals)} proposals")

    # ── 4. Process each holo reference ───────────────────────────────────
    print(f"\n[4] Processing {len(holos)} holo references...")
    all_prism_rows:    list[dict] = []
    all_baseline_rows: list[dict] = []
    holo_summary:      list[dict] = []
    seen_holos:        set[str]   = set()

    for holo_spec in holos:
        pdb_id       = holo_spec["pdb"]
        prefer_chain = holo_spec.get("chain")
        note         = holo_spec.get("note", "")
        print(f"\n  [{pdb_id}] {note}")

        pdb_path = fetch_pdb(pdb_id, cache_dir)
        if not pdb_path:
            holo_summary.append({"pdb": pdb_id, "holo_chain": "?", "query_chain": "?",
                                  "rmsd_A": 999.0, "seq_identity": 0.0,
                                  "n_ligands": 0, "status": "DOWNLOAD_FAILED",
                                  "note": note})
            continue

        holo_parsed = parse_pdb(pdb_path)
        holo_chains = holo_parsed["chains"]

        qc, hc, R, t, rmsd, seqid = best_align_chains(
            query_chains, holo_chains, prefer_chain=prefer_chain)

        print(f"    RMSD={rmsd:.2f}Å  seqid={seqid:.2f}  qchain={qc}  hchain={hc}")

        if R is None or rmsd > ALIGN_RMSD_GATE or seqid < ALIGN_SEQID_GATE:
            status = "SKIPPED_QUALITY_GATE" if R is not None else "ALIGN_FAILED"
            print(f"    [SKIP] {status}")
            holo_summary.append({"pdb": pdb_id, "holo_chain": hc or "?",
                                  "query_chain": qc or "?",
                                  "rmsd_A": round(rmsd, 2),
                                  "seq_identity": round(seqid, 3),
                                  "n_ligands": 0, "status": status, "note": note})
            continue

        ligs = ligand_groups(holo_parsed["hetatm"])
        if not ligs:
            print(f"    [SKIP] no ligands (heavy atoms >= {MIN_LIGAND_HEAVY})")
            holo_summary.append({"pdb": pdb_id, "holo_chain": hc,
                                  "query_chain": qc,
                                  "rmsd_A": round(rmsd, 2),
                                  "seq_identity": round(seqid, 3),
                                  "n_ligands": 0, "status": "NO_LIGANDS", "note": note})
            continue

        print(f"    Ligands: {[(l['chain'], l['resnum'], l['resname']) for l in ligs]}")
        holo_summary.append({"pdb": pdb_id, "holo_chain": hc, "query_chain": qc,
                              "rmsd_A": round(rmsd, 2), "seq_identity": round(seqid, 3),
                              "n_ligands": len(ligs), "status": "OK", "note": note})
        seen_holos.add(pdb_id)

        for lig in ligs:
            lig_xyz_t = transform_atoms(
                [{"xyz": xyz} for xyz in lig["xyz"]], R, t)
            lig_key = (pdb_id, lig["chain"], lig["resnum"], lig["resname"])

            # Build shells on the query structure
            shells = {c: build_shell(query_chains, lig_xyz_t, c) for c in SHELL_CUTOFFS}
            shell_8 = shells[8.0]

            # Score PRISM sites (up to max_rank × 3 candidates, ranked by rank_K)
            ranked_sites = sorted(
                sites.values(),
                key=lambda s: (s["rank_K"] if s["rank_K"] else 9999.0)
            )[:args.max_rank * 3]

            for site in ranked_sites:
                row = score_site_vs_ligand(
                    site, kcc_residues, lig_xyz_t, shells, query_resnum_map)
                row["holo_pdb"]       = pdb_id
                row["holo_note"]      = note
                row["ligand_key"]     = lig_key
                row["ligand_resname"] = lig["resname"]
                row["align_rmsd_A"]   = round(rmsd, 3)
                row["align_seqid"]    = round(seqid, 3)
                # rank field for SR@K (use rank_K already set from site data)
                row["rank"]           = row["rank_K"]
                all_prism_rows.append(row)

            # Score fpocket proposals
            for prop in fp_proposals:
                sc = score_proposal_vs_ligand(prop, query_res_atoms, lig_xyz_t, shell_8)
                if sc is None:
                    continue
                sc["holo_pdb"]       = pdb_id
                sc["ligand_key"]     = lig_key
                sc["ligand_resname"] = lig["resname"]
                sc["align_rmsd_A"]   = round(rmsd, 3)
                sc["align_seqid"]    = round(seqid, 3)
                all_baseline_rows.append(sc)

            # Score p2rank proposals
            for prop in p2_proposals:
                sc = score_proposal_vs_ligand(prop, query_res_atoms, lig_xyz_t, shell_8)
                if sc is None:
                    continue
                sc["holo_pdb"]       = pdb_id
                sc["ligand_key"]     = lig_key
                sc["ligand_resname"] = lig["resname"]
                sc["align_rmsd_A"]   = round(rmsd, 3)
                sc["align_seqid"]    = round(seqid, 3)
                all_baseline_rows.append(sc)

    if not all_prism_rows:
        print("\n[WARN] No PRISM scoring rows generated. "
              "Check holo references and PDB downloads.")

    # ── 5. SR@K ───────────────────────────────────────────────────────────
    print(f"\n[5] Computing SR@K...")
    sr_prism = compute_sr_at_k(all_prism_rows, "PRISM")
    sr_fp    = compute_sr_at_k(
        [r for r in all_baseline_rows if r["method"] == "fpocket"], "fpocket")
    sr_p2    = compute_sr_at_k(
        [r for r in all_baseline_rows if r["method"] == "p2rank"],  "p2rank")

    for label, sr_list in [("PRISM", sr_prism), ("fpocket", sr_fp), ("p2rank", sr_p2)]:
        if sr_list:
            sr1 = next((r["SR@K"] for r in sr_list if r["K"] == 1), 0.0)
            sr5 = next((r["SR@K"] for r in sr_list if r["K"] == 5), 0.0)
            print(f"  {label:8s}  SR@1={sr1:.3f}  SR@5={sr5:.3f}")

    # ── 6. LORO blinded ───────────────────────────────────────────────────
    print(f"\n[6] Running LORO blinded validation...")
    loro_rows = loro_blinded(all_prism_rows, sorted(seen_holos),
                             max_rank=args.max_rank)
    n_blind   = sum(1 for r in loro_rows if r["status"] == "BLINDED_REDISCOVERY")
    print(f"  {len(loro_rows)} LORO tests | {n_blind} blinded rediscoveries")

    # ── 7. Family collapse ────────────────────────────────────────────────
    print(f"\n[7] Collapsing site families...")
    family_rows = collapse_families(all_prism_rows, max_rank=args.max_rank)
    print(f"  {len({r['site_id'] for r in all_prism_rows})} sites "
          f"→ {len(family_rows)} families")

    # ── 8. Write outputs ──────────────────────────────────────────────────
    print(f"\n[8] Writing outputs to {outdir}")

    def prep_for_csv(rows: list[dict]) -> list[dict]:
        """Convert tuple ligand_key to string for CSV serialisability."""
        out = []
        for r in rows:
            rr = dict(r)
            if "ligand_key" in rr and isinstance(rr["ligand_key"], tuple):
                lk = rr.pop("ligand_key")
                rr["ligand_key_str"] = "|".join(str(x) for x in lk)
            out.append(rr)
        return out

    write_csv(outdir / "prism_site_vs_ligand_shells.csv",   prep_for_csv(all_prism_rows))
    write_csv(outdir / "baseline_vs_ligand_shells.csv",     prep_for_csv(all_baseline_rows))
    write_csv(outdir / "sr_at_k_comparison.csv",            sr_prism + sr_fp + sr_p2)
    write_csv(outdir / "prism_family_vs_ligand_shells.csv", family_rows)
    write_csv(outdir / "blinded_loro.csv",                  loro_rows)
    write_csv(outdir / "holo_alignment_summary.csv",        holo_summary)

    write_report(outdir, target, all_prism_rows, all_baseline_rows,
                 loro_rows, family_rows, holo_summary,
                 sr_prism, sr_fp + sr_p2)
    write_pymol(outdir, target, query_pdb, all_prism_rows, loro_rows)

    print(f"\n  prism_site_vs_ligand_shells.csv  ({len(all_prism_rows)} rows)")
    print(f"  baseline_vs_ligand_shells.csv    ({len(all_baseline_rows)} rows)")
    print(f"  sr_at_k_comparison.csv")
    print(f"  prism_family_vs_ligand_shells.csv ({len(family_rows)} rows)")
    print(f"  blinded_loro.csv                 ({len(loro_rows)} rows)")
    print(f"  holo_alignment_summary.csv       ({len(holo_summary)} rows)")
    print(f"  validation_report.md")
    print(f"  pymol_overlay.pml")

    print(f"\n=== DONE: {target} ===")
    print(f"  Outdir: {outdir}")

    # Summary line for quick CI capture
    sr1_prism = next((r["SR@K"] for r in sr_prism if r["K"] == 1), 0.0)
    sr5_prism = next((r["SR@K"] for r in sr_prism if r["K"] == 5), 0.0)
    print(f"  PRISM SR@1={sr1_prism:.3f}  SR@5={sr5_prism:.3f}  "
          f"LORO_blinded={n_blind}/{len(loro_rows)}")

    if sr_fp:
        sr1_fp = next((r["SR@K"] for r in sr_fp if r["K"] == 1), 0.0)
        sr5_fp = next((r["SR@K"] for r in sr_fp if r["K"] == 5), 0.0)
        print(f"  fpocket SR@1={sr1_fp:.3f}  SR@5={sr5_fp:.3f}")
    if sr_p2:
        sr1_p2 = next((r["SR@K"] for r in sr_p2 if r["K"] == 1), 0.0)
        sr5_p2 = next((r["SR@K"] for r in sr_p2 if r["K"] == 5), 0.0)
        print(f"  p2rank  SR@1={sr1_p2:.3f}  SR@5={sr5_p2:.3f}")


if __name__ == "__main__":
    main()
