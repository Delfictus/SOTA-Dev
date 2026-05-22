#!/usr/bin/env python3
"""
prism_manifold_shell_validator.py
==================================
Publication-grade manifold-to-ligand-shell recovery validator for PRISM-4D.

Validates by manifold-to-ligand-shell recovery, NOT centroid distance alone.

Pipeline:
  1. Load PRISM sites from run dir (binding_sites.json + kcc_visualization.json + Arrow)
  2. Extract per-site: KCC driver residues, aromatic driver residues, high-kcc residues
  3. For each holo reference: download PDB, align backbone to query, transform ligand
  4. Build multi-shell residue sets (4/6/8 Å) using all-atom or CA distances
  5. Score every PRISM site: min_dist, centroid_dist, Jaccard@4/6/8, causal/KCC/aromatic
     driver in shell, phase-specific hit
  6. Collapse sites into manifold families via centroid proximity + lining overlap +
     KCC driver overlap
  7. Blinded leave-one-reference-out (LORO) validation
  8. Baseline scoring: fpocket / P2Rank on same query PDB vs same shells
  9. SR@K comparison, validation_report.md, pymol_overlay.pml

Usage:
  python prism_manifold_shell_validator.py \\
    --run-dir /mnt/storage/prism-outputs/runs/KRAS_G12C_chainA_20260512_194818 \\
    --query-pdb /path/to/KRAS_G12C_chainA_clean.pdb \\
    --target KRAS_G12C \\
    --outdir /tmp/pub_validation/KRAS_G12C
"""
from __future__ import annotations
import argparse, collections, csv, json, math, os, re, subprocess, sys, tempfile
import urllib.request, urllib.error
from pathlib import Path
from typing import Optional

import numpy as np

# ── Optional Arrow dependency ─────────────────────────────────────────────────
try:
    import pyarrow.ipc as _ipc
    import pyarrow.compute as _pc
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

# ── Holo reference table ──────────────────────────────────────────────────────
# Each entry: list of (pdb_id, preferred_chain, annotation)
# preferred_chain=None → use best-aligning chain via NW
# Ligands auto-detected from HETATM (>5 heavy atoms, not solvent/buffer)
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
        # 7T8I / 8BEL: alignment quality <0.50 seqid — skipped by gate
        {"pdb": "7T8I", "chain": "A", "note": "KRAS G12C + adagrasib MRTX849"},
        {"pdb": "5V9L", "chain": "A", "note": "KRAS G12C apo + GDP (pocket reference)"},
    ],
    "p53_Y220C": [
        {"pdb": "6GGE", "chain": "A", "note": "p53 Y220C + EYE stabilizer (0.16Å reference)"},
        {"pdb": "6SI3", "chain": "A", "note": "p53 Y220C + PK083 triazole stabilizer"},
        {"pdb": "7ATA", "chain": "A", "note": "p53 Y220C + aminobenzimidazole stabilizer"},
        # 7O70/6TP6: seqid<0.50 in tested alignment — skipped by gate
        {"pdb": "7O70", "chain": "A", "note": "p53 + iminoquinol (may need chain B)"},
        {"pdb": "6TP6", "chain": "A", "note": "p53 Y220C + fragment series"},
    ],
    "TEAD3": [
        # 6CDY: TEAD1 seqid=0.71 RMSD=1.48Å — only validated reference
        {"pdb": "6CDY", "chain": "B",  "note": "TEAD1 + EY1 palmitoyl flap-pocket (chain B, 207 CA)"},
        {"pdb": "5OAQ", "chain": "A",  "note": "TEAD2 + lipid-mimetic compound (flap)"},
        {"pdb": "6GE0", "chain": "A",  "note": "TEAD1/2 + VGX-1 YAP-interface allosteric"},
        # 5GQM / 6TUG: wrong protein (seqid=0.21) — kept to confirm gate works
        {"pdb": "5GQM", "chain": "A", "note": "TEAD2 + nucleotides (prob. artifact; gate check)"},
    ],
    "GLP1R": [
        # 7LCI chain A = GLP-1R (244 CA, 11-394) — same construct as query
        {"pdb": "7LCI", "chain": "A",  "note": "GLP-1R active state (UK4 GLP-1 peptide via chain R)"},
        # 6VCB chain A = GLP-1R (223 CA, 9-394); QW7=TT-OAD2 non-peptide agonist in chain R
        {"pdb": "6VCB", "chain": "A",  "note": "GLP-1R + TT-OAD2 non-peptide agonist (TMD)"},
        # 5NX2 chain A = Gαs (wrong protein); only useful if alignment gate rejects it
        {"pdb": "5NX2", "chain": "A",  "note": "Gαs structure (expect gate rejection, seqid check)"},
    ],
    "STING": [
        # Verified: 4KSY/4LOH seqid ~0.63 with our STING query (CDN-pocket CTD constructs)
        {"pdb": "4KSY", "chain": "A", "note": "hSTING CTD + c-di-AMP CDN pocket (1SY)"},
        {"pdb": "4LOH", "chain": "A", "note": "hSTING CTD + c-di-GMP CDN pocket (1SY)"},
        {"pdb": "5CFQ", "chain": "A", "note": "hSTING CTD + DMXAA analog (1SY)"},
        # 7BIQ / 7JU5: alignment quality too low for chain A — skipped by gate
        {"pdb": "7BIQ", "chain": "A", "note": "hSTING + SR-717 TVB (gate check — may fail)"},
    ],
    "AKT1": [
        # 5KCV: full PH+kinase construct (6-438), 6S1 = covalent allosteric inhibitor (33 ha)
        {"pdb": "5KCV", "chain": "A", "note": "AKT1 PH+kinase + 6S1 covalent allosteric"},
        # 4EKK: kinase domain (144-477), ANP = AMPPNP ATP mimic (31 ha)
        {"pdb": "4EKK", "chain": "A", "note": "AKT1 kinase + ANP (AMPPNP ATP-site mimic)"},
        # 3CQW: kinase domain (144-478), CQW = ATP-competitive inhibitor (19 ha)
        {"pdb": "3CQW", "chain": "A", "note": "AKT1 kinase + CQW ATP-competitive inhibitor"},
        # 4GV1: AKT2 allosteric (143-477), 0XZ = allosteric inhibitor (30 ha)
        {"pdb": "4GV1", "chain": "A", "note": "AKT2 kinase + 0XZ allosteric (isoform ref)"},
    ],
    "Kv31": [
        # 7PHH: exact Kv3.1 source structure (chain A 7-464, 396 CA); PCF = phosphatidylcholine
        # at pore-facing fenestration sites. Self-alignment RMSD=0.00.
        {"pdb": "7PHH", "chain": "A", "note": "Kv3.1 (7PHH) + PCF lipid at pore fenestration"},
        # 6Y7Y: Kv channel family member + SB4 blocker (25 ha) — gating pocket reference
        {"pdb": "6Y7Y", "chain": "A", "note": "Kv channel + SB4 blocker (gating/pore comparison)"},
    ],
    "TRPV1": [
        # 5IS0: TRPV1 + 6ET (25 ha) at vanilloid binding site; chains B-E, 335-751
        {"pdb": "5IS0", "chain": None, "note": "TRPV1 + 6ET capsaicin-site compound (vanilloid TM4/5)"},
        # 5IRZ: TRPV1 + 6O8/6ES/6OE (27-38 ha) at vanilloid pocket; chains B-E, 335-751
        {"pdb": "5IRZ", "chain": None, "note": "TRPV1 + 6O8/6ES/6OE RTX-site compounds (vanilloid)"},
    ],
}

SOLVENT_EXCL = {
    "HOH","WAT","DOD","SO4","PO4","GOL","EDO","PEG","ACT","FMT","MPD","BME",
    "ACE","NH2","NAG","FUC","DMS","IOD","CL","NA","MG","ZN","CA","K","MN","FE",
    "NI","CU","CO","CD","HG","SEP","TPO","PTR","ALY","MLY","MSE","UNK","CSO",
    "CME","OCS","KCX","LLP","PCA","HYP","DPN","NEP","SNN","4HY","2MR","EPE",
    "MES","TRS","HEZ","BTB","P6G","TAM","BOG","LMT","DIO","NDG","BMA",
}
MIN_LIGAND_HEAVY = 6
SHELL_CUTOFFS = (4.0, 6.0, 8.0)
KCC_HIGH_SCORE_THRESH = 0.55  # residues above this are "high-KCC"
FAMILY_CENTROID_DIST   = 10.0  # Å — max centroid distance to merge sites into family
FAMILY_LINING_JACCARD  = 0.12  # min lining Jaccard to merge
FAMILY_KCC_SHARED      = 1     # min shared KCC driver residues to merge


# ── Geometry ──────────────────────────────────────────────────────────────────
def dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def centroid(pts) -> list[float]:
    n = len(pts)
    return [sum(p[i] for p in pts) / n for i in range(3)]

def kabsch(P: np.ndarray, Q: np.ndarray):
    """Optimal rotation R, translation t: maps P → Q. Returns R, t, rmsd."""
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
    """Global NW alignment. Returns (score, aligned_a, aligned_b)."""
    MATCH, MISMATCH, GAP = 2, -1, -2
    n, m = len(a), len(b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0] = dp[i-1][0] + GAP
    for j in range(1, m + 1): dp[0][j] = dp[0][j-1] + GAP
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

AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "HID":"H","HIE":"H","HIP":"H","CYX":"C","MSE":"M","SEP":"S","TPO":"T",
    "PTR":"Y","HYP":"P","CSO":"C","CME":"C",
}

# ── PDB parsing ───────────────────────────────────────────────────────────────
def parse_pdb(path: Path) -> dict:
    """Returns {'atoms': [...], 'chains': {chain_id: {'seq': str, 'ca': [...]}}}."""
    atoms, hetatm = [], []
    chain_res: dict[str, dict] = collections.defaultdict(dict)  # chain → {resnum: (resname, {name: xyz})}

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
                if chain not in chain_res:
                    chain_res[chain] = {}
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
    """Group HETATM atoms into ligand instances (>= MIN_LIGAND_HEAVY heavy atoms)."""
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

def best_align_chains(query_chains: dict, holo_chains: dict,
                      prefer_chain: Optional[str] = None
                      ) -> tuple[str, str, np.ndarray, np.ndarray, float, float]:
    """
    Find best (query_chain, holo_chain) pair by NW sequence identity.
    Returns (q_chain, h_chain, R, t, rmsd, seq_identity).
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
            matches = sum(1 for a, b in zip(qa, ha) if a == b and a != "-")
            identity = matches / max(len(qdata["seq"]), len(hdata["seq"]))
            if best is None or identity > best[0]:
                best = (identity, qc, hc, qdata["ca"], hdata["ca"])
    if best is None:
        return None, None, None, None, 999.0, 0.0

    identity, qc, hc, qca, hca = best
    # Align positions where both chains have CA
    _, qa_aln, ha_aln = needleman(
        "".join(d["xyz"] and AA1.get(d["resname"], "X") for d in qca),
        "".join(d["xyz"] and AA1.get(d["resname"], "X") for d in hca),
    )
    # Pair aligned residues
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
    """Apply (R, t) transform to a list of xyz tuples."""
    result = []
    for a in atoms:
        v = np.array(a["xyz"]) if isinstance(a, dict) else np.array(a)
        tv = (R @ v) + t
        result.append(tuple(tv.tolist()))
    return result

def build_shell(query_chains: dict, lig_xyz_transformed: list[tuple],
                cutoff: float) -> set[int]:
    """
    Return set of query residue numbers within `cutoff` Å of any ligand atom.
    Uses all available atoms per residue, falls back to CA.
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
    key = pdb_id.upper()
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
    Load sites from binding_sites.json + kcc_visualization.json + Arrow.
    Returns dict keyed by site_id → site info.
    """
    bs_files = sorted(run_dir.glob("*.binding_sites.json"))
    kcc_files = sorted(run_dir.glob("*.kcc_visualization.json"))
    arrow_files = sorted(run_dir.glob("*.topology.spike_events.arrow"))
    therm_files = sorted(run_dir.glob("*.topology.prism_therm.json"))

    if not bs_files:
        raise FileNotFoundError(f"No binding_sites.json in {run_dir}")

    bs_data  = json.loads(bs_files[0].read_text())
    kcc_data = json.loads(kcc_files[0].read_text()) if kcc_files else {}

    # ── Build per-residue data from kcc_visualization ─────────────────────
    # kcc_residues keyed by kcc residue_id (PDB-author numbering).
    # binding_sites lining.resid and Arrow residue IDs use 0-based sequential.
    # seq_to_pdb: 0-based sequential index → PDB-author residue_id (translation key).
    kcc_residues: dict[int, dict] = {}  # pdb_author_resid → data
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
    # 0-based sequential → PDB-author ID (for translating lining/Arrow IDs)
    seq_to_pdb: list[int] = [r["residue_id"] for r in _kcc_raw]

    # ── kcc_viz site-level: driver_residue_id, candidate_residue_ids ──────
    kcc_site_drivers: dict[int, dict] = {}
    for s in kcc_data.get("sites", []):
        sid  = s.get("id", -1)
        kcc  = s.get("kcc") or {}
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

    # ── High-KCC residues set ─────────────────────────────────────────────
    high_kcc_resids = {
        rid for rid, rd in kcc_residues.items()
        if rd["kcc_score"] >= KCC_HIGH_SCORE_THRESH
    }

    # ── Arrow: per-site aromatic driver residues (top by count) ───────────
    arrow_drivers: dict[int, set[int]] = collections.defaultdict(set)  # site_id → aromatic driver resids
    if HAS_ARROW and arrow_files:
        try:
            table = _ipc.open_file(arrow_files[0]).read_all()
            # Use aromatic_residue_id (non-(-1)) as aromatic driver signal
            site_col  = table.column("site_id").to_pylist()
            arid_col  = table.column("aromatic_residue_id").to_pylist()
            ipct_col  = table.column("intensity_percentile").to_pylist()
            nr_col    = table.column("nearby_residues").to_pylist()

            # Per-site: collect aromatic driver IDs and high-intensity nearby residues
            site_arid_cnt:  dict[int, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
            site_nr_cnt:    dict[int, dict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
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

    # ── Assemble sites dict ───────────────────────────────────────────────
    sites: dict[int, dict] = {}
    for s in bs_data.get("sites", []):
        sid = s.get("id", -1)
        if sid < 0:
            continue
        lining = s.get("lining_residues", s.get("lining", []))
        # lining.resid is 0-based sequential; translate to PDB-author IDs
        lining_resids = [
            seq_to_pdb[r["resid"]] if 0 <= r["resid"] < len(seq_to_pdb) else r["resid"]
            for r in lining if isinstance(r, dict)
        ]
        lining_resnames = {
            (seq_to_pdb[r["resid"]] if 0 <= r["resid"] < len(seq_to_pdb) else r["resid"]): r.get("resname", "?")
            for r in lining if isinstance(r, dict)
        }

        # Get CA positions for lining residues from kcc_viz (now in PDB-author space)
        lining_ca = []
        for rid in lining_resids:
            if rid in kcc_residues:
                lining_ca.append(kcc_residues[rid]["ca_xyz"])

        kcc_info = kcc_site_drivers.get(sid, {})
        driver_rid = kcc_info.get("driver_residue_id", -1)
        candidates = set(kcc_info.get("candidate_residue_ids", []))

        # Use kcc_viz centroid if available, else from binding_sites
        kcc_cent = kcc_info.get("centroid")
        bs_cent = s.get("centroid", s.get("centroid_xyz", [0, 0, 0]))
        cent = kcc_cent if kcc_cent else bs_cent

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
        "sites":            sites,
        "kcc_residues":     kcc_residues,
        "high_kcc_resids":  high_kcc_resids,
        "stem":             bs_files[0].stem.replace(".binding_sites", ""),
    }

# ── Per-site scoring against one holo ligand ─────────────────────────────────
def score_site_vs_ligand(site: dict, kcc_residues: dict,
                         lig_xyz_t: list[tuple],
                         shells: dict[float, set[int]],
                         query_resnum_map: dict[int, int]) -> dict:
    """
    Score one PRISM site against a transformed holo ligand.
    query_resnum_map: topology_resid → PDB resnum in query (for shell lookup).
    """
    # Map site lining resids to query resnums for shell membership check
    lining_q = {query_resnum_map.get(r, r) for r in site["lining_resids"]}
    site_cent = site["centroid"]

    # min atom distance: min(dist(lining_CA, lig_atom)) for each lining residue CA
    min_dist = 999.0
    if site["lining_ca"]:
        for ca in site["lining_ca"]:
            for lpt in lig_xyz_t:
                d = dist(ca, lpt)
                if d < min_dist:
                    min_dist = d
    # centroid to ligand distance
    lig_cent = centroid(lig_xyz_t) if lig_xyz_t else [0,0,0]
    cent_dist = dist(site_cent, lig_cent)

    # Jaccard at each cutoff
    jaccard, intersection_cnt = {}, {}
    for c in SHELL_CUTOFFS:
        sh = shells[c]
        # map shell from PDB resnum back to topology space via inverse map
        # (shell is in PDB resnum space; lining_q is in PDB resnum space)
        inter = lining_q & sh
        union = lining_q | sh
        jaccard[c]         = len(inter) / max(1, len(union))
        intersection_cnt[c] = len(inter)

    # Causal/driver features
    shell_8 = shells[8.0]
    kcc_dr_q = query_resnum_map.get(site["kcc_driver_resid"], site["kcc_driver_resid"])
    kcc_cand_q = {query_resnum_map.get(r, r) for r in site["kcc_candidates"]}
    arrow_q   = {query_resnum_map.get(r, r) for r in site["arrow_driver_resids"]}

    kcc_driver_in_shell   = kcc_dr_q in shell_8 and kcc_dr_q >= 0
    kcc_cand_in_shell     = len(kcc_cand_q & shell_8)
    arrow_driver_in_shell = len(arrow_q & shell_8) > 0

    # High-KCC residues in 8Å shell (using topology IDs mapped to PDB resnums)
    high_kcc_in_8a_shell = 0
    for rid, rd in kcc_residues.items():
        if rd["kcc_score"] >= KCC_HIGH_SCORE_THRESH:
            qr = query_resnum_map.get(rid, rid)
            if qr in shell_8:
                high_kcc_in_8a_shell += 1

    # Verdict
    good     = min_dist <= 4.0 or jaccard[8.0] >= 0.20
    strong   = min_dist <= 3.0 and kcc_driver_in_shell
    # excellent set after LORO (placeholder here)

    if strong:
        verdict = "STRONG"
    elif min_dist <= 4.0 and jaccard[8.0] >= 0.20:
        verdict = "GOOD_BOTH"
    elif min_dist <= 4.0:
        verdict = "GOOD_DIST"
    elif jaccard[8.0] >= 0.20:
        verdict = "GOOD_JACCARD"
    elif min_dist <= 6.0 and intersection_cnt[8.0] >= 2:
        verdict = "NEAR"
    elif min_dist <= 8.0:
        verdict = "WEAK"
    else:
        verdict = "MISS"

    def is_good(v): return v in ("STRONG", "GOOD_BOTH", "GOOD_DIST", "GOOD_JACCARD")

    return {
        "site_id":                 site["id"],
        "therm_class":             site["therm_class"],
        "quality_score":           site["quality_score"],
        "hysteresis_asymmetry":    site["hysteresis_asymmetry"],
        "kcc_confidence":          site["kcc_confidence"],
        "burst_motion":            site["burst_motion"],
        "rank_K":                  site["rank_K"],
        "rank_T":                  site["rank_T"],
        "lining_size":             len(site["lining_resids"]),
        "min_lining_dist_A":       round(min_dist, 3),
        "centroid_to_ligand_A":    round(cent_dist, 3),
        "intersection_4A":         intersection_cnt[4.0],
        "intersection_6A":         intersection_cnt[6.0],
        "intersection_8A":         intersection_cnt[8.0],
        "jaccard_4A":              round(jaccard[4.0], 4),
        "jaccard_6A":              round(jaccard[6.0], 4),
        "jaccard_8A":              round(jaccard[8.0], 4),
        "kcc_driver_in_8A_shell":  kcc_driver_in_shell,
        "kcc_candidates_in_8A":    kcc_cand_in_shell,
        "arrow_driver_in_8A":      arrow_driver_in_shell,
        "high_kcc_resids_in_8A":   high_kcc_in_8a_shell,
        "verdict":                 verdict,
        "is_good":                 is_good(verdict),
    }

# ── Baseline: fpocket ─────────────────────────────────────────────────────────
def run_fpocket(query_pdb: Path, outdir: Path) -> list[dict]:
    """Run fpocket and return list of pocket proposals as {rank, xyz_list}."""
    import shutil as _shutil
    if not query_pdb or not query_pdb.exists():
        return []
    # fpocket always creates <name>_out/ next to the input file, so copy to outdir first
    local_pdb = outdir / query_pdb.name
    _shutil.copy2(query_pdb, local_pdb)
    try:
        r = subprocess.run(["fpocket", "-f", str(local_pdb)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    proposals = []
    for pf in sorted(outdir.glob("*_out/pockets/pocket*_atm.pdb")):
        m = re.search(r"pocket(\d+)", pf.name)
        rank = int(m.group(1)) if m else 999
        pts = []
        with open(pf) as fh:
            for line in fh:
                if line[:6] in ("ATOM  ", "HETATM"):
                    try:
                        pts.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                    except ValueError:
                        pass
        if pts:
            proposals.append({"method": "fpocket", "rank": rank, "xyz": pts,
                               "centroid": centroid(pts)})
    return sorted(proposals, key=lambda x: x["rank"])

def run_p2rank(query_pdb: Path, outdir: Path) -> list[dict]:
    """Run P2Rank and return pocket proposals."""
    if not query_pdb or not query_pdb.exists():
        return []
    p2out = outdir / "p2rank_out"
    p2out.mkdir(exist_ok=True)
    try:
        r = subprocess.run(
            ["/opt/p2rank/prank", "predict", "-f", str(query_pdb), "-o", str(p2out), "-threads", "4"],
            capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    pred_file = next(p2out.rglob("*_predictions.csv"), None)
    if not pred_file:
        return []
    proposals = []
    with open(pred_file) as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            try:
                cx = float(row.get("center_x", row.get(" center_x", 0)))
                cy = float(row.get("center_y", row.get(" center_y", 0)))
                cz = float(row.get("center_z", row.get(" center_z", 0)))
                proposals.append({"method": "p2rank", "rank": i + 1,
                                   "xyz": [(cx, cy, cz)], "centroid": [cx, cy, cz]})
            except (ValueError, KeyError):
                pass
    return proposals

def score_baseline_vs_ligand(proposal: dict, query_chains: dict,
                              lig_xyz_t: list[tuple], shells: dict) -> dict:
    """Score a baseline pocket proposal against a transformed ligand shell."""
    pts = proposal["xyz"]
    min_dist = min((dist(p, lpt) for p in pts for lpt in lig_xyz_t), default=999.0)
    # build 'near residues' for this proposal
    near_res = set()
    for chain, cdata in query_chains.items():
        for resnum, (_, atom_dict) in cdata["res"]:
            for p in pts:
                if any(dist(list(av), p) <= 4.0 for av in atom_dict.values()):
                    near_res.add(resnum)
                    break
    sh8 = shells[8.0]
    inter = near_res & sh8
    union = near_res | sh8
    jaccard_8 = len(inter) / max(1, len(union))
    good = min_dist <= 4.0 or jaccard_8 >= 0.20
    verdict = "GOOD" if min_dist <= 4.0 and jaccard_8 >= 0.10 else \
              "NEAR" if min_dist <= 8.0 else "MISS"
    return {
        "method":           proposal["method"],
        "rank":             proposal["rank"],
        "min_dist_A":       round(min_dist, 3),
        "intersection_8A":  len(inter),
        "jaccard_8A":       round(jaccard_8, 4),
        "verdict":          verdict,
        "is_good":          good,
    }

# ── SR@K ──────────────────────────────────────────────────────────────────────
def compute_sr_at_k(rows: list[dict], key_field: str = "ligand_key",
                    rank_field: str = "site_rank",
                    good_field: str = "is_good",
                    ks: tuple = (1, 3, 5, 10)) -> dict:
    """SR@K: fraction of unique ligands with ≥1 good hit in top-K ranked sites."""
    by_lig: dict = collections.defaultdict(list)
    for r in rows:
        by_lig[r[key_field]].append(r)
    result = {}
    total = len(by_lig)
    for k in ks:
        hits = sum(
            1 for lig_rows in by_lig.values()
            if any(r[good_field] for r in sorted(lig_rows, key=lambda x: x[rank_field])[:k])
        )
        result[f"SR@{k}"] = round(hits / max(1, total), 4)
    result["n_ligands"] = total
    return result

# ── Site family collapse ───────────────────────────────────────────────────────
def collapse_families(sites: dict) -> list[set]:
    """
    Union-find: merge sites sharing centroid proximity AND lining overlap
    AND KCC driver overlap.
    """
    sids = list(sites.keys())
    parent = {s: s for s in sids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i, si in enumerate(sids):
        for sj in sids[i+1:]:
            a, b = sites[si], sites[sj]
            cd = dist(a["centroid"], b["centroid"])
            if cd > FAMILY_CENTROID_DIST:
                continue
            li = a["lining_resids"] & b["lining_resids"]
            lu = a["lining_resids"] | b["lining_resids"]
            j  = len(li) / max(1, len(lu))
            shared_kcc = len(
                (a["kcc_candidates"] | {a["kcc_driver_resid"]}) &
                (b["kcc_candidates"] | {b["kcc_driver_resid"]}) -
                {-1}
            )
            if j >= FAMILY_LINING_JACCARD or shared_kcc >= FAMILY_KCC_SHARED:
                union(si, sj)

    families: dict = collections.defaultdict(set)
    for s in sids:
        families[find(s)].add(s)
    return list(families.values())

# ── LORO blinded validation ────────────────────────────────────────────────────
def loro_validation(all_rows: list[dict]) -> list[dict]:
    """
    Leave-one-reference-out: for each held-out holo, determine which
    trained site families (trained on all OTHER holos) recover it.
    """
    ref_ids = sorted({r["holo_pdb"] for r in all_rows})
    results = []
    for held_ref in ref_ids:
        train_rows = [r for r in all_rows if r["holo_pdb"] != held_ref and r["is_good"]]
        test_rows  = [r for r in all_rows if r["holo_pdb"] == held_ref]

        trained_sites = {r["site_id"] for r in train_rows}
        for lig_key in sorted({r["ligand_key"] for r in test_rows}):
            lig_test = [r for r in test_rows if r["ligand_key"] == lig_key]
            trained_hits  = [r for r in lig_test if r["site_id"] in trained_sites and r["is_good"]]
            any_good      = [r for r in lig_test if r["is_good"]]
            best_rank_all = min((r["site_rank"] for r in lig_test), default=9999)
            best_rank_trained = min((r["site_rank"] for r in trained_hits), default=9999) \
                                if trained_hits else 9999

            if trained_hits:
                status = "BLINDED_REDISCOVERY" if best_rank_trained <= 3 else \
                         "BLINDED_GOOD" if best_rank_trained <= 10 else \
                         "TRAINED_SITE_WEAK"
            elif any_good:
                status = "NEW_SITE_HIT"
            else:
                status = "FAIL"

            results.append({
                "withheld_ref":         held_ref,
                "ligand_key":           lig_key,
                "n_trained_refs":       len(ref_ids) - 1,
                "trained_sites":        len(trained_sites),
                "best_rank_any":        best_rank_all,
                "best_rank_trained":    best_rank_trained,
                "loro_status":          status,
                "excellent":            status == "BLINDED_REDISCOVERY",
            })
    return results

# ── Output writers ─────────────────────────────────────────────────────────────
def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def write_pymol(path: Path, query_pdb: Optional[Path], sites: dict,
                holo_summaries: list[dict], target: str):
    lines = [
        f"# PRISM-4D manifold-shell validation — {target}",
        f"# Generated by prism_manifold_shell_validator.py",
        "",
    ]
    if query_pdb and query_pdb.exists():
        lines.append(f'load {query_pdb}, query')
        lines.append("hide everything, query")
        lines.append("show cartoon, query")
        lines.append("color grey80, query")
        lines.append("")
    for h in holo_summaries:
        if h.get("pdb_path"):
            lines.append(f'load {h["pdb_path"]}, holo_{h["pdb_id"]}')
            lines.append(f'hide everything, holo_{h["pdb_id"]}')
            lines.append(f'show lines, holo_{h["pdb_id"]} and organic')
    lines.append("")
    colors = ["red","orange","yellow","green","cyan","blue","purple","magenta"]
    for i, (sid, site) in enumerate(sorted(sites.items(), key=lambda x: -x[1]["quality_score"])[:15]):
        c = colors[i % len(colors)]
        cx, cy, cz = site["centroid"]
        tc = site["therm_class"]
        lines.append(f"# Site {sid} [{tc}] q={site['quality_score']:.3f}")
        lines.append(f'pseudoatom site_{sid}, pos=[{cx:.2f},{cy:.2f},{cz:.2f}]')
        lines.append(f'color {c}, site_{sid}')
        lines.append(f'show sphere, site_{sid}')
        lines.append(f'set sphere_scale, 1.5, site_{sid}')
    lines += ["", "zoom query", ""]
    path.write_text("\n".join(lines))

def write_report(path: Path, target: str, sites: dict, all_rows: list[dict],
                 families: list[set], loro_rows: list[dict],
                 sr_prism: dict, sr_fpocket: dict, sr_p2rank: dict,
                 holo_summaries: list[dict]):
    n_total    = len(sites)
    n_cryptic  = sum(1 for s in sites.values() if s["therm_class"] == "CRYPTIC")
    n_responsive = sum(1 for s in sites.values() if s["therm_class"] == "RESPONSIVE")
    good_rows  = [r for r in all_rows if r["is_good"]]
    n_holos    = len({r["holo_pdb"] for r in all_rows})
    n_lig_inst = len({r["ligand_key"] for r in all_rows})
    n_excellents = sum(1 for r in loro_rows if r["excellent"])
    n_blinded  = sum(1 for r in loro_rows if "BLINDED" in r.get("loro_status", ""))

    lines = [
        f"# PRISM-4D Manifold-to-Ligand-Shell Validation Report",
        f"## Target: {target}",
        f"",
        f"**Sites**: {n_total}  "
        f"(CRYPTIC:{n_cryptic}  RESPONSIVE:{n_responsive})",
        f"**Holo references**: {n_holos}  **Ligand instances**: {n_lig_inst}",
        f"**Families collapsed**: {len(families)}",
        f"",
        f"### SR@K — PRISM vs baselines",
        f"",
        f"| Method | SR@1 | SR@3 | SR@5 | SR@10 | n_ligands |",
        f"|--------|------|------|------|-------|-----------|",
    ]
    for label, sr in [("PRISM", sr_prism), ("fpocket", sr_fpocket), ("P2Rank", sr_p2rank)]:
        if sr:
            lines.append(
                f"| {label:8s} | {sr.get('SR@1',0):.3f} | {sr.get('SR@3',0):.3f} | "
                f"{sr.get('SR@5',0):.3f} | {sr.get('SR@10',0):.3f} | "
                f"{sr.get('n_ligands','—')} |"
            )
    lines += [
        f"",
        f"### Blinded LORO Summary",
        f"",
        f"- Blinded rediscoveries (rank ≤ 3): **{n_excellents}** / {len(loro_rows)}",
        f"- Blinded good (rank ≤ 10): **{n_blinded}**",
        f"",
        f"### Holo Alignment Quality",
        f"",
        f"| PDB | RMSD (Å) | Seq identity | Ligands | Best PRISM site | Best verdict |",
        f"|-----|----------|--------------|---------|-----------------|--------------|",
    ]
    for h in holo_summaries:
        best = min(
            (r for r in all_rows if r["holo_pdb"] == h["pdb_id"]),
            key=lambda r: r["min_lining_dist_A"], default=None
        )
        best_site    = best["site_id"] if best else "—"
        best_verdict = best["verdict"] if best else "—"
        lines.append(
            f"| {h['pdb_id']} | {h.get('rmsd_A',999):.2f} | "
            f"{h.get('seq_identity',0):.2f} | {h.get('n_ligands',0)} | "
            f"{best_site} | {best_verdict} |"
        )
    lines += [
        f"",
        f"### Top CRYPTIC Sites",
        f"",
        f"| Site | Therm | Quality | Asym | KCC Conf | Burst | Best min dist (Å) | Best J@8 | Best verdict |",
        f"|------|-------|---------|------|----------|-------|-------------------|----------|--------------|",
    ]
    cryptic_sites = sorted(
        [s for s in sites.values() if s["therm_class"] in ("CRYPTIC", "RESPONSIVE")],
        key=lambda s: -s["quality_score"]
    )[:10]
    for s in cryptic_sites:
        site_rows = [r for r in all_rows if r["site_id"] == s["id"]]
        bd = min((r["min_lining_dist_A"] for r in site_rows), default=999)
        bj = max((r["jaccard_8A"] for r in site_rows), default=0)
        bv = min(site_rows, key=lambda r: r["min_lining_dist_A"], default={}).get("verdict", "—")
        lines.append(
            f"| {s['id']} | {s['therm_class']} | {s['quality_score']:.3f} | "
            f"{s['hysteresis_asymmetry']:.3f} | {s['kcc_confidence']:.3f} | "
            f"{s['burst_motion']:.3f} | {bd:.2f} | {bj:.3f} | {bv} |"
        )
    path.write_text("\n".join(lines) + "\n")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="PRISM manifold-to-ligand-shell validator")
    ap.add_argument("--run-dir",   required=True, help="PRISM run output directory")
    ap.add_argument("--query-pdb", required=True, help="Clean query PDB (prism-clean output)")
    ap.add_argument("--target",    required=True, help="Target name (must match TARGET_HOLOS key)")
    ap.add_argument("--outdir",    required=True, help="Output directory")
    ap.add_argument("--refs",      nargs="+",     help="Override holo PDB IDs")
    ap.add_argument("--fpocket-dir", default=None)
    ap.add_argument("--p2rank-dir",  default=None)
    ap.add_argument("--max-rank",  type=int, default=20, help="Max site rank to consider")
    ap.add_argument("--topology-offset", type=int, default=0,
                    help="topology_resid + offset = query PDB resnum (default 0)")
    args = ap.parse_args()

    run_dir   = Path(args.run_dir)
    query_pdb = Path(args.query_pdb)
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = outdir / ".pdb_cache"
    cache_dir.mkdir(exist_ok=True)

    target = args.target
    if target not in TARGET_HOLOS and not args.refs:
        print(f"[WARN] {target} not in TARGET_HOLOS and no --refs given.")
        holo_specs = []
    else:
        holo_specs = TARGET_HOLOS.get(target, [])
        if args.refs:
            holo_specs = [{"pdb": p, "chain": None, "note": "user-supplied"} for p in args.refs]

    print(f"\n=== PRISM Manifold-Shell Validator: {target} ===")
    print(f"  run_dir:   {run_dir}")
    print(f"  query_pdb: {query_pdb}")
    print(f"  outdir:    {outdir}")

    # ── 1. Load PRISM data ────────────────────────────────────────────────
    print("\n[1] Loading PRISM run data...")
    prism = load_prism_run(run_dir)
    sites = prism["sites"]
    kcc_residues = prism["kcc_residues"]
    print(f"  Sites: {len(sites)}  KCC residues: {len(kcc_residues)}")

    # ── 2. Parse query PDB ────────────────────────────────────────────────
    print("\n[2] Parsing query PDB...")
    query_parsed = parse_pdb(query_pdb) if query_pdb.exists() else {"chains": {}, "hetatm": []}
    query_chains = query_parsed["chains"]

    # Build topology_resid → PDB_resnum map
    # Strategy: if kcc_viz CA positions ≈ query PDB CA positions, offsets match.
    # Default: kcc_viz resid N → query PDB resnum N + topology_offset
    topo_offset = args.topology_offset
    # Try to auto-detect offset by comparing kcc_viz CA positions to query PDB CA
    if kcc_residues and query_chains:
        qc = next(iter(query_chains.values()))
        qca_by_rnum = {d["resnum"]: np.array(d["xyz"]) for d in qc["ca"]}
        # find offset that minimizes CA distance for first 5 shared residues
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
        print(f"  Auto-detected topology offset: {topo_offset} (mean CA err={best_err:.2f}Å)")

    query_resnum_map = {rid: rid + topo_offset for rid in kcc_residues}

    # ── 3. Process each holo reference ───────────────────────────────────
    print(f"\n[3] Processing {len(holo_specs)} holo references...")
    all_score_rows:    list[dict] = []
    all_baseline_fp:   list[dict] = []
    all_baseline_p2:   list[dict] = []
    holo_summaries:    list[dict] = []

    for spec in holo_specs:
        pdb_id   = spec["pdb"]
        pref_ch  = spec.get("chain")
        note     = spec.get("note", "")
        print(f"\n  Holo: {pdb_id}  ({note})")

        holo_path = fetch_pdb(pdb_id, cache_dir)
        if holo_path is None:
            print(f"    [SKIP] download failed")
            continue

        holo_parsed = parse_pdb(holo_path)
        ligs = ligand_groups(holo_parsed["hetatm"])
        if not ligs:
            print(f"    [SKIP] no ligands found (heavy atoms ≥ {MIN_LIGAND_HEAVY})")
            continue

        print(f"    Ligands: {[(l['chain'], l['resnum'], l['resname']) for l in ligs]}")

        # Align holo → query
        if not query_chains:
            print("    [WARN] no query chains — skipping alignment, using raw holo coords")
            R, t, rmsd, seqid = None, None, 999.0, 0.0
        else:
            qc, hc, R, t, rmsd, seqid = best_align_chains(
                query_chains, holo_parsed["chains"], pref_ch)
            if R is None:
                print(f"    [SKIP] alignment failed (seq identity={seqid:.2f})")
                continue
            print(f"    Aligned {hc}→{qc}  RMSD={rmsd:.2f}Å  seqid={seqid:.2f}")
            if rmsd > 5.0 or seqid < 0.50:
                print(f"    [SKIP] alignment quality too low (RMSD={rmsd:.2f}Å, seqid={seqid:.2f}) — spurious scoring risk")
                continue

        holo_summaries.append({
            "pdb_id": pdb_id, "note": note,
            "rmsd_A": rmsd, "seq_identity": seqid,
            "n_ligands": len(ligs), "pdb_path": str(holo_path),
        })

        for lig in ligs:
            lig_xyz_raw = lig["xyz"]
            # transform ligand to query frame
            if R is not None:
                lig_xyz_t = transform_atoms(
                    [{"xyz": xyz} for xyz in lig_xyz_raw], R, t)
            else:
                lig_xyz_t = list(lig_xyz_raw)

            lig_key = f"{pdb_id}:{lig['chain']}:{lig['resnum']}:{lig['resname']}"

            # Build shells on query
            shells: dict[float, set[int]] = {}
            if query_chains:
                for c in SHELL_CUTOFFS:
                    shells[c] = build_shell(query_chains, lig_xyz_t, c)
            else:
                # fall back: use kcc_viz CA positions
                for c in SHELL_CUTOFFS:
                    sh = set()
                    for rid, rd in kcc_residues.items():
                        if any(dist(rd["ca_xyz"], lpt) <= c for lpt in lig_xyz_t):
                            sh.add(rid + topo_offset)
                    shells[c] = sh

            # Score every PRISM site
            site_list = sorted(sites.values(), key=lambda s: -s["quality_score"])
            for rank_i, site in enumerate(site_list[:args.max_rank], 1):
                row = score_site_vs_ligand(
                    site, kcc_residues, lig_xyz_t, shells, query_resnum_map)
                row.update({
                    "holo_pdb":     pdb_id,
                    "holo_note":    note,
                    "ligand_key":   lig_key,
                    "lig_resname":  lig["resname"],
                    "alignment_rmsd_A":    round(rmsd, 3),
                    "alignment_seqid":     round(seqid, 3),
                    "site_rank":    rank_i,
                })
                all_score_rows.append(row)

            # Score baselines if proposals available
            if args.fpocket_dir:
                fps = [{"method":"fpocket","rank":i+1,"xyz":[p],"centroid":p}
                       for i,p in enumerate([])]  # placeholder from dir
            # run fpocket fresh if not pre-computed
            fp_proposals = run_fpocket(query_pdb, outdir) if query_pdb.exists() else []
            p2_proposals = run_p2rank(query_pdb, outdir)  if query_pdb.exists() else []

            for prop in fp_proposals[:15]:
                row = score_baseline_vs_ligand(prop, query_chains, lig_xyz_t, shells)
                row.update({"holo_pdb": pdb_id, "ligand_key": lig_key,
                             "lig_resname": lig["resname"]})
                all_baseline_fp.append(row)
            for prop in p2_proposals[:15]:
                row = score_baseline_vs_ligand(prop, query_chains, lig_xyz_t, shells)
                row.update({"holo_pdb": pdb_id, "ligand_key": lig_key,
                             "lig_resname": lig["resname"]})
                all_baseline_p2.append(row)

    if not all_score_rows:
        print("\n[WARN] No scoring rows generated — check holo references and PDB downloads.")
        return

    # ── 4. Assign final rank-adjusted verdicts ────────────────────────────
    print(f"\n[4] Scored {len(all_score_rows)} site×ligand pairs.")
    good_cnt = sum(1 for r in all_score_rows if r["is_good"])
    print(f"  Good hits (GOOD/STRONG): {good_cnt} "
          f"({100*good_cnt/len(all_score_rows):.1f}%)")

    # ── 5. Family collapse ────────────────────────────────────────────────
    print("\n[5] Collapsing site families...")
    families = collapse_families(sites)
    print(f"  {len(sites)} sites → {len(families)} families")
    # Family score rows: max over members
    fam_rows = []
    for fam_set in families:
        fam_id = min(fam_set)
        fam_site_rows = [r for r in all_score_rows if r["site_id"] in fam_set]
        if not fam_site_rows:
            continue
        by_lig: dict = collections.defaultdict(list)
        for r in fam_site_rows:
            by_lig[r["ligand_key"]].append(r)
        for lig_key, lig_rows in by_lig.items():
            best = min(lig_rows, key=lambda r: r["min_lining_dist_A"])
            fam_rows.append({
                "family_id":         fam_id,
                "family_size":       len(fam_set),
                "member_sites":      sorted(fam_set),
                "ligand_key":        lig_key,
                "min_lining_dist_A": best["min_lining_dist_A"],
                "jaccard_8A":        max(r["jaccard_8A"] for r in lig_rows),
                "max_intersection_8A": max(r["intersection_8A"] for r in lig_rows),
                "kcc_driver_in_8A":  any(r["kcc_driver_in_8A_shell"] for r in lig_rows),
                "arrow_driver_in_8A": any(r["arrow_driver_in_8A"] for r in lig_rows),
                "best_verdict":      best["verdict"],
                "is_good":           any(r["is_good"] for r in lig_rows),
                "holo_pdb":          best["holo_pdb"],
            })

    # ── 6. LORO blinded validation ────────────────────────────────────────
    print("\n[6] Running LORO blinded validation...")
    loro_rows = loro_validation(all_score_rows)
    n_excellent = sum(1 for r in loro_rows if r["excellent"])
    print(f"  {len(loro_rows)} ligand×holo LORO tests | "
          f"{n_excellent} blinded rediscoveries (rank ≤ 3)")

    # ── 7. SR@K ───────────────────────────────────────────────────────────
    sr_prism   = compute_sr_at_k(all_score_rows)
    sr_fpocket = compute_sr_at_k(all_baseline_fp) if all_baseline_fp else {}
    sr_p2rank  = compute_sr_at_k(all_baseline_p2) if all_baseline_p2 else {}

    print(f"\n[7] SR@K — PRISM: {sr_prism}")
    if sr_fpocket: print(f"         fpocket: {sr_fpocket}")
    if sr_p2rank:  print(f"         P2Rank:  {sr_p2rank}")

    # ── 8. Write all output files ─────────────────────────────────────────
    print(f"\n[8] Writing outputs to {outdir}")
    write_csv(outdir / "holo_alignment_summary.csv", holo_summaries)
    write_csv(outdir / "prism_site_vs_ligand_shells.csv", all_score_rows)
    write_csv(outdir / "prism_family_vs_ligand_shells.csv", fam_rows)
    write_csv(outdir / "baseline_fpocket_vs_ligands.csv", all_baseline_fp)
    write_csv(outdir / "baseline_p2rank_vs_ligands.csv",  all_baseline_p2)
    write_csv(outdir / "blinded_leave_one_holo_out.csv",  loro_rows)

    sr_rows = []
    for method, sr in [("PRISM", sr_prism), ("fpocket", sr_fpocket), ("P2Rank", sr_p2rank)]:
        if sr:
            row = {"method": method}; row.update(sr); sr_rows.append(row)
    write_csv(outdir / "sr_at_k_comparison.csv", sr_rows)

    write_pymol(outdir / "pymol_overlay.pml", query_pdb, sites,
                holo_summaries, target)
    write_report(outdir / "validation_report.md", target, sites, all_score_rows,
                 families, loro_rows, sr_prism, sr_fpocket, sr_p2rank,
                 holo_summaries)

    print(f"\n  ✓ holo_alignment_summary.csv")
    print(f"  ✓ prism_site_vs_ligand_shells.csv  ({len(all_score_rows)} rows)")
    print(f"  ✓ prism_family_vs_ligand_shells.csv ({len(fam_rows)} rows)")
    print(f"  ✓ baseline_fpocket_vs_ligands.csv  ({len(all_baseline_fp)} rows)")
    print(f"  ✓ baseline_p2rank_vs_ligands.csv   ({len(all_baseline_p2)} rows)")
    print(f"  ✓ sr_at_k_comparison.csv")
    print(f"  ✓ blinded_leave_one_holo_out.csv   ({len(loro_rows)} rows)")
    print(f"  ✓ validation_report.md")
    print(f"  ✓ pymol_overlay.pml")
    print(f"\n=== DONE: {target} ===\n")

if __name__ == "__main__":
    main()
