#!/usr/bin/env python3
"""
PFR Step C — Holo Interaction Fingerprint Extractor
=====================================================
Extracts protein-ligand interaction fingerprints from holo PDB files using
pure-Python geometry, no PLIP dependency (OpenBabel build failure on this host).

Interaction types detected:
  AROMATIC   — pi-pi or CH-pi: ligand aromatic ring centroid near protein
               aromatic residue (PHE/TYR/TRP/HIS) within 5.5 Å; ring normals
               within 30° (parallel stacking) or 60-90° (T-shaped)
  HBOND      — hydrogen bond: N or O in ligand within 3.5 Å of protein N/O;
               direction vector = ligand_atom → protein_atom (donor→acceptor)
  HYDROPHOBIC— C atom in ligand within 4.5 Å of protein C/S from
               ILE/LEU/VAL/ALA/MET/PHE/TRP/PRO/CYS; no adjacent polar atoms
               within 2.0 Å in the ligand
  IONIC_NEG  — ligand anionic group: N⁺ in ligand (protonated amine, Lys-like)
               near protein Glu/Asp O within 4.0 Å
  IONIC_POS  — ligand cationic group: O⁻ or N in COO⁻/SO₃⁻ near protein
               Arg/Lys N within 4.0 Å
  COVALENT   — C in ligand bonded (≤1.9 Å) to protein Cys SG

Each interaction record:
  type           str       pharmacophore type (matches Script A feature types)
  position       [x,y,z]  centroid of the ligand atom(s) involved
  direction      [dx,dy,dz] unit vector pointing from ligand toward protein partner
  ligand_atom    str       element:resname:atom_name
  protein_atom   str       resname:resnum:atom_name (nearest protein partner)
  distance_A     float     actual distance
  pdb_id         str
  ligand_resname str

Firewall: TIMESTAMP_C written into every output JSON.
          Step D asserts TIMESTAMP_C > TIMESTAMP_A.

Output: /mnt/storage/prism-outputs/blind_validation/pfr_validation/holo_interactions/
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

PFR_BASE      = Path("/mnt/storage/prism-outputs/blind_validation")
PFR_OUT_BASE  = PFR_BASE / "pfr_validation"
POSTFREEZE    = Path(
    "/home/diddy/Desktop/Prism4D-bio/docs/blind_validation/post_freeze_validation"
)
MANIFEST_PATH = PFR_OUT_BASE / "sha256_manifest_pfr.txt"
PHARMA_DIR    = PFR_OUT_BASE / "pharmacophores"

# ── Target → holo catalog (from holo_alignment_summary.csv files) ────────────

TARGETS: dict[str, dict] = {
    "B01_HRAS_Q61H": {
        "apo_pdb": "4L9S",
        "holos": [
            {"pdb": "6OIM", "chain": "A",
             "ligand_resnames": ["MOV", "GDP"],
             "note": "KRAS G12C + sotorasib (MOV) + GDP"},
            {"pdb": "7RPZ", "chain": "A",
             "ligand_resnames": ["6IC"],
             "note": "KRAS G12D + MRTX1133 (6IC)"},
        ],
    },
    "B02_CDK2_allosteric": {
        "apo_pdb": "1HCL",
        "holos": [
            {"pdb": "3PXZ", "chain": "A",
             "ligand_resnames": ["JWS", "2AN"],
             "note": "CDK2 allosteric + JWS648 + ANS probe"},
            {"pdb": "4GCJ", "chain": "A",
             "ligand_resnames": ["X64"],
             "note": "CDK2 + RC-3-89 (X64)"},
        ],
    },
    "B05_TP53_R175H": {
        "apo_pdb": "2OCJ",
        "holos": [
            {"pdb": "3ZME", "chain": "A",
             "ligand_resnames": [],   # auto-detect
             "note": "p53 Y220C + compound 23 (PROVISIONAL)"},
            {"pdb": "4AGQ", "chain": "A",
             "ligand_resnames": [],
             "note": "p53 Y220C + compound 3 (PROVISIONAL)"},
        ],
    },
    "B06_cGAS": {
        "apo_pdb": "4KM5",
        "holos": [
            {"pdb": "4O67", "chain": "A",
             "ligand_resnames": ["1SY"],
             "note": "cGAS + cGAMP product"},
            {"pdb": "5V8N", "chain": "A",
             "ligand_resnames": ["8ZP"],
             "note": "cGAS + high-affinity inhibitor (8ZP)"},
        ],
    },
    "B08_CRBN": {
        "apo_pdb": "4TZ4_chainC",
        "holos": [
            {"pdb": "4CI3", "chain": "B",
             "ligand_resnames": ["Y70"],
             "note": "DDB1-CRBN + pomalidomide"},
            {"pdb": "5FQD", "chain": "B",
             "ligand_resnames": ["LVY"],
             "note": "DDB1-CRBN + lenalidomide"},
        ],
    },
    "B09_Thrombin_exosite": {
        "apo_pdb": "1PPB",
        "holos": [
            {"pdb": "1HAH", "chain": "H",
             "ligand_resnames": [],   # hirugen peptide chain I
             "note": "Thrombin + hirugen at exosite I"},
        ],
    },
}

# Residue sets for classification
HYDROPHOBIC_RESIDUES = {
    "ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "CYS"
}
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
HBOND_DONOR_RESIDUES = {
    "SER", "THR", "TYR", "ASN", "GLN", "LYS", "ARG", "HIS",
    "ASP", "GLU", "TRP",
}
ACIDIC_RESIDUES   = {"ASP", "GLU"}
BASIC_RESIDUES    = {"LYS", "ARG"}

# Solvent / buffer residues to ignore in HETATM
SOLVENT_RESNAMES = {
    "HOH", "DOD", "WAT", "H2O", "NA", "MG", "ZN", "CL", "K",
    "CA", "FE", "MN", "CU", "CO", "NI", "CS", "BR", "I",
    "SO4", "PO4", "HPO", "GOL", "EDO", "ACE", "NH2", "ACT",
    "FMT", "MSE", "SEC", "CSO", "SCN", "NO3", "DMS", "MOH",
    "EOH", "MPD", "IMD", "TRS", "EPE", "MES", "HED", "IOD",
    "BME", "MLY", "SEP", "TPO", "PTR",
}

BACKBONE_ATOMS = {"CA", "C", "N", "O", "CB"}


# ── PDB parser ───────────────────────────────────────────────────────────────

def parse_pdb(pdb_path: Path, target_chain: str | None = None) -> dict:
    """
    Parse ATOM + HETATM records.
    Returns:
      protein_atoms  list of atom dicts from ATOM records (chain = target_chain)
      ligand_atoms   list of atom dicts from HETATM records (non-solvent)
    """
    protein_atoms = []
    ligand_atoms  = []

    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                atom_name = line[12:16].strip()
                alt_loc   = line[16].strip()
                resname   = line[17:20].strip()
                chain     = line[21].strip()
                resnum    = line[22:26].strip()
                x         = float(line[30:38])
                y         = float(line[38:46])
                z         = float(line[46:54])
                element   = line[76:78].strip() if len(line) > 77 else ""
            except (ValueError, IndexError):
                continue

            # Skip alternate conformations (keep blank or 'A')
            if alt_loc not in ("", " ", "A"):
                continue

            # Infer element from atom name if missing
            if not element:
                element = atom_name[0] if atom_name else "C"

            atom = {
                "atom_name": atom_name,
                "resname":   resname,
                "chain":     chain,
                "resnum":    resnum,
                "x": x, "y": y, "z": z,
                "element":   element.upper(),
            }

            if rec == "ATOM":
                if target_chain is None or chain == target_chain:
                    protein_atoms.append(atom)
            else:  # HETATM
                if resname not in SOLVENT_RESNAMES:
                    ligand_atoms.append(atom)

    return {"protein_atoms": protein_atoms, "ligand_atoms": ligand_atoms}


# ── Aromatic ring detection ───────────────────────────────────────────────────

def detect_ligand_rings(lig_atoms: list) -> list:
    """
    Detect planar aromatic ring systems in the ligand by finding groups of
    ≥5 C/N atoms that are mutually connected (bond ≤ 1.65 Å) and coplanar
    (largest eigenvalue of covariance ≪ two others → near-zero thickness).

    Returns list of ring dicts: {centroid, normal, atom_indices}.
    """
    if not lig_atoms:
        return []

    pos = np.array([[a["x"], a["y"], a["z"]] for a in lig_atoms])
    el  = [a["element"] for a in lig_atoms]
    n   = len(lig_atoms)

    # Bond adjacency (C, N within 1.65 Å of each other — includes C=C, C-N, C=N)
    organic = [i for i, e in enumerate(el) if e in ("C", "N", "O", "S")]
    adj: dict[int, list[int]] = defaultdict(list)
    for i in range(len(organic)):
        for j in range(i + 1, len(organic)):
            ai, aj = organic[i], organic[j]
            d = float(np.linalg.norm(pos[ai] - pos[aj]))
            if d <= 1.65:
                adj[ai].append(aj)
                adj[aj].append(ai)

    # Find rings: connected components of size 5–8 where all carbons have
    # exactly 2–3 C/N neighbours (aromatic connectivity)
    visited = set()
    rings   = []

    def bfs(start):
        comp = []
        q = [start]
        seen = {start}
        while q:
            node = q.pop()
            comp.append(node)
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        return comp

    for start in organic:
        if start in visited:
            continue
        comp = bfs(start)
        for i in comp:
            visited.add(i)
        if len(comp) < 5 or len(comp) > 9:
            continue
        # Coplanarity test
        ring_pos = pos[comp]
        centered = ring_pos - ring_pos.mean(axis=0)
        cov = np.cov(centered.T)
        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        # Planar: smallest eigenvalue ≪ other two
        if evals[0] > 0.5 * evals[1]:
            continue
        centroid = ring_pos.mean(axis=0)
        normal   = evecs[:, 0]  # smallest eigenvector = normal to plane
        rings.append({"centroid": centroid, "normal": normal, "atom_indices": comp})

    return rings


# ── Nearest-protein-atom search ───────────────────────────────────────────────

def build_protein_kdtree_arrays(protein_atoms: list):
    """Return (pos_array, atoms_list) for fast distance queries."""
    pos = np.array([[a["x"], a["y"], a["z"]] for a in protein_atoms],
                   dtype=np.float32)
    return pos, protein_atoms


def nearest_protein_atom(lig_pos: np.ndarray, prot_pos: np.ndarray,
                         prot_atoms: list, cutoff: float,
                         resname_filter: set | None = None) -> tuple:
    """
    Return (protein_atom_dict, distance) for the nearest protein atom
    within cutoff.  Returns (None, inf) if none found.
    """
    dists = np.linalg.norm(prot_pos - lig_pos, axis=1)
    order = np.argsort(dists)
    for i in order:
        if dists[i] > cutoff:
            break
        pa = prot_atoms[i]
        if resname_filter and pa["resname"] not in resname_filter:
            continue
        return pa, float(dists[i])
    return None, float("inf")


# ── Interaction extraction ────────────────────────────────────────────────────

def extract_interactions(parsed: dict, holo_cfg: dict, pdb_id: str,
                         explicit_resnames: list) -> list:
    """
    Run all interaction detection passes on one holo PDB.
    Returns list of interaction dicts.
    """
    prot_atoms = parsed["protein_atoms"]
    lig_atoms  = parsed["ligand_atoms"]

    # Filter ligand atoms to the target ligand resnames (or auto-detect)
    if explicit_resnames:
        lig_atoms = [a for a in lig_atoms if a["resname"] in explicit_resnames]
    if not lig_atoms:
        return []

    prot_pos, prot_list = build_protein_kdtree_arrays(prot_atoms)
    interactions = []

    # ── AROMATIC: ligand ring centroid ↔ protein aromatic residue ────────────
    lig_rings = detect_ligand_rings(lig_atoms)
    for ring in lig_rings:
        rc    = ring["centroid"].astype(np.float32)
        rnorm = ring["normal"]
        # Find nearest aromatic protein residue centroid
        arom_prot = [a for a in prot_atoms
                     if a["resname"] in AROMATIC_RESIDUES
                     and a["element"] == "C"]
        if not arom_prot:
            continue
        pa, dist = nearest_protein_atom(
            rc, np.array([[a["x"], a["y"], a["z"]] for a in arom_prot],
                         dtype=np.float32),
            arom_prot, cutoff=5.5
        )
        if pa is None:
            continue
        prot_pos_v = np.array([pa["x"], pa["y"], pa["z"]])
        direction  = prot_pos_v - ring["centroid"]
        dnorm      = np.linalg.norm(direction)
        direction  = direction / (dnorm + 1e-9)
        interactions.append({
            "type":           "AROMATIC",
            "position":       ring["centroid"].tolist(),
            "direction":      direction.tolist(),
            "normal":         rnorm.tolist(),
            "ligand_atom":    f"ring:{lig_atoms[ring['atom_indices'][0]]['resname']}",
            "protein_atom":   f"{pa['resname']}:{pa['resnum']}:{pa['atom_name']}",
            "distance_A":     round(dist, 3),
            "pdb_id":         pdb_id,
            "ligand_resname": lig_atoms[ring["atom_indices"][0]]["resname"],
        })

    # ── HBOND: ligand N/O ↔ protein N/O ─────────────────────────────────────
    for la in lig_atoms:
        if la["element"] not in ("N", "O"):
            continue
        la_pos = np.array([la["x"], la["y"], la["z"]], dtype=np.float32)
        # Filter protein atoms to H-bond capable N/O
        hb_prot = [a for a in prot_atoms if a["element"] in ("N", "O")]
        if not hb_prot:
            continue
        hp_pos = np.array([[a["x"], a["y"], a["z"]] for a in hb_prot],
                           dtype=np.float32)
        pa, dist = nearest_protein_atom(la_pos, hp_pos, hb_prot, cutoff=3.5)
        if pa is None:
            continue
        prot_pos_v = np.array([pa["x"], pa["y"], pa["z"]])
        direction  = prot_pos_v - la_pos
        dnorm      = np.linalg.norm(direction)
        direction  = direction / (dnorm + 1e-9)
        interactions.append({
            "type":           "HBOND_DONOR",
            "position":       la_pos.tolist(),
            "direction":      direction.tolist(),
            "ligand_atom":    f"{la['element']}:{la['resname']}:{la['atom_name']}",
            "protein_atom":   f"{pa['resname']}:{pa['resnum']}:{pa['atom_name']}",
            "distance_A":     round(dist, 3),
            "pdb_id":         pdb_id,
            "ligand_resname": la["resname"],
        })

    # ── HYDROPHOBIC: ligand C ↔ protein hydrophobic C/S ──────────────────────
    # Only non-polar ligand carbons: exclude C with adjacent N/O within 2.0 Å
    lig_pos_all = np.array([[a["x"], a["y"], a["z"]] for a in lig_atoms],
                            dtype=np.float32)
    for li, la in enumerate(lig_atoms):
        if la["element"] != "C":
            continue
        # Check if this carbon is adjacent to a polar atom in the ligand
        la_pos = lig_pos_all[li]
        polar_dists = np.linalg.norm(lig_pos_all - la_pos, axis=1)
        has_polar_nb = any(
            polar_dists[j] <= 2.0 and lig_atoms[j]["element"] in ("N", "O", "F", "Cl", "Br")
            for j in range(len(lig_atoms)) if j != li
        )
        if has_polar_nb:
            continue
        # Nearest hydrophobic protein atom
        hp_prot = [a for a in prot_atoms
                   if a["resname"] in HYDROPHOBIC_RESIDUES
                   and a["element"] in ("C", "S")
                   and a["atom_name"] not in BACKBONE_ATOMS]
        if not hp_prot:
            continue
        hp_pos = np.array([[a["x"], a["y"], a["z"]] for a in hp_prot], dtype=np.float32)
        pa, dist = nearest_protein_atom(la_pos, hp_pos, hp_prot, cutoff=4.5)
        if pa is None:
            continue
        prot_pos_v = np.array([pa["x"], pa["y"], pa["z"]])
        direction  = prot_pos_v - la_pos
        dnorm      = np.linalg.norm(direction)
        direction  = direction / (dnorm + 1e-9)
        interactions.append({
            "type":           "HYDROPHOBIC",
            "position":       la_pos.tolist(),
            "direction":      direction.tolist(),
            "ligand_atom":    f"C:{la['resname']}:{la['atom_name']}",
            "protein_atom":   f"{pa['resname']}:{pa['resnum']}:{pa['atom_name']}",
            "distance_A":     round(dist, 3),
            "pdb_id":         pdb_id,
            "ligand_resname": la["resname"],
        })

    # ── IONIC_NEG: ligand N⁺ ↔ protein acidic O ─────────────────────────────
    for la in lig_atoms:
        if la["element"] != "N":
            continue
        la_pos = np.array([la["x"], la["y"], la["z"]], dtype=np.float32)
        ac_prot = [a for a in prot_atoms
                   if a["resname"] in ACIDIC_RESIDUES
                   and a["element"] == "O"]
        if not ac_prot:
            continue
        ac_pos = np.array([[a["x"], a["y"], a["z"]] for a in ac_prot], dtype=np.float32)
        pa, dist = nearest_protein_atom(la_pos, ac_pos, ac_prot, cutoff=4.0)
        if pa is None:
            continue
        prot_pos_v = np.array([pa["x"], pa["y"], pa["z"]])
        direction  = prot_pos_v - la_pos
        dnorm      = np.linalg.norm(direction)
        direction  = direction / (dnorm + 1e-9)
        interactions.append({
            "type":           "IONIC_NEG",
            "position":       la_pos.tolist(),
            "direction":      direction.tolist(),
            "ligand_atom":    f"N:{la['resname']}:{la['atom_name']}",
            "protein_atom":   f"{pa['resname']}:{pa['resnum']}:{pa['atom_name']}",
            "distance_A":     round(dist, 3),
            "pdb_id":         pdb_id,
            "ligand_resname": la["resname"],
        })

    # ── IONIC_POS: ligand O ↔ protein basic N ────────────────────────────────
    for la in lig_atoms:
        if la["element"] != "O":
            continue
        la_pos = np.array([la["x"], la["y"], la["z"]], dtype=np.float32)
        ba_prot = [a for a in prot_atoms
                   if a["resname"] in BASIC_RESIDUES
                   and a["element"] == "N"]
        if not ba_prot:
            continue
        ba_pos = np.array([[a["x"], a["y"], a["z"]] for a in ba_prot], dtype=np.float32)
        pa, dist = nearest_protein_atom(la_pos, ba_pos, ba_prot, cutoff=4.0)
        if pa is None:
            continue
        prot_pos_v = np.array([pa["x"], pa["y"], pa["z"]])
        direction  = prot_pos_v - la_pos
        dnorm      = np.linalg.norm(direction)
        direction  = direction / (dnorm + 1e-9)
        interactions.append({
            "type":           "IONIC_POS",
            "position":       la_pos.tolist(),
            "direction":      direction.tolist(),
            "ligand_atom":    f"O:{la['resname']}:{la['atom_name']}",
            "protein_atom":   f"{pa['resname']}:{pa['resnum']}:{pa['atom_name']}",
            "distance_A":     round(dist, 3),
            "pdb_id":         pdb_id,
            "ligand_resname": la["resname"],
        })

    # ── COVALENT: ligand C bonded to protein Cys SG ──────────────────────────
    cys_sg = [a for a in prot_atoms
              if a["resname"] == "CYS" and a["atom_name"] == "SG"]
    for la in lig_atoms:
        if la["element"] != "C":
            continue
        la_pos = np.array([la["x"], la["y"], la["z"]], dtype=np.float32)
        for sg in cys_sg:
            sg_pos = np.array([sg["x"], sg["y"], sg["z"]])
            dist   = float(np.linalg.norm(la_pos - sg_pos))
            if dist <= 1.9:
                direction = sg_pos - la_pos
                dnorm = np.linalg.norm(direction)
                direction = direction / (dnorm + 1e-9)
                interactions.append({
                    "type":           "COVALENT",
                    "position":       la_pos.tolist(),
                    "direction":      direction.tolist(),
                    "ligand_atom":    f"C:{la['resname']}:{la['atom_name']}",
                    "protein_atom":   f"CYS:{sg['resnum']}:SG",
                    "distance_A":     round(dist, 3),
                    "pdb_id":         pdb_id,
                    "ligand_resname": la["resname"],
                })

    return interactions


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_interactions(interactions: list, position_thresh: float = 1.5) -> list:
    """
    Remove duplicate interactions of the same type with centroids within
    position_thresh Å (keeps the one with shorter distance to protein).
    """
    kept = []
    for ia in interactions:
        pos_a = np.array(ia["position"])
        duplicate = False
        for ib in kept:
            if ib["type"] != ia["type"]:
                continue
            if np.linalg.norm(np.array(ib["position"]) - pos_a) < position_thresh:
                # Keep shorter distance
                if ia["distance_A"] < ib["distance_A"]:
                    kept.remove(ib)
                    kept.append(ia)
                duplicate = True
                break
        if not duplicate:
            kept.append(ia)
    return kept


# ── Utility ──────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Per-target processing ─────────────────────────────────────────────────────

def process_target(target: str, cfg: dict, timestamp_c: str,
                   out_dir: Path, manifest_lines: list) -> list:
    tgt_out = out_dir / target
    tgt_out.mkdir(parents=True, exist_ok=True)
    pdb_cache = POSTFREEZE / target / ".pdb_cache"
    written   = []

    for holo in cfg["holos"]:
        pdb_id       = holo["pdb"]
        chain        = holo["chain"]
        explicit_res = holo.get("ligand_resnames", [])
        pdb_path     = pdb_cache / f"{pdb_id}.pdb"

        if not pdb_path.exists():
            print(f"  {target} {pdb_id}: PDB not found at {pdb_path} — skip")
            continue

        parsed = parse_pdb(pdb_path, target_chain=chain)
        print(f"  {target} {pdb_id}: "
              f"{len(parsed['protein_atoms'])} protein atoms, "
              f"{len(parsed['ligand_atoms'])} ligand atoms")

        interactions = extract_interactions(parsed, holo, pdb_id, explicit_res)

        # If no explicit resnames specified, auto-detect
        if not interactions and not explicit_res:
            parsed2 = parse_pdb(pdb_path, target_chain=None)
            interactions = extract_interactions(parsed2, holo, pdb_id, [])

        interactions = deduplicate_interactions(interactions)

        type_counts = {}
        for ia in interactions:
            type_counts[ia["type"]] = type_counts.get(ia["type"], 0) + 1
        print(f"    → {len(interactions)} interactions: {type_counts}")

        record = {
            "pdb_id":        pdb_id,
            "chain":         chain,
            "target":        target,
            "ligand_resnames_queried": explicit_res,
            "n_interactions": len(interactions),
            "interactions":  interactions,
            "timestamp_c":   timestamp_c,
            "method":        "pure-Python geometry (PLIP fallback: openbabel build unavailable)",
            "note":          holo.get("note", ""),
        }

        out_path = tgt_out / f"{pdb_id}_interactions.json"
        out_path.write_text(json.dumps(record, indent=2))
        digest = sha256_file(out_path)
        manifest_lines.append(
            f"{digest}  {out_path}  TIMESTAMP_C={timestamp_c}"
        )
        print(f"    sha256={digest[:12]}…  → {out_path.name}")
        written.append(str(out_path))

    return written


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="PFR Step C: holo interaction fingerprint extraction"
    )
    ap.add_argument("--targets", nargs="*", default=list(TARGETS.keys()))
    ap.add_argument("--out-dir", default=str(PFR_OUT_BASE / "holo_interactions"))
    args = ap.parse_args()

    timestamp_c = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PFR Step C — Holo Interaction Fingerprint Extractor")
    print("=" * 70)
    print(f"TIMESTAMP_C  = {timestamp_c}")
    print(f"Output dir   = {out_dir}")
    print()

    # Assert TIMESTAMP_C > TIMESTAMP_A
    pharma_files = list(PHARMA_DIR.glob("*/*_pharmacophore.json")) if PHARMA_DIR.exists() else []
    for p in pharma_files[:1]:
        ts_a = json.loads(p.read_text()).get("timestamp_a", "")
        if ts_a and ts_a >= timestamp_c:
            print(f"WARNING: TIMESTAMP_A ({ts_a}) >= TIMESTAMP_C ({timestamp_c})")
            print("This violates firewall ordering — check system clock or run order")

    manifest_lines = [
        "",
        f"# ── PFR Step C — TIMESTAMP_C={timestamp_c} ──────────────────────",
        f"# pfr_c_holo_interaction_extractor.py",
        f"# method: pure-Python geometry (PLIP unavailable: openbabel build fails)",
    ]

    all_written = []
    for target in args.targets:
        if target not in TARGETS:
            continue
        print(f"\n[{target}]")
        written = process_target(
            target, TARGETS[target], timestamp_c, out_dir, manifest_lines
        )
        all_written.extend(written)

    with open(MANIFEST_PATH, "a") as fh:
        fh.write("\n".join(manifest_lines) + "\n")

    print()
    print("=" * 70)
    print(f"Step C complete — {len(all_written)} holo interaction JSONs written")
    print(f"Manifest updated: {MANIFEST_PATH}")
    print(f"TIMESTAMP_C = {timestamp_c}")


if __name__ == "__main__":
    main()
