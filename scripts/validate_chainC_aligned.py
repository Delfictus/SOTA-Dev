#!/usr/bin/env python3
"""
Validate PRISM materialized sites against 6LU7 Chain C ligand after aligning
6LU7 Chain A C-alpha atoms into the PRISM processed Chain A coordinate frame.

Outputs:
  aligned_chainC_ligand_validation_report.json
  aligned_chainC_ligand_transformed.pdb

This script is intentionally dependency-light: numpy only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


Vec3 = Tuple[float, float, float]


# ----------------------------
# Generic helpers
# ----------------------------

def dist(a: Iterable[float], b: Iterable[float]) -> float:
    aa = list(a)
    bb = list(b)
    return math.sqrt(sum((aa[i] - bb[i]) ** 2 for i in range(3)))


def centroid(points: List[Vec3]) -> Optional[List[float]]:
    if not points:
        return None
    return [
        sum(p[i] for p in points) / len(points)
        for i in range(3)
    ]


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def as_vec3(x: Any) -> Optional[List[float]]:
    if not isinstance(x, (list, tuple)) or len(x) < 3:
        return None
    try:
        return [float(x[0]), float(x[1]), float(x[2])]
    except Exception:
        return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# PDB parsing
# ----------------------------

def parse_pdb_atoms(path: Path, chain: Optional[str] = None, hetatm_only: bool = False, atom_name: Optional[str] = None) -> List[Dict[str, Any]]:
    atoms: List[Dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        rec = line[0:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        if hetatm_only and rec != "HETATM":
            continue
        ch = line[21].strip() or "_"
        if chain is not None and ch != chain:
            continue
        aname = line[12:16].strip()
        if atom_name is not None and aname != atom_name:
            continue
        try:
            atoms.append({
                "record": rec,
                "serial": int(line[6:11]),
                "atom_name": aname,
                "residue_name": line[17:20].strip(),
                "chain_id": ch,
                "residue_id": int(line[22:26]),
                "icode": line[26].strip(),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "line": line,
            })
        except Exception:
            continue
    return atoms


def write_transformed_pdb(path: Path, atoms: List[Dict[str, Any]], transformed_xyz: List[Vec3]) -> None:
    out = []
    for atom, xyz in zip(atoms, transformed_xyz):
        line = atom["line"]
        x, y, z = xyz
        if len(line) < 80:
            line = line.ljust(80)
        newline = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        out.append(newline)
    out.append("END")
    path.write_text("\n".join(out) + "\n")


# ----------------------------
# PRISM topology parsing
# ----------------------------

def iter_json_atoms(obj: Any) -> List[Dict[str, Any]]:
    """
    Best-effort parser for common topology JSON shapes.

    Supports atoms under:
      atoms
      topology.atoms
      protein_atoms
      nodes

    Recognizes coordinate fields:
      x/y/z
      pos / position / xyz / coord / coordinates
    """
    if isinstance(obj, dict):
        for key in ("atoms", "protein_atoms", "nodes"):
            if isinstance(obj.get(key), list):
                return obj[key]
        if isinstance(obj.get("topology"), dict):
            return iter_json_atoms(obj["topology"])
        if isinstance(obj.get("system"), dict):
            return iter_json_atoms(obj["system"])
    return []


def atom_coord_from_json(a: Dict[str, Any]) -> Optional[List[float]]:
    for keys in (("x", "y", "z"),):
        if all(k in a for k in keys):
            try:
                return [float(a["x"]), float(a["y"]), float(a["z"])]
            except Exception:
                pass
    for key in ("position", "pos", "xyz", "coord", "coordinates"):
        v = a.get(key)
        vv = as_vec3(v)
        if vv is not None:
            return vv
    return None


def get_any(a: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in a:
            return a[k]
    return default


def parse_prism_topology_ca(topology_path: Path, chain: str = "A") -> Dict[int, Vec3]:
    obj = load_json(topology_path)
    out: Dict[int, Vec3] = {}

    # PRISM topology schema:
    # positions is a flat xyz array of length n_atoms * 3.
    # ca_indices is a list of atom indices for C-alpha atoms.
    # residue_ids / chain_ids / atom_names are per-atom arrays.
    if (
        isinstance(obj, dict)
        and isinstance(obj.get("positions"), list)
        and isinstance(obj.get("ca_indices"), list)
        and isinstance(obj.get("residue_ids"), list)
    ):
        positions = obj["positions"]
        residue_ids = obj["residue_ids"]
        chain_ids = obj.get("chain_ids", ["A"] * len(residue_ids))
        atom_names = obj.get("atom_names", [""] * len(residue_ids))

        for atom_idx in obj["ca_indices"]:
            atom_idx = int(atom_idx)
            if atom_idx < 0 or atom_idx >= len(residue_ids):
                continue
            chain_id = str(chain_ids[atom_idx]).strip() or chain
            if chain_id != chain:
                continue
            if atom_names and str(atom_names[atom_idx]).strip() not in ("CA", ""):
                continue
            j = atom_idx * 3
            if j + 2 >= len(positions):
                continue
            rid = int(residue_ids[atom_idx])
            out[rid] = (float(positions[j]), float(positions[j + 1]), float(positions[j + 2]))

    if out:
        return out

    # Fallback: object-array atom schema.
    atoms = iter_json_atoms(obj)
    for a in atoms:
        if not isinstance(a, dict):
            continue
        atom_name = str(get_any(a, ["atom_name", "name", "atom", "atomName"], "")).strip()
        residue_id = safe_int(get_any(a, ["residue_id", "resid", "residue_idx", "residue_index", "residue_number"]))
        chain_id = str(get_any(a, ["chain_id", "chain", "chainId"], chain)).strip() or chain
        if residue_id is None or chain_id != chain or atom_name != "CA":
            continue
        coord = atom_coord_from_json(a)
        if coord is not None:
            out[residue_id] = (coord[0], coord[1], coord[2])

    if not out:
        raise RuntimeError(
            f"No CA atoms parsed from {topology_path}. "
            "Expected PRISM flat positions + ca_indices schema or atom-object schema."
        )
    return out


def parse_prism_topology_residue_atoms(topology_path: Path, chain: str = "A") -> Dict[int, List[Vec3]]:
    obj = load_json(topology_path)
    out: Dict[int, List[Vec3]] = {}

    # PRISM flat topology schema.
    if (
        isinstance(obj, dict)
        and isinstance(obj.get("positions"), list)
        and isinstance(obj.get("residue_ids"), list)
    ):
        positions = obj["positions"]
        residue_ids = obj["residue_ids"]
        chain_ids = obj.get("chain_ids", ["A"] * len(residue_ids))

        for atom_idx, rid_raw in enumerate(residue_ids):
            chain_id = str(chain_ids[atom_idx]).strip() or chain
            if chain_id != chain:
                continue
            j = atom_idx * 3
            if j + 2 >= len(positions):
                continue
            rid = int(rid_raw)
            out.setdefault(rid, []).append((float(positions[j]), float(positions[j + 1]), float(positions[j + 2])))

    if out:
        return out

    # Fallback object-array schema.
    atoms = iter_json_atoms(obj)
    for a in atoms:
        if not isinstance(a, dict):
            continue
        residue_id = safe_int(get_any(a, ["residue_id", "resid", "residue_idx", "residue_index", "residue_number"]))
        chain_id = str(get_any(a, ["chain_id", "chain", "chainId"], chain)).strip() or chain
        if residue_id is None or chain_id != chain:
            continue
        coord = atom_coord_from_json(a)
        if coord is None:
            continue
        out.setdefault(residue_id, []).append((coord[0], coord[1], coord[2]))

    return out


# ----------------------------
# Alignment
# ----------------------------

def kabsch_fit(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return R, t such that P @ R + t approximates Q.
    P: moving coordinates, e.g. 6LU7 Chain A
    Q: fixed coordinates, e.g. PRISM topology Chain A
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc

    C = P0.T @ Q0
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    t = Qc - Pc @ R

    Pfit = P @ R + t
    rmsd = float(np.sqrt(np.mean(np.sum((Pfit - Q) ** 2, axis=1))))
    return R, t, rmsd


def apply_transform(points: List[Vec3], R: np.ndarray, t: np.ndarray) -> List[Vec3]:
    P = np.array(points, dtype=np.float64)
    X = P @ R + t
    return [(float(x), float(y), float(z)) for x, y, z in X]


def build_ca_pairs(ref_pdb: Path, prism_topology: Path, ref_chain: str, prism_chain: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    ref_ca_atoms = parse_pdb_atoms(ref_pdb, chain=ref_chain, atom_name="CA")
    ref_ca = {a["residue_id"]: (a["x"], a["y"], a["z"]) for a in ref_ca_atoms}
    prism_ca = parse_prism_topology_ca(prism_topology, chain=prism_chain)

    common = sorted(set(ref_ca) & set(prism_ca))
    if len(common) < 10:
        raise RuntimeError(
            f"Too few shared CA residue IDs for alignment: {len(common)}. "
            "Need residue mapping or check chain/topology numbering."
        )

    P = np.array([ref_ca[r] for r in common], dtype=np.float64)
    Q = np.array([prism_ca[r] for r in common], dtype=np.float64)
    return P, Q, common


# ----------------------------
# Site parsing and scoring
# ----------------------------

CENTROID_VIEW_KEYS = [
    "whole_site",
    "lining_mass",
    "causal_driver",
    "hot_phase",
    "cold_phase",
    "return_phase",
    "ligand_adjacent",
]


def extract_sites(materialized_path: Path) -> List[Dict[str, Any]]:
    data = load_json(materialized_path)
    if isinstance(data, dict):
        for key in ("binding_sites", "sites", "site_candidates"):
            if isinstance(data.get(key), list):
                return data[key]
    if isinstance(data, list):
        return data
    raise RuntimeError(f"Cannot find binding_sites/sites list in {materialized_path}")


def site_centroid_views(site: Dict[str, Any]) -> Dict[str, List[float]]:
    views: Dict[str, List[float]] = {}

    cm = site.get("centroid_manifold")
    if isinstance(cm, dict):
        for k in CENTROID_VIEW_KEYS:
            v = as_vec3(cm.get(k))
            if v is not None:
                views[k] = v

    for k in ("centroid_xyz", "centroid", "site_centroid_xyz"):
        v = as_vec3(site.get(k))
        if v is not None:
            views.setdefault(k, v)

    # Also search nested spatiotemporal block if present.
    st = site.get("spatiotemporal_so3_evidence")
    if isinstance(st, dict):
        cm2 = st.get("centroid_manifold")
        if isinstance(cm2, dict):
            for k in CENTROID_VIEW_KEYS:
                v = as_vec3(cm2.get(k))
                if v is not None:
                    views.setdefault(f"so3_{k}", v)

    return views


def extract_residue_ids_from_site(site: Dict[str, Any]) -> List[int]:
    ids: List[int] = []
    keys = [
        "representative_residues",
        "lining_residues",
        "driver_residues",
        "residue_support",
        "residues",
    ]
    for key in keys:
        v = site.get(key)
        if not isinstance(v, list):
            continue
        for item in v:
            if isinstance(item, int):
                ids.append(item)
            elif isinstance(item, dict):
                rid = safe_int(
                    item.get("residue_id")
                    or item.get("resid")
                    or item.get("residue_idx")
                    or item.get("residue_number")
                )
                if rid is not None:
                    ids.append(rid)
            else:
                rid = safe_int(item)
                if rid is not None:
                    ids.append(rid)
    return sorted(set(ids))


def min_distance_to_atoms(pt: List[float], atoms: List[Vec3]) -> float:
    return min(dist(pt, a) for a in atoms) if atoms else float("nan")


def ligand_contact_shell_residues(lig_atoms: List[Vec3], residue_atoms: Dict[int, List[Vec3]], cutoff: float) -> List[int]:
    shell = []
    for rid, atoms in residue_atoms.items():
        hit = False
        for la in lig_atoms:
            for aa in atoms:
                if dist(la, aa) <= cutoff:
                    hit = True
                    break
            if hit:
                break
        if hit:
            shell.append(rid)
    return sorted(shell)


def overlap_metrics(site_res: List[int], ref_shell: List[int]) -> Dict[str, Any]:
    A = set(site_res)
    B = set(ref_shell)
    inter = A & B
    union = A | B
    return {
        "site_residue_count": len(A),
        "reference_shell_count": len(B),
        "overlap_count": len(inter),
        "overlap_residues": sorted(inter),
        "jaccard": (len(inter) / len(union)) if union else None,
        "recall_vs_reference_shell": (len(inter) / len(B)) if B else None,
        "precision_vs_site_residues": (len(inter) / len(A)) if A else None,
    }


def grade_dcc(d: float) -> str:
    if d <= 2.0:
        return "EXCELLENT"
    if d <= 3.0:
        return "ACCEPTABLE"
    if d <= 5.0:
        return "USEFUL"
    if d <= 8.0:
        return "LENIENT"
    return "MISS"


def score_sites(
    sites: List[Dict[str, Any]],
    ligand_atoms: List[Vec3],
    ligand_centroid: List[float],
    ref_shell: List[int],
) -> List[Dict[str, Any]]:
    out = []
    for site in sites:
        views = site_centroid_views(site)
        view_scores = []
        for name, pt in views.items():
            dcc = dist(pt, ligand_centroid)
            min_atom = min_distance_to_atoms(pt, ligand_atoms)
            view_scores.append({
                "view": name,
                "centroid": pt,
                "dcc_to_ligand_centroid_A": dcc,
                "min_distance_to_ligand_atom_A": min_atom,
                "dcc_grade": grade_dcc(dcc),
            })
        view_scores.sort(key=lambda x: x["dcc_to_ligand_centroid_A"])

        site_res = extract_residue_ids_from_site(site)
        overlap = overlap_metrics(site_res, ref_shell)

        out.append({
            "site_id": site.get("site_id"),
            "rank": site.get("rank"),
            "materialization_status": site.get("materialization_status"),
            "best_centroid_view": view_scores[0] if view_scores else None,
            "all_centroid_views": view_scores,
            "contact_shell_overlap": overlap,
            "stream_support": site.get("stream_support"),
            "spatiotemporal_so3_evidence_status": (
                site.get("spatiotemporal_so3_evidence", {}).get("status")
                if isinstance(site.get("spatiotemporal_so3_evidence"), dict)
                else None
            ),
            "raw_site_summary": {
                "field_completeness": site.get("field_completeness"),
                "limitations": site.get("limitations"),
            },
        })

    out.sort(key=lambda r: (
        r["best_centroid_view"]["dcc_to_ligand_centroid_A"] if r["best_centroid_view"] else float("inf")
    ))
    return out


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prism-topology", required=True, help="PRISM processed topology JSON, e.g. data/targets/mpro_monomer.topology.json")
    ap.add_argument("--reference-pdb", required=True, help="Full 6LU7 PDB")
    ap.add_argument("--materialized-sites", required=True, help="binding_sites.materialized.json")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--reference-protein-chain", default="A")
    ap.add_argument("--prism-protein-chain", default="A")
    ap.add_argument("--ligand-chain", default="C")
    ap.add_argument("--contact-cutoff", type=float, default=5.0)
    ap.add_argument("--prism-residue-offset", type=int, default=0, help="Map PDB residue N to PRISM residue N + offset")
    args = ap.parse_args()

    prism_topology = Path(args.prism_topology)
    ref_pdb = Path(args.reference_pdb)
    materialized = Path(args.materialized_sites)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build CA pairs with optional residue-number offset: PDB residue N -> PRISM residue N + offset.
    ref_ca_atoms = parse_pdb_atoms(ref_pdb, chain=args.reference_protein_chain, atom_name="CA")
    ref_ca = {a["residue_id"]: (a["x"], a["y"], a["z"]) for a in ref_ca_atoms}
    prism_ca = parse_prism_topology_ca(prism_topology, chain=args.prism_protein_chain)
    common_res = sorted([rid for rid in ref_ca if (rid + args.prism_residue_offset) in prism_ca])
    if len(common_res) < 10:
        raise RuntimeError(f"Too few CA pairs with offset {args.prism_residue_offset}: {len(common_res)}")
    P = np.array([ref_ca[r] for r in common_res], dtype=np.float64)
    Q = np.array([prism_ca[r + args.prism_residue_offset] for r in common_res], dtype=np.float64)
    R, t, rmsd = kabsch_fit(P, Q)

    ligand_atoms_raw = parse_pdb_atoms(ref_pdb, chain=args.ligand_chain, hetatm_only=True)
    if not ligand_atoms_raw:
        raise RuntimeError(f"No HETATM ligand atoms found on chain {args.ligand_chain} in {ref_pdb}")

    ligand_xyz_raw = [(a["x"], a["y"], a["z"]) for a in ligand_atoms_raw]
    ligand_xyz_aligned = apply_transform(ligand_xyz_raw, R, t)
    ligand_centroid = centroid(ligand_xyz_aligned)
    assert ligand_centroid is not None

    transformed_pdb = outdir / "aligned_chainC_ligand_transformed.pdb"
    write_transformed_pdb(transformed_pdb, ligand_atoms_raw, ligand_xyz_aligned)

    residue_atoms = parse_prism_topology_residue_atoms(prism_topology, chain=args.prism_protein_chain)
    ref_shell = ligand_contact_shell_residues(ligand_xyz_aligned, residue_atoms, args.contact_cutoff)

    sites = extract_sites(materialized)
    results = score_sites(sites, ligand_xyz_aligned, ligand_centroid, ref_shell)

    report = {
        "schema_version": 2,
        "validation_type": "aligned_cross_chain_ligand_reference",
        "alignment": {
            "reference_pdb": str(ref_pdb),
            "reference_protein_chain": args.reference_protein_chain,
            "prism_topology": str(prism_topology),
            "prism_protein_chain": args.prism_protein_chain,
            "shared_ca_count": len(common_res),
            "prism_residue_offset": args.prism_residue_offset,
            "shared_residue_ids_first20": common_res[:20],
            "shared_residue_ids_last20": common_res[-20:],
            "ca_rmsd_A": rmsd,
            "note": "6LU7 Chain A CA atoms aligned to PRISM topology Chain A CA atoms; transform applied to Chain C HETATM ligand atoms.",
        },
        "reference_ligand": {
            "chain": args.ligand_chain,
            "hetatm_atom_count": len(ligand_xyz_aligned),
            "centroid_xyz_aligned": ligand_centroid,
            "transformed_pdb": str(transformed_pdb),
        },
        "reference_contact_shell": {
            "cutoff_A": args.contact_cutoff,
            "residue_count": len(ref_shell),
            "residue_ids": ref_shell,
        },
        "summary": {
            "n_sites_scored": len(results),
            "best_dcc_A": results[0]["best_centroid_view"]["dcc_to_ligand_centroid_A"] if results and results[0]["best_centroid_view"] else None,
            "best_dcc_site_id": results[0]["site_id"] if results else None,
            "best_dcc_grade": results[0]["best_centroid_view"]["dcc_grade"] if results and results[0]["best_centroid_view"] else None,
            "n_sites_dcc_le_2": sum(1 for r in results if r["best_centroid_view"] and r["best_centroid_view"]["dcc_to_ligand_centroid_A"] <= 2.0),
            "n_sites_dcc_le_3": sum(1 for r in results if r["best_centroid_view"] and r["best_centroid_view"]["dcc_to_ligand_centroid_A"] <= 3.0),
            "n_sites_dcc_le_5": sum(1 for r in results if r["best_centroid_view"] and r["best_centroid_view"]["dcc_to_ligand_centroid_A"] <= 5.0),
            "n_sites_dcc_le_8": sum(1 for r in results if r["best_centroid_view"] and r["best_centroid_view"]["dcc_to_ligand_centroid_A"] <= 8.0),
        },
        "results_sorted_by_best_dcc": results,
    }

    report_path = outdir / "aligned_chainC_ligand_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("alignment_shared_ca_count", len(common_res))
    print("alignment_ca_rmsd_A", round(rmsd, 4))
    print("ligand_atoms", len(ligand_xyz_aligned))
    print("ligand_centroid_aligned", [round(x, 3) for x in ligand_centroid])
    print("reference_shell_count", len(ref_shell))
    print("wrote", report_path)
    print("wrote", transformed_pdb)
    print()
    print("Top 10 sites by best aligned DCC:")
    for r in results[:10]:
        b = r["best_centroid_view"]
        print(
            r["site_id"],
            "rank=", r.get("rank"),
            "view=", b["view"] if b else None,
            "DCC=", round(b["dcc_to_ligand_centroid_A"], 3) if b else None,
            "grade=", b["dcc_grade"] if b else None,
            "min_atom=", round(b["min_distance_to_ligand_atom_A"], 3) if b else None,
            "overlap=", r["contact_shell_overlap"]["overlap_count"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
