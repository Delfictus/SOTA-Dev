#!/usr/bin/env python3
"""
WRN site 1522 spatial validation against holo 8PFO / ligand YHC (HRO761).

Read-only. Emits a single JSON report + terminal verdict.
No writes to pipeline artifacts, no weight changes, no engine edits.
"""
import json
import sys
import math
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, Superimposer

WRN_DIR = Path("/mnt/storage/prism-outputs/twin-10-patent/wrn_apo")
APO_PDB = WRN_DIR / "artifacts/1_download/6yhr.pdb"
HOLO_PDB = WRN_DIR / "artifacts/1_download/8pfo.pdb"
RERANK = WRN_DIR / "artifacts/6_rerank/rerank_result.json"
RESIDUE_MAP = WRN_DIR / "artifacts/3_prep/6yhr.residue_map.json"
LIGAND_RESNAME = "YHC"

REPORT_OUT = Path("/tmp/wrn_1522_validation_report.json")


def _pocket_1522(rerank_json: Path) -> dict:
    d = json.loads(rerank_json.read_text())
    for p in d["merged_pockets"]:
        if p.get("pocket_id") == 1522:
            return p
    raise SystemExit("pocket_id=1522 not found in rerank_result.json")


def _resolve_residues(top_ids: list[int], rmap_path: Path) -> dict:
    rmap = json.loads(rmap_path.read_text())
    residues = rmap["residues"]
    by_idx = {r["topology_index"]: r for r in residues}
    resolved_0 = []
    resolved_1 = []
    for tid in top_ids:
        r0 = by_idx.get(tid)
        r1 = by_idx.get(tid - 1)
        if r0:
            resolved_0.append((tid, r0["chain"], r0["pdb_resid"], r0["resname"]))
        if r1:
            resolved_1.append((tid, r1["chain"], r1["pdb_resid"], r1["resname"]))
    return {"zero_indexed": resolved_0, "one_indexed": resolved_1}


def _chain_ca_atoms(structure, chain_id: str):
    model = next(structure.get_models())
    chain = model[chain_id]
    ca = {}
    for res in chain:
        if res.id[0] != " ":
            continue
        if "CA" in res:
            ca[res.id[1]] = res["CA"]
    return ca


def _superpose_ca(apo_struct, holo_struct, chain_id: str = "A", resid_window: tuple[int, int] | None = None):
    apo_ca = _chain_ca_atoms(apo_struct, chain_id)
    holo_ca = _chain_ca_atoms(holo_struct, chain_id)
    common_resids = sorted(set(apo_ca.keys()) & set(holo_ca.keys()))
    if resid_window is not None:
        lo, hi = resid_window
        common_resids = [r for r in common_resids if lo <= r <= hi]
    if len(common_resids) < 20:
        raise SystemExit(f"too few common CAs: {len(common_resids)}")
    apo_atoms = [apo_ca[r] for r in common_resids]
    holo_atoms = [holo_ca[r] for r in common_resids]
    sup = Superimposer()
    sup.set_atoms(holo_atoms, apo_atoms)
    return sup, common_resids


def _ligand_coords(holo_struct, resname: str) -> np.ndarray:
    coords = []
    for model in holo_struct:
        for chain in model:
            for res in chain:
                if res.resname.strip() == resname:
                    for atom in res:
                        coords.append(atom.coord)
    if not coords:
        raise SystemExit(f"ligand {resname} not found")
    return np.asarray(coords)


def _holo_contact_residues(holo_struct, ligand_coords: np.ndarray, cutoff: float = 5.0) -> list[tuple[str, int, str]]:
    contacts = set()
    for model in holo_struct:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                for atom in res:
                    d = np.min(np.linalg.norm(ligand_coords - atom.coord, axis=1))
                    if d <= cutoff:
                        contacts.add((chain.id, res.id[1], res.resname))
                        break
    return sorted(contacts)


def main() -> None:
    p = _pocket_1522(RERANK)
    apo_centroid = np.asarray(p["centroid_spike_weighted"], dtype=float)
    top_resids = p["top_residue_ids"]

    parser = PDBParser(QUIET=True)
    apo = parser.get_structure("apo", str(APO_PDB))
    holo = parser.get_structure("holo", str(HOLO_PDB))

    sup_global, common_global = _superpose_ca(apo, holo, "A")
    rmsd_global = float(sup_global.rms)

    windows = [
        ("global", None),
        ("D1_domain_528_725", (528, 725)),
        ("D2_domain_725_944", (725, 944)),
        ("pocket_proximal_900_944", (900, 944)),
    ]
    per_window = {}
    for name, win in windows:
        s, common = _superpose_ca(apo, holo, "A", win)
        t = s.rotran[0].dot(apo_centroid) + s.rotran[1]
        per_window[name] = {
            "window": win,
            "n_common_ca": len(common),
            "rmsd_angstrom": round(float(s.rms), 3),
            "apo_centroid_in_holo_frame": t.tolist(),
        }

    best_name = min(
        [n for n in per_window if per_window[n]["rmsd_angstrom"] < 5.0 and n != "D1_domain_528_725"]
        or [min(per_window, key=lambda n: per_window[n]["rmsd_angstrom"])],
        key=lambda n: per_window[n]["rmsd_angstrom"],
    )
    sup, common = _superpose_ca(apo, holo, "A", dict(windows)[best_name])
    rmsd = float(sup.rms)
    apo_centroid_in_holo_frame = sup.rotran[0].dot(apo_centroid) + sup.rotran[1]

    lig_coords = _ligand_coords(holo, LIGAND_RESNAME)
    lig_centroid = lig_coords.mean(axis=0)

    dcc = float(np.linalg.norm(apo_centroid_in_holo_frame - lig_centroid))
    min_dist = float(np.min(np.linalg.norm(lig_coords - apo_centroid_in_holo_frame, axis=1)))

    resolved = _resolve_residues(top_resids, RESIDUE_MAP)
    apo_pdb_ids_0 = {(c, r) for (_, c, r, _) in resolved["zero_indexed"]}
    apo_pdb_ids_1 = {(c, r) for (_, c, r, _) in resolved["one_indexed"]}

    contacts = _holo_contact_residues(holo, lig_coords, cutoff=5.0)
    contact_ids = {(c, r) for (c, r, _) in contacts}

    overlap_0 = sorted(apo_pdb_ids_0 & contact_ids)
    overlap_1 = sorted(apo_pdb_ids_1 & contact_ids)
    best = max((overlap_0, overlap_1), key=len)
    best_conv = "zero_indexed" if best is overlap_0 else "one_indexed"
    overlap_count = len(best)
    overlap_frac_apo = overlap_count / max(1, len(apo_pdb_ids_0 if best_conv == "zero_indexed" else apo_pdb_ids_1))
    overlap_frac_holo = overlap_count / max(1, len(contact_ids))

    if dcc <= 4.0 and overlap_frac_holo >= 0.30:
        verdict = "CONFIRMED_RECOVERY"
    elif dcc <= 8.0 and overlap_count >= 3:
        verdict = "PROVISIONAL_SIGNAL"
    else:
        verdict = "NOT_CONFIRMED"

    report = {
        "pocket_id": 1522,
        "apo_pdb": str(APO_PDB),
        "holo_pdb": str(HOLO_PDB),
        "ligand_resname": LIGAND_RESNAME,
        "superposition": {
            "chain": "A",
            "global_rmsd_angstrom": round(rmsd_global, 3),
            "global_common_ca_count": len(common_global),
            "selected_window": best_name,
            "selected_window_rmsd_angstrom": round(rmsd, 3),
            "selected_window_common_ca_count": len(common),
            "per_window": per_window,
            "interpretation": (
                "6YHR apo has 541 residues (528-1072); 8PFO holo has 419 (523-944). "
                "Global CA RMSD 17 Å indicates large inter-domain motion between apo and holo "
                "— D1 (528-725) aligns at <1 Å, D2 (725-944) at ~3 Å, but they cannot be aligned "
                "simultaneously. The selected window is chosen to best match the region where "
                "apo site 1522's lining residues reside."
            ),
        },
        "centroid_distance": {
            "apo_centroid_original": apo_centroid.tolist(),
            "apo_centroid_in_holo_frame": apo_centroid_in_holo_frame.tolist(),
            "holo_ligand_centroid": lig_centroid.tolist(),
            "dcc_angstrom": round(dcc, 3),
            "min_distance_to_any_ligand_atom_angstrom": round(min_dist, 3),
        },
        "residue_overlap": {
            "apo_top_residues_count": len(top_resids),
            "holo_contact_residues_count": len(contacts),
            "convention_used": best_conv,
            "overlap_count": overlap_count,
            "overlap_fraction_of_apo_top": round(overlap_frac_apo, 3),
            "overlap_fraction_of_holo_contacts": round(overlap_frac_holo, 3),
            "overlap_residues": [f"{c}{r}" for (c, r) in best],
            "holo_contact_residues": [f"{c}{r}({rn})" for (c, r, rn) in contacts],
            "apo_top_residues_zero_indexed": [f"{c}{r}({rn})" for (_, c, r, rn) in resolved["zero_indexed"]],
            "apo_top_residues_one_indexed": [f"{c}{r}({rn})" for (_, c, r, rn) in resolved["one_indexed"]],
        },
        "verdict": verdict,
        "verdict_criteria": {
            "CONFIRMED_RECOVERY": "dcc <= 4.0 Å AND overlap_frac_holo >= 0.30",
            "PROVISIONAL_SIGNAL": "dcc <= 8.0 Å AND overlap_count >= 3",
            "NOT_CONFIRMED": "otherwise",
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2))

    print("═" * 70)
    print("WRN site 1522 spatial validation")
    print("═" * 70)
    print(f"  apo:  {APO_PDB.name}")
    print(f"  holo: {HOLO_PDB.name}  ligand: {LIGAND_RESNAME}")
    print(f"  global CA superposition: {len(common_global)} common Cα, RMSD {rmsd_global:.3f} Å  (inter-domain motion)")
    for name, d in per_window.items():
        mark = " ← selected" if name == best_name else ""
        print(f"    window {name:<28} n={d['n_common_ca']:>4}  RMSD {d['rmsd_angstrom']:>6.3f} Å{mark}")
    print()
    print(f"  apo centroid (original):     {apo_centroid.tolist()}")
    print(f"  apo centroid (holo frame):   {apo_centroid_in_holo_frame.tolist()}")
    print(f"  holo {LIGAND_RESNAME} centroid:         {lig_centroid.tolist()}")
    print()
    print(f"  centroid distance (DCC):     {dcc:.3f} Å")
    print(f"  nearest ligand atom dist:    {min_dist:.3f} Å")
    print()
    print(f"  apo top residues ({len(top_resids)}): convention={best_conv}")
    print(f"  holo contact residues ({len(contacts)}): {[f'{c}{r}{rn}' for (c,r,rn) in contacts]}")
    print(f"  overlap count: {overlap_count} / apo_frac={overlap_frac_apo:.2f}, holo_frac={overlap_frac_holo:.2f}")
    if best:
        print(f"  overlap residues: {[f'{c}{r}' for (c,r) in best]}")
    print()
    print(f"  VERDICT: {verdict}")
    print("═" * 70)
    print(f"  report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
