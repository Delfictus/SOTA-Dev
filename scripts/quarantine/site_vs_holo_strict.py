#!/usr/bin/env python3
"""Generalized strict site-vs-holo spatial verification.

Locked rules:
  * Residue convention: ONE_INDEXED. topology_resnum is 1-indexed in PRISM
    (per CLAUDE.md offsets). pdb_resid = residue_map[topology_index = tid-1].
  * Alignment: dual-pass. Global (all common Cα) AND local (common Cα within
    15 Å of the apo pocket centroid in apo frame). Both reported.
  * Residue overlap cutoff: 4.5 Å (holo residue within 4.5 Å of any ligand atom).
  * HARD_NOT_MATCH rule (overrides all others):
        if min(centroid_distance_local, centroid_distance_global) > 30 Å
           AND overlap_fraction < 0.1
        → verdict = HARD_NOT_MATCH
  * NOT_MATCH:  centroid_distance > 20 Å AND overlap_fraction < 0.1
  * MATCH:      centroid_distance < 8 Å AND overlap_fraction > 0.3
  * else PARTIAL_OR_AMBIGUOUS
  * No auto-picking of anything. No threshold-dependent hedging.

Read-only. Writes report to /tmp/<target_key>_strict.json.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, Superimposer

LIGAND_CONTACT_CUTOFF = 4.5
LOCAL_RADIUS = 15.0
HARD_DISTANCE = 30.0

TARGETS = {
    "wrn_1522": dict(
        target_dir="wrn_apo",
        apo_code="6yhr", holo_code="8pfo",
        ligand_resname="YHC", site_id=1522, chain="A",
    ),
    "smarca2_site2": dict(
        target_dir="smarca2_brd_apo",
        apo_code="4qy4", holo_code="5dkc",
        ligand_resname="5BW", site_id=2, chain="A",
    ),
    "menin_3529": dict(
        target_dir="menin_apo",
        apo_code="3re2", holo_code="7uj4",
        ligand_resname="OQ4", site_id=3529, chain="A",
    ),
}


def chain_ca(structure, ch):
    m = {}
    for model in structure:
        for chain in model:
            if chain.id != ch:
                continue
            for r in chain:
                if r.id[0] == " " and "CA" in r:
                    m[r.id[1]] = (r.resname, r["CA"])
        break
    return m


def apo_atoms_by_res(structure, ch):
    m = {}
    for model in structure:
        for chain in model:
            if chain.id != ch:
                continue
            for r in chain:
                if r.id[0] == " ":
                    m[r.id[1]] = list(r)
        break
    return m


def holo_ligand_coords(structure, resname):
    coords = []
    for model in structure:
        for chain in model:
            for r in chain:
                if r.resname.strip() == resname:
                    for a in r:
                        coords.append(a.coord)
    return np.asarray(coords)


def detect_ligand_if_unset(structure, override_resname: str | None):
    if override_resname:
        return override_resname
    DENY = {"HOH", "ZN", "MG", "CA", "NA", "K", "CL", "SO4", "PO4", "GOL", "EDO", "PEG"}
    counts = {}
    for model in structure:
        for chain in model:
            for r in chain:
                if r.id[0].startswith("H") and r.id[0] != " ":
                    resname = r.resname.strip()
                    if resname in DENY or resname.startswith("H"):
                        if resname in DENY:
                            continue
                    counts[resname] = counts.get(resname, 0) + sum(1 for _ in r)
    if not counts:
        raise SystemExit("FAIL: no ligand and no override")
    return max(counts.items(), key=lambda kv: kv[1])[0]


def resolve_residues_one_indexed(top_ids, rmap_path, chain):
    rmap = json.loads(rmap_path.read_text())
    by_idx = {r["topology_index"]: r for r in rmap["residues"]}
    out = []
    for tid in top_ids:
        r = by_idx.get(tid - 1)
        if r is not None and r["chain"] == chain:
            out.append((chain, r["pdb_resid"], r["resname"]))
    return out


def pocket_record(path, pid):
    d = json.loads(path.read_text())
    for p in d["merged_pockets"]:
        if p.get("pocket_id") == pid:
            return p
    raise SystemExit(f"FAIL: pocket_id={pid} not in {path}")


def do_alignment(apo_ca, holo_ca, resid_subset=None):
    common = sorted(set(apo_ca) & set(holo_ca))
    if resid_subset is not None:
        common = [r for r in common if r in resid_subset]
    if len(common) < 20:
        return None, common
    sup = Superimposer()
    sup.set_atoms([holo_ca[r][1] for r in common], [apo_ca[r][1] for r in common])
    return sup, common


def run(target_key: str):
    cfg = TARGETS[target_key]
    base = Path(f"/mnt/storage/prism-outputs/twin-10-patent/{cfg['target_dir']}")
    apo_pdb = base / f"artifacts/1_download/{cfg['apo_code']}.pdb"
    holo_pdb = base / f"artifacts/1_download/{cfg['holo_code']}.pdb"
    rerank = base / "artifacts/6_rerank/rerank_result.json"
    rmap = base / f"artifacts/3_prep/{cfg['apo_code']}.residue_map.json"
    chain = cfg["chain"]

    parser = PDBParser(QUIET=True)
    apo = parser.get_structure("apo", str(apo_pdb))
    holo = parser.get_structure("holo", str(holo_pdb))

    apo_ca = chain_ca(apo, chain)
    holo_ca = chain_ca(holo, chain)
    apo_atoms = apo_atoms_by_res(apo, chain)

    lig_resname = detect_ligand_if_unset(holo, cfg["ligand_resname"])
    lig_coords = holo_ligand_coords(holo, lig_resname)
    if lig_coords.size == 0:
        raise SystemExit(f"FAIL: ligand {lig_resname} not found in {holo_pdb.name}")
    lig_centroid = lig_coords.mean(axis=0)

    p = pocket_record(rerank, cfg["site_id"])
    apo_site_centroid = np.asarray(p["centroid_spike_weighted"], dtype=float)

    resolved = resolve_residues_one_indexed(p["top_residue_ids"], rmap, chain)
    site_resids = sorted({r for _, r, _ in resolved})

    sup_g, common_g = do_alignment(apo_ca, holo_ca)
    if sup_g is None:
        raise SystemExit("FAIL: global alignment — too few common Cα")
    rmsd_g = float(sup_g.rms)
    centroid_in_holo_g = sup_g.rotran[0].dot(apo_site_centroid) + sup_g.rotran[1]
    dcc_g = float(np.linalg.norm(centroid_in_holo_g - lig_centroid))

    local_resids = {rid for rid, (_, ca) in apo_ca.items()
                    if float(np.linalg.norm(ca.coord - apo_site_centroid)) <= LOCAL_RADIUS}
    sup_l, common_l = do_alignment(apo_ca, holo_ca, local_resids)
    if sup_l is None:
        rmsd_l = None
        centroid_in_holo_l = None
        dcc_l = None
    else:
        rmsd_l = float(sup_l.rms)
        centroid_in_holo_l = sup_l.rotran[0].dot(apo_site_centroid) + sup_l.rotran[1]
        dcc_l = float(np.linalg.norm(centroid_in_holo_l - lig_centroid))

    site_atoms_in_holo_g = []
    for rid in site_resids:
        for a in apo_atoms.get(rid, []):
            c = sup_g.rotran[0].dot(a.coord) + sup_g.rotran[1]
            site_atoms_in_holo_g.append(c)
    arr_g = np.asarray(site_atoms_in_holo_g) if site_atoms_in_holo_g else np.empty((0, 3))
    min_atom_g = float(np.linalg.norm(arr_g[:, None, :] - lig_coords[None, :, :], axis=-1).min()) if arr_g.size else float("inf")

    site_atoms_in_holo_l = []
    if sup_l is not None:
        for rid in site_resids:
            for a in apo_atoms.get(rid, []):
                c = sup_l.rotran[0].dot(a.coord) + sup_l.rotran[1]
                site_atoms_in_holo_l.append(c)
    arr_l = np.asarray(site_atoms_in_holo_l) if site_atoms_in_holo_l else np.empty((0, 3))
    min_atom_l = float(np.linalg.norm(arr_l[:, None, :] - lig_coords[None, :, :], axis=-1).min()) if arr_l.size else float("inf")

    holo_atoms_by_res = {}
    for model in holo:
        for chobj in model:
            if chobj.id != chain:
                continue
            for r in chobj:
                if r.id[0] == " ":
                    holo_atoms_by_res[r.id[1]] = list(r)
        break

    contact_resids = set()
    for rid, atoms in holo_atoms_by_res.items():
        for a in atoms:
            d = float(np.min(np.linalg.norm(lig_coords - a.coord, axis=1)))
            if d <= LIGAND_CONTACT_CUTOFF:
                contact_resids.add(rid)
                break

    overlap = sorted(set(site_resids) & contact_resids)
    overlap_count = len(overlap)
    overlap_fraction = (overlap_count / len(contact_resids)) if contact_resids else 0.0

    anchor = []
    for rid in contact_resids:
        if rid in holo_ca:
            d = float(np.linalg.norm(centroid_in_holo_g - holo_ca[rid][1].coord))
            anchor.append((rid, d, holo_ca[rid][0]))
    anchor.sort(key=lambda t: t[1])
    top3 = anchor[:3]

    min_dcc = min([d for d in (dcc_l, dcc_g) if d is not None]) if dcc_g is not None else dcc_g
    if min_dcc > HARD_DISTANCE and overlap_fraction < 0.1:
        verdict = "HARD_NOT_MATCH"
    elif dcc_g > 20.0 and overlap_fraction < 0.1:
        verdict = "NOT_MATCH"
    elif dcc_g < 8.0 and overlap_fraction > 0.3:
        verdict = "MATCH"
    else:
        verdict = "PARTIAL_OR_AMBIGUOUS"

    rec = {
        "target_key": target_key,
        "apo_pdb": str(apo_pdb),
        "holo_pdb": str(holo_pdb),
        "ligand_resname": lig_resname,
        "site_id": cfg["site_id"],
        "residue_convention_used": "one_indexed",
        "alignment": {
            "global": {
                "n_common_ca": len(common_g),
                "rmsd_angstrom": round(rmsd_g, 3),
                "centroid_distance_angstrom": round(dcc_g, 3),
                "min_atom_distance_angstrom": round(min_atom_g, 3),
            },
            "local_15A_around_site_centroid": {
                "n_common_ca": len(common_l) if sup_l is not None else 0,
                "rmsd_angstrom": round(rmsd_l, 3) if rmsd_l is not None else None,
                "centroid_distance_angstrom": round(dcc_l, 3) if dcc_l is not None else None,
                "min_atom_distance_angstrom": round(min_atom_l, 3) if min_atom_l != float("inf") else None,
            },
        },
        "centroid_distance_global": round(dcc_g, 3),
        "centroid_distance_local": round(dcc_l, 3) if dcc_l is not None else None,
        "ligand_centroid": [round(x, 3) for x in lig_centroid.tolist()],
        "site_centroid_apo_frame": [round(x, 3) for x in apo_site_centroid.tolist()],
        "site_centroid_holo_frame_global": [round(x, 3) for x in centroid_in_holo_g.tolist()],
        "site_centroid_holo_frame_local": [round(x, 3) for x in centroid_in_holo_l.tolist()] if centroid_in_holo_l is not None else None,
        "ligand_contact_cutoff_angstrom": LIGAND_CONTACT_CUTOFF,
        "ligand_contact_residue_count": len(contact_resids),
        "ligand_contact_residues": sorted(contact_resids),
        "site_residue_count": len(site_resids),
        "site_residues": site_resids,
        "overlap_count": overlap_count,
        "overlap_fraction": round(overlap_fraction, 3),
        "overlap_residues": overlap,
        "nearest_residues": [r for r, _, _ in top3],
        "nearest_distances": [round(d, 3) for _, d, _ in top3],
        "nearest_resnames": [rn for _, _, rn in top3],
        "engine_rank": p.get("engine_rank"),
        "rerank_rank": p.get("rerank_rank"),
        "rank_shift": p.get("rank_shift"),
        "therm_class": p.get("therm_class"),
        "composite": p.get("rerank_composite"),
        "verdict": verdict,
        "rules": {
            "HARD_NOT_MATCH": "min(dcc_local, dcc_global) > 30 Å AND overlap_fraction < 0.1 (overrides)",
            "NOT_MATCH": "dcc_global > 20 Å AND overlap_fraction < 0.1",
            "MATCH": "dcc_global < 8 Å AND overlap_fraction > 0.3",
            "PARTIAL_OR_AMBIGUOUS": "otherwise",
        },
    }
    out_path = Path(f"/tmp/{target_key}_strict.json")
    out_path.write_text(json.dumps(rec, indent=2))

    print("═" * 72)
    print(f"STRICT VERIFICATION: {target_key}")
    print("═" * 72)
    print(f"  apo = {apo_pdb.name}   holo = {holo_pdb.name}   ligand = {lig_resname}")
    print(f"  residue_convention_used = one_indexed (locked)")
    print(f"  site_id = {cfg['site_id']}   therm_class = {p.get('therm_class')}   composite = {round(p.get('rerank_composite',0),3)}")
    print(f"  engine_rank = {p.get('engine_rank')}   rerank_rank = {p.get('rerank_rank')}   rank_shift = {p.get('rank_shift')}")
    print()
    print(f"  ALIGNMENT dual-pass:")
    print(f"    global: n={len(common_g)}  RMSD={rmsd_g:.3f} Å   centroid_distance={dcc_g:.3f} Å   min_atom={min_atom_g:.3f} Å")
    if sup_l is None:
        print(f"    local:  insufficient common Cα within {LOCAL_RADIUS} Å")
    else:
        print(f"    local:  n={len(common_l)}  RMSD={rmsd_l:.3f} Å   centroid_distance={dcc_l:.3f} Å   min_atom={min_atom_l:.3f} Å")
    print()
    print(f"  ligand_contact_residue_count = {len(contact_resids)}   site_residue_count = {len(site_resids)}")
    print(f"  overlap_count = {overlap_count}   overlap_fraction = {overlap_fraction:.3f}")
    print(f"  nearest_residues = {[r for r,_,_ in top3]}   distances = {[round(d,2) for _,d,_ in top3]}")
    print()
    print(f"  VERDICT = {verdict}")
    print("═" * 72)
    print(f"  report: {out_path}")
    print()
    print("RAW OUTPUT BLOCK")
    print(json.dumps(rec, indent=2))

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target_key", choices=sorted(TARGETS.keys()))
    args = ap.parse_args()
    run(args.target_key)


if __name__ == "__main__":
    main()
