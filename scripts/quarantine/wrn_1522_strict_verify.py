#!/usr/bin/env python3
"""Strict verification: WRN site 1522 vs 8PFO/HRO761 (YHC).

Follows NON-NEGOTIABLE VERIFICATION MODE spec exactly. No narrative.
Read-only; writes only /tmp report + stdout metrics.
"""
import json
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, Superimposer

APO_PDB = Path("/mnt/storage/prism-outputs/twin-10-patent/wrn_apo/artifacts/1_download/6yhr.pdb")
HOLO_PDB = Path("/mnt/storage/prism-outputs/twin-10-patent/wrn_apo/artifacts/1_download/8pfo.pdb")
RERANK = Path("/mnt/storage/prism-outputs/twin-10-patent/wrn_apo/artifacts/6_rerank/rerank_result.json")
RESIDUE_MAP = Path("/mnt/storage/prism-outputs/twin-10-patent/wrn_apo/artifacts/3_prep/6yhr.residue_map.json")
LIGAND_RESNAME = "YHC"
LIGAND_CONTACT_CUTOFF = 4.5
SITE_ID = 1522

REPORT_OUT = Path("/tmp/wrn_1522_strict_verify.json")


def chain_ca_atoms(structure, chain_id: str = "A"):
    out = {}
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] == " " and "CA" in res:
                    out[res.id[1]] = (res.resname, res["CA"])
        break
    return out


def apo_chain_atoms(structure, chain_id: str = "A"):
    per_res_atoms = {}
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] != " ":
                    continue
                per_res_atoms[res.id[1]] = [atom for atom in res]
        break
    return per_res_atoms


def holo_chain_atoms(structure, chain_id: str = "A"):
    per_res_atoms = {}
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if res.id[0] != " ":
                    continue
                per_res_atoms[res.id[1]] = {"resname": res.resname, "atoms": [a for a in res]}
        break
    return per_res_atoms


def ligand_coords(structure, resname: str) -> np.ndarray:
    coords = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.resname.strip() == resname:
                    for a in res:
                        coords.append(a.coord)
    if not coords:
        print("FAIL: ligand not found", file=sys.stderr)
        sys.exit(1)
    return np.asarray(coords)


def pocket_record(path: Path, pid: int) -> dict:
    d = json.loads(path.read_text())
    for p in d["merged_pockets"]:
        if p.get("pocket_id") == pid:
            return p
    print("FAIL: pocket not found", file=sys.stderr)
    sys.exit(1)


def resolve_top_residues(top_ids, rmap_path):
    rmap = json.loads(rmap_path.read_text())
    by_idx = {r["topology_index"]: r for r in rmap["residues"]}
    zero = []
    one = []
    for tid in top_ids:
        r0 = by_idx.get(tid)
        r1 = by_idx.get(tid - 1)
        if r0:
            zero.append(("A", r0["pdb_resid"], r0["resname"]))
        if r1:
            one.append(("A", r1["pdb_resid"], r1["resname"]))
    return zero, one


def pick_convention(zero_resolved, one_resolved, apo_atoms, apo_centroid):
    def mean_ca_dist(resolved):
        ds = []
        for _, rid, _ in resolved:
            atoms = apo_atoms.get(rid, [])
            for a in atoms:
                if a.get_name() == "CA":
                    ds.append(float(np.linalg.norm(a.coord - apo_centroid)))
                    break
        return (np.mean(ds) if ds else float("inf")), len(ds)

    m0, n0 = mean_ca_dist(zero_resolved)
    m1, n1 = mean_ca_dist(one_resolved)
    if m0 <= m1:
        return "zero_indexed", zero_resolved, m0, n0, m1, n1
    return "one_indexed", one_resolved, m1, n1, m0, n0


def main():
    parser = PDBParser(QUIET=True)
    apo = parser.get_structure("apo", str(APO_PDB))
    holo = parser.get_structure("holo", str(HOLO_PDB))

    apo_ca = chain_ca_atoms(apo, "A")
    holo_ca = chain_ca_atoms(holo, "A")
    common = sorted(set(apo_ca) & set(holo_ca))
    sup = Superimposer()
    sup.set_atoms([holo_ca[r][1] for r in common], [apo_ca[r][1] for r in common])
    rmsd = float(sup.rms)
    n_align = len(common)

    print("STEP 1 — ALIGNMENT (global Cα)")
    print(f"  alignment_rmsd_A = {rmsd:.3f}")
    print(f"  n_residues_used  = {n_align}")
    if rmsd > 5.0:
        print("  WARNING: alignment_rmsd > 5 Å")
    print()

    lig_coords = ligand_coords(holo, LIGAND_RESNAME)
    lig_centroid = lig_coords.mean(axis=0)
    print("STEP 2 — LIGAND CENTROID")
    print(f"  ligand_centroid_xyz = {lig_centroid.tolist()}")
    print()

    p = pocket_record(RERANK, SITE_ID)
    site_centroid_apo_frame = np.asarray(p["centroid_spike_weighted"], dtype=float)
    site_centroid_holo_frame = sup.rotran[0].dot(site_centroid_apo_frame) + sup.rotran[1]
    print("STEP 3 — SITE 1522 CENTROID")
    print(f"  site_centroid_apo_frame   = {site_centroid_apo_frame.tolist()}")
    print(f"  site_centroid_holo_frame  = {site_centroid_holo_frame.tolist()}")
    print()

    apo_atoms_by_res = apo_chain_atoms(apo, "A")
    holo_atoms_by_res = holo_chain_atoms(holo, "A")

    zero_resolved, one_resolved = resolve_top_residues(p["top_residue_ids"], RESIDUE_MAP)
    convention, resolved, best_m, best_n, other_m, other_n = pick_convention(
        zero_resolved, one_resolved, apo_atoms_by_res, site_centroid_apo_frame
    )
    print("STEP 3b — RESIDUE CONVENTION RESOLUTION (disambiguation)")
    print(f"  zero_indexed_mean_ca_to_centroid_A = {(best_m if convention=='zero_indexed' else other_m):.3f}  n={ (best_n if convention=='zero_indexed' else other_n) }")
    print(f"  one_indexed_mean_ca_to_centroid_A  = {(best_m if convention=='one_indexed' else other_m):.3f}  n={ (best_n if convention=='one_indexed' else other_n) }")
    print(f"  selected_convention = {convention}")
    print()

    site_resids = sorted({rid for _, rid, _ in resolved})
    # transform apo atom coords into holo frame
    site_atom_coords_holo = []
    for rid in site_resids:
        for a in apo_atoms_by_res.get(rid, []):
            c = sup.rotran[0].dot(a.coord) + sup.rotran[1]
            site_atom_coords_holo.append(c)
    site_atom_coords_holo = np.asarray(site_atom_coords_holo) if site_atom_coords_holo else np.empty((0, 3))

    centroid_distance = float(np.linalg.norm(site_centroid_holo_frame - lig_centroid))
    if site_atom_coords_holo.size == 0:
        min_atom_distance = float("inf")
    else:
        dists = np.linalg.norm(site_atom_coords_holo[:, None, :] - lig_coords[None, :, :], axis=-1)
        min_atom_distance = float(dists.min())

    print("STEP 4 — DISTANCE METRICS")
    print(f"  centroid_distance_angstrom = {centroid_distance:.3f}")
    print(f"  min_atom_distance_angstrom = {min_atom_distance:.3f}")
    print()

    ligand_contact_residues = set()
    for rid, info in holo_atoms_by_res.items():
        atoms = info["atoms"]
        for a in atoms:
            d = float(np.min(np.linalg.norm(lig_coords - a.coord, axis=1)))
            if d <= LIGAND_CONTACT_CUTOFF:
                ligand_contact_residues.add(rid)
                break

    site_resid_set = set(site_resids)
    overlap_set = site_resid_set & ligand_contact_residues
    overlap_count = len(overlap_set)
    overlap_fraction = (overlap_count / len(ligand_contact_residues)) if ligand_contact_residues else 0.0

    print("STEP 5 — RESIDUE OVERLAP (4.5 Å ligand-contact cutoff)")
    print(f"  ligand_contact_residue_count = {len(ligand_contact_residues)}")
    print(f"  site_residue_count           = {len(site_resid_set)}")
    print(f"  overlap_count                = {overlap_count}")
    print(f"  overlap_fraction             = {overlap_fraction:.3f}")
    print(f"  ligand_contact_residues      = {sorted(ligand_contact_residues)}")
    print(f"  site_residues                = {sorted(site_resid_set)}")
    print(f"  overlap_residues             = {sorted(overlap_set)}")
    print()

    anchor_dists = []
    for rid in ligand_contact_residues:
        if rid in holo_ca:
            ca_coord = holo_ca[rid][1].coord
            d = float(np.linalg.norm(site_centroid_holo_frame - ca_coord))
            anchor_dists.append((rid, d, holo_ca[rid][0]))
    anchor_dists.sort(key=lambda t: t[1])
    top3 = anchor_dists[:3]

    print("STEP 6 — ANCHOR DISTANCES (top-3 nearest ligand-contact Cα)")
    print(f"  nearest_residue_ids = {[r for r,_,_ in top3]}")
    print(f"  nearest_distances   = {[round(d,3) for _,d,_ in top3]}")
    print(f"  nearest_resnames    = {[rn for _,_,rn in top3]}")
    print()

    if centroid_distance > 20.0 and overlap_fraction < 0.1:
        verdict = "NOT_MATCH"
    elif centroid_distance < 8.0 and overlap_fraction > 0.3:
        verdict = "MATCH"
    else:
        verdict = "PARTIAL_OR_AMBIGUOUS"

    print("STEP 7 — FINAL CLASSIFICATION")
    print(f"  verdict = {verdict}")
    print()

    engine_rank = p.get("engine_rank")
    rerank_rank = p.get("rerank_rank")
    rank_shift = p.get("rank_shift")
    therm_class = p.get("therm_class")
    composite = p.get("rerank_composite")

    print("STEP 8 — RANK CONTEXT")
    print(f"  engine_rank     = {engine_rank}")
    print(f"  rerank_rank     = {rerank_rank}")
    print(f"  rank_shift      = {rank_shift}")
    print(f"  therm_class     = {therm_class}")
    print(f"  composite_score = {composite}")
    print()

    record = {
        "alignment_rmsd": round(rmsd, 3),
        "alignment_n_residues": n_align,
        "alignment_window": "global_chainA_all_common_CA",
        "ligand_resname": LIGAND_RESNAME,
        "ligand_centroid": [round(x, 3) for x in lig_centroid.tolist()],
        "site_id": SITE_ID,
        "site_centroid_apo_frame": [round(x, 3) for x in site_centroid_apo_frame.tolist()],
        "site_centroid_holo_frame": [round(x, 3) for x in site_centroid_holo_frame.tolist()],
        "site_centroid": [round(x, 3) for x in site_centroid_holo_frame.tolist()],
        "centroid_distance": round(centroid_distance, 3),
        "min_atom_distance": round(min_atom_distance, 3),
        "ligand_contact_cutoff_angstrom": LIGAND_CONTACT_CUTOFF,
        "ligand_contact_residues": sorted(ligand_contact_residues),
        "site_residues": sorted(site_resid_set),
        "overlap_count": overlap_count,
        "overlap_fraction": round(overlap_fraction, 3),
        "overlap_residues": sorted(overlap_set),
        "nearest_residues": [r for r, _, _ in top3],
        "nearest_distances": [round(d, 3) for _, d, _ in top3],
        "nearest_resnames": [rn for _, _, rn in top3],
        "residue_convention_used": convention,
        "engine_rank": engine_rank,
        "rerank_rank": rerank_rank,
        "rank_shift": rank_shift,
        "therm_class": therm_class,
        "composite": composite,
        "verdict": verdict,
    }
    REPORT_OUT.write_text(json.dumps(record, indent=2))
    print("STEP 9 — RAW OUTPUT BLOCK")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
