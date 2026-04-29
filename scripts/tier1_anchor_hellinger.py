"""
Tier 1: MD Anchor Diagnostic + Hellinger Distribution Distance.

Pure post-processor module. Operates on already-emitted Arrow + topology
JSON + reference PDBs. NO_POST_MD_LOOPS-compliant: this is the Python
analysis layer; no Rust engine modifications.

  1a (diagnose_md_anchor):
        Replaces opaque 8-11Å "drift" reports with a named diagnosis:
        TIGHT_ANCHOR | NUMBERING_OFFSET_<N> | LOOSE_ANCHOR |
        COORDINATE_FRAME_SHIFT | GENUINE_DRIFT.

  1b (compute_pocket_reference_hellinger):
        Alignment-invariant pocket-to-reference comparison via Hellinger
        distance over residue-occupancy probability distributions. Adds
        Bhattacharyya overlap coefficient + Jensen-Shannon divergence
        as cross-validation metrics.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# =============================================================================
# Tier 1a — MD anchor diagnostic
# =============================================================================

def load_topology_atom_positions(md_pdb_path):
    """
    Returns dict: residue_id -> np.array of all heavy-atom coords for that residue.
    Operates on the MD-output PDB (chain A, all heavy atoms, alt-loc A or blank).
    """
    by_residue = defaultdict(list)
    if not md_pdb_path or not Path(md_pdb_path).exists():
        return {}
    with open(md_pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                atom_name = line[12:16].strip()
                alt_loc = line[16:17].strip()
                chain = line[21:22].strip()
                resseq = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]
            except (ValueError, IndexError):
                continue
            if element == "H" or atom_name.startswith("H"):
                continue
            if alt_loc not in ("", "A"):
                continue
            if chain != "A":
                continue
            by_residue[resseq].append([x, y, z])
    return {k: np.asarray(v) for k, v in by_residue.items()}


def load_residue_to_atom_map(topology_json_path):
    """
    Returns dict: residue_id (int) -> list[atom_index (int)]
    From G4-emitted topology JSON. Engine numbering.
    """
    if not topology_json_path or not Path(topology_json_path).exists():
        return {}
    with open(topology_json_path) as f:
        topo = json.load(f)
    rmap = topo.get("residue_to_atom_indices", {})
    return {int(k): list(v) for k, v in rmap.items()}


def diagnose_md_anchor(pocket_residues, spike_centroids_per_res,
                      md_atom_positions, residue_to_atom_map):
    """
    For each pocket residue, compare the spike-attributed centroid to the
    MD PDB atom positions for that same residue ID.

    Returns:
      {
        "status": "OK" | "BLOCKED",
        "per_residue": {residue_id: {centroid_to_atoms_min_A, ...}},
        "diagnosis": "NUMBERING_OFFSET_<N>" | "COORDINATE_FRAME_SHIFT" |
                     "TIGHT_ANCHOR" | "LOOSE_ANCHOR" | "GENUINE_DRIFT",
        "best_offset": int | None,
        "evidence": {...}
      }
    """
    if not md_atom_positions or not residue_to_atom_map:
        return {"status": "BLOCKED",
                "gate": "Need MD PDB and topology JSON",
                "result": None}

    per_residue = {}
    direct_match_offsets = []

    for rid in pocket_residues:
        if rid not in spike_centroids_per_res:
            continue
        centroid = np.asarray(spike_centroids_per_res[rid])

        # Direct match: assume engine_id == pdb_resseq
        if rid in md_atom_positions:
            atoms = md_atom_positions[rid]
            min_d = float(np.linalg.norm(atoms - centroid, axis=1).min())
            direct_match_offsets.append(min_d)
            per_residue[int(rid)] = {
                "centroid_xyz": centroid.tolist(),
                "n_atoms_in_md_pdb": len(atoms),
                "min_centroid_to_atom_A_direct": min_d,
            }

    if not per_residue:
        return {"status": "BLOCKED",
                "gate": "No pocket residues found in MD PDB",
                "result": None}

    direct_mean = float(np.mean(direct_match_offsets))
    direct_median = float(np.median(direct_match_offsets))

    # If direct match is tight, no further diagnostic needed.
    if direct_mean < 3.0:
        return {"status": "OK", "per_residue": per_residue,
                "diagnosis": "TIGHT_ANCHOR",
                "best_offset": 0,
                "evidence": {
                    "direct_match_mean_offset_A": direct_mean,
                    "direct_match_median_offset_A": direct_median,
                }}

    # Direct match is loose. Try numbering offsets ±20 to detect shift.
    best_offset = 0
    best_mean = direct_mean
    for off in range(-20, 21):
        if off == 0:
            continue
        offset_offsets = []
        for rid in pocket_residues:
            if rid not in spike_centroids_per_res:
                continue
            centroid = np.asarray(spike_centroids_per_res[rid])
            target_resseq = rid + off
            if target_resseq in md_atom_positions:
                atoms = md_atom_positions[target_resseq]
                offset_offsets.append(
                    float(np.linalg.norm(atoms - centroid, axis=1).min())
                )
        if offset_offsets and len(offset_offsets) >= len(pocket_residues) // 2:
            mean_off = float(np.mean(offset_offsets))
            if mean_off < best_mean:
                best_mean = mean_off
                best_offset = off

    if best_offset != 0 and best_mean < 3.0:
        diagnosis = f"NUMBERING_OFFSET_{best_offset:+d}"
    elif direct_mean < 6.0:
        diagnosis = "LOOSE_ANCHOR"
    elif direct_mean > 8.0 and best_mean > 8.0:
        diagnosis = "COORDINATE_FRAME_SHIFT"
    else:
        diagnosis = "GENUINE_DRIFT"

    return {
        "status": "OK", "per_residue": per_residue,
        "diagnosis": diagnosis,
        "best_offset": best_offset,
        "evidence": {
            "direct_match_mean_offset_A": direct_mean,
            "direct_match_median_offset_A": direct_median,
            "best_offset_mean_A": best_mean,
        },
    }


# =============================================================================
# Tier 1b — Hellinger residue-distribution distance
# =============================================================================

def pocket_occupancy_distribution(pocket_residue_energies, all_residues):
    """
    Build P_pocket(residue_id) over the union of all residues seen in run.
    Returns numpy array indexed by residue_id ordering.
    """
    total = sum(pocket_residue_energies.values())
    if total == 0:
        return np.zeros(len(all_residues))
    p = np.zeros(len(all_residues))
    for i, rid in enumerate(all_residues):
        p[i] = pocket_residue_energies.get(rid, 0.0) / total
    return p


def reference_contact_shell_distribution(ref_pdb_path, ligand_het_id,
                                          all_residues, cutoff_A=4.5,
                                          residue_offset=0):
    """
    Build P_ref(residue_id) from holo PDB: residues within cutoff of any
    ligand heavy atom, weighted by inverse distance.
    residue_offset: holo numbering - engine numbering (from Kabsch alignment).
    """
    res_atoms = defaultdict(list)
    lig_atoms = []
    with open(ref_pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                atom_name = line[12:16].strip()
                alt_loc = line[16:17].strip()
                res_name = line[17:20].strip()
                chain = line[21:22].strip()
                resseq = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]
            except (ValueError, IndexError):
                continue
            if element == "H" or atom_name.startswith("H"):
                continue
            if alt_loc not in ("", "A"):
                continue
            if chain != "A":
                continue
            if line.startswith("ATOM"):
                res_atoms[resseq].append([x, y, z])
            elif res_name == ligand_het_id:
                lig_atoms.append([x, y, z])

    if not lig_atoms:
        return np.zeros(len(all_residues))

    lig_xyz = np.asarray(lig_atoms)
    p = np.zeros(len(all_residues))

    for i, engine_rid in enumerate(all_residues):
        pdb_resseq = engine_rid + residue_offset
        if pdb_resseq not in res_atoms:
            continue
        res_xyz = np.asarray(res_atoms[pdb_resseq])
        d_min = np.linalg.norm(
            res_xyz[:, None, :] - lig_xyz[None, :, :], axis=-1
        ).min()
        if d_min < cutoff_A:
            # Inverse-distance weight (saturates at 1.0 for direct contact).
            p[i] = 1.0 / max(d_min, 1.0)

    if p.sum() > 0:
        p = p / p.sum()
    return p


def hellinger_distance(p, q):
    """
    Hellinger distance between two probability distributions.
    H(P, Q) = sqrt(1 - sum(sqrt(P*Q)))
    Range: [0, 1]. 0 = identical, 1 = orthogonal (zero overlap).
    """
    bc = float(np.sum(np.sqrt(p * q)))  # Bhattacharyya coefficient
    bc = min(max(bc, 0.0), 1.0)
    return float(np.sqrt(1.0 - bc))


def compute_pocket_reference_hellinger(pocket_residue_energies,
                                       references_dict, ref_pdb_dir,
                                       all_residues,
                                       residue_offset_per_ref):
    """
    Returns per-reference Hellinger distance + Bhattacharyya overlap +
    Jensen-Shannon divergence. Skip-with-status when ref PDB or ligand
    atoms are missing.
    """
    P_pocket = pocket_occupancy_distribution(pocket_residue_energies, all_residues)
    out = {}
    for pdb_id, meta in references_dict.items():
        ref_path = Path(ref_pdb_dir) / f"{pdb_id}.pdb"
        if not ref_path.exists():
            out[pdb_id] = {"status": "BLOCKED", "gate": f"PDB missing: {ref_path}"}
            continue
        offset = residue_offset_per_ref.get(pdb_id, 0)
        P_ref = reference_contact_shell_distribution(
            str(ref_path), meta["het"], all_residues,
            cutoff_A=4.5, residue_offset=offset,
        )
        if P_ref.sum() == 0:
            out[pdb_id] = {"status": "BLOCKED", "gate": "No ligand atoms in PDB"}
            continue
        H = hellinger_distance(P_pocket, P_ref)
        # Bhattacharyya overlap coefficient.
        bc = float(np.sum(np.sqrt(P_pocket * P_ref)))
        # Symmetrized KL (Jensen-Shannon, complementary metric).
        m = 0.5 * (P_pocket + P_ref)
        eps = 1e-12
        js = 0.5 * float(np.sum(P_pocket * np.log((P_pocket + eps) / (m + eps)))) + \
             0.5 * float(np.sum(P_ref * np.log((P_ref + eps) / (m + eps))))
        out[pdb_id] = {
            "status": "OK",
            "hellinger_distance": H,
            "bhattacharyya_overlap": bc,
            "jensen_shannon_divergence": js,
            "interpretation": (
                "STRONG_OVERLAP" if H < 0.3 else
                "MODERATE_OVERLAP" if H < 0.6 else
                "WEAK_OVERLAP" if H < 0.85 else
                "ORTHOGONAL_DISTRIBUTIONS"
            ),
        }
    return out
