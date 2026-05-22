#!/usr/bin/env python3
"""Validate 7c8r v2 candidates against TG3 (the bound 3CL protease inhibitor)."""
import json
import sys
from Bio.PDB import PDBParser

PDB_FILE = "/home/diddy/.cache/prism4d/holo_pdbs/7c8r.pdb"
LIGAND_RESNAME = "TG3"
TARGET_CHAIN = "A"
CANDIDATES_JSON = "/mnt/storage/7c8r_dimer_discovery_v9c_prime_20260508_064301/candidate_region_extracted/binding_site_candidates.v2_ranked.json"
CUTOFF = 8.0


def get_contact_shell():
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("holo", PDB_FILE)
    ligand_atoms = []
    protein_atoms = []
    for model in structure:
        for chain in model:
            if chain.id != TARGET_CHAIN:
                continue
            for residue in chain:
                if residue.resname == LIGAND_RESNAME:
                    ligand_atoms.extend(residue.get_atoms())
                elif residue.id[0] == " ":
                    protein_atoms.extend(residue.get_atoms())
    if not ligand_atoms:
        print(f"Error: ligand {LIGAND_RESNAME} not found in chain {TARGET_CHAIN} of {PDB_FILE}", file=sys.stderr)
        sys.exit(1)
    contact = set()
    for pa in protein_atoms:
        for la in ligand_atoms:
            if pa - la <= CUTOFF:
                contact.add(pa.get_parent().id[1])
                break
    return sorted(contact), len(ligand_atoms)


def main():
    shell_res, n_atoms = get_contact_shell()
    gt = set(shell_res)
    print(f"Target: 7C8R | Chain: {TARGET_CHAIN} | Ligand: {LIGAND_RESNAME} ({n_atoms} atoms)")
    print(f"Ground-truth contact shell ({CUTOFF} A): {len(gt)} residues")
    print(f"Shell: {sorted(gt)}")
    print()

    data = json.load(open(CANDIDATES_JSON))
    cands = sorted(data.get("candidates", []), key=lambda x: x.get("v2_score", 0), reverse=True)

    hdr = f"{'rk':>3} | {'candidate':<24} | {'v2_score':>10} | {'P':>5} | {'R':>5} | {'F1':>5} | TP"
    print(hdr)
    print("-" * len(hdr))
    for i, c in enumerate(cands[:15], 1):
        pred = set(c.get("residue_ids", []))
        if not pred:
            continue
        tp = pred & gt
        P = len(tp) / len(pred) if pred else 0.0
        R = len(tp) / len(gt) if gt else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        print(f"{i:>3} | {c.get('candidate_id', '?'):<24} | {int(c.get('v2_score', 0)):>10} | {P:5.2f} | {R:5.2f} | {F1:5.2f} | {sorted(tp)}")


if __name__ == "__main__":
    main()
