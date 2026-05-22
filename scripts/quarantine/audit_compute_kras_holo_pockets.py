#!/usr/bin/env python3
"""[DIAGNOSTIC] Compute ligand-contact protein residues for 4 KRAS reference PDBs.

Substrate-cleanliness audit support: produces references/holo_pockets_engine_indexed.json
with per-reference ligand contact residue lists in (a) that reference PDB's own
residue numbering, and (b) annotated with the 4LPK offset (+1 from
topology.residue_id to pdb_resid) so the bridge can translate to engine-space.

Reads only PDB files in references/. Writes one JSON to references/.
No source files modified. No producer-side changes.
"""

import json
import sys
from pathlib import Path

# Per Pass 1.3 audit findings — confirmed in the PDBs themselves
LIGAND_CODES = {
    "6oim": ("MOV", "Sotorasib (AMG 510)"),
    "6ut0": ("M1X", "Adagrasib (MRTX849)"),
    "7rpz": ("6IC", "MRTX1133"),
    "6gj8": ("F0K", "BI-2852"),
}

CONTACT_THRESHOLD_A = 4.0


def parse_pdb(path):
    """Return (protein_atoms, hetatm_by_resname) where each is a list of
    (resid:int, atom_name:str, x:float, y:float, z:float, resname:str, chain:str).
    Skips altloc != ' ' or 'A' to match prism-clean's behavior.
    Heavy atoms only (drops element=H)."""
    protein, hetatm = [], {}
    with open(path) as fh:
        for line in fh:
            rec = line[0:6].rstrip()
            if rec not in ("ATOM", "HETATM"):
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
            if element == "H" or atom_name.startswith("H"):
                continue
            resname = line[17:20].strip()
            chain = line[21:22]
            try:
                resid = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            tup = (resid, atom_name, x, y, z, resname, chain)
            if rec == "ATOM":
                protein.append(tup)
            else:
                hetatm.setdefault(resname, []).append(tup)
    return protein, hetatm


def contacts_within(ligand_atoms, protein_atoms, threshold):
    """Return sorted set of unique protein resids whose any heavy atom is
    within `threshold` Angstroms of any ligand heavy atom."""
    t2 = threshold * threshold
    contacts = {}
    for la in ligand_atoms:
        lx, ly, lz = la[2], la[3], la[4]
        for pa in protein_atoms:
            dx = pa[2] - lx
            dy = pa[3] - ly
            dz = pa[4] - lz
            if dx*dx + dy*dy + dz*dz <= t2:
                resid = pa[0]
                contacts[resid] = pa[5]  # resid -> resname
    return contacts


def first_chain(protein):
    """Return the chain id of the first protein atom (proxy for 'KRAS chain A')."""
    return protein[0][6] if protein else None


def main():
    refs_dir = Path("references")
    out_path = refs_dir / "holo_pockets_engine_indexed.json"

    out = {
        "schema_version": "1.0.0",
        "generator": "scripts/quarantine/audit_compute_kras_holo_pockets.py",
        "contact_threshold_angstrom": CONTACT_THRESHOLD_A,
        "fourLpk_offset_note": (
            "For the 4LPK target topology (KRAS WT GDP-bound), the constant rule "
            "is pdb_resid = topology.residue_id + 1. Single PDB-numbering gap at "
            "engine_idx 59->60 (residue_id 59->70, 10 residues skipped — Switch-II "
            "disordered loop). This offset is target-specific; each reference PDB "
            "(6OIM/6UT0/7RPZ/6GJ8) has its own chain-A numbering reported below. "
            "When the bridge's forensic test runs against a particular target topology, "
            "the engine-residue translation is engine_idx = (target topology lookup "
            "of pdb_resid - 1). For 4LPK with its single Switch-II gap, see "
            "the gap-aware mapping rule in the audit report."
        ),
        "references": {},
    }

    for prefix, (lig_code, drug_name) in LIGAND_CODES.items():
        pdb = refs_dir / f"{prefix}.pdb"
        if not pdb.exists():
            print(f"  MISSING: {pdb}", file=sys.stderr)
            continue
        protein, hetatm = parse_pdb(pdb)
        if lig_code not in hetatm:
            print(f"  WARN: {prefix} has no {lig_code} HETATM "
                  f"(found: {sorted(hetatm.keys())})", file=sys.stderr)
            continue
        chain = first_chain(protein)
        ligand_atoms = hetatm[lig_code]
        # Filter protein to first chain only (KRAS chain A); if mixed,
        # all chains are reported.
        protein_chainA = [a for a in protein if a[6] == chain]
        contacts = contacts_within(ligand_atoms, protein_chainA, CONTACT_THRESHOLD_A)

        first_resid = min(a[0] for a in protein_chainA)
        last_resid = max(a[0] for a in protein_chainA)

        out["references"][prefix] = {
            "pdb_id": prefix.upper(),
            "ligand_resname": lig_code,
            "drug_name": drug_name,
            "ligand_atom_count": len(ligand_atoms),
            "chain": chain,
            "chain_first_resid": first_resid,
            "chain_last_resid": last_resid,
            "contact_threshold_angstrom": CONTACT_THRESHOLD_A,
            "n_contact_residues": len(contacts),
            "pdb_residues": [
                {"resid": r, "resname": contacts[r]}
                for r in sorted(contacts.keys())
            ],
            "pdb_residue_ids_only": sorted(contacts.keys()),
            "engine_residues_4lpk_offset_applied": (
                "Target-specific; see fourLpk_offset_note. For 4LPK target topology, "
                "engine_residue_id = pdb_resid - 1, valid for residues outside "
                "the Switch-II 61-70 gap. Per-reference engine_residues lists are NOT "
                "produced here because the bridge consumes pdb_residues directly and "
                "applies the target topology's residue_id mapping at runtime."
            ),
        }

    out_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote {out_path} ({out_path.stat().st_size} bytes)")
    for pdb_id, info in out["references"].items():
        print(f"  {pdb_id.upper()}: ligand={info['ligand_resname']:>4s} "
              f"({info['drug_name']}) "
              f"contacts={info['n_contact_residues']:>3d} "
              f"chain={info['chain']} "
              f"resid_range=[{info['chain_first_resid']},{info['chain_last_resid']}]")


if __name__ == "__main__":
    main()
