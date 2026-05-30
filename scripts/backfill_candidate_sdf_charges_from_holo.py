#!/usr/bin/env python3
"""Recover precomputed ligand charges from an observed holo topology and stamp them into an SDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sdf", type=Path, required=True)
    parser.add_argument("--holo-topology", type=Path, required=True)
    parser.add_argument("--output-sdf", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path, default=None)
    parser.add_argument("--candidate-id", default=None)
    return parser.parse_args()


def load_sdf(path: Path) -> Any:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = supplier[0] if len(supplier) else None
    if mol is None:
        raise ValueError(f"failed to parse SDF: {path}")
    return mol


def load_holo(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"holo topology is not a JSON object: {path}")
    return payload


def ligand_tail_from_holo(payload: dict[str, Any]) -> tuple[list[str], list[float]]:
    ligand_atom_count = int(payload["ligand_atom_count"])
    elements = [str(value) for value in payload["elements"][-ligand_atom_count:]]
    charges = [float(value) for value in payload["charges"][-ligand_atom_count:]]
    if len(elements) != ligand_atom_count or len(charges) != ligand_atom_count:
        raise ValueError("holo topology tail does not contain a full ligand element/charge vector")
    return elements, charges


def stamp_charges(mol: Any, charges: list[float], tool_label: str) -> None:
    if len(charges) != int(mol.GetNumAtoms()):
        raise ValueError("charge vector length does not match SDF atom count")
    for atom_index, charge in enumerate(charges):
        atom = mol.GetAtomWithIdx(atom_index)
        atom.SetDoubleProp("AM1BCCCharge", float(charge))
        atom.SetProp("am1bcc_charge", f"{float(charge):.12f}")
    mol.SetProp("am1bcc_charges_json", json.dumps([float(charge) for charge in charges], separators=(",", ":")))
    mol.SetProp("am1bcc_tool", tool_label)
    mol.SetProp("charge_method", "AM1-BCC")


def write_sdf(path: Path, mol: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(mol)
    finally:
        writer.close()


def main() -> int:
    args = parse_args()
    mol = load_sdf(args.input_sdf)
    holo = load_holo(args.holo_topology)
    holo_elements, holo_charges = ligand_tail_from_holo(holo)
    sdf_elements = [mol.GetAtomWithIdx(index).GetSymbol() for index in range(int(mol.GetNumAtoms()))]
    if sdf_elements != holo_elements:
        raise ValueError("SDF atom ordering/elements do not match holo ligand tail; refusing charge backfill")

    generated_ligand_id = holo.get("generated_ligand_id")
    if args.candidate_id and generated_ligand_id and str(args.candidate_id) != str(generated_ligand_id):
        raise ValueError(
            f"candidate mismatch: requested {args.candidate_id} but holo topology records {generated_ligand_id}"
        )

    tool_label = "recovered_from_observed_holo"
    stamp_charges(mol, holo_charges, tool_label)
    write_sdf(args.output_sdf, mol)

    evidence = {
        "schema_version": "prism.backfill_candidate_sdf_charges_from_holo.v1",
        "input_sdf": args.input_sdf.as_posix(),
        "holo_topology": args.holo_topology.as_posix(),
        "output_sdf": args.output_sdf.as_posix(),
        "candidate_id": args.candidate_id or generated_ligand_id,
        "ligand_atom_count": int(mol.GetNumAtoms()),
        "element_match_verified": True,
        "holo_generated_ligand_id": generated_ligand_id,
        "recovered_charge_method": str(holo.get("ligand_charge_method")),
        "backfilled_charge_source_label": tool_label,
    }
    evidence_path = args.evidence_json or args.output_sdf.with_suffix(".charge_backfill.json")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
