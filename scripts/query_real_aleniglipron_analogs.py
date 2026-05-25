#!/usr/bin/env python3
"""Query public chemistry APIs for real GSBR-1290/Aleniglipron analog SMILES."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors


DEFAULT_OUTPUT = Path(
    "campaigns/glp1r_aleniglipron/track_0_manual_emulation/analog_registry.json"
)

QUERY_NAMES = (
    "Aleniglipron",
    "GSBR-1290",
    "GSBR1290",
    "GSBR-1001290",
    "GSBR1001290",
)

REQUIRED_ROLES = ("Baseline", "Bulky", "Rigidified")


@dataclass(frozen=True)
class Candidate:
    source: str
    source_id: str
    matched_query: str
    preferred_name: str | None
    smiles: str
    synonyms: list[str]


def normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def api_get_json(url: str, *, params: dict[str, str | int] | None = None) -> dict[str, Any] | None:
    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    loaded = response.json()
    if not isinstance(loaded, dict):
        raise ValueError(f"unexpected non-object JSON from {url}")
    return loaded


def pubchem_synonyms(cid: int) -> list[str]:
    data = api_get_json(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
    )
    if data is None:
        return []
    information = data.get("InformationList", {}).get("Information", [])
    if not isinstance(information, list) or not information:
        return []
    synonyms = information[0].get("Synonym", [])
    return [str(item) for item in synonyms] if isinstance(synonyms, list) else []


def query_pubchem() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for query in QUERY_NAMES:
        data = api_get_json(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{requests.utils.quote(query)}/property/IsomericSMILES,CanonicalSMILES,"
            "MolecularFormula,MolecularWeight/JSON"
        )
        if data is None:
            continue
        properties = data.get("PropertyTable", {}).get("Properties", [])
        if not isinstance(properties, list):
            continue
        for row in properties:
            if not isinstance(row, dict):
                continue
            cid = int(row.get("CID", 0))
            if cid <= 0 or f"pubchem:{cid}" in seen:
                continue
            synonyms = pubchem_synonyms(cid)
            normalized_synonyms = {normalize_name(item) for item in synonyms}
            if normalize_name(query) not in normalized_synonyms:
                continue
            smiles = str(row.get("IsomericSMILES") or row.get("CanonicalSMILES") or row.get("SMILES") or "")
            if not smiles:
                continue
            seen.add(f"pubchem:{cid}")
            candidates.append(
                Candidate(
                    source="PubChem",
                    source_id=str(cid),
                    matched_query=query,
                    preferred_name=synonyms[0] if synonyms else query,
                    smiles=smiles,
                    synonyms=synonyms,
                )
            )
    return candidates


def chembl_molecule(chembl_id: str) -> dict[str, Any]:
    data = api_get_json(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json")
    if data is None:
        raise ValueError(f"ChEMBL molecule disappeared after search: {chembl_id}")
    return data


def chembl_synonyms(molecule: dict[str, Any]) -> list[str]:
    values: list[str] = []
    pref_name = molecule.get("pref_name")
    if isinstance(pref_name, str):
        values.append(pref_name)
    for row in molecule.get("molecule_synonyms") or []:
        if isinstance(row, dict):
            for key in ("molecule_synonym", "synonyms"):
                value = row.get(key)
                if isinstance(value, str):
                    values.append(value)
    for row in molecule.get("cross_references") or []:
        if isinstance(row, dict):
            for key in ("xref_name", "xref_id"):
                value = row.get(key)
                if isinstance(value, str):
                    values.append(value)
    return sorted(set(values))


def query_chembl() -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for query in QUERY_NAMES:
        data = api_get_json(
            "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json",
            params={"q": query, "limit": 25},
        )
        if data is None:
            continue
        molecules = data.get("molecules", [])
        if not isinstance(molecules, list):
            continue
        for row in molecules:
            if not isinstance(row, dict):
                continue
            chembl_id = str(row.get("molecule_chembl_id") or "")
            if not chembl_id or f"chembl:{chembl_id}" in seen:
                continue
            molecule = chembl_molecule(chembl_id)
            synonyms = chembl_synonyms(molecule)
            normalized_synonyms = {normalize_name(item) for item in synonyms}
            if normalize_name(query) not in normalized_synonyms:
                continue
            structures = molecule.get("molecule_structures") or {}
            smiles = structures.get("canonical_smiles") if isinstance(structures, dict) else None
            if not isinstance(smiles, str) or not smiles:
                continue
            seen.add(f"chembl:{chembl_id}")
            candidates.append(
                Candidate(
                    source="ChEMBL",
                    source_id=chembl_id,
                    matched_query=query,
                    preferred_name=molecule.get("pref_name") if isinstance(molecule.get("pref_name"), str) else None,
                    smiles=smiles,
                    synonyms=synonyms,
                )
            )
    return candidates


def validate_smiles(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "valid": False,
            "num_atoms": None,
            "molecular_weight": None,
            "failure_reasons": ["RDKit MolFromSmiles failed"],
        }
    num_atoms = int(mol.GetNumAtoms())
    molecular_weight = float(Descriptors.MolWt(mol))
    reasons: list[str] = []
    if num_atoms <= 15:
        reasons.append("NumAtoms <= 15")
    if num_atoms >= 100:
        reasons.append("NumAtoms >= 100")
    if molecular_weight <= 200.0:
        reasons.append("MW <= 200")
    if molecular_weight >= 800.0:
        reasons.append("MW >= 800")
    return {
        "valid": not reasons,
        "num_atoms": num_atoms,
        "molecular_weight": molecular_weight,
        "failure_reasons": reasons,
    }


def unique_candidates(candidates: list[Candidate]) -> list[Candidate]:
    unique: list[Candidate] = []
    seen_smiles: set[str] = set()
    for candidate in candidates:
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(candidate.smiles), isomericSmiles=True)
        if canonical in seen_smiles:
            continue
        seen_smiles.add(canonical)
        unique.append(candidate)
    return unique


def build_registry() -> dict[str, Any]:
    raw_candidates = query_pubchem() + query_chembl()
    candidates = unique_candidates(raw_candidates)
    records: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []
    for candidate in candidates:
        validation = validate_smiles(candidate.smiles)
        record = {
            **asdict(candidate),
            "validation": validation,
        }
        records.append(record)
        if validation["valid"]:
            valid_records.append(record)

    status = "pass" if len(valid_records) >= len(REQUIRED_ROLES) else "requires_manual_smiles"
    assigned: list[dict[str, Any]] = []
    if status == "pass":
        for role, record in zip(REQUIRED_ROLES, valid_records, strict=True):
            assigned.append({**record, "role": role})

    return {
        "schema_version": "track0.analog_registry.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "required_roles": list(REQUIRED_ROLES),
        "queries": list(QUERY_NAMES),
        "validation_constraints": {
            "num_atoms_min_exclusive": 15,
            "num_atoms_max_exclusive": 100,
            "mw_min_exclusive": 200.0,
            "mw_max_exclusive": 800.0,
        },
        "accepted_analogs": assigned,
        "api_candidates": records,
        "manual_smiles_prompt": (
            "Provide three real, source-traceable SMILES strings for Baseline, Bulky, and "
            "Rigidified Structure Therapeutics GLP-1R agonist analogs. No placeholder or "
            "invented analogs were written."
        )
        if status != "pass"
        else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = build_registry()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": args.output.as_posix(), "status": registry["status"]}, indent=2))
    if registry["status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
