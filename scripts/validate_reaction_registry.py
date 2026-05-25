#!/usr/bin/env python3
"""Validate the PRISM Track A SMARTS reaction registry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from rdkit import Chem
from rdkit.Chem import AllChem


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "00_registry/chemistry/reaction_rules.v1.yml"


class RegistryValidationError(RuntimeError):
    """Raised when the SMARTS reaction registry is malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RegistryValidationError(f"{label} must be a mapping")
    return value


def require_sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise RegistryValidationError(f"{label} must be a list")
    return value


def atom_maps_from_mol(mol: Chem.Mol) -> set[int]:
    maps: set[int] = set()
    for atom in mol.GetAtoms():
        map_num = int(atom.GetAtomMapNum())
        if map_num > 0:
            if map_num in maps:
                raise RegistryValidationError(f"duplicate atom map {map_num} in SMARTS")
            maps.add(map_num)
    return maps


def validate_required_smarts(smarts: str, label: str) -> set[int]:
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        raise RegistryValidationError(f"{label} required_smarts does not compile: {smarts}")
    maps = atom_maps_from_mol(mol)
    if not maps:
        raise RegistryValidationError(f"{label} required_smarts contains no atom maps")
    return maps


def validate_reaction(reaction: Mapping[str, Any]) -> str:
    reaction_id = str(reaction.get("reaction_id", ""))
    if not reaction_id:
        raise RegistryValidationError("reaction missing reaction_id")
    smarts = str(reaction.get("smarts", ""))
    if not smarts:
        raise RegistryValidationError(f"{reaction_id} missing smarts")
    try:
        rd_rxn = AllChem.ReactionFromSmarts(smarts)
    except Exception as exc:  # noqa: BLE001 - surface RDKit parse reason.
        raise RegistryValidationError(f"{reaction_id} SMARTS failed RDKit parse: {exc}") from exc
    if rd_rxn is None:
        raise RegistryValidationError(f"{reaction_id} SMARTS did not compile")

    reactant_roles = require_mapping(reaction.get("reactant_roles"), f"{reaction_id}.reactant_roles")
    if set(reactant_roles) != {"scaffold", "synthon"}:
        raise RegistryValidationError(f"{reaction_id} must define scaffold and synthon roles")
    role_maps: dict[str, set[int]] = {}
    for role_name, role_obj in reactant_roles.items():
        role = require_mapping(role_obj, f"{reaction_id}.{role_name}")
        maps = validate_required_smarts(str(role.get("required_smarts", "")), f"{reaction_id}.{role_name}")
        reactive_map = int(role.get("reactive_atom_map", -1))
        if reactive_map not in maps:
            raise RegistryValidationError(f"{reaction_id}.{role_name} reactive atom map missing from role SMARTS")
        leaving_maps = [int(value) for value in require_sequence(role.get("leaving_group_atom_maps"), "leaving maps")]
        missing = [value for value in leaving_maps if value not in maps]
        if missing:
            raise RegistryValidationError(f"{reaction_id}.{role_name} leaving maps missing from role SMARTS: {missing}")
        bond_vector_reference = [int(value) for value in require_sequence(role.get("bond_vector_reference"), "bond vector")]
        if reactive_map not in bond_vector_reference:
            raise RegistryValidationError(f"{reaction_id}.{role_name} bond_vector_reference must include reactive atom")
        role_maps[str(role_name)] = maps

    product_bond = require_mapping(reaction.get("product_bond"), f"{reaction_id}.product_bond")
    atom_map_a = int(product_bond.get("atom_map_a", -1))
    atom_map_b = int(product_bond.get("atom_map_b", -1))
    all_role_maps = set().union(*role_maps.values())
    if atom_map_a not in all_role_maps or atom_map_b not in all_role_maps:
        raise RegistryValidationError(f"{reaction_id} product bond atom maps are absent from reactant roles")
    if float(product_bond.get("ideal_bond_length_A", 0.0)) <= 0.0:
        raise RegistryValidationError(f"{reaction_id} ideal_bond_length_A must be positive")
    torsion = require_mapping(product_bond.get("torsion_policy"), f"{reaction_id}.torsion_policy")
    if str(torsion.get("mode", "")) != "discrete_grid":
        raise RegistryValidationError(f"{reaction_id} torsion_policy.mode must be discrete_grid")
    if len(require_sequence(torsion.get("dihedral_deg"), f"{reaction_id}.dihedral_deg")) == 0:
        raise RegistryValidationError(f"{reaction_id} torsion grid must be nonempty")
    if bool(reaction.get("enabled")) and not isinstance(reaction.get("guards"), dict):
        raise RegistryValidationError(f"{reaction_id} enabled reaction lacks guards")
    return reaction_id


def main() -> int:
    args = parse_args()
    registry = yaml.safe_load(Path(args.registry).read_text())
    root = require_mapping(registry, "registry")
    reactions = require_sequence(root.get("reactions"), "reactions")
    disclaimer = str(root.get("disclaimer", ""))
    if "not guaranteed experimental success" not in disclaimer:
        raise RegistryValidationError("registry disclaimer must state experimental success is not guaranteed")
    seen: set[str] = set()
    enabled: list[str] = []
    for reaction_obj in reactions:
        reaction = require_mapping(reaction_obj, "reaction")
        reaction_id = validate_reaction(reaction)
        if reaction_id in seen:
            raise RegistryValidationError(f"duplicate reaction_id: {reaction_id}")
        seen.add(reaction_id)
        if bool(reaction.get("enabled")):
            enabled.append(reaction_id)
    if not enabled:
        raise RegistryValidationError("no enabled reactions")
    print(
        "reaction_registry_valid "
        f"path={Path(args.registry)} reactions={len(reactions)} enabled={','.join(enabled)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
