"""Strict loader for PRISM-FORGE reaction-rule registries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import yaml


SCHEMA_VERSION: Final[str] = "PRISM.reaction_rules.v1"
LEGACY_SCHEMA_VERSION: Final[str] = "reaction_rules.v1"
FreeRotation = Literal["free_rotation"]
DihedralOmega = float | FreeRotation


@dataclass(frozen=True)
class KinematicConstraints:
    bond_length_A: float
    dihedral_omega_rad: DihedralOmega


@dataclass(frozen=True)
class ReactionRule:
    rule_id: str
    name: str
    smarts: str
    scaffold_leaving_group: str
    synthon_leaving_group: str
    kinematic_constraints: KinematicConstraints


@dataclass(frozen=True)
class ReactionRegistry:
    schema_version: str
    reactions: tuple[ReactionRule, ...]

    def by_rule_id(self) -> dict[str, ReactionRule]:
        return {reaction.rule_id: reaction for reaction in self.reactions}


def load_reaction_registry(path: Path) -> ReactionRegistry:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = _mapping(loaded, "reaction registry")
    schema_version = _string(payload.get("schema_version"), "schema_version")
    if schema_version not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION} or {LEGACY_SCHEMA_VERSION}, got {schema_version}"
        )
    raw_reactions = _list(payload.get("reactions"), "reactions")
    reactions = tuple(_parse_reaction(item, index) for index, item in enumerate(raw_reactions))
    if not reactions:
        raise ValueError("reactions must contain at least one rule")
    rule_ids = [reaction.rule_id for reaction in reactions]
    duplicate_rule_ids = sorted({rule_id for rule_id in rule_ids if rule_ids.count(rule_id) > 1})
    if duplicate_rule_ids:
        raise ValueError(f"duplicate reaction rule_id values: {duplicate_rule_ids}")
    return ReactionRegistry(schema_version=schema_version, reactions=reactions)


def _parse_reaction(value: object, index: int) -> ReactionRule:
    payload = _mapping(value, f"reactions[{index}]")
    constraints = _extract_kinematic_constraints(payload, index)
    return ReactionRule(
        rule_id=_first_non_empty_string(
            payload,
            ("rule_id", "reaction_id"),
            f"reactions[{index}].rule_id",
        ),
        name=_first_non_empty_string(
            payload,
            ("name", "reaction_name"),
            f"reactions[{index}].name",
        ),
        smarts=_non_empty_string(payload.get("smarts"), f"reactions[{index}].smarts"),
        scaffold_leaving_group=_extract_leaving_group(
            payload,
            role="scaffold",
            fallback_key="scaffold_leaving_group",
            label=f"reactions[{index}].scaffold_leaving_group",
        ),
        synthon_leaving_group=_extract_leaving_group(
            payload,
            role="synthon",
            fallback_key="synthon_leaving_group",
            label=f"reactions[{index}].synthon_leaving_group",
        ),
        kinematic_constraints=KinematicConstraints(
            bond_length_A=_positive_float(
                constraints.get("bond_length_A"),
                f"reactions[{index}].kinematic_constraints.bond_length_A",
            ),
            dihedral_omega_rad=_dihedral(
                constraints.get("dihedral_omega_rad"),
                f"reactions[{index}].kinematic_constraints.dihedral_omega_rad",
            ),
        ),
    )


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _non_empty_string(value: object, label: str) -> str:
    parsed = _string(value, label)
    if parsed == "":
        raise ValueError(f"{label} must not be empty")
    return parsed


def _first_non_empty_string(payload: dict[str, object], keys: tuple[str, ...], label: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value != "":
            return value
    raise ValueError(f"{label} must not be empty")


def _extract_leaving_group(
    payload: dict[str, object],
    *,
    role: str,
    fallback_key: str,
    label: str,
) -> str:
    fallback = payload.get(fallback_key)
    if isinstance(fallback, str) and fallback != "":
        return fallback
    roles = payload.get("reactant_roles")
    if isinstance(roles, dict):
        role_payload = roles.get(role)
        if isinstance(role_payload, dict):
            maps = role_payload.get("leaving_group_atom_maps")
            if isinstance(maps, list):
                return ",".join(str(item) for item in maps)
    raise ValueError(f"{label} must not be empty")


def _extract_kinematic_constraints(payload: dict[str, object], index: int) -> dict[str, object]:
    raw_constraints = payload.get("kinematic_constraints")
    if isinstance(raw_constraints, dict):
        return cast(dict[str, object], raw_constraints)
    product_bond = _mapping(payload.get("product_bond"), f"reactions[{index}].product_bond")
    torsion_policy = _mapping(
        product_bond.get("torsion_policy"),
        f"reactions[{index}].product_bond.torsion_policy",
    )
    dihedrals = torsion_policy.get("dihedral_deg")
    first_dihedral_rad: object = "free_rotation"
    if isinstance(dihedrals, list) and dihedrals:
        raw_first = dihedrals[0]
        if isinstance(raw_first, int | float) and not isinstance(raw_first, bool):
            first_dihedral_rad = float(raw_first) * 3.141592653589793 / 180.0
    return {
        "bond_length_A": product_bond.get("ideal_bond_length_A"),
        "dihedral_omega_rad": first_dihedral_rad,
    }


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not parsed > 0.0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _dihedral(value: object, label: str) -> DihedralOmega:
    if value == "free_rotation":
        return "free_rotation"
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric radians or 'free_rotation'")
    return float(value)


__all__ = [
    "DihedralOmega",
    "KinematicConstraints",
    "ReactionRegistry",
    "ReactionRule",
    "SCHEMA_VERSION",
    "load_reaction_registry",
]
