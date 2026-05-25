"""Strict loader for PRISM-FORGE reaction-rule registries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import yaml


SCHEMA_VERSION: Final[str] = "PRISM.reaction_rules.v1"
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
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {schema_version}")
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
    constraints = _mapping(payload.get("kinematic_constraints"), f"reactions[{index}].kinematic_constraints")
    return ReactionRule(
        rule_id=_non_empty_string(payload.get("rule_id"), f"reactions[{index}].rule_id"),
        name=_non_empty_string(payload.get("name"), f"reactions[{index}].name"),
        smarts=_non_empty_string(payload.get("smarts"), f"reactions[{index}].smarts"),
        scaffold_leaving_group=_non_empty_string(
            payload.get("scaffold_leaving_group"),
            f"reactions[{index}].scaffold_leaving_group",
        ),
        synthon_leaving_group=_non_empty_string(
            payload.get("synthon_leaving_group"),
            f"reactions[{index}].synthon_leaving_group",
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
