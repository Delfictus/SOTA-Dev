"""Universal materials action-space primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MaterialsAction:
    """A deterministic materials edit descriptor."""

    action_type: str
    default_parameters: dict[str, float] = field(default_factory=dict)

    def apply(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        base = dict(state or {})
        history = list(base.get("action_history", []))
        history.append(self.action_type)
        base["action_history"] = history
        base["last_action_parameters"] = dict(self.default_parameters)
        return base


class UniversalMaterialsActionSpace:
    """Production-safe action catalog spanning organic and inorganic edits."""

    action_types: tuple[str, ...] = (
        "doping",
        "vacancy",
        "interstitial",
        "polymer_modification",
        "metal_coordination",
        "ligand_exchange",
        "surface_passivation",
        "solvent_shell_rewire",
        "grain_boundary_stabilization",
    )

    def instantiate(self, action_type: str) -> MaterialsAction:
        if action_type not in self.action_types:
            raise ValueError(f"unsupported materials action: {action_type}")
        return MaterialsAction(action_type=action_type, default_parameters={"magnitude": 1.0})

    def all_actions(self) -> list[MaterialsAction]:
        return [self.instantiate(action_type) for action_type in self.action_types]
