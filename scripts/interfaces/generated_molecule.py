"""GeneratedMolecule interface — output of PhoreGen / PGMG generative models.

Consumed by WT-3 (filtering), WT-2 (docking → FEP), and WT-4 (orchestrator).
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class GeneratedMolecule:
    """A single molecule produced by a pharmacophore-guided generative model.

    Attributes:
        smiles:                     Canonical SMILES string.
        mol_block:                  3-D SDF / mol-block (V2000 or V3000).
        source:                     Generator identifier ("phoregen" | "pgmg").
        pharmacophore_match_score:  Fraction of pharmacophore features matched
                                    (0.0–1.0).
        matched_features:           List of matched feature type codes
                                    (e.g. ["AR", "HBD", "NI"]).
        generation_batch_id:        UUID or run-tag for the generation batch.
        generation_timestamp:       ISO-8601 UTC timestamp of generation.
    """
    smiles: str
    mol_block: str
    source: str
    pharmacophore_match_score: float
    matched_features: List[str]
    generation_batch_id: str
    generation_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GeneratedMolecule:
        return cls(**copy.deepcopy(d))

    @classmethod
    def from_json(cls, s: str) -> GeneratedMolecule:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> GeneratedMolecule:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
