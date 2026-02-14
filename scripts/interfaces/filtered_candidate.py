"""FilteredCandidate interface — output of multi-stage filtering pipeline.

Wraps a GeneratedMolecule with computed drug-likeness metrics, PAINS
alerts, novelty scores, and cluster assignment.  Consumed by WT-2
(docking / FEP) and WT-4 (orchestrator / reporting).
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .generated_molecule import GeneratedMolecule


@dataclass
class FilteredCandidate:
    """A molecule that has passed (or failed) multi-stage filtering.

    Attributes:
        molecule:                   The underlying GeneratedMolecule.
        qed_score:                  Quantitative Estimate of Drug-likeness
                                    (0.0–1.0).
        sa_score:                   Synthetic Accessibility score (1 easy – 10
                                    hard; Ertl & Schuffenhauer 2009).
        lipinski_violations:        Number of Lipinski Rule-of-Five violations
                                    (0–4).
        pains_alerts:               List of PAINS filter hit names (empty if
                                    clean).
        tanimoto_to_nearest_known:  Maximum Tanimoto similarity (ECFP4) to
                                    the reference compound set.
        nearest_known_cid:          PubChem CID of the nearest known compound
                                    (string to handle large CIDs).
        cluster_id:                 Butina / Taylor cluster assignment index.
        passed_all_filters:         True if the candidate passed every filter.
        rejection_reason:           Human-readable rejection reason, or None if
                                    passed.
    """
    molecule: GeneratedMolecule
    qed_score: float
    sa_score: float
    lipinski_violations: int
    pains_alerts: List[str]
    tanimoto_to_nearest_known: float
    nearest_known_cid: str
    cluster_id: int
    passed_all_filters: bool
    rejection_reason: Optional[str] = None

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FilteredCandidate:
        data = copy.deepcopy(d)
        data["molecule"] = GeneratedMolecule(**data["molecule"])
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> FilteredCandidate:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> FilteredCandidate:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
