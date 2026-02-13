"""FEPResult interface — output of the OpenFE ABFE/RBFE pipeline.

Consumed by WT-4 (orchestrator / reporting) and WT-8 (interactive viewer).
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class FEPResult:
    """Result of a single Free Energy Perturbation calculation.

    Attributes:
        compound_id:                Unique compound identifier.
        delta_g_bind:               Predicted binding free energy (kcal/mol).
        delta_g_error:              Statistical uncertainty ± (kcal/mol).
        method:                     FEP method used ("ABFE" | "RBFE").
        n_repeats:                  Number of independent FEP repeats.
        convergence_passed:         True if BAR/MBAR convergence criteria met.
        hysteresis_kcal:            Forward–reverse hysteresis (kcal/mol).
        overlap_minimum:            Minimum overlap between adjacent lambda
                                    windows (0.0–1.0).
        max_protein_rmsd:           Maximum protein backbone RMSD during the
                                    FEP simulation (Angstrom).
        restraint_correction:       Boresch/orientational restraint correction
                                    (kcal/mol).
        charge_correction:          Finite-size charge correction (kcal/mol).
        vina_score_deprecated:      Legacy Vina score, if available.
        spike_pharmacophore_match:  Human-readable match summary, e.g.
                                    "4/5 features within 2.0A".
        classification:             Result classification:
                                    NOVEL_HIT | RECAPITULATED | WEAK_BINDER |
                                    FAILED_QC
        raw_data_path:              Filesystem path to the OpenFE output
                                    directory.
        fep_timestamp:              ISO-8601 UTC timestamp.
    """
    compound_id: str
    delta_g_bind: float
    delta_g_error: float
    method: str
    n_repeats: int
    convergence_passed: bool
    hysteresis_kcal: float
    overlap_minimum: float
    max_protein_rmsd: float
    restraint_correction: float
    charge_correction: float
    vina_score_deprecated: Optional[float]
    spike_pharmacophore_match: str
    classification: str
    raw_data_path: str
    fep_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Derived properties ───────────────────────────────────────────────

    @property
    def corrected_delta_g(self) -> float:
        """Binding free energy with restraint + charge corrections applied."""
        return self.delta_g_bind + self.restraint_correction + self.charge_correction

    @property
    def passed_qc(self) -> bool:
        """True if all quality-control gates are satisfied.

        QC gates:
          - Convergence check passed
          - Hysteresis < 1.0 kcal/mol
          - Overlap minimum >= 0.03
          - Max protein RMSD < 3.0 A
        """
        return (
            self.convergence_passed
            and abs(self.hysteresis_kcal) < 1.0
            and self.overlap_minimum >= 0.03
            and self.max_protein_rmsd < 3.0
        )

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["corrected_delta_g"] = self.corrected_delta_g
        d["passed_qc"] = self.passed_qc
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FEPResult:
        data = copy.deepcopy(d)
        # Remove computed properties that aren't constructor args
        data.pop("corrected_delta_g", None)
        data.pop("passed_qc", None)
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> FEPResult:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> FEPResult:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
