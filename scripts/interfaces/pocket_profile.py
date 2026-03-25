"""PocketProfile interface — descriptive pocket chemistry summary.

Dataclasses
-----------
PocketProfile
    Descriptive chemistry and geometry profile for a binding site.
    Purely observational — no predictions or recommendations.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class PocketProfile:
    """Descriptive chemistry and geometry profile for a binding site.

    All values are directly computed from spike pharmacophore data and
    lining residue composition.  No heuristic interpretation.

    Attributes:
        site_id:              Zero-based site index.
        aromatic_fraction:    Fraction of lining residues that are aromatic.
        polar_fraction:       Fraction that are polar (donor or acceptor).
        hydrophobic_fraction: Fraction that are hydrophobic.
        charged_positive_fraction: Fraction that are positively charged.
        charged_negative_fraction: Fraction that are negatively charged.
        charge_bias:          (n_pos - n_neg) / n_total. >0 = basic, <0 = acidic.
        volume:               Pocket volume (Angstrom^3).
        enclosure:            Burial/enclosure score (0-1).
        n_lining_residues:    Number of lining residues.
        feature_coupling:     Spatial clustering entropy of pharmacophore
                              features. Lower = more clustered.
        mw_class:             "fragment" (<300), "lead" (300-500), "beyond_ro5" (>500).
        polarity_class:       "hydrophobic", "mixed", "polar".
        water_displacement_energy: Sum of positive dG from water map (kcal/mol).
                              0.0 if no water map available.
    """

    site_id: int
    aromatic_fraction: float
    polar_fraction: float
    hydrophobic_fraction: float
    charged_positive_fraction: float
    charged_negative_fraction: float
    charge_bias: float
    volume: float
    enclosure: float
    n_lining_residues: int
    feature_coupling: float
    mw_class: str
    polarity_class: str
    water_displacement_energy: float

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PocketProfile:
        return cls(**copy.deepcopy(d))

    @classmethod
    def from_json(cls, s: str) -> PocketProfile:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> PocketProfile:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
