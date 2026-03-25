"""AnchorPoint interface — spike-to-residue interaction anchors.

Dataclasses
-----------
AnchorPoint
    A single anchored interaction between a PRISM spike and a lining residue.
AnchorPointMap
    Collection of anchor points for one binding site.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


# -- Spike type → interaction type mapping ----------------------------------
SPIKE_TYPE_TO_INTERACTION: Dict[str, str] = {
    "BNZ": "PI_STACK",
    "PHE": "HYDROPHOBIC",
    "TYR": "H_BOND_DONOR",
    "TRP": "PI_STACK",
    "CATION": "SALT_BRIDGE",
    "ANION": "SALT_BRIDGE",
    "UNK": "HYDROPHOBIC",
    "SS": "COVALENT",
}

# -- Ideal distance ranges per interaction type (Angstrom) ------------------
IDEAL_DISTANCE: Dict[str, Tuple[float, float]] = {
    "H_BOND_DONOR": (2.5, 3.5),
    "H_BOND_ACCEPTOR": (2.5, 3.5),
    "PI_STACK": (3.5, 5.5),
    "HYDROPHOBIC": (3.5, 5.0),
    "SALT_BRIDGE": (3.0, 5.0),
    "COVALENT": (1.5, 3.0),
}


@dataclass
class AnchorPoint:
    """A single anchored interaction at a binding site.

    Represents a high-intensity spike that overlaps a lining residue,
    identifying a specific protein-ligand interaction opportunity.

    Attributes:
        residue_name:        Three-letter residue code (e.g. "TYR").
        residue_id:          Topology residue ID.
        chain:               Chain identifier.
        atom_label:          Atom label (e.g. "TYR142_OH").
        interaction_type:    HBD, HBA, PI_STACK, HYDROPHOBIC, SALT_BRIDGE, COVALENT.
        x, y, z:             Anchor position (Angstrom).
        distance_to_centroid: Distance from pocket centroid (Angstrom).
        spike_intensity:     Associated spike intensity (0+).
        temporal_persistence: Fraction of frames where this anchor is
                             observed (0-1).
        geometric_alignment: How well geometry matches ideal (0-1).
        stability_stddev:    Stddev of spike-residue distance over frames (A).
        confidence:          intensity * persistence * alignment * (1/dist).
    """

    residue_name: str
    residue_id: int
    chain: str
    atom_label: str
    interaction_type: str
    x: float
    y: float
    z: float
    distance_to_centroid: float
    spike_intensity: float
    temporal_persistence: float
    geometric_alignment: float
    stability_stddev: float
    confidence: float

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AnchorPoint:
        return cls(**copy.deepcopy(d))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> AnchorPoint:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> AnchorPoint:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj


@dataclass
class AnchorPointMap:
    """Collection of anchor points for one binding site.

    Attributes:
        site_id:       Zero-based site index.
        pocket_centroid: (x, y, z) of the pocket centroid.
        anchors:       List of AnchorPoint objects, sorted by confidence desc.
        n_anchors:     Total number of anchor points.
        anchor_density: anchors per lining residue.
    """

    site_id: int
    pocket_centroid: Tuple[float, float, float]
    anchors: List[AnchorPoint]
    n_anchors: int
    anchor_density: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "pocket_centroid": list(self.pocket_centroid),
            "anchors": [a.to_dict() for a in self.anchors],
            "n_anchors": self.n_anchors,
            "anchor_density": self.anchor_density,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AnchorPointMap:
        data = copy.deepcopy(d)
        data["anchors"] = [AnchorPoint.from_dict(a) for a in data["anchors"]]
        data["pocket_centroid"] = tuple(data["pocket_centroid"])
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> AnchorPointMap:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> AnchorPointMap:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
