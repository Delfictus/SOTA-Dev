"""GrowthVector and SubPocket interfaces.

Dataclasses
-----------
GrowthVector
    A directional expansion opportunity from an anchor point.
SubPocket
    A spatially clustered region of pharmacophore features within a pocket.
GrowthVectorMap
    Collection of growth vectors and subpockets for one binding site.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class GrowthVector:
    """A directional growth/expansion opportunity from an anchor.

    Attributes:
        origin:            (x,y,z) start point (from anchor position).
        direction:         (dx,dy,dz) unit vector.
        free_length:       Angstrom before hitting protein wall or solvent.
        contact_density:   Protein atoms per Angstrom along the ray.
        expansion_stability: Does the pocket maintain shape along this
                            direction? (0-1, higher = more stable).
        exits_to_solvent:  True if the ray escapes to solvent.
        vector_score:      free_length * (1/(contact_density+0.1)) *
                           expansion_stability.  NOT a composite ranking
                           score — used only for filtering bad vectors.
        source_anchor_label: Label of the anchor this vector originates from.
    """

    origin: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    free_length: float
    contact_density: float
    expansion_stability: float
    exits_to_solvent: bool
    vector_score: float
    source_anchor_label: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["origin"] = list(d["origin"])
        d["direction"] = list(d["direction"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GrowthVector:
        data = copy.deepcopy(d)
        data["origin"] = tuple(data["origin"])
        data["direction"] = tuple(data["direction"])
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> GrowthVector:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> GrowthVector:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj


@dataclass
class SubPocket:
    """A spatially clustered region of pharmacophore features.

    Attributes:
        sub_pocket_id:       Sub-pocket index within the parent pocket.
        centroid:            (x,y,z) centroid of the sub-pocket.
        volume:              Estimated volume (Angstrom^3).
        feature_types:       List of pharmacophore feature types present.
        n_features:          Number of features in this sub-pocket.
        dominant_interaction: Most common interaction type.
    """

    sub_pocket_id: int
    centroid: Tuple[float, float, float]
    volume: float
    feature_types: List[str]
    n_features: int
    dominant_interaction: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["centroid"] = list(d["centroid"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SubPocket:
        data = copy.deepcopy(d)
        data["centroid"] = tuple(data["centroid"])
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> SubPocket:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> SubPocket:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj


@dataclass
class GrowthVectorMap:
    """Collection of growth vectors and subpockets for one binding site.

    Attributes:
        site_id:      Zero-based site index.
        vectors:      Growth vectors (filtered: no solvent-exiting vectors).
        sub_pockets:  Spatially clustered feature regions.
        n_vectors:    Number of valid growth vectors.
        n_sub_pockets: Number of identified sub-pockets.
    """

    site_id: int
    vectors: List[GrowthVector]
    sub_pockets: List[SubPocket]
    n_vectors: int
    n_sub_pockets: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "vectors": [v.to_dict() for v in self.vectors],
            "sub_pockets": [sp.to_dict() for sp in self.sub_pockets],
            "n_vectors": self.n_vectors,
            "n_sub_pockets": self.n_sub_pockets,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GrowthVectorMap:
        data = copy.deepcopy(d)
        data["vectors"] = [GrowthVector.from_dict(v) for v in data["vectors"]]
        data["sub_pockets"] = [
            SubPocket.from_dict(sp) for sp in data["sub_pockets"]
        ]
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> GrowthVectorMap:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> GrowthVectorMap:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
