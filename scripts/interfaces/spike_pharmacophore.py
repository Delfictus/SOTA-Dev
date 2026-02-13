"""SpikePharmacophore interface — PRISM spike output to pharmacophore format.

Defines the canonical data contract for pharmacophore features extracted from
PRISM-4D spike events.  Consumed by WT-1 (PhoreGen/PGMG), WT-2 (FEP), WT-3
(filtering), and WT-4 (orchestrator).

Feature-type codes follow IUPHAR/Catalyst convention:
    AR  — Aromatic ring
    PI  — Positive ionizable
    NI  — Negative ionizable
    HBD — Hydrogen-bond donor
    HBA — Hydrogen-bond acceptor
    HY  — Hydrophobic
    XB  — Exclusion / steric block
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ── Spike-type → pharmacophore feature-type mapping ──────────────────────
# Mirrors the physics basis in scripts/spike_pharmacophore_map.py
SPIKE_TYPE_TO_FEATURE: Dict[str, str] = {
    "BNZ": "AR",      # pi-stacking hotspot
    "PHE": "HY",      # pure hydrophobic
    "TYR": "HBD",     # H-bond donor + pi (dual, but primary is HBD)
    "TRP": "AR",      # large aromatic
    "CATION": "NI",   # pocket is cationic → wants anionic ligand group
    "ANION": "PI",    # pocket is anionic → wants cationic ligand group
    "UNK": "HY",      # shape complementarity → hydrophobic fill
    "SS": "HBD",      # disulfide / thiol-reactive → covalent (mapped as HBD)
}


@dataclass
class PharmacophoreFeature:
    """A single pharmacophore feature point derived from PRISM spike events.

    Attributes:
        feature_type:       Pharmacophore type code (AR, PI, NI, HBD, HBA, HY).
        x, y, z:            Cartesian coordinates in Angstrom (model frame).
        intensity:          PRISM spike intensity, normalised 0-1.
        source_spike_type:  Raw spike type from PRISM (BNZ, TYR, CATION, …).
        source_residue_id:  Topology residue ID of the source residue.
        source_residue_name: Human-readable residue label, e.g. "TYR142".
        wavelength_nm:      UV excitation wavelength that generated this spike.
        water_density:      Local solvent accessibility at the feature site.
    """
    feature_type: str
    x: float
    y: float
    z: float
    intensity: float
    source_spike_type: str
    source_residue_id: int
    source_residue_name: str
    wavelength_nm: float
    water_density: float

    def distance_to(self, other: PharmacophoreFeature) -> float:
        """Euclidean distance to another feature (Angstrom)."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return (x, y, z) position tuple."""
        return (self.x, self.y, self.z)


@dataclass
class ExclusionSphere:
    """A steric exclusion volume that ligand atoms must avoid.

    Attributes:
        x, y, z:       Centre of the exclusion sphere (Angstrom).
        radius:        Exclusion radius (Angstrom).
        source_atom:   Atom label, e.g. "CA:ALA145".
    """
    x: float
    y: float
    z: float
    radius: float
    source_atom: str

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class SpikePharmacophore:
    """Complete pharmacophore model for a single PRISM-detected binding pocket.

    This is the primary output of WT-0/WT-1 and the primary input for
    generative molecule design (PhoreGen, PGMG) and docking preparation.

    Attributes:
        target_name:            Target identifier, e.g. "KRAS_G12C".
        pdb_id:                 PDB accession of the input structure.
        pocket_id:              Zero-based pocket index from PRISM detection.
        features:               Ordered list of pharmacophore feature points.
        exclusion_spheres:      Steric exclusion volumes.
        pocket_centroid:        (x, y, z) centroid of the pocket (Angstrom).
        pocket_lining_residues: Topology residue IDs lining the pocket.
        prism_run_hash:         SHA-256 hex digest of PRISM binary + input.
        creation_timestamp:     ISO-8601 UTC timestamp of creation.
    """
    target_name: str
    pdb_id: str
    pocket_id: int
    features: List[PharmacophoreFeature]
    exclusion_spheres: List[ExclusionSphere]
    pocket_centroid: Tuple[float, float, float]
    pocket_lining_residues: List[int]
    prism_run_hash: str
    creation_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert to a plain dict (JSON-safe)."""
        d = asdict(self)
        # Tuple → list for JSON
        d["pocket_centroid"] = list(d["pocket_centroid"])
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SpikePharmacophore:
        """Reconstruct from a plain dict (e.g. parsed JSON)."""
        data = copy.deepcopy(d)
        data["features"] = [
            PharmacophoreFeature(**f) for f in data["features"]
        ]
        data["exclusion_spheres"] = [
            ExclusionSphere(**e) for e in data["exclusion_spheres"]
        ]
        data["pocket_centroid"] = tuple(data["pocket_centroid"])
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> SpikePharmacophore:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        """Serialize to pickle bytes."""
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> SpikePharmacophore:
        """Deserialize from pickle bytes."""
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    # ── Downstream conversion methods ────────────────────────────────────

    def to_phoregen_json(self) -> Dict[str, Any]:
        """Convert to PhoreGen input format.

        PhoreGen expects a JSON dict with:
            features: [{type, x, y, z, radius, optional}, ...]
            exclusions: [{x, y, z, radius}, ...]
            bounds: {center: [x,y,z], size: [sx,sy,sz]}

        Feature radii default to 1.5 A (pharmacophore tolerance).
        """
        phoregen_features = []
        for feat in self.features:
            phoregen_features.append({
                "type": feat.feature_type,
                "x": round(feat.x, 3),
                "y": round(feat.y, 3),
                "z": round(feat.z, 3),
                "radius": 1.5,
                "optional": feat.intensity < 0.3,
            })

        exclusions = []
        for exc in self.exclusion_spheres:
            exclusions.append({
                "x": round(exc.x, 3),
                "y": round(exc.y, 3),
                "z": round(exc.z, 3),
                "radius": round(exc.radius, 3),
            })

        cx, cy, cz = self.pocket_centroid
        box = self.to_docking_box()
        bounds = {
            "center": [round(cx, 3), round(cy, 3), round(cz, 3)],
            "size": [box["size_x"], box["size_y"], box["size_z"]],
        }

        return {
            "target": self.target_name,
            "pdb_id": self.pdb_id,
            "pocket_id": self.pocket_id,
            "features": phoregen_features,
            "exclusions": exclusions,
            "bounds": bounds,
        }

    def to_pgmg_posp(self) -> str:
        """Convert to PGMG .posp pharmacophore specification string.

        PGMG uses a line-based format:
            FEATURE <type> <x> <y> <z> <radius> <weight>
            EXCLUSION <x> <y> <z> <radius>
            CENTROID <x> <y> <z>
        """
        lines: List[str] = []
        lines.append(f"# PGMG pharmacophore for {self.target_name} pocket {self.pocket_id}")
        lines.append(f"# PDB: {self.pdb_id}  Hash: {self.prism_run_hash}")

        for feat in self.features:
            weight = round(feat.intensity, 4)
            lines.append(
                f"FEATURE {feat.feature_type} "
                f"{feat.x:.3f} {feat.y:.3f} {feat.z:.3f} "
                f"1.500 {weight}"
            )

        for exc in self.exclusion_spheres:
            lines.append(
                f"EXCLUSION {exc.x:.3f} {exc.y:.3f} {exc.z:.3f} "
                f"{exc.radius:.3f}"
            )

        cx, cy, cz = self.pocket_centroid
        lines.append(f"CENTROID {cx:.3f} {cy:.3f} {cz:.3f}")
        lines.append("")  # trailing newline
        return "\n".join(lines)

    def to_docking_box(self, padding: float = 4.0) -> Dict[str, float]:
        """Compute an axis-aligned docking box enclosing all features.

        Matches the dict format used by ``gpu_dock.extract_docking_boxes``
        and ``docking_prep.compute_docking_box``:
            center_x, center_y, center_z, size_x, size_y, size_z

        Args:
            padding: Extra padding per side in Angstrom (default 4.0).

        Returns:
            Dict with center_x/y/z and size_x/y/z (Angstrom), values
            clamped to a maximum of 40.0 per dimension (Vina limit).
        """
        if not self.features:
            cx, cy, cz = self.pocket_centroid
            return {
                "center_x": round(cx, 3),
                "center_y": round(cy, 3),
                "center_z": round(cz, 3),
                "size_x": 20.0,
                "size_y": 20.0,
                "size_z": 20.0,
            }

        xs = [f.x for f in self.features]
        ys = [f.y for f in self.features]
        zs = [f.z for f in self.features]

        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        cz = (min(zs) + max(zs)) / 2.0

        sx = max((max(xs) - min(xs)) + 2 * padding, 20.0)
        sy = max((max(ys) - min(ys)) + 2 * padding, 20.0)
        sz = max((max(zs) - min(zs)) + 2 * padding, 20.0)

        return {
            "center_x": round(cx, 3),
            "center_y": round(cy, 3),
            "center_z": round(cz, 3),
            "size_x": round(min(sx, 40.0), 1),
            "size_y": round(min(sy, 40.0), 1),
            "size_z": round(min(sz, 40.0), 1),
        }
