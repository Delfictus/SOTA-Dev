"""Lock-reaching pharmacophore assessment for static Track A candidates."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence, cast

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors


Coordinate3D = tuple[float, float, float]


@dataclass(frozen=True)
class BiasPharmacophoreAssessment:
    """Static pharmacophore checks aligned to the Epoch 015 lock-reaching goal."""

    has_aromatic_core: bool
    intracellular_reach_angstrom: float
    distal_steric_volume_angstrom3: float
    directional_rotatable_bonds: int
    electrostatic_complement_proxy: float
    hydrophobic_match_proxy: float
    matches_required: bool
    projection_bonus: float


def assess_bias_pharmacophore(
    smiles: str,
    *,
    coordinates_json: str | None = None,
    attachment_point: Coordinate3D | None = None,
) -> BiasPharmacophoreAssessment:
    """Assess whether a candidate has the static features of a lock wedge."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return BiasPharmacophoreAssessment(
            has_aromatic_core=False,
            intracellular_reach_angstrom=0.0,
            distal_steric_volume_angstrom3=0.0,
            directional_rotatable_bonds=99,
            electrostatic_complement_proxy=0.0,
            hydrophobic_match_proxy=0.0,
            matches_required=False,
            projection_bonus=0.0,
        )

    mol_any = cast(Any, mol)
    aromatic_atoms = [atom.GetIdx() for atom in mol_any.GetAtoms() if bool(atom.GetIsAromatic())]
    has_aromatic_core = len(aromatic_atoms) > 0
    coordinates = coordinates_from_json(coordinates_json)
    reach = max_atomic_span(coordinates, attachment_point)
    heavy_atoms = float(mol.GetNumHeavyAtoms())
    distal_volume = max(0.0, heavy_atoms * 12.0)
    rotatable = int(cast(Any, rdMolDescriptors).CalcNumRotatableBonds(mol))
    hydrophobic = max(0.0, float(cast(Any, Crippen).MolLogP(mol)) * 10.0)
    charge_proxy = -float(cast(Any, Descriptors).NumHAcceptors(mol)) + float(
        cast(Any, Descriptors).NumHDonors(mol)
    ) * 0.5
    electrostatic = max(0.0, -charge_proxy)
    matches_required = has_aromatic_core and reach >= 12.0 and distal_volume >= 100.0 and rotatable <= 3
    projection_bonus = (
        (0.25 if has_aromatic_core else 0.0)
        + min(reach / 24.0, 0.25)
        + min(distal_volume / 400.0, 0.25)
        + (0.25 if rotatable <= 3 else max(0.0, 0.25 - 0.05 * float(rotatable - 3)))
    )
    return BiasPharmacophoreAssessment(
        has_aromatic_core=has_aromatic_core,
        intracellular_reach_angstrom=reach,
        distal_steric_volume_angstrom3=distal_volume,
        directional_rotatable_bonds=rotatable,
        electrostatic_complement_proxy=electrostatic,
        hydrophobic_match_proxy=hydrophobic,
        matches_required=matches_required,
        projection_bonus=min(projection_bonus, 1.0),
    )


def assessment_to_dict(assessment: BiasPharmacophoreAssessment) -> dict[str, Any]:
    """Return a JSON-ready representation."""

    return asdict(assessment)


def coordinates_from_json(coordinates_json: str | None) -> tuple[Coordinate3D, ...]:
    """Decode the PRISM coordinate JSON format used by survivor rows."""

    if not coordinates_json:
        return ()
    decoded = json.loads(coordinates_json)
    if not isinstance(decoded, list):
        return ()
    coords: list[Coordinate3D] = []
    for raw in decoded:
        if not isinstance(raw, list) or len(raw) < 3:
            continue
        coords.append((_float_value(raw[0]), _float_value(raw[1]), _float_value(raw[2])))
    return tuple(coords)


def max_atomic_span(
    coordinates: Sequence[Coordinate3D],
    attachment_point: Coordinate3D | None = None,
) -> float:
    """Return max distance from attachment point, or full pairwise span."""

    if not coordinates:
        return 0.0
    if attachment_point is not None:
        return max(distance(attachment_point, coord) for coord in coordinates)
    max_span = 0.0
    for index, left in enumerate(coordinates):
        for right in coordinates[index + 1 :]:
            max_span = max(max_span, distance(left, right))
    return max_span


def distance(left: Coordinate3D, right: Coordinate3D) -> float:
    """Euclidean distance between two coordinates."""

    dx = left[0] - right[0]
    dy = left[1] - right[1]
    dz = left[2] - right[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def assess_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Assess one row from a candidate or synthon parquet."""

    smiles = str(row.get("canonical_smiles", row.get("smiles", "")))
    coordinates = row.get("coordinates_json")
    return assessment_to_dict(
        assess_bias_pharmacophore(
            smiles,
            coordinates_json=coordinates if isinstance(coordinates, str) else None,
        )
    )


def _float_value(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"coordinate value must be numeric, got {type(value).__name__}")
