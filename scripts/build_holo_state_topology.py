#!/usr/bin/env python3
"""Merge the WT GLP-1R topology with aligned Aleniglipron into a holo topology."""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib import import_module
from itertools import combinations
from pathlib import Path
from typing import Any, TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APO_TOPOLOGY = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/"
    "20260518T031002Z/04_TOPOLOGIES/glp1r_6XOX_WT.topology.json"
)
DEFAULT_LIGAND_SDF = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/conformers/ALENI-PARENT_6XOX_frame_minimized.sdf"
)
DEFAULT_OUTPUT = REPO_ROOT / "04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json"
LIGAND_RESIDUE_NAME = "ALN"
LIGAND_CHAIN_ID = "L"
COLLISION_MIN_HEAVY_DISTANCE_A = 1.5
COLLISION_MAX_HEAVY_DISTANCE_A = 4.0

JsonObject: TypeAlias = dict[str, Any]
Point: TypeAlias = tuple[float, float, float]


Chem = cast(Any, import_module("rdkit.Chem"))
AllChem = cast(Any, import_module("rdkit.Chem.AllChem"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apo-topology", type=Path, default=DEFAULT_APO_TOPOLOGY)
    parser.add_argument("--ligand-sdf", type=Path, default=DEFAULT_LIGAND_SDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-heavy-distance-a", type=float, default=COLLISION_MIN_HEAVY_DISTANCE_A)
    parser.add_argument("--max-heavy-distance-a", type=float, default=COLLISION_MAX_HEAVY_DISTANCE_A)
    parser.add_argument("--allow-non-am1bcc", action="store_true")
    return parser.parse_args()


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def read_json_object(path: Path) -> JsonObject:
    decoded = json.loads(path.read_text())
    if not isinstance(decoded, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return cast(JsonObject, decoded)


def ensure_list(topology: JsonObject, key: str) -> list[Any]:
    value = topology.setdefault(key, [])
    if not isinstance(value, list):
        raise TypeError(f"topology[{key!r}] must be a list")
    return value


def ensure_dict(topology: JsonObject, key: str) -> dict[str, Any]:
    value = topology.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"topology[{key!r}] must be a dict")
    return cast(dict[str, Any], value)


def flat_positions(points: list[Point]) -> list[float]:
    values: list[float] = []
    for x_coord, y_coord, z_coord in points:
        values.extend([x_coord, y_coord, z_coord])
    return values


def topology_points(topology: JsonObject) -> list[Point]:
    raw = ensure_list(topology, "positions")
    if len(raw) % 3 != 0:
        raise ValueError("topology positions must be a flat xyz array")
    values = [float(item) for item in raw]
    return [(values[index], values[index + 1], values[index + 2]) for index in range(0, len(values), 3)]


def load_ligand(path: Path) -> Any:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = supplier[0] if len(supplier) else None
    if mol is None:
        raise ValueError(f"failed to parse ligand SDF: {path}")
    if int(mol.GetNumConformers()) < 1:
        raise ValueError(f"ligand has no conformers: {path}")
    return mol


def ligand_points(mol: Any) -> list[Point]:
    conformer = mol.GetConformer(0)
    points: list[Point] = []
    for atom_index in range(int(mol.GetNumAtoms())):
        pos = conformer.GetAtomPosition(atom_index)
        points.append((float(pos.x), float(pos.y), float(pos.z)))
    return points


def periodic_table() -> Any:
    return Chem.GetPeriodicTable()


def atomic_mass(element: str) -> float:
    return float(periodic_table().GetAtomicWeight(element))


def gb_radius(element: str) -> float:
    radii = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "F": 1.47,
        "P": 1.80,
        "S": 1.80,
        "CL": 1.75,
        "BR": 1.85,
        "I": 1.98,
        "B": 1.92,
    }
    return radii.get(element.upper(), 1.70)


def lj_params(element: str) -> JsonObject:
    params: dict[str, tuple[float, float]] = {
        "H": (1.069078461768407, 0.0157),
        "C": (3.3996695084235347, 0.1094),
        "N": (3.2499985237759583, 0.17),
        "O": (2.95992190192351, 0.21),
        "F": (3.118145513491187, 0.061),
        "P": (3.741774101539736, 0.20),
        "S": (3.563594872561357, 0.25),
        "CL": (3.470936522384323, 0.265),
        "BR": (3.519049376418701, 0.32),
        "I": (4.009954999, 0.40),
        "B": (3.50, 0.095),
    }
    sigma, epsilon = params.get(element.upper(), (3.50, 0.10))
    return {"sigma": sigma, "epsilon": epsilon}


def molecule_charges(mol: Any) -> tuple[list[float], str]:
    atom_props = ("AM1BCCCharge", "am1bcc_charge")
    for prop_name in atom_props:
        charges: list[float] = []
        for atom in mol.GetAtoms():
            if not atom.HasProp(prop_name):
                break
            charges.append(float(atom.GetProp(prop_name)))
        if len(charges) == int(mol.GetNumAtoms()):
            return charges, "AM1-BCC"

    if mol.HasProp("am1bcc_charges_json"):
        raw = json.loads(str(mol.GetProp("am1bcc_charges_json")))
        if isinstance(raw, list) and len(raw) == int(mol.GetNumAtoms()):
            return [float(value) for value in raw], "AM1-BCC"

    legacy_props = ("PartialCharge", "_TriposPartialCharge")
    for prop_name in legacy_props:
        charges = []
        for atom in mol.GetAtoms():
            if not atom.HasProp(prop_name):
                break
            charges.append(float(atom.GetProp(prop_name)))
        if len(charges) == int(mol.GetNumAtoms()):
            return charges, prop_name

    raise ValueError("ligand SDF does not contain AM1-BCC charges")


def distance(left: Point, right: Point) -> float:
    return math.sqrt(
        (left[0] - right[0]) * (left[0] - right[0])
        + (left[1] - right[1]) * (left[1] - right[1])
        + (left[2] - right[2]) * (left[2] - right[2])
    )


def vector(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def norm(v: Point) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def angle_radians(a: Point, b: Point, c: Point) -> float:
    ba = vector(a, b)
    bc = vector(c, b)
    denom = max(norm(ba) * norm(bc), 1.0e-12)
    cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]) / denom))
    return math.acos(cosine)


def ligand_bonds(mol: Any, points: list[Point], offset: int) -> list[JsonObject]:
    bonds: list[JsonObject] = []
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        order = float(bond.GetBondTypeAsDouble())
        bonds.append(
            {
                "i": offset + i,
                "j": offset + j,
                "r0": distance(points[i], points[j]),
                "k": 700.0 + (order - 1.0) * 150.0,
                "source": "ligand_sdf",
            }
        )
    return bonds


def adjacency_from_mol(mol: Any) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {index: set() for index in range(int(mol.GetNumAtoms()))}
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def ligand_angles(mol: Any, points: list[Point], offset: int) -> list[JsonObject]:
    adjacency = adjacency_from_mol(mol)
    angles: list[JsonObject] = []
    for center, neighbors in adjacency.items():
        for i, k_idx in combinations(sorted(neighbors), 2):
            angles.append(
                {
                    "i": offset + i,
                    "j": offset + center,
                    "k_idx": offset + k_idx,
                    "theta0": angle_radians(points[i], points[center], points[k_idx]),
                    "force_k": 80.0,
                    "source": "ligand_sdf_geometry",
                }
            )
    return angles


def ligand_dihedrals(mol: Any, offset: int) -> list[JsonObject]:
    adjacency = adjacency_from_mol(mol)
    dihedrals: list[JsonObject] = []
    seen: set[tuple[int, int, int, int]] = set()
    for bond in mol.GetBonds():
        b = int(bond.GetBeginAtomIdx())
        c = int(bond.GetEndAtomIdx())
        for a in sorted(adjacency[b] - {c}):
            for d in sorted(adjacency[c] - {b}):
                key = (a, b, c, d)
                reverse_key = (d, c, b, a)
                if key in seen or reverse_key in seen:
                    continue
                seen.add(key)
                dihedrals.append(
                    {
                        "i": offset + a,
                        "j": offset + b,
                        "k_idx": offset + c,
                        "l": offset + d,
                        "periodicity": 3,
                        "phase": 0.0,
                        "force_k": 0.15555555555555556,
                        "source": "ligand_sdf_topology",
                    }
                )
    return dihedrals


def ligand_exclusions(mol: Any, offset: int) -> list[list[int]]:
    adjacency = adjacency_from_mol(mol)
    exclusions: list[set[int]] = [set() for _ in range(int(mol.GetNumAtoms()))]
    for i, neighbors in adjacency.items():
        for j in neighbors:
            exclusions[i].add(offset + j)
        second_neighbors: set[int] = set()
        for j in neighbors:
            second_neighbors.update(adjacency[j])
        second_neighbors.discard(i)
        for j in second_neighbors:
            exclusions[i].add(offset + j)
    return [sorted(values) for values in exclusions]


def heavy_atom_indices(elements: list[str]) -> list[int]:
    return [index for index, element in enumerate(elements) if element.upper() not in {"H", "D"}]


def collision_guard(
    receptor_points: list[Point],
    receptor_elements: list[str],
    ligand_coords: list[Point],
    ligand_elements: list[str],
    min_threshold: float,
    max_threshold: float,
) -> tuple[float, tuple[int, int]]:
    min_distance = math.inf
    min_pair = (-1, -1)
    receptor_heavy = heavy_atom_indices(receptor_elements)
    ligand_heavy = heavy_atom_indices(ligand_elements)
    for rec_index in receptor_heavy:
        rec_point = receptor_points[rec_index]
        for lig_index in ligand_heavy:
            current = distance(rec_point, ligand_coords[lig_index])
            if current < min_distance:
                min_distance = current
                min_pair = (rec_index, lig_index)
    if min_distance < min_threshold:
        raise ValueError(
            "ligand/receptor collision guard failed: "
            f"min_heavy_distance_A={min_distance:.3f} receptor_atom={min_pair[0]} ligand_atom={min_pair[1]}"
        )
    if min_distance > max_threshold:
        raise ValueError(
            "ligand/receptor contact guard failed: "
            f"min_heavy_distance_A={min_distance:.3f} exceeds max_allowed_A={max_threshold:.3f}"
        )
    return min_distance, min_pair


def max_residue_id(topology: JsonObject) -> int:
    residue_ids = [int(value) for value in ensure_list(topology, "residue_ids")]
    return max(residue_ids, default=0)


def append_ligand(
    topology: JsonObject,
    mol: Any,
    ligand_sdf: Path,
    min_heavy_distance_a: float,
    max_heavy_distance_a: float,
    require_am1bcc: bool,
) -> JsonObject:
    receptor_points = topology_points(topology)
    offset = int(topology.get("n_atoms", len(receptor_points)))
    if offset != len(receptor_points):
        raise ValueError("n_atoms does not match position array length")

    points = ligand_points(mol)
    ligand_elements = [str(atom.GetSymbol()) for atom in mol.GetAtoms()]
    receptor_elements = [str(value) for value in ensure_list(topology, "elements")]
    min_distance, min_pair = collision_guard(
        receptor_points,
        receptor_elements,
        points,
        ligand_elements,
        min_heavy_distance_a,
        max_heavy_distance_a,
    )
    charges, charge_method = molecule_charges(mol)
    if require_am1bcc and charge_method != "AM1-BCC":
        raise ValueError(f"ligand_charge_method must be AM1-BCC for MD, got {charge_method}")
    ligand_atom_count = int(mol.GetNumAtoms())
    selected_source_condition = (
        str(mol.GetProp("selected_source_condition"))
        if bool(mol.HasProp("selected_source_condition"))
        else "glp1r_6XOX_WT"
    )
    ligand_residue_index = int(topology.get("n_residues", len(ensure_list(topology, "residues"))))
    ligand_residue_id = max_residue_id(topology) + 1

    ensure_list(topology, "positions").extend(flat_positions(points))
    ensure_list(topology, "masses").extend([atomic_mass(element) for element in ligand_elements])
    ensure_list(topology, "elements").extend(ligand_elements)
    ensure_list(topology, "atom_names").extend(
        [f"{element}{atom_index + 1}" for atom_index, element in enumerate(ligand_elements)]
    )
    ensure_list(topology, "residue_names").extend([LIGAND_RESIDUE_NAME] * ligand_atom_count)
    ensure_list(topology, "residue_ids").extend([ligand_residue_id] * ligand_atom_count)
    ensure_list(topology, "chain_ids").extend([LIGAND_CHAIN_ID] * ligand_atom_count)
    ensure_list(topology, "charges").extend(charges)
    ensure_list(topology, "lj_params").extend([lj_params(element) for element in ligand_elements])
    ensure_list(topology, "gb_radii").extend([gb_radius(element) for element in ligand_elements])
    ensure_list(topology, "bonds").extend(ligand_bonds(mol, points, offset))
    ensure_list(topology, "angles").extend(ligand_angles(mol, points, offset))
    ensure_list(topology, "dihedrals").extend(ligand_dihedrals(mol, offset))
    ensure_list(topology, "exclusions").extend(ligand_exclusions(mol, offset))

    ensure_list(topology, "residues").append(
        {
            "residue_idx": ligand_residue_index,
            "residue_name": LIGAND_RESIDUE_NAME,
            "residue_id": ligand_residue_id,
            "chain_id": LIGAND_CHAIN_ID,
            "atom_start": offset,
            "atom_end": offset + ligand_atom_count - 1,
        }
    )
    ensure_dict(topology, "residue_to_atom_indices")[str(ligand_residue_index)] = list(
        range(offset, offset + ligand_atom_count)
    )
    serial_map = ensure_dict(topology, "pdb_atom_serial_to_topo_index")
    serial_start = max((int(key) for key in serial_map), default=0) + 1
    for atom_offset in range(ligand_atom_count):
        serial_map[str(serial_start + atom_offset)] = offset + atom_offset

    topology["n_atoms"] = int(offset + ligand_atom_count)
    topology["n_residues"] = ligand_residue_index + 1
    topology["n_chains"] = max(int(topology.get("n_chains", 1)), 2)
    topology["holo_state"] = True
    topology["condition_id"] = "glp1r_6XOX_HOLO_ALENI"
    topology["ligand_name"] = "ALENIGLIPRON"
    topology["ligand_residue_name"] = LIGAND_RESIDUE_NAME
    topology["ligand_atom_count"] = int(ligand_atom_count)
    topology["ligand_atoms"] = int(ligand_atom_count)
    topology["ligand_charge_method"] = str(charge_method)
    topology["ligand_source_sdf"] = ligand_sdf.as_posix()
    topology["selected_source_condition"] = str(selected_source_condition)
    topology["collision_guard_min_heavy_atom_distance_A"] = float(min_distance)
    topology["min_heavy_distance_A"] = float(min_distance)
    topology["collision_guard_required_min_A"] = float(min_heavy_distance_a)
    topology["collision_guard_required_max_A"] = float(max_heavy_distance_a)
    topology["collision_guard_min_pair"] = {
        "receptor_topology_atom_index": int(min_pair[0]),
        "ligand_local_atom_index": int(min_pair[1]),
    }
    topology["topology_merge_note"] = (
        "Holo topology produced by appending aligned Aleniglipron conformer 0 to the WT receptor topology. "
        "Ligand coordinates remain in the receptor coordinate frame from the source SDF."
    )
    return topology


def main() -> int:
    args = parse_args()
    apo_topology = Path(args.apo_topology)
    ligand_sdf = Path(args.ligand_sdf)
    output = Path(args.output)
    if not apo_topology.exists():
        raise FileNotFoundError(apo_topology)
    if not ligand_sdf.exists():
        raise FileNotFoundError(ligand_sdf)

    topology = read_json_object(apo_topology)
    mol = load_ligand(ligand_sdf)
    merged = append_ligand(
        topology,
        mol,
        ligand_sdf,
        float(args.min_heavy_distance_a),
        float(args.max_heavy_distance_a),
        not bool(args.allow_non_am1bcc),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2, sort_keys=False) + "\n")
    emit(
        f"wrote {output} n_atoms={merged['n_atoms']} ligand_atoms={merged['ligand_atom_count']} "
        f"ligand_charge_method={merged['ligand_charge_method']} "
        f"min_heavy_distance_A={float(merged['collision_guard_min_heavy_atom_distance_A']):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
