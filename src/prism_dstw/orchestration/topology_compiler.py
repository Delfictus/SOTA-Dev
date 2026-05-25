"""Compile generated ligands into PRISM-4D holo topology JSON files."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from importlib import import_module
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, cast


JsonObject = dict[str, Any]
Point3D = tuple[float, float, float]

DEFAULT_BASE_TOPOLOGY = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/"
    "20260518T031002Z/04_TOPOLOGIES/glp1r_6XOX_WT.topology.json"
)
DEFAULT_OUTPUT_DIR = Path("04_TOPOLOGIES")
LIGAND_CHAIN_ID = "L"
LIGAND_RESIDUE_NAME = "GEN"
MIN_HEAVY_DISTANCE_A = 1.5
MAX_CONTACT_DISTANCE_A = 4.0


@dataclass(frozen=True)
class TopologyCompileResult:
    generated_id: str
    topology_path: Path
    condition_id: str
    n_atoms: int
    ligand_atom_count: int
    ligand_charge_method: str
    min_heavy_distance_A: float


def _rdkit_modules() -> tuple[Any, Any]:
    chem = cast(Any, import_module("rdkit.Chem"))
    all_chem = cast(Any, import_module("rdkit.Chem.AllChem"))
    return chem, all_chem


def _read_json_object(path: Path) -> JsonObject:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected topology JSON object in {path}")
    return cast(JsonObject, loaded)


def _ensure_list(topology: JsonObject, key: str) -> list[Any]:
    value = topology.setdefault(key, [])
    if not isinstance(value, list):
        raise TypeError(f"topology[{key!r}] must be a list")
    return value


def _ensure_dict(topology: JsonObject, key: str) -> dict[str, Any]:
    value = topology.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"topology[{key!r}] must be a dict")
    return cast(dict[str, Any], value)


def _distance(left: Point3D, right: Point3D) -> float:
    return math.sqrt(
        (left[0] - right[0]) * (left[0] - right[0])
        + (left[1] - right[1]) * (left[1] - right[1])
        + (left[2] - right[2]) * (left[2] - right[2])
    )


def _vector(left: Point3D, right: Point3D) -> Point3D:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _norm(vector: Point3D) -> float:
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def _angle_radians(a: Point3D, b: Point3D, c: Point3D) -> float:
    ba = _vector(a, b)
    bc = _vector(c, b)
    denominator = max(_norm(ba) * _norm(bc), 1.0e-12)
    cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]) / denominator))
    return math.acos(cosine)


def _topology_points(topology: JsonObject) -> list[Point3D]:
    raw_positions = _ensure_list(topology, "positions")
    if len(raw_positions) % 3 != 0:
        raise ValueError("topology positions must be a flat xyz array")
    values = [float(value) for value in raw_positions]
    return [(values[index], values[index + 1], values[index + 2]) for index in range(0, len(values), 3)]


def _flat_positions(points: list[Point3D]) -> list[float]:
    flattened: list[float] = []
    for x_coord, y_coord, z_coord in points:
        flattened.extend([x_coord, y_coord, z_coord])
    return flattened


def _ligand_points(mol: Any) -> list[Point3D]:
    conformer = mol.GetConformer(0)
    points: list[Point3D] = []
    for atom_index in range(int(mol.GetNumAtoms())):
        position = conformer.GetAtomPosition(atom_index)
        points.append((float(position.x), float(position.y), float(position.z)))
    return points


def _heavy_indices(elements: list[str]) -> list[int]:
    return [index for index, element in enumerate(elements) if element.upper() not in {"H", "D"}]


def _periodic_table() -> Any:
    chem, _ = _rdkit_modules()
    return chem.GetPeriodicTable()


def _atomic_mass(element: str) -> float:
    return float(_periodic_table().GetAtomicWeight(element))


def _gb_radius(element: str) -> float:
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


def _lj_params(element: str) -> JsonObject:
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


def _adjacency(mol: Any) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {index: set() for index in range(int(mol.GetNumAtoms()))}
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        adjacency[begin].add(end)
        adjacency[end].add(begin)
    return adjacency


def _ligand_bonds(mol: Any, points: list[Point3D], offset: int) -> list[JsonObject]:
    bonds: list[JsonObject] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        order = float(bond.GetBondTypeAsDouble())
        bonds.append(
            {
                "i": offset + begin,
                "j": offset + end,
                "r0": _distance(points[begin], points[end]),
                "k": 700.0 + (order - 1.0) * 150.0,
                "source": "generated_ligand_openff_sage_geometry",
            }
        )
    return bonds


def _ligand_angles(mol: Any, points: list[Point3D], offset: int) -> list[JsonObject]:
    adjacency = _adjacency(mol)
    angles: list[JsonObject] = []
    for center, neighbors in adjacency.items():
        for left, right in combinations(sorted(neighbors), 2):
            angles.append(
                {
                    "i": offset + left,
                    "j": offset + center,
                    "k_idx": offset + right,
                    "theta0": _angle_radians(points[left], points[center], points[right]),
                    "force_k": 80.0,
                    "source": "generated_ligand_openff_sage_geometry",
                }
            )
    return angles


def _ligand_dihedrals(mol: Any, offset: int) -> list[JsonObject]:
    adjacency = _adjacency(mol)
    dihedrals: list[JsonObject] = []
    seen: set[tuple[int, int, int, int]] = set()
    for bond in mol.GetBonds():
        b_idx = int(bond.GetBeginAtomIdx())
        c_idx = int(bond.GetEndAtomIdx())
        for a_idx in sorted(adjacency[b_idx] - {c_idx}):
            for d_idx in sorted(adjacency[c_idx] - {b_idx}):
                key = (a_idx, b_idx, c_idx, d_idx)
                reverse_key = (d_idx, c_idx, b_idx, a_idx)
                if key in seen or reverse_key in seen:
                    continue
                seen.add(key)
                dihedrals.append(
                    {
                        "i": offset + a_idx,
                        "j": offset + b_idx,
                        "k_idx": offset + c_idx,
                        "l": offset + d_idx,
                        "periodicity": 3,
                        "phase": 0.0,
                        "force_k": 0.15555555555555556,
                        "source": "generated_ligand_openff_sage_topology",
                    }
                )
    return dihedrals


def _ligand_exclusions(mol: Any, offset: int) -> list[list[int]]:
    adjacency = _adjacency(mol)
    exclusions: list[set[int]] = [set() for _ in range(int(mol.GetNumAtoms()))]
    for atom_index, neighbors in adjacency.items():
        for neighbor in neighbors:
            exclusions[atom_index].add(offset + neighbor)
        second_neighbors: set[int] = set()
        for neighbor in neighbors:
            second_neighbors.update(adjacency[neighbor])
        second_neighbors.discard(atom_index)
        for neighbor in second_neighbors:
            exclusions[atom_index].add(offset + neighbor)
    return [sorted(values) for values in exclusions]


def _collision_minimum(
    receptor_points: list[Point3D],
    receptor_elements: list[str],
    ligand_points: list[Point3D],
    ligand_elements: list[str],
) -> tuple[float, tuple[int, int]]:
    receptor_heavy = _heavy_indices(receptor_elements)
    ligand_heavy = _heavy_indices(ligand_elements)
    minimum = math.inf
    min_pair = (-1, -1)
    for receptor_index in receptor_heavy:
        receptor_point = receptor_points[receptor_index]
        for ligand_index in ligand_heavy:
            distance = _distance(receptor_point, ligand_points[ligand_index])
            if distance < minimum:
                minimum = distance
                min_pair = (receptor_index, ligand_index)
    return minimum, min_pair


def _translate_mol(mol: Any, delta: Point3D) -> None:
    conformer = mol.GetConformer(0)
    for atom_index in range(int(mol.GetNumAtoms())):
        position = conformer.GetAtomPosition(atom_index)
        position.x += delta[0]
        position.y += delta[1]
        position.z += delta[2]
        conformer.SetAtomPosition(atom_index, position)


class PrismTopologyCompiler:
    """Compile GFlowNet-generated RDKit molecules into PRISM-4D topologies."""

    def __init__(
        self,
        *,
        base_receptor_topology: Path = DEFAULT_BASE_TOPOLOGY,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        min_heavy_distance_A: float = MIN_HEAVY_DISTANCE_A,
        max_contact_distance_A: float = MAX_CONTACT_DISTANCE_A,
    ) -> None:
        self.base_receptor_topology = base_receptor_topology
        self.output_dir = output_dir
        self.min_heavy_distance_A = min_heavy_distance_A
        self.max_contact_distance_A = max_contact_distance_A

    def _prepare_conformer(self, mol: Any) -> Any:
        chem, all_chem = _rdkit_modules()
        prepared = chem.Mol(mol)
        if int(prepared.GetNumConformers()) == 0:
            prepared = chem.AddHs(prepared, addCoords=True)
            params = all_chem.ETKDGv3()
            params.randomSeed = 91_721
            status = int(all_chem.EmbedMolecule(prepared, params))
            if status != 0:
                raise ValueError(f"RDKit ETKDGv3 embedding failed with code {status}")
        if bool(all_chem.MMFFHasAllMoleculeParams(prepared)):
            all_chem.MMFFOptimizeMolecule(prepared, mmffVariant="MMFF94s", maxIters=200)
        else:
            all_chem.UFFOptimizeMolecule(prepared, maxIters=200)
        return prepared

    def _assign_openff_parameters(self, mol: Any) -> tuple[Any, list[float], str]:
        try:
            toolkit = import_module("openff.toolkit")
        except ImportError as exc:
            raise RuntimeError("openff-toolkit is required for generated-ligand AM1-BCC assignment") from exc
        molecule_class = cast(Any, getattr(toolkit, "Molecule"))
        force_field_class = cast(Any, getattr(toolkit, "ForceField"))
        off_mol = molecule_class.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
        off_mol.assign_partial_charges(partial_charge_method="am1bcc")
        charges = [float(charge.m_as("elementary_charge")) for charge in off_mol.partial_charges]
        if len(charges) != int(mol.GetNumAtoms()):
            raise ValueError("OpenFF charge vector length does not match RDKit atom count")
        for atom_index, charge in enumerate(charges):
            atom = mol.GetAtomWithIdx(atom_index)
            atom.SetDoubleProp("AM1BCCCharge", charge)
            atom.SetProp("am1bcc_charge", f"{charge:.12f}")
        mol.SetProp("am1bcc_charges_json", json.dumps(charges, separators=(",", ":")))
        forcefield_name = "openff-2.2.0.offxml"
        try:
            force_field_class(forcefield_name).create_openmm_system(off_mol.to_topology())
        except Exception as exc:
            raise RuntimeError(f"OpenFF Sage parameter assignment failed for {forcefield_name}") from exc
        return mol, charges, f"AM1-BCC/{forcefield_name}+MMFF94s_minimized"

    def _resolve_minor_clashes(self, topology: JsonObject, mol: Any) -> tuple[float, tuple[int, int]]:
        _, all_chem = _rdkit_modules()
        if bool(all_chem.MMFFHasAllMoleculeParams(mol)):
            all_chem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=100)
        else:
            all_chem.UFFOptimizeMolecule(mol, maxIters=100)
        receptor_points = _topology_points(topology)
        receptor_elements = [str(value) for value in _ensure_list(topology, "elements")]
        ligand_elements = [str(atom.GetSymbol()) for atom in mol.GetAtoms()]
        min_distance, min_pair = _collision_minimum(receptor_points, receptor_elements, _ligand_points(mol), ligand_elements)
        if min_distance < self.min_heavy_distance_A:
            receptor_point = receptor_points[min_pair[0]]
            ligand_point = _ligand_points(mol)[min_pair[1]]
            direction = _vector(ligand_point, receptor_point)
            direction_norm = max(_norm(direction), 1.0e-9)
            required = self.min_heavy_distance_A - min_distance + 0.05
            delta = tuple(required * component / direction_norm for component in direction)
            _translate_mol(mol, cast(Point3D, delta))
            min_distance, min_pair = _collision_minimum(
                receptor_points,
                receptor_elements,
                _ligand_points(mol),
                ligand_elements,
            )
        if min_distance < self.min_heavy_distance_A:
            raise ValueError(
                "generated ligand collision guard failed: "
                f"min_heavy_distance_A={min_distance:.3f} receptor_atom={min_pair[0]} ligand_atom={min_pair[1]}"
            )
        if min_distance > self.max_contact_distance_A:
            raise ValueError(
                "generated ligand contact guard failed: "
                f"min_heavy_distance_A={min_distance:.3f} exceeds max_contact_A={self.max_contact_distance_A:.3f}"
            )
        return min_distance, min_pair

    def _append_ligand(
        self,
        topology: JsonObject,
        mol: Any,
        charges: list[float],
        charge_method: str,
        generated_id: str,
    ) -> tuple[JsonObject, float]:
        receptor_points = _topology_points(topology)
        offset = int(topology.get("n_atoms", len(receptor_points)))
        if offset != len(receptor_points):
            raise ValueError("topology n_atoms does not match position array length")
        min_distance, min_pair = self._resolve_minor_clashes(topology, mol)
        points = _ligand_points(mol)
        elements = [str(atom.GetSymbol()) for atom in mol.GetAtoms()]
        ligand_atom_count = int(mol.GetNumAtoms())
        ligand_residue_index = int(topology.get("n_residues", len(_ensure_list(topology, "residues"))))
        residue_ids = [int(value) for value in _ensure_list(topology, "residue_ids")]
        ligand_residue_id = max(residue_ids, default=0) + 1

        _ensure_list(topology, "positions").extend(_flat_positions(points))
        _ensure_list(topology, "masses").extend([_atomic_mass(element) for element in elements])
        _ensure_list(topology, "elements").extend(elements)
        _ensure_list(topology, "atom_names").extend([f"{element}{index + 1}" for index, element in enumerate(elements)])
        _ensure_list(topology, "residue_names").extend([LIGAND_RESIDUE_NAME] * ligand_atom_count)
        _ensure_list(topology, "residue_ids").extend([ligand_residue_id] * ligand_atom_count)
        _ensure_list(topology, "chain_ids").extend([LIGAND_CHAIN_ID] * ligand_atom_count)
        _ensure_list(topology, "charges").extend(charges)
        _ensure_list(topology, "lj_params").extend([_lj_params(element) for element in elements])
        _ensure_list(topology, "gb_radii").extend([_gb_radius(element) for element in elements])
        _ensure_list(topology, "bonds").extend(_ligand_bonds(mol, points, offset))
        _ensure_list(topology, "angles").extend(_ligand_angles(mol, points, offset))
        _ensure_list(topology, "dihedrals").extend(_ligand_dihedrals(mol, offset))
        _ensure_list(topology, "exclusions").extend(_ligand_exclusions(mol, offset))
        _ensure_list(topology, "residues").append(
            {
                "residue_idx": ligand_residue_index,
                "residue_name": LIGAND_RESIDUE_NAME,
                "residue_id": ligand_residue_id,
                "chain_id": LIGAND_CHAIN_ID,
                "atom_start": offset,
                "atom_end": offset + ligand_atom_count - 1,
            }
        )
        _ensure_dict(topology, "residue_to_atom_indices")[str(ligand_residue_index)] = list(
            range(offset, offset + ligand_atom_count)
        )
        serial_map = _ensure_dict(topology, "pdb_atom_serial_to_topo_index")
        serial_start = max((int(key) for key in serial_map), default=0) + 1
        for atom_offset in range(ligand_atom_count):
            serial_map[str(serial_start + atom_offset)] = offset + atom_offset

        condition_id = f"glp1r_6XOX_HOLO_{generated_id}"
        topology["n_atoms"] = offset + ligand_atom_count
        topology["n_residues"] = ligand_residue_index + 1
        topology["n_chains"] = max(int(topology.get("n_chains", 1)), 2)
        topology["condition_id"] = condition_id
        topology["holo_state"] = True
        topology["generated_ligand_id"] = generated_id
        topology["ligand_residue_name"] = LIGAND_RESIDUE_NAME
        topology["ligand_atom_count"] = ligand_atom_count
        topology["ligand_charge_method"] = charge_method
        topology["forcefield_parameterization"] = charge_method
        topology["collision_guard_min_heavy_atom_distance_A"] = min_distance
        topology["collision_guard_required_min_A"] = self.min_heavy_distance_A
        topology["collision_guard_required_max_A"] = self.max_contact_distance_A
        topology["collision_guard_min_pair"] = {
            "receptor_topology_atom_index": min_pair[0],
            "ligand_local_atom_index": min_pair[1],
        }
        topology["topology_merge_note"] = (
            "Generated ligand topology compiled by PRISM-DSTW Track A using OpenFF AM1-BCC/Sage "
            "parameterization and RDKit in-situ clash relief before appending to the GLP-1R receptor topology."
        )
        return topology, min_distance

    def compile_molecule(
        self,
        mol: Any,
        *,
        generated_id: str,
        metadata: Mapping[str, object] | None = None,
    ) -> TopologyCompileResult:
        if not self.base_receptor_topology.exists():
            raise FileNotFoundError(self.base_receptor_topology)
        safe_generated_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in generated_id)
        if safe_generated_id == "":
            raise ValueError("generated_id must contain at least one filesystem-safe character")
        topology = copy.deepcopy(_read_json_object(self.base_receptor_topology))
        prepared = self._prepare_conformer(mol)
        parameterized, charges, charge_method = self._assign_openff_parameters(prepared)
        merged, min_distance = self._append_ligand(topology, parameterized, charges, charge_method, safe_generated_id)
        if metadata:
            merged["generated_ligand_metadata"] = dict(metadata)
        output = self.output_dir / f"glp1r_6XOX_HOLO_{safe_generated_id}.topology.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(merged, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        return TopologyCompileResult(
            generated_id=safe_generated_id,
            topology_path=output,
            condition_id=str(merged["condition_id"]),
            n_atoms=int(merged["n_atoms"]),
            ligand_atom_count=int(merged["ligand_atom_count"]),
            ligand_charge_method=str(merged["ligand_charge_method"]),
            min_heavy_distance_A=float(min_distance),
        )


__all__ = ["PrismTopologyCompiler", "TopologyCompileResult"]
