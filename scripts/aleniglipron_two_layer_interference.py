#!/usr/bin/env python3
"""Run two-layer whole-molecule and BRICS-fragment interference for aleniglipron."""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import polars as pl
from jinja2 import Environment, StrictUndefined
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors, Lipinski, SaltRemover, SDWriter
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import sha256_path, write_provenance_parquet
from prism_dstw.propagation_ledger import append_propagation_entry, build_entry


JsonObject: TypeAlias = dict[str, object]
Coordinate: TypeAlias = tuple[float, float, float]

TRACK0_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation"
N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
DEFAULT_GRID_MAPPING = TRACK0_DIR / "grid_coordinate_mapping.json"
DEFAULT_BINDING_SITE = TRACK0_DIR / "binding_site_reference.json"
DEFAULT_STERIC_ENV = TRACK0_DIR / "interface_steric_environment.parquet"
DEFAULT_RISK_MAP = N80_DIR / "receptor_durability_risk_map.parquet"
DEFAULT_SIGNAL_GRID = N80_DIR / "signal_grid_variance_channel.parquet"
CONFORMER_DIR = TRACK0_DIR / "conformers"
LAYER1_DIR = TRACK0_DIR / "layer1_whole_molecule"
LAYER2_DIR = TRACK0_DIR / "layer2_fragments"
ATTRIBUTION_PATH = TRACK0_DIR / "fragment_interference_attribution.parquet"
SUMMARY_PATH = TRACK0_DIR / "aleniglipron_interference_summary.md"
ANALOG_REGISTRY_PATH = TRACK0_DIR / "analog_registry.json"
PUBCHEM_RESPONSE_PATH = TRACK0_DIR / "pubchem_cid_164809721_canonical_smiles.json"
FRAGMENT_REGISTRY_PATH = TRACK0_DIR / "aleniglipron_brics_fragment_registry.json"
GATE_REPORT_PATH = TRACK0_DIR / "aleniglipron_two_layer_gate_report.json"
ALENIGLIPRON_CID = 164809721
K_CONFORMERS = 10

SUMMARY_TEMPLATE = """## Aleniglipron Thermodynamic Field Interference Analysis

### Layer 1: Whole-Molecule Scoring
- Total Pi_clash across 9 critical edges: {{ "%.4f"|format(total_clash) }}
- Total Pi_complement: {{ "%.4f"|format(total_complement) }}
- Projected durability score: {{ "%.4f"|format(projected_score) }} +/- {{ "%.4f"|format(uncertainty) }}
- Confidence class: {{ confidence_class }}

### Layer 2: Fragment Decomposition (BRICS)
- Fragments identified: {{ n_fragments }}
- Dominant liability fragment: {{ worst_fragment_id }} ({{ worst_fragment_smiles }}, {{ "%.1f"|format(worst_fraction * 100.0) }}% of total clash)
- Inter-fragment coupling: {{ "%.4f"|format(total_coupling) }}

### Per-Edge Attribution
| Edge | Whole Clash | Sum Fragment Clash | Coupling | Dominant Fragment | Dominant Fraction |
|---|---:|---:|---:|---|---:|
{% for row in attribution_rows -%}
| {{ row.edge_id }} | {{ "%.4f"|format(row.whole_molecule_clash) }} | {{ "%.4f"|format(row.sum_fragment_clash) }} | {{ "%.4f"|format(row.inter_fragment_coupling) }} | {{ row.dominant_fragment }} | {{ "%.1f"|format(row.dominant_fraction * 100.0) }}% |
{% endfor %}

### Structural Rationales
{% for edge in attribution_rows -%}
**{{ edge.edge_id }}:** {{ edge.rationale }}

{% endfor -%}

### Recommendation
{{ recommendation_text }}
"""


@dataclass(frozen=True)
class CleanMolecule:
    raw_smiles: str
    clean_smiles: str
    mol: Any
    mw: float
    n_heavy_atoms: int
    rotatable_bonds: int
    salt_stripped: bool


@dataclass(frozen=True)
class ConformerSet:
    mol_h: Any
    top_conformers: list[tuple[int, float]]
    sdf_path: Path
    alignment_method: str
    energy_range: tuple[float, float]
    energy_spread: float
    first_atom_validation: JsonObject


@dataclass(frozen=True)
class FragmentInfo:
    fragment_id: str
    brics_smiles: str
    parent_atom_indices: tuple[int, ...]
    n_heavy_atoms: int
    sdf_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--binding-site", type=Path, default=DEFAULT_BINDING_SITE)
    parser.add_argument("--steric-env", type=Path, default=DEFAULT_STERIC_ENV)
    parser.add_argument("--risk-map", type=Path, default=DEFAULT_RISK_MAP)
    parser.add_argument("--signal-grid", type=Path, default=DEFAULT_SIGNAL_GRID)
    parser.add_argument("--beta-f", default="auto")
    parser.add_argument("--beta-s", default="auto")
    parser.add_argument("--regenerate-conformers", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def as_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError(f"{label} must be numeric")


def as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got bool")
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError(f"{label} must be an integer")


def json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a JSON list")
    return value


def coordinate_from_json(value: object, label: str) -> Coordinate:
    raw = json_list(value, label)
    if len(raw) != 3:
        raise ValueError(f"{label} must contain 3 values")
    return (
        as_float(raw[0], f"{label}[0]"),
        as_float(raw[1], f"{label}[1]"),
        as_float(raw[2], f"{label}[2]"),
    )


def write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def retrieve_pubchem_smiles() -> tuple[str, JsonObject]:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{ALENIGLIPRON_CID}/property/CanonicalSMILES/JSON"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Prism4D/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("PubChem response was not a JSON object")
    write_json(PUBCHEM_RESPONSE_PATH, payload)
    property_table = json_object(payload["PropertyTable"], "PropertyTable")
    rows = json_list(property_table["Properties"], "Properties")
    first = json_object(rows[0], "Properties[0]")
    raw_smiles = first.get("CanonicalSMILES", first.get("ConnectivitySMILES"))
    if raw_smiles is None:
        raise ValueError(f"PubChem response did not contain a SMILES field: {first}")
    smiles = str(raw_smiles)
    return smiles, payload


def clean_and_validate_molecule(raw_smiles: str) -> CleanMolecule:
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse PubChem SMILES: {raw_smiles}")
    remover = SaltRemover.SaltRemover()  # type: ignore[no-untyped-call]
    stripped = remover.StripMol(mol)  # type: ignore[no-untyped-call]
    salt_stripped = Chem.MolToSmiles(stripped) != Chem.MolToSmiles(mol)
    if "." in Chem.MolToSmiles(stripped):
        fragments = Chem.GetMolFrags(stripped, asMols=True)
        stripped = max(fragments, key=lambda item: item.GetNumAtoms())
        salt_stripped = True
    clean_smiles = Chem.MolToSmiles(stripped)
    clean_mol = Chem.MolFromSmiles(clean_smiles)
    if clean_mol is None:
        raise ValueError(f"sanitization failed for desalted SMILES: {clean_smiles}")
    mw = float(Descriptors.MolWt(clean_mol))  # type: ignore[attr-defined]
    n_heavy = int(clean_mol.GetNumHeavyAtoms())
    rotatable = int(Lipinski.NumRotatableBonds(clean_mol))  # type: ignore[attr-defined]
    if n_heavy <= 15:
        raise ValueError(f"too few atoms ({n_heavy}) for parent aleniglipron")
    if n_heavy >= 150:
        raise ValueError(f"too many atoms ({n_heavy}) for parent aleniglipron")
    if not 200.0 < mw < 1200.0:
        raise ValueError(f"MW {mw:.1f} outside extended drug range")
    return CleanMolecule(
        raw_smiles=raw_smiles,
        clean_smiles=clean_smiles,
        mol=clean_mol,
        mw=mw,
        n_heavy_atoms=n_heavy,
        rotatable_bonds=rotatable,
        salt_stripped=salt_stripped,
    )


def write_analog_registry(clean: CleanMolecule) -> None:
    write_json(
        ANALOG_REGISTRY_PATH,
        {
            "schema_version": 2,
            "retrieval_timestamp": datetime.now(UTC).isoformat(),
            "source": f"PubChem CID {ALENIGLIPRON_CID}",
            "raw_smiles": clean.raw_smiles,
            "desalted_smiles": clean.clean_smiles,
            "mw_desalted": clean.mw,
            "n_heavy_atoms": clean.n_heavy_atoms,
            "rotatable_bonds": clean.rotatable_bonds,
            "salt_stripped": clean.salt_stripped,
            "validation_status": "accepted_relaxed_gate",
            "analogs": [
                {
                    "analog_id": "ALENI-PARENT",
                    "canonical_smiles": clean.clean_smiles,
                    "source": f"PubChem CID {ALENIGLIPRON_CID}",
                    "modification_class": "parent_clinical_candidate",
                    "mw": clean.mw,
                }
            ],
        },
    )


def load_binding_site_center(path: Path) -> tuple[Coordinate, float, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not decode to a JSON object")
    center_value = payload.get("binding_site_center_angstrom")
    if center_value is None:
        center_value = json_object(payload["reference"], "reference")["centroid_xyz_angstrom"]
    radius_value = payload.get("binding_site_radius_angstrom", 15.0)
    ref_ligand = payload.get("reference_ligand_coords")
    if ref_ligand is None:
        return coordinate_from_json(center_value, "binding_site_center"), as_float(radius_value, "binding_site_radius"), "centroid_translation_to_binding_site"
    return coordinate_from_json(center_value, "binding_site_center"), as_float(radius_value, "binding_site_radius"), "O3A_to_reference_ligand"


def generate_parent_conformers(clean: CleanMolecule, binding_site: Path, grid_mapping: Path, signal_grid: Path) -> ConformerSet:
    mol_h = Chem.AddHs(clean.mol)
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = 42
    params.numThreads = 0
    params.pruneRmsThresh = 0.5
    generated = [int(conf_id) for conf_id in AllChem.EmbedMultipleConfs(mol_h, 50, params)]  # type: ignore[attr-defined]
    if len(generated) < K_CONFORMERS:
        raise ValueError(f"only generated {len(generated)} conformers for aleniglipron")
    if not AllChem.MMFFHasAllMoleculeParams(mol_h):  # type: ignore[attr-defined]
        raise ValueError("MMFF parameters are unavailable for aleniglipron")
    results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=0, maxIters=1000)  # type: ignore[attr-defined]
    energies = [
        (conf_id, float(result[1]))
        for conf_id, result in zip(generated, results, strict=True)
        if int(result[0]) == 0
    ]
    if len(energies) < K_CONFORMERS:
        raise ValueError(f"only {len(energies)} conformers optimized cleanly")
    energies.sort(key=lambda item: item[1])
    top_k = energies[:K_CONFORMERS]
    for conf_idx, _energy in top_k:
        positions = mol_h.GetConformer(conf_idx).GetPositions()
        min_dist = float(pdist(positions).min())
        if min_dist <= 0.7:
            raise ValueError(f"conformer {conf_idx} has atom clash: min_dist={min_dist:.2f} A")
    center, radius, alignment_method = load_binding_site_center(binding_site)
    for conf_idx, _energy in top_k:
        conf = mol_h.GetConformer(conf_idx)
        positions = conf.GetPositions()
        centroid = positions.mean(axis=0)
        translation = (
            center[0] - float(centroid[0]),
            center[1] - float(centroid[1]),
            center[2] - float(centroid[2]),
        )
        for atom_idx in range(mol_h.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            conf.SetAtomPosition(
                atom_idx,
                (
                    float(pos.x) + translation[0],
                    float(pos.y) + translation[1],
                    float(pos.z) + translation[2],
                ),
            )
        shifted = conf.GetPositions().mean(axis=0)
        distance_to_site = math.dist((float(shifted[0]), float(shifted[1]), float(shifted[2])), center)
        if distance_to_site >= radius:
            raise ValueError(f"conformer {conf_idx} centroid is {distance_to_site:.2f} A from binding site center")
    validation = validate_first_heavy_atom(mol_h, top_k[0][0], grid_mapping, signal_grid)
    if validation["in_bounds"] is not True or validation["variance_class"] == "void":
        raise ValueError(f"alignment validation failed: {validation}")
    CONFORMER_DIR.mkdir(parents=True, exist_ok=True)
    sdf_path = CONFORMER_DIR / "ALENI-PARENT_whole_molecule_aligned.sdf"
    writer = SDWriter(str(sdf_path))
    for conf_idx, energy in top_k:
        mol_h.SetProp("_Name", f"ALENI-PARENT_conf{conf_idx}")
        mol_h.SetProp("mmff_energy_kcal", f"{energy:.2f}")
        mol_h.SetProp("layer", "whole_molecule")
        mol_h.SetProp("alignment_method", alignment_method)
        mol_h.SetProp("aligned_to_receptor", "true")
        writer.write(mol_h, confId=conf_idx)
    writer.close()
    return ConformerSet(
        mol_h=mol_h,
        top_conformers=top_k,
        sdf_path=sdf_path,
        alignment_method=alignment_method,
        energy_range=(top_k[0][1], top_k[-1][1]),
        energy_spread=top_k[-1][1] - top_k[0][1],
        first_atom_validation=validation,
    )


def validate_first_heavy_atom(mol_h: Any, conf_id: int, grid_mapping: Path, signal_grid: Path) -> JsonObject:
    payload = json.loads(grid_mapping.read_text(encoding="utf-8"))
    conditions = json_object(json_object(payload, "grid_mapping")["conditions"], "conditions")
    condition_id = "glp1r_5VEX_WT"
    geometry = json_object(conditions[condition_id], f"conditions[{condition_id}]")
    origin = coordinate_from_json(geometry["origin_xyz_angstrom"], "origin_xyz_angstrom")
    spacing = as_float(geometry["spacing_angstrom"], "spacing_angstrom")
    grid_dim = as_int(geometry["grid_dim"], "grid_dim")
    heavy_indices = [int(atom.GetIdx()) for atom in mol_h.GetAtoms() if int(atom.GetAtomicNum()) > 1]
    atom_idx = heavy_indices[0]
    pos = mol_h.GetConformer(conf_id).GetAtomPosition(atom_idx)
    coord = (float(pos.x), float(pos.y), float(pos.z))
    ix = math.trunc((coord[0] - origin[0]) / spacing)
    iy = math.trunc((coord[1] - origin[1]) / spacing)
    iz = math.trunc((coord[2] - origin[2]) / spacing)
    in_bounds = 0 <= ix < grid_dim and 0 <= iy < grid_dim and 0 <= iz < grid_dim
    voxel_idx = iz * grid_dim * grid_dim + iy * grid_dim + ix
    variance_class = "out_of_bounds"
    if in_bounds:
        rows = (
            pl.scan_parquet(signal_grid)
            .filter((pl.col("condition_id") == condition_id) & (pl.col("voxel_idx") == voxel_idx))
            .select("variance_class")
            .collect()
            .get_column("variance_class")
            .to_list()
        )
        if rows:
            variance_class = str(rows[0])
    return {
        "condition_id": condition_id,
        "atom_idx": atom_idx,
        "xyz_angstrom": [coord[0], coord[1], coord[2]],
        "grid_indices": [ix, iy, iz],
        "grid_dim": grid_dim,
        "voxel_idx": voxel_idx,
        "in_bounds": in_bounds,
        "variance_class": variance_class,
    }


def run_interference_tool(
    sdf_path: Path,
    out_dir: Path,
    grid_mapping: Path,
    binding_site: Path,
    steric_env: Path,
    risk_map: Path,
    beta_f: float | str,
    beta_s: float | str,
    normalization_source: Path,
) -> None:
    command = [
        sys.executable,
        "scripts/compute_scaffold_interference.py",
        "--sdf",
        sdf_path.as_posix(),
        "--grid-mapping",
        grid_mapping.as_posix(),
        "--binding-site",
        binding_site.as_posix(),
        "--steric-env",
        steric_env.as_posix(),
        "--risk-map",
        risk_map.as_posix(),
        "--normalization-source",
        normalization_source.as_posix(),
        "--beta-f",
        str(beta_f),
        "--beta-s",
        str(beta_s),
        "--out-dir",
        out_dir.as_posix(),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def validate_layer_output(out_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    edges = pl.read_parquet(out_dir / "per_edge_interference.parquet")
    projection = pl.read_parquet(out_dir / "analog_durability_projection.parquet")
    if edges.height != 9:
        raise ValueError(f"{out_dir} scored {edges.height} edges, expected 9")
    nan_count = edges.select(
        pl.sum_horizontal(
            [
                pl.col("E_pi_clash").is_nan().cast(pl.UInt32),
                pl.col("E_pi_complement").is_nan().cast(pl.UInt32),
                pl.col("U_pose").is_nan().cast(pl.UInt32),
                pl.col("te_multiplier").is_nan().cast(pl.UInt32),
            ]
        )
        .sum()
        .alias("nan_count")
    )["nan_count"][0]
    if as_int(nan_count, "nan_count") != 0:
        raise ValueError(f"{out_dir} contains NaN edge scores")
    total = as_float(projection["total_projected_durability"][0], "total_projected_durability")
    if total < 0.0:
        raise ValueError(f"{out_dir} produced negative total projected durability: {total}")
    return edges, projection


def fragment_parent_atom_sets(clean_mol: Any) -> list[tuple[str, tuple[int, ...]]]:
    brics_bonds = list(BRICS.FindBRICSBonds(clean_mol))  # type: ignore[no-untyped-call]
    bond_indices: list[int] = []
    for atom_pair, _labels in brics_bonds:
        bond = clean_mol.GetBondBetweenAtoms(int(atom_pair[0]), int(atom_pair[1]))
        if bond is None:
            raise ValueError(f"BRICS bond {atom_pair} not present in parent")
        bond_indices.append(int(bond.GetIdx()))
    if not bond_indices:
        raise ValueError("BRICS found no bonds to cut")
    fragmented = Chem.FragmentOnBonds(clean_mol, bond_indices, addDummies=True)
    atom_sets: list[tuple[str, tuple[int, ...]]] = []
    for frag_atoms in Chem.GetMolFrags(fragmented, asMols=False, sanitizeFrags=True):
        parent_indices = tuple(
            sorted(
                int(atom_idx)
                for atom_idx in frag_atoms
                if int(atom_idx) < int(clean_mol.GetNumAtoms())
                and int(clean_mol.GetAtomWithIdx(int(atom_idx)).GetAtomicNum()) > 1
            )
        )
        if len(parent_indices) < 3:
            continue
        brics_smiles = Chem.MolFragmentToSmiles(
            fragmented,
            atomsToUse=list(frag_atoms),
            isomericSmiles=True,
            canonical=True,
        )
        atom_sets.append((brics_smiles, parent_indices))
    unique: dict[tuple[int, ...], str] = {}
    for brics_smiles, parent_indices in atom_sets:
        unique.setdefault(parent_indices, brics_smiles)
    return [(smiles, atoms) for atoms, smiles in sorted(unique.items(), key=lambda item: item[0])]


def build_fragment_mol(clean_mol: Any, parent_atom_indices: tuple[int, ...]) -> tuple[Any, dict[int, int]]:
    index_map: dict[int, int] = {}
    rw_mol = Chem.RWMol()
    for parent_idx in parent_atom_indices:
        parent_atom = clean_mol.GetAtomWithIdx(parent_idx)
        atom = Chem.Atom(int(parent_atom.GetAtomicNum()))
        atom.SetFormalCharge(int(parent_atom.GetFormalCharge()))
        fragment_idx = int(rw_mol.AddAtom(atom))
        index_map[parent_idx] = fragment_idx
    fragment = rw_mol.GetMol()
    fragment.UpdatePropertyCache(strict=False)
    return fragment, index_map


def slice_fragments(clean: CleanMolecule, parent_sdf: Path) -> list[FragmentInfo]:
    supplier = Chem.SDMolSupplier(str(parent_sdf), removeHs=False)
    parent_conformers = [mol for mol in supplier if mol is not None]
    if len(parent_conformers) < K_CONFORMERS:
        raise ValueError(f"expected {K_CONFORMERS} parent conformers, got {len(parent_conformers)}")
    clean_parent = Chem.RemoveHs(parent_conformers[0])
    atom_sets = fragment_parent_atom_sets(clean_parent)
    fragments: list[FragmentInfo] = []
    for idx, (brics_smiles, parent_indices) in enumerate(atom_sets):
        fragment_id = f"FRAG-{chr(65 + idx)}"
        fragment_mol, index_map = build_fragment_mol(clean_parent, parent_indices)
        sdf_path = CONFORMER_DIR / f"{fragment_id}_sliced.sdf"
        writer = SDWriter(str(sdf_path))
        for conf_i, parent_conf_mol in enumerate(parent_conformers[:K_CONFORMERS]):
            parent_conf = parent_conf_mol.GetConformer()
            mol_copy = Chem.Mol(fragment_mol)
            conformer = Chem.Conformer(mol_copy.GetNumAtoms())
            for parent_idx in parent_indices:
                pos = parent_conf.GetAtomPosition(parent_idx)
                conformer.SetAtomPosition(
                    index_map[parent_idx],
                    (float(pos.x), float(pos.y), float(pos.z)),
                )
            mol_copy.AddConformer(conformer, assignId=True)
            mol_copy.SetProp("_Name", f"{fragment_id}_parent_conf{conf_i}")
            mol_copy.SetProp("layer", "brics_fragment_sliced")
            mol_copy.SetProp("parent", "ALENI-PARENT")
            mol_copy.SetProp("sliced_from_parent_conf", str(conf_i))
            mol_copy.SetProp("alignment_inherited", "true")
            mol_copy.SetProp("brics_smiles", brics_smiles)
            writer.write(mol_copy, confId=0)
        writer.close()
        fragments.append(
            FragmentInfo(
                fragment_id=fragment_id,
                brics_smiles=brics_smiles,
                parent_atom_indices=parent_indices,
                n_heavy_atoms=len(parent_indices),
                sdf_path=sdf_path,
            )
        )
    total_frag_heavy = sum(fragment.n_heavy_atoms for fragment in fragments)
    if abs(total_frag_heavy - clean.n_heavy_atoms) >= 10:
        raise ValueError(f"fragment atom count mismatch: {total_frag_heavy} vs {clean.n_heavy_atoms}")
    validate_fragment_coordinate_identity(parent_conformers[0], fragments)
    write_json(
        FRAGMENT_REGISTRY_PATH,
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "parent": "ALENI-PARENT",
            "parent_heavy_atoms": clean.n_heavy_atoms,
            "sum_fragment_heavy_atoms": total_frag_heavy,
            "fragments": [
                {
                    "fragment_id": fragment.fragment_id,
                    "brics_smiles": fragment.brics_smiles,
                    "parent_atom_indices": list(fragment.parent_atom_indices),
                    "n_heavy_atoms": fragment.n_heavy_atoms,
                    "sdf_path": fragment.sdf_path.relative_to(REPO_ROOT).as_posix(),
                }
                for fragment in fragments
            ],
        },
    )
    return fragments


def validate_fragment_coordinate_identity(parent_conf_mol: Any, fragments: list[FragmentInfo]) -> None:
    parent_conf = parent_conf_mol.GetConformer()
    for fragment in fragments:
        supplier = Chem.SDMolSupplier(str(fragment.sdf_path), removeHs=False)
        frag_mol = next(mol for mol in supplier if mol is not None)
        frag_conf = frag_mol.GetConformer()
        for fragment_idx, parent_idx in enumerate(fragment.parent_atom_indices[:3]):
            parent_pos = parent_conf.GetAtomPosition(parent_idx)
            frag_pos = frag_conf.GetAtomPosition(fragment_idx)
            distance = math.dist(
                (float(parent_pos.x), float(parent_pos.y), float(parent_pos.z)),
                (float(frag_pos.x), float(frag_pos.y), float(frag_pos.z)),
            )
            if distance > 1.0e-3:
                raise ValueError(
                    f"{fragment.fragment_id} coordinate identity failed for parent atom {parent_idx}: {distance:.6f} A"
                )


def validate_first_heavy_atom_from_mol(mol: Any, grid_mapping: Path, signal_grid: Path) -> JsonObject:
    payload = json.loads(grid_mapping.read_text(encoding="utf-8"))
    conditions = json_object(json_object(payload, "grid_mapping")["conditions"], "conditions")
    condition_id = "glp1r_5VEX_WT"
    geometry = json_object(conditions[condition_id], f"conditions[{condition_id}]")
    origin = coordinate_from_json(geometry["origin_xyz_angstrom"], "origin_xyz_angstrom")
    spacing = as_float(geometry["spacing_angstrom"], "spacing_angstrom")
    grid_dim = as_int(geometry["grid_dim"], "grid_dim")
    heavy_indices = [int(atom.GetIdx()) for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) > 1]
    atom_idx = heavy_indices[0]
    pos = mol.GetConformer().GetAtomPosition(atom_idx)
    coord = (float(pos.x), float(pos.y), float(pos.z))
    ix = math.trunc((coord[0] - origin[0]) / spacing)
    iy = math.trunc((coord[1] - origin[1]) / spacing)
    iz = math.trunc((coord[2] - origin[2]) / spacing)
    in_bounds = 0 <= ix < grid_dim and 0 <= iy < grid_dim and 0 <= iz < grid_dim
    voxel_idx = iz * grid_dim * grid_dim + iy * grid_dim + ix
    variance_class = "out_of_bounds"
    if in_bounds:
        rows = (
            pl.scan_parquet(signal_grid)
            .filter((pl.col("condition_id") == condition_id) & (pl.col("voxel_idx") == voxel_idx))
            .select("variance_class")
            .collect()
            .get_column("variance_class")
            .to_list()
        )
        if rows:
            variance_class = str(rows[0])
    return {
        "condition_id": condition_id,
        "atom_idx": atom_idx,
        "xyz_angstrom": [coord[0], coord[1], coord[2]],
        "grid_indices": [ix, iy, iz],
        "grid_dim": grid_dim,
        "voxel_idx": voxel_idx,
        "in_bounds": in_bounds,
        "variance_class": variance_class,
    }


def load_existing_conformer_set(parent_sdf: Path, grid_mapping: Path, signal_grid: Path) -> ConformerSet:
    supplier = Chem.SDMolSupplier(str(parent_sdf), removeHs=False)
    mols = [mol for mol in supplier if mol is not None]
    if len(mols) < K_CONFORMERS:
        raise ValueError(f"existing parent SDF has {len(mols)} conformers, expected at least {K_CONFORMERS}")
    energies = [
        (idx, float(mol.GetProp("mmff_energy_kcal")) if mol.HasProp("mmff_energy_kcal") else 0.0)
        for idx, mol in enumerate(mols[:K_CONFORMERS])
    ]
    method = (
        mols[0].GetProp("alignment_method")
        if mols[0].HasProp("alignment_method")
        else "sdf_receptor_frame_prealigned"
    )
    validation = validate_first_heavy_atom_from_mol(mols[0], grid_mapping, signal_grid)
    if validation["in_bounds"] is not True or validation["variance_class"] == "void":
        raise ValueError(f"existing aligned SDF failed validation: {validation}")
    energy_values = [energy for _idx, energy in energies]
    return ConformerSet(
        mol_h=mols[0],
        top_conformers=energies,
        sdf_path=parent_sdf,
        alignment_method=str(method),
        energy_range=(min(energy_values), max(energy_values)),
        energy_spread=max(energy_values) - min(energy_values),
        first_atom_validation=validation,
    )


def load_existing_fragments(path: Path) -> list[FragmentInfo]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not decode to a JSON object")
    fragments: list[FragmentInfo] = []
    for raw_fragment in json_list(payload["fragments"], "fragments"):
        fragment = json_object(raw_fragment, "fragment")
        sdf_path = REPO_ROOT / str(fragment["sdf_path"])
        if not sdf_path.exists():
            raise FileNotFoundError(sdf_path)
        fragments.append(
            FragmentInfo(
                fragment_id=str(fragment["fragment_id"]),
                brics_smiles=str(fragment["brics_smiles"]),
                parent_atom_indices=tuple(
                    as_int(value, "parent_atom_index")
                    for value in json_list(fragment["parent_atom_indices"], "parent_atom_indices")
                ),
                n_heavy_atoms=as_int(fragment["n_heavy_atoms"], "n_heavy_atoms"),
                sdf_path=sdf_path,
            )
        )
    return fragments


def collect_fragment_scores(fragments: list[FragmentInfo]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for fragment in fragments:
        edges = pl.read_parquet(LAYER2_DIR / fragment.fragment_id / "per_edge_interference.parquet")
        for row in edges.to_dicts():
            rows.append(
                {
                    "fragment_id": fragment.fragment_id,
                    "brics_smiles": fragment.brics_smiles,
                    "edge_id": str(row["edge_id"]),
                    "frag_E_pi_clash": as_float(row["E_pi_clash"], "E_pi_clash"),
                    "frag_E_pi_complement": as_float(row["E_pi_complement"], "E_pi_complement"),
                    "frag_U_pose": as_float(row["U_pose"], "U_pose"),
                }
            )
    return pl.DataFrame(rows)


def validate_frozen_beta_projection_scale(fragments: list[FragmentInfo], whole_projection: pl.DataFrame) -> None:
    whole_score = abs(as_float(whole_projection["total_projected_durability"][0], "whole_projected_score"))
    fragment_score_sum = 0.0
    for fragment in fragments:
        projection = pl.read_parquet(LAYER2_DIR / fragment.fragment_id / "analog_durability_projection.parquet")
        fragment_score_sum += abs(as_float(projection["total_projected_durability"][0], "fragment_projected_score"))
    if whole_score <= 0.0:
        return
    ratio = fragment_score_sum / whole_score
    if ratio > 10.0 or ratio < 0.1:
        raise ValueError(
            f"fragment projected score sum and whole score differ by >10x: ratio={ratio:.4f}"
        )


def generate_fragment_rationale(row: dict[str, object]) -> str:
    edge = str(row["edge_id"])
    dom_frag = str(row["dominant_fragment"])
    dom_clash = as_float(row["dominant_fragment_clash"], "dominant_fragment_clash")
    wm_clash = as_float(row["whole_molecule_clash"], "whole_molecule_clash")
    coupling = as_float(row["inter_fragment_coupling"], "inter_fragment_coupling")
    fraction = as_float(row["dominant_fraction"], "dominant_fraction")
    if wm_clash < 0.1:
        return (
            f"Aleniglipron has low projected steric interference at {edge} "
            f"(Pi_clash = {wm_clash:.3f}). No fragment contributes a high projected clash signal; "
            "this chemistry claim requires assay falsification."
        )
    parts = [
        f"At {edge}, aleniglipron shows Pi_clash = {wm_clash:.2f}. "
        f"{dom_frag} is the dominant contributor ({dom_clash:.2f}, {fraction * 100.0:.0f}% of total)."
    ]
    if abs(coupling) > 0.1:
        mode = "coupled steric load" if coupling > 0 else "partial projected steric relief"
        parts.append(
            f"Inter-fragment coupling contributes {coupling:.2f} projected clash, indicating {mode} when fragments are connected."
        )
    return " ".join(parts)


def attribution(whole_edges: pl.DataFrame, fragment_scores: pl.DataFrame, fragments: list[FragmentInfo]) -> pl.DataFrame:
    fragment_smiles = {fragment.fragment_id: fragment.brics_smiles for fragment in fragments}
    rows: list[dict[str, object]] = []
    for whole_row in whole_edges.sort("edge_id").to_dicts():
        edge_id = str(whole_row["edge_id"])
        wm_clash = as_float(whole_row["E_pi_clash"], "whole E_pi_clash")
        wm_complement = as_float(whole_row["E_pi_complement"], "whole E_pi_complement")
        frag_rows = fragment_scores.filter(pl.col("edge_id") == edge_id)
        sum_frag_clash = as_float(frag_rows["frag_E_pi_clash"].sum(), "sum_frag_clash")
        sum_frag_complement = as_float(frag_rows["frag_E_pi_complement"].sum(), "sum_frag_complement")
        dominant = frag_rows.sort("frag_E_pi_clash", descending=True).head(1).to_dicts()[0]
        dominant_fragment = str(dominant["fragment_id"])
        dominant_clash = as_float(dominant["frag_E_pi_clash"], "dominant_clash")
        row = {
            "edge_id": edge_id,
            "whole_molecule_clash": wm_clash,
            "whole_molecule_complement": wm_complement,
            "sum_fragment_clash": sum_frag_clash,
            "sum_fragment_complement": sum_frag_complement,
            "inter_fragment_coupling": wm_clash - sum_frag_clash,
            "inter_fragment_complement_coupling": wm_complement - sum_frag_complement,
            "dominant_fragment": dominant_fragment,
            "dominant_fragment_smiles": fragment_smiles[dominant_fragment],
            "dominant_fragment_clash": dominant_clash,
            "dominant_fraction": dominant_clash / wm_clash if wm_clash > 0.0 else 0.0,
        }
        row["rationale"] = generate_fragment_rationale(row)
        rows.append(row)
    return pl.DataFrame(rows)


def confidence_class(projection: pl.DataFrame) -> str:
    score = as_float(projection["total_projected_durability"][0], "score")
    uncertainty = as_float(projection["total_projected_uncertainty"][0], "uncertainty")
    if score <= 0.0:
        return "invalid_nonpositive_score"
    ratio = uncertainty / score
    if ratio < 0.1:
        return "high"
    if ratio < 0.25:
        return "moderate"
    return "low"


def render_summary(
    whole_edges: pl.DataFrame,
    whole_projection: pl.DataFrame,
    attribution_df: pl.DataFrame,
    fragments: list[FragmentInfo],
) -> None:
    env = Environment(undefined=StrictUndefined, autoescape=False)
    template = env.from_string(SUMMARY_TEMPLATE)
    worst = attribution_df.sort("dominant_fragment_clash", descending=True).head(1).to_dicts()[0]
    total_clash = as_float(whole_edges["E_pi_clash"].sum(), "total_clash")
    total_complement = as_float(whole_edges["E_pi_complement"].sum(), "total_complement")
    projected_score = as_float(whole_projection["total_projected_durability"][0], "projected_score")
    uncertainty = as_float(whole_projection["total_projected_uncertainty"][0], "uncertainty")
    total_coupling = as_float(attribution_df["inter_fragment_coupling"].sum(), "total_coupling")
    worst_fragment = str(worst["dominant_fragment"])
    recommendation = (
        f"Prioritize modifications to {worst_fragment} if reducing receptor-side steric clash is the objective; "
        "protect fragments with low dominant clash unless potency SAR requires changes."
        if total_clash >= 0.1
        else "The parent shows negligible receptor-side steric clash in this field map; preserve the current scaffold and prioritize potency/selectivity SAR."
    )
    rendered = template.render(
        total_clash=total_clash,
        total_complement=total_complement,
        projected_score=projected_score,
        uncertainty=uncertainty,
        confidence_class=confidence_class(whole_projection),
        n_fragments=len(fragments),
        worst_fragment_id=worst_fragment,
        worst_fragment_smiles=str(worst["dominant_fragment_smiles"]),
        worst_fraction=as_float(worst["dominant_fraction"], "worst_fraction"),
        total_coupling=total_coupling,
        attribution_rows=attribution_df.to_dicts(),
        recommendation_text=recommendation,
    )
    SUMMARY_PATH.write_text(rendered, encoding="utf-8")


def lineage_inputs(
    fragments: list[FragmentInfo],
    args: argparse.Namespace,
    parent_sdf: Path,
    extra_parquets: list[Path],
) -> dict[str, Path]:
    inputs: dict[str, Path] = {
        "pubchem_api_response": PUBCHEM_RESPONSE_PATH,
        "analog_registry": ANALOG_REGISTRY_PATH,
        "whole_molecule_sdf": parent_sdf,
        "grid_coordinate_mapping": args.grid_mapping,
        "binding_site_reference": args.binding_site,
        "interface_steric_environment": args.steric_env,
        "receptor_durability_risk_map": args.risk_map,
        "signal_grid_variance_channel": args.signal_grid,
        "fragment_registry": FRAGMENT_REGISTRY_PATH,
    }
    for fragment in fragments:
        inputs[f"{fragment.fragment_id}_sdf"] = fragment.sdf_path
    for idx, path in enumerate(extra_parquets):
        inputs[f"scoring_parquet_{idx}"] = path
    return inputs


def seal_lineage(
    parquet_path: Path,
    inputs: dict[str, Path],
    module: str,
    output_rows: int,
) -> None:
    append_propagation_entry(
        parquet_path.with_suffix(".propagation.jsonl"),
        build_entry(
            module=module,
            operation="lineage_seal",
            inputs=dict[str, Path | str](inputs),
            parameters={
                "lineage_requirement": "PubChem -> RDKit -> Conformers -> Alignment -> Voxel Mapping -> Interference Scoring -> Fragment Attribution",
                "input_sha256": {name: sha256_path(path) for name, path in sorted(inputs.items())},
            },
            output_value={"output_path": parquet_path, "rows": output_rows},
            output_uncertainty=None,
            gate_status={"lineage_sealed": True, "sha256_complete": True},
            repo_root=REPO_ROOT,
        ),
        repo_root=REPO_ROOT,
    )


def write_attribution(attribution_df: pl.DataFrame, source_parquets: list[Path], lineage: dict[str, Path]) -> None:
    write_provenance_parquet(
        attribution_df,
        ATTRIBUTION_PATH,
        producer_script=Path(__file__),
        source_parquets=source_parquets,
        schema_version="aleniglipron_fragment_interference_attribution.v1",
        pipeline_stage="track0_two_layer_interference_attribution",
        partition_keys=["edge_id"],
        ledger_parameters={
            "pubchem_cid": ALENIGLIPRON_CID,
            "non_parquet_input_sha256": {
                name: sha256_path(path)
                for name, path in sorted(lineage.items())
                if path.suffix != ".parquet"
            },
        },
        ledger_output_value={"rows": attribution_df.height, "output_path": ATTRIBUTION_PATH},
        repo_root=REPO_ROOT,
    )


def gate_report_payload(
    clean: CleanMolecule,
    conformers: ConformerSet,
    fragments: list[FragmentInfo],
    whole_edges: pl.DataFrame,
    whole_projection: pl.DataFrame,
    attribution_df: pl.DataFrame,
) -> JsonObject:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "gate_1": {
            "desalted_smiles": clean.clean_smiles,
            "mw": clean.mw,
            "n_heavy_atoms": clean.n_heavy_atoms,
            "rotatable_bonds": clean.rotatable_bonds,
            "status": "PASS",
        },
        "gate_2": {
            "n_conformers": len(conformers.top_conformers),
            "energy_min_kcal_mol": conformers.energy_range[0],
            "energy_max_kcal_mol": conformers.energy_range[1],
            "energy_spread_kcal_mol": conformers.energy_spread,
            "alignment_method": conformers.alignment_method,
            "alignment_validation": conformers.first_atom_validation,
            "status": "PASS",
        },
        "gate_3": {
            "edges_scored": whole_edges.height,
            "projected_durability_score": as_float(whole_projection["total_projected_durability"][0], "score"),
            "projected_durability_uncertainty": as_float(whole_projection["total_projected_uncertainty"][0], "uncertainty"),
            "status": "PASS",
        },
        "gate_4": {
            "n_fragments": len(fragments),
            "parent_heavy_atoms": clean.n_heavy_atoms,
            "sum_fragment_heavy_atoms": sum(fragment.n_heavy_atoms for fragment in fragments),
            "status": "PASS",
        },
        "gate_5": {
            "fragment_attribution_rows": attribution_df.height,
            "total_inter_fragment_coupling": as_float(attribution_df["inter_fragment_coupling"].sum(), "total_coupling"),
            "status": "PASS",
        },
        "outputs": {
            "analog_registry": ANALOG_REGISTRY_PATH.relative_to(REPO_ROOT).as_posix(),
            "whole_sdf": conformers.sdf_path.relative_to(REPO_ROOT).as_posix(),
            "fragment_registry": FRAGMENT_REGISTRY_PATH.relative_to(REPO_ROOT).as_posix(),
            "attribution_parquet": ATTRIBUTION_PATH.relative_to(REPO_ROOT).as_posix(),
            "summary_markdown": SUMMARY_PATH.relative_to(REPO_ROOT).as_posix(),
        },
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=str(args.log_level).upper(), format="%(levelname)s %(message)s")
    CONFORMER_DIR.mkdir(parents=True, exist_ok=True)
    LAYER1_DIR.mkdir(parents=True, exist_ok=True)
    LAYER2_DIR.mkdir(parents=True, exist_ok=True)

    raw_smiles, _payload = retrieve_pubchem_smiles()
    clean = clean_and_validate_molecule(raw_smiles)
    write_analog_registry(clean)
    logging.info("Gate 1 PASS: MW %.2f, heavy atoms %s", clean.mw, clean.n_heavy_atoms)

    parent_sdf = CONFORMER_DIR / "ALENI-PARENT_whole_molecule_aligned.sdf"
    reuse_existing = parent_sdf.exists() and FRAGMENT_REGISTRY_PATH.exists() and not bool(args.regenerate_conformers)
    if reuse_existing:
        conformers = load_existing_conformer_set(parent_sdf, args.grid_mapping, args.signal_grid)
    else:
        conformers = generate_parent_conformers(clean, args.binding_site, args.grid_mapping, args.signal_grid)
    logging.info(
        "Gate 2 PASS: %s aligned conformers, energy spread %.2f kcal/mol%s",
        len(conformers.top_conformers),
        conformers.energy_spread,
        " (reused existing SDF)" if reuse_existing else "",
    )

    run_interference_tool(
        conformers.sdf_path,
        LAYER1_DIR,
        args.grid_mapping,
        args.binding_site,
        args.steric_env,
        args.risk_map,
        str(args.beta_f),
        str(args.beta_s),
        args.signal_grid,
    )
    whole_edges, whole_projection = validate_layer_output(LAYER1_DIR)
    frozen_beta_f = as_float(whole_projection["beta_f"][0], "frozen_beta_f")
    frozen_beta_s = as_float(whole_projection["beta_s"][0], "frozen_beta_s")
    logging.info("Gate 3 PASS: whole molecule scored %s edges", whole_edges.height)

    fragments = load_existing_fragments(FRAGMENT_REGISTRY_PATH) if reuse_existing else slice_fragments(clean, conformers.sdf_path)
    logging.info("Gate 4 PASS: sliced %s BRICS fragments", len(fragments))

    for fragment in fragments:
        run_interference_tool(
            fragment.sdf_path,
            LAYER2_DIR / fragment.fragment_id,
            args.grid_mapping,
            args.binding_site,
            args.steric_env,
            args.risk_map,
            frozen_beta_f,
            frozen_beta_s,
            args.signal_grid,
        )
        validate_layer_output(LAYER2_DIR / fragment.fragment_id)
        fragment_projection = pl.read_parquet(LAYER2_DIR / fragment.fragment_id / "analog_durability_projection.parquet")
        if as_float(fragment_projection["beta_f"][0], "fragment_beta_f") != frozen_beta_f:
            raise ValueError(f"{fragment.fragment_id} beta_f did not match frozen whole-molecule beta")
        if as_float(fragment_projection["beta_s"][0], "fragment_beta_s") != frozen_beta_s:
            raise ValueError(f"{fragment.fragment_id} beta_s did not match frozen whole-molecule beta")
    validate_frozen_beta_projection_scale(fragments, whole_projection)
    fragment_scores = collect_fragment_scores(fragments)
    attribution_df = attribution(whole_edges, fragment_scores, fragments)
    logging.info("Gate 5 PASS: computed %s attribution rows", attribution_df.height)

    scoring_parquets = [
        LAYER1_DIR / "per_edge_interference.parquet",
        LAYER1_DIR / "analog_durability_projection.parquet",
        *[
            path
            for fragment in fragments
            for path in [
                LAYER2_DIR / fragment.fragment_id / "per_edge_interference.parquet",
                LAYER2_DIR / fragment.fragment_id / "analog_durability_projection.parquet",
            ]
        ],
    ]
    source_parquets = [args.steric_env, args.risk_map, args.signal_grid, *scoring_parquets]
    lineage = lineage_inputs(fragments, args, conformers.sdf_path, scoring_parquets)
    write_attribution(attribution_df, source_parquets, lineage)
    render_summary(whole_edges, whole_projection, attribution_df, fragments)
    write_json(GATE_REPORT_PATH, gate_report_payload(clean, conformers, fragments, whole_edges, whole_projection, attribution_df))
    for parquet_path in [*scoring_parquets, ATTRIBUTION_PATH]:
        seal_lineage(
            parquet_path,
            lineage,
            "track0_two_layer_aleniglipron_interference",
            pl.read_parquet(parquet_path).height,
        )
    logging.info("Gate 6/7 PASS: wrote summary and sealed lineage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
