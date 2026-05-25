#!/usr/bin/env python3
"""Ingest Track A calibration anchors and generate one low-energy 3D conformer per entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, TypeAlias, cast

import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Descriptors


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import sha256_path, write_provenance_parquet
from prism_dstw.ontology import EpistemicClass, epistemic_metadata


CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK0_DIR = CAMPAIGN_DIR / "track_0_manual_emulation"
TRACK_A_DIR = CAMPAIGN_DIR / "track_a_generative"
DEFAULT_INPUT = Path("/home/diddy/prism4d_analysis/library/fragments.smi")
DEFAULT_OUTPUT = TRACK_A_DIR / "calibration_anchors_3d.parquet"
ANALOG_REGISTRY = TRACK0_DIR / "analog_registry.json"
BRICS_REGISTRY = TRACK0_DIR / "aleniglipron_brics_fragment_registry.json"
CONFORMER_DIR = TRACK0_DIR / "conformers"
FRAGMENT_ATTRIBUTION = CAMPAIGN_DIR / "interference/fragment_interference_attribution.parquet"
DEFAULT_TARGET_SIZE = 512
MIN_CALIBRATION_SIZE = 500
MAX_CALIBRATION_SIZE = 1000
DEFAULT_CHUNKSIZE = 16
DEFAULT_CHARGE_TIMEOUT_SECONDS = 90
DEFAULT_AM1_MAX_HEAVY_ATOMS = 18
MAX_HEAVY_ATOMS = 45
EPSILON = 1.0e-9
ANTECHAMBER = Path(
    os.environ.get("PRISM_ANTECHAMBER", "/home/diddy/miniconda3/envs/prism_dock/bin/antechamber")
)

CHEM = cast(Any, Chem)
ALL_CHEM = cast(Any, AllChem)
BRICS_MOD = cast(Any, BRICS)
DESCRIPTORS = cast(Any, Descriptors)

JsonObject: TypeAlias = dict[str, object]
Coordinate: TypeAlias = tuple[float, float, float]


@dataclass(frozen=True)
class CalibrationSeed:
    source_id: str
    smiles: str
    source_kind: str


@dataclass(frozen=True)
class AnchorTask:
    anchor_idx: int
    source_id: str
    smiles: str
    source_kind: str
    require_am1_bcc: bool
    charge_timeout_seconds: int
    am1_max_heavy_atoms: int


@dataclass(frozen=True)
class TaskBundle:
    tasks: list[AnchorTask]
    input_seed_count: int
    augmented: bool
    source_kind_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--min-calibration-size", type=int, default=MIN_CALIBRATION_SIZE)
    parser.add_argument("--processes", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--charge-timeout-seconds", type=int, default=DEFAULT_CHARGE_TIMEOUT_SECONDS)
    parser.add_argument("--am1-max-heavy-atoms", type=int, default=DEFAULT_AM1_MAX_HEAVY_ATOMS)
    parser.add_argument("--require-am1-bcc", action="store_true")
    return parser.parse_args()


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def emit_err(message: str) -> None:
    sys.stderr.write(message + "\n")


def file_sha256(path: Path) -> str:
    return sha256_path(path) if path.resolve().is_relative_to(REPO_ROOT.resolve()) else sha256_external_path(path)


def sha256_external_path(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def antechamber_available() -> bool:
    return ANTECHAMBER.exists() and os.access(ANTECHAMBER, os.X_OK)


def json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def canonicalize_smiles(smiles: str) -> str | None:
    mol = CHEM.MolFromSmiles(smiles)
    if mol is None:
        return None
    if any(int(atom.GetAtomicNum()) == 0 for atom in mol.GetAtoms()):
        stripped = strip_dummy_atoms(mol)
        if stripped is None:
            return None
        mol = stripped
    try:
        CHEM.SanitizeMol(mol)
    except (RuntimeError, ValueError):
        return None
    fragments = CHEM.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if fragments:
        mol = max(fragments, key=lambda fragment: int(fragment.GetNumHeavyAtoms()))
    if int(mol.GetNumHeavyAtoms()) < 2 or int(mol.GetNumHeavyAtoms()) > MAX_HEAVY_ATOMS:
        return None
    if abs(int(CHEM.GetFormalCharge(mol))) > 2:
        return None
    return str(CHEM.MolToSmiles(mol, canonical=True))


def add_seed(candidates: dict[str, CalibrationSeed], source_id: str, smiles: str, source_kind: str) -> None:
    canonical = canonicalize_smiles(smiles)
    if canonical is None or canonical in candidates:
        return
    candidates[canonical] = CalibrationSeed(source_id=source_id, smiles=canonical, source_kind=source_kind)


def read_line_smiles(path: Path) -> list[CalibrationSeed]:
    seeds: list[CalibrationSeed] = []
    if not path.exists():
        return seeds
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            source_id = parts[1] if len(parts) > 1 else f"fragment_seed_{line_idx:04d}"
            canonical = canonicalize_smiles(parts[0])
            if canonical is None:
                continue
            seeds.append(CalibrationSeed(source_id=source_id, smiles=canonical, source_kind="local_fragment_seed"))
    return seeds


def registry_smiles() -> list[CalibrationSeed]:
    if not ANALOG_REGISTRY.exists():
        return []
    payload = json_object(json.loads(ANALOG_REGISTRY.read_text(encoding="utf-8")), "analog_registry")
    seeds: list[CalibrationSeed] = []
    for key in ("desalted_smiles", "raw_smiles"):
        value = payload.get(key)
        if isinstance(value, str):
            seeds.append(CalibrationSeed(source_id=f"ALENI-PARENT:{key}", smiles=value, source_kind="local_glp1r_agonist"))
    for raw_analog in json_list(payload.get("analogs", []), "analogs"):
        analog = json_object(raw_analog, "analog")
        analog_id = str(analog.get("analog_id", f"analog_{len(seeds) + 1:04d}"))
        for key in ("canonical_smiles", "raw_smiles", "desalted_smiles"):
            value = analog.get(key)
            if isinstance(value, str):
                seeds.append(CalibrationSeed(source_id=f"{analog_id}:{key}", smiles=value, source_kind="local_glp1r_agonist"))
    return seeds


def sdf_smiles(path: Path, source_kind: str) -> list[CalibrationSeed]:
    if not path.exists():
        return []
    supplier = CHEM.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    seeds: list[CalibrationSeed] = []
    for mol_idx, mol in enumerate(supplier, start=1):
        if mol is None:
            continue
        source_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{path.stem}:{mol_idx:04d}"
        smiles = str(CHEM.MolToSmiles(CHEM.RemoveHs(mol), canonical=True))
        seeds.append(CalibrationSeed(source_id=source_id, smiles=smiles, source_kind=source_kind))
    return seeds


def brics_registry_smiles() -> list[CalibrationSeed]:
    if not BRICS_REGISTRY.exists():
        return []
    payload = json_object(json.loads(BRICS_REGISTRY.read_text(encoding="utf-8")), "brics_registry")
    seeds: list[CalibrationSeed] = []
    for raw_fragment in json_list(payload.get("fragments", []), "fragments"):
        fragment = json_object(raw_fragment, "fragment")
        brics_smiles = fragment.get("brics_smiles")
        fragment_id = str(fragment.get("fragment_id", f"fragment_{len(seeds) + 1:04d}"))
        if isinstance(brics_smiles, str):
            seeds.append(CalibrationSeed(source_id=f"{fragment_id}:brics_registry", smiles=brics_smiles, source_kind="local_brics_fragment"))
        sdf_path = fragment.get("sdf_path")
        if isinstance(sdf_path, str):
            seeds.extend(sdf_smiles(REPO_ROOT / sdf_path, "local_sliced_fragment_sdf"))
    return seeds


def attribution_fragment_smiles() -> list[CalibrationSeed]:
    if not FRAGMENT_ATTRIBUTION.exists():
        return []
    columns = pl.scan_parquet(FRAGMENT_ATTRIBUTION).collect_schema().names()
    if "dominant_fragment_smiles" not in columns:
        return []
    frame = (
        pl.scan_parquet(FRAGMENT_ATTRIBUTION)
        .select(pl.col("dominant_fragment_smiles").drop_nulls().unique())
        .collect()
    )
    seeds: list[CalibrationSeed] = []
    for idx, row in enumerate(frame.iter_rows(named=True), start=1):
        smiles = row.get("dominant_fragment_smiles")
        if isinstance(smiles, str):
            seeds.append(
                CalibrationSeed(
                    source_id=f"fragment_attribution:{idx:04d}",
                    smiles=smiles,
                    source_kind="projected_fragment_attribution_seed",
                )
            )
    return seeds


def local_seed_molecules(input_path: Path) -> tuple[list[CalibrationSeed], int]:
    input_seeds = read_line_smiles(input_path)
    seeds: list[CalibrationSeed] = []
    seeds.extend(input_seeds)
    seeds.extend(registry_smiles())
    seeds.extend(sdf_smiles(CONFORMER_DIR / "ALENI-PARENT_whole_molecule_aligned.sdf", "local_glp1r_agonist_sdf"))
    seeds.extend(brics_registry_smiles())
    seeds.extend(attribution_fragment_smiles())
    return seeds, len(input_seeds)


def strip_dummy_atoms(mol: Any) -> Any | None:
    editable = CHEM.RWMol(mol)
    dummy_indices = [int(atom.GetIdx()) for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) == 0]
    for atom_idx in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(atom_idx)
    stripped = editable.GetMol()
    try:
        CHEM.SanitizeMol(stripped)
    except (RuntimeError, ValueError):
        return None
    return stripped


def brics_fragment_pool(seeds: list[CalibrationSeed]) -> list[Any]:
    fragments: dict[str, Any] = {}
    for seed in seeds:
        mol = CHEM.MolFromSmiles(seed.smiles)
        if mol is None:
            continue
        try:
            raw_fragments = BRICS_MOD.BRICSDecompose(mol)
        except (RuntimeError, ValueError):
            continue
        for fragment_smiles in sorted(raw_fragments):
            fragment_mol = CHEM.MolFromSmiles(fragment_smiles)
            if fragment_mol is None:
                continue
            fragment_key = str(CHEM.MolToSmiles(fragment_mol, canonical=True))
            fragments.setdefault(fragment_key, fragment_mol)
    return list(fragments.values())


def add_stripped_brics_fragments(candidates: dict[str, CalibrationSeed], fragments: list[Any]) -> None:
    for idx, fragment_mol in enumerate(fragments, start=1):
        stripped = strip_dummy_atoms(fragment_mol)
        if stripped is None:
            continue
        smiles = str(CHEM.MolToSmiles(stripped, canonical=True))
        add_seed(candidates, f"BRICS-FRAGMENT-{idx:04d}", smiles, "brics_stripped_anchor")


def augment_with_brics(candidates: dict[str, CalibrationSeed], seeds: list[CalibrationSeed], target_size: int) -> None:
    fragments = brics_fragment_pool(seeds)
    add_stripped_brics_fragments(candidates, fragments)
    if len(candidates) >= target_size or not fragments:
        return
    for depth in (2, 3, 4):
        builder = BRICS_MOD.BRICSBuild(fragments, maxDepth=depth, scrambleReagents=False)
        for built_mol in builder:
            smiles = str(CHEM.MolToSmiles(built_mol, canonical=True))
            add_seed(candidates, f"BRICS-RECOMB-{len(candidates) + 1:06d}", smiles, "brics_recombinant_anchor")
            if len(candidates) >= target_size:
                return


def source_kind_counts(seeds: Iterable[CalibrationSeed]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seed in seeds:
        counts[seed.source_kind] = counts.get(seed.source_kind, 0) + 1
    return counts


def build_tasks(
    *,
    input_path: Path,
    target_size: int,
    min_calibration_size: int,
    require_am1_bcc: bool,
    charge_timeout_seconds: int,
    am1_max_heavy_atoms: int,
) -> TaskBundle:
    seeds, input_seed_count = local_seed_molecules(input_path)
    candidates: dict[str, CalibrationSeed] = {}
    for seed in seeds:
        add_seed(candidates, seed.source_id, seed.smiles, seed.source_kind)
    augmented = False
    if len(candidates) < min_calibration_size:
        augment_with_brics(candidates, seeds, target_size)
        augmented = True
    if len(candidates) < min_calibration_size:
        raise ValueError(
            f"calibration library yielded {len(candidates)} anchors, below requested minimum {min_calibration_size}"
        )
    selected = list(candidates.values())[:target_size]
    tasks = [
        AnchorTask(
            anchor_idx=idx,
            source_id=seed.source_id,
            smiles=seed.smiles,
            source_kind=seed.source_kind,
            require_am1_bcc=require_am1_bcc,
            charge_timeout_seconds=charge_timeout_seconds,
            am1_max_heavy_atoms=am1_max_heavy_atoms,
        )
        for idx, seed in enumerate(selected)
    ]
    return TaskBundle(
        tasks=tasks,
        input_seed_count=input_seed_count,
        augmented=augmented,
        source_kind_counts=source_kind_counts(selected),
    )


def atom_coordinates(mol: Any, atom_idx: int) -> Coordinate:
    pos = mol.GetConformer().GetAtomPosition(atom_idx)
    return (float(pos.x), float(pos.y), float(pos.z))


def coordinate_payload(mol: Any, charges: list[float]) -> str:
    atoms: list[dict[str, object]] = []
    for atom in mol.GetAtoms():
        atomic_num = int(atom.GetAtomicNum())
        if atomic_num <= 1:
            continue
        atom_idx = int(atom.GetIdx())
        coord = atom_coordinates(mol, atom_idx)
        atoms.append(
            {
                "atom_idx": atom_idx,
                "atomic_num": atomic_num,
                "symbol": str(atom.GetSymbol()),
                "x": coord[0],
                "y": coord[1],
                "z": coord[2],
                "partial_charge": charges[atom_idx],
            }
        )
    return json.dumps(atoms, sort_keys=True, separators=(",", ":"))


def bounding_box(mol: Any) -> tuple[float, float, float]:
    coords = [atom_coordinates(mol, int(atom.GetIdx())) for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) > 1]
    if not coords:
        return (0.0, 0.0, 0.0)
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    zs = [coord[2] for coord in coords]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def approximate_volume(mol: Any) -> float:
    radii = {
        1: 1.20,
        6: 1.70,
        7: 1.55,
        8: 1.52,
        9: 1.47,
        15: 1.80,
        16: 1.80,
        17: 1.75,
        35: 1.85,
        53: 1.98,
    }
    total = 0.0
    for atom in mol.GetAtoms():
        radius = radii.get(int(atom.GetAtomicNum()), 1.70)
        total += (4.0 / 3.0) * math.pi * radius * radius * radius
    return total


def compute_volume(mol: Any) -> float:
    try:
        return float(ALL_CHEM.ComputeMolVolume(mol))
    except (RuntimeError, ValueError):
        return approximate_volume(mol)


def parse_mol2_charges(path: Path, atom_count: int) -> list[float]:
    charges: list[float] = []
    in_atoms = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            break
        if not in_atoms or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        charges.append(float(parts[-1]))
    if len(charges) != atom_count:
        raise ValueError(f"mol2 charge count {len(charges)} did not match atom count {atom_count}")
    return charges


def compute_am1_bcc_charges(mol: Any, formal_charge: int, timeout_seconds: int) -> list[float]:
    if not antechamber_available():
        raise RuntimeError("AmberTools antechamber is unavailable")
    with tempfile.TemporaryDirectory(prefix="prism_calibration_anchor_") as tmpdir:
        workdir = Path(tmpdir)
        sdf_path = workdir / "anchor.sdf"
        mol2_path = workdir / "anchor_charged.mol2"
        CHEM.MolToMolFile(mol, str(sdf_path))
        env = dict(os.environ)
        env["PATH"] = f"{ANTECHAMBER.parent}:{env.get('PATH', '')}"
        command = [
            str(ANTECHAMBER),
            "-i",
            str(sdf_path),
            "-fi",
            "sdf",
            "-o",
            str(mol2_path),
            "-fo",
            "mol2",
            "-c",
            "bcc",
            "-nc",
            str(formal_charge),
            "-s",
            "0",
        ]
        result = subprocess.run(
            command,
            cwd=workdir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode != 0 or not mol2_path.exists():
            diagnostic = result.stderr.strip()[-240:] or result.stdout.strip()[-240:]
            raise RuntimeError(f"antechamber failed with code {result.returncode}: {diagnostic}")
        return parse_mol2_charges(mol2_path, int(mol.GetNumAtoms()))


def compute_gasteiger_charges(mol: Any) -> list[float]:
    ALL_CHEM.ComputeGasteigerCharges(mol)
    charges: list[float] = []
    for atom in mol.GetAtoms():
        raw_charge = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0.0"
        charge = float(raw_charge) if raw_charge not in {"nan", "-nan", "inf", "-inf"} else 0.0
        charges.append(charge)
    return charges


def compute_charges(
    mol: Any,
    *,
    formal_charge: int,
    require_am1_bcc: bool,
    timeout_seconds: int,
    am1_max_heavy_atoms: int,
) -> tuple[list[float], str]:
    heavy_atoms = int(sum(1 for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) > 1))
    if heavy_atoms > am1_max_heavy_atoms and not require_am1_bcc:
        return compute_gasteiger_charges(mol), "rdkit_gasteiger_fallback_size_gate"
    try:
        return compute_am1_bcc_charges(mol, formal_charge, timeout_seconds), "am1-bcc"
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError):
        if require_am1_bcc:
            raise
    return compute_gasteiger_charges(mol), "rdkit_gasteiger_fallback"


def optimize_conformer(mol_h: Any, seed: int) -> tuple[Any, float, str]:
    params = ALL_CHEM.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    embed_status = int(ALL_CHEM.EmbedMolecule(mol_h, params))
    if embed_status != 0:
        raise RuntimeError(f"ETKDGv3 embedding failed with code {embed_status}")

    energy = 0.0
    force_field = "MMFF94s"
    if bool(ALL_CHEM.MMFFHasAllMoleculeParams(mol_h)):
        ALL_CHEM.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94s", maxIters=300)
        props = ALL_CHEM.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        ff = ALL_CHEM.MMFFGetMoleculeForceField(mol_h, props)
        energy = float(ff.CalcEnergy()) if ff is not None else 0.0
    else:
        force_field = "UFF"
        ALL_CHEM.UFFOptimizeMolecule(mol_h, maxIters=300)
        ff = ALL_CHEM.UFFGetMoleculeForceField(mol_h)
        energy = float(ff.CalcEnergy()) if ff is not None else 0.0
    return mol_h, energy, force_field


def failure_row(task: AnchorTask, status: str) -> JsonObject:
    return {
        "anchor_idx": task.anchor_idx,
        "anchor_id": f"CAL-ANCHOR-{task.anchor_idx + 1:06d}",
        "source_anchor_id": task.source_id,
        "source_kind": task.source_kind,
        "input_smiles": task.smiles,
        "canonical_smiles": "",
        "n_heavy_atoms": 0,
        "molecular_weight": 0.0,
        "formal_charge": 0,
        "steric_volume_A3": 0.0,
        "bbox_x_A": 0.0,
        "bbox_y_A": 0.0,
        "bbox_z_A": 0.0,
        "mmff_energy_kcal_mol": 0.0,
        "force_field": "",
        "partial_charge_method": "unassigned",
        "partial_charges_json": "[]",
        "conformer_atoms_json": "[]",
        "generation_status": status,
        "Epistemic_Class": EpistemicClass.HYPOTHESIZED.value,
        "epistemic_class": EpistemicClass.HYPOTHESIZED.value,
    }


def process_anchor(task: AnchorTask) -> JsonObject:
    mol = CHEM.MolFromSmiles(task.smiles)
    if mol is None:
        return failure_row(task, "invalid_smiles")
    canonical = str(CHEM.MolToSmiles(mol, canonical=True))
    mol_h = CHEM.AddHs(mol)
    formal_charge = int(CHEM.GetFormalCharge(mol))
    try:
        optimized, energy, force_field = optimize_conformer(mol_h, seed=17_001 + task.anchor_idx)
        charges, charge_method = compute_charges(
            optimized,
            formal_charge=formal_charge,
            require_am1_bcc=task.require_am1_bcc,
            timeout_seconds=task.charge_timeout_seconds,
            am1_max_heavy_atoms=task.am1_max_heavy_atoms,
        )
        volume = compute_volume(optimized)
    except (RuntimeError, ValueError, TypeError, OSError, subprocess.SubprocessError) as exc:
        return failure_row(task, f"conformer_failed:{type(exc).__name__}")
    bbox_x, bbox_y, bbox_z = bounding_box(optimized)
    partial_charges = json.dumps(charges, separators=(",", ":"))
    return {
        "anchor_idx": task.anchor_idx,
        "anchor_id": f"CAL-ANCHOR-{task.anchor_idx + 1:06d}",
        "source_anchor_id": task.source_id,
        "source_kind": task.source_kind,
        "input_smiles": task.smiles,
        "canonical_smiles": canonical,
        "n_heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "molecular_weight": float(DESCRIPTORS.MolWt(mol)),
        "formal_charge": formal_charge,
        "steric_volume_A3": volume,
        "bbox_x_A": bbox_x,
        "bbox_y_A": bbox_y,
        "bbox_z_A": bbox_z,
        "mmff_energy_kcal_mol": energy,
        "force_field": force_field,
        "partial_charge_method": charge_method,
        "partial_charges_json": partial_charges,
        "conformer_atoms_json": coordinate_payload(optimized, charges),
        "generation_status": "ok",
        "Epistemic_Class": EpistemicClass.HYPOTHESIZED.value,
        "epistemic_class": EpistemicClass.HYPOTHESIZED.value,
    }


def process_tasks(tasks: list[AnchorTask], processes: int, chunk_size: int) -> pl.DataFrame:
    rows: list[JsonObject] = []
    if processes <= 1:
        iterable: Iterable[JsonObject] = (process_anchor(task) for task in tasks)
        for row in iterable:
            rows.append(row)
            if len(rows) % 64 == 0:
                emit(f"processed anchors={len(rows)}")
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=processes) as pool:
            for row in pool.imap_unordered(process_anchor, tasks, chunksize=chunk_size):
                rows.append(row)
                if len(rows) % 64 == 0:
                    emit(f"processed anchors={len(rows)}")
    return pl.DataFrame(rows).sort("anchor_idx")


def write_anchor_parquet(frame: pl.DataFrame, bundle: TaskBundle, args: argparse.Namespace) -> Path:
    status_counts = {
        str(row["generation_status"]): int(row["len"])
        for row in frame.group_by("generation_status").len().to_dicts()
    }
    charge_method_counts = {
        str(row["partial_charge_method"]): int(row["len"])
        for row in frame.group_by("partial_charge_method").len().to_dicts()
    }
    ok_rows = int(frame.filter(pl.col("generation_status") == "ok").height)
    am1_rows = int(frame.filter(pl.col("partial_charge_method") == "am1-bcc").height)
    return write_provenance_parquet(
        frame,
        args.output,
        producer_script=Path(__file__),
        source_parquets=[],
        schema_version="calibration_anchors_3d.v1",
        pipeline_stage="track_a_calibration_anchor_ingestion",
        partition_keys=["anchor_id"],
        extra_metadata={
            **epistemic_metadata(EpistemicClass.HYPOTHESIZED),
            "requested_charge_method": "AM1-BCC",
            "primary_charge_backend": "AmberTools antechamber",
            "anchor_set": "calibration",
        },
        ledger_parameters={
            "input_source_file_name": args.input.name,
            "input_source_sha256": file_sha256(args.input),
            "input_seed_count": bundle.input_seed_count,
            "requested_anchor_count": int(args.target_size),
            "processed_anchor_count": frame.height,
            "ok_anchor_count": ok_rows,
            "am1_bcc_anchor_count": am1_rows,
            "source_kind_counts": bundle.source_kind_counts,
            "status_counts": status_counts,
            "charge_method_counts": charge_method_counts,
            "brics_augmentation_enabled": bundle.augmented,
            "am1_bcc_backend_available": antechamber_available(),
            "am1_max_heavy_atoms": int(args.am1_max_heavy_atoms),
        },
        ledger_gate_status={
            "epistemic_metadata": True,
            "target_count_reached": frame.height == int(args.target_size),
            "minimum_calibration_count_reached": frame.height >= int(args.min_calibration_size),
            "all_ok_rows_have_3d": ok_rows == int(frame.filter(pl.col("conformer_atoms_json") != "[]").height),
            "am1_bcc_backend_available": antechamber_available(),
        },
    )


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        emit_err(f"Calibration fragment source is missing: {args.input}")
        return 2
    target_size = max(int(args.target_size), int(args.min_calibration_size))
    target_size = min(target_size, MAX_CALIBRATION_SIZE)
    args.target_size = target_size
    if args.require_am1_bcc and not antechamber_available():
        emit_err("AM1-BCC backend is unavailable: AmberTools antechamber was not found")
        return 2
    try:
        bundle = build_tasks(
            input_path=args.input,
            target_size=target_size,
            min_calibration_size=int(args.min_calibration_size),
            require_am1_bcc=bool(args.require_am1_bcc),
            charge_timeout_seconds=int(args.charge_timeout_seconds),
            am1_max_heavy_atoms=int(args.am1_max_heavy_atoms),
        )
    except ValueError as exc:
        emit_err(str(exc))
        return 2

    emit(
        "source="
        f"{args.input.name} input_seeds={bundle.input_seed_count} "
        f"processing={len(bundle.tasks)} augmented={bundle.augmented} processes={args.processes}"
    )
    frame = process_tasks(bundle.tasks, int(args.processes), int(args.chunk_size))
    written = write_anchor_parquet(frame, bundle, args)
    ok_rows = int(frame.filter(pl.col("generation_status") == "ok").height)
    am1_rows = int(frame.filter(pl.col("partial_charge_method") == "am1-bcc").height)
    emit(f"wrote {written.relative_to(REPO_ROOT)} rows={frame.height} ok_rows={ok_rows} am1_bcc_rows={am1_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
