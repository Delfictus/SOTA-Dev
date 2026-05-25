"""PRISM-FORGE thermodynamic reward wrapper."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt
import polars as pl


VarianceCode = np.int8
FloatArray = npt.NDArray[np.float32]
UIntArray = npt.NDArray[np.uint32]
Int8Array = npt.NDArray[np.int8]

VARIANCE_CODES: Mapping[str, int] = {
    "stable_occupied": 1,
    "thermally_activated": 2,
    "thermally_destabilized": 3,
}


@dataclass(frozen=True)
class RewardField:
    condition_id: str
    grid_dim: int
    origin_xyz: tuple[float, float, float]
    spacing_A: float
    voxel_indices: UIntArray
    variance_codes: Int8Array

    @classmethod
    def from_files(cls, *, grid_mapping: Path, signal_grid: Path, condition_id: str) -> "RewardField":
        mapping = _mapping(json.loads(grid_mapping.read_text(encoding="utf-8")), "grid_coordinate_mapping")
        conditions = _mapping(mapping.get("conditions"), "conditions")
        geometry = _mapping(conditions.get(condition_id), f"conditions[{condition_id}]")
        origin = _triple(geometry.get("origin_xyz_angstrom"), "origin_xyz_angstrom")
        grid_dim = _positive_int(geometry.get("grid_dim"), "grid_dim")
        spacing = _positive_float(geometry.get("spacing_angstrom"), "spacing_angstrom")
        frame = (
            pl.scan_parquet(signal_grid)
            .filter(pl.col("condition_id") == condition_id)
            .filter(pl.col("variance_class").is_in(list(VARIANCE_CODES)))
            .select(["voxel_idx", "variance_class"])
            .collect()
        )
        voxel_indices = np.ascontiguousarray(
            np.asarray([int(value) for value in frame.get_column("voxel_idx").to_list()], dtype=np.uint32)
        )
        variance_codes = np.ascontiguousarray(
            np.asarray(
                [VARIANCE_CODES[str(value)] for value in frame.get_column("variance_class").to_list()],
                dtype=np.int8,
            )
        )
        return cls(
            condition_id=condition_id,
            grid_dim=grid_dim,
            origin_xyz=origin,
            spacing_A=spacing,
            voxel_indices=voxel_indices,
            variance_codes=variance_codes,
        )


@dataclass(frozen=True)
class RewardBreakdown:
    rust_reward: float
    synthetic_accessibility_score: float
    final_reward: float


def compute_reward_3d(
    *,
    coordinates: FloatArray,
    charges: FloatArray,
    field: RewardField,
    smiles: str,
    lambda_sa: float = 0.08,
    beta_f: float = 1.0,
    beta_s: float = 1.0,
    rust_module: ModuleType | None = None,
) -> RewardBreakdown:
    if lambda_sa < 0.0:
        raise ValueError("lambda_sa must be non-negative")
    module = rust_module or load_prism_forge_extension()
    coords = _float32_contiguous(coordinates, "coordinates")
    charge_array = _float32_contiguous(charges, "charges")
    origin = _float32_contiguous(np.asarray(field.origin_xyz, dtype=np.float32), "origin_xyz")
    rust_reward = float(
        module.compute_thermodynamic_reward_3d(
            coords,
            charge_array,
            origin,
            float(field.spacing_A),
            int(field.grid_dim),
            field.voxel_indices,
            field.variance_codes,
            float(beta_f),
            float(beta_s),
        )
    )
    sa_score = synthetic_accessibility_score(smiles)
    final_reward = rust_reward * math.exp(-lambda_sa * sa_score)
    if not math.isfinite(final_reward) or final_reward <= 0.0:
        raise ValueError("final thermodynamic reward must be finite and positive")
    return RewardBreakdown(
        rust_reward=rust_reward,
        synthetic_accessibility_score=sa_score,
        final_reward=final_reward,
    )


def load_prism_forge_extension(*, repo_root: Path | None = None, build_if_missing: bool = False) -> ModuleType:
    try:
        return importlib.import_module("prism_forge")
    except ImportError:
        pass
    root = repo_root or Path(__file__).resolve().parents[3]
    extension = _find_local_extension(root)
    if extension is None and build_if_missing:
        subprocess.run(["cargo", "build", "-p", "prism-forge"], cwd=root, check=True)
        extension = _find_local_extension(root)
    if extension is None:
        raise ImportError("prism_forge extension was not importable and no local build artifact was found")
    return _load_extension(extension)


def synthetic_accessibility_score(smiles: str) -> float:
    try:
        sascorer = importlib.import_module("rdkit.Contrib.SA_Score.sascorer")
        chem = importlib.import_module("rdkit.Chem")
        mol = chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        return float(sascorer.calculateScore(mol))
    except Exception:
        heavy_atoms = sum(1 for char in smiles if char.isalpha() and char.isupper())
        ring_tokens = sum(1 for char in smiles if char.isdigit())
        return max(1.0, min(10.0, 1.0 + 0.12 * float(heavy_atoms) + 0.20 * float(ring_tokens)))


def _find_local_extension(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "target/debug/libprism_forge.so",
        repo_root / "target/release/libprism_forge.so",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_extension(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("prism_forge", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["prism_forge"] = module
    spec.loader.exec_module(module)
    return module


def _float32_contiguous(value: npt.ArrayLike, label: str) -> FloatArray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if array.ndim == 0:
        raise ValueError(f"{label} must not be scalar")
    return array


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _triple(value: object, label: str) -> tuple[float, float, float]:
    sequence = _sequence(value, label)
    if len(sequence) != 3:
        raise ValueError(f"{label} must have exactly 3 elements")
    return (_float(sequence[0], label), _float(sequence[1], label), _float(sequence[2], label))


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _positive_float(value: object, label: str) -> float:
    parsed = _float(value, label)
    if not parsed > 0.0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


__all__ = [
    "RewardBreakdown",
    "RewardField",
    "compute_reward_3d",
    "load_prism_forge_extension",
    "synthetic_accessibility_score",
]
