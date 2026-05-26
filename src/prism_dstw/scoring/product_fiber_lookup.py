"""Direct thermodynamic field lookup for product atom fiber tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import torch
from torch import Tensor


PHASE_WEIGHTS_V1 = np.array(
    [
        [1.2, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.8, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.6, 1.2, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.8, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.2, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)

PHASE_WEIGHTS_V2 = np.array(
    [
        [1.2, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0],
        [0.8, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 1.0, 1.0, 1.0],
        [0.6, 1.2, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.8, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 1.0, 1.0, 1.0],
        [1.2, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class GridSpec:
    """Cartesian mapping for one thermodynamic signal grid."""

    origin: np.ndarray
    spacing: float
    dims: tuple[int, int, int]


@dataclass
class VoxelThermodynamicProfile:
    """Unified per-voxel observatory profile used by field-stack v2."""

    voxel_idx: int
    classification: str
    cold_mean: float
    warm_mean: float
    delta: float
    consensus_type: str = "unknown"
    consensus_bonus: float = 0.0
    shear_stress: float = 0.0
    shear_stress_max: float = 0.0
    hysteresis_score: float = 0.0
    reversibility: float = 1.0
    on_activation_pathway: bool = False
    pathway_score: float = 0.0
    pathway_residue_idx: int | None = None
    pathway_provenance: str = "UNAVAILABLE"
    species_conservation_score: float = 1.0
    hysteresis_provenance: str = "UNAVAILABLE"
    phase_spikes: dict[str, float] = field(default_factory=dict)


class SignalGridFiberLookup:
    """Map atom coordinates directly to `[5, 8]` signal-grid fiber vectors.

    Scaffold atoms may keep their pre-computed residue phase tensors. New product
    atoms are mapped to the thermodynamic signal grid by voxel index, so the
    fallback is explicit `L0_VOID` rather than a nearest-scaffold broadcast.
    """

    def __init__(
        self,
        signal_grid_path: str | Path,
        grid_config_path: str | Path,
        *,
        condition_id: str = "glp1r_6XOX_WT",
    ) -> None:
        self.signal_grid_path = Path(signal_grid_path)
        self.grid_config_path = Path(grid_config_path)
        self.condition_id = condition_id
        self.spec = self._load_grid_spec(self.grid_config_path, condition_id)
        grid_df = pl.read_parquet(self.signal_grid_path)
        self.grid = self._load_grid_rows(grid_df)

    @staticmethod
    def _load_grid_spec(path: Path, condition_id: str) -> GridSpec:
        payload = json.loads(path.read_text())
        config: Mapping[str, Any]
        if isinstance(payload, dict) and "conditions" in payload:
            conditions = payload.get("conditions")
            if not isinstance(conditions, dict) or condition_id not in conditions:
                raise ValueError(f"grid mapping missing condition_id={condition_id}")
            config = conditions[condition_id]
        else:
            config = payload
        origin_raw = config.get("origin_xyz_angstrom", config.get("origin", config.get("origin_xyz")))
        if origin_raw is None:
            origin_raw = [config["Ox"], config["Oy"], config["Oz"]]
        origin = np.array(origin_raw, dtype=np.float32)
        spacing = float(config.get("spacing_angstrom", config.get("spacing_A", config.get("spacing"))))
        nx = int(config.get("nx", config.get("grid_dim")))
        ny = int(config.get("ny", config.get("grid_dim")))
        nz = int(config.get("nz", config.get("grid_dim")))
        return GridSpec(origin=origin, spacing=spacing, dims=(nx, ny, nz))

    @staticmethod
    def _load_grid_rows(grid_df: pl.DataFrame) -> dict[int, dict[str, object]]:
        rows: dict[int, dict[str, object]] = {}
        for row in grid_df.iter_rows(named=True):
            voxel_idx = int(row["voxel_idx"])
            cold = _float(row.get("hit_count_cold_mean"))
            warm = _float(row.get("hit_count_warm_mean"))
            delta = _float(row.get("hit_count_delta"), warm - cold)
            classification = str(
                row.get(
                    "variance_classification",
                    row.get("variance_class", row.get("consensus_raw_variance_class", "void")),
                )
            )
            rows[voxel_idx] = {
                "cold_mean": cold,
                "warm_mean": warm,
                "delta": delta,
                "classification": classification,
                "consensus_bonus": _float(row.get("consensus_complement_bonus")),
            }
        return rows

    def xyz_to_voxel(self, xyz: np.ndarray) -> int | None:
        """Map Cartesian coordinates to a signal-grid voxel index."""

        if not bool(np.isfinite(xyz).all()):
            raise ValueError("xyz contains non-finite coordinates")
        ijk = np.floor((xyz.astype(np.float32) - self.spec.origin) / self.spec.spacing).astype(int)
        nx, ny, nz = self.spec.dims
        if not (0 <= ijk[0] < nx and 0 <= ijk[1] < ny and 0 <= ijk[2] < nz):
            return None
        return int(ijk[0] + ijk[1] * nx + ijk[2] * nx * ny)

    def lookup_atom_fiber(self, xyz: np.ndarray) -> np.ndarray:
        """Return a `[5, 8]` fiber vector for one atom coordinate."""

        voxel_idx = self.xyz_to_voxel(xyz)
        if voxel_idx is None or voxel_idx not in self.grid:
            return _void_fiber()
        voxel = self.grid[voxel_idx]
        class_vec = _classification_one_hot(str(voxel["classification"]))
        base_vec = np.array(
            [
                _float(voxel["cold_mean"]),
                _float(voxel["warm_mean"]),
                _float(voxel["delta"]),
                *class_vec,
                _float(voxel["consensus_bonus"]),
            ],
            dtype=np.float32,
        )
        return np.tile(base_vec, (5, 1)) * PHASE_WEIGHTS_V1

    def lookup_product_fiber(
        self,
        scaffold_phase: Tensor,
        product_xyz: np.ndarray,
        *,
        n_scaffold: int,
    ) -> Tensor:
        """Return `[N_product, 5, 8]` fiber data for scaffold plus product atoms."""

        if scaffold_phase.ndim != 3 or int(scaffold_phase.shape[1]) != 5:
            raise ValueError("scaffold_phase must have shape [N_scaffold, 5, D]")
        if int(scaffold_phase.shape[2]) != 8:
            raise ValueError("scaffold_phase feature dimension must be 8")
        if product_xyz.ndim != 2 or int(product_xyz.shape[1]) != 3:
            raise ValueError("product_xyz must have shape [N_product, 3]")
        if not bool(np.isfinite(product_xyz).all()):
            raise ValueError("product_xyz contains non-finite coordinates")
        if n_scaffold < 0 or n_scaffold > int(product_xyz.shape[0]):
            raise ValueError("n_scaffold must be within product atom count")
        if int(scaffold_phase.shape[0]) < n_scaffold:
            raise ValueError("scaffold_phase does not contain n_scaffold rows")

        product_fiber = torch.zeros((int(product_xyz.shape[0]), 5, 8), dtype=torch.float32)
        product_fiber[:n_scaffold] = scaffold_phase[:n_scaffold].detach().to(dtype=torch.float32)
        for atom_index in range(n_scaffold, int(product_xyz.shape[0])):
            product_fiber[atom_index] = torch.from_numpy(self.lookup_atom_fiber(product_xyz[atom_index]))
        return product_fiber

    def field_stats_for_coordinates(self, product_xyz: np.ndarray, *, n_scaffold: int) -> dict[str, float]:
        """Return v1-compatible aggregate metrics for selected product atoms."""

        if product_xyz.ndim != 2 or int(product_xyz.shape[1]) != 3:
            return _empty_field_stats()
        if not bool(np.isfinite(product_xyz).all()):
            raise ValueError("product_xyz contains non-finite coordinates")
        fibers = [
            self.lookup_atom_fiber(product_xyz[index])
            for index in range(max(0, n_scaffold), int(product_xyz.shape[0]))
        ]
        if not fibers:
            return _empty_field_stats()
        stacked = np.stack(fibers, axis=0)
        return {
            **_empty_field_stats(),
            "consensus_complement_bonus": float(stacked[:, :, 7].mean()),
        }


class ThermodynamicFieldStack:
    """Unified lookup for signal grid, shear stress, hysteresis, and pathway fields.

    The signal grid is voxel-native. Shear and pathway are also voxel-keyed in the
    current campaign data. The hysteresis tensor is residue-keyed, so it is mapped
    onto pathway voxels by residue index and tagged with
    ``RESIDUE_PATHWAY_MAPPED`` provenance rather than treated as direct voxel data.
    """

    phase_feature_dim = 12

    def __init__(
        self,
        signal_grid_path: str | Path,
        grid_config_path: str | Path,
        *,
        shear_stress_path: str | Path,
        hysteresis_tensor_path: str | Path,
        translation_pathway_path: str | Path,
        cross_species_path: str | Path | None = None,
        condition_id: str = "glp1r_6XOX_WT",
    ) -> None:
        self.signal_grid_path = Path(signal_grid_path)
        self.grid_config_path = Path(grid_config_path)
        self.shear_stress_path = Path(shear_stress_path)
        self.hysteresis_tensor_path = Path(hysteresis_tensor_path)
        self.translation_pathway_path = Path(translation_pathway_path)
        self.cross_species_path = Path(cross_species_path) if cross_species_path is not None else None
        self.condition_id = condition_id
        self.spec = SignalGridFiberLookup._load_grid_spec(self.grid_config_path, condition_id)
        self.field: dict[int, VoxelThermodynamicProfile] = {}
        self.grid: dict[int, dict[str, object]] = {}
        self.pathway_voxels: set[int] = set()
        self.residue_hysteresis: dict[int, dict[str, float]] = {}
        self.species_conservation: dict[int, dict[str, str]] = {}

        self._load_signal_grid(pl.read_parquet(self.signal_grid_path))
        self._merge_shear(pl.read_parquet(self.shear_stress_path))
        self._load_residue_hysteresis(pl.read_parquet(self.hysteresis_tensor_path))
        if self.cross_species_path is not None and self.cross_species_path.exists():
            self._load_cross_species(self.cross_species_path)
        self._merge_pathway(pl.read_parquet(self.translation_pathway_path))
        self.grid = {
            voxel_idx: {
                "classification": profile.classification,
                "consensus_bonus": profile.consensus_bonus,
                "shear_stress": profile.shear_stress,
            }
            for voxel_idx, profile in self.field.items()
        }

    def _load_signal_grid(self, signal_df: pl.DataFrame) -> None:
        for row in signal_df.iter_rows(named=True):
            voxel_idx = int(row["voxel_idx"])
            cold = _float(row.get("hit_count_cold_mean"))
            warm = _float(row.get("hit_count_warm_mean"))
            self.field[voxel_idx] = VoxelThermodynamicProfile(
                voxel_idx=voxel_idx,
                classification=str(_first_present(row, ("variance_classification", "variance_class", "consensus_raw_variance_class"), "void")),
                cold_mean=cold,
                warm_mean=warm,
                delta=_float(row.get("hit_count_delta"), warm - cold),
                consensus_type=str(_first_present(row, ("consensus_band", "consensus_type"), "unknown")),
                consensus_bonus=_float(row.get("consensus_complement_bonus")),
            )

    def _merge_shear(self, shear_df: pl.DataFrame) -> None:
        if "condition_id" in shear_df.columns:
            preferred = shear_df.filter(pl.col("condition_id") == self.condition_id)
            if preferred.is_empty() and self.condition_id == "glp1r_6XOX_WT":
                preferred = shear_df.filter(pl.col("condition_id").str.contains("6XOX|6X1A"))
            if not preferred.is_empty():
                shear_df = preferred
        for row in shear_df.iter_rows(named=True):
            voxel_idx = int(row["voxel_idx"])
            profile = self.field.get(voxel_idx)
            if profile is None:
                continue
            profile.shear_stress = _float(_first_present(row, ("shear_stress", "frobenius_norm"), 0.0))
            profile.shear_stress_max = _float(row.get("shear_stress_max"), profile.shear_stress)

    def _load_residue_hysteresis(self, hysteresis_df: pl.DataFrame) -> None:
        if "condition_id" in hysteresis_df.columns:
            hysteresis_df = hysteresis_df.filter(pl.col("condition_id") == self.condition_id)
        if "protocol_group" in hysteresis_df.columns:
            protocol_filtered = hysteresis_df.filter(pl.col("protocol_group") == "D_Hysteresis")
            if not protocol_filtered.is_empty():
                hysteresis_df = protocol_filtered
        residue_column = "primary_residue_idx" if "primary_residue_idx" in hysteresis_df.columns else "residue_idx"
        for row in hysteresis_df.iter_rows(named=True):
            if residue_column not in row or row[residue_column] is None:
                continue
            residue_idx = int(row[residue_column])
            cold_hold = _float(_first_present(row, ("cold_hold_spikes", "spike_count_cold_hold"), 0.0))
            ramp_up = _float(_first_present(row, ("ramp_up_spikes", "spike_count_ramp_up"), 1.0), 1.0)
            cold_return = _float(_first_present(row, ("cold_return_spikes", "spike_count_cold_return"), cold_hold))
            if "thermal_irreversibility" in row:
                hysteresis = _float(row.get("thermal_irreversibility"))
            else:
                hysteresis = abs(cold_return - cold_hold) / max(ramp_up, 1.0)
            self.residue_hysteresis[residue_idx] = {
                "hysteresis_score": hysteresis,
                "reversibility": max(0.0, min(1.0, 1.0 / (1.0 + hysteresis))),
                "cold_hold": cold_hold,
                "ramp_up": ramp_up,
                "warm_hold": _float(_first_present(row, ("warm_hold_spikes", "spike_count_warm_hold"), 0.0)),
                "ramp_down": _float(_first_present(row, ("ramp_down_spikes", "spike_count_ramp_down"), 0.0)),
                "cold_return": cold_return,
            }

    def _merge_pathway(self, pathway_df: pl.DataFrame) -> None:
        if "condition_id" in pathway_df.columns:
            pathway_df = pathway_df.filter(pl.col("condition_id") == self.condition_id)
        for row in pathway_df.iter_rows(named=True):
            raw_voxel = row.get("voxel_idx")
            if raw_voxel is None:
                continue
            voxel_idx = int(raw_voxel)
            if voxel_idx < 0:
                continue
            profile = self.field.get(voxel_idx)
            if profile is None:
                continue
            residue_idx = int(row["residue_idx"]) if row.get("residue_idx") is not None else None
            profile.on_activation_pathway = True
            profile.pathway_provenance = "DIRECT_PATHWAY_VOXEL"
            profile.pathway_score = max(
                _float(row.get("wire_score")),
                _float(row.get("load_normalized")),
                _float(row.get("kcc_normalized")),
                1.0,
            )
            profile.pathway_residue_idx = residue_idx
            if residue_idx is not None:
                profile.species_conservation_score = self._conservation_score_for_residue(residue_idx)
            self.pathway_voxels.add(voxel_idx)
            if residue_idx is not None and residue_idx in self.residue_hysteresis:
                hysteresis = self.residue_hysteresis[residue_idx]
                profile.hysteresis_score = hysteresis["hysteresis_score"]
                profile.reversibility = hysteresis["reversibility"]
                profile.phase_spikes = {
                    "cold_hold": hysteresis["cold_hold"],
                    "ramp_up": hysteresis["ramp_up"],
                    "warm_hold": hysteresis["warm_hold"],
                    "ramp_down": hysteresis["ramp_down"],
                    "cold_return": hysteresis["cold_return"],
                }
                profile.hysteresis_provenance = "RESIDUE_PATHWAY_MAPPED"

    def _conservation_score_for_residue(self, residue_idx: int) -> float:
        row = self.species_conservation.get(residue_idx)
        if row is None:
            return 1.0
        try:
            value = float(row.get("conservation_score", "1.0"))
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, min(1.0, value))

    def _load_cross_species(self, path: Path) -> None:
        import csv

        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    residue_idx = int(row["residue_position"])
                except (KeyError, TypeError, ValueError):
                    continue
                self.species_conservation[residue_idx] = row

    def xyz_to_voxel(self, xyz: np.ndarray) -> int | None:
        if not bool(np.isfinite(xyz).all()):
            raise ValueError("xyz contains non-finite coordinates")
        ijk = np.floor((xyz.astype(np.float32) - self.spec.origin) / self.spec.spacing).astype(int)
        nx, ny, nz = self.spec.dims
        if not (0 <= ijk[0] < nx and 0 <= ijk[1] < ny and 0 <= ijk[2] < nz):
            return None
        return int(ijk[0] + ijk[1] * nx + ijk[2] * nx * ny)

    def voxel_to_xyz(self, voxel_idx: int) -> np.ndarray:
        nx, ny, _ = self.spec.dims
        ix = voxel_idx % nx
        iy = (voxel_idx // nx) % ny
        iz = voxel_idx // (nx * ny)
        xyz: np.ndarray = np.asarray(
            self.spec.origin + np.array([ix, iy, iz], dtype=np.float32) * self.spec.spacing,
            dtype=np.float32,
        )
        return xyz

    def _nearest_pathway_profile(self, voxel_idx: int) -> VoxelThermodynamicProfile | None:
        if not self.pathway_voxels:
            return None
        origin = self.voxel_to_xyz(voxel_idx)
        nearest: tuple[float, VoxelThermodynamicProfile] | None = None
        for pathway_voxel in self.pathway_voxels:
            profile = self.field.get(pathway_voxel)
            if profile is None:
                continue
            dist = float(np.linalg.norm(self.voxel_to_xyz(pathway_voxel) - origin))
            if nearest is None or dist < nearest[0]:
                nearest = (dist, profile)
        # The current pathway tensor is sparse at WT 6XOX resolution
        # (one condition-specific pathway voxel). Treat nearby receptor-space
        # atoms as pathway-neighborhood contacts with explicit provenance
        # rather than pretending they are direct pathway voxel hits.
        if nearest is None or nearest[0] > 30.0:
            return None
        return nearest[1]

    def lookup_atom_profile(self, xyz: np.ndarray) -> VoxelThermodynamicProfile | None:
        voxel_idx = self.xyz_to_voxel(xyz)
        if voxel_idx is None:
            return None
        profile = self.field.get(voxel_idx)
        if profile is None:
            return None
        if profile.on_activation_pathway or profile.hysteresis_score > 0.0:
            return profile
        nearest = self._nearest_pathway_profile(voxel_idx)
        if nearest is None:
            return profile
        proxied = VoxelThermodynamicProfile(
            voxel_idx=profile.voxel_idx,
            classification=profile.classification,
            cold_mean=profile.cold_mean,
            warm_mean=profile.warm_mean,
            delta=profile.delta,
            consensus_type=profile.consensus_type,
            consensus_bonus=profile.consensus_bonus,
            shear_stress=profile.shear_stress,
            shear_stress_max=profile.shear_stress_max,
            hysteresis_score=nearest.hysteresis_score,
            reversibility=nearest.reversibility,
            on_activation_pathway=True,
            pathway_score=nearest.pathway_score * 0.5,
            pathway_residue_idx=nearest.pathway_residue_idx,
            pathway_provenance="NEAREST_PATHWAY_NEIGHBORHOOD",
            species_conservation_score=nearest.species_conservation_score,
            hysteresis_provenance="NEAREST_PATHWAY_MAPPED",
            phase_spikes=dict(nearest.phase_spikes),
        )
        return proxied

    def lookup_atom_fiber(self, xyz: np.ndarray) -> np.ndarray:
        profile = self.lookup_atom_profile(xyz)
        if profile is None:
            return _void_fiber_v2()
        class_vec = _classification_one_hot(profile.classification)
        base_vec = np.array(
            [
                profile.cold_mean,
                profile.warm_mean,
                profile.delta,
                *class_vec,
                profile.consensus_bonus,
                profile.shear_stress,
                profile.hysteresis_score,
                profile.pathway_score * profile.species_conservation_score if profile.on_activation_pathway else 0.0,
                profile.reversibility,
            ],
            dtype=np.float32,
        )
        return np.tile(base_vec, (5, 1)) * PHASE_WEIGHTS_V2

    def lookup_product_fiber(
        self,
        scaffold_phase: Tensor,
        product_xyz: np.ndarray,
        *,
        n_scaffold: int,
    ) -> Tensor:
        if scaffold_phase.ndim != 3 or int(scaffold_phase.shape[1]) != 5:
            raise ValueError("scaffold_phase must have shape [N_scaffold, 5, D]")
        if int(scaffold_phase.shape[2]) != 12:
            raise ValueError("field-stack v2 scaffold_phase feature dimension must be 12")
        if product_xyz.ndim != 2 or int(product_xyz.shape[1]) != 3:
            raise ValueError("product_xyz must have shape [N_product, 3]")
        if not bool(np.isfinite(product_xyz).all()):
            raise ValueError("product_xyz contains non-finite coordinates")
        if n_scaffold < 0 or n_scaffold > int(product_xyz.shape[0]):
            raise ValueError("n_scaffold must be within product atom count")
        if int(scaffold_phase.shape[0]) < n_scaffold:
            raise ValueError("scaffold_phase does not contain n_scaffold rows")
        product_fiber = torch.zeros((int(product_xyz.shape[0]), 5, 12), dtype=torch.float32)
        product_fiber[:n_scaffold] = scaffold_phase[:n_scaffold].detach().to(dtype=torch.float32)
        for atom_index in range(n_scaffold, int(product_xyz.shape[0])):
            product_fiber[atom_index] = torch.from_numpy(self.lookup_atom_fiber(product_xyz[atom_index]))
        return product_fiber

    def field_stats_for_coordinates(self, product_xyz: np.ndarray, *, n_scaffold: int) -> dict[str, float]:
        if product_xyz.ndim != 2 or int(product_xyz.shape[1]) != 3:
            return _empty_field_stats()
        if not bool(np.isfinite(product_xyz).all()):
            raise ValueError("product_xyz contains non-finite coordinates")
        profiles = [
            self.lookup_atom_profile(product_xyz[index])
            for index in range(max(0, n_scaffold), int(product_xyz.shape[0]))
        ]
        present = [profile for profile in profiles if profile is not None]
        if not present:
            return _empty_field_stats()
        pathway_scores = [profile.pathway_score if profile.on_activation_pathway else 0.0 for profile in present]
        direct_pathway_scores = [
            profile.pathway_score
            for profile in present
            if profile.pathway_provenance == "DIRECT_PATHWAY_VOXEL"
        ]
        neighborhood_pathway_scores = [
            profile.pathway_score
            for profile in present
            if profile.pathway_provenance == "NEAREST_PATHWAY_NEIGHBORHOOD"
        ]
        conservation_scores = [
            profile.species_conservation_score
            for profile in present
            if profile.on_activation_pathway and profile.pathway_residue_idx is not None
        ]
        return {
            "sigma_shear_mean": _mean_float([profile.shear_stress for profile in present]),
            "hysteresis_mean": _mean_float([profile.hysteresis_score for profile in present]),
            "reversibility_mean": _mean_float([profile.reversibility for profile in present], default=1.0),
            "pathway_voxels_occupied": float(len(direct_pathway_scores)),
            "pathway_neighborhood_contacts": float(len(neighborhood_pathway_scores)),
            "pathway_neighborhood_score_mean": _mean_float(neighborhood_pathway_scores),
            "pathway_score_mean": _mean_float(pathway_scores),
            "species_conservation_score_mean": _mean_float(conservation_scores, default=1.0),
            "consensus_complement_bonus": _mean_float([profile.consensus_bonus for profile in present]),
        }


def _classification_one_hot(classification: str) -> list[float]:
    normalized = classification.lower()
    if normalized == "stable_occupied":
        return [1.0, 0.0, 0.0, 0.0]
    if normalized == "thermally_destabilized":
        return [0.0, 1.0, 0.0, 0.0]
    if normalized == "thermally_activated":
        return [0.0, 0.0, 1.0, 0.0]
    if normalized == "variant_disputed":
        return [0.5, 0.5, 0.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


def _void_fiber() -> np.ndarray:
    fiber = np.zeros((5, 8), dtype=np.float32)
    fiber[:, 6] = 1.0
    return fiber


def _void_fiber_v2() -> np.ndarray:
    fiber = np.zeros((5, 12), dtype=np.float32)
    fiber[:, 6] = 1.0
    fiber[:, 11] = 1.0
    return fiber


def _empty_field_stats() -> dict[str, float]:
    return {
        "sigma_shear_mean": 0.0,
        "hysteresis_mean": 0.0,
        "reversibility_mean": 1.0,
        "pathway_voxels_occupied": 0.0,
        "pathway_neighborhood_contacts": 0.0,
        "pathway_neighborhood_score_mean": 0.0,
        "pathway_score_mean": 0.0,
        "species_conservation_score_mean": 1.0,
        "consensus_complement_bonus": 0.0,
    }


def _mean_float(values: Sequence[float], default: float = 0.0) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return default
    return float(sum(finite) / len(finite))


def _first_present(row: Mapping[str, object], names: Sequence[str], default: object) -> object:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return default


def _float(value: object, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int | float | str):
        parsed = float(value)
        return parsed if np.isfinite(parsed) else default
    return default


__all__ = [
    "GridSpec",
    "SignalGridFiberLookup",
    "ThermodynamicFieldStack",
    "VoxelThermodynamicProfile",
]
