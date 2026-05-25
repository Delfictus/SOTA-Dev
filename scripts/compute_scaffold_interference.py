#!/usr/bin/env python3
"""Score an aligned or alignable SDF against receptor-side steric environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import polars as pl
from jinja2 import Environment, StrictUndefined
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import rdMolAlign


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import write_provenance_parquet


Coordinate: TypeAlias = tuple[float, float, float]
FloatArray: TypeAlias = NDArray[np.float64]

TRACK0_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation"
N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
DEFAULT_GRID_MAPPING = TRACK0_DIR / "grid_coordinate_mapping.json"
DEFAULT_BINDING_SITE = TRACK0_DIR / "binding_site_reference.json"
DEFAULT_STERIC_ENV = TRACK0_DIR / "interface_steric_environment.parquet"
DEFAULT_RISK_MAP = N80_DIR / "receptor_durability_risk_map.parquet"

REPORT_TEMPLATE = """# Scaffold Interference Report

Analog: `{{ analog_id }}`

Alignment mode: `{{ alignment_mode }}`

| Edge | Class | Risk | E[Pi_clash] | E[Pi_complement] | U_pose | Multiplier | Projected score |
|---|---:|---:|---:|---:|---:|---:|---:|
{% for row in edges -%}
| {{ row.edge_label }} | {{ row.edge_class }} | {{ "%.4f"|format(row.durability_risk_score_raw) }} | {{ "%.4f"|format(row.expected_pi_clash) }} | {{ "%.4f"|format(row.expected_pi_complement) }} | {{ "%.4f"|format(row.u_pose) }} | {{ "%.4f"|format(row.multiplier) }} | {{ "%.4f"|format(row.projected_edge_score) }} |
{% endfor %}

Projected durability score: `{{ "%.4f"|format(projected_durability_score) }}`

Projected durability std: `{{ "%.4f"|format(projected_durability_std) }}`
"""


@dataclass(frozen=True)
class GridGeometry:
    condition_id: str
    grid_dim: int
    origin: Coordinate
    spacing: float


@dataclass(frozen=True)
class EdgeReference:
    edge_id: str
    condition_id: str
    edge_label: str
    edge_class: str
    edge_from_residue: int
    edge_to_residue: int
    reference_points: tuple[Coordinate, ...]


@dataclass(frozen=True)
class EdgeRisk:
    edge_id: str
    condition_id: str
    edge_label: str
    edge_class: str
    edge_from_residue: int
    edge_to_residue: int
    durability_risk_score_raw: float
    signed_te_mean: float


@dataclass(frozen=True)
class VoxelDensity:
    variance_class: str
    hit_count_cold_mean: float
    hit_count_warm_mean: float
    hit_count_delta: float


@dataclass(frozen=True)
class NormalizationConstants:
    quantile: float
    cold_mean_p95: float
    delta_abs_p95: float


@dataclass(frozen=True)
class BetaCalibration:
    beta_f: float
    beta_s: float
    beta_f_mode: str
    beta_s_mode: str
    max_clash: float
    max_complement: float
    method: str


@dataclass(frozen=True)
class ConformerRecord:
    mol: Any
    conf_id: int
    label: str
    prealigned: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--binding-site", type=Path, default=DEFAULT_BINDING_SITE)
    parser.add_argument("--steric-env", type=Path, default=DEFAULT_STERIC_ENV)
    parser.add_argument("--risk-map", type=Path, default=DEFAULT_RISK_MAP)
    parser.add_argument("--normalization-source", type=Path, default=None)
    parser.add_argument("--normalization-quantile", type=float, default=0.95)
    parser.add_argument("--beta-f", default="auto")
    parser.add_argument("--beta-s", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not decode to a JSON object")
    return loaded


def json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return cast(dict[str, object], value)


def json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a list")
    return value


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


def coordinate_from_json(value: object, label: str) -> Coordinate:
    raw = json_list(value, label)
    if len(raw) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    return (
        as_float(raw[0], f"{label}[0]"),
        as_float(raw[1], f"{label}[1]"),
        as_float(raw[2], f"{label}[2]"),
    )


def load_geometries(path: Path) -> dict[str, GridGeometry]:
    payload = load_json_object(path)
    conditions = json_object(payload.get("conditions"), f"{path}:conditions")
    geometries: dict[str, GridGeometry] = {}
    for condition_id, raw_geometry in conditions.items():
        geometry = json_object(raw_geometry, f"{path}:conditions[{condition_id}]")
        geometries[condition_id] = GridGeometry(
            condition_id=condition_id,
            grid_dim=as_int(geometry["grid_dim"], "grid_dim"),
            origin=coordinate_from_json(
                geometry.get("origin_xyz_angstrom"),
                f"{path}:conditions[{condition_id}].origin_xyz_angstrom",
            ),
            spacing=as_float(geometry["spacing_angstrom"], "spacing_angstrom"),
        )
    return geometries


def load_edge_references(path: Path) -> dict[str, EdgeReference]:
    payload = load_json_object(path)
    references: dict[str, EdgeReference] = {}
    for raw_edge in json_list(payload.get("critical_edges"), f"{path}:critical_edges"):
        edge = json_object(raw_edge, f"{path}:critical_edges[]")
        points: list[Coordinate] = []
        for key in ("from_atom_coordinates", "to_atom_coordinates"):
            for raw_atom in json_list(edge.get(key), f"{path}:{key}"):
                atom = json_object(raw_atom, f"{path}:{key}[]")
                points.append(coordinate_from_json(atom.get("xyz_angstrom"), f"{path}:{key}.xyz"))
        edge_id = str(edge["edge_id"])
        references[edge_id] = EdgeReference(
            edge_id=edge_id,
            condition_id=str(edge["condition_id"]),
            edge_label=str(edge["edge_label"]),
            edge_class=str(edge["edge_class"]),
            edge_from_residue=as_int(edge["edge_from_residue"], "edge_from_residue"),
            edge_to_residue=as_int(edge["edge_to_residue"], "edge_to_residue"),
            reference_points=tuple(points),
        )
    return references


def sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def heavy_atom_indices(mol: Any) -> list[int]:
    return [int(atom.GetIdx()) for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) > 1]


def conformer_coordinates(mol: Any, conf_id: int, atom_indices: list[int]) -> FloatArray:
    conf = mol.GetConformer(conf_id)
    rows: list[list[float]] = []
    for atom_idx in atom_indices:
        pos = conf.GetAtomPosition(atom_idx)
        rows.append([float(pos.x), float(pos.y), float(pos.z)])
    return np.asarray(rows, dtype=np.float64)


def load_sdf_conformers(path: Path) -> list[ConformerRecord]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    conformers: list[ConformerRecord] = []
    for mol_idx, mol in enumerate(supplier):
        if mol is None:
            continue
        if mol.GetNumConformers() > 1:
            rdMolAlign.AlignMolConformers(mol)
        for conf in mol.GetConformers():
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{path.stem}_{mol_idx}"
            prealigned = bool(
                (mol.HasProp("aligned_to_receptor") and mol.GetProp("aligned_to_receptor").lower() == "true")
                or (mol.HasProp("alignment_inherited") and mol.GetProp("alignment_inherited").lower() == "true")
            )
            conformers.append(
                ConformerRecord(
                    mol=mol,
                    conf_id=int(conf.GetId()),
                    label=name,
                    prealigned=prealigned,
                )
            )
    if not conformers:
        raise ValueError(f"{path} did not contain any valid RDKit conformers")
    return conformers


def axes_for_points(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    if points.shape[0] < 3:
        return centroid, np.eye(3, dtype=np.float64)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = np.asarray(vh, dtype=np.float64)
    if axes.shape != (3, 3):
        padded = np.eye(3, dtype=np.float64)
        padded[: axes.shape[0], :] = axes
        axes = padded
    return centroid, axes


def align_to_reference_shell(coords: FloatArray, reference_points: tuple[Coordinate, ...]) -> FloatArray:
    ref = np.asarray(reference_points, dtype=np.float64)
    mol_centroid, mol_axes = axes_for_points(coords)
    ref_centroid, ref_axes = axes_for_points(ref)
    rotation = mol_axes.T @ ref_axes
    return (coords - mol_centroid) @ rotation + ref_centroid


def map_coordinate_to_voxel(coord: Coordinate, geometry: GridGeometry) -> int | None:
    ix = math.trunc((coord[0] - geometry.origin[0]) / geometry.spacing)
    iy = math.trunc((coord[1] - geometry.origin[1]) / geometry.spacing)
    iz = math.trunc((coord[2] - geometry.origin[2]) / geometry.spacing)
    if ix < 0 or iy < 0 or iz < 0:
        return None
    if ix >= geometry.grid_dim or iy >= geometry.grid_dim or iz >= geometry.grid_dim:
        return None
    return iz * geometry.grid_dim * geometry.grid_dim + iy * geometry.grid_dim + ix


def variance_class_column(path: Path) -> str:
    columns = set(pl.scan_parquet(path).collect_schema().names())
    if "variance_classification" in columns:
        return "variance_classification"
    if "variance_class" in columns:
        return "variance_class"
    raise ValueError(f"{path} has no variance class column")


def hit_delta_expr(path: Path) -> pl.Expr:
    columns = set(pl.scan_parquet(path).collect_schema().names())
    if "hit_count_delta" in columns:
        return pl.col("hit_count_delta")
    return pl.col("hit_count_warm_mean") - pl.col("hit_count_cold_mean")


def compute_normalization_constants(path: Path, quantile: float) -> NormalizationConstants:
    class_col = variance_class_column(path)
    frame = (
        pl.scan_parquet(path)
        .with_columns(hit_delta_expr(path).alias("hit_count_delta_for_normalization"))
        .filter(pl.col(class_col) != "void")
    )
    cold_value = frame.select(pl.col("hit_count_cold_mean").quantile(quantile)).collect().item()
    delta_value = (
        frame.select(pl.col("hit_count_delta_for_normalization").abs().quantile(quantile))
        .collect()
        .item()
    )
    return NormalizationConstants(
        quantile=quantile,
        cold_mean_p95=max(as_float(cold_value, "cold_mean_quantile"), 1.0),
        delta_abs_p95=max(as_float(delta_value, "delta_abs_quantile"), 1.0),
    )


def steric_lookup(path: Path, risk_map: Path) -> tuple[dict[str, dict[int, VoxelDensity]], list[EdgeRisk]]:
    class_col = variance_class_column(path)
    frame = pl.scan_parquet(path).select(
        [
            "edge_id",
            "condition_id",
            "edge_label",
            "edge_class",
            "edge_from_residue",
            "edge_to_residue",
            "voxel_idx",
            pl.col(class_col).alias("variance_class"),
            "hit_count_cold_mean",
            "hit_count_warm_mean",
            hit_delta_expr(path).alias("hit_count_delta"),
        ]
    )
    risk = pl.scan_parquet(risk_map).select(
        [
            "condition_id",
            "edge_from_residue",
            "edge_to_residue",
            "durability_risk_score_raw",
            "signed_te_mean",
        ]
    )
    joined = frame.join(
        risk,
        on=["condition_id", "edge_from_residue", "edge_to_residue"],
        how="left",
    ).collect()
    lookup: dict[str, dict[int, VoxelDensity]] = {}
    for row in joined.select(
        [
            "edge_id",
            "voxel_idx",
            "variance_class",
            "hit_count_cold_mean",
            "hit_count_warm_mean",
            "hit_count_delta",
        ]
    ).to_dicts():
        edge_id = str(row["edge_id"])
        lookup.setdefault(edge_id, {})[as_int(row["voxel_idx"], "voxel_idx")] = VoxelDensity(
            variance_class=str(row["variance_class"]),
            hit_count_cold_mean=as_float(row["hit_count_cold_mean"], "hit_count_cold_mean"),
            hit_count_warm_mean=as_float(row["hit_count_warm_mean"], "hit_count_warm_mean"),
            hit_count_delta=as_float(row["hit_count_delta"], "hit_count_delta"),
        )
    risks: list[EdgeRisk] = []
    for row in (
        joined.select(
            [
                "edge_id",
                "condition_id",
                "edge_label",
                "edge_class",
                "edge_from_residue",
                "edge_to_residue",
                "durability_risk_score_raw",
                "signed_te_mean",
            ]
        )
        .unique()
        .sort("edge_id")
        .to_dicts()
    ):
        risks.append(
            EdgeRisk(
                edge_id=str(row["edge_id"]),
                condition_id=str(row["condition_id"]),
                edge_label=str(row["edge_label"]),
                edge_class=str(row["edge_class"]),
                edge_from_residue=as_int(row["edge_from_residue"], "edge_from_residue"),
                edge_to_residue=as_int(row["edge_to_residue"], "edge_to_residue"),
                durability_risk_score_raw=as_float(row["durability_risk_score_raw"], "durability_risk_score_raw"),
                signed_te_mean=as_float(row["signed_te_mean"], "signed_te_mean"),
            )
        )
    return lookup, risks


def score_atom_interference(
    hit_count_cold_mean: float,
    hit_count_warm_mean: float,
    hit_count_delta: float,
    variance_classification: str,
    signed_te_mean: float,
    cold_mean_p95: float,
    delta_abs_p95: float,
) -> tuple[float, float]:
    """Return density-weighted clash/complement contributions for one atom.

    The formula normalizes cold receptor occupancy and absolute warm-cold
    density change by distribution-derived quantiles. Stable occupied voxels
    always contribute clash proportional to cold density. Thermally activated
    voxels always contribute complement proportional to density change.
    Thermally destabilized voxels use signed transfer entropy as context:
    positive TE treats melting density as pocket-proximal complement, while
    negative TE treats melting lock density as clash risk.
    """

    _ = hit_count_warm_mean
    cold_normalized = hit_count_cold_mean / max(cold_mean_p95, 1.0)
    delta_normalized = abs(hit_count_delta) / max(delta_abs_p95, 1.0)
    if variance_classification == "stable_occupied":
        return (cold_normalized, 0.0)
    if variance_classification == "thermally_destabilized":
        if signed_te_mean >= 0.0:
            return (0.0, delta_normalized * 0.5)
        return (delta_normalized * 0.5, 0.0)
    if variance_classification == "thermally_activated":
        return (0.0, delta_normalized)
    return (0.0, 0.0)


def population_variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / float(len(values))
    return sum((value - mean) ** 2 for value in values) / float(len(values))


def score_edges(
    conformers: list[ConformerRecord],
    edge_risks: list[EdgeRisk],
    edge_refs: dict[str, EdgeReference],
    geometries: dict[str, GridGeometry],
    lookup: dict[str, dict[int, VoxelDensity]],
    normalization: NormalizationConstants,
    beta_f_override: float | None,
    beta_s_override: float | None,
) -> tuple[pl.DataFrame, BetaCalibration]:
    raw_rows: list[dict[str, object]] = []
    for edge in edge_risks:
        reference = edge_refs[edge.edge_id]
        geometry = geometries[edge.condition_id]
        edge_lookup = lookup[edge.edge_id]
        pi_clash_values: list[float] = []
        pi_complement_values: list[float] = []
        for conformer in conformers:
            atom_indices = heavy_atom_indices(conformer.mol)
            coords = conformer_coordinates(conformer.mol, conformer.conf_id, atom_indices)
            scoring_coords = coords if conformer.prealigned else align_to_reference_shell(coords, reference.reference_points)
            pi_clash = 0.0
            pi_complement = 0.0
            for row in scoring_coords:
                current_voxel = map_coordinate_to_voxel((float(row[0]), float(row[1]), float(row[2])), geometry)
                if current_voxel is None:
                    continue
                density = edge_lookup.get(current_voxel)
                if density is None:
                    continue
                clash_increment, complement_increment = score_atom_interference(
                    density.hit_count_cold_mean,
                    density.hit_count_warm_mean,
                    density.hit_count_delta,
                    density.variance_class,
                    edge.signed_te_mean,
                    normalization.cold_mean_p95,
                    normalization.delta_abs_p95,
                )
                pi_clash += clash_increment
                pi_complement += complement_increment
            pi_clash_values.append(pi_clash)
            pi_complement_values.append(pi_complement)
        expected_clash = sum(pi_clash_values) / float(len(pi_clash_values))
        expected_complement = sum(pi_complement_values) / float(len(pi_complement_values))
        raw_rows.append(
            {
                "edge": edge,
                "expected_clash": expected_clash,
                "expected_complement": expected_complement,
                "pi_clash_values": pi_clash_values,
                "pi_complement_values": pi_complement_values,
            }
        )
    max_clash = max([as_float(row["expected_clash"], "expected_clash") for row in raw_rows], default=0.0)
    max_complement = max(
        [as_float(row["expected_complement"], "expected_complement") for row in raw_rows],
        default=0.0,
    )
    beta_f = beta_f_override if beta_f_override is not None else math.log(10.0) / max(max_clash, 0.1)
    beta_s = beta_s_override if beta_s_override is not None else math.log(10.0) / max(max_complement, 0.1)
    calibration = BetaCalibration(
        beta_f=beta_f,
        beta_s=beta_s,
        beta_f_mode="manual" if beta_f_override is not None else "auto",
        beta_s_mode="manual" if beta_s_override is not None else "auto",
        max_clash=max_clash,
        max_complement=max_complement,
        method="ln(10)/max(max_score,0.1)",
    )
    rows: list[dict[str, int | float | str]] = []
    for raw_row in raw_rows:
        raw_edge_obj = raw_row["edge"]
        if not isinstance(raw_edge_obj, EdgeRisk):
            raise TypeError("raw row edge was not EdgeRisk")
        edge = raw_edge_obj
        pi_clash_values = [as_float(value, "pi_clash") for value in json_list(raw_row["pi_clash_values"], "pi_clash_values")]
        pi_complement_values = [
            as_float(value, "pi_complement")
            for value in json_list(raw_row["pi_complement_values"], "pi_complement_values")
        ]
        expected_clash = as_float(raw_row["expected_clash"], "expected_clash")
        expected_complement = as_float(raw_row["expected_complement"], "expected_complement")
        multiplier_values = [
            math.exp(beta_f * clash) * math.exp(-beta_s * complement)
            for clash, complement in zip(pi_clash_values, pi_complement_values, strict=True)
        ]
        multiplier = math.exp(beta_f * expected_clash) * math.exp(-beta_s * expected_complement)
        multiplier_variance = population_variance(multiplier_values)
        u_pose = population_variance(
            [clash - complement for clash, complement in zip(pi_clash_values, pi_complement_values, strict=True)]
        )
        rows.append(
            {
                "edge_id": edge.edge_id,
                "condition_id": edge.condition_id,
                "edge_label": edge.edge_label,
                "edge_class": edge.edge_class,
                "edge_from_residue": edge.edge_from_residue,
                "edge_to_residue": edge.edge_to_residue,
                "durability_risk_score_raw": edge.durability_risk_score_raw,
                "signed_te_mean": edge.signed_te_mean,
                "n_conformers": len(conformers),
                "expected_pi_clash": expected_clash,
                "expected_pi_complement": expected_complement,
                "u_pose": u_pose,
                "multiplier": multiplier,
                "E_pi_clash": expected_clash,
                "E_pi_complement": expected_complement,
                "U_pose": u_pose,
                "te_multiplier": multiplier,
                "multiplier_variance": multiplier_variance,
                "projected_edge_score": edge.durability_risk_score_raw * multiplier,
                "beta_f": beta_f,
                "beta_s": beta_s,
            }
        )
    return pl.DataFrame(rows), calibration


def projection_frame(edge_scores: pl.DataFrame, analog_id: str, beta_f: float, beta_s: float) -> pl.DataFrame:
    score = as_float(edge_scores["projected_edge_score"].sum(), "projected_edge_score_sum")
    variance = as_float(
        (
            edge_scores["durability_risk_score_raw"]
            * edge_scores["durability_risk_score_raw"]
            * edge_scores["multiplier_variance"]
        ).sum(),
        "projection_variance",
    )
    return pl.DataFrame(
        [
            {
                "analog_id": analog_id,
                "n_edges": edge_scores.height,
                "n_conformers": as_int(edge_scores["n_conformers"].max(), "n_conformers"),
                "beta_f": beta_f,
                "beta_s": beta_s,
                "projected_durability_score": score,
                "projected_durability_std": math.sqrt(max(variance, 0.0)),
                "total_projected_durability": score,
                "total_projected_uncertainty": math.sqrt(max(variance, 0.0)),
            }
        ]
    )


def render_report(path: Path, analog_id: str, edge_scores: pl.DataFrame, projection: pl.DataFrame, alignment_mode: str) -> None:
    env = Environment(undefined=StrictUndefined, autoescape=False)
    template = env.from_string(REPORT_TEMPLATE)
    projection_row = projection.to_dicts()[0]
    rendered = template.render(
        analog_id=analog_id,
        alignment_mode=alignment_mode,
        edges=edge_scores.sort("projected_edge_score", descending=True).to_dicts(),
        projected_durability_score=as_float(
            projection_row["projected_durability_score"],
            "projected_durability_score",
        ),
        projected_durability_std=as_float(
            projection_row["projected_durability_std"],
            "projected_durability_std",
        ),
    )
    path.write_text(rendered, encoding="utf-8")


def validate_risk_edges(risk_map: Path, edge_scores: pl.DataFrame) -> None:
    keys = edge_scores.select(["condition_id", "edge_from_residue", "edge_to_residue", "edge_class"]).unique()
    risk_matches = (
        keys.lazy()
        .join(
            pl.scan_parquet(risk_map).select(
                ["condition_id", "edge_from_residue", "edge_to_residue", "edge_class"]
            ),
            on=["condition_id", "edge_from_residue", "edge_to_residue", "edge_class"],
            how="inner",
        )
        .collect()
    )
    if risk_matches.height != keys.height:
        raise ValueError("risk map did not contain every edge present in the steric environment")


def parse_beta_argument(value: object, label: str) -> float | None:
    text = str(value).strip().lower()
    if text == "auto":
        return None
    return as_float(text, label)


def unique_parquet_sources(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            out.append(path)
            seen.add(resolved)
    return out


def calibration_parameters(
    normalization: NormalizationConstants,
    calibration: BetaCalibration,
    edge_scores: pl.DataFrame,
) -> dict[str, object]:
    return {
        "normalization": {
            "quantile": normalization.quantile,
            "cold_mean_p95": normalization.cold_mean_p95,
            "delta_abs_p95": normalization.delta_abs_p95,
        },
        "beta_calibration": {
            "beta_f": calibration.beta_f,
            "beta_s": calibration.beta_s,
            "beta_f_mode": calibration.beta_f_mode,
            "beta_s_mode": calibration.beta_s_mode,
            "max_clash": calibration.max_clash,
            "max_complement": calibration.max_complement,
            "method": calibration.method,
        },
        "signed_te_by_edge": {
            str(row["edge_id"]): as_float(row["signed_te_mean"], "signed_te_mean")
            for row in edge_scores.select(["edge_id", "signed_te_mean"]).to_dicts()
        },
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=str(args.log_level).upper(), format="%(levelname)s %(message)s")
    geometries = load_geometries(args.grid_mapping)
    edge_refs = load_edge_references(args.binding_site)
    normalization_source = args.normalization_source or args.steric_env
    normalization = compute_normalization_constants(normalization_source, float(args.normalization_quantile))
    lookup, edge_risks = steric_lookup(args.steric_env, args.risk_map)
    conformers = load_sdf_conformers(args.sdf)
    alignment_mode = (
        "sdf_receptor_frame_prealigned"
        if all(conformer.prealigned for conformer in conformers)
        else "receptor_shell_principal_axes_no_ligand_reference"
    )
    beta_f_override = parse_beta_argument(args.beta_f, "beta_f")
    beta_s_override = parse_beta_argument(args.beta_s, "beta_s")
    edge_scores, calibration = score_edges(
        conformers,
        edge_risks,
        edge_refs,
        geometries,
        lookup,
        normalization,
        beta_f_override,
        beta_s_override,
    )
    validate_risk_edges(args.risk_map, edge_scores)
    projection = projection_frame(edge_scores, args.sdf.stem, calibration.beta_f, calibration.beta_s)
    sys.stdout.write(
        "Normalization constants: "
        f"cold_mean_p95={normalization.cold_mean_p95:.6f} "
        f"delta_abs_p95={normalization.delta_abs_p95:.6f}\n"
    )
    sys.stdout.write(
        "Active beta: "
        f"beta_f={calibration.beta_f:.12f} beta_s={calibration.beta_s:.12f} "
        f"mode_f={calibration.beta_f_mode} mode_s={calibration.beta_s_mode}\n"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_edge_path = args.out_dir / "per_edge_interference.parquet"
    projection_path = args.out_dir / "analog_durability_projection.parquet"
    report_path = args.out_dir / "interference_report.md"
    source_parquets = unique_parquet_sources([args.steric_env, args.risk_map, normalization_source])
    calibration_metadata = calibration_parameters(normalization, calibration, edge_scores)
    ledger_parameters = {
        "sdf_path": args.sdf.as_posix(),
        "sdf_sha256": sha256_path(args.sdf),
        "grid_mapping_json": args.grid_mapping.as_posix(),
        "binding_site_reference_json": args.binding_site.as_posix(),
        "beta_f": calibration.beta_f,
        "beta_s": calibration.beta_s,
        "alignment_mode": alignment_mode,
        "density_weighted_scoring": "cold density and absolute warm-cold density delta normalized by quantile denominators; TE sign directs thermally_destabilized context",
        **calibration_metadata,
    }
    write_provenance_parquet(
        edge_scores,
        per_edge_path,
        producer_script=Path(__file__),
        source_parquets=source_parquets,
        schema_version="scaffold_interference_per_edge.v1",
        pipeline_stage="track0_scaffold_interference_per_edge",
        partition_keys=["edge_id"],
        extra_metadata=calibration_metadata,
        ledger_parameters=ledger_parameters,
        ledger_output_value={"rows": edge_scores.height, "output_path": per_edge_path.as_posix()},
        repo_root=REPO_ROOT,
    )
    write_provenance_parquet(
        projection,
        projection_path,
        producer_script=Path(__file__),
        source_parquets=source_parquets,
        schema_version="scaffold_interference_projection.v1",
        pipeline_stage="track0_scaffold_interference_projection",
        partition_keys=["analog_id"],
        extra_metadata=calibration_metadata,
        ledger_parameters=ledger_parameters,
        ledger_output_value={"rows": projection.height, "output_path": projection_path.as_posix()},
        repo_root=REPO_ROOT,
    )
    render_report(
        report_path,
        args.sdf.stem,
        edge_scores,
        projection,
        alignment_mode,
    )
    logging.info("wrote %s", per_edge_path)
    logging.info("wrote %s", projection_path)
    logging.info("wrote %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
