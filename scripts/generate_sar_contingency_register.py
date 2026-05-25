#!/usr/bin/env python3
"""Generate phase-aware SAR rules by thermodynamic ray-casting ligand exit vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import polars as pl
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import write_provenance_parquet


CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
TRACK0_DIR = CAMPAIGN_DIR / "track_0_manual_emulation"
DEFAULT_SDF = TRACK0_DIR / "conformers/ALENI-PARENT_whole_molecule_aligned.sdf"
DEFAULT_OUTPUT = TRACK0_DIR / "sar_contingency_register.parquet"
EPSILON = 1.0e-9

Coordinate: TypeAlias = tuple[float, float, float]


@dataclass(frozen=True)
class GridGeometry:
    condition_id: str
    grid_dim: int
    origin: Coordinate
    spacing: float


@dataclass(frozen=True)
class EdgeContext:
    edge_id: str
    condition_id: str
    edge_label: str
    edge_class: str
    edge_from_residue: int
    edge_to_residue: int
    signed_te_mean: float
    edge_u_pose: float
    cooperative_cluster_id: int
    cluster_confidence: float


@dataclass(frozen=True)
class VoxelField:
    pi_clash: float
    pi_complement: float


@dataclass(frozen=True)
class RayDirection:
    ray_id: str
    dx: float
    dy: float
    dz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steric-env", type=Path, default=TRACK0_DIR / "interface_steric_environment.parquet")
    parser.add_argument("--sdf", type=Path, default=DEFAULT_SDF)
    parser.add_argument("--grid-mapping", type=Path, default=TRACK0_DIR / "grid_coordinate_mapping.json")
    parser.add_argument("--break-clusters", type=Path, default=N80_DIR / "probabilistic_break_clusters.parquet")
    parser.add_argument("--edge-interference", type=Path, default=TRACK0_DIR / "layer1_whole_molecule/per_edge_interference.parquet")
    parser.add_argument("--shear-stress", type=Path, default=N80_DIR / "shear_stress_field.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ray-length-A", type=float, default=3.0)
    parser.add_argument("--ray-step-A", type=float, default=0.5)
    parser.add_argument("--epistemic-quantile", type=float, default=0.75)
    parser.add_argument("--permitted-score-threshold", type=float, default=0.5)
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


def sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not decode to an object")
    return cast(dict[str, object], loaded)


def json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def coordinate_from_json(value: object, label: str) -> Coordinate:
    raw = json_list(value, label)
    if len(raw) != 3:
        raise ValueError(f"{label} must contain three values")
    return (as_float(raw[0], f"{label}[0]"), as_float(raw[1], f"{label}[1]"), as_float(raw[2], f"{label}[2]"))


def load_geometries(path: Path) -> dict[str, GridGeometry]:
    payload = load_json(path)
    conditions = json_object(payload.get("conditions"), f"{path}:conditions")
    out: dict[str, GridGeometry] = {}
    for condition_id, raw_geometry in conditions.items():
        geometry = json_object(raw_geometry, f"{path}:conditions[{condition_id}]")
        out[str(condition_id)] = GridGeometry(
            condition_id=str(condition_id),
            grid_dim=as_int(geometry["grid_dim"], "grid_dim"),
            origin=coordinate_from_json(geometry.get("origin_xyz_angstrom"), "origin_xyz_angstrom"),
            spacing=as_float(geometry["spacing_angstrom"], "spacing_angstrom"),
        )
    return out


def coordinate_to_voxel(coord: Coordinate, geometry: GridGeometry) -> int | None:
    ix = math.trunc((coord[0] - geometry.origin[0]) / geometry.spacing)
    iy = math.trunc((coord[1] - geometry.origin[1]) / geometry.spacing)
    iz = math.trunc((coord[2] - geometry.origin[2]) / geometry.spacing)
    if ix < 0 or iy < 0 or iz < 0 or ix >= geometry.grid_dim or iy >= geometry.grid_dim or iz >= geometry.grid_dim:
        return None
    return iz * geometry.grid_dim * geometry.grid_dim + iy * geometry.grid_dim + ix


def condition_family(condition_id: str) -> str:
    for marker in ("6LN2", "6XOX", "5VEX", "6X1A"):
        if marker in condition_id:
            return f"glp1r_{marker}"
    return condition_id


def normalize(vector: Coordinate) -> Coordinate | None:
    norm = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if norm <= EPSILON:
        return None
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def add_direction(directions: list[RayDirection], label: str, vector: Coordinate) -> None:
    unit = normalize(vector)
    if unit is None:
        return
    for current in directions:
        dot = current.dx * unit[0] + current.dy * unit[1] + current.dz * unit[2]
        if abs(dot) > 0.98:
            return
    directions.append(RayDirection(ray_id=label, dx=unit[0], dy=unit[1], dz=unit[2]))


def load_sdf_molecules(sdf_path: Path) -> list[Any]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [item for item in supplier if item is not None]
    if not mols:
        raise ValueError(f"{sdf_path} did not contain valid conformers")
    return mols


def heavy_atom_indices(mol: Any) -> list[int]:
    return [int(atom.GetIdx()) for atom in cast(Any, mol).GetAtoms() if int(atom.GetAtomicNum()) > 1]


def atom_coordinate(mol: Any, atom_idx: int) -> Coordinate:
    pos = mol.GetConformer().GetAtomPosition(atom_idx)
    return (float(pos.x), float(pos.y), float(pos.z))


def ligand_centroid(mol: Any, atom_indices: list[int]) -> Coordinate:
    coords = [atom_coordinate(mol, atom_idx) for atom_idx in atom_indices]
    return (
        sum(coord[0] for coord in coords) / float(len(coords)),
        sum(coord[1] for coord in coords) / float(len(coords)),
        sum(coord[2] for coord in coords) / float(len(coords)),
    )


def ray_directions_for_atom(mol: Any, atom_idx: int, centroid: Coordinate) -> list[RayDirection]:
    atom = mol.GetAtomWithIdx(atom_idx)
    atom_coord = atom_coordinate(mol, atom_idx)
    directions: list[RayDirection] = []
    neighbor_coords = [
        atom_coordinate(mol, int(neighbor.GetIdx()))
        for neighbor in atom.GetNeighbors()
        if int(neighbor.GetAtomicNum()) > 1
    ]
    if neighbor_coords:
        neighbor_centroid = (
            sum(coord[0] for coord in neighbor_coords) / float(len(neighbor_coords)),
            sum(coord[1] for coord in neighbor_coords) / float(len(neighbor_coords)),
            sum(coord[2] for coord in neighbor_coords) / float(len(neighbor_coords)),
        )
        add_direction(
            directions,
            "bond_opposite_exit",
            (
                atom_coord[0] - neighbor_centroid[0],
                atom_coord[1] - neighbor_centroid[1],
                atom_coord[2] - neighbor_centroid[2],
            ),
        )
    radial = (atom_coord[0] - centroid[0], atom_coord[1] - centroid[1], atom_coord[2] - centroid[2])
    add_direction(directions, "radial_surface_exit", radial)
    if neighbor_coords:
        base = normalize((directions[0].dx, directions[0].dy, directions[0].dz))
        radial_unit = normalize(radial)
        if base is not None and radial_unit is not None:
            add_direction(
                directions,
                "blended_exit",
                (base[0] + radial_unit[0], base[1] + radial_unit[1], base[2] + radial_unit[2]),
            )
    return directions


def normalization_constants(steric_env: Path) -> tuple[float, float]:
    frame = pl.scan_parquet(steric_env).filter(pl.col("variance_class") != "void")
    cold = frame.select(pl.col("hit_count_cold_mean").quantile(0.95).alias("q")).collect().item()
    delta = frame.select(pl.col("hit_count_delta").abs().quantile(0.95).alias("q")).collect().item()
    return max(as_float(cold, "cold_q95"), 1.0), max(as_float(delta, "delta_q95"), 1.0)


def voxel_field_lookup(steric_env: Path, edge_interference: Path) -> dict[tuple[str, int], VoxelField]:
    cold_q95, delta_q95 = normalization_constants(steric_env)
    signed_te = {
        str(row["edge_id"]): as_float(row["signed_te_mean"], "signed_te_mean")
        for row in pl.read_parquet(edge_interference).select(["edge_id", "signed_te_mean"]).to_dicts()
    }
    rows = pl.read_parquet(steric_env).select(
        ["edge_id", "voxel_idx", "variance_class", "hit_count_cold_mean", "hit_count_delta"]
    )
    lookup: dict[tuple[str, int], VoxelField] = {}
    for row in rows.iter_rows(named=True):
        edge_id = str(row["edge_id"])
        variance_class = str(row["variance_class"])
        cold_norm = as_float(row["hit_count_cold_mean"], "hit_count_cold_mean") / cold_q95
        delta_norm = abs(as_float(row["hit_count_delta"], "hit_count_delta")) / delta_q95
        pi_clash = 0.0
        pi_complement = 0.0
        if variance_class == "stable_occupied":
            pi_clash = cold_norm
        elif variance_class == "thermally_activated":
            pi_complement = delta_norm
        elif variance_class == "thermally_destabilized":
            if signed_te.get(edge_id, 0.0) < 0.0:
                pi_clash = 0.5 * delta_norm
            else:
                pi_complement = 0.5 * delta_norm
        lookup[(edge_id, as_int(row["voxel_idx"], "voxel_idx"))] = VoxelField(pi_clash=pi_clash, pi_complement=pi_complement)
    return lookup


def shear_lookup(path: Path) -> tuple[dict[tuple[str, int], float], dict[str, float]]:
    frame = pl.read_parquet(path).select(["condition_id", "voxel_idx", "shear_stress"])
    thresholds = (
        frame.lazy()
        .with_columns(pl.col("shear_stress").abs().alias("shear_abs"))
        .group_by("condition_id")
        .agg(pl.col("shear_abs").quantile(0.90).alias("p90"))
        .collect()
    )
    threshold_lookup = {str(row["condition_id"]): max(as_float(row["p90"], "p90"), EPSILON) for row in thresholds.to_dicts()}
    values = {
        (str(row["condition_id"]), as_int(row["voxel_idx"], "voxel_idx")): abs(as_float(row["shear_stress"], "shear_stress"))
        for row in frame.to_dicts()
    }
    return values, threshold_lookup


def edge_contexts(
    phase_edges: Path,
    edge_interference: Path,
    break_clusters: Path,
) -> list[EdgeContext]:
    edge_pose = {
        str(row["edge_id"]): {
            "signed_te_mean": as_float(row["signed_te_mean"], "signed_te_mean"),
            "edge_u_pose": as_float(row["U_pose"], "U_pose"),
        }
        for row in pl.read_parquet(edge_interference).select(["edge_id", "signed_te_mean", "U_pose"]).to_dicts()
    }
    cluster_rows = (
        pl.scan_parquet(break_clusters)
        .select(["condition_id", "primary_residue_idx", "cluster_id", "cluster_confidence"])
        .group_by(["condition_id", "primary_residue_idx"])
        .agg([pl.col("cluster_id").first(), pl.col("cluster_confidence").max()])
        .collect()
    )
    cluster_lookup = {
        (str(row["condition_id"]), as_int(row["primary_residue_idx"], "primary_residue_idx")): (
            as_int(row["cluster_id"], "cluster_id"),
            as_float(row["cluster_confidence"], "cluster_confidence"),
        )
        for row in cluster_rows.to_dicts()
    }
    contexts: list[EdgeContext] = []
    for row in pl.read_parquet(phase_edges).select(
        ["edge_id", "condition_id", "edge_label", "edge_class", "edge_from_residue", "edge_to_residue"]
    ).to_dicts():
        edge_id = str(row["edge_id"])
        from_key = (str(row["condition_id"]), as_int(row["edge_from_residue"], "edge_from_residue"))
        to_key = (str(row["condition_id"]), as_int(row["edge_to_residue"], "edge_to_residue"))
        from_cluster = cluster_lookup.get(from_key, (-1, 0.0))
        to_cluster = cluster_lookup.get(to_key, (-1, 0.0))
        cluster = from_cluster if from_cluster[1] >= to_cluster[1] else to_cluster
        pose = edge_pose.get(edge_id, {"signed_te_mean": 0.0, "edge_u_pose": 0.0})
        contexts.append(
            EdgeContext(
                edge_id=edge_id,
                condition_id=str(row["condition_id"]),
                edge_label=str(row["edge_label"]),
                edge_class=str(row["edge_class"]),
                edge_from_residue=as_int(row["edge_from_residue"], "edge_from_residue"),
                edge_to_residue=as_int(row["edge_to_residue"], "edge_to_residue"),
                signed_te_mean=as_float(pose["signed_te_mean"], "signed_te_mean"),
                edge_u_pose=as_float(pose["edge_u_pose"], "edge_u_pose"),
                cooperative_cluster_id=cluster[0],
                cluster_confidence=cluster[1],
            )
        )
    return contexts


def population_variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / float(len(values))
    return sum((value - mean) * (value - mean) for value in values) / float(len(values))


def atom_edge_pose_uncertainty(
    mols: list[Any],
    atom_idx: int,
    context: EdgeContext,
    geometry: GridGeometry,
    fields: dict[tuple[str, int], VoxelField],
) -> float:
    contributions: list[float] = []
    for mol in mols:
        voxel_idx = coordinate_to_voxel(atom_coordinate(mol, atom_idx), geometry)
        field = fields.get((context.edge_id, voxel_idx), VoxelField(0.0, 0.0)) if voxel_idx is not None else VoxelField(0.0, 0.0)
        contributions.append(field.pi_clash - field.pi_complement)
    return population_variance(contributions)


def ray_points(origin: Coordinate, direction: RayDirection, ray_length_a: float, ray_step_a: float) -> list[Coordinate]:
    points: list[Coordinate] = []
    steps = max(int(math.ceil(ray_length_a / ray_step_a)), 1)
    for step in range(1, steps + 1):
        distance = min(float(step) * ray_step_a, ray_length_a)
        points.append(
            (
                origin[0] + direction.dx * distance,
                origin[1] + direction.dy * distance,
                origin[2] + direction.dz * distance,
            )
        )
    return points


def integrate_ray(
    context: EdgeContext,
    geometry: GridGeometry,
    origin: Coordinate,
    direction: RayDirection,
    fields: dict[tuple[str, int], VoxelField],
    shear_values: dict[tuple[str, int], float],
    shear_thresholds: dict[str, float],
    ray_length_a: float,
    ray_step_a: float,
) -> tuple[float, float, float, float, int]:
    complement = 0.0
    clash = 0.0
    shear_normalized = 0.0
    shear_raw = 0.0
    sampled = 0
    threshold = shear_thresholds.get(context.condition_id, 1.0)
    for point in ray_points(origin, direction, ray_length_a, ray_step_a):
        voxel_idx = coordinate_to_voxel(point, geometry)
        if voxel_idx is None:
            continue
        sampled += 1
        field = fields.get((context.edge_id, voxel_idx), VoxelField(0.0, 0.0))
        complement += field.pi_complement
        clash += field.pi_clash
        shear = shear_values.get((context.condition_id, voxel_idx), 0.0)
        shear_raw += shear
        shear_normalized += shear / threshold
    return complement, clash, shear_raw, shear_normalized, sampled


def classify_ray(score: float, clash: float, shear_normalized: float, sampled_points: int, permitted_threshold: float) -> str:
    if shear_normalized >= float(max(sampled_points, 1)):
        return "shear_fracture_vector"
    if clash > 0.0:
        return "prohibited_rigidification_zone"
    if score > permitted_threshold:
        return "permitted_growth_vector"
    return "neutral_low_reward_vector"


def register_frame(
    sdf_path: Path,
    grid_mapping: Path,
    phase_edges: Path,
    steric_env: Path,
    edge_interference: Path,
    break_clusters: Path,
    shear_stress: Path,
    ray_length_a: float,
    ray_step_a: float,
    epistemic_quantile: float,
    permitted_threshold: float,
) -> pl.DataFrame:
    mols = load_sdf_molecules(sdf_path)
    reference_mol = mols[0]
    atom_indices = heavy_atom_indices(reference_mol)
    centroid = ligand_centroid(reference_mol, atom_indices)
    geometries = load_geometries(grid_mapping)
    contexts = edge_contexts(phase_edges, edge_interference, break_clusters)
    fields = voxel_field_lookup(steric_env, edge_interference)
    shear_values, shear_thresholds = shear_lookup(shear_stress)
    atom_uncertainties: list[float] = []
    uncertainty_lookup: dict[tuple[int, str], float] = {}
    for context in contexts:
        geometry = geometries[context.condition_id]
        for atom_idx in atom_indices:
            u_pose = atom_edge_pose_uncertainty(mols, atom_idx, context, geometry, fields)
            uncertainty_lookup[(atom_idx, context.edge_id)] = u_pose
            atom_uncertainties.append(u_pose)
    sorted_uncertainties = sorted(atom_uncertainties)
    threshold_idx = min(max(int(round(epistemic_quantile * float(len(sorted_uncertainties) - 1))), 0), len(sorted_uncertainties) - 1)
    epistemic_threshold = max(sorted_uncertainties[threshold_idx], EPSILON)
    rows: list[dict[str, object]] = []
    rule_idx = 1
    for context in contexts:
        geometry = geometries[context.condition_id]
        for atom_idx in atom_indices:
            atom = reference_mol.GetAtomWithIdx(atom_idx)
            origin = atom_coordinate(reference_mol, atom_idx)
            for direction in ray_directions_for_atom(reference_mol, atom_idx, centroid):
                atom_u_pose = uncertainty_lookup[(atom_idx, context.edge_id)]
                if atom_u_pose > epistemic_threshold:
                    action_class = "epistemic_mirage"
                    complement = 0.0
                    clash = 0.0
                    shear_raw = 0.0
                    shear_normalized = 0.0
                    sampled = 0
                    score = 0.0
                    directive = (
                        f"Defer expansion at Vector {rule_idx}; atom-level pose uncertainty exceeds the ray-casting "
                        "threshold and requires scaffold rigidification before growth-vector optimization."
                    )
                else:
                    complement, clash, shear_raw, shear_normalized, sampled = integrate_ray(
                        context,
                        geometry,
                        origin,
                        direction,
                        fields,
                        shear_values,
                        shear_thresholds,
                        ray_length_a,
                        ray_step_a,
                    )
                    score = complement - clash - shear_normalized
                    action_class = classify_ray(score, clash, shear_normalized, sampled, permitted_threshold)
                    if action_class == "permitted_growth_vector":
                        directive = (
                            f"Elaborate the scaffold along Vector {rule_idx}; the ray integrates positive complement "
                            "with no clash penalty and low shear penalty."
                        )
                    elif action_class == "shear_fracture_vector":
                        directive = (
                            f"Avoid expansion along Vector {rule_idx}; the ray integrates high structural deformation "
                            "gradient and may reduce ligand residence stability."
                        )
                    elif action_class == "prohibited_rigidification_zone":
                        directive = (
                            f"Halt rigidifying expansion at Vector {rule_idx}; the ray intersects a positive clash field "
                            "near a receptor lock interface."
                        )
                    else:
                        directive = (
                            f"No priority expansion at Vector {rule_idx}; the ray does not integrate enough complement "
                            "to justify added scaffold complexity."
                        )
                rows.append(
                    {
                        "sar_rule_id": f"SAR-{rule_idx:03d}",
                        "growth_vector_id": f"Vector {rule_idx}",
                        "action_class": action_class,
                        "condition_family": condition_family(context.condition_id),
                        "condition_id": context.condition_id,
                        "edge_id": context.edge_id,
                        "edge_label": context.edge_label,
                        "edge_class": context.edge_class,
                        "ligand_atom_idx": atom_idx,
                        "ligand_atom_symbol": str(atom.GetSymbol()),
                        "ray_id": direction.ray_id,
                        "ray_dx": direction.dx,
                        "ray_dy": direction.dy,
                        "ray_dz": direction.dz,
                        "cooperative_cluster_id": context.cooperative_cluster_id,
                        "cluster_confidence": context.cluster_confidence,
                        "atom_edge_u_pose": atom_u_pose,
                        "epistemic_u_pose_threshold": epistemic_threshold,
                        "pi_complement_integral": complement,
                        "pi_clash_integral": clash,
                        "shear_stress_integral": shear_raw,
                        "shear_stress_normalized_integral": shear_normalized,
                        "vector_score": score,
                        "sampled_points": sampled,
                        "medchem_directive": directive,
                    }
                )
                rule_idx += 1
    frame = pl.DataFrame(rows)
    support = (
        frame.lazy()
        .filter(pl.col("action_class") == "permitted_growth_vector")
        .group_by(["ligand_atom_idx", "ray_id"])
        .agg(pl.col("condition_family").n_unique().alias("variant_family_support_count"))
        .collect()
    )
    target_support = max(
        int(frame.select(pl.col("condition_family").n_unique().alias("n")).item()),
        1,
    )
    return (
        frame.lazy()
        .join(support.lazy(), on=["ligand_atom_idx", "ray_id"], how="left")
        .with_columns(pl.col("variant_family_support_count").fill_null(0))
        .with_columns(
            (
                (pl.col("action_class") == "permitted_growth_vector")
                & (pl.col("variant_family_support_count") >= min(target_support, 2))
            ).alias("universal_growth_vector")
        )
        .with_columns(
            pl.when(pl.col("universal_growth_vector"))
            .then(pl.lit("universal_growth_vector"))
            .otherwise(pl.col("action_class"))
            .alias("action_class")
        )
        .sort(["action_class", "vector_score"], descending=[False, True])
        .collect()
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    phase_edges = N80_DIR / "phase_manifold_edge_validation.parquet"
    frame = register_frame(
        Path(args.sdf),
        Path(args.grid_mapping),
        phase_edges,
        Path(args.steric_env),
        Path(args.edge_interference),
        Path(args.break_clusters),
        Path(args.shear_stress),
        float(args.ray_length_A),
        float(args.ray_step_A),
        float(args.epistemic_quantile),
        float(args.permitted_score_threshold),
    )
    write_provenance_parquet(
        frame,
        output,
        producer_script=Path(__file__),
        source_parquets=[
            phase_edges,
            Path(args.steric_env),
            Path(args.edge_interference),
            Path(args.break_clusters),
            Path(args.shear_stress),
        ],
        schema_version="sar_contingency_register.v2",
        pipeline_stage="sar_contingency_register",
        partition_keys=["action_class", "condition_id"],
        ledger_parameters={
            "sdf_path": Path(args.sdf).relative_to(REPO_ROOT).as_posix(),
            "sdf_sha256": sha256_path(Path(args.sdf)),
            "ray_length_A": float(args.ray_length_A),
            "ray_step_A": float(args.ray_step_A),
            "score": "sum(Pi_complement)-sum(Pi_clash)-sum(ShearStress/p90_condition_shear)",
            "epistemic_gate": "atom-edge U_pose from conformer contribution variance over Layer 1 field lookup",
        },
        ledger_output_value={"rows": frame.height, "output_path": output.as_posix()},
        repo_root=REPO_ROOT,
    )
    sys.stdout.write(f"wrote {output} rows={frame.height}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
