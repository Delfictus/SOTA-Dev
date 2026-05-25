#!/usr/bin/env python3
"""Render Deliverable A from geometry-joined GLP-1R durability targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from jinja2 import Environment, FileSystemLoader, StrictUndefined


N80_DIR = Path("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale")
TRACK0_DIR = Path("campaigns/glp1r_aleniglipron/track_0_manual_emulation")
TOPOLOGY_DIR = Path("campaigns/glp1r_aleniglipron/topology")
FULL_DIR = Path("campaigns/glp1r_aleniglipron/integrated_spike_events/full")
SOURCE_TOPOLOGY_DIR = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/"
    "20260518T031002Z/04_TOPOLOGIES"
)
SOURCE_OVERLAY_DIR = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/"
    "20260518T031002Z/06_ANALYSIS/glp1r_posthoc_lock_eval_latest"
)

DEFAULT_RISK_MAP = N80_DIR / "receptor_durability_risk_map.parquet"
DEFAULT_CHANNEL_SUMMARY = N80_DIR / "receptor_durability_channel_summary.parquet"
DEFAULT_TEMPLATE = Path("00_registry/templates/deliverable_a_v0.1.md.j2")
DEFAULT_MAPPING = TOPOLOGY_DIR / "residue_index_mapping_matrix.parquet"
DEFAULT_TOPOLOGY_REGISTER = FULL_DIR / "sar_steric_interface_catalog.parquet"
DEFAULT_REPORT = TRACK0_DIR / "Deliverable_A_Receptor_Durability_Audit.md"
DEFAULT_WORKBOOK = TRACK0_DIR / "Track_0_Interference_Workbook.csv"

ANALOG_COUNT = 5
STREAM_COUNT = 1_600
POCKET_TARGET_COUNT = 4
DOWNSTREAM_TARGET_COUNT = 5
CONTACT_CUTOFF_ANGSTROM = 5.0
CENTROID_PREFILTER_ANGSTROM = 18.0
MIN_SEQUENCE_SEPARATION = 3

WORKBOOK_COLUMNS = [
    "analog_id",
    "condition_id",
    "edge_label",
    "edge_from_amino_acid",
    "edge_from_sequence_number",
    "edge_to_amino_acid",
    "edge_to_sequence_number",
    "edge_class",
    "clash_assessment",
    "complement_assessment",
    "pose_confidence",
    "structural_rationale",
    "analyst_id",
]

REQUIRED_RISK_COLUMNS = {
    "condition_id",
    "edge_from_residue",
    "edge_to_residue",
    "edge_class",
    "durability_class",
    "durability_risk_score_raw",
    "durability_risk_percentile",
    "signed_te_mean",
    "variance_risk_penalty",
    "temporal_durability_risk",
    "autonomous_steering_prior_multiplier",
    "aromatic_uv_penalty",
    "thermally_destabilized_fraction",
    "thermally_activated_fraction",
    "mean_abs_mechanical_load",
    "active_load_fraction",
    "mean_survival_time_ps",
    "short_lived_regime_fraction",
    "dt_drop_count",
    "violent_dt_drop_count",
    "steering_weight_sum",
}

SOURCE_PARQUET_NAMES = [
    "receptor_durability_risk_map.parquet",
    "receptor_durability_channel_summary.parquet",
    "spike_events_snr_masked.parquet",
    "signal_grid_variance_channel.parquet",
    "mechanical_load_network.parquet",
    "bocpd_survival_regimes.parquet",
    "kinetic_strain_events.parquet",
    "autonomous_steering_tensor.parquet",
    "aromatic_reorganization_tensor.parquet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-map", type=Path, default=DEFAULT_RISK_MAP)
    parser.add_argument("--channel-summary", type=Path, default=DEFAULT_CHANNEL_SUMMARY)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--topology-register", type=Path, default=DEFAULT_TOPOLOGY_REGISTER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--source-dir", type=Path, default=N80_DIR)
    parser.add_argument("--source-topology-dir", type=Path, default=SOURCE_TOPOLOGY_DIR)
    parser.add_argument("--source-overlay-dir", type=Path, default=SOURCE_OVERLAY_DIR)
    return parser.parse_args()


def condition_ids(risk_map: Path) -> list[str]:
    return [
        str(value)
        for value in (
            pl.scan_parquet(risk_map)
            .select(pl.col("condition_id").unique().sort())
            .collect()
            .get_column("condition_id")
            .to_list()
        )
    ]


def require_columns(lf: pl.LazyFrame, required: set[str], path: Path) -> None:
    columns = set(lf.collect_schema().names())
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def source_file(directory: Path, condition_id: str, suffix: str) -> Path:
    path = directory / f"{condition_id}{suffix}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not decode to a JSON object")
    return loaded


def materialize_residue_mapping(
    conditions: list[str],
    output_path: Path,
    source_topology_dir: Path,
) -> None:
    rows: list[dict[str, int | str]] = []
    for condition_id in conditions:
        residue_map_path = source_file(source_topology_dir, condition_id, ".residue_map.json")
        residue_map = load_json(residue_map_path)
        residues = residue_map.get("residues")
        if not isinstance(residues, list):
            raise ValueError(f"{residue_map_path} does not contain a residues list")
        for residue in residues:
            topology_index = int(residue["topology_index"])
            amino_acid = str(residue["resname"]).upper()
            sequence_number = int(residue["pdb_resid"])
            rows.append(
                {
                    "condition_id": condition_id,
                    "residue_idx": topology_index,
                    "chain_id": str(residue["chain"]),
                    "amino_acid_3letter": amino_acid,
                    "biological_sequence_number": sequence_number,
                    "canonical_residue_label": f"{amino_acid}{sequence_number}",
                    "source_residue_map": residue_map_path.as_posix(),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        pl.DataFrame(rows)
        .with_columns(
            [
                pl.col("residue_idx").cast(pl.UInt32),
                pl.col("biological_sequence_number").cast(pl.Int32),
            ]
        )
        .sort(["condition_id", "residue_idx"])
        .write_parquet(output_path)
    )


def residue_heavy_atom_coordinates(topology: dict[str, Any]) -> dict[int, list[tuple[float, float, float]]]:
    positions = topology["positions"]
    elements = topology["elements"]
    residue_to_atoms = topology["residue_to_atom_indices"]
    coordinates: dict[int, list[tuple[float, float, float]]] = {}
    for residue_idx_text, atom_indices in residue_to_atoms.items():
        residue_idx = int(residue_idx_text)
        residue_coordinates: list[tuple[float, float, float]] = []
        for atom_idx in atom_indices:
            atom_index = int(atom_idx)
            if str(elements[atom_index]).upper() == "H":
                continue
            residue_coordinates.append(
                (
                    float(positions[3 * atom_index]),
                    float(positions[3 * atom_index + 1]),
                    float(positions[3 * atom_index + 2]),
                )
            )
        if residue_coordinates:
            coordinates[residue_idx] = residue_coordinates
    return coordinates


def centroid(coordinates: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    count = float(len(coordinates))
    return (
        sum(item[0] for item in coordinates) / count,
        sum(item[1] for item in coordinates) / count,
        sum(item[2] for item in coordinates) / count,
    )


def squared_distance(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return (
        (left[0] - right[0]) ** 2
        + (left[1] - right[1]) ** 2
        + (left[2] - right[2]) ** 2
    )


def minimum_heavy_atom_distance(
    left: list[tuple[float, float, float]],
    right: list[tuple[float, float, float]],
    cutoff_squared: float,
) -> float | None:
    minimum = math.inf
    for left_coordinate in left:
        for right_coordinate in right:
            distance_squared = squared_distance(left_coordinate, right_coordinate)
            if distance_squared < minimum:
                minimum = distance_squared
            if distance_squared <= cutoff_squared:
                return math.sqrt(minimum)
    return None


def edge_class_from_partition(partition: str) -> str | None:
    if "ORTHOSTERIC" in partition:
        return "pocket_vector"
    if "LOWER_TM" in partition:
        return "downstream_lock"
    return None


def materialize_topology_register(
    conditions: list[str],
    output_path: Path,
    source_topology_dir: Path,
    source_overlay_dir: Path,
) -> None:
    rows: list[dict[str, int | float | str]] = []
    cutoff_squared = CONTACT_CUTOFF_ANGSTROM * CONTACT_CUTOFF_ANGSTROM
    prefilter_squared = CENTROID_PREFILTER_ANGSTROM * CENTROID_PREFILTER_ANGSTROM
    for condition_id in conditions:
        topology_path = source_file(source_topology_dir, condition_id, ".topology.json")
        overlay_path = source_file(source_overlay_dir, condition_id, ".residue_overlay.csv")
        topology = load_json(topology_path)
        coordinates = residue_heavy_atom_coordinates(topology)
        centroids = {residue_idx: centroid(values) for residue_idx, values in coordinates.items()}
        overlay = (
            pl.read_csv(overlay_path)
            .select(
                [
                    pl.col("residue_id").cast(pl.UInt32).alias("edge_to_residue"),
                    pl.col("partition").cast(pl.Utf8),
                ]
            )
            .with_columns(
                pl.col("partition").map_elements(
                    edge_class_from_partition,
                    return_dtype=pl.Utf8,
                ).alias("edge_class")
            )
            .drop_nulls("edge_class")
        )
        class_by_to = {
            int(row["edge_to_residue"]): str(row["edge_class"])
            for row in overlay.to_dicts()
        }
        for edge_from, from_coordinates in coordinates.items():
            for edge_to, edge_class in class_by_to.items():
                if edge_to not in coordinates:
                    continue
                if edge_from == edge_to or abs(edge_from - edge_to) < MIN_SEQUENCE_SEPARATION:
                    continue
                if squared_distance(centroids[edge_from], centroids[edge_to]) > prefilter_squared:
                    continue
                contact_distance = minimum_heavy_atom_distance(
                    from_coordinates,
                    coordinates[edge_to],
                    cutoff_squared,
                )
                if contact_distance is None:
                    continue
                rows.append(
                    {
                        "condition_id": condition_id,
                        "edge_from_residue": edge_from,
                        "edge_to_residue": edge_to,
                        "edge_class": edge_class,
                        "minimum_heavy_atom_distance_angstrom": contact_distance,
                        "contact_cutoff_angstrom": CONTACT_CUTOFF_ANGSTROM,
                        "source_topology_json": topology_path.as_posix(),
                        "source_overlay_csv": overlay_path.as_posix(),
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        pl.DataFrame(rows)
        .with_columns(
            [
                pl.col("edge_from_residue").cast(pl.UInt32),
                pl.col("edge_to_residue").cast(pl.UInt32),
                pl.col("minimum_heavy_atom_distance_angstrom").cast(pl.Float64),
            ]
        )
        .sort(["condition_id", "edge_class", "edge_from_residue", "edge_to_residue"])
        .write_parquet(output_path)
    )


def mapping_lookup(mapping_path: Path, side: str) -> pl.LazyFrame:
    residue_column = f"edge_{side}_residue"
    return pl.scan_parquet(mapping_path).select(
        [
            "condition_id",
            pl.col("residue_idx").alias(residue_column),
            pl.col("amino_acid_3letter").alias(f"edge_{side}_amino_acid"),
            pl.col("biological_sequence_number").alias(f"edge_{side}_sequence_number"),
            pl.col("canonical_residue_label").alias(f"edge_{side}_label"),
        ]
    )


def selected_edge_frame(risk_map: Path, mapping_path: Path, topology_register: Path) -> pl.DataFrame:
    risk_lf = pl.scan_parquet(risk_map)
    require_columns(risk_lf, REQUIRED_RISK_COLUMNS, risk_map)
    register_lf = pl.scan_parquet(topology_register)
    require_columns(
        register_lf,
        {
            "condition_id",
            "edge_from_residue",
            "edge_to_residue",
            "edge_class",
            "minimum_heavy_atom_distance_angstrom",
        },
        topology_register,
    )
    critical = (
        risk_lf.filter(pl.col("durability_class") == "critical_durability_risk")
        .with_columns(
            [
                pl.col("edge_from_residue").cast(pl.UInt32),
                pl.col("edge_to_residue").cast(pl.UInt32),
            ]
        )
        .join(
            register_lf.select(
                [
                    "condition_id",
                    "edge_from_residue",
                    "edge_to_residue",
                    "edge_class",
                ]
            ).unique(),
            on=["condition_id", "edge_from_residue", "edge_to_residue", "edge_class"],
            how="inner",
        )
        .join(mapping_lookup(mapping_path, "from"), on=["condition_id", "edge_from_residue"], how="inner")
        .join(mapping_lookup(mapping_path, "to"), on=["condition_id", "edge_to_residue"], how="inner")
        .with_columns(
            [
                (
                    pl.col("edge_from_label")
                    + pl.lit(" -> ")
                    + pl.col("edge_to_label")
                ).alias("edge_label"),
                (
                    pl.col("condition_id")
                    + pl.lit(":")
                    + pl.col("edge_from_label")
                    + pl.lit("->")
                    + pl.col("edge_to_label")
                    + pl.lit(":")
                    + pl.col("edge_class")
                ).alias("edge_id"),
            ]
        )
    )
    selection_columns = [
        "condition_id",
        "edge_from_residue",
        "edge_to_residue",
        "edge_from_amino_acid",
        "edge_from_sequence_number",
        "edge_from_label",
        "edge_to_amino_acid",
        "edge_to_sequence_number",
        "edge_to_label",
        "edge_label",
        "edge_id",
        "edge_class",
        "minimum_heavy_atom_distance_angstrom",
        "durability_risk_score_raw",
        "durability_risk_percentile",
        "signed_te_mean",
        "variance_risk_penalty",
        "temporal_durability_risk",
        "autonomous_steering_prior_multiplier",
        "aromatic_uv_penalty",
        "thermally_destabilized_fraction",
        "thermally_activated_fraction",
        "mean_abs_mechanical_load",
        "active_load_fraction",
        "mean_survival_time_ps",
        "short_lived_regime_fraction",
        "dt_drop_count",
        "violent_dt_drop_count",
        "steering_weight_sum",
    ]
    sort_columns = [
        "durability_risk_score_raw",
        "minimum_heavy_atom_distance_angstrom",
        "edge_from_sequence_number",
        "edge_to_sequence_number",
    ]
    pocket = (
        critical.filter(pl.col("edge_class") == "pocket_vector")
        .sort(sort_columns, descending=[True, False, False, False])
        .head(POCKET_TARGET_COUNT)
        .select(selection_columns)
        .collect()
    )
    downstream = (
        critical.filter(pl.col("edge_class") == "downstream_lock")
        .sort(sort_columns, descending=[True, False, False, False])
        .head(DOWNSTREAM_TARGET_COUNT)
        .select(selection_columns)
        .collect()
    )
    if pocket.height != POCKET_TARGET_COUNT:
        raise ValueError(f"expected {POCKET_TARGET_COUNT} pocket_vector targets, found {pocket.height}")
    if downstream.height != DOWNSTREAM_TARGET_COUNT:
        raise ValueError(
            f"expected {DOWNSTREAM_TARGET_COUNT} downstream_lock targets, found {downstream.height}"
        )
    return pl.concat([pocket, downstream], how="diagonal_relaxed")


def edge_records(selected: pl.DataFrame) -> list[dict[str, str | int | float]]:
    records: list[dict[str, str | int | float]] = selected.to_dicts()
    class_offsets = {"pocket_vector": 0, "downstream_lock": POCKET_TARGET_COUNT}
    for edge_class in ("pocket_vector", "downstream_lock"):
        class_rows = [row for row in records if row["edge_class"] == edge_class]
        for class_rank, row in enumerate(class_rows, start=1):
            row["class_rank"] = class_rank
            row["selection_rank"] = class_offsets[edge_class] + class_rank
    return sorted(records, key=lambda row: int(row["selection_rank"]))


def workbook_rows(edges: list[dict[str, str | int | float]]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for analog_idx in range(1, ANALOG_COUNT + 1):
        analog_id = f"placeholder_analog_{analog_idx:02d}"
        for edge in edges:
            rows.append(
                {
                    "analog_id": analog_id,
                    "condition_id": str(edge["condition_id"]),
                    "edge_label": str(edge["edge_label"]),
                    "edge_from_amino_acid": str(edge["edge_from_amino_acid"]),
                    "edge_from_sequence_number": int(edge["edge_from_sequence_number"]),
                    "edge_to_amino_acid": str(edge["edge_to_amino_acid"]),
                    "edge_to_sequence_number": int(edge["edge_to_sequence_number"]),
                    "edge_class": str(edge["edge_class"]),
                    "clash_assessment": "none",
                    "complement_assessment": "none",
                    "pose_confidence": "unresolvable",
                    "structural_rationale": (
                        "Pending expert-in-the-loop Track 0 scoring for "
                        f"{edge['edge_label']} generated by strict SAR topology join."
                    ),
                    "analyst_id": "pending_track0_analyst",
                }
            )
    return rows


def write_workbook(path: Path, edges: list[dict[str, str | int | float]]) -> int:
    rows = workbook_rows(edges)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).select(WORKBOOK_COLUMNS).write_csv(path)
    return len(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lineage_hashes(
    source_dir: Path,
    risk_map: Path,
    channel_summary: Path,
    mapping_path: Path,
    topology_register: Path,
) -> list[dict[str, str]]:
    paths = [source_dir / name for name in SOURCE_PARQUET_NAMES]
    paths.extend([mapping_path, topology_register])
    resolved = []
    for path in paths:
        if path.name == risk_map.name:
            path = risk_map
        elif path.name == channel_summary.name:
            path = channel_summary
        if not path.exists():
            raise FileNotFoundError(path)
        resolved.append(path)
    return [{"path": path.as_posix(), "sha256": sha256_file(path)} for path in resolved]


def channel_summary_records(path: Path) -> list[dict[str, str | int | float]]:
    return (
        pl.scan_parquet(path)
        .sort(["condition_id", "durability_class"])
        .collect()
        .to_dicts()
    )


def mechanically_pruned_count(path: Path) -> int:
    return int(
        pl.scan_parquet(path)
        .filter(pl.col("mechanically_pruned"))
        .select(pl.len().alias("n"))
        .collect()
        .item()
    )


def render_report(
    template_path: Path,
    report_path: Path,
    workbook_path: Path,
    edges: list[dict[str, str | int | float]],
    channel_summaries: list[dict[str, str | int | float]],
    workbook_row_count: int,
    lineage: list[dict[str, str]],
    pruned_count: int,
) -> None:
    environment = Environment(
        loader=FileSystemLoader(template_path.parent.as_posix()),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    template = environment.get_template(template_path.name)
    rendered = template.render(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        stream_count=STREAM_COUNT,
        channel_summary_count=len(channel_summaries),
        mechanically_pruned_count=pruned_count,
        channel_summaries=channel_summaries,
        critical_edges=edges,
        edge_selection_policy=(
            "critical_durability_risk rows inner-joined to "
            "sar_steric_interface_catalog.parquet on condition_id, edge_from_residue, "
            "edge_to_residue, and edge_class; edge scores come from the edge-level evaluator, "
            "not from node-to-edge broadcasting."
        ),
        analog_count=ANALOG_COUNT,
        workbook_row_count=workbook_row_count,
        workbook_path=workbook_path.as_posix(),
        lineage_hashes=lineage,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")


def main() -> None:
    args = parse_args()
    conditions = condition_ids(args.risk_map)
    materialize_residue_mapping(conditions, args.mapping, args.source_topology_dir)
    materialize_topology_register(
        conditions,
        args.topology_register,
        args.source_topology_dir,
        args.source_overlay_dir,
    )
    selected = selected_edge_frame(args.risk_map, args.mapping, args.topology_register)
    edges = edge_records(selected)
    workbook_row_count = write_workbook(args.workbook, edges)
    summaries = channel_summary_records(args.channel_summary)
    pruned_count = mechanically_pruned_count(args.risk_map)
    lineage = lineage_hashes(
        args.source_dir,
        args.risk_map,
        args.channel_summary,
        args.mapping,
        args.topology_register,
    )
    render_report(
        args.template,
        args.report,
        args.workbook,
        edges,
        summaries,
        workbook_row_count,
        lineage,
        pruned_count,
    )
    print(f"report={args.report}")
    print(f"workbook={args.workbook}")
    print(f"mapping={args.mapping}")
    print(f"topology_register={args.topology_register}")
    print(f"workbook_rows={workbook_row_count}")
    for edge in edges:
        print(
            f"{edge['selection_rank']}\t{edge['edge_class']}\t{edge['edge_label']}\t"
            f"{float(edge['durability_risk_score_raw']):.6f}"
        )


if __name__ == "__main__":
    main()
