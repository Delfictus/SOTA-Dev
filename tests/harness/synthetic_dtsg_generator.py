#!/usr/bin/env python3
"""Generate synthetic Chem_Perturbed_DTSG parquet fixtures for EiV MPNN tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


DEFAULT_OUTPUT = Path("tests/harness/Chem_Perturbed_DTSG.parquet")
ASSAY_FLOOR = 0.10
TENSOR_COLUMNS = [
    "signed_te",
    "hysteresis_delta",
    "hydration_variance",
    "mechanical_load",
    "bocpd_survival",
    "kinetic_strain",
    "steering_prior",
    "aromatic_reorganization",
]
SCALAR = int | float | str | bool


def base_edges() -> list[tuple[int, int, str]]:
    return [
        (144, 145, "pocket_vector"),
        (145, 196, "pocket_vector"),
        (196, 241, "pocket_vector"),
        (241, 316, "downstream_lock"),
        (316, 317, "downstream_lock"),
        (317, 372, "downstream_lock"),
        (372, 241, "downstream_lock"),
        (241, 144, "pocket_vector"),
    ]


def tensor_row(
    *,
    graph_id: str,
    analog_id: str,
    edge_index: int,
    edge_from: int,
    edge_to: int,
    edge_class: str,
    pose_uncertainty: float,
    graph_scale: float,
    assay_value: float,
    is_left_censored: bool,
) -> dict[str, SCALAR]:
    class_scale = 1.0 if edge_class == "pocket_vector" else 0.82
    ligand_severing = 0.32 if edge_index in {3, 4} else 1.0
    signed_te = graph_scale * class_scale * ligand_severing * (1.0 + float(edge_index) * 0.035)
    return {
        "graph_id": graph_id,
        "analog_id": analog_id,
        "edge_from_residue": edge_from,
        "edge_to_residue": edge_to,
        "edge_class": edge_class,
        "signed_te": signed_te,
        "hysteresis_delta": graph_scale * (0.18 + float(edge_index) * 0.017),
        "hydration_variance": 0.09 + graph_scale * float(edge_index % 3) * 0.021,
        "mechanical_load": graph_scale * (1.7 + float(edge_index) * 0.13),
        "bocpd_survival": 0.004 + graph_scale * float(edge_index % 2) * 0.003,
        "kinetic_strain": graph_scale * (0.11 + float(edge_index) * 0.025),
        "steering_prior": 1.0 + graph_scale * float(edge_index % 4) * 0.09,
        "aromatic_reorganization": graph_scale * (0.02 + float(edge_index % 5) * 0.011),
        "u_pose": pose_uncertainty,
        "assay_floor": ASSAY_FLOOR,
        "assay_value": assay_value,
        "is_left_censored": is_left_censored,
    }


def graph_specs() -> list[tuple[str, float, float, float, bool]]:
    specs: list[tuple[str, float, float, float, bool]] = []
    graph_ids = [
        "synthetic_low_u_pose",
        "synthetic_high_u_pose",
        "synthetic_variant_02",
        "synthetic_variant_03",
        "synthetic_variant_04",
        "synthetic_variant_05",
        "synthetic_variant_06",
        "synthetic_variant_07",
        "synthetic_variant_08",
        "synthetic_variant_09",
    ]
    for graph_index, graph_id in enumerate(graph_ids):
        pose_uncertainty = 0.01 if graph_index == 0 else 2.50 if graph_index == 1 else 0.08 + float(graph_index) * 0.17
        graph_scale = 0.78 + float(graph_index) * 0.055
        is_left_censored = graph_index in {0, 4, 8}
        assay_value = ASSAY_FLOOR if is_left_censored else 0.18 + float(graph_index) * 0.045
        specs.append((graph_id, pose_uncertainty, graph_scale, assay_value, is_left_censored))
    return specs


def generate_synthetic_dtsg(output_path: Path = DEFAULT_OUTPUT) -> Path:
    rows: list[dict[str, SCALAR]] = []
    for graph_id, pose_uncertainty, graph_scale, assay_value, is_left_censored in graph_specs():
        analog_id = graph_id.replace("synthetic_", "analog_")
        for edge_index, (edge_from, edge_to, edge_class) in enumerate(base_edges()):
            rows.append(
                tensor_row(
                    graph_id=graph_id,
                    analog_id=analog_id,
                    edge_index=edge_index,
                    edge_from=edge_from,
                    edge_to=edge_to,
                    edge_class=edge_class,
                    pose_uncertainty=pose_uncertainty,
                    graph_scale=graph_scale,
                    assay_value=assay_value,
                    is_left_censored=is_left_censored,
                )
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        pl.DataFrame(rows)
        .with_columns(
            [
                pl.col("edge_from_residue").cast(pl.UInt32),
                pl.col("edge_to_residue").cast(pl.UInt32),
                *[
                    pl.col(column).cast(pl.Float64)
                    for column in [*TENSOR_COLUMNS, "u_pose", "assay_floor", "assay_value"]
                ],
                pl.col("is_left_censored").cast(pl.Boolean),
            ]
        )
        .write_parquet(output_path)
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written = generate_synthetic_dtsg(args.output)
    print(written.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
