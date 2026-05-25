#!/usr/bin/env python3
"""Translate the aligned Aleniglipron scaffold from 5VEX to 6XOX pocket space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_TOPOLOGY_DIR = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/04_TOPOLOGIES"
)
DEFAULT_SOURCE_TOPOLOGY = ORIGINAL_TOPOLOGY_DIR / "glp1r_5VEX_WT.topology.json"
DEFAULT_TARGET_TOPOLOGY = ORIGINAL_TOPOLOGY_DIR / "glp1r_6XOX_WT.topology.json"
DEFAULT_INPUT_SDF = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/conformers/ALENI-PARENT_whole_molecule_aligned.sdf"
)
DEFAULT_OUTPUT_SDF = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_a_generative/conformers/ALENI-PARENT_6XOX_translated.sdf"
)
POCKET_RESIDUES: tuple[tuple[str, int], ...] = (
    ("PHE", 143),
    ("ILE", 147),
    ("TYR", 148),
    ("TYR", 152),
)

Vector3: TypeAlias = tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-topology", type=Path, default=DEFAULT_SOURCE_TOPOLOGY)
    parser.add_argument("--target-topology", type=Path, default=DEFAULT_TARGET_TOPOLOGY)
    parser.add_argument("--input-sdf", type=Path, default=DEFAULT_INPUT_SDF)
    parser.add_argument("--output-sdf", type=Path, default=DEFAULT_OUTPUT_SDF)
    return parser.parse_args()


def as_float_list(value: object, label: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    out: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            raise TypeError(f"{label} contains a non-numeric value")
        out.append(float(item))
    return out


def as_int_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    out: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise TypeError(f"{label} contains a non-integer value")
        out.append(item)
    return out


def topology_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return cast(dict[str, object], payload)


def residue_entries(payload: dict[str, object]) -> list[dict[str, object]]:
    residues_raw = payload.get("residues")
    if not isinstance(residues_raw, list):
        raise TypeError("topology missing residues list")
    residues: list[dict[str, object]] = []
    for item in residues_raw:
        if not isinstance(item, dict):
            raise TypeError("residues contains a non-object entry")
        residues.append(cast(dict[str, object], item))
    return residues


def residue_int(residue: dict[str, object], key: str) -> int:
    value = residue.get(key)
    if not isinstance(value, int):
        raise TypeError(f"residue missing integer {key}")
    return value


def residue_name(residue: dict[str, object]) -> str:
    value = residue.get("residue_name")
    if not isinstance(value, str):
        raise TypeError("residue missing string residue_name")
    return value


def ca_coordinate(payload: dict[str, object], residue_label: tuple[str, int]) -> Vector3:
    wanted_name, wanted_number = residue_label
    residues = residue_entries(payload)
    ca_indices = as_int_list(payload.get("ca_indices"), "ca_indices")
    positions = as_float_list(payload.get("positions"), "positions")
    if len(ca_indices) != len(residues):
        raise ValueError("ca_indices length does not match residues length")

    candidate_indices: list[int] = []
    for offset in (0, -1):
        candidate_number = wanted_number + offset
        candidate_indices = [
            idx
            for idx, residue in enumerate(residues)
            if residue_name(residue) == wanted_name and residue_int(residue, "residue_id") == candidate_number
        ]
        if candidate_indices:
            break
    if len(candidate_indices) != 1:
        raise ValueError(f"could not uniquely resolve {wanted_name}{wanted_number}: {candidate_indices}")

    ca_idx = ca_indices[candidate_indices[0]]
    start = 3 * ca_idx
    if start + 2 >= len(positions):
        raise ValueError(f"CA atom index {ca_idx} is outside positions array")
    return (positions[start], positions[start + 1], positions[start + 2])


def centroid(points: list[Vector3]) -> Vector3:
    if not points:
        raise ValueError("cannot compute centroid for empty point list")
    scale = 1.0 / float(len(points))
    return (
        sum(point[0] for point in points) * scale,
        sum(point[1] for point in points) * scale,
        sum(point[2] for point in points) * scale,
    )


def sub(lhs: Vector3, rhs: Vector3) -> Vector3:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def translate_atom_line(line: str, translation: Vector3) -> str:
    x = float(line[0:10]) + translation[0]
    y = float(line[10:20]) + translation[1]
    z = float(line[20:30]) + translation[2]
    return f"{x:10.4f}{y:10.4f}{z:10.4f}{line[30:]}"


def translate_sdf(input_sdf: Path, output_sdf: Path, translation: Vector3) -> None:
    lines = input_sdf.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise ValueError(f"{input_sdf} is too short to be an SDF")
    atom_count = int(lines[3][0:3])
    out = list(lines)
    for index in range(4, 4 + atom_count):
        out[index] = translate_atom_line(out[index], translation)
    output_sdf.parent.mkdir(parents=True, exist_ok=True)
    output_sdf.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = topology_payload(args.source_topology)
    target = topology_payload(args.target_topology)
    source_points = [ca_coordinate(source, residue) for residue in POCKET_RESIDUES]
    target_points = [ca_coordinate(target, residue) for residue in POCKET_RESIDUES]
    translation = sub(centroid(target_points), centroid(source_points))
    translate_sdf(args.input_sdf, args.output_sdf, translation)
    print(
        "wrote "
        f"{args.output_sdf} translation=[{translation[0]:.6f},{translation[1]:.6f},{translation[2]:.6f}]"
    )


if __name__ == "__main__":
    main()
