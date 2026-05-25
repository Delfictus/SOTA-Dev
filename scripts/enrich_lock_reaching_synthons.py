#!/usr/bin/env python3
"""Tag synthons with static lock-reaching pharmacophore priors."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from prism_dstw.pharmacophore.bias_pharmacophore import max_atomic_span, coordinates_from_json


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_INPUT = TRACK_A / "enamine_115k_synthons_3d.parquet"
DEFAULT_OUTPUT = TRACK_A / "enamine_115k_synthons_3d_lock_reaching.parquet"
DEFAULT_REPORT = TRACK_A / "lock_reaching_synthon_enrichment_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pl.read_parquet(Path(args.input))
    if args.limit is not None:
        frame = frame.head(int(args.limit))
    rows = [tag_row(row) for row in frame.iter_rows(named=True)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    pl.DataFrame(rows).write_parquet(tmp_output)
    tmp_output.replace(output)
    tagged_count = sum(1 for row in rows if bool(row["lock_reaching_candidate"]))
    report = {
        "schema_version": "PRISM.lock_reaching_synthon_enrichment.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(Path(args.input)),
        "output": str(output),
        "rows": len(rows),
        "lock_reaching_candidate_count": tagged_count,
        "criteria": {
            "max_atomic_span_angstrom": ">8",
            "aromatic_ring_count": ">=1",
            "rotatable_bonds": "<=4",
            "functional_group_count": ">=2 when available",
        },
    }
    atomic_write_json(Path(args.report), report)
    print(f"lock_reaching_synthons_tagged total={len(rows)} tagged={tagged_count} output={output}")
    return 0


def tag_row(row: dict[str, Any]) -> dict[str, Any]:
    smiles = str(row.get("canonical_smiles", row.get("smiles", "")))
    mol = Chem.MolFromSmiles(smiles)
    aromatic_count = 0
    rotatable = 99
    if mol is not None:
        mol_any = cast(Any, mol)
        aromatic_count = len([atom for atom in mol_any.GetAtoms() if bool(atom.GetIsAromatic())])
        rotatable = int(cast(Any, rdMolDescriptors).CalcNumRotatableBonds(mol))
    coordinates = coordinates_from_json(row.get("coordinates_json") if isinstance(row.get("coordinates_json"), str) else None)
    span = max_atomic_span(coordinates)
    functional_groups = int_value(
        row.get("compatible_functional_group_count", row.get("functional_group_count", 2))
    )
    lock_reaching = span > 8.0 and aromatic_count >= 1 and rotatable <= 4 and functional_groups >= 2
    tagged = dict(row)
    tagged.update(
        {
            "max_atomic_span_A": span,
            "aromatic_atom_count": aromatic_count,
            "rotatable_bonds": rotatable,
            "compatible_functional_group_count": functional_groups,
            "lock_reaching_candidate": lock_reaching,
            "lock_reaching_synthon_boost": 2.0 if lock_reaching else 1.0,
        }
    )
    return tagged


def int_value(value: object) -> int:
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(float(value))
    return 0


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
