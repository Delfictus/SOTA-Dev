#!/usr/bin/env python3
"""Validate CCNS dispatch inputs and materialize a GPU-side run manifest."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rdkit import Chem

from lib.prism_runtime import resolve_prism_scratch_root

SAFE_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--protocol", default="ccns_5phase_validation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=resolve_prism_scratch_root() / "ccns_validation_manifests",
    )
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_id = str(args.candidate_id)
    if not SAFE_CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError(f"unsafe candidate_id for validation manifest: {candidate_id}")
    sdf = Path(args.sdf)
    if not sdf.is_file():
        raise FileNotFoundError(f"SDF not found: {sdf}")
    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False)
    mol = supplier[0] if len(supplier) > 0 else None
    if mol is None:
        raise ValueError(f"SDF could not be parsed: {sdf}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = Path(args.raw_output_dir) if args.raw_output_dir is not None else output_dir / "ccns_raw" / candidate_id
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "PRISM.ccns_validation_manifest.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "candidate_id": candidate_id,
        "sdf": str(sdf),
        "replicas": int(args.replicas),
        "protocol": str(args.protocol),
        "raw_output_dir": str(raw_output_dir),
        "atom_count": int(mol.GetNumAtoms()),
        "status": "inputs_validated_candidate_ccns_raw_required",
        "required_raw_artifacts": ["signal_grid.bin", "warp_matrix.bin"],
        "phases": ["cold_hold", "ramp_up", "warm_hold", "ramp_down", "cold_return"],
    }
    output = output_dir / f"{candidate_id}.json"
    atomic_write_json(output, manifest)
    print(f"ccns_validation_manifest_written candidate={candidate_id} output={output}")
    return 0


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
