#!/usr/bin/env python3
"""Validate CCNS dispatch inputs and materialize a GPU-side run manifest."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--protocol", default="ccns_5phase_validation")
    parser.add_argument("--output-dir", type=Path, default=Path(".scratch/ccns_validation_manifests"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sdf = Path(args.sdf)
    if not sdf.is_file():
        raise FileNotFoundError(f"SDF not found: {sdf}")
    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False)
    mol = supplier[0] if len(supplier) > 0 else None
    if mol is None:
        raise ValueError(f"SDF could not be parsed: {sdf}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "PRISM.ccns_validation_manifest.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "candidate_id": str(args.candidate_id),
        "sdf": str(sdf),
        "replicas": int(args.replicas),
        "protocol": str(args.protocol),
        "atom_count": int(mol.GetNumAtoms()),
        "status": "inputs_validated_pending_cluster_md",
        "phases": ["cold_hold", "ramp_up", "warm_hold", "ramp_down", "cold_return"],
    }
    output = output_dir / f"{args.candidate_id}.json"
    atomic_write_json(output, manifest)
    print(f"ccns_validation_manifest_written candidate={args.candidate_id} output={output}")
    return 0


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
