#!/usr/bin/env python3
"""Generate SLURM validation dispatch scripts for lock-positive candidates."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_PROFILES = TRACK_A / "gflownet_top_50_tripartite_profiles.parquet"
DEFAULT_OUTPUT = TRACK_A / "gpu_dispatch"
DEFAULT_MANIFEST = TRACK_A / "gpu_dispatch_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replicas", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = pl.read_parquet(Path(args.profiles)).filter(pl.col("lock_geometry_score") > 0.0)
    output_dir = Path(args.output_dir)
    sdf_dir = output_dir / "sdf"
    launch_dir = output_dir / "launch"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    launch_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for row in profiles.sort("bald_information_value", descending=True).iter_rows(named=True):
        manifest_rows.append(write_dispatch_assets(row, sdf_dir=sdf_dir, launch_dir=launch_dir, replicas=int(args.replicas)))

    manifest = {
        "schema_version": "PRISM.gpu_dispatch.tripartite.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "profiles": str(Path(args.profiles)),
        "replicas": int(args.replicas),
        "dispatch_count": len(manifest_rows),
        "dispatches": manifest_rows,
    }
    atomic_write_json(Path(args.manifest), manifest)
    print(f"gpu_dispatch_batch_generated count={len(manifest_rows)} manifest={args.manifest}")
    return 0


def write_dispatch_assets(
    row: Mapping[str, Any],
    *,
    sdf_dir: Path,
    launch_dir: Path,
    replicas: int,
) -> dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    smiles = str(row["canonical_smiles"])
    sdf_path = sdf_dir / f"{candidate_id}.sdf"
    script_path = launch_dir / f"launch-n{replicas}-validate-{candidate_id}.sh"
    write_sdf(smiles, sdf_path)
    script_path.write_text(slurm_script(candidate_id, sdf_path, replicas), encoding="utf-8")
    script_path.chmod(0o755)
    return {
        "candidate_id": candidate_id,
        "canonical_smiles": smiles,
        "sdf": str(sdf_path),
        "launch_script": str(script_path),
        "bald_information_value": float(row["bald_information_value"]),
        "lock_geometry_score": float(row["lock_geometry_score"]),
        "epistemic_confidence": str(row["epistemic_confidence"]),
    }


def write_sdf(smiles: str, output: Path) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES for SDF generation: {smiles}")
    mol_h = Chem.AddHs(mol)
    all_chem = cast(Any, AllChem)
    status = int(all_chem.EmbedMolecule(mol_h, randomSeed=20260524))
    if status != 0:
        all_chem.Compute2DCoords(mol_h)
    else:
        all_chem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    writer = Chem.SDWriter(str(output))
    writer.write(mol_h)
    writer.close()


def slurm_script(candidate_id: str, sdf_path: Path, replicas: int) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=prism-{candidate_id}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

CANDIDATE_ID="{candidate_id}"
SDF_PATH="{sdf_path}"
REPLICAS="{replicas}"
PROTOCOL="ccns_5phase_validation"

echo "gpu_validation_start candidate=${{CANDIDATE_ID}} replicas=${{REPLICAS}} sdf=${{SDF_PATH}} protocol=${{PROTOCOL}}"
python3 scripts/run_ccns_validation_md.py \\
  --candidate-id "${{CANDIDATE_ID}}" \\
  --sdf "${{SDF_PATH}}" \\
  --replicas "${{REPLICAS}}" \\
  --protocol "${{PROTOCOL}}"
echo "gpu_validation_complete candidate=${{CANDIDATE_ID}}"
"""


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
