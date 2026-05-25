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
DEFAULT_PROTOCOL = "ccns_5phase"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--candidates", type=Path, default=None, help="Alias for --profiles used by Epoch 016.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument("--n-replicas", type=int, default=None)
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lock-positive-only", action="store_true", default=False)
    parser.add_argument("--bald-ranking", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles_path = Path(args.candidates) if args.candidates is not None else Path(args.profiles)
    profiles = normalize_profiles(pl.read_parquet(profiles_path))
    if bool(args.lock_positive_only):
        profiles = profiles.filter(pl.col("lock_geometry_score") > 0.0)
    output_dir = Path(args.output_dir)
    sdf_dir = output_dir / "sdf"
    launch_dir = output_dir / "launch"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    launch_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    replicas = int(args.n_replicas) if args.n_replicas is not None else int(args.replicas)
    sort_column = "bald_information_value" if bool(args.bald_ranking) else "lock_geometry_score"
    for row in profiles.sort(sort_column, descending=True).iter_rows(named=True):
        manifest_rows.append(
            write_dispatch_assets(
                row,
                sdf_dir=sdf_dir,
                launch_dir=launch_dir,
                replicas=replicas,
                protocol=str(args.protocol),
            )
        )

    manifest = {
        "schema_version": "PRISM.gpu_dispatch.tripartite.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "profiles": str(profiles_path),
        "replicas": replicas,
        "protocol": str(args.protocol),
        "lock_positive_only": bool(args.lock_positive_only),
        "bald_ranking": bool(args.bald_ranking),
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
    protocol: str,
) -> dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    smiles = str(row["canonical_smiles"])
    sdf_path = sdf_dir / f"{candidate_id}.sdf"
    script_path = launch_dir / f"launch-n{replicas}-validate-{candidate_id}.sh"
    write_sdf(smiles, sdf_path)
    script_path.write_text(slurm_script(candidate_id, sdf_path, replicas, protocol), encoding="utf-8")
    script_path.chmod(0o755)
    return {
        "candidate_id": candidate_id,
        "canonical_smiles": smiles,
        "sdf": str(sdf_path),
        "launch_script": str(script_path),
        "bald_information_value": float(row.get("bald_information_value", 0.0)),
        "lock_geometry_score": float(row.get("lock_geometry_score", 0.0)),
        "epistemic_confidence": str(row.get("epistemic_confidence", "L1")),
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


def slurm_script(candidate_id: str, sdf_path: Path, replicas: int, protocol: str) -> str:
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
PROTOCOL="{protocol}"

echo "gpu_validation_start candidate=${{CANDIDATE_ID}} replicas=${{REPLICAS}} sdf=${{SDF_PATH}} protocol=${{PROTOCOL}}"
python3 scripts/run_ccns_validation_md.py \\
  --candidate-id "${{CANDIDATE_ID}}" \\
  --sdf "${{SDF_PATH}}" \\
  --replicas "${{REPLICAS}}" \\
  --protocol "${{PROTOCOL}}"
echo "gpu_validation_complete candidate=${{CANDIDATE_ID}}"
"""


def normalize_profiles(frame: pl.DataFrame) -> pl.DataFrame:
    """Fill dispatch columns when upstream input is a candidate parquet."""

    if "canonical_smiles" not in frame.columns and "smiles" in frame.columns:
        frame = frame.with_columns(pl.col("smiles").alias("canonical_smiles"))
    if "candidate_id" not in frame.columns:
        frame = frame.with_row_index("candidate_rank", offset=1).with_columns(
            pl.concat_str(
                [
                    pl.lit("cand_"),
                    pl.col("candidate_rank").cast(pl.Utf8),
                    pl.lit("_"),
                    pl.col("canonical_smiles").hash(seed=20260524).cast(pl.Utf8).str.slice(0, 8),
                ]
            ).alias("candidate_id")
        )
    if "lock_geometry_score" not in frame.columns:
        if "pi_clash_lock" in frame.columns:
            frame = frame.with_columns(pl.col("pi_clash_lock").alias("lock_geometry_score"))
        else:
            frame = frame.with_columns(pl.lit(0.0).alias("lock_geometry_score"))
    if "bald_information_value" not in frame.columns:
        projection = pl.col("bias_projection_score") if "bias_projection_score" in frame.columns else pl.lit(0.5)
        frame = frame.with_columns(
            (
                pl.col("lock_geometry_score")
                * (1.0 - (projection - 0.5).abs() * 2.0).clip(0.0, 1.0)
                * 0.75
            ).alias("bald_information_value")
        )
    if "epistemic_confidence" not in frame.columns:
        frame = frame.with_columns(pl.lit("L1").alias("epistemic_confidence"))
    return frame


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
