#!/usr/bin/env python3
"""Generate four-channel GPU validation dispatch scripts for lock-positive candidates."""

from __future__ import annotations

import argparse
import json
import re
import shlex
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
DEFAULT_PROTOCOL_STATE_SUMMARY = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/protocol_state_summary.parquet"
)
DEFAULT_RAW_ROOT = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/05_RESULTS/glp1r_aleniglipron_risk_map"
)
DEFAULT_TOPOLOGY_ROOT = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/04_TOPOLOGIES"
)
SAFE_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--candidates", type=Path, default=None, help="Alias for --profiles used by Epoch 016.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument("--n-replicas", type=int, default=None)
    parser.add_argument("--protocol", type=str, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-state-summary", type=Path, default=DEFAULT_PROTOCOL_STATE_SUMMARY)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--topology-root", type=Path, default=DEFAULT_TOPOLOGY_ROOT)
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
    topology_dir = output_dir / "topologies"
    launch_dir = output_dir / "launch"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    topology_dir.mkdir(parents=True, exist_ok=True)
    launch_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    replicas = int(args.n_replicas) if args.n_replicas is not None else int(args.replicas)
    sort_column = "bald_information_value" if bool(args.bald_ranking) else "lock_geometry_score"
    for row in profiles.sort(sort_column, descending=True).iter_rows(named=True):
        manifest_rows.append(
            write_dispatch_assets(
                row,
                sdf_dir=sdf_dir,
                topology_dir=topology_dir,
                launch_dir=launch_dir,
                replicas=replicas,
                protocol=str(args.protocol),
                protocol_state_summary=Path(args.protocol_state_summary),
                raw_root=Path(args.raw_root),
                topology_root=Path(args.topology_root),
            )
        )

    manifest = {
        "schema_version": "PRISM.gpu_dispatch.tripartite.v2",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "profiles": str(profiles_path),
        "replicas": replicas,
        "protocol": str(args.protocol),
        "protocol_state_summary": str(Path(args.protocol_state_summary)),
        "raw_root": str(Path(args.raw_root)),
        "topology_root": str(Path(args.topology_root)),
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
    topology_dir: Path,
    launch_dir: Path,
    replicas: int,
    protocol: str,
    protocol_state_summary: Path,
    raw_root: Path,
    topology_root: Path,
) -> dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    validate_candidate_id(candidate_id)
    smiles = str(row["canonical_smiles"])
    sdf_path = sdf_dir / f"{candidate_id}.sdf"
    topology_path = topology_dir / f"{candidate_id}.json"
    script_path = launch_dir / f"launch-n{replicas}-validate-{candidate_id}.sh"
    write_sdf(smiles, sdf_path)
    write_topology_json(topology_path, candidate_id, sdf_path, row)
    script_path.write_text(
        slurm_script(
            candidate_id,
            sdf_path,
            topology_path,
            replicas,
            protocol,
            protocol_state_summary,
            raw_root,
            topology_root,
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    return {
        "candidate_id": candidate_id,
        "canonical_smiles": smiles,
        "sdf": str(sdf_path),
        "topology_json": str(topology_path),
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


def write_topology_json(
    path: Path,
    candidate_id: str,
    sdf_path: Path,
    row: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": "PRISM.gpu_dispatch_topology.v2",
        "candidate_id": candidate_id,
        "ligand_sdf": str(sdf_path),
        "canonical_smiles": str(row["canonical_smiles"]),
        "lock_geometry_score": float(row.get("lock_geometry_score", 0.0)),
        "bald_information_value": float(row.get("bald_information_value", 0.0)),
        "epistemic_confidence": str(row.get("epistemic_confidence", "L1")),
        "provenance": "generated_for_four_channel_gpu_dispatch",
    }
    atomic_write_json(path, payload)


def slurm_script(
    candidate_id: str,
    sdf_path: Path,
    topology_path: Path,
    replicas: int,
    protocol: str,
    protocol_state_summary: Path,
    raw_root: Path,
    topology_root: Path,
) -> str:
    repo_root = shell_literal(str(REPO_ROOT))
    default_protocol_state_summary = shell_literal(path_for_script(protocol_state_summary))
    reference_raw_root = shell_literal(path_for_script(raw_root))
    default_topology_root = shell_literal(path_for_script(topology_root))
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=prism-{candidate_id}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT={repo_root}
cd "$REPO_ROOT"

CANDIDATE_ID={shell_literal(candidate_id)}
SDF_PATH={shell_literal(path_for_script(sdf_path))}
TOPOLOGY_JSON={shell_literal(path_for_script(topology_path))}
REPLICAS="{replicas}"
PROTOCOL={shell_literal(protocol)}
DISPATCH_ROOT="$(dirname "$SDF_PATH")/.."
VALIDATION_MANIFEST_DIR="${{PRISM_VALIDATION_MANIFEST_DIR:-${{DISPATCH_ROOT}}/validation_manifests}}"
RESULTS_ROOT="${{PRISM_GPU_RESULTS_ROOT:-${{DISPATCH_ROOT}}/results}}"
OUTPUT_DIR="${{RESULTS_ROOT}}/${{CANDIDATE_ID}}"
CHANNEL_DIR="${{OUTPUT_DIR}}/channels"
DEFAULT_PROTOCOL_STATE_SUMMARY={default_protocol_state_summary}
REFERENCE_RAW_ROOT={reference_raw_root}
DEFAULT_TOPOLOGY_ROOT={default_topology_root}
PROTOCOL_STATE_SUMMARY="${{PRISM_PROTOCOL_STATE_SUMMARY:-$DEFAULT_PROTOCOL_STATE_SUMMARY}}"
CANDIDATE_RAW_ROOT="${{PRISM_CANDIDATE_RAW_ROOT:-${{OUTPUT_DIR}}/ccns_raw}}"
RAW_ROOT="${{PRISM_RAW_ROOT:-$CANDIDATE_RAW_ROOT}}"
TOPOLOGY_ROOT="${{PRISM_TOPOLOGY_ROOT:-$DEFAULT_TOPOLOGY_ROOT}}"
ALLOW_SHARED_RAW_ROOT="${{PRISM_ALLOW_SHARED_RAW_ROOT:-0}}"
BIFURCATE_CHANNELS="pocket,lock,pathway"
SIGNAL_GRID_DIR="${{CHANNEL_DIR}}/channel_1_signal_grid"
WARP_JACOBIAN_DIR="${{CHANNEL_DIR}}/channel_2_warp_jacobian"
HYSTERESIS_DIR="${{CHANNEL_DIR}}/channel_3_hysteresis"
PATHWAY_DIR="${{CHANNEL_DIR}}/channel_4_pathway"
TIMESTEP_JSON="${{OUTPUT_DIR}}/protocol_timesteps.json"
SIGNAL_GRID_PARQUET="${{SIGNAL_GRID_DIR}}/signal_grid_variance_channel.parquet"
WARP_JACOBIAN_PARQUET="${{WARP_JACOBIAN_DIR}}/shear_stress_field.parquet"
HYSTERESIS_JSON="${{HYSTERESIS_DIR}}/hysteresis_analysis.json"
PATHWAY_JSON="${{PATHWAY_DIR}}/pathway_analysis.json"
TRIPARTITE_UPGRADE_JSON="${{OUTPUT_DIR}}/tripartite_upgrade.json"

mkdir -p "$VALIDATION_MANIFEST_DIR" "$SIGNAL_GRID_DIR" "$WARP_JACOBIAN_DIR" "$HYSTERESIS_DIR" "$PATHWAY_DIR"

assert_candidate_raw_root() {{
  if [ "$ALLOW_SHARED_RAW_ROOT" != "1" ]; then
    if [ "$RAW_ROOT" = "$REFERENCE_RAW_ROOT" ]; then
      echo "PRISM_RAW_ROOT points at shared campaign raw root; set PRISM_CANDIDATE_RAW_ROOT/PRISM_RAW_ROOT to candidate-specific CCNS output or PRISM_ALLOW_SHARED_RAW_ROOT=1" >&2
      return 2
    fi
    case "$RAW_ROOT" in
      *"$CANDIDATE_ID"*) ;;
      *) echo "RAW_ROOT must include candidate id ($CANDIDATE_ID) unless PRISM_ALLOW_SHARED_RAW_ROOT=1: $RAW_ROOT" >&2; return 2 ;;
    esac
  fi
  if ! find "$RAW_ROOT" -name signal_grid.bin -print -quit 2>/dev/null | grep -q .; then
    echo "candidate-specific RAW_ROOT has no signal_grid.bin: $RAW_ROOT" >&2
    return 2
  fi
  if ! find "$RAW_ROOT" -name warp_matrix.bin -print -quit 2>/dev/null | grep -q .; then
    echo "candidate-specific RAW_ROOT has no warp_matrix.bin: $RAW_ROOT" >&2
    return 2
  fi
}}

python3 scripts/process_gpu_dispatch_results.py \\
  --mode timestep_extraction \\
  --protocol-state-summary "$PROTOCOL_STATE_SUMMARY" \\
  --output "$TIMESTEP_JSON"

extract_json_list() {{
  local json_path="$1"
  local key="$2"
  python3 - "$json_path" "$key" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
values = payload.get(sys.argv[2], [])
if isinstance(values, list):
    print(",".join(str(int(value)) for value in values))
else:
    print("")
PY
}}

EQUILIBRATED_FRAMES="$(extract_json_list "$TIMESTEP_JSON" equilibrated_frames)"
RAMP_PHASE_FRAMES="$(extract_json_list "$TIMESTEP_JSON" ramp_frames)"

run_prism_nhs_bin() {{
  local bin_name="$1"
  shift
  cargo run -p prism-nhs --bin "$bin_name" -- "$@"
}}

signal_grid_differential() {{
  local raw_root=""
  local out_dir=""
  local protocol_state_summary=""
  local frame_scope=""
  local bifurcate=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --raw-root) raw_root="$2"; shift 2 ;;
      --out-dir) out_dir="$2"; shift 2 ;;
      --protocol-state-summary) protocol_state_summary="$2"; shift 2 ;;
      --frame-scope) frame_scope="$2"; shift 2 ;;
      --bifurcate) bifurcate="$2"; shift 2 ;;
      *) echo "signal_grid_differential unsupported arg: $1" >&2; return 2 ;;
    esac
  done
  run_prism_nhs_bin signal_grid_differential \\
    --raw-root "$raw_root" \\
    --out-dir "$out_dir" \\
    --protocol-state-summary "$protocol_state_summary" \\
    --frame-scope "$frame_scope" \\
    --bifurcate "$bifurcate"
  python3 scripts/process_gpu_dispatch_results.py \\
    --mode channel_metadata \\
    --channel signal_grid_differential \\
    --protocol-state-summary "$protocol_state_summary" \\
    --frame-scope "$frame_scope" \\
    --bifurcate "$bifurcate" \\
    --output "${{out_dir}}/dispatch_channel_metadata.json"
}}

warp_jacobian() {{
  local raw_root=""
  local out_dir=""
  local topology_root=""
  local protocol_state_summary=""
  local frames=""
  local bifurcate=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --raw-root) raw_root="$2"; shift 2 ;;
      --out-dir) out_dir="$2"; shift 2 ;;
      --topology-root) topology_root="$2"; shift 2 ;;
      --protocol-state-summary) protocol_state_summary="$2"; shift 2 ;;
      --frames) frames="$2"; shift 2 ;;
      --bifurcate) bifurcate="$2"; shift 2 ;;
      *) echo "warp_jacobian unsupported arg: $1" >&2; return 2 ;;
    esac
  done
  run_prism_nhs_bin warp_jacobian \\
    --raw-root "$raw_root" \\
    --out-dir "$out_dir" \\
    --topology-root "$topology_root" \\
    --protocol-state-summary "$protocol_state_summary" \\
    --frames "$frames" \\
    --bifurcate "$bifurcate"
  python3 scripts/process_gpu_dispatch_results.py \\
    --mode channel_metadata \\
    --channel warp_jacobian \\
    --frames "$frames" \\
    --bifurcate "$bifurcate" \\
    --output "${{out_dir}}/dispatch_channel_metadata.json"
}}

hysteresis_analysis() {{
  run_prism_nhs_bin hysteresis_analysis "$@"
}}

pathway_analysis() {{
  run_prism_nhs_bin pathway_analysis "$@"
}}

echo "gpu_validation_start candidate=${{CANDIDATE_ID}} replicas=${{REPLICAS}} sdf=${{SDF_PATH}} protocol=${{PROTOCOL}}"
python3 scripts/run_ccns_validation_md.py \\
  --candidate-id "${{CANDIDATE_ID}}" \\
  --sdf "${{SDF_PATH}}" \\
  --replicas "${{REPLICAS}}" \\
  --protocol "${{PROTOCOL}}" \\
  --output-dir "${{VALIDATION_MANIFEST_DIR}}" \\
  --raw-output-dir "${{CANDIDATE_RAW_ROOT}}"

assert_candidate_raw_root

echo "gpu_dispatch_channel_1_start candidate=${{CANDIDATE_ID}}"
signal_grid_differential \\
  --raw-root "${{RAW_ROOT}}" \\
  --out-dir "${{SIGNAL_GRID_DIR}}" \\
  --protocol-state-summary "${{PROTOCOL_STATE_SUMMARY}}" \\
  --frame-scope "all" \\
  --bifurcate "${{BIFURCATE_CHANNELS}}"

echo "gpu_dispatch_channel_2_start candidate=${{CANDIDATE_ID}}"
warp_jacobian \\
  --raw-root "${{RAW_ROOT}}" \\
  --out-dir "${{WARP_JACOBIAN_DIR}}" \\
  --topology-root "${{TOPOLOGY_ROOT}}" \\
  --protocol-state-summary "${{PROTOCOL_STATE_SUMMARY}}" \\
  --frames "${{EQUILIBRATED_FRAMES}}" \\
  --bifurcate "${{BIFURCATE_CHANNELS}}"

echo "gpu_dispatch_channel_3_start candidate=${{CANDIDATE_ID}}"
hysteresis_analysis \\
  --candidate-id "${{CANDIDATE_ID}}" \\
  --signal-grid "${{SIGNAL_GRID_PARQUET}}" \\
  --protocol-state-summary "${{PROTOCOL_STATE_SUMMARY}}" \\
  --output "${{HYSTERESIS_JSON}}" \\
  --bifurcate "${{BIFURCATE_CHANNELS}}"

echo "gpu_dispatch_channel_4_start candidate=${{CANDIDATE_ID}}"
pathway_analysis \\
  --candidate-id "${{CANDIDATE_ID}}" \\
  --signal-grid "${{SIGNAL_GRID_PARQUET}}" \\
  --warp-jacobian "${{WARP_JACOBIAN_PARQUET}}" \\
  --protocol-state-summary "${{PROTOCOL_STATE_SUMMARY}}" \\
  --phase-filter "${{RAMP_PHASE_FRAMES}}" \\
  --output "${{PATHWAY_JSON}}" \\
  --bifurcate "${{BIFURCATE_CHANNELS}}"

python3 scripts/process_gpu_dispatch_results.py \\
  --mode tripartite_upgrade \\
  --candidate-id "${{CANDIDATE_ID}}" \\
  --dispatch-dir "${{DISPATCH_ROOT}}" \\
  --result-dir "${{OUTPUT_DIR}}" \\
  --signal-grid "${{SIGNAL_GRID_PARQUET}}" \\
  --warp-jacobian "${{WARP_JACOBIAN_PARQUET}}" \\
  --hysteresis "${{HYSTERESIS_JSON}}" \\
  --pathway "${{PATHWAY_JSON}}" \\
  --output "${{TRIPARTITE_UPGRADE_JSON}}" \\
  --bifurcate "${{BIFURCATE_CHANNELS}}"
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


def validate_candidate_id(candidate_id: str) -> None:
    if not SAFE_CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError(f"unsafe candidate_id for shell dispatch: {candidate_id}")


def shell_literal(value: str) -> str:
    return shlex.quote(value)


def path_for_script(path: Path) -> str:
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
