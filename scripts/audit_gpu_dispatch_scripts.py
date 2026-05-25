#!/usr/bin/env python3
"""Audit and repair GPU dispatch scripts so they call the production NHS engine path."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_DISPATCH_DIR = TRACK_A / "gpu_dispatch"
DEFAULT_MANIFEST = TRACK_A / "gpu_dispatch_manifest.json"
DEFAULT_REPORT = TRACK_A / "gpu_dispatch_audit_report.json"
DEFAULT_ENGINE = "demo/nhs_rt_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-dir", type=Path, default=DEFAULT_DISPATCH_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--engine", type=str, default=DEFAULT_ENGINE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dispatch_dir = Path(args.dispatch_dir)
    launch_dir = dispatch_dir / "launch"
    corrected_dir = dispatch_dir / "launch_corrected"
    topology_dir = dispatch_dir / "topologies"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    topology_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(Path(args.manifest)) if Path(args.manifest).is_file() else {"dispatches": []}
    dispatches = manifest.get("dispatches", [])
    if not isinstance(dispatches, list):
        dispatches = []
    rows: list[dict[str, Any]] = []
    corrected_count = 0
    ready_count = 0
    high_priority_count = 0
    for item in dispatches:
        if not isinstance(item, dict):
            continue
        script_path = Path(str(item.get("launch_script", "")))
        if not script_path.is_absolute():
            script_path = REPO_ROOT / script_path
        candidate_id = str(item.get("candidate_id", script_path.stem.removeprefix("launch-n10-validate-")))
        sdf_path = Path(str(item.get("sdf", "")))
        if not sdf_path.is_absolute():
            sdf_path = REPO_ROOT / sdf_path
        bald_value = float_value(item.get("bald_information_value"))
        high_priority = bald_value >= 0.50
        if high_priority:
            high_priority_count += 1
        syntax_ok = bash_syntax_ok(script_path) if script_path.is_file() else False
        engine_ok = script_references_engine(script_path, str(args.engine)) if script_path.is_file() else False
        corrected_script = None
        if not syntax_ok or not engine_ok:
            corrected_script = corrected_dir / script_path.name
            topology_json = topology_dir / f"{candidate_id}.json"
            write_topology_json(topology_json, candidate_id, sdf_path, item)
            corrected_script.write_text(
                corrected_slurm_script(
                    candidate_id=candidate_id,
                    sdf_path=sdf_path,
                    topology_json=topology_json,
                    replicas=int(float_value(item.get("replicas"), 10.0)),
                    engine=str(args.engine),
                ),
                encoding="utf-8",
            )
            corrected_script.chmod(0o755)
            syntax_ok = bash_syntax_ok(corrected_script)
            engine_ok = script_references_engine(corrected_script, str(args.engine))
            corrected_count += 1
        dispatch_ready = syntax_ok and engine_ok and sdf_path.is_file()
        if dispatch_ready:
            ready_count += 1
        rows.append(
            {
                "candidate_id": candidate_id,
                "original_script": script_path.as_posix(),
                "corrected_script": corrected_script.as_posix() if corrected_script is not None else None,
                "sdf_exists": sdf_path.is_file(),
                "syntax_ok": syntax_ok,
                "engine_ok": engine_ok,
                "dispatch_ready": dispatch_ready,
                "bald_information_value": bald_value,
                "priority_class": "HIGH" if high_priority else "STANDARD",
            }
        )
    report = {
        "schema_version": "PRISM.gpu_dispatch_audit.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dispatch_dir": dispatch_dir.as_posix(),
        "manifest": Path(args.manifest).as_posix(),
        "engine": str(args.engine),
        "dispatch_count": len(rows),
        "dispatch_ready_count": ready_count,
        "corrected_script_count": corrected_count,
        "high_priority_count": high_priority_count,
        "status": "PASS" if rows and ready_count == len(rows) else "WARN",
        "note": "HIGH priority uses BALD information value >= 0.50; lower values are reported honestly as STANDARD.",
        "dispatches": rows,
    }
    atomic_write_json(Path(args.report), report)
    print(
        "gpu_dispatch_audit "
        f"status={report['status']} count={len(rows)} ready={ready_count} corrected={corrected_count} "
        f"high_priority={high_priority_count} report={args.report}"
    )
    return 0


def corrected_slurm_script(
    *,
    candidate_id: str,
    sdf_path: Path,
    topology_json: Path,
    replicas: int,
    engine: str,
) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=prism-{candidate_id}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

CANDIDATE_ID="{candidate_id}"
SDF_PATH="{sdf_path}"
TOPOLOGY_JSON="{topology_json}"
REPLICAS="{replicas}"
PROTOCOL="ccns_5phase"
ENGINE="${{PRISM_NHS_ENGINE:-{engine}}}"
OUTPUT_DIR="$(dirname "$TOPOLOGY_JSON")/../results/${{CANDIDATE_ID}}"
mkdir -p "$OUTPUT_DIR"

echo "gpu_validation_start candidate=${{CANDIDATE_ID}} replicas=${{REPLICAS}} engine=${{ENGINE}}"
"$ENGINE" \\
  --topology "$TOPOLOGY_JSON" \\
  --ligand-sdf "$SDF_PATH" \\
  --protocol "$PROTOCOL" \\
  --n-replicas "$REPLICAS" \\
  --multi-differential \\
  --hysteresis \\
  --prism-therm \\
  --save-trajectory-interval 50 \\
  --output-dir "$OUTPUT_DIR"
echo "gpu_validation_complete candidate=${{CANDIDATE_ID}} output_dir=${{OUTPUT_DIR}}"
"""


def write_topology_json(path: Path, candidate_id: str, sdf_path: Path, row: dict[str, Any]) -> None:
    payload = {
        "schema_version": "PRISM.gpu_dispatch_topology.v1",
        "candidate_id": candidate_id,
        "ligand_sdf": sdf_path.as_posix(),
        "canonical_smiles": row.get("canonical_smiles"),
        "lock_geometry_score": row.get("lock_geometry_score"),
        "bald_information_value": row.get("bald_information_value"),
        "epistemic_confidence": row.get("epistemic_confidence"),
        "provenance": "generated_for_ccns_validation_dispatch",
    }
    atomic_write_json(path, payload)


def script_references_engine(path: Path, engine: str) -> bool:
    text = path.read_text(encoding="utf-8")
    return engine in text and "--multi-differential" in text and "--hysteresis" in text and "--prism-therm" in text


def bash_syntax_ok(path: Path) -> bool:
    result = subprocess.run(["bash", "-n", str(path)], cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    return result.returncode == 0


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return loaded


def float_value(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int | float | str):
        return float(value)
    return default


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
