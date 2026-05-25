#!/usr/bin/env python3
"""Launch representative Phase 2C de novo captures with trajectory output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
DEFAULT_WORKSPACE = Path("/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z")
DEFAULT_RUNTIME_ENV = DEFAULT_WORKSPACE / "02_RUNTIME_CONFIG/glp1r_runtime.env"
DEFAULT_OUTPUT_ROOT = CAMPAIGN_DIR / "phase_2c_de_novo_capture"
DEFAULT_CAPTURE_MANIFEST = DEFAULT_OUTPUT_ROOT / "de_novo_capture_manifest.json"
TARGET_CONDITIONS = ("glp1r_6XOX_WT", "glp1r_6LN2_A316T")
EQUILIBRIUM_STREAM_ID = 0
MAX_STEPS = 6015
JsonObject: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class CaptureCommand:
    condition_id: str
    topology_json: Path
    topology_sha256: str
    output_dir: Path
    replica_seed: int
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--runtime-env", type=Path, default=DEFAULT_RUNTIME_ENV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--capture-manifest", type=Path, default=DEFAULT_CAPTURE_MANIFEST)
    parser.add_argument("--base-seed", type=int, default=91_000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_runtime_env(runtime_env: Path) -> dict[str, str]:
    keys = [
        "PRISM_ENGINE_BIN",
        "PRISM_ENGINE_LIB_DIR",
        "PRISM_PTX_DIR",
        "PRISM4D_PTX_DIR",
        "LD_LIBRARY_PATH",
    ]
    key_literal = json.dumps(keys)
    command = (
        "set -euo pipefail; "
        f"source {shlex.quote(str(runtime_env))}; "
        f"python3 -c {shlex.quote('import json, os; keys=' + key_literal + '; print(json.dumps({k: os.environ.get(k, \"\") for k in keys}, sort_keys=True))')}"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to source {runtime_env}:\n{result.stderr}")
    decoded = json.loads(result.stdout)
    if not isinstance(decoded, dict):
        raise ValueError("runtime environment dump was not a JSON object")
    return {str(key): str(value) for key, value in decoded.items()}


def topology_path(workspace: Path, condition_id: str) -> Path:
    path = workspace / "04_TOPOLOGIES" / f"{condition_id}.topology.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def nma_modes_path(topology_json: Path) -> Path | None:
    candidate = topology_json.with_name(f"{topology_json.stem.removesuffix('.topology')}_nma_modes.json")
    return candidate if candidate.exists() else None


def build_commands(args: argparse.Namespace, env_values: dict[str, str]) -> tuple[CaptureCommand, ...]:
    workspace = Path(args.workspace)
    output_root = Path(args.output_root)
    engine_bin = Path(env_values["PRISM_ENGINE_BIN"])
    commands: list[CaptureCommand] = []
    for offset, condition_id in enumerate(TARGET_CONDITIONS):
        topology_json = topology_path(workspace, condition_id)
        nma_modes = nma_modes_path(topology_json)
        output_dir = output_root / "single_stream_representative" / condition_id
        seed = int(args.base_seed) + offset
        command = [
            str(engine_bin),
            "-t",
            str(topology_json),
            "-o",
            str(output_dir),
            "--steps",
            str(MAX_STEPS),
            "--replicas",
            "1",
            "--replica-seed",
            str(seed),
            "--ensemble-campaign-id",
            "glp1r_aleniglipron_de_novo_capture",
            "--ensemble-base-seed",
            str(args.base_seed),
            "--ensemble-replica-id",
            "0",
            "--save-trajectory-interval",
            "1",
            "--cryo-temp",
            "150.0",
            "--temperature",
            "310.0",
            "--spike-percentile",
            "70",
            "--fused-steps",
            "6",
            "--hmr",
            "--adaptive-dt",
            "--site-ranker",
            "phase-manifold",
            "--uv-burst-energy",
            "25.0",
            "--uv-burst-step",
            "400",
            "--uv-wavelengths",
            "280,274,258,254,211",
            "--uv-wavelength-dwell-steps",
            "400",
            "--nma-amplification",
            "3.0",
            "--nma-scan-fraction",
            "0.3",
        ]
        if nma_modes is not None:
            command.extend(["--nma-perturb", str(nma_modes)])
        commands.append(
            CaptureCommand(
                condition_id=condition_id,
                topology_json=topology_json,
                topology_sha256=sha256_file(topology_json),
                output_dir=output_dir,
                replica_seed=seed,
                command=tuple(command),
            )
        )
    return tuple(commands)


def launch_env(env_values: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env["PRISM_VALIDATED"] = "1"
    for key in ("PRISM_PTX_DIR", "PRISM4D_PTX_DIR", "LD_LIBRARY_PATH"):
        value = env_values.get(key)
        if value:
            env[key] = value
    return env


def capture_outputs(output_dir: Path) -> list[Path]:
    suffixes = (".pdb", ".frames.bin", "_v2_frames.bin")
    return sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name.endswith(suffixes))


def command_to_json(command: CaptureCommand, outputs: list[Path]) -> JsonObject:
    return {
        "condition_id": command.condition_id,
        "representative_capture": True,
        "extracted_protocol_group": "single_stream_equilibrium_like",
        "equilibrium_stream_id": EQUILIBRIUM_STREAM_ID,
        "requested_max_steps": MAX_STEPS,
        "protocol_note": (
            "Pinned binary exposes TWIN Equilibrium only through multi-differential graph mode, "
            "which does not emit interval snapshots. This single-stream run uses Equilibrium-like "
            "temperature and UV settings strictly as a representative Cartesian anchor."
        ),
        "trajectory_save_interval": 1,
        "replica_seed": command.replica_seed,
        "topology_json": str(command.topology_json),
        "topology_sha256": command.topology_sha256,
        "output_dir": str(command.output_dir),
        "command": list(command.command),
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in outputs
        ],
    }


def write_capture_manifest(
    *,
    path: Path,
    workspace: Path,
    runtime_env: Path,
    env_values: dict[str, str],
    commands: tuple[CaptureCommand, ...],
    completed: dict[str, list[Path]],
) -> None:
    engine_bin = Path(env_values["PRISM_ENGINE_BIN"])
    payload: JsonObject = {
        "campaign_id": "glp1r_aleniglipron",
        "phase": "phase_2c_de_novo_representative_capture",
        "epistemic_class": "REPRESENTATIVE_CAPTURE",
        "rationale": (
            "Original 80-replica campaign lacked checkpoints. De Novo capture executed to provide Cartesian alignment "
            "anchor. Thermodynamic scoring remains anchored to the original 80-replica variance field."
        ),
        "workspace": str(workspace),
        "runtime_env": str(runtime_env),
        "runtime_env_sha256": sha256_file(runtime_env),
        "engine_binary": str(engine_bin),
        "engine_binary_sha256": sha256_file(engine_bin),
        "targets": [
            command_to_json(command, completed.get(command.condition_id, []))
            for command in commands
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_commands(
    commands: tuple[CaptureCommand, ...],
    env: dict[str, str],
    log_root: Path,
    workspace: Path,
) -> dict[str, list[Path]]:
    completed: dict[str, list[Path]] = {}
    log_root.mkdir(parents=True, exist_ok=True)
    for command in commands:
        command.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{command.condition_id}.log"
        emit(f"launching de_novo_capture condition={command.condition_id} seed={command.replica_seed}")
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("$ " + shlex.join(command.command) + "\n")
            result = subprocess.run(
                list(command.command),
                cwd=workspace,
                env=env,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"de novo capture failed for {command.condition_id}; log={log_path}")
        outputs = capture_outputs(command.output_dir)
        if not outputs:
            raise FileNotFoundError(f"no trajectory outputs found under {command.output_dir}")
        completed[command.condition_id] = outputs
        emit(f"completed condition={command.condition_id} outputs={len(outputs)}")
    return completed


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace)
    runtime_env = Path(args.runtime_env)
    if not workspace.exists():
        raise FileNotFoundError(workspace)
    if not runtime_env.exists():
        raise FileNotFoundError(runtime_env)
    env_values = source_runtime_env(runtime_env)
    engine_bin = Path(env_values["PRISM_ENGINE_BIN"])
    if not engine_bin.exists():
        raise FileNotFoundError(engine_bin)
    commands = build_commands(args, env_values)
    completed: dict[str, list[Path]] = {command.condition_id: capture_outputs(command.output_dir) for command in commands}
    write_capture_manifest(
        path=Path(args.capture_manifest),
        workspace=workspace,
        runtime_env=runtime_env,
        env_values=env_values,
        commands=commands,
        completed=completed if args.dry_run else {},
    )
    if args.dry_run:
        emit(f"capture_manifest={args.capture_manifest}")
        emit("dry_run=true")
        return 0
    completed = run_commands(commands, launch_env(env_values), Path(args.output_root) / "logs", workspace)
    write_capture_manifest(
        path=Path(args.capture_manifest),
        workspace=workspace,
        runtime_env=runtime_env,
        env_values=env_values,
        commands=commands,
        completed=completed,
    )
    emit(f"capture_manifest={args.capture_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
