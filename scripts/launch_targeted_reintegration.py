#!/usr/bin/env python3
"""Launch Phase 2C targeted reintegration only after checkpoint parity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
DEFAULT_WORKSPACE = Path("/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z")
DEFAULT_RUN_SCRIPT = DEFAULT_WORKSPACE / "bin/run-one-glp1r-replica.sh"
DEFAULT_RUNTIME_ENV = DEFAULT_WORKSPACE / "02_RUNTIME_CONFIG/glp1r_runtime.env"
DEFAULT_MANIFEST = CAMPAIGN_DIR / "phase_2c_snapshot_triggers.json"
DEFAULT_SNAPSHOT_ROOT = CAMPAIGN_DIR / "phase_2c_snapshots"
DEFAULT_PARITY_RECORD = CAMPAIGN_DIR / "phase_2c_reintegration_parity.json"
FRAME_SUFFIXES = (".pdb", ".cif", ".dcd", ".frames.bin")
CHECKPOINT_SUFFIXES = (".chk", ".state", ".restart")
CHECKPOINT_FLAG_CANDIDATES = (
    "--checkpoint",
    "--checkpoint-file",
    "--restart",
    "--restart-file",
    "--state",
    "--state-file",
    "--load-state",
    "--resume-from",
)
JsonObject: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class TriggerTarget:
    condition_id: str
    replica_id: int
    stream_ids: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointFile:
    path: Path
    sha256: str
    size_bytes: int
    stream_id: int | None


@dataclass(frozen=True)
class LaunchCommand:
    target: TriggerTarget
    topology_json: Path
    topology_sha256: str
    nma_modes_json: Path | None
    nma_modes_sha256: str | None
    checkpoint_files: tuple[CheckpointFile, ...]
    replica_seed: int
    output_dir: Path
    command: tuple[str, ...]
    completion_json: Path | None
    completion_status: str | None
    original_run_id: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--run-script", type=Path, default=DEFAULT_RUN_SCRIPT)
    parser.add_argument("--runtime-env", type=Path, default=DEFAULT_RUNTIME_ENV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--parity-record", type=Path, default=DEFAULT_PARITY_RECORD)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unsupported-reintegrate",
        action="store_true",
        help="Bypass the engine help preflight. This is unsafe unless the binary supports hidden reintegration flags.",
    )
    return parser.parse_args()


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def as_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got bool")
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"{label} must be an integer")


def json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def load_json(path: Path) -> JsonObject:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not decode to a JSON object")
    return cast(JsonObject, loaded)


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
        "PRISM_CAMPAIGN_ID",
        "PRISM_BASE_SEED",
        "PRISM_STREAMS_PER_REPLICA",
        "PRISM_PATH_A_MAX_WALL_SECONDS",
        "PRISM_GLP1R_OUTPUT_ROOT",
        "PRISM_NMA_AMPLIFICATION",
        "PRISM_NMA_SCAN_FRACTION",
        "PRISM_ENGINE_FLAGS",
        "PRISM_GLP1R_WORKSPACE",
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


def unique_targets(manifest_path: Path) -> tuple[TriggerTarget, ...]:
    payload = load_json(manifest_path)
    triggers = json_list(payload.get("triggers"), "triggers")
    streams_by_target: dict[tuple[str, int], set[int]] = {}
    for raw_trigger in triggers:
        if not isinstance(raw_trigger, dict):
            raise TypeError("trigger entries must be JSON objects")
        condition_id = as_str(raw_trigger.get("condition_id"), "condition_id")
        replica_id = as_int(raw_trigger.get("replica_id"), "replica_id")
        stream_id = as_int(raw_trigger.get("stream_id"), "stream_id")
        streams_by_target.setdefault((condition_id, replica_id), set()).add(stream_id)
    return tuple(
        TriggerTarget(condition_id=condition_id, replica_id=replica_id, stream_ids=tuple(sorted(stream_ids)))
        for (condition_id, replica_id), stream_ids in sorted(streams_by_target.items())
    )


def topology_path(workspace: Path, condition_id: str) -> Path:
    path = workspace / "04_TOPOLOGIES" / f"{condition_id}.topology.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def completion_path(workspace: Path, campaign_id: str, target: TriggerTarget) -> Path | None:
    path = (
        workspace
        / "05_RESULTS"
        / campaign_id
        / target.condition_id
        / f"replica_{target.replica_id}"
        / f"{target.condition_id}_path_a_completion.json"
    )
    return path if path.exists() else None


def completion_metadata(path: Path | None) -> tuple[str | None, str | None]:
    if path is None:
        return (None, None)
    payload = load_json(path)
    return (
        str(payload.get("completion_status")) if payload.get("completion_status") is not None else None,
        str(payload.get("run_id")) if payload.get("run_id") is not None else None,
    )


def engine_help(engine_bin: Path) -> str:
    result = subprocess.run(
        [str(engine_bin), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=20,
    )
    return result.stdout


def supported_checkpoint_flag(help_text: str) -> str | None:
    for flag in CHECKPOINT_FLAG_CANDIDATES:
        if flag in help_text:
            return flag
    return None


def stream_id_from_path(path: Path) -> int | None:
    lowered = path.as_posix().lower()
    for pattern in (r"stream[_-]?0*(\d+)(?:\D|$)", r"prism_v2_[^/]*_0*(\d+)(?:\D|$)"):
        match = re.search(pattern, lowered)
        if match is not None:
            return int(match.group(1))
    return None


def discover_checkpoints(workspace: Path, campaign_id: str, target: TriggerTarget) -> tuple[CheckpointFile, ...]:
    replica_dir = workspace / "05_RESULTS" / campaign_id / target.condition_id / f"replica_{target.replica_id}"
    if not replica_dir.exists():
        return ()
    stream_set = set(target.stream_ids)
    checkpoint_paths = [
        path
        for path in replica_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
    ]
    checkpoints: list[CheckpointFile] = []
    for path in sorted(checkpoint_paths):
        stream_id = stream_id_from_path(path)
        if stream_id is not None and stream_id not in stream_set:
            continue
        checkpoints.append(
            CheckpointFile(
                path=path,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                stream_id=stream_id,
            )
        )
    return tuple(checkpoints)


def without_save_interval(flags: tuple[str, ...]) -> tuple[str, ...]:
    cleaned: list[str] = []
    skip_next = False
    for flag in flags:
        if skip_next:
            skip_next = False
            continue
        if flag == "--save-trajectory-interval":
            skip_next = True
            continue
        if flag.startswith("--save-trajectory-interval="):
            continue
        cleaned.append(flag)
    return tuple(cleaned)


def build_commands(
    *,
    workspace: Path,
    env_values: dict[str, str],
    manifest_path: Path,
    snapshot_root: Path,
    checkpoint_flag: str | None,
) -> tuple[LaunchCommand, ...]:
    engine_bin = Path(env_values["PRISM_ENGINE_BIN"])
    engine_flags = without_save_interval(tuple(shlex.split(env_values["PRISM_ENGINE_FLAGS"])))
    campaign_id = env_values["PRISM_CAMPAIGN_ID"]
    base_seed = int(env_values["PRISM_BASE_SEED"])
    commands: list[LaunchCommand] = []
    for target in unique_targets(manifest_path):
        topology = topology_path(workspace, target.condition_id)
        checkpoints = discover_checkpoints(workspace, campaign_id, target)
        nma_modes = topology.with_name(f"{topology.stem.removesuffix('.topology')}_nma_modes.json")
        nma_path = nma_modes if nma_modes.exists() else None
        output_dir = snapshot_root / target.condition_id / f"replica_{target.replica_id}"
        seed = base_seed + target.replica_id
        command = [
            str(engine_bin),
            "-t",
            str(topology),
            "-o",
            str(output_dir),
            "--replica-seed",
            str(seed),
            "--ensemble-campaign-id",
            campaign_id,
            "--ensemble-base-seed",
            str(base_seed),
            "--ensemble-replica-id",
            str(target.replica_id),
            *engine_flags,
        ]
        if nma_path is not None:
            command.extend(["--nma-perturb", str(nma_path)])
        if checkpoint_flag is not None:
            for checkpoint in checkpoints:
                command.extend([checkpoint_flag, str(checkpoint.path)])
        command.extend(["--reintegrate", "--manifest", str(manifest_path), "--save-trajectory-interval", "1"])
        completion = completion_path(workspace, campaign_id, target)
        status, run_id = completion_metadata(completion)
        commands.append(
            LaunchCommand(
                target=target,
                topology_json=topology,
                topology_sha256=sha256_file(topology),
                nma_modes_json=nma_path,
                nma_modes_sha256=sha256_file(nma_path) if nma_path is not None else None,
                checkpoint_files=checkpoints,
                replica_seed=seed,
                output_dir=output_dir,
                command=tuple(command),
                completion_json=completion,
                completion_status=status,
                original_run_id=run_id,
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


def command_to_json(command: LaunchCommand) -> JsonObject:
    return {
        "condition_id": command.target.condition_id,
        "replica_id": command.target.replica_id,
        "replica_seed": command.replica_seed,
        "topology_json": str(command.topology_json),
        "topology_sha256": command.topology_sha256,
        "nma_modes_json": str(command.nma_modes_json) if command.nma_modes_json is not None else None,
        "nma_modes_sha256": command.nma_modes_sha256,
        "checkpoint_files": [
            {
                "path": str(checkpoint.path),
                "sha256": checkpoint.sha256,
                "size_bytes": checkpoint.size_bytes,
                "stream_id": checkpoint.stream_id,
            }
            for checkpoint in command.checkpoint_files
        ],
        "output_dir": str(command.output_dir),
        "completion_json": str(command.completion_json) if command.completion_json is not None else None,
        "completion_status": command.completion_status,
        "original_run_id": command.original_run_id,
        "command": list(command.command),
    }


def write_parity_record(
    *,
    path: Path,
    workspace: Path,
    run_script: Path,
    runtime_env: Path,
    manifest_path: Path,
    snapshot_root: Path,
    env_values: dict[str, str],
    commands: tuple[LaunchCommand, ...],
    engine_supports_reintegrate: bool,
    engine_supports_save_trajectory_interval: bool,
    checkpoint_flag: str | None,
) -> None:
    engine_bin = Path(env_values["PRISM_ENGINE_BIN"])
    payload: JsonObject = {
        "campaign_id": "glp1r_aleniglipron",
        "phase": "phase_2c_targeted_reintegration",
        "workspace": str(workspace),
        "run_script": str(run_script),
        "run_script_sha256": sha256_file(run_script),
        "runtime_env": str(runtime_env),
        "runtime_env_sha256": sha256_file(runtime_env),
        "engine_binary": str(engine_bin),
        "engine_binary_sha256": sha256_file(engine_bin),
        "engine_supports_reintegrate": engine_supports_reintegrate,
        "engine_supports_save_trajectory_interval": engine_supports_save_trajectory_interval,
        "checkpoint_flag": checkpoint_flag,
        "checkpoint_suffixes_scanned": list(CHECKPOINT_SUFFIXES),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "snapshot_root": str(snapshot_root),
        "base_seed": int(env_values["PRISM_BASE_SEED"]),
        "engine_flags": env_values["PRISM_ENGINE_FLAGS"],
        "commands": [command_to_json(command) for command in commands],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_commands(commands: tuple[LaunchCommand, ...], env: dict[str, str], log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    for command in commands:
        command.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{command.target.condition_id}_replica_{command.target.replica_id}.log"
        emit(f"launching {command.target.condition_id} replica={command.target.replica_id} seed={command.replica_seed}")
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("$ " + shlex.join(command.command) + "\n")
            result = subprocess.run(
                list(command.command),
                cwd=command.output_dir,
                env=env,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"engine reintegration failed for {command.target.condition_id} replica={command.target.replica_id}; "
                f"log={log_path}"
            )


def validate_frames(snapshot_root: Path) -> list[Path]:
    if not snapshot_root.exists():
        raise FileNotFoundError(snapshot_root)
    frames = sorted(path for path in snapshot_root.rglob("*") if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES)
    if not frames:
        raise FileNotFoundError(f"no PDB/CIF/DCD frames emitted under {snapshot_root}")
    return frames


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace)
    run_script = Path(args.run_script)
    runtime_env = Path(args.runtime_env)
    manifest_path = Path(args.manifest).resolve()
    snapshot_root = Path(args.snapshot_root)
    for required in (workspace, run_script, runtime_env, manifest_path):
        if not required.exists():
            raise FileNotFoundError(required)

    env_values = source_runtime_env(runtime_env)
    engine_bin = Path(env_values["PRISM_ENGINE_BIN"])
    if not engine_bin.exists():
        raise FileNotFoundError(engine_bin)
    help_text = engine_help(engine_bin)
    supports_reintegrate = "--reintegrate" in help_text
    supports_save_interval = "--save-trajectory-interval" in help_text
    checkpoint_flag = supported_checkpoint_flag(help_text)
    commands = build_commands(
        workspace=workspace,
        env_values=env_values,
        manifest_path=manifest_path,
        snapshot_root=snapshot_root,
        checkpoint_flag=checkpoint_flag,
    )
    write_parity_record(
        path=Path(args.parity_record),
        workspace=workspace,
        run_script=run_script,
        runtime_env=runtime_env,
        manifest_path=manifest_path,
        snapshot_root=snapshot_root,
        env_values=env_values,
        commands=commands,
        engine_supports_reintegrate=supports_reintegrate,
        engine_supports_save_trajectory_interval=supports_save_interval,
        checkpoint_flag=checkpoint_flag,
    )
    emit(f"parity_record={args.parity_record}")
    emit(f"engine_sha256={sha256_file(engine_bin)}")
    for command in commands:
        emit(
            f"target={command.target.condition_id} replica={command.target.replica_id} "
            f"streams={','.join(str(stream_id) for stream_id in command.target.stream_ids)} "
            f"seed={command.replica_seed} topology_sha256={command.topology_sha256} "
            f"checkpoints={len(command.checkpoint_files)}"
        )
    if args.dry_run:
        emit("dry_run=true")
        return 0
    missing_checkpoint_targets = [
        f"{command.target.condition_id}/replica_{command.target.replica_id}"
        for command in commands
        if not command.checkpoint_files
    ]
    if missing_checkpoint_targets:
        emit("WARN: no checkpoints for: " + ",".join(missing_checkpoint_targets))
        emit("WARN: de novo reintegration — fresh simulation at trigger windows")
    if checkpoint_flag is None:
        emit("WARN: engine does not advertise checkpoint flag — de novo mode")
    if not supports_save_interval:
        emit("ERROR: pinned engine binary does not advertise --save-trajectory-interval")
        emit("ERROR: parity record was written; engine subprocesses were not started")
        return 5
    if not supports_reintegrate and not bool(args.allow_unsupported_reintegrate):
        emit("ERROR: pinned engine binary does not advertise --reintegrate; refusing unsupported reintegration launch")
        emit("ERROR: parity record was written; engine subprocesses were not started")
        return 2

    run_commands(commands, launch_env(env_values), CAMPAIGN_DIR / "logs/phase_2c_reintegration")
    frames = validate_frames(snapshot_root)
    emit(f"validated_frames={len(frames)} snapshot_root={snapshot_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
