#!/usr/bin/env python3
"""Run Log-SubTB spectral GFlowNet over captured graph tile actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import polars as pl
import torch
from torch import Tensor

from prism_dstw.gflownet.bsr_operator_state import BSROperatorState, make_demo_operator
from prism_dstw.gflownet.captured_graph_tiles import (
    CapturedGraphTileRegistry,
)
from prism_dstw.gflownet.cuda_graph_tile_runtime import CUDAGraphTileRuntime, TileCaptureError
from prism_dstw.gflownet.spectral_reward_manager import SpectralRewardManager, log_subtb_loss


TRACK_B_ROOT = Path("campaigns/glp1r_aleniglipron/track_b_chronological")
SUBTB_ROOT = TRACK_B_ROOT / "subtb_spectral"
CAPTURE_DIR = SUBTB_ROOT / "captured_graph_tiles"
FULL_OPERATOR_PATH = CAPTURE_DIR / "restricted_c6_operator_state.json"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_json_hash(path: Path) -> str:
    payload = json.loads(path.read_text())
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _operator_payload_hash(rows: list[list[dict[str, float | int]]], state_count: int) -> str:
    payload = json.dumps(
        {
            "restricted_operator_target": "W_without_arr(Pi)",
            "rows": rows,
            "state_count": state_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _c6_payload_hash(*, basin_weights: list[float], operator_hash: str, solver: str) -> str:
    payload = json.dumps(
        {
            "basin_weights": basin_weights,
            "operator_hash": operator_hash,
            "solver": solver,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _variant_panel_path(campaign: str, override: str | None) -> Path:
    if override:
        return Path(override)
    return Path("campaigns") / campaign / "track_b_chronological" / "genealogical_variant_panel.json"


def _guard_binary(*, required: bool) -> Path | None:
    for path in [Path("target/release/log_subtb_tile_guard"), Path("target/debug/log_subtb_tile_guard")]:
        if path.exists():
            return path
    if required:
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: Rust tile boundary guard binary is missing")
    return None


def verify_full_mode_rust_canonical_artifacts(
    args: argparse.Namespace,
    *,
    registry_path: Path,
    operator_path: Path,
) -> dict[str, str]:
    if not registry_path.exists():
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: captured graph tile registry is missing")
    if not operator_path.exists():
        raise TileCaptureError(f"LOG_SUBTB_BLOCKED_TILE_CAPTURE: restricted operator artifact is missing: {operator_path}")
    panel_path = _variant_panel_path(str(args.campaign), args.variant_panel_path)
    if not panel_path.exists():
        raise TileCaptureError(f"LOG_SUBTB_BLOCKED_TILE_CAPTURE: missing variant panel {panel_path}")
    guard_bin = _guard_binary(required=True)
    scratch_parent = Path("/mnt/storage/tmp")
    if scratch_parent.exists():
        tmp_context = tempfile.TemporaryDirectory(prefix="log_subtb_rust_canonical_", dir=str(scratch_parent))
    else:
        tmp_context = tempfile.TemporaryDirectory(prefix="log_subtb_rust_canonical_")
    with tmp_context as tmp_dir:
        expected_registry = Path(tmp_dir) / "captured_tile_registry_source.json"
        expected_operator = Path(tmp_dir) / "restricted_c6_operator_state.json"
        subprocess.run(
            [
                str(guard_bin),
                "--build-from-variant-panel",
                str(panel_path),
                "--registry-output",
                str(expected_registry),
                "--operator-output",
                str(expected_operator),
                "--max-tiles",
                str(int(args.max_tiles)),
            ],
            check=True,
        )
        supplied_registry_hash = _canonical_json_hash(registry_path)
        expected_registry_hash = _canonical_json_hash(expected_registry)
        supplied_operator_hash = _canonical_json_hash(operator_path)
        expected_operator_hash = _canonical_json_hash(expected_operator)
        if supplied_registry_hash != expected_registry_hash:
            raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: registry is not canonical Rust output")
        if supplied_operator_hash != expected_operator_hash:
            raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: restricted operator is not canonical Rust output")
    return {
        "variant_panel_path": str(panel_path),
        "variant_panel_sha256": _sha256_file(panel_path),
        "rust_canonical_registry_hash": supplied_registry_hash,
        "rust_canonical_operator_hash": supplied_operator_hash,
    }


def load_restricted_operator_artifact(path: Path) -> tuple[BSROperatorState, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "prism.log_subtb.restricted_operator.v1":
        raise TileCaptureError(f"LOG_SUBTB_BLOCKED_TILE_CAPTURE: unsupported restricted operator schema in {path}")
    if payload.get("operator_generation_owner") != "prism-forge/log_subtb_tile_guard":
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: restricted operator was not generated by prism-forge")
    if payload.get("restricted_operator_target") != "W_without_arr(Pi)":
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: unsupported restricted operator target")
    rows_raw = payload.get("rows")
    state_count = int(payload.get("state_count", 0))
    if not isinstance(rows_raw, list) or state_count <= 0 or len(rows_raw) != state_count:
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: malformed restricted operator rows")
    rows: list[list[dict[str, float | int]]] = []
    dense = torch.zeros((state_count, state_count), device="cuda", dtype=torch.float64)
    for row_index, row in enumerate(rows_raw):
        if not isinstance(row, list) or not row:
            raise TileCaptureError(f"LOG_SUBTB_BLOCKED_OPERATOR_STOCHASTICITY: empty operator row {row_index}")
        canonical_row: list[dict[str, float | int]] = []
        for entry in row:
            if not isinstance(entry, dict):
                raise TileCaptureError(f"LOG_SUBTB_BLOCKED_TILE_CAPTURE: malformed operator row entry {row_index}")
            col = int(entry.get("col", -1))
            value = float(entry.get("value", float("nan")))
            if col < 0 or col >= state_count or not math.isfinite(value) or value < 0.0:
                raise TileCaptureError(f"LOG_SUBTB_BLOCKED_OPERATOR_STOCHASTICITY: invalid operator entry row {row_index}")
            dense[row_index, col] = value
            canonical_row.append({"col": col, "value": value})
        rows.append(canonical_row)
    expected_hash = _operator_payload_hash(rows, state_count)
    if payload.get("operator_hash") != expected_hash:
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: restricted operator hash mismatch")
    solver = str(payload.get("c6_reward_solver", ""))
    basin_raw = payload.get("basin_weights")
    if solver != "restricted_dirichlet_gpu_v1":
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: unsupported C6 reward solver")
    if not isinstance(basin_raw, list) or len(basin_raw) != state_count:
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: malformed C6 basin weights")
    basin_weights = [float(value) for value in basin_raw]
    if any((not math.isfinite(value)) or value < 0.0 for value in basin_weights):
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: invalid C6 basin weight")
    expected_c6_hash = _c6_payload_hash(
        basin_weights=basin_weights,
        operator_hash=expected_hash,
        solver=solver,
    )
    if payload.get("c6_operator_hash") != expected_c6_hash:
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: C6 operator hash mismatch")
    state = BSROperatorState.from_dense(dense, device="cuda", dtype=torch.float64)
    state.update_hashes(extra_state=f"restricted_operator_artifact:{expected_hash}")
    return state, payload


def load_or_build_registry(args: argparse.Namespace, *, output_registry: Path) -> CapturedGraphTileRegistry:
    registry_path = Path(args.tile_registry_path) if args.tile_registry_path else output_registry
    if registry_path.exists():
        return CapturedGraphTileRegistry.from_json(registry_path)
    if args.mode == "full":
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: captured graph tile registry is missing")
    registry = CapturedGraphTileRegistry.demo(tile_count=int(args.max_tiles))
    registry.write_json(registry_path)
    return registry


def _memory_metrics() -> dict[str, float]:
    return {
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024.0 * 1024.0),
        "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024.0 * 1024.0),
        "max_gpu_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise TileCaptureError("LOG_SUBTB_BLOCKED_GPU_PLACEMENT: CUDA is required for captured graph tile runtime")
    torch.manual_seed(int(args.seed))
    output_root = (
        Path(str(args.output_root))
        if args.output_root
        else (
            SUBTB_ROOT
            if str(args.campaign) == "glp1r_aleniglipron"
            else Path("campaigns") / str(args.campaign) / "track_b_chronological" / "subtb_spectral"
        )
    )
    capture_dir = output_root / "captured_graph_tiles"
    default_registry = (
        capture_dir / f"captured_tile_registry_source_{int(args.max_tiles)}.json"
        if args.mode == "full"
        else capture_dir / "captured_tile_registry.json"
    )
    registry_path = Path(args.tile_registry_path) if args.tile_registry_path else default_registry
    operator_path = Path(str(args.restricted_operator_path)) if args.restricted_operator_path else FULL_OPERATOR_PATH
    rust_canonical_evidence: dict[str, str] = {}
    if args.mode == "full":
        rust_canonical_evidence = verify_full_mode_rust_canonical_artifacts(
            args,
            registry_path=registry_path,
            operator_path=operator_path,
        )
    registry = load_or_build_registry(args, output_registry=registry_path)
    if len(registry.tiles) == 0:
        raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: captured graph tile registry has zero tiles")

    device = torch.device("cuda")
    operator_artifact: dict[str, Any]
    if args.mode == "full":
        state, operator_artifact = load_restricted_operator_artifact(operator_path)
        for tile in registry.tiles.values():
            if tile.restricted_operator_hash != operator_artifact.get("operator_hash"):
                raise TileCaptureError(f"LOG_SUBTB_BLOCKED_TILE_CAPTURE: {tile.tile_id} targets a different restricted operator")
            if tile.c6_operator_hash != operator_artifact.get("c6_operator_hash"):
                raise TileCaptureError(f"LOG_SUBTB_BLOCKED_TILE_CAPTURE: {tile.tile_id} targets a different C6 operator")
        basin_weights = torch.tensor(
            [float(value) for value in operator_artifact["basin_weights"]],
            device=device,
            dtype=torch.float64,
        )
    else:
        state = make_demo_operator(len(registry.tiles), device=device)
        operator_artifact = {
            "schema_version": "prism.log_subtb.restricted_operator.synthetic.v1",
            "id": "synthetic_demo_operator",
            "operator_generation_owner": "python.synthetic_test_fixture",
            "restricted_operator_target": "W_without_arr(Pi)",
            "operator_hash": state.operator_hash,
            "c6_operator_hash": _sha256_text(f"synthetic-c6|{state.operator_hash}"),
            "provenance_class": "L2_PROJECTED",
        }
        basin_weights = None
    if state.device.type != "cuda":
        raise TileCaptureError("LOG_SUBTB_BLOCKED_GPU_PLACEMENT: restricted operator is not cuda")
    runtime = CUDAGraphTileRuntime(
        state=state,
        allow_eager_cuda_tiles=bool(args.allow_eager_cuda_tiles),
        require_all_production_tiles_captured=bool(args.require_all_production_tiles_captured),
        warmup_iterations=int(args.warmup_capture_iterations),
    )
    capture_ids = runtime.capture_registry(registry)
    captured_registry = registry.with_captures(capture_ids)
    captured_registry.write_json(capture_dir / "captured_tile_registry.json")
    rust_boundary_report = capture_dir / "rust_tile_boundary_report.json"
    guard_bin = _guard_binary(required=args.mode == "full")
    if guard_bin is not None and guard_bin.exists():
        subprocess.run(
            [
                str(guard_bin),
                "--registry",
                str(capture_dir / "captured_tile_registry.json"),
                "--operator-block-count",
                str(int(state.values.shape[0])),
                "--output",
                str(rust_boundary_report),
            ],
            check=True,
        )
    tile_manifest = captured_registry.manifest(
        capture_ids=capture_ids,
        state=state,
        warmup_iterations=int(args.warmup_capture_iterations),
    )
    _write_json(capture_dir / "tile_capture_manifest.json", tile_manifest)

    tile_ids = list(captured_registry.tiles)
    action_logits = torch.nn.Parameter(torch.zeros(len(tile_ids), device=device, dtype=torch.float64))
    optimizer = torch.optim.AdamW([action_logits], lr=0.03)
    reward_manager = SpectralRewardManager(state=state, telemetry=runtime.telemetry, basin_weights=basin_weights)
    initial_values = state.values.detach().clone()
    rows: list[dict[str, Any]] = []
    sequence_len = min(max(int(args.capture_bucket_size), 1), len(tile_ids), 4)

    for epoch in range(int(args.epochs)):
        state.values.copy_(initial_values)
        state.update_hashes(extra_state=f"epoch:{epoch}:reset")
        runtime.reset_trajectory_history()
        optimizer.zero_grad(set_to_none=True)
        log_probs = torch.log_softmax(action_logits, dim=0)
        ranked_actions = torch.argsort(log_probs, descending=True)
        if args.mode == "full":
            offset = epoch % len(tile_ids)
            selected = torch.stack(
                [ranked_actions[(offset + action_index) % len(tile_ids)] for action_index in range(sequence_len)]
            )
        else:
            selected = ranked_actions[:sequence_len]
        forward_log_probs: list[Tensor] = []
        backward_log_probs: list[Tensor] = []
        log_rewards: list[Tensor] = []
        cache_hits = 0
        tile_sequence: list[str] = []
        for selected_index in selected.tolist():
            tile_id = tile_ids[int(selected_index)]
            tile_sequence.append(tile_id)
            event = runtime.replay_tile(tile_id)
            reward_record = reward_manager.reward_for_event(event)
            cache_hits += int(reward_record.cache_hit)
            forward_log_probs.append(log_probs[int(selected_index)])
            backward_log_probs.append(torch.full((), -math.log(float(sequence_len)), device=device, dtype=torch.float64))
            log_rewards.append(reward_record.log_reward)
        loss = log_subtb_loss(forward_log_probs, backward_log_probs, log_rewards)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        telemetry = runtime.production_invariant_report()
        mem = _memory_metrics()
        rows.append(
            {
                "epoch": epoch + 1,
                "tb_loss": float(loss.detach().cpu().item()),
                "captured_tile_replay_count": int(runtime.telemetry.captured_graph_replay_count),
                "uncaptured_tile_fallback_count": int(runtime.telemetry.uncaptured_fallback_count),
                "cuda_graph_replay_ms": float(telemetry["avg_tile_exec_ms"]),
                "bsr_update_ms": float(telemetry["avg_tile_exec_ms"]),
                "row_renorm_ms": 0.0,
                "operator_hash_update_ms": 0.0,
                "reward_cache_hit_rate": reward_manager.cache_hit_rate,
                "c6_solve_count": int(runtime.telemetry.gpu_solve_count),
                "gpu_solve_count": int(runtime.telemetry.gpu_solve_count),
                "cpu_solve_count": int(runtime.telemetry.cpu_solve_count),
                "gpu_memory_allocated_mb": mem["gpu_memory_allocated_mb"],
                "gpu_memory_reserved_mb": mem["gpu_memory_reserved_mb"],
                "max_gpu_memory_allocated_mb": mem["max_gpu_memory_allocated_mb"],
                "state_hash": state.state_hash,
                "operator_hash": state.operator_hash,
                "tile_sequence": ",".join(tile_sequence),
                "reward_event_source": "captured_tile_delta_hashes",
                "cache_hits_this_epoch": cache_hits,
            }
        )

    metrics_path = output_root / "subtb_training_metrics.parquet"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(metrics_path)
    summary = runtime.production_invariant_report()
    summary.update(
        {
            "schema_version": "prism.log_subtb.spectral_run_manifest.v1",
            "mode": str(args.mode),
            "captured_graph_tile_runtime_enabled": True,
            "captured_tile_count": int(tile_manifest["captured_tile_count"]),
            "cuda_graph_count": int(tile_manifest["cuda_graph_count"]),
            "capture_bucket_count": int(tile_manifest["capture_bucket_count"]),
            "uncaptured_tile_count": int(tile_manifest["uncaptured_tile_count"]),
            "bsr_operator_device": str(state.device),
            "bsr_operator_dtype": str(state.dtype).replace("torch.", ""),
            "production_cpu_fallback_used": runtime.telemetry.cpu_solve_count > 0
            or runtime.telemetry.uncaptured_fallback_count > 0,
            "subtb_training_metrics": str(metrics_path),
            "tile_registry_path": str(capture_dir / "captured_tile_registry.json"),
            "tile_capture_manifest": str(capture_dir / "tile_capture_manifest.json"),
            "rust_tile_boundary_guard": str(rust_boundary_report),
            "rust_tile_boundary_verified": rust_boundary_report.exists(),
            "spectral_reward_manager": reward_manager.manifest(),
            "reward_events_derive_from_captured_tile_deltas": reward_manager.event_count > 0,
            "final_epoch": rows[-1] if rows else None,
            "restricted_operator_source": operator_artifact.get("id", "synthetic_demo_operator"),
            "restricted_operator_path": str(args.restricted_operator_path or FULL_OPERATOR_PATH)
            if args.mode == "full"
            else "synthetic_fixture",
            "restricted_operator_hash": operator_artifact.get("operator_hash"),
            "c6_operator_hash": operator_artifact.get("c6_operator_hash"),
            "c6_reward_solver": "restricted_dirichlet_gpu_v1"
            if args.mode == "full"
            else "synthetic_restricted_dirichlet_gpu_v1",
            "operator_generation_owner": operator_artifact.get("operator_generation_owner"),
            "rust_canonical_artifact_verification": rust_canonical_evidence,
        }
    )
    if args.mode == "full":
        if summary["captured_graph_replay_count"] <= 0:
            raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: captured_graph_replay_count = 0")
        if summary["gpu_solve_count"] <= 0:
            raise TileCaptureError("LOG_SUBTB_BLOCKED_GPU_PLACEMENT: gpu_solve_count = 0")
        if summary["uncaptured_fallback_count"] != 0:
            raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: uncaptured fallback used")
        if summary["cpu_solve_count"] != 0:
            raise TileCaptureError("LOG_SUBTB_BLOCKED_GPU_PLACEMENT: CPU solve used")
        if not summary["reward_events_derive_from_captured_tile_deltas"]:
            raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: reward manager bypassed captured tile events")
        if not summary["rust_tile_boundary_verified"]:
            raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: Rust tile boundary guard did not verify registry")
    summary["status"] = "LOG_SUBTB_CAPTURED_TILE_RUNTIME_VERIFIED"
    manifest_path = output_root / "subtb_run_manifest.json"
    _write_json(manifest_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="glp1r_aleniglipron")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--loss-type", default="subtrajectory_balance")
    parser.add_argument("--reward-type", default="c6_spectral_log")
    parser.add_argument("--detach-spectral-reward", type=lambda value: value.lower() == "true", default=True)
    parser.add_argument("--intermediate-spectral-rewards", default="event_triggered_intermediate")
    parser.add_argument("--use-captured-graph-tiles", type=lambda value: value.lower() == "true", default=True)
    parser.add_argument("--tile-registry-path")
    parser.add_argument("--restricted-operator-path")
    parser.add_argument("--c6-operator-path")
    parser.add_argument("--capture-bucket-size", type=int, default=3)
    parser.add_argument("--warmup-capture-iterations", type=int, default=3)
    parser.add_argument("--allow-eager-cuda-tiles", type=lambda value: value.lower() == "true", default=False)
    parser.add_argument("--require-all-production-tiles-captured", type=lambda value: value.lower() == "true", default=True)
    parser.add_argument("--build-registry-if-missing", type=lambda value: value.lower() == "true", default=False)
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--mode", choices=["full", "synthetic"], default="full")
    parser.add_argument("--variant-panel-path")
    parser.add_argument("--seed", type=int, default=25016)
    parser.add_argument("--output-root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.loss_type) != "subtrajectory_balance":
        raise SystemExit("LOG_SUBTB_BLOCKED_CONFIG: --loss-type must be subtrajectory_balance")
    if str(args.reward_type) != "c6_spectral_log":
        raise SystemExit("LOG_SUBTB_BLOCKED_CONFIG: --reward-type must be c6_spectral_log")
    if str(args.intermediate_spectral_rewards) != "event_triggered_intermediate":
        raise SystemExit("LOG_SUBTB_BLOCKED_CONFIG: reward events must be event_triggered_intermediate")
    if not bool(args.use_captured_graph_tiles):
        raise SystemExit("LOG_SUBTB_BLOCKED_TILE_CAPTURE: --use-captured-graph-tiles is required")
    summary = run_training(args)
    print(
        "LOG_SUBTB_CAPTURED_TILE_RUNTIME "
        f"status={summary['status']} "
        f"captured_graph_replay_count={summary['captured_graph_replay_count']} "
        f"gpu_solve_count={summary['gpu_solve_count']} "
        f"uncaptured_fallback_count={summary['uncaptured_fallback_count']} "
        f"cpu_solve_count={summary['cpu_solve_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
