from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch

from scripts.run_log_subtb_spectral_gflownet import _c6_payload_hash, _operator_payload_hash, parse_args, run_training
from prism_dstw.gflownet.captured_graph_tiles import CapturedGraphTileRegistry, build_tile
from prism_dstw.gflownet.cuda_graph_tile_runtime import TileCaptureError


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for captured tile runtime")


def _write_test_operator(path: Path, *, tile_count: int = 4) -> tuple[str, str]:
    state_count = max(tile_count + 2, 8)
    rows: list[list[dict[str, float | int]]] = []
    basin_weights: list[float] = []
    for row in range(state_count):
        self_weight = 0.62 + (0.02 * float(row % 3))
        rows.append(
            [
                {"col": row, "value": self_weight},
                {"col": (row + 1) % state_count, "value": 1.0 - self_weight},
            ]
        )
        basin_weights.append(1.0 + (0.1 * float(row % 4)))
    operator_hash = _operator_payload_hash(rows, state_count)
    c6_operator_hash = _c6_payload_hash(
        basin_weights=basin_weights,
        operator_hash=operator_hash,
        solver="restricted_dirichlet_gpu_v1",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "prism.log_subtb.restricted_operator.v1",
                "id": "test_restricted_operator",
                "provenance_class": "L3_DERIVED",
                "operator_generation_owner": "prism-forge/log_subtb_tile_guard",
                "restricted_operator_target": "W_without_arr(Pi)",
                "c6_operator_id": "restricted_dirichlet_c6_v1",
                "c6_reward_solver": "restricted_dirichlet_gpu_v1",
                "source_artifacts": ["unit_test"],
                "state_count": state_count,
                "blocksize": [1, 1],
                "dtype": "float64",
                "rows": rows,
                "basin_weights": basin_weights,
                "operator_hash": operator_hash,
                "c6_operator_hash": c6_operator_hash,
            },
            sort_keys=True,
        )
    )
    return operator_hash, c6_operator_hash


def _write_test_registry(path: Path, *, operator_hash: str, c6_operator_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = {}
    for index in range(4):
        tile = build_tile(
            tile_id=f"test_tile_{index}",
            tile_type="holo_tile_fusion",
            topology_region="TE_HUBS",
            perturbation_family="SEVERING_PROBE",
            affected_voxel_ids=[100 + index],
            affected_state_ids=[index, index + 1],
            affected_bsr_blocks=[2 * index, (2 * index) + 1],
            delta_values=[[[0.0125]], [[-0.0040]]],
            restricted_operator_target="W_without_arr(Pi)",
            capture_shape_bucket="rows1_blocks2_block1_float64",
            topology_delta=f"topology_{index}",
            basin_delta=f"basin_{index}",
            restricted_operator_hash=operator_hash,
            c6_operator_hash=c6_operator_hash,
        )
        tiles[tile.tile_id] = tile
    CapturedGraphTileRegistry(tiles=tiles).write_json(path)


def _args(output_root: Path) -> Any:
    import sys

    old_argv = sys.argv
    sys.argv = [
        "run_log_subtb_spectral_gflownet.py",
        "--campaign",
        "glp1r_aleniglipron",
        "--epochs",
        "5",
        "--capture-bucket-size",
        "2",
        "--warmup-capture-iterations",
        "1",
        "--max-tiles",
        "4",
        "--mode",
        "synthetic",
        "--output-root",
        str(output_root),
        "--tile-registry-path",
        str(output_root / "captured_graph_tiles" / "source_registry.json"),
        "--restricted-operator-path",
        str(output_root / "captured_graph_tiles" / "restricted_c6_operator_state.json"),
    ]
    try:
        return parse_args()
    finally:
        sys.argv = old_argv


def test_log_subtb_training_uses_captured_tiles_and_improves(tmp_path: Path) -> None:
    output_root = tmp_path / "subtb_spectral"
    operator_hash, c6_operator_hash = _write_test_operator(output_root / "captured_graph_tiles" / "restricted_c6_operator_state.json")
    _write_test_registry(
        output_root / "captured_graph_tiles" / "source_registry.json",
        operator_hash=operator_hash,
        c6_operator_hash=c6_operator_hash,
    )
    summary = run_training(_args(output_root))
    assert summary["captured_graph_tile_runtime_enabled"] is True
    assert summary["captured_graph_replay_count"] > 0
    assert summary["gpu_solve_count"] > 0
    assert summary["uncaptured_fallback_count"] == 0
    assert summary["cpu_solve_count"] == 0
    assert summary["production_cpu_fallback_used"] is False
    assert summary["reward_events_derive_from_captured_tile_deltas"] is True
    metrics = pl.read_parquet(output_root / "subtb_training_metrics.parquet")
    assert metrics.height == 5
    assert metrics.get_column("tb_loss")[metrics.height - 1] < metrics.get_column("tb_loss")[0]
    assert set(metrics.get_column("reward_event_source").unique().to_list()) == {"captured_tile_delta_hashes"}


def test_run_manifest_has_required_capture_fields(tmp_path: Path) -> None:
    output_root = tmp_path / "subtb_spectral"
    operator_hash, c6_operator_hash = _write_test_operator(output_root / "captured_graph_tiles" / "restricted_c6_operator_state.json")
    _write_test_registry(
        output_root / "captured_graph_tiles" / "source_registry.json",
        operator_hash=operator_hash,
        c6_operator_hash=c6_operator_hash,
    )
    run_training(_args(output_root))
    manifest_path = output_root / "subtb_run_manifest.json"
    payload = json.loads(manifest_path.read_text())
    for key in [
        "captured_graph_tile_runtime_enabled",
        "captured_tile_count",
        "cuda_graph_count",
        "capture_bucket_count",
        "uncaptured_tile_count",
        "captured_graph_replay_count",
        "uncaptured_fallback_count",
        "bsr_operator_device",
        "production_cpu_fallback_used",
    ]:
        assert key in payload
    assert payload["bsr_operator_device"].startswith("cuda")


def test_full_mode_missing_registry_fails_closed(tmp_path: Path) -> None:
    args = _args(tmp_path / "missing_registry_case")
    args.mode = "full"
    args.tile_registry_path = str(tmp_path / "missing_registry_case" / "does_not_exist.json")
    _write_test_operator(Path(str(args.restricted_operator_path)))
    with pytest.raises(TileCaptureError, match="captured graph tile registry is missing"):
        run_training(args)
