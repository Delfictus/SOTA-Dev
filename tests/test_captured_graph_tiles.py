from __future__ import annotations

import pytest
import torch
import json

from prism_dstw.gflownet.bsr_operator_state import make_demo_operator
from prism_dstw.gflownet.captured_graph_tiles import CapturedGraphTileRegistry, build_tile
from prism_dstw.gflownet.tile_operator_delta import TileDeltaError


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for captured tile runtime")


def test_captured_tile_registry_rejects_invalid_bsr_blocks() -> None:
    state = make_demo_operator(2, device="cuda")
    tile = build_tile(
        tile_id="bad_tile",
        tile_type="holo_tile_fusion",
        topology_region="TE_HUBS",
        perturbation_family="SEVERING_PROBE",
        affected_voxel_ids=[1],
        affected_state_ids=[1],
        affected_bsr_blocks=[99999],
        delta_values=[[[0.1]]],
        restricted_operator_target="W_without_arr(Pi)",
        capture_shape_bucket="bad",
        topology_delta="topology",
        basin_delta="basin",
        restricted_operator_hash="restricted",
        c6_operator_hash="c6",
    )
    registry = CapturedGraphTileRegistry(tiles={"bad_tile": tile})
    with pytest.raises(TileDeltaError, match="outside allowed BSR block"):
        registry.validate_for_state(state)


def test_registry_json_roundtrip_preserves_hashes(tmp_path: object) -> None:
    path = tmp_path / "captured_tile_registry.json"  # type: ignore[operator]
    registry = CapturedGraphTileRegistry.demo(tile_count=3)
    registry.write_json(path)
    loaded = CapturedGraphTileRegistry.from_json(path)
    assert set(loaded.tiles) == set(registry.tiles)
    assert loaded.get("captured_tile_000").captured_graph_tile_hash == registry.get("captured_tile_000").captured_graph_tile_hash


def test_registry_json_does_not_trust_persisted_cuda_graph_id(tmp_path: object) -> None:
    path = tmp_path / "captured_tile_registry.json"  # type: ignore[operator]
    CapturedGraphTileRegistry.demo(tile_count=1).write_json(path)
    payload = json.loads(path.read_text())
    payload["tiles"][0]["cuda_graph_id"] = "cuda_graph::forged_without_capture"
    path.write_text(json.dumps(payload))
    loaded = CapturedGraphTileRegistry.from_json(path)
    assert loaded.get("captured_tile_000").cuda_graph_id is None
