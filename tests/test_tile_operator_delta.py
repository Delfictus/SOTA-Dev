from __future__ import annotations

import pytest
import torch

from prism_dstw.gflownet.bsr_operator_state import make_demo_operator
from prism_dstw.gflownet.captured_graph_tiles import CapturedGraphTileRegistry


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for captured tile runtime")


def test_tile_delta_changes_only_affected_blocks_and_preserves_rows() -> None:
    state = make_demo_operator(5, device="cuda")
    tile = CapturedGraphTileRegistry.demo(tile_count=1).get("captured_tile_000")
    delta = tile.to_delta(state=state)
    before = state.values.detach().clone()
    old_hash = state.state_hash
    changed_rows, changed_blocks = delta.apply_to_bsr_state(state)
    after = state.values.detach().clone()
    changed = torch.nonzero((after - before).abs().reshape(after.shape[0], -1).sum(dim=1) > 1.0e-12).flatten()
    assert set(changed.detach().cpu().tolist()).issubset(set(changed_blocks))
    assert changed_rows
    assert state.state_hash != old_hash
    assert bool((state.values >= 0).all().item())
    assert bool(torch.allclose(state.dense_for_gpu_solve().sum(dim=1), torch.ones(state.shape[0], device="cuda", dtype=torch.float64)))


def test_duplicate_tile_sequence_gives_same_state_hash() -> None:
    registry = CapturedGraphTileRegistry.demo(tile_count=2)
    hashes: list[str] = []
    for _ in range(2):
        state = make_demo_operator(5, device="cuda")
        for tile_id in ["captured_tile_000", "captured_tile_001"]:
            registry.get(tile_id).to_delta(state=state).apply_to_bsr_state(state)
        hashes.append(state.state_hash)
    assert hashes[0] == hashes[1]

