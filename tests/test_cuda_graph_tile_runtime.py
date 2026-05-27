from __future__ import annotations

import pytest
import torch

from prism_dstw.gflownet.bsr_operator_state import make_demo_operator
from prism_dstw.gflownet.captured_graph_tiles import CapturedGraphTileRegistry
from prism_dstw.gflownet.cuda_graph_tile_runtime import CUDAGraphTileRuntime, TileCaptureError, TileReplayResult
from prism_dstw.gflownet.spectral_reward_manager import SpectralRewardError, SpectralRewardManager


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for captured tile runtime")


def test_cuda_graph_tile_runtime_replays_and_emits_state_hashes() -> None:
    state = make_demo_operator(4, device="cuda")
    registry = CapturedGraphTileRegistry.demo(tile_count=2)
    runtime = CUDAGraphTileRuntime(
        state=state,
        allow_eager_cuda_tiles=False,
        require_all_production_tiles_captured=True,
        warmup_iterations=1,
    )
    capture_ids = runtime.capture_registry(registry)
    assert len(capture_ids) == 2
    result = runtime.replay_tile("captured_tile_000")
    assert result.state_hash == state.state_hash
    assert result.captured_graph_tile_hash
    assert runtime.telemetry.captured_graph_replay_count == 1
    assert runtime.telemetry.uncaptured_fallback_count == 0


def test_reward_manager_uses_captured_tile_state_hash_cache() -> None:
    state = make_demo_operator(4, device="cuda")
    registry = CapturedGraphTileRegistry.demo(tile_count=1)
    runtime = CUDAGraphTileRuntime(
        state=state,
        allow_eager_cuda_tiles=False,
        require_all_production_tiles_captured=True,
        warmup_iterations=1,
    )
    runtime.capture_registry(registry)
    manager = SpectralRewardManager(state=state, telemetry=runtime.telemetry)
    event = runtime.replay_tile("captured_tile_000")
    first = manager.reward_for_event(event)
    second = manager.reward_for_event(event)
    assert first.cache_hit is False
    assert second.cache_hit is True
    cache_key = next(iter(manager.cache))
    assert cache_key.state_hash == event.state_hash
    assert cache_key.operator_hash == event.operator_hash
    assert cache_key.captured_graph_tile_hash == event.captured_graph_tile_hash
    assert runtime.telemetry.gpu_solve_count == 1
    assert runtime.telemetry.cpu_solve_count == 0


def test_reward_manager_rejects_never_captured_forged_event() -> None:
    state = make_demo_operator(4, device="cuda")
    manager = SpectralRewardManager(state=state, telemetry=CUDAGraphTileRuntime(
        state=state,
        allow_eager_cuda_tiles=False,
        require_all_production_tiles_captured=True,
        warmup_iterations=1,
    ).telemetry)
    forged = TileReplayResult(
        tile_id="never_captured",
        state_hash=state.state_hash,
        operator_hash=state.operator_hash,
        restricted_operator_hash="restricted",
        c6_operator_hash="c6",
        captured_graph_tile_hash="captured",
        event_provenance_hash="forged",
        topology_delta="topology",
        basin_delta="basin",
        changed_rows=(0,),
        changed_blocks=(0,),
        replay_ms=0.0,
        bsr_update_ms=0.0,
        row_renorm_ms=0.0,
        operator_hash_update_ms=0.0,
        device="cuda",
    )
    with pytest.raises(SpectralRewardError, match="not produced by captured tile runtime"):
        manager.reward_for_event(forged)


def test_uncaptured_production_tile_blocks_full_run() -> None:
    state = make_demo_operator(3, device="cuda")
    runtime = CUDAGraphTileRuntime(
        state=state,
        allow_eager_cuda_tiles=False,
        require_all_production_tiles_captured=True,
        warmup_iterations=1,
    )
    with pytest.raises(TileCaptureError, match="fallback forbidden"):
        runtime.replay_tile("missing_tile")


def test_failed_replay_rolls_back_last_good_state() -> None:
    state = make_demo_operator(2, device="cuda")
    registry = CapturedGraphTileRegistry.demo(tile_count=1)
    runtime = CUDAGraphTileRuntime(
        state=state,
        allow_eager_cuda_tiles=False,
        require_all_production_tiles_captured=True,
        warmup_iterations=1,
    )
    runtime.capture_registry(registry)
    observed_failure = False
    for _ in range(100):
        last_good_values = state.values.detach().clone()
        last_good_hash = state.state_hash
        try:
            runtime.replay_tile("captured_tile_000")
        except Exception:
            observed_failure = True
            assert bool(torch.allclose(last_good_values, state.values))
            assert state.state_hash == last_good_hash
            state.validate()
            break
    assert observed_failure
