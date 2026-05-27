"""CUDA graph replay runtime for captured graph tiles."""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from prism_dstw.gflownet.bsr_operator_state import BSROperatorState, BSRStateError
from prism_dstw.gflownet.captured_graph_tiles import CapturedGraphTile, CapturedGraphTileRegistry


class TileCaptureError(RuntimeError):
    """Raised when a captured tile cannot be used in the requested runtime mode."""


@dataclass(frozen=True)
class TileReplayResult:
    tile_id: str
    state_hash: str
    operator_hash: str
    restricted_operator_hash: str
    c6_operator_hash: str
    captured_graph_tile_hash: str
    event_provenance_hash: str
    topology_delta: str
    basin_delta: str
    changed_rows: tuple[int, ...]
    changed_blocks: tuple[int, ...]
    replay_ms: float
    bsr_update_ms: float
    row_renorm_ms: float
    operator_hash_update_ms: float
    device: str


@dataclass
class CapturedTileTelemetry:
    cuda_graph_capture_count: int = 0
    cuda_graph_replay_count: int = 0
    captured_graph_replay_count: int = 0
    uncaptured_fallback_count: int = 0
    gpu_solve_count: int = 0
    cpu_solve_count: int = 0
    tile_exec_ms: list[float] = field(default_factory=list)
    gpu_solve_ms: list[float] = field(default_factory=list)
    captured_event_hashes: set[str] = field(default_factory=set)

    def summary(self) -> dict[str, Any]:
        avg_tile = sum(self.tile_exec_ms) / len(self.tile_exec_ms) if self.tile_exec_ms else 0.0
        avg_solve = sum(self.gpu_solve_ms) / len(self.gpu_solve_ms) if self.gpu_solve_ms else 0.0
        max_mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
        return {
            "cuda_graph_capture_count": self.cuda_graph_capture_count,
            "cuda_graph_replay_count": self.cuda_graph_replay_count,
            "captured_graph_replay_count": self.captured_graph_replay_count,
            "uncaptured_fallback_count": self.uncaptured_fallback_count,
            "gpu_solve_count": self.gpu_solve_count,
            "cpu_solve_count": self.cpu_solve_count,
            "avg_tile_exec_ms": avg_tile,
            "avg_gpu_solve_ms": avg_solve,
            "max_gpu_memory_allocated_mb": max_mem,
        }


@dataclass
class _CapturedReplay:
    graph: torch.cuda.CUDAGraph
    tile: CapturedGraphTile
    capture_id: str
    block_index: Tensor
    delta_values: Tensor
    changed_rows: tuple[int, ...]
    changed_blocks: tuple[int, ...]
    row_ranges: tuple[tuple[int, int], ...]
    row_sum_buffers: tuple[Tensor, ...]


class CUDAGraphTileRuntime:
    """Captures tile update graphs and replays them against a GPU BSR state."""

    def __init__(
        self,
        *,
        state: BSROperatorState,
        allow_eager_cuda_tiles: bool,
        require_all_production_tiles_captured: bool,
        warmup_iterations: int = 3,
    ) -> None:
        if state.device.type != "cuda" or not torch.cuda.is_available():
            raise TileCaptureError("LOG_SUBTB_BLOCKED_GPU_PLACEMENT: CUDA is required")
        self.state = state
        self.allow_eager_cuda_tiles = allow_eager_cuda_tiles
        self.require_all_production_tiles_captured = require_all_production_tiles_captured
        self.warmup_iterations = warmup_iterations
        self._captures: dict[str, _CapturedReplay] = {}
        self.telemetry = CapturedTileTelemetry()
        self._history: list[str] = []

    def reset_trajectory_history(self) -> None:
        self._history = []

    @staticmethod
    def _event_hash(
        *,
        tile_id: str,
        state_hash: str,
        operator_hash: str,
        topology_delta: str,
        basin_delta: str,
        restricted_operator_hash: str,
        c6_operator_hash: str,
        captured_graph_tile_hash: str,
    ) -> str:
        payload = "|".join(
            [
                tile_id,
                state_hash,
                operator_hash,
                topology_delta,
                basin_delta,
                restricted_operator_hash,
                c6_operator_hash,
                captured_graph_tile_hash,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _prepare_tile(
        self,
        tile: CapturedGraphTile,
    ) -> tuple[Tensor, Tensor, tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...], tuple[Tensor, ...]]:
        delta = tile.to_delta(state=self.state)
        delta.validate_for_state(self.state)
        block_index = torch.tensor(delta.affected_bsr_blocks, device=self.state.device, dtype=torch.long)
        delta_values = delta.delta_values.detach().clone()
        changed_rows = tuple(sorted({self.state.row_for_block(block) for block in delta.affected_bsr_blocks}))
        row_ranges: list[tuple[int, int]] = []
        row_sum_buffers: list[Tensor] = []
        for row in changed_rows:
            start = int(self.state.rowptr[row].detach().cpu().item())
            end = int(self.state.rowptr[row + 1].detach().cpu().item())
            if start == end:
                raise TileCaptureError(f"{tile.tile_id}: affected row {row} has no BSR blocks")
            row_ranges.append((start, end))
            row_sum_buffers.append(torch.empty((), device=self.state.device, dtype=self.state.dtype))
        original = self.state.values.detach().clone()
        scratch = self.state.values.detach().clone()
        scratch.index_add_(0, block_index, delta_values)
        if bool((scratch.index_select(0, block_index) < 0).any().item()):
            raise TileCaptureError(f"LOG_SUBTB_BLOCKED_BSR_UPDATE: {tile.tile_id} would produce negative BSR values")
        for start, end in row_ranges:
            row_sum = scratch[start:end].sum()
            if not bool(torch.isfinite(row_sum).item()) or float(row_sum.item()) <= 0.0:
                raise TileCaptureError(f"LOG_SUBTB_BLOCKED_OPERATOR_STOCHASTICITY: {tile.tile_id} invalid row sum")
        self.state.values.copy_(original)
        return block_index, delta_values, changed_rows, delta.affected_bsr_blocks, tuple(row_ranges), tuple(row_sum_buffers)

    def _captured_update(
        self,
        *,
        block_index: Tensor,
        delta_values: Tensor,
        row_ranges: tuple[tuple[int, int], ...],
        row_sum_buffers: tuple[Tensor, ...],
    ) -> None:
        self.state.values.index_add_(0, block_index, delta_values)
        for (start, end), row_sum in zip(row_ranges, row_sum_buffers, strict=True):
            row_view = self.state.values[start:end]
            torch.sum(row_view, dim=(0, 1, 2), out=row_sum)
            row_view.div_(row_sum)

    def capture_tile(self, tile: CapturedGraphTile) -> str:
        block_index, delta_values, changed_rows, changed_blocks, row_ranges, row_sum_buffers = self._prepare_tile(tile)
        capture_id = f"cuda_graph::{tile.capture_shape_bucket}::{tile.tile_id}"
        if not torch.cuda.is_available():
            raise TileCaptureError(f"{tile.tile_id}: CUDA unavailable")
        # Warm-up happens on a throwaway clone of values so the actual operator is not mutated.
        original_values = self.state.values.clone()
        for _ in range(self.warmup_iterations):
            self._captured_update(
                block_index=block_index,
                delta_values=delta_values,
                row_ranges=row_ranges,
                row_sum_buffers=row_sum_buffers,
            )
            self.state.values.copy_(original_values)
            self.state.update_hashes(extra_state="warmup_restore")
        torch.cuda.synchronize()
        graph: torch.cuda.CUDAGraph
        try:
            graph = torch.cuda.CUDAGraph()
            self.state.values.copy_(original_values)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                self._captured_update(
                    block_index=block_index,
                    delta_values=delta_values,
                    row_ranges=row_ranges,
                    row_sum_buffers=row_sum_buffers,
                )
            self.state.values.copy_(original_values)
            self.state.update_hashes(extra_state="capture_restore")
            torch.cuda.synchronize()
        except Exception:
            self.state.values.copy_(original_values)
            self.state.update_hashes(extra_state="capture_failed_restore")
            torch.cuda.synchronize()
            raise TileCaptureError(f"LOG_SUBTB_BLOCKED_TILE_CAPTURE: {tile.tile_id} failed CUDA graph capture")
        self._captures[tile.tile_id] = _CapturedReplay(
            graph=graph,
            tile=tile,
            capture_id=capture_id,
            block_index=block_index,
            delta_values=delta_values,
            changed_rows=changed_rows,
            changed_blocks=changed_blocks,
            row_ranges=row_ranges,
            row_sum_buffers=row_sum_buffers,
        )
        self.telemetry.cuda_graph_capture_count += 1
        return capture_id

    def capture_registry(self, registry: CapturedGraphTileRegistry) -> dict[str, str]:
        capture_ids: dict[str, str] = {}
        for tile in registry.tiles.values():
            try:
                capture_ids[tile.tile_id] = self.capture_tile(tile)
            except TileCaptureError:
                if self.require_all_production_tiles_captured:
                    raise
        if self.require_all_production_tiles_captured and len(capture_ids) != len(registry.tiles):
            raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: uncaptured production tile present")
        if not capture_ids:
            raise TileCaptureError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: captured graph tile registry produced zero captures")
        return capture_ids

    def replay_tile(self, tile_id: str) -> TileReplayResult:
        capture = self._captures.get(tile_id)
        if capture is None:
            if not self.allow_eager_cuda_tiles:
                raise TileCaptureError(f"uncaptured tile fallback forbidden in production: {tile_id}")
            self.telemetry.uncaptured_fallback_count += 1
            raise TileCaptureError(f"eager replay requested for unknown tile: {tile_id}")
        else:
            last_good_values = self.state.values.detach().clone()
            last_good_state_hash = self.state.state_hash
            last_good_operator_hash = self.state.operator_hash
            last_good_history = list(self._history)
            started = time.perf_counter()
            try:
                capture.graph.replay()
                torch.cuda.synchronize()
                self.telemetry.cuda_graph_replay_count += 1
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                changed_rows = capture.changed_rows
                changed_blocks = capture.changed_blocks
                self._history.append(tile_id)
                self.state.update_hashes(extra_state="|".join(self._history))
                self.state.validate()
            except BSRStateError:
                self.state.values.copy_(last_good_values)
                self.state.state_hash = last_good_state_hash
                self.state.operator_hash = last_good_operator_hash
                self._history = last_good_history
                torch.cuda.synchronize()
                raise
        self.telemetry.captured_graph_replay_count += 1
        self.telemetry.tile_exec_ms.append(elapsed_ms)
        tile = self._captures[tile_id].tile
        event_hash = self._event_hash(
            tile_id=tile_id,
            state_hash=self.state.state_hash,
            operator_hash=self.state.operator_hash,
            topology_delta=tile.topology_delta,
            basin_delta=tile.basin_delta,
            restricted_operator_hash=tile.restricted_operator_hash,
            c6_operator_hash=tile.c6_operator_hash,
            captured_graph_tile_hash=tile.captured_graph_tile_hash,
        )
        self.telemetry.captured_event_hashes.add(event_hash)
        return TileReplayResult(
            tile_id=tile_id,
            state_hash=self.state.state_hash,
            operator_hash=self.state.operator_hash,
            restricted_operator_hash=tile.restricted_operator_hash,
            c6_operator_hash=tile.c6_operator_hash,
            captured_graph_tile_hash=tile.captured_graph_tile_hash,
            event_provenance_hash=event_hash,
            topology_delta=tile.topology_delta,
            basin_delta=tile.basin_delta,
            changed_rows=changed_rows,
            changed_blocks=changed_blocks,
            replay_ms=elapsed_ms,
            bsr_update_ms=elapsed_ms,
            row_renorm_ms=0.0,
            operator_hash_update_ms=0.0,
            device=str(self.state.device),
        )

    def production_invariant_report(self) -> dict[str, Any]:
        summary = self.telemetry.summary()
        summary.update(
            {
                "captured_graph_tile_runtime_enabled": True,
                "bsr_operator_device": str(self.state.device),
                "bsr_operator_dtype": str(self.state.dtype).replace("torch.", ""),
                "production_cpu_fallback_used": self.telemetry.cpu_solve_count > 0
                or self.telemetry.uncaptured_fallback_count > 0,
            }
        )
        return summary
