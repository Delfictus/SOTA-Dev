"""Sparse block deltas applied by captured graph tiles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor

from prism_dstw.gflownet.bsr_operator_state import BSROperatorState, BSRStateError


class TileDeltaError(RuntimeError):
    """Raised when a tile delta violates operator-update invariants."""


@dataclass(frozen=True)
class TileOperatorDelta:
    """Sparse BSR block delta for one captured tile action."""

    tile_id: str
    affected_bsr_blocks: tuple[int, ...]
    delta_values: Tensor
    topology_delta: str
    basin_delta: str
    restricted_operator_target: str

    @classmethod
    def from_lists(
        cls,
        *,
        tile_id: str,
        affected_bsr_blocks: list[int],
        delta_values: list[list[list[float]]],
        topology_delta: str,
        basin_delta: str,
        restricted_operator_target: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "TileOperatorDelta":
        return cls(
            tile_id=tile_id,
            affected_bsr_blocks=tuple(int(value) for value in affected_bsr_blocks),
            delta_values=torch.tensor(delta_values, device=device, dtype=dtype),
            topology_delta=topology_delta,
            basin_delta=basin_delta,
            restricted_operator_target=restricted_operator_target,
        )

    @property
    def tile_delta_hash(self) -> str:
        payload = self.delta_values.detach().contiguous().cpu().numpy().tobytes()
        prefix = "|".join(
            [
                self.tile_id,
                ",".join(str(block) for block in self.affected_bsr_blocks),
                self.topology_delta,
                self.basin_delta,
                self.restricted_operator_target,
            ]
        ).encode("utf-8")
        return hashlib.sha256(prefix + payload).hexdigest()

    def validate_for_state(self, state: BSROperatorState) -> None:
        if not self.affected_bsr_blocks:
            raise TileDeltaError(f"{self.tile_id}: affected rows empty")
        if self.delta_values.device.type != state.device.type:
            raise TileDeltaError(f"{self.tile_id}: delta tensor device mismatch")
        if self.delta_values.dtype != state.dtype:
            raise TileDeltaError(f"{self.tile_id}: delta tensor dtype mismatch")
        if tuple(self.delta_values.shape) != (len(self.affected_bsr_blocks), *state.blocksize):
            raise TileDeltaError(f"{self.tile_id}: delta shape does not match affected BSR blocks")
        for block in self.affected_bsr_blocks:
            if block not in state.allowed_bsr_blocks:
                raise TileDeltaError(f"{self.tile_id}: tile writes outside allowed BSR block set: {block}")
        if not bool(torch.isfinite(self.delta_values).all().item()):
            raise TileDeltaError(f"{self.tile_id}: delta contains NaN/Inf")

    def apply_to_bsr_state(self, state: BSROperatorState, *, validate: bool = True) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Apply this delta using GPU tensor mutation and affected-row renormalization."""

        self.validate_for_state(state)
        block_index = torch.tensor(self.affected_bsr_blocks, device=state.device, dtype=torch.long)
        candidate_values = state.values.clone()
        candidate_values.index_add_(0, block_index, self.delta_values)
        if bool((candidate_values.index_select(0, block_index) < 0).any().item()):
            raise BSRStateError(f"{self.tile_id}: negative BSR value after delta application")
        changed_rows = tuple(sorted({state.row_for_block(block) for block in self.affected_bsr_blocks}))
        for row in changed_rows:
            row_blocks = state.row_blocks[row]
            if not row_blocks:
                raise BSRStateError(f"{self.tile_id}: affected row {row} has no blocks")
            row_index = torch.tensor(row_blocks, device=state.device, dtype=torch.long)
            row_values = candidate_values.index_select(0, row_index)
            row_sum = row_values.sum()
            if not bool(torch.isfinite(row_sum).item()) or float(row_sum.item()) <= 0.0:
                raise BSRStateError(f"{self.tile_id}: invalid row sum during renormalization")
            normalized = row_values / row_sum
            candidate_values.index_copy_(0, row_index, normalized)
        state.values.copy_(candidate_values)
        state.update_hashes(extra_state=f"tile:{self.tile_id}:{self.tile_delta_hash}")
        if validate:
            state.validate()
        return changed_rows, self.affected_bsr_blocks
