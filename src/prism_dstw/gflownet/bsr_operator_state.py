"""GPU-resident BSR operator state for spectral tile execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


class BSRStateError(RuntimeError):
    """Raised when a BSR operator invariant is violated."""


def _tensor_hash(tensor: Tensor) -> str:
    data = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


@dataclass
class BSROperatorState:
    """Owns the current restricted operator as mutable GPU BSR block values."""

    rowptr: Tensor
    colind: Tensor
    values: Tensor
    blocksize: tuple[int, int]
    device: torch.device
    dtype: torch.dtype
    allowed_bsr_blocks: frozenset[int]
    row_blocks: dict[int, tuple[int, ...]]
    block_rows: tuple[int, ...]
    state_hash: str
    operator_hash: str

    @classmethod
    def from_dense(
        cls,
        dense: Tensor,
        *,
        blocksize: tuple[int, int] = (1, 1),
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> "BSROperatorState":
        target_device = torch.device(device or dense.device)
        if target_device.type != "cuda":
            raise BSRStateError("BSROperatorState requires cuda device")
        if blocksize != (1, 1):
            raise BSRStateError("only blocksize=(1, 1) is currently production-captured")
        matrix = dense.to(device=target_device, dtype=dtype)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise BSRStateError("base operator must be a square rank-2 tensor")
        if not bool(torch.isfinite(matrix).all().item()):
            raise BSRStateError("base operator contains NaN/Inf")
        if bool((matrix < 0).any().item()):
            raise BSRStateError("base operator contains negative values")
        row_sums = matrix.sum(dim=1, keepdim=True)
        if bool((row_sums <= 0).any().item()):
            raise BSRStateError("base operator has an empty row")
        matrix = matrix / row_sums
        bsr = matrix.to_sparse_bsr(blocksize=blocksize)
        rowptr = bsr.crow_indices().to(device=target_device)
        colind = bsr.col_indices().to(device=target_device)
        values = bsr.values().clone().to(device=target_device, dtype=dtype)
        row_blocks: dict[int, tuple[int, ...]] = {}
        block_rows: list[int] = []
        rowptr_cpu = rowptr.detach().cpu().tolist()
        for row in range(len(rowptr_cpu) - 1):
            start = int(rowptr_cpu[row])
            end = int(rowptr_cpu[row + 1])
            row_blocks[row] = tuple(range(start, end))
            block_rows.extend([row] * (end - start))
        state = cls(
            rowptr=rowptr,
            colind=colind,
            values=values,
            blocksize=blocksize,
            device=target_device,
            dtype=dtype,
            allowed_bsr_blocks=frozenset(range(int(values.shape[0]))),
            row_blocks=row_blocks,
            block_rows=tuple(block_rows),
            state_hash="",
            operator_hash="",
        )
        state.update_hashes(extra_state="initial")
        state.validate()
        return state

    @property
    def shape(self) -> tuple[int, int]:
        rows = int(self.rowptr.shape[0] - 1) * self.blocksize[0]
        cols = rows
        return rows, cols

    def sparse_bsr(self) -> Tensor:
        return torch.sparse_bsr_tensor(
            self.rowptr,
            self.colind,
            self.values,
            size=self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def dense_for_gpu_solve(self) -> Tensor:
        return self.sparse_bsr().to_dense()

    def validate(self, *, tolerance: float = 1.0e-8) -> None:
        if self.device.type != "cuda":
            raise BSRStateError("BSR operator state is not cuda-resident")
        if self.values.device.type != "cuda" or self.rowptr.device.type != "cuda" or self.colind.device.type != "cuda":
            raise BSRStateError("BSR tensors must all be cuda-resident")
        if self.values.dtype != self.dtype:
            raise BSRStateError("BSR values dtype mismatch")
        if not bool(torch.isfinite(self.values).all().item()):
            raise BSRStateError("BSR values contain NaN/Inf")
        if bool((self.values < 0).any().item()):
            raise BSRStateError("BSR values contain negative entries")
        dense = self.dense_for_gpu_solve()
        row_sums = dense.sum(dim=1)
        max_error = torch.max(torch.abs(row_sums - 1.0)).item()
        if float(max_error) > tolerance:
            raise BSRStateError(f"row stochasticity error {max_error:.3e} exceeds tolerance {tolerance:.3e}")

    def update_hashes(self, *, extra_state: str) -> None:
        operator_hash = _tensor_hash(self.values)
        payload = "|".join(
            [
                operator_hash,
                _tensor_hash(self.rowptr),
                _tensor_hash(self.colind),
                str(self.blocksize),
                extra_state,
            ]
        )
        self.operator_hash = operator_hash
        self.state_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def row_for_block(self, block_index: int) -> int:
        try:
            return self.block_rows[block_index]
        except IndexError as exc:
            raise BSRStateError(f"BSR block {block_index} is outside operator state") from exc

    def manifest(self) -> dict[str, Any]:
        return {
            "blocksize": list(self.blocksize),
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": str(self.device),
            "nnz_blocks": int(self.values.shape[0]),
            "operator_hash": self.operator_hash,
            "state_hash": self.state_hash,
        }


def make_demo_operator(tile_count: int, *, device: torch.device | str = "cuda") -> BSROperatorState:
    """Create a deterministic row-stochastic BSR operator for runtime validation."""

    n = max(tile_count + 2, 8)
    base = torch.eye(n, dtype=torch.float64)
    for row in range(n):
        base[row, (row + 1) % n] = 0.25
        base[row, row] = 0.75
    return BSROperatorState.from_dense(base, device=device, dtype=torch.float64)

