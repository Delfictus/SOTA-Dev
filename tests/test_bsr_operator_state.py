from __future__ import annotations

import pytest
import torch

from prism_dstw.gflownet.bsr_operator_state import BSROperatorState, make_demo_operator


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for captured tile runtime")


def test_bsr_operator_state_is_cuda_float64_and_stochastic() -> None:
    state = make_demo_operator(5, device="cuda")
    state.validate()
    dense = state.dense_for_gpu_solve()
    assert state.values.device.type == "cuda"
    assert state.dtype == torch.float64
    assert tuple(state.blocksize) == (1, 1)
    assert bool(torch.allclose(dense.sum(dim=1), torch.ones(dense.shape[0], device="cuda", dtype=torch.float64)))
    assert state.operator_hash
    assert state.state_hash


def test_bsr_operator_state_rejects_cpu_placement() -> None:
    with pytest.raises(Exception, match="cuda"):
        BSROperatorState.from_dense(torch.eye(4, dtype=torch.float64), device="cpu")

