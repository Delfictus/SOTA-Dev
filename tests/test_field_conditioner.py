from __future__ import annotations

import torch

from prism_dstw.orchestration.field_conditioner import FieldConditioner


def test_step_0_returns_scaffold() -> None:
    scaffold_phase = torch.randn(4, 5, 8)
    scaffold_xyz = torch.randn(4, 3)
    conditioner = FieldConditioner(scaffold_phase, scaffold_xyz)
    result = conditioner.condition_at_step(0)
    assert torch.allclose(result, scaffold_phase)


def test_step_1_returns_product_fiber() -> None:
    scaffold_phase = torch.randn(4, 5, 8)
    scaffold_xyz = torch.randn(4, 3)
    product_xyz = torch.cat([scaffold_xyz, torch.randn(3, 3)], dim=0)
    conditioner = FieldConditioner(scaffold_phase, scaffold_xyz)
    result = conditioner.condition_at_step(1, product_xyz, n_scaffold=len(scaffold_xyz))
    assert result.shape == (7, 5, 8)
    assert torch.allclose(result[:4], scaffold_phase)
