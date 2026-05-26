from __future__ import annotations

import torch

from prism_dstw.orchestration.field_conditioner import FieldConditioner


def test_step_0_returns_scaffold() -> None:
    scaffold_phase = torch.randn(4, 5, 8)
    scaffold_xyz = torch.randn(4, 3)
    conditioner = FieldConditioner(scaffold_phase, scaffold_xyz)
    result = conditioner.condition_at_step(0)
    assert torch.allclose(result, scaffold_phase)
    result.zero_()
    assert torch.allclose(conditioner.condition_at_step(0), scaffold_phase)


def test_negative_step_is_rejected() -> None:
    conditioner = FieldConditioner(torch.zeros((2, 5, 8)), torch.zeros((2, 3)))

    try:
        conditioner.condition_at_step(-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("FieldConditioner accepted a negative step")


def test_step_1_returns_product_fiber() -> None:
    scaffold_phase = torch.randn(4, 5, 8)
    scaffold_xyz = torch.randn(4, 3)
    product_xyz = torch.cat([scaffold_xyz, torch.randn(3, 3)], dim=0)
    conditioner = FieldConditioner(scaffold_phase, scaffold_xyz)
    result = conditioner.condition_at_step(1, product_xyz, n_scaffold=len(scaffold_xyz))
    assert result.shape == (7, 5, 8)
    assert torch.allclose(result[:4], scaffold_phase)


def test_step_1_requires_product_coordinates() -> None:
    conditioner = FieldConditioner(torch.zeros((2, 5, 8)), torch.zeros((2, 3)))

    try:
        conditioner.condition_at_step(1)
    except ValueError as exc:
        assert "current_product_xyz" in str(exc)
    else:
        raise AssertionError("FieldConditioner accepted missing product coordinates after step 0")


def test_conditioner_rejects_scaffold_row_mismatch() -> None:
    try:
        FieldConditioner(torch.zeros((3, 5, 8)), torch.zeros((2, 3)))
    except ValueError as exc:
        assert "matching atom counts" in str(exc)
    else:
        raise AssertionError("FieldConditioner accepted mismatched scaffold rows")


def test_conditioner_rejects_nonfinite_scaffold_inputs() -> None:
    for phase, xyz in (
        (torch.full((1, 5, 8), float("nan")), torch.zeros((1, 3))),
        (torch.zeros((1, 5, 8)), torch.tensor([[0.0, float("inf"), 0.0]])),
    ):
        try:
            FieldConditioner(phase, xyz)
        except ValueError as exc:
            assert "non-finite" in str(exc)
        else:
            raise AssertionError("FieldConditioner accepted non-finite scaffold input")


def test_explicit_zero_scaffold_count_is_preserved() -> None:
    scaffold_phase = torch.ones((2, 5, 8), dtype=torch.float32)
    scaffold_xyz = torch.zeros((2, 3), dtype=torch.float32)
    product_xyz = torch.zeros((3, 3), dtype=torch.float32)
    conditioner = FieldConditioner(scaffold_phase, scaffold_xyz)

    result = conditioner.condition_at_step(1, product_xyz, n_scaffold=0)

    assert result.shape == (3, 5, 8)
    assert float(result[:, :, 6].sum().item()) == 15.0
    assert float(result[:, :, :6].sum().item()) == 0.0


def test_v2_fallback_marks_new_atoms_void_and_reversible() -> None:
    scaffold_phase = torch.zeros((2, 5, 12), dtype=torch.float32)
    scaffold_xyz = torch.zeros((2, 3), dtype=torch.float32)
    product_xyz = torch.zeros((4, 3), dtype=torch.float32)
    conditioner = FieldConditioner(scaffold_phase, scaffold_xyz)

    result = conditioner.condition_at_step(1, product_xyz, n_scaffold=2)

    assert result.shape == (4, 5, 12)
    assert float(result[2:, :, 6].sum().item()) == 10.0
    assert float(result[2:, :, 10].sum().item()) == 0.0
    assert float(result[2:, :, 11].sum().item()) == 10.0


def test_conditioner_rejects_nonfinite_product_coordinates() -> None:
    scaffold_phase = torch.zeros((2, 5, 8), dtype=torch.float32)
    scaffold_xyz = torch.zeros((2, 3), dtype=torch.float32)
    product_xyz = torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]], dtype=torch.float32)
    conditioner = FieldConditioner(scaffold_phase, scaffold_xyz)

    try:
        conditioner.condition_at_step(1, product_xyz, n_scaffold=1)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("FieldConditioner accepted non-finite product coordinates")


def test_fallback_rejects_invalid_scaffold_atom_count() -> None:
    scaffold_phase = torch.zeros((2, 5, 12), dtype=torch.float32)
    scaffold_xyz = torch.zeros((2, 3), dtype=torch.float32)
    product_xyz = torch.zeros((3, 3), dtype=torch.float32)
    conditioner = FieldConditioner(scaffold_phase, scaffold_xyz)

    for n_scaffold in (-1, 4, 3):
        try:
            conditioner.condition_at_step(1, product_xyz, n_scaffold=n_scaffold)
        except ValueError as exc:
            assert "n_scaffold" in str(exc)
        else:
            raise AssertionError(f"FieldConditioner accepted invalid n_scaffold={n_scaffold}")
