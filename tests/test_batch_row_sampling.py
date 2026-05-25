from __future__ import annotations

import torch

from scripts.train_gflownet_policy import (
    sample_actions_per_row,
    selected_forward_log_probs,
)


def test_different_rows_produce_different_actions() -> None:
    """Each row's action must come from its own logit distribution."""

    fake_logits = torch.full((2, 10), -100.0)
    fake_logits[0, 0] = 100.0
    fake_logits[1, 5] = 100.0
    actions = sample_actions_per_row(fake_logits, torch.ones(10, dtype=torch.bool), temperature=1.0)

    assert int(actions[0].item()) == 0
    assert int(actions[1].item()) == 5


def test_forward_selected_matches_action_row() -> None:
    """Selected log-prob for row i must come from row i's distribution."""

    torch.manual_seed(19)
    fake_logits = torch.randn(4, 10)
    actions = sample_actions_per_row(fake_logits, torch.ones(10, dtype=torch.bool), temperature=1.0)
    log_probs = torch.log_softmax(fake_logits, dim=-1)
    forward_selected = selected_forward_log_probs(log_probs, actions)

    for row_idx in range(4):
        expected = log_probs[row_idx, actions[row_idx]]
        assert torch.allclose(forward_selected[row_idx], expected)


def test_duplicate_repair_still_uses_row_distribution() -> None:
    """Duplicate rows are repaired from each row's remaining valid mass."""

    fake_logits = torch.full((2, 4), -100.0)
    fake_logits[0, 0] = 100.0
    fake_logits[1, 0] = 100.0
    fake_logits[1, 2] = 90.0

    actions = sample_actions_per_row(fake_logits, torch.ones(4, dtype=torch.bool), temperature=1.0)

    assert int(actions[0].item()) == 0
    assert int(actions[1].item()) == 2
