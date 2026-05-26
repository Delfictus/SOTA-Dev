from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
from torch_geometric.data import Data  # type: ignore[import-untyped]

from scripts.train_gflownet_policy import (
    ActionSpace,
    AssembledState,
    AssemblyHistoryStep,
    ScaffoldGraph,
    backward_policy_inputs_for_states,
    current_policy_anchor_embeddings,
    selected_backward_log_probs_for_growth,
)


class _PolicyWithAnchors(torch.nn.Module):
    def __init__(self, anchors: torch.Tensor) -> None:
        super().__init__()
        self.anchor_embeddings = torch.nn.Parameter(anchors.clone())


def _state(action_idx: int) -> AssembledState:
    data = Data(
        x_base=torch.zeros((2, 4)),
        x_phase=torch.zeros((2, 5, 8)),
        xyz=torch.zeros((2, 3)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1)),
        active_dendrite_mask=torch.empty((0,), dtype=torch.bool),
        num_nodes=2,
    )
    scaffold = ScaffoldGraph(
        scaffold_id="test",
        ligand_sdf=Path(__file__),
        data=data,
        exit_node_idx=0,
        base_feature_dim=4,
        phase_feature_dim=8,
        edge_feature_dim=1,
        retained_atom_count=2,
    )
    history = (
        AssemblyHistoryStep(
            step=0,
            synthon_id="a",
            exit_atom_idx=0,
            action_idx=action_idx,
            node_count=2,
        ),
    )
    return AssembledState(
        scaffold=scaffold,
        data=data,
        exit_node_idx=1,
        history=history,
        last_growth_action_idx=action_idx,
        canonical_smiles="CC",
        anchor_id="a",
        coordinates_json="[]",
        score_atom_offset=0,
        survivor_lookup_smiles=None,
    )


def _action_space() -> ActionSpace:
    anchors = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    return ActionSpace(
        table=pl.DataFrame({"anchor_id": [str(i) for i in range(6)]}),
        anchor_embeddings=anchors,
        valid_mask=torch.ones((6,), dtype=torch.bool),
        reward_targets=torch.ones((6,), dtype=torch.float32) / 6.0,
    )


def test_backward_candidates_use_current_policy_anchor_embeddings() -> None:
    action_space = _action_space()
    model = _PolicyWithAnchors(torch.ones_like(action_space.anchor_embeddings) * 7.0)

    mask, attachments, targets = backward_policy_inputs_for_states(
        [_state(2), _state(4)],
        action_space,
        max_backward_actions=4,
        anchor_embeddings=current_policy_anchor_embeddings(model, action_space),
    )

    assert mask.shape == (2, 4)
    assert targets.tolist() == [0, 0]
    assert torch.allclose(attachments[:, 0], torch.full((2, 4), 7.0))
    assert attachments.requires_grad


def test_stop_transitions_get_zero_backward_log_prob() -> None:
    backward_log_probs = torch.tensor(
        [
            [-2.0, -1.0],
            [-3.0, -0.5],
            [-4.0, -0.25],
        ]
    )
    target_indices = torch.tensor([0, 0, 0], dtype=torch.long)
    growth_rows = torch.tensor([True, False, True])

    selected = selected_backward_log_probs_for_growth(backward_log_probs, target_indices, growth_rows)

    assert selected.tolist() == [-2.0, 0.0, -4.0]
