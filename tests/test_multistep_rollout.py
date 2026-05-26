from __future__ import annotations

import torch
from torch_geometric.data import Data  # type: ignore[import-untyped]
import polars as pl
from pathlib import Path
import pytest

from scripts.train_gflownet_policy import (
    ActionSpace,
    ScaffoldGraph,
    condition_states_for_step,
    sample_actions_per_row,
    grow_state_with_action,
    initial_assembled_state,
    proposals_for_terminal_states,
    rust_zmatrix_attach_fragment,
    rollout_forward_action_mask,
    state_with_conditioned_phase,
)
from prism_dstw.orchestration.field_conditioner import FieldConditioner


def _graph() -> ScaffoldGraph:
    data = Data(
        x_base=torch.zeros((2, 12), dtype=torch.float32),
        x_phase=torch.zeros((2, 5, 8), dtype=torch.float32),
        xyz=torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.ones((2, 6), dtype=torch.float32),
        active_dendrite_mask=torch.tensor([False, False]),
        num_nodes=2,
    )
    return ScaffoldGraph(
        scaffold_id="SCAF",
        ligand_sdf=Path(__file__),
        data=data,
        exit_node_idx=1,
        base_feature_dim=12,
        phase_feature_dim=8,
        edge_feature_dim=6,
        retained_atom_count=2,
    )


def _action_space() -> ActionSpace:
    table = pl.DataFrame(
        {
            "anchor_id": ["A0"],
            "canonical_smiles": ["CCO"],
            "coordinates_json": ['[[0,0,0],[1,0,0],[2,0,0],[3,0,0]]'],
            "n_heavy_atoms": [2],
            "action_valid": [True],
        }
    )
    atom_features = torch.zeros((1, 2, 12), dtype=torch.float32)
    atom_features[0, :, 0] = 0.06
    atom_mask = torch.tensor([[True, True]])
    return ActionSpace(
        table=table,
        anchor_embeddings=torch.zeros((1, 4), dtype=torch.float32),
        valid_mask=torch.tensor([True]),
        reward_targets=torch.tensor([1.0]),
        action_atom_features=atom_features,
        action_atom_mask=atom_mask,
    )


def test_growth_feeds_product_state_forward() -> None:
    graph = _graph()
    state0 = initial_assembled_state(graph)

    state1 = grow_state_with_action(state0, action_space=_action_space(), action_idx=0, step=0)
    state2 = grow_state_with_action(state1, action_space=_action_space(), action_idx=0, step=1)

    assert state1.data.x_base.shape[0] == 4
    assert state2.data.x_base.shape[0] == 6
    assert state1.exit_node_idx == 3
    assert state2.exit_node_idx == 5
    first_step_new_xyz = state1.data.xyz[2:]
    second_step_new_xyz = state2.data.xyz[4:]
    assert float(second_step_new_xyz[:, 0].min().item()) > float(first_step_new_xyz[:, 0].max().item())
    assert state2.score_atom_offset == graph.retained_atom_count
    assert state2.canonical_smiles != "CCO"
    assert "." not in state2.canonical_smiles
    assert [(item.step, item.synthon_id, item.exit_atom_idx) for item in state2.history] == [
        (0, "A0", 1),
        (1, "A0", 3),
    ]


def test_growth_uses_rust_zmatrix_subprocess_coordinates() -> None:
    graph = _graph()
    fragment = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]], dtype=torch.float32)

    transformed = rust_zmatrix_attach_fragment(fragment, current_xyz=graph.data.xyz, exit_node_idx=graph.exit_node_idx)

    assert transformed.shape == (2, 3)
    assert torch.isfinite(transformed).all()
    assert not torch.allclose(transformed, fragment)


def test_growth_does_not_admit_nonfinite_action_coordinates() -> None:
    graph = _graph()
    action_space = _action_space()
    table = action_space.table.with_columns(pl.lit("[[NaN,0,0],[1,0,0]]").alias("coordinates_json"))
    poisoned = ActionSpace(
        table=table,
        anchor_embeddings=action_space.anchor_embeddings,
        valid_mask=action_space.valid_mask,
        reward_targets=action_space.reward_targets,
        action_atom_features=action_space.action_atom_features,
        action_atom_mask=action_space.action_atom_mask,
        action_rows=tuple(table.to_dicts()),
    )

    state = grow_state_with_action(initial_assembled_state(graph), action_space=poisoned, action_idx=0, step=0)

    assert torch.isfinite(state.data.xyz).all()
    assert torch.isfinite(state.data.edge_attr).all()


class SpyFieldConditioner(FieldConditioner):
    def __init__(self, scaffold_phase: torch.Tensor, scaffold_xyz: torch.Tensor) -> None:
        super().__init__(scaffold_phase, scaffold_xyz)
        self.calls: list[tuple[int, torch.Tensor | None]] = []

    def condition_at_step(
        self,
        step: int,
        current_product_xyz: torch.Tensor | None = None,
        n_scaffold: int | None = None,
    ) -> torch.Tensor:
        self.calls.append((step, None if current_product_xyz is None else current_product_xyz.detach().clone()))
        return super().condition_at_step(step, current_product_xyz, n_scaffold)


def test_step_conditioning_receives_actual_product_coordinates() -> None:
    graph = _graph()
    state0 = initial_assembled_state(graph)
    state1 = grow_state_with_action(state0, action_space=_action_space(), action_idx=0, step=0)
    conditioner = SpyFieldConditioner(graph.data.x_phase, graph.data.xyz)

    conditioned0, count0, events0 = condition_states_for_step([state0], [conditioner], step=0)
    conditioned1, count1, events1 = condition_states_for_step([state1], [conditioner], step=1)

    assert count0 == 1
    assert count1 == 1
    assert events0 == ((0, 0, int(state0.data.x_base.shape[0]), int(state0.data.x_phase.shape[2])),)
    assert events1 == ((0, 1, int(state1.data.x_base.shape[0]), int(state1.data.x_phase.shape[2])),)
    assert conditioned0[0].data.x_phase.shape[0] == state0.data.x_base.shape[0]
    assert conditioned1[0].data.x_phase.shape[0] == state1.data.x_base.shape[0]
    assert conditioner.calls[0][0] == 0
    assert conditioner.calls[0][1] is None
    assert conditioner.calls[1][0] == 1
    assert conditioner.calls[1][1] is not None
    assert torch.allclose(conditioner.calls[1][1], state1.data.xyz)
    assert int(conditioned1[0].data.x_phase.shape[0]) > int(conditioned0[0].data.x_phase.shape[0])


def test_step_conditioning_skips_inactive_rows() -> None:
    graph = _graph()
    state0 = initial_assembled_state(graph)
    state1 = grow_state_with_action(state0, action_space=_action_space(), action_idx=0, step=0)
    active_conditioner = SpyFieldConditioner(graph.data.x_phase, graph.data.xyz)
    inactive_conditioner = SpyFieldConditioner(graph.data.x_phase, graph.data.xyz)

    conditioned, call_count, events = condition_states_for_step(
        [state1, state0],
        [active_conditioner, inactive_conditioner],
        step=1,
        active_rows=torch.tensor([True, False]),
    )

    assert call_count == 1
    assert events == ((0, 1, int(state1.data.x_base.shape[0]), int(state1.data.x_phase.shape[2])),)
    assert len(active_conditioner.calls) == 1
    assert len(inactive_conditioner.calls) == 0
    assert torch.allclose(conditioned[0].data.x_phase[:2], graph.data.x_phase)
    assert conditioned[1] is state0


def test_step_conditioning_requires_1d_bool_active_mask() -> None:
    graph = _graph()
    state = initial_assembled_state(graph)
    conditioner = SpyFieldConditioner(graph.data.x_phase, graph.data.xyz)

    for bad_mask in (torch.tensor([1]), torch.tensor([1.0]), torch.tensor([[True]])):
        with pytest.raises(ValueError, match="1D bool"):
            condition_states_for_step([state], [conditioner], step=0, active_rows=bad_mask)


def test_step_conditioning_rejects_negative_step() -> None:
    graph = _graph()
    state = initial_assembled_state(graph)
    conditioner = SpyFieldConditioner(graph.data.x_phase, graph.data.xyz)

    with pytest.raises(ValueError, match="non-negative"):
        condition_states_for_step([state], [conditioner], step=-1)


def test_state_conditioning_rejects_phase_feature_dim_mismatch() -> None:
    graph = _graph()
    state = initial_assembled_state(graph)
    bad_phase = torch.zeros((int(state.data.x_base.shape[0]), 5, 7), dtype=torch.float32)

    try:
        state_with_conditioned_phase(state, bad_phase)
    except ValueError as exc:
        assert "feature dimension" in str(exc)
    else:
        raise AssertionError("state_with_conditioned_phase accepted a wrong phase feature dimension")


def test_terminal_live_proposal_uses_product_identity_and_full_growth_offset() -> None:
    graph = _graph()
    state = initial_assembled_state(graph)
    state = grow_state_with_action(state, action_space=_action_space(), action_idx=0, step=0)
    state = grow_state_with_action(state, action_space=_action_space(), action_idx=0, step=1)

    proposal = proposals_for_terminal_states([state], oracle_mode="live_signal_grid")[0]

    assert proposal.canonical_smiles == state.canonical_smiles
    assert proposal.canonical_smiles != "CCO"
    assert proposal.score_atom_offset == 0


def test_terminal_survivor_lookup_keeps_survivor_key_for_single_step_compatibility() -> None:
    graph = _graph()
    state = grow_state_with_action(initial_assembled_state(graph), action_space=_action_space(), action_idx=0, step=0)

    proposal = proposals_for_terminal_states([state], oracle_mode="survivor_lookup")[0]

    assert proposal.canonical_smiles == "CCO"


def test_product_identity_does_not_collide_on_action_modulo() -> None:
    graph = _graph()
    action_space_0 = _action_space()
    state0 = grow_state_with_action(initial_assembled_state(graph), action_space=action_space_0, action_idx=0, step=0)

    table = pl.DataFrame(
        {
            "anchor_id": [f"A{i}" for i in range(201)],
            "canonical_smiles": ["CCO" for _ in range(201)],
            "coordinates_json": ['[[0,0,0],[1,0,0],[2,0,0],[3,0,0]]' for _ in range(201)],
            "n_heavy_atoms": [2 for _ in range(201)],
            "action_valid": [True for _ in range(201)],
        }
    )
    atom_features = torch.zeros((201, 2, 12), dtype=torch.float32)
    atom_features[:, :, 0] = 0.06
    atom_mask = torch.ones((201, 2), dtype=torch.bool)
    action_space_200 = ActionSpace(
        table=table,
        anchor_embeddings=torch.zeros((201, 4), dtype=torch.float32),
        valid_mask=torch.ones(201, dtype=torch.bool),
        reward_targets=torch.full((201,), 1.0 / 201.0),
        action_atom_features=atom_features,
        action_atom_mask=atom_mask,
        action_rows=tuple(table.to_dicts()),
    )
    state200 = grow_state_with_action(
        initial_assembled_state(graph),
        action_space=action_space_200,
        action_idx=200,
        step=0,
    )

    assert state0.canonical_smiles != state200.canonical_smiles


def test_batch_rows_may_select_same_growth_action_independently() -> None:
    logits = torch.full((2, 2), -100.0)
    logits[:, 0] = 100.0
    mask = torch.tensor([[True, False], [True, False]])

    actions = sample_actions_per_row(logits, mask, temperature=1.0, require_unique=False)

    assert actions.tolist() == [0, 0]


def test_rollout_mask_fails_early_without_growth_actions() -> None:
    action_space = _action_space()
    action_space = ActionSpace(
        table=action_space.table,
        anchor_embeddings=action_space.anchor_embeddings,
        valid_mask=torch.tensor([False]),
        reward_targets=action_space.reward_targets,
    )
    with pytest.raises(ValueError, match="valid growth action"):
        rollout_forward_action_mask(action_space, active_rows=torch.tensor([True]), step=0, max_steps=3)


def test_stop_action_is_last_logit_and_not_available_at_step_zero() -> None:
    action_space = _action_space()
    active = torch.tensor([True, True])

    step0 = rollout_forward_action_mask(action_space, active_rows=active, step=0, max_steps=3)
    final_step = rollout_forward_action_mask(action_space, active_rows=active, step=2, max_steps=3)

    assert step0.shape == (2, 2)
    assert not bool(step0[:, -1].any())
    assert bool(step0[:, 0].all())
    assert bool(final_step[:, -1].all())
    assert not bool(final_step[:, 0].any())
