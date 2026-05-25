from __future__ import annotations

import torch

from prism_dstw.hierarchical_bayes.gflownet_policy import FieldConditionedDualChannelGFlowNetPolicy


def _policy() -> FieldConditionedDualChannelGFlowNetPolicy:
    torch.manual_seed(7)
    return FieldConditionedDualChannelGFlowNetPolicy(
        base_feature_dim=4,
        phase_feature_dim=8,
        edge_feature_dim=1,
        anchor_embeddings=torch.randn(6, 16),
        hidden_dim=24,
        embedding_dim=16,
        learn_anchor_embeddings=True,
    )


def _graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(4, 4)
    x_phase = torch.randn(4, 5, 8)
    edge_index = torch.tensor([[0, 1, 3, 2], [1, 0, 2, 3]], dtype=torch.long)
    edge_attr = torch.ones(4, 1)
    active = torch.tensor([True, True, False, False])
    batch = torch.zeros(4, dtype=torch.long)
    return x, x_phase, edge_index, edge_attr, active, batch


def test_electronic_channel_masks_aliphatic() -> None:
    policy = _policy().eval()
    x, x_phase, edge_index, edge_attr, active, batch = _graph()
    out_a = policy.embed_graph(
        node_features=x,
        x_phase=x_phase,
        edge_index=edge_index,
        edge_distance_a=edge_attr[:, 0],
        active_dendrite_mask=active,
        batch_index=batch,
    )
    x_perturbed = x.clone()
    x_perturbed[3] += 10.0
    out_b = policy.embed_graph(
        node_features=x_perturbed,
        x_phase=x_phase,
        edge_index=edge_index,
        edge_distance_a=edge_attr[:, 0],
        active_dendrite_mask=active,
        batch_index=batch,
    )
    electronic_a = out_a[1]
    electronic_b = out_b[1]
    assert torch.allclose(electronic_a[2], electronic_b[2], atol=1.0e-6)


def test_kinematic_channel_sees_all_atoms() -> None:
    policy = _policy().eval()
    x, x_phase, edge_index, edge_attr, active, batch = _graph()
    out_a = policy.embed_graph(
        node_features=x,
        x_phase=x_phase,
        edge_index=edge_index,
        edge_distance_a=edge_attr[:, 0],
        active_dendrite_mask=active,
        batch_index=batch,
    )
    x_perturbed = x.clone()
    x_perturbed[3] += 10.0
    out_b = policy.embed_graph(
        node_features=x_perturbed,
        x_phase=x_phase,
        edge_index=edge_index,
        edge_distance_a=edge_attr[:, 0],
        active_dendrite_mask=active,
        batch_index=batch,
    )
    steric_a = out_a[2]
    steric_b = out_b[2]
    assert not torch.allclose(steric_a[2], steric_b[2], atol=1.0e-6)


def test_dual_channel_fusion_produces_valid_logits() -> None:
    policy = _policy()
    x, x_phase, edge_index, edge_attr, active, batch = _graph()
    output = policy(
        x_base=x,
        x_phase=x_phase,
        edge_index=edge_index,
        edge_attr=edge_attr,
        active_dendrite_mask=active,
        batch_index=batch,
        exit_node_indices=torch.tensor([0]),
        forward_action_mask=torch.ones(1, 6, dtype=torch.bool),
        backward_action_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    assert output.forward_log_probs.shape == (1, 6)
    assert output.phase_global_embeddings.shape == (1, 16)
    assert output.channel_gate.shape == (4, 16)
    output.forward_log_probs.sum().backward()


def test_action_phase_features_condition_logits() -> None:
    policy = _policy().eval()
    x, x_phase, edge_index, edge_attr, active, batch = _graph()
    common = {
        "x_base": x,
        "x_phase": x_phase,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "active_dendrite_mask": active,
        "batch_index": batch,
        "exit_node_indices": torch.tensor([0]),
        "forward_action_mask": torch.ones(1, 6, dtype=torch.bool),
        "backward_action_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    neutral = torch.zeros(6, 5, 8)
    enriched = neutral.clone()
    enriched[3, :, 2] = 10.0
    out_neutral = policy(**common, action_phase_features=neutral)
    out_enriched = policy(**common, action_phase_features=enriched)
    assert out_neutral.action_field_embeddings is not None
    assert out_enriched.action_field_embeddings is not None
    assert not torch.allclose(out_neutral.forward_logits, out_enriched.forward_logits)


def test_action_base_charge_features_condition_logits() -> None:
    policy = _policy().eval()
    x, x_phase, edge_index, edge_attr, active, batch = _graph()
    common = {
        "x_base": x,
        "x_phase": x_phase,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "active_dendrite_mask": active,
        "batch_index": batch,
        "exit_node_indices": torch.tensor([0]),
        "forward_action_mask": torch.ones(1, 6, dtype=torch.bool),
        "backward_action_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    neutral = torch.zeros(6, 4)
    charged = neutral.clone()
    charged[2, -1] = 2.5
    out_neutral = policy(**common, action_base_features=neutral)
    out_charged = policy(**common, action_base_features=charged)
    assert out_charged.action_base_embeddings is not None
    assert not torch.allclose(out_neutral.forward_logits, out_charged.forward_logits)


def test_action_atom_charge_features_condition_logits() -> None:
    policy = _policy().eval()
    x, x_phase, edge_index, edge_attr, active, batch = _graph()
    common = {
        "x_base": x,
        "x_phase": x_phase,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "active_dendrite_mask": active,
        "batch_index": batch,
        "exit_node_indices": torch.tensor([0]),
        "forward_action_mask": torch.ones(1, 6, dtype=torch.bool),
        "backward_action_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    neutral_atoms = torch.zeros(6, 3, 4)
    charged_atoms = neutral_atoms.clone()
    charged_atoms[4, :, -1] = torch.tensor([0.3, -0.5, 0.8])
    atom_mask = torch.ones(6, 3, dtype=torch.bool)

    out_neutral = policy(**common, action_atom_features=neutral_atoms, action_atom_mask=atom_mask)
    out_charged = policy(**common, action_atom_features=charged_atoms, action_atom_mask=atom_mask)

    assert out_charged.action_atom_embeddings is not None
    assert not torch.allclose(out_neutral.forward_logits, out_charged.forward_logits)
