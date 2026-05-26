from __future__ import annotations

import asyncio
from pathlib import Path

import polars as pl
import pytest
import torch
from torch_geometric.data import Batch, Data  # type: ignore[import-untyped]

from prism_dstw.hierarchical_bayes.gflownet_policy import (
    BackwardPolicyHead,
    FiberBundleGFlowNetPolicy,
    OrthogonalMessagePassing,
    PhaseResolvedFiberEmbedder,
    masked_log_softmax,
)
from prism_dstw.hierarchical_bayes.trajectory_balance import TrajectoryBalanceLoss
from prism_dstw.orchestration.rust_reward_oracle import BatchedRustOracle, OracleProposal


def test_phase_resolved_fiber_embedder_shapes() -> None:
    embedder = PhaseResolvedFiberEmbedder(phase_feature_dim=8, hidden_dim=16, embedding_dim=12)
    x_phase = torch.randn(7, 5, 8)
    output = embedder(x_phase)
    assert output.h_fiber_per_phase.shape == (7, 5, 12)
    assert output.h_fiber_summary.shape == (7, 12)
    assert output.hysteresis_embedding.shape == (7, 12)
    assert output.activation_gradient_embedding.shape == (7, 12)


def test_orthogonal_message_passing_preserves_separation() -> None:
    embedder = PhaseResolvedFiberEmbedder(phase_feature_dim=8, hidden_dim=16, embedding_dim=12)
    phase = embedder(torch.randn(5, 5, 8))
    module = OrthogonalMessagePassing(
        base_feature_dim=10,
        fiber_embedding_dim=12,
        edge_feature_dim=4,
        hidden_dim=16,
        output_dim=12,
    )
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)
    active = torch.tensor([True, False, True, False])
    output = module(
        x_base=torch.randn(5, 10),
        phase_output=phase,
        edge_index=edge_index,
        edge_attr=edge_attr,
        active_dendrite_mask=active,
    )
    assert output.h_base.shape == (5, 12)
    assert output.h_fiber.shape == (5, 12)
    assert not torch.allclose(output.h_base, output.h_fiber)


def test_action_mask_blocks_invalid_anchors_and_tb_loss_is_finite() -> None:
    model = FiberBundleGFlowNetPolicy(
        base_feature_dim=10,
        phase_feature_dim=8,
        edge_feature_dim=4,
        anchor_embeddings=torch.randn(6, 12),
        hidden_dim=16,
        embedding_dim=12,
    )
    data = Data(
        x_base=torch.randn(4, 10),
        x_phase=torch.randn(4, 5, 8),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        edge_attr=torch.randn(4, 4),
        active_dendrite_mask=torch.tensor([True, False, True, False]),
        num_nodes=4,
    )
    batch = Batch.from_data_list([data, data.clone()])
    forward_mask = torch.tensor([[True, False, True, True, False, True], [True, True, False, True, False, True]])
    output = model(
        x_base=batch.x_base,
        x_phase=batch.x_phase,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        active_dendrite_mask=batch.active_dendrite_mask,
        batch_index=batch.batch,
        exit_node_indices=torch.tensor([0, 4], dtype=torch.long),
        forward_action_mask=forward_mask,
        backward_action_mask=torch.ones((2, 1), dtype=torch.bool),
    )
    assert output.forward_probs[0, 1].item() == pytest.approx(0.0)
    selected_forward = output.forward_log_probs[torch.arange(2), torch.tensor([0, 1])].unsqueeze(1)
    selected_backward = output.backward_log_probs[:, 0].unsqueeze(1)
    tb = TrajectoryBalanceLoss()(output.log_z, selected_forward, selected_backward, torch.tensor([1.0, 2.0]))
    assert torch.isfinite(tb.loss)


def test_backward_policy_head_uses_graph_and_attachment_embeddings() -> None:
    torch.manual_seed(13)
    head = BackwardPolicyHead(embedding_dim=8, hidden_dim=16, max_backward_actions=4)
    graph_embeddings = torch.randn(2, 8)
    attachment_embeddings = torch.randn(2, 4, 8)
    mask = torch.ones((2, 4), dtype=torch.bool)

    logits = head(graph_embeddings, mask, attachment_embeddings)
    log_probs = masked_log_softmax(logits, mask)

    assert logits.shape == (2, 4)
    assert float(log_probs[:, 0].abs().sum().item()) > 0.0
    assert not torch.allclose(log_probs[0], log_probs[1])


def test_fiber_policy_backward_logits_are_attachment_conditioned() -> None:
    torch.manual_seed(17)
    model = FiberBundleGFlowNetPolicy(
        base_feature_dim=10,
        phase_feature_dim=8,
        edge_feature_dim=4,
        anchor_embeddings=torch.randn(6, 12),
        hidden_dim=16,
        embedding_dim=12,
        max_backward_actions=4,
    )
    data = Data(
        x_base=torch.randn(4, 10),
        x_phase=torch.randn(4, 5, 8),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        edge_attr=torch.randn(4, 4),
        active_dendrite_mask=torch.tensor([True, False, True, False]),
        num_nodes=4,
    )
    batch = Batch.from_data_list([data, data.clone()])
    forward_mask = torch.ones((2, 6), dtype=torch.bool)
    backward_mask = torch.ones((2, 4), dtype=torch.bool)
    attachments = torch.randn(2, 4, 12)

    output = model(
        x_base=batch.x_base,
        x_phase=batch.x_phase,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        active_dendrite_mask=batch.active_dendrite_mask,
        batch_index=batch.batch,
        exit_node_indices=torch.tensor([0, 4], dtype=torch.long),
        forward_action_mask=forward_mask,
        backward_action_mask=backward_mask,
        backward_attachment_embeddings=attachments,
    )

    assert output.backward_log_probs.shape == (2, 4)
    assert float(output.backward_log_probs[:, 0].abs().sum().item()) > 0.0


def test_rust_oracle_scores_32_trajectories_if_built() -> None:
    oracle_bin = Path("target/release/oracle_scorer")
    if not oracle_bin.exists():
        pytest.skip("oracle_scorer binary has not been built")
    survivors_path = Path(
        "campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_real512_o3a_zmatrix.parquet"
    )
    survivors = pl.read_parquet(survivors_path).sort("score", descending=True).head(32)
    proposals = [
        OracleProposal(
            anchor_id=str(row["anchor_id"]),
            canonical_smiles=str(row["canonical_smiles"]),
            trajectory_id=f"test-{idx:03d}",
        )
        for idx, row in enumerate(survivors.to_dicts())
    ]

    async def score() -> None:
        oracle = BatchedRustOracle(survivor_corpus=survivors_path, max_batch_size=32)
        result = await oracle.score_batch(proposals)
        assert result.rewards.shape == (32,)
        assert bool((result.rewards > 0).all().item())
        assert result.telemetry.invalid_reward_count == 0

    asyncio.run(score())
