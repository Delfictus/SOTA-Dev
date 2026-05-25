#!/usr/bin/env python3
"""Autonomous PRISM-FORGE daemon state machine."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Protocol, cast

import numpy as np
import polars as pl
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.chemistry.reaction_parser import ReactionRegistry, load_reaction_registry
from prism_dstw.hierarchical_bayes.gflownet_policy import (  # noqa: E402
    TrilinearAttentionGFlowNetPolicy,
    TrilinearPolicyOutput,
)
from prism_dstw.hierarchical_bayes.trajectory_balance import (  # noqa: E402
    trajectory_balance_loss_from_action_tables,
)
from prism_dstw.ontology import EpistemicClass  # noqa: E402
from prism_dstw.orchestration.reward_function import (  # noqa: E402
    RewardBreakdown,
    RewardField,
    compute_reward_3d,
    load_prism_forge_extension,
)
from prism_dstw.persistence.cloudflare_client import CloudflareManifoldClient, JsonObject  # noqa: E402


class ForgeState(str, Enum):
    INIT = "INIT"
    SAMPLE = "SAMPLE"
    FORGE = "FORGE"
    SCORE = "SCORE"
    LEARN = "LEARN"
    PERSIST = "PERSIST"


@dataclass(frozen=True)
class ForgeDaemonConfig:
    anchors_csv: Path = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/115k_curated_anchors.csv"
    reaction_rules: Path = REPO_ROOT / "00_registry/chemistry/reaction_rules.v1.yml"
    grid_mapping: Path = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json"
    signal_grid: Path = (
        REPO_ROOT
        / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/signal_grid_variance_channel.parquet"
    )
    condition_id: str = "glp1r_5VEX_WT"
    batch_size: int = 64
    trajectory_length: int = 1
    anchor_limit: int = 115_000
    embedding_dim: int = 16
    reward_threshold: float = 0.25
    learning_rate: float = 3.0e-4


@dataclass(frozen=True)
class ForgeDaemonPassResult:
    state_trace: tuple[ForgeState, ...]
    product_coordinates: np.ndarray[Any, np.dtype[np.float32]]
    product_charges: np.ndarray[Any, np.dtype[np.float32]]
    reward: RewardBreakdown
    tb_loss: float
    gradient_l1: float
    uploaded_count: int


class ForgeUploadSink(Protocol):
    async def upload_hypothesized_molecule(
        self,
        *,
        smiles: str,
        coordinates: list[list[float]],
        reward: float,
        metadata: Mapping[str, str],
    ) -> Mapping[str, object]:
        ...


class CloudflareForgeSink:
    def __init__(self, client: CloudflareManifoldClient) -> None:
        self._client = client

    async def upload_hypothesized_molecule(
        self,
        *,
        smiles: str,
        coordinates: list[list[float]],
        reward: float,
        metadata: Mapping[str, str],
    ) -> Mapping[str, object]:
        scaffold_id = f"forge-{abs(hash((smiles, round(reward, 8))))}"
        payload: JsonObject = {
            "smiles": smiles,
            "coordinates_json": json.dumps(coordinates, separators=(",", ":")),
            "reward": reward,
            "metadata": dict(metadata),
        }
        return await self._client.commit_scaffold_update(scaffold_id, expected_version=0, data=payload)


class MockForgeUploadSink:
    def __init__(self) -> None:
        self.uploads: list[Mapping[str, object]] = []

    async def upload_hypothesized_molecule(
        self,
        *,
        smiles: str,
        coordinates: list[list[float]],
        reward: float,
        metadata: Mapping[str, str],
    ) -> Mapping[str, object]:
        payload: Mapping[str, object] = {
            "smiles": smiles,
            "coordinates": coordinates,
            "reward": reward,
            "metadata": dict(metadata),
        }
        self.uploads.append(payload)
        return payload


def load_anchor_embeddings(path: Path, *, limit: int, embedding_dim: int) -> Tensor:
    frame = pl.scan_csv(path).select(["mw", "clogp"]).head(limit).collect()
    if frame.height == 0:
        raise ValueError(f"no anchors loaded from {path}")
    base = torch.tensor(frame.to_numpy(), dtype=torch.float32)
    mw = base[:, 0:1] / 500.0
    clogp = base[:, 1:2] / 6.0
    features = [mw, clogp, mw * clogp, torch.sin(mw * 3.14159), torch.cos(clogp * 3.14159)]
    while sum(int(item.shape[1]) for item in features) < embedding_dim:
        index = len(features) + 1
        features.append(torch.sin(mw * float(index)) + torch.cos(clogp * float(index)))
    embeddings = torch.cat(features, dim=1)[:, :embedding_dim]
    return (embeddings - embeddings.mean(dim=0, keepdim=True)) / embeddings.std(dim=0, keepdim=True).clamp_min(1.0e-6)


def scaffold_batch(anchor_embeddings: Tensor, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
    nodes_per_graph = 2
    repeats = (batch_size + int(anchor_embeddings.shape[0]) - 1) // int(anchor_embeddings.shape[0])
    base = anchor_embeddings.repeat((repeats, 1))[:batch_size]
    node_offsets = torch.tensor([-0.08, 0.08], dtype=torch.float32).view(1, nodes_per_graph, 1)
    node_features = (base[:, None, :].repeat(1, nodes_per_graph, 1) + node_offsets).reshape(
        batch_size * nodes_per_graph,
        int(anchor_embeddings.shape[1]),
    )
    start = torch.arange(0, batch_size * nodes_per_graph, nodes_per_graph, dtype=torch.long)
    edge_index = torch.stack([torch.cat([start, start + 1]), torch.cat([start + 1, start])], dim=0)
    batch_index = torch.arange(batch_size, dtype=torch.long).repeat_interleave(nodes_per_graph)
    return node_features, edge_index, batch_index


def reaction_mask(batch_index: Tensor, num_synthons: int, registry: ReactionRegistry) -> Tensor:
    node_ids = torch.arange(batch_index.shape[0], dtype=torch.long).view(-1, 1, 1)
    synthon_ids = torch.arange(num_synthons, dtype=torch.long).view(1, -1, 1)
    rule_ids = torch.arange(len(registry.reactions), dtype=torch.long).view(1, 1, -1)
    mask = ((node_ids + synthon_ids + rule_ids) % 3) != 0
    first_node_per_graph = torch.zeros_like(mask)
    first_node_per_graph[::2, 0, 0] = True
    return mask | first_node_per_graph


def execute_demo_reaction(rust_module: ModuleType) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
    scaffold_coordinates = np.ascontiguousarray(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    )
    scaffold_charges = np.ascontiguousarray(np.asarray([-0.2, 0.35, -0.1], dtype=np.float32))
    synthon_coordinates = np.ascontiguousarray(
        np.asarray([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    )
    synthon_charges = np.ascontiguousarray(np.asarray([-0.3, 0.1, 0.05], dtype=np.float32))
    product_coordinates, product_charges = rust_module.execute_3d_reaction(
        scaffold_coordinates,
        scaffold_charges,
        synthon_coordinates,
        synthon_charges,
        1,
        0,
        np.ascontiguousarray(np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        np.ascontiguousarray(np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)),
        1.33,
        None,
        1,
        0,
        2,
        3.14159,
    )
    return (
        cast(np.ndarray[Any, np.dtype[np.float32]], product_coordinates),
        cast(np.ndarray[Any, np.dtype[np.float32]], product_charges),
    )


async def run_one_prism_forge_daemon_pass(
    config: ForgeDaemonConfig,
    *,
    rust_module: ModuleType | None = None,
    reward_field: RewardField | None = None,
    upload_sink: ForgeUploadSink | None = None,
) -> ForgeDaemonPassResult:
    state_trace: list[ForgeState] = [ForgeState.INIT]
    module = rust_module or load_prism_forge_extension(build_if_missing=True)
    registry = load_reaction_registry(config.reaction_rules)
    field = reward_field or RewardField.from_files(
        grid_mapping=config.grid_mapping,
        signal_grid=config.signal_grid,
        condition_id=config.condition_id,
    )
    anchor_embeddings = load_anchor_embeddings(
        config.anchors_csv,
        limit=config.anchor_limit,
        embedding_dim=config.embedding_dim,
    )
    policy = TrilinearAttentionGFlowNetPolicy(
        node_feature_dim=config.embedding_dim,
        synthon_embeddings=anchor_embeddings,
        num_reaction_rules=len(registry.reactions),
        hidden_dim=32,
        embedding_dim=config.embedding_dim,
        max_backward_actions=1,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)

    state_trace.append(ForgeState.SAMPLE)
    node_features, edge_index, batch_index = scaffold_batch(anchor_embeddings, config.batch_size)
    forward_mask = reaction_mask(batch_index, int(anchor_embeddings.shape[0]), registry)
    backward_mask = torch.ones((config.batch_size, 1), dtype=torch.bool)
    output = cast(
        TrilinearPolicyOutput,
        policy(node_features, edge_index, batch_index, forward_mask, backward_mask),
    )
    action_table = output.forward_log_probs.reshape(config.batch_size, 1, -1)
    chosen_forward_actions = action_table.argmax(dim=2)
    backward_table = output.backward_log_probs.unsqueeze(1)
    chosen_backward_actions = torch.zeros((config.batch_size, 1), dtype=torch.long)

    state_trace.append(ForgeState.FORGE)
    product_coordinates, product_charges = execute_demo_reaction(module)
    if bool(np.isnan(product_coordinates).any()):
        raise ValueError("Rust execute_3d_reaction returned NaN coordinates")

    state_trace.append(ForgeState.SCORE)
    reward = compute_reward_3d(
        coordinates=product_coordinates,
        charges=product_charges,
        field=field,
        smiles="CC(=O)NC",
        rust_module=module,
    )

    state_trace.append(ForgeState.LEARN)
    terminal_rewards = torch.full((config.batch_size,), float(reward.final_reward), dtype=torch.float32)
    tb_output = trajectory_balance_loss_from_action_tables(
        log_z=output.log_z,
        forward_log_prob_tables=action_table,
        backward_log_prob_tables=backward_table,
        forward_action_indices=chosen_forward_actions,
        backward_action_indices=chosen_backward_actions,
        terminal_rewards=terminal_rewards,
    )
    optimizer.zero_grad(set_to_none=True)
    tb_output.loss.backward()  # type: ignore[no-untyped-call]
    gradient_l1 = sum(
        float(parameter.grad.detach().abs().sum().item())
        for parameter in policy.parameters()
        if parameter.grad is not None
    )
    optimizer.step()

    state_trace.append(ForgeState.PERSIST)
    sink = upload_sink
    uploaded_count = 0
    if reward.final_reward > config.reward_threshold:
        coordinates_payload = [[float(value) for value in row] for row in product_coordinates.tolist()]
        metadata = {"Epistemic_Class": EpistemicClass.HYPOTHESIZED.value}
        if sink is not None:
            await sink.upload_hypothesized_molecule(
                smiles="CC(=O)NC",
                coordinates=coordinates_payload,
                reward=reward.final_reward,
                metadata=metadata,
            )
            uploaded_count = 1
    return ForgeDaemonPassResult(
        state_trace=tuple(state_trace),
        product_coordinates=product_coordinates,
        product_charges=product_charges,
        reward=reward,
        tb_loss=float(tb_output.loss.detach().item()),
        gradient_l1=gradient_l1,
        uploaded_count=uploaded_count,
    )


async def run_prism_forge_daemon_forever(config: ForgeDaemonConfig) -> None:
    async with CloudflareManifoldClient() as client:
        sink = CloudflareForgeSink(client)
        while True:
            await run_one_prism_forge_daemon_pass(config, upload_sink=sink)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--anchor-limit", type=int, default=115_000)
    parser.add_argument("--reward-threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ForgeDaemonConfig(
        batch_size=int(args.batch_size),
        anchor_limit=int(args.anchor_limit),
        reward_threshold=float(args.reward_threshold),
    )
    if bool(args.once):
        result = asyncio.run(run_one_prism_forge_daemon_pass(config, upload_sink=MockForgeUploadSink()))
        print(f"R_sn={result.reward.final_reward:.8f} L_TB={result.tb_loss:.8f}")
        return 0
    asyncio.run(run_prism_forge_daemon_forever(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
