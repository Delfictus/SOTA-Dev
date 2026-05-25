from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
from torch import Tensor

from prism_dstw.hierarchical_bayes.mpnn_embedding import DTSGGraphTensors, EiVMPNNEmbedding
from tests.harness.synthetic_dtsg_generator import generate_synthetic_dtsg


def int_values(series: pl.Series) -> list[int]:
    return [int(value) for value in series.to_list()]


def float_values(series: pl.Series) -> list[float]:
    return [float(value) for value in series.to_list()]


def node_feature_tensor(residue_ids: list[int]) -> Tensor:
    values = torch.tensor(residue_ids, dtype=torch.float32)
    scaled = values / 400.0
    return torch.stack(
        [
            scaled,
            torch.sin(scaled * torch.pi),
            torch.cos(scaled * torch.pi),
            torch.ones_like(scaled),
        ],
        dim=1,
    )


def load_graph(path: Path, graph_id: str) -> DTSGGraphTensors:
    graph_df = pl.read_parquet(path).filter(pl.col("graph_id") == graph_id)
    edge_from_values = int_values(graph_df.get_column("edge_from_residue"))
    edge_to_values = int_values(graph_df.get_column("edge_to_residue"))
    residue_ids = sorted({*edge_from_values, *edge_to_values})
    node_lookup = {residue_id: node_idx for node_idx, residue_id in enumerate(residue_ids)}
    edge_index = torch.tensor(
        [
            [node_lookup[residue_id] for residue_id in edge_from_values],
            [node_lookup[residue_id] for residue_id in edge_to_values],
        ],
        dtype=torch.long,
    )
    return DTSGGraphTensors(
        node_features=node_feature_tensor(residue_ids),
        edge_index=edge_index,
        signed_te_mean=torch.tensor(float_values(graph_df.get_column("signed_te")), dtype=torch.float32),
        u_pose=torch.tensor(float_values(graph_df.get_column("u_pose")), dtype=torch.float32),
    )


def signed_te_gradient_norm(graph: DTSGGraphTensors) -> float:
    torch.manual_seed(41)
    model = EiVMPNNEmbedding(input_channels=4, hidden_channels=12, output_channels=3)
    signed_te_mean = graph.signed_te_mean.detach().clone().requires_grad_(True)
    output = model(
        graph.node_features,
        graph.edge_index,
        signed_te_mean,
        graph.u_pose,
    )
    loss = output.square().mean()
    loss.backward()
    gradient = signed_te_mean.grad
    assert gradient is not None
    return float(torch.linalg.vector_norm(gradient).item())


def test_eiv_pose_uncertainty_flattens_signed_te_gradients(tmp_path: Path) -> None:
    parquet_path = generate_synthetic_dtsg(tmp_path / "Chem_Perturbed_DTSG.parquet")
    low_uncertainty_graph = load_graph(parquet_path, "synthetic_low_u_pose")
    high_uncertainty_graph = load_graph(parquet_path, "synthetic_high_u_pose")

    assert float(low_uncertainty_graph.u_pose.mean().item()) < 0.02
    assert float(high_uncertainty_graph.u_pose.mean().item()) > 2.0

    low_gradient_norm = signed_te_gradient_norm(low_uncertainty_graph)
    high_gradient_norm = signed_te_gradient_norm(high_uncertainty_graph)
    print(
        f"low_gradient_norm={low_gradient_norm:.8f} "
        f"high_gradient_norm={high_gradient_norm:.8f}"
    )

    assert high_gradient_norm < low_gradient_norm * 0.25
