from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import gpytorch  # type: ignore[import-untyped]
import polars as pl
import torch
from torch import Tensor, nn

from prism_dstw.hierarchical_bayes.dkl_model import DTSGDeepKernelGP
from prism_dstw.hierarchical_bayes.mpnn_embedding import DTSGGraphTensors
from prism_dstw.hierarchical_bayes.tobit_likelihood import TobitLikelihood
from tests.harness.synthetic_dtsg_generator import TENSOR_COLUMNS, generate_synthetic_dtsg


def int_values(series: pl.Series) -> list[int]:
    return [int(value) for value in series.to_list()]


def float_values(series: pl.Series) -> list[float]:
    return [float(value) for value in series.to_list()]


def bool_values(series: pl.Series) -> list[bool]:
    return [bool(value) for value in series.to_list()]


def string_values(series: pl.Series) -> list[str]:
    return [str(value) for value in series.to_list()]


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


def tensor_columns(frame: pl.DataFrame, columns: list[str]) -> Tensor:
    column_values = [float_values(frame.get_column(column)) for column in columns]
    row_values = [list(row) for row in zip(*column_values, strict=True)]
    return torch.tensor(row_values, dtype=torch.float32)


def load_graph(frame: pl.DataFrame, graph_id: str) -> tuple[DTSGGraphTensors, float, bool, float]:
    graph_df = frame.filter(pl.col("graph_id") == graph_id)
    edge_from_values = int_values(graph_df.get_column("edge_from_residue"))
    edge_to_values = int_values(graph_df.get_column("edge_to_residue"))
    residue_ids = sorted({*edge_from_values, *edge_to_values})
    node_lookup = {residue_id: node_index for node_index, residue_id in enumerate(residue_ids)}
    edge_index = torch.tensor(
        [
            [node_lookup[residue_id] for residue_id in edge_from_values],
            [node_lookup[residue_id] for residue_id in edge_to_values],
        ],
        dtype=torch.long,
    )
    graph = DTSGGraphTensors(
        node_features=node_feature_tensor(residue_ids),
        edge_index=edge_index,
        signed_te_mean=torch.tensor(float_values(graph_df.get_column("signed_te")), dtype=torch.float32),
        u_pose=torch.tensor(float_values(graph_df.get_column("u_pose")), dtype=torch.float32),
        edge_tensors=tensor_columns(graph_df, TENSOR_COLUMNS),
    )
    assay_value = float_values(graph_df.get_column("assay_value"))[0]
    is_left_censored = bool_values(graph_df.get_column("is_left_censored"))[0]
    assay_floor = float_values(graph_df.get_column("assay_floor"))[0]
    return graph, assay_value, is_left_censored, assay_floor


def load_batch(path: Path) -> tuple[list[DTSGGraphTensors], Tensor, Tensor, float]:
    frame = pl.read_parquet(path)
    graph_ids = sorted(set(string_values(frame.get_column("graph_id"))))
    graphs: list[DTSGGraphTensors] = []
    labels: list[float] = []
    censored_flags: list[bool] = []
    assay_floor = 0.0
    for graph_id in graph_ids:
        graph, assay_value, is_left_censored, graph_floor = load_graph(frame, graph_id)
        graphs.append(graph)
        labels.append(assay_value)
        censored_flags.append(is_left_censored)
        assay_floor = graph_floor
    return (
        graphs,
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(censored_flags, dtype=torch.bool),
        assay_floor,
    )


def finite_grad_norm(named_parameters: Iterable[tuple[str, nn.Parameter]], name_fragment: str) -> float:
    total_norm = 0.0
    found = False
    for name, parameter in named_parameters:
        if name_fragment not in name:
            continue
        gradient = parameter.grad
        assert gradient is not None, f"missing gradient for {name}"
        assert bool(torch.isfinite(gradient).all().item()), f"non-finite gradient for {name}"
        total_norm += float(torch.linalg.vector_norm(gradient).item())
        found = True
    assert found, f"no parameters matched {name_fragment}"
    assert total_norm > 0.0
    return total_norm


def test_dkl_elbo_and_tobit_log_ndtr_gradients(tmp_path: Path) -> None:
    torch.manual_seed(101)
    parquet_path = generate_synthetic_dtsg(tmp_path / "Chem_Perturbed_DTSG.parquet")
    graphs, labels, censored_mask, assay_floor = load_batch(parquet_path)

    assert int(censored_mask.sum().item()) == 3

    model = DTSGDeepKernelGP(
        node_feature_dim=4,
        latent_dim=4,
        hidden_channels=12,
        num_inducing=6,
    )
    likelihood = TobitLikelihood(assay_floor=assay_floor, initial_noise=0.02)
    model.train()
    likelihood.train()

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=labels.numel())
    latent_z = model.embed_graphs(graphs)
    output = model(latent_z)
    regularization = model.wasserstein_regularization(
        graphs,
        latent_z,
        regularization_lambda=0.04,
        epsilon=0.08,
        iterations=12,
    )
    loss = -mll(output, labels) + regularization
    print(f"dkl_loss={float(loss.item()):.8f} wasserstein_reg={float(regularization.item()):.8f}")

    assert bool(torch.isfinite(loss).item())
    loss.backward()

    mpnn_grad_norm = finite_grad_norm(model.named_parameters(), "feature_extractor")
    gp_grad_norm = finite_grad_norm(model.named_parameters(), "covar_module")
    likelihood_grad_norm = finite_grad_norm(likelihood.named_parameters(), "raw_noise")
    print(
        f"mpnn_grad_norm={mpnn_grad_norm:.8f} "
        f"gp_grad_norm={gp_grad_norm:.8f} "
        f"likelihood_grad_norm={likelihood_grad_norm:.8f}"
    )

    probe_mean = torch.zeros_like(labels, requires_grad=True)
    probe_variance = torch.full_like(labels, 0.04)
    censored_log_prob = likelihood.tobit_log_prob(labels, probe_mean, probe_variance)[censored_mask].sum()
    censored_objective = censored_log_prob * -1.0
    torch.autograd.backward(censored_objective)

    censored_gradients = probe_mean.grad
    assert censored_gradients is not None
    assert bool(torch.isfinite(censored_gradients[censored_mask]).all().item())
    assert bool((censored_gradients[censored_mask].abs() > 0.0).all().item())
