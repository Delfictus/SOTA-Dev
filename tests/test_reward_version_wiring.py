from __future__ import annotations

from pathlib import Path

import pytest
import torch

from prism_dstw.scoring.tripartite_bias_scorer import compute_tripartite_bias
import scripts.train_gflownet_policy as train_policy
from scripts.train_gflownet_policy import (
    action_base_features_from_table,
    action_atom_features_from_table,
    anchor_embeddings_from_table,
    compute_effective_reward_tensor,
    pose_penalty_from_row,
    pose_penalty_source_from_row,
    resolve_survivor_corpus_for_reward,
)


def _patch_survivor_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    track_dir = tmp_path / "track_a_generative"
    track_dir.mkdir()
    default_survivors = track_dir / "vspace_survivors_full_scale.parquet"
    population_survivors = track_dir / "vspace_survivors_population_consensus_action_corpus.parquet"
    scaffold_survivors = track_dir / "vspace_survivors_scaffold_consensus_action_corpus.parquet"
    for path in (default_survivors, population_survivors, scaffold_survivors):
        path.touch()
    monkeypatch.setattr(train_policy, "DEFAULT_SURVIVORS", default_survivors)
    monkeypatch.setattr(train_policy, "DEFAULT_POPULATION_CONSENSUS_SURVIVORS", population_survivors)
    monkeypatch.setattr(train_policy, "DEFAULT_SCAFFOLD_CONSENSUS_SURVIVORS", scaffold_survivors)
    return default_survivors, population_survivors, scaffold_survivors


def test_v3_consensus_uses_consensus_bonus() -> None:
    """Changing consensus_bonus_weight must change the effective reward."""

    mock_row = {
        "pi_complement": 10.0,
        "pi_clash_pocket": 2.0,
        "lock_geometry_score": 0.5,
        "sigma_shear": 0.1,
        "consensus_complement_bonus": 5.0,
    }
    raw_rewards = torch.tensor([1.0], dtype=torch.float32)
    tripartite_scores = [compute_tripartite_bias(mock_row)]

    reward_w0, metrics_w0 = compute_effective_reward_tensor(
        reward_version="v3_population_consensus",
        oracle_rows=[mock_row],
        raw_rewards=raw_rewards,
        tripartite_scores=tripartite_scores,
        consensus_bonus_weight=0.0,
    )
    reward_w2, metrics_w2 = compute_effective_reward_tensor(
        reward_version="v3_population_consensus",
        oracle_rows=[mock_row],
        raw_rewards=raw_rewards,
        tripartite_scores=tripartite_scores,
        consensus_bonus_weight=2.0,
    )

    difference = float((reward_w2 - reward_w0).item())
    assert float(reward_w2.item()) > float(reward_w0.item())
    assert abs(difference - 10.0) < 0.01
    assert metrics_w0["consensus_bonus_mean"] == 5.0
    assert metrics_w2["consensus_bonus_mean"] == 5.0


def test_v3_scaffold_consensus_uses_scaffold_bonus() -> None:
    mock_row = {
        "pi_complement": 4.0,
        "pi_clash_pocket": 1.0,
        "lock_geometry_score": 0.0,
        "consensus_complement_bonus": 0.5,
        "scaffold_consensus_bonus": 3.0,
    }
    reward, metrics = compute_effective_reward_tensor(
        reward_version="v3_scaffold_consensus",
        oracle_rows=[mock_row],
        raw_rewards=torch.tensor([1.0], dtype=torch.float32),
        tripartite_scores=[compute_tripartite_bias(mock_row)],
        consensus_bonus_weight=2.0,
    )

    assert abs(float(reward.item()) - 9.0) < 0.01
    assert metrics["consensus_bonus_mean"] == 3.0


def test_v1_base_uses_raw_oracle_reward() -> None:
    """The v1 reward path preserves the raw Rust oracle reward."""

    mock_row = {"pi_complement": 100.0, "consensus_complement_bonus": 100.0}
    raw_rewards = torch.tensor([7.25], dtype=torch.float32)
    reward, _ = compute_effective_reward_tensor(
        reward_version="v1_base",
        oracle_rows=[mock_row],
        raw_rewards=raw_rewards,
        tripartite_scores=[compute_tripartite_bias(mock_row)],
        consensus_bonus_weight=10.0,
    )

    assert torch.allclose(reward, raw_rewards)


def test_v1_base_sanitizes_non_finite_raw_oracle_reward() -> None:
    mock_row = {"pi_complement": 1.0}
    reward, _ = compute_effective_reward_tensor(
        reward_version="v1_base",
        oracle_rows=[mock_row],
        raw_rewards=torch.tensor([float("nan")], dtype=torch.float32),
        tripartite_scores=[compute_tripartite_bias(mock_row)],
        consensus_bonus_weight=1.0,
    )

    assert torch.isfinite(reward).all()
    assert float(reward.item()) > 0.0


def test_v4_full_field_penalizes_shear_hysteresis_and_pose() -> None:
    base_row = {
        "pi_complement": 10.0,
        "pi_clash_pocket": 1.0,
        "lock_geometry_score": 0.5,
        "bias_projection_score": 0.5,
        "consensus_complement_bonus": 2.0,
        "pathway_voxels_occupied": 1.0,
        "pathway_score_mean": 0.5,
        "reversibility_mean": 1.0,
        "sigma_shear_mean": 0.0,
        "u_pose": 0.0,
    }
    stressed = {
        **base_row,
        "sigma_shear_mean": 10.0,
        "reversibility_mean": 0.4,
        "u_pose": 1.5,
        "pathway_voxels_occupied": 0.0,
        "pathway_score_mean": 0.0,
    }
    raw_rewards = torch.tensor([1.0, 1.0], dtype=torch.float32)
    rewards, metrics = compute_effective_reward_tensor(
        reward_version="v4_full_field",
        oracle_rows=[base_row, stressed],
        raw_rewards=raw_rewards,
        tripartite_scores=[compute_tripartite_bias(base_row), compute_tripartite_bias(stressed)],
        consensus_bonus_weight=2.0,
    )
    assert float(rewards[0].item()) > float(rewards[1].item())
    assert metrics["shear_mean"] == 5.0
    assert metrics["pathway_voxels_occupied"] == 0.5
    assert metrics["u_pose_mean"] == 0.75


def test_reward_defaults_reject_non_finite_optional_values() -> None:
    mock_row = {
        "pi_complement": 5.0,
        "pi_clash_pocket": 1.0,
        "sigma_shear": float("nan"),
        "consensus_complement_bonus": float("inf"),
        "u_pose": float("-inf"),
    }
    rewards, metrics = compute_effective_reward_tensor(
        reward_version="v4_full_field",
        oracle_rows=[mock_row],
        raw_rewards=torch.tensor([1.0], dtype=torch.float32),
        tripartite_scores=[compute_tripartite_bias(mock_row)],
        consensus_bonus_weight=1.0,
    )

    assert torch.isfinite(rewards).all()
    assert metrics["shear_mean"] == 0.0
    assert metrics["consensus_bonus_mean"] == 0.0
    assert metrics["u_pose_mean"] == 0.0


def test_reward_accepts_directive_field_name_aliases() -> None:
    row = {
        "pi_complement": 5.0,
        "pi_clash_pocket": 1.0,
        "shear_stress": 10.0,
        "hysteresis_score": 0.75,
        "reversibility": 0.2,
        "am1bcc_charge": 0.33,
    }
    _, metrics = compute_effective_reward_tensor(
        reward_version="v4_full_field",
        oracle_rows=[row],
        raw_rewards=torch.tensor([1.0], dtype=torch.float32),
        tripartite_scores=[compute_tripartite_bias(row)],
        consensus_bonus_weight=1.0,
    )

    assert metrics["shear_mean"] == 10.0
    assert metrics["hysteresis_mean"] == 0.75
    assert metrics["reversibility_mean"] == 0.2
    assert metrics["charge_feature_mean"] == 0.33


def test_consensus_reward_auto_selects_population_survivor_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """v3/v4 runs must not silently use the WT survivor corpus when consensus data exists."""

    default_survivors, population_survivors, _ = _patch_survivor_defaults(monkeypatch, tmp_path)

    selected = resolve_survivor_corpus_for_reward(
        requested_survivors=default_survivors,
        reward_version="v4_full_field",
        signal_grid=population_survivors.with_name("signal_grid_population_consensus.parquet"),
    )

    assert selected == population_survivors


def test_scaffold_consensus_reward_auto_selects_scaffold_survivor_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    default_survivors, _, scaffold_survivors = _patch_survivor_defaults(monkeypatch, tmp_path)

    selected = resolve_survivor_corpus_for_reward(
        requested_survivors=default_survivors,
        reward_version="v3_scaffold_consensus",
        signal_grid=scaffold_survivors.with_name("signal_grid_scaffold_consensus.parquet"),
    )

    assert selected == scaffold_survivors


def test_u_pose_uses_surviving_rotamers_when_present() -> None:
    assert pose_penalty_source_from_row({"n_surviving_rotamers": 6}) == "n_surviving_rotamers"
    assert abs(pose_penalty_from_row({"n_surviving_rotamers": 6}) - 0.0) < 1.0e-8
    assert pose_penalty_from_row({"n_surviving_rotamers": 1}) > 1.5


def test_u_pose_proxy_source_is_explicit_when_survivor_count_absent() -> None:
    row = {"best_rotamer_rank": 6}
    assert pose_penalty_source_from_row(row) == "best_rotamer_rank_proxy"
    assert pose_penalty_from_row(row) > 1.5


def test_action_atom_features_preserve_per_atom_charge_lane() -> None:
    import json
    import polars as pl

    table = pl.DataFrame(
        {
            "conformer_atoms_json": [
                json.dumps(
                    [
                        {"atomic_num": 6, "partial_charge": -0.25},
                        {"atomic_num": 8, "partial_charge": 0.42},
                    ]
                )
            ]
        }
    )
    features, mask = action_atom_features_from_table(table, base_feature_dim=13)

    assert tuple(features.shape) == (1, 2, 13)
    assert tuple(mask.shape) == (1, 2)
    assert bool(mask.all().item())
    assert torch.allclose(features[0, :, -1], torch.tensor([-0.25, 0.42]))


def test_single_row_action_feature_normalization_is_finite() -> None:
    import polars as pl

    table = pl.DataFrame(
        {
            "canonical_smiles": ["CCO"],
            "n_heavy_atoms": [3.0],
            "formal_charge": [0.0],
            "partial_charge_sum": [0.1],
            "partial_charge_span": [0.2],
            "partial_charge_mean_abs": [0.15],
        }
    )

    anchors = anchor_embeddings_from_table(table, embedding_dim=8)
    base = action_base_features_from_table(table, base_feature_dim=13)

    assert torch.isfinite(anchors).all()
    assert torch.isfinite(base).all()


def test_empty_action_table_fails_clearly() -> None:
    import polars as pl
    import pytest

    table = pl.DataFrame(
        schema={
            "canonical_smiles": pl.String,
            "n_heavy_atoms": pl.Float64,
            "formal_charge": pl.Float64,
            "partial_charge_sum": pl.Float64,
            "partial_charge_span": pl.Float64,
            "partial_charge_mean_abs": pl.Float64,
        }
    )

    with pytest.raises(ValueError, match="action table"):
        anchor_embeddings_from_table(table, embedding_dim=8)
    with pytest.raises(ValueError, match="action table"):
        action_base_features_from_table(table, base_feature_dim=13)
