from pathlib import Path

import polars as pl
import pytest

from scripts.audit_pgx_resilience import DEFAULT_SURVIVORS
from scripts.repair_wt_projection_parity import (
    build_parity_metrics,
    load_projection_frame,
    native_wt_reward_from_candidate,
    resolve_survivor_corpus_for_candidates,
)


def test_default_survivor_corpus_uses_candidate_training_corpus(tmp_path: Path) -> None:
    preferred = tmp_path / "candidate_recorded_survivors.parquet"
    preferred.write_bytes(b"placeholder")
    candidates = tmp_path / "candidates.parquet"
    pl.DataFrame({"training_survivor_corpus": [str(preferred), str(preferred)]}).write_parquet(candidates)

    resolved, source = resolve_survivor_corpus_for_candidates(candidates, DEFAULT_SURVIVORS)

    assert resolved == preferred
    assert source == "candidate_training_survivor_corpus"


def test_default_survivor_corpus_rejects_multiple_candidate_corpora(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    first.write_bytes(b"placeholder")
    second.write_bytes(b"placeholder")
    candidates = tmp_path / "candidates.parquet"
    pl.DataFrame({"training_survivor_corpus": [str(first), str(second)]}).write_parquet(candidates)

    with pytest.raises(RuntimeError, match="multiple training_survivor_corpus"):
        resolve_survivor_corpus_for_candidates(candidates, DEFAULT_SURVIVORS)


def test_default_survivor_corpus_rejects_missing_candidate_corpus(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    candidates = tmp_path / "candidates.parquet"
    pl.DataFrame({"training_survivor_corpus": [str(missing)]}).write_parquet(candidates)

    with pytest.raises(RuntimeError, match="does not exist"):
        resolve_survivor_corpus_for_candidates(candidates, DEFAULT_SURVIVORS)


def test_explicit_survivor_corpus_overrides_candidate_training_corpus(tmp_path: Path) -> None:
    preferred = tmp_path / "candidate_recorded_survivors.parquet"
    explicit = tmp_path / "cli_survivors.parquet"
    preferred.write_bytes(b"placeholder")
    explicit.write_bytes(b"placeholder")
    candidates = tmp_path / "candidates.parquet"
    pl.DataFrame({"training_survivor_corpus": [str(preferred)]}).write_parquet(candidates)

    resolved, source = resolve_survivor_corpus_for_candidates(candidates, explicit)

    assert resolved == explicit
    assert source == "cli_argument"


def test_raw_wt_parity_is_native_rust_not_coordinate_projection() -> None:
    metrics = build_parity_metrics(
        stored_rewards=[16.0, 14.0],
        stored_native_rewards=[10.0, 8.0],
        rust_rewards=[10.0, 8.0],
        projection_rewards=[1.0e-8, 1.0e-8],
    )

    assert metrics["raw_projection_status"] == "WT_NATIVE_RAW_PARITY_CONFIRMED"
    assert metrics["raw_wt_projection_ratio_mean"] == pytest.approx(1.0)
    assert metrics["stored_reward_status"] == "STORED_REWARD_INCLUDES_NON_WT_BONUSES"
    assert metrics["coordinate_projection_status"] == "WT_COORDINATE_PROJECTION_COLLAPSE"
    assert float(metrics["coordinate_projection_vs_native_ratio_mean"]) < 1.0e-6


def test_raw_wt_parity_fails_when_stored_native_reward_mismatches_rust() -> None:
    metrics = build_parity_metrics(
        stored_rewards=[999.0, -999.0],
        stored_native_rewards=[9.0, 19.0],
        rust_rewards=[10.0, 20.0],
        projection_rewards=[0.0, 0.0],
    )

    assert metrics["raw_projection_status"] == "WT_NATIVE_RAW_PARITY_FAILED"
    assert metrics["raw_wt_projection_ratio_mean"] != pytest.approx(1.0)


def test_native_wt_reward_removes_consensus_bonus() -> None:
    assert native_wt_reward_from_candidate(
        {
            "reward": 16.0,
            "consensus_bonus_weight": 2.0,
            "scaffold_consensus_bonus": 3.0,
        }
    ) == pytest.approx(10.0)


def test_native_wt_reward_accepts_duplicate_equal_bonus_columns() -> None:
    assert native_wt_reward_from_candidate(
        {
            "reward": 16.0,
            "consensus_bonus_weight": 2.0,
            "scaffold_consensus_bonus": 3.0,
            "consensus_complement_bonus": 3.0,
        }
    ) == pytest.approx(10.0)


def test_native_wt_reward_rejects_ambiguous_bonus_columns() -> None:
    with pytest.raises(RuntimeError, match="ambiguous consensus bonus"):
        native_wt_reward_from_candidate(
            {
                "reward": 20.0,
                "consensus_bonus_weight": 2.0,
                "scaffold_consensus_bonus": 1.0,
                "consensus_complement_bonus": 5.0,
            }
        )


def test_native_wt_reward_rejects_corrupt_bonus_values() -> None:
    for field_value in (float("nan"), float("inf"), -1.0):
        with pytest.raises(RuntimeError):
            native_wt_reward_from_candidate(
                {
                    "reward": 20.0,
                    "consensus_bonus_weight": 2.0,
                    "scaffold_consensus_bonus": field_value,
                }
            )


def test_native_wt_reward_rejects_negative_bonus_weight() -> None:
    with pytest.raises(RuntimeError, match="consensus_bonus_weight"):
        native_wt_reward_from_candidate(
            {
                "reward": 20.0,
                "consensus_bonus_weight": -1.0,
                "scaffold_consensus_bonus": 3.0,
            }
        )


def test_projection_uses_candidate_coordinates_before_duplicate_survivors(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    survivors = tmp_path / "survivors.parquet"
    grid = tmp_path / "grid.parquet"
    mapping = tmp_path / "grid.json"
    pl.DataFrame(
        {
            "canonical_smiles": ["CC"],
            "anchor_id": ["a0"],
            "reward": [1.0],
            "coordinates_json": ["[[0.0,0.0,0.0]]"],
        }
    ).write_parquet(candidates)
    pl.DataFrame(
        {
            "canonical_smiles": ["CC", "CC"],
            "coordinates_json": ["[[99.0,0.0,0.0]]", "[[100.0,0.0,0.0]]"],
        }
    ).write_parquet(survivors)
    mapping.write_text(
        '{"conditions":{"WT":{"nx":2,"ny":2,"nz":2,"origin_xyz_angstrom":[0,0,0],"spacing_angstrom":1.0}}}',
        encoding="utf-8",
    )
    pl.DataFrame(
        {
            "condition_id": ["WT"],
            "voxel_idx": [0],
            "variance_class": ["thermally_activated"],
            "hit_count_cold_mean": [2.0],
            "hit_count_warm_mean": [1.0],
        }
    ).write_parquet(grid)

    result = load_projection_frame(
        candidates=candidates,
        survivors=survivors,
        signal_grid=grid,
        grid_mapping=mapping,
        wt_condition="WT",
        n=1,
    )

    assert result.get_column("projection_reward_WT").to_list() == [1.0]


def test_projection_fallback_rejects_conflicting_duplicate_survivor_coordinates(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    survivors = tmp_path / "survivors.parquet"
    grid = tmp_path / "grid.parquet"
    mapping = tmp_path / "grid.json"
    pl.DataFrame({"canonical_smiles": ["CC"], "anchor_id": ["a0"], "reward": [1.0]}).write_parquet(candidates)
    pl.DataFrame(
        {
            "canonical_smiles": ["CC", "CC"],
            "coordinates_json": ["[[0.0,0.0,0.0]]", "[[1.0,0.0,0.0]]"],
        }
    ).write_parquet(survivors)
    mapping.write_text(
        '{"conditions":{"WT":{"nx":2,"ny":2,"nz":2,"origin_xyz_angstrom":[0,0,0],"spacing_angstrom":1.0}}}',
        encoding="utf-8",
    )
    pl.DataFrame(
        {
            "condition_id": ["WT"],
            "voxel_idx": [0],
            "variance_class": ["thermally_activated"],
            "hit_count_cold_mean": [2.0],
            "hit_count_warm_mean": [1.0],
        }
    ).write_parquet(grid)

    with pytest.raises(RuntimeError, match="duplicate canonical_smiles"):
        load_projection_frame(
            candidates=candidates,
            survivors=survivors,
            signal_grid=grid,
            grid_mapping=mapping,
            wt_condition="WT",
            n=1,
        )
