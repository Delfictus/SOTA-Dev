from __future__ import annotations

import polars as pl

from prism_dstw.orchestration.rust_reward_oracle import (
    BatchedRustOracle,
    LiveSignalGridOracle,
    OracleProposal,
    RustOracleError,
    SurvivorCorpusOracle,
    proposals_from_rows,
)


def test_backward_alias_points_to_survivor_corpus_oracle() -> None:
    assert BatchedRustOracle is SurvivorCorpusOracle


def test_default_oracle_paths_are_instance_isolated() -> None:
    oracle_a = SurvivorCorpusOracle(max_batch_size=2)
    oracle_b = SurvivorCorpusOracle(max_batch_size=2)

    assert oracle_a.batch_path != oracle_b.batch_path
    assert oracle_a.reward_path != oracle_b.reward_path
    assert oracle_a.batch_path.name == "oracle_batch.parquet"
    assert oracle_a.reward_path.name == "oracle_rewards.parquet"
    assert "oracle_runs" in oracle_a.batch_path.parts


def test_lock_phase_provenance_tagged_for_replicated_rows() -> None:
    oracle = SurvivorCorpusOracle(max_batch_size=2)
    df = pl.DataFrame(
        {
            "lock_occupancy_cold_hold": [1.0, 1.0],
            "lock_occupancy_ramp_up": [1.0, 2.0],
            "lock_occupancy_warm_hold": [1.0, 3.0],
            "lock_occupancy_ramp_down": [1.0, 4.0],
            "lock_occupancy_cold_return": [1.0, 5.0],
        }
    )
    tagged = oracle.annotate_lock_phase_provenance(df)
    assert tagged.get_column("lock_phase_provenance").to_list() == [
        "REPLICATED_AGGREGATE",
        "PHASE_RESOLVED",
    ]


def test_live_signal_grid_oracle_requires_coordinates() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)
    proposals = [
        OracleProposal(anchor_id="a0", canonical_smiles="CC", trajectory_id="t0"),
    ]

    try:
        oracle.prepare_batch(proposals)
    except RustOracleError as exc:
        assert "coordinates_json" in str(exc)
    else:
        raise AssertionError("live oracle accepted a proposal without coordinates_json")


def test_live_signal_grid_oracle_batch_contains_coordinates() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)
    proposals = [
        OracleProposal(
            anchor_id="a0",
            canonical_smiles="CC",
            trajectory_id="t0",
            coordinates_json="[[0.0,0.0,0.0]]",
        ),
    ]

    batch = oracle.prepare_batch(proposals)
    assert batch.get_column("coordinates_json").to_list() == ["[[0.0,0.0,0.0]]"]


def test_live_signal_grid_oracle_command_includes_survivor_reference() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)

    command = oracle.build_command()

    assert "--live-scoring" in command
    assert "--survivors" in command
    assert command[command.index("--survivors") + 1] == str(oracle.survivor_corpus)


def test_proposals_from_rows_preserves_projection_coordinates() -> None:
    rows = pl.DataFrame(
        {
            "anchor_id": ["a0"],
            "canonical_smiles": ["CC"],
            "coordinates_json": ["[[1.0,2.0,3.0]]"],
            "score_atom_offset": [4],
        }
    )

    proposals = proposals_from_rows(rows, [0])

    assert proposals[0].coordinates_json == "[[1.0,2.0,3.0]]"
    assert proposals[0].score_atom_offset == 4
