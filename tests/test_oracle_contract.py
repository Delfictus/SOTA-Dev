from __future__ import annotations

import polars as pl

from prism_dstw.orchestration.rust_reward_oracle import BatchedRustOracle, SurvivorCorpusOracle


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
