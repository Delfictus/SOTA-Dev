from __future__ import annotations

import polars as pl

from prism_dstw.orchestration.rust_reward_oracle import BatchedRustOracle, SurvivorCorpusOracle


def test_backward_alias_points_to_survivor_corpus_oracle() -> None:
    assert BatchedRustOracle is SurvivorCorpusOracle


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
