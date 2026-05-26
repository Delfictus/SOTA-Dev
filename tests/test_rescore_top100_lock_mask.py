from __future__ import annotations

import polars as pl

from scripts.rescore_top100_lock_mask import prepare_legacy_input, phase_distinct_count


def test_prepare_legacy_input_is_idempotent_for_rescored_frames() -> None:
    frame = pl.DataFrame(
        {
            "rank": [1],
            "legacy_reward": [2.0],
            "legacy_pi_complement": [3.0],
            "legacy_adjusted_pi_clash": [1.0],
            "legacy_pi_clash_pocket": [0.5],
            "legacy_pi_clash_lock": [0.25],
            "reward": [9.0],
            "pi_complement": [8.0],
            "pi_clash_lock": [7.0],
            "lock_phase_provenance": ["REPLICATED_AGGREGATE"],
        }
    )

    cleaned = prepare_legacy_input(frame)

    assert cleaned.get_column("legacy_reward").to_list() == [2.0]
    assert "reward" not in cleaned.columns
    assert "lock_phase_provenance" not in cleaned.columns


def test_prepare_legacy_input_renames_first_run_reward_columns() -> None:
    frame = pl.DataFrame(
        {
            "rank": [1],
            "reward": [2.0],
            "pi_complement": [3.0],
            "adjusted_pi_clash": [1.0],
            "pi_clash_pocket": [0.5],
            "pi_clash_lock": [0.25],
        }
    )

    cleaned = prepare_legacy_input(frame)

    assert cleaned.get_column("legacy_reward").to_list() == [2.0]
    assert cleaned.get_column("legacy_pi_clash_lock").to_list() == [0.25]


def test_phase_distinct_count_requires_four_resolved_values_for_gate() -> None:
    row = {
        "lock_occupancy_cold_hold": 29.0,
        "lock_occupancy_ramp_up": 15.0,
        "lock_occupancy_warm_hold": 1.0,
        "lock_occupancy_ramp_down": 15.0,
        "lock_occupancy_cold_return": 30.0,
    }

    assert phase_distinct_count(row) == 4
