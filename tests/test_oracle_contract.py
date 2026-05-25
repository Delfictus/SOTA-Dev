from __future__ import annotations

import polars as pl

from prism_dstw.orchestration.rust_reward_oracle import (
    BatchedRustOracle,
    LiveSignalGridOracle,
    OracleProposal,
    RustOracleError,
    SurvivorCorpusOracle,
    live_score_coordinate_count,
    proposals_from_rows,
    strict_score_atom_offset,
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


def test_live_signal_grid_oracle_rejects_empty_coordinate_array() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)
    proposals = [
        OracleProposal(
            anchor_id="a0",
            canonical_smiles="CC",
            trajectory_id="t0",
            coordinates_json="[]",
        ),
    ]

    try:
        oracle.prepare_batch(proposals)
    except RustOracleError as exc:
        assert "coordinates_json" in str(exc)
    else:
        raise AssertionError("live oracle accepted an empty coordinates_json array")


def test_live_signal_grid_oracle_rejects_malformed_coordinate_rows() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)
    proposals = [
        OracleProposal(
            anchor_id="a0",
            canonical_smiles="CC",
            trajectory_id="t0",
            coordinates_json="[[1.0, 2.0]]",
        ),
    ]

    try:
        oracle.prepare_batch(proposals)
    except RustOracleError as exc:
        assert "coordinates_json" in str(exc)
    else:
        raise AssertionError("live oracle accepted malformed coordinate rows")


def test_live_signal_grid_oracle_rejects_offset_that_scores_no_atoms() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)
    proposals = [
        OracleProposal(
            anchor_id="a0",
            canonical_smiles="CC",
            trajectory_id="t0",
            coordinates_json="[[0.0,0.0,0.0]]",
            score_atom_offset=1,
        ),
    ]

    try:
        oracle.prepare_batch(proposals)
    except RustOracleError as exc:
        assert "score_atom_offset" in str(exc)
    else:
        raise AssertionError("live oracle accepted an offset with no atoms to score")


def test_live_signal_grid_oracle_rejects_offset_removed_by_context_exclusion() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)
    proposals = [
        OracleProposal(
            anchor_id="a0",
            canonical_smiles="CC",
            trajectory_id="t0",
            coordinates_json="[[0.0,0.0,0.0],[0.1,0.0,0.0]]",
            score_atom_offset=1,
        ),
    ]

    try:
        oracle.prepare_batch(proposals)
    except RustOracleError as exc:
        assert "score_atom_offset" in str(exc)
    else:
        raise AssertionError("live oracle accepted a fragment removed by context exclusion")


def test_live_score_coordinate_count_keeps_separated_fragment_atoms() -> None:
    positions = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (6.0, 0.0, 0.0)]

    assert live_score_coordinate_count(positions, score_atom_offset=1) == 2


def test_strict_score_atom_offset_rejects_non_integer_offsets() -> None:
    for value in (True, 1.0, 1.5, -1, "1"):
        try:
            strict_score_atom_offset(value)
        except RustOracleError:
            pass
        else:
            raise AssertionError(f"accepted invalid score_atom_offset={value!r}")


def test_live_signal_grid_oracle_rejects_direct_float_offset() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)
    proposals = [
        OracleProposal(
            anchor_id="a0",
            canonical_smiles="CC",
            trajectory_id="t0",
            coordinates_json="[[0.0,0.0,0.0],[5.0,0.0,0.0]]",
            score_atom_offset=1.5,  # type: ignore[arg-type]
        ),
    ]

    try:
        oracle.prepare_batch(proposals)
    except RustOracleError as exc:
        assert "score_atom_offset" in str(exc)
    else:
        raise AssertionError("live oracle accepted direct float score_atom_offset")


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


def test_live_signal_grid_oracle_allows_duplicate_live_identities() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)
    proposals = [
        OracleProposal(
            anchor_id="a0",
            canonical_smiles="CC",
            trajectory_id="t0",
            coordinates_json="[[0.0,0.0,0.0]]",
        ),
        OracleProposal(
            anchor_id="a1",
            canonical_smiles="CC",
            trajectory_id="t1",
            coordinates_json="[[1.0,0.0,0.0]]",
        ),
    ]

    batch = oracle.prepare_batch(proposals)

    assert batch.get_column("canonical_smiles").to_list() == ["CC", "CC"]


def _live_reward_frame(*, trajectory_ids: list[str], smiles: list[str] | None = None) -> pl.DataFrame:
    count = len(trajectory_ids)
    canonical = smiles if smiles is not None else [f"C{i}" for i in range(count)]
    return pl.DataFrame(
        {
            "trajectory_id": trajectory_ids,
            "anchor_id": [f"a{i}" for i in range(count)],
            "canonical_smiles": canonical,
            "reward": [1.0 for _ in range(count)],
            "pi_complement": [1.0 for _ in range(count)],
            "adjusted_pi_clash": [0.0 for _ in range(count)],
            "pi_clash_pocket": [0.0 for _ in range(count)],
            "pi_clash_lock": [0.0 for _ in range(count)],
            "pi_clash_lock_cold_hold": [0.0 for _ in range(count)],
            "pi_clash_lock_ramp_up": [0.0 for _ in range(count)],
            "pi_clash_lock_warm_hold": [0.0 for _ in range(count)],
            "pi_clash_lock_ramp_down": [0.0 for _ in range(count)],
            "pi_clash_lock_cold_return": [0.0 for _ in range(count)],
            "lock_geometry_score": [0.0 for _ in range(count)],
            "lock_geometry_atom_count": [0.0 for _ in range(count)],
            "lock_voxel_indices_json": ["[]" for _ in range(count)],
            "lock_occupancy_cold_hold": [0.0 for _ in range(count)],
            "lock_occupancy_ramp_up": [1.0 for _ in range(count)],
            "lock_occupancy_warm_hold": [0.0 for _ in range(count)],
            "lock_occupancy_ramp_down": [0.5 for _ in range(count)],
            "lock_occupancy_cold_return": [0.0 for _ in range(count)],
            "intracellular_penetration_depth_angstrom": [0.0 for _ in range(count)],
            "lock_steric_volume_angstrom3": [0.0 for _ in range(count)],
            "cryptic_bonus": [0.0 for _ in range(count)],
            "consensus_complement_bonus": [0.0 for _ in range(count)],
            "pathway_voxels": [0 for _ in range(count)],
            "void_atom_count": [0 for _ in range(count)],
            "lock_phase_provenance": ["PHASE_RESOLVED" for _ in range(count)],
            "survival_tier": ["live_signal_grid" for _ in range(count)],
            "selected_dihedral_deg": [0.0 for _ in range(count)],
            "reward_components_json": ["{}" for _ in range(count)],
            "oracle_valid": [True for _ in range(count)],
        }
    )


def test_live_signal_grid_oracle_rejects_reordered_duplicate_reward_rows() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)
    proposals = [
        OracleProposal(anchor_id="a0", canonical_smiles="CC", trajectory_id="t0", coordinates_json="[[0,0,0]]"),
        OracleProposal(anchor_id="a1", canonical_smiles="CC", trajectory_id="t1", coordinates_json="[[1,0,0]]"),
    ]
    rewards = _live_reward_frame(trajectory_ids=["t1", "t0"], smiles=["CC", "CC"])

    try:
        oracle.validate_rewards(
            proposals=proposals,
            rewards_df=rewards,
            oracle_latency_ms=1.0,
            rust_scoring_time_ms=1.0,
            parquet_write_ms=1.0,
            parquet_read_ms=1.0,
        )
    except RustOracleError as exc:
        assert "trajectory_id" in str(exc)
    else:
        raise AssertionError("live oracle accepted reordered duplicate reward rows")


def test_live_signal_grid_oracle_validates_duplicate_rewards_by_trajectory_id() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)
    proposals = [
        OracleProposal(anchor_id="a0", canonical_smiles="CC", trajectory_id="t0", coordinates_json="[[0,0,0]]"),
        OracleProposal(anchor_id="a1", canonical_smiles="CC", trajectory_id="t1", coordinates_json="[[1,0,0]]"),
    ]
    rewards = _live_reward_frame(trajectory_ids=["t0", "t1"], smiles=["CC", "CC"])

    telemetry = oracle.validate_rewards(
        proposals=proposals,
        rewards_df=rewards,
        oracle_latency_ms=1.0,
        rust_scoring_time_ms=1.0,
        parquet_write_ms=1.0,
        parquet_read_ms=1.0,
    )

    assert telemetry.duplicate_smiles_count == 1


def test_live_signal_grid_oracle_command_includes_survivor_reference() -> None:
    oracle = LiveSignalGridOracle(max_batch_size=2)

    command = oracle.build_command()

    assert "--live-scoring" in command
    assert "--survivors" in command
    assert command[command.index("--survivors") + 1] == str(oracle.survivor_corpus)
    assert "--translation-pathway" in command
    assert command[command.index("--translation-pathway") + 1] == str(oracle.translation_pathway)


def test_live_signal_grid_oracle_none_inputs_are_explicitly_disabled() -> None:
    oracle = LiveSignalGridOracle(
        max_batch_size=2,
        shear_stress=None,
        translation_pathway=None,
        lock_mask=None,
    )

    command = oracle.build_command()

    assert "--no-shear-stress" in command
    assert "--no-translation-pathway" in command
    assert "--no-lock-mask" in command
    assert "--shear-stress" not in command
    assert "--translation-pathway" not in command
    assert "--lock-mask" not in command


def test_reward_validation_requires_live_payload_component_columns() -> None:
    oracle = SurvivorCorpusOracle(max_batch_size=2)
    proposals = [OracleProposal(anchor_id="a0", canonical_smiles="CC", trajectory_id="t0")]
    reward_row = {
        "canonical_smiles": ["CC"],
        "reward": [1.0],
        "pi_complement": [1.0],
        "adjusted_pi_clash": [0.0],
        "pi_clash_pocket": [0.0],
        "pi_clash_lock": [0.0],
        "pi_clash_lock_cold_hold": [0.0],
        "pi_clash_lock_ramp_up": [0.0],
        "pi_clash_lock_warm_hold": [0.0],
        "pi_clash_lock_ramp_down": [0.0],
        "pi_clash_lock_cold_return": [0.0],
        "lock_geometry_score": [0.0],
        "lock_geometry_atom_count": [0.0],
        "lock_voxel_indices_json": ["[]"],
        "lock_occupancy_cold_hold": [0.0],
        "lock_occupancy_ramp_up": [0.0],
        "lock_occupancy_warm_hold": [0.0],
        "lock_occupancy_ramp_down": [0.0],
        "lock_occupancy_cold_return": [0.0],
        "intracellular_penetration_depth_angstrom": [0.0],
        "lock_steric_volume_angstrom3": [0.0],
        "cryptic_bonus": [0.0],
        "consensus_complement_bonus": [0.0],
        "lock_phase_provenance": ["PHASE_RESOLVED"],
        "survival_tier": ["live_signal_grid"],
        "selected_dihedral_deg": [0.0],
        "reward_components_json": ["{}"],
        "oracle_valid": [True],
    }

    try:
        oracle.validate_rewards(
            proposals=proposals,
            rewards_df=pl.DataFrame(reward_row),
            oracle_latency_ms=1.0,
            rust_scoring_time_ms=1.0,
            parquet_write_ms=1.0,
            parquet_read_ms=1.0,
        )
    except RustOracleError as exc:
        message = str(exc)
        assert "pathway_voxels" in message
        assert "void_atom_count" in message
    else:
        raise AssertionError("oracle validation accepted rewards missing live payload columns")


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


def test_proposals_from_rows_rejects_float_score_atom_offset() -> None:
    rows = pl.DataFrame(
        {
            "anchor_id": ["a0"],
            "canonical_smiles": ["CC"],
            "coordinates_json": ["[[1.0,2.0,3.0]]"],
            "score_atom_offset": [1.5],
        }
    )

    try:
        proposals_from_rows(rows, [0])
    except RustOracleError as exc:
        assert "score_atom_offset" in str(exc)
    else:
        raise AssertionError("proposal construction accepted float score_atom_offset")
