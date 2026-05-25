from prism_dstw.orchestration.rust_reward_oracle import (
    DEFAULT_BATCH_PATH,
    DEFAULT_REWARD_PATH,
    SurvivorCorpusOracle,
)


def test_survivor_oracle_uses_instance_isolated_scratch_paths() -> None:
    oracle_a = SurvivorCorpusOracle()
    oracle_b = SurvivorCorpusOracle()

    assert oracle_a.batch_path != DEFAULT_BATCH_PATH
    assert oracle_a.reward_path != DEFAULT_REWARD_PATH
    assert oracle_b.batch_path != DEFAULT_BATCH_PATH
    assert oracle_b.reward_path != DEFAULT_REWARD_PATH
    assert oracle_a.batch_path != oracle_b.batch_path
    assert oracle_a.reward_path != oracle_b.reward_path
    assert oracle_a.batch_path.name == "oracle_batch.parquet"
    assert oracle_a.reward_path.name == "oracle_rewards.parquet"
    assert oracle_a.batch_path.parent == oracle_a.reward_path.parent
    assert "oracle_runs" in oracle_a.batch_path.parts
