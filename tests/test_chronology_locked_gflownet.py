import json
from pathlib import Path

import polars as pl


REPORT = Path("campaigns/glp1r_aleniglipron/track_b_chronological/chronology_locked_training_report.json")
CANDIDATES = Path("campaigns/glp1r_aleniglipron/track_b_chronological/chronology_locked_top_100_candidates.parquet")


def test_chronology_training_emits_finite_losses() -> None:
    payload = json.loads(REPORT.read_text())
    assert payload["runtime_training_mode"] == "LIVE_ORACLE_CANDIDATE_POLICY_TB"
    assert payload["live_oracle_runtime_scored"] is True
    assert payload["optimizer_steps"] == 50
    assert payload["epochs_completed"] == 50
    final = payload["final_epoch"]
    assert final["tb_loss"] > 0.0
    assert final["reward_mean"] == final["reward_mean"]
    assert final["unique_smiles"] > 100
    assert final["dot_smiles_count"] == 0
    assert final["continuity_admissibility_rate"] >= 0.0
    assert final["backward_log_prob_std"] >= 0.0
    assert payload["candidate_action_count"] > 100


def test_chronology_candidates_carry_live_continuity_scores() -> None:
    frame = pl.read_parquet(CANDIDATES)
    required = {
        "coordinates_json",
        "continuity_admissibility",
        "nma_disruption_penalty",
        "hydration_blockade_penalty",
        "thermodynamic_trap_penalty",
        "continuity_reward_v1",
        "u_pose",
        "u_pose_input",
        "u_pose_provenance",
        "track_b_chronology_locked_score",
        "runtime_training_mode",
        "live_oracle_runtime_scored",
    }
    assert frame.height == 100
    assert required.issubset(frame.columns)
    assert frame.get_column("live_oracle_runtime_scored").all()
    u_pose_max = frame.select(pl.col("u_pose").max()).item()
    u_pose_input_max = frame.select(pl.col("u_pose_input").max()).item()
    assert isinstance(u_pose_max, (int, float))
    assert isinstance(u_pose_input_max, (int, float))
    assert u_pose_max > 0.0
    assert u_pose_input_max > 0.0
    assert set(frame.get_column("u_pose_provenance").unique().to_list()) == {
        "best_rotamer_rank_proxy_from_track_a_survivors"
    }
    assert set(frame.get_column("runtime_training_mode").unique().to_list()) == {
        "LIVE_ORACLE_CANDIDATE_POLICY_TB"
    }
    assert frame.get_column("track_b_chronology_locked_score").is_not_null().all()
