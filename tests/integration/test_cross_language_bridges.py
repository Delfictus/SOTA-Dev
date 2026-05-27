from __future__ import annotations

import json
import subprocess
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[2]
BRIDGE_BINARY_PACKAGES = {
    "log_subtb_tile_guard": "prism-forge",
    "oracle_scorer": "prism-forge",
    "warp_jacobian": "prism-nhs",
}


def _binary(name: str) -> Path:
    path = ROOT / "target/release" / name
    if not path.exists():
        package = BRIDGE_BINARY_PACKAGES[name]
        result = subprocess.run(
            ["cargo", "build", "-p", package, "--release", "--bin", name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    assert path.exists(), f"missing release binary: {path}"
    return path


def test_oracle_bridge_roundtrip(tmp_path: Path) -> None:
    batch = tmp_path / "oracle_batch.parquet"
    rewards = tmp_path / "oracle_rewards.parquet"
    pl.DataFrame(
        {
            "trajectory_id": ["t0", "t1", "t2"],
            "anchor_id": ["a0", "a1", "a2"],
            "canonical_smiles": ["CCO", "CCCO", "c1ccccc1"],
            "coordinates_json": [
                json.dumps([[130.0, 134.8, 110.1], [130.5, 134.8, 110.1], [131.0, 134.8, 110.1]]),
                json.dumps([[129.8, 134.5, 110.0], [130.2, 134.5, 110.0], [130.6, 134.5, 110.0], [131.0, 134.5, 110.0]]),
                json.dumps([[130.0, 135.0, 110.0], [131.0, 135.0, 110.0], [132.0, 135.0, 110.0]]),
            ],
        }
    ).write_parquet(batch)
    result = subprocess.run(
        [
            _binary("oracle_scorer").as_posix(),
            "--live-scoring",
            "--batch",
            batch.as_posix(),
            "--signal-grid",
            "campaigns/glp1r_aleniglipron/track_a_generative/signal_grid_population_consensus.parquet",
            "--grid-config",
            "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json",
            "--no-shear-stress",
            "--no-translation-pathway",
            "--no-lock-mask",
            "--output",
            rewards.as_posix(),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    frame = pl.read_parquet(rewards)
    assert frame.height == 3
    for column in ["reward", "pi_clash_pocket", "pi_complement", "survival_tier"]:
        assert column in frame.columns
    assert frame.get_column("reward").null_count() == 0
    assert set(frame.get_column("survival_tier").to_list()) == {"live_signal_grid"}


def test_subtb_tile_guard_roundtrip(tmp_path: Path) -> None:
    report = tmp_path / "rust_tile_boundary_report.json"
    result = subprocess.run(
        [
            _binary("log_subtb_tile_guard").as_posix(),
            "--registry",
            "campaigns/glp1r_aleniglipron/track_b_chronological/subtb_spectral/captured_graph_tiles/captured_tile_registry.json",
            "--operator-block-count",
            "100000",
            "--output",
            report.as_posix(),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(report.read_text())
    assert payload["status"] == "RUST_TILE_BOUNDARY_VERIFIED"
    assert payload["tile_count"] > 0


def test_subtb_artifacts_are_python_parseable() -> None:
    registry = json.loads(
        (ROOT / "campaigns/glp1r_aleniglipron/track_b_chronological/subtb_spectral/captured_graph_tiles/captured_tile_registry.json").read_text()
    )
    operator = json.loads(
        (ROOT / "campaigns/glp1r_aleniglipron/track_b_chronological/subtb_spectral/captured_graph_tiles/restricted_c6_operator_state.json").read_text()
    )
    metrics = pl.read_parquet(ROOT / "campaigns/glp1r_aleniglipron/track_b_chronological/subtb_spectral/subtb_training_metrics.parquet")
    assert len(registry["tiles"]) > 0
    assert operator["restricted_operator_target"] == "W_without_arr(Pi)"
    assert metrics.get_column("captured_tile_replay_count").sum() > 0
    assert metrics.get_column("cpu_solve_count").sum() == 0


def test_warp_jacobian_bridge_binary_is_callable() -> None:
    result = subprocess.run(
        [_binary("warp_jacobian").as_posix(), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "--bifurcate" in result.stdout
