from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gpu_dispatch_generation_includes_four_channels(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.parquet"
    output_dir = tmp_path / "dispatch"
    manifest = tmp_path / "dispatch_manifest.json"
    pl.DataFrame(
        {
            "candidate_id": ["cand_keep", "cand_skip"],
            "canonical_smiles": ["CCO", "CCN"],
            "lock_geometry_score": [1.5, 0.0],
            "bias_projection_score": [0.7, 0.4],
            "epistemic_confidence": ["L2", "L1"],
        }
    ).write_parquet(profiles)

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_gpu_dispatch_batch.py",
            "--profiles",
            str(profiles),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest),
            "--lock-positive-only",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dispatch_count"] == 1
    dispatch = payload["dispatches"][0]
    script_path = REPO_ROOT / dispatch["launch_script"]
    topology_json = REPO_ROOT / dispatch["topology_json"]
    text = script_path.read_text(encoding="utf-8")

    assert topology_json.is_file()
    for needle in (
        "signal_grid_differential",
        "warp_jacobian",
        "hysteresis_analysis",
        "pathway_analysis",
        "--bifurcate",
        "process_gpu_dispatch_results.py",
        "timestep_extraction",
        'cd "$REPO_ROOT"',
        "run_prism_nhs_bin hysteresis_analysis",
        "run_prism_nhs_bin pathway_analysis",
        '--frame-scope "$frame_scope"',
        '--frames "$frames"',
        'cd "$REPO_ROOT"',
        "assert_candidate_raw_root",
        "--raw-output-dir",
        "PRISM_ALLOW_SHARED_RAW_ROOT",
        '--protocol-state-summary "$protocol_state_summary"',
    ):
        assert needle in text

    subprocess.run(["bash", "-n", str(script_path)], cwd=REPO_ROOT, check=True)


def test_timestep_extraction_derives_phase_windows(tmp_path: Path) -> None:
    summary = tmp_path / "protocol_state_summary.parquet"
    output = tmp_path / "timesteps.json"
    pl.DataFrame(
        {
            "cold_hold_end": [7000, 7000],
            "ramp_end": [14000, 14000],
            "warm_hold_end": [21000, 21000],
            "ramp_down_end": [28000, 28000],
            "total_steps": [35000, 35000],
        }
    ).write_parquet(summary)

    subprocess.run(
        [
            sys.executable,
            "scripts/process_gpu_dispatch_results.py",
            "--mode",
            "timestep_extraction",
            "--protocol-state-summary",
            str(summary),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["equilibrated_frames"] == [6999, 20999, 34999]
    assert payload["ramp_frames"] == [10500, 24500]


def test_timestep_extraction_accepts_multi_condition_windows(tmp_path: Path) -> None:
    summary = tmp_path / "protocol_state_summary.parquet"
    output = tmp_path / "timesteps.json"
    pl.DataFrame(
        {
            "thermal_phase": ["Cold_Hold", "Ramp_Up", "Cold_Hold", "Ramp_Up"],
            "current_step": [6006, 7506, 8006, 9006],
            "cold_hold_end": [6000, 6000, 8000, 8000],
            "ramp_end": [12000, 12000, 14000, 14000],
            "warm_hold_end": [18000, 18000, 26000, 26000],
            "ramp_down_end": [26000, 26000, 35000, 35000],
            "total_steps": [34000, 34000, 37000, 37000],
        }
    ).write_parquet(summary)

    subprocess.run(
        [
            sys.executable,
            "scripts/process_gpu_dispatch_results.py",
            "--mode",
            "timestep_extraction",
            "--protocol-state-summary",
            str(summary),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["condition_window_count"] == 2
    assert 6006 in payload["equilibrated_frames"]
    assert 7506 in payload["ramp_frames"]


def test_timestep_extraction_accepts_phase_step_only_summary(tmp_path: Path) -> None:
    summary = tmp_path / "protocol_state_summary.parquet"
    output = tmp_path / "timesteps.json"
    pl.DataFrame(
        {
            "condition_id": ["cond_a", "cond_a", "cond_b", "cond_b"],
            "thermal_phase": ["Cold_Hold", "Ramp_Up", "Warm_Hold", "Ramp_Down"],
            "current_step": [100, 200, 300, 400],
        }
    ).write_parquet(summary)

    subprocess.run(
        [
            sys.executable,
            "scripts/process_gpu_dispatch_results.py",
            "--mode",
            "timestep_extraction",
            "--protocol-state-summary",
            str(summary),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["condition_window_count"] == 2
    assert payload["equilibrated_frames"] == [100, 300]
    assert payload["ramp_frames"] == [200, 400]


def test_timestep_extraction_accepts_real_protocol_summary_if_present(tmp_path: Path) -> None:
    summary = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/protocol_state_summary.parquet"
    if not summary.is_file():
        pytest.skip("campaign protocol_state_summary.parquet is not present")
    output = tmp_path / "timesteps.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/process_gpu_dispatch_results.py",
            "--mode",
            "timestep_extraction",
            "--protocol-state-summary",
            str(summary),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["condition_window_count"] >= 4
    assert payload["equilibrated_frames"]
    assert payload["ramp_frames"]


def test_gpu_dispatch_rejects_unsafe_candidate_id(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.parquet"
    output_dir = tmp_path / "dispatch"
    manifest = tmp_path / "dispatch_manifest.json"
    pl.DataFrame(
        {
            "candidate_id": ["cand_$(printf injected >&2)"],
            "canonical_smiles": ["CCO"],
            "lock_geometry_score": [1.5],
            "bias_projection_score": [0.7],
            "epistemic_confidence": ["L2"],
        }
    ).write_parquet(profiles)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_gpu_dispatch_batch.py",
            "--profiles",
            str(profiles),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest),
            "--lock-positive-only",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unsafe candidate_id for shell dispatch" in result.stderr


def test_ccns_validation_rejects_unsafe_candidate_id_before_writing(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ccns_validation_md.py",
            "--candidate-id",
            "../escape",
            "--sdf",
            str(tmp_path / "missing.sdf"),
            "--replicas",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
            "--raw-output-dir",
            str(tmp_path / "raw"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unsafe candidate_id for validation manifest" in result.stderr
    assert not (tmp_path / "escape.json").exists()


def test_tripartite_upgrade_requires_channel_artifacts(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/process_gpu_dispatch_results.py",
            "--mode",
            "tripartite_upgrade",
            "--candidate-id",
            "cand_ok",
            "--signal-grid",
            str(tmp_path / "missing_signal.parquet"),
            "--warp-jacobian",
            str(tmp_path / "missing_warp.parquet"),
            "--hysteresis",
            str(tmp_path / "missing_hysteresis.json"),
            "--pathway",
            str(tmp_path / "missing_pathway.json"),
            "--output",
            str(tmp_path / "upgrade.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "required artifact not found" in result.stderr
    assert not (tmp_path / "upgrade.json").exists()


def test_tripartite_upgrade_rejects_semantic_bogus_channel_artifacts(tmp_path: Path) -> None:
    signal_grid = tmp_path / "signal.parquet"
    warp = tmp_path / "warp.parquet"
    hysteresis = tmp_path / "hysteresis.json"
    pathway = tmp_path / "pathway.json"

    pl.DataFrame({"voxel_idx": [1], "unrelated": [1.0]}).write_parquet(signal_grid)
    pl.DataFrame({"voxel_idx": [1], "gradient_mean": [0.1]}).write_parquet(warp)
    hysteresis.write_text(json.dumps({"mode": "not_hysteresis"}), encoding="utf-8")
    pathway.write_text(json.dumps({"mode": "pathway_analysis"}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/process_gpu_dispatch_results.py",
            "--mode",
            "tripartite_upgrade",
            "--candidate-id",
            "cand_bad",
            "--signal-grid",
            str(signal_grid),
            "--warp-jacobian",
            str(warp),
            "--hysteresis",
            str(hysteresis),
            "--pathway",
            str(pathway),
            "--output",
            str(tmp_path / "upgrade.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "required metric column" in result.stderr


def test_tripartite_upgrade_accepts_typed_channel_artifacts(tmp_path: Path) -> None:
    signal_grid = tmp_path / "signal.parquet"
    warp = tmp_path / "warp.parquet"
    hysteresis = tmp_path / "hysteresis.json"
    pathway = tmp_path / "pathway.json"
    output = tmp_path / "upgrade.json"

    pl.DataFrame(
        {
            "voxel_idx": [1],
            "hit_count_cold_mean": [1.0],
            "hit_count_warm_mean": [0.5],
            "variance_class": ["thermally_activated"],
        }
    ).write_parquet(signal_grid)
    pl.DataFrame({"voxel_idx": [1], "gradient_mean": [0.1]}).write_parquet(warp)
    hysteresis.write_text(
        json.dumps(
            {
                "mode": "hysteresis_analysis",
                "schema_version": "prism.gpu_dispatch.hysteresis_analysis.v1",
            }
        ),
        encoding="utf-8",
    )
    pathway.write_text(
        json.dumps(
            {
                "mode": "pathway_analysis",
                "schema_version": "prism.gpu_dispatch.pathway_analysis.v1",
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/process_gpu_dispatch_results.py",
            "--mode",
            "tripartite_upgrade",
            "--candidate-id",
            "cand_ok",
            "--signal-grid",
            str(signal_grid),
            "--warp-jacobian",
            str(warp),
            "--hysteresis",
            str(hysteresis),
            "--pathway",
            str(pathway),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "assembled_from_four_channel_dispatch"
    assert payload["artifacts"]["hysteresis"]["mode"] == "hysteresis_analysis"


def test_hysteresis_rejects_empty_signal_grid(tmp_path: Path) -> None:
    signal_grid = tmp_path / "empty_signal.parquet"
    pl.DataFrame(
        {
            "voxel_idx": pl.Series("voxel_idx", [], dtype=pl.Int64),
            "hit_count_cold_mean": pl.Series("hit_count_cold_mean", [], dtype=pl.Float64),
            "hit_count_warm_mean": pl.Series("hit_count_warm_mean", [], dtype=pl.Float64),
            "hit_count_delta": pl.Series("hit_count_delta", [], dtype=pl.Float64),
            "variance_classification": pl.Series("variance_classification", [], dtype=pl.Utf8),
        }
    ).write_parquet(signal_grid)

    result = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "prism-nhs",
            "--bin",
            "hysteresis_analysis",
            "--",
            "--candidate-id",
            "cand_empty",
            "--signal-grid",
            str(signal_grid),
            "--output",
            str(tmp_path / "hysteresis.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "signal grid parquet contains zero rows" in result.stderr


def test_timestep_extraction_rejects_non_monotonic_boundaries(tmp_path: Path) -> None:
    summary = tmp_path / "protocol_state_summary.parquet"
    output = tmp_path / "timesteps.json"
    pl.DataFrame(
        {
            "cold_hold_end": [100],
            "ramp_end": [50],
            "warm_hold_end": [40],
            "ramp_down_end": [200],
            "total_steps": [80],
        }
    ).write_parquet(summary)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/process_gpu_dispatch_results.py",
            "--mode",
            "timestep_extraction",
            "--protocol-state-summary",
            str(summary),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid protocol phase boundaries" in result.stderr


def test_timestep_extraction_rejects_staggered_partial_rows(tmp_path: Path) -> None:
    summary = tmp_path / "protocol_state_summary.parquet"
    output = tmp_path / "timesteps.json"
    pl.DataFrame(
        {
            "cold_hold_end": [100, None],
            "ramp_end": [None, 200],
            "warm_hold_end": [300, None],
            "ramp_down_end": [None, 400],
            "total_steps": [500, None],
        }
    ).write_parquet(summary)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/process_gpu_dispatch_results.py",
            "--mode",
            "timestep_extraction",
            "--protocol-state-summary",
            str(summary),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "protocol phase boundaries require one complete non-null row" in result.stderr
