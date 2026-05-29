from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from scripts.build_continuity_maps import _build_hydration


ROOT = Path(__file__).resolve().parents[1]


def test_v2_hydration_extractor_attaches_dstw_context(tmp_path: Path) -> None:
    event_path = tmp_path / "spike_events_snr_masked.parquet"
    out_dir = tmp_path / "out"
    pl.DataFrame(
        {
            "condition_id": ["c1", "c1", "c2"],
            "primary_residue_idx": [262, 262, 111],
            "voxel_idx": [700, 700, 999],
            "water_density": [1.0, 3.0, 2.0],
            "wd_change": [0.5, -1.0, 0.25],
            "intensity": [2.0, 4.0, 8.0],
        }
    ).write_parquet(event_path)

    topology_registry = tmp_path / "topology_region_registry.json"
    topology_registry.write_text(
        json.dumps(
            {
                "schema_version": "track_b.topology_region_registry.v1",
                "regions": {
                    "HYDRATION_CORRIDOR": {
                        "residues": [{"residue_id": "GLU262"}],
                    }
                },
            }
        )
    )

    captured_registry = tmp_path / "captured_tile_registry.json"
    captured_registry.write_text(
        json.dumps(
            {
                "schema_version": "prism.log_subtb.captured_tile_registry.v1",
                "tiles": [
                    {
                        "tile_id": "tile_hydration_001",
                        "tile_type": "hydration_channel_preservation",
                        "topology_region": "HYDRATION_CORRIDOR",
                        "perturbation_family": "HYDRATION_WIRE_PROBE",
                        "affected_voxel_ids": [700],
                        "affected_state_ids": [1, 2],
                        "affected_bsr_blocks": [3, 4],
                        "delta_values": [[[0.1]], [[-0.05]]],
                        "restricted_operator_target": "W_without_arr(Pi)",
                        "capture_shape_bucket": "rows2_blocks2_block1_float64",
                        "cuda_graph_id": "cuda_graph::test",
                        "tile_delta_hash": "tile_delta_hash",
                        "provenance_hash": "provenance_hash",
                        "topology_delta": "topology_delta_hash",
                        "basin_delta": "basin_delta_hash",
                        "restricted_operator_hash": "restricted_hash",
                        "c6_operator_hash": "c6_hash",
                        "captured_graph_tile_hash": "captured_hash",
                    }
                ],
            }
        )
    )

    spectral_metrics = tmp_path / "subtb_training_metrics.parquet"
    pl.DataFrame(
        {
            "captured_tile_replay_count": [3],
            "gpu_solve_count": [2],
            "cpu_solve_count": [0],
            "reward_cache_hit_rate": [0.5],
            "reward_event_source": ["captured_tile_delta_hashes"],
        }
    ).write_parquet(spectral_metrics)
    manifest = tmp_path / "subtb_run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "LOG_SUBTB_CAPTURED_TILE_RUNTIME_VERIFIED",
                "spectral_reward_manager": {
                    "event_trigger_source": "captured_tile_delta_hashes",
                    "reward_cache_hit_rate": 0.5,
                    "gpu_solve_count": 2,
                    "cpu_solve_count": 0,
                },
                "captured_graph_replay_count": 3,
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prism_v2_hydration_extractor.py"),
            "--event-parquet",
            str(event_path),
            "--out-dir",
            str(out_dir),
            "--topology-region-registry",
            str(topology_registry),
            "--captured-tile-registry",
            str(captured_registry),
            "--spectral-metrics",
            str(spectral_metrics),
            "--subtb-run-manifest",
            str(manifest),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dstw_context_loaded" in result.stderr

    frame = pl.read_parquet(out_dir / "hydration_statistics.parquet").sort("condition_id")
    assert frame.height == 2
    matched = frame.row(0, named=True)
    assert matched["condition_id"] == "c1"
    assert matched["topology_region"] == "HYDRATION_CORRIDOR"
    assert matched["captured_tile_ids"] == "tile_hydration_001"
    assert matched["captured_tile_match_basis"] == "voxel_idx"
    assert matched["spectral_reward_event_source"] == "captured_tile_delta_hashes"
    assert matched["spectral_gpu_solve_count"] == 2
    assert matched["dstw_integration_status"] == "DSTW_CAPTURED_GRAPH_SPECTRAL_LINKED"

    continuity = _build_hydration([out_dir / "hydration_statistics.parquet"]).sort("residue_id")
    continuity_row = continuity.row(1, named=True)
    assert continuity_row["residue_id"] == "262"
    assert continuity_row["solvent_wire_importance"] == 2.5
    assert continuity_row["occlusion_risk"] == 2.0
    assert continuity_row["topology_region"] == "HYDRATION_CORRIDOR"
    assert continuity_row["captured_graph_tile_hashes"] == "captured_hash"
