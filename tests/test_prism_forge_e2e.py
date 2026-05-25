from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_dstw.orchestration.reward_function import RewardField, load_prism_forge_extension
from scripts.run_prism_forge_daemon import (
    ForgeDaemonConfig,
    MockForgeUploadSink,
    run_one_prism_forge_daemon_pass,
)


def synthetic_reward_field() -> RewardField:
    return RewardField(
        condition_id="synthetic_e2e",
        grid_dim=8,
        origin_xyz=(-2.0, -2.0, -2.0),
        spacing_A=1.0,
        voxel_indices=np.ascontiguousarray(np.asarray([140, 146, 147, 148, 155], dtype=np.uint32)),
        variance_codes=np.ascontiguousarray(np.asarray([2, 1, 2, 1, 3], dtype=np.int8)),
    )


def test_prism_forge_e2e_forward_backward_and_mock_persist() -> None:
    rust_module = load_prism_forge_extension(repo_root=REPO_ROOT, build_if_missing=True)
    sink = MockForgeUploadSink()
    config = ForgeDaemonConfig(
        batch_size=4,
        anchor_limit=16,
        embedding_dim=8,
        reward_threshold=0.0,
    )
    result = asyncio.run(
        run_one_prism_forge_daemon_pass(
            config,
            rust_module=rust_module,
            reward_field=synthetic_reward_field(),
            upload_sink=sink,
        )
    )

    assert result.product_coordinates.shape[1] == 3
    assert not bool(np.isnan(result.product_coordinates).any())
    assert result.reward.rust_reward > 0.0
    assert result.reward.final_reward > 0.0
    assert math.isfinite(result.tb_loss)
    assert result.gradient_l1 > 0.0
    assert result.uploaded_count == 1
    assert len(sink.uploads) == 1
    assert sink.uploads[0]["metadata"] == {"Epistemic_Class": "HYPOTHESIZED"}

    print(f"Thermodynamic Reward R(sn)={result.reward.final_reward:.8f} Rust_R={result.reward.rust_reward:.8f}")
    print(f"Trajectory Balance Loss L_TB={result.tb_loss:.8f}")
