from __future__ import annotations

import torch

from prism_dstw.orchestration.multi_scaffold_entropy_router import (
    MultiScaffoldEntropyRouter,
    phase_occupancy_from_fiber_bundle,
)


def test_exploration_bonus_starts_balanced() -> None:
    torch.manual_seed(7)
    router = MultiScaffoldEntropyRouter(["A", "B", "C"])
    batch = router.sample_batch(3000)
    counts = {name: batch.count(name) for name in ["A", "B", "C"]}
    for count in counts.values():
        assert 700 < count < 1300


def test_pgx_exclusion_zeroes_probability() -> None:
    torch.manual_seed(7)
    router = MultiScaffoldEntropyRouter(["A", "B", "C"], pgx_exclusions={"A316T": {"C"}})
    batch = router.sample_batch(1000, active_variant="A316T")
    assert "C" not in batch


def test_underexplored_scaffold_gets_boosted() -> None:
    torch.manual_seed(7)
    router = MultiScaffoldEntropyRouter(["A", "B", "C"])
    for index in range(1000):
        router.update("A", [f"a_{index}"], [1.0], ["amide"], [10.0])
        router.update("B", [f"b_{index}"], [1.0], ["suzuki"], [10.0])
    batch = router.sample_batch(100)
    assert batch.count("C") > 50


def test_collapsed_scaffold_gets_deprioritized() -> None:
    torch.manual_seed(7)
    router = MultiScaffoldEntropyRouter(["A", "B", "C"])
    for index in range(100):
        router.update("A", [f"a_{index}"], [1.0], ["amide"], [10.0])
        router.update("B", [f"b_{index}"], [float(index)], ["suzuki"], [10.0])
        router.update("C", [f"c_{index}"], [float(index) * 0.5], ["bh"], [10.0])
    batch = router.sample_batch(300)
    assert batch.count("A") < 80


def test_phase_occupancy_from_fiber_bundle_shape() -> None:
    fiber = torch.zeros((2, 5, 4), dtype=torch.float32)
    fiber[0, :, 2] = torch.tensor([0.0, 0.5, 1.0, 0.5, 0.25])
    fiber[1, :, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    rows = phase_occupancy_from_fiber_bundle(fiber)
    assert rows[0]["warm_hold"] == 1.0
    assert rows[1]["cold_hold"] == 1.0


def test_dstw_telemetry_contains_phase_fields() -> None:
    router = MultiScaffoldEntropyRouter(["ALENI", "ORFOR"])
    router.update(
        "ALENI",
        ["CCO"],
        [5.0],
        ["amide"],
        [10.0],
        phase_occupancy_batch=[{"cold_hold": 1.0, "warm_hold": 1.0}],
        lock_clash_phase_batch=[
            {
                "cold_hold": 8.0,
                "ramp_up": 10.0,
                "warm_hold": 12.0,
                "ramp_down": 9.0,
                "cold_return": 7.0,
            }
        ],
        channel_a_activations=[2],
        channel_b_activations=[5],
    )
    line = router.telemetry_lines()[0]
    assert "scaffold_router_dstw" in line
    assert "phase_entropy=" in line
    assert "lock_min_phase=" in line
