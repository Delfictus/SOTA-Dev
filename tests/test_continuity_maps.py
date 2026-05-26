from pathlib import Path

import polars as pl


ROOT = Path("campaigns/glp1r_aleniglipron/track_b_chronological")


def test_all_three_continuity_maps_exist() -> None:
    for name in (
        "nma_continuity_map.parquet",
        "hydration_continuity_map.parquet",
        "thermodynamic_continuity_map.parquet",
    ):
        assert (ROOT / name).is_file()


def test_hydration_missing_is_explicitly_blocked_not_fake_observed() -> None:
    frame = pl.read_parquet(ROOT / "hydration_continuity_map.parquet")
    assert "blocked_with_hard_evidence" in frame.columns
    assert bool(frame.get_column("blocked_with_hard_evidence").any())
    assert "L5_OBSERVED" not in set(frame.get_column("provenance_class").to_list())

