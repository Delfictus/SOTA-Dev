import json
from pathlib import Path

import polars as pl


ROOT = Path("campaigns/glp1r_aleniglipron/track_b_chronological")


def test_coverage_matrix_includes_all_six_regions() -> None:
    matrix = pl.read_parquet(ROOT / "variant_manifold_coverage_matrix.parquet")
    regions = set(matrix.get_column("topology_space").unique().to_list())
    assert regions == {
        "ECD_TM1_GATEWAY",
        "TM3_TM5_CORE",
        "INTRACELLULAR_LOCK_BASIN",
        "HYDRATION_CORRIDOR",
        "TE_HUBS",
        "BISTATE_RETAINED_RESIDUES",
    }


def test_coverage_gaps_are_explicit() -> None:
    report = json.loads((ROOT / "variant_manifold_coverage_report.json").read_text())
    assert "coverage_gaps" in report

