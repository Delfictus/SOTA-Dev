from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


def test_consensus_bonus_varies_in_population_survivor_corpus() -> None:
    path = Path("campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_population_consensus_action_corpus.parquet")
    if not path.exists():
        pytest.skip(f"population consensus corpus not present: {path}")
    df = pl.read_parquet(path, columns=["consensus_complement_bonus"])
    bonuses = df.get_column("consensus_complement_bonus")
    assert float(bonuses.max()) > 0.0
    assert float(bonuses.max()) > float(bonuses.min())
