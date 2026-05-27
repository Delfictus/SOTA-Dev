from __future__ import annotations

import json
from pathlib import Path

import polars as pl


REGISTRY = Path(
    "campaigns/glp1r_aleniglipron/track_b_chronological/motif_intelligence/thermodynamic_motif_registry.parquet"
)
REPORT = Path(
    "campaigns/glp1r_aleniglipron/track_b_chronological/motif_intelligence/thermodynamic_motif_registry_report.json"
)
DOSSIER = Path("campaigns/glp1r_aleniglipron/M3_Lead_Optimization_Dossier.md")


def test_runtime_motif_registry_has_all_four_methods_and_completeness() -> None:
    payload = json.loads(REPORT.read_text())
    frame = pl.read_parquet(REGISTRY)

    assert payload["came_integrated_gradients"] is True
    assert payload["came_attention_weights_used"] is False
    assert payload["motif_count"] >= 20
    assert payload["completeness_mean"] > 0.5
    assert frame.height == payload["motif_count"]
    methods = set(frame.get_column("discovery_method").unique().to_list())
    assert {"TFGD", "CAME", "PR_MCS", "SAD"}.issubset(methods)
    assert frame.filter(pl.col("thermodynamic_role") == "LOCK_WEDGE").height > 0


def test_m3_dossier_renders_motif_intelligence_section() -> None:
    text = DOSSIER.read_text()

    assert "## 13. Thermodynamic Motif Intelligence" in text
    assert "Causal Attribution Motif Extraction via integrated gradients" in text
    assert "Synthon ordering recommendations" in text
    assert "thermodynamic motif registry" in text
