from __future__ import annotations

import polars as pl
import pytest

from prism_dstw.motif.registry import MotifEntry, MotifRegistry, motif_id_for_smarts


def test_motif_registry_upsert_enrich_and_completeness(tmp_path) -> None:
    registry = MotifRegistry(tmp_path / "motifs.parquet")
    entry = MotifEntry(
        motif_id=motif_id_for_smarts("[#6]-[#8]"),
        canonical_smarts="[#6]-[#8]",
        discovery_method="TFGD",
        thermodynamic_role="LOCK_WEDGE",
        lock_geometry_contribution=0.7,
        phase_profile=[0.1, 0.2, 0.3, 0.2, 0.1],
        synthon_sources=["EN-001"],
        exit_vector_preference={2: 1.0},
        enrichment_ratio=2.1,
    ).with_completeness()

    motif_id = registry.register(entry)
    assert motif_id == entry.motif_id
    assert registry.all()[0].completeness_score > 0.0

    registry.enrich(motif_id, consensus_resilience=0.86, worst_case_resilience=0.81)
    loaded = registry.all()[0]
    assert loaded.consensus_resilience == pytest.approx(0.86)
    assert loaded.exit_vector_preference == {2: 1.0}
    assert registry.query_lock_enriched(min_enrichment=2.0)[0].motif_id == motif_id
    assert registry.query_by_phase_profile(pl.Series([0.1, 0.2, 0.3, 0.2, 0.1]).to_numpy(), 1.0e-9)
