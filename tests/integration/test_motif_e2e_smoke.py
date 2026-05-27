from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from prism_dstw.motif.causal_attribution import extract_causal_motifs
from prism_dstw.motif.feedback import compute_motif_bonus, compute_motif_diversity_penalty
from prism_dstw.motif.functional_groups import AtomThermoAnnotation, classify_atom_roles, extract_tfg_with_neighborhood
from prism_dstw.motif.genealogy import compute_genealogy
from prism_dstw.motif.phase_resolved_mcs import extract_phase_resolved_mcs
from prism_dstw.motif.registry import MotifEntry, MotifRegistry
from prism_dstw.motif.synthon_ancestry import compute_synthon_ancestry


def _entry(smarts: str, motif_id: str = "motif_001") -> MotifEntry:
    return MotifEntry(
        motif_id=motif_id,
        canonical_smarts=smarts,
        discovery_method="TFGD",
        thermodynamic_role="LOCK_WEDGE",
        lock_geometry_contribution=1.0,
        n_occurrences_top100=2,
        n_occurrences_lock_positive=2,
        enrichment_ratio=2.0,
        first_seen_epoch=24,
        last_seen_epoch=24,
        confidence="L3",
        provenance="DERIVED",
    )


def test_motif_extraction_pipeline_imports() -> None:
    assert callable(extract_tfg_with_neighborhood)
    assert callable(extract_causal_motifs)
    assert callable(extract_phase_resolved_mcs)
    assert callable(compute_synthon_ancestry)


def test_motif_registry_crud(tmp_path: Path) -> None:
    registry = MotifRegistry(tmp_path / "test_motif_registry.parquet")
    registry.register(_entry("[C]-[N]"))
    results = registry.query_by_role("LOCK_WEDGE")
    assert len(results) == 1
    assert results[0].motif_id == "motif_001"


def test_motif_feedback_wiring(tmp_path: Path) -> None:
    registry = MotifRegistry(tmp_path / "motifs.parquet")
    entry = _entry("[C]-[O]")
    registry.register(entry)
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    assert compute_motif_bonus(mol, registry) > 0.0
    assert isinstance(compute_motif_diversity_penalty([mol, Chem.MolFromSmiles("CCN")], registry), float)


def test_motif_genealogy_runs(tmp_path: Path) -> None:
    older = MotifRegistry(tmp_path / "older.parquet")
    newer = MotifRegistry(tmp_path / "newer.parquet")
    older.register(_entry("[C]-[N]", "old"))
    newer.register(_entry("[C]-[N]-[C]", "new"))
    assert compute_genealogy(older, newer)


def test_tfgd_smoke_extracts_role_group() -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    annotations = {
        0: AtomThermoAnnotation(pi_complement=0.0, pi_clash=0.0, lock_geometry=0.0, shear_stress=0.0),
        1: AtomThermoAnnotation(pi_complement=0.0, pi_clash=0.0, lock_geometry=1.0, shear_stress=0.0),
        2: AtomThermoAnnotation(
            pi_complement=1.0,
            pi_clash=0.0,
            lock_geometry=0.0,
            shear_stress=0.0,
            phase_profile=(0.0, 0.0, 0.5, 0.75, 1.0),
        ),
    }
    roles = classify_atom_roles(annotations)
    motifs = extract_tfg_with_neighborhood(mol, roles, annotations)
    assert motifs
