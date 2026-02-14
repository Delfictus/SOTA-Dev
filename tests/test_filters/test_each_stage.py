"""Unit tests for each filter stage independently."""
from __future__ import annotations

import pytest
from rdkit import Chem

from scripts.interfaces import (
    GeneratedMolecule,
    FilterConfig,
    PharmacophoreFeature,
    SpikePharmacophore,
    ExclusionSphere,
)
from scripts.filters import (
    stage1_validity,
    stage2_druglikeness,
    stage3_pains,
    stage4_pharmacophore,
    stage5_novelty,
    stage6_diversity,
    ranking,
)


# ── Stage 1: Validity ────────────────────────────────────────────────────

class TestStage1Validity:
    def test_valid_smiles_pass(self, valid_molecules):
        passed, rejected = stage1_validity.run(valid_molecules)
        assert len(passed) == len(valid_molecules)
        assert len(rejected) == 0

    def test_invalid_smiles_rejected(self, invalid_molecules):
        passed, rejected = stage1_validity.run(invalid_molecules)
        assert len(passed) == 0
        assert len(rejected) == len(invalid_molecules)

    def test_mixed_input(self, mixed_molecules, valid_molecules, invalid_molecules):
        passed, rejected = stage1_validity.run(mixed_molecules)
        assert len(passed) == len(valid_molecules)
        assert len(rejected) == len(invalid_molecules)

    def test_empty_input(self):
        passed, rejected = stage1_validity.run([])
        assert passed == []
        assert rejected == []

    def test_rejection_reasons_populated(self, invalid_molecules):
        _, rejected = stage1_validity.run(invalid_molecules)
        for mol, reason in rejected:
            assert isinstance(reason, str)
            assert len(reason) > 0

    def test_pentavalent_carbon_rejected(self):
        mol = GeneratedMolecule(
            smiles="C(C)(C)(C)(C)(C)", mol_block="", source="test",
            pharmacophore_match_score=0.5, matched_features=["AR"],
            generation_batch_id="bad",
        )
        passed, rejected = stage1_validity.run([mol])
        assert len(passed) == 0
        assert len(rejected) == 1


# ── Stage 2: Drug-likeness ────────────────────────────────────────────────

class TestStage2DrugLikeness:
    def test_aspirin_passes_default(self, default_filter_config):
        mol = GeneratedMolecule(
            smiles="CC(=O)Oc1ccccc1C(O)=O", mol_block="", source="test",
            pharmacophore_match_score=0.5, matched_features=["AR"],
            generation_batch_id="t",
        )
        passed, rejected = stage2_druglikeness.run([mol], default_filter_config)
        assert len(passed) == 1
        _, metrics = passed[0]
        assert metrics.qed > 0.3
        assert metrics.sa_score < 6.0
        assert metrics.lipinski_violations <= 1

    def test_huge_molecule_rejected(self, default_filter_config):
        # Very long alkane chain — fails Lipinski MW and logP
        mol = GeneratedMolecule(
            smiles="C" * 80, mol_block="", source="test",
            pharmacophore_match_score=0.5, matched_features=["HY"],
            generation_batch_id="t",
        )
        passed, rejected = stage2_druglikeness.run([mol], default_filter_config)
        assert len(rejected) == 1

    def test_strict_qed_threshold(self):
        config = FilterConfig(qed_min=0.9)  # very strict
        mol = GeneratedMolecule(
            smiles="CC(=O)Oc1ccccc1C(O)=O", mol_block="", source="test",
            pharmacophore_match_score=0.5, matched_features=["AR"],
            generation_batch_id="t",
        )
        passed, rejected = stage2_druglikeness.run([mol], config)
        # Aspirin QED ~0.55, should fail with 0.9 threshold
        assert len(rejected) == 1

    def test_compute_metrics_sanity(self):
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        metrics = stage2_druglikeness.compute_metrics(mol)
        assert 0.0 <= metrics.qed <= 1.0
        assert 1.0 <= metrics.sa_score <= 10.0
        assert metrics.lipinski_violations >= 0
        assert metrics.mw > 0

    def test_lipinski_counting(self):
        # Benzene: MW=78, logP~1.6, HBD=0, HBA=0 → 0 violations
        mol = Chem.MolFromSmiles("c1ccccc1")
        violations = stage2_druglikeness.compute_lipinski_violations(mol)
        assert violations == 0


# ── Stage 3: PAINS ────────────────────────────────────────────────────────

class TestStage3PAINS:
    def test_clean_molecule_passes(self, default_filter_config):
        mol = GeneratedMolecule(
            smiles="c1ccccc1", mol_block="", source="test",
            pharmacophore_match_score=0.5, matched_features=["AR"],
            generation_batch_id="t",
        )
        metrics = stage2_druglikeness.compute_metrics(Chem.MolFromSmiles("c1ccccc1"))
        passed, rejected = stage3_pains.run([(mol, metrics)], reject_pains=True)
        assert len(passed) == 1
        _, _, alerts = passed[0]
        assert alerts == []

    def test_rhodanine_rejected(self):
        mol = GeneratedMolecule(
            smiles="O=C1CSC(=S)N1", mol_block="", source="test",
            pharmacophore_match_score=0.5, matched_features=["HBA"],
            generation_batch_id="t",
        )
        metrics = stage2_druglikeness.compute_metrics(Chem.MolFromSmiles("O=C1CSC(=S)N1"))
        passed, rejected = stage3_pains.run([(mol, metrics)], reject_pains=True)
        assert len(rejected) == 1
        assert "PAINS" in rejected[0][1]

    def test_rhodanine_passes_when_pains_off(self):
        mol = GeneratedMolecule(
            smiles="O=C1CSC(=S)N1", mol_block="", source="test",
            pharmacophore_match_score=0.5, matched_features=["HBA"],
            generation_batch_id="t",
        )
        metrics = stage2_druglikeness.compute_metrics(Chem.MolFromSmiles("O=C1CSC(=S)N1"))
        passed, rejected = stage3_pains.run([(mol, metrics)], reject_pains=False)
        assert len(passed) == 1
        _, _, alerts = passed[0]
        assert len(alerts) > 0  # alerts still recorded

    def test_get_pains_alerts_directly(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        alerts = stage3_pains.get_pains_alerts(mol)
        assert alerts == []


# ── Stage 4: Pharmacophore ────────────────────────────────────────────────

class TestStage4Pharmacophore:
    def test_feature_extraction(self):
        mol = Chem.MolFromSmiles("c1ccc(O)cc1")  # phenol: AR + HBD
        mol = stage4_pharmacophore._ensure_3d(mol)
        feats = stage4_pharmacophore.extract_molecule_features(mol)
        types = {f[0] for f in feats}
        assert "AR" in types
        assert "HBD" in types

    def test_feature_extraction_no_conformer(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        # Don't embed — should return empty
        feats = stage4_pharmacophore.extract_molecule_features(mol)
        assert feats == []

    def test_matching_exact_overlap(self, sample_pharmacophore):
        # Create features exactly at pharmacophore positions
        mol_features = [
            (f.feature_type, (f.x, f.y, f.z))
            for f in sample_pharmacophore.features
        ]
        n_matched, matched = stage4_pharmacophore.match_pharmacophore(
            mol_features, sample_pharmacophore, distance_tolerance=1.5
        )
        assert n_matched == len(sample_pharmacophore.features)

    def test_matching_no_overlap(self, sample_pharmacophore):
        # Features far away
        mol_features = [("AR", (100.0, 100.0, 100.0))]
        n_matched, matched = stage4_pharmacophore.match_pharmacophore(
            mol_features, sample_pharmacophore, distance_tolerance=1.5
        )
        assert n_matched == 0

    def test_ensure_3d(self):
        mol = Chem.MolFromSmiles("CCO")
        mol_3d = stage4_pharmacophore._ensure_3d(mol)
        assert mol_3d is not None
        assert mol_3d.GetNumConformers() > 0

    def test_run_with_relaxed_tolerance(self, sample_pharmacophore):
        mol = GeneratedMolecule(
            smiles="c1ccc(NC(=O)c2ccccc2)cc1", mol_block="", source="test",
            pharmacophore_match_score=0.8, matched_features=["AR", "HBD"],
            generation_batch_id="t",
        )
        metrics = stage2_druglikeness.compute_metrics(Chem.MolFromSmiles(mol.smiles))
        input_list = [(mol, metrics, [])]

        # Very relaxed tolerance to ensure pass
        passed, rejected = stage4_pharmacophore.run(
            input_list, sample_pharmacophore,
            min_matches=1, distance_tolerance=100.0,
        )
        assert len(passed) == 1


# ── Stage 5: Novelty ─────────────────────────────────────────────────────

class TestStage5Novelty:
    def test_no_references_passes_all(self):
        mol = GeneratedMolecule(
            smiles="c1ccccc1", mol_block="", source="test",
            pharmacophore_match_score=0.5, matched_features=["AR"],
            generation_batch_id="t",
        )
        metrics = stage2_druglikeness.compute_metrics(Chem.MolFromSmiles("c1ccccc1"))
        input_list = [(mol, metrics, [], 3, ["AR", "HBD", "HBA"], )]

        # Explicitly pass empty reference list
        passed, rejected = stage5_novelty.run(input_list, reference_fps=[])
        assert len(passed) == 1
        _, _, _, _, _, tanimoto, cid = passed[0]
        assert tanimoto == 0.0

    def test_identical_reference_rejected(self):
        mol = GeneratedMolecule(
            smiles="c1ccccc1", mol_block="", source="test",
            pharmacophore_match_score=0.5, matched_features=["AR"],
            generation_batch_id="t",
        )
        metrics = stage2_druglikeness.compute_metrics(Chem.MolFromSmiles("c1ccccc1"))
        input_list = [(mol, metrics, [], 3, ["AR", "HBD", "HBA"])]

        # Reference is the same molecule
        ref_mol = Chem.MolFromSmiles("c1ccccc1")
        ref_fp = stage5_novelty.compute_morgan_fp(ref_mol)
        ref_fps = [("CID_BENZENE", ref_fp)]

        passed, rejected = stage5_novelty.run(
            input_list, tanimoto_max=0.85, reference_fps=ref_fps
        )
        assert len(rejected) == 1
        assert "Tanimoto" in rejected[0][1]

    def test_different_molecule_passes(self):
        mol = GeneratedMolecule(
            smiles="CC(=O)Oc1ccccc1C(O)=O", mol_block="", source="test",
            pharmacophore_match_score=0.5, matched_features=["AR"],
            generation_batch_id="t",
        )
        metrics = stage2_druglikeness.compute_metrics(Chem.MolFromSmiles(mol.smiles))
        input_list = [(mol, metrics, [], 3, ["AR", "HBD", "HBA"])]

        # Very different reference
        ref_mol = Chem.MolFromSmiles("CC(C)Cc1ccc(CC(C)C(O)=O)cc1")
        ref_fp = stage5_novelty.compute_morgan_fp(ref_mol)
        ref_fps = [("CID_IBU", ref_fp)]

        passed, rejected = stage5_novelty.run(
            input_list, tanimoto_max=0.85, reference_fps=ref_fps
        )
        assert len(passed) == 1

    def test_morgan_fp_deterministic(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        fp1 = stage5_novelty.compute_morgan_fp(mol)
        fp2 = stage5_novelty.compute_morgan_fp(mol)
        from rdkit import DataStructs
        assert DataStructs.TanimotoSimilarity(fp1, fp2) == 1.0


# ── Stage 6: Diversity ────────────────────────────────────────────────────

class TestStage6Diversity:
    def test_cluster_single_molecule(self):
        from scripts.filters.stage5_novelty import compute_morgan_fp
        mol = Chem.MolFromSmiles("c1ccccc1")
        fps = [compute_morgan_fp(mol)]
        clusters = stage6_diversity.cluster_molecules(fps, cutoff=0.4)
        assert len(clusters) == 1
        assert clusters[0] == [0]

    def test_cluster_empty(self):
        clusters = stage6_diversity.cluster_molecules([], cutoff=0.4)
        assert clusters == []

    def test_diverse_set_preserved(self):
        """Structurally diverse molecules should form separate clusters."""
        from scripts.filters.stage5_novelty import compute_morgan_fp
        smiles = [
            "c1ccccc1",                            # benzene
            "CC(=O)Oc1ccccc1C(O)=O",              # aspirin
            "CC(C)Cc1ccc(CC(C)C(O)=O)cc1",        # ibuprofen
        ]
        fps = [compute_morgan_fp(Chem.MolFromSmiles(s)) for s in smiles]
        clusters = stage6_diversity.cluster_molecules(fps, cutoff=0.4)
        assert len(clusters) == 3  # each molecule is its own cluster

    def test_run_top_n_limits_output(self):
        """Stage 6 should not return more than top_n molecules."""
        molecules = []
        for i, smi in enumerate(["c1ccccc1", "CCO", "CC=O", "CCCO", "CC(=O)O"]):
            mol = GeneratedMolecule(
                smiles=smi, mol_block="", source="test",
                pharmacophore_match_score=0.8 - i * 0.1,
                matched_features=["AR"], generation_batch_id=f"t-{i}",
            )
            metrics = stage2_druglikeness.compute_metrics(Chem.MolFromSmiles(smi))
            molecules.append((mol, metrics, [], 3, ["AR"], 0.2, "", ))

        passed, rejected = stage6_diversity.run(molecules, top_n=3)
        assert len(passed) <= 3


# ── Ranking ───────────────────────────────────────────────────────────────

class TestRanking:
    def test_pareto_simple(self):
        # A dominates B in all objectives
        objs = [(1.0, 1.0), (0.5, 0.5)]
        fronts = ranking.pareto_fronts(objs)
        assert fronts[0] == [0]
        assert fronts[1] == [1]

    def test_pareto_incomparable(self):
        # Neither dominates the other
        objs = [(1.0, 0.0), (0.0, 1.0)]
        fronts = ranking.pareto_fronts(objs)
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1}

    def test_composite_score_range(self):
        score = ranking.composite_score((1.0, 1.0, 1.0, 1.0))
        assert 0.99 <= score <= 1.01  # weights sum to 1.0

        score_zero = ranking.composite_score((0.0, 0.0, 0.0, 0.0))
        assert score_zero == 0.0

    def test_rank_candidates_empty(self):
        result = ranking.rank_candidates([], top_n=5)
        assert result == []

    def test_rank_candidates_produces_filtered_candidates(self):
        molecules = []
        for i, smi in enumerate(["c1ccccc1", "CCO"]):
            mol = GeneratedMolecule(
                smiles=smi, mol_block="", source="test",
                pharmacophore_match_score=0.8 - i * 0.1,
                matched_features=["AR"], generation_batch_id=f"t-{i}",
            )
            metrics = stage2_druglikeness.compute_metrics(Chem.MolFromSmiles(smi))
            molecules.append((mol, metrics, [], 3, ["AR"], 0.2, "CID_X", 0))

        result = ranking.rank_candidates(molecules, top_n=5)
        assert len(result) == 2
        for c in result:
            assert isinstance(c, __import__("scripts.interfaces", fromlist=["FilteredCandidate"]).FilteredCandidate)
            assert c.passed_all_filters is True
