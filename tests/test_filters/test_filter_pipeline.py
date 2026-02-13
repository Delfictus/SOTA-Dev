"""Integration tests for the full filter pipeline."""
from __future__ import annotations

import json
import os
import tempfile

import pytest
from rdkit import Chem

from scripts.interfaces import (
    GeneratedMolecule,
    FilterConfig,
    FilteredCandidate,
    SpikePharmacophore,
    PharmacophoreFeature,
    ExclusionSphere,
)
from scripts.filters.filter_pipeline import run_pipeline, main


class TestRunPipeline:
    """Tests for the run_pipeline() function."""

    def _make_molecules(self, smiles_list):
        return [
            GeneratedMolecule(
                smiles=smi, mol_block="", source="phoregen",
                pharmacophore_match_score=0.8 - i * 0.05,
                matched_features=["AR", "HBD", "HBA"],
                generation_batch_id=f"int-test-{i}",
            )
            for i, smi in enumerate(smiles_list)
        ]

    def test_full_pipeline_returns_candidates(self, sample_pharmacophore):
        molecules = self._make_molecules([
            "CC(=O)Oc1ccccc1C(O)=O",   # aspirin
            "CC(=O)Nc1ccc(O)cc1",       # acetaminophen
            "CC(C)Cc1ccc(CC(C)C(O)=O)cc1",  # ibuprofen
        ])

        candidates, stats = run_pipeline(
            molecules, sample_pharmacophore,
            top_n=5,
            min_pharmacophore_matches=1,
            pharmacophore_distance_tolerance=100.0,
        )

        assert isinstance(candidates, list)
        assert all(isinstance(c, FilteredCandidate) for c in candidates)
        assert stats["input"] == 3
        assert stats["stage1_passed"] == 3
        assert stats["final_output"] <= 5

    def test_all_invalid_yields_empty(self, sample_pharmacophore):
        molecules = self._make_molecules(["not_valid", "", "[bad]"])

        candidates, stats = run_pipeline(
            molecules, sample_pharmacophore, top_n=5,
        )

        assert len(candidates) == 0
        assert stats["stage1_passed"] == 0

    def test_empty_input(self, sample_pharmacophore):
        candidates, stats = run_pipeline(
            [], sample_pharmacophore, top_n=5,
        )
        assert candidates == []
        assert stats["input"] == 0

    def test_stats_accounting(self, sample_pharmacophore):
        """Each stage's passed + rejected should equal previous stage's passed."""
        molecules = self._make_molecules([
            "CC(=O)Oc1ccccc1C(O)=O",
            "c1ccccc1",
            "not_valid",
        ])

        _, stats = run_pipeline(
            molecules, sample_pharmacophore,
            top_n=5,
            min_pharmacophore_matches=1,
            pharmacophore_distance_tolerance=100.0,
        )

        assert stats["stage1_passed"] + stats["stage1_rejected"] == stats["input"]
        assert stats["stage2_passed"] + stats["stage2_rejected"] == stats["stage1_passed"]
        assert stats["stage3_passed"] + stats["stage3_rejected"] == stats["stage2_passed"]
        assert stats["stage4_passed"] + stats["stage4_rejected"] == stats["stage3_passed"]
        assert stats["stage5_passed"] + stats["stage5_rejected"] == stats["stage4_passed"]
        assert stats["stage6_passed"] + stats["stage6_rejected"] == stats["stage5_passed"]

    def test_top_n_respected(self, sample_pharmacophore):
        molecules = self._make_molecules([
            "CC(=O)Oc1ccccc1C(O)=O",
            "CC(=O)Nc1ccc(O)cc1",
            "CC(C)Cc1ccc(CC(C)C(O)=O)cc1",
            "c1ccc(NC(=O)c2ccccc2)cc1",
            "O=C(O)c1ccccc1O",
            "c1ccc(Oc2ccccc2)cc1",
        ])

        candidates, _ = run_pipeline(
            molecules, sample_pharmacophore,
            top_n=2,
            min_pharmacophore_matches=1,
            pharmacophore_distance_tolerance=100.0,
        )

        assert len(candidates) <= 2

    def test_custom_filter_config(self, sample_pharmacophore):
        molecules = self._make_molecules(["CC(=O)Oc1ccccc1C(O)=O"])

        # Very strict QED that aspirin won't pass
        strict_config = FilterConfig(qed_min=0.99)
        candidates, stats = run_pipeline(
            molecules, sample_pharmacophore,
            config=strict_config, top_n=5,
        )

        assert stats["stage2_rejected"] == 1
        assert len(candidates) == 0

    def test_candidates_serializable(self, sample_pharmacophore):
        """FilteredCandidate output must serialize to JSON."""
        molecules = self._make_molecules(["CC(=O)Oc1ccccc1C(O)=O"])

        candidates, _ = run_pipeline(
            molecules, sample_pharmacophore,
            top_n=5,
            min_pharmacophore_matches=1,
            pharmacophore_distance_tolerance=100.0,
        )

        if candidates:
            for c in candidates:
                json_str = c.to_json()
                roundtrip = FilteredCandidate.from_json(json_str)
                assert roundtrip.molecule.smiles == c.molecule.smiles
                assert roundtrip.qed_score == c.qed_score


class TestCLI:
    """Tests for the CLI entry point."""

    def _write_test_files(self, tmp_dir, sample_pharmacophore):
        """Write molecules and pharmacophore JSON for CLI testing."""
        molecules = [
            GeneratedMolecule(
                smiles="CC(=O)Oc1ccccc1C(O)=O", mol_block="", source="phoregen",
                pharmacophore_match_score=0.8, matched_features=["AR", "HBD"],
                generation_batch_id="cli-test",
            ),
            GeneratedMolecule(
                smiles="CC(=O)Nc1ccc(O)cc1", mol_block="", source="phoregen",
                pharmacophore_match_score=0.7, matched_features=["AR", "HBD"],
                generation_batch_id="cli-test",
            ),
        ]

        mol_path = os.path.join(tmp_dir, "molecules.json")
        with open(mol_path, "w") as f:
            json.dump([m.to_dict() for m in molecules], f)

        pharm_path = os.path.join(tmp_dir, "pharmacophore.json")
        with open(pharm_path, "w") as f:
            f.write(sample_pharmacophore.to_json())

        return mol_path, pharm_path

    def test_cli_basic(self, sample_pharmacophore):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mol_path, pharm_path = self._write_test_files(tmp_dir, sample_pharmacophore)
            output_path = os.path.join(tmp_dir, "output.json")

            main([
                "--molecules", mol_path,
                "--pharmacophore", pharm_path,
                "--output", output_path,
                "--top-n", "5",
                "--min-pharm-matches", "1",
                "--distance-tolerance", "100.0",
            ])

            assert os.path.exists(output_path)
            with open(output_path) as f:
                data = json.load(f)
            assert "candidates" in data
            assert "stats" in data
            assert isinstance(data["candidates"], list)

    def test_cli_with_molecules_dict_format(self, sample_pharmacophore):
        """Test loading molecules from {molecules: [...]} format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            molecules = [
                GeneratedMolecule(
                    smiles="CC(=O)Oc1ccccc1C(O)=O", mol_block="", source="phoregen",
                    pharmacophore_match_score=0.8, matched_features=["AR"],
                    generation_batch_id="cli-test",
                ),
            ]

            mol_path = os.path.join(tmp_dir, "molecules.json")
            with open(mol_path, "w") as f:
                json.dump({"molecules": [m.to_dict() for m in molecules]}, f)

            pharm_path = os.path.join(tmp_dir, "pharmacophore.json")
            with open(pharm_path, "w") as f:
                f.write(sample_pharmacophore.to_json())

            output_path = os.path.join(tmp_dir, "output.json")
            main([
                "--molecules", mol_path,
                "--pharmacophore", pharm_path,
                "--output", output_path,
                "--min-pharm-matches", "1",
                "--distance-tolerance", "100.0",
            ])

            with open(output_path) as f:
                data = json.load(f)
            assert len(data["candidates"]) >= 0  # may pass or not, but should not crash
