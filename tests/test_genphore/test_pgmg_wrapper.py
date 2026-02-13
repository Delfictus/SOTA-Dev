"""Tests for scripts.genphore.run_pgmg."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.genphore.run_pgmg import (
    _MAX_PGMG_FEATURES,
    _parse_pgmg_output,
    _smiles_to_mol_block,
    _truncate_pharmacophore,
    run_pgmg,
)
from scripts.genphore.spike_to_pharmacophore import convert
from scripts.interfaces import GeneratedMolecule, SpikePharmacophore

FIXTURES = Path(__file__).parent / "fixtures"
MOCK_SPIKES = FIXTURES / "mock_spike_output.json"
MOCK_BINDING = FIXTURES / "mock_binding_sites.json"

# Mock PGMG output: one SMILES per line, optional score
MOCK_PGMG_OUTPUT = """\
c1ccccc1 0.85
CC(=O)Oc1ccccc1C(=O)O 0.72
CCO 0.55
c1ccncc1 0.90
"""


@pytest.fixture
def pharmacophore():
    return convert(str(MOCK_SPIKES), str(MOCK_BINDING), target_name="TEST")


# ── _truncate_pharmacophore ──────────────────────────────────────────────

class TestTruncatePharmacophore:
    def test_no_truncation_needed(self, pharmacophore):
        # Mock fixture has 9 features, above max 8
        posp = _truncate_pharmacophore(pharmacophore)
        lines = [l for l in posp.splitlines() if l.startswith("FEATURE")]
        assert len(lines) == _MAX_PGMG_FEATURES

    def test_truncation_keeps_highest_intensity(self, pharmacophore):
        posp = _truncate_pharmacophore(pharmacophore)
        lines = [l for l in posp.splitlines() if l.startswith("FEATURE")]
        # First feature should have highest weight
        weights = []
        for line in lines:
            parts = line.split()
            weights.append(float(parts[-1]))
        assert weights == sorted(weights, reverse=True)

    def test_has_centroid_and_exclusion(self, pharmacophore):
        posp = _truncate_pharmacophore(pharmacophore)
        assert "CENTROID" in posp
        assert "EXCLUSION" in posp

    def test_small_pharmacophore_no_truncation(self):
        """Pharmacophore with <=8 features uses to_pgmg_posp directly."""
        from scripts.interfaces import PharmacophoreFeature, ExclusionSphere
        pharm = SpikePharmacophore(
            target_name="T",
            pdb_id="XXXX",
            pocket_id=0,
            features=[
                PharmacophoreFeature("AR", 1, 2, 3, 0.9, "BNZ", 10, "BNZ10", 258, 0.005),
                PharmacophoreFeature("HBD", 4, 5, 6, 0.7, "TYR", 20, "TYR20", 274, 0.01),
            ],
            exclusion_spheres=[],
            pocket_centroid=(2.5, 3.5, 4.5),
            pocket_lining_residues=[10, 20],
            prism_run_hash="abc123",
        )
        posp = _truncate_pharmacophore(pharm)
        lines = [l for l in posp.splitlines() if l.startswith("FEATURE")]
        assert len(lines) == 2


# ── _smiles_to_mol_block ────────────────────────────────────────────────

class TestSmilesToMolBlock:
    def test_valid_smiles(self):
        try:
            from rdkit import Chem
        except ImportError:
            pytest.skip("RDKit not available")

        block = _smiles_to_mol_block("c1ccccc1")
        assert len(block) > 0
        assert "V2000" in block or "V3000" in block

    def test_invalid_smiles(self):
        block = _smiles_to_mol_block("INVALID_SMILES_XXX")
        assert block == ""

    def test_complex_molecule(self):
        try:
            from rdkit import Chem
        except ImportError:
            pytest.skip("RDKit not available")

        block = _smiles_to_mol_block("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        assert len(block) > 0


# ── _parse_pgmg_output ──────────────────────────────────────────────────

class TestParsePgmgOutput:
    def test_parses_smiles_with_scores(self, pharmacophore, tmp_path):
        out = tmp_path / "output.smi"
        out.write_text(MOCK_PGMG_OUTPUT)
        mols = _parse_pgmg_output(out, "batch-1", pharmacophore)
        assert len(mols) == 4

    def test_source_tag(self, pharmacophore, tmp_path):
        out = tmp_path / "output.smi"
        out.write_text(MOCK_PGMG_OUTPUT)
        mols = _parse_pgmg_output(out, "batch-1", pharmacophore)
        for m in mols:
            assert m.source == "pgmg"

    def test_match_score_parsed(self, pharmacophore, tmp_path):
        out = tmp_path / "output.smi"
        out.write_text(MOCK_PGMG_OUTPUT)
        mols = _parse_pgmg_output(out, "batch-1", pharmacophore)
        assert mols[0].pharmacophore_match_score == pytest.approx(0.85)
        assert mols[3].pharmacophore_match_score == pytest.approx(0.90)

    def test_empty_file(self, pharmacophore, tmp_path):
        out = tmp_path / "output.smi"
        out.write_text("")
        mols = _parse_pgmg_output(out, "batch-1", pharmacophore)
        assert len(mols) == 0

    def test_missing_file(self, pharmacophore, tmp_path):
        out = tmp_path / "nonexistent.smi"
        mols = _parse_pgmg_output(out, "batch-1", pharmacophore)
        assert len(mols) == 0

    def test_smiles_only_no_score(self, pharmacophore, tmp_path):
        out = tmp_path / "output.smi"
        out.write_text("c1ccccc1\nCCO\n")
        mols = _parse_pgmg_output(out, "batch-1", pharmacophore)
        assert len(mols) == 2
        assert mols[0].pharmacophore_match_score == 0.0

    def test_batch_id(self, pharmacophore, tmp_path):
        out = tmp_path / "output.smi"
        out.write_text("CCO 0.5\n")
        mols = _parse_pgmg_output(out, "test-batch", pharmacophore)
        assert mols[0].generation_batch_id == "test-batch"


# ── run_pgmg ────────────────────────────────────────────────────────────

class TestRunPgmg:
    def test_missing_tool_dir_raises(self, pharmacophore):
        with pytest.raises(FileNotFoundError, match="PGMG not found"):
            run_pgmg(pharmacophore, pgmg_dir="/nonexistent/PGMG")

    def test_subprocess_called_correctly(self, pharmacophore, tmp_path):
        tool_dir = tmp_path / "PGMG"
        tool_dir.mkdir()
        (tool_dir / "generate.py").write_text("")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        smi_output = output_dir / "pgmg_output.smi"

        def fake_run(cmd, **kwargs):
            smi_output.write_text(MOCK_PGMG_OUTPUT)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("scripts.genphore.run_pgmg.subprocess.run", side_effect=fake_run):
            mols = run_pgmg(
                pharmacophore,
                n_molecules=100,
                output_dir=str(output_dir),
                pgmg_dir=str(tool_dir),
            )

        assert len(mols) == 4
        for m in mols:
            assert m.source == "pgmg"

    def test_posp_file_written(self, pharmacophore, tmp_path):
        tool_dir = tmp_path / "PGMG"
        tool_dir.mkdir()
        (tool_dir / "generate.py").write_text("")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        smi_output = output_dir / "pgmg_output.smi"

        def fake_run(cmd, **kwargs):
            smi_output.write_text(MOCK_PGMG_OUTPUT)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("scripts.genphore.run_pgmg.subprocess.run", side_effect=fake_run):
            run_pgmg(
                pharmacophore,
                output_dir=str(output_dir),
                pgmg_dir=str(tool_dir),
            )

        posp = output_dir / "pharmacophore.posp"
        assert posp.exists()
        content = posp.read_text()
        assert "FEATURE" in content
        assert "CENTROID" in content

    def test_subprocess_failure_raises(self, pharmacophore, tmp_path):
        tool_dir = tmp_path / "PGMG"
        tool_dir.mkdir()
        (tool_dir / "generate.py").write_text("")

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=1, stdout="", stderr="error")

        with patch("scripts.genphore.run_pgmg.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="PGMG failed"):
                run_pgmg(
                    pharmacophore,
                    output_dir=str(tmp_path / "out"),
                    pgmg_dir=str(tool_dir),
                )

    def test_timeout_raises(self, pharmacophore, tmp_path):
        import subprocess as sp

        tool_dir = tmp_path / "PGMG"
        tool_dir.mkdir()
        (tool_dir / "generate.py").write_text("")

        with patch(
            "scripts.genphore.run_pgmg.subprocess.run",
            side_effect=sp.TimeoutExpired(cmd="test", timeout=5),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                run_pgmg(
                    pharmacophore,
                    output_dir=str(tmp_path / "out"),
                    pgmg_dir=str(tool_dir),
                    timeout=5,
                )

    def test_no_output_file_raises(self, pharmacophore, tmp_path):
        tool_dir = tmp_path / "PGMG"
        tool_dir.mkdir()
        (tool_dir / "generate.py").write_text("")

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("scripts.genphore.run_pgmg.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="no output"):
                run_pgmg(
                    pharmacophore,
                    output_dir=str(tmp_path / "out"),
                    pgmg_dir=str(tool_dir),
                )
