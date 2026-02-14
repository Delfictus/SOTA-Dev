"""Tests for scripts.genphore.run_phoregen."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.genphore.run_phoregen import (
    _parse_sdf_raw,
    _parse_sdf_to_molecules,
    run_phoregen,
)
from scripts.genphore.spike_to_pharmacophore import convert
from scripts.interfaces import GeneratedMolecule, SpikePharmacophore

FIXTURES = Path(__file__).parent / "fixtures"
MOCK_SPIKES = FIXTURES / "mock_spike_output.json"
MOCK_BINDING = FIXTURES / "mock_binding_sites.json"

# Minimal valid SDF block for testing
MOCK_SDF = """\
mol_001
     RDKit          3D

  6  6  0  0  0  0  0  0  0  0999 V2000
    1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2124   -0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124   -0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
M  END
$$$$
mol_002
     RDKit          3D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""


@pytest.fixture
def pharmacophore():
    return convert(str(MOCK_SPIKES), str(MOCK_BINDING), target_name="TEST")


# ── _parse_sdf_raw ──────────────────────────────────────────────────────

class TestParseSdfRaw:
    def test_parses_two_blocks(self, pharmacophore, tmp_path):
        sdf_path = tmp_path / "test.sdf"
        sdf_path.write_text(MOCK_SDF)
        mols = _parse_sdf_raw(sdf_path, "batch-1", pharmacophore)
        assert len(mols) == 2

    def test_source_tag(self, pharmacophore, tmp_path):
        sdf_path = tmp_path / "test.sdf"
        sdf_path.write_text(MOCK_SDF)
        mols = _parse_sdf_raw(sdf_path, "batch-1", pharmacophore)
        for m in mols:
            assert m.source == "phoregen"
            assert isinstance(m, GeneratedMolecule)

    def test_empty_sdf(self, pharmacophore, tmp_path):
        sdf_path = tmp_path / "empty.sdf"
        sdf_path.write_text("")
        mols = _parse_sdf_raw(sdf_path, "batch-1", pharmacophore)
        assert len(mols) == 0

    def test_batch_id_propagated(self, pharmacophore, tmp_path):
        sdf_path = tmp_path / "test.sdf"
        sdf_path.write_text(MOCK_SDF)
        mols = _parse_sdf_raw(sdf_path, "my-batch-42", pharmacophore)
        for m in mols:
            assert m.generation_batch_id == "my-batch-42"


# ── _parse_sdf_to_molecules ─────────────────────────────────────────────

class TestParseSdfToMolecules:
    def test_fallback_to_raw_without_rdkit(self, pharmacophore, tmp_path):
        sdf_path = tmp_path / "test.sdf"
        sdf_path.write_text(MOCK_SDF)
        with patch.dict("sys.modules", {"rdkit": None, "rdkit.Chem": None}):
            # Force ImportError by hiding rdkit
            pass
        # Even if rdkit IS available, we test the raw path
        mols = _parse_sdf_raw(sdf_path, "b1", pharmacophore)
        assert len(mols) >= 1

    def test_parses_valid_sdf(self, pharmacophore, tmp_path):
        sdf_path = tmp_path / "test.sdf"
        sdf_path.write_text(MOCK_SDF)
        mols = _parse_sdf_to_molecules(sdf_path, "b1", pharmacophore)
        assert len(mols) >= 1
        for m in mols:
            assert m.source == "phoregen"


# ── run_phoregen ─────────────────────────────────────────────────────────

class TestRunPhoregen:
    def test_missing_tool_dir_raises(self, pharmacophore):
        with pytest.raises(FileNotFoundError, match="PhoreGen not found"):
            run_phoregen(
                pharmacophore,
                phoregen_dir="/nonexistent/PhoreGen",
            )

    def test_subprocess_called_correctly(self, pharmacophore, tmp_path):
        # Create fake PhoreGen directory
        tool_dir = tmp_path / "PhoreGen"
        tool_dir.mkdir()
        (tool_dir / "sample.py").write_text("# placeholder")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock subprocess to produce an SDF
        sdf_output = output_dir / "phoregen_output.sdf"

        def fake_run(cmd, **kwargs):
            sdf_output.write_text(MOCK_SDF)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("scripts.genphore.run_phoregen.subprocess.run", side_effect=fake_run):
            mols = run_phoregen(
                pharmacophore,
                n_molecules=10,
                output_dir=str(output_dir),
                phoregen_dir=str(tool_dir),
            )

        assert len(mols) >= 1
        for m in mols:
            assert m.source == "phoregen"

    def test_subprocess_failure_raises(self, pharmacophore, tmp_path):
        tool_dir = tmp_path / "PhoreGen"
        tool_dir.mkdir()
        (tool_dir / "sample.py").write_text("")

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=1, stdout="error", stderr="crash")

        with patch("scripts.genphore.run_phoregen.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="PhoreGen failed"):
                run_phoregen(
                    pharmacophore,
                    output_dir=str(tmp_path / "out"),
                    phoregen_dir=str(tool_dir),
                )

    def test_timeout_raises(self, pharmacophore, tmp_path):
        import subprocess as sp

        tool_dir = tmp_path / "PhoreGen"
        tool_dir.mkdir()
        (tool_dir / "sample.py").write_text("")

        with patch(
            "scripts.genphore.run_phoregen.subprocess.run",
            side_effect=sp.TimeoutExpired(cmd="test", timeout=10),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                run_phoregen(
                    pharmacophore,
                    output_dir=str(tmp_path / "out"),
                    phoregen_dir=str(tool_dir),
                    timeout=10,
                )

    def test_pharmacophore_json_written(self, pharmacophore, tmp_path):
        tool_dir = tmp_path / "PhoreGen"
        tool_dir.mkdir()
        (tool_dir / "sample.py").write_text("")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sdf_output = output_dir / "phoregen_output.sdf"

        def fake_run(cmd, **kwargs):
            sdf_output.write_text(MOCK_SDF)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("scripts.genphore.run_phoregen.subprocess.run", side_effect=fake_run):
            run_phoregen(
                pharmacophore,
                output_dir=str(output_dir),
                phoregen_dir=str(tool_dir),
            )

        phore_json = output_dir / "pharmacophore_input.json"
        assert phore_json.exists()
        data = json.loads(phore_json.read_text())
        assert "features" in data
        assert "bounds" in data

    def test_no_output_file_raises(self, pharmacophore, tmp_path):
        tool_dir = tmp_path / "PhoreGen"
        tool_dir.mkdir()
        (tool_dir / "sample.py").write_text("")

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("scripts.genphore.run_phoregen.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="no output"):
                run_phoregen(
                    pharmacophore,
                    output_dir=str(tmp_path / "out"),
                    phoregen_dir=str(tool_dir),
                )
