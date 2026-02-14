"""Tests for protein_fixer module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.preprocessing.protein_fixer import (
    FixerResult,
    fix_protein,
    _check_crystal_contacts,
    _NONSTD_MAP,
    _STANDARD_RESIDUES,
)


class TestFixerResult:
    """Tests for FixerResult dataclass."""

    def test_construction(self):
        r = FixerResult(
            input_path="/tmp/in.pdb",
            output_path="/tmp/out.pdb",
            method="fallback",
        )
        assert r.missing_residues_added == 0
        assert r.nonstandard_replaced == []
        assert r.method == "fallback"

    def test_to_dict(self):
        r = FixerResult(
            input_path="/tmp/in.pdb",
            output_path="/tmp/out.pdb",
            nonstandard_replaced=["MSE→MET"],
            warnings=["Crystal contacts detected"],
        )
        d = r.to_dict()
        assert d["nonstandard_replaced"] == ["MSE→MET"]
        assert "Crystal contacts" in d["warnings"][0]

    def test_to_json(self):
        r = FixerResult(
            input_path="/tmp/in.pdb", output_path="/tmp/out.pdb",
        )
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["method"] == "pdbfixer"


class TestNonstandardResidueFix:
    """Test non-standard residue replacement in fallback mode."""

    def test_mse_to_met(self, tmp_path):
        pdb = tmp_path / "mse.pdb"
        pdb.write_text(
            "CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1\n"
            "ATOM      1  CA  MSE A   1       0.0   0.0   0.0  1.00  0.0\n"
            "END\n"
        )
        out = tmp_path / "fixed.pdb"
        result = fix_protein(str(pdb), str(out))
        assert "MSE→MET" in result.nonstandard_replaced
        content = out.read_text()
        assert "MET" in content
        assert "MSE" not in content

    def test_multiple_nonstandard(self, tmp_path):
        pdb = tmp_path / "multi.pdb"
        pdb.write_text(
            "CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1\n"
            "ATOM      1  CA  MSE A   1       0.0   0.0   0.0  1.00  0.0\n"
            "ATOM      2  CA  SEP A   2       3.0   0.0   0.0  1.00  0.0\n"
            "ATOM      3  CA  ALA A   3       6.0   0.0   0.0  1.00  0.0\n"
            "END\n"
        )
        out = tmp_path / "fixed.pdb"
        result = fix_protein(str(pdb), str(out))
        assert "MSE→MET" in result.nonstandard_replaced
        assert "SEP→SER" in result.nonstandard_replaced
        assert len(result.nonstandard_replaced) == 2


class TestAltConformation:
    """Test alternate conformation handling."""

    def test_keeps_altloc_a(self, tmp_path):
        pdb = tmp_path / "altloc.pdb"
        pdb.write_text(
            "CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1\n"
            "ATOM      1  CA AALA A   1       0.0   0.0   0.0  1.00  0.0\n"
            "ATOM      2  CA BALA A   1       0.1   0.1   0.1  1.00  0.0\n"
            "END\n"
        )
        out = tmp_path / "fixed.pdb"
        result = fix_protein(str(pdb), str(out))
        assert result.altlocs_resolved > 0
        content = out.read_text()
        lines = [l for l in content.split("\n") if l.startswith("ATOM")]
        assert len(lines) == 1  # only 'A' kept (B dropped)

    def test_keeps_blank_altloc(self, tmp_path):
        pdb = tmp_path / "no_altloc.pdb"
        pdb.write_text(
            "CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1\n"
            "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.0\n"
            "END\n"
        )
        out = tmp_path / "fixed.pdb"
        result = fix_protein(str(pdb), str(out))
        content = out.read_text()
        assert "ALA" in content


class TestHeterogenRemoval:
    """Test HETATM removal options."""

    def test_remove_heterogens(self, tmp_path):
        pdb = tmp_path / "het.pdb"
        pdb.write_text(
            "CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1\n"
            "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.0\n"
            "HETATM    2  C1  LIG A 200       5.0   5.0   5.0  1.00  0.0\n"
            "HETATM    3  O   HOH A 300      10.0  10.0  10.0  1.00  0.0\n"
            "END\n"
        )
        out = tmp_path / "fixed.pdb"
        result = fix_protein(str(pdb), str(out), remove_heterogens=True)
        assert result.heterogens_removed >= 2
        content = out.read_text()
        assert "HETATM" not in content

    def test_keep_water(self, tmp_path):
        pdb = tmp_path / "water.pdb"
        pdb.write_text(
            "CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1\n"
            "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.0\n"
            "HETATM    2  C1  LIG A 200       5.0   5.0   5.0  1.00  0.0\n"
            "HETATM    3  O   HOH A 300      10.0  10.0  10.0  1.00  0.0\n"
            "END\n"
        )
        out = tmp_path / "fixed.pdb"
        result = fix_protein(
            str(pdb), str(out), remove_heterogens=True, keep_water=True,
        )
        assert result.heterogens_removed == 1  # only LIG removed
        content = out.read_text()
        assert "HOH" in content
        assert "LIG" not in content


class TestCrystalContacts:
    """Test crystal contact detection."""

    def test_small_cell_warns(self, tmp_path):
        pdb = tmp_path / "small_cell.pdb"
        pdb.write_text(
            "CRYST1   25.000   25.000   25.000  90.00  90.00  90.00 P 1\n"
            "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.0\n"
            "END\n"
        )
        assert _check_crystal_contacts(str(pdb)) is True

    def test_large_cell_ok(self, tmp_path):
        pdb = tmp_path / "large_cell.pdb"
        pdb.write_text(
            "CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1\n"
            "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.0\n"
            "END\n"
        )
        assert _check_crystal_contacts(str(pdb)) is False

    def test_crystal_warning_in_result(self, tmp_path):
        pdb = tmp_path / "contact.pdb"
        pdb.write_text(
            "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1\n"
            "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.0\n"
            "END\n"
        )
        out = tmp_path / "fixed.pdb"
        result = fix_protein(str(pdb), str(out))
        assert result.crystal_contacts_warned is True
        assert any("Crystal contacts" in w for w in result.warnings)


class TestDefaultOutputPath:
    """Test auto-generated output path."""

    def test_default_path(self, tmp_path):
        pdb = tmp_path / "input.pdb"
        pdb.write_text(
            "CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1\n"
            "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.0\n"
            "END\n"
        )
        result = fix_protein(str(pdb))
        assert result.output_path.endswith("input_fixed.pdb")
        assert Path(result.output_path).exists()


class TestFallbackWarning:
    """Test that fallback mode warns about limitations."""

    def test_fallback_warning(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1\n"
            "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.0\n"
            "END\n"
        )
        out = tmp_path / "fixed.pdb"
        result = fix_protein(str(pdb), str(out))
        if result.method == "fallback":
            assert any("PDBFixer not available" in w for w in result.warnings)


class TestCLI:
    """Test CLI entrypoint."""

    def test_cli_basic(self, kras_pdb, tmp_path, capsys):
        from scripts.preprocessing.protein_fixer import main
        out = tmp_path / "fixed.pdb"
        main(["--pdb", kras_pdb, "--output", str(out)])
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert "input_path" in parsed
        assert "output_path" in parsed

    def test_cli_remove_heterogens(self, kras_pdb, tmp_path, capsys):
        from scripts.preprocessing.protein_fixer import main
        out = tmp_path / "fixed.pdb"
        main(["--pdb", kras_pdb, "--output", str(out), "--remove-heterogens"])
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed["method"] in ("pdbfixer", "fallback")
