"""Tests for target_classifier module."""
from __future__ import annotations

import json
import tempfile

import pytest

from scripts.preprocessing.target_classifier import (
    ClassificationResult,
    classify_target,
    _detect_hydrophobicity_belt,
    _extract_pdb_id,
    _KD_HYDROPHOBICITY,
)


class TestClassificationResult:
    """Tests for the ClassificationResult dataclass."""

    def test_construction(self):
        r = ClassificationResult(
            pdb_path="/tmp/test.pdb",
            classification="membrane",
            confidence="definitive",
            method="opm_database",
        )
        assert r.classification == "membrane"
        assert r.confidence == "definitive"

    def test_to_dict(self):
        r = ClassificationResult(
            pdb_path="/tmp/test.pdb",
            classification="soluble",
            confidence="low",
            method="hydrophobicity_belt",
        )
        d = r.to_dict()
        assert d["classification"] == "soluble"
        assert d["opm_id"] is None

    def test_json_round_trip(self):
        r = ClassificationResult(
            pdb_path="/tmp/test.pdb",
            classification="membrane",
            confidence="high",
            method="uniprot_annotation",
            uniprot_accession="P07550",
            uniprot_location="membrane",
        )
        j = r.to_json()
        r2 = ClassificationResult.from_dict(json.loads(j))
        assert r2.classification == "membrane"
        assert r2.uniprot_accession == "P07550"

    def test_details_field(self):
        r = ClassificationResult(
            pdb_path="/tmp/test.pdb",
            classification="membrane",
            confidence="definitive",
            method="opm_database",
            details={"family": "GPCR", "tilt": 5.0},
        )
        assert r.details["family"] == "GPCR"


class TestManualOverride:
    """Test tier-0 manual override."""

    def test_membrane_override(self, kras_pdb):
        r = classify_target(kras_pdb, override="membrane")
        assert r.classification == "membrane"
        assert r.confidence == "manual"
        assert r.method == "manual_override"

    def test_soluble_override(self, beta2_pdb):
        r = classify_target(beta2_pdb, override="soluble")
        assert r.classification == "soluble"
        assert r.confidence == "manual"

    def test_invalid_override_ignored(self, kras_pdb):
        # Invalid override falls through to auto-detect
        r = classify_target(kras_pdb, override="invalid", skip_remote=True)
        assert r.method == "hydrophobicity_belt"


class TestHydrophobicityBelt:
    """Test tier-3 hydrophobicity belt detection."""

    def test_membrane_protein_detected(self, beta2_pdb):
        """Beta2-adrenergic fixture has a hydrophobic stretch."""
        detected = _detect_hydrophobicity_belt(beta2_pdb)
        assert detected is True

    def test_soluble_protein_not_detected(self, kras_pdb):
        """KRAS fixture has mixed hydrophilicity → no belt."""
        detected = _detect_hydrophobicity_belt(kras_pdb)
        assert detected is False

    def test_tiny_protein_no_belt(self, tmp_path):
        """Protein with fewer residues than window size → no belt."""
        pdb = tmp_path / "tiny.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "END\n"
        )
        assert _detect_hydrophobicity_belt(str(pdb)) is False

    def test_custom_threshold(self, beta2_pdb):
        """Very high threshold should return False."""
        assert _detect_hydrophobicity_belt(beta2_pdb, threshold=5.0) is False


class TestExtractPdbId:
    """Test PDB ID extraction from HEADER and filename."""

    def test_from_header(self, beta2_pdb):
        pdb_id = _extract_pdb_id(beta2_pdb)
        assert pdb_id == "2RH1"

    def test_from_filename(self, tmp_path):
        pdb = tmp_path / "4OBE_processed.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1   0.0  0.0  0.0\nEND\n")
        pdb_id = _extract_pdb_id(str(pdb))
        assert pdb_id == "4OBE"

    def test_no_id_found(self, tmp_path):
        pdb = tmp_path / "unknown.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1   0.0  0.0  0.0\nEND\n")
        assert _extract_pdb_id(str(pdb)) is None


class TestClassifyOffline:
    """Integration tests with skip_remote=True."""

    def test_membrane_protein_offline(self, beta2_pdb):
        r = classify_target(beta2_pdb, skip_remote=True)
        assert r.classification == "membrane"
        assert r.method == "hydrophobicity_belt"
        assert r.hydrophobicity_belt_detected is True

    def test_soluble_protein_offline(self, kras_pdb):
        r = classify_target(kras_pdb, skip_remote=True)
        assert r.classification == "soluble"
        assert r.method == "hydrophobicity_belt"
        assert r.hydrophobicity_belt_detected is False

    def test_pdb_path_is_absolute(self, kras_pdb):
        r = classify_target(kras_pdb, skip_remote=True)
        assert r.pdb_path.startswith("/")


class TestCLI:
    """Test CLI entrypoint."""

    def test_cli_offline(self, kras_pdb, capsys):
        from scripts.preprocessing.target_classifier import main
        main(["--pdb", kras_pdb, "--offline"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["classification"] in ("membrane", "soluble")

    def test_cli_membrane_override(self, kras_pdb, capsys):
        from scripts.preprocessing.target_classifier import main
        main(["--pdb", kras_pdb, "--membrane"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["classification"] == "membrane"
        assert parsed["confidence"] == "manual"
