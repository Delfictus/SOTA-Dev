"""Tests for WT-8 interactive HTML5/Mol* viewer builder.

Covers:
- Payload loading from JSON file and dict
- HTML generation with all sections present
- Component JS inlining
- File size < 10 MB limit
- Placeholder substitution
- Minimal / empty payload edge cases
- CLI smoke test
- Round-trip: payload JSON → HTML → extractable payload
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.interfaces.viewer_payload import ViewerPayload
from scripts.viewer.build_viewer import (
    MAX_FILE_SIZE_BYTES,
    build_html,
    build_html_from_dict,
    build_viewer,
    _inline_components,
    _validate_payload,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Payload loading
# ═══════════════════════════════════════════════════════════════════════════


class TestPayloadLoading:
    """Verify ViewerPayload deserialises correctly from fixture."""

    def test_load_from_json_file(self, mock_payload_path: Path) -> None:
        text = mock_payload_path.read_text(encoding="utf-8")
        payload = ViewerPayload.from_json(text)
        assert payload.target_name == "KRAS_G12C"
        assert isinstance(payload.pdb_structure, str)
        assert len(payload.pdb_structure) > 0

    def test_load_from_dict(self, mock_payload_dict: dict) -> None:
        payload = ViewerPayload.from_dict(mock_payload_dict)
        assert payload.target_name == "KRAS_G12C"
        assert len(payload.pocket_surfaces) == 1
        assert len(payload.spike_positions) == 3
        assert len(payload.water_map_sites) == 3
        assert len(payload.ligand_poses) == 3
        assert payload.lining_residues == [140, 141, 142, 145, 148]
        assert payload.p_open == 0.72

    def test_metadata_preserved(self, mock_payload: ViewerPayload) -> None:
        assert mock_payload.metadata["qa_score"] == 0.92
        assert mock_payload.metadata["druggability_class"] == "STABLE_OPEN"
        assert mock_payload.metadata["n_displaceable_waters"] == 1

    def test_pocket_surface_structure(self, mock_payload: ViewerPayload) -> None:
        surf = mock_payload.pocket_surfaces[0]
        assert "vertices" in surf
        assert "triangles" in surf
        assert "color" in surf
        assert len(surf["vertices"]) == 9
        assert len(surf["triangles"]) == 6

    def test_spike_structure(self, mock_payload: ViewerPayload) -> None:
        spike = mock_payload.spike_positions[0]
        assert spike["type"] == "BNZ"
        assert spike["intensity"] == 0.85
        assert spike["residue"] == "TYR142"
        assert len(spike["position"]) == 3

    def test_water_site_structure(self, mock_payload: ViewerPayload) -> None:
        happy = mock_payload.water_map_sites[0]
        assert happy["classification"] == "CONSERVED_HAPPY"
        assert happy["delta_g"] == -2.1
        unhappy = mock_payload.water_map_sites[1]
        assert unhappy["classification"] == "CONSERVED_UNHAPPY"
        assert unhappy["delta_g"] == 1.8

    def test_ligand_structure(self, mock_payload: ViewerPayload) -> None:
        lig = mock_payload.ligand_poses[0]
        assert lig["smiles"] == "c1ccc(CC(=O)O)cc1"
        assert lig["dg_kcal"] == -8.2
        assert lig["classification"] == "NOVEL_HIT"


# ═══════════════════════════════════════════════════════════════════════════
#  Component inlining
# ═══════════════════════════════════════════════════════════════════════════


class TestComponentInlining:
    """Verify JS components are read and inlined."""

    def test_inline_components_returns_string(self) -> None:
        result = _inline_components()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_all_components_present(self) -> None:
        result = _inline_components()
        for name in [
            "pocket_surface.js",
            "spike_overlay.js",
            "water_map_layer.js",
            "ligand_viewer.js",
            "report_panel.js",
            "controls.js",
        ]:
            assert name in result, f"Component {name} not found in inlined output"

    def test_components_wrapped_in_script_tags(self) -> None:
        result = _inline_components()
        assert result.count("<script>") == 6
        assert result.count("</script>") == 6

    def test_component_objects_defined(self) -> None:
        result = _inline_components()
        for obj in [
            "PocketSurface",
            "SpikeOverlay",
            "WaterMapLayer",
            "LigandViewer",
            "ReportPanel",
            "Controls",
        ]:
            assert obj in result, f"JS object '{obj}' not found"


# ═══════════════════════════════════════════════════════════════════════════
#  Payload validation
# ═══════════════════════════════════════════════════════════════════════════


class TestPayloadValidation:

    def test_valid_payload_no_warnings(self, mock_payload: ViewerPayload) -> None:
        warnings = _validate_payload(mock_payload)
        assert len(warnings) == 0

    def test_empty_pdb_warns(self, empty_payload: ViewerPayload) -> None:
        warnings = _validate_payload(empty_payload)
        assert any("pdb_structure" in w for w in warnings)

    def test_no_surfaces_warns(self, minimal_payload: ViewerPayload) -> None:
        warnings = _validate_payload(minimal_payload)
        assert any("pocket surfaces" in w.lower() for w in warnings)

    def test_no_spikes_warns(self, minimal_payload: ViewerPayload) -> None:
        warnings = _validate_payload(minimal_payload)
        assert any("spike" in w.lower() for w in warnings)

    def test_no_ligands_warns(self, minimal_payload: ViewerPayload) -> None:
        warnings = _validate_payload(minimal_payload)
        assert any("ligand" in w.lower() for w in warnings)


# ═══════════════════════════════════════════════════════════════════════════
#  HTML generation
# ═══════════════════════════════════════════════════════════════════════════


class TestHTMLGeneration:
    """Core HTML builder tests."""

    def test_build_html_returns_string(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert isinstance(html, str)

    def test_html_is_valid_document(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html

    def test_target_name_in_title(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert "KRAS_G12C" in html
        assert "<title>PRISM-4D Viewer" in html

    def test_payload_json_embedded(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert 'id="payload-data"' in html
        assert "application/json" in html
        # Payload should be parseable
        m = re.search(
            r'<script id="payload-data" type="application/json">\s*(.+?)\s*</script>',
            html,
            re.DOTALL,
        )
        assert m is not None
        embedded = json.loads(m.group(1))
        assert embedded["target_name"] == "KRAS_G12C"
        assert len(embedded["spike_positions"]) == 3

    def test_components_inlined(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert "PocketSurface" in html
        assert "SpikeOverlay" in html
        assert "WaterMapLayer" in html
        assert "LigandViewer" in html
        assert "ReportPanel" in html
        assert "Controls" in html

    def test_molstar_cdn_referenced(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert "molstar" in html.lower()
        assert "cdn.jsdelivr.net" in html

    def test_no_unresolved_placeholders(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert "{{TARGET_NAME}}" not in html
        assert "{{PAYLOAD_JSON}}" not in html
        assert "{{COMPONENT_SCRIPTS}}" not in html

    def test_viewer_container_present(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert 'id="viewer-3d"' in html
        assert 'id="viewer-container"' in html

    def test_controls_slot_present(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert 'id="controls-slot"' in html

    def test_report_slot_present(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert 'id="report-slot"' in html

    def test_responsive_meta_viewport(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert 'name="viewport"' in html

    def test_loading_overlay_present(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert 'id="loading-overlay"' in html


# ═══════════════════════════════════════════════════════════════════════════
#  File size constraint
# ═══════════════════════════════════════════════════════════════════════════


class TestFileSize:

    def test_under_10mb(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        size = len(html.encode("utf-8"))
        assert size < MAX_FILE_SIZE_BYTES, f"HTML is {size / 1024:.1f} KB"

    def test_mock_payload_reasonable_size(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        size_kb = len(html.encode("utf-8")) / 1024
        # Mock payload should be well under 1 MB
        assert size_kb < 1024, f"Mock HTML is {size_kb:.1f} KB — unexpectedly large"

    def test_oversized_payload_raises(self) -> None:
        """A payload with enormous PDB text should trigger size guard."""
        giant_pdb = "ATOM" * (3 * 1024 * 1024)  # ~12 MB of text
        payload = ViewerPayload(
            target_name="GIANT",
            pdb_structure=giant_pdb,
            pocket_surfaces=[],
            spike_positions=[],
            water_map_sites=[],
            ligand_poses=[],
            lining_residues=[],
        )
        with pytest.raises(ValueError, match="MB limit"):
            build_html(payload)


# ═══════════════════════════════════════════════════════════════════════════
#  build_html_from_dict convenience
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildFromDict:

    def test_from_dict(self, mock_payload_dict: dict) -> None:
        html = build_html_from_dict(mock_payload_dict)
        assert "KRAS_G12C" in html
        assert "<!DOCTYPE html>" in html


# ═══════════════════════════════════════════════════════════════════════════
#  build_viewer (file I/O)
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildViewer:

    def test_writes_html_file(self, mock_payload_path: Path, tmp_path: Path) -> None:
        out = tmp_path / "test_report.html"
        result = build_viewer(mock_payload_path, out)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "KRAS_G12C" in content

    def test_creates_parent_dirs(self, mock_payload_path: Path, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "dir" / "report.html"
        result = build_viewer(mock_payload_path, out)
        assert result.exists()

    def test_output_under_10mb(self, mock_payload_path: Path, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        result = build_viewer(mock_payload_path, out)
        assert result.stat().st_size < MAX_FILE_SIZE_BYTES

    def test_nonexistent_payload_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_viewer("/nonexistent/payload.json", tmp_path / "out.html")


# ═══════════════════════════════════════════════════════════════════════════
#  Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_minimal_payload_builds(self, minimal_payload: ViewerPayload) -> None:
        html = build_html(minimal_payload)
        assert "MINIMAL_TEST" in html
        assert "<!DOCTYPE html>" in html

    def test_empty_payload_builds(self, empty_payload: ViewerPayload) -> None:
        html = build_html(empty_payload)
        assert "<!DOCTYPE html>" in html

    def test_null_p_open(self) -> None:
        payload = ViewerPayload(
            target_name="NULL_POPEN",
            pdb_structure="END\n",
            pocket_surfaces=[],
            spike_positions=[],
            water_map_sites=[],
            ligand_poses=[],
            lining_residues=[],
            p_open=None,
        )
        html = build_html(payload)
        assert "NULL_POPEN" in html

    def test_special_chars_in_target_name(self) -> None:
        payload = ViewerPayload(
            target_name="BRAF_V600E <test> & \"quoted\"",
            pdb_structure="END\n",
            pocket_surfaces=[],
            spike_positions=[],
            water_map_sites=[],
            ligand_poses=[],
            lining_residues=[],
        )
        html = build_html(payload)
        assert "<!DOCTYPE html>" in html

    def test_unicode_in_metadata(self) -> None:
        payload = ViewerPayload(
            target_name="UNICODE_TEST",
            pdb_structure="END\n",
            pocket_surfaces=[],
            spike_positions=[],
            water_map_sites=[],
            ligand_poses=[],
            lining_residues=[],
            metadata={"note": "Delta \u0394G = -8.2 kcal/mol"},
        )
        html = build_html(payload)
        assert "UNICODE_TEST" in html


# ═══════════════════════════════════════════════════════════════════════════
#  Round-trip: embedded JSON extractable from HTML
# ═══════════════════════════════════════════════════════════════════════════


class TestRoundTrip:

    def test_embedded_payload_round_trip(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        m = re.search(
            r'<script id="payload-data" type="application/json">\s*(.+?)\s*</script>',
            html,
            re.DOTALL,
        )
        assert m is not None
        extracted = json.loads(m.group(1))
        restored = ViewerPayload.from_dict(extracted)

        assert restored.target_name == mock_payload.target_name
        assert restored.p_open == mock_payload.p_open
        assert restored.lining_residues == mock_payload.lining_residues
        assert len(restored.spike_positions) == len(mock_payload.spike_positions)
        assert len(restored.ligand_poses) == len(mock_payload.ligand_poses)
        assert len(restored.water_map_sites) == len(mock_payload.water_map_sites)
        assert restored.metadata == mock_payload.metadata


# ═══════════════════════════════════════════════════════════════════════════
#  CSS / styling checks
# ═══════════════════════════════════════════════════════════════════════════


class TestStyling:

    def test_responsive_breakpoint(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert "@media" in html
        assert "900px" in html

    def test_dark_theme_colors(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert "#0d1117" in html  # GitHub dark background
        assert "#c9d1d9" in html  # text color

    def test_print_styles(self, mock_payload: ViewerPayload) -> None:
        html = build_html(mock_payload)
        assert "@media print" in html


# ═══════════════════════════════════════════════════════════════════════════
#  CLI smoke test
# ═══════════════════════════════════════════════════════════════════════════


class TestCLI:

    def test_cli_builds_file(self, mock_payload_path: Path, tmp_path: Path) -> None:
        out = tmp_path / "cli_report.html"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.viewer.build_viewer",
                "--payload",
                str(mock_payload_path),
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "KRAS_G12C" in content

    def test_cli_missing_payload_fails(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.viewer.build_viewer",
                "--payload",
                "/nonexistent.json",
                "--output",
                str(tmp_path / "out.html"),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert result.returncode != 0
