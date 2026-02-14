#!/usr/bin/env python3
"""WT-8 Viewer Builder — convert a ViewerPayload JSON into a self-contained HTML5 file.

Usage
-----
    python scripts/viewer/build_viewer.py \\
        --payload tests/test_viewer/fixtures/mock_viewer_payload.json \\
        --output /tmp/test_report.html

The generated HTML embeds:
* Mol* from CDN (with offline fallback)
* All JS components inlined
* PDB structure + pipeline data as embedded JSON

Target: < 10 MB per file, works on iPad Safari.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Allow running as ``python scripts/viewer/build_viewer.py`` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.interfaces.viewer_payload import ViewerPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIEWER_DIR = Path(__file__).resolve().parent
_TEMPLATE_PATH = _VIEWER_DIR / "viewer_template.html"
_COMPONENTS_DIR = _VIEWER_DIR / "components"

# Order matters — dependencies first
_COMPONENT_FILES = [
    "pocket_surface.js",
    "spike_overlay.js",
    "water_map_layer.js",
    "ligand_viewer.js",
    "report_panel.js",
    "controls.js",
]

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB hard limit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    """Read a text file, raising a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def _inline_components() -> str:
    """Read all JS component files and wrap each in a <script> tag."""
    parts: list[str] = []
    for name in _COMPONENT_FILES:
        js_path = _COMPONENTS_DIR / name
        js_code = _read_text(js_path)
        parts.append(f"<!-- {name} -->\n<script>\n{js_code}\n</script>")
    return "\n".join(parts)


def _validate_payload(payload: ViewerPayload) -> list[str]:
    """Return a list of warnings (non-fatal) about the payload."""
    warnings: list[str] = []
    if not payload.pdb_structure or not payload.pdb_structure.strip():
        warnings.append("pdb_structure is empty — 3D viewer will show fallback")
    if not payload.pocket_surfaces:
        warnings.append("No pocket surfaces provided")
    if not payload.spike_positions:
        warnings.append("No spike positions provided")
    if not payload.ligand_poses:
        warnings.append("No ligand poses provided")
    return warnings


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_html(payload: ViewerPayload) -> str:
    """Build a self-contained HTML string from *payload*.

    Parameters
    ----------
    payload:
        A fully-populated :class:`ViewerPayload` instance.

    Returns
    -------
    str
        Complete HTML document as a string.

    Raises
    ------
    FileNotFoundError
        If template or component JS files are missing.
    ValueError
        If the generated HTML exceeds *MAX_FILE_SIZE_BYTES*.
    """
    # Validate
    for w in _validate_payload(payload):
        logger.warning("Payload warning: %s", w)

    # Load template
    template = _read_text(_TEMPLATE_PATH)

    # Inline JS components
    component_scripts = _inline_components()

    # Serialise payload
    payload_json = payload.to_json(indent=None)

    # Substitute placeholders
    html = template.replace("{{TARGET_NAME}}", payload.target_name or "PRISM-4D")
    html = html.replace("{{PAYLOAD_JSON}}", payload_json)
    html = html.replace("{{COMPONENT_SCRIPTS}}", component_scripts)

    # Size check
    size_bytes = len(html.encode("utf-8"))
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Generated HTML is {size_bytes / 1024 / 1024:.1f} MB, "
            f"exceeding {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB limit"
        )

    return html


def build_viewer(
    payload_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """End-to-end builder: read payload JSON, write HTML file.

    Parameters
    ----------
    payload_path:
        Path to a ViewerPayload JSON file.
    output_path:
        Destination path for the generated HTML file.

    Returns
    -------
    Path
        The resolved output path that was written.
    """
    payload_path = Path(payload_path)
    output_path = Path(output_path)

    logger.info("Loading payload from %s", payload_path)
    payload_json = _read_text(payload_path)
    payload = ViewerPayload.from_json(payload_json)

    logger.info("Building HTML for target: %s", payload.target_name)
    html = build_html(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    size_kb = len(html.encode("utf-8")) / 1024
    logger.info("Wrote %s (%.1f KB)", output_path, size_kb)
    return output_path.resolve()


def build_html_from_dict(data: Dict[str, Any]) -> str:
    """Build HTML directly from a payload dict (convenience for tests)."""
    payload = ViewerPayload.from_dict(data)
    return build_html(payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build PRISM-4D interactive HTML5 viewer"
    )
    parser.add_argument(
        "--payload",
        required=True,
        help="Path to ViewerPayload JSON file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output HTML file path",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output = build_viewer(args.payload, args.output)
    print(f"Viewer built: {output}")


if __name__ == "__main__":
    main()
