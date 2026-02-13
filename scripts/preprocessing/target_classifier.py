"""Target classifier — auto-detect membrane vs soluble protein.

Hierarchy (first definitive match wins):
    1. Manual override (``--membrane`` / ``--soluble`` flag)
    2. OPM database lookup (https://opm.phar.umich.edu)
    3. UniProt subcellular-location annotation
    4. Hydrophobicity-belt heuristic on the PDB structure

Returns a :class:`ClassificationResult` consumed by ``membrane_builder.py``
and the orchestrator.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Output of the target classifier."""

    pdb_path: str
    classification: str  # "membrane" | "soluble"
    confidence: str      # "definitive" | "high" | "medium" | "low" | "manual"
    method: str          # which tier produced the answer
    opm_id: Optional[str] = None
    opm_tilt_angle: Optional[float] = None
    opm_thickness: Optional[float] = None
    uniprot_accession: Optional[str] = None
    uniprot_location: Optional[str] = None
    hydrophobicity_belt_detected: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ClassificationResult:
        return cls(**d)


# ---------------------------------------------------------------------------
# Tier 1: OPM database lookup
# ---------------------------------------------------------------------------

def _query_opm(pdb_id: str) -> Optional[Dict[str, Any]]:
    """Query the OPM database for a PDB entry.

    Returns dict with tilt_angle, thickness, and classification if found.
    Returns None on miss or network error.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed — skipping OPM lookup")
        return None

    url = f"https://opm.phar.umich.edu/api/search?search={pdb_id}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or not isinstance(data, list):
            return None
        for entry in data:
            if entry.get("pdbid", "").upper() == pdb_id.upper():
                return {
                    "opm_id": entry.get("pdbid"),
                    "tilt_angle": float(entry.get("tilt", 0.0)),
                    "thickness": float(entry.get("thickness", 0.0)),
                    "family": entry.get("family", ""),
                    "type_name": entry.get("type_name", ""),
                }
    except Exception as exc:
        logger.debug("OPM query failed for %s: %s", pdb_id, exc)
    return None


# ---------------------------------------------------------------------------
# Tier 2: UniProt subcellular location
# ---------------------------------------------------------------------------

_MEMBRANE_KEYWORDS = frozenset({
    "membrane", "transmembrane", "integral membrane",
    "multi-pass membrane", "single-pass membrane",
    "lipid-anchor", "gpi-anchor",
})


def _query_uniprot(pdb_id: str) -> Optional[Dict[str, Any]]:
    """Map PDB → UniProt and check subcellular location for membrane terms."""
    try:
        import requests
    except ImportError:
        return None

    # PDB → UniProt mapping via SIFTS
    mapping_url = (
        f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    )
    try:
        resp = requests.get(mapping_url, timeout=10)
        if resp.status_code != 200:
            return None
        mapping = resp.json()
        entry = mapping.get(pdb_id.lower(), {}).get("UniProt", {})
        if not entry:
            return None
        accession = next(iter(entry))
    except Exception as exc:
        logger.debug("UniProt mapping failed for %s: %s", pdb_id, exc)
        return None

    # Fetch UniProt entry for subcellular location
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        resp = requests.get(uniprot_url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        comments = data.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                locations = comment.get("subcellularLocations", [])
                for loc in locations:
                    loc_val = loc.get("location", {}).get("value", "").lower()
                    for kw in _MEMBRANE_KEYWORDS:
                        if kw in loc_val:
                            return {
                                "accession": accession,
                                "location": loc_val,
                                "is_membrane": True,
                            }
                # Found SUBCELLULAR LOCATION but no membrane keywords
                first_loc = ""
                if locations:
                    first_loc = (
                        locations[0].get("location", {}).get("value", "")
                    )
                return {
                    "accession": accession,
                    "location": first_loc,
                    "is_membrane": False,
                }
    except Exception as exc:
        logger.debug("UniProt fetch failed for %s: %s", accession, exc)
    return None


# ---------------------------------------------------------------------------
# Tier 3: Hydrophobicity-belt heuristic
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity
_KD_HYDROPHOBICITY = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}


def _detect_hydrophobicity_belt(pdb_path: str,
                                window: int = 20,
                                threshold: float = 1.2) -> bool:
    """Scan CA atoms along z-axis for a hydrophobic belt.

    A belt is detected when a sliding window of ``window`` consecutive
    residues (sorted by z-coordinate) has mean Kyte-Doolittle
    hydrophobicity >= ``threshold``.
    """
    residues: List[tuple] = []  # (z_coord, resname)
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip() == "CA":
                resname = line[17:20].strip()
                try:
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                residues.append((z, resname))

    if len(residues) < window:
        return False

    residues.sort(key=lambda r: r[0])

    for i in range(len(residues) - window + 1):
        window_res = residues[i : i + window]
        mean_hyd = sum(
            _KD_HYDROPHOBICITY.get(r[1], 0.0) for r in window_res
        ) / window
        if mean_hyd >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Extract PDB ID from file
# ---------------------------------------------------------------------------

def _extract_pdb_id(pdb_path: str) -> Optional[str]:
    """Extract 4-letter PDB ID from HEADER record or filename."""
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("HEADER"):
                # PDB ID is at columns 63-66 in HEADER record
                pdb_id = line[62:66].strip()
                if len(pdb_id) == 4:
                    return pdb_id.upper()
                break

    # Fallback: extract from filename
    stem = Path(pdb_path).stem.upper()
    match = re.search(r"([0-9][A-Z0-9]{3})", stem)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_target(
    pdb_path: str,
    *,
    override: Optional[str] = None,
    skip_remote: bool = False,
) -> ClassificationResult:
    """Classify a protein target as membrane or soluble.

    Parameters
    ----------
    pdb_path : str
        Path to the input PDB file.
    override : str, optional
        ``"membrane"`` or ``"soluble"`` to bypass auto-detection.
    skip_remote : bool
        If True, skip OPM and UniProt lookups (offline mode / testing).
    """
    pdb_path = str(Path(pdb_path).resolve())

    # Tier 0: Manual override
    if override in ("membrane", "soluble"):
        return ClassificationResult(
            pdb_path=pdb_path,
            classification=override,
            confidence="manual",
            method="manual_override",
        )

    pdb_id = _extract_pdb_id(pdb_path)

    # Tier 1: OPM
    if pdb_id and not skip_remote:
        opm = _query_opm(pdb_id)
        if opm:
            return ClassificationResult(
                pdb_path=pdb_path,
                classification="membrane",
                confidence="definitive",
                method="opm_database",
                opm_id=opm["opm_id"],
                opm_tilt_angle=opm["tilt_angle"],
                opm_thickness=opm["thickness"],
                details=opm,
            )

    # Tier 2: UniProt
    if pdb_id and not skip_remote:
        up = _query_uniprot(pdb_id)
        if up:
            cls_val = "membrane" if up["is_membrane"] else "soluble"
            return ClassificationResult(
                pdb_path=pdb_path,
                classification=cls_val,
                confidence="high",
                method="uniprot_annotation",
                uniprot_accession=up["accession"],
                uniprot_location=up["location"],
                details=up,
            )

    # Tier 3: Hydrophobicity belt
    belt = _detect_hydrophobicity_belt(pdb_path)
    return ClassificationResult(
        pdb_path=pdb_path,
        classification="membrane" if belt else "soluble",
        confidence="medium" if belt else "low",
        method="hydrophobicity_belt",
        hydrophobicity_belt_detected=belt,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify protein target as membrane or soluble."
    )
    parser.add_argument("--pdb", required=True, help="Path to PDB file")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--membrane", action="store_true",
        help="Force membrane classification",
    )
    grp.add_argument(
        "--soluble", action="store_true",
        help="Force soluble classification",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip OPM/UniProt remote lookups",
    )
    args = parser.parse_args(argv)

    override = None
    if args.membrane:
        override = "membrane"
    elif args.soluble:
        override = "soluble"

    result = classify_target(
        args.pdb, override=override, skip_remote=args.offline,
    )
    print(result.to_json())


if __name__ == "__main__":
    main()
