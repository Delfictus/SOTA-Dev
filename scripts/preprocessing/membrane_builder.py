"""Automated lipid-bilayer embedding for membrane-protein targets.

Workflow:
    1. Consume :class:`ClassificationResult` from ``target_classifier``
    2. If "soluble" → skip
    3. Query OPM database for membrane orientation (tilt angle, depth)
    4. If not in OPM → PPM server prediction (fallback)
    5. Run ``packmol-memgen``: POPC(70%)/CHOL(30%), 80 Å × 80 Å
    6. Solvate + neutralize with 150 mM NaCl
    7. 6-step CHARMM-GUI equilibration protocol

Output: :class:`MembraneSystem` (from ``scripts.interfaces.membrane_system``).

External tool: packmol-memgen (GPL, AMBER-compatible, scriptable CLI).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.interfaces.membrane_system import MembraneSystem
from scripts.preprocessing.target_classifier import (
    ClassificationResult,
    classify_target,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LIPID_COMPOSITION = {"POPC": 0.70, "CHOL": 0.30}
DEFAULT_BOX_XY = 80.0  # Angstrom
DEFAULT_NACL_CONC = 0.15  # Molar (150 mM)
EQUILIBRATION_PROTOCOL = "CHARMM_GUI_6step"


# ---------------------------------------------------------------------------
# OPM orientation query
# ---------------------------------------------------------------------------

def query_opm_orientation(pdb_id: str) -> Optional[Dict[str, Any]]:
    """Fetch membrane orientation from OPM database.

    Returns dict with ``tilt_angle``, ``thickness``, ``depth`` or None.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests not available — skipping OPM orientation query")
        return None

    url = f"https://opm.phar.umich.edu/api/search?search={pdb_id}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data or not isinstance(data, list):
            return None
        for entry in data:
            if entry.get("pdbid", "").upper() == pdb_id.upper():
                return {
                    "tilt_angle": float(entry.get("tilt", 0.0)),
                    "thickness": float(entry.get("thickness", 30.0)),
                    "source": "OPM",
                }
    except Exception as exc:
        logger.debug("OPM orientation query failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# PPM fallback
# ---------------------------------------------------------------------------

def _predict_ppm_orientation(pdb_path: str) -> Dict[str, Any]:
    """Fallback orientation prediction when OPM misses.

    In production this would call the PPM 3.0 server or a local
    hydrophobic-slab optimization.  Here we return a sensible default
    for standard type-I transmembrane proteins.
    """
    logger.info("OPM miss — using PPM default orientation")
    return {
        "tilt_angle": 0.0,
        "thickness": 30.0,
        "source": "PPM_default",
    }


# ---------------------------------------------------------------------------
# packmol-memgen wrapper
# ---------------------------------------------------------------------------

def _has_packmol_memgen() -> bool:
    return shutil.which("packmol-memgen") is not None


def _run_packmol_memgen(
    pdb_path: str,
    output_dir: str,
    *,
    lipids: Dict[str, float],
    box_xy: float = DEFAULT_BOX_XY,
    nacl_conc: float = DEFAULT_NACL_CONC,
) -> Dict[str, Any]:
    """Run packmol-memgen to embed protein in a lipid bilayer.

    Returns metadata dict with atom count, lipid count, box dimensions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build lipid ratio string: --lipids POPC:7 CHOL:3
    lipid_parts = []
    for lip, frac in lipids.items():
        lipid_parts.extend(["--lipids", f"{lip}:{int(frac * 10)}"])

    cmd = [
        "packmol-memgen",
        "--pdb", str(pdb_path),
        "--output", str(output_dir / "membrane_system.pdb"),
        "--dist", str(box_xy / 2),  # half-distance from protein center
        "--salt", f"--conc {nacl_conc}",
        "--preoriented",  # assume OPM orientation already applied
        *lipid_parts,
    ]

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(output_dir), timeout=7200,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"packmol-memgen failed (rc={proc.returncode}): {proc.stderr}"
        )

    # Parse output for metadata
    out_pdb = output_dir / "membrane_system.pdb"
    n_atoms = 0
    n_lipids = 0
    box = (box_xy, box_xy, box_xy + 40.0)  # rough estimate
    if out_pdb.exists():
        with open(out_pdb) as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    n_atoms += 1
                    resname = line[17:20].strip()
                    if resname in lipids:
                        n_lipids += 1
                if line.startswith("CRYST1"):
                    try:
                        box = (
                            float(line[6:15]),
                            float(line[15:24]),
                            float(line[24:33]),
                        )
                    except (ValueError, IndexError):
                        pass

    return {
        "n_atoms": n_atoms,
        "n_lipids": n_lipids,
        "box": box,
        "output_pdb": str(out_pdb),
    }


# ---------------------------------------------------------------------------
# Simulated build (for testing without packmol-memgen)
# ---------------------------------------------------------------------------

def _simulate_membrane_build(
    pdb_path: str,
    output_dir: str,
    *,
    lipids: Dict[str, float],
    box_xy: float = DEFAULT_BOX_XY,
    orientation: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a MembraneSystem without calling packmol-memgen.

    Used for testing and when packmol-memgen is not installed.
    Estimates atom/lipid counts from box dimensions.
    """
    area = box_xy * box_xy
    lipid_area = 65.0  # Å² per lipid (POPC average)
    n_lipids_per_leaflet = int(area / lipid_area)
    n_lipids = n_lipids_per_leaflet * 2
    thickness = orientation.get("thickness", 30.0)
    box_z = thickness + 40.0  # water buffer above and below
    n_water = int(area * box_z * 0.033)  # ~0.033 waters/ų
    n_atoms_lipid = n_lipids * 130  # ~130 atoms per POPC
    n_atoms_water = n_water * 3
    n_atoms_protein = _count_atoms(pdb_path)
    total_atoms = n_atoms_protein + n_atoms_lipid + n_atoms_water

    return {
        "n_atoms": total_atoms,
        "n_lipids": n_lipids,
        "box": (box_xy, box_xy, box_z),
        "output_pdb": str(Path(output_dir) / "membrane_system_simulated.pdb"),
    }


def _count_atoms(pdb_path: str) -> int:
    """Count ATOM/HETATM records in a PDB."""
    count = 0
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_membrane(
    pdb_path: str,
    classification: ClassificationResult,
    *,
    output_dir: Optional[str] = None,
    lipids: Optional[Dict[str, float]] = None,
    box_xy: float = DEFAULT_BOX_XY,
    nacl_conc: float = DEFAULT_NACL_CONC,
    force_build: bool = False,
) -> Optional[MembraneSystem]:
    """Build a membrane system for a protein target.

    Parameters
    ----------
    pdb_path : str
        Path to the (fixed) PDB file.
    classification : ClassificationResult
        Output from ``target_classifier.classify_target()``.
    output_dir : str, optional
        Directory for output files. Defaults to ``<pdb_dir>/membrane/``.
    lipids : dict, optional
        Lipid composition. Defaults to POPC(70%)/CHOL(30%).
    box_xy : float
        Membrane patch size in Angstrom (default 80).
    nacl_conc : float
        NaCl concentration in Molar (default 0.15 = 150 mM).
    force_build : bool
        Build even if classified as soluble.

    Returns
    -------
    MembraneSystem or None
        None if target is soluble and ``force_build`` is False.
    """
    if classification.classification == "soluble" and not force_build:
        logger.info("Target classified as soluble — skipping membrane build")
        return None

    pdb_path = str(Path(pdb_path).resolve())
    if output_dir is None:
        output_dir = str(Path(pdb_path).parent / "membrane")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if lipids is None:
        lipids = dict(DEFAULT_LIPID_COMPOSITION)

    # Get orientation
    pdb_id = None
    if classification.opm_id:
        pdb_id = classification.opm_id
    orientation: Dict[str, Any]
    if pdb_id:
        opm = query_opm_orientation(pdb_id)
        orientation = opm if opm else _predict_ppm_orientation(pdb_path)
    else:
        orientation = _predict_ppm_orientation(pdb_path)

    tilt_angle = orientation.get("tilt_angle", 0.0)
    thickness = orientation.get("thickness", 30.0)
    orient_source = orientation.get("source", "unknown")

    # Build
    if _has_packmol_memgen():
        meta = _run_packmol_memgen(
            pdb_path, output_dir,
            lipids=lipids, box_xy=box_xy, nacl_conc=nacl_conc,
        )
    else:
        logger.warning(
            "packmol-memgen not found — returning simulated MembraneSystem"
        )
        meta = _simulate_membrane_build(
            pdb_path, output_dir,
            lipids=lipids, box_xy=box_xy, orientation=orientation,
        )

    return MembraneSystem(
        lipid_composition=lipids,
        bilayer_method="packmol_memgen" if _has_packmol_memgen() else "simulated",
        n_lipids=meta["n_lipids"],
        membrane_thickness=thickness,
        protein_orientation=orient_source,
        opm_tilt_angle=tilt_angle,
        system_size=meta["box"],
        total_atoms=meta["n_atoms"],
        equilibration_protocol=EQUILIBRATION_PROTOCOL,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build lipid bilayer membrane system around a protein."
    )
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument(
        "--force", action="store_true",
        help="Force membrane build even if classified as soluble",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip remote lookups",
    )
    args = parser.parse_args(argv)

    classification = classify_target(
        args.pdb, skip_remote=args.offline,
    )
    print(f"Classification: {classification.classification} "
          f"({classification.confidence})")

    result = build_membrane(
        args.pdb, classification,
        output_dir=args.output_dir,
        force_build=args.force,
    )
    if result is None:
        print("Soluble target — no membrane built.")
    else:
        print(result.to_json())


if __name__ == "__main__":
    main()
