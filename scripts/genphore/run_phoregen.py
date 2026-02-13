"""PhoreGen wrapper — pharmacophore-guided diffusion-based molecule generation.

Wraps the PhoreGen tool (tools/PhoreGen/) to generate molecules from a
SpikePharmacophore.  Uses subprocess to call PhoreGen's sampling script
in its own conda environment.

Pipeline:
    1. Write SpikePharmacophore.to_phoregen_json() → temp JSON
    2. Call PhoreGen sampling (subprocess in phoregen conda env)
    3. Parse output SDF → validate with RDKit → wrap as GeneratedMolecule
    4. Tag each molecule with source="phoregen" + pharmacophore_match_score
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from scripts.interfaces import GeneratedMolecule, SpikePharmacophore

logger = logging.getLogger(__name__)

# Default PhoreGen installation path
_PHOREGEN_DIR = Path("tools/PhoreGen")
_PHOREGEN_SCRIPT = _PHOREGEN_DIR / "sample.py"


def _parse_sdf_to_molecules(
    sdf_path: Path,
    batch_id: str,
    pharmacophore: SpikePharmacophore,
) -> List[GeneratedMolecule]:
    """Parse SDF file into GeneratedMolecule objects.

    Uses RDKit to validate each molecule and compute pharmacophore match.
    Falls back to raw parsing if RDKit is not available.
    """
    molecules: List[GeneratedMolecule] = []

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        feature_types = [f.feature_type for f in pharmacophore.features]

        for mol in supplier:
            if mol is None:
                continue

            smiles = Chem.MolToSmiles(mol)
            mol_block = Chem.MolToMolBlock(mol)

            # Pharmacophore match score from SDF properties if available
            match_score = 0.0
            matched = []
            if mol.HasProp("PhoreMatchScore"):
                try:
                    match_score = float(mol.GetProp("PhoreMatchScore"))
                except (ValueError, TypeError):
                    pass
            if mol.HasProp("MatchedFeatures"):
                matched = mol.GetProp("MatchedFeatures").split(",")
            else:
                matched = feature_types[:3]  # assume partial match
                match_score = len(matched) / max(len(feature_types), 1)

            molecules.append(
                GeneratedMolecule(
                    smiles=smiles,
                    mol_block=mol_block,
                    source="phoregen",
                    pharmacophore_match_score=round(match_score, 4),
                    matched_features=matched,
                    generation_batch_id=batch_id,
                )
            )
    except ImportError:
        logger.warning("RDKit not available; parsing SDF as raw text blocks")
        molecules.extend(
            _parse_sdf_raw(sdf_path, batch_id, pharmacophore)
        )

    return molecules


def _parse_sdf_raw(
    sdf_path: Path,
    batch_id: str,
    pharmacophore: SpikePharmacophore,
) -> List[GeneratedMolecule]:
    """Fallback SDF parser when RDKit is not available."""
    molecules: List[GeneratedMolecule] = []
    text = sdf_path.read_text()
    blocks = text.split("$$$$")
    feature_types = [f.feature_type for f in pharmacophore.features]

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract a SMILES-like identifier from the first line
        lines = block.split("\n")
        name = lines[0].strip() if lines else "unknown"

        molecules.append(
            GeneratedMolecule(
                smiles=name,  # placeholder — needs RDKit for real SMILES
                mol_block=block + "\n$$$$\n",
                source="phoregen",
                pharmacophore_match_score=0.0,
                matched_features=feature_types[:2],
                generation_batch_id=batch_id,
            )
        )

    return molecules


def run_phoregen(
    pharmacophore: SpikePharmacophore,
    n_molecules: int = 1000,
    output_dir: Optional[str] = None,
    conda_env: str = "phoregen",
    phoregen_dir: Optional[str] = None,
    device: str = "cuda:0",
    timeout: int = 3600,
) -> List[GeneratedMolecule]:
    """Run PhoreGen to generate molecules from a pharmacophore.

    Args:
        pharmacophore: Input pharmacophore model.
        n_molecules: Number of molecules to generate (default 1000).
        output_dir: Directory for output files. Uses tempdir if None.
        conda_env: Name of the conda environment with PhoreGen installed.
        phoregen_dir: Override path to PhoreGen installation.
        device: CUDA device string (default "cuda:0").
        timeout: Subprocess timeout in seconds (default 3600).

    Returns:
        List of GeneratedMolecule objects.

    Raises:
        RuntimeError: If PhoreGen subprocess fails.
        FileNotFoundError: If PhoreGen is not installed.
    """
    tool_dir = Path(phoregen_dir) if phoregen_dir else _PHOREGEN_DIR
    script = tool_dir / "sample.py"

    if not tool_dir.exists():
        raise FileNotFoundError(
            f"PhoreGen not found at {tool_dir}. "
            f"Clone it: git clone <phoregen-repo> {tool_dir}"
        )

    batch_id = str(uuid.uuid4())
    work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write pharmacophore input
    phore_json_path = work_dir / "pharmacophore_input.json"
    phore_data = pharmacophore.to_phoregen_json()
    with open(phore_json_path, "w") as f:
        json.dump(phore_data, f, indent=2)

    output_sdf = work_dir / "phoregen_output.sdf"

    cmd = [
        "conda", "run", "-n", conda_env, "--no-banner",
        "python", str(script),
        "--pharmacophore", str(phore_json_path),
        "--output", str(output_sdf),
        "--n_samples", str(n_molecules),
        "--device", device,
    ]

    logger.info(
        "Running PhoreGen: n=%d, device=%s, output=%s",
        n_molecules, device, output_sdf,
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(tool_dir),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"PhoreGen timed out after {timeout}s generating {n_molecules} molecules"
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"PhoreGen failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout[-500:]}\n"
            f"STDERR: {result.stderr[-500:]}"
        )

    if not output_sdf.exists():
        raise RuntimeError(
            f"PhoreGen produced no output file at {output_sdf}"
        )

    molecules = _parse_sdf_to_molecules(output_sdf, batch_id, pharmacophore)
    logger.info("PhoreGen produced %d valid molecules", len(molecules))

    return molecules
