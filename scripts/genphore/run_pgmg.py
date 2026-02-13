"""PGMG wrapper — pharmacophore-guided VAE-based molecule generation.

Wraps the PGMG tool (tools/PGMG/) to generate molecules from a
SpikePharmacophore.  PGMG uses a Variational Autoencoder to sample
SMILES conditioned on pharmacophore constraints.

Pipeline:
    1. Write SpikePharmacophore.to_pgmg_posp() → temp .posp file
    2. Call PGMG generate.py (subprocess in pgmg conda env)
    3. Parse output SMILES → 3D conformer via RDKit ETKDG
    4. Wrap as GeneratedMolecule with source="pgmg"

PGMG limitation: max 8 pharmacophore features.  Features are truncated
by intensity (highest first) if the pharmacophore has more than 8.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from scripts.interfaces import GeneratedMolecule, SpikePharmacophore

logger = logging.getLogger(__name__)

_PGMG_DIR = Path("tools/PGMG")
_PGMG_SCRIPT = _PGMG_DIR / "generate.py"
_MAX_PGMG_FEATURES = 8


def _truncate_pharmacophore(pharmacophore: SpikePharmacophore) -> str:
    """Generate .posp string, truncating to top 8 features by intensity.

    PGMG cannot handle more than 8 features.  We keep the highest-intensity
    features since they represent the strongest pharmacophoric constraints.
    """
    if len(pharmacophore.features) <= _MAX_PGMG_FEATURES:
        return pharmacophore.to_pgmg_posp()

    # Create a temporary copy with truncated features
    truncated_features = sorted(
        pharmacophore.features, key=lambda f: f.intensity, reverse=True
    )[:_MAX_PGMG_FEATURES]

    logger.info(
        "Truncating %d features to %d for PGMG (dropped types: %s)",
        len(pharmacophore.features),
        _MAX_PGMG_FEATURES,
        [f.feature_type for f in pharmacophore.features[_MAX_PGMG_FEATURES:]],
    )

    lines: List[str] = []
    lines.append(
        f"# PGMG pharmacophore for {pharmacophore.target_name} "
        f"pocket {pharmacophore.pocket_id}"
    )
    lines.append(
        f"# PDB: {pharmacophore.pdb_id}  "
        f"Hash: {pharmacophore.prism_run_hash}"
    )
    lines.append(f"# Truncated from {len(pharmacophore.features)} features")

    for feat in truncated_features:
        weight = round(feat.intensity, 4)
        lines.append(
            f"FEATURE {feat.feature_type} "
            f"{feat.x:.3f} {feat.y:.3f} {feat.z:.3f} "
            f"1.500 {weight}"
        )

    for exc in pharmacophore.exclusion_spheres:
        lines.append(
            f"EXCLUSION {exc.x:.3f} {exc.y:.3f} {exc.z:.3f} "
            f"{exc.radius:.3f}"
        )

    cx, cy, cz = pharmacophore.pocket_centroid
    lines.append(f"CENTROID {cx:.3f} {cy:.3f} {cz:.3f}")
    lines.append("")
    return "\n".join(lines)


def _smiles_to_mol_block(smiles: str) -> str:
    """Convert SMILES to 3D mol block via RDKit ETKDG.

    Falls back to empty string if RDKit unavailable or conformer fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            # Retry with less strict parameters
            params.useRandomCoords = True
            status = AllChem.EmbedMolecule(mol, params)
            if status != 0:
                return ""

        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToMolBlock(mol)
    except ImportError:
        logger.warning("RDKit not available; skipping 3D conformer generation")
        return ""
    except Exception as e:
        logger.warning("3D conformer failed for %s: %s", smiles, e)
        return ""


def _parse_pgmg_output(
    output_path: Path,
    batch_id: str,
    pharmacophore: SpikePharmacophore,
) -> List[GeneratedMolecule]:
    """Parse PGMG output (one SMILES per line) into GeneratedMolecule list."""
    molecules: List[GeneratedMolecule] = []
    feature_types = [f.feature_type for f in pharmacophore.features]

    if not output_path.exists():
        return molecules

    lines = output_path.read_text().strip().splitlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        smiles = parts[0]

        # PGMG may output score in second column
        match_score = 0.0
        if len(parts) > 1:
            try:
                match_score = float(parts[1])
            except (ValueError, TypeError):
                pass

        mol_block = _smiles_to_mol_block(smiles)

        molecules.append(
            GeneratedMolecule(
                smiles=smiles,
                mol_block=mol_block,
                source="pgmg",
                pharmacophore_match_score=round(match_score, 4),
                matched_features=feature_types[:_MAX_PGMG_FEATURES],
                generation_batch_id=batch_id,
            )
        )

    return molecules


def run_pgmg(
    pharmacophore: SpikePharmacophore,
    n_molecules: int = 10000,
    output_dir: Optional[str] = None,
    conda_env: str = "pgmg",
    pgmg_dir: Optional[str] = None,
    timeout: int = 1800,
) -> List[GeneratedMolecule]:
    """Run PGMG to generate molecules from a pharmacophore.

    Args:
        pharmacophore: Input pharmacophore model.
        n_molecules: Number of molecules to generate (default 10000).
        output_dir: Directory for output files. Uses tempdir if None.
        conda_env: Name of the conda environment with PGMG installed.
        pgmg_dir: Override path to PGMG installation.
        timeout: Subprocess timeout in seconds (default 1800).

    Returns:
        List of GeneratedMolecule objects.

    Raises:
        RuntimeError: If PGMG subprocess fails.
        FileNotFoundError: If PGMG is not installed.
    """
    tool_dir = Path(pgmg_dir) if pgmg_dir else _PGMG_DIR

    if not tool_dir.exists():
        raise FileNotFoundError(
            f"PGMG not found at {tool_dir}. "
            f"Clone it: git clone <pgmg-repo> {tool_dir}"
        )

    batch_id = str(uuid.uuid4())
    work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write pharmacophore in PGMG format (truncated to 8 features)
    posp_path = work_dir / "pharmacophore.posp"
    posp_content = _truncate_pharmacophore(pharmacophore)
    posp_path.write_text(posp_content)

    output_file = work_dir / "pgmg_output.smi"

    script = tool_dir / "generate.py"
    cmd = [
        "conda", "run", "-n", conda_env, "--no-banner",
        "python", str(script),
        "--pharmacophore", str(posp_path),
        "--output", str(output_file),
        "--n_samples", str(n_molecules),
    ]

    logger.info(
        "Running PGMG: n=%d, output=%s, features=%d",
        n_molecules, output_file, min(len(pharmacophore.features), _MAX_PGMG_FEATURES),
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
            f"PGMG timed out after {timeout}s generating {n_molecules} molecules"
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"PGMG failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout[-500:]}\n"
            f"STDERR: {result.stderr[-500:]}"
        )

    if not output_file.exists():
        raise RuntimeError(f"PGMG produced no output file at {output_file}")

    molecules = _parse_pgmg_output(output_file, batch_id, pharmacophore)
    logger.info("PGMG produced %d valid molecules", len(molecules))

    return molecules
