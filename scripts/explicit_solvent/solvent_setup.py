"""Explicit-solvent system preparation for pocket-stability MD.

Builds an OpenMM ``Simulation`` from a PDB file:
    1. Fix missing atoms/residues via PDBFixer.
    2. Solvate in a truncated-octahedron (or cubic) box with the chosen water
       model (TIP3P, OPC, TIP4P-Ew).
    3. Add 150 mM NaCl (neutralise first, then add excess ions).
    4. Return an :class:`ExplicitSolventSystem` with topology, positions,
       and a fully-parameterised OpenMM ``System`` ready for minimisation.

All heavy lifting is deferred to OpenMM — this module is thin glue.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
WATER_MODELS: Dict[str, Dict[str, str]] = {
    "TIP3P": {
        "ff_water": "tip3p.xml",
        "model_name": "tip3p",
    },
    "OPC": {
        "ff_water": "opc.xml",
        "model_name": "opc",
    },
    "TIP4P-Ew": {
        "ff_water": "tip4pew.xml",
        "model_name": "tip4pew",
    },
}

FORCE_FIELDS: Dict[str, str] = {
    "ff19SB": "amber/ff19SB.xml",
    "ff14SB": "amber14/protein.ff14SB.xml",
    "CHARMM36m": "charmm36.xml",
}

DEFAULT_IONIC_STRENGTH_M = 0.15   # 150 mM NaCl
DEFAULT_BOX_PADDING_NM = 1.2      # 12 Angstrom
DEFAULT_FORCE_FIELD = "ff14SB"
DEFAULT_WATER_MODEL = "TIP3P"


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class ExplicitSolventSystem:
    """A fully-parameterised, solvated OpenMM system ready for simulation.

    Attributes
    ----------
    topology : object
        OpenMM ``Topology`` (kept as ``object`` to avoid hard import at
        class-definition time).
    positions : np.ndarray
        Particle positions, shape (n_atoms, 3), in nanometres.
    system : object
        OpenMM ``System`` with force-field parameters applied.
    water_model : str
        Water model name used (TIP3P / OPC / TIP4P-Ew).
    force_field : str
        Protein force-field label (ff14SB / ff19SB / CHARMM36m).
    n_waters : int
        Number of water molecules added.
    n_cations : int
        Number of Na+ ions.
    n_anions : int
        Number of Cl- ions.
    box_vectors : np.ndarray
        Periodic box vectors, shape (3, 3), nanometres.
    """
    topology: object
    positions: np.ndarray
    system: object
    water_model: str
    force_field: str
    n_waters: int
    n_cations: int
    n_anions: int
    box_vectors: np.ndarray


# ── Public API ────────────────────────────────────────────────────────────

def prepare_solvated_system(
    pdb_path: str,
    water_model: str = DEFAULT_WATER_MODEL,
    force_field: str = DEFAULT_FORCE_FIELD,
    padding_nm: float = DEFAULT_BOX_PADDING_NM,
    ionic_strength: float = DEFAULT_IONIC_STRENGTH_M,
    box_shape: str = "dodecahedron",
) -> ExplicitSolventSystem:
    """Build a solvated, ion-neutralised OpenMM system from a PDB file.

    Parameters
    ----------
    pdb_path : str
        Path to the input PDB file.
    water_model : str
        Water model: ``"TIP3P"`` (default), ``"OPC"``, or ``"TIP4P-Ew"``.
    force_field : str
        Protein force field: ``"ff14SB"`` (default), ``"ff19SB"``, or
        ``"CHARMM36m"``.
    padding_nm : float
        Minimum distance from solute to box edge, in nm (default 1.2 = 12 A).
    ionic_strength : float
        Ionic strength in mol/L (default 0.15 = 150 mM NaCl).
    box_shape : str
        ``"dodecahedron"`` (truncated octahedron, default) or ``"cube"``.

    Returns
    -------
    ExplicitSolventSystem
        Ready-to-simulate system with topology, positions, and OpenMM System.

    Raises
    ------
    ValueError
        If water_model or force_field is not recognised.
    ImportError
        If OpenMM is not installed.
    """
    # Lazy imports — keep module importable without OpenMM for testing
    try:
        from openmm import app as omm_app
        from openmm import unit as omm_unit
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise ImportError(
            "OpenMM and PDBFixer are required for solvent setup. "
            "Install via: conda install -c conda-forge openmm pdbfixer"
        ) from exc

    # Validate inputs
    if water_model not in WATER_MODELS:
        raise ValueError(
            f"Unknown water model '{water_model}'. "
            f"Choose from: {list(WATER_MODELS)}"
        )
    if force_field not in FORCE_FIELDS:
        raise ValueError(
            f"Unknown force field '{force_field}'. "
            f"Choose from: {list(FORCE_FIELDS)}"
        )

    pdb_file = Path(pdb_path)
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    logger.info("Preparing solvated system from %s", pdb_path)
    logger.info("  Water model: %s | Force field: %s", water_model, force_field)
    logger.info("  Padding: %.1f nm | Ionic strength: %.3f M", padding_nm, ionic_strength)

    # 1. Fix missing atoms / residues
    fixer = PDBFixer(filename=str(pdb_file))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)
    logger.info("  PDBFixer: missing atoms added, hydrogens at pH 7.4")

    # 2. Load force field
    wm_info = WATER_MODELS[water_model]
    ff_xml = FORCE_FIELDS[force_field]
    forcefield = omm_app.ForceField(ff_xml, wm_info["ff_water"])

    # 3. Solvate
    modeller = omm_app.Modeller(fixer.topology, fixer.positions)

    if box_shape == "dodecahedron":
        modeller.addSolvent(
            forcefield,
            model=wm_info["model_name"],
            padding=padding_nm * omm_unit.nanometers,
            boxShape="dodecahedron",
            ionicStrength=ionic_strength * omm_unit.molar,
        )
    else:
        modeller.addSolvent(
            forcefield,
            model=wm_info["model_name"],
            padding=padding_nm * omm_unit.nanometers,
            ionicStrength=ionic_strength * omm_unit.molar,
        )

    # Count waters and ions
    n_waters = 0
    n_na = 0
    n_cl = 0
    for residue in modeller.topology.residues():
        name = residue.name
        if name == "HOH" or name == "WAT" or name == "SOL":
            n_waters += 1
        elif name == "NA" or name == "Na+":
            n_na += 1
        elif name == "CL" or name == "Cl-":
            n_cl += 1

    logger.info("  Solvation: %d waters, %d Na+, %d Cl-", n_waters, n_na, n_cl)

    # 4. Create parameterised system
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=omm_app.PME,
        nonbondedCutoff=1.0 * omm_unit.nanometers,
        constraints=omm_app.HBonds,
        hydrogenMass=1.5 * omm_unit.amu,  # HMR for 4 fs timestep
    )

    # Extract positions and box vectors as numpy arrays
    positions = np.array(
        modeller.positions.value_in_unit(omm_unit.nanometers)
    )
    box_vectors = np.array(
        modeller.topology.getPeriodicBoxVectors().value_in_unit(omm_unit.nanometers)
    )

    logger.info("  System: %d particles, PME, HBonds, HMR 1.5 amu",
                system.getNumParticles())

    return ExplicitSolventSystem(
        topology=modeller.topology,
        positions=positions,
        system=system,
        water_model=water_model,
        force_field=force_field,
        n_waters=n_waters,
        n_cations=n_na,
        n_anions=n_cl,
        box_vectors=box_vectors,
    )


def save_solvated_pdb(
    solvent_system: ExplicitSolventSystem,
    output_path: str,
) -> str:
    """Write the solvated system to a PDB file.

    Parameters
    ----------
    solvent_system : ExplicitSolventSystem
        System returned by :func:`prepare_solvated_system`.
    output_path : str
        Output PDB path.

    Returns
    -------
    str
        Absolute path to the written PDB.
    """
    try:
        from openmm import app as omm_app
        from openmm import unit as omm_unit
    except ImportError as exc:
        raise ImportError("OpenMM required to write PDB") from exc

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    positions_quantity = solvent_system.positions * omm_unit.nanometers
    with open(out, "w") as f:
        omm_app.PDBFile.writeFile(
            solvent_system.topology,
            positions_quantity,
            f,
        )
    logger.info("Solvated PDB written to %s", out)
    return str(out.resolve())
