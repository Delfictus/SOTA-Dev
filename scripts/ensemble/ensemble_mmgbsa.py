"""Ensemble-averaged MM-GBSA free-energy scoring.

Replaces single-snapshot MM-GBSA with ensemble averaging over MD trajectory
snapshots (typically 100 frames at 100 ps intervals from a 10 ns production
run).  Uses AmberTools MMPBSA.py or gmx_MMPBSA as the per-frame engine.

Blueprint spec (WT-7)
---------------------
1. Extract snapshots every ``snapshot_interval_ps`` (default 100 ps).
2. Per frame: strip solvent, compute MM-GBSA decomposition.
3. Average: dG_mean, dG_std, dG_sem (block analysis for correlated data).
4. Per-residue decomposition for hot-spot identification.

BEFORE: Single pose → −28 kcal/mol (unrealistic)
AFTER:  100 frames → −8.2 ± 1.4 kcal/mol (realistic)

References
----------
- Genheden S, Ryde U. Expert Opin Drug Discov. 2015;10(5):449-461.
- Miller BR III et al. J Chem Theory Comput. 2012;8(9):3314-3321.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.interfaces import EnsembleMMGBSA

from .block_analysis import BlockAverageResult, block_average

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_SNAPSHOT_INTERVAL_PS = 100.0
DEFAULT_GB_MODEL = "igb5"  # OBC2 (igb=5) — standard for MM-GBSA
DEFAULT_SALT_CONC = 0.15  # 150 mM NaCl

# ── Energy components tracked ────────────────────────────────────────────

ENERGY_COMPONENTS = ("vdw", "elec", "gb", "sa")


@dataclass
class EnsembleMMGBSAConfig:
    """Configuration for ensemble MM-GBSA calculation.

    Attributes
    ----------
    snapshot_interval_ps : float
        Interval between trajectory snapshots in picoseconds.
    gb_model : str
        Generalised Born model (igb=2 or igb=5).
    salt_concentration : float
        Ionic strength in mol/L for GB screening.
    decompose_residues : bool
        Whether to compute per-residue energy decomposition.
    method : str
        "MMGBSA_ensemble" or "MMPBSA_ensemble".
    engine : str
        Backend engine: "amber" for MMPBSA.py, "gmx" for gmx_MMPBSA.
    """

    snapshot_interval_ps: float = DEFAULT_SNAPSHOT_INTERVAL_PS
    gb_model: str = DEFAULT_GB_MODEL
    salt_concentration: float = DEFAULT_SALT_CONC
    decompose_residues: bool = True
    method: str = "MMGBSA_ensemble"
    engine: str = "amber"


def select_frames(
    n_total_frames: int,
    interval_ps: float,
    timestep_ps: float = 2.0,
    save_interval: int = 500,
) -> List[int]:
    """Select evenly-spaced frame indices from a trajectory.

    Parameters
    ----------
    n_total_frames : int
        Total number of frames in the trajectory.
    interval_ps : float
        Desired interval between selected frames in picoseconds.
    timestep_ps : float
        MD integration timestep in picoseconds.
    save_interval : int
        Trajectory save frequency (every N steps).

    Returns
    -------
    list of int
        0-based frame indices to extract.
    """
    ps_per_frame = timestep_ps * save_interval
    frame_stride = max(1, int(round(interval_ps / ps_per_frame)))
    frames = list(range(0, n_total_frames, frame_stride))
    if not frames:
        frames = [0]
    return frames


def _parse_mmpbsa_output(output_path: str) -> Dict[str, Any]:
    """Parse MMPBSA.py FINAL_RESULTS_MMPBSA.dat output.

    Parameters
    ----------
    output_path : str
        Path to the MMPBSA output file.

    Returns
    -------
    dict
        Per-frame energies and decomposition data.

    Raises
    ------
    FileNotFoundError
        If output file does not exist.
    RuntimeError
        If the output file cannot be parsed.
    """
    path = Path(output_path)
    if not path.exists():
        raise FileNotFoundError(f"MMPBSA output not found: {output_path}")

    results: Dict[str, Any] = {
        "per_frame_total": [],
        "decomposition": {comp: [] for comp in ENERGY_COMPONENTS},
        "per_residue": {},
    }

    text = path.read_text()

    # Parse the standard MMPBSA.py output format
    in_results = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("DELTA TOTAL"):
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    results["per_frame_total"].append(float(parts[2]))
                except (ValueError, IndexError):
                    pass
            in_results = True
        elif in_results and stripped.startswith("VDWAALS"):
            parts = stripped.split()
            if len(parts) >= 2:
                results["decomposition"]["vdw"].append(float(parts[1]))
        elif in_results and stripped.startswith("EEL"):
            parts = stripped.split()
            if len(parts) >= 2:
                results["decomposition"]["elec"].append(float(parts[1]))
        elif in_results and stripped.startswith("EGB"):
            parts = stripped.split()
            if len(parts) >= 2:
                results["decomposition"]["gb"].append(float(parts[1]))
        elif in_results and stripped.startswith("ESURF"):
            parts = stripped.split()
            if len(parts) >= 2:
                results["decomposition"]["sa"].append(float(parts[1]))

    return results


def compute_ensemble_mmgbsa(
    per_frame_energies: np.ndarray,
    decomposition: Dict[str, np.ndarray],
    per_residue_contributions: Optional[Dict[int, np.ndarray]] = None,
    compound_id: str = "unknown",
    snapshot_interval_ps: float = DEFAULT_SNAPSHOT_INTERVAL_PS,
    method: str = "MMGBSA_ensemble",
) -> EnsembleMMGBSA:
    """Compute ensemble-averaged MM-GBSA from per-frame energies.

    This is the core calculation, independent of the backend engine.
    It takes pre-computed per-frame energies and produces the ensemble
    average with proper error estimation via block averaging.

    Parameters
    ----------
    per_frame_energies : np.ndarray
        1-D array of MM-GBSA total energies per frame (kcal/mol).
    decomposition : dict
        Per-component energies: keys "vdw", "elec", "gb", "sa" each
        mapping to a 1-D array of per-frame values.
    per_residue_contributions : dict, optional
        Mapping of topology residue IDs to per-frame energy arrays.
    compound_id : str
        Compound identifier.
    snapshot_interval_ps : float
        Time between snapshots in picoseconds.
    method : str
        Scoring method label.

    Returns
    -------
    EnsembleMMGBSA
        Interface-compliant ensemble scoring result.

    Raises
    ------
    ValueError
        If arrays are empty or have mismatched lengths.
    """
    energies = np.asarray(per_frame_energies, dtype=np.float64).ravel()
    n_snapshots = len(energies)

    if n_snapshots == 0:
        raise ValueError("No per-frame energies provided")

    # Ensemble average with block-averaged SEM
    if n_snapshots >= 5:
        ba = block_average(energies)
        delta_g_mean = ba.mean
        delta_g_sem = ba.sem
    else:
        delta_g_mean = float(np.mean(energies))
        delta_g_sem = float(np.std(energies, ddof=1) / np.sqrt(n_snapshots))

    delta_g_std = float(np.std(energies, ddof=1)) if n_snapshots > 1 else 0.0

    # Component averages
    avg_decomposition: Dict[str, float] = {}
    for comp in ENERGY_COMPONENTS:
        if comp in decomposition and len(decomposition[comp]) > 0:
            avg_decomposition[comp] = float(np.mean(decomposition[comp]))
        else:
            avg_decomposition[comp] = 0.0

    # Per-residue averages
    avg_per_residue: Dict[int, float] = {}
    if per_residue_contributions:
        for resid, values in per_residue_contributions.items():
            arr = np.asarray(values, dtype=np.float64)
            avg_per_residue[int(resid)] = float(np.mean(arr))

    return EnsembleMMGBSA(
        compound_id=compound_id,
        delta_g_mean=delta_g_mean,
        delta_g_std=delta_g_std,
        delta_g_sem=delta_g_sem,
        n_snapshots=n_snapshots,
        snapshot_interval_ps=snapshot_interval_ps,
        decomposition=avg_decomposition,
        per_residue_contributions=avg_per_residue,
        method=method,
    )


def run_ensemble_mmgbsa(
    topology_path: str,
    trajectory_path: str,
    compound_id: str,
    config: Optional[EnsembleMMGBSAConfig] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
) -> EnsembleMMGBSA:
    """Run the full ensemble MM-GBSA pipeline.

    Parameters
    ----------
    topology_path : str
        Path to the protein-ligand topology file (prmtop or gro).
    trajectory_path : str
        Path to the production trajectory (dcd, xtc, nc).
    compound_id : str
        Compound identifier.
    config : EnsembleMMGBSAConfig, optional
        Calculation configuration.  Uses defaults if *None*.
    output_dir : str, optional
        Directory for output files.  Uses a temp dir if *None*.
    dry_run : bool
        If *True*, return a synthetic result without running MMPBSA.

    Returns
    -------
    EnsembleMMGBSA
        Ensemble scoring result.

    Raises
    ------
    FileNotFoundError
        If topology or trajectory files are missing.
    RuntimeError
        If the MMPBSA engine fails.
    """
    if config is None:
        config = EnsembleMMGBSAConfig()

    top_path = Path(topology_path)
    traj_path = Path(trajectory_path)

    if not top_path.exists():
        raise FileNotFoundError(f"Topology not found: {topology_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="ensemble_mmgbsa_")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return _generate_dry_run_result(compound_id, config)

    # Real execution path — calls MMPBSA.py or gmx_MMPBSA
    logger.info(
        "Running ensemble MM-GBSA: %s frames at %s ps intervals",
        "auto",
        config.snapshot_interval_ps,
    )

    if config.engine == "amber":
        result = _run_amber_mmpbsa(
            top_path, traj_path, compound_id, config, out_dir
        )
    elif config.engine == "gmx":
        result = _run_gmx_mmpbsa(
            top_path, traj_path, compound_id, config, out_dir
        )
    else:
        raise ValueError(f"Unknown engine: {config.engine!r}")

    # Write result JSON
    result_file = out_dir / f"{compound_id}_ensemble_mmgbsa.json"
    result_file.write_text(result.to_json())
    logger.info("Result written to %s", result_file)

    return result


def _generate_dry_run_result(
    compound_id: str, config: EnsembleMMGBSAConfig
) -> EnsembleMMGBSA:
    """Generate a synthetic result for dry-run mode."""
    rng = np.random.default_rng(42)
    n_frames = 100
    # Simulate realistic MM-GBSA energies centred around -8 kcal/mol
    energies = rng.normal(-8.0, 2.5, n_frames)
    decomp = {
        "vdw": rng.normal(-25.0, 3.0, n_frames),
        "elec": rng.normal(-15.0, 5.0, n_frames),
        "gb": rng.normal(30.0, 4.0, n_frames),
        "sa": rng.normal(-3.0, 0.5, n_frames),
    }
    per_residue = {
        12: rng.normal(-2.5, 0.3, n_frames),
        34: rng.normal(-1.8, 0.4, n_frames),
        60: rng.normal(-0.5, 0.2, n_frames),
    }
    return compute_ensemble_mmgbsa(
        per_frame_energies=energies,
        decomposition={k: np.asarray(v) for k, v in decomp.items()},
        per_residue_contributions={k: np.asarray(v) for k, v in per_residue.items()},
        compound_id=compound_id,
        snapshot_interval_ps=config.snapshot_interval_ps,
        method=config.method,
    )


def _run_amber_mmpbsa(
    topology: Path,
    trajectory: Path,
    compound_id: str,
    config: EnsembleMMGBSAConfig,
    out_dir: Path,
) -> EnsembleMMGBSA:
    """Execute MMPBSA.py from AmberTools.

    Raises RuntimeError if the external process fails.
    """
    # Write MMPBSA input file
    input_text = (
        f"&general\n"
        f"  interval=1, verbose=2,\n"
        f"/\n"
        f"&gb\n"
        f"  igb={config.gb_model.replace('igb', '')}, saltcon={config.salt_concentration},\n"
        f"/\n"
    )
    if config.decompose_residues:
        input_text += "&decomp\n  idecomp=1, dec_verbose=0,\n/\n"

    input_file = out_dir / "mmpbsa.in"
    input_file.write_text(input_text)

    output_file = out_dir / "FINAL_RESULTS_MMPBSA.dat"
    cmd = [
        "MMPBSA.py",
        "-i", str(input_file),
        "-sp", str(topology),
        "-y", str(trajectory),
        "-o", str(output_file),
    ]

    logger.info("Executing: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(out_dir))

    if proc.returncode != 0:
        raise RuntimeError(
            f"MMPBSA.py failed (rc={proc.returncode}):\n{proc.stderr[:2000]}"
        )

    parsed = _parse_mmpbsa_output(str(output_file))
    energies = np.array(parsed["per_frame_total"])
    decomp = {k: np.array(v) for k, v in parsed["decomposition"].items()}

    return compute_ensemble_mmgbsa(
        per_frame_energies=energies,
        decomposition=decomp,
        compound_id=compound_id,
        snapshot_interval_ps=config.snapshot_interval_ps,
        method=config.method,
    )


def _run_gmx_mmpbsa(
    topology: Path,
    trajectory: Path,
    compound_id: str,
    config: EnsembleMMGBSAConfig,
    out_dir: Path,
) -> EnsembleMMGBSA:
    """Execute gmx_MMPBSA (GROMACS wrapper).

    Raises RuntimeError if the external process fails.
    """
    input_text = (
        f"&general\n"
        f"  sys_name=\"{compound_id}\", interval=1,\n"
        f"/\n"
        f"&gb\n"
        f"  igb={config.gb_model.replace('igb', '')}, saltcon={config.salt_concentration},\n"
        f"/\n"
    )
    if config.decompose_residues:
        input_text += "&decomp\n  idecomp=1, dec_verbose=0,\n/\n"

    input_file = out_dir / "mmpbsa.in"
    input_file.write_text(input_text)

    output_file = out_dir / "FINAL_RESULTS_MMPBSA.dat"
    cmd = [
        "gmx_MMPBSA",
        "-i", str(input_file),
        "-cs", str(topology),
        "-ct", str(trajectory),
        "-o", str(output_file),
    ]

    logger.info("Executing: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(out_dir))

    if proc.returncode != 0:
        raise RuntimeError(
            f"gmx_MMPBSA failed (rc={proc.returncode}):\n{proc.stderr[:2000]}"
        )

    parsed = _parse_mmpbsa_output(str(output_file))
    energies = np.array(parsed["per_frame_total"])
    decomp = {k: np.array(v) for k, v in parsed["decomposition"].items()}

    return compute_ensemble_mmgbsa(
        per_frame_energies=energies,
        decomposition=decomp,
        compound_id=compound_id,
        snapshot_interval_ps=config.snapshot_interval_ps,
        method=config.method,
    )


def get_hotspot_residues(
    result: EnsembleMMGBSA,
    threshold_kcal: float = -1.0,
) -> List[Tuple[int, float]]:
    """Identify hot-spot residues contributing strongly to binding.

    Parameters
    ----------
    result : EnsembleMMGBSA
        Ensemble scoring result with per-residue decomposition.
    threshold_kcal : float
        Residues with average contribution below this threshold
        (more negative = more stabilising) are hot-spots.

    Returns
    -------
    list of (int, float)
        (residue_id, average_contribution) sorted by contribution
        (most stabilising first).
    """
    hotspots = [
        (resid, energy)
        for resid, energy in result.per_residue_contributions.items()
        if energy < threshold_kcal
    ]
    return sorted(hotspots, key=lambda x: x[1])
