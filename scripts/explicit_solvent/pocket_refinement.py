"""Pocket stability QC gate via explicit-solvent MD.

**This is the single most important QC gate in the entire PRISM-4D pipeline.**

Protocol
--------
1. Load PRISM output structure + spike JSON (pocket lining residues).
2. Solvate with TIP3P/OPC (truncated octahedron, 12 A buffer, 150 mM NaCl).
3. Energy minimise (5000 steps steepest descent).
4. NVT equilibration (500 ps, 300 K, Langevin thermostat, 2 fs step).
5. NPT equilibration (500 ps, 1 bar, Monte Carlo barostat, 4 fs step).
6. Production NPT MD (10–50 ns, 4 fs timestep, save every 10 ps).
7. Analyse:
   a. Pocket RMSD (Cα of lining residues over trajectory).
   b. Pocket volume (alpha-sphere or convex-hull on each frame).
   c. Classify:
      - RMSD < 2.0 Å  AND  volume σ < 20% → **STABLE**
      - RMSD 2.0–3.5 Å  OR  σ 20–40%     → **METASTABLE**
      - RMSD > 3.5 Å  OR  pocket closes   → **COLLAPSED**
   d. Identify structural waters (>80% occupancy).

Output
------
:class:`~scripts.interfaces.ExplicitSolventResult`

CLI
---
::

    python scripts/explicit_solvent/pocket_refinement.py \\
        --pdb tests/fixtures/kras.pdb \\
        --spike-json tests/fixtures/kras_spikes.json \\
        --time-ns 10 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Interface contracts (read-only)
from scripts.interfaces import ExplicitSolventResult, SpikePharmacophore

logger = logging.getLogger(__name__)

# ── Stability classification constants ────────────────────────────────────
RMSD_STABLE_THRESHOLD = 2.0       # Angstrom
RMSD_METASTABLE_THRESHOLD = 3.5   # Angstrom
VOLUME_CV_STABLE = 0.20           # 20% coefficient of variation
VOLUME_CV_METASTABLE = 0.40       # 40%

CLASSIFICATION_STABLE = "STABLE"
CLASSIFICATION_METASTABLE = "METASTABLE"
CLASSIFICATION_COLLAPSED = "COLLAPSED"


# ── Analysis helpers ──────────────────────────────────────────────────────

def classify_pocket_stability(
    rmsd_mean: float,
    volume_mean: float,
    volume_std: float,
) -> str:
    """Classify pocket stability from RMSD and volume statistics.

    Parameters
    ----------
    rmsd_mean : float
        Mean pocket RMSD over production trajectory (Angstrom).
    volume_mean : float
        Mean pocket volume (Angstrom^3).
    volume_std : float
        Standard deviation of pocket volume (Angstrom^3).

    Returns
    -------
    str
        ``"STABLE"``, ``"METASTABLE"``, or ``"COLLAPSED"``.
    """
    volume_cv = volume_std / volume_mean if volume_mean > 0 else 1.0

    # Blueprint thresholds (strict interpretation):
    #   RMSD < 2.0 AND CV < 20%  → STABLE
    #   RMSD 2.0–3.5 OR CV 20–40% → METASTABLE
    #   RMSD > 3.5  OR CV > 40%  → COLLAPSED
    if rmsd_mean > RMSD_METASTABLE_THRESHOLD or volume_cv > VOLUME_CV_METASTABLE:
        return CLASSIFICATION_COLLAPSED
    if rmsd_mean >= RMSD_STABLE_THRESHOLD or volume_cv >= VOLUME_CV_STABLE:
        return CLASSIFICATION_METASTABLE
    return CLASSIFICATION_STABLE


def compute_pocket_rmsd_trajectory(
    trajectory: Any,
    lining_residue_indices: List[int],
    reference_frame: int = 0,
) -> np.ndarray:
    """Compute per-frame Cα RMSD of pocket-lining residues.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        Production trajectory.
    lining_residue_indices : list of int
        Topology residue indices of pocket-lining residues.
    reference_frame : int
        Frame index to use as reference (default 0 = first production frame).

    Returns
    -------
    np.ndarray
        RMSD values in Angstrom, shape (n_frames,).
    """
    import mdtraj as md

    # Select Cα atoms of lining residues
    ca_indices = trajectory.topology.select(
        "name CA and resid " + " ".join(str(r) for r in lining_residue_indices)
    )
    if len(ca_indices) == 0:
        raise ValueError(
            f"No Cα atoms found for lining residues {lining_residue_indices}. "
            "Check residue indexing (0-based MDTraj vs 1-based PDB)."
        )

    ref = trajectory[reference_frame]
    rmsds = md.rmsd(trajectory, ref, atom_indices=ca_indices)
    # mdtraj returns nm → convert to Angstrom
    return rmsds * 10.0


def compute_pocket_volumes(
    trajectory: Any,
    pocket_centroid: Tuple[float, float, float],
    lining_residue_indices: List[int],
    radius_angstrom: float = 10.0,
) -> np.ndarray:
    """Estimate pocket volume per frame using alpha-sphere convex hull.

    Uses Cα positions of lining residues per frame, computes the convex hull
    volume.  This is a fast heuristic — POVME or fpocket could be used for
    higher accuracy but are much slower per-frame.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        Production trajectory.
    pocket_centroid : tuple
        (x, y, z) centroid of the pocket in Angstrom.
    lining_residue_indices : list of int
        Topology residue indices.
    radius_angstrom : float
        Radius around centroid to include atoms (default 10 A).

    Returns
    -------
    np.ndarray
        Volume in Angstrom^3 per frame, shape (n_frames,).
    """
    from scipy.spatial import ConvexHull

    ca_indices = trajectory.topology.select(
        "name CA and resid " + " ".join(str(r) for r in lining_residue_indices)
    )
    if len(ca_indices) < 4:
        raise ValueError(
            f"Need at least 4 Cα atoms for convex hull, got {len(ca_indices)}."
        )

    volumes = np.zeros(trajectory.n_frames)
    for i in range(trajectory.n_frames):
        # positions in nm → Angstrom
        coords = trajectory.xyz[i, ca_indices, :] * 10.0
        try:
            hull = ConvexHull(coords)
            volumes[i] = hull.volume
        except Exception:
            # Degenerate hull (coplanar points) — use 0
            volumes[i] = 0.0

    return volumes


def count_structural_waters(
    trajectory: Any,
    pocket_centroid: Tuple[float, float, float],
    radius_nm: float = 0.8,
    occupancy_threshold: float = 0.80,
    grid_spacing_nm: float = 0.05,
) -> int:
    """Count water sites with occupancy above threshold near the pocket.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        Explicit-solvent trajectory.
    pocket_centroid : tuple
        (x, y, z) centroid in Angstrom.
    radius_nm : float
        Radius around centroid in nm (default 0.8 = 8 A).
    occupancy_threshold : float
        Minimum fraction of frames a site must be occupied (default 0.80).
    grid_spacing_nm : float
        Grid resolution in nm (default 0.05 = 0.5 A).

    Returns
    -------
    int
        Number of high-occupancy water sites.
    """
    centroid_nm = np.array(pocket_centroid) / 10.0  # Angstrom → nm

    # Select water oxygen atoms
    water_oxy = trajectory.topology.select("water and name O")
    if len(water_oxy) == 0:
        return 0

    # Build a 3D histogram of water oxygen positions near the pocket
    n_bins = int(2.0 * radius_nm / grid_spacing_nm) + 1
    counts = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    origin = centroid_nm - radius_nm

    for frame_idx in range(trajectory.n_frames):
        coords = trajectory.xyz[frame_idx, water_oxy, :]
        # Filter to waters within the sphere
        dists = np.linalg.norm(coords - centroid_nm, axis=1)
        near = coords[dists < radius_nm]
        if len(near) == 0:
            continue
        # Bin positions
        grid_idx = ((near - origin) / grid_spacing_nm).astype(int)
        valid = np.all((grid_idx >= 0) & (grid_idx < n_bins), axis=1)
        for gi in grid_idx[valid]:
            counts[gi[0], gi[1], gi[2]] += 1

    # Normalise by number of frames → occupancy
    occupancy = counts / trajectory.n_frames
    return int(np.sum(occupancy >= occupancy_threshold))


def select_snapshot_frames(
    n_frames: int,
    n_snapshots: int = 10,
) -> List[int]:
    """Select evenly-spaced frames for ensemble scoring.

    Parameters
    ----------
    n_frames : int
        Total number of production frames.
    n_snapshots : int
        Number of snapshots to select (default 10).

    Returns
    -------
    list of int
        Frame indices.
    """
    if n_frames <= n_snapshots:
        return list(range(n_frames))
    step = n_frames // n_snapshots
    return list(range(step // 2, n_frames, step))[:n_snapshots]


# ── Main simulation runner ────────────────────────────────────────────────

def run_pocket_refinement(
    pdb_path: str,
    spike_json_path: str,
    time_ns: float = 10.0,
    water_model: str = "TIP3P",
    force_field: str = "ff14SB",
    output_dir: str = "output/explicit_solvent",
    platform_name: str = "CUDA",
    dry_run: bool = False,
) -> ExplicitSolventResult:
    """Run the full explicit-solvent pocket refinement protocol.

    Parameters
    ----------
    pdb_path : str
        Path to receptor PDB file.
    spike_json_path : str
        Path to PRISM spike JSON (SpikePharmacophore serialisation).
    time_ns : float
        Production simulation time in nanoseconds (default 10).
    water_model : str
        Water model: TIP3P (default), OPC, TIP4P-Ew.
    force_field : str
        Protein force field (default ff14SB).
    output_dir : str
        Directory for output files.
    platform_name : str
        OpenMM platform: CUDA (default), OpenCL, CPU, Reference.
    dry_run : bool
        If True, validate inputs and return a mock result without running MD.

    Returns
    -------
    ExplicitSolventResult
        Pocket stability result with classification.

    Raises
    ------
    FileNotFoundError
        If PDB or spike JSON files do not exist.
    RuntimeError
        If simulation fails.
    """
    pdb_file = Path(pdb_path)
    spike_file = Path(spike_json_path)
    out_dir = Path(output_dir)

    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if not spike_file.exists():
        raise FileNotFoundError(f"Spike JSON not found: {spike_json_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load spike pharmacophore
    with open(spike_file) as f:
        spike_data = json.load(f)
    pharmacophore = SpikePharmacophore.from_dict(spike_data)

    pocket_id = pharmacophore.pocket_id
    lining_residues = pharmacophore.pocket_lining_residues
    pocket_centroid = pharmacophore.pocket_centroid

    logger.info("=== POCKET REFINEMENT QC GATE ===")
    logger.info("Target: %s | Pocket: %d", pharmacophore.target_name, pocket_id)
    logger.info("Lining residues (%d): %s", len(lining_residues), lining_residues)
    logger.info("Centroid: (%.2f, %.2f, %.2f)", *pocket_centroid)
    logger.info("Simulation: %.1f ns | Water: %s | FF: %s",
                time_ns, water_model, force_field)

    # ── Dry-run mode: validate inputs, return mock result ─────────────
    if dry_run:
        logger.info("[DRY RUN] Skipping MD simulation")
        result = ExplicitSolventResult(
            pocket_id=pocket_id,
            simulation_time_ns=time_ns,
            water_model=water_model,
            force_field=force_field,
            pocket_stable=True,
            pocket_rmsd_mean=1.2,
            pocket_rmsd_std=0.3,
            pocket_volume_mean=450.0,
            pocket_volume_std=45.0,
            n_structural_waters=3,
            trajectory_path=str(out_dir / "production.dcd"),
            snapshot_frames=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        )
        classification = classify_pocket_stability(
            result.pocket_rmsd_mean,
            result.pocket_volume_mean,
            result.pocket_volume_std,
        )
        logger.info("[DRY RUN] Classification: %s (mock)", classification)
        # Write result JSON
        result_path = out_dir / f"pocket_{pocket_id}_result.json"
        result_path.write_text(result.to_json())
        logger.info("[DRY RUN] Result written to %s", result_path)
        return result

    # ── Full simulation ───────────────────────────────────────────────
    try:
        import mdtraj as md
        import openmm as omm
        from openmm import app as omm_app
        from openmm import unit as omm_unit
    except ImportError as exc:
        raise ImportError(
            "OpenMM and MDTraj are required. "
            "Install via: conda env create -f envs/explicit_solvent.yml"
        ) from exc

    from .solvent_setup import prepare_solvated_system

    # Step 1: Solvate
    logger.info("Step 1/6: Solvation")
    solvent_sys = prepare_solvated_system(
        pdb_path=pdb_path,
        water_model=water_model,
        force_field=force_field,
    )

    # Write solvated PDB for reference
    solvated_pdb = out_dir / "solvated.pdb"
    positions_quantity = solvent_sys.positions * omm_unit.nanometers
    with open(solvated_pdb, "w") as f:
        omm_app.PDBFile.writeFile(solvent_sys.topology, positions_quantity, f)

    # Step 2: Energy minimisation
    logger.info("Step 2/6: Energy minimisation (5000 steps)")
    integrator = omm.LangevinMiddleIntegrator(
        300 * omm_unit.kelvin,
        1.0 / omm_unit.picoseconds,
        0.002 * omm_unit.picoseconds,   # 2 fs for minimisation
    )
    platform = omm.Platform.getPlatformByName(platform_name)
    simulation = omm_app.Simulation(
        solvent_sys.topology, solvent_sys.system, integrator, platform,
    )
    simulation.context.setPositions(solvent_sys.positions * omm_unit.nanometers)
    simulation.minimizeEnergy(maxIterations=5000)
    logger.info("  Minimisation complete")

    # Step 3: NVT equilibration (500 ps, 300K)
    logger.info("Step 3/6: NVT equilibration (500 ps, 300 K)")
    simulation.context.setVelocitiesToTemperature(300 * omm_unit.kelvin)
    nvt_steps = int(500.0 / 0.002)   # 500 ps / 2 fs = 250,000 steps
    simulation.step(nvt_steps)
    logger.info("  NVT equilibration complete")

    # Step 4: NPT equilibration (500 ps, 1 bar)
    logger.info("Step 4/6: NPT equilibration (500 ps, 1 bar)")
    # Add barostat for NPT
    barostat = omm.MonteCarloBarostat(
        1.0 * omm_unit.bar, 300 * omm_unit.kelvin, 25,
    )
    solvent_sys.system.addForce(barostat)
    # Re-create integrator with 4 fs step (HMR enables this)
    integrator_npt = omm.LangevinMiddleIntegrator(
        300 * omm_unit.kelvin,
        1.0 / omm_unit.picoseconds,
        0.004 * omm_unit.picoseconds,   # 4 fs with HMR
    )
    simulation_npt = omm_app.Simulation(
        solvent_sys.topology, solvent_sys.system, integrator_npt, platform,
    )
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    simulation_npt.context.setPositions(state.getPositions())
    simulation_npt.context.setVelocities(state.getVelocities())

    npt_equil_steps = int(500.0 / 0.004)  # 500 ps / 4 fs = 125,000 steps
    simulation_npt.step(npt_equil_steps)
    logger.info("  NPT equilibration complete")

    # Step 5: Production NPT MD
    production_ps = time_ns * 1000.0
    dt_ps = 0.004
    total_steps = int(production_ps / dt_ps)
    save_interval_ps = 10.0  # save every 10 ps
    report_interval = int(save_interval_ps / dt_ps)  # = 2500 steps

    traj_path = out_dir / "production.dcd"
    logger.info("Step 5/6: Production MD (%.1f ns, %d steps, saving every %d steps)",
                time_ns, total_steps, report_interval)

    simulation_npt.reporters.append(
        md.reporters.DCDReporter(str(traj_path), report_interval)
    )
    simulation_npt.reporters.append(
        omm_app.StateDataReporter(
            str(out_dir / "production.log"),
            report_interval * 10,
            step=True, time=True, potentialEnergy=True,
            temperature=True, speed=True,
        )
    )

    t0 = time.time()
    simulation_npt.step(total_steps)
    wall_time = time.time() - t0
    logger.info("  Production complete in %.1f s (%.1f ns/day)",
                wall_time, time_ns / (wall_time / 86400))

    # Step 6: Analysis
    logger.info("Step 6/6: Trajectory analysis")

    # Load trajectory
    traj = md.load(str(traj_path), top=str(solvated_pdb))
    logger.info("  Loaded %d frames", traj.n_frames)

    # 6a. Pocket RMSD
    rmsds = compute_pocket_rmsd_trajectory(traj, lining_residues)
    rmsd_mean = float(np.mean(rmsds))
    rmsd_std = float(np.std(rmsds))

    # 6b. Pocket volume
    volumes = compute_pocket_volumes(traj, pocket_centroid, lining_residues)
    vol_mean = float(np.mean(volumes))
    vol_std = float(np.std(volumes))

    # 6c. Classification
    classification = classify_pocket_stability(rmsd_mean, vol_mean, vol_std)
    pocket_stable = classification != CLASSIFICATION_COLLAPSED

    # 6d. Structural waters
    n_struct_waters = count_structural_waters(traj, pocket_centroid)

    # Select snapshot frames for ensemble scoring
    snapshots = select_snapshot_frames(traj.n_frames)

    logger.info("=== POCKET REFINEMENT RESULT ===")
    logger.info("  RMSD:   %.2f +/- %.2f A", rmsd_mean, rmsd_std)
    logger.info("  Volume: %.1f +/- %.1f A^3 (CV=%.1f%%)",
                vol_mean, vol_std,
                (vol_std / vol_mean * 100) if vol_mean > 0 else 0)
    logger.info("  Structural waters: %d", n_struct_waters)
    logger.info("  *** CLASSIFICATION: %s ***", classification)
    if classification == CLASSIFICATION_COLLAPSED:
        logger.warning("  >>> POCKET COLLAPSED — pipeline STOPS here <<<")

    result = ExplicitSolventResult(
        pocket_id=pocket_id,
        simulation_time_ns=time_ns,
        water_model=water_model,
        force_field=force_field,
        pocket_stable=pocket_stable,
        pocket_rmsd_mean=rmsd_mean,
        pocket_rmsd_std=rmsd_std,
        pocket_volume_mean=vol_mean,
        pocket_volume_std=vol_std,
        n_structural_waters=n_struct_waters,
        trajectory_path=str(traj_path.resolve()),
        snapshot_frames=snapshots,
    )

    # Write result JSON
    result_path = out_dir / f"pocket_{pocket_id}_result.json"
    result_path.write_text(result.to_json())
    logger.info("Result written to %s", result_path)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for pocket refinement."""
    parser = argparse.ArgumentParser(
        description="Explicit-solvent pocket stability QC gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pdb", required=True, help="Receptor PDB file")
    parser.add_argument("--spike-json", required=True,
                        help="PRISM spike JSON (SpikePharmacophore)")
    parser.add_argument("--time-ns", type=float, default=10.0,
                        help="Production MD time in ns (default: 10)")
    parser.add_argument("--water-model", default="TIP3P",
                        choices=["TIP3P", "OPC", "TIP4P-Ew"],
                        help="Water model (default: TIP3P)")
    parser.add_argument("--force-field", default="ff14SB",
                        choices=["ff14SB", "ff19SB", "CHARMM36m"],
                        help="Protein force field (default: ff14SB)")
    parser.add_argument("--output-dir", default="output/explicit_solvent",
                        help="Output directory")
    parser.add_argument("--platform", default="CUDA",
                        choices=["CUDA", "OpenCL", "CPU", "Reference"],
                        help="OpenMM platform (default: CUDA)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs, return mock result (no MD)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = run_pocket_refinement(
        pdb_path=args.pdb,
        spike_json_path=args.spike_json,
        time_ns=args.time_ns,
        water_model=args.water_model,
        force_field=args.force_field,
        output_dir=args.output_dir,
        platform_name=args.platform,
        dry_run=args.dry_run,
    )

    classification = classify_pocket_stability(
        result.pocket_rmsd_mean,
        result.pocket_volume_mean,
        result.pocket_volume_std,
    )

    # Exit code: 0=STABLE, 1=COLLAPSED (pipeline stop), 2=METASTABLE
    if classification == CLASSIFICATION_COLLAPSED:
        sys.exit(1)
    elif classification == CLASSIFICATION_METASTABLE:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
