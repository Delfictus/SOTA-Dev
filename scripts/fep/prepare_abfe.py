"""ABFE protocol preparation — configure alchemical lambda schedules.

Sets up:
    - Complex leg:  42 lambda windows (charge + vdW decoupling)
    - Solvent leg:   31 lambda windows
    - 5 ns per window, 3 independent repeats
    - Optional REST2 enhanced sampling for flexible pockets
    - Boresch orientational restraint configuration

Can be run as a CLI:
    python scripts/fep/prepare_abfe.py \\
        --protein tests/test_fep/fixtures/mock_protein.pdb \\
        --ligand tests/test_fep/fixtures/mock_ligand.sdf \\
        --output-dir /tmp/fep_test/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LambdaSchedule:
    """Lambda window schedule for one alchemical leg."""
    name: str
    n_windows: int
    # Electrostatic lambda values (charge decoupling)
    lambda_electrostatics: List[float] = field(default_factory=list)
    # Van der Waals lambda values (steric decoupling)
    lambda_sterics: List[float] = field(default_factory=list)
    # Restraint lambda values (Boresch restraint)
    lambda_restraints: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _make_complex_schedule(n_windows: int = 42) -> LambdaSchedule:
    """Generate the complex-leg lambda schedule.

    Uses a 3-stage protocol:
      Stage 1: Turn on restraints (5 windows)
      Stage 2: Discharge ligand (12 windows)
      Stage 3: Decouple vdW (25 windows, soft-core)
    """
    # Stage 1: Restraint on (elec=1, vdw=1, restraint 0→1)
    n_restraint = 5
    lam_restraint_stage = [i / (n_restraint - 1) for i in range(n_restraint)]

    # Stage 2: Discharge (elec 1→0, vdw=1, restraint=1)
    n_elec = 12
    lam_elec_stage = [1.0 - i / (n_elec - 1) for i in range(n_elec)]

    # Stage 3: Decouple vdW (elec=0, vdw 1→0, restraint=1)
    n_vdw = n_windows - n_restraint - n_elec  # 25
    lam_vdw_stage = [1.0 - i / (n_vdw - 1) for i in range(n_vdw)]

    # Compose full schedule
    lambda_elec = (
        [1.0] * n_restraint
        + lam_elec_stage
        + [0.0] * n_vdw
    )
    lambda_sterics = (
        [1.0] * n_restraint
        + [1.0] * n_elec
        + lam_vdw_stage
    )
    lambda_restraints = (
        lam_restraint_stage
        + [1.0] * n_elec
        + [1.0] * n_vdw
    )

    return LambdaSchedule(
        name="complex",
        n_windows=n_windows,
        lambda_electrostatics=lambda_elec,
        lambda_sterics=lambda_sterics,
        lambda_restraints=lambda_restraints,
    )


def _make_solvent_schedule(n_windows: int = 31) -> LambdaSchedule:
    """Generate the solvent-leg lambda schedule.

    2-stage protocol (no restraints in solvent):
      Stage 1: Discharge ligand (12 windows)
      Stage 2: Decouple vdW (19 windows, soft-core)
    """
    n_elec = 12
    n_vdw = n_windows - n_elec  # 19

    lam_elec = [1.0 - i / (n_elec - 1) for i in range(n_elec)]
    lam_vdw = [1.0 - i / (n_vdw - 1) for i in range(n_vdw)]

    lambda_elec = lam_elec + [0.0] * n_vdw
    lambda_sterics = [1.0] * n_elec + lam_vdw
    lambda_restraints = [0.0] * n_windows  # No restraints in solvent

    return LambdaSchedule(
        name="solvent",
        n_windows=n_windows,
        lambda_electrostatics=lambda_elec,
        lambda_sterics=lambda_sterics,
        lambda_restraints=lambda_restraints,
    )


@dataclass
class ABFEProtocol:
    """Complete ABFE protocol specification.

    Attributes:
        compound_id: Unique compound identifier.
        complex_schedule: Lambda schedule for protein-ligand complex.
        solvent_schedule: Lambda schedule for solvated ligand.
        simulation_time_ns: Simulation time per lambda window.
        n_repeats: Number of independent repeats.
        timestep_fs: Integration timestep in femtoseconds.
        temperature_k: Simulation temperature in Kelvin.
        pressure_atm: Simulation pressure in atmospheres.
        use_rest2: Enable REST2 enhanced sampling.
        rest2_scale_factor: REST2 temperature scaling factor.
        soft_core_alpha: Soft-core alpha parameter for vdW.
        equilibration_ns: Equilibration time per window before production.
        protein_pdb: Path to protein PDB.
        ligand_sdf: Path or mol-block of docked ligand.
        output_dir: Output directory for simulation files.
        restraint_info: Boresch restraint parameters.
        network_json: Path to the network spec JSON.
    """
    compound_id: str
    complex_schedule: LambdaSchedule
    solvent_schedule: LambdaSchedule
    simulation_time_ns: float = 5.0
    n_repeats: int = 3
    timestep_fs: float = 4.0
    temperature_k: float = 298.15
    pressure_atm: float = 1.0
    use_rest2: bool = False
    rest2_scale_factor: float = 0.3
    soft_core_alpha: float = 0.5
    equilibration_ns: float = 1.0
    protein_pdb: str = ""
    ligand_sdf: str = ""
    output_dir: str = ""
    restraint_info: Dict[str, Any] = field(default_factory=dict)
    network_json: str = ""

    @property
    def total_windows(self) -> int:
        return self.complex_schedule.n_windows + self.solvent_schedule.n_windows

    @property
    def total_simulation_time_ns(self) -> float:
        return self.total_windows * self.simulation_time_ns * self.n_repeats

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_windows"] = self.total_windows
        d["total_simulation_time_ns"] = self.total_simulation_time_ns
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Optional[str] = None) -> str:
        if path is None:
            path = os.path.join(self.output_dir, f"{self.compound_id}_abfe_protocol.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved ABFE protocol: %s", path)
        return path


def prepare_abfe(
    compound_id: str,
    protein_pdb: str,
    ligand_sdf: str,
    output_dir: str,
    *,
    n_complex_windows: int = 42,
    n_solvent_windows: int = 31,
    simulation_time_ns: float = 5.0,
    n_repeats: int = 3,
    use_rest2: bool = False,
    restraint_info: Optional[Dict[str, Any]] = None,
    network_json: str = "",
) -> ABFEProtocol:
    """Prepare a complete ABFE protocol for one compound.

    Args:
        compound_id: Unique compound identifier.
        protein_pdb: Path to receptor PDB file.
        ligand_sdf: Path to docked ligand SDF or mol-block string.
        output_dir: Directory for output files.
        n_complex_windows: Lambda windows for complex leg.
        n_solvent_windows: Lambda windows for solvent leg.
        simulation_time_ns: Per-window production time (ns).
        n_repeats: Number of independent repeats.
        use_rest2: Enable REST2 enhanced sampling.
        restraint_info: Boresch restraint dict.
        network_json: Path to network spec JSON.

    Returns:
        Fully configured ABFEProtocol.
    """
    complex_schedule = _make_complex_schedule(n_complex_windows)
    solvent_schedule = _make_solvent_schedule(n_solvent_windows)

    protocol = ABFEProtocol(
        compound_id=compound_id,
        complex_schedule=complex_schedule,
        solvent_schedule=solvent_schedule,
        simulation_time_ns=simulation_time_ns,
        n_repeats=n_repeats,
        use_rest2=use_rest2,
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf,
        output_dir=output_dir,
        restraint_info=restraint_info or {},
        network_json=network_json,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    protocol.save()

    logger.info(
        "Prepared ABFE for %s: %d complex + %d solvent windows, "
        "%d repeats, %.1f ns/window = %.0f ns total",
        compound_id,
        complex_schedule.n_windows,
        solvent_schedule.n_windows,
        n_repeats,
        simulation_time_ns,
        protocol.total_simulation_time_ns,
    )

    return protocol


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ABFE protocol for a protein-ligand system",
    )
    parser.add_argument("--protein", required=True, help="Protein PDB path")
    parser.add_argument("--ligand", required=True, help="Ligand SDF path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--compound-id", default="compound_001", help="Compound ID")
    parser.add_argument("--complex-windows", type=int, default=42)
    parser.add_argument("--solvent-windows", type=int, default=31)
    parser.add_argument("--time-ns", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rest2", action="store_true", help="Enable REST2")
    parser.add_argument("--network-json", default="", help="Network spec JSON")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.isfile(args.protein):
        logger.error("Protein PDB not found: %s", args.protein)
        sys.exit(1)
    if not os.path.isfile(args.ligand):
        logger.error("Ligand SDF not found: %s", args.ligand)
        sys.exit(1)

    protocol = prepare_abfe(
        compound_id=args.compound_id,
        protein_pdb=args.protein,
        ligand_sdf=args.ligand,
        output_dir=args.output_dir,
        n_complex_windows=args.complex_windows,
        n_solvent_windows=args.solvent_windows,
        simulation_time_ns=args.time_ns,
        n_repeats=args.repeats,
        use_rest2=args.rest2,
        network_json=args.network_json,
    )

    print(f"ABFE protocol saved to {args.output_dir}")
    print(f"  Total windows: {protocol.total_windows}")
    print(f"  Total simulation time: {protocol.total_simulation_time_ns:.0f} ns")


if __name__ == "__main__":
    main()
