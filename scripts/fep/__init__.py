"""WT-2: OpenFE ABFE/RBFE Pipeline.

Provides the alchemical free-energy perturbation infrastructure for the
PRISM-4D pipeline:
    - prism_to_openfe: Bridge from PRISM DockingResult → OpenFE systems
    - prepare_abfe:    ABFE lambda-schedule configuration
    - prepare_rbfe:    RBFE star-map network setup
    - run_fep:         Execution engine (local GPU / SLURM)
    - analyze_fep:     Result gathering, corrections, classification
    - fep_qc:          Quality-control gate implementations
    - restraint_selector: Boresch restraint atom selection
"""

from .fep_qc import (
    FEPQCReport,
    QCGateResult,
    check_hysteresis,
    check_ligand_in_pocket,
    check_overlap,
    check_protein_rmsd,
    check_repeat_convergence,
    run_all_qc_gates,
)
from .prepare_abfe import ABFEProtocol, LambdaSchedule, prepare_abfe
from .prepare_rbfe import AtomMapping, RBFEEdge, RBFEProtocol, prepare_rbfe
from .prism_to_openfe import (
    AlchemicalNetworkSpec,
    ChemicalSystemSpec,
    build_network_from_prism,
    build_rbfe_network,
)
from .restraint_selector import (
    AtomInfo,
    BoreshRestraint,
    score_restraint,
    select_boresch_restraint,
)
from .run_fep import (
    ExecutionBackend,
    FEPExecutionPlan,
    SimulationJob,
    build_execution_plan,
    execute_plan,
)

__all__ = [
    # fep_qc
    "QCGateResult",
    "FEPQCReport",
    "check_overlap",
    "check_hysteresis",
    "check_protein_rmsd",
    "check_ligand_in_pocket",
    "check_repeat_convergence",
    "run_all_qc_gates",
    # prepare_abfe
    "LambdaSchedule",
    "ABFEProtocol",
    "prepare_abfe",
    # prepare_rbfe
    "AtomMapping",
    "RBFEEdge",
    "RBFEProtocol",
    "prepare_rbfe",
    # prism_to_openfe
    "ChemicalSystemSpec",
    "AlchemicalNetworkSpec",
    "build_network_from_prism",
    "build_rbfe_network",
    # restraint_selector
    "AtomInfo",
    "BoreshRestraint",
    "score_restraint",
    "select_boresch_restraint",
    # run_fep
    "ExecutionBackend",
    "SimulationJob",
    "FEPExecutionPlan",
    "build_execution_plan",
    "execute_plan",
]
