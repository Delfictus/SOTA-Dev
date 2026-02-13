"""PRISM-4D Interface Contracts — stable data types for inter-worktree communication.

All pipeline data flows between worktrees MUST use the dataclasses defined
here.  These interfaces are frozen after merge to ``sota-dev``.  Changes
require a hotfix branch and rebase of all dependent worktrees.

WT-0 interfaces (this package):
    SpikePharmacophore, PharmacophoreFeature, ExclusionSphere
    GeneratedMolecule
    FilteredCandidate
    DockingResult, DockingPose
    FEPResult
    PipelineConfig, DockingConfig, FilterConfig, FEPConfig
    ResidueMapping, ResidueEntry

WT-9 V2 interfaces (additive — no modifications to WT-0 types):
    TautomerState, TautomerEnsemble
    ExplicitSolventResult
    HydrationSite, WaterMap
    EnsembleMMGBSA, InteractionEntropy
    PocketDynamics
    MembraneSystem
    ViewerPayload
"""

from .spike_pharmacophore import (
    ExclusionSphere,
    PharmacophoreFeature,
    SpikePharmacophore,
    SPIKE_TYPE_TO_FEATURE,
)
from .generated_molecule import GeneratedMolecule
from .filtered_candidate import FilteredCandidate
from .docking_result import DockingPose, DockingResult
from .fep_result import FEPResult
from .pipeline_config import (
    DockingConfig,
    FEPConfig,
    FilterConfig,
    PipelineConfig,
)
from .residue_mapping import ResidueEntry, ResidueMapping

# ── V2 interfaces (WT-9) ─────────────────────────────────────────────────
from .tautomer_state import TautomerState, TautomerEnsemble
from .explicit_solvent_result import ExplicitSolventResult
from .water_map import HydrationSite, WaterMap
from .ensemble_score import EnsembleMMGBSA, InteractionEntropy
from .pocket_dynamics import PocketDynamics
from .membrane_system import MembraneSystem
from .viewer_payload import ViewerPayload

__all__ = [
    # spike_pharmacophore
    "PharmacophoreFeature",
    "ExclusionSphere",
    "SpikePharmacophore",
    "SPIKE_TYPE_TO_FEATURE",
    # generated_molecule
    "GeneratedMolecule",
    # filtered_candidate
    "FilteredCandidate",
    # docking_result
    "DockingPose",
    "DockingResult",
    # fep_result
    "FEPResult",
    # pipeline_config
    "DockingConfig",
    "FilterConfig",
    "FEPConfig",
    "PipelineConfig",
    # residue_mapping
    "ResidueEntry",
    "ResidueMapping",
    # ── V2 (WT-9) ──
    # tautomer_state
    "TautomerState",
    "TautomerEnsemble",
    # explicit_solvent_result
    "ExplicitSolventResult",
    # water_map
    "HydrationSite",
    "WaterMap",
    # ensemble_score
    "EnsembleMMGBSA",
    "InteractionEntropy",
    # pocket_dynamics
    "PocketDynamics",
    # membrane_system
    "MembraneSystem",
    # viewer_payload
    "ViewerPayload",
]
