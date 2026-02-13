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

WT-9 will add V2 interfaces in separate modules (tautomer_state,
explicit_solvent_result, water_map, ensemble_score, pocket_dynamics,
membrane_system, viewer_payload).  Those are NOT part of this package
until WT-9 merges.
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
]
