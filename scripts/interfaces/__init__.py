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

V3 gating-stack interfaces:
    ContactReorgResult
    ResponseProfile
    SiteGateDecision, GatingResult

V4 design-layer interfaces:
    AnchorPoint, AnchorPointMap
    GrowthVector, SubPocket, GrowthVectorMap
    PocketProfile
    RankedSite, SiteRanking
    DesignBrief


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

# ── V3 interfaces (gating stack) ──────────────────────────────────────────
from .contact_reorg_result import ContactReorgResult
from .response_profile import ResponseProfile
from .gating_result import SiteGateDecision, GatingResult

# ── V4 interfaces (design layers) ────────────────────────────────────────
from .anchor_point import AnchorPoint, AnchorPointMap, SPIKE_TYPE_TO_INTERACTION
from .growth_vector import GrowthVector, SubPocket, GrowthVectorMap
from .pocket_profile import PocketProfile
from .site_ranking import RankedSite, SiteRanking
from .design_brief import DesignBrief

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
    # ── V3 (gating stack) ──
    # contact_reorg_result
    "ContactReorgResult",
    # response_profile
    "ResponseProfile",
    # gating_result
    "SiteGateDecision",
    "GatingResult",
    # ── V4 (design layers) ──
    # anchor_point
    "AnchorPoint",
    "AnchorPointMap",
    "SPIKE_TYPE_TO_INTERACTION",
    # growth_vector
    "GrowthVector",
    "SubPocket",
    "GrowthVectorMap",
    # pocket_profile
    "PocketProfile",
    # site_ranking
    "RankedSite",
    "SiteRanking",
    # design_brief
    "DesignBrief",
]
