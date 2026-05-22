"""Ontological type wrappers for PRISM-DSTW computational boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType


CampaignId = NewType("CampaignId", str)
RunLabel = NewType("RunLabel", str)
StructureId = NewType("StructureId", str)
StreamId = NewType("StreamId", int)
ScopeType = NewType("ScopeType", str)
ScopeId = NewType("ScopeId", str)

ResidueIdx = NewType("ResidueIdx", int)
EdgeIdx = NewType("EdgeIdx", int)
ConformerIdx = NewType("ConformerIdx", int)
AnalogIdx = NewType("AnalogIdx", int)
AnalogId = NewType("AnalogId", str)
MotifClass = NewType("MotifClass", str)
ScaffoldHash = NewType("ScaffoldHash", str)
ConformerEnsembleURI = NewType("ConformerEnsembleURI", str)
ScaffoldIdx = NewType("ScaffoldIdx", int)
VoxelIdx = NewType("VoxelIdx", int)

CausalCoupling = NewType("CausalCoupling", float)
HysteresisEnthalpy = NewType("HysteresisEnthalpy", float)
HydrationVariance = NewType("HydrationVariance", float)
SpatialVariance = NewType("SpatialVariance", float)
FrustrationPenalty = NewType("FrustrationPenalty", float)
ComplementPenalty = NewType("ComplementPenalty", float)
PoseUncertainty = NewType("PoseUncertainty", float)
ScalingConstant = NewType("ScalingConstant", float)
AngstromDistance = NewType("AngstromDistance", float)


@dataclass(frozen=True)
class PartitionKey:
    campaign_id: CampaignId
    run_label: RunLabel
    structure_id: StructureId
    stream_id: StreamId
    scope_type: ScopeType
    scope_id: ScopeId


@dataclass(frozen=True)
class AnalogIntakeRecord:
    analog_id: AnalogId
    canonical_smiles: str
    motif_class: MotifClass
    scaffold_hash: ScaffoldHash
    conformer_ensemble_uri: ConformerEnsembleURI
    conformer_count: ConformerIdx
    charge_method: str


@dataclass(frozen=True)
class DTSGEdge:
    from_residue: ResidueIdx
    to_residue: ResidueIdx
    te_out: CausalCoupling
    te_in: CausalCoupling
    delta_hc: HysteresisEnthalpy
    sigma_hyd: HydrationVariance
    spatial_var: SpatialVariance
    hysteresis_persist: float


@dataclass(frozen=True)
class PerturbedEdge:
    edge: DTSGEdge
    te_out_perturbed: CausalCoupling
    te_in_perturbed: CausalCoupling
    delta_hc_perturbed: HysteresisEnthalpy
    u_pose_te: PoseUncertainty
    u_pose_hc: PoseUncertainty
