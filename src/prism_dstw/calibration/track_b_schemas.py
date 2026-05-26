"""Strict Track B production schemas.

The schema objects in this module are intentionally small dataclasses rather
than runtime-only dictionaries. Every production artifact written by the Track B
pipeline carries explicit provenance, source artifacts, evidence paths, creation
time, and schema version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, TypeAlias

ProvenanceClass: TypeAlias = Literal[
    "L5_OBSERVED",
    "L4_RUNTIME_TELEMETRY",
    "L3_DERIVED",
    "L2_PROJECTED",
    "L0_MISSING",
]

PROVENANCE_CLASSES: tuple[str, ...] = (
    "L5_OBSERVED",
    "L4_RUNTIME_TELEMETRY",
    "L3_DERIVED",
    "L2_PROJECTED",
    "L0_MISSING",
)

SCHEMA_VERSION = "track_b.production.v1"


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with timezone."""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ProvenanceRecord:
    """Common provenance envelope required by every Track B schema object."""

    id: str
    provenance_class: ProvenanceClass
    source_artifacts: list[str]
    evidence_paths: list[str]
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InputArtifact(ProvenanceRecord):
    artifact_name: str = ""
    path: str = ""
    exists: bool = False
    row_count: int | None = None
    schema: dict[str, str] = field(default_factory=dict)
    sha256: str | None = None
    size_bytes: int | None = None
    usable_for: list[str] = field(default_factory=list)
    blocking_if_missing: bool = False
    fallback_allowed: bool = False
    fallback_rule: str | None = None


@dataclass(frozen=True)
class TopologyRegion(ProvenanceRecord):
    region: str = ""
    purpose: str = ""
    residue_ids: list[str] = field(default_factory=list)
    evidence_columns: list[str] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    status: str = "COVERED"


@dataclass(frozen=True)
class VariantFamily(ProvenanceRecord):
    family: str = ""
    description: str = ""
    perturbation_types: list[str] = field(default_factory=list)
    selection_features: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Perturbation(ProvenanceRecord):
    variant_id: str = ""
    residue_id: str = ""
    topology_region: str = ""
    perturbation_family: str = ""
    genotype_axis: str = ""
    observability_channels: list[str] = field(default_factory=list)
    selection_features: list[str] = field(default_factory=list)
    projected_mutation: str = ""


@dataclass(frozen=True)
class CoverageRow(ProvenanceRecord):
    genotype_space: str = ""
    topology_space: str = ""
    perturbation_space: str = ""
    observability_space: str = ""
    variant_id: str = ""
    covered: bool = False
    coverage_score: float = 0.0


@dataclass(frozen=True)
class AdmissibilityResult(ProvenanceRecord):
    verdict: Literal[
        "CALIBRATION_MANIFOLD_ADEQUATE",
        "CALIBRATION_MANIFOLD_PARTIAL_BLOCKED_WITH_EVIDENCE",
        "CALIBRATION_MANIFOLD_REJECTED_TOO_SPARSE",
    ] = "CALIBRATION_MANIFOLD_REJECTED_TOO_SPARSE"
    passed_rules: list[str] = field(default_factory=list)
    failed_rules: list[str] = field(default_factory=list)
    blocked_rules: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TransitionEvent(ProvenanceRecord):
    true_md_step: int = 0
    time_ps: float = 0.0
    condition: str = ""
    replica: int = 0
    stream: int = 0
    residues: list[int] = field(default_factory=list)
    voxel_idx: int | None = None
    event_type: str = ""
    temporal_overlap_entropy: float = 0.0
    event_source: str = ""
    confidence: float = 0.0
    upstream_artifact: str = ""


@dataclass(frozen=True)
class ContinuityMapRecord(ProvenanceRecord):
    map_type: Literal["NMA", "HYDRATION", "THERMODYNAMIC"] = "THERMODYNAMIC"
    residue_id: str | None = None
    voxel_idx: int | None = None
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)
    blocked_with_hard_evidence: bool = False


@dataclass(frozen=True)
class ChronologyTrainingRecord(ProvenanceRecord):
    epoch: int = 0
    tb_loss: float = 0.0
    reward_mean: float = 0.0
    chronology_multiplier_mean: float = 0.0
    continuity_admissibility_rate: float = 0.0
    target_window_start: int = 0
    target_window_end: int = 0
    unique_smiles: int = 0
    dot_smiles_count: int = 0


@dataclass(frozen=True)
class TrackBManifest(ProvenanceRecord):
    product_root: str = ""
    calibration_manifold: str | None = None
    variant_panel: str | None = None
    coverage_matrix: str | None = None
    adequacy_gate: str | None = None
    transition_tensor: str | None = None
    continuity_maps: list[str] = field(default_factory=list)
    candidate_audit: str | None = None
    dossier: str | None = None


@dataclass(frozen=True)
class DeploymentBundleManifest(ProvenanceRecord):
    runtime_root: str = ""
    artifact_manifest: str = ""
    cloud_sync_manifest: str = ""
    vectorize_manifest: str = ""
    release_package: str | None = None
    release_sha256: str | None = None
