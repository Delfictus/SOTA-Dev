"""Persistent parquet-backed thermodynamic motif registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias, get_args

import numpy as np
import polars as pl

DiscoveryMethod: TypeAlias = Literal["TFGD", "CAME", "PR_MCS", "SAD"]
ThermodynamicRole: TypeAlias = Literal[
    "COMPLEMENT_ANCHOR",
    "CLASH_DRIVER",
    "LOCK_WEDGE",
    "SHEAR_SENTINEL",
    "PHASE_PIVOT",
    "NEUTRAL",
    "MIXED",
]
ConfidenceTag: TypeAlias = Literal["L1", "L2", "L3", "L4", "L5"]
ProvenanceTag: TypeAlias = Literal["OBSERVED", "DERIVED", "PROJECTED"]

OPTIONAL_FIELD_NAMES: tuple[str, ...] = (
    "pi_complement_contribution",
    "pi_clash_contribution",
    "lock_geometry_contribution",
    "shear_stress_mean",
    "phase_profile",
    "hysteresis_score",
    "consensus_resilience",
    "variant_resilience",
    "attribution_score_mean",
    "synthon_sources",
    "exit_vector_preference",
)

JSON_FIELD_NAMES: tuple[str, ...] = (
    "phase_profile",
    "variant_resilience",
    "synthon_sources",
    "reaction_classes",
    "exit_vector_preference",
    "parent_smiles",
)


@dataclass(frozen=True)
class MotifEntry:
    """One progressively enriched thermodynamic motif."""

    motif_id: str
    canonical_smarts: str
    discovery_method: DiscoveryMethod
    thermodynamic_role: ThermodynamicRole
    pi_complement_contribution: float | None = None
    pi_clash_contribution: float | None = None
    lock_geometry_contribution: float | None = None
    shear_stress_mean: float | None = None
    phase_profile: list[float] | None = None
    hysteresis_score: float | None = None
    consensus_resilience: float | None = None
    variant_resilience: dict[str, float] | None = None
    worst_case_variant: str | None = None
    worst_case_resilience: float | None = None
    is_evolutionary_invariant: bool | None = None
    attribution_score_mean: float | None = None
    attribution_score_std: float | None = None
    causal_direction: str | None = None
    channel_a_ratio: float | None = None
    channel_b_ratio: float | None = None
    synthon_sources: list[str] | None = None
    reaction_classes: list[str] | None = None
    synthetic_accessibility: float | None = None
    exit_vector_preference: dict[int, float] | None = None
    n_occurrences_top100: int = 0
    n_occurrences_lock_positive: int = 0
    enrichment_ratio: float | None = None
    first_seen_epoch: int = 24
    last_seen_epoch: int = 24
    confidence: ConfidenceTag = "L3"
    provenance: ProvenanceTag = "DERIVED"
    completeness_score: float = 0.0
    parent_smiles: list[str] | None = None

    def compute_completeness(self) -> float:
        """Return the fraction of optional annotations currently populated."""

        values = [getattr(self, name) for name in OPTIONAL_FIELD_NAMES]
        populated = sum(1 for value in values if value is not None)
        return populated / float(len(values))

    def with_completeness(self) -> "MotifEntry":
        """Return an entry with refreshed completeness."""

        return replace(self, completeness_score=self.compute_completeness())

    def to_storage_dict(self) -> dict[str, Any]:
        """Serialize to a parquet-friendly row."""

        row = asdict(self.with_completeness())
        for name in JSON_FIELD_NAMES:
            row[name] = json.dumps(row[name], sort_keys=True) if row[name] is not None else None
        return row

    @classmethod
    def from_storage_dict(cls, row: dict[str, Any]) -> "MotifEntry":
        """Deserialize from a parquet row."""

        payload = dict(row)
        for name in JSON_FIELD_NAMES:
            raw = payload.get(name)
            payload[name] = json.loads(raw) if isinstance(raw, str) else None
        ev_pref = payload.get("exit_vector_preference")
        if isinstance(ev_pref, dict):
            payload["exit_vector_preference"] = {int(k): float(v) for k, v in ev_pref.items()}
        valid_names = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in payload.items() if key in valid_names}
        return cls(**filtered).with_completeness()


@dataclass(frozen=True)
class MotifDiff:
    """Registry diff between two epochs."""

    added: list[str]
    removed: list[str]
    persistent: list[str]


def motif_id_for_smarts(smarts: str) -> str:
    """Return a stable motif ID from canonical SMARTS."""

    digest = hashlib.sha256(smarts.encode("utf-8")).hexdigest()
    return f"motif_{digest[:16]}"


class MotifRegistry:
    """Persistent, versioned, queryable motif registry."""

    def __init__(self, parquet_path: str | Path, d1_url: str | None = None) -> None:
        self.parquet_path = Path(parquet_path)
        self.d1_url = d1_url

    def all(self) -> list[MotifEntry]:
        """Load all entries."""

        if not self.parquet_path.exists():
            return []
        rows = pl.read_parquet(self.parquet_path).to_dicts()
        return [MotifEntry.from_storage_dict(row) for row in rows]

    def register(self, entry: MotifEntry) -> str:
        """Add or update a motif by motif_id."""

        refreshed = entry.with_completeness()
        entries = {item.motif_id: item for item in self.all()}
        entries[refreshed.motif_id] = refreshed
        self._write_entries(list(entries.values()))
        return refreshed.motif_id

    def enrich(self, motif_id: str, **kwargs: Any) -> None:
        """Update specific fields of an existing motif entry."""

        entries = {item.motif_id: item for item in self.all()}
        if motif_id not in entries:
            raise KeyError(f"unknown motif_id: {motif_id}")
        allowed = {field.name for field in fields(MotifEntry)}
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise ValueError(f"unknown MotifEntry fields: {unknown}")
        entries[motif_id] = replace(entries[motif_id], **kwargs).with_completeness()
        self._write_entries(list(entries.values()))

    def query_by_role(self, role: str) -> list[MotifEntry]:
        """Return motifs with a thermodynamic role."""

        if role not in get_args(ThermodynamicRole):
            raise ValueError(f"unknown thermodynamic role: {role}")
        return [entry for entry in self.all() if entry.thermodynamic_role == role]

    def query_by_phase_profile(self, target: np.ndarray, tolerance: float) -> list[MotifEntry]:
        """Return motifs whose phase profile is near target."""

        if target.shape != (5,):
            raise ValueError("phase-profile target must have shape [5]")
        matches: list[MotifEntry] = []
        for entry in self.all():
            if entry.phase_profile is None:
                continue
            profile = np.array(entry.phase_profile, dtype=np.float64)
            if profile.shape == (5,) and float(np.linalg.norm(profile - target)) <= tolerance:
                matches.append(entry)
        return matches

    def query_variant_resilient(self, min_resilience: float = 0.85) -> list[MotifEntry]:
        """Return motifs whose worst-case variant resilience clears threshold."""

        return [
            entry
            for entry in self.all()
            if entry.worst_case_resilience is not None and entry.worst_case_resilience >= min_resilience
        ]

    def query_lock_enriched(self, min_enrichment: float = 2.0) -> list[MotifEntry]:
        """Return lock-enriched motifs."""

        return [
            entry
            for entry in self.all()
            if entry.enrichment_ratio is not None and entry.enrichment_ratio >= min_enrichment
        ]

    def diff(self, epoch_a: int, epoch_b: int) -> MotifDiff:
        """Compare motif presence across two epochs."""

        a = {
            entry.motif_id
            for entry in self.all()
            if entry.first_seen_epoch <= epoch_a <= entry.last_seen_epoch
        }
        b = {
            entry.motif_id
            for entry in self.all()
            if entry.first_seen_epoch <= epoch_b <= entry.last_seen_epoch
        }
        return MotifDiff(added=sorted(b - a), removed=sorted(a - b), persistent=sorted(a & b))

    def export_for_vectorize(self) -> list[dict[str, Any]]:
        """Export registry metadata for a Vectorize similarity index."""

        rows: list[dict[str, Any]] = []
        for entry in self.all():
            vector = [
                float(entry.pi_complement_contribution or 0.0),
                float(entry.pi_clash_contribution or 0.0),
                float(entry.lock_geometry_contribution or 0.0),
                float(entry.shear_stress_mean or 0.0),
                float(entry.hysteresis_score or 0.0),
                float(entry.consensus_resilience or 0.0),
                float(entry.enrichment_ratio or 0.0),
                float(entry.completeness_score),
            ]
            rows.append(
                {
                    "id": entry.motif_id,
                    "canonical_smarts": entry.canonical_smarts,
                    "metadata": {
                        "role": entry.thermodynamic_role,
                        "method": entry.discovery_method,
                        "confidence": entry.confidence,
                        "provenance": entry.provenance,
                    },
                    "values": vector,
                }
            )
        return rows

    def _write_entries(self, entries: list[MotifEntry]) -> None:
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [entry.to_storage_dict() for entry in sorted(entries, key=lambda item: item.motif_id)]
        pl.DataFrame(rows).write_parquet(self.parquet_path)
