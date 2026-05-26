"""Four-axis translational calibration manifold for Track B."""

from __future__ import annotations

from dataclasses import dataclass

GENOTYPE_SPACE: tuple[str, ...] = (
    "WT",
    "known_campaign_variants",
    "population_variants",
    "lineage_related_variants",
    "species_divergent_residues",
    "conserved_but_dynamically_sensitive_residues",
)

TOPOLOGY_SPACE: tuple[str, ...] = (
    "ECD_TM1_GATEWAY",
    "TM3_TM5_CORE",
    "INTRACELLULAR_LOCK_BASIN",
    "HYDRATION_CORRIDOR",
    "TE_HUBS",
    "BISTATE_RETAINED_RESIDUES",
)

PERTURBATION_SPACE: tuple[str, ...] = (
    "SEVERING_PROBE",
    "RIGIDIFYING_PROBE",
    "FLEXIBILIZING_PROBE",
    "CHARGE_INVERSION_PROBE",
    "HYDRATION_WIRE_PROBE",
    "CONSERVATIVE_CONTROL",
    "VOLUME_EXPANSION_PROBE",
    "AROMATIC_STACKING_PROBE",
)

OBSERVABILITY_SPACE: tuple[str, ...] = (
    "signal_grid",
    "shear_stress",
    "hysteresis",
    "phase_coherence",
    "translation_pathway",
    "hydration_continuity",
    "NMA_continuity",
    "thermodynamic_continuity",
    "transition_chronology",
    "species_conservation",
)

REGION_PURPOSES: dict[str, str] = {
    "ECD_TM1_GATEWAY": "upstream lock perturbation",
    "TM3_TM5_CORE": "propagation continuity",
    "INTRACELLULAR_LOCK_BASIN": "quiet-lock persistence",
    "HYDRATION_CORRIDOR": "solvent continuity",
    "TE_HUBS": "causal rerouting",
    "BISTATE_RETAINED_RESIDUES": "transition continuity",
}


@dataclass(frozen=True)
class ManifoldAssignment:
    variant_id: str
    genotype_axis: str
    topology_region: str
    perturbation_family: str
    observability_channels: tuple[str, ...]

    def validate(self) -> None:
        if self.genotype_axis not in GENOTYPE_SPACE:
            raise ValueError(f"unknown genotype axis: {self.genotype_axis}")
        if self.topology_region not in TOPOLOGY_SPACE:
            raise ValueError(f"unknown topology region: {self.topology_region}")
        if self.perturbation_family not in PERTURBATION_SPACE:
            raise ValueError(f"unknown perturbation family: {self.perturbation_family}")
        if not self.observability_channels:
            raise ValueError(f"{self.variant_id} has no observability channels")
        unknown = [c for c in self.observability_channels if c not in OBSERVABILITY_SPACE]
        if unknown:
            raise ValueError(f"{self.variant_id} has unknown observability channels: {unknown}")


def validate_assignments(assignments: list[ManifoldAssignment]) -> None:
    """Enforce that every variant maps to all four manifold axes."""
    if not assignments:
        raise ValueError("calibration manifold has no variants")
    seen: set[str] = set()
    for assignment in assignments:
        assignment.validate()
        if assignment.variant_id in seen:
            raise ValueError(f"duplicate variant_id: {assignment.variant_id}")
        seen.add(assignment.variant_id)


def default_observability_for_region(region: str) -> tuple[str, ...]:
    """Return non-orphan observability channels for a topology region."""
    base = ["signal_grid", "shear_stress", "phase_coherence", "species_conservation"]
    if region == "TE_HUBS":
        base.extend(["translation_pathway", "transition_chronology"])
    elif region == "HYDRATION_CORRIDOR":
        base.extend(["thermodynamic_continuity"])
    elif region == "INTRACELLULAR_LOCK_BASIN":
        base.extend(["hysteresis", "thermodynamic_continuity"])
    elif region == "BISTATE_RETAINED_RESIDUES":
        base.extend(["hysteresis", "NMA_continuity", "transition_chronology"])
    else:
        base.extend(["hysteresis", "NMA_continuity"])
    return tuple(dict.fromkeys(base))
