from prism_dstw.calibration.translational_calibration_manifold import (
    OBSERVABILITY_SPACE,
    PERTURBATION_SPACE,
    TOPOLOGY_SPACE,
    ManifoldAssignment,
    validate_assignments,
)


def test_manifold_axes_are_complete() -> None:
    assert "TE_HUBS" in TOPOLOGY_SPACE
    assert "HYDRATION_CORRIDOR" in TOPOLOGY_SPACE
    assert "SEVERING_PROBE" in PERTURBATION_SPACE
    assert "transition_chronology" in OBSERVABILITY_SPACE


def test_assignment_requires_all_axes() -> None:
    assignment = ManifoldAssignment(
        variant_id="v1",
        genotype_axis="WT",
        topology_region="TE_HUBS",
        perturbation_family="SEVERING_PROBE",
        observability_channels=("signal_grid", "transition_chronology"),
    )
    validate_assignments([assignment])

