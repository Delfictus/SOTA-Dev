"""Shared mock data fixtures for interface tests."""
from __future__ import annotations

import pytest

from scripts.interfaces.spike_pharmacophore import (
    ExclusionSphere,
    PharmacophoreFeature,
    SpikePharmacophore,
)
from scripts.interfaces.generated_molecule import GeneratedMolecule
from scripts.interfaces.filtered_candidate import FilteredCandidate
from scripts.interfaces.docking_result import DockingPose, DockingResult
from scripts.interfaces.fep_result import FEPResult
from scripts.interfaces.pipeline_config import (
    DockingConfig,
    FEPConfig,
    FilterConfig,
    PipelineConfig,
)
from scripts.interfaces.residue_mapping import ResidueEntry, ResidueMapping


# ── Pharmacophore fixtures ───────────────────────────────────────────────

@pytest.fixture
def sample_feature() -> PharmacophoreFeature:
    return PharmacophoreFeature(
        feature_type="AR",
        x=12.5,
        y=-3.2,
        z=8.7,
        intensity=0.85,
        source_spike_type="BNZ",
        source_residue_id=142,
        source_residue_name="TYR142",
        wavelength_nm=258.0,
        water_density=0.32,
    )


@pytest.fixture
def sample_feature_2() -> PharmacophoreFeature:
    return PharmacophoreFeature(
        feature_type="NI",
        x=15.5,
        y=-1.2,
        z=10.7,
        intensity=0.60,
        source_spike_type="CATION",
        source_residue_id=145,
        source_residue_name="LYS145",
        wavelength_nm=0.0,
        water_density=0.55,
    )


@pytest.fixture
def sample_exclusion() -> ExclusionSphere:
    return ExclusionSphere(
        x=14.0, y=-2.0, z=9.0, radius=2.0, source_atom="CA:ALA145"
    )


@pytest.fixture
def sample_pharmacophore(
    sample_feature, sample_feature_2, sample_exclusion
) -> SpikePharmacophore:
    return SpikePharmacophore(
        target_name="KRAS_G12C",
        pdb_id="6GJ8",
        pocket_id=0,
        features=[sample_feature, sample_feature_2],
        exclusion_spheres=[sample_exclusion],
        pocket_centroid=(13.5, -2.0, 9.5),
        pocket_lining_residues=[140, 141, 142, 145, 148],
        prism_run_hash="a1b2c3d4e5f6",
        creation_timestamp="2026-02-13T00:00:00+00:00",
    )


# ── Molecule fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_molecule() -> GeneratedMolecule:
    return GeneratedMolecule(
        smiles="c1ccc(CC(=O)O)cc1",
        mol_block=(
            "\n     RDKit          3D\n\n"
            " 11 11  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "M  END\n"
        ),
        source="phoregen",
        pharmacophore_match_score=0.80,
        matched_features=["AR", "NI"],
        generation_batch_id="batch-test-001",
        generation_timestamp="2026-02-13T00:00:00+00:00",
    )


# ── Filtered candidate fixtures ─────────────────────────────────────────

@pytest.fixture
def sample_candidate(sample_molecule) -> FilteredCandidate:
    return FilteredCandidate(
        molecule=sample_molecule,
        qed_score=0.72,
        sa_score=3.1,
        lipinski_violations=0,
        pains_alerts=[],
        tanimoto_to_nearest_known=0.42,
        nearest_known_cid="2244",
        cluster_id=3,
        passed_all_filters=True,
    )


@pytest.fixture
def sample_rejected_candidate(sample_molecule) -> FilteredCandidate:
    return FilteredCandidate(
        molecule=sample_molecule,
        qed_score=0.15,
        sa_score=8.5,
        lipinski_violations=3,
        pains_alerts=["catechol_A(92)"],
        tanimoto_to_nearest_known=0.91,
        nearest_known_cid="5090",
        cluster_id=0,
        passed_all_filters=False,
        rejection_reason="QED < 0.3; SA > 6.0; Lipinski > 1; PAINS hit; Tanimoto > 0.85",
    )


# ── Docking fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_pose() -> DockingPose:
    return DockingPose(
        pose_rank=1,
        mol_block="mock_sdf_block",
        vina_score=-8.5,
        cnn_score=0.92,
        cnn_affinity=-9.3,
        rmsd_lb=0.0,
        rmsd_ub=0.0,
    )


@pytest.fixture
def sample_docking_result(sample_pose) -> DockingResult:
    pose2 = DockingPose(
        pose_rank=2, mol_block="mock_sdf_2", vina_score=-7.8,
        cnn_score=0.85, cnn_affinity=-8.1,
    )
    return DockingResult(
        compound_id="cmpd-001",
        smiles="c1ccc(CC(=O)O)cc1",
        site_id=0,
        receptor_pdb="/data/receptor.pdb",
        poses=[sample_pose, pose2],
        best_vina_score=-8.5,
        best_cnn_affinity=-9.3,
        docking_engine="unidock+gnina",
        box_center=(13.5, -2.0, 9.5),
        box_size=(25.0, 25.0, 25.0),
        exhaustiveness=32,
        docking_timestamp="2026-02-13T00:00:00+00:00",
    )


# ── FEP fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_fep_pass() -> FEPResult:
    return FEPResult(
        compound_id="cmpd-001",
        delta_g_bind=-7.2,
        delta_g_error=0.45,
        method="ABFE",
        n_repeats=3,
        convergence_passed=True,
        hysteresis_kcal=0.3,
        overlap_minimum=0.15,
        max_protein_rmsd=1.2,
        restraint_correction=-0.4,
        charge_correction=0.1,
        vina_score_deprecated=-8.5,
        spike_pharmacophore_match="4/5 features within 2.0A",
        classification="NOVEL_HIT",
        raw_data_path="/data/fep/cmpd-001",
        fep_timestamp="2026-02-13T00:00:00+00:00",
    )


@pytest.fixture
def sample_fep_fail() -> FEPResult:
    return FEPResult(
        compound_id="cmpd-bad",
        delta_g_bind=-2.1,
        delta_g_error=1.8,
        method="ABFE",
        n_repeats=3,
        convergence_passed=False,
        hysteresis_kcal=2.5,
        overlap_minimum=0.01,
        max_protein_rmsd=4.5,
        restraint_correction=-0.3,
        charge_correction=0.05,
        vina_score_deprecated=-4.0,
        spike_pharmacophore_match="1/5 features within 2.0A",
        classification="FAILED_QC",
        raw_data_path="/data/fep/cmpd-bad",
    )


# ── Config fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_config() -> PipelineConfig:
    return PipelineConfig(
        project_name="KRAS-G12C Screen",
        target_name="KRAS_G12C",
        pdb_id="6GJ8",
        receptor_pdb="/data/6GJ8_prepped.pdb",
        topology_json="/data/6GJ8.topology.json",
        binding_sites_json="/data/6GJ8.binding_sites.json",
        output_dir="/output/kras_g12c",
    )


# ── Residue mapping fixtures ────────────────────────────────────────────

@pytest.fixture
def sample_residue_mapping() -> ResidueMapping:
    entries = [
        ResidueEntry(
            topology_id=0, pdb_resid=1, pdb_chain="A",
            pdb_insertion_code="", uniprot_position=1, residue_name="MET",
        ),
        ResidueEntry(
            topology_id=1, pdb_resid=2, pdb_chain="A",
            pdb_insertion_code="", uniprot_position=2, residue_name="THR",
        ),
        ResidueEntry(
            topology_id=2, pdb_resid=3, pdb_chain="A",
            pdb_insertion_code="", uniprot_position=3, residue_name="GLU",
        ),
        ResidueEntry(
            topology_id=3, pdb_resid=27, pdb_chain="A",
            pdb_insertion_code="A", uniprot_position=27, residue_name="ALA",
        ),
    ]
    return ResidueMapping(
        pdb_id="6GJ8",
        uniprot_id="P01116",
        chains=["A"],
        entries=entries,
        topology_residue_count=169,
        pdb_residue_count=169,
        mapping_source="SIFTS",
    )
