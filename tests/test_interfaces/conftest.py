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

# ── V2 imports (WT-9) ──────────────────────────────────────────────────
from scripts.interfaces.tautomer_state import TautomerState, TautomerEnsemble
from scripts.interfaces.explicit_solvent_result import ExplicitSolventResult
from scripts.interfaces.water_map import HydrationSite, WaterMap
from scripts.interfaces.ensemble_score import EnsembleMMGBSA, InteractionEntropy
from scripts.interfaces.pocket_dynamics import PocketDynamics
from scripts.interfaces.membrane_system import MembraneSystem
from scripts.interfaces.viewer_payload import ViewerPayload


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


# ═══════════════════════════════════════════════════════════════════════════
#  V2 fixtures (WT-9)
# ═══════════════════════════════════════════════════════════════════════════


# ── Tautomer fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_tautomer_neutral() -> TautomerState:
    return TautomerState(
        smiles="CC(=O)O",
        parent_smiles="CC(=O)O",
        protonation_ph=7.4,
        charge=0,
        pka_shifts=[(3, 4.76)],
        population_fraction=0.01,
        source_tool="dimorphite_dl",
    )


@pytest.fixture
def sample_tautomer_anion() -> TautomerState:
    return TautomerState(
        smiles="CC(=O)[O-]",
        parent_smiles="CC(=O)O",
        protonation_ph=7.4,
        charge=-1,
        pka_shifts=[(3, 4.76)],
        population_fraction=0.99,
        source_tool="dimorphite_dl",
    )


@pytest.fixture
def sample_tautomer_ensemble(
    sample_tautomer_neutral, sample_tautomer_anion
) -> TautomerEnsemble:
    return TautomerEnsemble(
        parent_smiles="CC(=O)O",
        states=[sample_tautomer_neutral, sample_tautomer_anion],
        dominant_state=sample_tautomer_anion,
        target_ph=7.4,
        enumeration_method="dimorphite_dl_rdk_mstandardize",
    )


# ── Water map fixtures ────────────────────────────────────────────────

@pytest.fixture
def sample_happy_water() -> HydrationSite:
    return HydrationSite(
        x=10.0, y=12.5, z=8.3,
        occupancy=0.92,
        delta_g_transfer=-2.1,
        entropy_contribution=-1.5,
        enthalpy_contribution=-0.6,
        n_hbonds_mean=2.8,
        classification="CONSERVED_HAPPY",
        displaceable=False,
    )


@pytest.fixture
def sample_unhappy_water() -> HydrationSite:
    return HydrationSite(
        x=11.5, y=14.0, z=7.1,
        occupancy=0.85,
        delta_g_transfer=1.8,
        entropy_contribution=0.5,
        enthalpy_contribution=1.3,
        n_hbonds_mean=1.1,
        classification="CONSERVED_UNHAPPY",
        displaceable=True,
    )


@pytest.fixture
def sample_water_map(sample_happy_water, sample_unhappy_water) -> WaterMap:
    return WaterMap(
        pocket_id=0,
        hydration_sites=[sample_happy_water, sample_unhappy_water],
        n_displaceable=1,
        max_displacement_energy=1.8,
        total_displacement_energy=1.8,
        grid_resolution=0.5,
        analysis_frames=1000,
    )


# ── Explicit solvent fixtures ────────────────────────────────────────

@pytest.fixture
def sample_explicit_solvent_stable(sample_water_map) -> ExplicitSolventResult:
    return ExplicitSolventResult(
        pocket_id=0,
        simulation_time_ns=10.0,
        water_model="TIP3P",
        force_field="ff19SB",
        pocket_stable=True,
        pocket_rmsd_mean=1.2,
        pocket_rmsd_std=0.3,
        pocket_volume_mean=450.0,
        pocket_volume_std=30.0,
        n_structural_waters=4,
        trajectory_path="/data/traj/pocket0.nc",
        snapshot_frames=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        water_map=sample_water_map,
    )


@pytest.fixture
def sample_explicit_solvent_collapsed() -> ExplicitSolventResult:
    return ExplicitSolventResult(
        pocket_id=1,
        simulation_time_ns=10.0,
        water_model="OPC",
        force_field="ff14SB",
        pocket_stable=False,
        pocket_rmsd_mean=4.2,
        pocket_rmsd_std=1.5,
        pocket_volume_mean=180.0,
        pocket_volume_std=95.0,
        n_structural_waters=0,
        trajectory_path="/data/traj/pocket1.nc",
        snapshot_frames=[],
    )


# ── Ensemble score fixtures ──────────────────────────────────────────

@pytest.fixture
def sample_ensemble_mmgbsa() -> EnsembleMMGBSA:
    return EnsembleMMGBSA(
        compound_id="cmpd-001",
        delta_g_mean=-8.2,
        delta_g_std=1.4,
        delta_g_sem=0.14,
        n_snapshots=100,
        snapshot_interval_ps=100.0,
        decomposition={"vdw": -12.0, "elec": -5.0, "gb": 8.0, "sa": -1.2},
        per_residue_contributions={142: -2.5, 145: -1.8, 148: -0.9},
        method="MMGBSA_ensemble",
    )


@pytest.fixture
def sample_interaction_entropy() -> InteractionEntropy:
    return InteractionEntropy(
        compound_id="cmpd-001",
        minus_t_delta_s=-3.2,
        delta_h=-5.0,
        delta_g_ie=-8.2,
        n_frames=100,
        convergence_block_std=0.35,
    )


# ── Pocket dynamics fixtures ────────────────────────────────────────

@pytest.fixture
def sample_pocket_dynamics_stable() -> PocketDynamics:
    return PocketDynamics(
        pocket_id=0,
        p_open=0.72,
        p_open_error=0.06,
        mean_open_lifetime_ns=3.5,
        mean_closed_lifetime_ns=1.2,
        n_opening_events=48,
        druggability_classification="STABLE_OPEN",
        volume_autocorrelation_ns=0.8,
        msm_state_weights={0: 0.6, 1: 0.25, 2: 0.15},
    )


@pytest.fixture
def sample_pocket_dynamics_rare() -> PocketDynamics:
    return PocketDynamics(
        pocket_id=2,
        p_open=0.05,
        p_open_error=0.02,
        mean_open_lifetime_ns=0.3,
        mean_closed_lifetime_ns=8.0,
        n_opening_events=3,
        druggability_classification="RARE_EVENT",
        volume_autocorrelation_ns=5.2,
    )


# ── Membrane system fixtures ────────────────────────────────────────

@pytest.fixture
def sample_membrane_system() -> MembraneSystem:
    return MembraneSystem(
        lipid_composition={"POPC": 0.7, "CHOL": 0.3},
        bilayer_method="packmol_memgen",
        n_lipids=300,
        membrane_thickness=35.0,
        protein_orientation="OPM",
        opm_tilt_angle=12.5,
        system_size=(80.0, 80.0, 110.0),
        total_atoms=85000,
        equilibration_protocol="CHARMM_GUI_6step",
    )


# ── Viewer payload fixtures ─────────────────────────────────────────

@pytest.fixture
def sample_viewer_payload() -> ViewerPayload:
    return ViewerPayload(
        target_name="KRAS_G12C",
        pdb_structure="ATOM      1  CA  ALA A   1      10.000  12.500   8.300  1.00  0.00\nEND\n",
        pocket_surfaces=[
            {
                "vertices": [[10, 12, 8], [11, 13, 9], [12, 14, 10]],
                "triangles": [[0, 1, 2]],
                "color": "rgba(0,100,255,0.4)",
            }
        ],
        spike_positions=[
            {"position": [12.5, -3.2, 8.7], "type": "BNZ", "intensity": 0.85, "residue": "TYR142"},
        ],
        water_map_sites=[
            {"position": [11.5, 14.0, 7.1], "occupancy": 0.85, "delta_g": 1.8,
             "classification": "CONSERVED_UNHAPPY", "color": "red"},
        ],
        ligand_poses=[
            {"smiles": "c1ccc(CC(=O)O)cc1", "mol_block": "mock_sdf",
             "dg_kcal": -8.2, "dg_error": 1.4, "classification": "NOVEL_HIT"},
        ],
        lining_residues=[140, 141, 142, 145, 148],
        p_open=0.72,
        metadata={"qa_score": 0.92, "n_displaceable_waters": 1, "druggability_class": "STABLE_OPEN"},
    )
