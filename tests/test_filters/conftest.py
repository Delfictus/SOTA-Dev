"""Shared fixtures for WT-3 filter tests."""
from __future__ import annotations

import math
import pytest

from scripts.interfaces import (
    GeneratedMolecule,
    FilteredCandidate,
    PharmacophoreFeature,
    ExclusionSphere,
    SpikePharmacophore,
    FilterConfig,
)


# ---------------------------------------------------------------------------
# Valid drug-like SMILES (known to pass RDKit sanitization + drug-likeness)
# ---------------------------------------------------------------------------
VALID_SMILES = [
    "c1ccc(NC(=O)c2ccccc2)cc1",           # benzanilide — simple, drug-like
    "CC(=O)Oc1ccccc1C(O)=O",              # aspirin
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",   # large PAH (will fail drug-likeness)
    "CC(C)Cc1ccc(CC(C)C(O)=O)cc1",        # ibuprofen
    "O=C(O)c1ccccc1O",                     # salicylic acid
    "c1ccc(Oc2ccccc2)cc1",                 # diphenyl ether
    "CC(=O)Nc1ccc(O)cc1",                  # acetaminophen
    "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",      # p-terphenyl
]

INVALID_SMILES = [
    "not_a_molecule",
    "",
    "C(C)(C)(C)(C)(C)",   # pentavalent carbon
    "[invalid]",
]


def _make_mol_block(smiles: str) -> str:
    """Generate a 3D mol block for testing; returns placeholder if rdkit fails."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return ""


@pytest.fixture
def valid_molecules() -> list[GeneratedMolecule]:
    """A list of GeneratedMolecule objects with valid SMILES."""
    mols = []
    for i, smi in enumerate(VALID_SMILES):
        mols.append(
            GeneratedMolecule(
                smiles=smi,
                mol_block=_make_mol_block(smi),
                source="phoregen",
                pharmacophore_match_score=0.8 - i * 0.05,
                matched_features=["AR", "HBD", "HBA"][:max(1, 3 - i % 3)],
                generation_batch_id=f"batch-test-{i:03d}",
            )
        )
    return mols


@pytest.fixture
def invalid_molecules() -> list[GeneratedMolecule]:
    """GeneratedMolecule objects with broken SMILES."""
    return [
        GeneratedMolecule(
            smiles=smi,
            mol_block="",
            source="phoregen",
            pharmacophore_match_score=0.5,
            matched_features=["AR"],
            generation_batch_id="batch-invalid",
        )
        for smi in INVALID_SMILES
    ]


@pytest.fixture
def mixed_molecules(valid_molecules, invalid_molecules) -> list[GeneratedMolecule]:
    """Mix of valid and invalid molecules."""
    return valid_molecules + invalid_molecules


@pytest.fixture
def sample_pharmacophore() -> SpikePharmacophore:
    """A realistic spike pharmacophore with 5 features for testing."""
    features = [
        PharmacophoreFeature(
            feature_type="AR", x=10.0, y=12.0, z=8.0,
            intensity=0.9, source_spike_type="BNZ",
            source_residue_id=142, source_residue_name="PHE142",
            wavelength_nm=280.0, water_density=0.2,
        ),
        PharmacophoreFeature(
            feature_type="HBD", x=12.0, y=14.0, z=9.0,
            intensity=0.85, source_spike_type="TYR",
            source_residue_id=145, source_residue_name="TYR145",
            wavelength_nm=275.0, water_density=0.3,
        ),
        PharmacophoreFeature(
            feature_type="HBA", x=8.0, y=10.0, z=7.0,
            intensity=0.7, source_spike_type="ANION",
            source_residue_id=150, source_residue_name="ASP150",
            wavelength_nm=260.0, water_density=0.4,
        ),
        PharmacophoreFeature(
            feature_type="HY", x=14.0, y=11.0, z=10.0,
            intensity=0.6, source_spike_type="PHE",
            source_residue_id=155, source_residue_name="LEU155",
            wavelength_nm=265.0, water_density=0.1,
        ),
        PharmacophoreFeature(
            feature_type="NI", x=9.0, y=13.0, z=6.0,
            intensity=0.5, source_spike_type="CATION",
            source_residue_id=160, source_residue_name="LYS160",
            wavelength_nm=270.0, water_density=0.5,
        ),
    ]

    exclusion_spheres = [
        ExclusionSphere(x=11.0, y=11.0, z=11.0, radius=2.0, source_atom="CA:ALA148"),
    ]

    return SpikePharmacophore(
        target_name="TEST_TARGET",
        pdb_id="1ABC",
        pocket_id=0,
        features=features,
        exclusion_spheres=exclusion_spheres,
        pocket_centroid=(10.6, 12.0, 8.0),
        pocket_lining_residues=[142, 145, 148, 150, 155, 160],
        prism_run_hash="a" * 64,
    )


@pytest.fixture
def default_filter_config() -> FilterConfig:
    """Default FilterConfig matching blueprint defaults."""
    return FilterConfig(
        qed_min=0.3,
        sa_max=6.0,
        lipinski_max_violations=1,
        pains_reject=True,
        tanimoto_max=0.85,
        cluster_diversity_min=5,
    )
