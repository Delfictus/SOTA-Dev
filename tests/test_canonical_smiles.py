from __future__ import annotations

from pathlib import Path

import polars as pl
from rdkit import Chem

from scripts.canonicalize_survivor_smiles import canonicalize_survivor_parquet


def test_no_dot_in_canonical(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO", "c1ccccc1"]}).write_parquet(input_path)
    df = canonicalize_survivor_parquet(input_path, output_path)
    valid = df.filter(pl.col("canonicalization_status") == "OK")
    for smiles in valid.get_column("canonical_smiles_rdkit").to_list():
        assert "." not in smiles


def test_canonical_roundtrips(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": ["C(C)O", "NCC(=O)O"]}).write_parquet(input_path)
    df = canonicalize_survivor_parquet(input_path, output_path)
    valid = df.filter(pl.col("canonicalization_status") == "OK")
    for smiles in valid.get_column("canonical_smiles_rdkit").to_list():
        assert Chem.MolFromSmiles(smiles) is not None


def test_reconstructs_from_reaction_smarts(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    rules_path = tmp_path / "reaction_rules.v1.yml"
    rules_path.write_text(
        """
schema_version: reaction_rules.v1
reactions:
  - reaction_id: RXN_AMIDE
    enabled: true
    smarts: "[C:1](=[O:2])[O:3].[N:4]>>[C:1](=[O:2])[N:4]"
"""
    )
    pl.DataFrame(
        {
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)
    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)
    assert row["canonicalization_status"] == "OK"
    assert row["canonicalization_source"] == "reaction_smarts"
    assert row["canonicalization_detail"] == "OK_RECONSTRUCTED"
    assert row["canonical_smiles_rdkit"] == "CC(N)=O"
