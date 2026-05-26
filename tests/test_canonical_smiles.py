from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest
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
    assert row["canonicalization_detail"] == "OK_RECONSTRUCTED:RXN_AMIDE"
    assert row["canonical_smiles_rdkit"] == "CC(N)=O"


def test_projected_reaction_uses_matching_rule_id(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    rules_path = tmp_path / "reaction_rules.v1.yml"
    rules_path.write_text(
        """
schema_version: reaction_rules.v1
reactions:
  - reaction_id: RXN_WRONG
    enabled: true
    smarts: "[C:1](=[O:2])[O:3].[N:4]>>[C:1](=[O:2])[O:3]"
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
    assert row["canonicalization_detail"] == "OK_RECONSTRUCTED:RXN_AMIDE"
    assert row["canonical_smiles_rdkit"] == "CC(N)=O"


def test_duplicate_reaction_ids_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    rules_path = tmp_path / "reaction_rules.v1.yml"
    rules_path.write_text(
        """
schema_version: reaction_rules.v1
reactions:
  - reaction_id: RXN_AMIDE
    enabled: true
    smarts: "[C:1](=[O:2])[O:3].[N:4]>>[C:1](=[O:2])[O:3]"
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

    with pytest.raises(ValueError, match="duplicate enabled reaction_id: RXN_AMIDE"):
        canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)


def test_non_projected_rows_do_not_reconstruct_from_side_fields(tmp_path: Path) -> None:
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
            "canonical_smiles": ["CCO"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "OK"
    assert row["canonicalization_source"] == "existing_survivor_product"
    assert row["canonical_smiles_rdkit"] == "CCO"


def test_explicit_reaction_id_alone_does_not_reconstruct_side_fields(tmp_path: Path) -> None:
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
            "reaction_id": ["RXN_AMIDE"],
            "canonical_smiles": ["CCO"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "OK"
    assert row["canonicalization_source"] == "existing_survivor_product"
    assert row["canonical_smiles_rdkit"] == "CCO"


def test_reaction_id_mismatch_rejected(tmp_path: Path) -> None:
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
  - reaction_id: RXN_KEEP_ACID
    enabled: true
    smarts: "[C:1](=[O:2])[O:3].[N:4]>>[C:1](=[O:2])[O:3]"
"""
    )
    pl.DataFrame(
        {
            "reaction_id": ["RXN_AMIDE"],
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_KEEP_ACID"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="reaction_id mismatch: RXN_AMIDE != RXN_KEEP_ACID"):
        canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)


def test_projected_reaction_with_multiple_products_fails(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    rules_path = tmp_path / "reaction_rules.v1.yml"
    rules_path.write_text(
        """
schema_version: reaction_rules.v1
reactions:
  - reaction_id: RXN_SPLIT
    enabled: true
    smarts: "[C:1].[O:2]>>[C:1].[O:2]"
"""
    )
    pl.DataFrame(
        {
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_SPLIT"],
            "scaffold_smiles": ["C"],
            "synthon_a_smiles": ["O"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_RECONSTRUCT"
    assert row["canonicalization_source"] == "failed"
    assert row["canonicalization_detail"] == "FAILED_RECONSTRUCT:RXN_SPLIT"
    assert row["canonical_smiles_rdkit"] is None


def test_secondary_projected_marker_cannot_override_primary_smiles(tmp_path: Path) -> None:
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
            "canonical_smiles": ["CCO"],
            "product_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="secondary projected marker conflicts"):
        canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)


def test_secondary_projected_marker_cannot_override_survivor_smiles(tmp_path: Path) -> None:
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
            "survivor_smiles": ["CCO"],
            "product_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="secondary projected marker conflicts with primary survivor_smiles"):
        canonicalize_survivor_parquet(
            input_path,
            output_path,
            reaction_rules=rules_path,
            require_all_ok=True,
            expected_rows=1,
        )


def test_conflicting_projected_markers_rejected_when_primary_is_projected(tmp_path: Path) -> None:
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
  - reaction_id: RXN_KEEP_ACID
    enabled: true
    smarts: "[C:1](=[O:2])[O:3].[N:4]>>[C:1](=[O:2])[O:3]"
"""
    )
    pl.DataFrame(
        {
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE"],
            "product_smiles": ["PROJECTED_RULE_PRODUCT::RXN_KEEP_ACID"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="conflicting projected markers"):
        canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)


def test_missing_projected_reaction_rule_fails(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    rules_path = tmp_path / "reaction_rules.v1.yml"
    rules_path.write_text(
        """
schema_version: reaction_rules.v1
reactions:
  - reaction_id: RXN_WRONG
    enabled: true
    smarts: "[C:1](=[O:2])[O:3].[N:4]>>[C:1](=[O:2])[O:3]"
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

    assert row["canonicalization_status"] == "FAILED_RECONSTRUCT"
    assert row["canonicalization_source"] == "failed"
    assert row["canonicalization_detail"] == "MISSING_REACTION_RULE:RXN_AMIDE"
    assert row["canonical_smiles_rdkit"] is None


def test_projected_marker_with_synthon_suffix_uses_rule_id(tmp_path: Path) -> None:
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
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE::ANCHOR_0001"],
            "synthon_a_id": ["ANCHOR_0001"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "OK"
    assert row["canonicalization_detail"] == "OK_RECONSTRUCTED:RXN_AMIDE"
    assert row["canonical_smiles_rdkit"] == "CC(N)=O"


def test_projected_marker_synthon_suffix_matches_metadata(tmp_path: Path) -> None:
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
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE::SYN_001"],
            "reaction_id": ["RXN_AMIDE"],
            "synthon_a_id": ["SYN_001"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "OK"
    assert row["canonicalization_detail"] == "OK_RECONSTRUCTED:RXN_AMIDE"
    assert row["canonical_smiles_rdkit"] == "CC(N)=O"


@pytest.mark.parametrize(
    "marker",
    [
        "PROJECTED_RULE_PRODUCT::RXN_AMIDE::SYN_EXPECTED::EXTRA",
        "PROJECTED_RULE_PRODUCT::",
        "PROJECTED_RULE_PRODUCT::RXN_AMIDE::",
        "PROJECTED_RULE_PRODUCT::RXN_AMIDE::SYN 001",
    ],
)
def test_malformed_projected_marker_suffixes_rejected(tmp_path: Path, marker: str) -> None:
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
            "canonical_smiles": [marker],
            "reaction_id": ["RXN_AMIDE"],
            "synthon_a_id": ["SYN_EXPECTED"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="malformed projected marker"):
        canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)


def test_projected_marker_synthon_mismatch_rejected(tmp_path: Path) -> None:
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
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE::SYN_EXPECTED"],
            "reaction_id": ["RXN_AMIDE"],
            "synthon_a_id": ["DIFFERENT_SYNTHON"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="projected marker synthon mismatch: SYN_EXPECTED"):
        canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)


def test_projected_marker_synthon_b_mismatch_rejected(tmp_path: Path) -> None:
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
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE::BAD_SYNTHON"],
            "reaction_id": ["RXN_AMIDE"],
            "synthon_b_id": ["GOOD_SYNTHON"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_b_smiles": ["N"],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="projected marker synthon mismatch: BAD_SYNTHON"):
        canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)


def test_duplicate_synthon_ids_across_roles_rejected(tmp_path: Path) -> None:
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
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE::SYN_DUP"],
            "reaction_id": ["RXN_AMIDE"],
            "synthon_a_id": ["SYN_DUP"],
            "synthon_a_smiles": ["N"],
            "synthon_b_id": ["SYN_DUP"],
            "synthon_b_smiles": ["O"],
            "scaffold_smiles": ["CC(=O)O"],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="duplicate synthon id across roles: SYN_DUP"):
        canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)


def test_duplicate_synthon_ids_with_same_smiles_allowed(tmp_path: Path) -> None:
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
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE::SYN_DUP"],
            "reaction_id": ["RXN_AMIDE"],
            "synthon_a_id": ["SYN_DUP"],
            "synthon_a_smiles": ["N"],
            "synthon_b_id": ["SYN_DUP"],
            "synthon_b_smiles": ["N"],
            "scaffold_smiles": ["CC(=O)O"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "OK"
    assert row["canonicalization_source"] == "reaction_smarts"
    assert row["canonical_smiles_rdkit"] == "CC(N)=O"


def test_marker_synthon_suffix_cannot_reconstruct_from_other_role(tmp_path: Path) -> None:
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
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE::SYN_B"],
            "reaction_id": ["RXN_AMIDE"],
            "synthon_a_id": ["SYN_A"],
            "synthon_a_smiles": ["N"],
            "synthon_b_id": ["SYN_B"],
            "synthon_b_smiles": ["O"],
            "scaffold_smiles": ["CC(=O)O"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_RECONSTRUCT"
    assert row["canonicalization_source"] == "failed"
    assert row["canonicalization_detail"] == "FAILED_RECONSTRUCT:RXN_AMIDE"
    assert row["canonical_smiles_rdkit"] is None


def test_projected_marker_missing_synthon_rejected_when_metadata_present(tmp_path: Path) -> None:
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
            "reaction_id": ["RXN_AMIDE"],
            "synthon_a_id": ["SYN_EXPECTED"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="projected marker missing synthon id"):
        canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)


def test_projected_marker_conflict_includes_synthon_suffix(tmp_path: Path) -> None:
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
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE::SYN_A"],
            "product_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE::SYN_B"],
            "reaction_id": ["RXN_AMIDE"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="conflicting projected markers"):
        canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)


def test_disconnected_smiles_not_persisted_as_canonical(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO.O"]}).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_DISCONNECTED"
    assert row["canonicalization_source"] == "failed"
    assert row["canonical_smiles_rdkit"] is None


@pytest.mark.parametrize(
    "bad_smiles",
    [
        "CCO not-a-smiles",
        "CCO .CCN",
        "CCO PROJECTED_RULE_PRODUCT::RXN_AMIDE",
    ],
)
def test_trailing_smiles_name_text_is_rejected(tmp_path: Path, bad_smiles: str) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": [bad_smiles]}).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_PARSE"
    assert row["canonicalization_source"] == "failed"
    assert row["canonical_smiles_rdkit"] is None


@pytest.mark.parametrize(
    "bad_smiles",
    [
        " CCO",
        "CCO ",
        "CCO\nnot-a-smiles",
        "CCO\r\n.O",
        "CCO\tjunk",
        "CCO |atomProp:0.foo.bar|",
        "CCO |atomProp:0.foo.bar| junk",
    ],
)
def test_control_whitespace_and_cxsmiles_are_rejected(tmp_path: Path, bad_smiles: str) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": [bad_smiles]}).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_PARSE"
    assert row["canonicalization_source"] == "failed"
    assert row["canonical_smiles_rdkit"] is None


@pytest.mark.parametrize("bad_smiles", ["CCO~CCN", "*CCO"])
def test_query_like_smiles_are_rejected(tmp_path: Path, bad_smiles: str) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": [bad_smiles]}).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_QUERY"
    assert row["canonicalization_source"] == "failed"
    assert row["canonical_smiles_rdkit"] is None


def test_non_whitespace_control_character_smiles_are_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO\x01"]}).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_PARSE"
    assert row["canonicalization_source"] == "failed"
    assert row["canonical_smiles_rdkit"] is None


def test_reconstruction_rejects_trailing_text_reactants(tmp_path: Path) -> None:
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
            "scaffold_smiles": ["CC(=O)O .Cl"],
            "synthon_a_smiles": ["N junk"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_RECONSTRUCT"
    assert row["canonicalization_source"] == "failed"
    assert row["canonicalization_detail"] == "FAILED_SCAFFOLD_PARSE"
    assert row["canonical_smiles_rdkit"] is None


def test_reconstruction_rejects_disconnected_scaffold_reactants(tmp_path: Path) -> None:
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
            "scaffold_smiles": ["CC(=O)O.Cl"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_RECONSTRUCT"
    assert row["canonicalization_source"] == "failed"
    assert row["canonicalization_detail"] == "FAILED_SCAFFOLD_DISCONNECTED"
    assert row["canonical_smiles_rdkit"] is None


def test_reconstruction_rejects_disconnected_synthon_reactants(tmp_path: Path) -> None:
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
            "synthon_a_smiles": ["N.Cl"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_RECONSTRUCT"
    assert row["canonicalization_source"] == "failed"
    assert row["canonicalization_detail"] == "FAILED_RECONSTRUCT:RXN_AMIDE"
    assert row["canonical_smiles_rdkit"] is None


@pytest.mark.parametrize(
    "hidden_marker",
    [
        " PROJECTED_RULE_PRODUCT::RXN_AMIDE",
        "\tPROJECTED_RULE_PRODUCT::RXN_AMIDE",
        "\u200bPROJECTED_RULE_PRODUCT::RXN_AMIDE",
        "\x01PROJECTED_RULE_PRODUCT::RXN_AMIDE",
        "PROJECTED_RULE\x01_PRODUCT::RXN_AMIDE",
    ],
)
def test_hidden_projected_markers_are_rejected_without_reaction_rules(
    tmp_path: Path, hidden_marker: str
) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame(
        {
            "canonical_smiles": ["CCO"],
            "product_smiles": [hidden_marker],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="malformed projected marker in product_smiles"):
        canonicalize_survivor_parquet(input_path, output_path)


def test_projected_marker_control_character_suffix_rejected(tmp_path: Path) -> None:
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
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_AMIDE::SYN_001\x01"],
            "reaction_id": ["RXN_AMIDE"],
            "synthon_a_id": ["SYN_001"],
            "scaffold_smiles": ["CC(=O)O"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    with pytest.raises(ValueError, match="malformed projected marker"):
        canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)


def test_reconstruction_rejects_newline_reactants(tmp_path: Path) -> None:
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
            "scaffold_smiles": ["CC(=O)O\n.Cl"],
            "synthon_a_smiles": ["N\nnot-a-smiles"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_RECONSTRUCT"
    assert row["canonicalization_source"] == "failed"
    assert row["canonicalization_detail"] == "FAILED_SCAFFOLD_PARSE"
    assert row["canonical_smiles_rdkit"] is None


def test_ambiguous_single_product_reconstruction_fails(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    rules_path = tmp_path / "reaction_rules.v1.yml"
    rules_path.write_text(
        """
schema_version: reaction_rules.v1
reactions:
  - reaction_id: RXN_HALIDE_AMINATION
    enabled: true
    smarts: "[c:1][Br,Cl,I:2].[N:3]>>[c:1]-[N:3]"
"""
    )
    pl.DataFrame(
        {
            "canonical_smiles": ["PROJECTED_RULE_PRODUCT::RXN_HALIDE_AMINATION"],
            "scaffold_smiles": ["Clc1ccc(Br)cc1"],
            "synthon_a_smiles": ["N"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path, reaction_rules=rules_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_RECONSTRUCT"
    assert row["canonicalization_source"] == "failed"
    assert row["canonicalization_detail"] == "AMBIGUOUS_RECONSTRUCT:RXN_HALIDE_AMINATION"
    assert row["canonical_smiles_rdkit"] is None


def test_existing_rdkit_column_is_not_trusted_as_source(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame(
        {
            "canonical_smiles_rdkit": ["CCO"],
            "canonical_smiles": ["not a smiles"],
        }
    ).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path)
    row = df.row(0, named=True)

    assert row["canonicalization_status"] == "FAILED_PARSE"
    assert row["canonicalization_source"] == "failed"
    assert row["canonical_smiles_rdkit"] is None


def test_output_path_cannot_overwrite_input(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO"]}).write_parquet(input_path)

    with pytest.raises(ValueError, match="must not overwrite input parquet"):
        canonicalize_survivor_parquet(input_path, input_path)


def test_tmp_suffix_input_is_preserved(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet.tmp"
    output_path = tmp_path / "survivors.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO"]}).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path)

    assert input_path.exists()
    assert output_path.exists()
    assert df.height == 1
    assert df.row(0, named=True)["canonicalization_status"] == "OK"


def test_empty_input_preserves_canonical_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": pl.Series([], dtype=pl.String)}).write_parquet(input_path)

    df = canonicalize_survivor_parquet(input_path, output_path)

    assert df.height == 0
    assert df.schema["canonical_smiles_rdkit"] == pl.String
    assert df.schema["canonicalization_status"] == pl.String


def test_require_any_ok_rejects_empty_input(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": pl.Series([], dtype=pl.String)}).write_parquet(input_path)

    with pytest.raises(ValueError, match="canonicalization produced zero OK rows"):
        canonicalize_survivor_parquet(input_path, output_path, require_any_ok=True)


def test_negative_limit_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO"]}).write_parquet(input_path)

    with pytest.raises(ValueError, match="limit must be non-negative"):
        canonicalize_survivor_parquet(input_path, output_path, limit=-1)


def test_cli_rejects_all_failed_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": ["not a smiles", "still not smiles"]}).write_parquet(input_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/canonicalize_survivor_smiles.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "canonicalization produced failed rows" in result.stderr
    assert not output_path.exists()


def test_cli_rejects_empty_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": pl.Series([], dtype=pl.String)}).write_parquet(input_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/canonicalize_survivor_smiles.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "canonicalization produced zero rows" in result.stderr
    assert not output_path.exists()


def test_cli_rejects_truncated_top100_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO"]}).write_parquet(input_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/canonicalize_survivor_smiles.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "canonicalization produced 1 rows; expected 100" in result.stderr
    assert not output_path.exists()


def test_cli_expected_rows_override_is_explicit(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO"]}).write_parquet(input_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/canonicalize_survivor_smiles.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--expected-rows",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    df = pl.read_parquet(output_path)
    assert df.height == 1
    assert df.row(0, named=True)["canonicalization_status"] == "OK"


def test_cli_rejects_mixed_failed_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO", "not a smiles"]}).write_parquet(input_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/canonicalize_survivor_smiles.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "canonicalization produced failed rows" in result.stderr
    assert not output_path.exists()


def test_cli_has_no_limit_bypass(tmp_path: Path) -> None:
    input_path = tmp_path / "survivors.parquet"
    output_path = tmp_path / "survivors_canonical.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO", "not a smiles"]}).write_parquet(input_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/canonicalize_survivor_smiles.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--limit",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unrecognized arguments: --limit" in result.stderr
    assert not output_path.exists()


def test_final_top100_artifact_is_canonicalized() -> None:
    artifact = Path(
        "campaigns/glp1r_aleniglipron/track_a_generative/"
        "gflownet_top100_pgx_parity_validated_canonical.parquet"
    )
    if not artifact.is_file():
        pytest.skip("D08 canonical Top100 artifact is not present")

    df = pl.read_parquet(artifact)
    assert df.height == 100
    assert "canonical_smiles_rdkit" in df.columns
    assert "canonicalization_status" in df.columns
    ok = df.filter(pl.col("canonicalization_status") == "OK")
    assert ok.height == 100
    for smiles in ok.get_column("canonical_smiles_rdkit").to_list():
        assert "." not in smiles
        assert Chem.MolFromSmiles(smiles) is not None
