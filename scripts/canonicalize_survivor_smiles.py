#!/usr/bin/env python3
"""Add RDKit-canonical product SMILES columns to a survivor parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
import yaml


def canonicalize_smiles(value: object) -> tuple[str | None, str]:
    """Return RDKit canonical SMILES and a status tag."""

    if not isinstance(value, str) or not value.strip():
        return None, "FAILED_EMPTY"
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None, "FAILED_PARSE"
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "FAILED_SANITIZE"
    canonical = Chem.MolToSmiles(mol, canonical=True)
    if "." in canonical:
        return canonical, "FAILED_DISCONNECTED"
    return canonical, "OK"


def load_reaction_smarts(path: Path | None) -> list[Any]:
    """Load enabled RDKit reactions from the PRISM reaction-rule registry."""

    if path is None:
        return []
    payload = yaml.safe_load(path.read_text())
    reactions: list[Any] = []
    for entry in payload.get("reactions", []) if isinstance(payload, dict) else []:
        if not bool(entry.get("enabled", True)):
            continue
        smarts = entry.get("smarts")
        if not isinstance(smarts, str) or not smarts:
            continue
        reaction_from_smarts: Any = getattr(AllChem, "ReactionFromSmarts")
        reaction = reaction_from_smarts(smarts)
        if reaction is not None:
            reactions.append(reaction)
    return reactions


def _first_string(row: Mapping[str, Any], columns: tuple[str, ...]) -> str | None:
    for column in columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip() and not value.startswith("PROJECTED_RULE_PRODUCT::"):
            return value
    return None


def reconstruct_product_smiles(row: Mapping[str, Any], reactions: list[Any]) -> tuple[str | None, str]:
    """Try scaffold + synthon SMARTS reconstruction before falling back to stored products."""

    if not reactions:
        return None, "NO_REACTION_RULES"
    scaffold = _first_string(
        row,
        (
            "scaffold_smiles",
            "scaffold_canonical_smiles",
            "parent_scaffold_smiles",
            "parent_smiles",
        ),
    )
    synthons = [
        value
        for value in (
            _first_string(row, ("synthon_smiles", "synthon_a_smiles", "synthon_a_canonical_smiles")),
            _first_string(row, ("synthon_b_smiles", "synthon_b_canonical_smiles")),
        )
        if value is not None
    ]
    if scaffold is None or not synthons:
        return None, "MISSING_REACTANTS"
    scaffold_mol = Chem.MolFromSmiles(scaffold)
    if scaffold_mol is None:
        return None, "FAILED_SCAFFOLD_PARSE"
    for synthon in synthons:
        synthon_mol = Chem.MolFromSmiles(synthon)
        if synthon_mol is None:
            continue
        for reaction in reactions:
            for reactants in ((scaffold_mol, synthon_mol), (synthon_mol, scaffold_mol)):
                try:
                    product_sets = reaction.RunReactants(reactants)
                except Exception:
                    continue
                for product_set in product_sets:
                    if not product_set:
                        continue
                    product = product_set[0]
                    try:
                        Chem.SanitizeMol(product)
                    except Exception:
                        continue
                    canonical = Chem.MolToSmiles(product, canonical=True)
                    if canonical and "." not in canonical:
                        return canonical, "OK_RECONSTRUCTED"
    return None, "FAILED_RECONSTRUCT"


def choose_source_smiles(row: Mapping[str, Any]) -> object:
    """Pick the best available product-like SMILES from a survivor row."""

    for column in (
        "canonical_smiles_rdkit",
        "canonical_smiles",
        "survivor_smiles",
        "product_smiles",
        "smiles",
    ):
        value = row.get(column)
        if isinstance(value, str) and value and not value.startswith("PROJECTED_RULE_PRODUCT::"):
            return value
    return row.get("canonical_smiles")


def canonicalize_survivor_parquet(
    input_path: Path,
    output_path: Path,
    *,
    reaction_rules: Path | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Canonicalize survivor rows and write an enriched parquet."""

    df = pl.read_parquet(input_path)
    rows = df.head(limit).to_dicts() if limit is not None else df.to_dicts()
    reactions = load_reaction_smarts(reaction_rules)
    canonical_values: list[str | None] = []
    statuses: list[str] = []
    sources: list[str] = []
    details: list[str] = []
    for row in rows:
        canonical, detail = reconstruct_product_smiles(row, reactions)
        if canonical is not None:
            status = "OK"
            source = "reaction_smarts"
        else:
            canonical, status = canonicalize_smiles(choose_source_smiles(row))
            source = "existing_survivor_product" if status == "OK" else "failed"
            detail = detail if status == "OK" else f"{detail};{status}"
        canonical_values.append(canonical)
        statuses.append(status)
        sources.append(source)
        details.append(detail)
    output = df.head(limit) if limit is not None else df
    output = output.with_columns(
        pl.Series("canonical_smiles_rdkit", canonical_values),
        pl.Series("canonicalization_status", statuses),
        pl.Series("canonicalization_source", sources),
        pl.Series("canonicalization_detail", details),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output.write_parquet(tmp_path)
    tmp_path.replace(output_path)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reaction-rules", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.reaction_rules is not None and not args.reaction_rules.exists():
        raise FileNotFoundError(args.reaction_rules)
    output = canonicalize_survivor_parquet(
        args.input,
        args.output,
        reaction_rules=args.reaction_rules,
        limit=args.limit,
    )
    ok = output.filter(pl.col("canonicalization_status") == "OK").height
    reconstructed = output.filter(pl.col("canonicalization_source") == "reaction_smarts").height
    print(
        "canonicalize_survivor_smiles_complete "
        f"rows={output.height} ok={ok} reconstructed={reconstructed} "
        f"failed={output.height - ok} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
