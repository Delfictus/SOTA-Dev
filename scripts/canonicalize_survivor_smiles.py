#!/usr/bin/env python3
"""Add RDKit-canonical product SMILES columns to a survivor parquet."""

from __future__ import annotations

import argparse
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
import yaml


@dataclass(frozen=True)
class ReactionRule:
    reaction_id: str
    reaction: Any


@dataclass(frozen=True)
class ProjectedMarker:
    reaction_id: str
    synthon_id: str | None


PROJECTED_MARKER_PREFIX = "PROJECTED_RULE_PRODUCT::"
PRODUCT_COLUMNS = ("canonical_smiles", "survivor_smiles", "product_smiles", "smiles")


def _strip_format_controls(value: str) -> str:
    return "".join(char for char in value if unicodedata.category(char) != "Cf")


def _has_control_character(value: str) -> bool:
    return any(unicodedata.category(char).startswith("C") for char in value)


def _strip_all_control_characters(value: str) -> str:
    return "".join(char for char in value if not unicodedata.category(char).startswith("C"))


def _mol_from_smiles_strict(value: str) -> Any | None:
    """Parse SMILES without allowing RDKit's whitespace-delimited name suffix."""

    stripped = value.strip()
    if stripped != value or any(char.isspace() for char in stripped) or _has_control_character(stripped):
        return None
    params = Chem.SmilesParserParams()
    params.allowCXSMILES = False
    params.parseName = False
    return Chem.MolFromSmiles(stripped, params)


def _has_query_or_unspecified_terms(mol: Any) -> bool:
    """Reject query-like molecules that are parseable but not concrete products."""

    for atom in mol.GetAtoms():
        if atom.HasQuery() or atom.GetAtomicNum() == 0:
            return True
    for bond in mol.GetBonds():
        if bond.HasQuery() or bond.GetBondType() == Chem.BondType.UNSPECIFIED:
            return True
    return False


def _has_multiple_fragments(mol: Any) -> bool:
    return len(Chem.GetMolFrags(mol)) != 1


def canonicalize_smiles(value: object) -> tuple[str | None, str]:
    """Return RDKit canonical SMILES and a status tag."""

    if not isinstance(value, str) or not value.strip():
        return None, "FAILED_EMPTY"
    mol = _mol_from_smiles_strict(value)
    if mol is None:
        return None, "FAILED_PARSE"
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "FAILED_SANITIZE"
    if _has_query_or_unspecified_terms(mol):
        return None, "FAILED_QUERY"
    if _has_multiple_fragments(mol):
        return None, "FAILED_DISCONNECTED"
    canonical = Chem.MolToSmiles(mol, canonical=True)
    if "." in canonical:
        return None, "FAILED_DISCONNECTED"
    return canonical, "OK"


def load_reaction_smarts(path: Path | None) -> list[ReactionRule]:
    """Load enabled RDKit reactions from the PRISM reaction-rule registry."""

    if path is None:
        return []
    payload = yaml.safe_load(path.read_text())
    reactions: list[ReactionRule] = []
    reaction_ids_seen: set[str] = set()
    for entry in payload.get("reactions", []) if isinstance(payload, dict) else []:
        if not bool(entry.get("enabled", True)):
            continue
        reaction_id = entry.get("reaction_id")
        if not isinstance(reaction_id, str) or not reaction_id.strip():
            continue
        reaction_id = reaction_id.strip()
        if reaction_id in reaction_ids_seen:
            raise ValueError(f"duplicate enabled reaction_id: {reaction_id}")
        reaction_ids_seen.add(reaction_id)
        smarts = entry.get("smarts")
        if not isinstance(smarts, str) or not smarts:
            continue
        reaction_from_smarts: Any = getattr(AllChem, "ReactionFromSmarts")
        reaction = reaction_from_smarts(smarts)
        if reaction is not None:
            reactions.append(ReactionRule(reaction_id=reaction_id, reaction=reaction))
    return reactions


def _first_string(row: Mapping[str, Any], columns: tuple[str, ...]) -> str | None:
    for column in columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip() and not value.startswith(PROJECTED_MARKER_PREFIX):
            return value
    return None


def _projected_reaction_id(row: Mapping[str, Any]) -> str | None:
    marker = _projected_marker(row)
    return marker.reaction_id if marker is not None else None


def _projected_marker(row: Mapping[str, Any]) -> ProjectedMarker | None:
    explicit = row.get("reaction_id")
    explicit_reaction_id = explicit.strip() if isinstance(explicit, str) and explicit.strip() else None
    primary_column = _first_product_like_column(row)
    primary = row.get(primary_column) if primary_column is not None else None
    markers: list[ProjectedMarker] = []
    for column in PRODUCT_COLUMNS:
        marker = _parse_projected_marker(row.get(column), column)
        if marker is not None:
            markers.append(marker)
    if isinstance(primary, str) and primary.strip():
        if not _is_projected_marker(primary) and markers:
            raise ValueError(f"secondary projected marker conflicts with primary {primary_column}")
        if not _is_projected_marker(primary) and _looks_like_malformed_projected_marker(primary):
            raise ValueError(f"malformed projected marker in primary {primary_column}")
    unique_markers = sorted(set(markers), key=lambda marker: (marker.reaction_id, marker.synthon_id or ""))
    if len(unique_markers) > 1:
        labels = [
            f"{marker.reaction_id}::{marker.synthon_id}" if marker.synthon_id else marker.reaction_id
            for marker in unique_markers
        ]
        raise ValueError(f"conflicting projected markers: {' != '.join(labels)}")
    marker = unique_markers[0] if unique_markers else None
    if explicit_reaction_id is not None and marker is not None and explicit_reaction_id != marker.reaction_id:
        raise ValueError(f"reaction_id mismatch: {explicit_reaction_id} != {marker.reaction_id}")
    if marker is not None:
        _validate_marker_synthon_id(row, marker)
    return marker


def _validate_projected_marker_format(row: Mapping[str, Any]) -> None:
    for column in PRODUCT_COLUMNS:
        _parse_projected_marker(row.get(column), column)


def _first_product_like_column(row: Mapping[str, Any]) -> str | None:
    for column in PRODUCT_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return column
    return None


def _is_projected_marker(value: object) -> bool:
    return isinstance(value, str) and value.startswith(PROJECTED_MARKER_PREFIX)


def _looks_like_malformed_projected_marker(value: object) -> bool:
    if not isinstance(value, str) or _is_projected_marker(value):
        return False
    normalized = _strip_all_control_characters(value).strip()
    return normalized.startswith(PROJECTED_MARKER_PREFIX)


def _parse_projected_marker(value: object, column: str) -> ProjectedMarker | None:
    if not isinstance(value, str):
        return None
    if _looks_like_malformed_projected_marker(value):
        raise ValueError(f"malformed projected marker in {column}")
    if not value.startswith(PROJECTED_MARKER_PREFIX):
        return None
    if (
        value != value.strip()
        or value != _strip_format_controls(value)
        or _has_control_character(value)
        or any(char.isspace() for char in value)
    ):
        raise ValueError(f"malformed projected marker in {column}")
    parts = value.removeprefix(PROJECTED_MARKER_PREFIX).split("::")
    if len(parts) not in (1, 2) or any(not part or part != part.strip() for part in parts):
        raise ValueError(f"malformed projected marker in {column}")
    return ProjectedMarker(reaction_id=parts[0], synthon_id=parts[1] if len(parts) == 2 else None)


def _validate_marker_synthon_id(row: Mapping[str, Any], marker: ProjectedMarker) -> None:
    synthon_records: list[tuple[str, str, str | None]] = []
    for id_column, smile_columns in (
        ("synthon_a_id", ("synthon_a_smiles", "synthon_a_canonical_smiles")),
        ("synthon_b_id", ("synthon_b_smiles", "synthon_b_canonical_smiles")),
        ("synthon_id", ("synthon_smiles", "synthon_canonical_smiles")),
    ):
        value = row.get(id_column)
        if not isinstance(value, str) or not value.strip():
            continue
        synthon_records.append((id_column, value.strip(), _first_string(row, smile_columns)))
    synthon_ids = {synthon_id for _, synthon_id, _ in synthon_records}
    normalized_by_id: dict[str, set[str | None]] = {}
    for _, synthon_id, smiles in synthon_records:
        normalized_smiles: str | None = None
        if smiles is not None:
            mol = _mol_from_smiles_strict(smiles)
            normalized_smiles = Chem.MolToSmiles(mol, canonical=True) if mol is not None else smiles.strip()
        normalized_by_id.setdefault(synthon_id, set()).add(normalized_smiles)
    for synthon_id, normalized_values in normalized_by_id.items():
        if len(normalized_values) > 1 or None in normalized_values:
            raise ValueError(f"duplicate synthon id across roles: {synthon_id}")
    if synthon_ids and marker.synthon_id is None:
        raise ValueError("projected marker missing synthon id")
    if marker.synthon_id is not None and synthon_ids and marker.synthon_id not in synthon_ids:
        raise ValueError(f"projected marker synthon mismatch: {marker.synthon_id}")


def reconstruct_product_smiles(row: Mapping[str, Any], reactions: list[ReactionRule]) -> tuple[str | None, str]:
    """Try scaffold + synthon SMARTS reconstruction before falling back to stored products."""

    _validate_projected_marker_format(row)
    if not reactions:
        return None, "NO_REACTION_RULES"
    marker = _projected_marker(row)
    if marker is None:
        return None, "NO_PROJECTED_REACTION"
    projected_reaction_id = marker.reaction_id
    candidate_reactions = (
        [rule for rule in reactions if rule.reaction_id == projected_reaction_id]
    )
    if not candidate_reactions:
        return None, f"MISSING_REACTION_RULE:{projected_reaction_id}"
    scaffold = _first_string(
        row,
        (
            "scaffold_smiles",
            "scaffold_canonical_smiles",
            "parent_scaffold_smiles",
            "parent_smiles",
        ),
    )
    synthon_candidates = [
        (
            row.get("synthon_a_id"),
            _first_string(row, ("synthon_smiles", "synthon_a_smiles", "synthon_a_canonical_smiles")),
        ),
        (
            row.get("synthon_b_id"),
            _first_string(row, ("synthon_b_smiles", "synthon_b_canonical_smiles")),
        ),
    ]
    synthons = [
        smiles
        for synthon_id, smiles in synthon_candidates
        if smiles is not None
        and (
            marker.synthon_id is None
            or (isinstance(synthon_id, str) and synthon_id.strip() == marker.synthon_id)
        )
    ]
    if scaffold is None or not synthons:
        return None, "MISSING_REACTANTS"
    scaffold_mol = _mol_from_smiles_strict(scaffold)
    if scaffold_mol is None:
        return None, "FAILED_SCAFFOLD_PARSE"
    if _has_multiple_fragments(scaffold_mol):
        return None, "FAILED_SCAFFOLD_DISCONNECTED"
    valid_products: set[str] = set()
    for synthon in synthons:
        synthon_mol = _mol_from_smiles_strict(synthon)
        if synthon_mol is None:
            continue
        if _has_multiple_fragments(synthon_mol):
            continue
        for rule in candidate_reactions:
            for reactants in ((scaffold_mol, synthon_mol), (synthon_mol, scaffold_mol)):
                try:
                    product_sets = rule.reaction.RunReactants(reactants)
                except Exception:
                    continue
                for product_set in product_sets:
                    if len(product_set) != 1:
                        continue
                    product = product_set[0]
                    try:
                        Chem.SanitizeMol(product)
                    except Exception:
                        continue
                    if _has_query_or_unspecified_terms(product):
                        continue
                    canonical = Chem.MolToSmiles(product, canonical=True)
                    if canonical and "." not in canonical:
                        valid_products.add(canonical)
    if len(valid_products) == 1:
        return next(iter(valid_products)), f"OK_RECONSTRUCTED:{projected_reaction_id}"
    if len(valid_products) > 1:
        return None, f"AMBIGUOUS_RECONSTRUCT:{projected_reaction_id}"
    detail = f"FAILED_RECONSTRUCT:{projected_reaction_id}" if projected_reaction_id else "FAILED_RECONSTRUCT"
    return None, detail


def choose_source_smiles(row: Mapping[str, Any]) -> object:
    """Pick the best available product-like SMILES from a survivor row."""

    for column in PRODUCT_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value and not value.startswith(PROJECTED_MARKER_PREFIX):
            return value
    return row.get("canonical_smiles")


def canonicalize_survivor_parquet(
    input_path: Path,
    output_path: Path,
    *,
    reaction_rules: Path | None = None,
    limit: int | None = None,
    require_any_ok: bool = False,
    require_all_ok: bool = False,
    expected_rows: int | None = None,
) -> pl.DataFrame:
    """Canonicalize survivor rows and write an enriched parquet."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if expected_rows is not None and expected_rows < 0:
        raise ValueError("expected_rows must be non-negative")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("output path must not overwrite input parquet")
    df = pl.read_parquet(input_path)
    rows = df.head(limit).to_dicts() if limit is not None else df.to_dicts()
    reactions = load_reaction_smarts(reaction_rules)
    canonical_values: list[str | None] = []
    statuses: list[str] = []
    sources: list[str] = []
    details: list[str] = []
    for row in rows:
        _validate_projected_marker_format(row)
        canonical, detail = reconstruct_product_smiles(row, reactions)
        if canonical is not None:
            status = "OK"
            source = "reaction_smarts"
        elif _projected_reaction_id(row) is not None:
            status = "FAILED_RECONSTRUCT"
            source = "failed"
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
        pl.Series("canonical_smiles_rdkit", canonical_values, dtype=pl.String),
        pl.Series("canonicalization_status", statuses, dtype=pl.String),
        pl.Series("canonicalization_source", sources, dtype=pl.String),
        pl.Series("canonicalization_detail", details, dtype=pl.String),
    )
    if require_all_ok and output.height == 0:
        raise ValueError("canonicalization produced zero rows")
    if require_all_ok and any(status != "OK" for status in statuses):
        raise ValueError("canonicalization produced failed rows")
    if expected_rows is not None and output.height != expected_rows:
        raise ValueError(f"canonicalization produced {output.height} rows; expected {expected_rows}")
    if require_any_ok and "OK" not in statuses:
        raise ValueError("canonicalization produced zero OK rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=output_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        if tmp_path.resolve() == input_path.resolve():
            raise ValueError("temporary output path collides with input parquet")
        output.write_parquet(tmp_path)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reaction-rules", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=100)
    args = parser.parse_args()
    if args.reaction_rules is not None and not args.reaction_rules.exists():
        raise FileNotFoundError(args.reaction_rules)
    output = canonicalize_survivor_parquet(
        args.input,
        args.output,
        reaction_rules=args.reaction_rules,
        require_all_ok=True,
        expected_rows=int(args.expected_rows),
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
