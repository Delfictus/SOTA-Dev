"""Schema contract validation for PRISM-DSTW Arrow/Polars boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml


PolarsDType = type[pl.DataType]


POLARS_DTYPE_MAP: dict[str, PolarsDType] = {
    "string": pl.String,
    "utf8": pl.String,
    "int16": pl.Int16,
    "int32": pl.Int32,
    "int64": pl.Int64,
    "uint32": pl.UInt32,
    "uint64": pl.UInt64,
    "float64": pl.Float64,
    "boolean": pl.Boolean,
    "bool": pl.Boolean,
}


@dataclass(frozen=True)
class SchemaColumn:
    name: str
    dtype: str
    ontology: str
    unit: str
    nullable: bool
    allowed: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SchemaContract:
    schema_version: str
    columns: dict[str, SchemaColumn]


def load_schema_contract(path: Path) -> SchemaContract:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Schema file {path} did not parse to a mapping.")
    schema_version = raw.get("schema_version")
    raw_columns = raw.get("columns")
    if not isinstance(schema_version, str) or not isinstance(raw_columns, dict):
        raise ValueError(f"Schema file {path} is missing schema_version or columns.")
    columns: dict[str, SchemaColumn] = {}
    for name, spec in raw_columns.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Schema column {name} in {path} is not a mapping.")
        allowed_values = spec.get("allowed", [])
        if not isinstance(allowed_values, list):
            raise ValueError(f"Schema column {name} allowed values must be a list.")
        columns[str(name)] = SchemaColumn(
            name=str(name),
            dtype=str(spec["dtype"]),
            ontology=str(spec["ontology"]),
            unit=str(spec["unit"]),
            nullable=bool(spec["nullable"]),
            allowed=tuple(str(value) for value in allowed_values) or None,
        )
    return SchemaContract(schema_version=schema_version, columns=columns)


def validate_schema(
    frame: pl.DataFrame | pl.LazyFrame,
    schema_path: Path,
) -> pl.DataFrame | pl.LazyFrame:
    contract = load_schema_contract(schema_path)
    actual = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    for column in contract.columns.values():
        if column.name not in actual:
            raise ValueError(f"Missing column {column.name} for schema {contract.schema_version}.")
        expected_dtype = POLARS_DTYPE_MAP.get(column.dtype.lower())
        if expected_dtype is not None and actual[column.name] != expected_dtype:
            raise TypeError(
                f"Type mismatch for {column.name}: expected {expected_dtype}, got {actual[column.name]}."
            )
        if isinstance(frame, pl.DataFrame) and not column.nullable:
            null_count = int(frame.select(pl.col(column.name).null_count()).item())
            if null_count:
                raise ValueError(f"Column {column.name} has {null_count} nulls but is non-nullable.")
            if actual[column.name] == pl.String:
                empty_count = int(
                    frame.select(
                        (pl.col(column.name).str.strip_chars() == "").sum().alias("empty_count")
                    ).item()
                )
                if empty_count:
                    raise ValueError(
                        f"Column {column.name} has {empty_count} empty strings but is non-nullable."
                    )
        if isinstance(frame, pl.DataFrame) and column.allowed:
            observed = frame.select(pl.col(column.name).drop_nulls().unique()).to_series().to_list()
            invalid = sorted({str(value) for value in observed if str(value) not in column.allowed})
            if invalid:
                raise ValueError(
                    f"Column {column.name} has values outside allowed set {list(column.allowed)}: {invalid}."
                )
    return frame


def schema_ontology_map(schema_path: Path) -> dict[str, dict[str, Any]]:
    contract = load_schema_contract(schema_path)
    return {
        name: {
            "dtype": column.dtype,
            "ontology": column.ontology,
            "unit": column.unit,
            "nullable": column.nullable,
            "allowed": list(column.allowed) if column.allowed else None,
        }
        for name, column in contract.columns.items()
    }


def schema_dtype_overrides(schema_path: Path) -> dict[str, PolarsDType]:
    contract = load_schema_contract(schema_path)
    overrides: dict[str, PolarsDType] = {}
    for column in contract.columns.values():
        dtype = POLARS_DTYPE_MAP.get(column.dtype.lower())
        if dtype is not None:
            overrides[column.name] = dtype
    return overrides


def read_schema_csv(path: Path, schema_path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        schema_overrides=schema_dtype_overrides(schema_path),
        null_values=[""],
    )
