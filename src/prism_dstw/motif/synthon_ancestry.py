"""Synthon-ancestry decomposition and lock enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
from scipy.stats import fisher_exact  # type: ignore[import-untyped]


@dataclass(frozen=True)
class SynthonAncestry:
    """Lock enrichment statistics for one synthon/source building block."""

    synthon_smiles: str
    enamine_id: str
    reaction_class: str
    lock_positive_rate: float
    lock_negative_rate: float
    enrichment_ratio: float
    p_value: float
    thermodynamic_role: str
    mean_reward_when_present: float
    mean_reward_when_absent: float
    exit_vector_preference: dict[int, float]


def compute_synthon_ancestry(
    candidates: pl.DataFrame,
    *,
    lock_threshold: float = 0.0,
    min_occurrences: int = 3,
) -> list[SynthonAncestry]:
    """Compute Fisher-exact lock enrichment for synthons."""

    required = {"synthon_smiles", "lock_geometry_score", "reward"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"candidate table missing required columns: {sorted(missing)}")
    if candidates.height == 0:
        return []
    frame = _with_optional_columns(candidates).with_columns(
        (pl.col("lock_geometry_score") > lock_threshold).alias("lock_positive")
    )
    n_lock_pos = int(frame.filter(pl.col("lock_positive")).height)
    n_lock_neg = int(frame.filter(~pl.col("lock_positive")).height)
    expected_rate = n_lock_pos / max(n_lock_pos + n_lock_neg, 1)
    grouped = frame.group_by("synthon_smiles").agg(
        pl.col("lock_positive").sum().alias("n_pos"),
        pl.col("lock_positive").count().alias("n_total"),
        pl.col("reward").mean().alias("mean_reward"),
        pl.col("enamine_id").first().alias("enamine_id"),
        pl.col("reaction_class").first().alias("reaction_class"),
        pl.col("exit_vector_idx").mode().first().alias("preferred_exit_vector"),
    )
    n_tests = max(grouped.height, 1)
    results: list[SynthonAncestry] = []
    for row in grouped.iter_rows(named=True):
        total = int(row["n_total"])
        if total < min_occurrences:
            continue
        n_pos = int(row["n_pos"])
        n_neg = total - n_pos
        table = [[n_pos, n_neg], [n_lock_pos - n_pos, n_lock_neg - n_neg]]
        _, p_value = fisher_exact(table, alternative="greater")
        p_corrected = min(float(p_value) * float(n_tests), 1.0)
        lock_pos_rate = n_pos / max(total, 1)
        enrichment = lock_pos_rate / max(expected_rate, 1.0e-6)
        synthon = str(row["synthon_smiles"])
        absent_mean = frame.filter(pl.col("synthon_smiles") != synthon).get_column("reward").mean()
        exit_idx = _int(row.get("preferred_exit_vector"), 0)
        results.append(
            SynthonAncestry(
                synthon_smiles=synthon,
                enamine_id=str(row.get("enamine_id") or synthon),
                reaction_class=str(row.get("reaction_class") or "unknown"),
                lock_positive_rate=lock_pos_rate,
                lock_negative_rate=1.0 - lock_pos_rate,
                enrichment_ratio=enrichment,
                p_value=p_corrected,
                thermodynamic_role="LOCK_WEDGE" if enrichment > 1.0 else "NEUTRAL",
                mean_reward_when_present=_float(row.get("mean_reward"), 0.0),
                mean_reward_when_absent=_float(absent_mean, 0.0),
                exit_vector_preference={exit_idx: 1.0},
            )
        )
    return sorted(results, key=lambda item: (item.enrichment_ratio, -item.p_value), reverse=True)


def _with_optional_columns(frame: pl.DataFrame) -> pl.DataFrame:
    output = frame
    if "enamine_id" not in output.columns:
        output = output.with_columns(pl.col("synthon_smiles").alias("enamine_id"))
    if "reaction_class" not in output.columns:
        output = output.with_columns(pl.lit("inferred_from_trajectory").alias("reaction_class"))
    if "exit_vector_idx" not in output.columns:
        output = output.with_columns(pl.lit(0).alias("exit_vector_idx"))
    return output


def _int(value: Any, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int | float | str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _float(value: Any, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
