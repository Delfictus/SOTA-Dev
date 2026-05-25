#!/usr/bin/env python3
"""Compute population-weighted and ancestry-stratified GLP1R consensus grids."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
POP_DIR = TRACK_A / "population_pgx"
N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
DEFAULT_WT_GRID = N80_DIR / "signal_grid_variance_channel.parquet"
DEFAULT_MANIFEST = POP_DIR / "variant_perturbation_manifest.json"
DEFAULT_OUTPUT = TRACK_A / "signal_grid_population_consensus.parquet"
DEFAULT_ANCESTRY_DIR = POP_DIR / "consensus_grids"
DEFAULT_REPORT = POP_DIR / "population_consensus_grid_report.json"

JsonObject: TypeAlias = dict[str, Any]
ANCESTRIES = ("EUR", "AFR", "EAS", "SAS", "AMR")
CLASS_ORDER = ("void", "thermally_destabilized", "thermally_activated", "stable_occupied")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wt-grid", type=Path, default=DEFAULT_WT_GRID)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ancestry-dir", type=Path, default=DEFAULT_ANCESTRY_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def atomic_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_grid(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def load_wt(path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .filter(pl.col("condition_id") == "glp1r_6XOX_WT")
        .collect()
        .sort("voxel_idx")
    )


def load_grid(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).sort("voxel_idx")


def variant_weight(variant: JsonObject, ancestry: str | None = None) -> float:
    consensus = float(cast(Any, variant.get("consensus_weight", 0.0)))
    if consensus <= 0.0:
        return 0.0
    if ancestry is None:
        maf = float(cast(Any, variant.get("maf_global", 0.0)))
    else:
        raw = variant.get("maf_by_ancestry", {})
        maf = float(cast(Any, cast(dict[str, object], raw).get(ancestry, 0.0))) if isinstance(raw, dict) else 0.0
    return consensus * math.sqrt(max(maf, 0.0))


def compute_consensus(
    *,
    wt: pl.DataFrame,
    variants: list[JsonObject],
    ancestry: str | None,
) -> tuple[pl.DataFrame, JsonObject]:
    frames: list[tuple[str, float, pl.DataFrame]] = [("WT", 1.0, wt)]
    for variant in variants:
        if int(variant["tier"]) > 2:
            continue
        weight = variant_weight(variant, ancestry)
        if weight <= 0.0:
            continue
        frames.append((str(variant["mutation"]), weight, load_grid(Path(str(variant["grid_path"])))))

    total_weight = sum(weight for _, weight, _ in frames)
    if total_weight <= 0.0:
        raise RuntimeError("consensus has no positive weights")

    n = wt.height
    votes = {name: np.zeros(n, dtype=np.float64) for name in CLASS_ORDER}
    cold = np.zeros(n, dtype=np.float64)
    warm = np.zeros(n, dtype=np.float64)
    disputed = np.zeros(n, dtype=np.float64)
    for _, weight, frame in frames:
        classes = np.array(frame.get_column("variance_class").to_list(), dtype=object)
        for class_name in CLASS_ORDER:
            votes[class_name] += np.where(classes == class_name, weight, 0.0)
        cold += frame.get_column("hit_count_cold_mean").to_numpy().astype(np.float64) * weight
        warm += frame.get_column("hit_count_warm_mean").to_numpy().astype(np.float64) * weight
    cold /= total_weight
    warm /= total_weight

    vote_matrix = np.stack([votes[name] for name in CLASS_ORDER], axis=1)
    class_indices = np.argmax(vote_matrix, axis=1)
    consensus_classes = [CLASS_ORDER[int(index)] for index in class_indices.tolist()]
    raw_consensus_classes = list(consensus_classes)
    strengths = np.max(vote_matrix, axis=1) / total_weight
    disputed = np.where(strengths <= 0.60, 1.0, 0.0)
    bands = np.where(strengths > 0.80, "STRONG_CONSENSUS", np.where(strengths > 0.60, "MODERATE_CONSENSUS", "WEAK_CONSENSUS"))
    class_array = np.array(consensus_classes, dtype=object)
    fallback_mask = np.zeros(n, dtype=bool)
    if not np.any(class_array == "thermally_activated"):
        gain_delta = warm - cold
        fallback_mask = (gain_delta > 0.0) & (warm > 0.0) & (class_array != "stable_occupied")
        class_array = np.where(fallback_mask, "thermally_activated", class_array)
        consensus_classes = [str(value) for value in class_array.tolist()]
    complement_bonus = np.where(
        class_array == "thermally_activated",
        np.where(fallback_mask, 0.5, np.where(strengths > 0.80, 2.0, np.where(strengths > 0.60, 1.0, 0.0))),
        0.0,
    )
    penalty_multiplier = np.where(
        np.array(consensus_classes, dtype=object) == "stable_occupied",
        1.5,
        np.where(disputed > 0.0, 1.2, 1.0),
    )
    output = wt.with_columns(
        pl.lit(f"glp1r_6XOX_CONSENSUS_{ancestry or 'GLOBAL'}").alias("condition_id"),
        pl.Series("hit_count_cold_mean", cold),
        pl.Series("hit_count_warm_mean", warm),
        pl.Series("variance_class", consensus_classes),
        pl.Series("consensus_raw_variance_class", raw_consensus_classes),
        pl.Series("consensus_activation_fallback", fallback_mask),
        pl.Series("consensus_strength", strengths),
        pl.Series("consensus_band", [str(value) for value in bands.tolist()]),
        pl.Series("consensus_complement_bonus", complement_bonus),
        pl.Series("consensus_penalty_multiplier", penalty_multiplier),
        pl.Series("variant_disputed", disputed.astype(bool)),
    )
    metrics: JsonObject = {
        "ancestry": ancestry or "GLOBAL",
        "source_count": len(frames),
        "total_weight": total_weight,
        "mean_consensus_strength": float(np.mean(strengths)),
        "weak_consensus_voxels": int(np.count_nonzero(disputed)),
        "thermally_activated_voxels": int(sum(1 for value in consensus_classes if value == "thermally_activated")),
        "activation_fallback_voxels": int(np.count_nonzero(fallback_mask)),
        "stable_occupied_voxels": int(sum(1 for value in consensus_classes if value == "stable_occupied")),
        "weights": {name: weight for name, weight, _ in frames},
    }
    return output, metrics


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    variants = cast(list[JsonObject], manifest["variants"])
    wt = load_wt(args.wt_grid)
    global_grid, global_metrics = compute_consensus(wt=wt, variants=variants, ancestry=None)
    write_grid(args.output, global_grid)
    ancestry_metrics: dict[str, JsonObject] = {}
    for ancestry in ANCESTRIES:
        grid, metrics = compute_consensus(wt=wt, variants=variants, ancestry=ancestry)
        output = args.ancestry_dir / f"signal_grid_consensus_{ancestry}.parquet"
        write_grid(output, grid)
        metrics["path"] = output.as_posix()
        ancestry_metrics[ancestry] = metrics
        print(
            "ancestry_consensus_grid "
            f"ancestry={ancestry} mean_strength={float(metrics['mean_consensus_strength']):.4f} "
            f"activated={metrics['thermally_activated_voxels']} path={output}"
        )
    report: JsonObject = {
        "schema_version": "PRISM.population_consensus_grid.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if int(global_metrics["source_count"]) >= 10 else "WARN_REVIEW",
        "output": args.output.as_posix(),
        "manifest": args.manifest.as_posix(),
        "global": global_metrics,
        "ancestry": ancestry_metrics,
        "tier_scope": "WT plus Tier 1 and Tier 2 variants",
        "weighting": "consensus_weight * sqrt(MAF)",
    }
    atomic_json(args.report, report)
    print(
        "population_consensus_grid "
        f"status={report['status']} sources={global_metrics['source_count']} "
        f"activated={global_metrics['thermally_activated_voxels']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
