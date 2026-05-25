#!/usr/bin/env python3
"""Model temporal-cascade transition chronology with a 1D Gaussian mixture."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import TypeAlias

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import write_provenance_parquet


CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
DEFAULT_OUTPUT = N80_DIR / "probabilistic_break_clusters.parquet"
EPSILON = 1.0e-9
JsonObject: TypeAlias = dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-cascade", type=Path, default=N80_DIR / "temporal_cascade.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=80)
    return parser.parse_args()


def gaussian_pdf(value: float, mean: float, variance: float) -> float:
    safe_variance = max(variance, EPSILON)
    return math.exp(-0.5 * (value - mean) * (value - mean) / safe_variance) / math.sqrt(2.0 * math.pi * safe_variance)


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    position = min(max(int(round(fraction * float(len(values) - 1))), 0), len(values) - 1)
    return sorted(values)[position]


def fit_gmm(values: list[float], k: int, max_iterations: int) -> tuple[list[float], list[float], list[float], float]:
    n = len(values)
    means = [quantile(values, (idx + 1.0) / (k + 1.0)) for idx in range(k)]
    total_mean = sum(values) / float(max(n, 1))
    total_variance = max(sum((value - total_mean) ** 2 for value in values) / float(max(n, 1)), EPSILON)
    variances = [total_variance for _idx in range(k)]
    weights = [1.0 / float(k) for _idx in range(k)]
    log_likelihood = -math.inf
    for _iteration in range(max_iterations):
        responsibilities: list[list[float]] = []
        current_log_likelihood = 0.0
        for value in values:
            raw = [weights[idx] * gaussian_pdf(value, means[idx], variances[idx]) for idx in range(k)]
            denominator = max(sum(raw), EPSILON)
            responsibilities.append([item / denominator for item in raw])
            current_log_likelihood += math.log(denominator)
        totals = [sum(row[idx] for row in responsibilities) for idx in range(k)]
        for idx in range(k):
            total = max(totals[idx], EPSILON)
            weights[idx] = total / float(n)
            means[idx] = sum(resp[idx] * value for resp, value in zip(responsibilities, values, strict=True)) / total
            variances[idx] = max(
                sum(resp[idx] * (value - means[idx]) ** 2 for resp, value in zip(responsibilities, values, strict=True))
                / total,
                EPSILON,
            )
        if abs(current_log_likelihood - log_likelihood) < 1.0e-6:
            log_likelihood = current_log_likelihood
            break
        log_likelihood = current_log_likelihood
    order = sorted(range(k), key=lambda idx: means[idx])
    return (
        [weights[idx] for idx in order],
        [means[idx] for idx in order],
        [variances[idx] for idx in order],
        log_likelihood,
    )


def bic(log_likelihood: float, n: int, k: int) -> float:
    parameter_count = 3 * k - 1
    return parameter_count * math.log(float(max(n, 1))) - 2.0 * log_likelihood


def best_model(values: list[float], min_clusters: int, max_clusters: int, max_iterations: int) -> tuple[list[float], list[float], list[float]]:
    best_score = math.inf
    best: tuple[list[float], list[float], list[float]] | None = None
    for k in range(min_clusters, max_clusters + 1):
        weights, means, variances, log_likelihood = fit_gmm(values, k, max_iterations)
        score = bic(log_likelihood, len(values), k)
        if score < best_score:
            best_score = score
            best = (weights, means, variances)
    if best is None:
        raise ValueError("failed to fit temporal mixture model")
    return best


def probabilities(value: float, weights: list[float], means: list[float], variances: list[float]) -> list[float]:
    raw = [weights[idx] * gaussian_pdf(value, means[idx], variances[idx]) for idx in range(len(weights))]
    denominator = max(sum(raw), EPSILON)
    return [item / denominator for item in raw]


def entropy(probs: list[float]) -> float:
    if len(probs) <= 1:
        return 0.0
    raw = -sum(prob * math.log(max(prob, EPSILON)) for prob in probs)
    return raw / math.log(float(len(probs)))


def chronology_frame(path: Path, min_clusters: int, max_clusters: int, max_iterations: int) -> pl.DataFrame:
    source = pl.read_parquet(path).select(
        ["condition_id", "protocol_group", "primary_residue_idx", "first_ramp_md_step", "supporting_stream_count"]
    )
    values = [float(item) for item in source.get_column("first_ramp_md_step").to_list()]
    weights, means, variances = best_model(values, min_clusters, max_clusters, max_iterations)
    rows: list[dict[str, object]] = []
    for row in source.iter_rows(named=True):
        value = float(row["first_ramp_md_step"])
        probs = probabilities(value, weights, means, variances)
        assigned = max(range(len(probs)), key=lambda idx: probs[idx])
        rows.append(
            {
                **row,
                "cluster_id": assigned + 1,
                "cluster_centroid_md_step": means[assigned],
                "cluster_probability": probs[assigned],
                "row_overlap_entropy": entropy(probs),
            }
        )
    assigned_frame = pl.DataFrame(rows)
    metrics = (
        assigned_frame.lazy()
        .group_by("cluster_id")
        .agg(
            [
                pl.col("cluster_probability").mean().alias("cluster_confidence"),
                pl.col("row_overlap_entropy").mean().alias("temporal_overlap_entropy"),
                pl.col("first_ramp_md_step").mean().alias("observed_cluster_centroid_md_step"),
                pl.col("first_ramp_md_step").var().fill_null(0.0).alias("within_cluster_variance"),
                pl.col("supporting_stream_count").mean().alias("mean_supporting_stream_count"),
                pl.len().alias("cluster_member_count"),
            ]
        )
        .with_columns(
            (
                pl.col("within_cluster_variance")
                / pl.max_horizontal([pl.col("mean_supporting_stream_count").cast(pl.Float64), pl.lit(1.0)])
            ).alias("inter_replicate_stability")
        )
    )
    return (
        assigned_frame.lazy()
        .join(metrics, on="cluster_id", how="left")
        .sort(["cluster_id", "condition_id", "primary_residue_idx"])
        .collect()
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    frame = chronology_frame(
        Path(args.temporal_cascade),
        int(args.min_clusters),
        int(args.max_clusters),
        int(args.max_iterations),
    )
    write_provenance_parquet(
        frame,
        output,
        producer_script=Path(__file__),
        source_parquets=[Path(args.temporal_cascade)],
        schema_version="probabilistic_break_clusters.v1",
        pipeline_stage="probabilistic_break_clusters",
        partition_keys=["cluster_id", "condition_id"],
        ledger_parameters={
            "model": "1D Gaussian mixture selected by BIC",
            "inter_replicate_stability": "within-cluster centroid variance scaled by supporting stream count because temporal_cascade is an aggregate tensor",
        },
        ledger_output_value={"rows": frame.height, "output_path": output.as_posix()},
        repo_root=REPO_ROOT,
    )
    sys.stdout.write(f"wrote {output} rows={frame.height}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
