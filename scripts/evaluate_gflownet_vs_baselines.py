#!/usr/bin/env python3
"""Phase 4 — trained policy vs. baselines.

Baselines:
  1. Random policy over valid anchors (uniform over the action space).
  2. Reward-weighted replay from the real512 survivor corpus (sampling
     proportional to existing rewards — represents what a memorizer would
     do).
  3. Top real512 O3A-Zmatrix survivors (deterministic top-K by reward).

All three are scored against the same Rust oracle. Required: trained
policy beats random on reward_p95, top_reward, nontrivial unique count,
and consensus-stable candidate count.
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path("/home/diddy/Desktop/Prism4D-bio")
sys.path.insert(0, str(REPO / "src"))
from prism_dstw.orchestration.rust_reward_oracle import (  # noqa: E402
    BatchedRustOracle, OracleProposal,
)

TRACK_A = REPO / "campaigns/glp1r_aleniglipron/track_a_generative"
SURVIVORS = TRACK_A / "vspace_survivors_real512_o3a_zmatrix.parquet"
TRAINED  = TRACK_A / "gflownet_oracle_consensus_scores.parquet"
ORACLE_BIN = REPO / "target/release/oracle_scorer"

OUT_JSON = TRACK_A / "gflownet_vs_baseline_comparison.json"
OUT_MD   = TRACK_A / "gflownet_vs_baseline_comparison.md"
N_BASELINE_SAMPLES = 5000


async def score_smiles_batch(smiles_and_anchors: list[tuple[str, str]]) -> pl.DataFrame:
    oracle = BatchedRustOracle(
        oracle_binary=ORACLE_BIN, survivor_corpus=SURVIVORS, max_batch_size=64
    )
    results = []
    buf: list[OracleProposal] = []
    for i, (smi, anch) in enumerate(smiles_and_anchors):
        buf.append(OracleProposal(anchor_id=anch, canonical_smiles=smi,
                                  trajectory_id=f"bl-{i:06d}"))
        if len(buf) >= 64:
            res = await oracle.score_batch(buf)
            results.append(res.rows)
            buf = []
    if buf:
        # Oracle rejects duplicate SMILES in one batch. De-dup the buffer.
        seen = set()
        deduped = []
        for p in buf:
            if p.canonical_smiles in seen:
                continue
            seen.add(p.canonical_smiles)
            deduped.append(p)
        if deduped:
            res = await oracle.score_batch(deduped)
            results.append(res.rows)
    return pl.concat(results) if results else pl.DataFrame()


def reward_distribution_stats(df: pl.DataFrame, reward_col: str = "reward") -> dict[str, float]:
    if df.is_empty():
        return {"n": 0, "mean": 0.0, "p95": 0.0, "max": 0.0, "unique": 0, "consensus_stable": 0}
    n = df.height
    arr = df.get_column(reward_col).to_numpy()
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "p95":  float(np.percentile(arr, 95)),
        "max":  float(np.max(arr)),
        "unique": df.unique(subset=["canonical_smiles"], keep="first").height,
        # "Consensus-stable" = SMILES with positive valid reward and oracle_valid.
        "consensus_stable": df.filter(
            (pl.col(reward_col) > 0.0) & (pl.col("oracle_valid") if "oracle_valid" in df.columns else pl.lit(True))
        ).height,
    }


async def main() -> int:
    survivors = pl.read_parquet(SURVIVORS)
    print("=== Phase 4 — baselines vs trained policy ===")
    rng = random.Random(20260524)

    # ---- 1. Random over valid anchors (uniform sample over corpus) ----
    rand_idx = rng.sample(range(survivors.height), min(N_BASELINE_SAMPLES, survivors.height))
    rand_sample = survivors[rand_idx]
    # Many corpus entries share canonical_smiles; dedup before oracle.
    rand_unique = rand_sample.unique(subset=["canonical_smiles"], keep="first")
    print(f"  random: {rand_sample.height} samples, {rand_unique.height} unique SMILES")
    random_scores = await score_smiles_batch([
        (row["canonical_smiles"], row["anchor_id"])
        for row in rand_unique.iter_rows(named=True)
    ])

    # ---- 2. Reward-weighted replay (sampling proportional to existing reward) ----
    if "score" in survivors.columns:
        w = survivors.get_column("score").to_numpy()
    elif "reward" in survivors.columns:
        w = survivors.get_column("reward").to_numpy()
    else:
        # Fall back to fragment_pi_complement as a proxy.
        w = survivors.get_column("fragment_pi_complement").to_numpy()
    w = np.clip(w - w.min() + 1e-6, 1e-6, None)
    w = w / w.sum()
    np_rng = np.random.default_rng(20260525)
    replay_idx = np_rng.choice(survivors.height, size=min(N_BASELINE_SAMPLES, survivors.height),
                                replace=False, p=w)
    replay_sample = survivors[replay_idx.tolist()]
    replay_unique = replay_sample.unique(subset=["canonical_smiles"], keep="first")
    print(f"  replay: {replay_sample.height} samples, {replay_unique.height} unique SMILES")
    replay_scores = await score_smiles_batch([
        (row["canonical_smiles"], row["anchor_id"])
        for row in replay_unique.iter_rows(named=True)
    ])

    # ---- 3. Top real512 survivors by existing reward ----
    sort_col = "score" if "score" in survivors.columns else "fragment_pi_complement"
    top_sv = survivors.sort(sort_col, descending=True).head(100)
    top_unique = top_sv.unique(subset=["canonical_smiles"], keep="first")
    print(f"  top512 by {sort_col}: {top_unique.height} unique SMILES")
    top_scores = await score_smiles_batch([
        (row["canonical_smiles"], row["anchor_id"])
        for row in top_unique.iter_rows(named=True)
    ])

    # ---- Trained policy ----
    if not TRAINED.is_file():
        print(f"  WARN: trained policy oracle scores not yet present at {TRAINED} — "
              f"trained stats will be empty.")
        trained = pl.DataFrame()
    else:
        trained = pl.read_parquet(TRAINED)

    rand_stats   = reward_distribution_stats(random_scores)
    replay_stats = reward_distribution_stats(replay_scores)
    top_stats    = reward_distribution_stats(top_scores)
    trained_stats = reward_distribution_stats(trained, reward_col="reward_mean") if not trained.is_empty() else \
                    {"n":0,"mean":0,"p95":0,"max":0,"unique":0,"consensus_stable":0}

    # Verdict: trained must beat random on 4 metrics.
    beats_random = {
        "reward_p95":      trained_stats["p95"]  > rand_stats["p95"],
        "top_reward":      trained_stats["max"]  > rand_stats["max"],
        "nontrivial_unique_count":   trained_stats["unique"] > rand_stats["unique"],
        "consensus_stable_count":    trained_stats["consensus_stable"] > rand_stats["consensus_stable"],
    }
    verdict = "PASS" if all(beats_random.values()) else "FAIL"

    report = {
        "package":          "PRISM_TRACK_A_GFLOWNET_V1_INFERENCE",
        "phase":            "4_baseline_comparison",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baselines": {
            "random_uniform":     rand_stats,
            "reward_weighted_replay": replay_stats,
            "top_real512":        top_stats,
        },
        "trained":          trained_stats,
        "beats_random":     beats_random,
        "verdict_vs_random": verdict,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    md = [
        "# GFlowNet v1 — Trained-Policy vs. Baselines",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Reward distribution stats",
        "",
        "| baseline | n | mean | p95 | max | unique | consensus stable |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| random uniform        | {rand_stats['n']} | {rand_stats['mean']:.3f} | {rand_stats['p95']:.3f} | {rand_stats['max']:.3f} | {rand_stats['unique']} | {rand_stats['consensus_stable']} |",
        f"| reward-weighted replay| {replay_stats['n']} | {replay_stats['mean']:.3f} | {replay_stats['p95']:.3f} | {replay_stats['max']:.3f} | {replay_stats['unique']} | {replay_stats['consensus_stable']} |",
        f"| top real512           | {top_stats['n']} | {top_stats['mean']:.3f} | {top_stats['p95']:.3f} | {top_stats['max']:.3f} | {top_stats['unique']} | {top_stats['consensus_stable']} |",
        f"| **trained policy**    | {trained_stats['n']} | **{trained_stats['mean']:.3f}** | **{trained_stats['p95']:.3f}** | **{trained_stats['max']:.3f}** | **{trained_stats['unique']}** | **{trained_stats['consensus_stable']}** |",
        "",
        "## Beats-random gate (required by directive)",
        "",
        "| metric | beats random? |",
        "|---|---|",
    ]
    for k, v in beats_random.items():
        md.append(f"| {k} | {'YES' if v else 'NO'} |")
    md.append("")
    md.append(f"**Overall verdict vs random:** {verdict}")
    md.append("")
    OUT_MD.write_text("\n".join(md) + "\n")

    print(f"  -> {OUT_JSON}")
    print(f"  -> {OUT_MD}")
    print(f"  verdict vs random: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
