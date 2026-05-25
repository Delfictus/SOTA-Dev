#!/usr/bin/env python3
"""Phase 8 — visual + review-card artifacts."""
from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path("/home/diddy/Desktop/Prism4D-bio")
TRACK_A = REPO / "campaigns/glp1r_aleniglipron/track_a_generative"
PLOT_DIR = TRACK_A / "review_artifacts"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CONSENSUS = TRACK_A / "gflownet_oracle_consensus_scores.parquet"
TOP100    = TRACK_A / "gflownet_top_100_candidates.parquet"
SAMPLES   = TRACK_A / "gflownet_raw_policy_samples.parquet"


def fig_savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> int:
    if not CONSENSUS.is_file() or not TOP100.is_file() or not SAMPLES.is_file():
        print("HARD-FAIL: required inputs missing for review artifacts", file=sys.stderr)
        return 2
    consensus = pl.read_parquet(CONSENSUS)
    top100    = pl.read_parquet(TOP100)
    samples   = pl.read_parquet(SAMPLES)

    print(f"=== Phase 8 — review artifacts ===")

    # 1. reward_distribution.png
    plt.figure(figsize=(8, 4.5))
    plt.hist(consensus.get_column("reward_mean").to_numpy(), bins=50,
             alpha=0.7, edgecolor="black")
    plt.axvline(top100.get_column("reward_mean").min(), color="red", linestyle="--",
                label=f"top-100 floor = {top100.get_column('reward_mean').min():.3f}")
    plt.xlabel("reward_mean (oracle)")
    plt.ylabel("candidates")
    plt.title("Reward distribution across consensus-scored candidates")
    plt.legend()
    fig_savefig(PLOT_DIR / "reward_distribution.png")

    # 2. reward_vs_uncertainty.png
    plt.figure(figsize=(8, 4.5))
    plt.scatter(consensus.get_column("reward_cv").to_numpy(),
                consensus.get_column("reward_mean").to_numpy(),
                s=10, alpha=0.5)
    plt.xlabel("reward_cv (deterministic backend → 0)")
    plt.ylabel("reward_mean")
    plt.title("Reward vs uncertainty")
    fig_savefig(PLOT_DIR / "reward_vs_uncertainty.png")

    # 3. reward_vs_clash.png
    plt.figure(figsize=(8, 4.5))
    plt.scatter(consensus.get_column("adjusted_pi_clash_mean").to_numpy(),
                consensus.get_column("reward_mean").to_numpy(),
                s=10, alpha=0.5)
    plt.xlabel("adjusted_pi_clash_mean")
    plt.ylabel("reward_mean")
    plt.title("Reward vs adjusted-clash")
    fig_savefig(PLOT_DIR / "reward_vs_clash.png")

    # 4. cryptic_bonus_vs_reward.png
    plt.figure(figsize=(8, 4.5))
    plt.scatter(consensus.get_column("cryptic_bonus_mean").to_numpy(),
                consensus.get_column("reward_mean").to_numpy(),
                s=10, alpha=0.5)
    plt.xlabel("cryptic_bonus_mean")
    plt.ylabel("reward_mean")
    plt.title("Cryptic-bonus vs reward")
    fig_savefig(PLOT_DIR / "cryptic_bonus_vs_reward.png")

    # 5. top100_cluster_summary.png
    if "scaffold_cluster_id" in top100.columns:
        cluster_counts = top100.group_by("scaffold_cluster_id").len().sort("len", descending=True)
        plt.figure(figsize=(8, 4.5))
        plt.bar(range(cluster_counts.height),
                cluster_counts.get_column("len").to_numpy())
        plt.xlabel("scaffold cluster index (sorted by size)")
        plt.ylabel("candidates in cluster")
        plt.title(f"Top-100 scaffold-cluster distribution (n_clusters={cluster_counts.height})")
        fig_savefig(PLOT_DIR / "top100_cluster_summary.png")

    # 6. temperature_source_distribution.png
    if "regime" in top100.columns:
        regime_counts = top100.group_by("regime").len().sort("len", descending=True)
        plt.figure(figsize=(8, 4.5))
        plt.bar(regime_counts.get_column("regime").to_list(),
                regime_counts.get_column("len").to_numpy())
        plt.xlabel("sampling regime")
        plt.ylabel("top-100 candidates")
        plt.title("Top-100 by sampling regime")
        fig_savefig(PLOT_DIR / "temperature_source_distribution.png")

    # 7. trajectory_entropy_distribution.png
    if "trajectory_entropy" in samples.columns:
        plt.figure(figsize=(8, 4.5))
        for regime, group in samples.group_by("regime"):
            arr = group.get_column("trajectory_entropy").to_numpy()
            plt.hist(arr, bins=40, alpha=0.5, label=regime[0] if isinstance(regime, tuple) else str(regime))
        plt.xlabel("trajectory_entropy")
        plt.ylabel("count")
        plt.legend()
        plt.title("Trajectory entropy by regime")
        fig_savefig(PLOT_DIR / "trajectory_entropy_distribution.png")

    # 8. review cards (HTML + MD)
    cards_html_path = TRACK_A / "gflownet_top100_review_cards.html"
    cards_md_path   = TRACK_A / "gflownet_top100_review_cards.md"
    md_lines = ["# GFlowNet v1 — Top-100 Review Cards", ""]
    html_cards = []
    for row in top100.iter_rows(named=True):
        rank = row.get("rank", "?")
        smi = row.get("canonical_smiles", "")
        rm  = row.get("reward_mean", 0)
        rs  = row.get("reward_std", 0)
        cl  = row.get("adjusted_pi_clash_mean", 0)
        co  = row.get("fragment_pi_complement_mean", 0)
        cb  = row.get("cryptic_bonus_mean", 0)
        bucket = row.get("selection_bucket", "")
        epi    = row.get("epistemic_class", "PROJECTED")
        md_lines.extend([
            f"## #{rank}",
            f"- SMILES: `{smi}`",
            f"- reward_mean: {rm:.3f}  reward_std: {rs:.3f}",
            f"- adjusted_pi_clash: {cl:.3f}  fragment_pi_complement: {co:.3f}  cryptic_bonus: {cb:.3f}",
            f"- bucket: {bucket}",
            f"- epistemic: {epi}",
            f"- falsification note: Treat as PROJECTED. No biological assertion. Subject to wet-lab gates.",
            "",
        ])
        html_cards.append(
            f"<div class='card'><h3>#{rank}</h3>"
            f"<div class='smiles'><code>{html.escape(smi)}</code></div>"
            f"<div>reward μ={rm:.3f} σ={rs:.3f}</div>"
            f"<div>clash {cl:.3f}  complement {co:.3f}  cryptic {cb:.3f}</div>"
            f"<div class='bucket'>{html.escape(str(bucket))}</div>"
            f"<div class='epi'>{html.escape(str(epi))}</div>"
            f"<div class='falsi'>Falsification gate required before biological claim.</div></div>"
        )
    cards_md_path.write_text("\n".join(md_lines) + "\n")
    cards_html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>GFlowNet v1 review cards</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;background:#0e1320;color:#e8ecf4;margin:24px;}"
        ".card{background:#141a2c;border:1px solid #1f2a44;border-radius:8px;padding:12px 16px;margin:10px 0;}"
        ".card h3{margin:0 0 4px;font-size:14px;color:#5b9dff;}"
        ".smiles{font-size:12px;word-break:break-all;margin-bottom:4px;}"
        ".bucket{color:#ffb454;font-size:12px;}.epi{color:#ff6e6e;font-size:12px;}"
        ".falsi{color:#8b94a9;font-size:11px;margin-top:4px;}"
        "</style></head><body><h1>GFlowNet v1 — Top-100 Review Cards</h1>"
        + "\n".join(html_cards) + "</body></html>"
    )

    print(f"  plots -> {PLOT_DIR}")
    print(f"  cards -> {cards_html_path}")
    print(f"  cards -> {cards_md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
