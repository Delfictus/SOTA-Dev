#!/usr/bin/env python3
"""Phase 9 — validate the bundle + build the final delivery tarball."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO = Path("/home/diddy/Desktop/Prism4D-bio")
TRACK_A = REPO / "campaigns/glp1r_aleniglipron/track_a_generative"
OUT_TAR = REPO / "PRISM_TRACK_A_GFLOWNET_V1_INFERENCE_AUDIT_v1.0.tar.gz"
OUT_SHA = REPO / "PRISM_TRACK_A_GFLOWNET_V1_INFERENCE_AUDIT_v1.0.tar.gz.sha256"
OUT_REPORT = TRACK_A / "gflownet_inference_validation_report.md"

REQUIRED_FILES = {
    "model":            TRACK_A / "gflownet_policy_v1.pt",
    "manifest":         TRACK_A / "gflownet_v1_inference_manifest.json",
    "raw_samples":      TRACK_A / "gflownet_raw_policy_samples.parquet",
    "consensus_scores": TRACK_A / "gflownet_oracle_consensus_scores.parquet",
    "filtered":         TRACK_A / "gflownet_medchem_filtered_candidates.parquet",
    "top100_parquet":   TRACK_A / "gflownet_top_100_candidates.parquet",
    "top100_csv":       TRACK_A / "gflownet_top_100_candidates.csv",
    "top100_md":        TRACK_A / "gflownet_top_100_candidates.md",
    "top25_high":       TRACK_A / "gflownet_top_25_high_confidence.parquet",
    "top25_expl":       TRACK_A / "gflownet_top_25_exploratory.parquet",
    "medchem_report":   TRACK_A / "gflownet_medchem_filter_report.md",
    "baseline_json":    TRACK_A / "gflownet_vs_baseline_comparison.json",
    "baseline_md":      TRACK_A / "gflownet_vs_baseline_comparison.md",
    "audit_json":       TRACK_A / "gflownet_candidate_audit.json",
    "audit_md":         TRACK_A / "gflownet_candidate_audit.md",
    "failure_matrix":   TRACK_A / "gflownet_failure_mode_matrix.md",
    "review_cards_html": TRACK_A / "gflownet_top100_review_cards.html",
    "review_cards_md":   TRACK_A / "gflownet_top100_review_cards.md",
}
PLOT_FILES = [
    "reward_distribution.png",
    "reward_vs_uncertainty.png",
    "reward_vs_clash.png",
    "cryptic_bonus_vs_reward.png",
    "top100_cluster_summary.png",
    "temperature_source_distribution.png",
    "trajectory_entropy_distribution.png",
]
FORBIDDEN_PHRASES = [
    "mechanistic proof",
    "confirmed biological efficacy",
    "wet-lab validated",
    "clinically validated",
    "guaranteed",
    "patient response",
    "clinical outcome",
]


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    findings: list[str] = []
    fail = 0
    warn = 0
    ok = 0
    def OK(msg): nonlocal ok; ok += 1; findings.append(f"- [OK] {msg}")
    def FAIL(msg): nonlocal fail; fail += 1; findings.append(f"- [FAIL] {msg}")
    def WARN(msg): nonlocal warn; warn += 1; findings.append(f"- [WARN] {msg}")

    print(f"=== Phase 9 — validation + packaging ===")

    # 1. All required files exist
    for key, path in REQUIRED_FILES.items():
        if path.is_file():
            OK(f"required file present: {key} ({path.stat().st_size:,} B)")
        else:
            FAIL(f"required file MISSING: {key} -> {path}")

    # 2. Top-100 unique-SMILES == 100
    p = REQUIRED_FILES["top100_parquet"]
    if p.is_file():
        top = pl.read_parquet(p)
        if top.height >= 1:
            n_unique = top.unique(subset=["canonical_smiles"], keep="first").height
            (OK if n_unique == top.height else FAIL)(
                f"top100 unique SMILES = {n_unique} (rows={top.height})"
            )
            # Heavy-atom floor
            if "heavy_atom_count" in top.columns:
                min_ha = top.get_column("heavy_atom_count").min()
                (OK if (min_ha or 0) >= 8 else FAIL)(
                    f"top100 heavy_atom_count floor = {min_ha} (require >=8)"
                )
            # Anchor-family domination
            if "anchor_id" in top.columns:
                ac = top.group_by("anchor_id").len().sort("len", descending=True)
                share = (ac.row(0)[1] / top.height) if top.height else 0
                (OK if share < 0.25 else WARN)(
                    f"top100 max anchor share = {share*100:.1f}% (require <25%)"
                )
            # No OBSERVED labels
            if "epistemic_class" in top.columns:
                bad = top.filter(pl.col("epistemic_class") == "OBSERVED").height
                (OK if bad == 0 else FAIL)(
                    f"top100 OBSERVED labels = {bad} (must be 0)"
                )

    # 3. Consensus parquet: oracle_valid_all
    p = REQUIRED_FILES["consensus_scores"]
    if p.is_file():
        c = pl.read_parquet(p)
        if "oracle_valid_all" in c.columns:
            bad = c.filter(~pl.col("oracle_valid_all")).height
            (OK if bad == 0 else FAIL)(f"consensus rows with oracle_valid_all=false = {bad}")

    # 4. Plots present
    for pf in PLOT_FILES:
        path = TRACK_A / "review_artifacts" / pf
        (OK if path.is_file() else WARN)(f"plot present: {pf}")

    # 5. Forbidden-phrase scan across MD/JSON
    md_to_scan = [
        REQUIRED_FILES["top100_md"], REQUIRED_FILES["medchem_report"],
        REQUIRED_FILES["audit_md"], REQUIRED_FILES["failure_matrix"],
        REQUIRED_FILES["review_cards_md"], REQUIRED_FILES["baseline_md"],
    ]
    hits = 0
    for p in md_to_scan:
        if not p.is_file():
            continue
        text = p.read_text(errors="ignore").lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text:
                FAIL(f"forbidden phrase `{phrase}` in {p.name}")
                hits += 1
    if hits == 0:
        OK("no forbidden overclaim phrases in any narrative MD")

    # 6. Build the tarball.
    include = [
        REQUIRED_FILES["model"], REQUIRED_FILES["manifest"],
        REQUIRED_FILES["raw_samples"], REQUIRED_FILES["consensus_scores"],
        REQUIRED_FILES["filtered"],
        REQUIRED_FILES["top100_parquet"], REQUIRED_FILES["top100_csv"], REQUIRED_FILES["top100_md"],
        REQUIRED_FILES["top25_high"], REQUIRED_FILES["top25_expl"],
        REQUIRED_FILES["medchem_report"],
        REQUIRED_FILES["baseline_json"], REQUIRED_FILES["baseline_md"],
        REQUIRED_FILES["audit_json"], REQUIRED_FILES["audit_md"], REQUIRED_FILES["failure_matrix"],
        REQUIRED_FILES["review_cards_html"], REQUIRED_FILES["review_cards_md"],
    ]
    if fail > 0:
        WARN(f"{fail} validation FAILs; tarball still being built so the audit can be reviewed.")

    with tarfile.open(OUT_TAR, "w:gz") as tar:
        for p in include:
            if p.is_file():
                tar.add(p, arcname=p.relative_to(REPO))
        # plots
        for pf in PLOT_FILES:
            pp = TRACK_A / "review_artifacts" / pf
            if pp.is_file():
                tar.add(pp, arcname=pp.relative_to(REPO))
        # validation report (write below first, then re-add)
    # write the validation report
    summary = {
        "package":          "PRISM_TRACK_A_GFLOWNET_V1_INFERENCE_AUDIT",
        "version":          "v1.0",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_counts":   {"ok": ok, "warn": warn, "fail": fail},
        "verdict":          "PASS" if fail == 0 else "FAIL",
    }
    md = [
        "# GFlowNet v1 — Inference Audit Validation Report",
        "",
        f"Generated: {summary['generated_at_utc']}",
        f"Summary: OK={ok}  WARN={warn}  FAIL={fail}",
        f"Verdict: **{summary['verdict']}**",
        "",
        "## Findings",
        "",
        *findings,
    ]
    OUT_REPORT.write_text("\n".join(md) + "\n")

    # re-archive with the validation report added
    with tarfile.open(OUT_TAR, "a:") if False else tarfile.open(OUT_TAR, "w:gz") as tar:
        # Re-add everything (including the validation report).
        for p in include + [OUT_REPORT]:
            if p.is_file():
                tar.add(p, arcname=p.relative_to(REPO))
        for pf in PLOT_FILES:
            pp = TRACK_A / "review_artifacts" / pf
            if pp.is_file():
                tar.add(pp, arcname=pp.relative_to(REPO))

    tar_sha = sha256_of(OUT_TAR)
    OUT_SHA.write_text(f"{tar_sha}  {OUT_TAR.name}\n")

    print(f"  OK={ok}  WARN={warn}  FAIL={fail}")
    print(f"  -> {OUT_REPORT}")
    print(f"  -> {OUT_TAR}  ({OUT_TAR.stat().st_size:,} B)")
    print(f"  -> {OUT_SHA}")
    print(f"  sha256: {tar_sha}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
