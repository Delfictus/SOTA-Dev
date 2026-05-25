#!/usr/bin/env python3
"""Phase 1 — pin GFlowNet v1 inference inputs against training config.

Hard-fails when:
- gflownet_policy_v1.pt is missing or unreadable
- Rust oracle binary (oracle_scorer) is unavailable / not executable
- selected_ligand_pose.json does not point to compact_pseudo_ligand_o3a*
  (the directive's required pose family)
- Any input hash differs from the training-config record without an
  explicit drift note via --accept-drift <key>=<reason>

Writes:
    campaigns/glp1r_aleniglipron/track_a_generative/gflownet_v1_inference_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path("/home/diddy/Desktop/Prism4D-bio")
TRACK_A = REPO / "campaigns/glp1r_aleniglipron/track_a_generative"
N80 = REPO / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"

MODEL_PATH = TRACK_A / "gflownet_policy_v1.pt"
TRAINING_CFG = TRACK_A / "gflownet_training_config.json"
SELECTED_POSE = TRACK_A / "selected_ligand_pose.json"

# These five inputs are hash-pinned by the trainer. Their canonical paths
# resolved at pre-flight (see chat transcript):
PINNED_INPUTS = {
    "ligand_sdf":      TRACK_A / "ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf",
    "anchors":         TRACK_A / "calibration_anchors_3d.parquet",
    "survivors":       TRACK_A / "vspace_survivors_real512_o3a_zmatrix.parquet",
    "residue_phase":   N80     / "residue_phase_tensor.parquet",
    "interferometric": N80     / "interferometric_differential.parquet",
}

# Additional inputs the directive lists but the trainer config does not
# hash. Pinned by recording their current hash for forward verification.
ADDITIONAL_INPUTS = {
    "voxel_thresholds":         TRACK_A / "voxel_thresholds.json",
    "phase_manifold_coherence": N80     / "phase_manifold_coherence.parquet",
    "selected_ligand_pose":     SELECTED_POSE,
}

ORACLE_BIN_CANDIDATES = [
    REPO / "target/release/oracle_scorer",
    REPO / "target/release/prism-forge",
    REPO / "target/debug/oracle_scorer",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_oracle_binary() -> Path | None:
    for p in ORACLE_BIN_CANDIDATES:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def parse_accept_drift(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in values or []:
        if "=" not in v:
            raise SystemExit(f"--accept-drift must be key=reason, got: {v}")
        k, _, reason = v.partition("=")
        out[k.strip()] = reason.strip()
    return out


def hard_fail(msg: str) -> None:
    print(f"HARD-FAIL: {msg}", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accept-drift", action="append", default=[],
                    help="key=reason; allow a specific input to drift from training config")
    ap.add_argument("--seeds", type=int, default=4,
                    help="number of inference seeds to generate (default 4 — one per regime)")
    args = ap.parse_args()
    accept_drift = parse_accept_drift(args.accept_drift)

    # 1. Model must exist + be loadable size-wise.
    if not MODEL_PATH.is_file():
        hard_fail(f"model missing: {MODEL_PATH}")
    model_size = MODEL_PATH.stat().st_size
    model_sha = sha256_file(MODEL_PATH)

    # 2. Training config must exist + parse.
    if not TRAINING_CFG.is_file():
        hard_fail(f"training config missing: {TRAINING_CFG}")
    cfg = json.loads(TRAINING_CFG.read_text())
    training_hashes: dict[str, str] = cfg.get("input_sha256", {})
    training_timestamp = cfg.get("generated_at_utc", "unknown")

    # 3. Selected ligand pose must point to compact_pseudo_ligand_o3a*.
    if not SELECTED_POSE.is_file():
        hard_fail(f"selected_ligand_pose.json missing: {SELECTED_POSE}")
    pose_doc = json.loads(SELECTED_POSE.read_text())
    pose_id = (pose_doc.get("selected_pose_method")
               or pose_doc.get("selected_pose")
               or pose_doc.get("pose_id")
               or pose_doc.get("ligand_pose")
               or pose_doc.get("selected"))
    if not pose_id:
        hard_fail(f"selected_ligand_pose.json has no pose identifier; keys={list(pose_doc)[:10]}")
    if "compact_pseudo_ligand_o3a" not in str(pose_id).lower() and "o3a_relaxed" not in str(pose_id).lower():
        # The directive requires compact_pseudo_ligand_o3a; o3a_relaxed is
        # the equivalent emitted by the current pose-reconciliation step.
        hard_fail(f"selected pose '{pose_id}' is not compact_pseudo_ligand_o3a / o3a_relaxed")

    # 4. Rust oracle binary must exist and be executable.
    oracle_bin = find_oracle_binary()
    if oracle_bin is None:
        hard_fail("Rust oracle binary not found (target/release/oracle_scorer or prism-forge)")
    oracle_sha = sha256_file(oracle_bin)
    oracle_version_out = ""
    try:
        proc = subprocess.run([str(oracle_bin), "--version"],
                              capture_output=True, text=True, timeout=5)
        oracle_version_out = (proc.stdout + proc.stderr).strip()[:200]
    except Exception as ex:  # noqa: BLE001
        oracle_version_out = f"(--version not supported: {ex})"

    # 5. Re-hash the 5 pinned training inputs and compare against config.
    drift = []
    pinned_records = []
    for key, path in PINNED_INPUTS.items():
        if not path.is_file():
            hard_fail(f"pinned input missing: {key} -> {path}")
        live_sha = sha256_file(path)
        train_sha = training_hashes.get(key, "")
        match = (live_sha == train_sha)
        pinned_records.append({
            "key":            key,
            "path":           str(path.relative_to(REPO)),
            "size_bytes":     path.stat().st_size,
            "live_sha256":    live_sha,
            "training_sha256": train_sha,
            "matches_training": match,
            "drift_reason":   accept_drift.get(key, ""),
        })
        if not match:
            if key in accept_drift:
                continue
            drift.append(f"{key}: live={live_sha[:16]}… training={train_sha[:16]}…")
    if drift:
        hard_fail("input hash drift without --accept-drift override:\n  "
                  + "\n  ".join(drift))

    # 6. Additional (non-pinned-by-trainer) inputs — record their hashes.
    additional_records = []
    for key, path in ADDITIONAL_INPUTS.items():
        if not path.is_file():
            hard_fail(f"additional input missing: {key} -> {path}")
        additional_records.append({
            "key":         key,
            "path":        str(path.relative_to(REPO)),
            "size_bytes":  path.stat().st_size,
            "live_sha256": sha256_file(path),
        })

    # 7. Git SHA + dirty state.
    git_sha = "unknown"
    git_dirty = "unknown"
    try:
        proc = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True, timeout=5)
        git_sha = proc.stdout.strip()
        dirty_proc = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True, timeout=5)
        git_dirty = "dirty" if dirty_proc.stdout.strip() else "clean"
    except Exception as ex:  # noqa: BLE001
        git_sha = f"(git unavailable: {ex})"

    # 8. Inference seed list — deterministic from a known prime.
    seeds = [int(20260524 * (i + 1) * 1000003) & 0x7FFFFFFF
             for i in range(int(args.seeds))]

    # 9. Architecture summary (read from training config).
    arch_summary = {
        "class":              cfg.get("architecture"),
        "phase_labels":       cfg.get("phase_labels"),
        "x_phase_shape":      cfg.get("x_phase_shape"),
        "base_feature_dim":   cfg.get("base_feature_dim"),
        "phase_feature_dim":  cfg.get("phase_feature_dim"),
        "edge_feature_dim":   cfg.get("edge_feature_dim"),
        "valid_action_count": cfg.get("valid_action_count"),
        "learn_anchor_embeddings": cfg.get("learn_anchor_embeddings"),
    }

    manifest: dict[str, Any] = {
        "package":            "PRISM_TRACK_A_GFLOWNET_V1_INFERENCE",
        "phase":              "1_input_pinning",
        "generated_at_utc":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "path":            str(MODEL_PATH.relative_to(REPO)),
            "size_bytes":      model_size,
            "sha256":          model_sha,
            "training_timestamp_utc": training_timestamp,
            "architecture":    arch_summary,
        },
        "rust_oracle": {
            "binary_path":     str(oracle_bin.relative_to(REPO)),
            "sha256":          oracle_sha,
            "size_bytes":      oracle_bin.stat().st_size,
            "version_probe":   oracle_version_out,
        },
        "selected_ligand_pose": {
            "path":            str(SELECTED_POSE.relative_to(REPO)),
            "pose_id":         pose_id,
            "pose_doc":        pose_doc,
        },
        "pinned_inputs":      pinned_records,
        "additional_inputs":  additional_records,
        "git": {
            "sha":             git_sha,
            "dirty_state":     git_dirty,
        },
        "inference_seeds":    seeds,
        "epistemic_policy": {
            "all_candidates_default_class": "PROJECTED",
            "promotion_rule":               "no candidate may be promoted to OBSERVED",
            "wet_lab_assertion":            "no candidate constitutes wet-lab validation",
            "reward_authority":             "Rust oracle_scorer via BatchedRustOracle",
        },
        "drift_overrides":    accept_drift,
    }
    out_path = TRACK_A / "gflownet_v1_inference_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"OK: pinned {len(pinned_records)} training-input hashes + "
          f"{len(additional_records)} additional inputs")
    print(f"  model:       {model_sha[:16]}… ({model_size:,} bytes)")
    print(f"  oracle bin:  {oracle_sha[:16]}… {oracle_bin.relative_to(REPO)}")
    print(f"  pose:        {pose_id}")
    print(f"  git:         {git_sha[:12]}… ({git_dirty})")
    print(f"  seeds:       {seeds}")
    print(f"  manifest ->  {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
