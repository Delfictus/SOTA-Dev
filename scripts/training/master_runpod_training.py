#!/usr/bin/env python3
"""Master RunPod orchestrator — PRISM-4D teacher v004 + VN-EGNN v001.

Single pod lifecycle, sequential phases, with **per-phase resume + continuous
R2 sync**. Designed to run as the entrypoint of a RunPod H100/A100 session.

Resume logic: every phase writes a completion sentinel
(`/workspace/phase_<n>_done.json`). On restart, each phase checks its
sentinel and skips if present. This means a pod crash mid-VN-EGNN does not
re-run ESM-2 or teacher training.

Continuous R2 sync: a background rclone loop mirrors /workspace/ to
`r2:prism-archive/training-runs/runpod-YYYYMMDD/` every 10 minutes,
capturing every *.pt, *.onnx, *.json, *.log. If the pod dies at any point,
the latest artifacts are on R2 within 10 minutes.

Expected env vars:
    R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET,
    GIT_REPO, GIT_BRANCH, BUNDLE_PREFIX,
    RUNPOD_API_KEY, RUNPOD_POD_ID  (optional — for self-terminate)

Phases:
  0. Bootstrap                   → /workspace/phase_0_done.json
  1. Download bundles            → /workspace/features/*.npz
  2. ESM-2 extraction            → in-place add 'esm2' key to each .npz
                                   → /workspace/phase_2_manifest.json (all 372 have esm2)
  3. Teacher training (cluster-aware LOTO) → /workspace/models/teacher_v004/
                                             → phase_3_done after evaluation.json written
  4. VN-EGNN training            → /workspace/models/vn_egnn_v001/
  5. Upload artifacts to R2      → /workspace/models/ → models/*
  6. Self-terminate              → RunPod GraphQL podTerminate

All file paths are under /workspace/ (the persistent network volume). The
continuous sync writes to r2://prism-archive/training-runs/runpod-YYYYMMDD/.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────────────────────────
#  Config — ALL paths under /workspace (persistent volume)
# ─────────────────────────────────────────────────────────────

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
REPO_DIR = WORKSPACE / "Prism4D-bio"
FEATURES_DIR = WORKSPACE / "features"
MODELS_DIR = WORKSPACE / "models"
LOGS_DIR = WORKSPACE / "logs"
SENTINEL_DIR = WORKSPACE / "phase_sentinels"
CLUSTER_CACHE = WORKSPACE / "seq_clusters.json"

GIT_REPO = os.environ.get("GIT_REPO", "https://github.com/Delfictus/SOTA-Dev.git")
GIT_BRANCH = os.environ.get("GIT_BRANCH", "feat/twin-data-integrity")
R2_BUCKET = os.environ.get("R2_BUCKET", "prism-archive")
BUNDLE_PREFIX = os.environ.get("BUNDLE_PREFIX",
                                f"r2:{R2_BUCKET}/training-data/pct95-v1")
RUN_TAG = time.strftime("%Y%m%d_%H%M%S")
RUN_DATE = time.strftime("%Y%m%d")
CONTINUOUS_SYNC_DEST = f"r2:{R2_BUCKET}/training-runs/runpod-{RUN_DATE}"

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")

PHASE_TIMINGS: dict = {}
_SYNC_PID: Optional[int] = None


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True,
        timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check, timeout=timeout)


def log_phase(name: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}\n[{ts}] PHASE: {name}\n{'='*70}", flush=True)


def sentinel_path(n: int) -> Path:
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    return SENTINEL_DIR / f"phase_{n}_done.json"


def phase_already_done(n: int) -> bool:
    return sentinel_path(n).exists()


def mark_phase_done(n: int, **info) -> None:
    data = {"phase": n, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            **info}
    sentinel_path(n).write_text(json.dumps(data, indent=2, default=str))


def sync_phase(local: Path, remote_suffix: str, transfers: int = 8) -> None:
    """Per-phase R2 sync. High-throughput (no bwlimit)."""
    if not local.exists():
        return
    run(["rclone", "copy", str(local), f"r2:{R2_BUCKET}/{remote_suffix}",
         "--transfers", str(transfers), "--checkers", str(transfers),
         "--buffer-size", "128M", "--quiet"],
        timeout=1800, check=False)


# ─────────────────────────────────────────────────────────────
#  Continuous background sync — every 10 min while we're alive
# ─────────────────────────────────────────────────────────────

def start_continuous_sync() -> None:
    """Fork a background rclone-every-10-min loop that captures checkpoints."""
    global _SYNC_PID
    script = WORKSPACE / "continuous_sync.sh"
    script.write_text(f"""#!/bin/bash
set -u
while true; do
  rclone copy {WORKSPACE} {CONTINUOUS_SYNC_DEST} \\
    --include "*.pt" --include "*.onnx" --include "*.json" \\
    --include "*.log" --include "*.npz" --include "*.txt" \\
    --transfers 8 --checkers 8 --buffer-size 128M --quiet \\
    || true
  sleep 600
done
""")
    script.chmod(0o755)
    log_path = LOGS_DIR / "continuous_sync.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        ["bash", str(script)],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _SYNC_PID = p.pid
    print(f"  continuous R2 sync started (PID {p.pid}, dest={CONTINUOUS_SYNC_DEST})")


def stop_continuous_sync() -> None:
    global _SYNC_PID
    if _SYNC_PID:
        try:
            os.killpg(os.getpgid(_SYNC_PID), signal.SIGTERM)
        except ProcessLookupError:
            pass
        _SYNC_PID = None


# ─────────────────────────────────────────────────────────────
#  Phase 0 — Bootstrap
# ─────────────────────────────────────────────────────────────

def phase_bootstrap() -> None:
    log_phase("0 / Bootstrap")
    if phase_already_done(0):
        print("  sentinel present, skipping")
        return
    t0 = time.time()

    for d in (WORKSPACE, FEATURES_DIR, MODELS_DIR, LOGS_DIR, SENTINEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # OS packages
    try:
        subprocess.run(["rclone", "version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        run(["bash", "-c", "curl -fsSL https://rclone.org/install.sh | bash"])
    try:
        subprocess.run(["mmseqs", "version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "mmseqs2"])

    # rclone config
    rclone_conf_dir = Path.home() / ".config" / "rclone"
    rclone_conf_dir.mkdir(parents=True, exist_ok=True)
    (rclone_conf_dir / "rclone.conf").write_text(
        f"[r2]\ntype = s3\nprovider = Cloudflare\n"
        f"access_key_id = {os.environ['R2_ACCESS_KEY']}\n"
        f"secret_access_key = {os.environ['R2_SECRET_KEY']}\n"
        f"endpoint = {os.environ['R2_ENDPOINT']}\n"
        f"acl = private\n"
    )
    run(["rclone", "lsd", f"r2:{R2_BUCKET}"], timeout=60)

    # Repo
    if not REPO_DIR.exists():
        run(["git", "clone", "-b", GIT_BRANCH, "--depth", "1",
             GIT_REPO, str(REPO_DIR)])
    else:
        run(["git", "-C", str(REPO_DIR), "pull", "origin", GIT_BRANCH])

    # Python deps. --break-system-packages needed on Ubuntu 24.04 containers
    # that mark the system Python as externally-managed (PEP 668).
    pip_cmd = ["pip", "install", "--quiet", "--no-cache-dir",
               "--break-system-packages",
               "numpy", "scipy", "scikit-learn", "pandas", "pyarrow",
               "fair-esm", "onnx", "onnxruntime"]
    try:
        run(pip_cmd)
    except subprocess.CalledProcessError:
        # Older containers without PEP 668 don't recognize the flag
        pip_cmd.remove("--break-system-packages")
        run(pip_cmd)

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  gpu={gpu}")

    PHASE_TIMINGS["bootstrap"] = time.time() - t0
    mark_phase_done(0, gpu=gpu)


# ─────────────────────────────────────────────────────────────
#  Phase 1 — Download bundles from R2
# ─────────────────────────────────────────────────────────────

def phase_download_bundles() -> List[str]:
    log_phase("1 / Download pre-extracted .npz bundles from R2")
    # Always run this — it's a resumable rclone copy (skips matched files)
    t0 = time.time()
    run(["rclone", "copy", BUNDLE_PREFIX, str(FEATURES_DIR),
         "--include", "*.npz", "--transfers", "16", "--checkers", "16",
         "--buffer-size", "128M", "--quiet"],
        timeout=1800)
    files = sorted(FEATURES_DIR.glob("*_features.npz"))
    print(f"  bundles available: {len(files)}")
    if len(files) < 300:
        raise RuntimeError(f"Only {len(files)} bundles — expected ~372")
    PHASE_TIMINGS["download"] = time.time() - t0
    mark_phase_done(1, n_bundles=len(files))
    return [p.stem.replace("_features", "") for p in files]


# ─────────────────────────────────────────────────────────────
#  Phase 2 — ESM-2 extraction, idempotent per-target
# ─────────────────────────────────────────────────────────────

def phase_esm2(targets: List[str]) -> None:
    log_phase("2 / ESM-2 extraction (esm2_t33_650M)")
    if phase_already_done(2):
        print("  sentinel present, skipping")
        return
    t0 = time.time()

    import numpy as np
    import torch
    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    batcher = alphabet.get_batch_converter()
    print(f"  ESM-2 loaded on {device}")

    AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")
    n_added = n_cached = 0

    for i, t in enumerate(targets):
        npz_path = FEATURES_DIR / f"{t}_features.npz"
        if not npz_path.exists():
            continue
        try:
            d = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            print(f"  [{i+1}/{len(targets)}] {t}: load failed: {e}")
            continue

        # Idempotent: skip if already has esm2 matching coord count
        if ("esm2" in d.files and d["esm2"].ndim == 2
                and d["esm2"].shape[0] == d["coords"].shape[0]):
            n_cached += 1
            continue

        seq = str(d["sequence"].item() if d["sequence"].ndim == 0 else d["sequence"])
        if not seq or len(seq) < 10:
            continue
        seq_clean = "".join(c if c in AA_VALID else "X" for c in seq)

        try:
            _, _, tokens = batcher([(t, seq_clean)])
            tokens = tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[33], return_contacts=False)
            reps = out["representations"][33][0, 1:-1, :].cpu().numpy().astype(np.float32)
        except RuntimeError as e:
            print(f"  [{i+1}/{len(targets)}] {t}: ESM RuntimeError ({str(e)[:80]}); zero-pad")
            reps = np.zeros((d["coords"].shape[0], 1280), dtype=np.float32)

        N = d["coords"].shape[0]
        if reps.shape[0] > N:
            reps = reps[:N]
        elif reps.shape[0] < N:
            pad = np.zeros((N - reps.shape[0], 1280), dtype=np.float32)
            reps = np.concatenate([reps, pad], axis=0)

        existing = {k: d[k] for k in d.files}
        existing["esm2"] = reps
        np.savez_compressed(npz_path, **existing)
        n_added += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(targets):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(targets) - i - 1)
            print(f"  [{i+1}/{len(targets)}] added={n_added} cached={n_cached} "
                  f"elapsed={elapsed/60:.1f}m ETA={eta/60:.0f}m", flush=True)

    # Manifest
    manifest = {
        "n_targets": len(targets),
        "n_added": n_added,
        "n_already_cached": n_cached,
        "n_missing": sum(1 for t in targets
                         if not (FEATURES_DIR / f"{t}_features.npz").exists()),
    }
    manifest_path = WORKSPACE / "phase_2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  ESM-2 manifest: {manifest}")

    # Per-phase R2 sync — upload the enriched .npz bundles
    sync_phase(FEATURES_DIR, "training-data/pct95-v1-esm2", transfers=16)

    PHASE_TIMINGS["esm2"] = time.time() - t0
    mark_phase_done(2, **manifest)


# ─────────────────────────────────────────────────────────────
#  Phase 3 — Teacher training (cluster-aware LOTO)
# ─────────────────────────────────────────────────────────────

def phase_train_teacher() -> Path:
    log_phase("3 / Teacher v004 — cluster-aware LOTO training")
    out = MODELS_DIR / "teacher_v004"
    out.mkdir(parents=True, exist_ok=True)

    if phase_already_done(3) and (out / "evaluation.json").exists():
        print("  sentinel present + evaluation.json exists, skipping")
        return out

    t0 = time.time()
    log_file = LOGS_DIR / f"teacher_{RUN_TAG}.log"
    with open(log_file, "w") as logf:
        r = subprocess.run(
            [sys.executable, str(REPO_DIR / "scripts" / "training" / "train_teacher.py"),
             "--features-dir", str(FEATURES_DIR),
             "--out-dir", str(out),
             "--epochs", "30",
             "--batch-size", "1024",
             "--lr", "1e-3",
             "--gate-auroc", "0.723",
             "--cluster-cache-path", str(CLUSTER_CACHE)],
            stdout=logf, stderr=subprocess.STDOUT, check=False,
        )
    PHASE_TIMINGS["teacher_training"] = time.time() - t0
    sync_phase(out, "models/teacher-v004", transfers=8)
    sync_phase(log_file, "training-runs/logs", transfers=2)

    if r.returncode not in (0, 2):
        raise RuntimeError(f"train_teacher.py exited {r.returncode} (see {log_file})")
    if r.returncode == 2:
        print("  WARN: teacher gate FAILED — continuing to VN-EGNN anyway")
    mark_phase_done(3, gate_passed=(r.returncode == 0))
    return out


# ─────────────────────────────────────────────────────────────
#  Phase 4 — VN-EGNN training
# ─────────────────────────────────────────────────────────────

def phase_train_vnegnn() -> Path:
    log_phase("4 / VN-EGNN v001 training")
    out = MODELS_DIR / "vn_egnn_v001"
    out.mkdir(parents=True, exist_ok=True)

    if phase_already_done(4) and (out / "evaluation.json").exists():
        print("  sentinel present + evaluation.json exists, skipping")
        return out

    t0 = time.time()
    log_file = LOGS_DIR / f"vnegnn_{RUN_TAG}.log"
    with open(log_file, "w") as logf:
        r = subprocess.run(
            [sys.executable, str(REPO_DIR / "scripts" / "training" / "vn_egnn" / "train.py"),
             "--features-dir", str(FEATURES_DIR),
             "--out-dir", str(out),
             "--epochs", "200",
             "--lr", "5e-4",
             "--patience", "20",
             "--gate-sr8", "0.55",
             "--cluster-cache-path", str(CLUSTER_CACHE)],
            stdout=logf, stderr=subprocess.STDOUT, check=False,
        )
    PHASE_TIMINGS["vn_egnn_training"] = time.time() - t0
    sync_phase(out, "models/vnegnn-v001", transfers=8)
    sync_phase(log_file, "training-runs/logs", transfers=2)

    if r.returncode not in (0, 2):
        raise RuntimeError(f"vn_egnn/train.py exited {r.returncode} (see {log_file})")
    if r.returncode == 2:
        print("  WARN: VN-EGNN gate FAILED — uploading for inspection")
    mark_phase_done(4, gate_passed=(r.returncode == 0))
    return out


# ─────────────────────────────────────────────────────────────
#  Phase 5 — Final upload (idempotent; ensures stamped copy)
# ─────────────────────────────────────────────────────────────

def phase_upload(teacher_dir: Path, vnegnn_dir: Path) -> None:
    log_phase("5 / Final artifact upload (stamped)")
    t0 = time.time()
    # Canonical latest paths
    sync_phase(teacher_dir, "models/teacher-v004", transfers=8)
    sync_phase(vnegnn_dir,  "models/vnegnn-v001", transfers=8)
    # Run-stamped history
    sync_phase(teacher_dir, f"models/teacher-v004_{RUN_TAG}", transfers=8)
    sync_phase(vnegnn_dir,  f"models/vnegnn-v001_{RUN_TAG}", transfers=8)

    manifest = {
        "run_timestamp": RUN_TAG,
        "phases": PHASE_TIMINGS,
        "teacher_dir": "r2://prism-archive/models/teacher-v004",
        "vnegnn_dir": "r2://prism-archive/models/vnegnn-v001",
    }
    (WORKSPACE / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    sync_phase(WORKSPACE / "run_manifest.json", "models/runs", transfers=1)
    PHASE_TIMINGS["upload"] = time.time() - t0
    mark_phase_done(5)


# ─────────────────────────────────────────────────────────────
#  Phase 6 — Self-terminate
# ─────────────────────────────────────────────────────────────

def phase_terminate() -> None:
    log_phase("6 / Self-terminate")
    stop_continuous_sync()
    if not RUNPOD_API_KEY or not RUNPOD_POD_ID:
        print("  RUNPOD_API_KEY or RUNPOD_POD_ID missing — manual delete required:")
        print(f"    runpodctl pod delete {RUNPOD_POD_ID or '<pod-id>'}")
        return
    gql = {
        "query": "mutation($input: PodTerminateInput!) { podTerminate(input: $input) }",
        "variables": {"input": {"podId": RUNPOD_POD_ID}},
    }
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps(gql).encode("utf-8"),
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"  Termination request sent: {r.read().decode()[:200]}")
    except urllib.error.URLError as e:
        print(f"  Termination request failed: {e}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main() -> int:
    start = time.time()
    print(f"PRISM-4D RunPod orchestrator starting at "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}  run_tag={RUN_TAG}", flush=True)

    # Keep orchestrator log under /workspace/logs/
    try:
        phase_bootstrap()
        start_continuous_sync()  # fires now — captures everything hereafter

        targets = phase_download_bundles()
        if len(targets) < 300:
            raise RuntimeError(f"Only {len(targets)} bundles — abort")
        phase_esm2(targets)
        teacher_dir = phase_train_teacher()
        vnegnn_dir = phase_train_vnegnn()
        phase_upload(teacher_dir, vnegnn_dir)
    except Exception as e:
        print(f"\nFATAL: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        PHASE_TIMINGS["fatal_error"] = str(e)
        # Emergency partial-artifact sync before terminate
        try:
            sync_phase(MODELS_DIR, f"models/partial_{RUN_TAG}", transfers=8)
        except Exception:
            pass
        phase_terminate()
        return 1

    total_m = (time.time() - start) / 60
    print(f"\n  Total pipeline time: {total_m:.1f} minutes")
    print(f"  Phase timings: {json.dumps(PHASE_TIMINGS, indent=2, default=str)}")
    phase_terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
