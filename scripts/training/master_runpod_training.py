#!/usr/bin/env python3
"""Master RunPod orchestrator — PRISM-4D teacher v004 + VN-EGNN v001.

Single pod lifecycle, sequential phases. Designed to run as the entrypoint
of a RunPod A100 session started by `runpodctl pod create`.

Bundles have been pre-extracted on the workstation and uploaded to
r2:prism-archive/training-data/pct95-v1/ (372 .npz files, ~30 MB total).
The pod downloads these and only needs to add ESM-2 embeddings before
training.

Expected env vars (injected at pod creation):
    R2_ENDPOINT            https://<account>.r2.cloudflarestorage.com
    R2_ACCESS_KEY          Cloudflare R2 access key
    R2_SECRET_KEY          Cloudflare R2 secret key
    R2_BUCKET              prism-archive
    GIT_REPO               https://github.com/Delfictus/SOTA-Dev.git
    GIT_BRANCH             feat/twin-data-integrity
    BUNDLE_PREFIX          r2:prism-archive/training-data/pct95-v1
    RUNPOD_API_KEY         for self-termination
    RUNPOD_POD_ID          auto-populated by RunPod runtime

Phases:
  0.  Bootstrap (git clone, pip install, rclone config, apt deps)
  1.  Download pre-extracted .npz bundles from R2 (~30 MB, 372 files)
  2.  ESM-2 extraction — run esm2_t33_650M on each target's sequence, save
      embedding as a new "esm2" key inside the .npz
  3.  Teacher v004 cluster-aware LOTO training (gate AUROC ≥ 0.723)
  4.  VN-EGNN v001 training
  5.  Export ONNX for both models
  6.  Upload artifacts to R2 (models/teacher-v004/, models/vnegnn-v001/)
  7.  Self-terminate via RunPod GraphQL API

Exit codes propagate — any phase failure triggers a partial-upload + pod
termination so we don't pay for idle.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
REPO_DIR = WORKSPACE / "Prism4D-bio"
FEATURES_DIR = WORKSPACE / "features"
MODELS_DIR = WORKSPACE / "models"

GIT_REPO = os.environ.get("GIT_REPO", "https://github.com/Delfictus/SOTA-Dev.git")
GIT_BRANCH = os.environ.get("GIT_BRANCH", "feat/twin-data-integrity")
R2_BUCKET = os.environ.get("R2_BUCKET", "prism-archive")
BUNDLE_PREFIX = os.environ.get("BUNDLE_PREFIX",
                                f"r2:{R2_BUCKET}/training-data/pct95-v1")

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")

PHASE_TIMINGS: dict = {}


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


# ─────────────────────────────────────────────────────────────
#  Phase 0 — Bootstrap
# ─────────────────────────────────────────────────────────────

def phase_bootstrap() -> None:
    log_phase("0 / Bootstrap")
    t0 = time.time()

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # OS packages: rclone, mmseqs2 (for cluster splits)
    try:
        subprocess.run(["rclone", "version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        run(["bash", "-c", "curl -fsSL https://rclone.org/install.sh | bash"])
    try:
        subprocess.run(["mmseqs", "version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "mmseqs2"])

    # rclone remote config
    rclone_conf_dir = Path.home() / ".config" / "rclone"
    rclone_conf_dir.mkdir(parents=True, exist_ok=True)
    (rclone_conf_dir / "rclone.conf").write_text(
        f"[r2]\n"
        f"type = s3\n"
        f"provider = Cloudflare\n"
        f"access_key_id = {os.environ['R2_ACCESS_KEY']}\n"
        f"secret_access_key = {os.environ['R2_SECRET_KEY']}\n"
        f"endpoint = {os.environ['R2_ENDPOINT']}\n"
        f"acl = private\n"
    )
    run(["rclone", "lsd", f"r2:{R2_BUCKET}"], timeout=60)

    # Clone repo
    if not REPO_DIR.exists():
        run(["git", "clone", "-b", GIT_BRANCH, "--depth", "1",
             GIT_REPO, str(REPO_DIR)])
    else:
        run(["git", "-C", str(REPO_DIR), "pull", "origin", GIT_BRANCH])

    # Python deps
    run(["pip", "install", "--quiet", "--no-cache-dir",
         "numpy", "scipy", "scikit-learn", "pandas", "pyarrow",
         "torch", "fair-esm", "onnx", "onnxruntime"])

    import torch
    print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    PHASE_TIMINGS["bootstrap"] = time.time() - t0


# ─────────────────────────────────────────────────────────────
#  Phase 1 — Download pre-extracted bundles
# ─────────────────────────────────────────────────────────────

def phase_download_bundles() -> List[str]:
    log_phase("1 / Download pre-extracted .npz bundles from R2")
    t0 = time.time()
    run(["rclone", "copy", BUNDLE_PREFIX, str(FEATURES_DIR),
         "--include", "*.npz", "--transfers", "16", "--quiet"],
        timeout=600)
    # Count
    files = sorted(FEATURES_DIR.glob("*_features.npz"))
    print(f"  Downloaded {len(files)} bundles")
    if len(files) < 300:
        raise RuntimeError(f"Only {len(files)} bundles — expected ~372")
    PHASE_TIMINGS["download"] = time.time() - t0
    return [p.stem.replace("_features", "") for p in files]


# ─────────────────────────────────────────────────────────────
#  Phase 2 — ESM-2 extraction (GPU), write into each .npz
# ─────────────────────────────────────────────────────────────

def phase_esm2(targets: List[str]) -> None:
    log_phase("2 / ESM-2 extraction (esm2_t33_650M, batched on GPU)")
    t0 = time.time()

    import numpy as np
    import torch
    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    batcher = alphabet.get_batch_converter()
    print(f"  loaded ESM-2 t33 on {device}")

    AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")
    for i, t in enumerate(targets):
        npz_path = FEATURES_DIR / f"{t}_features.npz"
        if not npz_path.exists():
            continue
        try:
            d = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            print(f"  [{i+1}/{len(targets)}] {t}: load failed: {e}")
            continue
        if "esm2" in d.files and d["esm2"].shape[0] == d["coords"].shape[0]:
            # Already has ESM-2 (idempotent)
            continue

        seq = str(d["sequence"].item() if d["sequence"].ndim == 0 else d["sequence"])
        if not seq or len(seq) < 10:
            continue
        # Replace non-standard AAs with X (ESM tolerates X)
        seq_clean = "".join(c if c in AA_VALID else "X" for c in seq)

        try:
            _, _, tokens = batcher([(t, seq_clean)])
            tokens = tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[33], return_contacts=False)
            reps = out["representations"][33][0, 1:-1, :].cpu().numpy().astype(np.float32)
        except RuntimeError as e:
            print(f"  [{i+1}/{len(targets)}] {t}: ESM failed ({e}); zero-pad")
            reps = np.zeros((d["coords"].shape[0], 1280), dtype=np.float32)

        # Pad/trim to match coords
        N = d["coords"].shape[0]
        if reps.shape[0] > N:
            reps = reps[:N]
        elif reps.shape[0] < N:
            pad = np.zeros((N - reps.shape[0], 1280), dtype=np.float32)
            reps = np.concatenate([reps, pad], axis=0)

        # Re-save npz with esm2 key
        existing = {k: d[k] for k in d.files}
        existing["esm2"] = reps
        np.savez_compressed(npz_path, **existing)

        if (i + 1) % 25 == 0 or (i + 1) == len(targets):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(targets) - i - 1)
            print(f"  [{i+1}/{len(targets)}] elapsed={elapsed/60:.1f}m ETA={eta/60:.0f}m",
                  flush=True)

    PHASE_TIMINGS["esm2"] = time.time() - t0


# ─────────────────────────────────────────────────────────────
#  Phase 3 — Teacher training
# ─────────────────────────────────────────────────────────────

def phase_train_teacher() -> Path:
    log_phase("3 / Teacher v004 — cluster-aware LOTO training")
    t0 = time.time()
    out = MODELS_DIR / "teacher_v004"
    out.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        [sys.executable, str(REPO_DIR / "scripts" / "training" / "train_teacher.py"),
         "--features-dir", str(FEATURES_DIR),
         "--out-dir", str(out),
         "--epochs", "30",
         "--batch-size", "1024",
         "--lr", "1e-3",
         "--gate-auroc", "0.723",
         "--cluster-cache-path", str(WORKSPACE / "seq_clusters.json")],
        check=False,
    )
    PHASE_TIMINGS["teacher_training"] = time.time() - t0
    if r.returncode not in (0, 2):
        raise RuntimeError(f"train_teacher.py exited {r.returncode}")
    if r.returncode == 2:
        print("  WARN: teacher gate FAILED — continuing to VN-EGNN anyway")
    return out


# ─────────────────────────────────────────────────────────────
#  Phase 4 — VN-EGNN training
# ─────────────────────────────────────────────────────────────

def phase_train_vnegnn() -> Path:
    log_phase("4 / VN-EGNN v001 training")
    t0 = time.time()
    out = MODELS_DIR / "vn_egnn_v001"
    out.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        [sys.executable, str(REPO_DIR / "scripts" / "training" / "vn_egnn" / "train.py"),
         "--features-dir", str(FEATURES_DIR),
         "--out-dir", str(out),
         "--epochs", "200",
         "--lr", "5e-4",
         "--patience", "20",
         "--gate-sr8", "0.55",
         "--cluster-cache-path", str(WORKSPACE / "seq_clusters.json")],
        check=False,
    )
    PHASE_TIMINGS["vn_egnn_training"] = time.time() - t0
    if r.returncode not in (0, 2):
        raise RuntimeError(f"vn_egnn/train.py exited {r.returncode}")
    if r.returncode == 2:
        print("  WARN: VN-EGNN gate FAILED — uploading artifacts anyway for inspection")
    return out


# ─────────────────────────────────────────────────────────────
#  Phase 5 — Upload to R2
# ─────────────────────────────────────────────────────────────

def phase_upload(teacher_dir: Path, vnegnn_dir: Path) -> None:
    log_phase("5 / Upload artifacts to R2")
    t0 = time.time()
    tag = time.strftime("%Y%m%d_%H%M%S")

    def upload(local: Path, remote_prefix: str):
        run(["rclone", "copy", str(local), f"r2:{R2_BUCKET}/{remote_prefix}",
             "--transfers", "8", "--quiet"], timeout=600)
        run(["rclone", "copy", str(local), f"r2:{R2_BUCKET}/{remote_prefix}_{tag}",
             "--transfers", "8", "--quiet"], timeout=600)

    upload(teacher_dir, "models/teacher-v004")
    upload(vnegnn_dir,  "models/vnegnn-v001")

    manifest = {
        "run_timestamp": tag,
        "phases": PHASE_TIMINGS,
        "teacher_dir": "r2://prism-archive/models/teacher-v004",
        "vnegnn_dir": "r2://prism-archive/models/vnegnn-v001",
    }
    man_path = WORKSPACE / "run_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    run(["rclone", "copy", str(man_path), f"r2:{R2_BUCKET}/models/runs/"])

    PHASE_TIMINGS["upload"] = time.time() - t0


# ─────────────────────────────────────────────────────────────
#  Phase 6 — Self-terminate
# ─────────────────────────────────────────────────────────────

def phase_terminate() -> None:
    log_phase("6 / Self-terminate")
    if not RUNPOD_API_KEY or not RUNPOD_POD_ID:
        print("  NOTE: RUNPOD_API_KEY or RUNPOD_POD_ID missing — skipping self-termination.")
        print(f"    runpodctl pod delete {RUNPOD_POD_ID or '<pod-id>'}")
        return

    gql = {
        "query": "mutation($input: PodTerminateInput!) { podTerminate(input: $input) }",
        "variables": {"input": {"podId": RUNPOD_POD_ID}},
    }
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps(gql).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
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
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    try:
        phase_bootstrap()
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
        try:
            if MODELS_DIR.exists():
                run(["rclone", "copy", str(MODELS_DIR),
                     f"r2:{R2_BUCKET}/models/partial_{time.strftime('%Y%m%d_%H%M%S')}",
                     "--transfers", "8", "--quiet"], check=False)
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
