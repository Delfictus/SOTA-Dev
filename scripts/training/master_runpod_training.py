#!/usr/bin/env python3
"""Master RunPod orchestrator — PRISM-4D teacher v004 + VN-EGNN v001.

Single pod lifecycle, sequential phases. Designed to run as the entrypoint
of a RunPod A100 session started by `runpodctl pod create`.

Expected env vars (injected at pod creation):
    R2_ENDPOINT            https://<account>.r2.cloudflarestorage.com
    R2_ACCESS_KEY          Cloudflare R2 access key
    R2_SECRET_KEY          Cloudflare R2 secret key
    R2_BUCKET              prism-archive
    D1_WORKER_URL          https://prism-feature-pipeline.is-0b9.workers.dev
    GIT_REPO               https://github.com/Delfictus/SOTA-Dev.git
    GIT_BRANCH             feat/twin-data-integrity
    RUNPOD_API_KEY         for self-termination
    RUNPOD_POD_ID          auto-populated by RunPod runtime

Phases:
  0.  Bootstrap (git clone, pip install, rclone config, ESM-2 warm)
  1.  Feature extraction for all pct95 targets (dcc_grade != POOR)
  2.  Teacher v004 LOTO training
  3.  VN-EGNN v001 training
  4.  Upload artifacts to R2 (models/teacher-v004/, models/vnegnn-v001/)
  5.  Self-terminate

Exit codes propagate — any phase failure halts the pipeline and terminates
the pod (you pay for compute up to the failure, not for idle).
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
REPO_DIR = WORKSPACE / "Prism4D-bio"
CACHE_DIR = WORKSPACE / "cache"
FEATURES_DIR = WORKSPACE / "features"
MODELS_DIR = WORKSPACE / "models"

GIT_REPO = os.environ.get("GIT_REPO", "https://github.com/Delfictus/SOTA-Dev.git")
GIT_BRANCH = os.environ.get("GIT_BRANCH", "feat/twin-data-integrity")
D1_WORKER_URL = os.environ.get("D1_WORKER_URL", "https://prism-feature-pipeline.is-0b9.workers.dev")
R2_BUCKET = os.environ.get("R2_BUCKET", "prism-archive")

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")

# Phase timings are logged but not enforced
PHASE_TIMINGS: dict = {}


# ─────────────────────────────────────────────────────────────
#  Shell helper
# ─────────────────────────────────────────────────────────────

def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None,
        check: bool = True, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=check, timeout=timeout)


def log_phase(name: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}\n[{ts}] PHASE: {name}\n{'='*70}", flush=True)


def api_get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 prism4d-orchestrator"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ─────────────────────────────────────────────────────────────
#  Phase 0 — Bootstrap
# ─────────────────────────────────────────────────────────────

def phase_bootstrap() -> None:
    log_phase("0 / Bootstrap")
    t0 = time.time()

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) OS packages: rclone, mkdssp
    try:
        subprocess.run(["rclone", "version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        run(["bash", "-c", "curl -fsSL https://rclone.org/install.sh | bash"])
    try:
        subprocess.run(["mkdssp", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "dssp", "freesasa"])

    # 2) rclone remote config
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

    # 3) Clone repo
    if not REPO_DIR.exists():
        run(["git", "clone", "-b", GIT_BRANCH, "--depth", "1", GIT_REPO, str(REPO_DIR)])
    else:
        run(["git", "-C", str(REPO_DIR), "pull", "origin", GIT_BRANCH])

    # 4) Python deps
    run(["pip", "install", "--quiet", "--no-cache-dir",
         "numpy", "scipy", "scikit-learn", "pandas", "pyarrow",
         "torch", "fair-esm", "prody", "freesasa", "onnx", "onnxruntime"])

    # 5) Smoke-test torch CUDA
    import torch
    print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    PHASE_TIMINGS["bootstrap"] = time.time() - t0


# ─────────────────────────────────────────────────────────────
#  Phase 1 — Feature extraction
# ─────────────────────────────────────────────────────────────

def _extract_one(target: str) -> Optional[str]:
    """Worker: extract features for one target. ESM-2 runs on GPU inside this process."""
    sys.path.insert(0, str(REPO_DIR / "scripts" / "training"))
    from feature_extractor import extract_target, save_bundle

    out_path = FEATURES_DIR / f"{target}.features.npz"
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return target  # already extracted

    bundle = extract_target(target, CACHE_DIR, compute_esm=True)
    if bundle is None:
        return None
    save_bundle(bundle, FEATURES_DIR)
    # Drop the per-target cache dir to save disk (parquets are large)
    tdir = CACHE_DIR / target
    if tdir.exists():
        shutil.rmtree(tdir, ignore_errors=True)
    return target


def phase_extract_features() -> List[str]:
    log_phase("1 / Feature extraction")
    t0 = time.time()

    # Pull valid targets from D1 (non-POOR)
    data = api_get(f"{D1_WORKER_URL}/dcc")
    valid = sorted({r["target"] for r in data.get("records", [])
                    if r.get("dcc_grade") != "POOR"})
    print(f"  Valid targets (non-POOR): {len(valid)}")

    # Serial execution — ESM-2 on GPU serializes regardless; parquet I/O benefits
    # from the GPU being idle between ESM calls. Running in-process avoids
    # reloading the ESM weights for each target.
    sys.path.insert(0, str(REPO_DIR / "scripts" / "training"))
    from feature_extractor import extract_target, save_bundle

    ok, failed = [], []
    for i, t in enumerate(valid):
        out_path = FEATURES_DIR / f"{t}.features.npz"
        if out_path.exists() and out_path.stat().st_size > 10_000:
            ok.append(t)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(valid)}] cached", flush=True)
            continue
        try:
            bundle = extract_target(t, CACHE_DIR, compute_esm=True)
        except Exception as e:
            print(f"  [{i+1}/{len(valid)}] {t} FAILED: {e}", flush=True)
            failed.append(t)
            continue
        if bundle is None:
            failed.append(t)
            print(f"  [{i+1}/{len(valid)}] {t} SKIPPED (pipeline returned None)", flush=True)
            continue
        save_bundle(bundle, FEATURES_DIR)
        ok.append(t)
        # Purge per-target cache to stay under 200 GB
        tdir = CACHE_DIR / t
        if tdir.exists():
            shutil.rmtree(tdir, ignore_errors=True)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(valid) - i - 1)
            print(f"  [{i+1}/{len(valid)}] ok={len(ok)} fail={len(failed)} "
                  f"elapsed={elapsed/60:.1f}m ETA={eta/60:.0f}m", flush=True)

    print(f"  DONE: {len(ok)} extracted / {len(failed)} failed  ({(time.time()-t0)/60:.1f}m)")
    PHASE_TIMINGS["feature_extraction"] = time.time() - t0
    return ok


# ─────────────────────────────────────────────────────────────
#  Phase 2 — Teacher training
# ─────────────────────────────────────────────────────────────

def phase_train_teacher() -> Path:
    log_phase("2 / Teacher v004 — LOTO training")
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
         "--gate-auroc", "0.723"],
        check=False,
    )
    PHASE_TIMINGS["teacher_training"] = time.time() - t0
    if r.returncode not in (0, 2):
        raise RuntimeError(f"train_teacher.py exited {r.returncode}")
    # exit 2 = gate-fail, still continue to VN-EGNN (we still want the export)
    if r.returncode == 2:
        print("  WARN: teacher gate FAILED — continuing to VN-EGNN phase anyway")
    return out


# ─────────────────────────────────────────────────────────────
#  Phase 3 — VN-EGNN training
# ─────────────────────────────────────────────────────────────

def phase_train_vnegnn() -> Path:
    log_phase("3 / VN-EGNN v001 training")
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
         "--gate-sr8", "0.55"],
        check=False,
    )
    PHASE_TIMINGS["vn_egnn_training"] = time.time() - t0
    if r.returncode not in (0, 2):
        raise RuntimeError(f"vn_egnn/train.py exited {r.returncode}")
    if r.returncode == 2:
        print("  WARN: VN-EGNN gate FAILED — uploading artifacts anyway for inspection")
    return out


# ─────────────────────────────────────────────────────────────
#  Phase 4 — Upload to R2
# ─────────────────────────────────────────────────────────────

def phase_upload(teacher_dir: Path, vnegnn_dir: Path) -> None:
    log_phase("4 / Upload artifacts to R2")
    t0 = time.time()

    run_tag = time.strftime("%Y%m%d_%H%M%S")

    def upload(local: Path, remote_prefix: str):
        dest = f"r2:{R2_BUCKET}/{remote_prefix}"
        run(["rclone", "copy", str(local), dest, "--transfers", "8", "--quiet"])
        # Also save a run-stamped mirror for history
        stamped = f"r2:{R2_BUCKET}/{remote_prefix}_{run_tag}"
        run(["rclone", "copy", str(local), stamped, "--transfers", "8", "--quiet"])

    upload(teacher_dir, "models/teacher-v004")
    upload(vnegnn_dir,  "models/vnegnn-v001")

    # Upload feature_stats so downstream inference can normalize identically
    stats = teacher_dir / "feature_stats.json"
    if stats.exists():
        run(["rclone", "copy", str(stats), f"r2:{R2_BUCKET}/models/teacher-v004"])

    # Upload a summary manifest
    manifest = {
        "run_timestamp": run_tag,
        "phases": PHASE_TIMINGS,
        "teacher_dir": "r2://prism-archive/models/teacher-v004",
        "vnegnn_dir": "r2://prism-archive/models/vnegnn-v001",
    }
    man_path = WORKSPACE / "run_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    run(["rclone", "copy", str(man_path), f"r2:{R2_BUCKET}/models/runs/"])

    PHASE_TIMINGS["upload"] = time.time() - t0


# ─────────────────────────────────────────────────────────────
#  Phase 5 — Self-terminate
# ─────────────────────────────────────────────────────────────

def phase_terminate() -> None:
    log_phase("5 / Self-terminate")
    if not RUNPOD_API_KEY or not RUNPOD_POD_ID:
        print("  NOTE: RUNPOD_API_KEY or RUNPOD_POD_ID missing — skipping self-termination.")
        print("  Pod will continue to incur charges. Terminate manually:")
        print(f"    runpodctl pod delete {RUNPOD_POD_ID or '<pod-id>'}")
        return

    # Post to RunPod GraphQL podTerminate mutation
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
            print(f"  Termination request sent. Response: {r.read().decode()[:200]}")
    except urllib.error.URLError as e:
        print(f"  Termination request failed: {e}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main() -> int:
    start = time.time()
    print(f"PRISM-4D training orchestrator starting at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    try:
        phase_bootstrap()
        ok_targets = phase_extract_features()
        if len(ok_targets) < 50:
            raise RuntimeError(f"Only {len(ok_targets)} targets extracted — abort")
        teacher_dir = phase_train_teacher()
        vnegnn_dir = phase_train_vnegnn()
        phase_upload(teacher_dir, vnegnn_dir)
    except Exception as e:
        print(f"\nFATAL: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        PHASE_TIMINGS["fatal_error"] = str(e)
        # Still upload whatever we have
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
