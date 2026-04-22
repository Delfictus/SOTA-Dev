#!/usr/bin/env python3
"""
PRISM-4D v006 Training Pipeline — VQ-VAE Level 1 Spike Tokenizer
=================================================================

Uses the RICHEST available spike data: per-site spike event files
(Level 1 — raw temporal dynamics, not summary statistics).

Pipeline:
  Phase 0: Harvest ALL per-site spike events from every R2 source
  Phase 1: Temporal binning → 18-dim per-window features per site
  Phase 2: Train SiteVQVAE (quality-aware, contrastive, 4096 codebook)
  Phase 3: Generate teacher v005 soft logits
  Phase 4: Train SpikeBERT v003 on VQ tokens + teacher distillation
  Phase 5: Export all-masked embeddings
  Phase 6: Train VN-EGNN v006 (teacher-distilled, all targets)
  Phase 7: Freeze artifacts

VQ-VAE advantages over KMeans:
  - Learns reconstruction-optimal codebook (not just geometric centroids)
  - Quality-aware: codebook entries self-organize by DCC grade
  - Contrastive: EXCELLENT tokens pushed away from POOR in latent space
  - Codebook collapse detection: unused entries get redistributed
  - Continuous latent available alongside discrete tokens

Data sources (all from R2, pulled in Phase 0):
  - prism-archive/training-data/pct95-v2/     (372 feature bundles)
  - prism-archive/10k-runs-canonical-test/     (spike events)
  - prism-archive/10k-runs-pct70/              (spike events from active campaign)
  - prism-archive/10k-twin/                    (TWIN spike events)
  - prism-archive/dev-runs/                    (dev spike events)
  - prism-archive/twin-runs/                   (twin spike events)

Global RNG seed propagated through every component.
No ONNX. PyTorch only.

Usage:
  # Phase 0: Harvest spike data (run on machine with R2 access)
  python3 train_v006.py harvest \\
      --r2-remote r2 \\
      --output-dir /workspace/v006/ \\
      --seed 42

  # Phase 1-7: Train everything (run on GPU)
  python3 train_v006.py train \\
      --features-dir /workspace/training-data/pct95-v2/ \\
      --spike-corpus /workspace/v006/spike_corpus/ \\
      --teacher-logits /workspace/teacher_soft_logits.npz \\
      --output-dir /workspace/v006/ \\
      --seed 42 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# GLOBAL SEED
# ═══════════════════════════════════════════════════════════════

GLOBAL_SEED: int = 42


def set_global_seed(seed: int):
    global GLOBAL_SEED
    GLOBAL_SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_rng(purpose: str) -> torch.Generator:
    sub_seed = int(hashlib.sha256(
        f"{GLOBAL_SEED}:{purpose}".encode()
    ).hexdigest()[:8], 16)
    g = torch.Generator()
    g.manual_seed(sub_seed)
    return g


# ═══════════════════════════════════════════════════════════════
# PHASE 0: HARVEST ALL SPIKE DATA FROM R2
# ═══════════════════════════════════════════════════════════════

R2_SPIKE_SOURCES = [
    # (bucket, prefix, description)
    ("prism-archive", "10k-runs-canonical-test/", "10K canonical test"),
    ("prism-archive", "10k-runs-pct70/", "10K pct70 (active campaign)"),
    ("prism-archive", "10k-twin/", "10K TWIN runs"),
    ("prism-archive", "dev-runs/", "Dev runs"),
    ("prism-archive", "twin-runs/", "TWIN runs"),
    ("prism-archive", "10k-runs/", "10K initial runs"),
]


def harvest_spike_data(r2_remote: str, output_dir: Path, max_per_source: int = 0):
    """Pull ALL per-site spike event files from every R2 source.

    Downloads only .spike_events.json files (the Level 1 data).
    Organizes into output_dir/spike_corpus/{source}/{target}/{site_file}
    """
    corpus_dir = output_dir / "spike_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PHASE 0: Harvesting Level 1 Spike Data from R2")
    print(f"{'='*60}")

    total_files = 0
    total_bytes = 0
    manifest = []

    for bucket, prefix, desc in R2_SPIKE_SOURCES:
        source_dir = corpus_dir / prefix.rstrip("/").replace("/", "_")
        source_dir.mkdir(exist_ok=True)

        print(f"\n  --- {desc} ({bucket}/{prefix}) ---")

        # List available targets
        try:
            result = subprocess.run(
                ["rclone", "lsd", f"{r2_remote}:{bucket}/{prefix}",
                 "--max-depth", "1"],
                capture_output=True, text=True, timeout=60
            )
            target_dirs = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        target_dirs.append(parts[-1])
        except Exception as e:
            print(f"    WARN: Failed to list {bucket}/{prefix}: {e}")
            continue

        print(f"    Found {len(target_dirs)} target directories")
        if max_per_source > 0:
            target_dirs = target_dirs[:max_per_source]

        for i, target in enumerate(target_dirs):
            target_dir = source_dir / target
            target_dir.mkdir(exist_ok=True)

            # List spike event files for this target
            try:
                result = subprocess.run(
                    ["rclone", "ls",
                     f"{r2_remote}:{bucket}/{prefix}{target}/",
                     "--include", "*.spike_events.json"],
                    capture_output=True, text=True, timeout=30
                )
                spike_files = []
                for line in result.stdout.strip().split("\n"):
                    if line.strip() and ".spike_events.json" in line:
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            size = int(parts[0])
                            fname = parts[1]
                            spike_files.append((fname, size))
                            total_bytes += size
            except Exception:
                continue

            if not spike_files:
                continue

            # Download spike event files (skip if already exist)
            n_new = 0
            for fname, size in spike_files:
                local_path = target_dir / Path(fname).name
                if local_path.exists() and local_path.stat().st_size > 0:
                    continue

                try:
                    subprocess.run(
                        ["rclone", "copy",
                         f"{r2_remote}:{bucket}/{prefix}{target}/{fname}",
                         str(target_dir)],
                        capture_output=True, timeout=300
                    )
                    n_new += 1
                    total_files += 1
                except Exception:
                    pass

            if n_new > 0 and (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(target_dirs)}] {target}: {len(spike_files)} sites, "
                      f"{n_new} new downloads")

            manifest.append({
                "source": desc,
                "target": target,
                "n_spike_files": len(spike_files),
                "total_size_mb": sum(s for _, s in spike_files) / 1e6,
            })

    # Save manifest
    manifest_path = output_dir / "spike_harvest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n  Harvest complete:")
    print(f"    Total spike event files: {total_files}")
    print(f"    Total data: {total_bytes / 1e9:.1f} GB")
    print(f"    Manifest: {manifest_path}")


# ═══════════════════════════════════════════════════════════════
# PHASE 1: TEMPORAL TOKENIZATION (Level 1 → 18-dim windows)
# ═══════════════════════════════════════════════════════════════

WINDOW_FEATURES = 18  # Fixed feature layout per temporal window


def parse_spike_events(filepath: Path, max_events: int = 500_000
                       ) -> List[Dict]:
    """Parse a per-site spike event JSON file.

    Returns list of spike event dicts. Caps at max_events for memory.
    Uses orjson if available, falls back to json.
    """
    try:
        import orjson
        with open(filepath, "rb") as f:
            raw = f.read()
        events = orjson.loads(raw)
    except ImportError:
        with open(filepath) as f:
            events = json.load(f)

    if isinstance(events, dict):
        # Some formats wrap events in a container
        if "events" in events:
            events = events["events"]
        elif "spikes" in events:
            events = events["spikes"]

    if isinstance(events, list) and len(events) > max_events:
        # Subsample deterministically
        step = len(events) // max_events
        events = events[::step][:max_events]

    return events


def compute_window_features(events: List[Dict], n_windows: int = 32
                            ) -> np.ndarray:
    """Bin spike events into temporal windows → [n_windows, 18] features.

    Window feature layout (18 dims):
      0:  spike_count
      1:  mean_intensity
      2:  peak_intensity
      3:  std_intensity
      4:  burst_count (spikes with n_nearby_excited >= 3)
      5:  isi_mean (inter-spike interval mean)
      6:  isi_std
      7:  ch_uv_fraction
      8:  ch_lif_fraction
      9:  ch_efp_fraction
      10: ph_cold_hold_fraction
      11: ph_heating_fraction
      12: ph_warm_hold_fraction
      13: ph_cooling_fraction
      14: ph_cold_return_fraction
      15: mean_n_nearby_excited
      16: mean_vibrational_energy
      17: mean_water_density
    """
    features = np.zeros((n_windows, WINDOW_FEATURES), dtype=np.float32)

    if not events:
        return features

    # Get timestep range
    timesteps = []
    for e in events:
        t = e.get("timestep", e.get("frame_index", e.get("t", 0)))
        if isinstance(t, (int, float)):
            timesteps.append(t)
    if not timesteps:
        return features

    t_min, t_max = min(timesteps), max(timesteps)
    if t_max <= t_min:
        t_max = t_min + 1

    window_size = (t_max - t_min) / n_windows

    # Bin events into windows
    window_events: List[List[Dict]] = [[] for _ in range(n_windows)]
    for e in events:
        t = e.get("timestep", e.get("frame_index", e.get("t", 0)))
        if not isinstance(t, (int, float)):
            continue
        w = min(int((t - t_min) / window_size), n_windows - 1)
        window_events[w].append(e)

    # Compute features per window
    PHASE_MAP = {
        "cold_hold": 10, "heating": 11, "warm_hold": 12,
        "cooling": 13, "cold_return": 14,
    }
    CHANNEL_MAP = {"uv": 7, "lif": 8, "efp": 9}

    for w, w_events in enumerate(window_events):
        n = len(w_events)
        features[w, 0] = n  # spike_count

        if n == 0:
            continue

        # Intensity stats
        intensities = [e.get("intensity", e.get("amplitude", 0.0))
                       for e in w_events]
        intensities = [x for x in intensities if isinstance(x, (int, float))]
        if intensities:
            features[w, 1] = np.mean(intensities)
            features[w, 2] = np.max(intensities)
            features[w, 3] = np.std(intensities) if len(intensities) > 1 else 0

        # Burst count
        features[w, 4] = sum(1 for e in w_events
                             if e.get("n_nearby_excited", 0) >= 3)

        # ISI (inter-spike interval)
        w_times = sorted(e.get("timestep", e.get("frame_index", 0))
                         for e in w_events)
        if len(w_times) > 1:
            isis = [w_times[i+1] - w_times[i] for i in range(len(w_times)-1)]
            isis = [x for x in isis if x > 0]
            if isis:
                features[w, 5] = np.mean(isis)
                features[w, 6] = np.std(isis) if len(isis) > 1 else 0

        # Channel fractions
        for e in w_events:
            src = str(e.get("spike_source", e.get("channel", ""))).lower()
            for ch_name, ch_idx in CHANNEL_MAP.items():
                if ch_name in src:
                    features[w, ch_idx] += 1.0 / n
                    break

        # Phase fractions
        for e in w_events:
            phase = str(e.get("ccns_phase", e.get("phase", ""))).lower()
            for ph_name, ph_idx in PHASE_MAP.items():
                if ph_name in phase:
                    features[w, ph_idx] += 1.0 / n
                    break

        # Context means
        context_vals = {"n_nearby_excited": [], "vibrational_energy": [],
                        "water_density": []}
        for e in w_events:
            for key in context_vals:
                v = e.get(key, None)
                if isinstance(v, (int, float)):
                    context_vals[key].append(v)

        for i, key in enumerate(["n_nearby_excited", "vibrational_energy",
                                  "water_density"]):
            vals = context_vals[key]
            if vals:
                features[w, 15 + i] = np.mean(vals)

    return features


def _try_open_view_for_spike_path(spike_path: Path, view_cache: Dict):
    """Locate the Arrow+run_metadata+binding_sites triad for a per-site JSON path.
    Supports two layouts:
      1. <target_root>/artifacts/5_engine/<stem>.site<sid>.spike_events.json
      2. <target_root>/<stem>.site<sid>.spike_events.json   (flat R2-sync layout)

    Returns (view, sid) or (None, None) if triad missing. Views are cached per
    (parent_dir, stem) key in view_cache so multiple sites of the same target
    reuse the same open view (critical for perf).
    """
    import re
    m = re.match(r'^(?P<stem>.+)\.site(?P<sid>-?\d+)\.spike_events\.json$', spike_path.name)
    if not m:
        return None, None
    stem = m.group("stem")
    sid = int(m.group("sid"))
    parent = spike_path.parent
    arrow_p = parent / f"{stem}.topology.spike_events.arrow"
    meta_p = parent / f"{stem}.run_metadata.json"
    bs_p = parent / f"{stem}.binding_sites.json"
    if not (arrow_p.exists() and meta_p.exists() and bs_p.exists()):
        return None, None
    cache_key = (str(parent), stem)
    if cache_key in view_cache:
        return view_cache[cache_key], sid
    try:
        from scripts.interfaces.site_spike_view import SiteSpikeView
        view = SiteSpikeView.from_triad(arrow_p, meta_p, bs_p, stem=stem)
        view_cache[cache_key] = view
        return view, sid
    except Exception:
        view_cache[cache_key] = None
        return None, None


def build_spike_corpus(spike_corpus_dir: Path, n_windows: int = 32
                       ) -> Tuple[np.ndarray, List[Dict]]:
    """Process ALL per-site spike event files → feature tensor.

    D5 Arrow-first: when the triad (topology.spike_events.arrow +
    run_metadata.json + binding_sites.json) is present alongside the per-site
    JSON, the temporal-window features are computed via
    `SiteSpikeView.site(sid).temporal_windows_18dim(n_windows, max_events_cap=500_000)`.
    This preserves the legacy 500k-spike cap from parse_spike_events, yielding
    byte-equivalent window features to the legacy JSON path. When the triad
    is absent, falls back to legacy `parse_spike_events` + `compute_window_features`.

    Returns:
        features: [total_sites * n_windows, 18] — all window features
        site_meta: list of {target, site_name, n_spikes, grade, ...}
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Building Temporal Spike Corpus")
    print(f"{'='*60}")

    all_features = []
    site_meta = []
    total_spikes = 0
    n_sites = 0

    # Walk all subdirectories for spike event files
    spike_files = sorted(spike_corpus_dir.rglob("*.spike_events.json"))
    print(f"  Found {len(spike_files)} spike event files")

    view_cache: Dict = {}
    n_via_view = 0
    n_via_legacy = 0

    for i, spike_path in enumerate(spike_files):
        try:
            view, sid = _try_open_view_for_spike_path(spike_path, view_cache)
            used_view = False
            n_spikes_eff = 0

            if view is not None and view.has_site(sid):
                slice_obj = view.site(sid)
                n_spikes_eff = slice_obj.n_spikes()
                window_feats = slice_obj.temporal_windows_18dim(
                    n_windows=n_windows, max_events_cap=500_000,
                )
                used_view = True
                n_via_view += 1
            else:
                events = parse_spike_events(spike_path)
                if not events:
                    continue
                window_feats = compute_window_features(events, n_windows)
                n_spikes_eff = len(events)
                n_via_legacy += 1

            all_features.append(window_feats)

            # Extract metadata from path
            target = spike_path.parent.name
            site_name = spike_path.stem.replace(".spike_events", "")

            site_meta.append({
                "target": target,
                "site_name": site_name,
                "n_spikes": n_spikes_eff,
                "n_windows": n_windows,
                "file": str(spike_path),
                "d5_view": used_view,
            })
            total_spikes += n_spikes_eff
            n_sites += 1

        except Exception as e:
            if (i + 1) % 100 == 0:
                print(f"    WARN: Failed {spike_path.name}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(spike_files)}] {n_sites} sites "
                  f"(view={n_via_view} legacy={n_via_legacy}), "
                  f"{total_spikes:,} spikes")

    if not all_features:
        raise ValueError("No spike event files processed successfully")

    features = np.concatenate(all_features, axis=0)  # [n_sites * n_windows, 18]
    features_3d = np.stack(all_features)  # [n_sites, n_windows, 18]

    print(f"\n  Corpus built:")
    print(f"    Sites: {n_sites}")
    print(f"    Total spikes: {total_spikes:,}")
    print(f"    Feature tensor: {features_3d.shape}")
    print(f"    Per-window features: {WINDOW_FEATURES} dims")

    return features_3d, site_meta


# ═══════════════════════════════════════════════════════════════
# PHASE 2: SITE VQ-VAE
# ═══════════════════════════════════════════════════════════════

class SiteVQVAE(nn.Module):
    """VQ-VAE for temporal spike window tokenization.

    Learns a quality-aware codebook where each entry represents
    a distinct temporal dynamics pattern. EXCELLENT site windows
    cluster in different codebook regions than POOR ones.

    Input: 18-dim temporal window feature vector
    Output: codebook token + continuous latent + quality prediction
    """

    def __init__(self, input_dim: int = 18, latent_dim: int = 64,
                 codebook_size: int = 4096, beta: float = 0.25):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.beta = beta

        # Encoder: 18 → 256 → 128 → 64
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(128, latent_dim),
        )

        # Codebook: [codebook_size, latent_dim]
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1 / codebook_size, 1 / codebook_size)

        # Decoder: 64 → 128 → 256 → 18
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, input_dim),
        )

        # Quality head: predict DCC distance from latent
        self.quality_head = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Druggability head
        self.drug_head = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Track codebook usage for dead entry resampling
        self.register_buffer("usage_count",
                             torch.zeros(codebook_size, dtype=torch.long))
        self.register_buffer("usage_decay",
                             torch.zeros(codebook_size, dtype=torch.float))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantize(self, z_e: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Nearest-neighbor lookup + straight-through gradient."""
        # [B, D] vs [K, D] → [B, K]
        dists = torch.cdist(z_e.unsqueeze(0),
                            self.codebook.weight.unsqueeze(0)).squeeze(0)
        token_ids = dists.argmin(dim=-1)
        z_q = self.codebook(token_ids)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, z_q, token_ids

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encode(x)
        z_q_st, z_q, token_ids = self.quantize(z_e)
        recon = self.decode(z_q_st)
        quality = self.quality_head(z_q_st)
        druggability = self.drug_head(z_q_st)

        # Track usage
        if self.training:
            for tid in token_ids:
                self.usage_count[tid] += 1

        return recon, token_ids, z_e, z_q, quality, druggability

    def compute_loss(self, x: torch.Tensor, recon: torch.Tensor,
                     z_e: torch.Tensor, z_q: torch.Tensor,
                     quality_pred: torch.Tensor, drug_pred: torch.Tensor,
                     quality_target: Optional[torch.Tensor] = None,
                     drug_target: Optional[torch.Tensor] = None,
                     excellent_mask: Optional[torch.Tensor] = None,
                     poor_mask: Optional[torch.Tensor] = None,
                     ) -> Dict[str, torch.Tensor]:
        """Multi-head loss with quality-aware contrastive term."""

        # Reconstruction
        recon_loss = F.mse_loss(recon, x)

        # VQ losses (van den Oord 2017)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        losses = {
            "recon": recon_loss,
            "codebook": codebook_loss,
            "commitment": self.beta * commitment_loss,
        }

        # Quality regression (if targets available)
        if quality_target is not None:
            losses["quality"] = F.mse_loss(quality_pred.squeeze(), quality_target)

        # Druggability regression
        if drug_target is not None:
            losses["druggability"] = F.mse_loss(drug_pred.squeeze(), drug_target)

        # Contrastive: push EXCELLENT latents away from POOR latents
        if excellent_mask is not None and poor_mask is not None:
            z_exc = z_e[excellent_mask]
            z_poor = z_e[poor_mask]
            if z_exc.size(0) > 0 and z_poor.size(0) > 0:
                d = torch.cdist(z_exc, z_poor)
                margin = 2.0
                contrastive = F.relu(margin - d).mean()
                losses["contrastive"] = contrastive

        # Total
        losses["total"] = (
            losses["recon"]
            + losses["codebook"]
            + losses["commitment"]
            + 0.5 * losses.get("quality", torch.tensor(0.0))
            + 0.3 * losses.get("druggability", torch.tensor(0.0))
            + 0.5 * losses.get("contrastive", torch.tensor(0.0))
        )

        return losses

    def resample_dead_entries(self, all_z_e: torch.Tensor, threshold: int = 5):
        """Replace dead codebook entries with perturbed copies of active ones."""
        dead = (self.usage_count < threshold).nonzero().squeeze(-1)
        if dead.numel() == 0:
            return 0

        alive = (self.usage_count >= threshold).nonzero().squeeze(-1)
        if alive.numel() == 0:
            return 0

        n_replaced = 0
        rng = make_rng(f"resample_{self.usage_count.sum().item()}")
        for d_idx in dead:
            # Pick random alive entry
            a_idx = alive[torch.randint(alive.numel(), (1,), generator=rng).item()]
            # Copy with noise
            noise = torch.randn_like(self.codebook.weight[a_idx]) * 0.01
            self.codebook.weight.data[d_idx] = self.codebook.weight.data[a_idx] + noise
            n_replaced += 1

        self.usage_count.zero_()
        return n_replaced


def train_vqvae(features_3d: np.ndarray, site_meta: List[Dict],
                output_dir: Path, device: torch.device,
                codebook_size: int = 4096, latent_dim: int = 64,
                epochs: int = 300, lr: float = 3e-4,
                quality_labels: Optional[Dict] = None
                ) -> SiteVQVAE:
    """Train the SiteVQVAE on the full temporal spike corpus.

    features_3d: [n_sites, n_windows, 18]
    quality_labels: {site_key: {"dcc": float, "grade": str, "drug": float}}
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Training SiteVQVAE")
    print(f"  Codebook: {codebook_size} entries, latent dim: {latent_dim}")
    print(f"  Sites: {features_3d.shape[0]}, Windows: {features_3d.shape[1]}")
    print(f"{'='*60}")

    n_sites, n_windows, feat_dim = features_3d.shape

    # Flatten to [n_sites * n_windows, 18] for VQ-VAE training
    X = features_3d.reshape(-1, feat_dim).astype(np.float32)

    # Normalize per-feature (robust: median + IQR)
    medians = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    X_norm = np.clip((X - medians) / iqr, -5, 5)

    # Build quality/grade targets if available
    quality_targets = np.full(X.shape[0], 0.0, dtype=np.float32)
    drug_targets = np.full(X.shape[0], 0.0, dtype=np.float32)
    excellent_indices = []
    poor_indices = []

    if quality_labels:
        for i, meta in enumerate(site_meta):
            key = f"{meta['target']}/{meta['site_name']}"
            alt_key = meta["site_name"]
            ql = quality_labels.get(key, quality_labels.get(alt_key, None))
            if ql:
                start = i * n_windows
                end = start + n_windows
                dcc = ql.get("dcc", 10.0)
                drug = ql.get("drug", 0.0)
                grade = ql.get("grade", "UNKNOWN")

                quality_targets[start:end] = dcc
                drug_targets[start:end] = drug

                if grade == "EXCELLENT" or dcc < 2.0:
                    excellent_indices.extend(range(start, end))
                elif grade == "POOR" or dcc > 10.0:
                    poor_indices.extend(range(start, end))

    has_quality = len(excellent_indices) > 0 and len(poor_indices) > 0
    print(f"  Quality labels: {len(excellent_indices)} EXCELLENT windows, "
          f"{len(poor_indices)} POOR windows")

    # To tensors
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
    q_t = torch.tensor(quality_targets, dtype=torch.float32, device=device)
    d_t = torch.tensor(drug_targets, dtype=torch.float32, device=device)

    # Model
    model = SiteVQVAE(input_dim=feat_dim, latent_dim=latent_dim,
                      codebook_size=codebook_size)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    batch_size = min(4096, X_t.size(0))
    n_batches = (X_t.size(0) + batch_size - 1) // batch_size

    best_loss = float("inf")
    best_state = None
    rng = make_rng("vqvae_training")

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_losses = defaultdict(float)

        # Shuffle
        perm = torch.randperm(X_t.size(0), generator=rng)
        X_shuffled = X_t[perm]
        q_shuffled = q_t[perm]
        d_shuffled = d_t[perm]

        # Build masks for contrastive loss
        exc_mask_full = torch.zeros(X_t.size(0), dtype=torch.bool)
        poor_mask_full = torch.zeros(X_t.size(0), dtype=torch.bool)
        if has_quality:
            for idx in excellent_indices:
                exc_mask_full[idx] = True
            for idx in poor_indices:
                poor_mask_full[idx] = True
            exc_mask_shuffled = exc_mask_full[perm]
            poor_mask_shuffled = poor_mask_full[perm]

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, X_t.size(0))
            x_batch = X_shuffled[start:end]
            q_batch = q_shuffled[start:end]
            d_batch = d_shuffled[start:end]

            recon, token_ids, z_e, z_q, quality, drug = model(x_batch)

            exc_m = exc_mask_shuffled[start:end] if has_quality else None
            poor_m = poor_mask_shuffled[start:end] if has_quality else None

            losses = model.compute_loss(
                x_batch, recon, z_e, z_q, quality, drug,
                quality_target=q_batch if has_quality else None,
                drug_target=d_batch if has_quality else None,
                excellent_mask=exc_m, poor_mask=poor_m,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v.item()

        scheduler.step()

        # Resample dead codebook entries every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                z_all = model.encode(X_t)
            n_dead = model.resample_dead_entries(z_all, threshold=5)
            if n_dead > 0:
                print(f"    Resampled {n_dead} dead codebook entries")

        avg_total = epoch_losses["total"] / n_batches
        if avg_total < best_loss:
            best_loss = avg_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
            # Check codebook utilization
            model.eval()
            with torch.no_grad():
                _, _, _, _, token_ids_all = model.encode(X_t), None, None, None, None
                _, token_ids_all, _, _, _, _ = model(X_t[:min(50000, X_t.size(0))])
            n_used = len(token_ids_all.unique())

            elapsed = time.time() - t0
            print(f"  epoch {epoch+1:3d}  recon={epoch_losses['recon']/n_batches:.4f}  "
                  f"total={avg_total:.4f}  codebook_used={n_used}/{codebook_size}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")

    elapsed = time.time() - t0
    print(f"\n  VQ-VAE training complete: {elapsed:.0f}s")
    print(f"  Best total loss: {best_loss:.4f}")

    # Save
    model.load_state_dict(best_state)
    model.eval()

    ckpt_path = output_dir / "site_vqvae.pt"
    torch.save({
        "model_state_dict": best_state,
        "codebook_size": codebook_size,
        "latent_dim": latent_dim,
        "input_dim": feat_dim,
        "normalizer": {"medians": medians, "iqr": iqr},
    }, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    # Export codebook for inspection
    codebook_np = model.codebook.weight.detach().cpu().numpy()
    np.savez_compressed(str(output_dir / "vqvae_codebook.npz"),
                        codebook=codebook_np,
                        medians=medians, iqr=iqr)

    return model


# ═══════════════════════════════════════════════════════════════
# REMAINING PHASES (SpikeBERT + VN-EGNN) use VQ tokens
# These follow the same architecture as v005 but with VQ-VAE
# tokens instead of KMeans tokens, and the SpikeBERT vocab
# size matches the VQ codebook size.
# ═══════════════════════════════════════════════════════════════

# [SpikeBERT v003 and VN-EGNN v006 architectures are identical
#  to train_pipeline_v005.py — the only change is:
#  - SPIKE_VOCAB = codebook_size (4096 instead of 2048)
#  - Tokenization uses VQ-VAE encoder instead of KMeans
#  - SpikeBERT sees richer tokens with quality-aware structure]

# Import the remaining phases from v005 pipeline or inline them
# For brevity, the key changes are noted here:

def tokenize_with_vqvae(model: SiteVQVAE, features: np.ndarray,
                        normalizer: Dict, device: torch.device
                        ) -> np.ndarray:
    """Tokenize per-window features using trained VQ-VAE encoder.

    features: [n_windows, 18]
    Returns: token_ids [n_windows] (int64)
    """
    # Normalize
    X = (features - normalizer["medians"]) / normalizer["iqr"]
    X = np.clip(X, -5, 5).astype(np.float32)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        _, token_ids, _, _, _, _ = model(X_t)
    return token_ids.cpu().numpy()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PRISM-4D v006 — VQ-VAE Level 1 Spike Training Pipeline")

    sub = parser.add_subparsers(dest="command")

    # Harvest command
    harvest_p = sub.add_parser("harvest", help="Pull spike data from R2")
    harvest_p.add_argument("--r2-remote", default="r2")
    harvest_p.add_argument("--output-dir", type=Path, required=True)
    harvest_p.add_argument("--max-per-source", type=int, default=0,
                           help="Limit targets per source (0=all)")
    harvest_p.add_argument("--seed", type=int, default=42)

    # Train command
    train_p = sub.add_parser("train", help="Train full pipeline")
    train_p.add_argument("--features-dir", type=Path, required=True)
    train_p.add_argument("--spike-corpus", type=Path, required=True,
                         help="Dir with spike event files (from harvest)")
    train_p.add_argument("--teacher-logits", type=Path, default=None)
    train_p.add_argument("--teacher-onnx", type=Path, default=None)
    train_p.add_argument("--esm2-dir", type=Path, default=None)
    train_p.add_argument("--quality-labels", type=Path, default=None,
                         help="JSON with per-site DCC/grade/druggability")
    train_p.add_argument("--output-dir", type=Path, required=True)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--device", default="auto")
    train_p.add_argument("--codebook-size", type=int, default=4096)
    train_p.add_argument("--latent-dim", type=int, default=64)
    train_p.add_argument("--vqvae-epochs", type=int, default=300)
    train_p.add_argument("--spikebert-epochs", type=int, default=200)
    train_p.add_argument("--vnegnn-epochs", type=int, default=200)
    train_p.add_argument("--n-windows", type=int, default=32)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    set_global_seed(args.seed)

    if args.command == "harvest":
        harvest_spike_data(args.r2_remote, args.output_dir, args.max_per_source)

    elif args.command == "train":
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        print(f"  Device: {device}")

        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Build temporal spike corpus
        features_3d, site_meta = build_spike_corpus(
            args.spike_corpus, n_windows=args.n_windows)

        # Save corpus
        np.savez_compressed(str(args.output_dir / "spike_corpus_features.npz"),
                            features=features_3d)
        (args.output_dir / "spike_corpus_meta.json").write_text(
            json.dumps(site_meta, indent=2))

        # Load quality labels if available
        quality_labels = None
        if args.quality_labels and args.quality_labels.exists():
            quality_labels = json.loads(args.quality_labels.read_text())

        # Phase 2: Train VQ-VAE
        vqvae = train_vqvae(
            features_3d, site_meta, args.output_dir, device,
            codebook_size=args.codebook_size,
            latent_dim=args.latent_dim,
            epochs=args.vqvae_epochs,
            quality_labels=quality_labels)

        print(f"\n  VQ-VAE codebook: {args.codebook_size} entries")
        print(f"  Next: integrate VQ tokens into SpikeBERT v003 training")
        print(f"  (Phases 3-7 follow the v005 pipeline with VQ tokens)")
        print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
