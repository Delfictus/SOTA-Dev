"""Pocket open-probability (P_open) from PRISM multi-stream trajectories.

Quantifies the fraction of time a cryptic or allosteric pocket remains open
across PRISM's 20 independent cryo-thermal streams.  Produces druggability
classifications for downstream decision-making.

Blueprint spec (WT-7)
---------------------
1. Per stream: compute pocket volume at each frame (POVME/fpocket).
2. Binary trajectory: open[t] = V(t) > threshold.
3. Aggregate: P_open = mean(fraction_open), error = bootstrap across streams.
4. Lifetimes: mean open/closed durations.
5. Classify:
       P_open > 0.5  → STABLE_OPEN  (good druggability)
   0.1 < P_open ≤ 0.5 → TRANSIENT    (moderate)
       P_open ≤ 0.1  → RARE_EVENT   (poor)

PRISM leverage: 20 streams = 20 independent estimates + cryo-thermal sampling.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.interfaces import PocketDynamics

logger = logging.getLogger(__name__)

# ── Classification thresholds ─────────────────────────────────────────────

POPEN_STABLE_OPEN_THRESHOLD = 0.5
POPEN_TRANSIENT_THRESHOLD = 0.1

CLASSIFICATION_STABLE_OPEN = "STABLE_OPEN"
CLASSIFICATION_TRANSIENT = "TRANSIENT"
CLASSIFICATION_RARE_EVENT = "RARE_EVENT"

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_VOLUME_THRESHOLD_A3 = 200.0  # Angstrom^3 — pocket "open" if V > this
DEFAULT_N_BOOTSTRAP = 10000
DEFAULT_BOOTSTRAP_CI = 0.95


def classify_druggability(p_open: float) -> str:
    """Classify pocket druggability based on P_open.

    Parameters
    ----------
    p_open : float
        Pocket open probability (0–1).

    Returns
    -------
    str
        One of STABLE_OPEN, TRANSIENT, or RARE_EVENT.
    """
    if p_open > POPEN_STABLE_OPEN_THRESHOLD:
        return CLASSIFICATION_STABLE_OPEN
    elif p_open > POPEN_TRANSIENT_THRESHOLD:
        return CLASSIFICATION_TRANSIENT
    else:
        return CLASSIFICATION_RARE_EVENT


def compute_binary_trajectory(
    volumes: np.ndarray,
    threshold: float = DEFAULT_VOLUME_THRESHOLD_A3,
) -> np.ndarray:
    """Convert volume time-series to binary open/closed trajectory.

    Parameters
    ----------
    volumes : np.ndarray
        Per-frame pocket volumes in Angstrom^3.
    threshold : float
        Volume threshold: open if V > threshold.

    Returns
    -------
    np.ndarray
        Boolean array: True = open, False = closed.
    """
    return np.asarray(volumes, dtype=np.float64) > threshold


def compute_lifetimes(
    binary_traj: np.ndarray,
    dt_ns: float = 0.001,
) -> Tuple[List[float], List[float]]:
    """Compute open and closed state lifetimes from binary trajectory.

    Parameters
    ----------
    binary_traj : np.ndarray
        Boolean array (True = open).
    dt_ns : float
        Time per frame in nanoseconds.

    Returns
    -------
    tuple of (list[float], list[float])
        (open_lifetimes_ns, closed_lifetimes_ns).
    """
    traj = np.asarray(binary_traj, dtype=bool)
    if len(traj) == 0:
        return [], []

    open_lifetimes: List[float] = []
    closed_lifetimes: List[float] = []

    current_state = traj[0]
    run_length = 1

    for i in range(1, len(traj)):
        if traj[i] == current_state:
            run_length += 1
        else:
            duration = run_length * dt_ns
            if current_state:
                open_lifetimes.append(duration)
            else:
                closed_lifetimes.append(duration)
            current_state = traj[i]
            run_length = 1

    # Final run
    duration = run_length * dt_ns
    if current_state:
        open_lifetimes.append(duration)
    else:
        closed_lifetimes.append(duration)

    return open_lifetimes, closed_lifetimes


def compute_volume_autocorrelation(
    volumes: np.ndarray,
    dt_ns: float = 0.001,
) -> float:
    """Estimate the volume autocorrelation time.

    Parameters
    ----------
    volumes : np.ndarray
        Per-frame pocket volumes.
    dt_ns : float
        Time per frame in nanoseconds.

    Returns
    -------
    float
        Estimated autocorrelation time in nanoseconds.
    """
    v = np.asarray(volumes, dtype=np.float64)
    v = v - np.mean(v)
    n = len(v)
    if n < 2 or np.var(v) < 1e-30:
        return 0.0

    # Normalised autocorrelation via FFT
    fft_v = np.fft.fft(v, n=2 * n)
    acf = np.real(np.fft.ifft(fft_v * np.conj(fft_v)))[:n]
    acf /= acf[0]

    # Integrate until first negative crossing
    tau_frames = 0.0
    for i in range(n):
        if acf[i] < 0:
            break
        tau_frames += acf[i]

    return float(tau_frames * dt_ns)


def bootstrap_popen(
    per_stream_fractions: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_BOOTSTRAP_CI,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap P_open and its error from per-stream open fractions.

    Parameters
    ----------
    per_stream_fractions : np.ndarray
        Open fraction for each independent stream.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence interval (e.g. 0.95).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (float, float)
        (p_open, p_open_error) where error is the half-width of the CI.
    """
    fractions = np.asarray(per_stream_fractions, dtype=np.float64)
    n_streams = len(fractions)

    if n_streams == 0:
        return 0.0, 0.0
    if n_streams == 1:
        return float(fractions[0]), 0.0

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(fractions, size=n_streams, replace=True)
        boot_means[i] = np.mean(sample)

    p_open = float(np.mean(fractions))
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    error = (hi - lo) / 2.0

    return p_open, error


def compute_popen(
    stream_volumes: List[np.ndarray],
    pocket_id: int = 0,
    volume_threshold: float = DEFAULT_VOLUME_THRESHOLD_A3,
    dt_ns: float = 0.001,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> PocketDynamics:
    """Compute P_open from multiple trajectory streams.

    This is the main entry point: takes per-stream volume time-series and
    returns a PocketDynamics interface object.

    Parameters
    ----------
    stream_volumes : list of np.ndarray
        Per-stream pocket volume trajectories (Angstrom^3).
        Typically 20 PRISM streams.
    pocket_id : int
        Pocket identifier.
    volume_threshold : float
        Volume threshold for open/closed classification.
    dt_ns : float
        Time per frame in nanoseconds.
    n_bootstrap : int
        Number of bootstrap resamples for error estimation.
    seed : int
        Random seed.

    Returns
    -------
    PocketDynamics
        Interface-compliant pocket dynamics result.

    Raises
    ------
    ValueError
        If no streams are provided.
    """
    if not stream_volumes:
        raise ValueError("No volume streams provided")

    per_stream_fractions = []
    all_open_lifetimes: List[float] = []
    all_closed_lifetimes: List[float] = []
    total_opening_events = 0
    all_volumes = []

    for stream_idx, volumes in enumerate(stream_volumes):
        v = np.asarray(volumes, dtype=np.float64)
        if len(v) == 0:
            logger.warning("Stream %d has no frames, skipping", stream_idx)
            continue

        binary = compute_binary_trajectory(v, volume_threshold)
        frac_open = float(np.mean(binary))
        per_stream_fractions.append(frac_open)

        open_lt, closed_lt = compute_lifetimes(binary, dt_ns)
        all_open_lifetimes.extend(open_lt)
        all_closed_lifetimes.extend(closed_lt)

        # Count opening events (transitions from closed to open)
        transitions = np.diff(binary.astype(int))
        total_opening_events += int(np.sum(transitions == 1))

        all_volumes.append(v)

    if not per_stream_fractions:
        raise ValueError("All streams were empty")

    fractions_arr = np.array(per_stream_fractions)
    p_open, p_open_error = bootstrap_popen(fractions_arr, n_bootstrap, seed=seed)

    mean_open_lt = float(np.mean(all_open_lifetimes)) if all_open_lifetimes else 0.0
    mean_closed_lt = float(np.mean(all_closed_lifetimes)) if all_closed_lifetimes else 0.0

    # Volume autocorrelation from concatenated streams
    if all_volumes:
        concat_v = np.concatenate(all_volumes)
        vol_autocorr = compute_volume_autocorrelation(concat_v, dt_ns)
    else:
        vol_autocorr = 0.0

    classification = classify_druggability(p_open)

    return PocketDynamics(
        pocket_id=pocket_id,
        p_open=p_open,
        p_open_error=p_open_error,
        mean_open_lifetime_ns=mean_open_lt,
        mean_closed_lifetime_ns=mean_closed_lt,
        n_opening_events=total_opening_events,
        druggability_classification=classification,
        volume_autocorrelation_ns=vol_autocorr,
        msm_state_weights=None,
    )


def popen_from_trajectory_files(
    trajectory_paths: List[str],
    pocket_lining_residues: List[int],
    pocket_id: int = 0,
    volume_threshold: float = DEFAULT_VOLUME_THRESHOLD_A3,
    volume_calculator: str = "fpocket",
    dry_run: bool = False,
) -> PocketDynamics:
    """Compute P_open from trajectory files (high-level entry point).

    Parameters
    ----------
    trajectory_paths : list of str
        Paths to multi-stream trajectory files.
    pocket_lining_residues : list of int
        Residue IDs defining the pocket lining.
    pocket_id : int
        Pocket identifier.
    volume_threshold : float
        Volume threshold in Angstrom^3.
    volume_calculator : str
        "fpocket" or "povme" — tool for pocket volume calculation.
    dry_run : bool
        If True, return synthetic result without loading trajectories.

    Returns
    -------
    PocketDynamics
        Pocket dynamics result.
    """
    if dry_run:
        return _generate_dry_run_popen(pocket_id, len(trajectory_paths))

    # Real execution path — requires MDAnalysis/MDTraj + fpocket/POVME
    logger.info(
        "Computing P_open: %d streams, %d lining residues, threshold=%.1f A^3",
        len(trajectory_paths),
        len(pocket_lining_residues),
        volume_threshold,
    )

    stream_volumes = []
    for traj_path in trajectory_paths:
        volumes = _compute_pocket_volumes(
            traj_path, pocket_lining_residues, volume_calculator
        )
        stream_volumes.append(volumes)

    return compute_popen(
        stream_volumes,
        pocket_id=pocket_id,
        volume_threshold=volume_threshold,
    )


def _compute_pocket_volumes(
    trajectory_path: str,
    lining_residues: List[int],
    calculator: str,
) -> np.ndarray:
    """Compute per-frame pocket volumes from a trajectory.

    This is a placeholder for the real implementation that calls
    fpocket or POVME on each frame of the trajectory.
    """
    raise NotImplementedError(
        f"Pocket volume calculation with {calculator!r} requires "
        f"MDAnalysis + {calculator}. Use compute_popen() with "
        f"pre-computed volume arrays, or pass dry_run=True."
    )


def _generate_dry_run_popen(pocket_id: int, n_streams: int) -> PocketDynamics:
    """Generate synthetic P_open result for dry-run / testing."""
    rng = np.random.default_rng(42)
    n_streams = max(n_streams, 5)
    n_frames = 500

    stream_volumes = []
    for i in range(n_streams):
        # Simulate pocket that's mostly open with occasional closures
        base_vol = 350.0 + rng.normal(0, 30, n_frames)
        # Inject closed periods (volume drops)
        for _ in range(rng.integers(1, 5)):
            start = rng.integers(0, n_frames - 20)
            length = rng.integers(5, 30)
            end = min(start + length, n_frames)
            base_vol[start:end] *= 0.3
        stream_volumes.append(base_vol)

    return compute_popen(
        stream_volumes,
        pocket_id=pocket_id,
        volume_threshold=DEFAULT_VOLUME_THRESHOLD_A3,
    )
