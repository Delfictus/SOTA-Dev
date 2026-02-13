"""Markov State Model (MSM) builder for ensemble pose weighting.

**Priority: P3 (optional)** — defer unless targeting IDP clients.

Planned workflow (when implemented):
    1. TICA dimensionality reduction on pocket coordinates
    2. k-means clustering in TICA space
    3. MSM estimation (implied timescales validation)
    4. Stationary distribution → weight ensemble poses

Dependencies (not required until implementation):
    - deeptime >= 0.4  (or PyEMMA)
    - scikit-learn (for k-means)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from scripts.interfaces import PocketDynamics

logger = logging.getLogger(__name__)


class MSMNotAvailableError(NotImplementedError):
    """Raised when MSM functionality is requested but not yet implemented."""

    def __init__(self) -> None:
        super().__init__(
            "MSM builder is P3 priority and not yet implemented. "
            "Use compute_popen() for pocket dynamics analysis."
        )


def build_msm(
    trajectories: List[np.ndarray],
    pocket_id: int = 0,
    n_tica_components: int = 3,
    n_clusters: int = 50,
    lag_time_frames: int = 10,
) -> PocketDynamics:
    """Build a Markov State Model from trajectory data.

    .. note:: P3 priority — raises ``MSMNotAvailableError``.

    Parameters
    ----------
    trajectories : list of np.ndarray
        Per-stream coordinate data (n_frames × n_features).
    pocket_id : int
        Pocket identifier.
    n_tica_components : int
        Number of TICA components for dimensionality reduction.
    n_clusters : int
        Number of k-means clusters.
    lag_time_frames : int
        MSM lag time in frames.

    Returns
    -------
    PocketDynamics
        Result with msm_state_weights populated.

    Raises
    ------
    MSMNotAvailableError
        Always, until implemented.
    """
    raise MSMNotAvailableError()


def validate_msm_lag_time(
    implied_timescales: np.ndarray,
    lag_times: np.ndarray,
    n_timescales: int = 5,
) -> bool:
    """Validate MSM lag time selection via implied timescales.

    .. note:: P3 stub — returns False.
    """
    logger.warning("MSM lag time validation not implemented (P3)")
    return False
