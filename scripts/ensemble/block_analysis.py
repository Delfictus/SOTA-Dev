"""Block averaging for convergence assessment.

Implements the block-averaging method of Flyvbjerg & Petersen (JCP 1989) for
estimating the standard error of correlated time-series data.  Used by
ensemble_mmgbsa and interaction_entropy to validate convergence of free-energy
estimates.

References
----------
- Flyvbjerg H, Petersen HG. J Chem Phys. 1989;91(1):461–466.
- Grossfield A, Zuckerman DM. Annu Rep Comput Chem. 2009;5:23–48.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of blocks required for a reliable SEM estimate.
MIN_BLOCKS = 5


@dataclass
class BlockAverageResult:
    """Result of a block-averaging convergence analysis.

    Attributes
    ----------
    mean : float
        Sample mean of the input data.
    sem : float
        Standard error of the mean from block averaging (plateau value).
    naive_sem : float
        Naive SEM assuming uncorrelated samples (σ / √N).
    optimal_block_size : int
        Block size at which the SEM estimate plateaued.
    n_samples : int
        Total number of input data points.
    block_sizes : list of int
        All block sizes evaluated.
    block_sems : list of float
        SEM estimate at each block size.
    converged : bool
        Whether a clear plateau was detected.
    """

    mean: float
    sem: float
    naive_sem: float
    optimal_block_size: int
    n_samples: int
    block_sizes: List[int]
    block_sems: List[float]
    converged: bool


def block_average(
    data: np.ndarray,
    max_block_fraction: float = 0.25,
    min_blocks: int = MIN_BLOCKS,
) -> BlockAverageResult:
    """Compute the block-averaged standard error of the mean.

    Parameters
    ----------
    data : np.ndarray
        1-D array of time-series data (e.g. per-frame MM-GBSA energies).
    max_block_fraction : float
        Largest block size as a fraction of total samples (default 0.25).
    min_blocks : int
        Minimum number of blocks required at each block size (default 5).

    Returns
    -------
    BlockAverageResult
        Convergence analysis with plateau SEM and diagnostics.

    Raises
    ------
    ValueError
        If *data* has fewer than ``min_blocks`` elements.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    n = len(data)

    if n < min_blocks:
        raise ValueError(
            f"Need at least {min_blocks} samples for block averaging, got {n}"
        )

    mean = float(np.mean(data))
    naive_sem = float(np.std(data, ddof=1) / np.sqrt(n))

    max_block = max(1, int(n * max_block_fraction))
    block_sizes: List[int] = []
    block_sems: List[float] = []

    for bs in range(1, max_block + 1):
        n_blocks = n // bs
        if n_blocks < min_blocks:
            break
        # Compute block means
        truncated = data[: n_blocks * bs].reshape(n_blocks, bs)
        block_means = truncated.mean(axis=1)
        sem = float(np.std(block_means, ddof=1) / np.sqrt(n_blocks))
        block_sizes.append(bs)
        block_sems.append(sem)

    if not block_sizes:
        # Fallback: cannot block-average, return naive SEM
        return BlockAverageResult(
            mean=mean,
            sem=naive_sem,
            naive_sem=naive_sem,
            optimal_block_size=1,
            n_samples=n,
            block_sizes=[1],
            block_sems=[naive_sem],
            converged=False,
        )

    # Find plateau: block size where SEM stabilises.
    # Use the last third of the SEM curve and take the median as plateau.
    sems_arr = np.array(block_sems)
    plateau_start = max(1, len(sems_arr) * 2 // 3)
    plateau_region = sems_arr[plateau_start:]

    if len(plateau_region) >= 2:
        plateau_sem = float(np.median(plateau_region))
        # Check convergence: CV of plateau region < 20% → converged
        plateau_cv = float(np.std(plateau_region) / (np.mean(plateau_region) + 1e-30))
        converged = plateau_cv < 0.20
        # Optimal block = first block size where SEM reaches within 10% of plateau
        threshold = plateau_sem * 0.9
        optimal_idx = 0
        for i, s in enumerate(block_sems):
            if s >= threshold:
                optimal_idx = i
                break
        optimal_block_size = block_sizes[optimal_idx]
    else:
        plateau_sem = block_sems[-1]
        optimal_block_size = block_sizes[-1]
        converged = False

    return BlockAverageResult(
        mean=mean,
        sem=plateau_sem,
        naive_sem=naive_sem,
        optimal_block_size=optimal_block_size,
        n_samples=n,
        block_sizes=block_sizes,
        block_sems=block_sems,
        converged=converged,
    )


def compute_block_sem(data: np.ndarray, block_size: int) -> float:
    """Compute SEM for a specific block size.

    Parameters
    ----------
    data : np.ndarray
        1-D array of time-series values.
    block_size : int
        Number of consecutive samples per block.

    Returns
    -------
    float
        Block-averaged SEM.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    n_blocks = len(data) // block_size
    if n_blocks < 2:
        return float(np.std(data, ddof=1) / np.sqrt(len(data)))
    truncated = data[: n_blocks * block_size].reshape(n_blocks, block_size)
    block_means = truncated.mean(axis=1)
    return float(np.std(block_means, ddof=1) / np.sqrt(n_blocks))
