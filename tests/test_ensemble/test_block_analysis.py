"""Tests for block_analysis.py — convergence assessment utility."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.ensemble.block_analysis import (
    MIN_BLOCKS,
    BlockAverageResult,
    block_average,
    compute_block_sem,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Core block averaging
# ═══════════════════════════════════════════════════════════════════════════

class TestBlockAverage:
    def test_uncorrelated_data(self):
        """For IID data, block SEM should be close to naive SEM."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        result = block_average(data)
        # Should be within 2x of naive SEM for uncorrelated data
        assert result.sem < result.naive_sem * 2.5

    def test_correlated_data_larger_sem(self):
        """For correlated data, block SEM should exceed naive SEM."""
        rng = np.random.default_rng(42)
        # Highly correlated: random walk
        data = np.cumsum(rng.normal(0, 1, 500))
        result = block_average(data)
        assert result.sem > result.naive_sem

    def test_result_fields(self):
        """All result fields should be populated."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, 100)
        result = block_average(data)

        assert isinstance(result, BlockAverageResult)
        assert abs(result.mean - 5.0) < 1.0
        assert result.sem > 0
        assert result.naive_sem > 0
        assert result.optimal_block_size >= 1
        assert result.n_samples == 100
        assert len(result.block_sizes) > 0
        assert len(result.block_sems) == len(result.block_sizes)
        assert isinstance(result.converged, bool)

    def test_constant_data(self):
        """Constant data should give SEM ≈ 0."""
        data = np.ones(50)
        result = block_average(data)
        assert result.sem < 1e-10
        assert result.mean == 1.0

    def test_too_few_samples_raises(self):
        """Fewer than MIN_BLOCKS samples should raise ValueError."""
        with pytest.raises(ValueError, match="at least"):
            block_average(np.array([1.0, 2.0, 3.0]))

    def test_exactly_min_blocks(self):
        """Exactly MIN_BLOCKS samples should work."""
        data = np.arange(MIN_BLOCKS, dtype=float)
        result = block_average(data)
        assert result.n_samples == MIN_BLOCKS

    def test_mean_matches_numpy(self):
        """Block average mean should match np.mean."""
        rng = np.random.default_rng(123)
        data = rng.normal(10.0, 3.0, 200)
        result = block_average(data)
        assert abs(result.mean - np.mean(data)) < 1e-10

    def test_block_sems_monotonic_tendency(self):
        """For correlated data, SEMs should generally increase with block size."""
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.normal(0, 1, 500))
        result = block_average(data)
        # First SEM should be smaller than last SEM
        assert result.block_sems[0] < result.block_sems[-1]


# ═══════════════════════════════════════════════════════════════════════════
#  Specific block SEM
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeBlockSem:
    def test_block_size_1(self):
        """Block size 1 should give naive SEM."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        sem = compute_block_sem(data, block_size=1)
        expected = np.std(data, ddof=1) / np.sqrt(len(data))
        assert abs(sem - expected) < 1e-10

    def test_large_block_fewer_blocks(self):
        """Larger blocks → fewer blocks → larger SEM variance."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        sem_small = compute_block_sem(data, 2)
        sem_large = compute_block_sem(data, 20)
        # Both should be positive
        assert sem_small > 0
        assert sem_large > 0

    def test_block_larger_than_data(self):
        """Block size larger than data should fall back to naive SEM."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sem = compute_block_sem(data, block_size=10)
        expected = np.std(data, ddof=1) / np.sqrt(len(data))
        assert abs(sem - expected) < 1e-10

    def test_exact_division(self):
        """When data length is exact multiple of block size."""
        data = np.arange(20, dtype=float)
        sem = compute_block_sem(data, 5)
        # 4 blocks of 5
        block_means = np.array([2.0, 7.0, 12.0, 17.0])
        expected = np.std(block_means, ddof=1) / np.sqrt(4)
        assert abs(sem - expected) < 1e-10
