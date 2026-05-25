"""DSTW-aligned entropy router for multi-scaffold Track A training."""

from __future__ import annotations

import json
import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor


PHASES: tuple[str, ...] = ("cold_hold", "ramp_up", "warm_hold", "ramp_down", "cold_return")
DEFAULT_REACTION_PRIOR = math.log(3.0)
DEFAULT_PHASE_PRIOR = math.log(float(len(PHASES)))


@dataclass
class PhaseStats:
    """Per-phase exploration statistics for one scaffold."""

    phase_voxel_counts: dict[str, float] = field(default_factory=lambda: {phase: 0.0 for phase in PHASES})

    @property
    def phase_coverage_entropy(self) -> float:
        total = sum(self.phase_voxel_counts.values())
        if total <= 0.0:
            return DEFAULT_PHASE_PRIOR
        probabilities = [count / total for count in self.phase_voxel_counts.values() if count > 0.0]
        return -sum(probability * math.log(probability + 1.0e-10) for probability in probabilities)


@dataclass
class ChannelStats:
    """Track Resonate-and-Fire channel activation balance."""

    total_channel_a_activations: int = 0
    total_channel_b_activations: int = 0

    @property
    def rf_activation_ratio(self) -> float:
        total = max(self.total_channel_b_activations, 1)
        if self.total_channel_b_activations == 0:
            return 0.5
        return self.total_channel_a_activations / float(total)


@dataclass
class PhaseResolvedLockStats:
    """Track lock-clash strength over the five CCNS phases."""

    lock_clash_per_phase: dict[str, float] = field(default_factory=lambda: {phase: 0.0 for phase in PHASES})
    trajectory_count: int = 0

    @property
    def phase_resolved_lock_score(self) -> float:
        if self.trajectory_count == 0:
            return 0.0
        means = [
            self.lock_clash_per_phase[phase] / max(self.trajectory_count, 1)
            for phase in PHASES
        ]
        return min(means)

    @property
    def hysteresis_variance(self) -> float:
        if self.trajectory_count == 0:
            return 1.0
        cold_hold = self.lock_clash_per_phase["cold_hold"] / max(self.trajectory_count, 1)
        cold_return = self.lock_clash_per_phase["cold_return"] / max(self.trajectory_count, 1)
        ramp_up = self.lock_clash_per_phase["ramp_up"] / max(self.trajectory_count, 1)
        return abs(cold_return - cold_hold) / max(ramp_up, 0.01)


@dataclass
class ScaffoldStats:
    """Per-scaffold running statistics, updated after oracle scoring."""

    name: str
    total_trajectories: int = 0
    novel_embedding_count: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    reaction_class_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    lock_clash_sum: float = 0.0
    last_50_rewards: list[float] = field(default_factory=list)
    phase_stats: PhaseStats = field(default_factory=PhaseStats)
    channel_stats: ChannelStats = field(default_factory=ChannelStats)
    lock_stats: PhaseResolvedLockStats = field(default_factory=PhaseResolvedLockStats)
    fingerprint_centroids: list[set[int]] = field(default_factory=list)

    @property
    def reward_mean(self) -> float:
        return self.reward_sum / max(self.total_trajectories, 1)

    @property
    def reward_std(self) -> float:
        if self.total_trajectories < 2:
            return 1.0
        variance = (self.reward_sq_sum / self.total_trajectories) - self.reward_mean**2
        return math.sqrt(max(variance, 0.0))

    @property
    def embedding_novelty(self) -> float:
        if self.total_trajectories == 0:
            return 1.0
        return self.novel_embedding_count / max(self.total_trajectories, 1)

    @property
    def reaction_entropy(self) -> float:
        total = sum(self.reaction_class_counts.values())
        if total == 0:
            return DEFAULT_REACTION_PRIOR
        probabilities = [count / total for count in self.reaction_class_counts.values()]
        return -sum(probability * math.log(probability + 1.0e-10) for probability in probabilities)

    @property
    def lock_clash_mean(self) -> float:
        return self.lock_clash_sum / max(self.total_trajectories, 1)


class MultiScaffoldEntropyRouter:
    """Entropy-balanced scaffold selector aligned to the DSTW phase manifold."""

    def __init__(
        self,
        scaffold_names: Sequence[str],
        *,
        alpha_explore: float = 2.0,
        alpha_manifold: float = 1.8,
        alpha_hyster: float = 1.5,
        alpha_novelty: float = 1.2,
        alpha_channel: float = 1.0,
        alpha_reaction: float = 0.8,
        alpha_lock: float = 0.5,
        alpha_redund: float = 1.0,
        temperature: float = 1.0,
        pgx_exclusions: Mapping[str, set[str]] | None = None,
        fingerprint_novelty_distance: float = 0.7,
        max_fingerprint_centroids: int = 1000,
    ) -> None:
        if not scaffold_names:
            raise ValueError("scaffold_names cannot be empty")
        if len(set(scaffold_names)) != len(scaffold_names):
            raise ValueError("scaffold_names must be unique")
        self.stats = {name: ScaffoldStats(name=name) for name in scaffold_names}
        self.alpha_explore = alpha_explore
        self.alpha_manifold = alpha_manifold
        self.alpha_hyster = alpha_hyster
        self.alpha_novelty = alpha_novelty
        self.alpha_channel = alpha_channel
        self.alpha_reaction = alpha_reaction
        self.alpha_lock = alpha_lock
        self.alpha_redund = alpha_redund
        self.temperature = temperature
        self.pgx_exclusions = {variant: set(names) for variant, names in (pgx_exclusions or {}).items()}
        self.fingerprint_novelty_distance = fingerprint_novelty_distance
        self.max_fingerprint_centroids = max_fingerprint_centroids

    @staticmethod
    def load_pgx_exclusions(path: Path | None) -> dict[str, set[str]]:
        """Load optional PGx scaffold exclusions from a JSON manifest.

        The current Phase3 manifest primarily contains variant risk status. If
        future manifests add explicit scaffold exclusions, this parser will
        consume either ``pgx_scaffold_exclusions`` or per-variant
        ``excluded_scaffolds`` lists. Absence of those keys means no hard mask.
        """

        if path is None or not path.is_file():
            return {}
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            return {}
        explicit = decoded.get("pgx_scaffold_exclusions")
        exclusions: dict[str, set[str]] = {}
        if isinstance(explicit, dict):
            for variant, values in explicit.items():
                if isinstance(values, list):
                    exclusions[str(variant)] = {str(value) for value in values}
        for row in decoded.get("high_risk_variants", []):
            if not isinstance(row, dict):
                continue
            values = row.get("excluded_scaffolds")
            if isinstance(values, list):
                exclusions[str(row.get("variant", "WT"))] = {str(value) for value in values}
        return exclusions

    def compute_channel_diversity(self) -> dict[str, float]:
        ratios = {name: stats.channel_stats.rf_activation_ratio for name, stats in self.stats.items()}
        mean_ratio = sum(ratios.values()) / max(len(ratios), 1)
        return {name: abs(ratio - mean_ratio) for name, ratio in ratios.items()}

    def compute_scores(self, active_variant: str = "WT") -> dict[str, float]:
        channel_diversity = self.compute_channel_diversity()
        scores: dict[str, float] = {}
        excluded = self.pgx_exclusions.get(active_variant, set())
        for name, stats in self.stats.items():
            if name in excluded:
                scores[name] = float("-inf")
                continue
            explore = self.alpha_explore / math.sqrt(max(stats.total_trajectories, 1))
            manifold = self.alpha_manifold * stats.phase_stats.phase_coverage_entropy
            channel = self.alpha_channel * channel_diversity.get(name, 0.0)
            novelty = self.alpha_novelty * stats.embedding_novelty
            reaction = self.alpha_reaction * stats.reaction_entropy
            lock = self.alpha_lock * (stats.lock_stats.phase_resolved_lock_score / 50.0)
            hysteresis = self.alpha_hyster * stats.lock_stats.hysteresis_variance
            redundancy = self.alpha_redund * math.log1p(1.0 / (stats.reward_std + 1.0e-6))
            scores[name] = explore + manifold + channel + novelty + reaction + lock + hysteresis - redundancy
        return scores

    def sample_batch(self, batch_size: int, active_variant: str = "WT") -> list[str]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        scores = self.compute_scores(active_variant)
        names = list(scores)
        finite_scores = [score for score in scores.values() if math.isfinite(score)]
        if not finite_scores:
            raise RuntimeError(f"all scaffolds are PGx-excluded for variant {active_variant}")
        max_score = max(finite_scores)
        exp_scores = [
            0.0 if not math.isfinite(scores[name]) else math.exp((scores[name] - max_score) / self.temperature)
            for name in names
        ]
        total = sum(exp_scores)
        if total <= 0.0:
            raise RuntimeError(f"no scaffold probability mass for variant {active_variant}")
        probabilities = torch.tensor([value / total for value in exp_scores], dtype=torch.float32)
        indices = torch.multinomial(probabilities, batch_size, replacement=True)
        return [names[int(index)] for index in indices.tolist()]

    def update(
        self,
        scaffold_name: str,
        smiles_batch: Sequence[str],
        rewards: Sequence[float],
        reaction_classes: Sequence[str],
        lock_clashes: Sequence[float],
        *,
        phase_occupancy_batch: Sequence[Mapping[str, float]] | None = None,
        lock_clash_phase_batch: Sequence[Mapping[str, float]] | None = None,
        channel_a_activations: Sequence[int] | None = None,
        channel_b_activations: Sequence[int] | None = None,
    ) -> None:
        if scaffold_name not in self.stats:
            raise KeyError(f"unknown scaffold {scaffold_name}")
        lengths = {len(smiles_batch), len(rewards), len(reaction_classes), len(lock_clashes)}
        if len(lengths) != 1:
            raise ValueError("router update batches must have matching lengths")
        stats = self.stats[scaffold_name]
        phase_occupancy_batch = phase_occupancy_batch or [{} for _ in smiles_batch]
        lock_clash_phase_batch = lock_clash_phase_batch or [{} for _ in smiles_batch]
        channel_a_activations = channel_a_activations or [0 for _ in smiles_batch]
        channel_b_activations = channel_b_activations or [1 for _ in smiles_batch]
        for index, (smiles, reward, reaction_class, lock_clash) in enumerate(
            zip(smiles_batch, rewards, reaction_classes, lock_clashes, strict=True)
        ):
            stats.total_trajectories += 1
            stats.reward_sum += float(reward)
            stats.reward_sq_sum += float(reward) * float(reward)
            stats.reaction_class_counts[str(reaction_class)] += 1
            stats.lock_clash_sum += float(lock_clash)
            if self._is_embedding_novel(stats, smiles):
                stats.novel_embedding_count += 1
            for phase in PHASES:
                stats.phase_stats.phase_voxel_counts[phase] += float(phase_occupancy_batch[index].get(phase, 0.0))
                stats.lock_stats.lock_clash_per_phase[phase] += float(lock_clash_phase_batch[index].get(phase, lock_clash))
            stats.lock_stats.trajectory_count += 1
            stats.channel_stats.total_channel_a_activations += int(channel_a_activations[index])
            stats.channel_stats.total_channel_b_activations += max(int(channel_b_activations[index]), 1)
            stats.last_50_rewards.append(float(reward))
            if len(stats.last_50_rewards) > 50:
                stats.last_50_rewards.pop(0)

    def get_telemetry(self, active_variant: str = "WT") -> dict[str, dict[str, float | int | bool]]:
        channel_diversity = self.compute_channel_diversity()
        excluded = self.pgx_exclusions.get(active_variant, set())
        return {
            name: {
                "trajectories": stats.total_trajectories,
                "unique_embed": stats.novel_embedding_count,
                "novelty": round(stats.embedding_novelty, 4),
                "reward_mean": round(stats.reward_mean, 4),
                "reward_std": round(stats.reward_std, 4),
                "rxn_entropy": round(stats.reaction_entropy, 4),
                "phase_entropy": round(stats.phase_stats.phase_coverage_entropy, 4),
                "rf_ratio": round(stats.channel_stats.rf_activation_ratio, 4),
                "channel_div": round(channel_diversity.get(name, 0.0), 4),
                "lock_min_phase": round(stats.lock_stats.phase_resolved_lock_score, 4),
                "hysteresis": round(stats.lock_stats.hysteresis_variance, 4),
                "pgx_excluded": name in excluded,
            }
            for name, stats in self.stats.items()
        }

    def telemetry_lines(self, active_variant: str = "WT") -> list[str]:
        telemetry = self.get_telemetry(active_variant)
        lines: list[str] = []
        for scaffold_name, stats in telemetry.items():
            lines.append(
                "scaffold_router_dstw "
                f"scaffold={scaffold_name} trajectories={stats['trajectories']} "
                f"unique_embed={stats['unique_embed']} novelty={stats['novelty']} "
                f"reward_mean={stats['reward_mean']} reward_std={stats['reward_std']} "
                f"rxn_entropy={stats['rxn_entropy']} phase_entropy={stats['phase_entropy']} "
                f"rf_ratio={stats['rf_ratio']} channel_div={stats['channel_div']} "
                f"lock_min_phase={stats['lock_min_phase']} hysteresis={stats['hysteresis']} "
                f"pgx_excluded={str(stats['pgx_excluded']).lower()}"
            )
        return lines

    def _is_embedding_novel(self, stats: ScaffoldStats, smiles: str) -> bool:
        fingerprint = self._fingerprint(smiles)
        if fingerprint is None:
            return smiles not in {str(item) for item in stats.fingerprint_centroids}
        if not stats.fingerprint_centroids:
            stats.fingerprint_centroids.append(fingerprint)
            return True
        max_similarity = max(self._tanimoto(fingerprint, centroid) for centroid in stats.fingerprint_centroids)
        is_novel = (1.0 - max_similarity) >= self.fingerprint_novelty_distance
        if is_novel and len(stats.fingerprint_centroids) < self.max_fingerprint_centroids:
            stats.fingerprint_centroids.append(fingerprint)
        return is_novel

    @staticmethod
    def _fingerprint(smiles: str) -> set[int] | None:
        # A dependency-light Morgan proxy: atom-pair-ish hashed character
        # shingles. This keeps the router strict-typed and deterministic while
        # avoiding RDKit imports in the training hot path.
        if not smiles:
            return None
        values: set[int] = set()
        for index in range(max(1, len(smiles) - 2)):
            shingle = smiles[index : index + 3].encode("utf-8")
            values.add(int.from_bytes(hashlib.sha256(shingle).digest()[:4], "big") % 4096)
        return values

    @staticmethod
    def _tanimoto(lhs: set[int], rhs: set[int]) -> float:
        union = len(lhs | rhs)
        if union == 0:
            return 0.0
        return len(lhs & rhs) / float(union)


def phase_occupancy_from_fiber_bundle(fiber_bundle: Tensor) -> list[dict[str, float]]:
    """Aggregate oracle terminal fiber tensors into phase occupancy maps."""

    if fiber_bundle.ndim != 3 or int(fiber_bundle.shape[1]) != len(PHASES):
        raise ValueError(f"expected [batch, 5, D] fiber tensor, got {tuple(fiber_bundle.shape)}")
    rows: list[dict[str, float]] = []
    for batch_idx in range(int(fiber_bundle.shape[0])):
        values: dict[str, float] = {}
        for phase_idx, phase in enumerate(PHASES):
            phase_vector = fiber_bundle[batch_idx, phase_idx, :]
            if int(phase_vector.numel()) >= 3:
                values[phase] = max(float(phase_vector[2].item()), 0.0)
            else:
                values[phase] = 0.0
        rows.append(values)
    return rows


def lock_phase_maps_from_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, float]]:
    phase_maps: list[dict[str, float]] = []
    for row in rows:
        fallback = _float_value(row.get("pi_clash_lock", 0.0))
        phase_maps.append(
            {
                phase: _float_value(row.get(f"pi_clash_lock_{phase}", fallback))
                for phase in PHASES
            }
        )
    return phase_maps


def _float_value(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, int | float | str):
        return float(value)
    return 0.0
