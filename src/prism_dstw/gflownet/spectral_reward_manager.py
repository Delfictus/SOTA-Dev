"""Event-triggered C6 spectral reward manager for captured tile states."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from prism_dstw.gflownet.bsr_operator_state import BSROperatorState
from prism_dstw.gflownet.cuda_graph_tile_runtime import CapturedTileTelemetry, TileReplayResult


class SpectralRewardError(RuntimeError):
    """Raised when spectral reward causality or placement is invalid."""


@dataclass
class SpectralRewardRecord:
    state_hash: str
    log_reward: Tensor
    cache_hit: bool
    topology_delta: str
    basin_delta: str
    restricted_operator_hash: str
    c6_operator_hash: str
    captured_graph_tile_hash: str


@dataclass(frozen=True)
class SpectralRewardCacheKey:
    state_hash: str
    operator_hash: str
    topology_delta: str
    basin_delta: str
    restricted_operator_hash: str
    c6_operator_hash: str
    captured_graph_tile_hash: str


@dataclass
class SpectralRewardManager:
    """Computes cached rewards only for states produced by captured tile events."""

    state: BSROperatorState
    telemetry: CapturedTileTelemetry
    basin_weights: Tensor | None = None
    cache: dict[SpectralRewardCacheKey, Tensor] = field(default_factory=dict)
    event_count: int = 0

    def _gpu_spectral_solve(self) -> Tensor:
        if self.state.device.type != "cuda":
            self.telemetry.cpu_solve_count += 1
            raise SpectralRewardError("LOG_SUBTB_BLOCKED_GPU_PLACEMENT: reward state is not cuda")
        started = time.perf_counter()
        dense = self.state.dense_for_gpu_solve()
        if self.basin_weights is None:
            basin = torch.linspace(0.25, 1.0, steps=dense.shape[0], device=self.state.device, dtype=self.state.dtype)
        else:
            basin = self.basin_weights.to(device=self.state.device, dtype=self.state.dtype)
            if basin.ndim != 1 or int(basin.shape[0]) != int(dense.shape[0]):
                raise SpectralRewardError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: C6 basin vector shape mismatch")
            if not bool(torch.isfinite(basin).all().item()):
                raise SpectralRewardError("LOG_SUBTB_BLOCKED_TILE_CAPTURE: C6 basin vector contains NaN/Inf")
        identity = torch.eye(dense.shape[0], device=self.state.device, dtype=self.state.dtype)
        restricted_dirichlet = identity - dense
        dirichlet_energy = basin.matmul(restricted_dirichlet).matmul(basin) / float(dense.shape[0])
        c6_response = dense.matmul(basin)
        retention = torch.mean(c6_response)
        reward = torch.log(
            torch.clamp(retention + 1.0, min=torch.tensor(1.0e-12, device=self.state.device, dtype=self.state.dtype))
        ) - torch.clamp(dirichlet_energy, min=0.0)
        torch.cuda.synchronize()
        self.telemetry.gpu_solve_ms.append((time.perf_counter() - started) * 1000.0)
        self.telemetry.gpu_solve_count += 1
        return reward

    def reward_for_event(self, event: TileReplayResult) -> SpectralRewardRecord:
        required = [
            event.topology_delta,
            event.basin_delta,
            event.restricted_operator_hash,
            event.c6_operator_hash,
            event.captured_graph_tile_hash,
        ]
        if any(not value for value in required):
            raise SpectralRewardError("spectral reward manager bypassed captured tile event metadata")
        if event.event_provenance_hash not in self.telemetry.captured_event_hashes:
            raise SpectralRewardError("spectral reward event was not produced by captured tile runtime")
        if event.state_hash != self.state.state_hash or event.operator_hash != self.state.operator_hash:
            raise SpectralRewardError("spectral reward event does not match live operator state")
        self.event_count += 1
        cache_key = SpectralRewardCacheKey(
            state_hash=event.state_hash,
            operator_hash=event.operator_hash,
            topology_delta=event.topology_delta,
            basin_delta=event.basin_delta,
            restricted_operator_hash=event.restricted_operator_hash,
            c6_operator_hash=event.c6_operator_hash,
            captured_graph_tile_hash=event.captured_graph_tile_hash,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return SpectralRewardRecord(
                state_hash=event.state_hash,
                log_reward=cached,
                cache_hit=True,
                topology_delta=event.topology_delta,
                basin_delta=event.basin_delta,
                restricted_operator_hash=event.restricted_operator_hash,
                c6_operator_hash=event.c6_operator_hash,
                captured_graph_tile_hash=event.captured_graph_tile_hash,
            )
        reward = self._gpu_spectral_solve()
        if not bool(torch.isfinite(reward).item()):
            raise SpectralRewardError("spectral reward produced NaN/Inf")
        self.cache[cache_key] = reward.detach()
        return SpectralRewardRecord(
            state_hash=event.state_hash,
            log_reward=reward,
            cache_hit=False,
            topology_delta=event.topology_delta,
            basin_delta=event.basin_delta,
            restricted_operator_hash=event.restricted_operator_hash,
            c6_operator_hash=event.c6_operator_hash,
            captured_graph_tile_hash=event.captured_graph_tile_hash,
        )

    @property
    def cache_hit_rate(self) -> float:
        if self.event_count == 0:
            return 0.0
        misses = len(self.cache)
        hits = max(self.event_count - misses, 0)
        return float(hits) / float(self.event_count)

    def manifest(self) -> dict[str, Any]:
        if self.telemetry.captured_graph_replay_count == 0:
            status = "LOG_SUBTB_BLOCKED_TILE_CAPTURE"
        elif self.telemetry.cpu_solve_count > 0:
            status = "LOG_SUBTB_BLOCKED_GPU_PLACEMENT"
        else:
            status = "LOG_SUBTB_SPECTRAL_REWARD_ACTIVE"
        return {
            "status": status,
            "reward_cache_size": len(self.cache),
            "reward_cache_hit_rate": self.cache_hit_rate,
            "gpu_solve_count": self.telemetry.gpu_solve_count,
            "cpu_solve_count": self.telemetry.cpu_solve_count,
            "event_count": self.event_count,
            "event_trigger_source": "captured_tile_delta_hashes",
        }


def log_subtb_loss(forward_log_probs: list[Tensor], backward_log_probs: list[Tensor], log_rewards: list[Tensor]) -> Tensor:
    if not forward_log_probs or not log_rewards:
        raise SpectralRewardError("SubTB loss requires at least one captured tile transition")
    forward = torch.stack(forward_log_probs).sum()
    backward = torch.stack(backward_log_probs).sum()
    reward = torch.stack(log_rewards).sum()
    residual = forward - backward - reward
    if not bool(torch.isfinite(residual).item()):
        raise SpectralRewardError("SubTB residual is NaN/Inf")
    return residual.square() + torch.tensor(math.log1p(len(log_rewards)), device=reward.device, dtype=reward.dtype)
