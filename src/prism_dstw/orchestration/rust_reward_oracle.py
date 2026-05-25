"""Async batched bridge to the PRISM-FORGE Rust reward oracle."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl
import torch
from torch import Tensor


DEFAULT_BATCH_PATH = Path(".scratch/oracle_batch.parquet")
DEFAULT_REWARD_PATH = Path(".scratch/oracle_rewards.parquet")
DEFAULT_ORACLE_BIN = Path("target/release/oracle_scorer")
DEFAULT_SURVIVOR_CORPUS = Path(
    "campaigns/glp1r_aleniglipron/track_a_generative/"
    "vspace_survivors_full_scale.parquet"
)


class RustOracleError(RuntimeError):
    """Raised when the Rust reward oracle returns invalid output."""


@dataclass(frozen=True)
class OracleProposal:
    """One molecule proposal sent to the Rust oracle."""

    anchor_id: str
    canonical_smiles: str
    trajectory_id: str


@dataclass(frozen=True)
class OracleTelemetry:
    """Latency and reward telemetry for one oracle batch."""

    oracle_batch_size: int
    oracle_latency_ms: float
    rust_scoring_time_ms: float
    parquet_write_ms: float
    parquet_read_ms: float
    reward_mean: float
    reward_std: float
    invalid_reward_count: int
    duplicate_smiles_count: int


@dataclass(frozen=True)
class OracleBatchResult:
    """Validated reward tensor plus component table."""

    rewards: Tensor
    fiber_bundle: Tensor
    rows: pl.DataFrame
    telemetry: OracleTelemetry


@dataclass
class BatchedRustOracle:
    """Strict reward-authority wrapper around ``oracle_scorer``.

    The Python training loop never computes final rewards directly. It writes
    parquet proposals, invokes the Rust binary, validates the returned parquet,
    and only then exposes a reward tensor to PyTorch.
    """

    oracle_binary: Path = DEFAULT_ORACLE_BIN
    survivor_corpus: Path = DEFAULT_SURVIVOR_CORPUS
    batch_path: Path = DEFAULT_BATCH_PATH
    reward_path: Path = DEFAULT_REWARD_PATH
    max_batch_size: int = 64
    extra_args: Sequence[str] = field(default_factory=tuple)
    last_telemetry: OracleTelemetry | None = None

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")

    async def score_batch(self, proposals: Sequence[OracleProposal]) -> OracleBatchResult:
        """Score a batch of molecular proposals through the Rust oracle."""

        batch_df = self.prepare_batch(proposals)
        start = time.perf_counter()
        write_start = time.perf_counter()
        self.batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch_df.write_parquet(self.batch_path)
        parquet_write_ms = (time.perf_counter() - write_start) * 1000.0
        rust_scoring_time_ms = await self.invoke_rust()
        read_start = time.perf_counter()
        rewards_df = self.read_rewards()
        parquet_read_ms = (time.perf_counter() - read_start) * 1000.0
        telemetry = self.validate_rewards(
            proposals=proposals,
            rewards_df=rewards_df,
            oracle_latency_ms=(time.perf_counter() - start) * 1000.0,
            rust_scoring_time_ms=rust_scoring_time_ms,
            parquet_write_ms=parquet_write_ms,
            parquet_read_ms=parquet_read_ms,
        )
        self.last_telemetry = telemetry
        rewards = torch.tensor(rewards_df.get_column("reward").to_list(), dtype=torch.float32)
        fiber_bundle = self.fiber_bundle_from_rewards(rewards_df)
        return OracleBatchResult(
            rewards=rewards,
            fiber_bundle=fiber_bundle,
            rows=rewards_df,
            telemetry=telemetry,
        )

    def fiber_bundle_from_rewards(self, rewards_df: pl.DataFrame) -> Tensor:
        """Return a compact five-phase tensor emitted from Rust reward components.

        The training policy obtains its full scaffold fiber features from the
        protocol-aware receptor tensors. The oracle still returns a phase-shaped
        terminal tensor so callers can persist reward-conditioned state without
        collapsing the Track A interface to a scalar-only API.
        """

        phase_rows: list[list[list[float]]] = []
        for row in rewards_df.iter_rows(named=True):
            reward = float(row["reward"])
            complement = float(row["pi_complement"])
            clash = float(row["pi_clash_pocket"]) if "pi_clash_pocket" in row else float(row["adjusted_pi_clash"])
            lock_clash = float(row["pi_clash_lock"]) if "pi_clash_lock" in row else 0.0
            lock_phase = [
                float(row.get("lock_occupancy_cold_hold", row.get("pi_clash_lock_cold_hold", lock_clash)) or 0.0),
                float(row.get("lock_occupancy_ramp_up", row.get("pi_clash_lock_ramp_up", lock_clash)) or 0.0),
                float(row.get("lock_occupancy_warm_hold", row.get("pi_clash_lock_warm_hold", lock_clash)) or 0.0),
                float(row.get("lock_occupancy_ramp_down", row.get("pi_clash_lock_ramp_down", lock_clash)) or 0.0),
                float(
                    row.get("lock_occupancy_cold_return", row.get("pi_clash_lock_cold_return", lock_clash))
                    or 0.0
                ),
            ]
            cryptic = float(row["cryptic_bonus"])
            dihedral = float(row["selected_dihedral_deg"])
            phase_rows.append(
                [
                    [0.0, clash, 0.0, lock_phase[0]],
                    [0.25, clash, complement * 0.25, lock_phase[1]],
                    [0.50, clash, complement, lock_phase[2]],
                    [0.75, clash * 0.5, complement, lock_phase[3] + cryptic],
                    [1.0, clash * 0.25, complement * 0.5, lock_phase[4] + dihedral / 360.0],
                ]
            )
        return torch.tensor(phase_rows, dtype=torch.float32)

    def prepare_batch(self, proposals: Sequence[OracleProposal]) -> pl.DataFrame:
        """Create a parquet-ready proposal batch."""

        if len(proposals) == 0:
            raise RustOracleError("empty oracle batch")
        if len(proposals) > self.max_batch_size:
            raise RustOracleError(
                f"oracle batch size {len(proposals)} exceeds max_batch_size {self.max_batch_size}"
            )
        duplicate_count = len(proposals) - len({proposal.canonical_smiles for proposal in proposals})
        if duplicate_count > 0:
            raise RustOracleError(f"oracle batch contains {duplicate_count} duplicate SMILES")
        return pl.DataFrame(
            {
                "trajectory_id": [proposal.trajectory_id for proposal in proposals],
                "anchor_id": [proposal.anchor_id for proposal in proposals],
                "canonical_smiles": [proposal.canonical_smiles for proposal in proposals],
            }
        )

    async def invoke_rust(self) -> float:
        """Run the Rust scorer subprocess and return its wall time in ms."""

        if not self.oracle_binary.is_file():
            raise RustOracleError(f"Rust oracle binary not found: {self.oracle_binary}")
        if not self.survivor_corpus.is_file():
            raise RustOracleError(f"survivor corpus not found: {self.survivor_corpus}")
        if self.reward_path.exists():
            self.reward_path.unlink()
        command = [
            str(self.oracle_binary),
            "--batch",
            str(self.batch_path),
            "--rewards",
            str(self.reward_path),
            "--survivors",
            str(self.survivor_corpus),
            *self.extra_args,
        ]
        start = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if process.returncode != 0:
            raise RustOracleError(
                "Rust oracle exited nonzero "
                f"code={process.returncode}\nstdout={stdout.decode()}\nstderr={stderr.decode()}"
            )
        return elapsed_ms

    def read_rewards(self) -> pl.DataFrame:
        """Read the Rust oracle reward parquet."""

        if not self.reward_path.is_file():
            raise RustOracleError(f"Rust oracle did not write rewards parquet: {self.reward_path}")
        return pl.read_parquet(self.reward_path)

    def validate_rewards(
        self,
        *,
        proposals: Sequence[OracleProposal],
        rewards_df: pl.DataFrame,
        oracle_latency_ms: float,
        rust_scoring_time_ms: float,
        parquet_write_ms: float,
        parquet_read_ms: float,
    ) -> OracleTelemetry:
        """Validate reward shape, finiteness, positivity, and authority flags."""

        required = {
            "canonical_smiles",
            "reward",
            "pi_complement",
            "adjusted_pi_clash",
            "pi_clash_pocket",
            "pi_clash_lock",
            "pi_clash_lock_cold_hold",
            "pi_clash_lock_ramp_up",
            "pi_clash_lock_warm_hold",
            "pi_clash_lock_ramp_down",
            "pi_clash_lock_cold_return",
            "lock_geometry_score",
            "lock_geometry_atom_count",
            "lock_voxel_indices_json",
            "lock_occupancy_cold_hold",
            "lock_occupancy_ramp_up",
            "lock_occupancy_warm_hold",
            "lock_occupancy_ramp_down",
            "lock_occupancy_cold_return",
            "intracellular_penetration_depth_angstrom",
            "lock_steric_volume_angstrom3",
            "cryptic_bonus",
            "consensus_complement_bonus",
            "survival_tier",
            "selected_dihedral_deg",
            "reward_components_json",
            "oracle_valid",
        }
        missing = required.difference(rewards_df.columns)
        if missing:
            raise RustOracleError(f"Rust oracle rewards missing columns: {sorted(missing)}")
        if rewards_df.height != len(proposals):
            raise RustOracleError(
                f"batch size mismatch: sent {len(proposals)}, received {rewards_df.height}"
            )
        duplicate_smiles_count = rewards_df.height - rewards_df.select("canonical_smiles").unique().height
        if duplicate_smiles_count > 0:
            raise RustOracleError(f"Rust oracle emitted {duplicate_smiles_count} duplicate SMILES rows")
        rewards = [float(value) for value in rewards_df.get_column("reward").to_list()]
        invalid_reward_count = sum(1 for value in rewards if not math.isfinite(value) or value < 0.0)
        if invalid_reward_count > 0:
            raise RustOracleError(f"Rust oracle emitted {invalid_reward_count} invalid rewards")
        invalid_flags = rewards_df.filter(~pl.col("oracle_valid"))
        if invalid_flags.height > 0:
            raise RustOracleError(f"Rust oracle marked {invalid_flags.height} rows invalid")
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        return OracleTelemetry(
            oracle_batch_size=len(proposals),
            oracle_latency_ms=oracle_latency_ms,
            rust_scoring_time_ms=rust_scoring_time_ms,
            parquet_write_ms=parquet_write_ms,
            parquet_read_ms=parquet_read_ms,
            reward_mean=float(reward_tensor.mean().item()),
            reward_std=float(reward_tensor.std(unbiased=False).item()),
            invalid_reward_count=invalid_reward_count,
            duplicate_smiles_count=duplicate_smiles_count,
        )

    async def shutdown(self) -> None:
        """Placeholder for future persistent oracle processes."""

        return None


def proposals_from_rows(rows: pl.DataFrame, indices: Sequence[int]) -> list[OracleProposal]:
    """Build proposal objects from an anchor/action dataframe."""

    selected = rows[["anchor_id", "canonical_smiles"]].to_dicts()
    proposals: list[OracleProposal] = []
    for batch_index, action_index in enumerate(indices):
        row = selected[action_index]
        anchor_id = row.get("anchor_id")
        canonical_smiles = row.get("canonical_smiles")
        if not isinstance(anchor_id, str) or not isinstance(canonical_smiles, str):
            raise RustOracleError("proposal row missing anchor_id or canonical_smiles")
        proposals.append(
            OracleProposal(
                anchor_id=anchor_id,
                canonical_smiles=canonical_smiles,
                trajectory_id=f"trajectory-{batch_index:06d}",
            )
        )
    return proposals


def telemetry_to_dict(telemetry: OracleTelemetry) -> dict[str, float | int]:
    """Serialize oracle telemetry for JSON/CSV outputs."""

    return {
        "oracle_batch_size": telemetry.oracle_batch_size,
        "oracle_latency_ms": telemetry.oracle_latency_ms,
        "rust_scoring_time_ms": telemetry.rust_scoring_time_ms,
        "parquet_write_ms": telemetry.parquet_write_ms,
        "parquet_read_ms": telemetry.parquet_read_ms,
        "reward_mean": telemetry.reward_mean,
        "reward_std": telemetry.reward_std,
        "invalid_reward_count": telemetry.invalid_reward_count,
        "duplicate_smiles_count": telemetry.duplicate_smiles_count,
    }


__all__ = [
    "BatchedRustOracle",
    "OracleBatchResult",
    "OracleProposal",
    "OracleTelemetry",
    "RustOracleError",
    "proposals_from_rows",
    "telemetry_to_dict",
]
