"""Async batched bridge to the PRISM-FORGE Rust reward oracle."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl
import torch
from torch import Tensor

DEFAULT_SCRATCH_ROOT = Path(os.environ.get("PRISM_SCRATCH_ROOT", "/mnt/storage/prism-scratch/Prism4D-bio"))

DEFAULT_BATCH_PATH = DEFAULT_SCRATCH_ROOT / "oracle_bridge" / "oracle_batch.parquet"
DEFAULT_REWARD_PATH = DEFAULT_SCRATCH_ROOT / "oracle_bridge" / "oracle_rewards.parquet"
DEFAULT_ORACLE_BIN = Path("target/release/oracle_scorer")
DEFAULT_SURVIVOR_CORPUS = Path(
    "campaigns/glp1r_aleniglipron/track_a_generative/"
    "vspace_survivors_full_scale.parquet"
)
FRAGMENT_CONTEXT_EXCLUSION_A = 2.32


class RustOracleError(RuntimeError):
    """Raised when the Rust reward oracle returns invalid output."""


@dataclass(frozen=True)
class OracleProposal:
    """One molecule proposal sent to the Rust oracle."""

    anchor_id: str
    canonical_smiles: str
    trajectory_id: str
    coordinates_json: str | None = None
    score_atom_offset: int = 0
    u_pose: float = 0.0


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
class SurvivorCorpusOracle:
    """Strict survivor-corpus reward wrapper around ``oracle_scorer``.

    This is not a live signal-grid scorer. Proposed molecules are scored by
    lookup against the immutable survivor parquet produced by ``vspace_pruner``.
    That corpus was computed from signal-grid voxel mapping offline, and this
    runtime bridge preserves the full component row returned by Rust.
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
        if self.batch_path == DEFAULT_BATCH_PATH and self.reward_path == DEFAULT_REWARD_PATH:
            run_id = f"{os.getpid()}_{id(self):x}"
            scratch_dir = DEFAULT_BATCH_PATH.parent / "oracle_runs" / run_id
            self.batch_path = scratch_dir / "oracle_batch.parquet"
            self.reward_path = scratch_dir / "oracle_rewards.parquet"

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
        rewards_df = self.annotate_lock_phase_provenance(self.read_rewards())
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
                "coordinates_json": [proposal.coordinates_json or "" for proposal in proposals],
                "score_atom_offset": [int(proposal.score_atom_offset) for proposal in proposals],
                "u_pose": [strict_nonnegative_finite_float(proposal.u_pose, "u_pose") for proposal in proposals],
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
        command = self.build_command()
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

    def build_command(self) -> list[str]:
        """Return the exact Rust command used for survivor-corpus scoring."""

        return [
            str(self.oracle_binary),
            "--batch",
            str(self.batch_path),
            "--rewards",
            str(self.reward_path),
            "--survivors",
            str(self.survivor_corpus),
            *self.extra_args,
        ]

    def read_rewards(self) -> pl.DataFrame:
        """Read the Rust oracle reward parquet."""

        if not self.reward_path.is_file():
            raise RustOracleError(f"Rust oracle did not write rewards parquet: {self.reward_path}")
        return pl.read_parquet(self.reward_path)

    def annotate_lock_phase_provenance(self, rewards_df: pl.DataFrame) -> pl.DataFrame:
        """Ensure each oracle row carries explicit lock phase provenance."""

        if "lock_phase_provenance" in rewards_df.columns:
            return rewards_df
        phase_columns = [
            "lock_occupancy_cold_hold",
            "lock_occupancy_ramp_up",
            "lock_occupancy_warm_hold",
            "lock_occupancy_ramp_down",
            "lock_occupancy_cold_return",
        ]
        if not set(phase_columns).issubset(set(rewards_df.columns)):
            return rewards_df.with_columns(pl.lit("UNKNOWN").alias("lock_phase_provenance"))
        tags: list[str] = []
        for row in rewards_df.select(phase_columns).iter_rows():
            finite_values = [float(value or 0.0) for value in row]
            tags.append(
                "REPLICATED_AGGREGATE"
                if len({round(value, 12) for value in finite_values}) <= 1
                else "PHASE_RESOLVED"
            )
        return rewards_df.with_columns(pl.Series("lock_phase_provenance", tags))

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
            "pathway_voxels",
            "void_atom_count",
            "lock_phase_provenance",
            "survival_tier",
            "selected_dihedral_deg",
            "reward_components_json",
            "oracle_valid",
        }
        missing = required.difference(rewards_df.columns)
        if missing:
            raise RustOracleError(f"Rust oracle rewards missing columns: {sorted(missing)}")
        if rewards_df.height != len(proposals):
            observed = (
                rewards_df.select("canonical_smiles").head(8).to_series().to_list()
                if "canonical_smiles" in rewards_df.columns
                else []
            )
            expected = [proposal.canonical_smiles for proposal in proposals[:8]]
            raise RustOracleError(
                f"batch size mismatch: sent {len(proposals)}, received {rewards_df.height}; "
                f"batch_path={self.batch_path} reward_path={self.reward_path} "
                f"expected_head={expected} observed_head={observed}"
            )
        observed_smiles = [str(value) for value in rewards_df.get_column("canonical_smiles").to_list()]
        expected_smiles = [proposal.canonical_smiles for proposal in proposals]
        if observed_smiles != expected_smiles:
            raise RustOracleError(
                "survivor oracle reward rows are not aligned by canonical_smiles; "
                f"expected_head={expected_smiles[:8]} observed_head={observed_smiles[:8]}"
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


@dataclass
class LiveSignalGridOracle(SurvivorCorpusOracle):
    """Live signal-grid scorer backed by ``oracle_scorer --live-scoring``.

    This mode scores proposal atom coordinates directly against the loaded
    signal grid at runtime. It is intentionally separate from survivor lookup
    so callers must opt in before scoring molecules outside the survivor
    corpus.
    """

    signal_grid: Path = Path(
        "campaigns/glp1r_aleniglipron/track_a_generative/signal_grid_population_consensus.parquet"
    )
    grid_config: Path = Path(
        "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json"
    )
    shear_stress: Path | None = Path(
        "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/shear_stress_field.parquet"
    )
    translation_pathway: Path | None = Path(
        "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/translation_pathway_nodes.parquet"
    )
    nma_continuity_map: Path | None = None
    hydration_continuity_map: Path | None = None
    thermodynamic_continuity_map: Path | None = None
    continuity_admissibility: bool = False
    lock_mask: Path | None = Path("campaigns/glp1r_aleniglipron/track_a_generative/lock_region_mask.json")

    def prepare_batch(self, proposals: Sequence[OracleProposal]) -> pl.DataFrame:
        """Create a live-scoring batch with mandatory atom coordinates."""

        if len(proposals) == 0:
            raise RustOracleError("empty oracle batch")
        if len(proposals) > self.max_batch_size:
            raise RustOracleError(
                f"oracle batch size {len(proposals)} exceeds max_batch_size {self.max_batch_size}"
            )
        missing_or_invalid: list[str] = []
        for proposal in proposals:
            if not proposal.coordinates_json:
                missing_or_invalid.append(proposal.trajectory_id)
                continue
            try:
                positions = parse_live_coordinates_json(proposal.coordinates_json)
            except RustOracleError:
                missing_or_invalid.append(proposal.trajectory_id)
                continue
            try:
                score_atom_offset = strict_score_atom_offset(proposal.score_atom_offset)
            except RustOracleError:
                missing_or_invalid.append(proposal.trajectory_id)
                continue
            try:
                live_score_coordinate_count(positions, score_atom_offset)
            except RustOracleError:
                missing_or_invalid.append(proposal.trajectory_id)
        if missing_or_invalid:
            raise RustOracleError(
                "live signal-grid oracle requires non-empty finite coordinates_json and a "
                "score_atom_offset that leaves at least one atom to score; "
                f"invalid={missing_or_invalid[:5]}"
            )
        return pl.DataFrame(
            {
                "trajectory_id": [proposal.trajectory_id for proposal in proposals],
                "anchor_id": [proposal.anchor_id for proposal in proposals],
                "canonical_smiles": [proposal.canonical_smiles for proposal in proposals],
                "coordinates_json": [proposal.coordinates_json or "" for proposal in proposals],
                "score_atom_offset": [int(proposal.score_atom_offset) for proposal in proposals],
                "u_pose": [strict_nonnegative_finite_float(proposal.u_pose, "u_pose") for proposal in proposals],
            }
        )

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
        """Validate live-scoring rows while allowing duplicate product identities."""

        if "trajectory_id" not in rewards_df.columns:
            raise RustOracleError("live oracle rewards missing trajectory_id column")
        expected_trajectory_ids = [proposal.trajectory_id for proposal in proposals]
        observed_trajectory_ids = [str(value) for value in rewards_df.get_column("trajectory_id").to_list()]
        if observed_trajectory_ids != expected_trajectory_ids:
            raise RustOracleError(
                "live oracle reward rows are not aligned by trajectory_id; "
                f"expected_head={expected_trajectory_ids[:8]} observed_head={observed_trajectory_ids[:8]}"
            )
        validation_rewards = deduplicate_live_reward_identities(rewards_df)
        validation_proposals = [
            OracleProposal(
                anchor_id=proposal.anchor_id,
                canonical_smiles=str(validation_rewards.get_column("canonical_smiles")[index]),
                trajectory_id=proposal.trajectory_id,
                coordinates_json=proposal.coordinates_json,
                score_atom_offset=proposal.score_atom_offset,
                u_pose=proposal.u_pose,
            )
            for index, proposal in enumerate(proposals)
        ]
        telemetry = super().validate_rewards(
            proposals=validation_proposals,
            rewards_df=validation_rewards,
            oracle_latency_ms=oracle_latency_ms,
            rust_scoring_time_ms=rust_scoring_time_ms,
            parquet_write_ms=parquet_write_ms,
            parquet_read_ms=parquet_read_ms,
        )
        if self.continuity_admissibility:
            required_continuity = {
                "nma_disruption_penalty",
                "hydration_blockade_penalty",
                "thermodynamic_trap_penalty",
                "pathway_bonus",
                "u_pose",
                "continuity_admissibility",
                "continuity_reward_v1",
                "continuity_provenance",
            }
            missing_continuity = required_continuity.difference(rewards_df.columns)
            if missing_continuity:
                raise RustOracleError(
                    f"live continuity rewards missing columns: {sorted(missing_continuity)}"
                )
        duplicate_count = rewards_df.height - rewards_df.select("canonical_smiles").unique().height
        return OracleTelemetry(
            oracle_batch_size=telemetry.oracle_batch_size,
            oracle_latency_ms=telemetry.oracle_latency_ms,
            rust_scoring_time_ms=telemetry.rust_scoring_time_ms,
            parquet_write_ms=telemetry.parquet_write_ms,
            parquet_read_ms=telemetry.parquet_read_ms,
            reward_mean=telemetry.reward_mean,
            reward_std=telemetry.reward_std,
            invalid_reward_count=telemetry.invalid_reward_count,
            duplicate_smiles_count=duplicate_count,
        )

    async def invoke_rust(self) -> float:
        """Run the Rust scorer in live signal-grid mode."""

        if not self.oracle_binary.is_file():
            raise RustOracleError(f"Rust oracle binary not found: {self.oracle_binary}")
        if not self.signal_grid.is_file():
            raise RustOracleError(f"signal grid not found: {self.signal_grid}")
        if not self.grid_config.is_file():
            raise RustOracleError(f"grid config not found: {self.grid_config}")
        if self.shear_stress is not None and not self.shear_stress.is_file():
            raise RustOracleError(f"shear stress field not found: {self.shear_stress}")
        if self.translation_pathway is not None and not self.translation_pathway.is_file():
            raise RustOracleError(f"translation pathway not found: {self.translation_pathway}")
        if self.continuity_admissibility:
            required_maps = {
                "nma_continuity_map": self.nma_continuity_map,
                "hydration_continuity_map": self.hydration_continuity_map,
                "thermodynamic_continuity_map": self.thermodynamic_continuity_map,
            }
            for name, path in required_maps.items():
                if path is None:
                    raise RustOracleError(f"continuity admissibility requires {name}")
                if not path.is_file():
                    raise RustOracleError(f"{name} not found: {path}")
        if self.lock_mask is not None and not self.lock_mask.is_file():
            raise RustOracleError(f"lock mask not found: {self.lock_mask}")
        if self.reward_path.exists():
            self.reward_path.unlink()
        command = self.build_command()
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
                "Rust live oracle exited nonzero "
                f"code={process.returncode}\nstdout={stdout.decode()}\nstderr={stderr.decode()}"
            )
        return elapsed_ms

    def build_command(self) -> list[str]:
        """Return the exact Rust command used for live signal-grid scoring."""

        return [
            str(self.oracle_binary),
            "--live-scoring",
            "--input",
            str(self.batch_path),
            "--output",
            str(self.reward_path),
            "--signal-grid",
            str(self.signal_grid),
            "--grid-config",
            str(self.grid_config),
            *(
                ["--shear-stress", str(self.shear_stress)]
                if self.shear_stress is not None
                else ["--no-shear-stress"]
            ),
            *(
                ["--translation-pathway", str(self.translation_pathway)]
                if self.translation_pathway is not None
                else ["--no-translation-pathway"]
            ),
            *(
                [
                    "--continuity-admissibility",
                    "--nma-continuity-map",
                    str(self.nma_continuity_map),
                    "--hydration-continuity-map",
                    str(self.hydration_continuity_map),
                    "--thermodynamic-continuity-map",
                    str(self.thermodynamic_continuity_map),
                ]
                if self.continuity_admissibility
                else []
            ),
            *(["--lock-mask", str(self.lock_mask)] if self.lock_mask is not None else ["--no-lock-mask"]),
            *self.extra_args,
        ]


BatchedRustOracle = SurvivorCorpusOracle


def deduplicate_live_reward_identities(rewards_df: pl.DataFrame) -> pl.DataFrame:
    """Return a validation-only view with unique IDs for duplicate live rows."""

    if "canonical_smiles" not in rewards_df.columns:
        return rewards_df
    seen: dict[str, int] = {}
    identifiers: list[str] = []
    for value in rewards_df.get_column("canonical_smiles").to_list():
        base = str(value)
        count = seen.get(base, 0)
        seen[base] = count + 1
        identifiers.append(base if count == 0 else f"{base}__live_duplicate_{count}")
    return rewards_df.with_columns(pl.Series("canonical_smiles", identifiers))


def parse_live_coordinates_json(raw: str) -> list[tuple[float, float, float]]:
    """Validate the coordinate payload accepted by live Rust scoring."""

    try:
        decoded: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RustOracleError(f"invalid coordinates_json: {exc}") from exc
    if not isinstance(decoded, list) or len(decoded) == 0:
        raise RustOracleError("coordinates_json must contain at least one coordinate")

    positions: list[tuple[float, float, float]] = []
    if all(isinstance(row, list) for row in decoded):
        for row in decoded:
            if not isinstance(row, list) or len(row) != 3:
                raise RustOracleError("coordinate rows must contain exactly 3 values")
            positions.append(
                (
                    finite_coordinate_value(row[0]),
                    finite_coordinate_value(row[1]),
                    finite_coordinate_value(row[2]),
                )
            )
        return positions

    if len(decoded) % 3 != 0:
        raise RustOracleError("flat coordinates_json length must be divisible by 3")
    values = [finite_coordinate_value(value) for value in decoded]
    for index in range(0, len(values), 3):
        positions.append((values[index], values[index + 1], values[index + 2]))
    return positions


def finite_coordinate_value(value: object) -> float:
    """Return a finite float coordinate, rejecting bools and non-numeric values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RustOracleError("coordinate value is not numeric")
    coordinate = float(value)
    if not math.isfinite(coordinate):
        raise RustOracleError("coordinate value is not finite")
    return coordinate


def live_score_coordinate_count(
    positions: Sequence[tuple[float, float, float]], score_atom_offset: object
) -> int:
    """Return the number of atoms Rust will score after scaffold-context exclusion."""

    offset = strict_score_atom_offset(score_atom_offset)
    if len(positions) == 0:
        raise RustOracleError("live scoring requires at least one coordinate")
    if offset == 0:
        return len(positions)
    if offset >= len(positions):
        raise RustOracleError("score_atom_offset leaves no atoms to score")
    scaffold = positions[:offset]
    fragment_count = sum(
        1
        for xyz in positions[offset:]
        if min_distance_to_live_coordinates(xyz, scaffold) > FRAGMENT_CONTEXT_EXCLUSION_A
    )
    if fragment_count == 0:
        raise RustOracleError("score_atom_offset leaves no atoms after scaffold-context exclusion")
    return fragment_count


def min_distance_to_live_coordinates(
    xyz: tuple[float, float, float], positions: Sequence[tuple[float, float, float]]
) -> float:
    """Mirror Rust live-fragment scaffold-context exclusion distance."""

    return min(
        math.sqrt(
            (xyz[0] - other[0]) ** 2
            + (xyz[1] - other[1]) ** 2
            + (xyz[2] - other[2]) ** 2
        )
        for other in positions
    )


def strict_score_atom_offset(value: object) -> int:
    """Parse score_atom_offset without truncation or bool coercion."""

    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise RustOracleError("score_atom_offset must be a non-negative integer")
    if value < 0:
        raise RustOracleError("score_atom_offset must be non-negative")
    return value


def strict_nonnegative_finite_float(value: object, name: str) -> float:
    """Parse non-negative finite float inputs without bool coercion."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RustOracleError(f"{name} must be a non-negative finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise RustOracleError(f"{name} must be a non-negative finite number")
    return parsed


def proposals_from_rows(rows: pl.DataFrame, indices: Sequence[int]) -> list[OracleProposal]:
    """Build proposal objects from an anchor/action dataframe."""

    selected_columns = ["anchor_id", "canonical_smiles"]
    for optional_column in ("coordinates_json", "score_atom_offset", "u_pose"):
        if optional_column in rows.columns:
            selected_columns.append(optional_column)
    selected = rows.select(selected_columns).to_dicts()
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
                coordinates_json=(
                    str(row.get("coordinates_json"))
                    if isinstance(row.get("coordinates_json"), str)
                    else None
                ),
                score_atom_offset=strict_score_atom_offset(row.get("score_atom_offset")),
                u_pose=strict_nonnegative_finite_float(row.get("u_pose", 0.0), "u_pose"),
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
    "LiveSignalGridOracle",
    "SurvivorCorpusOracle",
    "OracleBatchResult",
    "OracleProposal",
    "OracleTelemetry",
    "RustOracleError",
    "proposals_from_rows",
    "telemetry_to_dict",
]
