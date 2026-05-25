"""Tripartite biased-agonism scoring with explicit epistemic separation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence, cast


EpistemicConfidence = Literal["L1", "L2", "L3"]
PHASES: tuple[str, ...] = ("cold_hold", "ramp_up", "warm_hold", "ramp_down", "cold_return")


@dataclass(frozen=True)
class TripartiteBiasScore:
    """Three-layer decomposition of biased-agonism potential.

    The observed layer is static geometry from the Rust oracle. The derived
    layer is a proxy from the five-phase lock occupancy profile. The projected
    layer combines the first two layers with literature priors and must not be
    presented as dynamic MD confirmation.
    """

    lock_geometry_score: float
    lock_geometry_atoms: int
    lock_geometry_voxels: list[int]
    lock_persistence_score: float
    lock_phase_profile: dict[str, float]
    lock_hysteresis_asymmetry: float
    bias_projection_score: float
    intracellular_penetration_depth: float
    projected_tm6_blockade_volume: float
    epistemic_confidence: EpistemicConfidence
    mask_version: str = "v2_residue_frame_corrected"
    scoring_method: str = "tripartite_bias_v1"


@dataclass(frozen=True)
class LiteraturePriors:
    """Static prior weights used only for projection, not evidence promotion."""

    w_geo: float = 2.0
    w_per: float = 1.5
    w_depth: float = 1.0
    w_flex: float = 0.8
    w_prior: float = 0.5
    baseline_bias_prior: float = 0.3


@dataclass(frozen=True)
class RewardV2Weights:
    """Weights for the tripartite-aware terminal reward."""

    complement: float = 1.0
    clash_pocket: float = 1.0
    lock_geometry: float = 1.0
    lock_projection: float = 0.5
    shear: float = 1.0
    oral: float = 1.0


def sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""

    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def compute_tripartite_bias(
    oracle_output: Mapping[str, Any],
    literature_priors: LiteraturePriors | None = None,
) -> TripartiteBiasScore:
    """Compute observed, derived, and projected biased-agonism scores."""

    priors = literature_priors or LiteraturePriors()
    geo_score = _float_value(
        oracle_output.get("lock_geometry_score", oracle_output.get("pi_clash_lock", 0.0))
    )
    geo_atoms = _int_value(oracle_output.get("lock_geometry_atom_count", 0))
    phases = _phase_profile(oracle_output, geo_score)

    cold_hold = phases["cold_hold"]
    cold_return = phases["cold_return"]
    ramp_up = phases["ramp_up"]
    if ramp_up < 1.0e-6:
        hysteresis = 0.0
        persistence = 0.0
    else:
        hysteresis = abs(cold_return - cold_hold) / ramp_up
        persistence = 1.0 / (1.0 + hysteresis)

    depth = _float_value(oracle_output.get("intracellular_penetration_depth_angstrom", 0.0))
    volume = _float_value(oracle_output.get("lock_steric_volume_angstrom3", 0.0))
    rotatable = _float_value(oracle_output.get("rotatable_bonds_in_lock_region", 0.0))
    projection = sigmoid(
        priors.w_geo * math.log1p(max(geo_score, 0.0))
        + priors.w_per * persistence
        + priors.w_depth * max(depth, 0.0) / 10.0
        - priors.w_flex * rotatable / 5.0
        + priors.w_prior * priors.baseline_bias_prior
    )

    if geo_atoms >= 3 and persistence > 0.5 and depth > 3.0:
        confidence: EpistemicConfidence = "L3"
    elif geo_atoms >= 1 and depth > 1.0:
        confidence = "L2"
    else:
        confidence = "L1"

    return TripartiteBiasScore(
        lock_geometry_score=geo_score,
        lock_geometry_atoms=geo_atoms,
        lock_geometry_voxels=_voxel_list(oracle_output.get("lock_voxel_indices_json")),
        lock_persistence_score=persistence,
        lock_phase_profile=phases,
        lock_hysteresis_asymmetry=hysteresis,
        bias_projection_score=projection,
        intracellular_penetration_depth=depth,
        projected_tm6_blockade_volume=volume,
        epistemic_confidence=confidence,
    )


def compute_reward_v2(
    oracle_output: Mapping[str, Any],
    bias_score: TripartiteBiasScore,
    weights: RewardV2Weights | None = None,
) -> float:
    """Compute the tripartite-aware terminal reward.

    Persistence is intentionally excluded from the reward because it is a
    derived static proxy. It remains a ranking and dossier field until GPU MD
    validates or refutes the dynamic wedge.
    """

    w = weights or RewardV2Weights()
    pi_complement = _float_value(oracle_output.get("pi_complement", 0.0))
    pi_clash_pocket = _float_value(oracle_output.get("pi_clash_pocket", 0.0))
    sigma_shear = _float_value(oracle_output.get("sigma_shear", 0.0))
    oral_violation = _float_value(oracle_output.get("oral_violation", 0.0))
    u_pose = _float_value(oracle_output.get("u_pose", 0.0))
    reward = (
        w.complement * pi_complement
        - w.clash_pocket * pi_clash_pocket
        + w.lock_geometry * bias_score.lock_geometry_score
        + w.lock_projection * bias_score.bias_projection_score
        - w.shear * math.log1p(max(sigma_shear, 0.0))
        - w.oral * oral_violation
        - u_pose
    )
    return max(reward, 1.0e-8)


def score_row(
    row: Mapping[str, Any],
    literature_priors: LiteraturePriors | None = None,
    reward_weights: RewardV2Weights | None = None,
) -> dict[str, Any]:
    """Return JSON-ready tripartite fields for one oracle/candidate row."""

    bias = compute_tripartite_bias(row, literature_priors)
    output = asdict(bias)
    output["reward_v2_tripartite"] = compute_reward_v2(row, bias, reward_weights)
    return output


def score_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Score multiple rows with default priors and weights."""

    return [score_row(row) for row in rows]


def _phase_profile(row: Mapping[str, Any], fallback: float) -> dict[str, float]:
    profile: dict[str, float] = {}
    for phase in PHASES:
        profile[phase] = _float_value(
            row.get(f"lock_occupancy_{phase}", row.get(f"pi_clash_lock_{phase}", fallback))
        )
    return profile


def _voxel_list(raw_value: object) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        if raw_value == "":
            return []
        decoded = json.loads(raw_value)
    else:
        decoded = raw_value
    if not isinstance(decoded, list):
        return []
    voxels: list[int] = []
    for value in decoded:
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float | str):
            voxels.append(int(value))
    return voxels


def _float_value(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float | str):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else 0.0
    return 0.0


def _int_value(value: object) -> int:
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(float(value))
    return 0


def tripartite_bias_from_json(payload: str) -> TripartiteBiasScore:
    """Load a score from JSON emitted by dossier/profile scripts."""

    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("tripartite bias payload must decode to an object")
    data = cast(dict[str, Any], decoded)
    confidence = data.get("epistemic_confidence", "L1")
    if confidence not in ("L1", "L2", "L3"):
        confidence = "L1"
    raw_phase_profile = data.get("lock_phase_profile", {})
    phase_profile = cast(Mapping[str, Any], raw_phase_profile) if isinstance(raw_phase_profile, dict) else {}
    return TripartiteBiasScore(
        lock_geometry_score=_float_value(data.get("lock_geometry_score")),
        lock_geometry_atoms=_int_value(data.get("lock_geometry_atoms")),
        lock_geometry_voxels=_voxel_list(data.get("lock_geometry_voxels")),
        lock_persistence_score=_float_value(data.get("lock_persistence_score")),
        lock_phase_profile={
            phase: _float_value(phase_profile.get(phase))
            for phase in PHASES
        },
        lock_hysteresis_asymmetry=_float_value(data.get("lock_hysteresis_asymmetry")),
        bias_projection_score=_float_value(data.get("bias_projection_score")),
        intracellular_penetration_depth=_float_value(data.get("intracellular_penetration_depth")),
        projected_tm6_blockade_volume=_float_value(data.get("projected_tm6_blockade_volume")),
        epistemic_confidence=cast(EpistemicConfidence, confidence),
        mask_version=str(data.get("mask_version", "v2_residue_frame_corrected")),
        scoring_method=str(data.get("scoring_method", "tripartite_bias_v1")),
    )
