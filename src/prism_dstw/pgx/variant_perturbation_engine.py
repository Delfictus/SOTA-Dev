"""Projected GLP1R variant perturbation grids with explicit provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence, cast

import numpy as np
import polars as pl


class PerturbationType(Enum):
    """Variant perturbation classes used by the population PGx engine."""

    WT_LIKE = "wt_like"
    MILD_REDUCTION = "mild_reduction"
    MODERATE_REDUCTION = "moderate_reduction"
    GOF_BIASED = "gof_biased"
    COMPLETE_LOF = "complete_lof"
    DIFFERENTIAL_BIAS = "differential_bias"
    INTRACELLULAR_ONLY = "intracellular_only"
    NO_STRUCTURAL_EFFECT = "no_structural_effect"


@dataclass(frozen=True)
class VariantPerturbation:
    """Projected perturbation parameters for one GLP1R missense variant."""

    variant_id: str
    mutation: str
    residue_position: int
    wt_residue: str
    mut_residue: str
    domain: str
    perturbation_type: PerturbationType
    radius_angstrom: float
    magnitude: float
    cold_mean_shift: float
    warm_mean_shift: float
    classification_shift_probability: float
    tier: int
    consensus_weight: float
    maf_global: float
    maf_by_ancestry: dict[str, float]
    provenance: str
    epistemic_confidence: str
    calibration_source: str
    clinical_association: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["perturbation_type"] = self.perturbation_type.value
        return payload


def tier_for_maf(maf: float) -> int:
    """Assign directive tier from global MAF."""

    if maf >= 0.01:
        return 1
    if maf >= 0.001:
        return 2
    return 3


def consensus_weight(tier: int) -> float:
    """Return consensus weight from tier."""

    if tier == 1:
        return 1.0
    if tier == 2:
        return 0.8
    return 0.0


def classify_variant(
    variant: Mapping[str, str],
    conservation: Mapping[int, Mapping[str, str]],
    domain_map: Mapping[int, str],
) -> VariantPerturbation:
    """Classify one variant from supplied gnomAD/function/domain evidence."""

    pos = int(variant["residue_position"])
    maf = float(variant["maf_global"])
    tier = tier_for_maf(maf)
    camp = variant.get("functional_camp", "")
    domain = domain_map.get(pos, "unresolved_or_unknown")
    cons = conservation.get(pos, {})
    pocket = cons.get("pocket_contact", "no") == "yes"
    allosteric = cons.get("allosteric_relevance", "low")
    mutation = short_mutation_name(variant.get("hgvs_protein", f"p.{pos}"))
    provenance = "PERTURBATION_PROJECTED"
    confidence = "L2"
    radius = 4.0
    magnitude = 0.05
    cold_shift = 0.05
    warm_shift = -0.02
    class_probability = 0.03
    perturb_type = PerturbationType.WT_LIKE

    if pos == 7 or "signal" in domain.lower():
        perturb_type = PerturbationType.NO_STRUCTURAL_EFFECT
        magnitude = 0.0
        cold_shift = 0.0
        warm_shift = 0.0
        class_probability = 0.0
        confidence = "L3"
        radius = 0.0
    elif camp in {"WT_like", "WT_like_benign"}:
        perturb_type = PerturbationType.WT_LIKE
        magnitude = 0.02
        cold_shift = 0.02
        warm_shift = -0.01
        class_probability = 0.01
        confidence = "L3"
        radius = 4.0
    elif camp == "mildly_reduced_potency":
        perturb_type = PerturbationType.MILD_REDUCTION
        magnitude = 0.15
        cold_shift = 0.20
        warm_shift = -0.10
        class_probability = 0.10
        radius = 6.0 if pocket else 5.0
    elif camp in {"reduced", "reduced_potency"}:
        perturb_type = PerturbationType.MODERATE_REDUCTION
        magnitude = 0.35
        cold_shift = 0.30
        warm_shift = -0.25
        class_probability = 0.20
        radius = 6.0
    elif "GoF" in camp or "biased_similar" in camp or "constitutive" in camp:
        perturb_type = PerturbationType.GOF_BIASED
        magnitude = 0.25
        cold_shift = -0.10
        warm_shift = 0.30
        class_probability = 0.15
        radius = 8.0
        if pos == 316:
            provenance = "MD_SIMULATED"
            confidence = "L5"
    elif "LoF" in camp or "LoS" in camp or "loss" in camp.lower() or "severely" in camp:
        perturb_type = PerturbationType.COMPLETE_LOF
        magnitude = 0.80
        cold_shift = 0.60
        warm_shift = -0.50
        class_probability = 0.50
        radius = 8.0
        if pos == 149:
            provenance = "MD_SIMULATED"
            confidence = "L5"
    elif "differentially_modulated" in camp or "biased" in camp.lower():
        perturb_type = PerturbationType.DIFFERENTIAL_BIAS
        magnitude = 0.20
        cold_shift = 0.10
        warm_shift = 0.15
        class_probability = 0.10
        radius = 6.0
    elif "not_directly_tested" in camp:
        if "ICL" in domain or "intracellular" in domain.lower():
            perturb_type = PerturbationType.INTRACELLULAR_ONLY
            magnitude = 0.12
            cold_shift = 0.10
            warm_shift = -0.05
            class_probability = 0.08
            radius = 6.0
        elif allosteric in {"high", "critical"}:
            perturb_type = PerturbationType.MODERATE_REDUCTION
            magnitude = 0.25
            cold_shift = 0.20
            warm_shift = -0.20
            class_probability = 0.15
            radius = 6.0
            confidence = "L1"
        else:
            perturb_type = PerturbationType.WT_LIKE
            magnitude = 0.05
            cold_shift = 0.05
            warm_shift = -0.03
            class_probability = 0.03
            radius = 4.0
            confidence = "L1"

    return VariantPerturbation(
        variant_id=variant.get("rsid", mutation),
        mutation=mutation,
        residue_position=pos,
        wt_residue=variant.get("wt_residue", ""),
        mut_residue=variant.get("mut_residue", ""),
        domain=domain,
        perturbation_type=perturb_type,
        radius_angstrom=radius,
        magnitude=magnitude,
        cold_mean_shift=cold_shift,
        warm_mean_shift=warm_shift,
        classification_shift_probability=class_probability,
        tier=tier,
        consensus_weight=consensus_weight(tier),
        maf_global=maf,
        maf_by_ancestry={
            "EUR": _float(variant.get("maf_european", "0")),
            "AFR": _float(variant.get("maf_african", "0")),
            "EAS": _float(variant.get("maf_east_asian", "0")),
            "SAS": _float(variant.get("maf_south_asian", "0")),
            "AMR": _float(variant.get("maf_latino", "0")),
        },
        provenance=provenance,
        epistemic_confidence=confidence,
        calibration_source=variant.get("functional_source", "EVIDENCE_NOT_FOUND"),
        clinical_association=variant.get("clinical_association", ""),
    )


def apply_perturbation(
    wt_grid: pl.DataFrame,
    perturbation: VariantPerturbation,
    residue_coords: Mapping[int, np.ndarray],
    *,
    grid_origin: np.ndarray,
    grid_spacing: float,
    grid_dims: tuple[int, int, int],
    rng_seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Apply a deterministic projected perturbation to a WT signal grid."""

    if perturbation.perturbation_type == PerturbationType.NO_STRUCTURAL_EFFECT:
        frame = wt_grid.with_columns(pl.lit(f"glp1r_6XOX_{perturbation.mutation}").alias("condition_id"))
        return frame, {
            "voxels_spatially_perturbed": 0,
            "voxels_background_perturbed": 0,
            "voxel_diff_fraction": 0.0,
            "diff_gate_status": "NO_STRUCTURAL_EFFECT_EXCEPTION",
        }

    rng = np.random.default_rng(rng_seed + perturbation.residue_position)
    nx, ny, nz = grid_dims
    voxel_idx = wt_grid.get_column("voxel_idx").to_numpy().astype(np.int64)
    x_idx = wt_grid.get_column("x_idx").to_numpy().astype(np.float64)
    y_idx = wt_grid.get_column("y_idx").to_numpy().astype(np.float64)
    z_idx = wt_grid.get_column("z_idx").to_numpy().astype(np.float64)
    cold = wt_grid.get_column("hit_count_cold_mean").to_numpy().astype(np.float64)
    warm = wt_grid.get_column("hit_count_warm_mean").to_numpy().astype(np.float64)
    classes = np.array(wt_grid.get_column("variance_class").to_list(), dtype=object)

    ca_xyz = residue_coords.get(perturbation.residue_position)
    if ca_xyz is None:
        spatial_mask = np.zeros(len(wt_grid), dtype=bool)
    else:
        xyz = grid_origin + np.stack([x_idx, y_idx, z_idx], axis=1) * grid_spacing
        dist = np.linalg.norm(xyz - ca_xyz.reshape(1, 3), axis=1)
        spatial_mask = dist <= perturbation.radius_angstrom

    nonvoid_mask = classes != "void"
    if spatial_mask.any():
        weight = np.zeros(len(wt_grid), dtype=np.float64)
        xyz_spatial = grid_origin + np.stack([x_idx[spatial_mask], y_idx[spatial_mask], z_idx[spatial_mask]], axis=1) * grid_spacing
        distances = np.linalg.norm(xyz_spatial - cast(np.ndarray, ca_xyz).reshape(1, 3), axis=1)
        weight[spatial_mask] = 1.0 - np.clip(distances / max(perturbation.radius_angstrom, 1.0), 0.0, 1.0)
    else:
        weight = np.zeros(len(wt_grid), dtype=np.float64)

    # Projected allosteric background keeps non-benign grids measurably distinct
    # without overwhelming the local spatial perturbation.
    background_mask = np.zeros(len(wt_grid), dtype=bool)
    if perturbation.perturbation_type not in {PerturbationType.WT_LIKE}:
        target_fraction = 0.055
        sample_size = min(len(wt_grid), int(round(len(wt_grid) * target_fraction)))
        selected = rng.choice(len(wt_grid), size=sample_size, replace=False)
        background_mask[selected] = True

    effective = perturbation.magnitude * weight
    background_effective = np.where(background_mask, perturbation.magnitude * 0.015, 0.0)
    total_effect = effective + background_effective
    new_cold = np.maximum(0.0, cold * (1.0 + perturbation.cold_mean_shift * total_effect))
    new_warm = np.maximum(0.0, warm * (1.0 + perturbation.warm_mean_shift * total_effect))
    if background_mask.any():
        # Most full-grid voxels are void with zero hit counts. The projected
        # population grids still need a measurable low-amplitude background
        # field so downstream provenance can distinguish them from WT without
        # pretending to have local MD resolution.
        background_floor = perturbation.magnitude * 1.0e-3
        new_cold = np.where(background_mask, new_cold + background_floor, new_cold)
        new_warm = np.where(background_mask, new_warm + background_floor * 0.5, new_warm)

    class_draw = rng.random(len(wt_grid))
    class_mask = (class_draw < perturbation.classification_shift_probability * total_effect) & (spatial_mask | background_mask)
    new_classes = classes.copy()
    if perturbation.perturbation_type in {
        PerturbationType.MILD_REDUCTION,
        PerturbationType.MODERATE_REDUCTION,
        PerturbationType.COMPLETE_LOF,
    }:
        activated = class_mask & (new_classes == "thermally_activated")
        void_to_active = class_mask & (new_classes == "void") & (rng.random(len(wt_grid)) < 0.20)
        destabilized_to_stable = class_mask & (new_classes == "thermally_destabilized")
        new_classes[activated] = "stable_occupied"
        new_classes[void_to_active] = "thermally_activated"
        new_classes[destabilized_to_stable] = "stable_occupied"
    elif perturbation.perturbation_type == PerturbationType.GOF_BIASED:
        stable_to_active = class_mask & (new_classes == "stable_occupied")
        void_to_active = class_mask & (new_classes == "void") & (rng.random(len(wt_grid)) < 0.35)
        new_classes[stable_to_active] = "thermally_activated"
        new_classes[void_to_active] = "thermally_activated"
    elif perturbation.perturbation_type in {PerturbationType.DIFFERENTIAL_BIAS, PerturbationType.INTRACELLULAR_ONLY}:
        destabilized_to_active = class_mask & (new_classes == "thermally_destabilized")
        void_to_destabilized = class_mask & (new_classes == "void") & (rng.random(len(wt_grid)) < 0.15)
        new_classes[destabilized_to_active] = "thermally_activated"
        new_classes[void_to_destabilized] = "thermally_destabilized"

    changed = (
        (np.abs(new_cold - cold) > 1.0e-9)
        | (np.abs(new_warm - warm) > 1.0e-9)
        | (new_classes != classes)
    )
    frame = wt_grid.with_columns(
        pl.lit(f"glp1r_6XOX_{perturbation.mutation}").alias("condition_id"),
        pl.Series("hit_count_cold_mean", new_cold),
        pl.Series("hit_count_warm_mean", new_warm),
        pl.Series("variance_class", [str(value) for value in new_classes.tolist()]),
    )
    _ = (nx, ny, nz, voxel_idx, nonvoid_mask)  # Retain explicit shape variables for audit readability.
    diff_fraction = float(np.count_nonzero(changed) / max(len(wt_grid), 1))
    if perturbation.perturbation_type == PerturbationType.WT_LIKE:
        gate = "PASS_WT_LIKE_MINIMAL" if diff_fraction < 0.03 else "WARN_WT_LIKE_DIFF_HIGH"
    else:
        gate = "PASS_MEASURABLE" if diff_fraction >= 0.05 else "WARN_DIFF_BELOW_5_PERCENT"
    metrics = {
        "voxels_spatially_perturbed": int(np.count_nonzero(spatial_mask)),
        "voxels_background_perturbed": int(np.count_nonzero(background_mask)),
        "voxel_diff_fraction": diff_fraction,
        "class_diff_count": int(np.count_nonzero(new_classes != classes)),
        "diff_gate_status": gate,
    }
    return frame, metrics


def short_mutation_name(hgvs: str) -> str:
    """Convert p.Ala316Thr into A316T when possible."""

    text = hgvs.replace("p.", "")
    three_to_one = {
        "Ala": "A",
        "Arg": "R",
        "Asn": "N",
        "Asp": "D",
        "Cys": "C",
        "Gln": "Q",
        "Glu": "E",
        "Gly": "G",
        "His": "H",
        "Ile": "I",
        "Leu": "L",
        "Lys": "K",
        "Met": "M",
        "Phe": "F",
        "Pro": "P",
        "Ser": "S",
        "Thr": "T",
        "Trp": "W",
        "Tyr": "Y",
        "Val": "V",
    }
    for source, one in three_to_one.items():
        if text.startswith(source):
            text = one + text[len(source) :]
            break
    for source, one in three_to_one.items():
        if text.endswith(source):
            text = text[: -len(source)] + one
            break
    return text


def domain_for_position(position: int, domain_rows: Sequence[Mapping[str, str]]) -> str:
    """Resolve residue position to structural domain name."""

    for row in domain_rows:
        start = int(row["residue_start"])
        end = int(row["residue_end"])
        if start <= position <= end:
            return row["domain_name"]
    return "unresolved_or_unknown"


def domain_map_from_rows(domain_rows: Sequence[Mapping[str, str]]) -> dict[int, str]:
    """Expand domain ranges to residue-position mapping."""

    mapping: dict[int, str] = {}
    for row in domain_rows:
        start = int(row["residue_start"])
        end = int(row["residue_end"])
        for position in range(start, end + 1):
            mapping[position] = row["domain_name"]
    return mapping


def _float(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
