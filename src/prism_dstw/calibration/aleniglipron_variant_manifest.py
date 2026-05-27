"""Expanded Aleniglipron variant run-manifest construction.

This module turns the compact Track B calibration panel into an execution
manifest for true perturbed-trajectory work. It is intentionally conservative:
projected probes are queued as work, not labeled as observed trajectory data.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl

from prism_dstw.calibration.track_b_artifacts import read_json, sha256_file, write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso
from prism_dstw.calibration.translational_calibration_manifold import (
    PERTURBATION_SPACE,
    default_observability_for_region,
)

AA3_TO_1: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

FAMILY_TARGETS: dict[str, tuple[str, ...]] = {
    "SEVERING_PROBE": ("G", "P", "A"),
    "RIGIDIFYING_PROBE": ("W", "P", "F"),
    "FLEXIBILIZING_PROBE": ("S", "G", "A"),
    "CHARGE_INVERSION_PROBE": ("K", "R", "D", "E"),
    "HYDRATION_WIRE_PROBE": ("T", "S", "N"),
    "CONSERVATIVE_CONTROL": ("V", "I", "L", "A"),
    "VOLUME_EXPANSION_PROBE": ("F", "W", "Y"),
    "AROMATIC_STACKING_PROBE": ("Y", "F", "W"),
}

FAMILY_PURPOSE: dict[str, str] = {
    "SEVERING_PROBE": "break propagation edges and expose causal fragility",
    "RIGIDIFYING_PROBE": "test whether stiffening preserves chronological control",
    "FLEXIBILIZING_PROBE": "test whether added local motion restores continuity",
    "CHARGE_INVERSION_PROBE": "stress electrostatic relay and salt-bridge directionality",
    "HYDRATION_WIRE_PROBE": "probe solvent-wire preservation and occlusion liability",
    "CONSERVATIVE_CONTROL": "control for backbone/context sensitivity without gross chemistry change",
    "VOLUME_EXPANSION_PROBE": "test steric crowding and cryptic-pocket fracture",
    "AROMATIC_STACKING_PROBE": "test aromatic relay, pi-stack, and activation-pathway support",
}

NO_FLY_RESIDUES = {182}
MAX_RESIDUE = 500


@dataclass(frozen=True)
class ResidueSeed:
    """One residue/topology context eligible for variant expansion."""

    residue_id: str
    residue_position: int
    source_aa: str
    topology_region: str
    genotype_axis: str
    selection_features: tuple[str, ...]
    score_components: dict[str, float]
    source_artifacts: tuple[str, ...]
    seed_kind: str


@dataclass(frozen=True)
class BuildResult:
    """All expanded variant-run outputs before writing to disk."""

    manifest: dict[str, Any]
    variant_rows: list[dict[str, Any]]
    target_rows: list[dict[str, Any]]
    target_report: dict[str, Any]
    run_plan: dict[str, Any]
    validation_report: dict[str, Any]
    runbook_markdown: str


def parse_residue_identity(residue_id: str) -> tuple[str, int]:
    """Return one-letter amino acid and residue position from IDs like PHE367."""

    match = re.match(r"^([A-Z]{3})(\d+)$", residue_id)
    if match is None:
        digits = re.findall(r"\d+", residue_id)
        position = int(digits[-1]) if digits else 0
        return "X", position
    position = int(match.group(2))
    if position < 1 or position > MAX_RESIDUE:
        return "X", position
    return AA3_TO_1.get(match.group(1), "X"), position


def mutation_label(source_aa: str, residue_position: int, target_aa: str) -> str:
    """Build compact mutation label such as A316T."""

    return f"{source_aa}{residue_position}{target_aa}"


def region_for_position(position: int) -> str:
    """Map a GLP1R residue position to the nearest Track B topology region."""

    if position <= 144:
        return "ECD_TM1_GATEWAY"
    if position <= 199:
        return "TM3_TM5_CORE"
    if position <= 260:
        return "HYDRATION_CORRIDOR"
    if position <= 330:
        return "BISTATE_RETAINED_RESIDUES"
    if position <= 390:
        return "TE_HUBS"
    return "INTRACELLULAR_LOCK_BASIN"


def targets_for_family(family: str, source_aa: str, *, limit: int = 3) -> tuple[str, ...]:
    """Return non-self target amino acids for one perturbation family."""

    pool: tuple[str, ...]
    if family == "CHARGE_INVERSION_PROBE":
        pool = ("D", "E", "K", "R") if source_aa in {"K", "R", "H"} else ("K", "R", "D", "E")
    else:
        pool = FAMILY_TARGETS[family]
    targets = tuple(target for target in pool if target != source_aa)
    return targets[:limit]


def infer_family(source_aa: str, target_aa: str, functional_text: str = "") -> str:
    """Infer a perturbation family for an observed or known PGx mutation."""

    charged = {"D", "E", "K", "R", "H"}
    aromatic = {"F", "Y", "W"}
    text = functional_text.lower()
    if source_aa in charged and target_aa in charged and source_aa != target_aa:
        return "CHARGE_INVERSION_PROBE"
    if any(word in text for word in ("complete_loss", "severely", "loss", "lof", "los")):
        return "SEVERING_PROBE"
    if target_aa == "P" or target_aa in aromatic:
        return "RIGIDIFYING_PROBE"
    if target_aa in {"G", "S"}:
        return "FLEXIBILIZING_PROBE"
    if target_aa in {"T", "N", "Q"}:
        return "HYDRATION_WIRE_PROBE"
    return "CONSERVATIVE_CONTROL"


def _stable_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def _normalize(value: float, scale: float) -> float:
    if not math.isfinite(value) or scale <= 0.0:
        return 0.0
    return max(0.0, min(value / scale, 1.0))


def _as_float(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, int | float | str):
        try:
            parsed = float(value)
        except ValueError:
            return 0.0
        return parsed if math.isfinite(parsed) else 0.0
    return 0.0


def _json_dump(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _path_sha(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }


def _topology_aa_by_position(topology_path: Path) -> dict[int, str]:
    if not topology_path.exists():
        return {}
    payload = json.loads(topology_path.read_text(encoding="utf-8"))
    mapping: dict[int, str] = {}
    for residue in payload.get("residues", []):
        if not isinstance(residue, dict):
            continue
        residue_id = int(residue.get("residue_id", 0) or 0)
        residue_name = str(residue.get("residue_name", ""))
        one_letter = AA3_TO_1.get(residue_name.upper())
        if residue_id and one_letter is not None:
            mapping[residue_id] = one_letter
    return mapping


def _conservation_by_position(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        int(row["residue_position"]): row
        for row in pl.read_csv(path).to_dicts()
        if row.get("residue_position") is not None
    }


def _population_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pl.read_csv(path).to_dicts()


def _phase3_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    rows = payload.get("high_risk_variants", [])
    return [row for row in rows if isinstance(row, dict)]


def _aggregate_propagation(path: Path) -> tuple[dict[str, dict[str, float]], dict[int, dict[str, float]]]:
    if not path.exists():
        return {}, {}
    frame = pl.read_parquet(path)
    by_mutation: dict[str, dict[str, float]] = {}
    if frame.height:
        for row in frame.group_by("mutation_label").agg(
            pl.len().alias("edge_count"),
            pl.col("propagation_delta_risk").mean().alias("mean_delta_risk"),
            pl.col("propagation_delta_risk").max().alias("max_delta_risk"),
            pl.col("resilience_shift_signed_te").abs().mean().alias("mean_abs_te_shift"),
        ).to_dicts():
            by_mutation[str(row["mutation_label"])] = {
                "propagation_edge_count": _as_float(row["edge_count"]),
                "propagation_mean_delta_risk": _as_float(row["mean_delta_risk"]),
                "propagation_max_delta_risk": _as_float(row["max_delta_risk"]),
                "propagation_mean_abs_te_shift": _as_float(row["mean_abs_te_shift"]),
            }
    edge_rows = frame.select(["edge_from_residue", "edge_to_residue", "propagation_delta_risk"]).to_dicts()
    by_residue: dict[int, dict[str, float]] = {}
    for edge in edge_rows:
        for key in ("edge_from_residue", "edge_to_residue"):
            position = int(edge[key])
            entry = by_residue.setdefault(position, {"edge_count": 0.0, "max_delta_risk": 0.0, "sum_delta_risk": 0.0})
            delta = abs(_as_float(edge["propagation_delta_risk"]))
            entry["edge_count"] += 1.0
            entry["sum_delta_risk"] += delta
            entry["max_delta_risk"] = max(entry["max_delta_risk"], delta)
    for entry in by_residue.values():
        count = max(entry["edge_count"], 1.0)
        entry["mean_delta_risk"] = entry["sum_delta_risk"] / count
    return by_mutation, by_residue


def _aggregate_chronology(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    frame = pl.read_parquet(path)
    if "residues" not in frame.columns or frame.height == 0:
        return {}
    exploded = frame.explode("residues").filter(pl.col("residues").is_not_null())
    if exploded.height == 0:
        return {}
    rows = exploded.group_by("residues").agg(
        pl.len().alias("event_count"),
        pl.col("temporal_overlap_entropy").mean().alias("mean_temporal_overlap_entropy"),
        pl.col("confidence").mean().alias("mean_confidence"),
    ).to_dicts()
    return {
        int(row["residues"]): {
            "chronology_event_count": _as_float(row["event_count"]),
            "chronology_mean_entropy": _as_float(row["mean_temporal_overlap_entropy"]),
            "chronology_mean_confidence": _as_float(row["mean_confidence"]),
        }
        for row in rows
    }


def _aggregate_thermodynamic(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    frame = pl.read_parquet(path)
    if "residue_id" not in frame.columns or frame.height == 0:
        return {}
    rows = (
        frame.with_columns(
            pl.col("residue_id").str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False).alias("residue_position")
        )
        .filter(pl.col("residue_position").is_not_null())
        .group_by("residue_position")
        .agg(
            pl.col("trap_risk").mean().alias("trap_risk"),
            pl.col("hysteresis_score").mean().alias("hysteresis_score"),
            pl.col("transition_event_density").mean().alias("transition_event_density"),
            pl.col("recovery_likelihood").mean().alias("recovery_likelihood"),
        )
        .to_dicts()
    )
    return {
        int(row["residue_position"]): {
            "trap_risk": _as_float(row["trap_risk"]),
            "hysteresis_score": _as_float(row["hysteresis_score"]),
            "transition_event_density": _as_float(row["transition_event_density"]),
            "recovery_likelihood": _as_float(row["recovery_likelihood"]),
        }
        for row in rows
    }


def _aggregate_nma(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    frame = pl.read_parquet(path)
    if "residue_id" not in frame.columns or frame.height == 0:
        return {}
    rows = (
        frame.with_columns(pl.col("residue_id").cast(pl.Int64, strict=False).alias("residue_position"))
        .filter(pl.col("residue_position").is_not_null())
        .group_by("residue_position")
        .agg(
            pl.col("hinge_disruption_risk").mean().alias("hinge_disruption_risk"),
            pl.col("mode_displacement_norm").mean().alias("mode_displacement_norm"),
            pl.col("hinge_residue_flag").cast(pl.Int64).max().alias("hinge_residue_flag"),
        )
        .to_dicts()
    )
    return {
        int(row["residue_position"]): {
            "hinge_disruption_risk": _as_float(row["hinge_disruption_risk"]),
            "mode_displacement_norm": _as_float(row["mode_displacement_norm"]),
            "hinge_residue_flag": _as_float(row["hinge_residue_flag"]),
        }
        for row in rows
    }


def _build_registry_seeds(
    registry_path: Path,
    topology_aa: Mapping[int, str],
    conservation: Mapping[int, Mapping[str, Any]],
) -> list[ResidueSeed]:
    if not registry_path.exists():
        return []
    registry = read_json(registry_path)
    seeds: list[ResidueSeed] = []
    seen: set[tuple[str, int, str]] = set()
    for region, payload in dict(registry.get("regions", {})).items():
        for residue in payload.get("residues", []):
            if not isinstance(residue, dict):
                continue
            residue_id = str(residue.get("residue_id", ""))
            parsed_aa, position = parse_residue_identity(residue_id)
            source_aa = parsed_aa if parsed_aa != "X" else topology_aa.get(position, "X")
            if source_aa == "X":
                continue
            conservation_row = conservation.get(position)
            genotype_axis = (
                "species_divergent_residues"
                if conservation_row and _as_float(conservation_row.get("conservation_score")) < 1.0
                else "conserved_but_dynamically_sensitive_residues"
            )
            score_components = {
                key: _as_float(value)
                for key, value in dict(residue.get("score_components", {})).items()
            }
            features = tuple(sorted(set(residue.get("evidence_columns", []) or ["topology_region_registry"])))
            key = (str(region), position, source_aa)
            if key in seen:
                continue
            seen.add(key)
            seeds.append(
                ResidueSeed(
                    residue_id=residue_id,
                    residue_position=position,
                    source_aa=source_aa,
                    topology_region=str(region),
                    genotype_axis=genotype_axis,
                    selection_features=features,
                    score_components=score_components,
                    source_artifacts=(registry_path.as_posix(),),
                    seed_kind="track_b_topology_registry",
                )
            )
    return seeds


def _build_population_seeds(
    rows: Iterable[Mapping[str, Any]],
    conservation: Mapping[int, Mapping[str, Any]],
    population_path: Path,
) -> list[ResidueSeed]:
    seeds: list[ResidueSeed] = []
    for row in rows:
        position = int(row.get("residue_position", 0) or 0)
        source_aa = str(row.get("wt_residue", "X") or "X")
        target_aa = str(row.get("mut_residue", "X") or "X")
        if not position or source_aa == "X" or target_aa == "X":
            continue
        region = region_for_position(position)
        conservation_row = conservation.get(position)
        features = {
            "population_frequency",
            "functional_assay_annotation",
            "structural_domain_mapping",
        }
        if conservation_row is not None:
            features.add("species_conservation")
        clinical_text = " ".join(
            str(row.get(key, ""))
            for key in ("functional_camp", "functional_barr2", "functional_erk", "clinical_association")
        ).lower()
        functional_risk = 1.0 if any(token in clinical_text for token in ("loss", "reduced", "severe", "lof", "los")) else 0.4
        if "gof" in clinical_text or "constitutive" in clinical_text:
            functional_risk = max(functional_risk, 0.7)
        score_components = {
            "maf_global": _as_float(row.get("maf_global")),
            "functional_risk": functional_risk,
            "exact_population_variant": 1.0,
            "species_divergence": 1.0 - _as_float(conservation_row.get("conservation_score")) if conservation_row else 0.0,
        }
        seeds.append(
            ResidueSeed(
                residue_id=f"{source_aa}{position}",
                residue_position=position,
                source_aa=source_aa,
                topology_region=region,
                genotype_axis="population_variants",
                selection_features=tuple(sorted(features)),
                score_components=score_components,
                source_artifacts=(population_path.as_posix(),),
                seed_kind="population_pgx_source",
            )
        )
    return seeds


def _build_chronology_seeds(
    chronology: Mapping[int, Mapping[str, float]],
    topology_aa: Mapping[int, str],
    chronology_path: Path,
    *,
    limit: int = 24,
) -> list[ResidueSeed]:
    ranked = sorted(
        [(position, metrics) for position, metrics in chronology.items() if position > 0],
        key=lambda item: item[1].get("chronology_event_count", 0.0),
        reverse=True,
    )[:limit]
    seeds: list[ResidueSeed] = []
    for position, metrics in ranked:
        source_aa = topology_aa.get(position, "X")
        if source_aa == "X":
            continue
        seeds.append(
            ResidueSeed(
                residue_id=f"{source_aa}{position}",
                residue_position=position,
                source_aa=source_aa,
                topology_region=region_for_position(position),
                genotype_axis="lineage_related_variants",
                selection_features=("transition_chronology", "temporal_overlap_entropy", "runtime_event_density"),
                score_components=dict(metrics),
                source_artifacts=(chronology_path.as_posix(),),
                seed_kind="transition_chronology_hotspot",
            )
        )
    return seeds


def _dedupe_seeds(seeds: Iterable[ResidueSeed]) -> list[ResidueSeed]:
    merged: dict[tuple[str, int, str], ResidueSeed] = {}
    for seed in seeds:
        key = (seed.topology_region, seed.residue_position, seed.source_aa)
        existing = merged.get(key)
        if existing is None:
            merged[key] = seed
            continue
        features = tuple(sorted(set(existing.selection_features) | set(seed.selection_features)))
        score_components = dict(existing.score_components)
        score_components.update(seed.score_components)
        source_artifacts = tuple(sorted(set(existing.source_artifacts) | set(seed.source_artifacts)))
        genotype_axis = existing.genotype_axis
        if seed.genotype_axis == "population_variants":
            genotype_axis = seed.genotype_axis
        merged[key] = ResidueSeed(
            residue_id=existing.residue_id,
            residue_position=existing.residue_position,
            source_aa=existing.source_aa,
            topology_region=existing.topology_region,
            genotype_axis=genotype_axis,
            selection_features=features,
            score_components=score_components,
            source_artifacts=source_artifacts,
            seed_kind=f"{existing.seed_kind}+{seed.seed_kind}",
        )
    return sorted(merged.values(), key=lambda seed: (seed.topology_region, seed.residue_position, seed.source_aa))


def _observed_grid_lookup(pop_manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not pop_manifest_path.exists():
        return {}
    payload = read_json(pop_manifest_path)
    lookup: dict[str, dict[str, Any]] = {}
    for row in payload.get("variants", []):
        if not isinstance(row, dict):
            continue
        mutation = str(row.get("mutation", ""))
        if mutation:
            lookup[mutation] = row
    return lookup


def _variant_score(
    seed: ResidueSeed,
    family: str,
    mutation: str,
    conservation: Mapping[int, Mapping[str, Any]],
    propagation_by_mutation: Mapping[str, Mapping[str, float]],
    propagation_by_residue: Mapping[int, Mapping[str, float]],
    chronology_by_residue: Mapping[int, Mapping[str, float]],
    thermo_by_residue: Mapping[int, Mapping[str, float]],
    nma_by_residue: Mapping[int, Mapping[str, float]],
) -> dict[str, float]:
    scores = dict(seed.score_components)
    position = seed.residue_position
    scores.update({f"propagation_{key}": value for key, value in propagation_by_residue.get(position, {}).items()})
    scores.update({f"exact_mutation_{key}": value for key, value in propagation_by_mutation.get(mutation, {}).items()})
    scores.update(chronology_by_residue.get(position, {}))
    scores.update(thermo_by_residue.get(position, {}))
    scores.update(nma_by_residue.get(position, {}))
    conservation_row = conservation.get(position)
    conservation_score = _as_float(conservation_row.get("conservation_score")) if conservation_row else 1.0
    scores["species_divergence"] = max(scores.get("species_divergence", 0.0), 1.0 - conservation_score)
    scores["no_fly_zone"] = 1.0 if position in NO_FLY_RESIDUES else 0.0
    scores["family_risk_weight"] = {
        "SEVERING_PROBE": 0.85,
        "CHARGE_INVERSION_PROBE": 0.75,
        "VOLUME_EXPANSION_PROBE": 0.65,
        "RIGIDIFYING_PROBE": 0.55,
        "FLEXIBILIZING_PROBE": 0.45,
        "AROMATIC_STACKING_PROBE": 0.45,
        "HYDRATION_WIRE_PROBE": 0.35,
        "CONSERVATIVE_CONTROL": 0.15,
    }[family]
    translation = max(scores.get("translation_pathway", 0.0), scores.get("pathway_rank_score", 0.0))
    phase = max(scores.get("phase_coherence_rank", 0.0), scores.get("coherence_score", 0.0))
    shear = _normalize(scores.get("mean_abs_load", 0.0), 3000.0)
    chronology = _normalize(scores.get("chronology_event_count", 0.0), 1000.0)
    propagation = _normalize(
        max(scores.get("propagation_max_delta_risk", 0.0), scores.get("exact_mutation_propagation_max_delta_risk", 0.0)),
        13.5,
    )
    trap = _normalize(scores.get("trap_risk", 0.0), 1.0)
    functional = _normalize(scores.get("functional_risk", 0.0), 1.0)
    population = min(math.sqrt(max(scores.get("maf_global", 0.0), 0.0)), 1.0)
    hinge = _normalize(scores.get("hinge_disruption_risk", 0.0), 1.0)
    species = _normalize(scores.get("species_divergence", 0.0), 1.0)
    targetability = (
        0.22 * translation
        + 0.16 * phase
        + 0.16 * chronology
        + 0.16 * propagation
        + 0.10 * functional
        + 0.10 * population
        + 0.10 * species
        - 0.14 * trap
        - 0.12 * shear
    )
    avoidance = (
        0.32 * scores["no_fly_zone"]
        + 0.22 * trap
        + 0.16 * shear
        + 0.12 * scores["family_risk_weight"]
        + 0.08 * hinge
        + 0.10 * functional
    )
    priority = max(0.0, min(1.0, targetability + 0.35 * avoidance))
    scores["targetability_score"] = max(0.0, min(1.0, targetability))
    scores["avoidance_score"] = max(0.0, min(1.0, avoidance))
    scores["priority_score"] = max(0.0, min(1.0, priority))
    return scores


def _target_decision(seed: ResidueSeed, scores: Mapping[str, float], family: str) -> tuple[str, str, str]:
    if scores.get("no_fly_zone", 0.0) > 0.0:
        return (
            "AVOID_DIRECT_SMALL_MOLECULE_PRESSURE",
            "ASN182 no-fly geometry makes direct occupancy a design liability; keep as falsification sentinel.",
            "negative_control_or_exclusion_boundary",
        )
    if seed.topology_region == "HYDRATION_CORRIDOR":
        return (
            "PRESERVE_DO_NOT_OCCLUDE",
            "Hydration corridor should be protected unless the perturbation is a hydration-wire control.",
            "solvent_continuity_guardrail",
        )
    if scores.get("avoidance_score", 0.0) >= 0.72 and family in {"SEVERING_PROBE", "CHARGE_INVERSION_PROBE"}:
        return (
            "AVOID_OR_USE_AS_FALSIFICATION_CONTROL",
            "High trap/shear/functional-risk score makes this useful as an avoid boundary, not a primary design target.",
            "falsification_control",
        )
    if seed.topology_region in {"TE_HUBS", "INTRACELLULAR_LOCK_BASIN"} and scores.get("targetability_score", 0.0) >= 0.30:
        return (
            "TARGET_FOR_BIASED_CHRONOLOGICAL_CONTROL",
            "Topology and chronology channels support causal rerouting or quiet-lock control without relying on endpoint-only scoring.",
            "primary_design_target",
        )
    if seed.topology_region == "BISTATE_RETAINED_RESIDUES":
        return (
            "TARGET_FOR_TRANSITION_CONTINUITY_TEST",
            "Bistate-retained region is suited for transition-continuity perturbation and chronology-window falsification.",
            "transition_continuity_target",
        )
    if seed.genotype_axis == "population_variants":
        return (
            "MONITOR_AS_PGX_BACKGROUND",
            "Population/functional evidence makes this a resilience-screening background for candidate triage.",
            "pgx_resilience_background",
        )
    return (
        "SECONDARY_CALIBRATION_TARGET",
        "Multi-channel evidence is sufficient for calibration but lower priority than TE/lock/PGx sentinels.",
        "secondary_design_target",
    )


def _run_priority(scores: Mapping[str, float], execution_status: str) -> str:
    if execution_status.startswith("BLOCKED"):
        return "BLOCKED"
    if scores.get("no_fly_zone", 0.0) > 0.0:
        return "P0_AVOID_SENTINEL"
    if execution_status.startswith("OBSERVED_GRID_AVAILABLE"):
        return "P0_OBSERVED_EXTENSION"
    priority = scores.get("priority_score", 0.0)
    if priority >= 0.62:
        return "P0_TRUE_TRAJECTORY"
    if priority >= 0.46:
        return "P1_TRUE_TRAJECTORY"
    if priority >= 0.30:
        return "P2_TRUE_TRAJECTORY"
    return "P3_CONTROL_TRAJECTORY"


def _engine_commands(variant_id: str, topology_path: str, output_dir: str) -> dict[str, str]:
    base = (
        "scripts/prism-validate-and-run.sh "
        f"-t {topology_path} "
        f"-o {output_dir} "
        "--fast --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70 "
        "--fused-steps 6 --hmr --adaptive-dt --multi-differential "
        "--closed-loop-steering --asymmetric-steering --use-xgb-ranker "
        "--replica-seed 42 -v"
    )
    return {
        "pilot_command": base,
        "full_trajectory_command_template": base.replace("--multi-stream 8", "--multi-stream 20"),
        "candidate_id": variant_id,
    }


def _make_row(
    seed: ResidueSeed,
    family: str,
    target_aa: str,
    conservation: Mapping[int, Mapping[str, Any]],
    propagation_by_mutation: Mapping[str, Mapping[str, float]],
    propagation_by_residue: Mapping[int, Mapping[str, float]],
    chronology_by_residue: Mapping[int, Mapping[str, float]],
    thermo_by_residue: Mapping[int, Mapping[str, float]],
    nma_by_residue: Mapping[int, Mapping[str, float]],
    observed_grid_lookup: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    mutation = mutation_label(seed.source_aa, seed.residue_position, target_aa)
    variant_id = _stable_id(f"ALENI-{seed.topology_region}-{family}-{mutation}")
    scores = _variant_score(
        seed,
        family,
        mutation,
        conservation,
        propagation_by_mutation,
        propagation_by_residue,
        chronology_by_residue,
        thermo_by_residue,
        nma_by_residue,
    )
    decision, rationale, design_role = _target_decision(seed, scores, family)
    observed_grid = observed_grid_lookup.get(mutation)
    if seed.source_aa == "X":
        execution_status = "BLOCKED_SOURCE_AA_UNKNOWN"
    elif seed.residue_position < 24:
        execution_status = "BLOCKED_UNMAPPED_SIGNAL_PEPTIDE"
    elif observed_grid and str(observed_grid.get("provenance")) == "MD_SIMULATED":
        execution_status = "OBSERVED_GRID_AVAILABLE_QUEUE_TRUE_TRAJECTORY_EXTENSION"
    elif observed_grid:
        execution_status = "PROJECTED_GRID_AVAILABLE_QUEUE_TRUE_TRAJECTORY"
    else:
        execution_status = "QUEUED_TRUE_PERTURBED_TRAJECTORY"
    run_priority = _run_priority(scores, execution_status)
    topology_path = output_root / "variant_topologies" / f"{variant_id}.topology.json"
    trajectory_output_dir = output_root / "trajectory_outputs" / variant_id
    commands = _engine_commands(variant_id, topology_path.as_posix(), trajectory_output_dir.as_posix())
    features = sorted(
        set(seed.selection_features)
        | {
            "topology_region",
            "perturbation_family",
            "observability_space",
        }
    )
    if conservation.get(seed.residue_position) is not None:
        features.append("species_conservation")
    if scores.get("chronology_event_count", 0.0) > 0.0:
        features.append("transition_chronology")
    if scores.get("propagation_edge_count", 0.0) > 0.0:
        features.append("propagation_delta")
    provenance = "L5_OBSERVED" if execution_status.startswith("OBSERVED_GRID_AVAILABLE") else "L2_PROJECTED"
    source_artifacts = sorted(
        set(seed.source_artifacts)
        | {
            path
            for row in [observed_grid]
            if row is not None
            for path in [str(row.get("grid_path", ""))]
            if path
        }
    )
    return {
        "id": variant_id,
        "variant_id": variant_id,
        "mutation": mutation,
        "residue_id": seed.residue_id,
        "residue_position": seed.residue_position,
        "source_amino_acid": seed.source_aa,
        "target_amino_acid": target_aa,
        "topology_region": seed.topology_region,
        "genotype_axis": seed.genotype_axis,
        "perturbation_family": family,
        "perturbation_purpose": FAMILY_PURPOSE[family],
        "observability_channels": list(default_observability_for_region(seed.topology_region)),
        "selection_features": sorted(set(features)),
        "selection_feature_count": len(set(features)),
        "score_components": scores,
        "priority_score": scores["priority_score"],
        "targetability_score": scores["targetability_score"],
        "avoidance_score": scores["avoidance_score"],
        "target_decision": decision,
        "target_rationale": rationale,
        "design_role": design_role,
        "run_priority": run_priority,
        "trajectory_execution_status": execution_status,
        "trajectory_claim_status": "NOT_OBSERVED_UNTIL_PRISM_OUTPUT_EXISTS",
        "observed_grid_available": observed_grid is not None,
        "observed_grid_path": str(observed_grid.get("grid_path", "")) if observed_grid else None,
        "variant_topology_path": topology_path.as_posix(),
        "trajectory_output_dir": trajectory_output_dir.as_posix(),
        "pilot_command": commands["pilot_command"],
        "full_trajectory_command_template": commands["full_trajectory_command_template"],
        "replicas_required_for_full_observatory": 80,
        "streams_required_for_full_observatory": 1600,
        "ccns_protocol": "5-phase CCNS hysteresis",
        "seed_kind": seed.seed_kind,
        "provenance_class": provenance,
        "source_artifacts": source_artifacts,
        "evidence_paths": source_artifacts,
        "created_at": utc_now_iso(),
        "schema_version": "track_b.aleniglipron_expanded_variant_run.v1",
    }


def build_expanded_variant_run(
    *,
    topology_registry: Path,
    genealogical_panel: Path,
    population_variants: Path,
    population_manifest: Path,
    phase3_manifest: Path,
    conservation_path: Path,
    propagation_deltas: Path,
    chronology_tensor: Path,
    thermodynamic_continuity: Path,
    nma_continuity: Path,
    topology_path: Path,
    output_root: Path,
    max_target_choices_per_family: int = 3,
) -> BuildResult:
    """Build an expanded run manifest from Track B and PGx source artifacts."""

    topology_aa = _topology_aa_by_position(topology_path)
    conservation = _conservation_by_position(conservation_path)
    population_rows = _population_rows(population_variants)
    propagation_by_mutation, propagation_by_residue = _aggregate_propagation(propagation_deltas)
    chronology_by_residue = _aggregate_chronology(chronology_tensor)
    thermo_by_residue = _aggregate_thermodynamic(thermodynamic_continuity)
    nma_by_residue = _aggregate_nma(nma_continuity)
    observed_grid_lookup = _observed_grid_lookup(population_manifest)
    registry_seeds = _build_registry_seeds(topology_registry, topology_aa, conservation)
    population_seeds = _build_population_seeds(population_rows, conservation, population_variants)
    chronology_seeds = _build_chronology_seeds(chronology_by_residue, topology_aa, chronology_tensor)
    seeds = _dedupe_seeds([*registry_seeds, *population_seeds, *chronology_seeds])

    rows_by_id: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        for family in PERTURBATION_SPACE:
            for target_aa in targets_for_family(family, seed.source_aa, limit=max_target_choices_per_family):
                row = _make_row(
                    seed,
                    family,
                    target_aa,
                    conservation,
                    propagation_by_mutation,
                    propagation_by_residue,
                    chronology_by_residue,
                    thermo_by_residue,
                    nma_by_residue,
                    observed_grid_lookup,
                    output_root,
                )
                rows_by_id[str(row["variant_id"])] = row

    # Ensure exact population variants are always present, even if their target
    # amino acid is outside the synthetic family target pool.
    for row in population_rows:
        position = int(row.get("residue_position", 0) or 0)
        source_aa = str(row.get("wt_residue", "X") or "X")
        target_aa = str(row.get("mut_residue", "X") or "X")
        if position <= 0 or source_aa == "X" or target_aa == "X":
            continue
        clinical_text = " ".join(str(row.get(key, "")) for key in row)
        family = infer_family(source_aa, target_aa, clinical_text)
        seed = ResidueSeed(
            residue_id=f"{source_aa}{position}",
            residue_position=position,
            source_aa=source_aa,
            topology_region=region_for_position(position),
            genotype_axis="population_variants",
            selection_features=(
                "population_frequency",
                "functional_assay_annotation",
                "clinical_association",
                "topology_region",
            ),
            score_components={
                "maf_global": _as_float(row.get("maf_global")),
                "functional_risk": 1.0,
                "exact_population_variant": 1.0,
            },
            source_artifacts=(population_variants.as_posix(),),
            seed_kind="exact_population_variant_required",
        )
        exact = _make_row(
            seed,
            family,
            target_aa,
            conservation,
            propagation_by_mutation,
            propagation_by_residue,
            chronology_by_residue,
            thermo_by_residue,
            nma_by_residue,
            observed_grid_lookup,
            output_root,
        )
        exact["variant_id"] = _stable_id(f"ALENI-PGX-{exact['mutation']}")
        exact["id"] = exact["variant_id"]
        exact["run_priority"] = "P0_PGX_SENTINEL" if exact["trajectory_execution_status"].startswith("OBSERVED") else exact["run_priority"]
        rows_by_id[str(exact["variant_id"])] = exact

    rows = sorted(rows_by_id.values(), key=lambda row: (-float(row["priority_score"]), str(row["variant_id"])))
    target_rows = _build_target_rows(rows)
    target_report = _build_target_report(target_rows, rows)
    run_plan = _build_run_plan(rows, output_root, phase3_manifest)
    validation_report = _validate_rows(rows, target_rows, genealogical_panel, population_variants, phase3_manifest)
    manifest = {
        "id": "aleniglipron_expanded_variant_run_manifest",
        "schema_version": "track_b.aleniglipron_expanded_variant_run_manifest.v1",
        "created_at": utc_now_iso(),
        "provenance_class": "L3_DERIVED",
        "source_artifacts": [
            _path_sha(path)
            for path in (
                topology_registry,
                genealogical_panel,
                population_variants,
                population_manifest,
                phase3_manifest,
                conservation_path,
                propagation_deltas,
                chronology_tensor,
                thermodynamic_continuity,
                nma_continuity,
                topology_path,
            )
        ],
        "evidence_paths": [
            path.as_posix()
            for path in (
                topology_registry,
                genealogical_panel,
                population_variants,
                population_manifest,
                phase3_manifest,
                conservation_path,
                propagation_deltas,
                chronology_tensor,
                thermodynamic_continuity,
                nma_continuity,
                topology_path,
            )
        ],
        "variant_count": len(rows),
        "baseline_genealogical_panel_variant_count": _panel_variant_count(genealogical_panel),
        "population_variant_count": len(population_rows),
        "phase3_high_risk_count": len(_phase3_rows(phase3_manifest)),
        "trajectory_queue_total_count": sum(
            1
            for row in rows
            if "QUEUE_TRUE_TRAJECTORY" in str(row["trajectory_execution_status"])
            or row["trajectory_execution_status"] == "QUEUED_TRUE_PERTURBED_TRAJECTORY"
        ),
        "true_trajectory_queue_count": sum(
            1 for row in rows if str(row["trajectory_execution_status"]).endswith("TRUE_PERTURBED_TRAJECTORY")
        ),
        "observed_grid_extension_count": sum(
            1 for row in rows if str(row["trajectory_execution_status"]).startswith("OBSERVED_GRID_AVAILABLE")
        ),
        "claim_boundary": "Rows queue true perturbed PRISM trajectories; they are not observed trajectory outputs until trajectory_output_dir contains engine artifacts.",
        "variants": rows,
        "run_plan": run_plan,
        "target_avoidance_report": target_report,
        "validation_report": validation_report,
    }
    return BuildResult(
        manifest=manifest,
        variant_rows=rows,
        target_rows=target_rows,
        target_report=target_report,
        run_plan=run_plan,
        validation_report=validation_report,
        runbook_markdown=_runbook(manifest, output_root),
    )


def _panel_variant_count(path: Path) -> int:
    if not path.exists():
        return 0
    payload = read_json(path)
    return int(payload.get("variant_count", len(payload.get("variants", []))))


def _build_target_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["topology_region"]), int(row["residue_position"]), str(row["source_amino_acid"]))
        grouped.setdefault(key, []).append(row)
    target_rows: list[dict[str, Any]] = []
    for (region, position, source_aa), members in sorted(grouped.items()):
        decisions = {str(row["target_decision"]) for row in members}
        max_target = max(float(row["targetability_score"]) for row in members)
        max_avoid = max(float(row["avoidance_score"]) for row in members)
        max_priority = max(float(row["priority_score"]) for row in members)
        if any(decision.startswith("AVOID") for decision in decisions):
            recommendation = "AVOID_OR_USE_AS_FALSIFICATION_CONTROL"
        elif any(decision == "TARGET_FOR_BIASED_CHRONOLOGICAL_CONTROL" for decision in decisions):
            recommendation = "TARGET"
        elif any(decision == "PRESERVE_DO_NOT_OCCLUDE" for decision in decisions):
            recommendation = "PRESERVE"
        else:
            recommendation = "SECONDARY_TARGET"
        target_rows.append(
            {
                "id": f"target-{region}-{source_aa}{position}",
                "residue_position": position,
                "residue_id": f"{source_aa}{position}",
                "topology_region": region,
                "variant_count": len(members),
                "max_targetability_score": max_target,
                "max_avoidance_score": max_avoid,
                "max_priority_score": max_priority,
                "recommendation": recommendation,
                "decision_set": sorted(decisions),
                "top_variant_ids": [str(row["variant_id"]) for row in members[:5]],
                "provenance_class": "L3_DERIVED",
                "created_at": utc_now_iso(),
                "schema_version": "track_b.aleniglipron_receptor_target_avoidance.v1",
            }
        )
    return target_rows


def _build_target_report(target_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_rec: dict[str, int] = {}
    for row in target_rows:
        recommendation = str(row["recommendation"])
        by_rec[recommendation] = by_rec.get(recommendation, 0) + 1
    top_targets = sorted(
        [row for row in target_rows if row["recommendation"] in {"TARGET", "SECONDARY_TARGET"}],
        key=lambda row: (str(row["recommendation"]) != "TARGET", -float(row["max_priority_score"])),
    )[:20]
    avoid = sorted(
        [row for row in target_rows if str(row["recommendation"]).startswith("AVOID")],
        key=lambda row: -float(row["max_avoidance_score"]),
    )[:20]
    return {
        "id": "aleniglipron_receptor_target_avoidance_report",
        "schema_version": "track_b.aleniglipron_receptor_target_avoidance_report.v1",
        "created_at": utc_now_iso(),
        "provenance_class": "L3_DERIVED",
        "target_count": len(target_rows),
        "variant_count": len(rows),
        "recommendation_counts": by_rec,
        "top_receptor_regions_to_target": [
            {
                "residue_id": row["residue_id"],
                "topology_region": row["topology_region"],
                "score": row["max_priority_score"],
                "recommendation": row["recommendation"],
            }
            for row in top_targets
        ],
        "receptor_regions_to_avoid_or_use_as_controls": [
            {
                "residue_id": row["residue_id"],
                "topology_region": row["topology_region"],
                "avoidance_score": row["max_avoidance_score"],
                "recommendation": row["recommendation"],
            }
            for row in avoid
        ],
        "claim_boundary": "Computational target/avoid triage only; no biological efficacy claim.",
    }


def _build_run_plan(rows: list[dict[str, Any]], output_root: Path, phase3_manifest: Path) -> dict[str, Any]:
    priority_order = [
        "P0_PGX_SENTINEL",
        "P0_OBSERVED_EXTENSION",
        "P0_AVOID_SENTINEL",
        "P0_TRUE_TRAJECTORY",
        "P1_TRUE_TRAJECTORY",
        "P2_TRUE_TRAJECTORY",
        "P3_CONTROL_TRAJECTORY",
        "BLOCKED",
    ]
    batches: list[dict[str, Any]] = []
    for priority in priority_order:
        selected = [row for row in rows if row["run_priority"] == priority]
        if not selected:
            continue
        batches.append(
            {
                "batch_id": f"aleniglipron-{priority.lower()}",
                "run_priority": priority,
                "variant_count": len(selected),
                "variant_ids": [str(row["variant_id"]) for row in selected[:256]],
                "topology_manifest_path": (output_root / "variant_topology_materialization_manifest.json").as_posix(),
                "trajectory_manifest_path": (output_root / f"{priority.lower()}_trajectory_manifest.json").as_posix(),
                "execution_status": "READY_AFTER_VARIANT_TOPOLOGIES_EXIST" if priority != "BLOCKED" else "BLOCKED_INPUT_MAPPING",
            }
        )
    return {
        "id": "aleniglipron_true_perturbed_trajectory_run_plan",
        "schema_version": "track_b.aleniglipron_true_trajectory_run_plan.v1",
        "created_at": utc_now_iso(),
        "provenance_class": "L3_DERIVED",
        "phase3_manifest": phase3_manifest.as_posix(),
        "replicas_required_for_full_observatory": 80,
        "streams_required_for_full_observatory": 1600,
        "trajectory_protocol": "5-phase CCNS hysteresis with shear, hysteresis, translation-pathway, and continuity extraction",
        "batch_count": len(batches),
        "batches": batches,
        "hard_gate": "Do not label queued rows L5_OBSERVED until PRISM trajectory artifacts exist under trajectory_output_dir.",
    }


def _validate_rows(
    rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    genealogical_panel: Path,
    population_variants: Path,
    phase3_manifest: Path,
) -> dict[str, Any]:
    sequence_only = [
        row["variant_id"]
        for row in rows
        if set(row.get("selection_features", [])) <= {"species_conservation", "population_frequency"}
    ]
    missing_axes = [
        row["variant_id"]
        for row in rows
        if not row.get("topology_region") or not row.get("perturbation_family") or not row.get("observability_channels")
    ]
    projected_claimed_observed = [
        row["variant_id"]
        for row in rows
        if row["trajectory_execution_status"] == "QUEUED_TRUE_PERTURBED_TRAJECTORY"
        and row["provenance_class"] == "L5_OBSERVED"
    ]
    verdict = "PASS" if not sequence_only and not missing_axes and not projected_claimed_observed else "FAIL"
    return {
        "id": "aleniglipron_expanded_variant_run_validation",
        "schema_version": "track_b.aleniglipron_expanded_variant_run_validation.v1",
        "created_at": utc_now_iso(),
        "provenance_class": "L3_DERIVED",
        "source_artifacts": [
            genealogical_panel.as_posix(),
            population_variants.as_posix(),
            phase3_manifest.as_posix(),
        ],
        "variant_count": len(rows),
        "target_row_count": len(target_rows),
        "sequence_only_variant_count": len(sequence_only),
        "missing_axis_count": len(missing_axes),
        "projected_claimed_observed_count": len(projected_claimed_observed),
        "run_priority_counts": _counts(row["run_priority"] for row in rows),
        "trajectory_status_counts": _counts(row["trajectory_execution_status"] for row in rows),
        "topology_region_counts": _counts(row["topology_region"] for row in rows),
        "perturbation_family_counts": _counts(row["perturbation_family"] for row in rows),
        "verdict": verdict,
        "failures": {
            "sequence_only": sequence_only[:20],
            "missing_axes": missing_axes[:20],
            "projected_claimed_observed": projected_claimed_observed[:20],
        },
    }


def _counts(values: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _runbook(manifest: dict[str, Any], output_root: Path) -> str:
    return "\n".join(
        [
            "# Aleniglipron Expanded Variant Runbook",
            "",
            "This runbook queues true perturbed PRISM trajectories. It does not claim",
            "that queued variants have observed trajectory data until the referenced",
            "`trajectory_output_dir` contains PRISM engine outputs.",
            "",
            f"- Variant count: {manifest['variant_count']}",
            f"- Output root: {output_root.as_posix()}",
            f"- Full observatory target: {manifest['run_plan']['replicas_required_for_full_observatory']} replicas / {manifest['run_plan']['streams_required_for_full_observatory']} streams",
            "",
            "## Execution Order",
            "",
            "1. Materialize each `variant_topology_path` from the validated WT holo topology.",
            "2. Run P0 batches first: PGx sentinels, observed-grid extensions, avoid sentinels, and primary true trajectories.",
            "3. Promote to P1/P2/P3 only after P0 trajectory extraction produces signal grid, shear, hysteresis, and pathway outputs.",
            "4. Never relabel `QUEUED_TRUE_PERTURBED_TRAJECTORY` rows as `L5_OBSERVED` without raw PRISM output artifacts.",
            "",
            "## Claim Boundary",
            "",
            "Computational target/avoid triage only. No biological efficacy claim is made.",
            "",
        ]
    )


def rows_to_parquet_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Return a parquet-friendly DataFrame for expanded variant rows."""

    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = dict(row)
        for key in ("observability_channels", "selection_features", "score_components", "source_artifacts", "evidence_paths"):
            flat[f"{key}_json"] = _json_dump(flat.pop(key))
        flat_rows.append(flat)
    return pl.DataFrame(flat_rows)


def target_rows_to_parquet_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Return a parquet-friendly DataFrame for target/avoid rows."""

    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = dict(row)
        for key in ("decision_set", "top_variant_ids"):
            flat[f"{key}_json"] = _json_dump(flat.pop(key))
        flat_rows.append(flat)
    return pl.DataFrame(flat_rows)


def write_build_result(
    result: BuildResult,
    *,
    output_json: Path,
    output_parquet: Path,
    target_matrix: Path,
    target_report: Path,
    run_plan: Path,
    validation_report: Path,
    runbook: Path,
) -> None:
    """Write all expanded manifest artifacts."""

    write_json(output_json, result.manifest)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    rows_to_parquet_frame(result.variant_rows).write_parquet(output_parquet)
    target_matrix.parent.mkdir(parents=True, exist_ok=True)
    target_rows_to_parquet_frame(result.target_rows).write_parquet(target_matrix)
    write_json(target_report, result.target_report)
    write_json(run_plan, result.run_plan)
    write_json(validation_report, result.validation_report)
    runbook.parent.mkdir(parents=True, exist_ok=True)
    runbook.write_text(result.runbook_markdown, encoding="utf-8")
    _write_execution_manifests(result, run_plan.parent)


def _write_execution_manifests(result: BuildResult, output_root: Path) -> None:
    """Write per-priority execution manifests referenced by the run plan."""

    topology_jobs = [
        {
            "variant_id": row["variant_id"],
            "mutation": row["mutation"],
            "source_amino_acid": row["source_amino_acid"],
            "target_amino_acid": row["target_amino_acid"],
            "residue_position": row["residue_position"],
            "base_topology": "04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json",
            "variant_topology_path": row["variant_topology_path"],
            "status": "PENDING_TOPOLOGY_MATERIALIZATION",
            "claim_boundary": "No true perturbed trajectory can run until this topology exists and passes prism-prep validation.",
        }
        for row in result.variant_rows
        if row["run_priority"] != "BLOCKED"
    ]
    write_json(
        output_root / "variant_topology_materialization_manifest.json",
        {
            "id": "aleniglipron_variant_topology_materialization_manifest",
            "schema_version": "track_b.aleniglipron_variant_topology_materialization.v1",
            "created_at": utc_now_iso(),
            "provenance_class": "L3_DERIVED",
            "job_count": len(topology_jobs),
            "jobs": topology_jobs,
        },
    )
    for batch in result.run_plan["batches"]:
        priority = str(batch["run_priority"])
        selected = [row for row in result.variant_rows if row["run_priority"] == priority]
        selected_jobs = [
            {
                "variant_id": row["variant_id"],
                "mutation": row["mutation"],
                "topology_region": row["topology_region"],
                "perturbation_family": row["perturbation_family"],
                "target_decision": row["target_decision"],
                "trajectory_execution_status": row["trajectory_execution_status"],
                "variant_topology_path": row["variant_topology_path"],
                "trajectory_output_dir": row["trajectory_output_dir"],
                "pilot_command": row["pilot_command"],
                "full_trajectory_command_template": row["full_trajectory_command_template"],
            }
            for row in selected
        ]
        write_json(
            Path(str(batch["trajectory_manifest_path"])),
            {
                "id": str(batch["batch_id"]),
                "schema_version": "track_b.aleniglipron_variant_trajectory_batch.v1",
                "created_at": utc_now_iso(),
                "provenance_class": "L3_DERIVED",
                "run_priority": priority,
                "variant_count": len(selected_jobs),
                "execution_status": batch["execution_status"],
                "claim_boundary": "Batch manifests queue PRISM work; they are not observed trajectory evidence.",
                "variants": selected_jobs,
            },
        )
