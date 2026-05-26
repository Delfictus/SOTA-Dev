#!/usr/bin/env python3
"""Generate a non-sequence-only genealogical variant panel for Track B."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import polars as pl

from prism_dstw.calibration.track_b_artifacts import read_json, write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso
from prism_dstw.calibration.translational_calibration_manifold import (
    PERTURBATION_SPACE,
    default_observability_for_region,
)

FAMILY_MUTATIONS: dict[str, tuple[str, str]] = {
    "SEVERING_PROBE": ("A", "G"),
    "RIGIDIFYING_PROBE": ("P", "W"),
    "FLEXIBILIZING_PROBE": ("G", "S"),
    "CHARGE_INVERSION_PROBE": ("D", "K"),
    "HYDRATION_WIRE_PROBE": ("S", "T"),
    "CONSERVATIVE_CONTROL": ("A", "V"),
    "VOLUME_EXPANSION_PROBE": ("A", "F"),
    "AROMATIC_STACKING_PROBE": ("F", "Y"),
}

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
    "RES": "X",
}

MAX_GLP1R_RESIDUE = 500


def _resolve_conservation_path(path: Path) -> Path:
    if path.exists():
        return path
    name = path.name
    campaign = Path("campaigns/glp1r_aleniglipron")
    candidates = [
        campaign / "track_a_generative" / "population_pgx" / "source" / name,
        *campaign.glob(f"**/{name}"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cross-species conservation artifact not found: {path}")


def _load_conservation(path: Path) -> tuple[dict[int, dict[str, Any]], Path]:
    resolved = _resolve_conservation_path(path)
    rows = pl.read_csv(resolved).to_dicts()
    return (
        {int(row["residue_position"]): row for row in rows if row.get("residue_position") is not None},
        resolved,
    )


def _selection_features(region: str, residue: dict[str, Any], conservation: dict[str, Any] | None) -> list[str]:
    features = [
        "propagation_adjacency",
        "topology_state_sensitivity",
        "phase_coherence",
    ]
    scores = residue.get("score_components", {})
    if isinstance(scores, dict):
        if float(scores.get("thermal_irreversibility", 0.0)) > 0.0:
            features.append("manifold_bifurcation_probability")
        if float(scores.get("mean_abs_load", 0.0)) > 0.0:
            features.append("shear_fracture_risk")
        if float(scores.get("translation_pathway", 0.0)) > 0.0:
            features.append("signaling_divergence_risk")
    if region == "HYDRATION_CORRIDOR":
        features.append("solvent_continuity_proxy")
    if region == "INTRACELLULAR_LOCK_BASIN":
        features.append("lock_basin_persistence")
    if conservation is not None:
        features.append("evolutionary_conservation")
    return sorted(set(features))


def _parse_residue_identity(residue_id: str) -> tuple[str, int]:
    match = re.match(r"^([A-Z]{3})(\d+)", residue_id)
    if match is None:
        digits = re.findall(r"\d+", residue_id)
        plausible_digits = [int(value) for value in digits if 0 < int(value) <= MAX_GLP1R_RESIDUE]
        return "X", plausible_digits[-1] if plausible_digits else 0
    position = int(match.group(2))
    if not (0 < position <= MAX_GLP1R_RESIDUE):
        raise ValueError(f"implausible residue identity rejected: {residue_id}")
    return AA3_TO_1.get(match.group(1), "X"), position


def _fallback_conservation(
    residue_position: int, conservation: dict[int, dict[str, Any]]
) -> tuple[dict[str, Any] | None, str]:
    if residue_position in conservation:
        return conservation[residue_position], "exact"
    if not conservation:
        return None, "none"
    nearest_position = min(conservation, key=lambda pos: abs(pos - residue_position))
    if abs(nearest_position - residue_position) <= 8:
        return conservation[nearest_position], "nearest_within_8"
    return None, "none"


def _target_aa(family: str, source_aa: str) -> str:
    if family == "CHARGE_INVERSION_PROBE":
        candidates = ["D", "E"] if source_aa in {"K", "R", "H"} else ["K", "R"]
    else:
        candidates_by_family = {
            "SEVERING_PROBE": ["G", "P", "A"],
            "RIGIDIFYING_PROBE": ["W", "P", "F"],
            "FLEXIBILIZING_PROBE": ["S", "G", "A"],
            "HYDRATION_WIRE_PROBE": ["T", "S", "N"],
            "CONSERVATIVE_CONTROL": ["V", "I", "L", "A"],
            "VOLUME_EXPANSION_PROBE": ["F", "W", "Y"],
            "AROMATIC_STACKING_PROBE": ["Y", "F", "W"],
        }
        candidates = candidates_by_family[family]
    for candidate in candidates:
        if candidate != source_aa:
            return candidate
    raise ValueError(f"no non-self target amino acid for {family} from {source_aa}")


def build_panel(region_registry: Path, conservation_path: Path) -> dict[str, Any]:
    registry = read_json(region_registry)
    conservation, resolved_conservation_path = _load_conservation(conservation_path)
    variants: list[dict[str, Any]] = []
    region_order = list(registry["regions"].keys())
    for region_index, region in enumerate(region_order):
        region_payload = registry["regions"][region]
        residues = [
            residue
            for residue in list(region_payload.get("residues", []))
            if _parse_residue_identity(str(residue.get("residue_id", "")))[0] != "X"
        ]
        if not residues:
            continue
        required_families = list(PERTURBATION_SPACE)
        for family_index, family in enumerate(required_families):
            for replicate in range(2):
                residue = residues[(family_index + replicate) % len(residues)]
                residue_id = str(residue["residue_id"])
                source_aa, residue_position = _parse_residue_identity(residue_id)
                conservation_row, conservation_match_type = _fallback_conservation(residue_position, conservation)
                from_aa = source_aa
                if from_aa == "X" and conservation_row is not None and conservation_match_type == "exact":
                    from_aa = str(conservation_row.get("human_aa", source_aa))
                to_aa = _target_aa(family, from_aa)
                if from_aa == to_aa:
                    raise ValueError(f"no-op variant rejected: {residue_id} {family} {from_aa}->{to_aa}")
                features = _selection_features(region, residue, conservation_row)
                if features == ["evolutionary_conservation"]:
                    raise ValueError(f"sequence-only variant rejected: {residue_id}")
                genotype_axis = (
                    "species_divergent_residues"
                    if conservation_row and float(conservation_row.get("conservation_score", 1.0)) < 1.0
                    else "conserved_but_dynamically_sensitive_residues"
                )
                variants.append(
                    {
                        "id": f"TB-{region}-{family}-{replicate + 1}-{residue_id}",
                        "variant_id": f"TB-{region}-{family}-{replicate + 1}-{residue_id}",
                        "residue_id": residue_id,
                        "residue_position": residue_position,
                        "topology_region": region,
                        "perturbation_family": family,
                        "perturbation_type": f"{from_aa}{residue_position}{to_aa}",
                        "source_amino_acid": from_aa,
                        "target_amino_acid": to_aa,
                        "projected_mutation": f"{from_aa}{residue_position}{to_aa}",
                        "genotype_axis": genotype_axis,
                        "observability_channels": list(default_observability_for_region(region)),
                        "selection_features": features,
                        "selection_feature_count": len(features),
                        "conservation_used": conservation_row is not None,
                        "conservation_match_type": conservation_match_type,
                        "region_priority": region_index,
                        "provenance_class": "L3_DERIVED",
                        "source_artifacts": [str(region_registry), str(resolved_conservation_path)],
                        "evidence_paths": [str(region_registry), str(resolved_conservation_path)],
                        "created_at": utc_now_iso(),
                        "schema_version": "track_b.genealogical_variant_panel.v1",
                    }
                )
    return {
        "id": "track_b_genealogical_variant_panel",
        "schema_version": "track_b.genealogical_variant_panel.v1",
        "created_at": utc_now_iso(),
        "provenance_class": "L3_DERIVED",
        "source_artifacts": [str(region_registry), str(resolved_conservation_path)],
        "evidence_paths": [str(region_registry), str(resolved_conservation_path)],
        "variant_count": len(variants),
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-registry", type=Path, required=True)
    parser.add_argument("--conservation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel = build_panel(args.region_registry, args.conservation)
    write_json(args.output, panel)
    regions = sorted({str(v["topology_region"]) for v in panel["variants"]})
    families = sorted({str(v["perturbation_family"]) for v in panel["variants"]})
    print(
        "genealogical_variant_panel "
        f"variants={panel['variant_count']} regions={len(regions)} families={len(families)} output={args.output}"
    )


if __name__ == "__main__":
    main()
