#!/usr/bin/env python3
"""Build an expanded Aleniglipron variant run manifest for true trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from prism_dstw.calibration.aleniglipron_variant_manifest import (
    build_expanded_variant_run,
    write_build_result,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_B = CAMPAIGN / "track_b_chronological"
N80 = CAMPAIGN / "integrated_spike_events/n80_full_scale"
TRACK_A = CAMPAIGN / "track_a_generative"
POP_SOURCE = TRACK_A / "population_pgx/source"
OUTPUT_ROOT = TRACK_B / "expanded_variant_run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-registry", type=Path, default=TRACK_B / "topology_region_registry.json")
    parser.add_argument("--genealogical-panel", type=Path, default=TRACK_B / "genealogical_variant_panel.json")
    parser.add_argument("--population-variants", type=Path, default=POP_SOURCE / "gnomAD_GLP1R_missense_variants.csv")
    parser.add_argument("--population-manifest", type=Path, default=TRACK_A / "population_pgx/variant_perturbation_manifest.json")
    parser.add_argument("--phase3-manifest", type=Path, default=CAMPAIGN / "Phase3_PGx_Exclusion_Manifest.json")
    parser.add_argument("--conservation", type=Path, default=POP_SOURCE / "GLP1R_cross_species_conservation.csv")
    parser.add_argument("--propagation-deltas", type=Path, default=N80 / "variant_propagation_deltas.parquet")
    parser.add_argument("--chronology-tensor", type=Path, default=N80 / "transition_chronology_tensor.parquet")
    parser.add_argument("--thermodynamic-continuity", type=Path, default=TRACK_B / "thermodynamic_continuity_map.parquet")
    parser.add_argument("--nma-continuity", type=Path, default=TRACK_B / "nma_continuity_map.parquet")
    parser.add_argument("--topology", type=Path, default=REPO_ROOT / "04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--max-target-choices-per-family",
        type=int,
        default=3,
        help="Number of non-self amino-acid targets emitted per perturbation family.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    result = build_expanded_variant_run(
        topology_registry=Path(args.topology_registry),
        genealogical_panel=Path(args.genealogical_panel),
        population_variants=Path(args.population_variants),
        population_manifest=Path(args.population_manifest),
        phase3_manifest=Path(args.phase3_manifest),
        conservation_path=Path(args.conservation),
        propagation_deltas=Path(args.propagation_deltas),
        chronology_tensor=Path(args.chronology_tensor),
        thermodynamic_continuity=Path(args.thermodynamic_continuity),
        nma_continuity=Path(args.nma_continuity),
        topology_path=Path(args.topology),
        output_root=output_root,
        max_target_choices_per_family=int(args.max_target_choices_per_family),
    )
    write_build_result(
        result,
        output_json=output_root / "aleniglipron_expanded_variant_run_manifest.json",
        output_parquet=output_root / "aleniglipron_expanded_variant_run_manifest.parquet",
        target_matrix=output_root / "aleniglipron_receptor_target_avoidance_matrix.parquet",
        target_report=output_root / "aleniglipron_receptor_target_avoidance_report.json",
        run_plan=output_root / "true_perturbed_trajectory_run_plan.json",
        validation_report=output_root / "aleniglipron_expanded_variant_run_validation.json",
        runbook=output_root / "TRUE_PERTURBED_TRAJECTORY_RUNBOOK.md",
    )
    print(
        "aleniglipron_expanded_variant_run_manifest "
        f"variants={result.manifest['variant_count']} "
        f"baseline={result.manifest['baseline_genealogical_panel_variant_count']} "
        f"queued={result.manifest['true_trajectory_queue_count']} "
        f"observed_grid_extensions={result.manifest['observed_grid_extension_count']} "
        f"targets={result.target_report['target_count']} "
        f"validation={result.validation_report['verdict']} "
        f"output_root={output_root}"
    )
    return 0 if result.validation_report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
