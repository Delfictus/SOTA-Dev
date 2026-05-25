#!/usr/bin/env python3
"""Project ligand interference onto temporal/fatigue fibers with tri-state labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import write_provenance_parquet


CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
TRACK0_DIR = CAMPAIGN_DIR / "track_0_manual_emulation"
DEFAULT_OUTPUT = N80_DIR / "tri_state_ligand_fiber_graph.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment-attribution", type=Path, default=TRACK0_DIR / "fragment_interference_attribution.parquet")
    parser.add_argument("--edge-interference", type=Path, default=TRACK0_DIR / "layer1_whole_molecule/per_edge_interference.parquet")
    parser.add_argument("--phase-edge-validation", type=Path, default=N80_DIR / "phase_manifold_edge_validation.parquet")
    parser.add_argument("--break-clusters", type=Path, default=N80_DIR / "probabilistic_break_clusters.parquet")
    parser.add_argument("--occupancy-fatigue", type=Path, default=N80_DIR / "occupancy_fatigue_risk.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def endpoint_cluster(prefix: str, clusters: pl.LazyFrame, residue_col: str) -> pl.LazyFrame:
    return (
        clusters.group_by(["condition_id", "primary_residue_idx"])
        .agg(
            [
                pl.col("cluster_id").first().alias(f"{prefix}_cluster_id"),
                pl.col("cluster_probability").max().alias(f"{prefix}_cluster_probability"),
                pl.col("cluster_confidence").max().alias(f"{prefix}_cluster_confidence"),
                pl.col("temporal_overlap_entropy").mean().alias(f"{prefix}_temporal_overlap_entropy"),
            ]
        )
        .rename({"primary_residue_idx": residue_col})
    )


def endpoint_fatigue(prefix: str, fatigue: pl.LazyFrame, residue_col: str) -> pl.LazyFrame:
    return (
        fatigue.group_by(["condition_id", "primary_residue_idx"])
        .agg(pl.col("occupancy_fatigue_index").max().alias(f"{prefix}_occupancy_fatigue_index"))
        .rename({"primary_residue_idx": residue_col})
    )


def fiber_frame(
    fragment_attribution: Path,
    edge_interference: Path,
    phase_edge_validation: Path,
    break_clusters: Path,
    occupancy_fatigue: Path,
) -> pl.LazyFrame:
    phase = pl.scan_parquet(phase_edge_validation).select(
        [
            "edge_id",
            "condition_id",
            "edge_label",
            "edge_class",
            pl.col("edge_from_residue").cast(pl.Int64),
            pl.col("edge_to_residue").cast(pl.Int64),
            "edge_coherence_score",
            "validation_status",
        ]
    )
    interference = pl.scan_parquet(edge_interference).select(
        [
            "edge_id",
            "expected_pi_clash",
            "expected_pi_complement",
            "te_multiplier",
            "projected_edge_score",
        ]
    )
    fragments = pl.scan_parquet(fragment_attribution).select(
        [
            "edge_id",
            "dominant_fragment",
            "dominant_fraction",
            "inter_fragment_coupling",
            "whole_molecule_clash",
            "whole_molecule_complement",
        ]
    )
    clusters = pl.scan_parquet(break_clusters).select(
        [
            "condition_id",
            "primary_residue_idx",
            "cluster_id",
            "cluster_probability",
            "cluster_confidence",
            "temporal_overlap_entropy",
        ]
    )
    fatigue = pl.scan_parquet(occupancy_fatigue).select(
        ["condition_id", "primary_residue_idx", "occupancy_fatigue_index"]
    )
    joined = (
        phase.join(interference, on="edge_id", how="left")
        .join(fragments, on="edge_id", how="left")
        .join(endpoint_cluster("from", clusters, "edge_from_residue"), on=["condition_id", "edge_from_residue"], how="left")
        .join(endpoint_cluster("to", clusters, "edge_to_residue"), on=["condition_id", "edge_to_residue"], how="left")
        .join(endpoint_fatigue("from", fatigue, "edge_from_residue"), on=["condition_id", "edge_from_residue"], how="left")
        .join(endpoint_fatigue("to", fatigue, "edge_to_residue"), on=["condition_id", "edge_to_residue"], how="left")
    )
    interference_load = pl.max_horizontal(
        [
            pl.col("expected_pi_clash").fill_null(0.0).abs(),
            pl.col("expected_pi_complement").fill_null(0.0).abs(),
            pl.col("whole_molecule_clash").fill_null(0.0).abs(),
            pl.col("whole_molecule_complement").fill_null(0.0).abs(),
        ]
    )
    return (
        joined.with_columns(
            [
                interference_load.alias("ligand_interference_load"),
                pl.max_horizontal(
                    [
                        pl.col("from_occupancy_fatigue_index").fill_null(0.0),
                        pl.col("to_occupancy_fatigue_index").fill_null(0.0),
                    ]
                ).alias("edge_occupancy_fatigue_index"),
                pl.when(pl.col("from_cluster_confidence").fill_null(0.0) >= pl.col("to_cluster_confidence").fill_null(0.0))
                .then(pl.col("from_cluster_id"))
                .otherwise(pl.col("to_cluster_id"))
                .alias("dominant_temporal_cluster"),
            ]
        )
        .with_columns(
            pl.when((pl.col("validation_status") == "validated_constitutive") & (pl.col("ligand_interference_load") < 0.1))
            .then(pl.lit("constitutive_fiber"))
            .when((pl.col("validation_status") == "validated_constitutive") & (pl.col("te_multiplier").fill_null(1.0) >= 2.0))
            .then(pl.lit("ligand_amplified_fiber"))
            .when((pl.col("edge_coherence_score") < 0.45) & (pl.col("ligand_interference_load") >= 0.1))
            .then(pl.lit("induced_fiber"))
            .otherwise(pl.lit("weak_or_unresolved_fiber"))
            .alias("fiber_type")
        )
        .select(
            [
                "edge_id",
                "condition_id",
                "edge_label",
                "edge_class",
                "edge_coherence_score",
                "validation_status",
                "expected_pi_clash",
                "expected_pi_complement",
                "te_multiplier",
                "projected_edge_score",
                "dominant_fragment",
                "dominant_fraction",
                "inter_fragment_coupling",
                "ligand_interference_load",
                "dominant_temporal_cluster",
                "from_cluster_confidence",
                "to_cluster_confidence",
                "from_temporal_overlap_entropy",
                "to_temporal_overlap_entropy",
                "edge_occupancy_fatigue_index",
                "fiber_type",
            ]
        )
        .sort(["fiber_type", "projected_edge_score"], descending=[False, True])
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    frame = fiber_frame(
        Path(args.fragment_attribution),
        Path(args.edge_interference),
        Path(args.phase_edge_validation),
        Path(args.break_clusters),
        Path(args.occupancy_fatigue),
    )
    rows = frame.select(pl.len().alias("n")).collect(engine="streaming").item()
    write_provenance_parquet(
        frame,
        output,
        producer_script=Path(__file__),
        source_parquets=[
            Path(args.fragment_attribution),
            Path(args.edge_interference),
            Path(args.phase_edge_validation),
            Path(args.break_clusters),
            Path(args.occupancy_fatigue),
        ],
        schema_version="tri_state_ligand_fiber_graph.v1",
        pipeline_stage="tri_state_ligand_fiber_graph",
        partition_keys=["fiber_type", "condition_id"],
        ledger_parameters={
            "constitutive_fiber": "validated constitutive phase edge with near-zero ligand interference",
            "ligand_amplified_fiber": "validated constitutive phase edge with te_multiplier >= 2",
            "induced_fiber": "low-coherence edge with nonzero ligand interference load",
        },
        ledger_output_value={"rows": int(rows), "output_path": output.as_posix()},
        repo_root=REPO_ROOT,
    )
    sys.stdout.write(f"wrote {output} rows={rows}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
