"""WT-3 Filter Pipeline — 6-stage cascade + Pareto ranking.

Input:  List[GeneratedMolecule] + SpikePharmacophore + PipelineConfig/FilterConfig
Output: List[FilteredCandidate] (ranked, top-N)

Pipeline:  Stage1 → Stage2 → Stage3 → Stage4 → Stage5 → Stage6 → Rank

Each stage takes a list, returns (passed, rejected), logs N_in/N_out.
Never modifies molecules — only annotates + filters.

CLI usage:
    python scripts/filters/filter_pipeline.py \\
        --molecules molecules_meta.json \\
        --pharmacophore pharmacophore.json \\
        --top-n 5 --output candidates.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.interfaces import (
    FilterConfig,
    FilteredCandidate,
    GeneratedMolecule,
    SpikePharmacophore,
)
from scripts.filters import (
    stage1_validity,
    stage2_druglikeness,
    stage3_pains,
    stage4_pharmacophore,
    stage5_novelty,
    stage6_diversity,
    ranking,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    molecules: List[GeneratedMolecule],
    pharmacophore: SpikePharmacophore,
    config: Optional[FilterConfig] = None,
    top_n: int = 5,
    min_pharmacophore_matches: int = 3,
    pharmacophore_distance_tolerance: float = 1.5,
    diversity_cutoff: float = 0.4,
    reference_fps=None,
) -> Tuple[List[FilteredCandidate], Dict[str, int]]:
    """Run the full 6-stage filter pipeline.

    Args:
        molecules: Input molecules from generation stage.
        pharmacophore: Reference spike pharmacophore.
        config: Filter thresholds (uses defaults if None).
        top_n: Number of final candidates to return.
        min_pharmacophore_matches: Min 3D feature matches (stage 4).
        pharmacophore_distance_tolerance: Distance tolerance in Angstrom.
        diversity_cutoff: Butina clustering distance cutoff.
        reference_fps: Pre-loaded reference fingerprints for novelty.

    Returns:
        (candidates, stats) — ranked FilteredCandidate list and per-stage stats.
    """
    if config is None:
        config = FilterConfig()

    stats: Dict[str, int] = {"input": len(molecules)}

    # Stage 1: Validity
    s1_passed, s1_rejected = stage1_validity.run(molecules)
    stats["stage1_passed"] = len(s1_passed)
    stats["stage1_rejected"] = len(s1_rejected)

    # Stage 2: Drug-likeness
    s2_passed, s2_rejected = stage2_druglikeness.run(s1_passed, config)
    stats["stage2_passed"] = len(s2_passed)
    stats["stage2_rejected"] = len(s2_rejected)

    # Stage 3: PAINS
    s3_passed, s3_rejected = stage3_pains.run(s2_passed, reject_pains=config.pains_reject)
    stats["stage3_passed"] = len(s3_passed)
    stats["stage3_rejected"] = len(s3_rejected)

    # Stage 4: Pharmacophore re-validation
    s4_passed, s4_rejected = stage4_pharmacophore.run(
        s3_passed,
        pharmacophore,
        min_matches=min_pharmacophore_matches,
        distance_tolerance=pharmacophore_distance_tolerance,
    )
    stats["stage4_passed"] = len(s4_passed)
    stats["stage4_rejected"] = len(s4_rejected)

    # Stage 5: Novelty
    s5_passed, s5_rejected = stage5_novelty.run(
        s4_passed,
        tanimoto_max=config.tanimoto_max,
        reference_fps=reference_fps,
    )
    stats["stage5_passed"] = len(s5_passed)
    stats["stage5_rejected"] = len(s5_rejected)

    # Stage 6: Diversity
    s6_passed, s6_rejected = stage6_diversity.run(
        s5_passed,
        diversity_cutoff=diversity_cutoff,
        top_n=top_n,
    )
    stats["stage6_passed"] = len(s6_passed)
    stats["stage6_rejected"] = len(s6_rejected)

    # Ranking
    candidates = ranking.rank_candidates(s6_passed, top_n=top_n)
    stats["final_output"] = len(candidates)

    logger.info("Pipeline complete: %d → %d candidates", len(molecules), len(candidates))
    for stage_name, count in stats.items():
        logger.info("  %s: %d", stage_name, count)

    return candidates, stats


def _load_molecules(path: str) -> List[GeneratedMolecule]:
    """Load molecules from JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [GeneratedMolecule.from_dict(d) for d in data]
    if isinstance(data, dict) and "molecules" in data:
        return [GeneratedMolecule.from_dict(d) for d in data["molecules"]]
    raise ValueError(f"Unexpected JSON structure in {path}")


def _load_pharmacophore(path: str) -> SpikePharmacophore:
    """Load pharmacophore from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return SpikePharmacophore.from_dict(data)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="WT-3: Multi-stage filtering + ranking pipeline"
    )
    parser.add_argument(
        "--molecules", required=True,
        help="Path to molecules JSON (list of GeneratedMolecule dicts)",
    )
    parser.add_argument(
        "--pharmacophore", required=True,
        help="Path to SpikePharmacophore JSON",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of top candidates (default: 5)")
    parser.add_argument("--output", required=True, help="Output path for candidates JSON")
    parser.add_argument(
        "--qed-min", type=float, default=None,
        help="Override QED minimum threshold",
    )
    parser.add_argument(
        "--sa-max", type=float, default=None,
        help="Override SA score maximum threshold",
    )
    parser.add_argument(
        "--tanimoto-max", type=float, default=None,
        help="Override Tanimoto novelty cutoff",
    )
    parser.add_argument(
        "--min-pharm-matches", type=int, default=3,
        help="Minimum pharmacophore feature matches (default: 3)",
    )
    parser.add_argument(
        "--distance-tolerance", type=float, default=1.5,
        help="Pharmacophore distance tolerance in Angstrom (default: 1.5)",
    )
    parser.add_argument(
        "--diversity-cutoff", type=float, default=0.4,
        help="Butina clustering distance cutoff (default: 0.4)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load inputs
    molecules = _load_molecules(args.molecules)
    pharmacophore = _load_pharmacophore(args.pharmacophore)

    # Build config with overrides
    config = FilterConfig()
    if args.qed_min is not None:
        config.qed_min = args.qed_min
    if args.sa_max is not None:
        config.sa_max = args.sa_max
    if args.tanimoto_max is not None:
        config.tanimoto_max = args.tanimoto_max

    # Run pipeline
    candidates, stats = run_pipeline(
        molecules=molecules,
        pharmacophore=pharmacophore,
        config=config,
        top_n=args.top_n,
        min_pharmacophore_matches=args.min_pharm_matches,
        pharmacophore_distance_tolerance=args.distance_tolerance,
        diversity_cutoff=args.diversity_cutoff,
    )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "candidates": [c.to_dict() for c in candidates],
        "stats": stats,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Wrote %d candidates to %s", len(candidates), output_path)


if __name__ == "__main__":
    main()
