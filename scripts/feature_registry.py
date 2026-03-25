"""PRISM4D — Canonical Feature Registry.

Every validated feature MUST be registered here and executed in the pipeline.
If a feature is implemented, tested, and committed — it is NOT optional.

The pipeline calls mark() after each step and assert_all() at the end.
A missing mark is a hard failure.
"""
from __future__ import annotations

from typing import Dict, List


# -- Canonical feature list -------------------------------------------------
# If it's in this dict, it MUST execute.  No exceptions.
CANONICAL_FEATURES: Dict[str, str] = {
    # Data loading
    "binding_sites_loaded": "Load binding_sites.json from Rust engine",
    "spike_events_loaded": "Load spike_events/ from Rust engine",
    "kcc_loaded": "Load kcc_visualization.json from Rust engine",
    "trajectory_loaded": "Load ensemble_trajectory.pdb (or mark absent)",
    # Gating stack
    "gating_therm": "Therm gate evaluated for all sites",
    "gating_coherence": "Coherence gate evaluated for all sites (soft)",
    "gating_localization": "Localization gate evaluated for all sites",
    "gating_contact_reorg": "Contact Reorg gate evaluated for all sites",
    "gating_response_selectivity": "Response Selectivity gate evaluated for all sites",
    # Design layers (on passed sites only)
    "anchor_points": "AnchorPoint computed for all passed sites",
    "growth_vectors": "GrowthVector computed for all passed sites",
    "pocket_profiles": "PocketProfile computed for all passed sites",
    "site_ranking": "Lexicographic SiteRanker executed",
    "design_briefs": "DesignBrief generated for all passed sites",
}


class PipelineRegistry:
    """Tracks which features have been executed in the current pipeline run.

    Usage:
        reg = PipelineRegistry()
        reg.mark("binding_sites_loaded")
        ...
        reg.assert_all()  # fails hard if anything is missing
    """

    def __init__(self) -> None:
        self._executed: Dict[str, bool] = {
            k: False for k in CANONICAL_FEATURES
        }

    def mark(self, feature: str) -> None:
        """Mark a feature as executed."""
        if feature not in self._executed:
            raise KeyError(
                f"Unknown feature '{feature}'. "
                f"Valid features: {list(CANONICAL_FEATURES.keys())}"
            )
        self._executed[feature] = True

    def is_marked(self, feature: str) -> bool:
        return self._executed.get(feature, False)

    def missing(self) -> List[str]:
        """Return list of features that were NOT executed."""
        return [k for k, v in self._executed.items() if not v]

    def assert_all(self) -> None:
        """Fail hard if any canonical feature was not executed."""
        gaps = self.missing()
        if gaps:
            report = "\n".join(
                f"  ❌ {f}: {CANONICAL_FEATURES[f]}" for f in gaps
            )
            raise RuntimeError(
                f"PIPELINE INTEGRITY VIOLATION — {len(gaps)} features not executed:\n"
                f"{report}\n\n"
                f"Every validated feature MUST run. No exceptions."
            )

    def summary(self) -> str:
        """Return human-readable execution summary."""
        lines = []
        for feature, desc in CANONICAL_FEATURES.items():
            status = "✔" if self._executed[feature] else "❌"
            lines.append(f"  {status} {feature}: {desc}")
        n_done = sum(1 for v in self._executed.values() if v)
        n_total = len(self._executed)
        lines.insert(0, f"Pipeline Registry: {n_done}/{n_total} features executed")
        return "\n".join(lines)
