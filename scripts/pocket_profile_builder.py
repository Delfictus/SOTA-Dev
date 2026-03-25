#!/usr/bin/env python3
"""PRISM4D — PocketProfile Builder.

Computes a descriptive chemistry and geometry profile for each binding site
from lining residue composition, pharmacophore features, and optional water
map data.  Purely observational — no predictions.

Usage (standalone):
    python3 scripts/pocket_profile_builder.py \\
        --binding-sites /path/to/binding_sites.json \\
        [--out /path/to/pocket_profiles.json]

Programmatic:
    from scripts.pocket_profile_builder import PocketProfileBuilder
    builder = PocketProfileBuilder()
    profiles = builder.compute_all(sites)
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import Any, Dict, List, Optional

from scripts.interfaces.pocket_profile import PocketProfile


# ---------------------------------------------------------------------------
# Residue classification tables
# ---------------------------------------------------------------------------
AROMATIC = {"TRP", "TYR", "PHE", "HIS"}
POLAR = {"ASP", "GLU", "ASN", "GLN", "HIS", "SER", "THR", "TYR", "ARG", "LYS"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
CHARGED_POS = {"ARG", "LYS"}
CHARGED_NEG = {"ASP", "GLU"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lining_composition(
    lining: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute residue class fractions from lining_residues."""
    if not lining:
        return {
            "aromatic": 0.0,
            "polar": 0.0,
            "hydrophobic": 0.0,
            "charged_pos": 0.0,
            "charged_neg": 0.0,
        }

    n = len(lining)
    counts = Counter(r.get("resname", "UNK") for r in lining)

    return {
        "aromatic": sum(counts.get(r, 0) for r in AROMATIC) / n,
        "polar": sum(counts.get(r, 0) for r in POLAR) / n,
        "hydrophobic": sum(counts.get(r, 0) for r in HYDROPHOBIC) / n,
        "charged_pos": sum(counts.get(r, 0) for r in CHARGED_POS) / n,
        "charged_neg": sum(counts.get(r, 0) for r in CHARGED_NEG) / n,
    }


def _feature_coupling(features: List[Dict[str, Any]]) -> float:
    """Spatial clustering entropy of pharmacophore features.

    Lower entropy = more clustered (features are spatially correlated).
    Higher entropy = dispersed (features are spread out).
    Uses feature-type distribution as a proxy for spatial coupling.
    """
    if not features:
        return 1.0

    types = [f.get("feature_type", f.get("type", "UNK")) for f in features]
    counts = Counter(types)
    n = len(types)
    if n == 0:
        return 1.0

    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalize by max entropy (uniform distribution over observed types)
    max_entropy = math.log2(max(len(counts), 1))
    if max_entropy > 0:
        return round(entropy / max_entropy, 4)
    return 0.0


def _classify_mw(volume: float, n_features: int) -> str:
    """Estimate MW class from pocket volume.

    Rough mapping: pocket volume → ligand MW range.
    fragment (<300 Da) ~= volume < 300 A^3
    lead (300-500 Da)  ~= volume 300-800 A^3
    beyond_ro5 (>500)  ~= volume > 800 A^3
    """
    if volume < 300:
        return "fragment"
    elif volume < 800:
        return "lead"
    else:
        return "beyond_ro5"


def _classify_polarity(comp: Dict[str, float]) -> str:
    """Classify pocket polarity from residue composition."""
    hydro = comp["hydrophobic"]
    polar = comp["polar"]

    if hydro > 0.5:
        return "hydrophobic"
    elif polar > 0.5:
        return "polar"
    else:
        return "mixed"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class PocketProfileBuilder:
    """Computes PocketProfile for binding sites."""

    def compute(
        self,
        site: Dict[str, Any],
        pharmacophore_features: Optional[List[Dict[str, Any]]] = None,
        water_displacement_energy: float = 0.0,
    ) -> PocketProfile:
        """Compute pocket profile for one site."""
        site_id = site.get("id", -1)
        lining = site.get("lining_residues", [])
        volume = site.get("volume", 0.0)
        enclosure = site.get("burial_score", 0.0)

        comp = _lining_composition(lining)

        n_lining = len(lining)
        charge_bias = (
            (comp["charged_pos"] - comp["charged_neg"])
            if n_lining > 0
            else 0.0
        )

        features = pharmacophore_features or []
        coupling = _feature_coupling(features)
        mw_class = _classify_mw(volume, len(features))
        polarity = _classify_polarity(comp)

        return PocketProfile(
            site_id=site_id,
            aromatic_fraction=round(comp["aromatic"], 4),
            polar_fraction=round(comp["polar"], 4),
            hydrophobic_fraction=round(comp["hydrophobic"], 4),
            charged_positive_fraction=round(comp["charged_pos"], 4),
            charged_negative_fraction=round(comp["charged_neg"], 4),
            charge_bias=round(charge_bias, 4),
            volume=round(volume, 1),
            enclosure=round(enclosure, 4),
            n_lining_residues=n_lining,
            feature_coupling=coupling,
            mw_class=mw_class,
            polarity_class=polarity,
            water_displacement_energy=round(water_displacement_energy, 3),
        )

    def compute_all(
        self,
        sites: List[Dict[str, Any]],
        water_energies: Optional[Dict[int, float]] = None,
    ) -> Dict[int, PocketProfile]:
        """Compute pocket profiles for all sites."""
        we = water_energies or {}
        results: Dict[int, PocketProfile] = {}
        for i, site in enumerate(sites):
            site_id = site.get("id", i)
            wd = we.get(site_id, 0.0)
            results[site_id] = self.compute(site, water_displacement_energy=wd)
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D PocketProfile Builder"
    )
    parser.add_argument(
        "--binding-sites", required=True, help="Path to binding_sites.json"
    )
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    with open(args.binding_sites) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get("sites", [])

    builder = PocketProfileBuilder()
    results = builder.compute_all(sites)

    output = {str(sid): r.to_dict() for sid, r in sorted(results.items())}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {len(output)} profiles to {args.out}")
    else:
        for sid, pp in sorted(results.items()):
            print(
                f"Site {sid}: {pp.polarity_class} {pp.mw_class} "
                f"V={pp.volume:.0f}A^3 "
                f"aro={pp.aromatic_fraction:.2f} "
                f"pol={pp.polar_fraction:.2f} "
                f"hyd={pp.hydrophobic_fraction:.2f} "
                f"Q={pp.charge_bias:+.2f}"
            )


if __name__ == "__main__":
    main()
