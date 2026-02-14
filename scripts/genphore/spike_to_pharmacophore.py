"""Convert PRISM spike event JSON to SpikePharmacophore.

Algorithm:
    1. Parse spike events JSON → group by spike type (BNZ/TYR/CATION/ANION/…)
    2. Per type: compute intensity-weighted centroid within pocket
    3. Map spike_type → pharmacophore feature_type via SPIKE_TYPE_TO_FEATURE
    4. Apply dual-feature rule: TYR→AR+HBD/HBA based on water_density
    5. Generate exclusion spheres from lining residue positions
    6. Validate: ≥2 features required, coordinates in protein reference frame
    7. Return SpikePharmacophore ready for .to_phoregen_json() / .to_pgmg_posp()
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.interfaces import (
    ExclusionSphere,
    PharmacophoreFeature,
    SpikePharmacophore,
    SPIKE_TYPE_TO_FEATURE,
)

logger = logging.getLogger(__name__)

# Water-density threshold above which TYR spikes also produce an HBA feature.
_WATER_DENSITY_HBA_THRESHOLD = 0.015

# Minimum spike intensity to include in centroid computation.
_MIN_SPIKE_INTENSITY = 0.05

# Default exclusion sphere radius for lining-residue heavy atoms (Angstrom).
_EXCLUSION_RADIUS = 1.7


def _compute_weighted_centroid(
    spikes: List[Dict[str, Any]],
) -> Tuple[float, float, float, float, float, float]:
    """Intensity-weighted centroid of a list of spike events.

    Returns:
        (cx, cy, cz, mean_intensity, mean_wavelength, mean_water_density)
    """
    wx = wy = wz = w_total = 0.0
    wavelength_sum = water_density_sum = 0.0
    n = 0
    for s in spikes:
        intensity = s["intensity"]
        if intensity < _MIN_SPIKE_INTENSITY:
            continue
        wx += s["x"] * intensity
        wy += s["y"] * intensity
        wz += s["z"] * intensity
        w_total += intensity
        wavelength_sum += s["wavelength_nm"]
        water_density_sum += s["water_density"]
        n += 1

    if w_total == 0.0 or n == 0:
        raise ValueError("No spikes above intensity threshold for centroid")

    return (
        wx / w_total,
        wy / w_total,
        wz / w_total,
        w_total / n,
        wavelength_sum / n,
        water_density_sum / n,
    )


def _dominant_residue(spikes: List[Dict[str, Any]]) -> Tuple[int, str]:
    """Find the residue contributing most spikes (by intensity sum).

    Returns:
        (residue_id, residue_label) where label is "<RESNAME><RESID>" style.
        Falls back to (-1, "UNK-1") if no aromatic residue is identified.
    """
    counts: Dict[int, float] = defaultdict(float)
    for s in spikes:
        rid = s.get("aromatic_residue_id", -1)
        counts[rid] += s["intensity"]

    if not counts:
        return -1, "UNK-1"

    best_rid = max(counts, key=counts.get)  # type: ignore[arg-type]
    # Build a human-readable label from the spike type
    spike_type = spikes[0]["type"] if spikes else "UNK"
    return best_rid, f"{spike_type}{best_rid}"


def _spikes_to_features(
    grouped: Dict[str, List[Dict[str, Any]]],
) -> List[PharmacophoreFeature]:
    """Convert grouped spikes into PharmacophoreFeature list.

    Applies the SPIKE_TYPE_TO_FEATURE mapping plus the dual-feature rule:
    TYR spikes with high water density additionally produce an HBA feature.
    """
    features: List[PharmacophoreFeature] = []

    for spike_type, spikes in grouped.items():
        if not spikes:
            continue

        primary_feature = SPIKE_TYPE_TO_FEATURE.get(spike_type)
        if primary_feature is None:
            logger.warning("Unknown spike type %r, skipping", spike_type)
            continue

        try:
            cx, cy, cz, mean_int, mean_wl, mean_wd = _compute_weighted_centroid(
                spikes
            )
        except ValueError:
            logger.debug("All spikes below threshold for type %s", spike_type)
            continue

        res_id, res_name = _dominant_residue(spikes)

        features.append(
            PharmacophoreFeature(
                feature_type=primary_feature,
                x=round(cx, 3),
                y=round(cy, 3),
                z=round(cz, 3),
                intensity=round(mean_int, 4),
                source_spike_type=spike_type,
                source_residue_id=res_id,
                source_residue_name=res_name,
                wavelength_nm=round(mean_wl, 1),
                water_density=round(mean_wd, 6),
            )
        )

        # Dual-feature rule: TYR with high water density → extra HBA
        if spike_type == "TYR" and mean_wd >= _WATER_DENSITY_HBA_THRESHOLD:
            features.append(
                PharmacophoreFeature(
                    feature_type="HBA",
                    x=round(cx, 3),
                    y=round(cy, 3),
                    z=round(cz, 3),
                    intensity=round(mean_int * 0.7, 4),  # reduced weight
                    source_spike_type=spike_type,
                    source_residue_id=res_id,
                    source_residue_name=res_name,
                    wavelength_nm=round(mean_wl, 1),
                    water_density=round(mean_wd, 6),
                )
            )

    # Sort by intensity descending for downstream truncation (PGMG max 8)
    features.sort(key=lambda f: f.intensity, reverse=True)
    return features


def _build_exclusion_spheres(
    lining_residues: List[Dict[str, Any]],
    radius: float = _EXCLUSION_RADIUS,
) -> List[ExclusionSphere]:
    """Build exclusion spheres from binding-site lining residues.

    Uses the residue centroid approximated from the site centroid + offset
    based on min_distance.  In production, PDB coordinates would be used;
    here we approximate from the binding_sites.json metadata.
    """
    spheres: List[ExclusionSphere] = []
    for res in lining_residues:
        resid = res["resid"]
        resname = res.get("resname", "UNK")
        chain = res.get("chain", "A")
        label = f"CA:{resname}{resid}:{chain}"

        # Use min_distance as a proxy — exclusion spheres at the pocket
        # boundary.  Real implementation would read PDB heavy-atom coords.
        # For now, we record a placeholder that will be overridden when
        # PDB coords are available.
        spheres.append(
            ExclusionSphere(
                x=0.0,  # placeholder — set by _place_exclusion_spheres
                y=0.0,
                z=0.0,
                radius=round(radius, 3),
                source_atom=label,
            )
        )
    return spheres


def _place_exclusion_spheres(
    spheres: List[ExclusionSphere],
    pocket_centroid: Tuple[float, float, float],
    lining_residues: List[Dict[str, Any]],
) -> List[ExclusionSphere]:
    """Place exclusion spheres radially around the pocket centroid.

    Uses min_distance from each lining residue to estimate position along
    a radial direction from centroid.  Distributes residues evenly in angle
    space when no PDB coords are available.
    """
    import math

    cx, cy, cz = pocket_centroid
    n = len(spheres)
    placed: List[ExclusionSphere] = []

    for i, (sphere, res) in enumerate(zip(spheres, lining_residues)):
        dist = res.get("min_distance", 5.0)
        # Distribute evenly on a sphere surface using golden angle
        phi = math.acos(1 - 2.0 * (i + 0.5) / max(n, 1))
        theta = math.pi * (1 + 5**0.5) * i

        x = cx + dist * math.sin(phi) * math.cos(theta)
        y = cy + dist * math.sin(phi) * math.sin(theta)
        z = cz + dist * math.cos(phi)

        placed.append(
            ExclusionSphere(
                x=round(x, 3),
                y=round(y, 3),
                z=round(z, 3),
                radius=sphere.radius,
                source_atom=sphere.source_atom,
            )
        )
    return placed


def _compute_run_hash(spike_path: str, binding_sites_path: Optional[str]) -> str:
    """SHA-256 hash of the input file paths for provenance tracking."""
    h = hashlib.sha256()
    h.update(spike_path.encode())
    if binding_sites_path:
        h.update(binding_sites_path.encode())
    return h.hexdigest()


def convert(
    spike_json_path: str,
    binding_sites_path: Optional[str] = None,
    target_name: str = "UNKNOWN",
    pdb_id: str = "UNKNOWN",
    pocket_index: int = 0,
) -> SpikePharmacophore:
    """Convert PRISM spike events JSON to a SpikePharmacophore.

    Args:
        spike_json_path: Path to ``*.spike_events.json`` file.
        binding_sites_path: Optional path to ``*.binding_sites.json`` for
            lining residues / exclusion spheres.
        target_name: Target identifier (e.g. "KRAS_G12C").
        pdb_id: PDB accession of the input structure.
        pocket_index: Which pocket to extract (for multi-pocket binding sites).

    Returns:
        A validated SpikePharmacophore instance.

    Raises:
        ValueError: If fewer than 2 pharmacophore features are found.
        FileNotFoundError: If input files don't exist.
    """
    spike_path = Path(spike_json_path)
    if not spike_path.exists():
        raise FileNotFoundError(f"Spike events file not found: {spike_json_path}")

    with open(spike_path) as f:
        spike_data = json.load(f)

    centroid = tuple(spike_data["centroid"])
    raw_spikes: List[Dict[str, Any]] = spike_data["spikes"]

    logger.info(
        "Loaded %d spike events from %s (centroid=%.1f, %.1f, %.1f)",
        len(raw_spikes),
        spike_path.name,
        *centroid,
    )

    # Group by spike type
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in raw_spikes:
        grouped[s["type"]].append(s)

    logger.info(
        "Spike type distribution: %s",
        {k: len(v) for k, v in sorted(grouped.items())},
    )

    # Convert to pharmacophore features
    features = _spikes_to_features(grouped)

    if len(features) < 2:
        raise ValueError(
            f"Only {len(features)} pharmacophore feature(s) found; need >=2. "
            f"Check spike data quality or intensity threshold."
        )

    # Lining residues + exclusion spheres
    lining_residue_ids: List[int] = []
    exclusion_spheres: List[ExclusionSphere] = []

    if binding_sites_path:
        bs_path = Path(binding_sites_path)
        if not bs_path.exists():
            raise FileNotFoundError(
                f"Binding sites file not found: {binding_sites_path}"
            )
        with open(bs_path) as f:
            bs_data = json.load(f)

        sites = bs_data.get("sites", [])
        if pocket_index >= len(sites):
            raise IndexError(
                f"pocket_index={pocket_index} but only {len(sites)} sites available"
            )
        site = sites[pocket_index]
        lining_res = site.get("lining_residues", [])
        lining_residue_ids = [r["resid"] for r in lining_res]

        raw_spheres = _build_exclusion_spheres(lining_res)
        exclusion_spheres = _place_exclusion_spheres(
            raw_spheres, centroid, lining_res
        )
        logger.info(
            "Built %d exclusion spheres from %d lining residues",
            len(exclusion_spheres),
            len(lining_res),
        )

    run_hash = _compute_run_hash(spike_json_path, binding_sites_path)

    pharmacophore = SpikePharmacophore(
        target_name=target_name,
        pdb_id=pdb_id,
        pocket_id=spike_data.get("site_id", pocket_index),
        features=features,
        exclusion_spheres=exclusion_spheres,
        pocket_centroid=centroid,
        pocket_lining_residues=lining_residue_ids,
        prism_run_hash=run_hash,
    )

    logger.info(
        "Created SpikePharmacophore: %d features, %d exclusion spheres, "
        "pocket_id=%d, target=%s",
        len(features),
        len(exclusion_spheres),
        pharmacophore.pocket_id,
        target_name,
    )

    return pharmacophore
