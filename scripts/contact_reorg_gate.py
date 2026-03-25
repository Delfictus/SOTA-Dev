#!/usr/bin/env python3
"""PRISM4D — Contact Reorganization Gate.

Computes per-site local contact reorganization metrics from ensemble
trajectory PDBs and applies a hard gate.  A real binding site shows
localized structural rearrangement (contact formation/breakage preferentially
near the pocket), not just global thermal noise.

Usage (standalone):
    python3 scripts/contact_reorg_gate.py \\
        --binding-sites /path/to/binding_sites.json \\
        --trajectory /path/to/ensemble_trajectory.pdb \\
        [--contact-cutoff 6.0] [--site-radius 12.0] \\
        [--max-frames 20] [--out /path/to/contact_reorg.json]

Programmatic:
    from scripts.contact_reorg_gate import ContactReorgGate
    gate = ContactReorgGate()
    results = gate.evaluate_all(sites, trajectory_path)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from scripts.interfaces.contact_reorg_result import ContactReorgResult


# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
@dataclass
class ContactReorgThresholds:
    """Gate thresholds for contact reorganization.

    A site passes if:
        localization_ratio  >= min_localization_ratio  AND
        contact_change_density >= min_contact_change_density
    """

    min_localization_ratio: float = 0.03
    min_contact_change_density: float = 0.3
    contact_cutoff_angstrom: float = 6.0
    site_radius_angstrom: float = 12.0
    max_frames: int = 20


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------
def parse_trajectory_ca(
    pdb_path: str, max_frames: int = 20
) -> List[Dict[str, Tuple[float, float, float]]]:
    """Parse multi-model PDB, extract CA positions per frame.

    Returns list of dicts mapping "chain:resid" -> (x, y, z).
    """
    frames: List[Dict[str, Tuple[float, float, float]]] = []
    current_cas: Dict[str, Tuple[float, float, float]] = {}

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current_cas = {}
            elif line.startswith("ENDMDL"):
                if current_cas:
                    frames.append(current_cas)
                if len(frames) >= max_frames:
                    break
            elif line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain = line[21:22].strip() or "_"
                resi = line[22:26].strip()
                key = f"{chain}:{resi}"
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current_cas[key] = (x, y, z)

    # Single-model PDB (no MODEL/ENDMDL)
    if not frames and current_cas:
        frames.append(current_cas)

    return frames


# ---------------------------------------------------------------------------
# Contact computation
# ---------------------------------------------------------------------------
def _dist(
    p1: Tuple[float, float, float], p2: Tuple[float, float, float]
) -> float:
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    )


def compute_contacts(
    cas: Dict[str, Tuple[float, float, float]], cutoff: float
) -> Set[Tuple[str, str]]:
    """Compute CA-CA contact set (pairs within cutoff, skipping adjacent)."""
    keys = sorted(cas.keys())
    contacts: Set[Tuple[str, str]] = set()
    for i in range(len(keys)):
        for j in range(i + 2, len(keys)):
            if _dist(cas[keys[i]], cas[keys[j]]) < cutoff:
                contacts.add((keys[i], keys[j]))
    return contacts


def local_contacts(
    contacts: Set[Tuple[str, str]],
    cas: Dict[str, Tuple[float, float, float]],
    centroid: Tuple[float, float, float],
    radius: float,
) -> Set[Tuple[str, str]]:
    """Filter contacts to those with at least one residue near centroid."""
    local: Set[Tuple[str, str]] = set()
    for r1, r2 in contacts:
        if r1 not in cas or r2 not in cas:
            continue
        if _dist(cas[r1], centroid) < radius or _dist(cas[r2], centroid) < radius:
            local.add((r1, r2))
    return local


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
class ContactReorgGate:
    """Evaluates contact reorganization for binding sites."""

    def __init__(self, thresholds: Optional[ContactReorgThresholds] = None):
        self.t = thresholds or ContactReorgThresholds()

    def evaluate(
        self,
        site: Dict[str, Any],
        frames: List[Dict[str, Tuple[float, float, float]]],
    ) -> ContactReorgResult:
        """Compute contact reorg metrics and gate decision for one site."""
        site_id = site.get("id", -1)
        centroid_list = site.get("centroid", [0.0, 0.0, 0.0])
        centroid = (centroid_list[0], centroid_list[1], centroid_list[2])

        if len(frames) < 3:
            return ContactReorgResult(
                site_id=site_id,
                contact_change_density=0.0,
                localization_ratio=0.0,
                persistence=0.0,
                boundary_growth=0.0,
                n_frames_analyzed=len(frames),
                gate_pass=True,
                gate_reason="insufficient_frames (<3) — gate bypassed",
            )

        cutoff = self.t.contact_cutoff_angstrom
        radius = self.t.site_radius_angstrom

        ref_contacts = compute_contacts(frames[0], cutoff)
        ref_local = local_contacts(ref_contacts, frames[0], centroid, radius)

        formed_local: List[int] = []
        broken_local: List[int] = []
        total_formed: List[int] = []
        total_broken: List[int] = []

        for i in range(1, len(frames)):
            fc = compute_contacts(frames[i], cutoff)
            formed = fc - ref_contacts
            broken = ref_contacts - fc
            total_formed.append(len(formed))
            total_broken.append(len(broken))
            formed_local.append(
                len(local_contacts(formed, frames[i], centroid, radius))
            )
            broken_local.append(
                len(local_contacts(broken, frames[0], centroid, radius))
            )

        n = len(frames) - 1

        # 1. Contact change density
        ccd = sum(f + b for f, b in zip(formed_local, broken_local)) / n

        # 2. Localization ratio
        total_change = sum(f + b for f, b in zip(total_formed, total_broken))
        local_change = sum(f + b for f, b in zip(formed_local, broken_local))
        lr = local_change / max(total_change, 1)

        # 3. Persistence (early-formed contacts surviving to late frames)
        if len(frames) >= 6:
            early_formed = compute_contacts(frames[2], cutoff) - ref_contacts
            early_local = local_contacts(early_formed, frames[2], centroid, radius)
            late_contacts = compute_contacts(frames[-1], cutoff)
            persisted = early_local & late_contacts
            persistence = len(persisted) / max(len(early_local), 1)
        else:
            persistence = 0.0

        # 4. Boundary growth
        lc_first = len(local_contacts(ref_contacts, frames[0], centroid, radius))
        lc_last = len(
            local_contacts(
                compute_contacts(frames[-1], cutoff),
                frames[-1],
                centroid,
                radius,
            )
        )
        bg = (lc_last - lc_first) / max(lc_first, 1)

        # Gate decision
        passes_lr = lr >= self.t.min_localization_ratio
        passes_ccd = ccd >= self.t.min_contact_change_density

        if passes_lr and passes_ccd:
            gate_pass = True
            reason = "pass"
        elif not passes_lr and not passes_ccd:
            gate_pass = False
            reason = (
                f"localization_ratio={lr:.4f} < {self.t.min_localization_ratio} "
                f"AND contact_change_density={ccd:.2f} < {self.t.min_contact_change_density}"
            )
        elif not passes_lr:
            gate_pass = False
            reason = f"localization_ratio={lr:.4f} < {self.t.min_localization_ratio}"
        else:
            gate_pass = False
            reason = f"contact_change_density={ccd:.2f} < {self.t.min_contact_change_density}"

        return ContactReorgResult(
            site_id=site_id,
            contact_change_density=round(ccd, 4),
            localization_ratio=round(lr, 6),
            persistence=round(persistence, 4),
            boundary_growth=round(bg, 4),
            n_frames_analyzed=n,
            gate_pass=gate_pass,
            gate_reason=reason,
        )

    def evaluate_all(
        self,
        sites: List[Dict[str, Any]],
        trajectory_path: Optional[str] = None,
    ) -> Dict[int, ContactReorgResult]:
        """Evaluate contact reorg gate for all sites.

        If trajectory_path is None, all sites pass (gate is advisory).
        """
        if trajectory_path is None or not Path(trajectory_path).exists():
            return {
                s.get("id", i): ContactReorgResult(
                    site_id=s.get("id", i),
                    contact_change_density=0.0,
                    localization_ratio=0.0,
                    persistence=0.0,
                    boundary_growth=0.0,
                    n_frames_analyzed=0,
                    gate_pass=True,
                    gate_reason="no_trajectory — gate bypassed",
                )
                for i, s in enumerate(sites)
            }

        frames = parse_trajectory_ca(trajectory_path, self.t.max_frames)
        return {
            s.get("id", i): self.evaluate(s, frames)
            for i, s in enumerate(sites)
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D Contact Reorganization Gate"
    )
    parser.add_argument(
        "--binding-sites", required=True, help="Path to binding_sites.json"
    )
    parser.add_argument(
        "--trajectory", default=None, help="Path to ensemble_trajectory.pdb"
    )
    parser.add_argument("--contact-cutoff", type=float, default=6.0)
    parser.add_argument("--site-radius", type=float, default=12.0)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    with open(args.binding_sites) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get("sites", [])

    thresholds = ContactReorgThresholds(
        contact_cutoff_angstrom=args.contact_cutoff,
        site_radius_angstrom=args.site_radius,
        max_frames=args.max_frames,
    )
    gate = ContactReorgGate(thresholds)
    results = gate.evaluate_all(sites, args.trajectory)

    output = {
        sid: r.to_dict() for sid, r in sorted(results.items())
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {len(output)} results to {args.out}")
    else:
        passed = sum(1 for r in results.values() if r.gate_pass)
        blocked = len(results) - passed
        print(f"Contact Reorg Gate: {passed} passed, {blocked} blocked / {len(results)} sites")
        for sid, r in sorted(results.items()):
            status = "PASS" if r.gate_pass else "BLOCK"
            print(
                f"  site {sid:>3}: {status}  "
                f"ccd={r.contact_change_density:.2f}  "
                f"lr={r.localization_ratio:.4f}  "
                f"persist={r.persistence:.2f}  "
                f"bg={r.boundary_growth:+.3f}"
            )


if __name__ == "__main__":
    main()
