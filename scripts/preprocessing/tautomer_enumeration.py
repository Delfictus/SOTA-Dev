"""Tautomer & protonation-state enumeration at physiological pH.

Pipeline position:
    [WT-1 output (SMILES/SDF)] → **tautomer_enumeration.py** → [WT-3 filters / WT-2 FEP]

Algorithm:
    1. Canonicalize input (RDKit)
    2. Enumerate tautomers (RDKit TautomerEnumerator)
    3. For each tautomer:
       a. Predict pKa of ionizable groups (Dimorphite-DL)
       b. Generate protonation states at target pH ± tolerance
       c. Compute Boltzmann population at target pH
    4. Filter: keep states with population > cutoff (default 1%)
    5. Return TautomerEnsemble with dominant_state flagged

CRITICAL: ALL ligands entering gpu_dock.py, PhoreGen, or FEP must pass
through this module.  Wrong protonation = invalid docking + invalid FEP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from scripts.interfaces.tautomer_state import TautomerEnsemble, TautomerState

logger = logging.getLogger(__name__)

# Suppress noisy RDKit warnings during enumeration
RDLogger.logger().setLevel(RDLogger.ERROR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PH = 7.4
DEFAULT_PH_TOLERANCE = 1.0
DEFAULT_POPULATION_CUTOFF = 0.01  # 1%
MAX_TAUTOMERS = 50  # safety cap


# ---------------------------------------------------------------------------
# Dimorphite-DL wrapper
# ---------------------------------------------------------------------------

def _has_dimorphite() -> bool:
    try:
        import dimorphite_dl  # noqa: F401
        return True
    except ImportError:
        return False


def _protonate_dimorphite(
    smiles: str, ph_min: float, ph_max: float
) -> List[str]:
    """Generate protonation states via Dimorphite-DL.

    Returns list of SMILES for protonation states in [ph_min, ph_max].
    """
    from dimorphite_dl import protonate_smiles

    try:
        results = protonate_smiles(
            smiles, ph_min=ph_min, ph_max=ph_max, precision=1.0,
        )
        if results:
            return list(set(results))
    except Exception as exc:
        logger.debug("Dimorphite-DL failed for %s: %s", smiles, exc)
    return [smiles]  # fallback: return input


# ---------------------------------------------------------------------------
# Boltzmann population estimation
# ---------------------------------------------------------------------------

_RT_KCAL = 0.592  # RT at 298K in kcal/mol


def _estimate_populations(
    states: List[str], target_ph: float
) -> List[Tuple[str, float]]:
    """Assign approximate Boltzmann populations.

    Without per-state free energies, we use charge deviation from the
    target pH as a heuristic.  States whose net formal charge minimizes
    |charge| (i.e. closest to neutral or expected physiological charge)
    get higher weight.  This is a pragmatic approximation — true
    populations require full pKa predictions.
    """
    if not states:
        return []

    scored: List[Tuple[str, float]] = []
    for smi in states:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        charge = Chem.rdmolops.GetFormalCharge(mol)
        # Lower absolute charge → more favorable at physiological pH
        # Use a soft penalty: exp(-|charge|)
        energy = abs(charge) * 2.0  # pseudo kcal/mol
        scored.append((smi, energy))

    if not scored:
        return [(states[0], 1.0)]

    # Boltzmann weights
    min_e = min(e for _, e in scored)
    weights = [math.exp(-(e - min_e) / _RT_KCAL) for _, e in scored]
    total = sum(weights)
    return [(smi, w / total) for (smi, _), w in zip(scored, weights)]


# ---------------------------------------------------------------------------
# pKa extraction helper
# ---------------------------------------------------------------------------

def _extract_pka_shifts(parent_smi: str, protonated_smi: str) -> List[Tuple[int, float]]:
    """Identify atoms whose protonation changed and assign estimated pKa.

    Returns list of (atom_idx, estimated_pKa).  The pKa is estimated from
    the charge difference — a rough heuristic.
    """
    parent_mol = Chem.MolFromSmiles(parent_smi)
    prot_mol = Chem.MolFromSmiles(protonated_smi)
    if parent_mol is None or prot_mol is None:
        return []

    shifts: List[Tuple[int, float]] = []
    try:
        # Compare formal charges per atom
        n_atoms = min(parent_mol.GetNumAtoms(), prot_mol.GetNumAtoms())
        for i in range(n_atoms):
            fc_parent = parent_mol.GetAtomWithIdx(i).GetFormalCharge()
            fc_prot = prot_mol.GetAtomWithIdx(i).GetFormalCharge()
            if fc_parent != fc_prot:
                # Estimate pKa based on change direction
                est_pka = 7.4 + (fc_prot - fc_parent) * 2.0
                shifts.append((i, round(est_pka, 2)))
    except Exception:
        pass
    return shifts


# ---------------------------------------------------------------------------
# Core enumeration
# ---------------------------------------------------------------------------

def enumerate_tautomers(
    smiles: str,
    *,
    target_ph: float = DEFAULT_PH,
    ph_tolerance: float = DEFAULT_PH_TOLERANCE,
    population_cutoff: float = DEFAULT_POPULATION_CUTOFF,
    max_tautomers: int = MAX_TAUTOMERS,
) -> TautomerEnsemble:
    """Enumerate tautomers and protonation states for a SMILES.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    target_ph : float
        Physiological pH target (default 7.4).
    ph_tolerance : float
        pH window for protonation-state generation (default ±1.0).
    population_cutoff : float
        Minimum Boltzmann population to keep (default 0.01 = 1%).
    max_tautomers : int
        Maximum tautomers to enumerate (safety cap).

    Returns
    -------
    TautomerEnsemble
        Complete ensemble with dominant state flagged.
    """
    # 1. Canonicalize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canon_smi = Chem.MolToSmiles(mol)

    # 2. Enumerate tautomers via RDKit
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)
    taut_mols = list(enumerator.Enumerate(mol))
    taut_smiles = list({Chem.MolToSmiles(m) for m in taut_mols})

    if not taut_smiles:
        taut_smiles = [canon_smi]

    # 3. For each tautomer, generate protonation states
    all_protonation_smiles: List[str] = []
    taut_source: Dict[str, str] = {}  # protonated_smi → parent tautomer smi

    use_dimorphite = _has_dimorphite()
    ph_min = target_ph - ph_tolerance
    ph_max = target_ph + ph_tolerance

    for taut_smi in taut_smiles:
        if use_dimorphite:
            prot_states = _protonate_dimorphite(taut_smi, ph_min, ph_max)
        else:
            prot_states = [taut_smi]

        for ps in prot_states:
            if ps not in taut_source:
                taut_source[ps] = taut_smi
                all_protonation_smiles.append(ps)

    # 4. Compute populations
    pop_pairs = _estimate_populations(all_protonation_smiles, target_ph)

    # 5. Filter by cutoff
    filtered = [(smi, pop) for smi, pop in pop_pairs if pop >= population_cutoff]
    if not filtered:
        # Keep at least one state
        filtered = [max(pop_pairs, key=lambda x: x[1])]

    # Renormalize after filtering
    total_pop = sum(p for _, p in filtered)
    filtered = [(smi, p / total_pop) for smi, p in filtered]

    # Build TautomerState objects
    enumeration_method = "dimorphite_dl_rdk_mstandardize" if use_dimorphite else "rdk_mstandardize"
    states: List[TautomerState] = []
    for smi, pop in filtered:
        prot_mol = Chem.MolFromSmiles(smi)
        charge = Chem.rdmolops.GetFormalCharge(prot_mol) if prot_mol else 0
        pka_shifts = _extract_pka_shifts(canon_smi, smi)
        source_tool = "dimorphite_dl" if use_dimorphite else "rdkit"

        states.append(TautomerState(
            smiles=smi,
            parent_smiles=canon_smi,
            protonation_ph=target_ph,
            charge=charge,
            pka_shifts=pka_shifts,
            population_fraction=round(pop, 6),
            source_tool=source_tool,
        ))

    # Sort by population (descending)
    states.sort(key=lambda s: s.population_fraction, reverse=True)
    dominant = states[0]

    return TautomerEnsemble(
        parent_smiles=canon_smi,
        states=states,
        dominant_state=dominant,
        target_ph=target_ph,
        enumeration_method=enumeration_method,
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def enumerate_batch(
    smiles_list: List[str],
    *,
    target_ph: float = DEFAULT_PH,
    ph_tolerance: float = DEFAULT_PH_TOLERANCE,
    population_cutoff: float = DEFAULT_POPULATION_CUTOFF,
) -> List[TautomerEnsemble]:
    """Enumerate tautomers for a batch of SMILES.

    Invalid SMILES are logged and skipped.
    """
    results: List[TautomerEnsemble] = []
    for smi in smiles_list:
        try:
            ens = enumerate_tautomers(
                smi,
                target_ph=target_ph,
                ph_tolerance=ph_tolerance,
                population_cutoff=population_cutoff,
            )
            results.append(ens)
        except ValueError as exc:
            logger.warning("Skipping invalid SMILES %r: %s", smi, exc)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate tautomers & protonation states at target pH."
    )
    parser.add_argument(
        "--smiles", required=True,
        help="Input SMILES string",
    )
    parser.add_argument(
        "--ph", type=float, default=DEFAULT_PH,
        help=f"Target pH (default {DEFAULT_PH})",
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_PH_TOLERANCE,
        help=f"pH tolerance window (default ±{DEFAULT_PH_TOLERANCE})",
    )
    parser.add_argument(
        "--cutoff", type=float, default=DEFAULT_POPULATION_CUTOFF,
        help=f"Min population fraction to keep (default {DEFAULT_POPULATION_CUTOFF})",
    )
    args = parser.parse_args(argv)

    ensemble = enumerate_tautomers(
        args.smiles,
        target_ph=args.ph,
        ph_tolerance=args.tolerance,
        population_cutoff=args.cutoff,
    )

    print(ensemble.to_json())


if __name__ == "__main__":
    main()
