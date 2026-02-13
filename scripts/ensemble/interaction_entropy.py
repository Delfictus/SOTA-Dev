"""Interaction Entropy (IE) method for entropy estimation.

Implements the IE method of Duan et al. (JACS 2016) which replaces the
obsolete and expensive Normal Mode Analysis (NMA) for estimating the
entropic contribution to binding free energy.

Key equation
------------
    -TΔS_IE = kT · ln < exp[ β(E_int - <E_int>) ] >

where E_int is the protein–ligand interaction energy (VdW + Coulomb),
β = 1/(kT), and the average is over MD trajectory frames.

Advantages over NMA
-------------------
- No Hessian computation (10–100× faster)
- Captures anharmonic effects
- Reuses the same MD trajectory as MM-GBSA
- Negligible additional wall time

References
----------
- Duan L, Liu X, Zhang JZH. JACS. 2016;138(17):5722-5728.
- Yan Y et al. J Chem Inf Model. 2017;57(5):1112-1122.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from scripts.interfaces import InteractionEntropy

from .block_analysis import BlockAverageResult, block_average

logger = logging.getLogger(__name__)

# Physical constants
KB_KCAL = 0.001987204  # Boltzmann constant in kcal/(mol·K)
DEFAULT_TEMPERATURE_K = 300.0

# Convergence threshold: block SEM should be < this fraction of |-TdS|
CONVERGENCE_FRACTION = 0.30


def compute_interaction_energy(
    e_complex: np.ndarray,
    e_receptor: np.ndarray,
    e_ligand: np.ndarray,
) -> np.ndarray:
    """Compute per-frame interaction energies.

    Parameters
    ----------
    e_complex : np.ndarray
        Per-frame energies of the complex (VdW + Coulomb).
    e_receptor : np.ndarray
        Per-frame energies of the receptor alone.
    e_ligand : np.ndarray
        Per-frame energies of the ligand alone.

    Returns
    -------
    np.ndarray
        Per-frame interaction energies: E_int = E_complex - E_receptor - E_ligand.

    Raises
    ------
    ValueError
        If arrays have different lengths.
    """
    e_c = np.asarray(e_complex, dtype=np.float64)
    e_r = np.asarray(e_receptor, dtype=np.float64)
    e_l = np.asarray(e_ligand, dtype=np.float64)

    if not (len(e_c) == len(e_r) == len(e_l)):
        raise ValueError(
            f"Array length mismatch: complex={len(e_c)}, "
            f"receptor={len(e_r)}, ligand={len(e_l)}"
        )

    return e_c - e_r - e_l


def compute_ie(
    e_interaction: np.ndarray,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> Tuple[float, float]:
    """Compute the interaction entropy −TΔS from per-frame interaction energies.

    Parameters
    ----------
    e_interaction : np.ndarray
        Per-frame protein–ligand interaction energies (kcal/mol).
    temperature_k : float
        Temperature in Kelvin.

    Returns
    -------
    tuple of (float, float)
        (minus_t_delta_s, mean_e_interaction) in kcal/mol.

    Raises
    ------
    ValueError
        If input array is empty.
    """
    e_int = np.asarray(e_interaction, dtype=np.float64).ravel()
    if len(e_int) == 0:
        raise ValueError("Empty interaction energy array")

    kT = KB_KCAL * temperature_k
    beta = 1.0 / kT

    mean_e = np.mean(e_int)
    delta_e = e_int - mean_e

    # Use logsumexp trick for numerical stability:
    # ln <exp(β·δE)> = ln(1/N · Σ exp(β·δE))
    #                = -ln(N) + ln(Σ exp(β·δE))
    # Using scipy-like logsumexp:
    beta_de = beta * delta_e
    max_bde = np.max(beta_de)
    log_mean_exp = max_bde + np.log(np.mean(np.exp(beta_de - max_bde)))

    minus_t_delta_s = kT * log_mean_exp

    return float(minus_t_delta_s), float(mean_e)


def compute_interaction_entropy(
    e_interaction: np.ndarray,
    delta_h: Optional[float] = None,
    compound_id: str = "unknown",
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> InteractionEntropy:
    """Full interaction entropy calculation with convergence analysis.

    Parameters
    ----------
    e_interaction : np.ndarray
        Per-frame protein–ligand interaction energies (kcal/mol).
    delta_h : float, optional
        Enthalpy from MM-GBSA.  If *None*, uses the mean interaction energy.
    compound_id : str
        Compound identifier.
    temperature_k : float
        Temperature in Kelvin.

    Returns
    -------
    InteractionEntropy
        Interface-compliant IE result.

    Raises
    ------
    ValueError
        If fewer than 5 frames are provided.
    """
    e_int = np.asarray(e_interaction, dtype=np.float64).ravel()
    n_frames = len(e_int)

    if n_frames < 5:
        raise ValueError(f"Need at least 5 frames for IE, got {n_frames}")

    minus_t_ds, mean_e = compute_ie(e_int, temperature_k)

    if delta_h is None:
        delta_h = mean_e

    delta_g_ie = delta_h + minus_t_ds

    # Convergence: block-average the per-frame −TΔS contributions
    # We assess convergence by block-averaging the raw E_int series
    ba = block_average(e_int)
    convergence_block_std = ba.sem

    return InteractionEntropy(
        compound_id=compound_id,
        minus_t_delta_s=minus_t_ds,
        delta_h=delta_h,
        delta_g_ie=delta_g_ie,
        n_frames=n_frames,
        convergence_block_std=convergence_block_std,
    )


def ie_from_mmgbsa_components(
    vdw_energies: np.ndarray,
    elec_energies: np.ndarray,
    delta_h_mmgbsa: float,
    compound_id: str = "unknown",
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> InteractionEntropy:
    """Convenience: compute IE directly from MM-GBSA VdW + Coulomb arrays.

    The interaction energy is approximated as E_int ≈ VdW + Coulomb
    (gas-phase terms from MM-GBSA decomposition).  This avoids recomputing
    interaction energies separately.

    Parameters
    ----------
    vdw_energies : np.ndarray
        Per-frame VdW energies from MM-GBSA decomposition.
    elec_energies : np.ndarray
        Per-frame electrostatic energies from MM-GBSA decomposition.
    delta_h_mmgbsa : float
        Enthalpy (ΔH) from ensemble MM-GBSA mean.
    compound_id : str
        Compound identifier.
    temperature_k : float
        Temperature in Kelvin.

    Returns
    -------
    InteractionEntropy
        IE result with ΔG_IE = ΔH_MMGBSA + (−TΔS_IE).
    """
    vdw = np.asarray(vdw_energies, dtype=np.float64)
    elec = np.asarray(elec_energies, dtype=np.float64)

    if len(vdw) != len(elec):
        raise ValueError(
            f"VdW and elec array length mismatch: {len(vdw)} vs {len(elec)}"
        )

    e_int = vdw + elec
    return compute_interaction_entropy(
        e_int,
        delta_h=delta_h_mmgbsa,
        compound_id=compound_id,
        temperature_k=temperature_k,
    )
