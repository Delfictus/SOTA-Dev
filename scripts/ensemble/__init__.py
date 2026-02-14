"""WT-7: Ensemble Scoring + P_open + Interaction Entropy.

Modules
-------
block_analysis
    Block-averaging convergence analysis (Flyvbjerg & Petersen).
ensemble_mmgbsa
    Ensemble-averaged MM-GBSA free-energy scoring.
interaction_entropy
    Interaction Entropy method (Duan et al., JACS 2016).
pocket_popen
    Pocket open-probability from PRISM multi-stream trajectories.
msm_builder
    Markov State Model builder (P3 stub).
"""
from .block_analysis import BlockAverageResult, block_average, compute_block_sem
from .ensemble_mmgbsa import (
    EnsembleMMGBSAConfig,
    compute_ensemble_mmgbsa,
    get_hotspot_residues,
    run_ensemble_mmgbsa,
    select_frames,
)
from .interaction_entropy import (
    compute_ie,
    compute_interaction_energy,
    compute_interaction_entropy,
    ie_from_mmgbsa_components,
)
from .pocket_popen import (
    CLASSIFICATION_RARE_EVENT,
    CLASSIFICATION_STABLE_OPEN,
    CLASSIFICATION_TRANSIENT,
    bootstrap_popen,
    classify_druggability,
    compute_binary_trajectory,
    compute_lifetimes,
    compute_popen,
    compute_volume_autocorrelation,
    popen_from_trajectory_files,
)

__all__ = [
    # block_analysis
    "BlockAverageResult",
    "block_average",
    "compute_block_sem",
    # ensemble_mmgbsa
    "EnsembleMMGBSAConfig",
    "compute_ensemble_mmgbsa",
    "get_hotspot_residues",
    "run_ensemble_mmgbsa",
    "select_frames",
    # interaction_entropy
    "compute_ie",
    "compute_interaction_energy",
    "compute_interaction_entropy",
    "ie_from_mmgbsa_components",
    # pocket_popen
    "CLASSIFICATION_RARE_EVENT",
    "CLASSIFICATION_STABLE_OPEN",
    "CLASSIFICATION_TRANSIENT",
    "bootstrap_popen",
    "classify_druggability",
    "compute_binary_trajectory",
    "compute_lifetimes",
    "compute_popen",
    "compute_volume_autocorrelation",
    "popen_from_trajectory_files",
]
