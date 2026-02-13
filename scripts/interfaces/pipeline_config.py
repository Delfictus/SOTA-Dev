"""PipelineConfig interface — centralised paths, thresholds, and feature flags.

Consumed by WT-4 (orchestrator) to wire together all pipeline stages, and
by individual WTs to read runtime configuration.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DockingConfig:
    """Docking-stage configuration.

    Attributes:
        engine:             Docking engine ("unidock" | "gnina" | "unidock+gnina").
        exhaustiveness:     Search exhaustiveness (default 32).
        num_modes:          Max poses per ligand (default 20).
        box_padding:        Extra box padding in Angstrom (default 4.0).
        cnn_scoring:        GNINA CNN scoring mode ("rescore" | "refinement" |
                            "none").
        max_box_side:       Maximum box dimension in Angstrom (default 40.0,
                            Vina limit).
    """
    engine: str = "unidock+gnina"
    exhaustiveness: int = 32
    num_modes: int = 20
    box_padding: float = 4.0
    cnn_scoring: str = "rescore"
    max_box_side: float = 40.0


@dataclass
class FilterConfig:
    """Filtering-stage thresholds.

    Attributes:
        qed_min:                Minimum QED score to pass (default 0.3).
        sa_max:                 Maximum SA score to pass (default 6.0).
        lipinski_max_violations: Max Lipinski violations (default 1).
        pains_reject:           Reject PAINS-flagged compounds (default True).
        tanimoto_max:           Maximum Tanimoto similarity to known compounds
                                (novelty filter; default 0.85).
        cluster_diversity_min:  Minimum number of unique clusters to keep
                                (default 5).
    """
    qed_min: float = 0.3
    sa_max: float = 6.0
    lipinski_max_violations: int = 1
    pains_reject: bool = True
    tanimoto_max: float = 0.85
    cluster_diversity_min: int = 5


@dataclass
class FEPConfig:
    """FEP-stage configuration.

    Attributes:
        method:                 FEP method ("ABFE" | "RBFE").
        n_repeats:              Independent repeats per compound (default 3).
        n_lambda_windows:       Lambda windows for alchemical transformation
                                (default 20).
        simulation_time_ns:     Per-window simulation time in ns (default 5.0).
        convergence_threshold:  Max acceptable ΔΔG between repeats (kcal/mol;
                                default 1.0).
        overlap_minimum:        Min acceptable overlap between windows
                                (default 0.03).
        max_protein_rmsd:       Max acceptable backbone RMSD (Angstrom;
                                default 3.0).
    """
    method: str = "ABFE"
    n_repeats: int = 3
    n_lambda_windows: int = 20
    simulation_time_ns: float = 5.0
    convergence_threshold: float = 1.0
    overlap_minimum: float = 0.03
    max_protein_rmsd: float = 3.0


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration tying all stages together.

    Attributes:
        project_name:       Human-readable project name.
        target_name:        Target identifier (e.g. "KRAS_G12C").
        pdb_id:             Input PDB accession.
        receptor_pdb:       Absolute path to the receptor PDB file.
        topology_json:      Absolute path to the PRISM topology JSON.
        binding_sites_json: Absolute path to PRISM binding_sites.json.
        output_dir:         Root output directory.
        conda_env_prefix:   Conda env prefix directory (for subprocess
                            activation).
        docking:            Docking-stage configuration.
        filtering:          Filtering-stage configuration.
        fep:                FEP-stage configuration.
        stages_enabled:     Which pipeline stages are active.
        random_seed:        Global random seed (default 42).
    """
    project_name: str
    target_name: str
    pdb_id: str
    receptor_pdb: str
    topology_json: str
    binding_sites_json: str
    output_dir: str
    conda_env_prefix: str = ""
    docking: DockingConfig = field(default_factory=DockingConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    fep: FEPConfig = field(default_factory=FEPConfig)
    stages_enabled: List[str] = field(
        default_factory=lambda: [
            "pharmacophore", "generation", "filtering", "docking", "fep",
        ]
    )
    random_seed: int = 42

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PipelineConfig:
        data = copy.deepcopy(d)
        data["docking"] = DockingConfig(**data.get("docking", {}))
        data["filtering"] = FilterConfig(**data.get("filtering", {}))
        data["fep"] = FEPConfig(**data.get("fep", {}))
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> PipelineConfig:
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_json_file(cls, path: str) -> PipelineConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str) -> None:
        """Write configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> PipelineConfig:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
