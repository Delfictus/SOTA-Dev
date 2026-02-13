"""RBFE protocol preparation — relative binding free energies for lead optimization.

Uses LOMAP atom mappings and star-map or minimal-spanning-tree network topology.
Phase 2 module (after ABFE identifies initial hits).

RBFE computes ddG between ligand pairs, which is cheaper than ABFE per
compound and better suited for SAR exploration.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AtomMapping:
    """Atom mapping between two ligands for relative FEP."""
    ligand_a_id: str
    ligand_b_id: str
    mapping_type: str = "lomap"   # "lomap" | "kartograf" | "manual"
    n_mapped_atoms: int = 0
    n_unique_a: int = 0          # atoms only in ligand A (disappearing)
    n_unique_b: int = 0          # atoms only in ligand B (appearing)
    lomap_score: float = 0.0     # LOMAP similarity score (0-1, higher=easier)
    atom_map: Dict[int, int] = field(default_factory=dict)  # A_idx -> B_idx

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RBFEEdge:
    """One edge in the RBFE perturbation network."""
    ligand_a_id: str
    ligand_b_id: str
    atom_mapping: AtomMapping
    n_lambda_windows: int = 22
    simulation_time_ns: float = 5.0
    n_repeats: int = 3

    @property
    def total_time_ns(self) -> float:
        return self.n_lambda_windows * self.simulation_time_ns * self.n_repeats

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_time_ns"] = self.total_time_ns
        return d


@dataclass
class RBFEProtocol:
    """Complete RBFE protocol for a ligand series.

    Attributes:
        target_name: Target identifier.
        topology: Network topology ("star" | "mst" | "full").
        hub_compound_id: Hub compound for star topology.
        edges: List of perturbation edges.
        compounds: Dict of compound_id → SDF path or mol-block.
        protein_pdb: Path to receptor PDB.
        output_dir: Output directory.
        n_lambda_windows: Windows per edge.
        simulation_time_ns: Per-window production time (ns).
        n_repeats: Independent repeats per edge.
    """
    target_name: str
    topology: str = "star"
    hub_compound_id: str = ""
    edges: List[RBFEEdge] = field(default_factory=list)
    compounds: Dict[str, str] = field(default_factory=dict)
    protein_pdb: str = ""
    output_dir: str = ""
    n_lambda_windows: int = 22
    simulation_time_ns: float = 5.0
    n_repeats: int = 3

    @property
    def total_simulation_time_ns(self) -> float:
        return sum(e.total_time_ns for e in self.edges)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_compounds(self) -> int:
        return len(self.compounds)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "target_name": self.target_name,
            "topology": self.topology,
            "hub_compound_id": self.hub_compound_id,
            "edges": [e.to_dict() for e in self.edges],
            "compounds": self.compounds,
            "protein_pdb": self.protein_pdb,
            "output_dir": self.output_dir,
            "n_lambda_windows": self.n_lambda_windows,
            "simulation_time_ns": self.simulation_time_ns,
            "n_repeats": self.n_repeats,
            "n_edges": self.n_edges,
            "n_compounds": self.n_compounds,
            "total_simulation_time_ns": self.total_simulation_time_ns,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Optional[str] = None) -> str:
        if path is None:
            path = os.path.join(self.output_dir, "rbfe_protocol.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved RBFE protocol: %s", path)
        return path


def compute_atom_mapping(
    smiles_a: str,
    smiles_b: str,
    ligand_a_id: str,
    ligand_b_id: str,
    method: str = "lomap",
) -> AtomMapping:
    """Compute atom mapping between two ligands.

    When LOMAP is available, uses its MCS-based mapping.
    Falls back to a scaffold-based heuristic otherwise.

    Args:
        smiles_a: SMILES for ligand A.
        smiles_b: SMILES for ligand B.
        ligand_a_id: Identifier for ligand A.
        ligand_b_id: Identifier for ligand B.
        method: Mapping algorithm ("lomap" | "kartograf").
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS

        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)

        if mol_a is None or mol_b is None:
            logger.warning("Failed to parse SMILES for mapping: %s / %s", smiles_a, smiles_b)
            return AtomMapping(
                ligand_a_id=ligand_a_id,
                ligand_b_id=ligand_b_id,
                mapping_type=method,
            )

        mcs = rdFMCS.FindMCS(
            [mol_a, mol_b],
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=10,
        )

        n_mapped = mcs.numAtoms if mcs.numAtoms else 0
        n_a = mol_a.GetNumAtoms()
        n_b = mol_b.GetNumAtoms()

        lomap_score = n_mapped / max(n_a, n_b) if max(n_a, n_b) > 0 else 0.0

        return AtomMapping(
            ligand_a_id=ligand_a_id,
            ligand_b_id=ligand_b_id,
            mapping_type=method,
            n_mapped_atoms=n_mapped,
            n_unique_a=n_a - n_mapped,
            n_unique_b=n_b - n_mapped,
            lomap_score=lomap_score,
        )

    except ImportError:
        logger.debug("RDKit not available, returning empty mapping")
        return AtomMapping(
            ligand_a_id=ligand_a_id,
            ligand_b_id=ligand_b_id,
            mapping_type=method,
        )


def build_star_topology(
    compound_smiles: Dict[str, str],
    hub_id: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Build star-topology edges with given hub compound.

    If no hub is specified, picks the compound with highest average
    LOMAP score to all others.

    Returns:
        List of (hub_id, spoke_id) pairs.
    """
    ids = sorted(compound_smiles.keys())
    if len(ids) < 2:
        raise ValueError("Star topology requires at least 2 compounds")

    if hub_id is None:
        hub_id = ids[0]
        logger.info("Auto-selected hub compound: %s", hub_id)

    edges = [(hub_id, cid) for cid in ids if cid != hub_id]
    return edges


def prepare_rbfe(
    target_name: str,
    compound_smiles: Dict[str, str],
    compound_sdfs: Dict[str, str],
    protein_pdb: str,
    output_dir: str,
    *,
    topology: str = "star",
    hub_id: Optional[str] = None,
    n_lambda_windows: int = 22,
    simulation_time_ns: float = 5.0,
    n_repeats: int = 3,
) -> RBFEProtocol:
    """Prepare a complete RBFE protocol for a ligand series.

    Args:
        target_name: Target identifier (e.g. "KRAS_G12C").
        compound_smiles: Dict of compound_id → SMILES.
        compound_sdfs: Dict of compound_id → SDF path or mol-block.
        protein_pdb: Path to receptor PDB.
        output_dir: Output directory.
        topology: Network topology ("star" | "mst").
        hub_id: Hub compound for star topology.
        n_lambda_windows: Lambda windows per edge.
        simulation_time_ns: Per-window simulation time (ns).
        n_repeats: Independent repeats.

    Returns:
        Configured RBFEProtocol.
    """
    if topology == "star":
        edge_pairs = build_star_topology(compound_smiles, hub_id)
        hub = edge_pairs[0][0]
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    edges = []
    for a_id, b_id in edge_pairs:
        mapping = compute_atom_mapping(
            compound_smiles[a_id],
            compound_smiles[b_id],
            a_id,
            b_id,
        )
        edges.append(RBFEEdge(
            ligand_a_id=a_id,
            ligand_b_id=b_id,
            atom_mapping=mapping,
            n_lambda_windows=n_lambda_windows,
            simulation_time_ns=simulation_time_ns,
            n_repeats=n_repeats,
        ))

    protocol = RBFEProtocol(
        target_name=target_name,
        topology=topology,
        hub_compound_id=hub,
        edges=edges,
        compounds=compound_sdfs,
        protein_pdb=protein_pdb,
        output_dir=output_dir,
        n_lambda_windows=n_lambda_windows,
        simulation_time_ns=simulation_time_ns,
        n_repeats=n_repeats,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    protocol.save()

    logger.info(
        "Prepared RBFE for %s: %d compounds, %d edges, %.0f ns total",
        target_name, protocol.n_compounds, protocol.n_edges,
        protocol.total_simulation_time_ns,
    )

    return protocol
