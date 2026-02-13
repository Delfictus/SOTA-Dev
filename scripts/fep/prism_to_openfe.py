"""PRISM-to-OpenFE bridge — convert docking results to alchemical networks.

Input:  FilteredCandidate + DockingResult + SpikePharmacophore
Output: OpenFE-compatible protocol setup dict (AlchemicalNetwork spec)

When OpenFE is available, this produces real gufe/openfe objects.
When running without OpenFE (e.g. in tests), it produces equivalent
plain-dict representations that serialize identically.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ── Lazy OpenFE imports (graceful fallback) ───────────────────────────────

_HAS_OPENFE = False
try:
    import gufe
    import openfe
    from openfe import ChemicalSystem, SmallMoleculeComponent, ProteinComponent, SolventComponent
    _HAS_OPENFE = True
except ImportError:
    logger.debug("OpenFE not installed — using dict-based fallback")


@dataclass
class ChemicalSystemSpec:
    """Plain-dict representation of an OpenFE ChemicalSystem.

    Works without OpenFE installed.  When OpenFE is available, can be
    converted to real objects via ``to_openfe()``.
    """
    protein_pdb_path: str
    ligand_sdf_block: str
    ligand_name: str
    solvent_model: str = "tip3p"
    ion_concentration_m: float = 0.15
    forcefield: str = "openff-2.2.0"
    protein_forcefield: str = "amber/ff14SB.xml"
    water_model: str = "amber/tip3p_standard.xml"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protein_pdb_path": self.protein_pdb_path,
            "ligand_sdf_block": self.ligand_sdf_block,
            "ligand_name": self.ligand_name,
            "solvent_model": self.solvent_model,
            "ion_concentration_m": self.ion_concentration_m,
            "forcefield": self.forcefield,
            "protein_forcefield": self.protein_forcefield,
            "water_model": self.water_model,
        }

    def to_openfe(self) -> Any:
        """Convert to real OpenFE ChemicalSystem (requires openfe)."""
        if not _HAS_OPENFE:
            raise RuntimeError("OpenFE is not installed")

        from openfe import ProteinComponent, SolventComponent
        from openfe import SmallMoleculeComponent
        from rdkit import Chem

        protein = ProteinComponent.from_pdb_file(self.protein_pdb_path)
        mol = Chem.MolFromMolBlock(self.ligand_sdf_block, removeHs=False)
        if mol is None:
            raise ValueError(f"Failed to parse ligand SDF for {self.ligand_name}")
        ligand = SmallMoleculeComponent(mol, name=self.ligand_name)
        solvent = SolventComponent(
            positive_ion="Na+",
            negative_ion="Cl-",
            ion_concentration=self.ion_concentration_m,
        )

        return ChemicalSystem(
            {"protein": protein, "ligand": ligand, "solvent": solvent},
            name=f"{self.ligand_name}_complex",
        )


@dataclass
class AlchemicalNetworkSpec:
    """Specification for an ABFE alchemical network.

    Contains everything needed to set up and run ABFE for one compound.
    """
    compound_id: str
    complex_system: ChemicalSystemSpec
    solvent_system: ChemicalSystemSpec
    restraint_info: Dict[str, Any] = field(default_factory=dict)
    pharmacophore_match: str = ""
    protocol_name: str = "AbsoluteSolvationProtocol"
    network_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compound_id": self.compound_id,
            "complex_system": self.complex_system.to_dict(),
            "solvent_system": self.solvent_system.to_dict(),
            "restraint_info": self.restraint_info,
            "pharmacophore_match": self.pharmacophore_match,
            "protocol_name": self.protocol_name,
            "network_hash": self.network_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AlchemicalNetworkSpec:
        d = dict(d)
        d["complex_system"] = ChemicalSystemSpec(**d["complex_system"])
        d["solvent_system"] = ChemicalSystemSpec(**d["solvent_system"])
        return cls(**d)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved network spec: %s", path)

    @classmethod
    def load(cls, path: str) -> AlchemicalNetworkSpec:
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _compute_pharmacophore_match(
    docking_poses: list,
    pharmacophore_features: list,
    tolerance_a: float = 2.0,
) -> str:
    """Compute a human-readable pharmacophore match summary.

    This is a lightweight geometric check — how many pharmacophore features
    have a docked ligand atom within ``tolerance_a`` Angstrom.
    """
    if not pharmacophore_features:
        return "0/0 features"

    # In a real implementation, we would parse the SDF mol block and
    # check atom positions against feature coordinates.
    # For now, return the feature count as a template.
    n_features = len(pharmacophore_features)
    return f"0/{n_features} features within {tolerance_a}A (requires RDKit atom coords)"


def build_network_from_prism(
    docking_result: Any,
    pharmacophore: Any,
    restraint_dict: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
) -> AlchemicalNetworkSpec:
    """Build an AlchemicalNetworkSpec from PRISM pipeline outputs.

    Args:
        docking_result: DockingResult interface object.
        pharmacophore: SpikePharmacophore interface object.
        restraint_dict: Pre-computed Boresch restraint (from restraint_selector).
        output_dir: Optional directory to save the network JSON.

    Returns:
        AlchemicalNetworkSpec ready for prepare_abfe.
    """
    if not docking_result.poses:
        raise ValueError(f"No docking poses for {docking_result.compound_id}")

    best_pose = docking_result.poses[0]

    # Compute network hash for reproducibility tracking
    hash_input = (
        docking_result.compound_id
        + docking_result.receptor_pdb
        + best_pose.mol_block[:200]
        + pharmacophore.prism_run_hash
    )
    network_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    # Pharmacophore match summary
    phore_match = _compute_pharmacophore_match(
        docking_result.poses,
        pharmacophore.features,
    )

    complex_sys = ChemicalSystemSpec(
        protein_pdb_path=docking_result.receptor_pdb,
        ligand_sdf_block=best_pose.mol_block,
        ligand_name=docking_result.compound_id,
    )

    # Solvent system (ligand only, no protein)
    solvent_sys = ChemicalSystemSpec(
        protein_pdb_path="",  # No protein in solvent leg
        ligand_sdf_block=best_pose.mol_block,
        ligand_name=docking_result.compound_id,
    )

    network = AlchemicalNetworkSpec(
        compound_id=docking_result.compound_id,
        complex_system=complex_sys,
        solvent_system=solvent_sys,
        restraint_info=restraint_dict or {},
        pharmacophore_match=phore_match,
        network_hash=network_hash,
    )

    if output_dir:
        out_path = os.path.join(output_dir, f"{docking_result.compound_id}_network.json")
        network.save(out_path)

    return network


def build_rbfe_network(
    docking_results: Sequence[Any],
    pharmacophore: Any,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an RBFE star-map network from multiple docking results.

    For RBFE, we need atom mappings between ligand pairs. This function
    produces the network topology (which pairs to run).

    Args:
        docking_results: List of DockingResult objects for the ligand series.
        pharmacophore: SpikePharmacophore for the target pocket.
        output_dir: Optional directory to save the network JSON.

    Returns:
        Dict with star-map edges and per-compound network specs.
    """
    if len(docking_results) < 2:
        raise ValueError("RBFE requires at least 2 compounds")

    # Star map: use the first compound as the hub
    hub = docking_results[0]
    edges = []
    compounds = {}

    for dr in docking_results:
        spec = build_network_from_prism(dr, pharmacophore)
        spec.protocol_name = "RelativeHybridTopologyProtocol"
        compounds[dr.compound_id] = spec.to_dict()

        if dr.compound_id != hub.compound_id:
            edges.append({
                "ligand_a": hub.compound_id,
                "ligand_b": dr.compound_id,
                "mapping_type": "lomap",
            })

    rbfe_network = {
        "topology": "star",
        "hub_compound": hub.compound_id,
        "edges": edges,
        "compounds": compounds,
        "n_compounds": len(docking_results),
        "n_edges": len(edges),
    }

    if output_dir:
        out_path = os.path.join(output_dir, "rbfe_network.json")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rbfe_network, f, indent=2)
        logger.info("Saved RBFE network: %s", out_path)

    return rbfe_network
