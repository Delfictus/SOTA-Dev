"""ResidueMapping interface — PDB ↔ Topology ↔ UniProt residue reconciliation.

Critical for cross-database comparisons (see CLAUDE.md Residue ID QC).
PDB residue IDs ≠ Topology residue IDs ≠ UniProt sequence positions.

Consumed by all WTs that reference residue positions: WT-1 (pharmacophore
lining), WT-2 (FEP restraints), WT-5 (preprocessing), WT-6 (explicit
solvent), WT-7 (ensemble scoring), and WT-4 (reporting).
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ResidueEntry:
    """Mapping for a single residue across three numbering systems.

    Attributes:
        topology_id:    Residue index in the PRISM topology (0-based).
        pdb_resid:      PDB residue sequence number (may include insertion
                        codes via pdb_insertion_code).
        pdb_chain:      PDB chain identifier (e.g. "A").
        pdb_insertion_code: PDB insertion code (e.g. "A" in 27A), or empty
                        string if none.
        uniprot_position: 1-based position in the UniProt canonical sequence,
                        or None if unmapped.
        residue_name:   Three-letter amino-acid code (e.g. "ALA").
    """
    topology_id: int
    pdb_resid: int
    pdb_chain: str
    pdb_insertion_code: str
    uniprot_position: Optional[int]
    residue_name: str

    @property
    def pdb_label(self) -> str:
        """Human-readable PDB label, e.g. 'A:ALA142' or 'A:ALA27A'."""
        ins = self.pdb_insertion_code if self.pdb_insertion_code else ""
        return f"{self.pdb_chain}:{self.residue_name}{self.pdb_resid}{ins}"


@dataclass
class ResidueMapping:
    """Complete residue-numbering map for a structure.

    Provides bidirectional lookup between topology IDs, PDB residue numbers,
    and UniProt sequence positions.  Must be validated after prism-prep
    (see CLAUDE.md Residue QC Checklist).

    Attributes:
        pdb_id:             PDB accession.
        uniprot_id:         UniProt accession (e.g. "P01116").
        chains:             List of chain identifiers present.
        entries:            Complete list of ResidueEntry records.
        topology_residue_count: Total residues in the PRISM topology.
        pdb_residue_count:  Total residues in the PDB ATOM records.
        mapping_source:     How the mapping was generated (e.g. "SIFTS",
                            "prism-prep", "manual").
    """
    pdb_id: str
    uniprot_id: str
    chains: List[str]
    entries: List[ResidueEntry]
    topology_residue_count: int
    pdb_residue_count: int
    mapping_source: str = "prism-prep"

    # ── Lookup helpers ───────────────────────────────────────────────────

    def topology_to_pdb(self, topology_id: int) -> Optional[ResidueEntry]:
        """Look up a residue by topology ID."""
        for e in self.entries:
            if e.topology_id == topology_id:
                return e
        return None

    def pdb_to_topology(
        self, pdb_resid: int, chain: str, insertion_code: str = ""
    ) -> Optional[ResidueEntry]:
        """Look up a residue by PDB resid + chain + insertion code."""
        for e in self.entries:
            if (
                e.pdb_resid == pdb_resid
                and e.pdb_chain == chain
                and e.pdb_insertion_code == insertion_code
            ):
                return e
        return None

    def uniprot_to_entries(self, uniprot_pos: int) -> List[ResidueEntry]:
        """Find entries matching a UniProt position (may span chains)."""
        return [
            e for e in self.entries
            if e.uniprot_position == uniprot_pos
        ]

    def topology_ids_to_pdb_labels(self, topo_ids: List[int]) -> List[str]:
        """Batch-convert topology IDs to PDB labels for reporting."""
        labels = []
        for tid in topo_ids:
            entry = self.topology_to_pdb(tid)
            labels.append(entry.pdb_label if entry else f"?:{tid}")
        return labels

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResidueMapping:
        data = copy.deepcopy(d)
        data["entries"] = [ResidueEntry(**e) for e in data["entries"]]
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> ResidueMapping:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> ResidueMapping:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
