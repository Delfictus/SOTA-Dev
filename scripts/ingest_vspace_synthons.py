#!/usr/bin/env python3
"""Ingest V-space building blocks with CPU conformer generation and NAGL charges."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import polars as pl
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "00_registry/chemistry/reaction_rules.v1.yml"
DEFAULT_OUTPUT = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/enamine_130k_synthons_3d.parquet"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_a_generative/enamine_130k_synthons_ingest_report.json"
)
DEFAULT_NAGL_MODEL = "openff-gnn-am1bcc-1.0.0.pt"

_WORKER_REGISTRY: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SourceRow:
    synthon_id: str
    vendor_id: str
    smiles: str


@dataclass(frozen=True)
class RoleMatch:
    reaction_id: str
    role_name: str
    reactive_atom_idx: int
    leaving_group_atom_indices: list[int]
    reference_atom_idx: int | None
    atom_map_to_atom_idx: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reaction-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--cpu-chunksize", type=int, default=128)
    parser.add_argument("--nagl-batch-size", type=int, default=10_000)
    parser.add_argument("--nagl-model", type=str, default=DEFAULT_NAGL_MODEL)
    parser.add_argument("--allow-cpu-nagl", action="store_true")
    return parser.parse_args()


def load_registry(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or not payload.get("reactions"):
        raise ValueError(f"invalid reaction registry: {path}")
    return cast(Mapping[str, Any], payload)


def source_rows(path: Path, limit: int | None) -> list[SourceRow]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".smi", ".smiles"}:
        rows: list[SourceRow] = []
        with path.open() as handle:
            for idx, line in enumerate(handle):
                if limit is not None and idx >= limit:
                    break
                parts = line.strip().split()
                if not parts:
                    continue
                smiles = parts[0]
                synthon_id = parts[1] if len(parts) > 1 else f"SYNTHON_{idx:06d}"
                rows.append(SourceRow(synthon_id=synthon_id, vendor_id=synthon_id, smiles=smiles))
        return rows

    lazy = pl.scan_csv(path)
    frame = lazy.head(limit).collect() if limit is not None else lazy.collect(streaming=True)
    columns = set(frame.columns)
    smiles_col = "smiles" if "smiles" in columns else "SMILES"
    id_col = "synthon_id" if "synthon_id" in columns else "anchor_id" if "anchor_id" in columns else "id"
    vendor_col = "vendor_id" if "vendor_id" in columns else id_col
    missing = [column for column in (smiles_col, id_col) if column not in columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return [
        SourceRow(
            synthon_id=str(row[id_col]),
            vendor_id=str(row[vendor_col]),
            smiles=str(row[smiles_col]),
        )
        for row in frame.iter_rows(named=True)
    ]


def canonical_smiles(mol: Chem.Mol) -> str:
    return str(Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True))


def atom_map_lookup(query: Chem.Mol, match: Sequence[int]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for query_atom in query.GetAtoms():
        atom_map = int(query_atom.GetAtomMapNum())
        if atom_map > 0:
            lookup[atom_map] = int(match[int(query_atom.GetIdx())])
    return lookup


def reference_atom_for(mol: Chem.Mol, reactive_atom_idx: int, leaving: set[int]) -> int | None:
    atom = mol.GetAtomWithIdx(reactive_atom_idx)
    heavy_neighbors = [
        int(neighbor.GetIdx())
        for neighbor in atom.GetNeighbors()
        if int(neighbor.GetAtomicNum()) > 1 and int(neighbor.GetIdx()) not in leaving
    ]
    return heavy_neighbors[0] if heavy_neighbors else None


def reaction_role_matches(mol: Chem.Mol, registry: Mapping[str, Any]) -> list[RoleMatch]:
    matches: list[RoleMatch] = []
    for reaction_obj in cast(Sequence[Mapping[str, Any]], registry["reactions"]):
        if not bool(reaction_obj.get("enabled")):
            continue
        reaction_id = str(reaction_obj["reaction_id"])
        roles = cast(Mapping[str, Mapping[str, Any]], reaction_obj["reactant_roles"])
        for role_name, role in roles.items():
            query = Chem.MolFromSmarts(str(role["required_smarts"]))
            if query is None:
                raise ValueError(f"{reaction_id}:{role_name} SMARTS failed to compile")
            for match in mol.GetSubstructMatches(query, uniquify=True):
                atom_map_to_idx = atom_map_lookup(query, match)
                reactive_map = int(role["reactive_atom_map"])
                if reactive_map not in atom_map_to_idx:
                    continue
                leaving = [
                    atom_map_to_idx[int(atom_map)]
                    for atom_map in cast(Sequence[int], role["leaving_group_atom_maps"])
                    if int(atom_map) in atom_map_to_idx
                ]
                reactive = atom_map_to_idx[reactive_map]
                matches.append(
                    RoleMatch(
                        reaction_id=reaction_id,
                        role_name=str(role_name),
                        reactive_atom_idx=reactive,
                        leaving_group_atom_indices=leaving,
                        reference_atom_idx=reference_atom_for(mol, reactive, set(leaving)),
                        atom_map_to_atom_idx=atom_map_to_idx,
                    )
                )
    return matches


def embed_molecule(mol: Chem.Mol) -> tuple[Chem.Mol | None, str | None]:
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 73_001
    params.useRandomCoords = True
    status = int(AllChem.EmbedMolecule(mol_h, params))
    if status != 0:
        return None, f"failed_conformer_etkdg_{status}"
    try:
        if bool(AllChem.MMFFHasAllMoleculeParams(mol_h)):
            AllChem.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94s", maxIters=100)
        else:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=100)
    except Exception as exc:  # noqa: BLE001 - preserve RDKit reason.
        return None, f"failed_conformer_optimize:{exc}"
    return mol_h, None


def conformer_atoms_json(mol: Chem.Mol) -> str:
    conformer = mol.GetConformer()
    rows: list[dict[str, object]] = []
    for atom in mol.GetAtoms():
        atom_idx = int(atom.GetIdx())
        pos = conformer.GetAtomPosition(atom_idx)
        rows.append(
            {
                "atom_idx": atom_idx,
                "atomic_num": int(atom.GetAtomicNum()),
                "symbol": str(atom.GetSymbol()),
                "x": float(pos.x),
                "y": float(pos.y),
                "z": float(pos.z),
            }
        )
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def attachment_vector_json(mol: Chem.Mol, match: RoleMatch) -> str:
    conformer = mol.GetConformer()
    reactive = conformer.GetAtomPosition(match.reactive_atom_idx)
    reference_idx = match.leaving_group_atom_indices[0] if match.leaving_group_atom_indices else match.reference_atom_idx
    if reference_idx is None:
        return json.dumps([1.0, 0.0, 0.0], separators=(",", ":"))
    reference = conformer.GetAtomPosition(reference_idx)
    return json.dumps(
        [float(reference.x - reactive.x), float(reference.y - reactive.y), float(reference.z - reactive.z)],
        separators=(",", ":"),
    )


def sa_score_if_available(mol: Chem.Mol) -> float | None:
    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore[import-not-found]
    except Exception:
        return None
    return float(sascorer.calculateScore(Chem.RemoveHs(mol)))


def base_row(source: SourceRow) -> dict[str, object]:
    return {
        "synthon_id": source.synthon_id,
        "vendor_id": source.vendor_id,
        "canonical_smiles": "",
        "is_valid_mol": False,
        "compatible_reactions_json": "[]",
        "compatible_reaction_roles_json": "[]",
        "reaction_match_atoms_json": "[]",
        "leaving_group_atoms_json": "[]",
        "conformer_atoms_json": "[]",
        "formal_charge": 0,
        "heavy_atom_count": 0,
        "molecular_weight": 0.0,
        "sa_score": None,
        "charge_method": "",
        "partial_charges_json": "[]",
        "ingest_status": "rejected",
        "reject_reason": "",
        "reaction_rule_id": "",
        "reaction_role": "",
        "reaction_tags_json": "[]",
        "attachment_atom_idx": None,
        "leaving_group_atom_idx": None,
        "dihedral_reference_atom_idx": None,
        "attachment_vector_json": "[1.0,0.0,0.0]",
        "leaving_group_formal_charge": 0.0,
        "partial_charge_method": "",
        "conformer_count": 0,
        "_mol_block": "",
    }


def init_worker(registry_path: str) -> None:
    global _WORKER_REGISTRY
    _WORKER_REGISTRY = load_registry(Path(registry_path))


def conformer_worker(source: SourceRow) -> dict[str, object]:
    registry = _WORKER_REGISTRY
    if registry is None:
        raise RuntimeError("worker registry was not initialized")

    row = base_row(source)
    mol = Chem.MolFromSmiles(source.smiles)
    if mol is None:
        return {**row, "reject_reason": "invalid_smiles"}

    canonical = canonical_smiles(mol)
    common = {
        "canonical_smiles": canonical,
        "is_valid_mol": True,
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "molecular_weight": float(Descriptors.MolWt(mol)),
    }

    matches = reaction_role_matches(mol, registry)
    if not matches:
        return {**row, **common, "reject_reason": "no_compatible_reaction_role"}

    embedded, failure = embed_molecule(mol)
    compatible_reactions = sorted({m.reaction_id for m in matches})
    compatible_roles = [f"{m.reaction_id}:{m.role_name}" for m in matches]
    if embedded is None:
        return {
            **row,
            **common,
            "compatible_reactions_json": json.dumps(compatible_reactions),
            "compatible_reaction_roles_json": json.dumps(compatible_roles),
            "reject_reason": failure or "failed_conformer",
            "ingest_status": "failed_conformer",
        }

    first_match = matches[0]
    embedded_no_h = Chem.RemoveHs(embedded)
    return {
        **row,
        "canonical_smiles": canonical_smiles(embedded),
        "is_valid_mol": True,
        "compatible_reactions_json": json.dumps(compatible_reactions),
        "compatible_reaction_roles_json": json.dumps(compatible_roles),
        "reaction_match_atoms_json": json.dumps(
            [
                {
                    "reaction_id": m.reaction_id,
                    "role_name": m.role_name,
                    "reactive_atom_idx": m.reactive_atom_idx,
                    "reference_atom_idx": m.reference_atom_idx,
                    "atom_map_to_atom_idx": m.atom_map_to_atom_idx,
                    "multi_match_enumeration_required": len(matches) > 1,
                }
                for m in matches
            ],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "leaving_group_atoms_json": json.dumps(
            [
                {
                    "reaction_id": m.reaction_id,
                    "role_name": m.role_name,
                    "leaving_group_atom_indices": m.leaving_group_atom_indices,
                }
                for m in matches
            ],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "conformer_atoms_json": conformer_atoms_json(embedded),
        "formal_charge": int(Chem.GetFormalCharge(embedded_no_h)),
        "heavy_atom_count": int(embedded_no_h.GetNumHeavyAtoms()),
        "molecular_weight": float(Descriptors.MolWt(embedded_no_h)),
        "sa_score": sa_score_if_available(embedded),
        "ingest_status": "pending_nagl_charge",
        "reject_reason": "",
        "reaction_rule_id": first_match.reaction_id,
        "reaction_role": first_match.role_name,
        "reaction_tags_json": json.dumps(compatible_roles),
        "attachment_atom_idx": first_match.reactive_atom_idx,
        "leaving_group_atom_idx": first_match.leaving_group_atom_indices[0]
        if first_match.leaving_group_atom_indices
        else None,
        "dihedral_reference_atom_idx": first_match.reference_atom_idx,
        "attachment_vector_json": attachment_vector_json(embedded, first_match),
        "leaving_group_formal_charge": 0.0,
        "conformer_count": 1,
        "_mol_block": Chem.MolToMolBlock(embedded),
    }


def iter_batches(indices: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def rdkit_to_pyg_data(mol: Chem.Mol) -> Any:
    import torch
    from torch_geometric.data import Data

    x = torch.tensor(
        [
            [
                float(atom.GetAtomicNum()),
                float(atom.GetFormalCharge()),
                float(atom.GetTotalDegree()),
                float(atom.GetIsAromatic()),
            ]
            for atom in mol.GetAtoms()
        ],
        dtype=torch.float32,
    )
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        edges.append((begin, end))
        edges.append((end, begin))
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def charge_rows_with_nagl(
    rows: list[dict[str, object]],
    batch_size: int,
    charge_model: str,
    allow_cpu_nagl: bool,
) -> dict[str, Any]:
    import torch
    from openff.nagl import GNNModel
    from openff.nagl.molecule._graph.molecule import GraphMolecule, NXMolHeteroGraph
    from openff.nagl_models._dynamic_fetch import get_model
    from openff.toolkit import Molecule
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
    from torch_geometric.data import Batch

    cuda_available = bool(torch.cuda.is_available())
    cuda_device = str(torch.cuda.get_device_name(0)) if cuda_available else ""
    if not cuda_available and not allow_cpu_nagl:
        raise RuntimeError("OpenFF NAGL requested, but torch.cuda.is_available() is false")

    pending = [idx for idx, row in enumerate(rows) if row["ingest_status"] == "pending_nagl_charge"]
    wrapper = NAGLToolkitWrapper()
    if charge_model not in wrapper.supported_charge_methods:
        raise RuntimeError(f"NAGL model {charge_model!r} is not supported by this OpenFF NAGL install")
    model_path = get_model(filename=charge_model)
    nagl_model = GNNModel.load(model_path, eval_mode=True)

    print(
        "nagl_backend_initialized "
        f"model={charge_model} model_path={model_path} torch_cuda_available={cuda_available} "
        f"torch_cuda_device={json.dumps(cuda_device)} pyg_batching=true batch_size={batch_size}",
        flush=True,
    )

    charge_methods = Counter()
    failures = Counter()
    charged = 0
    charge_start = time.perf_counter()
    for batch_index, batch_indices in enumerate(iter_batches(pending, batch_size), start=1):
        mols: list[tuple[int, Chem.Mol]] = []
        for idx in batch_indices:
            mol_block = str(rows[idx]["_mol_block"])
            mol = Chem.MolFromMolBlock(mol_block, removeHs=False, sanitize=True)
            if mol is None:
                rows[idx]["ingest_status"] = "failed_charge"
                rows[idx]["reject_reason"] = "failed_molblock_roundtrip"
                failures["failed_molblock_roundtrip"] += 1
                continue
            mols.append((idx, mol))

        pyg_batch = Batch.from_data_list([rdkit_to_pyg_data(mol) for _, mol in mols])
        pyg_device = "cpu"
        if cuda_available:
            pyg_batch = pyg_batch.to("cuda")
            torch.cuda.synchronize()
            pyg_device = "cuda"
        print(
            "pyg_dendritic_batch_ready "
            f"batch_index={batch_index} molecules={len(mols)} nodes={pyg_batch.num_nodes} "
            f"edges={pyg_batch.num_edges} device={pyg_device}",
            flush=True,
        )

        print(
            "nagl_charge_batch_start "
            f"batch_index={batch_index} molecules={len(mols)} model={charge_model}",
            flush=True,
        )
        batch_start = time.perf_counter()
        off_mols: list[tuple[int, Chem.Mol, Molecule]] = []
        for idx, mol in mols:
            try:
                off_mol = Molecule.from_rdkit(
                    mol,
                    allow_undefined_stereo=True,
                    hydrogens_are_explicit=True,
                )
            except Exception as exc:  # noqa: BLE001 - molecule-specific NAGL failure should not kill the batch.
                rows[idx]["ingest_status"] = "failed_charge"
                rows[idx]["reject_reason"] = f"failed_openff_conversion:{type(exc).__name__}:{exc}"
                failures["failed_openff_conversion"] += 1
                continue
            off_mols.append((idx, mol, off_mol))

        try:
            graphs = [
                NXMolHeteroGraph.from_openff(
                    off_mol,
                    atom_features=nagl_model.config.atom_features,
                    bond_features=nagl_model.config.bond_features,
                )
                for _, _, off_mol in off_mols
            ]
            batch_graph = NXMolHeteroGraph._batch(graphs)
            graph_molecule = GraphMolecule(graph=batch_graph, n_representations=1, mapped_smiles="batch")
            raw_tensor = nagl_model.forward(graph_molecule)["am1bcc_charges"]
            raw_values = [float(value) for value in raw_tensor.detach().cpu().numpy().flatten()]
        except Exception as exc:  # noqa: BLE001 - batch-level backend failure should stop the run.
            raise RuntimeError(f"batched NAGL forward failed: {type(exc).__name__}: {exc}") from exc

        offset = 0
        for idx, mol, off_mol in off_mols:
            atom_count = int(off_mol.n_atoms)
            charges = raw_values[offset : offset + atom_count]
            offset += atom_count
            if len(charges) != atom_count:
                rows[idx]["ingest_status"] = "failed_charge"
                rows[idx]["reject_reason"] = "failed_nagl_charge:charge_count_mismatch"
                failures["failed_nagl_charge"] += 1
                continue
            formal_charge = float(Chem.GetFormalCharge(Chem.RemoveHs(mol)))
            correction = (formal_charge - sum(charges)) / float(len(charges))
            charges = [value + correction for value in charges]
            rows[idx]["ingest_status"] = "ok"
            rows[idx]["charge_method"] = f"openff_nagl:{charge_model}"
            rows[idx]["partial_charge_method"] = f"openff_nagl:{charge_model}"
            rows[idx]["partial_charges_json"] = json.dumps(charges, separators=(",", ":"))
            charge_methods[f"openff_nagl:{charge_model}"] += 1
            charged += 1
        if offset != len(raw_values):
            raise RuntimeError(
                f"batched NAGL returned {len(raw_values)} charges but consumed {offset}; batch graph is inconsistent"
            )
        print(
            "nagl_charge_batch_complete "
            f"batch_index={batch_index} charged={charged} batch_seconds={time.perf_counter() - batch_start:.3f}",
            flush=True,
        )

    return {
        "nagl_charge_seconds": time.perf_counter() - charge_start,
        "charge_method_counts": dict(charge_methods),
        "charge_failure_counts": dict(failures),
        "torch_cuda_available": cuda_available,
        "torch_cuda_device": cuda_device,
        "nagl_model": charge_model,
        "nagl_batch_size": batch_size,
    }


def build_rows_parallel(
    sources: Sequence[SourceRow],
    registry_path: Path,
    processes: int,
    chunksize: int,
) -> tuple[list[dict[str, object]], float]:
    print(
        "hpc_multiprocessing_pool_start "
        f"processes={processes} chunksize={chunksize} molecules={len(sources)}",
        flush=True,
    )
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    completed = 0
    with mp.Pool(
        processes=processes,
        initializer=init_worker,
        initargs=(str(registry_path),),
    ) as pool:
        for row in pool.imap_unordered(conformer_worker, sources, chunksize=chunksize):
            rows.append(row)
            completed += 1
            if completed % 10_000 == 0 or completed == len(sources):
                print(
                    "hpc_conformer_generation_progress "
                    f"completed={completed} total={len(sources)} elapsed_seconds={time.perf_counter() - started:.3f}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    print(
        "hpc_conformer_generation_complete "
        f"completed={completed} total={len(sources)} elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return rows, elapsed


def reject_reason_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return dict(Counter(str(row.get("reject_reason") or "accepted") for row in rows))


def build_report(rows: Sequence[Mapping[str, object]], conformer_seconds: float, charge_report: Mapping[str, Any]) -> dict[str, Any]:
    per_reaction = Counter()
    for row in rows:
        if row.get("ingest_status") != "ok":
            continue
        for reaction_id in cast(list[str], json.loads(str(row.get("compatible_reactions_json") or "[]"))):
            per_reaction[reaction_id] += 1
    total = len(rows)
    conformer_success = sum(1 for row in rows if row["ingest_status"] in {"pending_nagl_charge", "ok", "failed_charge"})
    return {
        "total_input": total,
        "valid_mol_count": sum(1 for row in rows if bool(row["is_valid_mol"])),
        "compatible_synthon_count": sum(1 for row in rows if row["ingest_status"] == "ok"),
        "per_reaction_match_counts": dict(per_reaction),
        "conformer_success_rate": conformer_success / max(total, 1),
        "conformer_generation_seconds": conformer_seconds,
        "charge_method_counts": dict(charge_report.get("charge_method_counts", {})),
        "charge_failure_counts": dict(charge_report.get("charge_failure_counts", {})),
        "reject_reason_counts": reject_reason_counts(rows),
        "hpc": {
            "processes": None,
            "cpu_conformer_pool": True,
            "openff_nagl": True,
            "pyg_batching": True,
            **dict(charge_report),
        },
    }


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(payload)
    tmp.replace(path)


def atomic_write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    stripped = [{key: value for key, value in row.items() if key != "_mol_block"} for row in rows]
    pl.DataFrame(stripped).write_parquet(tmp, compression="zstd")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    processes = int(args.processes)
    if processes <= 0:
        raise ValueError("--processes must be positive")
    batch_size = int(args.nagl_batch_size)
    if batch_size <= 0:
        raise ValueError("--nagl-batch-size must be positive")

    registry_path = cast(Path, args.reaction_registry)
    load_registry(registry_path)
    sources = source_rows(cast(Path, args.input), cast(int | None, args.limit))
    if not sources:
        raise ValueError("zero input synthons")

    rows, conformer_seconds = build_rows_parallel(
        sources=sources,
        registry_path=registry_path,
        processes=processes,
        chunksize=int(args.cpu_chunksize),
    )
    charge_report = charge_rows_with_nagl(
        rows=rows,
        batch_size=batch_size,
        charge_model=str(args.nagl_model),
        allow_cpu_nagl=bool(args.allow_cpu_nagl),
    )
    report = build_report(rows, conformer_seconds, charge_report)
    report["hpc"]["processes"] = processes

    valid = int(report["valid_mol_count"])
    compatible = int(report["compatible_synthon_count"])
    if valid == 0:
        raise ValueError("zero valid synthons")
    if compatible == 0:
        raise ValueError("zero compatible synthons")

    output = cast(Path, args.output)
    atomic_write_parquet(output, rows)
    report_path = cast(Path, args.report)
    atomic_write_text(report_path, json.dumps(report, indent=2, sort_keys=True))
    print(
        "vspace_synthons_ingested "
        f"input={args.input} output={output} total={len(sources)} compatible={compatible} "
        f"charge_methods={json.dumps(report['charge_method_counts'], sort_keys=True)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
