#!/usr/bin/env python3
"""Ingest Phase 2C PDB snapshots into a dynamic alignment reference frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "glp1r_aleniglipron"
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
TRACK_A_DIR = CAMPAIGN_DIR / "track_a_generative"
DEFAULT_RISK_MAP = N80_DIR / "receptor_durability_risk_map.parquet"
DEFAULT_TRIGGERS = CAMPAIGN_DIR / "phase_2c_snapshot_triggers.json"
DEFAULT_SNAPSHOT_ROOT = CAMPAIGN_DIR / "phase_2c_snapshots"
DEFAULT_OUTPUT = TRACK_A_DIR / "dynamic_alignment_reference.json"
DEFAULT_PARITY_RECORD = CAMPAIGN_DIR / "phase_2c_reintegration_parity.json"
DEFAULT_LEDGER = TRACK_A_DIR / "dynamic_alignment_reference.propagation.jsonl"
DEFAULT_LEDGER_RATIONALE = (
    "Original 80-replica campaign lacked checkpoints. De Novo capture executed to provide Cartesian alignment anchor. "
    "Thermodynamic scoring remains anchored to the original 80-replica variance field."
)
DEFAULT_CONDITION = "glp1r_6XOX_WT"
DEFAULT_CHAIN = "A"
POCKET_EDGE_CLASS = "pocket_vector"
POCKET_EDGE_COUNT = 4
JsonObject: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class CaAtom:
    topology_residue_idx: int
    pdb_residue_number: int
    insertion_code: str
    residue_name: str
    chain_id: str
    x: float
    y: float
    z: float

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class HeavyAtom:
    atom_serial: int
    atom_name: str
    residue_name: str
    chain_id: str
    pdb_residue_number: int
    insertion_code: str
    element: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class SnapshotSelection:
    condition_id: str
    replica_id: int
    stream_id: int
    baseline_step: int
    metastable_step: int
    desensitized_step: int


@dataclass(frozen=True)
class SelectedPdbModel:
    lines: list[str]
    timestep: int | None
    model_index: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-map", type=Path, default=DEFAULT_RISK_MAP)
    parser.add_argument("--triggers", type=Path, default=DEFAULT_TRIGGERS)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parity-record", type=Path, default=DEFAULT_PARITY_RECORD)
    parser.add_argument("--ledger-output", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--epistemic-class", default="REPRESENTATIVE_CAPTURE")
    parser.add_argument("--ledger-rationale", default=DEFAULT_LEDGER_RATIONALE)
    parser.add_argument("--trajectory-file", action="append", type=Path, default=[])
    parser.add_argument("--condition-id", default=DEFAULT_CONDITION)
    parser.add_argument("--replica-id", type=int, default=None)
    parser.add_argument("--stream-id", type=int, default=None)
    parser.add_argument("--chain-id", default=DEFAULT_CHAIN)
    parser.add_argument("--metastable-step", type=int, default=None)
    parser.add_argument("--baseline-step", type=int, default=None)
    parser.add_argument("--desensitized-step", type=int, default=None)
    parser.add_argument("--metastable-pdb", type=Path, default=None)
    parser.add_argument("--baseline-pdb", type=Path, default=None)
    parser.add_argument("--desensitized-pdb", type=Path, default=None)
    return parser.parse_args()


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def as_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got bool")
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"{label} must be an integer")


def as_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric, got bool")
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"{label} must be numeric")


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_json(path: Path) -> JsonObject:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not decode to an object")
    return cast(JsonObject, loaded)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def top_pocket_edges(path: Path, condition_id: str) -> list[JsonObject]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("condition_id") == condition_id)
        .filter(pl.col("edge_class") == POCKET_EDGE_CLASS)
        .select(
            [
                "condition_id",
                "edge_from_residue",
                "edge_to_residue",
                "durability_risk_score_raw",
                "durability_risk_percentile",
                "validation_status",
            ]
        )
        .unique(subset=["condition_id", "edge_from_residue", "edge_to_residue"])
        .sort(
            ["durability_risk_percentile", "durability_risk_score_raw", "edge_from_residue", "edge_to_residue"],
            descending=[True, True, False, False],
        )
        .head(POCKET_EDGE_COUNT)
        .collect()
    )
    if frame.height != POCKET_EDGE_COUNT:
        raise ValueError(f"{condition_id}: expected {POCKET_EDGE_COUNT} pocket-vector edges, found {frame.height}")
    return cast(list[JsonObject], frame.to_dicts())


def pocket_residue_indices(edges: list[JsonObject]) -> list[int]:
    residues: set[int] = set()
    for edge in edges:
        residues.add(as_int(edge["edge_from_residue"], "edge_from_residue"))
        residues.add(as_int(edge["edge_to_residue"], "edge_to_residue"))
    return sorted(residues)


def window_step(windows: list[object], rationale_token: str) -> int:
    for raw_window in windows:
        if not isinstance(raw_window, dict):
            continue
        rationale = str(raw_window.get("rationale", ""))
        if rationale_token in rationale:
            start_step = as_int(raw_window["start_step"], "start_step")
            end_step = as_int(raw_window["end_step"], "end_step")
            if start_step == end_step:
                return start_step
            if start_step <= 6005 <= end_step:
                return 6005
            return (start_step + end_step) // 2
    raise ValueError(f"trigger did not contain window matching {rationale_token!r}")


def choose_snapshot_selection(args: argparse.Namespace) -> SnapshotSelection:
    payload = load_json(args.triggers)
    triggers = json_list(payload.get("triggers"), "triggers")
    condition_id = str(args.condition_id)
    for raw_trigger in triggers:
        if not isinstance(raw_trigger, dict):
            continue
        if raw_trigger.get("condition_id") != condition_id:
            continue
        replica_id = as_int(raw_trigger["replica_id"], "replica_id")
        stream_id = as_int(raw_trigger["stream_id"], "stream_id")
        if args.replica_id is not None and replica_id != int(args.replica_id):
            continue
        if args.stream_id is not None and stream_id != int(args.stream_id):
            continue
        windows = json_list(raw_trigger.get("windows"), "windows")
        return SnapshotSelection(
            condition_id=condition_id,
            replica_id=replica_id,
            stream_id=stream_id,
            baseline_step=int(args.baseline_step) if args.baseline_step is not None else window_step(windows, "Baseline"),
            metastable_step=int(args.metastable_step) if args.metastable_step is not None else window_step(windows, "Rupture"),
            desensitized_step=int(args.desensitized_step)
            if args.desensitized_step is not None
            else window_step(windows, "Desensitized"),
        )
    raise ValueError(f"no trigger matched condition={condition_id} replica={args.replica_id} stream={args.stream_id}")


def pdb_match_score(path: Path, selection: SnapshotSelection, step: int) -> int:
    lowered = path.as_posix().lower()
    condition = selection.condition_id.lower()
    score = 0
    if condition in lowered:
        score += 50
    if re.search(rf"replica[_-]?0*{selection.replica_id}(?:\D|$)", lowered):
        score += 20
    if re.search(rf"stream[_-]?0*{selection.stream_id}(?:\D|$)", lowered):
        score += 20
    if re.search(rf"(?:step|mdstep|md_step|frame)[_-]?0*{step}(?:\D|$)", lowered):
        score += 30
    elif re.search(rf"(?:\D|^)0*{step}(?:\D|$)", lowered):
        score += 10
    return score


def discover_pdb(snapshot_root: Path, selection: SnapshotSelection, step: int) -> Path:
    if not snapshot_root.exists():
        raise FileNotFoundError(f"snapshot root does not exist yet: {snapshot_root}")
    candidates = [path for path in snapshot_root.rglob("*.pdb") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no PDB snapshots found under {snapshot_root}")
    ranked = sorted(
        ((pdb_match_score(path, selection, step), path) for path in candidates),
        key=lambda item: (-item[0], item[1].as_posix()),
    )
    best_score, best_path = ranked[0]
    if best_score < 80:
        raise FileNotFoundError(
            f"no PDB matched condition={selection.condition_id} replica={selection.replica_id} "
            f"stream={selection.stream_id} step={step}; best={best_path} score={best_score}"
        )
    return best_path


def resolve_pdbs(args: argparse.Namespace, selection: SnapshotSelection) -> tuple[Path, Path, Path]:
    baseline = args.baseline_pdb or discover_pdb(args.snapshot_root, selection, selection.baseline_step)
    metastable = args.metastable_pdb or discover_pdb(args.snapshot_root, selection, selection.metastable_step)
    desensitized = args.desensitized_pdb or discover_pdb(args.snapshot_root, selection, selection.desensitized_step)
    return (Path(baseline), Path(metastable), Path(desensitized))


def select_pdb_model(path: Path, target_step: int | None) -> SelectedPdbModel:
    fallback_lines: list[str] = []
    has_model = False
    current_lines: list[str] = []
    current_timestep: int | None = None
    current_model_index: int | None = None
    best_lines: list[str] | None = None
    best_timestep: int | None = None
    best_model_index: int | None = None
    best_distance: int | None = None
    best_model_index_distance: int | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if not has_model:
                fallback_lines.append(line)
                if len(fallback_lines) > 200_000:
                    fallback_lines = fallback_lines[-1_000:]
            if not line.startswith("MODEL") and not current_lines:
                continue

            if line.startswith("MODEL"):
                has_model = True
                current_lines = [line]
                current_timestep = None
                model_token = line[5:].strip()
                current_model_index = int(model_token) if model_token.isdigit() else None
                continue
            if current_lines:
                current_lines.append(line)
                if line.startswith("REMARK") and "TIMESTEP" in line:
                    tokens = line.split()
                    if tokens and tokens[-1].lstrip("-").isdigit():
                        current_timestep = int(tokens[-1])
                if line.startswith("ENDMDL"):
                    if target_step is None:
                        best_lines = current_lines.copy()
                        best_timestep = current_timestep
                        best_model_index = current_model_index
                    elif current_timestep is not None:
                        distance = abs(current_timestep - target_step)
                        if distance == 0:
                            return SelectedPdbModel(
                                lines=current_lines.copy(),
                                timestep=current_timestep,
                                model_index=current_model_index,
                            )
                        if best_distance is None or distance < best_distance:
                            best_lines = current_lines.copy()
                            best_timestep = current_timestep
                            best_model_index = current_model_index
                            best_distance = distance
                    elif current_model_index is not None:
                        distance = abs(current_model_index - target_step)
                        if best_model_index_distance is None or distance < best_model_index_distance:
                            best_lines = current_lines.copy()
                            best_timestep = current_timestep
                            best_model_index = current_model_index
                            best_model_index_distance = distance
                    current_lines = []
                    current_timestep = None
                    current_model_index = None
    if best_lines is not None:
        return SelectedPdbModel(lines=best_lines, timestep=best_timestep, model_index=best_model_index)
    if current_lines:
        return SelectedPdbModel(lines=current_lines, timestep=current_timestep, model_index=current_model_index)
    return SelectedPdbModel(lines=fallback_lines, timestep=None, model_index=None)


def parse_pdb_ca_atoms(path: Path, chain_id: str, target_step: int | None = None) -> list[CaAtom]:
    atoms: list[CaAtom] = []
    selected = select_pdb_model(path, target_step)
    for line in selected.lines:
        record = line[0:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        atom_name = line[12:16].strip()
        alternate_location = line[16:17].strip()
        chain = line[21:22].strip() or "_"
        if atom_name != "CA" or chain != chain_id or alternate_location not in {"", "A"}:
            continue
        atoms.append(
            CaAtom(
                topology_residue_idx=len(atoms),
                pdb_residue_number=int(line[22:26]),
                insertion_code=line[26:27].strip(),
                residue_name=line[17:20].strip(),
                chain_id=chain,
                x=float(line[30:38]),
                y=float(line[38:46]),
                z=float(line[46:54]),
            )
        )
    if not atoms:
        raise ValueError(f"{path}: no C-alpha atoms found for chain {chain_id}")
    return atoms


def parse_pdb_heavy_atoms(path: Path, chain_id: str, target_step: int | None = None) -> list[HeavyAtom]:
    atoms: list[HeavyAtom] = []
    selected = select_pdb_model(path, target_step)
    for line in selected.lines:
        record = line[0:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        alternate_location = line[16:17].strip()
        chain = line[21:22].strip() or "_"
        if chain != chain_id or alternate_location not in {"", "A"}:
            continue
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or re.sub(r"[^A-Za-z]", "", atom_name)[:1]).upper()
        if element in {"H", "D"} or atom_name.upper().startswith(("H", "D")):
            continue
        atoms.append(
            HeavyAtom(
                atom_serial=int(line[6:11]),
                atom_name=atom_name,
                residue_name=line[17:20].strip(),
                chain_id=chain,
                pdb_residue_number=int(line[22:26]),
                insertion_code=line[26:27].strip(),
                element=element,
                x=float(line[30:38]),
                y=float(line[38:46]),
                z=float(line[46:54]),
            )
        )
    if not atoms:
        raise ValueError(f"{path}: no heavy atoms found for chain {chain_id}")
    return atoms


def atoms_by_residue(atoms: list[CaAtom], residue_indices: list[int], path: Path) -> dict[int, CaAtom]:
    by_index = {atom.topology_residue_idx: atom for atom in atoms}
    missing = [residue_idx for residue_idx in residue_indices if residue_idx not in by_index]
    if missing:
        raise ValueError(f"{path}: missing topology residue indices {missing}")
    return {residue_idx: by_index[residue_idx] for residue_idx in residue_indices}


def coordinate_rows(atoms: dict[int, CaAtom], residue_indices: list[int]) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for residue_idx in residue_indices:
        atom = atoms[residue_idx]
        rows.append(
            {
                "residue_idx": residue_idx,
                "pdb_residue_number": atom.pdb_residue_number,
                "insertion_code": atom.insertion_code,
                "residue_name": atom.residue_name,
                "chain_id": atom.chain_id,
                "ca_xyz": [atom.x, atom.y, atom.z],
            }
        )
    return rows


def heavy_atom_coordinate_rows(
    heavy_atoms: list[HeavyAtom],
    ca_atoms: dict[int, CaAtom],
    residue_indices: list[int],
) -> list[JsonObject]:
    residue_keys = {
        residue_idx: (
            ca_atoms[residue_idx].chain_id,
            ca_atoms[residue_idx].pdb_residue_number,
            ca_atoms[residue_idx].insertion_code,
        )
        for residue_idx in residue_indices
    }
    topology_idx_by_pdb_key = {pdb_key: residue_idx for residue_idx, pdb_key in residue_keys.items()}
    rows: list[JsonObject] = []
    for atom in heavy_atoms:
        residue_idx = topology_idx_by_pdb_key.get((atom.chain_id, atom.pdb_residue_number, atom.insertion_code))
        if residue_idx is None:
            continue
        rows.append(
            {
                "residue_idx": residue_idx,
                "pdb_residue_number": atom.pdb_residue_number,
                "insertion_code": atom.insertion_code,
                "residue_name": atom.residue_name,
                "chain_id": atom.chain_id,
                "atom_serial": atom.atom_serial,
                "atom_name": atom.atom_name,
                "element": atom.element,
                "xyz": [atom.x, atom.y, atom.z],
            }
        )
    if not rows:
        raise ValueError("no heavy atoms matched the requested pocket residues")
    return rows


def euclidean_delta(start_atom: CaAtom, end_atom: CaAtom) -> float:
    dx = end_atom.x - start_atom.x
    dy = end_atom.y - start_atom.y
    dz = end_atom.z - start_atom.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def deformation_delta_rows(
    start_atoms: dict[int, CaAtom],
    end_atoms: dict[int, CaAtom],
    residue_indices: list[int],
) -> tuple[list[JsonObject], float, float]:
    rows: list[JsonObject] = []
    squared_sum = 0.0
    max_delta = 0.0
    for residue_idx in residue_indices:
        start_atom = start_atoms[residue_idx]
        end_atom = end_atoms[residue_idx]
        delta = euclidean_delta(start_atom, end_atom)
        max_delta = max(max_delta, delta)
        squared_sum += delta * delta
        rows.append(
            {
                "residue_idx": residue_idx,
                "delta_r_A": delta,
                "start_ca_xyz": [start_atom.x, start_atom.y, start_atom.z],
                "end_ca_xyz": [end_atom.x, end_atom.y, end_atom.z],
            }
        )
    rms_delta = math.sqrt(squared_sum / max(len(residue_indices), 1))
    return rows, rms_delta, max_delta


def edge_rows(edges: list[JsonObject], atoms: dict[int, CaAtom]) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for edge_rank, edge in enumerate(edges, start=1):
        from_idx = as_int(edge["edge_from_residue"], "edge_from_residue")
        to_idx = as_int(edge["edge_to_residue"], "edge_to_residue")
        from_atom = atoms[from_idx]
        to_atom = atoms[to_idx]
        rows.append(
            {
                "edge_rank": edge_rank,
                "edge_from_residue": from_idx,
                "edge_to_residue": to_idx,
                "durability_risk_score_raw": as_float(edge["durability_risk_score_raw"], "durability_risk_score_raw"),
                "durability_risk_percentile": as_float(edge["durability_risk_percentile"], "durability_risk_percentile"),
                "from_ca_xyz": [from_atom.x, from_atom.y, from_atom.z],
                "to_ca_xyz": [to_atom.x, to_atom.y, to_atom.z],
            }
        )
    return rows


def matching_checkpoint_records(parity_record: Path, selection: SnapshotSelection) -> list[JsonObject]:
    if not parity_record.exists():
        return []
    payload = load_json(parity_record)
    commands = json_list(payload.get("commands"), "commands")
    records: list[JsonObject] = []
    for raw_command in commands:
        if not isinstance(raw_command, dict):
            continue
        if raw_command.get("condition_id") != selection.condition_id:
            continue
        if as_int(raw_command.get("replica_id"), "replica_id") != selection.replica_id:
            continue
        raw_checkpoints = raw_command.get("checkpoint_files", [])
        if not isinstance(raw_checkpoints, list):
            continue
        for raw_checkpoint in raw_checkpoints:
            if isinstance(raw_checkpoint, dict):
                records.append(cast(JsonObject, raw_checkpoint))
    return records


def write_ledger(
    *,
    path: Path,
    reference_output: Path,
    selection: SnapshotSelection,
    source_pdbs: tuple[Path, Path, Path],
    checkpoint_records: list[JsonObject],
    trajectory_files: list[Path],
    epistemic_class: str,
    rationale: str,
) -> None:
    baseline_pdb, metastable_pdb, desensitized_pdb = source_pdbs
    payload: JsonObject = {
        "event": "dynamic_alignment_reference_ingested",
        "campaign_id": CAMPAIGN_ID,
        "epistemic_class": epistemic_class,
        "rationale": rationale,
        "condition_id": selection.condition_id,
        "replica_id": selection.replica_id,
        "stream_id": selection.stream_id,
        "reference_output": relative_path(reference_output),
        "reference_output_sha256": sha256_file(reference_output),
        "source_pdbs": {
            "baseline": {"path": relative_path(baseline_pdb), "sha256": sha256_file(baseline_pdb)},
            "metastable": {"path": relative_path(metastable_pdb), "sha256": sha256_file(metastable_pdb)},
            "desensitized": {"path": relative_path(desensitized_pdb), "sha256": sha256_file(desensitized_pdb)},
        },
        "checkpoint_files": checkpoint_records,
        "trajectory_files": [
            {"path": relative_path(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in trajectory_files
            if path.exists()
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def build_reference(args: argparse.Namespace) -> JsonObject:
    selection = choose_snapshot_selection(args)
    baseline_pdb, metastable_pdb, desensitized_pdb = resolve_pdbs(args, selection)
    for path in (baseline_pdb, metastable_pdb, desensitized_pdb):
        if not path.exists():
            raise FileNotFoundError(path)

    pocket_edges = top_pocket_edges(args.risk_map, selection.condition_id)
    residue_indices = pocket_residue_indices(pocket_edges)
    baseline_atoms = atoms_by_residue(
        parse_pdb_ca_atoms(baseline_pdb, args.chain_id, selection.baseline_step),
        residue_indices,
        baseline_pdb,
    )
    metastable_atoms = atoms_by_residue(
        parse_pdb_ca_atoms(metastable_pdb, args.chain_id, selection.metastable_step),
        residue_indices,
        metastable_pdb,
    )
    desensitized_atoms = atoms_by_residue(
        parse_pdb_ca_atoms(desensitized_pdb, args.chain_id, selection.desensitized_step), residue_indices, desensitized_pdb
    )
    baseline_to_metastable, baseline_rms, baseline_max = deformation_delta_rows(
        baseline_atoms, metastable_atoms, residue_indices
    )
    metastable_to_desensitized, desensitized_rms, desensitized_max = deformation_delta_rows(
        metastable_atoms, desensitized_atoms, residue_indices
    )
    heavy_atom_rows = heavy_atom_coordinate_rows(
        parse_pdb_heavy_atoms(metastable_pdb, args.chain_id, selection.metastable_step),
        metastable_atoms,
        residue_indices,
    )
    selected_metastable = select_pdb_model(metastable_pdb, selection.metastable_step)
    coordinate_array = [[metastable_atoms[idx].x, metastable_atoms[idx].y, metastable_atoms[idx].z] for idx in residue_indices]
    return {
        "campaign_id": CAMPAIGN_ID,
        "condition_id": selection.condition_id,
        "replica_id": selection.replica_id,
        "stream_id": selection.stream_id,
        "md_step": selection.metastable_step,
        "metastable_md_step": selection.metastable_step,
        "selected_metastable_timestep": selected_metastable.timestep,
        "selected_metastable_model_index": selected_metastable.model_index,
        "baseline_md_step": selection.baseline_step,
        "desensitized_md_step": selection.desensitized_step,
        "chain_id": args.chain_id,
        "coordinate_frame": "topology_order_heavy_atom_residue_mapped",
        "epistemic_class": args.epistemic_class,
        "epistemic_rationale": args.ledger_rationale,
        "source_pdbs": {
            "baseline": relative_path(baseline_pdb),
            "metastable": relative_path(metastable_pdb),
            "desensitized": relative_path(desensitized_pdb),
        },
        "source_pdb_sha256": {
            "baseline": sha256_file(baseline_pdb),
            "metastable": sha256_file(metastable_pdb),
            "desensitized": sha256_file(desensitized_pdb),
        },
        "pocket_vector_edges": edge_rows(pocket_edges, metastable_atoms),
        "pocket_residue_indices": residue_indices,
        "pocket_vector_coordinates": coordinate_array,
        "pocket_residue_coordinates": coordinate_rows(metastable_atoms, residue_indices),
        "pocket_heavy_atom_coordinates": heavy_atom_rows,
        "heavy_atom_count": len(heavy_atom_rows),
        "deformation_deltas": {
            "baseline_to_metastable": {
                "rms_delta_A": baseline_rms,
                "max_delta_A": baseline_max,
                "per_residue": baseline_to_metastable,
            },
            "metastable_to_desensitized": {
                "rms_delta_A": desensitized_rms,
                "max_delta_A": desensitized_max,
                "per_residue": metastable_to_desensitized,
            },
        },
    }


def main() -> int:
    args = parse_args()
    selection = choose_snapshot_selection(args)
    reference = build_reference(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reference, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    baseline_pdb, metastable_pdb, desensitized_pdb = resolve_pdbs(args, selection)
    checkpoint_records = matching_checkpoint_records(Path(args.parity_record), selection)
    write_ledger(
        path=Path(args.ledger_output),
        reference_output=output,
        selection=selection,
        source_pdbs=(baseline_pdb, metastable_pdb, desensitized_pdb),
        checkpoint_records=checkpoint_records,
        trajectory_files=[Path(path) for path in args.trajectory_file],
        epistemic_class=str(args.epistemic_class),
        rationale=str(args.ledger_rationale),
    )
    emit(
        f"wrote {output.relative_to(REPO_ROOT)} condition={reference['condition_id']} "
        f"md_step={reference['md_step']} residues={len(cast(list[object], reference['pocket_residue_indices']))} "
        f"heavy_atoms={reference['heavy_atom_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
