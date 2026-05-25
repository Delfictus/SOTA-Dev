#!/usr/bin/env python3
"""Single-shot Track A trial harness over the first 10k finalized anchors."""

from __future__ import annotations

import asyncio
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import polars as pl
import torch
from torch import Tensor

from prism_dstw.hierarchical_bayes.gflownet_policy import AnchorAttentionGFlowNetPolicy
from prism_dstw.orchestration.campaign_dispatcher import AsyncCampaignDispatcher, CampaignTask
from prism_dstw.orchestration.topology_compiler import PrismTopologyCompiler, TopologyCompileResult
from prism_dstw.persistence.cloudflare_client import CloudflareManifoldClient, JsonValue


REPO_ROOT = Path(__file__).resolve().parents[1]
TRIAL_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/trial_run"
TRIAL_ANCHORS = TRIAL_DIR / "10k_trial_anchors_3d.parquet"
TRIAL_TOPOLOGY_DIR = TRIAL_DIR / "topologies"
TRIAL_OUTPUT_ROOT = TRIAL_DIR / "mvp_campaigns"
TRIAL_EXTRACT_DIR = TRIAL_DIR / "extracted"
BASE_RECEPTOR_TOPOLOGY = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/"
    "20260518T031002Z/04_TOPOLOGIES/glp1r_6XOX_WT.topology.json"
)
RUNTIME_ENV = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/"
    "20260518T031002Z/02_RUNTIME_CONFIG/glp1r_runtime.env"
)
MVP_STREAMS = ((0, "ThermalShock"), (5, "UvAromatic"), (10, "Equilibrium"), (15, "Hysteresis"))
EMBEDDING_DIM = 8


Point3D = tuple[float, float, float]
JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SelectedAnchor:
    row_index: int
    anchor_id: str
    smiles: str
    status: str
    coordinates: list[Point3D]
    charges: list[float]
    policy_logit: float


@dataclass(frozen=True)
class TrialTelemetry:
    selected_anchor: SelectedAnchor
    compiled: TopologyCompileResult
    engine_exit_code: int
    extraction_exit_code: int
    upload_key: str
    upload_part_count: int
    upload_id: str


def _rdkit_modules() -> tuple[Any, Any]:
    chem = cast(Any, import_module("rdkit.Chem"))
    all_chem = cast(Any, import_module("rdkit.Chem.AllChem"))
    return chem, all_chem


def _read_json_object(path: Path) -> JsonObject:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(JsonObject, loaded)


def _float_list(value: object, *, label: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must decode to a list")
    return [float(item) for item in value]


def _point_list(value: object, *, label: str) -> list[Point3D]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must decode to a list")
    points: list[Point3D] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 3:
            raise TypeError(f"{label} entries must be xyz lists")
        points.append((float(item[0]), float(item[1]), float(item[2])))
    return points


def _distance(left: Point3D, right: Point3D) -> float:
    return math.sqrt(
        (left[0] - right[0]) * (left[0] - right[0])
        + (left[1] - right[1]) * (left[1] - right[1])
        + (left[2] - right[2]) * (left[2] - right[2])
    )


def _norm(vector: Point3D) -> float:
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def _add(left: Point3D, right: Point3D) -> Point3D:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _subtract(left: Point3D, right: Point3D) -> Point3D:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _scale(vector: Point3D, factor: float) -> Point3D:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _dot(left: Point3D, right: Point3D) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _centroid(points: Sequence[Point3D]) -> Point3D:
    count = float(len(points))
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def _topology_points_and_elements(topology_path: Path) -> tuple[list[Point3D], list[str]]:
    topology = _read_json_object(topology_path)
    positions_raw = topology.get("positions")
    elements_raw = topology.get("elements")
    if not isinstance(positions_raw, list) or len(positions_raw) % 3 != 0:
        raise ValueError(f"{topology_path}: positions must be a flat xyz list")
    if not isinstance(elements_raw, list):
        raise ValueError(f"{topology_path}: elements must be a list")
    positions = [float(value) for value in positions_raw]
    points = [(positions[index], positions[index + 1], positions[index + 2]) for index in range(0, len(positions), 3)]
    elements = [str(value) for value in elements_raw]
    if len(points) != len(elements):
        raise ValueError(f"{topology_path}: positions/elements length mismatch")
    return points, elements


def _mol_points(mol: Any) -> list[Point3D]:
    conformer = mol.GetConformer(0)
    points: list[Point3D] = []
    for atom_index in range(int(mol.GetNumAtoms())):
        position = conformer.GetAtomPosition(atom_index)
        points.append((float(position.x), float(position.y), float(position.z)))
    return points


def _translate_mol(mol: Any, delta: Point3D) -> None:
    conformer = mol.GetConformer(0)
    for atom_index in range(int(mol.GetNumAtoms())):
        position = conformer.GetAtomPosition(atom_index)
        conformer.SetAtomPosition(
            atom_index,
            (float(position.x) + delta[0], float(position.y) + delta[1], float(position.z) + delta[2]),
        )


def _minimum_heavy_distance(
    receptor_points: Sequence[Point3D],
    receptor_elements: Sequence[str],
    ligand_points: Sequence[Point3D],
    ligand_elements: Sequence[str],
) -> float:
    receptor_heavy = [index for index, element in enumerate(receptor_elements) if element.upper() not in {"H", "D"}]
    ligand_heavy = [index for index, element in enumerate(ligand_elements) if element.upper() not in {"H", "D"}]
    minimum = math.inf
    for receptor_index in receptor_heavy:
        receptor_point = receptor_points[receptor_index]
        for ligand_index in ligand_heavy:
            minimum = min(minimum, _distance(receptor_point, ligand_points[ligand_index]))
    return minimum


class TrialPrismTopologyCompiler(PrismTopologyCompiler):
    """Trial compiler using precomputed 10k-anchor charges without mutating production compiler."""

    def _assign_openff_parameters(self, mol: Any) -> tuple[Any, list[float], str]:
        if bool(mol.HasProp("trial_precomputed_charges_json")):
            decoded = json.loads(str(mol.GetProp("trial_precomputed_charges_json")))
            charges = _float_list(decoded, label="trial_precomputed_charges_json")
            if len(charges) == int(mol.GetNumAtoms()):
                for atom_index, charge in enumerate(charges):
                    atom = mol.GetAtomWithIdx(atom_index)
                    atom.SetDoubleProp("AM1BCCCharge", charge)
                    atom.SetProp("am1bcc_charge", f"{charge:.12f}")
                return mol, charges, "trial_precomputed_am1bcc_from_10k_action_space+MMFF94s_minimized"

        _, all_chem = _rdkit_modules()
        all_chem.ComputeGasteigerCharges(mol)
        fallback_charges: list[float] = []
        for atom in mol.GetAtoms():
            raw = str(atom.GetProp("_GasteigerCharge")) if bool(atom.HasProp("_GasteigerCharge")) else "0.0"
            charge = 0.0 if raw.lower() in {"nan", "inf", "-inf"} else float(raw)
            atom.SetDoubleProp("AM1BCCCharge", charge)
            atom.SetProp("am1bcc_charge", f"{charge:.12f}")
            fallback_charges.append(charge)
        return mol, fallback_charges, "trial_rdkit_gasteiger_fallback+MMFF94s_minimized"


class MVPTrialDispatcher(AsyncCampaignDispatcher):
    """Dedicated 1-replica MVP dispatcher with direct engine flags."""

    def launch_script_path(self, generated_id: str) -> Path:
        return self.launch_dir / "launch-mvp-trial.sh"

    def expected_output_dir(self, generated_id: str) -> Path:
        safe_generated_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in generated_id)
        return self.output_root / safe_generated_id / "05_RESULTS"

    def write_launch_script(self, compiled: TopologyCompileResult) -> Path:
        output_dir = self.expected_output_dir(compiled.generated_id)
        script = self.launch_script_path(compiled.generated_id)
        stream_label = ",".join(f"{stream_id}:{name}" for stream_id, name in MVP_STREAMS)
        text = f"""#!/usr/bin/env bash
set -euo pipefail

# Generated by scripts/run_track_a_trial_harness.py.
# MVP logical stream plan: {stream_label}

REPO_ROOT="${{PRISM_REPO_ROOT:-{REPO_ROOT.as_posix()}}}"
RUNTIME_ENV="${{PRISM_RUNTIME_ENV:-{RUNTIME_ENV.as_posix()}}}"
CONDITION_ID="{compiled.condition_id}"
TOPOLOGY_JSON="{compiled.topology_path.as_posix()}"
OUTPUT_ROOT="{output_dir.as_posix()}"
REPLICA_ID=0

if [[ -f "$RUNTIME_ENV" ]]; then
  source "$RUNTIME_ENV"
fi

export PRISM_CAMPAIGN_ID="track_a_trial_{compiled.generated_id}"
export PRISM_REPLICAS=1
export PRISM_STREAMS_PER_REPLICA=4
export PRISM_GLP1R_OUTPUT_ROOT="$OUTPUT_ROOT"
export PRISM_GLP1R_RAW_ARCHIVE_AFTER_REPLICA=0
export PRISM_GLP1R_REQUIRE_R2_PREFLIGHT=0
export PRISM_GLP1R_COMPACT_ARCHIVE_AFTER_MATERIALIZE=0
export PRISM_GLP1R_PRUNE_AFTER_REMOTE_VERIFY=0
export PRISM_VALIDATED=1

if [[ ! -x "${{PRISM_ENGINE_BIN:-}}" ]]; then
  echo "missing executable PRISM_ENGINE_BIN: ${{PRISM_ENGINE_BIN:-unset}}" >&2
  exit 2
fi

if [[ ! -f "$TOPOLOGY_JSON" ]]; then
  echo "missing topology: $TOPOLOGY_JSON" >&2
  exit 2
fi

seed="${{PRISM_BASE_SEED:-42}}"
out_dir="$OUTPUT_ROOT/$CONDITION_ID/replica_${{REPLICA_ID}}"
log_dir="{(TRIAL_DIR / "logs").as_posix()}/$CONDITION_ID"
mkdir -p "$out_dir" "$log_dir"

cmd=(
  "$PRISM_ENGINE_BIN"
  -t "$TOPOLOGY_JSON"
  -o "$out_dir"
  --replica-seed "$seed"
  --ensemble-campaign-id "$PRISM_CAMPAIGN_ID"
  --ensemble-base-seed "${{PRISM_BASE_SEED:-42}}"
  --ensemble-replica-id "$REPLICA_ID"
  --fast-25k
  --hysteresis
  --prism-therm
  --multi-stream 4
  --multi-differential
  --spike-percentile 70
  --fused-steps 6
  --hmr
  --adaptive-dt
  --closed-loop-steering
  --asymmetric-steering
  --site-ranker phase-manifold
  --md-only-evidence
  --path-a-production-profile
  --path-a-max-wall-seconds "${{PRISM_TRIAL_MAX_WALL_SECONDS:-120}}"
  --path-a-chunk-size "${{PRISM_TRIAL_CHUNK_SIZE:-50}}"
  --save-trajectory-interval 1
  --uv-wavelengths 280,274,258,254,211
  --nma-amplification "${{PRISM_NMA_AMPLIFICATION:-3.0}}"
  --nma-scan-fraction "${{PRISM_NMA_SCAN_FRACTION:-0.3}}"
)

echo "condition_id=$CONDITION_ID"
echo "topology_json=$TOPOLOGY_JSON"
echo "output_root=$OUTPUT_ROOT"
echo "logical_stream_plan={stream_label}"
printf 'command:'
printf ' %q' "${{cmd[@]}}"
echo

"${{cmd[@]}}" 2>&1 | tee "$log_dir/replica_${{REPLICA_ID}}.run.log"

manifest="$out_dir/md_evidence_manifest.json"
if [[ ! -f "$manifest" ]]; then
  echo "missing md_evidence_manifest.json for $CONDITION_ID replica $REPLICA_ID" >&2
  exit 1
fi
echo "$manifest" > "$log_dir/replica_${{REPLICA_ID}}.manifest.path"
echo "$manifest"
"""
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(text, encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return script


class TrackATrialHarness:
    """Single-shot Generate -> Compile -> Simulate -> Extract -> Upload trial."""

    def __init__(
        self,
        *,
        action_space_path: Path = TRIAL_ANCHORS,
        worker_url: str = "http://127.0.0.1:8787",
    ) -> None:
        self.action_space_path = action_space_path
        self.worker_url = worker_url
        self.compiler = TrialPrismTopologyCompiler(
            base_receptor_topology=BASE_RECEPTOR_TOPOLOGY,
            output_dir=TRIAL_TOPOLOGY_DIR,
        )
        self.dispatcher = MVPTrialDispatcher(
            launch_dir=REPO_ROOT / "bin",
            output_root=TRIAL_OUTPUT_ROOT,
            runtime_env=RUNTIME_ENV,
            max_parallel=1,
        )

    def load_action_space(self) -> pl.DataFrame:
        if not self.action_space_path.is_file():
            raise FileNotFoundError(
                f"{self.action_space_path} is missing; run scripts/materialize_10k_trial_space.py first"
            )
        frame = pl.read_parquet(self.action_space_path)
        required = {"anchor_id", "smiles", "status", "coordinates_json", "am1bcc_charges_json"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"trial action space missing required columns: {missing}")
        successful = frame.filter(pl.col("status") == "success")
        if successful.is_empty():
            raise ValueError("trial action space contains no successful anchors")
        return successful

    def _anchor_features(self, rows: Sequence[Mapping[str, object]]) -> Tensor:
        chem, _ = _rdkit_modules()
        features: list[list[float]] = []
        for row in rows:
            coordinates = _point_list(json.loads(str(row["coordinates_json"])), label="coordinates_json")
            charges = _float_list(json.loads(str(row["am1bcc_charges_json"])), label="am1bcc_charges_json")
            xs = [point[0] for point in coordinates]
            ys = [point[1] for point in coordinates]
            zs = [point[2] for point in coordinates]
            centroid = _centroid(coordinates)
            radius = max(_distance(point, centroid) for point in coordinates)
            mol = chem.MolFromSmiles(str(row["smiles"]))
            heavy_atoms = float(mol.GetNumHeavyAtoms()) if mol is not None else 0.0
            atom_count = float(len(coordinates))
            charge_abs_mean = sum(abs(charge) for charge in charges) / max(float(len(charges)), 1.0)
            features.append(
                [
                    atom_count,
                    heavy_atoms,
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                    max(zs) - min(zs),
                    radius,
                    sum(charges),
                    charge_abs_mean,
                ]
            )
        tensor = torch.tensor(features, dtype=torch.float32)
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True).clamp_min(1.0e-6)
        return (tensor - mean) / std

    def select_anchor(self, frame: pl.DataFrame) -> SelectedAnchor:
        rows = cast(list[dict[str, object]], frame.to_dicts())
        embeddings = self._anchor_features(rows)
        torch.manual_seed(20260523)
        policy = AnchorAttentionGFlowNetPolicy(
            node_feature_dim=EMBEDDING_DIM,
            anchor_embeddings=embeddings,
            hidden_dim=64,
            max_backward_actions=1,
        )
        policy.eval()
        state_nodes = torch.stack(
            [
                embeddings.mean(dim=0),
                embeddings.std(dim=0),
                embeddings.max(dim=0).values,
            ],
            dim=0,
        ).unsqueeze(0)
        node_mask = torch.ones((1, state_nodes.shape[1]), dtype=torch.bool)
        forward_mask = torch.ones((1, embeddings.shape[0]), dtype=torch.bool)
        backward_mask = torch.ones((1, 1), dtype=torch.bool)
        with torch.no_grad():
            output = policy(state_nodes, node_mask, forward_mask, backward_mask)
            ranked_indices = torch.argsort(output.forward_logits[0], descending=True).tolist()

        for row_index in ranked_indices:
            row = rows[int(row_index)]
            coordinates = _point_list(json.loads(str(row["coordinates_json"])), label="coordinates_json")
            charges = _float_list(json.loads(str(row["am1bcc_charges_json"])), label="am1bcc_charges_json")
            smiles = str(row["smiles"])
            if self._mol_from_anchor(smiles, coordinates, charges) is not None:
                return SelectedAnchor(
                    row_index=int(row_index),
                    anchor_id=str(row["anchor_id"]),
                    smiles=smiles,
                    status=str(row["status"]),
                    coordinates=coordinates,
                    charges=charges,
                    policy_logit=float(output.forward_logits[0, int(row_index)].item()),
                )
        raise RuntimeError("GFlowNet policy did not rank any RDKit-parseable trial anchor")

    def _mol_from_anchor(self, smiles: str, coordinates: Sequence[Point3D], charges: Sequence[float]) -> Any | None:
        chem, _ = _rdkit_modules()
        base = chem.MolFromSmiles(smiles)
        if base is None:
            return None
        mol = chem.AddHs(base)
        if int(mol.GetNumAtoms()) != len(coordinates):
            return None
        conformer = chem.Conformer(int(mol.GetNumAtoms()))
        for atom_index, point in enumerate(coordinates):
            conformer.SetAtomPosition(atom_index, point)
        mol.RemoveAllConformers()
        mol.AddConformer(conformer, assignId=True)
        mol.SetProp("trial_precomputed_charges_json", json.dumps([float(charge) for charge in charges]))
        for atom_index, charge in enumerate(charges):
            atom = mol.GetAtomWithIdx(atom_index)
            atom.SetDoubleProp("AM1BCCCharge", float(charge))
            atom.SetProp("am1bcc_charge", f"{float(charge):.12f}")
        return mol

    def _place_molecule_for_trial(self, mol: Any) -> None:
        receptor_points, receptor_elements = _topology_points_and_elements(BASE_RECEPTOR_TOPOLOGY)
        receptor_center = _centroid(receptor_points)
        ligand_elements = [str(atom.GetSymbol()) for atom in mol.GetAtoms()]
        ligand_heavy = [index for index, element in enumerate(ligand_elements) if element.upper() not in {"H", "D"}]
        if not ligand_heavy:
            raise ValueError("selected trial anchor contains no heavy atoms")

        surface_candidates = sorted(
            (
                (index, _distance(point, receptor_center))
                for index, (point, element) in enumerate(zip(receptor_points, receptor_elements, strict=True))
                if element.upper() not in {"H", "D"}
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:256]
        offsets = (1.8, 2.0, 2.3, 2.7, 3.2)
        best_delta: Point3D | None = None
        best_distance = 0.0
        original_points = _mol_points(mol)
        for receptor_index, _ in surface_candidates:
            receptor_point = receptor_points[receptor_index]
            normal_raw = _subtract(receptor_point, receptor_center)
            normal_length = max(_norm(normal_raw), 1.0e-9)
            normal = _scale(normal_raw, 1.0 / normal_length)
            ligand_points = original_points
            projections = [(_dot(ligand_points[index], normal), index) for index in ligand_heavy]
            _, ligand_index = min(projections, key=lambda item: item[0])
            for offset in offsets:
                target = _add(receptor_point, _scale(normal, offset))
                delta = _subtract(target, ligand_points[ligand_index])
                translated_points = [_add(point, delta) for point in ligand_points]
                min_distance = _minimum_heavy_distance(
                    receptor_points,
                    receptor_elements,
                    translated_points,
                    ligand_elements,
                )
                if 1.55 <= min_distance <= 3.8:
                    _translate_mol(mol, delta)
                    return
                if min_distance > best_distance:
                    best_distance = min_distance
                    best_delta = delta
        if best_delta is None:
            raise RuntimeError("could not place selected anchor near receptor surface")
        _translate_mol(mol, best_delta)

    def compile_anchor(self, selected: SelectedAnchor) -> TopologyCompileResult:
        mol = self._mol_from_anchor(selected.smiles, selected.coordinates, selected.charges)
        if mol is None:
            raise RuntimeError(f"selected anchor is not RDKit-constructible: {selected.smiles}")
        self._place_molecule_for_trial(mol)
        return self.compiler.compile_molecule(
            mol,
            generated_id=f"TRIAL_{selected.anchor_id}",
            metadata={
                "source": "TrackATrialHarness",
                "source_anchor_id": selected.anchor_id,
                "source_smiles": selected.smiles,
                "policy_logit": selected.policy_logit,
                "action_space_path": self.action_space_path.as_posix(),
            },
        )

    async def simulate(self, compiled: TopologyCompileResult) -> tuple[CampaignTask, int]:
        task = await self.dispatcher.dispatch(compiled)
        exit_code = await task.process.wait()
        return task, int(exit_code)

    async def extract(self, task: CampaignTask) -> int:
        TRIAL_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PRISM_RESULTS_DIR"] = task.expected_output_dir.as_posix()
        env["PRISM_EXTRACT_OUTPUT_DIR"] = TRIAL_EXTRACT_DIR.as_posix()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "scripts/scratch/topo_regenerate.py",
            cwd=REPO_ROOT,
            env=env,
        )
        return int(await process.wait())

    def upload_artifact(self, task: CampaignTask, compiled: TopologyCompileResult) -> Path:
        candidates = sorted(task.expected_output_dir.rglob("Chem_Perturbed_DTSG.parquet"))
        if candidates:
            return candidates[0]
        parquet_outputs = sorted(task.expected_output_dir.rglob("*.parquet"))
        if parquet_outputs:
            return parquet_outputs[0]
        manifest_paths = sorted(task.expected_output_dir.rglob("md_evidence_manifest.json"))
        if manifest_paths:
            summary_path = TRIAL_EXTRACT_DIR / "track_a_trial_manifest_summary.parquet"
            TRIAL_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "generated_id": [compiled.generated_id],
                    "condition_id": [compiled.condition_id],
                    "manifest_path": [manifest_paths[0].as_posix()],
                    "epistemic_class": ["OBSERVED_MANIFEST_SUMMARY"],
                }
            ).write_parquet(summary_path)
            return summary_path
        raise FileNotFoundError(f"no parquet or md_evidence_manifest.json found under {task.expected_output_dir}")

    async def upload_to_cloudflare(self, artifact: Path, selected: SelectedAnchor) -> tuple[str, int, str]:
        metadata: dict[str, JsonValue] = {
            "campaign_id": "glp1r_aleniglipron",
            "trial_harness": True,
            "anchor_id": selected.anchor_id,
            "smiles": selected.smiles,
        }
        object_key = f"track_a_trial/{artifact.name}"
        async with CloudflareManifoldClient(
            base_url=self.worker_url,
            access_client_id="dev",
            access_client_secret="dev",
        ) as client:
            result = await client.upload_tensor_multipart(
                artifact,
                key=object_key,
                chunk_size=8 * 1024 * 1024,
                concurrency=4,
                metadata=metadata,
            )
        return result["key"], int(result["part_count"]), result["upload_id"]

    async def run_once(self) -> TrialTelemetry:
        frame = self.load_action_space()
        selected = self.select_anchor(frame)
        print(f"GFLOWNET_CHOSEN_ANCHOR_ID={selected.anchor_id}", flush=True)
        print(f"GFLOWNET_CHOSEN_SMILES={selected.smiles}", flush=True)
        print(f"GFLOWNET_POLICY_LOGIT={selected.policy_logit:.8f}", flush=True)
        compiled = self.compile_anchor(selected)
        print(f"TOPOLOGY_PATH={compiled.topology_path.as_posix()}", flush=True)
        print(f"TOPOLOGY_MIN_HEAVY_DISTANCE_A={compiled.min_heavy_distance_A:.3f}", flush=True)
        task, engine_exit_code = await self.simulate(compiled)
        print(f"ENGINE_EXIT_CODE={engine_exit_code}", flush=True)
        if engine_exit_code != 0:
            raise RuntimeError(f"PRISM-4D MVP engine run failed with exit code {engine_exit_code}")
        extraction_exit_code = await self.extract(task)
        print(f"EXTRACTION_EXIT_CODE={extraction_exit_code}", flush=True)
        if extraction_exit_code != 0:
            raise RuntimeError(f"trial extraction DAG failed with exit code {extraction_exit_code}")
        artifact = self.upload_artifact(task, compiled)
        print(f"UPLOAD_ARTIFACT={artifact.as_posix()}", flush=True)
        upload_key, upload_part_count, upload_id = await self.upload_to_cloudflare(artifact, selected)
        print(
            f"CLOUDFLARE_MPU_UPLOAD=success key={upload_key} parts={upload_part_count} upload_id={upload_id}",
            flush=True,
        )
        return TrialTelemetry(
            selected_anchor=selected,
            compiled=compiled,
            engine_exit_code=engine_exit_code,
            extraction_exit_code=extraction_exit_code,
            upload_key=upload_key,
            upload_part_count=upload_part_count,
            upload_id=upload_id,
        )


def main() -> int:
    telemetry = asyncio.run(TrackATrialHarness().run_once())
    print(
        "TRACK_A_TRIAL_COMPLETE "
        f"anchor_id={telemetry.selected_anchor.anchor_id} "
        f"engine_exit_code={telemetry.engine_exit_code} "
        f"upload_key={telemetry.upload_key}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
