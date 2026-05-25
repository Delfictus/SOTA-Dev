#!/usr/bin/env python3
"""Autonomous Track A closed-loop dispatcher state machine.

This script is infrastructure only until explicitly executed. It consumes
GFlowNet-exported candidate molecules, compiles PRISM-4D topologies, launches
active-learning campaigns, triggers extraction, uploads observed tensors to R2,
and then returns to candidate generation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from prism_dstw.orchestration.campaign_dispatcher import AsyncCampaignDispatcher, CampaignTask
from prism_dstw.orchestration.topology_compiler import PrismTopologyCompiler, TopologyCompileResult
from prism_dstw.persistence.cloudflare_client import CloudflareManifoldClient, JsonValue


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_QUEUE = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/gflownet_generated_candidates.jsonl"
DEFAULT_DTSG_NAME = "Chem_Perturbed_DTSG.parquet"


class DaemonState(str, Enum):
    GENERATE = "GENERATE"
    COMPILE = "COMPILE"
    SIMULATE = "SIMULATE"
    EXTRACT = "EXTRACT"
    LEARN = "LEARN"


@dataclass(frozen=True)
class GeneratedCandidate:
    generated_id: str
    smiles: str
    reward: float
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class CompiledCandidate:
    candidate: GeneratedCandidate
    topology: TopologyCompileResult


@dataclass(frozen=True)
class CompletedCampaign:
    compiled: CompiledCandidate
    task: CampaignTask
    results_dir: Path


def _rdkit_chem() -> Any:
    return cast(Any, import_module("rdkit.Chem"))


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _float_value(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError(f"{label} must be numeric")


class GFlowNetCandidateProvider:
    """Read GFlowNet-generated molecules from an append-only JSONL queue."""

    def __init__(self, queue_path: Path, poll_interval_seconds: float) -> None:
        self.queue_path = queue_path
        self.poll_interval_seconds = poll_interval_seconds
        self._offset = 0

    def _parse_candidate(self, line: str) -> GeneratedCandidate:
        payload = _json_object(json.loads(line), "gflownet candidate")
        generated_id = str(payload["generated_id"])
        smiles = str(payload["smiles"])
        reward = _float_value(payload.get("reward", 0.0), "reward")
        metadata = _json_object(payload.get("metadata", {}), "candidate metadata")
        return GeneratedCandidate(generated_id=generated_id, smiles=smiles, reward=reward, metadata=metadata)

    async def next_batch(self, batch_size: int) -> list[GeneratedCandidate]:
        while True:
            if self.queue_path.exists():
                lines = self.queue_path.read_text(encoding="utf-8").splitlines()
                unread = [line for line in lines[self._offset :] if line.strip()]
                if unread:
                    selected = unread[:batch_size]
                    self._offset += len(selected)
                    return [self._parse_candidate(line) for line in selected]
            await asyncio.sleep(self.poll_interval_seconds)


class AutonomousTrackADaemon:
    """Closed-loop Track A state machine."""

    def __init__(
        self,
        *,
        candidate_provider: GFlowNetCandidateProvider,
        topology_compiler: PrismTopologyCompiler,
        dispatcher: AsyncCampaignDispatcher,
        cloud_client: CloudflareManifoldClient,
        batch_size: int,
        dtsg_name: str = DEFAULT_DTSG_NAME,
    ) -> None:
        self.candidate_provider = candidate_provider
        self.topology_compiler = topology_compiler
        self.dispatcher = dispatcher
        self.cloud_client = cloud_client
        self.batch_size = batch_size
        self.dtsg_name = dtsg_name

    async def state_generate(self) -> list[GeneratedCandidate]:
        sys.stdout.write(f"STATE {DaemonState.GENERATE.value}: querying GFlowNet candidate provider\n")
        return await self.candidate_provider.next_batch(self.batch_size)

    async def state_compile(self, candidates: Sequence[GeneratedCandidate]) -> list[CompiledCandidate]:
        sys.stdout.write(f"STATE {DaemonState.COMPILE.value}: compiling {len(candidates)} candidate topology file(s)\n")
        chem = _rdkit_chem()
        compiled: list[CompiledCandidate] = []
        for candidate in candidates:
            mol = chem.MolFromSmiles(candidate.smiles)
            if mol is None:
                raise ValueError(f"RDKit failed to parse generated candidate {candidate.generated_id}")
            topology = await asyncio.to_thread(
                self.topology_compiler.compile_molecule,
                mol,
                generated_id=candidate.generated_id,
                metadata={"reward": candidate.reward, **dict(candidate.metadata)},
            )
            compiled.append(CompiledCandidate(candidate=candidate, topology=topology))
        return compiled

    async def state_simulate(self, compiled: Sequence[CompiledCandidate]) -> list[CompletedCampaign]:
        sys.stdout.write(f"STATE {DaemonState.SIMULATE.value}: dispatching {len(compiled)} PRISM-4D campaign(s)\n")
        completed: list[CompletedCampaign] = []
        for item in compiled:
            task = await self.dispatcher.dispatch(item.topology)
            exit_code = await task.process.wait()
            if exit_code != 0:
                raise RuntimeError(
                    f"PRISM-4D campaign failed generated_id={item.candidate.generated_id} "
                    f"pid={task.pid} exit_code={exit_code}"
                )
            completed.append(CompletedCampaign(compiled=item, task=task, results_dir=task.expected_output_dir))
        return completed

    async def state_extract(self, campaigns: Sequence[CompletedCampaign]) -> list[CompletedCampaign]:
        sys.stdout.write(f"STATE {DaemonState.EXTRACT.value}: triggering extraction DAG for {len(campaigns)} campaign(s)\n")
        for campaign in campaigns:
            env = os.environ.copy()
            env["PRISM_RESULTS_DIR"] = campaign.results_dir.as_posix()
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "scripts/scratch/topo_regenerate.py",
                cwd=REPO_ROOT,
                env=env,
            )
            exit_code = await process.wait()
            if exit_code != 0:
                raise RuntimeError(
                    f"extraction DAG failed generated_id={campaign.compiled.candidate.generated_id} "
                    f"results_dir={campaign.results_dir} exit_code={exit_code}"
                )
        return list(campaigns)

    def _find_observed_dtsg(self, results_dir: Path) -> Path:
        candidates = sorted(results_dir.rglob(self.dtsg_name))
        if not candidates:
            raise FileNotFoundError(f"{self.dtsg_name} not found under {results_dir}")
        return candidates[0]

    async def state_learn(self, campaigns: Sequence[CompletedCampaign]) -> None:
        sys.stdout.write(f"STATE {DaemonState.LEARN.value}: uploading observed tensors and triggering posterior update\n")
        for campaign in campaigns:
            dtsg_path = self._find_observed_dtsg(campaign.results_dir)
            object_key = (
                "track_a_active_learning/"
                f"{campaign.compiled.candidate.generated_id}/"
                f"{dtsg_path.name}"
            )
            await self.cloud_client.upload_tensor_multipart(
                dtsg_path,
                key=object_key,
                content_type="application/vnd.apache.parquet",
                metadata={
                    "generated_id": campaign.compiled.candidate.generated_id,
                    "condition_id": campaign.compiled.topology.condition_id,
                    "epistemic_transition": "HYPOTHESIZED_TO_OBSERVED",
                },
            )
            await self.trigger_dkl_posterior_update(campaign.compiled.candidate.generated_id, object_key)

    async def trigger_dkl_posterior_update(self, generated_id: str, object_key: str) -> None:
        sys.stdout.write(f"DKL_POSTERIOR_UPDATE generated_id={generated_id} object_key={object_key}\n")

    async def run_forever(self) -> None:
        while True:
            candidates = await self.state_generate()
            compiled = await self.state_compile(candidates)
            campaigns = await self.state_simulate(compiled)
            extracted = await self.state_extract(campaigns)
            await self.state_learn(extracted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-queue", type=Path, default=DEFAULT_CANDIDATE_QUEUE)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--base-receptor-topology", type=Path, default=PrismTopologyCompiler().base_receptor_topology)
    parser.add_argument("--topology-output-dir", type=Path, default=PrismTopologyCompiler().output_dir)
    parser.add_argument("--campaign-output-root", type=Path, default=AsyncCampaignDispatcher().output_root)
    parser.add_argument("--max-parallel", type=int, default=1)
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    provider = GFlowNetCandidateProvider(Path(args.candidate_queue), float(args.poll_interval_seconds))
    compiler = PrismTopologyCompiler(
        base_receptor_topology=Path(args.base_receptor_topology),
        output_dir=Path(args.topology_output_dir),
    )
    dispatcher = AsyncCampaignDispatcher(
        output_root=Path(args.campaign_output_root),
        max_parallel=int(args.max_parallel),
    )
    async with CloudflareManifoldClient() as cloud_client:
        daemon = AutonomousTrackADaemon(
            candidate_provider=provider,
            topology_compiler=compiler,
            dispatcher=dispatcher,
            cloud_client=cloud_client,
            batch_size=int(args.batch_size),
        )
        await daemon.run_forever()


def main() -> int:
    asyncio.run(async_main(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
