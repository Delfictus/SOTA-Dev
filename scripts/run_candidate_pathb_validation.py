#!/usr/bin/env python3
"""Run a generated GLP-1R candidate through the current Path-B validation chain.

Pipeline:
  candidate SDF -> OpenFF holo topology -> PRISM md-only evidence ->
  prism-materialize-sites -> dstw_export_wt_pathb -> validated summary JSON.

The PRISM wrapper can currently return non-zero after a successful md-only
handoff because its postflight still expects legacy binding_sites.json outputs.
This runner treats the engine stage as accepted only when the md evidence
manifest is present and validates required artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_BASE_TOPOLOGY = REPO_ROOT / "campaigns/glp1r_aleniglipron/topologies/glp1r_6XOX_WT.topology.json"
DEFAULT_RESIDUE_MAP = REPO_ROOT / "campaigns/glp1r_aleniglipron/topologies/glp1r_6XOX_WT.residue_map.json"
DEFAULT_OUTPUT_ROOT = Path("/mnt/storage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--base-receptor-topology", type=Path, default=DEFAULT_BASE_TOPOLOGY)
    parser.add_argument("--residue-map", type=Path, default=DEFAULT_RESIDUE_MAP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--topology-python",
        type=Path,
        default=Path(os.environ.get("PRISM_TOPOLOGY_PYTHON", sys.executable)),
        help="Python executable with OpenFF installed for holo topology compilation.",
    )
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument(
        "--replica-start-index",
        type=int,
        default=0,
        help="First replica index to run. Use with an existing --run-label to resume without rerunning replica 0.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a replica when its summary outputs already exist and validate.",
    )
    parser.add_argument("--replica-seed-base", type=int, default=42)
    parser.add_argument("--path-a-max-wall-seconds", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument("--materialized-top-n", type=int, default=5)
    parser.add_argument("--top-k-candidates", type=int, default=100)
    parser.add_argument("--target", default="GLP1R")
    parser.add_argument("--uniprot-accession", default="P43220")
    parser.add_argument("--pdb-anchor-active", default="6XOX")
    parser.add_argument("--pdb-anchor-inactive", default=None)
    parser.add_argument(
        "--nma-perturb",
        type=Path,
        default=None,
        help="Optional NMA modes JSON to pass to the engine. Defaults to the receptor sidecar when present.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_id = str(args.candidate_id)
    if not SAFE_ID.fullmatch(candidate_id):
        raise ValueError(f"unsafe candidate id: {candidate_id}")
    if int(args.replicas) < 1:
        raise ValueError("--replicas must be >= 1")
    if int(args.replica_start_index) < 0:
        raise ValueError("--replica-start-index must be >= 0")

    sdf = resolve_existing(args.sdf)
    base_topology = resolve_existing(args.base_receptor_topology)
    residue_map = resolve_existing(args.residue_map)
    topology_python = resolve_existing(args.topology_python)
    nma_perturb = resolve_nma_perturb(args.nma_perturb, base_topology)
    receptor_chain_id = infer_residue_map_chain(residue_map)
    receptor_anchor_id = topology_anchor_id(base_topology)
    holo_anchor_id = f"{receptor_anchor_id}_HOLO_{candidate_id}"
    label = args.run_label or f"{candidate_id}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_root = Path(args.output_root) / f"prism_candidate_pathb_{label}"
    topo_dir = run_root / "holo_topologies_openff"
    engine_root = run_root / "engine_openff"
    materialized_root = run_root / "materialized_openff"
    run_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    ptx_dir = REPO_ROOT / "target/ptx"
    env.setdefault("PRISM4D_PTX_DIR", str(ptx_dir))
    env.setdefault("PRISM_PTX_DIR", str(ptx_dir))
    topology_env = env.copy()
    topology_env["PATH"] = f"{topology_python.parent}:{env.get('PATH', '')}"
    topology_env.setdefault("CONDA_PREFIX", str(topology_python.parent.parent))

    summary_path = run_root / "candidate_pathb_validation_summary.json"
    existing_summary = load_existing_summary(summary_path)
    summary: dict[str, Any] = {
        "schema_version": "PRISM.candidate_pathb_validation.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "candidate_id": candidate_id,
        "sdf": str(sdf),
        "base_receptor_topology": str(base_topology),
        "receptor_anchor_id": receptor_anchor_id,
        "holo_anchor_id": holo_anchor_id,
        "residue_map": str(residue_map),
        "topology_python": str(topology_python),
        "nma_perturb": str(nma_perturb) if nma_perturb else None,
        "receptor_chain_id": receptor_chain_id,
        "run_root": str(run_root),
        "replicas_requested": int(args.replicas),
        "replica_start_index": int(args.replica_start_index),
        "skip_existing": bool(args.skip_existing),
        "replicas": existing_summary.get("replicas", []) if existing_summary else [],
    }

    compile_cmd = [
        str(topology_python),
        str(REPO_ROOT / "scripts/compile_candidate_holo_topology.py"),
        "--candidate-id",
        candidate_id,
        "--sdf",
        str(sdf),
        "--base-receptor-topology",
        str(base_topology),
        "--output-dir",
        str(topo_dir),
        "--condition-prefix",
        f"{receptor_anchor_id}_HOLO",
    ]
    if args.dry_run:
        summary["dry_run"] = True
        summary["compile_command"] = compile_cmd
        write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    holo_topology = topo_dir / f"{holo_anchor_id}.topology.json"
    if holo_topology.exists():
        compile_rc = 0
    else:
        compile_rc = run_logged(compile_cmd, run_root / "compile_holo_topology.log", env=topology_env)
        if compile_rc != 0:
            raise RuntimeError(f"holo topology compile failed with exit {compile_rc}")
    holo_info = inspect_topology(holo_topology)
    summary["holo_topology"] = {"path": str(holo_topology), **holo_info}

    replica_records: dict[int, dict[str, Any]] = {
        int(row["replica"]): row for row in summary.get("replicas", []) if isinstance(row, dict) and "replica" in row
    }
    for replica in range(int(args.replica_start_index), int(args.replica_start_index) + int(args.replicas)):
        seed = int(args.replica_seed_base) + replica * 1000
        replica_id = f"{candidate_id}_{receptor_anchor_id}_replica_{replica}"
        engine_dir = engine_root / replica_id
        materialized_dir = materialized_root / replica_id
        engine_dir.mkdir(parents=True, exist_ok=True)
        materialized_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_existing:
            existing = validate_existing_replica(replica, seed, engine_dir, materialized_dir)
            if existing is not None:
                replica_records[replica] = existing
                summary["replicas"] = [replica_records[index] for index in sorted(replica_records)]
                write_json(summary_path, summary)
                continue

        engine_cmd = [
            "timeout",
            str(int(args.timeout_seconds)),
            str(REPO_ROOT / "scripts/prism-validate-and-run.sh"),
            "-t",
            str(holo_topology),
            "-o",
            str(engine_dir),
            "--fast-25k",
            "--hysteresis",
            "--prism-therm",
            "--multi-stream",
            "8",
            "--spike-percentile",
            "70",
            "--fused-steps",
            "6",
            "--hmr",
            "--adaptive-dt",
            "--multi-differential",
            "--closed-loop-steering",
            "--asymmetric-steering",
            "--site-ranker",
            "phase-manifold",
            "--md-only-evidence",
            "--path-a-production-profile",
            "--path-a-max-wall-seconds",
            str(int(args.path_a_max_wall_seconds)),
            "--uv-wavelengths",
            "280,274,258,254,211",
            "--nma-amplification",
            "3.0",
            "--nma-scan-fraction",
            "0.3",
            "--replica-seed",
            str(seed),
            "-v",
        ]
        if nma_perturb is not None:
            engine_cmd.extend(["--nma-perturb", str(nma_perturb)])
        engine_rc = run_logged(engine_cmd, engine_dir / "engine.log", env=env)
        md_manifest = engine_dir / "md_evidence_manifest.json"
        md_status = validate_md_manifest(md_manifest)
        if not md_status["accepted"]:
            raise RuntimeError(f"engine did not produce accepted md evidence: {md_status}")

        materialize_cmd = [
            str(REPO_ROOT / "target/release/prism-materialize-sites"),
            "--manifest",
            str(md_manifest),
            "--output-dir",
            str(materialized_dir),
            "--top-k-candidates",
            str(int(args.top_k_candidates)),
            "--materialized-top-n",
            str(int(args.materialized_top_n)),
            "--verbose",
        ]
        materialize_rc = run_logged(materialize_cmd, materialized_dir / "prism_materialize_sites.log", env=env)
        if materialize_rc != 0:
            raise RuntimeError(f"materializer failed with exit {materialize_rc}")
        materialized = materialized_dir / "binding_sites.materialized.json"
        materialized_status = validate_materialized(materialized)
        if not materialized_status["accepted"]:
            raise RuntimeError(f"materialized sites invalid: {materialized_status}")

        export_cmd = [
            str(REPO_ROOT / "target/release/dstw_export_wt_pathb"),
            "--materialized-sites",
            str(materialized),
            "--residue-map",
            str(residue_map),
            "--topology-json",
            str(holo_topology),
            "--target",
            str(args.target),
            "--uniprot-accession",
            str(args.uniprot_accession),
            "--structure-anchor-id",
            holo_anchor_id,
            "--receptor-chain-id",
            receptor_chain_id,
            "--prism-run-id",
            replica_id,
            "--pdb-anchor-active",
            str(args.pdb_anchor_active),
            "--out-json",
            str(materialized_dir / "candidate_physical_profile.json"),
            "--out-parquet",
            str(materialized_dir / "candidate_physics_payload.parquet"),
            "--out-contact-parquet",
            str(materialized_dir / "candidate_contact_graph.parquet"),
        ]
        if args.pdb_anchor_inactive:
            export_cmd.extend(["--pdb-anchor-inactive", str(args.pdb_anchor_inactive)])
        export_rc = run_logged(export_cmd, materialized_dir / "dstw_export_candidate_pathb.log", env=env)
        if export_rc != 0:
            raise RuntimeError(f"DSTW export failed with exit {export_rc}")

        physics = parquet_status(materialized_dir / "candidate_physics_payload.parquet")
        contacts = parquet_status(materialized_dir / "candidate_contact_graph.parquet")
        if physics["rows"] <= 0 or contacts["rows"] <= 0:
            raise RuntimeError(f"empty DSTW parquet output: physics={physics}, contacts={contacts}")

        replica_records[replica] = {
            "replica": replica,
            "seed": seed,
            "engine_returncode": engine_rc,
            "engine_returncode_note": (
                "accepted via md evidence manifest; wrapper may fail legacy postflight" if engine_rc != 0 else "zero"
            ),
            "engine_dir": str(engine_dir),
            "materialized_dir": str(materialized_dir),
            "md_evidence": md_status,
            "materialized": materialized_status,
            "physics_parquet": physics,
            "contact_parquet": contacts,
        }
        summary["replicas"] = [replica_records[index] for index in sorted(replica_records)]
        write_json(summary_path, summary)

    summary["status"] = "PASS"
    summary["replicas_completed"] = len(summary.get("replicas", []))
    write_json(summary_path, summary)
    print(f"candidate_pathb_validation=PASS summary={summary_path}")
    return 0


def resolve_existing(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def infer_residue_map_chain(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    residues = data.get("residues")
    if not isinstance(residues, list):
        raise ValueError(f"{path}: residues must be a list")
    chains = sorted({str(row.get("chain")) for row in residues if isinstance(row, dict) and row.get("chain")})
    if len(chains) != 1:
        raise ValueError(f"{path}: expected exactly one receptor chain, got {chains}")
    return chains[0]


def topology_anchor_id(path: Path) -> str:
    name = path.name
    if name.endswith(".topology.json"):
        return name[: -len(".topology.json")]
    return path.stem


def resolve_nma_perturb(requested: Path | None, base_topology: Path) -> Path | None:
    if requested is not None:
        return resolve_existing(requested)
    name = base_topology.name
    if name.endswith(".topology.json"):
        candidate = base_topology.with_name(name[: -len(".topology.json")] + "_nma_modes.json")
        if candidate.exists():
            return candidate
    return None


def load_existing_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def inspect_topology(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "n_atoms": int(data.get("n_atoms", 0)),
        "n_residues": int(data.get("n_residues", 0)),
        "ligand_atom_count": int(data.get("ligand_atom_count", 0)),
        "ligand_charge_method": data.get("ligand_charge_method"),
        "collision_guard_min_heavy_atom_distance_A": data.get("collision_guard_min_heavy_atom_distance_A"),
    }


def validate_md_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"accepted": False, "path": str(path), "reason": "missing"}
    data = json.loads(path.read_text(encoding="utf-8"))
    accepted = (
        data.get("schema_kind") == "md_evidence_manifest"
        and data.get("required_artifacts_complete") is True
        and int(data.get("stream_count", 0)) > 0
        and int(data.get("streams_serialized", 0)) == int(data.get("stream_count", -1))
        and int(data.get("total_spikes_md", 0)) > 0
    )
    return {
        "accepted": accepted,
        "path": str(path),
        "schema_kind": data.get("schema_kind"),
        "target": data.get("target"),
        "run_id": data.get("run_id"),
        "stream_count": data.get("stream_count"),
        "streams_serialized": data.get("streams_serialized"),
        "total_spikes_md": data.get("total_spikes_md"),
        "required_artifacts_complete": data.get("required_artifacts_complete"),
        "validation_status": data.get("validation_status"),
    }


def validate_materialized(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"accepted": False, "path": str(path), "reason": "missing"}
    data = json.loads(path.read_text(encoding="utf-8"))
    sites = data.get("binding_sites")
    site_count = len(sites) if isinstance(sites, list) else 0
    accepted = data.get("schema_kind") == "pathb_binding_sites_materialized" and site_count > 0
    return {
        "accepted": accepted,
        "path": str(path),
        "schema_kind": data.get("schema_kind"),
        "schema_version": data.get("schema_version"),
        "target": data.get("target"),
        "run_id": data.get("run_id"),
        "binding_sites": site_count,
    }


def parquet_status(path: Path) -> dict[str, Any]:
    meta = pq.read_metadata(path)
    schema = pq.read_schema(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": meta.num_rows,
        "columns": schema.names,
    }


def validate_existing_replica(
    replica: int,
    seed: int,
    engine_dir: Path,
    materialized_dir: Path,
) -> dict[str, Any] | None:
    md_status = validate_md_manifest(engine_dir / "md_evidence_manifest.json")
    materialized_status = validate_materialized(materialized_dir / "binding_sites.materialized.json")
    physics_path = materialized_dir / "candidate_physics_payload.parquet"
    contacts_path = materialized_dir / "candidate_contact_graph.parquet"
    if not (
        md_status.get("accepted")
        and materialized_status.get("accepted")
        and physics_path.exists()
        and contacts_path.exists()
    ):
        return None
    physics = parquet_status(physics_path)
    contacts = parquet_status(contacts_path)
    if physics["rows"] <= 0 or contacts["rows"] <= 0:
        return None
    return {
        "replica": replica,
        "seed": seed,
        "engine_returncode": None,
        "engine_returncode_note": "skipped_existing_validated_outputs",
        "engine_dir": str(engine_dir),
        "materialized_dir": str(materialized_dir),
        "md_evidence": md_status,
        "materialized": materialized_status,
        "physics_parquet": physics,
        "contact_parquet": contacts,
        "skipped_existing": True,
    }


def run_logged(cmd: list[str], log_path: Path, *, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            print(line, end="")
        rc = proc.wait()
        log.write(f"\nEXIT={rc}\n")
    return rc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
