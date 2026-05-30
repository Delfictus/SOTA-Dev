#!/usr/bin/env python3
"""Verify the canonical single-env PRISM runtime using bounded real smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_ROOT = Path("/mnt/storage/prism_env_copies/prism_dock_portable_20260529")
DEFAULT_SCRATCH_ROOT = Path("/mnt/storage/prism-scratch/Prism4D-bio")
DEFAULT_REPORT_BASE = Path("/mnt/storage/tmp")
DEFAULT_REGISTRY_PATH = REPO_ROOT / "manifests/prism_canonical_runtime_executables.json"
PHASE_MANIFOLD_FIXTURE_DIR = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_phase_manifold_smoke_fixture"
)


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def git_head_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "nogit"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--report-root", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--skip-gflownet-train", action="store_true")
    return parser.parse_args()


def make_env(env_root: Path, scratch_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    env["PRISM_DOCK_ENV"] = str(env_root)
    env["PRISM_DOCK_PYTHON"] = str(env_root / "bin/python")
    env["PRISM_AMBERHOME"] = str(env_root)
    env["PRISM_ANTECHAMBER"] = str(env_root / "bin/antechamber")
    env["PRISM_OBABEL"] = str(env_root / "bin/obabel")
    env["PRISM_UNIDOCK"] = str(env_root / "bin/unidock")
    env["PRISM_GNINA"] = str(env_root / "bin/gnina")
    env["BABEL_LIBDIR"] = str(env_root / "lib/openbabel/3.1.0")
    env["BABEL_DATADIR"] = str(env_root / "share/openbabel/3.1.0")
    env["PRISM_SCRATCH_ROOT"] = str(scratch_root)
    env["PRISM_CANONICAL_RUNTIME_STRICT"] = "1"
    env["PATH"] = f"{env_root / 'bin'}:{env.get('PATH', '')}"
    return env


def run_step(
    name: str,
    command: list[str],
    report_root: Path,
    env: dict[str, str],
    expected_outputs: list[Path] | None = None,
    accepted_returncodes: set[int] | None = None,
) -> dict[str, Any]:
    step_dir = report_root / name
    step_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = step_dir / "stdout.log"
    stderr_path = step_dir / "stderr.log"
    t0 = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    elapsed = round(time.perf_counter() - t0, 3)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    outputs = []
    for path in expected_outputs or []:
        outputs.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "sha256": sha256(path) if path.exists() and path.is_file() else None,
            }
        )
    ok_returncodes = accepted_returncodes or {0}
    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "expected_outputs": outputs,
        "accepted_returncodes": sorted(ok_returncodes),
        "passed": proc.returncode in ok_returncodes and all(item["exists"] for item in outputs),
    }


def run_custom_step(
    name: str,
    report_root: Path,
    payload: dict[str, Any],
    *,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    step_dir = report_root / name
    step_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = step_dir / "stdout.log"
    stderr_path = step_dir / "stderr.log"
    details_path = step_dir / "details.json"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    details_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "name": name,
        "command": payload.get("command"),
        "returncode": payload.get("returncode", 0),
        "elapsed_seconds": payload.get("elapsed_seconds", 0.0),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "details_json": str(details_path),
        "expected_outputs": payload.get("expected_outputs", []),
        "accepted_returncodes": payload.get("accepted_returncodes", [0]),
        "passed": bool(payload.get("passed")),
    }


def load_registry(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("executables")
    if not isinstance(entries, list):
        raise ValueError(f"runtime registry missing executables list: {path}")
    return [entry for entry in entries if isinstance(entry, dict)]


def run_registry_gate(registry_path: Path, report_root: Path, env: dict[str, str]) -> dict[str, Any]:
    python_exec = Path(env["PRISM_DOCK_PYTHON"])
    entries = load_registry(registry_path)
    details: list[dict[str, Any]] = []
    overall_pass = True
    for entry in entries:
        rel_path = str(entry["path"])
        path = (REPO_ROOT / rel_path).resolve()
        kind = str(entry.get("kind"))
        exists = path.exists()
        record: dict[str, Any] = {
            "id": entry.get("id"),
            "path": rel_path,
            "kind": kind,
            "validation_mode": entry.get("validation_mode"),
            "exists": exists,
            "returncode": None,
            "passed": False,
        }
        if not exists:
            overall_pass = False
            details.append(record)
            continue
        if kind == "python_script":
            proc = subprocess.run(
                [str(python_exec), "-m", "py_compile", str(path)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            record["returncode"] = proc.returncode
            record["stdout"] = proc.stdout
            record["stderr"] = proc.stderr
            record["passed"] = proc.returncode == 0
        elif kind == "shell_script":
            proc = subprocess.run(
                ["bash", "-n", str(path)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            record["returncode"] = proc.returncode
            record["stdout"] = proc.stdout
            record["stderr"] = proc.stderr
            record["passed"] = proc.returncode == 0
        elif kind == "binary":
            record["size_bytes"] = path.stat().st_size if path.is_file() else None
            record["executable"] = os.access(path, os.X_OK)
            record["passed"] = bool(record["size_bytes"]) and bool(record["executable"])
            record["returncode"] = 0 if record["passed"] else 1
        else:
            record["returncode"] = 1
            record["stderr"] = f"unknown registry kind: {kind}"
        overall_pass = overall_pass and bool(record["passed"])
        details.append(record)
    payload = {
        "registry_path": str(registry_path),
        "entry_count": len(entries),
        "entries": details,
        "passed": overall_pass,
    }
    return run_custom_step(
        "runtime_registry_gate",
        report_root,
        payload,
        stdout=json.dumps(
            {
                "registry_path": str(registry_path),
                "entry_count": len(entries),
                "passed_entries": sum(1 for item in details if item["passed"]),
            },
            indent=2,
        )
        + "\n",
        stderr="" if overall_pass else "runtime registry gate failed\n",
    )


def run_env_relocation_gate(env_root: Path, report_root: Path) -> dict[str, Any]:
    manifest_path = env_root.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    source_env = manifest.get("source_env")
    matches: list[str] = []
    if isinstance(source_env, str) and source_env:
        for path in sorted((env_root / "bin").iterdir()):
            if not path.is_file() or path.is_symlink():
                continue
            raw = path.read_bytes()
            if b"\x00" in raw:
                continue
            if source_env.encode("utf-8") in raw:
                matches.append(path.name)
    payload = {
        "env_root": str(env_root),
        "manifest_path": str(manifest_path),
        "source_env": source_env,
        "remaining_source_backreferences_in_bin": len(matches),
        "remaining_files": matches,
        "passed": len(matches) == 0,
    }
    return run_custom_step(
        "env_relocation_gate",
        report_root,
        payload,
        stdout=json.dumps(
            {
                "env_root": str(env_root),
                "remaining_source_backreferences_in_bin": len(matches),
            },
            indent=2,
        )
        + "\n",
        stderr="" if not matches else "\n".join(matches) + "\n",
    )


def prepare_hydration_fixture(report_root: Path) -> dict[str, Path]:
    fixture_root = report_root / "_fixtures" / "hydration_continuity"
    fixture_root.mkdir(parents=True, exist_ok=True)
    event_path = fixture_root / "spike_events_snr_masked.parquet"
    out_dir = fixture_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "condition_id": ["c1", "c1", "c2"],
            "primary_residue_idx": [262, 262, 111],
            "voxel_idx": [700, 700, 999],
            "water_density": [1.0, 3.0, 2.0],
            "wd_change": [0.5, -1.0, 0.25],
            "intensity": [2.0, 4.0, 8.0],
        }
    ).write_parquet(event_path)

    topology_registry = fixture_root / "topology_region_registry.json"
    topology_registry.write_text(
        json.dumps(
            {
                "schema_version": "track_b.topology_region_registry.v1",
                "regions": {"HYDRATION_CORRIDOR": {"residues": [{"residue_id": "GLU262"}]}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    captured_registry = fixture_root / "captured_tile_registry.json"
    captured_registry.write_text(
        json.dumps(
            {
                "schema_version": "prism.log_subtb.captured_tile_registry.v1",
                "tiles": [
                    {
                        "tile_id": "tile_hydration_001",
                        "tile_type": "hydration_channel_preservation",
                        "topology_region": "HYDRATION_CORRIDOR",
                        "perturbation_family": "HYDRATION_WIRE_PROBE",
                        "affected_voxel_ids": [700],
                        "affected_state_ids": [1, 2],
                        "affected_bsr_blocks": [3, 4],
                        "delta_values": [[[0.1]], [[-0.05]]],
                        "restricted_operator_target": "W_without_arr(Pi)",
                        "capture_shape_bucket": "rows2_blocks2_block1_float64",
                        "cuda_graph_id": "cuda_graph::test",
                        "tile_delta_hash": "tile_delta_hash",
                        "provenance_hash": "provenance_hash",
                        "topology_delta": "topology_delta_hash",
                        "basin_delta": "basin_delta_hash",
                        "restricted_operator_hash": "restricted_hash",
                        "c6_operator_hash": "c6_hash",
                        "captured_graph_tile_hash": "captured_hash"
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    spectral_metrics = fixture_root / "subtb_training_metrics.parquet"
    pl.DataFrame(
        {
            "captured_tile_replay_count": [3],
            "gpu_solve_count": [2],
            "cpu_solve_count": [0],
            "reward_cache_hit_rate": [0.5],
            "reward_event_source": ["captured_tile_delta_hashes"],
        }
    ).write_parquet(spectral_metrics)
    subtb_manifest = fixture_root / "subtb_run_manifest.json"
    subtb_manifest.write_text(
        json.dumps(
            {
                "status": "LOG_SUBTB_CAPTURED_TILE_RUNTIME_VERIFIED",
                "spectral_reward_manager": {
                    "event_trigger_source": "captured_tile_delta_hashes",
                    "reward_cache_hit_rate": 0.5,
                    "gpu_solve_count": 2,
                    "cpu_solve_count": 0,
                },
                "captured_graph_replay_count": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    chronology = fixture_root / "transition_chronology_tensor.parquet"
    pl.DataFrame(
        {
            "condition": ["c1", "c2"],
            "residues": [[262], [111]],
            "voxel_idx": [700, 999],
        }
    ).write_parquet(chronology)
    hysteresis = fixture_root / "hysteresis_tensor.parquet"
    pl.DataFrame(
        {
            "condition_id": ["c1", "c2"],
            "primary_residue_idx": [262, 111],
            "thermal_irreversibility": [0.2, 0.1],
            "hysteresis_delta": [0.4, 0.3],
        }
    ).write_parquet(hysteresis)
    signal_grid = fixture_root / "signal_grid_variance_channel.parquet"
    pl.DataFrame({"voxel_idx": [700, 999], "variance": [0.1, 0.2]}).write_parquet(signal_grid)
    nma_probe = fixture_root / "nma_probe.json"
    nma_probe.write_text(
        json.dumps(
            {
                "residue_ids": [111, 262],
                "modes": [
                    {"eigenvalue": 1.0, "displacements": [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]},
                    {"eigenvalue": 4.0, "displacements": [[0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "fixture_root": fixture_root,
        "event_parquet": event_path,
        "out_dir": out_dir,
        "topology_registry": topology_registry,
        "captured_registry": captured_registry,
        "spectral_metrics": spectral_metrics,
        "subtb_manifest": subtb_manifest,
        "chronology": chronology,
        "hysteresis": hysteresis,
        "signal_grid": signal_grid,
        "nma_probe": nma_probe,
    }


def main() -> int:
    args = parse_args()
    env_root = args.env_root.resolve()
    scratch_root = args.scratch_root.resolve()
    registry_path = args.registry.resolve()
    if args.report_root is None:
        report_root = DEFAULT_REPORT_BASE / f"prism_canonical_runtime_verification_{now_utc()}_{git_head_short()}"
    else:
        report_root = args.report_root.resolve()
    if report_root.exists():
        shutil.rmtree(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    env = make_env(env_root, scratch_root)

    control_number = f"PRISM-CANONICAL-RUNTIME-{now_utc()}-{git_head_short()}"
    summary: dict[str, Any] = {
        "schema_version": "prism.canonical_runtime_verification.v1",
        "control_number": control_number,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_head": git_head_short(),
        "env_root": str(env_root),
        "scratch_root": str(scratch_root),
        "steps": [],
    }

    env_manifest = env_root.with_suffix(".manifest.json")
    summary["env_manifest"] = {
        "path": str(env_manifest),
        "exists": env_manifest.exists(),
        "sha256": sha256(env_manifest) if env_manifest.exists() else None,
    }
    summary["runtime_registry"] = {
        "path": str(registry_path),
        "exists": registry_path.exists(),
        "sha256": sha256(registry_path) if registry_path.exists() else None,
    }

    python_exec = env_root / "bin/python"
    imports_cmd = [
        str(python_exec),
        "-c",
        (
            "import importlib; "
            "mods=['openff','openff.toolkit','openff.nagl','rdkit','openmm','torch','torch_geometric','polars','pyarrow','pandas','scipy','sklearn','meeko','plip','yaml']; "
            "missing=[]; "
            "[(importlib.import_module(m), print(f'{m}=OK')) for m in mods]; "
            "print('IMPORT_GATE=PASS')"
        ),
    ]
    summary["steps"].append(run_step("imports", imports_cmd, report_root, env))
    summary["steps"].append(run_env_relocation_gate(env_root, report_root))
    summary["steps"].append(run_registry_gate(registry_path, report_root, env))
    summary["steps"].append(
        run_step(
            "engine_wrapper_usage_gate",
            ["bash", "scripts/prism-validate-and-run.sh"],
            report_root,
            env,
            accepted_returncodes={1},
        )
    )

    prep_dir = report_root / "prep_smoke"
    prep_topology = prep_dir / "4lpk_test.topology.json"
    summary["steps"].append(
        run_step(
            "prep_smoke",
            [
                str(python_exec),
                "scripts/prism-prep",
                "vendor/test_targets/4lpk/4lpk_raw.pdb",
                str(prep_topology),
                "--hmr",
                "--quiet",
            ],
            report_root,
            env,
            expected_outputs=[
                prep_topology,
                prep_dir / "4lpk_test.residue_map.json",
                prep_dir / "4lpk_test.atom_to_residue.json",
            ],
        )
    )

    compile_dir = report_root / "compile_smoke"
    compiled_topology = compile_dir / "glp1r_6XOX_HOLO_cand_015_bccda098.topology.json"
    summary["steps"].append(
        run_step(
            "compile_smoke",
            [
                str(python_exec),
                "scripts/compile_candidate_holo_topology.py",
                "--candidate-id",
                "cand_015_bccda098",
                "--sdf",
                "campaigns/glp1r_aleniglipron/track_a_generative/gpu_dispatch/sdf/cand_015_bccda098.sdf",
                "--base-receptor-topology",
                "campaigns/glp1r_aleniglipron/topologies/glp1r_6XOX_A316T.topology.json",
                "--output-dir",
                str(compile_dir),
                "--condition-prefix",
                "glp1r_6XOX_HOLO",
            ],
            report_root,
            env,
            expected_outputs=[compiled_topology],
        )
    )

    dstw_dir = report_root / "dstw_smoke"
    summary["steps"].append(
        run_step(
            "dstw_smoke",
            [
                "target/release/dstw_export_wt_pathb",
                "--materialized-sites",
                "/mnt/storage/tmp/glp1r_candidate_md_smoke_20260527_233437/materialized_openff/cand_015_bccda098_glp1r_6XOX_WT_replica_0/binding_sites.materialized.json",
                "--residue-map",
                "campaigns/glp1r_aleniglipron/topologies/glp1r_6XOX_WT.residue_map.json",
                "--topology-json",
                "campaigns/glp1r_aleniglipron/topologies/glp1r_6XOX_WT.topology.json",
                "--target",
                "GLP1R",
                "--uniprot-accession",
                "P43220",
                "--structure-anchor-id",
                "6XOX",
                "--receptor-chain-id",
                "R",
                "--prism-run-id",
                "cand015_canonical_runtime_smoke",
                "--out-json",
                str(dstw_dir / "dstw.json"),
                "--out-parquet",
                str(dstw_dir / "dstw.parquet"),
                "--out-contact-parquet",
                str(dstw_dir / "dstw_contacts.parquet"),
            ],
            report_root,
            env,
            expected_outputs=[
                dstw_dir / "dstw.json",
                dstw_dir / "dstw.parquet",
                dstw_dir / "dstw_contacts.parquet",
            ],
        )
    )

    gflownet_sample_dir = report_root / "gflownet_sample_smoke"
    summary["steps"].append(
        run_step(
            "gflownet_sample_smoke",
            [
                str(python_exec),
                "scripts/sample_gflownet_candidates.py",
                "--num-samples",
                "8",
                "--batch-size",
                "4",
                "--skip-hard-fail-gates",
                "--output",
                str(gflownet_sample_dir / "top500.parquet"),
                "--top100-output",
                str(gflownet_sample_dir / "top100.parquet"),
                "--raw-output",
                str(gflownet_sample_dir / "raw.parquet"),
                "--summary-output",
                str(gflownet_sample_dir / "summary.json"),
                "--top500-csv-output",
                str(gflownet_sample_dir / "top500.csv"),
                "--top100-csv-output",
                str(gflownet_sample_dir / "top100.csv"),
                "--top500-md-output",
                str(gflownet_sample_dir / "top500.md"),
            ],
            report_root,
            env,
            expected_outputs=[
                gflownet_sample_dir / "raw.parquet",
                gflownet_sample_dir / "summary.json",
                gflownet_sample_dir / "top500.parquet",
                gflownet_sample_dir / "top100.parquet",
            ],
        )
    )

    if not args.skip_gflownet_train:
        gflownet_train_dir = report_root / "gflownet_train_smoke"
        summary["steps"].append(
            run_step(
                "gflownet_train_smoke",
                [
                    str(python_exec),
                    "scripts/train_gflownet_policy.py",
                    "--epochs",
                    "1",
                    "--warm-start-epochs",
                    "0",
                    "--batch-size",
                    "4",
                    "--torch-threads",
                    "2",
                    "--reward-version",
                    "v3_scaffold_consensus",
                    "--consensus-bonus-weight",
                    "2.0",
                    "--signal-grid",
                    "campaigns/glp1r_aleniglipron/track_a_generative/signal_grid_scaffold_consensus.parquet",
                    "--survivors",
                    "campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_scaffold_consensus_action_corpus.parquet",
                    "--oracle-mode",
                    "survivor_lookup",
                    "--output-dir",
                    str(gflownet_train_dir / "output"),
                    "--output-policy",
                    str(gflownet_train_dir / "output/policy.pt"),
                    "--checkpoint-dir",
                    str(gflownet_train_dir / "checkpoints"),
                ],
                report_root,
                env,
                expected_outputs=[
                    gflownet_train_dir / "output/policy.pt",
                    gflownet_train_dir / "output/gflownet_learning_validation.json",
                ],
            )
        )

    dock_dir = report_root / "dock_smoke"
    summary["steps"].append(
        run_step(
            "dock_smoke",
            [
                str(python_exec),
                "scripts/gpu_dock.py",
                "--receptor",
                "vendor/test_targets/4lpk/4lpk_clean.pdb",
                "--sites",
                "tests/test_genphore/fixtures/mock_binding_sites.json",
                "--ligands",
                "campaigns/glp1r_aleniglipron/track_a_generative/ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf",
                "--output",
                str(dock_dir),
                "--gnina-only",
            ],
            report_root,
            env,
            expected_outputs=[
                dock_dir / "4lpk_clean.pdbqt",
                dock_dir / "site0/site0_docking_results.json",
                dock_dir / "site0/gnina_docked.sdf",
            ],
        )
    )

    summary["steps"].append(
        run_step(
            "md_only_postflight_contract",
            [
                "python3",
                "scripts/prism-postflight-md-only.py",
                "/mnt/storage/tmp/glp1r_candidate_md_smoke_20260527_233437/engine_openff/cand_015_bccda098_glp1r_6XOX_WT_replica_0",
                "glp1r_6XOX_HOLO_cand_015_bccda098",
            ],
            report_root,
            env,
        )
    )

    phase_dir = (
        REPO_ROOT
        / "campaigns/glp1r_aleniglipron/integrated_spike_events/runtime_verifier_phase_smoke"
        / report_root.name
    )
    summary["steps"].append(
        run_step(
            "phase_manifold_smoke",
            [
                str(python_exec),
                "scripts/prism_phase_manifold_coherence.py",
                "--spike-events",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "spike_events_snr_masked.parquet"),
                "--stream-phase-counts",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "stream_level_phase_counts.parquet"),
                "--steering",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "autonomous_steering_tensor.parquet"),
                "--mechanical-load",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "mechanical_load_network.parquet"),
                "--kcc",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "kcc_residue_fields.parquet"),
                "--bocpd",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "bocpd_survival_regimes.parquet"),
                "--kinetic-strain",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "kinetic_strain_events.parquet"),
                "--aromatic",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "aromatic_reorganization_tensor.parquet"),
                "--signal-grid",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "signal_grid_variance_channel.parquet"),
                "--snr-masks",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "stream_snr_masks.parquet"),
                "--spike-residue-fields",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "spike_residue_fields.parquet"),
                "--protocol-state",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "protocol_state_summary.parquet"),
                "--channel-summary",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "receptor_durability_channel_summary.parquet"),
                "--risk-map",
                str(PHASE_MANIFOLD_FIXTURE_DIR / "receptor_durability_risk_map.parquet"),
                "--mapping",
                "campaigns/glp1r_aleniglipron/topology/residue_index_mapping_matrix.parquet",
                "--grid-mapping",
                "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json",
                "--critical-edges-json",
                "campaigns/glp1r_aleniglipron/track_0_manual_emulation/binding_site_reference.json",
                "--output",
                str(phase_dir / "phase_manifold_coherence.parquet"),
                "--edge-validation-out",
                str(phase_dir / "phase_manifold_edge_validation.parquet"),
                "--log-level",
                "INFO",
            ],
            report_root,
            env,
            expected_outputs=[
                phase_dir / "phase_manifold_coherence.parquet",
                phase_dir / "phase_manifold_edge_validation.parquet",
            ],
        )
    )

    fixture = prepare_hydration_fixture(report_root)
    summary["steps"].append(
        run_step(
            "hydration_extractor_smoke",
            [
                str(python_exec),
                "scripts/prism_v2_hydration_extractor.py",
                "--event-parquet",
                str(fixture["event_parquet"]),
                "--out-dir",
                str(fixture["out_dir"]),
                "--topology-region-registry",
                str(fixture["topology_registry"]),
                "--captured-tile-registry",
                str(fixture["captured_registry"]),
                "--spectral-metrics",
                str(fixture["spectral_metrics"]),
                "--subtb-run-manifest",
                str(fixture["subtb_manifest"]),
            ],
            report_root,
            env,
            expected_outputs=[fixture["out_dir"] / "hydration_statistics.parquet"],
        )
    )

    continuity_dir = report_root / "continuity_maps_smoke"
    summary["steps"].append(
        run_step(
            "continuity_maps_smoke",
            [
                str(python_exec),
                "scripts/build_continuity_maps.py",
                "--hysteresis",
                str(fixture["hysteresis"]),
                "--chronology",
                str(fixture["chronology"]),
                "--signal-grid",
                str(fixture["signal_grid"]),
                "--nma-glob",
                str(fixture["fixture_root"] / "nma*.json"),
                "--hydration-glob",
                str(fixture["out_dir"] / "*.parquet"),
                "--output-dir",
                str(continuity_dir),
            ],
            report_root,
            env,
            expected_outputs=[
                continuity_dir / "nma_continuity_map.parquet",
                continuity_dir / "hydration_continuity_map.parquet",
                continuity_dir / "thermodynamic_continuity_map.parquet",
                continuity_dir / "continuity_map_manifest.json",
            ],
        )
    )

    subtb_dir = report_root / "subtb_spectral_smoke"
    summary["steps"].append(
        run_step(
            "subtb_spectral_smoke",
            [
                str(python_exec),
                "scripts/run_log_subtb_spectral_gflownet.py",
                "--campaign",
                "glp1r_aleniglipron",
                "--epochs",
                "1",
                "--mode",
                "synthetic",
                "--max-tiles",
                "2",
                "--output-root",
                str(subtb_dir),
            ],
            report_root,
            env,
            expected_outputs=[
                subtb_dir / "captured_graph_tiles/captured_tile_registry.json",
                subtb_dir / "captured_graph_tiles/tile_capture_manifest.json",
                subtb_dir / "subtb_training_metrics.parquet",
                subtb_dir / "subtb_run_manifest.json",
            ],
        )
    )

    passed = all(step["passed"] for step in summary["steps"])
    summary["status"] = "PASS" if passed else "FAIL"
    summary["passed_steps"] = sum(1 for step in summary["steps"] if step["passed"])
    summary["total_steps"] = len(summary["steps"])

    json_path = report_root / "PRISM_CANONICAL_RUNTIME_VERIFICATION.json"
    md_path = report_root / "PRISM_CANONICAL_RUNTIME_VERIFICATION.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        f"# PRISM Canonical Runtime Verification",
        "",
        f"- control_number: `{control_number}`",
        f"- status: `{summary['status']}`",
        f"- env_root: `{env_root}`",
        f"- scratch_root: `{scratch_root}`",
        f"- git_head: `{summary['git_head']}`",
        f"- passed_steps: `{summary['passed_steps']}/{summary['total_steps']}`",
        "",
        "| step | returncode | passed | stdout | stderr |",
        "|---|---:|---:|---|---|",
    ]
    for step in summary["steps"]:
        md_lines.append(
            f"| {step['name']} | {step['returncode']} | {str(step['passed']).lower()} | "
            f"`{step['stdout_log']}` | `{step['stderr_log']}` |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "report_root": str(report_root), "json": str(json_path)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
