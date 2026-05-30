#!/usr/bin/env python3
"""Stage an authority-first Tier 3 GLP1R PoV runbook for later cloud execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN_ROOT / "track_a_generative"
PHASE3_INDEX = (
    CAMPAIGN_ROOT
    / "track_b_chronological/expanded_variant_run/phase3_topology_runnable_target_index.json"
)
EXPANDED_EXACT_INDEX = (
    CAMPAIGN_ROOT
    / "track_b_chronological/expanded_variant_run/expanded_6xox_exact_target_runnable_index.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=CAMPAIGN_ROOT,
        help="Campaign root used for staged runbook generation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CAMPAIGN_ROOT / "tier3_pov" / "staged_runs",
        help="Directory that will receive the staged runbook.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional fixed run ID. Defaults to tier3-pov-<UTC timestamp>.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=10,
        help="Replicate count used when building Loop 2/Loop 3 manifests.",
    )
    parser.add_argument(
        "--pods-loop2",
        type=int,
        default=10,
        help="Recommended pod count for Loop 2 shard planning.",
    )
    parser.add_argument(
        "--pods-loop3",
        type=int,
        default=5,
        help="Recommended pod count for Loop 3 shard planning.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_short_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def make_run_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"tier3-pov-{stamp}"


def make_control_number(run_id: str) -> str:
    return f"PRISM-T3POV-{run_id.replace('tier3-pov-', '')}-{git_short_head()}".upper()


def sha256_path(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rel(repo_path: Path) -> str:
    return str(repo_path.relative_to(REPO_ROOT))


def file_record(path: Path, *, phase: str, role: str, required: bool = True, notes: str = "") -> dict[str, Any]:
    exists = path.exists()
    return {
        "logical_id": path.name,
        "path": rel(path),
        "phase": phase,
        "role": role,
        "required": required,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256_path(path) if exists else None,
        "notes": notes,
    }


def load_phase3_index() -> dict[str, Any]:
    return json.loads(PHASE3_INDEX.read_text(encoding="utf-8"))


def load_expanded_exact_index() -> dict[str, Any]:
    return json.loads(EXPANDED_EXACT_INDEX.read_text(encoding="utf-8"))


def phase3_path_set(index: dict[str, Any]) -> set[str]:
    return {record["path"] for record in index.get("targets", [])}


def exact_target_map(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["variant_id"]): record for record in index.get("targets", []) if record.get("variant_id")}


def expected_object_prefix(run_id: str, loop_name: str, row_id: str) -> str:
    return f"r2://prism-archive/tier3-pov/{run_id}/{loop_name}/{row_id}/"


def shard_ranges(total_rows: int, pod_count: int) -> list[tuple[int, int]]:
    per_pod = math.ceil(total_rows / pod_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_rows:
        end = min(total_rows, start + per_pod)
        ranges.append((start, end))
        start = end
    return ranges


def build_target_catalog(
    campaign_root: Path,
    phase3_paths: set[str],
    expanded_exact_targets: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    n182_exact = expanded_exact_targets["ALENI-ECD_TM1_GATEWAY-SEVERING_PROBE-N182G"]
    n182_exact_path = Path(str(n182_exact["path"]))
    entries = {
        "glp1r_6XOX_WT": {
            "path": campaign_root / "topologies/glp1r_6XOX_WT.topology.json",
            "authority_tier": "AUTHORITY_A_PENDING_PROMOTION",
            "loop": "loop0/loop2",
            "panel_role": "active_baseline",
            "notes": "WT baseline control lane; must clear supplemental topology audit because it is outside the historical Phase 3 variant index.",
            "supplemental_audit_allowed": True,
        },
        "glp1r_6XOX_A316T": {
            "path": campaign_root / "topologies/materialized_phase3/glp1r_6XOX_A316T.topology.json",
            "authority_tier": "AUTHORITY_A",
            "loop": "loop0/loop2",
            "panel_role": "pgx_sentinel_active",
            "notes": "Strongest existing rescue-conditioning target.",
        },
        "glp1r_6XOX_T149M": {
            "path": campaign_root / "topologies/materialized_phase3/glp1r_6XOX_T149M.topology.json",
            "authority_tier": "AUTHORITY_A",
            "loop": "loop0/loop2",
            "panel_role": "pgx_sentinel_active",
            "notes": "Second strongest PGx rescue-conditioning target.",
        },
        "glp1r_5VEX_A316T": {
            "path": campaign_root / "topologies/materialized_phase3/glp1r_5VEX_A316T.topology.json",
            "authority_tier": "AUTHORITY_A",
            "loop": "loop2",
            "panel_role": "inactive_crosscheck",
            "notes": "Active-vs-inactive selectivity lane.",
        },
        "glp1r_5VEX_T149M": {
            "path": campaign_root / "topologies/materialized_phase3/glp1r_5VEX_T149M.topology.json",
            "authority_tier": "AUTHORITY_A",
            "loop": "loop2",
            "panel_role": "inactive_crosscheck",
            "notes": "Second inactive-state cross-check lane.",
        },
        "glp1r_6XOX_R227H": {
            "path": campaign_root / "topologies/materialized_phase3/glp1r_6XOX_R227H.topology.json",
            "authority_tier": "AUTHORITY_B",
            "loop": "loop0/loop2",
            "panel_role": "hydration_stress",
            "notes": "Materialized topology exists; conditioning source is weaker than A316T/T149M.",
        },
        "glp1r_6XOX_R421W": {
            "path": campaign_root / "topologies/materialized_phase3/glp1r_6XOX_R421W.topology.json",
            "authority_tier": "AUTHORITY_B",
            "loop": "loop0/loop2",
            "panel_role": "signaling_stress",
            "notes": "Materialized topology exists; conditioning source is weaker than A316T/T149M.",
        },
        "glp1r_6X1A_clean_A316T": {
            "path": campaign_root / "topologies/materialized_phase3/glp1r_6X1A_clean_A316T.topology.json",
            "authority_tier": "AUTHORITY_A",
            "loop": "loop2",
            "panel_role": "anchor_portability",
            "notes": "Cross-anchor portability lane.",
        },
        "glp1r_6XOX_W297R": {
            "path": campaign_root / "topologies/materialized_phase3/glp1r_6XOX_W297R.topology.json",
            "authority_tier": "AUTHORITY_A",
            "loop": "loop3",
            "panel_role": "falsification_stress",
            "notes": "Aromatic relay stress lane.",
        },
        "glp1r_6XOX_Y291C": {
            "path": campaign_root / "topologies/materialized_phase3/glp1r_6XOX_Y291C.topology.json",
            "authority_tier": "AUTHORITY_A",
            "loop": "loop3",
            "panel_role": "falsification_stress",
            "notes": "Bistate relay perturbation lane.",
        },
        "glp1r_6XOX_C226R": {
            "path": campaign_root / "topologies/materialized_phase3/glp1r_6XOX_C226R.topology.json",
            "authority_tier": "AUTHORITY_A",
            "loop": "loop3",
            "panel_role": "falsification_stress",
            "notes": "Hydration/entryway stress lane.",
        },
        "glp1r_6XOX_R190Q": {
            "path": campaign_root / "topologies/materialized_phase3/glp1r_6XOX_R190Q.topology.json",
            "authority_tier": "AUTHORITY_A",
            "loop": "loop3",
            "panel_role": "falsification_stress",
            "notes": "Relay perturbation lane for finalist translation.",
        },
        "glp1r_6XOX_N182G": {
            "path": n182_exact_path,
            "authority_tier": "AUTHORITY_A",
            "loop": "loop0/loop3",
            "panel_role": "expected_fail_falsification",
            "notes": "ECD_TM1_GATEWAY exact falsification lane sourced from the audited expanded 6XOX exact runnable index.",
            "variant_id": str(n182_exact["variant_id"]),
            "supplemental_audit_allowed": False,
        },
    }
    for target_id, payload in entries.items():
        path = payload["path"]
        payload["path"] = rel(path)
        payload["exists"] = path.exists()
        payload["size_bytes"] = path.stat().st_size if path.exists() else None
        payload["sha256"] = sha256_path(path) if path.exists() else None
        payload["in_phase3_runnable_index"] = payload["path"] in phase3_paths
        payload["in_expanded_exact_runnable_index"] = bool(
            target_id == "glp1r_6XOX_N182G" and path == n182_exact_path
        )
        payload["promotion_required"] = payload["authority_tier"].endswith("PENDING_PROMOTION")
        payload["source_regime"] = (
            "observed_embedded_n80_materialized"
            if target_id in {"glp1r_6XOX_A316T", "glp1r_6XOX_T149M"}
            else "expanded_6xox_exact_runnable_index"
            if target_id == "glp1r_6XOX_N182G"
            else "materialized_phase3"
            if target_id != "glp1r_6XOX_WT"
            else "supplemental_control_audit_required"
        )
    return entries


def build_loop1_heads(campaign_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "head_id": "generalist_consensus_head",
            "base_config": rel(TRACK_A / "gflownet_training_config.json"),
            "active_variant_override": "WT",
            "signal_grid_override": rel(TRACK_A / "signal_grid_population_consensus.parquet"),
            "objective_authority": "AUTHORITY_A",
            "selection_goal": "consensus_winner_across_WT_A316T_T149M",
            "scaffold_policy": "multi_scaffold_pool",
        },
        {
            "head_id": "A316T_rescue_head",
            "base_config": rel(TRACK_A / "gflownet_training_config.json"),
            "active_variant_override": "A316T",
            "signal_grid_override": rel(
                campaign_root
                / "integrated_spike_events/n80_full_scale/signal_grid_variance_channel_A316T.parquet"
            ),
            "objective_authority": "AUTHORITY_A",
            "selection_goal": "pgx_rescue_specialist",
            "scaffold_policy": "multi_scaffold_pool",
        },
        {
            "head_id": "T149M_rescue_head",
            "base_config": rel(TRACK_A / "gflownet_training_config.json"),
            "active_variant_override": "T149M",
            "signal_grid_override": rel(
                campaign_root
                / "integrated_spike_events/n80_full_scale/signal_grid_variance_channel_T149M.parquet"
            ),
            "objective_authority": "AUTHORITY_A",
            "selection_goal": "pgx_rescue_specialist",
            "scaffold_policy": "multi_scaffold_pool",
        },
        {
            "head_id": "ALENI_scaffold_A316T_head",
            "base_config": rel(TRACK_A / "gflownet_training_config.json"),
            "active_variant_override": "A316T",
            "signal_grid_override": rel(
                campaign_root
                / "integrated_spike_events/n80_full_scale/signal_grid_variance_channel_A316T.parquet"
            ),
            "objective_authority": "AUTHORITY_A",
            "selection_goal": "scaffold_preserving_pgx_rescue",
            "scaffold_policy": "aleniglipron_core_frozen",
            "scaffold_manifest": rel(
                TRACK_A / "scaffold_bound/scaffold_bound_grid_manifest.json"
            ),
        },
        {
            "head_id": "ALENI_scaffold_R227H_head",
            "base_config": rel(TRACK_A / "gflownet_training_config.json"),
            "active_variant_override": "R227H",
            "signal_grid_override": rel(
                TRACK_A / "population_pgx/variant_grids/signal_grid_variance_channel_R227H.parquet"
            ),
            "objective_authority": "AUTHORITY_B",
            "selection_goal": "scaffold_preserving_hydration_rescue",
            "scaffold_policy": "aleniglipron_core_frozen",
            "scaffold_manifest": rel(
                TRACK_A / "scaffold_bound/scaffold_bound_grid_manifest.json"
            ),
        },
        {
            "head_id": "R421W_signaling_safety_head",
            "base_config": rel(TRACK_A / "gflownet_training_config.json"),
            "active_variant_override": "R421W",
            "signal_grid_override": rel(
                TRACK_A / "population_pgx/variant_grids/signal_grid_variance_channel_R421W.parquet"
            ),
            "objective_authority": "AUTHORITY_B",
            "selection_goal": "signaling_safe_specialist",
            "scaffold_policy": "multi_scaffold_pool",
        },
    ]


def build_loop2_rows(run_id: str, replicates: int, targets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    molecules = [
        {
            "molecule_slot": "ALENI_PARENT",
            "source_kind": "fixed_reference",
            "molecule_path": rel(TRACK_A / "ALENI-PARENT_6XOX_frame_minimized.sdf"),
            "selection_origin": "incumbent_baseline",
        },
        {
            "molecule_slot": "ORFORGLIPRON_BENCHMARK",
            "source_kind": "fixed_reference",
            "molecule_path": rel(TRACK_A / "conformers/benchmark_ORFORGLIPRON_LY3502970.sdf"),
            "selection_origin": "competitor_benchmark",
        },
        {
            "molecule_slot": "CAND_015_BCCDA098",
            "source_kind": "fixed_reference",
            "molecule_path": rel(TRACK_A / "gpu_dispatch/sdf/cand_015_bccda098.sdf"),
            "selection_origin": "historical_de_novo_champion",
        },
        {
            "molecule_slot": "CAND_GENERALIST_V2",
            "source_kind": "loop1_dynamic_slot",
            "molecule_path": "__LOOP1__/cand_generalist_v2.sdf",
            "selection_origin": "generalist_consensus_head",
        },
        {
            "molecule_slot": "CAND_RESCUE_PGX",
            "source_kind": "loop1_dynamic_slot",
            "molecule_path": "__LOOP1__/cand_rescue_pgx.sdf",
            "selection_origin": "A316T_or_T149M_rescue_head_post_dedup",
        },
        {
            "molecule_slot": "CAND_ALENI_SCAFFOLD_RESCUE_V2",
            "source_kind": "loop1_dynamic_slot",
            "molecule_path": "__LOOP1__/cand_aleni_scaffold_rescue_v2.sdf",
            "selection_origin": "ALENI_scaffold_A316T_head_or_ALENI_scaffold_R227H_head_post_selection",
        },
    ]
    target_ids = [
        "glp1r_6XOX_WT",
        "glp1r_6XOX_A316T",
        "glp1r_6XOX_T149M",
        "glp1r_5VEX_A316T",
        "glp1r_5VEX_T149M",
        "glp1r_6XOX_R227H",
        "glp1r_6XOX_R421W",
        "glp1r_6X1A_clean_A316T",
    ]
    rows: list[dict[str, Any]] = []
    for molecule in molecules:
        for target_id in target_ids:
            target = targets[target_id]
            for replicate in range(1, replicates + 1):
                row_id = (
                    f"loop2-{molecule['molecule_slot'].lower()}-"
                    f"{target_id.lower()}-rep{replicate:02d}"
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "loop": "loop2",
                        "row_id": row_id,
                        "molecule_slot": molecule["molecule_slot"],
                        "molecule_path": molecule["molecule_path"],
                        "molecule_source_kind": molecule["source_kind"],
                        "selection_origin": molecule["selection_origin"],
                        "target_id": target_id,
                        "topology_path": target["path"],
                        "replicate": replicate,
                        "authority_tier": target["authority_tier"],
                        "promotion_required": target["promotion_required"],
                        "panel_role": target["panel_role"],
                        "expected_fail": False,
                        "object_prefix": expected_object_prefix(run_id, "loop2", row_id),
                    }
                )
    return rows


def build_loop3_rows(run_id: str, replicates: int, targets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    molecules = [
        {
            "molecule_slot": "ALENI_PARENT",
            "molecule_path": rel(TRACK_A / "ALENI-PARENT_6XOX_frame_minimized.sdf"),
            "selection_origin": "incumbent_baseline",
        },
        {
            "molecule_slot": "BEST_DE_NOVO_LOOP2",
            "molecule_path": "__LOOP2__/best_de_novo.sdf",
            "selection_origin": "loop2_finalist_selection",
        },
        {
            "molecule_slot": "BEST_SCAFFOLD_RESCUE_LOOP2",
            "molecule_path": "__LOOP2__/best_scaffold_rescue.sdf",
            "selection_origin": "loop2_finalist_selection",
        },
    ]
    target_ids = [
        "glp1r_6XOX_W297R",
        "glp1r_6XOX_Y291C",
        "glp1r_6XOX_C226R",
        "glp1r_6XOX_R190Q",
        "glp1r_6XOX_N182G",
    ]
    rows: list[dict[str, Any]] = []
    for molecule in molecules:
        for target_id in target_ids:
            target = targets[target_id]
            for replicate in range(1, replicates + 1):
                row_id = (
                    f"loop3-{molecule['molecule_slot'].lower()}-"
                    f"{target_id.lower()}-rep{replicate:02d}"
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "loop": "loop3",
                        "row_id": row_id,
                        "molecule_slot": molecule["molecule_slot"],
                        "molecule_path": molecule["molecule_path"],
                        "selection_origin": molecule["selection_origin"],
                        "target_id": target_id,
                        "topology_path": target["path"],
                        "replicate": replicate,
                        "authority_tier": target["authority_tier"],
                        "promotion_required": target["promotion_required"],
                        "panel_role": target["panel_role"],
                        "expected_fail": target_id == "glp1r_6XOX_N182G",
                        "object_prefix": expected_object_prefix(run_id, "loop3", row_id),
                    }
                )
    return rows


def build_pod_assignment(loop_name: str, rows: list[dict[str, Any]], pod_count: int) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for pod_index, (start, end) in enumerate(shard_ranges(len(rows), pod_count), start=1):
        shard_rows = rows[start:end]
        assignments.append(
            {
                "loop": loop_name,
                "pod_id": f"{loop_name}-pod-{pod_index:02d}",
                "row_start": start,
                "row_end_exclusive": end,
                "row_count": len(shard_rows),
                "first_row_id": shard_rows[0]["row_id"],
                "last_row_id": shard_rows[-1]["row_id"],
            }
        )
    return assignments


def build_verification_gates() -> dict[str, Any]:
    return {
        "schema_version": "prism.tier3_pov.verification_gates.v1",
        "generated_at_utc": now_utc(),
        "gates": [
            "loop0_target_authority_gate",
            "loop0_wt_promotion_gate",
            "loop0_n182_promotion_gate",
            "loop1_override_manifest_gate",
            "loop1_scaffold_core_invariance_gate",
            "loop1_chemotype_dedup_gate",
            "loop1_dynamic_slot_nomination_gate",
            "loop2_matrix_completeness_gate",
            "loop2_no_silent_worker_drop_gate",
            "loop2_per_row_receipt_gate",
            "loop3_expected_fail_flag_gate",
            "loop3_transition_chronology_presence_gate",
            "loop3_motif_attribution_presence_gate",
            "final_dossier_provenance_gate",
        ],
    }


def build_runtime_schema() -> dict[str, Any]:
    return {
        "schema_version": "prism.tier3_pov.runtime_event.v1",
        "required_fields": [
            "run_id",
            "loop",
            "row_id",
            "pod_id",
            "worker_pid",
            "state",
            "timestamp_utc",
            "object_prefix",
            "stdout_object",
            "stderr_object",
            "artifact_receipt_object",
            "checksum_receipt_object",
        ],
        "state_machine": [
            "CLAIMED",
            "STARTED",
            "HEARTBEAT",
            "OUTPUT_WRITTEN",
            "CHECKSUM_VERIFIED",
            "VERIFIED",
            "FAILED",
            "REQUEUED",
        ],
    }


def build_container_contract(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": "prism.tier3_pov.container_contract.v1",
        "generated_at_utc": now_utc(),
        "run_id": run_id,
        "entrypoint_command": (
            "scripts/prism-validate-and-run.sh -t <topology.json> -o <output_dir> "
            "--fast --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70 "
            "--fused-steps 6 --hmr --adaptive-dt --multi-differential "
            "--closed-loop-steering --asymmetric-steering --use-xgb-ranker "
            "--replica-seed 42 -v"
        ),
        "forbidden": [
            "direct nhs_rt_full invocation",
            "undeclared workstation absolute paths",
            "local-only scratch dependencies outside container contract",
        ],
        "required_env_keys": [
            "PRISM_RUN_ID",
            "PRISM_LOOP",
            "PRISM_TASK_ID",
            "PRISM_OBJECT_PREFIX",
            "PRISM_INPUT_MANIFEST",
            "PRISM_TOPOLOGY_PATH",
            "PRISM_LIGAND_SDF_PATH",
            "PRISM_VALIDATED",
        ],
    }


def build_r2_keyspace(run_id: str) -> dict[str, Any]:
    base = f"tier3-pov/{run_id}"
    return {
        "schema_version": "prism.tier3_pov.r2_keyspace.v1",
        "generated_at_utc": now_utc(),
        "bucket": "prism-archive",
        "prefixes": {
            "control_plane": f"{base}/control-plane/",
            "loop0": f"{base}/loop0/",
            "loop1": f"{base}/loop1/",
            "loop2": f"{base}/loop2/",
            "loop3": f"{base}/loop3/",
            "verification": f"{base}/verification/",
            "logs": f"{base}/logs/",
        },
    }


def build_checksum_policy() -> dict[str, Any]:
    return {
        "schema_version": "prism.tier3_pov.checksum_policy.v1",
        "generated_at_utc": now_utc(),
        "primary_digest": "blake3_preferred_at_execution",
        "compatibility_digest": "sha256",
        "receipt_required_for_row_completion": True,
    }


def build_pid_registry() -> dict[str, Any]:
    return {
        "schema_version": "prism.tier3_pov.pid_registry.v1",
        "generated_at_utc": now_utc(),
        "control_plane": {
            "image_build_pid": None,
            "manifest_build_pid": None,
            "dispatch_submission_pid": None,
        },
        "pod_processes": [],
        "notes": "Populate at execution time; do not infer task success from exit code alone.",
    }


def build_filetag_seed(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": "prism.tier3_pov.filetag_seed.v1",
        "generated_at_utc": now_utc(),
        "run_id": run_id,
        "tag_roots": [
            "loop0",
            "loop1",
            "loop2",
            "loop3",
            "tracking",
            "verification",
            "cloud",
        ],
        "recommended_commands": [
            "python3 scripts/prism_filetag.py tag-all --root <portable_bundle_root> --include-hidden",
            "python3 scripts/prism_filetag.py snapshot --root <portable_bundle_root> > <portable_bundle_root>/filetag_manifest.json",
            "python3 scripts/prism_filetag.py verify --root <portable_bundle_root> --manifest <portable_bundle_root>/filetag_manifest.json --strict",
        ],
    }


def build_core_inventory(campaign_root: Path) -> list[dict[str, Any]]:
    files = [
        file_record(TRACK_A / "gflownet_training_config.json", phase="loop0", role="base_training_config"),
        file_record(
            TRACK_A / "gflownet_top100_pgx_parity_validated_report.json",
            phase="loop0",
            role="current_variant_screen_evidence",
        ),
        file_record(
            TRACK_A / "scaffold_bound/scaffold_bound_grid_manifest.json",
            phase="loop0",
            role="scaffold_generation_manifest",
        ),
        file_record(
            TRACK_A / "ALENI-PARENT_6XOX_frame_minimized.sdf",
            phase="loop2",
            role="incumbent_reference_ligand",
        ),
        file_record(
            TRACK_A / "conformers/benchmark_ORFORGLIPRON_LY3502970.sdf",
            phase="loop2",
            role="competitor_reference_ligand",
        ),
        file_record(
            TRACK_A / "gpu_dispatch/sdf/cand_015_bccda098.sdf",
            phase="loop2",
            role="historical_de_novo_reference",
        ),
        file_record(
            TRACK_A / "candidate_dossiers/cand_015_bccda098.json",
            phase="loop2",
            role="historical_de_novo_dossier",
        ),
        file_record(
            campaign_root
            / "track_b_chronological/expanded_variant_run/phase3_topology_runnable_target_index.json",
            phase="loop0",
            role="audited_target_index",
        ),
        file_record(
            campaign_root
            / "track_b_chronological/expanded_variant_run/expanded_6xox_exact_target_runnable_index.json",
            phase="loop0",
            role="audited_exact_falsification_index",
        ),
        file_record(
            campaign_root / "track_b_chronological/genealogical_variant_panel.json",
            phase="loop0",
            role="mechanistic_variant_panel",
        ),
        file_record(
            campaign_root / "pgx_full_landscape_report.json",
            phase="loop0",
            role="population_pgx_landscape",
        ),
    ]
    return files


def build_readme(run_id: str, loop2_rows: list[dict[str, Any]], loop3_rows: list[dict[str, Any]]) -> str:
    return f"""# Tier 3 PoV Staged Runbook

Run ID: `{run_id}`

This directory is a staged control-plane bundle. It does not execute science by itself.

## What is staged

- Loop 0 authority-promotion plan
- Loop 1 head overrides and nomination policy
- Loop 2 mixed-crucible matrix template
- Loop 3 falsification/translation matrix template
- PID, file, checksum, tagging, and verification scaffolds
- cloud container contract and object-storage layout

## Matrix sizes

- Loop 2 staged rows: `{len(loop2_rows)}`
- Loop 3 staged rows: `{len(loop3_rows)}`

## Recommended execution sequence

1. Pass Loop 0 authority gates.
2. Build the cloud image and pass the single-pod container completeness gate.
3. Run Loop 1 and lock the three dynamic molecule slots.
4. Materialize the final Loop 2 manifest from the staged template.
5. Dispatch Loop 2 to the pod shard plan.
6. Select finalists and materialize Loop 3.
7. Dispatch Loop 3.
8. Assemble the translational dossier from verified artifacts only.

## Critical truth boundary

- Do not claim `R227H` / `R421W` conditioning has the same authority as `A316T` / `T149M`.
- Do not claim `glp1r_6XOX_WT` is fully ready until the supplemental control audit closes.
- Do not swap out the audited ECD_TM1_GATEWAY `N182G` falsification lane for a generic exact-bank topology without recording the downgrade.
- Do not keep duplicate rescue chemotypes just to preserve the ceiling matrix size.
"""


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    run_id = make_run_id(args.run_id)
    control_number = make_control_number(run_id)
    run_root = args.output_root / run_id
    logging.info("staging_runbook run_id=%s output=%s", run_id, run_root)

    index = load_phase3_index()
    expanded_exact_index = load_expanded_exact_index()
    phase3_paths = phase3_path_set(index)
    targets = build_target_catalog(
        args.campaign_root,
        phase3_paths,
        exact_target_map(expanded_exact_index),
    )
    loop1_heads = build_loop1_heads(args.campaign_root)
    loop2_rows = build_loop2_rows(run_id, args.replicates, targets)
    loop3_rows = build_loop3_rows(run_id, args.replicates, targets)
    pod_plan = build_pod_assignment("loop2", loop2_rows, args.pods_loop2) + build_pod_assignment(
        "loop3", loop3_rows, args.pods_loop3
    )

    for subdir in [
        "loop0",
        "loop1",
        "loop2",
        "loop3",
        "tracking",
        "verification",
        "cloud",
        "logs",
    ]:
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    write_json(
        run_root / "RUNBOOK_STATUS.json",
        {
            "schema_version": "prism.tier3_pov.runbook_status.v1",
            "generated_at_utc": now_utc(),
            "run_id": run_id,
            "control_number": control_number,
            "status": "STAGED_NOT_EXECUTED",
            "loop2_row_count": len(loop2_rows),
            "loop3_row_count": len(loop3_rows),
            "pods_loop2": args.pods_loop2,
            "pods_loop3": args.pods_loop3,
            "notes": [
                "This bundle stages the control plane only.",
                "Loop 0 promotion is mandatory before cloud execution.",
            ],
        },
    )
    write_json(
        run_root / "loop0" / "authority_manifest.json",
        {
            "schema_version": "prism.tier3_pov.loop0_authority_manifest.v1",
            "generated_at_utc": now_utc(),
            "run_id": run_id,
            "control_number": control_number,
            "targets": targets,
        },
    )
    write_json(
        run_root / "loop1" / "head_specs.json",
        {
            "schema_version": "prism.tier3_pov.loop1_head_specs.v1",
            "generated_at_utc": now_utc(),
            "run_id": run_id,
            "control_number": control_number,
            "heads": loop1_heads,
            "nomination_policy": {
                "generalist_slot": "highest_consensus_resilience_head_winner",
                "pgx_rescue_slot": "best_A316T_T149M_union_after_chemotype_dedup",
                "scaffold_rescue_slot": "best_A316T_R227H_scaffold_head_after_core_invariance_gate",
                "matrix_shrink_allowed_if_rescue_heads_converge": True,
            },
        },
    )
    write_json(
        run_root / "loop2" / "execution_manifest.template.json",
        {
            "schema_version": "prism.tier3_pov.loop2_execution_manifest_template.v1",
            "generated_at_utc": now_utc(),
            "run_id": run_id,
            "control_number": control_number,
            "row_count": len(loop2_rows),
            "ceiling_matrix": "6x8x10",
            "allowed_shrink_matrix": "5x8x10_if_rescue_heads_converge",
            "rows": loop2_rows,
        },
    )
    write_json(
        run_root / "loop3" / "execution_manifest.template.json",
        {
            "schema_version": "prism.tier3_pov.loop3_execution_manifest_template.v1",
            "generated_at_utc": now_utc(),
            "run_id": run_id,
            "control_number": control_number,
            "row_count": len(loop3_rows),
            "matrix": "3x5x10",
            "rows": loop3_rows,
        },
    )
    write_json(run_root / "tracking" / "pid_registry.template.json", build_pid_registry())
    write_json(run_root / "tracking" / "runtime_events.schema.json", build_runtime_schema())
    write_json(run_root / "tracking" / "checksum_policy.json", build_checksum_policy())
    write_json(run_root / "tracking" / "filetag_manifest.seed.json", build_filetag_seed(run_id))
    write_json(
        run_root / "tracking" / "control_number.json",
        {
            "schema_version": "prism.tier3_pov.control_number.v1",
            "generated_at_utc": now_utc(),
            "run_id": run_id,
            "control_number": control_number,
            "git_short_head": git_short_head(),
        },
    )
    (run_root / "CONTROL_NUMBER.txt").write_text(control_number + "\n", encoding="utf-8")
    write_json(
        run_root / "tracking" / "core_input_inventory.json",
        {
            "schema_version": "prism.tier3_pov.core_input_inventory.v1",
            "generated_at_utc": now_utc(),
            "run_id": run_id,
            "control_number": control_number,
            "records": build_core_inventory(args.campaign_root),
        },
    )
    write_jsonl(
        run_root / "tracking" / "file_registry.jsonl",
        build_core_inventory(args.campaign_root)
        + [
            {
                "logical_id": "loop0_authority_manifest",
                "path": "loop0/authority_manifest.json",
                "phase": "loop0",
                "role": "staged_control_artifact",
                "required": True,
                "exists": True,
                "notes": "Generated by stage_tier3_pov_runbook.py",
            },
            {
                "logical_id": "loop2_execution_manifest_template",
                "path": "loop2/execution_manifest.template.json",
                "phase": "loop2",
                "role": "staged_control_artifact",
                "required": True,
                "exists": True,
                "notes": "Replace loop1 placeholders after candidate nomination.",
            },
            {
                "logical_id": "loop3_execution_manifest_template",
                "path": "loop3/execution_manifest.template.json",
                "phase": "loop3",
                "role": "staged_control_artifact",
                "required": True,
                "exists": True,
                "notes": "Replace loop2 finalist placeholders after finalist selection.",
            },
        ],
    )
    write_json(run_root / "cloud" / "container_contract.json", build_container_contract(run_id))
    write_json(run_root / "cloud" / "r2_keyspace_plan.json", build_r2_keyspace(run_id))
    write_csv(run_root / "cloud" / "pod_assignment_plan.csv", pod_plan)
    write_json(run_root / "verification" / "verification_gates.json", build_verification_gates())
    (run_root / "tracking" / "worker_heartbeats.jsonl").write_text("", encoding="utf-8")
    (run_root / "tracking" / "artifact_receipts.jsonl").write_text("", encoding="utf-8")
    (run_root / "README.md").write_text(
        build_readme(run_id, loop2_rows, loop3_rows), encoding="utf-8"
    )

    logging.info(
        "staged_runbook_complete run_id=%s loop2_rows=%d loop3_rows=%d",
        run_id,
        len(loop2_rows),
        len(loop3_rows),
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "control_number": control_number,
                "run_root": str(run_root),
                "loop2_rows": len(loop2_rows),
                "loop3_rows": len(loop3_rows),
                "pods_loop2": args.pods_loop2,
                "pods_loop3": args.pods_loop3,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
