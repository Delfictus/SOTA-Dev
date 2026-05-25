"""Freeze a Track A gate by pinning required artifacts with hashes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml


REPO = Path(__file__).resolve().parents[1]
TRACK_A = REPO / "campaigns/glp1r_aleniglipron/track_a_generative"
GATE_ID = "SMARTS_ZMATRIX_FORGE_VALIDATED"
GATE_DIR = TRACK_A / "gates" / GATE_ID
MANIFEST_PATH = GATE_DIR / "gate_manifest.json"
LEDGER_PATH = GATE_DIR / "gate_manifest.propagation.jsonl"

ARTIFACTS: dict[str, Path] = {
    "reaction_registry": REPO / "00_registry/chemistry/reaction_rules.v1.yml",
    "tagged_synthons": TRACK_A / "enamine_130k_synthons_3d.parquet",
    "bounded_survivors": TRACK_A / "vspace_survivors_smarts_bounded.parquet",
    "bounded_telemetry": TRACK_A / "vspace_smarts_bounded_telemetry.json",
}


JsonObject = dict[str, Any]


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty_state() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO,
            text=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return "unknown"
    return "dirty" if result.stdout.strip() else "clean"


def parquet_summary(path: Path) -> JsonObject:
    frame = pl.scan_parquet(path)
    summary = frame.select(pl.len().alias("row_count")).collect()
    result: JsonObject = {"row_count": int(summary["row_count"][0])}
    columns = frame.collect_schema().names()
    result["columns"] = columns
    if "canonical_smiles" in columns:
        unique = frame.select(pl.col("canonical_smiles").n_unique().alias("unique_smiles")).collect()
        result["unique_smiles_count"] = int(unique["unique_smiles"][0])
    if "synthon_id" in columns:
        synthon_stats = frame.select(
            pl.col("synthon_id").n_unique().alias("unique_synthon_count"),
            pl.col("synthon_id").str.contains("MOCK").sum().alias("mock_id_count"),
        ).collect()
        result["unique_synthon_count"] = int(synthon_stats["unique_synthon_count"][0])
        result["mock_id_count"] = int(synthon_stats["mock_id_count"][0])
    if "ingest_status" in columns:
        status_counts = (
            frame.group_by("ingest_status")
            .agg(pl.len().alias("count"))
            .collect()
            .to_dicts()
        )
        result["ingest_status_counts"] = {
            str(row["ingest_status"]): int(row["count"]) for row in status_counts
        }
    if "survival_tier" in columns:
        tier_counts = (
            frame.group_by("survival_tier")
            .agg(pl.len().alias("count"))
            .collect()
            .to_dicts()
        )
        result["survival_tier_counts"] = {
            str(row["survival_tier"]): int(row["count"]) for row in tier_counts
        }
    return result


def json_summary(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    keys = [
        "assembly_mode",
        "real_anchor_count_loaded",
        "mock_anchor_count",
        "attempted_pairs",
        "rotamers_evaluated",
        "z_matrix_active_count",
        "rigid_fallback_count",
        "survivors",
        "candidate_survivors",
        "survived_normal",
        "survived_cryptic_rescue",
        "complement_voxels",
        "stable_occupied",
        "no_fly_voxels",
        "ligand_inside_grid",
    ]
    return {key: payload.get(key) for key in keys if key in payload}


def yaml_summary(path: Path) -> JsonObject:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected YAML object: {path}")
    reactions = payload.get("reactions", [])
    if not isinstance(reactions, list):
        reactions = []
    enabled = [
        str(reaction.get("reaction_id"))
        for reaction in reactions
        if isinstance(reaction, dict) and reaction.get("enabled") is True
    ]
    return {
        "schema_version": payload.get("schema_version"),
        "registry_name": payload.get("registry_name"),
        "enabled_reactions": enabled,
        "reaction_rule_count": len(reactions),
        "disclaimer": payload.get("disclaimer"),
    }


def artifact_record(key: str, path: Path) -> JsonObject:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    record: JsonObject = {
        "key": key,
        "path": repo_relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }
    if suffix == ".parquet":
        record["summary"] = parquet_summary(path)
    elif suffix == ".json":
        record["summary"] = json_summary(path)
    elif suffix in {".yml", ".yaml"}:
        record["summary"] = yaml_summary(path)
    return record


def build_manifest() -> JsonObject:
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    artifacts = [artifact_record(key, path) for key, path in ARTIFACTS.items()]
    telemetry = next(item for item in artifacts if item["key"] == "bounded_telemetry")["summary"]
    synthons = next(item for item in artifacts if item["key"] == "tagged_synthons")["summary"]
    survivors = next(item for item in artifacts if item["key"] == "bounded_survivors")["summary"]
    gates = {
        "reaction_registry_present": True,
        "tagged_synthons_present": True,
        "bounded_survivors_present": True,
        "bounded_telemetry_present": True,
        "smarts_zmatrix_mode": telemetry.get("assembly_mode") == "smarts_zmatrix",
        "real_synthons_loaded": int(telemetry.get("real_anchor_count_loaded", 0)) >= 512,
        "mock_synthons_absent": int(telemetry.get("mock_anchor_count", 1)) == 0
        and int(synthons.get("mock_id_count", 1)) == 0,
        "zmatrix_active": int(telemetry.get("z_matrix_active_count", 0)) > 0,
        "rigid_fallback_absent": int(telemetry.get("rigid_fallback_count", 1)) == 0,
        "bounded_survivors_nonzero": int(survivors.get("row_count", 0)) > 0,
        "full_scale_not_executed": True,
    }
    return {
        "schema_version": "PRISM.track_a_gate_manifest.v1",
        "gate_id": GATE_ID,
        "gate_status": "PASS" if all(gates.values()) else "FAIL",
        "generated_at_utc": generated_at,
        "campaign_id": "glp1r_aleniglipron",
        "track": "Track A",
        "scope": "SMARTS reaction grammar, synthon tagging, SMARTS-anchored Z-matrix assembly, bounded V-space smoke test.",
        "epistemic_boundary": (
            "This gate validates rule-consistent virtual assembly and PRISM-field pruning. "
            "It does not claim guaranteed wet-lab synthetic success or experimental activity."
        ),
        "scaling_boundary": {
            "full_scale_38b_loop_executed": False,
            "full_scale_requires_later_explicit_invocation": True,
            "max_pairs_safety_retained": True,
        },
        "git": {"sha": git_sha(), "dirty_state": git_dirty_state()},
        "validation_gates": gates,
        "artifacts": artifacts,
    }


def main() -> None:
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger_entry = {
        "entry_id": str(uuid.uuid4()),
        "module": "freeze_track_a_gate",
        "operation": "freeze_gate",
        "gate_id": GATE_ID,
        "gate_status": manifest["gate_status"],
        "manifest_path": repo_relative(MANIFEST_PATH),
        "manifest_sha256": sha256_path(MANIFEST_PATH),
        "artifact_sha256": {
            artifact["key"]: artifact["sha256"] for artifact in manifest["artifacts"]
        },
        "timestamp": manifest["generated_at_utc"],
    }
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ledger_entry, sort_keys=True) + "\n")
    print(
        "track_a_gate_frozen "
        f"gate_id={GATE_ID} "
        f"status={manifest['gate_status']} "
        f"manifest={repo_relative(MANIFEST_PATH)} "
        f"sha256={ledger_entry['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
