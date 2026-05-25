#!/usr/bin/env python3
"""Plan dendritic multi-component V-space execution without running Rust."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import polars as pl
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_REQUESTED_INPUT = TRACK_DIR / "enamine_130k_synthons_3d.parquet"
DEFAULT_FALLBACK_INPUT = TRACK_DIR / "enamine_115k_synthons_3d.parquet"
DEFAULT_REACTION_REGISTRY = REPO_ROOT / "00_registry/chemistry/reaction_rules.v1.yml"
DEFAULT_SHARD_PAIRS = 100_000_000
DEFAULT_SCAFFOLD_ANCHORS = 2
BENCHMARK_PAIRS = 100_000
BENCHMARK_SECONDS = 14.0
DEFAULT_DIHEDRAL_SAMPLES = 6
DEFAULT_SURVIVAL_RATE = 0.001
DEFAULT_PARQUET_ROW_BYTES = 2_048
FULL_INPUT_MIN_OK_ROWS = 100_000


@dataclass(frozen=True)
class ReactionRule:
    reaction_id: str
    reaction_name: str
    reaction_class: str
    enabled: bool
    scaffold_role_label: str
    synthon_role_label: str
    torsion_degrees: tuple[int, ...]


@dataclass(frozen=True)
class PathwayCount:
    pathway_id: str
    pathway_type: str
    first_reaction_id: str
    second_reaction_id: str
    first_reaction_class: str
    second_reaction_class: str
    scaffold_anchor_count: int
    first_synthon_role: str
    bridge_scaffold_role: str
    terminal_synthon_role: str
    first_synthon_count: int
    bifunctional_bridge_count: int
    terminal_synthon_count: int
    total_valid_pairs: int
    estimated_rotamers: int
    estimated_runtime_hours: float
    estimated_survivors: int
    estimated_disk_gb: float


@dataclass(frozen=True)
class ShardRow:
    shard_id: int
    pathway_id: str
    pathway_type: str
    reaction_1: str
    reaction_2: str
    global_start_pair_idx: int
    global_end_pair_idx_exclusive: int
    pathway_start_pair_idx: int
    pathway_end_pair_idx_exclusive: int
    synthon_a_start_idx: int
    synthon_a_end_idx: int
    synthon_b_start_idx: int
    synthon_b_end_idx: int
    pair_count: int
    estimated_pairs: int
    estimated_rotamers: int
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_REQUESTED_INPUT)
    parser.add_argument("--synthon-parquet", dest="input", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--fallback-input", type=Path, default=DEFAULT_FALLBACK_INPUT)
    parser.add_argument("--reaction-registry", type=Path, default=DEFAULT_REACTION_REGISTRY)
    parser.add_argument("--reaction-rules", dest="reaction_registry", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=TRACK_DIR)
    parser.add_argument("--output-plan", type=Path)
    parser.add_argument("--output-shards", type=Path)
    parser.add_argument("--output-resume", type=Path)
    parser.add_argument("--shard-pairs", type=int, default=DEFAULT_SHARD_PAIRS)
    parser.add_argument("--max-shard-size", dest="shard_pairs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--scaffold-anchor-count", type=int, default=DEFAULT_SCAFFOLD_ANCHORS)
    parser.add_argument("--dihedral-samples", type=int, default=DEFAULT_DIHEDRAL_SAMPLES)
    parser.add_argument("--survival-rate", type=float, default=DEFAULT_SURVIVAL_RATE)
    parser.add_argument("--row-bytes", type=int, default=DEFAULT_PARQUET_ROW_BYTES)
    parser.add_argument("--full-input-min-ok-rows", type=int, default=FULL_INPUT_MIN_OK_ROWS)
    return parser.parse_args()


def load_reaction_rules(path: Path) -> list[ReactionRule]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"invalid reaction registry payload: {path}")
    reactions = payload.get("reactions")
    if not isinstance(reactions, list):
        raise ValueError(f"reaction registry missing reactions list: {path}")
    rules: list[ReactionRule] = []
    for reaction in reactions:
        if not isinstance(reaction, dict):
            raise ValueError(f"reaction entry is not a mapping in {path}")
        product_bond = reaction.get("product_bond")
        torsion_policy = product_bond.get("torsion_policy") if isinstance(product_bond, dict) else {}
        dihedral_raw = torsion_policy.get("dihedral_deg") if isinstance(torsion_policy, dict) else []
        dihedral = tuple(int(value) for value in dihedral_raw) if isinstance(dihedral_raw, list) else tuple()
        reaction_id = str(reaction["reaction_id"])
        rules.append(
            ReactionRule(
                reaction_id=reaction_id,
                reaction_name=str(reaction.get("reaction_name", reaction_id)),
                reaction_class=str(reaction.get("reaction_class", "")),
                enabled=bool(reaction.get("enabled", False)),
                scaffold_role_label=chemical_role_label(reaction_id, "scaffold"),
                synthon_role_label=chemical_role_label(reaction_id, "synthon"),
                torsion_degrees=dihedral,
            )
        )
    return [rule for rule in rules if rule.enabled]


def chemical_role_label(reaction_id: str, registry_role: str) -> str:
    labels: dict[tuple[str, str], str] = {
        ("RXN_AMIDE_COUPLING", "scaffold"): "Carboxylic_Acid",
        ("RXN_AMIDE_COUPLING", "synthon"): "Amine",
        ("RXN_SUZUKI_ARYL_ARYL", "scaffold"): "Aryl_Halide",
        ("RXN_SUZUKI_ARYL_ARYL", "synthon"): "Boronic_Acid_or_Ester",
        ("RXN_BUCHWALD_HARTWIG", "scaffold"): "Aryl_Halide",
        ("RXN_BUCHWALD_HARTWIG", "synthon"): "Amine",
        ("RXN_SULFONAMIDE", "scaffold"): "Sulfonyl_Chloride",
        ("RXN_SULFONAMIDE", "synthon"): "Amine",
    }
    return labels.get((reaction_id, registry_role), registry_role)


def ok_row_expr() -> pl.Expr:
    return pl.col("ingest_status").eq("ok")


def has_tag_expr(reaction_id: str, role: str) -> pl.Expr:
    tag = f'"{reaction_id}:{role}"'
    return pl.col("reaction_tags_json").str.contains(re.escape(tag))


def parquet_ok_count(path: Path) -> int:
    if not path.exists():
        return 0
    frame = (
        pl.scan_parquet(path)
        .select((ok_row_expr()).sum().alias("ok_rows"))
        .collect()
    )
    return int(frame.item())


def parquet_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    frame = pl.scan_parquet(path).select(pl.len().alias("rows")).collect()
    return int(frame.item())


def select_input_parquet(requested: Path, fallback: Path, min_ok_rows: int) -> tuple[Path, dict[str, Any]]:
    requested_rows = parquet_row_count(requested)
    requested_ok = parquet_ok_count(requested)
    fallback_rows = parquet_row_count(fallback)
    fallback_ok = parquet_ok_count(fallback)
    if requested_ok >= min_ok_rows:
        selected = requested
        reason = "requested_input_meets_full_library_threshold"
    elif fallback_ok >= min_ok_rows:
        selected = fallback
        reason = "requested_input_below_full_library_threshold_selected_fallback"
    else:
        raise ValueError(
            "No full synthon library found: "
            f"{requested} ok_rows={requested_ok}, {fallback} ok_rows={fallback_ok}, "
            f"min_ok_rows={min_ok_rows}"
        )
    return selected, {
        "requested_input": str(requested),
        "requested_rows": requested_rows,
        "requested_ok_rows": requested_ok,
        "fallback_input": str(fallback),
        "fallback_rows": fallback_rows,
        "fallback_ok_rows": fallback_ok,
        "selected_input": str(selected),
        "selection_reason": reason,
    }


def count_tag(lf: pl.LazyFrame, reaction_id: str, role: str) -> int:
    frame = (
        lf.filter(ok_row_expr() & has_tag_expr(reaction_id, role))
        .select(pl.len().alias("count"))
        .collect()
    )
    return int(frame.item())


def count_bridge(lf: pl.LazyFrame, first_reaction_id: str, second_reaction_id: str) -> int:
    frame = (
        lf.filter(
            ok_row_expr()
            & has_tag_expr(first_reaction_id, "synthon")
            & has_tag_expr(second_reaction_id, "scaffold")
        )
        .select(pl.len().alias("count"))
        .collect()
    )
    return int(frame.item())


def estimate_runtime_hours(pair_count: int) -> float:
    return (float(pair_count) / float(BENCHMARK_PAIRS)) * BENCHMARK_SECONDS / 3600.0


def estimate_survivors(pair_count: int, survival_rate: float) -> int:
    return int(math.ceil(float(pair_count) * survival_rate))


def estimate_disk_gb(survivors: int, row_bytes: int) -> float:
    return float(survivors * row_bytes) / 1_000_000_000.0


def build_pathway_counts(
    lf: pl.LazyFrame,
    rules: Sequence[ReactionRule],
    scaffold_anchor_count: int,
    dihedral_samples: int,
    survival_rate: float,
    row_bytes: int,
) -> tuple[list[PathwayCount], dict[str, dict[str, int]]]:
    synthon_counts = {rule.reaction_id: count_tag(lf, rule.reaction_id, "synthon") for rule in rules}
    scaffold_counts = {rule.reaction_id: count_tag(lf, rule.reaction_id, "scaffold") for rule in rules}
    role_counts = {
        rule.reaction_id: {
            "synthon": synthon_counts[rule.reaction_id],
            "scaffold": scaffold_counts[rule.reaction_id],
        }
        for rule in rules
    }
    pathways: list[PathwayCount] = []
    for rule in rules:
        first_count = synthon_counts[rule.reaction_id]
        total_pairs = scaffold_anchor_count * first_count
        survivors = estimate_survivors(total_pairs, survival_rate)
        pathways.append(
            PathwayCount(
                pathway_id=f"ONE_STEP__{rule.reaction_id}",
                pathway_type="1-step",
                first_reaction_id=rule.reaction_id,
                second_reaction_id="",
                first_reaction_class=rule.reaction_class,
                second_reaction_class="",
                scaffold_anchor_count=scaffold_anchor_count,
                first_synthon_role=rule.synthon_role_label,
                bridge_scaffold_role="",
                terminal_synthon_role="",
                first_synthon_count=first_count,
                bifunctional_bridge_count=0,
                terminal_synthon_count=0,
                total_valid_pairs=total_pairs,
                estimated_rotamers=total_pairs * dihedral_samples,
                estimated_runtime_hours=estimate_runtime_hours(total_pairs),
                estimated_survivors=survivors,
                estimated_disk_gb=estimate_disk_gb(survivors, row_bytes),
            )
        )

    for first_rule in rules:
        for second_rule in rules:
            bridge_count = count_bridge(lf, first_rule.reaction_id, second_rule.reaction_id)
            terminal_count = synthon_counts[second_rule.reaction_id]
            if bridge_count == 0 or terminal_count == 0:
                continue
            total_pairs = scaffold_anchor_count * bridge_count * terminal_count
            survivors = estimate_survivors(total_pairs, survival_rate)
            pathways.append(
                PathwayCount(
                    pathway_id=f"TWO_STEP__{first_rule.reaction_id}__{second_rule.reaction_id}",
                    pathway_type="2-step",
                    first_reaction_id=first_rule.reaction_id,
                    second_reaction_id=second_rule.reaction_id,
                    first_reaction_class=first_rule.reaction_class,
                    second_reaction_class=second_rule.reaction_class,
                    scaffold_anchor_count=scaffold_anchor_count,
                    first_synthon_role=first_rule.synthon_role_label,
                    bridge_scaffold_role=second_rule.scaffold_role_label,
                    terminal_synthon_role=second_rule.synthon_role_label,
                    first_synthon_count=synthon_counts[first_rule.reaction_id],
                    bifunctional_bridge_count=bridge_count,
                    terminal_synthon_count=terminal_count,
                    total_valid_pairs=total_pairs,
                    estimated_rotamers=total_pairs * dihedral_samples,
                    estimated_runtime_hours=estimate_runtime_hours(total_pairs),
                    estimated_survivors=survivors,
                    estimated_disk_gb=estimate_disk_gb(survivors, row_bytes),
                )
            )
    return pathways, role_counts


def build_shards(pathways: Sequence[PathwayCount], shard_pairs: int, dihedral_samples: int) -> list[ShardRow]:
    shards: list[ShardRow] = []
    global_cursor = 0
    shard_id = 0
    for pathway in pathways:
        local_cursor = 0
        while local_cursor < pathway.total_valid_pairs:
            pair_count = min(shard_pairs, pathway.total_valid_pairs - local_cursor)
            shards.append(
                ShardRow(
                    shard_id=shard_id,
                    pathway_id=pathway.pathway_id,
                    pathway_type=pathway.pathway_type,
                    reaction_1=pathway.first_reaction_id,
                    reaction_2=pathway.second_reaction_id,
                    global_start_pair_idx=global_cursor,
                    global_end_pair_idx_exclusive=global_cursor + pair_count,
                    pathway_start_pair_idx=local_cursor,
                    pathway_end_pair_idx_exclusive=local_cursor + pair_count,
                    synthon_a_start_idx=local_cursor,
                    synthon_a_end_idx=local_cursor + pair_count,
                    synthon_b_start_idx=0,
                    synthon_b_end_idx=pair_count,
                    pair_count=pair_count,
                    estimated_pairs=pair_count,
                    estimated_rotamers=pair_count * dihedral_samples,
                    status="pending",
                )
            )
            shard_id += 1
            global_cursor += pair_count
            local_cursor += pair_count
    return shards


def git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    pl.DataFrame(list(rows)).write_parquet(tmp, compression="zstd")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    output_dir = cast(Path, args.output_dir)
    rules = load_reaction_rules(cast(Path, args.reaction_registry))
    selected_input, input_selection = select_input_parquet(
        cast(Path, args.input),
        cast(Path, args.fallback_input),
        int(args.full_input_min_ok_rows),
    )
    lf = pl.scan_parquet(selected_input)
    pathways, role_counts = build_pathway_counts(
        lf=lf,
        rules=rules,
        scaffold_anchor_count=int(args.scaffold_anchor_count),
        dihedral_samples=int(args.dihedral_samples),
        survival_rate=float(args.survival_rate),
        row_bytes=int(args.row_bytes),
    )
    if not pathways:
        raise ValueError("zero compatible dendritic pathways")

    shards = build_shards(pathways, int(args.shard_pairs), int(args.dihedral_samples))
    total_valid_pairs = sum(pathway.total_valid_pairs for pathway in pathways)
    total_rotamers = total_valid_pairs * int(args.dihedral_samples)
    total_survivors = estimate_survivors(total_valid_pairs, float(args.survival_rate))
    total_disk_gb = estimate_disk_gb(total_survivors, int(args.row_bytes))
    two_step_pairs = sum(pathway.total_valid_pairs for pathway in pathways if pathway.pathway_type == "2-step")
    one_step_pairs = sum(pathway.total_valid_pairs for pathway in pathways if pathway.pathway_type == "1-step")

    reaction_pair_counts_path = output_dir / "reaction_pair_counts.parquet"
    shard_plan_path = cast(Path | None, args.output_shards) or output_dir / "shard_plan.parquet"
    plan_path = cast(Path | None, args.output_plan) or output_dir / "vspace_38b_dendritic_plan.json"
    resume_path = cast(Path | None, args.output_resume) or output_dir / "resume_ledger.json"

    write_parquet(reaction_pair_counts_path, [asdict(pathway) for pathway in pathways])
    write_parquet(shard_plan_path, [asdict(shard) for shard in shards])

    generated_at = datetime.now(UTC).isoformat()
    plan: dict[str, Any] = {
        "gate": "DENDRITIC_38B_PLANNER_VALIDATED",
        "generated_at_utc": generated_at,
        "git_sha": git_sha(),
        "reaction_registry": str(cast(Path, args.reaction_registry)),
        "input_selection": input_selection,
        "enabled_reactions": [asdict(rule) for rule in rules],
        "role_counts": role_counts,
        "pathway_count": len(pathways),
        "one_step_pathway_count": sum(1 for pathway in pathways if pathway.pathway_type == "1-step"),
        "two_step_pathway_count": sum(1 for pathway in pathways if pathway.pathway_type == "2-step"),
        "one_step_pairs": one_step_pairs,
        "two_step_pairs": two_step_pairs,
        "total_valid_pairs": total_valid_pairs,
        "estimated_rotamers": total_rotamers,
        "benchmark_pairs": BENCHMARK_PAIRS,
        "benchmark_seconds_per_100k_pairs": BENCHMARK_SECONDS,
        "estimated_runtime_hours": estimate_runtime_hours(total_valid_pairs),
        "estimated_runtime_days": estimate_runtime_hours(total_valid_pairs) / 24.0,
        "survival_rate_assumption": float(args.survival_rate),
        "estimated_survivors": total_survivors,
        "parquet_row_bytes_assumption": int(args.row_bytes),
        "estimated_disk_gb": total_disk_gb,
        "estimated_disk_gib": float(total_survivors * int(args.row_bytes)) / float(1024**3),
        "shard_pair_count": int(args.shard_pairs),
        "shard_count": len(shards),
        "dihedral_samples": int(args.dihedral_samples),
        "scaffold_anchor_count": int(args.scaffold_anchor_count),
        "planning_artifacts": {
            "vspace_38b_dendritic_plan": str(plan_path),
            "reaction_pair_counts": str(reaction_pair_counts_path),
            "shard_plan": str(shard_plan_path),
            "resume_ledger": str(resume_path),
        },
        "assumptions": [
            "Counts are SMARTS-role-compatible virtual combinations, not guaranteed wet-lab syntheses.",
            "2-step pathways count Scaffold + bifunctional bridge synthon + terminal synthon combinations.",
            "Bifunctional bridge synthons are rows tagged as first reaction synthon and second reaction scaffold.",
            "No Rust V-space pruning was executed by this planner.",
            "Runtime estimate uses the supplied benchmark of 100,000 pairs in 14 seconds.",
        ],
    }
    write_json(plan_path, plan)

    resume_ledger: dict[str, Any] = {
        "gate": "DENDRITIC_38B_PLANNER_VALIDATED",
        "generated_at_utc": generated_at,
        "plan_path": str(plan_path),
        "shard_plan_path": str(shard_plan_path),
        "status": "not_started",
        "total_shards": len(shards),
        "completed_shards": [],
        "in_progress_shards": [],
        "failed_shards": [],
        "next_pending_shard_id": 0 if shards else None,
    }
    write_json(resume_path, resume_ledger)

    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
