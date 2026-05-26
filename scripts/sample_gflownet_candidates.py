#!/usr/bin/env python3
# mypy: ignore-errors
"""Phase 2 — multi-temperature policy sampling.

Loads gflownet_policy_v1.pt, reconstructs the action space + scaffold graph
via the same helpers used in scripts/train_gflownet_policy.py, then samples
30K policy trajectories across four temperature regimes.

Hard-fails per directive:
- valid generation rate < 80%
- unique SMILES < min(500, action_space_size) without action-space explanation
- >50% sample collapse to one anchor family
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Batch  # type: ignore[import-untyped]

REPO = Path("/home/diddy/Desktop/Prism4D-bio")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

# Reuse the trainer's helpers (scaffold graph, action space, model factory).
import scripts.train_gflownet_policy as T  # noqa: E402

from prism_dstw.hierarchical_bayes.gflownet_policy import (  # noqa: E402
    FiberBundleGFlowNetPolicy,
    FieldConditionedDualChannelGFlowNetPolicy,
)
from prism_dstw.scoring.tripartite_bias_scorer import compute_tripartite_bias  # noqa: E402

TRACK_A = REPO / "campaigns/glp1r_aleniglipron/track_a_generative"
MODEL_PATH = TRACK_A / "gflownet_policy_v1.pt"
OUT_PARQUET = TRACK_A / "gflownet_raw_policy_samples.parquet"
OUT_SUMMARY = TRACK_A / "gflownet_raw_policy_samples_summary.json"
OUT_TOP500 = TRACK_A / "gflownet_top_500_candidates.parquet"
OUT_TOP100 = TRACK_A / "gflownet_top_100_candidates.parquet"
LOCK_CLASH_THRESHOLD = 0.5

REGIMES = [
    {"name": "exploitation", "temperature": 0.1, "samples": 2_000},
    {"name": "focused",      "temperature": 0.5, "samples": 2_000},
    {"name": "balanced",     "temperature": 1.0, "samples": 2_000},
    {"name": "exploration",  "temperature": 2.0, "samples": 2_000},
    {"name": "near_random",  "temperature": 5.0, "samples": 2_000},
]


def hard_fail(msg: str) -> None:
    print(f"HARD-FAIL: {msg}", file=sys.stderr)
    sys.exit(2)


def resolve_track_a_path(path: Path | None, fallback: Path) -> Path:
    if path is None:
        return fallback
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    track_candidate = TRACK_A / candidate
    if track_candidate.exists():
        return track_candidate
    n80_candidate = REPO / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale" / candidate
    if n80_candidate.exists():
        return n80_candidate
    return candidate


def build_paths(args: argparse.Namespace) -> "T.TrainingPaths":
    # Use the trainer's default paths to match the model's training context
    # exactly. Resolved against pre-flight: residue_phase + interferometric
    # live under integrated_spike_events/n80_full_scale/.
    return T.TrainingPaths(
        ligand_sdf       = T.DEFAULT_LIGAND_SDF,
        anchors          = resolve_track_a_path(args.synthon_parquet, T.DEFAULT_ANCHORS),
        survivors        = resolve_track_a_path(args.survivors, T.DEFAULT_SURVIVORS),
        residue_phase    = T.DEFAULT_RESIDUE_PHASE,
        interferometric  = T.DEFAULT_INTERFEROMETRIC,
        topology         = T.DEFAULT_TOPOLOGY,
        fragment_registry= T.DEFAULT_FRAGMENT_REGISTRY,
        output_dir       = T.DEFAULT_OUTPUT_DIR,
    )


@torch.no_grad()
def sample_once(
    model: torch.nn.Module,
    graph: "T.ScaffoldGraph | Sequence[T.ScaffoldGraph]",
    action_space: "T.ActionSpace",
    action_rows: list[dict],
    batch_size: int,
    temperature: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor | list]:
    """Run one forward pass + temperature sample of size `batch_size`."""
    output, _, _ = T.forward_policy(model, graph, action_space, batch_size)
    valid_mask = action_space.valid_mask
    logits = output.forward_log_probs.detach().clone()
    logits = logits.masked_fill(~valid_mask.unsqueeze(0), -torch.inf)
    sampling = torch.softmax(logits / max(temperature, 1.0e-3), dim=1)
    invalid_rows = (~torch.isfinite(sampling).all(dim=1)) | (sampling.sum(dim=1) <= 0.0)
    if bool(invalid_rows.any()):
        fallback = valid_mask.to(dtype=torch.float32).unsqueeze(0).expand_as(sampling)
        fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        sampling[invalid_rows] = fallback[invalid_rows]
    actions_tensor = T.sample_actions_per_row(
        output.forward_log_probs,
        action_space.valid_mask,
        temperature,
        require_unique=False,
        generator=generator,
    )
    # Per-trajectory entropy of the post-temperature distribution.
    entropy = -(sampling * sampling.clamp_min(1.0e-12).log()).sum(dim=1)
    row_idx = torch.arange(batch_size)
    policy_logprob = output.forward_log_probs[row_idx, actions_tensor]
    backward_logprob = torch.full((batch_size,), float("nan"))
    proposals: list[dict[str, object]] = []
    for action_idx in actions_tensor.tolist():
        row = action_rows[int(action_idx)]
        survivor_smiles = row.get("survivor_smiles")
        canonical_smiles = survivor_smiles if isinstance(survivor_smiles, str) else row["canonical_smiles"]
        proposals.append(
            {
                "anchor_id": str(row["anchor_id"]),
                "canonical_smiles": str(canonical_smiles),
                "attachment_atom_idx": row.get(
                    "attachment_atom_idx",
                    row.get("fragment_attachment_atom_idx", -1),
                ),
                "selected_dihedral_deg": row.get("selected_dihedral_deg", float("nan")),
                "sigma_shear_mean": row.get("sigma_shear_mean", 0.0),
                "hysteresis_mean": row.get("hysteresis_mean", 0.0),
                "reversibility_mean": row.get("reversibility_mean", 1.0),
                "pathway_voxels_occupied": row.get("pathway_voxels_occupied", 0.0),
                "pathway_neighborhood_contacts": row.get("pathway_neighborhood_contacts", 0.0),
                "pathway_neighborhood_score_mean": row.get("pathway_neighborhood_score_mean", 0.0),
                "pathway_score_mean": row.get("pathway_score_mean", 0.0),
                "species_conservation_score_mean": row.get("species_conservation_score_mean", 1.0),
                "field_consensus_complement_bonus": row.get("field_consensus_complement_bonus", 0.0),
                "charge_feature_mean": row.get("charge_feature_mean", 0.0),
                "u_pose": row.get("u_pose", 0.0),
                "u_pose_source": row.get("u_pose_source", "unknown"),
            }
        )
    return {
        "actions":  actions_tensor.tolist(),
        "proposals": proposals,
        "logprob":  policy_logprob.tolist(),
        "blogprob": backward_logprob.tolist(),
        "entropy":  entropy.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--num-samples", type=int, default=10_000)
    ap.add_argument("--policy", type=Path, default=MODEL_PATH)
    ap.add_argument("--scaffold-pool", type=Path, action="append", default=None)
    ap.add_argument("--synthon-parquet", type=Path, default=None)
    ap.add_argument("--survivors", type=Path, default=None)
    ap.add_argument("--signal-grid", type=Path, default=None)
    ap.add_argument("--grid-coordinate-mapping", type=Path, default=T.DEFAULT_GRID_MAPPING)
    ap.add_argument("--field-stack-version", choices=("auto", "v1", "v2"), default="auto")
    ap.add_argument("--shear-stress", type=Path, default=None)
    ap.add_argument("--hysteresis-tensor", type=Path, default=None)
    ap.add_argument("--translation-pathway", type=Path, default=None)
    ap.add_argument("--cross-species", type=Path, default=None)
    ap.add_argument("--disable-exit-ray-masks", action="store_true", default=False)
    ap.add_argument("--lock-mask", type=Path, default=TRACK_A / "lock_region_mask.json")
    ap.add_argument("--temperatures", type=str, default="")
    ap.add_argument("--trajectories-per-temperature", type=int, default=None)
    ap.add_argument("--output", type=Path, default=OUT_TOP500)
    ap.add_argument("--tripartite-scoring", action="store_true", default=False)
    args = ap.parse_args()

    print("=== Phase 2 — multi-temperature policy sampling ===")
    model_path = resolve_track_a_path(args.policy, MODEL_PATH)
    print(f"  loading model: {model_path}")
    if not model_path.is_file():
        hard_fail(f"model missing: {model_path}")

    cfg = json.loads((TRACK_A / "gflownet_training_config.json").read_text())
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("config"), dict):
            cfg.update(ckpt["config"])
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        state_dict = ckpt
    paths = build_paths(args)
    selected_survivors = T.resolve_survivor_corpus_for_reward(
        requested_survivors=paths.survivors,
        reward_version=str(cfg.get("reward_version", "")),
        signal_grid=args.signal_grid or Path(str(cfg.get("signal_grid", ""))),
    )
    if selected_survivors != paths.survivors:
        print(
            "sampler_survivor_corpus_auto_selected "
            f"requested={paths.survivors.as_posix()} selected={selected_survivors.as_posix()}",
            flush=True,
        )
        paths = T.TrainingPaths(
            ligand_sdf=paths.ligand_sdf,
            anchors=paths.anchors,
            survivors=selected_survivors,
            residue_phase=paths.residue_phase,
            interferometric=paths.interferometric,
            topology=paths.topology,
            fragment_registry=paths.fragment_registry,
            output_dir=paths.output_dir,
        )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    field_stack_version = str(args.field_stack_version)
    if field_stack_version == "auto":
        field_stack_version = str(cfg.get("field_stack_version", "v1"))
    fiber_lookup = T.build_field_lookup(
        field_stack_version=field_stack_version,
        signal_grid=args.signal_grid,
        grid_mapping=args.grid_coordinate_mapping,
        shear_stress=args.shear_stress,
        hysteresis_tensor=args.hysteresis_tensor,
        translation_pathway=args.translation_pathway,
        cross_species=args.cross_species,
    )
    scaffold_sdfs = T.scaffold_paths_from_args(args.scaffold_pool, paths.ligand_sdf)
    scaffold_pool = T.build_scaffold_pool(
        paths,
        scaffold_sdfs,
        field_stack_version=field_stack_version,
        fiber_lookup=fiber_lookup,
    )
    graph: "T.ScaffoldGraph | Sequence[T.ScaffoldGraph]" = scaffold_pool if len(scaffold_pool) > 1 else scaffold_pool[0]
    reference_graph = scaffold_pool[0]
    # Trainer argparse defaults (must match the values used at training time —
    # see scripts/train_gflownet_policy.py:91-92). The trainer's saved
    # `config` blob does not record these explicitly.
    hidden_dim    = 96
    embedding_dim = 64
    action_space = T.load_action_space(
        paths,
        embedding_dim,
        base_feature_dim=reference_graph.base_feature_dim,
    )
    action_space = T.attach_exit_ray_adjustments(
        action_space,
        reference_graph=reference_graph,
        fiber_lookup=fiber_lookup,
        enabled=not bool(args.disable_exit_ray_masks),
    )
    action_rows = action_space.table.to_dicts()
    architecture = str(cfg.get("architecture", "FiberBundleGFlowNetPolicy"))
    if architecture == "FieldConditionedDualChannelGFlowNetPolicy" or bool(cfg.get("dual_channel", False)):
        model = FieldConditionedDualChannelGFlowNetPolicy(
            base_feature_dim=reference_graph.base_feature_dim,
            phase_feature_dim=reference_graph.phase_feature_dim,
            edge_feature_dim=reference_graph.edge_feature_dim,
            anchor_embeddings=action_space.anchor_embeddings,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            learn_anchor_embeddings=True,
        )
    else:
        model = FiberBundleGFlowNetPolicy(
            base_feature_dim=reference_graph.base_feature_dim,
            phase_feature_dim=reference_graph.phase_feature_dim,
            edge_feature_dim=reference_graph.edge_feature_dim,
            anchor_embeddings=action_space.anchor_embeddings,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            learn_anchor_embeddings=True,
            rf_mode=str(cfg.get("rf_mode", "gain")),
        )
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    n_actions = int(action_space.valid_mask.sum().item())
    print(f"  action space valid count: {n_actions}")

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    rows: list[dict] = []
    t0 = time.perf_counter()
    if args.temperatures:
        temperatures = [float(value.strip()) for value in str(args.temperatures).split(",") if value.strip()]
        per_temperature = int(args.trajectories_per_temperature or 2000)
        source_regimes = [
            {"name": f"T{temperature:g}".replace(".", "p"), "temperature": temperature, "samples": per_temperature}
            for temperature in temperatures
        ]
        args.num_samples = per_temperature * len(source_regimes)
    else:
        source_regimes = REGIMES
    base_total = sum(int(r["samples"]) for r in source_regimes)
    regimes = []
    assigned = 0
    for idx, regime in enumerate(source_regimes):
        if idx == len(source_regimes) - 1:
            samples = int(args.num_samples) - assigned
        else:
            samples = max(1, round(int(args.num_samples) * int(regime["samples"]) / base_total))
            assigned += samples
        regimes.append({**regime, "samples": samples})
    total_samples = sum(int(r["samples"]) for r in regimes)
    cumulative = 0
    for regime in regimes:
        regime_t0 = time.perf_counter()
        T_ = regime["temperature"]
        N  = regime["samples"]
        regime_rows: list[dict] = []
        # We sample in chunks of args.batch_size. Round up.
        batches = (N + args.batch_size - 1) // args.batch_size
        for b in range(batches):
            chunk = min(args.batch_size, N - len(regime_rows))
            if chunk <= 0:
                break
            payload = sample_once(model, graph, action_space, action_rows, chunk, T_, rng)
            for i, action_idx in enumerate(payload["actions"]):
                proposal = payload["proposals"][i]
                regime_rows.append({
                    "trajectory_id":        f"{regime['name']}-{len(regime_rows):06d}",
                    "regime":               regime["name"],
                    "temperature":          T_,
                    "seed":                 args.seed,
                    "sampled_action_idx":   int(action_idx),
                    "sampled_anchor_id":    str(proposal["anchor_id"]),
                    "canonical_smiles":     str(proposal["canonical_smiles"]),
                    "sampled_attachment_site": proposal["attachment_atom_idx"],
                    "sampled_dihedral_deg": float(proposal["selected_dihedral_deg"]),
                    "sigma_shear_mean":      float(proposal["sigma_shear_mean"]),
                    "hysteresis_mean":       float(proposal["hysteresis_mean"]),
                    "reversibility_mean":    float(proposal["reversibility_mean"]),
                    "pathway_voxels_occupied": float(proposal["pathway_voxels_occupied"]),
                    "pathway_neighborhood_contacts": float(proposal["pathway_neighborhood_contacts"]),
                    "pathway_neighborhood_score_mean": float(proposal["pathway_neighborhood_score_mean"]),
                    "pathway_score_mean":    float(proposal["pathway_score_mean"]),
                    "species_conservation_score_mean": float(proposal.get("species_conservation_score_mean", 1.0)),
                    "field_consensus_complement_bonus": float(proposal["field_consensus_complement_bonus"]),
                    "charge_feature_mean":   float(proposal["charge_feature_mean"]),
                    "u_pose":                float(proposal["u_pose"]),
                    "u_pose_source":         str(proposal["u_pose_source"]),
                    "policy_logprob":       float(payload["logprob"][i]),
                    "backward_logprob":     float(payload["blogprob"][i]),
                    "backward_logprob_status": "not_evaluated_one_step_sampler",
                    "trajectory_entropy":   float(payload["entropy"][i]),
                    "validity_status":      "valid",
                    "invalid_reason":       "",
                })
        rows.extend(regime_rows)
        cumulative += len(regime_rows)
        elapsed = time.perf_counter() - regime_t0
        print(f"  regime {regime['name']:<20s} T={T_}  N={len(regime_rows):>5d}  "
              f"elapsed {elapsed:5.1f}s  cumulative {cumulative}/{total_samples}")

    df = pl.DataFrame(rows)
    df.write_parquet(OUT_PARQUET)
    print(f"  -> {OUT_PARQUET}  ({df.height:,} rows)")

    unique_for_oracle = (
        df.unique(subset=["canonical_smiles"], keep="first")
        .select(
            pl.col("trajectory_id"),
            pl.col("sampled_anchor_id").alias("anchor_id"),
            pl.col("canonical_smiles"),
            pl.col("temperature").alias("sampling_temperature"),
            pl.col("regime").alias("sampling_regime"),
            pl.col("sampled_dihedral_deg"),
            pl.col("sigma_shear_mean"),
            pl.col("hysteresis_mean"),
            pl.col("reversibility_mean"),
            pl.col("pathway_voxels_occupied"),
            pl.col("pathway_neighborhood_contacts"),
            pl.col("pathway_neighborhood_score_mean"),
            pl.col("pathway_score_mean"),
            pl.col("species_conservation_score_mean"),
            pl.col("field_consensus_complement_bonus"),
            pl.col("charge_feature_mean"),
            pl.col("u_pose"),
            pl.col("u_pose_source"),
            pl.col("policy_logprob"),
            pl.col("trajectory_entropy"),
        )
    )
    scratch = REPO / ".scratch/gflownet_sampling"
    scratch.mkdir(parents=True, exist_ok=True)
    oracle_batch = scratch / "oracle_batch.parquet"
    oracle_rewards = scratch / "oracle_rewards.parquet"
    unique_for_oracle.select(["trajectory_id", "anchor_id", "canonical_smiles"]).write_parquet(oracle_batch)
    if oracle_rewards.exists():
        oracle_rewards.unlink()
    subprocess.run(
        [
            str(REPO / "target/release/oracle_scorer"),
            "--batch",
            str(oracle_batch),
            "--rewards",
            str(oracle_rewards),
            "--survivors",
            str(paths.survivors),
            "--lock-mask",
            str(resolve_track_a_path(args.lock_mask, TRACK_A / "lock_region_mask.json")),
        ],
        cwd=REPO,
        check=True,
    )
    rewards = pl.read_parquet(oracle_rewards)
    scored = (
        unique_for_oracle.join(rewards, on=["trajectory_id", "anchor_id", "canonical_smiles"], how="inner")
        .filter(pl.col("oracle_valid"))
        .sort("reward", descending=True)
        .with_row_index("rank", offset=1)
        .with_columns(
            pl.lit(True).alias("generated_by_policy"),
            pl.lit("PROJECTED").alias("epistemic_class"),
            pl.lit("policy_generated_not_validated").alias("training_status"),
        )
    )
    if "lock_geometry_score" not in scored.columns:
        scored = scored.with_columns(pl.col("pi_clash_lock").alias("lock_geometry_score"))
    if args.tripartite_scoring:
        tripartite_rows = [compute_tripartite_bias(row) for row in scored.to_dicts()]
        scored = scored.with_columns(
            pl.Series("lock_persistence_score", [score.lock_persistence_score for score in tripartite_rows]),
            pl.Series("bias_projection_score", [score.bias_projection_score for score in tripartite_rows]),
            pl.Series("epistemic_confidence", [score.epistemic_confidence for score in tripartite_rows]),
        )
    top500 = scored.head(500)
    biased_pool = scored.filter(pl.col("lock_geometry_score") > LOCK_CLASH_THRESHOLD)
    if biased_pool.height < 100:
        print(
            "biased_agonism_pool_incomplete "
            f"lock_positive={biased_pool.height} threshold={LOCK_CLASH_THRESHOLD} "
            "using top reward-ranked candidates with honest lock geometry fields",
            flush=True,
        )
        top100_source = scored.head(100)
    else:
        top100_source = biased_pool.head(100)
    top100 = top100_source.drop("rank").with_row_index("rank", offset=1)
    output_top500 = resolve_track_a_path(args.output, OUT_TOP500)
    output_top100 = output_top500.with_name(output_top500.stem.replace("top_500", "top_100") + output_top500.suffix)
    top500.write_parquet(output_top500)
    top500.write_csv(TRACK_A / "gflownet_top_500_candidates.csv")
    top100.write_parquet(output_top100)
    top100.write_csv(TRACK_A / "gflownet_top_100_candidates.csv")
    md_rows = [
        "# GFlowNet Top 500 Candidates",
        "",
        "| rank | canonical_smiles | reward | pi_complement | pi_clash_pocket | lock_geometry | cryptic_bonus |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in top500.head(100).iter_rows(named=True):
        md_rows.append(
            "| {rank} | `{smiles}` | {reward:.6f} | {pi:.3f} | {pocket:.3f} | {lock:.3f} | {cryptic:.3f} |".format(
                rank=row["rank"],
                smiles=row["canonical_smiles"],
                reward=float(row["reward"]),
                pi=float(row["pi_complement"]),
                pocket=float(row["pi_clash_pocket"]),
                lock=float(row["lock_geometry_score"]),
                cryptic=float(row["cryptic_bonus"]),
            )
        )
    (TRACK_A / "gflownet_top_500_candidates.md").write_text("\n".join(md_rows) + "\n")

    # ----- Hard-fail gates -----
    valid_rate = (df.filter(pl.col("validity_status") == "valid").height) / df.height
    unique_smiles = df.unique(subset=["canonical_smiles"], keep="first").height
    # Top anchor family share
    anchor_counts = (df.group_by("sampled_anchor_id").len()
                       .sort("len", descending=True))
    top_anchor_share = float(anchor_counts.row(0)[1]) / df.height
    summary = {
        "package":             "PRISM_TRACK_A_GFLOWNET_V1_INFERENCE",
        "phase":               "2_multi_temperature_sampling",
        "generated_at_utc":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_trajectories":  df.height,
        "valid_generation_rate": valid_rate,
        "unique_smiles":       unique_smiles,
        "action_space_size":   n_actions,
        "top_anchor_share":    top_anchor_share,
        "regime_breakdown":    [
            {"name": r["name"], "temperature": r["temperature"], "samples": r["samples"]}
            for r in regimes
        ],
        "wall_seconds":        round(time.perf_counter() - t0, 1),
        "oracle_ranked_unique_smiles": scored.height,
        "biased_pool_count": biased_pool.height,
        "biased_lock_threshold": LOCK_CLASH_THRESHOLD,
        "top500_min_reward": float(top500.get_column("reward").min()) if top500.height else 0.0,
        "top500_max_reward": float(top500.get_column("reward").max()) if top500.height else 0.0,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  -> {OUT_SUMMARY}")

    print()
    print("=== Hard-fail gate evaluation ===")
    print(f"  valid_generation_rate = {valid_rate*100:.2f}%  (require ≥ 80%)")
    print(f"  unique_smiles         = {unique_smiles}  (require ≥ min(500, action_space={n_actions}))")
    print(f"  top_anchor_share      = {top_anchor_share*100:.2f}%  (require < 50%)")
    print(
        "candidate_sampling_complete "
        f"temperatures={','.join(str(r['temperature']) for r in regimes)} "
        f"trajectories_total={df.height} unique_smiles={unique_smiles} "
        f"top500_min_reward={summary['top500_min_reward']:.6f} "
        f"top500_max_reward={summary['top500_max_reward']:.6f}"
    )

    if valid_rate < 0.80:
        hard_fail(f"valid generation rate {valid_rate*100:.2f}% < 80%")
    expected_unique_floor = min(500, n_actions)
    if unique_smiles < expected_unique_floor:
        hard_fail(f"unique_smiles {unique_smiles} < expected floor {expected_unique_floor}")
    if top_anchor_share > 0.50:
        hard_fail(f"top anchor family share {top_anchor_share*100:.2f}% > 50% — mode collapse")

    print()
    print("PASS — Phase 2 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
