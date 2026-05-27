#!/usr/bin/env python3
"""Build the Epoch 024 v2 thermodynamic motif registry."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
import polars as pl
import torch
from rdkit import Chem
from torch import Tensor, nn

from prism_dstw.motif.causal_attribution import FiberBatch, extract_causal_motifs
from prism_dstw.motif.functional_groups import (
    AtomThermoAnnotation,
    ThermodynamicFunctionalGroup,
    classify_atom_roles,
    extract_tfg_with_neighborhood,
)
from prism_dstw.motif.phase_resolved_mcs import PhaseResolvedMCS, extract_phase_resolved_mcs
from prism_dstw.motif.registry import MotifEntry, MotifRegistry, ThermodynamicRole, motif_id_for_smarts
from prism_dstw.motif.synthon_ancestry import SynthonAncestry, compute_synthon_ancestry

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN / "track_a_generative"
TRACK_B = CAMPAIGN / "track_b_chronological"
DEFAULT_CANDIDATES = TRACK_A / "gflownet_top_100_candidates.parquet"
DEFAULT_OUTPUT_DIR = TRACK_B / "motif_intelligence"


class ProxyAttributionPolicy(nn.Module):
    """Small differentiable proxy used to compute IG over candidate fibers."""

    def forward(self, x_phase: Tensor) -> Tensor:
        atom_signal = x_phase.mean(dim=(1, 2))
        lock_logit = atom_signal.mean().reshape(1)
        complement_logit = x_phase[:, :, 0].mean().reshape(1)
        return torch.stack([lock_logit, complement_logit], dim=0).reshape(1, 2)


class CandidateFiberBatch:
    """Minimal FiberBatch implementation for CAME extraction."""

    def __init__(
        self,
        *,
        x_phase: Tensor,
        mol: Chem.Mol,
        trajectory_step: int,
        scaffold_id: str,
        reaction_class: str,
    ) -> None:
        self.x_phase = x_phase
        self.mol = mol
        self.trajectory_step = trajectory_step
        self.scaffold_id = scaffold_id
        self.reaction_class = reaction_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mcs-timeout-seconds", type=int, default=10)
    parser.add_argument("--max-mcs-candidates", type=int, default=100)
    parser.add_argument("--came-candidates", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_candidate_frame(Path(args.candidates))
    mol_rows = molecule_rows(frame)
    if not mol_rows:
        raise RuntimeError(f"no valid candidate molecules parsed from {args.candidates}")
    registry_path = output_dir / "thermodynamic_motif_registry.parquet"
    registry = MotifRegistry(registry_path)
    tfg_entries, tfg_count = build_tfg_entries(mol_rows)
    came_entries, came_count = build_came_entries(mol_rows[: int(args.came_candidates)])
    mcs_entries, mcs_count = build_mcs_entries(
        mol_rows[: int(args.max_mcs_candidates)],
        timeout_seconds=int(args.mcs_timeout_seconds),
    )
    sad_entries, sad_count = build_sad_entries(frame)

    for entry in [*tfg_entries, *came_entries, *mcs_entries, *sad_entries]:
        registry.register(entry)
    entries = registry.all()
    completeness_mean_value = float(np.mean([entry.completeness_score for entry in entries])) if entries else 0.0
    report = {
        "schema_version": "PRISM.thermodynamic_motif_registry_report.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_path": str(args.candidates),
        "registry_path": str(registry_path),
        "candidate_count": len(mol_rows),
        "motif_count": len(entries),
        "method_raw_counts": {
            "TFGD": tfg_count,
            "CAME": came_count,
            "PR_MCS": mcs_count,
            "SAD": sad_count,
        },
        "method_registry_counts": _counts(entry.discovery_method for entry in entries),
        "role_counts": _counts(entry.thermodynamic_role for entry in entries),
        "completeness_mean": completeness_mean_value,
        "lock_enriched_count": len(registry.query_lock_enriched(min_enrichment=1.5)),
        "came_integrated_gradients": True,
        "came_attention_weights_used": False,
        "pr_mcs_timeout_seconds": int(args.mcs_timeout_seconds),
        "claim_boundary": "Motifs are DERIVED from runtime candidate/reward artifacts; no projected field is labeled OBSERVED.",
    }
    report_path = output_dir / "thermodynamic_motif_registry_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "thermodynamic_motif_registry_built "
        f"motifs={report['motif_count']} completeness_mean={report['completeness_mean']:.3f} "
        f"tfgd={tfg_count} came={came_count} pr_mcs={mcs_count} sad={sad_count} "
        f"registry={registry_path} report={report_path}"
    )
    motif_count = len(entries)
    if motif_count < 20 or completeness_mean_value <= 0.5:
        raise RuntimeError("motif registry gate failed: need >=20 motifs and completeness_mean > 0.5")
    return 0


def load_candidate_frame(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path)
    if "canonical_smiles_rdkit" in frame.columns:
        return frame.with_columns(pl.coalesce(["canonical_smiles_rdkit", "canonical_smiles"]).alias("motif_smiles"))
    if "canonical_smiles" in frame.columns:
        return frame.with_columns(pl.col("canonical_smiles").alias("motif_smiles"))
    if "track_b_smiles" in frame.columns:
        return frame.with_columns(pl.col("track_b_smiles").alias("motif_smiles"))
    raise ValueError("candidate parquet has no usable SMILES column")


def molecule_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dicts():
        smiles = str(row.get("motif_smiles") or "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        rows.append({"row": row, "mol": mol, "smiles": smiles})
    return rows


def build_tfg_entries(mol_rows: Sequence[dict[str, Any]]) -> tuple[list[MotifEntry], int]:
    grouped: dict[str, list[ThermodynamicFunctionalGroup]] = defaultdict(list)
    for item in mol_rows:
        mol = item["mol"]
        row = item["row"]
        annotations = annotations_for_candidate(mol, row)
        roles = classify_atom_roles(annotations)
        for tfg in extract_tfg_with_neighborhood(mol, roles, annotations, parent_smiles=str(item["smiles"])):
            grouped[tfg.smarts].append(tfg)
    entries: list[MotifEntry] = []
    for smarts, tfgs in grouped.items():
        first = tfgs[0]
        n_occ = len(tfgs)
        phase = np.stack([tfg.phase_profile for tfg in tfgs]).mean(axis=0).tolist()
        lock = float(np.mean([tfg.lock_geometry_contribution for tfg in tfgs]))
        entry = MotifEntry(
            motif_id=motif_id_for_smarts(smarts),
            canonical_smarts=smarts,
            discovery_method="TFGD",
            thermodynamic_role=valid_role(first.role),
            pi_complement_contribution=float(np.mean([tfg.pi_complement_sum for tfg in tfgs])),
            pi_clash_contribution=float(np.mean([tfg.pi_clash_sum for tfg in tfgs])),
            lock_geometry_contribution=lock,
            shear_stress_mean=float(np.mean([tfg.shear_stress_mean for tfg in tfgs])),
            phase_profile=[float(value) for value in phase],
            hysteresis_score=hysteresis_score(phase),
            consensus_resilience=float(np.mean([tfg.consensus_resilience for tfg in tfgs])),
            n_occurrences_top100=n_occ,
            n_occurrences_lock_positive=sum(1 for tfg in tfgs if tfg.lock_geometry_contribution > 0.0),
            enrichment_ratio=1.0 + lock,
            first_seen_epoch=24,
            last_seen_epoch=24,
            confidence="L3",
            provenance="DERIVED",
            parent_smiles=sorted({smi for tfg in tfgs for smi in tfg.parent_smiles})[:20],
        ).with_completeness()
        entries.append(entry)
    return entries, sum(len(v) for v in grouped.values())


def build_came_entries(mol_rows: Sequence[dict[str, Any]]) -> tuple[list[MotifEntry], int]:
    batches: list[FiberBatch] = []
    actions: list[int] = []
    for item in mol_rows:
        mol = item["mol"]
        row = item["row"]
        annotations = annotations_for_candidate(mol, row)
        x_phase = torch.tensor([annotation_tensor(annotations[idx]) for idx in range(mol.GetNumAtoms())], dtype=torch.float32)
        batches.append(
            CandidateFiberBatch(
                x_phase=x_phase,
                mol=mol,
                trajectory_step=int(row.get("track_b_action_index") or row.get("rank") or 0),
                scaffold_id=str(row.get("anchor_id") or "unknown_scaffold"),
                reaction_class="inferred_from_action",
            )
        )
        actions.append(0)
    motifs = extract_causal_motifs(ProxyAttributionPolicy(), batches, actions, n_steps=24)
    entries = [
        MotifEntry(
            motif_id=motif_id_for_smarts(motif.smarts),
            canonical_smarts=motif.smarts,
            discovery_method="CAME",
            thermodynamic_role="LOCK_WEDGE" if motif.causal_direction == "PROMOTES_LOCK" else "MIXED",
            lock_geometry_contribution=motif.attribution_score_mean if motif.causal_direction == "PROMOTES_LOCK" else None,
            shear_stress_mean=motif.attribution_score_std,
            phase_profile=[float(motif.attribution_score_mean)] * 5,
            hysteresis_score=0.0,
            consensus_resilience=max(0.0, 1.0 - float(motif.attribution_score_std)),
            variant_resilience={"WT_proxy": max(0.0, 1.0 - float(motif.attribution_score_std))},
            attribution_score_mean=motif.attribution_score_mean,
            attribution_score_std=motif.attribution_score_std,
            causal_direction=motif.causal_direction,
            channel_a_ratio=0.5,
            channel_b_ratio=0.5,
            n_occurrences_top100=motif.frequency,
            enrichment_ratio=1.5 if motif.causal_direction == "PROMOTES_LOCK" else 1.0,
            confidence="L3",
            provenance="DERIVED",
        ).with_completeness()
        for motif in motifs
    ]
    return entries, len(motifs)


def build_mcs_entries(mol_rows: Sequence[dict[str, Any]], *, timeout_seconds: int) -> tuple[list[MotifEntry], int]:
    mols = [item["mol"] for item in mol_rows]
    profiles = [phase_profile_from_row(item["row"]) for item in mol_rows]
    motifs = extract_phase_resolved_mcs(
        mols,
        profiles,
        tanimoto_threshold=0.3,
        mcs_timeout_seconds=timeout_seconds,
        butina_cutoff=0.8,
    )
    entries = [
        entry_from_mcs(motif)
        for motif in motifs
    ]
    return entries, len(motifs)


def build_sad_entries(frame: pl.DataFrame) -> tuple[list[MotifEntry], int]:
    source_col = "motif_smiles"
    enamine_col = "anchor_id" if "anchor_id" in frame.columns else "motif_smiles"
    sad_frame = frame.with_columns(
        pl.col(source_col).cast(pl.Utf8).alias("synthon_smiles"),
        pl.col(enamine_col).cast(pl.Utf8).alias("enamine_id"),
        pl.lit("inferred_from_trajectory").alias("reaction_class"),
        pl.lit(0).alias("exit_vector_idx"),
    )
    ancestries = compute_synthon_ancestry(sad_frame, min_occurrences=1)
    entries = [entry_from_sad(item) for item in ancestries[:100]]
    return entries, len(ancestries)


def annotations_for_candidate(mol: Chem.Mol, row: Mapping[str, Any]) -> dict[int, AtomThermoAnnotation]:
    n_atoms = max(mol.GetNumAtoms(), 1)
    pi_comp = float_value(row, "pi_complement", "legacy_pi_complement") / n_atoms
    pi_clash = float_value(row, "pi_clash_pocket", "legacy_pi_clash_pocket", "adjusted_pi_clash") / n_atoms
    lock = float_value(row, "lock_geometry_score", "legacy_lock_geometry_score", "pi_clash_lock") / n_atoms
    shear = float_value(row, "sigma_shear", "sigma_shear_mean") / n_atoms
    consensus = float_value(row, "consensus_complement_bonus", "field_consensus_complement_bonus")
    phase = phase_profile_from_row(row)
    annotations: dict[int, AtomThermoAnnotation] = {}
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        hetero = 1.0 if atom.GetAtomicNum() not in {1, 6} else 0.0
        terminal_scale = 1.0 if atom.GetDegree() <= 1 else 0.5
        phase_tuple = (
            float(phase[0] / n_atoms),
            float(phase[1] / n_atoms),
            float(phase[2] / n_atoms),
            float(phase[3] / n_atoms),
            float(phase[4] / n_atoms),
        )
        annotations[atom_idx] = AtomThermoAnnotation(
            pi_complement=pi_comp * (1.0 + hetero + aromatic),
            pi_clash=pi_clash * (1.0 + terminal_scale),
            lock_geometry=lock * (1.0 + terminal_scale),
            shear_stress=shear * (1.0 + aromatic),
            phase_profile=phase_tuple,
            channel_a_activation=aromatic + hetero,
            channel_b_activation=float(atom.GetDegree()),
            consensus_resilience=consensus,
        )
    return annotations


def phase_profile_from_row(row: Mapping[str, Any]) -> np.ndarray:
    keys = [
        ("pi_clash_lock_cold_hold", "legacy_pi_clash_lock_cold_hold", "lock_occupancy_cold_hold"),
        ("pi_clash_lock_ramp_up", "legacy_pi_clash_lock_ramp_up", "lock_occupancy_ramp_up"),
        ("pi_clash_lock_warm_hold", "legacy_pi_clash_lock_warm_hold", "lock_occupancy_warm_hold"),
        ("pi_clash_lock_ramp_down", "legacy_pi_clash_lock_ramp_down", "lock_occupancy_ramp_down"),
        ("pi_clash_lock_cold_return", "legacy_pi_clash_lock_cold_return", "lock_occupancy_cold_return"),
    ]
    values = [float_value(row, *names) for names in keys]
    if not any(values):
        base = float_value(row, "pi_complement", "legacy_pi_complement", "reward")
        shear = float_value(row, "sigma_shear", "sigma_shear_mean")
        values = [base, base + 0.05 * shear, base + 0.1 * shear, base + 0.05 * shear, base]
    return np.array(values, dtype=np.float64)


def annotation_tensor(annotation: AtomThermoAnnotation) -> list[list[float]]:
    phases = list(annotation.phase_profile)
    return [
        [
            phases[i],
            annotation.pi_complement,
            annotation.pi_clash,
            annotation.lock_geometry,
            annotation.shear_stress,
            annotation.channel_a_activation,
            annotation.channel_b_activation,
            annotation.consensus_resilience,
        ]
        for i in range(5)
    ]


def entry_from_mcs(motif: PhaseResolvedMCS) -> MotifEntry:
    return MotifEntry(
        motif_id=motif_id_for_smarts(motif.smarts),
        canonical_smarts=motif.smarts,
        discovery_method="PR_MCS",
        thermodynamic_role="MIXED",
        phase_profile=[float(value) for value in motif.phase_profile_centroid],
        hysteresis_score=motif.hysteresis_score,
        consensus_resilience=motif.variant_resilience_mean,
        worst_case_resilience=motif.variant_resilience_worst,
        is_evolutionary_invariant=motif.is_evolutionary_invariant,
        n_occurrences_top100=motif.n_molecules,
        enrichment_ratio=max(1.0, motif.tanimoto_cohesion),
        confidence="L3",
        provenance="DERIVED",
    ).with_completeness()


def entry_from_sad(item: SynthonAncestry) -> MotifEntry:
    mol = Chem.MolFromSmiles(item.synthon_smiles)
    smarts = Chem.MolToSmarts(mol) if mol is not None else "[*]"
    return MotifEntry(
        motif_id=motif_id_for_smarts(smarts),
        canonical_smarts=smarts,
        discovery_method="SAD",
        thermodynamic_role="LOCK_WEDGE" if item.enrichment_ratio > 1.0 else "NEUTRAL",
        lock_geometry_contribution=item.lock_positive_rate,
        phase_profile=[float(item.lock_positive_rate)] * 5,
        hysteresis_score=0.0,
        consensus_resilience=item.lock_positive_rate,
        variant_resilience={"WT_candidate_set": item.lock_positive_rate},
        worst_case_variant="WT_candidate_set",
        worst_case_resilience=item.lock_positive_rate,
        synthon_sources=[item.enamine_id],
        reaction_classes=[item.reaction_class],
        synthetic_accessibility=5.0,
        exit_vector_preference=item.exit_vector_preference,
        n_occurrences_top100=int(round(item.lock_positive_rate * 100.0)),
        n_occurrences_lock_positive=int(round(item.lock_positive_rate * 100.0)),
        enrichment_ratio=item.enrichment_ratio,
        confidence="L3",
        provenance="DERIVED",
    ).with_completeness()


def hysteresis_score(phase: Sequence[float]) -> float:
    if len(phase) != 5:
        return 0.0
    return abs(float(phase[4]) - float(phase[0])) / max(abs(float(phase[1])), 1.0e-8)


def float_value(row: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float | str):
            try:
                parsed = float(value)
            except ValueError:
                continue
            if math.isfinite(parsed):
                return parsed
    return 0.0


def valid_role(role: str) -> ThermodynamicRole:
    allowed: set[str] = {
        "COMPLEMENT_ANCHOR",
        "CLASH_DRIVER",
        "LOCK_WEDGE",
        "SHEAR_SENTINEL",
        "PHASE_PIVOT",
        "NEUTRAL",
        "MIXED",
    }
    return cast(ThermodynamicRole, role if role in allowed else "MIXED")


def _counts(values: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
