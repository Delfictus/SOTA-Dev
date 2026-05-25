#!/usr/bin/env python3
"""Generate tripartite candidate profiles and per-candidate markdown dossiers."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import polars as pl
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

from prism_dstw.pharmacophore.bias_pharmacophore import assess_bias_pharmacophore
from prism_dstw.scoring.tripartite_bias_scorer import compute_reward_v2, compute_tripartite_bias


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_INPUT = TRACK_A / "gflownet_top_100_candidates_lockmask_rescored.parquet"
DEFAULT_OUTPUT_DIR = TRACK_A / "candidate_dossiers"
DEFAULT_PROFILE_PARQUET = TRACK_A / "gflownet_top_50_tripartite_profiles.parquet"
DEFAULT_REPORT = TRACK_A / "gflownet_top_50_tripartite_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidates", type=Path, default=None, help="Alias for --input used by Epoch 016.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile-parquet", type=Path, default=DEFAULT_PROFILE_PARQUET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--tripartite", action="store_true", default=False)
    parser.add_argument(
        "--full-field-stack",
        action="store_true",
        default=False,
        help="Include Epoch 021 shear, hysteresis, pathway, charge, species, and u_pose fields when present.",
    )
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.candidates) if args.candidates is not None else Path(args.input)
    frame = pl.read_parquet(input_path).head(int(args.top_n))
    rows = [profile_for_row(row, rank=index + 1) for index, row in enumerate(frame.iter_rows(named=True))]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        write_candidate_files(output_dir, row)

    profile_path = Path(args.profile_parquet)
    tmp_profile = profile_path.with_suffix(profile_path.suffix + ".tmp")
    pl.DataFrame(rows).write_parquet(tmp_profile)
    tmp_profile.replace(profile_path)

    report = {
        "schema_version": "PRISM.candidate_dossiers.tripartite.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "profile_parquet": str(profile_path),
        "candidate_count": len(rows),
        "lock_positive": sum(1 for row in rows if float(row["lock_geometry_score"]) > 0.0),
        "full_field_stack": bool(args.full_field_stack),
        "full_field_metrics": full_field_report(rows),
        "confidence_counts": confidence_counts(rows),
        "epistemic_note": (
            "L1-L3 are static/proxy confidence tags. They do not represent GPU MD confirmation."
        ),
    }
    atomic_write_json(Path(args.report), report)
    if args.output_summary is not None:
        summary_path = Path(args.output_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_markdown(rows, report), encoding="utf-8")
    print(
        "candidate_dossiers_generated "
        f"count={len(rows)} lock_positive={report['lock_positive']} "
        f"profile_parquet={profile_path} output_dir={output_dir}"
    )
    return 0


def profile_for_row(row: Mapping[str, Any], *, rank: int) -> dict[str, Any]:
    smiles = str(row.get("canonical_smiles", row.get("smiles", "")))
    bias = compute_tripartite_bias(row)
    pharmacophore = assess_bias_pharmacophore(
        smiles,
        coordinates_json=row.get("coordinates_json") if isinstance(row.get("coordinates_json"), str) else None,
    )
    medchem = medchem_properties(smiles)
    reward_v2 = compute_reward_v2(row, bias)
    bald_value = bald_information_value(bias.epistemic_confidence, bias.bias_projection_score, bias.lock_geometry_score)
    return {
        "rank": rank,
        "candidate_id": stable_candidate_id(smiles, rank),
        "canonical_smiles": smiles,
        "reward": float_value(row.get("reward", 0.0)),
        "reward_v2_tripartite": reward_v2,
        "pi_complement": float_value(row.get("pi_complement", 0.0)),
        "pi_clash_pocket": float_value(row.get("pi_clash_pocket", 0.0)),
        "sigma_shear": float_value(row.get("sigma_shear", 0.0)),
        "lock_geometry_score": bias.lock_geometry_score,
        "lock_geometry_atoms": bias.lock_geometry_atoms,
        "lock_persistence_score": bias.lock_persistence_score,
        "lock_hysteresis_asymmetry": bias.lock_hysteresis_asymmetry,
        "bias_projection_score": bias.bias_projection_score,
        "intracellular_penetration_depth": bias.intracellular_penetration_depth,
        "projected_tm6_blockade_volume": bias.projected_tm6_blockade_volume,
        "epistemic_confidence": bias.epistemic_confidence,
        "tripartite_bias_json": json.dumps(asdict(bias), sort_keys=True),
        "pharmacophore_json": json.dumps(asdict(pharmacophore), sort_keys=True),
        "pharmacophore_matches_required": pharmacophore.matches_required,
        "bald_information_value": bald_value,
        "gpu_dispatch_status": "pending" if bias.lock_geometry_score > 0.0 else "not_lock_positive",
        "sigma_shear_mean": float_value(row.get("sigma_shear_mean", row.get("sigma_shear", 0.0))),
        "hysteresis_mean": float_value(row.get("hysteresis_mean", row.get("lock_hysteresis_asymmetry", 0.0))),
        "reversibility_mean": float_value(row.get("reversibility_mean", 1.0)),
        "pathway_voxels_occupied": float_value(row.get("pathway_voxels_occupied", 0.0)),
        "pathway_neighborhood_contacts": float_value(row.get("pathway_neighborhood_contacts", 0.0)),
        "pathway_neighborhood_score_mean": float_value(row.get("pathway_neighborhood_score_mean", 0.0)),
        "pathway_score_mean": float_value(row.get("pathway_score_mean", 0.0)),
        "charge_feature_mean": float_value(row.get("charge_feature_mean", 0.0)),
        "u_pose": float_value(row.get("u_pose", 0.0)),
        "u_pose_source": str(row.get("u_pose_source", "unknown")),
        "species_selectivity_score": float_value(row.get("species_selectivity_score", 0.0)),
        "predicted_active_in": str(row.get("predicted_active_in", "")),
        **medchem,
    }


def write_candidate_files(output_dir: Path, row: Mapping[str, Any]) -> None:
    candidate_id = str(row["candidate_id"])
    json_path = output_dir / f"{candidate_id}.json"
    md_path = output_dir / f"{candidate_id}.md"
    atomic_write_json(json_path, dict(row))
    md_path.write_text(markdown_for_profile(row), encoding="utf-8")


def markdown_for_profile(row: Mapping[str, Any]) -> str:
    return (
        f"# Candidate {row['rank']}: {row['candidate_id']}\n\n"
        f"- SMILES: `{row['canonical_smiles']}`\n"
        f"- Reward: `{float(row['reward']):.4f}`\n"
        f"- Tripartite reward v2: `{float(row['reward_v2_tripartite']):.4f}`\n"
        f"- Observed lock geometry score: `{float(row['lock_geometry_score']):.4f}`\n"
        f"- Derived lock persistence: `{float(row['lock_persistence_score']):.4f}`\n"
        f"- Projected bias score: `{float(row['bias_projection_score']):.4f}`\n"
        f"- Epistemic confidence: `{row['epistemic_confidence']}`\n"
        f"- BALD information value: `{float(row['bald_information_value']):.4f}`\n\n"
        "## Full Thermodynamic Field Stack\n\n"
        f"- Shear stress mean: `{float(row['sigma_shear_mean']):.4f}`\n"
        f"- Hysteresis mean: `{float(row['hysteresis_mean']):.4f}`\n"
        f"- Reversibility mean: `{float(row['reversibility_mean']):.4f}`\n"
        f"- Direct activation pathway voxels occupied: `{float(row['pathway_voxels_occupied']):.1f}`\n"
        f"- Activation pathway-neighborhood contacts: `{float(row['pathway_neighborhood_contacts']):.1f}`\n"
        f"- AM1-BCC charge feature mean: `{float(row['charge_feature_mean']):.4f}`\n"
        f"- u_pose penalty: `{float(row['u_pose']):.4f}` ({row['u_pose_source']})\n"
        f"- Species selectivity score: `{float(row['species_selectivity_score']):.4f}`\n"
        f"- Predicted active species: `{row['predicted_active_in']}`\n\n"
        "## Med Chem\n\n"
        f"- MW: `{float(row['mw']):.2f}`\n"
        f"- TPSA: `{float(row['tpsa']):.2f}`\n"
        f"- HBD/HBA: `{int(row['hbd'])}/{int(row['hba'])}`\n"
        f"- Rotatable bonds: `{int(row['rotatable_bonds'])}`\n"
        f"- cLogP: `{float(row['clogp']):.2f}`\n"
        f"- QED: `{float(row['qed']):.3f}`\n"
        f"- Oral compliant: `{bool(row['oral_compliant'])}`\n\n"
        "## Provenance\n\n"
        "The lock geometry field is the corrected residue-mask static overlap. "
        "Projected bias is an inference and requires GPU MD validation before promotion.\n"
    )


def medchem_properties(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "mw": 0.0,
            "tpsa": 0.0,
            "hbd": 0,
            "hba": 0,
            "rotatable_bonds": 0,
            "clogp": 0.0,
            "qed": 0.0,
            "oral_compliant": False,
        }
    mw = float(cast(Any, Descriptors).MolWt(mol))
    tpsa = float(cast(Any, rdMolDescriptors).CalcTPSA(mol))
    hbd = int(cast(Any, Descriptors).NumHDonors(mol))
    hba = int(cast(Any, Descriptors).NumHAcceptors(mol))
    rotatable = int(cast(Any, rdMolDescriptors).CalcNumRotatableBonds(mol))
    clogp = float(cast(Any, Crippen).MolLogP(mol))
    qed = float(cast(Any, QED).qed(mol))
    oral = mw <= 650.0 and tpsa <= 160.0 and hbd <= 5 and hba <= 12 and rotatable <= 12
    return {
        "mw": mw,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rotatable,
        "clogp": clogp,
        "qed": qed,
        "oral_compliant": oral,
    }


def bald_information_value(confidence: str, projection: float, lock_geometry: float) -> float:
    uncertainty = {"L1": 0.75, "L2": 0.50, "L3": 0.25}.get(confidence, 0.75)
    return uncertainty * max(lock_geometry, 0.0) * (1.0 - abs(projection - 0.5) * 2.0)


def confidence_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["epistemic_confidence"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def full_field_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "mean_shear": 0.0,
            "mean_hysteresis": 0.0,
            "mean_reversibility": 0.0,
            "pathway_contact_count": 0,
            "mean_u_pose": 0.0,
            "mean_species_selectivity": 0.0,
        }
    n_rows = float(len(rows))
    return {
        "mean_shear": sum(float(row["sigma_shear_mean"]) for row in rows) / n_rows,
        "mean_hysteresis": sum(float(row["hysteresis_mean"]) for row in rows) / n_rows,
        "mean_reversibility": sum(float(row["reversibility_mean"]) for row in rows) / n_rows,
        "pathway_contact_count": sum(
            1
            for row in rows
            if float(row["pathway_voxels_occupied"]) > 0.0
            or float(row["pathway_neighborhood_contacts"]) > 0.0
        ),
        "mean_u_pose": sum(float(row["u_pose"]) for row in rows) / n_rows,
        "mean_species_selectivity": sum(float(row["species_selectivity_score"]) for row in rows) / n_rows,
    }


def summary_markdown(rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> str:
    lines = [
        "# M3 Candidate Summary",
        "",
        f"Generated candidates: `{report['candidate_count']}`",
        f"Corrected lock-positive candidates: `{report['lock_positive']}`",
        "",
        "| rank | candidate | reward v2 | lock geometry | projection | confidence |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in rows[:50]:
        lines.append(
            "| {rank} | `{candidate_id}` | {reward:.3f} | {lock:.3f} | {projection:.3f} | {confidence} |".format(
                rank=int(row["rank"]),
                candidate_id=str(row["candidate_id"]),
                reward=float(row["reward_v2_tripartite"]),
                lock=float(row["lock_geometry_score"]),
                projection=float(row["bias_projection_score"]),
                confidence=str(row["epistemic_confidence"]),
            )
        )
    lines.extend(
        [
            "",
            "## Full Field Stack",
            "",
            "| rank | shear | hysteresis | reversibility | direct pathway | pathway neighborhood | charge mean | u_pose | species selectivity |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows[:50]:
        lines.append(
            "| {rank} | {shear:.3f} | {hyst:.4f} | {rev:.3f} | {path:.1f} | {neighborhood:.1f} | {charge:.4f} | {upose:.3f} | {species:.3f} |".format(
                rank=int(row["rank"]),
                shear=float(row["sigma_shear_mean"]),
                hyst=float(row["hysteresis_mean"]),
                rev=float(row["reversibility_mean"]),
                path=float(row["pathway_voxels_occupied"]),
                neighborhood=float(row["pathway_neighborhood_contacts"]),
                charge=float(row["charge_feature_mean"]),
                upose=float(row["u_pose"]),
                species=float(row["species_selectivity_score"]),
            )
        )
    lines.extend(
        [
            "",
            "Epistemic note: projected bias values are inference-layer signals and remain pending GPU MD or wet-lab falsification.",
            "",
        ]
    )
    return "\n".join(lines)


def stable_candidate_id(smiles: str, rank: int) -> str:
    digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:8]
    return f"cand_{rank:03d}_{digest}"


def float_value(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float | str):
        return float(value)
    return 0.0


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
