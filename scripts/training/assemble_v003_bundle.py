#!/usr/bin/env python3
"""
v003 training bundle assembler.

Combines per-target:
  - v5 PRISM-engine soft labels (parquet from prism_only_feature_extractor_v5.py)
  - structural v2 NPZ (PDB-derivable input features)
Into a unified per-target NPZ with:
  INPUTS (PDB-derivable, model sees at inference):
    - input_structural (n_res, 26) + input_structural_ext (n_res, 25) = 51
    - input_nma (n_res, 26) + input_nma_ext (n_res, 15) = 41
    - input_perturbed_nma (n_res, 5)
    - input_ca_xyz (n_res, 3) + input_sidechain_xyz (n_res, 3)
    - input_resname (n_res,) integer
    Total: ~103 PDB-derivable input dims/residue.
  TARGETS (engine-derived soft labels, model learns to predict):
    Grouped by signal source — preserves semantics for multi-head supervision.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


TARGET_GROUPS = {
    "kcc": (
        "kcc_score", "active_causal_steps", "burst_motion", "causal_lag",
        "direction_score", "lag_corr_peak", "local_cov", "motion_efficiency",
        "sum_motion", "total_steps", "net_dx_norm",
        "nearest_site_gtck", "nearest_kcc_confidence", "nearest_temporal_corr",
        "nearest_site_burst_motion", "nearest_site_causal_lag",
        "nearest_site_direction_score", "nearest_site_lag_corr_peak",
        "nearest_site_local_cov", "nearest_site_motion_efficiency",
    ),
    "therm": (
        "therm_class", "is_cryptic", "pocket_ccns_tau",
        "pocket_druggability", "pocket_hysteresis_asym",
        "nearest_pocket_dist", "max_transfer_entropy",
        "sum_transfer_entropy", "pocket_top_count",
    ),
    "asc": (
        "asc_s_pc", "asc_n_groups", "asc_in_consensus",
    ),
    "gcpid": (
        "gcpid_n_samples", "gcpid_redundancy_nats", "gcpid_synergy_nats",
        "gcpid_synergy_fraction", "gcpid_total_mi_nats",
        "gcpid_unique_a_nats", "gcpid_unique_b_nats",
    ),
    "phasors": (
        "phasor_mag_0", "phasor_mag_1", "phasor_mag_2", "phasor_mag_3",
        "phasor_phase_0", "phasor_phase_1", "phasor_phase_2", "phasor_phase_3",
        "phasor_count_0", "phasor_count_1", "phasor_count_2", "phasor_count_3",
        "phasor_coherence_01", "phasor_coherence_02", "phasor_coherence_03",
        "phasor_coherence_12", "phasor_coherence_13", "phasor_coherence_23",
        "phasor_phase_diff_01", "phasor_phase_diff_02", "phasor_phase_diff_03",
        "phasor_phase_diff_12", "phasor_phase_diff_13", "phasor_phase_diff_23",
        "phasor_mean_mag", "phasor_total_count",
        "phasor_scout_observer_coherence",
    ),
    "phase_manifold": (
        "max_phase_manifold_score", "top1_site_score",
        "top1_site_classification", "top1_site_centroid_view", "top1_site_rank",
        "is_in_top1_phase_site", "is_in_top5_phase_site", "is_in_top10_phase_site",
        "n_sites_containing_residue",
        "is_all_region", "n_sites_as_all_region",
        "is_lining", "n_sites_as_lining",
        "is_kcc_driver", "n_sites_as_kcc_driver",
        "is_hot_phase", "n_sites_as_hot_phase",
        "is_cold_phase", "n_sites_as_cold_phase",
        "is_burst_motion", "n_sites_as_burst_motion",
        "is_validation_contact", "n_sites_as_validation_contact",
    ),
    "stream": (
        "stream_entropy", "stream_dominant_id", "stream_max_fraction",
        "effective_n_streams", "scout_mean_spikes", "observer_mean_spikes",
        "scout_observer_contrast",
    ),
    "phase_bit": (
        "phase_bit_entropy", "ccns_phase_entropy",
        "ccns_dominant_phase", "ccns_max_fraction", "phase_popcount_mean",
    ),
    "druggability": (
        "residue_druggability_pdb", "residue_druggability_seen",
        "nearest_site_druggability", "nearest_site_classification",
    ),
    "ground_truth": (
        "gt_dist_to_ligand", "gt_in_contact_5A", "gt_in_contact_8A",
        "gt_has_ground_truth", "gt_valid_for_dcc", "gt_ligand_n_atoms",
    ),
    "p2rank": (
        "p2rank_score", "p2rank_zscore", "p2rank_probability",
        "p2rank_pocket", "p2rank_has_data",
    ),
}

TARGET_LEVEL_SCALARS = (
    "target_sdst_event_count", "target_total_pockets",
    "target_cryptic_pockets", "target_tide_residues_mapped",
    "target_n_streams_actual", "target_n_consensus_sites",
    "target_total_spikes", "target_n_streams_healthy",
    "kv_top1_vs_top2_separation", "kv_n_validated_sites",
    "kv_n_residues_tracked", "kv_n_residues_with_causal",
    "target_acl_mean", "target_acl_max", "target_acl_std", "target_acl_n_chunks",
    "gcpid_n_observer_groups", "gcpid_n_scout_groups", "gcpid_total_streams",
)


def get_cols(df: pl.DataFrame, names) -> np.ndarray:
    arrs = []
    for n in names:
        if n in df.columns:
            arrs.append(df[n].to_numpy())
        else:
            arrs.append(np.zeros(df.height, dtype=np.float32))
    return np.stack([a.astype(np.float32, copy=False) for a in arrs], axis=1)


def assemble_bundle(target: str, v5_parquet: Path, structural_npz: Optional[Path], output: Path) -> bool:
    df = pl.read_parquet(v5_parquet)
    n_res_engine = df.height
    n_res = n_res_engine

    bundle = {}

    if structural_npz and structural_npz.exists():
        s = np.load(structural_npz)
        n_res_struct = int(s["n_residues"])
        if n_res_struct != n_res_engine:
            print(f"  [warn] {target}: engine n_res={n_res_engine}, structural n_res={n_res_struct} — using min")
            n_res = min(n_res_engine, n_res_struct)
        for k in ("structural", "nma", "perturbed_nma", "structural_ext", "nma_ext",
                  "ca_xyz", "sidechain_xyz", "resname"):
            if k in s.files:
                arr = s[k]
                if arr.ndim == 1:
                    bundle[f"input_{k}"] = arr[:n_res]
                else:
                    bundle[f"input_{k}"] = arr[:n_res]
    else:
        n_res_struct = 0
        for k, shape_dim in (("structural", 26), ("nma", 26), ("perturbed_nma", 5),
                              ("structural_ext", 25), ("nma_ext", 15)):
            bundle[f"input_{k}"] = np.zeros((n_res, shape_dim), dtype=np.float32)
        bundle["input_ca_xyz"] = np.zeros((n_res, 3), dtype=np.float32)
        bundle["input_sidechain_xyz"] = np.zeros((n_res, 3), dtype=np.float32)
        bundle["input_resname"] = np.full(n_res, -1, dtype=np.int8)

    df = df.head(n_res)
    for group, cols in TARGET_GROUPS.items():
        arr = get_cols(df, cols)
        bundle[f"target_{group}"] = arr

    target_meta = {}
    for s in TARGET_LEVEL_SCALARS:
        if s in df.columns:
            v = df[s].to_numpy()
            if v.size > 0:
                target_meta[s] = float(v[0])
            else:
                target_meta[s] = 0.0
        else:
            target_meta[s] = 0.0
    bundle["target_metadata_keys"] = np.array(list(target_meta.keys()))
    bundle["target_metadata_values"] = np.array(list(target_meta.values()), dtype=np.float32)
    bundle["target_name"] = np.array([target])
    bundle["n_residues"] = np.int32(n_res)
    bundle["n_residues_structural"] = np.int32(n_res_struct)
    bundle["has_structural"] = np.int8(1 if structural_npz and structural_npz.exists() else 0)
    bundle["has_ground_truth"] = np.int8(int(target_meta.get("gt_has_ground_truth_first", 0)))

    if "gt_has_ground_truth" in df.columns:
        bundle["has_ground_truth"] = np.int8(1 if df["gt_has_ground_truth"][0] else 0)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **bundle)
    n_input_dims = sum(arr.shape[1] if arr.ndim > 1 else 1 for k, arr in bundle.items() if k.startswith("input_") and isinstance(arr, np.ndarray) and arr.dtype != np.dtype("O"))
    n_target_dims = sum(arr.shape[1] if arr.ndim > 1 else 1 for k, arr in bundle.items() if k.startswith("target_") and isinstance(arr, np.ndarray) and arr.dtype != np.dtype("O"))
    return True, n_input_dims, n_target_dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-dir", type=Path, required=True, help="dir with *_v5.parquet")
    ap.add_argument("--structural-dir", type=Path, required=True, help="dir with *_structural_v2.npz")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    v5_files = sorted(args.v5_dir.glob("*_v5.parquet"))
    ok = 0
    fail = 0
    for v5 in v5_files:
        name = v5.stem.replace("_v5", "")
        candidates = [
            args.structural_dir / f"{name}_structural_v2.npz",
            args.structural_dir / f"{re.sub(r'_[^_]+$', '', name)}_structural_v2.npz",
            args.structural_dir / f"{name.split('_')[0]}_structural_v2.npz",
        ]
        if name.split("_chainA_")[0] != name:
            candidates.append(args.structural_dir / f"{name.split('_chainA_')[0]}_chainA_structural_v2.npz")
        for tok_count in (4, 3, 2):
            tokens = name.split("_")
            if len(tokens) > tok_count:
                candidates.append(args.structural_dir / f"{'_'.join(tokens[:tok_count])}_structural_v2.npz")
        # Glob fallback: first NPZ that starts with first token
        first = name.split("_")[0]
        glob_match = sorted(args.structural_dir.glob(f"{first}*_structural_v2.npz"))
        candidates.extend(glob_match)
        struct_npz = None
        for c in candidates:
            if c.exists():
                struct_npz = c
                break

        out = args.output_dir / f"{name}_bundle.npz"
        try:
            success, n_in, n_tgt = assemble_bundle(name, v5, struct_npz, out)
            if success:
                sz = out.stat().st_size / 1024
                struct_flag = "✓" if struct_npz else "—"
                print(f"  {name:45s} input={n_in:3d} target={n_tgt:4d}  {sz:6.1f} KB  struct={struct_flag}")
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            fail += 1

    print(f"\n=== bundle assembly: {ok} ok, {fail} fail ===")


if __name__ == "__main__":
    main()
