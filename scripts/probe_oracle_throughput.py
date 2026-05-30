#!/usr/bin/env python3
"""Pre-Phase-2 throughput probe.

Runs one oracle batch via BatchedRustOracle and times the round-trip so we
can extrapolate to 30K samples and the multi-pose / multi-dihedral rescoring
load before committing the operator's compute.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import polars as pl

# Add repo src to path so we can import the prism_dstw package.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_dstw.orchestration.rust_reward_oracle import (  # noqa: E402
    BatchedRustOracle,
    OracleProposal,
)

SURVIVORS = REPO / "campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_real512_o3a_zmatrix.parquet"
ORACLE_BIN = REPO / "target/release/oracle_scorer"


async def probe(batch_size: int) -> dict[str, float | int]:
    survivors_df = pl.read_parquet(SURVIVORS)
    # Pick `batch_size` distinct rows (the survivor corpus has unique SMILES already).
    if "canonical_smiles" not in survivors_df.columns:
        raise SystemExit(f"survivors parquet missing canonical_smiles column. "
                         f"columns: {survivors_df.columns}")
    if "anchor_id" not in survivors_df.columns:
        anchor_col = next((c for c in ("anchor_id", "source_anchor_id", "anchor_idx") if c in survivors_df.columns), None)
        if anchor_col is None:
            raise SystemExit(f"no anchor column in survivors. columns: {survivors_df.columns}")
    else:
        anchor_col = "anchor_id"

    sample = survivors_df.unique(subset=["canonical_smiles"], keep="first").head(batch_size)
    proposals = [
        OracleProposal(
            anchor_id=str(row[anchor_col]),
            canonical_smiles=str(row["canonical_smiles"]),
            trajectory_id=f"probe-{i:04d}",
        )
        for i, row in enumerate(sample.iter_rows(named=True))
    ]
    if len(proposals) != batch_size:
        print(f"  WARN: requested {batch_size} but only got {len(proposals)} unique SMILES")
    oracle = BatchedRustOracle(
        oracle_binary=ORACLE_BIN,
        survivor_corpus=SURVIVORS,
        max_batch_size=batch_size,
    )
    t0 = time.perf_counter()
    result = await oracle.score_batch(proposals)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    tel = result.telemetry
    return {
        "batch_size":        len(proposals),
        "wall_ms":           wall_ms,
        "oracle_latency_ms": tel.oracle_latency_ms,
        "rust_scoring_ms":   tel.rust_scoring_time_ms,
        "parquet_write_ms":  tel.parquet_write_ms,
        "parquet_read_ms":   tel.parquet_read_ms,
        "reward_mean":       tel.reward_mean,
        "reward_std":        tel.reward_std,
        "invalid_count":     tel.invalid_reward_count,
        "duplicate_count":   tel.duplicate_smiles_count,
    }


async def main() -> int:
    print("=== oracle throughput probe ===")
    for bs in (8, 32, 64):
        try:
            stats = await probe(bs)
            print(f"  bs={bs:3d}  wall={stats['wall_ms']:7.1f}ms  "
                  f"rust={stats['rust_scoring_ms']:7.1f}ms  "
                  f"write={stats['parquet_write_ms']:5.1f}ms  "
                  f"read={stats['parquet_read_ms']:5.1f}ms  "
                  f"reward μ={stats['reward_mean']:.3f} σ={stats['reward_std']:.3f}  "
                  f"invalid={stats['invalid_count']}")
        except Exception as ex:  # noqa: BLE001
            print(f"  bs={bs}  FAIL: {ex}")
            import traceback
            traceback.print_exc()
            return 1

    print()
    print("=== extrapolation ===")
    # Re-time with bs=64 (the trainer's batch size) for the final estimate.
    stats = await probe(64)
    per_call = stats["wall_ms"]
    per_sample = per_call / 64
    samples_30k = 30_000
    rescoring_factor = 10  # 5 poses × 2 dihedral grids
    sample_time_min = (samples_30k * per_sample) / 1000 / 60
    rescore_time_min = (samples_30k * per_sample * rescoring_factor) / 1000 / 60
    # Real rescore uses unique candidates only; expect ~500-2000 unique.
    rescore_low_min  = (500  * per_sample * rescoring_factor) / 1000 / 60
    rescore_high_min = (2000 * per_sample * rescoring_factor) / 1000 / 60
    print(f"  per oracle call (bs=64): {per_call:.1f} ms")
    print(f"  per-sample amortized:    {per_sample:.2f} ms")
    print(f"  30K samples:             {sample_time_min:.1f} min  (Phase 2)")
    print(f"  rescoring 30K × 10:      {rescore_time_min:.1f} min  (Phase 3 worst case)")
    print(f"  rescoring 500 unique × 10:  {rescore_low_min:.1f} min  (Phase 3 likely low)")
    print(f"  rescoring 2000 unique × 10: {rescore_high_min:.1f} min  (Phase 3 likely high)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
