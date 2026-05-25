#!/usr/bin/env python3
"""Materialize the finalized first-10 chunk Track A trial action space."""

from __future__ import annotations

from pathlib import Path

import polars as pl


CHUNK_DIR = Path("campaigns/glp1r_aleniglipron/track_a_generative/anchors_3d")
OUTPUT = Path("campaigns/glp1r_aleniglipron/track_a_generative/trial_run/10k_trial_anchors_3d.parquet")
CHUNK_COUNT = 10


def finalized_chunk_paths() -> list[Path]:
    """Return the explicitly allowed, already-finalized chunk paths."""

    paths = [CHUNK_DIR / f"chunk_{index:04d}.parquet" for index in range(CHUNK_COUNT)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_text = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(f"missing finalized trial chunks: {missing_text}")
    return paths


def materialize_trial_space() -> int:
    """Read chunk_0000..chunk_0009, keep successful rows, and write the trial parquet."""

    lazy_frames = [pl.scan_parquet(path) for path in finalized_chunk_paths()]
    combined = pl.concat(lazy_frames, how="vertical_relaxed")
    schema = combined.collect_schema()
    if "status" not in schema:
        raise ValueError(f"trial anchor chunks must contain a status column; found {list(schema.names())}")

    successful = combined.filter(pl.col("status") == pl.lit("success"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    successful.sink_parquet(OUTPUT)
    row_count = int(pl.scan_parquet(OUTPUT).select(pl.len()).collect().item())
    return row_count


def main() -> int:
    row_count = materialize_trial_space()
    print(f"TRIAL_ACTION_SPACE={OUTPUT.as_posix()}")
    print(f"TRIAL_ACTION_SPACE_ROWS={row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
