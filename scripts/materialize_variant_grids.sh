#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="${1:-campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/signal_grid_variance_channel.parquet}"
OUT_DIR="${2:-campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale}"

PYTHONPATH=src python3 scripts/extract_embedded_variant_grids.py \
  --input "$INPUT" \
  --output-dir "$OUT_DIR" \
  --variants "A316T,T149M"

PYTHONPATH=src python3 scripts/validate_variant_grids.py \
  --wt-grid "$INPUT" \
  --variant-grid "$OUT_DIR/signal_grid_variance_channel_A316T.parquet" \
  --variant-grid "$OUT_DIR/signal_grid_variance_channel_T149M.parquet"
