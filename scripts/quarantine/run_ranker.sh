#!/usr/bin/env bash
set -u
mode="${1:-}"
case "$mode" in
  single|md_asym) ;;
  twin)
    echo "ERROR: twin run wrote coupled_spikes.json (no Arrow, no site_id, no ccns_phase)." >&2
    echo "       The phase-manifold ranker cannot run on twin output without an adapter." >&2
    exit 2
    ;;
  *)
    echo "usage: $0 {single|md_asym}" >&2
    exit 1
    ;;
esac

ROOT="/home/diddy/Desktop/Prism4D-bio"
BASE="/home/diddy/Desktop/TEST 5-8/6OIM/${mode}"
PREFIX="6oim_chainA"

ARROW="${BASE}/${PREFIX}.topology.spike_events.arrow"
SITES="${BASE}/${PREFIX}.binding_sites.json"
KCC="${BASE}/${PREFIX}.kcc_visualization.json"
OUT="${BASE}/phase_manifold_ranked"

for f in "$ARROW" "$SITES" "$KCC"; do
  if [ ! -e "$f" ]; then
    echo "MISSING: $f" >&2
    exit 3
  fi
done

mkdir -p "$OUT"

cd "$ROOT" || exit 4
exec python3 scripts/phase_manifold_ranker.py \
    --arrow "$ARROW" \
    --binding-sites "$SITES" \
    --kcc "$KCC" \
    --outdir "$OUT"
