#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMPAIGN_DIR="$REPO_ROOT/campaigns/glp1r_aleniglipron"
APP_DIR="$REPO_ROOT/apps/glp1r-teaser-visualizer"
VISUALIZER_TARGET="$CAMPAIGN_DIR/visualizer_app"
ARCHIVE="${PRISM_RELEASE_ARCHIVE:-$REPO_ROOT/PRISM_GLP1R_M3_FINAL_RELEASE_v2.0.tar.gz}"
SIGNATURE="$ARCHIVE.sha256"

cd "$APP_DIR"
npm run build

rm -rf "$VISUALIZER_TARGET"
mkdir -p "$VISUALIZER_TARGET"
cp -R "$APP_DIR/dist/." "$VISUALIZER_TARGET/"

cd "$REPO_ROOT"
python3 scripts/build_enterprise_positioning.py
python3 scripts/build_master_data_room.py
python3 scripts/generate_m3_dossier.py
python3 scripts/build_campaign_cbom.py

tmp_list="$(mktemp)"
trap 'rm -f "$tmp_list"' EXIT

find campaigns/glp1r_aleniglipron \
  -path '*/node_modules/*' -prune -o \
  -type f \
  ! -path 'campaigns/glp1r_aleniglipron/track_a_generative/fullscale_shards/shard_*.parquet' \
  \( -name '*.parquet' -o -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.sdf' -o -name '*.md' -o -name '*.yml' -o -name '*.yaml' -o -name '*.txt' -o -name '*.html' -o -name '*.js' -o -name '*.css' -o -name '*.wasm' \) \
  ! -name '.*.tmp.parquet' \
  ! -name 'PRISM_GLP1R_M2_Release_v1.0.tar.gz' \
  ! -name 'PRISM_GLP1R_M2_Release_v1.0.tar.gz.sha256' \
  ! -name 'PRISM_GLP1R_M2_EXECUTIVE_RELEASE_v1.1.tar.gz' \
  ! -name 'PRISM_GLP1R_M2_EXECUTIVE_RELEASE_v1.1.tar.gz.sha256' \
  ! -name 'PRISM_GLP1R_M3_FINAL_RELEASE_v2.0.tar.gz' \
  ! -name 'PRISM_GLP1R_M3_FINAL_RELEASE_v2.0.tar.gz.sha256' \
  | sort > "$tmp_list"

tar -czf "$ARCHIVE" --files-from "$tmp_list"
sha256sum "$ARCHIVE" > "$SIGNATURE"

echo "wrote $ARCHIVE"
echo "wrote $SIGNATURE"
