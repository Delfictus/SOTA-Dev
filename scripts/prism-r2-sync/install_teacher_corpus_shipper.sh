#!/usr/bin/env bash
set -euo pipefail

PRISM_DIR="${PRISM_DIR:-$HOME/Desktop/Prism4D-bio}"
SERVICE_NAME="prism-teacher-corpus-shipper"
BUCKET="${PRISM_TEACHER_SPIKE_BUCKET:-prism-teacher-spikes-20260516}"
ROOT="${PRISM_TEACHER_CORPUS_ROOT:-/mnt/storage/prism-outputs/teacher-corpus}"

cd "$PRISM_DIR"

echo "[1/6] Building Rust shipper"
cargo build -p prism-spike-shipper --release

echo "[2/6] Preparing local corpus root: $ROOT"
mkdir -p "$ROOT" /mnt/storage/prism-outputs/.spike-shipper-parquet-cache

echo "[3/6] Ensuring R2 bucket exists: r2:$BUCKET"
if rclone lsd "r2:$BUCKET" >/dev/null 2>&1; then
  echo "  bucket exists"
else
  if command -v wrangler >/dev/null 2>&1; then
    set -a
    # shellcheck disable=SC1090
    source "$HOME/.config/prism/credentials.env"
    set +a
    CLOUDFLARE_API_TOKEN="$CLOUDFLARE_API_TOKEN" wrangler r2 bucket create "$BUCKET"
  else
    rclone mkdir "r2:$BUCKET"
  fi
  echo "  bucket create requested"
fi

echo "[4/6] Installing R2-only env file and systemd service"
(
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.config/prism/credentials.env"
  set +a
  umask 077
  {
    printf 'R2_ACCESS_KEY_ID=%s\n' "$R2_ACCESS_KEY_ID"
    printf 'R2_SECRET_ACCESS_KEY=%s\n' "$R2_SECRET_ACCESS_KEY"
    printf 'R2_ENDPOINT=%s\n' "$R2_ENDPOINT"
  } > "$HOME/.config/prism/r2.env"
)
sudo cp "$PRISM_DIR/scripts/prism-r2-sync/prism-teacher-corpus-shipper.service" \
  /etc/systemd/system/prism-teacher-corpus-shipper.service
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo "[5/6] Dry-run inventory"
"$PRISM_DIR/target/release/prism-spike-shipper" \
  --no-default-watch-dirs \
  --route "$ROOT=teacher-corpus/raw-spikes/v1" \
  --spike-bucket "$BUCKET" \
  --archive-bucket prism-archive \
  --archive-prefix teacher-corpus/raw-spike-archive \
  --inventory-only

echo "[6/6] Starting service"
sudo systemctl restart "$SERVICE_NAME"
sudo systemctl status "$SERVICE_NAME" --no-pager
