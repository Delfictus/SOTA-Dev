#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# PRISM Spike Watcher — Install Script
# ============================================================================
# Run on Prism4D machine as diddy (will sudo where needed).
#
# What this does:
#   1. Installs deps (pyarrow, inotify_simple)
#   2. Copies watcher script to scripts/
#   3. Installs systemd service
#   4. Enables and starts the daemon
#   5. Runs a dry-run test
#
# Usage:
#   chmod +x install_spike_watcher.sh
#   ./install_spike_watcher.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRISM_DIR="$HOME/Desktop/Prism4D-bio"
SCRIPTS_DIR="$PRISM_DIR/scripts"
SERVICE_NAME="prism-spike-watcher"

echo "=============================================="
echo "  PRISM Spike Watcher — Installer"
echo "=============================================="
echo ""

# ---- 1. Install Python deps ----
echo "[1/5] Installing Python dependencies..."
pip install pyarrow inotify_simple --break-system-packages --quiet 2>/dev/null || \
  pip install pyarrow inotify_simple --quiet

python3 -c "import pyarrow; print(f'  pyarrow {pyarrow.__version__}: OK')"
python3 -c "import inotify_simple; print('  inotify_simple: OK')"

# ---- 2. Copy watcher script ----
echo "[2/5] Installing watcher script..."
mkdir -p "$SCRIPTS_DIR"
cp "$SCRIPT_DIR/prism_spike_watcher.py" "$SCRIPTS_DIR/prism_spike_watcher.py"
chmod +x "$SCRIPTS_DIR/prism_spike_watcher.py"
echo "  Installed: $SCRIPTS_DIR/prism_spike_watcher.py"

# ---- 3. Verify rclone ----
echo "[3/5] Verifying rclone R2 connectivity..."
if ! command -v rclone &>/dev/null; then
  echo "  ERROR: rclone not found. Install it first."
  exit 1
fi

if rclone lsd r2: &>/dev/null; then
  echo "  rclone R2: OK"
  echo "  Buckets:"
  rclone lsd r2: 2>/dev/null | awk '{print "    " $NF}'
else
  echo "  ERROR: rclone cannot reach R2. Check ~/.config/rclone/rclone.conf"
  exit 1
fi

# ---- 4. Install systemd service ----
echo "[4/5] Installing systemd service..."
sudo cp "$SCRIPT_DIR/prism-spike-watcher.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
echo "  Service installed and enabled"

# ---- 5. Dry-run test ----
echo "[5/5] Running dry-run test..."
echo ""
python3 "$SCRIPTS_DIR/prism_spike_watcher.py" --foreground --dry-run --retroactive-only 2>&1 | head -50
echo ""
echo "=============================================="
echo "  INSTALL COMPLETE"
echo "=============================================="
echo ""
echo "Commands:"
echo "  Start:    sudo systemctl start $SERVICE_NAME"
echo "  Stop:     sudo systemctl stop $SERVICE_NAME"
echo "  Status:   sudo systemctl status $SERVICE_NAME"
echo "  Logs:     journalctl -u $SERVICE_NAME -f"
echo "  Dry-run:  python3 $SCRIPTS_DIR/prism_spike_watcher.py --foreground --dry-run"
echo "  Retro:    python3 $SCRIPTS_DIR/prism_spike_watcher.py --foreground --retroactive"
echo ""
echo "The daemon will:"
echo "  • Watch /mnt/storage/prism-outputs/runs/ and twin-runs/"
echo "  • Upload raw JSON + Parquet to R2 for every spike file"
echo "  • NEVER delete local files until BOTH are verified on R2"
echo "  • Auto-restart on failure"
echo "  • Run retroactive scan on every startup"
echo ""
echo "Start it now?"
read -p "  [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  sudo systemctl start "$SERVICE_NAME"
  sleep 2
  sudo systemctl status "$SERVICE_NAME" --no-pager
fi
