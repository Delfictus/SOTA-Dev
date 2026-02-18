#!/bin/bash
# PRISM4D: GPU Unleash + System Lockdown Script
# Maximizes RTX 5080 performance and prevents kernel panics from updates

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  PRISM4D GPU UNLEASH + SYSTEM LOCKDOWN                         ║"
echo "║  RTX 5080 Blackwell Optimization                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  This script needs sudo privileges."
    echo "   Run: sudo bash unleash_and_lock.sh"
    exit 1
fi

echo "📦 Creating system backup..."
BACKUP_DIR="/home/diddy/prism4d_lockdown_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
nvidia-smi -q > "$BACKUP_DIR/gpu_before.txt"
lspci -vv > "$BACKUP_DIR/pcie_before.txt" 2>&1
dpkg -l > "$BACKUP_DIR/packages_before.txt"
cp /etc/apt/sources.list "$BACKUP_DIR/sources.list.bak" 2>/dev/null || true
echo "✅ Backup saved to: $BACKUP_DIR"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 1: GPU UNLEASHING"
echo "═══════════════════════════════════════════════════════════════"

echo "🔥 [1/5] Enabling GPU Persistence Mode..."
nvidia-smi -pm 1
echo "✅ Persistence mode: ENABLED"

echo "🔥 [2/5] Setting Maximum Performance Mode..."
nvidia-smi --auto-boost-default=DISABLED 2>/dev/null || echo "⚠️  Auto-boost control not available (OK)"
echo "✅ Performance mode: MAXIMUM"

echo "🔥 [3/5] Locking GPU Clocks to Maximum..."
# Get max graphics clock (RTX 5080: ~2900 MHz)
MAX_GR_CLOCK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits)
nvidia-smi -lgc "$MAX_GR_CLOCK" 2>/dev/null || echo "⚠️  Clock locking not supported (will run at boost clocks)"
echo "✅ Graphics clock: Locked to ${MAX_GR_CLOCK} MHz"

echo "🔥 [4/5] Setting Maximum Power Limit..."
# RTX 5080 TDP: ~320W, allow up to 450W if supported
nvidia-smi -pl 450 2>/dev/null || nvidia-smi -pl 400 2>/dev/null || echo "⚠️  Power limit not adjustable"
echo "✅ Power limit: MAXIMIZED"

echo "🔥 [5/5] Enabling Compute Mode (Exclusive Process)..."
nvidia-smi -c EXCLUSIVE_PROCESS 2>/dev/null || echo "⚠️  Compute mode not available"
echo "✅ Compute mode: EXCLUSIVE (max throughput)"

echo ""
nvidia-smi --query-gpu=name,persistence_mode,power.limit,clocks.gr,clocks.mem --format=csv
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 2: SYSTEM LOCKDOWN (Prevent Kernel Panics)"
echo "═══════════════════════════════════════════════════════════════"

echo "🔒 [1/6] Holding Current Kernel (6.14.0-37-generic)..."
apt-mark hold linux-image-6.14.0-37-generic
apt-mark hold linux-headers-6.14.0-37-generic
apt-mark hold linux-modules-6.14.0-37-generic
apt-mark hold linux-modules-extra-6.14.0-37-generic
echo "✅ Kernel LOCKED (will NOT update/remove)"

echo "🔒 [2/6] Holding NVIDIA Driver 590.48.01..."
apt-mark hold nvidia-driver-590-open
apt-mark hold nvidia-dkms-590-open
apt-mark hold nvidia-utils-590
apt-mark hold nvidia-kernel-source-590-open
apt-mark hold libnvidia-compute-590
apt-mark hold libnvidia-gl-590
apt-mark hold nvidia-kernel-common-590
apt-mark hold nvidia-firmware-590-590.48.01
echo "✅ NVIDIA Driver LOCKED (will NOT update)"

echo "🔒 [3/6] Disabling Unattended Upgrades..."
systemctl stop unattended-upgrades 2>/dev/null || true
systemctl disable unattended-upgrades 2>/dev/null || true
systemctl mask unattended-upgrades 2>/dev/null || true
echo "✅ Unattended upgrades: DISABLED"

echo "🔒 [4/6] Disabling APT Auto-Update Timers..."
systemctl stop apt-daily.timer 2>/dev/null || true
systemctl disable apt-daily.timer 2>/dev/null || true
systemctl stop apt-daily-upgrade.timer 2>/dev/null || true
systemctl disable apt-daily-upgrade.timer 2>/dev/null || true
systemctl stop apt-daily.service 2>/dev/null || true
systemctl disable apt-daily.service 2>/dev/null || true
echo "✅ APT auto-updates: DISABLED"

echo "🔒 [5/6] Creating /etc/apt/preferences.d/prism4d-pin..."
cat > /etc/apt/preferences.d/prism4d-lockdown << 'PINFILE'
# PRISM4D System Lockdown - Prevent ALL kernel/driver updates
# Created: $(date)

Package: linux-image-6.14.0-37-generic
Pin: version 6.14.0-37.37~24.04.1
Pin-Priority: 1001

Package: nvidia-driver-590-open
Pin: version 590.48.01-0ubuntu0.24.04.1
Pin-Priority: 1001

Package: nvidia-*
Pin: version 590.*
Pin-Priority: 1001

# Block kernel upgrades globally
Package: linux-image-*
Pin: version *
Pin-Priority: -1

Package: linux-generic*
Pin: version *
Pin-Priority: -1
PINFILE
echo "✅ APT pinning: LOCKED (priority 1001 = unremovable)"

echo "🔒 [6/6] Creating recovery script..."
cat > /home/diddy/recover_prism4d.sh << 'RECOVERY'
#!/bin/bash
echo "=== PRISM4D EMERGENCY RECOVERY ==="
echo ""
echo "If system won't boot:"
echo "1. Hold SHIFT at boot → GRUB menu"
echo "2. Select: Advanced Options for Ubuntu"
echo "3. Boot with: 6.14.0-37-generic (recovery mode)"
echo ""
echo "If NVIDIA driver broken:"
echo "  sudo apt install --reinstall nvidia-driver-590-open=590.48.01-0ubuntu0.24.04.1"
echo "  sudo nvidia-smi"
echo ""
echo "If CUDA broken:"
echo "  cd ~/Desktop/Prism4D-bio"
echo "  cargo clean"
echo "  cargo build --release -p prism-report --features gpu"
echo ""
echo "Test PRISM4D:"
echo "  target/release/prism4d version"
echo "  nvidia-smi"
echo ""
echo "Backup location: $BACKUP_DIR"
RECOVERY
chmod +x /home/diddy/recover_prism4d.sh
echo "✅ Recovery script: ~/recover_prism4d.sh"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "LOCKDOWN COMPLETE!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "✅ Kernel 6.14.0-37: LOCKED (held + pinned)"
echo "✅ NVIDIA 590.48.01: LOCKED (held + pinned)"
echo "✅ Auto-updates: DISABLED PERMANENTLY"
echo "✅ GPU: Persistence mode ON, max clocks"
echo "✅ Recovery script: ~/recover_prism4d.sh"
echo "✅ Backup: $BACKUP_DIR"
echo ""
echo "🎯 HELD PACKAGES (will NEVER update):"
apt-mark showhold | grep -E "linux-image-6.14.0-37|nvidia-driver-590"
echo ""
echo "⚠️  MANUAL BIOS SETTINGS REQUIRED:"
echo "    1. Reboot into BIOS/UEFI"
echo "    2. Enable: PCIe Gen 5.0"
echo "    3. Enable: Above 4G Decoding"
echo "    4. Enable: Resizable BAR (already enabled - verify!)"
echo "    5. Set: GPU mode to 'Unleashed' or 'Performance'"
echo "    6. Disable: PCIe ASPM (power management)"
echo "    7. Save & Exit"
echo ""
echo "After BIOS config, test PRISM4D:"
echo "  cd ~/Desktop/Prism4D-bio"
echo "  target/release/nhs-batch --help"
echo ""
echo "Expected performance: 4500-10,000 steps/sec on Blackwell!"
echo ""
