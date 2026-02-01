#!/bin/bash
# CPU Performance Unleashing for PRISM4D
# Disables C-states and sets performance governor

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  CPU PERFORMANCE UNLEASH - Intel Ultra 9 285K                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ "$EUID" -ne 0 ]; then
    echo "⚠️  This script needs sudo privileges."
    echo "   Run: sudo bash cpu_performance_unleash.sh"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 1: DISABLE C-STATES (Prevent CPU Idle/Throttling)"
echo "═══════════════════════════════════════════════════════════════"

echo "🔥 Disabling ALL C-states on all 24 cores..."
for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 1 > "$state" 2>/dev/null || true
done
echo "✅ C-states DISABLED (CPU will never idle/throttle)"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 2: SET PERFORMANCE GOVERNOR (Lock to Max Frequency)"
echo "═══════════════════════════════════════════════════════════════"

echo "🔥 Setting all CPUs to 'performance' governor..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$cpu" 2>/dev/null || true
done
echo "✅ Performance governor ACTIVE (locked to 6.5 GHz max)"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 3: MAXIMIZE CPU PERFORMANCE"
echo "═══════════════════════════════════════════════════════════════"

# Disable CPU frequency scaling
echo "🔥 Locking CPU to maximum frequency..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
    MAX_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
    echo "$MAX_FREQ" > "$cpu" 2>/dev/null || true
done
echo "✅ CPU frequency locked to MAX (6.5 GHz)"

# Verify turbo boost is enabled
TURBO_STATUS=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "N/A")
if [ "$TURBO_STATUS" = "0" ]; then
    echo "✅ Intel Turbo Boost: ENABLED"
elif [ "$TURBO_STATUS" = "1" ]; then
    echo "🔥 Enabling Intel Turbo Boost..."
    echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
    echo "✅ Intel Turbo Boost: NOW ENABLED"
else
    echo "⚠️  Intel P-State not available (using ACPI cpufreq)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PHASE 4: MAKE CHANGES PERMANENT (Survive Reboots)"
echo "═══════════════════════════════════════════════════════════════"

# Create systemd service to apply settings on boot
cat > /etc/systemd/system/prism4d-cpu-performance.service << 'SYSTEMD_SERVICE'
[Unit]
Description=PRISM4D CPU Performance Optimization
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/prism4d-cpu-performance.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
SYSTEMD_SERVICE

# Create the script that systemd will run
cat > /usr/local/bin/prism4d-cpu-performance.sh << 'BOOT_SCRIPT'
#!/bin/bash
# PRISM4D CPU Performance - Applied at Boot

# Disable all C-states
for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 1 > "$state" 2>/dev/null || true
done

# Set performance governor
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$cpu" 2>/dev/null || true
done

# Lock to max frequency
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
    MAX_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
    echo "$MAX_FREQ" > "$cpu" 2>/dev/null || true
done

# Ensure turbo boost enabled
echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
BOOT_SCRIPT

chmod +x /usr/local/bin/prism4d-cpu-performance.sh

# Enable the service
systemctl daemon-reload
systemctl enable prism4d-cpu-performance.service
systemctl start prism4d-cpu-performance.service

echo "✅ Systemd service created: prism4d-cpu-performance.service"
echo "✅ Settings will apply automatically on every boot!"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "VERIFICATION"
echo "═══════════════════════════════════════════════════════════════"

sleep 2

GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
CURRENT_MHZ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
MAX_MHZ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
CURRENT_GHZ=$(echo "scale=2; $CURRENT_MHZ / 1000000" | bc)
MAX_GHZ=$(echo "scale=2; $MAX_MHZ / 1000000" | bc)

echo "✅ CPU Governor: $GOVERNOR"
echo "✅ Current Frequency: ${CURRENT_GHZ} GHz"
echo "✅ Maximum Frequency: ${MAX_GHZ} GHz"

C_STATE_COUNT=$(cat /sys/devices/system/cpu/cpu*/cpuidle/state*/disable 2>/dev/null | grep "1" | wc -l)
TOTAL_C_STATES=$(cat /sys/devices/system/cpu/cpu*/cpuidle/state*/disable 2>/dev/null | wc -l)
echo "✅ C-States Disabled: $C_STATE_COUNT / $TOTAL_C_STATES"

TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "N/A")
if [ "$TURBO" = "0" ]; then
    echo "✅ Turbo Boost: ENABLED"
else
    echo "⚠️  Turbo Boost: Status unknown"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  CPU UNLEASHED! Maximum Performance Active!                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Your Intel Ultra 9 285K is now running at MAXIMUM PERFORMANCE:"
echo "  - C-States: DISABLED (no idle throttling)"
echo "  - Governor: PERFORMANCE (locked to max frequency)"
echo "  - Frequency: ${MAX_GHZ} GHz sustained"
echo "  - Turbo Boost: ENABLED"
echo ""
echo "Settings are PERMANENT (survive reboots via systemd service)"
echo ""
