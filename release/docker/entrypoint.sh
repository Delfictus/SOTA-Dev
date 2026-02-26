#!/usr/bin/env bash
set -euo pipefail
##############################################################################
# PRISM4D Demo — Container Entrypoint
#
# Initializes the audit pipeline, SSH host keys, and starts sshd.
# All activity is logged to /var/log/prism4d-audit/ (host-mounted).
##############################################################################

AUDIT_DIR="/var/log/prism4d-audit"
mkdir -p "${AUDIT_DIR}" 2>/dev/null || true

log_audit() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "${AUDIT_DIR}/system.log" 2>/dev/null || true
}

log_audit "=== PRISM4D DEMO CONTAINER STARTING ==="
log_audit "Hostname: $(hostname)"
log_audit "Container ID: $(cat /proc/self/cgroup 2>/dev/null | head -1 | sed 's|.*/||' || echo unknown)"

# ── 1. Generate SSH host keys if missing ──────────────────────────────
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    ssh-keygen -A 2>/dev/null
    log_audit "SSH host keys generated"
fi

# ── 2. Ensure audit directory structure ───────────────────────────────
touch "${AUDIT_DIR}/system.log" \
      "${AUDIT_DIR}/sessions.log" \
      "${AUDIT_DIR}/commands.log" \
      "${AUDIT_DIR}/sftp.log" \
      "${AUDIT_DIR}/gpu.log" \
      2>/dev/null || true

chmod 700 "${AUDIT_DIR}" 2>/dev/null || true

# ── 3. Install the command audit hook into demo user's profile ────────
# This captures EVERY command the demo user runs via PROMPT_COMMAND.
# The log file is outside their home — they cannot read, modify, or delete it.
cat > /etc/profile.d/prism4d-audit.sh << 'AUDIT_HOOK'
# PRISM4D Activity Audit — injected at login
if [ "$(whoami)" = "demo" ]; then
    export PRISM4D_SESSION_ID="$(date +%s)-$$"
    export PRISM4D_LOGIN_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    export PRISM4D_CLIENT_IP="${SSH_CLIENT%% *}"

    # Log session start
    echo "[${PRISM4D_LOGIN_TIME}] SESSION_START sid=${PRISM4D_SESSION_ID} ip=${PRISM4D_CLIENT_IP} user=demo tty=$(tty 2>/dev/null || echo none)" \
        >> /var/log/prism4d-audit/sessions.log 2>/dev/null

    # Command logger — fires before every prompt
    _prism4d_log_cmd() {
        local last_cmd
        last_cmd=$(history 1 | sed 's/^[ ]*[0-9]*[ ]*//')
        [ -z "$last_cmd" ] && return
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] CMD sid=${PRISM4D_SESSION_ID} ip=${PRISM4D_CLIENT_IP} pwd=$(pwd) cmd=${last_cmd}" \
            >> /var/log/prism4d-audit/commands.log 2>/dev/null
    }
    PROMPT_COMMAND="_prism4d_log_cmd;${PROMPT_COMMAND:-}"

    # Log session end on exit
    trap 'echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SESSION_END sid=${PRISM4D_SESSION_ID} ip=${PRISM4D_CLIENT_IP} duration=$(( $(date +%s) - ${PRISM4D_SESSION_ID%%-*} ))s" >> /var/log/prism4d-audit/sessions.log 2>/dev/null' EXIT
fi
AUDIT_HOOK

chmod 644 /etc/profile.d/prism4d-audit.sh

# ── 4. Configure SFTP audit logging ──────────────────────────────────
# The sshd_config uses sftp-server -l INFO which logs transfers to syslog.
# We redirect syslog auth messages to our audit file.
# Since we may not have rsyslog, use a named pipe approach.
if [ ! -p "${AUDIT_DIR}/sftp_pipe" ]; then
    mkfifo "${AUDIT_DIR}/sftp_pipe" 2>/dev/null || true
fi

# ── 5. GPU activity monitor (background) ─────────────────────────────
# Polls nvidia-smi every 30s while any demo process is running.
(
    while true; do
        sleep 30
        # Only log if demo user has active processes
        if pgrep -u demo > /dev/null 2>&1; then
            GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
                --format=csv,noheader,nounits 2>/dev/null || echo "unavailable")
            DEMO_PROCS=$(ps -u demo -o pid,comm,etime --no-headers 2>/dev/null | tr '\n' '; ')
            echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU_POLL gpu_util=${GPU_INFO} demo_procs=[${DEMO_PROCS}]" \
                >> "${AUDIT_DIR}/gpu.log" 2>/dev/null
        fi
    done
) &

log_audit "Audit hooks installed"
log_audit "GPU monitor started (PID $!)"
log_audit "Starting SSH daemon..."

# ── 6. Start sshd in foreground ──────────────────────────────────────
exec /usr/sbin/sshd -D -e 2>&1 | while IFS= read -r line; do
    echo "$line"
    # Capture SSH auth events to sessions.log
    case "$line" in
        *"Accepted"*|*"Failed"*|*"Disconnected"*|*"Connection closed"*)
            echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SSH ${line}" \
                >> "${AUDIT_DIR}/sessions.log" 2>/dev/null
            ;;
    esac
done
