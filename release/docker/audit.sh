#!/usr/bin/env bash
##############################################################################
# PRISM4D Demo — Audit & Activity Logger
#
# Logs ALL demo user activity:
#   - SSH login/logout with IP, timestamp, session duration
#   - Every command executed (via bash PROMPT_COMMAND)
#   - File access (reads, writes, downloads via SFTP)
#   - GPU usage during sessions
#   - Binary invocations with full arguments
#
# Logs are written to /var/log/prism4d-audit/ on the HOST (bind-mounted)
# and are NOT visible to the demo user inside the container.
##############################################################################

AUDIT_DIR="/var/log/prism4d-audit"
mkdir -p "${AUDIT_DIR}"

# Rotate logs if > 50MB
for f in "${AUDIT_DIR}"/*.log; do
    [ -f "$f" ] || continue
    SIZE=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 52428800 ]; then
        mv "$f" "${f}.$(date +%Y%m%d%H%M%S).bak"
    fi
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Audit system initialized" >> "${AUDIT_DIR}/system.log"
