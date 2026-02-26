#!/usr/bin/env bash
##############################################################################
# PRISM4D Demo — Audit Log Viewer
#
# Real-time and historical views of all demo user activity.
#
# Usage:
#   ./audit_viewer.sh live       # Real-time command feed
#   ./audit_viewer.sh sessions   # All login/logout events
#   ./audit_viewer.sh commands   # Full command history
#   ./audit_viewer.sh gpu        # GPU usage log
#   ./audit_viewer.sh report     # Summary report
#   ./audit_viewer.sh tail       # Live tail all logs
##############################################################################

AUDIT_DIR="/var/log/prism4d-audit"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

if [ ! -d "${AUDIT_DIR}" ]; then
    echo "ERROR: Audit directory not found: ${AUDIT_DIR}"
    echo "Has the demo container been started?"
    exit 1
fi

case "${1:-report}" in

    live)
        echo -e "${CYAN}=== PRISM4D Live Activity Feed ===${NC}"
        echo "Watching: commands.log + sessions.log"
        echo "Press Ctrl+C to stop"
        echo ""
        tail -f "${AUDIT_DIR}/commands.log" "${AUDIT_DIR}/sessions.log" 2>/dev/null
        ;;

    sessions)
        echo -e "${CYAN}=== SSH Session History ===${NC}"
        echo ""
        if [ -s "${AUDIT_DIR}/sessions.log" ]; then
            cat "${AUDIT_DIR}/sessions.log"
        else
            echo "(no sessions recorded yet)"
        fi
        ;;

    commands|cmds)
        echo -e "${CYAN}=== Command History ===${NC}"
        echo ""
        if [ -s "${AUDIT_DIR}/commands.log" ]; then
            cat "${AUDIT_DIR}/commands.log"
        else
            echo "(no commands recorded yet)"
        fi
        ;;

    gpu)
        echo -e "${CYAN}=== GPU Usage Log ===${NC}"
        echo ""
        if [ -s "${AUDIT_DIR}/gpu.log" ]; then
            cat "${AUDIT_DIR}/gpu.log"
        else
            echo "(no GPU activity recorded yet)"
        fi
        ;;

    tail)
        echo -e "${CYAN}=== Live Tail — All Audit Logs ===${NC}"
        echo "Press Ctrl+C to stop"
        echo ""
        tail -f "${AUDIT_DIR}"/*.log 2>/dev/null
        ;;

    report)
        echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║  PRISM4D Demo Audit Report                                  ║${NC}"
        echo -e "${CYAN}║  Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)                            ║${NC}"
        echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
        echo ""

        # Session summary
        TOTAL_SESSIONS=$(grep -c "SESSION_START" "${AUDIT_DIR}/sessions.log" 2>/dev/null || echo 0)
        ACTIVE_SESSIONS=$(grep -c "SESSION_START" "${AUDIT_DIR}/sessions.log" 2>/dev/null || echo 0)
        ENDED_SESSIONS=$(grep -c "SESSION_END" "${AUDIT_DIR}/sessions.log" 2>/dev/null || echo 0)
        FAILED_LOGINS=$(grep -c "Failed" "${AUDIT_DIR}/sessions.log" 2>/dev/null || echo 0)

        echo -e "${GREEN}Sessions:${NC}"
        echo "  Total logins:     ${TOTAL_SESSIONS}"
        echo "  Completed:        ${ENDED_SESSIONS}"
        echo "  Failed attempts:  ${FAILED_LOGINS}"
        echo ""

        # Unique IPs
        echo -e "${GREEN}Client IPs:${NC}"
        grep "SESSION_START" "${AUDIT_DIR}/sessions.log" 2>/dev/null | \
            grep -oP 'ip=\K[^ ]+' | sort -u | while read ip; do
            COUNT=$(grep -c "ip=${ip}" "${AUDIT_DIR}/sessions.log" 2>/dev/null || echo 0)
            echo "  ${ip} (${COUNT} sessions)"
        done
        echo ""

        # Command summary
        TOTAL_CMDS=$(wc -l < "${AUDIT_DIR}/commands.log" 2>/dev/null || echo 0)
        echo -e "${GREEN}Commands Executed:${NC} ${TOTAL_CMDS} total"

        if [ "${TOTAL_CMDS}" -gt 0 ]; then
            echo ""
            echo -e "${GREEN}Top 10 Commands:${NC}"
            grep -oP 'cmd=\K.*' "${AUDIT_DIR}/commands.log" 2>/dev/null | \
                sed 's/ .*//' | sort | uniq -c | sort -rn | head -10 | \
                while read count cmd; do
                    printf "  %4d  %s\n" "$count" "$cmd"
                done

            echo ""
            echo -e "${GREEN}Directories Visited:${NC}"
            grep -oP 'pwd=\K[^ ]+' "${AUDIT_DIR}/commands.log" 2>/dev/null | \
                sort | uniq -c | sort -rn | head -5 | \
                while read count dir; do
                    printf "  %4d  %s\n" "$count" "$dir"
                done

            echo ""
            echo -e "${GREEN}PRISM4D Invocations:${NC}"
            grep -E "cmd=(prism4d|nhs_rt_full|prism4d detect)" "${AUDIT_DIR}/commands.log" 2>/dev/null | \
                tail -10 || echo "  (none)"
        fi

        echo ""
        echo -e "${GREEN}GPU Activity:${NC}"
        GPU_POLLS=$(wc -l < "${AUDIT_DIR}/gpu.log" 2>/dev/null || echo 0)
        echo "  GPU poll entries: ${GPU_POLLS}"
        if [ "${GPU_POLLS}" -gt 0 ]; then
            echo "  Latest:"
            tail -3 "${AUDIT_DIR}/gpu.log" 2>/dev/null | sed 's/^/    /'
        fi

        echo ""
        echo -e "${YELLOW}Log Files:${NC}"
        for f in "${AUDIT_DIR}"/*.log; do
            [ -f "$f" ] || continue
            SIZE=$(du -h "$f" | cut -f1)
            LINES=$(wc -l < "$f")
            printf "  %-20s %6s  %6d lines\n" "$(basename "$f")" "${SIZE}" "${LINES}"
        done
        echo ""
        ;;

    *)
        echo "Usage: $0 {live|sessions|commands|gpu|tail|report}"
        exit 1
        ;;
esac
