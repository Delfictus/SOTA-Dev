#!/bin/bash
## PRISM4D Demo Analytics & Metrics Dashboard
## Tracks all visitor activity via cloudflared, ttyd, and container logs
## Usage: sudo bash demo-analytics.sh [live|report|tail]

MODE=${1:-report}
LOG_DIR="/var/log/prism4d-demo"
mkdir -p "$LOG_DIR"

# ============================================================
# REPORT MODE — Summary of all tracked activity
# ============================================================
if [ "$MODE" = "report" ]; then
    echo "============================================"
    echo "  PRISM4D DEMO — ANALYTICS REPORT"
    echo "  $(date)"
    echo "============================================"
    echo ""

    # --- Cloudflare Tunnel Metrics ---
    echo "=== CLOUDFLARE TUNNEL METRICS ==="
    METRICS=$(curl -s http://127.0.0.1:20241/metrics 2>/dev/null)
    if [ -n "$METRICS" ]; then
        echo "  Active connections:"
        echo "$METRICS" | grep "cloudflared_tunnel_active_streams" | tail -1 | awk '{print "    Streams: "$2}'
        echo ""
        echo "  Total requests served:"
        echo "$METRICS" | grep "cloudflared_tunnel_total_requests" | tail -1 | awk '{print "    Total: "$2}'
        echo ""
        echo "  Request status codes:"
        echo "$METRICS" | grep "cloudflared_tunnel_response_by_code" | while read -r line; do
            code=$(echo "$line" | grep -oP 'status_code="\K[^"]+')
            count=$(echo "$line" | awk '{print $NF}')
            [ -n "$code" ] && printf "    HTTP %s: %s\n" "$code" "$count"
        done
        echo ""
        echo "  Tunnel uptime:"
        echo "$METRICS" | grep "cloudflared_tunnel_server_info" | head -1 | grep -oP 'version="[^"]+"' | sed 's/version="/    Version: /;s/"//'
        echo ""
        echo "  Data transferred:"
        RX=$(echo "$METRICS" | grep "cloudflared_tunnel_request_body_size_bytes_sum" | tail -1 | awk '{printf "%.2f MB", $2/1048576}')
        TX=$(echo "$METRICS" | grep "cloudflared_tunnel_response_body_size_bytes_sum" | tail -1 | awk '{printf "%.2f MB", $2/1048576}')
        echo "    Received: $RX"
        echo "    Sent: $TX"
    else
        echo "  Tunnel metrics unavailable (check: curl http://127.0.0.1:20241/metrics)"
    fi

    echo ""
    echo "=== CONTAINER METRICS ==="
    if docker ps --filter name=prism4d-demo --format "{{.Status}}" | grep -q "Up"; then
        echo "  Status: $(docker ps --filter name=prism4d-demo --format '{{.Status}}')"
        echo ""
        echo "  Resource usage:"
        docker stats prism4d-demo --no-stream --format "    CPU: {{.CPUPerc}} | Memory: {{.MemUsage}} | Net I/O: {{.NetIO}} | PIDs: {{.PIDs}}"
        echo ""
        echo "  ttyd connections (from logs):"
        CONN_COUNT=$(docker logs prism4d-demo 2>&1 | grep -c "HTTP /")
        WS_COUNT=$(docker logs prism4d-demo 2>&1 | grep -c "LWS_CALLBACK_ESTABLISHED\|WSI_STATE_ESTABLISHED\|new wsi")
        echo "    HTTP requests: $CONN_COUNT"
        echo "    WebSocket sessions: $WS_COUNT"
        echo ""
        echo "  Recent visitor activity (last 20 log entries):"
        docker logs prism4d-demo --tail 20 2>&1 | while read -r line; do
            echo "    $line"
        done
    else
        echo "  Container is not running!"
    fi

    echo ""
    echo "=== ACCESS LOG (host-level proxy tracking) ==="
    if [ -f "$LOG_DIR/access.log" ]; then
        TOTAL=$(wc -l < "$LOG_DIR/access.log")
        UNIQUE_IPS=$(awk '{print $1}' "$LOG_DIR/access.log" | sort -u | wc -l)
        echo "  Total requests logged: $TOTAL"
        echo "  Unique IPs: $UNIQUE_IPS"
        echo ""
        echo "  Top 10 visitors by IP:"
        awk '{print $1}' "$LOG_DIR/access.log" | sort | uniq -c | sort -rn | head -10 | while read -r count ip; do
            # Try to get geo info from whois
            country=$(whois "$ip" 2>/dev/null | grep -i "country:" | head -1 | awk '{print $NF}')
            org=$(whois "$ip" 2>/dev/null | grep -i "orgname\|org-name\|descr" | head -1 | sed 's/.*:\s*//')
            printf "    %-6s %-18s %-5s %s\n" "$count" "$ip" "${country:-??}" "${org:-unknown}"
        done
        echo ""
        echo "  Requests by hour (last 24h):"
        awk '{print $4}' "$LOG_DIR/access.log" | cut -d: -f1-2 | sort | uniq -c | tail -24 | while read -r count hour; do
            bar=$(printf '%0.s█' $(seq 1 $((count > 50 ? 50 : count))))
            printf "    %-20s %4s %s\n" "$hour" "$count" "$bar"
        done
    else
        echo "  No access logs yet. Start the logger: sudo bash demo-analytics.sh live"
    fi

    echo ""
    echo "=== CLOUDFLARE DASHBOARD ==="
    echo "  For full web analytics (geographic, browser, threat data):"
    echo "  https://dash.cloudflare.com → delfictus.com → Analytics & Logs"
    echo ""

# ============================================================
# LIVE MODE — Real-time request monitoring
# ============================================================
elif [ "$MODE" = "live" ]; then
    echo "============================================"
    echo "  PRISM4D DEMO — LIVE MONITORING"
    echo "  Press Ctrl+C to stop"
    echo "============================================"
    echo ""

    # Start tcpdump on localhost to capture all demo traffic with timestamps
    echo "Logging all demo traffic to $LOG_DIR/access.log"
    echo "Monitoring tunnel metrics every 30s..."
    echo ""

    # Background: capture HTTP requests to demo ports
    tcpdump -i lo -nn -l "tcp port 7681 or tcp port 8080" 2>/dev/null | \
        awk '{print strftime("%Y-%m-%d %H:%M:%S"), $0}' >> "$LOG_DIR/access.log" &
    TCPDUMP_PID=$!

    # Background: poll tunnel metrics
    while true; do
        METRICS=$(curl -s http://127.0.0.1:20241/metrics 2>/dev/null)
        STREAMS=$(echo "$METRICS" | grep "cloudflared_tunnel_active_streams" | tail -1 | awk '{print $2}')
        REQS=$(echo "$METRICS" | grep "cloudflared_tunnel_total_requests" | tail -1 | awk '{print $2}')
        echo "[$(date '+%H:%M:%S')] Active streams: ${STREAMS:-0} | Total requests: ${REQS:-0}"

        # Also log container stats
        if docker ps --filter name=prism4d-demo -q | grep -q .; then
            docker stats prism4d-demo --no-stream --format "[$(date '+%H:%M:%S')] CPU: {{.CPUPerc}} | Mem: {{.MemUsage}} | Net: {{.NetIO}}"
        fi

        sleep 30
    done &
    METRICS_PID=$!

    # Foreground: tail container logs
    echo "--- Live container logs ---"
    docker logs -f prism4d-demo 2>&1 | while read -r line; do
        echo "[$(date '+%H:%M:%S')] $line"
    done

    # Cleanup on exit
    kill $TCPDUMP_PID $METRICS_PID 2>/dev/null

# ============================================================
# TAIL MODE — Just tail the container logs
# ============================================================
elif [ "$MODE" = "tail" ]; then
    echo "=== Tailing prism4d-demo container logs ==="
    docker logs -f prism4d-demo 2>&1

else
    echo "Usage: sudo bash demo-analytics.sh [report|live|tail]"
    echo ""
    echo "  report  — Summary of all tracked metrics (default)"
    echo "  live    — Real-time monitoring with traffic capture"
    echo "  tail    — Just tail container logs"
fi
