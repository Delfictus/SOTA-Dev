#!/bin/bash
## PRISM4D Demo — JSON Metrics Dump
## Dumps all activity since last dump into a timestamped JSON file
## Each dump is incremental (only new data since last dump)

DUMP_DIR="/var/log/prism4d-demo/dumps"
STATE_FILE="/var/log/prism4d-demo/.last_dump_ts"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TIMESTAMP_FILE=$(date +"%Y%m%d_%H%M%S")
DUMP_FILE="$DUMP_DIR/demo_metrics_${TIMESTAMP_FILE}.json"

mkdir -p "$DUMP_DIR"

# Get last dump timestamp (or epoch if first run)
if [ -f "$STATE_FILE" ]; then
    LAST_DUMP=$(cat "$STATE_FILE")
    LAST_DUMP_EPOCH=$(date -d "$LAST_DUMP" +%s 2>/dev/null || echo 0)
else
    LAST_DUMP="never"
    LAST_DUMP_EPOCH=0
fi

# --- Collect tunnel metrics ---
METRICS_RAW=$(curl -s http://127.0.0.1:20241/metrics 2>/dev/null)
TUNNEL_TOTAL_REQUESTS=$(echo "$METRICS_RAW" | grep "^cloudflared_tunnel_total_requests " | awk '{print $2+0}')
TUNNEL_ACTIVE_STREAMS=$(echo "$METRICS_RAW" | grep "^cloudflared_tunnel_active_streams " | awk '{print $2+0}')
TUNNEL_RX_BYTES=$(echo "$METRICS_RAW" | grep "^cloudflared_tunnel_request_body_size_bytes_sum " | awk '{print $2+0}')
TUNNEL_TX_BYTES=$(echo "$METRICS_RAW" | grep "^cloudflared_tunnel_response_body_size_bytes_sum " | awk '{print $2+0}')

# Response codes
RESPONSE_CODES=$(echo "$METRICS_RAW" | grep "^cloudflared_tunnel_response_by_code" | \
    sed 's/.*status_code="\([^"]*\)".* \(.*\)/\1:\2/' | \
    awk -F: '{printf "%s\"%s\": %d", (NR>1?", ":""), $1, $2+0}')

# --- Collect container metrics ---
CONTAINER_STATUS="stopped"
CONTAINER_CPU="0"
CONTAINER_MEM="0"
CONTAINER_NET_IN="0"
CONTAINER_NET_OUT="0"
CONTAINER_PIDS="0"

if docker ps --filter name=prism4d-demo -q 2>/dev/null | grep -q .; then
    CONTAINER_STATUS="running"
    STATS=$(docker stats prism4d-demo --no-stream --format "{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}|{{.PIDs}}" 2>/dev/null)
    CONTAINER_CPU=$(echo "$STATS" | cut -d'|' -f1 | tr -d '%')
    CONTAINER_MEM=$(echo "$STATS" | cut -d'|' -f2)
    CONTAINER_NET=$(echo "$STATS" | cut -d'|' -f3)
    CONTAINER_NET_IN=$(echo "$CONTAINER_NET" | cut -d'/' -f1 | tr -d ' ')
    CONTAINER_NET_OUT=$(echo "$CONTAINER_NET" | cut -d'/' -f2 | tr -d ' ')
    CONTAINER_PIDS=$(echo "$STATS" | cut -d'|' -f4)
fi

# --- Collect container logs since last dump ---
if [ "$LAST_DUMP_EPOCH" -gt 0 ]; then
    SINCE_FLAG="--since $(date -d @$LAST_DUMP_EPOCH -Iseconds)"
else
    SINCE_FLAG=""
fi

# Parse ttyd logs for connection events
CONTAINER_LOGS=$(docker logs prism4d-demo $SINCE_FLAG 2>&1)
HTTP_REQUESTS=$(echo "$CONTAINER_LOGS" | grep -c "HTTP /" 2>/dev/null || echo 0)
WS_SESSIONS=$(echo "$CONTAINER_LOGS" | grep -c "CALLBACK_ESTABLISHED\|new wsi\|WSI_STATE" 2>/dev/null || echo 0)

# Build connection events array from ttyd logs
EVENTS=""
while IFS= read -r line; do
    [ -z "$line" ] && continue
    # Extract timestamp and event type from ttyd log lines
    if echo "$line" | grep -qE "HTTP /|CALLBACK|wsi"; then
        TS=$(echo "$line" | grep -oP '\[\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}' | tr -d '[' | sed 's|/|-|g')
        EVENT_TYPE="unknown"
        echo "$line" | grep -q "HTTP /" && EVENT_TYPE="http_request"
        echo "$line" | grep -q "CALLBACK_ESTABLISHED\|new wsi" && EVENT_TYPE="websocket_open"
        echo "$line" | grep -q "CALLBACK_CLOSED\|__lws_lc_untag" && EVENT_TYPE="websocket_close"
        ENDPOINT=$(echo "$line" | grep -oP 'HTTP \K/\S*' || echo "/")
        # Escape for JSON
        SAFE_LINE=$(echo "$line" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/\\t/g' | tr -d '\n\r')
        EVENTS="${EVENTS}${EVENTS:+,}
    {\"timestamp\": \"${TS:-unknown}\", \"type\": \"${EVENT_TYPE}\", \"endpoint\": \"${ENDPOINT}\", \"raw\": \"${SAFE_LINE}\"}"
    fi
done <<< "$CONTAINER_LOGS"

# --- System context ---
UPTIME=$(uptime -s)
LOAD=$(cat /proc/loadavg | awk '{print $1, $2, $3}')

# --- Build JSON ---
cat > "$DUMP_FILE" << JSONEOF
{
  "dump_metadata": {
    "generated_at": "$TIMESTAMP",
    "previous_dump": "$LAST_DUMP",
    "dump_file": "$DUMP_FILE",
    "hostname": "$(hostname)",
    "schema_version": "1.0"
  },
  "tunnel": {
    "total_requests": ${TUNNEL_TOTAL_REQUESTS:-0},
    "active_streams": ${TUNNEL_ACTIVE_STREAMS:-0},
    "bytes_received": ${TUNNEL_RX_BYTES:-0},
    "bytes_sent": ${TUNNEL_TX_BYTES:-0},
    "response_codes": {${RESPONSE_CODES:-}}
  },
  "container": {
    "status": "$CONTAINER_STATUS",
    "cpu_percent": "${CONTAINER_CPU:-0}",
    "memory_usage": "$CONTAINER_MEM",
    "network_in": "$CONTAINER_NET_IN",
    "network_out": "$CONTAINER_NET_OUT",
    "pids": "${CONTAINER_PIDS:-0}"
  },
  "activity_since_last_dump": {
    "http_requests": $HTTP_REQUESTS,
    "websocket_sessions": $WS_SESSIONS,
    "events": [${EVENTS}
    ]
  },
  "system": {
    "uptime_since": "$UPTIME",
    "load_average": "$LOAD",
    "tunnel_pid": $(pgrep -f "cloudflared.*tunnel run" || echo 0),
    "suricata_active": $(systemctl is-active suricata 2>/dev/null | grep -c "^active"),
    "ufw_active": $(ufw status 2>/dev/null | grep -c "^Status: active")
  }
}
JSONEOF

# Update last dump timestamp
echo "$TIMESTAMP" > "$STATE_FILE"

# Validate JSON
if python3 -m json.tool "$DUMP_FILE" > /dev/null 2>&1; then
    echo "Dump saved: $DUMP_FILE"
    echo "Period: $LAST_DUMP → $TIMESTAMP"
    echo ""
    # Pretty summary
    python3 -c "
import json
with open('$DUMP_FILE') as f:
    d = json.load(f)
print('=== PRISM4D Demo Metrics Dump ===')
print(f'  Period: {d[\"dump_metadata\"][\"previous_dump\"]} → {d[\"dump_metadata\"][\"generated_at\"]}')
print(f'  Tunnel requests: {d[\"tunnel\"][\"total_requests\"]}')
print(f'  Active streams: {d[\"tunnel\"][\"active_streams\"]}')
print(f'  Container: {d[\"container\"][\"status\"]} (CPU: {d[\"container\"][\"cpu_percent\"]}%)')
print(f'  HTTP requests (period): {d[\"activity_since_last_dump\"][\"http_requests\"]}')
print(f'  WebSocket sessions (period): {d[\"activity_since_last_dump\"][\"websocket_sessions\"]}')
print(f'  Events logged: {len(d[\"activity_since_last_dump\"][\"events\"])}')
print(f'  File: {d[\"dump_metadata\"][\"dump_file\"]}')
"
else
    echo "WARNING: JSON validation failed. Raw file saved at $DUMP_FILE"
fi
