#!/usr/bin/env bash
set -euo pipefail
##############################################################################
# PRISM4D Demo — Cloudflare Tunnel Setup
#
# Exposes the SSH demo container at demo.delfictus.com via Cloudflare Tunnel.
# Clients can SSH from anywhere without port forwarding or VPN.
#
# Prerequisites:
#   1. delfictus.com added to Cloudflare (free plan is fine)
#   2. Nameservers at GoDaddy changed to Cloudflare's
#   3. Docker demo container running on port 2222
#
# Usage:
#   bash release/docker/cloudflare_tunnel_setup.sh setup    # First-time setup
#   bash release/docker/cloudflare_tunnel_setup.sh start    # Start tunnel
#   bash release/docker/cloudflare_tunnel_setup.sh stop     # Stop tunnel
#   bash release/docker/cloudflare_tunnel_setup.sh status   # Check tunnel status
#   bash release/docker/cloudflare_tunnel_setup.sh quick    # Quick tunnel (temp URL)
##############################################################################

PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLOUDFLARED="${HOME}/.local/bin/cloudflared"
CONFIG_DIR="${HOME}/.cloudflared"
TUNNEL_NAME="prism4d-demo"
DOMAIN="delfictus.com"
DEMO_SUBDOMAIN="demo.${DOMAIN}"
PIDFILE="/tmp/cloudflared-prism4d.pid"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Ensure cloudflared is available ─────────────────────────────────────
ensure_cloudflared() {
    if [ -x "${CLOUDFLARED}" ]; then
        return 0
    elif command -v cloudflared >/dev/null 2>&1; then
        CLOUDFLARED="$(command -v cloudflared)"
        return 0
    else
        echo -e "${RED}ERROR: cloudflared not found${NC}"
        echo "Install: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cf.deb && sudo dpkg -i /tmp/cf.deb"
        exit 1
    fi
}

# ── Check if demo container is running ──────────────────────────────────
check_demo() {
    if ! ss -tlnp 2>/dev/null | grep -q ":2222"; then
        echo -e "${YELLOW}WARNING: Nothing listening on port 2222${NC}"
        echo "Start the demo container first: docker compose -f release/docker/docker-compose.yml up -d"
        return 1
    fi
    return 0
}

# ── SETUP: Full Cloudflare Tunnel Configuration ────────────────────────
do_setup() {
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  PRISM4D — Cloudflare Tunnel Setup                          ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    ensure_cloudflared
    echo -e "${GREEN}[1/6] cloudflared: ${CLOUDFLARED}${NC}"
    ${CLOUDFLARED} --version
    echo ""

    # ── Step 1: Check if already authenticated ──
    if [ -f "${CONFIG_DIR}/cert.pem" ]; then
        echo -e "${GREEN}[2/6] Already authenticated with Cloudflare${NC}"
    else
        echo -e "${YELLOW}[2/6] Authenticating with Cloudflare...${NC}"
        echo ""
        echo "  A browser window will open. Log in to your Cloudflare account"
        echo "  and authorize cloudflared for the ${DOMAIN} zone."
        echo ""
        echo "  Press ENTER to continue..."
        read -r
        ${CLOUDFLARED} tunnel login
        echo ""
        if [ -f "${CONFIG_DIR}/cert.pem" ]; then
            echo -e "${GREEN}  Authentication successful!${NC}"
        else
            echo -e "${RED}  Authentication failed. Run this step again.${NC}"
            exit 1
        fi
    fi
    echo ""

    # ── Step 2: Create tunnel ──
    echo -e "${YELLOW}[3/6] Creating tunnel: ${TUNNEL_NAME}${NC}"
    EXISTING_TUNNEL=$(${CLOUDFLARED} tunnel list --output json 2>/dev/null | \
        python3 -c "import sys,json;tunnels=json.load(sys.stdin);[print(t['id']) for t in tunnels if t['name']=='${TUNNEL_NAME}']" 2>/dev/null || echo "")

    if [ -n "${EXISTING_TUNNEL}" ]; then
        TUNNEL_ID="${EXISTING_TUNNEL}"
        echo -e "${GREEN}  Tunnel already exists: ${TUNNEL_ID}${NC}"
    else
        # Create new tunnel
        ${CLOUDFLARED} tunnel create "${TUNNEL_NAME}"
        TUNNEL_ID=$(${CLOUDFLARED} tunnel list --output json 2>/dev/null | \
            python3 -c "import sys,json;tunnels=json.load(sys.stdin);[print(t['id']) for t in tunnels if t['name']=='${TUNNEL_NAME}']" 2>/dev/null)
        echo -e "${GREEN}  Tunnel created: ${TUNNEL_ID}${NC}"
    fi

    if [ -z "${TUNNEL_ID}" ]; then
        echo -e "${RED}  ERROR: Could not determine tunnel ID${NC}"
        exit 1
    fi
    echo ""

    # ── Step 3: Configure DNS ──
    echo -e "${YELLOW}[4/6] Configuring DNS: ${DEMO_SUBDOMAIN} → tunnel${NC}"
    ${CLOUDFLARED} tunnel route dns "${TUNNEL_NAME}" "${DEMO_SUBDOMAIN}" 2>&1 || true
    echo -e "${GREEN}  DNS CNAME: ${DEMO_SUBDOMAIN} → ${TUNNEL_ID}.cfargotunnel.com${NC}"
    echo ""

    # ── Step 4: Write tunnel config ──
    echo -e "${YELLOW}[5/6] Writing tunnel configuration${NC}"
    CRED_FILE="${CONFIG_DIR}/${TUNNEL_ID}.json"

    cat > "${CONFIG_DIR}/config.yml" << YAML
# PRISM4D Demo — Cloudflare Tunnel Configuration
# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

tunnel: ${TUNNEL_ID}
credentials-file: ${CRED_FILE}

ingress:
  # SSH demo access via demo.delfictus.com
  - hostname: ${DEMO_SUBDOMAIN}
    service: ssh://localhost:2222
  # Catch-all (required by cloudflared)
  - service: http_status:404
YAML

    echo -e "${GREEN}  Config written: ${CONFIG_DIR}/config.yml${NC}"
    echo ""

    # ── Step 5: Validate ──
    echo -e "${YELLOW}[6/6] Validating configuration${NC}"
    ${CLOUDFLARED} tunnel ingress validate 2>&1 || true
    echo ""

    # ── Summary ──
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  SETUP COMPLETE                                             ║${NC}"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║                                                             ║${NC}"
    echo -e "${CYAN}║  Tunnel ID:   ${TUNNEL_ID}            ║${NC}"
    echo -e "${CYAN}║  Hostname:    ${DEMO_SUBDOMAIN}                          ║${NC}"
    echo -e "${CYAN}║  Service:     ssh://localhost:2222                          ║${NC}"
    echo -e "${CYAN}║                                                             ║${NC}"
    echo -e "${CYAN}║  Start:  bash $0 start            ║${NC}"
    echo -e "${CYAN}║                                                             ║${NC}"
    echo -e "${CYAN}║  Client Connection (after start):                           ║${NC}"
    echo -e "${CYAN}║                                                             ║${NC}"
    echo -e "${CYAN}║  Option A — cloudflared on client:                          ║${NC}"
    echo -e "${CYAN}║    ssh -o ProxyCommand='cloudflared access ssh \\            ║${NC}"
    echo -e "${CYAN}║      --hostname ${DEMO_SUBDOMAIN}' demo@${DEMO_SUBDOMAIN}   ║${NC}"
    echo -e "${CYAN}║    Password: demo                                           ║${NC}"
    echo -e "${CYAN}║                                                             ║${NC}"
    echo -e "${CYAN}║  Option B — Browser SSH (after Access setup):               ║${NC}"
    echo -e "${CYAN}║    https://${DEMO_SUBDOMAIN}                                ║${NC}"
    echo -e "${CYAN}║                                                             ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Save tunnel ID for other commands
    echo "${TUNNEL_ID}" > "${CONFIG_DIR}/.prism4d-tunnel-id"
}

# ── START: Run the tunnel ──────────────────────────────────────────────
do_start() {
    ensure_cloudflared
    check_demo || true

    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
        echo -e "${GREEN}Tunnel already running (PID $(cat "${PIDFILE}"))${NC}"
        return 0
    fi

    if [ ! -f "${CONFIG_DIR}/config.yml" ]; then
        echo -e "${RED}ERROR: No tunnel config found. Run 'setup' first.${NC}"
        exit 1
    fi

    echo -e "${CYAN}Starting Cloudflare Tunnel...${NC}"

    # Run in background with log
    mkdir -p /tmp/prism4d-logs
    nohup ${CLOUDFLARED} tunnel run \
        --config "${CONFIG_DIR}/config.yml" \
        > /tmp/prism4d-logs/cloudflared.log 2>&1 &

    TUNNEL_PID=$!
    echo "${TUNNEL_PID}" > "${PIDFILE}"

    sleep 3

    if kill -0 "${TUNNEL_PID}" 2>/dev/null; then
        echo -e "${GREEN}Tunnel running (PID ${TUNNEL_PID})${NC}"
        echo -e "${GREEN}Log: /tmp/prism4d-logs/cloudflared.log${NC}"
        echo ""
        echo -e "SSH access: ${CYAN}ssh -o ProxyCommand='cloudflared access ssh --hostname demo.delfictus.com' demo@demo.delfictus.com${NC}"
        echo -e "Password:   ${CYAN}demo${NC}"
    else
        echo -e "${RED}Tunnel failed to start. Check log:${NC}"
        tail -20 /tmp/prism4d-logs/cloudflared.log
        rm -f "${PIDFILE}"
        exit 1
    fi
}

# ── STOP: Kill the tunnel ──────────────────────────────────────────────
do_stop() {
    if [ -f "${PIDFILE}" ]; then
        PID=$(cat "${PIDFILE}")
        if kill -0 "${PID}" 2>/dev/null; then
            kill "${PID}"
            echo -e "${GREEN}Tunnel stopped (PID ${PID})${NC}"
        else
            echo -e "${YELLOW}Tunnel was not running${NC}"
        fi
        rm -f "${PIDFILE}"
    else
        # Try to find and kill any cloudflared tunnel processes
        pkill -f "cloudflared tunnel run" 2>/dev/null && \
            echo -e "${GREEN}Tunnel processes stopped${NC}" || \
            echo -e "${YELLOW}No tunnel running${NC}"
    fi
}

# ── STATUS: Check tunnel health ────────────────────────────────────────
do_status() {
    ensure_cloudflared
    echo -e "${CYAN}=== Cloudflare Tunnel Status ===${NC}"
    echo ""

    # Check process
    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
        echo -e "  Process:    ${GREEN}Running (PID $(cat "${PIDFILE}"))${NC}"
    else
        echo -e "  Process:    ${RED}Not running${NC}"
    fi

    # Check demo container
    if ss -tlnp 2>/dev/null | grep -q ":2222"; then
        echo -e "  SSH Demo:   ${GREEN}Listening on :2222${NC}"
    else
        echo -e "  SSH Demo:   ${RED}Not listening${NC}"
    fi

    # Check config
    if [ -f "${CONFIG_DIR}/config.yml" ]; then
        echo -e "  Config:     ${GREEN}${CONFIG_DIR}/config.yml${NC}"
    else
        echo -e "  Config:     ${RED}Not found (run setup)${NC}"
    fi

    # Check auth
    if [ -f "${CONFIG_DIR}/cert.pem" ]; then
        echo -e "  Auth:       ${GREEN}Authenticated${NC}"
    else
        echo -e "  Auth:       ${RED}Not authenticated${NC}"
    fi

    # DNS check
    echo ""
    echo -e "${CYAN}DNS Resolution:${NC}"
    RESOLVED=$(dig +short "${DEMO_SUBDOMAIN}" CNAME 2>/dev/null || echo "unresolvable")
    if [ -n "${RESOLVED}" ] && [ "${RESOLVED}" != "unresolvable" ]; then
        echo -e "  ${DEMO_SUBDOMAIN} → ${GREEN}${RESOLVED}${NC}"
    else
        RESOLVED_A=$(dig +short "${DEMO_SUBDOMAIN}" A 2>/dev/null || echo "none")
        if [ -n "${RESOLVED_A}" ] && [ "${RESOLVED_A}" != "none" ]; then
            echo -e "  ${DEMO_SUBDOMAIN} → ${GREEN}${RESOLVED_A}${NC}"
        else
            echo -e "  ${DEMO_SUBDOMAIN} → ${RED}Not resolving${NC}"
            echo -e "  ${YELLOW}Ensure nameservers are set to Cloudflare and DNS has propagated${NC}"
        fi
    fi

    # Recent log
    if [ -f /tmp/prism4d-logs/cloudflared.log ]; then
        echo ""
        echo -e "${CYAN}Recent Log:${NC}"
        tail -5 /tmp/prism4d-logs/cloudflared.log 2>/dev/null | sed 's/^/  /'
    fi

    # List tunnels
    echo ""
    echo -e "${CYAN}Registered Tunnels:${NC}"
    ${CLOUDFLARED} tunnel list 2>/dev/null | head -10 | sed 's/^/  /' || echo "  (unable to list — run setup first)"
}

# ── QUICK: Temporary tunnel with auto-generated URL ────────────────────
do_quick() {
    ensure_cloudflared
    check_demo || exit 1

    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  PRISM4D — Quick Tunnel (Temporary URL)                     ║${NC}"
    echo -e "${CYAN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║  No Cloudflare account needed. URL expires when stopped.    ║${NC}"
    echo -e "${CYAN}║  Press Ctrl+C to stop.                                      ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Starting tunnel to ssh://localhost:2222 ..."
    echo "Watch for the URL below (*.trycloudflare.com)"
    echo ""

    ${CLOUDFLARED} tunnel --url ssh://localhost:2222 2>&1 | while IFS= read -r line; do
        echo "$line"
        if echo "$line" | grep -q "trycloudflare.com"; then
            QUICK_URL=$(echo "$line" | grep -oP 'https://[^ ]+trycloudflare\.com')
            if [ -n "${QUICK_URL}" ]; then
                echo ""
                echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
                echo -e "${GREEN}  Quick Tunnel Active!${NC}"
                echo -e "${GREEN}  URL: ${QUICK_URL}${NC}"
                echo ""
                echo -e "  Client command:"
                echo -e "  ${CYAN}ssh -o ProxyCommand='cloudflared access ssh --hostname ${QUICK_URL#https://}' demo@${QUICK_URL#https://}${NC}"
                echo -e "  Password: ${CYAN}demo${NC}"
                echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
            fi
        fi
    done
}

# ── INSTALL-SERVICE: Create systemd user service ───────────────────────
do_install_service() {
    ensure_cloudflared

    if [ ! -f "${CONFIG_DIR}/config.yml" ]; then
        echo -e "${RED}ERROR: No tunnel config. Run 'setup' first.${NC}"
        exit 1
    fi

    SYSTEMD_DIR="${HOME}/.config/systemd/user"
    mkdir -p "${SYSTEMD_DIR}"

    cat > "${SYSTEMD_DIR}/prism4d-tunnel.service" << EOF
[Unit]
Description=PRISM4D Demo Cloudflare Tunnel
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=${CLOUDFLARED} tunnel run --config ${CONFIG_DIR}/config.yml
Restart=on-failure
RestartSec=5
Environment=PATH=${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable prism4d-tunnel.service
    systemctl --user start prism4d-tunnel.service

    echo -e "${GREEN}Systemd user service installed and started${NC}"
    echo ""
    echo "  Status:  systemctl --user status prism4d-tunnel"
    echo "  Logs:    journalctl --user -u prism4d-tunnel -f"
    echo "  Stop:    systemctl --user stop prism4d-tunnel"
    echo "  Start:   systemctl --user start prism4d-tunnel"
}

# ── MAIN ───────────────────────────────────────────────────────────────
case "${1:-help}" in
    setup)      do_setup ;;
    start)      do_start ;;
    stop)       do_stop ;;
    status)     do_status ;;
    quick)      do_quick ;;
    service)    do_install_service ;;
    *)
        echo "Usage: $0 {setup|start|stop|status|quick|service}"
        echo ""
        echo "  setup    — Full tunnel setup (auth + create + DNS + config)"
        echo "  start    — Start the tunnel in background"
        echo "  stop     — Stop the tunnel"
        echo "  status   — Check tunnel, DNS, and container health"
        echo "  quick    — Temporary tunnel with auto URL (no account needed)"
        echo "  service  — Install as systemd user service (auto-start)"
        ;;
esac
