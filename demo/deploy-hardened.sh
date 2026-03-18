#!/bin/bash
## PRISM4D Demo — Hardened Deployment Script
## Run with: sudo bash deploy-hardened.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Run as root: sudo bash $0"
    exit 1
fi

# Make docker compose plugin available to root
# Plugin is installed in user's home dir, symlink it for system-wide access
PLUGIN_SRC="/home/diddy/.docker/cli-plugins/docker-compose"
PLUGIN_DST="/usr/libexec/docker/cli-plugins/docker-compose"
if [ -f "$PLUGIN_SRC" ] && [ ! -f "$PLUGIN_DST" ]; then
    mkdir -p /usr/libexec/docker/cli-plugins
    ln -sf "$PLUGIN_SRC" "$PLUGIN_DST"
fi

if docker compose version >/dev/null 2>&1; then
    DC="docker compose"
else
    echo "ERROR: 'docker compose' plugin not available"
    exit 1
fi
echo "Using: $DC ($(docker compose version))"

echo "============================================"
echo "  PRISM4D Hardened Demo Deployment"
echo "============================================"
echo ""

## -------- STEP 1: Stop current container --------
echo "[1/6] Stopping current container..."
docker stop prism4d-demo 2>/dev/null || true
docker rm prism4d-demo 2>/dev/null || true
# Also stop any compose-managed containers
$DC -f docker-compose.yml down 2>/dev/null || true
$DC -f docker-compose.hardened.yml down 2>/dev/null || true
echo "  Done."

## -------- STEP 2: Build hardened image --------
echo ""
echo "[2/6] Building hardened container image..."
$DC -f docker-compose.hardened.yml build --no-cache
echo "  Done."

## -------- STEP 3: UFW rules for demo isolation --------
echo ""
echo "[3/6] Configuring firewall rules..."

# Delete any existing rules for demo ports
ufw delete allow 7681 2>/dev/null || true
ufw delete allow 8080 2>/dev/null || true

# Block external access to demo ports (only localhost via tunnel)
ufw deny in on enp130s0 to any port 7681 comment "Block direct ttyd access" 2>/dev/null || true
ufw deny in on enp130s0 to any port 8080 comment "Block direct HTTP access" 2>/dev/null || true

# Block the demo container subnet from reaching the LAN
ufw deny in from 10.99.0.0/24 to 192.168.1.0/24 comment "Isolate demo from LAN" 2>/dev/null || true
ufw deny in from 10.99.0.0/24 to 172.17.0.0/16 comment "Isolate demo from docker0" 2>/dev/null || true
ufw deny in from 10.99.0.0/24 to 172.18.0.0/16 comment "Isolate demo from docker nets" 2>/dev/null || true

echo "  Firewall rules applied."

## -------- STEP 4: Update Cloudflare tunnel config --------
echo ""
echo "[4/6] Updating Cloudflare tunnel config..."

cp /etc/cloudflared/config.yml /etc/cloudflared/config.yml.bak.$(date +%s)

cat > /etc/cloudflared/config.yml << 'CFEOF'
tunnel: b9185477-3fcc-4706-aaae-f96f9ad2c91f
credentials-file: /home/diddy/.cloudflared/b9185477-3fcc-4706-aaae-f96f9ad2c91f.json

originRequest:
  noTLSVerify: false
  connectTimeout: 10s
  tlsTimeout: 10s

ingress:
  - hostname: demo.delfictus.com
    service: http://127.0.0.1:7681
    originRequest:
      connectTimeout: 10s
  - hostname: viewer.delfictus.com
    service: http://127.0.0.1:8080
    originRequest:
      connectTimeout: 10s
  - service: http_status:404
CFEOF

echo "  Tunnel config updated (backup saved)."

## -------- STEP 5: Restart services --------
echo ""
echo "[5/6] Starting hardened container..."
$DC -f docker-compose.hardened.yml up -d
echo "  Container started."

echo ""
echo "Restarting Cloudflare tunnel..."
systemctl restart cloudflared
echo "  Tunnel restarted."

## -------- STEP 6: Verify --------
echo ""
echo "[6/6] Verification..."
sleep 5

echo "  Container status:"
docker ps --filter name=prism4d-demo --format "  {{.Status}} | {{.Ports}}"

echo ""
echo "  Security checks:"
if ss -tlnp | grep -q "0.0.0.0:7681"; then
    echo "  [FAIL] Port 7681 is bound to 0.0.0.0"
else
    echo "  [PASS] Port 7681 bound to localhost only"
fi
if ss -tlnp | grep -q "0.0.0.0:8080"; then
    echo "  [FAIL] Port 8080 is bound to 0.0.0.0"
else
    echo "  [PASS] Port 8080 bound to localhost only"
fi

echo ""
echo "  Network isolation:"
docker exec prism4d-demo sh -c 'ping -c1 -W2 192.168.1.254 2>/dev/null && echo "  [FAIL] Can reach gateway" || echo "  [PASS] Cannot reach gateway"'
docker exec prism4d-demo sh -c 'ping -c1 -W2 8.8.8.8 2>/dev/null && echo "  [FAIL] Has internet" || echo "  [PASS] No internet access"'

echo ""
echo "  ttyd auth:"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7681/ 2>/dev/null)
if [ "$HTTP_CODE" = "401" ]; then
    echo "  [PASS] Requires authentication (HTTP 401)"
elif [ "$HTTP_CODE" = "200" ]; then
    echo "  [FAIL] OPEN without auth (HTTP 200)"
else
    echo "  [INFO] HTTP $HTTP_CODE (container may still be starting)"
fi

echo ""
echo "  Filesystem:"
docker exec prism4d-demo sh -c 'touch /test 2>/dev/null && echo "  [FAIL] Root FS writable" || echo "  [PASS] Root FS read-only"'

echo ""
echo "  Obfuscation:"
docker exec prism4d-demo sh -c 'ls /opt/prism4d 2>/dev/null && echo "  [FAIL] /opt/prism4d exists" || echo "  [PASS] /opt/prism4d removed"'
docker exec prism4d-demo sh -c 'ls /home/diddy 2>/dev/null && echo "  [FAIL] /home/diddy exists" || echo "  [PASS] Host paths scrubbed"'
docker exec prism4d-demo sh -c 'cat /var/lib/p4r/.re 2>/dev/null && echo "  [FAIL] Key file readable" || echo "  [PASS] Key file destroyed after boot"'
docker exec prism4d-demo sh -c 'ls /var/lib/p4r/k/ptx/*.ptx 2>/dev/null && echo "  [FAIL] Plaintext kernels on disk" || echo "  [PASS] Only encrypted kernels on disk"'

echo ""
echo "  Removed tools:"
for tool in curl wget apt dpkg su sudo nc strings objdump file hexdump base64 find ssh gdb strace; do
    docker exec prism4d-demo sh -c "which $tool 2>/dev/null" && echo "  [FAIL] $tool available" || echo "  [PASS] $tool removed"
done

echo ""
echo "============================================"
echo "  DEPLOYMENT COMPLETE"
echo "============================================"
echo ""
echo "  REMAINING MANUAL STEP:"
echo "  Set up Cloudflare Access (Zero Trust) at:"
echo "  https://one.dash.cloudflare.com → Access → Applications"
echo ""
echo "  1. Add Application → Self-hosted"
echo "  2. Domain: demo.delfictus.com"
echo "  3. Policy: Allow → One-time PIN → your email"
echo "  4. Repeat for viewer.delfictus.com"
echo ""
