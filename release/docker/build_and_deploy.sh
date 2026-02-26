#!/usr/bin/env bash
set -euo pipefail
##############################################################################
# PRISM4D Demo — Build Docker Image & Deploy SSH Demo Server
#
# This script:
#   1. Builds the demo release package (binaries, PTX, scripts)
#   2. Copies sample topology files for the demo
#   3. Builds the Docker image with full audit pipeline
#   4. Creates the host audit log directory
#   5. Starts the container with GPU access
#
# Usage:  bash release/docker/build_and_deploy.sh
# Access: ssh demo@localhost -p 2222  (password: demo)
# Audit:  tail -f /var/log/prism4d-audit/commands.log
##############################################################################

PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_DIR="${PROJ_ROOT}/release/docker"
DATE_TAG=$(date +%Y%m%d)

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PRISM4D Demo — Docker Image Builder & Deployer                 ║"
echo "║  ${DATE_TAG}                                                    ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# ── 0. Pre-flight ─────────────────────────────────────────────────────
echo "[0/6] Pre-flight checks..."
command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 1; }
docker info > /dev/null 2>&1 || { echo "ERROR: docker daemon not running"; exit 1; }

# Check NVIDIA Container Toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "WARNING: NVIDIA Container Toolkit may not be configured."
    echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    echo "Continuing anyway..."
fi

# ── 1. Build demo release if needed ──────────────────────────────────
DEMO_DIR="${PROJ_ROOT}/release/dist/prism4d-demo-${DATE_TAG}"
if [ ! -d "${DEMO_DIR}" ]; then
    echo "[1/6] Building demo release..."
    bash "${PROJ_ROOT}/release/build_demo_release.sh" "${PROJ_ROOT}/release/dist"
else
    echo "[1/6] Demo release already exists: ${DEMO_DIR}"
fi

# ── 2. Stage Docker build context ────────────────────────────────────
echo "[2/6] Staging Docker build context..."
BUILD_CTX="${DOCKER_DIR}/.build-context"
rm -rf "${BUILD_CTX}"
mkdir -p "${BUILD_CTX}"

# Copy Docker files
cp "${DOCKER_DIR}/Dockerfile" "${BUILD_CTX}/"
cp "${DOCKER_DIR}/sshd_config" "${BUILD_CTX}/"
cp "${DOCKER_DIR}/motd" "${BUILD_CTX}/"
cp "${DOCKER_DIR}/quickstart.md" "${BUILD_CTX}/"
cp "${DOCKER_DIR}/entrypoint.sh" "${BUILD_CTX}/"
chmod +x "${BUILD_CTX}/entrypoint.sh"

# Copy demo release
cp -r "${DEMO_DIR}" "${BUILD_CTX}/prism4d-demo"

# Copy conda env definition
mkdir -p "${BUILD_CTX}/envs"
cp "${PROJ_ROOT}/envs/preprocessing.yml" "${BUILD_CTX}/envs/" 2>/dev/null || \
    echo "WARNING: preprocessing.yml not found — conda env will use fallback"

# Stage sample topology files for the demo
echo "  Staging sample structures..."
mkdir -p "${BUILD_CTX}/samples"

# Pick a representative set of completed targets as samples
RESULTS_DIR="${PROJ_ROOT}/benchmarks/cryptobench/results"
TOPO_DIR="${PROJ_ROOT}/benchmarks/cryptobench/topologies"
SAMPLE_COUNT=0

for tgt in 2fhz 1arl 2x47 4ttp 5b0e 6eqj 1r3m 2fem; do
    TOPO="${TOPO_DIR}/${tgt}.topology.json"
    if [ -f "$TOPO" ]; then
        cp "$TOPO" "${BUILD_CTX}/samples/"
        ((SAMPLE_COUNT++))
        echo "  + ${tgt}.topology.json"
    fi
done

# Also include a sample binding_sites.json so they can see output format
for tgt in 2fhz 1arl; do
    BS="${RESULTS_DIR}/${tgt}/${tgt}.binding_sites.json"
    if [ -f "$BS" ]; then
        mkdir -p "${BUILD_CTX}/samples/example_results/${tgt}"
        cp "$BS" "${BUILD_CTX}/samples/example_results/${tgt}/"
        echo "  + ${tgt} example results"
    fi
done

echo "  ${SAMPLE_COUNT} sample topologies staged"

# ── 3. Build Docker image ────────────────────────────────────────────
echo "[3/6] Building Docker image (this may take 5-15 minutes)..."
cd "${BUILD_CTX}"
docker build \
    --tag prism4d-demo:latest \
    --tag prism4d-demo:${DATE_TAG} \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    . 2>&1 | tail -20

echo "  Image built: prism4d-demo:latest"
docker images prism4d-demo --format "  Size: {{.Size}}"

# ── 4. Create host audit directory ────────────────────────────────────
echo "[4/6] Setting up host audit logging..."
sudo mkdir -p /var/log/prism4d-audit
sudo chmod 755 /var/log/prism4d-audit
sudo chown "$(whoami):$(whoami)" /var/log/prism4d-audit

# Initialize audit log
cat > /var/log/prism4d-audit/system.log << EOF
[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PRISM4D Demo Audit System Initialized
[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Host: $(hostname)
[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Built: ${DATE_TAG}
[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)
[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo unknown)
EOF

touch /var/log/prism4d-audit/sessions.log \
      /var/log/prism4d-audit/commands.log \
      /var/log/prism4d-audit/sftp.log \
      /var/log/prism4d-audit/gpu.log

echo "  Audit logs: /var/log/prism4d-audit/"
echo "    system.log   — container lifecycle events"
echo "    sessions.log — SSH login/logout with IP, duration"
echo "    commands.log — every command the demo user runs"
echo "    gpu.log      — GPU utilization during sessions"

# ── 5. Stop existing container if running ─────────────────────────────
echo "[5/6] Managing container..."
if docker ps -q -f name=prism4d-demo | grep -q .; then
    echo "  Stopping existing container..."
    docker compose -f "${DOCKER_DIR}/docker-compose.yml" down
fi

# ── 6. Start the container ───────────────────────────────────────────
echo "[6/6] Starting PRISM4D demo container..."
cd "${DOCKER_DIR}"
docker compose up -d

# Wait for SSH to be ready
echo "  Waiting for SSH daemon..."
for i in $(seq 1 15); do
    if ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no \
           -o UserKnownHostsFile=/dev/null \
           demo@localhost -p 2222 exit 2>/dev/null; then
        break
    fi
    sleep 1
done

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PRISM4D DEMO SERVER RUNNING                                    ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║                                                                 ║"
echo "║  SSH Access (local):                                            ║"
echo "║    ssh demo@localhost -p 2222                                   ║"
echo "║    Password: demo                                               ║"
echo "║                                                                 ║"
echo "║  SSH Access (remote):                                           ║"
echo "║    ssh demo@$(hostname -I | awk '{print $1}') -p 2222                            ║"
echo "║    Password: demo                                               ║"
echo "║                                                                 ║"
echo "║  Audit Logs (YOUR EYES ONLY):                                   ║"
echo "║    tail -f /var/log/prism4d-audit/commands.log  # live commands ║"
echo "║    tail -f /var/log/prism4d-audit/sessions.log  # logins       ║"
echo "║    cat /var/log/prism4d-audit/gpu.log           # GPU usage    ║"
echo "║                                                                 ║"
echo "║  Management:                                                    ║"
echo "║    docker compose -f release/docker/docker-compose.yml logs -f  ║"
echo "║    docker compose -f release/docker/docker-compose.yml down     ║"
echo "║    docker compose -f release/docker/docker-compose.yml restart  ║"
echo "║                                                                 ║"
echo "║  Security:                                                      ║"
echo "║    - Client is inside a Docker container (no host access)       ║"
echo "║    - No sudo, no package installs, limited capabilities         ║"
echo "║    - PID limit: 100, Memory limit: 16GB                        ║"
echo "║    - All commands logged to /var/log/prism4d-audit/             ║"
echo "║    - SSH tunneling/forwarding disabled                          ║"
echo "║    - Audit logs are invisible to the demo user                  ║"
echo "║                                                                 ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

# Clean up build context
rm -rf "${BUILD_CTX}"
