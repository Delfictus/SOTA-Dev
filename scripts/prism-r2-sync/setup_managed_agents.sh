#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# PRISM Managed Agents Setup
# ============================================================================
# Creates PRISM-DataOps and CampaignRunner agents on Anthropic's cloud
# using the Managed Agents REST API (curl-based, not SDK).
#
# Prerequisites:
#   - ANTHROPIC_API_KEY set (via credential vault or export)
#   - jq installed
#
# Usage:
#   ./setup_managed_agents.sh
# ============================================================================

BETA_HEADER="managed-agents-2026-04-01"
API_BASE="https://api.anthropic.com/v1"
STATE_FILE="$HOME/.config/prism/managed-agents-state.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---- Preflight checks ----
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo -e "${RED}ERROR: ANTHROPIC_API_KEY not set. Run: source ~/.bashrc${NC}"
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo -e "${RED}ERROR: jq not installed. Run: sudo apt install jq${NC}"
    exit 1
fi

api_call() {
    local method="$1"
    local endpoint="$2"
    local data="${3:-}"

    local args=(
        -sS --fail-with-body
        -X "$method"
        -H "x-api-key: $ANTHROPIC_API_KEY"
        -H "anthropic-version: 2023-06-01"
        -H "anthropic-beta: $BETA_HEADER"
        -H "content-type: application/json"
    )

    if [[ -n "$data" ]]; then
        args+=(-d "$data")
    fi

    curl "${args[@]}" "${API_BASE}${endpoint}"
}

echo -e "${CYAN}=============================================="
echo "  PRISM Managed Agents Setup"
echo -e "==============================================${NC}"
echo ""

# ============================================================================
# Step 1: Create Environment
# ============================================================================
echo -e "${CYAN}[1/4] Creating environment...${NC}"

ENV_RESPONSE=$(api_call POST "/environments" '{
    "name": "prism-dataops-env",
    "config": {
        "type": "cloud",
        "networking": {"type": "unrestricted"},
        "packages": {
            "pip": ["pyarrow", "pandas", "numpy", "requests", "boto3"]
        }
    }
}')

ENVIRONMENT_ID=$(echo "$ENV_RESPONSE" | jq -er '.id')
echo -e "  ${GREEN}✓${NC} Environment ID: $ENVIRONMENT_ID"

# ============================================================================
# Step 2: Create PRISM-DataOps Agent
# ============================================================================
echo -e "${CYAN}[2/4] Creating PRISM-DataOps agent...${NC}"

DATAOPS_RESPONSE=$(api_call POST "/agents" '{
    "name": "PRISM-DataOps",
    "model": "claude-sonnet-4-6",
    "system": "You are the PRISM-DataOps agent for Delfictus IO. Your role is to manage data pipelines for the PRISM-4D neuromorphic molecular dynamics platform.\n\nYour responsibilities:\n1. Monitor and validate spike event data uploads to Cloudflare R2 buckets\n2. Run post-processing on completed PRISM-TWIN engine outputs\n3. Convert spike JSON to Parquet (zstd, lossless) and verify row counts\n4. Generate data quality reports and manifest files\n5. Track R2 bucket inventory and flag missing data\n\nR2 Bucket Structure:\n- prism-archive: Raw data, append-only, forever (cryptobench199/, twin-runs/, v1.1-physics/, 10k-runs/)\n- prism-production: Pipeline artifacts, versioned (models/, evaluations/, registries/)\n- prism-public: Public-facing demo assets\n\nCRITICAL RULES:\n- NEVER delete anything from R2. R2 is append-only.\n- Always verify uploads (size match within 1%) before reporting success.\n- Both raw JSON and Parquet must exist on R2 for every spike dataset.\n- Flag any data integrity issues immediately.",
    "tools": [
        {"type": "agent_toolset_20260401"}
    ]
}')

DATAOPS_AGENT_ID=$(echo "$DATAOPS_RESPONSE" | jq -er '.id')
DATAOPS_AGENT_VERSION=$(echo "$DATAOPS_RESPONSE" | jq -er '.version')
echo -e "  ${GREEN}✓${NC} PRISM-DataOps Agent ID: $DATAOPS_AGENT_ID (v${DATAOPS_AGENT_VERSION})"

# ============================================================================
# Step 3: Create PRISM-CampaignRunner Agent
# ============================================================================
echo -e "${CYAN}[3/4] Creating PRISM-CampaignRunner agent...${NC}"

CAMPAIGN_RESPONSE=$(api_call POST "/agents" '{
    "name": "PRISM-CampaignRunner",
    "model": "claude-sonnet-4-6",
    "system": "You are the PRISM-CampaignRunner agent for Delfictus IO. Your role is to orchestrate and monitor PRISM-4D engine campaigns across protein targets.\n\nYour responsibilities:\n1. Track campaign progress across target lists (CryptoBench199, 10K, TWIN runs)\n2. Parse and validate engine output files (binding_sites.json, spike_events, THERM classifications)\n3. Generate campaign status reports with per-target success/failure tracking\n4. Identify failed or incomplete runs that need retry\n5. Compute aggregate statistics (SR@1, SR@3, AUROC) across completed targets\n6. Compare results against known benchmarks (CryptoSite, PocketMiner, P2Rank)\n\nKey metrics:\n- SR@1 (Success Rate at rank 1): Best historical = 53.1%\n- SR@3: Best historical = 75.0%\n- PRISM-AI Model B AUROC: 0.889 across 18 unseen targets\n\nOutput formats: JSON reports, Markdown summaries, CSV for downstream ML.\n\nAlways verify data against actual files, never fabricate results.",
    "tools": [
        {"type": "agent_toolset_20260401"}
    ]
}')

CAMPAIGN_AGENT_ID=$(echo "$CAMPAIGN_RESPONSE" | jq -er '.id')
CAMPAIGN_AGENT_VERSION=$(echo "$CAMPAIGN_RESPONSE" | jq -er '.version')
echo -e "  ${GREEN}✓${NC} PRISM-CampaignRunner Agent ID: $CAMPAIGN_AGENT_ID (v${CAMPAIGN_AGENT_VERSION})"

# ============================================================================
# Step 4: Save state
# ============================================================================
echo -e "${CYAN}[4/4] Saving agent state...${NC}"

mkdir -p "$(dirname "$STATE_FILE")"
cat > "$STATE_FILE" << STATEEOF
{
    "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "environment": {
        "id": "$ENVIRONMENT_ID",
        "name": "prism-dataops-env"
    },
    "agents": {
        "dataops": {
            "id": "$DATAOPS_AGENT_ID",
            "version": $DATAOPS_AGENT_VERSION,
            "name": "PRISM-DataOps"
        },
        "campaign_runner": {
            "id": "$CAMPAIGN_AGENT_ID",
            "version": $CAMPAIGN_AGENT_VERSION,
            "name": "PRISM-CampaignRunner"
        }
    }
}
STATEEOF

chmod 600 "$STATE_FILE"
echo -e "  ${GREEN}✓${NC} State saved: $STATE_FILE"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${CYAN}=============================================="
echo "  SETUP COMPLETE"
echo -e "==============================================${NC}"
echo ""
echo "  Environment:      $ENVIRONMENT_ID"
echo "  DataOps Agent:    $DATAOPS_AGENT_ID (v${DATAOPS_AGENT_VERSION})"
echo "  Campaign Agent:   $CAMPAIGN_AGENT_ID (v${CAMPAIGN_AGENT_VERSION})"
echo "  State file:       $STATE_FILE"
echo ""
echo -e "${YELLOW}  To start a DataOps session:${NC}"
echo ""
echo "    SESSION=\$(curl -sS --fail-with-body ${API_BASE}/sessions \\"
echo "      -H \"x-api-key: \$ANTHROPIC_API_KEY\" \\"
echo "      -H \"anthropic-version: 2023-06-01\" \\"
echo "      -H \"anthropic-beta: $BETA_HEADER\" \\"
echo "      -H \"content-type: application/json\" \\"
echo "      -d '{\"agent\": \"$DATAOPS_AGENT_ID\", \"environment_id\": \"$ENVIRONMENT_ID\"}')"
echo ""
echo "    SESSION_ID=\$(echo \$SESSION | jq -er '.id')"
echo ""
echo -e "${YELLOW}  To send a task:${NC}"
echo ""
echo "    curl -sS --fail-with-body ${API_BASE}/sessions/\$SESSION_ID/events \\"
echo "      -H \"x-api-key: \$ANTHROPIC_API_KEY\" \\"
echo "      -H \"anthropic-version: 2023-06-01\" \\"
echo "      -H \"anthropic-beta: $BETA_HEADER\" \\"
echo "      -H \"content-type: application/json\" \\"
echo "      -d '{\"events\": [{\"type\": \"user.message\", \"content\": [{\"type\": \"text\", \"text\": \"Check R2 bucket inventory\"}]}]}'"
echo ""
echo -e "${YELLOW}  To stream responses:${NC}"
echo ""
echo "    curl -fsSN ${API_BASE}/sessions/\$SESSION_ID/stream \\"
echo "      -H \"x-api-key: \$ANTHROPIC_API_KEY\" \\"
echo "      -H \"anthropic-version: 2023-06-01\" \\"
echo "      -H \"anthropic-beta: $BETA_HEADER\""
