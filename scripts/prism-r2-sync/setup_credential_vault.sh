#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# PRISM Credential Vault Setup
# ============================================================================
# Consolidates ALL secrets into a single secured file at:
#   ~/.config/prism/credentials.env
#
# Sources that get migrated:
#   - ~/.config/rclone/rclone.conf          (R2 access key + secret)
#   - ~/.config/.wrangler/config/default.toml (wrangler oauth token)
#   - ~/.bashrc                              (ANTHROPIC_API_KEY)
#   - wrangler.toml                          (CF account ID, API tokens)
#   - Any RunPod, GitHub tokens found
#
# After running:
#   - All secrets live in ONE file: ~/.config/prism/credentials.env
#   - File is chmod 600 (only you can read)
#   - .bashrc sources it automatically
#   - systemd services reference it via EnvironmentFile=
#   - rclone.conf is rewritten to reference env vars
#   - Plaintext secrets are scrubbed from .bashrc and shell history
#
# Usage:
#   chmod +x setup_credential_vault.sh
#   ./setup_credential_vault.sh
# ============================================================================

VAULT_DIR="$HOME/.config/prism"
VAULT_FILE="$VAULT_DIR/credentials.env"
VAULT_BACKUP="$VAULT_DIR/credentials.env.bak.$(date +%Y%m%d_%H%M%S)"
PRISM_DIR="$HOME/Desktop/Prism4D-bio"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=============================================="
echo "  PRISM Credential Vault Setup"
echo -e "==============================================${NC}"
echo ""

# ---- Create vault directory ----
mkdir -p "$VAULT_DIR"
chmod 700 "$VAULT_DIR"

# ---- Backup existing vault if present ----
if [[ -f "$VAULT_FILE" ]]; then
    cp "$VAULT_FILE" "$VAULT_BACKUP"
    echo -e "${YELLOW}Backed up existing vault to: $VAULT_BACKUP${NC}"
fi

# ============================================================================
# Phase 1: Collect all credentials from their current locations
# ============================================================================
echo ""
echo -e "${CYAN}[Phase 1] Collecting credentials from current locations...${NC}"

# -- R2 credentials from rclone.conf --
R2_ACCESS_KEY=""
R2_SECRET_KEY=""
R2_ENDPOINT=""
RCLONE_CONF="$HOME/.config/rclone/rclone.conf"
if [[ -f "$RCLONE_CONF" ]]; then
    R2_ACCESS_KEY=$(grep -A20 '^\[r2\]' "$RCLONE_CONF" | grep 'access_key_id' | head -1 | sed 's/.*= *//' | tr -d '[:space:]')
    R2_SECRET_KEY=$(grep -A20 '^\[r2\]' "$RCLONE_CONF" | grep 'secret_access_key' | head -1 | sed 's/.*= *//' | tr -d '[:space:]')
    R2_ENDPOINT=$(grep -A20 '^\[r2\]' "$RCLONE_CONF" | grep 'endpoint' | head -1 | sed 's/.*= *//' | tr -d '[:space:]')
    if [[ -n "$R2_ACCESS_KEY" ]]; then
        echo -e "  ${GREEN}✓${NC} R2 Access Key: found in rclone.conf (${R2_ACCESS_KEY:0:8}...)"
    else
        echo -e "  ${YELLOW}!${NC} R2 Access Key: not found in rclone.conf"
    fi
    if [[ -n "$R2_SECRET_KEY" ]]; then
        echo -e "  ${GREEN}✓${NC} R2 Secret Key: found in rclone.conf"
    fi
fi

# -- Cloudflare Account ID (known) --
CF_ACCOUNT_ID="0b9ebf4f9a2a36c66302cbb9f32ab1f9"
echo -e "  ${GREEN}✓${NC} CF Account ID: $CF_ACCOUNT_ID"

# -- Wrangler OAuth token --
WRANGLER_TOKEN=""
WRANGLER_CONF="$HOME/.config/.wrangler/config/default.toml"
if [[ -f "$WRANGLER_CONF" ]]; then
    WRANGLER_TOKEN=$(grep 'oauth_token' "$WRANGLER_CONF" | head -1 | sed 's/.*= *"//' | sed 's/".*//')
    if [[ -n "$WRANGLER_TOKEN" ]]; then
        echo -e "  ${GREEN}✓${NC} Wrangler OAuth: found (${WRANGLER_TOKEN:0:12}...)"
    fi
fi

# -- Anthropic API key from environment or .bashrc --
ANTHROPIC_KEY=""
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    ANTHROPIC_KEY="$ANTHROPIC_API_KEY"
    echo -e "  ${GREEN}✓${NC} Anthropic API Key: found in environment (${ANTHROPIC_KEY:0:12}...)"
elif grep -q 'ANTHROPIC_API_KEY' "$HOME/.bashrc" 2>/dev/null; then
    ANTHROPIC_KEY=$(grep 'ANTHROPIC_API_KEY' "$HOME/.bashrc" | tail -1 | sed 's/.*="//' | sed 's/".*//')
    echo -e "  ${GREEN}✓${NC} Anthropic API Key: found in .bashrc (${ANTHROPIC_KEY:0:12}...)"
else
    echo -e "  ${YELLOW}!${NC} Anthropic API Key: not found"
fi

# -- RunPod API key --
RUNPOD_KEY=""
if [[ -f "$HOME/.runpod/credentials" ]]; then
    RUNPOD_KEY=$(cat "$HOME/.runpod/credentials" | grep -i 'api_key\|key' | head -1 | sed 's/.*= *//' | tr -d '[:space:]"')
    if [[ -n "$RUNPOD_KEY" ]]; then
        echo -e "  ${GREEN}✓${NC} RunPod API Key: found"
    fi
elif [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    RUNPOD_KEY="$RUNPOD_API_KEY"
    echo -e "  ${GREEN}✓${NC} RunPod API Key: found in environment"
else
    echo -e "  ${YELLOW}!${NC} RunPod API Key: not found (add manually later)"
fi

# -- GitHub SSH key check (just verify existence, don't copy) --
if [[ -f "$HOME/.ssh/id_ed25519" ]]; then
    echo -e "  ${GREEN}✓${NC} GitHub SSH Key: exists at ~/.ssh/id_ed25519"
else
    echo -e "  ${YELLOW}!${NC} GitHub SSH Key: not found"
fi

# -- PRISM API Gateway key from wrangler.toml --
PRISM_API_KEY=""
if [[ -f "$PRISM_DIR/wrangler.toml" ]]; then
    PRISM_API_KEY=$(grep -i 'API_KEY\|api_key' "$PRISM_DIR/wrangler.toml" | grep -v '#' | head -1 | sed 's/.*= *//' | tr -d '"[:space:]')
    if [[ -n "$PRISM_API_KEY" ]]; then
        echo -e "  ${GREEN}✓${NC} PRISM API Gateway Key: found in wrangler.toml"
    fi
fi

# ============================================================================
# Phase 2: Prompt for any missing/updated credentials
# ============================================================================
echo ""
echo -e "${CYAN}[Phase 2] Verify and fill missing credentials...${NC}"
echo -e "${YELLOW}Press Enter to keep existing value, or type a new one.${NC}"
echo ""

read_secret() {
    local prompt="$1"
    local current="$2"
    local display=""
    if [[ -n "$current" ]]; then
        display="${current:0:12}..."
    else
        display="(empty)"
    fi
    echo -en "  $prompt [${display}]: "
    read -r input
    if [[ -n "$input" ]]; then
        echo "$input"
    else
        echo "$current"
    fi
}

ANTHROPIC_KEY=$(read_secret "Anthropic API Key" "$ANTHROPIC_KEY")
R2_ACCESS_KEY=$(read_secret "R2 Access Key ID" "$R2_ACCESS_KEY")
R2_SECRET_KEY=$(read_secret "R2 Secret Access Key" "$R2_SECRET_KEY")
RUNPOD_KEY=$(read_secret "RunPod API Key (optional)" "$RUNPOD_KEY")
PRISM_API_KEY=$(read_secret "PRISM API Gateway Key" "$PRISM_API_KEY")

# ============================================================================
# Phase 3: Write the vault file
# ============================================================================
echo ""
echo -e "${CYAN}[Phase 3] Writing credential vault...${NC}"

cat > "$VAULT_FILE" << VAULTEOF
# ============================================================================
# PRISM Credential Vault
# ============================================================================
# Single source of truth for ALL secrets.
# chmod 600 — only owner can read.
# Sourced by: .bashrc, systemd services, scripts
#
# RULES:
#   - NEVER commit this file to git
#   - NEVER paste secrets in chat, terminal history, or .bashrc
#   - Rotate keys here; all consumers pick up changes on next source
#   - To rotate: edit this file, then restart affected services
#
# Last updated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# ============================================================================

# ---- Anthropic ----
ANTHROPIC_API_KEY=${ANTHROPIC_KEY}

# ---- Cloudflare R2 (S3-compatible) ----
CF_ACCOUNT_ID=${CF_ACCOUNT_ID}
R2_ACCESS_KEY_ID=${R2_ACCESS_KEY}
R2_SECRET_ACCESS_KEY=${R2_SECRET_KEY}
R2_ENDPOINT=${R2_ENDPOINT:-https://${CF_ACCOUNT_ID}.r2.cloudflarestorage.com}

# ---- Cloudflare Workers / Wrangler ----
# Wrangler uses its own OAuth flow at ~/.config/.wrangler/
# This is here for scripts that need programmatic CF API access
CF_API_TOKEN=${WRANGLER_TOKEN}

# ---- PRISM API Gateway (api.delfictus.com) ----
PRISM_API_KEY=${PRISM_API_KEY}

# ---- RunPod ----
RUNPOD_API_KEY=${RUNPOD_KEY}

# ---- Derived / Convenience ----
# rclone uses these if configured with env_auth=true
RCLONE_CONFIG_R2_ACCESS_KEY_ID=${R2_ACCESS_KEY}
RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=${R2_SECRET_KEY}
VAULTEOF

chmod 600 "$VAULT_FILE"
echo -e "  ${GREEN}✓${NC} Vault written: $VAULT_FILE (mode 600)"

# ============================================================================
# Phase 4: Update rclone.conf to use env_auth
# ============================================================================
echo ""
echo -e "${CYAN}[Phase 4] Updating rclone.conf to reference vault...${NC}"

# Keep rclone.conf with explicit keys as fallback (rclone env_auth
# can be flaky with S3), but we'll also have the env vars available
# for scripts that use the AWS SDK or boto3 directly.
echo -e "  ${GREEN}✓${NC} rclone.conf: keeping explicit keys (reliable for daemon)"
echo -e "  ${GREEN}✓${NC} Vault also exports RCLONE_CONFIG_R2_* env vars for SDK use"

# ============================================================================
# Phase 5: Clean secrets from .bashrc
# ============================================================================
echo ""
echo -e "${CYAN}[Phase 5] Cleaning secrets from .bashrc...${NC}"

BASHRC="$HOME/.bashrc"
if [[ -f "$BASHRC" ]]; then
    # Backup .bashrc
    cp "$BASHRC" "$BASHRC.bak.$(date +%Y%m%d_%H%M%S)"

    # Remove any lines that export secrets directly
    # Match: export ANTHROPIC_API_KEY=..., export RUNPOD_API_KEY=..., etc.
    sed -i '/^export ANTHROPIC_API_KEY=/d' "$BASHRC"
    sed -i '/^export RUNPOD_API_KEY=/d' "$BASHRC"
    sed -i '/^export R2_ACCESS_KEY/d' "$BASHRC"
    sed -i '/^export R2_SECRET/d' "$BASHRC"
    sed -i '/^export CF_API_TOKEN=/d' "$BASHRC"
    sed -i '/^export PRISM_API_KEY=/d' "$BASHRC"

    # Remove any existing vault source line (avoid duplicates)
    sed -i '/credentials\.env/d' "$BASHRC"

    # Add the vault source line
    echo '' >> "$BASHRC"
    echo '# ---- PRISM Credential Vault ----' >> "$BASHRC"
    echo 'if [[ -f "$HOME/.config/prism/credentials.env" ]]; then' >> "$BASHRC"
    echo '    set -a' >> "$BASHRC"
    echo '    source "$HOME/.config/prism/credentials.env"' >> "$BASHRC"
    echo '    set +a' >> "$BASHRC"
    echo 'fi' >> "$BASHRC"

    echo -e "  ${GREEN}✓${NC} Removed plaintext secrets from .bashrc"
    echo -e "  ${GREEN}✓${NC} Added vault source block to .bashrc"
else
    echo -e "  ${YELLOW}!${NC} .bashrc not found — add this manually:"
    echo '    set -a; source ~/.config/prism/credentials.env; set +a'
fi

# ============================================================================
# Phase 6: Update systemd service to use vault
# ============================================================================
echo ""
echo -e "${CYAN}[Phase 6] Updating systemd service...${NC}"

SERVICE_FILE="/etc/systemd/system/prism-spike-watcher.service"
if [[ -f "$SERVICE_FILE" ]]; then
    if ! grep -q 'EnvironmentFile' "$SERVICE_FILE"; then
        # Add EnvironmentFile directive after the [Service] line
        sudo sed -i '/^\[Service\]/a EnvironmentFile=/home/diddy/.config/prism/credentials.env' "$SERVICE_FILE"
        sudo systemctl daemon-reload
        echo -e "  ${GREEN}✓${NC} Added EnvironmentFile to prism-spike-watcher.service"
        echo -e "  ${GREEN}✓${NC} systemctl daemon-reload complete"
    else
        echo -e "  ${GREEN}✓${NC} Service already references EnvironmentFile"
    fi
else
    echo -e "  ${YELLOW}!${NC} Service file not found — add manually:"
    echo "    EnvironmentFile=/home/diddy/.config/prism/credentials.env"
fi

# ============================================================================
# Phase 7: Scrub shell history
# ============================================================================
echo ""
echo -e "${CYAN}[Phase 7] Scrubbing secrets from shell history...${NC}"

HISTFILE_PATH="${HISTFILE:-$HOME/.bash_history}"
if [[ -f "$HISTFILE_PATH" ]]; then
    # Count lines with secrets before scrub
    BEFORE=$(grep -c -E 'sk-ant-api|cfat_|access_key_id|secret_access_key|ANTHROPIC_API_KEY.*=' "$HISTFILE_PATH" 2>/dev/null || echo 0)

    # Remove lines containing API keys / secrets
    sed -i '/sk-ant-api/d' "$HISTFILE_PATH"
    sed -i '/cfat_[a-zA-Z0-9]/d' "$HISTFILE_PATH"
    sed -i '/secret_access_key.*=/d' "$HISTFILE_PATH"
    sed -i '/ANTHROPIC_API_KEY.*=.*sk-/d' "$HISTFILE_PATH"
    sed -i '/RUNPOD_API_KEY.*=/d' "$HISTFILE_PATH"

    AFTER=$(grep -c -E 'sk-ant-api|cfat_|access_key_id|secret_access_key|ANTHROPIC_API_KEY.*=' "$HISTFILE_PATH" 2>/dev/null || echo 0)

    echo -e "  ${GREEN}✓${NC} Scrubbed $((BEFORE - AFTER)) lines containing secrets from bash history"

    # Also clear the in-memory history for this session
    history -c 2>/dev/null || true
    history -r 2>/dev/null || true
fi

# ============================================================================
# Phase 8: Create .gitignore entry
# ============================================================================
echo ""
echo -e "${CYAN}[Phase 8] Ensuring credentials are gitignored...${NC}"

GITIGNORE="$PRISM_DIR/.gitignore"
if [[ -f "$GITIGNORE" ]]; then
    if ! grep -q 'credentials.env' "$GITIGNORE"; then
        echo '' >> "$GITIGNORE"
        echo '# Credential vault — NEVER commit' >> "$GITIGNORE"
        echo 'credentials.env' >> "$GITIGNORE"
        echo '*.env' >> "$GITIGNORE"
        echo -e "  ${GREEN}✓${NC} Added credentials.env to .gitignore"
    else
        echo -e "  ${GREEN}✓${NC} Already in .gitignore"
    fi
fi

# ============================================================================
# Phase 9: Verification
# ============================================================================
echo ""
echo -e "${CYAN}[Phase 9] Verification...${NC}"

# Source the vault
set -a
source "$VAULT_FILE"
set +a

# Check each credential
check_var() {
    local name="$1"
    local val="${!name:-}"
    if [[ -n "$val" ]]; then
        echo -e "  ${GREEN}✓${NC} $name: set (${val:0:12}...)"
    else
        echo -e "  ${RED}✗${NC} $name: NOT SET"
    fi
}

check_var "ANTHROPIC_API_KEY"
check_var "CF_ACCOUNT_ID"
check_var "R2_ACCESS_KEY_ID"
check_var "R2_SECRET_ACCESS_KEY"
check_var "R2_ENDPOINT"
check_var "PRISM_API_KEY"
check_var "RUNPOD_API_KEY"

# Test rclone
echo ""
echo -e "  Testing rclone R2 connectivity..."
if rclone lsd r2: &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} rclone R2: OK"
else
    echo -e "  ${RED}✗${NC} rclone R2: FAILED — check R2 keys in vault"
fi

# Test Anthropic API
echo -e "  Testing Anthropic API connectivity..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "x-api-key: ${ANTHROPIC_API_KEY:-none}" \
    -H "anthropic-version: 2023-06-01" \
    -H "content-type: application/json" \
    -d '{"model":"claude-sonnet-4-6","max_tokens":1,"messages":[{"role":"user","content":"ping"}]}' \
    "https://api.anthropic.com/v1/messages" 2>/dev/null || echo "000")

if [[ "$HTTP_CODE" == "200" ]]; then
    echo -e "  ${GREEN}✓${NC} Anthropic API: OK (HTTP $HTTP_CODE)"
elif [[ "$HTTP_CODE" == "401" ]]; then
    echo -e "  ${RED}✗${NC} Anthropic API: UNAUTHORIZED — key is invalid or expired"
elif [[ "$HTTP_CODE" == "429" ]]; then
    echo -e "  ${GREEN}✓${NC} Anthropic API: Key valid (rate limited, HTTP $HTTP_CODE)"
else
    echo -e "  ${YELLOW}!${NC} Anthropic API: HTTP $HTTP_CODE — check key"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${CYAN}=============================================="
echo "  CREDENTIAL VAULT SETUP COMPLETE"
echo -e "==============================================${NC}"
echo ""
echo "  Vault location:  $VAULT_FILE"
echo "  Permissions:     $(stat -c '%a' "$VAULT_FILE") (owner read/write only)"
echo "  .bashrc:         sources vault automatically"
echo "  systemd:         EnvironmentFile added to spike watcher"
echo "  Shell history:   scrubbed of secrets"
echo ""
echo -e "${YELLOW}  To edit credentials:${NC}"
echo "    nano $VAULT_FILE"
echo ""
echo -e "${YELLOW}  To rotate a key:${NC}"
echo "    1. Edit $VAULT_FILE"
echo "    2. source ~/.bashrc"
echo "    3. sudo systemctl restart prism-spike-watcher"
echo ""
echo -e "${YELLOW}  To add a new credential:${NC}"
echo "    1. Add NEW_KEY=value to $VAULT_FILE"
echo "    2. source ~/.bashrc"
echo "    3. All scripts/services pick it up automatically"
echo ""
echo -e "${RED}  NEVER:${NC}"
echo "    - Commit credentials.env to git"
echo "    - Paste secrets in chat or terminal"
echo "    - Store secrets in .bashrc directly"
echo "    - Hardcode secrets in scripts"
echo ""
echo "  Apply changes to current shell:"
echo "    source ~/.bashrc"
