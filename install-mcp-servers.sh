#!/bin/bash
set -euo pipefail

# ============================================================
# PRISM-4D SOTA MCP Server Installation
# Installs: AlphaFold, UniProt, PubChem, STRING, gget
# ============================================================

MCP_DIR="$HOME/mcp-servers"
mkdir -p "$MCP_DIR"

echo "============================================"
echo "  PRISM-4D MCP Server Installation"
echo "============================================"

# --- Pre-flight checks ---
echo ""
echo "[1/8] Pre-flight checks..."

# Check Node.js
if ! command -v node &>/dev/null; then
    echo "ERROR: Node.js not found. Install with: sudo apt install nodejs npm"
    exit 1
fi
NODE_VER=$(node --version)
echo "  Node.js: $NODE_VER"

# Check npm
if ! command -v npm &>/dev/null; then
    echo "ERROR: npm not found. Install with: sudo apt install npm"
    exit 1
fi
echo "  npm: $(npm --version)"

# Check uv (for gget-mcp)
if ! command -v uv &>/dev/null; then
    echo "  uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv: $(uv --version)"

# Check git
if ! command -v git &>/dev/null; then
    echo "ERROR: git not found."
    exit 1
fi
echo "  git: $(git --version | head -c 20)"

echo ""
echo "[2/8] Installing AlphaFold MCP Server..."
cd "$MCP_DIR"
if [ -d "AlphaFold-MCP-Server" ]; then
    echo "  Already cloned, pulling latest..."
    cd AlphaFold-MCP-Server && git pull
else
    git clone https://github.com/Augmented-Nature/AlphaFold-MCP-Server.git
    cd AlphaFold-MCP-Server
fi
npm install --silent 2>&1 | tail -1
npm run build 2>&1 | tail -1
echo "  ✔ AlphaFold ready: $MCP_DIR/AlphaFold-MCP-Server/build/index.js"

echo ""
echo "[3/8] Installing UniProt MCP Server..."
cd "$MCP_DIR"
if [ -d "UniProt-MCP-Server" ]; then
    echo "  Already cloned, pulling latest..."
    cd UniProt-MCP-Server && git pull
else
    git clone https://github.com/Augmented-Nature/Augmented-Nature-UniProt-MCP-Server.git UniProt-MCP-Server
    cd UniProt-MCP-Server
fi
npm install --silent 2>&1 | tail -1
npm run build 2>&1 | tail -1
echo "  ✔ UniProt ready: $MCP_DIR/UniProt-MCP-Server/build/index.js"

echo ""
echo "[4/8] Installing PubChem MCP Server..."
cd "$MCP_DIR"
if [ -d "PubChem-MCP-Server" ]; then
    echo "  Already cloned, pulling latest..."
    cd PubChem-MCP-Server && git pull
else
    git clone https://github.com/Augmented-Nature/PubChem-MCP-Server.git
    cd PubChem-MCP-Server
fi
npm install --silent 2>&1 | tail -1
npm run build 2>&1 | tail -1
echo "  ✔ PubChem ready: $MCP_DIR/PubChem-MCP-Server/build/index.js"

echo ""
echo "[5/8] Installing STRING-db MCP Server..."
cd "$MCP_DIR"
if [ -d "STRING-db-MCP-Server" ]; then
    echo "  Already cloned, pulling latest..."
    cd STRING-db-MCP-Server && git pull
else
    git clone https://github.com/Augmented-Nature/STRING-db-MCP-Server.git
    cd STRING-db-MCP-Server
fi
npm install --silent 2>&1 | tail -1
npm run build 2>&1 | tail -1
echo "  ✔ STRING-db ready: $MCP_DIR/STRING-db-MCP-Server/build/index.js"

echo ""
echo "[6/8] Installing gget MCP Server (Holy Bio)..."
cd "$MCP_DIR"
if [ -d "gget-mcp" ]; then
    echo "  Already cloned, pulling latest..."
    cd gget-mcp && git pull
else
    git clone https://github.com/longevity-genie/gget-mcp.git
    cd gget-mcp
fi
uv sync 2>&1 | tail -3
echo "  ✔ gget ready: $MCP_DIR/gget-mcp"

echo ""
echo "[7/8] Verifying all builds..."
FAIL=0

for server in AlphaFold-MCP-Server UniProt-MCP-Server PubChem-MCP-Server STRING-db-MCP-Server; do
    if [ -f "$MCP_DIR/$server/build/index.js" ]; then
        echo "  ✔ $server/build/index.js exists"
    else
        echo "  ✘ $server/build/index.js MISSING"
        FAIL=1
    fi
done

if [ -f "$MCP_DIR/gget-mcp/pyproject.toml" ]; then
    echo "  ✔ gget-mcp/pyproject.toml exists"
else
    echo "  ✘ gget-mcp MISSING"
    FAIL=1
fi

if [ "$FAIL" -eq 1 ]; then
    echo ""
    echo "ERROR: Some servers failed to build. Check output above."
    exit 1
fi

echo ""
echo "[8/8] Generating managed-mcp.json..."

cat > /tmp/managed-mcp.json << 'MCPEOF'
{
  "mcpServers": {
    "rcsb-pdb": {
      "type": "sse",
      "url": "https://rcsb-pdb-mcp-server.quentincody.workers.dev/sse"
    },
    "tooluniverse-literature": {
      "type": "stdio",
      "command": "tooluniverse-smcp-stdio",
      "args": ["--compact-mode", "--include-tools", "EuropePMC_search_articles,openalex_literature_search"]
    },
    "tooluniverse-drugs": {
      "type": "stdio",
      "command": "tooluniverse-smcp-stdio",
      "args": ["--compact-mode", "--include-tools", "ChEMBL_search_similar_molecules,search_clinical_trials"]
    },
    "pymol": {
      "type": "stdio",
      "command": "mcp",
      "args": ["run", "PYMOL_PATH/pymol_server.py"]
    },
    "alphafold": {
      "type": "stdio",
      "command": "node",
      "args": ["MCP_DIR/AlphaFold-MCP-Server/build/index.js"]
    },
    "uniprot": {
      "type": "stdio",
      "command": "node",
      "args": ["MCP_DIR/UniProt-MCP-Server/build/index.js"]
    },
    "pubchem": {
      "type": "stdio",
      "command": "node",
      "args": ["MCP_DIR/PubChem-MCP-Server/build/index.js"]
    },
    "string-db": {
      "type": "stdio",
      "command": "node",
      "args": ["MCP_DIR/STRING-db-MCP-Server/build/index.js"]
    },
    "gget": {
      "type": "stdio",
      "command": "UV_PATH",
      "args": ["--directory", "MCP_DIR/gget-mcp", "run", "stdio"]
    }
  }
}
MCPEOF

# Replace placeholders with actual paths
HOME_ESC=$(echo "$HOME" | sed 's/\//\\\//g')
MCP_ESC=$(echo "$MCP_DIR" | sed 's/\//\\\//g')
PYMOL_PATH="$HOME/molecule-mcp"
PYMOL_ESC=$(echo "$PYMOL_PATH" | sed 's/\//\\\//g')
UV_PATH=$(which uv)
UV_ESC=$(echo "$UV_PATH" | sed 's/\//\\\//g')

sed -i "s/MCP_DIR/$MCP_ESC/g" /tmp/managed-mcp.json
sed -i "s/PYMOL_PATH/$PYMOL_ESC/g" /tmp/managed-mcp.json
sed -i "s/UV_PATH/$UV_ESC/g" /tmp/managed-mcp.json

# Validate JSON
if jq . /tmp/managed-mcp.json > /dev/null 2>&1; then
    echo "  ✔ JSON valid"
else
    echo "  ✘ JSON invalid!"
    exit 1
fi

echo ""
echo "============================================"
echo "  Installation Complete"
echo "============================================"
echo ""
echo "Generated config at: /tmp/managed-mcp.json"
echo ""
echo "To deploy, run:"
echo "  sudo cp /tmp/managed-mcp.json /etc/claude-code/managed-mcp.json"
echo ""
echo "Then restart Claude Code and run /mcp to verify all 9 servers connect."
echo ""
echo "Servers installed:"
echo "  1. rcsb-pdb        (SSE - existing)"
echo "  2. tooluniverse-lit (stdio - existing)"
echo "  3. tooluniverse-drugs (stdio - existing)"
echo "  4. pymol           (stdio - existing)"
echo "  5. alphafold        (stdio - NEW)"
echo "  6. uniprot          (stdio - NEW)"
echo "  7. pubchem          (stdio - NEW)"
echo "  8. string-db        (stdio - NEW)"
echo "  9. gget             (stdio - NEW)"
echo " 10. bioRxiv          (claude.ai connector)"
