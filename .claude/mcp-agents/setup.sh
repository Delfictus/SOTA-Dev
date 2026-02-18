#!/bin/bash
# Prism4D MCP Agents Setup Script
# Sets up RAG database and configures MCP servers for Claude Code

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "🚀 Prism4D MCP Agents Setup"
echo "=========================="
echo "Root: $PRISM_ROOT"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📌 Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q

echo "✅ Dependencies installed"

# Index the codebase
echo ""
echo "🔍 Indexing Prism4D codebase for RAG..."
echo "   This may take a few minutes on first run..."
python3 "$SCRIPT_DIR/indexer.py"

# Generate Claude Code MCP configuration
echo ""
echo "⚙️  Generating Claude Code configuration..."

CLAUDE_CONFIG_DIR="$HOME/.claude"
mkdir -p "$CLAUDE_CONFIG_DIR"

# Create MCP servers configuration
cat > "$SCRIPT_DIR/mcp_config.json" << EOF
{
  "mcpServers": {
    "prism-rag": {
      "command": "$VENV_DIR/bin/python",
      "args": ["$SCRIPT_DIR/prism_rag_server.py"],
      "description": "Prism4D RAG - Semantic code search with LanceDB"
    },
    "prism-cuda": {
      "command": "$VENV_DIR/bin/python",
      "args": ["$SCRIPT_DIR/cuda_agent_server.py"],
      "description": "CUDA/GPU Specialist - Kernel analysis and optimization"
    },
    "prism-bench": {
      "command": "$VENV_DIR/bin/python",
      "args": ["$SCRIPT_DIR/bench_agent_server.py"],
      "description": "Validation & Benchmarking - Metrics and testing"
    },
    "prism-orchestrator": {
      "command": "$VENV_DIR/bin/python",
      "args": ["$SCRIPT_DIR/orchestrator_server.py"],
      "description": "Agent Orchestrator - Coordinate specialist agents"
    }
  }
}
EOF

echo "✅ MCP configuration generated: $SCRIPT_DIR/mcp_config.json"

# Check if Claude settings exist and offer to merge
CLAUDE_SETTINGS="$CLAUDE_CONFIG_DIR/settings.json"
if [ -f "$CLAUDE_SETTINGS" ]; then
    echo ""
    echo "📋 Existing Claude settings found at $CLAUDE_SETTINGS"
    echo "   To add MCP servers, merge the configuration from:"
    echo "   $SCRIPT_DIR/mcp_config.json"
    echo ""
    echo "   Or run: jq -s '.[0] * .[1]' $CLAUDE_SETTINGS $SCRIPT_DIR/mcp_config.json > tmp.json && mv tmp.json $CLAUDE_SETTINGS"
else
    echo ""
    echo "📋 No existing Claude settings found."
    echo "   Creating new settings with MCP servers..."

    cat > "$CLAUDE_SETTINGS" << EOF
{
  "permissions": {
    "defaultMode": "default"
  },
  "mcpServers": {
    "prism-rag": {
      "command": "$VENV_DIR/bin/python",
      "args": ["$SCRIPT_DIR/prism_rag_server.py"]
    },
    "prism-cuda": {
      "command": "$VENV_DIR/bin/python",
      "args": ["$SCRIPT_DIR/cuda_agent_server.py"]
    },
    "prism-bench": {
      "command": "$VENV_DIR/bin/python",
      "args": ["$SCRIPT_DIR/bench_agent_server.py"]
    },
    "prism-orchestrator": {
      "command": "$VENV_DIR/bin/python",
      "args": ["$SCRIPT_DIR/orchestrator_server.py"]
    }
  }
}
EOF
    echo "✅ Claude settings created: $CLAUDE_SETTINGS"
fi

echo ""
echo "=========================================="
echo "🎉 Setup Complete!"
echo "=========================================="
echo ""
echo "Available MCP Servers:"
echo "  • prism-rag         - Semantic code search (LanceDB RAG)"
echo "  • prism-cuda        - CUDA kernel specialist"
echo "  • prism-bench       - Validation & benchmarking"
echo "  • prism-orchestrator - Multi-agent coordination"
echo ""
echo "To use:"
echo "  1. Restart Claude Code to load new MCP servers"
echo "  2. The agents will appear in the 'agents' section"
echo "  3. Use tools like 'search_code', 'analyze_kernel', etc."
echo ""
echo "To re-index codebase after changes:"
echo "  python3 $SCRIPT_DIR/indexer.py"
echo ""
