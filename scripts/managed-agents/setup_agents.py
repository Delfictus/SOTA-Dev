#!/usr/bin/env python3
"""
PRISM4D Managed Agents — Setup & Orchestration
================================================

Creates and manages cloud-hosted Claude agents for the PRISM data pipeline:

  1. PRISM-DataOps:      Post-upload validation, R2 inventory, training manifest curation
  2. PRISM-CampaignRunner: Long-running batch orchestration with resume logic

Architecture:
  Local daemon (prism_spike_watcher.py) handles the hot path:
    inotify → detect spike files → upload JSON + Parquet → verify on R2 → delete local

  Managed agents handle the intelligence layer:
    R2 bucket monitoring → data validation → contamination detection →
    training manifest curation → campaign status reporting

Prerequisites:
  export ANTHROPIC_API_KEY="sk-ant-..."
  pip install anthropic>=0.79.0

Usage:
  # Create all agents and environments (one-time setup)
  python3 scripts/managed-agents/setup_agents.py setup

  # Start a DataOps validation session
  python3 scripts/managed-agents/setup_agents.py validate

  # Start a campaign monitoring session
  python3 scripts/managed-agents/setup_agents.py monitor

  # List all agents and sessions
  python3 scripts/managed-agents/setup_agents.py status
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: pip install anthropic")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENT_CONFIG_DIR = Path(__file__).parent / "agent_configs"
STATE_FILE = Path(__file__).parent / ".agent_state.json"

# R2 bucket structure (must match prism_spike_watcher.py routing)
R2_BUCKETS = {
    "prism-archive": "Main data archive (spike JSONs, Parquets, binding_sites)",
    "prism-production": "Validated production data for ML training",
    "prism-public": "Public demo platform data",
}

R2_PREFIXES = [
    "cryptobench199",
    "v1.1-physics",
    "twin-runs",
    "10k-runs",
    "blind_validation_100",
    "runs",
]

# Contaminated targets — never include in training data
CONTAMINATED_TARGETS = {
    # CryptoBench contamination (teacher model saw 143/199 targets)
    "cryptobench_teacher_contaminated": True,
    # SNDC overlap must never appear in training
    "sndc_targets": [
        "1w50", "1btl", "4obe", "1g1f", "1ade", "3k5v",
        "1a4q", "2wng", "1ere", "1hhp", "1bj4", "2gl7",
    ],
}


# ---------------------------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------------------------

DATAOPS_AGENT = {
    "name": "PRISM-DataOps",
    "model": "claude-sonnet-4-6",
    "system": """You are PRISM-DataOps, the data operations agent for the PRISM4D protein binding site detection pipeline.

YOUR RESPONSIBILITIES:
1. Validate spike data uploaded to Cloudflare R2 (prism-archive bucket)
2. Maintain the canonical training manifest for PrismAI student model training
3. Detect and flag contaminated data (SNDC overlap, CryptoBench teacher leakage)
4. Generate R2 inventory reports and data quality summaries

CRITICAL RULES:
- NEVER include SNDC targets in training manifests: 1w50, 1btl, 4obe, 1g1f, 1ade, 3k5v, 1a4q, 2wng, 1ere, 1hhp, 1bj4, 2gl7
- CryptoBench data is contaminated (teacher saw 143/199 targets). Use ONLY for evaluation, never training.
- Every spike Parquet must have matching row count to source JSON (lossless verification)
- The 216-feature spec is mandatory for all training data
- Topology residue IDs ≠ PDB residue IDs. Always note the offset.

DATA LIFECYCLE:
  spike_events.json (engine output)
  → spike_events.parquet (zstd lossless)
  → BOTH uploaded to r2:prism-archive/{category}/{target}/
  → Local JSON deleted only after R2 verification
  → Parquet kept as working copy

R2 BUCKET STRUCTURE:
  prism-archive/cryptobench199/{target}/  — CryptoBench 199 runs
  prism-archive/v1.1-physics/{target}/    — v1.1 physics validation
  prism-archive/twin-runs/{target}/       — TWIN multi-stream runs
  prism-archive/10k-runs/{target}/        — 10K proteome campaign
  prism-archive/runs/{target}/            — general runs

TOOLS AVAILABLE:
  - bash: run rclone commands to query/manage R2
  - file ops: read/write validation reports
  - web_fetch: access external APIs if needed

When validating, always report:
  - Total files on R2 (JSON count, Parquet count)
  - Missing Parquets (JSONs without matching Parquet)
  - Contaminated targets found
  - Data quality issues (empty files, truncated, wrong schema)
""",
    "tools": [
        {"type": "agent_toolset_20260401"},
        {
            "type": "custom",
            "name": "notify_local_daemon",
            "description": "Send a notification to the local PRISM machine's spike watcher daemon. Use this to trigger retroactive scans or report issues that need local attention.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["retroactive_scan", "pause", "resume", "status"],
                        "description": "Action for the local daemon",
                    },
                    "message": {
                        "type": "string",
                        "description": "Human-readable message about what triggered this notification",
                    },
                },
                "required": ["action", "message"],
            },
        },
        {
            "type": "custom",
            "name": "update_training_manifest",
            "description": "Update the canonical training manifest with validated spike data entries. The manifest tracks which R2 objects are approved for PrismAI model training.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "r2_path": {"type": "string"},
                                "target_pdb": {"type": "string"},
                                "category": {"type": "string"},
                                "row_count": {"type": "integer"},
                                "file_size_bytes": {"type": "integer"},
                                "validation_status": {
                                    "type": "string",
                                    "enum": ["approved", "rejected", "needs_review"],
                                },
                                "rejection_reason": {"type": "string"},
                            },
                            "required": ["r2_path", "target_pdb", "validation_status"],
                        },
                    },
                },
                "required": ["entries"],
            },
        },
    ],
}

CAMPAIGN_AGENT = {
    "name": "PRISM-CampaignRunner",
    "model": "claude-sonnet-4-6",
    "system": """You are PRISM-CampaignRunner, the batch orchestration agent for the PRISM4D pipeline.

YOUR RESPONSIBILITIES:
1. Monitor ongoing engine campaigns (CryptoBench, 10K proteome, blind validation)
2. Track completion status: expected vs actual targets
3. Detect stalled or failed runs and report them
4. Generate campaign progress reports
5. Manage R2 data organization for completed campaigns

CAMPAIGN STRUCTURE:
  Each campaign has a manifest listing all target PDBs.
  Engine runs produce: binding_sites.json, kcc_visualization.json, spike_events.json/parquet, trajectory data
  A target is "complete" when binding_sites.json exists AND spike files are on R2.

KNOWN CAMPAIGNS:
  - cryptobench199: 199 targets, CryptoBench benchmark
  - blind_validation_100: 100 blind validation targets
  - 10k_campaign: 10,000 proteome-scale targets
  - v1.1-physics: 33 original physics validation targets

TOOLS AVAILABLE:
  - bash: run rclone to query R2 bucket contents
  - file ops: write reports and manifests

When monitoring, report:
  - Targets complete / total
  - Targets with missing spike data
  - Targets with failed engine runs (no binding_sites.json)
  - Estimated data volume on R2
  - Any anomalies (unusually small files, missing expected outputs)
""",
    "tools": [
        {"type": "agent_toolset_20260401"},
    ],
}


# ---------------------------------------------------------------------------
# Environment Definition
# ---------------------------------------------------------------------------

PRISM_ENVIRONMENT = {
    "name": "prism-dataops-env",
    "config": {
        "type": "cloud",
        "packages": {
            "pip": ["pyarrow", "pandas", "boto3"],
            "apt": ["rclone", "jq", "curl"],
        },
        "networking": {"type": "unrestricted"},
    },
}


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load persisted agent/environment IDs."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    """Persist agent/environment IDs for reuse."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup(client: Anthropic):
    """Create agents and environment (idempotent — skips if already created)."""
    state = load_state()

    # --- Environment ---
    if "environment_id" not in state:
        print("Creating environment: prism-dataops-env")
        env = client.beta.environments.create(**PRISM_ENVIRONMENT)
        state["environment_id"] = env.id
        print(f"  Environment ID: {env.id}")
    else:
        print(f"Environment already exists: {state['environment_id']}")

    # --- DataOps Agent ---
    if "dataops_agent_id" not in state:
        print("Creating agent: PRISM-DataOps")
        agent = client.beta.agents.create(**DATAOPS_AGENT)
        state["dataops_agent_id"] = agent.id
        state["dataops_agent_version"] = agent.version
        print(f"  Agent ID: {agent.id}, version: {agent.version}")
    else:
        print(f"DataOps agent already exists: {state['dataops_agent_id']}")

    # --- CampaignRunner Agent ---
    if "campaign_agent_id" not in state:
        print("Creating agent: PRISM-CampaignRunner")
        agent = client.beta.agents.create(**CAMPAIGN_AGENT)
        state["campaign_agent_id"] = agent.id
        state["campaign_agent_version"] = agent.version
        print(f"  Agent ID: {agent.id}, version: {agent.version}")
    else:
        print(f"CampaignRunner agent already exists: {state['campaign_agent_id']}")

    save_state(state)
    print("\nSetup complete. State saved to:", STATE_FILE)
    return state


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------

def start_session(client: Anthropic, agent_key: str, title: str, message: str):
    """Start a managed agent session and stream results."""
    state = load_state()

    agent_id = state.get(f"{agent_key}_agent_id")
    env_id = state.get("environment_id")

    if not agent_id or not env_id:
        print("ERROR: Run 'setup' first to create agents and environment.")
        sys.exit(1)

    print(f"Starting session: {title}")
    print(f"  Agent: {agent_id}")
    print(f"  Environment: {env_id}")

    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=env_id,
        title=title,
    )
    print(f"  Session ID: {session.id}")

    # Open stream and send message
    with client.beta.sessions.events.stream(session.id) as stream:
        client.beta.sessions.events.send(
            session.id,
            events=[
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": message}],
                },
            ],
        )

        # Process streaming events
        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    print(block.text, end="")
            elif event.type == "agent.tool_use":
                print(f"\n[Tool: {event.name}]")
            elif event.type == "agent.tool_result":
                pass  # Tool results are handled internally
            elif event.type == "session.status_idle":
                print("\n\nSession complete.")
                break

    return session.id


def validate_r2(client: Anthropic):
    """Start a DataOps validation session."""
    # Build the rclone config that the agent will need
    rclone_setup = """First, configure rclone for Cloudflare R2. Run these bash commands:

mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << 'RCLONE_EOF'
[r2]
type = s3
provider = Cloudflare
env_auth = false
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
acl = private
no_check_bucket = true
RCLONE_EOF

Then validate the R2 data:

1. List all objects in r2:prism-archive/ — count JSONs vs Parquets per category
2. Check for orphaned JSONs (no matching Parquet) — these need conversion
3. Check for any SNDC target data that should NOT be in training paths
4. Check for empty or suspiciously small files (<1KB)
5. Generate a summary report with:
   - Total files by category (cryptobench199, v1.1-physics, twin-runs, etc.)
   - Total data volume
   - Missing Parquets count
   - Contamination warnings
   - Recommended actions
"""

    return start_session(
        client,
        agent_key="dataops",
        title="R2 Data Validation Sweep",
        message=rclone_setup,
    )


def monitor_campaigns(client: Anthropic):
    """Start a campaign monitoring session."""
    message = """Check the status of all PRISM campaigns on R2.

Configure rclone first (see system prompt for R2 bucket structure), then:

1. For each campaign prefix (cryptobench199, v1.1-physics, blind_validation_100, 10k-runs):
   a. List all target directories
   b. Check each for binding_sites.json (engine completed)
   c. Check each for spike_events.parquet (data archived)
   d. Flag any with JSON but no Parquet (conversion pending)

2. Generate a progress report:
   - Targets complete / expected
   - Data volume per campaign
   - Stalled targets (binding_sites exists but no spikes)
   - Failed targets (empty directories)

3. Write the report as campaign_status_report.json
"""
    return start_session(
        client,
        agent_key="campaign",
        title="Campaign Status Check",
        message=message,
    )


def show_status(client: Anthropic):
    """List all agents, environments, and active sessions."""
    state = load_state()
    if not state:
        print("No agents configured. Run 'setup' first.")
        return

    print("=== PRISM Managed Agents Status ===\n")
    print(f"Environment: {state.get('environment_id', 'NOT CREATED')}")
    print(f"DataOps Agent: {state.get('dataops_agent_id', 'NOT CREATED')}")
    print(f"CampaignRunner Agent: {state.get('campaign_agent_id', 'NOT CREATED')}")

    # List active sessions
    try:
        sessions = client.beta.sessions.list()
        active = [s for s in sessions.data if s.status == "active"]
        print(f"\nActive sessions: {len(active)}")
        for s in active:
            print(f"  {s.id} — {s.title} ({s.status})")
    except Exception as e:
        print(f"\nCould not list sessions: {e}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PRISM4D Managed Agents — Setup & Orchestration"
    )
    parser.add_argument(
        "command",
        choices=["setup", "validate", "monitor", "status"],
        help="Command to run",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    if args.command == "setup":
        setup(client)
    elif args.command == "validate":
        validate_r2(client)
    elif args.command == "monitor":
        monitor_campaigns(client)
    elif args.command == "status":
        show_status(client)


if __name__ == "__main__":
    main()
