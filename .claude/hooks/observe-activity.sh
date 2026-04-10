#!/usr/bin/env bash
# observe-activity.sh — Claude Code PostToolUse hook
# Fires after every tool call, POSTs observation to the PRISM-DataOps Worker.
# Non-blocking: curl with 2s timeout, failures silently ignored.
# The Worker batches observations and triggers the PRISM-Observer agent.

set -euo pipefail

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // "unknown"')
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // ""')
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // ""')

WORKER_URL="https://prism-dataops.is-0b9.workers.dev/observe"

# Classify the event type
EVENT_TYPE="UNKNOWN"
PAYLOAD="{}"

case "$TOOL_NAME" in
  Edit|Write)
    EVENT_TYPE="FILE_EDIT"
    PAYLOAD=$(jq -n --arg file "$FILE_PATH" --arg tool "$TOOL_NAME" \
      '{file: $file, tool: $tool}')
    ;;
  Bash)
    # Extract just the command, not full output
    CMD_SHORT=$(echo "$COMMAND" | head -c 500)
    EVENT_TYPE="BASH_COMMAND"
    PAYLOAD=$(jq -n --arg cmd "$CMD_SHORT" '{command: $cmd}')

    # Detect engine runs
    if echo "$COMMAND" | grep -q "nhs_rt_full\|prism-validate-and-run" 2>/dev/null; then
      EVENT_TYPE="ENGINE_RUN"
    fi

    # Detect build checks
    if echo "$COMMAND" | grep -qE "cargo (check|build|test|clippy)|pytest|python3 -m pytest" 2>/dev/null; then
      EVENT_TYPE="BUILD_CHECK"
    fi

    # Detect git commits
    if echo "$COMMAND" | grep -q "git commit" 2>/dev/null; then
      EVENT_TYPE="GIT_COMMIT"
      SHA=$(git -C /home/diddy/Desktop/Prism4D-bio rev-parse --short HEAD 2>/dev/null || echo "unknown")
      MSG=$(git -C /home/diddy/Desktop/Prism4D-bio log -1 --format=%s 2>/dev/null || echo "")
      PAYLOAD=$(jq -n --arg sha "$SHA" --arg msg "$MSG" --arg cmd "$CMD_SHORT" \
        '{sha: $sha, message: $msg, command: $cmd}')
    fi

    # Flag SOP violations
    if echo "$COMMAND" | grep -q "nhs_rt_full" 2>/dev/null; then
      if ! echo "$COMMAND" | grep -q "prism-validate-and-run" 2>/dev/null; then
        EVENT_TYPE="FLAG_ALERT"
        PAYLOAD=$(jq -n --arg cmd "$CMD_SHORT" \
          '{alert: "DIRECT_ENGINE_INVOCATION", severity: "HIGH", evidence: $cmd, rule: "Must use prism-validate-and-run.sh"}')
      fi
    fi
    ;;
  Read|Grep|Glob)
    # Skip read-only operations — too noisy
    exit 0
    ;;
  *)
    # Skip unknown tools
    exit 0
    ;;
esac

# Determine actor from session context
ACTOR="dev"
BRANCH=$(git -C /home/diddy/Desktop/Prism4D-bio branch --show-current 2>/dev/null || echo "unknown")
if echo "$BRANCH" | grep -q "twin\|coupled" 2>/dev/null; then
  ACTOR="twin"
elif echo "$BRANCH" | grep -q "devops\|ops" 2>/dev/null; then
  ACTOR="ops"
fi

# POST to Worker (non-blocking, 2s timeout, silent failure)
curl -s -o /dev/null -m 2 -X POST "$WORKER_URL" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg type "$EVENT_TYPE" --arg actor "$ACTOR" --argjson payload "$PAYLOAD" \
    '{type: $type, actor: $actor, payload: $payload}')" \
  2>/dev/null &

exit 0
