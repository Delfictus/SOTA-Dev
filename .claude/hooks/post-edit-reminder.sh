#!/usr/bin/env bash
# post-edit-reminder.sh
# PostToolUse hook for Edit/Write: Injects a reminder into Claude's context
# after any file edit. Exit 0 with stdout = message shown in verbose mode.
# The real enforcement is the Stop hook — this is a nudge.

set -euo pipefail

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // "unknown"')
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // "unknown"')

# Only care about Rust/TOML files
if echo "$FILE_PATH" | grep -qE '\.(rs|toml)$' 2>/dev/null; then
  echo "[VERIFY] $FILE_PATH was modified. Run 'cargo check' before claiming this works."
fi

exit 0
