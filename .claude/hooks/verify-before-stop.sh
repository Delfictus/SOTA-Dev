#!/usr/bin/env bash
set -euo pipefail
INPUT=$(cat)

STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path')

if [ ! -f "$TRANSCRIPT_PATH" ]; then
  exit 0
fi

RECENT=$(tail -300 "$TRANSCRIPT_PATH" 2>/dev/null || echo "")

EDITED_RUST=false
EDITED_PYTHON=false
EDITED_WEB=false

if echo "$RECENT" | grep -qE '\.(rs|toml)"' 2>/dev/null; then EDITED_RUST=true; fi
if echo "$RECENT" | grep -qE '\.(py)"' 2>/dev/null; then EDITED_PYTHON=true; fi
if echo "$RECENT" | grep -qE '\.(html|js|jsx|css)"' 2>/dev/null; then EDITED_WEB=true; fi

CARGO_RAN=false
PYTEST_RAN=false

if echo "$RECENT" | grep -qE 'cargo\s+(check|build|test|clippy)' 2>/dev/null; then CARGO_RAN=true; fi
if echo "$RECENT" | grep -qE 'pytest|python3?\s+-m\s+pytest' 2>/dev/null; then PYTEST_RAN=true; fi

CLAIMS=false
if echo "$RECENT" | grep -qiE '(implemented|complete[d]?|works|compiles|passes|success|fixed|resolved|done|pushed|deployed)' 2>/dev/null; then
  CLAIMS=true
fi

if [ "$STOP_HOOK_ACTIVE" = "true" ]; then
  if [ "$EDITED_RUST" = "true" ] && [ "$CARGO_RAN" = "false" ]; then
    echo "STOP BLOCKED (re-check): Rust edited, no cargo check." >&2; exit 2
  fi
  if [ "$EDITED_PYTHON" = "true" ] && [ "$PYTEST_RAN" = "false" ]; then
    echo "STOP BLOCKED (re-check): Python edited, no pytest." >&2; exit 2
  fi
  exit 0
fi

if [ "$EDITED_RUST" = "true" ] && [ "$CARGO_RAN" = "false" ]; then
  echo "STOP BLOCKED: Edited .rs/.toml but never ran cargo check/build/test." >&2; exit 2
fi

if [ "$EDITED_PYTHON" = "true" ] && [ "$PYTEST_RAN" = "false" ]; then
  echo "STOP BLOCKED: Edited .py files but never ran pytest. Run 'python3 -m pytest' now." >&2; exit 2
fi

if [ "$EDITED_WEB" = "true" ] && [ "$CLAIMS" = "true" ]; then
  echo "STOP BLOCKED: Edited HTML/JS and claimed it works without verification." >&2; exit 2
fi

exit 0
