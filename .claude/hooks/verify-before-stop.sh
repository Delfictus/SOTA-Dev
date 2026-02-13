#!/usr/bin/env bash
# verify-before-stop.sh
# Stop hook: Checks if Claude ran verification commands before claiming completion.
# Exit 0 = allow stop. Exit 2 = block stop (forces Claude to continue).

set -euo pipefail

INPUT=$(cat)

# Check if this is already a re-run from a previous stop hook block
STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path')

if [ ! -f "$TRANSCRIPT_PATH" ]; then
  # Can't verify without transcript, allow stop
  exit 0
fi

# Read the last ~200 lines of transcript (recent context)
RECENT_TRANSCRIPT=$(tail -200 "$TRANSCRIPT_PATH" 2>/dev/null || echo "")

# --- Check 1: Were any Rust files edited? ---
EDITED_RUST=false
if echo "$RECENT_TRANSCRIPT" | grep -qE '"tool_name"\s*:\s*"(Edit|Write|MultiEdit)"' 2>/dev/null; then
  if echo "$RECENT_TRANSCRIPT" | grep -qE '\.(rs|toml)"' 2>/dev/null; then
    EDITED_RUST=true
  fi
fi

# --- Check 2: Was cargo check actually run after edits? ---
CARGO_CHECK_RAN=false
if echo "$RECENT_TRANSCRIPT" | grep -qE 'cargo\s+(check|build|clippy)' 2>/dev/null; then
  CARGO_CHECK_RAN=true
fi

# --- Check 3: Was cargo test actually run? ---
CARGO_TEST_RAN=false
if echo "$RECENT_TRANSCRIPT" | grep -qE 'cargo\s+test' 2>/dev/null; then
  CARGO_TEST_RAN=true
fi

# --- Decision logic ---

# If this is a re-check (stop_hook_active=true), be more lenient
# but still verify the critical path
if [ "$STOP_HOOK_ACTIVE" = "true" ]; then
  # On re-check, only block if Rust was edited and STILL no cargo check
  if [ "$EDITED_RUST" = "true" ] && [ "$CARGO_CHECK_RAN" = "false" ]; then
    echo "STOP BLOCKED (re-check): Rust files were edited but cargo check/build was never run. Run 'cargo check' and show the actual output." >&2
    exit 2
  fi
  exit 0
fi

# First-time stop: enforce full verification
if [ "$EDITED_RUST" = "true" ]; then
  if [ "$CARGO_CHECK_RAN" = "false" ]; then
    echo "STOP BLOCKED: You edited .rs/.toml files but never ran cargo check or cargo build. Run 'cargo check' now and show the actual output before claiming anything works." >&2
    exit 2
  fi
fi

# If Claude's last assistant message contains "implemented", "complete", "works",
# "compiles", "passes" — check that corresponding verification happened
CLAIMS_PATTERN='(implemented|complete[d]?|works|compiles|passes|success|fixed|resolved)'
if echo "$RECENT_TRANSCRIPT" | grep -qiE "$CLAIMS_PATTERN" 2>/dev/null; then
  if [ "$CARGO_CHECK_RAN" = "false" ] && [ "$EDITED_RUST" = "true" ]; then
    echo "STOP BLOCKED: You made completion claims about Rust code but never ran cargo check. Verify before claiming." >&2
    exit 2
  fi
fi

# All checks passed
exit 0
