#!/usr/bin/env bash
# sync_and_purge_tmp.sh — archive /tmp/* to R2 at max throughput, then
# purge ONLY what was successfully verified on R2.
#
# Safe-list approach:
#   - skip protected system dirs (.X11-unix, .ICE-unix, .Test-unix,
#     .font-unix, .XIM-unix, systemd-private-*, snap-*, snap.*,
#     tmux-*)
#   - skip files currently open by any running process (Codex RC logs,
#     tmux-ops socket, claude session sockets)
#   - skip non-regular files (sockets, FIFOs, char/block devs)
#   - skip files owned by anything other than diddy

set -euo pipefail

REPO=/home/diddy/Desktop/Prism4D-bio
TAG="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname -s)"
R2_PREFIX="r2:prism-archive/tmp-snapshots/${HOST}/${TAG}"
STAGE="/mnt/storage/tmp-sync-staging-${TAG}"
MANIFEST="${STAGE}/MANIFEST.json"
LOG="/tmp/tmp_sync_purge.log"

# Use /mnt/storage for staging — it's at 34%, plenty of room — instead of
# the already-pressured root.
mkdir -p "$STAGE"

# Block-pattern list (POSIX find -path matchers, OR-joined manually below).
PROTECTED_PATTERNS=(
  "/tmp/.X11-unix"        "/tmp/.X11-unix/*"
  "/tmp/.ICE-unix"        "/tmp/.ICE-unix/*"
  "/tmp/.Test-unix"       "/tmp/.Test-unix/*"
  "/tmp/.font-unix"       "/tmp/.font-unix/*"
  "/tmp/.XIM-unix"        "/tmp/.XIM-unix/*"
  "/tmp/systemd-private-*" "/tmp/systemd-private-*/*"
  "/tmp/snap.*"           "/tmp/snap.*/*"
  "/tmp/snap-private-*"   "/tmp/snap-private-*/*"
  "/tmp/tmux-*"           "/tmp/tmux-*/*"
  "/tmp/.snap.lock"
  "/tmp/.font-unix/*"
  "/tmp/codex_rc_*.log"   # Codex remote-control daemon writes these
  "/tmp/codex-bwrap-*"
  "/tmp/codex_userns.conf"
  "/tmp/gflownet_inference_chain.log"  # may still be tail'd; keep
  "/tmp/chain.out"
  "/tmp/claude-1000"      "/tmp/claude-1000/*"
)

# Build a find expression that excludes the protected patterns.
EXCLUDE_ARGS=()
for p in "${PROTECTED_PATTERNS[@]}"; do
  EXCLUDE_ARGS+=( -not -path "$p" )
done

echo "=== /tmp sync+purge: tag=${TAG} ===" | tee "$LOG"
echo "  R2 target:   ${R2_PREFIX}/" | tee -a "$LOG"
echo "  staging dir: ${STAGE}" | tee -a "$LOG"

# 1. Inventory candidates (regular files, owned by diddy, NOT in protected list).
echo "  building candidate list…" | tee -a "$LOG"
mapfile -t CANDIDATES < <(
  find /tmp -maxdepth 3 -type f -user diddy "${EXCLUDE_ARGS[@]}" 2>/dev/null \
    | grep -vE "/(\.X11-unix|\.ICE-unix|\.Test-unix|\.font-unix|\.XIM-unix|systemd-private|snap\.|snap-private|tmux-|claude-1000|codex-bwrap)" \
    | sort
)
echo "  candidate count: ${#CANDIDATES[@]}" | tee -a "$LOG"
if [ "${#CANDIDATES[@]}" -eq 0 ]; then
  echo "  nothing to sync. exiting." | tee -a "$LOG"
  exit 0
fi

# 2. Skip any candidate that is currently open by any process.
OPEN_FILES_TMP=$(mktemp)
# /proc walk — safer than lsof for this scope and doesn't need extra perms.
for pid_dir in /proc/[0-9]*; do
  ls -l "${pid_dir}/fd/" 2>/dev/null \
    | awk '$NF ~ /^\/tmp\// {print $NF}'
done | sort -u > "$OPEN_FILES_TMP"
echo "  files currently open by some process: $(wc -l < $OPEN_FILES_TMP)" | tee -a "$LOG"

# 3. Filter candidates by open-file list.
SYNC_LIST=$(mktemp)
SKIPPED_LIST=$(mktemp)
for f in "${CANDIDATES[@]}"; do
  if grep -Fxq "$f" "$OPEN_FILES_TMP"; then
    echo "$f" >> "$SKIPPED_LIST"
  else
    echo "$f" >> "$SYNC_LIST"
  fi
done
N_SYNC=$(wc -l < "$SYNC_LIST")
N_SKIP=$(wc -l < "$SKIPPED_LIST")
echo "  to sync: ${N_SYNC}    skipping (in-use): ${N_SKIP}" | tee -a "$LOG"

# 4. Stage symlinks into a single tree mirroring /tmp.
mkdir -p "$STAGE/payload"
while IFS= read -r f; do
  rel="${f#/tmp/}"
  dest="$STAGE/payload/$rel"
  mkdir -p "$(dirname "$dest")"
  cp -a "$f" "$dest" 2>/dev/null || true
done < "$SYNC_LIST"

# 5. Write a manifest (path, size, sha256, mtime).
python3 - <<PY > "$MANIFEST"
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
sync_list = open("$SYNC_LIST").read().splitlines()
skip_list = open("$SKIPPED_LIST").read().splitlines() if os.path.exists("$SKIPPED_LIST") else []
entries = []
for orig in sync_list:
    if not os.path.isfile(orig):
        continue
    h = hashlib.sha256()
    with open(orig, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = os.stat(orig)
    entries.append({
        "original_path": orig,
        "r2_key":        f"tmp-snapshots/$HOST/$TAG/" + orig[len("/tmp/"):],
        "size_bytes":    st.st_size,
        "sha256":        h.hexdigest(),
        "mtime_utc":     datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
out = {
    "package":      "PRISM4D_TMP_SNAPSHOT",
    "host":         "$HOST",
    "snapshot_utc": "$TAG",
    "trigger":      "operator-directed disk-pressure relief (root 97%)",
    "purge_policy": "purge after successful R2 verify; skip files in-use",
    "skipped_in_use": skip_list,
    "entries":      entries,
}
print(json.dumps(out, indent=2))
PY
TOTAL_BYTES=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(sum(e['size_bytes'] for e in m['entries']))")
echo "  manifest written: ${MANIFEST}" | tee -a "$LOG"
echo "  total bytes to ship: ${TOTAL_BYTES}" | tee -a "$LOG"

# Also stage the manifest under the payload tree.
cp "$MANIFEST" "$STAGE/payload/_MANIFEST.json"

# 6. Upload to R2 at MAX parallelism.
echo "  uploading to ${R2_PREFIX}/…" | tee -a "$LOG"
SECONDS=0
rclone copy \
  "$STAGE/payload/" "$R2_PREFIX/" \
  --transfers 32 --checkers 32 --multi-thread-streams 8 \
  --header-upload "x-amz-meta-host:$HOST" \
  --header-upload "x-amz-meta-snapshot:$TAG" \
  --header-upload "x-amz-meta-purpose:tmp-disk-pressure-relief" \
  --header-upload "x-amz-meta-source:/tmp" \
  --stats=5s --stats-one-line \
  2>&1 | tee -a "$LOG" | grep -E "Transferred|ETA" | tail -5
echo "  upload elapsed: ${SECONDS}s" | tee -a "$LOG"

# 7. Verify each entry exists on R2 (cheap object stat).
echo "  verifying R2 objects…" | tee -a "$LOG"
VERIFY_OK=0
VERIFY_BAD=0
BAD_KEYS=()
while IFS= read -r key; do
  if rclone size "${R2_PREFIX}/${key}" --json 2>/dev/null | grep -q "\"count\":1"; then
    VERIFY_OK=$((VERIFY_OK+1))
  else
    VERIFY_BAD=$((VERIFY_BAD+1))
    BAD_KEYS+=( "$key" )
  fi
done < <(python3 -c "
import json
for e in json.load(open('$MANIFEST'))['entries']:
    print(e['original_path'][len('/tmp/'):])
")
echo "  verified-on-R2: ${VERIFY_OK}  failed: ${VERIFY_BAD}" | tee -a "$LOG"

# 8. Purge ONLY entries that verified.
if [ "$VERIFY_BAD" -gt 0 ]; then
  echo "  ABORTING PURGE — ${VERIFY_BAD} files did not verify on R2." | tee -a "$LOG"
  for k in "${BAD_KEYS[@]}"; do echo "    BAD: $k" | tee -a "$LOG"; done
  echo "  re-run after fixing R2 issues. staging kept at: ${STAGE}" | tee -a "$LOG"
  exit 1
fi

echo "  purging verified files from /tmp…" | tee -a "$LOG"
PURGED=0
while IFS= read -r f; do
  if [ -f "$f" ] && rm -f "$f"; then
    PURGED=$((PURGED+1))
  fi
done < "$SYNC_LIST"
echo "  purged: ${PURGED} files" | tee -a "$LOG"

# 9. Drop empty dirs in /tmp owned by diddy (skip protected).
find /tmp -maxdepth 3 -type d -user diddy -empty \
  -not -path "/tmp" \
  "${EXCLUDE_ARGS[@]}" \
  -delete 2>/dev/null || true

# 10. Cleanup staging.
rm -rf "$STAGE"
echo "=== done. log: $LOG ===" | tee -a "$LOG"

# Final state
df -h / | tail -1 | tee -a "$LOG"
echo "  R2 prefix: $R2_PREFIX" | tee -a "$LOG"
