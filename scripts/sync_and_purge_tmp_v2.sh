#!/usr/bin/env bash
# v2 — top-level granularity. For each /tmp top-level entry:
#   - skip if in protected list
#   - skip if in lsof-derived in-use set
#   - if file: copy as-is to R2, then rm
#   - if dir:  tar+zstd to /mnt/storage staging, upload one tarball,
#              then rm -rf the dir
# Throughput: 32 parallel rclone transfers, zstd-19 for max compression of
# regeneratable dirs (mypy cache compresses heavily).

set -euo pipefail

TAG="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname -s)"
R2_PREFIX="r2:prism-archive/tmp-snapshots/${HOST}/${TAG}"
STAGE="/mnt/storage/tmp-sync-staging-${TAG}"
LOG="/mnt/storage/tmp_sync_purge_${TAG}.log"
MANIFEST="${STAGE}/MANIFEST.json"
mkdir -p "$STAGE"

# Top-level entries we MUST NOT touch (active sessions, sockets, system).
declare -A PROTECTED=(
  [".X11-unix"]=1 [".ICE-unix"]=1 [".Test-unix"]=1 [".font-unix"]=1 [".XIM-unix"]=1
  ["claude-1000"]=1
  [".snap.lock"]=1
  ["codex_rc_default.log"]=1 ["codex_rc_full.log"]=1 ["codex_userns.conf"]=1
  ["codex-bwrap-synthetic-mount-targets-1000"]=1
  ["tmp_sync_purge.log"]=1     # ours, may be tail'd
  ["gflownet_inference_chain.log"]=1 ["chain.out"]=1   # may still be tail'd
  ["tmux-ops"]=1   # in case anything lands at this name; the real path is on a private socket
)

is_protected () {
  local n="$1"
  [ "${PROTECTED[$n]:-}" = "1" ] && return 0
  case "$n" in
    systemd-private-*|snap.*|snap-private-*|tmux-*|claude-*)
      return 0 ;;
  esac
  return 1
}

# Build the set of /tmp paths currently held open by any running process
# — using /proc walk (no sudo needed for our own pids; system service
# pids may be unreadable and that's fine — those are protected by name
# anyway).
OPEN_PATHS=$(mktemp)
set +e
{
  for pid_dir in /proc/[0-9]*; do
    for fd in "$pid_dir"/fd/*; do
      target=$(readlink "$fd" 2>/dev/null || true)
      case "$target" in
        /tmp/*) echo "$target" ;;
      esac
    done
  done
} 2>/dev/null | sort -u > "$OPEN_PATHS"
set -e

is_in_use () {
  local p="$1"
  if [ -d "$p" ]; then
    grep -q "^$p/" "$OPEN_PATHS" && return 0
  else
    grep -Fxq "$p" "$OPEN_PATHS" && return 0
  fi
  return 1
}

echo "=== /tmp sync+purge v2 ===" | tee "$LOG"
echo "  TAG=$TAG"  | tee -a "$LOG"
echo "  STAGE=$STAGE"  | tee -a "$LOG"
echo "  R2=$R2_PREFIX/"  | tee -a "$LOG"
echo "  open-paths: $(wc -l < $OPEN_PATHS)"  | tee -a "$LOG"

# Emit a manifest as we go.
echo "[" > "$MANIFEST"
first=1

UPLOAD_FILES=()
N_KEPT=0
N_PURGED=0
TOTAL_BYTES_PURGED=0

for entry in /tmp/* /tmp/.[!.]*; do
  [ -e "$entry" ] || continue
  name="$(basename "$entry")"
  # Skip ourselves
  [ "$name" = "$(basename "$STAGE")" ] && continue

  if is_protected "$name"; then
    echo "  [skip-protected] $name" | tee -a "$LOG"
    continue
  fi
  if is_in_use "$entry"; then
    echo "  [skip-in-use]    $name" | tee -a "$LOG"
    continue
  fi
  # Only deal with diddy-owned, regular file or dir
  owner=$(stat -c %U "$entry" 2>/dev/null || echo "")
  if [ "$owner" != "diddy" ]; then
    echo "  [skip-owner]     $name (owner=$owner)" | tee -a "$LOG"
    continue
  fi

  if [ -d "$entry" ]; then
    # Tar+zstd the directory into one archive in staging.
    arc="$STAGE/${name}.tar.zst"
    sz_before=$(du -sb "$entry" 2>/dev/null | awk '{print $1}')
    tar --use-compress-program='zstd -T0 -19' -cf "$arc" -C /tmp "$name" 2>/dev/null
    sz_arc=$(stat -c %s "$arc" 2>/dev/null)
    UPLOAD_FILES+=( "$arc" )
    echo "  [archive] $name  ${sz_before}B -> ${sz_arc}B  (${arc})" | tee -a "$LOG"
    [ "$first" -eq 0 ] && echo "  ," >> "$MANIFEST"
    cat >> "$MANIFEST" <<JSON
  {"name":"$name","type":"dir","tar":"${name}.tar.zst","size_before":$sz_before,"size_after":$sz_arc}
JSON
    first=0
  elif [ -f "$entry" ]; then
    cp -a "$entry" "$STAGE/$name"
    UPLOAD_FILES+=( "$STAGE/$name" )
    sz=$(stat -c %s "$entry" 2>/dev/null)
    echo "  [file]    $name  ${sz}B" | tee -a "$LOG"
    [ "$first" -eq 0 ] && echo "  ," >> "$MANIFEST"
    cat >> "$MANIFEST" <<JSON
  {"name":"$name","type":"file","size":$sz}
JSON
    first=0
  fi
done
echo "]" >> "$MANIFEST"
echo "  upload set: ${#UPLOAD_FILES[@]} archives/files" | tee -a "$LOG"
echo "  staging size: $(du -sh $STAGE | cut -f1)" | tee -a "$LOG"

# Upload everything in staging to R2 with max parallelism.
SECONDS=0
rclone copy "$STAGE/" "$R2_PREFIX/" \
  --transfers 32 --checkers 32 --multi-thread-streams 8 \
  --header-upload "x-amz-meta-host:$HOST" \
  --header-upload "x-amz-meta-snapshot:$TAG" \
  --header-upload "x-amz-meta-purpose:tmp-disk-pressure-relief" \
  --stats=2s --stats-one-line 2>&1 | tee -a "$LOG" | tail -3
echo "  upload elapsed: ${SECONDS}s" | tee -a "$LOG"

# Verify every uploaded basename exists on R2.
echo "  verifying R2…" | tee -a "$LOG"
R2_LISTING=$(mktemp)
rclone lsf "$R2_PREFIX/" > "$R2_LISTING" 2>/dev/null
N_VERIFY_OK=0
N_VERIFY_BAD=0
BAD=()
for f in "${UPLOAD_FILES[@]}"; do
  bn=$(basename "$f")
  if grep -Fxq "$bn" "$R2_LISTING"; then
    N_VERIFY_OK=$((N_VERIFY_OK+1))
  else
    N_VERIFY_BAD=$((N_VERIFY_BAD+1))
    BAD+=( "$bn" )
  fi
done
echo "  verified: $N_VERIFY_OK / failed: $N_VERIFY_BAD" | tee -a "$LOG"
if [ "$N_VERIFY_BAD" -gt 0 ]; then
  echo "  ABORTING PURGE — some uploads did not verify:" | tee -a "$LOG"
  printf "    BAD: %s\n" "${BAD[@]}" | tee -a "$LOG"
  exit 1
fi

# Purge the originals (only ones we successfully archived/uploaded).
for entry in /tmp/* /tmp/.[!.]*; do
  [ -e "$entry" ] || continue
  name="$(basename "$entry")"
  if is_protected "$name"; then continue; fi
  if is_in_use "$entry"; then continue; fi
  owner=$(stat -c %U "$entry" 2>/dev/null || echo "")
  [ "$owner" = "diddy" ] || continue
  # Only delete if we successfully uploaded the corresponding archive/file
  if [ -d "$entry" ]; then
    if grep -Fxq "${name}.tar.zst" "$R2_LISTING"; then
      sz=$(du -sb "$entry" 2>/dev/null | awk '{print $1}')
      rm -rf "$entry" && {
        TOTAL_BYTES_PURGED=$((TOTAL_BYTES_PURGED + sz))
        N_PURGED=$((N_PURGED+1))
        echo "  [purged-dir]  $name (${sz}B)" | tee -a "$LOG"
      }
    fi
  elif [ -f "$entry" ]; then
    if grep -Fxq "$name" "$R2_LISTING"; then
      sz=$(stat -c %s "$entry" 2>/dev/null)
      rm -f "$entry" && {
        TOTAL_BYTES_PURGED=$((TOTAL_BYTES_PURGED + sz))
        N_PURGED=$((N_PURGED+1))
        echo "  [purged-file] $name (${sz}B)" | tee -a "$LOG"
      }
    fi
  fi
done

# Cleanup staging.
rm -rf "$STAGE"

echo "=== summary ===" | tee -a "$LOG"
echo "  purged entries: $N_PURGED" | tee -a "$LOG"
echo "  bytes freed in /tmp: $TOTAL_BYTES_PURGED" | tee -a "$LOG"
echo "  R2 prefix: $R2_PREFIX/" | tee -a "$LOG"
echo "  log: $LOG" | tee -a "$LOG"
df -h / | tail -1 | tee -a "$LOG"
du -sh /tmp 2>/dev/null | tee -a "$LOG"
