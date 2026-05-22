#!/usr/bin/env bash
set -euo pipefail

BUCKET="${PRISM_DEEP_ARCHIVE_BUCKET:-prism-deep-archive-20260516}"
STATE_DIR="${PRISM_DEEP_ARCHIVE_STATE:-$HOME/.local/state/prism-deep-archive}"
RUN_ID_FILE="$STATE_DIR/run_id"
HOSTNAME_SAFE="$(hostname | tr -c 'A-Za-z0-9._-' '-')"

mkdir -p "$STATE_DIR/logs" "$STATE_DIR/checks"
if [[ -n "${PRISM_DEEP_ARCHIVE_RUN_ID:-}" ]]; then
  RUN_ID="$PRISM_DEEP_ARCHIVE_RUN_ID"
elif [[ -f "$RUN_ID_FILE" ]]; then
  RUN_ID="$(<"$RUN_ID_FILE")"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"
fi

PREFIX="${PRISM_DEEP_ARCHIVE_PREFIX:-workstation-prune/$HOSTNAME_SAFE/$RUN_ID}"
LOG="$STATE_DIR/logs/deep_archive_prune_${RUN_ID}.log"
MANIFEST="$STATE_DIR/deep_archive_manifest_${RUN_ID}.tsv"

RCLONE_COPY_FLAGS=(
  --transfers "${PRISM_DEEP_ARCHIVE_TRANSFERS:-64}"
  --checkers "${PRISM_DEEP_ARCHIVE_CHECKERS:-64}"
  --s3-chunk-size "${PRISM_DEEP_ARCHIVE_CHUNK_SIZE:-128M}"
  --s3-upload-concurrency "${PRISM_DEEP_ARCHIVE_UPLOAD_CONCURRENCY:-8}"
  --fast-list
  --links
  --retries 5
  --low-level-retries 20
  --stats 30s
)

RCLONE_CHECK_FLAGS=(
  --one-way
  --size-only
  --checkers "${PRISM_DEEP_ARCHIVE_CHECKERS:-64}"
)

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

require_safe_path() {
  local src="$1"
  case "$src" in
    /|/home|/home/diddy|/home/diddy/Desktop|/mnt|/mnt/storage|/mnt/storage/prism-outputs|/tmp)
      log "REFUSE unsafe broad delete path: $src"
      return 1
      ;;
  esac
  [[ "$src" == /* ]] || {
    log "REFUSE non-absolute path: $src"
    return 1
  }
}

dest_for() {
  local src="$1"
  local rel="${src#/}"
  printf 'r2:%s/%s/%s/' "$BUCKET" "$PREFIX" "$rel"
}

archive_dir_then_delete() {
  local src="$1"
  [[ -e "$src" ]] || {
    log "SKIP missing: $src"
    return 0
  }
  [[ -d "$src" ]] || {
    log "SKIP non-directory: $src"
    return 0
  }
  require_safe_path "$src" || return 1

  local dest
  dest="$(dest_for "$src")"
  local size
  size="$(du -sh "$src" 2>/dev/null | awk '{print $1}')"
  log "ARCHIVE start size=$size src=$src dest=$dest"
  printf 'start\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$size" "$src" "$dest" >> "$MANIFEST"

  rclone copy "$src" "$dest" "${RCLONE_COPY_FLAGS[@]}" 2>&1 | tee -a "$LOG"
  rclone check "$src" "$dest" "${RCLONE_CHECK_FLAGS[@]}" \
    --combined "$STATE_DIR/checks/$(echo "$src" | tr '/ ' '__').combined" 2>&1 | tee -a "$LOG"

  log "VERIFY passed; deleting local src=$src"
  rm -rf --one-file-system -- "$src"
  printf 'deleted\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$size" "$src" "$dest" >> "$MANIFEST"
}

archive_prism_outputs_then_prune() {
  local src="/mnt/storage/prism-outputs"
  [[ -d "$src" ]] || return 0
  local dest
  dest="$(dest_for "$src")"
  local size
  size="$(du -sh "$src" 2>/dev/null | awk '{print $1}')"
  log "ARCHIVE start size=$size src=$src dest=$dest excludes=teacher-corpus,.spike-shipper-parquet-cache,sync-manifests"
  printf 'start\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$size" "$src" "$dest" >> "$MANIFEST"

  local excludes=(
    --exclude '/teacher-corpus/**'
    --exclude '/.spike-shipper-parquet-cache/**'
    --exclude '/.r2-sync-manifest.jsonl'
    --exclude '/.r2-reject-cache.jsonl'
  )
  rclone copy "$src" "$dest" "${RCLONE_COPY_FLAGS[@]}" "${excludes[@]}" 2>&1 | tee -a "$LOG"
  rclone check "$src" "$dest" "${RCLONE_CHECK_FLAGS[@]}" "${excludes[@]}" \
    --combined "$STATE_DIR/checks/mnt_storage_prism_outputs.combined" 2>&1 | tee -a "$LOG"

  log "VERIFY passed; pruning /mnt/storage/prism-outputs except live teacher-corpus shipper state"
  find "$src" -mindepth 1 -maxdepth 1 \
    ! -name 'teacher-corpus' \
    ! -name '.spike-shipper-parquet-cache' \
    ! -name '.r2-sync-manifest.jsonl' \
    ! -name '.r2-reject-cache.jsonl' \
    -exec rm -rf --one-file-system -- {} +
  printf 'deleted_children\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$size" "$src" "$dest" >> "$MANIFEST"
}

archive_tmp_prism_then_prune() {
  shopt -s nullglob
  local dirs=(/tmp/prism_* /tmp/pytest-of-diddy)
  shopt -u nullglob
  for src in "${dirs[@]}"; do
    archive_dir_then_delete "$src"
  done
}

main() {
  log "deep archive prune starting bucket=$BUCKET prefix=$PREFIX manifest=$MANIFEST"
  rclone lsd "r2:$BUCKET" >/dev/null

  # Free the root volume first. The workstation is already over 90% on `/`,
  # while the teacher-corpus shipper depends on `/mnt/storage` staying healthy.
  archive_tmp_prism_then_prune

  local candidates=(
    /home/diddy/Desktop/Prism4D-bio/output
    /home/diddy/Desktop/Prism4D-bio/.prism_orchestration
    /home/diddy/Desktop/Prism4D-bio/benchmarks
    /home/diddy/Desktop/Prism4D-bio/benchmark
    /home/diddy/Desktop/Prism4D-bio/target
    /home/diddy/Desktop/Prism4D-v1.1-frozen
    /home/diddy/Desktop/prism-production-test
    /home/diddy/Desktop/ble-countersurveillance
    /home/diddy/prism-archive
    /home/diddy/forensics
    /home/diddy/prism-working
    /home/diddy/prism4d_training
    /home/diddy/p53_full_original
    /home/diddy/prism4d_analysis
    /home/diddy/trash_staging
    /mnt/storage/Prism4D-v1.1-frozen
    /mnt/storage/Prism4D-bio
    /mnt/storage/diddy_recovery_archive
    /mnt/storage/spike-audit
    /mnt/storage/prism_pub_runs
    /mnt/storage/PHASE_A_V2_SWEEP_20260508T185540Z
    /mnt/storage/PHASE_A_FULL_PASS_20260508T181727Z
    /mnt/storage/PHASE_A_V2_CAPTURE_20260508T184726Z
    /mnt/storage/tmp
    /mnt/storage/uspto
    /mnt/storage/7c8r_dimer_discovery_v9c_prime_20260508_064301
    /mnt/storage/prism_pathA_baseline_20260504_115113
    /mnt/storage/parquet-test
  )

  for src in "${candidates[@]}"; do
    archive_dir_then_delete "$src"
  done

  # Keep this large tree until after the targeted roots above so verified
  # deletion starts returning space sooner instead of waiting on a monolithic
  # 1T prism-outputs pass.
  archive_prism_outputs_then_prune

  log "deep archive prune complete"
  df -h / /mnt/storage | tee -a "$LOG"
}

main "$@"
