#!/usr/bin/env bash
# PRISM-4D Infrastructure Status Report — Hourly Deep Check
# Installed via cron: 0 * * * *
# Output: ~/Desktop/prism_infra_status_TIMESTAMP.txt

set -uo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT=~/Desktop/prism_infra_status_${TIMESTAMP}.txt
WORKER_URL="https://prism-feature-pipeline.is-0b9.workers.dev"
WORK_DIR=/mnt/storage/prism-outputs/10k-runs
LOG_DIR=/mnt/storage/prism-outputs/_corpus_runner_logs
R2_OUTPUT_PREFIX="10k-runs-pct70"
R2_SIZE_CACHE="/mnt/storage/prism-outputs/_corpus_runner_logs/.r2_size_cache"

# Find the active corpus runner log (most recent by PID match)
RUNNER_PID=$(pgrep -f "prism-corpus-runner.sh.*pct70" | head -1 || true)
ACTIVE_LOG=$(ls -t "$LOG_DIR"/run_*_per_target.log 2>/dev/null | head -1)

{
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║            PRISM-4D INFRASTRUCTURE STATUS REPORT               ║"
echo "║            $(date -Iseconds)                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

# ─────────────────────────────────────────────────────────────────────
# Section 1 — Campaign Health
# ─────────────────────────────────────────────────────────────────────
echo "═══ SECTION 1: CAMPAIGN HEALTH ═══"
echo

if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
    echo "Runner PID:      $RUNNER_PID — ALIVE"
else
    echo "Runner PID:      ${RUNNER_PID:-N/A} — DEAD ⚠"
fi

ENGINE_PID=$(pgrep -f "nhs_rt_full.*spike-percentile.70" | head -1 || true)
if [[ -n "$ENGINE_PID" ]] && kill -0 "$ENGINE_PID" 2>/dev/null; then
    ENGINE_ELAPSED=$(ps -o etimes= -p "$ENGINE_PID" 2>/dev/null | tr -d ' ')
    CURRENT_TARGET=$(ps -o args= -p "$ENGINE_PID" 2>/dev/null | grep -oP '10k-runs/\K[^/]+' || echo "?")
    echo "Engine PID:      $ENGINE_PID — ALIVE (${ENGINE_ELAPSED}s elapsed)"
    echo "Current target:  $CURRENT_TARGET"
else
    echo "Engine PID:      ${ENGINE_PID:-N/A} — IDLE"
    echo "Current target:  none"
fi

if [[ -f "$ACTIVE_LOG" ]]; then
    COMPLETED=$(wc -l < "$ACTIVE_LOG")
    N_OK=$(grep -c " OK " "$ACTIVE_LOG" 2>/dev/null || echo 0)
    N_FAIL=$((COMPLETED - N_OK))
    MANIFEST_TOTAL=$(cat "$LOG_DIR"/proteome_1000_pct70_372.txt 2>/dev/null | grep -cv '^[[:space:]]*\(#\|$\)' || echo "?")
    FAIL_RATE="0.0"
    if [[ "$COMPLETED" -gt 0 ]]; then
        FAIL_RATE=$(awk "BEGIN{printf \"%.1f\", ($N_FAIL/$COMPLETED)*100}")
    fi

    AVG_ENGINE=$(grep -oP 'engine=\K[0-9]+' "$ACTIVE_LOG" | awk '{s+=$1; n++} END{if(n>0) printf "%.0f", s/n; else print "?"}')
    REMAINING=$((MANIFEST_TOTAL - COMPLETED))
    if [[ "$AVG_ENGINE" =~ ^[0-9]+$ && "$REMAINING" -gt 0 ]]; then
        ETA_SECS=$((REMAINING * AVG_ENGINE * 115 / 100))
        ETA_HRS=$((ETA_SECS / 3600))
        ETA_MIN=$(( (ETA_SECS % 3600) / 60 ))
        ETA_DATE=$(date -d "+${ETA_SECS} seconds" '+%Y-%m-%d %H:%M')
    else
        ETA_HRS="?"
        ETA_MIN="?"
        ETA_DATE="?"
    fi

    echo
    echo "Completed:       $COMPLETED / $MANIFEST_TOTAL"
    echo "OK:              $N_OK"
    echo "Failed:          $N_FAIL  (rate: ${FAIL_RATE}%)"
    echo "Avg engine time: ${AVG_ENGINE}s"
    echo "ETA:             ${ETA_HRS}h ${ETA_MIN}m  (~${ETA_DATE})"
    echo
    echo "Last 3 completed:"
    tail -3 "$ACTIVE_LOG" | sed 's/^/  /'
else
    echo "No active per-target log found"
fi
echo

# ─────────────────────────────────────────────────────────────────────
# Section 2 — GPU Status
# ─────────────────────────────────────────────────────────────────────
echo "═══ SECTION 2: GPU STATUS ═══"
echo
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null | while IFS=, read -r name util mem_used mem_total temp power; do
        echo "GPU:             $name"
        echo "Utilization:     $util"
        echo "Memory:          $mem_used / $mem_total"
        echo "Temperature:     $temp"
        echo "Power:           $power"
    done
    echo
    echo "GPU processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  none"
else
    echo "nvidia-smi not found"
fi
echo

# ─────────────────────────────────────────────────────────────────────
# Section 3 — Disk Status
# ─────────────────────────────────────────────────────────────────────
echo "═══ SECTION 3: DISK STATUS ═══"
echo
echo "/mnt/storage:"
df -h /mnt/storage 2>/dev/null | tail -1 | awk '{printf "  Size: %s  Used: %s  Avail: %s  Use%%: %s\n", $2, $3, $4, $5}'
STORAGE_PCT=$(df /mnt/storage 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')

echo "/ (root):"
df -h / | tail -1 | awk '{printf "  Size: %s  Used: %s  Avail: %s  Use%%: %s\n", $2, $3, $4, $5}'

echo
RUNS_SIZE=$(du -sh "$WORK_DIR" 2>/dev/null | cut -f1)
RUNS_COUNT=$(ls -d "$WORK_DIR"/*/ 2>/dev/null | wc -l)
echo "10k-runs local:  $RUNS_SIZE across $RUNS_COUNT target dirs"

if [[ -n "$STORAGE_PCT" && "$STORAGE_PCT" -ge 80 ]]; then
    echo "⚠  WARNING: /mnt/storage at ${STORAGE_PCT}% — exceeds 80% threshold"
fi
echo

# ─────────────────────────────────────────────────────────────────────
# Section 4 — R2 Status
# ─────────────────────────────────────────────────────────────────────
echo "═══ SECTION 4: R2 STATUS ═══"
echo
R2_DIRS=$(rclone lsd "r2:prism-archive/$R2_OUTPUT_PREFIX/" 2>/dev/null | wc -l)
echo "Targets on R2:   $R2_DIRS"

LAST_R2=$(rclone lsd "r2:prism-archive/$R2_OUTPUT_PREFIX/" 2>/dev/null | sort -k2,3 | tail -1 | awk '{print $NF}')
echo "Last uploaded:    ${LAST_R2:-unknown}"

# R2 total size — cache to avoid hammering (run every 6 hours)
REFRESH_SIZE=false
if [[ -f "$R2_SIZE_CACHE" ]]; then
    CACHE_AGE=$(( $(date +%s) - $(stat -c %Y "$R2_SIZE_CACHE") ))
    if [[ "$CACHE_AGE" -gt 21600 ]]; then
        REFRESH_SIZE=true
    fi
else
    REFRESH_SIZE=true
fi

if [[ "$REFRESH_SIZE" == "true" ]]; then
    R2_SIZE_JSON=$(rclone size "r2:prism-archive/$R2_OUTPUT_PREFIX/" --json 2>/dev/null || echo '{"bytes":0,"count":0}')
    echo "$R2_SIZE_JSON" > "$R2_SIZE_CACHE"
else
    R2_SIZE_JSON=$(cat "$R2_SIZE_CACHE")
fi
R2_BYTES=$(echo "$R2_SIZE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('bytes',0))" 2>/dev/null || echo 0)
R2_GB=$(awk "BEGIN{printf \"%.1f\", $R2_BYTES/1073741824}")
R2_COUNT=$(echo "$R2_SIZE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('count',0))" 2>/dev/null || echo 0)
echo "R2 total size:   ${R2_GB} GB  ($R2_COUNT files)  [cache age: $(( $(date +%s) - $(stat -c %Y "$R2_SIZE_CACHE" 2>/dev/null || echo $(date +%s)) ))s]"

# Spot check latest target for spike data
if [[ -n "$LAST_R2" ]]; then
    SPIKE_CHECK=$(rclone lsf "r2:prism-archive/$R2_OUTPUT_PREFIX/$LAST_R2/" 2>/dev/null | grep -c "spike_events" || true)
    echo "Spot check ($LAST_R2): spike_events files = $SPIKE_CHECK"
fi
echo

# ─────────────────────────────────────────────────────────────────────
# Section 5 — Cloudflare Pipeline Status
# ─────────────────────────────────────────────────────────────────────
echo "═══ SECTION 5: CLOUDFLARE PIPELINE STATUS ═══"
echo

# D1 target count
D1_TARGETS=$(curl -s -m 10 "$WORKER_URL/targets" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else d.get('count','?'))" 2>/dev/null || echo "ERR")
echo "D1 targets:      $D1_TARGETS"

# D1 stats
D1_STATS=$(curl -s -m 10 "$WORKER_URL/stats" 2>/dev/null || echo "{}")
echo "D1 stats:        $D1_STATS" | head -1

# D1 site_features for latest target
if [[ -n "$LAST_R2" ]]; then
    SF_COUNT=$(curl -s -m 10 "$WORKER_URL/site-features/${LAST_R2%/}" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else '?')" 2>/dev/null || echo "ERR")
    echo "D1 site_features ($LAST_R2): $SF_COUNT rows"
fi

# Queue health: check for queue_consumer source
QUEUE_ROWS=$(curl -s -m 10 "$WORKER_URL/targets?source=queue_consumer" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if isinstance(d, list):
        print(len(d))
    else:
        print('?')
except:
    print('?')
" 2>/dev/null || echo "?")
echo "Queue consumer:  $QUEUE_ROWS targets ingested via queue"
if [[ "$QUEUE_ROWS" == "0" || "$QUEUE_ROWS" == "?" ]]; then
    COMPLETED_COUNT=${COMPLETED:-0}
    if [[ "$COMPLETED_COUNT" -gt 24 ]]; then
        echo "  ⚠  RED: Queue pipeline appears NOT firing (0 queue_consumer rows after $COMPLETED_COUNT targets)"
    fi
fi

# Campaign Durable Object
CAMPAIGN_STATUS=$(curl -s -m 10 "$WORKER_URL/campaign/status" 2>/dev/null || echo "ERR")
echo "Campaign DO:     $CAMPAIGN_STATUS" | head -1

echo "Analytics Engine: unverified (GraphQL-only, no direct query method)"
echo

# ─────────────────────────────────────────────────────────────────────
# Section 6 — Feature Extraction Status
# ─────────────────────────────────────────────────────────────────────
echo "═══ SECTION 6: FEATURE EXTRACTION STATUS ═══"
echo
EXTRACT_PID=$(pgrep -f "extract_all_features" | head -1 || true)
if [[ -n "$EXTRACT_PID" ]] && kill -0 "$EXTRACT_PID" 2>/dev/null; then
    EXTRACT_ELAPSED=$(ps -o etimes= -p "$EXTRACT_PID" 2>/dev/null | tr -d ' ')
    echo "extract_all_features: PID $EXTRACT_PID — RUNNING (${EXTRACT_ELAPSED}s)"
else
    echo "extract_all_features: NOT RUNNING"
fi

# Count .npz files in features output directories
for fdir in /mnt/storage/spike-audit/features-176gate /mnt/storage/spike-audit/features; do
    if [[ -d "$fdir" ]]; then
        NPZ_COUNT=$(find "$fdir" -name "*.npz" 2>/dev/null | wc -l)
        echo "  $fdir: $NPZ_COUNT .npz files"
    fi
done

# D1 residue_features count
RF_COUNT=$(curl -s -m 10 "$WORKER_URL/stats" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if isinstance(d, dict):
        print(d.get('residue_features', '?'))
    else:
        print('?')
except:
    print('?')
" 2>/dev/null || echo "?")
echo "D1 residue_features: $RF_COUNT rows"
echo

# ─────────────────────────────────────────────────────────────────────
# Section 7 — Training Pipeline Readiness
# ─────────────────────────────────────────────────────────────────────
echo "═══ SECTION 7: TRAINING PIPELINE READINESS ═══"
echo

PROJECT_DIR="/home/diddy/Desktop/Prism4D-bio"
echo "Training scripts in scripts/training/:"
if [[ -d "$PROJECT_DIR/scripts/training" ]]; then
    ls -lhS "$PROJECT_DIR/scripts/training/"*.py 2>/dev/null | awk '{printf "  %-50s %s\n", $NF, $5}'
else
    echo "  directory not found"
fi

echo
echo "RunPod status:"
if command -v runpodctl &>/dev/null; then
    runpodctl get pods 2>/dev/null | head -5 | sed 's/^/  /' || echo "  runpodctl failed"
elif [[ -f ~/.runpod/config.toml || -f ~/.runpod.toml ]]; then
    echo "  RunPod config exists but runpodctl not in PATH"
else
    echo "  No RunPod config found — not provisioned"
fi

echo
echo "Directive phases:"
for phase in "corpus_generation" "feature_extraction" "d1_population" "model_training" "inference_benchmark"; do
    case "$phase" in
        corpus_generation)
            if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
                echo "  $phase: IN PROGRESS ($COMPLETED/$MANIFEST_TOTAL)"
            elif [[ "${COMPLETED:-0}" -ge "${MANIFEST_TOTAL:-999}" ]]; then
                echo "  $phase: COMPLETE"
            else
                echo "  $phase: STOPPED ($COMPLETED/${MANIFEST_TOTAL:-?})"
            fi
            ;;
        feature_extraction)
            if [[ -n "$EXTRACT_PID" ]] && kill -0 "$EXTRACT_PID" 2>/dev/null; then
                echo "  $phase: IN PROGRESS"
            else
                echo "  $phase: PENDING"
            fi
            ;;
        *)
            echo "  $phase: PENDING"
            ;;
    esac
done
echo

# ─────────────────────────────────────────────────────────────────────
# Section 8 — Alerts
# ─────────────────────────────────────────────────────────────────────
echo "═══ SECTION 8: ALERTS ═══"
echo

RED_COUNT=0
YELLOW_COUNT=0

# RED checks
if [[ -n "${STORAGE_PCT:-}" && "$STORAGE_PCT" -ge 80 ]]; then
    echo "🔴 RED:    Disk /mnt/storage at ${STORAGE_PCT}%"
    RED_COUNT=$((RED_COUNT+1))
fi

FAIL_RATE_NUM=$(echo "${FAIL_RATE:-0}" | tr -d '.')
if [[ "${FAIL_RATE_NUM:-0}" -gt 20 ]]; then
    echo "🔴 RED:    Failure rate ${FAIL_RATE}% exceeds 2% threshold"
    RED_COUNT=$((RED_COUNT+1))
fi

if [[ -n "$ENGINE_PID" ]] && ! kill -0 "$ENGINE_PID" 2>/dev/null && [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "$GPU_UTIL" && "$GPU_UTIL" -lt 5 ]]; then
        echo "🔴 RED:    GPU idle during active campaign (${GPU_UTIL}%)"
        RED_COUNT=$((RED_COUNT+1))
    fi
fi

if [[ "$QUEUE_ROWS" == "0" || "$QUEUE_ROWS" == "?" ]]; then
    COMPLETED_COUNT=${COMPLETED:-0}
    if [[ "$COMPLETED_COUNT" -gt 24 ]]; then
        echo "🔴 RED:    Queue pipeline not firing ($QUEUE_ROWS queue_consumer rows after $COMPLETED_COUNT targets)"
        RED_COUNT=$((RED_COUNT+1))
    fi
fi

# YELLOW checks
if [[ -n "${STORAGE_PCT:-}" && "$STORAGE_PCT" -ge 60 && "$STORAGE_PCT" -lt 80 ]]; then
    echo "🟡 YELLOW: Disk /mnt/storage at ${STORAGE_PCT}%"
    YELLOW_COUNT=$((YELLOW_COUNT+1))
fi

if [[ -n "${ENGINE_ELAPSED:-}" && "$ENGINE_ELAPSED" -gt 600 ]]; then
    echo "🟡 YELLOW: Current target engine time ${ENGINE_ELAPSED}s (>10 min)"
    YELLOW_COUNT=$((YELLOW_COUNT+1))
fi

if [[ "$RED_COUNT" -eq 0 && "$YELLOW_COUNT" -eq 0 ]]; then
    echo "🟢 GREEN:  All systems nominal"
fi

echo
echo "────────────────────────────────────────────────────────────────"
echo "Report written: $REPORT"
echo "Next report:    $(date -d '+1 hour' '+%Y-%m-%d %H:%M')"

} > "$REPORT" 2>&1

# Also print to stdout when run interactively
cat "$REPORT"
