#!/bin/bash
# Non-naive cleanup daemon for pct70 campaign
# Checks every 2 minutes, validates before deleting, logs everything

LOG="/tmp/cleanup_daemon.log"
LOCK="/tmp/cleanup_daemon.lock"
LOCAL_DIR="/mnt/storage/prism-outputs/10k-runs"
R2_PREFIX="r2:prism-archive/10k-runs-pct70"
MIN_AGE_MIN=15          # Don't touch anything newer than 15 min
DISK_WARN_PCT=60        # Start aggressive cleanup above this
DISK_CRIT_PCT=80        # Emergency cleanup above this

exec >> "$LOG" 2>&1

echo "$(date) Cleanup daemon started (PID $$)"

# Prevent duplicate daemons
if [ -f "$LOCK" ]; then
    OLD_PID=$(cat "$LOCK")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "$(date) Another daemon running (PID $OLD_PID). Exiting."
        exit 1
    fi
fi
echo $$ > "$LOCK"
trap "rm -f $LOCK" EXIT

while true; do
    # Check disk pressure
    DISK_PCT=$(df /mnt/storage --output=pcent | tail -1 | tr -d ' %')

    if [ "$DISK_PCT" -ge "$DISK_CRIT_PCT" ]; then
        MODE="CRITICAL"
        MIN_AGE=5       # More aggressive — 5 min age threshold
        SLEEP=30        # Check every 30 seconds
    elif [ "$DISK_PCT" -ge "$DISK_WARN_PCT" ]; then
        MODE="WARN"
        MIN_AGE=10
        SLEEP=60
    else
        MODE="NORMAL"
        MIN_AGE=$MIN_AGE_MIN
        SLEEP=120
    fi

    # Get the currently active target from the runner (don't touch it)
    ACTIVE_TARGET=""
    ENGINE_PID=$(pgrep -f nhs_rt_full 2>/dev/null | head -1)
    if [ -n "$ENGINE_PID" ]; then
        # Extract target name from the engine's -o argument
        ACTIVE_TARGET=$(ps -p "$ENGINE_PID" -o args= 2>/dev/null | grep -oP '(?<=-o\s)\S+' | xargs basename 2>/dev/null)
    fi

    DELETED=0
    KEPT=0
    SKIPPED=0

    for d in "$LOCAL_DIR"/*/; do
        [ ! -d "$d" ] && continue
        t=$(basename "$d")

        # Never delete the active target
        if [ "$t" = "$ACTIVE_TARGET" ]; then
            SKIPPED=$((SKIPPED+1))
            continue
        fi

        # Check age — use the NEWEST file in the directory, not dir mtime
        NEWEST=$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)
        NOW=$(date +%s)
        if [ -n "$NEWEST" ]; then
            AGE_MIN=$(( (NOW - NEWEST) / 60 ))
        else
            AGE_MIN=999  # Empty dir, safe to delete
        fi

        if [ "$AGE_MIN" -lt "$MIN_AGE" ]; then
            SKIPPED=$((SKIPPED+1))
            continue
        fi

        # Compare file-by-file: every local file must exist on R2 with same size
        MISMATCH=0
        while IFS= read -r line; do
            size=$(echo "$line" | awk '{print $1}')
            name=$(echo "$line" | awk '{$1=""; print substr($0,2)}')
            # Check if R2 has this exact file with same size
            r2_size=$(rclone ls "$R2_PREFIX/$t/$name" 2>/dev/null | awk '{print $1}')
            if [ "$r2_size" != "$size" ]; then
                MISMATCH=$((MISMATCH+1))
                if [ "$MISMATCH" -ge 3 ]; then
                    break  # Don't check every file if already failing
                fi
            fi
        done < <(ls -l "$d" | tail -n +2 | awk '{print $5, $NF}')

        if [ "$MISMATCH" -eq 0 ]; then
            rm -rf "$d"
            DELETED=$((DELETED+1))
            echo "$(date +%H:%M:%S) DELETED $t (age=${AGE_MIN}m mode=$MODE)"
        else
            KEPT=$((KEPT+1))
            # In CRITICAL mode, force upload then retry
            if [ "$MODE" = "CRITICAL" ]; then
                echo "$(date +%H:%M:%S) CRITICAL_UPLOAD $t ($MISMATCH mismatches)"
                rclone copy "$d" "$R2_PREFIX/$t/" --transfers 8 --checkers 8 --buffer-size 64M --retries 3 --quiet
            fi
        fi
    done

    if [ $((DELETED + KEPT)) -gt 0 ]; then
        DISK_NOW=$(df /mnt/storage --output=pcent | tail -1 | tr -d ' %')
        echo "$(date +%H:%M:%S) [$MODE] disk=${DISK_NOW}% deleted=$DELETED kept=$KEPT skipped=$SKIPPED active=$ACTIVE_TARGET"
    fi

    sleep $SLEEP
done
