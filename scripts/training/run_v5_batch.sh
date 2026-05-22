#!/usr/bin/env bash
# v5 batch driver: runs prism_only_feature_extractor_v5.py over all local TWIN targets,
# auto-discovers companion files (binding_sites, kcc, therm, asc, gcpid, phasors,
# druggability, acl_contrast, ensemble_trajectory, kcc_validation, ground_truth, P2Rank)
# and launches phase_manifold_ranker per target.

set -u

OUT_DIR="/home/diddy/prism4d_training/prism_only_features"
PM_DIR="/home/diddy/prism4d_training/phase_manifold"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACTOR="$SCRIPT_DIR/prism_only_feature_extractor_v5.py"
RANKER="$(dirname "$SCRIPT_DIR")/phase_manifold_ranker.py"
LOG_DIR="/home/diddy/prism4d_training/extractor_logs"

mkdir -p "$OUT_DIR" "$PM_DIR" "$LOG_DIR"

SEARCH_ROOTS=(
    "/mnt/storage/prism-outputs/hect-family"
    "/mnt/storage/diddy_recovery_archive/m1_strict_dcc"
    "/mnt/storage/prism-outputs/m1-strict-dcc-panel"
    "/mnt/storage/prism-outputs/runs"
    "/mnt/storage/prism-outputs/blind_validation"
)

TARGETS=()
for root in "${SEARCH_ROOTS[@]}"; do
    [ -d "$root" ] || continue
    while IFS= read -r arrow_path; do
        [ -z "$arrow_path" ] && continue
        TARGETS+=("$arrow_path")
    done < <(find "$root" -name "*.topology.spike_events.arrow" -size +1G 2>/dev/null)
done

echo "v5 batch: ${#TARGETS[@]} TWIN-density arrow files"
echo "output: $OUT_DIR"
echo ""

done_count=0; fail_count=0
for arrow in "${TARGETS[@]}"; do
    dir="$(dirname "$arrow")"
    base="$(basename "$arrow" .topology.spike_events.arrow)"
    parent="$(dirname "$dir")"
    grandparent="$(dirname "$parent")"
    key="${base}_$(basename "$parent")"
    out="$OUT_DIR/${key}_v5.parquet"
    log="$LOG_DIR/${key}_v5.log"
    pm_outdir="$PM_DIR/${key}"

    if [ -f "$out" ]; then
        echo "[skip] $key"
        done_count=$((done_count+1))
        continue
    fi

    args=("--arrow" "$arrow" "--output" "$out" "--phase-manifold-script" "$RANKER")

    for sig in binding_sites kcc_visualization topology.prism_therm topology.asc_consensus \
               topology.gcpid_synergy topology.phasors topology.druggability \
               topology.acl_contrast ensemble_trajectory kcc_validation ; do
        path="${dir}/${base}.${sig}.json"
        case "$sig" in
            topology.phasors) path="${dir}/${base}.${sig}.bin" ;;
            topology.druggability) path="${dir}/${base}.${sig}.pdb" ;;
            topology.acl_contrast) path="${dir}/${base}.${sig}.bin" ;;
        esac
        if [ -f "$path" ]; then
            case "$sig" in
                binding_sites) args+=("--binding-sites" "$path") ;;
                kcc_visualization) args+=("--kcc" "$path") ;;
                topology.prism_therm) args+=("--therm" "$path") ;;
                topology.asc_consensus) args+=("--asc-consensus" "$path") ;;
                topology.gcpid_synergy) args+=("--gcpid" "$path") ;;
                topology.phasors) args+=("--phasors" "$path") ;;
                topology.druggability) args+=("--druggability-pdb" "$path") ;;
                topology.acl_contrast) args+=("--acl-contrast" "$path") ;;
                ensemble_trajectory) args+=("--ensemble-trajectory" "$path") ;;
                kcc_validation) args+=("--kcc-validation" "$path") ;;
            esac
        fi
    done

    # ground_truth.json (varies in name)
    for gt in "${dir}/${base}_ground_truth.json" "${parent}/${base}_ground_truth.json" "${grandparent}/${base}_ground_truth.json"; do
        [ -f "$gt" ] && args+=("--ground-truth" "$gt") && break
    done

    # P2Rank residues.csv
    for p2 in "${parent}/p2rank/${base}_clean.pdb_residues.csv" "${grandparent}/p2rank/${base}_clean.pdb_residues.csv" "${parent}/p2rank/${base}.pdb_residues.csv"; do
        [ -f "$p2" ] && args+=("--p2rank-residues" "$p2") && break
    done

    args+=("--phase-manifold-outdir" "$pm_outdir")

    echo "[run] $key  ($(date +%H:%M:%S))"
    python3 "$EXTRACTOR" "${args[@]}" > "$log" 2>&1
    if [ $? -eq 0 ]; then
        sz=$(du -h "$out" 2>/dev/null | cut -f1)
        echo "[done] $key  ($(date +%H:%M:%S)) — $sz"
        done_count=$((done_count+1))
    else
        echo "[FAIL] $key  rc=$?  (see $log)"
        fail_count=$((fail_count+1))
    fi
done

echo ""
echo "=== v5 batch summary ==="
echo "done: $done_count"
echo "fail: $fail_count"
ls -la "$OUT_DIR"/*_v5.parquet 2>/dev/null | wc -l
