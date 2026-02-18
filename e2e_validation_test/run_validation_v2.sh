#!/bin/bash
set -e

BASE="/home/diddy/Desktop/Prism4D-bio"
VALIDATION_DIR="/home/diddy/Desktop/Apo_Holo_pdb/PRISM4D_validation"
OUTPUT_DIR="$BASE/e2e_validation_test"
LOG="$OUTPUT_DIR/logs/pipeline_v2.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"/{prep,results,logs}

echo "╔══════════════════════════════════════════════════════════════════╗" | tee $LOG
echo "║     PRISM4D E2E Validation Pipeline v2 - $TIMESTAMP     ║" | tee -a $LOG
echo "║     10 structures × 3 replicas × 50K steps (--fast)              ║" | tee -a $LOG
echo "╚══════════════════════════════════════════════════════════════════╝" | tee -a $LOG

# Define structures
STRUCTURES=(
    "tier1_cryptosite/apo/1ade"
    "tier1_cryptosite/apo/1bj4"
    "tier2_dewetting/apo/1btl"
    "tier2_dewetting/apo/1hhp"
    "tier3_positive_controls/1a4q"
    "tier3_positive_controls/1ere"
    "tier4_allosteric/apo/1g1f"
    "tier4_allosteric/apo/1qmf"
    "tier5_negative_controls/1crn"
    "tier5_negative_controls/1igt"
)

cd "$BASE"

echo "" | tee -a $LOG
echo "=== STAGE 2: NHS-RT-FULL Pipeline (--fast = 50K steps) ===" | tee -a $LOG
echo "Config: --fast --rt-clustering --parallel --replicas 3 --adaptive-epsilon --multi-scale" | tee -a $LOG
echo "Started: $(date)" | tee -a $LOG

for topo in "$OUTPUT_DIR"/prep/*.topology.json; do
    name=$(basename "$topo" .topology.json)
    result_dir="$OUTPUT_DIR/results/$name"
    
    if [ -d "$result_dir" ] && [ -f "$result_dir/summary.json" ]; then
        echo "  [SKIP] $name - results exist" | tee -a $LOG
        continue
    fi
    
    echo "" | tee -a $LOG
    echo "  [RUN] $name (3 replicas, 50K steps, RT+BVH+Cryo-UV)..." | tee -a $LOG
    start_time=$(date +%s)
    
    target/release/nhs-rt-full \
        --topology "$topo" \
        --output "$result_dir" \
        --replicas 3 \
        --fast \
        --rt-clustering \
        --parallel \
        --adaptive-epsilon \
        --multi-scale \
        --verbose 2>&1 | tee "$OUTPUT_DIR/logs/${name}_v2.log" | grep -E "INFO|✓|✗|Complete|Error|steps" | tail -10 | tee -a $LOG
    
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "    → Completed in ${elapsed}s" | tee -a $LOG
    
    # Quick result summary
    if [ -f "$result_dir/summary.json" ]; then
        sites=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('total_sites', '?'))" 2>/dev/null || echo "?")
        echo "    → Sites found: $sites" | tee -a $LOG
    fi
done

echo "" | tee -a $LOG
echo "=== VALIDATION COMPLETE ===" | tee -a $LOG
echo "Finished: $(date)" | tee -a $LOG

# Summary table
echo "" | tee -a $LOG
echo "╔═══════════════════════════════════════════════════════════════════╗" | tee -a $LOG
echo "║                      RESULTS SUMMARY                              ║" | tee -a $LOG  
echo "╠═══════════════════════════════════════════════════════════════════╣" | tee -a $LOG
printf "║ %-12s │ %-8s │ %-10s │ %-10s │ %-8s ║\n" "Structure" "Tier" "Sites" "Druggable" "Status" | tee -a $LOG
echo "╠═══════════════════════════════════════════════════════════════════╣" | tee -a $LOG

declare -A TIERS
TIERS[1ade]="T1-Crypto" TIERS[1bj4]="T1-Crypto"
TIERS[1btl]="T2-Dewet" TIERS[1hhp]="T2-Dewet"
TIERS[1a4q]="T3-Posit" TIERS[1ere]="T3-Posit"
TIERS[1g1f]="T4-Allo" TIERS[1qmf]="T4-Allo"
TIERS[1crn]="T5-Negat" TIERS[1igt]="T5-Negat"

for result_dir in "$OUTPUT_DIR"/results/*/; do
    name=$(basename "$result_dir")
    tier=${TIERS[$name]:-"?"}
    if [ -f "$result_dir/summary.json" ]; then
        sites=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('total_sites', 0))" 2>/dev/null || echo "?")
        druggable=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('druggable_sites', 0))" 2>/dev/null || echo "?")
        status="✓ Done"
    else
        sites="-"
        druggable="-"
        status="✗ Failed"
    fi
    printf "║ %-12s │ %-8s │ %10s │ %10s │ %-8s ║\n" "$name" "$tier" "$sites" "$druggable" "$status" | tee -a $LOG
done
echo "╚═══════════════════════════════════════════════════════════════════╝" | tee -a $LOG
