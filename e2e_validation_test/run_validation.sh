#!/bin/bash
set -e

BASE="/home/diddy/Desktop/Prism4D-bio"
VALIDATION_DIR="/home/diddy/Desktop/Apo_Holo_pdb/PRISM4D_validation"
OUTPUT_DIR="$BASE/e2e_validation_test"
LOG="$OUTPUT_DIR/logs/pipeline.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"/{prep,results,logs}

echo "╔══════════════════════════════════════════════════════════════════╗" | tee $LOG
echo "║     PRISM4D E2E Validation Pipeline - $TIMESTAMP        ║" | tee -a $LOG
echo "║     10 structures × 3 replicas = 30 simulations                  ║" | tee -a $LOG
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

echo "" | tee -a $LOG
echo "=== STAGE 1: PRISM-PREP (Topology Generation) ===" | tee -a $LOG
echo "Started: $(date)" | tee -a $LOG

cd "$BASE"

for struct in "${STRUCTURES[@]}"; do
    name=$(basename "$struct")
    pdb_file="$VALIDATION_DIR/${struct}.pdb"
    topo_file="$OUTPUT_DIR/prep/${name}.topology.json"
    
    if [ -f "$topo_file" ]; then
        echo "  [SKIP] $name - topology exists" | tee -a $LOG
        continue
    fi
    
    echo "  [PREP] $name..." | tee -a $LOG
    python3 scripts/prism-prep "$pdb_file" "$topo_file" 2>&1 | tail -3 | tee -a $LOG
    
    if [ -f "$topo_file" ]; then
        atoms=$(grep -o '"n_atoms":[0-9]*' "$topo_file" | grep -o '[0-9]*')
        echo "    → Created: $atoms atoms" | tee -a $LOG
    else
        echo "    → FAILED" | tee -a $LOG
    fi
done

echo "" | tee -a $LOG
echo "=== STAGE 1B: Composition Analysis ===" | tee -a $LOG
echo "Started: $(date)" | tee -a $LOG

# Quick composition check (inline)
for topo in "$OUTPUT_DIR"/prep/*.topology.json; do
    name=$(basename "$topo" .topology.json)
    atoms=$(grep -o '"n_atoms":[0-9]*' "$topo" | grep -o '[0-9]*')
    residues=$(grep -o '"n_residues":[0-9]*' "$topo" | grep -o '[0-9]*')
    echo "  $name: $atoms atoms, $residues residues" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "=== STAGE 2: NHS-RT-FULL Pipeline ===" | tee -a $LOG
echo "Config: --fast --rt-clustering --parallel --replicas 3 --adaptive-epsilon --multi-scale --ultimate-mode" | tee -a $LOG
echo "Started: $(date)" | tee -a $LOG

for topo in "$OUTPUT_DIR"/prep/*.topology.json; do
    name=$(basename "$topo" .topology.json)
    result_dir="$OUTPUT_DIR/results/$name"
    
    if [ -d "$result_dir" ] && [ -f "$result_dir/summary.json" ]; then
        echo "  [SKIP] $name - results exist" | tee -a $LOG
        continue
    fi
    
    echo "  [RUN] $name (3 replicas, RT+BVH+Cryo-UV)..." | tee -a $LOG
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
        --ultimate-mode \
        --verbose 2>&1 | tee -a "$OUTPUT_DIR/logs/${name}.log" | tail -5 | tee -a $LOG
    
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "    → Completed in ${elapsed}s" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "=== VALIDATION COMPLETE ===" | tee -a $LOG
echo "Finished: $(date)" | tee -a $LOG

# Summary
echo "" | tee -a $LOG
echo "=== RESULTS SUMMARY ===" | tee -a $LOG
for result_dir in "$OUTPUT_DIR"/results/*/; do
    name=$(basename "$result_dir")
    if [ -f "$result_dir/summary.json" ]; then
        sites=$(grep -o '"total_sites":[0-9]*' "$result_dir/summary.json" 2>/dev/null | grep -o '[0-9]*' || echo "?")
        druggable=$(grep -o '"druggable_sites":[0-9]*' "$result_dir/summary.json" 2>/dev/null | grep -o '[0-9]*' || echo "?")
        echo "  $name: $sites sites ($druggable druggable)" | tee -a $LOG
    fi
done
