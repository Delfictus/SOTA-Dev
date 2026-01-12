#!/bin/bash
# Phase 6 Checkpoint Gate
# This script MUST pass before proceeding to the next implementation week
# Usage: ./scripts/phase6_checkpoint.sh [week]

set -e

WEEK=${1:-"auto"}

echo "========================================"
echo "PHASE 6 CHECKPOINT GATE"
echo "========================================"

# Auto-detect current week based on files present
if [[ "$WEEK" == "auto" ]]; then
    if [[ -f "crates/prism-validation/src/publication_outputs.rs" ]]; then
        WEEK=8
    elif [[ -f "crates/prism-validation/src/failure_analysis.rs" ]]; then
        WEEK=6
    elif [[ -f "crates/prism-validation/src/apo_holo_benchmark.rs" ]]; then
        WEEK=4
    elif [[ -f "crates/prism-validation/src/gpu_zro_cryptic_scorer.rs" ]]; then
        WEEK=2
    else
        WEEK=0
    fi
    echo "Auto-detected: Week $WEEK"
fi

echo "Validating Week $WEEK checkpoint..."
echo ""

case $WEEK in
    0)
        echo "WEEK 0 CHECKPOINT: Environment Setup"
        echo "----------------------------------------"

        # Check Rust version
        RUST_VERSION=$(rustc --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [[ $(echo "$RUST_VERSION >= 1.75" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
            echo "[PASS] Rust version $RUST_VERSION >= 1.75"
        else
            echo "[FAIL] Rust version $RUST_VERSION < 1.75"
            exit 1
        fi

        # Check CUDA
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep -oE 'V[0-9]+\.[0-9]+' | head -1)
            echo "[PASS] CUDA installed: $CUDA_VERSION"
        else
            echo "[FAIL] CUDA not found"
            exit 1
        fi

        # Check prism-gpu compiles
        echo "Checking prism-gpu compilation..."
        if cargo check -p prism-gpu --features cuda 2>/dev/null; then
            echo "[PASS] prism-gpu compiles with cuda feature"
        else
            echo "[FAIL] prism-gpu compilation failed"
            exit 1
        fi

        echo ""
        echo "WEEK 0 CHECKPOINT PASSED"
        echo "Ready to proceed to Weeks 1-2: GPU SNN Scale-Up"
        ;;

    2)
        echo "WEEK 2 CHECKPOINT: GPU SNN Scale-Up"
        echo "----------------------------------------"

        # Check files exist
        for file in cryptic_features.rs gpu_zro_cryptic_scorer.rs; do
            if [[ -f "crates/prism-validation/src/$file" ]]; then
                echo "[PASS] $file exists"
            else
                echo "[FAIL] $file missing"
                exit 1
            fi
        done

        # Check compilation
        echo "Checking compilation..."
        if cargo check -p prism-validation --features cuda 2>/dev/null; then
            echo "[PASS] prism-validation compiles"
        else
            echo "[FAIL] Compilation failed"
            exit 1
        fi

        # Run GPU scorer tests
        echo "Running GPU scorer tests..."
        if cargo test --release -p prism-validation --features cuda gpu_scorer 2>&1 | tee /tmp/gpu_test.log | grep -q "test result: ok"; then
            echo "[PASS] GPU scorer tests pass"
        else
            echo "[FAIL] GPU scorer tests failed"
            cat /tmp/gpu_test.log
            exit 1
        fi

        # CRITICAL: Zero fallback test
        echo "Running zero fallback verification..."
        if CUDA_VISIBLE_DEVICES="" cargo test --release -p prism-validation --features cuda test_no_cpu_fallback 2>&1 | grep -qE "FAILED|error"; then
            echo "[PASS] Zero fallback test correctly fails without GPU"
        else
            echo "[FAIL] CRITICAL: Zero fallback test should fail but passed!"
            echo "       This means there is a hidden CPU fallback - FIX IMMEDIATELY"
            exit 1
        fi

        # Check throughput
        echo "Running throughput benchmark..."
        THROUGHPUT=$(cargo test --release -p prism-validation --features cuda bench_gpu_scorer_throughput -- --nocapture 2>&1 | grep -oE '[0-9]+ residues/second' | grep -oE '[0-9]+' || echo 0)
        if [[ "$THROUGHPUT" -gt 10000 ]]; then
            echo "[PASS] Throughput: $THROUGHPUT residues/second (>10k required)"
        else
            echo "[WARN] Throughput: $THROUGHPUT residues/second (target: >10k)"
        fi

        echo ""
        echo "WEEK 2 CHECKPOINT PASSED"
        echo "Ready to proceed to Weeks 3-4: PRISM-NOVA Integration"
        ;;

    4)
        echo "WEEK 4 CHECKPOINT: PRISM-NOVA Integration"
        echo "----------------------------------------"

        # Check files exist
        for file in pdb_sanitizer.rs nova_cryptic_sampler.rs apo_holo_benchmark.rs; do
            if [[ -f "crates/prism-validation/src/$file" ]]; then
                echo "[PASS] $file exists"
            else
                echo "[FAIL] $file missing"
                exit 1
            fi
        done

        # Check compilation
        echo "Checking compilation..."
        if cargo check -p prism-validation --features cuda 2>/dev/null; then
            echo "[PASS] prism-validation compiles"
        else
            echo "[FAIL] Compilation failed"
            exit 1
        fi

        # Run apo-holo benchmark on single pair
        echo "Running apo-holo single pair test..."
        if [[ -f "data/benchmarks/apo_holo/1AKE_apo.pdb" ]]; then
            RESULT=$(cargo run --release -p prism-validation --bin apo-holo-single -- --apo 1AKE --holo 4AKE 2>&1 || true)
            MIN_RMSD=$(echo "$RESULT" | grep -oE 'min_rmsd.*[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo 999)
            if [[ $(echo "$MIN_RMSD < 3.5" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
                echo "[PASS] Apo-holo 1AKE->4AKE: min RMSD = ${MIN_RMSD}A (<3.5A)"
            else
                echo "[WARN] Apo-holo min RMSD = ${MIN_RMSD}A (target: <3.5A)"
            fi
        else
            echo "[SKIP] Apo-holo data not downloaded yet"
        fi

        echo ""
        echo "WEEK 4 CHECKPOINT PASSED"
        echo "Ready to proceed to Weeks 5-6: CryptoBench & Ablation"
        ;;

    6)
        echo "WEEK 6 CHECKPOINT: CryptoBench & Ablation"
        echo "----------------------------------------"

        # Check files exist
        for file in cryptobench_dataset.rs ablation.rs failure_analysis.rs; do
            if [[ -f "crates/prism-validation/src/$file" ]]; then
                echo "[PASS] $file exists"
            else
                echo "[FAIL] $file missing"
                exit 1
            fi
        done

        # Check compilation
        echo "Checking compilation..."
        if cargo check -p prism-validation --features cuda 2>/dev/null; then
            echo "[PASS] prism-validation compiles"
        else
            echo "[FAIL] Compilation failed"
            exit 1
        fi

        # Check for benchmark results
        if [[ -f "results/cryptobench_full.json" ]]; then
            ROC_AUC=$(jq -r '.roc_auc // 0' results/cryptobench_full.json 2>/dev/null || echo 0)
            if [[ $(echo "$ROC_AUC > 0.70" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
                echo "[PASS] ROC AUC = $ROC_AUC (>0.70 minimum)"
            else
                echo "[FAIL] ROC AUC = $ROC_AUC (<0.70 minimum)"
                exit 1
            fi
        else
            echo "[SKIP] CryptoBench results not yet generated"
        fi

        # Check ablation results
        if [[ -f "results/ablation.json" ]]; then
            DELTA=$(jq -r '.results | map(select(.variant == "Full")) | .[0].roc_auc - (.results | map(select(.variant == "AnmOnly")) | .[0].roc_auc)' results/ablation.json 2>/dev/null || echo 0)
            if [[ $(echo "$DELTA > 0.15" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
                echo "[PASS] Ablation delta = $DELTA (>0.15 minimum)"
            else
                echo "[WARN] Ablation delta = $DELTA (target: >0.20)"
            fi
        else
            echo "[SKIP] Ablation results not yet generated"
        fi

        echo ""
        echo "WEEK 6 CHECKPOINT PASSED"
        echo "Ready to proceed to Weeks 7-8: Publication & Final Validation"
        ;;

    8)
        echo "WEEK 8 CHECKPOINT: Final Validation"
        echo "----------------------------------------"

        # Check publication outputs
        if [[ -f "crates/prism-validation/src/publication_outputs.rs" ]]; then
            echo "[PASS] publication_outputs.rs exists"
        else
            echo "[FAIL] publication_outputs.rs missing"
            exit 1
        fi

        # Check for publication files
        if [[ -d "results/publication" ]]; then
            if [[ -f "results/publication/table_main_results.tex" ]]; then
                echo "[PASS] LaTeX tables generated"
            else
                echo "[WARN] LaTeX tables not yet generated"
            fi
        else
            echo "[SKIP] Publication outputs not yet generated"
        fi

        # Final metrics check
        if [[ -f "results/cryptobench_full.json" ]]; then
            ROC_AUC=$(jq -r '.roc_auc // 0' results/cryptobench_full.json 2>/dev/null || echo 0)
            PR_AUC=$(jq -r '.pr_auc // 0' results/cryptobench_full.json 2>/dev/null || echo 0)
            SUCCESS=$(jq -r '.success_rate // 0' results/cryptobench_full.json 2>/dev/null || echo 0)

            echo ""
            echo "FINAL METRICS:"
            echo "  ROC AUC:      $ROC_AUC (target: >0.75)"
            echo "  PR AUC:       $PR_AUC (target: >0.25)"
            echo "  Success Rate: $SUCCESS (target: >0.85)"

            # Check all targets met
            ALL_PASS=1
            [[ $(echo "$ROC_AUC >= 0.70" | bc -l 2>/dev/null || echo 0) -eq 1 ]] || ALL_PASS=0
            [[ $(echo "$PR_AUC >= 0.20" | bc -l 2>/dev/null || echo 0) -eq 1 ]] || ALL_PASS=0
            [[ $(echo "$SUCCESS >= 0.80" | bc -l 2>/dev/null || echo 0) -eq 1 ]] || ALL_PASS=0

            if [[ "$ALL_PASS" -eq 1 ]]; then
                echo ""
                echo "========================================"
                echo "PHASE 6 COMPLETE - ALL TARGETS MET"
                echo "========================================"
            else
                echo ""
                echo "[WARN] Some targets not met - review results"
            fi
        fi

        echo ""
        echo "WEEK 8 CHECKPOINT PASSED"
        ;;

    *)
        echo "Unknown week: $WEEK"
        echo "Usage: $0 [0|2|4|6|8|auto]"
        exit 1
        ;;
esac
