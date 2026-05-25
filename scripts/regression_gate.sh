#!/bin/bash
# scripts/regression_gate.sh
set -euo pipefail

DIRECTIVE_NAME="${1:-unknown}"
echo "=== REGRESSION GATE - after $DIRECTIVE_NAME ==="

FAIL=0

echo "[1/7] Rust clippy..."
cargo clippy -p prism-forge -- -D warnings || { echo "FAIL: clippy"; FAIL=1; }

echo "[2/7] Rust tests..."
cargo test -p prism-forge --release || { echo "FAIL: rust tests"; FAIL=1; }

echo "[3/7] Rust release build..."
cargo build --release -p prism-forge --bin oracle_scorer --bin vspace_pruner || { echo "FAIL: build"; FAIL=1; }

echo "[4/7] mypy strict..."
PYTHONPATH=src python3 -m mypy --strict \
    scripts/train_gflownet_policy.py \
    src/prism_dstw/orchestration/rust_reward_oracle.py \
    || { echo "FAIL: mypy"; FAIL=1; }

echo "[5/7] Root pytest collection..."
COLLECTION_ERRORS=$(PYTHONPATH=src python3 -m pytest --collect-only 2>&1 | grep -c "ERROR" || true)
if [ "$COLLECTION_ERRORS" -gt 0 ]; then
    echo "FAIL: $COLLECTION_ERRORS collection errors"
    FAIL=1
fi

echo "[6/7] Root pytest run..."
PYTHONPATH=src python3 -m pytest -q --tb=line || { echo "FAIL: pytest"; FAIL=1; }

echo "[7/7] Default trainer smoke..."
timeout 120 bash -c 'PYTHONPATH=src python3 scripts/train_gflownet_policy.py --epochs 1 --batch-size 4' \
    || { echo "FAIL: default trainer smoke"; FAIL=1; }

if [ "$FAIL" -eq 0 ]; then
    echo "=== REGRESSION GATE PASSED for $DIRECTIVE_NAME ==="
else
    echo "=== REGRESSION GATE FAILED for $DIRECTIVE_NAME - DO NOT COMMIT ==="
    exit 1
fi
