#!/bin/bash
# Phase 6 Compliance Verification Script
# Run this to verify Claude's implementation follows the plan

set -e

echo "========================================"
echo "PHASE 6 COMPLIANCE VERIFICATION"
echo "========================================"
echo ""

PASS=0
FAIL=0
WARN=0

check_pass() {
    echo "  [PASS] $1"
    ((PASS++))
}

check_fail() {
    echo "  [FAIL] $1"
    ((FAIL++))
}

check_warn() {
    echo "  [WARN] $1"
    ((WARN++))
}

# ========================================
# 1. ZERO FALLBACK VERIFICATION
# ========================================
echo "1. Zero Fallback Verification"
echo "----------------------------------------"

# Check that GPU scorer has no CPU fallback
if grep -r "fallback" crates/prism-validation/src/gpu_zro_cryptic_scorer.rs 2>/dev/null | grep -v "NOT fall back" | grep -v "NO.*fallback" | grep -q .; then
    check_fail "GPU scorer may contain fallback logic"
else
    check_pass "No fallback patterns in GPU scorer"
fi

# Check for bail! on missing GPU
if grep -q "bail!.*CUDA\|bail!.*GPU\|bail!.*device" crates/prism-validation/src/gpu_zro_cryptic_scorer.rs 2>/dev/null; then
    check_pass "GPU scorer fails explicitly on missing GPU"
else
    check_warn "Verify GPU scorer fails on missing device"
fi

# Check for todo!() or unimplemented!()
if grep -rE "todo!\(\)|unimplemented!\(\)" crates/prism-validation/src/*.rs 2>/dev/null | grep -v test | grep -q .; then
    check_fail "Found todo!() or unimplemented!() in production code"
else
    check_pass "No todo!/unimplemented! in production code"
fi

echo ""

# ========================================
# 2. FILE MANIFEST VERIFICATION
# ========================================
echo "2. File Manifest Verification"
echo "----------------------------------------"

WEEK1_FILES=(
    "crates/prism-validation/src/cryptic_features.rs"
    "crates/prism-validation/src/gpu_zro_cryptic_scorer.rs"
)

WEEK3_FILES=(
    "crates/prism-validation/src/pdb_sanitizer.rs"
    "crates/prism-validation/src/nova_cryptic_sampler.rs"
    "crates/prism-validation/src/apo_holo_benchmark.rs"
)

WEEK5_FILES=(
    "crates/prism-validation/src/cryptobench_dataset.rs"
    "crates/prism-validation/src/ablation.rs"
    "crates/prism-validation/src/failure_analysis.rs"
)

WEEK7_FILES=(
    "crates/prism-validation/src/publication_outputs.rs"
)

check_files() {
    local week=$1
    shift
    local files=("$@")

    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            check_pass "$file exists"
        else
            check_fail "$file MISSING (Week $week)"
        fi
    done
}

echo "Week 1-2 Files:"
check_files "1-2" "${WEEK1_FILES[@]}"

echo "Week 3-4 Files:"
check_files "3-4" "${WEEK3_FILES[@]}"

echo "Week 5-6 Files:"
check_files "5-6" "${WEEK5_FILES[@]}"

echo "Week 7-8 Files:"
check_files "7-8" "${WEEK7_FILES[@]}"

echo ""

# ========================================
# 3. CODE QUALITY VERIFICATION
# ========================================
echo "3. Code Quality Verification"
echo "----------------------------------------"

# Check for module documentation
for file in crates/prism-validation/src/cryptic_features.rs \
            crates/prism-validation/src/gpu_zro_cryptic_scorer.rs \
            crates/prism-validation/src/cryptobench_dataset.rs; do
    if [[ -f "$file" ]]; then
        if head -5 "$file" | grep -q "^//!"; then
            check_pass "$file has module documentation"
        else
            check_fail "$file missing module documentation"
        fi
    fi
done

# Check for proper error handling (no unwrap in main code paths)
UNWRAP_COUNT=$(grep -r "\.unwrap()" crates/prism-validation/src/*.rs 2>/dev/null | grep -v test | grep -v "\.ok().unwrap_or" | wc -l || echo 0)
if [[ "$UNWRAP_COUNT" -gt 10 ]]; then
    check_warn "Found $UNWRAP_COUNT unwrap() calls - review for safety"
else
    check_pass "Reasonable unwrap() usage ($UNWRAP_COUNT calls)"
fi

echo ""

# ========================================
# 4. TEST VERIFICATION
# ========================================
echo "4. Test Verification"
echo "----------------------------------------"

# Check test file exists
if [[ -f "crates/prism-validation/src/tests/gpu_scorer_tests.rs" ]]; then
    check_pass "GPU scorer tests exist"

    # Check for zero fallback test
    if grep -q "test_no_cpu_fallback\|no_cpu_fallback" crates/prism-validation/src/tests/gpu_scorer_tests.rs; then
        check_pass "Zero fallback test exists"
    else
        check_fail "Missing zero fallback test"
    fi

    # Check for RLS stability test
    if grep -q "rls_stability\|stability.*1000\|stability.*10000" crates/prism-validation/src/tests/gpu_scorer_tests.rs; then
        check_pass "RLS stability test exists"
    else
        check_fail "Missing RLS stability test"
    fi
else
    check_fail "GPU scorer tests file missing"
fi

echo ""

# ========================================
# 5. FEATURE VECTOR VERIFICATION
# ========================================
echo "5. Feature Vector Verification"
echo "----------------------------------------"

if [[ -f "crates/prism-validation/src/cryptic_features.rs" ]]; then
    # Check for 16-dim feature vector
    FEATURE_COUNT=$(grep -E "pub [a-z_]+: f32" crates/prism-validation/src/cryptic_features.rs | wc -l)
    if [[ "$FEATURE_COUNT" -eq 16 ]]; then
        check_pass "Feature vector has 16 dimensions"
    else
        check_fail "Feature vector has $FEATURE_COUNT dimensions (expected 16)"
    fi

    # Check for velocity encoding
    if grep -q "encode_with_velocity" crates/prism-validation/src/cryptic_features.rs; then
        check_pass "Velocity encoding implemented"
    else
        check_fail "Missing velocity encoding"
    fi
fi

echo ""

# ========================================
# 6. ARCHITECTURAL COMPLIANCE
# ========================================
echo "6. Architectural Compliance"
echo "----------------------------------------"

# Check for forbidden dependencies
if grep -rE "torch|tensorflow|pytorch|onnx" Cargo.toml crates/*/Cargo.toml 2>/dev/null | grep -v "#" | grep -q .; then
    check_fail "Found forbidden ML framework dependencies"
else
    check_pass "No forbidden ML framework dependencies"
fi

# Check reservoir size
if grep -q "512" crates/prism-validation/src/gpu_zro_cryptic_scorer.rs 2>/dev/null; then
    check_pass "512-neuron reservoir specified"
else
    check_warn "Verify 512-neuron reservoir size"
fi

# Check RLS lambda
if grep -q "0.99\|lambda.*0\.99" crates/prism-validation/src/gpu_zro_cryptic_scorer.rs 2>/dev/null; then
    check_pass "RLS lambda=0.99 specified"
else
    check_warn "Verify RLS lambda value"
fi

echo ""

# ========================================
# 7. DATA INTEGRITY
# ========================================
echo "7. Data Integrity Verification"
echo "----------------------------------------"

if [[ -d "data/benchmarks/cryptobench" ]]; then
    STRUCTURE_COUNT=$(find data/benchmarks/cryptobench -name "*.pdb" 2>/dev/null | wc -l)
    if [[ "$STRUCTURE_COUNT" -ge 1000 ]]; then
        check_pass "CryptoBench dataset present ($STRUCTURE_COUNT structures)"
    else
        check_warn "CryptoBench incomplete ($STRUCTURE_COUNT structures, expected ~1107)"
    fi
else
    check_warn "CryptoBench dataset not yet downloaded"
fi

if [[ -d "data/benchmarks/apo_holo" ]]; then
    APO_HOLO_COUNT=$(find data/benchmarks/apo_holo -name "*.pdb" 2>/dev/null | wc -l)
    if [[ "$APO_HOLO_COUNT" -ge 28 ]]; then
        check_pass "Apo-holo pairs present ($APO_HOLO_COUNT files)"
    else
        check_warn "Apo-holo incomplete ($APO_HOLO_COUNT files, expected 30)"
    fi
else
    check_warn "Apo-holo pairs not yet downloaded"
fi

echo ""

# ========================================
# 8. PARALLEL IMPLEMENTATION VERIFICATION
# ========================================
echo "8. Parallel Implementation Verification"
echo "----------------------------------------"

# Check contract.rs exists (THE LAW)
if [[ -f "crates/prism-validation/src/sampling/contract.rs" ]]; then
    check_pass "Contract file exists (THE LAW)"

    # Check SamplingBackend trait defined
    if grep -q "trait SamplingBackend" crates/prism-validation/src/sampling/contract.rs; then
        check_pass "SamplingBackend trait defined"
    else
        check_fail "SamplingBackend trait missing from contract.rs"
    fi
else
    check_warn "Contract file not yet created (Week 3)"
fi

# Check NOVA path exists (Greenfield)
if [[ -f "crates/prism-validation/src/sampling/paths/nova_path.rs" ]]; then
    check_pass "NOVA path exists (Greenfield)"

    # CRITICAL: Check NOVA doesn't import from AMBER
    if grep -q "amber_path\|AmberPath\|AmberBackend" crates/prism-validation/src/sampling/paths/nova_path.rs 2>/dev/null; then
        check_fail "ISOLATION VIOLATION: nova_path.rs imports from amber_path"
    else
        check_pass "NOVA path isolation verified"
    fi

    # Check NOVA implements SamplingBackend
    if grep -q "impl SamplingBackend for" crates/prism-validation/src/sampling/paths/nova_path.rs; then
        check_pass "NOVA implements SamplingBackend"
    else
        check_fail "NOVA does not implement SamplingBackend trait"
    fi
else
    check_warn "NOVA path not yet created (Week 3)"
fi

# Check AMBER path exists (Stable)
if [[ -f "crates/prism-validation/src/sampling/paths/amber_path.rs" ]]; then
    check_pass "AMBER path exists (Stable)"

    # CRITICAL: Check AMBER doesn't import from NOVA
    if grep -q "nova_path\|NovaPath\|NovaBackend" crates/prism-validation/src/sampling/paths/amber_path.rs 2>/dev/null; then
        check_fail "ISOLATION VIOLATION: amber_path.rs imports from nova_path"
    else
        check_pass "AMBER path isolation verified"
    fi

    # Check AMBER implements SamplingBackend
    if grep -q "impl SamplingBackend for" crates/prism-validation/src/sampling/paths/amber_path.rs; then
        check_pass "AMBER implements SamplingBackend"
    else
        check_fail "AMBER does not implement SamplingBackend trait"
    fi
else
    check_warn "AMBER path not yet created (Week 3)"
fi

# Check shadow comparator exists
if [[ -f "crates/prism-validation/src/sampling/shadow/comparator.rs" ]]; then
    check_pass "Shadow comparator exists"

    # Check DivergenceMetrics defined
    if grep -q "struct DivergenceMetrics\|DivergenceMetrics" crates/prism-validation/src/sampling/shadow/comparator.rs; then
        check_pass "DivergenceMetrics defined"
    else
        check_warn "DivergenceMetrics struct not found"
    fi
else
    check_warn "Shadow comparator not yet created (Week 4)"
fi

# Check migration feature flags exist
if [[ -f "crates/prism-validation/src/sampling/migration/feature_flags.rs" ]]; then
    check_pass "Migration feature flags exist"

    # Check MigrationStage enum defined
    if grep -q "enum MigrationStage\|MigrationStage" crates/prism-validation/src/sampling/migration/feature_flags.rs; then
        check_pass "MigrationStage enum defined"
    else
        check_warn "MigrationStage enum not found"
    fi
else
    check_warn "Migration feature flags not yet created (Week 4)"
fi

# Check router exists
if [[ -f "crates/prism-validation/src/sampling/router/mod.rs" ]]; then
    check_pass "Sampling router exists"
else
    check_warn "Sampling router not yet created (Week 3)"
fi

echo ""

# ========================================
# SUMMARY
# ========================================
echo "========================================"
echo "COMPLIANCE SUMMARY"
echo "========================================"
echo "  PASSED: $PASS"
echo "  FAILED: $FAIL"
echo "  WARNINGS: $WARN"
echo ""

if [[ "$FAIL" -gt 0 ]]; then
    echo "STATUS: COMPLIANCE VIOLATIONS DETECTED"
    echo "Review failed checks and fix before proceeding."
    exit 1
elif [[ "$WARN" -gt 3 ]]; then
    echo "STATUS: REVIEW NEEDED"
    echo "Multiple warnings detected - manual review recommended."
    exit 0
else
    echo "STATUS: COMPLIANT"
    echo "Implementation follows Phase 6 plan specifications."
    exit 0
fi
