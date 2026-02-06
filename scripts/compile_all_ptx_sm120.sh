#!/bin/bash
# Compile all CUDA kernels to sm_120 PTX for Blackwell (RTX 5080)

# Don't exit on error - we want to compile as many as possible

KERNEL_DIR="/home/diddy/Desktop/Prism4D-bio/crates/prism-gpu/src/kernels"
OUTPUT_DIR="/home/diddy/Desktop/Prism4D-bio/crates/prism-gpu/src/kernels"
NVCC_FLAGS="-ptx -arch=sm_120 -O3 --use_fast_math -Wno-deprecated-gpu-targets"

echo "═══════════════════════════════════════════════════════════════"
echo "  COMPILING ALL CUDA KERNELS FOR SM_120 (BLACKWELL)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

SUCCESS=0
FAILED=0
SKIPPED=0
FAILED_LIST=""

compile_kernel() {
    local cu_file="$1"
    local base_name=$(basename "$cu_file" .cu)
    local dir_name=$(dirname "$cu_file")
    local ptx_file="${dir_name}/${base_name}.ptx"

    echo -n "  Compiling ${base_name}.cu... "

    if nvcc $NVCC_FLAGS -o "$ptx_file" "$cu_file" 2>/dev/null; then
        echo "✓"
        SUCCESS=$((SUCCESS + 1))
    else
        # Try with relaxed flags
        if nvcc -ptx -arch=sm_120 -O2 -o "$ptx_file" "$cu_file" 2>/dev/null; then
            echo "✓ (relaxed)"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "✗ FAILED"
            FAILED=$((FAILED + 1))
            FAILED_LIST="$FAILED_LIST $base_name"
        fi
    fi
}

# Main kernels directory
echo "Main kernels:"
for cu_file in "$KERNEL_DIR"/*.cu; do
    if [[ -f "$cu_file" ]]; then
        compile_kernel "$cu_file"
    fi
done

# Cryptic subdirectory
echo ""
echo "Cryptic kernels:"
for cu_file in "$KERNEL_DIR"/cryptic/*.cu; do
    if [[ -f "$cu_file" ]]; then
        compile_kernel "$cu_file"
    fi
done

# Allosteric subdirectory
echo ""
echo "Allosteric kernels:"
for cu_file in "$KERNEL_DIR"/allosteric/*.cu; do
    if [[ -f "$cu_file" ]]; then
        compile_kernel "$cu_file"
    fi
done

# LBS subdirectory
echo ""
echo "LBS kernels:"
for cu_file in "$KERNEL_DIR"/lbs/*.cu; do
    if [[ -f "$cu_file" ]]; then
        compile_kernel "$cu_file"
    fi
done

# Also compile holographic_langevin from different location
echo ""
echo "Other kernels:"
OTHER_KERNEL="/home/diddy/Desktop/Prism4D-bio/crates/prism-gpu/kernels/holographic_langevin.cu"
if [[ -f "$OTHER_KERNEL" ]]; then
    compile_kernel "$OTHER_KERNEL"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  COMPILATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo "  Success: $SUCCESS"
echo "  Failed:  $FAILED"
if [[ -n "$FAILED_LIST" ]]; then
    echo "  Failed kernels:$FAILED_LIST"
fi
echo "═══════════════════════════════════════════════════════════════"
