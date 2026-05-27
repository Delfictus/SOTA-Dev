// Lanczos Eigensolver GPU Kernels - WGSL Implementation
// WebGPU Compute Shaders for PRISM-LBS Normal Mode Analysis
//
// Implements core linear algebra operations for Lanczos iteration:
// - Sparse matrix-vector multiplication (SpMV)
// - Vector dot product with parallel reduction
// - AXPY: y = alpha * x + y
// - Vector norm computation
// - Vector scaling
//
// Reference: Golub & Van Loan (2013) Matrix Computations, 4th ed.

const WORKGROUP_SIZE: u32 = 256u;
const WARP_SIZE: u32 = 32u;

// ============================================================================
// SPARSE MATRIX-VECTOR MULTIPLICATION (CSR FORMAT)
// ============================================================================

// CSR sparse matrix storage
@group(0) @binding(0) var<storage, read> csr_values: array<f32>;      // Non-zero values
@group(0) @binding(1) var<storage, read> csr_col_indices: array<u32>; // Column indices
@group(0) @binding(2) var<storage, read> csr_row_ptrs: array<u32>;    // Row pointers
@group(0) @binding(3) var<storage, read> x: array<f32>;               // Input vector
@group(0) @binding(4) var<storage, read_write> y: array<f32>;         // Output vector

struct SpMVParams {
    n_rows: u32,
    n_cols: u32,
    nnz: u32,
    alpha: f32,  // y = alpha * A * x + beta * y
    beta: f32,
}

@group(0) @binding(5) var<uniform> spmv_params: SpMVParams;

// One thread per row - good for matrices with variable row lengths
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn spmv_csr(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let row = global_id.x;

    if (row >= spmv_params.n_rows) {
        return;
    }

    let row_start = csr_row_ptrs[row];
    let row_end = csr_row_ptrs[row + 1u];

    var sum: f32 = 0.0;
    for (var j = row_start; j < row_end; j = j + 1u) {
        let col = csr_col_indices[j];
        let val = csr_values[j];
        sum = sum + val * x[col];
    }

    // y = alpha * A*x + beta * y
    y[row] = spmv_params.alpha * sum + spmv_params.beta * y[row];
}

// ============================================================================
// DENSE MATRIX-VECTOR MULTIPLICATION
// ============================================================================

@group(1) @binding(0) var<storage, read> dense_matrix: array<f32>;  // Row-major n x n
@group(1) @binding(1) var<storage, read> dense_x: array<f32>;       // Input vector
@group(1) @binding(2) var<storage, read_write> dense_y: array<f32>; // Output vector

struct DenseParams {
    n: u32,
    alpha: f32,
    beta: f32,
}

@group(1) @binding(3) var<uniform> dense_params: DenseParams;

// Shared memory for tiled matrix-vector product
var<workgroup> shared_x: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn matvec_dense(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.x;
    let n = dense_params.n;

    if (row >= n) {
        return;
    }

    var sum: f32 = 0.0;

    // Process in tiles for coalesced memory access
    let n_tiles = (n + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    for (var tile: u32 = 0u; tile < n_tiles; tile = tile + 1u) {
        let col_base = tile * WORKGROUP_SIZE;
        let col = col_base + local_id.x;

        // Load x tile into shared memory
        if (col < n) {
            shared_x[local_id.x] = dense_x[col];
        } else {
            shared_x[local_id.x] = 0.0;
        }
        workgroupBarrier();

        // Compute partial sum for this tile
        let tile_end = min(WORKGROUP_SIZE, n - col_base);
        for (var j: u32 = 0u; j < tile_end; j = j + 1u) {
            let mat_idx = row * n + col_base + j;
            sum = sum + dense_matrix[mat_idx] * shared_x[j];
        }
        workgroupBarrier();
    }

    // Write result
    dense_y[row] = dense_params.alpha * sum + dense_params.beta * dense_y[row];
}

// ============================================================================
// VECTOR DOT PRODUCT WITH PARALLEL REDUCTION
// ============================================================================

@group(2) @binding(0) var<storage, read> dot_x: array<f32>;
@group(2) @binding(1) var<storage, read> dot_y: array<f32>;
@group(2) @binding(2) var<storage, read_write> dot_result: array<f32>;  // Partial sums

struct DotParams {
    n: u32,
}

@group(2) @binding(3) var<uniform> dot_params: DotParams;

var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn dot_product(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = dot_params.n;

    // Each thread computes partial sum
    var sum: f32 = 0.0;
    if (gid < n) {
        sum = dot_x[gid] * dot_y[gid];
    }
    shared_sum[tid] = sum;
    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        workgroupBarrier();
    }

    // Write workgroup result to global memory
    if (tid == 0u) {
        dot_result[group_id.x] = shared_sum[0];
    }
}

// Final reduction of partial sums (called with single workgroup)
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn dot_reduce(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let n_groups = (dot_params.n + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Load partial sums
    var sum: f32 = 0.0;
    if (tid < n_groups) {
        sum = dot_result[tid];
    }
    shared_sum[tid] = sum;
    workgroupBarrier();

    // Reduce
    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        workgroupBarrier();
    }

    // Write final result
    if (tid == 0u) {
        dot_result[0] = shared_sum[0];
    }
}

// ============================================================================
// AXPY: y = alpha * x + y
// ============================================================================

@group(3) @binding(0) var<storage, read> axpy_x: array<f32>;
@group(3) @binding(1) var<storage, read_write> axpy_y: array<f32>;

struct AxpyParams {
    n: u32,
    alpha: f32,
}

@group(3) @binding(2) var<uniform> axpy_params: AxpyParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn axpy(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;

    if (i >= axpy_params.n) {
        return;
    }

    axpy_y[i] = axpy_params.alpha * axpy_x[i] + axpy_y[i];
}

// ============================================================================
// VECTOR SCALING: x = alpha * x
// ============================================================================

@group(4) @binding(0) var<storage, read_write> scale_x: array<f32>;

struct ScaleParams {
    n: u32,
    alpha: f32,
}

@group(4) @binding(1) var<uniform> scale_params: ScaleParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn scale(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;

    if (i >= scale_params.n) {
        return;
    }

    scale_x[i] = scale_params.alpha * scale_x[i];
}

// ============================================================================
// VECTOR NORM (uses dot product infrastructure)
// ============================================================================

// Norm is computed as sqrt(dot(x, x)) using the dot product kernel

// ============================================================================
// VECTOR COPY: y = x
// ============================================================================

@group(5) @binding(0) var<storage, read> copy_src: array<f32>;
@group(5) @binding(1) var<storage, read_write> copy_dst: array<f32>;

struct CopyParams {
    n: u32,
}

@group(5) @binding(2) var<uniform> copy_params: CopyParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn copy(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;

    if (i >= copy_params.n) {
        return;
    }

    copy_dst[i] = copy_src[i];
}

// ============================================================================
// ORTHOGONALIZATION: Modified Gram-Schmidt for single vector
// ============================================================================

// Orthogonalize v against columns of V: v = v - sum_i (V_i . v) * V_i
// This requires multiple kernel launches orchestrated from CPU/Rust

@group(6) @binding(0) var<storage, read> orth_V: array<f32>;      // m vectors of length n, column-major
@group(6) @binding(1) var<storage, read_write> orth_v: array<f32>; // Vector to orthogonalize
@group(6) @binding(2) var<storage, read_write> orth_coeffs: array<f32>; // Projection coefficients

struct OrthParams {
    n: u32,           // Vector length
    m: u32,           // Number of columns in V
    col_idx: u32,     // Current column to project against
}

@group(6) @binding(3) var<uniform> orth_params: OrthParams;

// Step 1: Compute coefficient = V[:,col_idx] . v
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn orth_dot(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = orth_params.n;
    let col = orth_params.col_idx;

    var sum: f32 = 0.0;
    if (gid < n) {
        let V_idx = col * n + gid;  // Column-major indexing
        sum = orth_V[V_idx] * orth_v[gid];
    }
    shared_sum[tid] = sum;
    workgroupBarrier();

    for (var s: u32 = WORKGROUP_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        orth_coeffs[group_id.x] = shared_sum[0];
    }
}

// Step 2: v = v - coeff * V[:,col_idx]
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn orth_subtract(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;
    let n = orth_params.n;
    let col = orth_params.col_idx;

    if (i >= n) {
        return;
    }

    // Read coefficient (assumed already reduced)
    let coeff = orth_coeffs[0];
    let V_idx = col * n + i;

    orth_v[i] = orth_v[i] - coeff * orth_V[V_idx];
}
