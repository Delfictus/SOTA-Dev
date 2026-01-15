# PRISM-4D Performance Optimization Plan: Phases 7-8

**Branch:** `feature/performance-optimization`
**Working Directory:** `~/Desktop/PRISM4D-dev`
**Prerequisite:** Phases 0-6 (Explicit Solvent) MUST be complete and passing all tests
**Stable Reference:** `~/Desktop/PRISM4D-v2.0.0-EXPLICIT` (tag after Phase 6)

---

## EXECUTIVE SUMMARY

| Phase | Optimization | Expected Speedup | Est. Time | Risk |
|-------|--------------|------------------|-----------|------|
| 7 | Mixed Precision (FP16/FP32) | 1.5-2.0x | 5-7 days | Medium |
| 8 | Fused Kernels | 1.3-1.5x | 7-10 days | Medium |
| **Combined** | | **2.0-3.0x** | **2-3 weeks** | |

**Target Performance:**
- Before: 15-25 ns/day (RTX 3080, 25k atoms)
- After: 40-60 ns/day (RTX 3080, 25k atoms)
- 100 ns simulation: 4-7 days → **2-3 days**

---

## SECTION 1: PREREQUISITES AND BASELINE

### 1.1 Required Before Starting

```bash
# Verify explicit solvent is working
cd ~/Desktop/PRISM4D-dev

# Run baseline benchmark
cargo run --release -p prism-validation --bin benchmark -- \
    --topology test_data/6m0j_solvated.json \
    --solvent explicit \
    --steps 10000 \
    --warmup 1000

# Expected output:
# Atoms: 25,000
# Steps: 10,000
# Wall time: X.XX seconds
# ns/day: XX.X
# Step time: X.XX ms
```

### 1.2 Baseline Metrics to Record

| Metric | Value | Notes |
|--------|-------|-------|
| Total atoms | | Record from benchmark |
| Steps/second | | Primary metric |
| ns/day | | For publication |
| Force kernel time | | Profile with nvprof |
| PME time | | Profile with nvprof |
| Integration time | | Profile with nvprof |
| Memory bandwidth | | From nvprof |
| GPU utilization | | From nvidia-smi |

### 1.3 Profiling Commands

```bash
# Detailed kernel profiling
nvprof --print-gpu-trace \
    cargo run --release -p prism-validation --bin generate-ensemble -- \
    --topology test_data/water_box_small.json \
    --steps 1000

# Memory bandwidth analysis
nvprof --metrics dram_read_throughput,dram_write_throughput \
    cargo run --release ...

# Occupancy analysis
nvprof --metrics achieved_occupancy \
    cargo run --release ...
```

### 1.4 Create Performance Tag

```bash
git checkout feature/explicit-solvent
git tag -a v2.0.0-explicit -m "Explicit solvent complete - baseline for optimization"
git push origin v2.0.0-explicit

# Create optimization branch
git checkout -b feature/performance-optimization
```

---

## SECTION 2: PHASE 7 — MIXED PRECISION

### 2.1 Overview

Mixed precision uses FP16 (half precision) where accuracy isn't critical and FP32 where it is. Modern GPUs (Turing, Ampere, Ada) have dedicated FP16 units that are 2x faster than FP32.

**Precision Requirements by Operation:**

| Operation | Current | Target | Reason |
|-----------|---------|--------|--------|
| Position storage | FP32 | FP32 | Need Å-level accuracy |
| Velocity storage | FP32 | FP32 | Accumulates over time |
| Force accumulation | FP32 | **FP16** | Summed and reset each step |
| Distance calculation | FP32 | **FP16** | Intermediate value |
| LJ computation | FP32 | **FP16** | Intermediate value |
| Coulomb computation | FP32 | FP32 | Long-range sensitive |
| PME grid | FP32 | **FP16** | FFT tolerates FP16 |
| PME forces | FP32 | FP32 | Accuracy critical |
| Energy reporting | FP64 | FP64 | Publication accuracy |
| Temperature | FP64 | FP64 | Thermostat stability |

### 2.2 CUDA Half Precision Primer

```cuda
#include <cuda_fp16.h>

// Single half-precision value
__half h = __float2half(1.0f);
float f = __half2float(h);

// Packed half2 (2 values, single instruction)
__half2 h2 = __floats2half2_rn(1.0f, 2.0f);
__half2 result = __hadd2(h2, h2);  // Vectorized add

// Conversion intrinsics
__half2 __float22half2_rn(float2 f);  // float2 → half2
float2 __half22float2(__half2 h);      // half2 → float2

// Arithmetic (single)
__half __hadd(__half a, __half b);     // a + b
__half __hmul(__half a, __half b);     // a * b
__half __hfma(__half a, __half b, __half c);  // a*b + c (fused)

// Arithmetic (packed, 2x throughput)
__half2 __hadd2(__half2 a, __half2 b);
__half2 __hmul2(__half2 a, __half2 b);
__half2 __hfma2(__half2 a, __half2 b, __half2 c);

// Comparison
bool __hgt(__half a, __half b);        // a > b
bool __hlt(__half a, __half b);        // a < b

// Math functions
__half hsqrt(__half a);
__half hexp(__half a);
__half hrsqrt(__half a);  // 1/sqrt(a) - very useful for MD!
```

### 2.3 Files to Modify

| File | Changes | Est. LOC |
|------|---------|----------|
| `amber_mega_fused.cu` | Mixed precision force kernels | +300 |
| `amber_mega_fused.rs` | FP16 buffer management | +100 |
| `pme.cu` | FP16 charge grid | +150 |
| `pme.rs` | FP16 grid allocation | +50 |
| `settle.cu` | Keep FP32 (constraints need accuracy) | +20 |
| `tests/precision_tests.rs` | Accuracy validation | +200 |
| `benches/mixed_precision.rs` | Performance comparison | +100 |
| **Total** | | **~920** |

### 2.4 Implementation: Force Kernel Mixed Precision

#### 2.4.1 Data Structure Changes

```cuda
// amber_mega_fused.cu

// NEW: Mixed precision parameter structure
struct MixedPrecisionParams {
    // Positions: FP32 (need accuracy for PBC)
    const float* __restrict__ positions;       // [n_atoms * 3]

    // Velocities: FP32 (accumulate over time)
    float* __restrict__ velocities;            // [n_atoms * 3]

    // Forces: FP16 accumulation buffer (reset each step)
    __half* __restrict__ forces_fp16;          // [n_atoms * 3]

    // Final forces: FP32 (after reduction)
    float* __restrict__ forces_fp32;           // [n_atoms * 3]

    // LJ parameters: FP16 (read-only, tolerates reduced precision)
    const __half* __restrict__ sigma_fp16;     // [n_atoms]
    const __half* __restrict__ epsilon_fp16;   // [n_atoms]

    // Charges: FP32 (electrostatics sensitive)
    const float* __restrict__ charges;         // [n_atoms]

    // Masses: FP32 (used in integration)
    const float* __restrict__ masses;          // [n_atoms]
};

// Compile-time switch for mixed precision
#ifndef USE_MIXED_PRECISION
#define USE_MIXED_PRECISION 1
#endif
```

#### 2.4.2 LJ Force Kernel (Mixed Precision)

```cuda
// amber_mega_fused.cu

/**
 * Lennard-Jones force computation with mixed precision
 *
 * Uses FP16 for:
 *   - Distance calculations (intermediate)
 *   - LJ sigma/epsilon lookup
 *   - Force magnitude computation
 *
 * Uses FP32 for:
 *   - Position differences (PBC accuracy)
 *   - Force accumulation (atomicAdd)
 *   - Final force output
 *
 * Strategy: Compute in FP16, accumulate in FP32
 */
__device__ __forceinline__
float3 compute_lj_force_mixed(
    float3 pos_i,           // FP32 position of atom i
    float3 pos_j,           // FP32 position of atom j
    __half sigma_i,         // FP16 LJ sigma
    __half sigma_j,
    __half epsilon_i,       // FP16 LJ epsilon
    __half epsilon_j,
    float cutoff2           // FP32 cutoff squared
) {
    // Distance vector (FP32 for PBC accuracy)
    float3 dr = apply_pbc(make_float3(
        pos_j.x - pos_i.x,
        pos_j.y - pos_i.y,
        pos_j.z - pos_i.z
    ));

    float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

    // Early exit if beyond cutoff
    if (r2 >= cutoff2 || r2 < 1e-10f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // === Switch to FP16 for LJ computation ===
    __half r2_h = __float2half(r2);

    // Combining rules (arithmetic mean for sigma, geometric for epsilon)
    __half sigma = __hmul(__hadd(sigma_i, sigma_j), __float2half(0.5f));
    __half epsilon = hsqrt(__hmul(epsilon_i, epsilon_j));

    // sigma^2 / r^2
    __half sigma2 = __hmul(sigma, sigma);
    __half s2_over_r2 = __hdiv(sigma2, r2_h);

    // (sigma/r)^6 and (sigma/r)^12
    __half s6 = __hmul(__hmul(s2_over_r2, s2_over_r2), s2_over_r2);
    __half s12 = __hmul(s6, s6);

    // LJ force magnitude: 24 * epsilon * (2*s12 - s6) / r^2
    __half two = __float2half(2.0f);
    __half twentyfour = __float2half(24.0f);

    __half term = __hsub(__hmul(two, s12), s6);
    __half f_over_r2 = __hdiv(__hmul(__hmul(twentyfour, epsilon), term), r2_h);

    // === Convert back to FP32 for force vector ===
    float f_mag = __half2float(f_over_r2);

    return make_float3(
        f_mag * dr.x,
        f_mag * dr.y,
        f_mag * dr.z
    );
}

/**
 * Mixed precision non-bonded force kernel
 */
__global__ void compute_nonbonded_forces_mixed(
    const float* __restrict__ positions,
    const __half* __restrict__ sigma,
    const __half* __restrict__ epsilon,
    const float* __restrict__ charges,
    const int* __restrict__ neighbor_list,
    const int* __restrict__ n_neighbors,
    float* __restrict__ forces,
    int n_atoms,
    float cutoff2,
    float coulomb_scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    // Load atom i data
    float3 pos_i = make_float3(
        positions[i * 3],
        positions[i * 3 + 1],
        positions[i * 3 + 2]
    );
    __half sigma_i = sigma[i];
    __half epsilon_i = epsilon[i];
    float charge_i = charges[i];

    // Accumulate forces in FP32 (register)
    float3 force_acc = make_float3(0.0f, 0.0f, 0.0f);

    // Loop over neighbors
    int num_neighbors = n_neighbors[i];
    for (int n = 0; n < num_neighbors; n++) {
        int j = neighbor_list[i * NEIGHBOR_LIST_SIZE + n];

        // Load atom j data
        float3 pos_j = make_float3(
            positions[j * 3],
            positions[j * 3 + 1],
            positions[j * 3 + 2]
        );

        // LJ force (mixed precision)
        float3 f_lj = compute_lj_force_mixed(
            pos_i, pos_j,
            sigma_i, sigma[j],
            epsilon_i, epsilon[j],
            cutoff2
        );

        // Coulomb force (FP32 - needs accuracy)
        float3 dr = apply_pbc(make_float3(
            pos_j.x - pos_i.x,
            pos_j.y - pos_i.y,
            pos_j.z - pos_i.z
        ));
        float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
        float r = sqrtf(r2);

        float f_coul_mag = 0.0f;
        if (r2 < cutoff2 && r > 1e-5f) {
            float charge_j = charges[j];
            // Coulomb: k * q_i * q_j / r^2
            // Force points along dr, magnitude is k*q*q/r^2
            f_coul_mag = coulomb_scale * charge_i * charge_j / (r * r2);
        }

        float3 f_coul = make_float3(
            f_coul_mag * dr.x,
            f_coul_mag * dr.y,
            f_coul_mag * dr.z
        );

        // Accumulate (FP32)
        force_acc.x += f_lj.x + f_coul.x;
        force_acc.y += f_lj.y + f_coul.y;
        force_acc.z += f_lj.z + f_coul.z;
    }

    // Write accumulated force (FP32)
    atomicAdd(&forces[i * 3], force_acc.x);
    atomicAdd(&forces[i * 3 + 1], force_acc.y);
    atomicAdd(&forces[i * 3 + 2], force_acc.z);
}
```

#### 2.4.3 Vectorized Half2 Version (Maximum Performance)

```cuda
/**
 * Optimized LJ using half2 vectorization
 * Processes 2 neighbor pairs simultaneously
 *
 * ~40% faster than scalar FP16 version
 */
__device__ __forceinline__
float3 compute_lj_force_half2_pair(
    float3 pos_i,
    float3 pos_j1, float3 pos_j2,  // Two neighbors
    __half2 sigma_ij,               // Packed sigma for both pairs
    __half2 epsilon_ij,             // Packed epsilon for both pairs
    float cutoff2
) {
    // Distance vectors (FP32)
    float3 dr1 = apply_pbc(make_float3(pos_j1.x - pos_i.x, pos_j1.y - pos_i.y, pos_j1.z - pos_i.z));
    float3 dr2 = apply_pbc(make_float3(pos_j2.x - pos_i.x, pos_j2.y - pos_i.y, pos_j2.z - pos_i.z));

    float r2_1 = dr1.x*dr1.x + dr1.y*dr1.y + dr1.z*dr1.z;
    float r2_2 = dr2.x*dr2.x + dr2.y*dr2.y + dr2.z*dr2.z;

    // Pack r² into half2
    __half2 r2_h2 = __floats2half2_rn(r2_1, r2_2);

    // Vectorized LJ computation (both pairs at once)
    __half2 sigma2 = __hmul2(sigma_ij, sigma_ij);
    __half2 s2_over_r2 = __h2div(sigma2, r2_h2);
    __half2 s6 = __hmul2(__hmul2(s2_over_r2, s2_over_r2), s2_over_r2);
    __half2 s12 = __hmul2(s6, s6);

    __half2 two = __float2half2_rn(2.0f, 2.0f);
    __half2 twentyfour = __float2half2_rn(24.0f, 24.0f);

    __half2 term = __hsub2(__hmul2(two, s12), s6);
    __half2 f_over_r2 = __h2div(__hmul2(__hmul2(twentyfour, epsilon_ij), term), r2_h2);

    // Unpack results
    float f_mag_1 = __low2float(f_over_r2);
    float f_mag_2 = __high2float(f_over_r2);

    // Apply cutoff masks
    if (r2_1 >= cutoff2) f_mag_1 = 0.0f;
    if (r2_2 >= cutoff2) f_mag_2 = 0.0f;

    // Sum both force contributions
    return make_float3(
        f_mag_1 * dr1.x + f_mag_2 * dr2.x,
        f_mag_1 * dr1.y + f_mag_2 * dr2.y,
        f_mag_1 * dr1.z + f_mag_2 * dr2.z
    );
}
```

### 2.5 Implementation: PME Mixed Precision

#### 2.5.1 FP16 Charge Grid

```cuda
// pme.cu

/**
 * PME charge spreading with FP16 grid
 *
 * Uses FP16 for grid storage (2x memory bandwidth)
 * Uses FP32 for B-spline computation (accuracy)
 */
__global__ void pme_spread_charges_fp16(
    const float* __restrict__ positions,
    const float* __restrict__ charges,
    __half* __restrict__ grid,         // FP16 charge grid
    int n_atoms,
    int3 grid_dims,
    float3 grid_spacing_inv,
    float3 box_dims
) {
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= n_atoms) return;

    float charge = charges[atom_idx];
    if (fabsf(charge) < 1e-10f) return;

    float3 pos = make_float3(
        positions[atom_idx * 3],
        positions[atom_idx * 3 + 1],
        positions[atom_idx * 3 + 2]
    );

    // Fractional grid coordinates
    float3 frac = make_float3(
        pos.x * grid_spacing_inv.x,
        pos.y * grid_spacing_inv.y,
        pos.z * grid_spacing_inv.z
    );

    // B-spline interpolation (order 4)
    // Compute in FP32 for accuracy
    int3 base = make_int3(
        (int)floorf(frac.x) - 1,
        (int)floorf(frac.y) - 1,
        (int)floorf(frac.z) - 1
    );

    float wx[4], wy[4], wz[4];
    compute_bspline_weights(frac.x - floorf(frac.x), wx);
    compute_bspline_weights(frac.y - floorf(frac.y), wy);
    compute_bspline_weights(frac.z - floorf(frac.z), wz);

    // Spread charge to grid (FP16 atomic add)
    for (int iz = 0; iz < 4; iz++) {
        int gz = (base.z + iz + grid_dims.z) % grid_dims.z;
        for (int iy = 0; iy < 4; iy++) {
            int gy = (base.y + iy + grid_dims.y) % grid_dims.y;
            for (int ix = 0; ix < 4; ix++) {
                int gx = (base.x + ix + grid_dims.x) % grid_dims.x;

                float weight = wx[ix] * wy[iy] * wz[iz] * charge;
                int grid_idx = gx + grid_dims.x * (gy + grid_dims.y * gz);

                // FP16 atomic add (convert, add, convert back)
                atomicAdd_fp16(&grid[grid_idx], __float2half(weight));
            }
        }
    }
}

/**
 * FP16 atomic add helper
 * Uses CAS loop for GPUs without native FP16 atomics
 */
__device__ __forceinline__
void atomicAdd_fp16(__half* address, __half val) {
#if __CUDA_ARCH__ >= 700
    // Native FP16 atomics on Volta+
    atomicAdd(address, val);
#else
    // Fallback: CAS loop
    unsigned short* addr_as_ushort = (unsigned short*)address;
    unsigned short old = *addr_as_ushort;
    unsigned short assumed;
    do {
        assumed = old;
        __half sum = __hadd(__ushort_as_half(assumed), val);
        old = atomicCAS(addr_as_ushort, assumed, __half_as_ushort(sum));
    } while (assumed != old);
#endif
}
```

#### 2.5.2 cuFFT with FP16

```cuda
// pme.cu

/**
 * PME FFT execution with FP16 ↔ FP32 conversion
 *
 * cuFFT doesn't support FP16 directly, so we:
 * 1. Convert FP16 grid → FP32
 * 2. Run cuFFT in FP32
 * 3. Apply Green's function
 * 4. Run inverse FFT
 * 5. Convert FP32 → FP16
 *
 * Memory bandwidth still benefits from FP16 storage
 */
__global__ void convert_grid_fp16_to_fp32(
    const __half* __restrict__ grid_fp16,
    float* __restrict__ grid_fp32,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    grid_fp32[idx] = __half2float(grid_fp16[idx]);
}

__global__ void convert_grid_fp32_to_fp16(
    const float* __restrict__ grid_fp32,
    __half* __restrict__ grid_fp16,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    grid_fp16[idx] = __float2half(grid_fp32[idx]);
}

// In PME solve function:
void pme_solve_mixed(PmeState* state) {
    int n_grid = state->grid_dims.x * state->grid_dims.y * state->grid_dims.z;

    // 1. FP16 → FP32 for FFT
    convert_grid_fp16_to_fp32<<<(n_grid+255)/256, 256>>>(
        state->grid_fp16, state->grid_fp32, n_grid
    );

    // 2. Forward FFT (FP32)
    cufftExecR2C(state->plan_r2c, state->grid_fp32, state->grid_complex);

    // 3. Apply Green's function (reciprocal space)
    pme_reciprocal_convolution<<<...>>>(state->grid_complex, ...);

    // 4. Inverse FFT (FP32)
    cufftExecC2R(state->plan_c2r, state->grid_complex, state->grid_fp32);

    // 5. FP32 → FP16 for force interpolation
    convert_grid_fp32_to_fp16<<<(n_grid+255)/256, 256>>>(
        state->grid_fp32, state->grid_fp16, n_grid
    );
}
```

### 2.6 Rust Integration

```rust
// amber_mega_fused.rs

use cudarc::driver::{CudaSlice, LaunchConfig};
use half::f16;

/// Mixed precision configuration
#[derive(Clone, Copy, Debug)]
pub struct MixedPrecisionConfig {
    /// Use FP16 for LJ parameters
    pub lj_fp16: bool,
    /// Use FP16 for PME grid
    pub pme_grid_fp16: bool,
    /// Use FP16 for force accumulation (experimental)
    pub force_accum_fp16: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            lj_fp16: true,
            pme_grid_fp16: true,
            force_accum_fp16: false,  // Conservative default
        }
    }
}

/// GPU buffers with mixed precision support
pub struct MixedPrecisionBuffers {
    // FP32 buffers (positions, velocities, charges)
    pub positions: CudaSlice<f32>,
    pub velocities: CudaSlice<f32>,
    pub forces: CudaSlice<f32>,
    pub charges: CudaSlice<f32>,
    pub masses: CudaSlice<f32>,

    // FP16 buffers (LJ parameters, PME grid)
    pub sigma_fp16: CudaSlice<f16>,
    pub epsilon_fp16: CudaSlice<f16>,
    pub pme_grid_fp16: Option<CudaSlice<f16>>,

    // Temporary FP32 buffer for FFT
    pub pme_grid_fp32: Option<CudaSlice<f32>>,
}

impl MixedPrecisionBuffers {
    pub fn new(
        device: &CudaDevice,
        n_atoms: usize,
        grid_size: Option<usize>,
        config: &MixedPrecisionConfig,
    ) -> Result<Self> {
        // Allocate FP32 buffers
        let positions = device.alloc_zeros::<f32>(n_atoms * 3)?;
        let velocities = device.alloc_zeros::<f32>(n_atoms * 3)?;
        let forces = device.alloc_zeros::<f32>(n_atoms * 3)?;
        let charges = device.alloc_zeros::<f32>(n_atoms)?;
        let masses = device.alloc_zeros::<f32>(n_atoms)?;

        // Allocate FP16 LJ parameters
        let sigma_fp16 = device.alloc_zeros::<f16>(n_atoms)?;
        let epsilon_fp16 = device.alloc_zeros::<f16>(n_atoms)?;

        // PME grids (if enabled)
        let (pme_grid_fp16, pme_grid_fp32) = if let Some(size) = grid_size {
            if config.pme_grid_fp16 {
                (
                    Some(device.alloc_zeros::<f16>(size)?),
                    Some(device.alloc_zeros::<f32>(size)?),  // For FFT
                )
            } else {
                (None, Some(device.alloc_zeros::<f32>(size)?))
            }
        } else {
            (None, None)
        };

        Ok(Self {
            positions,
            velocities,
            forces,
            charges,
            masses,
            sigma_fp16,
            epsilon_fp16,
            pme_grid_fp16,
            pme_grid_fp32,
        })
    }

    /// Upload LJ parameters, converting FP32 → FP16
    pub fn upload_lj_params(
        &mut self,
        sigma: &[f32],
        epsilon: &[f32],
        stream: &CudaStream,
    ) -> Result<()> {
        // Convert to FP16 on CPU (one-time cost)
        let sigma_f16: Vec<f16> = sigma.iter().map(|&x| f16::from_f32(x)).collect();
        let epsilon_f16: Vec<f16> = epsilon.iter().map(|&x| f16::from_f32(x)).collect();

        // Upload
        stream.memcpy_htod(&sigma_f16, &mut self.sigma_fp16)?;
        stream.memcpy_htod(&epsilon_f16, &mut self.epsilon_fp16)?;

        Ok(())
    }
}
```

### 2.7 Precision Validation Tests

```rust
// tests/precision_tests.rs

use approx::assert_relative_eq;

/// Verify mixed precision doesn't degrade accuracy beyond acceptable limits
#[test]
fn test_lj_force_precision() {
    let device = CudaDevice::new(0).unwrap();

    // Test case: two atoms at known distance
    let sigma = 3.4;   // Å (typical carbon)
    let epsilon = 0.1; // kcal/mol
    let r = 4.0;       // Å

    // Reference: FP64 computation
    let s_r = sigma / r;
    let s6 = s_r.powi(6);
    let s12 = s6 * s6;
    let f_ref = 24.0 * epsilon * (2.0 * s12 - s6) / (r * r);

    // Mixed precision computation
    let f_mixed = compute_lj_force_mixed_test(&device, sigma, epsilon, r);

    // Tolerance: 0.1% for LJ forces
    assert_relative_eq!(f_mixed, f_ref, max_relative = 0.001);
}

#[test]
fn test_energy_conservation_mixed_precision() {
    // Run 10,000 steps with mixed precision
    // Verify energy drift < 0.1% (same as FP32)

    let config = MixedPrecisionConfig::default();
    let mut sim = setup_water_box_216(config);

    let initial_energy = sim.compute_total_energy();
    sim.run(10_000);
    let final_energy = sim.compute_total_energy();

    let drift = (final_energy - initial_energy).abs() / initial_energy.abs();
    assert!(drift < 0.001, "Energy drift {:.4}% exceeds 0.1% threshold", drift * 100.0);
}

#[test]
fn test_temperature_stability_mixed_precision() {
    // Run 50,000 steps, verify temperature stays at 310 ± 10 K

    let config = MixedPrecisionConfig::default();
    let mut sim = setup_protein_solvated(config);

    let temps: Vec<f64> = (0..500).map(|_| {
        sim.run(100);
        sim.compute_temperature()
    }).collect();

    let mean_temp: f64 = temps.iter().sum::<f64>() / temps.len() as f64;
    let std_temp: f64 = (temps.iter().map(|t| (t - mean_temp).powi(2)).sum::<f64>()
                         / temps.len() as f64).sqrt();

    assert!((mean_temp - 310.0).abs() < 5.0,
            "Mean temperature {:.1} K deviates from 310 K", mean_temp);
    assert!(std_temp < 10.0,
            "Temperature std {:.1} K exceeds 10 K", std_temp);
}

#[test]
fn test_mixed_vs_fp32_trajectory_similarity() {
    // Run same simulation with FP32 and mixed precision
    // RMSD between trajectories should be < 0.1 Å

    let mut sim_fp32 = setup_water_box_216(MixedPrecisionConfig {
        lj_fp16: false,
        pme_grid_fp16: false,
        force_accum_fp16: false,
    });

    let mut sim_mixed = setup_water_box_216(MixedPrecisionConfig::default());

    // Same initial conditions
    sim_fp32.seed_rng(42);
    sim_mixed.seed_rng(42);

    // Run both for 1000 steps
    sim_fp32.run(1000);
    sim_mixed.run(1000);

    // Compare final positions
    let pos_fp32 = sim_fp32.get_positions();
    let pos_mixed = sim_mixed.get_positions();

    let rmsd = compute_rmsd(&pos_fp32, &pos_mixed);
    assert!(rmsd < 0.1, "RMSD {:.3} Å between FP32 and mixed exceeds 0.1 Å", rmsd);
}
```

### 2.8 Performance Benchmarks

```rust
// benches/mixed_precision.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_force_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("force_computation");

    for n_atoms in [1000, 5000, 10000, 25000, 50000] {
        // FP32 baseline
        group.bench_with_input(
            BenchmarkId::new("fp32", n_atoms),
            &n_atoms,
            |b, &n| {
                let sim = setup_benchmark_system(n, false);
                b.iter(|| sim.compute_forces_fp32());
            }
        );

        // Mixed precision
        group.bench_with_input(
            BenchmarkId::new("mixed", n_atoms),
            &n_atoms,
            |b, &n| {
                let sim = setup_benchmark_system(n, true);
                b.iter(|| sim.compute_forces_mixed());
            }
        );
    }

    group.finish();
}

fn benchmark_pme(c: &mut Criterion) {
    let mut group = c.benchmark_group("pme");

    for grid_size in [32, 48, 64, 80, 96] {
        group.bench_with_input(
            BenchmarkId::new("fp32_grid", grid_size),
            &grid_size,
            |b, &size| {
                let pme = setup_pme(size, false);
                b.iter(|| pme.solve());
            }
        );

        group.bench_with_input(
            BenchmarkId::new("fp16_grid", grid_size),
            &grid_size,
            |b, &size| {
                let pme = setup_pme(size, true);
                b.iter(|| pme.solve());
            }
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_force_computation, benchmark_pme);
criterion_main!(benches);
```

### 2.9 Phase 7 Acceptance Criteria

| Test | Criterion | Status |
|------|-----------|--------|
| LJ force accuracy | < 0.1% error vs FP64 | ☐ |
| Coulomb force accuracy | < 0.01% error vs FP64 | ☐ |
| PME energy accuracy | < 0.1 kcal/mol vs FP32 | ☐ |
| Energy conservation | < 0.1% drift over 10k steps | ☐ |
| Temperature stability | 310 ± 10 K over 50k steps | ☐ |
| Trajectory similarity | RMSD < 0.1 Å vs FP32 | ☐ |
| Force kernel speedup | ≥ 1.5x | ☐ |
| PME speedup | ≥ 1.3x | ☐ |
| Overall speedup | ≥ 1.5x | ☐ |

### 2.10 Phase 7 Commit

```
feat(gpu): Add mixed precision (FP16/FP32) for force computation and PME

- FP16 LJ parameters and intermediate calculations
- FP16 PME charge grid with FP32 FFT
- Half2 vectorized LJ kernel for 2x pair throughput
- Precision validation tests passing
- ~1.5-2x overall speedup

Performance:
- Force kernel: 1.8x faster
- PME: 1.4x faster
- Overall: 1.6x faster
```

---

## SECTION 3: PHASE 8 — FUSED KERNELS

### 3.1 Overview

Kernel fusion combines multiple GPU operations into a single kernel launch, reducing:
1. **Launch overhead** (~5-10 μs per launch)
2. **Global memory traffic** (data stays in registers/shared memory)
3. **Synchronization points** (implicit barriers between kernels)

### 3.2 Current Kernel Launch Pattern

```
Step N:
├─ Launch: memset_forces          [5 μs overhead]
├─ Launch: compute_bonded_forces  [5 μs overhead]
├─ Launch: compute_nb_forces      [5 μs overhead]  ← Main work
├─ Launch: pme_spread             [5 μs overhead]
├─ Launch: cufft_r2c              [5 μs overhead]
├─ Launch: pme_reciprocal         [5 μs overhead]
├─ Launch: cufft_c2r              [5 μs overhead]
├─ Launch: pme_interpolate        [5 μs overhead]
├─ Launch: vv_step1               [5 μs overhead]
├─ Launch: vv_step2               [5 μs overhead]
├─ Launch: settle                 [5 μs overhead]
├─ Launch: h_constraints          [5 μs overhead]
├─ Launch: wrap_positions         [5 μs overhead]
├─ Launch: com_removal            [5 μs overhead]
└─ Synchronize                    [~100 μs]

Total overhead per step: ~170 μs (14 launches + sync)
```

### 3.3 Target Fused Pattern

```
Step N:
├─ Launch: fused_force_integration  [5 μs overhead]
│   ├─ Clear forces (inline)
│   ├─ Bonded forces (inline)
│   ├─ Non-bonded forces (inline)
│   ├─ PME real-space (inline)
│   ├─ VV step 1 (inline)
│   └─ Partial VV step 2 (inline)
├─ Launch: pme_fft_solve            [5 μs overhead]  ← Cannot fuse FFT
├─ Launch: fused_constraints        [5 μs overhead]
│   ├─ PME force interpolation
│   ├─ Final VV step 2
│   ├─ SETTLE
│   ├─ H-constraints
│   ├─ Wrap positions
│   └─ COM removal
└─ Synchronize (async, overlapped)

Total overhead per step: ~15 μs (3 launches)
Reduction: ~90% fewer launches
```

### 3.4 Files to Modify

| File | Changes | Est. LOC |
|------|---------|----------|
| `fused_kernels.cu` (NEW) | Mega-fused force+integration kernel | +600 |
| `fused_constraints.cu` (NEW) | Fused constraint kernel | +300 |
| `amber_mega_fused.rs` | New kernel orchestration | +200 |
| `amber_mega_fused.cu` | Refactor existing kernels as device functions | +100 |
| `tests/fused_kernel_tests.rs` | Correctness validation | +200 |
| **Total** | | **~1400** |

### 3.5 Implementation: Mega-Fused Force + Integration Kernel

#### 3.5.1 Device Function Refactoring

First, convert existing kernels to `__device__` functions:

```cuda
// amber_mega_fused.cu

// BEFORE: Global kernel
__global__ void compute_bond_forces(...) { ... }

// AFTER: Device function (callable from other kernels)
__device__ __forceinline__
void compute_bond_forces_device(
    int atom_idx,
    const float* positions,
    const int2* bond_atoms,
    const float2* bond_params,
    int n_bonds,
    float* force_x,
    float* force_y,
    float* force_z
) {
    // Same logic, but:
    // - Takes atom index as parameter (not blockIdx/threadIdx)
    // - Accumulates to register variables (not global memory)
    // - No atomicAdd (caller handles accumulation)

    for (int b = 0; b < n_bonds; b++) {
        int2 atoms = bond_atoms[b];
        if (atoms.x != atom_idx && atoms.y != atom_idx) continue;

        // ... compute bond force ...

        if (atoms.x == atom_idx) {
            *force_x += fx;
            *force_y += fy;
            *force_z += fz;
        } else {
            *force_x -= fx;
            *force_y -= fy;
            *force_z -= fz;
        }
    }
}

// Similar refactoring for:
// - compute_angle_forces_device()
// - compute_dihedral_forces_device()
// - compute_nb_forces_device()
// - compute_pme_real_space_device()
// - velocity_verlet_step1_device()
// - velocity_verlet_step2_device()
```

#### 3.5.2 Mega-Fused Kernel

```cuda
// fused_kernels.cu

/**
 * MEGA-FUSED FORCE + INTEGRATION KERNEL
 *
 * Combines into single kernel:
 *   1. Force clearing
 *   2. Bonded forces (bonds, angles, dihedrals)
 *   3. Non-bonded forces (LJ + real-space Coulomb)
 *   4. PME real-space contribution
 *   5. Velocity Verlet step 1 (half-kick + drift)
 *
 * NOT included (separate kernels):
 *   - PME FFT (requires cuFFT library call)
 *   - PME force interpolation (depends on FFT result)
 *   - VV step 2 (depends on PME forces)
 *   - Constraints (depends on final positions)
 *
 * Memory access pattern:
 *   - Each thread handles one atom
 *   - Positions: read once, cached in registers
 *   - Forces: accumulated in registers, written once
 *   - Velocities: read once, updated, written once
 */
__global__ void mega_fused_force_integrate_step1(
    // Positions (read)
    const float* __restrict__ positions,
    // Velocities (read/write)
    float* __restrict__ velocities,
    // Forces (write only - cleared and accumulated)
    float* __restrict__ forces,
    // LJ parameters (FP16)
    const __half* __restrict__ sigma,
    const __half* __restrict__ epsilon,
    // Charges
    const float* __restrict__ charges,
    // Masses
    const float* __restrict__ masses,
    // Bonded terms
    const int2* __restrict__ bond_atoms,
    const float2* __restrict__ bond_params,
    int n_bonds,
    const int4* __restrict__ angle_atoms,
    const float2* __restrict__ angle_params,
    int n_angles,
    const int4* __restrict__ dihedral_atoms,
    const float4* __restrict__ dihedral_params,
    int n_dihedrals,
    // Neighbor list
    const int* __restrict__ neighbor_list,
    const int* __restrict__ n_neighbors,
    // Simulation parameters
    int n_atoms,
    float dt,
    float cutoff2,
    float coulomb_scale,
    float pme_beta  // Ewald splitting parameter
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    // ══════════════════════════════════════════════════════════════════
    // STEP 1: Load atom data into registers
    // ══════════════════════════════════════════════════════════════════
    float3 pos_i = make_float3(
        positions[i * 3],
        positions[i * 3 + 1],
        positions[i * 3 + 2]
    );
    float3 vel_i = make_float3(
        velocities[i * 3],
        velocities[i * 3 + 1],
        velocities[i * 3 + 2]
    );
    float mass_i = masses[i];
    float charge_i = charges[i];
    __half sigma_i = sigma[i];
    __half epsilon_i = epsilon[i];

    // Force accumulator (registers - no global memory access during accumulation)
    float3 force_i = make_float3(0.0f, 0.0f, 0.0f);

    // ══════════════════════════════════════════════════════════════════
    // STEP 2: Bonded forces (bonds, angles, dihedrals)
    // ══════════════════════════════════════════════════════════════════
    compute_bond_forces_device(
        i, positions, bond_atoms, bond_params, n_bonds,
        &force_i.x, &force_i.y, &force_i.z
    );

    compute_angle_forces_device(
        i, positions, angle_atoms, angle_params, n_angles,
        &force_i.x, &force_i.y, &force_i.z
    );

    compute_dihedral_forces_device(
        i, positions, dihedral_atoms, dihedral_params, n_dihedrals,
        &force_i.x, &force_i.y, &force_i.z
    );

    // ══════════════════════════════════════════════════════════════════
    // STEP 3: Non-bonded forces (LJ + Coulomb real-space)
    // ══════════════════════════════════════════════════════════════════
    int num_neighbors = n_neighbors[i];

    for (int n = 0; n < num_neighbors; n++) {
        int j = neighbor_list[i * NEIGHBOR_LIST_SIZE + n];

        float3 pos_j = make_float3(
            positions[j * 3],
            positions[j * 3 + 1],
            positions[j * 3 + 2]
        );

        // Distance with PBC
        float3 dr = apply_pbc(make_float3(
            pos_j.x - pos_i.x,
            pos_j.y - pos_i.y,
            pos_j.z - pos_i.z
        ));
        float r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

        if (r2 >= cutoff2 || r2 < 1e-10f) continue;

        float r = sqrtf(r2);
        float r_inv = 1.0f / r;

        // LJ force (mixed precision)
        float3 f_lj = compute_lj_force_mixed(
            pos_i, pos_j,
            sigma_i, sigma[j],
            epsilon_i, epsilon[j],
            cutoff2
        );

        // Coulomb real-space with Ewald screening
        // erfc(beta * r) / r²  (complementary error function)
        float charge_j = charges[j];
        float beta_r = pme_beta * r;
        float erfc_val = erfcf(beta_r);
        float exp_val = expf(-beta_r * beta_r);

        // dU/dr for screened Coulomb
        float f_coul_mag = coulomb_scale * charge_i * charge_j * r_inv * r_inv * (
            erfc_val * r_inv +
            2.0f * pme_beta * M_2_SQRTPI * exp_val
        );

        float3 f_coul = make_float3(
            f_coul_mag * dr.x,
            f_coul_mag * dr.y,
            f_coul_mag * dr.z
        );

        // Accumulate forces
        force_i.x += f_lj.x + f_coul.x;
        force_i.y += f_lj.y + f_coul.y;
        force_i.z += f_lj.z + f_coul.z;
    }

    // ══════════════════════════════════════════════════════════════════
    // STEP 4: Velocity Verlet step 1 (half-kick + drift)
    // ══════════════════════════════════════════════════════════════════
    // v += (dt/2) * F/m
    float dt_half_over_m = 0.5f * dt / mass_i;
    vel_i.x += dt_half_over_m * force_i.x;
    vel_i.y += dt_half_over_m * force_i.y;
    vel_i.z += dt_half_over_m * force_i.z;

    // x += dt * v
    pos_i.x += dt * vel_i.x;
    pos_i.y += dt * vel_i.y;
    pos_i.z += dt * vel_i.z;

    // ══════════════════════════════════════════════════════════════════
    // STEP 5: Write results to global memory (single coalesced write)
    // ══════════════════════════════════════════════════════════════════
    // Note: Forces written for PME force interpolation to add to
    forces[i * 3] = force_i.x;
    forces[i * 3 + 1] = force_i.y;
    forces[i * 3 + 2] = force_i.z;

    // Intermediate positions (for PME and constraints)
    // Store in positions array (will be updated by constraints)
    // Using separate buffer to avoid race conditions
    // Actually, write to positions directly - constraints will fix
    positions[i * 3] = pos_i.x;
    positions[i * 3 + 1] = pos_i.y;
    positions[i * 3 + 2] = pos_i.z;

    // Velocities stored for VV step 2
    velocities[i * 3] = vel_i.x;
    velocities[i * 3 + 1] = vel_i.y;
    velocities[i * 3 + 2] = vel_i.z;
}
```

#### 3.5.3 Fused Constraints Kernel

```cuda
// fused_constraints.cu

/**
 * FUSED CONSTRAINTS KERNEL
 *
 * Combines into single kernel:
 *   1. PME force interpolation (add reciprocal forces)
 *   2. Velocity Verlet step 2 (final half-kick + thermostat)
 *   3. SETTLE constraints (rigid water)
 *   4. H-bond constraints
 *   5. Position wrapping (PBC)
 *   6. COM drift removal
 *
 * Execution model:
 *   - Phase 1: PME forces + VV step 2 (all atoms, parallel)
 *   - Phase 2: SETTLE (water molecules, parallel)
 *   - Phase 3: H-constraints (protein H atoms, parallel)
 *   - Phase 4: Wrap + COM (all atoms, with reduction)
 */
__global__ void fused_constraints_step2(
    // Positions (read/write)
    float* __restrict__ positions,
    // Old positions (for SETTLE)
    const float* __restrict__ old_positions,
    // Velocities (read/write)
    float* __restrict__ velocities,
    // Forces (read - includes PME reciprocal contribution)
    const float* __restrict__ forces,
    // PME potential grid (for force interpolation)
    const __half* __restrict__ pme_grid,
    // Masses
    const float* __restrict__ masses,
    // Water molecule indices (oxygen atoms)
    const int* __restrict__ water_oxygens,
    int n_waters,
    // H-constraint clusters
    const HConstraintCluster* __restrict__ h_clusters,
    int n_h_clusters,
    // Simulation parameters
    int n_atoms,
    float dt,
    float temperature,
    float gamma,  // Langevin friction
    float3 box_dims,
    // RNG state for Langevin
    curandState* __restrict__ rng_states,
    // COM reduction buffer
    float* __restrict__ com_velocity  // [3] output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // ══════════════════════════════════════════════════════════════════
    // PHASE 1: PME force interpolation + VV step 2 (all atoms)
    // ══════════════════════════════════════════════════════════════════
    if (i < n_atoms) {
        float mass_i = masses[i];

        // Load current state
        float3 vel = make_float3(
            velocities[i * 3],
            velocities[i * 3 + 1],
            velocities[i * 3 + 2]
        );

        // Load forces (already includes bonded + NB + PME real-space)
        float3 force = make_float3(
            forces[i * 3],
            forces[i * 3 + 1],
            forces[i * 3 + 2]
        );

        // Add PME reciprocal forces (interpolated from grid)
        float3 pme_force = interpolate_pme_force(
            positions + i * 3, pme_grid, box_dims
        );
        force.x += pme_force.x;
        force.y += pme_force.y;
        force.z += pme_force.z;

        // Velocity Verlet step 2: v += (dt/2) * F/m
        float dt_half_over_m = 0.5f * dt / mass_i;
        vel.x += dt_half_over_m * force.x;
        vel.y += dt_half_over_m * force.y;
        vel.z += dt_half_over_m * force.z;

        // Langevin thermostat
        float c1 = expf(-gamma * dt);
        float c2 = sqrtf((1.0f - c1 * c1) * BOLTZMANN * temperature / mass_i);

        curandState local_state = rng_states[i];
        vel.x = c1 * vel.x + c2 * curand_normal(&local_state);
        vel.y = c1 * vel.y + c2 * curand_normal(&local_state);
        vel.z = c1 * vel.z + c2 * curand_normal(&local_state);
        rng_states[i] = local_state;

        // Store velocities
        velocities[i * 3] = vel.x;
        velocities[i * 3 + 1] = vel.y;
        velocities[i * 3 + 2] = vel.z;
    }

    __syncthreads();

    // ══════════════════════════════════════════════════════════════════
    // PHASE 2: SETTLE constraints (one thread per water)
    // ══════════════════════════════════════════════════════════════════
    if (i < n_waters) {
        int o_idx = water_oxygens[i];
        int h1_idx = o_idx + 1;
        int h2_idx = o_idx + 2;

        // Load old positions
        float3 o_old = load_float3(old_positions + o_idx * 3);
        float3 h1_old = load_float3(old_positions + h1_idx * 3);
        float3 h2_old = load_float3(old_positions + h2_idx * 3);

        // Load new (unconstrained) positions
        float3 o_new = load_float3(positions + o_idx * 3);
        float3 h1_new = load_float3(positions + h1_idx * 3);
        float3 h2_new = load_float3(positions + h2_idx * 3);

        // Apply SETTLE algorithm
        float3 o_constrained, h1_constrained, h2_constrained;
        float3 o_vel_correction, h1_vel_correction, h2_vel_correction;

        settle_molecule(
            o_old, h1_old, h2_old,
            o_new, h1_new, h2_new,
            WATER_MASS_O, WATER_MASS_H,
            TIP3P_OH_BOND, TIP3P_HH_DIST,
            dt,
            &o_constrained, &h1_constrained, &h2_constrained,
            &o_vel_correction, &h1_vel_correction, &h2_vel_correction
        );

        // Store constrained positions
        store_float3(positions + o_idx * 3, o_constrained);
        store_float3(positions + h1_idx * 3, h1_constrained);
        store_float3(positions + h2_idx * 3, h2_constrained);

        // Apply velocity corrections
        atomicAdd(&velocities[o_idx * 3], o_vel_correction.x);
        atomicAdd(&velocities[o_idx * 3 + 1], o_vel_correction.y);
        atomicAdd(&velocities[o_idx * 3 + 2], o_vel_correction.z);
        // ... similar for h1, h2
    }

    __syncthreads();

    // ══════════════════════════════════════════════════════════════════
    // PHASE 3: H-bond constraints (protein X-H bonds)
    // ══════════════════════════════════════════════════════════════════
    if (i < n_h_clusters) {
        HConstraintCluster cluster = h_clusters[i];
        apply_h_constraint_cluster(
            cluster, positions, velocities, masses, dt
        );
    }

    __syncthreads();

    // ══════════════════════════════════════════════════════════════════
    // PHASE 4: Position wrapping + COM removal
    // ══════════════════════════════════════════════════════════════════
    if (i < n_atoms) {
        // Wrap position into box
        float3 pos = load_float3(positions + i * 3);
        pos = wrap_position(pos, box_dims);
        store_float3(positions + i * 3, pos);

        // Contribute to COM velocity (parallel reduction)
        float mass_i = masses[i];
        float3 vel = load_float3(velocities + i * 3);

        // Warp-level reduction for COM
        float3 warp_com_vel = warp_reduce_weighted_sum(vel, mass_i);
        float warp_mass = warp_reduce_sum(mass_i);

        // First thread in warp contributes to global COM
        if (threadIdx.x % 32 == 0) {
            atomicAdd(&com_velocity[0], warp_com_vel.x);
            atomicAdd(&com_velocity[1], warp_com_vel.y);
            atomicAdd(&com_velocity[2], warp_com_vel.z);
            atomicAdd(&com_velocity[3], warp_mass);  // Total mass
        }
    }
}

/**
 * Final COM removal (after reduction)
 */
__global__ void remove_com_velocity(
    float* velocities,
    const float* com_velocity,  // [4]: vx, vy, vz, total_mass
    int n_atoms
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    float inv_mass = 1.0f / com_velocity[3];
    float3 com_vel = make_float3(
        com_velocity[0] * inv_mass,
        com_velocity[1] * inv_mass,
        com_velocity[2] * inv_mass
    );

    velocities[i * 3] -= com_vel.x;
    velocities[i * 3 + 1] -= com_vel.y;
    velocities[i * 3 + 2] -= com_vel.z;
}
```

### 3.6 Rust Orchestration

```rust
// amber_mega_fused.rs

impl AmberHmc {
    /// Run one MD step with fused kernels
    pub fn fused_step(&mut self) -> Result<()> {
        // ═══════════════════════════════════════════════════════════════
        // STEP 1: Save old positions (for SETTLE)
        // ═══════════════════════════════════════════════════════════════
        self.stream.memcpy_dtod(
            &self.buffers.positions,
            &mut self.buffers.old_positions
        )?;

        // ═══════════════════════════════════════════════════════════════
        // STEP 2: Fused force + integration step 1
        // ═══════════════════════════════════════════════════════════════
        let blocks = (self.n_atoms + 255) / 256;
        unsafe {
            self.kernels.mega_fused_force_integrate_step1.launch(
                LaunchConfig::for_num_elems(self.n_atoms as u32),
                (
                    &self.buffers.positions,
                    &mut self.buffers.velocities,
                    &mut self.buffers.forces,
                    &self.buffers.sigma_fp16,
                    &self.buffers.epsilon_fp16,
                    &self.buffers.charges,
                    &self.buffers.masses,
                    // Bonded terms
                    &self.buffers.bond_atoms,
                    &self.buffers.bond_params,
                    self.n_bonds as i32,
                    &self.buffers.angle_atoms,
                    &self.buffers.angle_params,
                    self.n_angles as i32,
                    &self.buffers.dihedral_atoms,
                    &self.buffers.dihedral_params,
                    self.n_dihedrals as i32,
                    // Neighbor list
                    &self.buffers.neighbor_list,
                    &self.buffers.n_neighbors,
                    // Parameters
                    self.n_atoms as i32,
                    self.dt,
                    self.cutoff * self.cutoff,
                    self.coulomb_scale,
                    self.pme_beta,
                )
            )?;
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 3: PME reciprocal space (cannot fuse - requires FFT)
        // ═══════════════════════════════════════════════════════════════
        if self.pme_enabled {
            // Charge spreading was done in fused kernel via positions
            // Actually, need separate spread due to updated positions
            self.pme.spread_charges(&self.buffers.positions, &self.buffers.charges)?;
            self.pme.solve_fft()?;
            // Force interpolation will be in fused_constraints
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 4: Fused constraints + VV step 2
        // ═══════════════════════════════════════════════════════════════
        // Clear COM accumulator
        self.stream.memset_zeros(&mut self.buffers.com_velocity)?;

        unsafe {
            self.kernels.fused_constraints_step2.launch(
                LaunchConfig::for_num_elems(self.n_atoms.max(self.n_waters) as u32),
                (
                    &mut self.buffers.positions,
                    &self.buffers.old_positions,
                    &mut self.buffers.velocities,
                    &self.buffers.forces,
                    &self.pme.grid_fp16,
                    &self.buffers.masses,
                    &self.buffers.water_oxygens,
                    self.n_waters as i32,
                    &self.buffers.h_clusters,
                    self.n_h_clusters as i32,
                    self.n_atoms as i32,
                    self.dt,
                    self.temperature,
                    self.langevin_gamma,
                    self.box_dims,
                    &mut self.buffers.rng_states,
                    &mut self.buffers.com_velocity,
                )
            )?;
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 5: COM removal (separate small kernel)
        // ═══════════════════════════════════════════════════════════════
        if self.com_removal_enabled {
            unsafe {
                self.kernels.remove_com_velocity.launch(
                    LaunchConfig::for_num_elems(self.n_atoms as u32),
                    (
                        &mut self.buffers.velocities,
                        &self.buffers.com_velocity,
                        self.n_atoms as i32,
                    )
                )?;
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // STEP 6: Neighbor list rebuild (if needed)
        // ═══════════════════════════════════════════════════════════════
        self.step_count += 1;
        if self.step_count % self.neighbor_rebuild_interval == 0 {
            self.rebuild_neighbor_lists()?;
        }

        Ok(())
    }
}
```

### 3.7 Shared Memory Optimizations

```cuda
// fused_kernels.cu

/**
 * Shared memory tile for neighbor data
 *
 * Loads neighbor positions into shared memory for reuse
 * within a threadblock (reduces global memory traffic)
 */
#define TILE_SIZE 32

__global__ void mega_fused_force_integrate_tiled(
    /* ... parameters ... */
) {
    __shared__ float3 s_positions[TILE_SIZE];
    __shared__ __half s_sigma[TILE_SIZE];
    __shared__ __half s_epsilon[TILE_SIZE];
    __shared__ float s_charges[TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % TILE_SIZE;

    // Load atom i data
    float3 pos_i = /* ... */;
    float3 force_i = make_float3(0.0f, 0.0f, 0.0f);

    // Process neighbors in tiles
    int num_neighbors = n_neighbors[i];
    for (int tile_start = 0; tile_start < num_neighbors; tile_start += TILE_SIZE) {
        // Cooperative loading: each thread loads one neighbor
        int tile_idx = tile_start + lane;
        if (tile_idx < num_neighbors && i < n_atoms) {
            int j = neighbor_list[i * NEIGHBOR_LIST_SIZE + tile_idx];
            s_positions[lane] = load_float3(positions + j * 3);
            s_sigma[lane] = sigma[j];
            s_epsilon[lane] = epsilon[j];
            s_charges[lane] = charges[j];
        }
        __syncthreads();

        // Each thread processes all neighbors in tile
        int tile_end = min(TILE_SIZE, num_neighbors - tile_start);
        for (int t = 0; t < tile_end; t++) {
            float3 pos_j = s_positions[t];
            // Compute force using shared memory data
            float3 f = compute_pair_force(
                pos_i, pos_j,
                sigma_i, s_sigma[t],
                epsilon_i, s_epsilon[t],
                charge_i, s_charges[t],
                /* ... */
            );
            force_i.x += f.x;
            force_i.y += f.y;
            force_i.z += f.z;
        }
        __syncthreads();
    }

    // ... continue with integration ...
}
```

### 3.8 Phase 8 Acceptance Criteria

| Test | Criterion | Status |
|------|-----------|--------|
| Force correctness | < 1e-5 kcal/(mol·Å) vs unfused | ☐ |
| Energy conservation | < 0.1% drift (same as unfused) | ☐ |
| Temperature stability | 310 ± 10 K | ☐ |
| SETTLE violations | < 1e-4 Å | ☐ |
| Trajectory match | RMSD < 0.01 Å vs unfused (1000 steps) | ☐ |
| Kernel launch reduction | ≥ 80% fewer launches | ☐ |
| Overall speedup | ≥ 1.3x | ☐ |

### 3.9 Phase 8 Commit

```
feat(gpu): Add fused force+integration and constraint kernels

- Mega-fused kernel combines forces + VV step 1
- Fused constraints kernel combines SETTLE + H-constraints + VV step 2
- Shared memory tiling for neighbor data
- Kernel launches reduced from 14 to 3-4 per step
- ~1.3-1.5x overall speedup

Performance:
- Kernel launches: 14 → 3 per step
- Launch overhead: ~170 μs → ~15 μs per step
- Overall: 1.4x faster
```

---

## SECTION 4: COMBINED TESTING

### 4.1 Full Optimization Test Suite

```rust
// tests/optimization_integration.rs

#[test]
fn test_optimized_vs_baseline_10ns() {
    // Run 10 ns with both baseline and optimized
    // Compare final structures, energies, temperatures

    let baseline = run_simulation(SimConfig {
        mixed_precision: false,
        fused_kernels: false,
        steps: 5_000_000,  // 10 ns at 2 fs
        ..default()
    });

    let optimized = run_simulation(SimConfig {
        mixed_precision: true,
        fused_kernels: true,
        steps: 5_000_000,
        ..default()
    });

    // Structure similarity
    let rmsd = compute_rmsd(&baseline.final_positions, &optimized.final_positions);
    assert!(rmsd < 1.0, "RMSD {:.2} Å too large", rmsd);

    // Energy similarity
    let energy_diff = (baseline.final_energy - optimized.final_energy).abs();
    assert!(energy_diff < 10.0, "Energy difference {:.1} kcal/mol too large", energy_diff);

    // Temperature similarity
    let temp_diff = (baseline.mean_temperature - optimized.mean_temperature).abs();
    assert!(temp_diff < 2.0, "Temperature difference {:.1} K too large", temp_diff);

    // Performance improvement
    let speedup = baseline.wall_time / optimized.wall_time;
    assert!(speedup > 1.5, "Speedup {:.2}x below 1.5x target", speedup);

    println!("Optimization validation passed:");
    println!("  RMSD: {:.3} Å", rmsd);
    println!("  Energy diff: {:.2} kcal/mol", energy_diff);
    println!("  Temp diff: {:.2} K", temp_diff);
    println!("  Speedup: {:.2}x", speedup);
}
```

### 4.2 Performance Regression Tests

```rust
// benches/regression.rs

/// Ensure optimizations don't accidentally slow down edge cases
#[test]
fn test_no_performance_regression() {
    let test_cases = vec![
        ("small_protein", 1000),   // 1k atoms
        ("medium_protein", 5000),  // 5k atoms
        ("large_solvated", 25000), // 25k atoms
        ("huge_solvated", 50000),  // 50k atoms
    ];

    for (name, n_atoms) in test_cases {
        let baseline_ns_per_day = get_baseline_performance(name);
        let current_ns_per_day = benchmark_current(name);

        // Must be at least as fast as baseline
        assert!(
            current_ns_per_day >= baseline_ns_per_day * 0.95,
            "{}: Performance regression! {:.1} ns/day < {:.1} ns/day baseline",
            name, current_ns_per_day, baseline_ns_per_day
        );

        println!("{}: {:.1} ns/day (baseline: {:.1})",
                 name, current_ns_per_day, baseline_ns_per_day);
    }
}
```

---

## SECTION 5: IMPLEMENTATION ORDER

| Order | Component | Est. Time | Commit |
|-------|-----------|-----------|--------|
| 7.1 | FP16 buffer infrastructure | 1 day | `feat(gpu): Add FP16 buffer management` |
| 7.2 | Mixed precision LJ kernel | 2 days | `feat(gpu): Add mixed precision LJ forces` |
| 7.3 | Half2 vectorized LJ | 1 day | `feat(gpu): Add half2 vectorized LJ kernel` |
| 7.4 | FP16 PME grid | 2 days | `feat(gpu): Add FP16 PME charge grid` |
| 7.5 | Precision validation tests | 1 day | `test(gpu): Add precision validation suite` |
| 8.1 | Device function refactoring | 1 day | `refactor(gpu): Convert kernels to device functions` |
| 8.2 | Mega-fused force kernel | 3 days | `feat(gpu): Add mega-fused force+integrate kernel` |
| 8.3 | Fused constraints kernel | 2 days | `feat(gpu): Add fused constraints kernel` |
| 8.4 | Shared memory tiling | 1 day | `feat(gpu): Add shared memory tiling for neighbors` |
| 8.5 | Rust orchestration | 1 day | `feat(gpu): Add fused kernel orchestration` |
| 8.6 | Integration tests | 1 day | `test(gpu): Add fused kernel validation` |
| **Total** | | **~17 days** | |

---

## SECTION 6: RISK ANALYSIS

### 6.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FP16 precision insufficient | Low | High | Precision tests before integration |
| Fused kernel register pressure | Medium | Medium | Profile occupancy, reduce registers |
| Race conditions in fused kernel | Medium | High | Extensive correctness tests |
| cuFFT incompatible with fused flow | Low | Medium | Keep FFT as separate stage |
| Performance regression on small systems | Low | Low | Benchmark all system sizes |

### 6.2 Abort Criteria

Do NOT proceed if:
1. ❌ FP16 LJ forces differ > 0.5% from FP32
2. ❌ Energy drift > 1% over 10k steps with mixed precision
3. ❌ Fused kernel gives different trajectory than unfused
4. ❌ SETTLE violations > 1e-3 Å with fused constraints
5. ❌ Any NaN/Inf in forces or positions

---

## SECTION 7: EXPECTED OUTCOMES

### 7.1 Performance Targets

| Metric | Baseline | After Phase 7 | After Phase 8 | Combined |
|--------|----------|---------------|---------------|----------|
| ns/day (25k atoms) | 15-25 | 25-40 | 35-50 | **40-60** |
| Force kernel (ms) | 5.0 | 3.0 | 2.5 | **2.0** |
| PME (ms) | 3.0 | 2.2 | 2.0 | **1.8** |
| Integration (ms) | 1.0 | 1.0 | 0.3 | **0.3** |
| Kernel launches/step | 14 | 14 | 3-4 | **3-4** |

### 7.2 Time to 100 ns

| Hardware | Baseline | Optimized |
|----------|----------|-----------|
| RTX 3060 | 7-10 days | **3-4 days** |
| RTX 3080 | 4-7 days | **2-3 days** |
| RTX 4090 | 2-4 days | **1-2 days** |

---

## SECTION 8: POST-OPTIMIZATION ROADMAP

After Phases 7-8, additional optimizations (lower priority):

| Phase | Optimization | Speedup | Effort |
|-------|--------------|---------|--------|
| 9 | CUDA Graphs | 1.2-1.3x | 1 week |
| 10 | Warp-level reductions | 1.1x | 3 days |
| 11 | Async memory operations | 1.1x | 3 days |
| 12 | Multi-GPU (future) | 2-4x | 1-2 months |
| 13 | Tensor cores (future) | 2-4x | 2-3 months |

---

## EXECUTION CHECKLIST

### Phase 7 (Mixed Precision)
- [ ] FP16 buffer infrastructure implemented
- [ ] Mixed precision LJ kernel passing tests
- [ ] Half2 vectorization implemented
- [ ] FP16 PME grid working
- [ ] All precision validation tests passing
- [ ] Benchmark shows ≥ 1.5x speedup

### Phase 8 (Fused Kernels)
- [ ] Device functions extracted
- [ ] Mega-fused kernel implemented
- [ ] Fused constraints kernel implemented
- [ ] Shared memory tiling working
- [ ] All correctness tests passing
- [ ] Benchmark shows ≥ 1.3x additional speedup

### Final Validation
- [ ] 10 ns simulation matches baseline within tolerance
- [ ] No performance regression on any system size
- [ ] Combined speedup ≥ 2.0x
- [ ] Documentation updated

---

## SOVEREIGNTY DECLARATION

All optimizations in this plan:
- ❌ Require NO external libraries beyond CUDA/cuFFT
- ❌ Require NO proprietary code
- ✅ Are implemented entirely in Rust + CUDA
- ✅ Maintain full system sovereignty

---

*Document Version: 1.0*
*Created: 2026-01-14*
*Target Completion: 3 weeks after Phase 6*
