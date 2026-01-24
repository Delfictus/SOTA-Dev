/**
 * PME (Particle Mesh Ewald) CUDA Kernels
 *
 * Implements long-range electrostatics via Ewald summation with FFT.
 * Uses 4th-order B-spline charge spreading and force interpolation.
 *
 * Algorithm:
 * 1. Spread atom charges to grid using B-splines
 * 2. FFT grid to reciprocal space
 * 3. Apply Green's function (energy/potential)
 * 4. Inverse FFT back to real space
 * 5. Interpolate forces from grid to atoms
 */

extern "C" {

// Constants
#define PME_ORDER 4          // B-spline order (4th order = cubic)
#define ONE_4PI_EPS0 332.0636f  // kcal*Å/(mol*e²)

// Box dimensions are passed as kernel parameters, not device globals
// This allows PME to be compiled independently from amber_mega_fused.cu

/**
 * Compute 4th order B-spline values
 * Returns M_4(u) for u in [0, 4)
 */
__device__ __forceinline__ void bspline_4(float u, float* w) {
    // 4th order B-spline (cubic)
    // M_4(u) for u in [0,4)
    float u2 = u * u;
    float u3 = u2 * u;

    w[0] = (1.0f - u) * (1.0f - u) * (1.0f - u) / 6.0f;
    w[1] = (4.0f - 6.0f * u2 + 3.0f * u3) / 6.0f;
    w[2] = (1.0f + 3.0f * u + 3.0f * u2 - 3.0f * u3) / 6.0f;
    w[3] = u3 / 6.0f;
}

/**
 * Compute derivative of 4th order B-spline
 * dM_4(u)/du
 */
__device__ __forceinline__ void bspline_4_deriv(float u, float* dw) {
    float u2 = u * u;

    dw[0] = -0.5f * (1.0f - u) * (1.0f - u);
    dw[1] = u * (1.5f * u - 2.0f);
    dw[2] = 0.5f + u - 1.5f * u2;
    dw[3] = 0.5f * u2;
}

/**
 * Spread atom charges to PME grid using B-splines
 *
 * Each atom contributes charge to PME_ORDER^3 = 64 grid points
 * Uses atomicAdd for thread-safe accumulation
 */
__global__ void pme_spread_charges(
    const float* __restrict__ positions,  // [n_atoms * 3]
    const float* __restrict__ charges,    // [n_atoms]
    float* __restrict__ grid,             // [nx * ny * nz] - zeroed
    int n_atoms,
    int nx, int ny, int nz,
    float box_inv_x, float box_inv_y, float box_inv_z  // 1/L for each dimension
) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;

    float q = charges[atom];
    if (fabsf(q) < 1e-10f) return;  // Skip uncharged atoms

    // Get atom position
    float px = positions[atom * 3];
    float py = positions[atom * 3 + 1];
    float pz = positions[atom * 3 + 2];

    // Convert to fractional coordinates [0, 1)
    float fx = px * box_inv_x;
    float fy = py * box_inv_y;
    float fz = pz * box_inv_z;

    // Wrap to [0, 1)
    fx -= floorf(fx);
    fy -= floorf(fy);
    fz -= floorf(fz);

    // Scale to grid units [0, nx), [0, ny), [0, nz)
    float gx = fx * (float)nx;
    float gy = fy * (float)ny;
    float gz = fz * (float)nz;

    // Grid indices of nearest grid point below atom
    int i0 = (int)gx;
    int j0 = (int)gy;
    int k0 = (int)gz;

    // Fractional distance from nearest grid point
    float ux = gx - (float)i0;
    float uy = gy - (float)j0;
    float uz = gz - (float)k0;

    // Compute B-spline weights
    float wx[PME_ORDER], wy[PME_ORDER], wz[PME_ORDER];
    bspline_4(ux, wx);
    bspline_4(uy, wy);
    bspline_4(uz, wz);

    // Spread charge to PME_ORDER^3 grid points
    for (int di = 0; di < PME_ORDER; di++) {
        int i = (i0 - 1 + di + nx) % nx;  // Wrap with PBC
        for (int dj = 0; dj < PME_ORDER; dj++) {
            int j = (j0 - 1 + dj + ny) % ny;
            for (int dk = 0; dk < PME_ORDER; dk++) {
                int k = (k0 - 1 + dk + nz) % nz;

                float weight = wx[di] * wy[dj] * wz[dk];
                int grid_idx = i * ny * nz + j * nz + k;

                atomicAdd(&grid[grid_idx], q * weight);
            }
        }
    }
}

/**
 * Apply reciprocal space convolution (Green's function)
 *
 * After forward FFT, this kernel multiplies each k-space component
 * by the reciprocal space influence function:
 *
 * G(k) = (4π/k²) * exp(-k²/(4β²)) * |B(k)|²
 *
 * where β is the Ewald splitting parameter and B(k) is the B-spline
 * structure factor correction.
 *
 * NOTE: cuFFT outputs interleaved complex data: [re0, im0, re1, im1, ...]
 * So we index with idx*2 for real and idx*2+1 for imaginary.
 */
__global__ void pme_reciprocal_convolution(
    float* __restrict__ complex_grid,  // Interleaved complex: [re0, im0, re1, im1, ...]
    float* __restrict__ energy,        // Output: reciprocal energy
    float beta,                        // Ewald splitting parameter
    int nx, int ny, int nz,
    float box_inv_x, float box_inv_y, float box_inv_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * (nz / 2 + 1);  // R2C output is hermitian
    if (idx >= total) return;

    // Convert linear index to 3D
    int nz_complex = nz / 2 + 1;
    int k = idx % nz_complex;
    int j = (idx / nz_complex) % ny;
    int i = idx / (nz_complex * ny);

    // Interleaved indexing for complex data
    int re_idx = idx * 2;
    int im_idx = idx * 2 + 1;

    // Skip DC component (k=0)
    if (i == 0 && j == 0 && k == 0) {
        complex_grid[re_idx] = 0.0f;
        complex_grid[im_idx] = 0.0f;
        return;
    }

    // Compute wave vector (handle negative frequencies)
    int mi = (i <= nx / 2) ? i : i - nx;
    int mj = (j <= ny / 2) ? j : j - ny;
    int mk = k;  // k only goes to nz/2 due to R2C

    // k² in reciprocal space
    float kx = 2.0f * M_PI * (float)mi * box_inv_x;
    float ky = 2.0f * M_PI * (float)mj * box_inv_y;
    float kz = 2.0f * M_PI * (float)mk * box_inv_z;
    float k2 = kx * kx + ky * ky + kz * kz;

    if (k2 < 1e-10f) {
        complex_grid[re_idx] = 0.0f;
        complex_grid[im_idx] = 0.0f;
        return;
    }

    // Ewald screening: exp(-k²/(4β²))
    float four_beta2 = 4.0f * beta * beta;
    float ewald_factor = expf(-k2 / four_beta2);

    // B-spline structure factor correction
    // B(k) ≈ 1 for well-resolved k, we use simplified form
    float bx = (mi == 0) ? 1.0f : sinf(M_PI * (float)mi / (float)nx) / (M_PI * (float)mi / (float)nx);
    float by = (mj == 0) ? 1.0f : sinf(M_PI * (float)mj / (float)ny) / (M_PI * (float)mj / (float)ny);
    float bz = (mk == 0) ? 1.0f : sinf(M_PI * (float)mk / (float)nz) / (M_PI * (float)mk / (float)nz);
    float bx4 = bx * bx * bx * bx;  // 4th order B-spline
    float by4 = by * by * by * by;
    float bz4 = bz * bz * bz * bz;
    float b_corr = 1.0f / (bx4 * by4 * bz4);

    // Green's function: (4π/(V*k²)) * exp(-k²/(4β²)) / B(k)²
    // The 1/V factor is CRITICAL - without it, energy scales with box size!
    float inv_volume = box_inv_x * box_inv_y * box_inv_z;  // 1/(Lx*Ly*Lz)
    float green = 4.0f * M_PI * ewald_factor * b_corr * inv_volume / k2;

    // Load complex values from interleaved array
    float re = complex_grid[re_idx];
    float im = complex_grid[im_idx];

    // Accumulate energy: E = (1/2) * sum_k |ρ(k)|² * G(k)
    // Factor of 2 for k != 0 (hermitian symmetry)
    float rho2 = re * re + im * im;
    float e_contrib = 0.5f * rho2 * green;
    if (k > 0 && k < nz / 2) {
        e_contrib *= 2.0f;  // Account for hermitian conjugate
    }
    atomicAdd(energy, e_contrib * ONE_4PI_EPS0);

    // Multiply grid by Green's function and store back
    complex_grid[re_idx] = re * green;
    complex_grid[im_idx] = im * green;
}

/**
 * Interpolate forces from PME grid to atoms
 *
 * The force is the negative gradient of the potential:
 * F_i = -q_i * ∇φ(r_i)
 *
 * We compute ∇φ by differentiating the B-spline interpolation.
 */
__global__ void pme_interpolate_forces(
    const float* __restrict__ positions,
    const float* __restrict__ charges,
    const float* __restrict__ potential_grid,  // After inverse FFT
    float* __restrict__ forces,
    int n_atoms,
    int nx, int ny, int nz,
    float box_inv_x, float box_inv_y, float box_inv_z
) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;

    float q = charges[atom];
    if (fabsf(q) < 1e-10f) return;

    // Get atom position
    float px = positions[atom * 3];
    float py = positions[atom * 3 + 1];
    float pz = positions[atom * 3 + 2];

    // Convert to fractional coordinates
    float fx = px * box_inv_x;
    float fy = py * box_inv_y;
    float fz = pz * box_inv_z;

    // Wrap to [0, 1)
    fx -= floorf(fx);
    fy -= floorf(fy);
    fz -= floorf(fz);

    // Scale to grid units
    float gx = fx * (float)nx;
    float gy = fy * (float)ny;
    float gz = fz * (float)nz;

    int i0 = (int)gx;
    int j0 = (int)gy;
    int k0 = (int)gz;

    float ux = gx - (float)i0;
    float uy = gy - (float)j0;
    float uz = gz - (float)k0;

    // B-spline weights and derivatives
    float wx[PME_ORDER], wy[PME_ORDER], wz[PME_ORDER];
    float dwx[PME_ORDER], dwy[PME_ORDER], dwz[PME_ORDER];
    bspline_4(ux, wx);
    bspline_4(uy, wy);
    bspline_4(uz, wz);
    bspline_4_deriv(ux, dwx);
    bspline_4_deriv(uy, dwy);
    bspline_4_deriv(uz, dwz);

    // Interpolate gradient
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int di = 0; di < PME_ORDER; di++) {
        int i = (i0 - 1 + di + nx) % nx;
        for (int dj = 0; dj < PME_ORDER; dj++) {
            int j = (j0 - 1 + dj + ny) % ny;
            for (int dk = 0; dk < PME_ORDER; dk++) {
                int k = (k0 - 1 + dk + nz) % nz;

                int grid_idx = i * ny * nz + j * nz + k;
                float phi = potential_grid[grid_idx];

                // Gradient via chain rule: dφ/dr = dφ/du * du/dr
                // du/dr = n/L (grid points per box length)
                Fx += dwx[di] * wy[dj] * wz[dk] * phi;
                Fy += wx[di] * dwy[dj] * wz[dk] * phi;
                Fz += wx[di] * wy[dj] * dwz[dk] * phi;
            }
        }
    }

    // Scale by grid spacing and charge
    // F = -q * ∇φ * (n/L)
    float scale_x = -q * ONE_4PI_EPS0 * (float)nx * box_inv_x;
    float scale_y = -q * ONE_4PI_EPS0 * (float)ny * box_inv_y;
    float scale_z = -q * ONE_4PI_EPS0 * (float)nz * box_inv_z;

    // Add to forces (atomic for thread safety)
    atomicAdd(&forces[atom * 3 + 0], Fx * scale_x);
    atomicAdd(&forces[atom * 3 + 1], Fy * scale_y);
    atomicAdd(&forces[atom * 3 + 2], Fz * scale_z);
}

/**
 * Compute PME self-energy correction
 *
 * The self-interaction of charges with their own Gaussian screening
 * must be subtracted: E_self = -β/√π * Σ q_i²
 */
__global__ void pme_self_energy(
    const float* __restrict__ charges,
    float* __restrict__ energy,
    int n_atoms,
    float beta
) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;

    float q = charges[atom];
    float e_self = -beta / sqrtf(M_PI) * q * q * ONE_4PI_EPS0;

    atomicAdd(energy, e_self);
}

/**
 * Zero the PME grid
 */
__global__ void pme_zero_grid(
    float* __restrict__ grid,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        grid[idx] = 0.0f;
    }
}

/**
 * Normalize grid after inverse FFT
 * cuFFT doesn't normalize, so we divide by N
 */
__global__ void pme_normalize_grid(
    float* __restrict__ grid,
    int total_size,
    float norm_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        grid[idx] *= norm_factor;
    }
}

}  // extern "C"
