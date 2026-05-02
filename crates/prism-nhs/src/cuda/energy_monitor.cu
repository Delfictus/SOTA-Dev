// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / B.3.2 — Hamiltonian Auditor implementation
// ═══════════════════════════════════════════════════════════════════════════

#include "energy_monitor.cuh"
#include <cuda_runtime.h>
#include <cstdint>

// CUB device-wide reduction.  CUDA 13.x ships CUB under
// /usr/local/cuda/include/cccl/cub; build.rs adds that include path.
#include <cub/cub.cuh>

// ─── Window update kernel — f32→f64 promotion + roll ──────────────────────
//
// Single-thread <<<1,1>>>.  Reads *d_pe_scalar_f32 (the CUB-reduce
// f32 output) and rolls the energy window:
//   window->prev = window->cur
//   window->cur  = static_cast<double>(*d_pe_scalar_f32)
//
// Promoting to f64 once per launch (rather than maintaining f64 across
// the reduce) gives numerical stability over the 18-hour campaign
// without requiring a custom CUB reduce-with-init-type configuration.

extern "C"
__global__ void prism_energy_monitor_window_update_kernel(
    EnergyWindow* __restrict__ window,
    const float*  __restrict__ d_pe_scalar_f32
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (window == nullptr || d_pe_scalar_f32 == nullptr) return;

    const double new_cur = static_cast<double>(*d_pe_scalar_f32);
    window->prev = window->cur;
    window->cur  = new_cur;
}

// ─── Host-side launchers ──────────────────────────────────────────────────

extern "C"
int prism_energy_monitor_temp_storage_bytes(
    uint32_t n,
    size_t*  out_temp_bytes)
{
    if (out_temp_bytes == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    size_t bytes = 0;
    float*  d_in_dummy   = nullptr;
    float*  d_out_dummy  = nullptr;
    cudaError_t err = cub::DeviceReduce::Sum(
        /*d_temp_storage=*/      nullptr,
        /*temp_storage_bytes=*/  bytes,
        /*d_in=*/                d_in_dummy,
        /*d_out=*/               d_out_dummy,
        /*num_items=*/           static_cast<int>(n));
    if (err != cudaSuccess) return static_cast<int>(err);

    *out_temp_bytes = bytes;
    return static_cast<int>(cudaSuccess);
}

extern "C"
int prism_energy_monitor_launch_reduce(
    const float*    d_pe_components,
    uint32_t        n,
    void*           d_temp_storage,
    size_t          temp_storage_bytes,
    float*          d_pe_scalar_f32,
    EnergyWindow*   d_energy_window,
    void*           stream)
{
    if (d_pe_components == nullptr || d_pe_scalar_f32 == nullptr ||
        d_energy_window == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // CUB DeviceReduce::Sum on f32 → f32 scalar.  Captured into the
    // graph as a series of cub::DeviceReduce::Sum-internal nodes; CUDA
    // 12.4+ supports CUB device-wide algorithm capture so this works
    // inside cuStreamBeginCapture.
    cudaError_t err = cub::DeviceReduce::Sum(
        d_temp_storage,
        temp_storage_bytes,
        d_pe_components,
        d_pe_scalar_f32,
        static_cast<int>(n),
        s);
    if (err != cudaSuccess) return static_cast<int>(err);

    // f32 → f64 promotion + window roll, in a single 1-thread kernel.
    prism_energy_monitor_window_update_kernel<<<1, 1, 0, s>>>(
        d_energy_window, d_pe_scalar_f32);
    return static_cast<int>(cudaGetLastError());
}
