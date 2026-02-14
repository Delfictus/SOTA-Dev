// Placeholder CUDA kernel for surface accessibility (Shrake-Rupley style)
// This file documents the intended GPU offload; implementation pending.

extern "C" __global__ void surface_accessibility_kernel(
    const float* x,
    const float* y,
    const float* z,
    const float* radii,
    int num_atoms,
    int samples,
    float probe_radius,
    float* out_sasa)
{
    // Intentionally left unimplemented; production kernel should:
    // 1) generate Fibonacci sphere samples on device
    // 2) test occlusion against neighbor list
    // 3) accumulate exposed fraction and write SASA per atom
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_atoms) {
        out_sasa[idx] = 0.0f;
    }
}
