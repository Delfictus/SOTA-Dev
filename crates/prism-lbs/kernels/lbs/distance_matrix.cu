// Placeholder CUDA kernel for pairwise distance matrix

extern "C" __global__ void distance_matrix_kernel(
    const float* x,
    const float* y,
    const float* z,
    int n,
    float* out_dist)
{
    // out_dist is row-major (n x n). Only fill upper triangle in production.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float dx = x[i] - x[j];
        float dy = y[i] - y[j];
        float dz = z[i] - z[j];
        float d = sqrtf(dx * dx + dy * dy + dz * dz);
        out_dist[i * n + j] = d;
    }
}
