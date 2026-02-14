extern "C" __global__ void feature_merge_kernel(
    const float* __restrict__ main_features,    // [n_residues * 136]
    const float* __restrict__ cryptic_features, // [n_residues * 4]
    float* __restrict__ output,                 // [n_residues * 140]
    int n_residues
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_residues) return;

    // Base offsets
    int main_offset = idx * 136;
    int cryptic_offset = idx * 4;
    int out_offset = idx * 140;

    // Copy main features (unrolled loop for 136 elements)
    // 136 is divisible by 4, so we can use float4 copy if alignment allowed, but simple loop is fine for now.
    // #pragma unroll
    for (int i = 0; i < 136; i++) {
        output[out_offset + i] = main_features[main_offset + i];
    }

    // Copy cryptic features
    output[out_offset + 136] = cryptic_features[cryptic_offset + 0];
    output[out_offset + 137] = cryptic_features[cryptic_offset + 1];
    output[out_offset + 138] = cryptic_features[cryptic_offset + 2];
    output[out_offset + 139] = cryptic_features[cryptic_offset + 3];
}
