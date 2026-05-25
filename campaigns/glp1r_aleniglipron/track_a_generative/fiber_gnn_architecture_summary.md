# Fiber-Bundle GFlowNet Architecture

- Base space: PyG-packed scaffold atom graph with sparse `edge_index`.
- Fiber space: explicit `[N_atoms, 5, D]` CCNS phase tensor.
- Phase routing: GRU plus 1D convolution over Cold Hold, Ramp Up, Warm Hold, Ramp Down, Cold Return.
- Orthogonal routing: base graph messages and within-atom fiber messages remain separate before gated fusion.
- Action policy: exit-vector-conditioned dot-product attention over calibration anchor embeddings.
- Reward authority: Batched Rust Oracle; Python never computes terminal rewards.
