# PRISM4D: GPU-Accelerated Cryptic Pocket Discovery
#
# Multi-stage build for production-ready container
# Requires NVIDIA GPU with CUDA 12.0+ support

# =============================================================================
# Stage 1: Build environment (Rust + CUDA)
# =============================================================================
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set up working directory
WORKDIR /build

# Copy source code
COPY . .

# Build release binaries with CUDA support
RUN cargo build --release --features cuda -p prism-validation --bin generate-ensemble && \
    cargo build --release -p prism-validation --bin analyze_ensemble

# =============================================================================
# Stage 2: Python environment
# =============================================================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS python-builder

ENV DEBIAN_FRONTEND=noninteractive

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:${PATH}"

# Create conda environment with dependencies
RUN conda create -n prism4d python=3.11 -y && \
    conda run -n prism4d conda install -c conda-forge \
        openmm=8.1 \
        pdbfixer \
        numpy \
        requests \
        -y && \
    conda clean -afy

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

LABEL maintainer="PRISM4D Team"
LABEL description="PRISM4D: GPU-Accelerated Cryptic Pocket Discovery Pipeline"
LABEL version="1.2.0"

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment
COPY --from=python-builder /opt/conda /opt/conda
ENV PATH="/opt/conda/bin:${PATH}"

# Copy built binaries
COPY --from=builder /build/target/release/generate-ensemble /usr/local/bin/
COPY --from=builder /build/target/release/analyze_ensemble /usr/local/bin/

# Copy pipeline scripts
COPY scripts/prism_pipeline.py /opt/prism4d/scripts/
COPY scripts/stage1_sanitize.py /opt/prism4d/scripts/
COPY scripts/stage2_topology.py /opt/prism4d/scripts/

# Copy PTX kernels (required for CUDA)
COPY crates/prism-gpu/target/ptx/*.ptx /opt/prism4d/ptx/

# Set environment
ENV PRISM4D_HOME=/opt/prism4d
ENV PATH="/opt/prism4d/scripts:${PATH}"
ENV PYTHONPATH="/opt/prism4d/scripts:${PYTHONPATH}"

# Create working directory
WORKDIR /workspace

# Default command: show help
CMD ["conda", "run", "-n", "prism4d", "python", "/opt/prism4d/scripts/prism_pipeline.py", "--help"]
