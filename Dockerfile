# PRISM-4D Docker Container
# Sovereign GPU-Accelerated Molecular Dynamics Engine
#
# Build:   docker build -t prism4d:latest .
# Run:     docker run --gpus all -v $(pwd)/data:/app/data prism4d:latest
#
# Requirements:
#   - NVIDIA GPU with compute capability >= 7.0
#   - nvidia-container-toolkit installed on host
#   - Docker 19.03+ with GPU support

FROM nvidia/cuda:13.0-devel-ubuntu22.04

LABEL maintainer="PRISM-4D Team"
LABEL description="PRISM-4D: Sovereign GPU-Accelerated MD Engine"
LABEL version="1.0.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    # Rust dependencies
    libssl-dev \
    libclang-dev \
    # Python for analysis scripts
    python3 \
    python3-pip \
    python3-venv \
    # Visualization (optional, for headless rendering)
    xvfb \
    libgl1-mesa-glx \
    libglu1-mesa \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Verify Rust and Cargo
RUN rustc --version && cargo --version

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Verify CUDA installation
RUN nvcc --version

# Create application directory
WORKDIR /app

# Copy Cargo manifests first (for layer caching)
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build dependencies first (caching layer)
RUN cargo build --release 2>/dev/null || true

# Copy the rest of the source code
COPY . .

# Build PRISM-4D with CUDA support
RUN cargo build --release --features cuda

# Install Python dependencies for analysis
RUN pip3 install --no-cache-dir \
    numpy>=1.21.0 \
    pandas>=1.3.0 \
    matplotlib>=3.4.0 \
    scipy>=1.7.0 \
    seaborn>=0.11.0

# Create output directories
RUN mkdir -p /app/data/ensembles \
             /app/data/raw \
             /app/results \
             /app/publication

# Set default environment variables
ENV PRISM4D_LOG_LEVEL=info
ENV PRISM4D_GPU_DEVICE=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD cargo run --release -p prism-gpu -- --health-check || exit 1

# Default entry point - show help
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'PRISM-4D Container Ready' && \
     echo '' && \
     echo 'Available commands:' && \
     echo '  cargo run --release -p prism-validation --bin generate-ensemble -- --help' && \
     echo '  python3 scripts/generate_publication_data.py' && \
     echo '' && \
     echo 'GPU Status:' && \
     nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"]

# Example usage:
#
# 1. Run MD simulation:
#    docker run --gpus all -v $(pwd)/data:/app/data prism4d:latest \
#      "cargo run --release -p prism-validation --bin generate-ensemble -- \
#       --pdb data/raw/6M0J_RBD_fixed.pdb \
#       --output data/ensembles/6M0J_output.pdb \
#       --frames 1000 --temperature 310"
#
# 2. Generate publication data:
#    docker run --gpus all -v $(pwd):/app prism4d:latest \
#      "python3 scripts/generate_publication_data.py"
#
# 3. Interactive shell:
#    docker run --gpus all -it -v $(pwd):/app prism4d:latest /bin/bash
