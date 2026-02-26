##############################################################################
# PRISM4D Demo Image — Private Demonstration Release
#
# Provides a fully sandboxed, SSH-accessible environment with FULL activity
# auditing for prospective clients to evaluate the PRISM4D platform.
#
# Build:  docker build -t prism4d-demo .
# Run:    docker compose up -d
# Access: ssh demo@localhost -p 2222   (password: demo)
# Audit:  tail -f /var/log/prism4d-audit/commands.log
##############################################################################

FROM nvidia/cuda:12.6.3-base-ubuntu22.04

LABEL maintainer="PRISM4D Team"
LABEL description="PRISM4D Cryptic Pocket Detection — Private Demo"
LABEL version="1.0.0-demo"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ── 1. System packages ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    jq \
    less \
    vim-tiny \
    htop \
    tree \
    ca-certificates \
    libgomp1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# ── 2. SSH server setup ──────────────────────────────────────────────
RUN mkdir -p /run/sshd

COPY sshd_config /etc/ssh/sshd_config

# ── 3. Create sandboxed demo user ────────────────────────────────────
# - Standard bash shell (needed for PROMPT_COMMAND audit hook)
# - Cannot sudo
# - Home directory is the only writable location
# - Password: demo
RUN useradd -m -s /bin/bash -d /home/demo demo \
    && echo "demo:demo" | chpasswd \
    && mkdir -p /home/demo/workspace /home/demo/samples /home/demo/results

# ── 4. Install Miniconda for Python pipeline ─────────────────────────
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:${PATH}"

# ── 5. Create preprocessing conda environment ────────────────────────
COPY envs/preprocessing.yml /tmp/preprocessing.yml
RUN conda env create -f /tmp/preprocessing.yml || \
    conda create -n prism4d-preprocessing python=3.11 \
        numpy requests rdkit openmm pdbfixer -c conda-forge -y \
    && conda clean -afy \
    && rm /tmp/preprocessing.yml

# ── 6. Copy PRISM4D demo release ─────────────────────────────────────
COPY prism4d-demo/ /opt/prism4d/

# Ensure binaries are executable
RUN chmod +x /opt/prism4d/bin/* /opt/prism4d/prism4d /opt/prism4d/setup.sh 2>/dev/null || true

# Create PTX symlink for compile-time baked paths
RUN mkdir -p /home/diddy/Desktop/Prism4D-bio/crates/prism-gpu/target \
    && ln -sfn /opt/prism4d/kernels/ptx \
       /home/diddy/Desktop/Prism4D-bio/crates/prism-gpu/target/ptx

# ── 7. Copy sample structures for demo ───────────────────────────────
COPY samples/ /home/demo/samples/

# ── 8. Environment variables ─────────────────────────────────────────
ENV PRISM_PTX_DIR="/opt/prism4d/kernels/ptx"
ENV PRISM_OPTIXIR_DIR="/opt/prism4d/kernels/optixir"
ENV PATH="/opt/prism4d/bin:/opt/prism4d/scripts:/opt/prism4d:${PATH}"

# ── 9. Demo user profile ─────────────────────────────────────────────
COPY motd /etc/motd
RUN cat > /home/demo/.bashrc << 'BASHRC'
# PRISM4D Demo Environment
export PRISM_PTX_DIR="/opt/prism4d/kernels/ptx"
export PRISM_OPTIXIR_DIR="/opt/prism4d/kernels/optixir"
export PATH="/opt/prism4d/bin:/opt/prism4d/scripts:/opt/prism4d:/opt/conda/envs/prism4d-preprocessing/bin:${PATH}"
export HISTTIMEFORMAT="%Y-%m-%dT%H:%M:%S "
export HISTSIZE=10000
export HISTFILESIZE=10000
shopt -s histappend

alias prism='prism4d'
alias ll='ls -la'
alias results='ls -la ~/results/'

# Show welcome on login
cat /etc/motd

# Activate conda for prep
eval "$(/opt/conda/bin/conda shell.bash hook)" 2>/dev/null
conda activate prism4d-preprocessing 2>/dev/null

cd ~/workspace
BASHRC

# Copy quick-start guide into demo home
COPY quickstart.md /home/demo/README.md

# Fix ownership
RUN chown -R demo:demo /home/demo

# ── 10. Audit system ─────────────────────────────────────────────────
# Create audit directory (will be shadowed by host bind-mount at runtime)
RUN mkdir -p /var/log/prism4d-audit \
    && chmod 733 /var/log/prism4d-audit

# Install entrypoint with audit hooks
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh

# ── 11. Security hardening ───────────────────────────────────────────
# Remove package manager to prevent installs
RUN apt-get remove -y --purge wget curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Make system directories read-only for demo user
RUN chmod 755 /opt/prism4d \
    && find /opt/prism4d -type f -exec chmod 644 {} + \
    && find /opt/prism4d/bin -type f -exec chmod 755 {} + \
    && chmod 755 /opt/prism4d/prism4d /opt/prism4d/setup.sh \
    && find /opt/prism4d/scripts -name "*.sh" -exec chmod 755 {} +

# Demo user cannot read audit logs
RUN chown root:root /var/log/prism4d-audit \
    && chmod 733 /var/log/prism4d-audit

EXPOSE 22

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD pgrep sshd > /dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
