FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates less nano curl tmux \
    && pip3 install --break-system-packages nvitop \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sL https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 \
    -o /usr/local/bin/ttyd && chmod +x /usr/local/bin/ttyd

ARG DEMO_PASSWORD=demo
RUN useradd -m -s /bin/bash demo && echo "demo:${DEMO_PASSWORD}" | chpasswd

RUN mkdir -p /opt/prism4d/bin /opt/prism4d/scripts /opt/prism4d/samples \
    /opt/prism4d/topologies /opt/prism4d/output /opt/prism4d/kernels

COPY nhs_rt_full /opt/prism4d/bin/nhs_rt_full
RUN chmod +x /opt/prism4d/bin/nhs_rt_full
COPY kernels/ /opt/prism4d/kernels/
COPY samples/ /opt/prism4d/samples/
COPY topologies/ /opt/prism4d/topologies/
COPY run.sh /opt/prism4d/bin/run
COPY output.sh /opt/prism4d/bin/output
RUN chmod +x /opt/prism4d/bin/output
RUN chmod +x /opt/prism4d/bin/run
COPY tmux.conf /opt/prism4d/bin/tmux.conf

# Hardcoded absolute path symlink from build machine
RUN mkdir -p /home/diddy/Desktop/Prism4D-bio/crates/prism-gpu/target && \
    ln -sfn /opt/prism4d/kernels/ptx /home/diddy/Desktop/Prism4D-bio/crates/prism-gpu/target/ptx

# Relative path fallbacks
RUN mkdir -p /opt/prism4d/target/ptx && \
    ln -sfn /opt/prism4d/kernels/ptx/* /opt/prism4d/target/ptx/ && \
    mkdir -p /opt/prism4d/crates/prism-gpu/src/kernels && \
    ln -sfn /opt/prism4d/kernels/ptx/* /opt/prism4d/crates/prism-gpu/src/kernels/ && \
    mkdir -p /opt/prism4d/assets/ptx && \
    ln -sfn /opt/prism4d/kernels/ptx/* /opt/prism4d/assets/ptx/

ENV PRISM_PTX_DIR=/opt/prism4d/kernels/ptx
ENV PRISM4D_PTX_DIR=/opt/prism4d/kernels/ptx
ENV PRISM_OPTIXIR_DIR=/opt/prism4d/kernels/optixir
ENV PATH="/opt/prism4d/bin:${PATH}"

RUN echo 'cd /opt/prism4d' >> /home/demo/.bashrc

COPY motd /home/demo/.motd
RUN echo 'cat /home/demo/.motd' >> /home/demo/.bashrc

RUN chown -R demo:demo /opt/prism4d/output
RUN echo 'export TMOUT=1800' >> /home/demo/.bashrc && \
    echo 'readonly TMOUT' >> /home/demo/.bashrc

COPY demo-entrypoint.sh /usr/local/bin/demo-entrypoint.sh
RUN chmod +x /usr/local/bin/demo-entrypoint.sh

EXPOSE 7681
COPY generate_viewer.py /opt/prism4d/scripts/generate_viewer.py
RUN chmod +x /opt/prism4d/scripts/generate_viewer.py
COPY view.sh /opt/prism4d/bin/view
RUN chmod +x /opt/prism4d/bin/view
EXPOSE 8080
USER demo
WORKDIR /opt/prism4d
CMD ["/usr/local/bin/demo-entrypoint.sh"]
