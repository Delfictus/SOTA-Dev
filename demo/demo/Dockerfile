FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install SSH server and minimal deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    python3 \
    python3-pip \
    ca-certificates \
    less \
    nano \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /run/sshd

# Create demo user with restricted shell access
ARG DEMO_PASSWORD=prism4d-demo
RUN useradd -m -s /bin/bash demo && \
    echo "demo:${DEMO_PASSWORD}" | chpasswd

# SSH config: listen on 2222, disable root login
RUN sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    echo "AllowUsers demo" >> /etc/ssh/sshd_config

# Create PRISM-4D directory structure
RUN mkdir -p /opt/prism4d/bin /opt/prism4d/scripts /opt/prism4d/samples /opt/prism4d/output

# Copy binary
COPY nhs_rt_full /opt/prism4d/bin/nhs_rt_full
RUN chmod +x /opt/prism4d/bin/nhs_rt_full

# Copy prism-prep script
COPY prism-prep /opt/prism4d/scripts/prism-prep
RUN chmod +x /opt/prism4d/scripts/prism-prep

# Copy sample PDB files
COPY samples/ /opt/prism4d/samples/

# Add PRISM-4D to PATH for demo user
RUN echo 'export PATH="/opt/prism4d/bin:/opt/prism4d/scripts:$PATH"' >> /home/demo/.bashrc

# Set working directory for demo user
RUN echo 'cd /opt/prism4d' >> /home/demo/.bashrc

# MOTD / Welcome banner
COPY motd /etc/motd
RUN echo "PrintMotd yes" >> /etc/ssh/sshd_config

# Demo user owns the output directory
RUN chown -R demo:demo /opt/prism4d/output

# Read-only for everything except output
RUN chown -R root:root /opt/prism4d/bin /opt/prism4d/scripts /opt/prism4d/samples && \
    chmod -R 755 /opt/prism4d/bin /opt/prism4d/scripts && \
    chmod -R 644 /opt/prism4d/samples/*.pdb && \
    chmod 755 /opt/prism4d/samples

# Session timeout: 30 min idle
RUN echo 'export TMOUT=1800' >> /home/demo/.bashrc
RUN echo 'readonly TMOUT' >> /home/demo/.bashrc

EXPOSE 2222

CMD ["/usr/sbin/sshd", "-D", "-e"]
