#!/bin/bash
set -e

# Copy key to tmpfs (root FS is read-only, can't delete from there)
cp /var/lib/p4r/.re /run/.re_tmp
source /run/.re_tmp

RAMDIR="/run/p4k"
mkdir -p "$RAMDIR/ptx" "$RAMDIR/optixir"

for f in /var/lib/p4r/k/ptx/*.ptx.enc; do
    [ -f "$f" ] || continue
    bn=$(basename "$f" .enc)
    openssl enc -aes-256-cbc -d -salt -pbkdf2 -iter 100000 \
        -in "$f" -out "$RAMDIR/ptx/$bn" -pass "pass:${KERNEL_KEY}" 2>/dev/null
done
for f in /var/lib/p4r/k/optixir/*.optixir.enc; do
    [ -f "$f" ] || continue
    bn=$(basename "$f" .enc)
    openssl enc -aes-256-cbc -d -salt -pbkdf2 -iter 100000 \
        -in "$f" -out "$RAMDIR/optixir/$bn" -pass "pass:${KERNEL_KEY}" 2>/dev/null
done

# Set permissions: only demo user can read decrypted kernels
chown -R demo:demo "$RAMDIR"
chmod -R 500 "$RAMDIR"

# Fix ownership of tmpfs mounts (mounted as root at runtime)
chown demo:demo /var/lib/p4r/o 2>/dev/null || true
chown -R demo:demo /home/demo 2>/dev/null || true

# Destroy the key from tmpfs (original on read-only FS is inaccessible to demo user)
rm -f /run/.re_tmp

# Wipe key from environment
unset KERNEL_KEY

export PRISM4D_PTX_DIR="$RAMDIR/ptx"
export PRISM_PTX_DIR="$RAMDIR/ptx"
export PRISM_OPTIXIR_DIR="$RAMDIR/optixir"
export PATH="/var/lib/p4r/d:/var/lib/p4r/e:/usr/local/bin:/usr/bin:/bin"

# Drop to demo user — never run ttyd as root
exec setpriv --reuid=1001 --regid=1001 --init-groups --reset-env \
    env PATH="$PATH" \
        PRISM4D_PTX_DIR="$PRISM4D_PTX_DIR" \
        PRISM_PTX_DIR="$PRISM_PTX_DIR" \
        PRISM_OPTIXIR_DIR="$PRISM_OPTIXIR_DIR" \
        HOME=/home/demo \
        USER=demo \
        TTYD_PASSWORD="${TTYD_PASSWORD:-prism4d-demo-2026}" \
    "$@"
