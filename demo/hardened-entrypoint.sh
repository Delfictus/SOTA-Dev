#!/bin/bash
export PATH="/var/lib/p4r/d:/var/lib/p4r/e:/usr/local/bin:/usr/bin:/bin"

# Start HTTP server for viewer (on output dir only)
cd /var/lib/p4r/o
python3 -m http.server 8080 --bind 0.0.0.0 &

# Start ttyd — public demo (no auth)
exec ttyd \
    --port 7681 \
    --writable \
    --max-clients 3 \
    --client-option titleFixed="PRISM4D Demo" \
    --client-option fontFamily="JetBrains Mono,monospace" \
    --ping-interval 30 \
    --once \
    bash -l
