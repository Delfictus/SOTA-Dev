#!/bin/bash
export PRISM4D_PTX_DIR=/opt/prism4d/kernels/ptx
export PRISM_PTX_DIR=/opt/prism4d/kernels/ptx
export PRISM_OPTIXIR_DIR=/opt/prism4d/kernels/optixir
export PATH="/opt/prism4d/bin:/opt/prism4d/scripts:$PATH"

# Start HTTP server for viewer on port 8080
cd /opt/prism4d/output
python3 -m http.server 8080 --bind 0.0.0.0 &

# Start ttyd
exec ttyd --port 7681 --writable bash -l
