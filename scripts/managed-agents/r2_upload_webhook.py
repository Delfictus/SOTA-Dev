#!/usr/bin/env python3
"""
R2 Upload Webhook — Bridge between local spike watcher and managed agents.

When the local prism_spike_watcher.py daemon confirms a file on R2, it can
POST to this webhook which triggers a managed agent validation session.

This runs as a lightweight Flask/FastAPI server on the local machine, OR
as a Cloudflare Worker (see r2_webhook_worker.js).

For local use:
  python3 scripts/managed-agents/r2_upload_webhook.py --port 8787

The webhook receives upload notifications and batches them. When a batch
threshold is reached (or a timeout fires), it starts a managed agent
session to validate the batch.
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 10          # Start validation after this many uploads
BATCH_TIMEOUT_SECS = 300  # Or after this many seconds since first upload
STATE_FILE = Path(__file__).parent / ".agent_state.json"

# ---------------------------------------------------------------------------
# Batch accumulator
# ---------------------------------------------------------------------------

_batch_lock = threading.Lock()
_batch: list[dict] = []
_batch_first_ts: float | None = None
_timer: threading.Timer | None = None


def _load_agent_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _trigger_validation():
    """Start a managed agent session to validate the accumulated batch."""
    global _batch, _batch_first_ts, _timer

    with _batch_lock:
        if not _batch:
            return
        batch = _batch.copy()
        _batch = []
        _batch_first_ts = None
        if _timer:
            _timer.cancel()
            _timer = None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"[WARN] No ANTHROPIC_API_KEY — cannot trigger validation for {len(batch)} files")
        return

    state = _load_agent_state()
    agent_id = state.get("dataops_agent_id")
    env_id = state.get("environment_id")

    if not agent_id or not env_id:
        print(f"[WARN] Agents not set up — run setup_agents.py setup first")
        return

    client = Anthropic(api_key=api_key)

    # Build validation message
    file_list = "\n".join(f"  - {f['r2_path']} ({f.get('size_bytes', '?')} bytes)" for f in batch)
    message = f"""New spike data batch uploaded to R2 ({len(batch)} files).

Validate these files:
{file_list}

For each file:
1. Verify it exists on R2 (rclone ls)
2. If Parquet: read with pyarrow, check row count > 0, verify schema has required columns
3. Check target PDB against contamination list (SNDC targets must be flagged)
4. Update the training manifest via the update_training_manifest tool

Report summary when done.
"""

    print(f"[INFO] Starting validation session for {len(batch)} files")
    try:
        session = client.beta.sessions.create(
            agent=agent_id,
            environment_id=env_id,
            title=f"Batch validation — {len(batch)} files — {datetime.now(timezone.utc).isoformat()}",
        )

        # Fire-and-forget: send the message (agent runs autonomously)
        client.beta.sessions.events.send(
            session.id,
            events=[
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": message}],
                },
            ],
        )
        print(f"[INFO] Validation session started: {session.id}")
    except Exception as e:
        print(f"[ERROR] Failed to start validation session: {e}")


def _add_to_batch(upload_info: dict):
    """Add an upload notification to the batch."""
    global _batch_first_ts, _timer

    with _batch_lock:
        _batch.append(upload_info)

        if _batch_first_ts is None:
            _batch_first_ts = time.time()
            # Start timeout timer
            _timer = threading.Timer(BATCH_TIMEOUT_SECS, _trigger_validation)
            _timer.daemon = True
            _timer.start()

        if len(_batch) >= BATCH_SIZE:
            # Cancel timer and trigger immediately
            if _timer:
                _timer.cancel()
                _timer = None

    if len(_batch) >= BATCH_SIZE:
        _trigger_validation()


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/upload-complete":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                _add_to_batch(data)
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "queued"}).encode())
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            with _batch_lock:
                status = {
                    "pending_batch": len(_batch),
                    "batch_timeout_secs": BATCH_TIMEOUT_SECS,
                    "batch_size_threshold": BATCH_SIZE,
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[WEBHOOK] {args[0]}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="R2 Upload Webhook Server")
    parser.add_argument("--port", type=int, default=8787, help="Port to listen on")
    args = parser.parse_args()

    print(f"R2 Upload Webhook listening on :{args.port}")
    print(f"  POST /upload-complete — notify of R2 upload")
    print(f"  GET  /status          — check batch status")
    print(f"  Batch size: {BATCH_SIZE}, timeout: {BATCH_TIMEOUT_SECS}s")

    server = HTTPServer(("0.0.0.0", args.port), WebhookHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        _trigger_validation()  # Flush remaining batch
        server.server_close()


if __name__ == "__main__":
    main()
