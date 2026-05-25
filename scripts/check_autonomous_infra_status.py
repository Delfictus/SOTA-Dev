#!/usr/bin/env python3
"""Check PRISM autonomous/cloud infrastructure without redeploying anything."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback is not used in CI.
    tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = REPO_ROOT / "cloud/prism-manifold-worker"
WRANGLER_TOML = WORKER_DIR / "wrangler.toml"
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_JSON = TRACK_A / "autonomous_infra_status_epoch017.json"
DEFAULT_MD = TRACK_A / "autonomous_infra_status_epoch017.md"
WORKER_URL = "https://prism-manifold-worker.is-0b9.workers.dev/"

JsonObject: TypeAlias = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--timeout", type=int, default=45)
    return parser.parse_args()


def static_config() -> JsonObject:
    if not WRANGLER_TOML.is_file() or tomllib is None:
        return {"status": "MISSING", "path": WRANGLER_TOML.as_posix()}
    decoded = tomllib.loads(WRANGLER_TOML.read_text(encoding="utf-8"))
    return {
        "status": "FOUND",
        "path": WRANGLER_TOML.as_posix(),
        "worker_name": decoded.get("name"),
        "account_id": decoded.get("account_id"),
        "d1_databases": decoded.get("d1_databases", []),
        "vectorize": decoded.get("vectorize", []),
        "r2_buckets": decoded.get("r2_buckets", []),
        "kv_namespaces": decoded.get("kv_namespaces", []),
        "queues": decoded.get("queues", {}),
        "durable_objects": decoded.get("durable_objects", {}),
    }


def classify_failure(text: str) -> str:
    lowered = text.lower()
    auth_markers = (
        "not authenticated",
        "authentication",
        "unauthorized",
        "api token",
        "login",
        "permission",
        "forbidden",
        "expired",
    )
    if any(marker in lowered for marker in auth_markers):
        return "AUTH_BLOCKED"
    if "command not found" in lowered or "enoent" in lowered:
        return "COMMAND_UNAVAILABLE"
    return "COMMAND_FAILED"


def run_command(name: str, command: list[str], *, cwd: Path | None, timeout: int) -> JsonObject:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "status": "TIMEOUT",
            "command": command,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    except FileNotFoundError as exc:
        return {
            "name": name,
            "status": "COMMAND_UNAVAILABLE",
            "command": command,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "stdout": "",
            "stderr": str(exc),
        }
    output = f"{completed.stdout}\n{completed.stderr}"
    status = "OK" if completed.returncode == 0 else classify_failure(output)
    if name == "worker_http" and completed.returncode == 0:
        http_status = completed.stdout.strip()
        if http_status in {"401", "403"}:
            status = "ACCESS_PROTECTED"
        elif not http_status.startswith(("2", "3")):
            status = "COMMAND_FAILED"
    return {
        "name": name,
        "status": status,
        "returncode": completed.returncode,
        "command": command,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def command_available(executable: str) -> bool:
    return shutil.which(executable) is not None


def status_counts(checks: list[JsonObject]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        status = str(check.get("status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def render_markdown(report: JsonObject) -> str:
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        checks = []
    lines = [
        "# Autonomous Infrastructure Status",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Overall status: `{report['overall_status']}`",
        "",
        "## Static Cloudflare Bindings",
        "",
        "```json",
        json.dumps(report["static_config"], indent=2),
        "```",
        "",
        "## Live Checks",
        "",
        "| check | status | return code |",
        "|---|---|---:|",
    ]
    for check in checks:
        if isinstance(check, dict):
            lines.append(
                f"| {check.get('name', '')} | `{check.get('status', 'UNKNOWN')}` | {check.get('returncode', '')} |"
            )
    lines.extend(
        [
            "",
            "No redeploy was performed. AUTH_BLOCKED means the local terminal lacks usable wrangler credentials; the binding still exists in source configuration.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    checks: list[JsonObject] = []
    checks.append(
        run_command(
            "worker_http",
            ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", WORKER_URL],
            cwd=REPO_ROOT,
            timeout=int(args.timeout),
        )
    )
    if command_available("npx") and WORKER_DIR.is_dir():
        wrangler_prefix = ["npx", "wrangler"]
        checks.extend(
            [
                run_command(
                    "d1_candidate_count",
                    [
                        *wrangler_prefix,
                        "d1",
                        "execute",
                        "prism_metadata",
                        "--remote",
                        "--command",
                        "SELECT COUNT(*) AS n FROM gflownet_candidates",
                        "--json",
                    ],
                    cwd=WORKER_DIR,
                    timeout=int(args.timeout),
                ),
                run_command(
                    "vectorize_info",
                    [*wrangler_prefix, "vectorize", "info", "dkl_latent_space", "--json"],
                    cwd=WORKER_DIR,
                    timeout=int(args.timeout),
                ),
                run_command(
                    "r2_bucket_list",
                    [*wrangler_prefix, "r2", "bucket", "list"],
                    cwd=WORKER_DIR,
                    timeout=int(args.timeout),
                ),
                run_command(
                    "queue_list",
                    [*wrangler_prefix, "queues", "list"],
                    cwd=WORKER_DIR,
                    timeout=int(args.timeout),
                ),
            ]
        )
    else:
        checks.append(
            {
                "name": "wrangler_checks",
                "status": "COMMAND_UNAVAILABLE",
                "command": ["npx", "wrangler"],
                "stdout": "",
                "stderr": "npx unavailable or worker directory missing",
            }
        )
    counts = status_counts(checks)
    overall = "OK" if counts.get("COMMAND_FAILED", 0) == 0 and counts.get("TIMEOUT", 0) == 0 else "DEGRADED"
    if counts.get("AUTH_BLOCKED", 0) > 0:
        overall = "AUTH_BLOCKED"
    report: JsonObject = {
        "schema_version": "PRISM.autonomous_infra_status.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "overall_status": overall,
        "worker_url": WORKER_URL,
        "static_config": static_config(),
        "status_counts": counts,
        "checks": checks,
        "notes": [
            "This check is read-only and does not deploy the Worker.",
            "Static bindings are source-of-truth infrastructure configuration.",
            "Live wrangler checks require authenticated local Cloudflare credentials.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(args.output)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        "autonomous_infra_status "
        f"overall={overall} "
        f"counts={counts} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
