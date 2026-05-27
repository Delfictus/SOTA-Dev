#!/usr/bin/env python3
"""Audit Python and Rust dependency pinning for hardened release."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_REQ = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^=<>!~]+$")


def _requirements(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        rows.append(
            {
                "file": path.relative_to(REPO_ROOT).as_posix(),
                "line": lineno,
                "requirement": line,
                "pinned": bool(PINNED_REQ.match(line)),
            }
        )
    return rows


def _cargo_lock_packages(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    packages: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line == "[[package]]":
            if current:
                packages.append(current)
            current = {}
        elif line.startswith("name = "):
            current["name"] = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("version = "):
            current["version"] = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("source = "):
            current["source"] = line.split("=", 1)[1].strip().strip('"')
    if current:
        packages.append(current)
    return packages


def _tool_status(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except FileNotFoundError:
        return {"command": command, "status": "UNAVAILABLE_NOT_INSTALLED"}
    except subprocess.TimeoutExpired:
        return {"command": command, "status": "TIMEOUT"}
    return {
        "command": command,
        "status": "PASS" if result.returncode == 0 else "NONZERO",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def audit_dependencies(*, run_security: bool = False) -> dict[str, Any]:
    requirement_files = [REPO_ROOT / "requirements.txt", REPO_ROOT / "runpod_training/requirements.txt"]
    python_requirements = [row for path in requirement_files for row in _requirements(path)]
    floating_python = [row for row in python_requirements if not row["pinned"]]
    cargo_packages = _cargo_lock_packages(REPO_ROOT / "Cargo.lock")
    security = []
    if run_security:
        security.append(_tool_status(["pip-audit"]))
        security.append(_tool_status(["cargo", "audit"]))
    return {
        "schema_version": "PRISM.dependency_pinning_audit.v1",
        "python_requirements": python_requirements,
        "floating_python_requirements": floating_python,
        "cargo_lock": {
            "path": "Cargo.lock",
            "exists": (REPO_ROOT / "Cargo.lock").exists(),
            "package_count": len(cargo_packages),
            "sample": cargo_packages[:20],
            "pinning_authority": "Cargo.lock exact package versions",
        },
        "security_tools": security,
        "summary": {
            "python_requirement_count": len(python_requirements),
            "floating_python_count": len(floating_python),
            "cargo_lock_package_count": len(cargo_packages),
            "cargo_lock_present": (REPO_ROOT / "Cargo.lock").exists(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "release_artifacts/v0.25.0/dependency_pinning_report.json")
    parser.add_argument("--run-security", action="store_true")
    args = parser.parse_args()
    report = audit_dependencies(run_security=bool(args.run_security))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["summary"]
    print(
        "dependency_pinning_audit "
        f"python_requirements={summary['python_requirement_count']} "
        f"floating_python={summary['floating_python_count']} "
        f"cargo_lock_packages={summary['cargo_lock_package_count']} "
        f"report={args.output}"
    )
    return 1 if summary["floating_python_count"] or not summary["cargo_lock_present"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
