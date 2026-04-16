#!/usr/bin/env python3
"""
[PROVENANCE MODULE - TIER A]

BLAKE3 content-addressed provenance for the PRISM-TWIN-10 pipeline.

Every tool invocation emits a hash-chained provenance record. Records
reference upstream records by hash, forming a Merkle DAG over the full
execution. Any tampering of any intermediate artifact breaks the chain
at that exact point, and a reviewer can independently verify.

Schema: prism-twin-prov-v1
Hash: BLAKE3-256, unkeyed (reviewers can verify without secrets)
Self-hash: BLAKE3 of canonical JSON with self_blake3 field blanked
Canonical form: JSON sorted keys, compact separators, utf-8

Usage:
    with RunContext(target="kras_g12d_apo", stage="2_clean",
                    substage="pdbfixer", output_dir=out, prov_dir=prov,
                    upstream_prov=[prov/"1_download.prov.json"]) as ctx:
        ctx.add_input(Path("7F0W.pdb"), upstream_prov_ref="1_download")
        ctx.set_tool("pdbfixer", argv=["pdbfixer", "7F0W.pdb", "--output",
                                        "7F0W_fixed.pdb"])
        result = ctx.run(ctx.record.command["argv"])
        ctx.add_output(Path("7F0W_fixed.pdb"))
        ctx.set_gate("output_exists", "PASS")
        ctx.set_gate("exit_code_zero", "PASS" if result.returncode == 0 else "FAIL")
        ctx.set_verdict("PASS")

The context manager finalizes the record on exit: computes self_blake3,
writes {stage}.{substage}.prov.json to prov_dir.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from blake3 import blake3

PROV_SCHEMA_VERSION = "prism-twin-prov-v1"
CHUNK_SIZE = 1 << 20  # 1 MB chunks for streaming file hashing


# ─────────────────────────────────────────────────────────────────────
# Hashing primitives
# ─────────────────────────────────────────────────────────────────────

def blake3_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """BLAKE3 hash of a file, streamed to bound memory for large files."""
    h = blake3()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def blake3_bytes(data: bytes) -> str:
    return blake3(data).hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Canonical JSON form for reproducible hashing."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────
# Environment capture
# ─────────────────────────────────────────────────────────────────────

def capture_host() -> Dict[str, Any]:
    """Snapshot reproducibility-relevant host metadata."""
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "kernel": platform.uname().release,
        "os": platform.uname().system,
        "arch": platform.machine(),
        "python": platform.python_version(),
    }
    # CPU model
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        pass
    # GPU + driver + VRAM
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            info["gpu"] = r.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # CUDA toolkit
    try:
        r = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        for line in r.stdout.splitlines():
            if "release" in line.lower() or "V" in line:
                if "Cuda" in line or "release" in line:
                    info["cuda_toolkit"] = line.strip()
                    break
    except FileNotFoundError:
        pass
    # CUDA driver runtime version (from nvidia-smi)
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.stdout.strip():
            info["cuda_driver"] = r.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def capture_tool(name: str, binary_path: Optional[Path] = None,
                 extra_version_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Capture tool binary hash + version."""
    import shutil as _sh
    info: Dict[str, Any] = {"name": name}
    if binary_path is None:
        resolved = _sh.which(name)
        if resolved:
            binary_path = Path(resolved)
    if binary_path and Path(binary_path).exists():
        info["binary_path"] = str(binary_path)
        try:
            info["binary_blake3"] = blake3_file(Path(binary_path))
        except (PermissionError, OSError) as e:
            info["binary_blake3_error"] = str(e)
        version_attempts = [["--version"], ["-V"]]
        if extra_version_args:
            version_attempts.insert(0, extra_version_args)
        for args in version_attempts:
            try:
                r = subprocess.run(
                    [str(binary_path)] + args,
                    capture_output=True, text=True, timeout=5,
                )
                combined = (r.stdout + r.stderr).strip()
                if combined:
                    info["version_output"] = combined[:500]
                    break
            except (subprocess.TimeoutExpired, OSError):
                continue
    else:
        info["binary_path"] = None
        info["binary_blake3"] = None
    return info


# ─────────────────────────────────────────────────────────────────────
# ProvRecord — the immutable provenance record
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ProvRecord:
    schema_version: str = PROV_SCHEMA_VERSION
    target: str = ""
    stage: str = ""
    substage: str = ""
    timestamp_utc: str = ""
    host: Dict[str, Any] = field(default_factory=dict)
    tool: Dict[str, Any] = field(default_factory=dict)
    command: Dict[str, Any] = field(default_factory=dict)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    exec_stats: Dict[str, Any] = field(default_factory=dict)
    gates: Dict[str, str] = field(default_factory=dict)
    upstream_prov: List[str] = field(default_factory=list)
    verdict: str = ""
    notes: List[str] = field(default_factory=list)
    self_blake3: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_self_hash(self) -> str:
        d = self.to_dict()
        d["self_blake3"] = ""
        return blake3_bytes(canonical_json(d))

    def finalize(self) -> "ProvRecord":
        self.self_blake3 = self.compute_self_hash()
        return self

    def write(self, path: Path) -> Path:
        self.finalize()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return path


# ─────────────────────────────────────────────────────────────────────
# RunContext — provenance-wrapped tool invocation
# ─────────────────────────────────────────────────────────────────────

class RunContext:
    """Context manager emitting a hash-chained provenance record per tool run."""

    def __init__(
        self,
        target: str,
        stage: str,
        substage: str,
        output_dir: Path,
        prov_dir: Path,
        upstream_prov: Optional[List[Path]] = None,
    ):
        self.target = target
        self.stage = stage
        self.substage = substage
        self.output_dir = Path(output_dir)
        self.prov_dir = Path(prov_dir)
        self.upstream_prov = [str(p) for p in (upstream_prov or [])]
        self.record = ProvRecord(
            target=target, stage=stage, substage=substage,
            host=capture_host(),
            upstream_prov=self.upstream_prov,
        )
        self._t_start: Optional[float] = None
        self._proc_result: Optional[subprocess.CompletedProcess] = None

    def __enter__(self) -> "RunContext":
        self.record.timestamp_utc = datetime.now(timezone.utc).isoformat()
        self._t_start = time.monotonic()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prov_dir.mkdir(parents=True, exist_ok=True)
        return self

    def add_input(self, path: Path, upstream_prov_ref: Optional[str] = None) -> str:
        """Hash and record an input file. Returns the hash."""
        p = Path(path)
        if not p.exists():
            self.record.inputs.append({
                "path": str(p),
                "size_bytes": 0,
                "blake3": None,
                "present": False,
            })
            return ""
        h = blake3_file(p)
        entry = {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "blake3": h,
        }
        if upstream_prov_ref:
            entry["upstream_prov"] = upstream_prov_ref
        self.record.inputs.append(entry)
        return h

    def set_tool(self, name: str, argv: List[str],
                 binary_path: Optional[Path] = None,
                 extra_version_args: Optional[List[str]] = None):
        """Record tool binary + argv."""
        self.record.tool = capture_tool(name, binary_path, extra_version_args)
        self.record.command = {
            "argv": argv,
            "cwd": str(self.output_dir),
        }

    def set_env(self, env_pairs: Dict[str, str]):
        """Record environment variables relevant to determinism."""
        self.record.command["env_relevant"] = dict(env_pairs)

    def run(
        self,
        argv: List[str],
        timeout: int = 3600,
        env_overrides: Optional[Dict[str, str]] = None,
        stdout_file: Optional[Path] = None,
        stderr_file: Optional[Path] = None,
        cwd_override: Optional[Path] = None,
    ) -> subprocess.CompletedProcess:
        """Execute subprocess. Records stdout/stderr if files given.

        cwd_override: if provided, uses this as the subprocess working
        directory instead of self.output_dir. Needed for the engine which
        resolves PTX paths relative to CWD = repo_root.
        """
        merged_env = os.environ.copy()
        if env_overrides:
            merged_env.update(env_overrides)
            self.record.command.setdefault("env_relevant", {}).update(env_overrides)

        cwd = cwd_override if cwd_override is not None else self.output_dir
        self.record.command["cwd"] = str(cwd)

        stdout_h = open(stdout_file, "w") if stdout_file else subprocess.PIPE
        stderr_h = open(stderr_file, "w") if stderr_file else subprocess.PIPE
        try:
            result = subprocess.run(
                argv,
                stdout=stdout_h,
                stderr=stderr_h,
                text=True,
                timeout=timeout,
                env=merged_env,
                cwd=cwd,
            )
        finally:
            if stdout_file:
                stdout_h.close()
            if stderr_file:
                stderr_h.close()
        self._proc_result = result
        return result

    def add_output(self, path: Path, role: str = "output") -> str:
        """Hash and record an output file."""
        p = Path(path)
        if not p.exists():
            self.record.outputs.append({
                "path": str(p),
                "role": role,
                "size_bytes": 0,
                "blake3": None,
                "present": False,
            })
            return ""
        h = blake3_file(p)
        self.record.outputs.append({
            "path": str(p),
            "role": role,
            "size_bytes": p.stat().st_size,
            "blake3": h,
        })
        return h

    def set_gate(self, name: str, status: str, note: str = ""):
        self.record.gates[name] = f"{status} — {note}" if note else status

    def set_verdict(self, verdict: str):
        self.record.verdict = verdict

    def add_note(self, note: str):
        self.record.notes.append(note)

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.monotonic() - (self._t_start or time.monotonic())
        self.record.exec_stats["wall_sec"] = round(elapsed, 3)
        if self._proc_result is not None:
            self.record.exec_stats["exit_code"] = self._proc_result.returncode
        if exc_type is not None:
            self.record.notes.append(f"Exception: {exc_type.__name__}: {exc_val}")
            if not self.record.verdict:
                self.record.verdict = "FAIL"
        name = f"{self.stage}.{self.substage}.prov.json" if self.substage else f"{self.stage}.prov.json"
        self.record.write(self.prov_dir / name)
        return False  # do not suppress exceptions


# ─────────────────────────────────────────────────────────────────────
# Verification primitives
# ─────────────────────────────────────────────────────────────────────

def verify_record_self_hash(path: Path) -> Dict[str, Any]:
    """Reviewer-side: verify a single provenance record's self_blake3."""
    with open(path) as f:
        data = json.load(f)
    claimed = data.get("self_blake3", "")
    data_copy = dict(data)
    data_copy["self_blake3"] = ""
    actual = blake3_bytes(canonical_json(data_copy))
    return {
        "path": str(path),
        "claimed": claimed,
        "actual": actual,
        "valid": claimed == actual,
    }


def verify_artifact(path: Path, expected_blake3: str) -> Dict[str, Any]:
    """Reviewer-side: verify an artifact file's blake3 matches deposited."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "valid": False, "reason": "file missing"}
    actual = blake3_file(p)
    return {
        "path": str(p),
        "claimed": expected_blake3,
        "actual": actual,
        "valid": actual == expected_blake3,
    }


# ─────────────────────────────────────────────────────────────────────
# Pipeline manifest — DAG root
# ─────────────────────────────────────────────────────────────────────

def write_manifest(
    target: str,
    target_dir: Path,
    prov_records: List[Path],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Emit pipeline_manifest.json referencing all stage provenance records."""
    manifest_dir = target_dir / "prov"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "schema_version": PROV_SCHEMA_VERSION,
        "target": target,
        "target_dir": str(target_dir),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "host": capture_host(),
        "records": [],
    }
    for rec_path in prov_records:
        p = Path(rec_path)
        entry: Dict[str, Any] = {"path": str(p)}
        if p.exists():
            entry["blake3"] = blake3_file(p)
            entry["size_bytes"] = p.stat().st_size
            with open(p) as f:
                d = json.load(f)
            entry["stage"] = d.get("stage")
            entry["substage"] = d.get("substage")
            entry["verdict"] = d.get("verdict")
            entry["self_blake3"] = d.get("self_blake3")
        else:
            entry["present"] = False
        manifest["records"].append(entry)
    if extra:
        manifest["extra"] = extra

    # Compute root hash: BLAKE3 of canonical manifest with root_blake3 blanked
    manifest["root_blake3"] = ""
    root = blake3_bytes(canonical_json(manifest))
    manifest["root_blake3"] = root

    path = manifest_dir / "pipeline_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


if __name__ == "__main__":
    # Self-test: can we write a record and verify its self_hash?
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "input.txt").write_text("hello")
        with RunContext("test", "0_selftest", "demo", tdp / "out", tdp / "prov") as ctx:
            ctx.add_input(tdp / "input.txt")
            ctx.set_tool("echo", ["echo", "hello"])
            ctx.set_gate("ok", "PASS")
            ctx.set_verdict("PASS")
        prov_file = tdp / "prov" / "0_selftest.demo.prov.json"
        result = verify_record_self_hash(prov_file)
        print(f"self-hash verification: {'PASS' if result['valid'] else 'FAIL'}")
        print(f"  claimed: {result['claimed'][:32]}...")
        print(f"  actual:  {result['actual'][:32]}...")
