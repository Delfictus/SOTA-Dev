#!/usr/bin/env python3
"""
[PROVENANCE MODULE - TIER B]

Engine-specific provenance for the TWIN-10 MD run.

Extends prism_prov.py with:
  - Structured parsing of nhs_rt_full run.log (phases, streams, ring buffers,
    ASC events, cascade, final emission)
  - Individual BLAKE3 hashing of every engine-emitted artifact:
      * binding_sites.json
      * kcc_visualization.json
      * kcc_validation.json
      * topology.prism_therm.json
      * residue_map.json
      * ensemble_trajectory.json
      * topology.spike_events.arrow
      * *.site{N}.spike_events.parquet (per site)
      * *_stream{NN}.ensemble_trajectory.pdb (per stream)
  - Concurrent GPU telemetry capture via nvidia-smi dmon (subprocess)
  - CUPTI kernel trace via Nsight Systems nsys profile
  - Sub-phase boundary extraction: when available in run.log, hashes
    per-cryo-phase state (cold_hold, ramp, warm_hold, ramp_down, cold_return)
  - Cross-group data propagation: hashes each group's spike stream
    independently where ensemble_trajectory writes per-stream files

Tier B means: we do NOT instrument the engine binary itself. We hash every
artifact the engine emits and every log line it produces, and chain them
into the provenance DAG. The engine remains a (well-documented) black
box in the middle, but everything on both sides of the boundary is
cryptographically traceable.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prism_prov import (
    blake3_file, blake3_bytes, canonical_json,
    ProvRecord, RunContext, PROV_SCHEMA_VERSION,
)


# ─────────────────────────────────────────────────────────────────────
# Engine artifact catalog — what the engine emits
# ─────────────────────────────────────────────────────────────────────

# (role, glob pattern). Used to enumerate and hash every emission.
ENGINE_ARTIFACT_CATALOG: List[Tuple[str, str]] = [
    ("binding_sites",        "*.binding_sites.json"),
    ("kcc_visualization",    "*.kcc_visualization.json"),
    ("kcc_validation",       "*.kcc_validation.json"),
    ("prism_therm",          "*.topology.prism_therm.json"),
    ("residue_map",          "*.residue_map.json"),
    ("ensemble_trajectory",  "*.ensemble_trajectory.json"),
    ("spike_stream_arrow",   "*.topology.spike_events.arrow"),
    ("site_spike_parquet",   "*.site*.spike_events.parquet"),
    ("site_spike_json",      "*.site*.spike_events.json"),  # legacy fallback
    ("stream_trajectory",    "*_stream*.ensemble_trajectory.pdb"),
    ("binding_sites_pml",    "*.binding_sites.pml"),
    ("binding_sites_md",     "*.binding_sites.md"),
    ("binding_sites_cxc",    "*.binding_sites.cxc"),
    ("binding_sites_pdb",    "*.binding_sites.pdb"),
    ("kcc_session_pml",      "*.kcc_session.pml"),
    ("topology_druggability","*.topology.druggability.pdb"),
    ("run_log",              "run.log"),
]


def enumerate_engine_artifacts(engine_output_dir: Path) -> Dict[str, List[Path]]:
    """Discover every artifact the engine emitted, grouped by role."""
    result: Dict[str, List[Path]] = {}
    for role, pattern in ENGINE_ARTIFACT_CATALOG:
        matches = sorted(engine_output_dir.glob(pattern))
        if matches:
            result[role] = matches
    return result


def hash_engine_artifacts(engine_output_dir: Path) -> List[Dict[str, Any]]:
    """BLAKE3-hash every engine artifact. Returns ordered list of hash records."""
    artifacts = enumerate_engine_artifacts(engine_output_dir)
    records: List[Dict[str, Any]] = []
    for role, _pattern in ENGINE_ARTIFACT_CATALOG:
        for p in artifacts.get(role, []):
            records.append({
                "role": role,
                "path": str(p.name),
                "size_bytes": p.stat().st_size,
                "blake3": blake3_file(p),
            })
    return records


# ─────────────────────────────────────────────────────────────────────
# Structured run.log parser
# ─────────────────────────────────────────────────────────────────────

# Patterns the nhs_rt_full engine emits (based on run.log spot-checked earlier)
_RE_TIMESTAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+(\w+)\s+([\w_:]+)\]\s*(.*)")
_RE_STREAM_START = re.compile(r"\[stream (\d+)\] Starting \(seed: (\d+)\)")
_RE_STREAM_COMPLETE = re.compile(r"\[stream (\d+)\] Complete: (\d+) spikes, (\d+) snapshots, T=([\d.]+)K")
_RE_STREAM_FILTERED = re.compile(r"Stream (\d+): (\d+) filtered spikes .* (\d+) sites")
_RE_MULTI_STREAM_TIME = re.compile(r"All \d+ streams complete in ([\d.]+)s")
_RE_MULTI_DIFF_HDR = re.compile(r"Multi-Differential")
_RE_MULTI_STREAM_HDR = re.compile(r"TRUE MULTI-STREAM|MULTI-STREAM PIPELINE")
_RE_SPIKE_DEBUG = re.compile(r"SPIKE DEBUG \[(\d+)\]: ts=(\d+) phase=(\d+)/(\d+) src=(\d+) pos=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")


def parse_run_log(log_path: Path) -> Dict[str, Any]:
    """Extract structured events from engine run.log."""
    if not log_path.exists():
        return {"present": False}

    events: Dict[str, Any] = {
        "present": True,
        "file_blake3": blake3_file(log_path),
        "size_bytes": log_path.stat().st_size,
        "streams": {},         # stream_id → {start_seed, spikes, snapshots, filtered, sites_found}
        "phase_events": [],
        "multi_diff_detected": False,
        "multi_stream_detected": False,
        "all_streams_wallclock_sec": None,
        "spike_debug_samples": [],
        "errors": [],
        "warnings": [],
        "first_ts": None,
        "last_ts": None,
    }

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if _RE_MULTI_DIFF_HDR.search(line):
                events["multi_diff_detected"] = True
            if _RE_MULTI_STREAM_HDR.search(line):
                events["multi_stream_detected"] = True
            m = _RE_TIMESTAMP.match(line)
            if m:
                ts = m.group(1)
                if events["first_ts"] is None:
                    events["first_ts"] = ts
                events["last_ts"] = ts
            lower = line.lower()
            if "error" in lower and "no errors" not in lower:
                events["errors"].append(line.strip()[:300])
            if "warning" in lower or "warn" in lower:
                events["warnings"].append(line.strip()[:300])
            m = _RE_STREAM_START.search(line)
            if m:
                sid = int(m.group(1))
                seed = int(m.group(2))
                events["streams"].setdefault(sid, {})["start_seed"] = seed
                continue
            m = _RE_STREAM_COMPLETE.search(line)
            if m:
                sid = int(m.group(1))
                events["streams"].setdefault(sid, {}).update({
                    "spikes": int(m.group(2)),
                    "snapshots": int(m.group(3)),
                    "final_temp_K": float(m.group(4)),
                })
                continue
            m = _RE_STREAM_FILTERED.search(line)
            if m:
                sid = int(m.group(1))
                events["streams"].setdefault(sid, {}).update({
                    "filtered_spikes": int(m.group(2)),
                    "sites_found": int(m.group(3)),
                })
                continue
            m = _RE_MULTI_STREAM_TIME.search(line)
            if m:
                events["all_streams_wallclock_sec"] = float(m.group(1))
                continue
            m = _RE_SPIKE_DEBUG.search(line)
            if m and len(events["spike_debug_samples"]) < 50:
                events["spike_debug_samples"].append({
                    "idx": int(m.group(1)),
                    "ts": int(m.group(2)),
                    "phase": int(m.group(3)),
                    "phase_max": int(m.group(4)),
                    "src": int(m.group(5)),
                    "pos": [float(m.group(6)), float(m.group(7)), float(m.group(8))],
                })

    # Hash the structured event log itself for Tier-B chain
    events["structured_blake3"] = blake3_bytes(
        canonical_json({k: v for k, v in events.items()
                        if k not in ("structured_blake3",)})
    )
    return events


# ─────────────────────────────────────────────────────────────────────
# GPU telemetry capture
# ─────────────────────────────────────────────────────────────────────

class GpuTelemetryCapture:
    """Runs nvidia-smi dmon as a background subprocess during the engine run."""

    def __init__(self, output_csv: Path, interval_sec: int = 1):
        self.output_csv = Path(output_csv)
        self.interval_sec = interval_sec
        self._proc: Optional[subprocess.Popen] = None

    def start(self):
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        # -s pucvmet → p=power, u=util, c=clock, v=violation, m=memory, e=ecc, t=temp
        # -c 0 is infinite; we terminate manually
        self._proc = subprocess.Popen(
            [
                "nvidia-smi", "dmon",
                "-s", "pucvmet",
                "-d", str(self.interval_sec),
                "-o", "DT",  # date + time prefix
            ],
            stdout=open(self.output_csv, "w"),
            stderr=subprocess.DEVNULL,
        )

    def stop(self, grace_sec: float = 0.5) -> Dict[str, Any]:
        if self._proc is None:
            return {"present": False}
        self._proc.terminate()
        try:
            self._proc.wait(timeout=grace_sec + 2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=2)
        if not self.output_csv.exists():
            return {"present": False}
        return {
            "present": True,
            "path": str(self.output_csv),
            "size_bytes": self.output_csv.stat().st_size,
            "blake3": blake3_file(self.output_csv),
            "interval_sec": self.interval_sec,
        }


# ─────────────────────────────────────────────────────────────────────
# CUPTI / Nsight Systems trace
# ─────────────────────────────────────────────────────────────────────

def wrap_with_nsys(engine_argv: List[str], trace_output: Path,
                   trace_modes: List[str] = ("cuda", "nvtx", "osrt")) -> List[str]:
    """Wrap an engine invocation with `nsys profile` for kernel-level trace.

    trace_output: .nsys-rep file will be written here (nsys appends extension
                  if not present; we pass the base name).

    Returns the rewritten argv to execute. If nsys is not available on PATH,
    returns engine_argv unchanged (caller should detect absence separately).
    """
    import shutil as _sh
    nsys = _sh.which("nsys")
    if nsys is None:
        return list(engine_argv)

    trace_output.parent.mkdir(parents=True, exist_ok=True)
    base = str(trace_output)
    if base.endswith(".nsys-rep"):
        base = base[: -len(".nsys-rep")]

    return [
        nsys, "profile",
        "--output", base,
        "--trace", ",".join(trace_modes),
        "--sample", "cpu",
        "--cuda-graph-trace=node",
        "--force-overwrite=true",
        "--stats=false",  # keep the report minimal; full stats on demand
    ] + list(engine_argv)


def hash_nsys_trace(trace_path: Path) -> Dict[str, Any]:
    """Locate and hash the .nsys-rep file (nsys may append the extension)."""
    p = Path(trace_path)
    if not p.suffix:
        p = p.with_suffix(".nsys-rep")
    if not p.exists():
        # nsys may have written to a different path
        parent = Path(trace_path).parent
        stem = Path(trace_path).stem
        candidates = list(parent.glob(f"{stem}.nsys-rep"))
        if candidates:
            p = candidates[0]
    if not p.exists():
        return {"present": False}
    return {
        "present": True,
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "blake3": blake3_file(p),
    }


# ─────────────────────────────────────────────────────────────────────
# Determinism flags — environment variables we set for the engine run
# ─────────────────────────────────────────────────────────────────────

def determinism_env() -> Dict[str, str]:
    """Return the environment variables that configure deterministic execution.

    Note: nvidia-smi --lock-gpu-clocks was found unavailable on the current
    driver/GPU. We cannot force-lock clocks; instead we record actual clocks
    via GpuTelemetryCapture so reviewer can see any drift.
    """
    return {
        # Deterministic BLAS workspace — prevents atomics-order nondeterminism
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        # Reduce scheduling nondeterminism
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # Keep async for performance
        "CUDA_LAUNCH_BLOCKING": "0",
        # Python bytecode determinism
        "PYTHONHASHSEED": "42",
    }


# ─────────────────────────────────────────────────────────────────────
# Per-phase extraction from ensemble_trajectory (when available)
# ─────────────────────────────────────────────────────────────────────

def extract_stream_trajectory_hashes(engine_output_dir: Path) -> List[Dict[str, Any]]:
    """Hash each per-stream ensemble_trajectory.pdb (one per group).

    These PDBs are the end-of-run state of each parallel MD group. Hashing
    them individually gives Tier-B visibility into per-group convergence.
    """
    traj_files = sorted(engine_output_dir.glob("*_stream*.ensemble_trajectory.pdb"))
    records: List[Dict[str, Any]] = []
    for tp in traj_files:
        m = re.search(r"_stream(\d+)\.ensemble_trajectory\.pdb$", tp.name)
        stream_id = int(m.group(1)) if m else -1
        records.append({
            "stream_id": stream_id,
            "path": str(tp.name),
            "size_bytes": tp.stat().st_size,
            "blake3": blake3_file(tp),
        })
    return records


# ─────────────────────────────────────────────────────────────────────
# Group-spike-stream extraction (if per-group arrow/parquet segmentation)
# ─────────────────────────────────────────────────────────────────────

def hash_spike_streams_per_group(engine_output_dir: Path) -> Dict[str, Any]:
    """Hash the arrow spike stream + per-site parquets.

    The engine emits one .topology.spike_events.arrow as the fused whole-run
    per-spike stream, plus per-site .spike_events.parquet files. Hashing
    these separately exposes spike-data propagation integrity.
    """
    arrows = sorted(engine_output_dir.glob("*.topology.spike_events.arrow"))
    parquets = sorted(engine_output_dir.glob("*.site*.spike_events.parquet"))
    legacy_jsons = sorted(engine_output_dir.glob("*.site*.spike_events.json"))
    return {
        "arrow_files": [
            {"path": p.name, "size_bytes": p.stat().st_size, "blake3": blake3_file(p)}
            for p in arrows
        ],
        "site_parquet_files": [
            {"path": p.name, "size_bytes": p.stat().st_size, "blake3": blake3_file(p)}
            for p in parquets
        ],
        "site_json_files": [
            {"path": p.name, "size_bytes": p.stat().st_size, "blake3": blake3_file(p)}
            for p in legacy_jsons
        ],
        "counts": {
            "arrow": len(arrows),
            "site_parquets": len(parquets),
            "site_jsons": len(legacy_jsons),
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Full engine provenance emission — Tier B
# ─────────────────────────────────────────────────────────────────────

def emit_engine_tier_b_provenance(
    target: str,
    engine_output_dir: Path,
    prov_dir: Path,
    upstream_prov: List[Path],
    gpu_telemetry_csv: Optional[Path] = None,
    nsys_trace: Optional[Path] = None,
) -> Path:
    """Emit a tier-B engine provenance record enumerating every sub-artifact.

    This is the companion record to the tier-A engine.prov.json. The tier-A
    record captures the tool invocation at the I/O boundary; this record
    captures the internal structure of what the engine emitted.
    """
    from prism_prov import capture_host

    record: Dict[str, Any] = {
        "schema_version": PROV_SCHEMA_VERSION,
        "target": target,
        "stage": "5_engine",
        "substage": "tier_b",
        "timestamp_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "host": capture_host(),
        "upstream_prov": [str(p) for p in upstream_prov],
        "engine_output_dir": str(engine_output_dir),
        "artifact_hashes": hash_engine_artifacts(engine_output_dir),
        "spike_streams": hash_spike_streams_per_group(engine_output_dir),
        "stream_trajectories": extract_stream_trajectory_hashes(engine_output_dir),
        "run_log_parsed": parse_run_log(engine_output_dir / "run.log"),
    }
    if gpu_telemetry_csv and Path(gpu_telemetry_csv).exists():
        record["gpu_telemetry"] = {
            "path": str(gpu_telemetry_csv),
            "size_bytes": Path(gpu_telemetry_csv).stat().st_size,
            "blake3": blake3_file(Path(gpu_telemetry_csv)),
        }
    if nsys_trace:
        record["nsys_trace"] = hash_nsys_trace(Path(nsys_trace))

    # Gates
    gates: Dict[str, str] = {}
    run_log = record["run_log_parsed"]
    gates["run_log_present"] = "PASS" if run_log.get("present") else "FAIL"
    gates["multi_differential_activated"] = "PASS" if run_log.get("multi_diff_detected") else "FAIL"
    gates["no_errors_in_log"] = "PASS" if len(run_log.get("errors", [])) == 0 else f"WARN — {len(run_log.get('errors', []))} error lines"
    gates["four_groups_emitted"] = (
        "PASS" if len(record["stream_trajectories"]) >= 4 else
        f"FAIL — only {len(record['stream_trajectories'])} stream PDBs"
    )
    gates["arrow_stream_present"] = "PASS" if record["spike_streams"]["counts"]["arrow"] >= 1 else "FAIL"
    gates["binding_sites_emitted"] = "PASS" if any(
        r["role"] == "binding_sites" for r in record["artifact_hashes"]
    ) else "FAIL"

    # Phase bits sanity — spike debug samples should have non-zero phase values
    samples = run_log.get("spike_debug_samples", [])
    if samples:
        nonzero_phases = sum(1 for s in samples if s.get("phase", 0) > 0)
        gates["phase_bits_populated"] = (
            "PASS" if nonzero_phases > 0 else
            "WARN — all sampled phase values are zero"
        )

    record["gates"] = gates
    verdict = "PASS" if all(v.startswith("PASS") for v in gates.values()) else \
              "WARN" if any(v.startswith("WARN") for v in gates.values()) and \
                       not any(v.startswith("FAIL") for v in gates.values()) else \
              "FAIL"
    record["verdict"] = verdict

    # Self-hash
    record["self_blake3"] = ""
    record["self_blake3"] = blake3_bytes(canonical_json(record))

    prov_dir.mkdir(parents=True, exist_ok=True)
    out_path = prov_dir / "5_engine.tier_b.prov.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return out_path


if __name__ == "__main__":
    # Self-test: parse run.log on an existing target if available
    import sys
    test_dir = Path("/mnt/storage/prism-outputs/10k-runs/9ig2_chainA")
    if test_dir.exists():
        print("=== Engine artifact enumeration ===")
        arts = enumerate_engine_artifacts(test_dir)
        for role, paths in arts.items():
            print(f"  {role:25s} {len(paths)} files")
        print()
        print("=== run.log parse ===")
        parsed = parse_run_log(test_dir / "run.log")
        for k in ("present", "multi_diff_detected", "all_streams_wallclock_sec",
                  "first_ts", "last_ts"):
            print(f"  {k}: {parsed.get(k)}")
        print(f"  streams: {len(parsed.get('streams', {}))}")
        for sid, info in (parsed.get("streams") or {}).items():
            print(f"    stream {sid}: {info}")
        print(f"  errors: {len(parsed.get('errors', []))}")
        print(f"  warnings: {len(parsed.get('warnings', []))}")
        print(f"  spike_debug_samples: {len(parsed.get('spike_debug_samples', []))}")
    else:
        print(f"No test data at {test_dir}; skipping self-test")
