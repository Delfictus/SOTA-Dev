#!/usr/bin/env python3
"""D1 — run_metadata.json sidecar writer.

Post-engine, language-agnostic. Emits the sidecar from a completed target's
binding_sites.json + engine argv (captured in engine.stdout.log). The sidecar
is the single source of truth for per-run constants that cannot be recovered
from Arrow alone (lining_cutoff, total_frames) and for the enum decode tables
needed to regenerate legacy per-site JSON field values (type/spike_source/ccns_phase).

Canonical output path: <target_dir>/artifacts/5_engine/<stem>.run_metadata.json

No effect on engine default behavior. Does NOT flip --emit-spike-json.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

# Enum tables — mirror the Rust engine (crates/prism-nhs/src/bin/nhs_rt_full.rs)
# arom_type_name: nhs_rt_full.rs:10780
AROMATIC_TYPE_ENUM = {
    0: "TRP",
    1: "TYR",
    2: "PHE",
    3: "SS",
    4: "BNZ",
    5: "CATION",
    6: "ANION",
}
AROMATIC_TYPE_DEFAULT = "UNK"
# spike_source match arm: nhs_rt_full.rs:10841
SPIKE_SOURCE_ENUM = {
    1: "UV",
    3: "EFP",
    4: "LADD",
    5: "COFIRE",
}
SPIKE_SOURCE_DEFAULT = "LIF"
# phase_label (JSON): nhs_rt_full.rs:10784 — 5 states
CCNS_PHASE_JSON_ENUM = {
    "cold_hold": 0,
    "heating":   1,
    "warm_hold": 2,
    "cooling":   3,
    "cold_return": 4,
}
# ccns_phase_for_step (Arrow): spike_arrow_writer.rs:244 — 3 states
CCNS_PHASE_ARROW_ENUM = {
    0: "cold_hold",
    1: "ramp",   # Arrow label for timestep in [cold_hold, cold_hold+ramp)
    2: "warm_hold_or_later",
}


def _extract_protocol_from_stderr(engine_stderr_log: Path) -> dict:
    """Extract the MASTER protocol captured by the JSON phase_label closure
    (nhs_rt_full.rs:10784). This is the protocol built at line 2677 via
    CryoUvProtocol::fast_35k() + adaptive warm_hold + with_hysteresis().

    The closure uses:
      cold_hold_steps = 14000             (fast_35k default)
      ramp_steps      = 6000              (fast_35k default)
      warm_hold_steps = 15000 + extra_warm (logged as "Adaptive warm_hold: N steps")
      ramp_down_steps = ramp_steps = 6000 (set by with_hysteresis())
      cold_return_steps = cold_hold_steps = 14000 (set by with_hysteresis())

    The per-stream "Phases: cold_hold=X, ramp=Y, warm_hold=Z" lines are NOT
    the master protocol; they are per-stream engine variants. Using them
    produces wrong JSON phase_label regen (Gate A failure mode #1).
    """
    import re
    if not engine_stderr_log.exists():
        return {"error": "engine.stderr.log absent"}
    pat_adaptive = re.compile(r"Adaptive warm_hold:\s*(\d+)\s*steps")
    warm_hold = None
    line_hit = None
    for line in engine_stderr_log.open():
        m = pat_adaptive.search(line)
        if m:
            warm_hold = int(m.group(1))
            line_hit = line.rstrip("\n")
            break
    if warm_hold is None:
        return {"error": "no 'Adaptive warm_hold:' line in engine.stderr.log"}
    # fast_35k defaults + with_hysteresis
    cold_hold = 14000
    ramp = 6000
    ramp_down = 6000     # with_hysteresis: ramp_down = ramp
    cold_return = 14000  # with_hysteresis: cold_return = cold_hold
    return {
        "cold_hold_steps": cold_hold,
        "ramp_steps": ramp,
        "warm_hold_steps": warm_hold,
        "ramp_down_steps": ramp_down,
        "cold_return_steps": cold_return,
        "source": "fast_35k() + Adaptive warm_hold + with_hysteresis()",
        "source_line_warm_hold": line_hit,
        "total_steps_check": cold_hold + ramp + warm_hold + ramp_down + cold_return,
    }


def write_run_metadata(target_dir: Path, stem: str) -> Path:
    eng = target_dir / "artifacts/5_engine"
    bs_path = eng / f"{stem}.binding_sites.json"
    if not bs_path.exists():
        raise SystemExit(f"binding_sites.json absent at {bs_path}")
    bs = json.loads(bs_path.read_text())
    total_steps = bs.get("total_steps_per_stream") or 0
    # frame_index = timestep / 1000 (see nhs_rt_full.rs:10846 serde_json path)
    total_frames = int(total_steps / 1000) if total_steps else 0
    protocol_ref = _extract_protocol_from_stderr(eng / "engine.stderr.log")
    record = {
        "schema_version": "run_metadata_v1",
        "target": target_dir.name,
        "stem": stem,
        "lining_cutoff": bs.get("lining_residue_cutoff_angstroms"),
        "total_frames": total_frames,
        "n_streams": bs.get("n_streams"),
        "total_steps_per_stream": total_steps,
        "consensus_threshold": bs.get("consensus_threshold"),
        "simulation_time_sec": bs.get("simulation_time_sec"),
        "mode": bs.get("mode"),
        "aromatic_type_enum": AROMATIC_TYPE_ENUM,
        "aromatic_type_default": AROMATIC_TYPE_DEFAULT,
        "spike_source_enum": SPIKE_SOURCE_ENUM,
        "spike_source_default": SPIKE_SOURCE_DEFAULT,
        "ccns_phase_json_enum": CCNS_PHASE_JSON_ENUM,
        "ccns_phase_arrow_enum": CCNS_PHASE_ARROW_ENUM,
        "reference_protocol_for_json_phase_label": protocol_ref,
        "phase_label_rule_json": (
            "if ts < cold_hold: 'cold_hold'; "
            "elif ts < cold_hold+ramp: 'heating'; "
            "elif ts < cold_hold+ramp+warm_hold: 'warm_hold'; "
            "elif ts < cold_hold+ramp+warm_hold+ramp_down: 'cooling'; "
            "else: 'cold_return'"
        ),
        "enum_source_reference": (
            "crates/prism-nhs/src/bin/nhs_rt_full.rs "
            "arom_type_name (line 10780), phase_label (line 10784), "
            "spike_source match (line 10841); "
            "crates/prism-nhs/src/spike_arrow_writer.rs ccns_phase_for_step (line 244)"
        ),
    }
    out = eng / f"{stem}.run_metadata.json"
    out.write_text(json.dumps(record, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", required=True, type=Path)
    ap.add_argument("--stem", required=True)
    args = ap.parse_args()
    p = write_run_metadata(args.target_dir, args.stem)
    print(f"wrote: {p}")


if __name__ == "__main__":
    main()
