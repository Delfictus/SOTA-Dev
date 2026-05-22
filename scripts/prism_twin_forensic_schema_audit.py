#!/usr/bin/env python3
"""Forensic schema and ontology audit for the GLP-1R aleniglipron PRISM run.

This script inventories the PRISM Twin / Prism4D run artifacts that fed the
PRISM-DSTW aleniglipron campaign, decodes the documented binary envelopes, and
separates raw engine evidence from Path-B materialization, DSTW ingestion, and
SAR-facing derivative outputs.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - audit still writes useful partial output.
    pq = None


DSTW_ROOT = Path("/home/diddy/Desktop/PRISM-DSTW/prism-dstw-calibration")
BIO_ROOT = Path("/home/diddy/Desktop/Prism4D-bio")
RUN_ROOT = Path("/mnt/storage/prism4d_runs/glp1r_aleniglipron")
CAMPAIGN_ROOT = DSTW_ROOT / "campaigns/glp1r_aleniglipron"
OUT_MD = CAMPAIGN_ROOT / "PRISM_Twin_Forensic_Output_Schema_and_Spike_Ontology.md"
OUT_FACTPACK = (
    CAMPAIGN_ROOT / "PRISM_Twin_Forensic_Output_Schema_and_Spike_Ontology.factpack.json"
)

RUN_DIRS = {
    "inactive_5vex": RUN_ROOT / "wt_prime_5vex",
    "active_6x1a": RUN_ROOT / "wt_prime_6x1a",
}

SOURCE_CONTRACTS = {
    "GpuSpikeEvent": BIO_ROOT / "crates/prism-nhs/src/fused_engine.rs",
    "md_evidence_envelopes": BIO_ROOT / "crates/prism-nhs/src/bin/nhs_rt_full.rs",
    "canonical_spike_arrow": BIO_ROOT / "crates/prism-nhs/src/spike_arrow_writer.rs",
    "pathb_dstw_exporter": BIO_ROOT
    / "crates/prism-nhs/src/bin/dstw_export_wt_pathb.rs",
    "dstw_tso_handshake": DSTW_ROOT / "00_registry/prism_tso_handshake.yml",
    "dstw_tso_contract": DSTW_ROOT / "src/prism_dstw/tso/contract.py",
    "dstw_tso_field": DSTW_ROOT / "src/prism_dstw/manifold/tso_field.py",
}

SPIKE_BIN_SCHEMA = [
    ("timestep", "i32", "MD engine step number; converts to nominal time through ghost_time_map dt."),
    ("voxel_idx", "i32", "Engine voxel identifier in the simulation grid."),
    ("position[3]", "f32[3]", "Spike center in engine coordinates for this run/stream."),
    ("intensity", "f32", "Spike amplitude/intensity."),
    ("nearby_residues[8]", "i32[8]", "Nearby topology residue ids padded to eight slots."),
    ("n_residues", "i32", "Number of valid nearby_residues entries."),
    ("spike_source", "i32", "Mechanism/source code: writer maps UV/EFP/LADD/COFIRE/LIF classes."),
    ("wavelength_nm", "f32", "UV wavelength; zero for non-UV/LIF events."),
    ("aromatic_type", "i32", "0 TRP, 1 TYR, 2 PHE, 3 disulfide, -1 none."),
    ("aromatic_residue_id", "i32", "Closest excited aromatic residue, or -1."),
    ("water_density", "f32", "Local water-density signal around the spike."),
    ("vibrational_energy", "f32", "Deposited vibrational/UV energy where present."),
    ("n_nearby_excited", "i32", "Nearby excited aromatic count."),
    ("wd_change", "f32", "Absolute water density change versus prior sampled state."),
    ("phase_bits", "u32", "10-bit CCNS phase angle/protocol phase encoding."),
]

CANONICAL_SPIKE_ARROW_SCHEMA = [
    ("spike_id", "uint64", "Global sequential spike id."),
    ("replica_seed", "uint64", "Run replica seed."),
    ("stream_id", "uint8", "Engine stream index."),
    ("group_id", "uint8", "TWIN group id, e.g. TS/EQ/UV/HY grouping."),
    ("chunk_idx", "uint16", "Host-derived chunk index."),
    ("voxel_idx", "int32", "Engine voxel identifier."),
    ("timestep", "int32", "MD engine step number."),
    ("frame_index", "uint16", "Back-compat frame index, derived from timestep."),
    ("x", "float32", "Spike x coordinate."),
    ("y", "float32", "Spike y coordinate."),
    ("z", "float32", "Spike z coordinate."),
    ("intensity", "float32", "Spike amplitude/intensity."),
    ("spike_source", "int32", "Raw numeric source code."),
    ("mechanism_tag", "utf8", "Stable mechanism class for downstream ML."),
    ("aromatic_type", "int32", "Raw aromatic type code."),
    ("aromatic_residue_id", "int32", "Closest aromatic residue id."),
    ("phase_bits", "uint32", "CCNS phase bits."),
    ("n_residues", "uint8", "Nearby residue count."),
    ("nearby_residues", "fixed_size_list<int32,8>", "Eight nearby residue slots."),
    ("n_nearby_excited", "uint8", "Nearby excited aromatic count."),
    ("vibrational_energy", "float32", "Deposited vibrational/UV energy."),
    ("water_density", "float32", "Local water-density signal."),
    ("wd_change", "float32", "Water-density change feature."),
    ("wavelength_nm", "float32", "Excitation wavelength."),
    ("ccns_phase", "uint8", "Cold/ramp/warm/cooling phase label."),
    ("site_id", "int32", "Consensus site assignment, -1 for background."),
    ("nearest_site_id", "int32", "Nearest consensus site id."),
    ("nearest_site_dist", "float32", "Distance to nearest consensus site."),
    ("background_class", "uint8", "Primary/bulk/surface/near-miss/relabel bucket."),
    ("burial_score", "float32", "Atom-density burial proxy."),
    ("intensity_percentile", "uint8", "Per-channel intensity percentile."),
]

ENGINE_FILE_ONTOLOGY = [
    {
        "family": "spikes.bin",
        "class": "raw_event_stream",
        "schema": "PRSPK001 envelope containing repr(C, align(32)) GpuSpikeEvent records.",
        "represents": "Per-spike x/y/z/t/amplitude/source/residue/hydration/phase evidence.",
        "operationalized_now": "Partially, after Path-B materialization and residue graph reduction.",
    },
    {
        "family": "signal_grid.bin",
        "class": "voxel_signal_grid",
        "schema": "PRSGD001 envelope; grid_dim, voxel_count, then four int32 voxel grids.",
        "represents": "Voxel hit counts, coupled spike grid, primary residue id, primary residue count.",
        "operationalized_now": "Partially in materialized binding-site support; not as DSTW voxel TSO.",
    },
    {
        "family": "kcc_v2full.bin",
        "class": "per_residue_causal_kinematic_field",
        "schema": "PRKCC001 envelope with named f32/u32 per-residue fields.",
        "represents": "Temporal correlation, direction score, motion efficiency, causal lag, motion vectors, active-causal flags.",
        "operationalized_now": "Partially as materialized site kcc_driver support and Path-B residue lift.",
    },
    {
        "family": "bocpd.jsonl",
        "class": "temporal_changepoint_stream",
        "schema": "JSON lines with frame/chunk/stream observations and posterior reset/run-length statistics.",
        "represents": "Spike-rate change point and burst segmentation evidence.",
        "operationalized_now": "Present on disk; materialization report says not used in this commit.",
    },
    {
        "family": "protocol_state.json",
        "class": "run_protocol_state",
        "schema": "JSON protocol temperatures, phases, dt, steering, focus residues, processed spike counts.",
        "represents": "Thermal/steering lifecycle context for interpreting spike time and phase.",
        "operationalized_now": "Loaded for run configuration; not fully propagated as DSTW temporal features.",
    },
    {
        "family": "noise_floor.json",
        "class": "per_stream_noise_model",
        "schema": "JSON mu/sigma/samples for six noise-floor channels.",
        "represents": "Background/noise calibration for spike significance and teacher features.",
        "operationalized_now": "Required for post-MD teacher pack, but not present in current DSTW SAR payload.",
    },
    {
        "family": "adaptive_dt.bin",
        "class": "raw_time_step_sidecar",
        "schema": "Raw binary sidecar, no PRISM envelope detected in this run.",
        "represents": "Adaptive timestep information where parsed by producer/consumer.",
        "operationalized_now": "Present but deferred/not parsed in materialization completeness report.",
    },
    {
        "family": "asc_vectors.bin",
        "class": "raw_orientation_vector_sidecar",
        "schema": "Raw binary sidecar, no PRISM envelope detected in this run.",
        "represents": "Orientation/alignment vector evidence.",
        "operationalized_now": "Present but deferred/not parsed in materialization completeness report.",
    },
    {
        "family": "forces_final.bin",
        "class": "raw_force_vector_sidecar",
        "schema": "Raw binary sidecar, no PRISM envelope detected in this run.",
        "represents": "Final force/directional sidecar evidence.",
        "operationalized_now": "Present but deferred/not parsed in materialization completeness report.",
    },
    {
        "family": "warp_matrix.bin",
        "class": "raw_alignment_warp_sidecar",
        "schema": "Raw binary sidecar, no PRISM envelope detected in this run.",
        "represents": "Warp/SO3 alignment matrix evidence for spatial registration.",
        "operationalized_now": "Present but deferred/not parsed; no dynamic aligned voxel field admitted to DSTW.",
    },
    {
        "family": "binding_sites.materialized.json",
        "class": "pathb_materialized_site_register",
        "schema": "JSON site register with centroids, lining residues, spike support, phase support, score components.",
        "represents": "Post-MD binding-site/materialized pocket hypotheses derived from raw MD evidence.",
        "operationalized_now": "Yes, via Path-B exporter into residue physics/contact graph and SAR pocket accessibility.",
    },
]


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    value = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:,.2f} {unit}" if unit != "B" else f"{int(value):,} B"
        value /= 1024.0
    return f"{value:,.2f} TB"


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def parquet_schema(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return item
    item["size_bytes"] = path.stat().st_size
    if pq is None:
        item["error"] = "pyarrow unavailable"
        return item
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        item.update(
            {
                "num_rows": pf.metadata.num_rows,
                "num_columns": pf.metadata.num_columns,
                "columns": [
                    {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                    for field in schema
                ],
            }
        )
    except Exception as exc:
        item["error"] = repr(exc)
    return item


def classify_run_file(path: Path) -> str:
    name = path.name
    stream_match = re.search(r"_stream\d+_(.+)$", name)
    if stream_match:
        return stream_match.group(1)
    if re.match(r"prism_v2_.+_\d+\.bin$", name):
        return "prism_v2.bin"
    if re.match(r"prism_v2_.+_\d+\.bin\.audit\.json$", name):
        return "prism_v2.audit.json"
    return name


def parse_envelope_header(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    out["size_bytes"] = path.stat().st_size
    try:
        with path.open("rb") as fh:
            magic = fh.read(8)
            out["magic"] = magic.decode("ascii", errors="replace")
            if magic not in {b"PRSPK001", b"PRSGD001", b"PRKCC001"}:
                out["enveloped"] = False
                fh.seek(0)
                out["first_32_bytes_hex"] = fh.read(32).hex()
                return out
            schema_version, endian_marker, stream_id = struct.unpack("<III", fh.read(12))
            run_len = struct.unpack("<Q", fh.read(8))[0]
            run_id = fh.read(run_len).decode("utf-8", errors="replace")
            stem_len = struct.unpack("<Q", fh.read(8))[0]
            stem = fh.read(stem_len).decode("utf-8", errors="replace")
            record_count, byte_stride, payload_size = struct.unpack("<QQQ", fh.read(24))
            payload_offset = fh.tell()
            out.update(
                {
                    "enveloped": True,
                    "schema_version": schema_version,
                    "endian_marker_hex": hex(endian_marker),
                    "stream_id": stream_id,
                    "run_id": run_id,
                    "stem": stem,
                    "record_count": record_count,
                    "byte_stride": byte_stride,
                    "payload_size": payload_size,
                    "payload_offset": payload_offset,
                }
            )
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def parse_first_spike(path: Path) -> dict[str, Any]:
    header = parse_envelope_header(path)
    if not header.get("enveloped") or header.get("magic") != "PRSPK001":
        return {"header": header}
    fmt = "<ii3ff8iiifiiffifI"
    size = struct.calcsize(fmt)
    try:
        with path.open("rb") as fh:
            fh.seek(int(header["payload_offset"]))
            raw = fh.read(size)
        vals = struct.unpack(fmt, raw)
        first = {
            "timestep": vals[0],
            "voxel_idx": vals[1],
            "position": [vals[2], vals[3], vals[4]],
            "intensity": vals[5],
            "nearby_residues": list(vals[6:14]),
            "n_residues": vals[14],
            "spike_source": vals[15],
            "wavelength_nm": vals[16],
            "aromatic_type": vals[17],
            "aromatic_residue_id": vals[18],
            "water_density": vals[19],
            "vibrational_energy": vals[20],
            "n_nearby_excited": vals[21],
            "wd_change": vals[22],
            "phase_bits": vals[23],
        }
        return {"header": header, "first_record": first}
    except Exception as exc:
        return {"header": header, "error": repr(exc)}


def parse_signal_grid(path: Path) -> dict[str, Any]:
    header = parse_envelope_header(path)
    out = {"header": header}
    if not header.get("enveloped") or header.get("magic") != "PRSGD001":
        return out
    try:
        with path.open("rb") as fh:
            fh.seek(int(header["payload_offset"]))
            grid_dim, voxel_count = struct.unpack("<QQ", fh.read(16))
        out["payload_prefix"] = {"grid_dim": grid_dim, "voxel_count": voxel_count}
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def parse_kcc(path: Path) -> dict[str, Any]:
    header = parse_envelope_header(path)
    out = {"header": header, "fields": []}
    if not header.get("enveloped") or header.get("magic") != "PRKCC001":
        return out
    try:
        with path.open("rb") as fh:
            fh.seek(int(header["payload_offset"]))
            n_residues, field_count = struct.unpack("<QQ", fh.read(16))
            out["n_residues"] = n_residues
            out["field_count"] = field_count
            for _ in range(field_count):
                name_len = struct.unpack("<Q", fh.read(8))[0]
                name = fh.read(name_len).decode("utf-8", errors="replace")
                dtype_code = struct.unpack("<B", fh.read(1))[0]
                section_size = struct.unpack("<Q", fh.read(8))[0]
                fh.seek(section_size, 1)
                out["fields"].append(
                    {
                        "name": name,
                        "dtype": {1: "f32", 2: "u32"}.get(dtype_code, f"unknown({dtype_code})"),
                        "section_size_bytes": section_size,
                    }
                )
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def summarize_binding_sites(path: Path) -> dict[str, Any]:
    data = read_json(path)
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if data is None:
        return out
    sites = data.get("binding_sites") or []
    out.update(
        {
            "schema_kind": data.get("schema_kind"),
            "schema_version": data.get("schema_version"),
            "status": data.get("status"),
            "site_count": data.get("site_count"),
            "n_raw_peaks": data.get("n_raw_peaks"),
            "n_consolidated_regions": data.get("n_consolidated_regions"),
            "ground_truth_status": data.get("ground_truth_status"),
            "missing_fields": data.get("missing_fields"),
            "ranking_methodology": data.get("ranking_methodology"),
        }
    )
    if sites:
        first = sites[0]
        out["first_site_keys"] = sorted(first.keys())
        out["first_site_summary"] = {
            "site_id": first.get("site_id"),
            "materialization_level": first.get("materialization_level"),
            "centroid_xyz": first.get("centroid_xyz"),
            "n_spikes": first.get("n_spikes"),
            "lining_residues_count": len(first.get("lining_residues") or []),
            "driver_residues_count": len(first.get("driver_residues") or []),
            "phase_support": first.get("phase_support"),
            "refined_temporal_support": first.get("refined_temporal_support"),
            "spatiotemporal_so3_evidence": first.get("spatiotemporal_so3_evidence"),
        }
    return out


def summarize_json_keys(path: Path) -> dict[str, Any]:
    data = read_json(path)
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if data is None:
        return out
    out["top_level_keys"] = sorted(data.keys()) if isinstance(data, dict) else []
    for key in [
        "run_id",
        "target",
        "status",
        "validation_status",
        "required_artifacts_complete",
        "stream_count",
        "streams_serialized",
        "total_spikes_md",
        "schema_kind",
        "schema_version",
        "path_b_required",
        "post_md_aggregation_status",
        "site_materialization_status",
        "binding_sites_status",
    ]:
        if isinstance(data, dict) and key in data:
            out[key] = data[key]
    return out


def summarize_json_shape(path: Path) -> dict[str, Any]:
    data = read_json(path)
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if data is None:
        return out
    if isinstance(data, dict):
        out["top_level_keys"] = sorted(data.keys())
        out["key_types"] = {k: type(v).__name__ for k, v in sorted(data.items())}
        for key in [
            "current_step",
            "dt_ps",
            "current_temperature_K",
            "processed_spike_count",
            "n_samples",
            "phase",
            "status",
        ]:
            if key in data:
                out[key] = data[key]
    else:
        out["type"] = type(data).__name__
    return out


def summarize_jsonl_shape(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    try:
        line_count = 0
        first: dict[str, Any] | None = None
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                line_count += 1
                if first is None:
                    first = json.loads(line)
        out["line_count"] = line_count
        if isinstance(first, dict):
            out["first_record_keys"] = sorted(first.keys())
            out["first_record_types"] = {k: type(v).__name__ for k, v in sorted(first.items())}
            for key in [
                "frame_idx",
                "chunk_idx",
                "stream",
                "spike_delta",
                "observation",
                "posterior_max",
                "map_run_length",
                "reset_probability",
            ]:
                if key in first:
                    out[f"first_{key}"] = first[key]
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def run_summary(label: str, run_dir: Path) -> dict[str, Any]:
    files = [p for p in run_dir.iterdir()] if run_dir.exists() else []
    family_counts: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        grouped[classify_run_file(path)].append(path)
    for family, paths in sorted(grouped.items()):
        family_counts[family] = {
            "count": len(paths),
            "total_bytes": sum(p.stat().st_size for p in paths if p.exists()),
            "examples": [p.name for p in sorted(paths)[:3]],
        }

    stream0 = (
        sorted(
            p
            for p in files
            if re.search(r"_stream0+_", p.name)
        )
        if run_dir.exists()
        else []
    )
    stream0_by_family = {classify_run_file(p): p for p in stream0}
    raw_samples = {}
    for family in [
        "adaptive_dt.bin",
        "asc_vectors.bin",
        "forces_final.bin",
        "aromatic_centroids_final.bin",
        "warp_matrix.bin",
    ]:
        path = stream0_by_family.get(family)
        if path:
            raw_samples[family] = parse_envelope_header(path)

    return {
        "label": label,
        "path": str(run_dir),
        "exists": run_dir.exists(),
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files if p.exists()),
        "family_counts": family_counts,
        "md_evidence_manifest": summarize_json_keys(run_dir / "md_evidence_manifest.json"),
        "field_completeness_report": summarize_json_keys(run_dir / "field_completeness_report.json"),
        "materialization_field_completeness": summarize_json_keys(
            run_dir / "materialization_field_completeness.json"
        ),
        "ghost_lattice_routing_status": summarize_json_keys(
            run_dir / "ghost_lattice_routing_status.json"
        ),
        "ghost_time_map": summarize_json_keys(run_dir / "ghost_time_map.json"),
        "post_md_required_inputs": summarize_json_keys(run_dir / "post_md_required_inputs.json"),
        "binding_sites_materialized": summarize_binding_sites(
            run_dir / "binding_sites.materialized.json"
        ),
        "spikes_stream00": parse_first_spike(stream0_by_family.get("spikes.bin", Path("__missing__"))),
        "signal_grid_stream00": parse_signal_grid(
            stream0_by_family.get("signal_grid.bin", Path("__missing__"))
        ),
        "kcc_stream00": parse_kcc(stream0_by_family.get("kcc_v2full.bin", Path("__missing__"))),
        "raw_sidecar_stream00_samples": raw_samples,
        "protocol_state_stream00": summarize_json_shape(
            stream0_by_family.get("protocol_state.json", Path("__missing__"))
        ),
        "noise_floor_stream00": summarize_json_shape(
            stream0_by_family.get("noise_floor.json", Path("__missing__"))
        ),
        "bocpd_stream00": summarize_jsonl_shape(
            stream0_by_family.get("bocpd.jsonl", Path("__missing__"))
        ),
        "arrow_files_found": [str(p) for p in run_dir.rglob("*.arrow")] if run_dir.exists() else [],
        "parquet_files_found": [str(p) for p in run_dir.rglob("*.parquet")] if run_dir.exists() else [],
    }


def collect_dstw_payloads() -> dict[str, Any]:
    paths = [
        CAMPAIGN_ROOT / "prism4d_payload/wt_prime/wt_physics_payload.parquet",
        CAMPAIGN_ROOT / "prism4d_payload/wt_prime/wt_contact_graph.parquet",
        CAMPAIGN_ROOT / "orchestration/BALD_Round_000_Response.parquet",
        CAMPAIGN_ROOT / "orchestration/glp1r_wt_thermodynamic_strata.parquet",
        CAMPAIGN_ROOT / "orchestration/sar_topology/target_hinges.parquet",
        CAMPAIGN_ROOT / "orchestration/sar_topology/hinge_first_degree_contacts.parquet",
        CAMPAIGN_ROOT / "orchestration/sar_topology/topological_lock_interfaces.parquet",
        CAMPAIGN_ROOT / "orchestration/sar_topology/steric_wedge_interfaces.parquet",
        CAMPAIGN_ROOT / "orchestration/sar_topology/primary_pocket_accessible_interfaces.parquet",
        CAMPAIGN_ROOT / "orchestration/sar_topology/allosteric_downstream_interfaces.parquet",
        CAMPAIGN_ROOT / "orchestration/sar_topology/long_range_rejected_interfaces.parquet",
        BIO_ROOT / "output/dstw_payload/glp1r_aleniglipron/5vex/wt_physics_payload.parquet",
        BIO_ROOT / "output/dstw_payload/glp1r_aleniglipron/6x1a/wt_physics_payload.parquet",
    ]
    manifests = [
        CAMPAIGN_ROOT / "prism4d_payload/wt_prime/airgap_payload_manifest.json",
        CAMPAIGN_ROOT / "orchestration/sar_topology/sar_topology_discovery_summary.json",
        CAMPAIGN_ROOT / "orchestration/round_000_svi_posterior_summary.json",
    ]
    return {
        "parquets": [parquet_schema(path) for path in paths],
        "json_manifests": [summarize_json_keys(path) for path in manifests],
    }


def derive_axis_coverage(fact: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "axis": "x/y/z coordinates",
            "engine_raw": "Present in every PRSPK001 spike as position[3]; materialized sites also preserve centroid_xyz.",
            "path_b": "Used to form site centroids and residue shells; raw per-spike coordinates are not exported as a table.",
            "dstw": "Not admitted as absolute voxel arrays. DSTW receives residue graph/contact and pocket centroids.",
            "sar": "Used indirectly through mean_distance_angstrom, pocket proximity, steric wedge feasibility, CGO visualization.",
            "status": "preserved_raw_and_partially_operationalized",
        },
        {
            "axis": "t / timestep",
            "engine_raw": "Present as GpuSpikeEvent.timestep plus ghost_time_map nominal dt_fs and BOCPD chunk/frame logs.",
            "path_b": "Summarized into refined_temporal_support, persistence, entropy, phase support.",
            "dstw": "Collapsed into residue-level active/inactive channels; no per-event temporal table in current payload.",
            "sar": "Used indirectly as TE/delta_hc style static channels, not as interface-breaking timestamps.",
            "status": "preserved_raw_but_collapsed_for_dstw",
        },
        {
            "axis": "amplitude/intensity",
            "engine_raw": "Present as GpuSpikeEvent.intensity and signal_grid hit/coupled grids.",
            "path_b": "Used in density, spike_support, score components, site ranking.",
            "dstw": "Reduced to te_out/te_in/delta_hc/sigma proxy channels.",
            "sar": "Used through derived PRISM channels and interface scores.",
            "status": "operationalized_as_derived_support_not_raw_distribution",
        },
        {
            "axis": "hydration/water-density",
            "engine_raw": "Present as water_density and wd_change per spike.",
            "path_b": "Current Path-B exporter uses centroid_spread_a variance proxy for sigma_hydration_sq.",
            "dstw": "Receives sigma_hydration_sq but not the raw per-spike water-density distribution.",
            "sar": "Uses sigma_hydration_sq where present; do not interpret as direct water-density variance for this run.",
            "status": "raw_preserved_not_fully_operationalized",
        },
        {
            "axis": "kinematic/causal residue fields",
            "engine_raw": "Present in PRKCC001 fields: temporal_corr, direction_score, motion_efficiency, causal_lag, net_dx/net_dy/net_dz, active_causal, etc.",
            "path_b": "Partially used through kcc_driver and materialized site score components.",
            "dstw": "Collapsed into residue physics channels and contact graph support.",
            "sar": "TE-enriched SAR parquets include te_coupling_score/interface_te_differential after post-processing.",
            "status": "partially_operationalized_with_recoverable_raw_fields",
        },
        {
            "axis": "alignment/orientation/force sidecars",
            "engine_raw": "warp_matrix, asc_vectors, forces_final, adaptive_dt are present as raw sidecars.",
            "path_b": "Materialization completeness marks these format_not_parsed/deferred in this commit.",
            "dstw": "No dynamic_aligned_voxel_field was admitted for this campaign.",
            "sar": "Not used except through already materialized centroids and graph metrics.",
            "status": "preserved_raw_deferred",
        },
        {
            "axis": "variant functional deltas",
            "engine_raw": "Not from raw spike files; produced by DSTW/PRISM response layer.",
            "path_b": "N/A.",
            "dstw": "BALD_Round_000_Response carries delta_P_active, delta_P_lock, delta_P_ensemble and sigmas.",
            "sar": "Used in downstream decision layers, not in the SAR topology extraction directly.",
            "status": "dstw_operationalized_not_raw_spike",
        },
    ]


def build_factpack() -> dict[str, Any]:
    fact = {
        "campaign": "glp1r_aleniglipron",
        "scope": {
            "dstw_root": str(DSTW_ROOT),
            "bio_root": str(BIO_ROOT),
            "run_root": str(RUN_ROOT),
            "campaign_root": str(CAMPAIGN_ROOT),
            "run_dirs": {k: str(v) for k, v in RUN_DIRS.items()},
        },
        "source_contracts": {name: str(path) for name, path in SOURCE_CONTRACTS.items()},
        "spike_bin_schema": [
            {"field": f, "dtype": d, "ontology": o} for f, d, o in SPIKE_BIN_SCHEMA
        ],
        "canonical_spike_arrow_schema": [
            {"field": f, "dtype": d, "ontology": o}
            for f, d, o in CANONICAL_SPIKE_ARROW_SCHEMA
        ],
        "engine_file_ontology": ENGINE_FILE_ONTOLOGY,
        "runs": {label: run_summary(label, path) for label, path in RUN_DIRS.items()},
        "dstw_payloads": collect_dstw_payloads(),
    }
    fact["axis_coverage"] = derive_axis_coverage(fact)
    fact["integrity_findings"] = integrity_findings(fact)
    return fact


def integrity_findings(fact: dict[str, Any]) -> list[dict[str, str]]:
    findings = []
    for label, run in fact["runs"].items():
        manifest = run.get("md_evidence_manifest") or {}
        if manifest.get("required_artifacts_complete") is True:
            findings.append(
                {
                    "class": "run_completeness",
                    "run": label,
                    "finding": "Required MD evidence artifacts are complete for this run.",
                }
            )
        if run.get("arrow_files_found") == []:
            findings.append(
                {
                    "class": "canonical_spike_arrow_absent",
                    "run": label,
                    "finding": "No .spike_events.arrow file was found under the run directory; canonical per-spike Arrow is a schema in code, not the on-disk artifact for this campaign.",
                }
            )
        ghost = read_json(Path(run["path"]) / "ghost_lattice_routing_status.json")
        if isinstance(ghost, dict) and ghost.get("v2_was_live") is False:
            findings.append(
                {
                    "class": "v2_not_live",
                    "run": label,
                    "finding": "Captured-frame/ghost lattice V2 telemetry was not live for this run; prism_v2 audit files should not be treated as temporal field payloads.",
                }
            )
        mat = read_json(Path(run["path"]) / "materialization_field_completeness.json")
        if isinstance(mat, dict):
            sources = mat.get("evidence_sources") or {}
            deferred = [
                k
                for k, v in sources.items()
                if isinstance(v, dict)
                and str(v.get("status", "")).lower() in {"deferred", "partial"}
            ]
            if deferred:
                findings.append(
                    {
                        "class": "deferred_materialization_sources",
                        "run": label,
                        "finding": "Materialization reports deferred or partial use for: "
                        + ", ".join(sorted(deferred)),
                    }
                )
    findings.append(
        {
            "class": "dstw_boundary",
            "run": "campaign",
            "finding": "The DSTW handshake forbids raw absolute voxel time-series unless dynamically aligned. The current campaign accepted residue_graph_aggregated/contact graph payloads, not raw voxel TSO.",
        }
    )
    findings.append(
        {
            "class": "sigma_semantics",
            "run": "campaign",
            "finding": "For the current Path-B exporter, sigma_hydration_sq is a centroid-spread variance proxy, not a direct average of per-spike water_density or wd_change.",
        }
    )
    return findings


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(x: Any) -> str:
        if x is None:
            return ""
        text = str(x)
        return text.replace("\n", " ").replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(cell(x) for x in row) + " |")
    return lines


def evidence_status(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "" if value is None else str(value)


def write_markdown(fact: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# PRISM Twin Forensic Output Schema and Spike Ontology")
    lines.append("")
    lines.append("Campaign: `glp1r_aleniglipron`")
    lines.append("")
    lines.append(
        "This audit is a boundary accounting, not a biological claim. It separates raw PRISM Twin / Prism4D engine evidence from Path-B materialization, PRISM-DSTW ingestion, and SAR-facing derivative outputs. Any chronic durability interpretation remains hypothesis-generating until orthogonal biological assays and longer-timescale models are added."
    )
    lines.append("")
    lines.append("## 1. Constraint Gates")
    lines.extend(
        [
            "",
            "- Ontology class isolation: spike-event statistics, voxel-grid statistics, residue KCC fields, materialized-site summaries, DSTW residue channels, and SAR interface tables are separate classes. Averages must only be computed within a class unless a named projection/lift defines the mapping.",
            "- Coordinate-frame gate: DSTW forbids raw absolute voxel arrays unless each frame carries alignment transforms. The current aleniglipron handoff is residue-graph aggregated plus dynamic contact graph.",
            "- Temporal gate: raw spike timesteps exist, but current DSTW/SAR tables are mostly collapsed static residue/interface summaries. Do not describe them as full interface-breaking timestamps.",
            "- Hydration gate: per-spike water_density and wd_change exist in raw spikes. Current `inactive_sigma_hydration_sq` / `active_sigma_hydration_sq` in the Path-B export are centroid-spread proxy channels unless a future exporter recomputes direct water-density variance.",
            "- V2 gate: ghost lattice / captured-frame V2 telemetry was not live for these runs, so empty prism_v2 audit artifacts are not evidence of a usable 5D TSO field.",
        ]
    )
    lines.append("")
    lines.append("## 2. Executive Finding")
    lines.append("")
    lines.append(
        "Both local control runs are stabilized on disk in the sense that the required MD evidence families are present and complete. The engine output contains large raw x/y/z/t/amplitude/hydration/phase/kinematic evidence surfaces. The current PRISM-DSTW aleniglipron payload does not yet consume that full raw temporal field; it consumes a Path-B residue/contact reduction and then SAR-specific interface derivatives. The raw data can support deeper post-MD analytics, but additional decoders/exporters are required before DSTW has a first-class dynamic aligned voxel TSO or interface-breaking timestamp layer."
    )
    lines.append("")

    lines.append("## 3. Run-Level Evidence Surface")
    run_rows = []
    for label, run in fact["runs"].items():
        md = run.get("md_evidence_manifest") or {}
        bs = run.get("binding_sites_materialized") or {}
        run_rows.append(
            [
                label,
                run.get("path"),
                run.get("file_count"),
                fmt_bytes(run.get("total_bytes")),
                md.get("total_spikes_md"),
                md.get("stream_count"),
                evidence_status(md.get("required_artifacts_complete")),
                bs.get("site_count"),
                bs.get("status"),
            ]
        )
    lines.extend(
        md_table(
            [
                "Run",
                "Location",
                "Files",
                "Bytes",
                "Spikes",
                "Streams",
                "Required complete",
                "Materialized sites",
                "Materialization status",
            ],
            run_rows,
        )
    )
    lines.append("")

    lines.append("## 4. Engine File Families")
    family_rows = []
    families = sorted(
        {
            family
            for run in fact["runs"].values()
            for family in (run.get("family_counts") or {}).keys()
        }
    )
    for family in families:
        cells = [family]
        for label in RUN_DIRS:
            data = fact["runs"][label]["family_counts"].get(family)
            cells.append(
                ""
                if not data
                else f"{data['count']} files, {fmt_bytes(data['total_bytes'])}"
            )
        family_rows.append(cells)
    lines.extend(md_table(["Family", "inactive_5vex", "active_6x1a"], family_rows))
    lines.append("")

    lines.append("## 5. JSON and JSONL Artifact Schemas")
    json_rows = []
    json_artifacts = [
        ("md_evidence_manifest.json", "md_evidence_manifest", "Run-level MD evidence manifest and binary envelope declaration."),
        ("field_completeness_report.json", "field_completeness_report", "Required/deferred artifact completeness gate."),
        ("materialization_field_completeness.json", "materialization_field_completeness", "Path-B materialization source-use accounting."),
        ("ghost_lattice_routing_status.json", "ghost_lattice_routing_status", "Captured-frame/ghost lattice route status."),
        ("ghost_time_map.json", "ghost_time_map", "Nominal physical-time map for stream timesteps."),
        ("post_md_required_inputs.json", "post_md_required_inputs", "Declared post-MD aggregation inputs and output classes."),
        ("protocol_state.json", "protocol_state_stream00", "Stream protocol/temperature/steering state."),
        ("noise_floor.json", "noise_floor_stream00", "Stream noise-floor calibration samples."),
        ("bocpd.jsonl", "bocpd_stream00", "Temporal changepoint observations and posterior state."),
    ]
    for artifact, key, ontology in json_artifacts:
        row = [artifact, ontology]
        for label in RUN_DIRS:
            block = fact["runs"][label].get(key) or {}
            keys = block.get("top_level_keys") or block.get("first_record_keys") or []
            extras = []
            for extra_key in [
                "total_spikes_md",
                "required_artifacts_complete",
                "current_step",
                "dt_ps",
                "n_samples",
                "line_count",
                "first_frame_idx",
                "first_chunk_idx",
                "first_stream",
            ]:
                if extra_key in block:
                    extras.append(f"{extra_key}={block[extra_key]}")
            row.append(
                ", ".join(keys[:12])
                + (" ..." if len(keys) > 12 else "")
                + (f" ({'; '.join(extras)})" if extras else "")
            )
        json_rows.append(row)
    lines.extend(
        md_table(
            ["Artifact", "Ontology", "inactive_5vex schema/summary", "active_6x1a schema/summary"],
            json_rows,
        )
    )
    lines.append("")

    lines.append("## 6. Raw Binary Envelope Schemas")
    lines.append("")
    lines.append("Common PRISM MD evidence envelope:")
    lines.extend(
        [
            "",
            "```text",
            "magic[8], schema_version u32, endian_marker u32, stream_id u32,",
            "run_id_len u64, run_id utf8, stem_len u64, stem utf8,",
            "record_count u64, byte_stride u64, payload_size u64,",
            "payload[payload_size], fnv1a_64 u64, trailer[8]",
            "```",
            "",
            "The audited stream-00 envelope headers decode as:",
        ]
    )
    env_rows = []
    for label, run in fact["runs"].items():
        for key, block in [
            ("spikes.bin", run.get("spikes_stream00", {}).get("header")),
            ("signal_grid.bin", run.get("signal_grid_stream00", {}).get("header")),
            ("kcc_v2full.bin", run.get("kcc_stream00", {}).get("header")),
        ]:
            if block:
                env_rows.append(
                    [
                        label,
                        key,
                        block.get("magic"),
                        block.get("schema_version"),
                        block.get("record_count"),
                        block.get("byte_stride"),
                        fmt_bytes(block.get("payload_size")),
                    ]
                )
    lines.extend(
        md_table(
            ["Run", "File", "Magic", "Schema", "Records", "Stride", "Payload"],
            env_rows,
        )
    )
    lines.append("")

    lines.append("### GpuSpikeEvent Record")
    lines.extend(
        md_table(
            ["Field", "Type", "Ontology"],
            [[f, d, o] for f, d, o in SPIKE_BIN_SCHEMA],
        )
    )
    lines.append("")
    sample_rows = []
    for label, run in fact["runs"].items():
        first = run.get("spikes_stream00", {}).get("first_record") or {}
        sample_rows.append(
            [
                label,
                first.get("timestep"),
                first.get("voxel_idx"),
                first.get("position"),
                first.get("intensity"),
                first.get("nearby_residues"),
                first.get("water_density"),
                first.get("wd_change"),
                first.get("phase_bits"),
            ]
        )
    lines.append("First decoded stream-00 spike record:")
    lines.extend(
        md_table(
            [
                "Run",
                "timestep",
                "voxel_idx",
                "position",
                "intensity",
                "nearby residues",
                "water_density",
                "wd_change",
                "phase_bits",
            ],
            sample_rows,
        )
    )
    lines.append("")

    lines.append("### signal_grid.bin Payload")
    signal_rows = []
    for label, run in fact["runs"].items():
        prefix = run.get("signal_grid_stream00", {}).get("payload_prefix") or {}
        signal_rows.append([label, prefix.get("grid_dim"), prefix.get("voxel_count")])
    lines.extend(md_table(["Run", "grid_dim", "voxel_count"], signal_rows))
    lines.append("")

    lines.append("### kcc_v2full.bin Payload")
    kcc_fields = []
    for label, run in fact["runs"].items():
        kcc = run.get("kcc_stream00") or {}
        kcc_fields.append(
            [
                label,
                kcc.get("n_residues"),
                kcc.get("field_count"),
                ", ".join(f"{x['name']}:{x['dtype']}" for x in kcc.get("fields", [])),
            ]
        )
    lines.extend(md_table(["Run", "n_residues", "field_count", "fields"], kcc_fields))
    lines.append("")

    lines.append("## 7. Canonical Per-Spike Arrow Schema")
    lines.append("")
    lines.append(
        "The codebase defines a richer `.spike_events.arrow` schema for per-spike analytics. No `.arrow` files were found under the aleniglipron run directories, so this is an available producer schema, not the on-disk representation for this campaign."
    )
    lines.append("")
    lines.extend(
        md_table(
            ["Field", "Type", "Ontology"],
            [[f, d, o] for f, d, o in CANONICAL_SPIKE_ARROW_SCHEMA],
        )
    )
    lines.append("")

    lines.append("## 8. Engine Output Ontology")
    lines.extend(
        md_table(
            ["Family", "Ontology class", "Schema", "Represents", "Current operationalization"],
            [
                [
                    row["family"],
                    row["class"],
                    row["schema"],
                    row["represents"],
                    row["operationalized_now"],
                ]
                for row in ENGINE_FILE_ONTOLOGY
            ],
        )
    )
    lines.append("")

    lines.append("## 9. Path-B Materialization Ontology")
    mat_rows = []
    for label, run in fact["runs"].items():
        bs = run.get("binding_sites_materialized") or {}
        first = bs.get("first_site_summary") or {}
        mat_rows.append(
            [
                label,
                bs.get("schema_kind"),
                bs.get("site_count"),
                bs.get("n_raw_peaks"),
                bs.get("n_consolidated_regions"),
                first.get("site_id"),
                first.get("n_spikes"),
                first.get("centroid_xyz"),
                first.get("phase_support", {}).get("rayleigh_r_stat")
                if isinstance(first.get("phase_support"), dict)
                else "",
            ]
        )
    lines.extend(
        md_table(
            [
                "Run",
                "Schema kind",
                "Sites",
                "Raw peaks",
                "Consolidated",
                "First site",
                "First site spikes",
                "First centroid",
                "First Rayleigh r",
            ],
            mat_rows,
        )
    )
    lines.append("")
    lines.append(
        "Materialization is where raw spike clouds become site-level hypotheses. The first materialized site preserves centroid, spike support, phase support, temporal support, lining residues, and score components. However, the materialization completeness report marks orientation, force, warp/SO3, and adaptive-dt style sidecars as deferred or not parsed in this commit."
    )
    lines.append("")

    lines.append("## 10. PRISM-DSTW Ingest Ontology")
    parquet_rows = []
    for item in fact["dstw_payloads"]["parquets"]:
        if not item.get("exists"):
            continue
        parquet_rows.append(
            [
                Path(item["path"]).name,
                str(Path(item["path"]).relative_to(DSTW_ROOT))
                if str(item["path"]).startswith(str(DSTW_ROOT))
                else item["path"],
                item.get("num_rows"),
                item.get("num_columns"),
                ", ".join(col["name"] for col in item.get("columns", [])[:12])
                + (" ..." if len(item.get("columns", [])) > 12 else ""),
            ]
        )
    lines.extend(
        md_table(
            ["File", "Location", "Rows", "Columns", "Leading schema fields"],
            parquet_rows,
        )
    )
    lines.append("")
    lines.append(
        "The primary DSTW handoff is `shared_core_8d_wt_prime`: paired inactive/active residue physics plus a dynamic contact graph. SAR then derives target hinges, topological lock interfaces, steric wedge filters, primary pocket-accessible vectors, allosteric downstream interfaces, and long-range rejected correlations."
    )
    lines.append("")

    lines.append("## 11. Axis Propagation Matrix")
    lines.extend(
        md_table(
            ["Axis", "Engine raw", "Path-B", "DSTW", "SAR", "Status"],
            [
                [
                    row["axis"],
                    row["engine_raw"],
                    row["path_b"],
                    row["dstw"],
                    row["sar"],
                    row["status"],
                ]
                for row in fact["axis_coverage"]
            ],
        )
    )
    lines.append("")

    lines.append("## 12. Integrity Findings")
    lines.extend(
        md_table(
            ["Class", "Run", "Finding"],
            [[f["class"], f["run"], f["finding"]] for f in fact["integrity_findings"]],
        )
    )
    lines.append("")

    lines.append("## 13. What This Means for Chronic Receptor Durability")
    lines.append("")
    lines.append(
        "This data is strong mechanistic evidence for choosing local control topologies and SAR/path-sampling targets. It is not, by itself, full chronic receptor durability. It covers a short MD/interferometric captured-graph regime and produces raw spike-level temporal evidence plus residue/contact reductions. Chronic durability additionally requires ligand residence, receptor conformational cycling, G protein coupling, arrestin recruitment, desensitization, internalization, recycling, degradation, membrane context, cellular adaptation, and repeated-exposure kinetics on longer biological timescales."
    )
    lines.append("")
    lines.append("Minimum build required for full leverage:")
    lines.extend(
        [
            "",
            "1. Decode PRSPK001 into canonical Arrow/Parquet for every stream, preserving spike_id, stream_id, x/y/z/t, intensity, water_density, wd_change, phase_bits, nearby residues, and source tags.",
            "2. Parse and validate warp_matrix, asc_vectors, forces_final, and adaptive_dt sidecars, then emit a dynamic aligned voxel field with per-frame transforms that satisfies the DSTW handshake.",
            "3. Build interface-local event joins from spikes to SAR interfaces, so interface-breaking and interface-forming timestamps are extracted directly rather than inferred from static contacts.",
            "4. Recompute ontology-specific statistics only within the correct class: per-spike distributions for spike data, per-residue KCC summaries for KCC data, site-level summaries for materialized sites, and interface-level summaries for SAR tables.",
            "5. Add chronic durability bridge modules for residence time, G protein/arrestin coupling, internalization/recycling/degradation, membrane context, and repeated-dose cellular adaptation. These should consume the mechanistic PRISM evidence as priors or covariates, not as direct proof of chronic biology.",
        ]
    )
    lines.append("")

    lines.append("## 14. Source Contracts Used")
    lines.extend(md_table(["Contract", "Path"], [[k, v] for k, v in fact["source_contracts"].items()]))
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    fact = build_factpack()
    OUT_FACTPACK.write_text(json.dumps(fact, indent=2, sort_keys=True) + "\n")
    write_markdown(fact)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_FACTPACK}")


if __name__ == "__main__":
    main()
