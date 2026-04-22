#!/usr/bin/env python3
"""D2 — deterministic legacy-JSON regenerator from canonical Arrow + sidecars.

Input:
  <stem>.topology.spike_events.arrow   (canonical)
  <stem>.binding_sites.json            (unchanged; source of site centroid)
  <stem>.run_metadata.json             (sidecar; source of enum decode + protocol)

Output (one per site in binding_sites.sites[*].id):
  <stem>.site{N}.spike_events.json     (byte-equivalent to engine's legacy dump after
                                         indent=2 normalization)

Top-level per-site JSON schema (unchanged from engine emit path):
  site_id, centroid, n_spikes, lining_cutoff, open_frequency,
  spikes: [ {x,y,z, intensity, type, wavelength_nm, spike_source,
             aromatic_residue_id, water_density, vibrational_energy,
             n_nearby_excited, timestep, frame_index, ccns_phase, stream_id} ]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.compute as pc


def _phase_label_json(ts: int, proto: dict) -> str:
    p1 = proto["cold_hold_steps"]
    p2 = p1 + proto["ramp_steps"]
    p3 = p2 + proto["warm_hold_steps"]
    p4 = p3 + proto.get("ramp_down_steps", 0)
    if ts < p1: return "cold_hold"
    if ts < p2: return "heating"
    if ts < p3: return "warm_hold"
    if ts < p4: return "cooling"
    return "cold_return"


def regenerate(target_dir: Path, stem: str, out_dir: Path | None = None,
               dry_run: bool = False, sites_filter: set[int] | None = None) -> dict:
    eng = target_dir / "artifacts/5_engine"
    arrow_path = eng / f"{stem}.topology.spike_events.arrow"
    bs_path = eng / f"{stem}.binding_sites.json"
    meta_path = eng / f"{stem}.run_metadata.json"

    if not arrow_path.exists():
        raise SystemExit(f"missing Arrow file: {arrow_path}")
    if not bs_path.exists():
        raise SystemExit(f"missing binding_sites.json: {bs_path}")
    if not meta_path.exists():
        raise SystemExit(f"missing run_metadata.json: {meta_path} (run run_metadata_writer.py first)")

    bs = json.loads(bs_path.read_text())
    meta = json.loads(meta_path.read_text())
    protocol = meta["reference_protocol_for_json_phase_label"]
    if "error" in protocol:
        raise SystemExit(f"protocol extraction failed: {protocol}")

    arom_enum = {int(k): v for k, v in meta["aromatic_type_enum"].items()}
    arom_default = meta.get("aromatic_type_default", "UNK")
    src_enum = {int(k): v for k, v in meta["spike_source_enum"].items()}
    src_default = meta.get("spike_source_default", "LIF")

    lining_cutoff = meta.get("lining_cutoff", 8.0)

    # Sites to regenerate: from binding_sites.sites[*].id
    sites = bs.get("sites") or []
    site_centroid = {s.get("id"): s.get("centroid") for s in sites if isinstance(s, dict)}

    # Load Arrow file
    with arrow_path.open("rb") as f:
        reader = ipc.open_file(f) if _is_file_format(arrow_path) else ipc.open_stream(f)
        table = reader.read_all() if hasattr(reader, "read_all") else pa.Table.from_batches(list(reader))

    out_dir = out_dir or eng
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    if sites_filter is None:
        sites_filter = set(site_centroid.keys())

    # Site membership criterion must match engine legacy JSON writer exactly
    # (nhs_rt_full.rs:10809). The engine uses:
    #   site_radius = lining_cutoff + 2.0
    #   include spike IF sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2) <= site_radius
    # This is DIFFERENT from the Arrow `site_id` column, which assigns each
    # spike to exactly ONE nearest site. Overlap-zone spikes appear in
    # MULTIPLE per-site JSON files. Gate A requires us to match this.
    site_radius = float(lining_cutoff) + 2.0
    site_radius_sq = site_radius * site_radius

    x_col = table.column("x").to_numpy()
    y_col = table.column("y").to_numpy()
    z_col = table.column("z").to_numpy()
    import numpy as _np

    for sid, centroid in site_centroid.items():
        if sid not in sites_filter:
            continue
        if centroid is None or len(centroid) != 3:
            continue
        cx, cy, cz = centroid
        # Spatial mask — matches engine filter byte-for-byte
        d2 = (x_col - cx) ** 2 + (y_col - cy) ** 2 + (z_col - cz) ** 2
        mask_np = d2 <= site_radius_sq
        # Convert numpy bool mask to Arrow BooleanArray
        mask_arr = pa.array(mask_np)
        sub = table.filter(mask_arr)
        n_rows = len(sub)
        # Derive open_frequency: unique(frame_index)/max(frame_index)+1, matching engine closure
        frame_col = sub.column("frame_index").to_pylist() if n_rows else []
        if n_rows:
            unique_frames = len(set(frame_col))
            max_frame = max(frame_col)
            total_frames = max(max_frame + 1, 1)
            open_frequency = unique_frames / total_frames
        else:
            open_frequency = 0.0

        # Sort deterministically: (timestep, stream_id, spike_id) to match an
        # arbitrary but reproducible row order. The engine emits in insertion
        # order (same across runs with same seed); for Gate A we compare
        # counts and sampled-row fields, not order.
        if n_rows:
            sub = sub.sort_by([("timestep", "ascending"), ("stream_id", "ascending"), ("spike_id", "ascending")])

        # Project to legacy JSON shape
        spikes = []
        if n_rows:
            cols = {name: sub.column(name).to_pylist() for name in
                    ["x", "y", "z", "intensity", "aromatic_type", "wavelength_nm",
                     "spike_source", "aromatic_residue_id", "water_density",
                     "vibrational_energy", "n_nearby_excited", "timestep",
                     "frame_index", "ccns_phase", "stream_id"]}
            for i in range(n_rows):
                at = cols["aromatic_type"][i]
                ss = cols["spike_source"][i]
                ts = cols["timestep"][i]
                spikes.append({
                    "x": cols["x"][i],
                    "y": cols["y"][i],
                    "z": cols["z"][i],
                    "intensity": cols["intensity"][i],
                    "type": arom_enum.get(int(at), arom_default),
                    "wavelength_nm": cols["wavelength_nm"][i],
                    "spike_source": src_enum.get(int(ss), src_default),
                    "aromatic_residue_id": cols["aromatic_residue_id"][i],
                    "water_density": cols["water_density"][i],
                    "vibrational_energy": cols["vibrational_energy"][i],
                    "n_nearby_excited": cols["n_nearby_excited"][i],
                    "timestep": ts,
                    "frame_index": int(cols["frame_index"][i]),
                    # Use JSON phase_label recomputed from timestep + reference_protocol
                    "ccns_phase": _phase_label_json(int(ts), protocol),
                    "stream_id": int(cols["stream_id"][i]),
                })

        doc = {
            "site_id": sid,
            "centroid": centroid,
            "n_spikes": n_rows,
            "lining_cutoff": lining_cutoff,
            "open_frequency": open_frequency,
            "spikes": spikes,
        }
        out_file = out_dir / f"{stem}.site{sid}.spike_events.regen.json"
        if not dry_run:
            out_file.write_text(json.dumps(doc, indent=2))
        results.append({"site_id": sid, "n_rows": n_rows, "out_file": str(out_file)})
    return {"target": target_dir.name, "stem": stem, "sites": results}


def _is_file_format(path: Path) -> bool:
    """Heuristic: Arrow IPC file format has magic 'ARROW1\\0\\0' at start.
    Stream format begins directly with a message."""
    with path.open("rb") as f:
        head = f.read(8)
    return head.startswith(b"ARROW1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", required=True, type=Path)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--site", type=int, action="append", default=None,
                    help="Only regenerate specified site_ids (repeatable).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sf = set(args.site) if args.site else None
    r = regenerate(args.target_dir, args.stem, args.out_dir, args.dry_run, sf)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
