#!/usr/bin/env python3
"""Decode PRKCC001 per-residue causal/kinematic fields and project to SAR interfaces."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


KCC_FIELDS_F32 = [
    "temporal_corr",
    "direction_score",
    "motion_efficiency",
    "burst_motion",
    "phase_shift",
    "causal_lag",
    "lag_corr_peak",
    "local_cov",
    "net_dx",
    "net_dy",
    "net_dz",
    "sum_m",
]
KCC_FIELDS_U32 = ["residue_count", "active_causal"]
KCC_FIELDS = KCC_FIELDS_F32 + KCC_FIELDS_U32


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_residue_map(path: Path, structure_id: str, mapping_parquet: Path | None) -> dict[int, dict[str, Any]]:
    data = read_json(path)
    pdb_chain_res_to_uniprot: dict[tuple[str, str], int] = {}
    if mapping_parquet and mapping_parquet.exists():
        table = pq.read_table(
            mapping_parquet,
            columns=[
                "pdb_id",
                "auth_asym_id",
                "auth_seq_id",
                "uniprot_residue_index",
                "is_target_uniprot",
            ],
        ).to_pydict()
        for pdb_id, chain, auth_seq, uniprot, is_target in zip(
            table["pdb_id"],
            table["auth_asym_id"],
            table["auth_seq_id"],
            table["uniprot_residue_index"],
            table["is_target_uniprot"],
        ):
            if str(pdb_id).upper() == structure_id.upper() and bool(is_target):
                pdb_chain_res_to_uniprot[(str(chain), str(auth_seq))] = int(uniprot)
    out = {}
    for row in data.get("residues", []):
        topo = int(row["topology_index"])
        chain = str(row.get("chain", ""))
        pdb_resid = int(row["pdb_resid"])
        out[topo] = {
            "topology_residue_index": topo,
            "chain": chain,
            "pdb_resid": pdb_resid,
            "residue_name": str(row.get("resname", "")),
            "uniprot_residue_index": int(
                pdb_chain_res_to_uniprot.get((chain, str(pdb_resid)), pdb_resid)
            ),
        }
    return out


def parse_kcc(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with path.open("rb") as fh:
        magic = fh.read(8).decode("ascii", errors="replace")
        if magic != "PRKCC001":
            raise ValueError(f"{path}: expected PRKCC001, got {magic!r}")
        schema_version, endian_marker, stream_id = struct.unpack("<III", fh.read(12))
        if endian_marker != 0x01020304:
            raise ValueError(f"{path}: unsupported endian marker {hex(endian_marker)}")
        run_len = struct.unpack("<Q", fh.read(8))[0]
        run_id = fh.read(run_len).decode("utf-8", errors="replace")
        stem_len = struct.unpack("<Q", fh.read(8))[0]
        stem = fh.read(stem_len).decode("utf-8", errors="replace")
        record_count, byte_stride, payload_size = struct.unpack("<QQQ", fh.read(24))
        payload_start = fh.tell()
        n_residues, field_count = struct.unpack("<QQ", fh.read(16))
        fields: dict[str, np.ndarray] = {}
        for _ in range(field_count):
            name_len = struct.unpack("<Q", fh.read(8))[0]
            name = fh.read(name_len).decode("utf-8", errors="replace")
            dtype_code = struct.unpack("<B", fh.read(1))[0]
            section_size = struct.unpack("<Q", fh.read(8))[0]
            raw = fh.read(section_size)
            if dtype_code == 1:
                arr = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
            elif dtype_code == 2:
                arr = np.frombuffer(raw, dtype="<u4").astype(np.uint32, copy=True)
            else:
                raise ValueError(f"{path}: unknown KCC dtype code {dtype_code} for {name}")
            fields[name] = arr
        if int(n_residues) != int(record_count):
            raise ValueError(f"{path}: n_residues {n_residues} != record_count {record_count}")
    meta = {
        "path": str(path),
        "magic": magic,
        "schema_version": schema_version,
        "stream_id": stream_id,
        "run_id": run_id,
        "stem": stem,
        "record_count": int(record_count),
        "byte_stride": int(byte_stride),
        "payload_size": int(payload_size),
        "payload_offset": int(payload_start),
        "field_count": len(fields),
    }
    return meta, fields


def discover_kcc_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*_stream*_kcc_v2full.bin"))


def load_interfaces(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    out = []
    for row in rows:
        out.append(
            {
                "interface_id": f"{row['target_hinge_label']}__{row['neighbor_label']}",
                "interface_class": str(row.get("pocket_accessibility_class") or ""),
                "target_hinge_residue_index": int(row["target_hinge_residue_index"]),
                "neighbor_residue_index": int(row["neighbor_residue_index"]),
                "te_coupling_score": float(row.get("te_coupling_score") or 0.0),
                "lock_interface_score": float(row.get("lock_interface_score") or 0.0),
            }
        )
    return out


def residue_rows(
    *,
    campaign_id: str,
    run_label: str,
    structure_id: str,
    meta: dict[str, Any],
    fields: dict[str, np.ndarray],
    residue_map: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    n = meta["record_count"]
    rows = []
    for topo in range(n):
        mapped = residue_map.get(topo, {})
        row = {
            "campaign_id": campaign_id,
            "run_label": run_label,
            "structure_id": structure_id,
            "run_id": meta["run_id"],
            "stem": meta["stem"],
            "stream_id": int(meta["stream_id"]),
            "topology_residue_index": topo,
            "uniprot_residue_index": int(mapped.get("uniprot_residue_index", -1)),
            "pdb_resid": int(mapped.get("pdb_resid", -1)),
            "chain": str(mapped.get("chain", "")),
            "residue_name": str(mapped.get("residue_name", "")),
        }
        for name in KCC_FIELDS_F32:
            row[name] = float(fields.get(name, np.full(n, np.nan, dtype=np.float32))[topo])
        for name in KCC_FIELDS_U32:
            row[name] = int(fields.get(name, np.zeros(n, dtype=np.uint32))[topo])
        rows.append(row)
    return rows


def endpoint_and_delta_rows(
    residue_rows_: list[dict[str, Any]],
    interfaces: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {
        (int(r["stream_id"]), int(r["uniprot_residue_index"])): r
        for r in residue_rows_
        if int(r["uniprot_residue_index"]) >= 0
    }
    streams = sorted({int(r["stream_id"]) for r in residue_rows_})
    endpoint_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for stream_id in streams:
        for iface in interfaces:
            target = by_key.get((stream_id, iface["target_hinge_residue_index"]))
            neighbor = by_key.get((stream_id, iface["neighbor_residue_index"]))
            for role, source in [("target_hinge", target), ("neighbor", neighbor)]:
                if source is None:
                    continue
                row = {
                    "campaign_id": source["campaign_id"],
                    "run_label": source["run_label"],
                    "structure_id": source["structure_id"],
                    "run_id": source["run_id"],
                    "stream_id": stream_id,
                    "interface_id": iface["interface_id"],
                    "interface_class": iface["interface_class"],
                    "endpoint_role": role,
                    "uniprot_residue_index": source["uniprot_residue_index"],
                    "topology_residue_index": source["topology_residue_index"],
                    "residue_name": source["residue_name"],
                    "te_coupling_score": iface["te_coupling_score"],
                    "lock_interface_score": iface["lock_interface_score"],
                }
                for field in KCC_FIELDS:
                    row[field] = source[field]
                endpoint_rows.append(row)
            if target is None or neighbor is None:
                continue
            drow = {
                "campaign_id": target["campaign_id"],
                "run_label": target["run_label"],
                "structure_id": target["structure_id"],
                "run_id": target["run_id"],
                "stream_id": stream_id,
                "interface_id": iface["interface_id"],
                "interface_class": iface["interface_class"],
                "target_hinge_residue_index": iface["target_hinge_residue_index"],
                "neighbor_residue_index": iface["neighbor_residue_index"],
                "te_coupling_score": iface["te_coupling_score"],
                "lock_interface_score": iface["lock_interface_score"],
            }
            for field in KCC_FIELDS:
                tv = target[field]
                nv = neighbor[field]
                drow[f"target_{field}"] = tv
                drow[f"neighbor_{field}"] = nv
                drow[f"delta_target_minus_neighbor_{field}"] = float(tv) - float(nv)
            delta_rows.append(drow)
    return endpoint_rows, delta_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--structure-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--residue-map", type=Path, required=True)
    parser.add_argument("--residue-mapping-parquet", type=Path)
    parser.add_argument("--interfaces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    residue_map = load_residue_map(args.residue_map, args.structure_id, args.residue_mapping_parquet)
    interfaces = load_interfaces(args.interfaces)

    all_residue_rows: list[dict[str, Any]] = []
    stream_meta = []
    for path in discover_kcc_paths(args.run_dir):
        meta, fields = parse_kcc(path)
        stream_meta.append(meta)
        all_residue_rows.extend(
            residue_rows(
                campaign_id=args.campaign_id,
                run_label=args.run_label,
                structure_id=args.structure_id,
                meta=meta,
                fields=fields,
                residue_map=residue_map,
            )
        )

    endpoint_rows, delta_rows = endpoint_and_delta_rows(all_residue_rows, interfaces)

    residue_path = args.out_dir / f"{args.run_label}_kcc_residue_fields.parquet"
    endpoint_path = args.out_dir / f"{args.run_label}_interface_kcc_endpoint_fields.parquet"
    delta_path = args.out_dir / f"{args.run_label}_interface_kcc_pair_deltas.parquet"
    pq.write_table(pa.Table.from_pylist(all_residue_rows), residue_path, compression="zstd", compression_level=6)
    pq.write_table(pa.Table.from_pylist(endpoint_rows), endpoint_path, compression="zstd", compression_level=6)
    pq.write_table(pa.Table.from_pylist(delta_rows), delta_path, compression="zstd", compression_level=6)

    manifest = {
        "schema": "prism_kcc_causal_kinematic_decode.v1",
        "campaign_id": args.campaign_id,
        "run_label": args.run_label,
        "structure_id": args.structure_id,
        "run_dir": str(args.run_dir),
        "residue_map": str(args.residue_map),
        "residue_mapping_parquet": str(args.residue_mapping_parquet)
        if args.residue_mapping_parquet
        else None,
        "interfaces": str(args.interfaces),
        "outputs": {
            "kcc_residue_fields": str(residue_path),
            "interface_kcc_endpoint_fields": str(endpoint_path),
            "interface_kcc_pair_deltas": str(delta_path),
        },
        "counts": {
            "streams": len(stream_meta),
            "residue_rows": len(all_residue_rows),
            "interface_endpoint_rows": len(endpoint_rows),
            "interface_pair_delta_rows": len(delta_rows),
        },
        "fields": {
            "f32": KCC_FIELDS_F32,
            "u32": KCC_FIELDS_U32,
        },
        "stream_meta": stream_meta,
        "semantic_warnings": [
            "KCC fields are per-residue causal/kinematic fields and must not be averaged into spike-event hydration statistics without an explicit projection.",
            "Interface endpoint rows preserve target and neighbor residue roles separately.",
            "Pair delta rows are endpoint differences, not thermodynamic free energies.",
        ],
    }
    manifest_path = args.out_dir / f"{args.run_label}_kcc_decode_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_path}")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
