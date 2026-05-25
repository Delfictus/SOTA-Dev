#!/usr/bin/env python3
"""Raw PRKCC001 KCC binary dump without importing project decoders."""

from __future__ import annotations

import math
import struct
from pathlib import Path


KCC_PATH = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/"
    "20260518T031002Z/05_RESULTS/glp1r_aleniglipron_risk_map/"
    "glp1r_5VEX_WT/replica_0/glp1r_5VEX_WT_stream0_kcc_v2full.bin"
)


def read_u64(handle) -> int:
    raw = handle.read(8)
    if len(raw) != 8:
        raise EOFError("unexpected EOF while reading u64")
    return struct.unpack("<Q", raw)[0]


def read_header(handle) -> dict[str, object]:
    magic = handle.read(8).decode("ascii", errors="replace")
    schema_version, endian_marker, stream_id = struct.unpack("<III", handle.read(12))
    run_len = read_u64(handle)
    run_id = handle.read(run_len).decode("utf-8", errors="replace")
    stem_len = read_u64(handle)
    stem = handle.read(stem_len).decode("utf-8", errors="replace")
    record_count, byte_stride, payload_size = struct.unpack("<QQQ", handle.read(24))
    return {
        "magic": magic,
        "schema_version": schema_version,
        "endian_marker_hex": f"0x{endian_marker:08x}",
        "stream_id": stream_id,
        "run_id": run_id,
        "stem": stem,
        "record_count": record_count,
        "byte_stride": byte_stride,
        "payload_size": payload_size,
        "payload_offset": handle.tell(),
    }


def fmt_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(float(value))


def main() -> None:
    with KCC_PATH.open("rb") as handle:
        header = read_header(handle)
        print("HEADER")
        for key, value in header.items():
            print(f"  {key}: {value}")

        n_fields_from_stride = int(header["byte_stride"]) // 4
        first_record_raw = handle.read(int(header["byte_stride"]))
        first_record = struct.unpack(f"<{n_fields_from_stride}f", first_record_raw)
        print("\nMANDATED_BYTE_STRIDE_RECORD_0")
        print(f"  N_fields = byte_stride // 4 = {n_fields_from_stride}")
        print("  f32 =", [float(value) for value in first_record])

        handle.seek(int(header["payload_offset"]))
        n_residues = read_u64(handle)
        field_count = read_u64(handle)
        print("\nCOLUMNAR_PAYLOAD_HEADER")
        print(f"  n_residues: {n_residues}")
        print(f"  field_count: {field_count}")

        record0: list[tuple[str, str, float | int]] = []
        for _ in range(field_count):
            name_len = read_u64(handle)
            name = handle.read(name_len).decode("utf-8", errors="replace")
            dtype_code = struct.unpack("<B", handle.read(1))[0]
            section_size = read_u64(handle)
            raw = handle.read(section_size)
            if dtype_code == 1:
                value = struct.unpack("<f", raw[:4])[0]
                dtype = "f32"
            elif dtype_code == 2:
                value = struct.unpack("<I", raw[:4])[0]
                dtype = "u32"
            else:
                raise ValueError(f"unknown dtype_code {dtype_code} for field {name}")
            record0.append((name, dtype, value))

        print("\nRECORD_0_NAMED_FIELDS")
        for idx, (name, dtype, value) in enumerate(record0):
            rendered = fmt_float(value) if isinstance(value, float) else str(value)
            print(f"  [{idx:02d}] {name} ({dtype}) = {rendered}")

        f32_values = [value for _, dtype, value in record0 if dtype == "f32"]
        print("\nRECORD_0_F32_ARRAY_IN_SERIALIZATION_ORDER")
        print("  [" + ", ".join(fmt_float(float(value)) for value in f32_values) + "]")


if __name__ == "__main__":
    main()
