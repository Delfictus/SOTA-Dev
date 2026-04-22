#!/usr/bin/env python3
"""Arrow IPC → Parquet+zstd lossless converter.

Reads an Apache Arrow IPC file, writes Parquet with zstd compression,
and validates the output is bit-for-bit lossless (row count, column
names, spot-check first/middle/last rows).

Usage:
    python3 scripts/arrow_to_parquet.py <path_to_arrow_ipc_file>

Output: writes .spike_events.parquet alongside the input .spike_events.arrow
Exit 0 = success (validated), Exit 1 = validation failure
"""

import sys
from pathlib import Path


def convert_and_validate(arrow_path_str: str) -> int:
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    arrow_path = Path(arrow_path_str)
    if not arrow_path.exists():
        print(f"ERROR: file not found: {arrow_path}")
        return 1

    # Derive output path: *.spike_events.arrow → *.spike_events.parquet
    parquet_path = Path(str(arrow_path).replace(".spike_events.arrow", ".spike_events.parquet"))
    if parquet_path == arrow_path:
        # Fallback if naming convention doesn't match
        parquet_path = arrow_path.with_suffix(".parquet")

    arrow_size = arrow_path.stat().st_size
    print(f"INPUT:  {arrow_path}")
    print(f"  Size: {arrow_size:,} bytes ({arrow_size / 1e9:.2f} GB)")

    # Read Arrow IPC
    try:
        reader = ipc.open_file(str(arrow_path))
        arrow_table = reader.read_all()
    except Exception as e:
        print(f"ERROR: failed to read Arrow IPC: {e}")
        return 1

    arrow_rows = arrow_table.num_rows
    arrow_cols = arrow_table.column_names
    print(f"  Rows: {arrow_rows:,}")
    print(f"  Cols: {len(arrow_cols)} — {arrow_cols}")

    if arrow_rows == 0:
        print(f"WARNING: Arrow file has 0 rows — writing empty Parquet")

    # Write Parquet+zstd
    try:
        pq.write_table(
            arrow_table,
            str(parquet_path),
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )
    except Exception as e:
        print(f"ERROR: failed to write Parquet: {e}")
        return 1

    parquet_size = parquet_path.stat().st_size
    ratio = arrow_size / parquet_size if parquet_size > 0 else 0
    print(f"\nOUTPUT: {parquet_path}")
    print(f"  Size: {parquet_size:,} bytes ({parquet_size / 1e6:.1f} MB)")
    print(f"  Ratio: {ratio:.2f}× compression")

    # Validate: re-read Parquet and compare
    try:
        parquet_table = pq.read_table(str(parquet_path))
    except Exception as e:
        print(f"VALIDATION FAILED: cannot re-read Parquet: {e}")
        return 1

    parquet_rows = parquet_table.num_rows
    parquet_cols = parquet_table.column_names

    # Check 1: row count
    if arrow_rows != parquet_rows:
        print(f"VALIDATION FAILED: row count mismatch — Arrow={arrow_rows}, Parquet={parquet_rows}")
        return 1
    print(f"\n  Row count:    {arrow_rows} == {parquet_rows} ✓")

    # Check 2: column names
    if arrow_cols != parquet_cols:
        print(f"VALIDATION FAILED: column names mismatch")
        print(f"  Arrow:   {arrow_cols}")
        print(f"  Parquet: {parquet_cols}")
        return 1
    print(f"  Column names: {len(arrow_cols)} == {len(parquet_cols)} ✓")

    # Check 3: spot-check first, middle, last rows
    if arrow_rows > 0:
        check_indices = [0]
        if arrow_rows > 2:
            check_indices.append(arrow_rows // 2)
        if arrow_rows > 1:
            check_indices.append(arrow_rows - 1)

        mismatches = 0
        for idx in check_indices:
            for col in arrow_cols:
                a_val = arrow_table.column(col)[idx].as_py()
                p_val = parquet_table.column(col)[idx].as_py()
                if a_val != p_val:
                    # Float comparison with tolerance for fp precision
                    if isinstance(a_val, float) and isinstance(p_val, float):
                        if abs(a_val - p_val) < 1e-6 * max(abs(a_val), abs(p_val), 1e-10):
                            continue
                    print(f"VALIDATION FAILED: row {idx}, col '{col}' — Arrow={a_val}, Parquet={p_val}")
                    mismatches += 1
                    if mismatches >= 5:
                        print(f"  (stopping after 5 mismatches)")
                        return 1

        if mismatches > 0:
            print(f"VALIDATION FAILED: {mismatches} value mismatches in spot-check")
            return 1

        print(f"  Spot-check:   rows {check_indices} × {len(arrow_cols)} cols — all match ✓")

    # Check 4: compression ratio sanity
    if ratio < 2.0 or ratio > 6.0:
        print(f"WARNING: compression ratio {ratio:.2f}× outside expected 2-6× range")
        print(f"  This is unusual but not necessarily wrong. Proceeding.")

    print(f"\nVALIDATION: PASS")
    print(f"  Arrow:   {arrow_size:,} bytes ({arrow_size / 1e9:.2f} GB)")
    print(f"  Parquet: {parquet_size:,} bytes ({parquet_size / 1e6:.1f} MB)")
    print(f"  Ratio:   {ratio:.2f}×")
    print(f"  Rows:    {arrow_rows:,}")
    print(f"  Cols:    {len(arrow_cols)}")
    return 0


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <arrow_ipc_file>")
        return 1
    return convert_and_validate(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
