#!/usr/bin/env python3
"""Audit PRISM-4D raw stream classes without decoding payloads.

This scanner is intentionally conservative: it classifies only filename
families that are present on disk, normalizes stream IDs, and reports stream
payload classes separately from stream audit sidecars and replica-level files.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_ROOT = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/"
    "20260518T031002Z/05_RESULTS/glp1r_aleniglipron_risk_map"
)

STREAM_RE = re.compile(
    r"^(?P<target>.+)_stream(?P<stream>\d{1,2})_(?P<suffix>.+?)(?P<ext>\.bin|\.json|\.jsonl)$"
)
PRISM_V2_RE = re.compile(r"^prism_v2_(?P<run>\d+)_(?P<stream>\d+)\.bin$")
PRISM_V2_AUDIT_RE = re.compile(r"^prism_v2_(?P<run>\d+)_(?P<stream>\d+)\.bin\.audit\.json$")

STREAM_SIDECAR_CLASSES = {"prism_v2.bin.audit.json"}

REQUESTED_ABSENCE_PATTERNS = {
    "transfer_entropy": re.compile(r"transfer[_-]?entropy|(^|_)te($|_|\.)", re.I),
    "hysteresis": re.compile(r"hyster", re.I),
    "hydration_or_water": re.compile(r"hydr|water", re.I),
}

EXTRACTOR_MAP = {
    "adaptive_dt.bin": "Rust binary extractor required; no dedicated extractor found in current tree",
    "aromatic_centroids_final.bin": "Rust binary extractor required; no dedicated extractor found in current tree",
    "asc_vectors.bin": "Rust binary extractor required; no dedicated extractor found in current tree",
    "bocpd.jsonl": "Polars JSONL extractor required; source model in crates/prism-nhs/src/bocpd.rs",
    "forces_final.bin": "Rust binary extractor required; no dedicated extractor found in current tree",
    "kcc_v2full.bin": "Existing Python decoder scripts/prism_kcc_decoder.py; rewrite target: lazy Polars/Rust-safe provenance",
    "noise_floor.json": "Polars JSON extractor required; related Rust report bin crates/prism-nhs/src/bin/prism_noise_floor.rs",
    "prism_v2.bin": "Rust binary extractor required; stream audit sidecar present separately",
    "protocol_state.json": "Polars JSON extractor required; schema source crates/prism-nhs/src/protocol_state.rs",
    "signal_grid.bin": "Rust binary extractor required; audit parser exists in scripts/prism_twin_forensic_schema_audit.py",
    "spikes.bin": "Existing Python decoder scripts/prism_spike_event_integrator.py; rewrite target: lazy Polars",
    "warp_matrix.bin": "Rust binary extractor required; no dedicated extractor found in current tree",
}


@dataclass
class ClassStats:
    files: int = 0
    streams: set[tuple[str, int, int]] = field(default_factory=set)
    bytes_total: int = 0
    example: Path | None = None
    magic_counter: Counter[str] = field(default_factory=Counter)


def condition_replica(path: Path, root: Path) -> tuple[str, int] | None:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if len(rel.parts) < 3:
        return None
    condition = rel.parts[0]
    replica_name = rel.parts[1]
    if not replica_name.startswith("replica_"):
        return None
    try:
        replica = int(replica_name.split("_", 1)[1])
    except ValueError:
        return None
    return condition, replica


def stream_class(name: str) -> tuple[int, str] | None:
    match = STREAM_RE.match(name)
    if match:
        return int(match.group("stream")), f"{match.group('suffix')}{match.group('ext')}"
    match = PRISM_V2_RE.match(name)
    if match:
        return int(match.group("stream")), "prism_v2.bin"
    match = PRISM_V2_AUDIT_RE.match(name)
    if match:
        return int(match.group("stream")), "prism_v2.bin.audit.json"
    return None


def replica_class(name: str) -> str:
    if name.startswith("."):
        return f"marker:{name}"
    return name


def read_magic(path: Path) -> str:
    if path.suffix not in {".bin", ".json", ".jsonl"} and ".bin." not in path.name:
        return ""
    try:
        with path.open("rb") as handle:
            raw = handle.read(8)
    except OSError as exc:
        return f"unreadable:{exc.__class__.__name__}"
    if not raw:
        return "empty"
    if all(32 <= byte < 127 for byte in raw):
        return raw.decode("ascii", errors="replace")
    return raw.hex()


def audit(root: Path) -> tuple[dict[str, ClassStats], dict[str, ClassStats], dict[tuple[str, int, int], dict[str, Path]], Counter[str]]:
    stream_stats: dict[str, ClassStats] = defaultdict(ClassStats)
    replica_stats: dict[str, ClassStats] = defaultdict(ClassStats)
    stream_members: dict[tuple[str, int, int], dict[str, Path]] = defaultdict(dict)
    absence_hits: Counter[str] = Counter()

    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            rel_meta = condition_replica(path, root)
            if rel_meta is None:
                continue
            condition, replica = rel_meta

            for label, pattern in REQUESTED_ABSENCE_PATTERNS.items():
                if pattern.search(filename):
                    absence_hits[label] += 1

            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            magic = read_magic(path)

            classified = stream_class(filename)
            if classified:
                stream_id, class_name = classified
                key = (condition, replica, stream_id)
                stats = stream_stats[class_name]
                stats.files += 1
                stats.streams.add(key)
                stats.bytes_total += size
                stats.example = stats.example or path
                if magic:
                    stats.magic_counter[magic] += 1
                stream_members[key][class_name] = path
                continue

            class_name = replica_class(filename)
            stats = replica_stats[class_name]
            stats.files += 1
            stats.bytes_total += size
            stats.example = stats.example or path
            if magic:
                stats.magic_counter[magic] += 1

    return dict(stream_stats), dict(replica_stats), dict(stream_members), absence_hits


def fmt_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    n = float(value)
    for unit in units:
        if n < 1024.0 or unit == units[-1]:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{value} B"


def emit(message: str = "") -> None:
    sys.stdout.write(message + "\n")


def best_stream(stream_members: dict[tuple[str, int, int], dict[str, Path]]) -> tuple[tuple[str, int, int], dict[str, Path]] | None:
    if not stream_members:
        return None
    return max(
        stream_members.items(),
        key=lambda item: (
            len(set(item[1]) - STREAM_SIDECAR_CLASSES),
            len(item[1]),
            item[0][0],
            -item[0][1],
            -item[0][2],
        ),
    )


def print_class_table(title: str, stats: dict[str, ClassStats], root: Path, *, include_streams: bool) -> None:
    emit(f"\n{title}")
    emit("-" * len(title))
    for class_name in sorted(stats):
        row = stats[class_name]
        example = str(row.example.relative_to(root)) if row.example else ""
        magic = row.magic_counter.most_common(1)[0][0] if row.magic_counter else ""
        stream_text = f" streams={len(row.streams):5d}" if include_streams else ""
        emit(
            f"{class_name:34s} files={row.files:5d}{stream_text} "
            f"bytes={fmt_bytes(row.bytes_total):>10s} magic={magic!r} example={example}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        emit(f"ERROR: root does not exist: {root}")
        return 2

    stream_stats, replica_stats, stream_members, absence_hits = audit(root)
    payload_classes = sorted(set(stream_stats) - STREAM_SIDECAR_CLASSES)
    sidecar_classes = sorted(set(stream_stats) & STREAM_SIDECAR_CLASSES)
    streams_total = len(stream_members)

    emit(f"RAW_ROOT={root}")
    emit(f"STREAMS_OBSERVED={streams_total}")
    emit(f"STREAM_PAYLOAD_CLASS_COUNT={len(payload_classes)}")
    emit(f"STREAM_SIDECAR_CLASS_COUNT={len(sidecar_classes)}")
    emit(f"REPLICA_LEVEL_CLASS_COUNT={len(replica_stats)}")

    selected = best_stream(stream_members)
    if selected:
        stream_key, members = selected
        condition, replica, stream_id = stream_key
        emit(f"\nREPRESENTATIVE_STREAM={condition}/replica_{replica}/stream{stream_id}")
        emit("REPRESENTATIVE_STREAM_PAYLOAD_CLASSES")
        for class_name in payload_classes:
            marker = "present" if class_name in members else "MISSING"
            example = members.get(class_name)
            sample = example.name if example else ""
            emit(f"  {class_name:34s} {marker:7s} {sample}")
        if sidecar_classes:
            emit("REPRESENTATIVE_STREAM_SIDECARS")
            for class_name in sidecar_classes:
                marker = "present" if class_name in members else "MISSING"
                example = members.get(class_name)
                sample = example.name if example else ""
                emit(f"  {class_name:34s} {marker:7s} {sample}")

    payload_stats = {name: stream_stats[name] for name in payload_classes}
    sidecar_stats = {name: stream_stats[name] for name in sidecar_classes}
    print_class_table("STREAM PAYLOAD CLASSES", payload_stats, root, include_streams=True)
    if sidecar_stats:
        print_class_table("STREAM SIDECAR CLASSES", sidecar_stats, root, include_streams=True)
    print_class_table("REPLICA-LEVEL CLASSES", replica_stats, root, include_streams=False)

    emit("\nEXTRACTOR MAP")
    emit("-------------")
    for class_name in payload_classes:
        emit(f"{class_name:34s} -> {EXTRACTOR_MAP.get(class_name, 'UNMAPPED')}")

    emit("\nREQUESTED DTSG TOKEN PRESENCE")
    emit("-----------------------------")
    for label in sorted(REQUESTED_ABSENCE_PATTERNS):
        emit(f"{label:24s} filename_hits={absence_hits[label]}")

    missing_from_map = [name for name in payload_classes if name not in EXTRACTOR_MAP]
    if missing_from_map:
        emit(f"\nERROR: payload classes missing extractor map entries: {missing_from_map}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
