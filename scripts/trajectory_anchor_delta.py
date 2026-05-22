#!/usr/bin/env python3
"""Lossless PRISM trajectory anchor-delta codec.

This is an offline storage tool. It does not change engine execution,
ranking, validation, or CUDA graph behavior.

Supported inputs:
  * Gate G2 frames.bin:
      b"PRISM4D\\0" + u32 version + u32 n_atoms + u32 save_interval
      + f32 dt_ps + raw f32 LE positions
  * V2 streamed frames:
      u64 n_frames + repeated [u64 step, u32 n_floats, f32 LE positions]
  * Raw f32 positions with --n-atoms

The delta is not an arithmetic float delta. It is an XOR over the u32
IEEE-754 payload of each f32 relative to the most recent anchor frame.
That makes decoding bit-exact for NaNs, signed zero, and every ordinary
coordinate value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple


ARCHIVE_MAGIC = b"P4ADZ001"
FOOTER_MAGIC = b"P4ADIDX"
GATE_G2_MAGIC = b"PRISM4D\0"
SCHEMA = "prism_anchor_delta_v1"


@dataclass
class SourceInfo:
    kind: str
    n_frames: int
    n_atoms: int
    floats_per_frame: int
    frame_bytes: int
    header_hex: str = ""
    gate_version: Optional[int] = None
    save_interval: Optional[int] = None
    dt_ps: Optional[float] = None


@dataclass
class Frame:
    index: int
    step: int
    n_floats: int
    payload: bytes


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_exact(f: BinaryIO, n: int, what: str) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ValueError(f"short read while reading {what}: expected {n}, got {len(b)}")
    return b


def detect_source(path: Path, n_atoms: Optional[int]) -> SourceInfo:
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(24)

    if len(head) >= 24 and head[:8] == GATE_G2_MAGIC:
        version, atoms, save_interval = struct.unpack_from("<III", head, 8)
        (dt_ps,) = struct.unpack_from("<f", head, 20)
        floats_per_frame = atoms * 3
        frame_bytes = floats_per_frame * 4
        body = size - 24
        if frame_bytes <= 0 or body < 0 or body % frame_bytes != 0:
            raise ValueError(
                f"invalid Gate G2 frames.bin size: body={body}, frame_bytes={frame_bytes}"
            )
        return SourceInfo(
            kind="gate_g2_frames_bin",
            n_frames=body // frame_bytes,
            n_atoms=atoms,
            floats_per_frame=floats_per_frame,
            frame_bytes=frame_bytes,
            header_hex=head[:24].hex(),
            gate_version=version,
            save_interval=save_interval,
            dt_ps=dt_ps,
        )

    if size >= 8:
        try:
            return detect_v2_source(path)
        except ValueError:
            pass

    if n_atoms is None:
        raise ValueError("raw f32 input requires --n-atoms")
    floats_per_frame = n_atoms * 3
    frame_bytes = floats_per_frame * 4
    if size % frame_bytes != 0:
        raise ValueError(
            f"raw input size {size} is not divisible by frame size {frame_bytes}"
        )
    return SourceInfo(
        kind="raw_f32_positions",
        n_frames=size // frame_bytes,
        n_atoms=n_atoms,
        floats_per_frame=floats_per_frame,
        frame_bytes=frame_bytes,
    )


def detect_v2_source(path: Path) -> SourceInfo:
    size = path.stat().st_size
    with path.open("rb") as f:
        (n_frames,) = struct.unpack("<Q", read_exact(f, 8, "v2 n_frames"))
        if n_frames == 0:
            raise ValueError("v2 stream reports zero frames")
        n_floats_first: Optional[int] = None
        for i in range(n_frames):
            _step = struct.unpack("<Q", read_exact(f, 8, f"v2 frame {i} step"))[0]
            (n_floats,) = struct.unpack("<I", read_exact(f, 4, f"v2 frame {i} n_floats"))
            if n_floats == 0 or n_floats % 3 != 0:
                raise ValueError(f"invalid v2 n_floats={n_floats} at frame {i}")
            if n_floats_first is None:
                n_floats_first = n_floats
            elif n_floats != n_floats_first:
                raise ValueError("variable-length V2 frames are not supported by this codec")
            skip = n_floats * 4
            if f.seek(skip, os.SEEK_CUR) < 0:
                raise ValueError("seek failed")
        if f.tell() != size:
            raise ValueError(f"trailing bytes after V2 stream: pos={f.tell()} size={size}")
    assert n_floats_first is not None
    return SourceInfo(
        kind="v2_streamed_frames",
        n_frames=n_frames,
        n_atoms=n_floats_first // 3,
        floats_per_frame=n_floats_first,
        frame_bytes=n_floats_first * 4,
    )


def iter_frames(path: Path, info: SourceInfo) -> Iterable[Frame]:
    with path.open("rb") as f:
        if info.kind == "gate_g2_frames_bin":
            f.seek(24)
            assert info.save_interval is not None
            for i in range(info.n_frames):
                payload = read_exact(f, info.frame_bytes, f"gate_g2 frame {i}")
                yield Frame(i, i * info.save_interval, info.floats_per_frame, payload)
        elif info.kind == "v2_streamed_frames":
            (n_frames,) = struct.unpack("<Q", read_exact(f, 8, "v2 n_frames"))
            for i in range(n_frames):
                (step,) = struct.unpack("<Q", read_exact(f, 8, f"v2 frame {i} step"))
                (n_floats,) = struct.unpack("<I", read_exact(f, 4, f"v2 frame {i} n_floats"))
                payload = read_exact(f, n_floats * 4, f"v2 frame {i} payload")
                yield Frame(i, step, n_floats, payload)
        elif info.kind == "raw_f32_positions":
            for i in range(info.n_frames):
                payload = read_exact(f, info.frame_bytes, f"raw frame {i}")
                yield Frame(i, i, info.floats_per_frame, payload)
        else:
            raise ValueError(f"unsupported source kind: {info.kind}")


def xor_delta(frame: bytes, anchor: bytes) -> bytes:
    if len(frame) != len(anchor) or len(frame) % 4 != 0:
        raise ValueError("frame/anchor byte lengths must match and be f32-aligned")
    out = bytearray(len(frame))
    for off in range(0, len(frame), 4):
        a = struct.unpack_from("<I", anchor, off)[0]
        b = struct.unpack_from("<I", frame, off)[0]
        struct.pack_into("<I", out, off, a ^ b)
    return bytes(out)


def xor_apply(anchor: bytes, delta: bytes) -> bytes:
    return xor_delta(delta, anchor)


def choose_codec(name: str) -> str:
    if name != "auto":
        return name
    try:
        import zstandard  # noqa: F401

        return "zstd"
    except ImportError:
        return "zlib"


def compress_block(data: bytes, codec: str, level: int) -> bytes:
    if codec == "none":
        return data
    if codec == "zlib":
        return zlib.compress(data, level=max(0, min(level, 9)))
    if codec == "zstd":
        try:
            import zstandard as zstd
        except ImportError as e:
            raise RuntimeError("codec zstd requested but Python module zstandard is missing") from e
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data)
    raise ValueError(f"unknown codec: {codec}")


def decompress_block(data: bytes, codec: str) -> bytes:
    if codec == "none":
        return data
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "zstd":
        try:
            import zstandard as zstd
        except ImportError as e:
            raise RuntimeError("archive uses zstd but Python module zstandard is missing") from e
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    raise ValueError(f"unknown codec: {codec}")


def encode_chunk(frames: List[Frame], anchor_stride: int) -> Tuple[bytes, List[int], int]:
    raw = bytearray()
    steps: List[int] = []
    anchor: Optional[bytes] = None
    anchors_in_chunk = 0
    for local_idx, frame in enumerate(frames):
        must_anchor = local_idx == 0 or (anchor_stride > 0 and local_idx % anchor_stride == 0)
        if must_anchor or anchor is None:
            raw.append(ord("A"))
            raw.extend(frame.payload)
            anchor = frame.payload
            anchors_in_chunk += 1
        else:
            raw.append(ord("X"))
            raw.extend(xor_delta(frame.payload, anchor))
        steps.append(frame.step)
    return bytes(raw), steps, anchors_in_chunk


def encode(args: argparse.Namespace) -> None:
    src = Path(args.input)
    out = Path(args.output)
    info = detect_source(src, args.n_atoms)
    codec = choose_codec(args.codec)

    if info.n_frames == 0:
        raise ValueError("refusing to encode an empty trajectory")
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    if args.anchor_stride <= 0:
        raise ValueError("--anchor-stride must be positive")

    source_sha = sha256_file(src)
    chunks: List[Dict[str, int]] = []
    frame_steps: List[int] = []
    total_anchors = 0
    total_raw_chunk_bytes = 0
    total_compressed_bytes = 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as wf:
        wf.write(ARCHIVE_MAGIC)
        offset = len(ARCHIVE_MAGIC)
        buf: List[Frame] = []

        def flush_chunk() -> None:
            nonlocal offset, total_anchors, total_raw_chunk_bytes, total_compressed_bytes
            if not buf:
                return
            raw_chunk, steps, anchors = encode_chunk(buf, args.anchor_stride)
            compressed = compress_block(raw_chunk, codec, args.level)
            wf.write(compressed)
            chunks.append(
                {
                    "offset": offset,
                    "compressed_size": len(compressed),
                    "raw_size": len(raw_chunk),
                    "first_frame": buf[0].index,
                    "n_frames": len(buf),
                    "anchors": anchors,
                }
            )
            frame_steps.extend(steps)
            offset += len(compressed)
            total_anchors += anchors
            total_raw_chunk_bytes += len(raw_chunk)
            total_compressed_bytes += len(compressed)

        for frame in iter_frames(src, info):
            buf.append(frame)
            if len(buf) >= args.chunk_frames:
                flush_chunk()
                buf.clear()
        flush_chunk()

        manifest = {
            "schema": SCHEMA,
            "source": {
                "name": src.name,
                "kind": info.kind,
                "size_bytes": src.stat().st_size,
                "sha256": source_sha,
                "header_hex": info.header_hex,
                "gate_version": info.gate_version,
                "save_interval": info.save_interval,
                "dt_ps": info.dt_ps,
            },
            "layout": {
                "n_frames": info.n_frames,
                "n_atoms": info.n_atoms,
                "floats_per_frame": info.floats_per_frame,
                "frame_bytes": info.frame_bytes,
                "anchor_stride": args.anchor_stride,
                "chunk_frames": args.chunk_frames,
                "chunks": chunks,
                "frame_steps": frame_steps if info.kind == "v2_streamed_frames" else None,
                "total_anchors": total_anchors,
            },
            "compression": {
                "codec": codec,
                "level": args.level,
                "raw_anchor_delta_bytes": total_raw_chunk_bytes,
                "compressed_payload_bytes": total_compressed_bytes,
            },
        }
        footer = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        wf.write(footer)
        wf.write(struct.pack("<Q", len(footer)))
        wf.write(FOOTER_MAGIC)

    report_archive(out, as_json=True)


def read_manifest(path: Path) -> Dict:
    with path.open("rb") as f:
        magic = read_exact(f, len(ARCHIVE_MAGIC), "archive magic")
        if magic != ARCHIVE_MAGIC:
            raise ValueError(f"not a PRISM anchor-delta archive: {path}")
        f.seek(-(8 + len(FOOTER_MAGIC)), os.SEEK_END)
        footer_len = struct.unpack("<Q", read_exact(f, 8, "footer length"))[0]
        footer_magic = read_exact(f, len(FOOTER_MAGIC), "footer magic")
        if footer_magic != FOOTER_MAGIC:
            raise ValueError("missing archive footer magic")
        f.seek(-(8 + len(FOOTER_MAGIC) + footer_len), os.SEEK_END)
        footer = read_exact(f, footer_len, "footer json")
    manifest = json.loads(footer)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported schema: {manifest.get('schema')}")
    return manifest


def decode_chunk(raw_chunk: bytes, n_frames: int, frame_bytes: int) -> List[bytes]:
    frames: List[bytes] = []
    pos = 0
    anchor: Optional[bytes] = None
    for _ in range(n_frames):
        if pos >= len(raw_chunk):
            raise ValueError("chunk ended before expected frame count")
        flag = raw_chunk[pos]
        pos += 1
        body = raw_chunk[pos : pos + frame_bytes]
        if len(body) != frame_bytes:
            raise ValueError("short frame body in chunk")
        pos += frame_bytes
        if flag == ord("A"):
            frame = bytes(body)
            anchor = frame
        elif flag == ord("X"):
            if anchor is None:
                raise ValueError("delta frame appeared before anchor")
            frame = xor_apply(anchor, bytes(body))
        else:
            raise ValueError(f"unknown frame flag {flag!r}")
        frames.append(frame)
    if pos != len(raw_chunk):
        raise ValueError(f"trailing bytes in decoded chunk: {len(raw_chunk) - pos}")
    return frames


def decode(args: argparse.Namespace) -> None:
    archive = Path(args.input)
    out = Path(args.output)
    manifest = read_manifest(archive)
    sha = decode_archive(archive, out, manifest)
    expected = manifest["source"]["sha256"]
    if sha != expected:
        raise ValueError(f"decoded sha256 mismatch: got {sha}, expected {expected}")
    print(json.dumps({"decoded": str(out), "sha256": sha, "verified": True}, indent=2))


def decode_archive(archive: Path, out: Path, manifest: Optional[Dict] = None) -> str:
    manifest = manifest or read_manifest(archive)
    source = manifest["source"]
    layout = manifest["layout"]
    codec = manifest["compression"]["codec"]
    frame_bytes = int(layout["frame_bytes"])
    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    frame_index = 0

    with archive.open("rb") as af, out.open("wb") as wf:
        if source["kind"] == "gate_g2_frames_bin":
            header = bytes.fromhex(source["header_hex"])
            wf.write(header)
            h.update(header)
        elif source["kind"] == "v2_streamed_frames":
            hdr = struct.pack("<Q", int(layout["n_frames"]))
            wf.write(hdr)
            h.update(hdr)

        for chunk in layout["chunks"]:
            af.seek(int(chunk["offset"]))
            compressed = read_exact(af, int(chunk["compressed_size"]), "compressed chunk")
            raw_chunk = decompress_block(compressed, codec)
            frames = decode_chunk(raw_chunk, int(chunk["n_frames"]), frame_bytes)
            for frame in frames:
                if source["kind"] == "v2_streamed_frames":
                    step = int(layout["frame_steps"][frame_index])
                    n_floats = int(layout["floats_per_frame"])
                    rec = struct.pack("<QI", step, n_floats)
                    wf.write(rec)
                    h.update(rec)
                wf.write(frame)
                h.update(frame)
                frame_index += 1

    return h.hexdigest()


def report_archive(path: Path, as_json: bool = False) -> None:
    manifest = read_manifest(path)
    source_size = int(manifest["source"]["size_bytes"])
    archive_size = path.stat().st_size
    ratio = source_size / archive_size if archive_size else 0.0
    out = {
        "archive": str(path),
        "source_kind": manifest["source"]["kind"],
        "source_bytes": source_size,
        "archive_bytes": archive_size,
        "compression_ratio": round(ratio, 4),
        "n_frames": manifest["layout"]["n_frames"],
        "n_atoms": manifest["layout"]["n_atoms"],
        "chunk_count": len(manifest["layout"]["chunks"]),
        "anchor_stride": manifest["layout"]["anchor_stride"],
        "codec": manifest["compression"]["codec"],
    }
    if as_json:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(out, indent=2))


def verify(args: argparse.Namespace) -> None:
    archive = Path(args.input)
    manifest = read_manifest(archive)
    with tempfile.TemporaryDirectory(prefix="prism_traj_verify_") as td:
        decoded = Path(td) / "decoded.bin"
        decoded_sha = decode_archive(archive, decoded, manifest)
        expected = manifest["source"]["sha256"]
        source_sha = sha256_file(Path(args.source)) if args.source else expected
        ok = decoded_sha == expected == source_sha
        print(
            json.dumps(
                {
                    "archive": str(archive),
                    "decoded_sha256": decoded_sha,
                    "manifest_source_sha256": expected,
                    "source_sha256": source_sha,
                    "verified": ok,
                },
                indent=2,
            )
        )
        if not ok:
            raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="Encode a trajectory into anchor-delta archive")
    enc.add_argument("input")
    enc.add_argument("-o", "--output", required=True)
    enc.add_argument("--n-atoms", type=int, default=None, help="Required for raw f32 input")
    enc.add_argument("--anchor-stride", type=int, default=128)
    enc.add_argument("--chunk-frames", type=int, default=512)
    enc.add_argument("--codec", choices=("auto", "zstd", "zlib", "none"), default="auto")
    enc.add_argument("--level", type=int, default=9)
    enc.set_defaults(func=encode)

    dec = sub.add_parser("decode", help="Decode archive back to original bytes")
    dec.add_argument("input")
    dec.add_argument("-o", "--output", required=True)
    dec.set_defaults(func=decode)

    rep = sub.add_parser("report", help="Print archive report")
    rep.add_argument("input")
    rep.set_defaults(func=lambda a: report_archive(Path(a.input), as_json=True))

    ver = sub.add_parser("verify", help="Decode to temp file and verify sha256")
    ver.add_argument("input")
    ver.add_argument("--source", default=None)
    ver.set_defaults(func=verify)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
