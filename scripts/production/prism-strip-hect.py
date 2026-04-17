#!/usr/bin/env python3
"""
prism-strip-hect — emit a HECT-only PDB from a full-complex cryo-EM/XRD file.

Part of the production pipeline (scripts/production/). Reads the HECT
family manifest (hect_family_manifest.json), pulls a per-target spec,
and produces a cleaned PDB that retains only:
  * ATOM records on the target's HECT chain
  * residues inside the configured range (full-construct or HECT-domain-only)
while stripping:
  * all other chains (ubiquitin, substrate, scaffold, etc.)
  * any HETATM residues whose resname is on the strip-list
    (K29 linkage analogs, aminolysis mimics, crosslinkers)
  * TER/END records adjusted to the new chain layout

The output is suitable for the existing `prism-clean` → `prism-prep` →
`prism-validate-and-run.sh` pipeline. A BLAKE3 sidecar is written so the
stripped PDB is content-addressed and the strip operation is reviewer-
verifiable.

Usage
-----

    # Full manifest-driven strip (recommended)
    scripts/production/prism-strip-hect.py \\
        --target trip12 \\
        --input  /mnt/storage/.../9gkn.pdb \\
        --output /tmp/trip12_stripped/9gkn_hect.pdb \\
        --mode   hect_only         # or 'full_construct'

    # Manual override (skip the manifest)
    scripts/production/prism-strip-hect.py \\
        --input  9gkn.pdb \\
        --output 9gkn_hect.pdb \\
        --target-chain A \\
        --residue-range 1600 2040 \\
        --strip-chains B C \\
        --strip-hetatm SY8

Modes
-----
  * `full_construct` — use `chain_a_residue_range` (keeps regulatory
    domains; preserves full biological construct minus Ub).
  * `hect_only`      — use `hect_domain_residue_range` (just the HECT
    catalytic domain; ~400-450 residues → KRAS-like wall-clock).

Exit codes
----------
    0 — success; stripped PDB written; BLAKE3 sidecar written
    2 — input file missing / unreadable
    3 — manifest missing or target not in manifest
    4 — empty output (no atoms survived the filter — likely bad config)
    5 — CLI/argument error
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable


# ── provenance (BLAKE3 if available, else fall back to SHA-256) ────────────
try:
    import blake3  # type: ignore
    _HASH_NAME = "blake3"
    def _hexdigest(data: bytes) -> str:
        return blake3.blake3(data).hexdigest()
except ImportError:  # pragma: no cover
    _HASH_NAME = "sha256"
    def _hexdigest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


# ── manifest ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "scripts/production/hect_family_manifest.json"


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open() as f:
        return json.load(f)


def find_target(manifest: dict, target_id: str) -> dict:
    for t in manifest.get("targets", []):
        if t.get("id") == target_id:
            return t
    raise KeyError(
        f"target id {target_id!r} not found in manifest "
        f"(available: {[t['id'] for t in manifest.get('targets', [])]})"
    )


# ── PDB filter ─────────────────────────────────────────────────────────────
def keep_line(
    line: str,
    target_chain: str,
    residue_range: tuple[int, int] | None,
    strip_chains: set[str],
    strip_hetatm_resnames: set[str],
) -> bool:
    """Decide whether a single PDB line is retained.

    PDB column indices (1-indexed, standard PDB v3.3):
      * 13-16  atom name
      * 18-20  residue name
      * 22     chain id
      * 23-26  residue seq number
    """
    if line.startswith(("ATOM  ", "ANISOU")):
        ch = line[21:22]
        try:
            resnum = int(line[22:26])
        except ValueError:
            return False
        if ch != target_chain:
            return False
        if residue_range is not None:
            lo, hi = residue_range
            if resnum < lo or resnum > hi:
                return False
        return True

    if line.startswith("HETATM"):
        ch = line[21:22]
        resname = line[17:20].strip()
        if ch in strip_chains:
            return False
        if resname in strip_hetatm_resnames:
            return False
        # If a HETATM survives strip-chains + strip-resnames and sits on
        # the target chain inside the residue range, keep it (could be a
        # structural cofactor like a Zn that must stay).
        if ch != target_chain:
            return False
        try:
            resnum = int(line[22:26])
        except ValueError:
            return False
        if residue_range is not None:
            lo, hi = residue_range
            if resnum < lo or resnum > hi:
                return False
        return True

    # Always keep structural records; prism-clean will normalize downstream.
    if line.startswith(("HEADER", "TITLE ", "REMARK", "CRYST1", "ORIGX", "SCALE",
                        "MTRIX", "MODEL ", "ENDMDL")):
        return True

    # TER / END / MASTER / CONECT — re-emitted only if the preceding atom
    # survived; simpler to drop and let prism-clean rebuild.
    return False


def strip_pdb(
    input_path: Path,
    target_chain: str,
    residue_range: tuple[int, int] | None,
    strip_chains: set[str],
    strip_hetatm_resnames: set[str],
) -> tuple[bytes, dict]:
    """Returns (output_bytes, stats)."""
    kept = 0
    dropped_atom = 0
    dropped_hetatm = 0
    chains_seen: set[str] = set()
    out_lines: list[str] = []

    with input_path.open() as f:
        for line in f:
            if line.startswith("ATOM  "):
                chains_seen.add(line[21:22])
                if keep_line(line, target_chain, residue_range,
                             strip_chains, strip_hetatm_resnames):
                    out_lines.append(line)
                    kept += 1
                else:
                    dropped_atom += 1
            elif line.startswith("HETATM"):
                if keep_line(line, target_chain, residue_range,
                             strip_chains, strip_hetatm_resnames):
                    out_lines.append(line)
                    kept += 1
                else:
                    dropped_hetatm += 1
            elif keep_line(line, target_chain, residue_range,
                           strip_chains, strip_hetatm_resnames):
                out_lines.append(line)

    out_lines.append("END\n")
    out_bytes = "".join(out_lines).encode("utf-8")

    stats = {
        "input_chains_observed": sorted(chains_seen),
        "target_chain": target_chain,
        "residue_range": residue_range,
        "strip_chains": sorted(strip_chains),
        "strip_hetatm_resnames": sorted(strip_hetatm_resnames),
        "atoms_kept": kept,
        "atoms_dropped": dropped_atom,
        "hetatms_dropped": dropped_hetatm,
    }
    return out_bytes, stats


# ── CLI ────────────────────────────────────────────────────────────────────
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="prism-strip-hect",
        description="Emit a HECT-only PDB by retaining one chain (optionally "
                    "within a residue range) and stripping Ub/substrate chains "
                    "+ linker-analog HETATMs.",
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Path to the full-complex PDB (input)")
    p.add_argument("--output", type=Path, required=True,
                   help="Destination PDB path (output)")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                   help=f"HECT family manifest JSON (default: {DEFAULT_MANIFEST})")
    p.add_argument("--target", type=str, default=None,
                   help="Manifest target id (e.g. 'trip12'). If omitted, use "
                        "--target-chain / --residue-range / --strip-* flags.")
    p.add_argument("--mode", choices=["full_construct", "hect_only"],
                   default="full_construct",
                   help="When using --target, whether to keep the full chain "
                        "residue range or restrict to the HECT catalytic domain.")
    p.add_argument("--target-chain", type=str,
                   help="[manual override] single chain id to retain")
    p.add_argument("--residue-range", type=int, nargs=2, metavar=("LO", "HI"),
                   help="[manual override] inclusive residue range to retain")
    p.add_argument("--strip-chains", type=str, nargs="*", default=[],
                   help="[manual override] chain ids to strip")
    p.add_argument("--strip-hetatm", type=str, nargs="*", default=[],
                   help="[manual override] HETATM resnames to strip")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    if not args.input.is_file():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    # Resolve config: manifest-driven or manual.
    if args.target is not None:
        try:
            manifest = load_manifest(args.manifest)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3
        try:
            spec = find_target(manifest, args.target)
        except KeyError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3

        target_chain = spec["target_chain"]
        if args.mode == "hect_only":
            rr = spec.get("hect_domain_residue_range")
        else:
            rr = spec.get("chain_a_residue_range")
        residue_range = tuple(rr) if rr else None
        strip_chains = set(spec.get("strip_chains") or [])
        strip_hetatm = set(spec.get("strip_hetatm_resnames") or [])
        source = f"manifest:{args.target}:{args.mode}"
    else:
        if not args.target_chain:
            print("ERROR: must pass either --target or --target-chain",
                  file=sys.stderr)
            return 5
        target_chain = args.target_chain
        residue_range = (tuple(args.residue_range)
                         if args.residue_range else None)
        strip_chains = set(args.strip_chains)
        strip_hetatm = set(args.strip_hetatm)
        source = "manual"

    # Execute the filter.
    out_bytes, stats = strip_pdb(
        args.input, target_chain, residue_range, strip_chains, strip_hetatm,
    )

    if stats["atoms_kept"] == 0:
        print("ERROR: 0 atoms survived the filter — likely wrong chain id or "
              "residue range. Check manifest / CLI config.", file=sys.stderr)
        print(f"  observed chains: {stats['input_chains_observed']}",
              file=sys.stderr)
        return 4

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(out_bytes)

    # Content-addressed provenance sidecar (.strip_provenance.json)
    sidecar = args.output.with_suffix(args.output.suffix + ".strip_provenance.json")
    with args.input.open("rb") as fh:
        input_digest = _hexdigest(fh.read())
    output_digest = _hexdigest(out_bytes)
    prov = {
        "schema_version": "prism-strip-hect-v1",
        "tool": "prism-strip-hect",
        "hash_algorithm": _HASH_NAME,
        "source": source,
        "input": {
            "path": str(args.input.resolve()),
            f"{_HASH_NAME}": input_digest,
        },
        "output": {
            "path": str(args.output.resolve()),
            f"{_HASH_NAME}": output_digest,
            "size_bytes": len(out_bytes),
        },
        "config": stats,
    }
    sidecar.write_text(json.dumps(prov, indent=2))

    # Human-readable summary.
    print(f"✓ wrote {args.output}  ({len(out_bytes):,} bytes)")
    print(f"  atoms kept:      {stats['atoms_kept']:,}")
    print(f"  atoms dropped:   {stats['atoms_dropped']:,}")
    print(f"  hetatms dropped: {stats['hetatms_dropped']:,}")
    print(f"  target chain:    {stats['target_chain']}")
    print(f"  residue range:   {stats['residue_range']}")
    print(f"  strip chains:    {stats['strip_chains']}")
    print(f"  strip hetatm:    {stats['strip_hetatm_resnames']}")
    print(f"  provenance:      {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
