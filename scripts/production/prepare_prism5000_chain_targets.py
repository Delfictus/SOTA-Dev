#!/usr/bin/env python3
"""
Prepare and verify chain-level PRISM teacher targets.

Input is the JSONL manifest produced by curate_prism5000_chain_targets.py.
For each chain target this runner:

1. downloads the source PDB file from RCSB
2. extracts the requested author chain with scripts/prism-clean.py
3. builds a PRISM topology with scripts/prism-prep
4. runs structural/topology sanity checks plus repo validators
5. writes per-target logs and a ready_manifest.jsonl

The script is resumable. Existing ready targets are skipped unless --force is set.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import gemmi
except ImportError:  # pragma: no cover - dependency checked at runtime
    gemmi = None


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PrepResult:
    target_id: str
    pdb_id: str
    auth_asym_id: str
    ready: bool
    status: str
    error: str
    paths: dict[str, str]
    metrics: dict[str, Any]
    started_at: float
    finished_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "pdb_id": self.pdb_id,
            "auth_asym_id": self.auth_asym_id,
            "ready": self.ready,
            "status": self.status,
            "error": self.error,
            "paths": self.paths,
            "metrics": self.metrics,
            "elapsed_sec": round(self.finished_at - self.started_at, 3),
        }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def download_file(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "prism5000-prep/1.0"})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                payload = response.read()
            if len(payload) < 1000:
                raise RuntimeError(f"download too small: {len(payload)} bytes")
            tmp.write_bytes(payload)
            os.replace(tmp, dest)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last_error}")


def extract_chain_from_cif(cif_path: Path, out_pdb: Path, auth_chain: str, asym_chain: str) -> str:
    if gemmi is None:
        raise RuntimeError("gemmi is required for mmCIF chain extraction")
    structure = gemmi.read_structure(str(cif_path))
    if not structure:
        raise RuntimeError(f"empty mmCIF structure: {cif_path}")
    wanted = [auth_chain, asym_chain]
    for wanted_chain in wanted:
        if not wanted_chain:
            continue
        out = gemmi.Structure()
        out.name = structure.name
        out.cell = structure.cell
        out.spacegroup_hm = structure.spacegroup_hm
        model_out = gemmi.Model("1")
        found = False
        for chain in structure[0]:
            if chain.name == wanted_chain:
                model_out.add_chain(chain.clone())
                found = True
        if found:
            out.add_model(model_out)
            out_pdb.parent.mkdir(parents=True, exist_ok=True)
            out.write_pdb(str(out_pdb))
            if out_pdb.stat().st_size <= 1000:
                raise RuntimeError(f"extracted chain PDB too small: {out_pdb}")
            return wanted_chain
    raise RuntimeError(
        f"chain not found in mmCIF: auth={auth_chain!r} asym={asym_chain!r} file={cif_path}"
    )


def materialize_source_pdb(
    pdb_id: str,
    auth_chain: str,
    asym_chain: str,
    raw_pdb: Path,
    raw_cif: Path,
    extracted_pdb: Path,
) -> tuple[Path, str]:
    pdb_url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    if download_file(pdb_url, raw_pdb):
        return raw_pdb, "legacy_pdb"

    cif_url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    if not download_file(cif_url, raw_cif):
        cif_url = f"https://files.rcsb.org/download/pdb_0000{pdb_id.lower()}.cif"
        if not download_file(cif_url, raw_cif):
            raise RuntimeError(f"neither legacy PDB nor mmCIF is available for {pdb_id}")
    used_chain = extract_chain_from_cif(raw_cif, extracted_pdb, auth_chain, asym_chain)
    return extracted_pdb, f"mmcif_extracted_chain:{used_chain}"


def run_cmd(cmd: list[str], log_path: Path, *, timeout_sec: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
    return proc.returncode


def topology_metrics(topology_path: Path) -> tuple[bool, dict[str, Any], str]:
    try:
        topology = json.loads(topology_path.read_text())
    except Exception as exc:
        return False, {}, f"topology_json_unreadable: {exc}"

    def array_len(*names: str) -> int:
        for name in names:
            value = topology.get(name)
            if isinstance(value, list):
                return len(value)
        return 0

    atom_count = (
        int(topology.get("n_atoms") or 0)
        or array_len("atoms", "atom_names", "masses", "charges")
    )
    residue_names = topology.get("residue_names")
    residue_count = int(topology.get("n_residues") or 0)
    if not residue_count and isinstance(residue_names, list):
        residue_count = len(residue_names)

    positions_count = array_len("positions", "coordinates", "xyz")
    masses_count = array_len("masses")
    charges_count = array_len("charges")
    atom_names_count = array_len("atom_names")
    chain_ids = topology.get("chain_ids") if isinstance(topology.get("chain_ids"), list) else []
    unique_chain_ids = sorted(set(str(x) for x in chain_ids if str(x).strip()))

    metrics = {
        "atom_count": atom_count,
        "residue_count": residue_count,
        "positions_count": positions_count,
        "masses_count": masses_count,
        "charges_count": charges_count,
        "atom_names_count": atom_names_count,
        "unique_chain_ids": unique_chain_ids,
        "has_bonds": bool(topology.get("bonds")),
        "has_angles": bool(topology.get("angles")),
        "has_dihedrals": bool(topology.get("dihedrals")),
    }

    if atom_count <= 0:
        return False, metrics, "zero_atom_topology"
    if residue_count <= 0:
        return False, metrics, "zero_residue_topology"
    if positions_count not in (0, atom_count, atom_count * 3):
        return False, metrics, "position_count_mismatch"
    if masses_count not in (0, atom_count):
        return False, metrics, "mass_count_mismatch"
    if charges_count not in (0, atom_count):
        return False, metrics, "charge_count_mismatch"
    if atom_names_count not in (0, atom_count):
        return False, metrics, "atom_name_count_mismatch"
    return True, metrics, ""


def prepare_one(
    row: dict[str, Any],
    out_dir: Path,
    *,
    strict_prep: bool,
    use_amber: bool,
    hmr: bool,
    force: bool,
    timeout_sec: int,
) -> PrepResult:
    started = time.time()
    target_id = str(row["target_id"])
    pdb_id = str(row["pdb_id"]).lower()
    chain = str(row["auth_asym_id"])
    asym_chain = str(row.get("asym_id") or chain)
    target_dir = out_dir / "targets" / target_id
    raw_pdb = out_dir / "raw_pdb" / f"{pdb_id}.pdb"
    raw_cif = out_dir / "raw_cif" / f"{pdb_id}.cif"
    extracted_pdb = target_dir / f"{target_id}.source_chain.pdb"
    clean_pdb = target_dir / f"{target_id}.clean.pdb"
    topology = target_dir / f"{target_id}.topology.json"
    status_path = target_dir / "prep_status.json"
    logs = target_dir / "logs"

    paths = {
        "raw_pdb": str(raw_pdb),
        "raw_cif": str(raw_cif),
        "source_chain_pdb": str(extracted_pdb),
        "clean_pdb": str(clean_pdb),
        "topology_json": str(topology),
        "prep_status": str(status_path),
        "clean_log": str(logs / "clean.log"),
        "prep_log": str(logs / "prism_prep.log"),
        "verify_log": str(logs / "verify_topology.log"),
        "validate_log": str(logs / "validate_topology.log"),
    }

    if status_path.exists() and not force:
        try:
            existing = json.loads(status_path.read_text())
            if existing.get("ready") is True and topology.exists():
                return PrepResult(
                    target_id,
                    pdb_id,
                    chain,
                    True,
                    "ready_cached",
                    "",
                    paths,
                    existing.get("metrics") or {},
                    started,
                    time.time(),
                )
        except Exception:
            pass

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        source_pdb, source_kind = materialize_source_pdb(
            pdb_id,
            chain,
            asym_chain,
            raw_pdb,
            raw_cif,
            extracted_pdb,
        )

        clean_rc = run_cmd(
            [sys.executable, "scripts/prism-clean.py", str(source_pdb), str(clean_pdb), chain],
            logs / "clean.log",
            timeout_sec=timeout_sec,
        )
        if clean_rc != 0 or not clean_pdb.exists():
            raise RuntimeError(f"prism-clean failed rc={clean_rc}")

        prep_cmd = ["scripts/prism-prep", str(clean_pdb), str(topology), "--mode", "cryptic"]
        if strict_prep:
            prep_cmd.append("--strict")
        if use_amber:
            prep_cmd.append("--use-amber")
        if hmr:
            prep_cmd.append("--hmr")
        prep_rc = run_cmd(prep_cmd, logs / "prism_prep.log", timeout_sec=timeout_sec)
        if prep_rc != 0 or not topology.exists():
            raise RuntimeError(f"prism-prep failed rc={prep_rc}")

        ok, metrics, metric_error = topology_metrics(topology)
        metrics["source_kind"] = source_kind
        if not ok:
            raise RuntimeError(metric_error)

        verify_rc = run_cmd(
            [sys.executable, "scripts/verify_topology.py", str(topology)],
            logs / "verify_topology.log",
            timeout_sec=timeout_sec,
        )
        if verify_rc != 0:
            raise RuntimeError(f"verify_topology failed rc={verify_rc}")

        validate_cmd = [sys.executable, "scripts/validate_topology.py", str(topology), "--quiet"]
        validate_rc = run_cmd(validate_cmd, logs / "validate_topology.log", timeout_sec=timeout_sec)
        if validate_rc != 0:
            raise RuntimeError(f"validate_topology failed rc={validate_rc}")

        result = PrepResult(
            target_id,
            pdb_id,
            chain,
            True,
            "ready",
            "",
            paths,
            metrics,
            started,
            time.time(),
        )
    except Exception as exc:
        result = PrepResult(
            target_id,
            pdb_id,
            chain,
            False,
            "failed",
            str(exc),
            paths,
            {},
            started,
            time.time(),
        )

    write_json(status_path, result.as_dict())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--strict-prep", action="store_true", default=True)
    parser.add_argument("--no-strict-prep", dest="strict_prep", action="store_false")
    parser.add_argument("--use-amber", action="store_true")
    parser.add_argument("--hmr", action="store_true", default=True)
    parser.add_argument("--no-hmr", dest="hmr", action="store_false")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    status_jsonl = args.out_dir / "prep_status.jsonl"
    ready_jsonl = args.out_dir / "ready_manifest.jsonl"
    failed_jsonl = args.out_dir / "failed_manifest.jsonl"

    print(
        f"[prep] manifest={args.manifest} targets={len(rows)} out={args.out_dir} "
        f"parallel={args.parallel}",
        file=sys.stderr,
    )

    results: list[PrepResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = [
            pool.submit(
                prepare_one,
                row,
                args.out_dir,
                strict_prep=args.strict_prep,
                use_amber=args.use_amber,
                hmr=args.hmr,
                force=args.force,
                timeout_sec=args.timeout_sec,
            )
            for row in rows
        ]
        with status_jsonl.open("a") as status_fh:
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                status_fh.write(json.dumps(result.as_dict(), sort_keys=True) + "\n")
                status_fh.flush()
                if idx % 10 == 0 or idx == len(futures):
                    ready = sum(1 for r in results if r.ready)
                    failed = len(results) - ready
                    print(f"[prep] {idx}/{len(futures)} ready={ready} failed={failed}", file=sys.stderr)

    ready_rows = [r.as_dict() for r in sorted(results, key=lambda x: x.target_id) if r.ready]
    failed_rows = [r.as_dict() for r in sorted(results, key=lambda x: x.target_id) if not r.ready]
    with ready_jsonl.open("w") as fh:
        for row in ready_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with failed_jsonl.open("w") as fh:
        for row in failed_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    report = {
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "requested_targets": len(rows),
        "ready_targets": len(ready_rows),
        "failed_targets": len(failed_rows),
        "strict_prep": args.strict_prep,
        "hmr": args.hmr,
        "use_amber": args.use_amber,
        "ready_manifest": str(ready_jsonl),
        "failed_manifest": str(failed_jsonl),
        "status_jsonl": str(status_jsonl),
    }
    write_json(args.out_dir / "prep_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
    return 0 if len(ready_rows) > 0 or len(rows) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
