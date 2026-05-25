#!/usr/bin/env python3
"""Stream ChEMBL bulk SMILES through RDKit and cap to 115k curated anchors.

The pipeline uses multiprocessing workers for RDKit standardization and
descriptor filtering, then Polars lazy scanning for deduplication and sampling.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import os
import sys
import time
from collections.abc import Iterator
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski
from rdkit.Chem.MolStandardize import rdMolStandardize


DEFAULT_INPUT = Path("/home/diddy/prism4d_analysis/library/chembl_36_chemreps.txt.gz")
DEFAULT_FILTERED = Path("/home/diddy/prism4d_analysis/library/chembl_36_ro3_filtered_anchors.csv")
DEFAULT_OUTPUT = Path(
    "campaigns/glp1r_aleniglipron/track_a_generative/115k_curated_anchors.csv"
)
DEFAULT_SOURCE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"
    "chembl_36_chemreps.txt.gz"
)
TARGET_COUNT = 115_000
DEFAULT_SEED = 20260523
DEFAULT_CHUNKSIZE = 512
CATALOG_LOOKUP = {
    "PAINS_A": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
    "PAINS_B": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
    "PAINS_C": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C,
    "BRENK": FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
    "NIH": FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH,
    "ZINC": FilterCatalog.FilterCatalogParams.FilterCatalogs.ZINC,
}

_UNCHARGER: Any = None
_FILTER_CATALOG: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", default=[DEFAULT_INPUT])
    parser.add_argument("--filtered-output", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--processes", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument(
        "--filter-catalogs",
        default="PAINS_A,PAINS_B,PAINS_C",
        help="Comma-separated RDKit FilterCatalog names. Default: PAINS_A,PAINS_B,PAINS_C.",
    )
    parser.add_argument("--reuse-filtered", action="store_true")
    return parser.parse_args()


def init_worker(catalog_names: tuple[str, ...]) -> None:
    global _UNCHARGER, _FILTER_CATALOG
    RDLogger.DisableLog("rdApp.*")
    _UNCHARGER = rdMolStandardize.Uncharger()
    params = FilterCatalog.FilterCatalogParams()
    for name in catalog_names:
        params.AddCatalog(CATALOG_LOOKUP[name])
    _FILTER_CATALOG = FilterCatalog.FilterCatalog(params)


def parse_catalog_names(raw: str) -> tuple[str, ...]:
    names = tuple(name.strip().upper() for name in raw.split(",") if name.strip())
    unknown = sorted(set(names).difference(CATALOG_LOOKUP))
    if unknown:
        raise ValueError(f"Unknown RDKit FilterCatalog names: {', '.join(unknown)}")
    if not names:
        raise ValueError("At least one RDKit FilterCatalog name is required.")
    return names


def open_text(path: Path) -> Any:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def iter_tasks_from_path(path: Path) -> Iterator[tuple[str, str]]:
    with open_text(path) as handle:
        first_line = handle.readline()
        if not first_line:
            return
        header = first_line.rstrip("\n").split("\t")
        if "chembl_id" in header and "canonical_smiles" in header:
            reader = csv.DictReader(itertools.chain([first_line], handle), delimiter="\t")
            for row in reader:
                chembl_id = (row.get("chembl_id") or "").strip()
                smiles = (row.get("canonical_smiles") or "").strip()
                if chembl_id and smiles:
                    yield chembl_id, smiles
            return
        for line_index, line in enumerate(itertools.chain([first_line], handle), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            smiles = parts[0]
            source_id = parts[1] if len(parts) > 1 else f"{path.stem}_{line_index:08d}"
            yield source_id, smiles


def iter_tasks(paths: list[Path]) -> Iterator[tuple[str, str]]:
    for path in paths:
        yield from iter_tasks_from_path(path)


def open_bulk_table(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row


def prepare_mol(raw_smiles: str) -> Any | None:
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.FragmentParent(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol = _UNCHARGER.uncharge(mol)
    Chem.SanitizeMol(mol)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def curate_one(task: tuple[str, str]) -> dict[str, str | float] | None:
    source_id, raw_smiles = task
    try:
        mol = prepare_mol(raw_smiles)
        if mol is None:
            return None
        if _FILTER_CATALOG.HasMatch(mol):
            return None

        mw = float(Descriptors.MolWt(mol))
        clogp = float(Crippen.MolLogP(mol))
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        rotb = int(Lipinski.NumRotatableBonds(mol))

        if mw > 300.0 or clogp > 3.0 or hbd > 3 or hba > 3 or rotb > 3:
            return None

        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return {
            "source_id": source_id,
            "smiles": smiles,
            "mw": round(mw, 4),
            "clogp": round(clogp, 4),
            "hbd": hbd,
            "hba": hba,
            "rotb": rotb,
            "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        }
    except Exception:
        return None


def write_filtered(
    input_paths: list[Path],
    filtered_path: Path,
    processes: int,
    chunksize: int,
    catalog_names: tuple[str, ...],
) -> int:
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_id", "smiles", "mw", "clogp", "hbd", "hba", "rotb", "heavy_atoms"]
    scanned = 0
    kept = 0
    started = time.time()
    last_report = started

    with filtered_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        ctx = get_context("spawn")
        with ctx.Pool(processes=processes, initializer=init_worker, initargs=(catalog_names,)) as pool:
            for result in pool.imap_unordered(curate_one, iter_tasks(input_paths), chunksize=chunksize):
                scanned += 1
                if result is not None:
                    writer.writerow(result)
                    kept += 1
                now = time.time()
                if now - last_report >= 30:
                    rate = scanned / max(now - started, 1.0)
                    print(f"curation_progress\tscanned={scanned}\tkept={kept}\trate={rate:.1f}/s")
                    output_handle.flush()
                    last_report = now
    print(f"curation_complete\tscanned={scanned}\tkept={kept}\tfiltered={filtered_path}")
    return kept


def cap_with_polars(filtered_path: Path, output_path: Path, target_count: int, seed: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lazy_unique = (
        pl.scan_csv(
            filtered_path,
            schema_overrides={"source_id": pl.Utf8, "smiles": pl.Utf8, "mw": pl.Float64, "clogp": pl.Float64},
        )
        .select(["smiles", "mw", "clogp"])
        .drop_nulls()
        .unique(subset=["smiles"], keep="first")
        .sort("smiles")
    )
    unique_count = int(lazy_unique.select(pl.len()).collect().item())
    if unique_count < target_count:
        raise RuntimeError(f"Only {unique_count} unique curated anchors available; need {target_count}.")

    sampled = lazy_unique.collect().sample(n=target_count, seed=seed, shuffle=True).sort("smiles")
    sampled = sampled.with_columns(
        pl.Series("anchor_id", [f"ANCHOR_{idx:06d}" for idx in range(1, target_count + 1)])
    ).select(["anchor_id", "smiles", "mw", "clogp"])
    sampled.write_csv(output_path)
    final_count = int(pl.scan_csv(output_path).select(pl.len()).collect().item())
    if final_count != target_count:
        raise RuntimeError(f"Expected {target_count} rows in {output_path}, observed {final_count}.")
    print(f"cap_complete\tunique={unique_count}\tfinal={final_count}\toutput={output_path}")
    return final_count


def main() -> int:
    started = time.time()
    args = parse_args()
    catalog_names = parse_catalog_names(args.filter_catalogs)
    missing = [path for path in args.input if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing input bulk file: {path}", file=sys.stderr)
        return 2

    RDLogger.DisableLog("rdApp.*")
    print(f"processes\t{args.processes}")
    print(f"filter_catalogs\t{','.join(catalog_names)}")
    print("source_files\t" + ",".join(str(path) for path in args.input))
    if args.reuse_filtered and args.filtered_output.exists():
        kept = int(pl.scan_csv(args.filtered_output).select(pl.len()).collect().item())
        print(f"reuse_filtered\tkept={kept}\tfiltered={args.filtered_output}")
    else:
        kept = write_filtered(args.input, args.filtered_output, args.processes, args.chunksize, catalog_names)
    if kept < args.target_count:
        print(f"Filtered row count {kept} is below requested target {args.target_count}.", file=sys.stderr)
        return 3
    final_count = cap_with_polars(args.filtered_output, args.output, args.target_count, args.seed)
    print(f"source_url\t{args.source_url}")
    print(f"final_row_count\t{final_count}")
    print(f"processing_time_seconds\t{time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
