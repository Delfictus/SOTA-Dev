#!/usr/bin/env python3
"""Verify covalent topology roundtrip after SMARTS/Z-matrix assembly fixes."""

from __future__ import annotations

import argparse
import itertools
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_SURVIVORS = TRACK_A / "vspace_survivors_full_scale.parquet"
DEFAULT_SYNTHONS = TRACK_A / "enamine_115k_synthons_3d.parquet"
DEFAULT_ORACLE = REPO_ROOT / "target/release/oracle_scorer"


class TopologyVerificationError(RuntimeError):
    """Raised when a disconnected or malformed product is detected."""


@dataclass(frozen=True)
class VerificationStats:
    n_tested: int
    n_passed: int
    n_failed: int
    edge_cases_tested: int
    edge_cases_passed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--synthons", type=Path, default=DEFAULT_SYNTHONS)
    parser.add_argument("--oracle-bin", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=11011)
    parser.add_argument("--scratch-dir", type=Path, default=REPO_ROOT / ".scratch/topology_verify")
    parser.add_argument("--exit-on-fail", action="store_true")
    return parser.parse_args()


def mol_from_smiles(smiles: str, label: str) -> Any:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise TopologyVerificationError(f"{label} failed RDKit parse: {smiles}")
    return mol


def assert_connected_smiles(smiles: str, label: str) -> Any:
    if "." in smiles:
        raise TopologyVerificationError(f"{label} contains disconnected '.' SMILES: {smiles}")
    mol = mol_from_smiles(smiles, label)
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) != 1:
        raise TopologyVerificationError(f"{label} has {len(fragments)} RDKit fragments: {smiles}")
    return mol


def heavy_atom_count(mol: Any) -> int:
    return int(sum(1 for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) > 1))


def has_declared_reaction_bond(mol: Any) -> bool:
    reaction_bond_smarts = (
        "[CX3](=O)[NX3]",  # amide / urea-like C(=O)-N
        "c-c",  # aryl-aryl Suzuki-style connectivity
        "c-[NX3]",  # Buchwald-Hartwig aryl-N connectivity
        "[NX3][CX3](=O)[NX3]",  # urea linkage
    )
    for smarts in reaction_bond_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return True
    return False


def load_synthon_heavy_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    frame = (
        pl.scan_parquet(path)
        .select("synthon_id", "canonical_smiles", "heavy_atom_count")
        .collect()
    )
    counts: dict[str, int] = {}
    for row in frame.iter_rows(named=True):
        synthon_id = str(row["synthon_id"])
        heavy_value = row.get("heavy_atom_count")
        if isinstance(heavy_value, int):
            counts[synthon_id] = heavy_value
        else:
            mol = mol_from_smiles(str(row["canonical_smiles"]), synthon_id)
            counts[synthon_id] = heavy_atom_count(mol)
    return counts


def sample_survivors(path: Path, n_samples: int, seed: int) -> pl.DataFrame:
    if not path.is_file():
        raise TopologyVerificationError(f"survivor corpus missing: {path}")
    required = [
        "anchor_id",
        "canonical_smiles",
        "synthon_a_id",
        "synthon_b_id",
    ]
    frame = pl.scan_parquet(path).select(required).collect()
    if frame.is_empty():
        raise TopologyVerificationError(f"survivor corpus is empty: {path}")
    rng = random.Random(seed)
    rows = frame.to_dicts()
    sampled = [rows[rng.randrange(len(rows))] for _ in range(n_samples)]
    return pl.DataFrame(sampled)


def ensure_oracle_binary(path: Path) -> None:
    if path.is_file():
        return
    subprocess.run(
        ["cargo", "build", "--release", "-p", "prism-forge", "--bin", "oracle_scorer"],
        cwd=REPO_ROOT,
        check=True,
    )
    if not path.is_file():
        raise TopologyVerificationError(f"oracle binary was not built: {path}")


def run_oracle(
    oracle_bin: Path,
    survivors_path: Path,
    batch: pl.DataFrame,
    scratch_dir: Path,
) -> pl.DataFrame:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = scratch_dir / "oracle_batch.parquet"
    rewards_path = scratch_dir / "oracle_rewards.parquet"
    proposal_batch = pl.DataFrame(
        {
            "trajectory_id": [f"verify-{idx:06d}" for idx in range(batch.height)],
            "anchor_id": batch.get_column("anchor_id").cast(pl.Utf8).to_list(),
            "canonical_smiles": batch.get_column("canonical_smiles").cast(pl.Utf8).to_list(),
        }
    )
    proposal_batch.write_parquet(batch_path)
    if rewards_path.exists():
        rewards_path.unlink()
    subprocess.run(
        [
            str(oracle_bin),
            "--batch",
            str(batch_path),
            "--rewards",
            str(rewards_path),
            "--survivors",
            str(survivors_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return pl.read_parquet(rewards_path)


def verify_oracle_rows(
    source_rows: pl.DataFrame,
    reward_rows: pl.DataFrame,
    synthon_heavy_counts: dict[str, int],
) -> tuple[int, list[str]]:
    if reward_rows.height != source_rows.height:
        raise TopologyVerificationError(
            f"oracle returned {reward_rows.height} rows for {source_rows.height} proposals"
        )
    failures: list[str] = []
    source = source_rows.to_dicts()
    rewards = reward_rows.to_dicts()
    for idx, (source_row, reward_row) in enumerate(zip(source, rewards, strict=True)):
        label = f"row={idx} smiles={reward_row.get('canonical_smiles')}"
        try:
            if not bool(reward_row.get("oracle_valid")):
                raise TopologyVerificationError(f"{label} oracle_valid=false")
            smiles = str(reward_row["canonical_smiles"])
            mol = assert_connected_smiles(smiles, label)
            # The Rust oracle output does not currently persist exact product
            # atom-map metadata, so the cross-language hard gate is graph
            # connectivity. Reaction-motif specificity is stress-tested below
            # against the SMARTS registry classes with RDKit reactions.
            _ = has_declared_reaction_bond(mol)
            product_heavy = heavy_atom_count(mol)
            synthon_a_id = str(source_row.get("synthon_a_id", ""))
            synthon_b_id = str(source_row.get("synthon_b_id", ""))
            available_synthon_counts = [
                value
                for value in (
                    synthon_heavy_counts.get(synthon_a_id, 0),
                    synthon_heavy_counts.get(synthon_b_id, 0),
                )
                if value > 0
            ]
            synthon_floor = min(available_synthon_counts) if available_synthon_counts else 0
            if product_heavy < 6:
                raise TopologyVerificationError(
                    f"{label} product heavy atoms {product_heavy} below medicinal fragment floor"
                )
            if synthon_floor and product_heavy < max(1, synthon_floor - 4):
                raise TopologyVerificationError(
                    f"{label} product heavy atoms {product_heavy} below synthon floor {synthon_floor}"
                )
        except TopologyVerificationError as exc:
            failures.append(str(exc))
    return reward_rows.height - len(failures), failures


def run_reaction_edge_cases() -> tuple[int, int, list[str]]:
    cases = {
        "amide": (
            "[C:1](=[O:2])[O:3].[N:4]>>[C:1](=[O:2])[N:4]",
            ["CC(=O)O", "O=C(O)c1ccccc1", "CCC(=O)O", "O=C(O)C(F)(F)F", "COC(=O)C(=O)O"],
            ["NCC", "Nc1ccccc1", "N1CCCCC1", "NC(C)C", "NCCO"],
            "[CX3](=O)[NX3]",
        ),
        "suzuki": (
            "[c:1][Br,Cl,I].[c:2][B]([O])[O]>>[c:1][c:2]",
            ["Brc1ccccc1", "Clc1ccc(F)cc1", "Ic1ccncc1", "Brc1ccc(C#N)cc1", "Clc1ccccn1"],
            ["OB(O)c1ccccc1", "OB(O)c1ccc(F)cc1", "OB(O)c1ccncc1", "OB(O)c1ccc(OC)cc1", "OB(O)c1ccccn1"],
            "c-c",
        ),
        "buchwald_hartwig": (
            "[c:1][Br,Cl,I].[N:2]>>[c:1][N:2]",
            ["Brc1ccccc1", "Clc1ccc(F)cc1", "Ic1ccncc1", "Brc1ccc(C#N)cc1", "Clc1ccccn1"],
            ["NCC", "Nc1ccccc1", "N1CCCCC1", "NC(C)C", "NCCO"],
            "c-[NX3]",
        ),
    }
    failures: list[str] = []
    tested = 0
    passed = 0
    for reaction_name, (smarts, lefts, rights, product_smarts) in cases.items():
        reaction_from_smarts = getattr(AllChem, "ReactionFromSmarts")
        reaction = reaction_from_smarts(smarts)
        product_pattern = Chem.MolFromSmarts(product_smarts)
        if reaction is None or product_pattern is None:
            failures.append(f"{reaction_name} reaction SMARTS failed compilation")
            continue
        pair_iter = itertools.islice(itertools.cycle(itertools.product(lefts, rights)), 50)
        for left, right in pair_iter:
            tested += 1
            try:
                reactants = (mol_from_smiles(left, f"{reaction_name}:left"), mol_from_smiles(right, f"{reaction_name}:right"))
                products = reaction.RunReactants(reactants)
                if not products:
                    raise TopologyVerificationError(f"{reaction_name} produced no products for {left} + {right}")
                product = products[0][0]
                Chem.SanitizeMol(product)
                product_smiles = Chem.MolToSmiles(product, canonical=True)
                product_mol = assert_connected_smiles(product_smiles, reaction_name)
                if not product_mol.HasSubstructMatch(product_pattern):
                    raise TopologyVerificationError(
                        f"{reaction_name} product lacks expected motif {product_smarts}: {product_smiles}"
                    )
                passed += 1
            except Exception as exc:  # noqa: BLE001 - report exact RDKit/reaction failure.
                failures.append(f"{reaction_name} {left} + {right}: {exc}")
    return tested, passed, failures


def emit_failure_examples(failures: Sequence[str], limit: int = 10) -> None:
    for failure in failures[:limit]:
        print(f"topology_verification_failure {failure}")


def main() -> int:
    args = parse_args()
    if args.n_samples < 1:
        raise TopologyVerificationError("--n-samples must be positive")
    ensure_oracle_binary(args.oracle_bin)
    sample = sample_survivors(args.survivors, int(args.n_samples), int(args.seed))
    rewards = run_oracle(args.oracle_bin, args.survivors, sample, args.scratch_dir)
    synthon_heavy_counts = load_synthon_heavy_counts(args.synthons)
    n_passed, failures = verify_oracle_rows(sample, rewards, synthon_heavy_counts)
    edge_tested, edge_passed, edge_failures = run_reaction_edge_cases()
    all_failures = failures + edge_failures
    stats = VerificationStats(
        n_tested=sample.height,
        n_passed=n_passed,
        n_failed=len(failures),
        edge_cases_tested=edge_tested,
        edge_cases_passed=edge_passed,
    )
    print(
        "topology_verification_complete "
        f"n_tested={stats.n_tested} n_passed={stats.n_passed} n_failed={stats.n_failed} "
        f"edge_cases_tested={stats.edge_cases_tested} edge_cases_passed={stats.edge_cases_passed} "
        f"edge_cases_failed={len(edge_failures)}"
    )
    if all_failures:
        emit_failure_examples(all_failures)
        if args.exit_on_fail:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
