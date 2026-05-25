#!/usr/bin/env python3
"""Assign AM1-BCC charges to aligned Aleniglipron and persist them on the SDF."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/conformers/ALENI-PARENT_whole_molecule_aligned.sdf"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/conformers/ALENI-PARENT_am1bcc.sdf"
)
DEFAULT_ANTECHAMBER = Path("/home/diddy/miniconda3/envs/prism_dock/bin/antechamber")
DEFAULT_AMBERHOME = Path("/home/diddy/miniconda3/envs/prism_dock")


Chem = cast(Any, import_module("rdkit.Chem"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--antechamber", type=Path, default=DEFAULT_ANTECHAMBER)
    parser.add_argument("--amberhome", type=Path, default=DEFAULT_AMBERHOME)
    parser.add_argument("--net-charge", type=int, default=None)
    return parser.parse_args()


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def load_single_mol(path: Path) -> Any:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = supplier[0] if len(supplier) else None
    if mol is None:
        raise ValueError(f"failed to parse SDF: {path}")
    if int(mol.GetNumConformers()) != 1:
        raise ValueError(f"expected exactly one conformer in {path}, found {mol.GetNumConformers()}")
    return mol


def formal_charge(mol: Any) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def amber_env(amberhome: Path, antechamber: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["AMBERHOME"] = str(amberhome)
    env["PATH"] = f"{antechamber.parent}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{amberhome / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def run_antechamber(input_sdf: Path, output_mol2: Path, antechamber: Path, amberhome: Path, net_charge: int) -> str:
    if not antechamber.exists():
        raise FileNotFoundError(antechamber)
    command = [
        str(antechamber),
        "-i",
        str(input_sdf),
        "-fi",
        "sdf",
        "-o",
        str(output_mol2),
        "-fo",
        "mol2",
        "-c",
        "bcc",
        "-nc",
        str(net_charge),
        "-s",
        "2",
        "-pf",
        "y",
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=amber_env(amberhome, antechamber),
        cwd=output_mol2.parent,
    )
    if result.returncode != 0:
        raise RuntimeError("antechamber AM1-BCC failed:\n" + result.stdout)
    return result.stdout


def parse_mol2_charges(path: Path) -> list[float]:
    charges: list[float] = []
    in_atom_block = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "@<TRIPOS>ATOM":
            in_atom_block = True
            continue
        if stripped.startswith("@<TRIPOS>") and in_atom_block:
            break
        if not in_atom_block or not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 9:
            raise ValueError(f"invalid MOL2 atom line in {path}: {line}")
        charges.append(float(fields[-1]))
    if not charges:
        raise ValueError(f"no MOL2 charges parsed from {path}")
    return charges


def persist_charges(mol: Any, charges: list[float], output: Path, antechamber_log: str) -> None:
    if len(charges) != int(mol.GetNumAtoms()):
        raise ValueError(f"charge count mismatch: charges={len(charges)} atoms={mol.GetNumAtoms()}")
    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("AM1BCCCharge", float(charge))
        atom.SetDoubleProp("PartialCharge", float(charge))
    if hasattr(Chem, "CreateAtomDoublePropertyList"):
        Chem.CreateAtomDoublePropertyList(mol, "AM1BCCCharge")
        Chem.CreateAtomDoublePropertyList(mol, "PartialCharge")
    mol.SetProp("charge_method", "AM1-BCC")
    mol.SetProp("am1bcc_charges_json", json.dumps(charges, separators=(",", ":")))
    mol.SetProp("am1bcc_total_charge", f"{sum(charges):.8f}")
    mol.SetProp("am1bcc_tool", "AmberTools antechamber -c bcc")
    mol.SetProp("am1bcc_antechamber_log_tail", "\n".join(antechamber_log.splitlines()[-20:]))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output))
    writer.write(mol)
    writer.close()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    antechamber = Path(args.antechamber)
    amberhome = Path(args.amberhome)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    mol = load_single_mol(input_path)
    net_charge = formal_charge(mol) if args.net_charge is None else int(args.net_charge)
    with tempfile.TemporaryDirectory(prefix="prism_am1bcc_") as tmp_dir:
        tmp = Path(tmp_dir)
        work_sdf = tmp / "aleni_input.sdf"
        work_mol2 = tmp / "aleni_am1bcc.mol2"
        shutil.copy2(input_path, work_sdf)
        log = run_antechamber(work_sdf, work_mol2, antechamber, amberhome, net_charge)
        charges = parse_mol2_charges(work_mol2)
    persist_charges(mol, charges, output_path, log)
    emit(
        f"wrote {output_path} charge_method=AM1-BCC atoms={mol.GetNumAtoms()} "
        f"net_charge={sum(charges):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
