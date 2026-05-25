#!/usr/bin/env python3
"""Build the expert-seeded GLP-1R ligand benchmark panel and 3D SDF conformers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
DEFAULT_MANIFEST_PARQUET = CAMPAIGN_DIR / "GLP1R_LIGAND_SET_MANIFEST_v1.parquet"
DEFAULT_MANIFEST_CSV = CAMPAIGN_DIR / "GLP1R_LIGAND_SET_MANIFEST_v1.csv"
DEFAULT_CONFORMER_DIR = CAMPAIGN_DIR / "track_a_generative/conformers"
DEFAULT_BINDING_SITE_REFERENCE = CAMPAIGN_DIR / "track_0_manual_emulation/binding_site_reference.json"
DEFAULT_ANTECHAMBER = Path("/home/diddy/miniconda3/envs/prism_dock/bin/antechamber")
DEFAULT_AMBERHOME = Path("/home/diddy/miniconda3/envs/prism_dock")
CONFORMER_COUNT = 10

JsonObject: TypeAlias = dict[str, Any]
Point: TypeAlias = tuple[float, float, float]
Matrix: TypeAlias = list[list[float]]
Vector: TypeAlias = list[float]

Chem = cast(Any, import_module("rdkit.Chem"))
AllChem = cast(Any, import_module("rdkit.Chem.AllChem"))
Descriptors = cast(Any, import_module("rdkit.Chem.Descriptors"))
np = cast(Any, import_module("numpy"))


@dataclass(frozen=True)
class BenchmarkLigand:
    ligand_id: str
    compound_name: str
    smiles: str
    ec50_nm: float


BENCHMARKS: tuple[BenchmarkLigand, ...] = (
    BenchmarkLigand(
        ligand_id="ORFORGLIPRON_LY3502970",
        compound_name="Orforglipron (LY3502970)",
        smiles=(
            "CC1=C(C=C(C=C1)F)C2=C(N=C(N2C3=CC=C(C=C3)C(F)(F)F)C4=CC=C(C=C4)OC)"
            "C(=O)N[C@H](CC5=CC=CC=C5)C(=O)O"
        ),
        ec50_nm=3.0,
    ),
    BenchmarkLigand(
        ligand_id="DANUGLIPRON_PF06882961",
        compound_name="Danuglipron (PF-06882961)",
        smiles="c1cc(ccc1c2c(n(cn2)C3CCC(CC3)O)c4ccc(cc4)C#N)Cl",
        ec50_nm=4.5,
    ),
    BenchmarkLigand(
        ligand_id="LOTIGLIPRON_PF07081532",
        compound_name="Lotiglipron (PF-07081532)",
        smiles=(
            "CC1=CN=C(C(=C1)OC)C2=NC(=C(N2C3=CC=C(C=C3)C#N)C4=CC=C(C=C4)Cl)"
            "C(=O)N[C@H](CC5=CC=CC=C5)C(=O)O"
        ),
        ec50_nm=4.8,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-parquet", type=Path, default=DEFAULT_MANIFEST_PARQUET)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--conformer-dir", type=Path, default=DEFAULT_CONFORMER_DIR)
    parser.add_argument("--binding-site-reference", type=Path, default=DEFAULT_BINDING_SITE_REFERENCE)
    parser.add_argument("--antechamber", type=Path, default=DEFAULT_ANTECHAMBER)
    parser.add_argument("--amberhome", type=Path, default=DEFAULT_AMBERHOME)
    parser.add_argument("--conformers", type=int, default=CONFORMER_COUNT)
    parser.add_argument("--charge-timeout-seconds", type=int, default=3600)
    return parser.parse_args()


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse benchmark SMILES: {smiles}")
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def benchmark_rows() -> list[JsonObject]:
    rows: list[JsonObject] = []
    for ligand in BENCHMARKS:
        rows.append(
            {
                "ligand_id": ligand.ligand_id,
                "compound_name": ligand.compound_name,
                "canonical_smiles": canonical_smiles(ligand.smiles),
                "source_database": "Clinical_Literature",
                "known_target": "GLP-1R",
                "known_activity_type": "EC50",
                "known_activity_value": float(ligand.ec50_nm),
                "activity_units": "nM",
                "confidence_class": "high_confidence_clinical",
                "included_in_track": "Track_A_Benchmark_Set",
            }
        )
    return rows


def manifest_lazy_frame() -> pl.LazyFrame:
    return (
        pl.LazyFrame(benchmark_rows())
        .with_columns(
            pl.col("ligand_id").cast(pl.Utf8),
            pl.col("compound_name").cast(pl.Utf8),
            pl.col("canonical_smiles").cast(pl.Utf8),
            pl.col("source_database").cast(pl.Utf8),
            pl.col("known_target").cast(pl.Utf8),
            pl.col("known_activity_type").cast(pl.Utf8),
            pl.col("known_activity_value").cast(pl.Float64),
            pl.col("activity_units").cast(pl.Utf8),
            pl.col("confidence_class").cast(pl.Utf8),
            pl.col("included_in_track").cast(pl.Utf8),
        )
        .select(
            "ligand_id",
            "compound_name",
            "canonical_smiles",
            "source_database",
            "known_target",
            "known_activity_type",
            "known_activity_value",
            "activity_units",
            "confidence_class",
            "included_in_track",
        )
    )


def write_manifest(parquet_path: Path, csv_path: Path) -> pl.DataFrame:
    lazy = manifest_lazy_frame()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    lazy.sink_parquet(parquet_path)
    frame = lazy.collect()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(csv_path)
    return frame


def read_json_object(path: Path) -> JsonObject:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(JsonObject, decoded)


def point_from_raw(raw: object, label: str) -> Point:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{label} must be an xyz array")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def reference_points(payload: JsonObject) -> list[Point]:
    reference = payload.get("reference")
    if not isinstance(reference, dict):
        return [point_from_raw(payload.get("binding_site_center_angstrom"), "binding_site_center_angstrom")]
    raw_points = reference.get("alignment_point_coordinates")
    if not isinstance(raw_points, list) or len(raw_points) < 3:
        return [point_from_raw(reference.get("centroid_xyz_angstrom"), "reference.centroid_xyz_angstrom")]
    return [point_from_raw(item, "reference.alignment_point_coordinates[]") for item in raw_points]


def as_array(points: list[Point]) -> Any:
    return np.array(points, dtype=np.float64)


def matrix_to_list(matrix: Any) -> Matrix:
    values = np.asarray(matrix, dtype=np.float64).tolist()
    return [[float(item) for item in row] for row in values]


def vector_to_list(vector: Any) -> Vector:
    return [float(item) for item in np.asarray(vector, dtype=np.float64).tolist()]


def principal_axes(points: Any) -> Any:
    centered = points - points.mean(axis=0)
    if int(centered.shape[0]) < 3:
        return np.eye(3, dtype=np.float64)
    covariance = np.matmul(centered.T, centered) / max(float(centered.shape[0] - 1), 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    if float(np.linalg.det(axes)) < 0.0:
        axes[:, 2] *= -1.0
    return axes


def conformer_heavy_points(mol: Any, conformer_id: int) -> Any:
    conformer = mol.GetConformer(conformer_id)
    coords: list[Point] = []
    for atom in mol.GetAtoms():
        if int(atom.GetAtomicNum()) <= 1:
            continue
        pos = conformer.GetAtomPosition(int(atom.GetIdx()))
        coords.append((float(pos.x), float(pos.y), float(pos.z)))
    if len(coords) < 3:
        raise ValueError("benchmark conformer must contain at least three heavy atoms for alignment")
    return as_array(coords)


def transform_conformer_to_reference(
    mol: Any,
    conformer_id: int,
    reference_center: Any,
    reference_axes: Any,
) -> tuple[Matrix, Vector]:
    conformer = mol.GetConformer(conformer_id)
    ligand_points = conformer_heavy_points(mol, conformer_id)
    ligand_center = ligand_points.mean(axis=0)
    ligand_axes = principal_axes(ligand_points)
    rotation = np.matmul(reference_axes, ligand_axes.T)
    translation = reference_center - np.matmul(rotation, ligand_center)
    for atom_index in range(int(mol.GetNumAtoms())):
        position = conformer.GetAtomPosition(atom_index)
        current = np.array([float(position.x), float(position.y), float(position.z)], dtype=np.float64)
        transformed = np.matmul(rotation, current) + translation
        conformer.SetAtomPosition(
            atom_index,
            (float(transformed[0]), float(transformed[1]), float(transformed[2])),
        )
    return matrix_to_list(rotation), vector_to_list(translation)


def formal_charge(mol: Any) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def amber_env(amberhome: Path, antechamber: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["AMBERHOME"] = str(amberhome)
    env["PATH"] = f"{antechamber.parent}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{amberhome / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def parse_mol2_charges(path: Path, atom_count: int) -> list[float]:
    charges: list[float] = []
    in_atom_block = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
            raise ValueError(f"invalid MOL2 atom line: {line}")
        charges.append(float(fields[-1]))
    if len(charges) != atom_count:
        raise ValueError(f"MOL2 charge count {len(charges)} did not match atom count {atom_count}")
    return charges


def compute_am1bcc_charges(
    mol: Any,
    *,
    antechamber: Path,
    amberhome: Path,
    net_charge: int,
    timeout_seconds: int,
) -> tuple[list[float], str]:
    if not antechamber.exists():
        raise FileNotFoundError(antechamber)
    with tempfile.TemporaryDirectory(prefix="prism_glp1r_benchmark_am1bcc_") as tmp_name:
        tmp_dir = Path(tmp_name)
        input_sdf = tmp_dir / "input.sdf"
        output_mol2 = tmp_dir / "charged.mol2"
        Chem.MolToMolFile(mol, str(input_sdf))
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
            cwd=tmp_dir,
            env=amber_env(amberhome, antechamber),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("antechamber AM1-BCC failed:\n" + result.stdout[-4000:])
        charges = parse_mol2_charges(output_mol2, int(mol.GetNumAtoms()))
    return charges, result.stdout


def set_charge_properties(mol: Any, charges: list[float]) -> None:
    if len(charges) != int(mol.GetNumAtoms()):
        raise ValueError("charge vector length does not match molecule atom count")
    for atom, charge in zip(mol.GetAtoms(), charges, strict=True):
        atom.SetDoubleProp("AM1BCCCharge", float(charge))
        atom.SetDoubleProp("PartialCharge", float(charge))
        atom.SetProp("am1bcc_charge", f"{float(charge):.12f}")
    if hasattr(Chem, "CreateAtomDoublePropertyList"):
        Chem.CreateAtomDoublePropertyList(mol, "AM1BCCCharge")
        Chem.CreateAtomDoublePropertyList(mol, "PartialCharge")
    mol.SetProp("charge_method", "AM1-BCC")
    mol.SetProp("am1bcc_charges_json", json.dumps(charges, separators=(",", ":")))
    mol.SetProp("am1bcc_total_charge", f"{sum(charges):.8f}")
    mol.SetProp("am1bcc_tool", "AmberTools antechamber -c bcc")


def single_conformer_mol(mol: Any, conformer_id: int) -> Any:
    copy = Chem.Mol(mol)
    conformer = Chem.Conformer(mol.GetConformer(conformer_id))
    conformer.SetId(0)
    copy.RemoveAllConformers()
    copy.AddConformer(conformer, assignId=True)
    return copy


def optimize_conformers(mol: Any, conformer_ids: list[int]) -> tuple[str, dict[int, float]]:
    energy_by_conf: dict[int, float] = {}
    if bool(AllChem.MMFFHasAllMoleculeParams(mol)):
        for conf_id in conformer_ids:
            AllChem.MMFFOptimizeMolecule(mol, confId=int(conf_id), mmffVariant="MMFF94s", maxIters=500)
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            force_field = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(conf_id))
            energy_by_conf[int(conf_id)] = float(force_field.CalcEnergy()) if force_field is not None else 0.0
        return "MMFF94s", energy_by_conf
    for conf_id in conformer_ids:
        AllChem.UFFOptimizeMolecule(mol, confId=int(conf_id), maxIters=500)
        force_field = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
        energy_by_conf[int(conf_id)] = float(force_field.CalcEnergy()) if force_field is not None else 0.0
    return "UFF", energy_by_conf


def generate_benchmark_sdf(
    row: dict[str, Any],
    *,
    conformer_dir: Path,
    reference_payload: JsonObject,
    binding_site_reference: Path,
    antechamber: Path,
    amberhome: Path,
    conformer_count: int,
    charge_timeout_seconds: int,
) -> Path:
    ligand_id = str(row["ligand_id"])
    canonical = str(row["canonical_smiles"])
    base_mol = Chem.MolFromSmiles(canonical)
    if base_mol is None:
        raise ValueError(f"RDKit could not parse canonical SMILES for {ligand_id}")
    mol = Chem.AddHs(base_mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 30_071 + sum(ord(char) for char in ligand_id)
    params.useRandomCoords = True
    params.pruneRmsThresh = -1.0
    conformer_ids = [int(conf_id) for conf_id in AllChem.EmbedMultipleConfs(mol, numConfs=conformer_count, params=params)]
    if len(conformer_ids) != conformer_count:
        raise RuntimeError(f"{ligand_id}: generated {len(conformer_ids)} conformers, expected {conformer_count}")
    force_field, energy_by_conf = optimize_conformers(mol, conformer_ids)

    points = reference_points(reference_payload)
    target_array = as_array(points)
    reference_center = np.array(
        point_from_raw(reference_payload.get("binding_site_center_angstrom"), "binding_site_center_angstrom"),
        dtype=np.float64,
    )
    reference_axes = principal_axes(target_array) if len(points) >= 3 else np.eye(3, dtype=np.float64)
    rotations: dict[int, Matrix] = {}
    translations: dict[int, Vector] = {}
    for conf_id in conformer_ids:
        rotation, translation = transform_conformer_to_reference(mol, conf_id, reference_center, reference_axes)
        rotations[int(conf_id)] = rotation
        translations[int(conf_id)] = translation

    charge_source_conf = int(conformer_ids[0])
    charge_source_mol = single_conformer_mol(mol, charge_source_conf)
    charges, antechamber_log = compute_am1bcc_charges(
        charge_source_mol,
        antechamber=antechamber,
        amberhome=amberhome,
        net_charge=formal_charge(base_mol),
        timeout_seconds=charge_timeout_seconds,
    )

    output = conformer_dir / f"benchmark_{ligand_id}.sdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output))
    try:
        for conformer_index, conf_id in enumerate(conformer_ids):
            conformer_mol = single_conformer_mol(mol, conf_id)
            set_charge_properties(conformer_mol, charges)
            conformer_mol.SetProp("_Name", f"{ligand_id}_conf{conformer_index:02d}")
            conformer_mol.SetProp("ligand_id", ligand_id)
            conformer_mol.SetProp("compound_name", str(row["compound_name"]))
            conformer_mol.SetProp("canonical_smiles", canonical)
            conformer_mol.SetProp("known_activity_type", str(row["known_activity_type"]))
            conformer_mol.SetProp("known_activity_value", f"{float(row['known_activity_value']):.6g}")
            conformer_mol.SetProp("activity_units", str(row["activity_units"]))
            conformer_mol.SetProp("benchmark_conformer_index", str(conformer_index))
            conformer_mol.SetProp("force_field", force_field)
            conformer_mol.SetProp("mmff_energy_kcal_mol", f"{energy_by_conf[int(conf_id)]:.8f}")
            conformer_mol.SetProp("binding_site_reference", binding_site_reference.as_posix())
            conformer_mol.SetProp("alignment_method", "principal_axes_to_binding_site_reference_shell")
            conformer_mol.SetProp(
                "alignment_rotation_matrix",
                json.dumps(rotations[int(conf_id)], separators=(",", ":")),
            )
            conformer_mol.SetProp(
                "alignment_translation_vector",
                json.dumps(translations[int(conf_id)], separators=(",", ":")),
            )
            conformer_mol.SetProp("am1bcc_charge_source_conformer_index", str(charge_source_conf))
            conformer_mol.SetProp("am1bcc_antechamber_log_tail", " | ".join(antechamber_log.splitlines()[-10:]))
            writer.write(conformer_mol)
    finally:
        writer.close()
    return output


def existing_sdf_is_valid(path: Path, expected_conformers: int) -> bool:
    if not path.exists():
        return False
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    molecules: list[Any] = [mol for mol in supplier if mol is not None]
    if len(molecules) != expected_conformers:
        return False
    for mol in molecules:
        if not mol.HasProp("charge_method") or mol.GetProp("charge_method") != "AM1-BCC":
            return False
        if not mol.HasProp("am1bcc_charges_json"):
            return False
        if not mol.HasProp("alignment_method"):
            return False
    return True


def main() -> int:
    args = parse_args()
    if int(args.conformers) != CONFORMER_COUNT:
        raise ValueError("benchmark panel release requires exactly K=10 conformers per ligand")
    frame = write_manifest(Path(args.manifest_parquet), Path(args.manifest_csv))
    reference_payload = read_json_object(Path(args.binding_site_reference))
    written_sdfs: list[Path] = []
    for row in frame.to_dicts():
        ligand_id = str(row["ligand_id"])
        expected_sdf = Path(args.conformer_dir) / f"benchmark_{ligand_id}.sdf"
        if existing_sdf_is_valid(expected_sdf, CONFORMER_COUNT):
            written_sdfs.append(expected_sdf)
            emit(f"reused {expected_sdf} conformers={CONFORMER_COUNT} charge_method=AM1-BCC")
            continue
        sdf_path = generate_benchmark_sdf(
            row,
            conformer_dir=Path(args.conformer_dir),
            reference_payload=reference_payload,
            binding_site_reference=Path(args.binding_site_reference),
            antechamber=Path(args.antechamber),
            amberhome=Path(args.amberhome),
            conformer_count=int(args.conformers),
            charge_timeout_seconds=int(args.charge_timeout_seconds),
        )
        written_sdfs.append(sdf_path)
        emit(f"wrote {sdf_path} conformers={CONFORMER_COUNT} charge_method=AM1-BCC")
    emit(f"wrote {args.manifest_parquet} rows={frame.height}")
    emit(f"wrote {args.manifest_csv} rows={frame.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
