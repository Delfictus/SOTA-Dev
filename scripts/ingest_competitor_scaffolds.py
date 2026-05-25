#!/usr/bin/env python3
"""Generate O3A-aligned 6XOX-frame competitor scaffold conformers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A_DIR = CAMPAIGN_DIR / "track_a_generative"
DEFAULT_LIGAND_MANIFEST = CAMPAIGN_DIR / "GLP1R_LIGAND_SET_MANIFEST_v1.parquet"
DEFAULT_REFERENCE = TRACK_A_DIR / "6XOX_PRISM_compact_negative_image_pseudoligand.sdf"
DEFAULT_OUTPUT_DIR = TRACK_A_DIR / "conformers"
DEFAULT_OUTPUT_MANIFEST = TRACK_A_DIR / "competitor_scaffold_o3a_manifest.json"
TARGET_LIGANDS = {
    "ORFORGLIPRON_LY3502970": "ORFOR-PARENT",
    "DANUGLIPRON_PF06882961": "DANU-PARENT",
}

JsonObject: TypeAlias = dict[str, Any]

Chem = cast(Any, import_module("rdkit.Chem"))
AllChem = cast(Any, import_module("rdkit.Chem.AllChem"))
rdMolAlign = cast(Any, import_module("rdkit.Chem.rdMolAlign"))


@dataclass(frozen=True)
class AlignedScaffold:
    ligand_id: str
    compound_name: str
    output_path: Path
    o3a_score: float
    o3a_rmsd: float
    selected_conformer_id: int
    constrained_embed_status: str
    relaxation_method: str
    heavy_atom_count: int
    centroid_xyz: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-manifest", type=Path, default=DEFAULT_LIGAND_MANIFEST)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--conformers", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260524)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def load_single_mol(path: Path) -> Any:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = supplier[0] if len(supplier) else None
    if mol is None:
        raise RuntimeError(f"could not read SDF: {path}")
    if int(mol.GetNumConformers()) == 0:
        raise RuntimeError(f"SDF has no conformer: {path}")
    return mol


def write_single_sdf(path: Path, mol: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    writer = Chem.SDWriter(str(tmp_path))
    writer.write(mol)
    writer.close()
    tmp_path.replace(path)


def centroid_xyz(mol: Any) -> tuple[float, float, float]:
    conformer = mol.GetConformer(0)
    values = [0.0, 0.0, 0.0]
    count = 0
    for atom in mol.GetAtoms():
        if int(atom.GetAtomicNum()) <= 1:
            continue
        pos = conformer.GetAtomPosition(int(atom.GetIdx()))
        values[0] += float(pos.x)
        values[1] += float(pos.y)
        values[2] += float(pos.z)
        count += 1
    if count == 0:
        raise RuntimeError("molecule has no heavy atoms")
    return (values[0] / count, values[1] / count, values[2] / count)


def heavy_atom_count(mol: Any) -> int:
    return sum(1 for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) > 1)


def smiles_to_3d_mol(smiles: str, conformer_count: int, seed: int) -> Any:
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise RuntimeError(f"invalid SMILES: {smiles}")
    mol = Chem.AddHs(base)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    conformer_ids = [int(cid) for cid in AllChem.EmbedMultipleConfs(mol, numConfs=conformer_count, params=params)]
    if not conformer_ids:
        raise RuntimeError("ETKDGv3 failed to generate conformers")
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    for conformer_id in conformer_ids:
        if properties is not None:
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", confId=conformer_id, maxIters=300)
        else:
            AllChem.UFFOptimizeMolecule(mol, confId=conformer_id, maxIters=300)
    return mol


def one_conformer_copy(mol: Any, conformer_id: int) -> Any:
    selected = Chem.Mol(mol)
    selected.RemoveAllConformers()
    selected.AddConformer(Chem.Conformer(mol.GetConformer(conformer_id)), assignId=True)
    return selected


def score_o3a_conformers(mol: Any, reference: Any) -> tuple[Any, int, float, float]:
    if int(reference.GetNumConformers()) == 0:
        raise RuntimeError("reference molecule has no conformer")
    best_score = -math.inf
    best_rmsd = math.inf
    best_conformer_id = -1
    for conformer in mol.GetConformers():
        conformer_id = int(conformer.GetId())
        try:
            o3a = rdMolAlign.GetO3A(mol, reference, prbCid=conformer_id, refCid=0)
            rmsd = float(o3a.Align())
            score = float(o3a.Score())
        except Exception as exc:  # pragma: no cover - RDKit raises extension exceptions.
            raise RuntimeError(f"RDKit O3A failed for conformer {conformer_id}: {exc}") from exc
        if score > best_score:
            best_score = score
            best_rmsd = rmsd
            best_conformer_id = conformer_id
    if best_conformer_id < 0:
        raise RuntimeError("no O3A conformer was scored")
    return one_conformer_copy(mol, best_conformer_id), best_conformer_id, best_score, best_rmsd


def constrained_relax(mol: Any, seed: int) -> tuple[Any, str, str]:
    """Use RDKit ConstrainedEmbed when possible, then finish with MMFF/UFF."""

    try:
        core = Chem.RemoveHs(Chem.Mol(mol), sanitize=False)
        relaxed = AllChem.ConstrainedEmbed(Chem.Mol(mol), core, randomseed=int(seed), useTethers=True)
        constrained_status = "ConstrainedEmbed_success_heavy_atom_core"
    except Exception as exc:  # pragma: no cover - RDKit raises extension exceptions.
        relaxed = Chem.Mol(mol)
        constrained_status = f"ConstrainedEmbed_fallback:{type(exc).__name__}"
    properties = AllChem.MMFFGetMoleculeProperties(relaxed, mmffVariant="MMFF94s")
    if properties is not None:
        AllChem.MMFFOptimizeMolecule(relaxed, mmffVariant="MMFF94s", confId=0, maxIters=500)
        method = "MMFF94s_post_o3a_relaxation"
    else:
        AllChem.UFFOptimizeMolecule(relaxed, confId=0, maxIters=500)
        method = "UFF_post_o3a_relaxation"
    return relaxed, constrained_status, method


def load_target_ligands(path: Path) -> list[dict[str, str]]:
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("ligand_id").is_in(list(TARGET_LIGANDS)))
        .select(["ligand_id", "compound_name", "canonical_smiles"])
        .collect()
    )
    rows = cast(list[dict[str, object]], frame.to_dicts())
    missing = sorted(set(TARGET_LIGANDS).difference(str(row["ligand_id"]) for row in rows))
    if missing:
        raise RuntimeError(f"ligand manifest missing required ligands: {missing}")
    return [
        {
            "ligand_id": str(row["ligand_id"]),
            "compound_name": str(row["compound_name"]),
            "canonical_smiles": str(row["canonical_smiles"]),
        }
        for row in rows
    ]


def align_ligand(row: dict[str, str], reference: Any, output_dir: Path, conformers: int, seed: int) -> AlignedScaffold:
    ligand_id = row["ligand_id"]
    prefix = TARGET_LIGANDS[ligand_id]
    mol = smiles_to_3d_mol(row["canonical_smiles"], conformers, seed + len(ligand_id))
    aligned, conformer_id, score, rmsd = score_o3a_conformers(mol, reference)
    relaxed, constrained_status, relaxation_method = constrained_relax(aligned, seed + conformer_id)
    relaxed.SetProp("ligand_id", ligand_id)
    relaxed.SetProp("compound_name", row["compound_name"])
    relaxed.SetProp("alignment_method", "compact_pseudoligand_o3a")
    relaxed.SetProp("o3a_score", f"{score:.6f}")
    output_path = output_dir / f"{prefix}_6XOX_o3a.sdf"
    write_single_sdf(output_path, relaxed)
    return AlignedScaffold(
        ligand_id=ligand_id,
        compound_name=row["compound_name"],
        output_path=output_path,
        o3a_score=score,
        o3a_rmsd=rmsd,
        selected_conformer_id=conformer_id,
        constrained_embed_status=constrained_status,
        relaxation_method=relaxation_method,
        heavy_atom_count=heavy_atom_count(relaxed),
        centroid_xyz=centroid_xyz(relaxed),
    )


def main() -> int:
    args = parse_args()
    reference = load_single_mol(args.reference)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_target_ligands(Path(args.ligand_manifest))
    aligned = [
        align_ligand(row, reference, output_dir, int(args.conformers), int(args.seed))
        for row in sorted(rows, key=lambda item: item["ligand_id"])
    ]
    output_hashes = {item.output_path.as_posix(): sha256_file(item.output_path) for item in aligned}
    manifest: JsonObject = {
        "schema_version": "PRISM.competitor_scaffold_o3a_manifest.v1",
        "epistemic_class": "DERIVED",
        "selected_alignment_method": "compact_pseudoligand_o3a",
        "reference_pseudoligand": Path(args.reference).as_posix(),
        "reference_sha256": sha256_file(Path(args.reference)),
        "ligand_manifest": Path(args.ligand_manifest).as_posix(),
        "ligand_manifest_sha256": sha256_file(Path(args.ligand_manifest)),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "conformer_count_requested": int(args.conformers),
        "outputs": [
            {
                "ligand_id": item.ligand_id,
                "compound_name": item.compound_name,
                "output_path": item.output_path.as_posix(),
                "o3a_score": item.o3a_score,
                "o3a_rmsd_A": item.o3a_rmsd,
                "selected_conformer_id": item.selected_conformer_id,
                "constrained_embed_status": item.constrained_embed_status,
                "relaxation_method": item.relaxation_method,
                "heavy_atom_count": item.heavy_atom_count,
                "centroid_xyz": list(item.centroid_xyz),
                "sha256": output_hashes[item.output_path.as_posix()],
            }
            for item in aligned
        ],
        "notes": [
            "O3A was run against the compact PRISM negative-image pseudo-ligand, not a true co-crystal ligand.",
            "Outputs are DERIVED aligned scaffold poses for policy initialization and remain subject to PRISM field scoring.",
        ],
    }
    atomic_write_text(Path(args.manifest), json.dumps(manifest, indent=2) + "\n")
    for item in aligned:
        print(
            "competitor_scaffold_o3a_written "
            f"ligand_id={item.ligand_id} output={item.output_path} "
            f"o3a_score={item.o3a_score:.6f} rmsd_A={item.o3a_rmsd:.6f} "
            f"constrained_embed_status={item.constrained_embed_status}"
        )
    print(f"competitor_scaffold_manifest_written path={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
