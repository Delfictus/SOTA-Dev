from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_grid(
    path: Path,
    scaffold: str,
    classes: list[str],
    condition: str | None = None,
    *,
    voxel_ids: list[int] | None = None,
    bonuses: list[float] | None = None,
    cold_means: list[float] | None = None,
    warm_means: list[float] | None = None,
) -> None:
    voxel_ids = voxel_ids or list(range(len(classes)))
    bonuses = bonuses or [3.0 if value == "thermally_activated" else 0.0 for value in classes]
    cold_means = cold_means or [0.0] * len(classes)
    warm_means = warm_means or [1.0 if value == "thermally_activated" else 0.0 for value in classes]
    pl.DataFrame(
        {
            "campaign_id": ["test"] * len(classes),
            "condition_id": [condition or f"glp1r_6XOX_SCAFFOLD_{scaffold}"] * len(classes),
            "voxel_idx": voxel_ids,
            "x_idx": [0, 1, 0, 1][: len(classes)],
            "y_idx": [0, 0, 1, 1][: len(classes)],
            "z_idx": [0] * len(classes),
            "hit_count_cold_mean": cold_means,
            "hit_count_warm_mean": warm_means,
            "variance_class": classes,
            "variance_classification": classes,
            "scaffold_consensus_bonus": bonuses,
        }
    ).write_parquet(path)


def write_mapping(path: Path) -> None:
    conditions = {
        name: {
            "nx": 2,
            "ny": 2,
            "nz": 1,
            "origin_xyz_angstrom": [0.0, 0.0, 0.0],
            "spacing_angstrom": 1.0,
        }
        for name in (
            "glp1r_6XOX_WT",
            "glp1r_6XOX_SCAFFOLD_ALENI",
            "glp1r_6XOX_SCAFFOLD_DANU",
            "glp1r_6XOX_SCAFFOLD_ORFOR",
        )
    }
    path.write_text(json.dumps({"conditions": conditions}), encoding="utf-8")


def test_compute_scaffold_consensus_grid_marks_invariant_voxels(tmp_path: Path) -> None:
    grid_dir = tmp_path / "grids"
    grid_dir.mkdir()
    grids = {
        "ALENI": grid_dir / "aleni.parquet",
        "DANU": grid_dir / "danu.parquet",
        "ORFOR": grid_dir / "orfor.parquet",
    }
    write_grid(grids["ALENI"], "ALENI", ["thermally_activated", "thermally_activated", "stable_occupied"])
    write_grid(grids["DANU"], "DANU", ["thermally_activated", "thermally_activated", "void"])
    write_grid(grids["ORFOR"], "ORFOR", ["thermally_activated", "stable_occupied", "thermally_destabilized"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scaffolds": [
                    {"scaffold_id": scaffold, "grid_path": str(path)}
                    for scaffold, path in grids.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "consensus.parquet"
    report = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/compute_scaffold_consensus_grid.py",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    df = pl.read_parquet(output)
    assert df.filter(pl.col("scaffold_consensus_type") == "SCAFFOLD_INVARIANT").height == 1
    invariant = df.filter(pl.col("scaffold_consensus_type") == "SCAFFOLD_INVARIANT").row(0, named=True)
    assert invariant["variance_classification"] == "thermally_activated"
    assert invariant["scaffold_consensus_bonus"] == 3.0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["metrics"]["scaffold_invariant_activated_voxels"] == 1


def test_compute_scaffold_consensus_rejects_duplicate_scaffold_ids(tmp_path: Path) -> None:
    grid = tmp_path / "grid.parquet"
    write_grid(grid, "DUP", ["thermally_activated"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scaffolds": [
                    {"scaffold_id": "DUP", "grid_path": str(grid)},
                    {"scaffold_id": "dup", "grid_path": str(grid)},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_scaffold_consensus_grid.py",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out.parquet"),
            "--report",
            str(tmp_path / "report.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "duplicate scaffold_id in manifest" in result.stderr


def test_compute_scaffold_consensus_rejects_voxel_set_mismatch(tmp_path: Path) -> None:
    grid_a = tmp_path / "a.parquet"
    grid_b = tmp_path / "b.parquet"
    write_grid(grid_a, "A", ["thermally_activated"], voxel_ids=[0])
    write_grid(grid_b, "B", ["thermally_activated", "void"], voxel_ids=[0, 1])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scaffolds": [
                    {"scaffold_id": "A", "grid_path": str(grid_a)},
                    {"scaffold_id": "B", "grid_path": str(grid_b)},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_scaffold_consensus_grid.py",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out.parquet"),
            "--report",
            str(tmp_path / "report.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "scaffold grid voxel_idx mismatch" in result.stderr


def test_compute_scaffold_consensus_rejects_non_finite_signal_values(tmp_path: Path) -> None:
    grid_a = tmp_path / "a.parquet"
    grid_b = tmp_path / "b.parquet"
    write_grid(grid_a, "A", ["thermally_activated"], cold_means=[float("inf")])
    write_grid(grid_b, "B", ["thermally_activated"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scaffolds": [
                    {"scaffold_id": "A", "grid_path": str(grid_a)},
                    {"scaffold_id": "B", "grid_path": str(grid_b)},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compute_scaffold_consensus_grid.py",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out.parquet"),
            "--report",
            str(tmp_path / "report.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "hit_count_cold_mean" in result.stderr
    assert "non-finite" in result.stderr


def test_build_scaffold_consensus_survivor_corpus_adds_bonus(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    write_mapping(mapping)
    consensus = tmp_path / "consensus.parquet"
    pl.DataFrame(
        {
            "voxel_idx": [0],
            "variance_classification": ["thermally_activated"],
            "scaffold_consensus_bonus": [3.0],
        }
    ).write_parquet(consensus)
    survivors = tmp_path / "survivors.parquet"
    pl.DataFrame(
        {
            "canonical_smiles": ["CCO"],
            "coordinates_json": [json.dumps([[0.5, 0.5, 0.5]])],
        }
    ).write_parquet(survivors)
    output = tmp_path / "out.parquet"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_scaffold_consensus_survivor_corpus.py",
            "--survivors",
            str(survivors),
            "--consensus-grid",
            str(consensus),
            "--grid-mapping",
            str(mapping),
            "--output",
            str(output),
            "--report",
            str(tmp_path / "report.json"),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    df = pl.read_parquet(output)
    assert df["scaffold_consensus_bonus"].to_list() == [3.0]
    assert df["consensus_complement_bonus"].to_list() == [3.0]


def test_build_scaffold_consensus_survivor_corpus_rejects_nan_bonus(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    write_mapping(mapping)
    consensus = tmp_path / "consensus.parquet"
    pl.DataFrame(
        {
            "voxel_idx": [0],
            "variance_classification": ["thermally_activated"],
            "scaffold_consensus_bonus": [float("nan")],
        }
    ).write_parquet(consensus)
    survivors = tmp_path / "survivors.parquet"
    pl.DataFrame(
        {
            "canonical_smiles": ["CCO"],
            "coordinates_json": [json.dumps([[0.5, 0.5, 0.5]])],
        }
    ).write_parquet(survivors)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_scaffold_consensus_survivor_corpus.py",
            "--survivors",
            str(survivors),
            "--consensus-grid",
            str(consensus),
            "--grid-mapping",
            str(mapping),
            "--output",
            str(tmp_path / "out.parquet"),
            "--report",
            str(tmp_path / "report.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "scaffold_consensus_bonus must be finite" in result.stderr


def test_build_scaffold_consensus_survivor_corpus_rejects_duplicate_voxels(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    write_mapping(mapping)
    consensus = tmp_path / "consensus.parquet"
    pl.DataFrame(
        {
            "voxel_idx": [0, 0],
            "variance_classification": ["thermally_activated", "thermally_activated"],
            "scaffold_consensus_bonus": [3.0, 3.0],
        }
    ).write_parquet(consensus)
    survivors = tmp_path / "survivors.parquet"
    pl.DataFrame(
        {
            "canonical_smiles": ["CCO"],
            "coordinates_json": [json.dumps([[0.5, 0.5, 0.5]])],
        }
    ).write_parquet(survivors)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_scaffold_consensus_survivor_corpus.py",
            "--survivors",
            str(survivors),
            "--consensus-grid",
            str(consensus),
            "--grid-mapping",
            str(mapping),
            "--output",
            str(tmp_path / "out.parquet"),
            "--report",
            str(tmp_path / "report.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "duplicate voxel_idx" in result.stderr


def test_cross_screen_multi_scaffold_uses_grid_scores(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    write_mapping(mapping)
    grid_dir = tmp_path / "grids"
    grid_dir.mkdir()
    grids = {
        "ALENI": grid_dir / "aleni.parquet",
        "DANU": grid_dir / "danu.parquet",
        "ORFOR": grid_dir / "orfor.parquet",
    }
    write_grid(grids["ALENI"], "ALENI", ["stable_occupied"])
    write_grid(grids["DANU"], "DANU", ["thermally_activated"])
    write_grid(grids["ORFOR"], "ORFOR", ["thermally_activated"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scaffolds": [
                    {"scaffold_id": scaffold, "grid_path": str(path)}
                    for scaffold, path in grids.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.parquet"
    pl.DataFrame(
        {
            "candidate_id": ["cand_1"],
            "canonical_smiles": ["CCO"],
            "coordinates_json": [json.dumps([[0.5, 0.5, 0.5]])],
        }
    ).write_parquet(candidates)
    output = tmp_path / "cross.parquet"
    report = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/cross_screen_multi_scaffold.py",
            "--input",
            str(candidates),
            "--scaffold-manifest",
            str(manifest),
            "--grid-mapping",
            str(mapping),
            "--positive-threshold",
            "1.0",
            "--output",
            str(output),
            "--report",
            str(report),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    df = pl.read_parquet(output)
    assert df["cross_scaffold_evidence"].to_list() == ["THERMODYNAMIC_SCAFFOLD_BOUND_GRID"]
    assert df["n_scaffolds_positive"].to_list() == [2]
    assert df["positive_danu"].to_list() == [True]
    assert df["positive_orfor"].to_list() == [True]


def test_cross_screen_rejects_negative_score_atom_offset(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    pl.DataFrame(
        {
            "candidate_id": ["cand_1"],
            "canonical_smiles": ["CCO"],
            "coordinates_json": [json.dumps([[0.5, 0.5, 0.5]])],
        }
    ).write_parquet(candidates)
    manifest = tmp_path / "missing_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/cross_screen_multi_scaffold.py",
            "--input",
            str(candidates),
            "--scaffold-manifest",
            str(manifest),
            "--score-atom-offset",
            "-1",
            "--output",
            str(tmp_path / "out.parquet"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--score-atom-offset must be non-negative" in result.stderr


def test_cross_screen_rejects_duplicate_manifest_scaffold_ids(tmp_path: Path) -> None:
    grid = tmp_path / "grid.parquet"
    write_grid(grid, "DUP", ["thermally_activated"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scaffolds": [
                    {"scaffold_id": "DUP", "grid_path": str(grid)},
                    {"scaffold_id": "dup", "grid_path": str(grid)},
                ]
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.parquet"
    pl.DataFrame(
        {
            "candidate_id": ["cand_1"],
            "canonical_smiles": ["CCO"],
            "coordinates_json": [json.dumps([[0.5, 0.5, 0.5]])],
        }
    ).write_parquet(candidates)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/cross_screen_multi_scaffold.py",
            "--input",
            str(candidates),
            "--scaffold-manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out.parquet"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "duplicate scaffold_id in manifest" in result.stderr


def test_cross_screen_rejects_nan_grid_values(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    write_mapping(mapping)
    grid = tmp_path / "grid.parquet"
    write_grid(grid, "ALENI", ["thermally_activated"], bonuses=[float("nan")])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"scaffolds": [{"scaffold_id": "ALENI", "grid_path": str(grid)}]}),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.parquet"
    pl.DataFrame(
        {
            "candidate_id": ["cand_1"],
            "canonical_smiles": ["CCO"],
            "coordinates_json": [json.dumps([[0.5, 0.5, 0.5]])],
        }
    ).write_parquet(candidates)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/cross_screen_multi_scaffold.py",
            "--input",
            str(candidates),
            "--scaffold-manifest",
            str(manifest),
            "--grid-mapping",
            str(mapping),
            "--output",
            str(tmp_path / "out.parquet"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "expected finite float value" in result.stderr


def test_cross_screen_rejects_zero_candidates(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    write_mapping(mapping)
    grid = tmp_path / "grid.parquet"
    write_grid(grid, "ALENI", ["thermally_activated"])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"scaffolds": [{"scaffold_id": "ALENI", "grid_path": str(grid)}]}),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.parquet"
    pl.DataFrame(
        {
            "candidate_id": pl.Series("candidate_id", [], dtype=pl.Utf8),
            "canonical_smiles": pl.Series("canonical_smiles", [], dtype=pl.Utf8),
            "coordinates_json": pl.Series("coordinates_json", [], dtype=pl.Utf8),
        }
    ).write_parquet(candidates)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/cross_screen_multi_scaffold.py",
            "--input",
            str(candidates),
            "--scaffold-manifest",
            str(manifest),
            "--grid-mapping",
            str(mapping),
            "--output",
            str(tmp_path / "out.parquet"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "cross-screen input contains zero candidates" in result.stderr
