from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from scripts.compute_species_selectivity import (
    DEFAULT_CONSERVATION,
    compute_species_selectivity,
    compute_selectivity_for_row,
    load_conservation,
    load_receptor_residues,
    load_topology_residue_id_map,
)


def test_explicit_contacts_score_human_specific_residues() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)
    result = compute_selectivity_for_row(
        {"contact_residues_json": json.dumps([190, 149])},
        conservation,
    )
    assert result.method == "explicit_contact_residue_columns"
    assert result.evidence_level == "L2"
    assert result.species_selectivity_score == 0.5
    assert result.predicted_active_in[0:2] == ["Human", "NHP"]


def test_variant_resilience_columns_are_contact_evidence() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)
    result = compute_selectivity_for_row(
        {
            "resilience_R190Q": 0.2,
            "resilience_T149M": 1.0,
        },
        conservation,
    )
    assert result.method == "variant_resilience_sensitivity"
    assert result.species_selectivity_score == 1.0
    assert result.predicted_active_in == ["Human", "NHP"]


def test_coordinates_map_to_nearest_receptor_residue(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  LEU R 190     155.000 129.000 125.000  1.00 99.76           C\n"
        "ATOM      2  CA  THR R 149     136.000 134.000 123.000  1.00 99.76           C\n"
        "ENDMDL\n",
        encoding="utf-8",
    )
    residues = load_receptor_residues(receptor)
    conservation = load_conservation(DEFAULT_CONSERVATION)

    result = compute_selectivity_for_row(
        {"coordinates_json": json.dumps([[155.2, 129.1, 124.9]])},
        conservation,
        residues,
        contact_cutoff_a=2.0,
    )

    assert result.method == "atom_coordinate_nearest_residue_mapping"
    assert result.contact_residues.keys() == {190}
    assert result.species_selectivity_score == 1.0


def test_topology_maps_pdb_indices_to_biological_residue_ids(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdb"
    topology = tmp_path / "topology.json"
    receptor.write_text(
        "ATOM      1  CA  LEU R   0     155.000 129.000 125.000  1.00 99.76           C\n"
        "ENDMDL\n",
        encoding="utf-8",
    )
    topology.write_text(
        json.dumps({"residues": [{"residue_idx": 0, "residue_name": "LEU", "residue_id": 190}]}),
        encoding="utf-8",
    )

    assert load_topology_residue_id_map(topology) == {0: 190}
    residues = load_receptor_residues(receptor, topology)
    conservation = load_conservation(DEFAULT_CONSERVATION)
    result = compute_selectivity_for_row(
        {"coordinates_json": json.dumps([[155.2, 129.1, 124.9]])},
        conservation,
        residues,
        contact_cutoff_a=2.0,
    )

    assert result.method == "atom_coordinate_nearest_residue_mapping"
    assert result.contact_residues.keys() == {190}
    assert result.unscored_contact_residues == {}
    assert result.species_selectivity_score == 1.0


def test_coordinate_contacts_keep_unscored_provenance(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  GLY R 999     155.000 129.000 125.000  1.00 99.76           C\n"
        "ENDMDL\n",
        encoding="utf-8",
    )
    residues = load_receptor_residues(receptor)
    conservation = load_conservation(DEFAULT_CONSERVATION)

    result = compute_selectivity_for_row(
        {"coordinates_json": json.dumps([[155.2, 129.1, 124.9]]), "resilience_R190Q": 0.1},
        conservation,
        residues,
        contact_cutoff_a=2.0,
    )

    assert result.method == "atom_coordinate_nearest_residue_mapping"
    assert result.contact_residues.keys() == {999}
    assert result.unscored_contact_residues.keys() == {999}
    assert result.unscored_contact_fraction == 1.0


def test_coordinates_without_nearby_residue_do_not_fall_back_to_resilience(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  LEU R 190     155.000 129.000 125.000  1.00 99.76           C\n"
        "ENDMDL\n",
        encoding="utf-8",
    )
    residues = load_receptor_residues(receptor)
    conservation = load_conservation(DEFAULT_CONSERVATION)

    result = compute_selectivity_for_row(
        {"coordinates_json": json.dumps([[500.0, 500.0, 500.0]]), "resilience_R190Q": 0.1},
        conservation,
        residues,
        contact_cutoff_a=2.0,
    )

    assert result.method == "atom_coordinate_no_residue_within_cutoff"
    assert result.contact_residues == {}
    assert result.species_selectivity_score == 0.0


def test_compute_species_selectivity_writes_enriched_parquet(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    output = tmp_path / "species.parquet"
    pl.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCC"],
            "contact_residues_json": [json.dumps([190]), json.dumps([149])],
        }
    ).write_parquet(candidates)

    df = compute_species_selectivity(candidates, DEFAULT_CONSERVATION, output)

    assert output.exists()
    assert "species_selectivity_score" in df.columns
    assert "predicted_active_in" in df.columns
    scores = df.get_column("species_selectivity_score").to_list()
    assert scores == [1.0, 0.0]


def test_aleniglipron_reference_is_human_selective() -> None:
    """Reference contact pattern should reproduce Aleniglipron's Human/NHP selectivity."""

    conservation = load_conservation(DEFAULT_CONSERVATION)
    result = compute_selectivity_for_row(
        {
            "contact_residues_json": json.dumps(
                [190, 233, 328, 330, 339, 349, 352]
            )
        },
        conservation,
    )

    assert result.method == "explicit_contact_residue_columns"
    assert result.species_selectivity_score > 0.7
    assert result.predicted_active_in == ["Human", "NHP"]
