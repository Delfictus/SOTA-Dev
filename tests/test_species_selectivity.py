from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from scripts.compute_species_selectivity import (
    DEFAULT_CONSERVATION,
    DEFAULT_RECEPTOR_PDB,
    REGION_WEIGHTS,
    ConservationRecord,
    ReceptorResidue,
    compute_species_selectivity,
    compute_species_selectivity_v3,
    compute_selectivity_for_row,
    distance_decay,
    extract_variant_contacts,
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
    assert abs(result.species_selectivity_score - (3.0 / 13.0)) < 1.0e-9
    assert result.region_classes == {190: "allosteric", 149: "pocket_contact"}
    assert result.region_weighted_contacts == {190: 3.0, 149: 10.0}
    assert result.divergent_residue_weights == {190: 3.0}
    assert result.predicted_active_in[0:2] == ["Human", "NHP"]


def test_region_weighting_matches_d07_four_region_scheme() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)

    assert conservation[149].region_class == "pocket_contact"
    assert conservation[149].region_weight == REGION_WEIGHTS["pocket_contact"] == 10.0
    assert conservation[131].region_class == "ecd"
    assert conservation[131].region_weight == REGION_WEIGHTS["ecd"] == 5.0
    assert conservation[141].region_class == "ecd"
    assert conservation[141].region_weight == REGION_WEIGHTS["ecd"] == 5.0
    assert conservation[190].region_class == "allosteric"
    assert conservation[190].region_weight == REGION_WEIGHTS["allosteric"] == 3.0
    assert conservation[257].region_class == "surface"
    assert conservation[257].region_weight == REGION_WEIGHTS["surface"] == 0.0
    assert conservation[190].divergent_species_count == 3


def test_repeated_residue_contacts_are_saturated() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)
    result = compute_selectivity_for_row(
        {"contact_residues_json": json.dumps([149, 149, 149, 190])},
        conservation,
    )

    assert result.region_weighted_contacts == {149: 10.0, 190: 3.0}
    assert abs(result.species_selectivity_score - (3.0 / 13.0)) < 1.0e-9


def test_coordinate_contact_distance_decay_is_monotonic() -> None:
    assert distance_decay(0.0, 8.0) == 1.0
    assert distance_decay(2.0, 8.0) > distance_decay(5.0, 8.0)
    assert distance_decay(9.0, 8.0) == 0.0


def test_coordinate_contacts_are_residue_centric_not_nearest_atom_only(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  LEU R 190       0.000   0.000   0.000  1.00 99.76           C\n"
        "ATOM      2  CA  THR R 149       1.000   0.000   0.000  1.00 99.76           C\n"
        "ENDMDL\n",
        encoding="utf-8",
    )
    residues = load_receptor_residues(receptor)
    conservation = load_conservation(DEFAULT_CONSERVATION)

    result = compute_selectivity_for_row(
        {"coordinates_json": json.dumps([[0.2, 0.0, 0.0]])},
        conservation,
        residues,
        contact_cutoff_a=2.0,
    )

    assert result.contact_residues.keys() == {190, 149}
    assert result.region_classes == {190: "allosteric", 149: "pocket_contact"}


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

    assert result.method == "atom_coordinate_residue_distance_decay"
    assert result.contact_residues.keys() == {190}
    assert result.species_selectivity_score == 1.0


def test_conserved_ecd_contact_contributes_to_denominator() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)
    result = compute_selectivity_for_row(
        {"contact_residues_json": json.dumps([141, 190])},
        conservation,
    )

    assert result.region_classes == {141: "ecd", 190: "allosteric"}
    assert result.region_weighted_contacts == {141: 5.0, 190: 3.0}
    assert abs(result.species_selectivity_score - (3.0 / 8.0)) < 1.0e-9


def test_ecd_classification_ignores_note_substrings_outside_domain_range() -> None:
    conservation = {
        190: ConservationRecord(
            190,
            human_aa="A",
            nhp_aa="A",
            rat_aa="V",
            mouse_aa="V",
            dog_aa="V",
            conservation_score=0.2,
            pocket_contact=False,
            allosteric_relevance="high",
            note="not_ecd_control",
        ),
        141: ConservationRecord(
            141,
            human_aa="A",
            nhp_aa="A",
            rat_aa="A",
            mouse_aa="A",
            dog_aa="A",
            conservation_score=1.0,
            pocket_contact=False,
            allosteric_relevance="low",
            note="",
        ),
    }
    result = compute_selectivity_for_row(
        {"contact_residues_json": json.dumps([190, 141])},
        conservation,
    )

    assert result.region_classes == {190: "allosteric", 141: "ecd"}
    assert result.region_weighted_contacts == {190: 3.0, 141: 5.0}
    assert abs(result.species_selectivity_score - (3.0 / 8.0)) < 1.0e-9


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

    assert result.method == "atom_coordinate_residue_distance_decay"
    assert result.contact_residues.keys() == {190}
    assert result.unscored_contact_residues == {}
    assert result.species_selectivity_score == 1.0


def test_topology_file_is_strict_when_provided(tmp_path: Path) -> None:
    missing = tmp_path / "missing_topology.json"
    with pytest.raises(ValueError, match="topology file not found"):
        load_topology_residue_id_map(missing)

    malformed = tmp_path / "malformed_topology.json"
    malformed.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid topology JSON"):
        load_topology_residue_id_map(malformed)

    missing_residues = tmp_path / "missing_residues.json"
    missing_residues.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing residues list"):
        load_topology_residue_id_map(missing_residues)

    wrong_shape = tmp_path / "wrong_shape_topology.json"
    wrong_shape.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="topology JSON must be an object"):
        load_topology_residue_id_map(wrong_shape)

    duplicate = tmp_path / "duplicate_topology.json"
    duplicate.write_text(
        json.dumps(
            {
                "residues": [
                    {"residue_idx": 1, "residue_name": "LEU", "residue_id": 190},
                    {"residue_idx": 1, "residue_name": "THR", "residue_id": 149},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate topology residue_idx"):
        load_topology_residue_id_map(duplicate)

    invalid_cases = [
        {"residue_idx": True, "residue_name": "LEU", "residue_id": 190},
        {"residue_idx": 1.9, "residue_name": "LEU", "residue_id": 190},
        {"residue_idx": "1", "residue_name": "LEU", "residue_id": 190},
        {"residue_idx": -1, "residue_name": "LEU", "residue_id": 190},
        {"residue_idx": 1, "residue_name": "LEU", "residue_id": "190"},
        {"residue_idx": 1, "residue_name": "LEU", "residue_id": 0},
        {"residue_idx": 1, "residue_id": 190},
        {"residue_idx": 1, "residue_name": "", "residue_id": 190},
    ]
    for idx, residue in enumerate(invalid_cases):
        path = tmp_path / f"invalid_topology_{idx}.json"
        path.write_text(json.dumps({"residues": [residue]}), encoding="utf-8")
        with pytest.raises(ValueError, match="malformed topology residue entry|invalid topology"):
            load_topology_residue_id_map(path)

    duplicate_residue_id = tmp_path / "duplicate_residue_id_topology.json"
    duplicate_residue_id.write_text(
        json.dumps(
            {
                "residues": [
                    {"residue_idx": 1, "residue_name": "LEU", "residue_id": 190},
                    {"residue_idx": 2, "residue_name": "THR", "residue_id": 190},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate topology residue_id"):
        load_topology_residue_id_map(duplicate_residue_id)

    valid_ligand_skip = tmp_path / "valid_ligand_skip.json"
    valid_ligand_skip.write_text(
        json.dumps(
            {
                "residues": [
                    {"residue_idx": 1, "residue_name": "LEU", "residue_id": 190},
                    {"residue_idx": 380, "residue_name": "ALN", "residue_id": 190},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_topology_residue_id_map(valid_ligand_skip) == {1: 190}

    malformed_ligand = tmp_path / "malformed_ligand_topology.json"
    malformed_ligand.write_text(
        json.dumps(
            {
                "residues": [
                    {"residue_idx": 1, "residue_name": "LEU", "residue_id": 190},
                    {"residue_idx": True, "residue_name": "ALN", "residue_id": -1},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed topology residue entry"):
        load_topology_residue_id_map(malformed_ligand)


def test_topology_validated_even_when_receptor_pdb_missing(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    output = tmp_path / "species.parquet"
    bad_topology = tmp_path / "bad_topology.json"
    pl.DataFrame({"canonical_smiles": ["CCO"], "contact_residues_json": [json.dumps([190])]}).write_parquet(
        candidates
    )
    bad_topology.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid topology JSON"):
        compute_species_selectivity_v3(
            candidates,
            DEFAULT_CONSERVATION,
            output,
            tmp_path / "missing_receptor.pdb",
            bad_topology,
        )


def test_receptor_topology_mode_rejects_unmapped_ca_indices(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdb"
    topology = tmp_path / "topology.json"
    receptor.write_text(
        "ATOM      1  CA  LEU R 190       0.000   0.000   0.000  1.00 99.76           C\n"
        "ENDMDL\n",
        encoding="utf-8",
    )
    topology.write_text(
        json.dumps({"residues": [{"residue_idx": 0, "residue_name": "LEU", "residue_id": 149}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="residue index missing from topology: 190"):
        load_receptor_residues(receptor, topology)


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

    assert result.method == "atom_coordinate_residue_distance_decay+variant_resilience_sensitivity"
    assert result.contact_residues.keys() == {999, 190}
    assert result.unscored_contact_residues.keys() == {999}
    assert 0.0 < result.unscored_contact_fraction < 1.0
    assert result.species_selectivity_score == 1.0


def test_coordinates_without_nearby_residue_can_still_use_variant_evidence(tmp_path: Path) -> None:
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

    assert result.method == "variant_resilience_sensitivity"
    assert result.contact_residues.keys() == {190}
    assert result.species_selectivity_score == 1.0


def test_invalid_candidate_evidence_does_not_use_conservation_fallback() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)

    result = compute_selectivity_for_row({}, conservation)
    assert result.method == "candidate_contact_evidence_empty_or_invalid"
    assert result.contact_residues == {}
    assert result.species_selectivity_score == 0.0
    assert result.predicted_active_in == []

    for row in (
        {"coordinates_json": ""},
        {"coordinates_json": None},
        {"contact_residues_json": "[]"},
        {"contact_residues_json": ""},
        {"contact_residues_json": None},
        {"resilience_R190Q": 1.0},
        {"resilience_R190Q": None},
    ):
        result = compute_selectivity_for_row(row, conservation)
        assert result.method == "candidate_contact_evidence_empty_or_invalid"
        assert result.contact_residues == {}
        assert result.species_selectivity_score == 0.0
        assert result.predicted_active_in == []


def test_nullable_resilience_with_missing_conservation_fails_closed() -> None:
    result = compute_selectivity_for_row({"resilience_R190Q": None}, None)
    assert result.method == "candidate_contact_evidence_empty_or_invalid"
    assert result.contact_residues == {}
    assert result.species_selectivity_score == 0.0
    assert result.predicted_active_in == []


def test_malformed_contact_json_is_rejected() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)

    with pytest.raises(ValueError):
        compute_selectivity_for_row({"contact_residues_json": "[190"}, conservation)

    with pytest.raises(ValueError, match="invalid JSON constant"):
        compute_selectivity_for_row({"contact_residues_json": "[NaN, 190]"}, conservation)

    with pytest.raises(ValueError, match="non-integer residue id"):
        compute_selectivity_for_row({"contact_residues_json": json.dumps(["bad residue R190Q"])}, conservation)

    with pytest.raises(ValueError, match="boolean token"):
        compute_selectivity_for_row({"contact_residues_json": json.dumps([True, 190])}, conservation)

    with pytest.raises(ValueError, match="non-integer residue id"):
        compute_selectivity_for_row({"contact_residues_json": json.dumps([190.5, 190])}, conservation)

    with pytest.raises(ValueError, match="must be a list"):
        compute_selectivity_for_row({"contact_residues_json": json.dumps({"residue": 190})}, conservation)

    with pytest.raises(ValueError, match="must be a list"):
        compute_selectivity_for_row({"contact_residues_json": "190"}, conservation)

    with pytest.raises(ValueError, match="must be a list"):
        compute_selectivity_for_row({"contact_residues_json": (190,)}, conservation)

    with pytest.raises(ValueError, match="must be a list"):
        compute_selectivity_for_row({"contact_residues_json": {190}}, conservation)

    with pytest.raises(ValueError, match="non-positive residue id"):
        compute_selectivity_for_row({"contact_residues_json": json.dumps([0, 190])}, conservation)


def test_malformed_coordinate_json_is_rejected() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)
    residues = [ReceptorResidue(190, "LEU", (0.0, 0.0, 0.0))]

    with pytest.raises(ValueError):
        compute_selectivity_for_row({"coordinates_json": "[1"}, conservation, residues)

    with pytest.raises(ValueError, match="coordinate row"):
        compute_selectivity_for_row({"coordinates_json": json.dumps([1])}, conservation, residues)

    with pytest.raises(ValueError, match="coordinate triples"):
        compute_selectivity_for_row({"coordinates_json": ((0, 0, 0),)}, conservation, residues)

    with pytest.raises(ValueError, match="malformed coordinate row"):
        compute_selectivity_for_row({"coordinates_json": [(0, 0, 0)]}, conservation, residues)

    with pytest.raises(ValueError, match="malformed coordinate row"):
        compute_selectivity_for_row({"coordinates_json": [{0, 0, 0}]}, conservation, residues)

    with pytest.raises(ValueError, match="exactly 3 values"):
        compute_selectivity_for_row({"coordinates_json": json.dumps([[0, 0, 0, 999]])}, conservation, residues)

    with pytest.raises(ValueError, match="exactly x/y/z keys"):
        compute_selectivity_for_row(
            {"coordinates_json": json.dumps([{"x": 0, "y": 0, "z": 0, "extra": 999}])},
            conservation,
            residues,
        )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        compute_selectivity_for_row(
            {"coordinates_json": '[{"x":0,"x":999,"y":0,"z":0}]'},
            conservation,
            residues,
        )

    with pytest.raises(ValueError, match="non-numeric coordinate"):
        compute_selectivity_for_row({"coordinates_json": json.dumps([["0", "0", "0"]])}, conservation, residues)

    with pytest.raises(ValueError, match="non-numeric coordinate"):
        compute_selectivity_for_row(
            {"coordinates_json": json.dumps([{"x": "0", "y": 0, "z": 0}])},
            conservation,
            residues,
        )


def test_coordinate_evidence_requires_receptor_residues() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)

    with pytest.raises(ValueError, match="receptor residues are unavailable"):
        compute_selectivity_for_row(
            {"coordinates_json": json.dumps([[155.2, 129.1, 124.9]])},
            conservation,
            receptor_residues=[],
        )


def test_empty_conservation_file_rejected(tmp_path: Path) -> None:
    conservation = tmp_path / "empty_conservation.csv"
    conservation.write_text(
        "residue_position,human_aa,nhp_aa,rat_aa,mouse_aa,dog_aa,"
        "conservation_score,pocket_contact,allosteric_relevance,note\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains no records"):
        load_conservation(conservation)


def test_malformed_conservation_amino_acids_rejected(tmp_path: Path) -> None:
    conservation = tmp_path / "blank_aa_conservation.csv"
    conservation.write_text(
        "residue_position,human_aa,nhp_aa,rat_aa,mouse_aa,dog_aa,"
        "conservation_score,pocket_contact,allosteric_relevance,note\n"
        "900,A,A,,A,A,1.0,yes,high,blank_rat_aa\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid rat_aa"):
        load_conservation(conservation)


def test_duplicate_conservation_rows_rejected(tmp_path: Path) -> None:
    conservation = tmp_path / "duplicate_conservation.csv"
    conservation.write_text(
        "residue_position,human_aa,nhp_aa,rat_aa,mouse_aa,dog_aa,"
        "conservation_score,pocket_contact,allosteric_relevance,note\n"
        "190,L,L,A,A,A,0.2,no,high,row1\n"
        "190,L,L,L,L,L,1.0,no,low,row2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate conservation residue_position"):
        load_conservation(conservation)


def test_malformed_conservation_csv_structure_rejected(tmp_path: Path) -> None:
    duplicate_header = tmp_path / "duplicate_header.csv"
    duplicate_header.write_text(
        "residue_position,residue_position,human_aa,nhp_aa,rat_aa,mouse_aa,dog_aa,"
        "conservation_score,pocket_contact,allosteric_relevance,note\n"
        "999,190,R,R,H,H,Q,0.60,no,high,note\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate headers"):
        load_conservation(duplicate_header)

    extra_field = tmp_path / "extra_field.csv"
    extra_field.write_text(
        "residue_position,human_aa,nhp_aa,rat_aa,mouse_aa,dog_aa,"
        "conservation_score,pocket_contact,allosteric_relevance,note\n"
        "190,R,R,H,H,Q,0.60,no,high,note,extra\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="extra fields"):
        load_conservation(extra_field)

    nonpositive = tmp_path / "nonpositive_residue.csv"
    nonpositive.write_text(
        "residue_position,human_aa,nhp_aa,rat_aa,mouse_aa,dog_aa,"
        "conservation_score,pocket_contact,allosteric_relevance,note\n"
        "0,R,R,H,H,Q,0.60,no,high,note\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid residue_position"):
        load_conservation(nonpositive)

    missing_required = tmp_path / "missing_required_value.csv"
    missing_required.write_text(
        "residue_position,human_aa,nhp_aa,rat_aa,mouse_aa,dog_aa,"
        "conservation_score,pocket_contact,allosteric_relevance,note\n"
        "190,R,R,H,H,Q,0.60,no\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing value for required column"):
        load_conservation(missing_required)

    missing_note = tmp_path / "missing_optional_note.csv"
    missing_note.write_text(
        "residue_position,human_aa,nhp_aa,rat_aa,mouse_aa,dog_aa,"
        "conservation_score,pocket_contact,allosteric_relevance,note\n"
        "900,A,A,V,V,V,0.0,no,low\n",
        encoding="utf-8",
    )
    records = load_conservation(missing_note)
    assert records[900].note == ""
    assert records[900].region_class == "surface"


def test_negative_contact_cutoff_rejected() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)

    with pytest.raises(ValueError, match="positive finite"):
        compute_selectivity_for_row({"contact_residues_json": json.dumps([190])}, conservation, contact_cutoff_a=-1.0)


def test_nhp_divergence_not_predicted_active_for_human_specific_high_score() -> None:
    conservation = {
        900: ConservationRecord(
            residue_position=900,
            human_aa="A",
            nhp_aa="V",
            rat_aa="V",
            mouse_aa="V",
            dog_aa="V",
            conservation_score=0.0,
            pocket_contact=True,
            allosteric_relevance="high",
            note="synthetic_human_specific",
        )
    }

    result = compute_selectivity_for_row({"contact_residues_json": json.dumps([900])}, conservation)

    assert result.species_selectivity_score == 1.0
    assert result.predicted_active_in == ["Human"]


def test_nhp_divergence_not_predicted_active_for_low_score() -> None:
    conservation = {
        900: ConservationRecord(
            residue_position=900,
            human_aa="A",
            nhp_aa="V",
            rat_aa="A",
            mouse_aa="A",
            dog_aa="A",
            conservation_score=0.8,
            pocket_contact=True,
            allosteric_relevance="high",
            note="synthetic_nhp_divergent_universal_non_nhp",
        )
    }

    result = compute_selectivity_for_row({"contact_residues_json": json.dumps([900])}, conservation)

    assert result.species_selectivity_score == 0.0
    assert result.predicted_active_in == ["Human", "Rat", "Mouse", "Dog"]


def test_rat_and_mouse_prediction_are_independent() -> None:
    conservation = {
        900: ConservationRecord(
            residue_position=900,
            human_aa="A",
            nhp_aa="A",
            rat_aa="A",
            mouse_aa="V",
            dog_aa="A",
            conservation_score=0.8,
            pocket_contact=True,
            allosteric_relevance="high",
            note="synthetic_mouse_only_divergent",
        )
    }

    result = compute_selectivity_for_row({"contact_residues_json": json.dumps([900])}, conservation)

    assert result.species_selectivity_score == 1.0 / 3.0
    assert result.predicted_active_in == ["Human", "NHP", "Rat", "Dog"]


def test_zero_weight_surface_contacts_do_not_affect_species_prediction() -> None:
    conservation = {
        900: ConservationRecord(900, "A", "V", "V", "V", "V", 0.0, True, "high", "human_specific_pocket"),
        901: ConservationRecord(901, "A", "A", "A", "A", "A", 1.0, False, "low", "surface_zero_weight"),
    }

    result = compute_selectivity_for_row({"contact_residues_json": json.dumps([900, 901])}, conservation)

    assert result.species_selectivity_score == 1.0
    assert result.region_weighted_contacts == {900: 10.0, 901: 0.0}
    assert result.predicted_active_in == ["Human"]


def test_noninformative_contacts_do_not_predict_human_only() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)

    for row in (
        {"contact_residues_json": json.dumps([257])},
        {"contact_residues_json": json.dumps([999])},
        {"contact_residues_json": json.dumps([])},
    ):
        result = compute_selectivity_for_row(row, conservation)
        assert result.species_selectivity_score == 0.0
        assert result.predicted_active_in == []


def test_empty_candidate_output_uses_stable_schema(tmp_path: Path) -> None:
    empty = tmp_path / "empty.parquet"
    nonempty = tmp_path / "nonempty.parquet"
    empty_output = tmp_path / "empty_species.parquet"
    nonempty_output = tmp_path / "nonempty_species.parquet"
    pl.DataFrame(
        {
            "canonical_smiles": pl.Series([], dtype=pl.String),
            "contact_residues_json": pl.Series([], dtype=pl.String),
        }
    ).write_parquet(empty)
    pl.DataFrame({"canonical_smiles": ["CCO"], "contact_residues_json": [json.dumps([190])]}).write_parquet(nonempty)

    empty_df = compute_species_selectivity_v3(empty, DEFAULT_CONSERVATION, empty_output)
    nonempty_df = compute_species_selectivity_v3(nonempty, DEFAULT_CONSERVATION, nonempty_output)

    assert empty_df.schema["species_selectivity_score"] == pl.Float64
    assert empty_df.schema["human_specific_voxels"] == pl.Int64
    assert empty_df.schema["species_selectivity_model"] == pl.String
    assert pl.concat([empty_df, nonempty_df]).height == 1


def test_candidate_parquet_without_evidence_schema_rejected(tmp_path: Path) -> None:
    candidates = tmp_path / "no_evidence.parquet"
    output = tmp_path / "no_evidence_species.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO"], "reward": [1.0]}).write_parquet(candidates)

    with pytest.raises(ValueError, match="no species-selectivity evidence columns"):
        compute_species_selectivity_v3(candidates, DEFAULT_CONSERVATION, output)


def test_output_temp_path_does_not_delete_input_with_tmp_suffix(tmp_path: Path) -> None:
    output = tmp_path / "species.parquet"
    candidates = tmp_path / "species.parquet.tmp"
    pl.DataFrame({"canonical_smiles": ["CCO"], "contact_residues_json": [json.dumps([190])]}).write_parquet(candidates)

    df = compute_species_selectivity_v3(candidates, DEFAULT_CONSERVATION, output)

    assert candidates.exists()
    assert output.exists()
    assert df.height == 1


def test_output_path_cannot_overwrite_input(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    pl.DataFrame({"canonical_smiles": ["CCO"], "contact_residues_json": [json.dumps([190])]}).write_parquet(candidates)

    with pytest.raises(ValueError, match="must not overwrite candidate parquet"):
        compute_species_selectivity_v3(candidates, DEFAULT_CONSERVATION, candidates)


def test_any_out_of_range_variant_resilience_rejected() -> None:
    with pytest.raises(ValueError, match="resilience value out of range"):
        extract_variant_contacts({"resilience_R190Q": 0.1, "resilience_T149M": 2.0})

    conservation = load_conservation(DEFAULT_CONSERVATION)
    with pytest.raises(ValueError, match="resilience value out of range"):
        compute_selectivity_for_row({"resilience_R190Q": 0.1, "resilience_T149M": 2.0}, conservation)


def test_malformed_variant_resilience_rejected() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)

    with pytest.raises(ValueError, match="invalid resilience mutation suffix"):
        compute_selectivity_for_row({"resilience_bad": 0.1}, conservation)

    with pytest.raises(ValueError, match="invalid resilience value"):
        compute_selectivity_for_row({"contact_residues_json": json.dumps([190]), "resilience_R190Q": "bad"}, conservation)

    with pytest.raises(ValueError, match="invalid resilience value"):
        compute_selectivity_for_row({"resilience_R190Q": False}, conservation)

    with pytest.raises(ValueError, match="invalid resilience mutation suffix"):
        compute_selectivity_for_row({"resilience_B190Z": 0.1}, conservation)

    with pytest.raises(ValueError, match="invalid resilience mutation suffix"):
        compute_selectivity_for_row({"resilience_R190R": 0.1}, conservation)

    with pytest.raises(ValueError, match="does not match human residue"):
        compute_selectivity_for_row({"resilience_A190Q": 0.1}, conservation)

    with pytest.raises(ValueError, match="invalid resilience mutation suffix"):
        compute_selectivity_for_row({"resilience_R0Q": 0.1}, conservation)


def test_known_variant_source_conflict_is_unscored_not_fatal() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)

    assert compute_selectivity_for_row({"resilience_R380C": 0.0}, conservation).contact_residues == {}


def test_known_variant_source_conflict_still_validates_value() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)

    for bad_value in ("bad", False, -0.1, 2.0):
        with pytest.raises(ValueError):
            compute_selectivity_for_row({"resilience_R380C": bad_value}, conservation)
        with pytest.raises(ValueError):
            compute_selectivity_for_row(
                {"contact_residues_json": json.dumps([190]), "resilience_R380C": bad_value},
                conservation,
            )


def test_super_resilience_values_are_zero_contact_not_invalid() -> None:
    assert extract_variant_contacts({"resilience_R190Q": 1.02}) == {}


def test_mixed_nonfinite_coordinate_payload_rejected() -> None:
    conservation = load_conservation(DEFAULT_CONSERVATION)

    with pytest.raises(ValueError, match="invalid JSON constant|non-finite coordinate"):
        compute_selectivity_for_row(
            {"coordinates_json": json.dumps([[155.0, 129.0, 125.0], [float("nan"), 0.0, 0.0]])},
            conservation,
            [ReceptorResidue(190, "LEU", (155.0, 129.0, 125.0))],
            contact_cutoff_a=2.0,
        )


def test_receptor_pdb_nonfinite_coordinates_rejected(tmp_path: Path) -> None:
    receptor = tmp_path / "nan_receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  LEU R 190         nan   0.000   0.000  1.00 99.76           C\n"
        "ENDMDL\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite CA coordinates"):
        load_receptor_residues(receptor)


def test_receptor_pdb_malformed_ca_record_rejected(tmp_path: Path) -> None:
    receptor = tmp_path / "bad_receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  LEU R 190       badxx   0.000   0.000  1.00 99.76           C\n"
        "ATOM      2  CA  THR R 149       0.000   0.000   0.000  1.00 99.76           C\n"
        "ENDMDL\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed CA record"):
        load_receptor_residues(receptor)


def test_compute_species_selectivity_writes_enriched_parquet(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    output = tmp_path / "species.parquet"
    pl.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCC"],
            "contact_residues_json": [json.dumps([190]), json.dumps([149])],
        }
    ).write_parquet(candidates)

    df = compute_species_selectivity_v3(candidates, DEFAULT_CONSERVATION, output)

    assert output.exists()
    assert "species_selectivity_score" in df.columns
    assert "predicted_active_in" in df.columns
    assert "species_region_weighted_contacts_json" in df.columns
    assert df["species_selectivity_model"].unique().to_list() == ["v3_region_weighted"]
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
    assert result.species_selectivity_score > 0.5
    assert result.predicted_active_in == ["Human", "NHP"]


def test_top100_candidate_range_gate(tmp_path: Path) -> None:
    candidates = Path(
        "campaigns/glp1r_aleniglipron/track_a_generative/"
        "gflownet_top100_pgx_parity_validated.parquet"
    )
    if not candidates.is_file():
        pytest.skip("D07 Top100 candidate artifact is not present")
    if not DEFAULT_RECEPTOR_PDB.is_file():
        pytest.skip(f"D07 Top100 coordinate range gate requires external receptor PDB: {DEFAULT_RECEPTOR_PDB}")

    df = compute_species_selectivity_v3(
        candidates,
        DEFAULT_CONSERVATION,
        tmp_path / "top100_species.parquet",
    )
    scores = df.get_column("species_selectivity_score")

    assert float(scores.max() - scores.min()) > 0.15
