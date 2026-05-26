import json
import re
from pathlib import Path


PANEL = Path("campaigns/glp1r_aleniglipron/track_b_chronological/genealogical_variant_panel.json")

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def test_no_variant_lacks_required_axes() -> None:
    payload = json.loads(PANEL.read_text())
    assert payload["variant_count"] > 0
    for variant in payload["variants"]:
        assert variant["topology_region"]
        assert variant["perturbation_family"]
        assert len(variant["observability_channels"]) >= 2


def test_projected_variants_are_not_marked_observed() -> None:
    payload = json.loads(PANEL.read_text())
    assert all(variant["provenance_class"] != "L5_OBSERVED" for variant in payload["variants"])


def test_variant_residue_identity_and_mutation_strings_are_physical() -> None:
    payload = json.loads(PANEL.read_text())
    for variant in payload["variants"]:
        residue_id = str(variant["residue_id"])
        match = re.match(r"^([A-Z]{3})(\d+)$", residue_id)
        assert match is not None
        assert int(match.group(2)) == int(variant["residue_position"])
        assert 1 <= int(variant["residue_position"]) <= 500
        assert variant["source_amino_acid"] == AA3_TO_1[match.group(1)]
        assert variant["source_amino_acid"] != "X"
        assert variant["source_amino_acid"] != variant["target_amino_acid"]
        assert variant["perturbation_type"] == (
            f"{variant['source_amino_acid']}{variant['residue_position']}{variant['target_amino_acid']}"
        )


def test_conservation_is_used_without_rewriting_residue_chemistry() -> None:
    payload = json.loads(PANEL.read_text())
    assert sum(1 for variant in payload["variants"] if variant["conservation_used"]) > 0
    assert all(
        variant["conservation_match_type"] in {"exact", "nearest_within_8", "none"}
        for variant in payload["variants"]
    )
