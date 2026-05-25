from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import polars as pl


DEFAULT_CONSERVATION = Path(
    "campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/source/"
    "GLP1R_cross_species_conservation.csv"
)
DEFAULT_RECEPTOR_PDB = Path(
    "campaigns/glp1r_aleniglipron/phase_2c_de_novo_capture/single_stream_representative/"
    "glp1r_6XOX_WT/glp1r_6XOX_WT.ensemble_trajectory.pdb"
)
DEFAULT_TOPOLOGY = Path("04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json")

CONTACT_COLUMNS = (
    "contact_residues_json",
    "occupied_residues_json",
    "nearest_residue_positions_json",
    "residue_positions_json",
    "lock_residue_positions_json",
)

MUTATION_RE = re.compile(r"^[A-Z](\d+)[A-Z]$")


@dataclass(frozen=True)
class ConservationRecord:
    residue_position: int
    human_aa: str
    nhp_aa: str
    rat_aa: str
    mouse_aa: str
    dog_aa: str
    conservation_score: float
    pocket_contact: bool
    allosteric_relevance: str
    note: str

    @property
    def is_human_nhp_conserved(self) -> bool:
        return self.human_aa == self.nhp_aa

    @property
    def rodent_divergent(self) -> bool:
        return self.rat_aa != self.human_aa or self.mouse_aa != self.human_aa

    @property
    def dog_divergent(self) -> bool:
        return self.dog_aa != self.human_aa

    @property
    def human_specific_weight(self) -> float:
        """0.0 = universal residue, 1.0 = human/NHP-specific contact."""
        if not self.is_human_nhp_conserved:
            return 0.25
        divergent_species = int(self.rat_aa != self.human_aa)
        divergent_species += int(self.mouse_aa != self.human_aa)
        divergent_species += int(self.dog_aa != self.human_aa)
        return min(1.0, divergent_species / 3.0)


@dataclass(frozen=True)
class SelectivityResult:
    species_selectivity_score: float
    human_specific_voxels: int
    universal_voxels: int
    predicted_active_in: list[str]
    evidence_level: str
    method: str
    contact_residues: dict[int, float]
    unscored_contact_residues: dict[int, float]
    unscored_contact_fraction: float


@dataclass(frozen=True)
class ReceptorResidue:
    residue_position: int
    residue_name: str
    xyz: tuple[float, float, float]


def load_topology_residue_id_map(path: Path | None) -> dict[int, int]:
    """Map trajectory/PDB residue indices to biological GLP-1R residue IDs."""

    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw_residues = data.get("residues")
    if not isinstance(raw_residues, list):
        return {}
    mapping: dict[int, int] = {}
    for raw in raw_residues:
        if not isinstance(raw, dict):
            continue
        if raw.get("residue_name") == "ALN":
            continue
        try:
            residue_idx = int(raw["residue_idx"])
            residue_id = int(raw["residue_id"])
        except (KeyError, TypeError, ValueError):
            continue
        mapping[residue_idx] = residue_id
    return mapping


def load_conservation(path: Path) -> dict[int, ConservationRecord]:
    records: dict[int, ConservationRecord] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pos = int(row["residue_position"])
            records[pos] = ConservationRecord(
                residue_position=pos,
                human_aa=row["human_aa"],
                nhp_aa=row["nhp_aa"],
                rat_aa=row["rat_aa"],
                mouse_aa=row["mouse_aa"],
                dog_aa=row["dog_aa"],
                conservation_score=float(row["conservation_score"]),
                pocket_contact=row.get("pocket_contact", "no").lower() == "yes",
                allosteric_relevance=row.get("allosteric_relevance", "low").lower(),
                note=row.get("note", ""),
            )
    return records


def load_receptor_residues(path: Path, topology_path: Path | None = None) -> list[ReceptorResidue]:
    """Load C-alpha coordinates from the first PDB model.

    The observatory trajectory PDB uses topology indices as residue numbers.
    When a topology JSON is provided, those indices are converted to biological
    GLP-1R residue IDs before contact scoring against the conservation CSV.
    """

    residues: list[ReceptorResidue] = []
    if not path.is_file():
        return residues
    residue_id_map = load_topology_residue_id_map(topology_path)
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                pdb_residue_position = int(line[22:26])
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            residue_position = residue_id_map.get(pdb_residue_position, pdb_residue_position)
            residues.append(
                ReceptorResidue(
                    residue_position=residue_position,
                    residue_name=line[17:20].strip(),
                    xyz=xyz,
                )
            )
    return residues


def parse_mutation_position(mutation: str) -> int | None:
    match = MUTATION_RE.match(mutation)
    if match is None:
        return None
    return int(match.group(1))


def _numbers_from_jsonish(value: object) -> list[int]:
    if value is None:
        return []
    parsed: object
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [int(part) for part in re.findall(r"\d+", value)]
    else:
        parsed = value

    numbers: list[int] = []

    def visit(obj: object) -> None:
        if isinstance(obj, bool):
            return
        if isinstance(obj, int):
            numbers.append(obj)
        elif isinstance(obj, float) and obj.is_integer():
            numbers.append(int(obj))
        elif isinstance(obj, str):
            numbers.extend(int(part) for part in re.findall(r"\d+", obj))
        elif isinstance(obj, dict):
            for item in obj.values():
                visit(item)
        elif isinstance(obj, Iterable):
            for item in obj:
                visit(item)

    visit(parsed)
    return numbers


def _coordinates_from_jsonish(value: object) -> list[tuple[float, float, float]]:
    if value is None:
        return []
    parsed: object
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    else:
        parsed = value
    coords: list[tuple[float, float, float]] = []
    if not isinstance(parsed, Iterable) or isinstance(parsed, str | bytes | dict):
        return coords
    for item in parsed:
        raw_xyz: list[object]
        if isinstance(item, dict):
            raw_xyz = [cast(object, item.get("x")), cast(object, item.get("y")), cast(object, item.get("z"))]
        elif isinstance(item, Iterable) and not isinstance(item, str | bytes):
            raw_xyz = [cast(object, value) for value in list(item)[:3]]
        else:
            continue
        if len(raw_xyz) != 3:
            continue
        try:
            x = _object_to_float(raw_xyz[0])
            y = _object_to_float(raw_xyz[1])
            z = _object_to_float(raw_xyz[2])
        except (TypeError, ValueError):
            continue
        xyz = (x, y, z)
        if all(math.isfinite(value_) for value_ in xyz):
            coords.append(xyz)
    return coords


def _object_to_float(value: object) -> float:
    if isinstance(value, bool) or value is None:
        raise TypeError("not a numeric coordinate")
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError("not a numeric coordinate")


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(
        (a[0] - b[0]) * (a[0] - b[0])
        + (a[1] - b[1]) * (a[1] - b[1])
        + (a[2] - b[2]) * (a[2] - b[2])
    )


def extract_coordinate_contacts(
    row: dict[str, Any],
    receptor_residues: Sequence[ReceptorResidue],
    *,
    contact_cutoff_a: float,
) -> dict[int, float]:
    if not receptor_residues:
        return {}
    coords = _coordinates_from_jsonish(row.get("coordinates_json"))
    if not coords:
        return {}
    contacts: dict[int, float] = {}
    for atom_xyz in coords:
        nearest: tuple[float, ReceptorResidue] | None = None
        for residue in receptor_residues:
            dist = _distance(atom_xyz, residue.xyz)
            if nearest is None or dist < nearest[0]:
                nearest = (dist, residue)
        if nearest is None or nearest[0] > contact_cutoff_a:
            continue
        residue_position = nearest[1].residue_position
        contacts[residue_position] = contacts.get(residue_position, 0.0) + (1.0 / (1.0 + nearest[0]))
    return contacts


def extract_explicit_contacts(row: dict[str, Any]) -> dict[int, float]:
    contacts: dict[int, float] = {}
    for column in CONTACT_COLUMNS:
        if column not in row:
            continue
        for residue in _numbers_from_jsonish(row.get(column)):
            contacts[residue] = contacts.get(residue, 0.0) + 1.0
    return contacts


def extract_variant_contacts(row: dict[str, Any]) -> dict[int, float]:
    contacts: dict[int, float] = {}
    for key, value in row.items():
        if not key.startswith("resilience_"):
            continue
        mutation = key.removeprefix("resilience_")
        position = parse_mutation_position(mutation)
        if position is None:
            continue
        try:
            resilience = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(resilience):
            continue
        contact_weight = max(0.0, 1.0 - resilience)
        if contact_weight > 0.0:
            contacts[position] = contacts.get(position, 0.0) + contact_weight
    return contacts


def fallback_contacts(conservation: dict[int, ConservationRecord]) -> dict[int, float]:
    """Conservative fallback when candidate rows do not carry contact provenance."""
    contacts: dict[int, float] = {}
    for pos, record in conservation.items():
        weight = 0.0
        if record.pocket_contact:
            weight += 1.0
        if record.allosteric_relevance == "critical":
            weight += 0.75
        elif record.allosteric_relevance == "high":
            weight += 0.5
        elif record.allosteric_relevance == "moderate":
            weight += 0.25
        if weight > 0.0:
            contacts[pos] = weight
    return contacts


def infer_contact_residues(
    row: dict[str, Any],
    conservation: dict[int, ConservationRecord],
    receptor_residues: Sequence[ReceptorResidue] = (),
    *,
    contact_cutoff_a: float = 8.0,
) -> tuple[dict[int, float], str, str]:
    coordinate_rows_present = bool(_coordinates_from_jsonish(row.get("coordinates_json")))
    coordinate_contacts = extract_coordinate_contacts(
        row,
        receptor_residues,
        contact_cutoff_a=contact_cutoff_a,
    )
    if coordinate_contacts:
        return coordinate_contacts, "atom_coordinate_nearest_residue_mapping", "L2"
    if coordinate_rows_present:
        return {}, "atom_coordinate_no_residue_within_cutoff", "L1"

    explicit = extract_explicit_contacts(row)
    if explicit:
        return explicit, "explicit_contact_residue_columns", "L2"

    variant_contacts = extract_variant_contacts(row)
    if variant_contacts:
        return variant_contacts, "variant_resilience_sensitivity", "L2"

    return fallback_contacts(conservation), "conservation_fallback_no_candidate_contacts", "L1"


def predict_active_species(score: float, contact_records: Sequence[ConservationRecord]) -> list[str]:
    if score >= 0.65:
        return ["Human", "NHP"]
    if score <= 0.25:
        return ["Human", "NHP", "Rat", "Mouse", "Dog"]

    active = ["Human", "NHP"]
    if any(not record.rodent_divergent for record in contact_records):
        active.extend(["Rat", "Mouse"])
    if any(not record.dog_divergent for record in contact_records):
        active.append("Dog")
    return list(dict.fromkeys(active))


def compute_selectivity_for_row(
    row: dict[str, Any],
    conservation: dict[int, ConservationRecord],
    receptor_residues: Sequence[ReceptorResidue] = (),
    *,
    contact_cutoff_a: float = 8.0,
) -> SelectivityResult:
    contacts, method, evidence = infer_contact_residues(
        row,
        conservation,
        receptor_residues,
        contact_cutoff_a=contact_cutoff_a,
    )
    total_weight = 0.0
    human_specific_weight = 0.0
    universal_weight = 0.0
    contact_records: list[ConservationRecord] = []
    unscored_contacts: dict[int, float] = {}
    unscored_weight = 0.0

    for residue, contact_weight in contacts.items():
        record = conservation.get(residue)
        if record is None:
            weight = max(float(contact_weight), 0.0)
            if weight > 0.0:
                unscored_contacts[residue] = weight
                unscored_weight += weight
            continue
        contact_records.append(record)
        weight = max(float(contact_weight), 0.0)
        total_weight += weight
        residue_specific = record.human_specific_weight * weight
        human_specific_weight += residue_specific
        universal_weight += (1.0 - record.human_specific_weight) * weight

    if total_weight <= 0.0:
        score = 0.0
    else:
        score = max(0.0, min(1.0, human_specific_weight / total_weight))
    all_contact_weight = total_weight + unscored_weight
    unscored_fraction = unscored_weight / all_contact_weight if all_contact_weight > 0.0 else 0.0

    return SelectivityResult(
        species_selectivity_score=score,
        human_specific_voxels=int(round(human_specific_weight)),
        universal_voxels=int(round(universal_weight)),
        predicted_active_in=predict_active_species(score, contact_records),
        evidence_level=evidence,
        method=method,
        contact_residues=contacts,
        unscored_contact_residues=unscored_contacts,
        unscored_contact_fraction=unscored_fraction,
    )


def compute_species_selectivity(
    candidates_path: Path,
    conservation_path: Path,
    output_path: Path,
    receptor_pdb_path: Path = DEFAULT_RECEPTOR_PDB,
    topology_path: Path | None = DEFAULT_TOPOLOGY,
    contact_cutoff_a: float = 8.0,
) -> pl.DataFrame:
    conservation = load_conservation(conservation_path)
    receptor_residues = load_receptor_residues(receptor_pdb_path, topology_path)
    df = pl.read_parquet(candidates_path)
    rows = df.to_dicts()
    results = [
        compute_selectivity_for_row(
            row,
            conservation,
            receptor_residues,
            contact_cutoff_a=contact_cutoff_a,
        )
        for row in rows
    ]

    enriched = df.with_columns(
        pl.Series("species_selectivity_score", [r.species_selectivity_score for r in results]),
        pl.Series("human_specific_voxels", [r.human_specific_voxels for r in results]),
        pl.Series("universal_voxels", [r.universal_voxels for r in results]),
        pl.Series("predicted_active_in", [json.dumps(r.predicted_active_in) for r in results]),
        pl.Series("species_selectivity_evidence_level", [r.evidence_level for r in results]),
        pl.Series("species_selectivity_method", [r.method for r in results]),
        pl.Series(
            "species_contact_residues_json",
            [json.dumps(r.contact_residues, sort_keys=True) for r in results],
        ),
        pl.Series(
            "species_unscored_contact_residues_json",
            [json.dumps(r.unscored_contact_residues, sort_keys=True) for r in results],
        ),
        pl.Series("species_unscored_contact_fraction", [r.unscored_contact_fraction for r in results]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    enriched.write_parquet(tmp_path)
    tmp_path.replace(output_path)
    return enriched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute GLP-1R cross-species selectivity.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--conservation", type=Path, default=DEFAULT_CONSERVATION)
    parser.add_argument("--receptor-pdb", type=Path, default=DEFAULT_RECEPTOR_PDB)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--contact-cutoff-a", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidates = cast(Path, args.candidates)
    conservation = cast(Path, args.conservation)
    receptor_pdb = cast(Path, args.receptor_pdb)
    topology = cast(Path, args.topology)
    contact_cutoff_a = float(args.contact_cutoff_a)
    output = cast(Path, args.output)
    df = compute_species_selectivity(candidates, conservation, output, receptor_pdb, topology, contact_cutoff_a)
    raw_mean_score = cast(float | None, df.get_column("species_selectivity_score").mean())
    mean_score = float(raw_mean_score) if raw_mean_score is not None else 0.0
    print(
        "species_selectivity_complete "
        f"rows={len(df)} output={str(output)} mean_score={mean_score:.4f}"
    )


if __name__ == "__main__":
    main()
