from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NoReturn, Sequence, cast

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

MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
REQUIRED_CONSERVATION_COLUMNS = {
    "residue_position",
    "human_aa",
    "nhp_aa",
    "rat_aa",
    "mouse_aa",
    "dog_aa",
    "conservation_score",
    "pocket_contact",
    "allosteric_relevance",
}
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
VALID_POCKET_CONTACT_VALUES = {"yes", "no"}
VALID_ALLOSTERIC_RELEVANCE = {"low", "moderate", "high", "critical"}
SELECTIVITY_MODEL_VERSION = "v3_region_weighted"
COORDINATE_DISTANCE_DECAY_EXPONENT = 0.15
KNOWN_VARIANT_SOURCE_CONFLICTS = {
    # The committed PGx landscape contains an R380C resilience column while
    # the cross-species conservation table and gnomAD source identify residue
    # 380 as F. Treat this single upstream source conflict as unscored evidence
    # rather than letting it abort coordinate-driven Top 100 scoring.
    "R380C",
}
REGION_WEIGHTS = {
    "pocket_contact": 10.0,
    "ecd": 5.0,
    "allosteric": 3.0,
    "surface": 0.0,
}
N_TERMINAL_ECD_START = 24
N_TERMINAL_ECD_END = 144


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
    def rat_divergent(self) -> bool:
        return self.rat_aa != self.human_aa

    @property
    def mouse_divergent(self) -> bool:
        return self.mouse_aa != self.human_aa

    @property
    def dog_divergent(self) -> bool:
        return self.dog_aa != self.human_aa

    @property
    def human_specific_weight(self) -> float:
        """0.0 = universal residue, 1.0 = rat/mouse/dog-divergent human residue."""
        return min(1.0, self.divergent_species_count / 3.0)

    @property
    def divergent_species_count(self) -> int:
        """Count rat, mouse, and dog amino-acid divergence from human GLP-1R."""
        divergent_species = int(self.rat_aa != self.human_aa)
        divergent_species += int(self.mouse_aa != self.human_aa)
        divergent_species += int(self.dog_aa != self.human_aa)
        return divergent_species

    @property
    def region_class(self) -> str:
        """Classify the residue into the D07 four-region weighting scheme."""

        if self.pocket_contact:
            return "pocket_contact"
        if N_TERMINAL_ECD_START <= self.residue_position <= N_TERMINAL_ECD_END:
            return "ecd"
        if self.allosteric_relevance in {"critical", "high", "moderate"}:
            return "allosteric"
        return "surface"

    @property
    def region_weight(self) -> float:
        return REGION_WEIGHTS[self.region_class]


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
    region_weighted_contacts: dict[int, float]
    region_classes: dict[int, str]
    divergent_residue_weights: dict[int, float]
    model_version: str


@dataclass(frozen=True)
class ReceptorResidue:
    residue_position: int
    residue_name: str
    xyz: tuple[float, float, float]


def load_topology_residue_id_map(path: Path | None) -> dict[int, int]:
    """Map trajectory/PDB residue indices to biological GLP-1R residue IDs."""

    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"topology file not found: {path}")
    try:
        data = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid topology JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("topology JSON must be an object")
    raw_residues = data.get("residues")
    if not isinstance(raw_residues, list):
        raise ValueError("topology JSON missing residues list")
    mapping: dict[int, int] = {}
    residue_indices_seen: set[int] = set()
    residue_ids_seen: set[int] = set()
    for raw in raw_residues:
        if not isinstance(raw, dict):
            raise ValueError("topology residue entry must be an object")
        residue_name = raw.get("residue_name")
        if not isinstance(residue_name, str) or not residue_name.strip():
            raise ValueError(f"malformed topology residue entry: {raw!r}")
        if "residue_idx" not in raw or "residue_id" not in raw:
            raise ValueError(f"malformed topology residue entry: {raw!r}")
        residue_idx = _topology_int(raw["residue_idx"], "residue_idx", raw, minimum=0)
        residue_id = _topology_int(raw["residue_id"], "residue_id", raw, minimum=1)
        if residue_idx in residue_indices_seen:
            raise ValueError(f"duplicate topology residue_idx: {residue_idx}")
        residue_indices_seen.add(residue_idx)
        if residue_name == "ALN":
            continue
        if residue_id in residue_ids_seen:
            raise ValueError(f"duplicate topology residue_id: {residue_id}")
        mapping[residue_idx] = residue_id
        residue_ids_seen.add(residue_id)
    if not mapping:
        raise ValueError("topology JSON contains no receptor residue mappings")
    return mapping


def _topology_int(value: Any, field: str, raw: dict[str, Any], *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"malformed topology residue entry: {raw!r}")
    if value < minimum:
        raise ValueError(f"invalid topology {field}: {value}")
    return value


def load_conservation(path: Path) -> dict[int, ConservationRecord]:
    records: dict[int, ConservationRecord] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_fieldnames = reader.fieldnames or []
        if len(raw_fieldnames) != len(set(raw_fieldnames)):
            raise ValueError("conservation file contains duplicate headers")
        fieldnames = set(raw_fieldnames)
        missing = REQUIRED_CONSERVATION_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError(f"conservation file missing required columns: {sorted(missing)}")
        for row in reader:
            if None in row:
                raise ValueError("conservation file row has extra fields")
            for column in REQUIRED_CONSERVATION_COLUMNS:
                if row.get(column) is None:
                    raise ValueError(f"conservation file row missing value for required column: {column}")
            pos = int(row["residue_position"])
            if pos <= 0:
                raise ValueError(f"invalid residue_position: {pos}")
            if pos in records:
                raise ValueError(f"duplicate conservation residue_position: {pos}")
            human_aa = _validated_amino_acid(row["human_aa"], pos, "human_aa")
            nhp_aa = _validated_amino_acid(row["nhp_aa"], pos, "nhp_aa")
            rat_aa = _validated_amino_acid(row["rat_aa"], pos, "rat_aa")
            mouse_aa = _validated_amino_acid(row["mouse_aa"], pos, "mouse_aa")
            dog_aa = _validated_amino_acid(row["dog_aa"], pos, "dog_aa")
            conservation_score = float(row["conservation_score"])
            if not math.isfinite(conservation_score) or not 0.0 <= conservation_score <= 1.0:
                raise ValueError(f"invalid conservation_score for residue {pos}: {row['conservation_score']!r}")
            pocket_contact_raw = row.get("pocket_contact", "").strip().lower()
            if pocket_contact_raw not in VALID_POCKET_CONTACT_VALUES:
                raise ValueError(f"invalid pocket_contact for residue {pos}: {row.get('pocket_contact')!r}")
            allosteric_relevance = row.get("allosteric_relevance", "").strip().lower()
            if allosteric_relevance not in VALID_ALLOSTERIC_RELEVANCE:
                raise ValueError(
                    f"invalid allosteric_relevance for residue {pos}: {row.get('allosteric_relevance')!r}"
                )
            records[pos] = ConservationRecord(
                residue_position=pos,
                human_aa=human_aa,
                nhp_aa=nhp_aa,
                rat_aa=rat_aa,
                mouse_aa=mouse_aa,
                dog_aa=dog_aa,
                conservation_score=conservation_score,
                pocket_contact=pocket_contact_raw == "yes",
                allosteric_relevance=allosteric_relevance,
                note=row.get("note") or "",
            )
    if not records:
        raise ValueError("conservation file contains no records")
    return records


def _validated_amino_acid(value: str, residue_position: int, column: str) -> str:
    aa = value.strip().upper()
    if len(aa) != 1 or aa not in VALID_AMINO_ACIDS:
        raise ValueError(f"invalid {column} for residue {residue_position}: {value!r}")
    return aa


def load_receptor_residues(path: Path, topology_path: Path | None = None) -> list[ReceptorResidue]:
    """Load C-alpha coordinates from the first PDB model.

    The observatory trajectory PDB uses topology indices as residue numbers.
    When a topology JSON is provided, those indices are converted to biological
    GLP-1R residue IDs before contact scoring against the conservation CSV.
    """

    residue_id_map = load_topology_residue_id_map(topology_path)
    residues: list[ReceptorResidue] = []
    if not path.is_file():
        return residues
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
            except ValueError as exc:
                raise ValueError(f"receptor PDB contains malformed CA record: {line.rstrip()}") from exc
            if not all(math.isfinite(value_) for value_ in xyz):
                raise ValueError(f"receptor PDB contains non-finite CA coordinates at residue {pdb_residue_position}")
            if topology_path is not None and pdb_residue_position not in residue_id_map:
                raise ValueError(f"receptor PDB residue index missing from topology: {pdb_residue_position}")
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
    from_aa, position, to_aa = match.groups()
    position_int = int(position)
    if position_int <= 0 or from_aa not in VALID_AMINO_ACIDS or to_aa not in VALID_AMINO_ACIDS or from_aa == to_aa:
        return None
    return position_int


def _numbers_from_jsonish(value: object) -> list[int]:
    if value is None:
        return []
    parsed: object
    if isinstance(value, str):
        if not value.strip():
            return []
        parsed = _strict_json_loads(value)
    else:
        parsed = value

    if not isinstance(parsed, list):
        raise ValueError("contact residue JSON must be a list of positive integer residue IDs")

    numbers: list[int] = []
    for item in parsed:
        if isinstance(item, bool):
            raise ValueError("contact residue JSON contains a boolean token")
        if not isinstance(item, int):
            raise ValueError(f"contact residue JSON contains a non-integer residue id: {item!r}")
        if item <= 0:
            raise ValueError(f"contact residue JSON contains a non-positive residue id: {item!r}")
        numbers.append(item)
    return numbers


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"invalid JSON constant: {value}")


def _strict_json_loads(value: str) -> object:
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_json_object_keys,
    )


def _reject_duplicate_json_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, item in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON object key: {key}")
        parsed[key] = item
    return parsed


def _value_has_content(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Iterable):
        try:
            return any(True for _ in value)
        except TypeError:
            return True
    return True


def _coordinates_from_jsonish(value: object) -> list[tuple[float, float, float]]:
    if value is None:
        return []
    parsed: object
    if isinstance(value, str):
        if not value.strip():
            return []
        parsed = _strict_json_loads(value)
    else:
        parsed = value
    coords: list[tuple[float, float, float]] = []
    if not isinstance(parsed, list):
        raise ValueError("coordinates_json must be a list of coordinate triples")
    for item in parsed:
        raw_xyz: list[object]
        if isinstance(item, dict):
            if set(item) != {"x", "y", "z"}:
                raise ValueError("coordinates_json contains a coordinate object without exactly x/y/z keys")
            raw_xyz = [cast(object, item.get("x")), cast(object, item.get("y")), cast(object, item.get("z"))]
        elif isinstance(item, list):
            raw_xyz = [cast(object, value) for value in item]
        else:
            raise ValueError("coordinates_json contains a malformed coordinate row")
        if len(raw_xyz) != 3:
            raise ValueError("coordinates_json contains a coordinate row without exactly 3 values")
        try:
            x = _object_to_float(raw_xyz[0])
            y = _object_to_float(raw_xyz[1])
            z = _object_to_float(raw_xyz[2])
        except (TypeError, ValueError):
            raise ValueError("coordinates_json contains a non-numeric coordinate") from None
        xyz = (x, y, z)
        if not all(math.isfinite(value_) for value_ in xyz):
            raise ValueError("coordinates_json contains a non-finite coordinate")
        coords.append(xyz)
    return coords


def _object_to_float(value: object) -> float:
    if isinstance(value, bool) or value is None:
        raise TypeError("not a numeric coordinate")
    if isinstance(value, int | float):
        return float(value)
    raise TypeError("not a numeric coordinate")


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(
        (a[0] - b[0]) * (a[0] - b[0])
        + (a[1] - b[1]) * (a[1] - b[1])
        + (a[2] - b[2]) * (a[2] - b[2])
    )


def distance_decay(distance_a: float, cutoff_a: float) -> float:
    """Distance decay for coordinate-derived residue contacts."""

    if cutoff_a <= 0.0 or distance_a < 0.0 or not math.isfinite(distance_a):
        return 0.0
    if distance_a > cutoff_a:
        return 0.0
    base = 1.0 / (1.0 + distance_a)
    return math.pow(base, COORDINATE_DISTANCE_DECAY_EXPONENT)


def extract_coordinate_contacts(
    row: dict[str, Any],
    receptor_residues: Sequence[ReceptorResidue],
    *,
    contact_cutoff_a: float,
    scorable_residue_positions: set[int] | None = None,
) -> dict[int, float]:
    if not receptor_residues:
        return {}
    residues = (
        [residue for residue in receptor_residues if residue.residue_position in scorable_residue_positions]
        if scorable_residue_positions is not None
        else list(receptor_residues)
    )
    if not residues:
        return {}
    coords = _coordinates_from_jsonish(row.get("coordinates_json"))
    if not coords:
        return {}
    contacts: dict[int, float] = {}
    for residue in residues:
        nearest_distance = min((_distance(atom_xyz, residue.xyz) for atom_xyz in coords), default=math.inf)
        weight = distance_decay(nearest_distance, contact_cutoff_a)
        if weight <= 0.0:
            continue
        contacts[residue.residue_position] = max(contacts.get(residue.residue_position, 0.0), weight)
    return contacts


def extract_explicit_contacts(row: dict[str, Any]) -> dict[int, float]:
    contacts: dict[int, float] = {}
    for column in CONTACT_COLUMNS:
        if column not in row:
            continue
        for residue in _numbers_from_jsonish(row.get(column)):
            contacts[residue] = contacts.get(residue, 0.0) + 1.0
    return contacts


def extract_variant_contacts(
    row: dict[str, Any],
    conservation: dict[int, ConservationRecord] | None = None,
) -> dict[int, float]:
    contacts: dict[int, float] = {}
    for key, value in row.items():
        if not key.startswith("resilience_"):
            continue
        if value is None:
            continue
        mutation = key.removeprefix("resilience_")
        position = parse_mutation_position(mutation)
        if position is None:
            raise ValueError(f"invalid resilience mutation suffix for {key}")
        record = conservation.get(position) if conservation is not None else None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"invalid resilience value for {key}: {value!r}")
        try:
            resilience = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"invalid resilience value for {key}: {value!r}") from None
        if not math.isfinite(resilience) or resilience < 0.0 or resilience > 1.5:
            raise ValueError(f"resilience value out of range for {key}: {value!r}")
        if record is not None and mutation[0] != record.human_aa:
            if mutation in KNOWN_VARIANT_SOURCE_CONFLICTS:
                continue
            raise ValueError(
                f"resilience mutation {mutation} does not match human residue "
                f"{record.human_aa}{position}"
            )
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
    conservation: dict[int, ConservationRecord] | None,
    receptor_residues: Sequence[ReceptorResidue] = (),
    *,
    contact_cutoff_a: float = 8.0,
) -> tuple[dict[int, float], str, str]:
    conservation_map = conservation or {}
    coordinate_evidence_present = "coordinates_json" in row
    coordinate_rows = _coordinates_from_jsonish(row.get("coordinates_json"))
    coordinate_rows_present = bool(coordinate_rows)
    contact_evidence_present = any(column in row for column in CONTACT_COLUMNS)
    variant_evidence_present = any(key.startswith("resilience_") for key in row)
    if coordinate_rows_present and not receptor_residues:
        raise ValueError("coordinate evidence present but receptor residues are unavailable")
    explicit = extract_explicit_contacts(row)
    variant_contacts = extract_variant_contacts(row, conservation)
    raw_coordinate_contacts = extract_coordinate_contacts(
        row,
        receptor_residues,
        contact_cutoff_a=contact_cutoff_a,
    )
    scorable_coordinate_contacts = extract_coordinate_contacts(
        row,
        receptor_residues,
        contact_cutoff_a=contact_cutoff_a,
        scorable_residue_positions=set(conservation_map),
    )
    coordinate_contacts = dict(raw_coordinate_contacts)
    for residue, weight in scorable_coordinate_contacts.items():
        coordinate_contacts[residue] = max(coordinate_contacts.get(residue, 0.0), weight)
    if coordinate_contacts:
        merged = dict(coordinate_contacts)
        for evidence_contacts in (explicit, variant_contacts):
            for residue, weight in evidence_contacts.items():
                merged[residue] = max(merged.get(residue, 0.0), weight)
        method = "atom_coordinate_residue_distance_decay"
        if explicit:
            method += "+explicit_contact_residue_columns"
        if variant_contacts:
            method += "+variant_resilience_sensitivity"
        return merged, method, "L2"

    if explicit:
        merged = dict(explicit)
        for residue, weight in variant_contacts.items():
            merged[residue] = max(merged.get(residue, 0.0), weight)
        method = "explicit_contact_residue_columns"
        if variant_contacts:
            method += "+variant_resilience_sensitivity"
        return merged, method, "L2"
    if variant_contacts:
        return variant_contacts, "variant_resilience_sensitivity", "L2"
    if coordinate_evidence_present or contact_evidence_present or variant_evidence_present:
        return {}, "candidate_contact_evidence_empty_or_invalid", "L1"

    return {}, "candidate_contact_evidence_empty_or_invalid", "L1"


def predict_active_species(score: float, contact_records: Sequence[ConservationRecord]) -> list[str]:
    if not contact_records:
        return []
    active = ["Human"]
    if any(record.is_human_nhp_conserved for record in contact_records):
        active.append("NHP")
    if any(not record.rat_divergent for record in contact_records):
        active.append("Rat")
    if any(not record.mouse_divergent for record in contact_records):
        active.append("Mouse")
    if any(not record.dog_divergent for record in contact_records):
        active.append("Dog")
    if score >= 0.65:
        return active[:2] if len(active) > 1 and active[1] == "NHP" else ["Human"]
    if score <= 0.25:
        return list(dict.fromkeys(active))

    return list(dict.fromkeys(active))


def compute_selectivity_for_row(
    row: dict[str, Any],
    conservation: dict[int, ConservationRecord] | None,
    receptor_residues: Sequence[ReceptorResidue] = (),
    *,
    contact_cutoff_a: float = 8.0,
) -> SelectivityResult:
    if not math.isfinite(contact_cutoff_a) or contact_cutoff_a <= 0.0:
        raise ValueError("contact_cutoff_a must be a positive finite value")
    conservation_map = conservation or {}
    contacts, method, evidence = infer_contact_residues(
        row,
        conservation_map,
        receptor_residues,
        contact_cutoff_a=contact_cutoff_a,
    )
    total_weight = 0.0
    human_specific_weight = 0.0
    universal_weight = 0.0
    contact_records: list[ConservationRecord] = []
    unscored_contacts: dict[int, float] = {}
    unscored_weight = 0.0
    region_weighted_contacts: dict[int, float] = {}
    region_classes: dict[int, str] = {}
    divergent_residue_weights: dict[int, float] = {}

    for residue, contact_weight in contacts.items():
        record = conservation_map.get(residue)
        if record is None:
            weight = max(float(contact_weight), 0.0)
            if weight > 0.0:
                unscored_contacts[residue] = weight
                unscored_weight += weight
            continue
        # Species selectivity is residue-occupancy evidence. Multiple ligand
        # atoms near the same residue should not let one conserved contact
        # numerically swamp a distinct species-divergent contact.
        raw_weight = min(max(float(contact_weight), 0.0), 1.0)
        region_classes[residue] = record.region_class
        weight = raw_weight * record.region_weight
        region_weighted_contacts[residue] = weight
        if weight <= 0.0:
            continue
        contact_records.append(record)
        total_weight += weight
        residue_specific = record.human_specific_weight * weight
        if residue_specific > 0.0:
            divergent_residue_weights[residue] = residue_specific
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
        region_weighted_contacts=region_weighted_contacts,
        region_classes=region_classes,
        divergent_residue_weights=divergent_residue_weights,
        model_version=SELECTIVITY_MODEL_VERSION,
    )


def compute_species_selectivity(
    candidates_path: Path,
    conservation_path: Path,
    output_path: Path,
    receptor_pdb_path: Path = DEFAULT_RECEPTOR_PDB,
    topology_path: Path | None = DEFAULT_TOPOLOGY,
    contact_cutoff_a: float = 8.0,
) -> pl.DataFrame:
    if not math.isfinite(contact_cutoff_a) or contact_cutoff_a <= 0.0:
        raise ValueError("contact_cutoff_a must be a positive finite value")
    if output_path.resolve() == candidates_path.resolve():
        raise ValueError("output path must not overwrite candidate parquet")
    conservation = load_conservation(conservation_path)
    receptor_residues = load_receptor_residues(receptor_pdb_path, topology_path)
    df = pl.read_parquet(candidates_path)
    if not _candidate_dataframe_has_evidence_schema(df):
        raise ValueError("candidate parquet contains no species-selectivity evidence columns")
    rows = df.to_dicts()
    if any("coordinates_json" in row and _value_has_content(row.get("coordinates_json")) for row in rows) and not receptor_residues:
        raise ValueError("coordinate evidence present but receptor residues are unavailable")
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
        pl.Series("species_selectivity_score", [r.species_selectivity_score for r in results], dtype=pl.Float64),
        pl.Series("human_specific_voxels", [r.human_specific_voxels for r in results], dtype=pl.Int64),
        pl.Series("universal_voxels", [r.universal_voxels for r in results], dtype=pl.Int64),
        pl.Series("predicted_active_in", [json.dumps(r.predicted_active_in) for r in results], dtype=pl.String),
        pl.Series("species_selectivity_evidence_level", [r.evidence_level for r in results], dtype=pl.String),
        pl.Series("species_selectivity_method", [r.method for r in results], dtype=pl.String),
        pl.Series(
            "species_contact_residues_json",
            [json.dumps(r.contact_residues, sort_keys=True) for r in results],
            dtype=pl.String,
        ),
        pl.Series(
            "species_unscored_contact_residues_json",
            [json.dumps(r.unscored_contact_residues, sort_keys=True) for r in results],
            dtype=pl.String,
        ),
        pl.Series("species_unscored_contact_fraction", [r.unscored_contact_fraction for r in results], dtype=pl.Float64),
        pl.Series(
            "species_region_weighted_contacts_json",
            [json.dumps(r.region_weighted_contacts, sort_keys=True) for r in results],
            dtype=pl.String,
        ),
        pl.Series(
            "species_region_classes_json",
            [json.dumps(r.region_classes, sort_keys=True) for r in results],
            dtype=pl.String,
        ),
        pl.Series(
            "species_divergent_residue_weights_json",
            [json.dumps(r.divergent_residue_weights, sort_keys=True) for r in results],
            dtype=pl.String,
        ),
        pl.Series("species_selectivity_model", [r.model_version for r in results], dtype=pl.String),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=output_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        if tmp_path.resolve() == candidates_path.resolve():
            raise ValueError("temporary output path collides with candidate parquet")
        enriched.write_parquet(tmp_path)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return enriched


def _candidate_dataframe_has_evidence_schema(df: pl.DataFrame) -> bool:
    columns = set(df.columns)
    if "coordinates_json" in columns:
        return True
    if any(column in columns for column in CONTACT_COLUMNS):
        return True
    return any(column.startswith("resilience_") for column in columns)


def compute_species_selectivity_v3(
    candidates_path: Path,
    conservation_path: Path,
    output_path: Path,
    receptor_pdb_path: Path = DEFAULT_RECEPTOR_PDB,
    topology_path: Path | None = DEFAULT_TOPOLOGY,
    contact_cutoff_a: float = 8.0,
) -> pl.DataFrame:
    """Compute D07 species selectivity with four-region residue weighting."""

    return compute_species_selectivity(
        candidates_path,
        conservation_path,
        output_path,
        receptor_pdb_path,
        topology_path,
        contact_cutoff_a,
    )


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
    raw_max_score = cast(float | int | None, df.get_column("species_selectivity_score").max())
    raw_min_score = cast(float | int | None, df.get_column("species_selectivity_score").min())
    max_score = float(raw_max_score) if raw_max_score is not None else 0.0
    min_score = float(raw_min_score) if raw_min_score is not None else 0.0
    score_range = max_score - min_score
    print(
        "species_selectivity_complete "
        f"rows={len(df)} output={str(output)} mean_score={mean_score:.4f} "
        f"score_range={score_range:.4f} model={SELECTIVITY_MODEL_VERSION}"
    )


if __name__ == "__main__":
    main()
