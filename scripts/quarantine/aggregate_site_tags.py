#!/usr/bin/env python3
"""
Aggregate all present tags / labels / values across the per-site spike-event
files and the topology / KCC / ensemble files produced by a PRISM-4D run,
for a requested set of site IDs.

Usage
-----
  python3 scripts/quarantine/aggregate_site_tags.py \
      --output-dir output/4lpk_phase2.1_audit_verify \
      --prefix 4lpk_clean \
      --sites 0,1,2 \
      [--json-out report.json] \
      [--top-k 20] \
      [--reservoir 10000]

Design notes (non-naive)
------------------------
* The per-site `*.site<N>.spike_events.json` files are multi-GB.  They are
  parsed with `ijson` in streaming mode so memory usage stays O(1) w.r.t.
  spike count.  Do NOT attempt `json.load` on those files.
* Every discovered field is classified on the fly:
    - string values  -> categorical tag aggregator (Counter)
    - bool values    -> boolean aggregator
    - int / float    -> numeric aggregator (Welford online stats + reservoir
                         sample for quantiles); if cardinality stays below a
                         threshold it is additionally reported as a
                         small-cardinality categorical view.
    - list values    -> recurse into elements using the same rules
    - dict values    -> recurse with dotted-path field names
    - null values    -> missing counter
* No field is hard-coded.  Schema drift is reported, not hidden.
* For the six non-spike files, site filtering is applied where the file
  has a per-site array (`prism_therm.pockets[pocket_id]`,
  `kcc_visualization.sites[id]`, `kcc_validation.sites[site_id]`).  Files
  without a per-site dimension (`asc_consensus`, `gcpid_synergy`,
  `ensemble_trajectory`) are aggregated whole-file with a clear
  "site_filter: N/A" note.
* Output is structured and machine-readable (JSON) as well as human-readable
  (text).  No pretty-printing tricks that lose precision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import ijson  # streaming JSON parser
except ImportError:
    sys.stderr.write(
        "FATAL: ijson is required for streaming the multi-GB spike files.\n"
        "Install with:  pip install ijson\n"
    )
    sys.exit(2)


# ----------------------------------------------------------------------------
# Field aggregators
# ----------------------------------------------------------------------------

# A single int that looks "categorical" (e.g. stream_id, aromatic_residue_id)
# gets a Counter up to this cardinality.  Beyond that, only numeric stats.
CATEGORICAL_INT_MAX_CARDINALITY = 64

# Reservoir size for float/int quantile sampling.  10k gives sub-percentile
# accuracy on p50 / p90 / p99 for populations up to ~1e9 rows.
DEFAULT_RESERVOIR = 10_000


@dataclass
class NumericAggregator:
    """Welford online stats + reservoir sample for quantiles."""

    count: int = 0
    min_v: float = math.inf
    max_v: float = -math.inf
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations (Welford)
    reservoir: List[float] = field(default_factory=list)
    reservoir_capacity: int = DEFAULT_RESERVOIR
    rng: random.Random = field(default_factory=lambda: random.Random(0xC0DE))
    categorical_peek: Optional[Counter] = None  # for small-cardinality ints
    categorical_aborted: bool = False

    def add(self, v: float) -> None:
        self.count += 1
        if v < self.min_v:
            self.min_v = v
        if v > self.max_v:
            self.max_v = v
        delta = v - self.mean
        self.mean += delta / self.count
        delta2 = v - self.mean
        self.m2 += delta * delta2

        # Reservoir sampling for quantile estimation
        if len(self.reservoir) < self.reservoir_capacity:
            self.reservoir.append(v)
        else:
            j = self.rng.randint(0, self.count - 1)
            if j < self.reservoir_capacity:
                self.reservoir[j] = v

        # Small-cardinality integer peek (only if v is integer-valued)
        if not self.categorical_aborted:
            if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
                if self.categorical_peek is None:
                    self.categorical_peek = Counter()
                self.categorical_peek[int(v)] += 1
                if len(self.categorical_peek) > CATEGORICAL_INT_MAX_CARDINALITY:
                    self.categorical_peek = None
                    self.categorical_aborted = True
            else:
                self.categorical_peek = None
                self.categorical_aborted = True

    @property
    def stddev(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def quantiles(self, qs: Iterable[float]) -> Dict[str, float]:
        if not self.reservoir:
            return {f"p{int(q * 100)}": float("nan") for q in qs}
        sorted_r = sorted(self.reservoir)
        n = len(sorted_r)
        out = {}
        for q in qs:
            # nearest-rank method
            idx = max(0, min(n - 1, int(round(q * (n - 1)))))
            out[f"p{int(q * 100)}"] = sorted_r[idx]
        return out

    def to_dict(self, top_k: int) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "kind": "numeric",
            "count": self.count,
            "min": None if self.count == 0 else self.min_v,
            "max": None if self.count == 0 else self.max_v,
            "mean": None if self.count == 0 else self.mean,
            "stddev": self.stddev,
        }
        if self.count:
            d.update(self.quantiles([0.01, 0.50, 0.90, 0.99]))
        if self.categorical_peek is not None:
            d["small_cardinality_view"] = {
                "unique": len(self.categorical_peek),
                "top": [
                    {"value": int(v), "count": c}
                    for v, c in self.categorical_peek.most_common(top_k)
                ],
            }
        return d


@dataclass
class CategoricalAggregator:
    """Counter over hashable (string / bool) values."""

    counter: Counter = field(default_factory=Counter)
    total: int = 0

    def add(self, v: Any) -> None:
        self.counter[v] += 1
        self.total += 1

    def to_dict(self, top_k: int) -> Dict[str, Any]:
        return {
            "kind": "categorical",
            "total": self.total,
            "unique": len(self.counter),
            "top": [
                {"value": v, "count": c}
                for v, c in self.counter.most_common(top_k)
            ],
            "truncated": len(self.counter) > top_k,
        }


@dataclass
class BooleanAggregator:
    counter: Counter = field(default_factory=lambda: Counter({True: 0, False: 0}))
    total: int = 0

    def add(self, v: bool) -> None:
        self.counter[bool(v)] += 1
        self.total += 1

    def to_dict(self, top_k: int = 0) -> Dict[str, Any]:
        return {
            "kind": "boolean",
            "total": self.total,
            "true": self.counter[True],
            "false": self.counter[False],
        }


@dataclass
class NullAggregator:
    count: int = 0

    def add(self, _v: None = None) -> None:
        self.count += 1

    def to_dict(self, top_k: int = 0) -> Dict[str, Any]:
        return {"kind": "null", "count": self.count}


@dataclass
class WeightedNumericAggregator:
    """
    Support-weighted numeric aggregator for the --project-via mode.

    Tracks both the unweighted distribution (count / min / max / mean / stddev /
    reservoir quantiles) AND the support-weighted view:

        weighted_mean = sum(value_i * w_i) / sum(w_i)

    where `w_i` is the "amount of spikes" / "total consensus metric" carried
    by each row (n_samples / n_groups / active_causal_steps / n_causal_spikes).

    Also records `support_sum = sum(w_i)` so downstream consumers can see
    the absolute weight contributed by the matched residues for this site.
    """

    count: int = 0
    min_v: float = math.inf
    max_v: float = -math.inf
    mean: float = 0.0
    m2: float = 0.0  # Welford
    support_sum: float = 0.0
    weighted_numerator: float = 0.0  # sum(value_i * w_i)
    reservoir: List[float] = field(default_factory=list)
    reservoir_capacity: int = DEFAULT_RESERVOIR
    rng: random.Random = field(default_factory=lambda: random.Random(0xC0DE))

    def add(self, value: float, weight: float) -> None:
        self.count += 1
        v = float(value)
        w = float(weight) if weight is not None else 0.0
        if v < self.min_v:
            self.min_v = v
        if v > self.max_v:
            self.max_v = v
        delta = v - self.mean
        self.mean += delta / self.count
        delta2 = v - self.mean
        self.m2 += delta * delta2
        self.support_sum += w
        self.weighted_numerator += v * w
        if len(self.reservoir) < self.reservoir_capacity:
            self.reservoir.append(v)
        else:
            j = self.rng.randint(0, self.count - 1)
            if j < self.reservoir_capacity:
                self.reservoir[j] = v

    @property
    def stddev(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    @property
    def weighted_mean(self) -> Optional[float]:
        if self.support_sum <= 0:
            return None
        return self.weighted_numerator / self.support_sum

    def quantiles(self, qs: Iterable[float]) -> Dict[str, float]:
        if not self.reservoir:
            return {f"p{int(q * 100)}": float("nan") for q in qs}
        s = sorted(self.reservoir)
        n = len(s)
        return {
            f"p{int(q * 100)}": s[max(0, min(n - 1, int(round(q * (n - 1)))))]
            for q in qs
        }

    def to_dict(self, top_k: int = 0) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "kind": "weighted_numeric",
            "count": self.count,
            "min": None if self.count == 0 else self.min_v,
            "max": None if self.count == 0 else self.max_v,
            "unweighted_mean": None if self.count == 0 else self.mean,
            "stddev": self.stddev,
            "weighted_mean": self.weighted_mean,
            "support_sum": self.support_sum,
        }
        if self.count:
            d.update(self.quantiles([0.01, 0.50, 0.90, 0.99]))
        return d


@dataclass
class WeightedCategoricalAggregator:
    """
    Categorical aggregator that also tracks the total support per category.

    For `role` in prism_therm.top_residues: counts how many residues carry
    each role label AND the sum of `n_causal_spikes` across residues with
    that label.
    """

    counts: Counter = field(default_factory=Counter)
    support: Dict[Any, float] = field(default_factory=dict)
    total_count: int = 0
    total_support: float = 0.0

    def add(self, value: Any, weight: float) -> None:
        self.counts[value] += 1
        self.support[value] = self.support.get(value, 0.0) + float(weight or 0.0)
        self.total_count += 1
        self.total_support += float(weight or 0.0)

    def to_dict(self, top_k: int = 20) -> Dict[str, Any]:
        return {
            "kind": "weighted_categorical",
            "total_count": self.total_count,
            "total_support": self.total_support,
            "unique": len(self.counts),
            "top": [
                {
                    "value": v,
                    "count": c,
                    "support_sum": self.support.get(v, 0.0),
                }
                for v, c in self.counts.most_common(top_k)
            ],
            "truncated": len(self.counts) > top_k,
        }


@dataclass
class MixedAggregator:
    """Catches values that have more than one primitive type at the same path."""

    sub_agg: Dict[str, Any] = field(default_factory=dict)
    total: int = 0

    def dispatch(self, v: Any) -> None:
        self.total += 1
        if v is None:
            key = "null"
            if key not in self.sub_agg:
                self.sub_agg[key] = NullAggregator()
            self.sub_agg[key].add()
        elif isinstance(v, bool):
            key = "bool"
            if key not in self.sub_agg:
                self.sub_agg[key] = BooleanAggregator()
            self.sub_agg[key].add(v)
        elif isinstance(v, str):
            key = "str"
            if key not in self.sub_agg:
                self.sub_agg[key] = CategoricalAggregator()
            self.sub_agg[key].add(v)
        elif isinstance(v, (int, float)):
            key = "num"
            if key not in self.sub_agg:
                self.sub_agg[key] = NumericAggregator()
            self.sub_agg[key].add(float(v))
        elif isinstance(v, (list, tuple)):
            key = "list_lengths"
            if key not in self.sub_agg:
                self.sub_agg[key] = NumericAggregator()
            self.sub_agg[key].add(float(len(v)))
        elif isinstance(v, dict):
            key = "dict_keys_count"
            if key not in self.sub_agg:
                self.sub_agg[key] = NumericAggregator()
            self.sub_agg[key].add(float(len(v)))
        else:
            key = f"unknown_type:{type(v).__name__}"
            if key not in self.sub_agg:
                self.sub_agg[key] = NullAggregator()
            self.sub_agg[key].add()

    def to_dict(self, top_k: int) -> Dict[str, Any]:
        return {
            "kind": "mixed",
            "total": self.total,
            "subtypes": {k: v.to_dict(top_k) for k, v in self.sub_agg.items()},
        }


# ----------------------------------------------------------------------------
# Aggregation driver
# ----------------------------------------------------------------------------

class FieldRegistry:
    """Stores one aggregator per dotted path; dispatches by runtime value type."""

    def __init__(self) -> None:
        self.fields: Dict[str, MixedAggregator] = {}

    def add(self, path: str, value: Any) -> None:
        if path not in self.fields:
            self.fields[path] = MixedAggregator()
        self.fields[path].dispatch(value)

    def to_dict(self, top_k: int) -> Dict[str, Any]:
        return {p: a.to_dict(top_k) for p, a in sorted(self.fields.items())}


def walk_and_aggregate(obj: Any, registry: FieldRegistry, path: str = "") -> None:
    """Recurse into dicts and lists, aggregating every leaf encountered."""

    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{path}.{k}" if path else k
            walk_and_aggregate(v, registry, child)
    elif isinstance(obj, list):
        # Record list length at this path, then recurse into each element
        registry.add(f"{path}[].length_per_parent", len(obj))
        for elem in obj:
            walk_and_aggregate(elem, registry, f"{path}[]")
    else:
        registry.add(path, obj)


# ----------------------------------------------------------------------------
# Streaming spike file aggregation (ijson)
# ----------------------------------------------------------------------------

def aggregate_spike_file(path: Path, site_id: int, reservoir: int) -> Dict[str, Any]:
    """Stream the top-level fields and every element of `spikes[]`."""

    registry = FieldRegistry()
    header: Dict[str, Any] = {}
    spike_count = 0

    # First pass: grab the non-`spikes` top-level fields.  They are small and
    # we want them reported even though we stream the rest.
    with path.open("rb") as fh:
        # `kvitems` yields (key, value) pairs at the top-level object.
        # For the `spikes` key we deliberately just record its length via
        # the streaming `item` pass below rather than materializing it here.
        for key, value in ijson.kvitems(fh, ""):
            if key == "spikes":
                continue  # handled in streaming pass
            header[key] = value

    # Second pass: stream each element of the spikes array.
    with path.open("rb") as fh:
        for spike in ijson.items(fh, "spikes.item"):
            spike_count += 1
            walk_and_aggregate(spike, registry, path="spike")

    return {
        "path": str(path),
        "site_id": site_id,
        "header": header,  # centroid, lining_cutoff, n_spikes, etc.
        "spike_count_observed": spike_count,
        "n_spikes_claimed": header.get("n_spikes"),
        "count_matches_claim": header.get("n_spikes") == spike_count,
        "fields": registry.to_dict(top_k=20),
    }


# ----------------------------------------------------------------------------
# Small-file site-filtered aggregation
# ----------------------------------------------------------------------------

# Where the per-site array lives in each file, and which key carries the site id.
SITE_FILTERS: Dict[str, Tuple[str, str]] = {
    "topology.prism_therm.json":    ("pockets", "pocket_id"),
    "kcc_visualization.json":       ("sites",   "id"),
    "kcc_validation.json":          ("sites",   "site_id"),
}

# Files that have no per-site dimension — aggregated whole-file, site filter N/A.
NO_SITE_DIM = {
    "topology.asc_consensus.json",
    "topology.gcpid_synergy.json",
    "ensemble_trajectory.json",
}


# ----------------------------------------------------------------------------
# --project-via configuration
# ----------------------------------------------------------------------------
#
# Three site→residue attribution maps already present in the run artifacts.
# Each map entry tells us (a) which file carries the map, (b) the name of the
# top-level list that contains per-site entries, (c) the key under which that
# entry stores the site id, (d) the nested list of residue records, and
# (e) the key inside each residue record that holds the residue id.
#
# The keys differ across files on purpose — the engine historically used
# different conventions in different modules.  We normalize to a Python set
# of integer residue ids per site.

SITE_RESIDUE_MAP_SPECS: Dict[str, Dict[str, str]] = {
    "lining": {
        "source_suffix": "binding_sites.json",
        "sites_key": "sites",
        "site_id_key": "id",
        "residues_key": "lining_residues",
        "residue_id_key": "resid",
    },
    "therm_top": {
        "source_suffix": "topology.prism_therm.json",
        "sites_key": "pockets",
        "site_id_key": "pocket_id",
        "residues_key": "top_residues",
        "residue_id_key": "residue_id",
    },
    "topk_driver": {
        "source_suffix": "kcc_validation.json",
        "sites_key": "sites",
        "site_id_key": "site_id",
        "residues_key": "topk_residues",
        "residue_id_key": "residue_id",
    },
}


# Residue-level data carriers that we project through the maps above.
#
# For each source: which file, where its residue list lives, which key holds
# the residue id inside each row, and which scalar field is the "support"
# (the natural weight: spike count, sample count, consensus group count, etc.)
# that the user asked to be carried alongside every aggregated value.

RESIDUE_LEVEL_SOURCES: List[Dict[str, Any]] = [
    {
        "name": "asc_consensus.consensus_residues",
        "source_suffix": "topology.asc_consensus.json",
        "list_path": ["consensus_residues"],
        "residue_id_key": "residue_id",
        "support_field": "n_groups",
        "support_semantics": "count of consensus groups this residue appeared in",
    },
    {
        "name": "gcpid_synergy.residues",
        "source_suffix": "topology.gcpid_synergy.json",
        "list_path": ["residues"],
        "residue_id_key": "residue_id",
        "support_field": "n_samples",
        "support_semantics": "sample count used to compute the Gaussian-copula PID",
    },
    {
        "name": "kcc_visualization.residues",
        "source_suffix": "kcc_visualization.json",
        "list_path": ["residues"],
        "residue_id_key": "residue_id",
        "support_field": "active_causal_steps",
        "support_semantics": "simulation steps in which this residue saw causal signal (spike-analog)",
    },
]


# prism_therm.top_residues is *already* site-keyed (no external map needed).
# We emit it separately, once per requested site, using its own support field.
DIRECT_SITE_KEYED_SOURCE: Dict[str, Any] = {
    "name": "prism_therm.pockets[*].top_residues",
    "source_suffix": "topology.prism_therm.json",
    "sites_key": "pockets",
    "site_id_key": "pocket_id",
    "residues_key": "top_residues",
    "residue_id_key": "residue_id",
    "support_field": "n_causal_spikes",
    "support_semantics": "number of causal spikes attributed to this residue at this pocket",
}


def _find_artifact(out_dir: Path, prefix: str, suffix: str) -> Optional[Path]:
    """Locate `<out_dir>/<prefix>.<suffix>` if present."""
    p = out_dir / f"{prefix}.{suffix}"
    return p if p.is_file() else None


def _load_json(path: Path) -> Any:
    with path.open("r") as fh:
        return json.load(fh)


def _resolve_path(root: Any, path: List[str]) -> Any:
    cur = root
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur


def build_site_residue_maps(
    out_dir: Path, prefix: str, sites: List[int]
) -> Dict[str, Dict[int, List[int]]]:
    """
    Build `{map_name: {site_id: [residue_ids]}}` for all three site→residue
    maps defined in SITE_RESIDUE_MAP_SPECS.

    Missing sites, missing residue lists, and string residue ids are handled
    defensively — a missing site yields an empty list, not an exception.
    """
    out: Dict[str, Dict[int, List[int]]] = {}
    for map_name, spec in SITE_RESIDUE_MAP_SPECS.items():
        src_path = _find_artifact(out_dir, prefix, spec["source_suffix"])
        if src_path is None:
            sys.stderr.write(
                f"[warn] map '{map_name}': source {prefix}.{spec['source_suffix']} not found\n"
            )
            out[map_name] = {sid: [] for sid in sites}
            continue
        root = _load_json(src_path)
        sites_arr = root.get(spec["sites_key"], [])
        per_site: Dict[int, List[int]] = {sid: [] for sid in sites}
        for entry in sites_arr:
            if not isinstance(entry, dict):
                continue
            sid = entry.get(spec["site_id_key"])
            if sid not in per_site:
                continue
            resid_list = entry.get(spec["residues_key"], [])
            ids: List[int] = []
            for r in resid_list:
                if isinstance(r, dict):
                    v = r.get(spec["residue_id_key"])
                    if isinstance(v, int):
                        ids.append(v)
                elif isinstance(r, int):
                    ids.append(r)
            per_site[sid] = ids
        out[map_name] = per_site
    return out


def _aggregate_rows_weighted(
    rows: List[Dict[str, Any]],
    support_field: str,
    residue_id_key: str,
) -> Dict[str, Any]:
    """
    Aggregate a flat list of residue-row dicts with support-weighting.

    For every top-level scalar field in each row:
      - numeric (non-bool)  -> WeightedNumericAggregator
      - bool                -> WeightedCategoricalAggregator (true/false counts)
      - string              -> WeightedCategoricalAggregator
      - null                -> counted separately as missing

    The support weight is taken from `row[support_field]`.  Rows missing the
    support field contribute with weight=0 (they still count toward `count`
    but do not influence `support_sum` or `weighted_mean`).

    Fields that are themselves lists or dicts are recorded as mixed-subtype
    metadata (length only) so the top-level weighted view stays flat and
    meaningful.
    """
    numeric: Dict[str, WeightedNumericAggregator] = {}
    categorical: Dict[str, WeightedCategoricalAggregator] = {}
    null_counts: Counter = Counter()
    non_scalar_lens: Dict[str, WeightedNumericAggregator] = {}

    support_sum_total = 0.0
    matched = 0
    missing_support = 0

    for row in rows:
        matched += 1
        w_raw = row.get(support_field)
        if w_raw is None:
            missing_support += 1
            w = 0.0
        else:
            try:
                w = float(w_raw)
            except (TypeError, ValueError):
                missing_support += 1
                w = 0.0
        support_sum_total += w

        for k, v in row.items():
            if k == support_field or k == residue_id_key:
                continue
            if v is None:
                null_counts[k] += 1
            elif isinstance(v, bool):
                categorical.setdefault(k, WeightedCategoricalAggregator()).add(v, w)
            elif isinstance(v, (int, float)):
                numeric.setdefault(k, WeightedNumericAggregator()).add(float(v), w)
            elif isinstance(v, str):
                categorical.setdefault(k, WeightedCategoricalAggregator()).add(v, w)
            elif isinstance(v, (list, tuple)):
                non_scalar_lens.setdefault(k + ".len", WeightedNumericAggregator()).add(
                    float(len(v)), w
                )
            elif isinstance(v, dict):
                non_scalar_lens.setdefault(k + ".keys_count", WeightedNumericAggregator()).add(
                    float(len(v)), w
                )

    return {
        "matched_rows": matched,
        "missing_support": missing_support,
        "support_sum": support_sum_total,
        "numeric_fields": {k: a.to_dict() for k, a in sorted(numeric.items())},
        "categorical_fields": {k: a.to_dict() for k, a in sorted(categorical.items())},
        "null_counts_by_field": dict(null_counts),
        "non_scalar_field_summaries": {k: a.to_dict() for k, a in sorted(non_scalar_lens.items())},
    }


def aggregate_projection(
    out_dir: Path,
    prefix: str,
    sites: List[int],
    maps_to_use: List[str],
) -> Dict[str, Any]:
    """
    Full --project-via aggregation.

    Returns:
      {
        "site_residue_maps": {map_name: {site_id: [residue_ids], map_size: N}},
        "projections":       [ per (site, map, source) aggregate ],
        "direct_site_keyed": [ per site aggregate of prism_therm top_residues ],
      }
    """
    site_res_maps = build_site_residue_maps(out_dir, prefix, sites)

    # Preload each indirect source once so we don't re-read the JSON per site.
    source_rows: Dict[str, List[Dict[str, Any]]] = {}
    source_meta: Dict[str, Dict[str, Any]] = {}
    for src in RESIDUE_LEVEL_SOURCES:
        p = _find_artifact(out_dir, prefix, src["source_suffix"])
        if p is None:
            sys.stderr.write(
                f"[warn] source '{src['name']}': {prefix}.{src['source_suffix']} not found\n"
            )
            source_rows[src["name"]] = []
            source_meta[src["name"]] = {"path": None, "total_rows": 0}
            continue
        root = _load_json(p)
        rows = _resolve_path(root, src["list_path"]) or []
        if not isinstance(rows, list):
            rows = []
        source_rows[src["name"]] = [r for r in rows if isinstance(r, dict)]
        source_meta[src["name"]] = {"path": str(p), "total_rows": len(source_rows[src["name"]])}

    projections: List[Dict[str, Any]] = []

    for map_name in maps_to_use:
        per_site_ids = site_res_maps.get(map_name, {})
        for sid in sites:
            ids_for_site = per_site_ids.get(sid, [])
            id_set = set(ids_for_site)
            for src in RESIDUE_LEVEL_SOURCES:
                all_rows = source_rows.get(src["name"], [])
                matched = [r for r in all_rows if r.get(src["residue_id_key"]) in id_set]
                agg = _aggregate_rows_weighted(
                    matched,
                    support_field=src["support_field"],
                    residue_id_key=src["residue_id_key"],
                )
                projections.append({
                    "site_id": sid,
                    "map": map_name,
                    "source": src["name"],
                    "source_path": source_meta[src["name"]]["path"],
                    "source_total_rows": source_meta[src["name"]]["total_rows"],
                    "site_map_size": len(ids_for_site),
                    "support_field": src["support_field"],
                    "support_semantics": src["support_semantics"],
                    **agg,
                    "match_rate_of_source": (
                        agg["matched_rows"] / source_meta[src["name"]]["total_rows"]
                        if source_meta[src["name"]]["total_rows"]
                        else 0.0
                    ),
                    "match_rate_of_map": (
                        agg["matched_rows"] / len(ids_for_site)
                        if ids_for_site
                        else 0.0
                    ),
                })

    # Direct site-keyed source: prism_therm.pockets[pocket_id].top_residues[]
    direct: List[Dict[str, Any]] = []
    pt_spec = DIRECT_SITE_KEYED_SOURCE
    pt_path = _find_artifact(out_dir, prefix, pt_spec["source_suffix"])
    if pt_path is not None:
        pt_root = _load_json(pt_path)
        pockets = pt_root.get(pt_spec["sites_key"], [])
        by_id: Dict[int, List[Dict[str, Any]]] = {}
        for p in pockets:
            if not isinstance(p, dict):
                continue
            pid = p.get(pt_spec["site_id_key"])
            tr = p.get(pt_spec["residues_key"], []) or []
            by_id[pid] = [r for r in tr if isinstance(r, dict)]
        for sid in sites:
            rows = by_id.get(sid, [])
            agg = _aggregate_rows_weighted(
                rows,
                support_field=pt_spec["support_field"],
                residue_id_key=pt_spec["residue_id_key"],
            )
            direct.append({
                "site_id": sid,
                "source": pt_spec["name"],
                "source_path": str(pt_path),
                "support_field": pt_spec["support_field"],
                "support_semantics": pt_spec["support_semantics"],
                "residues_in_pocket_top": len(rows),
                **agg,
            })
    else:
        sys.stderr.write(
            f"[warn] direct source: {prefix}.{pt_spec['source_suffix']} not found\n"
        )

    return {
        "site_residue_maps": {
            map_name: {
                "per_site": {
                    str(sid): {
                        "residue_ids": ids,
                        "size": len(ids),
                    }
                    for sid, ids in per_site_ids.items()
                }
            }
            for map_name, per_site_ids in site_res_maps.items()
        },
        "projections": projections,
        "direct_site_keyed": direct,
    }


def _file_category(fname: str) -> Tuple[str, Optional[Tuple[str, str]]]:
    """Return (category, site_filter_spec_or_None) for the given filename suffix."""
    for suffix, spec in SITE_FILTERS.items():
        if fname.endswith(suffix):
            return ("site_filtered", spec)
    for suffix in NO_SITE_DIM:
        if fname.endswith(suffix):
            return ("no_site_dim", None)
    return ("unknown", None)


def aggregate_small_file(path: Path, sites: List[int]) -> Dict[str, Any]:
    """Whole-file walk, with per-site filtering where the schema admits it."""
    with path.open("r") as fh:
        root = json.load(fh)

    category, spec = _file_category(path.name)

    out: Dict[str, Any] = {
        "path": str(path),
        "category": category,
    }

    if category == "site_filtered":
        assert spec is not None
        arr_key, id_key = spec
        arr = root.get(arr_key)
        if not isinstance(arr, list):
            out["error"] = f"expected list at '{arr_key}', found {type(arr).__name__}"
            return out
        out["site_filter"] = {"array_key": arr_key, "id_key": id_key, "requested": sites}
        out["per_site"] = {}
        for sid in sites:
            matches = [e for e in arr if isinstance(e, dict) and e.get(id_key) == sid]
            reg = FieldRegistry()
            for m in matches:
                walk_and_aggregate(m, reg, path=arr_key)
            out["per_site"][str(sid)] = {
                "match_count": len(matches),
                "fields": reg.to_dict(top_k=20),
            }
        # Also aggregate non-site-keyed global sections so the output is complete.
        global_reg = FieldRegistry()
        for k, v in root.items():
            if k == arr_key:
                continue  # handled per-site above
            walk_and_aggregate(v, global_reg, path=k)
        out["global_fields"] = global_reg.to_dict(top_k=20)

    elif category == "no_site_dim":
        out["site_filter"] = "N/A — file has no per-site array dimension"
        reg = FieldRegistry()
        walk_and_aggregate(root, reg, path="")
        out["fields"] = reg.to_dict(top_k=20)

    else:
        out["site_filter"] = "UNKNOWN — file not recognized by this script"
        reg = FieldRegistry()
        walk_and_aggregate(root, reg, path="")
        out["fields"] = reg.to_dict(top_k=20)

    return out


# ----------------------------------------------------------------------------
# Human-readable renderer
# ----------------------------------------------------------------------------

def _fmt_num(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if abs(v) >= 1e6 or (abs(v) < 1e-3 and v != 0.0):
            return f"{v:.4e}"
        return f"{v:.6g}"
    return str(v)


def render_field(name: str, agg: Dict[str, Any], indent: str = "  ") -> List[str]:
    lines: List[str] = []
    kind = agg.get("kind")
    if kind == "mixed":
        lines.append(f"{indent}{name}: MIXED types (total={agg['total']})")
        for subtype, sub in agg["subtypes"].items():
            lines.extend(render_field(f"<{subtype}>", sub, indent + "  "))
    elif kind == "numeric":
        lines.append(
            f"{indent}{name}: NUMERIC n={agg['count']} "
            f"min={_fmt_num(agg['min'])} max={_fmt_num(agg['max'])} "
            f"mean={_fmt_num(agg['mean'])} sd={_fmt_num(agg['stddev'])} "
            f"p50={_fmt_num(agg.get('p50'))} p90={_fmt_num(agg.get('p90'))} "
            f"p99={_fmt_num(agg.get('p99'))}"
        )
        scv = agg.get("small_cardinality_view")
        if scv:
            top = ", ".join(f"{e['value']}×{e['count']}" for e in scv["top"])
            lines.append(f"{indent}  (small-cardinality int, {scv['unique']} unique): {top}")
    elif kind == "categorical":
        top = ", ".join(f"{e['value']!r}×{e['count']}" for e in agg["top"])
        tail = " [truncated]" if agg.get("truncated") else ""
        lines.append(
            f"{indent}{name}: CATEGORICAL total={agg['total']} unique={agg['unique']}: {top}{tail}"
        )
    elif kind == "boolean":
        lines.append(
            f"{indent}{name}: BOOLEAN total={agg['total']} true={agg['true']} false={agg['false']}"
        )
    elif kind == "null":
        lines.append(f"{indent}{name}: NULL count={agg['count']}")
    else:
        lines.append(f"{indent}{name}: {agg}")
    return lines


def render_file(file_summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"╭─ {file_summary.get('path')}")

    if "error" in file_summary:
        lines.append(f"│  ERROR: {file_summary['error']}")
        lines.append("╰─")
        return "\n".join(lines)

    category = file_summary.get("category")
    if category is None:
        # spike file
        lines.append(f"│  site_id: {file_summary['site_id']}")
        lines.append(
            f"│  spikes observed: {file_summary['spike_count_observed']:,}  "
            f"(claimed: {file_summary['n_spikes_claimed']!r}  "
            f"match: {file_summary['count_matches_claim']})"
        )
        lines.append(f"│  header: {file_summary['header']}")
        lines.append("│  fields:")
        for fname, agg in file_summary["fields"].items():
            for ln in render_field(fname, agg, indent="│    "):
                lines.append(ln)

    elif category == "site_filtered":
        lines.append(f"│  site_filter: {file_summary['site_filter']}")
        for sid, block in file_summary["per_site"].items():
            lines.append(f"│  ── site {sid} (matches={block['match_count']}) ──")
            for fname, agg in block["fields"].items():
                for ln in render_field(fname, agg, indent="│    "):
                    lines.append(ln)
        lines.append("│  ── global (non-site-keyed) fields ──")
        for fname, agg in file_summary["global_fields"].items():
            for ln in render_field(fname, agg, indent="│    "):
                lines.append(ln)

    elif category == "no_site_dim":
        lines.append(f"│  site_filter: {file_summary['site_filter']}")
        for fname, agg in file_summary["fields"].items():
            for ln in render_field(fname, agg, indent="│  "):
                lines.append(ln)

    else:
        lines.append(f"│  site_filter: {file_summary.get('site_filter', '?')}")
        for fname, agg in file_summary.get("fields", {}).items():
            for ln in render_field(fname, agg, indent="│  "):
                lines.append(ln)

    lines.append("╰─")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Projection renderer
# ----------------------------------------------------------------------------

def _fmt_weighted_numeric(name: str, agg: Dict[str, Any], indent: str) -> List[str]:
    return [
        f"{indent}{name}: n={agg['count']} "
        f"min={_fmt_num(agg['min'])} max={_fmt_num(agg['max'])} "
        f"unweighted_mean={_fmt_num(agg['unweighted_mean'])} "
        f"weighted_mean={_fmt_num(agg['weighted_mean'])} "
        f"sd={_fmt_num(agg['stddev'])} "
        f"p50={_fmt_num(agg.get('p50'))} p90={_fmt_num(agg.get('p90'))} "
        f"p99={_fmt_num(agg.get('p99'))} "
        f"support_sum={_fmt_num(agg['support_sum'])}"
    ]


def _fmt_weighted_categorical(name: str, agg: Dict[str, Any], indent: str) -> List[str]:
    top_items = ", ".join(
        f"{e['value']!r}×{e['count']} (support={_fmt_num(e['support_sum'])})"
        for e in agg["top"]
    )
    tail = " [truncated]" if agg.get("truncated") else ""
    return [
        f"{indent}{name}: CATEG total_count={agg['total_count']} "
        f"total_support={_fmt_num(agg['total_support'])} unique={agg['unique']}",
        f"{indent}  {top_items}{tail}",
    ]


def render_projection_block(block: Dict[str, Any], header: str) -> List[str]:
    lines: List[str] = [header]
    lines.append(
        f"    source:             {block['source']}"
    )
    if "source_path" in block and block["source_path"]:
        lines.append(f"    source_path:        {block['source_path']}")
    lines.append(f"    support_field:      {block['support_field']}")
    lines.append(f"    support_semantics:  {block['support_semantics']}")
    if "site_map_size" in block:
        lines.append(
            f"    map size for site:  {block['site_map_size']} residue ids"
        )
    if "residues_in_pocket_top" in block:
        lines.append(
            f"    residues in pocket top_residues:  {block['residues_in_pocket_top']}"
        )
    if "source_total_rows" in block:
        lines.append(
            f"    source total rows:  {block['source_total_rows']}"
        )
    lines.append(
        f"    matched rows:       {block['matched_rows']}  "
        f"(match_rate_of_source={_fmt_num(block.get('match_rate_of_source'))}, "
        f"match_rate_of_map={_fmt_num(block.get('match_rate_of_map'))})"
    )
    lines.append(
        f"    missing support:    {block['missing_support']} rows  "
        f"(support_sum for matched: {_fmt_num(block['support_sum'])})"
    )

    num_fields = block.get("numeric_fields", {})
    cat_fields = block.get("categorical_fields", {})
    null_fields = block.get("null_counts_by_field", {})
    nonscalar = block.get("non_scalar_field_summaries", {})

    if num_fields:
        lines.append("    numeric fields (unweighted + support-weighted):")
        for fname in sorted(num_fields):
            lines.extend(_fmt_weighted_numeric(fname, num_fields[fname], "      "))
    if cat_fields:
        lines.append("    categorical fields (counts + support sum per label):")
        for fname in sorted(cat_fields):
            lines.extend(_fmt_weighted_categorical(fname, cat_fields[fname], "      "))
    if nonscalar:
        lines.append("    non-scalar field lengths:")
        for fname in sorted(nonscalar):
            lines.extend(_fmt_weighted_numeric(fname, nonscalar[fname], "      "))
    if null_fields:
        lines.append("    null counts per field: " + ", ".join(
            f"{k}={v}" for k, v in sorted(null_fields.items())
        ))
    if not (num_fields or cat_fields or nonscalar or null_fields):
        lines.append("    (no matched rows — nothing to aggregate)")
    return lines


def render_projection(proj: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("  SITE-PROJECTED RESIDUE-LEVEL AGGREGATION")
    lines.append("=" * 78)

    # Map sizes summary
    lines.append("")
    lines.append("  Site → residue maps (source of projection):")
    for map_name, body in proj["site_residue_maps"].items():
        lines.append(f"    [{map_name}]")
        for sid_s, info in body["per_site"].items():
            lines.append(
                f"      site {sid_s}: size={info['size']}  residue_ids={info['residue_ids'][:30]}"
                + ("..." if info["size"] > 30 else "")
            )

    # Grouped: (site, map) → list of source projections
    by_site_map: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for pb in proj["projections"]:
        by_site_map.setdefault((pb["site_id"], pb["map"]), []).append(pb)

    sites_ordered = sorted({pb["site_id"] for pb in proj["projections"]})
    maps_ordered = [m for m in ("lining", "therm_top", "topk_driver") if any(pb["map"] == m for pb in proj["projections"])]

    for sid in sites_ordered:
        for map_name in maps_ordered:
            blocks = by_site_map.get((sid, map_name), [])
            if not blocks:
                continue
            lines.append("")
            lines.append("─" * 78)
            lines.append(f"  SITE {sid}  via map '{map_name}'")
            lines.append("─" * 78)
            for b in blocks:
                lines.extend(render_projection_block(b, header=f"  ── {b['source']} ──"))

    if proj["direct_site_keyed"]:
        lines.append("")
        lines.append("─" * 78)
        lines.append("  DIRECT SITE-KEYED (prism_therm.top_residues — no external map needed)")
        lines.append("─" * 78)
        for b in proj["direct_site_keyed"]:
            lines.append("")
            lines.extend(render_projection_block(b, header=f"  ── site {b['site_id']}  {b['source']} ──"))

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate tags/labels per site across PRISM-4D run artifacts."
    )
    ap.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory containing a PRISM-4D run's JSON artifacts.",
    )
    ap.add_argument(
        "--prefix", required=True,
        help="Filename prefix (e.g. '4lpk_clean').",
    )
    ap.add_argument(
        "--sites", required=True,
        help="Comma-separated site IDs to include (e.g. '0,1,2').",
    )
    ap.add_argument(
        "--json-out", type=Path, default=None,
        help="Optional path to write the full structured report as JSON.",
    )
    ap.add_argument(
        "--top-k", type=int, default=20,
        help="Top-K retention for categorical fields (default 20).",
    )
    ap.add_argument(
        "--reservoir", type=int, default=DEFAULT_RESERVOIR,
        help="Reservoir sample size for numeric quantile estimation.",
    )
    ap.add_argument(
        "--skip-spikes", action="store_true",
        help="Skip the multi-GB spike files (only aggregate the six small files).",
    )
    ap.add_argument(
        "--project-via", default="off",
        choices=["off", "all", "lining", "therm_top", "topk_driver"],
        help=(
            "Emit a separate site-projected residue-level aggregation via the "
            "chosen site→residue map(s): lining (binding_sites.lining_residues), "
            "therm_top (prism_therm.pockets.top_residues), topk_driver "
            "(kcc_validation.sites.topk_residues), or 'all' (all three "
            "independently, not unioned). The prism_therm.top_residues source "
            "is also emitted per site directly (it is already site-keyed). "
            "Every aggregated field carries both the unweighted stats and the "
            "support-weighted view (weight = n_samples / n_groups / "
            "active_causal_steps / n_causal_spikes depending on source)."
        ),
    )
    args = ap.parse_args()

    try:
        sites = [int(s.strip()) for s in args.sites.split(",") if s.strip()]
    except ValueError as e:
        sys.stderr.write(f"invalid --sites: {e}\n")
        return 2

    out_dir: Path = args.output_dir.resolve()
    if not out_dir.is_dir():
        sys.stderr.write(f"--output-dir not found: {out_dir}\n")
        return 2

    prefix = args.prefix
    report: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "prefix": prefix,
        "sites_requested": sites,
        "files": [],
    }
    text_sections: List[str] = []

    # ------------------------------------------------------------------
    # Per-site spike files
    # ------------------------------------------------------------------
    if not args.skip_spikes:
        for sid in sites:
            spike_path = out_dir / f"{prefix}.site{sid}.spike_events.json"
            if not spike_path.is_file():
                sys.stderr.write(f"[warn] missing: {spike_path}\n")
                continue
            sys.stderr.write(f"[info] streaming {spike_path} (this may take a few minutes)...\n")
            summary = aggregate_spike_file(spike_path, sid, args.reservoir)
            report["files"].append(summary)
            text_sections.append(render_file(summary))

    # ------------------------------------------------------------------
    # Six non-spike files
    # ------------------------------------------------------------------
    small_files = [
        f"{prefix}.topology.asc_consensus.json",
        f"{prefix}.topology.gcpid_synergy.json",
        f"{prefix}.topology.prism_therm.json",
        f"{prefix}.kcc_visualization.json",
        f"{prefix}.kcc_validation.json",
        f"{prefix}.ensemble_trajectory.json",
    ]
    for name in small_files:
        p = out_dir / name
        if not p.is_file():
            sys.stderr.write(f"[warn] missing: {p}\n")
            continue
        sys.stderr.write(f"[info] reading {p}\n")
        summary = aggregate_small_file(p, sites)
        report["files"].append(summary)
        text_sections.append(render_file(summary))

    # ------------------------------------------------------------------
    # --project-via residue-level site projection (optional)
    # ------------------------------------------------------------------
    if args.project_via != "off":
        if args.project_via == "all":
            maps_to_use = ["lining", "therm_top", "topk_driver"]
        else:
            maps_to_use = [args.project_via]
        sys.stderr.write(
            f"[info] running --project-via {args.project_via} "
            f"(maps={maps_to_use}) for sites {sites}\n"
        )
        proj = aggregate_projection(out_dir, prefix, sites, maps_to_use)
        report["projection"] = proj
        text_sections.append(render_projection(proj))

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------
    print("\n\n".join(text_sections))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as fh:
            json.dump(report, fh, indent=2, default=str)
        sys.stderr.write(f"[info] structured report written to {args.json_out}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
