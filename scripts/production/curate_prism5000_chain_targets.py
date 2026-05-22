#!/usr/bin/env python3
"""
Build a chain-level PRISM teacher target bank from live RCSB metadata.

The output unit is a single protein chain instance, not a PDB entry. The
selection logic is intentionally campaign-oriented:

* chain-level records with auth/asym/entity identifiers
* sequence-cluster diversity using RCSB 30/50/70/90% clusters
* category quotas across drug-discovery and cryptic-site relevant families
* explicit hard-target/novel-site tags, not just ligand-bound easy examples
* manifest files that can be consumed by the PRISM prep verifier

This script does not run MD and does not write topologies. It creates the
curated queue and metadata cache on the selected staging volume.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

CATEGORY_QUOTAS = {
    "oncology_signaling": 850,
    "epigenetic_ddr": 450,
    "proteostasis_e3": 400,
    "immunology_inflammation": 500,
    "neuro_gpcr_ion": 450,
    "infectious_disease": 600,
    "membrane_transport": 400,
    "enzyme_allostery": 700,
    "ppi_scaffold": 350,
    "apo_novel": 300,
}

CATEGORY_TERMS = {
    "oncology_signaling": (
        "kinase", "phosphatase", "ras", "raf", "mek", "erk", "pi3k", "akt",
        "mtor", "jak", "stat", "bcl", "p53", "mdm2", "myc", "tead",
        "yap", "parp", "wnt", "notch", "hedgehog", "mapk",
    ),
    "epigenetic_ddr": (
        "bromodomain", "chromodomain", "methyltransferase", "demethylase",
        "histone", "deacetylase", "hdac", "acetyltransferase", "parp",
        "atm", "atr", "dna repair", "helicase", "nuclease",
    ),
    "proteostasis_e3": (
        "ubiquitin", "ligase", "e3", "deubiquitinase", "dub", "proteasome",
        "cereblon", "vhl", "ring", "hect", "f-box", "chaperone",
    ),
    "immunology_inflammation": (
        "interleukin", "cytokine", "chemokine", "sting", "cgas", "toll",
        "inflammasome", "complement", "checkpoint", "pd-1", "pd-l1", "ctla",
        "hla", "mhc", "jak", "stat", "nf-kappa", "tnf",
    ),
    "neuro_gpcr_ion": (
        "gpcr", "g protein-coupled", "receptor", "ion channel", "channel",
        "transporter", "neuro", "serotonin", "dopamine", "glutamate",
        "gaba", "acetylcholine", "trp channel", "sodium channel",
    ),
    "infectious_disease": (
        "viral", "virus", "coronavirus", "influenza", "hiv", "hepatitis",
        "bacterial", "mycobacterium", "plasmodium", "malaria", "parasite",
        "protease", "polymerase", "replicase", "integrase", "capsid",
    ),
    "membrane_transport": (
        "transmembrane", "membrane", "transporter", "pump", "channel",
        "porin", "symporter", "antiporter", "abc transporter",
    ),
    "enzyme_allostery": (
        "allosteric", "enzyme", "dehydrogenase", "synthase", "synthetase",
        "transferase", "hydrolase", "isomerase", "oxidoreductase", "lyase",
        "binding protein", "metabolic",
    ),
    "ppi_scaffold": (
        "adapter", "adaptor", "scaffold", "sh2", "sh3", "pdz", "ww domain",
        "14-3-3", "coiled coil", "protein-protein", "interaction domain",
        "repeat protein",
    ),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def clean_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "UNK"


def lower_text(*parts: Any) -> str:
    return " ".join(str(p or "") for p in parts).lower()


def safe_min_resolution(entry: dict[str, Any]) -> float | None:
    values = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return min(nums) if nums else None


def request_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
    retries: int = 4,
) -> dict[str, Any]:
    cache_key = None
    if cache_dir is not None:
        raw_key = f"{method}:{url}:{json.dumps(body, sort_keys=True) if body else ''}"
        cache_key = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Accept": "application/json",
        "User-Agent": "prism5000-chain-curator/1.0",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"

    last_error: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
            if cache_dir is not None and cache_key is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / f"{cache_key}.json").write_text(json.dumps(data, sort_keys=True))
            return data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"RCSB request failed after {retries} attempts: {url}: {last_error}")


def rcsb_search_entry_ids(
    cache_dir: Path,
    *,
    min_len: int,
    max_len: int,
    max_entries: int,
) -> list[str]:
    nodes: list[dict[str, Any]] = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "operator": "exact_match",
                "value": "Protein (only)",
                "attribute": "rcsb_entry_info.selected_polymer_entity_types",
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "operator": "range",
                "value": {
                    "from": min_len,
                    "to": max_len,
                    "include_lower": True,
                    "include_upper": True,
                },
                "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
            },
        },
    ]

    all_ids: list[str] = []
    page_rows = 1000
    for start in range(0, max_entries, page_rows):
        body = {
            "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": start, "rows": min(page_rows, max_entries - start)},
                "results_content_type": ["experimental"],
            },
        }
        data = request_json(RCSB_SEARCH_URL, method="POST", body=body, cache_dir=cache_dir)
        page = [str(row["identifier"]).lower() for row in data.get("result_set", [])]
        if not page:
            break
        all_ids.extend(page)
        if len(page) < page_rows:
            break
    return list(dict.fromkeys(all_ids))[:max_entries]


def rcsb_category_entry_ids(
    cache_dir: Path,
    *,
    category: str,
    min_len: int,
    max_len: int,
    max_entries: int,
) -> list[str]:
    term_nodes: list[dict[str, Any]] = []
    # Keep the query bounded. The full category term list still drives local
    # scoring after metadata fetch.
    for term in CATEGORY_TERMS.get(category, ())[:14]:
        for attr in ("struct.title", "struct_keywords.text", "rcsb_polymer_entity.pdbx_description"):
            term_nodes.append(
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "operator": "contains_phrase",
                        "value": term,
                        "attribute": attr,
                    },
                }
            )

    if not term_nodes:
        return []

    query = {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "operator": "exact_match",
                    "value": "Protein (only)",
                    "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "operator": "range",
                    "value": {
                        "from": min_len,
                        "to": max_len,
                        "include_lower": True,
                        "include_upper": True,
                    },
                    "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                },
            },
            {"type": "group", "logical_operator": "or", "nodes": term_nodes},
        ],
    }

    all_ids: list[str] = []
    page_rows = 500
    for start in range(0, max_entries, page_rows):
        body = {
            "query": query,
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": start, "rows": min(page_rows, max_entries - start)},
                "results_content_type": ["experimental"],
            },
        }
        try:
            data = request_json(RCSB_SEARCH_URL, method="POST", body=body, cache_dir=cache_dir)
        except RuntimeError as exc:
            print(f"[warn] category search failed {category}: {exc}", file=sys.stderr)
            break
        page = [str(row["identifier"]).lower() for row in data.get("result_set", [])]
        if not page:
            break
        all_ids.extend(page)
        if len(page) < page_rows:
            break
    return list(dict.fromkeys(all_ids))[:max_entries]


def cluster_ids(entity: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in entity.get("rcsb_cluster_membership") or []:
        identity = str(item.get("identity") or "")
        cluster_id = item.get("cluster_id")
        if identity and cluster_id:
            out[f"cluster_{identity}_id"] = str(cluster_id)
    return out


def reference_uniprot_ids(entity: dict[str, Any]) -> list[str]:
    ids = entity.get("rcsb_polymer_entity_container_identifiers") or {}
    refs = ids.get("reference_sequence_identifiers") or []
    found: list[str] = []
    for ref in refs:
        if str(ref.get("database_name", "")).lower() == "uniprot":
            acc = ref.get("database_accession")
            if acc:
                found.append(str(acc))
    return sorted(set(found))


def chain_pairs(entity: dict[str, Any]) -> list[tuple[str, str]]:
    ids = entity.get("rcsb_polymer_entity_container_identifiers") or {}
    asym_ids = [str(x) for x in ids.get("asym_ids") or []]
    auth_ids = [str(x) for x in ids.get("auth_asym_ids") or []]
    if not auth_ids:
        strand = ((entity.get("entity_poly") or {}).get("pdbx_strand_id") or "")
        auth_ids = [x.strip() for x in strand.split(",") if x.strip()]
    if not asym_ids:
        asym_ids = auth_ids[:]
    if len(asym_ids) == len(auth_ids):
        return list(zip(asym_ids, auth_ids))
    return [(auth_id, auth_id) for auth_id in auth_ids]


def sequence_ok(sequence: str, min_unique: int) -> tuple[bool, dict[str, Any]]:
    seq = re.sub(r"\s+", "", sequence or "").upper()
    if not seq:
        return False, {"reject_reason": "empty_sequence"}
    noncanonical = sum(1 for aa in seq if aa not in VALID_AA)
    unique_valid = len(set(seq) & VALID_AA)
    aromatic = sum(1 for aa in seq if aa in "FWY")
    stats = {
        "sequence_length": len(seq),
        "unique_aa": unique_valid,
        "noncanonical_fraction": noncanonical / max(1, len(seq)),
        "aromatic_count": aromatic,
    }
    if unique_valid < min_unique:
        return False, stats | {"reject_reason": "low_residue_diversity"}
    if stats["noncanonical_fraction"] > 0.03:
        return False, stats | {"reject_reason": "too_many_noncanonical_residues"}
    if aromatic < 2:
        return False, stats | {"reject_reason": "weak_uv_lif_observability"}
    return True, stats


def category_hits(text: str, entry_nonpolymer_count: int) -> dict[str, int]:
    hits: dict[str, int] = {}
    for category, terms in CATEGORY_TERMS.items():
        score = 0
        for term in terms:
            if term in text:
                score += 2 if " " in term else 1
        if score:
            hits[category] = score
    if entry_nonpolymer_count == 0:
        hits["apo_novel"] = hits.get("apo_novel", 0) + 4
    return hits


def primary_category(hits: dict[str, int], nonpolymer_count: int) -> str:
    if hits:
        return sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return "enzyme_allostery" if nonpolymer_count > 0 else "apo_novel"


def difficulty_tags(
    *,
    text: str,
    length: int,
    nonpolymer_count: int,
    resolution: float | None,
    release_date: str,
    cluster30: str,
) -> list[str]:
    tags: list[str] = []
    if nonpolymer_count == 0:
        tags.append("apo_or_ligandless")
    else:
        tags.append("ligand_or_cofactor_present")
    if length >= 600:
        tags.append("large_chain")
    elif length >= 300:
        tags.append("mid_large_chain")
    if any(term in text for term in CATEGORY_TERMS["membrane_transport"]):
        tags.append("membrane_or_transport_challenge")
    if any(term in text for term in CATEGORY_TERMS["ppi_scaffold"]):
        tags.append("ppi_surface_risk")
    if "allosteric" in text or "cryptic" in text:
        tags.append("allosteric_or_cryptic_prior")
    if release_date >= "2019":
        tags.append("recent_structure")
    if resolution is None:
        tags.append("no_reported_resolution")
    elif resolution <= 2.2:
        tags.append("high_resolution")
    elif resolution >= 3.5:
        tags.append("low_resolution_challenge")
    if cluster30:
        tags.append("sequence_cluster_tracked")
    return tags


def selection_score(
    *,
    category: str,
    hit_score: int,
    length: int,
    nonpolymer_count: int,
    resolution: float | None,
    release_date: str,
    text: str,
) -> float:
    score = 0.0
    score += min(10.0, hit_score * 1.5)
    if nonpolymer_count > 0:
        score += 4.0
    else:
        score += 3.0
    if 120 <= length <= 550:
        score += 3.0
    elif length > 550:
        score += 1.5
    if resolution is not None:
        score += max(0.0, 4.0 - abs(resolution - 2.2))
    if release_date >= "2019":
        score += 2.0
    if "allosteric" in text or "cryptic" in text:
        score += 3.0
    if category in {"membrane_transport", "neuro_gpcr_ion", "ppi_scaffold"}:
        score += 1.5
    return round(score, 4)


@dataclass
class Candidate:
    target_id: str
    pdb_id: str
    entity_id: str
    asym_id: str
    auth_asym_id: str
    sequence_length: int
    resolution_angstrom: float | None
    experimental_methods: list[str]
    release_date: str
    title: str
    pdb_keywords: str
    entity_description: str
    organism: str
    uniprot_ids: list[str]
    nonpolymer_entity_count: int
    cluster_ids: dict[str, str]
    primary_category: str
    categories: list[str]
    difficulty_tags: list[str]
    selection_score: float
    sequence_stats: dict[str, Any]
    ligand_neighbor_count: int | None = None
    instance_feature_count: int | None = None
    prep_status: str = "selected"
    paths: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "pdb_id": self.pdb_id,
            "entity_id": self.entity_id,
            "asym_id": self.asym_id,
            "auth_asym_id": self.auth_asym_id,
            "sequence_length": self.sequence_length,
            "resolution_angstrom": self.resolution_angstrom,
            "experimental_methods": self.experimental_methods,
            "release_date": self.release_date,
            "title": self.title,
            "pdb_keywords": self.pdb_keywords,
            "entity_description": self.entity_description,
            "organism": self.organism,
            "uniprot_ids": self.uniprot_ids,
            "nonpolymer_entity_count": self.nonpolymer_entity_count,
            "cluster_ids": self.cluster_ids,
            "primary_category": self.primary_category,
            "categories": self.categories,
            "difficulty_tags": self.difficulty_tags,
            "selection_score": self.selection_score,
            "sequence_stats": self.sequence_stats,
            "ligand_neighbor_count": self.ligand_neighbor_count,
            "instance_feature_count": self.instance_feature_count,
            "prep_status": self.prep_status,
            "paths": self.paths,
        }


def build_candidates_for_entry(
    pdb_id: str,
    cache_dir: Path,
    *,
    min_len: int,
    max_len: int,
    min_unique_aa: int,
    include_redundant_chains: bool,
) -> list[Candidate]:
    try:
        entry = request_json(f"{RCSB_DATA_URL}/entry/{pdb_id}", cache_dir=cache_dir)
    except RuntimeError as exc:
        print(f"[warn] entry fetch failed {pdb_id}: {exc}", file=sys.stderr)
        return []

    info = entry.get("rcsb_entry_info") or {}
    ids = entry.get("rcsb_entry_container_identifiers") or {}
    entity_ids = [str(x) for x in ids.get("polymer_entity_ids") or []]
    resolution = safe_min_resolution(entry)
    methods = [str(x.get("method")) for x in entry.get("exptl") or [] if x.get("method")]
    nonpolymer_count = int(info.get("nonpolymer_entity_count") or 0)
    release_date = str((entry.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10]
    title = str((entry.get("struct") or {}).get("title") or "")
    keywords_data = entry.get("struct_keywords") or {}
    keywords = " ".join(
        str(keywords_data.get(key) or "") for key in ("pdbx_keywords", "text")
    )

    candidates: list[Candidate] = []
    for entity_id in entity_ids:
        try:
            entity = request_json(
                f"{RCSB_DATA_URL}/polymer_entity/{pdb_id}/{entity_id}",
                cache_dir=cache_dir,
            )
        except RuntimeError as exc:
            print(f"[warn] entity fetch failed {pdb_id}/{entity_id}: {exc}", file=sys.stderr)
            continue

        entity_poly = entity.get("entity_poly") or {}
        polymer_type = str(entity_poly.get("type") or "").lower()
        if "polypeptide" not in polymer_type:
            continue
        length = int(entity_poly.get("rcsb_sample_sequence_length") or 0)
        if length < min_len or length > max_len:
            continue
        sequence = entity_poly.get("pdbx_seq_one_letter_code_can") or entity_poly.get(
            "pdbx_seq_one_letter_code"
        ) or ""
        ok, seq_stats = sequence_ok(sequence, min_unique_aa)
        if not ok:
            continue

        entity_desc = str((entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "")
        organism = ""
        srcs = entity.get("rcsb_entity_source_organism") or []
        if srcs:
            organism = str(srcs[0].get("scientific_name") or "")
        text = lower_text(title, keywords, entity_desc, organism)
        hits = category_hits(text, nonpolymer_count)
        category = primary_category(hits, nonpolymer_count)
        cids = cluster_ids(entity)
        hit_score = hits.get(category, 0)
        score = selection_score(
            category=category,
            hit_score=hit_score,
            length=length,
            nonpolymer_count=nonpolymer_count,
            resolution=resolution,
            release_date=release_date,
            text=text,
        )
        tags = difficulty_tags(
            text=text,
            length=length,
            nonpolymer_count=nonpolymer_count,
            resolution=resolution,
            release_date=release_date,
            cluster30=cids.get("cluster_30_id", ""),
        )
        pairs = chain_pairs(entity)
        if not include_redundant_chains and pairs:
            pairs = pairs[:1]
        for asym_id, auth_id in pairs:
            target_id = f"{pdb_id.lower()}_chain_{clean_token(auth_id)}"
            candidates.append(
                Candidate(
                    target_id=target_id,
                    pdb_id=pdb_id.lower(),
                    entity_id=entity_id,
                    asym_id=asym_id,
                    auth_asym_id=auth_id,
                    sequence_length=length,
                    resolution_angstrom=resolution,
                    experimental_methods=methods,
                    release_date=release_date,
                    title=title,
                    pdb_keywords=keywords,
                    entity_description=entity_desc,
                    organism=organism,
                    uniprot_ids=reference_uniprot_ids(entity),
                    nonpolymer_entity_count=nonpolymer_count,
                    cluster_ids=cids,
                    primary_category=category,
                    categories=sorted(hits),
                    difficulty_tags=tags,
                    selection_score=score,
                    sequence_stats=seq_stats,
                )
            )
    return candidates


def enrich_instance(candidate: Candidate, cache_dir: Path) -> Candidate:
    try:
        instance = request_json(
            f"{RCSB_DATA_URL}/polymer_entity_instance/{candidate.pdb_id}/{candidate.asym_id}",
            cache_dir=cache_dir,
        )
    except RuntimeError:
        return candidate
    candidate.ligand_neighbor_count = len(instance.get("rcsb_ligand_neighbors") or [])
    features = instance.get("rcsb_polymer_instance_feature") or []
    candidate.instance_feature_count = len(features)
    if candidate.ligand_neighbor_count:
        candidate.difficulty_tags = sorted(
            set(candidate.difficulty_tags + ["chain_level_ligand_neighbors"])
        )
    return candidate


def select_balanced(candidates: list[Candidate], n_targets: int) -> list[Candidate]:
    candidates = sorted(candidates, key=lambda c: (-c.selection_score, c.pdb_id, c.auth_asym_id))
    by_category: dict[str, list[Candidate]] = {key: [] for key in CATEGORY_QUOTAS}
    for cand in candidates:
        by_category.setdefault(cand.primary_category, []).append(cand)

    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    cluster30_counts: dict[str, int] = {}

    def can_take(c: Candidate, relaxed: bool = False) -> bool:
        if c.target_id in selected_ids:
            return False
        cluster = c.cluster_ids.get("cluster_30_id") or f"entry:{c.pdb_id}:{c.entity_id}"
        limit = 2 if relaxed else 1
        return cluster30_counts.get(cluster, 0) < limit

    def take(c: Candidate) -> None:
        selected.append(c)
        selected_ids.add(c.target_id)
        cluster = c.cluster_ids.get("cluster_30_id") or f"entry:{c.pdb_id}:{c.entity_id}"
        cluster30_counts[cluster] = cluster30_counts.get(cluster, 0) + 1

    scale = n_targets / sum(CATEGORY_QUOTAS.values())
    scaled_quotas = {
        cat: max(1, int(math.floor(quota * scale))) for cat, quota in CATEGORY_QUOTAS.items()
    }
    deficit = n_targets - sum(scaled_quotas.values())
    for cat in sorted(scaled_quotas, key=lambda k: CATEGORY_QUOTAS[k], reverse=True)[:deficit]:
        scaled_quotas[cat] += 1

    for category, quota in scaled_quotas.items():
        for cand in by_category.get(category, []):
            if len([x for x in selected if x.primary_category == category]) >= quota:
                break
            if can_take(cand):
                take(cand)
            if len(selected) >= n_targets:
                return selected

    for relaxed in (False, True):
        for cand in candidates:
            if len(selected) >= n_targets:
                return selected
            if can_take(cand, relaxed=relaxed):
                take(cand)
    return selected


def write_outputs(out_dir: Path, selected: list[Candidate], all_candidates: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "prism5000_chain_manifest.jsonl"
    csv_path = out_dir / "prism5000_chain_manifest.csv"
    prep_path = out_dir / "prism5000_prep_plan.tsv"
    report_path = out_dir / "prism5000_category_report.json"

    rows = [cand.as_dict() for cand in selected]
    with jsonl_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = dict(row)
        flat["categories"] = ",".join(row["categories"])
        flat["difficulty_tags"] = ",".join(row["difficulty_tags"])
        flat["uniprot_ids"] = ",".join(row["uniprot_ids"])
        flat["cluster_30_id"] = row["cluster_ids"].get("cluster_30_id", "")
        flat["cluster_50_id"] = row["cluster_ids"].get("cluster_50_id", "")
        flat["cluster_70_id"] = row["cluster_ids"].get("cluster_70_id", "")
        flat["cluster_90_id"] = row["cluster_ids"].get("cluster_90_id", "")
        del flat["cluster_ids"]
        del flat["sequence_stats"]
        del flat["paths"]
        flat_rows.append(flat)

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    with prep_path.open("w") as fh:
        fh.write("target_id\tpdb_id\tauth_asym_id\tasym_id\tentity_id\tprimary_category\tsequence_length\n")
        for cand in selected:
            fh.write(
                "\t".join(
                    [
                        cand.target_id,
                        cand.pdb_id,
                        cand.auth_asym_id,
                        cand.asym_id,
                        cand.entity_id,
                        cand.primary_category,
                        str(cand.sequence_length),
                    ]
                )
                + "\n"
            )

    category_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    cluster30 = set()
    length_bins = {"80_150": 0, "151_300": 0, "301_600": 0, "601_plus": 0}
    for cand in selected:
        category_counts[cand.primary_category] = category_counts.get(cand.primary_category, 0) + 1
        for tag in cand.difficulty_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if cand.cluster_ids.get("cluster_30_id"):
            cluster30.add(cand.cluster_ids["cluster_30_id"])
        if cand.sequence_length <= 150:
            length_bins["80_150"] += 1
        elif cand.sequence_length <= 300:
            length_bins["151_300"] += 1
        elif cand.sequence_length <= 600:
            length_bins["301_600"] += 1
        else:
            length_bins["601_plus"] += 1

    report = {
        "generated_at_utc": utc_stamp(),
        "selected_targets": len(selected),
        "candidate_chains_considered": all_candidates,
        "category_counts": dict(sorted(category_counts.items())),
        "difficulty_tag_counts": dict(sorted(tag_counts.items())),
        "unique_cluster_30_count": len(cluster30),
        "length_bins": length_bins,
        "outputs": {
            "jsonl_manifest": str(jsonl_path),
            "csv_manifest": str(csv_path),
            "prep_plan_tsv": str(prep_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-targets", type=int, default=5000)
    parser.add_argument("--max-entry-candidates", type=int, default=18000)
    parser.add_argument("--category-entry-candidates", type=int, default=2500)
    parser.add_argument("--disable-category-search", action="store_true")
    parser.add_argument("--min-len", type=int, default=80)
    parser.add_argument("--max-len", type=int, default=900)
    parser.add_argument("--min-unique-aa", type=int, default=15)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--include-redundant-chains", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.out_dir / "rcsb_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[curate] output={args.out_dir}", file=sys.stderr)
    print(f"[curate] querying up to {args.max_entry_candidates} RCSB protein entries", file=sys.stderr)
    entry_ids = rcsb_search_entry_ids(
        cache_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        max_entries=args.max_entry_candidates,
    )
    print(f"[curate] broad entry candidates={len(entry_ids)}", file=sys.stderr)
    if not args.disable_category_search:
        for category in CATEGORY_QUOTAS:
            if category == "apo_novel":
                continue
            category_ids = rcsb_category_entry_ids(
                cache_dir,
                category=category,
                min_len=args.min_len,
                max_len=args.max_len,
                max_entries=args.category_entry_candidates,
            )
            before = len(entry_ids)
            entry_ids.extend(category_ids)
            entry_ids = list(dict.fromkeys(entry_ids))
            print(
                f"[curate] {category} targeted entries +{len(entry_ids) - before} "
                f"(pool={len(entry_ids)})",
                file=sys.stderr,
            )
    print(f"[curate] total unique entry candidates={len(entry_ids)}", file=sys.stderr)

    all_candidates: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                build_candidates_for_entry,
                pdb_id,
                cache_dir,
                min_len=args.min_len,
                max_len=args.max_len,
                min_unique_aa=args.min_unique_aa,
                include_redundant_chains=args.include_redundant_chains,
            ): pdb_id
            for pdb_id in entry_ids
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            all_candidates.extend(future.result())
            if idx % 250 == 0:
                print(
                    f"[curate] metadata {idx}/{len(futures)} entries, "
                    f"{len(all_candidates)} chain candidates",
                    file=sys.stderr,
                )

    print(f"[curate] chain candidates after hard filters={len(all_candidates)}", file=sys.stderr)
    selected = select_balanced(all_candidates, args.n_targets)
    if len(selected) < args.n_targets:
        print(
            f"[warn] selected only {len(selected)} targets; increase --max-entry-candidates "
            "or relax filters if exactly 5000 are required",
            file=sys.stderr,
        )

    print(f"[curate] enriching {len(selected)} selected chain instances", file=sys.stderr)
    enriched: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(enrich_instance, cand, cache_dir) for cand in selected]
        for idx, future in enumerate(as_completed(futures), start=1):
            enriched.append(future.result())
            if idx % 500 == 0:
                print(f"[curate] enriched {idx}/{len(selected)} instances", file=sys.stderr)

    enriched = sorted(enriched, key=lambda c: c.target_id)
    write_outputs(args.out_dir, enriched, len(all_candidates))
    print(f"[curate] wrote {len(enriched)} chain targets to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
