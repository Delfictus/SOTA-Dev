#!/usr/bin/env python3
"""Read-only PRISM-DSTW ontology/lineage/claim/provenance audit.

This script inventories existing repository artifacts and emits compact audit
indices. It does not regenerate scientific outputs, delete files, package
release artifacts, or upgrade provenance tiers.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TMPDIR", "/mnt/storage/tmp")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - fallback path is runtime dependent
    pa = None
    pq = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


ROOT = Path(__file__).resolve().parents[1]
ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"
TEXT_EXTS = {
    ".py",
    ".rs",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tsv",
    ".sh",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cu",
    ".ptx",
    ".js",
    ".ts",
    ".tsx",
    ".css",
    ".html",
    ".xml",
    ".svg",
    ".dot",
    ".graphml",
}

INVENTORY_DIRS = [
    "campaigns/glp1r_aleniglipron",
    "src",
    "scripts",
    "crates",
    "tests",
    ".audit-reports",
    "docs",
    "benchmarks",
]

DOMAINS = [
    "raw_observatory_data",
    "active_sets",
    "spectral_transfer_operator",
    "markov_validity_C4",
    "metastable_states_C1",
    "coarse_DTSG",
    "C6_dirichlet_reward",
    "C6_7_perron_guard",
    "captured_graph_tiles",
    "BSR_operator_runtime",
    "spectral_reward_manager",
    "Log_SubTB_training",
    "tile_credit_assignment",
    "active_learning_feedback",
    "variant_durability",
    "genealogical_receptor_panel",
    "chemical_tile_registry",
    "molecule_design_surface",
    "reports/dossiers/manifests",
    "tests/gates/subagent_reports",
    "release_packages/hashes/tags",
    "IP_secret_exposure",
]

DOMAIN_PATTERNS: list[tuple[str, list[str]]] = [
    ("raw_observatory_data", ["raw", "observ", "spike_events", "snr", "bocpd", "trajectory", "protocol_state", "forces_final"]),
    ("active_sets", ["active_set", "active_sets", "basin", "membership", "chi"]),
    ("spectral_transfer_operator", ["w_dir", "w_spec", "transfer", "transition", "operator", "spectral"]),
    ("markov_validity_C4", ["c4", "timescale", "markov", "convergence"]),
    ("metastable_states_C1", ["c1", "metastable", "chi", "membership"]),
    ("coarse_DTSG", ["dtsg", "coarse"]),
    ("C6_dirichlet_reward", ["c6", "dirichlet", "reward", "survival"]),
    ("C6_7_perron_guard", ["c6.7", "c6_7", "perron", "robust"]),
    ("captured_graph_tiles", ["captured", "tile", "graph_tile"]),
    ("BSR_operator_runtime", ["bsr", "block_sparse", "runtime"]),
    ("spectral_reward_manager", ["spectral_reward_manager", "reward_manager"]),
    ("Log_SubTB_training", ["log_subtb", "subtb", "gflownet", "training", "metrics"]),
    ("tile_credit_assignment", ["tile_credit", "credit_assignment", "motif_index"]),
    ("active_learning_feedback", ["active_learning", "acquisition", "uncertainty", "feedback", "bald"]),
    ("variant_durability", ["variant", "durability", "wt_vs_variant", "pgx"]),
    ("genealogical_receptor_panel", ["genealogical", "receptor_panel", "cross_species", "species"]),
    ("chemical_tile_registry", ["chemical_tile", "tile_registry", "fragment_registry", "brics", "smiles"]),
    ("molecule_design_surface", ["molecule", "candidate", "smiles", "rdkit", "medchem", "pains", "synthetic"]),
    ("reports/dossiers/manifests", ["report", "dossier", "manifest", "summary", "deliverable"]),
    ("tests/gates/subagent_reports", ["test", "gate", "subagent", "bug_hunter", "pytest", "clippy"]),
    ("release_packages/hashes/tags", ["release", "hash", "sha256", "tag", "package"]),
    ("IP_secret_exposure", ["credential", "secret", "token", "password", "patent", "trade_secret"]),
]

FORMAL_SECTIONS = [
    "Section I non-overclaim boundary",
    "Section II provenance tiers",
    "Axiom 8.5 estimator",
    "C0 transfer operator",
    "C1 metastable state extraction",
    "C2 chronology/eigenvalue decay",
    "C3 bisimulation/lumpability",
    "C4 timescale convergence",
    "C5 calibration/memory kernel",
    "C6 restricted Dirichlet survival",
    "C6.7 Perron robustness",
    "retained federated operators",
    "identity caveat",
    "open physical blockers",
]

SPEC_NAME_PATTERNS = [
    re.compile(r"^DSTW_FORMAL_SPECIFICATION_v1\.md$", re.I),
    re.compile(r"PRISM[-_ ]DSTW.*Formal.*Specification.*v1", re.I),
]
SPEC_CONTENT_PATTERNS = [
    re.compile(r"PRISM-DSTW Formal Specification v1", re.I),
    re.compile(r"Canonical Spectral-Spine Formalism", re.I),
]

CLAIM_PATTERNS: dict[str, re.Pattern[str]] = {
    "chronology": re.compile(r"\b(chronolog|temporal order|time[- ]ordered|trajectory)\b", re.I),
    "causality": re.compile(r"\b(causal|causality|causes|drives|mechanism)\b", re.I),
    "entropy": re.compile(r"\b(entropy|entropic)\b", re.I),
    "free_energy": re.compile(r"\b(free energy|delta g|dg|thermodynamic)\b", re.I),
    "reward": re.compile(r"\b(reward|dirichlet|survival)\b", re.I),
    "molecule_design": re.compile(r"\b(molecule design|de novo|candidate|smiles|medchem|rdkit)\b", re.I),
    "variant_resilience": re.compile(r"\b(variant|resilien|mutation|pgx)\b", re.I),
    "durability": re.compile(r"\b(durability|durable|robust)\b", re.I),
    "biological_effect": re.compile(r"\b(biological effect|agonis|antagonis|efficacy|potency)\b", re.I),
    "clinical_effect": re.compile(r"\b(clinical|patient|therapeutic|weight loss|diabetes)\b", re.I),
    "assay_prediction": re.compile(r"\b(assay|wet.?lab|prediction|predicted)\b", re.I),
    "calibration": re.compile(r"\b(calibration|calibrated|anchor)\b", re.I),
    "operational_status": re.compile(r"\b(operational|production[- ]ready|ready|complete|validated)\b", re.I),
}

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "CLOUDFLARE_TOKEN": re.compile(r"\b(cf(?:at|ut)_[A-Za-z0-9_-]{20,})\b"),
    "AWS_ACCESS_KEY": re.compile(r"\b(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16})\b"),
    "PRIVATE_KEY": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GENERIC_SECRET_ASSIGNMENT": re.compile(
        r"(?i)\b(api[_-]?key|secret|password|passwd|token|access[_-]?key|secret[_-]?key)\b\s*[:=]\s*['\"]?([^'\"\s]{8,})"
    ),
    "OPENAI_STYLE_KEY": re.compile(r"\b(sk-[A-Za-z0-9_-]{20,})\b"),
}

IP_PATTERNS: dict[str, re.Pattern[str]] = {
    "PATENT_RISK_DISCLOSURE": re.compile(r"\b(patentable|patent pending|claim construction|novel mechanism|invention)\b", re.I),
    "TRADE_SECRET_LEAK": re.compile(r"\b(trade secret|proprietary fragment|saleable backend|unpublished)\b", re.I),
    "CLIENT_SENSITIVE_DATA": re.compile(r"\b(client confidential|customer confidential|client data)\b", re.I),
}

CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
RUST_FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\b")
RUST_STRUCT_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)\b")
RUST_ENUM_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)\b")
RUST_TRAIT_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)\b")
RUST_MOD_RE = re.compile(r"^\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\b")
RUST_USE_RE = re.compile(r"^\s*use\s+(.+?);")


def utc_now() -> str:
    return dt.datetime.utcnow().strftime(ISO_UTC)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def iso_mtime(path: Path) -> str:
    return dt.datetime.utcfromtimestamp(path.stat().st_mtime).strftime(ISO_UTC)


def run_git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout


def get_git_maps() -> tuple[set[str], dict[str, str]]:
    tracked = set(run_git(["ls-files"]).splitlines())
    status_map: dict[str, str] = {}
    raw = run_git(["status", "--porcelain=v1", "-z"])
    parts = raw.split("\0")
    idx = 0
    while idx < len(parts):
        entry = parts[idx]
        idx += 1
        if not entry:
            continue
        code = entry[:2]
        path = entry[3:]
        if code.startswith("R") or code.startswith("C"):
            if idx < len(parts):
                new_path = parts[idx]
                idx += 1
                path = new_path
        status_map[path] = code.strip() or "CLEAN"
    return tracked, status_map


def git_status_for(path: str, tracked: set[str], status_map: dict[str, str]) -> str:
    if path in status_map:
        return status_map[path]
    if path in tracked:
        return "TRACKED_CLEAN"
    return "UNTRACKED"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def redacted_fingerprint(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()
    prefix = value[:4] if len(value) >= 4 else value
    suffix = value[-4:] if len(value) >= 8 else ""
    return f"{prefix}...{suffix}|sha256:{digest[:16]}|len:{len(value)}"


def text_sample(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        with path.open("rb") as fh:
            data = fh.read(max_bytes)
        if b"\0" in data[:8192]:
            return ""
        return data.decode("utf-8", "replace")
    except Exception:
        return ""


def is_textual(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTS


def iter_scope_files(audit_root: Path) -> Iterable[Path]:
    audit_root_resolved = audit_root.resolve()
    for dirname in INVENTORY_DIRS:
        base = ROOT / dirname
        if not base.exists():
            continue
        if base.is_file():
            yield base
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            d = Path(dirpath)
            resolved_d = d.resolve()
            if resolved_d == audit_root_resolved or audit_root_resolved in resolved_d.parents:
                dirnames[:] = []
                continue
            if ".git" in d.parts:
                dirnames[:] = []
                continue
            for name in filenames:
                yield d / name


def infer_domain(path_str: str) -> str:
    lower = path_str.lower()
    scores: Counter[str] = Counter()
    for domain, patterns in DOMAIN_PATTERNS:
        for pat in patterns:
            if pat in lower:
                scores[domain] += 1
    if scores:
        return scores.most_common(1)[0][0]
    if "/tests/" in lower or lower.startswith("tests/"):
        return "tests/gates/subagent_reports"
    if lower.endswith((".md", ".json", ".yaml", ".yml")):
        return "reports/dossiers/manifests"
    return "raw_observatory_data"


def infer_semantic_role(path_str: str) -> str:
    lower = path_str.lower()
    role_patterns = [
        ("W_dir", ["w_dir", "directed"]),
        ("W_spec", ["w_spec", "spectral_operator"]),
        ("transition_operator", ["transition", "operator"]),
        ("C4_report", ["c4", "timescale", "markov"]),
        ("chi_membership", ["chi", "membership"]),
        ("DTSG", ["dtsg"]),
        ("C6_reward", ["c6", "dirichlet", "reward"]),
        ("C6_7_verdict", ["c6.7", "c6_7", "perron"]),
        ("captured_graph_manifest", ["captured", "graph", "tile"]),
        ("reward_diagnostic", ["reward", "diagnostic"]),
        ("training_metric", ["subtb", "gflownet", "metric", "training"]),
        ("report", ["report", "dossier", "summary", "deliverable"]),
        ("release_package", ["release", "package", "tag"]),
        ("schema_table", [".parquet", ".csv", ".tsv"]),
        ("runtime_manifest", ["manifest", "protocol_state", "provenance"]),
        ("source_code", [".py", ".rs", ".sh"]),
    ]
    for role, pats in role_patterns:
        if any(p in lower for p in pats):
            return role
    return "artifact"


def infer_provenance_tier(path_str: str) -> str:
    lower = path_str.lower()
    if any(k in lower for k in ["w_dir", "w_spec", "current", "asymmetry", "survival", "reward", "dissipation", "dirichlet"]):
        return "L3_DERIVED"
    if any(k in lower for k in ["spike_events", "forces_final", "trajectory", ".bin", "raw", "pdb", "sdf", "protocol_state", "bocpd"]):
        return "L5_OBSERVED"
    if any(k in lower for k in ["report", "dossier", "summary", "manifest", "metric", "operator", "transition", "variant", "tile"]):
        return "L3_DERIVED"
    return "UNKNOWN_TIER"


def infer_canonical_status(path_str: str, spec_found: bool, before_spec: bool | None) -> str:
    lower = path_str.lower()
    if not spec_found:
        return "BLOCKED"
    if any(k in lower for k in ["backup", "old", "deprecated", "legacy", "pre_spec", "prespec"]):
        return "PRE_SPEC"
    if before_spec:
        return "STALE_UNDER_V1"
    if any(k in lower for k in ["canonical", "formal", "dstw"]):
        return "CANONICAL_CONFORMANT"
    return "PARTIAL"


def detect_schema(path: Path) -> tuple[str, int | None, list[str], dict[str, Any]]:
    suffix = path.suffix.lower()
    notes: dict[str, Any] = {}
    try:
        if suffix == ".parquet" and pq is not None:
            pf = pq.ParquetFile(path)
            row_count = pf.metadata.num_rows if pf.metadata else None
            cols = [field.name for field in pf.schema_arrow]
            notes["null_rates"] = "NOT_COMPUTED_METADATA_ONLY"
            notes["duplicate_keys"] = "NOT_COMPUTED_METADATA_ONLY"
            return "parquet", row_count, cols, notes
        if suffix in {".csv", ".tsv"}:
            delim = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.reader(fh, delimiter=delim)
                header = next(reader, [])
                rows = sum(1 for _ in reader)
            notes["null_rates"] = "NOT_COMPUTED_LIGHTWEIGHT"
            notes["duplicate_keys"] = "NOT_COMPUTED_LIGHTWEIGHT"
            return suffix[1:], rows, header, notes
        if suffix in {".json", ".jsonl"}:
            if suffix == ".jsonl":
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    first = fh.readline()
                    row_count = 1 + sum(1 for _ in fh) if first else 0
                cols: list[str] = []
                if first:
                    obj = json.loads(first)
                    if isinstance(obj, dict):
                        cols = list(obj.keys())
                return "jsonl", row_count, cols, {"null_rates": "NOT_COMPUTED_METADATA_ONLY"}
            if path.stat().st_size <= 20_000_000:
                obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(obj, dict):
                    return "json", 1, list(obj.keys()), {"json_type": "object"}
                if isinstance(obj, list):
                    cols = list(obj[0].keys()) if obj and isinstance(obj[0], dict) else []
                    return "json", len(obj), cols, {"json_type": "array"}
            return "json", None, [], {"schema_warning": "JSON_TOO_LARGE_FOR_FULL_READ"}
        if suffix in {".yaml", ".yml"} and yaml is not None and path.stat().st_size <= 5_000_000:
            obj = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, dict):
                return "yaml", 1, list(obj.keys()), {"yaml_type": "object"}
            if isinstance(obj, list):
                return "yaml", len(obj), [], {"yaml_type": "array"}
            return "yaml", None, [], {"yaml_type": type(obj).__name__}
    except Exception as exc:
        return suffix[1:] if suffix else "unknown", None, [], {"schema_error": str(exc)[:240]}
    return "", None, [], {}


def safe_json_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_markdown_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> None:
    shown = rows if limit is None else rows[:limit]
    lines = [f"# {title}", "", f"row_count: {len(rows)}", ""]
    if rows:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in shown:
            vals = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, (list, dict)):
                    val = json.dumps(val, ensure_ascii=True, sort_keys=True)
                text = str(val).replace("|", "\\|").replace("\n", " ")
                if len(text) > 220:
                    text = text[:217] + "..."
                vals.append(text)
            lines.append("| " + " | ".join(vals) + " |")
        if limit is not None and len(rows) > limit:
            lines.extend(["", f"Truncated to first {limit} rows in Markdown. Full data is in JSON/Parquet/SQLite."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if pa is None or pq is None:
        fallback = path.with_suffix(path.suffix + ".json_fallback")
        write_json(fallback, rows)
        return
    normalized: list[dict[str, Any]] = []
    columns: set[str] = set()
    for row in rows:
        out = {k: safe_json_value(v) for k, v in row.items()}
        columns.update(out.keys())
        normalized.append(out)
    all_cols = sorted(columns)
    table = pa.Table.from_pylist([{col: row.get(col) for col in all_cols} for row in normalized])
    pq.write_table(table, path, compression="zstd")


def find_canonical_spec(tracked: set[str], status_map: dict[str, str]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        d = Path(dirpath)
        if any(part in {".git", "target", "__pycache__", "node_modules", ".venv", ".figvenv"} for part in d.parts):
            dirnames[:] = []
            continue
        for name in filenames:
            p = d / name
            r = rel(p)
            by_name = any(rx.search(name) or rx.search(r) for rx in SPEC_NAME_PATTERNS)
            by_content = False
            matched_phrases: list[str] = []
            if by_name or (p.suffix.lower() in {".md", ".txt", ".rst"} and p.stat().st_size <= 10_000_000):
                sample = text_sample(p, 10_000_000)
                for rx in SPEC_CONTENT_PATTERNS:
                    if rx.search(sample):
                        by_content = True
                        matched_phrases.append(rx.pattern)
            if by_name or by_content:
                formal_detected = []
                sample = text_sample(p, 10_000_000)
                for section in FORMAL_SECTIONS:
                    simple = section.split(" ", 2)[0]
                    if section.lower() in sample.lower() or simple.lower() in sample.lower():
                        formal_detected.append(section)
                candidates.append(
                    {
                        "spec_path": r,
                        "sha256": sha256_file(p),
                        "git_status": git_status_for(r, tracked, status_map),
                        "modified_time": iso_mtime(p),
                        "formal_sections_detected": formal_detected,
                        "matched_by_name": by_name,
                        "matched_by_content": by_content,
                        "matched_phrases": matched_phrases,
                    }
                )
    if not candidates:
        return {
            "spec_path": None,
            "sha256": None,
            "git_status": None,
            "modified_time": None,
            "formal_sections_detected": [],
            "canonical_status": "BLOCKED_NO_CANONICAL_SPEC",
            "candidate_count": 0,
            "candidates": [],
        }
    exact = [c for c in candidates if c["matched_by_content"] or Path(c["spec_path"]).name == "DSTW_FORMAL_SPECIFICATION_v1.md"]
    chosen = exact[0] if exact else candidates[0]
    status = "CANONICAL_SPEC_FOUND" if len(exact) == 1 else "BLOCKED_CANONICAL_AMBIGUITY"
    return {
        "spec_path": chosen["spec_path"],
        "sha256": chosen["sha256"],
        "git_status": chosen["git_status"],
        "modified_time": chosen["modified_time"],
        "formal_sections_detected": chosen["formal_sections_detected"],
        "canonical_status": status,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def make_audit_scope(spec_status: str, artifacts_by_domain: Counter[str]) -> list[dict[str, Any]]:
    rows = []
    for domain in DOMAINS:
        count = artifacts_by_domain.get(domain, 0)
        if spec_status == "BLOCKED_NO_CANONICAL_SPEC":
            status = "BLOCKED" if count else "MISSING"
        elif count == 0:
            status = "MISSING"
        else:
            status = "PARTIAL"
        rows.append(
            {
                "domain": domain,
                "status": status,
                "artifact_count": count,
                "canonical_status": spec_status,
                "notes": "No canonical v1 spec found" if spec_status == "BLOCKED_NO_CANONICAL_SPEC" else "",
            }
        )
    return rows


def inventory_artifacts(audit_root: Path, spec: dict[str, Any], tracked: set[str], status_map: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec_mtime = None
    if spec.get("spec_path"):
        try:
            spec_mtime = (ROOT / spec["spec_path"]).stat().st_mtime
        except Exception:
            spec_mtime = None
    rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    sha_groups: defaultdict[str, list[str]] = defaultdict(list)
    files = list(iter_scope_files(audit_root))
    total = len(files)
    started = time.time()
    for idx, path in enumerate(files, 1):
        if idx == 1 or idx % 250 == 0 or idx == total:
            print(f"[audit] inventory {idx}/{total}: {rel(path)}", file=sys.stderr, flush=True)
        path_str = rel(path)
        stat = path.stat()
        digest = sha256_file(path)
        sha_groups[digest].append(path_str)
        schema_type, row_count, columns, schema_notes = detect_schema(path)
        domain = infer_domain(path_str)
        role = infer_semantic_role(path_str)
        before_spec = None if spec_mtime is None else stat.st_mtime < spec_mtime
        tier = infer_provenance_tier(path_str)
        generated_by = infer_generated_by(path_str)
        consumed_by = infer_consumed_by(path_str)
        row = {
            "artifact_id": f"ART-{idx:06d}",
            "path": path_str,
            "file_type": path.suffix.lower().lstrip(".") or "no_extension",
            "size_bytes": stat.st_size,
            "sha256": digest,
            "modified_time": iso_mtime(path),
            "git_tracked": path_str in tracked,
            "git_status": git_status_for(path_str, tracked, status_map),
            "created_before_spec_lock": before_spec,
            "owning_domain": domain,
            "semantic_role": role,
            "provenance_tier": tier,
            "canonical_status": infer_canonical_status(path_str, spec.get("canonical_status") != "BLOCKED_NO_CANONICAL_SPEC", bool(before_spec)),
            "duplicate_group_id": "",
            "source_of_truth_status": "UNKNOWN",
            "generated_by_script": generated_by,
            "consumed_by_script": consumed_by,
            "schema_detected": schema_type,
            "row_count_if_table": row_count,
            "columns_if_table": columns,
            "blocker_notes": "",
        }
        if schema_type:
            schema_rows.append(
                {
                    "path": path_str,
                    "schema_detected": schema_type,
                    "row_count_if_table": row_count,
                    "columns_if_table": columns,
                    "required_columns_status": required_columns_status(path_str, columns),
                    "type_mismatches": schema_notes.get("schema_error", ""),
                    "null_rates": schema_notes.get("null_rates", ""),
                    "duplicate_keys": schema_notes.get("duplicate_keys", ""),
                    "semantic_key_consistency": semantic_key_consistency(path_str, columns),
                    "notes": json.dumps(schema_notes, ensure_ascii=True, sort_keys=True),
                }
            )
        rows.append(row)
    duplicate_ids: dict[str, str] = {}
    for n, (digest, paths) in enumerate((item for item in sha_groups.items() if len(item[1]) > 1), 1):
        duplicate_ids[digest] = f"DUP-SHA-{n:05d}"
    for row in rows:
        if row["sha256"] in duplicate_ids:
            row["duplicate_group_id"] = duplicate_ids[row["sha256"]]
            row["source_of_truth_status"] = "DUPLICATE_NEEDS_REVIEW"
        else:
            row["source_of_truth_status"] = "SINGLE_CANDIDATE"
    print(f"[audit] inventory complete: {len(rows)} artifacts in {time.time() - started:.1f}s", file=sys.stderr, flush=True)
    return rows, schema_rows


def infer_generated_by(path_str: str) -> str:
    lower = path_str.lower()
    script_hints = {
        "generate_genealogical_variant_panel.py": ["genealogical", "variant_panel"],
        "build_variant_manifold_coverage_matrix.py": ["variant_manifold", "coverage_matrix"],
        "build_transition_trajectory_tensor.py": ["transition_trajectory_tensor"],
        "run_log_subtb_spectral_gflownet.py": ["subtb", "gflownet"],
        "compute_species_selectivity.py": ["species_selectivity"],
        "build_gflownet_reward_landscape.py": ["reward_landscape"],
        "generate_track_b_release_manifest.py": ["track_b_release_manifest"],
        "instantiate_track_b_runtime.py": ["track_b_runtime"],
    }
    for script, keys in script_hints.items():
        if any(k in lower for k in keys):
            return f"scripts/{script}"
    return ""


def infer_consumed_by(path_str: str) -> str:
    lower = path_str.lower()
    if any(k in lower for k in ["spike_events", "transition_chronology", "phase_manifold"]):
        return "scripts/build_transition_trajectory_tensor.py"
    if any(k in lower for k in ["variant", "pgx", "cross_species"]):
        return "scripts/generate_genealogical_variant_panel.py"
    if any(k in lower for k in ["reward", "gflownet", "subtb"]):
        return "scripts/run_log_subtb_spectral_gflownet.py"
    if any(k in lower for k in ["smiles", "fragment", "candidate"]):
        return "scripts/filter_gflownet_medchem_plausibility.py"
    return ""


def required_columns_status(path_str: str, columns: list[str]) -> str:
    lower = path_str.lower()
    cols = {c.lower() for c in columns}
    requirements = {
        "spike": {"residue", "time"},
        "transition": {"source", "target"},
        "reward": {"reward"},
        "variant": {"variant"},
        "smiles": {"smiles"},
        "tile": {"tile_id"},
    }
    for key, req in requirements.items():
        if key in lower:
            if req <= cols:
                return "REQUIRED_COLUMNS_PRESENT"
            return f"MISSING_REQUIRED_COLUMNS:{','.join(sorted(req - cols))}"
    return "NO_SPECIAL_GATE"


def semantic_key_consistency(path_str: str, columns: list[str]) -> str:
    lower = path_str.lower()
    cols = {c.lower() for c in columns}
    if "tile" in lower and "tile_id" not in cols:
        return "MISSING_TILE_ID"
    if "variant" in lower and not ({"variant", "variant_id", "mutation"} & cols):
        return "MISSING_VARIANT_KEY"
    if "smiles" in lower and "smiles" not in cols and "canonical_smiles" not in cols:
        return "MISSING_SMILES_KEY"
    return "NOT_FLAGGED"


def python_symbols(path: Path, path_str: str) -> list[dict[str, Any]]:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except Exception as exc:
        return [
            {
                "symbol_id": "",
                "file_path": path_str,
                "language": "python",
                "symbol_type": "PARSE_ERROR",
                "symbol_name": "PARSE_ERROR",
                "fully_qualified_name": f"{path_str}:PARSE_ERROR",
                "line_start": 0,
                "line_end": 0,
                "imports_used": [],
                "calls_detected": [],
                "owning_domain": infer_domain(path_str),
                "canonical_spec_mapping": "",
                "risk_flags": [f"PARSE_ERROR:{str(exc)[:120]}"],
            }
        ]
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.extend(f"{mod}.{alias.name}".strip(".") for alias in node.names)
    rows = []
    module = path_str[:-3].replace("/", ".") if path_str.endswith(".py") else path_str.replace("/", ".")
    for node in ast.walk(tree):
        symbol_type = ""
        name = ""
        if isinstance(node, ast.ClassDef):
            symbol_type = "class"
            name = node.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbol_type = "function"
            name = node.name
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name) and t.id.isupper()]
            if targets:
                symbol_type = "constant"
                name = ",".join(targets)
        if not symbol_type:
            continue
        calls: list[str] = []
        decorators: list[str] = []
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    calls.append(call_name(child.func))
            for dec in getattr(node, "decorator_list", []):
                decorators.append(call_name(dec))
            if any(d.endswith("dataclass") or d == "dataclass" for d in decorators) and symbol_type == "class":
                symbol_type = "dataclass"
        risk = risk_flags_for(path_str, name, "\n".join(src.splitlines()[max(getattr(node, "lineno", 1) - 5, 0) : getattr(node, "end_lineno", getattr(node, "lineno", 1)) + 5]))
        rows.append(
            {
                "symbol_id": "",
                "file_path": path_str,
                "language": "python",
                "symbol_type": symbol_type,
                "symbol_name": name,
                "fully_qualified_name": f"{module}.{name}",
                "line_start": getattr(node, "lineno", 0),
                "line_end": getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                "imports_used": sorted(set(imports)),
                "calls_detected": sorted(set(c for c in calls if c)),
                "decorators": sorted(set(d for d in decorators if d)),
                "owning_domain": infer_domain(path_str),
                "canonical_spec_mapping": map_symbol_to_spec(path_str, name),
                "risk_flags": risk,
            }
        )
    if "if __name__" in src and "__main__" in src:
        rows.append(
            {
                "symbol_id": "",
                "file_path": path_str,
                "language": "python",
                "symbol_type": "CLI entrypoint",
                "symbol_name": "__main__",
                "fully_qualified_name": f"{module}.__main__",
                "line_start": 0,
                "line_end": 0,
                "imports_used": sorted(set(imports)),
                "calls_detected": sorted(set(CALL_RE.findall(src))),
                "decorators": [],
                "owning_domain": infer_domain(path_str),
                "canonical_spec_mapping": map_symbol_to_spec(path_str, "__main__"),
                "risk_flags": risk_flags_for(path_str, "__main__", src),
            }
        )
    return rows


def call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return ""


def rust_symbols(path: Path, path_str: str) -> list[dict[str, Any]]:
    src = text_sample(path, 5_000_000)
    lines = src.splitlines()
    imports: list[str] = []
    rows = []
    for i, line in enumerate(lines, 1):
        use_match = RUST_USE_RE.search(line)
        if use_match:
            imports.append(use_match.group(1))
        matches = [
            ("function", RUST_FN_RE.search(line)),
            ("struct", RUST_STRUCT_RE.search(line)),
            ("enum", RUST_ENUM_RE.search(line)),
            ("trait", RUST_TRAIT_RE.search(line)),
            ("module", RUST_MOD_RE.search(line)),
        ]
        for symbol_type, match in matches:
            if not match:
                continue
            name = match.group(1)
            context = "\n".join(lines[max(0, i - 5) : min(len(lines), i + 20)])
            rows.append(
                {
                    "symbol_id": "",
                    "file_path": path_str,
                    "language": "rust",
                    "symbol_type": symbol_type,
                    "symbol_name": name,
                    "fully_qualified_name": rust_fqn(path_str, name),
                    "line_start": i,
                    "line_end": estimate_rust_end(lines, i),
                    "imports_used": sorted(set(imports)),
                    "calls_detected": sorted(set(CALL_RE.findall(context))),
                    "owning_domain": infer_domain(path_str),
                    "canonical_spec_mapping": map_symbol_to_spec(path_str, name),
                    "risk_flags": risk_flags_for(path_str, name, context),
                    "public_api": line.lstrip().startswith("pub "),
                    "binary": "/src/bin/" in path_str or path.name == "main.rs",
                }
            )
    return rows


def rust_fqn(path_str: str, name: str) -> str:
    no_ext = re.sub(r"\.rs$", "", path_str)
    parts = [p for p in no_ext.split("/") if p not in {"src", "bin", "crates"}]
    return "::".join(parts + [name])


def estimate_rust_end(lines: list[str], start_line: int) -> int:
    depth = 0
    seen = False
    for idx in range(start_line - 1, min(len(lines), start_line + 400)):
        line = lines[idx]
        depth += line.count("{")
        if "{" in line:
            seen = True
        depth -= line.count("}")
        if seen and depth <= 0:
            return idx + 1
    return start_line


def risk_flags_for(path_str: str, symbol_name: str, context: str) -> list[str]:
    lower = f"{path_str}\n{symbol_name}\n{context}".lower()
    flags = []
    if any(k in lower for k in ["heuristic", "magic", "threshold", "fallback"]):
        flags.append("HIDDEN_HEURISTIC_LOGIC")
    if "reward" in lower and not any(k in lower for k in ["c6", "dirichlet", "spectral_reward_manager"]):
        flags.append("NONCANONICAL_REWARD")
    if "operator" in lower and not any(k in lower for k in ["w_spec", "w_dir", "transfer", "spectral"]):
        flags.append("NONCANONICAL_OPERATOR")
    if "cpu" in lower and "fallback" in lower:
        flags.append("CPU_FALLBACK_PATH")
    if any(k in lower for k in ["/tmp", "tempfile", "scratch"]) and "manifest" not in lower:
        flags.append("UNTRACKED_RUNTIME_PATH")
    if any(k in lower for k in ["todo", "stub", "placeholder", "notimplemented"]):
        flags.append("PLACEHOLDER_STUB")
    if any(k in lower for k in ["v2", "old", "legacy", "duplicate"]):
        flags.append("DUPLICATE_IMPLEMENTATION")
    return sorted(set(flags))


def map_symbol_to_spec(path_str: str, symbol_name: str) -> str:
    lower = f"{path_str} {symbol_name}".lower()
    mappings = [
        ("C0 transfer operator", ["w_spec", "w_dir", "transfer", "operator", "transition"]),
        ("C1 metastable state extraction", ["metastable", "chi", "membership"]),
        ("C4 timescale convergence", ["c4", "markov", "timescale"]),
        ("C6 restricted Dirichlet survival", ["c6", "dirichlet", "reward", "survival"]),
        ("C6.7 Perron robustness", ["perron", "robust", "c6_7"]),
        ("Section II provenance tiers", ["provenance", "tier"]),
        ("retained federated operators", ["federated", "retained"]),
        ("identity caveat", ["identity", "caveat"]),
    ]
    for section, pats in mappings:
        if any(p in lower for p in pats):
            return section
    return ""


def index_code_symbols(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    source_paths = [ROOT / a["path"] for a in artifacts if a["path"].endswith((".py", ".rs"))]
    total = len(source_paths)
    for idx, path in enumerate(source_paths, 1):
        if idx == 1 or idx % 100 == 0 or idx == total:
            print(f"[audit] symbols {idx}/{total}: {rel(path)}", file=sys.stderr, flush=True)
        path_str = rel(path)
        if path.suffix == ".py":
            rows.extend(python_symbols(path, path_str))
        elif path.suffix == ".rs":
            rows.extend(rust_symbols(path, path_str))
    for idx, row in enumerate(rows, 1):
        row["symbol_id"] = f"SYM-{idx:06d}"
    return rows


def build_dependency_edges(symbols: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    name_to_symbols: defaultdict[str, list[str]] = defaultdict(list)
    for sym in symbols:
        name_to_symbols[sym["symbol_name"]].append(sym["fully_qualified_name"])
    edges = []
    for sym in symbols:
        caller = sym["fully_qualified_name"]
        calls = sym.get("calls_detected") or []
        imports = sym.get("imports_used") or []
        for imp in imports:
            edges.append(edge_row(caller, imp, "import", sym["file_path"], 0.75, sym["owning_domain"], ""))
        for call in calls:
            base = call.split(".")[-1]
            matches = name_to_symbols.get(base, [])
            if matches:
                for callee in matches[:5]:
                    if callee != caller:
                        edges.append(edge_row(caller, callee, "function_call", sym["file_path"], 0.65, sym["owning_domain"], callee))
            else:
                edges.append(edge_row(caller, call, "unresolved_call", sym["file_path"], 0.35, sym["owning_domain"], call))
    for artifact in artifacts:
        producer = artifact.get("generated_by_script")
        consumer = artifact.get("consumed_by_script")
        if producer and consumer:
            edges.append(edge_row(producer, consumer, "artifact_lineage_cli_boundary", artifact["path"], 0.55, artifact["owning_domain"], consumer))
    for idx, row in enumerate(edges, 1):
        row["edge_id"] = f"EDGE-{idx:07d}"
    return edges


def edge_row(caller: str, callee: str, edge_type: str, file_path: str, confidence: float, caller_domain: str, callee_hint: str) -> dict[str, Any]:
    callee_domain = infer_domain(callee)
    flags = []
    lower = f"{caller} {callee} {file_path}".lower()
    if edge_type == "unresolved_call":
        flags.append("UNDECLARED_DEPENDENCY")
    if caller_domain != callee_domain and callee_domain != "raw_observatory_data":
        domain_crossing = True
    else:
        domain_crossing = False
    if "reward" in lower and "spectral_reward_manager" not in lower:
        flags.append("BYPASSES_SPECTRAL_MANAGER")
    if "c6" in lower and "perron" not in lower and "c6_7" not in lower:
        flags.append("BYPASSES_C6_7_GUARD")
    if "tile" in lower and "captured" not in lower:
        flags.append("BYPASSES_CAPTURED_TILE_RUNTIME")
    if any(k in lower for k in ["noncanonical", "legacy", "fallback"]):
        flags.append("NONCANONICAL_CALL_PATH")
    return {
        "caller_symbol": caller,
        "callee_symbol": callee,
        "edge_type": edge_type,
        "file_path": file_path,
        "confidence": confidence,
        "domain_crossing": domain_crossing,
        "canonical_allowed": not flags,
        "risk_flags": sorted(set(flags)),
    }


def duplicate_conflicts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in artifacts:
        if row.get("duplicate_group_id"):
            groups[row["duplicate_group_id"]].append(row)
    semantic_targets = [
        "W_dir",
        "W_spec",
        "transition_operator",
        "C4_report",
        "chi_membership",
        "DTSG",
        "C6_reward",
        "C6_7_verdict",
        "tile_registry",
        "captured_graph_manifest",
        "reward_diagnostic",
        "training_metric",
        "report",
        "release_package",
    ]
    for target in semantic_targets:
        matches = [a for a in artifacts if target.lower() in f"{a['semantic_role']} {a['path']}".lower()]
        if len(matches) > 1:
            groups[f"SEM-{target}"].extend(matches)
    rows = []
    for group_id, items in groups.items():
        statuses = Counter(i.get("canonical_status", "") for i in items)
        if statuses.get("CANONICAL_CONFORMANT", 0) > 1:
            classification = "BLOCKED_CANONICAL_AMBIGUITY"
        elif group_id.startswith("DUP-SHA"):
            classification = "BACKUP"
        elif statuses.get("PRE_SPEC") or statuses.get("STALE_UNDER_V1"):
            classification = "STALE"
        elif statuses.get("BLOCKED"):
            classification = "NONCANONICAL"
        else:
            classification = "CONFLICTING" if len({i["sha256"] for i in items}) > 1 else "CANONICAL"
        rows.append(
            {
                "duplicate_group_id": group_id,
                "classification": classification,
                "artifact_count": len(items),
                "paths": [i["path"] for i in items[:50]],
                "sha256_count": len({i["sha256"] for i in items}),
                "canonical_candidate_count": statuses.get("CANONICAL_CONFORMANT", 0),
                "notes": "Path list truncated to 50 entries" if len(items) > 50 else "",
            }
        )
    return rows


def evidence_paths(artifacts: list[dict[str, Any]], *terms: str, limit: int = 25) -> list[str]:
    out = []
    for row in artifacts:
        lower = row["path"].lower()
        if all(term.lower() in lower for term in terms):
            out.append(row["path"])
            if len(out) >= limit:
                break
    return out


def evidence_any(artifacts: list[dict[str, Any]], terms: list[str], limit: int = 25) -> list[str]:
    out = []
    for row in artifacts:
        lower = row["path"].lower()
        if any(term.lower() in lower for term in terms):
            out.append(row["path"])
            if len(out) >= limit:
                break
    return out


def formal_conformance(artifacts: list[dict[str, Any]], symbols: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    symbol_text = " ".join(f"{s['file_path']} {s['symbol_name']} {s.get('canonical_spec_mapping','')}" for s in symbols).lower()
    rows = []
    section_terms = {
        "Section I non-overclaim boundary": ["overclaim", "claim", "caveat"],
        "Section II provenance tiers": ["provenance", "tier", "l5", "l3"],
        "Axiom 8.5 estimator": ["axiom", "8.5", "estimator"],
        "C0 transfer operator": ["w_spec", "w_dir", "transfer", "transition"],
        "C1 metastable state extraction": ["c1", "metastable", "chi", "membership"],
        "C2 chronology/eigenvalue decay": ["chronology", "eigenvalue", "decay"],
        "C3 bisimulation/lumpability": ["bisimulation", "lumpability"],
        "C4 timescale convergence": ["c4", "timescale", "markov", "convergence"],
        "C5 calibration/memory kernel": ["calibration", "memory_kernel", "kernel"],
        "C6 restricted Dirichlet survival": ["c6", "dirichlet", "survival", "reward"],
        "C6.7 Perron robustness": ["c6.7", "c6_7", "perron", "robust"],
        "retained federated operators": ["federated", "retained"],
        "identity caveat": ["identity", "caveat"],
        "open physical blockers": ["blocker", "physical"],
    }
    no_spec = spec.get("canonical_status") == "BLOCKED_NO_CANONICAL_SPEC"
    for section, terms in section_terms.items():
        paths = evidence_any(artifacts, terms)
        sym_hit = any(t in symbol_text for t in terms)
        if no_spec:
            status = "BLOCKED_BY_DATA"
        elif paths and sym_hit:
            status = "IMPLEMENTED_CANONICAL"
        elif paths or sym_hit:
            status = "PARTIAL"
        else:
            status = "MISSING"
        rows.append(
            {
                "formal_section": section,
                "status": status,
                "artifact_evidence": paths,
                "symbol_evidence_count": sum(1 for s in symbols if any(t in f"{s['file_path']} {s['symbol_name']}".lower() for t in terms)),
                "notes": "Canonical v1 spec missing; implementation evidence cannot be marked canonical" if no_spec else "",
            }
        )
    return rows


def provenance_audit(artifacts: list[dict[str, Any]], claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for art in artifacts:
        flags = []
        lower = art["path"].lower()
        tier = art["provenance_tier"]
        if any(k in lower for k in ["w_dir", "w_spec", "reward", "survival", "dissipation"]) and tier != "L3_DERIVED":
            flags.append("CLAIM_TIER_OVERREACH")
        if tier == "UNKNOWN_TIER":
            flags.append("UNKNOWN_TIER")
        if any(k in lower for k in ["source_of_truth", "canonical"]) and art.get("git_status") == "UNTRACKED":
            flags.append("FORGED_SOURCE_OF_TRUTH")
        rows.append(
            {
                "path": art["path"],
                "assigned_tier": tier,
                "rule_basis": "W-derived quantities forced L3_DERIVED" if any(k in lower for k in ["w_dir", "w_spec", "reward", "survival", "dissipation"]) else "path_role_heuristic",
                "risk_flags": flags,
                "hash": art["sha256"],
                "notes": "",
            }
        )
    for claim in claims:
        if any(f in claim.get("risk_flags", []) for f in ["EXACT_THERMODYNAMIC_CLAIM", "EXACT_BIOLOGICAL_CAUSALITY", "CLINICAL_EFFECT_CLAIM"]):
            rows.append(
                {
                    "path": claim["path"],
                    "assigned_tier": "UNKNOWN_TIER",
                    "rule_basis": "claim_text_requires_source_artifact_review",
                    "risk_flags": ["CLAIM_TIER_OVERREACH"],
                    "hash": "",
                    "notes": f"claim_line={claim['line']}",
                }
            )
    return rows


def lineage_chains(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chains = [
        ("spike events", "active sets", ["spike_events"], ["active_set", "active_sets"]),
        ("active sets", "transition counts", ["active_set", "chi", "membership"], ["transition", "count"]),
        ("transition counts", "W_dir/W_spec", ["transition"], ["w_dir", "w_spec"]),
        ("W_spec", "C4", ["w_spec"], ["c4", "timescale", "markov"]),
        ("W_spec", "C1 chi", ["w_spec"], ["chi", "membership", "metastable"]),
        ("W_dir + chi", "DTSG", ["w_dir", "chi"], ["dtsg"]),
        ("W_spec + basin mapping", "C6 restricted operator", ["w_spec", "basin"], ["c6", "operator"]),
        ("C6 operator", "Dirichlet eigensystem", ["c6", "operator"], ["dirichlet", "eigen"]),
        ("Dirichlet eigensystem", "C6 reward", ["dirichlet", "eigen"], ["c6", "reward"]),
        ("C6 reward", "spectral reward manager", ["c6", "reward"], ["spectral_reward_manager"]),
        ("captured graph tile", "BSR update", ["captured", "tile"], ["bsr"]),
        ("BSR update", "C6 reward solve/cache", ["bsr"], ["c6", "reward", "cache"]),
        ("C6 reward", "Log-SubTB", ["c6", "reward"], ["subtb", "gflownet"]),
        ("trajectory", "tile credit", ["trajectory"], ["tile_credit"]),
        ("tile credit", "motif index", ["tile_credit"], ["motif"]),
        ("chemical tile", "molecule candidate", ["chemical_tile", "tile_registry", "fragment"], ["molecule", "candidate", "smiles"]),
        ("variant operator", "variant durability", ["variant", "operator"], ["variant", "durability"]),
        ("active acquisition", "next generation/training batch", ["acquisition", "uncertainty"], ["next", "generation", "training"]),
        ("training run", "report/release/manifest", ["training", "subtb", "gflownet"], ["report", "release", "manifest"]),
    ]
    rows = []
    for idx, (src, dst, src_terms, dst_terms) in enumerate(chains, 1):
        src_ev = evidence_any(artifacts, src_terms, limit=12)
        dst_ev = evidence_any(artifacts, dst_terms, limit=12)
        if src_ev and dst_ev:
            status = "LINEAGE_COMPLETE"
        elif src_ev or dst_ev:
            status = "LINEAGE_PARTIAL"
        else:
            status = "LINEAGE_BROKEN"
        if any(t in {"w_spec", "w_dir", "c6"} for t in src_terms + dst_terms) and not evidence_any(artifacts, ["w_spec", "w_hat"], limit=1):
            if status == "LINEAGE_COMPLETE":
                status = "LINEAGE_NON_CANONICAL"
        rows.append(
            {
                "chain_id": f"LINEAGE-{idx:02d}",
                "source": src,
                "target": dst,
                "status": status,
                "source_evidence": src_ev,
                "target_evidence": dst_ev,
                "blocker_notes": "" if status == "LINEAGE_COMPLETE" else "Missing source or target artifact evidence in requested scope",
            }
        )
    return rows


def active_learning_audit(artifacts: list[dict[str, Any]], symbols: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    checks = [
        ("acquisition score exists", ["acquisition", "bald", "score"]),
        ("uncertainty signal exists", ["uncertainty", "variance", "entropy", "bald"]),
        ("selected candidates logged", ["selected", "candidate"]),
        ("selected candidates modify next batch", ["next", "batch", "feedback"]),
        ("next training/generation run consumes selected candidates", ["training", "generation", "selected"]),
        ("feedback manifest exists", ["feedback", "manifest"]),
        ("rejection/negative evidence preserved", ["rejection", "negative"]),
    ]
    rows = []
    pass_count = 0
    for check, terms in checks:
        art_ev = evidence_any(artifacts, terms, limit=15)
        sym_ev = [s["fully_qualified_name"] for s in symbols if any(t in f"{s['file_path']} {s['symbol_name']}".lower() for t in terms)][:15]
        status = "PASS" if art_ev or sym_ev else "MISSING"
        pass_count += status == "PASS"
        rows.append(
            {
                "check": check,
                "status": status,
                "artifact_evidence": art_ev,
                "symbol_evidence": sym_ev,
                "notes": "",
            }
        )
    if pass_count >= 6:
        overall = "ACTIVE_LEARNING_OPERATIONAL"
    elif pass_count >= 2 and not any(row["check"] == "selected candidates modify next batch" and row["status"] == "PASS" for row in rows):
        overall = "ACTIVE_LEARNING_SCORING_ONLY"
    elif pass_count >= 1:
        overall = "ACTIVE_LEARNING_PROJECTED_ONLY"
    else:
        overall = "BLOCKED_NO_UNCERTAINTY_SIGNAL"
    if not any(row["check"] == "selected candidates modify next batch" and row["status"] == "PASS" for row in rows):
        overall = "BLOCKED_NO_FEEDBACK_LOOP" if pass_count < 2 else "ACTIVE_LEARNING_SCORING_ONLY"
    return rows, overall


def scan_claims(audit_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_scope_files(audit_root):
        if path.suffix.lower() not in {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml"}:
            continue
        if path.stat().st_size > 25_000_000:
            continue
        path_str = rel(path)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line_no, line in enumerate(fh, 1):
                    cats = [name for name, rx in CLAIM_PATTERNS.items() if rx.search(line)]
                    if not cats:
                        continue
                    flags = claim_risk_flags(line, cats)
                    rows.append(
                        {
                            "path": path_str,
                            "line": line_no,
                            "claim_categories": cats,
                            "claim_excerpt": redact_text(line.strip())[:260],
                            "risk_flags": flags,
                            "provenance_status": "NEEDS_L5_REVIEW" if flags else "UNMAPPED",
                        }
                    )
        except Exception as exc:
            rows.append(
                {
                    "path": path_str,
                    "line": 0,
                    "claim_categories": ["scan_error"],
                    "claim_excerpt": "",
                    "risk_flags": [f"SCAN_ERROR:{str(exc)[:120]}"],
                    "provenance_status": "SCAN_ERROR",
                }
            )
    return rows


def claim_risk_flags(line: str, categories: list[str]) -> list[str]:
    lower = line.lower()
    flags = []
    if "exact" in lower and any(cat in categories for cat in ["entropy", "free_energy"]):
        flags.append("EXACT_THERMODYNAMIC_CLAIM")
    if any(word in lower for word in ["causes", "causal", "drives"]) and "biological_effect" in categories:
        flags.append("EXACT_BIOLOGICAL_CAUSALITY")
    if "clinical" in categories or any(word in lower for word in ["patient", "clinical", "therapeutic"]):
        flags.append("CLINICAL_EFFECT_CLAIM")
    if any(word in lower for word in ["t1", "type 1"]) and "l5" not in lower:
        flags.append("T1_WITHOUT_L5_EDGES")
    if "c6" in lower and "c4" not in lower:
        flags.append("C6_WITHOUT_C4_PASS")
    if "molecule" in lower and "chemical tile" not in lower:
        flags.append("MOLECULE_DESIGN_WITHOUT_CHEMICAL_TILE_REGISTRY")
    if "variant" in lower and "operator" not in lower:
        flags.append("VARIANT_DURABILITY_WITHOUT_VARIANT_OPERATOR_CONTEXT")
    if "full dstw" in lower and "w_hat" not in lower and "w-hat" not in lower:
        flags.append("FULL_DSTW_WITHOUT_W_HAT")
    return sorted(set(flags))


def redact_text(text: str) -> str:
    out = text
    for rx in SECRET_PATTERNS.values():
        out = rx.sub("[REDACTED_SECRET]", out)
    return out


def scan_ip_secrets(audit_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_scope_files(audit_root):
        if not is_textual(path):
            continue
        if path.stat().st_size > 25_000_000:
            continue
        path_str = rel(path)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line_no, line in enumerate(fh, 1):
                    for name, rx in SECRET_PATTERNS.items():
                        for match in rx.finditer(line):
                            secret_value = match.group(0)
                            if name == "GENERIC_SECRET_ASSIGNMENT" and match.lastindex and match.lastindex >= 2:
                                secret_value = match.group(2)
                            rows.append(
                                {
                                    "path": path_str,
                                    "line": line_no,
                                    "finding_type": name,
                                    "risk_flag": "CREDENTIAL_EXPOSED" if name != "PRIVATE_KEY" else "SECRET_EXPOSED",
                                    "redacted_fingerprint": redacted_fingerprint(secret_value),
                                    "notes": "Secret value intentionally omitted",
                                }
                            )
                    for name, rx in IP_PATTERNS.items():
                        if rx.search(line):
                            rows.append(
                                {
                                    "path": path_str,
                                    "line": line_no,
                                    "finding_type": name,
                                    "risk_flag": name,
                                    "redacted_fingerprint": redacted_fingerprint(line.strip()),
                                    "notes": "Line text intentionally omitted",
                                }
                            )
        except Exception as exc:
            rows.append(
                {
                    "path": path_str,
                    "line": 0,
                    "finding_type": "SCAN_ERROR",
                    "risk_flag": "SCAN_ERROR",
                    "redacted_fingerprint": "",
                    "notes": str(exc)[:200],
                }
            )
    return rows


def runtime_consistency(artifacts: list[dict[str, Any]], symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = [
        ("required manifests exist", ["manifest"]),
        ("hashes exist", ["sha256", "hash"]),
        ("statuses valid", ["status"]),
        ("no CPU fallback in production captured-tile run", ["cpu", "fallback"]),
        ("no uncaptured tile fallback", ["uncaptured", "fallback"]),
        ("reward source canonical", ["reward", "canonical"]),
        ("C6.7 verdict distribution present", ["c6.7", "c6_7", "perron", "verdict"]),
        ("tile credit exists", ["tile_credit", "credit_assignment"]),
        ("cache diagnostics present", ["cache", "diagnostic"]),
        ("operator owner consistent", ["operator", "owner"]),
        ("exact command provenance exists", ["command", "provenance"]),
    ]
    rows = []
    text = " ".join(a["path"] for a in artifacts).lower()
    symbol_text = " ".join(f"{s['file_path']} {s['symbol_name']} {' '.join(s.get('risk_flags', []))}" for s in symbols).lower()
    for check, terms in checks:
        ev = evidence_any(artifacts, terms, limit=20)
        flags = []
        status = "PASS" if ev else "MISSING"
        if "cpu fallback" in check and ("cpu_fallback_path" in symbol_text or ("cpu" in text and "fallback" in text)):
            status = "FAIL"
            flags.append("RUNTIME_CPU_FALLBACK")
        if "uncaptured tile fallback" in check and "uncaptured" in text and "fallback" in text:
            status = "FAIL"
            flags.append("RUNTIME_UNCAPTURED_TILE_FALLBACK")
        if "manifests" in check and not ev:
            flags.append("RUNTIME_MISSING_MANIFEST")
        if "hashes" in check and not ev:
            flags.append("RUNTIME_HASH_MISMATCH")
        if "reward source" in check and not ev:
            flags.append("RUNTIME_REWARD_SOURCE_NONCANONICAL")
        if "command provenance" in check and not ev:
            flags.append("RUNTIME_COMMAND_PROVENANCE_MISSING")
        rows.append({"check": check, "status": status, "evidence": ev, "risk_flags": flags})
    return rows


def molecule_design_audit(artifacts: list[dict[str, Any]], symbols: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    checks = [
        ("chemical tile registry", ["chemical_tile", "tile_registry"]),
        ("tile_id -> SMILES/molecular graph", ["tile_id", "smiles", "molecular_graph"]),
        ("attachment atoms", ["attachment", "atom"]),
        ("valency rules", ["valency", "valence"]),
        ("stereochemistry rules", ["stereo", "stereochemistry"]),
        ("RDKit sanitization", ["rdkit", "sanitize"]),
        ("canonical SMILES", ["canonical_smiles", "canonical smiles"]),
        ("duplicate detection", ["duplicate"]),
        ("fragment provenance", ["fragment", "provenance"]),
        ("PAINS/basic medchem filters", ["pains", "medchem"]),
        ("synthetic feasibility status", ["synthetic", "feasibility"]),
        ("tile-to-operator-delta mapping", ["tile", "operator", "delta"]),
        ("tile-to-C6 reward effect", ["tile", "c6", "reward"]),
        ("emitted molecule candidates", ["molecule", "candidate", "smiles"]),
    ]
    rows = []
    pass_count = 0
    for check, terms in checks:
        ev = evidence_any(artifacts, terms, limit=15)
        sym_ev = [s["fully_qualified_name"] for s in symbols if any(t in f"{s['file_path']} {s['symbol_name']}".lower() for t in terms)][:15]
        status = "PASS" if ev or sym_ev else "MISSING"
        pass_count += status == "PASS"
        rows.append({"check": check, "status": status, "artifact_evidence": ev, "symbol_evidence": sym_ev})
    registry_ok = rows[0]["status"] == "PASS"
    validity_ok = all(row["status"] == "PASS" for row in rows[2:8])
    mapping_ok = rows[11]["status"] == "PASS" and rows[12]["status"] == "PASS"
    if registry_ok and validity_ok and mapping_ok and pass_count >= 12:
        overall = "MOLECULE_DESIGN_READY"
    elif not registry_ok:
        if any(row["status"] == "PASS" for row in rows if "SMILES" in row["check"] or "fragment" in row["check"]):
            overall = "MOTIF_ONLY_READY"
        else:
            overall = "BLOCKED_NO_CHEMICAL_TILE_REGISTRY"
    elif not validity_ok:
        overall = "BLOCKED_NO_VALIDITY_FILTERS"
    elif not mapping_ok:
        overall = "BLOCKED_NO_TILE_TO_OPERATOR_MAPPING"
    else:
        overall = "MOTIF_ONLY_READY"
    return rows, overall


def variant_durability_audit(artifacts: list[dict[str, Any]], symbols: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    checks = [
        ("variant panel", ["variant_panel", "variant"]),
        ("genealogical grouping", ["genealogical"]),
        ("topology-region grouping", ["topology", "region"]),
        ("perturbation-family grouping", ["perturbation", "family"]),
        ("variant-specific operator context", ["variant", "operator"]),
        ("variant-specific C6 reward", ["variant", "c6", "reward"]),
        ("WT-vs-variant comparison", ["wt", "variant", "comparison"]),
        ("variant uncertainty", ["variant", "uncertainty"]),
        ("acquisition uses variant uncertainty", ["acquisition", "variant", "uncertainty"]),
    ]
    rows = []
    pass_count = 0
    for check, terms in checks:
        ev = evidence_any(artifacts, terms, limit=15)
        sym_ev = [s["fully_qualified_name"] for s in symbols if any(t in f"{s['file_path']} {s['symbol_name']}".lower() for t in terms)][:15]
        status = "PASS" if ev or sym_ev else "MISSING"
        pass_count += status == "PASS"
        rows.append({"check": check, "status": status, "artifact_evidence": ev, "symbol_evidence": sym_ev})
    if rows[0]["status"] != "PASS":
        overall = "BLOCKED_NO_GENEALOGICAL_PANEL"
    elif rows[4]["status"] != "PASS":
        overall = "BLOCKED_NO_VARIANT_OPERATOR"
    elif rows[5]["status"] != "PASS":
        overall = "BLOCKED_NO_VARIANT_C6"
    elif pass_count >= 8:
        overall = "VARIANT_DURABILITY_OPERATIONAL"
    elif pass_count >= 4:
        overall = "VARIANT_DURABILITY_PARTIAL"
    else:
        overall = "PROJECTED_ONLY"
    return rows, overall


def test_gate_audit(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = [
        ("pytest", [".py", "test_"]),
        ("mypy", ["mypy", "pyproject.toml", "mypy.ini"]),
        ("rust tests", [".rs", "tests/"]),
        ("clippy", ["clippy", "cargo"]),
        ("regression gates", ["regression", "gate"]),
        ("subagent reports", ["subagent", "bug_hunter", ".audit-reports"]),
    ]
    rows = []
    for name, terms in checks:
        ev = []
        for art in artifacts:
            lower = art["path"].lower()
            if all(t in lower for t in terms):
                ev.append(art["path"])
            elif name == "mypy" and any(t in lower for t in terms):
                ev.append(art["path"])
            elif name in {"clippy", "regression gates", "subagent reports"} and any(t in lower for t in terms):
                ev.append(art["path"])
            if len(ev) >= 20:
                break
        status = "STALE" if ev else "MISSING"
        rows.append(
            {
                "gate": name,
                "status": status,
                "evidence": ev,
                "notes": "Inventory-only audit; gates not executed to avoid mutating build/test artifacts on 99% full root filesystem" if ev else "",
            }
        )
    return rows


def build_blockers(
    spec: dict[str, Any],
    artifacts: list[dict[str, Any]],
    conformance: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    active_learning_status: str,
    molecule_status: str,
    variant_status: str,
    ip_secret_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(code: str, severity: str, evidence: list[str], notes: str) -> None:
        rows.append(
            {
                "blocker_id": f"BLOCKER-{len(rows)+1:04d}",
                "code": code,
                "severity": severity,
                "evidence": evidence[:25],
                "notes": notes,
            }
        )

    if spec.get("canonical_status") == "BLOCKED_NO_CANONICAL_SPEC":
        add("BLOCKED_NO_CANONICAL_SPEC", "CRITICAL", [], "No exact DSTW v1 canonical spec file or content phrase found")
    if any(r.get("classification") == "BLOCKED_CANONICAL_AMBIGUITY" for r in duplicate_rows):
        add("BLOCKED_CANONICAL_AMBIGUITY", "CRITICAL", [r["duplicate_group_id"] for r in duplicate_rows if r.get("classification") == "BLOCKED_CANONICAL_AMBIGUITY"], "Multiple canonical candidates detected")
    if not evidence_any(artifacts, ["w_hat", "w-hat", "w_spec"], limit=5):
        add("MISSING_W_HAT", "CRITICAL", [], "No W_hat/W_spec evidence found in requested scope")
    c4 = next((r for r in conformance if r["formal_section"].startswith("C4")), None)
    if c4 and c4["status"] not in {"IMPLEMENTED_CANONICAL"}:
        add("C4_NOT_PROVEN", "CRITICAL", c4.get("artifact_evidence", []), f"C4 status={c4['status']}")
    c1 = next((r for r in conformance if r["formal_section"].startswith("C1")), None)
    if c1 and c1["status"] not in {"IMPLEMENTED_CANONICAL"}:
        add("C1_NOT_PROVEN", "HIGH", c1.get("artifact_evidence", []), f"C1 status={c1['status']}")
    c6 = next((r for r in conformance if r["formal_section"].startswith("C6 restricted")), None)
    if c6 and c6["status"] not in {"IMPLEMENTED_CANONICAL"}:
        add("C6_NOT_CANONICAL", "CRITICAL", c6.get("artifact_evidence", []), f"C6 status={c6['status']}")
    if not evidence_any(artifacts, ["transition_edge", "edge_transition", "edge"], limit=5):
        add("NO_TRANSITION_EDGE_L5", "HIGH", [], "No direct transition-edge L5 artifact evidence found")
    if molecule_status == "BLOCKED_NO_CHEMICAL_TILE_REGISTRY":
        add("NO_CHEMICAL_TILE_REGISTRY", "HIGH", [], molecule_status)
    if variant_status in {"BLOCKED_NO_VARIANT_OPERATOR", "BLOCKED_NO_VARIANT_C6"}:
        add("NO_VARIANT_OPERATOR_CONTEXT", "HIGH", [], variant_status)
    overreach = [c for c in claim_rows if c.get("risk_flags")]
    if overreach:
        add("CLAIM_OVERREACH", "HIGH", [f"{c['path']}:{c['line']}" for c in overreach[:25]], f"{len(overreach)} claim risk flags")
    broken_lineage = [l for l in lineage if l["status"] not in {"LINEAGE_COMPLETE"}]
    if broken_lineage:
        add("PROVENANCE_BREAK", "HIGH", [l["chain_id"] for l in broken_lineage[:25]], f"{len(broken_lineage)} incomplete lineage chains")
    schema_missing = [s for s in schema_rows if str(s.get("required_columns_status", "")).startswith("MISSING")]
    if schema_missing:
        add("SCHEMA_MISSING", "MEDIUM", [s["path"] for s in schema_missing[:25]], f"{len(schema_missing)} schema gates missing required columns")
    runtime_fail = [r for r in runtime_rows if r["status"] in {"FAIL", "MISSING"}]
    if runtime_fail:
        add("RUNTIME_NONCANONICAL", "HIGH", [r["check"] for r in runtime_fail[:25]], f"{len(runtime_fail)} runtime checks failed or missing")
    secret_findings = [r for r in ip_secret_rows if r.get("risk_flag") in {"SECRET_EXPOSED", "CREDENTIAL_EXPOSED"}]
    if secret_findings:
        add("SECRET_EXPOSED", "CRITICAL", [f"{r['path']}:{r['line']}" for r in secret_findings[:25]], f"{len(secret_findings)} credential/secret findings")
    ip_findings = [r for r in ip_secret_rows if r.get("risk_flag") in {"IP_OVERDISCURE", "PATENT_RISK_DISCLOSURE", "TRADE_SECRET_LEAK", "CLIENT_SENSITIVE_DATA"}]
    if ip_findings:
        add("IP_OVERDISCLOSURE", "HIGH", [f"{r['path']}:{r['line']}" for r in ip_findings[:25]], f"{len(ip_findings)} IP-sensitive findings")
    if active_learning_status in {"BLOCKED_NO_FEEDBACK_LOOP", "ACTIVE_LEARNING_SCORING_ONLY", "ACTIVE_LEARNING_PROJECTED_ONLY", "BLOCKED_NO_UNCERTAINTY_SIGNAL"}:
        add("ACTIVE_LEARNING_NO_FEEDBACK", "MEDIUM", [], active_learning_status)
    return rows


def global_status(spec: dict[str, Any], blockers: list[dict[str, Any]], molecule_status: str, variant_status: str, active_learning_status: str) -> str:
    if spec.get("canonical_status") == "BLOCKED_NO_CANONICAL_SPEC":
        return "E2E_BLOCKED"
    codes = {b["code"] for b in blockers}
    if "SECRET_EXPOSED" in codes:
        return "E2E_PARTIAL_WITH_BLOCKERS"
    if molecule_status != "MOLECULE_DESIGN_READY":
        return "E2E_MOTIF_READY_NOT_MOLECULE_READY" if molecule_status == "MOTIF_ONLY_READY" else "E2E_PARTIAL_WITH_BLOCKERS"
    if variant_status != "VARIANT_DURABILITY_OPERATIONAL" or active_learning_status != "ACTIVE_LEARNING_OPERATIONAL":
        return "E2E_PARTIAL_WITH_BLOCKERS"
    if not {"MISSING_W_HAT", "C4_NOT_PROVEN"} & codes:
        return "E2E_CANONICAL_OPERATIONAL"
    return "E2E_OPERATIONAL_WITH_L3_LIMITS"


def make_graphs(audit_root: Path, artifacts: list[dict[str, Any]], symbols: list[dict[str, Any]], edges: list[dict[str, Any]], lineage: list[dict[str, Any]], blockers: list[dict[str, Any]]) -> tuple[int, int]:
    if nx is None:
        write_dot(audit_root / "e2e_ontology_graph.dot", [], [])
        for name in [
            "dependency_graph.graphml",
            "e2e_ontology_graph.graphml",
            "dependency_graph.svg",
            "e2e_ontology_graph.svg",
            "e2e_pipeline_lineage.svg",
            "blocker_dependency_graph.svg",
            "code_dependency_graph.svg",
        ]:
            (audit_root / name).write_text("graph export unavailable: networkx missing\n", encoding="utf-8")
        return 0, 0
    g = nx.DiGraph()
    for domain in DOMAINS:
        g.add_node(domain, node_type="domain")
    for art in artifacts:
        g.add_node(art["path"], node_type="artifact", domain=art["owning_domain"])
        g.add_edge(art["owning_domain"], art["path"], edge_type="owns")
    for sym in symbols:
        g.add_node(sym["fully_qualified_name"], node_type="symbol", domain=sym["owning_domain"])
        g.add_edge(sym["file_path"], sym["fully_qualified_name"], edge_type="defines")
    for edge in edges[:20000]:
        g.add_node(edge["caller_symbol"], node_type="symbol_or_external")
        g.add_node(edge["callee_symbol"], node_type="symbol_or_external")
        g.add_edge(edge["caller_symbol"], edge["callee_symbol"], edge_type=edge["edge_type"])
    for b in blockers:
        g.add_node(b["code"], node_type="blocker", severity=b["severity"])
    nx.write_graphml(g, audit_root / "e2e_ontology_graph.graphml")
    nx.write_graphml(g, audit_root / "dependency_graph.graphml")
    write_dot(audit_root / "e2e_ontology_graph.dot", list(g.nodes(data=True)), list(g.edges(data=True)))
    draw_graph(g, audit_root / "e2e_ontology_graph.svg", max_nodes=350)
    draw_graph(g, audit_root / "dependency_graph.svg", max_nodes=350)
    draw_graph(g, audit_root / "code_dependency_graph.svg", max_nodes=350)
    lg = nx.DiGraph()
    for row in lineage:
        lg.add_edge(row["source"], row["target"], status=row["status"])
    draw_graph(lg, audit_root / "e2e_pipeline_lineage.svg", max_nodes=80)
    bg = nx.DiGraph()
    for b in blockers:
        bg.add_node(b["code"], severity=b["severity"])
        for ev in b.get("evidence", [])[:10]:
            bg.add_node(ev, node_type="evidence")
            bg.add_edge(b["code"], ev)
    draw_graph(bg, audit_root / "blocker_dependency_graph.svg", max_nodes=150)
    return g.number_of_nodes(), g.number_of_edges()


def write_dot(path: Path, nodes: list[Any], edges: list[Any]) -> None:
    lines = ["digraph ontology {"]
    for node, attrs in nodes[:20000]:
        label = str(node).replace('"', '\\"')
        lines.append(f'  "{label}";')
    for u, v, attrs in edges[:50000]:
        uu = str(u).replace('"', '\\"')
        vv = str(v).replace('"', '\\"')
        label = str(attrs.get("edge_type", "")).replace('"', '\\"') if isinstance(attrs, dict) else ""
        lines.append(f'  "{uu}" -> "{vv}" [label="{label}"];')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_graph(g: Any, path: Path, max_nodes: int = 250) -> None:
    if plt is None or nx is None:
        path.write_text("svg export unavailable: matplotlib/networkx missing\n", encoding="utf-8")
        return
    if g.number_of_nodes() == 0:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return
    nodes = list(g.nodes())[:max_nodes]
    h = g.subgraph(nodes).copy()
    plt.figure(figsize=(18, 14))
    pos = nx.spring_layout(h, seed=42, k=None, iterations=40)
    colors = []
    for n in h.nodes:
        nt = h.nodes[n].get("node_type", "")
        severity = h.nodes[n].get("severity", "")
        if severity == "CRITICAL":
            colors.append("#cc3344")
        elif severity == "HIGH":
            colors.append("#ee8844")
        elif nt == "domain":
            colors.append("#4477aa")
        elif nt == "artifact":
            colors.append("#66aa77")
        elif nt == "symbol":
            colors.append("#aa77cc")
        else:
            colors.append("#999999")
    nx.draw_networkx_nodes(h, pos, node_size=70, node_color=colors, alpha=0.85)
    nx.draw_networkx_edges(h, pos, arrows=False, width=0.35, alpha=0.35)
    labels = {n: shorten_label(str(n), 42) for n in list(h.nodes())[:90]}
    nx.draw_networkx_labels(h, pos, labels=labels, font_size=6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, format="svg")
    plt.close()


def shorten_label(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "..." + text[-(limit - 3) :]


def sqlite_write(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        for name, rows in tables.items():
            columns = sorted({k for row in rows for k in row.keys()}) or ["empty"]
            col_defs = ", ".join(f'"{c}" TEXT' for c in columns)
            conn.execute(f'CREATE TABLE "{name}" ({col_defs})')
            if rows:
                placeholders = ", ".join(["?"] * len(columns))
                col_names = ", ".join(f'"{c}"' for c in columns)
                conn.executemany(
                    f'INSERT INTO "{name}" ({col_names}) VALUES ({placeholders})',
                    [[sqlite_value(row.get(c)) for c in columns] for row in rows],
                )
        conn.commit()
    finally:
        conn.close()


def ontology_sqlite_rows(
    artifacts: list[dict[str, Any]],
    symbols: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_map: dict[str, dict[str, Any]] = {}

    def node(node_id: str, node_type: str, **attrs: Any) -> None:
        if node_id not in node_map:
            node_map[node_id] = {"node_id": node_id, "node_type": node_type, **attrs}

    edge_rows: list[dict[str, Any]] = []

    for domain in DOMAINS:
        node(domain, "domain")
    for art in artifacts:
        node(art["path"], "artifact", domain=art["owning_domain"], canonical_status=art["canonical_status"])
        edge_rows.append({"source": art["owning_domain"], "target": art["path"], "edge_type": "owns", "confidence": "1.0"})
    for sym in symbols:
        node(sym["fully_qualified_name"], "symbol", domain=sym["owning_domain"], file_path=sym["file_path"])
        edge_rows.append({"source": sym["file_path"], "target": sym["fully_qualified_name"], "edge_type": "defines", "confidence": "1.0"})
    for dep in edges:
        node(dep["caller_symbol"], "symbol_or_external")
        node(dep["callee_symbol"], "symbol_or_external")
        edge_rows.append(
            {
                "source": dep["caller_symbol"],
                "target": dep["callee_symbol"],
                "edge_type": dep["edge_type"],
                "confidence": dep.get("confidence", ""),
                "canonical_allowed": dep.get("canonical_allowed", ""),
            }
        )
    for blocker in blockers:
        node(blocker["code"], "blocker", severity=blocker["severity"])
        for evidence in blocker.get("evidence", []):
            node(str(evidence), "evidence")
            edge_rows.append({"source": blocker["code"], "target": str(evidence), "edge_type": "blocked_by", "confidence": "1.0"})
    for idx, row in enumerate(edge_rows, 1):
        row["ontology_edge_id"] = f"ONTEDGE-{idx:07d}"
    return list(node_map.values()), edge_rows


def sqlite_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def write_final_report(
    path: Path,
    summary: dict[str, Any],
    spec: dict[str, Any],
    scope_rows: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    symbols: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    conformance: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    molecule_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    ip_rows: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
) -> None:
    domain_counts = Counter(a["owning_domain"] for a in artifacts)
    risk_counts = Counter(flag for s in symbols for flag in s.get("risk_flags", []))
    claim_flags = Counter(flag for c in claims for flag in c.get("risk_flags", []))
    ip_flags = Counter(r["risk_flag"] for r in ip_rows)
    lines = [
        "# PRISM-DSTW E2E Ontology Forensic Audit v2",
        "",
        "## 1. Executive verdict",
        "",
        f"- global_status: {summary['global_status']}",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- artifact_count: {summary['artifact_count']}",
        f"- symbol_count: {summary['symbol_count']}",
        f"- blocker_count: {summary['blocker_count']}",
        "",
        "## 2. Canonical spec discovery",
        "",
        fenced_json(spec),
        "",
        "## 3. Canonical pipeline map",
        "",
        summarize_rows(lineage, ["chain_id", "source", "target", "status"], 30),
        "",
        "## 4. File artifact inventory",
        "",
        summarize_counter(domain_counts, "domain", "artifact_count"),
        "",
        "## 5. Code symbol index summary",
        "",
        summarize_counter(Counter(s["symbol_type"] for s in symbols), "symbol_type", "count"),
        "",
        "Risk flags:",
        "",
        summarize_counter(risk_counts, "risk_flag", "count"),
        "",
        "## 6. Dependency graph summary",
        "",
        f"- dependency_edge_count: {len(edges)}",
        f"- ontology_node_count: {summary['ontology_node_count']}",
        f"- ontology_edge_count: {summary['ontology_edge_count']}",
        "",
        "## 7. Duplicate/conflict audit",
        "",
        summarize_rows(duplicate_rows, ["duplicate_group_id", "classification", "artifact_count", "sha256_count"], 30),
        "",
        "## 8. Formal spec conformance",
        "",
        summarize_rows(conformance, ["formal_section", "status", "symbol_evidence_count"], 30),
        "",
        "## 9. Provenance tier audit",
        "",
        summarize_counter(Counter(p["assigned_tier"] for p in provenance), "assigned_tier", "count"),
        "",
        "## 10. Lineage chains",
        "",
        summarize_rows(lineage, ["chain_id", "source", "target", "status"], 30),
        "",
        "## 11. Active learning loop audit",
        "",
        f"- status: {summary['active_learning_status']}",
        "",
        summarize_rows(active_rows, ["check", "status"], 20),
        "",
        "## 12. Runtime consistency audit",
        "",
        summarize_rows(runtime_rows, ["check", "status", "risk_flags"], 20),
        "",
        "## 13. Captured graph tile audit",
        "",
        f"- status: {summary['captured_tile_status']}",
        "",
        "## 14. Log-SubTB audit",
        "",
        f"- status: {summary['log_subtb_status']}",
        "",
        "## 15. Molecule design readiness",
        "",
        f"- status: {summary['molecule_design_status']}",
        "",
        summarize_rows(molecule_rows, ["check", "status"], 30),
        "",
        "## 16. Variant/genealogical durability",
        "",
        f"- status: {summary['variant_durability_status']}",
        "",
        summarize_rows(variant_rows, ["check", "status"], 30),
        "",
        "## 17. Claim-provenance audit",
        "",
        summarize_counter(claim_flags, "claim_risk_flag", "count"),
        "",
        "## 18. IP/secret exposure audit",
        "",
        summarize_counter(ip_flags, "risk_flag", "count"),
        "",
        "Secret/IP values are intentionally omitted; see redacted fingerprints in `ip_secret_exposure_audit.*`.",
        "",
        "## 19. Test/gate audit",
        "",
        summarize_rows(tests, ["gate", "status"], 20),
        "",
        "## 20. Blocker index",
        "",
        summarize_rows(blockers, ["blocker_id", "code", "severity", "notes"], 50),
        "",
        "## 21. Recommended next actions",
        "",
        "\n".join(f"{idx}. {item}" for idx, item in enumerate(summary["next_actions"], 1)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_counter(counter: Counter[str], name_col: str, count_col: str) -> str:
    if not counter:
        return "_none_\n"
    rows = [{name_col: k, count_col: v} for k, v in counter.most_common()]
    return summarize_rows(rows, [name_col, count_col], 100)


def summarize_rows(rows: list[dict[str, Any]], columns: list[str], limit: int) -> str:
    if not rows:
        return "_none_\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:limit]:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=True, sort_keys=True)
            text = str(val).replace("|", "\\|").replace("\n", " ")
            if len(text) > 160:
                text = text[:157] + "..."
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > limit:
        lines.append("")
        lines.append(f"Truncated to first {limit} rows.")
    return "\n".join(lines) + "\n"


def fenced_json(obj: Any) -> str:
    return "```json\n" + json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n```"


def status_from_evidence(artifacts: list[dict[str, Any]], terms: list[str], operational: str, missing: str, partial: str | None = None) -> str:
    ev = evidence_any(artifacts, terms, limit=10)
    if not ev:
        return missing
    if partial:
        return partial
    return operational


def write_all_outputs(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = (ROOT / args.audit_root).resolve()
    audit_root.mkdir(parents=True, exist_ok=True)
    tracked, status_map = get_git_maps()
    spec = find_canonical_spec(tracked, status_map)
    write_json(audit_root / "canonical_spec_discovery.json", spec)
    write_markdown_table(audit_root / "canonical_spec_discovery.md", "Canonical Spec Discovery", [spec], ["spec_path", "sha256", "git_status", "modified_time", "canonical_status", "candidate_count"])

    artifacts, schema_rows = inventory_artifacts(audit_root, spec, tracked, status_map)
    scope_rows = make_audit_scope(spec["canonical_status"], Counter(a["owning_domain"] for a in artifacts))
    write_json(audit_root / "audit_scope.json", scope_rows)
    write_markdown_table(audit_root / "audit_scope.md", "Audit Scope Lock", scope_rows, ["domain", "status", "artifact_count", "canonical_status", "notes"])

    write_json(audit_root / "artifact_inventory.json", artifacts)
    write_parquet(audit_root / "artifact_inventory.parquet", artifacts)
    write_markdown_table(audit_root / "artifact_inventory.md", "Artifact Inventory", artifacts, ["artifact_id", "path", "file_type", "size_bytes", "sha256", "git_status", "owning_domain", "semantic_role", "provenance_tier", "canonical_status", "duplicate_group_id"], limit=300)

    symbols = index_code_symbols(artifacts) if args.index_symbols else []
    write_json(audit_root / "code_symbol_index.json", symbols)
    write_parquet(audit_root / "code_symbol_index.parquet", symbols)

    edges = build_dependency_edges(symbols, artifacts) if args.build_dependency_graph else []
    write_parquet(audit_root / "dependency_graph_edges.parquet", edges)

    duplicates = duplicate_conflicts(artifacts)
    write_parquet(audit_root / "duplicate_conflict_audit.parquet", duplicates)
    write_markdown_table(audit_root / "duplicate_conflict_audit.md", "Duplicate Conflict Audit", duplicates, ["duplicate_group_id", "classification", "artifact_count", "sha256_count", "canonical_candidate_count", "notes"], limit=300)

    conformance = formal_conformance(artifacts, symbols, spec)
    write_parquet(audit_root / "formal_spec_conformance_matrix.parquet", conformance)
    write_markdown_table(audit_root / "formal_spec_conformance_matrix.md", "Formal Spec Conformance Matrix", conformance, ["formal_section", "status", "symbol_evidence_count", "notes"], limit=100)

    claims = scan_claims(audit_root) if args.include_reports else []
    provenance = provenance_audit(artifacts, claims)
    write_parquet(audit_root / "provenance_tier_audit.parquet", provenance)
    write_markdown_table(audit_root / "provenance_tier_audit.md", "Provenance Tier Audit", provenance, ["path", "assigned_tier", "rule_basis", "risk_flags", "notes"], limit=300)

    lineage = lineage_chains(artifacts)
    write_parquet(audit_root / "lineage_chains.parquet", lineage)
    write_markdown_table(audit_root / "lineage_chains.md", "Lineage Chains", lineage, ["chain_id", "source", "target", "status", "blocker_notes"], limit=100)

    active_rows, active_status = active_learning_audit(artifacts, symbols)
    no_canonical_spec = spec["canonical_status"] == "BLOCKED_NO_CANONICAL_SPEC"
    if no_canonical_spec and active_status == "ACTIVE_LEARNING_OPERATIONAL":
        active_status = "ACTIVE_LEARNING_SCORING_ONLY"
    write_parquet(audit_root / "active_learning_loop_audit.parquet", active_rows)
    write_markdown_table(audit_root / "active_learning_loop_audit.md", "Active Learning Loop Audit", active_rows, ["check", "status", "artifact_evidence", "symbol_evidence"], limit=100)

    write_parquet(audit_root / "schema_forensics.parquet", schema_rows)
    write_markdown_table(audit_root / "schema_forensics.md", "Schema Forensics", schema_rows, ["path", "schema_detected", "row_count_if_table", "required_columns_status", "semantic_key_consistency", "type_mismatches"], limit=300)

    write_parquet(audit_root / "claim_provenance_audit.parquet", claims)
    write_markdown_table(audit_root / "claim_provenance_audit.md", "Claim Provenance Audit", claims, ["path", "line", "claim_categories", "risk_flags", "provenance_status", "claim_excerpt"], limit=300)

    ip_rows = scan_ip_secrets(audit_root) if args.scan_ip_secrets else []
    write_parquet(audit_root / "ip_secret_exposure_audit.parquet", ip_rows)
    write_markdown_table(audit_root / "ip_secret_exposure_audit.md", "IP Secret Exposure Audit", ip_rows, ["path", "line", "finding_type", "risk_flag", "redacted_fingerprint", "notes"], limit=300)

    runtime_rows = runtime_consistency(artifacts, symbols)
    write_parquet(audit_root / "runtime_consistency_audit.parquet", runtime_rows)
    write_markdown_table(audit_root / "runtime_consistency_audit.md", "Runtime Consistency Audit", runtime_rows, ["check", "status", "risk_flags", "evidence"], limit=100)

    molecule_rows, molecule_status = molecule_design_audit(artifacts, symbols)
    write_parquet(audit_root / "molecule_design_readiness_audit.parquet", molecule_rows)
    write_markdown_table(audit_root / "molecule_design_readiness_audit.md", "Molecule Design Readiness Audit", molecule_rows, ["check", "status", "artifact_evidence", "symbol_evidence"], limit=100)

    variant_rows, variant_status = variant_durability_audit(artifacts, symbols)
    if no_canonical_spec and variant_status == "VARIANT_DURABILITY_OPERATIONAL":
        variant_status = "VARIANT_DURABILITY_PARTIAL"
    write_parquet(audit_root / "variant_durability_audit.parquet", variant_rows)
    write_markdown_table(audit_root / "variant_durability_audit.md", "Variant Durability Audit", variant_rows, ["check", "status", "artifact_evidence", "symbol_evidence"], limit=100)

    tests = test_gate_audit(artifacts) if args.include_tests else []
    write_parquet(audit_root / "test_gate_audit.parquet", tests)
    write_markdown_table(audit_root / "test_gate_audit.md", "Test Gate Audit", tests, ["gate", "status", "evidence", "notes"], limit=100)

    blockers = build_blockers(spec, artifacts, conformance, lineage, active_status, molecule_status, variant_status, ip_rows, runtime_rows, claims, duplicates, schema_rows)
    write_parquet(audit_root / "blocker_index.parquet", blockers)
    write_json(audit_root / "blocker_index.json", blockers)
    write_markdown_table(audit_root / "blocker_index.md", "Blocker Index", blockers, ["blocker_id", "code", "severity", "evidence", "notes"], limit=200)

    ontology_node_count = 0
    ontology_edge_count = 0
    if args.write_graphs:
        ontology_node_count, ontology_edge_count = make_graphs(audit_root, artifacts, symbols, edges, lineage, blockers)

    spectral_spine_status = "BLOCKED_NO_CANONICAL_SPEC" if spec["canonical_status"] == "BLOCKED_NO_CANONICAL_SPEC" else ("MISSING_W_HAT" if any(b["code"] == "MISSING_W_HAT" for b in blockers) else "PARTIAL")
    log_subtb_status = status_from_evidence(artifacts, ["subtb", "gflownet"], "LOG_SUBTB_PRESENT", "MISSING", "LOG_SUBTB_ARTIFACTS_PRESENT_NOT_CANONICAL")
    captured_tile_status = status_from_evidence(artifacts, ["captured", "tile"], "CAPTURED_TILE_PRESENT", "MISSING", "CAPTURED_TILE_ARTIFACTS_PRESENT_NOT_CANONICAL")
    ip_secret_status = "SECRET_EXPOSED" if any(r["risk_flag"] in {"SECRET_EXPOSED", "CREDENTIAL_EXPOSED"} for r in ip_rows) else ("IP_RISK_FINDINGS" if ip_rows else "NO_FINDINGS")

    gstatus = global_status(spec, blockers, molecule_status, variant_status, active_status)
    severity_counts = Counter(b["severity"] for b in blockers)
    summary = {
        "generated_at_utc": utc_now(),
        "global_status": gstatus,
        "audit_root": rel(audit_root),
        "sqlite_index_path": rel(audit_root / "ontology_index.sqlite"),
        "final_report_path": rel(audit_root / "PRISM_DSTW_E2E_ONTOLOGY_FORENSIC_AUDIT_v2.md"),
        "artifact_count": len(artifacts),
        "symbol_count": len(symbols),
        "dependency_edge_count": len(edges),
        "ontology_node_count": ontology_node_count,
        "ontology_edge_count": ontology_edge_count,
        "blocker_count": len(blockers),
        "blocker_count_by_severity": dict(sorted(severity_counts.items())),
        "molecule_design_status": molecule_status,
        "variant_durability_status": variant_status,
        "active_learning_status": active_status,
        "spectral_spine_status": spectral_spine_status,
        "log_subtb_status": log_subtb_status,
        "captured_tile_status": captured_tile_status,
        "ip_secret_status": ip_secret_status,
        "top_10_blockers": [f"{b['severity']}:{b['code']}" for b in sorted(blockers, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(x["severity"], 9))[:10]],
        "next_action": "Recover or authoritatively place DSTW_FORMAL_SPECIFICATION_v1.md before marking any E2E path canonical" if spec["canonical_status"] == "BLOCKED_NO_CANONICAL_SPEC" else "Resolve critical blockers in blocker_index.md",
        "next_actions": [
            "Recover or authoritatively place DSTW_FORMAL_SPECIFICATION_v1.md and rerun this audit.",
            "Resolve MISSING_W_HAT/C4/C6 blockers before any operational DSTW claim.",
            "Triage credential/IP findings by path and line using redacted fingerprints only.",
            "Add chemical tile registry plus validity and tile-to-operator/C6 mappings before molecule-ready claims.",
            "Add variant-specific operator and C6 contexts before variant durability operational claims.",
        ],
    }

    ontology_sql_nodes, ontology_sql_edges = ontology_sqlite_rows(artifacts, symbols, edges, blockers)
    sqlite_tables = {
        "artifacts": artifacts,
        "code_symbols": symbols,
        "dependency_edges": edges,
        "ontology_nodes": ontology_sql_nodes,
        "ontology_edges": ontology_sql_edges,
        "formal_spec_conformance": conformance,
        "provenance_tiers": provenance,
        "lineage_chains": lineage,
        "schemas": schema_rows,
        "claims": claims,
        "blockers": blockers,
        "tests": tests,
        "runtime_metrics": runtime_rows,
        "releases": [a for a in artifacts if a["owning_domain"] == "release_packages/hashes/tags"],
        "ip_secret_findings": ip_rows,
    }
    if args.write_sqlite:
        sqlite_write(audit_root / "ontology_index.sqlite", sqlite_tables)

    write_final_report(
        audit_root / "PRISM_DSTW_E2E_ONTOLOGY_FORENSIC_AUDIT_v2.md",
        summary,
        spec,
        scope_rows,
        artifacts,
        symbols,
        edges,
        duplicates,
        conformance,
        provenance,
        lineage,
        active_rows,
        runtime_rows,
        molecule_rows,
        variant_rows,
        claims,
        ip_rows,
        tests,
        blockers,
    )
    write_json(audit_root / "audit_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--include-src", action="store_true")
    parser.add_argument("--include-rust", action="store_true")
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--include-reports", action="store_true")
    parser.add_argument("--write-sqlite", action="store_true")
    parser.add_argument("--write-graphs", action="store_true")
    parser.add_argument("--scan-ip-secrets", action="store_true")
    parser.add_argument("--index-symbols", action="store_true")
    parser.add_argument("--build-dependency-graph", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.campaign != "glp1r_aleniglipron":
        print(f"Unsupported campaign for this directive: {args.campaign}", file=sys.stderr)
        return 2
    summary = write_all_outputs(args)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
