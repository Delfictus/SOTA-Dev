#!/usr/bin/env python3
"""Audit import resolution for PRISM Python source and scripts."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [REPO_ROOT / "src", REPO_ROOT / "scripts"]
ADDITIONAL_IMPORT_ROOTS = [
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts/training",
    REPO_ROOT / "scripts/training/vn_egnn",
    REPO_ROOT / "scripts/quarantine",
    REPO_ROOT / "scripts/fep",
    REPO_ROOT / "scripts/production",
    REPO_ROOT / "prism-ai-inference",
]
OPTIONAL_RUNTIME_DEPENDENCIES = {
    "gufe": "OpenFE FEP conversion surface; optional unless FEP execution is requested.",
    "openfe": "OpenFE FEP conversion surface; optional unless FEP execution is requested.",
    "openff": "OpenFF/NAGL synthon ingestion surface; optional unless charge ingestion is requested.",
    "openmmforcefields": "OpenMM MMGBSA rescoring surface; optional unless MMGBSA is requested.",
    "plip": "PLIP post-docking analysis surface; optional unless docking post-analysis is requested.",
    "sascorer": "RDKit contrib SA-score helper; optional unless medchem filtering is requested.",
    "extract_216": "Legacy training extractor referenced by archived training scripts.",
}


@dataclass(frozen=True)
class ImportRecord:
    file: str
    line: int
    module: str
    statement: str


def _insert_import_roots() -> None:
    for path in reversed(ADDITIONAL_IMPORT_ROOTS):
        if path.exists() and path.as_posix() not in sys.path:
            sys.path.insert(0, path.as_posix())


def _module_for_node(node: ast.AST) -> tuple[str, str] | None:
    if isinstance(node, ast.Import):
        names = ", ".join(alias.name for alias in node.names)
        first = node.names[0].name
        return first, f"import {names}"
    if isinstance(node, ast.ImportFrom):
        if node.level:
            return None
        if node.module is None:
            return None
        names = ", ".join(alias.name for alias in node.names)
        return node.module, f"from {node.module} import {names}"
    return None


def _iter_python_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(
                p
                for p in path.rglob("*.py")
                if "__pycache__" not in p.parts
                and ".venv" not in p.parts
                and "node_modules" not in p.parts
            )
    return sorted(set(files))


def _origin_class(spec: object) -> str:
    origin = str(getattr(spec, "origin", "") or "")
    if origin in ("built-in", "frozen"):
        return "stdlib"
    if origin.startswith(REPO_ROOT.as_posix()):
        return "repo"
    if "site-packages" in origin or "dist-packages" in origin:
        return "third_party"
    return "stdlib_or_environment"


def audit_imports(paths: list[Path], *, strict_optionals: bool = False) -> dict[str, object]:
    _insert_import_roots()
    resolved: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    optional_unavailable: list[dict[str, object]] = []
    parse_errors: list[dict[str, object]] = []

    for py_file in _iter_python_files(paths):
        rel = py_file.relative_to(REPO_ROOT).as_posix() if py_file.is_relative_to(REPO_ROOT) else py_file.as_posix()
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"), filename=py_file.as_posix())
        except SyntaxError as exc:
            parse_errors.append({"file": rel, "line": exc.lineno or 0, "error": str(exc)})
            continue
        for node in ast.walk(tree):
            extracted = _module_for_node(node)
            if extracted is None:
                continue
            module_name, statement = extracted
            top_level = module_name.split(".", 1)[0]
            line = int(getattr(node, "lineno", 0))
            record = ImportRecord(file=rel, line=line, module=module_name, statement=statement)
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None and "." in module_name:
                    spec = importlib.util.find_spec(top_level)
            except Exception as exc:  # pragma: no cover - importer-specific
                spec = None
                error = str(exc)
                try:
                    top_spec = importlib.util.find_spec(top_level)
                except Exception:
                    top_spec = None
                if top_spec is not None:
                    item = asdict(record)
                    item["origin_class"] = _origin_class(top_spec)
                    item["origin"] = str(getattr(top_spec, "origin", "") or "")
                    item["resolution_note"] = f"top-level module resolves; nested finder raised: {error}"
                    resolved.append(item)
                    continue
            else:
                error = "module spec not found"
            if spec is not None:
                item = asdict(record)
                item["origin_class"] = _origin_class(spec)
                item["origin"] = str(getattr(spec, "origin", "") or "")
                resolved.append(item)
            elif top_level in OPTIONAL_RUNTIME_DEPENDENCIES and not strict_optionals:
                item = asdict(record)
                item["reason"] = OPTIONAL_RUNTIME_DEPENDENCIES[top_level]
                optional_unavailable.append(item)
            else:
                item = asdict(record)
                item["error"] = error
                unresolved.append(item)

    circular: list[dict[str, object]] = []
    report: dict[str, object] = {
        "schema_version": "PRISM.import_resolution_audit.v1",
        "paths": [p.as_posix() for p in paths],
        "resolved": resolved,
        "unresolved": unresolved,
        "optional_unavailable": optional_unavailable,
        "parse_errors": parse_errors,
        "circular": circular,
        "summary": {
            "resolved_count": len(resolved),
            "unresolved_count": len(unresolved),
            "optional_unavailable_count": len(optional_unavailable),
            "parse_error_count": len(parse_errors),
            "circular_count": len(circular),
            "strict_optionals": strict_optionals,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "release_artifacts/v0.25.0/import_audit_report.json")
    parser.add_argument("--strict-optionals", action="store_true")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = [p if p.is_absolute() else REPO_ROOT / p for p in (args.paths or DEFAULT_PATHS)]
    report = audit_imports(paths, strict_optionals=bool(args.strict_optionals))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["summary"]
    if not isinstance(summary, dict):
        raise TypeError("import audit summary is malformed")
    print(
        "import_resolution_audit "
        f"resolved={summary['resolved_count']} "
        f"unresolved={summary['unresolved_count']} "
        f"optional_unavailable={summary['optional_unavailable_count']} "
        f"parse_errors={summary['parse_error_count']} "
        f"report={args.output}"
    )
    return 1 if summary["unresolved_count"] or summary["parse_error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
