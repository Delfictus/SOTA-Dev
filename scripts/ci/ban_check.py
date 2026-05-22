#!/usr/bin/env python3
"""CI gate: fail on banned workspace implementation patterns."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path


PRODUCTION_PATHS = ["src", "crates", "campaigns", "scripts", "tests", "00_registry", "prism_dstw"]
PYTHON_SUFFIXES = {".py"}
TEXT_SUFFIXES = {".rs", ".toml", ".cfg"}
EXEMPTION_REGISTRY = Path("00_registry/ban_exemptions.yml")
DIRECT_PARQUET_WRITER_EXEMPT_PATHS = {"prism_dstw/io.py", "src/prism_dstw/io.py"}

EXEMPTION_MAX_DAYS = 90

BANNED_MODULES = {
    "duckdb": "banned_data_engine",
    "pandas": "banned_dataframe_engine",
    "sqlite3": "banned_sqlite_analytics",
    "csv": "banned_text_table_io",
    "openpyxl": "banned_excel_processing",
    "xlrd": "banned_excel_processing",
    "pickle": "banned_pickle_serialization",
    "cPickle": "banned_pickle_serialization",
    "tensorflow": "banned_ml_framework",
    "keras": "banned_ml_framework",
}

BANNED_FROM_IMPORTS = {
    ("numpy.linalg", "inv"): "banned_matrix_inverse",
    ("numpy.linalg", "det"): "banned_determinant",
    ("scipy.optimize", "minimize"): "minimize_requires_explicit_derivatives",
}

BANNED_CALLS = {
    ("json", "dump"): "json_dump_banned_for_analytical_data",
    ("yaml", "dump"): "yaml_dump_banned_for_data_output",
}

BANNED_DIRECT_WRITER_METHODS = {
    "write_parquet",
    "sink_parquet",
    "write_table",
    "ParquetWriter",
}

BANNED_NUMPY_LINALG_CALLS = {
    "inv": "banned_matrix_inverse",
    "det": "banned_determinant",
}

BANNED_DYNAMIC_IMPORT_MODULES = set(BANNED_MODULES) | {"sklearn"}

BANNED_TEXT_DEPENDENCIES = {
    "duckdb": "banned_duckdb_dependency",
    "rusqlite": "banned_rusqlite_dependency_except_d1",
    "libsqlite3-sys": "banned_sqlite_dependency_except_d1",
    "sqlx-sqlite": "banned_sqlite_dependency_except_d1",
    "\"sqlite\"": "banned_sqlite_feature_except_d1",
}

SKLEARN_ALLOWED_IMPORTS = {
    ("sklearn.preprocessing", "StandardScaler"),
    ("sklearn.metrics", "*"),
}


@dataclass(frozen=True)
class Violation:
    path: Path
    line_no: int
    rule: str
    detail: str


def emit_stdout(message: str) -> None:
    sys.stdout.write(message + "\n")


def emit_stderr(message: str) -> None:
    sys.stderr.write(message + "\n")


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in PRODUCTION_PATHS:
        base = root / rel
        if not base.exists():
            continue
        if base.is_file() and (base.suffix in PYTHON_SUFFIXES or base.suffix in TEXT_SUFFIXES):
            files.append(base)
            continue
        for path in base.rglob("*"):
            if path.is_file() and (path.suffix in PYTHON_SUFFIXES or path.suffix in TEXT_SUFFIXES):
                files.append(path)
    return sorted(files)


def parse_exemptions(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    today = date.today()
    exemptions: set[tuple[str, str, int]] = set()
    for item in raw.get("exemptions", []):
        current_path = str(item.get("path", ""))
        expires = date.fromisoformat(str(item.get("expires", "1970-01-01")))
        if not current_path or expires < today or (expires - today).days > EXEMPTION_MAX_DAYS:
            continue
        for rule_item in item.get("rules", []):
            if isinstance(rule_item, str):
                continue
            rule = str(rule_item.get("rule", ""))
            for line_no in rule_item.get("lines", []):
                exemptions.add((current_path, rule, int(line_no)))
    return exemptions


def module_root(name: str) -> str:
    return name.split(".", 1)[0]


def is_sklearn_allowed(module: str | None, imported_name: str | None) -> bool:
    if module is None:
        return False
    if (module, imported_name or "") in SKLEARN_ALLOWED_IMPORTS:
        return True
    if (module, "*") in SKLEARN_ALLOWED_IMPORTS:
        return True
    return False


def attribute_chain(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = attribute_chain(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def check_python(path: Path) -> list[Violation]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [Violation(path, exc.lineno or 1, "python_syntax_error", str(exc))]

    violations: list[Violation] = []
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = module_root(alias.name)
                aliases[alias.asname or root] = alias.name
                if root in BANNED_MODULES:
                    violations.append(Violation(path, node.lineno, BANNED_MODULES[root], f"import {alias.name}"))
                if root == "sklearn":
                    violations.append(Violation(path, node.lineno, "sklearn_model_import_requires_exemption", f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module_root(module)
            for alias in node.names:
                imported = alias.name
                aliases[alias.asname or imported] = f"{module}.{imported}" if module else imported
                if root in BANNED_MODULES:
                    violations.append(Violation(path, node.lineno, BANNED_MODULES[root], f"from {module} import {imported}"))
                if (module, imported) in BANNED_FROM_IMPORTS:
                    violations.append(Violation(path, node.lineno, BANNED_FROM_IMPORTS[(module, imported)], f"from {module} import {imported}"))
                if root == "sklearn" and not is_sklearn_allowed(module, imported):
                    violations.append(Violation(path, node.lineno, "sklearn_model_import_requires_exemption", f"from {module} import {imported}"))
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                violations.append(Violation(path, node.lineno, "bare_except_banned", "except:"))
            elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                violations.append(Violation(path, node.lineno, "broad_exception_banned", "except Exception"))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                violations.append(Violation(path, node.lineno, "print_banned_production_logging", "print()"))
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                key = (func.value.id, func.attr)
                if key in BANNED_CALLS:
                    violations.append(Violation(path, node.lineno, BANNED_CALLS[key], f"{func.value.id}.{func.attr}()"))
            chain = attribute_chain(func)
            if chain is None:
                continue
            parts = chain.split(".")
            if parts:
                resolved_root = aliases.get(parts[0], parts[0])
                resolved_chain = ".".join([resolved_root, *parts[1:]])
            else:
                resolved_chain = chain
            if resolved_chain in {"polars.sql", "pl.sql"} or resolved_chain.endswith(".SQLContext"):
                violations.append(Violation(path, node.lineno, "raw_sql_banned", f"{chain}()"))
            if resolved_chain.startswith("numpy.linalg."):
                operation = resolved_chain.rsplit(".", 1)[-1]
                if operation in BANNED_NUMPY_LINALG_CALLS:
                    violations.append(Violation(path, node.lineno, BANNED_NUMPY_LINALG_CALLS[operation], f"{chain}()"))
            if resolved_chain.startswith("np.linalg."):
                operation = resolved_chain.rsplit(".", 1)[-1]
                if operation in BANNED_NUMPY_LINALG_CALLS:
                    violations.append(Violation(path, node.lineno, BANNED_NUMPY_LINALG_CALLS[operation], f"{chain}()"))
            if resolved_chain in {"pytest.importorskip", "importlib.import_module"}:
                if node.args:
                    requested_module = string_literal(node.args[0])
                    if requested_module is not None:
                        requested_root = module_root(requested_module)
                        if requested_root in BANNED_DYNAMIC_IMPORT_MODULES:
                            rule = (
                                "sklearn_model_import_requires_exemption"
                                if requested_root == "sklearn"
                                else BANNED_MODULES[requested_root]
                            )
                            violations.append(
                                Violation(
                                    path,
                                    node.lineno,
                                    rule,
                                    f"{resolved_chain}({requested_module!r})",
                                )
                            )
            if parts[-1] in BANNED_DIRECT_WRITER_METHODS:
                path_text = path.as_posix()
                if not any(path_text.endswith(exempt) for exempt in DIRECT_PARQUET_WRITER_EXEMPT_PATHS):
                    violations.append(
                        Violation(path, node.lineno, "direct_parquet_writer_banned", f"{chain}()")
                    )
    return violations


def check_text(path: Path) -> list[Violation]:
    violations: list[Violation] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        lower = stripped.lower()
        for token, rule in BANNED_TEXT_DEPENDENCIES.items():
            if token in lower:
                violations.append(Violation(path, line_no, rule, stripped))
    return violations


def main() -> int:
    root = Path.cwd()
    exemptions = parse_exemptions(root / EXEMPTION_REGISTRY)
    violations: list[Violation] = []
    for path in iter_files(root):
        path_violations = check_python(path) if path.suffix in PYTHON_SUFFIXES else check_text(path)
        for violation in path_violations:
            rel = str(violation.path.relative_to(root))
            if (rel, violation.rule, violation.line_no) in exemptions:
                continue
            violations.append(violation)

    if violations:
        emit_stderr("BANNED IMPLEMENTATION CHECK FAILED")
        for violation in violations:
            emit_stderr(f"{violation.path}:{violation.line_no}: {violation.rule}: {violation.detail}")
        return 1

    emit_stdout("BANNED IMPLEMENTATION CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
