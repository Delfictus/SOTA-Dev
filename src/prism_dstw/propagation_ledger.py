"""Append-only propagation ledger primitives with workspace path boundaries."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .exceptions import FatalBoundaryError


JsonObject = dict[str, Any]
LedgerEntry = dict[str, Any]
LedgerValue = float | JsonObject | str | None


@dataclass(frozen=True)
class PropagationEntry:
    """Single append-only runtime lineage entry."""

    entry_id: str
    module: str
    operation: str
    inputs: dict[str, str]
    input_checksums: dict[str, str]
    parameters: JsonObject
    output_value: LedgerValue
    output_uncertainty: float | None
    timestamp: str
    gate_status: dict[str, bool]
    supersedes: str | None = None


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_within_repo(path: Path, repo_root: Path) -> Path:
    root = repo_root.resolve()
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FatalBoundaryError(f"path is outside repository boundary: {path}") from exc
    return candidate


def repo_relative_path(path: Path, repo_root: Path | None = None) -> str:
    """Return a stable repository-relative path or raise on boundary escape."""

    root = (repo_root or default_repo_root()).resolve()
    return _absolute_within_repo(path, root).relative_to(root).as_posix()


def _looks_like_embedded_absolute_path(value: str) -> bool:
    return "/home/" in value or "/Users/" in value or value.startswith("file:")


def _sanitize_ledger_value(value: Any, repo_root: Path) -> Any:
    if isinstance(value, Path):
        return repo_relative_path(value, repo_root)
    if isinstance(value, str):
        if value.startswith("/"):
            return repo_relative_path(Path(value), repo_root)
        if _looks_like_embedded_absolute_path(value):
            raise FatalBoundaryError(
                "ledger strings must not embed workstation-local paths; pass paths as Path values"
            )
        return value
    if isinstance(value, list):
        return [_sanitize_ledger_value(item, repo_root) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_ledger_value(item, repo_root) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _sanitize_ledger_value(item, repo_root) for key, item in value.items()}
    return value


def canonicalize_ledger_entry(
    entry: Mapping[str, Any] | PropagationEntry,
    repo_root: Path | None = None,
) -> LedgerEntry:
    """Return a JSON-serializable ledger entry with repository-relative paths only."""

    root = (repo_root or default_repo_root()).resolve()
    raw_entry = asdict(entry) if isinstance(entry, PropagationEntry) else dict(entry)
    canonical = _sanitize_ledger_value(raw_entry, root)
    if not isinstance(canonical, dict):
        raise TypeError("canonical ledger entry must remain a mapping")
    return canonical


def load_latest_ledger_entry(ledger_path: Path) -> LedgerEntry | None:
    """Load the most recent non-empty ledger entry without modifying the ledger."""

    if not ledger_path.exists():
        return None
    latest: LedgerEntry | None = None
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise FatalBoundaryError(f"ledger entry is not a JSON object: {ledger_path}")
            latest = loaded
    return latest


def append_ledger_entry(
    ledger_path: Path,
    entry: Mapping[str, Any] | PropagationEntry,
    repo_root: Path | None = None,
) -> None:
    """Append one canonical ledger entry; this function never truncates or rewrites."""

    root = (repo_root or default_repo_root()).resolve()
    output_path = _absolute_within_repo(ledger_path, root)
    canonical = canonicalize_ledger_entry(entry, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(canonical, sort_keys=True) + "\n")


def build_superseding_entry(
    previous: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    updates: Mapping[str, Any],
) -> LedgerEntry:
    """Build a new entry that supersedes a prior immutable entry."""

    root = (repo_root or default_repo_root()).resolve()
    previous_entry_id = str(previous.get("entry_id", "legacy-entry-without-id"))
    entry: LedgerEntry = {
        **dict(previous),
        **dict(updates),
        "entry_id": str(uuid.uuid4()),
        "supersedes": previous_entry_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    return canonicalize_ledger_entry(entry, root)


def _path_for_checksum(path_value: Path | str, repo_root: Path) -> Path | None:
    path = path_value if isinstance(path_value, Path) else Path(path_value)
    try:
        candidate = _absolute_within_repo(path, repo_root)
    except FatalBoundaryError:
        return None
    return candidate if candidate.is_file() else None


def build_entry(
    *,
    module: str,
    operation: str,
    inputs: dict[str, Path | str],
    parameters: JsonObject,
    output_value: LedgerValue,
    output_uncertainty: float | None,
    gate_status: dict[str, bool],
    supersedes: str | None = None,
    repo_root: Path | None = None,
) -> PropagationEntry:
    root = (repo_root or default_repo_root()).resolve()
    string_inputs = {
        name: repo_relative_path(value, root) if isinstance(value, Path) else str(value)
        for name, value in inputs.items()
    }
    checksums: dict[str, str] = {}
    for name, value in inputs.items():
        checksum_path = _path_for_checksum(value, root)
        if checksum_path is not None:
            checksums[name] = sha256_path(checksum_path)
    return PropagationEntry(
        entry_id=str(uuid.uuid4()),
        module=module,
        operation=operation,
        inputs=string_inputs,
        input_checksums=checksums,
        parameters=parameters,
        output_value=output_value,
        output_uncertainty=output_uncertainty,
        timestamp=datetime.now(UTC).isoformat(),
        gate_status=gate_status,
        supersedes=supersedes,
    )


def append_propagation_entry(
    path: Path,
    entry: PropagationEntry,
    repo_root: Path | None = None,
) -> None:
    append_ledger_entry(path, entry, repo_root)


__all__ = [
    "JsonObject",
    "LedgerEntry",
    "LedgerValue",
    "PropagationEntry",
    "append_ledger_entry",
    "append_propagation_entry",
    "build_entry",
    "build_superseding_entry",
    "canonicalize_ledger_entry",
    "load_latest_ledger_entry",
    "repo_relative_path",
    "sha256_path",
]
