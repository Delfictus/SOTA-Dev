from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRATCH_ROOT = Path("/mnt/storage/prism-scratch/Prism4D-bio")
DEFAULT_DOCK_ENV = Path("/mnt/storage/prism_env_copies/prism_dock_portable_20260529")


def canonical_runtime_strict() -> bool:
    return os.environ.get("PRISM_CANONICAL_RUNTIME_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ordered_unique(paths: list[Path | None]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def prism_dock_env_candidates() -> list[Path]:
    override = os.environ.get("PRISM_DOCK_ENV")
    executable = Path(sys.executable).resolve()
    inferred = executable.parent.parent if (executable.parent.parent / "bin").exists() else None
    if canonical_runtime_strict():
        return _ordered_unique(
            [
                Path(override) if override else None,
                inferred,
                DEFAULT_DOCK_ENV,
            ]
        )
    return _ordered_unique(
        [
            Path(override) if override else None,
            inferred,
            DEFAULT_DOCK_ENV,
            Path("/media/diddy/PRISM-LBS/PRISM-ISOLATED-20260528/envs/prism_dock"),
            Path("/home/diddy/miniconda3/envs/prism_dock"),
        ]
    )


def resolve_prism_dock_env() -> Path:
    for candidate in prism_dock_env_candidates():
        if (candidate / "bin" / "python").exists():
            return candidate
    if canonical_runtime_strict():
        raise FileNotFoundError(
            "canonical PRISM dock env not found; checked "
            + ", ".join(str(p) for p in prism_dock_env_candidates())
        )
    return Path("/home/diddy/miniconda3/envs/prism_dock")


def prism_scratch_candidates(repo_root: Path | None = None) -> list[Path]:
    override = os.environ.get("PRISM_SCRATCH_ROOT")
    if canonical_runtime_strict():
        return _ordered_unique(
            [
                Path(override) if override else None,
                DEFAULT_SCRATCH_ROOT,
            ]
        )
    root = repo_root or REPO_ROOT
    repo_scratch = root / ".scratch"
    resolved_repo_scratch: Path | None = None
    if repo_scratch.exists() and repo_scratch.is_dir():
        resolved_repo_scratch = repo_scratch
    elif repo_scratch.is_symlink():
        target = repo_scratch.resolve(strict=False)
        if target.exists() and target.is_dir():
            resolved_repo_scratch = target
    return _ordered_unique(
        [
            Path(override) if override else None,
            DEFAULT_SCRATCH_ROOT,
            resolved_repo_scratch,
        ]
    )


def resolve_prism_scratch_root(repo_root: Path | None = None) -> Path:
    for candidate in prism_scratch_candidates(repo_root):
        if candidate.exists() and candidate.is_dir():
            return candidate
    if canonical_runtime_strict():
        return Path(os.environ.get("PRISM_SCRATCH_ROOT", str(DEFAULT_SCRATCH_ROOT)))
    return Path(os.environ.get("PRISM_SCRATCH_ROOT", str(DEFAULT_SCRATCH_ROOT)))


def ensure_prism_scratch_subdir(*parts: str, repo_root: Path | None = None) -> Path:
    root = resolve_prism_scratch_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_prism_dock_python() -> Path:
    override = os.environ.get("PRISM_DOCK_PYTHON")
    candidates = _ordered_unique(
        [
            Path(override) if override else None,
            resolve_prism_dock_env() / "bin" / "python",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if canonical_runtime_strict():
        raise FileNotFoundError(
            "canonical PRISM dock python not found; checked "
            + ", ".join(str(p) for p in candidates)
        )
    return resolve_prism_dock_env() / "bin" / "python"


def resolve_amberhome() -> Path:
    override = os.environ.get("PRISM_AMBERHOME")
    if override:
        return Path(override)
    return resolve_prism_dock_env()


def resolve_antechamber() -> Path:
    override = os.environ.get("PRISM_ANTECHAMBER")
    candidates = _ordered_unique(
        [
            Path(override) if override else None,
            resolve_amberhome() / "bin" / "antechamber",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if canonical_runtime_strict():
        raise FileNotFoundError(
            "canonical antechamber not found; checked " + ", ".join(str(p) for p in candidates)
        )
    return resolve_amberhome() / "bin" / "antechamber"


def resolve_obabel() -> Path:
    override = os.environ.get("PRISM_OBABEL")
    candidates = _ordered_unique(
        [
            Path(override) if override else None,
            resolve_prism_dock_env() / "bin" / "obabel",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if canonical_runtime_strict():
        raise FileNotFoundError(
            "canonical obabel not found; checked " + ", ".join(str(p) for p in candidates)
        )
    return resolve_prism_dock_env() / "bin" / "obabel"


def resolve_unidock() -> Path:
    override = os.environ.get("PRISM_UNIDOCK")
    candidates = _ordered_unique(
        [
            Path(override) if override else None,
            resolve_prism_dock_env() / "bin" / "unidock",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if canonical_runtime_strict():
        raise FileNotFoundError(
            "canonical unidock not found; checked " + ", ".join(str(p) for p in candidates)
        )
    return resolve_prism_dock_env() / "bin" / "unidock"


def resolve_gnina() -> Path:
    override = os.environ.get("PRISM_GNINA")
    candidates = _ordered_unique(
        [
            Path(override) if override else None,
            resolve_prism_dock_env() / "bin" / "gnina",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if canonical_runtime_strict():
        raise FileNotFoundError(
            "canonical gnina not found; checked " + ", ".join(str(p) for p in candidates)
        )
    return resolve_prism_dock_env() / "bin" / "gnina"
