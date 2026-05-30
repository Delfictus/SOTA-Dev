#!/usr/bin/env python3
"""Shared helpers for sealed release packaging and verification."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


DATABASE_SUFFIXES = {
    ".db",
    ".duckdb",
    ".h5",
    ".hdf5",
    ".parquet",
    ".sqlite",
    ".sqlite3",
}
TEXT_SUFFIXES = {
    ".c",
    ".cfg",
    ".conf",
    ".cpp",
    ".csv",
    ".cu",
    ".cuh",
    ".env",
    ".ini",
    ".ipynb",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".pyi",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_KEYWORDS = (
    "AWS",
    "CLOUDFLARE",
    "PASSWORD",
    "PRIVATE KEY",
    "R2",
    "TOKEN",
    "API_KEY",
)
SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|token|password|secret|cloudflare|aws|r2)\b"),
    re.compile(r"(?i)-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(cf(at|ut)_[A-Za-z0-9_-]{8,})"),
    re.compile(r"(?i)\b(ghp_[A-Za-z0-9]{10,})"),
)
COMPUTE_PROCESS_PATTERNS = (
    re.compile(r"nhs_rt_full"),
    re.compile(r"prism", re.IGNORECASE),
    re.compile(r"phase2c", re.IGNORECASE),
    re.compile(r"dstw", re.IGNORECASE),
    re.compile(r"cargo run", re.IGNORECASE),
)


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(data), encoding="utf-8")


def append_log(log_path: Path | None, line: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{now_utc()} {line}\n")


def run(
    args: Iterable[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    log_path: Path | None = None,
    timeout: float | None = None,
) -> CommandResult:
    cmd = [str(part) for part in args]
    append_log(
        log_path,
        f"RUN cwd={cwd or Path.cwd()} cmd={' '.join(shlex.quote(part) for part in cmd)}",
    )
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    append_log(log_path, f"EXIT code={proc.returncode}")
    if proc.stdout:
        append_log(log_path, f"STDOUT {proc.stdout[:4000].rstrip()}")
    if proc.stderr:
        append_log(log_path, f"STDERR {proc.stderr[:4000].rstrip()}")
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd,
            output=proc.stdout,
            stderr=proc.stderr,
        )
    return CommandResult(cmd, proc.returncode, proc.stdout, proc.stderr)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except PermissionError:
        proc = subprocess.run(
            ["sudo", "-n", "sha256sum", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise
        return proc.stdout.split()[0]


def blake3_path(path: Path) -> str | None:
    try:
        import blake3  # type: ignore
    except Exception:
        return None
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def human_bytes(size_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def chunk_size_bytes(chunk_size_gib: float) -> int:
    return int(chunk_size_gib * (1024**3))


def file_mode_string(mode: int) -> str:
    return oct(stat.S_IMODE(mode))


def is_executable(mode: int) -> bool:
    return bool(mode & stat.S_IXUSR)


def is_database_path(path: Path) -> bool:
    return path.suffix.lower() in DATABASE_SUFFIXES


def is_text_candidate(path: Path, size_bytes: int) -> bool:
    if size_bytes > 5 * 1024 * 1024:
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return path.name.lower() in {
        "makefile",
        "dockerfile",
        ".env",
        ".gitignore",
        ".dockerignore",
        "cargo.lock",
        "cargo.toml",
        "pyproject.toml",
        "requirements.txt",
    }


def classify_path(relpath: str, path: Path) -> str:
    lowered = relpath.lower()
    suffix = path.suffix.lower()
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if is_database_path(path):
        return "database"
    if "/docs/" in lowered or suffix in {".md", ".pdf", ".docx"}:
        return "documentation"
    if lowered.startswith("scripts/") or suffix in {".sh", ".py"}:
        return "script"
    if lowered.startswith("src/") or lowered.startswith("crates/") or suffix in {".rs", ".c", ".cpp", ".h", ".cuh", ".cu"}:
        return "source"
    if suffix in {".ptx", ".cubin", ".fatbin", ".so"}:
        return "cuda_runtime"
    if "/campaigns/" in lowered:
        return "campaign_artifact"
    if lowered.startswith("target/"):
        return "build_output"
    if suffix in {".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "config_or_manifest"
    return "binary_or_other"


def git_status_map(repo_root: Path, *, log_path: Path | None = None) -> dict[str, str]:
    result = run(
        [
            "git",
            "-c",
            "core.quotePath=false",
            "status",
            "--porcelain=1",
            "--ignored=matching",
            "-uall",
        ],
        cwd=repo_root,
        log_path=log_path,
    )
    statuses: dict[str, str] = {}
    for raw_line in result.stdout.splitlines():
        if not raw_line:
            continue
        prefix = raw_line[:2]
        path_text = raw_line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        statuses[path_text] = prefix
    return statuses


def iter_inventory_roots(
    source_roots: list[tuple[str, Path]],
    *,
    exclude_dirs: set[Path],
) -> Iterable[tuple[str, Path, Path]]:
    for source_name, source_root in source_roots:
        if source_root.is_file():
            yield source_name, source_root.parent, source_root
            continue
        for dirpath, dirnames, filenames in os.walk(source_root, followlinks=False):
            current = Path(dirpath)
            retained_dirnames = []
            for name in dirnames:
                child = current / name
                if child in exclude_dirs or (current.name == ".git" and name == "objects"):
                    continue
                if child.is_symlink():
                    yield source_name, source_root, child
                    continue
                retained_dirnames.append(name)
            dirnames[:] = retained_dirnames
            yield source_name, source_root, current
            for filename in filenames:
                yield source_name, source_root, current / filename


def inventory_records(
    source_roots: list[tuple[str, Path]],
    *,
    repo_root: Path,
    exclude_dirs: set[Path],
    git_statuses: dict[str, str],
    compute_hashes: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_name, source_root, current in iter_inventory_roots(source_roots, exclude_dirs=exclude_dirs):
        stat_result = current.lstat()
        relpath = (
            current.relative_to(source_root)
            if current != source_root or source_root.is_file()
            else Path(".")
        )
        reltext = str(relpath)
        record: dict[str, Any] = {
            "source_root_name": source_name,
            "source_root_path": str(source_root),
            "relative_path": reltext,
            "absolute_path": str(current),
            "file_type": "symlink" if current.is_symlink() else "directory" if current.is_dir() else "file",
            "size_bytes": stat_result.st_size,
            "mtime_utc": datetime.fromtimestamp(stat_result.st_mtime, UTC).isoformat(),
            "permissions": file_mode_string(stat_result.st_mode),
            "executable": is_executable(stat_result.st_mode),
            "symlink_target": os.readlink(current) if current.is_symlink() else None,
            "sha256": None,
        }
        repo_relative = None
        try:
            repo_relative = str(current.relative_to(repo_root))
        except ValueError:
            repo_relative = None
        record["git_status"] = git_statuses.get(repo_relative or "", "UNTRACKED_EXTERNAL" if repo_relative is None else "  ")
        record["classification"] = classify_path(repo_relative or reltext, current)
        records.append(record)
    records.sort(key=lambda item: (item["source_root_name"], item["relative_path"]))
    if compute_hashes:
        file_records = [record for record in records if record["file_type"] == "file" and record["sha256"] is None]
        max_workers = min(8, max(1, (os.cpu_count() or 4) // 2))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for record, digest in zip(
                file_records,
                executor.map(lambda item: sha256_path(Path(item["absolute_path"])), file_records),
            ):
                record["sha256"] = digest
    return records


def default_destination_plan(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    size_mib = record["size_bytes"] / (1024 * 1024)
    classification = record["classification"]
    rel_lower = str(record.get("relative_path", "")).lower()
    repo_scoped = bool(record.get("repo_relative_path")) or record["source_root_name"] == "repo"
    github_include = True
    github_reason = "included"
    if any(marker in rel_lower for marker in ("/.claude/", "/.codex/", "/.wrangler/", "/.config/", "credentials.env", "rclone.conf", "settings.local.json", "secret_scan")):
        github_include = False
        github_reason = "sensitive_local_config_omitted_from_github"
    elif record["relative_path"].startswith(".") or "/." in record["relative_path"]:
        github_include = False
        github_reason = "hidden_local_state_omitted_from_github"
    elif classification == "database" and size_mib > 0:
        github_include = False
        github_reason = "database_omitted_from_github"
    elif size_mib > 50:
        github_include = False
        github_reason = "over_50mib_github_omission"
    elif not repo_scoped:
        github_include = False
        github_reason = "external_capture_omitted_from_github"
    return {
        "github": {"include": github_include, "reason": github_reason},
        "ssd": {"include": True, "reason": "full_release_target"},
        "r2": {"include": True, "reason": "full_release_target"},
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_policy(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"release policy must be JSON-compatible YAML: {path}") from exc
    loaded = yaml.safe_load(content)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"release policy must decode to a mapping: {path}")
    return loaded


def scan_secret_candidates(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if path.name.lower().endswith((".pem", ".key", ".p12")):
        findings.append({"path": str(path), "kind": "filename", "match": path.name})
        return findings
    try:
        size_bytes = path.stat().st_size
    except FileNotFoundError:
        return findings
    if not is_text_candidate(path, size_bytes):
        lowered = path.name.upper()
        if any(keyword in lowered for keyword in SECRET_KEYWORDS):
            findings.append({"path": str(path), "kind": "filename", "match": path.name})
        return findings
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings
    if any(keyword in path.name.upper() for keyword in SECRET_KEYWORDS):
        findings.append({"path": str(path), "kind": "filename", "match": path.name})
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(content):
            findings.append(
                {
                    "path": str(path),
                    "kind": "content",
                    "match": redact_secret(match.group(0)),
                }
            )
    return findings


def redact_secret(value: str) -> str:
    if len(value) <= 12:
        return "[REDACTED]"
    return f"{value[:4]}...[REDACTED]...{value[-4:]}"


def checkpoint_sqlite(path: Path) -> dict[str, Any]:
    import sqlite3

    result = {
        "path": str(path),
        "wal_checkpoint": "not_attempted",
        "backup_path": None,
        "backup_status": "not_attempted",
    }
    try:
        connection = sqlite3.connect(str(path))
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            result["wal_checkpoint"] = "ok"
        finally:
            connection.close()
    except Exception as exc:  # pragma: no cover - best effort
        result["wal_checkpoint"] = f"error:{exc}"
    return result


def sqlite_backup(source: Path, target: Path) -> dict[str, Any]:
    import sqlite3

    target.parent.mkdir(parents=True, exist_ok=True)
    result = {"source": str(source), "target": str(target), "status": "not_attempted"}
    try:
        src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        dst = sqlite3.connect(str(target))
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
        result["status"] = "ok"
    except Exception as exc:  # pragma: no cover - best effort
        result["status"] = f"error:{exc}"
    return result


def duckdb_checkpoint(path: Path) -> dict[str, Any]:
    result = {"path": str(path), "checkpoint": "not_attempted"}
    try:
        import duckdb  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        result["checkpoint"] = f"unavailable:{exc}"
        return result
    try:
        connection = duckdb.connect(str(path))
        connection.execute("CHECKPOINT")
        connection.close()
        result["checkpoint"] = "ok"
    except Exception as exc:  # pragma: no cover - best effort
        result["checkpoint"] = f"error:{exc}"
    return result


def find_active_compute_processes() -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        capture_output=True,
        text=True,
        check=True,
    )
    findings: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid, comm, args = (stripped.split(None, 2) + [""])[:3]
        if any(pattern.search(args) or pattern.search(comm) for pattern in COMPUTE_PROCESS_PATTERNS):
            findings.append({"pid": int(pid), "command": comm, "args": args})
    return findings


def gather_tool_versions(commands: list[str], *, log_path: Path | None = None) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for command in commands:
        resolved = shutil_which(command)
        if not resolved:
            versions[command] = {"available": False}
            continue
        version_output = ""
        for args in ([command, "--version"], [command, "-V"]):
            try:
                version_output = run(args, check=False, log_path=log_path).stdout or run(
                    args,
                    check=False,
                    log_path=log_path,
                ).stderr
                if version_output:
                    break
            except Exception:
                continue
        versions[command] = {
            "available": True,
            "path": resolved,
            "version_output": version_output.strip(),
        }
    return versions


def shutil_which(name: str) -> str | None:
    return shutil.which(name)


def rsync_copy(
    source: Path,
    destination: Path,
    *,
    log_path: Path | None = None,
    extra_args: list[str] | None = None,
    dereference: bool = False,
    check: bool = True,
) -> CommandResult:
    destination.parent.mkdir(parents=True, exist_ok=True)
    args = ["rsync", "-aHAX", "--partial", "--append-verify", "--numeric-ids"]
    if dereference:
        args.append("--copy-links")
    if extra_args:
        args.extend(extra_args)
    source_arg = f"{source}/" if source.is_dir() else str(source)
    destination_arg = f"{destination}/" if source.is_dir() else str(destination)
    args.extend([source_arg, destination_arg])
    return run(args, log_path=log_path, check=check)


def estimate_archive_bytes(records: list[dict[str, Any]]) -> int:
    total = 0
    for record in records:
        if record["file_type"] != "file":
            continue
        size_bytes = int(record["size_bytes"])
        path = Path(record["absolute_path"])
        suffix = path.suffix.lower()
        factor = 1.0
        if suffix in {".json", ".jsonl", ".md", ".py", ".rs", ".toml", ".txt", ".yaml", ".yml", ".csv", ".tsv", ".sql"}:
            factor = 0.35
        elif suffix in {".parquet", ".zip", ".gz", ".zst", ".pt", ".bin", ".so", ".ptx", ".cubin", ".fatbin"}:
            factor = 0.95
        elif suffix in {".sqlite", ".duckdb", ".db", ".h5", ".hdf5"}:
            factor = 0.85
        elif record["classification"] == "directory":
            factor = 0.0
        total += int(size_bytes * factor)
    return total


def compute_release_root_hash(records: list[dict[str, Any]]) -> str:
    canonical_lines = []
    for record in records:
        line = {
            "source_root_name": record["source_root_name"],
            "relative_path": record["relative_path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
            "permissions": record["permissions"],
            "mtime_utc": record["mtime_utc"],
            "classification": record["classification"],
            "destinations": record.get("destinations", {}),
        }
        canonical_lines.append(json.dumps(line, sort_keys=True))
    return sha256_text("\n".join(sorted(canonical_lines)))


def relative_records_tree(records: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{record['source_root_name']}:{record['relative_path']}"
        for record in records
    ) + "\n"


def ensure_temp_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def verify_git_bundle(bundle_path: Path, *, log_path: Path | None = None) -> dict[str, Any]:
    result = run(["git", "bundle", "verify", str(bundle_path)], check=False, log_path=log_path)
    return {
        "path": str(bundle_path),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "verified": result.returncode == 0,
    }


def verify_sha256_manifest(checksums_path: Path, *, cwd: Path | None = None, log_path: Path | None = None) -> dict[str, Any]:
    result = run(
        ["sha256sum", "-c", str(checksums_path)],
        cwd=cwd,
        check=False,
        log_path=log_path,
    )
    return {
        "path": str(checksums_path),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "verified": result.returncode == 0,
    }


def tar_extract(archive_path: Path, target_dir: Path, *, log_path: Path | None = None) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zst":
        cmd = [
            "bash",
            "-lc",
            f"zstd -dc {shlex.quote(str(archive_path))} | tar -xpf - -C {shlex.quote(str(target_dir))}",
        ]
    else:
        cmd = ["tar", "-xpf", str(archive_path), "-C", str(target_dir)]
    result = run(cmd, check=False, log_path=log_path)
    return {
        "archive": str(archive_path),
        "target_dir": str(target_dir),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "verified": result.returncode == 0,
    }


def detect_symlink_leaks(root: Path, forbidden_prefixes: list[str]) -> list[dict[str, Any]]:
    leaks: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        target = os.readlink(path)
        if any(target.startswith(prefix) for prefix in forbidden_prefixes):
            leaks.append({"path": str(path), "target": target})
    return leaks


def file_count_and_bytes(root: Path) -> tuple[int, int]:
    total_files = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total_files += 1
            total_bytes += path.stat().st_size
    return total_files, total_bytes


def resolve_prism_input_layout(input_root: Path) -> dict[str, Any]:
    resolved = input_root.resolve()
    canonical_repo_root = resolved / "absolute_root" / "home" / "diddy" / "Desktop" / "Prism4D-bio"
    canonical_env_root = resolved / "absolute_root" / "home" / "diddy" / "miniconda3"
    canonical_candidate_smoke_root = (
        resolved
        / "absolute_root"
        / "mnt"
        / "storage"
        / "tmp"
        / "glp1r_candidate_md_smoke_20260527_233437"
    )
    canonical_credentials = resolved / "absolute_root" / "home" / "diddy" / ".config" / "prism" / "credentials.env"
    canonical_rclone = resolved / "absolute_root" / "etc" / "rclone" / "rclone.conf"
    if canonical_repo_root.exists():
        return {
            "layout_kind": "canonical_platform_copy",
            "input_root": resolved,
            "repo_root": canonical_repo_root,
            "campaign_root": canonical_repo_root / "campaigns" / "glp1r_aleniglipron",
            "env_root": canonical_env_root if canonical_env_root.exists() else None,
            "candidate_smoke_root": canonical_candidate_smoke_root if canonical_candidate_smoke_root.exists() else None,
            "credentials_path": canonical_credentials if canonical_credentials.exists() else None,
            "rclone_config_path": canonical_rclone if canonical_rclone.exists() else None,
        }
    return {
        "layout_kind": "repo_only",
        "input_root": resolved,
        "repo_root": resolved,
        "campaign_root": resolved / "campaigns" / "glp1r_aleniglipron",
        "env_root": None,
        "candidate_smoke_root": None,
        "credentials_path": None,
        "rclone_config_path": None,
    }
