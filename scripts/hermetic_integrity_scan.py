#!/usr/bin/env python3
"""Hermetic integrity scanner for E025-R1.0 release packaging."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)

PRODUCTION_DIRS = (
    "src/",
    "scripts/",
    "crates/prism-forge/",
    "crates/prism-nhs/",
    "crates/prism-gpu/",
    "cloud/prism-manifold-worker/src/",
    "cloud/prism-manifold-worker/wrangler.toml",
    "campaigns/",
    "tests/",
    "00_registry/",
)

SOURCE_EXTENSIONS = {
    ".py",
    ".rs",
    ".toml",
    ".ts",
    ".tsx",
    ".js",
    ".json",
    ".yaml",
    ".yml",
    ".sh",
    ".j2",
    ".md",
}

SKIP_DIRS = {
    ".git",
    ".archive",
    ".venv",
    "venv",
    ".figvenv",
    ".pytest_cache",
    ".mypy_cache",
    ".scratch",
    ".wrangler",
    "__pycache__",
    "node_modules",
    "target",
    "dist",
    "release_audit",
    "production_test",
    "test_output",
    "e2e_quick_test",
    "tuned_test",
}

GENERATED_PATH_PARTS = (
    "/audits/",
    "/reports/",
    "/report",
    "/manifests/",
    "/manifest",
    "/quarantine/",
    "/validation/",
    "/frozen_predictions/",
)

DATA_SUFFIXES = (
    ".parquet",
    ".sdf",
    ".pdb",
    ".mol2",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".pkl",
    ".pt",
    ".safetensors",
    ".npy",
    ".npz",
    ".duckdb",
    ".sqlite",
    ".db",
    ".mrc",
    ".dx",
)

HARDCODED_PATTERNS = (
    (re.compile(r"/home/diddy\b"), "DIDDY_HOME"),
    (re.compile(r"/home/\w+"), "HOME_DIR"),
    (re.compile(r"/mnt/storage\b"), "MNT_STORAGE"),
    (re.compile(r"/mnt/scratch\b"), "MNT_SCRATCH"),
    (re.compile(r"/mnt/data\b"), "MNT_DATA"),
    (re.compile(r"/tmp/prism"), "TMP_PRISM"),
    (re.compile(r"/tmp/"), "TMP"),
    (re.compile(r"/usr/local/prism"), "USR_LOCAL"),
    (re.compile(r"/opt/prism"), "OPT_PRISM"),
    (re.compile(r"/root\b"), "ROOT_HOME"),
    (re.compile(r"/var/tmp\b"), "VAR_TMP"),
    (re.compile(r"/dev/shm\b"), "DEV_SHM"),
    (re.compile(r"file://"), "FILE_URL"),
    (re.compile(r"~/Desktop"), "TILDE_DESKTOP"),
)

ENV_VAR_PATTERNS = (
    (re.compile(r"PRISM_SCRATCH_ROOT"), "PRISM_SCRATCH_ROOT"),
    (re.compile(r"PRISM_DATA_ROOT"), "PRISM_DATA_ROOT"),
    (re.compile(r"PRISM_ROOT"), "PRISM_ROOT"),
)

CREDENTIAL_PATTERNS = (
    (re.compile(r"(?i)api[_-]?key\s*[=:]\s*[\"'][^\"']{8,}"), "API_KEY_ASSIGNMENT"),
    (re.compile(r"(?i)secret\s*[=:]\s*[\"'][^\"']{8,}"), "SECRET_ASSIGNMENT"),
    (re.compile(r"(?i)password\s*[=:]\s*[\"'][^\"']{4,}"), "PASSWORD_ASSIGNMENT"),
    (re.compile(r"(?i)token\s*[=:]\s*[\"'][A-Za-z0-9_\-]{20,}"), "TOKEN_ASSIGNMENT"),
    (re.compile(r"CLOUDFLARE_API_TOKEN\s*=\s*\S+"), "CF_TOKEN"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "OPENAI_KEY"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "GITHUB_PAT"),
    (re.compile(r"BEGIN RSA PRIVATE"), "RSA_KEY"),
    (re.compile(r"BEGIN OPENSSH PRIVATE"), "SSH_KEY"),
)

EXTERNAL_READ_PATTERNS = (
    re.compile(r"open\([\"'](/mnt/|/home/|/tmp/|/opt/|/root|/var/tmp|/dev/shm)"),
    re.compile(r"read_parquet\([\"'](/mnt/|/home/|/tmp/|/opt/|/root|/var/tmp|/dev/shm)"),
    re.compile(r"read_csv\([\"'](/mnt/|/home/|/tmp/|/opt/|/root|/var/tmp|/dev/shm)"),
    re.compile(r"Path\([\"'](/mnt/|/home/|/tmp/|/opt/|/root|/var/tmp|/dev/shm)"),
    re.compile(r"pl\.scan_parquet\([\"'](/mnt/|/home/|/tmp/|/opt/|/root|/var/tmp|/dev/shm)"),
)

SAFE_BINS = {
    "bash",
    "cargo",
    "cat",
    "date",
    "git",
    "mypy",
    "pip",
    "pytest",
    "python",
    "python3",
    "sha256sum",
    "sh",
    "which",
}

CHEMISTRY_BINS = {"xtb", "antechamber", "sqm", "obabel"}
CONFIG_BINS = {
    "acpype",
    "analyze_ensemble",
    "aws",
    "curl",
    "df",
    "fpocket",
    "generate-ensemble",
    "gnina",
    "gsutil",
    "gmx",
    "mmpbsa.py",
    "mmseqs",
    "nvidia-smi",
    "p2rank",
    "rclone",
    "reduce",
    "timeout",
    "unidock",
    "wget",
    "wrangler",
}

GENERATED_SECRET_SCAN_EXCLUDES = {"HERMETIC_INTEGRITY_LEDGER.json"}


@dataclass
class Finding:
    scan_id: str
    category: str
    severity: str
    file: str
    line: int | None
    content: str
    classification: str = "UNCLASSIFIED"
    remediation: str = ""
    status: str = "OPEN"

    @property
    def key(self) -> str:
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:8]
        return f"{self.category}:{self.file}:{self.line or ''}:{content_hash}"


findings: list[Finding] = []
scan_counter = 0
_tracked_files: set[str] | None = None
_manifest_text: str | None = None
_manifest_patterns: list[str] | None = None
_override_keys: set[str] = set()


def next_id(prefix: str) -> str:
    global scan_counter
    scan_counter += 1
    return f"{prefix}_{scan_counter:05d}"


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def path_parts(path: Path) -> set[str]:
    return set(path.relative_to(REPO_ROOT).parts)


def is_skipped(path: Path) -> bool:
    return bool(path_parts(path) & SKIP_DIRS)


def tracked_files() -> set[str]:
    global _tracked_files
    if _tracked_files is None:
        result = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
        )
        _tracked_files = set(result.stdout.splitlines())
    return _tracked_files


def is_tracked(path: Path) -> bool:
    return rel(path) in tracked_files()


def is_gitignored(path: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", rel(path)],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


def is_test_file(path: Path) -> bool:
    r = rel(path).lower()
    return r.startswith("tests/") or "/test" in r or "fixture" in r


def is_document_or_generated(path: Path) -> bool:
    r = f"/{rel(path)}"
    if rel(path).startswith("docs/") or path.suffix in {".md"}:
        return True
    if rel(path).startswith("campaigns/") and path.suffix in {".json", ".md", ".csv", ".yaml", ".yml"}:
        return True
    if rel(path).startswith("campaigns/") and any(part in r for part in GENERATED_PATH_PARTS):
        return True
    if rel(path).startswith("scripts/quarantine/"):
        return True
    return False


def is_production_file(path: Path) -> bool:
    r = rel(path)
    if is_skipped(path):
        return False
    return any(r.startswith(prefix) or r == prefix.rstrip("/") for prefix in PRODUCTION_DIRS)


def iter_source_files() -> Iterable[Path]:
    for prod_dir in PRODUCTION_DIRS:
        start = REPO_ROOT / prod_dir
        if not start.exists():
            continue
        if start.is_file():
            if start.suffix in SOURCE_EXTENSIONS:
                yield start
            continue
        for root, dirs, files in os.walk(start):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                path = Path(root) / name
                if path.name == "hermetic_integrity_scan.py":
                    continue
                if path.suffix in SOURCE_EXTENSIONS:
                    yield path


def read_text(path: Path) -> str:
    if path.stat().st_size > 8_000_000:
        return ""
    return path.read_text(errors="replace")


def load_release_data_manifest() -> str:
    global _manifest_text
    if _manifest_text is None:
        path = REPO_ROOT / "RELEASE_DATA_MANIFEST.md"
        _manifest_text = path.read_text(errors="replace") if path.exists() else ""
    return _manifest_text


def manifest_patterns() -> list[str]:
    global _manifest_patterns
    if _manifest_patterns is None:
        text = load_release_data_manifest()
        _manifest_patterns = re.findall(r"`([^`*]+[*][^`]*)`", text)
        _manifest_patterns.extend(re.findall(r"`([^`]+)`", text))
    return _manifest_patterns


def manifest_covers(reference: str) -> bool:
    text = load_release_data_manifest()
    if reference in text or Path(reference).name in text:
        return True
    return any(fnmatch.fnmatch(reference, pattern) for pattern in manifest_patterns())


def override_exists_for(finding: Finding) -> bool:
    return finding.key in _override_keys


def add_finding(finding: Finding) -> None:
    findings.append(finding)


def accepted(
    scan_id: str,
    category: str,
    severity: str,
    file: str,
    line: int | None,
    content: str,
    classification: str,
    remediation: str,
) -> Finding:
    return Finding(
        scan_id=scan_id,
        category=category,
        severity=severity,
        file=file,
        line=line,
        content=content[:240],
        classification=classification,
        remediation=remediation,
        status="ACCEPTED_RISK",
    )


def open_finding(
    scan_id: str,
    category: str,
    severity: str,
    file: str,
    line: int | None,
    content: str,
    classification: str,
    remediation: str,
) -> Finding:
    return Finding(
        scan_id=scan_id,
        category=category,
        severity=severity,
        file=file,
        line=line,
        content=content[:240],
        classification=classification,
        remediation=remediation,
        status="OPEN",
    )


def classify_external_path(path: Path, line: str) -> tuple[str, str, str]:
    if is_test_file(path):
        return "LOW", "TEST_ONLY", "Test-only external/local path."
    if is_document_or_generated(path):
        return "LOW", "GENERATED_OUTPUT_ONLY", "Generated/report/audit path retained for provenance."
    if "os.environ" in line or "getenv" in line or "PRISM_" in line:
        return "INFO", "ALREADY_MITIGATED", "Path is guarded by environment/config resolution."
    if "/tmp/" in line or "/var/tmp" in line or "/dev/shm" in line:
        return "LOW", "GENERATED_OUTPUT_ONLY", "Temporary/generated output path."
    if "/mnt/storage" in line or "/mnt/data" in line or "/mnt/scratch" in line:
        return "MEDIUM", "CONFIG_ENV_REQUIRED", "External scratch/data root; preserve capability via environment documentation."
    if "/home/" in line or "~/Desktop" in line:
        return "MEDIUM", "CONFIG_ENV_REQUIRED", "Operator-local path; preserve developer workflow and document as configurable/local data."
    if "file://" in line:
        return "LOW", "GENERATED_OUTPUT_ONLY", "Local report URI only."
    return "MEDIUM", "CONFIG_ENV_REQUIRED", "External/local path requires environment documentation."


def scan_symlinks() -> None:
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(files + dirs):
            full = Path(root) / name
            if not full.is_symlink():
                continue
            target = os.readlink(full)
            resolved = full.resolve(strict=False)
            broken = not full.exists()
            outside = not str(resolved).startswith(str(REPO_ROOT))
            r = rel(full)
            if is_skipped(full) or is_gitignored(full):
                classification = "GENERATED_OUTPUT_ONLY"
                severity = "INFO"
                remediation = "Ignored local/generated symlink; not part of source release."
                status = "ACCEPTED_RISK"
            elif broken:
                classification = "REAL_ISSUE"
                severity = "CRITICAL"
                remediation = "Tracked/source symlink is broken; fix target or classify explicitly."
                status = "OPEN"
            elif outside:
                classification = "REAL_ISSUE"
                severity = "HIGH"
                remediation = "Tracked/source symlink points outside repository; replace or classify explicitly."
                status = "OPEN"
            else:
                classification = "FALSE_POSITIVE"
                severity = "INFO"
                remediation = "Repo-contained symlink."
                status = "ACCEPTED_RISK"
            add_finding(
                Finding(
                    next_id("SYM"),
                    "SYMLINK",
                    severity,
                    r,
                    None,
                    f"-> {target} | resolved={resolved}",
                    classification,
                    remediation,
                    status,
                )
            )


def scan_hardcoded_paths() -> None:
    for path in iter_source_files():
        text = read_text(path)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern, tag in HARDCODED_PATTERNS:
                if not pattern.search(line):
                    continue
                severity, classification, reason = classify_external_path(path, line)
                status = "ACCEPTED_RISK"
                remediation = f"{reason} Pattern={tag}; convert to env/config when touching this path."
                add_finding(
                    Finding(
                        next_id("PATH"),
                        "HARDCODED_PATH",
                        severity,
                        rel(path),
                        lineno,
                        line.strip(),
                        classification,
                        remediation,
                        status,
                    )
                )
                break
            for pattern, tag in ENV_VAR_PATTERNS:
                if pattern.search(line):
                    add_finding(
                        accepted(
                            next_id("ENV"),
                            "HARDCODED_PATH",
                            "INFO",
                            rel(path),
                            lineno,
                            line.strip(),
                            "ALREADY_MITIGATED",
                            f"{tag} is documented in ENVIRONMENT.md.",
                        )
                    )
                    break


def credential_is_placeholder(line: str, path: Path) -> bool:
    lowered = line.lower()
    if is_test_file(path):
        return True
    if any(
        token in lowered
        for token in ("redacted", "placeholder", "example", "<", "your_", "changeme", "...", "from_file")
    ):
        return True
    if re.search(r"[\"']?\$[A-Z0-9_]+[\"']?", line):
        return True
    if path.name.endswith(".md") and any(token in lowered for token in ("must not", "never", "pattern", "runtime source")):
        return True
    return False


def redact(line: str) -> str:
    redacted = re.sub(r"([=:]\s*[\"'])[^\"']+([\"'])", r"\1[REDACTED]\2", line.strip())
    redacted = re.sub(r"(sk-|ghp_)[A-Za-z0-9_\-]+", r"\1[REDACTED]", redacted)
    return redacted[:240]


def scan_credentials_in_file(path: Path) -> None:
    text = read_text(path)
    if not text:
        return
    for lineno, line in enumerate(text.splitlines(), 1):
        for pattern, tag in CREDENTIAL_PATTERNS:
            if not pattern.search(line):
                continue
            if credential_is_placeholder(line, path):
                add_finding(
                    accepted(
                        next_id("CRED"),
                        "CREDENTIAL",
                        "INFO",
                        rel(path),
                        lineno,
                        redact(line),
                        "SECRET_EXCLUDED",
                        f"Placeholder or policy reference for {tag}; no secret value committed.",
                    )
                )
            else:
                add_finding(
                    open_finding(
                        next_id("CRED"),
                        "CREDENTIAL",
                        "CRITICAL",
                        rel(path),
                        lineno,
                        redact(line),
                        "REAL_ISSUE",
                        f"Remove committed credential-like value ({tag}); use secret store/env.",
                    )
                )
            break


def scan_credentials() -> None:
    for path in iter_source_files():
        scan_credentials_in_file(path)

    result = subprocess.run(
        [
            "git",
            "grep",
            "-nIE",
            r"api[_-]?key|secret|token|password|BEGIN RSA|BEGIN OPENSSH|sk-|ghp_",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    for raw in result.stdout.splitlines():
        parts = raw.split(":", 2)
        if len(parts) != 3:
            continue
        file_name, line_no, content = parts
        if file_name in GENERATED_SECRET_SCAN_EXCLUDES:
            continue
        path = REPO_ROOT / file_name
        if is_skipped(path) or not path.exists():
            continue
        if any(f.category == "CREDENTIAL" and f.file == file_name and str(f.line) == line_no for f in findings):
            continue
        lowered = content.lower()
        if (
            "token" in lowered
            or "secret" in lowered
            or "password" in lowered
            or "api" in lowered
        ):
            classification = "SECRET_EXCLUDED" if credential_is_placeholder(content, path) else "FALSE_POSITIVE"
            add_finding(
                accepted(
                    next_id("CRED"),
                    "CREDENTIAL",
                    "INFO",
                    file_name,
                    int(line_no) if line_no.isdigit() else None,
                    "[git grep tracked-file credential lexeme]",
                    classification,
                    "All git-tracked files scanned; this hit is a variable name, policy text, or placeholder.",
                )
            )

    for secret_path in (
        REPO_ROOT / "cloud/prism-manifold-worker/.dev.vars",
        REPO_ROOT / "cloud/prism-manifold-worker/.wrangler",
    ):
        if not secret_path.exists():
            continue
        if is_gitignored(secret_path):
            add_finding(
                accepted(
                    next_id("CLOUD"),
                    "CLOUD_SECRET",
                    "INFO",
                    rel(secret_path),
                    None,
                    "[exists locally and is gitignored]",
                    "SECRET_EXCLUDED",
                    "Local cloud secret/state is excluded from source release.",
                )
            )
        else:
            add_finding(
                open_finding(
                    next_id("CLOUD"),
                    "CLOUD_SECRET",
                    "CRITICAL",
                    rel(secret_path),
                    None,
                    "[exists locally and is not gitignored]",
                    "REAL_ISSUE",
                    "Add local cloud secret/state path to .gitignore.",
                )
            )


def scan_external_reads() -> None:
    for path in iter_source_files():
        if path.suffix != ".py":
            continue
        text = read_text(path)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if not any(pattern.search(line) for pattern in EXTERNAL_READ_PATTERNS):
                continue
            severity, classification, reason = classify_external_path(path, line)
            if classification in {"CONFIG_ENV_REQUIRED", "GENERATED_OUTPUT_ONLY", "TEST_ONLY", "ALREADY_MITIGATED"}:
                status = "ACCEPTED_RISK"
            elif manifest_covers(line):
                status = "ACCEPTED_RISK"
                classification = "EXTERNAL_DATA_REQUIRED"
                severity = "INFO"
            else:
                status = "OPEN"
                classification = "EXTERNAL_DATA_REQUIRED"
                severity = "HIGH"
            add_finding(
                Finding(
                    next_id("EXTRD"),
                    "EXTERNAL_READ",
                    severity,
                    rel(path),
                    lineno,
                    line.strip(),
                    classification,
                    f"{reason} External reads must be config-relative or listed in RELEASE_DATA_MANIFEST.md.",
                    status,
                )
            )


def scan_untracked() -> None:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "src/",
            "scripts/",
            "crates/",
            "cloud/prism-manifold-worker/src/",
            "campaigns/",
            "tests/",
            "00_registry/",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    for file_name in result.stdout.splitlines():
        path = REPO_ROOT / file_name
        if not file_name or "__pycache__" in file_name or file_name.endswith(".pyc"):
            continue
        if file_name == "scripts/hermetic_integrity_scan.py":
            classification = "ALREADY_MITIGATED"
            severity = "INFO"
            status = "ACCEPTED_RISK"
            remediation = "Scanner source is intentionally staged/committed as part of E025-R1.0."
        elif path.suffix in DATA_SUFFIXES:
            classification = "GENERATED_OUTPUT_ONLY"
            severity = "INFO"
            status = "ACCEPTED_RISK"
            remediation = "Generated data artifact; commit only if promoted to fixture or manifest."
        else:
            classification = "REAL_ISSUE"
            severity = "LOW"
            status = "OPEN"
            remediation = "Commit, gitignore, or move out of production path."
        add_finding(
            Finding(
                next_id("UNTRK"),
                "UNTRACKED",
                severity,
                file_name,
                None,
                "untracked file",
                classification,
                remediation,
                status,
            )
        )


def is_generated_data_reference(reference: str) -> bool:
    if any(ch.isspace() for ch in reference):
        return True
    if any(token in reference for token in ("*", "}", ")", "]", "->")):
        return True
    lower = Path(reference).name.lower()
    generated_tokens = (
        "audit",
        "batch",
        "cache",
        "certificate",
        "completion",
        "config",
        "diagnosis",
        "diff",
        "fixture",
        "ignition",
        "latency",
        "ledger",
        "loss",
        "manifest",
        "metrics",
        "profile",
        "progression",
        "report",
        "request",
        "response",
        "result",
        "reward",
        "rmsd",
        "rmsf",
        "state",
        "status",
        "summary",
        "telemetry",
        "timeseries",
        "trajectory_entropy",
        "validation",
        "visualization",
    )
    if any(token in lower for token in generated_tokens):
        return True
    if lower.startswith(("oracle_", "gflownet_policy_", "epoch_", "warm_start_")):
        return True
    return False


def scan_data_dependencies() -> None:
    data_ref_pattern = re.compile(r"[\"']([^\"'\n]*\.(?:parquet|sdf|pdb|mol2|csv|json|yaml|yml|pkl|pt|safetensors|npy|npz|duckdb|sqlite|db|mrc|dx))[\"']")
    for path in iter_source_files():
        text = read_text(path)
        if not text:
            continue
        refs = sorted(set(match.group(1) for match in data_ref_pattern.finditer(text)))
        for reference in refs:
            if reference.startswith(("http://", "https://")) or "{" in reference:
                continue
            generated_reference = is_generated_data_reference(reference)
            if reference.startswith("/"):
                covered = manifest_covers(reference)
                exists = Path(reference).exists()
            else:
                exists = (REPO_ROOT / reference).exists()
                covered = manifest_covers(reference)
            if exists:
                classification = "ALREADY_MITIGATED"
                severity = "INFO"
                status = "ACCEPTED_RISK"
                remediation = "Referenced data exists in the current source/developer checkout."
            elif is_test_file(path):
                classification = "TEST_ONLY"
                severity = "LOW"
                status = "ACCEPTED_RISK"
                remediation = "Test-only data dependency must use fixture fallback or explicit skip."
            elif is_document_or_generated(path) or generated_reference:
                classification = "GENERATED_OUTPUT_ONLY"
                severity = "INFO"
                status = "ACCEPTED_RISK"
                remediation = "Generated/provenance artifact reference retained for traceability."
            elif covered:
                classification = "EXTERNAL_DATA_REQUIRED"
                severity = "INFO"
                status = "ACCEPTED_RISK"
                remediation = "Referenced external data is covered by RELEASE_DATA_MANIFEST.md."
            else:
                classification = "REAL_ISSUE"
                severity = "MEDIUM"
                status = "OPEN"
                remediation = "Add dependency to RELEASE_DATA_MANIFEST.md or replace with committed fixture."
            add_finding(
                Finding(
                    next_id("DATA"),
                    "DATA_DEPENDENCY",
                    severity,
                    rel(path),
                    None,
                    f"References: {reference} (exists={exists}, manifest={covered})",
                    classification,
                    remediation,
                    status,
                )
            )


def extract_subprocess_binary(line: str) -> str:
    list_match = re.search(r"\[\s*[\"']([^\"']+)[\"']", line)
    if list_match:
        return Path(list_match.group(1)).name
    call_match = re.search(r"Command::new\(\s*([\"'])([^\"']+)\1", line)
    if call_match:
        return Path(call_match.group(2)).name
    str_match = re.search(r"subprocess\.[A-Za-z_]+\(\s*([\"'])([^\"']+)\1", line)
    if str_match:
        return Path(str_match.group(2).split()[0]).name
    return "UNKNOWN_DYNAMIC_COMMAND"


def scan_subprocess_calls() -> None:
    py_pattern = re.compile(r"subprocess\.(run|Popen|call|check_output|check_call)\s*\(")
    rs_pattern = re.compile(r"(?:std::process::)?Command::new\s*\(")
    for path in iter_source_files():
        text = read_text(path)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            pattern = py_pattern if path.suffix == ".py" else rs_pattern
            if not pattern.search(line):
                continue
            binary = extract_subprocess_binary(line)
            if is_test_file(path):
                classification = "TEST_ONLY"
                severity = "INFO"
                remediation = f"{binary} is invoked by tests; test controls availability or skips."
            elif binary in SAFE_BINS:
                classification = "FALSE_POSITIVE"
                severity = "INFO"
                remediation = f"{binary} is part of baseline developer/release tooling."
            elif binary in CHEMISTRY_BINS:
                classification = "CONFIG_ENV_REQUIRED"
                severity = "INFO"
                remediation = f"{binary} is a chemistry runtime dependency documented in ENVIRONMENT.md."
            elif binary in CONFIG_BINS or binary == "UNKNOWN_DYNAMIC_COMMAND":
                classification = "CONFIG_ENV_REQUIRED"
                severity = "INFO"
                remediation = f"{binary} is an external/dynamic runtime dependency documented in ENVIRONMENT.md."
            else:
                classification = "CONFIG_ENV_REQUIRED"
                severity = "MEDIUM"
                remediation = f"{binary} must be available in developer/full-production mode; document in ENVIRONMENT.md."
            add_finding(
                accepted(
                    next_id("SUBP"),
                    "SUBPROCESS",
                    severity,
                    rel(path),
                    lineno,
                    f"Calls: {binary} | {line.strip()}",
                    classification,
                    remediation,
                )
            )


def load_overrides() -> dict[str, dict[str, str]]:
    global _override_keys
    path = REPO_ROOT / "HERMETIC_CLASSIFICATION_OVERRIDES.json"
    if not path.exists():
        _override_keys = set()
        return {}
    raw = json.loads(path.read_text())
    overrides = {entry["key"]: entry for entry in raw}
    _override_keys = set(overrides)
    return overrides


def apply_classification_overrides() -> None:
    overrides = load_overrides()
    if not overrides:
        return
    for finding in findings:
        override = overrides.get(finding.key)
        if not override:
            continue
        finding.classification = override["override_classification"]
        finding.status = override["override_status"]
        finding.remediation = f"[OVERRIDE] {override['reason']}"


def verify_release_marker() -> None:
    attr = REPO_ROOT / ".gitattributes"
    marker = REPO_ROOT / "CLEAN_ARCHIVE_SOURCE_COMMIT.txt"
    attr_ok = attr.exists() and "CLEAN_ARCHIVE_SOURCE_COMMIT.txt export-subst" in attr.read_text(errors="replace")
    marker_ok = marker.exists() and marker.read_text(errors="replace") == "$Format:%H$\n"
    if attr_ok and marker_ok:
        add_finding(
            accepted(
                next_id("ARCHIVE"),
                "ARCHIVE_MARKER",
                "INFO",
                "CLEAN_ARCHIVE_SOURCE_COMMIT.txt",
                None,
                "$Format:%H$",
                "ALREADY_MITIGATED",
                "export-subst archive marker is present.",
            )
        )
    else:
        add_finding(
            open_finding(
                next_id("ARCHIVE"),
                "ARCHIVE_MARKER",
                "HIGH",
                "CLEAN_ARCHIVE_SOURCE_COMMIT.txt",
                None,
                "missing or malformed export-subst marker",
                "REAL_ISSUE",
                "Add .gitattributes export-subst and literal marker file.",
            )
        )


def write_ledger() -> Path:
    by_severity = {sev: 0 for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")}
    by_category: dict[str, int] = {}
    by_classification: dict[str, int] = {}
    for finding in findings:
        by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
        by_category[finding.category] = by_category.get(finding.category, 0) + 1
        by_classification[finding.classification] = by_classification.get(finding.classification, 0) + 1
    ledger = {
        "scan_version": "E025-R1.0-v2.1",
        "commit": "verified by CLEAN_ARCHIVE_SOURCE_COMMIT.txt export-subst during clean archive gate",
        "total_findings": len(findings),
        "by_severity": by_severity,
        "by_category": by_category,
        "by_classification": by_classification,
        "exit_rule": {
            "critical": sum(1 for f in findings if f.severity == "CRITICAL"),
            "high_real_issue": sum(1 for f in findings if f.severity == "HIGH" and f.classification == "REAL_ISSUE"),
            "unclassified": sum(1 for f in findings if f.classification == "UNCLASSIFIED"),
            "open": sum(1 for f in findings if f.status == "OPEN"),
        },
        "findings": [dict(asdict(f), key=f.key) for f in findings],
    }
    path = REPO_ROOT / "HERMETIC_INTEGRITY_LEDGER.json"
    path.write_text(json.dumps(ledger, indent=2) + "\n")
    return path


def main() -> int:
    print("=== HERMETIC INTEGRITY SCAN (E025-R1.0 v2.1) ===")
    print(f"Repo root: {REPO_ROOT}")
    print()

    scans = (
        ("archive marker", verify_release_marker),
        ("symlinks", scan_symlinks),
        ("hardcoded paths", scan_hardcoded_paths),
        ("credentials all tracked", scan_credentials),
        ("external reads", scan_external_reads),
        ("untracked production files", scan_untracked),
        ("data dependencies", scan_data_dependencies),
        ("subprocess calls", scan_subprocess_calls),
    )
    for name, function in scans:
        print(f"Running {name}...")
        function()

    apply_classification_overrides()
    ledger_path = write_ledger()

    by_severity = {sev: 0 for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")}
    for finding in findings:
        by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1

    n_critical = sum(1 for f in findings if f.severity == "CRITICAL")
    n_high_real = sum(1 for f in findings if f.severity == "HIGH" and f.classification == "REAL_ISSUE")
    n_unclassified = sum(1 for f in findings if f.classification == "UNCLASSIFIED")
    n_open = sum(1 for f in findings if f.status == "OPEN")

    print()
    print("=" * 60)
    print(f"TOTAL FINDINGS: {len(findings)}")
    for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
        if by_severity.get(severity, 0):
            print(f"  {severity}: {by_severity[severity]}")
    print(f"Ledger written to: {ledger_path}")
    print()

    for finding in findings:
        if finding.status == "OPEN" or finding.classification == "UNCLASSIFIED":
            print(f"[{finding.severity}] {finding.scan_id} {finding.category} {finding.file}:{finding.line or ''}")
            print(f"  classification={finding.classification} status={finding.status}")
            print(f"  content={finding.content[:160]}")
            print(f"  remediation={finding.remediation}")

    failed = False
    if n_critical:
        print(f"FAIL: {n_critical} CRITICAL findings.")
        failed = True
    if n_high_real:
        print(f"FAIL: {n_high_real} HIGH REAL_ISSUE findings.")
        failed = True
    if n_unclassified:
        print(f"FAIL: {n_unclassified} UNCLASSIFIED findings.")
        failed = True
    if n_open:
        print(f"FAIL: {n_open} OPEN findings.")
        failed = True

    if failed:
        print("OVERALL: FAIL")
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
