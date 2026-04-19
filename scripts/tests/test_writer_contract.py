"""CI test: no INSERT OR REPLACE on site_features in any active writer.

Pinned to v4 feature-service hardening contract, §1 writer ownership.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ACTIVE_WRITERS = [
    REPO / "cloudflare/workers/feature-pipeline/src/index.js",
    REPO / "scripts/training/post_campaign_analysis.py",
    REPO / "scripts/training/add_temporal_to_npz.py",
]
PAT = re.compile(r"INSERT\s+OR\s+REPLACE\s+INTO\s+site_features", re.IGNORECASE)


def test_no_insert_or_replace_on_site_features():
    violations = []
    for p in ACTIVE_WRITERS:
        assert p.exists(), f"active writer missing: {p}"
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            if PAT.search(line):
                violations.append(f"{p.relative_to(REPO)}:{ln}: {line.strip()}")
    assert not violations, \
        "INSERT OR REPLACE on site_features is forbidden:\n  " + "\n  ".join(violations)


def test_worker_uses_insert_or_ignore_plus_update():
    text = (REPO / "cloudflare/workers/feature-pipeline/src/index.js").read_text()
    assert "INSERT OR IGNORE INTO site_features" in text
    assert "UPDATE site_features SET" in text


def test_populate_d1_retired():
    p = REPO / "cloudflare/d1/populate_d1.py"
    if not p.exists():
        return   # absent is also acceptable
    first_line = p.read_text().splitlines()[0] if p.read_text().splitlines() else ""
    assert "# RETIRED" in first_line, \
        "populate_d1.py must have '# RETIRED' header; current first line: " + first_line
