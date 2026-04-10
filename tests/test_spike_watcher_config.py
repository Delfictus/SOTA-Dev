"""Minimal config validation for prism_spike_watcher.py WATCH_DIRS and R2_ROUTING."""
import ast
import re
from pathlib import Path

WATCHER_PATH = Path(__file__).parent.parent / "scripts" / "prism_spike_watcher.py"


def _load_config():
    source = WATCHER_PATH.read_text()
    tree = ast.parse(source)
    ns = {}
    exec(compile(tree, str(WATCHER_PATH), "exec"), ns)
    return ns


def test_syntax_valid():
    ast.parse(WATCHER_PATH.read_text())


def test_watch_dirs_contains_required_paths():
    ns = _load_config()
    dirs = ns["WATCH_DIRS"]
    assert "/mnt/storage/prism-outputs/runs" in dirs
    assert "/mnt/storage/prism-outputs/twin-runs" in dirs
    assert "/tmp" in dirs


def test_r2_routing_tmp_regex():
    ns = _load_config()
    routing = ns["R2_ROUTING"]
    assert any("tmp" in k for k in routing), "R2_ROUTING must have a /tmp rule"
    for pattern in routing:
        if "tmp" in pattern:
            assert re.search(pattern, "/tmp/some_output"), f"Pattern {pattern} must match /tmp/some_output"
            bucket, prefix = routing[pattern]
            assert bucket == "prism-archive"
            assert prefix == "dev-runs"
