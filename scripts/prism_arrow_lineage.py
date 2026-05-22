"""Compatibility wrapper for PRISM-DSTW Arrow-native lineage helpers."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_dstw.io import dependency_versions, sha256_path, write_provenance_parquet  # noqa: F401
