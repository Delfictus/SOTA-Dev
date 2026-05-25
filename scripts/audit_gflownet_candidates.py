#!/usr/bin/env python3
"""Compatibility entrypoint for the Track A GFlowNet candidate audit."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_gflownet_failure_modes import main


if __name__ == "__main__":
    sys.exit(main())
