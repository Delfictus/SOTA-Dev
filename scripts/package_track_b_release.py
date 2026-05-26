#!/usr/bin/env python3
"""Package Track B release artifacts into a tarball."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    release_root = "PRISM4D_TRACK_B_TRANSLATIONAL_CALIBRATION_RELEASE_v1"
    with tarfile.open(args.output, "w:gz") as tar:
        tar.add(args.track_b_root, arcname=release_root)
        chronology_tensor = Path(
            "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/transition_chronology_tensor.parquet"
        )
        if chronology_tensor.exists():
            tar.add(chronology_tensor, arcname=f"{release_root}/transition_chronology_tensor.parquet")
        oracle_binary = Path("target/release/oracle_scorer")
        if oracle_binary.exists():
            tar.add(oracle_binary, arcname=f"{release_root}/runtime/bin/oracle_scorer")
        audit_root = Path(".audit-reports")
        if audit_root.exists():
            for report in sorted(audit_root.glob("track_b_*")):
                if report.is_file():
                    tar.add(report, arcname=f"{release_root}/subagent_reports/{report.name}")
    print(f"track_b_release_packaged output={args.output}")


if __name__ == "__main__":
    main()
