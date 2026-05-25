#!/usr/bin/env python3
"""Stage the Amylin receptor campaign manifest for next-SOW authorization."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN_DIR = REPO_ROOT / "campaigns/amyr_calcitonin_combo"
DEFAULT_OUTPUT = DEFAULT_CAMPAIGN_DIR / "campaign_init_manifest.json"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_manifest(campaign_dir: Path) -> JsonObject:
    return {
        "schema_version": "PRISM.amyr_campaign_init.v1",
        "campaign_id": "amyr_calcitonin_combo",
        "target": "AMYR_CTR_RAMP1_Complex",
        "target_components": ["Calcitonin_Receptor", "RAMP1"],
        "status": "staged_pending_sow_authorization",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_dir": campaign_dir.relative_to(REPO_ROOT).as_posix(),
        "commercial_context": "Structure Therapeutics oral Amylin and GLP-1R / Amylin combination expansion readiness",
        "rationale": (
            "Staged for multi-receptor cross-talk analysis to support next-generation GLP-1R / "
            "Amylin co-agonism and combination therapy design."
        ),
        "planned_outputs": [
            "amyr_ctr_ramp1_topology_manifest.json",
            "amyr_signal_grid_variance_channel.parquet",
            "amyr_cross_talk_durability_map.parquet",
            "glp1r_amyr_combo_interference_report.md",
        ],
        "execution_boundary": (
            "This manifest stages infrastructure only. No AMYR simulation or biological claim is made "
            "until SOW authorization and topology approval."
        ),
    }


def main() -> int:
    args = parse_args()
    campaign_dir = Path(args.campaign_dir)
    output = Path(args.output)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(campaign_dir)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
