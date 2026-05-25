#!/usr/bin/env python3
"""Render the one-page enterprise positioning summary."""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined


CAMPAIGN_DIR = Path("campaigns/glp1r_aleniglipron")
TEMPLATE_DIR = Path("00_registry/templates")
DEFAULT_TEMPLATE = TEMPLATE_DIR / "enterprise_positioning_summary.md.j2"
DEFAULT_OUTPUT = CAMPAIGN_DIR / "ENTERPRISE_POSITIONING_SUMMARY.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--epistemic-contract", type=Path, default=Path("00_registry/epistemic_contract.yml"))
    parser.add_argument("--cbom", type=Path, default=CAMPAIGN_DIR / "PRISM_CBOM_v1.0.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    yaml_module = cast(Any, import_module("yaml"))
    loaded = yaml_module.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected mapping in {path}")
    return cast(dict[str, Any], loaded)


def epistemic_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    classes = contract.get("classes", {})
    if not isinstance(classes, dict):
        raise ValueError("epistemic contract classes must be a mapping")
    rows: list[dict[str, Any]] = []
    for name, payload in classes.items():
        if not isinstance(payload, dict):
            continue
        rows.append({"name": str(name), "level": payload.get("level"), "definition": payload.get("definition", "")})
    return sorted(rows, key=lambda row: int(row["level"]))


def render(args: argparse.Namespace) -> str:
    contract = load_yaml(args.epistemic_contract)
    cbom = json.loads(args.cbom.read_text(encoding="utf-8")) if args.cbom.exists() else {}
    env = Environment(loader=FileSystemLoader(str(args.template.parent)), undefined=StrictUndefined, autoescape=False)
    template = env.get_template(args.template.name)
    return template.render(
        epistemic_classes=epistemic_rows(contract),
        campaign_merkle_root=str(cbom.get("campaign_merkle_root", "CBOM_NOT_RENDERED")),
    )


def main() -> None:
    args = parse_args()
    rendered = render(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
