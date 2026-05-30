#!/usr/bin/env python3
"""Audit Tier 3 PoV target authority without falsely promoting unresolved lanes."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecars_for(topology_path: Path) -> dict[str, Path]:
    base = topology_path.name.removesuffix(".topology.json")
    return {
        "residue_map": topology_path.with_name(f"{base}.residue_map.json"),
        "atom_to_residue": topology_path.with_name(f"{base}.atom_to_residue.json"),
        "nma_modes": topology_path.with_name(f"{base}_nma_modes.json"),
    }


def summarize_topology(topology_path: Path) -> dict[str, Any]:
    if not topology_path.exists():
        return {"exists": False}
    body = load_json(topology_path)
    return {
        "exists": True,
        "size_bytes": topology_path.stat().st_size,
        "sha256": sha256(topology_path),
        "n_atoms": body.get("n_atoms"),
        "n_residues": body.get("n_residues"),
        "hmr_applied": body.get("hmr_applied"),
        "recommended_timestep_fs": body.get("recommended_timestep_fs"),
    }


def validate_topology(topology_path: Path) -> dict[str, Any]:
    command = [
        "python3",
        "scripts/validate_topology.py",
        str(topology_path),
        "--quiet",
    ]
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "command": " ".join(command),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    authority_path = run_root / "loop0/authority_manifest.json"
    authority = load_json(authority_path)
    targets: dict[str, Any] = authority["targets"]

    audited = []
    pending = []
    failing = []
    rows = []

    for target_id, payload in sorted(targets.items()):
        topology_path = Path(payload["path"])
        summary = summarize_topology(topology_path)
        validation = validate_topology(topology_path) if summary["exists"] else None
        sidecars = {}
        sidecars_complete = True
        for name, sidecar in sidecars_for(topology_path).items():
            present = sidecar.exists()
            sidecars[name] = {
                "path": str(sidecar),
                "exists": present,
                "size_bytes": sidecar.stat().st_size if present else None,
                "sha256": sha256(sidecar) if present else None,
            }
            sidecars_complete = sidecars_complete and present

        if not summary["exists"]:
            status = "FAIL_TOPOLOGY_MISSING"
            remediation = "restore_or_rebuild_topology_bundle"
            failing.append(target_id)
        elif validation and validation["returncode"] != 0:
            status = "FAIL_TOPOLOGY_VALIDATION"
            remediation = "repair_topology_payload_until_validate_topology_passes"
            failing.append(target_id)
        elif payload.get("in_phase3_runnable_index"):
            status = "PASS_AUDITED_PHASE3"
            remediation = None
            audited.append(target_id)
        elif payload.get("in_expanded_exact_runnable_index"):
            status = "PASS_AUDITED_EXPANDED_EXACT"
            remediation = None
            audited.append(target_id)
        elif payload.get("supplemental_audit_allowed") and sidecars_complete:
            status = "PASS_SUPPLEMENTAL_CONTROL_AUDITED"
            remediation = None
            audited.append(target_id)
        elif payload["promotion_required"] and sidecars_complete:
            status = "READY_PENDING_AUDIT"
            remediation = "run_index_promotion_audit_and_add_to_audited_runnable_set"
            pending.append(target_id)
        elif not sidecars_complete:
            status = "FAIL_SIDECARS_INCOMPLETE"
            remediation = "regenerate_missing_sidecars_before_promotion"
            failing.append(target_id)
        else:
            status = "PASS_PRESENT_WITHOUT_PROMOTION_REQUIREMENT"
            remediation = None
            audited.append(target_id)

        rows.append(
            {
                "target_id": target_id,
                "authority_tier": payload["authority_tier"],
                "in_phase3_runnable_index": payload["in_phase3_runnable_index"],
                "promotion_required": payload["promotion_required"],
                "status": status,
                "remediation": remediation,
                "panel_role": payload["panel_role"],
                "source_regime": payload["source_regime"],
                "topology": summary,
                "validation": validation,
                "sidecars": sidecars,
                "notes": payload["notes"],
            }
        )

    if failing:
        gate_status = "FAIL"
        verdict = "BLOCKED_BY_MISSING_OR_INCOMPLETE_TARGET_ARTIFACTS"
    elif pending:
        gate_status = "PARTIAL"
        verdict = "READY_PENDING_FORMAL_PROMOTION_AUDIT"
    else:
        gate_status = "PASS"
        verdict = "ALL_TARGETS_AUDITED_OR_ACCEPTABLY_CLASSIFIED"

    audit = {
        "schema_version": "prism.tier3_pov.loop0_authority_promotion_audit.v1",
        "generated_at_utc": now_utc(),
        "run_root": str(run_root),
        "rows": rows,
        "audited_count": len(audited),
        "pending_count": len(pending),
        "failing_count": len(failing),
        "pending_targets": pending,
        "failing_targets": failing,
        "gate_status": gate_status,
        "verdict": verdict,
    }
    gate = {
        "schema_version": "prism.tier3_pov.loop0_authority_gate_decision.v1",
        "generated_at_utc": now_utc(),
        "status": gate_status,
        "verdict": verdict,
        "blocking_targets": failing,
        "pending_promotion_targets": pending,
        "required_next_actions": [
            "Do not upgrade READY_PENDING_AUDIT to PASS without topology validation plus sidecar-complete audited evidence.",
            "If supplemental control lanes are used, preserve validate_topology proof and sidecar hashes in the authority audit.",
            "If expanded exact falsification lanes are used, preserve the exact runnable index provenance in the authority audit.",
        ],
    }
    markdown_lines = [
        "# Loop 0 Authority Promotion Audit",
        "",
        f"- Generated: `{audit['generated_at_utc']}`",
        f"- Status: `{gate_status}`",
        f"- Verdict: `{verdict}`",
        f"- Audited targets: `{len(audited)}`",
        f"- Pending promotion: `{len(pending)}`",
        f"- Failing targets: `{len(failing)}`",
        "",
        "## Pending promotion targets",
        "",
    ]
    if pending:
        markdown_lines.extend([f"- `{target}`" for target in pending])
    else:
        markdown_lines.append("- none")
    markdown_lines.extend(["", "## Failing targets", ""])
    if failing:
        markdown_lines.extend([f"- `{target}`" for target in failing])
    else:
        markdown_lines.append("- none")

    write_json(run_root / "loop0/authority_promotion_audit.json", audit)
    write_json(run_root / "loop0/authority_gate_decision.json", gate)
    (run_root / "loop0/authority_promotion_audit.md").write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": gate_status,
                "verdict": verdict,
                "audited_count": len(audited),
                "pending_count": len(pending),
                "failing_count": len(failing),
            }
        )
    )
    return 0 if gate_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
