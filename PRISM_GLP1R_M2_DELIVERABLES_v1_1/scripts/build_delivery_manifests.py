#!/usr/bin/env python3
"""Phase 7 — delivery manifests + human-facing index.

Walks every file under the delivery root and emits:
  - DELIVERY_MANIFEST.json   — machine-readable file ledger with hashes
  - DELIVERY_INDEX.md        — human-facing overview
  - KNOWN_LIMITATIONS.md     — explicit scope/epistemic caveats
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/diddy/Desktop/Prism4D-bio")
DROOT = REPO / "PRISM_GLP1R_M2_DELIVERABLES_v1_1"
MANIFEST_DIR = DROOT / "10_DELIVERY_MANIFESTS"

# What category does each top-level dir belong to?
CATEGORY = {
    "01_PDF_DELIVERABLES":      ("pdf",                "derived"),
    "02_LATEX_SOURCE":          ("latex_source",       "derived"),
    "03_MARKDOWN_SOURCE":       ("markdown_source",    "derived"),
    "04_TXT_SOURCE":            ("text_source",        "derived"),
    "05_GROUND_TRUTH_DATA":     ("ground_truth",       "source_of_truth"),
    "06_VISUALIZATION_PACKAGE": ("visualization",      "derived"),
    "07_AUDIT_AND_CBOM":        ("audit_cbom",         "source_of_truth"),
    "08_RELEASE_ARCHIVES":      ("release_archive",    "source_of_truth"),
    "09_TABLE_EXPORTS":         ("table_export",       "derived"),
    "10_DELIVERY_MANIFESTS":    ("delivery_manifest",  "derived"),
    "scripts":                  ("script",             "derived"),
}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def category_for(path: Path) -> tuple[str, str]:
    rel = path.relative_to(DROOT)
    top = rel.parts[0] if rel.parts else ""
    return CATEGORY.get(top, ("other", "derived"))


def main() -> int:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Walk and hash everything except manifest outputs (created here)
    entries = []
    for p in sorted(DROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(DROOT)
        if rel.parts[0] == "10_DELIVERY_MANIFESTS" and rel.name in (
            "DELIVERY_MANIFEST.json", "DELIVERY_INDEX.md", "KNOWN_LIMITATIONS.md",
            "PDF_DELIVERY_VALIDATION_REPORT.md",
        ):
            continue
        cat, sot = category_for(p)
        entries.append({
            "relative_path":   str(rel),
            "size_bytes":      p.stat().st_size,
            "sha256":          sha256_of(p),
            "category":        cat,
            "source_of_truth": (sot == "source_of_truth"),
        })

    # Trace pdf -> markdown -> ground-truth derivation
    md_to_gt_keywords = {
        "Executive_Dossier":           ["MASTER_DATA_ROOM_INDEX.md", "M2_Executive_Readout_Final.md", "ENTERPRISE_POSITIONING_SUMMARY.md"],
        "Pharmacological_Dynamics":    ["M2_Pharmacological_Dynamics_Intelligence_Report.md"],
        "MedChem_Action_Appendix":     ["aleniglipron_interference_summary.md", "teaser_solutions.parquet", "fragment_interference_attribution.parquet"],
        "CRO_Falsification_Handoff":   ["CRO_WetLab_Action_Plan.parquet", "M2_Triangulation_Dossier_Final.md", "claim_falsification_graph.json"],
        "Audit_CBOM_Appendix":         ["PRISM_CBOM_v1.0.json", "M2_Replayability_Manifest.json", "GROUND_TRUTH_FILE_MANIFEST.json"],
    }
    for e in entries:
        if e["relative_path"].startswith("01_PDF_DELIVERABLES/") and e["relative_path"].endswith(".pdf"):
            for tag, sources in md_to_gt_keywords.items():
                if tag in e["relative_path"]:
                    e["derived_from"] = sources
                    break

    total_bytes = sum(e["size_bytes"] for e in entries)
    manifest = {
        "package_version":   "PRISM_GLP1R_M2_DELIVERABLES_v1.1",
        "created_at_utc":    now,
        "delivery_root":     str(DROOT),
        "entry_count":       len(entries),
        "total_bytes":       total_bytes,
        "entries":           entries,
    }
    (MANIFEST_DIR / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    # Human-facing index ------------------------------------------
    by_cat = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)

    def fmt_section(title: str, cat: str, blurb: str) -> str:
        rows = ["", f"## {title}", "", blurb, "",
                "| file | size | sha256 (head) | source of truth |",
                "|---|---|---|---|"]
        for e in sorted(by_cat.get(cat, []), key=lambda x: x["relative_path"]):
            rows.append(
                f"| `{e['relative_path']}` | {e['size_bytes']:,} | "
                f"`{e['sha256'][:16]}…` | {'yes' if e['source_of_truth'] else 'no'} |"
            )
        return "\n".join(rows) + "\n"

    index = [
        "# PRISM GLP-1R M2 v1.1 — Delivery Index",
        "",
        f"- **Package:** `PRISM_GLP1R_M2_DELIVERABLES_v1.1`",
        f"- **Created (UTC):** {now}",
        f"- **Total files:** {len(entries):,}",
        f"- **Total size:** {total_bytes:,} bytes ({total_bytes/1024/1024:.1f} MiB)",
        f"- **Delivery root:** `{DROOT}`",
        "",
        "## Epistemic legend (canonical)",
        "",
        "Every claim and table row in this package inherits one class — this",
        "governs how the row may be cited.",
        "",
        "| class | meaning |",
        "|---|---|",
        "| OBSERVED | direct tensor measurement from PRISM-4D engine outputs |",
        "| DERIVED | deterministic transform of observed tensors |",
        "| INFERRED | multi-tensor interpretation; not a single-tensor measurement |",
        "| PROJECTED | translational extrapolation beyond simulated conditions |",
        "| HYPOTHESIZED | requires wet-lab falsification before any biological claim |",
        "",
    ]
    index.append(fmt_section("1. PDF deliverables", "pdf",
        "Polished, LaTeX-compiled deliverables. Derived from `03_MARKDOWN_SOURCE/`."))
    index.append(fmt_section("2. LaTeX sources", "latex_source",
        "Jinja2 template + generated `.tex` files. Reproducible build inputs."))
    index.append(fmt_section("3. Markdown sources", "markdown_source",
        "Stitched Markdown bodies fed into the PDF builder."))
    index.append(fmt_section("4. TXT sources", "text_source",
        "Plain-text mirrors of each PDF. Searchable, paste-friendly."))
    index.append(fmt_section("5. Ground-truth data", "ground_truth",
        "**Source of truth.** Verbatim snapshots of engine + downstream artifacts."))
    index.append(fmt_section("6. Visualization package", "visualization",
        "Static visualizer app + auto-generated review portal + parquet preview TSVs."))
    index.append(fmt_section("7. Audit / CBOM", "audit_cbom",
        "**Source of truth.** CBOM, replayability manifest, claim graph, release sha256."))
    index.append(fmt_section("8. Release archives", "release_archive",
        "**Source of truth.** The executive release tarball + signature, unpacked nowhere."))
    index.append(fmt_section("9. Table exports", "table_export",
        "Per-table CSV (full) + Markdown (compact) — derived from ground-truth parquets/JSON."))
    index.append(fmt_section("10. Delivery manifests", "delivery_manifest",
        "Index + machine-readable manifest + known limitations."))
    index.append(fmt_section("Scripts", "script",
        "Build / snapshot / table-export / validation scripts. Reproducible build inputs."))

    index.append("\n## Known limitations\n\n"
                 "See `KNOWN_LIMITATIONS.md` in this directory.\n")
    index.append("\n## Recommended handoff procedure\n\n"
                 "1. Verify `PRISM_GLP1R_M2_PDF_DELIVERABLES_v1.1.tar.gz` against\n"
                 "   its companion `.sha256` file before extraction.\n"
                 "2. Read `01_PDF_DELIVERABLES/PRISM_GLP1R_M2_Executive_Dossier.pdf`\n"
                 "   for the executive overview.\n"
                 "3. Open the static review portal at\n"
                 "   `06_VISUALIZATION_PACKAGE/ENTERPRISE_RELEASE_VIEWER.html` for\n"
                 "   the full browseable artifact tree.\n"
                 "4. Use `05_GROUND_TRUTH_DATA/GROUND_TRUTH_FILE_MANIFEST.json` to\n"
                 "   trace any cited claim back to its underlying tensor / JSON.\n"
                 "5. Treat every PROJECTED / HYPOTHESIZED row as a wet-lab\n"
                 "   falsification gate, not as confirmed biological evidence.\n")

    (MANIFEST_DIR / "DELIVERY_INDEX.md").write_text("\n".join(index))

    # Known limitations -------------------------------------------
    klimits = (
        "# PRISM GLP-1R M2 v1.1 — Known Limitations\n"
        "\n"
        "These limitations are part of the delivery itself, not concessions.\n"
        "Every downstream citation must remain consistent with them.\n"
        "\n"
        "## Derivation chain\n"
        "\n"
        "- The PDF layer is a *derived* artifact. Its source-of-truth is the\n"
        "  verified Markdown / Parquet / JSON content in `05_GROUND_TRUTH_DATA/`,\n"
        "  `07_AUDIT_AND_CBOM/`, and `08_RELEASE_ARCHIVES/`. If a PDF and the\n"
        "  ground-truth disagree, the ground-truth wins.\n"
        "- No claims have been added by the PDF layer. The PDF formatter\n"
        "  may re-order or section the content, but it does not introduce\n"
        "  new biological assertions.\n"
        "\n"
        "## Phase 2D — staged, not executed\n"
        "\n"
        "- The Phase 2D variant grid manifest in this package is in\n"
        "  `materialization_status: staged`. The engine has not yet been run\n"
        "  on those targets in the v1.1 timeline. The rows in\n"
        "  `09_TABLE_EXPORTS/Phase2D_Staged_Targets.csv` are a planning queue,\n"
        "  not an evidence set.\n"
        "\n"
        "## Zero-shot replacements — PROJECTED / HYPOTHESIZED\n"
        "\n"
        "- The Top-10 replacements in `ZeroShot_Top10_Replacements` are\n"
        "  computational projections from the manual emulation track.\n"
        "- They are **not validated compounds**, **not synthesis instructions**,\n"
        "  and **not biological recommendations**.\n"
        "- They are SAR-contingency shortlist inputs subject to medicinal-\n"
        "  chemistry review and wet-lab falsification.\n"
        "\n"
        "## CRO action plan — falsification gates only\n"
        "\n"
        "- Every row of `CRO_WetLab_Action_Plan` is a falsification gate.\n"
        "- The associated PRISM-4D claim is at risk if and only if the gate\n"
        "  fails as described in its `falsification_condition` field.\n"
        "- Priority score is a routing weight, not a probability of success.\n"
        "\n"
        "## No clinical / patient claims\n"
        "\n"
        "- Nothing in this package is a clinical-effect claim.\n"
        "- Nothing in this package is a patient-response prediction.\n"
        "- Nothing in this package is experimental validation.\n"
        "\n"
        "## Scope separation\n"
        "\n"
        "- This is the lightweight executive delivery. The full raw audit\n"
        "  archive (large, including raw spike events, full mechanical-load\n"
        "  networks, and raw `.bin` files) is intentionally **not** included\n"
        "  in this delivery tarball. It remains separately stored and is\n"
        "  available on request through the campaign's data-room procedure.\n"
        "\n"
        "## Visualizer epistemic overlays\n"
        "\n"
        "- The visualizer color-codes PROJECTED / HYPOTHESIZED layers so that\n"
        "  falsification gates are reviewable. Visibility in the viewer is\n"
        "  not a citation license — it is a review affordance.\n"
    )
    (MANIFEST_DIR / "KNOWN_LIMITATIONS.md").write_text(klimits)

    print(f"manifest: {len(entries)} files, {total_bytes:,} bytes")
    print(f"  -> {MANIFEST_DIR / 'DELIVERY_MANIFEST.json'}")
    print(f"  -> {MANIFEST_DIR / 'DELIVERY_INDEX.md'}")
    print(f"  -> {MANIFEST_DIR / 'KNOWN_LIMITATIONS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
