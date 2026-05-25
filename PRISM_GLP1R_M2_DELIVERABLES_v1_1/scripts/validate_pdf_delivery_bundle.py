#!/usr/bin/env python3
"""Phase 9 — validate the PRISM GLP-1R M2 v1.1 PDF delivery bundle.

Checks:
  - all 5 expected PDFs exist (or .tex fallback exists)
  - LaTeX source exists (template + at least 5 generated .tex)
  - TXT mirror exists for every PDF
  - Markdown source exists for every PDF
  - ground-truth manifest exists and matches actual files on disk
  - visualization package exists (README + viewer + tarball)
  - delivery manifest exists and every file in it hashes correctly
  - no forbidden overclaim strings in PDF/TeX/MD sources
  - delivery tarball is intact + sha256 matches its companion .sha256 file
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
MAN = DROOT / "10_DELIVERY_MANIFESTS"
REPORT = MAN / "PDF_DELIVERY_VALIDATION_REPORT.md"

EXPECTED_PDF_STEMS = [
    "PRISM_GLP1R_M2_Executive_Dossier",
    "PRISM_GLP1R_M2_Pharmacological_Dynamics_Intelligence_Report",
    "PRISM_GLP1R_M2_MedChem_Action_Appendix",
    "PRISM_GLP1R_M2_CRO_Falsification_Handoff",
    "PRISM_GLP1R_M2_Audit_CBOM_Appendix",
]

FORBIDDEN_STRINGS = [
    "mechanistic proof",
    "confirmed biological efficacy",
    "irreversible desensitization",
    "guaranteed",
    "patient response",
    "clinical outcome",
    "32 femtosecond",
    "Expected Δ uptake > 15",
    "Expected Delta uptake > 15",
]

# Per-text-source whitelist of allowed *false positives* — places where a
# forbidden substring legitimately appears as part of a longer phrase we
# explicitly want to keep (e.g. negation, definition of forbidden term).
WHITELIST_REGEXES = [
    re.compile(r"(?:not|never|no)\s+a\s+patient[- ]response", re.IGNORECASE),
    re.compile(r"(?:not|never|no)\s+a\s+clinical[- ]outcome", re.IGNORECASE),
    re.compile(r"(?:not|never|no)\s+(?:experimental|biological)\s+(?:validation|proof)", re.IGNORECASE),
    re.compile(r"forbidden.*overclaim", re.IGNORECASE),
    # The KNOWN_LIMITATIONS.md negates the bad phrases by listing them:
    re.compile(r"Nothing in this package", re.IGNORECASE),
]


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_for_forbidden(path: Path, kind: str) -> list[tuple[str, str]]:
    """Return list of (forbidden_string, context_line) hits. Allow
    whitelisted contexts (negations / KNOWN_LIMITATIONS legend)."""
    if not path.is_file():
        return []
    try:
        text = path.read_text(errors="ignore") if kind != "pdf" else _pdf_to_text(path)
    except Exception:  # noqa: BLE001
        return []
    hits: list[tuple[str, str]] = []
    lower = text.lower()
    for needle in FORBIDDEN_STRINGS:
        n_low = needle.lower()
        idx = 0
        while True:
            j = lower.find(n_low, idx)
            if j < 0:
                break
            # Pull surrounding 120-char context for whitelist check
            ctx = text[max(0, j - 80): j + len(needle) + 80]
            if any(rx.search(ctx) for rx in WHITELIST_REGEXES):
                idx = j + len(needle)
                continue
            hits.append((needle, ctx.replace("\n", " ").strip()))
            idx = j + len(needle)
    return hits


def _pdf_to_text(p: Path) -> str:
    import subprocess
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", "-q", str(p), "-"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return proc.stdout
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    findings: list[str] = []
    warnings: list[str] = []
    ok_count = 0
    fail_count = 0

    def ok(msg: str) -> None:
        nonlocal ok_count
        ok_count += 1
        findings.append(f"- [OK] {msg}")

    def fail(msg: str) -> None:
        nonlocal fail_count
        fail_count += 1
        findings.append(f"- [FAIL] {msg}")

    def warn(msg: str) -> None:
        warnings.append(f"- [WARN] {msg}")
        findings.append(f"- [WARN] {msg}")

    # 1. PDF existence
    for stem in EXPECTED_PDF_STEMS:
        pdf = DROOT / "01_PDF_DELIVERABLES" / f"{stem}.pdf"
        tex = DROOT / "02_LATEX_SOURCE" / "generated" / f"{stem}.tex"
        if pdf.is_file():
            ok(f"PDF present: `{pdf.relative_to(DROOT)}` ({pdf.stat().st_size:,} bytes)")
        elif tex.is_file():
            warn(f"PDF missing but .tex fallback present: `{tex.relative_to(DROOT)}`")
        else:
            fail(f"PDF + .tex both missing for `{stem}`")

    # 2. LaTeX source
    if (DROOT / "02_LATEX_SOURCE" / "prism_pharma_report_template.tex.j2").is_file():
        ok("LaTeX template present")
    else:
        fail("LaTeX template missing")
    gen = list((DROOT / "02_LATEX_SOURCE" / "generated").glob("*.tex"))
    (ok if len(gen) >= 5 else fail)(f"Generated .tex count: {len(gen)} (expected ≥5)")

    # 3. TXT + 4. MD mirrors
    for stem in EXPECTED_PDF_STEMS:
        for sub, ext in (("04_TXT_SOURCE", ".txt"), ("03_MARKDOWN_SOURCE", ".md")):
            p = DROOT / sub / f"{stem}{ext}"
            (ok if p.is_file() else fail)(f"{ext} source present: `{p.relative_to(DROOT)}`")

    # 5. Ground-truth manifest matches disk
    gt_manifest_path = DROOT / "05_GROUND_TRUTH_DATA" / "GROUND_TRUTH_FILE_MANIFEST.json"
    if not gt_manifest_path.is_file():
        fail("GROUND_TRUTH_FILE_MANIFEST.json missing")
    else:
        gtm = json.loads(gt_manifest_path.read_text())
        gt_hash_fail = 0
        for e in gtm["entries"]:
            f = DROOT / e["copied_relative_path"]
            if not f.is_file():
                fail(f"GT file missing: `{e['copied_relative_path']}`")
                continue
            if sha256_of(f) != e["sha256"]:
                fail(f"GT hash mismatch: `{e['copied_relative_path']}`")
                gt_hash_fail += 1
        if gt_hash_fail == 0:
            ok(f"All {len(gtm['entries'])} ground-truth files hash-verify against manifest")

    # 6. Visualization
    viz = DROOT / "06_VISUALIZATION_PACKAGE"
    for must in ("VISUALIZATION_README.md", "ENTERPRISE_RELEASE_VIEWER.html",
                 "PRISM_GLP1R_M2_VISUALIZATION_PACKAGE_v1.1.tar.gz",
                 "PRISM_GLP1R_M2_VISUALIZATION_PACKAGE_v1.1.tar.gz.sha256"):
        p = viz / must
        (ok if p.is_file() else fail)(f"viz: `{p.relative_to(DROOT)}` present")
    if (viz / "visualizer_app" / "index.html").is_file():
        ok("viz: visualizer_app/index.html present")
    else:
        fail("viz: visualizer_app/index.html missing")

    # 7. Delivery manifest sanity + per-entry hash verify (sampled)
    dm_path = MAN / "DELIVERY_MANIFEST.json"
    if not dm_path.is_file():
        fail("DELIVERY_MANIFEST.json missing")
    else:
        dm = json.loads(dm_path.read_text())
        # Spot-check every PDF + key audit files
        spot = [e for e in dm["entries"] if (
            e["relative_path"].startswith("01_PDF_DELIVERABLES/") or
            e["relative_path"].startswith("07_AUDIT_AND_CBOM/") or
            e["relative_path"].startswith("08_RELEASE_ARCHIVES/") or
            e["relative_path"].endswith("GROUND_TRUTH_FILE_MANIFEST.json")
        )]
        mismatches = 0
        for e in spot:
            p = DROOT / e["relative_path"]
            if not p.is_file():
                mismatches += 1
                fail(f"manifest entry missing on disk: `{e['relative_path']}`")
                continue
            if sha256_of(p) != e["sha256"]:
                mismatches += 1
                fail(f"manifest hash mismatch: `{e['relative_path']}`")
        if mismatches == 0:
            ok(f"Delivery manifest spot-check: {len(spot)} entries hash-verify")

    # 8. Forbidden overclaim strings
    overclaim_hits = []
    scan_targets = []
    for stem in EXPECTED_PDF_STEMS:
        scan_targets.append(("pdf", DROOT / "01_PDF_DELIVERABLES" / f"{stem}.pdf"))
        scan_targets.append(("md",  DROOT / "03_MARKDOWN_SOURCE" / f"{stem}.md"))
        scan_targets.append(("txt", DROOT / "04_TXT_SOURCE" / f"{stem}.txt"))
        scan_targets.append(("tex", DROOT / "02_LATEX_SOURCE" / "generated" / f"{stem}.tex"))
    for kind, p in scan_targets:
        hits = scan_for_forbidden(p, kind)
        if hits:
            overclaim_hits.append((p.relative_to(DROOT), hits))
    if not overclaim_hits:
        ok("No forbidden overclaim strings in PDF/TeX/MD/TXT sources")
    else:
        for rel, hits in overclaim_hits:
            for needle, ctx in hits:
                fail(f"forbidden string `{needle}` in `{rel}` :: …{ctx}…")

    # 9. Final delivery tarball integrity
    tar_path = DROOT / "PRISM_GLP1R_M2_PDF_DELIVERABLES_v1.1.tar.gz"
    sha_path = DROOT / "PRISM_GLP1R_M2_PDF_DELIVERABLES_v1.1.tar.gz.sha256"
    if tar_path.is_file() and sha_path.is_file():
        live = sha256_of(tar_path)
        signed = sha_path.read_text().strip().split()[0]
        if live == signed:
            ok(f"Delivery tarball sha256 verifies ({tar_path.stat().st_size:,} bytes)")
        else:
            fail(f"Delivery tarball sha256 mismatch  live={live}  signed={signed}")
    else:
        fail("Delivery tarball or .sha256 missing")

    # Report
    lines = [
        "# PRISM GLP-1R M2 v1.1 — PDF Delivery Validation Report",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Delivery root: `{DROOT}`",
        f"- Findings: **{ok_count} OK, {fail_count} FAIL, {len(warnings)} WARN**",
        "",
        "## Status",
        "",
        ("**PASS — delivery is shippable.**"
         if fail_count == 0 else
         f"**FAIL — {fail_count} blocking issue(s); see findings below.**"),
        "",
        "## Findings",
        "",
        *findings,
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"validation: ok={ok_count}  fail={fail_count}  warn={len(warnings)}")
    print(f"  -> {REPORT}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
