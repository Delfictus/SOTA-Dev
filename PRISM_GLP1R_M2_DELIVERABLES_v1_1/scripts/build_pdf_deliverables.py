#!/usr/bin/env python3
"""Phase 4 — generate 5 PDFs from the verified Markdown / table / JSON ground
truth using a Jinja2 LaTeX template + pandoc fragment conversion.

Each PDF is a *derived* artifact:
    Markdown (verified, epistemic-labeled) --> LaTeX fragment via pandoc
    --> Jinja2 stitch --> xelatex --> PDF.

No claim mutation. No projected->observed promotion. No invented data.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined

REPO = Path("/home/diddy/Desktop/Prism4D-bio")
DROOT = REPO / "PRISM_GLP1R_M2_DELIVERABLES_v1_1"
GT = DROOT / "05_GROUND_TRUTH_DATA"
TABLES = DROOT / "09_TABLE_EXPORTS"
LATEX_DIR = DROOT / "02_LATEX_SOURCE"
GEN_DIR = LATEX_DIR / "generated"
PDF_DIR = DROOT / "01_PDF_DELIVERABLES"
TXT_DIR = DROOT / "04_TXT_SOURCE"
MD_DIR = DROOT / "03_MARKDOWN_SOURCE"


_TEX_SPECIALS = str.maketrans({
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "^": r"\^{}",
    "~": r"\~{}",
})


def tex_escape(s: str) -> str:
    return str(s).translate(_TEX_SPECIALS)


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md_to_latex_fragment(md_path: Path) -> str:
    """pandoc Markdown -> LaTeX fragment (no preamble)."""
    proc = subprocess.run(
        ["pandoc", "-f", "gfm+pipe_tables", "-t", "latex",
         "--columns=100",
         "--top-level-division=section",
         "--listings",
         str(md_path)],
        check=True, capture_output=True, text=True,
    )
    return harden_latex_fragment(proc.stdout)


# --- LaTeX post-processor -----------------------------------------------
# Three transforms:
#   1. Convert wide pandoc longtables (>4 cols using only l/r/c spec) into
#      xltabular with equal-width X columns, so cells wrap instead of
#      overflowing the right margin.
#   2. Wrap any long unbroken hex/identifier blob inside texttt / lstinline
#      with \seqsplit so it can break per character.
#   3. Inside \passthrough{\lstinline!...!}, replace very long unbroken
#      blobs with a discretionary-break version.

_LONGTABLE_RX = re.compile(
    r"\\begin\{longtable\}\[\]\{@\{\}([lrc]+)@\{\}\}(.*?)\\end\{longtable\}",
    re.DOTALL,
)


# Bare long token regex used ONLY inside table-cell bodies (not on the
# whole document, because LaTeX section labels look textual and we
# must not wrap them in \seqsplit).
# Must contain at least one \_ or : or / (label anchors use hyphens only,
# so this discriminates).
_CELL_BARE_LONG_RX = re.compile(
    r"(?<!\\seqsplit\{)(?<!\\texttt\{)"
    r"([A-Za-z0-9][A-Za-z0-9]"
    r"(?:\\_|[A-Za-z0-9:/.-]){10,}"
    r"(?:\\_|[A-Za-z0-9:/.])"
    r")"
)


def _wrap_table_body_bare_long(cell_body: str) -> str:
    def repl(m: "re.Match[str]") -> str:
        token = m.group(1)
        # Require at least one \_, :, or / — without these, it's likely a
        # label or word-hyphenated phrase, not an unbreakable identifier.
        if ("\\_" not in token) and (":" not in token) and ("/" not in token):
            return token
        return r"\seqsplit{" + token + r"}"
    out = _CELL_BARE_LONG_RX.sub(repl, cell_body)
    # Convert `A\_B\_C:RES1-\textgreater RES2` style PAIR identifiers (very
    # common in edge/path tables, where `->` becomes `\textgreater`).
    pair_rx = re.compile(
        r"(?<!\\seqsplit\{)"
        r"([A-Za-z][A-Za-z0-9]+(?:\\_[A-Za-z0-9]+)+:[A-Za-z]+[0-9]+)"
        r"-\\textgreater\s*"
        r"([A-Za-z]+[0-9]+)"
    )
    out = pair_rx.sub(
        lambda m: r"\seqsplit{" + m.group(1) + r"}\hspace{0pt}\textgreater\hspace{0pt}\seqsplit{" + m.group(2) + r"}",
        out,
    )
    return out


def _convert_longtable_to_xltabular(match: "re.Match[str]") -> str:
    spec = match.group(1)
    body = match.group(2)
    ncols = len(spec)
    if ncols < 2:
        return match.group(0)
    body = _wrap_table_body_bare_long(body)
    cols_x = " ".join([">{\\RaggedRight\\arraybackslash}X"] * ncols)
    # Scale font down as column count grows — wider tables need smaller text
    # to keep multi-word cells from overflowing the per-column X width.
    if ncols >= 6:
        font_size = "\\scriptsize"
        col_sep = "2pt"
    elif ncols >= 4:
        font_size = "\\footnotesize"
        col_sep = "3pt"
    else:
        font_size = "\\small"
        col_sep = "4pt"
    return (
        "\\begingroup\n"
        f"{font_size}\n"
        "\\renewcommand{\\arraystretch}{1.05}\n"
        f"\\setlength{{\\tabcolsep}}{{{col_sep}}}\n"
        "\\begin{xltabular}{\\textwidth}{@{}" + cols_x + "@{}}"
        + body +
        "\\end{xltabular}\n"
        "\\endgroup\n"
    )


# Find a long unbroken hex / underscore-joined identifier (>=20 chars,
# no whitespace) inside a \texttt{...} group and re-wrap with \seqsplit.
_TEXTTT_LONG_RX = re.compile(
    r"\\texttt\{([A-Za-z0-9_\\.\-:/]{20,})\}"
)


def _wrap_long_texttt(match: "re.Match[str]") -> str:
    inner = match.group(1)
    return r"\texttt{\seqsplit{" + inner + r"}}"


# pandoc emits \passthrough{\lstinline!X!} for inline `code` spans.
# lstinline does NOT break mid-word even with breaklines=true. Long hex
# hashes or long identifiers blow out cell width. Convert ALL passthrough+
# lstinline to \texttt{\seqsplit{...}} which permits per-char line breaks.
_PASSTHROUGH_RX = re.compile(
    r"\\passthrough\{\\lstinline!([^!]+)!\}"
)


def _passthrough_to_seqsplit(match: "re.Match[str]") -> str:
    inner = match.group(1)
    if len(inner) < 8:
        # Short inline code (e.g. "True", "v1.0") — leave as plain texttt.
        return r"\texttt{" + inner + r"}"
    return r"\texttt{\seqsplit{" + inner + r"}}"


def harden_latex_fragment(tex: str) -> str:
    # Order matters:
    #   1. passthrough → seqsplit (covers most inline-code overflows)
    #   2. plain texttt → seqsplit (long hex/id blobs in monospace)
    #   3. longtable → xltabular (table widths) + cell-scoped bare-token
    #      wrap (we deliberately do NOT run bare-token wrap on the whole
    #      .tex because pandoc label anchors like
    #      `section-1--master-data-room-index` would otherwise be wrapped
    #      inside \hypertarget{} arguments and break the macro).
    tex = _PASSTHROUGH_RX.sub(_passthrough_to_seqsplit, tex)
    tex = _TEXTTT_LONG_RX.sub(_wrap_long_texttt, tex)
    tex = _LONGTABLE_RX.sub(_convert_longtable_to_xltabular, tex)
    return tex


def md_to_text(md_path: Path) -> str:
    """pandoc Markdown -> plain text (for 04_TXT_SOURCE/)."""
    proc = subprocess.run(
        ["pandoc", "-f", "gfm+pipe_tables", "-t", "plain",
         "--wrap=preserve",
         str(md_path)],
        check=True, capture_output=True, text=True,
    )
    return proc.stdout


def inline_md_string_to_latex(md: str) -> str:
    proc = subprocess.run(
        ["pandoc", "-f", "gfm+pipe_tables", "-t", "latex",
         "--columns=100",
         "--top-level-division=section",
         "--listings"],
        input=md, check=True, capture_output=True, text=True,
    )
    return harden_latex_fragment(proc.stdout)


def assemble_executive_dossier_md() -> str:
    parts: list[str] = []
    parts.append("# Section 1 — Master Data-Room Index\n\n")
    parts.append((GT / "campaigns/glp1r_aleniglipron/MASTER_DATA_ROOM_INDEX.md").read_text())
    parts.append("\n\n# Section 2 — Executive Readout\n\n")
    parts.append((GT / "campaigns/glp1r_aleniglipron/M2_Executive_Readout_Final.md").read_text())
    parts.append("\n\n# Section 3 — Enterprise Positioning\n\n")
    parts.append((GT / "campaigns/glp1r_aleniglipron/ENTERPRISE_POSITIONING_SUMMARY.md").read_text())
    parts.append("\n\n# Section 4 — CBOM Summary\n\n")
    parts.append((TABLES / "CBOM_Summary.md").read_text())
    parts.append("\n\n# Section 5 — Deliverable Status Snapshot\n\n")
    parts.append(
        "| Deliverable | Status | Epistemic role | Notes |\n"
        "|---|---|---|---|\n"
        "| PRISM CBOM v1.0 | present | OBSERVED | Cryptographic bill-of-materials |\n"
        "| M2 Replayability Manifest | present | OBSERVED | Environment + seed lineage |\n"
        "| Claim falsification graph | present | DERIVED | 8 claims, 16 edges, 20 nodes |\n"
        "| CRO Wet-Lab Action Plan | present | PROJECTED | 6 falsification gates |\n"
        "| Zero-shot replacements | present | PROJECTED / HYPOTHESIZED | Top-10 ranked |\n"
        "| Fragment interference | present | INFERRED | 9 edges |\n"
        "| Critical-edge validation | present | DERIVED | 9 edges |\n"
        "| Translation pathway nodes | present | DERIVED | 5 ranked |\n"
        "| Phase 2C metastable triggers | present | OBSERVED | 52 triggers |\n"
        "| Phase 2D staged variant grid | present | PROJECTED | Materialization status: staged, not executed |\n"
    )
    return "".join(parts)


def assemble_pharm_dynamics_md() -> str:
    parts: list[str] = []
    parts.append((GT / "campaigns/glp1r_aleniglipron/M2_Pharmacological_Dynamics_Intelligence_Report.md").read_text())
    parts.append("\n\n# Appendix A — Critical-Edge Validation\n\n")
    parts.append((TABLES / "Critical_Edge_Validation.md").read_text())
    parts.append("\n\n# Appendix B — Translation Pathway Nodes\n\n")
    parts.append((TABLES / "Translation_Pathway_Nodes.md").read_text())
    parts.append("\n\n# Appendix C — Phase 2C Metastable Triggers (sample)\n\n")
    parts.append((TABLES / "Phase2C_Metastable_Trigger_Summary.md").read_text())
    parts.append("\n\n# Appendix D — Claim / Falsification Graph\n\n")
    parts.append((TABLES / "Claim_Falsification_Graph.md").read_text())
    return "".join(parts)


def assemble_medchem_md() -> str:
    parts: list[str] = []
    parts.append("# Section 1 — Aleniglipron Interference Summary\n\n")
    parts.append((GT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/aleniglipron_interference_summary.md").read_text())
    parts.append("\n\n# Section 2 — Zero-Shot Top-10 Replacements\n\n")
    parts.append(
        "**SAR-contingency notice.** Every replacement below is PROJECTED "
        "or HYPOTHESIZED. These are not synthesis instructions, not "
        "validated compounds, and not biological recommendations. Use as "
        "a SAR shortlist input only, subject to medicinal-chemistry review "
        "and wet-lab falsification.\n\n"
    )
    parts.append((TABLES / "ZeroShot_Top10_Replacements.md").read_text())
    parts.append("\n\n# Section 3 — Fragment Interference Attribution\n\n")
    parts.append((TABLES / "Fragment_Interference_Attribution.md").read_text())
    parts.append("\n\n# Section 4 — Phase 2D Staged Variant Grid\n\n")
    parts.append(
        "**Materialization status.** Phase 2D targets are *staged* — "
        "they appear in the planning manifest but have not been engine-"
        "executed. Treat the table below as a queue, not an evidence set.\n\n"
    )
    parts.append((TABLES / "Phase2D_Staged_Targets.md").read_text())
    return "".join(parts)


def assemble_cro_md() -> str:
    parts: list[str] = []
    parts.append("# Section 1 — CRO Wet-Lab Action Plan\n\n")
    parts.append(
        "Each row below is a **falsification gate**. The associated "
        "PRISM-4D claim is at risk if and only if the gate fails as "
        "described in `falsification_condition`. None of these rows is a "
        "biological confirmation request. Priority score is a routing "
        "weight, not a probability of success.\n\n"
    )
    parts.append((TABLES / "CRO_WetLab_Action_Plan.md").read_text())
    parts.append("\n\n# Section 2 — Claim / Falsification Graph\n\n")
    parts.append((TABLES / "Claim_Falsification_Graph.md").read_text())
    parts.append("\n\n# Section 3 — Triangulation Dossier (verbatim)\n\n")
    parts.append((GT / "campaigns/glp1r_aleniglipron/M2_Triangulation_Dossier_Final.md").read_text())
    return "".join(parts)


def assemble_audit_md(cbom_merkle: str, exec_sha: str) -> str:
    parts: list[str] = []
    parts.append("# Section 1 — Release Identity\n\n")
    parts.append(
        f"- **Campaign ID:** `glp1r_aleniglipron`\n"
        f"- **Release version:** `PRISM_GLP1R_M2_DELIVERABLES_v1.1`\n"
        f"- **CBOM Merkle root:** `{cbom_merkle}`\n"
        f"- **Executive archive SHA-256:** `{exec_sha}`\n\n"
    )
    parts.append("# Section 2 — CBOM Summary\n\n")
    parts.append((TABLES / "CBOM_Summary.md").read_text())
    parts.append("\n\n# Section 3 — Replayability Manifest (verbatim, JSON)\n\n")
    rm = json.loads((GT / "campaigns/glp1r_aleniglipron/M2_Replayability_Manifest.json").read_text())
    parts.append("```\n" + json.dumps(rm, indent=2, sort_keys=False)[:8000] + "\n```\n\n")
    parts.append("# Section 4 — Ground-Truth Source Manifest (head)\n\n")
    gt_manifest = json.loads((GT / "GROUND_TRUTH_FILE_MANIFEST.json").read_text())
    rows = ["| epistemic | category | size | sha256 (head) | path |",
            "|---|---|---|---|---|"]
    for e in gt_manifest["entries"]:
        rows.append(
            f"| {e['epistemic_role']} | {e['artifact_category']} | "
            f"{e['size_bytes']:,} | `{e['sha256'][:16]}…` | `{e['copied_relative_path']}` |"
        )
    parts.append("\n".join(rows) + "\n")
    parts.append("\n\n# Section 5 — Environment Fingerprint (verbatim)\n\n")
    cbom = json.loads((GT / "campaigns/glp1r_aleniglipron/PRISM_CBOM_v1.0.json").read_text())
    env = cbom.get("environment", {})
    parts.append("```\n" + json.dumps(env, indent=2) + "\n```\n")
    return parts and "".join(parts)


# ---- PDF specs ----------------------------------------------------
PDFS = [
    {
        "stem":   "PRISM_GLP1R_M2_Executive_Dossier",
        "title":  "PRISM-4D — GLP-1R / Aleniglipron M2 Executive Dossier",
        "short":  "Executive Dossier",
        "subt":   "Boardroom-facing overview of the M2 release",
        "rtype":  "Executive — strategy and program-leadership review",
        "assembler": assemble_executive_dossier_md,
    },
    {
        "stem":   "PRISM_GLP1R_M2_Pharmacological_Dynamics_Intelligence_Report",
        "title":  "PRISM-4D — GLP-1R / Aleniglipron Pharmacological Dynamics Intelligence Report",
        "short":  "Pharm Dynamics Intelligence",
        "subt":   "Scientific / program-lead report on M2 dynamics evidence",
        "rtype":  "Scientific — pharmacology / program lead",
        "assembler": assemble_pharm_dynamics_md,
    },
    {
        "stem":   "PRISM_GLP1R_M2_MedChem_Action_Appendix",
        "title":  "PRISM-4D — GLP-1R / Aleniglipron Med-Chem Action Appendix",
        "short":  "Med-Chem Appendix",
        "subt":   "Medicinal-chemistry / SAR review of M2 replacements",
        "rtype":  "Medicinal chemistry — SAR review (PROJECTED / HYPOTHESIZED)",
        "assembler": assemble_medchem_md,
    },
    {
        "stem":   "PRISM_GLP1R_M2_CRO_Falsification_Handoff",
        "title":  "PRISM-4D — GLP-1R / Aleniglipron CRO Falsification Handoff",
        "short":  "CRO Falsification Handoff",
        "subt":   "Experimental validation planning for M2 falsification gates",
        "rtype":  "CRO — assay routing and falsification gates",
        "assembler": assemble_cro_md,
    },
    {
        "stem":   "PRISM_GLP1R_M2_Audit_CBOM_Appendix",
        "title":  "PRISM-4D — GLP-1R / Aleniglipron Audit & CBOM Appendix",
        "short":  "Audit / CBOM Appendix",
        "subt":   "IT / diligence audit appendix and CBOM identity",
        "rtype":  "Audit / IT / diligence",
        "assembler": None,  # built inline with hashes
    },
]


def compile_latex(tex_path: Path, work_dir: Path) -> tuple[Optional[Path], str]:
    """Compile LaTeX via xelatex (twice) or fall back to pdflatex.

    Returns (pdf_path_or_None, log_tail).
    """
    log_tail = ""
    for engine in ("xelatex", "pdflatex"):
        if not shutil.which(engine):
            continue
        try:
            for run in range(2):
                proc = subprocess.run(
                    [engine, "-interaction=nonstopmode", "-halt-on-error",
                     "-output-directory", str(work_dir), str(tex_path)],
                    check=False, capture_output=True, text=True,
                    cwd=str(work_dir),
                )
                if proc.returncode != 0:
                    log_tail = (proc.stdout + proc.stderr)[-2000:]
                    break
            pdf = work_dir / (tex_path.stem + ".pdf")
            if pdf.is_file():
                return pdf, ""
        except Exception as ex:  # noqa: BLE001
            log_tail = repr(ex)
            continue
    return None, log_tail


def main() -> int:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)

    # Identity values
    cbom = json.loads((GT / "campaigns/glp1r_aleniglipron/PRISM_CBOM_v1.0.json").read_text())
    cbom_merkle = cbom.get("campaign_merkle_root", "")
    exec_sha = (GT / "PRISM_GLP1R_M2_EXECUTIVE_RELEASE_v1.1.tar.gz.sha256").read_text().split()[0]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    env = Environment(
        loader=FileSystemLoader(str(LATEX_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        # LaTeX-friendly delimiters: avoid {# and #} which collide with
        # LaTeX command parameter references like {#1}.
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="{{",
        variable_end_string="}}",
        comment_start_string="<%--",
        comment_end_string="--%>",
    )
    template = env.get_template("prism_pharma_report_template.tex.j2")

    build_status = []
    for spec in PDFS:
        stem = spec["stem"]
        print(f"\n=== Building {stem} ===")
        # 1. Assemble Markdown body
        if spec["assembler"] is not None:
            body_md = spec["assembler"]()
        else:
            body_md = assemble_audit_md(cbom_merkle, exec_sha)
        # Mirror MD source
        md_out = MD_DIR / (stem + ".md")
        md_out.write_text(body_md)

        # 2. MD -> LaTeX fragment
        body_latex = inline_md_string_to_latex(body_md)

        # 3. Stitch into template. tex_escape() guards every text-mode
        # identifier; the listings block reads from `*_raw` variants and
        # is already verbatim so it needs no escaping.
        rendered = template.render(
            report_title             = tex_escape(spec["title"]),
            report_short_title       = tex_escape(spec["short"]),
            report_subtitle          = tex_escape(spec["subt"]),
            report_type              = tex_escape(spec["rtype"]),
            campaign_id              = tex_escape("glp1r_aleniglipron"),
            release_version          = tex_escape("PRISM_GLP1R_M2_DELIVERABLES_v1.1"),
            generated_at_utc         = tex_escape(generated),
            cbom_merkle_root         = tex_escape(cbom_merkle),
            executive_archive_sha256 = tex_escape(exec_sha),
            campaign_id_raw          = "glp1r_aleniglipron",
            release_version_raw      = "PRISM_GLP1R_M2_DELIVERABLES_v1.1",
            generated_at_utc_raw     = generated,
            cbom_merkle_root_raw     = cbom_merkle,
            executive_archive_sha256_raw = exec_sha,
            body                     = body_latex,
        )
        tex_path = GEN_DIR / (stem + ".tex")
        tex_path.write_text(rendered)

        # 4. Compile
        pdf, errlog = compile_latex(tex_path, GEN_DIR)
        if pdf is not None:
            final_pdf = PDF_DIR / (stem + ".pdf")
            shutil.copy2(pdf, final_pdf)
            status = "OK"
            note = f"sha256={sha256_of(final_pdf)[:16]}…"
        else:
            status = "TEX_FALLBACK"
            note = errlog.replace("\n", " ")[:240]

        # 5. TXT source
        try:
            (TXT_DIR / (stem + ".txt")).write_text(
                subprocess.run(
                    ["pandoc", "-f", "gfm+pipe_tables", "-t", "plain", "--wrap=preserve"],
                    input=body_md, check=True, capture_output=True, text=True,
                ).stdout
            )
        except Exception as ex:  # noqa: BLE001
            (TXT_DIR / (stem + ".txt")).write_text(body_md)
            note += f" (txt-from-md: {ex})"
        build_status.append({"pdf": stem, "status": status, "note": note})
        print(f"  -> {status}  {note}")

    summary_path = PDF_DIR / "BUILD_STATUS.json"
    summary_path.write_text(json.dumps({
        "built_at_utc": generated,
        "pdfs": build_status,
    }, indent=2))
    print(f"\nBuild summary -> {summary_path}")

    # Hint file if any fell back
    if any(b["status"] != "OK" for b in build_status):
        instr = (
            "# BUILD_INSTRUCTIONS.md\n\n"
            "Some PDFs fell back to .tex. To compile manually:\n\n"
            "```bash\n"
            f"cd {GEN_DIR}\n"
            "xelatex -interaction=nonstopmode <stem>.tex && \\\n"
            "xelatex -interaction=nonstopmode <stem>.tex\n"
            "```\n\n"
            "Or with latexmk:\n\n"
            "```bash\n"
            f"latexmk -pdf -xelatex -outdir={PDF_DIR} {GEN_DIR}/<stem>.tex\n"
            "```\n"
        )
        (PDF_DIR / "BUILD_INSTRUCTIONS.md").write_text(instr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
