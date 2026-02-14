#!/usr/bin/env python3
"""
Build PDF-ready markdown from the main report by inserting figure embeds
at appropriate locations, then invoke pandoc to generate the PDF.
"""

import re
import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path("/home/diddy/Desktop/Prism4D-bio/e2e_validation_test/results_myc")
REPORT_MD = REPORT_DIR / "PRISM4D_MYC_MAX_REPORT.md"
PDF_MD = REPORT_DIR / "PRISM4D_MYC_MAX_REPORT_pdf.md"
OUTPUT_PDF = REPORT_DIR / "PRISM4D_MYC_MAX_REPORT.pdf"

# Figure insertions: (after_line_pattern, figure_markdown)
# These get inserted AFTER the first line matching the pattern
FIGURE_INSERTIONS = [
    # Figure 1: After the visualization section header intro
    (
        r"^### 5\.1 PyMOL Session",
        "\n![**Figure 1.** Binding site overview — ray-traced rendering showing all three docked ligands "
        "(cyan = 10058-F4, hot pink = KJ-Pyr-9, orange = MYCi975) within the PRISM4D-detected pocket "
        "(purpleblue surface) at the c-MYC/MAX bHLH-LZ interface. Pharmacophore features shown as "
        "colored spheres. Yellow dashes indicate PLIP-detected hydrophobic contacts.]"
        "(analysis/sota_figure.png){width=100%}\n"
    ),
    # 2D structures: After the Detailed Molecular Profiles section header
    (
        r"^#### 3\.5\.3 Detailed Molecular Profiles",
        "\n![**Figure 5a.** 2D structure of 10058-F4 (MW 230.3, QED 0.82)](analysis/figures/2d_10058-F4.png){width=32%}"
        " ![**Figure 5b.** 2D structure of KJ-Pyr-9 (MW 371.5, QED 0.68)](analysis/figures/2d_KJ-Pyr-9.png){width=32%}"
        " ![**Figure 5c.** 2D structure of MYCi975 (MW 675.0, QED 0.21)](analysis/figures/2d_MYCi975.png){width=32%}\n"
    ),
    # Pharmacophore panel: After the pharmacophore map table
    (
        r"^\| \(68\.0, 68\.3, 36\.6\)",
        "\n![**Figure 2.** Pharmacophore map — 20 sub-clustered features from 606,656 spike events. "
        "Orange = pi-stacking (BNZ), green = H-bond+pi (TYR), gray = shape/flexibility (UNK), "
        "blue = cationic complementarity, red = anionic complementarity.]"
        "(analysis/figures/panel_pharmacophore.png){width=85%}\n"
    ),
    # Ligand close-ups: After the PLIP section
    (
        r"^### 3\.5 ADMET & Drug-Likeness",
        "\n![**Figure 3a.** 10058-F4 close-up — contacts with TYR53 (3.61 A), ALA52 (3.48 A), LEU55 (3.67 A)]"
        "(analysis/figures/panel_10058F4.png){width=48%}"
        " ![**Figure 3b.** KJ-Pyr-9 close-up — contact with ALA52 (3.42 A)]"
        "(analysis/figures/panel_KJPyr9.png){width=48%}\n"
        "\n![**Figure 3c.** MYCi975 close-up — contact with LEU55 (2.87 A)]"
        "(analysis/figures/panel_MYCi975.png){width=48%}"
        " ![**Figure 4.** Alternative view — 90-degree rotation of binding site]"
        "(analysis/figures/panel_rotated.png){width=48%}\n"
    ),
    # Scoring comparison: After the scoring method agreement section
    (
        r"^The partial discordance between scoring methods",
        "\n![**Figure 6.** Multi-method scoring comparison — normalized scores across Vina, CNN Affinity, "
        "CNN Pose Score, MM-GBSA, and QED for all three ligands.]"
        "(analysis/figures/multi_score_comparison.png){width=80%}\n"
        "\n![**Figure 7.** Predicted scores vs. experimental Kd — correlation analysis showing "
        "Vina and MM-GBSA dG against published binding constants.]"
        "(analysis/figures/scoring_correlation.png){width=80%}\n"
    ),
    # PLIP interaction diagrams: After each ligand's PLIP section
    (
        r"^10058-F4 makes three hydrophobic contacts",
        "\n![**Figure 8a.** PLIP interaction diagram — 10058-F4](analysis/10058-F4_interactions.png){width=60%}\n"
    ),
    (
        r"^\| Hydrophobic \| 1 \| A:ALA52 \(3\.42",
        "\n![**Figure 8b.** PLIP interaction diagram — KJ-Pyr-9](analysis/KJ-Pyr-9_interactions.png){width=60%}\n"
    ),
    (
        r"^\| Hydrophobic \| 1 \| A:LEU55 \(2\.87",
        "\n![**Figure 8c.** PLIP interaction diagram — MYCi975](analysis/MYCi975_interactions.png){width=60%}\n"
    ),
]


def build_pdf_markdown():
    """Insert figure embeds into the report markdown."""
    lines = REPORT_MD.read_text().splitlines()

    # Track which insertions have been made
    inserted = [False] * len(FIGURE_INSERTIONS)

    output_lines = []
    for line in lines:
        output_lines.append(line)
        for i, (pattern, figure_md) in enumerate(FIGURE_INSERTIONS):
            if not inserted[i] and re.search(pattern, line):
                output_lines.append(figure_md)
                inserted[i] = True
                break

    # Add YAML frontmatter for pandoc
    header = """---
title: "PRISM4D Computational Drug Discovery Report"
subtitle: "c-MYC/MAX Heterodimer — Binding Site Detection, Docking & Lead Profiling"
date: "2026-02-12"
geometry: margin=1in
fontsize: 11pt
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - \\usepackage{float}
  - \\usepackage{graphicx}
  - \\let\\origfigure\\figure
  - \\let\\endorigfigure\\endfigure
  - \\renewenvironment{figure}[1][]{\\origfigure[H]}{\\endorigfigure}
  - \\usepackage{fancyhdr}
  - \\pagestyle{fancy}
  - \\fancyhead[L]{PRISM4D Drug Discovery Report}
  - \\fancyhead[R]{c-MYC/MAX}
  - \\fancyfoot[C]{\\thepage}
  - \\fancyfoot[R]{CONFIDENTIAL}
---

"""

    pdf_content = header + "\n".join(output_lines)

    # Replace Unicode characters that pdflatex can't handle
    unicode_replacements = {
        "\u2264": "<=",      # ≤
        "\u2265": ">=",      # ≥
        "\u00b2": "^2",      # ²
        "\u00b3": "^3",      # ³
        "\u2192": "->",      # →
        "\u00d7": "x",       # ×
        "\u00b1": "+/-",     # ±
        "\u2014": "---",     # —
        "\u2013": "--",      # –
        "\u03b1": "alpha",   # α
        "\u03b2": "beta",    # β
        "\u03b3": "gamma",   # γ
        "\u03b4": "delta",   # δ
        "\u03bc": "u",       # μ → u (for uM etc.)
        "\u00c5": "A",       # Å → A (Angstrom)
        "\u2550": "=",       # ═
        "\u2551": "|",       # ║
        "\u2554": "+",       # ╔
        "\u2557": "+",       # ╗
        "\u255a": "+",       # ╚
        "\u255d": "+",       # ╝
        "\u2560": "+",       # ╠
        "\u2563": "+",       # ╣
        "\u251c": "|--",     # ├
        "\u2514": "|--",     # └
        "\u2502": "|",       # │
        "\u2500": "-",       # ─ (horizontal box drawing)
        "\u00b0": " deg",    # ° (degree)
    }
    for char, replacement in unicode_replacements.items():
        pdf_content = pdf_content.replace(char, replacement)

    PDF_MD.write_text(pdf_content)

    not_inserted = [i for i, done in enumerate(inserted) if not done]
    if not_inserted:
        print(f"WARNING: {len(not_inserted)} figure(s) not inserted (patterns not matched):")
        for i in not_inserted:
            print(f"  [{i}] {FIGURE_INSERTIONS[i][0]}")
    else:
        print(f"All {len(FIGURE_INSERTIONS)} figures inserted successfully.")

    return PDF_MD


def convert_to_pdf(md_path):
    """Convert markdown to PDF with pandoc."""
    cmd = [
        "pandoc",
        str(md_path),
        "-o", str(OUTPUT_PDF),
        "--pdf-engine=pdflatex",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        "-V", "documentclass=article",
        "-V", "classoption=oneside",
        "--highlight-style=tango",
        "--resource-path", str(REPORT_DIR),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPORT_DIR))

    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        return False

    if result.stderr:
        # pandoc often outputs warnings to stderr even on success
        print(f"Warnings:\n{result.stderr}")

    size_mb = OUTPUT_PDF.stat().st_size / (1024 * 1024)
    print(f"\nPDF generated: {OUTPUT_PDF}")
    print(f"Size: {size_mb:.1f} MB")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("PRISM4D Report → PDF Conversion")
    print("=" * 60)

    md_path = build_pdf_markdown()
    print(f"\nPDF-ready markdown: {md_path}")

    if convert_to_pdf(md_path):
        print("\nSUCCESS")
    else:
        print("\nFAILED — check errors above")
        sys.exit(1)
