#!/usr/bin/env python3
"""
Composite Figure 10B: 6-target grid of (real | scramble-null) PyMOL panels
with target labels and PFR percentages baked in.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

ROOT = Path("/home/diddy/Desktop/Prism4D-bio/prism4d_manuscript/pfr_assets/pymol_real/renders")

TARGETS = [
    ("CDK2_allosteric",    "CDK2 allosteric (3PXZ + 2AN probe)",      "42.3", "p ≤ 0.001"),
    ("Thrombin_exosite",   "Thrombin exosite (1HAH + TYS)",           "39.7", "p ≤ 0.001"),
    ("cGAS",               "cGAS (4O67 + 1SY)",                       "26.5", "p ≤ 0.001"),
    ("CRBN",               "CRBN (5FQD + LVY degrader)",              "19.8", "p ≤ 0.001"),
    ("HRAS_Q61H",          "HRAS Q61H (6OIM + GDP)",                  "17.6", "p ≤ 0.001"),
    ("TP53_apo",           "TP53 R175H (3ZME + QC5 stabilizer)",      "15.2", "p = 0.001"),
]


def main():
    n = len(TARGETS)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.5 * n), dpi=300)
    fig.patch.set_facecolor("white")

    col_headers = [
        "Phase-ordered PRISM4D manifold\n(directional pharmacophore vectors)",
        "Temporal-scramble null\n(same positions, randomized vectors)",
    ]
    for col, header in enumerate(col_headers):
        axes[0, col].set_title(header, fontsize=12, fontweight="bold",
                                color=("#0F5F22" if col == 0 else "#404040"),
                                pad=14)

    for row, (target, label, pfr, p) in enumerate(TARGETS):
        real_png = ROOT / f"{target}__real.png"
        null_png = ROOT / f"{target}__null.png"
        for col, png in enumerate([real_png, null_png]):
            ax = axes[row, col]
            if png.exists():
                ax.imshow(mpimg.imread(str(png)))
            ax.axis("off")
            # Banner with target name + PFR figures
            if col == 0:
                ax.text(0.02, 0.97, label, transform=ax.transAxes,
                        fontsize=11, fontweight="bold", color="#101010",
                        verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor="white", edgecolor="#101010", linewidth=0.8))
                ax.text(0.02, 0.02, f"Phase-ordered PFR = {pfr}%   ({p})",
                        transform=ax.transAxes,
                        fontsize=11, color="#0F5F22", fontweight="bold",
                        verticalalignment="bottom",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#E8F5E9", edgecolor="#0F5F22", linewidth=0.7))
            else:
                ax.text(0.02, 0.02, "Null PFR = 2.2%",
                        transform=ax.transAxes,
                        fontsize=11, color="#404040", fontweight="bold",
                        verticalalignment="bottom",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#F5F5F5", edgecolor="#666", linewidth=0.7))

    fig.suptitle("B. Phase-ordered vs temporal-scramble vectorial pharmacophore recovery (matched cameras)",
                 fontsize=13, fontweight="bold", x=0.06, ha="left", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out = ROOT / "Figure10B_phase_vs_scramble_grid.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
