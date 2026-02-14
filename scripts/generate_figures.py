#!/usr/bin/env python3
"""
generate_figures.py  --  Publication-quality figures for MYC inhibitor analysis.

Produces:
  1) 2D ligand structure PNGs (RDKit)
  2) Scoring-vs-Kd correlation scatter (matplotlib, dual Y-axis)
  3) Multi-method normalised bar chart (matplotlib)

All output lands in:
  e2e_validation_test/results_myc/analysis/figures/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import stats

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
LIGANDS = {
    "10058-F4": {
        "smiles": "Cc1cc(/C=C/c2ccsc2C)ccc1O",
        "kd_uM": 39.7,
        "mmgbsa_dG": -16.1,
        "vina": -4.13,
        "cnn_affinity": 4.82,
        "qed": 0.74,
        "pharma_overlap": 0.61,
    },
    "KJ-Pyr-9": {
        "smiles": "O=C(c1ccc2c(c1)CCCC2)N1CCN(c2nccc3ccccc23)CC1",
        "kd_uM": 0.0065,
        "mmgbsa_dG": -0.1,
        "vina": -2.90,
        "cnn_affinity": 5.21,
        "qed": 0.82,
        "pharma_overlap": 0.45,
    },
    "MYCi975": {
        "smiles": "Oc1c(I)cc(I)cc1-c1cc(Cl)cc(Cl)c1-c1nc2cc(C(F)(F)F)ccc2[nH]1",
        "kd_uM": 2.5,
        "mmgbsa_dG": -28.8,
        "vina": -0.47,
        "cnn_affinity": 6.10,
        "qed": 0.55,
        "pharma_overlap": 0.78,
    },
}

OUT_DIR = Path(__file__).resolve().parent.parent / \
    "e2e_validation_test" / "results_myc" / "analysis" / "figures"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_font():
    """Set publication-quality font defaults."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


# ---------------------------------------------------------------------------
# 1) 2D Ligand Structures
# ---------------------------------------------------------------------------

def generate_2d_structures():
    """Render each ligand as a clean 2D PNG using RDKit's MolDraw2DCairo."""
    print("[1/3] Generating 2D ligand structures ...")
    for name, data in LIGANDS.items():
        mol = Chem.MolFromSmiles(data["smiles"])
        if mol is None:
            print(f"  WARNING: Could not parse SMILES for {name}, skipping.")
            continue

        # Compute 2D coords
        AllChem.Compute2DCoords(mol)

        # Use the SVG/Cairo drawer for fine-grained control
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 400)
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        opts.addStereoAnnotation = True
        opts.bondLineWidth = 2.0
        opts.padding = 0.12
        opts.backgroundColour = (1.0, 1.0, 1.0, 1.0)  # white

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        safe_name = name.replace(" ", "_")
        out_path = OUT_DIR / f"2d_{safe_name}.png"
        with open(out_path, "wb") as fh:
            fh.write(drawer.GetDrawingText())
        print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# 2) Scoring-vs-Kd Correlation (dual Y-axis)
# ---------------------------------------------------------------------------

def generate_correlation_plot():
    """Scatter: Kd (X, log-scale) vs MM-GBSA dG and Vina (dual Y)."""
    print("[2/3] Generating scoring correlation plot ...")
    _setup_font()

    names = list(LIGANDS.keys())
    kd = np.array([LIGANDS[n]["kd_uM"] for n in names])
    mmgbsa = np.array([LIGANDS[n]["mmgbsa_dG"] for n in names])
    vina = np.array([LIGANDS[n]["vina"] for n in names])

    log_kd = np.log10(kd)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # -- Left Y: MM-GBSA --
    colour1 = "#2166ac"
    ax1.set_xlabel(r"Published $K_d$ ($\mu$M)", fontweight="bold")
    ax1.set_ylabel(r"MM-GBSA $\Delta G_{\mathrm{bind}}$ (kcal mol$^{-1}$)",
                   color=colour1, fontweight="bold")
    sc1 = ax1.scatter(kd, mmgbsa, s=120, c=colour1, edgecolors="k",
                      linewidths=0.8, zorder=5, label="MM-GBSA")
    ax1.tick_params(axis="y", labelcolor=colour1)
    ax1.set_xscale("log")

    # R^2 for MM-GBSA vs log(Kd)
    slope1, intercept1, r1, _, _ = stats.linregress(log_kd, mmgbsa)
    r2_mmgbsa = r1 ** 2
    # regression line
    xfit = np.logspace(np.log10(kd.min() * 0.5), np.log10(kd.max() * 2), 50)
    ax1.plot(xfit, slope1 * np.log10(xfit) + intercept1,
             ls="--", lw=1.2, color=colour1, alpha=0.5)

    # label points (left Y) -- per-point offsets to avoid overlap
    mmgbsa_offsets = {
        "10058-F4": (8, 8),
        "KJ-Pyr-9": (8, 10),
        "MYCi975": (-60, 10),
    }
    for i, n in enumerate(names):
        ax1.annotate(n, (kd[i], mmgbsa[i]),
                     textcoords="offset points",
                     xytext=mmgbsa_offsets.get(n, (8, 8)),
                     fontsize=9, color=colour1)

    # -- Right Y: Vina --
    colour2 = "#b2182b"
    ax2 = ax1.twinx()
    ax2.set_ylabel("Vina score (kcal mol$^{-1}$)",
                   color=colour2, fontweight="bold")
    sc2 = ax2.scatter(kd, vina, s=120, c=colour2, marker="D",
                      edgecolors="k", linewidths=0.8, zorder=5, label="Vina")
    ax2.tick_params(axis="y", labelcolor=colour2)

    slope2, intercept2, r2, _, _ = stats.linregress(log_kd, vina)
    r2_vina = r2 ** 2
    ax2.plot(xfit, slope2 * np.log10(xfit) + intercept2,
             ls="--", lw=1.2, color=colour2, alpha=0.5)

    # Per-point offsets to avoid label collisions / edge clipping
    vina_offsets = {
        "10058-F4": (-70, 8),
        "KJ-Pyr-9": (8, 10),
        "MYCi975": (8, -14),
    }
    for i, n in enumerate(names):
        ax2.annotate(n, (kd[i], vina[i]),
                     textcoords="offset points",
                     xytext=vina_offsets.get(n, (8, -14)),
                     fontsize=9, color=colour2)

    # -- annotation box with R^2 --
    textstr = (f"MM-GBSA  $R^2$ = {r2_mmgbsa:.3f}\n"
               f"Vina         $R^2$ = {r2_vina:.3f}")
    props = dict(boxstyle="round,pad=0.4", facecolor="white",
                 edgecolor="grey", alpha=0.9)
    ax1.text(0.03, 0.97, textstr, transform=ax1.transAxes,
             fontsize=10, verticalalignment="top", bbox=props)

    # -- combined legend --
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper center", framealpha=0.9, ncol=2,
               bbox_to_anchor=(0.55, 0.95))

    ax1.set_title("Scoring Function vs Experimental Affinity", fontweight="bold")
    fig.tight_layout()

    out_path = OUT_DIR / "scoring_correlation.png"
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path}  (R2_MMGBSA={r2_mmgbsa:.3f}, R2_Vina={r2_vina:.3f})")


# ---------------------------------------------------------------------------
# 3) Multi-method normalised bar chart
# ---------------------------------------------------------------------------

def generate_multi_score_comparison():
    """Grouped bar chart, each metric normalised 0-1 (best = 1)."""
    print("[3/3] Generating multi-score comparison chart ...")
    _setup_font()

    metrics = ["Vina score", "CNN affinity", "MM-GBSA dG", "QED",
               "Pharmacophore\noverlap"]
    raw_keys = ["vina", "cnn_affinity", "mmgbsa_dG", "qed", "pharma_overlap"]
    names = list(LIGANDS.keys())

    # Build raw matrix  (n_ligands x n_metrics)
    raw = np.array([[LIGANDS[n][k] for k in raw_keys] for n in names])

    # Normalise each column to [0, 1].
    # For Vina and MM-GBSA, more negative = better, so we negate before
    # normalising so that "best" maps to 1.
    # For the others, higher = better already.
    negate_cols = [0, 2]  # vina, mmgbsa
    normed = raw.copy()
    for c in negate_cols:
        normed[:, c] = -normed[:, c]

    for c in range(normed.shape[1]):
        col = normed[:, c]
        cmin, cmax = col.min(), col.max()
        if cmax - cmin > 1e-12:
            normed[:, c] = (col - cmin) / (cmax - cmin)
        else:
            normed[:, c] = 1.0  # all equal

    n_metrics = len(metrics)
    n_ligands = len(names)
    x = np.arange(n_metrics)
    width = 0.22

    colours = ["#4393c3", "#d6604d", "#5aae61"]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (name, colour) in enumerate(zip(names, colours)):
        offset = (i - (n_ligands - 1) / 2) * width
        bars = ax.bar(x + offset, normed[i], width, label=name,
                      color=colour, edgecolor="k", linewidth=0.6)
        # Value labels on top of each bar
        for bar, val in zip(bars, normed[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Normalised Score (best = 1)", fontweight="bold")
    ax.set_title("Multi-Method Scoring Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = OUT_DIR / "multi_score_comparison.png"
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    generate_2d_structures()
    generate_correlation_plot()
    generate_multi_score_comparison()

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
