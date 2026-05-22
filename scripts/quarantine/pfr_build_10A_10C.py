#!/usr/bin/env python3
"""
Build Figure 10A (Vectorial PFR scoring schematic) and Figure 10C
(per-target fold enrichment bar chart) from corrected PFR data.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

ROOT = Path("/home/diddy/Desktop/Prism4D-bio")
OUT = ROOT / "prism4d_manuscript" / "pfr_assets" / "pymol_real" / "renders"
OUT.mkdir(parents=True, exist_ok=True)


def fig_10A():
    """Vectorial PFR scoring criterion: 3.5 A sphere + 30 deg cone."""
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Origin: predicted apo pharmacophore feature
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])  # interaction vector
    dist = 3.5
    half_angle_deg = 30.0
    half_angle = np.radians(half_angle_deg)

    # 3.5 A scoring sphere (translucent grey, dashed wireframe)
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    xs = dist * np.outer(np.cos(u), np.sin(v))
    ys = dist * np.outer(np.sin(u), np.sin(v))
    zs = dist * np.outer(np.ones(36), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="0.5", alpha=0.16, linewidth=0.4, rstride=2, cstride=2)

    # 30 deg angular cone along +x
    cone_h = dist
    cone_r = math.tan(half_angle) * cone_h
    theta = np.linspace(0, 2 * np.pi, 64)
    h = np.linspace(0, cone_h, 24)
    H, T = np.meshgrid(h, theta)
    R = (cone_r / cone_h) * H
    Xc = H
    Yc = R * np.cos(T)
    Zc = R * np.sin(T)
    ax.plot_surface(Xc, Yc, Zc, color="#06A6C4", alpha=0.16, linewidth=0, edgecolor="none")

    # Feature origin sphere (red — acceptor)
    ax.scatter([0], [0], [0], color="#E60D1F", s=320, edgecolor="white",
               linewidth=2.0, zorder=10, label="predicted apo pharmacophore feature")

    # Interaction vector arrow (red)
    ax.quiver(0, 0, 0, 1.6, 0, 0, color="#E60D1F", linewidth=4.0,
              arrow_length_ratio=0.30)

    # HIT: matched holo ligand atom inside cone (within distance AND angle)
    hit_pt = np.array([2.4, 0.5, 0.3])
    ax.scatter(*hit_pt, color="#1E8A36", s=420, edgecolor="white",
               linewidth=2.5, zorder=11, label="matched holo feature (HIT)")
    ax.plot([0, hit_pt[0]], [0, hit_pt[1]], [0, hit_pt[2]],
            color="#1E8A36", linewidth=2.5, linestyle="--", alpha=0.75)

    # MISS-by-angle: within 3.5 A but outside 30 deg cone
    miss_angle = np.array([0.6, 2.4, 1.6])
    ax.scatter(*miss_angle, color="#888", s=260, edgecolor="white",
               linewidth=1.6, zorder=9, label="miss: outside 30° cone")

    # MISS-by-distance: outside the 3.5 A sphere even though along the vector
    miss_dist = np.array([5.0, 0.0, 0.0])
    ax.scatter(*miss_dist, color="#888", s=260, edgecolor="white",
               linewidth=1.6, zorder=9, alpha=0.45)
    ax.text(5.2, 0.0, 0.6, "miss: > 3.5 Å", fontsize=10, color="0.35")

    # Distance and angle annotations
    ax.text(1.8, 0.0, 3.3, "3.5 Å envelope", fontsize=11, color="0.30", style="italic")
    ax.text(2.5, 1.4, 1.2, "30° cone", fontsize=11, color="#057085", style="italic")
    ax.text(-0.7, -0.6, -0.2, "feature\norigin", fontsize=10, color="#900",
            ha="center", va="top")
    ax.text(2.0, 0.85, 0.6, "HIT\n(within 3.5 Å\n& ≤ 30°)", fontsize=10,
            color="#0F5F22", ha="center", va="bottom", fontweight="bold")

    # Axes
    L = 5.5
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis._axinfo['grid']['linewidth'] = 0.2
    ax.yaxis._axinfo['grid']['linewidth'] = 0.2
    ax.zaxis._axinfo['grid']['linewidth'] = 0.2
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=22, azim=-35)

    ax.set_title("A. Vectorial pharmacophore feature recovery (PFR) scoring criterion",
                 fontsize=12, loc="left", pad=12, fontweight="bold")

    # Manual legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#E60D1F",
                   markersize=14, markeredgecolor="white", label="Predicted apo pharmacophore"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1E8A36",
                   markersize=14, markeredgecolor="white", label="Recovered holo feature (HIT)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                   markersize=11, markeredgecolor="white", label="MISS (distance or angle violation)"),
        mpatches.Patch(color="#06A6C4", alpha=0.32, label="30° angular cone (direction tolerance)"),
        plt.Line2D([0], [0], color="0.5", lw=1.0, alpha=0.6, label="3.5 Å scoring envelope"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              framealpha=0.96, fancybox=True, edgecolor="0.7", bbox_to_anchor=(-0.08, 0.95))

    out = OUT / "Figure10A_vectorial_PFR_scoring.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
def fig_10C():
    """Per-target fold enrichment bar chart with significance labels."""
    rows = [
        ("CDK2_allosteric",    42.3, 2.2, 19.2, "p ≤ 0.001"),
        ("Thrombin_exosite",   39.7, 2.2, 18.0, "p ≤ 0.001"),
        ("cGAS",               26.5, 2.2, 12.0, "p ≤ 0.001"),
        ("CRBN",               19.8, 2.2,  9.0, "p ≤ 0.001"),
        ("HRAS_Q61H",          17.6, 2.2,  8.0, "p ≤ 0.001"),
        ("TP53_apo",           15.2, 2.2,  6.9, "p = 0.001"),
        ("Aggregate",          26.9, 2.2, 12.2, "p ≤ 0.001"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300,
                                    gridspec_kw=dict(width_ratios=[1.0, 0.9]))
    fig.patch.set_facecolor("white")

    targets = [r[0].replace("_", "\n") for r in rows]
    real = [r[1] for r in rows]
    null = [r[2] for r in rows]
    folds = [r[3] for r in rows]
    pvals = [r[4] for r in rows]

    # ---- Panel 1: PFR % bars (real vs null) ----
    x = np.arange(len(targets))
    w = 0.40
    colors_real = ["#1B5E20" if i < 6 else "#0D47A1" for i in range(len(rows))]
    bars_real = ax1.bar(x - w/2, real, w, label="Phase-ordered PRISM4D",
                         color=colors_real, edgecolor="white", linewidth=0.8)
    bars_null = ax1.bar(x + w/2, null, w, label="Temporal-scramble null",
                         color="#B0B0B0", edgecolor="white", linewidth=0.8)
    for i, (b, val, p) in enumerate(zip(bars_real, real, pvals)):
        ax1.text(b.get_x() + b.get_width()/2, val + 0.6, f"{val:.1f}%",
                 ha="center", fontsize=9, fontweight="bold", color="#1B3A0F")
    for b, val in zip(bars_null, null):
        ax1.text(b.get_x() + b.get_width()/2, val + 0.6, f"{val:.1f}%",
                 ha="center", fontsize=8, color="0.30")

    ax1.set_ylabel("Vectorial PFR (% holo features recovered)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets, fontsize=9)
    ax1.set_ylim(0, max(real) * 1.18)
    ax1.legend(fontsize=10, loc="upper right", frameon=True, edgecolor="0.7")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("C. Per-target PFR vs temporal-scramble null",
                  fontsize=12, loc="left", fontweight="bold", pad=10)
    # Divide aggregate from per-target
    ax1.axvline(x[-1] - 0.5, color="0.4", linestyle=":", linewidth=1)

    # ---- Panel 2: fold enrichment ----
    bars = ax2.barh(targets[::-1], folds[::-1], color="#2D6CD8",
                     edgecolor="white", linewidth=0.8)
    # Aggregate bar in different color
    bars[0].set_color("#0D47A1")
    for b, val, p in zip(bars, folds[::-1], pvals[::-1]):
        ax2.text(val + 0.4, b.get_y() + b.get_height()/2,
                 f"{val:.1f}×  ({p})",
                 va="center", fontsize=9, color="0.10")
    ax2.set_xlabel("Fold enrichment over temporal-scramble null", fontsize=11)
    ax2.set_xlim(0, max(folds) * 1.32)
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("Fold enrichment + corrected empirical p-value",
                  fontsize=11, loc="left", fontweight="bold", pad=10)
    ax2.axvline(1, color="0.4", linestyle=":", linewidth=1)
    ax2.text(1.05, len(targets) - 0.5, "no enrichment", fontsize=8,
              color="0.4", style="italic")

    plt.tight_layout()
    out = OUT / "Figure10C_PFR_enrichment.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    fig_10A()
    fig_10C()
