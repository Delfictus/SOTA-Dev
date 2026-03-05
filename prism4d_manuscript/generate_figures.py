#!/usr/bin/env python3
"""
PRISM-4D Manuscript Figure Generation Script
Generates all publication-quality figures as PDF files.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

# Publication settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for editing
plt.rcParams['ps.fonttype'] = 42

OUTPUT_DIR = Path("/home/diddy/Desktop/Prism4D-bio/prism4d_manuscript/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_figure(fig, filename):
    """Save figure as PDF with publication settings."""
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved {filename}")
    plt.close(fig)


def fig1_pipeline():
    """Figure 1: Pipeline Architecture flowchart."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')

    # Stage boxes
    stages = [
        {"x": 1.0, "label": "NHS Engine", "sub": "AMBER MD\nCryo-Thermal\nUV Excitation"},
        {"x": 3.5, "label": "Spike Detection", "sub": "3-Channel LIF\nUV/LIF/EFP"},
        {"x": 6.0, "label": "SNDC Clustering", "sub": "RT-DBSCAN\nWatershed\nEikonal BFS\nPeak Tracking"},
        {"x": 8.5, "label": "Site Scoring", "sub": "Druggability\nClassification\nCovalent ID"}
    ]

    box_width = 1.8
    box_height = 1.4
    box_y = 0.3

    colors = ['#2E8B99', '#3A9DAD', '#46B0C1', '#52C3D5']  # Teal gradient

    for i, stage in enumerate(stages):
        # Box
        box = FancyBboxPatch((stage["x"] - box_width/2, box_y), box_width, box_height,
                             boxstyle="round,pad=0.05",
                             edgecolor='#1a5f6f', facecolor=colors[i],
                             linewidth=2, alpha=0.9)
        ax.add_patch(box)

        # Label
        ax.text(stage["x"], box_y + box_height - 0.25, stage["label"],
                ha='center', va='top', fontsize=11, fontweight='bold', color='white')
        ax.text(stage["x"], box_y + 0.5, stage["sub"],
                ha='center', va='center', fontsize=8, color='white', linespacing=1.4)

        # Arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((stage["x"] + box_width/2 + 0.05, box_y + box_height/2),
                                   (stages[i+1]["x"] - box_width/2 - 0.05, box_y + box_height/2),
                                   arrowstyle='->', mutation_scale=25,
                                   linewidth=2.5, color='#1a5f6f', zorder=0)
            ax.add_patch(arrow)

    # Input/Output labels
    ax.text(0.1, box_y + box_height/2, "PDB Input",
            ha='left', va='center', fontsize=11, style='italic', color='#333')
    ax.text(9.9, box_y + box_height/2, "Druggable Sites",
            ha='right', va='center', fontsize=11, style='italic', color='#333')

    save_figure(fig, "fig1_pipeline.pdf")


def fig2_hysteresis():
    """Figure 2: Cryo-Thermal Hysteresis Protocol."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Phase boundaries (as % of total simulation)
    phases = [
        (0, 25.5, 50, "Cold Hold"),
        (25.5, 36.4, None, "Ramp Up"),  # Linear ramp
        (36.4, 63.6, 300, "Warm Hold"),
        (63.6, 74.5, None, "Ramp Down"),  # Linear ramp
        (74.5, 100, 50, "Cold Return")
    ]

    # Generate temperature curve
    progress = []
    temperature = []

    for start, end, temp, label in phases:
        n_points = 50
        x = np.linspace(start, end, n_points)
        progress.extend(x)

        if temp is not None:
            temperature.extend([temp] * n_points)
        elif "Up" in label:
            temperature.extend(np.linspace(50, 300, n_points))
        else:  # Ramp Down
            temperature.extend(np.linspace(300, 50, n_points))

    progress = np.array(progress)
    temperature = np.array(temperature)

    # Plot temperature curve with colors
    for i, (start, end, temp, label) in enumerate(phases):
        mask = (progress >= start) & (progress <= end)
        color = '#FF8C42' if "Up" in label else '#9B59B6' if "Down" in label else '#2E8B99'
        ax.plot(progress[mask], temperature[mask], linewidth=2.5, color=color, label=label if i < 2 else None)

    # UV burst positions (every 250 steps ~0.45% interval)
    uv_interval = 0.45
    uv_positions = np.arange(0, 100, uv_interval)
    uv_temps = np.interp(uv_positions, progress, temperature)
    ax.scatter(uv_positions, uv_temps, marker='^', s=15, color='#2E8B99', alpha=0.3, zorder=5)

    # Physiological line
    ax.axhline(300, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Physiological (300K)')

    # Phase annotations
    for start, end, temp, label in phases:
        mid = (start + end) / 2
        y_pos = 280 if temp == 300 else 70 if temp == 50 else 175
        ax.text(mid, y_pos, label, ha='center', va='center',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel('Simulation Progress (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 350)
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    save_figure(fig, "fig2_hysteresis.pdf")


def fig3_benchmark():
    """Figure 3: DCC Benchmark Bar Chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    proteins = ['BACE1\n(1W50)', 'TEM1\n(1BTL)', 'KRAS\n(4OBE)', 'PTP1B\n(1G1F)',
                'AdSS\n(1ADE)', 'Abl\n(3K5V)', 'IL-2\n(1A4Q)', 'SIRPα\n(2WNG)',
                'ERα\n(1ERE)', 'HIV-1\n(1HHP)', 'FKBP12\n(1BJ4)']

    dcc_values = [3.6, 3.7, 3.8, 4.8, 6.0, 6.2, 6.3, 7.1, 9.5, 9.8, 9.8]

    # Color coding
    colors = []
    for val in dcc_values:
        if val < 5:
            colors.append('#2ECC71')  # Green
        elif val < 8:
            colors.append('#2E8B99')  # Teal
        else:
            colors.append('#FF8C42')  # Orange

    x = np.arange(len(proteins))
    bars = ax.bar(x, dcc_values, color=colors, edgecolor='#333', linewidth=1.2, alpha=0.85)

    # Threshold lines
    ax.axhline(5, color='#2ECC71', linestyle='--', linewidth=1.5, alpha=0.6, label='Excellent (<5Å)')
    ax.axhline(8, color='#2E8B99', linestyle='--', linewidth=1.5, alpha=0.6, label='Good (<8Å)')
    ax.axhline(10, color='#FF8C42', linestyle='--', linewidth=1.5, alpha=0.6, label='Marginal (<10Å)')

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, dcc_values)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}Å',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(proteins, fontsize=9)
    ax.set_ylabel('DCC (Ångströms)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 11)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure(fig, "fig3_benchmark.pdf")


def fig4_accuracy():
    """Figure 4: Detection Accuracy Summary."""
    fig, ax = plt.subplots(figsize=(7, 5))

    categories = ['<5Å\n(Excellent)', '<8Å\n(Good)', '<10Å\n(Marginal)']
    counts = [4, 8, 11]
    total = 11
    percentages = [c/total*100 for c in counts]

    colors = ['#2ECC71', '#2E8B99', '#FF8C42']
    x = np.arange(len(categories))

    bars = ax.bar(x, counts, color=colors, edgecolor='#333', linewidth=1.5, alpha=0.85)

    # Labels
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        ax.text(bar.get_x() + bar.get_width()/2, count + 0.3,
                f'{count}/{total}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Number of Targets', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 13)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure(fig, "fig4_accuracy.pdf")


def fig5_reproducibility():
    """Figure 5: Reproducibility Analysis (3 panels)."""
    fig = plt.figure(figsize=(12, 4))

    # Panel A: Centroid scatter
    ax1 = plt.subplot(131)
    x_coords = [14.38, 14.40, 14.41, 14.39, 14.42]
    y_coords = [11.29, 11.30, 11.32, 11.28, 11.31]

    ax1.scatter(x_coords, y_coords, s=100, c='#2E8B99', edgecolors='#1a5f6f',
                linewidths=2, alpha=0.8, zorder=3)

    # Circle showing spread
    mean_x, mean_y = np.mean(x_coords), np.mean(y_coords)
    circle = Circle((mean_x, mean_y), 0.06, fill=False, edgecolor='red',
                    linewidth=2, linestyle='--', label='0.06Å radius')
    ax1.add_patch(circle)

    ax1.set_xlabel('X Position (Å)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Y Position (Å)', fontsize=10, fontweight='bold')
    ax1.set_title('A) Centroid Position', fontsize=11, fontweight='bold', loc='left')
    ax1.set_xlim(14.35, 14.45)
    ax1.set_ylim(11.25, 11.35)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Panel B: Spike count
    ax2 = plt.subplot(132)
    spike_counts = [3236, 3248, 3240, 3245, 3239]
    seeds = ['Seed 1', 'Seed 2', 'Seed 3', 'Seed 4', 'Seed 5']

    bars = ax2.bar(seeds, spike_counts, color='#2E8B99', edgecolor='#1a5f6f',
                   linewidth=1.5, alpha=0.85)

    for bar, count in zip(bars, spike_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, count + 2, str(count),
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    cv = np.std(spike_counts) / np.mean(spike_counts) * 100
    ax2.text(0.5, 0.95, f'CV = {cv:.1f}%', transform=ax2.transAxes,
            ha='center', va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    ax2.set_ylabel('Spike Count', fontsize=10, fontweight='bold')
    ax2.set_title('B) Spike Count Variation', fontsize=11, fontweight='bold', loc='left')
    ax2.set_ylim(3230, 3255)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: DCC histogram
    ax3 = plt.subplot(133)
    # 10 pairwise comparisons from 5 seeds (C(5,2) = 10)
    dcc_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.02, 0.03, 0.04, 0.01, 0.05]

    ax3.hist(dcc_values, bins=8, color='#2E8B99', edgecolor='#1a5f6f',
             linewidth=1.5, alpha=0.85, range=(0, 0.08))

    ax3.axvline(0.06, color='red', linestyle='--', linewidth=2, label='Max = 0.06Å')
    ax3.axvline(1.40, color='gray', linestyle=':', linewidth=2, label='C-C bond (1.40Å)', alpha=0.5)

    ax3.set_xlabel('Pairwise DCC (Å)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax3.set_title('C) Pairwise DCC Distribution', fontsize=11, fontweight='bold', loc='left')
    ax3.set_xlim(0, 1.6)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    save_figure(fig, "fig5_reproducibility.pdf")


def fig6_capability():
    """Figure 6: Method Comparison Capability Matrix."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['PRISM-4D', 'FTMap', 'fpocket', 'P2Rank', 'DeepSite', 'PocketMiner']
    capabilities = ['Physics\nSimulation', 'Cryptic\nPockets', 'Covalent\nID',
                   'UV\nSpectroscopy', 'Spike\nDetection', 'No Training\nData']

    # Data matrix (1=full, 0.5=partial, 0=none)
    data = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # PRISM-4D
        [0.5, 0.5, 0.0, 0.0, 0.0, 1.0],  # FTMap
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # fpocket
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # P2Rank
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # DeepSite
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # PocketMiner
    ])

    # Draw circles
    for i, method in enumerate(methods):
        for j, cap in enumerate(capabilities):
            value = data[i, j]
            if value == 1.0:
                circle = Circle((j, len(methods)-1-i), 0.35,
                              facecolor='#2E8B99', edgecolor='#1a5f6f', linewidth=2)
            elif value == 0.5:
                circle = mpatches.Wedge((j, len(methods)-1-i), 0.35, 90, 270,
                                       facecolor='#2E8B99', edgecolor='#1a5f6f', linewidth=2)
            else:
                circle = Circle((j, len(methods)-1-i), 0.35,
                              facecolor='white', edgecolor='#1a5f6f', linewidth=2)
            ax.add_patch(circle)

    # Labels
    ax.set_xticks(range(len(capabilities)))
    ax.set_xticklabels(capabilities, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods[::-1], fontsize=10, fontweight='bold')

    ax.set_xlim(-0.7, len(capabilities)-0.3)
    ax.set_ylim(-0.7, len(methods)-0.3)
    ax.set_aspect('equal')

    # Grid
    for i in range(len(methods)+1):
        ax.axhline(i-0.5, color='gray', linewidth=0.5, alpha=0.3)
    for j in range(len(capabilities)+1):
        ax.axvline(j-0.5, color='gray', linewidth=0.5, alpha=0.3)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Full Support',
                  markerfacecolor='#2E8B99', markersize=12, markeredgecolor='#1a5f6f', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', label='Partial Support',
                  markerfacecolor='#2E8B99', markersize=12, markeredgecolor='#1a5f6f',
                  markeredgewidth=2, fillstyle='left'),
        plt.Line2D([0], [0], marker='o', color='w', label='No Support',
                  markerfacecolor='white', markersize=12, markeredgecolor='#1a5f6f', markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

    ax.tick_params(axis='both', which='both', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    save_figure(fig, "fig6_capability.pdf")


def fig7_cost():
    """Figure 7: Hardware Cost Comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = ['PRISM-4D\n(RTX 5080)', 'Cloud HPC\n(per run)',
               'Schrödinger\n(annual)', 'D.E. Shaw\nAnton']
    costs = [999, 500, 30000, 100000000]

    colors = ['#2E8B99', '#3498DB', '#E67E22', '#E74C3C']

    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, costs, color=colors, edgecolor='#333', linewidth=1.5, alpha=0.85)

    # Value labels
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        if cost >= 1000000:
            label = f'${cost/1000000:.0f}M'
        elif cost >= 1000:
            label = f'${cost/1000:.0f}K'
        else:
            label = f'${cost:.0f}'

        ax.text(cost * 1.1, bar.get_y() + bar.get_height()/2, label,
                va='center', ha='left', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=11, fontweight='bold')
    ax.set_xlabel('Cost (USD, log scale)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(100, 200000000)
    ax.grid(True, axis='x', which='both', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure(fig, "fig7_cost.pdf")


def fig8_sirpa():
    """Figure 8: SIRPa Head-to-Head Comparison."""
    fig = plt.figure(figsize=(10, 5))

    # Panel A: DCC Comparison
    ax1 = plt.subplot(121)
    methods = ['PRISM-4D', 'P2Rank']
    dcc_values = [7.1, 9.0]
    colors = ['#2E8B99', '#95A5A6']

    bars = ax1.bar(methods, dcc_values, color=colors, edgecolor='#333',
                   linewidth=1.5, alpha=0.85, width=0.6)

    ax1.axhline(10, color='red', linestyle='--', linewidth=2, alpha=0.6, label='10Å Threshold')

    for bar, val in zip(bars, dcc_values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}Å',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('DCC (Ångströms)', fontsize=12, fontweight='bold')
    ax1.set_title('A) Detection Accuracy', fontsize=12, fontweight='bold', loc='left')
    ax1.set_ylim(0, 11)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Quality Metrics
    ax2 = plt.subplot(122)
    metrics = ['Quality\nScore', 'Druggability', 'Confidence']
    prism_values = [0.652, 0.652, 1.0]
    p2rank_values = [0.0, 0.77, 0.002]  # Quality N/A=0, confidence normalized

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax2.bar(x - width/2, prism_values, width, label='PRISM-4D',
                   color='#2E8B99', edgecolor='#1a5f6f', linewidth=1.5, alpha=0.85)
    bars2 = ax2.bar(x + width/2, p2rank_values, width, label='P2Rank',
                   color='#95A5A6', edgecolor='#333', linewidth=1.5, alpha=0.85)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.03, f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            else:
                ax2.text(bar.get_x() + bar.get_width()/2, 0.05, 'N/A' if height == 0 else f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8, style='italic')

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10)
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('B) Quality Metrics', fontsize=12, fontweight='bold', loc='left')
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    save_figure(fig, "fig8_sirpa.pdf")


def main():
    """Generate all figures."""
    print("Generating PRISM-4D Manuscript Figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)

    fig1_pipeline()
    fig2_hysteresis()
    fig3_benchmark()
    fig4_accuracy()
    fig5_reproducibility()
    fig6_capability()
    fig7_cost()
    fig8_sirpa()

    print("-" * 60)
    print(f"All figures saved to {OUTPUT_DIR}")
    print("Complete!")


if __name__ == "__main__":
    main()
